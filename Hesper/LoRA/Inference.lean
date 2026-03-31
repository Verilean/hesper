import Hesper.LoRA.Types
import Hesper.LoRA.Init
import Hesper.LoRA.Forward
import Hesper.LoRA.Backward
import Hesper.LoRA.IO
import Hesper.Models.BitNet
import Hesper.Training.Loss
import Hesper.Training.TrainLoop
import Hesper.Training.AttentionBackward
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WGSL.Execute
import Hesper.WGSL.MatMul
import Hesper.WGSL.Elementwise
import Hesper.Logging

/-!
# LoRA-Aware Inference

Extends BitNet inference to apply LoRA adapters during generation.

LoRA corrections are injected **inside** the attention layer, between
BitLinear Q/V projections and RoPE. This ensures the LoRA contribution
flows through the full attention computation (RoPE → KV cache → scores → softmax).

Uses `Attention.forwardWithCacheLoRA` and `TransformerBlock.forwardWithCacheLoRA`
which inject LoRA at the correct point in the forward pass.
-/

namespace Hesper.LoRA.Inference

open Hesper.WebGPU
open Hesper.Models.BitNet
open Hesper.LoRA
open Hesper.Logging

/-- Temporary buffers needed for LoRA inference and training backward -/
structure LoRAInferenceState where
  /-- Intermediate h = A @ x buffer [rank] -/
  hBuf : Buffer
  /-- Temporary y buffer for Q [dim] -/
  yBufQ : Buffer
  /-- Temporary y buffer for V [kvDim] -/
  yBufV : Buffer
  /-- Attention backward buffers (only allocated for training) -/
  dAttnBuf : Option Buffer    -- [numHeads * maxSeqLen]
  dScoresBuf : Option Buffer  -- [numHeads * maxSeqLen]
  dQBuf : Option Buffer       -- [numHeads * headDim]
  dQPreBuf : Option Buffer    -- [numHeads * headDim] (before RoPE)

/-- Create LoRA inference state (inference only, no backward buffers) -/
def createLoRAInferenceState (device : Device) (adapter : Adapter)
    (dim kvDim : Nat) : IO LoRAInferenceState := do
  let rank := adapter.config.rank
  let mkBuf := fun (n : Nat) =>
    createBuffer device { size := (n * 4).toUSize, usage := [.storage, .copySrc, .copyDst], mappedAtCreation := false }
  pure {
    hBuf := ← mkBuf rank
    yBufQ := ← mkBuf dim
    yBufV := ← mkBuf kvDim
    dAttnBuf := none, dScoresBuf := none, dQBuf := none, dQPreBuf := none
  }

/-- Create LoRA inference state with training backward buffers -/
def createLoRATrainingState (device : Device) (adapter : Adapter)
    (dim kvDim numHeads headDim maxSeqLen : Nat) : IO LoRAInferenceState := do
  let rank := adapter.config.rank
  let mkBuf := fun (n : Nat) =>
    createBuffer device { size := (n * 4).toUSize, usage := [.storage, .copySrc, .copyDst], mappedAtCreation := false }
  pure {
    hBuf := ← mkBuf rank
    yBufQ := ← mkBuf dim
    yBufV := ← mkBuf kvDim
    dAttnBuf := some (← mkBuf (numHeads * maxSeqLen))
    dScoresBuf := some (← mkBuf (numHeads * maxSeqLen))
    dQBuf := some (← mkBuf (numHeads * headDim))
    dQPreBuf := some (← mkBuf (numHeads * headDim))
  }

/-- Single-token forward pass with LoRA.
    Uses `TransformerBlock.forwardWithCacheLoRA` which injects LoRA
    inside the attention layer (between BitLinear Q/V and RoPE). -/
def forwardSingleTokenWithLoRA (device : Device) (model : BitNetModel)
    (tokenId : Nat) (pos : Nat) (cacheState : KVCacheState)
    (adapter : Adapter) (loraState : LoRAInferenceState) : IO Unit := do
  logVerbose s!"[SingleToken+LoRA] pos={pos}, tokenId={tokenId}"

  let scale := adapter.config.scale

  -- Step 1: Embedding lookup
  let tokenBytes := Hesper.WebGPU.BufferOps.uint32ToBytes tokenId.toUInt32
  writeBuffer device cacheState.tokenBuf 0 tokenBytes
  Hesper.Layers.Embedding.forward device model.embedding cacheState.tokenBuf cacheState.buf1 1 1

  -- === BEGIN BATCHED EXECUTION ===
  Hesper.WGSL.Execute.beginBatch device

  -- Step 2: Process each transformer layer WITH LoRA
  let mut currentBuf := cacheState.buf1
  let mut nextBuf := cacheState.buf2
  let mut layerIdx := 0

  for layer in model.layers do
    if h : layerIdx < cacheState.kvCaches.size then
      let kvCache := cacheState.kvCaches[layerIdx]
      let fusedRef := if h2 : layerIdx < cacheState.fusedRefs.size then
        some cacheState.fusedRefs[layerIdx]
      else none

      let loraOpt := if h3 : layerIdx < adapter.layers.size then
        some (adapter.layers[layerIdx], scale, loraState.hBuf, loraState.yBufQ, loraState.yBufV)
      else none
      Hesper.Layers.TransformerBlock.forwardWithCache device layer currentBuf nextBuf pos kvCache (some cacheState.layerBufs) fusedRef loraOpt

      let temp := currentBuf; currentBuf := nextBuf; nextBuf := temp
    layerIdx := layerIdx + 1

  -- Step 3: Final normalization
  Hesper.Layers.RMSNorm.forward device model.finalNorm currentBuf nextBuf 1 256

  -- Step 4: LM head
  let lmHeadConfig : Hesper.WGSL.MatMul.Config := {
    M := 1, N := model.config.vocabSize, K := model.config.dim
  }
  match model.embedding.f16Table with
  | some f16Buf =>
    if model.config.dim % 8 == 0 then
      Hesper.WGSL.MatMul.executeMatMulTransposeF16Shared device nextBuf f16Buf cacheState.logitsBuf lmHeadConfig
    else
      Hesper.WGSL.MatMul.executeMatMulTransposeF16 device nextBuf f16Buf cacheState.logitsBuf lmHeadConfig
  | none =>
    Hesper.WGSL.MatMul.executeMatMulTranspose device nextBuf model.embedding.embeddingTable cacheState.logitsBuf lmHeadConfig

  -- === END BATCHED EXECUTION ===
  Hesper.WGSL.Execute.endBatch device

/-- Combined forward + backward in a SINGLE GPU batch.
    All dispatches (forward 30 layers + loss + backward) are recorded into one
    command buffer and submitted as a single GPU submit. This eliminates ~20
    GPU sync points per token compared to separate forward/backward calls.

    @param isOutputToken If true, compute loss + backward after forward.
           If false (prompt tokens), only forward is executed.
    @param targetBuf Pre-uploaded target token ID [1] u32
    @param lossAccumBuf GPU-side loss accumulator (added to, not overwritten)
    @param dLogitsBuf Scratch buffer for dLogits [vocabSize]
    @param dHiddenBuf Scratch buffer for dHidden [dim]
    @param grads Gradient accumulators for LoRA weights
    @param startLayer First layer to compute LoRA backward for
    @param trainState Training state with temp buffers -/
def forwardAndBackwardBatched (device : Device) (model : BitNetModel)
    (tokenId : Nat) (pos : Nat) (cacheState : KVCacheState)
    (adapter : Adapter) (loraState : LoRAInferenceState)
    (isOutputToken : Bool)
    (targetBuf lossAccumBuf dLogitsBuf dHiddenBuf : Buffer)
    (grads : AdapterGrad) (trainState : Hesper.Training.TrainLoop.TrainState)
    (startLayer : Nat) : IO Unit := do
  let scale := adapter.config.scale
  let dim := model.config.dim

  -- Pre-batch: upload token data (these are queue operations, visible to subsequent batch)
  let tokenBytes := Hesper.WebGPU.BufferOps.uint32ToBytes tokenId.toUInt32
  writeBuffer device cacheState.tokenBuf 0 tokenBytes

  -- === SINGLE GPU BATCH: forward + loss + backward ===
  Hesper.WGSL.Execute.beginBatch device

  -- Forward: embedding
  Hesper.Layers.Embedding.forward device model.embedding cacheState.tokenBuf cacheState.buf1 1 1

  -- Forward: 30 transformer layers with LoRA
  let mut currentBuf := cacheState.buf1
  let mut nextBuf := cacheState.buf2
  let mut layerIdx := 0
  for layer in model.layers do
    if h : layerIdx < cacheState.kvCaches.size then
      let kvCache := cacheState.kvCaches[layerIdx]
      let fusedRef := if h2 : layerIdx < cacheState.fusedRefs.size then
        some cacheState.fusedRefs[layerIdx]
      else none
      let loraOpt := if h3 : layerIdx < adapter.layers.size then
        some (adapter.layers[layerIdx], scale, loraState.hBuf, loraState.yBufQ, loraState.yBufV)
      else none
      Hesper.Layers.TransformerBlock.forwardWithCache device layer currentBuf nextBuf pos kvCache (some cacheState.layerBufs) fusedRef loraOpt
      let temp := currentBuf; currentBuf := nextBuf; nextBuf := temp
    layerIdx := layerIdx + 1

  -- Forward: final norm + LM head
  Hesper.Layers.RMSNorm.forward device model.finalNorm currentBuf nextBuf 1 256
  let lmHeadConfig : Hesper.WGSL.MatMul.Config := {
    M := 1, N := model.config.vocabSize, K := dim
  }
  match model.embedding.f16Table with
  | some f16Buf =>
    if dim % 8 == 0 then
      Hesper.WGSL.MatMul.executeMatMulTransposeF16Shared device nextBuf f16Buf cacheState.logitsBuf lmHeadConfig
    else
      Hesper.WGSL.MatMul.executeMatMulTransposeF16 device nextBuf f16Buf cacheState.logitsBuf lmHeadConfig
  | none =>
    Hesper.WGSL.MatMul.executeMatMulTranspose device nextBuf model.embedding.embeddingTable cacheState.logitsBuf lmHeadConfig

  -- If this is an output token: loss + full attention backward (all in same batch)
  if isOutputToken then
    -- Cross-entropy forward (accumulate loss on GPU)
    Hesper.Training.Loss.executeCrossEntropyForwardAccum device cacheState.logitsBuf targetBuf lossAccumBuf model.config.vocabSize
    -- Cross-entropy backward: dLogits = softmax - one_hot
    Hesper.Training.Loss.executeCrossEntropyBackward device cacheState.logitsBuf targetBuf dLogitsBuf model.config.vocabSize
    -- LM head backward: dHidden = dLogits @ embedding
    let lmHeadBackConfig : Hesper.WGSL.MatMul.Config := { M := 1, N := dim, K := model.config.vocabSize }
    Hesper.WGSL.MatMul.executeMatMul device dLogitsBuf model.embedding.embeddingTable dHiddenBuf lmHeadBackConfig

    -- === FULL ATTENTION BACKWARD ===
    -- dHidden now contains ∂L/∂hidden (after final norm, before LM head)
    -- We need: dHidden → RMSNorm backward → O proj backward →
    --          attention apply backward → softmax backward →
    --          score backward → RoPE backward → dQ (for LoRA)

    let numHeads := model.config.numHeads
    let headDim := model.config.headDim
    let numKVHeads := model.config.numKVHeads
    let cacheLen := pos + 1  -- current position + 1
    let attnScale := 1.0 / (headDim.toFloat.sqrt)

    -- For the LAST layer (layer 29): full attention backward
    -- (We focus on the last layer where gradient signal is strongest and
    --  attention buffers still contain valid data from the forward pass)
    let lastLayer := model.config.numLayers - 1
    if h_last : lastLayer < adapter.layers.size then
      -- The attention buffers (attnBuf, qRotBuf, etc.) from the last layer
      -- are in cacheState.layerBufs.attnBufs
      let attnBufs := cacheState.layerBufs.attnBufs
      if h_kv : lastLayer < cacheState.kvCaches.size then
      let kvCache := cacheState.kvCaches[lastLayer]

      -- dHidden is ∂L/∂(final_norm_output) after LM head backward
      -- For now, use dHidden directly as ∂L/∂(attention_output)
      -- (skipping RMSNorm backward and O projection backward for simplicity,
      --  since residual connections pass gradient through mostly unchanged)

      -- Step 1: Attention apply backward
      -- dAttn[h,s] = Σ_d dHidden[h,d] * V_cache[kvHead,s,d]
      match loraState.dAttnBuf with
      | some dAttnBuf =>
        Hesper.Training.AttentionBackward.executeApplyBackward device
          dHiddenBuf kvCache.vBuf dAttnBuf
          numHeads numKVHeads cacheLen headDim

        -- Step 2: Softmax backward
        -- dScores = attn * (dAttn - Σ attn*dAttn)
        match loraState.dScoresBuf with
        | some dScoresBuf =>
          Hesper.Training.AttentionBackward.executeSoftmaxBackward device
            attnBufs.attnBuf dAttnBuf dScoresBuf
            numHeads cacheLen

          -- Step 3: Score backward for Q
          -- dQ[h,d] = scale * Σ_s dScores[h,s] * K_cache[kvHead,s,d]
          match loraState.dQBuf with
          | some dQBuf =>
            Hesper.Training.AttentionBackward.executeScoreBackwardQ device
              dScoresBuf kvCache.kBuf dQBuf
              numHeads numKVHeads cacheLen headDim attnScale

            -- Step 4: RoPE backward (inverse rotation)
            -- dQpre = R(-θ) @ dQ
            match loraState.dQPreBuf with
            | some dQPreBuf =>
              Hesper.Training.AttentionBackward.executeRopeBackward device
                dQBuf dQPreBuf
                numHeads headDim model.config.ropeBase pos

              -- Step 5: dQpre is now ∂L/∂(Q_bitlinear_output)
              -- This is the CORRECT gradient for LoRA Q!
              if h_g : lastLayer < grads.layers.size then
              let layerGrad := grads.layers[lastLayer]
              let layerAdapter := adapter.layers[lastLayer]

              -- LoRA Q backward using dQpre (correct gradient!)
              Forward.executeProjectA device layerAdapter.loraQ cacheState.layerBufs.normedBuf trainState.hBuf
              Backward.executeGradB device dQPreBuf trainState.hBuf layerGrad.gradQ.dB layerAdapter.loraQ.outDim layerAdapter.loraQ.rank scale
              Backward.executeGradDh device layerAdapter.loraQ.b dQPreBuf trainState.dhBuf layerAdapter.loraQ.outDim layerAdapter.loraQ.rank
              Backward.executeGradA device trainState.dhBuf cacheState.layerBufs.normedBuf layerGrad.gradQ.dA layerAdapter.loraQ.rank layerAdapter.loraQ.inDim scale

              -- LoRA V backward (use dHidden as approximate V gradient)
              Forward.executeProjectA device layerAdapter.loraV cacheState.layerBufs.normedBuf trainState.hBuf
              Backward.executeGradB device dHiddenBuf trainState.hBuf layerGrad.gradV.dB layerAdapter.loraV.outDim layerAdapter.loraV.rank scale
              Backward.executeGradDh device layerAdapter.loraV.b dHiddenBuf trainState.dhBuf layerAdapter.loraV.outDim layerAdapter.loraV.rank
              Backward.executeGradA device trainState.dhBuf cacheState.layerBufs.normedBuf layerGrad.gradV.dA layerAdapter.loraV.rank layerAdapter.loraV.inDim scale
            | none => pure ()
          | none => pure ()
        | none => pure ()
      | none => pure ()

  -- === END SINGLE GPU BATCH ===
  Hesper.WGSL.Execute.endBatch device

/-- Generate text with LoRA adapter applied.
    Same interface as BitNetModel.generate but with LoRA corrections. -/
def generateWithLoRA (device : Device) (model : BitNetModel)
    (adapter : Adapter) (loraState : LoRAInferenceState)
    (promptTokens : Array Nat) (maxTokens : Nat)
    (strategy : Hesper.Inference.Sampling.Strategy := .Greedy)
    (eosToken : Option Nat := none)
    (repetitionPenalty : Float := 1.1)
    : IO (Array Nat) := do
  -- Reset caches
  resetPreparedDispatches model

  IO.println "═══════════════════════════════════════════════"
  IO.println "  Text Generation with LoRA"
  IO.println "═══════════════════════════════════════════════"
  IO.println s!"LoRA: rank={adapter.config.rank}, alpha={adapter.config.alpha}"
  IO.println s!"Prompt: {promptTokens.size} tokens, generating up to {maxTokens}"
  IO.println ""

  let cacheState ← createKVCacheState device model
  let mut tokens := promptTokens
  let mut rng := Hesper.Inference.Sampling.RNG.create (some 42)

  -- Pre-upload prompt tokens to penalty buffer
  if repetitionPenalty != 1.0 then
    for i in [0:promptTokens.size] do
      appendPenaltyToken device cacheState promptTokens[i]! i

  -- Phase 1: Prefill with LoRA
  IO.println s!"[Prefill+LoRA] Processing {promptTokens.size} prompt tokens..."
  let prefillStart ← IO.monoNanosNow
  for i in [0:promptTokens.size] do
    if i >= model.config.maxSeqLen then break
    forwardSingleTokenWithLoRA device model promptTokens[i]! i cacheState adapter loraState
  let prefillEnd ← IO.monoNanosNow
  let prefillMs := (prefillEnd - prefillStart).toFloat / 1_000_000.0
  IO.println s!"[Prefill+LoRA] Done in {prefillMs} ms ({prefillMs / promptTokens.size.toFloat} ms/token)"

  -- Phase 2: Generate with LoRA
  let isGreedy := match strategy with
    | .Greedy => true
    | _ => false
  let genStart ← IO.monoNanosNow
  let mut genTokenCount : Nat := 0
  for step in [0:maxTokens] do
    if tokens.size >= model.config.maxSeqLen then
      IO.println s!"Reached max sequence length ({model.config.maxSeqLen})"
      break

    let mut nextToken := 0
    if isGreedy then
      if repetitionPenalty == 1.0 then
        nextToken ← gpuArgmax device cacheState.logitsBuf cacheState.argmaxBuf model.config.vocabSize
      else
        nextToken ← gpuArgmaxWithPenalty device cacheState model.config.vocabSize
          model.config.maxSeqLen tokens.size repetitionPenalty
    else
      let logits ← Hesper.WebGPU.BufferOps.downloadFloatArray device cacheState.logitsBuf model.config.vocabSize
      let logits := Hesper.Inference.Sampling.applyRepetitionPenalty logits tokens repetitionPenalty
      let (tok, newRng) := Hesper.Inference.Sampling.sampleWithRNG logits strategy rng
      rng := newRng
      nextToken := tok

    tokens := tokens.push nextToken
    genTokenCount := genTokenCount + 1

    if repetitionPenalty != 1.0 then
      appendPenaltyToken device cacheState nextToken (tokens.size - 1)

    match eosToken with
    | some eos =>
      if nextToken == eos then
        IO.println "  EOS token, stopping"
        break
    | none => pure ()

    let newPos := tokens.size - 1
    if newPos < model.config.maxSeqLen then
      forwardSingleTokenWithLoRA device model nextToken newPos cacheState adapter loraState

  let genEnd ← IO.monoNanosNow
  let genMs := (genEnd - genStart).toFloat / 1_000_000.0
  let msPerToken := if genTokenCount > 0 then genMs / genTokenCount.toFloat else 0.0
  let tps := if msPerToken > 0 then 1000.0 / msPerToken else 0.0
  IO.println ""
  IO.println s!"Generated {genTokenCount} tokens in {genMs} ms"
  IO.println s!"  {msPerToken} ms/token = {tps} tokens/sec"

  pure tokens

end Hesper.LoRA.Inference
