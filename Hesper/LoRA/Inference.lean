import Hesper.LoRA.Types
import Hesper.LoRA.Init
import Hesper.LoRA.Forward
import Hesper.LoRA.Backward
import Hesper.LoRA.IO
import Hesper.Models.BitNet
import Hesper.Training.Loss
import Hesper.Training.TrainLoop
import Hesper.Training.AttentionBackward
import Hesper.Training.BitLinearBackward
import Hesper.Training.FFNBackward
import Hesper.WGSL.Fusion
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
  /-- Per-layer saved normedBuf for multi-layer backward.
      savedNormed[i] = copy of normedBuf after RMSNorm, before attention layer i.
      This is the input to LoRA Q/V projections and is needed for gradient computation. -/
  savedNormed : Array Buffer  -- [numLayers] × [dim]
  /-- Per-layer saved attention weights for softmax backward.
      savedAttn[i] = copy of attnBuf (softmax output) for layer i.
      Needed for correct softmax backward: dScores = attn * (dAttn - Σ attn*dAttn) -/
  savedAttn : Array Buffer    -- [numLayers] × [numHeads * maxSeqLen]
  /-- Per-layer saved attention output (before sub-norm) for RMSNorm backward.
      savedAttnOut[i] = copy of qRotBuf after attention apply (= input to sub-norm).
      Needed for RMSNorm backward in the attention chain. -/
  savedAttnOut : Array Buffer -- [numLayers] × [numHeads * headDim]
  /-- Scratch buffer for dAttnOut (gradient after O backward, before RMSNorm backward) -/
  dAttnOutBuf : Option Buffer -- [numHeads * headDim]
  /-- Per-layer saved FFN activations for FFN backward -/
  savedGate : Array Buffer     -- [numLayers] × [ffnDim]
  savedUp : Array Buffer       -- [numLayers] × [ffnDim]
  savedHidden : Array Buffer   -- [numLayers] × [ffnDim] (pre sub-norm)
  savedResidual1 : Array Buffer -- [numLayers] × [dim] (pre ffn-norm)
  /-- Scratch buffers for FFN backward -/
  dFFNNormed : Option Buffer   -- [ffnDim]
  dFFNHidden : Option Buffer   -- [ffnDim]
  dGateBuf : Option Buffer     -- [ffnDim]
  dUpBuf : Option Buffer       -- [ffnDim]
  dNormed2Buf : Option Buffer  -- [dim]

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
    savedNormed := #[], savedAttn := #[], savedAttnOut := #[], dAttnOutBuf := none
    savedGate := #[], savedUp := #[], savedHidden := #[], savedResidual1 := #[]
    dFFNNormed := none, dFFNHidden := none, dGateBuf := none, dUpBuf := none, dNormed2Buf := none
  }

/-- Create LoRA inference state with training backward buffers -/
def createLoRATrainingState (device : Device) (adapter : Adapter)
    (dim kvDim numHeads headDim maxSeqLen numLayers : Nat) : IO LoRAInferenceState := do
  let rank := adapter.config.rank
  let mkBuf := fun (n : Nat) =>
    createBuffer device { size := (n * 4).toUSize, usage := [.storage, .copySrc, .copyDst], mappedAtCreation := false }
  -- Allocate per-layer saved buffers for multi-layer backward
  let mut savedNormed := #[]
  let mut savedAttn := #[]
  let mut savedAttnOut := #[]
  let mut savedGate := #[]
  let mut savedUp := #[]
  let mut savedHidden := #[]
  let mut savedResidual1 := #[]
  let ffnDim := dim * 27 / 10  -- 2560 * 2.7 = 6912 (BitNet FFN ratio)
  for _ in [:numLayers] do
    savedNormed := savedNormed.push (← mkBuf dim)
    savedAttn := savedAttn.push (← mkBuf (numHeads * maxSeqLen))
    savedAttnOut := savedAttnOut.push (← mkBuf (numHeads * headDim))
    savedGate := savedGate.push (← mkBuf ffnDim)
    savedUp := savedUp.push (← mkBuf ffnDim)
    savedHidden := savedHidden.push (← mkBuf ffnDim)
    savedResidual1 := savedResidual1.push (← mkBuf dim)
  pure {
    hBuf := ← mkBuf rank
    yBufQ := ← mkBuf dim
    yBufV := ← mkBuf kvDim
    dAttnBuf := some (← mkBuf (numHeads * maxSeqLen))
    dScoresBuf := some (← mkBuf (numHeads * maxSeqLen))
    dQBuf := some (← mkBuf (numHeads * headDim))
    dQPreBuf := some (← mkBuf (numHeads * headDim))
    savedNormed, savedAttn, savedAttnOut
    savedGate, savedUp, savedHidden, savedResidual1
    dAttnOutBuf := some (← mkBuf (numHeads * headDim))
    dFFNNormed := some (← mkBuf ffnDim)
    dFFNHidden := some (← mkBuf ffnDim)
    dGateBuf := some (← mkBuf ffnDim)
    dUpBuf := some (← mkBuf ffnDim)
    dNormed2Buf := some (← mkBuf dim)
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
      -- Output tokens: non-fused FFN (need gate/up buffers for FFN backward)
      -- Prompt tokens: use fused FFN for speed (no backward needed)
      let fusedRef := if isOutputToken then none
        else if h2 : layerIdx < cacheState.fusedRefs.size then
          some cacheState.fusedRefs[layerIdx]
        else none
      -- LoRA forward always active (weights affect output for all tokens)
      let loraOpt := if h3 : layerIdx < adapter.layers.size then
        some (adapter.layers[layerIdx], scale, loraState.hBuf, loraState.yBufQ, loraState.yBufV)
      else none
      Hesper.Layers.TransformerBlock.forwardWithCache device layer currentBuf nextBuf pos kvCache (some cacheState.layerBufs) fusedRef loraOpt

      -- Save activations for multi-layer backward (gradient checkpointing)
      if isOutputToken then
        -- Save activations (individual copies for reliability)
        if h_sn : layerIdx < loraState.savedNormed.size then
          Forward.saveActivation device cacheState.layerBufs.normedBuf loraState.savedNormed[layerIdx] dim
        if h_sa : layerIdx < loraState.savedAttn.size then
          let attnSize := model.config.numHeads * (pos + 1)
          Forward.saveActivation device cacheState.layerBufs.attnBufs.attnBuf loraState.savedAttn[layerIdx] attnSize
        if h_ao : layerIdx < loraState.savedAttnOut.size then
          Forward.saveActivation device cacheState.layerBufs.attnBufs.qRotBuf loraState.savedAttnOut[layerIdx] (model.config.numHeads * model.config.headDim)
        if h_sg : layerIdx < loraState.savedGate.size then
          Forward.saveActivation device cacheState.layerBufs.gateBuf loraState.savedGate[layerIdx] model.config.ffnDim
        if h_su : layerIdx < loraState.savedUp.size then
          Forward.saveActivation device cacheState.layerBufs.upBuf loraState.savedUp[layerIdx] model.config.ffnDim
        if h_sh : layerIdx < loraState.savedHidden.size then
          Forward.saveActivation device cacheState.layerBufs.hiddenBuf loraState.savedHidden[layerIdx] model.config.ffnDim
        if h_sr : layerIdx < loraState.savedResidual1.size then
          Forward.saveActivation device cacheState.layerBufs.residual1Buf loraState.savedResidual1[layerIdx] dim

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
    -- LM head backward: dNormOut = dLogits @ embedding
    let lmHeadBackConfig : Hesper.WGSL.MatMul.Config := { M := 1, N := dim, K := model.config.vocabSize }
    Hesper.WGSL.MatMul.executeMatMul device dLogitsBuf model.embedding.embeddingTable dHiddenBuf lmHeadBackConfig

    -- Final RMSNorm backward: dHidden = RMSNorm_bwd(lastLayerOutput, finalNorm.scale, dNormOut)
    -- currentBuf still holds the last layer's output (= final RMSNorm input)
    -- dHiddenBuf holds dNormOut, we write dHidden to dLogitsBuf (as temp, it's no longer needed)
    Hesper.Training.AttentionBackward.executeRmsNormBackward device
      currentBuf model.finalNorm.scale dHiddenBuf dLogitsBuf dim
    -- Copy result back to dHiddenBuf (swap dLogitsBuf → dHiddenBuf would work but
    -- dLogitsBuf is [vocabSize] and dHiddenBuf is [dim], sizes differ. Use saveActivation copy)
    Forward.saveActivation device dLogitsBuf dHiddenBuf dim

    -- === FULL MULTI-LAYER ATTENTION BACKWARD ===
    -- dHidden contains ∂L/∂hidden after LM head backward.
    -- We iterate ALL layers (reverse order) and compute LoRA gradients
    -- using per-layer saved normedBuf and per-layer KV cache.
    --
    -- Key insight: residual connections pass dHidden unchanged through
    -- non-LoRA components. At each LoRA layer, we compute the attention
    -- backward chain to get dQ, then compute LoRA gradients.

    let numHeads := model.config.numHeads
    let headDim := model.config.headDim
    let numKVHeads := model.config.numKVHeads
    let cacheLen := pos + 1
    let attnScale := 1.0 / (headDim.toFloat.sqrt)

    match loraState.dAttnBuf, loraState.dScoresBuf, loraState.dQBuf, loraState.dQPreBuf with
    | some dAttnBuf, some dScoresBuf, some dQBuf, some dQPreBuf =>
      -- Iterate layers in reverse (gradient flows backward)
      for li_rev in [:model.config.numLayers] do
        let li := model.config.numLayers - 1 - li_rev
        if h_a : li < adapter.layers.size then
          if h_g : li < grads.layers.size then
            if h_kv : li < cacheState.kvCaches.size then
              if h_sn : li < loraState.savedNormed.size then
                if h_sa : li < loraState.savedAttn.size then
                  if h_ao : li < loraState.savedAttnOut.size then
                let layerAdapter := adapter.layers[li]
                let layerGrad := grads.layers[li]
                let kvCache := cacheState.kvCaches[li]
                let savedNorm := loraState.savedNormed[li]
                let savedAttnWeights := loraState.savedAttn[li]

                -- Attention backward chain (verified specs in VerifiedBackward.lean):
                -- dHidden → RMSNorm backward (sub-norm) → apply backward →
                --   softmax backward → score backward → RoPE backward → dQ

                let savedAttnOutput := loraState.savedAttnOut[li]
                let dim := model.config.dim

                -- Step 0: RMSNorm backward (sub-norm)
                -- dHidden is ∂L/∂(O_projection_output). Through residual connection,
                -- it's also ∂L/∂(sub-norm output) (approximately, skipping O backward).
                -- RMSNorm backward: dAttnOut = RMSNorm_backward(savedAttnOut, gamma, dHidden)
                -- Step 0a: O projection backward: dSubNormOut = W_O^T @ dHidden
                -- Step 0b: RMSNorm backward: dAttnOut = RMSNorm_bwd(attnOutput, gamma, dSubNormOut)
                let dForApply ← match loraState.dAttnOutBuf with
                | some dAttnOutBuf =>
                  if h_layer : li < model.layers.size then
                    -- O projection backward: dAttnOutBuf = scale * W_O^T @ dHidden
                    let wO := model.layers[li].attention.wO
                    Hesper.Training.BitLinearBackward.executeBitLinearTranspose device
                      wO dHiddenBuf dAttnOutBuf
                    -- RMSNorm backward (sub-norm): dAttnOut → dAttnWeighted
                    let subNormScale := model.layers[li].attnSubNorm.scale
                    Hesper.Training.AttentionBackward.executeRmsNormBackward device
                      savedAttnOutput subNormScale dAttnOutBuf dScoresBuf
                      dim
                    pure dScoresBuf
                  else pure dHiddenBuf
                | none => pure dHiddenBuf

                -- Step 1: dAttn[h,s] = Σ_d dForApply[h,d] * V[kvHead,s,d]
                Hesper.Training.AttentionBackward.executeApplyBackward device
                  dForApply kvCache.vBuf dAttnBuf
                  numHeads numKVHeads cacheLen headDim

                -- Step 2: PROPER softmax backward using saved per-layer attention weights
                -- dScores[h,s] = attn[h,s] * (dAttn[h,s] - Σ_s' attn[h,s'] * dAttn[h,s'])
                Hesper.Training.AttentionBackward.executeSoftmaxBackward device
                  savedAttnWeights dAttnBuf dScoresBuf
                  numHeads cacheLen

                -- Step 3: dQ[h,d] = scale * Σ_s dScores[h,s] * K[kvHead,s,d]
                Hesper.Training.AttentionBackward.executeScoreBackwardQ device
                  dScoresBuf kvCache.kBuf dQBuf
                  numHeads numKVHeads cacheLen headDim attnScale

                -- Step 4: RoPE backward
                Hesper.Training.AttentionBackward.executeRopeBackward device
                  dQBuf dQPreBuf
                  numHeads headDim model.config.ropeBase pos

                -- Step 5: LoRA Q backward using dQpre + saved normedBuf
                Forward.executeProjectA device layerAdapter.loraQ savedNorm trainState.hBuf
                Backward.executeGradB device dQPreBuf trainState.hBuf layerGrad.gradQ.dB layerAdapter.loraQ.outDim layerAdapter.loraQ.rank scale
                Backward.executeGradDh device layerAdapter.loraQ.b dQPreBuf trainState.dhBuf layerAdapter.loraQ.outDim layerAdapter.loraQ.rank
                Backward.executeGradA device trainState.dhBuf savedNorm layerGrad.gradQ.dA layerAdapter.loraQ.rank layerAdapter.loraQ.inDim scale
                -- Note: dInput propagation through LoRA is not needed for residual backward.
                -- Residual connections pass dHidden unchanged; LoRA dInput only affects
                -- the LoRA parameter gradients (dA, dB), not the residual stream.

                -- Step 6: LoRA V backward using dForApply + saved normedBuf
                Forward.executeProjectA device layerAdapter.loraV savedNorm trainState.hBuf
                Backward.executeGradB device dForApply trainState.hBuf layerGrad.gradV.dB layerAdapter.loraV.outDim layerAdapter.loraV.rank scale
                Backward.executeGradDh device layerAdapter.loraV.b dForApply trainState.dhBuf layerAdapter.loraV.outDim layerAdapter.loraV.rank
                Backward.executeGradA device trainState.dhBuf savedNorm layerGrad.gradV.dA layerAdapter.loraV.rank layerAdapter.loraV.inDim scale
                -- Step 7: FFN backward
                if h_layer2 : li < model.layers.size then
                  if h_sg : li < loraState.savedGate.size then
                    if h_su : li < loraState.savedUp.size then
                      if h_sh : li < loraState.savedHidden.size then
                        if h_sr : li < loraState.savedResidual1.size then
                          match loraState.dFFNNormed, loraState.dFFNHidden, loraState.dGateBuf, loraState.dUpBuf, loraState.dNormed2Buf with
                          | some dFFNN, some dFFNH, some dG, some dU, some dN2 =>
                            let block := model.layers[li]
                            Hesper.Training.FFNBackward.executeFFNBackward device
                              block.ffnDown block.ffnGate block.ffnUp
                              block.ffnSubNorm.scale block.ffnNorm.scale
                              dHiddenBuf
                              loraState.savedHidden[li] loraState.savedResidual1[li]
                              loraState.savedGate[li] loraState.savedUp[li]
                              dFFNN dFFNH dG dU dN2 dHiddenBuf
                              dim model.config.ffnDim
                            -- dHiddenBuf now contains FFN's contribution to dResidual
                            -- Add it back to dHidden for the next (lower) layer
                            -- (The FFN backward writes to dHiddenBuf, which is used
                            --  as dOutput for the next layer iteration)
                          | _, _, _, _, _ => pure ()
    | _, _, _, _ => pure ()

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
