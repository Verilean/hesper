import Hesper.Models.BitNet
import Hesper.TTT.Types
import Hesper.TTT.Kernels
import Hesper.Training.SafeBuffer
import Hesper.Optimizer.AdamGPU
import Hesper.WebGPU.BufferOps
import Hesper.Backend.WebGPU
import Hesper.Inference.Sampling
import Hesper.WGSL.MatMul

/-!
# Hidden-Space MSE TTT for BitNet

Inspired by Stanford's original TTT paper, this variant completely
bypasses the LM Head for the TTT learning phase. Instead of predicting
the next token ID via Cross-Entropy (which dilutes gradients through
128K-dim softmax), the TTT module learns to predict the **residual
difference to the next token's hidden state** using MSE:

```
ttt_output = W_ttt @ hidden_t
target_residual = hidden_{t+1} - hidden_t
loss = MSE(ttt_output, target_residual)
gradient = (2/dim) * (ttt_output - target_residual)
```

Advantages over CE-based TTT:
- Gradient flows directly in dim-space (2560), not through 128K softmax
- MSE gradients are stable and proportional to reconstruction error
- No LM head backward needed (saves compute + avoids dilution)
- W_ttt is [dim×dim] = 26 MB (unchanged from CE variant)

At decode time, the frozen TTT weights correct the hidden state:
```
hidden' = hidden + W_ttt @ hidden
logits = LM_head(hidden')
```

Does NOT modify Hesper.Models.BitNet.
-/

namespace Hesper.TTT.HiddenSpace

open Hesper.WebGPU
open Hesper.Models.BitNet
open Hesper.TTT.Kernels
open Hesper.Inference
open Hesper.Training
open Hesper.Optimizer

/-- Hidden-space MSE TTT config. -/
structure HiddenTTTConfig where
  dim : Nat
  vocabSize : Nat     -- needed for decode-phase LM head
  innerLR : Float
  tau : Float          -- MSE threshold for gate (much smaller than CE: ~0.001-0.01)
  useAdam : Bool := true  -- Adam optimizer (momentum for noise resistance)
  adamBeta1 : Float := 0.9
  adamBeta2 : Float := 0.999
  adamEps : Float := 1e-7
  deriving Inhabited, Repr

/-- GPU buffers for hidden-space MSE TTT. -/
structure HiddenTTTBuffers where
  tttWeightBuf : Buffer       -- [dim × dim] f32, mutable, zero-init
  prevHiddenBuf : Buffer      -- [dim] f32, hidden_t (previous step)
  tttOutputBuf : Buffer       -- [dim] f32, W_ttt @ hidden_t
  correctedHiddenBuf : Buffer -- [dim] f32, hidden + W_ttt @ hidden (decode)
  gradBuf : Buffer            -- [dim] f32, d_ttt_output
  dWeightBuf : Buffer         -- [dim × dim] f32, outer product
  lossBuf : Buffer            -- [1] f32, scalar MSE loss
  -- Adam state (m and v for W_ttt)
  adamMBuf : Buffer           -- [dim × dim] f32, first moment
  adamVBuf : Buffer           -- [dim × dim] f32, second moment

def createHiddenTTTBuffers (device : Device) (config : HiddenTTTConfig) : IO HiddenTTTBuffers := do
  let mkF32Buf := fun (n : Nat) => createBuffer device {
    size := (n * 4).toUSize
    usage := [.storage, .copySrc, .copyDst]
    mappedAtCreation := false
  }
  let dimSq := config.dim * config.dim
  return {
    tttWeightBuf := ← do
      let buf ← mkF32Buf dimSq
      let mut zeros := ByteArray.empty
      for _ in [0:dimSq * 4] do
        zeros := zeros.push 0
      writeBuffer device buf 0 zeros
      pure buf
    prevHiddenBuf := ← mkF32Buf config.dim
    tttOutputBuf := ← mkF32Buf config.dim
    correctedHiddenBuf := ← mkF32Buf config.dim
    gradBuf := ← mkF32Buf config.dim
    dWeightBuf := ← mkF32Buf dimSq
    lossBuf := ← mkF32Buf 1
    -- Adam momentum buffers (zero-initialized by GPU default)
    adamMBuf := ← mkF32Buf dimSq
    adamVBuf := ← mkF32Buf dimSq
  }

/-- Which buffer holds post-norm hidden after forwardSingleToken -/
def postNormBuf (cacheState : KVCacheState) (numLayers : Nat) : Buffer :=
  if numLayers % 2 == 0 then cacheState.buf2 else cacheState.buf1

/-- One MSE-based hidden-space TTT step.

    Takes the previous hidden state (hidden_t) already in prevHiddenBuf
    and the current hidden state (hidden_{t+1}) in curHiddenBuf.

    Computes:
      ttt_output = W_ttt @ hidden_t
      loss = MSE(ttt_output, hidden_{t+1} - hidden_t)
      if loss > tau: update W_ttt (SGD or Adam)

    Returns (mseLoss, gateOpen, updatedAdamStep). -/
def hiddenTTTStep (device : Device) (config : HiddenTTTConfig) (bufs : HiddenTTTBuffers)
    (curHiddenBuf : Buffer) (adamStep : Nat)
    : IO (Float × Bool × Nat) := do
  let d := config.dim
  let n := d * d

  -- TTT forward: ttt_output = W_ttt @ hidden_t (prevHiddenBuf)
  executeMatVec device bufs.tttWeightBuf bufs.prevHiddenBuf bufs.tttOutputBuf d d

  -- Fused MSE loss + gradient
  executeMSEResidualLossAndGrad device
    bufs.tttOutputBuf bufs.prevHiddenBuf curHiddenBuf
    bufs.gradBuf bufs.lossBuf d

  -- CPU readback for gate decision
  let mseLoss ← SafeBuffer.safeReadF32 device bufs.lossBuf

  let gateOpen := mseLoss > config.tau

  if gateOpen then
    -- Weight gradient: dW = outer(grad, hidden_t)
    executeOuterProduct device bufs.gradBuf bufs.prevHiddenBuf bufs.dWeightBuf d d

    if config.useAdam then
      -- Adam update with momentum (noise-resistant)
      let adamConfig : AdamGPU.Config := {
        lr := config.innerLR
        beta1 := config.adamBeta1
        beta2 := config.adamBeta2
        eps := config.adamEps
        weightDecay := 0.0  -- no weight decay for TTT
      }
      AdamGPU.executeAdamUpdate device bufs.tttWeightBuf bufs.dWeightBuf
        bufs.adamMBuf bufs.adamVBuf n adamConfig (adamStep + 1)
      return (mseLoss, true, adamStep + 1)
    else
      -- SGD fallback
      executeSGDUpdate device bufs.tttWeightBuf bufs.dWeightBuf n config.innerLR
      return (mseLoss, true, adamStep)

  return (mseLoss, false, adamStep)

/-- Compute LM head using model's embedding weights -/
def computeLMHead (device : Device) (model : BitNetModel)
    (hiddenBuf logitsBuf : Buffer) : IO Unit := do
  let cfg := model.config
  let lmConfig : Hesper.WGSL.MatMul.Config := { M := 1, N := cfg.vocabSize, K := cfg.dim }
  match model.embedding.f16Table with
  | some f16Buf =>
    if cfg.dim % 8 == 0 then
      Hesper.WGSL.MatMul.executeMatMulTransposeF16Shared device hiddenBuf f16Buf logitsBuf lmConfig
    else
      Hesper.WGSL.MatMul.executeMatMulTransposeF16 device hiddenBuf f16Buf logitsBuf lmConfig
  | none =>
    Hesper.WGSL.MatMul.executeMatMulTranspose device hiddenBuf model.embedding.embeddingTable logitsBuf lmConfig

/-- Add TTT correction to hidden, then compute LM head (decode phase, frozen). -/
def addTTTAndComputeLogits (device : Device) (config : HiddenTTTConfig) (bufs : HiddenTTTBuffers)
    (model : BitNetModel) (postNormHiddenBuf logitsBuf : Buffer) : IO Unit := do
  let d := config.dim
  -- tttOut = W_ttt @ hidden
  executeMatVec device bufs.tttWeightBuf postNormHiddenBuf bufs.tttOutputBuf d d
  -- corrected = hidden + tttOut
  executeVecAdd device postNormHiddenBuf bufs.tttOutputBuf bufs.correctedHiddenBuf d
  -- LM head on corrected hidden
  computeLMHead device model bufs.correctedHiddenBuf logitsBuf

/-- Generate text with Hidden-Space MSE TTT. -/
def generateWithHiddenTTT (device : Device) (model : BitNetModel)
    (promptTokens : Array Nat) (maxTokens : Nat)
    (tttConfig : HiddenTTTConfig)
    (strategy : Sampling.Strategy := .Greedy)
    (eosToken : Option Nat := none)
    (verbose : Bool := true)
    : IO (Array Nat) := do
  resetPreparedDispatches model

  IO.println "╔══════════════════════════════════════════════════════════╗"
  IO.println "║  BitNet + Hidden-Space MSE TTT                         ║"
  IO.println "╚══════════════════════════════════════════════════════════╝"
  IO.println s!"  Model: {model.config.dim}d, {model.config.numLayers}L, vocab={model.config.vocabSize}"
  IO.println s!"  TTT:   dim={tttConfig.dim}×{tttConfig.dim} ({tttConfig.dim * tttConfig.dim * 4 / 1024} KB), lr={tttConfig.innerLR}, tau={tttConfig.tau}"
  IO.println s!"  Loss:  MSE(W_ttt @ h_t, h_t1 - h_t)  [no LM head backward!]"
  IO.println s!"  Prompt: {promptTokens.size} tokens, generate up to {maxTokens}"
  IO.println ""

  let cacheState ← createKVCacheState device model
  let bufs ← createHiddenTTTBuffers device tttConfig
  let hBuf := postNormBuf cacheState model.config.numLayers
  let mut tokens := promptTokens
  let mut gateOpenCount : Nat := 0
  let mut hasPrevHidden := false
  let mut adamStep : Nat := 0

  -- ═══════════════════════════════════════════
  -- Prefill with MSE TTT learning
  -- ═══════════════════════════════════════════
  let optStr := if tttConfig.useAdam then "Adam" else "SGD"
  IO.println s!"[Prefill+TTT] Processing {promptTokens.size} prompt tokens (optimizer: {optStr})..."
  let prefillStart ← IO.monoNanosNow

  for i in [0:promptTokens.size] do
    if i >= model.config.maxSeqLen then break

    -- Forward pass: fills KV cache + produces hidden state in hBuf
    forwardSingleToken device model promptTokens[i]! i cacheState

    -- TTT update: compare prev_hidden vs current_hidden
    if hasPrevHidden then
      let (mseLoss, gateOpen, newAdamStep) ← hiddenTTTStep device tttConfig bufs hBuf adamStep
      adamStep := newAdamStep

      if gateOpen then gateOpenCount := gateOpenCount + 1
      if verbose then
        let gateStr := if gateOpen then "OPEN ⚡" else "closed"
        IO.println s!"  Token {i}: mse_loss={mseLoss}, gate={gateStr}"

    -- Save current hidden as prev for next iteration
    executeCopy device hBuf bufs.prevHiddenBuf tttConfig.dim
    hasPrevHidden := true

  let prefillEnd ← IO.monoNanosNow
  let prefillMs := (prefillEnd - prefillStart).toFloat / 1_000_000.0
  IO.println s!"[Prefill+TTT] Done in {prefillMs} ms"
  IO.println s!"  Gate opened: {gateOpenCount} / {promptTokens.size - 1} tokens"
  IO.println ""

  -- ═══════════════════════════════════════════
  -- Decode with frozen TTT weights
  -- ═══════════════════════════════════════════
  IO.println "[Decode] Generating with frozen hidden-space TTT..."

  -- Pre-allocate a logits buffer for TTT-corrected output
  let tttLogitsBuf ← createBuffer device {
    size := (tttConfig.vocabSize * 4).toUSize
    usage := [.storage, .copySrc, .copyDst]
    mappedAtCreation := false
  }

  let genStart ← IO.monoNanosNow
  let mut genTokenCount : Nat := 0

  for step in [0:maxTokens] do
    if tokens.size >= model.config.maxSeqLen then break

    -- Compute corrected logits: LM_head(hidden + W_ttt @ hidden)
    addTTTAndComputeLogits device tttConfig bufs model hBuf tttLogitsBuf

    let logits ← BufferOps.downloadFloatArray device tttLogitsBuf tttConfig.vocabSize
    let (nextToken, _) := Sampling.sampleWithRNG logits strategy
      (Sampling.RNG.create (some (42 + step)))

    if verbose then IO.println s!"  Decode step {step}: token={nextToken}"

    tokens := tokens.push nextToken
    genTokenCount := genTokenCount + 1

    match eosToken with
    | some eos => if nextToken == eos then
        IO.println "  EOS token, stopping"; break
    | none => pure ()

    let newPos := tokens.size - 1
    if newPos < model.config.maxSeqLen then
      forwardSingleToken device model nextToken newPos cacheState

  let genEnd ← IO.monoNanosNow
  let genMs := (genEnd - genStart).toFloat / 1_000_000.0
  IO.println s!"\nGenerated {genTokenCount} tokens in {genMs} ms"

  pure tokens

end Hesper.TTT.HiddenSpace
