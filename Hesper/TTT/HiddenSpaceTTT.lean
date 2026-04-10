import Hesper.Models.BitNet
import Hesper.TTT.Types
import Hesper.TTT.Kernels
import Hesper.Training.Loss
import Hesper.Training.SafeBuffer
import Hesper.WebGPU.BufferOps
import Hesper.Inference.Sampling
import Hesper.WGSL.MatMul

/-!
# Hidden-Space TTT for BitNet

Instead of adding a residual to logits (vocabSize × dim = 1.3 GB),
this variant adds a residual to the **hidden state** before the LM head:

```
hidden' = hidden + W_ttt @ hidden     -- dim × dim = 26 MB
logits  = LM_head(hidden')            -- uses original LM head weights
```

This is 50,000× more memory-efficient than logit-space TTT for 128K
vocab, and the correction propagates to all tokens via the LM head.

The gradient flows back through the LM head:
```
dLogits  = softmax(logits) - one_hot(target)
dHidden' = LM_head^T @ dLogits
dW_ttt   = outer(dHidden', hidden)
W_ttt   -= lr * dW_ttt
```

Does NOT modify Hesper.Models.BitNet.
-/

namespace Hesper.TTT.HiddenSpace

open Hesper.WebGPU
open Hesper.Models.BitNet
open Hesper.TTT.Kernels
open Hesper.Inference
open Hesper.Training

/-- Hidden-space TTT config. Uses model's dim, not vocabSize. -/
structure HiddenTTTConfig where
  dim : Nat
  vocabSize : Nat     -- needed for CE loss/backward
  innerLR : Float
  tau : Float          -- static fallback threshold (used if dynamicGate=false)
  -- Dynamic EMA gate: gate opens when loss > emaLoss * gateMultiplier.
  -- Intended to filter haystack noise, but requires tuning per model.
  -- Default off: static tau works better on BitNet 2B where random
  -- token loss is already high (~10), making EMA-relative thresholding
  -- ineffective at distinguishing needles from noise.
  dynamicGate : Bool := false
  gateMultiplier : Float := 2.0   -- open when loss > k × EMA
  emaAlpha : Float := 0.1         -- smoothing factor for running average
  warmupSteps : Nat := 5          -- always OPEN for first N steps (EMA not stable yet)
  deriving Inhabited, Repr

/-- Mutable gate state for dynamic EMA thresholding. -/
structure GateState where
  emaLoss : Float := 0.0
  stepCount : Nat := 0
  deriving Inhabited, Repr

/-- GPU buffers for hidden-space TTT. W_ttt is [dim × dim]. -/
structure HiddenTTTBuffers where
  tttWeightBuf : Buffer     -- [dim × dim] f32, mutable, zero-init
  hiddenBuf : Buffer        -- [dim] f32, copy of post-norm hidden
  correctedHiddenBuf : Buffer -- [dim] f32, hidden + W_ttt @ hidden
  tttHiddenOutBuf : Buffer  -- [dim] f32, W_ttt @ hidden
  logitsBuf : Buffer        -- [vocabSize] f32, LM_head(corrected hidden)
  targetBuf : Buffer        -- [1] u32
  lossBuf : Buffer          -- [1] f32
  dLogitsBuf : Buffer       -- [vocabSize] f32, CE backward
  dHiddenBuf : Buffer       -- [dim] f32, LM_head^T @ dLogits
  dWeightBuf : Buffer       -- [dim × dim] f32, outer product

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
      -- Zero-init: initial residual = 0
      let mut zeros := ByteArray.empty
      for _ in [0:dimSq * 4] do
        zeros := zeros.push 0
      writeBuffer device buf 0 zeros
      pure buf
    hiddenBuf := ← mkF32Buf config.dim
    correctedHiddenBuf := ← mkF32Buf config.dim
    tttHiddenOutBuf := ← mkF32Buf config.dim
    logitsBuf := ← mkF32Buf config.vocabSize
    targetBuf := ← createBuffer device {
      size := 4, usage := [.storage, .copySrc, .copyDst], mappedAtCreation := false
    }
    lossBuf := ← mkF32Buf 1
    dLogitsBuf := ← mkF32Buf config.vocabSize
    dHiddenBuf := ← mkF32Buf config.dim
    dWeightBuf := ← mkF32Buf dimSq
  }

/-- Which buffer is the post-norm hidden after forwardSingleToken -/
def postNormBuf (cacheState : KVCacheState) (numLayers : Nat) : Buffer :=
  if numLayers % 2 == 0 then cacheState.buf2 else cacheState.buf1

/-- Compute LM head: logits = embedding @ hidden (transposed matmul).
    Reuses the model's own embedding weights. -/
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

/-- Compute LM head backward: dHidden = embedding^T @ dLogits.

    Math: `dHidden[d] = Σ_v embedding[v,d] * dLogits[v]`

    Since embedding is stored as [vocabSize, dim] (row-major), this is
    a **non-transposed** matmul: `C = A @ B` where
      A = dLogits [1, vocabSize]
      B = embedding [vocabSize, dim]
      C = dHidden [1, dim]

    Always uses the F32 embeddingTable (guaranteed to exist) via
    `executeMatMul` (not transpose). Previous version incorrectly
    used `executeMatMulTransposeF16` which swapped the matrix layout. -/
def computeLMHeadBackward (device : Device) (model : BitNetModel)
    (dLogitsBuf dHiddenBuf : Buffer) : IO Unit := do
  let cfg := model.config
  -- C[1, dim] = dLogits[1, vocabSize] @ embedding[vocabSize, dim]
  let matConfig : Hesper.WGSL.MatMul.Config := { M := 1, N := cfg.dim, K := cfg.vocabSize }
  Hesper.WGSL.MatMul.executeMatMul device dLogitsBuf model.embedding.embeddingTable dHiddenBuf matConfig

/-- One hidden-space TTT step with dynamic EMA gate.
    Returns (baseLoss, gateOpen, updatedGateState). -/
def hiddenTTTStep (device : Device) (config : HiddenTTTConfig) (bufs : HiddenTTTBuffers)
    (model : BitNetModel) (baseLogitsBuf postNormHiddenBuf : Buffer)
    (gateState : GateState)
    : IO (Float × Bool × GateState) := do
  let d := config.dim
  let v := config.vocabSize

  -- Step 1: Compute base loss from the ORIGINAL base logits
  Loss.executeCrossEntropyForward device baseLogitsBuf bufs.targetBuf bufs.lossBuf v
  let baseLoss ← SafeBuffer.safeReadF32 device bufs.lossBuf

  -- Step 2: Dynamic gate decision
  let gateOpen :=
    if config.dynamicGate then
      if gateState.stepCount < config.warmupSteps then
        true  -- warmup: always open (EMA not stable yet)
      else
        baseLoss > gateState.emaLoss * config.gateMultiplier
    else
      baseLoss > config.tau

  -- Update EMA (always, regardless of gate decision)
  let newEma :=
    if gateState.stepCount == 0 then baseLoss  -- init to first loss
    else config.emaAlpha * baseLoss + (1.0 - config.emaAlpha) * gateState.emaLoss
  let newGateState : GateState := {
    emaLoss := newEma
    stepCount := gateState.stepCount + 1
  }

  if gateOpen then
    -- Copy hidden state
    executeCopy device postNormHiddenBuf bufs.hiddenBuf d

    -- TTT forward: tttOut = W_ttt @ hidden
    executeMatVec device bufs.tttWeightBuf bufs.hiddenBuf bufs.tttHiddenOutBuf d d

    -- correctedHidden = hidden + tttOut
    executeVecAdd device bufs.hiddenBuf bufs.tttHiddenOutBuf bufs.correctedHiddenBuf d

    -- LM head on corrected hidden → logits
    computeLMHead device model bufs.correctedHiddenBuf bufs.logitsBuf

    -- CE backward: dLogits = softmax(logits) - one_hot(target)
    Loss.executeCrossEntropyBackward device bufs.logitsBuf bufs.targetBuf bufs.dLogitsBuf v

    -- Backprop through LM head: dHidden = emb^T @ dLogits (or emb @ dLogits for the right shape)
    computeLMHeadBackward device model bufs.dLogitsBuf bufs.dHiddenBuf

    -- Weight gradient: dW = outer(dHidden, hidden)
    executeOuterProduct device bufs.dHiddenBuf bufs.hiddenBuf bufs.dWeightBuf d d

    -- SGD update: W_ttt -= lr * dW
    executeSGDUpdate device bufs.tttWeightBuf bufs.dWeightBuf (d * d) config.innerLR

  return (baseLoss, gateOpen, newGateState)

/-- Add TTT correction to hidden, then compute LM head (decode phase, frozen). -/
def addTTTAndComputeLogits (device : Device) (config : HiddenTTTConfig) (bufs : HiddenTTTBuffers)
    (model : BitNetModel) (postNormHiddenBuf outputLogitsBuf : Buffer) : IO Unit := do
  let d := config.dim
  -- Copy hidden
  executeCopy device postNormHiddenBuf bufs.hiddenBuf d
  -- TTT forward: tttOut = W_ttt @ hidden
  executeMatVec device bufs.tttWeightBuf bufs.hiddenBuf bufs.tttHiddenOutBuf d d
  -- correctedHidden = hidden + tttOut
  executeVecAdd device bufs.hiddenBuf bufs.tttHiddenOutBuf bufs.correctedHiddenBuf d
  -- LM head on corrected hidden
  computeLMHead device model bufs.correctedHiddenBuf outputLogitsBuf

/-- Generate text with Hidden-Space TTT. -/
def generateWithHiddenTTT (device : Device) (model : BitNetModel)
    (promptTokens : Array Nat) (maxTokens : Nat)
    (tttConfig : HiddenTTTConfig)
    (strategy : Sampling.Strategy := .Greedy)
    (eosToken : Option Nat := none)
    (verbose : Bool := true)
    : IO (Array Nat) := do
  resetPreparedDispatches model

  IO.println "╔══════════════════════════════════════════════════════════╗"
  IO.println "║  BitNet + Hidden-Space TTT                             ║"
  IO.println "╚══════════════════════════════════════════════════════════╝"
  IO.println s!"  Model: {model.config.dim}d, {model.config.numLayers}L, vocab={model.config.vocabSize}"
  IO.println s!"  TTT:   dim={tttConfig.dim}×{tttConfig.dim} ({tttConfig.dim * tttConfig.dim * 4 / 1024} KB), lr={tttConfig.innerLR}, tau={tttConfig.tau}"
  IO.println s!"  Prompt: {promptTokens.size} tokens, generate up to {maxTokens}"
  IO.println ""

  let cacheState ← createKVCacheState device model
  let bufs ← createHiddenTTTBuffers device tttConfig
  let hBuf := postNormBuf cacheState model.config.numLayers
  let mut tokens := promptTokens
  let mut gateOpenCount : Nat := 0
  let mut gateState : GateState := {}

  -- Prefill with TTT learning
  let gateMode := if tttConfig.dynamicGate then s!"dynamic (EMA×{tttConfig.gateMultiplier})" else s!"static (tau={tttConfig.tau})"
  IO.println s!"[Prefill+TTT] Processing {promptTokens.size} prompt tokens (gate: {gateMode})..."
  let prefillStart ← IO.monoNanosNow

  for i in [0:promptTokens.size] do
    if i >= model.config.maxSeqLen then break
    forwardSingleToken device model promptTokens[i]! i cacheState

    if i + 1 < promptTokens.size then
      let target := promptTokens[i + 1]!
      let targetBytes := BufferOps.uint32ToBytes target.toUInt32
      writeBuffer device bufs.targetBuf 0 targetBytes

      let (baseLoss, gateOpen, newGS) ← hiddenTTTStep device tttConfig bufs model cacheState.logitsBuf hBuf gateState
      gateState := newGS

      if gateOpen then gateOpenCount := gateOpenCount + 1
      if verbose then
        let gateStr := if gateOpen then "OPEN ⚡" else "closed"
        let emaStr := if tttConfig.dynamicGate then s!" ema={gateState.emaLoss}" else ""
        IO.println s!"  Token {i}: loss={baseLoss}, gate={gateStr}{emaStr}"

  let prefillEnd ← IO.monoNanosNow
  let prefillMs := (prefillEnd - prefillStart).toFloat / 1_000_000.0
  IO.println s!"[Prefill+TTT] Done in {prefillMs} ms"
  IO.println s!"  Gate opened: {gateOpenCount} / {promptTokens.size - 1} tokens"
  IO.println ""

  -- Decode with frozen TTT weights
  IO.println "[Decode] Generating with frozen hidden-space TTT..."
  let genStart ← IO.monoNanosNow
  let mut genTokenCount : Nat := 0

  for step in [0:maxTokens] do
    if tokens.size >= model.config.maxSeqLen then break

    -- Compute corrected logits: LM_head(hidden + W_ttt @ hidden)
    addTTTAndComputeLogits device tttConfig bufs model hBuf bufs.logitsBuf

    let logits ← BufferOps.downloadFloatArray device bufs.logitsBuf tttConfig.vocabSize
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
