import Hesper.Backend.WebGPU
import Hesper.Models.BitNet
import Hesper.TTT.Types
import Hesper.TTT.Kernels
import Hesper.TTT.InnerLoop
import Hesper.WebGPU.BufferOps
import Hesper.Inference.Sampling

/-!
# BitNet + TTT Integration

Provides `generateWithTTT` that wraps the existing BitNet inference
pipeline with Surprise-Gated Residual TTT, without modifying
`Hesper.Models.BitNet` at all.

Architecture (LM-Head Residual TTT):
- During **prefill**: for each prompt token, run BitNet forward to get
  base logits, then call `tttStepGPU` with the next token as target.
  The TTT module learns from surprising tokens (gate opens when
  base_loss > tau).
- During **decode**: TTT weights are frozen (no target available).
  Simply add `tttWeights @ hidden` to the base logits before sampling.

The post-final-norm hidden state is always in `cacheState.buf2` for
even-layer-count models (BitNet 2B = 30 layers, 1.3B = 24 layers).
For odd-layer-count models, it would be `buf1`. We compute this from
the layer count to be safe.
-/

namespace Hesper.TTT.BitNetTTT

open Hesper.WebGPU
open Hesper.Models.BitNet
open Hesper.TTT
open Hesper.TTT.Kernels
open Hesper.Inference

/-- Determine which buffer holds the post-final-norm hidden state
    after `forwardSingleToken` returns. With N layers and a
    buf1/buf2 ping-pong starting at buf1, after N swaps the norm
    writes currentBuf → nextBuf:
      even N: currentBuf=buf1, nextBuf=buf2 → hidden in buf2
      odd N:  currentBuf=buf2, nextBuf=buf1 → hidden in buf1 -/
def hiddenBuf (cacheState : KVCacheState Buffer PreparedDispatch) (numLayers : Nat) : Buffer :=
  if numLayers % 2 == 0 then cacheState.buf2 else cacheState.buf1

/-- Upload a hidden state from a GPU buffer into the TTT hidden buffer.
    Copies `dim` f32 values from `srcBuf` to `tttBufs.hiddenBuf`. -/
def copyHiddenToTTT (device : Device) (srcBuf : Buffer) (tttBufs : TTTBuffers)
    (dim : Nat) : IO Unit :=
  Kernels.executeCopy device srcBuf tttBufs.hiddenBuf dim

/-- Run one TTT step using pre-computed base logits from BitNet's LM head.
    Skips the base matVec (BitNet already computed base logits via its
    own LM head) and jumps straight to CE loss + gate + TTT update.

    Returns (baseLoss, gateOpen). -/
def tttStepWithBaseLogits (device : Device) (config : TTTConfig) (tttBufs : TTTBuffers)
    (baseLogitsBuf : Buffer) : IO (Float × Bool) := do
  let v := config.vocabSize
  let h := config.hiddenDim
  let n := v * h

  -- Copy BitNet's base logits into TTT's baseLogitsBuf
  Kernels.executeCopy device baseLogitsBuf tttBufs.baseLogitsBuf v

  -- CE loss on base logits
  Hesper.Training.Loss.executeCrossEntropyForward device tttBufs.baseLogitsBuf tttBufs.targetBuf tttBufs.lossBuf v

  -- CPU readback for gate decision
  let baseLoss ← Hesper.Training.SafeBuffer.safeReadF32 device tttBufs.lossBuf

  let gateOpen := baseLoss > config.tau

  if gateOpen then
    -- TTT forward with current weights
    executeMatVec device tttBufs.tttWeightBuf tttBufs.hiddenBuf tttBufs.tttLogitsBuf v h

    -- Combined = base + ttt
    executeVecAdd device tttBufs.baseLogitsBuf tttBufs.tttLogitsBuf tttBufs.combinedLogitsBuf v

    -- CE backward
    Hesper.Training.Loss.executeCrossEntropyBackward device tttBufs.combinedLogitsBuf tttBufs.targetBuf tttBufs.dCombinedBuf v

    -- Weight gradient = outer(dCombined, hidden)
    executeOuterProduct device tttBufs.dCombinedBuf tttBufs.hiddenBuf tttBufs.dWeightBuf v h

    -- SGD update
    executeSGDUpdate device tttBufs.tttWeightBuf tttBufs.dWeightBuf n config.innerLR

    -- Recompute TTT logits with updated weights
    executeMatVec device tttBufs.tttWeightBuf tttBufs.hiddenBuf tttBufs.tttLogitsBuf v h

    -- Final = base + updated ttt
    executeVecAdd device tttBufs.baseLogitsBuf tttBufs.tttLogitsBuf tttBufs.finalLogitsBuf v
  else
    -- Gate closed: final = base + current ttt
    executeMatVec device tttBufs.tttWeightBuf tttBufs.hiddenBuf tttBufs.tttLogitsBuf v h
    executeVecAdd device tttBufs.baseLogitsBuf tttBufs.tttLogitsBuf tttBufs.finalLogitsBuf v

  return (baseLoss, gateOpen)

/-- Add TTT logits to base logits (frozen TTT, decode phase).
    Computes: `outputBuf[i] = baseLogitsBuf[i] + tttWeight @ hidden[i]`
    and writes to `outputBuf`. -/
def addTTTLogits (device : Device) (config : TTTConfig) (tttBufs : TTTBuffers)
    (baseLogitsBuf outputBuf : Buffer) : IO Unit := do
  let v := config.vocabSize
  let h := config.hiddenDim
  executeMatVec device tttBufs.tttWeightBuf tttBufs.hiddenBuf tttBufs.tttLogitsBuf v h
  executeVecAdd device baseLogitsBuf tttBufs.tttLogitsBuf outputBuf v

/-- Generate text with TTT (Surprise-Gated Residual Test-Time Training).

    During prefill, the TTT module learns from the prompt by observing
    which tokens surprise the base model (CE loss > tau). During decode,
    the learned TTT weights are frozen and simply added to base logits.

    Does NOT modify the BitNet model or its weights in any way. -/
def generateWithTTT (device : Device) (model : BitNetModel Buffer PreparedDispatch CompiledKernel)
    (promptTokens : Array Nat) (maxTokens : Nat)
    (tttConfig : TTTConfig)
    (strategy : Sampling.Strategy := .Greedy)
    (eosToken : Option Nat := none)
    (verbose : Bool := true)
    : IO (Array Nat) := do
  resetPreparedDispatches model

  IO.println "╔══════════════════════════════════════════════════════════╗"
  IO.println "║  BitNet + TTT Generation (Surprise-Gated Residual)     ║"
  IO.println "╚══════════════════════════════════════════════════════════╝"
  IO.println s!"  Model: {model.config.dim}d, {model.config.numLayers}L, vocab={model.config.vocabSize}"
  IO.println s!"  TTT:   lr={tttConfig.innerLR}, tau={tttConfig.tau}"
  IO.println s!"  Prompt: {promptTokens.size} tokens, generate up to {maxTokens}"
  IO.println ""

  -- Create standard BitNet KV cache state
  let cacheState ← createKVCacheState (β := Device) device model

  -- Create TTT buffers (tttWeightBuf zero-initialized)
  -- TTT operates at vocabSize × dim (same as LM head)
  let tttBufs ← createTTTBuffers device tttConfig

  let hBuf := hiddenBuf cacheState model.config.numLayers
  let mut tokens := promptTokens
  let mut gateOpenCount : Nat := 0

  -- ═══════════════════════════════════════════
  -- Phase 1: PREFILL with TTT learning
  -- ═══════════════════════════════════════════
  IO.println s!"[Prefill+TTT] Processing {promptTokens.size} prompt tokens..."
  let prefillStart ← IO.monoNanosNow

  for i in [0:promptTokens.size] do
    if i >= model.config.maxSeqLen then break

    -- Run standard BitNet forward pass (fills KV cache + produces logits)
    forwardSingleToken (β := Device) device model promptTokens[i]! i cacheState

    -- TTT learning: use next token as target (teacher forcing)
    if i + 1 < promptTokens.size then
      -- Copy post-norm hidden state to TTT buffer
      copyHiddenToTTT device hBuf tttBufs model.config.dim

      -- Upload target = next token
      let target := promptTokens[i + 1]!
      let targetBytes := BufferOps.uint32ToBytes target.toUInt32
      writeBuffer device tttBufs.targetBuf 0 targetBytes

      -- Run TTT step (gate decision + conditional update)
      let (baseLoss, gateOpen) ← tttStepWithBaseLogits device tttConfig tttBufs cacheState.logitsBuf

      if gateOpen then gateOpenCount := gateOpenCount + 1

      if verbose then
        let gateStr := if gateOpen then "OPEN ⚡" else "closed"
        IO.println s!"  Token {i}: loss={baseLoss}, gate={gateStr}"

  let prefillEnd ← IO.monoNanosNow
  let prefillMs := (prefillEnd - prefillStart).toFloat / 1_000_000.0
  IO.println s!"[Prefill+TTT] Done in {prefillMs} ms"
  IO.println s!"  Gate opened: {gateOpenCount} / {promptTokens.size - 1} tokens"
  IO.println ""

  -- ═══════════════════════════════════════════
  -- Phase 2: DECODE with frozen TTT weights
  -- ═══════════════════════════════════════════
  IO.println "[Decode] Generating with frozen TTT weights..."
  let genStart ← IO.monoNanosNow
  let mut genTokenCount : Nat := 0

  for step in [0:maxTokens] do
    if tokens.size >= model.config.maxSeqLen then
      IO.println s!"  Reached max sequence length ({model.config.maxSeqLen})"
      break

    -- Add TTT logits to base logits: finalLogits = baseLogits + tttW @ hidden
    copyHiddenToTTT device hBuf tttBufs model.config.dim
    addTTTLogits device tttConfig tttBufs cacheState.logitsBuf tttBufs.finalLogitsBuf

    -- Sample from finalLogitsBuf (with TTT contribution)
    let finalLogits ← BufferOps.downloadFloatArray device tttBufs.finalLogitsBuf tttConfig.vocabSize
    let (nextToken, _) := Sampling.sampleWithRNG finalLogits strategy
      (Sampling.RNG.create (some (42 + step)))

    if verbose then
      IO.println s!"  Decode step {step}: token={nextToken}"

    tokens := tokens.push nextToken
    genTokenCount := genTokenCount + 1

    -- Early stopping
    match eosToken with
    | some eos => if nextToken == eos then
        IO.println "  EOS token, stopping"
        break
    | none => pure ()

    -- Forward pass for the new token (extends KV cache)
    let newPos := tokens.size - 1
    if newPos < model.config.maxSeqLen then
      forwardSingleToken (β := Device) device model nextToken newPos cacheState

  let genEnd ← IO.monoNanosNow
  let genMs := (genEnd - genStart).toFloat / 1_000_000.0
  let msPerToken := if genTokenCount > 0 then genMs / genTokenCount.toFloat else 0.0
  IO.println ""
  IO.println s!"Generated {genTokenCount} tokens in {genMs} ms ({msPerToken} ms/tok)"

  pure tokens

end Hesper.TTT.BitNetTTT
