import Hesper.Models.BitNet
import Hesper.TTT.Kernels
import Hesper.TTT.HiddenSpaceTTT
import Hesper.Training.SafeBuffer
import Hesper.WebGPU.BufferOps
import Hesper.Inference.Sampling

/-!
# Surprise-Gated Smart KV-Cache

Uses the TTT MSE loss spike as a **sensor** to decide which tokens
get permanent KV cache slots (Attention Sinks) vs sliding window.

Architecture:
  - Sink slots [0, maxSinks): permanent, for "surprising" tokens
  - Ring buffer [maxSinks, maxSinks + windowSize): sliding window

Physical KV cache layout:
  [ SINK_0 | SINK_1 | ... | SINK_31 | RING_0 | RING_1 | ... | RING_255 ]
    permanent (surprise tokens)        sliding window (recent tokens)

The existing Flash Attention kernel reads positions 0..cacheLen-1
contiguously, so this layout works without any WGSL kernel changes.

Note: This PoC uses physicalPos for both RoPE and cache write.
RoPE angles won't reflect true absolute position for recycled ring
slots, but the content is correctly preserved. A production version
would decouple RoPE pos from cache write pos.
-/

namespace Hesper.TTT.SmartKV

open Hesper.WebGPU
open Hesper.Models.BitNet
open Hesper.TTT.Kernels
open Hesper.TTT.HiddenSpace
open Hesper.Inference
open Hesper.Training

/-- Smart KV-Cache position tracker. -/
structure SmartKVState where
  maxSinks : Nat       -- permanent slots for surprise tokens
  windowSize : Nat     -- sliding window for recent tokens
  sinkCount : Nat := 0
  ringCount : Nat := 0
  deriving Inhabited, Repr

/-- Get the physical KV cache position for a token.
    Returns (physicalPos, isSink, updatedState). -/
def getPhysicalPos (isSurprise : Bool) (state : SmartKVState)
    : Nat × Bool × SmartKVState :=
  if isSurprise && state.sinkCount < state.maxSinks then
    -- Route to permanent sink slot
    (state.sinkCount, true, { state with sinkCount := state.sinkCount + 1 })
  else
    -- Route to ring buffer (overwrites oldest)
    let ringPos := state.maxSinks + (state.ringCount % state.windowSize)
    (ringPos, false, { state with ringCount := state.ringCount + 1 })

/-- How many physical positions are currently populated -/
def cacheLen (state : SmartKVState) : Nat :=
  state.sinkCount + min state.ringCount state.windowSize

/-- Config for Smart KV experiment -/
structure SmartKVConfig where
  maxSinks : Nat := 32
  windowSize : Nat := 256
  tau : Float := 0.005      -- MSE threshold for surprise detection
  deriving Inhabited, Repr

/-- Generate with Surprise-Gated Smart KV-Cache.

    During prefill, each token is classified as "surprise" or "boring"
    based on MSE between consecutive hidden states. Surprise tokens get
    permanent KV cache slots; boring tokens use a sliding window.
    This allows perfect recall of important facts even after thousands
    of intervening tokens. -/
def generateWithSmartKV (device : Device) (model : BitNetModel)
    (promptTokens : Array Nat) (maxTokens : Nat)
    (config : SmartKVConfig)
    (strategy : Sampling.Strategy := .Greedy)
    (eosToken : Option Nat := none)
    (verbose : Bool := true)
    : IO (Array Nat) := do
  resetPreparedDispatches model

  let totalSlots := config.maxSinks + config.windowSize

  IO.println "╔══════════════════════════════════════════════════════════╗"
  IO.println "║  BitNet + Surprise-Gated Smart KV-Cache                ║"
  IO.println "╚══════════════════════════════════════════════════════════╝"
  IO.println s!"  Model: {model.config.dim}d, {model.config.numLayers}L"
  IO.println s!"  Sinks: {config.maxSinks}, Window: {config.windowSize}, Total: {totalSlots}"
  IO.println s!"  Surprise tau: {config.tau} (MSE threshold)"
  IO.println s!"  Prompt: {promptTokens.size} tokens, generate up to {maxTokens}"
  IO.println ""

  -- Use the ORIGINAL model for KV cache — full maxSeqLen allocation.
  -- The 288-slot limit is enforced purely by Lean-side routing logic.
  -- This avoids kernel/buffer-size mismatches.
  let cacheState ← createKVCacheState device model

  -- TTT buffers for MSE sensor (W_ttt stays zero — we only read loss)
  let tttConfig : HiddenTTTConfig := {
    dim := model.config.dim
    vocabSize := model.config.vocabSize
    innerLR := 0.0  -- no weight updates! sensor only
    tau := config.tau
    useAdam := false
  }
  let tttBufs ← createHiddenTTTBuffers device tttConfig
  -- Zero the tttOutputBuf explicitly (used as zero-vector in MSE sensor)
  let mut zeroBytes := ByteArray.empty
  for _ in [0:model.config.dim * 4] do
    zeroBytes := zeroBytes.push 0
  writeBuffer device tttBufs.tttOutputBuf 0 zeroBytes
  let hBuf := postNormBuf cacheState model.config.numLayers

  let mut tokens := promptTokens
  let mut kvState : SmartKVState := {
    maxSinks := config.maxSinks
    windowSize := config.windowSize
  }
  let mut hasPrevHidden := false
  let mut sinkTokens : Array (Nat × Nat) := #[]  -- (tokenIdx, physicalPos) for logging
  let mut isSurprise := false  -- carries over from previous iteration
  let mut mseLoss : Float := 0.0

  -- ═══════════════════════════════════════════
  -- Prefill: classify tokens and route to KV cache
  -- ═══════════════════════════════════════════
  IO.println s!"[Prefill] Processing {promptTokens.size} prompt tokens..."
  let prefillStart ← IO.monoNanosNow

  for i in [0:promptTokens.size] do

    -- Step 1: Forward pass at position within the smart window.
    -- Sinks occupy [0, maxSinks), ring occupies [maxSinks, totalSlots).
    -- Compute the physical position FIRST from current routing state,
    -- then run forward at that position.
    let (physPos, wasSink, newState) := getPhysicalPos isSurprise kvState
    kvState := newState
    forwardSingleToken device model promptTokens[i]! physPos cacheState

    -- Step 2: MSE sensor between consecutive hidden states
    if hasPrevHidden then
      executeMSEResidualLossAndGrad device
        tttBufs.tttOutputBuf  -- zeros (W=0)
        tttBufs.prevHiddenBuf -- hidden_t (from previous step)
        hBuf                  -- hidden_{t+1} (current step)
        tttBufs.gradBuf tttBufs.lossBuf model.config.dim
      mseLoss ← SafeBuffer.safeReadF32 device tttBufs.lossBuf
      -- This surprise flag will be used for the NEXT token's routing
      isSurprise := mseLoss > config.tau

    -- Save current hidden for next MSE comparison
    executeCopy device hBuf tttBufs.prevHiddenBuf model.config.dim
    hasPrevHidden := true

    if wasSink then
      sinkTokens := sinkTokens.push (i, physPos)

    if verbose then
      let typeStr := if wasSink then s!"SINK[{physPos}] ⚓"
        else s!"ring[{physPos}]"
      if i < 10 || wasSink || i >= promptTokens.size - 3 then
        IO.println s!"  Token {i} (id={promptTokens[i]!}): mse={mseLoss}, {typeStr}"

  let prefillEnd ← IO.monoNanosNow
  let prefillMs := (prefillEnd - prefillStart).toFloat / 1_000_000.0
  IO.println s!"[Prefill] Done in {prefillMs} ms"
  IO.println s!"  Sinks used: {kvState.sinkCount}/{config.maxSinks}"
  IO.println s!"  Ring count: {kvState.ringCount} (window={config.windowSize})"
  IO.println s!"  Cache len: {SmartKV.cacheLen kvState}"
  if sinkTokens.size > 0 then
    IO.println s!"  Sink tokens: {sinkTokens.map (·.1)}"
  IO.println ""

  -- ═══════════════════════════════════════════
  -- Decode: generate with the smart KV cache
  -- ═══════════════════════════════════════════
  IO.println "[Decode] Generating..."
  let genStart ← IO.monoNanosNow
  let mut genTokenCount : Nat := 0

  for step in [0:maxTokens] do
    -- Sample from logits (already computed by last forwardSingleToken)
    let logits ← BufferOps.downloadFloatArray device cacheState.logitsBuf model.config.vocabSize
    let (nextToken, _) := Sampling.sampleWithRNG logits strategy
      (Sampling.RNG.create (some (42 + step)))

    if verbose then IO.println s!"  Decode step {step}: token={nextToken}"

    tokens := tokens.push nextToken
    genTokenCount := genTokenCount + 1

    match eosToken with
    | some eos => if nextToken == eos then
        IO.println "  EOS token, stopping"; break
    | none => pure ()

    -- Forward pass for new token in the ring buffer
    let (physPos, _, newState) := getPhysicalPos false kvState
    kvState := newState
    forwardSingleToken device model nextToken physPos cacheState

  let genEnd ← IO.monoNanosNow
  let genMs := (genEnd - genStart).toFloat / 1_000_000.0
  IO.println s!"\nGenerated {genTokenCount} tokens in {genMs} ms"

  pure tokens

/-- Baseline: dumb sliding window (no smart routing).
    Uses original model's full KV cache but writes at pos % windowSize. -/
def generateWithDumbWindow (device : Device) (model : BitNetModel)
    (promptTokens : Array Nat) (maxTokens : Nat)
    (windowSize : Nat)
    (strategy : Sampling.Strategy := .Greedy)
    : IO (Array Nat) := do
  resetPreparedDispatches model
  let cacheState ← createKVCacheState device model

  -- Simple sliding window: pos = t % windowSize
  for i in [0:promptTokens.size] do
    let physPos := i % windowSize
    forwardSingleToken device model promptTokens[i]! physPos cacheState

  let mut tokens := promptTokens
  for step in [0:maxTokens] do
    let logits ← BufferOps.downloadFloatArray device cacheState.logitsBuf model.config.vocabSize
    let (nextToken, _) := Sampling.sampleWithRNG logits strategy
      (Sampling.RNG.create (some (42 + step)))
    tokens := tokens.push nextToken
    let physPos := tokens.size % windowSize
    forwardSingleToken device model nextToken physPos cacheState

  pure tokens

end Hesper.TTT.SmartKV
