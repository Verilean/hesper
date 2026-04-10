import Hesper.Models.Gemma4
import Hesper.TTT.Kernels
import Hesper.TTT.HiddenSpaceTTT
import Hesper.Training.SafeBuffer
import Hesper.WebGPU.BufferOps
import Hesper.Inference.Sampling

/-!
# Surprise-Gated Smart KV-Cache for Gemma 4

Port of SmartKVCache.lean (BitNet version) to the Gemma 4 pipeline.
Uses the MSE sensor to detect surprising tokens and route their KV
vectors to permanent "sink" slots, while boring tokens use a sliding
window ring buffer.

Does NOT modify Hesper.Models.Gemma4.
-/

namespace Hesper.TTT.SmartKVGemma4

open Hesper.WebGPU
open Hesper.Models.Gemma4
open Hesper.TTT.Kernels
open Hesper.TTT.HiddenSpace
open Hesper.Inference
open Hesper.Training

/-- Smart KV-Cache position tracker (same as BitNet version). -/
structure SmartKVState where
  maxSinks : Nat
  windowSize : Nat
  sinkCount : Nat := 0
  ringCount : Nat := 0
  deriving Inhabited, Repr

def getPhysicalPos (isSurprise : Bool) (state : SmartKVState)
    : Nat × Bool × SmartKVState :=
  if isSurprise && state.sinkCount < state.maxSinks then
    (state.sinkCount, true, { state with sinkCount := state.sinkCount + 1 })
  else
    let ringPos := state.maxSinks + (state.ringCount % state.windowSize)
    (ringPos, false, { state with ringCount := state.ringCount + 1 })

def cacheLen (state : SmartKVState) : Nat :=
  state.sinkCount + min state.ringCount state.windowSize

structure SmartKVConfig where
  maxSinks : Nat := 64
  windowSize : Nat := 256
  tau : Float := 0.003
  deriving Inhabited, Repr

/-- Post-norm hidden buffer: buf2 for even layers, buf1 for odd. -/
def postNormBuf (state : InferenceState) (numLayers : Nat) : Buffer :=
  if numLayers % 2 == 0 then state.buf2 else state.buf1

/-- Generate with Smart KV-Cache on Gemma 4. -/
def generateWithSmartKV (device : Device) (model : Gemma4Model)
    (promptTokens : Array Nat) (maxTokens : Nat)
    (config : SmartKVConfig)
    (strategy : Sampling.Strategy := .Greedy)
    (eosToken : Option Nat := none)
    (verbose : Bool := true)
    : IO (Array Nat) := do
  let totalSlots := config.maxSinks + config.windowSize
  let dim := model.config.hiddenSize

  IO.println "╔══════════════════════════════════════════════════════════╗"
  IO.println "║  Gemma 4 + Surprise-Gated Smart KV-Cache               ║"
  IO.println "╚══════════════════════════════════════════════════════════╝"
  IO.println s!"  Model: {dim}d, {model.config.numHiddenLayers}L, vocab={model.config.vocabSize}"
  IO.println s!"  Sinks: {config.maxSinks}, Window: {config.windowSize}, Total: {totalSlots}"
  IO.println s!"  Surprise tau: {config.tau}"
  IO.println s!"  Prompt: {promptTokens.size} tokens, generate up to {maxTokens}"
  IO.println ""

  -- Original model for KV cache (full maxSeqLen allocation)
  let state ← createInferenceState device model.config

  -- MSE sensor buffers (W_ttt stays zero — sensor only)
  let tttConfig : HiddenTTTConfig := {
    dim := dim
    vocabSize := model.config.vocabSize
    innerLR := 0.0
    tau := config.tau
    useAdam := false
  }
  let tttBufs ← createHiddenTTTBuffers device tttConfig
  let mut zeroBytes := ByteArray.empty
  for _ in [0:dim * 4] do
    zeroBytes := zeroBytes.push 0
  writeBuffer device tttBufs.tttOutputBuf 0 zeroBytes

  let hBuf := postNormBuf state model.config.numHiddenLayers

  let mut tokens := promptTokens
  let mut kvState : SmartKVState := {
    maxSinks := config.maxSinks, windowSize := config.windowSize
  }
  let mut hasPrevHidden := false
  let mut isSurprise := false
  let mut mseLoss : Float := 0.0
  let mut sinkTokens : Array (Nat × Nat) := #[]

  -- ═══════════════════════════════════════════
  -- Prefill with surprise-gated routing
  -- ═══════════════════════════════════════════
  IO.println s!"[Prefill] Processing {promptTokens.size} prompt tokens..."
  let prefillStart ← IO.monoNanosNow

  for i in [0:promptTokens.size] do
    let (physPos, wasSink, newState) := getPhysicalPos isSurprise kvState
    kvState := newState
    forwardSingleToken device model promptTokens[i]! physPos state

    if hasPrevHidden then
      executeMSEResidualLossAndGrad device
        tttBufs.tttOutputBuf tttBufs.prevHiddenBuf hBuf
        tttBufs.gradBuf tttBufs.lossBuf dim
      mseLoss ← SafeBuffer.safeReadF32 device tttBufs.lossBuf
      isSurprise := mseLoss > config.tau

    executeCopy device hBuf tttBufs.prevHiddenBuf dim
    hasPrevHidden := true

    if wasSink then sinkTokens := sinkTokens.push (i, physPos)
    if verbose then
      let typeStr := if wasSink then s!"SINK[{physPos}] ⚓" else s!"ring[{physPos}]"
      if i < 10 || wasSink || i >= promptTokens.size - 3 then
        IO.println s!"  Token {i} (id={promptTokens[i]!}): mse={mseLoss}, {typeStr}"

  let prefillEnd ← IO.monoNanosNow
  let prefillMs := (prefillEnd - prefillStart).toFloat / 1_000_000.0
  IO.println s!"[Prefill] Done in {prefillMs} ms"
  IO.println s!"  Sinks used: {kvState.sinkCount}/{config.maxSinks}"
  IO.println s!"  Ring count: {kvState.ringCount} (window={config.windowSize})"
  if sinkTokens.size > 0 && sinkTokens.size <= 40 then
    IO.println s!"  Sink tokens: {sinkTokens.map (·.1)}"
  IO.println ""

  -- ═══════════════════════════════════════════
  -- Decode
  -- ═══════════════════════════════════════════
  IO.println "[Decode] Generating..."
  let genStart ← IO.monoNanosNow
  let mut genTokenCount : Nat := 0

  for step in [0:maxTokens] do
    let logits ← BufferOps.downloadFloatArray device state.logitsBuf model.config.vocabSize
    let (nextToken, _) := Sampling.sampleWithRNG logits strategy
      (Sampling.RNG.create (some (42 + step)))

    if verbose then IO.println s!"  Decode step {step}: token={nextToken}"
    tokens := tokens.push nextToken
    genTokenCount := genTokenCount + 1

    match eosToken with
    | some eos => if nextToken == eos then IO.println "  EOS"; break
    | none => pure ()

    let (physPos, _, newState) := getPhysicalPos false kvState
    kvState := newState
    forwardSingleToken device model nextToken physPos state

  let genEnd ← IO.monoNanosNow
  let genMs := (genEnd - genStart).toFloat / 1_000_000.0
  IO.println s!"\nGenerated {genTokenCount} tokens in {genMs} ms"
  pure tokens

/-- Baseline: dumb sliding window for Gemma 4 -/
def generateWithDumbWindow (device : Device) (model : Gemma4Model)
    (promptTokens : Array Nat) (maxTokens : Nat)
    (windowSize : Nat)
    (strategy : Sampling.Strategy := .Greedy)
    : IO (Array Nat) := do
  let state ← createInferenceState device model.config

  for i in [0:promptTokens.size] do
    forwardSingleToken device model promptTokens[i]! (i % windowSize) state

  let mut tokens := promptTokens
  for step in [0:maxTokens] do
    let logits ← BufferOps.downloadFloatArray device state.logitsBuf model.config.vocabSize
    let (nextToken, _) := Sampling.sampleWithRNG logits strategy
      (Sampling.RNG.create (some (42 + step)))
    tokens := tokens.push nextToken
    forwardSingleToken device model nextToken (tokens.size % windowSize) state

  pure tokens

end Hesper.TTT.SmartKVGemma4
