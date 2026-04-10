import Hesper.Models.Gemma4
import Hesper.TTT.Kernels
import Hesper.Training.Loss
import Hesper.Training.SafeBuffer
import Hesper.WebGPU.BufferOps
import Hesper.Inference.Sampling

/-!
# Surprise-Gated Smart KV-Cache for Gemma 4

Uses cosine distance between consecutive hidden states as a
scale-invariant surprise sensor. Important tokens are protected
from eviction; boring tokens outside the window are zeroed out.

Strategy (NO modification to Gemma4.lean):
1. Call forwardSingleToken(pos=i) normally — KV written at row i,
   cacheLen = i+1 (Gemma's internal logic, untouched)
2. After forward, measure cosine surprise between h_t and h_{t+1}
3. If surprise > tau: mark position i as a "sink" (protected)
4. When i > windowSize: zero out K/V at position (i - windowSize)
   UNLESS that position is a protected sink
5. Zeroed K → dot(Q,K)=0 → minimal attention weight → effectively evicted

This approach works because:
- Gemma 4 always loops 0..cacheLen in attention
- Zeroed K rows produce near-zero attention scores
- Protected sink rows retain their full K/V content permanently
-/

namespace Hesper.TTT.SmartKVGemma4

open Hesper.WebGPU
open Hesper.Models.Gemma4
open Hesper.TTT.Kernels
open Hesper.Inference
open Hesper.Training

structure SmartKVConfig where
  windowSize : Nat := 256
  tau : Float := 5.0       -- CE loss threshold (high = very surprising)
  deriving Inhabited, Repr

/-- Post-norm hidden buffer for Gemma 4.
    Gemma 4's forwardSingleToken starts with currentBuf=buf2, nextBuf=buf1
    (opposite of BitNet). After N layers of ping-pong + final norm
    (currentBuf → nextBuf), the post-norm hidden ends up in:
      even layers: buf1  (start=buf2, 42 swaps → buf2 again, norm writes → buf1)
      odd layers:  buf2 -/
def postNormBuf (state : InferenceState) (numLayers : Nat) : Buffer :=
  if numLayers % 2 == 0 then state.buf1 else state.buf2

/-- Zero out KV cache at position `pos` across ALL layers.
    Each layer's KV cache is in its own Gemma4KVCache struct. -/
def evictPosition (device : Device) (state : InferenceState)
    (cfg : Config) (pos : Nat) : IO Unit := do
  for kvCache in state.kvCaches do
    -- Gemma 4 has varying numKVHeads per layer (full vs SWA).
    -- Use max heads to be safe; extra zeros beyond actual heads are harmless.
    let maxKVHeads := max cfg.numKeyValueHeadsFull cfg.numKeyValueHeadsSWA
    let maxHeadDim := max cfg.headDimFull cfg.headDimSWA
    executeZeroKVRow device kvCache.kBuf kvCache.vBuf maxKVHeads cfg.maxSeqLen maxHeadDim pos

/-- Generate with Smart KV-Cache on Gemma 4.
    Uses cosine surprise sensor + KV eviction. -/
def generateWithSmartKV (device : Device) (model : Gemma4Model)
    (promptTokens : Array Nat) (maxTokens : Nat)
    (config : SmartKVConfig)
    (strategy : Sampling.Strategy := .Greedy)
    (eosToken : Option Nat := none)
    (verbose : Bool := true)
    : IO (Array Nat) := do
  let dim := model.config.hiddenSize

  IO.println "╔══════════════════════════════════════════════════════════╗"
  IO.println "║  Gemma 4 + Cosine Surprise Smart KV-Cache              ║"
  IO.println "╚══════════════════════════════════════════════════════════╝"
  IO.println s!"  Model: {dim}d, {model.config.numHiddenLayers}L, vocab={model.config.vocabSize}"
  IO.println s!"  Window: {config.windowSize}, Surprise tau: {config.tau} (cosine distance)"
  IO.println s!"  Prompt: {promptTokens.size} tokens, generate up to {maxTokens}"
  IO.println ""

  -- Full KV cache (original maxSeqLen) — Gemma4's attention is untouched
  let state ← createInferenceState device model.config

  -- Sensor: CE loss between base logits and next token (same as BitNet TTT).
  -- This is the most reliable surprise metric — directly measures
  -- "how wrong was the model about the next token?"
  let vocabSize := model.config.vocabSize
  let lossBuf ← createBuffer device {
    size := 4, usage := [.storage, .copySrc, .copyDst, .mapRead], mappedAtCreation := false
  }
  let targetBuf ← createBuffer device {
    size := 4, usage := [.storage, .copySrc, .copyDst], mappedAtCreation := false
  }

  let mut tokens := promptTokens
  let mut hasPrevHidden := false
  let mut sinkPositions : Array Nat := #[]  -- protected absolute positions

  -- ═══════════════════════════════════════════
  -- Prefill: forward + sensor + eviction
  -- ═══════════════════════════════════════════
  IO.println s!"[Prefill] Processing {promptTokens.size} prompt tokens..."
  let prefillStart ← IO.monoNanosNow

  for i in [0:promptTokens.size] do
    if i >= model.config.maxSeqLen then break

    -- Standard forward at absolute position i
    forwardSingleToken device model promptTokens[i]! i state

    -- CE surprise sensor: how wrong was the model about the next token?
    -- CE loss > tau means the model was "surprised" by this token.
    let mut surprise : Float := 0.0
    let mut isSurprise := false
    if i + 1 < promptTokens.size then
      -- Target = next token in the prompt
      let target := promptTokens[i + 1]!
      let tBytes := BufferOps.uint32ToBytes target.toUInt32
      writeBuffer device targetBuf 0 tBytes
      -- CE loss of current logits vs next token
      Loss.executeCrossEntropyForward device state.logitsBuf targetBuf lossBuf vocabSize
      surprise ← SafeBuffer.safeReadF32 device lossBuf
      isSurprise := surprise > config.tau

    -- If surprise, protect this position
    if isSurprise then
      sinkPositions := sinkPositions.push i

    -- Eviction: zero out old positions outside the window
    if i >= config.windowSize then
      let evictPos := i - config.windowSize
      -- Only evict if NOT a protected sink
      if !sinkPositions.contains evictPos then
        evictPosition device state model.config evictPos

    if verbose then
      let typeStr := if isSurprise then s!"SURPRISE ⚓ (sink #{sinkPositions.size})"
        else s!"normal"
      if i < 10 || isSurprise || i >= promptTokens.size - 3 then
        IO.println s!"  Token {i} (id={promptTokens[i]!}): ce_loss={surprise}, {typeStr}"

  let prefillEnd ← IO.monoNanosNow
  let prefillMs := (prefillEnd - prefillStart).toFloat / 1_000_000.0
  IO.println s!"[Prefill] Done in {prefillMs} ms"
  IO.println s!"  Sink positions: {sinkPositions.size} protected tokens"
  if sinkPositions.size > 0 && sinkPositions.size <= 40 then
    IO.println s!"  Protected: {sinkPositions}"
  IO.println ""

  -- ═══════════════════════════════════════════
  -- Decode
  -- ═══════════════════════════════════════════
  IO.println "[Decode] Generating..."
  let genStart ← IO.monoNanosNow
  let mut genTokenCount : Nat := 0

  for step in [0:maxTokens] do
    if tokens.size >= model.config.maxSeqLen then break

    let logits ← BufferOps.downloadFloatArray device state.logitsBuf model.config.vocabSize
    let (nextToken, _) := Sampling.sampleWithRNG logits strategy
      (Sampling.RNG.create (some (42 + step)))

    if verbose then IO.println s!"  Decode step {step}: token={nextToken}"
    tokens := tokens.push nextToken
    genTokenCount := genTokenCount + 1

    match eosToken with
    | some eos => if nextToken == eos then IO.println "  EOS"; break
    | none => pure ()

    let newPos := tokens.size - 1
    if newPos < model.config.maxSeqLen then
      forwardSingleToken device model nextToken newPos state

  let genEnd ← IO.monoNanosNow
  let genMs := (genEnd - genStart).toFloat / 1_000_000.0
  IO.println s!"\nGenerated {genTokenCount} tokens in {genMs} ms"
  pure tokens

/-- Baseline: dumb sliding window with KV eviction (no surprise gate).
    Zeros out ALL positions older than windowSize — no protection. -/
def generateWithDumbWindow (device : Device) (model : Gemma4Model)
    (promptTokens : Array Nat) (maxTokens : Nat)
    (windowSize : Nat)
    (strategy : Sampling.Strategy := .Greedy)
    : IO (Array Nat) := do
  let state ← createInferenceState device model.config

  for i in [0:promptTokens.size] do
    if i >= model.config.maxSeqLen then break
    forwardSingleToken device model promptTokens[i]! i state
    -- Evict ALL old positions (no protection)
    if i >= windowSize then
      evictPosition device state model.config (i - windowSize)

  let mut tokens := promptTokens
  for step in [0:maxTokens] do
    if tokens.size >= model.config.maxSeqLen then break
    let logits ← BufferOps.downloadFloatArray device state.logitsBuf model.config.vocabSize
    let (nextToken, _) := Sampling.sampleWithRNG logits strategy
      (Sampling.RNG.create (some (42 + step)))
    tokens := tokens.push nextToken
    let newPos := tokens.size - 1
    if newPos < model.config.maxSeqLen then
      forwardSingleToken device model nextToken newPos state
      if newPos >= windowSize then
        evictPosition device state model.config (newPos - windowSize)

  pure tokens

end Hesper.TTT.SmartKVGemma4
