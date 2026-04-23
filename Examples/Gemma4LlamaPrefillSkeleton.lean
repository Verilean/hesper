import Hesper.Backend.CUDA
import Hesper.CUDA.FFI
import Hesper.Models.Gemma4
import Hesper.Models.Gemma4.LlamaForwardPrefill
import Hesper.Tokenizer.SentencePiece
import Hesper.GGUF.Parser
import Hesper.WebGPU.BufferOps

/-!
# Phase 0 v3 LlamaPath prefill dispatch-count driver + parity driver

Two modes:
* **No prompt** → dispatch-count check only.  Runs the stub with no token
  input; reports total dispatches vs llama.cpp's ~2016 reference.
* **With prompt** → parity mode.  Tokenises the prompt, runs the stub
  with real embeddings, and (if `HESPER_GOLDEN_DUMP_DIR` is set) writes
  matching llama.cpp `cb()`-named tensors to that directory for diffing
  against `llama-eval-callback` output.

Usage:
  # dispatch count only
  HESPER_DP4A=1 lake exe gemma4-llama-prefill-skeleton \
    data/gemma-4-e4b-it-Q4_K_M.gguf 50

  # parity mode
  HESPER_DP4A=1 HESPER_GOLDEN_DUMP_DIR=/tmp/hesper_dump \
    lake exe gemma4-llama-prefill-skeleton \
      data/gemma-4-e4b-it-Q4_K_M.gguf "Hello world how are you"
-/

open Hesper
open Hesper.Models.Gemma4

unsafe def main (args : List String) : IO Unit := do
  let ggufPath := args.getD 0 "data/gemma-4-e4b-it-Q4_K_M.gguf"
  -- Second arg: either an integer seqLen (dispatch-count mode) or a prompt string.
  let arg1 := args.getD 1 "50"
  let promptMode := arg1.toNat?.isNone
  let prompt := arg1
  -- Third arg: number of tokens to generate in parity mode (default 1).
  -- Each decode step re-runs the full prefill on `(prompt ++ generated)`,
  -- so this is O(N²) — fine for flow-correctness checks and small N.
  let maxTokens := (args.getD 2 "1").toNat!

  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║  Gemma 4 LlamaPath v3 Prefill Skeleton       ║"
  IO.println "╚══════════════════════════════════════════════╝"
  IO.println s!"  Model: {ggufPath}"
  if promptMode then
    IO.println s!"  Mode: parity (prompt=\"{prompt}\")"
  else
    IO.println s!"  Mode: dispatch-count (seqLen={prompt})"

  let ctx ← CUDAContext.init
  IO.println "[Load] Reading GGUF..."
  let ggufData ← Hesper.CUDA.readFileFast ggufPath
  let model ← Gemma4Model.fromGGUFData ctx ggufData
  let cfg := model.config
  IO.println s!"[Config] {cfg.numHiddenLayers} layers, hidden={cfg.hiddenSize}"

  let state ← createInferenceState ctx cfg

  if !promptMode then
    -- Dispatch-count-only: single forward, no prompt tokens, no sampling.
    let seqLen := prompt.toNat!
    Hesper.resetDispatchCounter
    let startNs ← IO.monoNanosNow
    let _ ← forwardPrefillLlamaCpp ctx model seqLen state
    let endNs ← IO.monoNanosNow
    let totalDisp ← Hesper.getDispatchCounter
    let wallMs := (endNs - startNs).toFloat / 1000000.0
    IO.println ""
    IO.println "───────────── Result ─────────────"
    IO.println s!"  total dispatches : {totalDisp}"
    IO.println s!"  wall clock (ms)  : {wallMs}"
    if totalDisp >= 1900 ∧ totalDisp <= 2250 then
      IO.println "✓ PASS: dispatch count near llama.cpp prefill reference (±10%)"
    else
      IO.println s!"✗ FAIL: expected 1900..2250 total dispatches, got {totalDisp}"
    return

  -- ─────────────────── Parity / greedy generation mode ───────────────────
  IO.println "[Tokenize] Loading tokenizer..."
  let gguf ← match Hesper.GGUF.Parser.parseGGUF ggufData with
    | .ok g => pure g
    | .error e => throw (IO.userError s!"GGUF parse error: {e}")
  let tokenizer ← Hesper.Tokenizer.SentencePiece.fromGGUF gguf
  let initialToks := Hesper.Tokenizer.SentencePiece.encode tokenizer prompt
  IO.println s!"[Tokenize] Prompt: {initialToks.size} tokens: {initialToks.toList}"

  let vocab := cfg.vocabSize
  -- Running token array (prompt + generated).
  let mut tokens : Array Nat := initialToks
  let mut generatedIds : Array Nat := #[]

  -- We allocate the token-id GPU buffer once at the maximum size we might
  -- need (initialToks.size + maxTokens) and re-upload each iteration.
  let maxSeq := initialToks.size + maxTokens
  let tokenIdsBuf ← GPUBackend.allocBuffer ctx (maxSeq * 4).toUSize

  let genStart ← IO.monoNanosNow
  let mut lastDisp : Nat := 0
  for step in [0:maxTokens] do
    let n := tokens.size
    -- Upload tokens[0..n).
    let mut bytes : ByteArray := ByteArray.empty
    for i in [0:n] do
      bytes := bytes ++ Hesper.WebGPU.BufferOps.uint32ToBytes tokens[i]!.toUInt32
    GPUBackend.writeBuffer ctx tokenIdsBuf bytes

    Hesper.resetDispatchCounter
    let stepStart ← IO.monoNanosNow
    let logitsOpt ← forwardPrefillLlamaCpp ctx model n state
      (tokenIdsBuf := some tokenIdsBuf)
    let stepEnd ← IO.monoNanosNow
    lastDisp ← Hesper.getDispatchCounter

    match logitsOpt with
    | some logitsBuf =>
      let lbytes ← GPUBackend.readBuffer ctx logitsBuf (vocab * 4).toUSize
      let mut bestIdx : Nat := 0
      let mut bestVal : Float := Float.ofNat 0
      let mut first := true
      for i in [0:vocab] do
        let fb ← Hesper.Basic.bytesToFloat32 lbytes (i * 4)
        if first ∨ fb > bestVal then
          bestVal := fb; bestIdx := i; first := false
      let decoded := Hesper.Tokenizer.SentencePiece.decode tokenizer #[bestIdx]
      let stepMs := (stepEnd - stepStart).toFloat / 1000000.0
      IO.println s!"[step {step}] seqLen={n} next={bestIdx} \"{decoded}\" ({stepMs}ms, {lastDisp} dispatches)"
      tokens := tokens.push bestIdx
      generatedIds := generatedIds.push bestIdx
    | none =>
      IO.println s!"[step {step}] forward returned no logits, stopping"
      break
  let genEnd ← IO.monoNanosNow
  let genMs := (genEnd - genStart).toFloat / 1000000.0
  let tps := if genMs > 0 then generatedIds.size.toFloat * 1000.0 / genMs else 0.0

  IO.println ""
  IO.println "───────────── Result ─────────────"
  IO.println s!"  prompt tokens    : {initialToks.size}"
  IO.println s!"  generated tokens : {generatedIds.size}"
  IO.println s!"  total time (ms)  : {genMs}"
  IO.println s!"  tokens / sec     : {tps}"
  IO.println s!"  last forward disp: {lastDisp}"
  let fullDecoded := Hesper.Tokenizer.SentencePiece.decode tokenizer generatedIds
  IO.println s!"  prompt  : \"{prompt}\""
  IO.println s!"  completion: \"{fullDecoded}\""
  match ← IO.getEnv "HESPER_GOLDEN_DUMP_DIR" with
  | some dir => IO.println s!"[Golden] Dumps from the LAST forward written to {dir}/"
  | none => pure ()
