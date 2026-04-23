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

  -- Determine seqLen and, if parity mode, prepare tokenIdsBuf.
  let (seqLen, tokenIdsBufOpt) ← if promptMode then do
    IO.println "[Tokenize] Loading tokenizer..."
    let gguf ← match Hesper.GGUF.Parser.parseGGUF ggufData with
      | .ok g => pure g
      | .error e => throw (IO.userError s!"GGUF parse error: {e}")
    let tokenizer ← Hesper.Tokenizer.SentencePiece.fromGGUF gguf
    let toks := Hesper.Tokenizer.SentencePiece.encode tokenizer prompt
    IO.println s!"[Tokenize] Prompt: {toks.size} tokens: {toks.toList}"
    let n := toks.size
    -- Allocate u32 buffer and upload token IDs.
    let buf ← GPUBackend.allocBuffer ctx (n * 4).toUSize
    -- Pack into ByteArray.
    let mut bytes : ByteArray := ByteArray.empty
    for i in [0:n] do
      bytes := bytes ++ Hesper.WebGPU.BufferOps.uint32ToBytes toks[i]!.toUInt32
    GPUBackend.writeBuffer ctx buf bytes
    pure (n, some buf)
  else
    pure (prompt.toNat!, none)

  Hesper.resetDispatchCounter
  let startNs ← IO.monoNanosNow
  forwardPrefillLlamaCpp ctx model seqLen state (tokenIdsBuf := tokenIdsBufOpt)
  let endNs ← IO.monoNanosNow
  let totalDisp ← Hesper.getDispatchCounter
  let wallMs := (endNs - startNs).toFloat / 1000000.0

  IO.println ""
  IO.println "───────────── Result ─────────────"
  IO.println s!"  total dispatches : {totalDisp}"
  IO.println s!"  wall clock (ms)  : {wallMs}"
  IO.println ""
  IO.println s!"  expected (v3 loop-faithful stub)  : ~2160 dispatches"
  IO.println s!"  llama.cpp reference               : ~2016 kernels/forward (nsys, graphs disabled)"
  if totalDisp >= 1900 ∧ totalDisp <= 2250 then
    IO.println "✓ PASS: dispatch count near llama.cpp prefill reference (±10%)"
  else
    IO.println s!"✗ FAIL: expected 1900..2250 total dispatches, got {totalDisp}"
  match ← IO.getEnv "HESPER_GOLDEN_DUMP_DIR" with
  | some dir => IO.println s!"[Golden] Dumps written to {dir}/"
  | none => pure ()
