import Hesper.Backend.CUDA
import Hesper.CUDA.FFI
import Hesper.Models.Gemma4
import Hesper.Models.Gemma4.LlamaForward
import Hesper.Tokenizer.SentencePiece
import Hesper.GGUF.Parser

/-!
# Phase 0 v3 LlamaPath dispatch-count driver

Calls `forwardTokenLlamaCpp` once and reports total dispatches.  v3 uses
batched-multilayer stubs so all 42 layers' compute happens inside the stub
kernels' gridY dimension; the host emits only ~17 dispatches per forward.

Usage:
  HESPER_DP4A=1 lake exe gemma4-llama-skeleton data/gemma-4-e4b-it-Q4_K_M.gguf
-/

open Hesper
open Hesper.Models.Gemma4

unsafe def main (args : List String) : IO Unit := do
  let ggufPath := args.getD 0 "data/gemma-4-e4b-it-Q4_K_M.gguf"

  IO.println "╔══════════════════════════════════════════╗"
  IO.println "║  Gemma 4 LlamaPath v3 Skeleton (batched) ║"
  IO.println "╚══════════════════════════════════════════╝"
  IO.println s!"  Model: {ggufPath}"

  let ctx ← CUDAContext.init
  IO.println "[Load] Reading GGUF..."
  let ggufData ← Hesper.CUDA.readFileFast ggufPath
  let model ← Gemma4Model.fromGGUFData ctx ggufData
  let cfg := model.config
  IO.println s!"[Config] {cfg.numHiddenLayers} layers, hidden={cfg.hiddenSize}"

  let state ← createInferenceState ctx cfg

  Hesper.resetDispatchCounter
  let startNs ← IO.monoNanosNow
  forwardTokenLlamaCpp ctx model 0 0 state
  let endNs ← IO.monoNanosNow
  let totalDisp ← Hesper.getDispatchCounter
  let wallMs := (endNs - startNs).toFloat / 1000000.0

  IO.println ""
  IO.println "───────────── Result ─────────────"
  IO.println s!"  total dispatches : {totalDisp}"
  IO.println s!"  wall clock (ms)  : {wallMs}"
  IO.println ""
  IO.println s!"  expected (v3 batched-multilayer) : ~17 dispatches per forward"
  IO.println s!"  llama.cpp reference             : ~113 kernels per forward (nsys)"
  if totalDisp >= 15 ∧ totalDisp <= 25 then
    IO.println "✓ PASS: dispatch count in batched-multilayer range"
  else
    IO.println s!"✗ FAIL: expected 15..25 total dispatches, got {totalDisp}"
