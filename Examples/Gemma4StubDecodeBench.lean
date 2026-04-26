import Hesper.Backend.CUDA
import Hesper.CUDA.FFI
import Hesper.Models.Gemma4
import Hesper.Models.Gemma4.LlamaForward

/-!
# Stub-forward decode bench (doc 57 H4b probe)

Loops `forwardTokenStubPerLayer` for N tokens.  The stub kernels are no-ops
on the GPU (1×1×1 workgroup with a trivial body), so the GPU contributes
near-zero per-token time.  The wall we observe is therefore **almost
entirely Lean-side host overhead**: cudaExecuteImpl, kcr lookup, args
Array build/expand, refcount work, FFI marshalling — the same per-block
cost the real `forwardSingleToken` pays on top of GPU work.

Comparison points:
- real `forwardSingleToken` graphs-OFF: ~5 ms/token forward host time
  (`HESPER_DECODE_SECT_TRACE=1`) — sits in series with the GPU drain.
- this bench at the same dispatch shape (~17 ops × 42 layers = ~714):
  what we measure here.

If this bench reports >> 0.5 ms / token, we have direct evidence that
the per-block hesper-specific overhead is real.  If it reports near
zero, the gap is somewhere else entirely.

Usage:
  HESPER_DP4A=1 lake exe gemma4-stub-decode-bench [model.gguf] [N=300]
  HESPER_STUB_KCR=1 …   — route through KernelCacheRefs (cache-hit cost)
-/

open Hesper
open Hesper.Models.Gemma4

unsafe def main (args : List String) : IO Unit := do
  let ggufPath := args.getD 0 "data/gemma-4-e4b-it-Q4_K_M.gguf"
  let n        := (args.getD 1 "300").toNat!

  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║  Gemma 4 Stub-Forward Decode Bench (H4b)     ║"
  IO.println "╚══════════════════════════════════════════════╝"
  IO.println s!"  Model     : {ggufPath}"
  IO.println s!"  N tokens  : {n}"

  let ctx ← CUDAContext.init
  IO.println "[Load] Reading GGUF..."
  let ggufData ← Hesper.CUDA.readFileFast ggufPath
  let model ← Gemma4Model.fromGGUFData ctx ggufData
  let cfg := model.config
  IO.println s!"[Config] {cfg.numHiddenLayers} layers, hidden={cfg.hiddenSize}"

  let state ← createInferenceState ctx cfg
  let useKcr := (← IO.getEnv "HESPER_STUB_KCR").isSome
  let kcrOpt ← if useKcr then do
    let k ← createKernelCacheRefs (β := CUDAContext)
    pure (some k)
  else pure none
  IO.println s!"  cache mode: {if useKcr then "kcr (cache-hit cost)" else "no-cache (cold path)"}"

  -- Warmup: first call does PTX JIT + module load.  Don't include in timing.
  IO.println "[Warmup] one stub forward..."
  forwardTokenStubPerLayer ctx model state (kcrOpt := kcrOpt)
  -- Drain the warmup to be safe.
  Hesper.CUDA.cuStreamSynchronize (0 : USize)

  -- Steady-state loop.
  Hesper.resetDispatchCounter
  let startNs ← IO.monoNanosNow
  for _i in [0:n] do
    forwardTokenStubPerLayer ctx model state (kcrOpt := kcrOpt)
  -- One drain at the very end so the wall captures all submitted work.
  Hesper.CUDA.cuStreamSynchronize (0 : USize)
  let endNs ← IO.monoNanosNow
  let dispTotal ← Hesper.getDispatchCounter
  let wallMs := (endNs - startNs).toFloat / 1e6
  let perTokMs := wallMs / n.toFloat

  IO.println ""
  IO.println "───────────── Result ─────────────"
  IO.println s!"  total dispatches        : {dispTotal}"
  IO.println s!"  dispatches per token    : {dispTotal / n}"
  IO.println s!"  total wall (ms)         : {wallMs}"
  IO.println s!"  wall per token (ms)     : {perTokMs}"
  IO.println s!"  effective TPS           : {1000.0 / perTokMs}"
  IO.println ""
  IO.println "Compare with HESPER_DECODE_SECT_TRACE=1 'forward' section on the"
  IO.println "real forwardSingleToken (~5 ms/tok at 60 TPS, graphs OFF). The"
  IO.println "delta is the GPU-side work; this number is the host-side floor."
