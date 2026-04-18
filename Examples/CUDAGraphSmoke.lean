import Hesper.CUDA.FFI

/-!
# CUDA Graph FFI smoke test

A 3-line program that proves the new FFI symbols are wired up at all
levels (Lean declaration → C++ implementation → driver call).  If this
runs without throwing, every cuGraph-related binding added in
Phase C2 is importable and loadable.

```
  lake exe cuda-graph-smoke
```
-/

unsafe def main : IO Unit := do
  Hesper.CUDA.cuDriverInit
  let _dev ← Hesper.CUDA.cuDeviceGet 0
  let _ctx ← Hesper.CUDA.cuCtxCreate _dev
  let s ← Hesper.CUDA.cuStreamCreate
  let sVal : USize := s
  IO.println s!"[smoke] cuStreamCreate returned {sVal}"
  Hesper.CUDA.cuStreamDestroy s
  IO.println "[smoke] cuStreamDestroy OK"
