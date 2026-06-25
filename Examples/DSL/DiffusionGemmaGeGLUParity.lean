import Hesper.Models.DiffusionGemma.Reference
import Hesper.Basic

/-!
# DiffusionGemma GeGLU parity vs ggml golden  (`gelu(gate) * up`)

Golden from `scripts/llama_parity/dump_dg_geglu_golden`.  ggml's CPU
`gelu` uses an f16 lookup table, so the exact-tanh reference differs by
~1e-3 — checked within a relaxed tolerance.  (The WGSL/CUDA kernel uses a
formula gelu and will match the reference more tightly.)

Run:
  ./scripts/llama_parity/dump_dg_geglu_golden /tmp/dg_golden/geglu
  lake exe diffusiongemma-geglu-parity
-/

open Hesper.Models.DiffusionGemma.Reference

def readF32Bin (path : String) : IO (Array Float) := do
  let bytes ← IO.FS.readBinFile path
  let n := bytes.size / 4
  let mut a := Array.replicate n 0.0
  for i in [0:n] do
    let b0 := (bytes.get! (i*4)    ).toUInt32
    let b1 := (bytes.get! (i*4 + 1)).toUInt32
    let b2 := (bytes.get! (i*4 + 2)).toUInt32
    let b3 := (bytes.get! (i*4 + 3)).toUInt32
    a := a.set! i (Hesper.Basic.float32BitsToFloat64 (b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24)))
  return a

def main (args : List String) : IO Unit := do
  let dir := args.head?.getD "/tmp/dg_golden/geglu"
  IO.println s!"[dg-geglu-parity] golden dir = {dir}"
  let gate ← readF32Bin s!"{dir}/gate.bin"
  let up   ← readF32Bin s!"{dir}/up.bin"
  let gold ← readF32Bin s!"{dir}/out.bin"
  let mut out := Array.replicate gate.size 0.0
  for i in [0:gate.size] do
    out := out.set! i (gelu gate[i]! * up[i]!)
  let mut err := 0.0
  for i in [0:gate.size] do
    let d := (out[i]! - gold[i]!).abs
    if d > err then err := d
  IO.println s!"  n={gate.size}  maxAbsErr={err}"
  let tol := 3e-3   -- ggml table-gelu vs exact-tanh
  if out.size == gold.size && err < tol then
    IO.println s!"✓ PASS (within table-gelu tolerance {tol})"
  else
    IO.println s!"✗ FAIL (err {err} ≥ tol {tol})"
    throw (IO.userError "geglu parity failed")
