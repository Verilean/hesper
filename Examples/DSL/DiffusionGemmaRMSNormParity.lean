import Hesper.Models.DiffusionGemma.Reference
import Hesper.Basic

/-!
# DiffusionGemma RMSNorm parity vs ggml golden

Loads the deterministic input + golden output produced by
`scripts/llama_parity/dump_dg_rmsnorm_golden` (ggml's
`rms_norm(x,eps) * weight`, the same op as llama.cpp's `build_norm`),
runs Hesper's RMSNorm on the SAME input, and compares.

This is the Gemma4/BitNet golden pattern: the reference value comes from
llama.cpp's CPU backend, so a passing test means our implementation
reproduces the reference op (not just an internally-consistent CPU model).

The CPU reference computes in f64; ggml in f32 — so we check within f32
precision rather than bit-exact.  The eventual WGSL/CUDA kernel (f32) is
held to a tighter, bit-exact bound against the same golden.

Run (after generating the golden):
  ./scripts/llama_parity/dump_dg_rmsnorm_golden /tmp/dg_golden/rmsnorm
  lake exe diffusiongemma-rmsnorm-parity
-/

open Hesper.Models.DiffusionGemma.Reference

/-- Read a raw little-endian f32 binary file into an `Array Float`. -/
def readF32Bin (path : String) : IO (Array Float) := do
  let bytes ← IO.FS.readBinFile path
  let n := bytes.size / 4
  let mut a := Array.replicate n 0.0
  for i in [0:n] do
    let b0 := (bytes.get! (i*4)    ).toUInt32
    let b1 := (bytes.get! (i*4 + 1)).toUInt32
    let b2 := (bytes.get! (i*4 + 2)).toUInt32
    let b3 := (bytes.get! (i*4 + 3)).toUInt32
    let bits := b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24)
    a := a.set! i (Hesper.Basic.float32BitsToFloat64 bits)
  return a

def maxAbsDiff (a b : Array Float) : Float := Id.run do
  let mut m := 0.0
  for i in [0:min a.size b.size] do
    let d := (a[i]! - b[i]!).abs
    if d > m then m := d
  return m

def main (args : List String) : IO Unit := do
  let dir := args.head?.getD "/tmp/dg_golden/rmsnorm"
  IO.println s!"[dg-rmsnorm-parity] golden dir = {dir}"
  let x ← readF32Bin s!"{dir}/x.bin"
  let w ← readF32Bin s!"{dir}/w.bin"
  let gold ← readF32Bin s!"{dir}/out.bin"
  let eps := 1e-6   -- diffusion-gemma.attention.layer_norm_rms_epsilon (matches dumper)

  let out := rmsNorm x (some w) eps
  let err := maxAbsDiff out gold
  IO.println s!"  n={x.size}  maxAbsErr={err}"
  IO.println s!"  ref[0..4]   = {(out.extract 0 4).toList}"
  IO.println s!"  gold[0..4]  = {(gold.extract 0 4).toList}"
  let tol := 1e-4
  if out.size == gold.size && err < tol then
    IO.println s!"✓ PASS (within f32 tolerance {tol})"
  else
    IO.println s!"✗ FAIL (err {err} ≥ tol {tol}, or size mismatch {out.size} vs {gold.size})"
    throw (IO.userError "rmsnorm parity failed")
