import Hesper
import Hesper.WebGPU.Device
import Hesper.Compute
import Hesper.WGSL.DSL
import Hesper.Basic

/-!
# DiffusionGemma logit-softcap parity on the GPU (WebGPU → Metal)

Runs the softcap op `30 * tanh(x / 30)` as a real Hesper **GPU** kernel via
`parallelForDSL` on the WebGPU/Dawn→Metal backend, and compares to the same
ggml golden used by the CPU parity test.  Demonstrates the per-module
kernel path runs on macOS — no CUDA required.

Run:
  ./scripts/llama_parity/dump_dg_softcap_golden /tmp/dg_golden/softcap
  lake exe diffusiongemma-softcap-gpu-parity
-/

open Hesper.WebGPU
open Hesper.WGSL

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
  let dir := args.head?.getD "/tmp/dg_golden/softcap"
  IO.println "[dg-softcap-GPU-parity] initializing WebGPU (Metal on macOS)..."
  let inst ← Hesper.init
  let device ← getDevice inst
  let x ← readF32Bin s!"{dir}/x.bin"
  let gold ← readF32Bin s!"{dir}/out.bin"

  -- GPU elementwise kernel: out = 30 * tanh(x * (1/30))
  let out ← Hesper.Compute.parallelForDSL device
    (fun v => Exp.mul (Exp.tanh (Exp.mul v (Exp.litF32 (1.0/30.0)))) (Exp.litF32 30.0)) x

  let mut err := 0.0
  for i in [0:x.size] do
    let d := (out[i]! - gold[i]!).abs
    if d > err then err := d
  IO.println s!"  n={x.size}  maxAbsErr={err}"
  IO.println s!"  gpu[0..4]  = {(out.extract 0 4).toList}"
  IO.println s!"  gold[0..4] = {(gold.extract 0 4).toList}"
  let tol := 1e-3
  if out.size == gold.size && err < tol then
    IO.println s!"✓ PASS on Metal (within f32 GPU tolerance {tol}) — no CUDA needed"
  else
    IO.println s!"✗ FAIL (err {err} ≥ tol {tol})"
    throw (IO.userError "softcap GPU parity failed")
