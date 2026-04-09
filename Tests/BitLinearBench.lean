import Hesper
import Hesper.Layers.BitLinear
import Hesper.Layers.BitLinearSpec
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WebGPU.BufferOps
import Hesper.Basic

/-!
# BitLinear Microbenchmark

Times `BitLinear.forward` at a variety of `numRows` values for a
BitNet-2B-style layer shape (`inDim=2560, outDim=2560`). Both the
subgroup-matrix path (M≥16) and the shared-memory tiled fallback path
(M<16 or adapter without SubgroupMatrix) are exercised depending on
`numRows`.

Output: ms/call and GFLOP/s for each configuration.

This is primarily useful as:
  - a regression test: a big drop in throughput after a kernel change
    is obvious from the numbers here.
  - a design knob test: lets us see at what `numRows` the subgroup
    matrix path starts to beat the fallback (if at all on a given
    adapter).
-/

open Hesper.WebGPU
open Hesper.Layers

namespace Tests.BitLinearBench

def benchCase (device : Device) (inDim outDim numRows : Nat) (warmups iters : Nat) : IO Unit := do
  -- Build a dummy layer (weights random ternary, deterministic seed).
  let seed : UInt32 := 0xDEADBEEF
  let mut rng := seed
  let xorshift (r : UInt32) : UInt32 :=
    let x1 := r ^^^ (r <<< 13)
    let x2 := x1 ^^^ (x1 >>> 17)
    x2 ^^^ (x2 <<< 5)
  let mut ternary : Array Int := Array.empty
  for _ in [:inDim * outDim] do
    rng := xorshift rng
    let v : Int := match rng.toNat % 3 with
      | 0 => -1
      | 1 => 0
      | _ => 1
    ternary := ternary.push v
  let packed := BitLinearSpec.packI2S ternary inDim outDim

  let cfg : BitLinear.Config :=
    { inDim := inDim, outDim := outDim, batchSize := numRows }
  let layer ← BitLinear.create device cfg packed 0.05

  -- Create buffers
  let inBuf ← createBuffer device {
    size := (numRows * inDim * 4).toUSize
    usage := [.storage, .copySrc, .copyDst]
    mappedAtCreation := false
  }
  let outBuf ← createBuffer device {
    size := (numRows * outDim * 4).toUSize
    usage := [.storage, .copySrc, .copyDst]
    mappedAtCreation := false
  }
  let mut inputArr : Array Float := Array.empty
  for i in [:numRows * inDim] do
    inputArr := inputArr.push ((i % 13).toFloat * 0.01 - 0.06)
  let inBytes ← Hesper.Basic.floatArrayToBytes inputArr
  writeBuffer device inBuf 0 inBytes

  -- Warm-up (JIT, pipeline cache, shader compile)
  for _ in [:warmups] do
    BitLinear.forward device layer inBuf outBuf numRows

  -- Timed loop
  let start ← IO.monoNanosNow
  for _ in [:iters] do
    BitLinear.forward device layer inBuf outBuf numRows
  -- Force GPU sync via a tiny readback
  let _ ← Hesper.WebGPU.BufferOps.downloadFloatArray device outBuf 1
  let stop ← IO.monoNanosNow

  let elapsedMs := (stop - start).toFloat / 1_000_000.0
  let msPerCall := elapsedMs / iters.toFloat
  let flops := 2.0 * numRows.toFloat * inDim.toFloat * outDim.toFloat
  let gflops := flops / (msPerCall * 1_000_000.0)

  IO.println s!"  inDim={inDim} outDim={outDim} numRows={numRows} : {msPerCall} ms/call, {gflops} GFLOP/s"

def main : IO Unit := do
  IO.println "═══════════════════════════════════════════════"
  IO.println "  BitLinear Microbenchmark"
  IO.println "═══════════════════════════════════════════════"
  -- Turn on the subgroup-matrix dispatch so we actually exercise it.
  BitLinear.subgroupMatrixOptInRef.set true
  let inst ← Hesper.init
  let device ← Hesper.WebGPU.getDevice inst
  IO.println ""
  IO.println "  BitNet layer shape (inDim=2560, outDim=2560):"
  for m in [1, 2, 4, 8, 16, 32, 64, 128, 256] do
    benchCase device 2560 2560 m 5 50
  IO.println ""
  IO.println "  Gemma-4-ish layer shape (inDim=2560, outDim=640):"
  for m in [1, 8, 16, 32, 64, 128] do
    benchCase device 2560 640 m 5 50

end Tests.BitLinearBench

def main : IO Unit := Tests.BitLinearBench.main
