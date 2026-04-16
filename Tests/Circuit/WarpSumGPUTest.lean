import Hesper.Backend.CUDA
import Hesper.Circuit.IR
import Hesper.Circuit.Lowering
import Hesper.Circuit.Passes
import Hesper.WebGPU.BufferOps
import Hesper.Basic

/-!
End-to-end GPU test for `ScalarExp.warpSum` — verifies that a warp-level
reduction (subgroupAdd) runs correctly inside a `Prim.scatter`.

Kernel (1 warp of 32 lanes, outShape = 32):
  value = warpSum(src[laneIdx])
  out[laneIdx] = value

Expected: every lane writes the same total ( Σ_{i=0..31} src[i] ).

Source values: src[i] = i + 1, so total = 32 * 33 / 2 = 528.
-/

open Hesper
open Hesper.Circuit

abbrev β := Hesper.CUDAContext

unsafe def main : IO Unit := do
  let n : Nat := 32
  let expected : Float := (n * (n + 1) / 2).toFloat  -- 528

  let ctx ← Hesper.CUDAContext.init

  let srcArr : Array Float := Array.ofFn (n := n) fun i => (i.val + 1).toFloat
  let srcBytes ← Hesper.WebGPU.BufferOps.floatArrayToBytes srcArr
  let srcBuf ← GPUBackend.allocBuffer (β := β) ctx (4 * n).toUSize
  GPUBackend.writeBuffer (β := β) ctx srcBuf srcBytes

  let zeros : Array Float := Array.ofFn (n := n) fun _ => 0.0
  let zeroBytes ← Hesper.WebGPU.BufferOps.floatArrayToBytes zeros
  let outBuf ← GPUBackend.allocBuffer (β := β) ctx (4 * n).toUSize
  GPUBackend.writeBuffer (β := β) ctx outBuf zeroBytes

  let ccRef : IO.Ref (Option (CompiledCircuit β)) ← IO.mkRef none
  Hesper.Circuit.runCachedFused ctx ccRef
    (do
      let src ← CircuitM.registerExternal
                  (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
                  srcBuf #[n] .f32 .Global
      -- value = warpSum(src[laneIdx])
      let _final ← CircuitM.map src (.warpSum (.input 0))
      pure ())
    [(0, srcBuf), (1, outBuf)]

  IO.println "Waiting for GPU..."

  let outBytes ← GPUBackend.readBuffer (β := β) ctx outBuf (4 * n).toUSize
  let mut ok := 0
  let mut firstErr : Option (Nat × Float) := none
  for i in [0:n] do
    let actual ← Hesper.Basic.bytesToFloat32 outBytes (i * 4)
    if (actual - expected).abs < 1e-3 then
      ok := ok + 1
    else if firstErr.isNone then firstErr := some (i, actual)

  if ok == n then
    IO.println s!"✓ PASS: all {n} lanes saw the same warp-sum total ({expected})"
  else
    IO.println s!"✗ FAIL: {ok}/{n} lanes matched (expected {expected})"
    match firstErr with
    | some (i, a) => IO.println s!"  first mismatch: lane {i} saw {a}"
    | none => pure ()
    IO.Process.exit 1
