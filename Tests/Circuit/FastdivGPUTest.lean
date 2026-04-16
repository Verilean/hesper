import Hesper.Backend.CUDA
import Hesper.Circuit.IR
import Hesper.Circuit.Lowering
import Hesper.Circuit.Passes
import Hesper.WebGPU.BufferOps
import Hesper.Basic

/-!
End-to-end GPU test for `ScalarExp.fastdiv` — the host-computed magic-number
integer division primitive (llama.cpp's Granlund-Montgomery trick).

For a set of divisors `D = [1, 2, 3, 7, 32, 100, 256, 4096]`, we run a
pointwise kernel where each lane computes `lane / d` three ways:
  - ground truth (CPU-side array)
  - `ScalarExp.fastdiv` (CUDA lowers to `mulhi + add + shr`)
We compare every output with the CPU reference; any mismatch fails.

Also exercises the `d == 1` fast-path (identity) and `d` power-of-two.
-/

open Hesper
open Hesper.Circuit

abbrev β := Hesper.CUDAContext

unsafe def runForDivisor (ctx : β) (d : Nat) : IO Bool := do
  let D : Nat := 1024

  -- Input: x[i] = i (we'll just use laneIdx directly inside the kernel)
  let xArr : Array Float := Array.ofFn (n := D) fun i => i.val.toFloat
  let xBytes ← Hesper.WebGPU.BufferOps.floatArrayToBytes xArr
  let xBuf ← GPUBackend.allocBuffer (β := β) ctx (4 * D).toUSize
  GPUBackend.writeBuffer (β := β) ctx xBuf xBytes

  let dstBuf ← GPUBackend.allocBuffer (β := β) ctx (4 * D).toUSize
  let zeros : Array Float := Array.ofFn (n := D) fun _ => 0.0
  GPUBackend.writeBuffer (β := β) ctx dstBuf (← Hesper.WebGPU.BufferOps.floatArrayToBytes zeros)

  let ccRef : IO.Ref (Option (CompiledCircuit β)) ← IO.mkRef none
  Hesper.Circuit.runCachedFused ctx ccRef
    (do
      let x ← CircuitM.registerExternal
                (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
                xBuf #[D] .f32 .Global
      let dst ← CircuitM.registerExternal
                  (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
                  dstBuf #[D] .f32 .Global
      -- Pure pointwise: dst[lane] = fastdiv(x[lane], d)
      let body : ScalarExp := ScalarExp.mkFastdiv (.input 0) d
      let _ ← CircuitM.scatterInto dst #[D] #[x] body .laneIdx
      pure ())
    [(0, xBuf), (1, dstBuf)]

  let outBytes ← GPUBackend.readBuffer (β := β) ctx dstBuf (4 * D).toUSize

  let mut ok := 0
  let mut firstErr : Option (Nat × Float × Float) := none
  for i in [0:D] do
    let actual ← Hesper.Basic.bytesToFloat32 outBytes (i * 4)
    let expected : Float := (i / d).toFloat
    if (actual - expected).abs < 0.5 then
      ok := ok + 1
    else if firstErr.isNone then
      firstErr := some (i, actual, expected)

  if ok == D then
    IO.println s!"  ✓ d={d}: all {D} values correct"
    return true
  else
    IO.println s!"  ✗ d={d}: {ok}/{D} correct"
    match firstErr with
    | some (i, a, e) => IO.println s!"    first mismatch: {i}/d={i/d}, expected {e}, got {a}"
    | none => pure ()
    return false

unsafe def main : IO Unit := do
  let ctx ← Hesper.CUDAContext.init
  IO.println "Testing ScalarExp.fastdiv (Granlund-Montgomery integer division)..."

  let divisors := [1, 2, 3, 7, 32, 100, 256, 4096]
  let mut allOk := true
  for d in divisors do
    let ok ← runForDivisor ctx d
    if !ok then allOk := false

  if allOk then
    IO.println "✓ PASS — all divisors verified"
  else
    IO.println "✗ FAIL"
    IO.Process.exit 1
