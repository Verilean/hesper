import Hesper.Backend.CUDA
import Hesper.Circuit.IR
import Hesper.Circuit.Lowering
import Hesper.Circuit.Passes
import Hesper.WebGPU.BufferOps
import Hesper.Basic

/-!
End-to-end GPU proof: a 3-op pointwise chain in CircuitM, compiled and
dispatched through the CUDA backend, with fusion enabled.

Correctness: the fused kernel's output must equal the mathematically
composed function `f(x) = ((x * 2) + 1)²` over every element of a
128-element f32 tensor initialized to x = i.

Diagnostic (informal): runs with and without `runCachedFused` paths
at the Lean level — both should land on the same compiled PTX once
fusePointwise has collapsed the chain.
-/

open Hesper
open Hesper.Circuit

abbrev β := Hesper.CUDAContext

unsafe def main : IO Unit := do
  let n : Nat := 128
  let ctx ← Hesper.CUDAContext.init

  let inputArr : Array Float := Array.ofFn (n := n) fun i => i.val.toFloat
  let inputBytes ← Hesper.WebGPU.BufferOps.floatArrayToBytes inputArr
  let inBuf ← GPUBackend.allocBuffer (β := β) ctx (4 * n).toUSize
  GPUBackend.writeBuffer (β := β) ctx inBuf inputBytes

  let outBuf ← GPUBackend.allocBuffer (β := β) ctx (4 * n).toUSize

  -- Build a 3-op pointwise chain, run through the fused pipeline.
  -- scale(x, 2) → add(_, 1) → square(_)
  let cacheRef : IO.Ref (Option (CompiledCircuit β)) ← IO.mkRef none
  Hesper.Circuit.runCachedFused ctx cacheRef
    (do
      let a ← CircuitM.registerExternal
                (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
                inBuf #[n] .f32 .Global
      let doubled ← CircuitM.scale a 2.0
      let inc     ← CircuitM.map doubled (.add (.input 0) (.const 1.0))
      let _final  ← CircuitM.map inc (.mul (.input 0) (.input 0))
      pure ())
    -- Tensor id assignment: 0 = external input, 1 = doubled, 2 = inc,
    -- 3 = final.  We protect id 3 so the fusion pass doesn't elide it.
    -- (Ids 1 and 2 are internal; fusePointwise will inline them away.)
    [(0, inBuf), (3, outBuf)]

  IO.println "Waiting for GPU..."

  -- Read back outputBuf
  let outBytes ← GPUBackend.readBuffer (β := β) ctx outBuf (4 * n).toUSize
  let mut okCount := 0
  let mut firstErr : Option (Nat × Float × Float) := none
  for i in [0:n] do
    let actual ← Hesper.Basic.bytesToFloat32 outBytes (i * 4)
    let expected : Float := ((i.toFloat * 2.0) + 1.0)
    let expected2 := expected * expected
    let err := (actual - expected2).abs
    if err < 1e-3 then
      okCount := okCount + 1
    else
      if firstErr.isNone then firstErr := some (i, actual, expected2)

  if okCount == n then
    IO.println s!"✓ PASS: all {n} elements match f(x) = ((x*2)+1)² within 1e-3"
  else
    IO.println s!"✗ FAIL: {okCount}/{n} matched"
    match firstErr with
    | some (i, a, e) => IO.println s!"  first mismatch: i={i} actual={a} expected={e}"
    | none => pure ()
    IO.Process.exit 1
