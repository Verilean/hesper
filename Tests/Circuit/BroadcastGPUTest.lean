import Hesper.Backend.CUDA
import Hesper.Circuit.IR
import Hesper.Circuit.Lowering
import Hesper.Circuit.Passes
import Hesper.WebGPU.BufferOps
import Hesper.Basic

/-!
End-to-end GPU proof that broadcast-scalar pointwise ops lower
correctly: a full-shape input `a` and a 1-element scalar input `s`
combined as `(a * s) * 1.0` through two CircuitM ops; after
fusePointwise it collapses into a single dispatch that reads `s` at
index 0 on every lane.
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

  let scaleBytes ← Hesper.WebGPU.BufferOps.floatArrayToBytes #[3.5]
  let scaleBuf ← GPUBackend.allocBuffer (β := β) ctx (4 : USize)
  GPUBackend.writeBuffer (β := β) ctx scaleBuf scaleBytes

  let outBuf ← GPUBackend.allocBuffer (β := β) ctx (4 * n).toUSize

  -- (a * s) * 1.0  — the * 1.0 step exists to force a 2-op chain so
  -- fusePointwise has something to collapse.
  let cacheRef : IO.Ref (Option (CompiledCircuit β)) ← IO.mkRef none
  Hesper.Circuit.runCachedFused ctx cacheRef
    (do
      let a ← CircuitM.registerExternal
                (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
                inBuf #[n] .f32 .Global
      let s ← CircuitM.registerExternal
                (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
                scaleBuf #[1] .f32 .Global
      let scaled ← CircuitM.scaleByBroadcast a s
      let _final ← CircuitM.map scaled (.mul (.input 0) (.const 1.0))
      pure ())
    -- Tensor ids: 0 = a (external), 1 = s (external, scalar),
    -- 2 = scaled (internal), 3 = final (caller-facing).
    [(0, inBuf), (1, scaleBuf), (3, outBuf)]

  let outBytes ← GPUBackend.readBuffer (β := β) ctx outBuf (4 * n).toUSize
  let mut okCount := 0
  let mut firstErr : Option (Nat × Float × Float) := none
  for i in [0:n] do
    let actual ← Hesper.Basic.bytesToFloat32 outBytes (i * 4)
    let expected := i.toFloat * 3.5
    let err := (actual - expected).abs
    if err < 1e-3 then okCount := okCount + 1
    else if firstErr.isNone then firstErr := some (i, actual, expected)

  if okCount == n then
    IO.println s!"✓ PASS: all {n} elements match f(x) = x * 3.5 within 1e-3"
  else
    IO.println s!"✗ FAIL: {okCount}/{n} matched"
    match firstErr with
    | some (i, a, e) => IO.println s!"  first mismatch: i={i} actual={a} expected={e}"
    | none => pure ()
    IO.Process.exit 1
