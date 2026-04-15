import Hesper.Backend.CUDA
import Hesper.Circuit.IR
import Hesper.Circuit.Lowering
import Hesper.Circuit.Passes
import Hesper.WebGPU.BufferOps
import Hesper.Basic

/-!
Validates that `CircuitM.rmsNorm` — a composite builder assembled from
`Prim.reduceLastAxis` + three `Prim.pointwise` ops — produces the
correct RMSNorm output when lowered and run on the CUDA backend.

Reference on CPU:
  out[i] = x[i] * rsqrt(mean(x²) + eps) * scale[i]
Tolerance: 1e-4 (f32 reduction has ~1e-6 round-off per element; over
D=256 that's ~5e-5 — 1e-4 is comfortable).
-/

open Hesper
open Hesper.Circuit

abbrev β := Hesper.CUDAContext

def cpuRmsNorm (x scale : Array Float) (eps : Float) : Array Float := Id.run do
  let D := x.size
  let mut sumSq : Float := 0.0
  for i in [0:D] do sumSq := sumSq + x[i]! * x[i]!
  let invRms : Float := 1.0 / Float.sqrt (sumSq / D.toFloat + eps)
  let mut out : Array Float := Array.replicate D 0.0
  for i in [0:D] do out := out.set! i (x[i]! * invRms * scale[i]!)
  return out

unsafe def main : IO Unit := do
  let D : Nat := 256
  let eps : Float := 1e-6
  let ctx ← Hesper.CUDAContext.init

  -- Build deterministic non-trivial input + scale.
  let xArr : Array Float :=
    Array.ofFn (n := D) fun i => (i.val.toFloat * 0.017) - 2.0
  let scaleArr : Array Float :=
    Array.ofFn (n := D) fun i => 0.5 + 0.01 * i.val.toFloat

  let xBytes     ← Hesper.WebGPU.BufferOps.floatArrayToBytes xArr
  let scaleBytes ← Hesper.WebGPU.BufferOps.floatArrayToBytes scaleArr
  let xBuf     ← GPUBackend.allocBuffer (β := β) ctx (4 * D).toUSize
  let scaleBuf ← GPUBackend.allocBuffer (β := β) ctx (4 * D).toUSize
  GPUBackend.writeBuffer (β := β) ctx xBuf     xBytes
  GPUBackend.writeBuffer (β := β) ctx scaleBuf scaleBytes

  let outBuf ← GPUBackend.allocBuffer (β := β) ctx (4 * D).toUSize

  let cacheRef : IO.Ref (Option (CompiledCircuit β)) ← IO.mkRef none
  Hesper.Circuit.runCachedFused ctx cacheRef
    (do
      let x ← CircuitM.registerExternal
                (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
                xBuf #[D] .f32 .Global
      let s ← CircuitM.registerExternal
                (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
                scaleBuf #[D] .f32 .Global
      let _y ← CircuitM.rmsNorm x s eps
      pure ())
    -- Tensor ids assigned by builder:
    --   0 = x (external)        -- full [D]
    --   1 = scale (external)    -- full [D]
    --   2 = sumSq               (reduce → [1])
    --   3 = meanSqPlusEps       (pointwise, fused away)
    --   4 = invRms              (pointwise, fused away)
    --   5 = final output        (pointwise, kept)
    [(0, xBuf), (1, scaleBuf), (5, outBuf)]

  let expected := cpuRmsNorm xArr scaleArr eps
  let outBytes ← GPUBackend.readBuffer (β := β) ctx outBuf (4 * D).toUSize
  let mut okCount := 0
  let mut maxErr : Float := 0.0
  let mut firstErr : Option (Nat × Float × Float) := none
  for i in [0:D] do
    let actual ← Hesper.Basic.bytesToFloat32 outBytes (i * 4)
    let e := expected[i]!
    let err := (actual - e).abs
    if err > maxErr then maxErr := err
    if err < 1e-4 then okCount := okCount + 1
    else if firstErr.isNone then firstErr := some (i, actual, e)

  IO.println s!"Max abs error: {maxErr}"
  if okCount == D then
    IO.println s!"✓ PASS: RMSNorm output matches CPU reference ({D} elements, tol 1e-4)"
  else
    IO.println s!"✗ FAIL: {okCount}/{D} matched"
    match firstErr with
    | some (i, a, e) => IO.println s!"  first mismatch: i={i} actual={a} expected={e}"
    | none => pure ()
    IO.Process.exit 1
