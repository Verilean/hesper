import Hesper.Backend.CUDA
import Hesper.Circuit.IR
import Hesper.Circuit.Lowering
import Hesper.Circuit.Passes
import Hesper.WebGPU.BufferOps
import Hesper.Basic

/-!
End-to-end GPU test for `Prim.reduceScatterEpilogue` (Level 3,
block-cooperative reduce + dynamic-address scatter).

Single workgroup, 64 threads cooperating:
  D = 64
  Phase 1: every thread loads x[lane], smem-tree-reduces to total = Σ x[lane]
  Phase 2: every lane writes
             dst[64 + lane] = (x[lane] / total) * 100.0
           (i.e. each lane writes its normalised value × 100 to a
            slot 64 elements into a 256-element destination buffer)

Expected:
  total = Σ_{i=0..63} (i + 1) = 64 * 65 / 2 = 2080
  dst[64 + i] = ((i + 1) / 2080) * 100
  dst[0..63] and dst[128..255] remain 0

This proves the DSL can express:
  - block-cooperative reduction (uses shared memory + barriers)
  - epilogue reading the reduced scalar broadcast
  - per-lane dynamic addr writing into an existing buffer
in a single dispatch.
-/

open Hesper
open Hesper.Circuit

abbrev β := Hesper.CUDAContext

unsafe def main : IO Unit := do
  let D : Nat := 64
  let dstSize : Nat := 256
  let writeOffset : Nat := 64
  let total : Float := (D * (D + 1) / 2).toFloat   -- 2080

  let ctx ← Hesper.CUDAContext.init

  -- Source: x[i] = i + 1
  let xArr : Array Float := Array.ofFn (n := D) fun i => (i.val + 1).toFloat
  let xBytes ← Hesper.WebGPU.BufferOps.floatArrayToBytes xArr
  let xBuf ← GPUBackend.allocBuffer (β := β) ctx (4 * D).toUSize
  GPUBackend.writeBuffer (β := β) ctx xBuf xBytes

  -- Pre-fill dst with zeros.
  let zeros : Array Float := Array.ofFn (n := dstSize) fun _ => 0.0
  let zeroBytes ← Hesper.WebGPU.BufferOps.floatArrayToBytes zeros
  let dstBuf ← GPUBackend.allocBuffer (β := β) ctx (4 * dstSize).toUSize
  GPUBackend.writeBuffer (β := β) ctx dstBuf zeroBytes

  let ccRef : IO.Ref (Option (CompiledCircuit β)) ← IO.mkRef none
  Hesper.Circuit.runCachedFused ctx ccRef
    (do
      let x ← CircuitM.registerExternal
                (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
                xBuf #[D] .f32 .Global
      let dst ← CircuitM.registerExternal
                  (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
                  dstBuf #[dstSize] .f32 .Global
      -- valueExpr = (x[lane] / total) * 100
      --   .input 0 = scalar reduction (= total)
      --   .input 1 = x[lane] (the epilogue input)
      let valueExpr : ScalarExp :=
        .mul (.div (.input 1) (.input 0)) (.const 100.0)
      let addrExpr : ScalarExp :=
        .add .laneIdx (.const writeOffset.toFloat)
      let _ ← CircuitM.reduceScatterEpilogue
                .sum x #[x] dst valueExpr addrExpr
      pure ())
    [(0, xBuf), (1, dstBuf)]

  IO.println "Waiting for GPU..."

  let outBytes ← GPUBackend.readBuffer (β := β) ctx dstBuf (4 * dstSize).toUSize
  let mut ok := 0
  let mut firstErr : Option (Nat × Float × Float) := none
  for i in [0:dstSize] do
    let actual ← Hesper.Basic.bytesToFloat32 outBytes (i * 4)
    let expected : Float :=
      if i >= writeOffset && i < writeOffset + D then
        let lane := i - writeOffset
        ((lane + 1).toFloat / total) * 100.0
      else
        0.0
    let diff := (actual - expected).abs
    if diff < 1e-3 then
      ok := ok + 1
    else if firstErr.isNone then firstErr := some (i, actual, expected)

  if ok == dstSize then
    IO.println s!"✓ PASS: reduce(sum)={total}, then per-lane (x[i]/total)*100 scattered to dst[{writeOffset}..{writeOffset+D}]"
  else
    IO.println s!"✗ FAIL: {ok}/{dstSize} cells matched"
    match firstErr with
    | some (i, a, e) => IO.println s!"  first mismatch: dst[{i}] = {a}, expected {e}"
    | none => pure ()
    IO.Process.exit 1
