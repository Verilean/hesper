import Hesper.Backend.CUDA
import Hesper.Circuit.IR
import Hesper.Circuit.Lowering
import Hesper.Circuit.Passes
import Hesper.WebGPU.BufferOps
import Hesper.Basic

/-!
End-to-end GPU test for a warp-level dot-product expressed purely in
`ScalarExp`:

  partial[lane] = a[lane] * b[lane]
  total = warpSum(partial)

Every lane's output equals `Σ_{i=0..31} a[i] * b[i]`.  This is the core
pattern llama.cpp's `warp_reduce_sum` implements — here we express it
as a single ScalarExp body inside a `scatter`.

Source values: a[i] = i.toFloat, b[i] = (i+1).toFloat,
                expected = Σ i*(i+1) for i=0..31 = 10912.
-/

open Hesper
open Hesper.Circuit

abbrev β := Hesper.CUDAContext

unsafe def main : IO Unit := do
  let n : Nat := 32
  -- Σ i*(i+1) for i=0..31 = Σ (i² + i) = Σi² + Σi
  --   Σi = 31*32/2 = 496
  --   Σi² = 31*32*63/6 = 10416
  --   total = 10912
  let expected : Float := 10912.0

  let ctx ← Hesper.CUDAContext.init

  let aArr : Array Float := Array.ofFn (n := n) fun i => i.val.toFloat
  let bArr : Array Float := Array.ofFn (n := n) fun i => (i.val + 1).toFloat
  let aBytes ← Hesper.WebGPU.BufferOps.floatArrayToBytes aArr
  let bBytes ← Hesper.WebGPU.BufferOps.floatArrayToBytes bArr

  let aBuf ← GPUBackend.allocBuffer (β := β) ctx (4 * n).toUSize
  let bBuf ← GPUBackend.allocBuffer (β := β) ctx (4 * n).toUSize
  GPUBackend.writeBuffer (β := β) ctx aBuf aBytes
  GPUBackend.writeBuffer (β := β) ctx bBuf bBytes

  let zeros : Array Float := Array.ofFn (n := n) fun _ => 0.0
  let zBytes ← Hesper.WebGPU.BufferOps.floatArrayToBytes zeros
  let outBuf ← GPUBackend.allocBuffer (β := β) ctx (4 * n).toUSize
  GPUBackend.writeBuffer (β := β) ctx outBuf zBytes

  let ccRef : IO.Ref (Option (CompiledCircuit β)) ← IO.mkRef none
  Hesper.Circuit.runCachedFused ctx ccRef
    (do
      let a ← CircuitM.registerExternal
                (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
                aBuf #[n] .f32 .Global
      let b ← CircuitM.registerExternal
                (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
                bBuf #[n] .f32 .Global
      -- body: warpSum(a[lane] * b[lane])
      -- Every lane sees the same total (warpSum broadcasts the result).
      let _final ← CircuitM.zip2 a b
                     (.warpSum (.mul (.input 0) (.input 1)))
      pure ())
    [(0, aBuf), (1, bBuf), (2, outBuf)]

  IO.println "Waiting for GPU..."

  let outBytes ← GPUBackend.readBuffer (β := β) ctx outBuf (4 * n).toUSize
  let mut ok := 0
  let mut firstErr : Option (Nat × Float) := none
  for i in [0:n] do
    let actual ← Hesper.Basic.bytesToFloat32 outBytes (i * 4)
    if (actual - expected).abs < 1e-2 then
      ok := ok + 1
    else if firstErr.isNone then firstErr := some (i, actual)

  if ok == n then
    IO.println s!"✓ PASS: warp-level dot product = {expected}, all {n} lanes agree"
  else
    IO.println s!"✗ FAIL: {ok}/{n} lanes matched (expected {expected})"
    match firstErr with
    | some (i, a) => IO.println s!"  first mismatch: lane {i} saw {a}"
    | none => pure ()
    IO.Process.exit 1
