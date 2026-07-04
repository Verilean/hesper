import Hesper
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WebGPU.BufferOps
import Hesper.Basic

open Hesper.WebGPU

/-- metal_replacer STEP 1+2+3 PoC — validate the Dawn→Metal interop end-to-end:
    STEP 1: MTLDevice behind the WGPUDevice.  STEP 2: MTLBuffer behind a Dawn buffer.
    STEP 3: run a hand-written custom Metal kernel (out[i]=in[i]*2) on our Dawn buffers.
    Proves we can dispatch tuned Metal kernels on our data (macOS DEBUG/REFERENCE tool — not shipped).
    See METAL_REPLACER_INTEGRATION.md. -/
def main : IO Unit := do
  IO.println "=== metal_replacer STEP 1+2+3 PoC — Dawn → Metal interop ==="
  let inst ← Hesper.init
  let device ← getDevice inst
  let info ← mtlDeviceName device
  IO.println s!"✅ STEP 1  MTLDevice: {info}"
  let n : Nat := 16
  let inArr : Array Float := #[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0]
  let inB ← createBuffer device { size := (n*4).toUSize, usage := [.storage, .copyDst, .copySrc], mappedAtCreation := false }
  let outB ← createBuffer device { size := (n*4).toUSize, usage := [.storage, .copyDst, .copySrc], mappedAtCreation := false }
  writeBuffer device inB 0 (← Hesper.Basic.floatArrayToBytes inArr)
  -- write outB too: marks it INITIALIZED in Dawn so its lazy-clear doesn't zero the custom kernel's output
  -- on the subsequent mapBufferRead.
  writeBuffer device outB 0 (← Hesper.Basic.floatArrayToBytes (Array.replicate n (0.0:Float)))
  let probe ← mtlBufferProbe inB
  IO.println s!"✅ STEP 2  buffer bridge (expect length=64): {probe}"
  -- sync the Dawn writes before the custom Metal kernel reads `in`; also verify the input round-trips.
  let inCheck ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device inB 0 (n*4).toUSize)
  IO.println s!"   (input readback: {inCheck.getD 0 0.0},{inCheck.getD 1 0.0},{inCheck.getD 2 0.0})"
  metalDispatchMul2 device inB outB n.toUInt32
  let outArr ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device outB 0 (n*4).toUSize)
  let ok := (List.range n).all (fun i => (outArr.getD i 0.0) == (inArr.getD i 0.0) * 2.0)
  let verdict := if ok then "✅ CORRECT" else "❌ WRONG"
  IO.println s!"STEP 3  custom Metal (out=in*2): out[0..3]={outArr.getD 0 0.0},{outArr.getD 1 0.0},{outArr.getD 2 0.0},{outArr.getD 3 0.0} → {verdict}"
  -- STEP 4: the CEILING — Apple's tuned MPS f16 matmul at our forward shapes, vs the WGSL reg.
  IO.println "\n=== STEP 4  MPS f16 matmul CEILING (vs WGSL reg) ==="
  let shapes : List (String × Nat × Nat × Nat) :=
    [("QKV-Q   ", 262, 8192, 2816), ("MoE g/up ", 6208, 1408, 2816), ("MoE down ", 6208, 2816, 704)]
  for (nm, m, nn, k) in shapes do
    let r ← mpsMatmulBench device m.toUInt32 nn.toUInt32 k.toUInt32 50
    IO.println s!"  MPS {nm}[M={m} N={nn} K={k}]: {r}"
  IO.println "  WGSL reg (harness): QKV-Q 1.75ms/44% | MoE g/up 4.75ms/67% | MoE down 2.4ms/66%"
