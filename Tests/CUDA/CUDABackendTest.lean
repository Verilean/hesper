import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.TTT.Kernels

/-!
# Backend Typeclass Test

Runs kernels through the `GPUBackend` typeclass API.
Same code works for both WebGPU and CUDA — only the context type differs.
-/

open Hesper
open Hesper.CUDA
open Hesper.TTT.Kernels
open Hesper.WGSL.Monad (ShaderM)

-- Float helpers
private def f64ToF32Bits (f : Float) : UInt32 :=
  let b := f.toBits; let s := (b >>> 63) &&& 1; let e := (b >>> 52) &&& 0x7FF
  let m := b &&& 0x000FFFFFFFFFFFFF
  if e == 0 then 0
  else let e32 := e.toNat - 1023 + 127
    if e32 <= 0 then 0 else if e32 >= 255 then (s.toUInt32 <<< 31) ||| ((0xFF : UInt32) <<< 23)
    else (s.toUInt32 <<< 31) ||| (e32.toUInt32 <<< 23) ||| ((m >>> 29).toUInt32 &&& (0x7FFFFF : UInt32))

private def f32BitsToF64 (bits : UInt32) : Float :=
  let e := (bits >>> 23) &&& 0xFF; let m := bits &&& (0x7FFFFF : UInt32); let s := bits >>> 31
  if e == 0 then 0.0 else
    let v := (1.0 + m.toNat.toFloat / 8388608.0) * Float.pow 2.0 (e.toNat.toFloat - 127.0)
    if s == 1 then -v else v

private def packFloats (arr : Array Float) : ByteArray :=
  arr.foldl (fun acc f => let b := f64ToF32Bits f
    acc.push b.toUInt8 |>.push (b>>>8).toUInt8 |>.push (b>>>16).toUInt8 |>.push (b>>>24).toUInt8
  ) ByteArray.empty

private def unpackFloat (ba : ByteArray) (i : Nat) : Float :=
  let o := i * 4
  let b0 : UInt32 := ba.get! o |>.toUInt32
  let b1 : UInt32 := ba.get! (o+1) |>.toUInt32
  let b2 : UInt32 := ba.get! (o+2) |>.toUInt32
  let b3 : UInt32 := ba.get! (o+3) |>.toUInt32
  f32BitsToF64 (b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24))

/-- Backend-agnostic kernel test. Same function works for WebGPU or CUDA. -/
def testVecAdd [GPUBackend β] (ctx : β) : IO Bool := do
  let n := 16
  let aBuf ← GPUBackend.allocBuffer ctx (n * 4).toUSize
  let bBuf ← GPUBackend.allocBuffer ctx (n * 4).toUSize
  let outBuf ← GPUBackend.allocBuffer ctx (n * 4).toUSize

  let aArr := Array.range n |>.map (fun i => (i + 1).toFloat)
  let bArr := Array.range n |>.map (fun i => ((i + 1) * 100).toFloat)
  GPUBackend.writeBuffer ctx aBuf (packFloats aArr)
  GPUBackend.writeBuffer ctx bBuf (packFloats bArr)

  GPUBackend.execute ctx (vecAddKernel n)
    [("a", aBuf), ("b", bBuf), ("output", outBuf)]
    (ExecConfig.dispatch1D n)

  let result ← GPUBackend.readBuffer ctx outBuf (n * 4).toUSize
  let mut ok := true
  for i in [0, 7, 15] do
    let got := unpackFloat result i
    let exp := (i + 1).toFloat + ((i + 1) * 100).toFloat
    if (got - exp).abs > 0.5 then ok := false
    IO.println s!"    out[{i}] = {got} (expect {exp})"

  GPUBackend.freeBuffer ctx aBuf
  GPUBackend.freeBuffer ctx bBuf
  GPUBackend.freeBuffer ctx outBuf
  return ok

/-- Backend-agnostic matVec test. -/
def testMatVec [GPUBackend β] (ctx : β) : IO Bool := do
  let outDim := 4; let inDim := 4
  let wBuf ← GPUBackend.allocBuffer ctx (outDim * inDim * 4).toUSize
  let xBuf ← GPUBackend.allocBuffer ctx (inDim * 4).toUSize
  let yBuf ← GPUBackend.allocBuffer ctx (outDim * 4).toUSize

  GPUBackend.writeBuffer ctx wBuf (packFloats #[1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16])
  GPUBackend.writeBuffer ctx xBuf (packFloats #[1,1,1,1])

  GPUBackend.execute ctx (matVecKernel outDim inDim)
    [("weight", wBuf), ("input", xBuf), ("output", yBuf)]
    (ExecConfig.dispatch1D outDim)

  let result ← GPUBackend.readBuffer ctx yBuf (outDim * 4).toUSize
  let expected : Array Float := #[10, 26, 42, 58]
  let mut ok := true
  for i in List.range outDim do
    let got := unpackFloat result i
    let exp := expected.getD i 0.0
    if (got - exp).abs > 0.01 then ok := false
    IO.println s!"    y[{i}] = {got} (expect {exp})"

  GPUBackend.freeBuffer ctx wBuf
  GPUBackend.freeBuffer ctx xBuf
  GPUBackend.freeBuffer ctx yBuf
  return ok

def main : IO Unit := do
  IO.println "═══ Backend Typeclass Test ═══"

  let backend ← detectBackend
  IO.println s!"Backend: {repr backend}"

  match backend with
  | .CUDA => do
    let ctx ← CUDAContext.init

    IO.println "\nTest A: vecAdd via GPUBackend typeclass"
    let okA ← testVecAdd ctx
    IO.println s!"  → {if okA then "PASSED" else "FAILED"}"

    IO.println "\nTest B: matVec via GPUBackend typeclass"
    let okB ← testMatVec ctx
    IO.println s!"  → {if okB then "PASSED" else "FAILED"}"

    if okA && okB then
      IO.println "\n✓ BACKEND TYPECLASS TESTS PASSED (CUDA)"
    else
      IO.println "\n✗ FAILED"
      IO.Process.exit 1

  | .WebGPU => do
    IO.println "WebGPU backend — skipping (set HESPER_BACKEND=cuda to test CUDA)"
