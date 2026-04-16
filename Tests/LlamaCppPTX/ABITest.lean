import Hesper.Backend.CUDA
import Hesper.Backend.LlamaCppPTX
import Hesper.CUDA.FFI
import Hesper.CUDA.Buffer
import Hesper.Basic
import Hesper.WebGPU.BufferOps

open Hesper
open Hesper.CUDA
open Hesper.LlamaCppPTX

unsafe def main : IO Unit := do
  let ctx ← CUDAContext.init
  let k ← loadKernels

  let D : Nat := 256
  let xArr : Array Float := Array.ofFn (n := D) fun i =>
    (i.val.toFloat - 128.0) / 128.0
  let xBytes ← Hesper.WebGPU.BufferOps.floatArrayToBytes xArr
  let xBuf ← createCUDABuffer (4 * D).toUSize
  writeCUDABuffer xBuf xBytes

  let nQ8Blocks := D / 32
  let q8Size := nQ8Blocks * 36
  let q8Buf ← createCUDABuffer q8Size.toUSize
  cuMemset q8Buf.ptr q8Size.toUSize

  IO.println s!"[Test 1] llama.cpp quantize_q8_1 on {D} floats..."
  launchQuantizeQ8_1 k xBuf.ptr q8Buf.ptr D

  let q8Bytes ← readCUDABuffer q8Buf q8Size.toUSize
  IO.println s!"  Q8_1 output: {q8Size} bytes"

  -- Block 0 header: half2 ds = (d, s) packed as 4 bytes
  for blk in [0:nQ8Blocks] do
    let off := blk * 36
    let b0 := q8Bytes.get! off
    let b1 := q8Bytes.get! (off + 1)
    let b2 := q8Bytes.get! (off + 2)
    let b3 := q8Bytes.get! (off + 3)
    IO.println s!"  Block {blk}: ds_bytes=[{b0},{b1},{b2},{b3}]"
    if blk == 0 then
      IO.print "    qs[0..31]: "
      for i in [0:32] do
        let b := q8Bytes.get! (off + 4 + i)
        let v : Int := if b.toNat >= 128 then (b.toNat : Int) - 256 else b.toNat
        IO.print s!"{v} "
      IO.println ""

  -- CPU ref for block 0: x[0..31] range [-1.0, ..., -0.7578125]
  -- amax = 1.0, d = 1/127 ≈ 0.007874
  IO.println s!"  CPU: d_f16(1/127) ≈ 0x1A03 or similar"

  -- Test 2: Zero-weight Q4_K matmul
  let inDim : Nat := 256
  let outDim : Nat := 4
  let bytesPerBlock := 144
  let weightBytes := outDim * (inDim / 256) * bytesPerBlock

  let mut wArr : ByteArray := ByteArray.empty
  for _ in [0:weightBytes] do
    wArr := wArr.push 0

  let wBuf ← createCUDABuffer weightBytes.toUSize
  writeCUDABuffer wBuf wArr

  let outBuf ← createCUDABuffer (outDim * 4).toUSize
  cuMemset outBuf.ptr (outDim * 4).toUSize

  IO.println s!"\n[Test 2] mul_mat_vec_q<Q4_K> with zero weights..."
  let q8Buf2 ← createCUDABuffer q8Size.toUSize
  cuMemset q8Buf2.ptr q8Size.toUSize
  launchQuantizeQ8_1 k xBuf.ptr q8Buf2.ptr inDim
  launchMulMatVecQ4K k wBuf.ptr q8Buf2.ptr outBuf.ptr inDim outDim

  let outBytes ← readCUDABuffer outBuf (outDim * 4).toUSize
  for i in [0:outDim] do
    let v ← Hesper.Basic.bytesToFloat32 outBytes (i * 4)
    IO.println s!"  out[{i}] = {v}  (expected 0.0)"

  -- Test 3: d=1.0 f16, scale[0]=1, qs=0x88, rest zero
  let mut w3 : ByteArray := ByteArray.empty
  w3 := w3.push 0x00; w3 := w3.push 0x3C  -- d=1.0 f16
  w3 := w3.push 0x00; w3 := w3.push 0x00  -- dmin=0.0
  w3 := w3.push 1  -- scales[0] = 1
  for _ in [1:12] do w3 := w3.push 0
  for _ in [0:128] do w3 := w3.push 0x88

  let mut wArr3 : ByteArray := ByteArray.empty
  for _ in [0:outDim] do
    wArr3 := wArr3 ++ w3

  writeCUDABuffer wBuf wArr3
  cuMemset outBuf.ptr (outDim * 4).toUSize
  launchQuantizeQ8_1 k xBuf.ptr q8Buf2.ptr inDim
  launchMulMatVecQ4K k wBuf.ptr q8Buf2.ptr outBuf.ptr inDim outDim

  let outBytes3 ← readCUDABuffer outBuf (outDim * 4).toUSize
  IO.println "\n[Test 3] d=1.0, scale[0]=1, qs=0x88:"
  for i in [0:outDim] do
    let v ← Hesper.Basic.bytesToFloat32 outBytes3 (i * 4)
    IO.println s!"  out[{i}] = {v}"

  IO.println "\nDone."
