import Hesper.CUDA.FFI
import Hesper.CUDA.Buffer
import Hesper.CUDA.CodeGen
import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.Layers.Linear

set_option maxRecDepth 2048

open Hesper
open Hesper.CUDA
open Hesper.Layers.Linear

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

private def packF32 (arr : Array Float) : ByteArray :=
  arr.foldl (fun acc f => let b := f64ToF32Bits f
    acc.push b.toUInt8 |>.push (b>>>8).toUInt8 |>.push (b>>>16).toUInt8 |>.push (b>>>24).toUInt8
  ) ByteArray.empty

def main : IO Unit := do
  IO.println "═══ Q8_1 Quantize Standalone Test ═══\n"

  -- Dump PTX first to catch codegen issues
  let inDim := 32
  let ptx := Hesper.CUDA.CodeGen.generatePTX "main" { x := 32, y := 1, z := 1 } (quantizeQ8_1Kernel inDim)
  IO.println "═══ Generated PTX ═══"
  IO.println ptx
  IO.println "═══ End PTX ═══\n"

  let cuda ← CUDAContext.init
  let input : Array Float := (Array.range inDim).map (fun i => i.toFloat * 0.1)
  IO.println s!"Input[0..7] = {input[0]!} {input[1]!} {input[2]!} {input[3]!} {input[4]!} {input[5]!} {input[6]!} {input[7]!}"
  IO.println s!"Expected: max|x| = {input[inDim-1]!} ≈ 3.1, d = 3.1/127 ≈ 0.0244"

  let inputBytes := packF32 input
  let inputBuf ← GPUBackend.allocBuffer cuda (4 * inDim).toUSize
  GPUBackend.writeBuffer cuda inputBuf inputBytes

  let nBlocks := inDim / 32
  let q8BufSize : USize := (nBlocks * 9 * 4).toUSize
  let q8Buf ← GPUBackend.allocBuffer cuda q8BufSize

  IO.println "Running quantizeQ8_1Kernel..."
  GPUBackend.execute cuda (quantizeQ8_1Kernel inDim)
    [("input", inputBuf), ("output", q8Buf)]
    { numWorkgroups := (nBlocks, 1, 1), workgroupSize := { x := 32, y := 1, z := 1 }
      extensions := ["subgroups"] }
  IO.println "Done!"

  let q8Data ← GPUBackend.readBuffer cuda q8Buf q8BufSize
  IO.print "Q8_1 block[0] u32s: "
  for i in [0:9] do
    let o := i * 4
    let b0 : UInt32 := q8Data.get! o |>.toUInt32
    let b1 : UInt32 := q8Data.get! (o+1) |>.toUInt32
    let b2 : UInt32 := q8Data.get! (o+2) |>.toUInt32
    let b3 : UInt32 := q8Data.get! (o+3) |>.toUInt32
    let v := b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24)
    IO.print s!"{v} "
  IO.println ""

  let hdrBits : UInt32 :=
    (q8Data.get! 0).toUInt32 ||| ((q8Data.get! 1).toUInt32 <<< 8) |||
    ((q8Data.get! 2).toUInt32 <<< 16) ||| ((q8Data.get! 3).toUInt32 <<< 24)
  let dVal := f32BitsToF64 hdrBits
  IO.println s!"d (header as f32) = {dVal}"

  -- Expected int8 values: q[i] = round(input[i] / d)
  -- q[0] = 0, q[31] ≈ 127
  IO.print "Quants[0..7]: "
  for i in [0:8] do
    let byteOff := 4 + i  -- skip header (4 bytes)
    let b := q8Data.get! byteOff
    let signed : Int := if b ≥ 128 then b.toNat - 256 else b.toNat
    IO.print s!"{signed} "
  IO.println ""

  GPUBackend.freeBuffer cuda inputBuf
  GPUBackend.freeBuffer cuda q8Buf
