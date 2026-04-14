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

private def toHex (v : UInt32) : String :=
  let hex := "0123456789ABCDEF"
  let h i := (hex.get ⟨Nat.min i.toNat 15⟩).toString
  h ((v >>> 28) &&& 0xF) ++ h ((v >>> 24) &&& 0xF) ++
  h ((v >>> 20) &&& 0xF) ++ h ((v >>> 16) &&& 0xF) ++
  h ((v >>> 12) &&& 0xF) ++ h ((v >>> 8) &&& 0xF) ++
  h ((v >>> 4) &&& 0xF) ++ h (v &&& 0xF)

-- Proper IEEE 754 binary64 → binary32 conversion with round-to-nearest-even.
private def f64ToF32Bits (f : Float) : UInt32 :=
  let b := f.toBits
  let s := (b >>> 63) &&& 1
  let e64 := (b >>> 52) &&& 0x7FF
  let m64 := b &&& 0x000FFFFFFFFFFFFF
  if e64 == 0 then 0
  else
    let eUnb : Int := Int.ofNat e64.toNat - 1023
    let e32i : Int := eUnb + 127
    if e32i ≤ 0 then 0
    else if e32i ≥ 255 then (s.toUInt32 <<< 31) ||| ((0xFF : UInt32) <<< 23)
    else
      let lower29 := m64 &&& (0x1FFFFFFF : UInt64)
      let m32Truncated := (m64 >>> 29).toUInt32 &&& (0x7FFFFF : UInt32)
      let halfway : UInt64 := 0x10000000
      let roundUp :=
        lower29 > halfway ||
        (lower29 == halfway && (m32Truncated &&& 1) == 1)
      let m32 := if roundUp then m32Truncated + 1 else m32Truncated
      let (m32Final, e32i') := if m32 == 0x800000 then (0, e32i + 1) else (m32, e32i)
      if e32i' ≥ 255 then (s.toUInt32 <<< 31) ||| ((0xFF : UInt32) <<< 23)
      else
        let e32 : UInt32 := e32i'.toNat.toUInt32
        (s.toUInt32 <<< 31) ||| (e32 <<< 23) ||| m32Final

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
  IO.print s!"Input bytes [0..11]: "
  for i in [0:12] do
    IO.print s!"{inputBytes.get! i} "
  IO.println ""
  let inputBuf ← GPUBackend.allocBuffer cuda (4 * inDim).toUSize
  GPUBackend.writeBuffer cuda inputBuf inputBytes

  -- Read back to verify what's on GPU
  let roundtrip ← GPUBackend.readBuffer cuda inputBuf (4 * inDim).toUSize
  IO.print s!"Roundtrip bytes [0..11]: "
  for i in [0:12] do
    IO.print s!"{roundtrip.get! i} "
  IO.println ""
  let rt1_bits : UInt32 :=
    (roundtrip.get! 4).toUInt32 ||| ((roundtrip.get! 5).toUInt32 <<< 8) |||
    ((roundtrip.get! 6).toUInt32 <<< 16) ||| ((roundtrip.get! 7).toUInt32 <<< 24)
  IO.println s!"Roundtrip input[1] bits = 0x{toHex rt1_bits} = {f32BitsToF64 rt1_bits}"

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
