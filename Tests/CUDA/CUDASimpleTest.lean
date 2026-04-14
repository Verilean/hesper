import Hesper.CUDA.FFI
import Hesper.CUDA.Buffer
import Hesper.CUDA.CodeGen
import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.WGSL.Monad
import Hesper.WGSL.Exp

set_option maxRecDepth 2048

open Hesper
open Hesper.CUDA
open Hesper.WGSL
open Hesper.WGSL.Monad (ShaderM)
open Hesper.WGSL.Monad.ShaderM

-- Simple kernel: output[tid] = (u32)(input[tid] * 10.0)
def simpleKernel : ShaderM Unit := do
  let lid ← localId
  let tid := Exp.vec3X lid
  let _input ← declareInputBuffer "input" (.array (.scalar .f32) 32)
  let _output ← declareOutputBuffer "output" (.array (.scalar .u32) 32)
  let x ← readBuffer (ty := .scalar .f32) (n := 32) "input" tid
  let mul10 := Exp.mul x (Exp.litF32 10.0)
  let asU32 := Exp.toU32 mul10
  writeBuffer (ty := .scalar .u32) "output" tid asU32

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
      -- Round-to-nearest-even: look at bits [28:0] of 52-bit mantissa.
      -- If > 0x10000000 (halfway bit + any below) → round up.
      -- If = 0x10000000 and result LSB is 1 → round up (tie to even).
      let lower29 := m64 &&& (0x1FFFFFFF : UInt64)
      let m32Truncated := (m64 >>> 29).toUInt32 &&& (0x7FFFFF : UInt32)
      let halfway : UInt64 := 0x10000000
      let roundUp :=
        lower29 > halfway ||
        (lower29 == halfway && (m32Truncated &&& 1) == 1)
      let m32 := if roundUp then m32Truncated + 1 else m32Truncated
      -- Handle mantissa overflow (m32 = 0x800000) → increment exponent
      let (m32Final, e32i') := if m32 == 0x800000 then (0, e32i + 1) else (m32, e32i)
      if e32i' ≥ 255 then (s.toUInt32 <<< 31) ||| ((0xFF : UInt32) <<< 23)
      else
        let e32 : UInt32 := e32i'.toNat.toUInt32
        (s.toUInt32 <<< 31) ||| (e32 <<< 23) ||| m32Final

private def packF32 (arr : Array Float) : ByteArray :=
  arr.foldl (init := ByteArray.empty) fun acc f => let b := f64ToF32Bits f
    acc.push b.toUInt8 |>.push (b>>>8).toUInt8 |>.push (b>>>16).toUInt8 |>.push (b>>>24).toUInt8

def main : IO Unit := do
  IO.println "═══ Simple mul_toU32 Test ═══"
  let ptx := Hesper.CUDA.CodeGen.generatePTX "main" { x := 32, y := 1, z := 1 } simpleKernel
  IO.println ptx
  IO.println "─── end PTX ───\n"
  let cuda ← CUDAContext.init
  let input : Array Float := (Array.range 32).map (fun i => i.toFloat * 0.1)
  let inputBytes := packF32 input
  let inputBuf ← GPUBackend.allocBuffer cuda 128
  GPUBackend.writeBuffer cuda inputBuf inputBytes
  let outBuf ← GPUBackend.allocBuffer cuda 128
  GPUBackend.execute cuda simpleKernel [("input", inputBuf), ("output", outBuf)]
    { numWorkgroups := (1,1,1), workgroupSize := { x := 32, y := 1, z := 1 } }
  let out ← GPUBackend.readBuffer cuda outBuf 128
  IO.print "Output[0..15]: "
  for i in [0:16] do
    let o := i * 4
    let b0 : UInt32 := (out.get! o).toUInt32
    let b1 : UInt32 := (out.get! (o+1)).toUInt32
    let b2 : UInt32 := (out.get! (o+2)).toUInt32
    let b3 : UInt32 := (out.get! (o+3)).toUInt32
    let v := b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24)
    IO.print s!"{v} "
  IO.println ""
  IO.println "Expected: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]"
