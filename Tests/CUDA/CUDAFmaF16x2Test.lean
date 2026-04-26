import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.CUDA.FFI
import Hesper.CUDA.CodeGen
import Hesper.WGSL.Monad
import Hesper.WGSL.Exp
import Hesper.WGSL.Shader
import Hesper

/-!
# Unit test for fma.rn.f16x2 primitive

Verifies:
1. The PTX dump contains `fma.rn.f16x2` (not two scalar half fmas).
2. The kernel computes `c = a*b + c` correctly with packed half2 inputs.

Layout:
  4 input u32 elements per buffer.
  Each u32 holds two f16 values (low half = lane 0, high = lane 1).
  Kernel: dst[i] = fmaF16x2(a[i], b[i], c[i]) for each thread i in [0,4).
-/

open Hesper
open Hesper.WGSL.Monad (ShaderM)
open Hesper.WGSL (Exp)

private def f64ToF16Bits (f : Float) : UInt16 :=
  -- Reuse the f64→f32→f16 path
  let b := f.toBits
  let s := (b >>> 63) &&& 1
  let e := (b >>> 52) &&& 0x7FF
  let m := b &&& 0x000FFFFFFFFFFFFF
  if e == 0 then 0
  else
    let e16 : Int := e.toNat - 1023 + 15
    if e16 <= 0 then 0
    else if e16 >= 31 then
      (s.toUInt16 <<< 15) ||| (0x7C00 : UInt16)
    else
      let m16 : UInt32 := (m >>> 42).toUInt32 &&& (0x3FF : UInt32)
      (s.toUInt16 <<< 15) ||| (e16.toNat.toUInt16 <<< 10) ||| m16.toUInt16

private def f16BitsToF32 (h : UInt16) : Float :=
  let s := (h.toUInt32 >>> 15) &&& 1
  let e := (h.toUInt32 >>> 10) &&& 0x1F
  let m := h.toUInt32 &&& 0x3FF
  if e == 0 then 0.0
  else if e == 31 then
    if s == 0 then 1.0 / 0.0 else -1.0 / 0.0
  else
    let fv := (1.0 + m.toNat.toFloat / 1024.0) * Float.pow 2.0 (e.toNat.toFloat - 15.0)
    if s == 1 then -fv else fv

/-- Pack two f16 values (as f32 input) into one u32. -/
private def packPairToU32 (a b : Float) : UInt32 :=
  let aH := f64ToF16Bits a
  let bH := f64ToF16Bits b
  aH.toUInt32 ||| (bH.toUInt32 <<< 16)

/-- Unpack one u32 into two f16-as-f32 values. -/
private def unpackU32ToPair (x : UInt32) : Float × Float :=
  let lo : UInt16 := (x &&& 0xFFFF).toUInt16
  let hi : UInt16 := ((x >>> 16) &&& 0xFFFF).toUInt16
  (f16BitsToF32 lo, f16BitsToF32 hi)

private def packU32 (arr : Array UInt32) : ByteArray :=
  arr.foldl (init := ByteArray.empty) (fun acc (x : UInt32) =>
    acc.push x.toUInt8
       |>.push (x >>> 8).toUInt8
       |>.push (x >>> 16).toUInt8
       |>.push (x >>> 24).toUInt8)

private def unpackU32 (ba : ByteArray) (n : Nat) : Array UInt32 := Id.run do
  let mut arr := #[]
  for i in [0:n] do
    let o := i * 4
    let b0 := ba.get! o |>.toUInt32
    let b1 := ba.get! (o+1) |>.toUInt32
    let b2 := ba.get! (o+2) |>.toUInt32
    let b3 := ba.get! (o+3) |>.toUInt32
    arr := arr.push (b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24))
  return arr

/-- The kernel: thread i reads a[i], b[i], c[i] (all u32 packed half2),
    computes fmaF16x2 and writes result to out[i].  4 threads. -/
def fmaF16x2Kernel : ShaderM Unit := do
  let lid ← ShaderM.localId
  let tid := Exp.vec3X lid
  let _a   ← ShaderM.declareInputBuffer "a"   (.array (.scalar .u32) 4)
  let _b   ← ShaderM.declareInputBuffer "b"   (.array (.scalar .u32) 4)
  let _c   ← ShaderM.declareInputBuffer "c"   (.array (.scalar .u32) 4)
  let _out ← ShaderM.declareOutputBuffer "out" (.array (.scalar .u32) 4)
  let aV ← ShaderM.readBuffer (ty := .scalar .u32) (n := 4) "a" tid
  let bV ← ShaderM.readBuffer (ty := .scalar .u32) (n := 4) "b" tid
  let cV ← ShaderM.readBuffer (ty := .scalar .u32) (n := 4) "c" tid
  let r := Exp.fmaF16x2 aV bV cV
  ShaderM.writeBuffer (ty := .scalar .u32) "out" tid r

unsafe def main : IO Unit := do
  IO.println "═══ fma.rn.f16x2 primitive test ═══"

  -- 1. PTX dump check
  let ptx := Hesper.CUDA.CodeGen.generatePTX "fma_f16x2_test"
               { x := 4, y := 1, z := 1 } fmaF16x2Kernel
  IO.FS.writeFile "/tmp/fma_f16x2.ptx" ptx
  IO.println s!"  (PTX written to /tmp/fma_f16x2.ptx, {ptx.length} bytes)"
  let ptxParts := ptx.splitOn "fma.rn.f16x2"
  if ptxParts.length < 2 then
    IO.println "✗ PTX does not contain `fma.rn.f16x2` instruction"
    IO.Process.exit 1
  else
    IO.println "✓ PTX contains `fma.rn.f16x2`"

  -- 2. Numeric check
  let ctx ← Hesper.CUDAContext.init
  -- Inputs: 4 lanes × 2 packed halves
  -- a = [(1.0,2.0), (3.0,4.0), (0.5,-1.5), (-2.0, 0.25)]
  -- b = [(2.0,3.0), (1.0,2.0), (4.0, 2.0), (-3.0, 8.0)]
  -- c = [(0.5,1.5), (-1.0,1.0), (1.0, 0.0), (10.0, -1.0)]
  let aPairs : Array (Float × Float) :=
    #[(1.0, 2.0), (3.0, 4.0), (0.5, -1.5), (-2.0, 0.25)]
  let bPairs : Array (Float × Float) :=
    #[(2.0, 3.0), (1.0, 2.0), (4.0, 2.0), (-3.0, 8.0)]
  let cPairs : Array (Float × Float) :=
    #[(0.5, 1.5), (-1.0, 1.0), (1.0, 0.0), (10.0, -1.0)]

  let aPacked := aPairs.map (fun (x, y) => packPairToU32 x y)
  let bPacked := bPairs.map (fun (x, y) => packPairToU32 x y)
  let cPacked := cPairs.map (fun (x, y) => packPairToU32 x y)

  let aBuf ← GPUBackend.allocBuffer ctx 16
  let bBuf ← GPUBackend.allocBuffer ctx 16
  let cBuf ← GPUBackend.allocBuffer ctx 16
  let oBuf ← GPUBackend.allocBuffer ctx 16
  GPUBackend.writeBuffer ctx aBuf (packU32 aPacked)
  GPUBackend.writeBuffer ctx bBuf (packU32 bPacked)
  GPUBackend.writeBuffer ctx cBuf (packU32 cPacked)

  let bufs : List (String × GPUBackend.Buf Hesper.CUDAContext) :=
    [ ("a", aBuf), ("b", bBuf), ("c", cBuf), ("out", oBuf) ]
  GPUBackend.execute ctx fmaF16x2Kernel bufs
    { workgroupSize := { x := 4 }, numWorkgroups := (1, 1, 1) }

  let resultBytes ← GPUBackend.readBuffer ctx oBuf 16
  let resultU32 := unpackU32 resultBytes 4

  let mut allOk := true
  for i in [0:4] do
    let (al, ah) := aPairs[i]!
    let (bl, bh) := bPairs[i]!
    let (cl, ch) := cPairs[i]!
    let expectedL := al * bl + cl
    let expectedH := ah * bh + ch
    let (gotL, gotH) := unpackU32ToPair resultU32[i]!
    let dL := (expectedL - gotL).abs
    let dH := (expectedH - gotH).abs
    -- f16 has ~3 decimal digits of precision; 1e-2 is generous but
    -- fine for these small-magnitude values.
    let ok := (dL < 1e-2) && (dH < 1e-2)
    if !ok then allOk := false
    let mark := if ok then "✓" else "✗"
    IO.println s!"  {mark} lane {i}: a*b+c = ({al}*{bl}+{cl}, {ah}*{bh}+{ch}) = ({expectedL}, {expectedH})  got ({gotL}, {gotH})  err ({dL}, {dH})"

  GPUBackend.freeBuffer ctx aBuf
  GPUBackend.freeBuffer ctx bBuf
  GPUBackend.freeBuffer ctx cBuf
  GPUBackend.freeBuffer ctx oBuf

  if allOk then
    IO.println "═══ ALL FMA F16X2 CASES PASS ═══"
  else
    IO.println "═══ FMA F16X2 TEST FAILED ═══"
    IO.Process.exit 1
