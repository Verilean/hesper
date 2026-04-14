import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.WGSL.Monad
import Hesper.WGSL.Exp
import Hesper.WGSL.Shader

/-!
# PTX Instruction-Level Tests

Tests every PTX code path with focus on rarely-used instructions and known bugs.
-/

open Hesper
open Hesper.CUDA
open Hesper.WGSL.Monad (ShaderM)
open Hesper.WGSL.Monad.ShaderM
open Hesper.WGSL (Exp)

-- ═══ Helpers ═══

private def f64ToF32Bits (f : Float) : UInt32 :=
  let b := f.toBits; let s := (b >>> 63) &&& 1; let e := (b >>> 52) &&& 0x7FF
  let m := b &&& 0x000FFFFFFFFFFFFF
  if e == 0 then 0
  else let e32 := e.toNat - 1023 + 127
    if e32 <= 0 then 0 else if e32 >= 255 then (s.toUInt32 <<< 31) ||| ((0xFF : UInt32) <<< 23)
    else (s.toUInt32 <<< 31) ||| (e32.toUInt32 <<< 23) ||| ((m >>> 29).toUInt32 &&& (0x7FFFFF : UInt32))

private def pf (arr : Array Float) : ByteArray :=
  arr.foldl (fun acc f => let b := f64ToF32Bits f
    acc.push b.toUInt8 |>.push (b>>>8).toUInt8 |>.push (b>>>16).toUInt8 |>.push (b>>>24).toUInt8
  ) ByteArray.empty

private def pu (arr : Array Nat) : ByteArray :=
  arr.foldl (init := ByteArray.empty) fun acc v =>
    acc.push (v % 256).toUInt8 |>.push ((v/256) % 256).toUInt8
      |>.push ((v/65536) % 256).toUInt8 |>.push ((v/16777216) % 256).toUInt8

private def uf (ba : ByteArray) (i : Nat) : Float :=
  let o := i * 4
  let b0 : UInt32 := ba.get! o |>.toUInt32
  let b1 : UInt32 := ba.get! (o+1) |>.toUInt32
  let b2 : UInt32 := ba.get! (o+2) |>.toUInt32
  let b3 : UInt32 := ba.get! (o+3) |>.toUInt32
  let bits := b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24)
  let e := (bits >>> 23) &&& 0xFF; let m := bits &&& (0x7FFFFF : UInt32); let s := bits >>> 31
  if e == 0 then 0.0 else
    let fv := (1.0 + m.toNat.toFloat / 8388608.0) * Float.pow 2.0 (e.toNat.toFloat - 127.0)
    if s == 1 then -fv else fv

private def uu (ba : ByteArray) (i : Nat) : Nat :=
  let o := i * 4
  let b0 := ba.get! o |>.toNat
  let b1 := ba.get! (o+1) |>.toNat
  let b2 := ba.get! (o+2) |>.toNat
  let b3 := ba.get! (o+3) |>.toNat
  b0 + b1 * 256 + b2 * 65536 + b3 * 16777216

-- ═══ Test infra ═══

structure Ctx where
  cuda : CUDAContext
  passed : IO.Ref Nat
  failed : IO.Ref Nat

def run1 (c : Ctx) (kernel : ShaderM Unit) (ins : Array (String × ByteArray))
    (outName : String) (outBytes : USize := 4)
    (wgSize : Nat := 1) (numWG : Nat := 1) : IO ByteArray := do
  let mut bufs : Array (String × GPUBackend.Buf CUDAContext) := #[]
  for h : i in [:ins.size] do
    let (name, data) := ins[i]
    let buf ← GPUBackend.allocBuffer c.cuda data.size.toUSize
    GPUBackend.writeBuffer c.cuda buf data
    bufs := bufs.push (name, buf)
  let outBuf ← GPUBackend.allocBuffer c.cuda outBytes
  bufs := bufs.push (outName, outBuf)
  GPUBackend.execute c.cuda kernel bufs.toList
    { workgroupSize := { x := wgSize }, numWorkgroups := (numWG, 1, 1),
      extensions := if wgSize == 32 then ["subgroups"] else [] }
  let result ← GPUBackend.readBuffer c.cuda outBuf outBytes
  for h : i in [:bufs.size] do
    GPUBackend.freeBuffer c.cuda (bufs[i]).2
  return result

def expectF (c : Ctx) (name : String) (got expected : Float) (tol : Float := 1e-4) : IO Unit := do
  let diff := (got - expected).abs
  let ok := diff < tol || (expected.abs > 1e-10 && diff / expected.abs < tol)
  if ok then
    IO.println s!"  ✓ {name}: {got}"
    c.passed.modify (· + 1)
  else
    IO.println s!"  ✗ {name}: got={got} expected={expected} diff={diff}"
    c.failed.modify (· + 1)

def expectU (c : Ctx) (name : String) (got expected : Nat) : IO Unit := do
  if got == expected then
    IO.println s!"  ✓ {name}: {got}"
    c.passed.modify (· + 1)
  else
    IO.println s!"  ✗ {name}: got={got} expected={expected}"
    c.failed.modify (· + 1)

-- ═══ Kernel builders ═══

def unaryF (f : Exp (.scalar .f32) → Exp (.scalar .f32)) : ShaderM Unit := do
  let _a ← declareInputBuffer "a" (.array (.scalar .f32) 1)
  let _o ← declareOutputBuffer "o" (.array (.scalar .f32) 1)
  let v ← readBuffer (ty := .scalar .f32) (n := 1) "a" (Exp.litU32 0)
  writeBuffer (ty := .scalar .f32) "o" (Exp.litU32 0) (f v)

def binF (f : Exp (.scalar .f32) → Exp (.scalar .f32) → Exp (.scalar .f32)) : ShaderM Unit := do
  let _a ← declareInputBuffer "a" (.array (.scalar .f32) 1)
  let _b ← declareInputBuffer "b" (.array (.scalar .f32) 1)
  let _o ← declareOutputBuffer "o" (.array (.scalar .f32) 1)
  let va ← readBuffer (ty := .scalar .f32) (n := 1) "a" (Exp.litU32 0)
  let vb ← readBuffer (ty := .scalar .f32) (n := 1) "b" (Exp.litU32 0)
  writeBuffer (ty := .scalar .f32) "o" (Exp.litU32 0) (f va vb)

def binU (f : Exp (.scalar .u32) → Exp (.scalar .u32) → Exp (.scalar .u32)) : ShaderM Unit := do
  let _a ← declareInputBuffer "a" (.array (.scalar .u32) 1)
  let _b ← declareInputBuffer "b" (.array (.scalar .u32) 1)
  let _o ← declareOutputBuffer "o" (.array (.scalar .u32) 1)
  let va ← readBuffer (ty := .scalar .u32) (n := 1) "a" (Exp.litU32 0)
  let vb ← readBuffer (ty := .scalar .u32) (n := 1) "b" (Exp.litU32 0)
  writeBuffer (ty := .scalar .u32) "o" (Exp.litU32 0) (f va vb)

def cmpF (cmp : Exp (.scalar .f32) → Exp (.scalar .f32) → Exp (.scalar .bool)) : ShaderM Unit := do
  let _a ← declareInputBuffer "a" (.array (.scalar .f32) 1)
  let _b ← declareInputBuffer "b" (.array (.scalar .f32) 1)
  let _o ← declareOutputBuffer "o" (.array (.scalar .f32) 1)
  let va ← readBuffer (ty := .scalar .f32) (n := 1) "a" (Exp.litU32 0)
  let vb ← readBuffer (ty := .scalar .f32) (n := 1) "b" (Exp.litU32 0)
  writeBuffer (ty := .scalar .f32) "o" (Exp.litU32 0) (Exp.select (cmp va vb) (Exp.litF32 1.0) (Exp.litF32 0.0))

-- ═══ Main ═══
set_option maxRecDepth 1024 in

def main : IO Unit := do
  IO.println "═══ PTX Instruction-Level Tests ═══\n"
  let cuda ← CUDAContext.init
  let c : Ctx := { cuda, passed := ← IO.mkRef 0, failed := ← IO.mkRef 0 }

  -- ── f32 arithmetic ──
  IO.println "── f32 arithmetic ──"
  let r ← run1 c (binF Exp.add) #[("a", pf #[3.0]), ("b", pf #[5.0])] "o"; expectF c "add(3,5)" (uf r 0) 8.0
  let r ← run1 c (binF Exp.sub) #[("a", pf #[5.0]), ("b", pf #[3.0])] "o"; expectF c "sub(5,3)" (uf r 0) 2.0
  let r ← run1 c (binF Exp.mul) #[("a", pf #[3.0]), ("b", pf #[5.0])] "o"; expectF c "mul(3,5)" (uf r 0) 15.0
  let r ← run1 c (binF Exp.div) #[("a", pf #[5.0]), ("b", pf #[2.0])] "o"; expectF c "div(5,2)" (uf r 0) 2.5
  let r ← run1 c (unaryF Exp.neg) #[("a", pf #[3.0])] "o"; expectF c "neg(3)" (uf r 0) (-3.0)

  -- ── f32 math ──
  IO.println "── f32 math ──"
  let r ← run1 c (unaryF Exp.sqrt) #[("a", pf #[9.0])] "o"; expectF c "sqrt(9)" (uf r 0) 3.0
  let r ← run1 c (unaryF Exp.abs) #[("a", pf #[-3.5])] "o"; expectF c "abs(-3.5)" (uf r 0) 3.5
  let r ← run1 c (binF Exp.max) #[("a", pf #[3.0]), ("b", pf #[5.0])] "o"; expectF c "max(3,5)" (uf r 0) 5.0
  let r ← run1 c (binF Exp.min) #[("a", pf #[3.0]), ("b", pf #[5.0])] "o"; expectF c "min(3,5)" (uf r 0) 3.0
  let r ← run1 c (unaryF Exp.exp) #[("a", pf #[1.0])] "o"; expectF c "exp(1)" (uf r 0) (Float.exp 1.0) 1e-3
  let r ← run1 c (unaryF Exp.exp2) #[("a", pf #[3.0])] "o"; expectF c "exp2(3)" (uf r 0) 8.0
  let r ← run1 c (unaryF Exp.log) #[("a", pf #[Float.exp 1.0])] "o"; expectF c "log(e)" (uf r 0) 1.0 1e-3
  let r ← run1 c (unaryF Exp.log2) #[("a", pf #[8.0])] "o"; expectF c "log2(8)" (uf r 0) 3.0 1e-3
  let r ← run1 c (unaryF Exp.sin) #[("a", pf #[3.14159265 / 2.0])] "o"; expectF c "sin(π/2)" (uf r 0) 1.0 2e-3
  let r ← run1 c (unaryF Exp.cos) #[("a", pf #[0.0])] "o"; expectF c "cos(0)" (uf r 0) 1.0 1e-3
  let r ← run1 c (unaryF Exp.tanh) #[("a", pf #[0.0])] "o"; expectF c "tanh(0)" (uf r 0) 0.0
  let r ← run1 c (unaryF Exp.floor) #[("a", pf #[3.7])] "o"; expectF c "floor(3.7)" (uf r 0) 3.0
  let r ← run1 c (unaryF Exp.ceil) #[("a", pf #[3.2])] "o"; expectF c "ceil(3.2)" (uf r 0) 4.0
  -- inverseSqrt: KNOWN BUG — emits rcp(x) not 1/sqrt(x)
  let r ← run1 c (unaryF Exp.inverseSqrt) #[("a", pf #[4.0])] "o"; expectF c "inverseSqrt(4)=0.5" (uf r 0) 0.5 1e-2
  let r ← run1 c (binF Exp.pow) #[("a", pf #[2.0]), ("b", pf #[3.0])] "o"; expectF c "pow(2,3)" (uf r 0) 8.0 1e-3

  -- fma
  let fmaK : ShaderM Unit := do
    let _a ← declareInputBuffer "a" (.array (.scalar .f32) 1)
    let _b ← declareInputBuffer "b" (.array (.scalar .f32) 1)
    let _c ← declareInputBuffer "c" (.array (.scalar .f32) 1)
    let _o ← declareOutputBuffer "o" (.array (.scalar .f32) 1)
    let va ← readBuffer (ty := .scalar .f32) (n := 1) "a" (Exp.litU32 0)
    let vb ← readBuffer (ty := .scalar .f32) (n := 1) "b" (Exp.litU32 0)
    let vc ← readBuffer (ty := .scalar .f32) (n := 1) "c" (Exp.litU32 0)
    writeBuffer (ty := .scalar .f32) "o" (Exp.litU32 0) (Exp.fma va vb vc)
  let r ← run1 c fmaK #[("a", pf #[3.0]), ("b", pf #[5.0]), ("c", pf #[2.0])] "o"
  expectF c "fma(3,5,2)=17" (uf r 0) 17.0

  -- clamp
  let clampK (v : Float) : ShaderM Unit := do
    let _a ← declareInputBuffer "a" (.array (.scalar .f32) 1)
    let _o ← declareOutputBuffer "o" (.array (.scalar .f32) 1)
    let x ← readBuffer (ty := .scalar .f32) (n := 1) "a" (Exp.litU32 0)
    writeBuffer (ty := .scalar .f32) "o" (Exp.litU32 0) (Exp.clamp x (Exp.litF32 1.0) (Exp.litF32 4.0))
  let r ← run1 c (clampK 0) #[("a", pf #[5.0])] "o"; expectF c "clamp(5,1,4)=4" (uf r 0) 4.0
  let r ← run1 c (clampK 0) #[("a", pf #[0.5])] "o"; expectF c "clamp(0.5,1,4)=1" (uf r 0) 1.0
  let r ← run1 c (clampK 0) #[("a", pf #[2.5])] "o"; expectF c "clamp(2.5,1,4)=2.5" (uf r 0) 2.5

  -- select
  let selK : ShaderM Unit := do
    let _a ← declareInputBuffer "a" (.array (.scalar .f32) 1)
    let _o ← declareOutputBuffer "o" (.array (.scalar .f32) 1)
    let v ← readBuffer (ty := .scalar .f32) (n := 1) "a" (Exp.litU32 0)
    writeBuffer (ty := .scalar .f32) "o" (Exp.litU32 0)
      (Exp.select (Exp.gt v (Exp.litF32 2.0)) (Exp.litF32 10.0) (Exp.litF32 20.0))
  let r ← run1 c selK #[("a", pf #[3.0])] "o"; expectF c "select(3>2,10,20)=10" (uf r 0) 10.0
  let r ← run1 c selK #[("a", pf #[1.0])] "o"; expectF c "select(1>2,10,20)=20" (uf r 0) 20.0

  -- select u32 (was broken: selp_f32 used for u32 values → bit reinterpretation)
  let selU32K : ShaderM Unit := do
    let _a ← declareInputBuffer "a" (.array (.scalar .u32) 1)
    let _o ← declareOutputBuffer "o" (.array (.scalar .u32) 1)
    let v ← readBuffer (ty := .scalar .u32) (n := 1) "a" (Exp.litU32 0)
    writeBuffer (ty := .scalar .u32) "o" (Exp.litU32 0)
      (Exp.select (Exp.gt v (Exp.litU32 5)) (Exp.litU32 100) (Exp.litU32 200))
  let r ← run1 c selU32K #[("a", pu #[10])] "o"; expectU c "select_u32(10>5,100,200)=100" (uu r 0) 100
  let r ← run1 c selU32K #[("a", pu #[3])] "o"; expectU c "select_u32(3>5,100,200)=200" (uu r 0) 200

  -- ── u32 arithmetic ──
  IO.println "── u32 arithmetic ──"
  let r ← run1 c (binU Exp.add) #[("a", pu #[7]), ("b", pu #[3])] "o"; expectU c "add_u32(7,3)" (uu r 0) 10
  let r ← run1 c (binU Exp.sub) #[("a", pu #[7]), ("b", pu #[3])] "o"; expectU c "sub_u32(7,3)" (uu r 0) 4
  let r ← run1 c (binU Exp.mul) #[("a", pu #[7]), ("b", pu #[3])] "o"; expectU c "mul_u32(7,3)" (uu r 0) 21
  let r ← run1 c (binU Exp.div) #[("a", pu #[20]), ("b", pu #[3])] "o"; expectU c "div_u32(20,3)" (uu r 0) 6
  let r ← run1 c (binU Exp.mod) #[("a", pu #[20]), ("b", pu #[3])] "o"; expectU c "rem_u32(20,3)" (uu r 0) 2

  -- ── dp4a (packed 4x int8 dot product) ──
  IO.println "── dp4a ──"
  -- dot4U8Packed(a, b) = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3] where each byte is uint8
  -- Example: a=[1,2,3,4], b=[5,6,7,8] → 5+12+21+32 = 70
  let a_bytes : Nat := 1 + 2 * 256 + 3 * 65536 + 4 * 16777216
  let b_bytes : Nat := 5 + 6 * 256 + 7 * 65536 + 8 * 16777216
  let r ← run1 c (binU Exp.dot4U8Packed) #[("a", pu #[a_bytes]), ("b", pu #[b_bytes])] "o"
  expectU c "dot4U8Packed([1,2,3,4],[5,6,7,8])=70" (uu r 0) 70
  -- All 1s against all 1s: 1+1+1+1 = 4
  let r ← run1 c (binU Exp.dot4U8Packed) #[("a", pu #[0x01010101]), ("b", pu #[0x01010101])] "o"
  expectU c "dot4U8Packed(0x01010101, 0x01010101)=4" (uu r 0) 4
  -- Against 0x01010101: gets sum of bytes (useful for sum-of-quants)
  let r ← run1 c (binU Exp.dot4U8Packed) #[("a", pu #[0x01010101]), ("b", pu #[a_bytes])] "o"
  expectU c "dot4U8Packed(ones, [1,2,3,4])=10" (uu r 0) 10

  -- ── bitwise ──
  IO.println "── bitwise ──"
  let r ← run1 c (binU Exp.bitAnd) #[("a", pu #[0xFF0F]), ("b", pu #[0x0FF0])] "o"; expectU c "bitAnd" (uu r 0) 0x0F00
  let r ← run1 c (binU Exp.bitOr)  #[("a", pu #[0xFF00]), ("b", pu #[0x00FF])] "o"; expectU c "bitOr" (uu r 0) 0xFFFF
  let r ← run1 c (binU Exp.bitXor) #[("a", pu #[0xFF00]), ("b", pu #[0xF0F0])] "o"; expectU c "bitXor" (uu r 0) 0x0FF0
  let r ← run1 c (binU Exp.shiftRight) #[("a", pu #[256]), ("b", pu #[4])] "o"; expectU c "shr(256,4)" (uu r 0) 16
  -- shiftLeft: KNOWN BUG — dynamic shift falls back to 0
  let r ← run1 c (binU Exp.shiftLeft) #[("a", pu #[1]), ("b", pu #[4])] "o"; expectU c "shl(1,4)=16" (uu r 0) 16

  -- ── comparisons ──
  IO.println "── comparisons ──"
  let r ← run1 c (cmpF Exp.lt) #[("a", pf #[3.0]), ("b", pf #[5.0])] "o"; expectF c "lt(3,5)=T" (uf r 0) 1.0
  let r ← run1 c (cmpF Exp.lt) #[("a", pf #[5.0]), ("b", pf #[3.0])] "o"; expectF c "lt(5,3)=F" (uf r 0) 0.0
  let r ← run1 c (cmpF Exp.le) #[("a", pf #[3.0]), ("b", pf #[3.0])] "o"; expectF c "le(3,3)=T" (uf r 0) 1.0
  let r ← run1 c (cmpF Exp.eq) #[("a", pf #[3.0]), ("b", pf #[3.0])] "o"; expectF c "eq(3,3)=T" (uf r 0) 1.0
  let r ← run1 c (cmpF Exp.eq) #[("a", pf #[3.0]), ("b", pf #[5.0])] "o"; expectF c "eq(3,5)=F" (uf r 0) 0.0
  let r ← run1 c (cmpF Exp.gt) #[("a", pf #[5.0]), ("b", pf #[3.0])] "o"; expectF c "gt(5,3)=T" (uf r 0) 1.0
  let r ← run1 c (cmpF Exp.ge) #[("a", pf #[3.0]), ("b", pf #[3.0])] "o"; expectF c "ge(3,3)=T" (uf r 0) 1.0
  let r ← run1 c (cmpF Exp.ne) #[("a", pf #[3.0]), ("b", pf #[5.0])] "o"; expectF c "ne(3,5)=T" (uf r 0) 1.0
  let r ← run1 c (cmpF Exp.ne) #[("a", pf #[3.0]), ("b", pf #[3.0])] "o"; expectF c "ne(3,3)=F" (uf r 0) 0.0

  -- ── type conversions ──
  IO.println "── type conversions ──"
  let cvtF32K : ShaderM Unit := do
    let _a ← declareInputBuffer "a" (.array (.scalar .u32) 1)
    let _o ← declareOutputBuffer "o" (.array (.scalar .f32) 1)
    let v ← readBuffer (ty := .scalar .u32) (n := 1) "a" (Exp.litU32 0)
    writeBuffer (ty := .scalar .f32) "o" (Exp.litU32 0) (Exp.toF32 v)
  let r ← run1 c cvtF32K #[("a", pu #[42])] "o"; expectF c "toF32(42u)" (uf r 0) 42.0

  let cvtU32K : ShaderM Unit := do
    let _a ← declareInputBuffer "a" (.array (.scalar .f32) 1)
    let _o ← declareOutputBuffer "o" (.array (.scalar .u32) 1)
    let v ← readBuffer (ty := .scalar .f32) (n := 1) "a" (Exp.litU32 0)
    writeBuffer (ty := .scalar .u32) "o" (Exp.litU32 0) (Exp.toU32 v)
  let r ← run1 c cvtU32K #[("a", pf #[42.7])] "o"; expectU c "toU32(42.7)" (uu r 0) 42

  -- ── f16 unpack ──
  IO.println "── f16 unpack ──"
  let f16K : ShaderM Unit := do
    let _a ← declareInputBuffer "a" (.array (.scalar .u32) 1)
    let _o ← declareOutputBuffer "o" (.array (.scalar .f32) 2)
    let packed ← readBuffer (ty := .scalar .u32) (n := 1) "a" (Exp.litU32 0)
    let unpacked := Exp.unpack2x16float packed
    writeBuffer (ty := .scalar .f32) "o" (Exp.litU32 0) (Exp.vecX unpacked)
    writeBuffer (ty := .scalar .f32) "o" (Exp.litU32 1) (Exp.vecY unpacked)
  -- f16: 1.5=0x3E00, 2.5=0x4100 → packed=0x41003E00
  let r ← run1 c f16K #[("a", pu #[0x41003E00])] "o" 8
  expectF c "unpack_lo(1.5)" (uf r 0) 1.5 1e-3
  expectF c "unpack_hi(2.5)" (uf r 1) 2.5 1e-3

  -- ── boolean logic (KNOWN BUGS) ──
  IO.println "── boolean logic ──"

  let notK : ShaderM Unit := do
    let _a ← declareInputBuffer "a" (.array (.scalar .f32) 1)
    let _o ← declareOutputBuffer "o" (.array (.scalar .f32) 1)
    let v ← readBuffer (ty := .scalar .f32) (n := 1) "a" (Exp.litU32 0)
    let cond := Exp.gt v (Exp.litF32 2.0)
    let negated := Exp.not cond
    writeBuffer (ty := .scalar .f32) "o" (Exp.litU32 0) (Exp.select negated (Exp.litF32 1.0) (Exp.litF32 0.0))
  let r ← run1 c notK #[("a", pf #[3.0])] "o"; expectF c "not(3>2)=F→0" (uf r 0) 0.0

  let orK : ShaderM Unit := do
    let _a ← declareInputBuffer "a" (.array (.scalar .f32) 1)
    let _o ← declareOutputBuffer "o" (.array (.scalar .f32) 1)
    let v ← readBuffer (ty := .scalar .f32) (n := 1) "a" (Exp.litU32 0)
    let c1 := Exp.gt v (Exp.litF32 10.0)
    let c2 := Exp.lt v (Exp.litF32 5.0)
    let combined := Exp.or c1 c2
    writeBuffer (ty := .scalar .f32) "o" (Exp.litU32 0) (Exp.select combined (Exp.litF32 1.0) (Exp.litF32 0.0))
  let r ← run1 c orK #[("a", pf #[3.0])] "o"; expectF c "or(F,T)=T→1" (uf r 0) 1.0

  let andK : ShaderM Unit := do
    let _a ← declareInputBuffer "a" (.array (.scalar .f32) 1)
    let _o ← declareOutputBuffer "o" (.array (.scalar .f32) 1)
    let v ← readBuffer (ty := .scalar .f32) (n := 1) "a" (Exp.litU32 0)
    let c1 := Exp.gt v (Exp.litF32 1.0)
    let c2 := Exp.lt v (Exp.litF32 5.0)
    writeBuffer (ty := .scalar .f32) "o" (Exp.litU32 0) (Exp.select (Exp.and c1 c2) (Exp.litF32 1.0) (Exp.litF32 0.0))
  let r ← run1 c andK #[("a", pf #[3.0])] "o"; expectF c "and(T,T)=T→1" (uf r 0) 1.0

  -- ── control flow ──
  IO.println "── control flow ──"

  let loopK : ShaderM Unit := do
    let _o ← declareOutputBuffer "o" (.array (.scalar .f32) 1)
    let acc ← var (.scalar .f32) (Exp.litF32 0.0)
    loop (Exp.litU32 0) (Exp.litU32 10) (Exp.litU32 1) fun i => do
      assign acc (Exp.add (Exp.var acc) (Exp.toF32 i))
    writeBuffer (ty := .scalar .f32) "o" (Exp.litU32 0) (Exp.var acc)
  let r ← run1 c loopK #[] "o"; expectF c "loop sum(0..9)=45" (uf r 0) 45.0

  let ifK : ShaderM Unit := do
    let _a ← declareInputBuffer "a" (.array (.scalar .f32) 1)
    let _o ← declareOutputBuffer "o" (.array (.scalar .f32) 1)
    let v ← readBuffer (ty := .scalar .f32) (n := 1) "a" (Exp.litU32 0)
    let acc ← var (.scalar .f32) (Exp.litF32 0.0)
    if_ (Exp.gt v (Exp.litF32 2.0)) (assign acc (Exp.litF32 100.0)) (assign acc (Exp.litF32 200.0))
    writeBuffer (ty := .scalar .f32) "o" (Exp.litU32 0) (Exp.var acc)
  let r ← run1 c ifK #[("a", pf #[3.0])] "o"; expectF c "if(3>2)→100" (uf r 0) 100.0
  let r ← run1 c ifK #[("a", pf #[1.0])] "o"; expectF c "if(1>2)→200" (uf r 0) 200.0

  -- ── shared memory ──
  IO.println "── shared memory ──"

  let shmK : ShaderM Unit := do
    let _a ← declareInputBuffer "a" (.array (.scalar .f32) 1)
    let _o ← declareOutputBuffer "o" (.array (.scalar .f32) 1)
    sharedNamed "s" (.array (.scalar .f32) 1)
    let v ← readBuffer (ty := .scalar .f32) (n := 1) "a" (Exp.litU32 0)
    writeWorkgroup (ty := .scalar .f32) "s" (Exp.litU32 0) (Exp.mul v (Exp.litF32 2.0))
    barrier
    let sv ← readWorkgroup (ty := .scalar .f32) (n := 1) "s" (Exp.litU32 0)
    writeBuffer (ty := .scalar .f32) "o" (Exp.litU32 0) sv
  let r ← run1 c shmK #[("a", pf #[3.0])] "o"; expectF c "shared_f32 roundtrip" (uf r 0) 6.0

  let shmUK : ShaderM Unit := do
    let _a ← declareInputBuffer "a" (.array (.scalar .u32) 1)
    let _o ← declareOutputBuffer "o" (.array (.scalar .u32) 1)
    sharedNamed "su" (.array (.scalar .u32) 1)
    let v ← readBuffer (ty := .scalar .u32) (n := 1) "a" (Exp.litU32 0)
    writeWorkgroup (ty := .scalar .u32) "su" (Exp.litU32 0) v
    barrier
    let sv ← readWorkgroup (ty := .scalar .u32) (n := 1) "su" (Exp.litU32 0)
    writeBuffer (ty := .scalar .u32) "o" (Exp.litU32 0) sv
  let r ← run1 c shmUK #[("a", pu #[12345])] "o"; expectU c "shared_u32 roundtrip" (uu r 0) 12345

  -- ── subgroupAdd (wg=32) ──
  IO.println "── subgroupAdd ──"

  let sgK : ShaderM Unit := do
    let lid ← localId
    let tid := Exp.vec3X lid
    let _o ← declareOutputBuffer "o" (.array (.scalar .f32) 1)
    varNamed "r" (.scalar .f32) (Exp.subgroupAdd (Exp.toF32 tid))
    if_ (Exp.eq tid (Exp.litU32 0)) (do
      writeBuffer (ty := .scalar .f32) "o" (Exp.litU32 0) (Exp.var "r")
    ) (pure ())
  let r ← run1 c sgK #[] "o" 4 32 1
  expectF c "subgroupAdd(tid)=496" (uf r 0) 496.0

  -- ═══ Summary ═══
  let p ← c.passed.get; let f ← c.failed.get
  IO.println s!"\n═══ {p} passed, {f} failed, {p+f} total ═══"
  if f > 0 then IO.println "✗ SOME TESTS FAILED"
  else IO.println "✓ ALL TESTS PASSED"
