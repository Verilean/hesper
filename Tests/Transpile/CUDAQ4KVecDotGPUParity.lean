import Hesper.Transpile.CUDA
import Hesper.Transpile.CUDA.Parse
import Hesper.Transpile.CUDA.LowerStmt
import Hesper.WGSL.Monad
import Hesper.WGSL.Shader
import Hesper.CUDA.CodeGen
import Hesper.CUDA.Buffer
import Hesper.Basic

/-! # Phase 7 GPU parity: transpiled vec_dot_q4_K_q8_1_impl_mmq

Closes the loop: parse llama.cpp's `vec_dot_q4_K_q8_1_impl_mmq` body
(vecdotq.cuh:527) → ShaderM → PTX → JIT-load on RTX 4070 Ti → run with
a small synthetic input → compare against a CPU reference.

**STATUS** (2026-04-30): PASSING on RTX 4070 Ti (max |err| = 4.9e-4,
within f16 ds8/dm4 round-trip tolerance).  Required fixing four
underlying CodeGen issues:

1. `i32` buffer reads emitted `ld.global.f32` then `cvt.rzi.u32.f32` —
   destroying the bit pattern.  Fix: register `.scalar .i32` buffers
   alongside `.scalar .u32` in `u32BufferNames`, so PTX emits
   `ld.global.u32` (signed/unsigned share the PTX register file).

2. `Exp.toU32` always assumed input was f32 (`cvt.u32.f32`), silently
   corrupting u32-valued inputs.  Fix: dispatch on input register
   variant — u32 → identity, f32 → cvt — so `u32(int*)` reads work.

3. `Stmt.varDecl name (.scalar .i32) init` fell through to the generic
   fallback that aliased the var directly to the init register —
   colliding with the immediate-cache for literal 0.  Fix: dedicated
   i32 case that allocates a fresh register and emits a mov.

4. `lowerF32` for an `.ident` whose name is bound as i32 silently
   returned `Exp.var name` typed-but-mismatched (an f32 expression
   pointing at a u32 register).  In `Exp.mul` the type-mismatch
   fallback dropped the operand entirely.  Fix: consult `env.i32`
   in `lowerF32 .ident` and wrap with `Exp.toF32` for the implicit
   `int → float` promotion (matches C semantics).

Plus a transpiler-level fix:

5. Decl of `float2 v = expr;` previously emitted `ShaderM.varNamed` for
   a vec2.f32 local — but the PTX backend has no native vec2.f32
   register form, so `v.x` reads returned a stale f32 register.
   Fix: inline-substitute the init expression.  `f32x2 v` is bound
   to the lowered init *expression*, not `Exp.var name`.  Subsequent
   `v.x` / `v.y` lower to `vecX/vecY <init-expr>`, which CodeGen
   already handles for `unpack2x16float`.

The transpile **smoke** test (`CUDAQ4KVecDotSmoke`) was already
passing pre-fix (proves WGSL emission); this GPU parity test was the
end-to-end validation that uncovered the five CodeGen-side issues.
-/
namespace Hesper.Transpile.CUDA.Q4KVecDotGPU

open Hesper Hesper.WGSL Hesper.WGSL.Monad Hesper.CUDA Hesper.CUDA.CodeGen Hesper.Transpile.CUDA

/-- The transpiled body — same as in CUDAQ4KVecDotSmoke. -/
def vecDotBody : String :=
"{
  float sumf_d = 0.0f;
  float sumf_m = 0.0f;

  for (int i = 0; i < 4; i = i + 1) {
    int sumi_d = 0;
    for (int j = 0; j < 4; j = j + 1) {
      sumi_d = ggml_cuda_dp4a((v[j] >> (4 * i)) & 0x0F0F0F0F, u[i * 4 + j], sumi_d);
    }

    const float2 ds8f = __half22float2(ds8[i]);

    sumf_d = sumf_d + ds8f.x * (sc[i] * sumi_d);
    sumf_m = sumf_m + ds8f.y * m[i];
  }

  const float2 dm4f = __half22float2(dm4);

  result = dm4f.x * sumf_d - dm4f.y * sumf_m;
}"

/-- Build the kernel.  Single thread (workgroup 1×1×1, grid 1×1×1) reads
    inputs from buffers, runs the transpiled body with `result` bound to
    a local var, then writes the local to `out_buf[0]`. -/
def buildKernel : ShaderM Unit := do
  let _v   ← ShaderM.declareReadOnlyBuffer "v_buf"   (.array (.scalar .i32) 4)
  let _u   ← ShaderM.declareReadOnlyBuffer "u_buf"   (.array (.scalar .i32) 16)
  let _sc  ← ShaderM.declareReadOnlyBuffer "sc_buf"  (.array (.scalar .u32) 4)
  let _m   ← ShaderM.declareReadOnlyBuffer "m_buf"   (.array (.scalar .u32) 4)
  let _ds8 ← ShaderM.declareReadOnlyBuffer "ds8_buf" (.array (.scalar .u32) 4)
  let _dm4 ← ShaderM.declareReadOnlyBuffer "dm4_buf" (.array (.scalar .u32) 1)
  let _out ← ShaderM.declareOutputBuffer  "out_buf" (.array (.scalar .f32) 1)

  let dm4Val ← ShaderM.readBuffer (ty := .scalar .u32) (n := 1) "dm4_buf" (Exp.litU32 0)
  ShaderM.varNamed "dm4_packed" (.scalar .u32) dm4Val

  ShaderM.varNamed "result" (.scalar .f32) (Exp.litF32 0.0)

  let envFor : Env := {
    bufs := fun n => match n with
      | "v"   => some { name := "v_buf",   elemTy := .scalar .i32 }
      | "u"   => some { name := "u_buf",   elemTy := .scalar .i32 }
      | "sc"  => some { name := "sc_buf",  elemTy := .scalar .u32 }
      | "m"   => some { name := "m_buf",   elemTy := .scalar .u32 }
      | "ds8" => some { name := "ds8_buf", elemTy := .scalar .u32 }
      | _ => none
    f32 := fun n => match n with
      | "result" => some (Exp.var "result")
      | _ => none
    u32 := fun n => match n with
      | "dm4" => some (Exp.var "dm4_packed")
      | _ => none
  }

  let stmt := match parseStmtStr vecDotBody with
    | .ok s => s
    | .error e => panic! s!"parseStmtStr failed: {e}"
  match lowerStmt envFor stmt with
  | .ok act => act
  | .error e => panic! s!"lowerStmt failed: {e}"

  ShaderM.writeBuffer (ty := .scalar .f32) "out_buf" (Exp.litU32 0) (Exp.var "result")

/-- Sign-extend a byte (held in low 8 bits of a UInt32) to a signed Int. -/
def sext8 (b : UInt32) : Int :=
  let n := b.toNat &&& 0xFF
  if n ≥ 128 then Int.ofNat n - 256 else Int.ofNat n

/-- Convert IEEE 754 half (16-bit) bits to Lean Float (f64) — exact. -/
def f16BitsToF64 (h : UInt16) : Float :=
  let sign : UInt16 := (h >>> 15) &&& 1
  let exp16 : UInt16 := (h >>> 10) &&& 0x1F
  let mant16 : UInt16 := h &&& 0x3FF
  if exp16 == 0 then
    if mant16 == 0 then
      if sign == 1 then -0.0 else 0.0
    else
      let v := mant16.toNat.toFloat / 1024.0 * (2.0 ^ (-14.0 : Float))
      if sign == 1 then -v else v
  else if exp16 == 0x1F then
    if mant16 == 0 then
      let infBits : UInt64 := 0x7FF0000000000000
      Float.ofBits (sign.toUInt64 <<< 63 ||| infBits)
    else
      Float.ofBits 0x7FF8000000000000
  else
    let f64Sign : UInt64 := sign.toUInt64 <<< 63
    let unbiased : Int := Int.ofNat exp16.toNat - 15
    let f64Exp : UInt64 := UInt64.ofNat (unbiased + 1023).toNat <<< 52
    let f64Mant : UInt64 := mant16.toUInt64 <<< 42
    Float.ofBits (f64Sign ||| f64Exp ||| f64Mant)

/-- Convert Lean Float (f64) to IEEE 754 half (16-bit) bits.  Adequate
    for the deterministic test inputs we use (powers of two, no
    rounding-to-nearest-even subtleties). -/
def f64ToF16Bits (f : Float) : UInt16 := Id.run do
  let bits64 := f.toBits
  let sign : UInt16 := UInt16.ofNat ((bits64 >>> 63).toNat &&& 1)
  let exp64 : UInt64 := (bits64 >>> 52) &&& 0x7FF
  let mant64 : UInt64 := bits64 &&& 0xFFFFFFFFFFFFF  -- 52 bits
  if exp64 == 0 then return sign <<< 15
  if exp64 == 0x7FF then
    if mant64 == 0 then return (sign <<< 15) ||| 0x7C00
    else return (sign <<< 15) ||| 0x7C01
  let unbiased : Int := Int.ofNat exp64.toNat - 1023
  if unbiased > 15 then return (sign <<< 15) ||| 0x7C00
  if unbiased < -14 then return sign <<< 15  -- denormals → 0 (test inputs avoid this)
  let exp16 : UInt16 := UInt16.ofNat (unbiased + 15).toNat
  let mantOut : UInt16 := UInt16.ofNat ((mant64 >>> 42).toNat &&& 0x3FF)
  return (sign <<< 15) ||| (exp16 <<< 10) ||| mantOut

/-- Pack two f64 values as half2 (low 16 = first, high 16 = second). -/
def packHalf2 (lo hi : Float) : UInt32 :=
  let loBits := f64ToF16Bits lo
  let hiBits := f64ToF16Bits hi
  loBits.toUInt32 ||| (hiBits.toUInt32 <<< 16)

/-- ByteArray helper — concat little-endian u32. -/
def u32ArrayToBytes (arr : Array UInt32) : ByteArray :=
  arr.foldl (fun acc x => acc ++ Hesper.Basic.uint32ToBytes x) ByteArray.empty

/-- CPU reference for vec_dot_q4_K_q8_1_impl_mmq.  Treats dm4 / ds8 as
    f16 (matching what the GPU's `unpack2x16float` produces). -/
def cpuVecDot
    (v  : Array UInt32) (u : Array UInt32)  -- u stored as u32 bit-pattern
    (sc : Array UInt8)  (m : Array UInt8)
    (dm4Packed : UInt32) (ds8Packed : Array UInt32)
    : Float := Id.run do
  let dm4d := f16BitsToF64 dm4Packed.toUInt16
  let dm4m := f16BitsToF64 (dm4Packed >>> 16).toUInt16
  let mut sumf_d : Float := 0.0
  let mut sumf_m : Float := 0.0
  for i in [0:4] do
    let mut sumi_d : Int := 0
    for j in [0:4] do
      let vij : UInt32 := (v[j]! >>> (UInt32.ofNat (4*i))) &&& 0x0F0F0F0F
      let uw : UInt32 := u[i*4 + j]!
      for k in [0:4] do
        let shift := UInt32.ofNat (8*k)
        let va := sext8 ((vij >>> shift) &&& 0xFF)
        let vb := sext8 ((uw  >>> shift) &&& 0xFF)
        sumi_d := sumi_d + va * vb
    let ds8f_x := f16BitsToF64 ds8Packed[i]!.toUInt16
    let ds8f_y := f16BitsToF64 (ds8Packed[i]! >>> 16).toUInt16
    let scF : Float := Float.ofNat sc[i]!.toNat
    let mF  : Float := Float.ofNat m[i]!.toNat
    let sumiF : Float := Float.ofInt sumi_d
    sumf_d := sumf_d + ds8f_x * (scF * sumiF)
    sumf_m := sumf_m + ds8f_y * mF
  return dm4d * sumf_d - dm4m * sumf_m

unsafe def main : IO Unit := do
  IO.println "═══ Phase 7 GPU parity: transpiled vec_dot_q4_K_q8_1_impl_mmq ═══"
  let (_dev, _ctx) ← initCUDA

  let v : Array UInt32 := #[0x12345678, 0x9ABCDEF0, 0x11223344, 0xAABBCCDD]
  -- u: 16 32-bit words.  We use 0x7F7F7F7F-masked positives so signed bytes are positive.
  let u : Array UInt32 := (List.range 16).toArray.map (fun i =>
    (UInt32.ofNat (i*7+3) * 0x01010101) &&& 0x7F7F7F7F)
  let sc : Array UInt8 := #[7, 13, 21, 5]
  let m  : Array UInt8 := #[2, 9, 17, 11]
  let dm4Packed : UInt32 := packHalf2 0.0625 0.0078125
  let ds8Packed : Array UInt32 := #[
    packHalf2 0.5 0.25,
    packHalf2 0.125 0.375,
    packHalf2 0.75 0.0625,
    packHalf2 1.0 0.5]

  let refResult := cpuVecDot v u sc m dm4Packed ds8Packed
  IO.println s!"  CPU ref result = {refResult}"

  let kernel := buildKernel
  let ptx := generatePTX
    (funcName := "vec_dot_q4k_transpiled")
    (workgroupSize := { x := 1, y := 1, z := 1 })
    (computation := kernel)
    (targetArch := "sm_89")
  IO.println s!"  PTX size: {ptx.length} chars"
  if (← IO.getEnv "DUMP_PTX").isSome then IO.println ptx

  let cudaMod ← cuModuleLoadData ptx
  IO.println "  ptxas JIT: OK ✓"
  let func ← cuModuleGetFunction cudaMod "vec_dot_q4k_transpiled"

  let vBytes  := u32ArrayToBytes v
  let uBytes  := u32ArrayToBytes u
  let scBytes := u32ArrayToBytes (sc.map (·.toUInt32))
  let mBytes  := u32ArrayToBytes (m.map (·.toUInt32))
  let ds8Bytes := u32ArrayToBytes ds8Packed
  let dm4Bytes := u32ArrayToBytes #[dm4Packed]

  let vBuf   ← createCUDABuffer 16
  let uBuf   ← createCUDABuffer 64
  let scBuf  ← createCUDABuffer 16
  let mBuf   ← createCUDABuffer 16
  let ds8Buf ← createCUDABuffer 16
  let dm4Buf ← createCUDABuffer 4
  let outBuf ← createCUDABuffer 4
  writeCUDABuffer vBuf vBytes
  writeCUDABuffer uBuf uBytes
  writeCUDABuffer scBuf scBytes
  writeCUDABuffer mBuf mBytes
  writeCUDABuffer ds8Buf ds8Bytes
  writeCUDABuffer dm4Buf dm4Bytes
  IO.println "  buffers initialized"

  cuLaunchKernel func 1 1 1 1 1 1 0
    #[vBuf.ptr, uBuf.ptr, scBuf.ptr, mBuf.ptr, ds8Buf.ptr, dm4Buf.ptr, outBuf.ptr]

  let outBytes ← readCUDABufferFull outBuf
  let outArr := Hesper.Basic.bytesToFloatArrayPure outBytes
  let gpuResult := outArr[0]!
  IO.println s!"  GPU result     = {gpuResult}"

  let err := (gpuResult - refResult).abs
  IO.println s!"  |err|          = {err}"

  freeCUDABuffer vBuf
  freeCUDABuffer uBuf
  freeCUDABuffer scBuf
  freeCUDABuffer mBuf
  freeCUDABuffer ds8Buf
  freeCUDABuffer dm4Buf
  freeCUDABuffer outBuf

  if err < 1e-3 then
    IO.println "✓ PASSED — transpiled vec_dot matches CPU reference"
  else
    IO.println "✗ FAILED — output diverges from CPU reference"
    IO.Process.exit 1

end Hesper.Transpile.CUDA.Q4KVecDotGPU

unsafe def main : IO Unit := Hesper.Transpile.CUDA.Q4KVecDotGPU.main
