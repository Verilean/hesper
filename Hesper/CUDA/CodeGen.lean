import Hesper.WGSL.Monad
import Hesper.WGSL.Exp
import Hesper.WGSL.Shader
import Hesper.WGSL.Types
import Hesper.CUDA.PTX

/-!
# PTX Code Generation (GADT DSL)

Converts ShaderM computations to typed PTX instructions.
Register type mismatches (e.g., `setp.lt.f32` on u32 operands)
are caught at Lean compile time via GADT-typed registers.

## Pipeline

```
ShaderM Unit → expToPTX/stmtToPTX → Array PTX.Inst (type-safe)
             → PTX.Module.render → String → cuModuleLoadData
```
-/

namespace Hesper.CUDA.CodeGen

open Hesper.WGSL
open Hesper.WGSL.Monad (ShaderM)
open Hesper.CUDA.PTX

abbrev ExpResult := AnyReg × GenState

/-- log₂ of n if n is a power of two (1, 2, 4, …, 2^31), else none.
    Used for peephole lowering of `div/mul/mod` by a constant power of two
    to `shr/shl/and`, which avoids the 20-cycle `div.u32`. -/
private def log2PowerOfTwo? (n : Nat) : Option Nat :=
  if n == 0 then none
  else if n &&& (n - 1) != 0 then none
  else Id.run do
    let mut m := n
    let mut acc := 0
    for _ in [0:32] do
      if m <= 1 then break
      m := m >>> 1
      acc := acc + 1
    return some acc

/-- Emit a typed comparison, dispatching by register type. -/
private def emitSetp (op : CmpOp) (ra rb : AnyReg) (s : GenState) : AnyReg × GenState :=
  let (p, s) := s.freshPred
  match ra with
  | .u32 a => match rb with
    | .u32 b => (.pred p, s.emit (.setp_u32 op p a b))
    | _ => (.pred p, s)  -- type mismatch fallback
  | .f32 a => match rb with
    | .f32 b => (.pred p, s.emit (.setp_f32 op p a b))
    | _ => (.pred p, s)
  | _ => (.pred p, s)

/-- Should this Exp constructor be memoized?  Pure-arithmetic constructors
    (whose lowering depends only on operand AnyRegs and the GenState's
    fresh-counter) are safe at the constructor level.  Loads/shuffles/var
    refs are NOT cseable: var values can change via `assign`, loads have
    ordering with surrounding writes, shuffles read other lanes' regs.

    NOTE: even though this returns true for `.add`, the CSE *cache itself*
    is invalidated whenever `Stmt.assign` runs (see stmtToPTX).  This is
    the safety belt: a cseable expression like `Exp.var "x" + 1` returns
    the same register on cache hit, but if `x` was assigned between the
    two evaluations, the cached register holds the *old* x + 1.  Clearing
    the cache at every assign avoids this stale-cache hazard. -/
def Exp.cseable {t : WGSLType} : Exp t → Bool
  | _ => false  -- temporarily disabled: see project_cse_assign_bug.md
  -- Original safe-list (re-enable after fixing the production hazard):
  -- | .add _ _ | .sub _ _ | .mul _ _ | .div _ _ | .mod _ _
  -- | .shiftLeft _ _ | .shiftRight _ _
  -- | .bitAnd _ _ | .bitOr _ _ | .bitXor _ _
  -- | .min _ _ | .max _ _
  -- | .neg _ | .toF32 _ | .toI32 _ | .toU32 _ | .toF16 _
  -- | .mulhiU32 _ _ => true
  -- | _ => false

/-- Generate PTX for an Exp, returning AnyReg + state. -/
partial def expToPTX (e : Exp t) (s : GenState) : ExpResult :=
  if Exp.cseable e then
    let key := Exp.toWGSL e
    match s.expCache.find? (·.1 == key) with
    | some (_, r) => (r, s)
    | none =>
      let (r, s) := expToPTX' e s
      (r, { s with expCache := (key, r) :: s.expCache })
  else
    expToPTX' e s

where
  expToPTX' (e : Exp t) (s : GenState) : ExpResult :=
  match e with
  -- Literals
  | .litF32 f =>
    let (r, s) := s.freshF32; (.f32 r, s.emit (.mov_f32_imm r (F32Imm.ofFloat f)))
  | .litU32 n =>
    let (r, s) := s.readImmU32 n; (.u32 r, s)
  | .litI32 n =>
    let (r, s) := s.readImmU32 n.toNat; (.u32 r, s)
  | .litBool b =>
    let (r, s) := s.freshPred
    let (t1, s) := s.freshU32; let (t2, s) := s.freshU32
    let s := s.emit (.mov_u32_imm t1 (if b then 1 else 0))
    let s := s.emit (.mov_u32_imm t2 1)
    (.pred r, s.emit (.setp_u32 .eq r t1 t2))

  -- Variables
  | .var name =>
    match name with
    | "global_invocation_id" | "local_invocation_id" | "workgroup_id" =>
      match s.lookupVar name with
      | some reg => (reg, s)
      | none => let (r, s) := s.freshU32; (.u32 r, s.bindVar name (.u32 r))
    | _ =>
      match s.lookupVar name with
      | some reg => (reg, s)
      | none => let (r, s) := s.freshU32; (.u32 r, s.bindVar name (.u32 r))

  -- f32 arithmetic
  | .add a b =>
    let (ra, s) := expToPTX a s; let (rb, s) := expToPTX b s
    match ra, rb with
    | .f32 a, .f32 b => let (r, s) := s.freshF32; (.f32 r, s.emit (.add_f32 r a b))
    | .u32 a, .u32 b => let (r, s) := s.freshU32; (.u32 r, s.emit (.add_u32 r a b))
    | _, _ => (ra, s)  -- fallback
  | .sub a b =>
    let (ra, s) := expToPTX a s; let (rb, s) := expToPTX b s
    match ra, rb with
    | .f32 a, .f32 b => let (r, s) := s.freshF32; (.f32 r, s.emit (.sub_f32 r a b))
    | .u32 a, .u32 b => let (r, s) := s.freshU32; (.u32 r, s.emit (.sub_u32 r a b))
    | _, _ => (ra, s)
  | .mul a b =>
    -- Peephole: u32 × 2^k → shl by k (avoids mul.lo.u32 latency + an extra reg).
    match b with
    | .litU32 n => match log2PowerOfTwo? n with
      | some k =>
        let (ra, s) := expToPTX a s
        match ra with
        | .u32 a => let (r, s) := s.freshU32; (.u32 r, s.emit (.shl_u32 r a k))
        | _ =>
          let (rb, s) := expToPTX b s
          match ra, rb with
          | .f32 a, .f32 b => let (r, s) := s.freshF32; (.f32 r, s.emit (.mul_f32 r a b))
          | .u32 a, .u32 b => let (r, s) := s.freshU32; (.u32 r, s.emit (.mul_lo_u32 r a b))
          | _, _ => (ra, s)
      | none =>
        let (ra, s) := expToPTX a s; let (rb, s) := expToPTX b s
        match ra, rb with
        | .f32 a, .f32 b => let (r, s) := s.freshF32; (.f32 r, s.emit (.mul_f32 r a b))
        | .u32 a, .u32 b => let (r, s) := s.freshU32; (.u32 r, s.emit (.mul_lo_u32 r a b))
        | _, _ => (ra, s)
    | _ =>
      let (ra, s) := expToPTX a s; let (rb, s) := expToPTX b s
      match ra, rb with
      | .f32 a, .f32 b => let (r, s) := s.freshF32; (.f32 r, s.emit (.mul_f32 r a b))
      | .u32 a, .u32 b => let (r, s) := s.freshU32; (.u32 r, s.emit (.mul_lo_u32 r a b))
      | _, _ => (ra, s)
  | .div a b =>
    -- Peephole: u32 / 2^k → shr by k (avoids 20-cycle div.u32).
    match b with
    | .litU32 n => match log2PowerOfTwo? n with
      | some k =>
        let (ra, s) := expToPTX a s
        match ra with
        | .u32 a =>
          let (kReg, s) := s.readImmU32 k
          let (r, s) := s.freshU32
          (.u32 r, s.emit (.shr_u32 r a kReg))
        | _ =>
          let (rb, s) := expToPTX b s
          match ra, rb with
          | .f32 a, .f32 b => let (r, s) := s.freshF32; (.f32 r, s.emit (.div_f32 r a b))
          | .u32 a, .u32 b => let (r, s) := s.freshU32; (.u32 r, s.emit (.div_u32 r a b))
          | _, _ => (ra, s)
      | none =>
        let (ra, s) := expToPTX a s; let (rb, s) := expToPTX b s
        match ra, rb with
        | .f32 a, .f32 b => let (r, s) := s.freshF32; (.f32 r, s.emit (.div_f32 r a b))
        | .u32 a, .u32 b => let (r, s) := s.freshU32; (.u32 r, s.emit (.div_u32 r a b))
        | _, _ => (ra, s)
    | _ =>
      let (ra, s) := expToPTX a s; let (rb, s) := expToPTX b s
      match ra, rb with
      | .f32 a, .f32 b => let (r, s) := s.freshF32; (.f32 r, s.emit (.div_f32 r a b))
      | .u32 a, .u32 b => let (r, s) := s.freshU32; (.u32 r, s.emit (.div_u32 r a b))
      | _, _ => (ra, s)
  | .mod a b =>
    -- Peephole: u32 % 2^k → and with (2^k - 1) (avoids rem.u32 latency).
    match b with
    | .litU32 n => match log2PowerOfTwo? n with
      | some _k =>
        let (ra, s) := expToPTX a s
        match ra with
        | .u32 a =>
          let (mReg, s) := s.readImmU32 (n - 1)
          let (r, s) := s.freshU32
          (.u32 r, s.emit (.and_u32 r a mReg))
        | _ =>
          let (rb, s) := expToPTX b s
          match ra, rb with
          | .u32 a, .u32 b => let (r, s) := s.freshU32; (.u32 r, s.emit (.rem_u32 r a b))
          | _, _ => (ra, s)
      | none =>
        let (ra, s) := expToPTX a s; let (rb, s) := expToPTX b s
        match ra, rb with
        | .u32 a, .u32 b => let (r, s) := s.freshU32; (.u32 r, s.emit (.rem_u32 r a b))
        | _, _ => (ra, s)
    | _ =>
      let (ra, s) := expToPTX a s; let (rb, s) := expToPTX b s
      match ra, rb with
      | .u32 a, .u32 b => let (r, s) := s.freshU32; (.u32 r, s.emit (.rem_u32 r a b))
      | _, _ => (ra, s)
  | .neg e =>
    let (re, s) := expToPTX e s
    match re with
    | .f32 a => let (r, s) := s.freshF32; (.f32 r, s.emit (.neg_f32 r a))
    | _ => (re, s)

  -- Comparisons (type-dispatched via emitSetp)
  | .lt a b => let (ra, s) := expToPTX a s; let (rb, s) := expToPTX b s; emitSetp .lt ra rb s
  | .le a b => let (ra, s) := expToPTX a s; let (rb, s) := expToPTX b s; emitSetp .le ra rb s
  | .eq a b => let (ra, s) := expToPTX a s; let (rb, s) := expToPTX b s; emitSetp .eq ra rb s
  | .gt a b => let (ra, s) := expToPTX a s; let (rb, s) := expToPTX b s; emitSetp .gt ra rb s
  | .ge a b => let (ra, s) := expToPTX a s; let (rb, s) := expToPTX b s; emitSetp .ge ra rb s
  | .ne a b => let (ra, s) := expToPTX a s; let (rb, s) := expToPTX b s; emitSetp .ne ra rb s

  -- Boolean
  | .and a b =>
    let (ra, s) := expToPTX a s; let (rb, s) := expToPTX b s
    let (r, s) := s.freshPred
    (.pred r, s.emit (.and_pred r ra.toPred! rb.toPred!))

  | .or a b =>
    -- Convert predicates to u32 via selp, or them, convert back
    let (pa, s) := expToPTX a s; let (pb, s) := expToPTX b s
    let (one, s) := s.freshU32; let (zero, s) := s.freshU32
    let s := s.emit (.mov_u32_imm one 1); let s := s.emit (.mov_u32_imm zero 0)
    let (va, s) := s.freshU32; let s := s.emit (.selp_u32 va one zero pa.toPred!)
    let (vb, s) := s.freshU32; let s := s.emit (.selp_u32 vb one zero pb.toPred!)
    let (r, s) := s.freshU32; let s := s.emit (.or_u32 r va vb)
    let (p, s) := s.freshPred; (.pred p, s.emit (.setp_u32 .ne p r zero))
  | .not e =>
    -- Negate a predicate: setp.eq %p, val, 0
    let (pe, s) := expToPTX e s
    let (zero, s) := s.freshU32; let s := s.emit (.mov_u32_imm zero 0)
    let (one, s) := s.freshU32; let s := s.emit (.mov_u32_imm one 1)
    let (val, s) := s.freshU32; let s := s.emit (.mov_u32_imm val 0)
    let (p, s) := s.freshPred; (.pred p, s.emit (.setp_u32 .eq p val one))

  -- Bitwise (u32)
  | .bitAnd a b =>
    -- Peephole: `(x >> s) & ((1<<w)-1)` → `bfe.u32 dst, x, s, w`.
    -- Detects extractScaleMin-style bit slicing; collapses shr+and (2-3
    -- cycles) into a single-cycle bfe instruction.
    match a, b with
    | .shiftRight x (.litU32 startBit), .litU32 mask =>
      let w := Nat.log2 (mask + 1)
      if mask + 1 == 1 <<< w && w > 0 && w <= 32 && startBit < 32 then
        let (rx, s) := expToPTX x s
        let (r, s) := s.freshU32
        (.u32 r, s.emit (.bfe_u32 r rx.toU32! startBit w))
      else
        let (ra, s) := expToPTX a s; let (rb, s) := expToPTX b s
        let (r, s) := s.freshU32; (.u32 r, s.emit (.and_u32 r ra.toU32! rb.toU32!))
    -- Also: `x & ((1<<w)-1)` (no shift) → `bfe.u32 dst, x, 0, w`.
    | x, .litU32 mask =>
      let w := Nat.log2 (mask + 1)
      if mask + 1 == 1 <<< w && w > 0 && w <= 32 then
        let (rx, s) := expToPTX x s
        let (r, s) := s.freshU32
        (.u32 r, s.emit (.bfe_u32 r rx.toU32! 0 w))
      else
        let (ra, s) := expToPTX a s; let (rb, s) := expToPTX b s
        let (r, s) := s.freshU32; (.u32 r, s.emit (.and_u32 r ra.toU32! rb.toU32!))
    | _, _ =>
      let (ra, s) := expToPTX a s; let (rb, s) := expToPTX b s
      let (r, s) := s.freshU32; (.u32 r, s.emit (.and_u32 r ra.toU32! rb.toU32!))
  | .bitOr a b =>
    let (ra, s) := expToPTX a s; let (rb, s) := expToPTX b s
    let (r, s) := s.freshU32; (.u32 r, s.emit (.or_u32 r ra.toU32! rb.toU32!))
  | .bitXor a b =>
    let (ra, s) := expToPTX a s; let (rb, s) := expToPTX b s
    let (r, s) := s.freshU32; (.u32 r, s.emit (.xor_u32 r ra.toU32! rb.toU32!))
  | .shiftRight a b =>
    let (ra, s) := expToPTX a s; let (rb, s) := expToPTX b s
    let (r, s) := s.freshU32; (.u32 r, s.emit (.shr_u32 r ra.toU32! rb.toU32!))
  | .mulhiU32 a b =>
    let (ra, s) := expToPTX a s; let (rb, s) := expToPTX b s
    let (r, s) := s.freshU32; (.u32 r, s.emit (.mul_hi_u32 r ra.toU32! rb.toU32!))
  | .shiftLeft a b =>
    let (ra, s) := expToPTX a s
    match b with
    | .litU32 n =>
      let (r, s) := s.freshU32; (.u32 r, s.emit (.shl_u32 r ra.toU32! n))
    | _ =>
      let (rb, s) := expToPTX b s
      let (r, s) := s.freshU32; (.u32 r, s.emit (.shl_u32_reg r ra.toU32! rb.toU32!))

  -- Type conversions
  | .toF32 e =>
    let (re, s) := expToPTX e s
    let (r, s) := s.freshF32; (.f32 r, s.emit (.cvt_f32_u32 r re.toU32!))
    -- Note: for negative i32, cvt.rn.f32.u32 reinterprets as unsigned.
    -- Use cvt_f32_s32 variant for signed if needed (added below).
  | .toF32U e =>
    -- True unsigned f32 conversion.  Lowers to PTX cvt.rn.f32.u32 which
    -- ptxas turns into a single I2FP.F32.U32 SASS instruction without the
    -- prepending SGXT.U32 sign-extend that the .s32 variant inserts when
    -- the source is a narrow bit-field (e.g. `(x >> 8) & 0x3F` for Q4_K
    -- scale).  Use ONLY when the value is semantically non-negative.
    let (re, s) := expToPTX e s
    let (r, s) := s.freshF32; (.f32 r, s.emit (.cvt_f32_u32_real r re.toU32!))
  | .toU32 e =>
    -- Polymorphic in input type:
    --   u32 → identity (reinterpret; PTX shares the s32/u32 register file)
    --   i32 → identity (same as above; the bit pattern stays put)
    --   f32 → cvt.rzi.u32.f32 (round-toward-zero conversion)
    -- Previously assumed input was f32, which silently corrupted u32-valued
    -- inputs (e.g. an `int` buffer load reinterpreted via `Exp.toU32`).
    let (re, s) := expToPTX e s
    match re with
    | .u32 u => (.u32 u, s)
    | .f32 f => let (r, s) := s.freshU32; (.u32 r, s.emit (.cvt_u32_f32 r f))
    | _ => (re, s)
  | .toI32 e =>
    -- f32 → i32 (round toward zero, like toU32 but signed).
    -- Also handles u32 → i32 (identity bitcast) and i32 → i32 (identity).
    let (re, s) := expToPTX e s
    match re with
    | .f32 f => let (r, s) := s.freshU32; (.u32 r, s.emit (.cvt_u32_f32 r f))
    | .u32 u => (.u32 u, s)  -- reinterpret (PTX s32/u32 share register file)
    | _ => (re, s)
  | .toF16 e =>
    -- f32 → f16 (stored as AnyReg.u32 holding the b16 value for shared mem writes)
    let (re, s) := expToPTX e s
    let (h, s) := s.freshB16; let s := s.emit (.cvt_f16_f32 h re.toF32!)
    -- Return as u32 for writeWorkgroup compatibility (b16 in low bits)
    let (r, s) := s.freshU32
    -- TODO: proper b16 → u32 zero-extend. For now, shared mem f16 writes
    -- need special handling. Return the f32 version instead.
    let (rf, s) := s.freshF32; (.f32 rf, s.emit (.cvt_f32_f16 rf h))

  -- Math functions
  | .sqrt e => let (re, s) := expToPTX e s; let (r, s) := s.freshF32; (.f32 r, s.emit (.sqrt_f32 r re.toF32!))
  | .abs e  => let (re, s) := expToPTX e s; let (r, s) := s.freshF32; (.f32 r, s.emit (.abs_f32 r re.toF32!))
  | .max a b =>
    let (ra, s) := expToPTX a s; let (rb, s) := expToPTX b s
    let (r, s) := s.freshF32; (.f32 r, s.emit (.max_f32 r ra.toF32! rb.toF32!))
  | .min a b =>
    let (ra, s) := expToPTX a s; let (rb, s) := expToPTX b s
    let (r, s) := s.freshF32; (.f32 r, s.emit (.min_f32 r ra.toF32! rb.toF32!))
  | .exp e =>
    let (re, s) := expToPTX e s
    let (log2eR, s) := s.freshF32; let s := s.emit (.mov_f32_imm log2eR F32Imm.log2e)
    let (scaled, s) := s.freshF32; let s := s.emit (.mul_f32 scaled re.toF32! log2eR)
    let (r, s) := s.freshF32; (.f32 r, s.emit (.ex2_f32 r scaled))
  | .exp2 e =>
    let (re, s) := expToPTX e s; let (r, s) := s.freshF32; (.f32 r, s.emit (.ex2_f32 r re.toF32!))
  | .log e =>
    -- log(x) = lg2(x) / lg2(e) = lg2(x) * ln(2)
    let (re, s) := expToPTX e s
    let (lg, s) := s.freshF32; let s := s.emit (.lg2_f32 lg re.toF32!)
    let (ln2, s) := s.freshF32; let s := s.emit (.mov_f32_imm ln2 ⟨"3F317218"⟩)  -- ln(2) ≈ 0.6931
    let (r, s) := s.freshF32; (.f32 r, s.emit (.mul_f32 r lg ln2))
  | .log2 e =>
    let (re, s) := expToPTX e s; let (r, s) := s.freshF32; (.f32 r, s.emit (.lg2_f32 r re.toF32!))
  | .sin e =>
    let (re, s) := expToPTX e s; let (r, s) := s.freshF32; (.f32 r, s.emit (.sin_f32 r re.toF32!))
  | .cos e =>
    let (re, s) := expToPTX e s; let (r, s) := s.freshF32; (.f32 r, s.emit (.cos_f32 r re.toF32!))
  | .tanh e =>
    let (re, s) := expToPTX e s; let (r, s) := s.freshF32; (.f32 r, s.emit (.tanh_f32 r re.toF32!))
  | .floor e =>
    let (re, s) := expToPTX e s; let (r, s) := s.freshF32; (.f32 r, s.emit (.floor_f32 r re.toF32!))
  | .ceil e =>
    let (re, s) := expToPTX e s; let (r, s) := s.freshF32; (.f32 r, s.emit (.ceil_f32 r re.toF32!))
  | .inverseSqrt e =>
    -- Emit a single `rsqrt.approx.f32` (1 PTX instruction) instead of
    -- `sqrt.rn` + `rcp.approx` (2 instructions).  Matches llama.cpp's
    -- rms_norm_f32 codegen.  rsqrt.approx.f32 is IEEE-relaxed but the
    -- precision is sufficient for RMSNorm (and already used by cuBLAS
    -- / cuDNN for the same purpose).
    let (re, s) := expToPTX e s
    let (r, s) := s.freshF32
    (.f32 r, s.emit (.rsqrt_f32 r re.toF32!))
  | .clamp x lo hi =>
    let (rx, s) := expToPTX x s; let (rlo, s) := expToPTX lo s; let (rhi, s) := expToPTX hi s
    let (t, s) := s.freshF32; let s := s.emit (.max_f32 t rx.toF32! rlo.toF32!)
    let (r, s) := s.freshF32; (.f32 r, s.emit (.min_f32 r t rhi.toF32!))
  | .pow a b =>
    -- pow(a,b) = ex2(b * lg2(a))
    let (ra, s) := expToPTX a s; let (rb, s) := expToPTX b s
    let (lg, s) := s.freshF32; let s := s.emit (.lg2_f32 lg ra.toF32!)
    let (t, s) := s.freshF32; let s := s.emit (.mul_f32 t rb.toF32! lg)
    let (r, s) := s.freshF32; (.f32 r, s.emit (.ex2_f32 r t))

  -- Select (type-dispatched: f32 or u32)
  | .select cond t f =>
    let (pc, s) := expToPTX cond s; let (rt, s) := expToPTX t s; let (rf, s) := expToPTX f s
    match rt, rf with
    | .u32 a, .u32 b =>
      let (r, s) := s.freshU32; (.u32 r, s.emit (.selp_u32 r a b pc.toPred!))
    | _, _ =>
      let (r, s) := s.freshF32; (.f32 r, s.emit (.selp_f32 r rt.toF32! rf.toF32! pc.toPred!))

  -- Vec2 component extraction (f16 unpack pattern)
  | .vecX (.unpack2x16float packed) =>
    -- unpack2x16float → extract lower f16 from u32, convert to f32
    -- PTX: mov.b32 {%h_lo, %h_hi}, %r; cvt.f32.f16 %f, %h_lo;
    let (rp, s) := expToPTX packed s
    let (hLo, s) := s.freshB16; let (hHi, s) := s.freshB16
    let s := s.emit (.mov_b32_unpack hLo hHi rp.toU32!)
    let (r, s) := s.freshF32; (.f32 r, s.emit (.cvt_f32_f16 r hLo))

  | .vecY (.unpack2x16float packed) =>
    -- extract upper f16
    let (rp, s) := expToPTX packed s
    let (hLo, s) := s.freshB16; let (hHi, s) := s.freshB16
    let s := s.emit (.mov_b32_unpack hLo hHi rp.toU32!)
    let (r, s) := s.freshF32; (.f32 r, s.emit (.cvt_f32_f16 r hHi))

  | .vecX v =>
    let (rv, s) := expToPTX v s; (rv, s)
  | .vecY v =>
    let (rv, s) := expToPTX v s; (rv, s)

  -- pack2x16float (vec2 f32 → u32 packed half2).  Used to produce
  -- llama.cpp-compatible Q8_1 block headers where `ds` = half2(d, sum).
  -- PTX: cvt.rn.f16.f32 %h_lo, %x; cvt.rn.f16.f32 %h_hi, %y;
  --      mov.b32 %r, {%h_lo, %h_hi};
  | .pack2x16float (.vec2 x y) =>
    let (rx, s) := expToPTX x s
    let (ry, s) := expToPTX y s
    let (hLo, s) := s.freshB16
    let s := s.emit (.cvt_f16_f32 hLo rx.toF32!)
    let (hHi, s) := s.freshB16
    let s := s.emit (.cvt_f16_f32 hHi ry.toF32!)
    let (r, s) := s.freshU32
    (.u32 r, s.emit (.mov_b32_pack r hLo hHi))
  | .pack2x16float v =>
    -- Fallback: pack a non-literal vec2 by extracting x/y first.
    let (rv, s) := expToPTX v s
    -- `v` should evaluate to a concrete vec2 f32; reuse it for both halves
    -- via an identical cvt (caller currently only constructs vec2 literals,
    -- so this branch is a safety net).
    let (hLo, s) := s.freshB16
    let s := s.emit (.cvt_f16_f32 hLo rv.toF32!)
    let (hHi, s) := s.freshB16
    let s := s.emit (.cvt_f16_f32 hHi rv.toF32!)
    let (r, s) := s.freshU32
    (.u32 r, s.emit (.mov_b32_pack r hLo hHi))

  -- Thread IDs.  Special registers are kernel-launch-invariant, so cache
  -- the register holding each sreg's value via `readSReg`; subsequent uses
  -- of the same sreg reuse the register instead of re-issuing `mov.u32`.
  | .vec3X v =>
    let (rv, s) := expToPTX v s
    let isGlobal := s.varMap.any (fun (n, reg) => reg == rv && n == "global_invocation_id")
    let isLocal  := s.varMap.any (fun (n, reg) => reg == rv && n == "local_invocation_id")
    let isWG     := s.varMap.any (fun (n, reg) => reg == rv && n == "workgroup_id")
    if isGlobal then
      let (c, s) := s.readSReg .ctaid_x
      let (n, s) := s.readSReg .ntid_x
      let (t, s) := s.readSReg .tid_x
      let (r, s) := s.freshU32
      (.u32 r, s.emit (.mad_lo_u32 r c n t))
    else if isLocal then let (r, s) := s.readSReg .tid_x; (.u32 r, s)
    else if isWG    then let (r, s) := s.readSReg .ctaid_x; (.u32 r, s)
    else let (r, s) := s.freshU32; (.u32 r, s)
  | .vec3Y v =>
    let (rv, s) := expToPTX v s
    let isGlobal := s.varMap.any (fun (n, reg) => reg == rv && n == "global_invocation_id")
    let isLocal  := s.varMap.any (fun (n, reg) => reg == rv && n == "local_invocation_id")
    let isWG     := s.varMap.any (fun (n, reg) => reg == rv && n == "workgroup_id")
    if isGlobal then
      let (c, s) := s.readSReg .ctaid_y
      let (n, s) := s.readSReg .ntid_y
      let (t, s) := s.readSReg .tid_y
      let (r, s) := s.freshU32
      (.u32 r, s.emit (.mad_lo_u32 r c n t))
    else if isLocal then let (r, s) := s.readSReg .tid_y; (.u32 r, s)
    else if isWG then let (r, s) := s.readSReg .ctaid_y; (.u32 r, s)
    else let (r, s) := s.freshU32; (.u32 r, s)

  -- Memory access
  | .index arr idx =>
    let (rArr, s) := expToPTX arr s; let (rIdx, s) := expToPTX idx s
    let arrName := match arr with | .var n => some n | _ => none
    let isShared := match arrName with | some n => s.isSharedVar n | none => false
    let isU32 := match arrName with | some n => s.isU32Buffer n | none => false
    let isRO := match arrName with | some n => s.isReadOnlyBuffer n | none => false
    if isShared then
      let name := arrName.getD "unknown"
      let (off, s) := s.freshU32; let s := s.emit (.shl_u32 off rIdx.toU32! 2)
      let (symR, s) := s.freshU32; let s := s.emit (.mov_shared_addr symR name)
      let (addr, s) := s.freshU32; let s := s.emit (.add_u32 addr symR off)
      if s.isU32Shared name then
        let (r, s) := s.freshU32; (.u32 r, s.emit (.ld_shared_sym_u32 r symR off addr))
      else
        let (r, s) := s.freshF32; (.f32 r, s.emit (.ld_shared_sym r symR off addr))
    else if isU32 then
      -- u32 buffer → ld.global[.nc].u32 (nc hint for readOnly buffers → read-only L1)
      let (r, s) := s.freshU32
      let (off, s) := s.freshU64; let s := s.emit (.mul_wide_u32 off rIdx.toU32! 4)
      let (addr, s) := s.freshU64; let s := s.emit (.add_u64 addr rArr.toU64! off)
      (.u32 r, s.emit (.ld_u32 .global r addr isRO))
    else
      -- f32 buffer → ld.global[.nc].f32
      let (r, s) := s.freshF32
      let (off, s) := s.freshU64; let s := s.emit (.mul_wide_u32 off rIdx.toU32! 4)
      let (addr, s) := s.freshU64; let s := s.emit (.add_u64 addr rArr.toU64! off)
      (.f32 r, s.emit (.ld_f32 .global r addr isRO))

  -- Byte-granularity load from a u32-typed global buffer.  The buffer is
  -- addressed by *byte* index (no ×4 stride).  Emits a single ld.global.u8.
  | .loadByteFromU32Buf (n := _n) name byteIdx =>
    let rArr := (s.varMap.find? (·.1 == name)).map (·.2) |>.getD default
    let (rIdx, s) := expToPTX byteIdx s
    let isRO := s.isReadOnlyBuffer name
    -- Widen the u32 byte offset to u64 so we can add it to the 64-bit base.
    let (off, s) := s.freshU64; let s := s.emit (.mul_wide_u32 off rIdx.toU32! 1)
    let (addr, s) := s.freshU64; let s := s.emit (.add_u64 addr rArr.toU64! off)
    let (r, s) := s.freshU32
    (.u32 r, s.emit (.ld_u8 .global r addr isRO))

  -- Halfword (16-bit) load from a u32-typed global buffer.  Same addressing
  -- pattern as loadByteFromU32Buf but emits ld.global.u16.
  | .loadU16FromU32Buf (n := _n) name byteIdx =>
    let rArr := (s.varMap.find? (·.1 == name)).map (·.2) |>.getD default
    let (rIdx, s) := expToPTX byteIdx s
    let isRO := s.isReadOnlyBuffer name
    let (off, s) := s.freshU64; let s := s.emit (.mul_wide_u32 off rIdx.toU32! 1)
    let (addr, s) := s.freshU64; let s := s.emit (.add_u64 addr rArr.toU64! off)
    let (r, s) := s.freshU32
    (.u32 r, s.emit (.ld_u16 .global r addr isRO))

  -- bufferArray 2-level indexing: arr[bufIdx][elemIdx].
  -- `arr` is a u64 holding the base of a device-side pointer table
  -- (N × 8 bytes, each a CUdeviceptr).  On read:
  --   1. ptrOff = bufIdx * 8
  --   2. ld.global.u64 bufPtr, [arr + ptrOff]  -- the layer's base
  --   3. elemOff = elemIdx * 4 (u32/f32 element)
  --   4. ld.global.u32/f32 val, [bufPtr + elemOff]
  | .indexBuf (elemTy := elemTy) arr bufIdx elemIdx =>
    let (rArr, s) := expToPTX arr s
    let (rBufIdx, s) := expToPTX bufIdx s
    let (rElemIdx, s) := expToPTX elemIdx s
    let arrName := match arr with | .var n => some n | _ => none
    let isRO := match arrName with | some n => s.isReadOnlyBuffer n | none => true
    let isU32 := match arrName with | some n => s.isU32Buffer n | none => false
    -- Step 1+2: load the per-layer buffer pointer.
    let (ptrOff, s) := s.freshU64; let s := s.emit (.mul_wide_u32 ptrOff rBufIdx.toU32! 8)
    let (ptrAddr, s) := s.freshU64; let s := s.emit (.add_u64 ptrAddr rArr.toU64! ptrOff)
    let (bufPtr, s) := s.freshU64
    let s := s.emit (.ld_u64 .global bufPtr ptrAddr isRO)
    -- Step 3+4: load the element from that buffer.
    match elemTy with
    | .scalar .u32 =>
      let (elemOff, s) := s.freshU64; let s := s.emit (.mul_wide_u32 elemOff rElemIdx.toU32! 4)
      let (addr, s) := s.freshU64; let s := s.emit (.add_u64 addr bufPtr elemOff)
      let (r, s) := s.freshU32; (.u32 r, s.emit (.ld_u32 .global r addr isRO))
    | .scalar .f32 =>
      let (elemOff, s) := s.freshU64; let s := s.emit (.mul_wide_u32 elemOff rElemIdx.toU32! 4)
      let (addr, s) := s.freshU64; let s := s.emit (.add_u64 addr bufPtr elemOff)
      let (r, s) := s.freshF32; (.f32 r, s.emit (.ld_f32 .global r addr isRO))
    | _ =>
      -- Fallback: emit as u32 (caller will bitcast).
      let _ := isU32
      let (elemOff, s) := s.freshU64; let s := s.emit (.mul_wide_u32 elemOff rElemIdx.toU32! 4)
      let (addr, s) := s.freshU64; let s := s.emit (.add_u64 addr bufPtr elemOff)
      let (r, s) := s.freshU32; (.u32 r, s.emit (.ld_u32 .global r addr isRO))

  -- Subgroup add (warp butterfly)
  | .subgroupAdd val =>
    let (rv, s) := expToPTX val s; let (r, s) := s.freshF32
    let s := s.emit (.mov_f32 r rv.toF32!)
    let (t1, s) := s.freshF32; let s := s.emit (.shfl_bfly_f32 t1 r 16); let s := s.emit (.add_f32 r r t1)
    let (t2, s) := s.freshF32; let s := s.emit (.shfl_bfly_f32 t2 r 8);  let s := s.emit (.add_f32 r r t2)
    let (t3, s) := s.freshF32; let s := s.emit (.shfl_bfly_f32 t3 r 4);  let s := s.emit (.add_f32 r r t3)
    let (t4, s) := s.freshF32; let s := s.emit (.shfl_bfly_f32 t4 r 2);  let s := s.emit (.add_f32 r r t4)
    let (t5, s) := s.freshF32; let s := s.emit (.shfl_bfly_f32 t5 r 1);  let s := s.emit (.add_f32 r r t5)
    (.f32 r, s)

  -- Subgroup max (warp butterfly with max)
  | .subgroupMax val =>
    let (rv, s) := expToPTX val s; let (r, s) := s.freshF32
    let s := s.emit (.mov_f32 r rv.toF32!)
    let (t1, s) := s.freshF32; let s := s.emit (.shfl_bfly_f32 t1 r 16); let s := s.emit (.max_f32 r r t1)
    let (t2, s) := s.freshF32; let s := s.emit (.shfl_bfly_f32 t2 r 8);  let s := s.emit (.max_f32 r r t2)
    let (t3, s) := s.freshF32; let s := s.emit (.shfl_bfly_f32 t3 r 4);  let s := s.emit (.max_f32 r r t3)
    let (t4, s) := s.freshF32; let s := s.emit (.shfl_bfly_f32 t4 r 2);  let s := s.emit (.max_f32 r r t4)
    let (t5, s) := s.freshF32; let s := s.emit (.shfl_bfly_f32 t5 r 1);  let s := s.emit (.max_f32 r r t5)
    (.f32 r, s)

  -- Subgroup shuffle (read value from specific lane)
  | .subgroupShuffle val laneIdx =>
    let (rv, s) := expToPTX val s
    let (rl, s) := expToPTX laneIdx s
    let (r, s) := s.freshF32; (.f32 r, s.emit (.shfl_idx_f32 r rv.toF32! rl.toU32!))

  -- Subgroup shuffle XOR (butterfly, single step at given mask).
  -- Used by `warpReduceSum n` for n < 32.  Without this case, the
  -- default fallback returned an undefined u32 register and the
  -- subsequent `add` silently dropped the shuffle (returns first
  -- operand on f32×u32 mismatch) — V11's parity bug 2026-04-27.
  | .subgroupShuffleXor val mask =>
    let (rv, s) := expToPTX val s
    let (_rm, s) := expToPTX mask s  -- mask must be a compile-time literal
    -- Try to extract the literal mask from the original Exp.  shfl.bfly's
    -- offset is encoded in the instruction, not a register.
    let maskLit : Nat := match mask with
      | .litU32 n => n
      | _ => 1  -- best-effort: caller should pass a literal
    let (r, s) := s.freshF32
    (.f32 r, s.emit (.shfl_bfly_f32 r rv.toF32! maskLit))

  -- Subgroup broadcast from lane 0 to all lanes.  Lowers to
  -- `shfl.sync.idx.b32 dst, src, 0, 31, 0xFFFFFFFF`.  Used by Circuit
  -- DSL's `warpBroadcast` lowering — silently broken on CUDA before
  -- this case existed (same `_ => freshU32` fallback as
  -- subgroupShuffleXor; bug found 2026-04-27 by audit after V11 fix).
  | .subgroupBroadcastFirst val =>
    let (rv, s) := expToPTX val s
    let (rZero, s) := s.readImmU32 0
    let (r, s) := s.freshF32
    (.f32 r, s.emit (.shfl_idx_f32 r rv.toF32! rZero))

  -- Barrier
  | .workgroupBarrier => let (r, s) := s.freshU32; (.u32 r, s.emit (.bar_sync 0))
  | .warpBarrier      => let (r, s) := s.freshU32; (.u32 r, s.emit .bar_warp_sync)

  -- Round-to-nearest-even f32 → i32 (matches llama.cpp's roundf semantics).
  | .roundToI32 v =>
    let (rv, s) := expToPTX v s
    let (r, s) := s.freshU32
    (.u32 r, s.emit (.cvt_rni_s32_f32 r rv.toF32!))

  -- Bitcast: reinterpret 32-bit register between f32/u32/i32.
  -- On PTX, all 32-bit types share register file, so bitcast is free:
  -- just return the register tagged as the target type.
  -- Note: we rely on context (e.g. read/write buffer) to know what type
  -- the caller wants. Since AnyReg carries the type tag, we need mov.b32
  -- to a new register of the target type inferred from usage.
  -- Simple approach: f32 → u32 via mov.b32; u32 → f32 via mov.b32.
  -- But we can't know the target type from Exp.bitcast's erased type param.
  -- Workaround: emit mov.b32 to both a u32 and f32 register, caller picks.
  -- Actually, simpler: just pass through the register, reinterpreting as u32
  -- if needed by subsequent ops. Most common case: f32 → u32 for packing.
  | .bitcast e =>
    let (re, s) := expToPTX e s
    match re with
    | .f32 f =>
      -- f32 → u32: use mov.b32 to copy bits (free on PTX hardware)
      let (r, s) := s.freshU32; (.u32 r, s.emit (.mov_b32_f32_to_u32 r f))
    | .u32 u =>
      -- u32 → f32: use mov.b32 to copy bits
      let (r, s) := s.freshF32; (.f32 r, s.emit (.mov_b32_u32_to_f32 r u))
    | _ => (re, s)

  -- dp4a: packed 4×int8 dot product
  | .dot4I8Packed a b =>
    let (ra, s) := expToPTX a s; let (rb, s) := expToPTX b s
    let (zero, s) := s.freshU32; let s := s.emit (.mov_u32_imm zero 0)
    let (r, s) := s.freshU32; (.u32 r, s.emit (.dp4a_s32 r ra.toU32! rb.toU32! zero))
  | .dot4U8Packed a b =>
    let (ra, s) := expToPTX a s; let (rb, s) := expToPTX b s
    let (zero, s) := s.freshU32; let s := s.emit (.mov_u32_imm zero 0)
    let (r, s) := s.freshU32; (.u32 r, s.emit (.dp4a_u32 r ra.toU32! rb.toU32! zero))

  -- __vsubss4: packed signed-saturating sub per byte (sm_70+)
  | .subSatS8x4 a b =>
    let (ra, s) := expToPTX a s; let (rb, s) := expToPTX b s
    let (r, s) := s.freshU32
    (.u32 r, s.emit (.sub_sat_s8x4 r ra.toU32! rb.toU32!))

  -- FMA
  | .fma a b c =>
    let (ra, s) := expToPTX a s; let (rb, s) := expToPTX b s; let (rc, s) := expToPTX c s
    let (r, s) := s.freshF32; (.f32 r, s.emit (.fma_f32 r ra.toF32! rb.toF32! rc.toF32!))

  -- Packed half2 FMA: dst = a*b + c, each u32 holds two f16.  Single PTX
  -- `fma.rn.f16x2` instruction → half the FMA-throughput cost vs two
  -- scalar f16 fmas.
  | .fmaF16x2 a b c =>
    let (ra, s) := expToPTX a s; let (rb, s) := expToPTX b s; let (rc, s) := expToPTX c s
    let (r, s) := s.freshU32
    (.u32 r, s.emit (.fma_rn_f16x2 r ra.toU32! rb.toU32! rc.toU32!))

  -- ── WMMA (Tensor Core) ──
  -- For m16n16k16 f16/f16/f32 (most common Tensor Core shape).
  -- Fragments flow first-class through `AnyReg.wmma WmmaFrag`, so
  -- `MultiplyAccumulate` and `Store` can recover the full per-thread
  -- register group. The Exp's `bufRef : String` is `"&A"` (from
  -- WGSL) — strip the leading `&` and look up the u64 base in varMap.
  | .subgroupMatrixLoad (st := _st) (m := _m) (k := _k) bufRef offset _trans stride =>
    let bufName := if bufRef.startsWith "&" then bufRef.drop 1 else bufRef
    let (rOff, s) := expToPTX offset s
    let (rStride, s) := expToPTX stride s
    let rArr := (s.varMap.find? (·.1 == bufName)).map (·.2) |>.getD default
    -- m16n16k16 .f16: A fragment is 8 .b32 regs per thread (verified
    -- against nvcc reference output 2026-04-30; the "4 regs" figure
    -- common in tutorials is per-pair-of-threads, not per-thread).
    let (a0, s) := s.freshU32; let (a1, s) := s.freshU32
    let (a2, s) := s.freshU32; let (a3, s) := s.freshU32
    let (a4, s) := s.freshU32; let (a5, s) := s.freshU32
    let (a6, s) := s.freshU32; let (a7, s) := s.freshU32
    -- Byte address = base + offset*2 (each f16 = 2 bytes).
    let (off64, s) := s.freshU64; let s := s.emit (.mul_wide_u32 off64 rOff.toU32! 2)
    let (addr, s)  := s.freshU64; let s := s.emit (.add_u64 addr rArr.toU64! off64)
    let regs := #[a0, a1, a2, a3, a4, a5, a6, a7]
    let s := s.emit (.wmma_load_a_f16 regs addr rStride.toU32!)
    let frag : WmmaFrag := { kind := .a, elemTy := .f16, m := 16, n := 16, k := 16, u32regs := regs }
    (.wmma frag, s)
  | .subgroupMatrixLoadRight (st := _st) (k := _k) (n := _n) bufRef offset _trans stride =>
    let bufName := if bufRef.startsWith "&" then bufRef.drop 1 else bufRef
    let (rOff, s) := expToPTX offset s
    let (rStride, s) := expToPTX stride s
    let rArr := (s.varMap.find? (·.1 == bufName)).map (·.2) |>.getD default
    let (b0, s) := s.freshU32; let (b1, s) := s.freshU32
    let (b2, s) := s.freshU32; let (b3, s) := s.freshU32
    let (b4, s) := s.freshU32; let (b5, s) := s.freshU32
    let (b6, s) := s.freshU32; let (b7, s) := s.freshU32
    let (off64, s) := s.freshU64; let s := s.emit (.mul_wide_u32 off64 rOff.toU32! 2)
    let (addr, s)  := s.freshU64; let s := s.emit (.add_u64 addr rArr.toU64! off64)
    let regs := #[b0, b1, b2, b3, b4, b5, b6, b7]
    let s := s.emit (.wmma_load_b_f16 regs addr rStride.toU32!)
    let frag : WmmaFrag := { kind := .b, elemTy := .f16, m := 16, n := 16, k := 16, u32regs := regs }
    (.wmma frag, s)
  | .subgroupMatrixZeroResult (st := _st) (m := _m) (n := _n) =>
    -- 8 f32 result regs zeroed.
    let (c0, s) := s.freshF32; let (c1, s) := s.freshF32
    let (c2, s) := s.freshF32; let (c3, s) := s.freshF32
    let (c4, s) := s.freshF32; let (c5, s) := s.freshF32
    let (c6, s) := s.freshF32; let (c7, s) := s.freshF32
    let regs := #[c0, c1, c2, c3, c4, c5, c6, c7]
    let s := s.emit (.wmma_zero_d_f32 regs)
    let frag : WmmaFrag := { kind := .c, elemTy := .f32, m := 16, n := 16, k := 16, f32regs := regs }
    (.wmma frag, s)
  | .subgroupMatrixMultiplyAccumulate (st := _st) (m := _m) (k := _k) (n := _n) a b c =>
    let (ra, s) := expToPTX a s
    let (rb, s) := expToPTX b s
    let (rc, s) := expToPTX c s
    -- Allocate fresh D fragment (8 f32 regs for m16n16k16).
    let (d0, s) := s.freshF32; let (d1, s) := s.freshF32
    let (d2, s) := s.freshF32; let (d3, s) := s.freshF32
    let (d4, s) := s.freshF32; let (d5, s) := s.freshF32
    let (d6, s) := s.freshF32; let (d7, s) := s.freshF32
    let dRegs := #[d0, d1, d2, d3, d4, d5, d6, d7]
    let aFrag := ra.toWmma!
    let bFrag := rb.toWmma!
    let cFrag := rc.toWmma!
    let s := s.emit (.wmma_mma_f32_f16_f16_f32 dRegs aFrag.u32regs bFrag.u32regs cFrag.f32regs)
    let frag : WmmaFrag := { kind := .d, elemTy := .f32, m := 16, n := 16, k := 16, f32regs := dRegs }
    (.wmma frag, s)
  | .subgroupMatrixStore (st := _st) (m := _m) (n := _n) bufRef offset matExp _trans stride =>
    let bufName := if bufRef.startsWith "&" then bufRef.drop 1 else bufRef
    let (rOff, s)    := expToPTX offset s
    let (rMat, s)    := expToPTX matExp s
    let (rStride, s) := expToPTX stride s
    let rArr := (s.varMap.find? (·.1 == bufName)).map (·.2) |>.getD default
    -- Byte address = base + offset*4 (f32 result element = 4 bytes).
    let (off64, s) := s.freshU64; let s := s.emit (.mul_wide_u32 off64 rOff.toU32! 4)
    let (addr, s)  := s.freshU64; let s := s.emit (.add_u64 addr rArr.toU64! off64)
    let dFrag := rMat.toWmma!
    let s := s.emit (.wmma_store_d_f32 addr dFrag.f32regs rStride.toU32!)
    -- The Exp's declared return type is `Exp (.scalar .u32)` (a unit
    -- placeholder); emit a dummy mov.u32 0 to satisfy that.
    let (r, s) := s.freshU32; let s := s.emit (.mov_u32_imm r 0)
    (.u32 r, s)

  -- Fallback
  | _ => let (r, s) := s.freshU32; (.u32 r, s)

/-- Convert Stmt to PTX instructions. -/
partial def stmtToPTX (stmt : Stmt) (s : GenState) : GenState :=
  match stmt with
  | .varDecl name _ty none =>
    let (r, s) := s.freshF32; s.bindVar name (.f32 r)
  | .varDecl name _ty (some ⟨.scalar .f32, init⟩) =>
    let (ri, s) := expToPTX init s
    let (r, s) := s.freshF32; let s := s.emit (.mov_f32 r ri.toF32!)
    s.bindVar name (.f32 r)
  | .varDecl name _ty (some ⟨.scalar .u32, init⟩) =>
    let (ri, s) := expToPTX init s
    let (r, s) := s.freshU32; let s := s.emit (.mov_u32 r ri.toU32!)
    s.bindVar name (.u32 r)
  | .varDecl name _ty (some ⟨.scalar .i32, init⟩) =>
    -- i32 shares PTX's u32 register file but the value is a *mutable* var,
    -- so we MUST fresh a register and emit a mov — directly aliasing the
    -- init register would share state with the literal-cache (e.g. literal
    -- 0u's cached register), producing catastrophic register aliasing where
    -- assigning the i32 var stomps on the literal.  This is the bug that
    -- broke vec_dot_q4_K's `int sumi_d = 0; for (j) sumi_d = …;` pattern.
    let (ri, s) := expToPTX init s
    let (r, s) := s.freshU32; let s := s.emit (.mov_u32 r ri.toU32!)
    s.bindVar name (.u32 r)
  | .varDecl name _ty (some ⟨_, init⟩) =>
    let (ri, s) := expToPTX init s; s.bindVar name ri

  | .assign name _ty value =>
    -- Special case: WMMA accumulator self-update
    --   c_frag := subgroupMatrixMultiplyAccumulate(a, b, c_frag)
    -- We want the mma to write its result D back into c_frag's existing
    -- registers (PTX `wmma.mma.sync` allows D and C to alias). Without
    -- this, each iteration of a K-loop would allocate 8 fresh f32 regs,
    -- exploding register usage at K=2560 (160 iters → 1280 regs).
    match value with
    | .subgroupMatrixMultiplyAccumulate aExp bExp (.var cName) =>
      match s.lookupVar name with
      | some (.wmma cFragOld) =>
        if cName == name then
          let (raAny, s) := expToPTX aExp s
          let (rbAny, s) := expToPTX bExp s
          let aFrag := raAny.toWmma!
          let bFrag := rbAny.toWmma!
          let s := { s with expCache := [] }
          s.emit (.wmma_mma_f32_f16_f16_f32 cFragOld.f32regs
                                              aFrag.u32regs bFrag.u32regs
                                              cFragOld.f32regs)
        else
          let (rv, s) := expToPTX value s
          let s := { s with expCache := [] }
          s.bindVar name rv
      | _ =>
        let (rv, s) := expToPTX value s
        let s := { s with expCache := [] }
        s.bindVar name rv
    | _ =>
      let (rv, s) := expToPTX value s
      -- Invalidate exp CSE cache: any cached result that depended on `name`
      -- (directly or transitively via another var that was already in cache)
      -- now holds a stale register.  Clearing the entire cache is conservative
      -- but correct; assign is rare enough in straight-line PTX that the
      -- per-block CSE benefit between assigns is preserved.
      let s := { s with expCache := [] }
      match s.lookupVar name with
      | some (.f32 r) => s.emit (.mov_f32 r rv.toF32!)
      | some (.u32 r) => s.emit (.mov_u32 r rv.toU32!)
      | _ => s.bindVar name rv

  | .assignIndex arrName idx _ty value =>
    let (rIdx, s) := expToPTX idx s; let (rVal, s) := expToPTX value s
    if s.isSharedVar arrName then
      let (off, s) := s.freshU32; let s := s.emit (.shl_u32 off rIdx.toU32! 2)
      let (symR, s) := s.freshU32; let s := s.emit (.mov_shared_addr symR arrName)
      let (addr, s) := s.freshU32; let s := s.emit (.add_u32 addr symR off)
      if s.isU32Shared arrName then
        s.emit (.st_shared_sym_u32 rVal.toU32! symR off addr)
      else
        s.emit (.st_shared_sym rVal.toF32! symR off addr)
    else if s.isU32Buffer arrName then match s.lookupVar arrName with
    | some (.u64 base) =>
      let (off, s) := s.freshU64; let s := s.emit (.mul_wide_u32 off rIdx.toU32! 4)
      let (addr, s) := s.freshU64; let s := s.emit (.add_u64 addr base off)
      s.emit (.st_u32 .global addr rVal.toU32!)
    | _ => s
    else match s.lookupVar arrName with
    | some (.u64 base) =>
      let (off, s) := s.freshU64; let s := s.emit (.mul_wide_u32 off rIdx.toU32! 4)
      let (addr, s) := s.freshU64; let s := s.emit (.add_u64 addr base off)
      s.emit (.st_f32 .global addr rVal.toF32!)
    | _ =>
      -- Assume shared memory as fallback
      let (off, s) := s.freshU32; let s := s.emit (.shl_u32 off rIdx.toU32! 2)
      let (symR, s) := s.freshU32; let s := s.emit (.mov_shared_addr symR arrName)
      let (addr, s) := s.freshU32; let s := s.emit (.add_u32 addr symR off)
      s.emit (.st_shared_sym rVal.toF32! symR off addr)

  | .assignIndexBuf arrName bufIdx elemIdx ty value =>
    -- bufferArray write: arr[bufIdx][elemIdx] = value.  `arr` is a u64
    -- param holding the pointer-table base.  Two-step indirection.
    let (rBufIdx, s) := expToPTX bufIdx s
    let (rElemIdx, s) := expToPTX elemIdx s
    let (rVal, s) := expToPTX value s
    match s.lookupVar arrName with
    | some (.u64 tableBase) =>
      -- Load the per-layer buffer pointer from the table.
      let (ptrOff, s) := s.freshU64; let s := s.emit (.mul_wide_u32 ptrOff rBufIdx.toU32! 8)
      let (ptrAddr, s) := s.freshU64; let s := s.emit (.add_u64 ptrAddr tableBase ptrOff)
      let (bufPtr, s) := s.freshU64
      -- Pointer table is read-only (nc hint OK).
      let s := s.emit (.ld_u64 .global bufPtr ptrAddr true)
      -- Store into that buffer.
      let (elemOff, s) := s.freshU64; let s := s.emit (.mul_wide_u32 elemOff rElemIdx.toU32! 4)
      let (addr, s) := s.freshU64; let s := s.emit (.add_u64 addr bufPtr elemOff)
      match ty with
      | .scalar .u32 => s.emit (.st_u32 .global addr rVal.toU32!)
      | _ => s.emit (.st_f32 .global addr rVal.toF32!)
    | _ => s

  | .forLoop varName init cond update body =>
    let (ri, s) := expToPTX init s
    let (r, s) := s.freshU32; let s := s.emit (.mov_u32 r ri.toU32!)
    let s := s.bindVar varName (.u32 r)
    let (loopL, s) := s.freshLabel; let (endL, s) := s.freshLabel
    let s := s.emit (.label loopL)
    let (pc, s) := expToPTX cond s; let s := s.emit (.bra_not pc.toPred! endL)
    let s := body.foldl (fun s st => stmtToPTX st s) s
    let (ru, s) := expToPTX update s; let s := s.emit (.mov_u32 r ru.toU32!)
    let s := s.emit (.bra loopL)
    s.emit (.label endL)

  | .ifStmt cond thenBody elseBody =>
    let (pc, s) := expToPTX cond s
    let (elseL, s) := s.freshLabel; let (endL, s) := s.freshLabel
    let s := s.emit (.bra_not pc.toPred! elseL)
    let s := thenBody.foldl (fun s st => stmtToPTX st s) s
    if !elseBody.isEmpty then
      let s := s.emit (.bra endL); let s := s.emit (.label elseL)
      let s := elseBody.foldl (fun s st => stmtToPTX st s) s
      s.emit (.label endL)
    else s.emit (.label elseL)

  | .varDeclLdV4U32 n0 n1 n2 n3 bufName u32Idx =>
    -- Lower to one ld.global.nc.v4.u32 instruction, binding 4 fresh u32
    -- regs to the named vars.  Caller guarantees u32Idx is 4-aligned and
    -- the buffer pointer is 16-byte aligned.
    let rArr := (s.varMap.find? (·.1 == bufName)).map (·.2) |>.getD default
    let (rIdx, s) := expToPTX u32Idx s
    let isRO := s.isReadOnlyBuffer bufName
    -- byte offset = u32Idx * 4
    let (off, s) := s.freshU64
    let s := s.emit (.mul_wide_u32 off rIdx.toU32! 4)
    let (addr, s) := s.freshU64
    let s := s.emit (.add_u64 addr rArr.toU64! off)
    let (r0, s) := s.freshU32
    let (r1, s) := s.freshU32
    let (r2, s) := s.freshU32
    let (r3, s) := s.freshU32
    let s := s.emit (.ld_v4_u32 .global r0 r1 r2 r3 addr isRO)
    let s := s.bindVar n0 (.u32 r0)
    let s := s.bindVar n1 (.u32 r1)
    let s := s.bindVar n2 (.u32 r2)
    s.bindVar n3 (.u32 r3)

  | .varDeclLdV4F32Shared n0 n1 n2 n3 sharedName f32Idx =>
    -- Lower to one ld.shared.v4.f32 instruction, binding 4 fresh f32 regs
    -- to the named vars.  Caller guarantees f32Idx is 4-aligned (so the
    -- byte offset is 16-aligned).  Mirrors ld_shared_sym addressing.
    let (rIdx, s) := expToPTX f32Idx s
    -- byte offset = f32Idx * 4
    let (off, s) := s.freshU32
    let s := s.emit (.shl_u32 off rIdx.toU32! 2)
    let (symR, s) := s.freshU32
    let s := s.emit (.mov_shared_addr symR sharedName)
    let (addr, s) := s.freshU32
    let s := s.emit (.add_u32 addr symR off)
    let (r0, s) := s.freshF32
    let (r1, s) := s.freshF32
    let (r2, s) := s.freshF32
    let (r3, s) := s.freshF32
    let s := s.emit (.ld_shared_sym_v4_f32 r0 r1 r2 r3 symR off addr)
    let s := s.bindVar n0 (.f32 r0)
    let s := s.bindVar n1 (.f32 r1)
    let s := s.bindVar n2 (.f32 r2)
    s.bindVar n3 (.f32 r3)

  | .exprStmt e => (expToPTX e s).2
  | .block stmts =>
    -- Enter a new PTX scope: assign a fresh scopeId, save outer reg
    -- counters and inst array, run inner stmts with reset counters
    -- (so inner regs get IDs 0..N-1 within the scope), then wrap the
    -- inner instructions in `Inst.scopeBlock` and append to the outer
    -- inst array.  The inner-scope regs use prefix `%bf{scopeId}_<id>`
    -- so they don't collide with outer-scope `%f<id>` regs.
    --
    -- This way:
    --   * function-top declares `.reg .f32 %f<outerCount>;` etc.
    --   * inside `{ ... }`, declares `.reg .f32 %bf{N}_0, %bf{N}_1, ...;`
    --   * ptxas treats inner regs as block-local → physical SASS register
    --     reuse → register pressure goes down.
    let outerScope := s.scopeId
    let outerInsts := s.insts
    let outerF := s.fRegs; let outerR := s.rRegs; let outerRd := s.rdRegs
    let outerP := s.pRegs; let outerH := s.hRegs
    let myScopeId := s.scopeIdNext
    -- Switch to inner scope: reset reg counters, clear inst buffer, bump
    -- scope id, reserve next scope id for any nested scope.
    let s := { s with
      scopeId := myScopeId
      scopeIdNext := myScopeId + 1
      insts := #[]
      fRegs := 0
      rRegs := 0
      rdRegs := 0
      pRegs := 0
      hRegs := 0
      -- Clear sreg / imm / exp caches: outer-scope reg refs aren't visible inside.
      sregCache := []
      immCache := []
      expCache := []
    }
    let s := stmts.foldl (fun s st => stmtToPTX st s) s
    -- Capture inner state before restoring outer.
    let innerInsts := s.insts
    let numF := s.fRegs
    let numR := s.rRegs
    let numRd := s.rdRegs
    let numP := s.pRegs
    let numH := s.hRegs
    -- Restore outer scope: reg counters and inst array as they were.
    -- scopeIdNext keeps the bumped value (we used myScopeId already).
    let s := { s with
      scopeId := outerScope
      insts := outerInsts
      fRegs := outerF
      rRegs := outerR
      rdRegs := outerRd
      pRegs := outerP
      hRegs := outerH
      sregCache := []
      immCache := []
      expCache := []
    }
    s.emit (.scopeBlock myScopeId numF numR numRd numP numH innerInsts)

/-- Generate a complete PTX module string from a ShaderM computation. -/
def generatePTX
    (funcName : String := "main")
    (workgroupSize : WorkgroupSize := {x := 256, y := 1, z := 1})
    (computation : ShaderM Unit)
    (ptxVersion : String := "8.0")
    (targetArch : String := "sm_89")
    (maxnreg : Option Nat := none)
    (minnctapersm : Option Nat := none)
    : String :=
  -- `__launch_bounds__` analogue: derive maxntid from the user's workgroup
  -- size so ptxas knows the launch config and can use more registers per
  -- thread when occupancy is low (see Module.minnctapersm doc).
  let maxntid : Option (Nat × Nat × Nat) :=
    if workgroupSize.x > 0 then some (workgroupSize.x, workgroupSize.y, workgroupSize.z)
    else none
  let state := ShaderM.exec computation
  let sharedDecls := state.sharedVars.foldl (fun (acc : Array SharedDecl) (name, ty) =>
    match ty with
    | .array (.scalar .f32) n => acc.push { name, elemType := "f32", count := n }
    | .array (.scalar .u32) n => acc.push { name, elemType := "u32", count := n }
    | _ => acc) #[]
  let paramNames := state.declaredBuffers.map (·.1) |>.toArray
  let sharedNames := state.sharedVars.map (·.1)
  -- Identify u32/i32-element buffers for correct ld.global.u32 generation.
  -- (i32 buffers also use ld.global.u32 at the PTX level — bit pattern is
  -- the same, sign interpretation comes via Exp.toI32 / signed PTX ops.
  -- Treating them as f32 — the previous default — silently corrupted the
  -- value via cvt.rzi.u32.f32.)
  let u32BufferNames := state.declaredBuffers.foldl (fun (acc : List String) (name, ty, _) =>
    match ty with
    | .array (.scalar .u32) _ => name :: acc
    | .array (.scalar .i32) _ => name :: acc
    | .array (.scalar .atomicU32) _ => name :: acc
    | _ => acc) []
  let u32SharedNames := state.sharedVars.foldl (fun (acc : List String) (name, ty) =>
    match ty with
    | .array (.scalar .u32) _ => name :: acc
    | _ => acc) []
  -- Buffers declared with access mode `.read` get `ld.global.nc` (read-only L1).
  let readOnlyBufferNames := state.declaredBuffers.foldl (fun (acc : List String) (name, _, mode) =>
    match mode with
    | .read => name :: acc
    | _ => acc) []
  let initState := state.declaredBuffers.foldl (fun (s : GenState) (name, _, _) =>
    let (r, s) := s.freshU64; let s := s.emit (.ld_param_u64 r name); s.bindVar name (.u64 r))
    ({ sharedNames, u32BufferNames, u32SharedNames, readOnlyBufferNames } : GenState)
  let finalState := (state.stmts.foldl (fun s st => stmtToPTX st s) initState).emit .ret
  (Module.mk ptxVersion targetArch funcName paramNames sharedDecls
    finalState.insts finalState.fRegs finalState.rRegs finalState.rdRegs
    finalState.pRegs finalState.hRegs maxnreg minnctapersm maxntid).render

end Hesper.CUDA.CodeGen
