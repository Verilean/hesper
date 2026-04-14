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

/-- Generate PTX for an Exp, returning AnyReg + state. -/
partial def expToPTX (e : Exp t) (s : GenState) : ExpResult :=
  match e with
  -- Literals
  | .litF32 f =>
    let (r, s) := s.freshF32; (.f32 r, s.emit (.mov_f32_imm r (F32Imm.ofFloat f)))
  | .litU32 n =>
    let (r, s) := s.freshU32; (.u32 r, s.emit (.mov_u32_imm r n))
  | .litI32 n =>
    let (r, s) := s.freshU32; (.u32 r, s.emit (.mov_u32_imm r n.toNat))
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
    let (ra, s) := expToPTX a s; let (rb, s) := expToPTX b s
    match ra, rb with
    | .f32 a, .f32 b => let (r, s) := s.freshF32; (.f32 r, s.emit (.mul_f32 r a b))
    | .u32 a, .u32 b => let (r, s) := s.freshU32; (.u32 r, s.emit (.mul_lo_u32 r a b))
    | _, _ => (ra, s)
  | .div a b =>
    let (ra, s) := expToPTX a s; let (rb, s) := expToPTX b s
    match ra, rb with
    | .f32 a, .f32 b => let (r, s) := s.freshF32; (.f32 r, s.emit (.div_f32 r a b))
    | .u32 a, .u32 b => let (r, s) := s.freshU32; (.u32 r, s.emit (.div_u32 r a b))
    | _, _ => (ra, s)
  | .mod a b =>
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
  | .toU32 e =>
    let (re, s) := expToPTX e s
    let (r, s) := s.freshU32; (.u32 r, s.emit (.cvt_u32_f32 r re.toF32!))
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
    let (re, s) := expToPTX e s
    let (sq, s) := s.freshF32; let s := s.emit (.sqrt_f32 sq re.toF32!)
    let (r, s) := s.freshF32; (.f32 r, s.emit (.rcp_f32 r sq))
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

  -- Thread IDs
  | .vec3X v =>
    let (rv, s) := expToPTX v s; let (r, s) := s.freshU32
    let isGlobal := s.varMap.any (fun (n, reg) => reg == rv && n == "global_invocation_id")
    let isLocal  := s.varMap.any (fun (n, reg) => reg == rv && n == "local_invocation_id")
    let isWG     := s.varMap.any (fun (n, reg) => reg == rv && n == "workgroup_id")
    if isGlobal then
      let (c, s) := s.freshU32; let s := s.emit (.mov_sreg c .ctaid_x)
      let (n, s) := s.freshU32; let s := s.emit (.mov_sreg n .ntid_x)
      let (t, s) := s.freshU32; let s := s.emit (.mov_sreg t .tid_x)
      (.u32 r, s.emit (.mad_lo_u32 r c n t))
    else if isLocal then (.u32 r, s.emit (.mov_sreg r .tid_x))
    else if isWG    then (.u32 r, s.emit (.mov_sreg r .ctaid_x))
    else (.u32 r, s)
  | .vec3Y v =>
    let (rv, s) := expToPTX v s; let (r, s) := s.freshU32
    let isGlobal := s.varMap.any (fun (n, reg) => reg == rv && n == "global_invocation_id")
    let isLocal  := s.varMap.any (fun (n, reg) => reg == rv && n == "local_invocation_id")
    let isWG     := s.varMap.any (fun (n, reg) => reg == rv && n == "workgroup_id")
    if isGlobal then
      let (c, s) := s.freshU32; let s := s.emit (.mov_sreg c .ctaid_y)
      let (n, s) := s.freshU32; let s := s.emit (.mov_sreg n .ntid_y)
      let (t, s) := s.freshU32; let s := s.emit (.mov_sreg t .tid_y)
      (.u32 r, s.emit (.mad_lo_u32 r c n t))
    else if isLocal then (.u32 r, s.emit (.mov_sreg r .tid_y))
    else if isWG then (.u32 r, s.emit (.mov_sreg r .ctaid_y))
    else (.u32 r, s)

  -- Memory access
  | .index arr idx =>
    let (rArr, s) := expToPTX arr s; let (rIdx, s) := expToPTX idx s
    let arrName := match arr with | .var n => some n | _ => none
    let isShared := match arrName with | some n => s.isSharedVar n | none => false
    let isU32 := match arrName with | some n => s.isU32Buffer n | none => false
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
      -- u32 buffer → ld.global.u32
      let (r, s) := s.freshU32
      let (off, s) := s.freshU64; let s := s.emit (.mul_wide_u32 off rIdx.toU32! 4)
      let (addr, s) := s.freshU64; let s := s.emit (.add_u64 addr rArr.toU64! off)
      (.u32 r, s.emit (.ld_u32 .global r addr))
    else
      -- f32 buffer → ld.global.f32
      let (r, s) := s.freshF32
      let (off, s) := s.freshU64; let s := s.emit (.mul_wide_u32 off rIdx.toU32! 4)
      let (addr, s) := s.freshU64; let s := s.emit (.add_u64 addr rArr.toU64! off)
      (.f32 r, s.emit (.ld_f32 .global r addr))

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

  -- Barrier
  | .workgroupBarrier => let (r, s) := s.freshU32; (.u32 r, s.emit (.bar_sync 0))

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

  -- FMA
  | .fma a b c =>
    let (ra, s) := expToPTX a s; let (rb, s) := expToPTX b s; let (rc, s) := expToPTX c s
    let (r, s) := s.freshF32; (.f32 r, s.emit (.fma_f32 r ra.toF32! rb.toF32! rc.toF32!))

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
  | .varDecl name _ty (some ⟨_, init⟩) =>
    let (ri, s) := expToPTX init s; s.bindVar name ri

  | .assign name _ty value =>
    let (rv, s) := expToPTX value s
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

  | .exprStmt e => (expToPTX e s).2
  | .block stmts => stmts.foldl (fun s st => stmtToPTX st s) s

/-- Generate a complete PTX module string from a ShaderM computation. -/
def generatePTX
    (funcName : String := "main")
    (_workgroupSize : WorkgroupSize := {x := 256, y := 1, z := 1})
    (computation : ShaderM Unit)
    (ptxVersion : String := "8.0")
    (targetArch : String := "sm_89")
    : String :=
  let state := ShaderM.exec computation
  let sharedDecls := state.sharedVars.foldl (fun (acc : Array SharedDecl) (name, ty) =>
    match ty with
    | .array (.scalar .f32) n => acc.push { name, elemType := "f32", count := n }
    | .array (.scalar .u32) n => acc.push { name, elemType := "u32", count := n }
    | _ => acc) #[]
  let paramNames := state.declaredBuffers.map (·.1) |>.toArray
  let sharedNames := state.sharedVars.map (·.1)
  -- Identify u32-element buffers for correct ld.global.u32 generation
  let u32BufferNames := state.declaredBuffers.foldl (fun (acc : List String) (name, ty, _) =>
    match ty with
    | .array (.scalar .u32) _ => name :: acc
    | .array (.scalar .atomicU32) _ => name :: acc
    | _ => acc) []
  let u32SharedNames := state.sharedVars.foldl (fun (acc : List String) (name, ty) =>
    match ty with
    | .array (.scalar .u32) _ => name :: acc
    | _ => acc) []
  let initState := state.declaredBuffers.foldl (fun (s : GenState) (name, _, _) =>
    let (r, s) := s.freshU64; let s := s.emit (.ld_param_u64 r name); s.bindVar name (.u64 r))
    ({ sharedNames, u32BufferNames, u32SharedNames } : GenState)
  let finalState := (state.stmts.foldl (fun s st => stmtToPTX st s) initState).emit .ret
  (Module.mk ptxVersion targetArch funcName paramNames sharedDecls
    finalState.insts finalState.fRegs finalState.rRegs finalState.rdRegs finalState.pRegs finalState.hRegs).render

end Hesper.CUDA.CodeGen
