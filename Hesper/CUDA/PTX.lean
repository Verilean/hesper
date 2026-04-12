/-!
# PTX Instruction DSL (GADT version)

Typed registers ensure correct PTX instructions at Lean compile time.
For example, `setp_f32` only accepts `RegF32` — passing a `RegU32`
is a compile error, not a runtime `CUDA_ERROR_INVALID_PTX`.

## Register types

| Type     | PTX    | Lean         |
|----------|--------|-------------|
| float32  | `%f`   | `RegF32`    |
| uint32   | `%r`   | `RegU32`    |
| uint64   | `%rd`  | `RegU64`    |
| predicate| `%p`   | `RegPred`   |
-/

namespace Hesper.CUDA.PTX

-- ============================================================================
-- Typed registers
-- ============================================================================

structure RegF32  where id : Nat deriving BEq, Repr, Inhabited
structure RegU32  where id : Nat deriving BEq, Repr, Inhabited
structure RegU64  where id : Nat deriving BEq, Repr, Inhabited
structure RegPred where id : Nat deriving BEq, Repr, Inhabited
structure RegB16  where id : Nat deriving BEq, Repr, Inhabited  -- 16-bit (f16/b16)

instance : ToString RegF32  where toString r := s!"%f{r.id}"
instance : ToString RegU32  where toString r := s!"%r{r.id}"
instance : ToString RegU64  where toString r := s!"%rd{r.id}"
instance : ToString RegPred where toString r := s!"%p{r.id}"
instance : ToString RegB16  where toString r := s!"%h{r.id}"

/-- Untyped register for varMap (bridges Exp's erased types to typed PTX). -/
inductive AnyReg where
  | f32  (r : RegF32)
  | u32  (r : RegU32)
  | u64  (r : RegU64)
  | pred (r : RegPred)
  deriving BEq, Repr, Inhabited

instance : ToString AnyReg where
  toString
  | .f32 r  => toString r
  | .u32 r  => toString r
  | .u64 r  => toString r
  | .pred r => toString r

def AnyReg.isU32 : AnyReg → Bool
  | .u32 _ => true | _ => false

def AnyReg.isF32 : AnyReg → Bool
  | .f32 _ => true | _ => false

def AnyReg.toF32! : AnyReg → RegF32
  | .f32 r => r | _ => default

def AnyReg.toU32! : AnyReg → RegU32
  | .u32 r => r | _ => default

def AnyReg.toU64! : AnyReg → RegU64
  | .u64 r => r | _ => default

def AnyReg.toPred! : AnyReg → RegPred
  | .pred r => r | _ => default

-- ============================================================================
-- Special registers & immediates
-- ============================================================================

inductive SReg where
  | tid_x | tid_y | tid_z
  | ntid_x | ntid_y | ntid_z
  | ctaid_x | ctaid_y | ctaid_z
  deriving BEq, Repr

instance : ToString SReg where
  toString
  | .tid_x => "%tid.x" | .tid_y => "%tid.y" | .tid_z => "%tid.z"
  | .ntid_x => "%ntid.x" | .ntid_y => "%ntid.y" | .ntid_z => "%ntid.z"
  | .ctaid_x => "%ctaid.x" | .ctaid_y => "%ctaid.y" | .ctaid_z => "%ctaid.z"

structure F32Imm where hex : String deriving BEq, Repr

namespace F32Imm
instance : ToString F32Imm where toString imm := s!"0F{imm.hex}"
def zero   : F32Imm := ⟨"00000000"⟩
def one    : F32Imm := ⟨"3F800000"⟩
def two    : F32Imm := ⟨"40000000"⟩
def half   : F32Imm := ⟨"3F000000"⟩
def negOne : F32Imm := ⟨"BF800000"⟩
def log2e  : F32Imm := ⟨"3FB8AA3B"⟩

def ofFloat (f : Float) : F32Imm :=
  if f == 0.0 then zero
  else if f == 1.0 then one
  else if f == 2.0 then two
  else if f == 0.5 then half
  else if f == -1.0 then negOne
  else
    let bits64 : UInt64 := f.toBits
    let sign64 := (bits64 >>> 63) &&& 1
    let exp64  := (bits64 >>> 52) &&& 0x7FF
    let mant64 := bits64 &&& 0x000FFFFFFFFFFFFF
    let bits32 : UInt32 :=
      if exp64 == 0 then 0
      else if exp64 == 0x7FF then
        (sign64.toUInt32 <<< 31) ||| ((0xFF : UInt32) <<< 23) ||| ((mant64 >>> 29).toUInt32 &&& (0x7FFFFF : UInt32))
      else
        let exp32 : Int := exp64.toNat - 1023 + 127
        if exp32 <= 0 then 0
        else if exp32 >= 255 then (sign64.toUInt32 <<< 31) ||| ((0xFF : UInt32) <<< 23)
        else (sign64.toUInt32 <<< 31) ||| (exp32.toNat.toUInt32 <<< 23) |||
             ((mant64 >>> 29).toUInt32 &&& (0x7FFFFF : UInt32))
    ⟨hexU32 bits32⟩
where
  hexDigit (n : UInt32) : Char :=
    if n < 10 then Char.ofNat (48 + n.toNat) else Char.ofNat (65 + n.toNat - 10)
  hexU32 (v : UInt32) : String :=
    String.ofList [
      hexDigit ((v >>> 28) &&& 0xF), hexDigit ((v >>> 24) &&& 0xF),
      hexDigit ((v >>> 20) &&& 0xF), hexDigit ((v >>> 16) &&& 0xF),
      hexDigit ((v >>> 12) &&& 0xF), hexDigit ((v >>> 8) &&& 0xF),
      hexDigit ((v >>> 4) &&& 0xF),  hexDigit (v &&& 0xF)]
end F32Imm

structure Label where id : Nat deriving BEq, Repr
instance : ToString Label where toString l := s!"L{l.id}"

-- ============================================================================
-- Instructions (GADT — type-safe operands)
-- ============================================================================

inductive AddrSpace where | global | shared | param deriving BEq, Repr

inductive CmpOp where | eq | ne | lt | le | gt | ge deriving BEq, Repr
instance : ToString CmpOp where
  toString | .eq => "eq" | .ne => "ne" | .lt => "lt" | .le => "le" | .gt => "gt" | .ge => "ge"

/-- PTX instruction with type-safe register operands.
    Compile errors if you pass wrong register type. -/
inductive Inst where
  -- ── f32 arithmetic ── (all operands: RegF32)
  | mov_f32     (dst src : RegF32)
  | mov_f32_imm (dst : RegF32) (imm : F32Imm)
  | add_f32     (dst src1 src2 : RegF32)
  | sub_f32     (dst src1 src2 : RegF32)
  | mul_f32     (dst src1 src2 : RegF32)
  | div_f32     (dst src1 src2 : RegF32)
  | fma_f32     (dst a b c : RegF32)
  | abs_f32     (dst src : RegF32)
  | neg_f32     (dst src : RegF32)
  | sqrt_f32    (dst src : RegF32)
  | max_f32     (dst src1 src2 : RegF32)
  | min_f32     (dst src1 src2 : RegF32)
  | ex2_f32     (dst src : RegF32)
  | lg2_f32     (dst src : RegF32)
  | sin_f32     (dst src : RegF32)
  | cos_f32     (dst src : RegF32)
  | tanh_f32    (dst src : RegF32)
  | rcp_f32     (dst src : RegF32)
  | floor_f32   (dst src : RegF32)
  | ceil_f32    (dst src : RegF32)
  | selp_f32    (dst ifTrue ifFalse : RegF32) (pred : RegPred)

  -- ── u32 arithmetic ── (all operands: RegU32)
  | mov_u32     (dst src : RegU32)
  | mov_u32_imm (dst : RegU32) (imm : Nat)
  | add_u32     (dst src1 src2 : RegU32)
  | sub_u32     (dst src1 src2 : RegU32)
  | mul_lo_u32  (dst src1 src2 : RegU32)
  | mad_lo_u32  (dst a b c : RegU32)
  | div_u32     (dst src1 src2 : RegU32)
  | rem_u32     (dst src1 src2 : RegU32)
  | shl_u32     (dst src : RegU32) (bits : Nat)
  | shr_u32     (dst src1 src2 : RegU32)
  | and_u32     (dst src1 src2 : RegU32)
  | or_u32      (dst src1 src2 : RegU32)
  | xor_u32     (dst src1 src2 : RegU32)
  | not_u32     (dst src : RegU32)

  -- ── u64 arithmetic ── (all operands: RegU64, except mul_wide src is RegU32)
  | mov_u64     (dst src : RegU64)
  | add_u64     (dst src1 src2 : RegU64)
  | mul_wide_u32 (dst : RegU64) (src : RegU32) (imm : Nat)

  -- ── type conversions ──
  | cvt_f32_u32 (dst : RegF32) (src : RegU32)
  | cvt_u32_f32 (dst : RegU32) (src : RegF32)
  | cvt_f32_f16 (dst : RegF32) (src : RegB16)   -- f16 → f32
  | cvt_f16_f32 (dst : RegB16) (src : RegF32)   -- f32 → f16

  -- ── f16 unpack (u32 → two b16) ──
  | mov_b32_unpack (lo hi : RegB16) (src : RegU32)  -- mov.b32 {%h_lo, %h_hi}, %r

  -- ── comparison → predicate ── (operands must match comparison type)
  | setp_f32    (op : CmpOp) (dst : RegPred) (src1 src2 : RegF32)
  | setp_u32    (op : CmpOp) (dst : RegPred) (src1 src2 : RegU32)

  -- ── predicate logic ──
  | and_pred    (dst p1 p2 : RegPred)

  -- ── special register read → RegU32 ──
  | mov_sreg    (dst : RegU32) (sreg : SReg)

  -- ── memory ──
  | ld_f32        (space : AddrSpace) (dst : RegF32) (addr : RegU64)
  | st_f32        (space : AddrSpace) (addr : RegU64) (val : RegF32)
  | ld_u32        (space : AddrSpace) (dst : RegU32) (addr : RegU64)
  | st_u32        (space : AddrSpace) (addr : RegU64) (val : RegU32)
  -- shared memory via symbol: mov.u32 %r, sym; add.u32 %r, %r, off; ld/st
  | ld_shared_sym (dst : RegF32) (symAddr : RegU32) (offset : RegU32) (addr : RegU32)
  | st_shared_sym (val : RegF32) (symAddr : RegU32) (offset : RegU32) (addr : RegU32)
  | ld_shared_sym_u32 (dst : RegU32) (symAddr : RegU32) (offset : RegU32) (addr : RegU32)
  | st_shared_sym_u32 (val : RegU32) (symAddr : RegU32) (offset : RegU32) (addr : RegU32)
  | mov_shared_addr (dst : RegU32) (symName : String)  -- mov.u32 %r, symbol
  | ld_param_u64  (dst : RegU64) (paramName : String)

  -- ── warp shuffle ──
  | shfl_bfly_f32 (dst src : RegF32) (offset : Nat)

  -- ── control flow ──
  | bar_sync     (id : Nat)
  | bra          (target : Label)
  | bra_not      (pred : RegPred) (target : Label)
  | label        (l : Label)
  | ret
  deriving Repr

-- ============================================================================
-- Pretty-printer
-- ============================================================================

private def spStr : AddrSpace → String
  | .global => "global" | .shared => "shared" | .param => "param"

def Inst.toString : Inst → String
  | .mov_f32 d s         => s!"  mov.f32 {d}, {s};"
  | .mov_f32_imm d imm   => s!"  mov.f32 {d}, {imm};"
  | .add_f32 d a b       => s!"  add.f32 {d}, {a}, {b};"
  | .sub_f32 d a b       => s!"  sub.f32 {d}, {a}, {b};"
  | .mul_f32 d a b       => s!"  mul.f32 {d}, {a}, {b};"
  | .div_f32 d a b       => s!"  div.rn.f32 {d}, {a}, {b};"
  | .fma_f32 d a b c     => s!"  fma.rn.f32 {d}, {a}, {b}, {c};"
  | .abs_f32 d s         => s!"  abs.f32 {d}, {s};"
  | .neg_f32 d s         => s!"  neg.f32 {d}, {s};"
  | .sqrt_f32 d s        => s!"  sqrt.rn.f32 {d}, {s};"
  | .max_f32 d a b       => s!"  max.f32 {d}, {a}, {b};"
  | .min_f32 d a b       => s!"  min.f32 {d}, {a}, {b};"
  | .ex2_f32 d s         => s!"  ex2.approx.f32 {d}, {s};"
  | .lg2_f32 d s         => s!"  lg2.approx.f32 {d}, {s};"
  | .sin_f32 d s         => s!"  sin.approx.f32 {d}, {s};"
  | .cos_f32 d s         => s!"  cos.approx.f32 {d}, {s};"
  | .tanh_f32 d s        => s!"  tanh.approx.f32 {d}, {s};"
  | .rcp_f32 d s         => s!"  rcp.approx.f32 {d}, {s};"
  | .floor_f32 d s       => s!"  cvt.rmi.f32.f32 {d}, {s};"
  | .ceil_f32 d s        => s!"  cvt.rpi.f32.f32 {d}, {s};"
  | .selp_f32 d t f p    => s!"  selp.f32 {d}, {t}, {f}, {p};"
  | .mov_u32 d s         => s!"  mov.u32 {d}, {s};"
  | .mov_u32_imm d n     => s!"  mov.u32 {d}, {n};"
  | .add_u32 d a b       => s!"  add.u32 {d}, {a}, {b};"
  | .sub_u32 d a b       => s!"  sub.u32 {d}, {a}, {b};"
  | .mul_lo_u32 d a b    => s!"  mul.lo.u32 {d}, {a}, {b};"
  | .mad_lo_u32 d a b c  => s!"  mad.lo.u32 {d}, {a}, {b}, {c};"
  | .div_u32 d a b       => s!"  div.u32 {d}, {a}, {b};"
  | .rem_u32 d a b       => s!"  rem.u32 {d}, {a}, {b};"
  | .shl_u32 d s n       => s!"  shl.b32 {d}, {s}, {n};"
  | .shr_u32 d a b       => s!"  shr.u32 {d}, {a}, {b};"
  | .and_u32 d a b       => s!"  and.b32 {d}, {a}, {b};"
  | .or_u32 d a b        => s!"  or.b32 {d}, {a}, {b};"
  | .xor_u32 d a b       => s!"  xor.b32 {d}, {a}, {b};"
  | .not_u32 d s         => s!"  not.b32 {d}, {s};"
  | .mov_u64 d s         => s!"  mov.u64 {d}, {s};"
  | .add_u64 d a b       => s!"  add.u64 {d}, {a}, {b};"
  | .mul_wide_u32 d s n  => s!"  mul.wide.u32 {d}, {s}, {n};"
  | .cvt_f32_u32 d s     => s!"  cvt.rn.f32.u32 {d}, {s};"
  | .cvt_u32_f32 d s     => s!"  cvt.rzi.u32.f32 {d}, {s};"
  | .cvt_f32_f16 d s     => s!"  cvt.f32.f16 {d}, {s};"
  | .cvt_f16_f32 d s     => s!"  cvt.rn.f16.f32 {d}, {s};"
  | .mov_b32_unpack l h src => s!"  mov.b32 " ++ "{" ++ s!"{l}, {h}" ++ "}" ++ s!", {src};"
  | .setp_f32 op d a b   => s!"  setp.{op}.f32 {d}, {a}, {b};"
  | .setp_u32 op d a b   => s!"  setp.{op}.u32 {d}, {a}, {b};"
  | .and_pred d a b      => s!"  and.pred {d}, {a}, {b};"
  | .mov_sreg d sr       => s!"  mov.u32 {d}, {sr};"
  | .ld_f32 sp d a       => s!"  ld.{spStr sp}.f32 {d}, [{a}];"
  | .st_f32 sp a v       => s!"  st.{spStr sp}.f32 [{a}], {v};"
  | .ld_u32 sp d a       => s!"  ld.{spStr sp}.u32 {d}, [{a}];"
  | .st_u32 sp a v       => s!"  st.{spStr sp}.u32 [{a}], {v};"
  | .ld_shared_sym d _sa _off addr => s!"  ld.shared.f32 {d}, [{addr}];"
  | .st_shared_sym v _sa _off addr => s!"  st.shared.f32 [{addr}], {v};"
  | .ld_shared_sym_u32 d _sa _off addr => s!"  ld.shared.u32 {d}, [{addr}];"
  | .st_shared_sym_u32 v _sa _off addr => s!"  st.shared.u32 [{addr}], {v};"
  | .mov_shared_addr d sym         => s!"  mov.u32 {d}, {sym};"
  | .ld_param_u64 d name => s!"  ld.param.u64 {d}, [param_{name}];"
  | .shfl_bfly_f32 d s off => s!"  shfl.sync.bfly.b32 {d}, {s}, {off}, 31, 0xFFFFFFFF;"
  | .bar_sync id         => s!"  bar.sync {id};"
  | .bra target          => s!"  bra {target};"
  | .bra_not p target    => s!"  @!{p} bra {target};"
  | .label l             => s!"{l}:"
  | .ret                 => "  ret;"

instance : ToString Inst where toString := Inst.toString

-- ============================================================================
-- Module assembly
-- ============================================================================

structure SharedDecl where
  name : String
  elemType : String
  count : Nat

structure Module where
  version : String := "8.0"
  target : String := "sm_89"
  funcName : String := "main"
  params : Array String
  sharedDecls : Array SharedDecl := #[]
  body : Array Inst
  fRegCount : Nat
  rRegCount : Nat
  rdRegCount : Nat
  pRegCount : Nat
  hRegCount : Nat := 0

def Module.render (m : Module) : String := Id.run do
  let mut s := s!".version {m.version}\n.target {m.target}\n.address_size 64\n\n"
  for sd in m.sharedDecls do
    s := s ++ s!".shared .{sd.elemType} {sd.name}[{sd.count}];\n"
  if !m.sharedDecls.isEmpty then s := s ++ "\n"
  s := s ++ s!".entry {m.funcName}(\n"
  for i in [:m.params.size] do
    if i > 0 then s := s ++ ",\n"
    s := s ++ s!"  .param .u64 param_{m.params[i]!}"
  s := s ++ "\n)\n{\n"
  if m.fRegCount > 0  then s := s ++ s!"  .reg .f32 %f<{m.fRegCount}>;\n"
  if m.rRegCount > 0  then s := s ++ s!"  .reg .u32 %r<{m.rRegCount}>;\n"
  if m.rdRegCount > 0 then s := s ++ s!"  .reg .u64 %rd<{m.rdRegCount}>;\n"
  if m.pRegCount > 0  then s := s ++ s!"  .reg .pred %p<{m.pRegCount}>;\n"
  if m.hRegCount > 0  then s := s ++ s!"  .reg .b16 %h<{m.hRegCount}>;\n"
  s := s ++ "\n"
  for inst in m.body do s := s ++ inst.toString ++ "\n"
  s := s ++ "}\n"
  return s

-- ============================================================================
-- Code generation state
-- ============================================================================

structure GenState where
  fRegs : Nat := 0
  rRegs : Nat := 0
  rdRegs : Nat := 0
  pRegs : Nat := 0
  hRegs : Nat := 0   -- b16 (f16) registers
  labels : Nat := 0
  insts : Array Inst := #[]
  varMap : List (String × AnyReg) := []
  sharedNames : List String := []
  /-- Buffer names declared with u32 element type (for ld.global.u32) -/
  u32BufferNames : List String := []
  /-- Shared memory names with u32 element type (for ld/st.shared.u32) -/
  u32SharedNames : List String := []
  deriving Inhabited

namespace GenState

def freshF32 (s : GenState) : RegF32 × GenState :=
  (⟨s.fRegs⟩, { s with fRegs := s.fRegs + 1 })

def freshU32 (s : GenState) : RegU32 × GenState :=
  (⟨s.rRegs⟩, { s with rRegs := s.rRegs + 1 })

def freshU64 (s : GenState) : RegU64 × GenState :=
  (⟨s.rdRegs⟩, { s with rdRegs := s.rdRegs + 1 })

def freshPred (s : GenState) : RegPred × GenState :=
  (⟨s.pRegs⟩, { s with pRegs := s.pRegs + 1 })

def freshB16 (s : GenState) : RegB16 × GenState :=
  (⟨s.hRegs⟩, { s with hRegs := s.hRegs + 1 })

def freshLabel (s : GenState) : Label × GenState :=
  (⟨s.labels⟩, { s with labels := s.labels + 1 })

def emit (s : GenState) (inst : Inst) : GenState :=
  { s with insts := s.insts.push inst }

def bindVar (s : GenState) (name : String) (reg : AnyReg) : GenState :=
  { s with varMap := (name, reg) :: s.varMap }

def lookupVar (s : GenState) (name : String) : Option AnyReg :=
  s.varMap.find? (·.1 == name) |>.map (·.2)

def isSharedVar (s : GenState) (name : String) : Bool :=
  s.sharedNames.any (· == name)

def isU32Buffer (s : GenState) (name : String) : Bool :=
  s.u32BufferNames.any (· == name)

def isU32Shared (s : GenState) (name : String) : Bool :=
  s.u32SharedNames.any (· == name)

end GenState

end Hesper.CUDA.PTX
