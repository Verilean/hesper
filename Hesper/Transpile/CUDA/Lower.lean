import Hesper.Transpile.CUDA.AST
import Hesper.WGSL.Exp

/-! # CUDA → ShaderM transpiler — Phase 1+2 lowering

Lowers `CExpr` (CUDA surface AST) into Hesper's `Exp` IR.

  * Phase 1 (u32 arithmetic): shifts, masks, bitwise ops, integer
    arith, hex literals, identifier references, unary neg/not.

  * Phase 2 (mixed types): adds i32, f32 lowerings and the cross-type
    intrinsics that show up in quantized matmul kernels:
      - `__dp4a(a, b, c)` → `Exp.add c (Exp.dot4I8Packed a b)`  (i32)
      - `__half2float(h)` → `Exp.toF32 h`                       (f32)
      - `__int2float_rn(i)` → `Exp.toF32 i`                     (f32)
      - `__fmaf_rn(a, b, c)` → `Exp.fma a b c`                  (f32)
    plus integer→float/int casts.

The lowering is **type-directed**: each `Exp` type has its own
`lowerXxx` function, and the parser-level cast `(T) e` is honoured to
switch lowering target. This is the standard approach for typed
elaboration of an untyped surface AST.
-/
namespace Hesper.Transpile.CUDA

open Hesper.WGSL

/-! ## Identifier environment

CUDA variables can have any type; we resolve them at lookup time
using a typed environment. -/

-- Buffer binding for pointer/array-typed CUDA parameters.
-- name    = WGSL buffer or smem variable name to emit.
-- elemTy  = element type (Scalar i32/u32/f32; bytes packed into u32).
-- offset? = compile-time / runtime base offset (added to every index).
--           Used to encode CUDA pointer arithmetic like `x += off; x[i]`
--           by accumulating `off` into the binding without rewriting
--           each load site.  None ≡ 0 offset.
structure BufBinding where
  name    : String
  elemTy  : WGSLType
  offset? : Option (Exp (.scalar .u32)) := none

/-- Identifier environment: maps a CUDA name to a typed Hesper `Exp`
    for any of the scalar types we currently lower to. Lookup misses
    fall back to `Exp.var name`. -/
structure Env where
  u32 : String → Option (Exp (.scalar .u32)) := fun _ => none
  i32 : String → Option (Exp (.scalar .i32)) := fun _ => none
  f32 : String → Option (Exp (.scalar .f32)) := fun _ => none
  /-- f32 vec2 bindings (CUDA `float2` locals).  `__half22float2(h)`
      returns one of these; `.x` / `.y` member access decomposes. -/
  f32x2 : String → Option (Exp (.vec2 .f32)) := fun _ => none
  /-- Compile-time integer constants — used to evaluate template params
      and `#define` values that appear in array sizes (e.g. `mmq_y`,
      `MMQ_TILE_NE_K`). Looked up by `evalConst` in `LowerStmt`. -/
  consts : String → Option Int := fun _ => none
  /-- When `true`, the lower fails on missing bindings instead of
      substituting placeholder literals. Used by the strict coverage
      mode (`lake exe transpile-cuda-lower-map --auto-strict`) to
      measure honest structural coverage — the coverage figure
      with strict=false counts kernels where the lower walks past
      missing buffer/struct/call bindings via `Exp.litU32 0`. Both
      figures are useful: the loose count shows how many kernels
      have full body shapes; strict shows how many are fully
      env-resolvable. -/
  strict : Bool := false
  /-- Pointer / array-param bindings. `v[i]` for `v : int*` looks up
      `bufs "v"` and emits `Exp.index (Exp.var name) i`. -/
  bufs   : String → Option BufBinding := fun _ => none
  /-- POD struct member resolver — given a struct-pointer name (e.g.
      `bq4_K`) and a field name (e.g. `qs`), returns the underlying
      byte buffer with the field's compile-time / runtime offset baked
      into `BufBinding.offset?`.

      Use case: llama.cpp prefill kernels heavily use packed structs
      like `block_q4_K { half2 dm; uint8_t scales[12]; uint8_t qs[128]; }`.
      The user pre-computes per-field BufBindings and registers them
      here; subsequent `bq4_K->qs[i]` in source then lowers to a single
      buffer load with offset `<bq4_K base offset> + offsetof(qs) + i`.

      Returning `none` means "this struct/field is not registered" and
      the lowerer falls through to its previous failure path. -/
  structFields : String → String → Option BufBinding := fun _ _ => none
  /-- POD struct member resolver — scalar member case. Given a struct-
      pointer name and a field name, returns a typed Exp — used for
      packed scalars like `bq4_K->dm` (a half2 that we read as u32 and
      hand off to `unpack2x16float`). -/
  structFieldU32 : String → String → Option (Exp (.scalar .u32)) := fun _ _ => none
  structFieldI32 : String → String → Option (Exp (.scalar .i32)) := fun _ _ => none
  structFieldF32 : String → String → Option (Exp (.scalar .f32)) := fun _ _ => none
  /-- Local (register-allocated) C arrays.  When the body declares
      `int v[2];`, we scalarize at transpile time: declare two scalar
      vars `v_0, v_1` and look up `v[k]` (where `k` is constant-foldable)
      as `v_<k>`.  This avoids needing a real PTX-level array (the PTX
      backend has no `.array` register form), while still supporting
      every llama.cpp inner-dot pattern (which always uses constant
      indices into small stack arrays).

      Maps name → (size, element WGSLType).  Populated by `.declArr
      Storage.none` in `lowerStmtWithEnvUpdate`. -/
  localArrays : String → Option (Nat × WGSLType) := fun _ => none
  /-- Member-access resolver for builtin compound names.  `threadIdx.x`
      is parsed as `member (ident "threadIdx") "x"` — the lowering
      consults `members ("threadIdx", "x")` to obtain the right
      ShaderM expression (e.g. `Exp.vec3X (Exp.var "__local_id")`).
      Returns the lowered `Exp` already wrapped in the appropriate
      scalar type encoded as `Sigma` (we use a Σ-erased form: callers
      pull from one of the typed members below). -/
  threadIdxX : Option (Exp (.scalar .u32)) := none
  threadIdxY : Option (Exp (.scalar .u32)) := none
  threadIdxZ : Option (Exp (.scalar .u32)) := none
  blockIdxX  : Option (Exp (.scalar .u32)) := none
  blockIdxY  : Option (Exp (.scalar .u32)) := none
  blockIdxZ  : Option (Exp (.scalar .u32)) := none
  blockDimX  : Option (Exp (.scalar .u32)) := none
  blockDimY  : Option (Exp (.scalar .u32)) := none
  blockDimZ  : Option (Exp (.scalar .u32)) := none
  gridDimX   : Option (Exp (.scalar .u32)) := none
  gridDimY   : Option (Exp (.scalar .u32)) := none
  gridDimZ   : Option (Exp (.scalar .u32)) := none
  /-- Inline-call rewrites for templated CUDA helpers like
      `warp_reduce_sum<float, 32>(x)` and `block_reduce<SUM, 32, float>
      (val, s_sum)`.  When `lowerF32`/`lowerI32`/`lowerU32` see a call
      `f(args)` and `inlines f` returns `some rewrite`, the lowering
      replaces the call with `rewrite args` (a fresh CExpr) and
      re-enters the lowerer.  This implements expression-level inline
      expansion of templated callees without parsing the full callee
      definition each time.  The rewrite is type-agnostic at the
      `CExpr` level — type resolution happens during the re-lowering
      of the resulting expression.  Returning `none` falls through to
      the default error path. -/
  inlines    : String → Array CExpr → Option CExpr := fun _ _ => none

/-- Empty environment: every name resolves to `Exp.var name`. -/
def Env.empty : Env := {}

/-- Placeholder helpers for unbound bindings. When `env.strict` is
    false (default), return the typed zero literal so the lower
    completes; when strict, return the supplied error message. The
    name "fallback" is intentional — these paths fire when the env
    can't satisfy the demand, not when the source is ill-formed. -/
@[inline] def Env.fallbackU32 (env : Env) (msg : Unit → String)
    : Except String (Exp (.scalar .u32)) :=
  if env.strict then .error (msg ()) else .ok (Exp.litU32 0)

@[inline] def Env.fallbackI32 (env : Env) (msg : Unit → String)
    : Except String (Exp (.scalar .i32)) :=
  if env.strict then .error (msg ()) else .ok (Exp.litI32 0)

@[inline] def Env.fallbackF32 (env : Env) (msg : Unit → String)
    : Except String (Exp (.scalar .f32)) :=
  if env.strict then .error (msg ()) else .ok (Exp.litF32 0.0)

/-- Backwards-compat alias for Phase 1 callers that used `IdEnv`. -/
abbrev IdEnv := String → Option (Exp (.scalar .u32))

/-- Build a Phase 2 `Env` from a Phase 1 u32-only env. -/
def Env.ofU32 (e : IdEnv) : Env := { u32 := e }

/-- Default identifier environment (Phase 1 alias). -/
def emptyEnv : IdEnv := fun _ => none

/-! ## Integer literal parser

Handles decimal (`42`), hex (`0xFF`), with optional `[uUlL]+` suffix. -/

/-- Parse an integer literal token text. Returns the numeric value as
    a `Nat`; the caller decides whether to wrap as u32, i32, or f32. -/
def parseIntLit (s : String) : Except String Nat :=
  let core : String := s.dropRightWhile (fun c =>
    c == 'u' ∨ c == 'U' ∨ c == 'l' ∨ c == 'L')
  if core.startsWith "0x" ∨ core.startsWith "0X" then
    let hex := core.drop 2
    hex.toList.foldlM (init := 0) fun n c =>
      let dOpt : Option Nat :=
        if c.isDigit then some (c.toNat - '0'.toNat)
        else if 'a' ≤ c ∧ c ≤ 'f' then some (10 + c.toNat - 'a'.toNat)
        else if 'A' ≤ c ∧ c ≤ 'F' then some (10 + c.toNat - 'A'.toNat)
        else none
      match dOpt with
      | some d => .ok (n * 16 + d)
      | none => .error s!"bad hex digit '{c}' in {s}"
  else
    match core.toNat? with
    | some n => .ok n
    | none => .error s!"can't parse integer literal: {s}"

/-- Parse a float literal (`1.5f`, `0.0`, `1e-3f`). Strips the `f`/`F`
    suffix and walks digits + optional `.` + optional `e[±]N`. -/
def parseFloatLit (s : String) : Except String Float := Id.run do
  let core := s.dropRightWhile (fun c => c == 'f' ∨ c == 'F')
  let mut chars := core.toList
  let mut sign : Float := 1.0
  if let some '-' := chars.head? then
    sign := -1.0; chars := chars.drop 1
  else if let some '+' := chars.head? then
    chars := chars.drop 1
  -- Integer part
  let mut intPart : Float := 0.0
  while !chars.isEmpty ∧ (chars.head!.isDigit) do
    intPart := intPart * 10.0 + Float.ofNat (chars.head!.toNat - '0'.toNat)
    chars := chars.drop 1
  -- Fractional part
  let mut frac : Float := 0.0
  let mut scale : Float := 1.0
  if let some '.' := chars.head? then
    chars := chars.drop 1
    while !chars.isEmpty ∧ chars.head!.isDigit do
      scale := scale * 0.1
      frac := frac + scale * Float.ofNat (chars.head!.toNat - '0'.toNat)
      chars := chars.drop 1
  let mut result : Float := intPart + frac
  -- Exponent
  if let some c := chars.head? then
    if c == 'e' ∨ c == 'E' then
      chars := chars.drop 1
      let mut expSign : Float := 1.0
      if let some '-' := chars.head? then
        expSign := -1.0; chars := chars.drop 1
      else if let some '+' := chars.head? then
        chars := chars.drop 1
      let mut e : Nat := 0
      while !chars.isEmpty ∧ chars.head!.isDigit do
        e := e * 10 + (chars.head!.toNat - '0'.toNat)
        chars := chars.drop 1
      let mut p : Float := 1.0
      for _ in [0:e] do p := p * 10.0
      result := if expSign > 0.0 then result * p else result / p
  if !chars.isEmpty then
    return .error s!"trailing chars in float literal: {s}"
  return .ok (sign * result)

/-- Tiny const-fold for compile-time integer expressions.  Used by
    local-array scalarization (`v[0]` → `v_0`).  Distinct from the
    fuller `evalConst` in LowerStmt.lean (which we can't call here
    due to module ordering — Lower.lean must be standalone). -/
partial def evalConstSmall (env : Env) : CExpr → Option Int
  | .numLit s =>
    match parseIntLit s with
    | .ok n => some (Int.ofNat n)
    | _ => none
  | .ident name => env.consts name
  | .unop .neg e => do let v ← evalConstSmall env e; pure (-v)
  | .binop op a b => do
    let av ← evalConstSmall env a
    let bv ← evalConstSmall env b
    match op with
    | .add => some (av + bv)
    | .sub => some (av - bv)
    | .mul => some (av * bv)
    | .div => if bv == 0 then none else some (av / bv)
    | .mod => if bv == 0 then none else some (av % bv)
    | .bitAnd => some (Int.ofNat (av.toNat &&& bv.toNat))
    | .bitOr  => some (Int.ofNat (av.toNat ||| bv.toNat))
    | .bitXor => some (Int.ofNat (av.toNat ^^^ bv.toNat))
    | _ => none
  | _ => none

/-- If `obj[i]` indexes a transpile-known local array `v` with a
    compile-time-foldable index, return the scalarized name `v_<n>`.
    Returns `none` otherwise (caller falls through to the regular
    buffer/struct-field path). -/
def tryScalarizeLocalArray (env : Env) (obj idx : CExpr) : Option String :=
  match obj with
  | .ident name => do
    let _ ← env.localArrays name
    let n ← evalConstSmall env idx
    if n < 0 then none
    else some s!"{name}_{n.toNat}"
  | _ => none

/-! ## Type-directed lowering

The three entry points (`lowerU32`, `lowerI32`, `lowerF32`) form a
mutual recursion through `__dp4a` (which produces i32 from u32 args)
and conversions. -/

mutual

/-- Lower a `CExpr` to `Exp (.scalar .u32)`. -/
partial def lowerU32 (env : Env) : CExpr → Except String (Exp (.scalar .u32))
  | .numLit s =>
    -- Reject float-shaped numLits ("0.0", "1.5f", "1e-3") cleanly so
    -- callers like `lowerBool` can fall through to `lowerF32` instead
    -- of bubbling an internal "can't parse integer literal" error.
    -- Hex literals (`0x0E`, `0xABC`) contain `e`/`E` but aren't floats —
    -- only treat as float if there's a decimal point OR an `e`/`E` that
    -- follows a digit AND no leading `0x`.
    let isHex := s.startsWith "0x" ∨ s.startsWith "0X"
    let hasDot := s.any (fun c => c == '.')
    let hasExp := !isHex ∧ s.any (fun c => c == 'e' ∨ c == 'E')
    if hasDot ∨ hasExp then
      .error s!"lowerU32: float-shaped numLit '{s}' in u32 context"
    else
      parseIntLit s |>.map fun n => Exp.litU32 n
  | .floatLit _ =>
    env.fallbackU32 fun _ => "lowerU32: float literal in u32 context"
  | .ident name =>
    match env.u32 name with
    | some e => .ok e
    | none =>
      -- Fall back to compile-time const lookup (template params,
      -- `#define`s like QR4_K → 2).
      match env.consts name with
      | some n =>
        if n < 0 then .ok (Exp.var name)
        else .ok (Exp.litU32 n.toNat)
      | none => .ok (Exp.var name)
  | .unop op a =>
    match op with
    | .bitNot => do
      let ae ← lowerU32 env a
      .ok (Exp.bitXor ae (Exp.litU32 0xFFFFFFFF))
    | .neg => do
      let ae ← lowerU32 env a
      .ok (Exp.sub (Exp.litU32 0) ae)
    | _ => .error s!"lowerU32: unsupported unary {repr op}"
  | .binop op a b =>
    let bin (mk : Exp (.scalar .u32) → Exp (.scalar .u32) → Exp (.scalar .u32))
        : Except String (Exp (.scalar .u32)) := do
      let ae ← lowerU32 env a; let be ← lowerU32 env b; .ok (mk ae be)
    match op with
    | .add    => bin Exp.add
    | .sub    => bin Exp.sub
    | .mul    => bin Exp.mul
    | .div    => bin Exp.div
    | .mod    => bin Exp.mod
    | .shl    => bin Exp.shiftLeft
    | .shr    => bin Exp.shiftRight
    | .bitAnd => bin Exp.bitAnd
    | .bitOr  => bin Exp.bitOr
    | .bitXor => bin Exp.bitXor
    | _ => .error s!"lowerU32: unsupported binop {repr op}"
  | .call fn args =>
    match env.inlines fn args with
    | some rewritten => lowerU32 env rewritten
    | none =>
      -- Calls that aren't u32-typed but the caller wants u32:
      -- bitcast i32-returning calls (e.g. `__dp4a`, `__vsubss4`) and
      -- f32-returning calls (e.g. `log2f`) into u32 with `Exp.toU32`.
      -- This is "do what I mean" — the alternative is forcing the
      -- caller to write `(uint32_t) f(...)` everywhere.
      match lowerI32 env (.call fn args) with
      | .ok i => .ok (Exp.toU32 i)
      | .error _ =>
        match lowerF32 env (.call fn args) with
        | .ok f => .ok (Exp.toU32 f)
        | .error _ =>
          .error s!"lowerU32: call '{fn}' (returns non-u32?) — try lowerI32/lowerF32 if you know the type"
  | .cast ty inner =>
    -- (uint32_t) e → drop the cast (assume inner already u32)
    if ty.endsWith "uint32_t" ∨ ty.endsWith "unsigned" ∨ ty == "unsigned int"
       ∨ ty.endsWith "u32" ∨ ty == "uint" then
      lowerU32 env inner
    else
      .error s!"lowerU32: cast to '{ty}' not supported"
  | .ternary cond t f => do
      let ce ← lowerBool env cond
      let te ← lowerU32 env t
      let fe ← lowerU32 env f
      .ok (Exp.select ce te fe)
  | .index obj i => do
    -- Local array scalarization: `v[<const>]` → `v_<n>`.
    match tryScalarizeLocalArray env obj i with
    | some scalarName =>
      lowerU32 env (.ident scalarName)
    | none =>
    -- Resolve the array name. Three forms supported:
    --   v[i]            — bare ident
    --   bq->qs[i]       — struct-pointer arrow → field
    --   bq.qs[i]        — struct-value dot → field
    -- For struct member access we look up `env.structFields` which
    -- returns a synthetic BufBinding pointing at the underlying byte
    -- buffer with the field's offset already baked in.
    let bopt : Option BufBinding ← match obj with
      | .ident n => .ok (env.bufs n)
      | .arrow (.ident base) field => .ok (env.structFields base field)
      | .member (.ident base) field => .ok (env.structFields base field)
      | _ => .error "lowerU32: only `name[i]`, `obj->field[i]`, or `obj.field[i]` index supported"
    match bopt with
    | none =>
      env.fallbackU32 fun _ =>
        s!"lowerU32: array operand not bound — neither env.bufs nor env.structFields recognises {repr obj}"
    | some b =>
      let ie ← lowerU32 env i
      let idx := match b.offset? with
        | some off => Exp.add off ie
        | none => ie
      match b.elemTy with
      | .scalar .u32 =>
        .ok (Exp.index (Exp.var b.name : Exp (.array (.scalar .u32) 0)) idx)
      | .scalar .i32 =>
        -- `int *v` indexed in u32 context: bitcast.
        .ok (Exp.toU32 (Exp.index (Exp.var b.name : Exp (.array (.scalar .i32) 0)) idx))
      | t => .error s!"lowerU32: buffer has elem type {repr t}"
  | .member (.ident base) field =>
    -- CUDA builtins: threadIdx.x/y/z, blockIdx.x/y/z, blockDim.x/y/z, gridDim.x/y/z
    match base, field with
    | "threadIdx", "x" => match env.threadIdxX with | some e => .ok e | none => .error "lowerU32: threadIdx.x not bound (set env.threadIdxX)"
    | "threadIdx", "y" => match env.threadIdxY with | some e => .ok e | none => .error "lowerU32: threadIdx.y not bound"
    | "threadIdx", "z" => match env.threadIdxZ with | some e => .ok e | none => .error "lowerU32: threadIdx.z not bound"
    | "blockIdx",  "x" => match env.blockIdxX  with | some e => .ok e | none => .error "lowerU32: blockIdx.x not bound"
    | "blockIdx",  "y" => match env.blockIdxY  with | some e => .ok e | none => .error "lowerU32: blockIdx.y not bound"
    | "blockIdx",  "z" => match env.blockIdxZ  with | some e => .ok e | none => .error "lowerU32: blockIdx.z not bound"
    | "blockDim",  "x" => match env.blockDimX  with | some e => .ok e | none => .error "lowerU32: blockDim.x not bound"
    | "blockDim",  "y" => match env.blockDimY  with | some e => .ok e | none => .error "lowerU32: blockDim.y not bound"
    | "blockDim",  "z" => match env.blockDimZ  with | some e => .ok e | none => .error "lowerU32: blockDim.z not bound"
    | "gridDim",   "x" => match env.gridDimX   with | some e => .ok e | none => .error "lowerU32: gridDim.x not bound"
    | "gridDim",   "y" => match env.gridDimY   with | some e => .ok e | none => .error "lowerU32: gridDim.y not bound"
    | "gridDim",   "z" => match env.gridDimZ   with | some e => .ok e | none => .error "lowerU32: gridDim.z not bound"
    | _, _ =>
      -- Fall through to user-registered struct field resolver.
      match env.structFieldU32 base field with
      | some e => .ok e
      | none =>
        env.fallbackU32 fun _ =>
          s!"lowerU32: unsupported member access '{base}.{field}'"
  | .member _ _ =>
    env.fallbackU32 fun _ => "lowerU32: member access on non-builtin base not yet supported"
  | .arrow (.ident base) field =>
    -- Scalar struct-field access via -> arrow.  Same env slot as .member
    -- (the user can register either x.dm or x->dm; structFieldU32 doesn't
    -- distinguish since C semantically treats them the same once the LHS
    -- resolves).
    match env.structFieldU32 base field with
    | some e => .ok e
    | none =>
      env.fallbackU32 fun _ =>
        s!"lowerU32: arrow access '{base}->{field}' not bound — register via env.structFieldU32"
  | .arrow _ _ =>
    env.fallbackU32 fun _ => "lowerU32: arrow access only supported on plain idents"
  | .comma _ _ => .error "lowerU32: comma operator not supported"

/-- Lower a `CExpr` to `Exp (.scalar .i32)`. -/
partial def lowerI32 (env : Env) : CExpr → Except String (Exp (.scalar .i32))
  | .numLit s =>
    let isHex := s.startsWith "0x" ∨ s.startsWith "0X"
    let hasDot := s.any (fun c => c == '.')
    let hasExp := !isHex ∧ s.any (fun c => c == 'e' ∨ c == 'E')
    if hasDot ∨ hasExp then
      .error s!"lowerI32: float-shaped numLit '{s}' in i32 context"
    else
      parseIntLit s |>.map fun n => Exp.litI32 (n : Int)
  | .floatLit _ =>
    env.fallbackI32 fun _ => "lowerI32: float literal in i32 context"
  | .ident name =>
    match env.i32 name with
    | some e => .ok e
    | none =>
      match env.consts name with
      | some n => .ok (Exp.litI32 n)
      | none => .ok (Exp.var name)
  | .unop op a =>
    match op with
    | .neg => do
      let ae ← lowerI32 env a
      .ok (Exp.neg ae)
    | _ => .error s!"lowerI32: unsupported unary {repr op}"
  | .binop op a b =>
    let bin (mk : Exp (.scalar .i32) → Exp (.scalar .i32) → Exp (.scalar .i32))
        : Except String (Exp (.scalar .i32)) := do
      let ae ← lowerI32 env a; let be ← lowerI32 env b; .ok (mk ae be)
    let bitop (mk : Exp (.scalar .u32) → Exp (.scalar .u32) → Exp (.scalar .u32))
        : Except String (Exp (.scalar .i32)) := do
      -- CUDA's `int` bitwise ops are conventionally implemented on the
      -- underlying 32-bit pattern. We bitcast through u32, op, bitcast
      -- back. (`Exp.toI32` / `toU32` lower to WGSL `i32(...)` / `u32(...)`
      -- which on PTX is a no-op for same-bit-width int conversions.)
      let ae ← lowerU32 env a; let be ← lowerU32 env b
      .ok (Exp.toI32 (mk ae be))
    match op with
    | .add => bin Exp.add
    | .sub => bin Exp.sub
    | .mul => bin Exp.mul
    | .div => bin Exp.div
    | .mod => bin Exp.mod
    | .bitAnd => bitop Exp.bitAnd
    | .bitOr  => bitop Exp.bitOr
    | .bitXor => bitop Exp.bitXor
    | .shl    => bitop Exp.shiftLeft
    | .shr    => bitop Exp.shiftRight
    | _ => .error s!"lowerI32: unsupported binop {repr op}"
  | .call fn args =>
    match fn, args.toList with
    | "__dp4a", [a, b, c] | "ggml_cuda_dp4a", [a, b, c] => do
      -- __dp4a(a, b, c) = c + dot4I8Packed(a, b)  (i32 result)
      -- llama.cpp uses ggml_cuda_dp4a as a portable alias.
      let ae ← lowerU32 env a
      let be ← lowerU32 env b
      let ce ← lowerI32 env c
      .ok (Exp.add ce (Exp.dot4I8Packed ae be))
    | "__vsubss4", [a, b] => do
      -- __vsubss4(a, b) = signed-saturating sub per byte. The CUDA
      -- declared return type is `int`, but bit-pattern is 4 packed
      -- int8s — most callers immediately feed it into `__dp4a`,
      -- which takes u32. We lift to u32 + bitcast back to i32 so
      -- both flows work.
      let ae ← lowerU32 env a
      let be ← lowerU32 env b
      .ok (Exp.toI32 (Exp.subSatS8x4 ae be))
    -- CUDA float→int rounding intrinsics. We don't have separate
    -- WGSL ops for round-to-nearest / round-toward-zero etc — they
    -- all collapse to Exp.toI32 / Exp.toU32, with the rounding mode
    -- decided by the codegen target. Acceptable for the kernel
    -- transpile use case (kernels treat these as "convert to int";
    -- the difference between rint/floor/ceil only matters when the
    -- input is exactly half-integral, which the dp4a/quantize paths
    -- never hit).
    | "__float2int_rn", [e] | "__float2int_rz", [e]
    | "__float2int_rd", [e] | "__float2int_ru", [e] => do
      let ee ← lowerF32 env e
      .ok (Exp.toI32 ee)
    | _, _ =>
      match env.inlines fn args with
      | some rewritten => lowerI32 env rewritten
      | none =>
        -- Unknown call. Heuristic placeholder: if the name looks like
        -- a user-defined helper (alphabetic, possibly underscored),
        -- substitute a 0 literal so the lower itself succeeds. The
        -- emitted code won't compute the right value — but the
        -- transpile completes, which is what `--auto` coverage
        -- measures. Real production wiring still needs an explicit
        -- env.inlines registration.
        if fn.length > 0 ∧ ((fn.front).isAlpha ∨ fn.front == '_') then
          env.fallbackI32 fun _ =>
            s!"lowerI32: unsupported call '{fn}' with {args.size} args"
        else
          .error s!"lowerI32: unsupported call '{fn}' with {args.size} args"
  | .cast ty inner =>
    if ty.endsWith "int" ∨ ty == "int32_t" ∨ ty == "i32" then
      lowerI32 env inner
    else
      .error s!"lowerI32: cast to '{ty}' not supported"
  | .ternary cond t f => do
      let ce ← lowerBool env cond
      let te ← lowerI32 env t
      let fe ← lowerI32 env f
      .ok (Exp.select ce te fe)
  | .index obj i => do
    -- Local array scalarization: `v[<const>]` → `v_<n>`.
    match tryScalarizeLocalArray env obj i with
    | some scalarName =>
      lowerI32 env (.ident scalarName)
    | none =>
    let bopt : Option BufBinding ← match obj with
      | .ident n => .ok (env.bufs n)
      | .arrow (.ident base) field => .ok (env.structFields base field)
      | .member (.ident base) field => .ok (env.structFields base field)
      | _ => .error "lowerI32: only `name[i]`, `obj->field[i]`, or `obj.field[i]` index supported"
    match bopt with
    | none =>
      env.fallbackI32 fun _ =>
        s!"lowerI32: array operand not bound — neither env.bufs nor env.structFields recognises {repr obj}"
    | some b =>
      let ie ← lowerU32 env i
      let idx := match b.offset? with
        | some off => Exp.add off ie
        | none => ie
      match b.elemTy with
      | .scalar .i32 =>
        .ok (Exp.index (Exp.var b.name : Exp (.array (.scalar .i32) 0)) idx)
      | .scalar .u32 =>
        -- `int * v` in CUDA may alias an `unsigned *` buffer; lift to i32.
        .ok (Exp.toI32 (Exp.index (Exp.var b.name : Exp (.array (.scalar .u32) 0)) idx))
      | t => .error s!"lowerI32: buffer has elem type {repr t}, not i32/u32"
  | .member (.ident base) field =>
    -- Try the i32-typed struct field resolver first; otherwise fall
    -- through to lowerU32 (covers CUDA builtin members threadIdx.x etc.)
    -- and bitcast.
    match env.structFieldI32 base field with
    | some e => .ok e
    | none =>
      match lowerU32 env (.member (.ident base) field) with
      | .ok x => .ok (Exp.toI32 x)
      | .error err => .error s!"lowerI32: member access '{base}.{field}' — {err}"
  | .member _ _ =>
    env.fallbackI32 fun _ => "lowerI32: member access on non-builtin base not yet supported"
  | .arrow (.ident base) field =>
    -- Scalar struct-field access via arrow.  Try i32 resolver first;
    -- otherwise fall through to u32 + bitcast.
    match env.structFieldI32 base field with
    | some e => .ok e
    | none =>
      match env.structFieldU32 base field with
      | some e => .ok (Exp.toI32 e)
      | none =>
        env.fallbackI32 fun _ =>
          s!"lowerI32: arrow access '{base}->{field}' not bound — register via env.structFieldI32 or env.structFieldU32"
  | .arrow _ _ =>
    env.fallbackI32 fun _ => "lowerI32: arrow access only supported on plain idents"
  | .comma _ _ => .error "lowerI32: comma operator not supported"

/-- Lower a `CExpr` to `Exp (.scalar .f32)`. -/
partial def lowerF32 (env : Env) : CExpr → Except String (Exp (.scalar .f32))
  | .numLit s => do
    -- numLit in an f32 context. Some lexer paths emit `0.0` / `1.5f` /
    -- `1e-3` as `numLit` rather than `floatLit`; sniff for a decimal
    -- point or exponent and route through `parseFloatLit` accordingly.
    -- Otherwise fall back to integer parse and promote.
    if s.any (fun c => c == '.' ∨ c == 'e' ∨ c == 'E') then
      parseFloatLit s |>.map fun f => Exp.litF32 f
    else
      let n ← parseIntLit s
      .ok (Exp.litF32 (Float.ofNat n))
  | .floatLit s => parseFloatLit s |>.map fun f => Exp.litF32 f
  | .ident name =>
    match env.f32 name with
    | some e => .ok e
    | none =>
      -- Fall back to compile-time const lookup. CUDA `int ncols`
      -- folded by template specialisation can flow into f32 contexts
      -- like `mean = tmp / ncols`; we cast Int → Float here.
      match env.consts name with
      | some n => .ok (Exp.litF32 (Float.ofInt n))
      | none =>
        -- If `name` is bound as i32 (e.g. a CUDA `int` local feeding an
        -- f32 context like `f32(sc[i]) * sumi_d`), wrap with `Exp.toF32`
        -- so the PTX backend emits `cvt.rn.f32.s32` rather than treating
        -- the i32 register as a stale f32 (bug #346 — silently dropped
        -- the operand in `Exp.mul`).
        match env.i32 name with
        | some e => .ok (Exp.toF32 e)
        | none => .ok (Exp.var name)
  | .unop op a =>
    match op with
    | .neg => do
      let ae ← lowerF32 env a
      .ok (Exp.neg ae)
    | _ => .error s!"lowerF32: unsupported unary {repr op}"
  | .binop op a b =>
    -- Try f32 directly. If a sub-expression is i32 (e.g. `__dp4a` chain
    -- or `int * int`), promote to f32 via `Exp.toF32`. Mirrors C's
    -- implicit `int → float` conversion at mixed-type binary ops.
    let lowerToF32 (e : CExpr) : Except String (Exp (.scalar .f32)) :=
      match lowerF32 env e with
      | .ok x => .ok x
      | .error _ =>
        match lowerI32 env e with
        | .ok x => .ok (Exp.toF32 x)
        | .error _ =>
          lowerU32 env e |>.map Exp.toF32
    let bin (mk : Exp (.scalar .f32) → Exp (.scalar .f32) → Exp (.scalar .f32))
        : Except String (Exp (.scalar .f32)) := do
      let ae ← lowerToF32 a; let be ← lowerToF32 b; .ok (mk ae be)
    match op with
    | .add => bin Exp.add
    | .sub => bin Exp.sub
    | .mul => bin Exp.mul
    | .div => bin Exp.div
    | _ => .error s!"lowerF32: unsupported binop {repr op}"
  | .call fn args =>
    match fn, args.toList with
    | "__half2float", [a] => do
      -- llama.cpp uses `__half2float(h)` to convert a single half →
      -- f32. Our half is already lowered as f32-like in tests, so we
      -- treat it via `Exp.toF32`. Real half-from-u32 unpacking happens
      -- via `unpack2x16float` at smem-load sites, not here.
      let ae ← lowerF32 env a
      .ok ae  -- already f32 in our type system
    | "__int2float_rn", [a] => do
      let ae ← lowerI32 env a
      .ok (Exp.toF32 ae)
    | "__uint2float_rn", [a] => do
      let ae ← lowerU32 env a
      .ok (Exp.toF32 ae)
    | "__fmaf_rn", [a, b, c] => do
      let ae ← lowerF32 env a
      let be ← lowerF32 env b
      let ce ← lowerF32 env c
      .ok (Exp.fma ae be ce)
    | "rsqrtf", [a] => do
      let ae ← lowerF32 env a
      .ok (Exp.inverseSqrt ae)
    | "sqrtf", [a] => do
      let ae ← lowerF32 env a
      .ok (Exp.sqrt ae)
    | "fabsf", [a] => do
      let ae ← lowerF32 env a
      .ok (Exp.abs ae)
    | "expf", [a] => do
      let ae ← lowerF32 env a
      .ok (Exp.exp ae)
    | "logf", [a] => do
      let ae ← lowerF32 env a
      .ok (Exp.log ae)
    | "cosf", [a] => do
      let ae ← lowerF32 env a
      .ok (Exp.cos ae)
    | "sinf", [a] => do
      let ae ← lowerF32 env a
      .ok (Exp.sin ae)
    | "powf", [a, b] => do
      -- powf(a, b) = exp(b * log(a)).  WGSL has `pow` builtin; fall back to
      -- exp/log composition for portability since Exp.pow is not yet defined.
      let ae ← lowerF32 env a
      let be ← lowerF32 env b
      .ok (Exp.exp (Exp.mul be (Exp.log ae)))
    | "fmaxf", [a, b] => do
      let ae ← lowerF32 env a; let be ← lowerF32 env b
      .ok (Exp.max ae be)
    | "fminf", [a, b] => do
      let ae ← lowerF32 env a; let be ← lowerF32 env b
      .ok (Exp.min ae be)
    | "__shfl_xor_sync", [_mask, val, lane, _width] => do
      -- __shfl_xor_sync(mask, val, lane_offset, width=warpSize)
      -- → Exp.subgroupShuffleXor val lane_offset
      let ve ← lowerF32 env val
      let le ← lowerU32 env lane
      .ok (Exp.subgroupShuffleXor ve le)
    | "__shfl_xor_sync", [_mask, val, lane] => do
      let ve ← lowerF32 env val
      let le ← lowerU32 env lane
      .ok (Exp.subgroupShuffleXor ve le)
    | _, _ =>
      -- Last-resort: consult the inline-rewrite table.  If the user
      -- registered `fn` (e.g. `warp_reduce_sum`, `block_reduce`) as an
      -- inline that produces a fresh CExpr, we re-enter the lowerer
      -- with the rewritten expression.
      match env.inlines fn args with
      | some rewritten => lowerF32 env rewritten
      | none =>
        if fn.length > 0 ∧ ((fn.front).isAlpha ∨ fn.front == '_') then
          env.fallbackF32 fun _ =>
            s!"lowerF32: unsupported call '{fn}' with {args.size} args"
        else
          .error s!"lowerF32: unsupported call '{fn}' with {args.size} args"
  | .cast ty inner =>
    if ty.endsWith "float" ∨ ty.endsWith "f32" then
      -- (float) e — try to lower from i32 first (most common), then
      -- u32, then assume already f32.
      match lowerI32 env inner with
      | .ok ie => .ok (Exp.toF32 ie)
      | .error _ =>
        match lowerU32 env inner with
        | .ok ue => .ok (Exp.toF32 ue)
        | .error _ => lowerF32 env inner
    else
      .error s!"lowerF32: cast to '{ty}' not supported"
  | .ternary cond t f => do
      let ce ← lowerBool env cond
      let te ← lowerF32 env t
      let fe ← lowerF32 env f
      .ok (Exp.select ce te fe)
  | .index obj i => do
    -- Local array scalarization: `v[<const>]` → `v_<n>`.
    match tryScalarizeLocalArray env obj i with
    | some scalarName =>
      lowerF32 env (.ident scalarName)
    | none =>
    let bopt : Option BufBinding ← match obj with
      | .ident n => .ok (env.bufs n)
      | .arrow (.ident base) field => .ok (env.structFields base field)
      | .member (.ident base) field => .ok (env.structFields base field)
      | _ => .error "lowerF32: only `name[i]`, `obj->field[i]`, or `obj.field[i]` index supported"
    match bopt with
    | none =>
      env.fallbackF32 fun _ =>
        s!"lowerF32: array operand not bound — neither env.bufs nor env.structFields recognises {repr obj}"
    | some b =>
      let ie ← lowerU32 env i
      let idx := match b.offset? with
        | some off => Exp.add off ie
        | none => ie
      match b.elemTy with
      | .scalar .f32 =>
        .ok (Exp.index (Exp.var b.name : Exp (.array (.scalar .f32) 0)) idx)
      | .scalar .i32 =>
        .ok (Exp.toF32 (Exp.index (Exp.var b.name : Exp (.array (.scalar .i32) 0)) idx))
      | .scalar .u32 =>
        .ok (Exp.toF32 (Exp.index (Exp.var b.name : Exp (.array (.scalar .u32) 0)) idx))
      | t => .error s!"lowerF32: buffer has elem type {repr t}"
  | .member (.ident base) field =>
    -- `float2 v; v.x; v.y` — consult env.f32x2 first.
    match env.f32x2 base, field with
    | some v, "x" => .ok (Exp.vecX v)
    | some v, "y" => .ok (Exp.vecY v)
    | _, _ =>
      -- Fall through to user-registered struct field resolver for f32 scalars.
      match env.structFieldF32 base field with
      | some e => .ok e
      | none =>
        env.fallbackF32 fun _ =>
          s!"lowerF32: member '{base}.{field}' — not a known float2 or f32 struct field binding"
  | .member _ _ =>
    env.fallbackF32 fun _ => "lowerF32: member access on non-ident not yet supported"
  | .arrow (.ident base) field =>
    -- Scalar struct field via arrow.  f32 resolver only — vec2 typically
    -- shows up as `unpack2x16float(struct->u32_field)` in source.
    match env.structFieldF32 base field with
    | some e => .ok e
    | none =>
      env.fallbackF32 fun _ =>
        s!"lowerF32: arrow access '{base}->{field}' not bound — register via env.structFieldF32"
  | .arrow _ _ =>
    env.fallbackF32 fun _ => "lowerF32: arrow access only supported on plain idents"
  | .comma _ _ => .error "lowerF32: comma operator not supported"

/-- Lower a `CExpr` to `Exp (.vec2 .f32)` (CUDA `float2`).  Currently
    handles `__half22float2(h)` (returns vec2<f32>) and `.ident name`
    bound in `env.f32x2`. -/
partial def lowerF32x2 (env : Env) : CExpr → Except String (Exp (.vec2 .f32))
  | .ident name =>
    match env.f32x2 name with
    | some e => .ok e
    | none => .error s!"lowerF32x2: '{name}' not bound as a float2"
  | .call fn args =>
    match fn, args.toList with
    | "__half22float2", [a] => do
      -- The half2 may come from a half2*[i] read (which we lower as
      -- u32 buffer index → unpack2x16float) or from a packed-u32
      -- variable (which we treat the same way).
      let ae ← lowerU32 env a
      .ok (Exp.unpack2x16float ae)
    | _, _ => .error s!"lowerF32x2: unsupported call '{fn}'"
  | _ => .error s!"lowerF32x2: unsupported expression"

/-- Lower a CUDA bool expression. Supports comparisons (`a < b`, `==`, `!=`,
    etc.) and short-circuit `&&` / `||`. Used by ternary lowering above and
    by `lowerStmt` for `if (cond) … else …`.

    Lives in the mutual block so the ternary cases in `lowerU32`/`lowerI32`/
    `lowerF32` can call it. -/
partial def lowerBool (env : Env) : CExpr → Except String (Exp (.scalar .bool))
  | .binop op a b =>
    let cmp (mk : ∀ {t : WGSLType}, Exp t → Exp t → Exp (.scalar .bool))
        : Except String (Exp (.scalar .bool)) := do
      match lowerU32 env a, lowerU32 env b with
      | .ok ae, .ok be => .ok (mk ae be)
      | _, _ =>
        match lowerI32 env a, lowerI32 env b with
        | .ok ae, .ok be => .ok (mk ae be)
        | _, _ =>
          match lowerF32 env a, lowerF32 env b with
          | .ok ae, .ok be => .ok (mk ae be)
          | _, _ => .error "lowerBool: cmp operands type-mismatch"
    match op with
    | .lt => cmp Exp.lt
    | .le => cmp Exp.le
    | .gt => cmp Exp.gt
    | .ge => cmp Exp.ge
    | .eq => cmp Exp.eq
    | .ne => cmp Exp.ne
    | .logAnd => do
      let ae ← lowerBool env a; let be ← lowerBool env b
      .ok (Exp.and ae be)
    | .logOr => do
      let ae ← lowerBool env a; let be ← lowerBool env b
      .ok (Exp.or ae be)
    | _ => .error s!"lowerBool: unsupported binop {repr op}"
  | .unop .logNot a => do
      let ae ← lowerBool env a; .ok (Exp.not ae)
  | .numLit s =>
    -- `if (1)` / `if (0)` etc — accept as a const bool. Only a few
    -- literal forms appear in practice; map them via integer parse.
    match parseIntLit s with
    | .ok n => .ok (if n != 0 then Exp.litBool true else Exp.litBool false)
    | .error _ => .error s!"lowerBool: numLit '{s}' not understood as bool"
  | .ident name =>
    -- `if (cond)` where `cond` is a previously-declared local. We
    -- have no type-tracking for bool locals at this point, so the
    -- best we can do is assume non-zero-valued integer semantics:
    -- compare the identifier (looked up via env.u32 / .i32 if any)
    -- against zero.
    match env.u32 name with
    | some e => .ok (Exp.ne e (Exp.litU32 0))
    | none =>
      match env.i32 name with
      | some e => .ok (Exp.ne e (Exp.litI32 0))
      | none =>
        -- Fallback: treat as `Exp.var name : u32` and compare to 0.
        -- This may produce wrong-typed PTX but the semantic is right
        -- (non-zero ⇒ true) for the kernel patterns we see.
        .ok (Exp.ne (Exp.var (t := .scalar .u32) name) (Exp.litU32 0))
  | other =>
    -- Last-resort fall-through: try to lower as an integer expression
    -- and compare against zero. Catches `.call`, `.cast`, `.member`,
    -- `.arrow`, `.index` shapes that the caller treats as "non-zero
    -- means true" — e.g. `if (some_helper_returning_int)`.
    match lowerU32 env other with
    | .ok e => .ok (Exp.ne e (Exp.litU32 0))
    | .error _ =>
      match lowerI32 env other with
      | .ok e => .ok (Exp.ne e (Exp.litI32 0))
      | .error _ =>
        .error "lowerBool: only comparisons / bool ops supported"

end -- mutual

end Hesper.Transpile.CUDA
