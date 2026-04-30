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
  /-- Compile-time integer constants — used to evaluate template params
      and `#define` values that appear in array sizes (e.g. `mmq_y`,
      `MMQ_TILE_NE_K`). Looked up by `evalConst` in `LowerStmt`. -/
  consts : String → Option Int := fun _ => none
  /-- Pointer / array-param bindings. `v[i]` for `v : int*` looks up
      `bufs "v"` and emits `Exp.index (Exp.var name) i`. -/
  bufs   : String → Option BufBinding := fun _ => none
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

/-! ## Type-directed lowering

The three entry points (`lowerU32`, `lowerI32`, `lowerF32`) form a
mutual recursion through `__dp4a` (which produces i32 from u32 args)
and conversions. -/

mutual

/-- Lower a `CExpr` to `Exp (.scalar .u32)`. -/
partial def lowerU32 (env : Env) : CExpr → Except String (Exp (.scalar .u32))
  | .numLit s => parseIntLit s |>.map fun n => Exp.litU32 n
  | .floatLit _ => .error "lowerU32: float literal in u32 context"
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
    | .shl    => bin Exp.shiftLeft
    | .shr    => bin Exp.shiftRight
    | .bitAnd => bin Exp.bitAnd
    | .bitOr  => bin Exp.bitOr
    | .bitXor => bin Exp.bitXor
    | _ => .error s!"lowerU32: unsupported binop {repr op}"
  | .call fn args =>
    match env.inlines fn args with
    | some rewritten => lowerU32 env rewritten
    | none => .error s!"lowerU32: call '{fn}' (returns non-u32?) — try lowerI32/lowerF32 if you know the type"
  | .cast ty inner =>
    -- (uint32_t) e → drop the cast (assume inner already u32)
    if ty.endsWith "uint32_t" ∨ ty.endsWith "unsigned" ∨ ty == "unsigned int"
       ∨ ty.endsWith "u32" ∨ ty == "uint" then
      lowerU32 env inner
    else
      .error s!"lowerU32: cast to '{ty}' not supported"
  | .ternary _ _ _ => .error "lowerU32: ternary not yet supported"
  | .index obj i => do
    let name ← match obj with
      | .ident n => .ok n
      | _ => .error "lowerU32: only `name[i]` index supported (no nested arrays)"
    match env.bufs name with
    | none => .error s!"lowerU32: '{name}' not bound as a buffer; add it to env.bufs"
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
      | t => .error s!"lowerU32: buffer '{name}' has elem type {repr t}"
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
    | _, _ => .error s!"lowerU32: unsupported member access '{base}.{field}'"
  | .member _ _ => .error "lowerU32: member access on non-builtin not yet supported"
  | .arrow _ _ => .error "lowerU32: arrow access not yet supported"
  | .comma _ _ => .error "lowerU32: comma operator not supported"

/-- Lower a `CExpr` to `Exp (.scalar .i32)`. -/
partial def lowerI32 (env : Env) : CExpr → Except String (Exp (.scalar .i32))
  | .numLit s =>
    parseIntLit s |>.map fun n => Exp.litI32 (n : Int)
  | .floatLit _ => .error "lowerI32: float literal in i32 context"
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
    | .bitAnd => bitop Exp.bitAnd
    | .bitOr  => bitop Exp.bitOr
    | .bitXor => bitop Exp.bitXor
    | .shl    => bitop Exp.shiftLeft
    | .shr    => bitop Exp.shiftRight
    | _ => .error s!"lowerI32: unsupported binop {repr op}"
  | .call fn args =>
    match fn, args.toList with
    | "__dp4a", [a, b, c] => do
      -- __dp4a(a, b, c) = c + dot4I8Packed(a, b)  (i32 result)
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
    | _, _ =>
      match env.inlines fn args with
      | some rewritten => lowerI32 env rewritten
      | none => .error s!"lowerI32: unsupported call '{fn}' with {args.size} args"
  | .cast ty inner =>
    if ty.endsWith "int" ∨ ty == "int32_t" ∨ ty == "i32" then
      lowerI32 env inner
    else
      .error s!"lowerI32: cast to '{ty}' not supported"
  | .ternary _ _ _ => .error "lowerI32: ternary not yet supported"
  | .index obj i => do
    let name ← match obj with
      | .ident n => .ok n
      | _ => .error "lowerI32: only `name[i]` index supported (no nested arrays)"
    match env.bufs name with
    | none => .error s!"lowerI32: '{name}' not bound as a buffer; add it to env.bufs"
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
      | t => .error s!"lowerI32: buffer '{name}' has elem type {repr t}, not i32/u32"
  | .member (.ident base) field =>
    -- Try lowering as u32 then bitcast to i32. The u32 lowering covers
    -- all CUDA builtin members (threadIdx.x, etc.).
    match lowerU32 env (.member (.ident base) field) with
    | .ok x => .ok (Exp.toI32 x)
    | .error err => .error s!"lowerI32: member access '{base}.{field}' — {err}"
  | .member _ _ => .error "lowerI32: member access on non-builtin not yet supported"
  | .arrow _ _ => .error "lowerI32: arrow access not yet supported"
  | .comma _ _ => .error "lowerI32: comma operator not supported"

/-- Lower a `CExpr` to `Exp (.scalar .f32)`. -/
partial def lowerF32 (env : Env) : CExpr → Except String (Exp (.scalar .f32))
  | .numLit s => do
    -- Bare integer used in f32 context: convert.
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
      | none => .error s!"lowerF32: unsupported call '{fn}' with {args.size} args"
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
  | .ternary _ _ _ => .error "lowerF32: ternary not yet supported"
  | .index obj i => do
    let name ← match obj with
      | .ident n => .ok n
      | _ => .error "lowerF32: only `name[i]` index supported"
    match env.bufs name with
    | none => .error s!"lowerF32: '{name}' not bound as a buffer; add it to env.bufs"
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
      | t => .error s!"lowerF32: buffer '{name}' has elem type {repr t}"
  | .member _ _ => .error "lowerF32: member access not yet supported"
  | .arrow _ _ => .error "lowerF32: arrow access not yet supported"
  | .comma _ _ => .error "lowerF32: comma operator not supported"

end -- mutual

end Hesper.Transpile.CUDA
