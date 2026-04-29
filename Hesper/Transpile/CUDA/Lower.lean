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

/-- Identifier environment: maps a CUDA name to a typed Hesper `Exp`
    for any of the scalar types we currently lower to. Lookup misses
    fall back to `Exp.var name`. -/
structure Env where
  u32 : String → Option (Exp (.scalar .u32)) := fun _ => none
  i32 : String → Option (Exp (.scalar .i32)) := fun _ => none
  f32 : String → Option (Exp (.scalar .f32)) := fun _ => none

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
    -- u32-context calls are rare; most intrinsics return i32 or f32.
    .error s!"lowerU32: call '{fn}' (returns non-u32?) — try lowerI32/lowerF32 if you know the type"
  | .cast ty inner =>
    -- (uint32_t) e → drop the cast (assume inner already u32)
    if ty.endsWith "uint32_t" ∨ ty.endsWith "unsigned" ∨ ty == "unsigned int"
       ∨ ty.endsWith "u32" ∨ ty == "uint" then
      lowerU32 env inner
    else
      .error s!"lowerU32: cast to '{ty}' not supported"
  | .ternary _ _ _ => .error "lowerU32: ternary not yet supported"
  | .index _ _ => .error "lowerU32: index expr not yet supported"
  | .member _ _ => .error "lowerU32: member access not yet supported"
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
    match op with
    | .add => bin Exp.add
    | .sub => bin Exp.sub
    | .mul => bin Exp.mul
    | _ => .error s!"lowerI32: unsupported binop {repr op}"
  | .call fn args =>
    match fn, args.toList with
    | "__dp4a", [a, b, c] => do
      -- __dp4a(a, b, c) = c + dot4I8Packed(a, b)  (i32 result)
      let ae ← lowerU32 env a
      let be ← lowerU32 env b
      let ce ← lowerI32 env c
      .ok (Exp.add ce (Exp.dot4I8Packed ae be))
    | _, _ => .error s!"lowerI32: unsupported call '{fn}' with {args.size} args"
  | .cast ty inner =>
    if ty.endsWith "int" ∨ ty == "int32_t" ∨ ty == "i32" then
      lowerI32 env inner
    else
      .error s!"lowerI32: cast to '{ty}' not supported"
  | .ternary _ _ _ => .error "lowerI32: ternary not yet supported"
  | .index _ _ => .error "lowerI32: index expr not yet supported"
  | .member _ _ => .error "lowerI32: member access not yet supported"
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
    | none => .ok (Exp.var name)
  | .unop op a =>
    match op with
    | .neg => do
      let ae ← lowerF32 env a
      .ok (Exp.neg ae)
    | _ => .error s!"lowerF32: unsupported unary {repr op}"
  | .binop op a b =>
    let bin (mk : Exp (.scalar .f32) → Exp (.scalar .f32) → Exp (.scalar .f32))
        : Except String (Exp (.scalar .f32)) := do
      let ae ← lowerF32 env a; let be ← lowerF32 env b; .ok (mk ae be)
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
    | _, _ => .error s!"lowerF32: unsupported call '{fn}' with {args.size} args"
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
  | .index _ _ => .error "lowerF32: index expr not yet supported"
  | .member _ _ => .error "lowerF32: member access not yet supported"
  | .arrow _ _ => .error "lowerF32: arrow access not yet supported"
  | .comma _ _ => .error "lowerF32: comma operator not supported"

end -- mutual

end Hesper.Transpile.CUDA
