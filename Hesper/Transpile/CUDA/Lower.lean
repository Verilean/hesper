import Hesper.Transpile.CUDA.AST
import Hesper.WGSL.Exp

/-! # CUDA → ShaderM transpiler — Phase 1 lowering

Lowers `CExpr` (CUDA surface AST) into Hesper's `Exp` IR. Phase 1
focuses on **u32 arithmetic** — the dominant expression class in
quantized matmul inner loops:

  shifts, masks, bitwise ops, integer arith, hex literals, identifier
  references (mapped via an environment), unary neg/not, and one
  intrinsic call: `__dp4a(a, b, c) → c + dot4I8Packed(a, b)`.

The lowering is **type-directed at one type**: this entry point assumes
the result type is `Exp (.scalar .u32)`. Multi-type lowering (i32,
f32, etc.) comes in a later phase along with cast / mixed-type ops.
-/
namespace Hesper.Transpile.CUDA

open Hesper.WGSL

/-- An identifier environment: maps CUDA variable names to Hesper
    `Exp (.scalar .u32)` expressions. Identifiers we don't know about
    fall back to `Exp.var name` (treated as an opaque ShaderM variable). -/
abbrev IdEnv := String → Option (Exp (.scalar .u32))

/-- Default identifier environment: known empty; everything resolves
    via `Exp.var`. -/
def emptyEnv : IdEnv := fun _ => none

/-- Parse an integer literal token text (decimal `42`, `42u`, `42UL`,
    or hex `0xFF`, `0x1F0Fu`). Returns the numeric value, ignoring
    `[uUlL]` suffixes. -/
def parseIntLit (s : String) : Except String Nat :=
  -- Strip suffix
  let core : String := s.dropRightWhile (fun c =>
    c == 'u' ∨ c == 'U' ∨ c == 'l' ∨ c == 'L')
  if core.startsWith "0x" ∨ core.startsWith "0X" then
    let hex := core.drop 2
    let res : Except String Nat := hex.toList.foldlM (init := 0) fun n c =>
      let dOpt : Option Nat :=
        if c.isDigit then some (c.toNat - '0'.toNat)
        else if 'a' ≤ c ∧ c ≤ 'f' then some (10 + c.toNat - 'a'.toNat)
        else if 'A' ≤ c ∧ c ≤ 'F' then some (10 + c.toNat - 'A'.toNat)
        else none
      match dOpt with
      | some d => Except.ok (n * 16 + d)
      | none => Except.error s!"bad hex digit '{c}' in {s}"
    res
  else
    match core.toNat? with
    | some n => .ok n
    | none => .error s!"can't parse integer literal: {s}"

/-- Lower a `CExpr` to `Exp (.scalar .u32)`. Phase 1: u32-only.

    Unsupported constructs (string lits, member access, ternary on
    non-trivial branches, unknown call) raise an error so that we can
    detect transpiler gaps explicitly rather than emit wrong code. -/
partial def lowerU32 (env : IdEnv) (e : CExpr) : Except String (Exp (.scalar .u32)) :=
  match e with
  | .numLit s =>
    parseIntLit s |>.map fun n => Exp.litU32 n
  | .floatLit _ =>
    .error "lowerU32: float literal in u32 context"
  | .ident name =>
    match env name with
    | some e' => .ok e'
    | none => .ok (Exp.var name)
  | .unop op a =>
    match op with
    | .bitNot =>
      -- ~a  →  a ^ 0xFFFFFFFF
      lowerU32 env a |>.map fun ae =>
        Exp.bitXor ae (Exp.litU32 0xFFFFFFFF)
    | .neg =>
      -- -a (u32) →  0 - a
      lowerU32 env a |>.map fun ae =>
        Exp.sub (Exp.litU32 0) ae
    | _ => .error s!"lowerU32: unsupported unary {repr op}"
  | .binop op a b =>
    let lowerBin (mk : Exp (.scalar .u32) → Exp (.scalar .u32) → Exp (.scalar .u32))
        : Except String (Exp (.scalar .u32)) := do
      let ae ← lowerU32 env a
      let be ← lowerU32 env b
      .ok (mk ae be)
    match op with
    | .add    => lowerBin Exp.add
    | .sub    => lowerBin Exp.sub
    | .mul    => lowerBin Exp.mul
    | .shl    => lowerBin Exp.shiftLeft
    | .shr    => lowerBin Exp.shiftRight
    | .bitAnd => lowerBin Exp.bitAnd
    | .bitOr  => lowerBin Exp.bitOr
    | .bitXor => lowerBin Exp.bitXor
    | _ => .error s!"lowerU32: unsupported binop {repr op}"
  | .call fn args =>
    -- `__dp4a` returns i32 — Phase 2 (mixed types) handles it.
    .error s!"lowerU32: call '{fn}' with {args.size} args not supported in u32 lowering"
  | .cast _ inner =>
    -- Drop the cast; assume the inner expression is already u32.
    lowerU32 env inner
  | .ternary _ _ _ => .error "lowerU32: ternary not yet supported"
  | .index _ _ => .error "lowerU32: index expr not yet supported in pure-expr lowering"
  | .member _ _ => .error "lowerU32: member access not yet supported"
  | .arrow _ _ => .error "lowerU32: arrow access not yet supported"
  | .comma _ _ => .error "lowerU32: comma operator not supported"

end Hesper.Transpile.CUDA
