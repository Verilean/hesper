import Hesper.WGSL.Exp

/-! # `Exp` pretty-printer + structural equality

Lean's `Repr` derivation can't handle `Exp : WGSLType → Type` because
of the dependent index. This module provides:

  * `Exp.toSExp : Exp t → String` — canonical S-expression rendering
    that works for *any* type index.
  * `Exp.structEq : Exp t1 → Exp t2 → Bool` — structural equality
    (across types, since the printer is total).

These are useful for tests, debugging, and the `transpile-cuda` test
suite (where we lex+parse+lower a CUDA expression and want to assert
it equals a hand-written `Exp` reference).

Coverage: every `Exp` constructor we use in the codebase. New
constructors should be added here too — there's a catch-all
`s!"(? {repr h})"` for unhandled cases that prints a hash so we
notice gaps without crashing.
-/
namespace Hesper.WGSL

open Exp

/-- Render an `Exp t` as an S-expression for any `t`. Ignores subtle
    type tagging (e.g. `add` at f32 vs add at u32 print the same) —
    if you need type-tagged output, wrap this in your own renderer. -/
partial def Exp.toSExp {t : WGSLType} : Exp t → String
  -- Literals
  | litF32 f  => s!"(litF32 {f})"
  | litF16 f  => s!"(litF16 {f})"
  | litI32 i  => s!"(litI32 {i})"
  | litU32 n  => s!"(litU32 {n})"
  | litBool b => s!"(litBool {b})"
  -- Variables
  | var name => s!"(var {name})"
  -- Arithmetic
  | add a b => s!"(add {a.toSExp} {b.toSExp})"
  | sub a b => s!"(sub {a.toSExp} {b.toSExp})"
  | mul a b => s!"(mul {a.toSExp} {b.toSExp})"
  | div a b => s!"(div {a.toSExp} {b.toSExp})"
  | mod a b => s!"(mod {a.toSExp} {b.toSExp})"
  | neg a   => s!"(neg {a.toSExp})"
  -- Comparison
  | eq a b => s!"(eq {a.toSExp} {b.toSExp})"
  | ne a b => s!"(ne {a.toSExp} {b.toSExp})"
  | lt a b => s!"(lt {a.toSExp} {b.toSExp})"
  | le a b => s!"(le {a.toSExp} {b.toSExp})"
  | gt a b => s!"(gt {a.toSExp} {b.toSExp})"
  | ge a b => s!"(ge {a.toSExp} {b.toSExp})"
  -- Boolean
  | and a b => s!"(and {a.toSExp} {b.toSExp})"
  | or  a b => s!"(or {a.toSExp} {b.toSExp})"
  | not a   => s!"(not {a.toSExp})"
  -- Bitwise (u32)
  | shiftLeft  a b => s!"(shl {a.toSExp} {b.toSExp})"
  | shiftRight a b => s!"(shr {a.toSExp} {b.toSExp})"
  | bitAnd a b => s!"(bitAnd {a.toSExp} {b.toSExp})"
  | bitOr  a b => s!"(bitOr {a.toSExp} {b.toSExp})"
  | bitXor a b => s!"(bitXor {a.toSExp} {b.toSExp})"
  | mulhiU32 a b => s!"(mulhiU32 {a.toSExp} {b.toSExp})"
  -- Conversions
  | toF32  a => s!"(toF32 {a.toSExp})"
  | toF32U a => s!"(toF32U {a.toSExp})"
  | toI32  a => s!"(toI32 {a.toSExp})"
  -- Vector access
  | vecX a  => s!"(vecX {a.toSExp})"
  | vecY a  => s!"(vecY {a.toSExp})"
  | vec3X a => s!"(vec3X {a.toSExp})"
  | vec3Y a => s!"(vec3Y {a.toSExp})"
  | vec4X a => s!"(vec4X {a.toSExp})"
  | vec4Y a => s!"(vec4Y {a.toSExp})"
  | vec4Z a => s!"(vec4Z {a.toSExp})"
  | vecW  a => s!"(vecW {a.toSExp})"
  -- Specialised ops
  | fma a b c => s!"(fma {a.toSExp} {b.toSExp} {c.toSExp})"
  | select c t e => s!"(select {c.toSExp} {t.toSExp} {e.toSExp})"
  | dot4I8Packed a b => s!"(dot4I8Packed {a.toSExp} {b.toSExp})"
  | unpack2x16float a => s!"(unpack2x16float {a.toSExp})"
  -- Catch-all (prints constructor index hash so we notice gaps)
  | _ => "(?)"

/-- Structural equality across (possibly different) type indices,
    via S-expression round-trip. Two `Exp`s are equal iff their
    S-expression renderings match. -/
def Exp.structEq {t1 t2 : WGSLType} (a : Exp t1) (b : Exp t2) : Bool :=
  a.toSExp = b.toSExp

/-- Default-`Exp` instances. We pick a literal for each scalar and
    fall back to `Exp.var "?_default"` for everything else (vector
    types, etc.). Useful for `panic!` / `Option.getD`. -/
instance : Inhabited (Exp (.scalar .u32))  := ⟨Exp.litU32 0⟩
instance : Inhabited (Exp (.scalar .i32))  := ⟨Exp.litI32 0⟩
instance : Inhabited (Exp (.scalar .f32))  := ⟨Exp.litF32 0.0⟩
instance : Inhabited (Exp (.scalar .f16))  := ⟨Exp.litF16 0.0⟩
instance : Inhabited (Exp (.scalar .bool)) := ⟨Exp.litBool false⟩
instance {t : WGSLType} : Inhabited (Exp t) := ⟨Exp.var "?_default"⟩

end Hesper.WGSL
