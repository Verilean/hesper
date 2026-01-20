import Hesper.AD.Reverse

/-!
# Debug AD Tape for Failing Tests
-/

namespace Examples.ADDebug

open Hesper.AD.Reverse

/-- Debug polynomial: f(x) = x² + 3x + 2 -/
def debug_polynomial : IO Unit := do
  IO.println "═══ Debug: f(x) = x² + 3x + 2 ═══"

  let f (ctx : ADContext) (x : Dual) : ADContext × Dual :=
    let (ctx, x2) := ctx.mul x x
    let three := Dual.const 3.0
    let (ctx, threex) := ctx.mul three x
    let (ctx, x2_plus_3x) := ctx.add x2 threex
    let two := Dual.const 2.0
    ctx.add x2_plus_3x two

  let x := 5.0
  let (value, grad) := gradWith f x

  IO.println s!"f({x}) = {value}, f'({x}) = {grad}"
  IO.println s!"Expected: f'(5) = 2*5 + 3 = 13"
  IO.println s!"Difference: {grad - 13.0}"
  IO.println ""

/-- Simpler test: f(x) = x + 3 -/
def debug_simple : IO Unit := do
  IO.println "═══ Debug: f(x) = x + 3 ═══"

  let f (ctx : ADContext) (x : Dual) : ADContext × Dual :=
    let three := Dual.const 3.0
    ctx.add x three

  let x := 5.0
  let (value, grad) := gradWith f x

  IO.println s!"f({x}) = {value}, f'({x}) = {grad}"
  IO.println s!"Expected: f'(x) = 1"
  IO.println ""

/-- Even simpler: f(x) = x -/
def debug_identity : IO Unit := do
  IO.println "═══ Debug: f(x) = x ═══"

  let f (ctx : ADContext) (x : Dual) : ADContext × Dual :=
    (ctx, x)

  let x := 5.0
  let (value, grad) := gradWith f x

  IO.println s!"f({x}) = {value}, f'({x}) = {grad}"
  IO.println s!"Expected: f'(x) = 1"
  IO.println ""

/-- Test: f(x) = x² -/
def debug_square : IO Unit := do
  IO.println "═══ Debug: f(x) = x² ═══"

  let f (ctx : ADContext) (x : Dual) : ADContext × Dual :=
    ctx.mul x x

  let x := 5.0
  let (value, grad) := gradWith f x

  IO.println s!"f({x}) = {value}, f'({x}) = {grad}"
  IO.println s!"Expected: f'(5) = 10"
  IO.println ""

/-- Test: f(x) = 3x -/
def debug_linear : IO Unit := do
  IO.println "═══ Debug: f(x) = 3x ═══"

  let f (ctx : ADContext) (x : Dual) : ADContext × Dual :=
    let three := Dual.const 3.0
    ctx.mul three x

  let x := 5.0
  let (value, grad) := gradWith f x

  IO.println s!"f({x}) = {value}, f'({x}) = {grad}"
  IO.println s!"Expected: f'(x) = 3"
  IO.println ""

def main : IO Unit := do
  IO.println "╔══════════════════════════════════════╗"
  IO.println "║   AD Tape Debugging                  ║"
  IO.println "╚══════════════════════════════════════╝"
  IO.println ""

  debug_identity
  debug_square
  debug_linear
  debug_simple
  debug_polynomial

end Examples.ADDebug

def main : IO Unit := Examples.ADDebug.main
