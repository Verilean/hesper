import Hesper.WGSL.Types
import Hesper.WGSL.Exp
import Hesper.WGSL.DSL

/-!
# WGSL DSL Basic Examples

This file demonstrates the type-safe WGSL DSL capabilities.
-/

namespace Examples.DSLBasics

open Hesper.WGSL

-- Example 1: Simple arithmetic expressions
def example1_simple : IO Unit := do
  -- Create type-safe expressions using the DSL
  let x : Exp (.scalar .f32) := var "x"
  let y : Exp (.scalar .f32) := var "y"

  -- Operator overloading makes it natural
  let sum := x + y
  let product := x * y
  let complex := (x + y) * (x - y)  -- (x + y) * (x - y)

  IO.println "=== Example 1: Simple Arithmetic ==="
  IO.println s!"x + y = {sum.toWGSL}"
  IO.println s!"x * y = {product.toWGSL}"
  IO.println s!"(x + y) * (x - y) = {complex.toWGSL}"
  IO.println ""

-- Example 2: Comparison and boolean logic
def example2_comparisons : IO Unit := do
  let a : Exp (.scalar .f32) := var "a"
  let b : Exp (.scalar .f32) := var "b"

  -- Type-safe comparisons (result is Exp (.scalar .bool))
  let less := a .<. b
  let greaterEq := a .>=. b
  let equals := a .==. b

  -- Boolean logic
  let condition := (a .<. b) .&&. (b .<. lit 10.0)

  IO.println "=== Example 2: Comparisons ==="
  IO.println s!"a < b = {less.toWGSL}"
  IO.println s!"a >= b = {greaterEq.toWGSL}"
  IO.println s!"a == b = {equals.toWGSL}"
  IO.println s!"(a < b) && (b < 10.0) = {condition.toWGSL}"
  IO.println ""

-- Example 3: Built-in math functions
def example3_builtins : IO Unit := do
  let x : Exp (.scalar .f32) := var "x"

  -- WGSL built-in functions
  let squared := x * x
  let sqrtX := Exp.sqrt x
  let absX := Exp.abs x
  let clampedX := Exp.clamp x (lit 0.0) (lit 1.0)
  let powX := Exp.pow x (lit 2.0)

  IO.println "=== Example 3: Built-in Functions ==="
  IO.println s!"x * x = {squared.toWGSL}"
  IO.println s!"sqrt(x) = {sqrtX.toWGSL}"
  IO.println s!"abs(x) = {absX.toWGSL}"
  IO.println s!"clamp(x, 0.0, 1.0) = {clampedX.toWGSL}"
  IO.println s!"pow(x, 2.0) = {powX.toWGSL}"
  IO.println ""

-- Example 4: Complex expression (demonstrating composition)
def example4_complex : IO Unit := do
  let x : Exp (.scalar .f32) := var "x"
  let y : Exp (.scalar .f32) := var "y"

  -- Complex mathematical expression
  -- result = sqrt(x*x + y*y) / (1.0 + abs(x - y))
  let numerator := sqrt (x * x + y * y)
  let denominator := lit 1.0 + abs (x - y)
  let result := numerator / denominator

  IO.println "=== Example 4: Complex Expression ==="
  IO.println "result = sqrt(x*x + y*y) / (1.0 + abs(x - y))"
  IO.println s!"Generated WGSL: {result.toWGSL}"
  IO.println ""

-- Example 5: Select (ternary) operation
def example5_select : IO Unit := do
  let x : Exp (.scalar .f32) := var "x"
  let y : Exp (.scalar .f32) := var "y"

  -- select(condition, true_value, false_value)
  -- Returns true_value if condition is true, else false_value
  let maxXY := select (x .>. y) x y
  let clampManual := select (x .<. lit 0.0) (lit 0.0)
                    (select (x .>. lit 1.0) (lit 1.0) x)

  IO.println "=== Example 5: Select (Ternary) ==="
  IO.println s!"max(x, y) using select: {maxXY.toWGSL}"
  IO.println s!"Manual clamp: {clampManual.toWGSL}"
  IO.println ""

-- Example 6: Demonstrating type safety
-- This won't compile (commented out):
-- def type_error : IO Unit := do
--   let f : Exp (.scalar .f32) := var "f"
--   let i : Exp (.scalar .i32) := var "i"
--   let wrong := f + i  -- ERROR: Can't add f32 and i32!

def example6_type_safety : IO Unit := do
  IO.println "=== Example 6: Type Safety ==="
  IO.println "The DSL prevents type errors at compile time:"
  IO.println "  ✓ Can add f32 + f32"
  IO.println "  ✓ Can add i32 + i32"
  IO.println "  ✗ CANNOT add f32 + i32 (compile error!)"
  IO.println "  ✗ CANNOT use f32 as boolean condition (compile error!)"
  IO.println ""

-- Run all examples
def main : IO Unit := do
  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║   Hesper WGSL DSL - Basic Examples          ║"
  IO.println "╚══════════════════════════════════════════════╝"
  IO.println ""

  example1_simple
  example2_comparisons
  example3_builtins
  example4_complex
  example5_select
  example6_type_safety

  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║   All examples completed!                    ║"
  IO.println "╚══════════════════════════════════════════════╝"

end Examples.DSLBasics

-- Top-level main for executable
def main : IO Unit := Examples.DSLBasics.main
