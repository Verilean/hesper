import Hesper.WGSL.Types
import Hesper.WGSL.Exp
import Hesper.WGSL.DSL
import Hesper.WGSL.Templates

/-!
# WGSL Shader Generation Examples

This file demonstrates generating complete WGSL compute shaders using the DSL.
-/

namespace Examples.ShaderGeneration

open Hesper.WGSL
open Hesper.WGSL.Templates

-- Example 1: Double each element (unary operation)
def example1_double : IO Unit := do
  IO.println "=== Example 1: Double Each Element ==="
  IO.println ""

  -- Generate shader using the DSL
  let shader := generateUnaryShader (fun x => x * lit 2.0)

  IO.println "Generated WGSL Shader:"
  IO.println "─────────────────────────────────────"
  IO.println shader
  IO.println "─────────────────────────────────────"
  IO.println ""

-- Example 2: Square root of each element
def example2_sqrt : IO Unit := do
  IO.println "=== Example 2: Square Root ==="
  IO.println ""

  let shader := generateUnaryShader (fun x => sqrt x)

  IO.println "Generated WGSL Shader:"
  IO.println "─────────────────────────────────────"
  IO.println shader
  IO.println "─────────────────────────────────────"
  IO.println ""

-- Example 3: Sigmoid-like activation function
def example3_sigmoid : IO Unit := do
  IO.println "=== Example 3: Sigmoid-like Activation ==="
  IO.println "f(x) = 1.0 / (1.0 + exp(-x))"
  IO.println ""

  let shader := generateUnaryShader (fun x =>
    lit 1.0 / (lit 1.0 + exp (Exp.neg x))
  )

  IO.println "Generated WGSL Shader:"
  IO.println "─────────────────────────────────────"
  IO.println shader
  IO.println "─────────────────────────────────────"
  IO.println ""

-- Example 4: ReLU activation function
def example4_relu : IO Unit := do
  IO.println "=== Example 4: ReLU Activation ==="
  IO.println "f(x) = max(0, x)"
  IO.println ""

  let shader := generateUnaryShader (fun x =>
    max (lit 0.0) x
  )

  IO.println "Generated WGSL Shader:"
  IO.println "─────────────────────────────────────"
  IO.println shader
  IO.println "─────────────────────────────────────"
  IO.println ""

-- Example 5: Clamp to range [0, 1]
def example5_clamp : IO Unit := do
  IO.println "=== Example 5: Clamp to [0, 1] ==="
  IO.println ""

  let shader := generateUnaryShader (fun x =>
    clamp x (lit 0.0) (lit 1.0)
  )

  IO.println "Generated WGSL Shader:"
  IO.println "─────────────────────────────────────"
  IO.println shader
  IO.println "─────────────────────────────────────"
  IO.println ""

-- Example 6: Vector addition (binary operation)
def example6_vector_add : IO Unit := do
  IO.println "=== Example 6: Vector Addition ==="
  IO.println "c[i] = a[i] + b[i]"
  IO.println ""

  let shader := generateBinaryShader (fun a b => a + b)

  IO.println "Generated WGSL Shader:"
  IO.println "─────────────────────────────────────"
  IO.println shader
  IO.println "─────────────────────────────────────"
  IO.println ""

-- Example 7: Weighted sum (AXPY: a * x + y)
def example7_axpy : IO Unit := do
  IO.println "=== Example 7: AXPY (a * x + y) ==="
  IO.println "c[i] = 2.0 * a[i] + b[i]"
  IO.println ""

  let shader := generateBinaryShader (fun a b =>
    lit 2.0 * a + b
  )

  IO.println "Generated WGSL Shader:"
  IO.println "─────────────────────────────────────"
  IO.println shader
  IO.println "─────────────────────────────────────"
  IO.println ""

-- Example 8: Element-wise multiply then add
def example8_fma : IO Unit := do
  IO.println "=== Example 8: Fused Multiply-Add ==="
  IO.println "c[i] = a[i] * b[i] + 1.0"
  IO.println ""

  let shader := generateBinaryShader (fun a b =>
    a * b + lit 1.0
  )

  IO.println "Generated WGSL Shader:"
  IO.println "─────────────────────────────────────"
  IO.println shader
  IO.println "─────────────────────────────────────"
  IO.println ""

-- Example 9: Complex expression
def example9_complex : IO Unit := do
  IO.println "=== Example 9: Complex Expression ==="
  IO.println "c[i] = sqrt(a[i]^2 + b[i]^2)  (Euclidean distance)"
  IO.println ""

  let shader := generateBinaryShader (fun a b =>
    sqrt (a * a + b * b)
  )

  IO.println "Generated WGSL Shader:"
  IO.println "─────────────────────────────────────"
  IO.println shader
  IO.println "─────────────────────────────────────"
  IO.println ""

-- Example 10: Conditional expression
def example10_conditional : IO Unit := do
  IO.println "=== Example 10: Conditional Max ==="
  IO.println "c[i] = max(a[i], b[i]) using select"
  IO.println ""

  let shader := generateBinaryShader (fun a b =>
    select (a .>. b) a b
  )

  IO.println "Generated WGSL Shader:"
  IO.println "─────────────────────────────────────"
  IO.println shader
  IO.println "─────────────────────────────────────"
  IO.println ""

-- Run all examples
def main : IO Unit := do
  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║   Hesper WGSL Shader Generation Examples    ║"
  IO.println "╚══════════════════════════════════════════════╝"
  IO.println ""

  example1_double
  example2_sqrt
  example3_sigmoid
  example4_relu
  example5_clamp
  example6_vector_add
  example7_axpy
  example8_fma
  example9_complex
  example10_conditional

  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║   All shader generation examples complete!   ║"
  IO.println "╚══════════════════════════════════════════════╝"

end Examples.ShaderGeneration

-- Top-level main for executable
def main : IO Unit := Examples.ShaderGeneration.main
