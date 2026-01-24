import Hesper
import Hesper.WGSL.Types
import Hesper.WGSL.Exp
import Hesper.Compute

open Hesper.WGSL

def showcaseDSL : IO Unit := do
  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║   Hesper WGSL DSL Showcase                   ║"
  IO.println "╚══════════════════════════════════════════════╝"
  IO.println ""

  -- Example 1: Simple expression
  IO.println "Example 1: Simple Arithmetic"
  let x : Exp (.scalar .f32) := Exp.var "x"
  let y : Exp (.scalar .f32) := Exp.var "y"
  let expr1 := (x + y) * (x - y)
  IO.println s!"  Expression: (x + y) * (x - y)"
  IO.println s!"  WGSL Code:  {expr1.toWGSL}"
  IO.println ""

  -- Example 2: Complex math
  IO.println "Example 2: Complex Math"
  let distance := Exp.sqrt (x * x + y * y)
  IO.println s!"  Expression: sqrt(x*x + y*y)"
  IO.println s!"  WGSL Code:  {distance.toWGSL}"
  IO.println ""

  -- Example 3: Activation function
  IO.println "Example 3: ReLU Activation"
  let relu := Exp.max (Exp.litF32 0.0) x
  IO.println s!"  Expression: max(0.0, x)"
  IO.println s!"  WGSL Code:  {relu.toWGSL}"
  IO.println ""

  -- Example 4: DSL shader generation available!
  IO.println "Example 4: Shader Generation"
  IO.println "  The DSL can generate complete WGSL shaders!"
  IO.println "  See Examples/ShaderGeneration.lean for details"
  IO.println ""

def main : IO Unit := do
  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║   Verilean Hesper - GPU Inference Engine     ║"
  IO.println "╚══════════════════════════════════════════════╝"
  IO.println ""

  -- Show DSL capabilities
  showcaseDSL

  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║   All Tests Complete!                        ║"
  IO.println "╚══════════════════════════════════════════════╝"
