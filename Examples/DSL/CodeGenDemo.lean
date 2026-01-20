import Hesper.WGSL.CodeGen

/-!
# Code Generation Demo

Demonstrates the complete WGSL code generation pipeline from ShaderM monads.

Features shown:
1. Automatic binding assignment for multiple buffers
2. Workgroup variable declarations
3. Custom function names and workgroup sizes
4. Extension directives
5. Complete module assembly

Usage:
  lake build codegen-demo && ./.lake/build/bin/codegen-demo
-/

open Hesper.WGSL
open Hesper.WGSL.Monad (ShaderM)
open Hesper.WGSL.Monad.ShaderM
open Hesper.WGSL.CodeGen

/-! Example 1: Simple kernel with automatic binding -/
def simpleKernel : ShaderM Unit := do
  let gid ← globalId
  let idx := Exp.vecZ gid

  let _input ← declareInputBuffer "input" (.array (.scalar .f32) 1024)
  let _output ← declareOutputBuffer "output" (.array (.scalar .f32) 1024)

  let val ← readBuffer (ty := .scalar .f32) (n := 1024) "input" idx
  let result := Exp.mul val (Exp.litF32 2.0)
  writeBuffer (ty := .scalar .f32) "output" idx result

/-! Example 2: Reduction with shared memory -/
def reductionKernel : ShaderM Unit := do
  let gid ← globalId
  let lid ← localId
  let globalIdx := Exp.vecZ gid
  let localIdx := Exp.vecZ lid

  let _input ← declareInputBuffer "input" (.array (.scalar .f32) 4096)
  let _output ← declareOutputBuffer "output" (.array (.scalar .f32) 16)

  -- Declare shared memory
  sharedNamed "sdata" (.array (.scalar .f32) 256)

  -- Load into shared memory
  let val ← readBuffer (ty := .scalar .f32) (n := 4096) "input" globalIdx
  writeWorkgroup (ty := .scalar .f32) "sdata" localIdx val

  -- Synchronize
  barrier

  -- Reduce (simplified - just write first element)
  if_ (Exp.eq localIdx (Exp.litU32 0))
    (do
      let sum ← readWorkgroup (ty := .scalar .f32) (n := 256) "sdata" (Exp.litU32 0)
      let wgId := Exp.vecZ (← workgroupId)
      writeBuffer (ty := .scalar .f32) "output" wgId sum)
    (pure ())

/-! Example 3: Complex compute with multiple operations -/
def complexCompute : ShaderM Unit := do
  let gid ← globalId
  let idx := Exp.vecZ gid

  -- Multiple buffers with automatic binding (binding 0, 1, 2, 3)
  let _inputA ← declareInputBuffer "inputA" (.array (.scalar .f32) 2048)
  let _inputB ← declareInputBuffer "inputB" (.array (.scalar .f32) 2048)
  let _inputC ← declareInputBuffer "inputC" (.array (.scalar .f32) 2048)
  let _output ← declareOutputBuffer "output" (.array (.scalar .f32) 2048)

  -- Read all inputs
  let a ← readBuffer (ty := .scalar .f32) (n := 2048) "inputA" idx
  let b ← readBuffer (ty := .scalar .f32) (n := 2048) "inputB" idx
  let c ← readBuffer (ty := .scalar .f32) (n := 2048) "inputC" idx

  -- Compute: result = (a * b) + c
  let tmp ← Monad.ShaderM.var (.scalar .f32) (Exp.mul a b)
  let result := Exp.add (Exp.var tmp) c

  writeBuffer (ty := .scalar .f32) "output" idx result

def main : IO Unit := do
  IO.println "╔════════════════════════════════════════════════╗"
  IO.println "║   WGSL Code Generation Demo                   ║"
  IO.println "╚════════════════════════════════════════════════╝"
  IO.println ""

  -- Example 1: Simple kernel
  IO.println "Example 1: Simple Kernel (Custom Name & Size)"
  IO.println "─────────────────────────────────────────────"
  let wgsl1 := generateWGSL "vectorDouble" {x := 64, y := 1, z := 1} [] simpleKernel
  IO.println wgsl1
  IO.println ""

  -- Example 2: Reduction with shared memory
  IO.println "Example 2: Reduction with Shared Memory"
  IO.println "─────────────────────────────────────────────"
  let wgsl2 := generateWGSL "reduce" {x := 256, y := 1, z := 1} [] reductionKernel
  IO.println wgsl2
  IO.println ""

  -- Example 3: Complex compute with multiple buffers
  IO.println "Example 3: Complex Compute (4 Buffers, Auto-Binding)"
  IO.println "─────────────────────────────────────────────"
  let wgsl3 := generateWGSL "fmaKernel" {x := 128, y := 1, z := 1} [] complexCompute
  IO.println wgsl3
  IO.println ""

  -- Example 4: Using generateWGSLSimple
  IO.println "Example 4: Simple Generation (Default Parameters)"
  IO.println "─────────────────────────────────────────────"
  let wgsl4 := generateWGSLSimple simpleKernel
  IO.println wgsl4
  IO.println ""

  IO.println "✅ Code generation demo complete!"
  IO.println ""
  IO.println "Key features demonstrated:"
  IO.println "  • Automatic binding assignment (@binding(0), @binding(1), ...)"
  IO.println "  • Workgroup variable declarations (var<workgroup>)"
  IO.println "  • Custom function names and workgroup sizes"
  IO.println "  • Complete WGSL module generation"
  IO.println "  • Multiple buffers with proper binding management"
