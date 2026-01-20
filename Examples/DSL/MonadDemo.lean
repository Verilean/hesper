import Hesper.WGSL.Monad

/-!
# ShaderM Monad Demo

Demonstrates building WGSL compute shaders imperatively using the ShaderM monad.

Features shown:
1. Declaring input/output buffers with automatic binding
2. Reading from built-in variables (globalId)
3. Variable declarations
4. Control flow (if, for loops)
5. Buffer read/write operations

Usage:
  lake build monad-demo && ./.lake/build/bin/monad-demo
-/

open Hesper.WGSL
open Hesper.WGSL.Monad (ShaderM ShaderState)
open Hesper.WGSL.Monad.ShaderM

/-! Example 1: Simple vector add with conditional -/
def vectorAddConditional : ShaderM Unit := do
  -- Get global invocation ID
  let gid ← globalId
  let idx := Exp.vecZ gid  -- Get x component (u32)

  -- Declare buffers (automatic binding assignment)
  let _inputA ← declareInputBuffer "inputA" (.array (.scalar .f32) 1024)
  let _inputB ← declareInputBuffer "inputB" (.array (.scalar .f32) 1024)
  let _output ← declareOutputBuffer "output" (.array (.scalar .f32) 1024)

  -- Read input values
  let valA ← readBuffer (ty := .scalar .f32) (n := 1024) "inputA" idx
  let valB ← readBuffer (ty := .scalar .f32) (n := 1024) "inputB" idx

  -- Compute sum
  let sum := Exp.add valA valB

  -- Conditional: clamp negative values to zero
  if_ (Exp.lt sum (Exp.litF32 0.0))
    (writeBuffer (ty := .scalar .f32) "output" idx (Exp.litF32 0.0))
    (writeBuffer (ty := .scalar .f32) "output" idx sum)

/-! Example 2: Complex computation with temporaries -/
def complexComputation : ShaderM Unit := do
  let gid ← globalId
  let idx := Exp.vecZ gid

  let _input ← declareInputBuffer "input" (.array (.scalar .f32) 1024)
  let _output ← declareOutputBuffer "output" (.array (.scalar .f32) 1024)

  -- Read input
  let val ← readBuffer (ty := .scalar .f32) (n := 1024) "input" idx

  -- Declare temporary variables
  let tmp1 ← Monad.ShaderM.var (.scalar .f32) (Exp.mul val (Exp.litF32 2.0))
  let tmp2 ← Monad.ShaderM.var (.scalar .f32) (Exp.add (Exp.var tmp1) (Exp.litF32 1.0))
  let result ← Monad.ShaderM.var (.scalar .f32) (Exp.max (t := .scalar .f32) (Exp.var tmp2) (Exp.litF32 0.0))

  -- Write result
  writeBuffer (ty := .scalar .f32) "output" idx (Exp.var result)

/-! Example 3: For loop example -/
def forLoopExample : ShaderM Unit := do
  let gid ← globalId
  let idx := Exp.vecZ gid

  let _output ← declareOutputBuffer "output" (.array (.scalar .f32) 1024)

  -- Initialize accumulator
  let acc ← Monad.ShaderM.var (.scalar .f32) (Exp.litF32 0.0)

  -- Loop from 0 to 10, adding i to accumulator
  loop (Exp.litU32 0) (Exp.litU32 10) (Exp.litU32 1) fun i => do
    let iAsF32 := Exp.toF32 i
    let newAcc := Exp.add (Exp.var acc) iAsF32
    assign acc newAcc

  -- Write result
  writeBuffer (ty := .scalar .f32) "output" idx (Exp.var acc)

/-! Convert ShaderM computation to ComputeShader -/
def shaderMToComputeShader (workgroupSize : WorkgroupSize) (computation : ShaderM Unit) : ComputeShader :=
  let state := exec computation

  -- Convert declared buffers to StorageBuffer list
  let buffers := state.declaredBuffers.mapIdx fun i (name, ty, _) =>
    { group := 0
      binding := i
      name := name
      elemType := ty
      readWrite := true }

  -- Convert shared vars to WorkgroupVar list
  let workgroupVars := state.sharedVars.map fun (name, ty) =>
    { name := name, type := ty }

  { extensions := []
    diagnostics := []
    structs := []
    buffers := buffers
    workgroupVars := workgroupVars
    workgroupSize := workgroupSize
    builtins := [{ builtin := BuiltinBinding.globalInvocationId, name := "global_invocation_id" },
                 { builtin := BuiltinBinding.localInvocationId, name := "local_invocation_id" },
                 { builtin := BuiltinBinding.workgroupId, name := "workgroup_id" }]
    body := state.stmts }

def main : IO Unit := do
  IO.println "╔════════════════════════════════════════════════╗"
  IO.println "║   ShaderM Monad Demo                          ║"
  IO.println "╚════════════════════════════════════════════════╝"
  IO.println ""

  -- Example 1: Vector add with conditional
  IO.println "Example 1: Vector Add with Conditional"
  IO.println "─────────────────────────────────────────────"
  let shader1 := shaderMToComputeShader { x := 256, y := 1, z := 1 } vectorAddConditional
  IO.println (shader1.toWGSL)
  IO.println ""

  -- Example 2: Complex computation with temporaries
  IO.println "Example 2: Complex Computation with Variables"
  IO.println "─────────────────────────────────────────────"
  let shader2 := shaderMToComputeShader { x := 256, y := 1, z := 1 } complexComputation
  IO.println (shader2.toWGSL)
  IO.println ""

  -- Example 3: For loop
  IO.println "Example 3: For Loop Accumulation"
  IO.println "─────────────────────────────────────────────"
  let shader3 := shaderMToComputeShader { x := 256, y := 1, z := 1 } forLoopExample
  IO.println (shader3.toWGSL)
  IO.println ""

  IO.println "✅ All shader generation successful!"
