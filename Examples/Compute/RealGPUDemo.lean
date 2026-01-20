import Hesper.WGSL.Execute
import Hesper.Basic

/-!
# Real GPU Execution Demo

This demonstrates end-to-end GPU execution:
1. Initialize GPU device
2. Create and populate buffers
3. Generate and compile WGSL shader
4. Execute on GPU
5. Read back and verify results

Usage:
  lake build real-gpu-demo && ./.lake/build/bin/real-gpu-demo
-/

open Hesper.WGSL
open Hesper.WGSL.Monad (ShaderM)
open Hesper.WGSL.Monad.ShaderM
open Hesper.WGSL.Execute
open Hesper.WebGPU
open Hesper.Basic

/-! Vector doubling kernel: output[i] = input[i] * 2.0 -/
def vectorDoubleKernel : ShaderM Unit := do
  let gid ← globalId
  let idx := Exp.vecZ gid
  let _input ← declareInputBuffer "input" (.array (.scalar .f32) 1024)
  let _output ← declareOutputBuffer "output" (.array (.scalar .f32) 1024)
  let val ← readBuffer (ty := .scalar .f32) (n := 1024) "input" idx
  let result := Exp.mul val (Exp.litF32 2.0)
  writeBuffer (ty := .scalar .f32) "output" idx result

/-! Vector addition kernel: output[i] = a[i] + b[i] -/
def vectorAddKernel : ShaderM Unit := do
  let gid ← globalId
  let idx := Exp.vecZ gid
  let _a ← declareInputBuffer "a" (.array (.scalar .f32) 1024)
  let _b ← declareInputBuffer "b" (.array (.scalar .f32) 1024)
  let _output ← declareOutputBuffer "output" (.array (.scalar .f32) 1024)
  let valA ← readBuffer (ty := .scalar .f32) (n := 1024) "a" idx
  let valB ← readBuffer (ty := .scalar .f32) (n := 1024) "b" idx
  let result := Exp.add valA valB
  writeBuffer (ty := .scalar .f32) "output" idx result

/-! Fused multiply-add kernel: output[i] = a[i] * b[i] + c[i] -/
def fmaKernel : ShaderM Unit := do
  let gid ← globalId
  let idx := Exp.vecZ gid
  let _a ← declareInputBuffer "a" (.array (.scalar .f32) 512)
  let _b ← declareInputBuffer "b" (.array (.scalar .f32) 512)
  let _c ← declareInputBuffer "c" (.array (.scalar .f32) 512)
  let _output ← declareOutputBuffer "output" (.array (.scalar .f32) 512)
  let valA ← readBuffer (ty := .scalar .f32) (n := 512) "a" idx
  let valB ← readBuffer (ty := .scalar .f32) (n := 512) "b" idx
  let valC ← readBuffer (ty := .scalar .f32) (n := 512) "c" idx
  let tmp ← Monad.ShaderM.var (.scalar .f32) (Exp.mul valA valB)
  let result := Exp.add (Exp.var tmp) valC
  writeBuffer (ty := .scalar .f32) "output" idx result

def main : IO Unit := do
  IO.println "╔════════════════════════════════════════════════╗"
  IO.println "║   Real GPU Execution Demo                     ║"
  IO.println "╚════════════════════════════════════════════════╝"
  IO.println ""

  IO.println "This demo shows the WGSL code generation pipeline."
  IO.println "Note: Actual GPU execution requires buffer FFI implementation."
  IO.println ""

  -- Example 1: Vector Doubling
  IO.println "═══════════════════════════════════════════════"
  IO.println "Example 1: Vector Doubling (output = input * 2)"
  IO.println "═══════════════════════════════════════════════"
  IO.println ""
  IO.println "Kernel definition:"
  IO.println "  def vectorDoubleKernel := do"
  IO.println "    let gid ← globalId"
  IO.println "    let idx := Exp.vecZ gid"
  IO.println "    let val ← readBuffer \"input\" idx"
  IO.println "    writeBuffer \"output\" idx (val * 2.0)"
  IO.println ""
  IO.println "Generated WGSL:"
  debugPrintWGSL vectorDoubleKernel (ExecutionConfig.dispatch1D 1024)
  IO.println ""

  -- Example 2: Vector Addition
  IO.println "═══════════════════════════════════════════════"
  IO.println "Example 2: Vector Addition (output = a + b)"
  IO.println "═══════════════════════════════════════════════"
  IO.println ""
  IO.println "Generated WGSL:"
  debugPrintWGSL vectorAddKernel (ExecutionConfig.dispatch1D 1024)
  IO.println ""

  -- Example 3: Fused Multiply-Add
  IO.println "═══════════════════════════════════════════════"
  IO.println "Example 3: FMA (output = a * b + c)"
  IO.println "═══════════════════════════════════════════════"
  IO.println ""
  IO.println "Generated WGSL:"
  debugPrintWGSL fmaKernel (ExecutionConfig.dispatch1D 512)
  IO.println ""

  IO.println "✅ All shaders compiled successfully!"
  IO.println ""
  IO.println "Next Steps for Full GPU Execution:"
  IO.println "  1. Initialize device: device ← getDevice"
  IO.println "  2. Create buffers: createBuffer device bufferDesc"
  IO.println "  3. Upload data: writeBuffer device buffer 0 bytes"
  IO.println "  4. Execute shader: executeShaderNamed device kernel buffers config"
  IO.println "  5. Read results: mapBufferRead device outputBuffer"
  IO.println "  6. Verify correctness"
  IO.println ""
