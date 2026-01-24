import Hesper.WGSL.Monad
import Hesper.WGSL.CodeGen
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WebGPU.Shader
import Hesper.WebGPU.Pipeline

namespace Hesper.WGSL.Execute

open Hesper.WGSL
open Hesper.WGSL.Monad (ShaderM)
open Hesper.WGSL.CodeGen
open Hesper.WebGPU

/-!
# WGSL Shader Execution Layer

Integration between ShaderM monad, code generation, and WebGPU execution.

This module provides high-level functions to:
1. Compile ShaderM computations to WGSL
2. Create GPU pipelines
3. Execute shaders with buffer management
4. Handle synchronization

Usage Pattern:
```lean
def myKernel : ShaderM Unit := do
  let gid ← globalId
  let idx := Exp.vecZ gid
  let _input ← declareInputBuffer "input" (.array (.scalar .f32) 1024)
  let _output ← declareOutputBuffer "output" (.array (.scalar .f32) 1024)
  let val ← readBuffer (ty := .scalar .f32) (n := 1024) "input" idx
  writeBuffer (ty := .scalar .f32) "output" idx (Exp.mul val (Exp.litF32 2.0))

-- Execute on GPU
executeShader device myKernel
  [("input", inputBuffer), ("output", outputBuffer)]
  {x := 256, y := 1, z := 1}
  (256, 1, 1)
```
-/

/-- GPU Buffer with name for binding -/
structure NamedBuffer where
  name : String
  buffer : Buffer

/-- Execution configuration for compute shaders -/
structure ExecutionConfig where
  funcName : String := "main"
  workgroupSize : WorkgroupSize := {x := 256, y := 1, z := 1}
  numWorkgroups : Nat × Nat × Nat
  extensions : List String := []
  diagnostics : List (String × String) := []

instance : Inhabited ExecutionConfig where
  default := {
    funcName := "main"
    workgroupSize := {x := 256, y := 1, z := 1}
    numWorkgroups := (1, 1, 1)
    extensions := []
    diagnostics := []
  }

namespace ExecutionConfig

/-- Create default execution config with specified workgroup count -/
def default (numWorkgroups : Nat × Nat × Nat) : ExecutionConfig :=
  { funcName := "main"
    workgroupSize := {x := 256, y := 1, z := 1}
    numWorkgroups := numWorkgroups }

/-- Create config for 1D dispatch -/
def dispatch1D (totalThreads : Nat) (workgroupSize : Nat := 256) : ExecutionConfig :=
  let numWorkgroups := (totalThreads + workgroupSize - 1) / workgroupSize
  { funcName := "main"
    workgroupSize := {x := workgroupSize, y := 1, z := 1}
    numWorkgroups := (numWorkgroups, 1, 1) }

end ExecutionConfig

/-- Compile a ShaderM computation to WGSL source code -/
def compileToWGSL
    (computation : ShaderM Unit)
    (funcName : String := "main")
    (workgroupSize : WorkgroupSize := {x := 256, y := 1, z := 1})
    (extensions : List String := [])
    (diagnostics : List (String × String) := [])
    : String :=
  generateWGSL funcName workgroupSize extensions diagnostics computation

/-- Create shader module from ShaderM computation -/
def createShaderFromComputation
    (device : Device)
    (computation : ShaderM Unit)
    (config : ExecutionConfig)
    : IO WebGPU.ShaderModule :=
  let wgslSource := compileToWGSL computation config.funcName config.workgroupSize []
  createShaderModule device wgslSource

/-- Execute a ShaderM computation on the GPU with named buffers.

This is the main high-level execution function. It:
1. Compiles the ShaderM computation to WGSL
2. Creates a shader module
3. Sets up the compute pipeline
4. Binds buffers by name
5. Dispatches the compute shader
6. Waits for completion
7. Cleans up resources

Parameters:
- device: GPU device
- computation: ShaderM monad defining the shader
- namedBuffers: List of (name, buffer) pairs for binding
- config: Execution configuration (workgroup size, dispatch size)

Example:
```lean
let kernel : ShaderM Unit := do
  let gid ← globalId
  let idx := Exp.vecZ gid
  let _input ← declareInputBuffer "input" (.array (.scalar .f32) 1024)
  let _output ← declareOutputBuffer "output" (.array (.scalar .f32) 1024)
  let val ← readBuffer (ty := .scalar .f32) (n := 1024) "input" idx
  writeBuffer (ty := .scalar .f32) "output" idx (Exp.mul val (Exp.litF32 2.0))

executeShaderNamed device kernel
  [("input", inputBuf), ("output", outputBuf)]
  (ExecutionConfig.dispatch1D 1024)
```
-/
def executeShaderNamed
    (device : Device)
    (computation : ShaderM Unit)
    (namedBuffers : List (String × Buffer))
    (config : ExecutionConfig)
    : IO Unit := do
  -- Compile to WGSL
  let wgslSource := compileToWGSL computation config.funcName config.workgroupSize config.extensions config.diagnostics

  IO.println s!"[Execute] Compiled shader ({wgslSource.length} bytes)"

  -- Create shader module
  let shaderModule ← createShaderModule device wgslSource

  -- Get buffer bindings from ShaderM state
  let state := ShaderM.exec computation
  let declaredNames := state.declaredBuffers.map (·.fst)

  -- Match buffers to bindings by name
  let sortedBuffers := declaredNames.filterMap fun name =>
    namedBuffers.find? (·.fst == name) |>.map (·.snd)

  -- Validate that we have all buffers
  if sortedBuffers.length != declaredNames.length then
    IO.println s!"[Execute] ERROR: Buffer count mismatch!"
    IO.println s!"  Expected: {declaredNames}"
    IO.println s!"  Provided: {namedBuffers.map (·.fst)}"
    throw <| IO.userError "Buffer binding mismatch"

  -- Create bind group layout
  let layoutEntries := List.range sortedBuffers.length |>.map fun i =>
    { binding := i.toUInt32
      visibility := ShaderStage.compute
      bindingType := BindingType.buffer false }  -- read_write storage

  let bindGroupLayout ← createBindGroupLayout device layoutEntries.toArray

  -- Create pipeline
  let pipelineDesc : ComputePipelineDescriptor := {
    shaderModule := shaderModule
    entryPoint := config.funcName
    bindGroupLayout := bindGroupLayout
  }
  let pipeline ← createComputePipeline device pipelineDesc

  -- Create bind group
  let bindEntries := sortedBuffers.mapIdx fun i buf =>
    { binding := i.toUInt32
      buffer := buf
      offset := 0
      size := 0 }  -- 0 means whole buffer

  let bindGroup ← createBindGroup device bindGroupLayout bindEntries.toArray

  -- Dispatch (async - returns Future)
  let (wx, wy, wz) := config.numWorkgroups
  IO.println s!"[Execute] Dispatching {wx}×{wy}×{wz} workgroups..."
  let future ← dispatchCompute device pipeline bindGroup wx.toUInt32 wy.toUInt32 wz.toUInt32

  -- Wait for completion
  deviceWait future

  -- Resources are automatically cleaned up by Lean's GC via External finalizers

  IO.println "[Execute] Completed successfully"

/-- Execute a simple ShaderM computation with a single input/output buffer.

Convenience wrapper for the common case of one input buffer and one output buffer.

Example:
```lean
let kernel : ShaderM Unit := do
  let gid ← globalId
  let idx := Exp.vecZ gid
  let _input ← declareInputBuffer "input" (.array (.scalar .f32) 1024)
  let _output ← declareOutputBuffer "output" (.array (.scalar .f32) 1024)
  let val ← readBuffer (ty := .scalar .f32) (n := 1024) "input" idx
  writeBuffer (ty := .scalar .f32) "output" idx (Exp.mul val (Exp.litF32 2.0))

executeShaderSimple device kernel inputBuf outputBuf 1024
```
-/
def executeShaderSimple
    (device : Device)
    (computation : ShaderM Unit)
    (inputBuffer : Buffer)
    (outputBuffer : Buffer)
    (numThreads : Nat)
    : IO Unit :=
  executeShaderNamed device computation
    [("input", inputBuffer), ("output", outputBuffer)]
    (ExecutionConfig.dispatch1D numThreads)

/-- Print generated WGSL for debugging -/
def debugPrintWGSL
    (computation : ShaderM Unit)
    (config : ExecutionConfig := ExecutionConfig.default (1, 1, 1))
    : IO Unit := do
  let wgsl := compileToWGSL computation config.funcName config.workgroupSize []
  IO.println "═══════════════════════════════════════════════"
  IO.println "Generated WGSL Shader:"
  IO.println "═══════════════════════════════════════════════"
  IO.println wgsl
  IO.println "═══════════════════════════════════════════════"

end Hesper.WGSL.Execute
