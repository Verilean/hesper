import Hesper.Basic
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WebGPU.Shader
import Hesper.WebGPU.Pipeline
import Hesper.WGSL.Types
import Hesper.WGSL.Exp
import Hesper.WGSL.DSL

namespace Hesper.Compute

open WebGPU
open WGSL
open Hesper.Basic

/-!
# High-level Compute API

This module provides a high-level API for running GPU compute kernels.
It handles device creation, buffer management, shader compilation,
and compute dispatch.
-/

/-- Compute kernel configuration -/
structure KernelConfig where
  workgroupSize : Nat × Nat × Nat := (256, 1, 1)  -- Default workgroup size
  numWorkgroups : Nat × Nat × Nat                  -- Number of workgroups to dispatch
  deriving Inhabited

/-- Create a simple compute kernel from WGSL source code.
    This is a convenience function that handles all setup:
    - Creates device (if not provided)
    - Compiles shader
    - Creates pipeline
    - Sets up bind group
    - Dispatches compute
    - Reads results

    Example:
    ```lean
    let shader := "
      @group(0) @binding(0) var<storage, read_write> data: array<f32>;

      @compute @workgroup_size(256)
      fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let i = gid.x;
        data[i] = data[i] * 2.0;
      }
    "
    let input := [1.0, 2.0, 3.0, 4.0]
    let result ← runSimpleKernel shader input 4
    ```
-/
def runSimpleKernel
  (shaderSource : String)
  (inputData : Array Float)
  (outputSize : Nat)
  (config : KernelConfig := { numWorkgroups := (4, 1, 1) }) : IO (Array Float) := do

  -- Initialize device
  let device ← getDevice

  -- Create buffers
  let bufferSize := inputData.size * 4  -- Float = 4 bytes
  let bufferDesc : BufferDescriptor := {
    size := bufferSize.toUSize
    usage := [.storage, .copyDst, .copySrc]
    mappedAtCreation := false
  }
  let buffer ← createBuffer device bufferDesc

  -- Upload input data
  let bytes := Hesper.Basic.floatArrayToBytes inputData
  writeBuffer device buffer 0 bytes

  -- Compile shader
  let shaderModule ← createShaderModule device shaderSource

  -- Create bind group layout
  let layoutEntry : BindGroupLayoutEntry := {
    binding := 0
    visibility := .compute
    bindingType := .buffer false  -- read_write storage buffer
  }
  let bindGroupLayout ← createBindGroupLayout device #[layoutEntry]

  -- Create pipeline
  let pipelineDesc : ComputePipelineDescriptor := {
    shaderModule := shaderModule
    entryPoint := "main"
    bindGroupLayout := bindGroupLayout
  }
  let pipeline ← createComputePipeline device pipelineDesc

  -- Create bind group
  let bindEntry : BindGroupEntry := {
    binding := 0
    buffer := buffer
    offset := 0
    size := bufferSize.toUSize
  }
  let bindGroup ← createBindGroup device bindGroupLayout #[bindEntry]

  -- Dispatch compute
  let (wx, wy, wz) := config.numWorkgroups
  dispatchCompute device pipeline bindGroup wx.toUInt32 wy.toUInt32 wz.toUInt32

  -- Wait for completion
  deviceWait device

  -- Read results
  let resultBytes ← mapBufferRead device buffer 0 ((outputSize * 4).toUSize)
  unmapBuffer buffer

  -- Resources are automatically cleaned up by Lean's GC via External finalizers

  return Hesper.Basic.bytesToFloatArray resultBytes

/-- Generate WGSL shader code for a simple unary operation.
    This takes a function body expression and wraps it in a complete shader.

    Example:
    ```lean
    import Hesper.WGSL.DSL
    open WGSL.DSL

    let shader := generateUnaryShader (fun x => x * lit 2.0)
    ```
-/
def generateUnaryShader (f : Exp (.scalar .f32) → Exp (.scalar .f32)) : String :=
  let x : Exp (.scalar .f32) := Exp.var "x"
  let body := f x
  let bodyCode := body.toWGSL
  "@group(0) @binding(0) var<storage, read_write> data: array<f32>;\n\n" ++
  "@compute @workgroup_size(256)\n" ++
  "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n" ++
  "  let i = gid.x;\n" ++
  "  if (i < arrayLength(&data)) {\n" ++
  "    let x = data[i];\n" ++
  s!"    data[i] = {bodyCode};\n" ++
  "  }\n" ++
  "}"

/-- Generate WGSL shader code for a binary operation (combining two arrays).

    Example:
    ```lean
    let shader := generateBinaryShader (fun a b => a + b)  -- Vector addition
    ```
-/
def generateBinaryShader (f : Exp (.scalar .f32) → Exp (.scalar .f32) → Exp (.scalar .f32)) : String :=
  let a : Exp (.scalar .f32) := Exp.var "a"
  let b : Exp (.scalar .f32) := Exp.var "b"
  let body := f a b
  let bodyCode := body.toWGSL
  "@group(0) @binding(0) var<storage, read_write> dataA: array<f32>;\n" ++
  "@group(0) @binding(1) var<storage, read_write> dataB: array<f32>;\n" ++
  "@group(0) @binding(2) var<storage, read_write> dataC: array<f32>;\n\n" ++
  "@compute @workgroup_size(256)\n" ++
  "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n" ++
  "  let i = gid.x;\n" ++
  "  if (i < arrayLength(&dataA)) {\n" ++
  "    let a = dataA[i];\n" ++
  "    let b = dataB[i];\n" ++
  s!"    dataC[i] = {bodyCode};\n" ++
  "  }\n" ++
  "}"

end Hesper.Compute
