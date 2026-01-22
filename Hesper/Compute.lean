import Hesper.Basic
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WebGPU.Shader
import Hesper.WebGPU.Pipeline
import Hesper.WGSL.Types
import Hesper.WGSL.Exp
import Hesper.WGSL.DSL
import Hesper.WGSL.Monad
import Hesper.WGSL.Execute

namespace Hesper.Compute

open WebGPU
open WGSL
open Hesper.Basic

/-!
# High-level Compute API

This module provides a high-level API for running GPU compute kernels with minimal boilerplate.
It abstracts away the low-level WebGPU details and handles:

- Device initialization and management
- Buffer creation, upload, and download
- Shader compilation and pipeline creation
- Bind group layout and resource binding
- Compute dispatch and synchronization
- Automatic resource cleanup via Lean's GC

## Usage

For simple compute operations, use `runSimpleKernel` with WGSL shader source.
For type-safe shader construction, combine with `Hesper.WGSL.DSL` module.

## Performance Considerations

- Default workgroup size is 256 threads (configurable via KernelConfig)
- Buffers use storage usage flags (read_write access from shaders)
- Synchronous execution with deviceWait (blocking until GPU completes)
- For async execution, use `Hesper.Async` module instead

## Examples

See `Examples/MainMatmul.lean` for production usage with WGSL DSL.
-/

/-- Compute kernel configuration for GPU dispatch.

Specifies the workgroup dimensions and number of workgroups to launch.

**Fields**:
- `workgroupSize`: Threads per workgroup (x, y, z). Default: (256, 1, 1)
  - Must match `@workgroup_size` in shader
  - Product must not exceed GPU limits (typically 256-1024)
  - Use (256, 1, 1) for 1D problems, (16, 16, 1) for 2D, (8, 8, 8) for 3D

- `numWorkgroups`: Number of workgroups to dispatch (x, y, z)
  - Total threads = workgroupSize * numWorkgroups (component-wise)
  - For N elements, use (⌈N / workgroupSize.x⌉, 1, 1)

**Example**:

    -- Process 10000 elements with 256 threads per workgroup
    let config : KernelConfig := {
      workgroupSize := (256, 1, 1)
      numWorkgroups := (40, 1, 1)  -- ceil(10000 / 256) = 40
    }
-/
structure KernelConfig where
  workgroupSize : Nat × Nat × Nat := (256, 1, 1)  -- Default workgroup size
  numWorkgroups : Nat × Nat × Nat                  -- Number of workgroups to dispatch
  deriving Inhabited

/-- Run a simple GPU compute kernel from WGSL shader source code.

This high-level function handles all GPU compute boilerplate:
1. Gets GPU device from instance
2. Creates storage buffer and uploads input data
3. Compiles WGSL shader to SPIR-V/IR
4. Creates compute pipeline with bind group layout
5. Binds buffer to @binding(0)
6. Dispatches compute with specified workgroups
7. Waits for GPU completion (blocking)
8. Downloads and returns result data

**Parameters**:
- `inst`: WebGPU instance from `Hesper.init`
- `shaderSource`: WGSL shader source code as string
  - Must have `@compute` entry point named "main"
  - Buffer must be `@group(0) @binding(0) var<storage, read_write> data: array<f32>`
- `inputData`: Input array of Float32 values
- `outputSize`: Number of elements to read back (usually same as input size)
- `config`: Kernel dispatch configuration (default: 4 workgroups of 256 threads)

**Returns**: Array of Float32 values read from GPU buffer

**Shader Requirements**:
- Entry point: `@compute fn main(@builtin(global_invocation_id) gid: vec3<u32>)`
- Binding: `@group(0) @binding(0) var<storage, read_write> data: array<f32>`
- Workgroup size must match config (default: `@workgroup_size(256)`)

**Example**:

    def main : IO Unit := do
      let inst ← Hesper.init

      let shader := "
        @group(0) @binding(0) var<storage, read_write> data: array<f32>;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
          let i = gid.x;
          if (i < arrayLength(&data)) {
            data[i] = data[i] * 2.0;  -- Double each element
          }
        }
      "

      let input := #[1.0, 2.0, 3.0, 4.0]
      let config := { numWorkgroups := (1, 1, 1) }  -- 256 threads enough for 4 elements
      let result ← runSimpleKernel inst shader input 4 config
      IO.println s!"Result: {result}"  -- Output: #[2.0, 4.0, 6.0, 8.0]

**Performance Note**: This function is synchronous and blocks until GPU completes.
For async/concurrent execution, use `Hesper.Async.runKernelAsync` instead.

**Safety**: Bounds checking must be done in shader (use `arrayLength` or `gid.x < N`).
-/
def runSimpleKernel
  (inst : Instance)
  (shaderSource : String)
  (inputData : Array Float)
  (outputSize : Nat)
  (config : KernelConfig := { numWorkgroups := (4, 1, 1) }) : IO (Array Float) := do

  -- Initialize device
  let device ← getDevice inst

  -- Create buffers
  let bufferSize := inputData.size * 4  -- Float = 4 bytes
  let bufferDesc : BufferDescriptor := {
    size := bufferSize.toUSize
    usage := [.storage, .copyDst, .copySrc]
    mappedAtCreation := false
  }
  let buffer ← createBuffer device bufferDesc

  -- Upload input data
  let bytes ← Hesper.Basic.floatArrayToBytes inputData
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

  -- Convert f32 bytes to f64 Float array
  Hesper.Basic.bytesToFloatArray resultBytes

/-- Generate WGSL shader code for a simple unary operation (map over array).

Generates a complete WGSL shader that applies a function to each element of an array in-place.
The shader operates on a single `var<storage, read_write> data: array<f32>` buffer.

**Parameters**:
- `f`: Function from scalar f32 to scalar f32 (using WGSL DSL Exp types)
  - Input is bound to variable "x"
  - Return value is written back to data[i]

**Generated Shader**:
- Entry point: main
- Workgroup size: 256
- Binding: @group(0) @binding(0) data array
- Bounds checking: if (i < arrayLength(&data))

**Example**:

    import Hesper.WGSL.Exp
    open WGSL

    -- Double each element
    let shader1 := generateUnaryShader (fun x => x * Exp.litF32 2.0)

    -- Apply tanh activation
    let shader2 := generateUnaryShader (fun x => Exp.tanh x)

    -- Complex function: f(x) = sqrt(abs(x)) + 1.0
    let shader3 := generateUnaryShader (fun x =>
      Exp.add (Exp.sqrt (Exp.abs x)) (Exp.litF32 1.0))

**Usage**:

    let inst ← Hesper.init
    let shader := generateUnaryShader (fun x => Exp.exp x)
    let input := #[0.0, 1.0, 2.0]
    let result ← runSimpleKernel inst shader input 3
    -- result ≈ #[1.0, 2.718, 7.389]

**Note**: For type-safe multi-step operations, use `Hesper.WGSL.Monad.ShaderM` instead.
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

/-- Generate WGSL shader code for a binary operation (combine two arrays element-wise).

Generates a complete WGSL shader that applies a binary function to corresponding elements
of two input arrays and stores results in a third array: `dataC[i] = f(dataA[i], dataB[i])`

**Parameters**:
- `f`: Binary function taking two scalar f32 values and returning scalar f32
  - First input is bound to variable "a" (from dataA)
  - Second input is bound to variable "b" (from dataB)
  - Return value is written to dataC[i]

**Generated Shader**:
- Entry point: main
- Workgroup size: 256
- Bindings:
  - @group(0) @binding(0): dataA (read)
  - @group(0) @binding(1): dataB (read)
  - @group(0) @binding(2): dataC (write)
- Bounds checking: if (i < arrayLength(&dataA))

**Example**:

    import Hesper.WGSL.Exp
    open WGSL

    -- Vector addition: C[i] = A[i] + B[i]
    let addShader := generateBinaryShader (fun a b => Exp.add a b)

    -- Multiplication: C[i] = A[i] * B[i]
    let mulShader := generateBinaryShader (fun a b => Exp.mul a b)

    -- Weighted sum: C[i] = 0.7*A[i] + 0.3*B[i]
    let weightedShader := generateBinaryShader (fun a b =>
      Exp.add (Exp.mul a (Exp.litF32 0.7)) (Exp.mul b (Exp.litF32 0.3)))

    -- Squared difference: C[i] = (A[i] - B[i])^2
    let diffSqShader := generateBinaryShader (fun a b =>
      let diff := Exp.sub a b
      Exp.mul diff diff)

**Note**: All three arrays (dataA, dataB, dataC) should have the same length.
The shader uses dataA's length for bounds checking.

**Warning**: This is a simplified helper for demonstration. For production code with
multiple buffers, use `Hesper.WGSL.Monad.ShaderM` for proper type-safe buffer management.
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

/-- High-level compute API on Device.

    Extension method for `Device` that executes a shader with named buffers.
    This provides a cleaner API similar to `gpu.compute()` in other frameworks.
-/
def _root_.Hesper.WebGPU.Device.compute
  (device : Device)
  (computation : Hesper.WGSL.Monad.ShaderM Unit)
  (namedBuffers : List (String × Buffer))
  (config : Hesper.WGSL.Execute.ExecutionConfig) : IO Unit :=
  Hesper.WGSL.Execute.executeShaderNamed device computation namedBuffers config

/-- High-level parallel-for API.

    Similar to `webgpu-dawn`, this function executes a shader over an array of data,
    handling all buffer creation, uploads, downloads, and synchronization.

    **Parameters**:
    - `device`: WebGPU device
    - `shaderSource`: WGSL shader source
    - `data`: Input array of Float32 values
    - `workgroupSize`: Threads per workgroup (default 256)

    **Returns**: Updated data array from the GPU
-/
def parallelFor
  (device : Device)
  (shaderSource : String)
  (data : Array Float)
  (workgroupSize : Nat := 256) : IO (Array Float) := do

  let count := data.size
  let numWorkgroups := (count + workgroupSize - 1) / workgroupSize

  -- Create buffer
  let bufferSize := count * 4
  let bufferDesc : BufferDescriptor := {
    size := bufferSize.toUSize
    usage := [.storage, .copyDst, .copySrc]
    mappedAtCreation := false
  }
  let buffer ← createBuffer device bufferDesc

  -- Upload data
  let bytes ← Hesper.Basic.floatArrayToBytes data
  writeBuffer device buffer 0 bytes

  -- Compile shader
  let shaderModule ← createShaderModule device shaderSource

  -- Pipeline setup
  let layoutEntry : BindGroupLayoutEntry := {
    binding := 0
    visibility := .compute
    bindingType := .buffer false
  }
  let bindGroupLayout ← createBindGroupLayout device #[layoutEntry]
  let pipelineDesc : ComputePipelineDescriptor := {
    shaderModule := shaderModule
    entryPoint := "main"
    bindGroupLayout := bindGroupLayout
  }
  let pipeline ← createComputePipeline device pipelineDesc

  -- Bind group
  let bindEntry : BindGroupEntry := {
    binding := 0
    buffer := buffer
    offset := 0
    size := bufferSize.toUSize
  }
  let bindGroup ← createBindGroup device bindGroupLayout #[bindEntry]

  -- Dispatch
  dispatchCompute device pipeline bindGroup numWorkgroups.toUInt32 1 1
  deviceWait device

  -- Read back results
  let resultBytes ← mapBufferRead device buffer 0 (bufferSize.toUSize)
  unmapBuffer buffer

  Hesper.Basic.bytesToFloatArray resultBytes

end Hesper.Compute
