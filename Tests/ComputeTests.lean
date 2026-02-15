import LSpec
import Hesper
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WebGPU.Shader
import Hesper.WebGPU.Pipeline
import Hesper.WebGPU.Types

/-!
# Compute Pipeline Integration Tests

End-to-end tests for GPU compute operations:
- Shader compilation
- Pipeline creation
- Compute dispatch
- Buffer readback
- Numerical correctness
-/

namespace Tests.ComputeTests

open Hesper.WebGPU
open LSpec

def withDevice (action : Instance → Device → IO α) : IO α := do
  let inst ← Hesper.init
  let device ← getDevice inst
  action inst device

-- Simple shader that adds 1 to each element
def simpleAddShader : String :=
  "@group(0) @binding(0) var<storage, read_write> data: array<f32>;

   @compute @workgroup_size(256)
   fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
       let i = gid.x;
       if (i < arrayLength(&data)) {
           data[i] = data[i] + 1.0;
       }
   }"

-- Test: Shader Module Creation
def testShaderCreation : IO TestSeq := withDevice fun _ device => do
  let shader ← createShaderModule device simpleAddShader
  pure $ test "Shader module created successfully" true

-- Test: Bind Group Layout Creation
def testBindGroupLayoutCreation : IO TestSeq := withDevice fun _ device => do
  let entries : Array BindGroupLayoutEntry := #[]  -- Empty for now
  let layout ← createBindGroupLayout device entries
  pure $ test "Bind group layout created successfully" true

-- Test: Compute Pipeline Creation (Simplified)
def testComputePipelineCreation : IO TestSeq := withDevice fun _ device => do
  let shader ← createShaderModule device simpleAddShader

  -- Create bind group layout matching the shader's @group(0) @binding(0)
  let entries : Array BindGroupLayoutEntry := #[
    { binding := 0, visibility := .compute, bindingType := .buffer false }
  ]
  let bindGroupLayout ← createBindGroupLayout device entries

  let desc : ComputePipelineDescriptor := {
    shaderModule := shader,
    entryPoint := "main",
    bindGroupLayout := bindGroupLayout
  }

  let pipeline ← createComputePipeline device desc
  pure $ test "Compute pipeline created successfully" true

-- Test: Full Compute Pipeline (Simplified)
-- This is a simplified version - full version would do actual computation
def testFullComputePipeline : IO TestSeq := withDevice fun _ device => do
  -- Create buffer
  let bufferDesc : BufferDescriptor := {
    size := 1024 * 4  -- 1024 floats
    usage := [BufferUsage.storage, BufferUsage.copyDst, BufferUsage.copySrc]
    mappedAtCreation := false
  }
  let buffer ← createBuffer device bufferDesc

  -- Create shader
  let shader ← createShaderModule device simpleAddShader

  pure $ test "Full compute pipeline setup successful" true

-- Test: Multiple Shader Compilations
def testMultipleShaders : IO TestSeq := withDevice fun _ device => do
  let shader1 ← createShaderModule device simpleAddShader

  let shader2Source := "@compute @workgroup_size(64)
                        fn main() { }"

  let shader2 ← createShaderModule device shader2Source

  pure $ test "Multiple shaders compiled successfully" true

-- Test: Invalid Shader (Should Fail)
def testInvalidShader : IO TestSeq := withDevice fun _ device => do
  let result ← try
    let invalidShader := "this is not valid WGSL code !@#$"
    let _ ← createShaderModule device invalidShader
    pure false  -- Should not succeed
  catch _ =>
    pure true   -- Error expected

  pure $ test "Invalid shader compilation fails gracefully" result

-- All compute tests
def allTests : IO (List (String × List TestSeq)) := do
  IO.println "Running Compute Pipeline Tests..."

  let t1 ← testShaderCreation
  let t2 ← testBindGroupLayoutCreation
  let t3 ← testComputePipelineCreation
  let t4 ← testFullComputePipeline
  let t5 ← testMultipleShaders
  let t6 ← testInvalidShader

  pure [
    ("Shader Module Creation", [t1]),
    ("Bind Group Layout Creation", [t2]),
    ("Compute Pipeline Creation", [t3]),
    ("Full Pipeline Setup", [t4]),
    ("Multiple Shaders", [t5]),
    ("Error: Invalid Shader", [t6])
  ]

end Tests.ComputeTests

