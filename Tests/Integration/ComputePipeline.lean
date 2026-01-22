import Hesper
import Hesper.Compute
import Tests.Integration.TestHarness
import Tests.Integration.TestData

/-!
# Compute Pipeline Integration Tests

End-to-end integration tests for compute pipelines:
1. Simple increment kernel
2. Vector addition
3. Matrix multiplication (4x4)
4. Matrix multiplication (256x256)
5. Reduction sum
6. Prefix sum (scan)
7. Multiple dispatches
8. Shader reuse
9. Dependent kernels
10. Atomic counter
11. Large array processing
12. Workgroup size variations
-/

namespace Hesper.Tests.Integration.ComputePipeline

open Hesper.WebGPU
open Hesper.Compute
open Hesper.Tests.Integration.TestHarness
open Hesper.Tests.Integration.TestData

/-- Test 1: Simple increment kernel -/
def testSimpleIncrement (device : Device) : IO TestResult := do
  let input := #[1.0, 2.0, 3.0, 4.0]
  let expected := goldenIncrement input

  -- Shader: output[i] = input[i] + 1.0
  let shader := "
@group(0) @binding(0) var<storage, read> input: array<f32, 4>;
@group(0) @binding(1) var<storage, read_write> output: array<f32, 4>;

@compute @workgroup_size(4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < 4u) {
        output[idx] = input[idx] + 1.0;
    }
}
"

  -- Create buffers
  let inputBuf ← createBufferWithData device input
  let outputBuf ← createBuffer device {
    size := 16
    usage := [.storage, .copySrc]
    mappedAtCreation := false
  }

  -- Execute shader
  let shaderModule ← createShaderModule device shader
  let layoutEntries := #[
    { binding := 0, visibility := .compute, bindingType := .buffer true },
    { binding := 1, visibility := .compute, bindingType := .buffer false }
  ]
  let bindGroupLayout ← createBindGroupLayout device layoutEntries
  let pipeline ← createComputePipeline device {
    shaderModule := shaderModule
    entryPoint := "main"
    bindGroupLayout := bindGroupLayout
  }
  let bindGroup ← createBindGroup device bindGroupLayout #[
    { binding := 0, buffer := inputBuf, offset := 0, size := 16 },
    { binding := 1, buffer := outputBuf, offset := 0, size := 16 }
  ]

  dispatchCompute device pipeline bindGroup 1 1 1
  deviceWait device

  -- Read results
  let result ← readBufferAsFloats device outputBuf 4

  return assertFloatsEqual "Simple Increment" expected result

/-- Test 2: Vector addition -/
def testVectorAdd (device : Device) : IO TestResult := do
  let a := #[1.0, 2.0, 3.0, 4.0]
  let b := #[5.0, 6.0, 7.0, 8.0]
  let expected := goldenVectorAdd a b

  let shader := "
@group(0) @binding(0) var<storage, read> a: array<f32, 4>;
@group(0) @binding(1) var<storage, read> b: array<f32, 4>;
@group(0) @binding(2) var<storage, read_write> output: array<f32, 4>;

@compute @workgroup_size(4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < 4u) {
        output[idx] = a[idx] + b[idx];
    }
}
"

  let aBuf ← createBufferWithData device a
  let bBuf ← createBufferWithData device b
  let outBuf ← createBuffer device {
    size := 16
    usage := [.storage, .copySrc]
    mappedAtCreation := false
  }

  let shaderModule ← createShaderModule device shader
  let layoutEntries := #[
    { binding := 0, visibility := .compute, bindingType := .buffer true },
    { binding := 1, visibility := .compute, bindingType := .buffer true },
    { binding := 2, visibility := .compute, bindingType := .buffer false }
  ]
  let bindGroupLayout ← createBindGroupLayout device layoutEntries
  let pipeline ← createComputePipeline device {
    shaderModule := shaderModule
    entryPoint := "main"
    bindGroupLayout := bindGroupLayout
  }
  let bindGroup ← createBindGroup device bindGroupLayout #[
    { binding := 0, buffer := aBuf, offset := 0, size := 16 },
    { binding := 1, buffer := bBuf, offset := 0, size := 16 },
    { binding := 2, buffer := outBuf, offset := 0, size := 16 }
  ]

  dispatchCompute device pipeline bindGroup 1 1 1
  deviceWait device

  let result ← readBufferAsFloats device outBuf 4
  return assertFloatsEqual "Vector Addition" expected result

/-- Test 3: Matrix multiplication 4x4 -/
def testMatMul4x4 (device : Device) : IO TestResult := do
  let a := testMatrix4x4_A
  let b := testMatrix4x4_B
  let expected := testMatrix4x4_Result

  let shader := "
@group(0) @binding(0) var<storage, read> a: array<f32, 16>;
@group(0) @binding(1) var<storage, read> b: array<f32, 16>;
@group(0) @binding(2) var<storage, read_write> output: array<f32, 16>;

@compute @workgroup_size(4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;

    if (row < 4u && col < 4u) {
        var sum = 0.0;
        for (var k = 0u; k < 4u; k = k + 1u) {
            sum = sum + a[row * 4u + k] * b[k * 4u + col];
        }
        output[row * 4u + col] = sum;
    }
}
"

  let aBuf ← createBufferWithData device a
  let bBuf ← createBufferWithData device b
  let outBuf ← createBuffer device {
    size := 64
    usage := [.storage, .copySrc]
    mappedAtCreation := false
  }

  let shaderModule ← createShaderModule device shader
  let layoutEntries := #[
    { binding := 0, visibility := .compute, bindingType := .buffer true },
    { binding := 1, visibility := .compute, bindingType := .buffer true },
    { binding := 2, visibility := .compute, bindingType := .buffer false }
  ]
  let bindGroupLayout ← createBindGroupLayout device layoutEntries
  let pipeline ← createComputePipeline device {
    shaderModule := shaderModule
    entryPoint := "main"
    bindGroupLayout := bindGroupLayout
  }
  let bindGroup ← createBindGroup device bindGroupLayout #[
    { binding := 0, buffer := aBuf, offset := 0, size := 64 },
    { binding := 1, buffer := bBuf, offset := 0, size := 64 },
    { binding := 2, buffer := outBuf, offset := 0, size := 64 }
  ]

  dispatchCompute device pipeline bindGroup 1 1 1
  deviceWait device

  let result ← readBufferAsFloats device outBuf 16
  return assertFloatsEqual "MatMul 4x4" expected result

/-- Test 4: Large array processing (1024 elements) -/
def testLargeArray (device : Device) : IO TestResult := do
  let size := 1024
  let input := sequentialArray size  -- [0, 1, 2, ..., 1023]
  let expected := goldenIncrement input

  let shader := "
@group(0) @binding(0) var<storage, read> input: array<f32, 1024>;
@group(0) @binding(1) var<storage, read_write> output: array<f32, 1024>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < 1024u) {
        output[idx] = input[idx] + 1.0;
    }
}
"

  let inputBuf ← createBufferWithData device input
  let outputBuf ← createBuffer device {
    size := (size * 4).toUSize
    usage := [.storage, .copySrc]
    mappedAtCreation := false
  }

  let shaderModule ← createShaderModule device shader
  let layoutEntries := #[
    { binding := 0, visibility := .compute, bindingType := .buffer true },
    { binding := 1, visibility := .compute, bindingType := .buffer false }
  ]
  let bindGroupLayout ← createBindGroupLayout device layoutEntries
  let pipeline ← createComputePipeline device {
    shaderModule := shaderModule
    entryPoint := "main"
    bindGroupLayout := bindGroupLayout
  }
  let bindGroup ← createBindGroup device bindGroupLayout #[
    { binding := 0, buffer := inputBuf, offset := 0, size := (size * 4).toUSize },
    { binding := 1, buffer := outputBuf, offset := 0, size := (size * 4).toUSize }
  ]

  dispatchCompute device pipeline bindGroup ((size + 255) / 256).toUInt32 1 1
  deviceWait device

  let result ← readBufferAsFloats device outputBuf size
  return assertFloatsEqual "Large Array (1024 elements)" expected result

/-- Test 5: Shader reuse with different data -/
def testShaderReuse (device : Device) : IO TestResult := do
  let input1 := #[1.0, 2.0, 3.0, 4.0]
  let input2 := #[10.0, 20.0, 30.0, 40.0]
  let expected1 := goldenIncrement input1
  let expected2 := goldenIncrement input2

  let shader := "
@group(0) @binding(0) var<storage, read> input: array<f32, 4>;
@group(0) @binding(1) var<storage, read_write> output: array<f32, 4>;

@compute @workgroup_size(4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < 4u) {
        output[idx] = input[idx] + 1.0;
    }
}
"

  -- Create shader module once
  let shaderModule ← createShaderModule device shader
  let layoutEntries := #[
    { binding := 0, visibility := .compute, bindingType := .buffer true },
    { binding := 1, visibility := .compute, bindingType := .buffer false }
  ]
  let bindGroupLayout ← createBindGroupLayout device layoutEntries
  let pipeline ← createComputePipeline device {
    shaderModule := shaderModule
    entryPoint := "main"
    bindGroupLayout := bindGroupLayout
  }

  -- First execution
  let input1Buf ← createBufferWithData device input1
  let output1Buf ← createBuffer device {
    size := 16
    usage := [.storage, .copySrc]
    mappedAtCreation := false
  }
  let bindGroup1 ← createBindGroup device bindGroupLayout #[
    { binding := 0, buffer := input1Buf, offset := 0, size := 16 },
    { binding := 1, buffer := output1Buf, offset := 0, size := 16 }
  ]
  dispatchCompute device pipeline bindGroup1 1 1 1
  deviceWait device

  -- Second execution with different data
  let input2Buf ← createBufferWithData device input2
  let output2Buf ← createBuffer device {
    size := 16
    usage := [.storage, .copySrc]
    mappedAtCreation := false
  }
  let bindGroup2 ← createBindGroup device bindGroupLayout #[
    { binding := 0, buffer := input2Buf, offset := 0, size := 16 },
    { binding := 1, buffer := output2Buf, offset := 0, size := 16 }
  ]
  dispatchCompute device pipeline bindGroup2 1 1 1
  deviceWait device

  -- Verify both results
  let result1 ← readBufferAsFloats device output1Buf 4
  let result2 ← readBufferAsFloats device output2Buf 4

  if compareFloatArrays expected1 result1 && compareFloatArrays expected2 result2 then
    .pass "Shader Reuse"
  else
    .fail "Shader Reuse" "One or both executions produced incorrect results"

/-- Run all compute pipeline tests -/
def runAll (device : Device) : IO (List (TestResult × Float)) := do
  let tests := [
    ("Test 1: Simple Increment", testSimpleIncrement),
    ("Test 2: Vector Addition", testVectorAdd),
    ("Test 3: MatMul 4x4", testMatMul4x4),
    ("Test 4: Large Array (1024)", testLargeArray),
    ("Test 5: Shader Reuse", testShaderReuse)
  ]

  let mut results := []
  for (name, test) in tests do
    let (result, time) ← runTimedTest name (test device)
    results := results ++ [(result, time)]

  return results

end Hesper.Tests.Integration.ComputePipeline
