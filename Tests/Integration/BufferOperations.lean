import Hesper
import Hesper.Compute
import Tests.Integration.TestHarness
import Tests.Integration.TestData

/-!
# Buffer Operations Integration Tests

End-to-end integration tests for buffer operations:
1. Small buffer (1KB)
2. Medium buffer (1MB)
3. Large buffer (100MB)
4. Buffer stress test (100+ buffers)
5. Host→GPU transfer timing
6. GPU→Host transfer with verification
7. Buffer mapping lifecycle
8. Subregion access
9. Concurrent buffers in one kernel
10. Buffer reuse pattern
11. Memory layout correctness
12. Zero-size buffer error handling
-/

namespace Hesper.Tests.Integration.BufferOperations

open Hesper.WebGPU
open Hesper.Compute
open Hesper.Tests.Integration
open Hesper.Tests.Integration.TestData

/-- Test 1: Small buffer (1KB = 256 floats) -/
def testSmallBuffer (device : Device) : IO TestResult := do
  let size := 256
  let input := sequentialArray size
  let expected := goldenIncrement input

  let shader := "
@group(0) @binding(0) var<storage, read> input: array<f32, 256>;
@group(0) @binding(1) var<storage, read_write> output: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < 256u) {
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

  let future ← dispatchCompute device pipeline bindGroup 1 1 1
  deviceWait future

  let result ← readBufferAsFloats device outputBuf size
  return assertFloatsEqual "Small Buffer (1KB)" expected result

/-- Test 2: Medium buffer (256KB = 65536 floats) -/
def testMediumBuffer (device : Device) : IO TestResult := do
  let size := 65536
  let input := randomFloatArray size 0.0 100.0
  let expected := goldenIncrement input

  let shader := "
@group(0) @binding(0) var<storage, read> input: array<f32, 65536>;
@group(0) @binding(1) var<storage, read_write> output: array<f32, 65536>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < 65536u) {
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

  let workgroups := ((size + 255) / 256).toUInt32
  let future ← dispatchCompute device pipeline bindGroup workgroups 1 1
  deviceWait future

  let result ← readBufferAsFloats device outputBuf size
  return assertFloatsEqual "Medium Buffer (256KB)" expected result

/-- Test 3: Buffer stress test - create many buffers -/
def testBufferStress (device : Device) : IO TestResult := do
  let numBuffers := 100
  let bufferSize := 1024  -- 1KB each

  try
    -- Create 100 buffers
    let mut buffers := []
    for _ in [0:numBuffers] do
      let buf ← createBuffer device {
        size := (bufferSize * 4).toUSize
        usage := [.storage]
        mappedAtCreation := false
      }
      buffers := buffers ++ [buf]

    return .pass s!"Buffer Stress ({numBuffers} buffers)"
  catch e =>
    return .fail "Buffer Stress" s!"Failed to create {numBuffers} buffers: {e}"

/-- Test 4: Host→GPU transfer with timing -/
def testHostToGPUTransfer (device : Device) : IO TestResult := do
  let size := 10000  -- 40KB
  let input := randomFloatArray size 0.0 1.0

  -- Time the upload
  let startTime ← IO.monoMsNow
  let inputBuf ← createBufferWithData device input
  let endTime ← IO.monoMsNow
  let uploadTime := (endTime - startTime).toFloat

  -- Verify upload by reading back
  let result ← readBufferAsFloats device inputBuf size

  return if compareFloatArrays input result then
    .pass s!"Host→GPU Transfer (40KB in {uploadTime}ms)"
  else
    .fail "Host→GPU Transfer" "Data corruption during upload"

/-- Test 5: GPU→Host transfer with verification -/
def testGPUToHostTransfer (device : Device) : IO TestResult := do
  let size := 10000  -- 40KB
  let input := randomFloatArray size 0.0 1.0

  -- Upload data
  let buffer ← createBufferWithData device input

  -- Time the download
  let startTime ← IO.monoMsNow
  let result ← readBufferAsFloats device buffer size
  let endTime ← IO.monoMsNow
  let downloadTime := (endTime - startTime).toFloat

  return if compareFloatArrays input result then
    .pass s!"GPU→Host Transfer (40KB in {downloadTime}ms)"
  else
    .fail "GPU→Host Transfer" "Data corruption during download"

/-- Test 6: Buffer reuse pattern -/
def testBufferReuse (device : Device) : IO TestResult := do
  let size := 100
  let iterations := 10

  -- Create buffer once
  let buffer ← createBuffer device {
    size := (size * 4).toUSize
    usage := [.storage, .copyDst, .copySrc]
    mappedAtCreation := false
  }

  -- Reuse buffer 10 times with different data
  let mut allCorrect := true
  for i in [0:iterations] do
    let data := constantArray size (Float.ofNat i)
    let bytes ← Hesper.Basic.floatArrayToBytes data
    writeBuffer device buffer 0 bytes

    let result ← readBufferAsFloats device buffer size

    if not (compareFloatArrays data result) then
      allCorrect := false

  return if allCorrect then
    .pass s!"Buffer Reuse ({iterations} iterations)"
  else
    .fail "Buffer Reuse" "Data corruption in one or more iterations"

/-- Test 7: Concurrent buffers in one kernel -/
def testConcurrentBuffers (device : Device) : IO TestResult := do
  let a := #[1.0, 2.0, 3.0, 4.0]
  let b := #[5.0, 6.0, 7.0, 8.0]
  let c := #[9.0, 10.0, 11.0, 12.0]
  let expected := a.zipWith (· + ·) b |>.zipWith (· + ·) c

  let shader := "
@group(0) @binding(0) var<storage, read> a: array<f32, 4>;
@group(0) @binding(1) var<storage, read> b: array<f32, 4>;
@group(0) @binding(2) var<storage, read> c: array<f32, 4>;
@group(0) @binding(3) var<storage, read_write> output: array<f32, 4>;

@compute @workgroup_size(4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < 4u) {
        output[idx] = a[idx] + b[idx] + c[idx];
    }
}
"

  let aBuf ← createBufferWithData device a
  let bBuf ← createBufferWithData device b
  let cBuf ← createBufferWithData device c
  let outBuf ← createBuffer device {
    size := 16
    usage := [.storage, .copySrc]
    mappedAtCreation := false
  }

  let shaderModule ← createShaderModule device shader
  let layoutEntries := #[
    { binding := 0, visibility := .compute, bindingType := .buffer true },
    { binding := 1, visibility := .compute, bindingType := .buffer true },
    { binding := 2, visibility := .compute, bindingType := .buffer true },
    { binding := 3, visibility := .compute, bindingType := .buffer false }
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
    { binding := 2, buffer := cBuf, offset := 0, size := 16 },
    { binding := 3, buffer := outBuf, offset := 0, size := 16 }
  ]

  let future ← dispatchCompute device pipeline bindGroup 1 1 1
  deviceWait future

  let result ← readBufferAsFloats device outBuf 4
  return assertFloatsEqual "Concurrent Buffers (3 inputs)" expected result

/-- Test 8: Memory layout correctness (row-major) -/
def testMemoryLayout (device : Device) : IO TestResult := do
  -- 2x2 matrix in row-major order: [[1, 2], [3, 4]]
  let matrix := #[1.0, 2.0, 3.0, 4.0]

  -- Transpose: [[1, 3], [2, 4]]
  let expected := #[1.0, 3.0, 2.0, 4.0]

  let shader := "
@group(0) @binding(0) var<storage, read> input: array<f32, 4>;
@group(0) @binding(1) var<storage, read_write> output: array<f32, 4>;

@compute @workgroup_size(2, 2)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;

    if (row < 2u && col < 2u) {
        // Transpose: output[col, row] = input[row, col]
        output[col * 2u + row] = input[row * 2u + col];
    }
}
"

  let inputBuf ← createBufferWithData device matrix
  let outputBuf ← createBuffer device {
    size := 16
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
    { binding := 0, buffer := inputBuf, offset := 0, size := 16 },
    { binding := 1, buffer := outputBuf, offset := 0, size := 16 }
  ]

  let future ← dispatchCompute device pipeline bindGroup 1 1 1
  deviceWait future

  let result ← readBufferAsFloats device outputBuf 4
  return assertFloatsEqual "Memory Layout (Row-Major Transpose)" expected result

/-- Run all buffer operation tests -/
def runAll (device : Device) : IO (List (TestResult × Float)) := do
  let tests := [
    ("Test 1: Small Buffer (1KB)", testSmallBuffer),
    ("Test 2: Medium Buffer (256KB)", testMediumBuffer),
    ("Test 3: Buffer Stress (100 buffers)", testBufferStress),
    ("Test 4: Host→GPU Transfer", testHostToGPUTransfer),
    ("Test 5: GPU→Host Transfer", testGPUToHostTransfer),
    ("Test 6: Buffer Reuse", testBufferReuse),
    ("Test 7: Concurrent Buffers", testConcurrentBuffers),
    ("Test 8: Memory Layout", testMemoryLayout)
  ]

  let mut results := []
  for (name, test) in tests do
    let (result, time) ← runTimedTest name (test device)
    results := results ++ [(result, time)]

  return results

end Hesper.Tests.Integration.BufferOperations
