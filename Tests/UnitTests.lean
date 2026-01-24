import Hesper
import Hesper.Basic
import Hesper.Compute

/-!
# Unit Tests for Hesper Components

Tests each component in isolation to verify correct functionality.
-/

namespace Tests.Unit
open Hesper.WebGPU
open Hesper.Compute

/-- Test 1: Float to bytes conversion (Lean side) -/
def testFloatToBytes : IO Unit := do
  IO.println "Test 1: Float to bytes conversion"

  let f : Float := 42.0
  let bytes := Hesper.Basic.floatToBytes f

  IO.println s!"  Input: {f}"
  IO.println s!"  Bytes size: {bytes.size}"
  IO.print "  Bytes (hex): "
  for i in [0:bytes.size] do
    let b := bytes.get! i
    IO.print s!"{b} "
  IO.println ""

/-- Test 2: Bytes to float conversion (Lean side - BROKEN) -/
def testBytesToFloat32 : IO Unit := do
  IO.println "\nTest 2: Bytes to Float32 conversion (KNOWN BUG)"

  -- Create byte array for 42.0 in IEEE 754 f32 format
  -- 42.0 = 0x42280000 in little-endian: 00 28 42 00
  let bytes : ByteArray := ByteArray.mk #[0x00, 0x28, 0x42, 0x00]

  IO.print "  Input bytes (hex): "
  for i in [0:bytes.size] do
    let b := bytes.get! i
    IO.print s!"{b} "
  IO.println ""

  let result := Hesper.Basic.bytesToFloat32 bytes
  IO.println s!"  Result: {result}"
  IO.println s!"  Expected: 42.0"

  if (result - 42.0).abs < 0.01 then
    IO.println "  ✅ PASS"
  else
    IO.println "  ❌ FAIL"

/-- Test 3: GPU buffer write -/
def testGPUBufferWrite : IO Unit := do
  IO.println "\nTest 3: GPU buffer write"

  let inst ← Hesper.init
  let device ← getDevice inst

  -- Create test data: [1.0, 2.0, 3.0, 4.0]
  let testData : Array Float := #[1.0, 2.0, 3.0, 4.0]
  let dataBytes := Hesper.Basic.floatArrayToBytes testData

  -- Create buffer
  let buffer ← createBuffer device {
    size := (dataBytes.size).toUSize
    usage := [.storage, .copyDst, .copySrc]
    mappedAtCreation := false
  }

  -- Write to buffer
  writeBuffer device buffer 0 dataBytes

  IO.println "  ✅ Buffer created and data written (check C++ debug output)"

/-- Test 4: GPU buffer read (C++ level) -/
def testGPUBufferRead : IO Unit := do
  IO.println "\nTest 4: GPU buffer read (C++ level)"

  let inst ← Hesper.init
  let device ← getDevice inst

  -- Create and write test data
  let testData : Array Float := #[1.0, 2.0, 3.0, 4.0]
  let dataBytes := Hesper.Basic.floatArrayToBytes testData

  let buffer ← createBuffer device {
    size := (dataBytes.size).toUSize
    usage := [.storage, .copyDst, .copySrc]
    mappedAtCreation := false
  }

  writeBuffer device buffer 0 dataBytes

  -- Read back
  let readBytes ← mapBufferRead device buffer 0 (dataBytes.size).toUSize

  IO.println s!"  Wrote {dataBytes.size} bytes, read {readBytes.size} bytes"
  IO.println "  ✅ Buffer read completed (check C++ debug output for actual values)"

/-- Test 5: GPU compute shader execution -/
def testGPUCompute : IO Unit := do
  IO.println "\nTest 5: GPU compute shader execution"

  let inst ← Hesper.init
  let device ← getDevice inst

  -- Create buffer with initial data
  let testData : Array Float := #[1.0, 2.0, 3.0, 4.0]
  let dataBytes := Hesper.Basic.floatArrayToBytes testData

  let buffer ← createBuffer device {
    size := (dataBytes.size).toUSize
    usage := [.storage, .copyDst, .copySrc]
    mappedAtCreation := false
  }

  writeBuffer device buffer 0 dataBytes

  -- Create shader that writes 42.0
  let shaderCode := "
@group(0) @binding(0) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&output)) {
        output[idx] = 42.0;
    }
}
"

  let shaderModule ← createShaderModule device shaderCode

  let layoutEntries := #[
    { binding := 0, visibility := .compute, bindingType := .buffer false : BindGroupLayoutEntry }
  ]
  let bindGroupLayout ← createBindGroupLayout device layoutEntries

  let pipeline ← createComputePipeline device {
    shaderModule := shaderModule
    entryPoint := "main"
    bindGroupLayout := bindGroupLayout
  }

  let bindEntries := #[
    { binding := 0, buffer := buffer, offset := 0, size := (dataBytes.size).toUSize : BindGroupEntry }
  ]
  let bindGroup ← createBindGroup device bindGroupLayout bindEntries

  -- Dispatch compute
  let future ← dispatchCompute device pipeline bindGroup 4 1 1
  deviceWait future

  -- Read back
  let readBytes ← mapBufferRead device buffer 0 (dataBytes.size).toUSize

  IO.println "  ✅ Compute shader executed (check C++ debug output for results)"
  IO.println "  Expected C++ output: 42.00, 42.00, 42.00, 42.00"

/-- Test 6: End-to-end with Lean conversion (currently fails) -/
def testEndToEnd : IO Unit := do
  IO.println "\nTest 6: End-to-end GPU compute + Lean conversion"

  let inst ← Hesper.init
  let device ← getDevice inst

  let testData : Array Float := #[1.0, 2.0, 3.0, 4.0]
  let dataBytes := Hesper.Basic.floatArrayToBytes testData

  let buffer ← createBuffer device {
    size := (dataBytes.size).toUSize
    usage := [.storage, .copyDst, .copySrc]
    mappedAtCreation := false
  }

  writeBuffer device buffer 0 dataBytes

  let shaderCode := "
@group(0) @binding(0) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&output)) {
        output[idx] = 42.0;
    }
}
"

  let shaderModule ← createShaderModule device shaderCode

  let layoutEntries := #[
    { binding := 0, visibility := .compute, bindingType := .buffer false : BindGroupLayoutEntry }
  ]
  let bindGroupLayout ← createBindGroupLayout device layoutEntries

  let pipeline ← createComputePipeline device {
    shaderModule := shaderModule
    entryPoint := "main"
    bindGroupLayout := bindGroupLayout
  }

  let bindEntries := #[
    { binding := 0, buffer := buffer, offset := 0, size := (dataBytes.size).toUSize : BindGroupEntry }
  ]
  let bindGroup ← createBindGroup device bindGroupLayout bindEntries

  let future ← dispatchCompute device pipeline bindGroup 4 1 1
  deviceWait future

  let readBytes ← mapBufferRead device buffer 0 (dataBytes.size).toUSize
  let results ← Hesper.Basic.bytesToFloatArray readBytes

  IO.println "  Results from Lean conversion:"
  for i in [0:results.size] do
    IO.println s!"    Element {i}: {results[i]!}"

  -- Check if all values are close to 42.0
  let allCorrect := results.all fun v => (v - 42.0).abs < 0.01

  if allCorrect then
    IO.println "  ✅ PASS - All values are 42.0"
  else
    IO.println "  ❌ FAIL - Conversion bug detected"

/-- Main test runner -/
def main : IO Unit := do
  IO.println "╔══════════════════════════════════════╗"
  IO.println "║   Hesper Unit Tests                  ║"
  IO.println "╚══════════════════════════════════════╝"
  IO.println ""

  testFloatToBytes
  testBytesToFloat32
  testGPUBufferWrite
  testGPUBufferRead
  testGPUCompute
  testEndToEnd

  IO.println ""
  IO.println "═══════════════════════════════════════"
  IO.println "Test Summary:"
  IO.println "  Tests 1-5: Component isolation tests"
  IO.println "  Test 6: End-to-end test (EXPECTED TO FAIL)"
  IO.println "═══════════════════════════════════════"

end Tests.Unit

-- Top-level main that delegates to the namespaced version
def main : IO Unit := Tests.Unit.main
