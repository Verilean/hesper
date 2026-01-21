import Hesper
import Hesper.Compute

/-!
# FFI Boundary Tests

Tests the Lean ↔ C++ data conversion boundary.

**Critical**: These tests verify the FFI layer, not just Lean code.
- Test 1: Lean writes → C++ reads (validates Lean→C++ direction)
- Test 2: C++ writes → Lean reads (validates C++→Lean direction)

Writing in Lean and reading in Lean doesn't test the FFI boundary!
-/

namespace Tests.FFIBoundary

open Hesper.WebGPU
open Hesper.Compute

/-- Test 1: Lean writes data, C++ reads and verifies it

    This tests: Lean ByteArray creation → C++ receives correct bytes
-/
def testLeanWriteCppRead : IO Unit := do
  IO.println "Test 1: Lean writes → C++ reads"
  IO.println "─────────────────────────────────"

  let inst ← Hesper.init
  let device ← getDevice inst

  -- Create test data in Lean: [1.0, 2.0, 3.0, 4.0] as f32 bytes
  let testData := ByteArray.mk #[
    0x00, 0x00, 0x80, 0x3F,  -- 1.0 in f32 little-endian
    0x00, 0x00, 0x00, 0x40,  -- 2.0
    0x00, 0x00, 0x40, 0x40,  -- 3.0
    0x00, 0x00, 0x80, 0x40   -- 4.0
  ]

  IO.println s!"Lean created {testData.size} bytes"

  -- Write to GPU buffer (C++ will log what it receives)
  let buffer ← createBuffer device {
    size := 16
    usage := [.storage, .copyDst]
    mappedAtCreation := false
  }

  IO.println "Writing data to GPU buffer..."
  IO.println "Check C++ debug output above for:"
  IO.println "  Expected: Data as floats: 1.00, 2.00, 3.00, 4.00"
  writeBuffer device buffer 0 testData

  IO.println "✓ Test 1 complete - check C++ debug log\n"

/-- Test 2: C++ writes data, Lean reads and verifies it

    This tests: GPU computation → C++ writes bytes → Lean receives correct bytes
    Note: We can't make C++ write arbitrary data, but we can use GPU to write
    known values and verify Lean reads the raw bytes correctly.
-/
def testCppWriteLeanRead : IO Unit := do
  IO.println "Test 2: C++ writes → Lean reads"
  IO.println "─────────────────────────────────"

  let inst ← Hesper.init
  let device ← getDevice inst

  -- Create buffer and have GPU write known pattern
  let buffer ← createBuffer device {
    size := 16
    usage := [.storage, .copySrc]
    mappedAtCreation := false
  }

  -- Shader that writes specific values we can verify byte-by-byte
  let shader := "
@group(0) @binding(0) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < 4u) {
        // Write distinctive values: 10.0, 20.0, 30.0, 40.0
        output[idx] = f32(idx + 1u) * 10.0;
    }
}
"

  let shaderModule ← createShaderModule device shader

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
    { binding := 0, buffer := buffer, offset := 0, size := 16 : BindGroupEntry }
  ]
  let bindGroup ← createBindGroup device bindGroupLayout bindEntries

  dispatchCompute device pipeline bindGroup 4 1 1
  deviceWait device

  -- Read back bytes from C++
  let bytes ← mapBufferRead device buffer 0 16
  unmapBuffer buffer

  IO.println s!"Lean received {bytes.size} bytes from C++"
  IO.println "Raw bytes received:"

  -- Verify byte-by-byte (not using float conversion!)
  -- 10.0 = 0x41200000, 20.0 = 0x41A00000, 30.0 = 0x41F00000, 40.0 = 0x42200000
  let expected : List (List UInt8) := [
    [0x00, 0x00, 0x20, 0x41],  -- 10.0
    [0x00, 0x00, 0xA0, 0x41],  -- 20.0
    [0x00, 0x00, 0xF0, 0x41],  -- 30.0
    [0x00, 0x00, 0x20, 0x42]   -- 40.0
  ]

  let mut allCorrect := true
  for i in [0:4] do
    let offset := i * 4
    let exp := expected[i]!
    let actual := [bytes.get! offset, bytes.get! (offset+1),
                   bytes.get! (offset+2), bytes.get! (offset+3)]

    let isMatch := exp == actual
    allCorrect := allCorrect && isMatch

    IO.print s!"  Float {i}: "
    for b in actual do
      IO.print s!"{b.toNat} "
    IO.println s!" {if isMatch then "✓" else "✗"}"

  if allCorrect then
    IO.println "✅ PASS: Lean correctly received bytes from C++\n"
  else
    IO.println "❌ FAIL: Byte mismatch\n"

/-- Test 3: Round-trip test

    Lean writes → GPU processes → C++ reads → returns bytes → Lean verifies
    This tests the complete FFI pipeline.
-/
def testRoundTrip : IO Unit := do
  IO.println "Test 3: Round-trip (Lean → GPU → C++ → Lean)"
  IO.println "─────────────────────────────────────────────"

  let inst ← Hesper.init
  let device ← getDevice inst

  -- Input: [5.0, 10.0, 15.0, 20.0]
  let inputData := ByteArray.mk #[
    0x00, 0x00, 0xA0, 0x40,  -- 5.0
    0x00, 0x00, 0x20, 0x41,  -- 10.0
    0x00, 0x00, 0x70, 0x41,  -- 15.0
    0x00, 0x00, 0xA0, 0x41   -- 20.0
  ]

  let inputBuf ← createBuffer device {
    size := 16
    usage := [.storage, .copyDst]
    mappedAtCreation := false
  }

  let outputBuf ← createBuffer device {
    size := 16
    usage := [.storage, .copySrc]
    mappedAtCreation := false
  }

  writeBuffer device inputBuf 0 inputData

  -- GPU doubles the values
  let shader := "
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < 4u) {
        output[idx] = input[idx] * 2.0;
    }
}
"

  let shaderModule ← createShaderModule device shader

  let layoutEntries := #[
    { binding := 0, visibility := .compute, bindingType := .buffer false : BindGroupLayoutEntry },
    { binding := 1, visibility := .compute, bindingType := .buffer false : BindGroupLayoutEntry }
  ]
  let bindGroupLayout ← createBindGroupLayout device layoutEntries

  let pipeline ← createComputePipeline device {
    shaderModule := shaderModule
    entryPoint := "main"
    bindGroupLayout := bindGroupLayout
  }

  let bindEntries := #[
    { binding := 0, buffer := inputBuf, offset := 0, size := 16 : BindGroupEntry },
    { binding := 1, buffer := outputBuf, offset := 0, size := 16 : BindGroupEntry }
  ]
  let bindGroup ← createBindGroup device bindGroupLayout bindEntries

  dispatchCompute device pipeline bindGroup 4 1 1
  deviceWait device

  let resultBytes ← mapBufferRead device outputBuf 0 16
  unmapBuffer outputBuf

  IO.println "Checking C++ debug output showed: 10.00, 20.00, 30.00, 40.00"
  IO.println s!"Lean received {resultBytes.size} bytes back"

  -- Expected: 10.0, 20.0, 30.0, 40.0 (doubled values)
  let expected : List (List UInt8) := [
    [0x00, 0x00, 0x20, 0x41],  -- 10.0
    [0x00, 0x00, 0xA0, 0x41],  -- 20.0
    [0x00, 0x00, 0xF0, 0x41],  -- 30.0
    [0x00, 0x00, 0x20, 0x42]   -- 40.0
  ]

  let mut allCorrect := true
  for i in [0:4] do
    let offset := i * 4
    let exp := expected[i]!
    let actual := [resultBytes.get! offset, resultBytes.get! (offset+1),
                   resultBytes.get! (offset+2), resultBytes.get! (offset+3)]

    let isMatch := exp == actual
    if !isMatch then
      allCorrect := false

    if !isMatch then
      IO.print s!"  Mismatch at {i}: got "
      for b in actual do IO.print s!"{b.toNat} "
      IO.print " expected "
      for b in exp do IO.print s!"{b.toNat} "
      IO.println ""

  if allCorrect then
    IO.println "✅ PASS: Round-trip maintains data integrity\n"
  else
    IO.println "❌ FAIL: Data corrupted in round-trip\n"

end Tests.FFIBoundary

def main : IO Unit := do
  IO.println "╔══════════════════════════════════════════╗"
  IO.println "║  FFI Boundary Tests (Lean ↔ C++)        ║"
  IO.println "╚══════════════════════════════════════════╝\n"

  Tests.FFIBoundary.testLeanWriteCppRead
  Tests.FFIBoundary.testCppWriteLeanRead
  Tests.FFIBoundary.testRoundTrip

  IO.println "═══════════════════════════════════════════"
  IO.println "All FFI boundary tests complete"
  IO.println "═══════════════════════════════════════════"
