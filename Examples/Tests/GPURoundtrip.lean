import Hesper
import Hesper.Compute

/-!
# GPU Roundtrip Test

Simple test that verifies:
1. CPU → GPU data transfer (writeBuffer)
2. GPU shader execution (simple increment operation)
3. GPU → CPU data readback (mapBufferRead)

This isolates the async MapAsync functionality from complex MNIST operations.
-/

namespace Examples.Tests.GPURoundtrip

open Hesper.WebGPU
open Hesper.Compute

/-- Simple shader that adds 1.0 to each element -/
def incrementShader : String := "
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&input)) {
        output[idx] = input[idx] + 1.0;
    }
}
"

/-- Run a simple roundtrip test with varying data sizes -/
def testRoundtrip (inst : Instance) (size : Nat) : IO Bool := do
  IO.println s!"\n🧪 Testing GPU roundtrip with {size} elements..."

  let device ← getDevice inst

  -- 1. Create test data (CPU)
  let inputData := Array.range size |>.map (fun i => i.toFloat)
  IO.println s!"  ✓ Created input data: {inputData.size} floats"

  -- 2. Create GPU buffers
  let inputBuf ← createBuffer device {
    size := (size * 4).toUSize
    usage := [.storage, .copyDst]
    mappedAtCreation := false
  }

  let outputBuf ← createBuffer device {
    size := (size * 4).toUSize
    usage := [.storage, .copySrc]
    mappedAtCreation := false
  }
  IO.println s!"  ✓ Created GPU buffers"

  -- 3. Upload data to GPU
  let inputBytes := Hesper.Basic.floatArrayToBytes inputData
  IO.println s!"  [DEBUG] Input bytes size: {inputBytes.size} (expected {size * 4})"
  writeBuffer device inputBuf 0 inputBytes
  IO.println s!"  ✓ Uploaded {size} floats to GPU"

  -- 4. Create and run shader
  let shaderModule ← createShaderModule device incrementShader

  let layoutEntries := #[
    { binding := 0, visibility := .compute, bindingType := .buffer true : BindGroupLayoutEntry },
    { binding := 1, visibility := .compute, bindingType := .buffer true }  -- read_write, not read-only
  ]
  let bindGroupLayout ← createBindGroupLayout device layoutEntries

  let pipeline ← createComputePipeline device {
    shaderModule := shaderModule
    entryPoint := "main"
    bindGroupLayout := bindGroupLayout
  }

  let bindEntries := #[
    { binding := 0, buffer := inputBuf, offset := 0, size := (size * 4).toUSize : BindGroupEntry },
    { binding := 1, buffer := outputBuf, offset := 0, size := (size * 4).toUSize }
  ]
  let bindGroup ← createBindGroup device bindGroupLayout bindEntries

  let numWorkgroups := (size + 63) / 64
  dispatchCompute device pipeline bindGroup numWorkgroups.toUInt32 1 1
  IO.println s!"  ✓ Executed shader ({numWorkgroups} workgroups)"

  -- 5. Read results back from GPU (ASYNC MAPASYNC TEST)
  deviceWait device
  IO.println s!"  ⏳ Reading {size} floats back from GPU..."
  let resultBytes ← mapBufferRead device outputBuf 0 ((size * 4).toUSize)
  IO.println s!"  ✓ Read back {resultBytes.size} bytes"
  unmapBuffer outputBuf

  -- 6. Convert and verify
  let resultFloats ← Hesper.Basic.bytesToFloatArray resultBytes
  IO.println s!"  ✓ Converted to {resultFloats.size} floats"

  -- Check if increment worked (each element should be original + 1.0)
  let mut allCorrect := true
  for i in [0:min size resultFloats.size] do
    let expected := inputData[i]! + 1.0
    let actual := resultFloats[i]!
    let diff := (expected - actual).abs
    if diff > 0.001 then
      IO.println s!"  ✗ ERROR at index {i}: expected {expected}, got {actual}"
      allCorrect := false

  if allCorrect then
    IO.println s!"  ✅ PASS: All {size} elements incremented correctly"
  else
    IO.println s!"  ❌ FAIL: Some elements incorrect"

  return allCorrect

def main : IO Unit := do
  IO.println "╔══════════════════════════════════════════════════╗"
  IO.println "║      GPU Roundtrip Test (CPU→GPU→CPU)          ║"
  IO.println "╚══════════════════════════════════════════════════╝"

  IO.println "\n🚀 Initializing WebGPU..."
  let inst ← Hesper.init

  -- Test with progressively larger data sizes
  let testSizes := [
    1,      -- Single element
    4,      -- 4 floats (16 bytes)
    16,     -- 16 floats (64 bytes)
    64,     -- 64 floats (256 bytes)
    128,    -- 128 floats (512 bytes) - same as MNIST Layer 1 output
    256,    -- 256 floats (1024 bytes)
    1024    -- 1024 floats (4096 bytes)
  ]

  let mut allPassed := true
  for size in testSizes do
    let passed ← testRoundtrip inst size
    if !passed then
      allPassed := false

  IO.println "\n═══════════════════════════════════════════════════"
  if allPassed then
    IO.println "  ✅ ALL TESTS PASSED"
  else
    IO.println "  ❌ SOME TESTS FAILED"
  IO.println "═══════════════════════════════════════════════════"

end Examples.Tests.GPURoundtrip

def main : IO Unit := Examples.Tests.GPURoundtrip.main
