import Hesper
import Hesper.Compute

/-!
# GPU Double Test

Test: GPU reads input array and doubles each element.
Input:  [1.0, 2.0, 3.0, 4.0]
Expected: [2.0, 4.0, 6.0, 8.0]

This verifies:
- GPU can read input values correctly at different positions
- GPU can write different values to different positions
- Order is preserved (no swap bugs)
- No position-specific bugs
-/

namespace Examples.Tests.SimpleWrite

open Hesper.WebGPU
open Hesper.Compute

/-- Shader that doubles each input element -/
def doubleShader : String := "
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&input)) {
        output[idx] = input[idx] * 2.0;
    }
}
"

def main : IO Unit := do
  IO.println "╔══════════════════════════════════════╗"
  IO.println "║   GPU Double Test (x2)               ║"
  IO.println "╚══════════════════════════════════════╝\n"

  IO.println "🚀 Initializing WebGPU..."
  let inst ← Hesper.init
  let device ← getDevice inst

  let size := 4  -- 4 floats
  IO.println s!"📝 Testing with {size} elements...\n"

  -- Create input data: [1.0, 2.0, 3.0, 4.0]
  let inputData := ByteArray.mk #[
    0x00, 0x00, 0x80, 0x3F,  -- 1.0 in float32 little-endian
    0x00, 0x00, 0x00, 0x40,  -- 2.0 in float32 little-endian
    0x00, 0x00, 0x40, 0x40,  -- 3.0 in float32 little-endian
    0x00, 0x00, 0x80, 0x40   -- 4.0 in float32 little-endian
  ]

  -- Create input buffer
  let inputBuf ← createBuffer device {
    size := (size * 4).toUSize
    usage := [.storage, .copyDst]
    mappedAtCreation := false
  }
  IO.println "  ✓ Created input buffer"

  -- Write input data
  writeBuffer device inputBuf 0 inputData
  IO.println "  ✓ Wrote input: [1.0, 2.0, 3.0, 4.0]"

  -- Create output buffer
  let outputBuf ← createBuffer device {
    size := (size * 4).toUSize
    usage := [.storage, .copySrc]
    mappedAtCreation := false
  }
  IO.println "  ✓ Created output buffer"

  -- Create shader
  IO.println "  📝 Creating compute shader..."
  let shaderModule ← createShaderModule device doubleShader
  IO.println "  ✓ Shader module created"

  -- Create bind group layout for 2 bindings (input + output)
  let layoutEntries := #[
    { binding := 0, visibility := .compute, bindingType := .buffer false : BindGroupLayoutEntry },   -- read-write input
    { binding := 1, visibility := .compute, bindingType := .buffer false : BindGroupLayoutEntry }   -- read-write output
  ]
  let bindGroupLayout ← createBindGroupLayout device layoutEntries

  -- Create pipeline
  let pipeline ← createComputePipeline device {
    shaderModule := shaderModule
    entryPoint := "main"
    bindGroupLayout := bindGroupLayout
  }

  -- Create bind group with both buffers
  let bindEntries := #[
    { binding := 0, buffer := inputBuf, offset := 0, size := (size * 4).toUSize : BindGroupEntry },
    { binding := 1, buffer := outputBuf, offset := 0, size := (size * 4).toUSize : BindGroupEntry }
  ]
  let bindGroup ← createBindGroup device bindGroupLayout bindEntries

  -- Dispatch - 4 workgroups, each processes one element
  dispatchCompute device pipeline bindGroup size.toUInt32 1 1
  IO.println s!"  ✓ Dispatched {size} workgroups"

  -- Read back
  deviceWait device
  IO.println "  ⏳ Reading results from GPU..."
  let resultBytes ← mapBufferRead device outputBuf 0 ((size * 4).toUSize)
  IO.println s!"  ✓ Read back {resultBytes.size} bytes"
  unmapBuffer outputBuf

  -- Convert to floats
  let resultFloats ← Hesper.Basic.bytesToFloatArray resultBytes
  IO.println s!"  ✓ Converted to {resultFloats.size} floats"

  -- Display results
  IO.println "\n📊 Results:"
  IO.println "  Input → Expected → Actual"
  let expected := #[2.0, 4.0, 6.0, 8.0]
  let input := #[1.0, 2.0, 3.0, 4.0]
  let mut allCorrect := true

  for i in [0:size] do
    let inp := input[i]!
    let exp := expected[i]!
    let actual := resultFloats[i]!
    let status := if (actual - exp).abs < 0.001 then "✓" else "✗"
    IO.println s!"  [{i}] {inp} → {exp} → {actual} {status}"
    if (actual - exp).abs > 0.001 then
      allCorrect := false

  IO.println ""
  if allCorrect then
    IO.println "✅ SUCCESS: GPU can read and double values correctly!"
  else
    IO.println "❌ FAIL: GPU doubling not working correctly"

end Examples.Tests.SimpleWrite

def main : IO Unit := Examples.Tests.SimpleWrite.main
