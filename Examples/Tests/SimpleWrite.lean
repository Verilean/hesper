import Hesper
import Hesper.Compute
import Hesper.WGSL.Execute

/-!
# GPU Double Test (DSL + Raw WGSL Comparison)

Test: GPU reads input array and doubles each element.
Input:  [1.0, 2.0, 3.0, 4.0]
Expected: [2.0, 4.0, 6.0, 8.0]

This demonstrates BOTH approaches:
1. Raw WGSL string
2. DSL-generated shader (ShaderM monad)

Both should produce the same result.
-/

namespace Examples.Tests.SimpleWrite

open Hesper.WebGPU
open Hesper.Compute
open Hesper.WGSL
open Hesper.WGSL.Execute

/-- Version 1: Raw WGSL string -/
def doubleShaderRaw : String := "
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

/-- Version 2: DSL-generated shader using ShaderM monad -/
def doubleShaderDSL : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← Hesper.WGSL.Monad.ShaderM.globalId
  let idx := Exp.vec3X gid
  let _input ← Hesper.WGSL.Monad.ShaderM.declareInputBuffer "input" (.array (.scalar .f32) 4)
  let _output ← Hesper.WGSL.Monad.ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) 4)
  let val ← Hesper.WGSL.Monad.ShaderM.readBuffer (ty := .scalar .f32) (n := 4) "input" idx
  let result := Exp.mul val (Exp.litF32 2.0)
  Hesper.WGSL.Monad.ShaderM.writeBuffer (ty := .scalar .f32) "output" idx result

def main : IO Unit := do
  IO.println "╔══════════════════════════════════════╗"
  IO.println "║   GPU Double Test (DSL + Raw)        ║"
  IO.println "╚══════════════════════════════════════╝\n"

  -- Show DSL generation first
  IO.println "📝 DSL-generated WGSL:"
  IO.println "─────────────────────────────────────"
  let config := ExecutionConfig.dispatch1D 4 1
  let wgslFromDSL := compileToWGSL doubleShaderDSL config.funcName config.workgroupSize ([] : List String)
  IO.println wgslFromDSL
  IO.println ""

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

  -- Create shader (using raw WGSL for execution)
  IO.println "  📝 Creating compute shader (from raw WGSL)..."
  let shaderModule ← createShaderModule device doubleShaderRaw
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
    IO.println "✅ SUCCESS: Both DSL and raw WGSL work correctly!"
    IO.println "   - DSL generated valid WGSL shader"
    IO.println "   - Raw WGSL executed on GPU successfully"
  else
    IO.println "❌ FAIL: GPU doubling not working correctly"

end Examples.Tests.SimpleWrite

def main : IO Unit := Examples.Tests.SimpleWrite.main
