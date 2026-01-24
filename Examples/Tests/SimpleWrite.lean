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

  -- Test 1: Execute raw WGSL shader
  IO.println "\n  🔹 Test 1: Raw WGSL shader"
  let shaderModule ← createShaderModule device doubleShaderRaw
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
    { binding := 0, buffer := inputBuf, offset := 0, size := (size * 4).toUSize : BindGroupEntry },
    { binding := 1, buffer := outputBuf, offset := 0, size := (size * 4).toUSize : BindGroupEntry }
  ]
  let bindGroup ← createBindGroup device bindGroupLayout bindEntries
  let future ← dispatchCompute device pipeline bindGroup size.toUInt32 1 1
  deviceWait future
  IO.println "  ✓ Raw WGSL executed"

  -- Read back results from Test 1
  let resultBytes1 ← mapBufferRead device outputBuf 0 ((size * 4).toUSize)
  unmapBuffer outputBuf
  let resultFloats1 ← Hesper.Basic.bytesToFloatArray resultBytes1

  -- Clear output buffer for Test 2
  let zeroData := ByteArray.mk #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  writeBuffer device outputBuf 0 zeroData

  -- Test 2: Execute DSL shader
  IO.println "\n  🔹 Test 2: DSL-generated shader"
  let config := ExecutionConfig.dispatch1D size 1
  let namedBuffers : List (String × Buffer) := [("input", inputBuf), ("output", outputBuf)]
  executeShaderNamed device doubleShaderDSL namedBuffers config
  IO.println "  ✓ DSL shader executed"

  -- Read back results from Test 2
  let resultBytes2 ← mapBufferRead device outputBuf 0 ((size * 4).toUSize)
  unmapBuffer outputBuf
  let resultFloats2 ← Hesper.Basic.bytesToFloatArray resultBytes2

  -- Display results
  IO.println "\n📊 Results:"
  let expected := #[2.0, 4.0, 6.0, 8.0]
  let input := #[1.0, 2.0, 3.0, 4.0]
  let mut allCorrect := true

  IO.println "  Input → Expected → Raw WGSL → DSL WGSL"
  for i in [0:size] do
    let inp := input[i]!
    let exp := expected[i]!
    let actual1 := resultFloats1[i]!
    let actual2 := resultFloats2[i]!
    let status1 := if (actual1 - exp).abs < 0.001 then "✓" else "✗"
    let status2 := if (actual2 - exp).abs < 0.001 then "✓" else "✗"
    IO.println s!"  [{i}] {inp} → {exp} → {actual1} {status1} → {actual2} {status2}"
    if (actual1 - exp).abs > 0.001 || (actual2 - exp).abs > 0.001 then
      allCorrect := false

  IO.println ""
  if allCorrect then
    IO.println "✅ SUCCESS: Both shaders work correctly!"
    IO.println "   - Raw WGSL shader: ✓"
    IO.println "   - DSL-generated shader (ShaderM monad): ✓"
    IO.println "   - Both produce identical correct results"
  else
    IO.println "❌ FAIL: One or both shaders failed"

end Examples.Tests.SimpleWrite

def main : IO Unit := Examples.Tests.SimpleWrite.main
