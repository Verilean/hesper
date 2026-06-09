import LSpec
import Hesper
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WebGPU.Shader
import Hesper.WebGPU.Pipeline
import Hesper.WebGPU.Types
import Hesper.Compute
import Hesper.WGSL.DSL

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

/-! ### Ch04 tutorial parity tests

The tutorial chapter `docs/tutorial/md/Ch04_HighLevelApi.md` ships a
`scaleByThousand` example that walks a user from "I have an Array
Float" to "I have the same array scaled by 1000.0 on the GPU." These
tests pin the exact code path that example uses so we don't break
the tutorial again — they were added after two regressions:

1. `generateUnaryShader` declared both an input AND an output buffer
   under the same WGSL identifier `data`, producing invalid WGSL
   ("'data' previously declared here") that Dawn rejected at
   CreateShaderModule.
2. `floatToWGSL 1000.0` returned the literal "1.0e2" (=100), not
   "1.0e3" (=1000), because `log abs / log 10` rounds *below* by
   a few ulps for powers of ten, leaving the mantissa at exactly
   10.0 instead of 1.0 with the exponent bumped.

If either regresses again, *both* the unit WGSL test and the full
GPU dispatch test below will fail. The WGSL test runs on CPU only
(no GPU required), so it traps a `floatToWGSL` / `generateUnaryShader`
regression even on machines where the GPU path is skipped.
-/

-- The exact DSL function the tutorial walks the user through.
def scaleByThousand : Hesper.WGSL.Exp (.scalar .f32) → Hesper.WGSL.Exp (.scalar .f32) :=
  fun x => x * Hesper.WGSL.Exp.litF32 1000.0

-- Test (CPU-only): WGSL codegen for the tutorial kernel.  Pins the
-- single-binding layout that `parallelFor` expects and the literal
-- `1.0e3` for the scaling constant.  No GPU required.
def testCh04WGSLCodegen : IO TestSeq := do
  let wgsl := Hesper.Compute.generateUnaryShader scaleByThousand
  -- A correct shader has exactly one storage binding named `data`
  -- and folds the literal 1000.0 as "1.0e3".  Any regression of the
  -- two bugs above flips one of these substring checks.
  -- `(s.splitOn sub).length - 1` counts non-overlapping occurrences of `sub` in `s`.
  let nDataBindings := (wgsl.splitOn "data: array<f32>").length - 1
  let hasLit1e3 := (wgsl.splitOn "1.0e3").length > 1
  let hasLit1e2 := (wgsl.splitOn "1.0e2").length > 1
  pure $
    test "Ch04 generateUnaryShader: exactly one `data` storage binding" (nDataBindings == 1)
    ++ test "Ch04 generateUnaryShader: literal 1.0e3 present" hasLit1e3
    ++ test "Ch04 generateUnaryShader: literal 1.0e2 absent (regression guard for floatToWGSL bug)" (! hasLit1e2)

-- Test (GPU): end-to-end `parallelForDSL` on the host's default GPU.
-- This is the cell the tutorial runs in xeus-lean and is what the
-- user expects to see produce `[0, 1000, …, 9000]`.
def testCh04ParallelForDSL : IO TestSeq := withDevice fun _ device => do
  let input : Array Float := (Array.range 10).map (·.toFloat)
  let out ← Hesper.Compute.parallelForDSL device scaleByThousand input
  let expected : Array Float := input.map (· * 1000.0)
  let sizesMatch := out.size == expected.size
  let valsMatch := sizesMatch && (Array.zip out expected).all (fun (a, b) =>
    -- Allow a tiny absolute tolerance — f32 round-trip on the GPU
    -- gives bit-exact results for these inputs, but be defensive.
    let d := a - b
    let abs_d := if d < 0.0 then 0.0 - d else d
    abs_d < 0.001)
  pure $ test s!"Ch04 parallelForDSL scaleByThousand → [0, 1000, …, 9000] (got {out})" valsMatch

-- All compute tests
def allTests : IO (List (String × List TestSeq)) := do
  IO.println "Running Compute Pipeline Tests..."

  let t1 ← testShaderCreation
  let t2 ← testBindGroupLayoutCreation
  let t3 ← testComputePipelineCreation
  let t4 ← testFullComputePipeline
  let t5 ← testMultipleShaders
  let t6 ← testInvalidShader
  let t7 ← testCh04WGSLCodegen
  let t8 ← testCh04ParallelForDSL

  pure [
    ("Shader Module Creation", [t1]),
    ("Bind Group Layout Creation", [t2]),
    ("Compute Pipeline Creation", [t3]),
    ("Full Pipeline Setup", [t4]),
    ("Multiple Shaders", [t5]),
    ("Error: Invalid Shader", [t6]),
    ("Ch04 tutorial WGSL codegen (CPU)", [t7]),
    ("Ch04 tutorial parallelForDSL (GPU)", [t8])
  ]

end Tests.ComputeTests

