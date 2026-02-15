import Hesper
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WGSL.Monad
import Hesper.WGSL.CodeGen
import Hesper.WGSL.Execute

/-!
# Minimal BindGroup Test

Ultra-minimal test to isolate GPU validation error.
Creates 3 small buffers, binds them, and dispatches a trivial compute shader.

This bypasses all model loading to quickly reproduce the BindGroup validation error.
-/

namespace Tests.MinimalBindGroup

open Hesper.WebGPU
open Hesper.WGSL
open Hesper.WGSL.Monad.ShaderM
open Hesper.WGSL.CodeGen
open Hesper.WGSL.Execute

/-- Trivial kernel that just reads from buffer and writes to output -/
def trivialKernel : Hesper.WGSL.Monad.ShaderM Unit := do
  let inputA ← declareInputBuffer "input_a" (.array (.scalar .u32) 1)
  let inputB ← declareInputBuffer "input_b" (.array (.scalar .f32) 32)
  let output ← declareOutputBuffer "output" (.array (.scalar .f32) 32)

  let gid ← globalId
  let idx := Exp.vec3X gid

  -- Just copy from input_b to output (ignore input_a)
  let val ← readBuffer (ty := .scalar .f32) (n := 32) inputB idx
  writeBuffer (ty := .scalar .f32) output idx val

def runMinimalTest : IO Unit := do
  IO.println "════════════════════════════════════════════════"
  IO.println "  Minimal BindGroup Test"
  IO.println "════════════════════════════════════════════════"

  -- Initialize WebGPU
  let inst ← Hesper.init
  let device ← getDevice inst
  IO.println "✓ Device created"

  -- Create 3 small buffers (same pattern as embedding: token_ids, table, output)
  -- Buffer 1: 4 bytes (1 u32)
  let buf1 ← createBuffer device {
    size := 4
    usage := [.storage, .copyDst]
    mappedAtCreation := false
  }
  IO.println "✓ Buffer 1 created (4 bytes)"

  -- Buffer 2: 128 bytes (32 floats)
  let buf2 ← createBuffer device {
    size := 128
    usage := [.storage, .copyDst]
    mappedAtCreation := false
  }
  IO.println "✓ Buffer 2 created (128 bytes)"

  -- Buffer 3: 128 bytes (32 floats)
  let buf3 ← createBuffer device {
    size := 128
    usage := [.storage, .copySrc]
    mappedAtCreation := false
  }
  IO.println "✓ Buffer 3 created (128 bytes)"

  -- Write some data to buffers
  let mut data1 := ByteArray.empty
  data1 := data1.push 42
  data1 := data1.push 0
  data1 := data1.push 0
  data1 := data1.push 0
  writeBuffer device buf1 0 data1
  IO.println "✓ Buffer 1 written"

  let mut data2 := ByteArray.empty
  for i in [0:32] do
    let val : Float := i.toFloat * 1.5
    let bits := val.toBits
    data2 := data2.push (bits.toUInt8)
    data2 := data2.push ((bits >>> 8).toUInt8)
    data2 := data2.push ((bits >>> 16).toUInt8)
    data2 := data2.push ((bits >>> 24).toUInt8)
  writeBuffer device buf2 0 data2
  IO.println "✓ Buffer 2 written"

  -- Execute shader
  IO.println "\n[TEST] Executing shader (will create BindGroup and dispatch)..."

  try
    let namedBuffers := [
      ("input_a", buf1),
      ("input_b", buf2),
      ("output", buf3)
    ]

    let execConfig := ExecutionConfig.dispatch1D 32 32

    executeShaderNamed device trivialKernel namedBuffers execConfig

    IO.println "✓ Shader executed successfully!"
    IO.println "\n════════════════════════════════════════════════"
    IO.println "✓ TEST PASSED - No BindGroup error!"
    IO.println "════════════════════════════════════════════════"

  catch e =>
    IO.println "\n════════════════════════════════════════════════"
    IO.println "✗ TEST FAILED"
    IO.println s!"Error: {e}"
    IO.println "════════════════════════════════════════════════"
    throw e

end Tests.MinimalBindGroup

def main : IO Unit := Tests.MinimalBindGroup.runMinimalTest
