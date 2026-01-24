import Hesper
import Hesper.Compute
import Hesper.WGSL.Execute

open Hesper.WebGPU
open Hesper.Compute
open Hesper.WGSL
open Hesper.WGSL.Execute

/-- DSL-generated shader for vector addition -/
def vectorAddDSL (size : Nat) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← Hesper.WGSL.Monad.ShaderM.globalId
  let idx := Exp.vec3X gid
  let _a ← Hesper.WGSL.Monad.ShaderM.declareInputBuffer "a" (.array (.scalar .f32) size)
  let _b ← Hesper.WGSL.Monad.ShaderM.declareInputBuffer "b" (.array (.scalar .f32) size)
  let _c ← Hesper.WGSL.Monad.ShaderM.declareOutputBuffer "c" (.array (.scalar .f32) size)

  let valA ← Hesper.WGSL.Monad.ShaderM.readBuffer (ty := .scalar .f32) (n := size) "a" idx
  let valB ← Hesper.WGSL.Monad.ShaderM.readBuffer (ty := .scalar .f32) (n := size) "b" idx
  let result := Exp.add valA valB
  Hesper.WGSL.Monad.ShaderM.writeBuffer (ty := .scalar .f32) "c" idx result

def main : IO Unit := do
  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║   Verilean Hesper - GPU Vector Addition      ║"
  IO.println "╚══════════════════════════════════════════════╝"
  IO.println ""

  -- Initialize WebGPU
  let inst ← Hesper.init
  let device ← getDevice inst

  let size := 1024
  IO.println s!"🚀 Running GPU vector addition with {size} elements..."

  -- Create input data (all 1.0s for A, all 2.0s for B)
  let aData ← Hesper.Basic.floatArrayToBytes (Array.range size |>.map fun _ => 1.0)
  let bData ← Hesper.Basic.floatArrayToBytes (Array.range size |>.map fun _ => 2.0)

  -- Create buffers
  let aBuf ← createBuffer device {
    size := (size * 4).toUSize
    usage := [.storage, .copyDst]
    mappedAtCreation := false
  }
  let bBuf ← createBuffer device {
    size := (size * 4).toUSize
    usage := [.storage, .copyDst]
    mappedAtCreation := false
  }
  let cBuf ← createBuffer device {
    size := (size * 4).toUSize
    usage := [.storage, .copySrc]
    mappedAtCreation := false
  }

  -- Write data
  writeBuffer device aBuf 0 aData
  writeBuffer device bBuf 0 bData

  -- Execute shader
  IO.println "  ✓ Dispatching compute shader..."
  let config := ExecutionConfig.dispatch1D size 64
  let namedBuffers := [("a", aBuf), ("b", bBuf), ("c", cBuf)]
  executeShaderNamed device (vectorAddDSL size) namedBuffers config

  -- Read back and verify
  IO.println "  ✓ Reading back results..."
  let resultBytes ← mapBufferRead device cBuf 0 ((size * 4).toUSize)
  unmapBuffer cBuf
  let results ← Hesper.Basic.bytesToFloatArray resultBytes

  -- Verify first few elements
  let mut allCorrect := true
  for i in [0:8] do
    let val := results[i]!
    if (val - 3.0).abs > 0.001 then
      IO.println s!"  ✗ Error at index {i}: expected 3.0, got {val}"
      allCorrect := false
    else
      IO.println s!"  [{i}] 1.0 + 2.0 = {val} ✓"

  if allCorrect then
    IO.println "\n✅ SUCCESS: Vector addition completed correctly on GPU!"
  else
    IO.println "\n❌ FAIL: Results were incorrect."

  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║   GPU Test Complete!                         ║"
  IO.println "╚══════════════════════════════════════════════╝"
