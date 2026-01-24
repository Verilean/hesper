import Hesper
import Hesper.Compute
import Hesper.WGSL.Execute

open Hesper.WebGPU
open Hesper.Compute
open Hesper.WGSL
open Hesper.WGSL.Execute

namespace Examples.Compute.MatmulSimple

/-!
# Simple Matrix Multiplication using ShaderM

This demonstrates matrix multiplication using the ShaderM monad with a naive algorithm.
Each thread computes one output element C[i,j] = sum_k A[i,k] * B[k,j].

**Dimensions**: A (M×K) × B (K×N) = C (M×N)
-/

/-- Naive matrix multiplication shader using ShaderM monad.

    Each thread computes one output element.
    No tiling or shared memory optimization.
-/
def naiveMatmulShader (M K N : Nat) : Hesper.WGSL.Monad.ShaderM Unit := do
  -- Get thread indices
  let gid ← Hesper.WGSL.Monad.ShaderM.globalId
  let row := Exp.vec3X gid
  let col := Exp.vec3Y gid

  -- Declare buffers (they will be auto-bound)
  let _A ← Hesper.WGSL.Monad.ShaderM.declareInputBuffer "A" (.array (.scalar .f32) (M * K))
  let _B ← Hesper.WGSL.Monad.ShaderM.declareInputBuffer "B" (.array (.scalar .f32) (K * N))
  let _C ← Hesper.WGSL.Monad.ShaderM.declareOutputBuffer "C" (.array (.scalar .f32) (M * N))

  -- Bounds check
  let inBounds := Exp.and
    (Exp.lt row ((M : Nat) : Exp (.scalar .u32)))
    (Exp.lt col ((N : Nat) : Exp (.scalar .u32)))

  Hesper.WGSL.Monad.ShaderM.if_ inBounds (do
    -- Initialize accumulator
    Hesper.WGSL.Monad.ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)

    -- Loop over K dimension: for (var k: u32 = 0u; k < K; k = k + 1u)
    Hesper.WGSL.Monad.ShaderM.loop
      (Exp.litU32 0)
      ((K : Nat) : Exp (.scalar .u32))
      (Exp.litU32 1)
      fun k => do
        -- Compute indices
        let aIdx := row * ((K : Nat) : Exp (.scalar .u32)) + k
        let bIdx := k * ((N : Nat) : Exp (.scalar .u32)) + col

        -- Load A[row,k] and B[k,col]
        let aVal ← Hesper.WGSL.Monad.ShaderM.readBuffer (ty := .scalar .f32) (n := M * K) "A" aIdx
        let bVal ← Hesper.WGSL.Monad.ShaderM.readBuffer (ty := .scalar .f32) (n := K * N) "B" bIdx

        -- Accumulate: acc += a * b
        let accVar : Exp (.scalar .f32) := Exp.var "acc"
        let newAcc := Exp.add accVar (Exp.mul aVal bVal)
        Hesper.WGSL.Monad.ShaderM.assign "acc" newAcc

    -- Write result to C[row,col]
    let outIdx := row * ((N : Nat) : Exp (.scalar .u32)) + col
    let accVar : Exp (.scalar .f32) := Exp.var "acc"
    Hesper.WGSL.Monad.ShaderM.writeBuffer (ty := .scalar .f32) "C" outIdx accVar
  ) (do
    -- Empty else branch for out-of-bounds threads
    pure ()
  )

def main : IO Unit := do
  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║   Hesper GPU Matrix Multiplication           ║"
  IO.println "║   (Naive Algorithm with ShaderM)             ║"
  IO.println "╚══════════════════════════════════════════════╝"
  IO.println ""

  -- Matrix dimensions
  let M := 128
  let K := 128
  let N := 128

  IO.println s!"Matrix dimensions: {M}×{K} × {K}×{N} = {M}×{N}"
  IO.println s!"Total operations: {2 * M * N * K}"
  IO.println ""

  -- Initialize WebGPU
  IO.println "Initializing WebGPU..."
  let inst ← Hesper.init
  let device ← getDevice inst

  -- Create buffers
  IO.println "Creating GPU buffers..."
  let aBuf ← createBuffer device {
    size := (M * K * 4).toUSize
    usage := [.storage, .copyDst]
    mappedAtCreation := false
  }
  let bBuf ← createBuffer device {
    size := (K * N * 4).toUSize
    usage := [.storage, .copyDst]
    mappedAtCreation := false
  }
  let cBuf ← createBuffer device {
    size := (M * N * 4).toUSize
    usage := [.storage, .copySrc]
    mappedAtCreation := false
  }

  -- Initialize matrix data: A[i,j] = i+1, B[i,j] = j+1
  IO.println "Uploading test data..."
  let aData ← Hesper.Basic.floatArrayToBytes (Array.range (M * K) |>.map fun idx =>
    let i := idx / K
    (i + 1).toFloat)
  let bData ← Hesper.Basic.floatArrayToBytes (Array.range (K * N) |>.map fun idx =>
    let j := idx % N
    (j + 1).toFloat)
  writeBuffer device aBuf 0 aData
  writeBuffer device bBuf 0 bData

  -- Compile shader using ShaderM
  IO.println "Compiling shader from ShaderM monad..."
  let shaderBody := naiveMatmulShader M K N
  let TILE_SIZE := 16
  let config : ExecutionConfig := {
    funcName := "main"
    workgroupSize := {x := TILE_SIZE, y := TILE_SIZE, z := 1}
    numWorkgroups := ((M + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE, 1)
  }

  -- Debug: Print generated WGSL
  let wgsl := compileToWGSL shaderBody config.funcName config.workgroupSize
  IO.println "\nGenerated WGSL (first 500 chars):"
  IO.println (wgsl.take 500 ++ "...")
  IO.println ""

  -- Execute shader
  IO.println "Running matrix multiplication on GPU..."
  let namedBuffers : List (String × Buffer) := [
    ("A", aBuf),
    ("B", bBuf),
    ("C", cBuf)
  ]
  executeShaderNamed device shaderBody namedBuffers config

  -- Read back and verify results
  IO.println "Reading back results..."
  let resultBytes ← mapBufferRead device cBuf 0 (M * N * 4).toUSize
  unmapBuffer cBuf
  let results ← Hesper.Basic.bytesToFloatArray resultBytes

  -- Expected: C[i,j] = sum_k A[i,k] * B[k,j] = sum_k (i+1) * (j+1) = (i+1) * (j+1) * K
  let testCases := [
    (0, 0, 1.0 * 1.0 * 128.0),   -- C[0,0] = 128
    (0, 1, 1.0 * 2.0 * 128.0),   -- C[0,1] = 256
    (1, 0, 2.0 * 1.0 * 128.0),   -- C[1,0] = 256
    (1, 1, 2.0 * 2.0 * 128.0),   -- C[1,1] = 512
    (10, 10, 11.0 * 11.0 * 128.0) -- C[10,10] = 15488
  ]

  let mut allPassed := true
  for (i, j, expected) in testCases do
    let idx := i * N + j
    let actual := results[idx]!
    let passed := (actual - expected).abs < 0.1
    if passed then
      IO.println s!"✅ C[{i},{j}] = {actual} (expected {expected})"
    else
      IO.println s!"❌ C[{i},{j}] = {actual} (expected {expected})"
      allPassed := false

  if allPassed then
    IO.println "\n✅ All verification tests passed!"
  else
    IO.println "\n❌ Some verification tests failed!"

  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║   Matrix multiplication complete!            ║"
  IO.println "╚══════════════════════════════════════════════╝"

end Examples.Compute.MatmulSimple

-- Top-level main for executable
def main : IO Unit := Examples.Compute.MatmulSimple.main
