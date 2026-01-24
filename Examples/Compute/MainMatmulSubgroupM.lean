import Hesper
import Hesper.Compute
import Hesper.WGSL.Execute

open Hesper.WebGPU
open Hesper.Compute
open Hesper.WGSL
open Hesper.WGSL.Execute

namespace Examples.Compute.MatmulSubgroupM

/-!
# Subgroup Matrix Multiplication using ShaderM

This demonstrates high-performance matrix multiplication using Chrome's
chromium_experimental_subgroup_matrix extension through the ShaderM monad.

**Algorithm**: Tiled matrix multiplication with subgroup matrix operations
- Uses hardware-accelerated 8×8 matrix tiles
- Each workgroup computes a (TM*8)×(TN*8) block of output
- Subgroup operations provide significant performance gains

**Dimensions**: A (M×K) × B (K×N) = C (M×N)
-/

/-- Subgroup matrix multiplication shader using ShaderM monad.

    Parameters:
    - M, K, N: Matrix dimensions
    - TM, TN: Number of 8×8 tiles per workgroup
    - LID0, LID1: Workgroup size (threads)
-/
def subgroupMatmulShader
    (M K N : Nat)
    (TM TN : Nat)
    (LID0 LID1 : Nat)
    : Hesper.WGSL.Monad.ShaderM Unit := do
  -- Get workgroup and local invocation IDs
  let wg ← Hesper.WGSL.Monad.ShaderM.workgroupId
  let localID ← Hesper.WGSL.Monad.ShaderM.localId

  let wgX := Exp.vec3X wg
  let wgY := Exp.vec3Y wg
  let localIDY := Exp.vec3Y localID

  -- Declare buffers
  let _A ← Hesper.WGSL.Monad.ShaderM.declareInputBuffer "A" (.array (.scalar .f32) (M * K))
  let _B ← Hesper.WGSL.Monad.ShaderM.declareInputBuffer "B" (.array (.scalar .f32) (K * N))
  let _C ← Hesper.WGSL.Monad.ShaderM.declareOutputBuffer "C" (.array (.scalar .f32) (M * N))

  -- Compute starting positions
  Hesper.WGSL.Monad.ShaderM.varNamed "rowStart" (.scalar .u32)
    (wgX * ((8 * TM) : Nat))
  Hesper.WGSL.Monad.ShaderM.varNamed "colStart" (.scalar .u32)
    ((wgY * (LID1 : Nat) + localIDY) * ((8 * TN) : Nat))

  let rowStartVar : Exp (.scalar .u32) := Exp.var "rowStart"
  let colStartVar : Exp (.scalar .u32) := Exp.var "colStart"

  -- Compute base offsets
  Hesper.WGSL.Monad.ShaderM.varNamed "baseA" (.scalar .u32)
    (rowStartVar * ((K : Nat) : Exp (.scalar .u32)))
  Hesper.WGSL.Monad.ShaderM.varNamed "baseB" (.scalar .u32)
    colStartVar
  Hesper.WGSL.Monad.ShaderM.varNamed "cBase" (.scalar .u32)
    (rowStartVar * ((N : Nat) : Exp (.scalar .u32)) + colStartVar)

  -- Declare subgroup matrix arrays
  Hesper.WGSL.Monad.ShaderM.declareMatrixLeftArray "Ax" .f32 8 8 TM
    Exp.subgroupMatrixZeroLeft
  Hesper.WGSL.Monad.ShaderM.declareMatrixRightArray "Bx" .f32 8 8 TN
    Exp.subgroupMatrixZeroRight
  Hesper.WGSL.Monad.ShaderM.declareMatrixResultArray "accxx" .f32 8 8 (TM * TN)
    Exp.subgroupMatrixZeroResult

  let baseAVar : Exp (.scalar .u32) := Exp.var "baseA"
  let baseBVar : Exp (.scalar .u32) := Exp.var "baseB"
  let cBaseVar : Exp (.scalar .u32) := Exp.var "cBase"

  -- Main computation loop over K dimension
  Hesper.WGSL.Monad.ShaderM.loop
    (Exp.litU32 0)
    ((K : Nat) : Exp (.scalar .u32))
    (Exp.litU32 8)
    fun kk => do
      -- Barrier before loading
      Hesper.WGSL.Monad.ShaderM.barrier

      -- Load tiles of A (left matrices)
      Hesper.WGSL.Monad.ShaderM.staticLoop TM fun i => do
        let offset := baseAVar + kk + ((8 * K * i) : Nat)
        Hesper.WGSL.Monad.ShaderM.loadMatrixLeft (st := .f32) (m := 8) (k := 8)
          "Ax" i "A" offset ((K : Nat) : Exp (.scalar .u32))

      -- Load tiles of B (right matrices)
      Hesper.WGSL.Monad.ShaderM.staticLoop TN fun i => do
        let offset := baseBVar + kk * ((N : Nat) : Exp (.scalar .u32)) + ((8 * i) : Nat)
        Hesper.WGSL.Monad.ShaderM.loadMatrixRight (st := .f32) (k := 8) (n := 8)
          "Bx" i "B" offset ((N : Nat) : Exp (.scalar .u32))

      -- Multiply-accumulate (nested loop over tiles)
      Hesper.WGSL.Monad.ShaderM.staticLoop2D TN TM fun j i => do
        let idx := j * TM + i
        Hesper.WGSL.Monad.ShaderM.matrixMultiplyAccumulate (st := .f32) (m := 8) (k := 8) (n := 8)
          "accxx" idx "Ax" i "Bx" j

  -- Barrier before storing
  Hesper.WGSL.Monad.ShaderM.barrier

  -- Store result matrices
  Hesper.WGSL.Monad.ShaderM.staticLoop2D TM TN fun i j => do
    let idx := j * TM + i
    let offset := cBaseVar + ((i * 8 * N + 8 * j) : Nat)
    Hesper.WGSL.Monad.ShaderM.storeMatrixResult (st := .f32) (m := 8) (n := 8)
      "accxx" idx "C" offset ((N : Nat) : Exp (.scalar .u32))

def main : IO Unit := do
  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║   Hesper GPU Matrix Multiplication           ║"
  IO.println "║   (Subgroup Operations with ShaderM)         ║"
  IO.println "╚══════════════════════════════════════════════╝"
  IO.println ""

  -- Matrix dimensions (matching MainMatmul.lean)
  let M := 128
  let K := 128
  let N := 128

  -- Tile configuration
  let TM := 4   -- 4 tiles of 8 rows = 32 rows per workgroup
  let TN := 2   -- 2 tiles of 8 cols = 16 cols per thread
  let LID0 := 32  -- Local workgroup size X
  let LID1 := 2   -- Local workgroup size Y

  IO.println s!"Matrix dimensions: {M}×{K} × {K}×{N} = {M}×{N}"
  IO.println s!"Tile configuration: TM={TM}, TN={TN}"
  IO.println s!"Workgroup size: {LID0}×{LID1}"
  IO.println s!"Total operations: {2 * M * N * K}"
  IO.println ""

  -- Initialize WebGPU with subgroup features
  IO.println "Initializing WebGPU with subgroup matrix support..."
  let inst ← Hesper.init
  let device ← getDeviceWithFeatures inst

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

  -- Compile shader using ShaderM with subgroup operations
  IO.println "Compiling shader from ShaderM monad (with subgroup matrix ops)..."
  let shaderBody := subgroupMatmulShader M K N TM TN LID0 LID1

  let numWorkgroupsX := (M + 31) / 32   -- ceil(128 / 32) = 4
  let numWorkgroupsY := (N + 63) / 64   -- ceil(128 / 64) = 2

  let config : ExecutionConfig := {
    funcName := "main"
    workgroupSize := {x := LID0, y := LID1, z := 1}
    numWorkgroups := (numWorkgroupsX, numWorkgroupsY, 1)
    extensions := ["chromium_experimental_subgroup_matrix"]
    diagnostics := [("off", "chromium.subgroup_matrix_uniformity")]
  }

  -- Debug: Print generated WGSL
  let wgsl := compileToWGSL shaderBody config.funcName config.workgroupSize config.extensions config.diagnostics
  IO.println "\nGenerated WGSL (first 500 chars):"
  IO.println (wgsl.take 500 ++ "...")
  IO.println ""

  -- Execute shader (need to add chromium_experimental_subgroup_matrix extension)
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

end Examples.Compute.MatmulSubgroupM

-- Top-level main for executable
def main : IO Unit := Examples.Compute.MatmulSubgroupM.main
