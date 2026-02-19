import LSpec
import Hesper
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WGSL.Execute
import Hesper.Layers.BitLinear
import Hesper.Basic

/-!
# Subgroup Fallback Tests

Tests that the shared-memory fallback kernels produce identical results
to the subgroup-based kernels. On devices without subgroup support,
only the shared-memory path is tested.

## Test Strategy:
1. Feature detection: verify `hasSubgroupSupport` caching works
2. Shared-memory kernel: always run, verify correctness
3. Both paths (if subgroups available): compare outputs
-/

namespace Tests.SubgroupFallbackTests

open Hesper.WebGPU
open Hesper.WGSL.Execute
open Hesper.Layers.BitLinear
open LSpec

/-- Create dummy i2_s packed weights for testing.
    All weights set to +1 (code=0b10), so output = scale * sum(input).
    For inDim elements: numBytes = inDim / 4, each byte = 0xAA (10_10_10_10). -/
def makeTestWeights (inDim outDim : Nat) : ByteArray × ByteArray :=
  let numBytes := outDim * inDim / 4
  let packed := ByteArray.mk (Array.replicate numBytes 0xAA)
  let scaleBytes := ByteArray.mk #[0x00, 0x00, 0x80, 0x3F]
  (packed, scaleBytes)

/-- Create test input: values 1.0 for all elements -/
def makeTestInput (dim : Nat) : IO ByteArray := do
  let mut bytes := ByteArray.empty
  for _ in [0:dim] do
    let fb ← Hesper.Basic.floatToBytes 1.0
    bytes := bytes ++ fb
  pure bytes

-- All subgroup fallback tests (share a single device to avoid pipeline cache conflicts)
def allTests : IO (List (String × List TestSeq)) := do
  IO.println "Running Subgroup Fallback Tests..."

  let inst ← Hesper.init
  let device ← getDevice inst

  -- Reset pipeline/bind group caches to avoid cross-test conflicts
  resetPipelineCache

  -- Test 1: Feature detection caching
  let t1 ← do
    let v1 ← hasSubgroupSupport device
    let v2 ← hasSubgroupSupport device
    IO.println s!"  Subgroup support: {v1}"
    pure $ test "Feature detection returns consistent results" (v1 == v2)

  -- Test 2: Shared-memory M=1 kernel produces correct results
  let t2 ← do
    let inDim := 128
    let outDim := 4
    let config : Config := { inDim, outDim, batchSize := 1 }
    let (packed, scaleBytes) := makeTestWeights inDim outDim

    let inputBytes ← makeTestInput inDim
    let inputBuf ← createBuffer device {
      size := (inDim * 4).toUSize
      usage := [.storage, .copyDst, .copySrc]
      mappedAtCreation := false
    }
    writeBuffer device inputBuf 0 inputBytes

    let outputBuf ← createBuffer device {
      size := (outDim * 4).toUSize
      usage := [.storage, .copyDst, .copySrc]
      mappedAtCreation := false
    }

    let weightsBuf ← createBuffer device {
      size := (packed.size.max 4).toUSize
      usage := [.storage, .copyDst]
      mappedAtCreation := false
    }
    writeBuffer device weightsBuf 0 packed

    let scaleBuf ← createBuffer device {
      size := 4
      usage := [.storage, .copyDst]
      mappedAtCreation := false
    }
    writeBuffer device scaleBuf 0 scaleBytes

    let shaderM := fusedBitLinearM1KernelSharedMem config
    let execConfig : ExecutionConfig := {
      workgroupSize := { x := 32, y := 1, z := 1 }
      numWorkgroups := (outDim, 1, 1)
    }
    let namedBuffers := [
      ("weights_packed", weightsBuf),
      ("scale", scaleBuf),
      ("input", inputBuf),
      ("output", outputBuf)
    ]
    executeShaderNamed device shaderM namedBuffers execConfig

    let resultBytes ← mapBufferRead device outputBuf 0 (outDim * 4).toUSize
    let results ← Hesper.Basic.bytesToFloatArray resultBytes

    let mut allCorrect := true
    for i in [0:outDim] do
      if h : i < results.size then
        let val := results[i]
        let diff := (val - 128.0).abs
        if diff > 0.1 then
          IO.println s!"  Output[{i}] = {val}, expected 128.0, diff = {diff}"
          allCorrect := false

    IO.println s!"  SharedMem results: {results.toList}"
    pure $ test "Shared-mem kernel: all +1 weights × all 1.0 inputs = 128.0" allCorrect

  -- Test 3: Both kernels produce same results (only when subgroups available)
  let t3 ← do
    let hasSubgroups ← hasSubgroupSupport device
    if !hasSubgroups then
      IO.println s!"  Skipping comparison test (no subgroup support on this device)"
      pure $ test "Both kernels match (skipped - no subgroups)" true
    else
      let inDim := 128
      let outDim := 4
      let config : Config := { inDim, outDim, batchSize := 1 }
      let (packed, scaleBytes) := makeTestWeights inDim outDim

      let inputBytes ← makeTestInput inDim

      let inputBuf ← createBuffer device {
        size := (inDim * 4).toUSize
        usage := [.storage, .copyDst, .copySrc]
        mappedAtCreation := false
      }
      writeBuffer device inputBuf 0 inputBytes

      let weightsBuf ← createBuffer device {
        size := (packed.size.max 4).toUSize
        usage := [.storage, .copyDst]
        mappedAtCreation := false
      }
      writeBuffer device weightsBuf 0 packed

      let scaleBuf ← createBuffer device {
        size := 4
        usage := [.storage, .copyDst]
        mappedAtCreation := false
      }
      writeBuffer device scaleBuf 0 scaleBytes

      let outputBuf1 ← createBuffer device {
        size := (outDim * 4).toUSize
        usage := [.storage, .copyDst, .copySrc]
        mappedAtCreation := false
      }
      let outputBuf2 ← createBuffer device {
        size := (outDim * 4).toUSize
        usage := [.storage, .copyDst, .copySrc]
        mappedAtCreation := false
      }

      -- Run subgroup kernel
      let subgroupConfig : ExecutionConfig := {
        workgroupSize := { x := 32, y := 1, z := 1 }
        numWorkgroups := (outDim, 1, 1)
        extensions := ["subgroups"]
        diagnostics := [("off", "chromium.subgroup_matrix_uniformity")]
      }
      let namedBufs1 := [("weights_packed", weightsBuf), ("scale", scaleBuf), ("input", inputBuf), ("output", outputBuf1)]
      executeShaderNamed device (fusedBitLinearM1Kernel config) namedBufs1 subgroupConfig

      -- Run shared-mem kernel
      let sharedMemConfig : ExecutionConfig := {
        workgroupSize := { x := 32, y := 1, z := 1 }
        numWorkgroups := (outDim, 1, 1)
      }
      let namedBufs2 := [("weights_packed", weightsBuf), ("scale", scaleBuf), ("input", inputBuf), ("output", outputBuf2)]
      executeShaderNamed device (fusedBitLinearM1KernelSharedMem config) namedBufs2 sharedMemConfig

      let result1Bytes ← mapBufferRead device outputBuf1 0 (outDim * 4).toUSize
      let result2Bytes ← mapBufferRead device outputBuf2 0 (outDim * 4).toUSize
      let results1 ← Hesper.Basic.bytesToFloatArray result1Bytes
      let results2 ← Hesper.Basic.bytesToFloatArray result2Bytes

      let mut allMatch := true
      for i in [0:outDim] do
        if h1 : i < results1.size then
          if h2 : i < results2.size then
            let diff := (results1[i] - results2[i]).abs
            if diff > 1e-5 then
              IO.println s!"  Mismatch at [{i}]: subgroup={results1[i]}, shared={results2[i]}, diff={diff}"
              allMatch := false

      IO.println s!"  Subgroup results: {results1.toList}"
      IO.println s!"  SharedMem results: {results2.toList}"
      pure $ test "Subgroup and shared-mem kernels produce same results" allMatch

  -- Test 4: BitLinear.create auto-selects correct kernel
  let t4 ← do
    let inDim := 128
    let outDim := 4
    let config : Config := { inDim, outDim, batchSize := 1 }
    let (packed, scaleBytes) := makeTestWeights inDim outDim

    let layer ← createFromBytes device config packed scaleBytes

    let inputBytes ← makeTestInput inDim
    let inputBuf ← createBuffer device {
      size := (inDim * 4).toUSize
      usage := [.storage, .copyDst, .copySrc]
      mappedAtCreation := false
    }
    writeBuffer device inputBuf 0 inputBytes

    let outputBuf ← createBuffer device {
      size := (outDim * 4).toUSize
      usage := [.storage, .copyDst, .copySrc]
      mappedAtCreation := false
    }

    forward device layer inputBuf outputBuf

    let resultBytes ← mapBufferRead device outputBuf 0 (outDim * 4).toUSize
    let results ← Hesper.Basic.bytesToFloatArray resultBytes

    let mut allCorrect := true
    for i in [0:outDim] do
      if h : i < results.size then
        let diff := (results[i] - 128.0).abs
        if diff > 0.1 then
          IO.println s!"  Output[{i}] = {results[i]}, expected 128.0"
          allCorrect := false

    let hasSubgroups ← hasSubgroupSupport device
    IO.println s!"  Auto-selected kernel (subgroups={hasSubgroups}), results: {results.toList}"
    pure $ test "BitLinear.create auto-selects correct kernel" allCorrect

  pure [
    ("Feature Detection Caching", [t1]),
    ("Shared-Mem Kernel Correctness", [t2]),
    ("Both Kernels Same Result", [t3]),
    ("Auto Kernel Selection", [t4])
  ]

end Tests.SubgroupFallbackTests
