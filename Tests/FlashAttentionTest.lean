import Hesper
import Hesper.WGSL.FlashAttention
import Hesper.Training.SafeBuffer

open Hesper.WGSL.FlashAttention
open Hesper.WebGPU
open Hesper.Training.SafeBuffer
open Hesper.Training.VerifiedBackward

def main : IO Unit := do
  IO.println "=== Flash Attention Tests ==="
  IO.println ""

  -- Test 1: CPU equivalence
  let cpuOk := verifyFlashEquivalence
  IO.println s!"1. CPU equivalence (flash spec == standard): {if cpuOk then "PASS" else "FAIL"}"

  -- Test 2: GPU kernel vs CPU spec
  IO.println "2. GPU kernel vs CPU spec:"
  let inst ← Hesper.init
  let device ← Hesper.WebGPU.getDevice inst

  -- Small test: 2 heads, 4 headDim, 3 cached positions
  let numHeads := 2
  let numKVHeads := 2
  let cacheLen := 3
  let headDim := 4
  let scale := 1.0 / (headDim.toFloat.sqrt)

  let mkBuf := fun (n : Nat) =>
    createBuffer device { size := (n * 4).toUSize, usage := [.storage, .copySrc, .copyDst, .mapRead], mappedAtCreation := false }

  -- Q: [numHeads * headDim] = [8]
  let qData := #[1.0, 0.5, -0.3, 0.8,   -- head 0
                  -0.2, 0.7, 0.4, -0.6]  -- head 1
  -- K cache: [numKVHeads * cacheLen * headDim] = [24]
  let kData := #[0.5, 1.0, 0.2, -0.5,   -- kv 0, pos 0
                  -0.3, 0.8, 1.0, 0.1,   -- kv 0, pos 1
                  0.7, -0.2, 0.5, 0.9,   -- kv 0, pos 2
                  0.3, 0.6, -0.4, 0.2,   -- kv 1, pos 0
                  0.8, -0.1, 0.7, -0.3,  -- kv 1, pos 1
                  -0.5, 0.4, 0.3, 0.6]   -- kv 1, pos 2
  -- V cache: same layout as K
  let vData := #[1.0, 0.0, 0.5, -0.3,
                  0.2, 1.0, -0.5, 0.8,
                  -0.1, 0.5, 1.0, 0.2,
                  0.4, -0.2, 0.8, 0.1,
                  -0.3, 0.6, 0.2, 0.9,
                  0.7, 0.1, -0.4, 0.5]

  let qBuf ← mkBuf (numHeads * headDim)
  let kBuf ← mkBuf (numKVHeads * cacheLen * headDim)
  let vBuf ← mkBuf (numKVHeads * cacheLen * headDim)
  let outBuf ← mkBuf (numHeads * headDim)

  writeBuffer device qBuf 0 (floatArrayToBytes qData)
  writeBuffer device kBuf 0 (floatArrayToBytes kData)
  writeBuffer device vBuf 0 (floatArrayToBytes vData)

  -- Run GPU flash attention
  executeFlashAttention device qBuf kBuf vBuf outBuf numHeads numKVHeads cacheLen headDim scale
  let gpuResult ← safeMapBufferReadF32 device outBuf (numHeads * headDim)

  -- CPU standard attention for comparison
  let q0 := #[1.0, 0.5, -0.3, 0.8]
  let q1 := #[-0.2, 0.7, 0.4, -0.6]
  let kCache0 := #[#[0.5, 1.0, 0.2, -0.5], #[-0.3, 0.8, 1.0, 0.1], #[0.7, -0.2, 0.5, 0.9]]
  let kCache1 := #[#[0.3, 0.6, -0.4, 0.2], #[0.8, -0.1, 0.7, -0.3], #[-0.5, 0.4, 0.3, 0.6]]
  let vCache0 := #[#[1.0, 0.0, 0.5, -0.3], #[0.2, 1.0, -0.5, 0.8], #[-0.1, 0.5, 1.0, 0.2]]
  let vCache1 := #[#[0.4, -0.2, 0.8, 0.1], #[-0.3, 0.6, 0.2, 0.9], #[0.7, 0.1, -0.4, 0.5]]

  let cpuOut0 := attentionForward q0 kCache0 vCache0 scale
  let cpuOut1 := attentionForward q1 kCache1 vCache1 scale
  let cpuResult := cpuOut0 ++ cpuOut1

  -- Compare
  let mut maxErr := 0.0
  let mut gpuOk := true
  for i in [:gpuResult.size] do
    let g := gpuResult.getD i 0.0
    let c := cpuResult.getD i 0.0
    if isNaN g then
      IO.println s!"   GPU[{i}] = NaN, CPU = {c}"
      gpuOk := false
    else
      let diff := if g - c < 0.0 then c - g else g - c
      let denom := (if g < 0.0 then -g else g) + (if c < 0.0 then -c else c)
      let err := if denom < 1e-10 then diff else diff / denom
      if err > maxErr then maxErr := err

  IO.println s!"   GPU result: {gpuResult.toList}"
  IO.println s!"   CPU result: {cpuResult.toList}"
  IO.println s!"   Max relative error: {maxErr}"
  if gpuOk && maxErr < 0.01 then
    IO.println "   ✓ GPU flash attention matches CPU spec"
  else
    IO.println "   ✗ GPU flash attention MISMATCH"

  IO.println ""
  if cpuOk && gpuOk && maxErr < 0.01 then
    IO.println "✓ All flash attention tests PASS"
  else
    IO.println "✗ Some tests FAILED"
