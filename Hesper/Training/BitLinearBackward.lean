import Hesper.WGSL.Monad
import Hesper.WGSL.Execute
import Hesper.WGSL.Exp
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.Layers.BitLinear

/-!
# BitLinear Backward (Transpose MatVec)

Computes dInput = scale * W^T @ dOutput for the O projection backward.

The weight matrix is in i2_s ternary format ({-1, 0, +1} packed as 2 bits).
The forward kernel has 1 workgroup per output row.
The transpose kernel has 1 workgroup per INPUT element (summing over output dim).

## i2_s Layout (same as forward)

W is [outDim, inDim]. Each row of 128 elements is packed into 32 bytes:
- Elements [0..31]:   bytes[0..31] >> 6 & 0x3
- Elements [32..63]:  bytes[0..31] >> 4 & 0x3
- Elements [64..95]:  bytes[0..31] >> 2 & 0x3
- Elements [96..127]: bytes[0..31] >> 0 & 0x3

Dequant: value = (code - 1) where code ∈ {0→-1, 1→0, 2→+1}

## Transpose Access Pattern

For dInput[j] = scale * Σ_i W[i,j] * dOutput[i]:
- We need to read column j across all rows i
- This is a strided access in the packed data
-/

namespace Hesper.Training.BitLinearBackward

open Hesper.WGSL
open Hesper.WGSL.Monad
open Hesper.WebGPU

/-- Transpose matmul kernel: dInput[j] = scale * Σ_i W[i,j] * dOutput[i]

    W is [outDim, inDim] in i2_s format.
    Each workgroup computes one element of dInput (one column sum).
    Uses shared memory reduction over outDim. -/
def bitLinearTransposeKernel (inDim outDim : Nat) (workgroupSize : Nat := 32) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let lid ← ShaderM.localId
  let wgid ← ShaderM.workgroupId
  let j := Exp.vec3X wgid   -- input index (one workgroup per j)
  let tid := Exp.vec3X lid   -- thread within workgroup

  -- Buffers
  let packedPerRow := inDim / 128 * 32 / 4  -- u32 words per row
  let totalPacked := outDim * packedPerRow
  let _weights ← ShaderM.declareInputBuffer "weights" (.array (.scalar .u32) totalPacked)
  let _scale ← ShaderM.declareInputBuffer "scale" (.array (.scalar .f32) 1)
  let _dOutput ← ShaderM.declareInputBuffer "dOutput" (.array (.scalar .f32) outDim)
  let _dInput ← ShaderM.declareOutputBuffer "dInput" (.array (.scalar .f32) inDim)

  ShaderM.sharedNamed "shared_acc" (.array (.scalar .f32) workgroupSize)

  let scaleVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := 1) "scale" (Exp.litU32 0)

  ShaderM.if_ (Exp.lt j (Exp.litU32 inDim)) (do
    -- Each thread accumulates partial sum over a strided subset of outDim
    let accVar ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    ShaderM.loop tid (Exp.litU32 outDim) (Exp.litU32 workgroupSize) fun i => do
      -- Read dOutput[i]
      let dOutVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := outDim) "dOutput" i

      -- Decode W[i, j] from i2_s packed format
      -- j is the column (input dimension)
      -- Block of 128 elements per row, j determines which block and position
      let blockIdx := Exp.div j (Exp.litU32 128)   -- which 128-element block
      let posInBlock := Exp.mod j (Exp.litU32 128)  -- position within block
      let byteIdx := Exp.mod posInBlock (Exp.litU32 32)  -- byte within block
      let shiftGroup := Exp.div posInBlock (Exp.litU32 32)  -- which 2-bit group (0-3)

      -- Word index in packed array: row_offset + block_offset + byte/4
      let rowOffset := Exp.mul i (Exp.litU32 packedPerRow)
      let blockOffset := Exp.mul blockIdx (Exp.litU32 32)  -- 32 u32 words per block... actually 8 u32 words per block
      -- Actually: 128 elements / 4 per byte = 32 bytes = 8 u32 words per block
      -- Wait, the i2_s format packs 4 values per byte (2 bits each).
      -- 128 elements / 4 = 32 bytes = 8 u32 words per 128-element block.
      -- But the layout interleaves: bytes[0..31] each contribute to 4 groups of 32.
      -- So for element j in block:
      --   byte_index = j % 32
      --   shift = (j / 32) * 2  (group 0: shift 6, group 1: shift 4, group 2: shift 2, group 3: shift 0)

      let wordIdx := Exp.add rowOffset (Exp.add (Exp.mul blockIdx (Exp.litU32 8)) (Exp.div byteIdx (Exp.litU32 4)))
      let byteShift := Exp.mul (Exp.mod byteIdx (Exp.litU32 4)) (Exp.litU32 8)
      let groupShift := Exp.sub (Exp.litU32 6) (Exp.mul shiftGroup (Exp.litU32 2))

      let packedWord ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalPacked) "weights" wordIdx
      let byteVal := Exp.bitAnd (Exp.shiftRight packedWord byteShift) (Exp.litU32 0xFF)
      let code := Exp.bitAnd (Exp.shiftRight byteVal groupShift) (Exp.litU32 3)
      let weight := Exp.sub (Exp.toF32 code) (Exp.litF32 1.0)

      -- Accumulate: acc += weight * dOutput[i]
      ShaderM.assign accVar (Exp.add (Exp.var accVar) (Exp.mul weight dOutVal))

    -- Shared memory reduction
    ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_acc" tid (Exp.var accVar)
    ShaderM.barrier

    let numSteps := Nat.log2 workgroupSize
    ShaderM.staticLoop numSteps fun step => do
      let s := workgroupSize >>> (step + 1)
      ShaderM.if_ (Exp.lt tid (Exp.litU32 s)) (do
        let other ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_acc" (Exp.add tid (Exp.litU32 s))
        let cur ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_acc" tid
        ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_acc" tid (Exp.add cur other)
      ) (pure ())
      ShaderM.barrier

    -- Thread 0 writes final result
    ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
      let result ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_acc" (Exp.litU32 0)
      ShaderM.writeBuffer (ty := .scalar .f32) "dInput" j (Exp.mul scaleVal result)
    ) (pure ())
  ) (pure ())

/-- Execute BitLinear transpose: dInput = scale * W^T @ dOutput -/
def executeBitLinearTranspose (device : Device) (layer : Hesper.Layers.BitLinear.BitLinear)
    (dOutputBuf dInputBuf : Buffer) : IO Unit := do
  let inDim := layer.config.inDim
  let outDim := layer.config.outDim
  let workgroupSize := 32
  let shader := bitLinearTransposeKernel inDim outDim workgroupSize
  let namedBuffers := [("weights", layer.weightsPacked), ("scale", layer.scaleBuf),
                       ("dOutput", dOutputBuf), ("dInput", dInputBuf)]
  let execConfig : Hesper.WGSL.Execute.ExecutionConfig := {
    workgroupSize := {x := workgroupSize, y := 1, z := 1}
    numWorkgroups := (inDim, 1, 1)
  }
  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig

end Hesper.Training.BitLinearBackward
