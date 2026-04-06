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

## i2_s Element Indexing (from forward kernel)

Given a row's u32 array, u32 index u32Idx decodes to 16 elements:
```
group128 = u32Idx / 8
groupPos = (u32Idx % 8) * 4

For byte b in [0..3], shift s in [0..3]:
  elemIdx = group128 * 128 + groupPos + b + s * 32
  code = ((packed >> (b*8)) >> (6 - s*2)) & 3
  weight = code - 1
```

For transpose (column j access across rows):
```
group128 = j / 128
posInGroup = j % 128
s = posInGroup / 32        (shift group)
subPos = posInGroup % 32   (within 32-element sub-group)
b = subPos % 4             (byte index within u32)
u32InGroup = subPos / 4    (u32 within group)
u32Idx = group128 * 8 + u32InGroup

byte_shift = b * 8
code_shift = 6 - s * 2
```
-/

namespace Hesper.Training.BitLinearBackward

open Hesper.WGSL
open Hesper.WGSL.Monad
open Hesper.WebGPU

/-- Transpose matmul kernel: dInput[j] = scale * Σ_i W[i,j] * dOutput[i]

    W is [outDim, inDim] in i2_s format.
    One workgroup per input element j, with threads cooperating over outDim.
    Uses shared memory reduction. -/
def bitLinearTransposeKernel (inDim outDim : Nat) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let wgid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let j := Exp.vec3X wgid    -- input element index (column)
  let tid := Exp.vec3X lid    -- thread within workgroup

  let u32PerRow := inDim / 16
  let totalPackedU32 := outDim * u32PerRow

  let _weights ← ShaderM.declareInputBuffer "weights" (.array (.scalar .u32) totalPackedU32)
  let _scale ← ShaderM.declareInputBuffer "scale" (.array (.scalar .f32) 1)
  let _dOutput ← ShaderM.declareInputBuffer "dOutput" (.array (.scalar .f32) outDim)
  let _dInput ← ShaderM.declareOutputBuffer "dInput" (.array (.scalar .f32) inDim)

  ShaderM.sharedNamed "shared_acc" (.array (.scalar .f32) workgroupSize)

  ShaderM.if_ (Exp.lt j (Exp.litU32 inDim)) (do
    -- Pre-compute column j's position in i2_s packed format
    -- These are constant for all rows (only j varies per workgroup)
    let group128 := Exp.div j (Exp.litU32 128)
    let posInGroup := Exp.mod j (Exp.litU32 128)
    let sGroup := Exp.div posInGroup (Exp.litU32 32)    -- shift group (0-3)
    let subPos := Exp.mod posInGroup (Exp.litU32 32)     -- position in sub-group
    let byteIdx := Exp.mod subPos (Exp.litU32 4)         -- byte within u32
    let u32InGroup := Exp.div subPos (Exp.litU32 4)      -- u32 within group

    let colU32Offset := Exp.add (Exp.mul group128 (Exp.litU32 8)) u32InGroup
    let byteShift := Exp.mul byteIdx (Exp.litU32 8)
    let codeShift := Exp.sub (Exp.litU32 6) (Exp.mul sGroup (Exp.litU32 2))

    -- Each thread accumulates partial sum over strided rows
    let accVar ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    ShaderM.loop tid (Exp.litU32 outDim) (Exp.litU32 workgroupSize) fun i => do
      let dOutVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := outDim) "dOutput" i

      -- Read W[i, j] from packed weights
      let rowBase := Exp.mul i (Exp.litU32 u32PerRow)
      let packedWord ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalPackedU32) "weights" (Exp.add rowBase colU32Offset)
      let theByte := Exp.bitAnd (Exp.shiftRight packedWord byteShift) (Exp.litU32 0xFF)
      let code := Exp.bitAnd (Exp.shiftRight theByte codeShift) (Exp.litU32 3)
      let weight := Exp.sub (Exp.toF32 code) (Exp.litF32 1.0)

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
      let scaleVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := 1) "scale" (Exp.litU32 0)
      ShaderM.writeBuffer (ty := .scalar .f32) "dInput" j (Exp.mul scaleVal result)
    ) (pure ())
  ) (pure ())

/-- Execute BitLinear transpose: dInput = scale * W^T @ dOutput -/
def executeBitLinearTranspose (device : Device) (layer : Hesper.Layers.BitLinear.BitLinear)
    (dOutputBuf dInputBuf : Buffer) : IO Unit := do
  let inDim := layer.config.inDim
  let outDim := layer.config.outDim
  let workgroupSize := 256
  let shader := bitLinearTransposeKernel inDim outDim workgroupSize
  let namedBuffers := [("weights", layer.weightsPacked), ("scale", layer.scaleBuf),
                       ("dOutput", dOutputBuf), ("dInput", dInputBuf)]
  let execConfig : Hesper.WGSL.Execute.ExecutionConfig := {
    workgroupSize := {x := workgroupSize, y := 1, z := 1}
    numWorkgroups := (inDim, 1, 1)
  }
  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig

end Hesper.Training.BitLinearBackward
