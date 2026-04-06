import Hesper.WGSL.Monad
import Hesper.WGSL.Execute
import Hesper.WGSL.Exp
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.Quantization.Q4_K_M
import Hesper.Logging

/-!
# Q4_K_M Linear Layer - Fused Dequantization + Matrix-Vector Multiply

Implements a linear (fully-connected) layer with Q4_K_M quantized weights.
The kernel reads packed Q4_K_M weights, dequantizes on-the-fly, and accumulates
the dot product in a single fused pass — no intermediate F32 weight buffer needed.

## Algorithm

Each workgroup computes one output element y[outIdx]:
1. Load input vector into shared memory (cooperative)
2. Each thread processes a stripe of the weight row:
   - Read Q4_K_M block header (d, dmin, scales)
   - For each element in its stripe: dequant + FMA
3. Tree reduction of partial sums
4. Thread 0 writes final result

## Memory Layout

Weights are stored as contiguous Q4_K_M blocks in row-major order:
- Row i occupies blocks [i * blocksPerRow .. (i+1) * blocksPerRow)
- Each block = 144 bytes = 256 elements
- blocksPerRow = inDim / 256

## References
- Hesper/Layers/BitLinear.lean (same pattern, different quant format)
- Hesper/Quantization/Q4_K_M.lean (dequant primitives)
-/

namespace Hesper.Layers.Linear

open Hesper.WGSL
open Hesper.WGSL.Monad
open Hesper.WebGPU
open Hesper.Quantization.Q4_K_M (fp16ToF32 getScaleMin)
open Hesper.Logging (logVerbose)

/-! ## Layer Configuration -/

structure Config where
  inDim : Nat
  outDim : Nat
  deriving Repr, Inhabited

/-! ## Fused Q4_K_M MatVec Kernel -/

/-- Fused Q4_K_M dequant + matrix-vector multiply kernel.

    One workgroup per output element. Each thread processes a subset of the
    input dimension, dequantizing Q4_K_M blocks on-the-fly.

    @param config Layer dimensions
    @param workgroupSize Threads per workgroup (default 256)
-/
def fusedQ4KMLinearKernel (config : Config) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid  -- one workgroup per output element
  let tid := Exp.vec3X lid

  -- Buffer sizes
  let blocksPerRow := config.inDim / 256  -- Q4_K block = 256 elements
  let totalWeightU32 := config.outDim * blocksPerRow * 36  -- 144 bytes = 36 u32s per block

  -- Declare buffers
  let _weights ← ShaderM.declareInputBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) config.inDim)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) config.outDim)

  -- Shared memory for reduction
  ShaderM.sharedNamed "shared_partial" (.array (.scalar .f32) workgroupSize)

  -- Bounds check
  let inBounds := Exp.lt outIdx (Exp.litU32 config.outDim)

  -- Accumulator
  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"

  -- Base u32 offset for this output row's weight blocks
  let rowBaseU32 := Exp.mul outIdx (Exp.litU32 (blocksPerRow * 36))

  -- Each thread processes blocks in a strided pattern
  ShaderM.loop tid (Exp.litU32 blocksPerRow) (Exp.litU32 workgroupSize) fun blockLocalIdx => do
    -- Block u32 offset
    let blockU32Base := Exp.add rowBaseU32 (Exp.mul blockLocalIdx (Exp.litU32 36))
    -- Element offset in input for this block
    let elemBase := Exp.mul blockLocalIdx (Exp.litU32 256)

    -- Read block header: d and dmin (first u32)
    let dmU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" blockU32Base
    let d := fp16ToF32 (Exp.bitAnd dmU32 (Exp.litU32 0xFFFF))
    let dmin := fp16ToF32 (Exp.shiftRight dmU32 (Exp.litU32 16))

    -- Read scales[12] as 3 u32s (bytes 4..15)
    let sc0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add blockU32Base (Exp.litU32 1))
    let sc1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add blockU32Base (Exp.litU32 2))
    let sc2 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add blockU32Base (Exp.litU32 3))

    -- Process 4 chunks of 64 elements each (matching dequant_row_q4_K pattern)
    -- Chunk c: elements [c*64 .. (c+1)*64), sub-blocks 2c and 2c+1
    for c in [0:4] do
      -- Sub-block indices for this chunk
      let is0 := c * 2      -- low nibble sub-block
      let is1 := c * 2 + 1  -- high nibble sub-block
      let (scaleA, minA) := getScaleMin is0 sc0 sc1 sc2
      let (scaleB, minB) := getScaleMin is1 sc0 sc1 sc2
      let d1 := Exp.mul d scaleA
      let m1 := Exp.mul dmin minA
      let d2 := Exp.mul d scaleB
      let m2 := Exp.mul dmin minB

      -- qs offset for this chunk: blockU32Base + 4 (skip header) + c*8 u32s
      let qsU32Base := Exp.add blockU32Base (Exp.litU32 (4 + c * 8))

      -- Process 32 elements with low nibble (sub-block is0)
      -- and 32 elements with high nibble (sub-block is1) from same qs bytes
      for l32 in [0:8] do  -- 8 u32s = 32 bytes = 32 element pairs
        let qsU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add qsU32Base (Exp.litU32 l32))
        -- Each u32 has 4 bytes, each byte has 2 nibbles
        for b in [0:4] do
          let byte := Exp.bitAnd (Exp.shiftRight qsU32 (Exp.litU32 (b * 8))) (Exp.litU32 0xFF)
          let qLow := Exp.bitAnd byte (Exp.litU32 0xF)
          let qHigh := Exp.shiftRight byte (Exp.litU32 4)
          -- Element indices in input
          let elemIdxLow := Exp.add elemBase (Exp.litU32 (c * 64 + l32 * 4 + b))
          let elemIdxHigh := Exp.add elemBase (Exp.litU32 (c * 64 + 32 + l32 * 4 + b))
          -- Read input values
          let inLow ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.inDim) "input" elemIdxLow
          let inHigh ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.inDim) "input" elemIdxHigh
          -- Dequant + FMA: y = d*sc*q - dmin*m
          let wLow := Exp.sub (Exp.mul d1 (Exp.toF32 qLow)) m1
          let wHigh := Exp.sub (Exp.mul d2 (Exp.toF32 qHigh)) m2
          ShaderM.assign "acc" (Exp.add acc (Exp.add (Exp.mul wLow inLow) (Exp.mul wHigh inHigh)))

  -- Write partial sum to shared memory for reduction
  ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_partial" tid acc
  ShaderM.barrier

  -- Tree reduction
  let mut stride := workgroupSize / 2
  while stride > 0 do
    ShaderM.if_ (Exp.lt tid (Exp.litU32 stride)) (do
      let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_partial" tid
      let b ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_partial" (Exp.add tid (Exp.litU32 stride))
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_partial" tid (Exp.add a b)
    ) (pure ())
    ShaderM.barrier
    stride := stride / 2

  -- Thread 0 writes result
  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    let totalSum ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_partial" (Exp.litU32 0)
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx totalSum
  ) (pure ())

/-! ## Layer Structure -/

/-- Q4_K_M linear layer -/
structure LinearLayer where
  config : Config
  weightBuf : Buffer    -- Raw Q4_K_M packed weights on GPU
  prepared : IO.Ref (Option Execute.PreparedDispatch)

/-- Execute the linear layer: output = input @ weights^T

    @param device WebGPU device
    @param layer The linear layer
    @param inputBuf GPU buffer with input vector [inDim]
    @param outputBuf GPU buffer for output vector [outDim]
-/
def LinearLayer.forward (device : Device) (layer : LinearLayer)
    (inputBuf outputBuf : Buffer) : IO Unit := do
  let namedBuffers := [
    ("weights", layer.weightBuf),
    ("input", inputBuf),
    ("output", outputBuf)
  ]
  let execConfig := Execute.ExecutionConfig.dispatch1D
    layer.config.outDim
    256
  let shader := fusedQ4KMLinearKernel layer.config
  Execute.executeShaderNamed device shader namedBuffers execConfig
    (preparedRef := some layer.prepared)

end Hesper.Layers.Linear
