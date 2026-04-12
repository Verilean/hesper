import Hesper.Backend
import Hesper.WGSL.Monad
import Hesper.WGSL.Exp
import Hesper.Quantization.TQ2_0
import Hesper.Basic
import Hesper.Logging
import Hesper.Layers.RMSNorm

/-!
# BitLinear Layer - Ternary Weight Matrix Multiplication (i2_s format)

Implements BitNet's BitLinear layer with on-the-fly i2_s dequantization on GPU.

## Key Innovation: Fused Kernel
Instead of:
1. Unpack i2_s → Float32 on CPU (slow, memory-intensive)
2. Matrix multiply Float32 × Float32

We do:
1. Upload raw packed i2_s bytes to GPU
2. **Read packed weights + compute matmul in same kernel**

## i2_s Packing Format

Encoding table:
| Ternary | 2-bit code |
|---------|-----------|
| -1      | 0b00 (0)  |
|  0      | 0b01 (1)  |
| +1      | 0b10 (2)  |

Dequantization: `float_value = (code - 1) * scale`

Layout (groups of 128 elements per 32 bytes):
- Elements [0..31]:    bytes[0..31] >> 6 & 3
- Elements [32..63]:   bytes[0..31] >> 4 & 3
- Elements [64..95]:   bytes[0..31] >> 2 & 3
- Elements [96..127]:  bytes[0..31] >> 0 & 3

Scale: single F32 at the END of tensor data.

## BitLinear Mathematics

For ternary weights w in {-1, 0, 1}, the matrix-vector product:

```
y[i] = scale * sum_j( w[i,j] * x[j] )
     = scale * (sum_{w=+1} x[j] - sum_{w=-1} x[j])
```

## References
- BitNet paper: https://arxiv.org/abs/2402.17764
- bitnet.cpp: i2_s format specification
-/

namespace Hesper.Layers.BitLinear

open Hesper.WGSL
open Hesper.WGSL.Monad
open Hesper
open Hesper.Logging (logVerbose)

/-- Counters for PreparedDispatch fast-path vs slow-path -/
initialize preparedHitsRef : IO.Ref Nat ← IO.mkRef 0
initialize preparedMissesRef : IO.Ref Nat ← IO.mkRef 0

/-- Runtime opt-in for the subgroup-matrix BitLinear kernel. Defaults
    to **on** when the device also has ShaderF16 + SubgroupMatrix (see
    dispatch logic in `forward`). Set to `false` to force the tiled
    fallback for debugging. -/
initialize subgroupMatrixOptInRef : IO.Ref Bool ← IO.mkRef true

/-! ## Layer Configuration -/

/-- BitLinear layer configuration -/
structure Config where
  inDim : Nat      -- Input dimension
  outDim : Nat     -- Output dimension
  batchSize : Nat  -- Batch size
  deriving Repr, Inhabited

/-! ## Fused BitLinear Kernel (i2_s format) -/

/-- Vectorized fused kernel: i2_s unpack + matrix-vector multiply

    **Algorithm** (vectorized, 16 weights per u32 read):
    ```
    for each output element y[out_idx]:
      1. Load input[0..inDim] into shared memory (all threads cooperate)
      2. acc = 0.0
      3. for each u32 in packed weights (strided over threads):
           Read 1 u32 = 4 bytes = 16 packed 2-bit weights
           Extract all 16 weights via compile-time unrolled shifts
           Accumulate 16 FMAs from shared memory input
      4. Tree reduction of partial sums
      5. y[out_idx] = scale * total_sum
    ```

    **Key optimizations over v1**:
    - 16x fewer weight buffer reads (1 u32 → 16 elements vs 1 u32 → 1 element)
    - Input cached in shared memory (10KB for dim=2560, fast random access)
    - Coalesced weight reads (consecutive threads read consecutive u32s)
    - Compile-time unrolled inner loop (16 FMAs with no branch overhead)
    - Tiled input loading for large dims (>3584 elements)

    **i2_s unpacking** (vectorized per u32):
    All 4 bytes of a u32 are always within the same group-128 block because
    u32 boundaries (4 bytes) never span a 32-byte group boundary.
    ```
    group128 = u32Idx / 8
    baseGroupPos = (u32Idx % 8) * 4
    For byte b in 0..4, shift s in [6,4,2,0]:
      elemIdx = group128 * 128 + baseGroupPos + b + (3-s/2) * 32
      code = (byte >> s) & 3
      ternary = code - 1
    ```

    @param config BitLinear layer configuration
    @param numRows Number of input rows (batch * seq_len)
    @param workgroupSize Threads per workgroup (default 256)
-/
def fusedBitLinearKernel (config : Config) (numRows : Nat) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let flatWgId := Exp.vec3X wid
  let tid := Exp.vec3X lid

  let totalOutputs := numRows * config.outDim

  -- Decode workgroup: each workgroup computes one output element
  let outIdx := Exp.mod flatWgId (Exp.litU32 config.outDim)
  let rowIdx := Exp.div flatWgId (Exp.litU32 config.outDim)

  -- Bounds check (for when numWorkgroups > totalOutputs due to rounding)
  let inBounds := Exp.lt flatWgId (Exp.litU32 totalOutputs)

  -- Buffer sizes
  let totalWeightElements := config.outDim * config.inDim
  let numPackedBytes := totalWeightElements / 4
  let numPackedU32 := (numPackedBytes + 3) / 4
  let totalInputElements := numRows * config.inDim

  -- Tiling: shared memory budget = ~15KB for input + 1KB for reduction = 16KB total
  -- Tile on 128-element boundaries (matching group-128 layout)
  let maxSharedInputElems := 3584  -- 128 * 28 = 14336 bytes, + 1024 reduction = 15360 < 16KB
  let tileElemSize :=
    if config.inDim ≤ maxSharedInputElems then config.inDim
    else (maxSharedInputElems / 128) * 128  -- round down to 128 boundary
  let numTiles := (config.inDim + tileElemSize - 1) / tileElemSize

  -- Declare shared memory for input cache and parallel reduction
  ShaderM.sharedNamed "shared_input" (.array (.scalar .f32) tileElemSize)
  ShaderM.sharedNamed "shared_partial" (.array (.scalar .f32) workgroupSize)

  -- Declare buffers
  let _packed ← ShaderM.declareInputBuffer "weights_packed" (.array (.scalar .u32) numPackedU32)
  let _scale ← ShaderM.declareInputBuffer "scale" (.array (.scalar .f32) 1)
  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) totalInputElements)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) totalOutputs)

  -- Accumulator for all tiles
  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"

  -- Weight buffer: row outIdx starts at u32 index outIdx * (inDim / 16)
  let u32PerRow := config.inDim / 16  -- 16 elements per u32 (4 bytes × 4 elements/byte)
  let rowBaseU32 := Exp.mul outIdx (Exp.litU32 u32PerRow)

  -- Input row base
  let inputRowBase := Exp.mul rowIdx (Exp.litU32 config.inDim)

  -- Process each tile (compile-time unrolled since numTiles is a Lean Nat)
  for t in [0:numTiles] do
    let tileStart := t * tileElemSize
    let tileEnd := min ((t + 1) * tileElemSize) config.inDim
    let actualTileSize := tileEnd - tileStart

    -- Step 1: Cooperatively load input tile into shared memory
    ShaderM.loop tid (Exp.litU32 actualTileSize) (Exp.litU32 workgroupSize) fun i => do
      let globalIdx := Exp.add inputRowBase (Exp.add (Exp.litU32 tileStart) i)
      let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalInputElements) "input" globalIdx
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_input" i val
    ShaderM.barrier

    -- Step 2: Process weight u32s for this tile
    -- Tile covers elements [tileStart, tileEnd), which maps to u32s [tileStart/16, tileEnd/16)
    let tileU32Start := tileStart / 16
    let tileU32Count := actualTileSize / 16

    ShaderM.loop tid (Exp.litU32 tileU32Count) (Exp.litU32 workgroupSize) fun localU32Idx => do
      -- Read one u32 = 4 bytes = 16 packed 2-bit weights
      let absU32Idx := Exp.add (Exp.litU32 tileU32Start) localU32Idx
      let packedU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := numPackedU32) "weights_packed" (Exp.add rowBaseU32 absU32Idx)

      -- Compute local element base within this tile
      -- All 4 bytes of a u32 are in the same group-128 block
      -- localElemIdx = (localU32Idx/8)*128 + (localU32Idx%8)*4 + byte + shift_offset
      let localGroup := Exp.div localU32Idx (Exp.litU32 8)
      let localGroupPos := Exp.mul (Exp.mod localU32Idx (Exp.litU32 8)) (Exp.litU32 4)
      let localElemBase := Exp.add (Exp.mul localGroup (Exp.litU32 128)) localGroupPos

      -- Process 4 bytes × 4 shifts = 16 elements (compile-time unrolled)
      for b in [0:4] do
        -- Extract byte b from the u32
        let theByte := Exp.bitAnd (Exp.shiftRight packedU32 (Exp.litU32 (b * 8))) (Exp.litU32 0xFF)
        for s in [0:4] do
          -- Extract 2-bit code at shift position (descending: 6, 4, 2, 0)
          let code := Exp.bitAnd (Exp.shiftRight theByte (Exp.litU32 (6 - s * 2))) (Exp.litU32 0x3)
          let ternaryF32 := Exp.sub (Exp.toF32 code) (Exp.litF32 1.0)
          -- Element offset: byte position + shift group * 32
          let localElemIdx := Exp.add localElemBase (Exp.litU32 (b + s * 32))
          let inputVal ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := tileElemSize) "shared_input" localElemIdx
          ShaderM.assign "acc" (Exp.add acc (Exp.mul ternaryF32 inputVal))

    ShaderM.barrier  -- Ensure all threads done before next tile overwrites shared memory

  -- Write partial sum to shared memory for reduction
  ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_partial" tid acc
  ShaderM.barrier

  -- Tree reduction in shared memory
  let mut stride := workgroupSize / 2
  while stride > 0 do
    ShaderM.if_ (Exp.lt tid (Exp.litU32 stride)) (do
      let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_partial" tid
      let b ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_partial" (Exp.add tid (Exp.litU32 stride))
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_partial" tid (Exp.add a b)
    ) (pure ())
    ShaderM.barrier
    stride := stride / 2

  -- Thread 0 of each workgroup writes the final result
  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    let totalSum ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_partial" (Exp.litU32 0)
    let scaleVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := 1) "scale" (Exp.litU32 0)
    let result := Exp.mul scaleVal totalSum
    let outputIdx := Exp.add (Exp.mul rowIdx (Exp.litU32 config.outDim)) outIdx
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outputIdx result
  ) (pure ())

/-! ## Shared-Memory Tree Reduction Helper -/

/-- Emit shared-memory tree reduction for a workgroup of 32 threads.
    Reads the accumulator value, writes it to shared memory, then performs
    5 rounds of tree reduction (stride 16, 8, 4, 2, 1).
    Returns an expression holding the final reduced sum (valid only for tid==0).

    @param tid Thread ID expression
    @param acc Accumulator expression to reduce
    @param sharedName Name for the shared memory variable
    @return Expression for the reduced sum (shared[0]) -/
private def sharedMemReduce32 (tid : Exp (.scalar .u32)) (acc : Exp (.scalar .f32))
    (sharedName : String := "shared_reduce") : ShaderM (Exp (.scalar .f32)) := do
  ShaderM.sharedNamed sharedName (.array (.scalar .f32) 32)
  ShaderM.writeWorkgroup (ty := .scalar .f32) sharedName tid acc
  ShaderM.barrier
  -- Tree reduction: 5 rounds for 32 threads
  for stride in [16, 8, 4, 2, 1] do
    ShaderM.if_ (Exp.lt tid (Exp.litU32 stride)) (do
      let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 32) sharedName tid
      let b ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 32) sharedName (Exp.add tid (Exp.litU32 stride))
      ShaderM.writeWorkgroup (ty := .scalar .f32) sharedName tid (Exp.add a b)
    ) (pure ())
    ShaderM.barrier
  ShaderM.readWorkgroup (ty := .scalar .f32) (n := 32) sharedName (Exp.litU32 0)

/-! ## M=1 Warp-Cooperative BitLinear Kernel (Single-Token Inference) -/

/-- M=1 warp-cooperative kernel: one subgroup (32 threads) per output element

    For single-token inference (M=1), uses subgroup-level cooperation:
    - 32 threads in a subgroup cooperatively process one output row
    - Weight reads are COALESCED: consecutive threads read consecutive u32s
    - Input reads are L2-cached: all subgroups read the same 10-27KB input vector
    - Reduction via hardware `subgroupAdd` (no shared memory, no barriers)

    **vs tiled kernel (256 threads/workgroup):**
    - No shared memory for input (10KB × outDim redundant loads eliminated)
    - No shared memory for reduction (subgroupAdd replaces tree reduction)
    - No barriers (subgroup ops are implicit)
    - 8x fewer threads (32 vs 256 per output)

    **Algorithm per subgroup (32 threads):**
    ```
    for u32Idx = tid; u32Idx < u32PerRow; u32Idx += 32:
      // Coalesced: thread k reads u32 at rowBase + k, k+32, k+64...
      packed = weights[outIdx * u32PerRow + u32Idx]
      unpack 16 ternary weights
      read 16 input values (L2-cached)
      acc += 16 FMAs
    total = subgroupAdd(acc)
    if tid == 0: output[outIdx] = scale * total
    ```

    @param config BitLinear layer configuration
-/
def fusedBitLinearM1Kernel (config : Config) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid  -- one workgroup (= one subgroup) per output element
  let tid := Exp.vec3X lid     -- lane within subgroup (0-31)

  let totalWeightElements := config.outDim * config.inDim
  let numPackedBytes := totalWeightElements / 4
  let numPackedU32 := (numPackedBytes + 3) / 4

  -- Declare buffers (NO shared memory!)
  let _packed ← ShaderM.declareInputBuffer "weights_packed" (.array (.scalar .u32) numPackedU32)
  let _scale ← ShaderM.declareInputBuffer "scale" (.array (.scalar .f32) 1)
  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) config.inDim)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) config.outDim)

  -- Accumulator (each thread accumulates partial sum)
  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"

  -- Weight u32s per row: inDim / 16 (16 elements per u32)
  let u32PerRow := config.inDim / 16
  let rowBaseU32 := Exp.mul outIdx (Exp.litU32 u32PerRow)

  -- Strided loop: thread k processes u32s at positions k, k+32, k+64, ...
  -- Consecutive threads read consecutive u32s → COALESCED memory access
  ShaderM.loop tid (Exp.litU32 u32PerRow) (Exp.litU32 32) fun u32Idx => do
    let packedU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := numPackedU32) "weights_packed"
      (Exp.add rowBaseU32 u32Idx)

    -- i2_s group-128 layout: compute element base from u32 position
    let localGroup := Exp.div u32Idx (Exp.litU32 8)
    let localGroupPos := Exp.mul (Exp.mod u32Idx (Exp.litU32 8)) (Exp.litU32 4)
    let elemBase := Exp.add (Exp.mul localGroup (Exp.litU32 128)) localGroupPos

    -- Extract 4 bytes × 4 shifts = 16 elements (compile-time unrolled)
    for b in [0:4] do
      let theByte := Exp.bitAnd (Exp.shiftRight packedU32 (Exp.litU32 (b * 8))) (Exp.litU32 0xFF)
      for s in [0:4] do
        let code := Exp.bitAnd (Exp.shiftRight theByte (Exp.litU32 (6 - s * 2))) (Exp.litU32 0x3)
        let ternaryF32 := Exp.sub (Exp.toF32 code) (Exp.litF32 1.0)
        let elemIdx := Exp.add elemBase (Exp.litU32 (b + s * 32))
        let inputVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.inDim) "input" elemIdx
        ShaderM.assign "acc" (Exp.add acc (Exp.mul ternaryF32 inputVal))

  -- Subgroup reduction: hardware-accelerated sum across 32 threads
  -- Assign to var before non-uniform branch so all threads execute subgroupAdd uniformly
  ShaderM.varNamed "totalSum" (.scalar .f32) (Exp.subgroupAdd acc)
  let totalSum : Exp (.scalar .f32) := Exp.var "totalSum"

  -- Thread 0 writes the final result
  let inBounds := Exp.lt outIdx (Exp.litU32 config.outDim)
  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    let scaleVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := 1) "scale" (Exp.litU32 0)
    let result := Exp.mul scaleVal totalSum
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx result
  ) (pure ())

/-- M=1 warp-cooperative kernel with fused residual add

    Same as fusedBitLinearM1Kernel but outputs: output = residual + scale * dot_product

    @param config BitLinear layer configuration
-/
def fusedBitLinearResidualM1Kernel (config : Config) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid

  let totalWeightElements := config.outDim * config.inDim
  let numPackedBytes := totalWeightElements / 4
  let numPackedU32 := (numPackedBytes + 3) / 4

  let _packed ← ShaderM.declareInputBuffer "weights_packed" (.array (.scalar .u32) numPackedU32)
  let _scale ← ShaderM.declareInputBuffer "scale" (.array (.scalar .f32) 1)
  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) config.inDim)
  let _residual ← ShaderM.declareInputBuffer "residual" (.array (.scalar .f32) config.outDim)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) config.outDim)

  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"

  let u32PerRow := config.inDim / 16
  let rowBaseU32 := Exp.mul outIdx (Exp.litU32 u32PerRow)

  ShaderM.loop tid (Exp.litU32 u32PerRow) (Exp.litU32 32) fun u32Idx => do
    let packedU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := numPackedU32) "weights_packed"
      (Exp.add rowBaseU32 u32Idx)

    let localGroup := Exp.div u32Idx (Exp.litU32 8)
    let localGroupPos := Exp.mul (Exp.mod u32Idx (Exp.litU32 8)) (Exp.litU32 4)
    let elemBase := Exp.add (Exp.mul localGroup (Exp.litU32 128)) localGroupPos

    for b in [0:4] do
      let theByte := Exp.bitAnd (Exp.shiftRight packedU32 (Exp.litU32 (b * 8))) (Exp.litU32 0xFF)
      for s in [0:4] do
        let code := Exp.bitAnd (Exp.shiftRight theByte (Exp.litU32 (6 - s * 2))) (Exp.litU32 0x3)
        let ternaryF32 := Exp.sub (Exp.toF32 code) (Exp.litF32 1.0)
        let elemIdx := Exp.add elemBase (Exp.litU32 (b + s * 32))
        let inputVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.inDim) "input" elemIdx
        ShaderM.assign "acc" (Exp.add acc (Exp.mul ternaryF32 inputVal))

  -- Assign to var before non-uniform branch so all threads execute subgroupAdd uniformly
  ShaderM.varNamed "totalSum" (.scalar .f32) (Exp.subgroupAdd acc)
  let totalSum : Exp (.scalar .f32) := Exp.var "totalSum"

  let inBounds := Exp.lt outIdx (Exp.litU32 config.outDim)
  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    let scaleVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := 1) "scale" (Exp.litU32 0)
    let residualVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.outDim) "residual" outIdx
    let result := Exp.add residualVal (Exp.mul scaleVal totalSum)
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx result
  ) (pure ())

/-! ## Fused RMSNorm + BitLinear + Residual Add M=1 Kernel -/

/-- Fused RMSNorm + BitLinear + Residual M=1 kernel.

    Combines three operations into one dispatch:
    1. RMSNorm: compute rmsInv = rsqrt(mean(input²) + eps)
    2. BitLinear dot product with inline normalization:
       dot = sum_j(ternary[outIdx,j] * input[j] * rmsInv * rmsScale[j])
    3. Residual add: output[outIdx] = residual[outIdx] + blScale * dot

    Saves 1 dispatch per call (was: RMSNorm + BitLinear = 2 dispatches).
    Used for: attn_sub_norm + O projection, ffn_sub_norm + down projection.

    **Algorithm per subgroup (32 threads):**
    ```
    Phase 1: Compute RMS via subgroupAdd
      partial_sq = 0
      for elemIdx = tid; elemIdx < inDim; elemIdx += 32:
        partial_sq += input[elemIdx]²
      totalSq = subgroupAdd(partial_sq)
      rmsInv = rsqrt(totalSq / dim + eps)

    Phase 2: BitLinear dot product with inline normalization
      for u32Idx = tid; u32Idx < u32PerRow; u32Idx += 32:
        packed = weights[outIdx * u32PerRow + u32Idx]
        for each of 16 elements:
          normalized = input[elemIdx] * rmsInv * rmsScale[elemIdx]
          acc += ternary * normalized
      total = subgroupAdd(acc)
      if tid == 0: output[outIdx] = residual[outIdx] + blScale * total
    ```

    @param config BitLinear layer configuration
    @param eps RMSNorm epsilon (typically 1e-5)
-/
def fusedRMSNormBitLinearResidualM1Kernel (config : Config) (eps : Float := 1e-5) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid

  let totalWeightElements := config.outDim * config.inDim
  let numPackedBytes := totalWeightElements / 4
  let numPackedU32 := (numPackedBytes + 3) / 4

  -- Declare buffers (6 bindings)
  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) config.inDim)
  let _rmsScale ← ShaderM.declareInputBuffer "rms_scale" (.array (.scalar .f32) config.inDim)
  let _packed ← ShaderM.declareInputBuffer "weights_packed" (.array (.scalar .u32) numPackedU32)
  let _blScale ← ShaderM.declareInputBuffer "bl_scale" (.array (.scalar .f32) 1)
  let _residual ← ShaderM.declareInputBuffer "residual" (.array (.scalar .f32) config.outDim)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) config.outDim)

  -- Phase 1: Compute RMS via subgroupAdd
  ShaderM.varNamed "partial_sq" (.scalar .f32) (Exp.litF32 0.0)
  let partialSq : Exp (.scalar .f32) := Exp.var "partial_sq"

  ShaderM.loop tid (Exp.litU32 config.inDim) (Exp.litU32 32) fun elemIdx => do
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.inDim) "input" elemIdx
    ShaderM.assign "partial_sq" (Exp.add partialSq (Exp.mul val val))

  -- subgroupAdd to get total sum of squares
  ShaderM.varNamed "totalSq" (.scalar .f32) (Exp.subgroupAdd partialSq)
  let totalSq : Exp (.scalar .f32) := Exp.var "totalSq"

  -- rmsInv = rsqrt(totalSq / dim + eps)
  let mean := Exp.div totalSq (Exp.litF32 config.inDim.toFloat)
  let rmsInv := Exp.inverseSqrt (Exp.add mean (Exp.litF32 eps))

  -- Phase 2: BitLinear dot product with inline normalization
  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"

  let u32PerRow := config.inDim / 16
  let rowBaseU32 := Exp.mul outIdx (Exp.litU32 u32PerRow)

  ShaderM.loop tid (Exp.litU32 u32PerRow) (Exp.litU32 32) fun u32Idx => do
    let packedU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := numPackedU32) "weights_packed"
      (Exp.add rowBaseU32 u32Idx)

    let localGroup := Exp.div u32Idx (Exp.litU32 8)
    let localGroupPos := Exp.mul (Exp.mod u32Idx (Exp.litU32 8)) (Exp.litU32 4)
    let elemBase := Exp.add (Exp.mul localGroup (Exp.litU32 128)) localGroupPos

    for b in [0:4] do
      let theByte := Exp.bitAnd (Exp.shiftRight packedU32 (Exp.litU32 (b * 8))) (Exp.litU32 0xFF)
      for s in [0:4] do
        let code := Exp.bitAnd (Exp.shiftRight theByte (Exp.litU32 (6 - s * 2))) (Exp.litU32 0x3)
        let ternaryF32 := Exp.sub (Exp.toF32 code) (Exp.litF32 1.0)
        let elemIdx := Exp.add elemBase (Exp.litU32 (b + s * 32))
        -- Read input and normalize inline: input[i] * rmsInv * rmsScale[i]
        let inputVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.inDim) "input" elemIdx
        let scaleVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.inDim) "rms_scale" elemIdx
        let normalized := Exp.mul (Exp.mul inputVal rmsInv) scaleVal
        ShaderM.assign "acc" (Exp.add acc (Exp.mul ternaryF32 normalized))

  -- Subgroup reduction
  ShaderM.varNamed "totalSum" (.scalar .f32) (Exp.subgroupAdd acc)
  let totalSum : Exp (.scalar .f32) := Exp.var "totalSum"

  -- Thread 0: output = residual + blScale * totalSum
  let inBounds := Exp.lt outIdx (Exp.litU32 config.outDim)
  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    let blScaleVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := 1) "bl_scale" (Exp.litU32 0)
    let residualVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.outDim) "residual" outIdx
    let result := Exp.add residualVal (Exp.mul blScaleVal totalSum)
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx result
  ) (pure ())

/-! ## Fused Gate+Up+ReLU²×Mul M=1 Kernel -/

/-- Fused gate + up + ReLU²×mul M=1 kernel.

    Combines three operations into one dispatch:
    1. gate_val = dot(gate_weights, input) * gate_scale
    2. up_val = dot(up_weights, input) * up_scale
    3. output = ReLU²(gate_val) * up_val = max(0, gate_val)² * up_val

    Saves 2 dispatches per call (was: gate BitLinear + up BitLinear + ReluSqrMul = 3).
    Input is read once and used for both gate and up dot products.

    **Algorithm per subgroup (32 threads):**
    ```
    gate_acc = 0, up_acc = 0
    for u32Idx = tid; u32Idx < u32PerRow; u32Idx += 32:
      // Read input once, unpack both gate and up weights
      gate_packed = gate_weights[outIdx * u32PerRow + u32Idx]
      up_packed = up_weights[outIdx * u32PerRow + u32Idx]
      for each of 16 elements:
        input_val = input[elemIdx]
        gate_acc += gate_ternary * input_val
        up_acc += up_ternary * input_val
    gate_total = subgroupAdd(gate_acc) * gate_scale
    up_total = subgroupAdd(up_acc) * up_scale
    if tid == 0:
      relu_val = max(0, gate_total)
      output[outIdx] = relu_val * relu_val * up_total
    ```

    @param config BitLinear layer configuration (inDim=dim, outDim=ffnDim)
-/
def fusedGateUpReluSqrMulM1Kernel (config : Config) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid

  let totalWeightElements := config.outDim * config.inDim
  let numPackedBytes := totalWeightElements / 4
  let numPackedU32 := (numPackedBytes + 3) / 4

  -- Declare buffers (6 bindings)
  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) config.inDim)
  let _gatePacked ← ShaderM.declareInputBuffer "gate_packed" (.array (.scalar .u32) numPackedU32)
  let _gateScale ← ShaderM.declareInputBuffer "gate_scale" (.array (.scalar .f32) 1)
  let _upPacked ← ShaderM.declareInputBuffer "up_packed" (.array (.scalar .u32) numPackedU32)
  let _upScale ← ShaderM.declareInputBuffer "up_scale" (.array (.scalar .f32) 1)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) config.outDim)

  -- Accumulators for gate and up dot products
  ShaderM.varNamed "gate_acc" (.scalar .f32) (Exp.litF32 0.0)
  ShaderM.varNamed "up_acc" (.scalar .f32) (Exp.litF32 0.0)
  let gateAcc : Exp (.scalar .f32) := Exp.var "gate_acc"
  let upAcc : Exp (.scalar .f32) := Exp.var "up_acc"

  let u32PerRow := config.inDim / 16
  let rowBaseU32 := Exp.mul outIdx (Exp.litU32 u32PerRow)

  ShaderM.loop tid (Exp.litU32 u32PerRow) (Exp.litU32 32) fun u32Idx => do
    -- Read both gate and up packed weights
    let gatePackedU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := numPackedU32) "gate_packed"
      (Exp.add rowBaseU32 u32Idx)
    let upPackedU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := numPackedU32) "up_packed"
      (Exp.add rowBaseU32 u32Idx)

    let localGroup := Exp.div u32Idx (Exp.litU32 8)
    let localGroupPos := Exp.mul (Exp.mod u32Idx (Exp.litU32 8)) (Exp.litU32 4)
    let elemBase := Exp.add (Exp.mul localGroup (Exp.litU32 128)) localGroupPos

    for b in [0:4] do
      let gateByte := Exp.bitAnd (Exp.shiftRight gatePackedU32 (Exp.litU32 (b * 8))) (Exp.litU32 0xFF)
      let upByte := Exp.bitAnd (Exp.shiftRight upPackedU32 (Exp.litU32 (b * 8))) (Exp.litU32 0xFF)
      for s in [0:4] do
        let gateCode := Exp.bitAnd (Exp.shiftRight gateByte (Exp.litU32 (6 - s * 2))) (Exp.litU32 0x3)
        let gateTernary := Exp.sub (Exp.toF32 gateCode) (Exp.litF32 1.0)
        let upCode := Exp.bitAnd (Exp.shiftRight upByte (Exp.litU32 (6 - s * 2))) (Exp.litU32 0x3)
        let upTernary := Exp.sub (Exp.toF32 upCode) (Exp.litF32 1.0)
        let elemIdx := Exp.add elemBase (Exp.litU32 (b + s * 32))
        -- Read input once, use for both
        let inputVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.inDim) "input" elemIdx
        ShaderM.assign "gate_acc" (Exp.add gateAcc (Exp.mul gateTernary inputVal))
        ShaderM.assign "up_acc" (Exp.add upAcc (Exp.mul upTernary inputVal))

  -- Subgroup reduction for both accumulators
  ShaderM.varNamed "gate_total" (.scalar .f32) (Exp.subgroupAdd gateAcc)
  ShaderM.varNamed "up_total" (.scalar .f32) (Exp.subgroupAdd upAcc)
  let gateTotal : Exp (.scalar .f32) := Exp.var "gate_total"
  let upTotal : Exp (.scalar .f32) := Exp.var "up_total"

  -- Thread 0: apply scales, ReLU², multiply, write output
  let inBounds := Exp.lt outIdx (Exp.litU32 config.outDim)
  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    let gateScaleVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := 1) "gate_scale" (Exp.litU32 0)
    let upScaleVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := 1) "up_scale" (Exp.litU32 0)
    let gateScaled := Exp.mul gateScaleVal gateTotal
    let upScaled := Exp.mul upScaleVal upTotal
    -- ReLU²(gate) × up
    let reluVal := Exp.max gateScaled (Exp.litF32 0.0)
    let result := Exp.mul (Exp.mul reluVal reluVal) upScaled
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx result
  ) (pure ())

/-! ## Shared-Memory Fallback M=1 Kernels (No Subgroup Support) -/

/-- M=1 BitLinear kernel using shared-memory reduction (no subgroups required).
    Same algorithm as `fusedBitLinearM1Kernel` but uses tree reduction instead of `subgroupAdd`. -/
def fusedBitLinearM1KernelSharedMem (config : Config) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid

  let totalWeightElements := config.outDim * config.inDim
  let numPackedBytes := totalWeightElements / 4
  let numPackedU32 := (numPackedBytes + 3) / 4

  let _packed ← ShaderM.declareInputBuffer "weights_packed" (.array (.scalar .u32) numPackedU32)
  let _scale ← ShaderM.declareInputBuffer "scale" (.array (.scalar .f32) 1)
  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) config.inDim)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) config.outDim)

  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"

  let u32PerRow := config.inDim / 16
  let rowBaseU32 := Exp.mul outIdx (Exp.litU32 u32PerRow)

  ShaderM.loop tid (Exp.litU32 u32PerRow) (Exp.litU32 32) fun u32Idx => do
    let packedU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := numPackedU32) "weights_packed"
      (Exp.add rowBaseU32 u32Idx)
    let localGroup := Exp.div u32Idx (Exp.litU32 8)
    let localGroupPos := Exp.mul (Exp.mod u32Idx (Exp.litU32 8)) (Exp.litU32 4)
    let elemBase := Exp.add (Exp.mul localGroup (Exp.litU32 128)) localGroupPos
    for b in [0:4] do
      let theByte := Exp.bitAnd (Exp.shiftRight packedU32 (Exp.litU32 (b * 8))) (Exp.litU32 0xFF)
      for s in [0:4] do
        let code := Exp.bitAnd (Exp.shiftRight theByte (Exp.litU32 (6 - s * 2))) (Exp.litU32 0x3)
        let ternaryF32 := Exp.sub (Exp.toF32 code) (Exp.litF32 1.0)
        let elemIdx := Exp.add elemBase (Exp.litU32 (b + s * 32))
        let inputVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.inDim) "input" elemIdx
        ShaderM.assign "acc" (Exp.add acc (Exp.mul ternaryF32 inputVal))

  -- Shared-memory tree reduction instead of subgroupAdd
  let totalSum ← sharedMemReduce32 tid acc "shared_reduce"

  let inBounds := Exp.lt outIdx (Exp.litU32 config.outDim)
  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    let scaleVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := 1) "scale" (Exp.litU32 0)
    let result := Exp.mul scaleVal totalSum
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx result
  ) (pure ())

/-- M=1 BitLinear+Residual kernel using shared-memory reduction (no subgroups required). -/
def fusedBitLinearResidualM1KernelSharedMem (config : Config) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid

  let totalWeightElements := config.outDim * config.inDim
  let numPackedBytes := totalWeightElements / 4
  let numPackedU32 := (numPackedBytes + 3) / 4

  let _packed ← ShaderM.declareInputBuffer "weights_packed" (.array (.scalar .u32) numPackedU32)
  let _scale ← ShaderM.declareInputBuffer "scale" (.array (.scalar .f32) 1)
  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) config.inDim)
  let _residual ← ShaderM.declareInputBuffer "residual" (.array (.scalar .f32) config.outDim)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) config.outDim)

  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"

  let u32PerRow := config.inDim / 16
  let rowBaseU32 := Exp.mul outIdx (Exp.litU32 u32PerRow)

  ShaderM.loop tid (Exp.litU32 u32PerRow) (Exp.litU32 32) fun u32Idx => do
    let packedU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := numPackedU32) "weights_packed"
      (Exp.add rowBaseU32 u32Idx)
    let localGroup := Exp.div u32Idx (Exp.litU32 8)
    let localGroupPos := Exp.mul (Exp.mod u32Idx (Exp.litU32 8)) (Exp.litU32 4)
    let elemBase := Exp.add (Exp.mul localGroup (Exp.litU32 128)) localGroupPos
    for b in [0:4] do
      let theByte := Exp.bitAnd (Exp.shiftRight packedU32 (Exp.litU32 (b * 8))) (Exp.litU32 0xFF)
      for s in [0:4] do
        let code := Exp.bitAnd (Exp.shiftRight theByte (Exp.litU32 (6 - s * 2))) (Exp.litU32 0x3)
        let ternaryF32 := Exp.sub (Exp.toF32 code) (Exp.litF32 1.0)
        let elemIdx := Exp.add elemBase (Exp.litU32 (b + s * 32))
        let inputVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.inDim) "input" elemIdx
        ShaderM.assign "acc" (Exp.add acc (Exp.mul ternaryF32 inputVal))

  let totalSum ← sharedMemReduce32 tid acc "shared_reduce"

  let inBounds := Exp.lt outIdx (Exp.litU32 config.outDim)
  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    let scaleVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := 1) "scale" (Exp.litU32 0)
    let residualVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.outDim) "residual" outIdx
    let result := Exp.add residualVal (Exp.mul scaleVal totalSum)
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx result
  ) (pure ())

/-- Fused RMSNorm+BitLinear+Residual M=1 kernel using shared-memory reduction (no subgroups). -/
def fusedRMSNormBitLinearResidualM1KernelSharedMem (config : Config) (eps : Float := 1e-5) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid

  let totalWeightElements := config.outDim * config.inDim
  let numPackedBytes := totalWeightElements / 4
  let numPackedU32 := (numPackedBytes + 3) / 4

  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) config.inDim)
  let _rmsScale ← ShaderM.declareInputBuffer "rms_scale" (.array (.scalar .f32) config.inDim)
  let _packed ← ShaderM.declareInputBuffer "weights_packed" (.array (.scalar .u32) numPackedU32)
  let _blScale ← ShaderM.declareInputBuffer "bl_scale" (.array (.scalar .f32) 1)
  let _residual ← ShaderM.declareInputBuffer "residual" (.array (.scalar .f32) config.outDim)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) config.outDim)

  -- Phase 1: Compute RMS via shared-memory reduction
  ShaderM.varNamed "partial_sq" (.scalar .f32) (Exp.litF32 0.0)
  let partialSq : Exp (.scalar .f32) := Exp.var "partial_sq"

  ShaderM.loop tid (Exp.litU32 config.inDim) (Exp.litU32 32) fun elemIdx => do
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.inDim) "input" elemIdx
    ShaderM.assign "partial_sq" (Exp.add partialSq (Exp.mul val val))

  let totalSq ← sharedMemReduce32 tid partialSq "shared_rms"
  -- Store totalSq in a var so all threads can use it after barrier
  ShaderM.varNamed "totalSq_val" (.scalar .f32) totalSq

  -- Broadcast totalSq to all threads via shared memory (thread 0 already has it in shared[0])
  -- After the reduction, shared_rms[0] contains the result, which all threads can read
  let totalSqBroadcast ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 32) "shared_rms" (Exp.litU32 0)
  let mean := Exp.div totalSqBroadcast (Exp.litF32 config.inDim.toFloat)
  let rmsInv := Exp.inverseSqrt (Exp.add mean (Exp.litF32 eps))

  -- Phase 2: BitLinear dot product with inline normalization
  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"

  let u32PerRow := config.inDim / 16
  let rowBaseU32 := Exp.mul outIdx (Exp.litU32 u32PerRow)

  ShaderM.loop tid (Exp.litU32 u32PerRow) (Exp.litU32 32) fun u32Idx => do
    let packedU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := numPackedU32) "weights_packed"
      (Exp.add rowBaseU32 u32Idx)
    let localGroup := Exp.div u32Idx (Exp.litU32 8)
    let localGroupPos := Exp.mul (Exp.mod u32Idx (Exp.litU32 8)) (Exp.litU32 4)
    let elemBase := Exp.add (Exp.mul localGroup (Exp.litU32 128)) localGroupPos
    for b in [0:4] do
      let theByte := Exp.bitAnd (Exp.shiftRight packedU32 (Exp.litU32 (b * 8))) (Exp.litU32 0xFF)
      for s in [0:4] do
        let code := Exp.bitAnd (Exp.shiftRight theByte (Exp.litU32 (6 - s * 2))) (Exp.litU32 0x3)
        let ternaryF32 := Exp.sub (Exp.toF32 code) (Exp.litF32 1.0)
        let elemIdx := Exp.add elemBase (Exp.litU32 (b + s * 32))
        let inputVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.inDim) "input" elemIdx
        let scaleVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.inDim) "rms_scale" elemIdx
        let normalized := Exp.mul (Exp.mul inputVal rmsInv) scaleVal
        ShaderM.assign "acc" (Exp.add acc (Exp.mul ternaryF32 normalized))

  -- Second shared-memory reduction for dot product
  let totalSum ← sharedMemReduce32 tid acc "shared_dot"

  let inBounds := Exp.lt outIdx (Exp.litU32 config.outDim)
  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    let blScaleVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := 1) "bl_scale" (Exp.litU32 0)
    let residualVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.outDim) "residual" outIdx
    let result := Exp.add residualVal (Exp.mul blScaleVal totalSum)
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx result
  ) (pure ())

/-- Fused gate+up+ReLU²×mul M=1 kernel using shared-memory reduction (no subgroups). -/
def fusedGateUpReluSqrMulM1KernelSharedMem (config : Config) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid

  let totalWeightElements := config.outDim * config.inDim
  let numPackedBytes := totalWeightElements / 4
  let numPackedU32 := (numPackedBytes + 3) / 4

  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) config.inDim)
  let _gatePacked ← ShaderM.declareInputBuffer "gate_packed" (.array (.scalar .u32) numPackedU32)
  let _gateScale ← ShaderM.declareInputBuffer "gate_scale" (.array (.scalar .f32) 1)
  let _upPacked ← ShaderM.declareInputBuffer "up_packed" (.array (.scalar .u32) numPackedU32)
  let _upScale ← ShaderM.declareInputBuffer "up_scale" (.array (.scalar .f32) 1)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) config.outDim)

  ShaderM.varNamed "gate_acc" (.scalar .f32) (Exp.litF32 0.0)
  ShaderM.varNamed "up_acc" (.scalar .f32) (Exp.litF32 0.0)
  let gateAcc : Exp (.scalar .f32) := Exp.var "gate_acc"
  let upAcc : Exp (.scalar .f32) := Exp.var "up_acc"

  let u32PerRow := config.inDim / 16
  let rowBaseU32 := Exp.mul outIdx (Exp.litU32 u32PerRow)

  ShaderM.loop tid (Exp.litU32 u32PerRow) (Exp.litU32 32) fun u32Idx => do
    let gatePackedU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := numPackedU32) "gate_packed"
      (Exp.add rowBaseU32 u32Idx)
    let upPackedU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := numPackedU32) "up_packed"
      (Exp.add rowBaseU32 u32Idx)
    let localGroup := Exp.div u32Idx (Exp.litU32 8)
    let localGroupPos := Exp.mul (Exp.mod u32Idx (Exp.litU32 8)) (Exp.litU32 4)
    let elemBase := Exp.add (Exp.mul localGroup (Exp.litU32 128)) localGroupPos
    for b in [0:4] do
      let gateByte := Exp.bitAnd (Exp.shiftRight gatePackedU32 (Exp.litU32 (b * 8))) (Exp.litU32 0xFF)
      let upByte := Exp.bitAnd (Exp.shiftRight upPackedU32 (Exp.litU32 (b * 8))) (Exp.litU32 0xFF)
      for s in [0:4] do
        let gateCode := Exp.bitAnd (Exp.shiftRight gateByte (Exp.litU32 (6 - s * 2))) (Exp.litU32 0x3)
        let gateTernary := Exp.sub (Exp.toF32 gateCode) (Exp.litF32 1.0)
        let upCode := Exp.bitAnd (Exp.shiftRight upByte (Exp.litU32 (6 - s * 2))) (Exp.litU32 0x3)
        let upTernary := Exp.sub (Exp.toF32 upCode) (Exp.litF32 1.0)
        let elemIdx := Exp.add elemBase (Exp.litU32 (b + s * 32))
        let inputVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.inDim) "input" elemIdx
        ShaderM.assign "gate_acc" (Exp.add gateAcc (Exp.mul gateTernary inputVal))
        ShaderM.assign "up_acc" (Exp.add upAcc (Exp.mul upTernary inputVal))

  -- Two separate shared-memory reductions (gate and up)
  let gateTotal ← sharedMemReduce32 tid gateAcc "shared_gate"
  let upTotal ← sharedMemReduce32 tid upAcc "shared_up"

  let inBounds := Exp.lt outIdx (Exp.litU32 config.outDim)
  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    let gateScaleVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := 1) "gate_scale" (Exp.litU32 0)
    let upScaleVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := 1) "up_scale" (Exp.litU32 0)
    let gateScaled := Exp.mul gateScaleVal gateTotal
    let upScaled := Exp.mul upScaleVal upTotal
    let reluVal := Exp.max gateScaled (Exp.litF32 0.0)
    let result := Exp.mul (Exp.mul reluVal reluVal) upScaled
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx result
  ) (pure ())

/-! ## Fused BitLinear + Residual Add Kernel -/

/-- Fused kernel: i2_s unpack + matrix-vector multiply + residual add

    Same as fusedBitLinearKernel but outputs:
      output[i] = residual[i] + scale * dot_product

    Eliminates a separate elementwise add dispatch per layer.
    Used for attention O-projection and FFN down-projection residual connections.

    @param config BitLinear layer configuration
    @param numRows Number of input rows
    @param workgroupSize Threads per workgroup (default 256)
-/
def fusedBitLinearResidualKernel (config : Config) (numRows : Nat) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let flatWgId := Exp.vec3X wid
  let tid := Exp.vec3X lid

  let totalOutputs := numRows * config.outDim
  let outIdx := Exp.mod flatWgId (Exp.litU32 config.outDim)
  let rowIdx := Exp.div flatWgId (Exp.litU32 config.outDim)
  let inBounds := Exp.lt flatWgId (Exp.litU32 totalOutputs)

  let totalWeightElements := config.outDim * config.inDim
  let numPackedBytes := totalWeightElements / 4
  let numPackedU32 := (numPackedBytes + 3) / 4
  let totalInputElements := numRows * config.inDim

  let maxSharedInputElems := 3584
  let tileElemSize :=
    if config.inDim ≤ maxSharedInputElems then config.inDim
    else (maxSharedInputElems / 128) * 128
  let numTiles := (config.inDim + tileElemSize - 1) / tileElemSize

  ShaderM.sharedNamed "shared_input" (.array (.scalar .f32) tileElemSize)
  ShaderM.sharedNamed "shared_partial" (.array (.scalar .f32) workgroupSize)

  let _packed ← ShaderM.declareInputBuffer "weights_packed" (.array (.scalar .u32) numPackedU32)
  let _scale ← ShaderM.declareInputBuffer "scale" (.array (.scalar .f32) 1)
  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) totalInputElements)
  let _residual ← ShaderM.declareInputBuffer "residual" (.array (.scalar .f32) totalOutputs)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) totalOutputs)

  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"

  let u32PerRow := config.inDim / 16
  let rowBaseU32 := Exp.mul outIdx (Exp.litU32 u32PerRow)
  let inputRowBase := Exp.mul rowIdx (Exp.litU32 config.inDim)

  for t in [0:numTiles] do
    let tileStart := t * tileElemSize
    let tileEnd := min ((t + 1) * tileElemSize) config.inDim
    let actualTileSize := tileEnd - tileStart

    ShaderM.loop tid (Exp.litU32 actualTileSize) (Exp.litU32 workgroupSize) fun i => do
      let globalIdx := Exp.add inputRowBase (Exp.add (Exp.litU32 tileStart) i)
      let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalInputElements) "input" globalIdx
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_input" i val
    ShaderM.barrier

    let tileU32Start := tileStart / 16
    let tileU32Count := actualTileSize / 16

    ShaderM.loop tid (Exp.litU32 tileU32Count) (Exp.litU32 workgroupSize) fun localU32Idx => do
      let absU32Idx := Exp.add (Exp.litU32 tileU32Start) localU32Idx
      let packedU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := numPackedU32) "weights_packed" (Exp.add rowBaseU32 absU32Idx)
      let localGroup := Exp.div localU32Idx (Exp.litU32 8)
      let localGroupPos := Exp.mul (Exp.mod localU32Idx (Exp.litU32 8)) (Exp.litU32 4)
      let localElemBase := Exp.add (Exp.mul localGroup (Exp.litU32 128)) localGroupPos

      for b in [0:4] do
        let theByte := Exp.bitAnd (Exp.shiftRight packedU32 (Exp.litU32 (b * 8))) (Exp.litU32 0xFF)
        for s in [0:4] do
          let code := Exp.bitAnd (Exp.shiftRight theByte (Exp.litU32 (6 - s * 2))) (Exp.litU32 0x3)
          let ternaryF32 := Exp.sub (Exp.toF32 code) (Exp.litF32 1.0)
          let localElemIdx := Exp.add localElemBase (Exp.litU32 (b + s * 32))
          let inputVal ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := tileElemSize) "shared_input" localElemIdx
          ShaderM.assign "acc" (Exp.add acc (Exp.mul ternaryF32 inputVal))

    ShaderM.barrier

  ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_partial" tid acc
  ShaderM.barrier

  let mut stride := workgroupSize / 2
  while stride > 0 do
    ShaderM.if_ (Exp.lt tid (Exp.litU32 stride)) (do
      let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_partial" tid
      let b ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_partial" (Exp.add tid (Exp.litU32 stride))
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_partial" tid (Exp.add a b)
    ) (pure ())
    ShaderM.barrier
    stride := stride / 2

  -- Thread 0: output = residual + scale * sum
  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    let totalSum ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_partial" (Exp.litU32 0)
    let scaleVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := 1) "scale" (Exp.litU32 0)
    let outputIdx := Exp.add (Exp.mul rowIdx (Exp.litU32 config.outDim)) outIdx
    let residualVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalOutputs) "residual" outputIdx
    let result := Exp.add residualVal (Exp.mul scaleVal totalSum)
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outputIdx result
  ) (pure ())

/-! ## Subgroup-Matrix Multi-row BitLinear Kernel (cooperative matmul) -/

/-- Multi-row BitLinear kernel using cooperative matrix operations.

    Targets the NVIDIA Ada / Ampere WMMA config (f16, f16) → f32 at 16×16×16.
    One workgroup (= one subgroup of 32 threads) computes a single 16×16
    output tile `Y[rowBase .. rowBase+16, outBase .. outBase+16]`.

    The math is `Y = X @ W^T` where X is the row-major `[numRows, inDim]`
    input and W is the ternary weight matrix (`[outDim, inDim]`). So the
    subgroup matrix layout is:

      A (left,  M×K = 16×16) = X[rowBase .. +16, kBase .. +16]
      B (right, K×N = 16×16) = W^T[kBase .. +16, outBase .. +16]
                             = W[outBase .. +16, kBase .. +16]^T
      C (result, M×N = 16×16)

    A is f16 (cast from f32 input), B is f16 (dequantized from i2_s
    ternary), C accumulates as f32 for precision. Preconditions:
    `numRows % 16 == 0`, `outDim % 16 == 0`, `inDim % 16 == 0`
    (and `inDim % 128 == 0` for i2_s). The caller (`forward`) must check
    these and fall back to the existing tiled kernel otherwise.

    Layout of shared memory:
      shared_A : array<f16, 256>   row-major 16 × 16 (M × K)
      shared_B : array<f16, 256>   row-major 16 × 16 (K × N)

    The subgroup matrix load reads f16 tiles from these with stride=16,
    does one 16×16×16 cooperative MAC per K-block, then at the end of
    the K loop stores the f32 result into another shared buffer
    `shared_C` of 256 f32 values, after which all 32 threads scale and
    write 8 elements each to the output buffer. -/
def fusedBitLinearSubgroupMatrixKernel (config : Config) (numRows : Nat) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let wgX := Exp.vec3X wid   -- tile column (outDim direction)
  let wgY := Exp.vec3Y wid   -- tile row    (row direction)
  let tid := Exp.vec3X lid   -- lane within the subgroup (0..31)

  let outTiles := config.outDim / 16
  let _ := outTiles  -- silence unused when not used directly
  let numKTiles := config.inDim / 16

  let totalWeightElements := config.outDim * config.inDim
  let numPackedBytes := totalWeightElements / 4
  let numPackedU32 := (numPackedBytes + 3) / 4
  let totalInputElements := numRows * config.inDim
  let totalOutputElements := numRows * config.outDim

  -- Buffers
  let _packed ← ShaderM.declareInputBuffer "weights_packed" (.array (.scalar .u32) numPackedU32)
  let _scale ← ShaderM.declareInputBuffer "scale" (.array (.scalar .f32) 1)
  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) totalInputElements)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) totalOutputElements)

  -- Workgroup-private matrix tiles (16×16 each).
  ShaderM.sharedNamed "shared_A" (.array (.scalar .f16) 256)
  ShaderM.sharedNamed "shared_B" (.array (.scalar .f16) 256)
  ShaderM.sharedNamed "shared_C" (.array (.scalar .f32) 256)

  -- Declare the one-element subgroup matrix arrays.
  ShaderM.declareMatrixLeftArray  "Ax" .f16 16 16 1 Exp.subgroupMatrixZeroLeft
  ShaderM.declareMatrixRightArray "Bx" .f16 16 16 1 Exp.subgroupMatrixZeroRight
  ShaderM.declareMatrixResultArray "Cx" .f32 16 16 1 Exp.subgroupMatrixZeroResult

  let rowBase := Exp.mul wgY (Exp.litU32 16)
  let outBase := Exp.mul wgX (Exp.litU32 16)

  -- Each row of W covers inDim / 4 bytes in the packed buffer.
  let bytesPerRow := config.inDim / 4

  -- Thread layout: 32 threads, 256 elements per tile → 8 elements/thread.
  -- Use flat index e = tid + s*32 for s in 0..8.
  -- Rows past `numRows` are zero-padded so the workgroup grid can cover any
  -- numRows (not just multiples of 16). Out-of-bounds output rows are
  -- skipped in the store phase below. outDim still needs to be a multiple of
  -- 16 (enforced by the dispatch threshold in `forward`).
  let numRowsU32 := Exp.litU32 numRows
  let loadTileA (kBase : Exp (.scalar .u32)) : ShaderM Unit := do
    for s in [0:8] do
      let e := Exp.add tid (Exp.litU32 (s * 32))
      let mi := Exp.div e (Exp.litU32 16)
      let ki := Exp.mod e (Exp.litU32 16)
      let row := Exp.add rowBase mi
      let col := Exp.add kBase ki
      let inRange := Exp.lt row numRowsU32
      ShaderM.if_ inRange (do
        let inIdx := Exp.add (Exp.mul row (Exp.litU32 config.inDim)) col
        let xf32 ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalInputElements) "input" inIdx
        let xf16 := Exp.toF16 xf32
        ShaderM.writeWorkgroup (ty := .scalar .f16) "shared_A" e xf16
      ) (do
        ShaderM.writeWorkgroup (ty := .scalar .f16) "shared_A" e (Exp.toF16 (Exp.litF32 0.0)))

  -- Decode a single i2_s ternary value at (outRow, col) into a plain f32 in {-1, 0, 1}.
  -- This mirrors BitLinearSpec.decodeI2S exactly.
  let decodeTernary (outRow col : Exp (.scalar .u32)) : ShaderM (Exp (.scalar .f32)) := do
    let rowStart := Exp.mul outRow (Exp.litU32 bytesPerRow)
    let group128 := Exp.div col (Exp.litU32 128)
    let colInGroup := Exp.mod col (Exp.litU32 128)
    let bytePos := Exp.mod colInGroup (Exp.litU32 32)
    let shiftIdx := Exp.div colInGroup (Exp.litU32 32)
    -- shift = 6 - shiftIdx*2
    let shift := Exp.sub (Exp.litU32 6) (Exp.mul shiftIdx (Exp.litU32 2))
    let byteOffset := Exp.add rowStart (Exp.add (Exp.mul group128 (Exp.litU32 32)) bytePos)
    -- Weights are packed as u32. Read the u32 that contains this byte.
    let u32Idx := Exp.div byteOffset (Exp.litU32 4)
    let byteInU32 := Exp.mod byteOffset (Exp.litU32 4)
    let packedU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := numPackedU32) "weights_packed" u32Idx
    -- Extract the right byte: (packedU32 >> (byteInU32 * 8)) & 0xFF
    let byteShift := Exp.mul byteInU32 (Exp.litU32 8)
    let theByte := Exp.bitAnd (Exp.shiftRight packedU32 byteShift) (Exp.litU32 0xFF)
    -- Extract 2-bit code at (shift .. shift+2)
    let code := Exp.bitAnd (Exp.shiftRight theByte shift) (Exp.litU32 0x3)
    pure (Exp.sub (Exp.toF32 code) (Exp.litF32 1.0))

  let loadTileB (kBase : Exp (.scalar .u32)) : ShaderM Unit := do
    for s in [0:8] do
      let e := Exp.add tid (Exp.litU32 (s * 32))
      -- B is stored row-major as [K=16, N=16]; B[ki, ni] = W^T[kBase+ki, outBase+ni]
      -- = W[outBase+ni, kBase+ki]. So we decode at (row = outBase+ni, col = kBase+ki).
      let ki := Exp.div e (Exp.litU32 16)
      let ni := Exp.mod e (Exp.litU32 16)
      let col := Exp.add kBase ki
      let row := Exp.add outBase ni
      let w ← decodeTernary row col
      let wf16 := Exp.toF16 w
      ShaderM.writeWorkgroup (ty := .scalar .f16) "shared_B" e wf16

  -- Main K loop — march over inDim in steps of 16.
  for kTile in [0:numKTiles] do
    let kBase := Exp.litU32 (kTile * 16)
    loadTileA kBase
    loadTileB kBase
    ShaderM.barrier
    ShaderM.loadMatrixLeft  (st := .f16) (m := 16) (k := 16) "Ax" 0 "shared_A" (Exp.litU32 0) (Exp.litU32 16)
    ShaderM.loadMatrixRight (st := .f16) (k := 16) (n := 16) "Bx" 0 "shared_B" (Exp.litU32 0) (Exp.litU32 16)
    ShaderM.matrixMultiplyAccumulateMixed
      (inSt := .f16) (outSt := .f32) (m := 16) (k := 16) (n := 16)
      "Cx" 0 "Ax" 0 "Bx" 0
    ShaderM.barrier

  -- Store the f32 accumulator tile into workgroup memory, then scale + write out.
  ShaderM.storeMatrixResult (st := .f32) (m := 16) (n := 16)
    "Cx" 0 "shared_C" (Exp.litU32 0) (Exp.litU32 16)
  ShaderM.barrier

  -- 32 threads, 256 tile elements → 8 per thread. Skip writes for rows past
  -- `numRows` (workgroup grid may cover padding rows when numRows % 16 ≠ 0).
  let scaleVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := 1) "scale" (Exp.litU32 0)
  for s in [0:8] do
    let e := Exp.add tid (Exp.litU32 (s * 32))
    let mi := Exp.div e (Exp.litU32 16)
    let ni := Exp.mod e (Exp.litU32 16)
    let row := Exp.add rowBase mi
    let col := Exp.add outBase ni
    ShaderM.if_ (Exp.lt row numRowsU32) (do
      let v ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 256) "shared_C" e
      let outIdx := Exp.add (Exp.mul row (Exp.litU32 config.outDim)) col
      ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx (Exp.mul scaleVal v)
    ) (pure ())

/-! ## High-Level API -/

/-- BitLinear layer structure -/
structure BitLinear (BufT : Type) (CacheT : Type := Unit) (KernelT : Type := Unit) where
  config : Config
  weightsPacked : BufT   -- i2_s packed weights (raw bytes as u32 array)
  scaleBuf : BufT        -- Single f32 scale value
  prepared : IO.Ref (Option CacheT)  -- Graph capture cache
  kernel : KernelT  -- Pre-compiled M=1 kernel

/-- Create BitLinear layer from i2_s packed data

    @param device WebGPU device
    @param config Layer configuration
    @param packedWeights Raw i2_s packed byte data from GGUF
    @param scale Float32 scale factor for the ternary weights
-/
def create [GPUBackend β] (ctx : β) (config : Config)
           (packedWeights : ByteArray) (scale : Float) : IO (BitLinear (GPUBackend.Buf β) (GPUBackend.CachedDispatch β) (GPUBackend.CompiledKernel β)) := do
  logVerbose s!"[BitLinear] Creating layer: {config.inDim} -> {config.outDim}, scale={scale}"

  -- Pad packed weights to u32 alignment if needed
  let paddedWeights ← do
    if packedWeights.size % 4 == 0 then pure packedWeights
    else do
      let padding := 4 - (packedWeights.size % 4)
      let mut w := packedWeights
      for _ in [0:padding] do
        w := w.push 0
      pure w

  let bufSize := if paddedWeights.size == 0 then 4 else paddedWeights.size
  let weightsBuf ← GPUBackend.allocBuffer ctx bufSize.toUSize
  if paddedWeights.size > 0 then
    GPUBackend.writeBuffer ctx weightsBuf paddedWeights
  let scaleBuf ← GPUBackend.allocBuffer ctx 4
  let scaleBytes ← Hesper.Basic.floatToBytes scale
  GPUBackend.writeBuffer ctx scaleBuf scaleBytes
  let prepared ← GPUBackend.newCacheRef (β := β)
  let useSubgroups ← GPUBackend.hasSubgroupSupport ctx
  IO.println s!"[DBG BitLinear.create] useSubgroups={useSubgroups}"
  let shaderM := if useSubgroups
    then fusedBitLinearM1Kernel config
    else fusedBitLinearM1KernelSharedMem config
  let buildCfg : Hesper.ExecConfig := {
    workgroupSize := { x := 32 }, numWorkgroups := (config.outDim, 1, 1),
    extensions := if useSubgroups then ["subgroups"] else [],
    diagnostics := if useSubgroups then [("off", "chromium.subgroup_matrix_uniformity")] else []
  }
  IO.println s!"[DBG BitLinear.create] extensions={buildCfg.extensions}"
  let kernel ← GPUBackend.buildKernel ctx shaderM buildCfg
  logVerbose s!"[BitLinear] Layer created: packed={paddedWeights.size} bytes (subgroups={useSubgroups})"
  pure { config, weightsPacked := weightsBuf, scaleBuf, prepared, kernel }

/-- Create BitLinear layer from packed data + scale ByteArrays

    Alternative constructor that takes pre-encoded scale bytes.
    Used when the caller has already extracted the raw data.

    @param device WebGPU device
    @param config Layer configuration
    @param packedWeights Raw i2_s packed byte data
    @param scaleBytes 4-byte little-endian F32 scale
-/
def createFromBytes [GPUBackend β] (ctx : β) (config : Config)
                    (packedWeights : ByteArray) (scaleBytes : ByteArray) : IO (BitLinear (GPUBackend.Buf β) (GPUBackend.CachedDispatch β) (GPUBackend.CompiledKernel β)) := do
  logVerbose s!"[BitLinear] Creating layer from bytes: {config.inDim} -> {config.outDim}"

  -- Pad packed weights to u32 alignment if needed
  let paddedWeights ← do
    if packedWeights.size % 4 == 0 then pure packedWeights
    else do
      let padding := 4 - (packedWeights.size % 4)
      let mut w := packedWeights
      for _ in [0:padding] do
        w := w.push 0
      pure w

  let bufSize := if paddedWeights.size == 0 then 4 else paddedWeights.size
  let weightsBuf ← GPUBackend.allocBuffer ctx bufSize.toUSize
  if paddedWeights.size > 0 then
    GPUBackend.writeBuffer ctx weightsBuf paddedWeights
  let scaleBuf ← GPUBackend.allocBuffer ctx (scaleBytes.size.max 4).toUSize
  GPUBackend.writeBuffer ctx scaleBuf scaleBytes
  let prepared ← GPUBackend.newCacheRef (β := β)
  let useSubgroups ← GPUBackend.hasSubgroupSupport ctx
  let shaderM := if useSubgroups
    then fusedBitLinearM1Kernel config
    else fusedBitLinearM1KernelSharedMem config
  let kernel ← GPUBackend.buildKernel ctx shaderM {
    workgroupSize := { x := 32 }, numWorkgroups := (config.outDim, 1, 1),
    extensions := if useSubgroups then ["subgroups"] else [],
    diagnostics := if useSubgroups then [("off", "chromium.subgroup_matrix_uniformity")] else []
  }
  logVerbose s!"[BitLinear] Layer created: packed={paddedWeights.size} bytes (subgroups={useSubgroups})"
  pure { config, weightsPacked := weightsBuf, scaleBuf, prepared, kernel }

/-- Execute forward pass

    @param device WebGPU device
    @param layer BitLinear layer
    @param inputBuf GPU buffer containing input (Float32)
    @param outputBuf GPU buffer for output (Float32)
-/
def forward [GPUBackend β] (ctx : β) (layer : BitLinear (GPUBackend.Buf β) (GPUBackend.CachedDispatch β) (GPUBackend.CompiledKernel β))
            (inputBuf outputBuf : GPUBackend.Buf β) (numRows : Nat := 1) : IO Unit := do
  -- Fast path: replay prepared dispatch (skips ALL Lean processing)
  if numRows == 1 then
    if let some p ← layer.prepared.get then
      preparedHitsRef.modify (· + 1)
      GPUBackend.replayCached ctx p (layer.config.outDim, 1, 1)
      return

  preparedMissesRef.modify (· + 1)
  logVerbose s!"[BitLinear] Executing forward pass ({numRows} rows, {layer.config.inDim}→{layer.config.outDim})..."

  if numRows == 1 then
    -- M=1 path: use pre-compiled kernel, direct buffer binding (no string matching)
    GPUBackend.dispatchCompiledKernel ctx layer.kernel
      #[layer.weightsPacked, layer.scaleBuf, inputBuf, outputBuf]
      (layer.config.outDim, 1, 1) (some layer.prepared)
  else
    -- M>1 path. Prefer the subgroup-matrix cooperative matmul kernel when:
    --   (a) the device exposes SubgroupMatrix + ShaderF16,
    --   (b) the shape is tile-friendly (numRows, inDim, outDim all % 16 == 0),
    --   (c) inDim is also % 128 == 0 (i2_s group layout).
    -- Otherwise fall back to the workgroup-cooperative tiled kernel.
    let hasSM ← GPUBackend.hasSubgroupSupport ctx
    let hasF16 ← GPUBackend.hasShaderF16Support ctx
    -- The kernel zero-pads the tail row tile, so numRows ≥ 16 is the only
    -- row requirement. outDim must be a multiple of 16 (no column-tail
    -- padding yet) and inDim must be a multiple of 128 (i2_s group layout).
    -- Measured ~800-2600 GFLOP/s for M ∈ [16, 128] on RTX 4070 Ti vs
    -- the tiled fallback's ~620 GFLOP/s peak; enabled by default.
    let tileFriendly :=
      numRows ≥ 16 && layer.config.outDim % 16 == 0 &&
      layer.config.inDim % 16 == 0 && layer.config.inDim % 128 == 0
    let optIn ← subgroupMatrixOptInRef.get
    if hasSM && hasF16 && tileFriendly && optIn then
      let shader := fusedBitLinearSubgroupMatrixKernel layer.config numRows
      let namedBuffers := [
        ("weights_packed", layer.weightsPacked),
        ("scale", layer.scaleBuf),
        ("input", inputBuf),
        ("output", outputBuf)
      ]
      let numRowTiles := (numRows + 15) / 16
      let execConfig : Hesper.ExecConfig := {
        workgroupSize := { x := 32, y := 1, z := 1 }
        numWorkgroups := (layer.config.outDim / 16, numRowTiles, 1)
      }
      let cacheKey : UInt64 :=
        hash ("bl-sm", layer.config.inDim, layer.config.outDim, numRows)
      GPUBackend.execute ctx shader namedBuffers execConfig
    else
      -- Workgroup-cooperative tiled kernel (existing fallback)
      let wgSize := 256
      let shader := fusedBitLinearKernel layer.config numRows wgSize
      let namedBuffers := [
        ("weights_packed", layer.weightsPacked),
        ("scale", layer.scaleBuf),
        ("input", inputBuf),
        ("output", outputBuf)
      ]
      let totalOutputs := numRows * layer.config.outDim
      let execConfig : Hesper.ExecConfig := {
        workgroupSize := { x := wgSize, y := 1, z := 1 }
        numWorkgroups := (totalOutputs, 1, 1)
      }
      let cacheKey : UInt64 := hash ("bl", layer.config.inDim, layer.config.outDim, numRows)
      GPUBackend.execute ctx shader namedBuffers execConfig

  logVerbose "[BitLinear] Forward pass complete"

/-- Execute forward pass with fused residual add: output = residual + scale * (weights @ input)

    Saves one elementwise add dispatch per call (2 dispatches/layer × 30 layers = 60 saved).

    @param device WebGPU device
    @param layer BitLinear layer
    @param inputBuf GPU buffer containing input (Float32)
    @param residualBuf GPU buffer containing residual to add (Float32)
    @param outputBuf GPU buffer for output (Float32)
    @param numRows Number of input rows
-/
def forwardWithResidual [GPUBackend β] (ctx : β) (layer : BitLinear (GPUBackend.Buf β) (GPUBackend.CachedDispatch β) (GPUBackend.CompiledKernel β))
            (inputBuf residualBuf outputBuf : GPUBackend.Buf β) (numRows : Nat := 1) : IO Unit := do
  -- Fast path: replay prepared dispatch (skips ALL Lean processing)
  if numRows == 1 then
    if let some p ← layer.prepared.get then
      preparedHitsRef.modify (· + 1)
      GPUBackend.replayCached ctx p (layer.config.outDim, 1, 1)
      return

  preparedMissesRef.modify (· + 1)
  logVerbose s!"[BitLinear] Executing fused forward+residual ({numRows} rows, {layer.config.inDim}→{layer.config.outDim})..."

  let namedBuffers := [
    ("weights_packed", layer.weightsPacked),
    ("scale", layer.scaleBuf),
    ("input", inputBuf),
    ("residual", residualBuf),
    ("output", outputBuf)
  ]

  if numRows == 1 then
    -- M=1 path: 32 threads per output, fused residual add
    let useSubgroups ← GPUBackend.hasSubgroupSupport ctx
    let shader := if useSubgroups
      then fusedBitLinearResidualM1Kernel layer.config
      else fusedBitLinearResidualM1KernelSharedMem layer.config
    let execConfig : Hesper.ExecConfig := {
      workgroupSize := { x := 32, y := 1, z := 1 }
      numWorkgroups := (layer.config.outDim, 1, 1)
    }
    let cacheKey : UInt64 := hash ("blrm1", layer.config.inDim, layer.config.outDim, useSubgroups)
    GPUBackend.executeWithConfigCached ctx shader namedBuffers execConfig cacheKey layer.prepared
  else
    -- M>1 path: workgroup-cooperative tiled kernel with residual
    let wgSize := 256
    let shader := fusedBitLinearResidualKernel layer.config numRows wgSize
    let totalOutputs := numRows * layer.config.outDim
    let execConfig : Hesper.ExecConfig := {
      workgroupSize := { x := wgSize, y := 1, z := 1 }
      numWorkgroups := (totalOutputs, 1, 1)
    }
    let cacheKey : UInt64 := hash ("blr", layer.config.inDim, layer.config.outDim, numRows)
    GPUBackend.execute ctx shader namedBuffers execConfig

  logVerbose "[BitLinear] Fused forward+residual complete"

/-- Execute fused RMSNorm + BitLinear + Residual forward pass (M=1 only).

    Combines: RMSNorm(input) → BitLinear dot product → residual add
    into a single GPU dispatch.

    output = residual + blScale * (weights @ RMSNorm(input))

    @param device WebGPU device
    @param layer BitLinear layer (weights + scale)
    @param rmsNorm RMSNorm layer (scale parameters)
    @param inputBuf Input buffer [inDim]
    @param residualBuf Residual buffer [outDim]
    @param outputBuf Output buffer [outDim]
    @param preparedRef Optional PreparedDispatch ref for fast-path replay
-/
def forwardFusedRMSNormResidual [GPUBackend β] (ctx : β) (layer : BitLinear (GPUBackend.Buf β) (GPUBackend.CachedDispatch β) (GPUBackend.CompiledKernel β))
    (rmsNorm : Hesper.Layers.RMSNorm.RMSNorm (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (inputBuf residualBuf outputBuf : GPUBackend.Buf β)
    (preparedRef : Option (IO.Ref (Option (GPUBackend.CachedDispatch β))) := none)
    : IO Unit := do
  -- Fast path: replay prepared dispatch
  if let some ref := preparedRef then
    if let some p ← ref.get then
      preparedHitsRef.modify (· + 1)
      GPUBackend.replayCached ctx p (layer.config.outDim, 1, 1)
      return

  preparedMissesRef.modify (· + 1)
  logVerbose s!"[BitLinear] Executing fused RMSNorm+BitLinear+Residual ({layer.config.inDim}→{layer.config.outDim})..."

  let useSubgroups ← GPUBackend.hasSubgroupSupport ctx
  let shader := if useSubgroups
    then fusedRMSNormBitLinearResidualM1Kernel layer.config rmsNorm.config.eps
    else fusedRMSNormBitLinearResidualM1KernelSharedMem layer.config rmsNorm.config.eps
  let namedBuffers := [
    ("input", inputBuf),
    ("rms_scale", rmsNorm.scale),
    ("weights_packed", layer.weightsPacked),
    ("bl_scale", layer.scaleBuf),
    ("residual", residualBuf),
    ("output", outputBuf)
  ]
  let execConfig : Hesper.ExecConfig := {
    workgroupSize := { x := 32, y := 1, z := 1 }
    numWorkgroups := (layer.config.outDim, 1, 1)
  }
  let cacheKey : UInt64 := hash ("frnblrm1", layer.config.inDim, layer.config.outDim, useSubgroups)
  match preparedRef with
  | some ref => GPUBackend.executeWithConfigCached ctx shader namedBuffers execConfig cacheKey ref
  | none => GPUBackend.execute ctx shader namedBuffers execConfig

/-- Execute fused Gate+Up+ReLU²×Mul forward pass (M=1 only).

    Combines: gate = BitLinear(input), up = BitLinear(input), output = ReLU²(gate) × up
    into a single GPU dispatch.

    @param device WebGPU device
    @param gateLayer Gate BitLinear layer
    @param upLayer Up BitLinear layer
    @param inputBuf Input buffer [inDim]
    @param outputBuf Output buffer [outDim]
    @param preparedRef Optional PreparedDispatch ref for fast-path replay
-/
def forwardFusedGateUpReluSqrMul [GPUBackend β] (ctx : β) (gateLayer upLayer : BitLinear (GPUBackend.Buf β) (GPUBackend.CachedDispatch β) (GPUBackend.CompiledKernel β))
    (inputBuf outputBuf : GPUBackend.Buf β)
    (preparedRef : Option (IO.Ref (Option (GPUBackend.CachedDispatch β))) := none)
    : IO Unit := do
  -- Fast path: replay prepared dispatch
  if let some ref := preparedRef then
    if let some p ← ref.get then
      preparedHitsRef.modify (· + 1)
      GPUBackend.replayCached ctx p (gateLayer.config.outDim, 1, 1)
      return

  preparedMissesRef.modify (· + 1)
  logVerbose s!"[BitLinear] Executing fused Gate+Up+ReLU²×Mul ({gateLayer.config.inDim}→{gateLayer.config.outDim})..."

  let useSubgroups ← GPUBackend.hasSubgroupSupport ctx
  let shader := if useSubgroups
    then fusedGateUpReluSqrMulM1Kernel gateLayer.config
    else fusedGateUpReluSqrMulM1KernelSharedMem gateLayer.config
  let namedBuffers := [
    ("input", inputBuf),
    ("gate_packed", gateLayer.weightsPacked),
    ("gate_scale", gateLayer.scaleBuf),
    ("up_packed", upLayer.weightsPacked),
    ("up_scale", upLayer.scaleBuf),
    ("output", outputBuf)
  ]
  let execConfig : Hesper.ExecConfig := {
    workgroupSize := { x := 32, y := 1, z := 1 }
    numWorkgroups := (gateLayer.config.outDim, 1, 1)
  }
  let cacheKey : UInt64 := hash ("fgurelum1", gateLayer.config.inDim, gateLayer.config.outDim, useSubgroups)
  match preparedRef with
  | some ref => GPUBackend.executeWithConfigCached ctx shader namedBuffers execConfig cacheKey ref
  | none => GPUBackend.execute ctx shader namedBuffers execConfig

end Hesper.Layers.BitLinear
