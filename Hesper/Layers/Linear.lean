import Hesper.WGSL.Monad
import Hesper.WGSL.Execute
import Hesper.WGSL.Exp
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.Quantization.Q4_K_M
import Hesper.Quantization.Q6_K
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

/-! ## Profiling counters

Gated counters that a caller can enable via `profilingRef.set true` to
measure how much total wall-clock time `LinearLayer.forward` consumes.
Only meaningful when dispatches run in non-batch mode (each call
deviceWaits), so the caller should profile without `beginBatch`. Reset
`totalNanosRef` between measurements. -/
initialize profilingRef  : IO.Ref Bool  ← IO.mkRef false
initialize totalNanosRef : IO.Ref UInt64 ← IO.mkRef 0
initialize callCountRef  : IO.Ref Nat    ← IO.mkRef 0
/-- Per-shape cumulative time, keyed by (inDim, outDim). Array of
    (inDim, outDim, cumNanos, callCount). Small N so linear scan is fine. -/
initialize perShapeRef : IO.Ref (Array (Nat × Nat × UInt64 × Nat)) ← IO.mkRef #[]

private def perShapeAdd (inDim outDim : Nat) (deltaNs : UInt64) : IO Unit := do
  let arr ← perShapeRef.get
  let mut found := false
  let mut out : Array (Nat × Nat × UInt64 × Nat) := Array.empty
  for entry in arr do
    let (i, o, ns, cnt) := entry
    if i == inDim && o == outDim then
      out := out.push (i, o, ns + deltaNs, cnt + 1)
      found := true
    else
      out := out.push entry
  if !found then
    out := out.push (inDim, outDim, deltaNs, 1)
  perShapeRef.set out

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

/-! ## Q4_K_M MatVec with Subgroup Reduction -/

/-- Subgroup-reduction variant of `fusedQ4KMLinearKernel`.

    Same math as the tree-reduction version, but uses 32 threads per
    workgroup and a single `subgroupAdd` at the end instead of a
    256-thread shared-memory tree. On adapters with subgroup support
    this eliminates all `workgroupBarrier` calls in the reduction phase
    and runs with an 8× smaller workgroup → more workgroups in flight
    → higher occupancy.

    Mirrors the pattern in `BitLinear.fusedBitLinearM1Kernel` — one
    subgroup per output element, strided block processing, hardware
    subgroupAdd across 32 lanes.

    @param config Layer dimensions
-/
def fusedQ4KMLinearSubgroupKernel (config : Config) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid

  let blocksPerRow := config.inDim / 256
  let totalWeightU32 := config.outDim * blocksPerRow * 36

  let _weights ← ShaderM.declareInputBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) config.inDim)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) config.outDim)

  let inBounds := Exp.lt outIdx (Exp.litU32 config.outDim)

  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"

  let rowBaseU32 := Exp.mul outIdx (Exp.litU32 (blocksPerRow * 36))

  -- Strided loop: 32 threads, stride = 32. Thread k handles blocks
  -- k, k+32, k+64, ... (consecutive threads hit consecutive blocks for
  -- coalesced weight reads).
  ShaderM.loop tid (Exp.litU32 blocksPerRow) (Exp.litU32 32) fun blockLocalIdx => do
    let blockU32Base := Exp.add rowBaseU32 (Exp.mul blockLocalIdx (Exp.litU32 36))
    let elemBase := Exp.mul blockLocalIdx (Exp.litU32 256)

    -- Read block header: d and dmin (first u32)
    let dmU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" blockU32Base
    let d := fp16ToF32 (Exp.bitAnd dmU32 (Exp.litU32 0xFFFF))
    let dmin := fp16ToF32 (Exp.shiftRight dmU32 (Exp.litU32 16))

    let sc0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add blockU32Base (Exp.litU32 1))
    let sc1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add blockU32Base (Exp.litU32 2))
    let sc2 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add blockU32Base (Exp.litU32 3))

    for c in [0:4] do
      let is0 := c * 2
      let is1 := c * 2 + 1
      let (scaleA, minA) := getScaleMin is0 sc0 sc1 sc2
      let (scaleB, minB) := getScaleMin is1 sc0 sc1 sc2
      let d1 := Exp.mul d scaleA
      let m1 := Exp.mul dmin minA
      let d2 := Exp.mul d scaleB
      let m2 := Exp.mul dmin minB

      let qsU32Base := Exp.add blockU32Base (Exp.litU32 (4 + c * 8))

      for l32 in [0:8] do
        let qsU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add qsU32Base (Exp.litU32 l32))
        for b in [0:4] do
          let byte := Exp.bitAnd (Exp.shiftRight qsU32 (Exp.litU32 (b * 8))) (Exp.litU32 0xFF)
          let qLow := Exp.bitAnd byte (Exp.litU32 0xF)
          let qHigh := Exp.shiftRight byte (Exp.litU32 4)
          let elemIdxLow := Exp.add elemBase (Exp.litU32 (c * 64 + l32 * 4 + b))
          let elemIdxHigh := Exp.add elemBase (Exp.litU32 (c * 64 + 32 + l32 * 4 + b))
          let inLow ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.inDim) "input" elemIdxLow
          let inHigh ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.inDim) "input" elemIdxHigh
          let wLow := Exp.sub (Exp.mul d1 (Exp.toF32 qLow)) m1
          let wHigh := Exp.sub (Exp.mul d2 (Exp.toF32 qHigh)) m2
          ShaderM.assign "acc" (Exp.add acc (Exp.add (Exp.mul wLow inLow) (Exp.mul wHigh inHigh)))

  -- Subgroup reduction: one hardware-accelerated sum across 32 lanes.
  ShaderM.varNamed "totalSum" (.scalar .f32) (Exp.subgroupAdd acc)
  let totalSum : Exp (.scalar .f32) := Exp.var "totalSum"

  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx totalSum
  ) (pure ())

/-! ## Q4_K_M MatVec, 2 Rows per Workgroup (Gemma 4 decode hot path) -/

/-- DEAD CODE — kept for documentation / benchmarking only.

    Two-rows-per-workgroup variant of `fusedQ4KMLinearSubgroupKernel`.
    Intended to mirror llama.cpp's `NUM_ROWS=2` Q4_K mat_vec kernel
    (`ggml-vulkan/vulkan-shaders/mul_mat_vec_q4_k.comp`) by computing
    two adjacent output rows per workgroup so each loaded `x` element
    is consumed twice (once per row) from registers.

    Result on RTX 4070 Ti + Tint/Dawn: **1.5× slower per call** than
    the 1-row kernel (0.77 ms vs 0.50 ms for 2560×10240). Total TPS
    dropped from 6.5 → 4.5. Hypothesis: the per-thread register pressure
    doubled (two accumulators + two sets of row-specific scales, mins,
    qs reads) and Tint's WGSL→SPIR-V path appears not to manage the
    doubled live range well, causing spills. Halving the workgroup
    count (`outDim/2` instead of `outDim`) also reduces parallelism on
    small shapes (`outDim=256` → only 128 WGs, starving the GPU).

    Kept in-tree so the next iteration can try a different split: e.g.
    two subgroups of 32 in a 64-thread workgroup, one subgroup per row,
    which should trade register pressure for L1-coincidence on the `x`
    reads. Not wired into `LinearLayer.forward`.
    @param config Layer dimensions (outDim must be even) -/
def fusedQ4KMLinear2RowSubgroupKernel (config : Config) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let pairIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid

  let blocksPerRow := config.inDim / 256
  let rowStrideU32 := blocksPerRow * 36
  let totalWeightU32 := config.outDim * rowStrideU32

  let _weights ← ShaderM.declareInputBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) config.inDim)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) config.outDim)

  -- Two output rows: row0 = 2*pairIdx, row1 = row0 + 1.
  let row0 := Exp.mul pairIdx (Exp.litU32 2)
  let row1 := Exp.add row0 (Exp.litU32 1)
  let inBounds := Exp.lt row1 (Exp.litU32 config.outDim)

  ShaderM.varNamed "acc0" (.scalar .f32) (Exp.litF32 0.0)
  ShaderM.varNamed "acc1" (.scalar .f32) (Exp.litF32 0.0)
  let acc0 : Exp (.scalar .f32) := Exp.var "acc0"
  let acc1 : Exp (.scalar .f32) := Exp.var "acc1"

  let row0BaseU32 := Exp.mul row0 (Exp.litU32 rowStrideU32)
  let row1BaseU32 := Exp.mul row1 (Exp.litU32 rowStrideU32)

  -- Inner processing of one block for one row. Only the acc and the row
  -- base change; the x reads are done outside and passed as references
  -- to the already-loaded input values via `elemBase` so they can be
  -- reused across rows. But since WGSL doesn't let us pass f32 ↦ f32
  -- arrays cleanly through Lean, we manually hoist the `x` reads.
  ShaderM.loop tid (Exp.litU32 blocksPerRow) (Exp.litU32 32) fun blockLocalIdx => do
    let blockOffsetU32 := Exp.mul blockLocalIdx (Exp.litU32 36)
    let elemBase := Exp.mul blockLocalIdx (Exp.litU32 256)

    -- Row0 block header
    let b0 := Exp.add row0BaseU32 blockOffsetU32
    let dm0U32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" b0
    let d0 := fp16ToF32 (Exp.bitAnd dm0U32 (Exp.litU32 0xFFFF))
    let dmin0 := fp16ToF32 (Exp.shiftRight dm0U32 (Exp.litU32 16))
    let sc0_0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add b0 (Exp.litU32 1))
    let sc0_1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add b0 (Exp.litU32 2))
    let sc0_2 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add b0 (Exp.litU32 3))

    -- Row1 block header
    let b1 := Exp.add row1BaseU32 blockOffsetU32
    let dm1U32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" b1
    let d1 := fp16ToF32 (Exp.bitAnd dm1U32 (Exp.litU32 0xFFFF))
    let dmin1 := fp16ToF32 (Exp.shiftRight dm1U32 (Exp.litU32 16))
    let sc1_0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add b1 (Exp.litU32 1))
    let sc1_1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add b1 (Exp.litU32 2))
    let sc1_2 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add b1 (Exp.litU32 3))

    for c in [0:4] do
      let is0 := c * 2
      let is1 := c * 2 + 1
      -- Row0 per-sub-block scales
      let (sA0, mA0) := getScaleMin is0 sc0_0 sc0_1 sc0_2
      let (sB0, mB0) := getScaleMin is1 sc0_0 sc0_1 sc0_2
      let dA0 := Exp.mul d0 sA0
      let mA0' := Exp.mul dmin0 mA0
      let dB0 := Exp.mul d0 sB0
      let mB0' := Exp.mul dmin0 mB0
      -- Row1 per-sub-block scales
      let (sA1, mA1) := getScaleMin is0 sc1_0 sc1_1 sc1_2
      let (sB1, mB1) := getScaleMin is1 sc1_0 sc1_1 sc1_2
      let dA1 := Exp.mul d1 sA1
      let mA1' := Exp.mul dmin1 mA1
      let dB1 := Exp.mul d1 sB1
      let mB1' := Exp.mul dmin1 mB1

      let qs0Base := Exp.add b0 (Exp.litU32 (4 + c * 8))
      let qs1Base := Exp.add b1 (Exp.litU32 (4 + c * 8))

      for l32 in [0:8] do
        let qs0U32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add qs0Base (Exp.litU32 l32))
        let qs1U32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add qs1Base (Exp.litU32 l32))
        for b in [0:4] do
          -- Unpack the (b-th) byte from each row's qs u32
          let byte0 := Exp.bitAnd (Exp.shiftRight qs0U32 (Exp.litU32 (b * 8))) (Exp.litU32 0xFF)
          let qLow0  := Exp.bitAnd byte0 (Exp.litU32 0xF)
          let qHigh0 := Exp.shiftRight byte0 (Exp.litU32 4)
          let byte1 := Exp.bitAnd (Exp.shiftRight qs1U32 (Exp.litU32 (b * 8))) (Exp.litU32 0xFF)
          let qLow1  := Exp.bitAnd byte1 (Exp.litU32 0xF)
          let qHigh1 := Exp.shiftRight byte1 (Exp.litU32 4)

          let elemIdxLow  := Exp.add elemBase (Exp.litU32 (c * 64 + l32 * 4 + b))
          let elemIdxHigh := Exp.add elemBase (Exp.litU32 (c * 64 + 32 + l32 * 4 + b))

          -- Input reads — done ONCE per block element, used for both rows.
          let inLow  ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.inDim) "input" elemIdxLow
          let inHigh ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.inDim) "input" elemIdxHigh

          -- Row0 dequant + FMA
          let w0Low  := Exp.sub (Exp.mul dA0 (Exp.toF32 qLow0))  mA0'
          let w0High := Exp.sub (Exp.mul dB0 (Exp.toF32 qHigh0)) mB0'
          ShaderM.assign "acc0" (Exp.add acc0 (Exp.add (Exp.mul w0Low inLow) (Exp.mul w0High inHigh)))

          -- Row1 dequant + FMA (reuses inLow / inHigh from registers)
          let w1Low  := Exp.sub (Exp.mul dA1 (Exp.toF32 qLow1))  mA1'
          let w1High := Exp.sub (Exp.mul dB1 (Exp.toF32 qHigh1)) mB1'
          ShaderM.assign "acc1" (Exp.add acc1 (Exp.add (Exp.mul w1Low inLow) (Exp.mul w1High inHigh)))

  -- Two subgroup reductions — one per row.
  ShaderM.varNamed "sum0" (.scalar .f32) (Exp.subgroupAdd acc0)
  ShaderM.varNamed "sum1" (.scalar .f32) (Exp.subgroupAdd acc1)
  let sum0 : Exp (.scalar .f32) := Exp.var "sum0"
  let sum1 : Exp (.scalar .f32) := Exp.var "sum1"

  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    ShaderM.writeBuffer (ty := .scalar .f32) "output" row0 sum0
    ShaderM.writeBuffer (ty := .scalar .f32) "output" row1 sum1
  ) (pure ())

/-! ## Q4_K_M Fused Gate+Up (Gemma 4 / LLaMA GeGLU FFN) -/

/-- Fused gate+up Q4_K_M kernel:
    `h[i] = GELU_tanh(x · W_gate[i]) * (x · W_up[i])`.

    Replaces three separate dispatches (gate matmul, up matmul,
    gelu-mul) with one. Both W_gate and W_up are Q4_K_M linears with
    the same (inDim, outDim) shape — Gemma 4's FFN layout — and both
    share the same input vector `x`. The fusion reuses the input loads
    between the two dot products, halves the dispatch overhead, and
    folds the element-wise GELU*mul step into the per-output
    computation (no separate kernel, no intermediate gate/up buffers).

    Constraint: both weight buffers must have the exact same
    (inDim, outDim); this is enforced at dispatch time by the caller
    (`forwardFusedGateUp` in `Hesper.Models.Gemma4`). The `config`
    argument gives those dims.

    @param config Layer dimensions (inDim, outDim must match both weight matrices)
-/
def fusedQ4KMGateUpSubgroupKernel (config : Config) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid

  let blocksPerRow := config.inDim / 256
  let totalWeightU32 := config.outDim * blocksPerRow * 36

  -- Two weight buffers with identical shape.
  let _weightsGate ← ShaderM.declareInputBuffer "weights_gate" (.array (.scalar .u32) totalWeightU32)
  let _weightsUp   ← ShaderM.declareInputBuffer "weights_up"   (.array (.scalar .u32) totalWeightU32)
  let _input       ← ShaderM.declareInputBuffer "input"        (.array (.scalar .f32) config.inDim)
  let _output      ← ShaderM.declareOutputBuffer "output"      (.array (.scalar .f32) config.outDim)

  let inBounds := Exp.lt outIdx (Exp.litU32 config.outDim)

  -- Two accumulators — one per dot product.
  ShaderM.varNamed "accG" (.scalar .f32) (Exp.litF32 0.0)
  ShaderM.varNamed "accU" (.scalar .f32) (Exp.litF32 0.0)
  let accG : Exp (.scalar .f32) := Exp.var "accG"
  let accU : Exp (.scalar .f32) := Exp.var "accU"

  let rowBaseU32 := Exp.mul outIdx (Exp.litU32 (blocksPerRow * 36))

  -- Decode one Q4_K_M block from `bufName` at `blockU32Base`, computing the
  -- dot-product contribution against the input elements starting at
  -- `elemBase`, and adding the result to `accName`. Input reads are
  -- performed in parallel for both weight buffers so they share memory
  -- bandwidth (the GPU L1/L2 caches handle the actual sharing).
  let processBlock (bufName : String) (accName : String) (acc : Exp (.scalar .f32))
      (blockU32Base : Exp (.scalar .u32)) (elemBase : Exp (.scalar .u32)) : ShaderM Unit := do
    -- Block header: d and dmin (packed as fp16 halves in u32[0]).
    let dmU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) bufName blockU32Base
    let d := fp16ToF32 (Exp.bitAnd dmU32 (Exp.litU32 0xFFFF))
    let dmin := fp16ToF32 (Exp.shiftRight dmU32 (Exp.litU32 16))
    let sc0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) bufName (Exp.add blockU32Base (Exp.litU32 1))
    let sc1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) bufName (Exp.add blockU32Base (Exp.litU32 2))
    let sc2 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) bufName (Exp.add blockU32Base (Exp.litU32 3))

    for c in [0:4] do
      let is0 := c * 2
      let is1 := c * 2 + 1
      let (scaleA, minA) := getScaleMin is0 sc0 sc1 sc2
      let (scaleB, minB) := getScaleMin is1 sc0 sc1 sc2
      let d1 := Exp.mul d scaleA
      let m1 := Exp.mul dmin minA
      let d2 := Exp.mul d scaleB
      let m2 := Exp.mul dmin minB

      let qsU32Base := Exp.add blockU32Base (Exp.litU32 (4 + c * 8))

      for l32 in [0:8] do
        let qsU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) bufName (Exp.add qsU32Base (Exp.litU32 l32))
        for b in [0:4] do
          let byte := Exp.bitAnd (Exp.shiftRight qsU32 (Exp.litU32 (b * 8))) (Exp.litU32 0xFF)
          let qLow := Exp.bitAnd byte (Exp.litU32 0xF)
          let qHigh := Exp.shiftRight byte (Exp.litU32 4)
          let elemIdxLow := Exp.add elemBase (Exp.litU32 (c * 64 + l32 * 4 + b))
          let elemIdxHigh := Exp.add elemBase (Exp.litU32 (c * 64 + 32 + l32 * 4 + b))
          let inLow ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.inDim) "input" elemIdxLow
          let inHigh ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.inDim) "input" elemIdxHigh
          let wLow := Exp.sub (Exp.mul d1 (Exp.toF32 qLow)) m1
          let wHigh := Exp.sub (Exp.mul d2 (Exp.toF32 qHigh)) m2
          ShaderM.assign accName (Exp.add acc (Exp.add (Exp.mul wLow inLow) (Exp.mul wHigh inHigh)))

  -- Strided loop over blocks. Each iteration processes one Q4_K block
  -- from W_gate and one from W_up for the SAME (outIdx, blockLocalIdx)
  -- pair. We do them in TWO SEPARATE inner loops so each loop keeps the
  -- register footprint of a single matmul (about half what an
  -- interleaved version would need). The gate-loop populates L1/L2
  -- with the input block, so the up-loop's input reads hit the cache.
  --
  -- NOTE: an earlier version interleaved the two `processBlock` calls
  -- inside a single loop iteration. On NVIDIA Vulkan that cost 2.8 ms
  -- per call vs 0.5 ms for a single matmul (5.6× / call, i.e. ~3×
  -- slower than the separate matmul baseline) because the interleaved
  -- form doubled live register usage and forced register spills.
  -- Keeping the two passes separate recovers the single-matmul
  -- throughput while still saving a dispatch.
  ShaderM.loop tid (Exp.litU32 blocksPerRow) (Exp.litU32 32) fun blockLocalIdx => do
    let blockU32Base := Exp.add rowBaseU32 (Exp.mul blockLocalIdx (Exp.litU32 36))
    let elemBase := Exp.mul blockLocalIdx (Exp.litU32 256)
    processBlock "weights_gate" "accG" accG blockU32Base elemBase
  ShaderM.loop tid (Exp.litU32 blocksPerRow) (Exp.litU32 32) fun blockLocalIdx => do
    let blockU32Base := Exp.add rowBaseU32 (Exp.mul blockLocalIdx (Exp.litU32 36))
    let elemBase := Exp.mul blockLocalIdx (Exp.litU32 256)
    processBlock "weights_up"   "accU" accU blockU32Base elemBase

  -- Subgroup reductions — one per accumulator. Each subgroupAdd is
  -- hardware-accelerated across 32 lanes.
  ShaderM.varNamed "gateSum" (.scalar .f32) (Exp.subgroupAdd accG)
  ShaderM.varNamed "upSum"   (.scalar .f32) (Exp.subgroupAdd accU)
  let gateSum : Exp (.scalar .f32) := Exp.var "gateSum"
  let upSum   : Exp (.scalar .f32) := Exp.var "upSum"

  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    -- GELU(tanh) — llama.cpp's approximation, matches `geluMulKernel`
    -- and `FusedFFNSpec.geluTanh`.
    let sqrt2OverPi := Exp.litF32 0.7978845608028654
    let z := gateSum
    let z3 := Exp.mul (Exp.mul z z) z
    let inner := Exp.mul sqrt2OverPi (Exp.add z (Exp.mul (Exp.litF32 0.044715) z3))
    let gelu := Exp.mul (Exp.mul (Exp.litF32 0.5) z) (Exp.add (Exp.litF32 1.0) (Exp.tanh inner))
    let result := Exp.mul gelu upSum
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx result
  ) (pure ())

/-! ## Layer Structure -/

/-- Quantization format for the weight tensor -/
inductive QuantFormat where
  | Q4_K   -- Q4_K_M: 144 bytes per 256 elements
  | Q6_K   -- Q6_K: 210 bytes per 256 elements
  deriving Repr, BEq, Inhabited

/-- Quantized linear layer (supports Q4_K and Q6_K) -/
structure LinearLayer where
  config : Config
  weightBuf : Buffer    -- Raw packed weights on GPU
  quantFormat : QuantFormat  -- Which dequant kernel to use
  prepared : IO.Ref (Option Execute.PreparedDispatch)

/-- Execute the linear layer: output = input @ weights^T

    Fast path: after the first call, the prepared dispatch is cached in
    `layer.prepared` and subsequent calls bypass WGSL regeneration / hash
    lookup entirely via `replayPreparedDispatch`. This is critical for
    Gemma 4 where Q4_K/Q6_K linears are called dozens of times per layer
    × 42 layers per token — the non-fast path would pay a ~tens-of-ms
    WGSL-emit-and-hash cost per call for these large unrolled kernels.

    @param device WebGPU device
    @param layer The linear layer
    @param inputBuf GPU buffer with input vector [inDim]
    @param outputBuf GPU buffer for output vector [outDim]
-/
def LinearLayer.forward (device : Device) (layer : LinearLayer)
    (inputBuf outputBuf : Buffer) : IO Unit := do
  let profiling ← profilingRef.get
  let startNs ← if profiling then IO.monoNanosNow else pure 0
  -- Fast path: instant replay if we already have a prepared dispatch and
  -- the input/output buffers match the prepared binding. The prepared
  -- dispatch binds specific buffer handles, so if the caller ever passes
  -- different buffers, we need to fall through to the slow path and
  -- re-prepare. In practice for Gemma 4 every layer gets its own
  -- state.* buffers that are stable across forward passes, so the fast
  -- path hits after the first call.
  if let some p ← layer.prepared.get then
    -- Use the same workgroup count that was used at prepare time.
    Execute.replayPreparedDispatch device p layer.config.outDim 1 1
    if profiling then
      let endNs ← IO.monoNanosNow
      let delta := (endNs - startNs).toUInt64
      totalNanosRef.modify (· + delta)
      callCountRef.modify (· + 1)
      perShapeAdd layer.config.inDim layer.config.outDim delta
    return

  let namedBuffers := [
    ("weights", layer.weightBuf),
    ("input", inputBuf),
    ("output", outputBuf)
  ]
  -- Pick subgroup-reduction (32 threads per workgroup, hardware
  -- `subgroupAdd`) if the device supports subgroups, else the
  -- original shared-memory tree-reduction kernel (256 threads).
  -- Both variants dispatch `outDim` workgroups, one per output row.
  let useSubgroups ← Execute.hasSubgroupSupport device
  let wgSize := if useSubgroups then 32 else 256
  let execConfig : Execute.ExecutionConfig := {
    numWorkgroups := (layer.config.outDim, 1, 1)
    workgroupSize := { x := wgSize, y := 1, z := 1 }
    extensions := if useSubgroups then ["subgroups"] else []
  }
  -- Stable cache key so the slow path (first call) skips per-call WGSL
  -- regeneration — Q4_K's kernel body is large and hashing it from
  -- scratch is noticeably expensive.
  let cacheKey : UInt64 := match layer.quantFormat with
    | .Q4_K => hash ("q4k-lin", layer.config.inDim, layer.config.outDim, useSubgroups)
    | .Q6_K => hash ("q6k-lin", layer.config.inDim, layer.config.outDim, useSubgroups)
  let shader := match layer.quantFormat, useSubgroups with
    | .Q4_K, true  => fusedQ4KMLinearSubgroupKernel layer.config
    | .Q4_K, false => fusedQ4KMLinearKernel layer.config
    | .Q6_K, true  => Hesper.Quantization.Q6_K.fusedQ6KLinearSubgroupKernel layer.config.inDim layer.config.outDim
    | .Q6_K, false => Hesper.Quantization.Q6_K.fusedQ6KLinearKernel layer.config.inDim layer.config.outDim
  Execute.executeShaderNamed device shader namedBuffers execConfig
    (cacheKey := some cacheKey) (preparedRef := some layer.prepared)
  if profiling then
    let endNs ← IO.monoNanosNow
    let delta := (endNs - startNs).toUInt64
    totalNanosRef.modify (· + delta)
    callCountRef.modify (· + 1)
    perShapeAdd layer.config.inDim layer.config.outDim delta

/-- Fused Q4_K gate+up forward pass.

    Computes `out[i] = GELU(x · W_gate[i]) * (x · W_up[i])` for all
    `i ∈ [0, outDim)` in a single dispatch. Both weight layers must be
    Q4_K and share the same `(inDim, outDim)` Config. This is Gemma 4's
    FFN gate+up+geluMul collapsed into one kernel.

    The fused kernel only supports the subgroup-reduction path
    (requires the device to expose subgroups). If subgroups aren't
    available, the caller should fall back to the 3-dispatch sequence:
    `gate.forward` + `up.forward` + separate `geluMulKernel`.

    @param preparedRef Shared prepared-dispatch cache for the fast path.
-/
def forwardFusedGateUp (device : Device)
    (gate up : LinearLayer)
    (inputBuf outputBuf : Buffer)
    (preparedRef : IO.Ref (Option Execute.PreparedDispatch))
    : IO Unit := do
  -- Preconditions (assert at the Lean level so kernel dispatch is well-formed).
  -- Both weight layers must be Q4_K and share the same shape.
  if gate.quantFormat != .Q4_K then
    throw (IO.userError s!"forwardFusedGateUp: gate must be Q4_K, got {repr gate.quantFormat}")
  if up.quantFormat != .Q4_K then
    throw (IO.userError s!"forwardFusedGateUp: up must be Q4_K, got {repr up.quantFormat}")
  if gate.config.inDim != up.config.inDim || gate.config.outDim != up.config.outDim then
    throw (IO.userError s!"forwardFusedGateUp: shape mismatch gate={gate.config.inDim}→{gate.config.outDim} up={up.config.inDim}→{up.config.outDim}")

  let profiling ← profilingRef.get
  let startNs ← if profiling then IO.monoNanosNow else pure 0

  -- Fast path: instant replay if prepared.
  if let some p ← preparedRef.get then
    Execute.replayPreparedDispatch device p gate.config.outDim 1 1
    if profiling then
      let endNs ← IO.monoNanosNow
      let delta := (endNs - startNs).toUInt64
      totalNanosRef.modify (· + delta)
      -- Attribute as two linear calls with the shared shape so the
      -- profiler still sees the gate+up workload.
      callCountRef.modify (· + 2)
      perShapeAdd gate.config.inDim gate.config.outDim delta
    return

  let namedBuffers := [
    ("weights_gate", gate.weightBuf),
    ("weights_up",   up.weightBuf),
    ("input",        inputBuf),
    ("output",       outputBuf)
  ]
  let execConfig : Execute.ExecutionConfig := {
    numWorkgroups := (gate.config.outDim, 1, 1)
    workgroupSize := { x := 32, y := 1, z := 1 }
    extensions := ["subgroups"]
  }
  let cacheKey : UInt64 :=
    hash ("q4k-gate-up", gate.config.inDim, gate.config.outDim)
  Execute.executeShaderNamed device
    (fusedQ4KMGateUpSubgroupKernel gate.config)
    namedBuffers execConfig
    (cacheKey := some cacheKey) (preparedRef := some preparedRef)
  if profiling then
    let endNs ← IO.monoNanosNow
    let delta := (endNs - startNs).toUInt64
    totalNanosRef.modify (· + delta)
    callCountRef.modify (· + 2)
    perShapeAdd gate.config.inDim gate.config.outDim delta

end Hesper.Layers.Linear
