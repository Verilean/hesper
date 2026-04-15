import Hesper.Backend
import Hesper.WGSL.Monad
import Hesper.WGSL.Exp
import Hesper.Quantization.Q4_K_M
import Hesper.Quantization.Q6_K
import Hesper.Logging
import Hesper.Layers.RMSNorm

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
open Hesper
open Hesper.Quantization.Q4_K_M (fp16ToF32 getScaleMin)
open Hesper.Logging (logVerbose)

/-! ## Profiling counters

Gated counters that a caller can enable via `profilingRef.set true` to
measure how much total wall-clock time `LinearLayer.forward` consumes.
Only meaningful when dispatches run in non-batch mode (each call
deviceWaits), so the caller should profile without `beginBatch`. Reset
`totalNanosRef` between measurements. -/
initialize profilingRef  : IO.Ref Bool  ← IO.mkRef false
/-- Toggle: use dp4a (Q8_1 quantize + INT8 SIMD matmul) for Q4_K linears.
    Off by default; enable via `dp4aEnabled.set true` before inference to
    activate llama.cpp-style accelerated matmul. -/
initialize dp4aEnabled   : IO.Ref Bool  ← IO.mkRef false
/-- Separately toggle Q6_K (lmHead) dp4a path. On by default when dp4aEnabled=true;
    set to false via HESPER_DP4A_Q6K=0 to debug Q6_K issues while keeping Q4_K dp4a. -/
initialize dp4aQ6KEnabled : IO.Ref Bool ← IO.mkRef true
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
  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareReadOnlyBuffer "input" (.array (.scalar .f32) config.inDim)
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

  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareReadOnlyBuffer "input" (.array (.scalar .f32) config.inDim)
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

  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareReadOnlyBuffer "input" (.array (.scalar .f32) config.inDim)
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

/-! ## Q4_K_M MatVec, Block-Cooperative (all 32 lanes active) -/

/-- Block-cooperative Q4_K_M mat-vec kernel.

    Fixes the subgroup-under-utilisation bug in
    `fusedQ4KMLinearSubgroupKernel`, whose outer loop was `tid → block`
    with stride 32 and therefore left lanes idle whenever
    `blocksPerRow = inDim/256 < 32`. For Gemma 4 that was nearly every
    linear: `inDim=2560 → blocksPerRow=10` (only 10/32 lanes working),
    `inDim=256 → blocksPerRow=1` (1/32 lanes).

    New scheme: 1 workgroup = 1 output row (as before), 32 threads per
    workgroup. The outer loop walks blocks **sequentially** and all 32
    lanes cooperate on each block. The Q4_K block has 32 qs u32s (4
    sub-block pairs × 8 u32 per pair) producing 256 dequantised
    weights, so the partition is one qs u32 per lane:

      c        = tid / 8          -- which sub-block pair (0..3)
      l32      = tid % 8          -- which u32 in that pair (0..7)

    Each lane reads its own `qsU32`, computes the 4 low-nibble + 4
    high-nibble weights it contains, multiplies by the matching 8 input
    elements, and FMAs into its private accumulator. The block header
    (d, dmin, sc0/1/2) is read by all 32 lanes from the same address;
    NVIDIA's memory subsystem broadcasts these redundant loads so the
    cost is one transaction per block, not 32.

    One subgroupAdd at the end reduces 32 partial sums.

    @param config Layer dimensions -/
def fusedQ4KMLinearBlockCoopKernel (config : Config) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid

  let blocksPerRow := config.inDim / 256
  let totalWeightU32 := config.outDim * blocksPerRow * 36

  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareReadOnlyBuffer "input" (.array (.scalar .f32) config.inDim)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) config.outDim)

  let inBounds := Exp.lt outIdx (Exp.litU32 config.outDim)

  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"

  let rowBaseU32 := Exp.mul outIdx (Exp.litU32 (blocksPerRow * 36))

  -- Per-lane partition of a single Q4_K block:
  --   c   = tid / 8  ∈ [0,4)  — sub-block pair
  --   l32 = tid % 8  ∈ [0,8)  — u32 within the pair
  let cLane := Exp.div tid (Exp.litU32 8)
  let l32Lane := Exp.sub tid (Exp.mul cLane (Exp.litU32 8))
  let qsOffsetInBlock := Exp.add (Exp.litU32 4)
                         (Exp.add (Exp.mul cLane (Exp.litU32 8)) l32Lane)

  -- ## Software-pipelined block loop
  --
  -- The naive version does `load → dequant → FMA` per block, which
  -- serialises memory latency with the FMA chain and leaves memory
  -- bandwidth at ~7 % of peak (see Bench/GpuFixedCost.lean). To overlap
  -- the next block's load with the current block's FMAs, we keep the
  -- five per-block u32 values (dm, sc0, sc1, sc2, qs) in mutable `var`s
  -- that are **prefetched one iteration ahead**:
  --
  --   pre-loop:          next* = weights[block 0 ...]
  --   inside loop:       curr* = next*                    (snapshot)
  --                      next* = weights[block+1 ...]     (issue only)
  --                      dequant/FMA using curr*          (overlaps with next*)
  --
  -- `ShaderM.varNamed` emits a WGSL `var`, `ShaderM.assign` rewrites
  -- it, so the SPIR-V backend sees the next-block loads as independent
  -- from the current-block math and can schedule them.
  -- Pre-loop prefetch of block 0 headers + qs.
  let nbBase0 := rowBaseU32
  let init0Dm  ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" nbBase0
  let init0Sc0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add nbBase0 (Exp.litU32 1))
  let init0Sc1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add nbBase0 (Exp.litU32 2))
  let init0Sc2 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add nbBase0 (Exp.litU32 3))
  let init0Qs  ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add nbBase0 qsOffsetInBlock)
  ShaderM.varNamed "nextDm"  (.scalar .u32) init0Dm
  ShaderM.varNamed "nextSc0" (.scalar .u32) init0Sc0
  ShaderM.varNamed "nextSc1" (.scalar .u32) init0Sc1
  ShaderM.varNamed "nextSc2" (.scalar .u32) init0Sc2
  ShaderM.varNamed "nextQs"  (.scalar .u32) init0Qs

  -- ## Runtime block loop (avoids compile-time unroll → register spill)
  ShaderM.varNamed "currDm"  (.scalar .u32) (Exp.var "nextDm")
  ShaderM.varNamed "currSc0" (.scalar .u32) (Exp.var "nextSc0")
  ShaderM.varNamed "currSc1" (.scalar .u32) (Exp.var "nextSc1")
  ShaderM.varNamed "currSc2" (.scalar .u32) (Exp.var "nextSc2")
  ShaderM.varNamed "currQs"  (.scalar .u32) (Exp.var "nextQs")

  ShaderM.loop (Exp.litU32 0) (Exp.litU32 blocksPerRow) (Exp.litU32 1) fun blockIdx => do
    let elemBase := Exp.mul blockIdx (Exp.litU32 256)

    -- Snapshot prefetched values
    ShaderM.assign "currDm"  (Exp.var (t := .scalar .u32) "nextDm")
    ShaderM.assign "currSc0" (Exp.var (t := .scalar .u32) "nextSc0")
    ShaderM.assign "currSc1" (Exp.var (t := .scalar .u32) "nextSc1")
    ShaderM.assign "currSc2" (Exp.var (t := .scalar .u32) "nextSc2")
    ShaderM.assign "currQs"  (Exp.var (t := .scalar .u32) "nextQs")
    let dmU32 : Exp (.scalar .u32) := Exp.var "currDm"
    let sc0   : Exp (.scalar .u32) := Exp.var "currSc0"
    let sc1   : Exp (.scalar .u32) := Exp.var "currSc1"
    let sc2   : Exp (.scalar .u32) := Exp.var "currSc2"
    let qsU32 : Exp (.scalar .u32) := Exp.var "currQs"
    let d := fp16ToF32 (Exp.bitAnd dmU32 (Exp.litU32 0xFFFF))
    let dmin := fp16ToF32 (Exp.shiftRight dmU32 (Exp.litU32 16))

    -- Prefetch next block (overlaps with current block's FMAs)
    let nextBlockIdx := Exp.add blockIdx (Exp.litU32 1)
    ShaderM.if_ (Exp.lt nextBlockIdx (Exp.litU32 blocksPerRow)) (do
      let nbBaseNext := Exp.add rowBaseU32 (Exp.mul nextBlockIdx (Exp.litU32 36))
      ShaderM.assign "nextDm"  (← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" nbBaseNext)
      ShaderM.assign "nextSc0" (← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add nbBaseNext (Exp.litU32 1)))
      ShaderM.assign "nextSc1" (← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add nbBaseNext (Exp.litU32 2)))
      ShaderM.assign "nextSc2" (← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add nbBaseNext (Exp.litU32 3)))
      ShaderM.assign "nextQs"  (← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add nbBaseNext qsOffsetInBlock))
    ) (pure ())

    -- Scale/min extraction (runtime cLane)
    let is0 := Exp.mul cLane (Exp.litU32 2)
    let is1 := Exp.add is0 (Exp.litU32 1)

    let extractScaleMin (is : Exp (.scalar .u32)) : Exp (.scalar .f32) × Exp (.scalar .f32) :=
      let isLow := Exp.lt is (Exp.litU32 4)
      let shift4 := Exp.mul is (Exp.litU32 8)
      let scaleLow := Exp.bitAnd (Exp.shiftRight sc0 shift4) (Exp.litU32 0x3F)
      let minLow   := Exp.bitAnd (Exp.shiftRight sc1 shift4) (Exp.litU32 0x3F)
      let isHi := Exp.sub is (Exp.litU32 4)
      let shiftHi := Exp.mul isHi (Exp.litU32 8)
      let scaleHiLo := Exp.bitAnd (Exp.shiftRight sc2 shiftHi) (Exp.litU32 0x0F)
      let scaleHiHi := Exp.shiftLeft
        (Exp.bitAnd (Exp.shiftRight sc0 (Exp.add shiftHi (Exp.litU32 6))) (Exp.litU32 0x03))
        (Exp.litU32 4)
      let scaleHigh := Exp.bitOr scaleHiLo scaleHiHi
      let minHiLo := Exp.bitAnd (Exp.shiftRight sc2 (Exp.add shiftHi (Exp.litU32 4))) (Exp.litU32 0x0F)
      let minHiHi := Exp.shiftLeft
        (Exp.bitAnd (Exp.shiftRight sc1 (Exp.add shiftHi (Exp.litU32 6))) (Exp.litU32 0x03))
        (Exp.litU32 4)
      let minHigh := Exp.bitOr minHiLo minHiHi
      let scaleU := Exp.select isLow scaleLow scaleHigh
      let minU   := Exp.select isLow minLow   minHigh
      (Exp.toF32 scaleU, Exp.toF32 minU)

    let (scaleA, minA) := extractScaleMin is0
    let (scaleB, minB) := extractScaleMin is1
    let d1 := Exp.mul d scaleA
    let m1 := Exp.mul dmin minA
    let d2 := Exp.mul d scaleB
    let m2 := Exp.mul dmin minB

    let elemOffset := Exp.add (Exp.mul cLane (Exp.litU32 64))
                      (Exp.mul l32Lane (Exp.litU32 4))
    let elemBaseAbs := Exp.add elemBase elemOffset

    -- 4 bytes → 8 weights (still compile-time unrolled — only 4 iterations)
    for b in [0:4] do
      let byte := Exp.bitAnd (Exp.shiftRight qsU32 (Exp.litU32 (b * 8))) (Exp.litU32 0xFF)
      let qLow := Exp.bitAnd byte (Exp.litU32 0xF)
      let qHigh := Exp.shiftRight byte (Exp.litU32 4)
      let elemIdxLow := Exp.add elemBaseAbs (Exp.litU32 b)
      let elemIdxHigh := Exp.add elemBaseAbs (Exp.litU32 (32 + b))
      let inLow ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.inDim) "input" elemIdxLow
      let inHigh ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.inDim) "input" elemIdxHigh
      let wLow := Exp.sub (Exp.mul d1 (Exp.toF32 qLow)) m1
      let wHigh := Exp.sub (Exp.mul d2 (Exp.toF32 qHigh)) m2
      ShaderM.assign "acc" (Exp.add acc (Exp.add (Exp.mul wLow inLow) (Exp.mul wHigh inHigh)))

  -- Subgroup reduction: 32 lanes → one scalar.
  ShaderM.varNamed "totalSum" (.scalar .f32) (Exp.subgroupAdd acc)
  let totalSum : Exp (.scalar .f32) := Exp.var "totalSum"

  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx totalSum
  ) (pure ())

/-! ## Q4_K_M MatVec, Block-Coop, 2 Rows per Workgroup -/

/-- Two-rows-per-workgroup extension of `fusedQ4KMLinearBlockCoopKernel`.

    Motivation: the single-row block-coop kernel already has every lane
    active during a block, but each WG only has 32 threads. On RTX 4070
    Ti (40 SMs × ~1536 resident threads), the `outDim`-many 32-thread
    WGs leave SMs underpopulated. Profiling showed the kernel runs at
    ~7% of memory bandwidth, suggesting occupancy-limited behaviour
    rather than bandwidth-limited.

    This variant doubles the WG size to 64 (two hardware subgroups on
    NVIDIA) and packs two output rows into each WG:

      subgroupId = tid / 32    -- 0 or 1, selects which row
      laneId     = tid % 32    -- 0..31, the block-coop lane

      outIdx     = pairIdx * 2 + subgroupId

    Because NVIDIA's subgroup size is 32, `subgroupAdd` reduces the two
    subgroups **independently and in parallel** — subgroup 0 sums row
    0's partial and subgroup 1 sums row 1's, with no cross-subgroup
    barrier needed. Each subgroup reads its own block headers and qs
    u32s; there is no register sharing between rows (so no pressure
    blowup like the earlier failed 2-row attempt).

    Dispatch: `outDim/2` workgroups × 64 threads. Requires
    `outDim % 2 == 0` and subgroup support. Caller must check both. -/
def fusedQ4KMLinearBlockCoop2RowKernel (config : Config) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let pairIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid

  -- Intra-WG decomposition: 64 threads = 2 subgroups of 32.
  let subgroupId := Exp.shiftRight tid (Exp.litU32 5)      -- tid / 32
  let laneId := Exp.bitAnd tid (Exp.litU32 31)             -- tid % 32
  let outIdx := Exp.add (Exp.mul pairIdx (Exp.litU32 2)) subgroupId

  let blocksPerRow := config.inDim / 256
  let totalWeightU32 := config.outDim * blocksPerRow * 36

  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareReadOnlyBuffer "input" (.array (.scalar .f32) config.inDim)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) config.outDim)

  let inBounds := Exp.lt outIdx (Exp.litU32 config.outDim)

  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"

  -- Row base in u32 units. Runtime value — `outIdx` depends on
  -- subgroupId at runtime, so unlike the 1-row kernel we can't bake
  -- `rowBaseU32` into a per-block constant offset.
  let rowBaseU32 := Exp.mul outIdx (Exp.litU32 (blocksPerRow * 36))

  -- Per-lane partition within a block — same as 1-row kernel, but using
  -- `laneId` instead of `tid`.
  let cLane := Exp.div laneId (Exp.litU32 8)
  let l32Lane := Exp.sub laneId (Exp.mul cLane (Exp.litU32 8))

  -- Runtime block loop (avoids compile-time unroll → register spill)
  ShaderM.loop (Exp.litU32 0) (Exp.litU32 blocksPerRow) (Exp.litU32 1) fun blockIdx => do
    let blockU32Base := Exp.add rowBaseU32 (Exp.mul blockIdx (Exp.litU32 36))
    let elemBase := Exp.mul blockIdx (Exp.litU32 256)

    let dmU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" blockU32Base
    let d := fp16ToF32 (Exp.bitAnd dmU32 (Exp.litU32 0xFFFF))
    let dmin := fp16ToF32 (Exp.shiftRight dmU32 (Exp.litU32 16))
    let sc0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add blockU32Base (Exp.litU32 1))
    let sc1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add blockU32Base (Exp.litU32 2))
    let sc2 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add blockU32Base (Exp.litU32 3))

    let is0 := Exp.mul cLane (Exp.litU32 2)
    let is1 := Exp.add is0 (Exp.litU32 1)

    let extractScaleMin (is : Exp (.scalar .u32)) : Exp (.scalar .f32) × Exp (.scalar .f32) :=
      let isLow := Exp.lt is (Exp.litU32 4)
      let shift4 := Exp.mul is (Exp.litU32 8)
      let scaleLow := Exp.bitAnd (Exp.shiftRight sc0 shift4) (Exp.litU32 0x3F)
      let minLow   := Exp.bitAnd (Exp.shiftRight sc1 shift4) (Exp.litU32 0x3F)
      let isHi := Exp.sub is (Exp.litU32 4)
      let shiftHi := Exp.mul isHi (Exp.litU32 8)
      let scaleHiLo := Exp.bitAnd (Exp.shiftRight sc2 shiftHi) (Exp.litU32 0x0F)
      let scaleHiHi := Exp.shiftLeft
        (Exp.bitAnd (Exp.shiftRight sc0 (Exp.add shiftHi (Exp.litU32 6))) (Exp.litU32 0x03))
        (Exp.litU32 4)
      let scaleHigh := Exp.bitOr scaleHiLo scaleHiHi
      let minHiLo := Exp.bitAnd (Exp.shiftRight sc2 (Exp.add shiftHi (Exp.litU32 4))) (Exp.litU32 0x0F)
      let minHiHi := Exp.shiftLeft
        (Exp.bitAnd (Exp.shiftRight sc1 (Exp.add shiftHi (Exp.litU32 6))) (Exp.litU32 0x03))
        (Exp.litU32 4)
      let minHigh := Exp.bitOr minHiLo minHiHi
      let scaleU := Exp.select isLow scaleLow scaleHigh
      let minU   := Exp.select isLow minLow   minHigh
      (Exp.toF32 scaleU, Exp.toF32 minU)

    let (scaleA, minA) := extractScaleMin is0
    let (scaleB, minB) := extractScaleMin is1
    let d1 := Exp.mul d scaleA
    let m1 := Exp.mul dmin minA
    let d2 := Exp.mul d scaleB
    let m2 := Exp.mul dmin minB

    let qsLaneIdx := Exp.add blockU32Base
                     (Exp.add (Exp.litU32 4)
                       (Exp.add (Exp.mul cLane (Exp.litU32 8)) l32Lane))
    let qsU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" qsLaneIdx

    let elemOffset := Exp.add (Exp.mul cLane (Exp.litU32 64))
                      (Exp.mul l32Lane (Exp.litU32 4))
    let elemBaseAbs := Exp.add elemBase elemOffset

    for b in [0:4] do
      let byte := Exp.bitAnd (Exp.shiftRight qsU32 (Exp.litU32 (b * 8))) (Exp.litU32 0xFF)
      let qLow := Exp.bitAnd byte (Exp.litU32 0xF)
      let qHigh := Exp.shiftRight byte (Exp.litU32 4)
      let elemIdxLow := Exp.add elemBaseAbs (Exp.litU32 b)
      let elemIdxHigh := Exp.add elemBaseAbs (Exp.litU32 (32 + b))
      let inLow ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.inDim) "input" elemIdxLow
      let inHigh ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.inDim) "input" elemIdxHigh
      let wLow := Exp.sub (Exp.mul d1 (Exp.toF32 qLow)) m1
      let wHigh := Exp.sub (Exp.mul d2 (Exp.toF32 qHigh)) m2
      ShaderM.assign "acc" (Exp.add acc (Exp.add (Exp.mul wLow inLow) (Exp.mul wHigh inHigh)))

  -- `subgroupAdd` reduces *within each subgroup*, so the two rows are
  -- summed independently without a workgroupBarrier.
  ShaderM.varNamed "totalSum" (.scalar .f32) (Exp.subgroupAdd acc)
  let totalSum : Exp (.scalar .f32) := Exp.var "totalSum"

  -- Lane 0 of each subgroup writes its row.
  ShaderM.if_ (Exp.and (Exp.eq laneId (Exp.litU32 0)) inBounds) (do
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx totalSum
  ) (pure ())

/-! ## Q4_K_M MatVec, 4 Subgroups per Row (High-Occupancy) -/

/-- 4-subgroup-per-row Q4_K_M mat-vec kernel.

    Motivation: for `ffnDown` (outDim=2560, inDim=10240), the 1-subgroup-per-row
    kernel (32 threads) only launches 81,920 threads — less than one full wave
    on RTX 4070 Ti (40 SMs × ~1536 resident threads ≈ 92k capacity). With
    insufficient warps per SM, the scheduler can't hide memory latency by
    swapping to a waiting warp, resulting in only 16% bandwidth utilization.

    This kernel uses **128 threads = 4 subgroups of 32** per workgroup, all
    cooperating on the same output row. The 40 Q4_K blocks (for blocksPerRow=40)
    are distributed across 4 subgroups in an interleaved pattern:
      subgroup 0: blocks 0, 4, 8, ...
      subgroup 1: blocks 1, 5, 9, ...
      subgroup 2: blocks 2, 6, 10, ...
      subgroup 3: blocks 3, 7, 11, ...

    Each subgroup internally is the same 32-thread block-coop kernel: each lane
    reads its slice of the Q4_K block header + qs, dequantizes 8 elements, and
    does FMA. After the block loop, `subgroupAdd` reduces within each subgroup.
    Lane 0 of each subgroup writes to shared memory (4 floats). After a barrier,
    thread 0 sums the 4 partials and writes the final output.

    Dispatch: `outDim` workgroups × 128 threads. Requires subgroup support
    (size=32). For ffnDown (2560 × 10240), this gives 327,680 total threads
    — matching llama.cpp's occupancy strategy.

    @param config Layer dimensions
    @param numSubgroups Number of subgroups (2 or 4). Default 4 for maximum occupancy. -/
def fusedQ4KMLinearMultiSubgroupKernel (config : Config) (numSubgroups : Nat := 4) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid

  let blocksPerRow := config.inDim / 256
  let totalWeightU32 := config.outDim * blocksPerRow * 36
  let wgSize := numSubgroups * 32

  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareReadOnlyBuffer "input" (.array (.scalar .f32) config.inDim)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) config.outDim)

  -- Shared memory for cross-subgroup reduction
  ShaderM.sharedNamed "shared_partial" (.array (.scalar .f32) numSubgroups)

  let inBounds := Exp.lt outIdx (Exp.litU32 config.outDim)

  -- Subgroup decomposition: 128 threads = 4 subgroups of 32
  let subgroupId := Exp.shiftRight tid (Exp.litU32 5)   -- tid / 32
  let laneId := Exp.bitAnd tid (Exp.litU32 31)          -- tid % 32

  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"

  let rowBaseU32 := Exp.mul outIdx (Exp.litU32 (blocksPerRow * 36))

  -- Per-lane partition within a block (same as 1-subgroup kernel)
  let cLane := Exp.div laneId (Exp.litU32 8)
  let l32Lane := Exp.sub laneId (Exp.mul cLane (Exp.litU32 8))

  -- Block loop: each subgroup processes blocks `subgroupId, subgroupId + numSubgroups, ...`
  ShaderM.loop subgroupId (Exp.litU32 blocksPerRow) (Exp.litU32 numSubgroups) fun blockIdx => do
    let blockU32Base := Exp.add rowBaseU32 (Exp.mul blockIdx (Exp.litU32 36))
    let elemBase := Exp.mul blockIdx (Exp.litU32 256)

    let dmU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" blockU32Base
    let d := fp16ToF32 (Exp.bitAnd dmU32 (Exp.litU32 0xFFFF))
    let dmin := fp16ToF32 (Exp.shiftRight dmU32 (Exp.litU32 16))
    let sc0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add blockU32Base (Exp.litU32 1))
    let sc1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add blockU32Base (Exp.litU32 2))
    let sc2 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add blockU32Base (Exp.litU32 3))

    let is0 := Exp.mul cLane (Exp.litU32 2)
    let is1 := Exp.add is0 (Exp.litU32 1)

    let extractScaleMin (is : Exp (.scalar .u32)) : Exp (.scalar .f32) × Exp (.scalar .f32) :=
      let isLow := Exp.lt is (Exp.litU32 4)
      let shift4 := Exp.mul is (Exp.litU32 8)
      let scaleLow := Exp.bitAnd (Exp.shiftRight sc0 shift4) (Exp.litU32 0x3F)
      let minLow   := Exp.bitAnd (Exp.shiftRight sc1 shift4) (Exp.litU32 0x3F)
      let isHi := Exp.sub is (Exp.litU32 4)
      let shiftHi := Exp.mul isHi (Exp.litU32 8)
      let scaleHiLo := Exp.bitAnd (Exp.shiftRight sc2 shiftHi) (Exp.litU32 0x0F)
      let scaleHiHi := Exp.shiftLeft
        (Exp.bitAnd (Exp.shiftRight sc0 (Exp.add shiftHi (Exp.litU32 6))) (Exp.litU32 0x03))
        (Exp.litU32 4)
      let scaleHigh := Exp.bitOr scaleHiLo scaleHiHi
      let minHiLo := Exp.bitAnd (Exp.shiftRight sc2 (Exp.add shiftHi (Exp.litU32 4))) (Exp.litU32 0x0F)
      let minHiHi := Exp.shiftLeft
        (Exp.bitAnd (Exp.shiftRight sc1 (Exp.add shiftHi (Exp.litU32 6))) (Exp.litU32 0x03))
        (Exp.litU32 4)
      let minHigh := Exp.bitOr minHiLo minHiHi
      let scaleU := Exp.select isLow scaleLow scaleHigh
      let minU   := Exp.select isLow minLow   minHigh
      (Exp.toF32 scaleU, Exp.toF32 minU)

    let (scaleA, minA) := extractScaleMin is0
    let (scaleB, minB) := extractScaleMin is1
    let d1 := Exp.mul d scaleA
    let m1 := Exp.mul dmin minA
    let d2 := Exp.mul d scaleB
    let m2 := Exp.mul dmin minB

    let qsLaneIdx := Exp.add blockU32Base
                     (Exp.add (Exp.litU32 4)
                       (Exp.add (Exp.mul cLane (Exp.litU32 8)) l32Lane))
    let qsU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" qsLaneIdx

    let elemOffset := Exp.add (Exp.mul cLane (Exp.litU32 64))
                      (Exp.mul l32Lane (Exp.litU32 4))
    let elemBaseAbs := Exp.add elemBase elemOffset

    for b in [0:4] do
      let byte := Exp.bitAnd (Exp.shiftRight qsU32 (Exp.litU32 (b * 8))) (Exp.litU32 0xFF)
      let qLow := Exp.bitAnd byte (Exp.litU32 0xF)
      let qHigh := Exp.shiftRight byte (Exp.litU32 4)
      let elemIdxLow := Exp.add elemBaseAbs (Exp.litU32 b)
      let elemIdxHigh := Exp.add elemBaseAbs (Exp.litU32 (32 + b))
      let inLow ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.inDim) "input" elemIdxLow
      let inHigh ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.inDim) "input" elemIdxHigh
      let wLow := Exp.sub (Exp.mul d1 (Exp.toF32 qLow)) m1
      let wHigh := Exp.sub (Exp.mul d2 (Exp.toF32 qHigh)) m2
      ShaderM.assign "acc" (Exp.add acc (Exp.add (Exp.mul wLow inLow) (Exp.mul wHigh inHigh)))

  -- Step 1: subgroupAdd reduces within each 32-lane subgroup
  ShaderM.varNamed "subgroupSum" (.scalar .f32) (Exp.subgroupAdd acc)
  let subgroupSum : Exp (.scalar .f32) := Exp.var "subgroupSum"

  -- Step 2: lane 0 of each subgroup writes partial to shared memory
  ShaderM.if_ (Exp.eq laneId (Exp.litU32 0)) (do
    ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_partial" subgroupId subgroupSum
  ) (pure ())

  ShaderM.barrier

  -- Step 3: thread 0 sums all subgroup partials and writes output
  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    ShaderM.varNamed "total" (.scalar .f32) (Exp.litF32 0.0)
    let total : Exp (.scalar .f32) := Exp.var "total"
    for sg in List.range numSubgroups do
      let sgPartial ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := numSubgroups) "shared_partial" (Exp.litU32 sg)
      ShaderM.assign "total" (Exp.add total sgPartial)
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx (Exp.var (t := .scalar .f32) "total")
  ) (pure ())

/-! ## Q8_1 Input Quantization & Q4_K × Q8_1 dp4a MatVec

    llama.cppと同じ戦略:
    1. 入力f32ベクトルをQ8_1 (ブロック=32要素: 2×fp16 + 32×int8) に変換
    2. Q4_K weight × Q8_1 input の dot product を dp4a で計算
       (1命令で 4×INT8 積和、f32 FMAの約4倍スループット)

    Q8_1 layout per 32-element block (36 bytes = 9 u32):
    - u32[0]: fp16(d) | fp16(s), s = d * sum(qs)
    - u32[1..8]: 32 int8 quants (packed 4 per u32)

    Q4_K layout per 256-element block (144 bytes = 36 u32):
    - u32[0]:  fp16(d) | fp16(dmin)
    - u32[1..3]: 12 bytes of 6-bit scales + mins (8 sub-blocks × 6 bits each)
    - u32[4..35]: 128 bytes of 4-bit quants (256 values × 4 bits)
-/

/-- Q8_1 quantization kernel: f32 input → Q8_1 packed u32 buffer.

    Each workgroup processes one 32-element block (one Q8_1 unit).
    Grid: (nBlocks, 1, 1) where nBlocks = inDim / 32.
    Workgroup size: 32 threads.

    Output layout per block (9 u32 = 36 bytes):
    - out[0] = pack2x16float(d, s)  — scale d + precomputed s = d*sum(qs)
    - out[1..8] = 8 u32, each packs 4×int8 quants

    Algorithm per block:
    1. Each thread reads 1 input f32 value
    2. subgroupMax of |x| → amax
    3. d = amax / 127
    4. q[tid] = round(x / d)  (clamped to [-127, 127])
    5. subgroupAdd of q → sum
    6. s = d * sum
    7. Pack q[0..3], q[4..7], ... into 8 u32s
    8. Thread 0 writes d|s header; threads 0,4,8,...,28 write packed quants.
-/
def quantizeQ8_1Kernel (inDim : Nat) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let blockIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid

  let nBlocks := inDim / 32
  let outU32Size := nBlocks * 9

  let _input ← ShaderM.declareReadOnlyBuffer "input" (.array (.scalar .f32) inDim)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .u32) outU32Size)

  -- Each thread reads one f32 element
  let elemIdx := Exp.add (Exp.mul blockIdx (Exp.litU32 32)) tid
  let x ← ShaderM.readBuffer (ty := .scalar .f32) (n := inDim) "input" elemIdx

  -- 1. Compute amax via subgroupMax. Materialize to var so the reduction
  -- is emitted ONCE in straight-line code (not inlined into downstream if-branches).
  let absX := Exp.select (Exp.lt x (Exp.litF32 0.0)) (Exp.sub (Exp.litF32 0.0) x) x
  ShaderM.varNamed "amax" (.scalar .f32) (Exp.subgroupMax absX)
  let amax : Exp (.scalar .f32) := Exp.var "amax"

  -- 2. Compute scale d = amax / 127, materialize.
  ShaderM.varNamed "d_q8" (.scalar .f32) (Exp.div amax (Exp.litF32 127.0))
  let d : Exp (.scalar .f32) := Exp.var "d_q8"

  -- 3. Quantize: q = round(x / d). Guard against d==0 → produce 0.
  -- Use round-to-nearest (matches llama.cpp's roundf). cvt.rzi (truncate
  -- toward zero) was producing ±0.5 systematic errors that accumulated
  -- over 256-element dot products into wildly wrong outputs.
  ShaderM.varNamed "qF32" (.scalar .f32)
    (Exp.select (Exp.eq d (Exp.litF32 0.0)) (Exp.litF32 0.0) (Exp.div x d))
  let qF32 : Exp (.scalar .f32) := Exp.var "qF32"
  -- roundToI32: round-to-nearest-even, two's-complement i32 stored as u32.
  -- Mask low 8 bits for int8 packing.
  ShaderM.varNamed "qByte" (.scalar .u32)
    (Exp.bitAnd (Exp.roundToI32 qF32) (Exp.litU32 0xFF))
  let qByte : Exp (.scalar .u32) := Exp.var "qByte"

  -- 4. Thread 0 writes header: d as f32 (bitcast to u32 for storage).
  let hdrOff := Exp.mul blockIdx (Exp.litU32 9)
  ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
    let dBits : Exp (.scalar .u32) := Exp.bitcast d
    ShaderM.writeBuffer (ty := .scalar .u32) "output" hdrOff dBits
  ) (pure ())

  -- 5. Pack quants via shared memory: all threads write their q byte to shared mem,
  -- then every 4th thread reads 4 consecutive values and packs them into one u32.
  ShaderM.sharedNamed "shared_q" (.array (.scalar .u32) 32)
  ShaderM.writeWorkgroup (ty := .scalar .u32) "shared_q" tid qByte
  ShaderM.barrier

  -- Every 4th thread (tid = 0, 4, 8, ..., 28) packs 4 consecutive quants.
  let laneQuarter := Exp.div tid (Exp.litU32 4)  -- 0..7
  let isQuarterLane := Exp.eq (Exp.sub tid (Exp.mul laneQuarter (Exp.litU32 4))) (Exp.litU32 0)
  ShaderM.if_ isQuarterLane (do
    -- Read 4 consecutive bytes from shared memory
    let base := Exp.mul laneQuarter (Exp.litU32 4)
    let b0 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := 32) "shared_q" base
    let b1 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := 32) "shared_q" (Exp.add base (Exp.litU32 1))
    let b2 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := 32) "shared_q" (Exp.add base (Exp.litU32 2))
    let b3 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := 32) "shared_q" (Exp.add base (Exp.litU32 3))
    let packed := Exp.bitOr (Exp.bitOr b0 (Exp.shiftLeft b1 (Exp.litU32 8)))
                            (Exp.bitOr (Exp.shiftLeft b2 (Exp.litU32 16)) (Exp.shiftLeft b3 (Exp.litU32 24)))
    let outIdx := Exp.add hdrOff (Exp.add (Exp.litU32 1) laneQuarter)
    ShaderM.writeBuffer (ty := .scalar .u32) "output" outIdx packed
  ) (pure ())

/-- Q4_K × Q8_1 mat-vec kernel using dp4a (INT8 SIMD dot product).

    llama.cppの `vec_dot_q4_K_q8_1_impl_vmmq` と同じアルゴリズム。
    各32スレッド subgroup が1出力要素 (1行) を計算。

    Per block (256 elements = 8 Q8_1 sub-blocks):
    - Q4_K header: u32[0] = fp16(d)|fp16(dmin), u32[1..3] = scales/mins
    - Q4_K quants: u32[4..35] = 128 bytes packed 4-bit values
    - Q8_1 header: u32[0] per sub-block = fp16(d8)|fp16(s8)
    - Q8_1 quants: u32[1..8] per sub-block = 32 int8 packed 4-per-u32

    Thread layout: lane `t` ∈ [0,32) processes 2 Q8_1 sub-blocks
    (sub-block pair `t/4`, element offset `t%4`).
    QR4_K = 2: each lane processes 2 adjacent sub-blocks (low/high nibbles).

    Grid: (outDim, 1, 1) workgroups × 32 threads.
-/
def fusedQ4KMLinearDP4AKernel (config : Config) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid

  let blocksPerRow := config.inDim / 256
  let totalWeightU32 := config.outDim * blocksPerRow * 36
  let q8BlocksPerRow := config.inDim / 32
  let q8InputU32Size := q8BlocksPerRow * 9

  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareReadOnlyBuffer "input_q8" (.array (.scalar .u32) q8InputU32Size)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) config.outDim)

  let inBounds := Exp.lt outIdx (Exp.litU32 config.outDim)

  -- Accumulator: acc = Σ_block (d*sumfD - dmin*sumfM)
  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"

  let rowBaseU32 := Exp.mul outIdx (Exp.litU32 (blocksPerRow * 36))

  -- Lane decomposition (matches llama.cpp's `iqs` partitioning):
  -- Each 32-lane subgroup handles 8 Q8_1 sub-blocks per Q4_K block = 32 lanes / 4 = 8 pairs
  -- iqs = 2 * tid (0,2,4,..,62). bq8_offset = QR4_K * (iqs/2 / (QI8_1/2)) = 2 * (tid/4)
  --   tid ∈ [0,4)  → bq8_offset = 0   (pair of sub-blocks 0,1)
  --   tid ∈ [4,8)  → bq8_offset = 2   (pair 2,3)
  --   tid ∈ [8,12) → bq8_offset = 4   (pair 4,5)
  --   tid ∈ [12,16)→ bq8_offset = 6   (pair 6,7)
  --   tid ∈ [16,20)→ bq8_offset = 0 (DUPLICATE!) — this is only 16 unique partitions
  -- NOTE: llama.cpp uses only 16 lanes; the 32-lane version here needs different mapping.
  -- llama.cpp's vec_dot_q4_K_q8_1 uses iqs ∈ {0, 2, ..., 30} = 16 values.
  -- bq8_offset = 2 * ((iqs/2) / 4) = 2 * (pairIdx), pairIdx ∈ {0..3}, 4 unique pairs.
  -- elemOff = (iqs/2) % 4 ∈ {0..3}.
  --
  -- So only 16 lanes (4 pairs × 4 elems) do useful work. For a 32-lane warp
  -- we duplicate: lanes 0..15 and lanes 16..31 each compute the same result.
  -- The final subgroupAdd then double-counts, so we divide by 2 at the end.
  --
  -- This exactly matches the reference kernel's logic and produces correct
  -- Q4_K × Q8_1 dot products.
  let laneLow := Exp.bitAnd tid (Exp.litU32 15)  -- tid % 16, 0..15
  let pairIdx := Exp.div laneLow (Exp.litU32 4)  -- 0..3
  let elemOff := Exp.sub laneLow (Exp.mul pairIdx (Exp.litU32 4))  -- 0..3
  let bq8Off := Exp.mul pairIdx (Exp.litU32 2)  -- 0,2,4,6 — valid sub-block indices

  ShaderM.loop (Exp.litU32 0) (Exp.litU32 blocksPerRow) (Exp.litU32 1) fun blockIdx => do
    let blockU32Base := Exp.add rowBaseU32 (Exp.mul blockIdx (Exp.litU32 36))
    -- Per-block Q4_K header
    let dmU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" blockU32Base
    let dF := fp16ToF32 (Exp.bitAnd dmU32 (Exp.litU32 0xFFFF))
    let dminF := fp16ToF32 (Exp.shiftRight dmU32 (Exp.litU32 16))

    -- Scales/mins: sc[8], m[8] — packed in u32[1..3] per Q4_K layout.
    -- llama.cpp treats them as uint16[6]; we extract via bit-ops.
    let sc0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add blockU32Base (Exp.litU32 1))
    let sc1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add blockU32Base (Exp.litU32 2))
    let sc2 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add blockU32Base (Exp.litU32 3))

    -- Extract scale[is] and min[is] for is = bq8Off and bq8Off+1
    let extractScaleMin (is : Exp (.scalar .u32)) : Exp (.scalar .f32) × Exp (.scalar .f32) :=
      let isLow := Exp.lt is (Exp.litU32 4)
      let shift4 := Exp.mul is (Exp.litU32 8)
      let scaleLow := Exp.bitAnd (Exp.shiftRight sc0 shift4) (Exp.litU32 0x3F)
      let minLow   := Exp.bitAnd (Exp.shiftRight sc1 shift4) (Exp.litU32 0x3F)
      let isHi := Exp.sub is (Exp.litU32 4)
      let shiftHi := Exp.mul isHi (Exp.litU32 8)
      let scaleHiLo := Exp.bitAnd (Exp.shiftRight sc2 shiftHi) (Exp.litU32 0x0F)
      let scaleHiHi := Exp.shiftLeft
        (Exp.bitAnd (Exp.shiftRight sc0 (Exp.add shiftHi (Exp.litU32 6))) (Exp.litU32 0x03))
        (Exp.litU32 4)
      let scaleHigh := Exp.bitOr scaleHiLo scaleHiHi
      let minHiLo := Exp.bitAnd (Exp.shiftRight sc2 (Exp.add shiftHi (Exp.litU32 4))) (Exp.litU32 0x0F)
      let minHiHi := Exp.shiftLeft
        (Exp.bitAnd (Exp.shiftRight sc1 (Exp.add shiftHi (Exp.litU32 6))) (Exp.litU32 0x03))
        (Exp.litU32 4)
      let minHigh := Exp.bitOr minHiLo minHiHi
      let scaleU := Exp.select isLow scaleLow scaleHigh
      let minU   := Exp.select isLow minLow   minHigh
      (Exp.toF32 scaleU, Exp.toF32 minU)

    let (scA, mA) := extractScaleMin bq8Off
    let (scB, mB) := extractScaleMin (Exp.add bq8Off (Exp.litU32 1))

    -- Read Q4_K quants for this lane's element pair:
    -- q4[0] and q4[4] per llama.cpp (v[0] and v[1])
    -- q4 base = bq4_K->qs + 16 * bq8_offset + 4 * elemOff (in bytes)
    -- In u32: offset = 4 + (16 * bq8Off + 4 * elemOff) / 4 = 4 + 4*bq8Off + elemOff
    let q4BaseIdx := Exp.add blockU32Base
      (Exp.add (Exp.litU32 4) (Exp.add (Exp.mul bq8Off (Exp.litU32 4)) elemOff))
    let v0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" q4BaseIdx
    -- v1 = q4[4] = 4 u32s later (each u32 covers 8 weights × 4 bits = 32 bits)
    let v1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add q4BaseIdx (Exp.litU32 4))

    -- Read Q8_1 quants: u[0], u[1] for sub-block bq8Off, u[2], u[3] for sub-block bq8Off+1
    let q8Sub0Base := Exp.add (Exp.mul blockIdx (Exp.litU32 (8 * 9))) (Exp.mul bq8Off (Exp.litU32 9))
    let q8Sub1Base := Exp.add q8Sub0Base (Exp.litU32 9)
    -- Q8_1 header at offset 0, quants at offset 1..8.
    -- llama.cpp: q8 = (int*)bq8i->qs + elemOff; u[0]=q8[0], u[1]=q8[4]
    -- In our u32 layout: sub_base + 1 + elemOff for u[0], sub_base + 1 + 4 + elemOff for u[1]
    let u0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" (Exp.add q8Sub0Base (Exp.add (Exp.litU32 1) elemOff))
    let u1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" (Exp.add q8Sub0Base (Exp.add (Exp.litU32 5) elemOff))
    let u2 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" (Exp.add q8Sub1Base (Exp.add (Exp.litU32 1) elemOff))
    let u3 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" (Exp.add q8Sub1Base (Exp.add (Exp.litU32 5) elemOff))

    -- Read d8 for each Q8_1 sub-block (f32, bitcast from u32 header)
    let q8Hdr0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" q8Sub0Base
    let q8Hdr1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" q8Sub1Base
    let d8A : Exp (.scalar .f32) := Exp.bitcast q8Hdr0
    let d8B : Exp (.scalar .f32) := Exp.bitcast q8Hdr1

    -- llama.cpp loop: QR4_K=2 iterations. i=0 uses v[0]>>0 & 0x0F..., u[0]+u[1], d8[0], sc[0], m[0]
    --                                       i=1 uses v[0]>>4 & 0x0F..., u[2]+u[3], d8[1], sc[1], m[1]
    -- i=0:
    let v0i0 := Exp.bitAnd v0 (Exp.litU32 0x0F0F0F0F)
    let v1i0 := Exp.bitAnd v1 (Exp.litU32 0x0F0F0F0F)
    -- Use signed dp4a: v0i/v1i are nibbles in [0,15] (fit int8 fine),
    -- u is signed int8 per Q8_1 spec. llama.cpp uses dp4a.s32.s32 throughout.
    let acc0 := Exp.dot4I8Packed v0i0 u0
    let dot1_0 := Exp.dot4I8Packed v1i0 u1
    let dot1_0Combined := Exp.add acc0 dot1_0
    let sumU_0 := Exp.add (Exp.dot4I8Packed (Exp.litU32 0x01010101) u0)
                          (Exp.dot4I8Packed (Exp.litU32 0x01010101) u1)
    let sumfD_0 := Exp.mul d8A (Exp.mul (Exp.toF32 dot1_0Combined) scA)
    let sumfM_0 := Exp.mul d8A (Exp.mul (Exp.toF32 sumU_0) mA)

    -- i=1: shift v by 4 then mask
    let v0i1 := Exp.bitAnd (Exp.shiftRight v0 (Exp.litU32 4)) (Exp.litU32 0x0F0F0F0F)
    let v1i1 := Exp.bitAnd (Exp.shiftRight v1 (Exp.litU32 4)) (Exp.litU32 0x0F0F0F0F)
    let acc1 := Exp.dot4I8Packed v0i1 u2
    let dot1_1 := Exp.dot4I8Packed v1i1 u3
    let dot1_1Combined := Exp.add acc1 dot1_1
    let sumU_1 := Exp.add (Exp.dot4I8Packed (Exp.litU32 0x01010101) u2)
                          (Exp.dot4I8Packed (Exp.litU32 0x01010101) u3)
    let sumfD_1 := Exp.mul d8B (Exp.mul (Exp.toF32 dot1_1Combined) scB)
    let sumfM_1 := Exp.mul d8B (Exp.mul (Exp.toF32 sumU_1) mB)

    -- Per-block contribution: d*(sumfD_0+sumfD_1) - dmin*(sumfM_0+sumfM_1)
    let blockSumfD := Exp.add sumfD_0 sumfD_1
    let blockSumfM := Exp.add sumfM_0 sumfM_1
    let blockContrib := Exp.sub (Exp.mul dF blockSumfD) (Exp.mul dminF blockSumfM)
    ShaderM.assign "acc" (Exp.add acc blockContrib)

  -- Subgroup reduction. Since 32 lanes compute duplicate work (lanes 0..15
  -- and 16..31 both cover all sub-blocks), divide by 2.
  ShaderM.varNamed "total" (.scalar .f32)
    (Exp.mul (Exp.subgroupAdd acc) (Exp.litF32 0.5))
  let total : Exp (.scalar .f32) := Exp.var "total"

  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx total
  ) (pure ())

/-- Q4_K dp4a matmul with a GELU-and-per-layer-input-mul epilogue.

    Body identical to `fusedQ4KMLinearDP4AKernel` (one 32-lane subgroup
    per output row) **except** for the final write: after the subgroupAdd
    + halving, lane 0 reads `per_layer_input[plOffset + outIdx]`, applies
    tanh-approx GELU to the dot product, and writes the product.

    Replaces the PLE `ple.inpGate` + `ple.geluGateMul` dispatch pair.
    `plTotalSize = embdPerLayer * numLayers`; `plOffset = li * embdPerLayer`
    is baked into the PTX for cache-friendly dispatch.

    TODO: this body is a copy of `fusedQ4KMLinearDP4AKernel`'s matmul
    portion with a custom epilogue.  Once the Circuit DSL gains a
    `Prim.matmulQ4KWithEpilogue` node, the two can be unified into one
    lowering parameterised by a `ScalarExp` tail. -/
def fusedQ4KMLinearDP4AGeluSliceKernel
    (config : Config) (plTotalSize plOffset : Nat) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid

  let blocksPerRow := config.inDim / 256
  let totalWeightU32 := config.outDim * blocksPerRow * 36
  let q8BlocksPerRow := config.inDim / 32
  let q8InputU32Size := q8BlocksPerRow * 9

  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareReadOnlyBuffer "input_q8" (.array (.scalar .u32) q8InputU32Size)
  let _plInput ← ShaderM.declareReadOnlyBuffer "per_layer_input" (.array (.scalar .f32) plTotalSize)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) config.outDim)

  let inBounds := Exp.lt outIdx (Exp.litU32 config.outDim)

  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"

  let rowBaseU32 := Exp.mul outIdx (Exp.litU32 (blocksPerRow * 36))

  let laneLow := Exp.bitAnd tid (Exp.litU32 15)
  let pairIdx := Exp.div laneLow (Exp.litU32 4)
  let elemOff := Exp.sub laneLow (Exp.mul pairIdx (Exp.litU32 4))
  let bq8Off := Exp.mul pairIdx (Exp.litU32 2)

  ShaderM.loop (Exp.litU32 0) (Exp.litU32 blocksPerRow) (Exp.litU32 1) fun blockIdx => do
    let blockU32Base := Exp.add rowBaseU32 (Exp.mul blockIdx (Exp.litU32 36))
    let dmU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" blockU32Base
    let dF := fp16ToF32 (Exp.bitAnd dmU32 (Exp.litU32 0xFFFF))
    let dminF := fp16ToF32 (Exp.shiftRight dmU32 (Exp.litU32 16))

    let sc0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add blockU32Base (Exp.litU32 1))
    let sc1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add blockU32Base (Exp.litU32 2))
    let sc2 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add blockU32Base (Exp.litU32 3))

    let extractScaleMin (is : Exp (.scalar .u32)) : Exp (.scalar .f32) × Exp (.scalar .f32) :=
      let isLow := Exp.lt is (Exp.litU32 4)
      let shift4 := Exp.mul is (Exp.litU32 8)
      let scaleLow := Exp.bitAnd (Exp.shiftRight sc0 shift4) (Exp.litU32 0x3F)
      let minLow   := Exp.bitAnd (Exp.shiftRight sc1 shift4) (Exp.litU32 0x3F)
      let isHi := Exp.sub is (Exp.litU32 4)
      let shiftHi := Exp.mul isHi (Exp.litU32 8)
      let scaleHiLo := Exp.bitAnd (Exp.shiftRight sc2 shiftHi) (Exp.litU32 0x0F)
      let scaleHiHi := Exp.shiftLeft
        (Exp.bitAnd (Exp.shiftRight sc0 (Exp.add shiftHi (Exp.litU32 6))) (Exp.litU32 0x03))
        (Exp.litU32 4)
      let scaleHigh := Exp.bitOr scaleHiLo scaleHiHi
      let minHiLo := Exp.bitAnd (Exp.shiftRight sc2 (Exp.add shiftHi (Exp.litU32 4))) (Exp.litU32 0x0F)
      let minHiHi := Exp.shiftLeft
        (Exp.bitAnd (Exp.shiftRight sc1 (Exp.add shiftHi (Exp.litU32 6))) (Exp.litU32 0x03))
        (Exp.litU32 4)
      let minHigh := Exp.bitOr minHiLo minHiHi
      let scaleU := Exp.select isLow scaleLow scaleHigh
      let minU   := Exp.select isLow minLow   minHigh
      (Exp.toF32 scaleU, Exp.toF32 minU)

    let (scA, mA) := extractScaleMin bq8Off
    let (scB, mB) := extractScaleMin (Exp.add bq8Off (Exp.litU32 1))

    let q4BaseIdx := Exp.add blockU32Base
      (Exp.add (Exp.litU32 4) (Exp.add (Exp.mul bq8Off (Exp.litU32 4)) elemOff))
    let v0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" q4BaseIdx
    let v1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add q4BaseIdx (Exp.litU32 4))

    let q8Sub0Base := Exp.add (Exp.mul blockIdx (Exp.litU32 (8 * 9))) (Exp.mul bq8Off (Exp.litU32 9))
    let q8Sub1Base := Exp.add q8Sub0Base (Exp.litU32 9)
    let u0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" (Exp.add q8Sub0Base (Exp.add (Exp.litU32 1) elemOff))
    let u1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" (Exp.add q8Sub0Base (Exp.add (Exp.litU32 5) elemOff))
    let u2 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" (Exp.add q8Sub1Base (Exp.add (Exp.litU32 1) elemOff))
    let u3 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" (Exp.add q8Sub1Base (Exp.add (Exp.litU32 5) elemOff))

    let q8Hdr0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" q8Sub0Base
    let q8Hdr1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" q8Sub1Base
    let d8A : Exp (.scalar .f32) := Exp.bitcast q8Hdr0
    let d8B : Exp (.scalar .f32) := Exp.bitcast q8Hdr1

    let v0i0 := Exp.bitAnd v0 (Exp.litU32 0x0F0F0F0F)
    let v1i0 := Exp.bitAnd v1 (Exp.litU32 0x0F0F0F0F)
    let acc0 := Exp.dot4I8Packed v0i0 u0
    let dot1_0 := Exp.dot4I8Packed v1i0 u1
    let dot1_0Combined := Exp.add acc0 dot1_0
    let sumU_0 := Exp.add (Exp.dot4I8Packed (Exp.litU32 0x01010101) u0)
                          (Exp.dot4I8Packed (Exp.litU32 0x01010101) u1)
    let sumfD_0 := Exp.mul d8A (Exp.mul (Exp.toF32 dot1_0Combined) scA)
    let sumfM_0 := Exp.mul d8A (Exp.mul (Exp.toF32 sumU_0) mA)

    let v0i1 := Exp.bitAnd (Exp.shiftRight v0 (Exp.litU32 4)) (Exp.litU32 0x0F0F0F0F)
    let v1i1 := Exp.bitAnd (Exp.shiftRight v1 (Exp.litU32 4)) (Exp.litU32 0x0F0F0F0F)
    let acc1 := Exp.dot4I8Packed v0i1 u2
    let dot1_1 := Exp.dot4I8Packed v1i1 u3
    let dot1_1Combined := Exp.add acc1 dot1_1
    let sumU_1 := Exp.add (Exp.dot4I8Packed (Exp.litU32 0x01010101) u2)
                          (Exp.dot4I8Packed (Exp.litU32 0x01010101) u3)
    let sumfD_1 := Exp.mul d8B (Exp.mul (Exp.toF32 dot1_1Combined) scB)
    let sumfM_1 := Exp.mul d8B (Exp.mul (Exp.toF32 sumU_1) mB)

    let blockSumfD := Exp.add sumfD_0 sumfD_1
    let blockSumfM := Exp.add sumfM_0 sumfM_1
    let blockContrib := Exp.sub (Exp.mul dF blockSumfD) (Exp.mul dminF blockSumfM)
    ShaderM.assign "acc" (Exp.add acc blockContrib)

  ShaderM.varNamed "total" (.scalar .f32)
    (Exp.mul (Exp.subgroupAdd acc) (Exp.litF32 0.5))
  let total : Exp (.scalar .f32) := Exp.var "total"

  -- Epilogue: gelu(total) * per_layer_input[plOffset + outIdx].
  -- Only lane 0 does the extra read + write (same as the base kernel).
  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    -- tanh-approx GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    let x := total
    let x3 := Exp.mul (Exp.mul x x) x
    let inner := Exp.mul (Exp.litF32 0.7978845608028654)
                         (Exp.add x (Exp.mul (Exp.litF32 0.044715) x3))
    let gelu := Exp.mul (Exp.mul (Exp.litF32 0.5) x)
                        (Exp.add (Exp.litF32 1.0) (Exp.tanh inner))
    let plIdx := Exp.add outIdx (Exp.litU32 plOffset)
    let plVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := plTotalSize) "per_layer_input" plIdx
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx (Exp.mul gelu plVal)
  ) (pure ())

/-- Fused Q4_K dp4a gate + up (dp4a path version of `fusedQ4KMLinearGateUpKernel`).

    Computes `out[i] = GELU(dot(W_gate[i], x_q8)) * dot(W_up[i], x_q8)` in a
    single kernel, sharing one Q8_1 input buffer + one per-block header /
    scale / quant decode pipeline between gate and up.  Saves one full pass
    over the Q8_1 input vs the two-dispatch gate.forward + up.forward pair.

    Buffers: weights_gate, weights_up (same shape), input_q8, output.
    Dispatch: (outDim, 1, 1) × 32 threads. Same lane-decomposition as
    fusedQ4KMLinearDP4AKernel. -/
def fusedQ4KMGateUpDP4AKernel (config : Config) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid

  let blocksPerRow := config.inDim / 256
  let totalWeightU32 := config.outDim * blocksPerRow * 36
  let q8BlocksPerRow := config.inDim / 32
  let q8InputU32Size := q8BlocksPerRow * 9

  let _weightsGate ← ShaderM.declareReadOnlyBuffer "weights_gate" (.array (.scalar .u32) totalWeightU32)
  let _weightsUp   ← ShaderM.declareReadOnlyBuffer "weights_up"   (.array (.scalar .u32) totalWeightU32)
  let _input       ← ShaderM.declareReadOnlyBuffer "input_q8"     (.array (.scalar .u32) q8InputU32Size)
  let _output      ← ShaderM.declareOutputBuffer "output"         (.array (.scalar .f32) config.outDim)

  let inBounds := Exp.lt outIdx (Exp.litU32 config.outDim)

  ShaderM.varNamed "accG" (.scalar .f32) (Exp.litF32 0.0)
  ShaderM.varNamed "accU" (.scalar .f32) (Exp.litF32 0.0)
  let accG : Exp (.scalar .f32) := Exp.var "accG"
  let accU : Exp (.scalar .f32) := Exp.var "accU"

  let rowBaseU32 := Exp.mul outIdx (Exp.litU32 (blocksPerRow * 36))

  let laneLow := Exp.bitAnd tid (Exp.litU32 15)
  let pairIdx := Exp.div laneLow (Exp.litU32 4)
  let elemOff := Exp.sub laneLow (Exp.mul pairIdx (Exp.litU32 4))
  let bq8Off := Exp.mul pairIdx (Exp.litU32 2)

  ShaderM.loop (Exp.litU32 0) (Exp.litU32 blocksPerRow) (Exp.litU32 1) fun blockIdx => do
    let blockU32Base := Exp.add rowBaseU32 (Exp.mul blockIdx (Exp.litU32 36))

    -- Per-block header + scales, read once per weight buffer.
    let processWeight (which : String) (acc : Exp (.scalar .f32))
        (u0 u1 u2 u3 : Exp (.scalar .u32)) (d8A d8B : Exp (.scalar .f32))
        : ShaderM (Exp (.scalar .f32)) := do
      let dmU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) which blockU32Base
      let dF := fp16ToF32 (Exp.bitAnd dmU32 (Exp.litU32 0xFFFF))
      let dminF := fp16ToF32 (Exp.shiftRight dmU32 (Exp.litU32 16))

      let sc0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) which (Exp.add blockU32Base (Exp.litU32 1))
      let sc1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) which (Exp.add blockU32Base (Exp.litU32 2))
      let sc2 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) which (Exp.add blockU32Base (Exp.litU32 3))

      let extractScaleMin (is : Exp (.scalar .u32)) : Exp (.scalar .f32) × Exp (.scalar .f32) :=
        let isLow := Exp.lt is (Exp.litU32 4)
        let shift4 := Exp.mul is (Exp.litU32 8)
        let scaleLow := Exp.bitAnd (Exp.shiftRight sc0 shift4) (Exp.litU32 0x3F)
        let minLow   := Exp.bitAnd (Exp.shiftRight sc1 shift4) (Exp.litU32 0x3F)
        let isHi := Exp.sub is (Exp.litU32 4)
        let shiftHi := Exp.mul isHi (Exp.litU32 8)
        let scaleHiLo := Exp.bitAnd (Exp.shiftRight sc2 shiftHi) (Exp.litU32 0x0F)
        let scaleHiHi := Exp.shiftLeft
          (Exp.bitAnd (Exp.shiftRight sc0 (Exp.add shiftHi (Exp.litU32 6))) (Exp.litU32 0x03))
          (Exp.litU32 4)
        let scaleHigh := Exp.bitOr scaleHiLo scaleHiHi
        let minHiLo := Exp.bitAnd (Exp.shiftRight sc2 (Exp.add shiftHi (Exp.litU32 4))) (Exp.litU32 0x0F)
        let minHiHi := Exp.shiftLeft
          (Exp.bitAnd (Exp.shiftRight sc1 (Exp.add shiftHi (Exp.litU32 6))) (Exp.litU32 0x03))
          (Exp.litU32 4)
        let minHigh := Exp.bitOr minHiLo minHiHi
        let scaleU := Exp.select isLow scaleLow scaleHigh
        let minU   := Exp.select isLow minLow   minHigh
        (Exp.toF32 scaleU, Exp.toF32 minU)

      let (scA, mA) := extractScaleMin bq8Off
      let (scB, mB) := extractScaleMin (Exp.add bq8Off (Exp.litU32 1))

      let q4BaseIdx := Exp.add blockU32Base
        (Exp.add (Exp.litU32 4) (Exp.add (Exp.mul bq8Off (Exp.litU32 4)) elemOff))
      let v0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) which q4BaseIdx
      let v1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) which (Exp.add q4BaseIdx (Exp.litU32 4))

      let v0i0 := Exp.bitAnd v0 (Exp.litU32 0x0F0F0F0F)
      let v1i0 := Exp.bitAnd v1 (Exp.litU32 0x0F0F0F0F)
      let acc0 := Exp.dot4I8Packed v0i0 u0
      let dot1_0 := Exp.dot4I8Packed v1i0 u1
      let dot1_0Combined := Exp.add acc0 dot1_0
      let sumU_0 := Exp.add (Exp.dot4I8Packed (Exp.litU32 0x01010101) u0)
                            (Exp.dot4I8Packed (Exp.litU32 0x01010101) u1)
      let sumfD_0 := Exp.mul d8A (Exp.mul (Exp.toF32 dot1_0Combined) scA)
      let sumfM_0 := Exp.mul d8A (Exp.mul (Exp.toF32 sumU_0) mA)

      let v0i1 := Exp.bitAnd (Exp.shiftRight v0 (Exp.litU32 4)) (Exp.litU32 0x0F0F0F0F)
      let v1i1 := Exp.bitAnd (Exp.shiftRight v1 (Exp.litU32 4)) (Exp.litU32 0x0F0F0F0F)
      let acc1 := Exp.dot4I8Packed v0i1 u2
      let dot1_1 := Exp.dot4I8Packed v1i1 u3
      let dot1_1Combined := Exp.add acc1 dot1_1
      let sumU_1 := Exp.add (Exp.dot4I8Packed (Exp.litU32 0x01010101) u2)
                            (Exp.dot4I8Packed (Exp.litU32 0x01010101) u3)
      let sumfD_1 := Exp.mul d8B (Exp.mul (Exp.toF32 dot1_1Combined) scB)
      let sumfM_1 := Exp.mul d8B (Exp.mul (Exp.toF32 sumU_1) mB)

      let blockSumfD := Exp.add sumfD_0 sumfD_1
      let blockSumfM := Exp.add sumfM_0 sumfM_1
      let blockContrib := Exp.sub (Exp.mul dF blockSumfD) (Exp.mul dminF blockSumfM)
      pure (Exp.add acc blockContrib)

    -- Q8_1 input read once; shared by both gate and up.
    let q8Sub0Base := Exp.add (Exp.mul blockIdx (Exp.litU32 (8 * 9))) (Exp.mul bq8Off (Exp.litU32 9))
    let q8Sub1Base := Exp.add q8Sub0Base (Exp.litU32 9)
    let u0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" (Exp.add q8Sub0Base (Exp.add (Exp.litU32 1) elemOff))
    let u1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" (Exp.add q8Sub0Base (Exp.add (Exp.litU32 5) elemOff))
    let u2 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" (Exp.add q8Sub1Base (Exp.add (Exp.litU32 1) elemOff))
    let u3 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" (Exp.add q8Sub1Base (Exp.add (Exp.litU32 5) elemOff))
    let q8Hdr0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" q8Sub0Base
    let q8Hdr1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" q8Sub1Base
    let d8A : Exp (.scalar .f32) := Exp.bitcast q8Hdr0
    let d8B : Exp (.scalar .f32) := Exp.bitcast q8Hdr1

    let newAccG ← processWeight "weights_gate" accG u0 u1 u2 u3 d8A d8B
    ShaderM.assign "accG" newAccG
    let newAccU ← processWeight "weights_up" accU u0 u1 u2 u3 d8A d8B
    ShaderM.assign "accU" newAccU

  -- Subgroup reduction for each accumulator (duplicate-work correction ×0.5).
  ShaderM.varNamed "totalG" (.scalar .f32)
    (Exp.mul (Exp.subgroupAdd accG) (Exp.litF32 0.5))
  ShaderM.varNamed "totalU" (.scalar .f32)
    (Exp.mul (Exp.subgroupAdd accU) (Exp.litF32 0.5))
  let totalG : Exp (.scalar .f32) := Exp.var "totalG"
  let totalU : Exp (.scalar .f32) := Exp.var "totalU"

  -- GELU(tanh) * up, written by lane 0 (matches llama.cpp's approximation
  -- and FusedFFNSpec.geluTanh).
  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    let sqrt2OverPi := Exp.litF32 0.7978845608028654
    let z := totalG
    let z3 := Exp.mul (Exp.mul z z) z
    let inner := Exp.mul sqrt2OverPi (Exp.add z (Exp.mul (Exp.litF32 0.044715) z3))
    let gelu := Exp.mul (Exp.mul (Exp.litF32 0.5) z) (Exp.add (Exp.litF32 1.0) (Exp.tanh inner))
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx (Exp.mul gelu totalU)
  ) (pure ())

/-- Fused Q4_K dp4a wK + wV.

    Computes `k[i] = dot(W_k[i], x_q8)` and `v[i] = dot(W_v[i], x_q8)` in
    a single kernel, writing to two separate output buffers.  Saves one
    full pass over the Q8_1 input + halves the per-layer dispatch count
    for attention KV projection (was 2 kernels → 1 kernel).

    Structure identical to `fusedQ4KMGateUpDP4AKernel` minus GELU: each
    WG computes one output row against both weight buffers sharing the
    same Q8_1 header/quant decode.

    Buffers: weights_k, weights_v (same shape), input_q8, output_k, output_v.
    Dispatch: (outDim, 1, 1) × 32 threads. -/
def fusedQ4KMKVDP4AKernel (config : Config) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid

  let blocksPerRow := config.inDim / 256
  let totalWeightU32 := config.outDim * blocksPerRow * 36
  let q8BlocksPerRow := config.inDim / 32
  let q8InputU32Size := q8BlocksPerRow * 9

  let _weightsK ← ShaderM.declareReadOnlyBuffer "weights_k" (.array (.scalar .u32) totalWeightU32)
  let _weightsV ← ShaderM.declareReadOnlyBuffer "weights_v" (.array (.scalar .u32) totalWeightU32)
  let _input    ← ShaderM.declareReadOnlyBuffer "input_q8" (.array (.scalar .u32) q8InputU32Size)
  let _outK     ← ShaderM.declareOutputBuffer "output_k" (.array (.scalar .f32) config.outDim)
  let _outV     ← ShaderM.declareOutputBuffer "output_v" (.array (.scalar .f32) config.outDim)

  let inBounds := Exp.lt outIdx (Exp.litU32 config.outDim)

  ShaderM.varNamed "accK" (.scalar .f32) (Exp.litF32 0.0)
  ShaderM.varNamed "accV" (.scalar .f32) (Exp.litF32 0.0)
  let accK : Exp (.scalar .f32) := Exp.var "accK"
  let accV : Exp (.scalar .f32) := Exp.var "accV"

  let rowBaseU32 := Exp.mul outIdx (Exp.litU32 (blocksPerRow * 36))

  let laneLow := Exp.bitAnd tid (Exp.litU32 15)
  let pairIdx := Exp.div laneLow (Exp.litU32 4)
  let elemOff := Exp.sub laneLow (Exp.mul pairIdx (Exp.litU32 4))
  let bq8Off := Exp.mul pairIdx (Exp.litU32 2)

  ShaderM.loop (Exp.litU32 0) (Exp.litU32 blocksPerRow) (Exp.litU32 1) fun blockIdx => do
    let blockU32Base := Exp.add rowBaseU32 (Exp.mul blockIdx (Exp.litU32 36))

    let processWeight (which : String) (acc : Exp (.scalar .f32))
        (u0 u1 u2 u3 : Exp (.scalar .u32)) (d8A d8B : Exp (.scalar .f32))
        : ShaderM (Exp (.scalar .f32)) := do
      let dmU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) which blockU32Base
      let dF := fp16ToF32 (Exp.bitAnd dmU32 (Exp.litU32 0xFFFF))
      let dminF := fp16ToF32 (Exp.shiftRight dmU32 (Exp.litU32 16))

      let sc0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) which (Exp.add blockU32Base (Exp.litU32 1))
      let sc1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) which (Exp.add blockU32Base (Exp.litU32 2))
      let sc2 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) which (Exp.add blockU32Base (Exp.litU32 3))

      let extractScaleMin (is : Exp (.scalar .u32)) : Exp (.scalar .f32) × Exp (.scalar .f32) :=
        let isLow := Exp.lt is (Exp.litU32 4)
        let shift4 := Exp.mul is (Exp.litU32 8)
        let scaleLow := Exp.bitAnd (Exp.shiftRight sc0 shift4) (Exp.litU32 0x3F)
        let minLow   := Exp.bitAnd (Exp.shiftRight sc1 shift4) (Exp.litU32 0x3F)
        let isHi := Exp.sub is (Exp.litU32 4)
        let shiftHi := Exp.mul isHi (Exp.litU32 8)
        let scaleHiLo := Exp.bitAnd (Exp.shiftRight sc2 shiftHi) (Exp.litU32 0x0F)
        let scaleHiHi := Exp.shiftLeft
          (Exp.bitAnd (Exp.shiftRight sc0 (Exp.add shiftHi (Exp.litU32 6))) (Exp.litU32 0x03))
          (Exp.litU32 4)
        let scaleHigh := Exp.bitOr scaleHiLo scaleHiHi
        let minHiLo := Exp.bitAnd (Exp.shiftRight sc2 (Exp.add shiftHi (Exp.litU32 4))) (Exp.litU32 0x0F)
        let minHiHi := Exp.shiftLeft
          (Exp.bitAnd (Exp.shiftRight sc1 (Exp.add shiftHi (Exp.litU32 6))) (Exp.litU32 0x03))
          (Exp.litU32 4)
        let minHigh := Exp.bitOr minHiLo minHiHi
        let scaleU := Exp.select isLow scaleLow scaleHigh
        let minU   := Exp.select isLow minLow   minHigh
        (Exp.toF32 scaleU, Exp.toF32 minU)

      let (scA, mA) := extractScaleMin bq8Off
      let (scB, mB) := extractScaleMin (Exp.add bq8Off (Exp.litU32 1))

      let q4BaseIdx := Exp.add blockU32Base
        (Exp.add (Exp.litU32 4) (Exp.add (Exp.mul bq8Off (Exp.litU32 4)) elemOff))
      let v0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) which q4BaseIdx
      let v1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) which (Exp.add q4BaseIdx (Exp.litU32 4))

      let v0i0 := Exp.bitAnd v0 (Exp.litU32 0x0F0F0F0F)
      let v1i0 := Exp.bitAnd v1 (Exp.litU32 0x0F0F0F0F)
      let acc0 := Exp.dot4I8Packed v0i0 u0
      let dot1_0 := Exp.dot4I8Packed v1i0 u1
      let dot1_0Combined := Exp.add acc0 dot1_0
      let sumU_0 := Exp.add (Exp.dot4I8Packed (Exp.litU32 0x01010101) u0)
                            (Exp.dot4I8Packed (Exp.litU32 0x01010101) u1)
      let sumfD_0 := Exp.mul d8A (Exp.mul (Exp.toF32 dot1_0Combined) scA)
      let sumfM_0 := Exp.mul d8A (Exp.mul (Exp.toF32 sumU_0) mA)

      let v0i1 := Exp.bitAnd (Exp.shiftRight v0 (Exp.litU32 4)) (Exp.litU32 0x0F0F0F0F)
      let v1i1 := Exp.bitAnd (Exp.shiftRight v1 (Exp.litU32 4)) (Exp.litU32 0x0F0F0F0F)
      let acc1 := Exp.dot4I8Packed v0i1 u2
      let dot1_1 := Exp.dot4I8Packed v1i1 u3
      let dot1_1Combined := Exp.add acc1 dot1_1
      let sumU_1 := Exp.add (Exp.dot4I8Packed (Exp.litU32 0x01010101) u2)
                            (Exp.dot4I8Packed (Exp.litU32 0x01010101) u3)
      let sumfD_1 := Exp.mul d8B (Exp.mul (Exp.toF32 dot1_1Combined) scB)
      let sumfM_1 := Exp.mul d8B (Exp.mul (Exp.toF32 sumU_1) mB)

      let blockSumfD := Exp.add sumfD_0 sumfD_1
      let blockSumfM := Exp.add sumfM_0 sumfM_1
      let blockContrib := Exp.sub (Exp.mul dF blockSumfD) (Exp.mul dminF blockSumfM)
      pure (Exp.add acc blockContrib)

    -- Q8_1 input shared between K and V.
    let q8Sub0Base := Exp.add (Exp.mul blockIdx (Exp.litU32 (8 * 9))) (Exp.mul bq8Off (Exp.litU32 9))
    let q8Sub1Base := Exp.add q8Sub0Base (Exp.litU32 9)
    let u0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" (Exp.add q8Sub0Base (Exp.add (Exp.litU32 1) elemOff))
    let u1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" (Exp.add q8Sub0Base (Exp.add (Exp.litU32 5) elemOff))
    let u2 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" (Exp.add q8Sub1Base (Exp.add (Exp.litU32 1) elemOff))
    let u3 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" (Exp.add q8Sub1Base (Exp.add (Exp.litU32 5) elemOff))
    let q8Hdr0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" q8Sub0Base
    let q8Hdr1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" q8Sub1Base
    let d8A : Exp (.scalar .f32) := Exp.bitcast q8Hdr0
    let d8B : Exp (.scalar .f32) := Exp.bitcast q8Hdr1

    let newAccK ← processWeight "weights_k" accK u0 u1 u2 u3 d8A d8B
    ShaderM.assign "accK" newAccK
    let newAccV ← processWeight "weights_v" accV u0 u1 u2 u3 d8A d8B
    ShaderM.assign "accV" newAccV

  -- Subgroup reduction for each accumulator (×0.5 duplicate-work correction).
  ShaderM.varNamed "totalK" (.scalar .f32)
    (Exp.mul (Exp.subgroupAdd accK) (Exp.litF32 0.5))
  ShaderM.varNamed "totalV" (.scalar .f32)
    (Exp.mul (Exp.subgroupAdd accV) (Exp.litF32 0.5))
  let totalK : Exp (.scalar .f32) := Exp.var "totalK"
  let totalV : Exp (.scalar .f32) := Exp.var "totalV"

  -- Lane 0 writes both outputs.
  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    ShaderM.writeBuffer (ty := .scalar .f32) "output_k" outIdx totalK
    ShaderM.writeBuffer (ty := .scalar .f32) "output_v" outIdx totalV
  ) (pure ())

/-- 4-warp cooperative variant of `fusedQ4KMGateUpDP4AKernel`.

    1 WG = 128 threads = 4 warps = 4 output rows. All 128 threads
    cooperatively load the Q8_1 input into shared memory, then each
    warp independently computes one row's (gate, up) dot products and
    writes GELU(gate) × up. The smem input is reused by 4 warps → 4×
    reduction in global-memory input traffic vs the 1-row variant.

    Weight traffic is unchanged (still 4 rows × 2 buffers per WG), so
    the win comes from (a) halved input BW, (b) fewer dispatch /
    launch overheads, and (c) better SM occupancy for small outDim.

    Dispatch: (⌈outDim / 4⌉, 1, 1) × 128 threads. -/
def fusedQ4KMGateUpDP4A4RowKernel (config : Config) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let quadIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid                           -- 0..127
  let subgroupId := Exp.shiftRight tid (Exp.litU32 5) -- 0..3 (row in quad)
  let laneId := Exp.bitAnd tid (Exp.litU32 31)        -- 0..31 (dp4a lane)
  let outIdx := Exp.add (Exp.mul quadIdx (Exp.litU32 4)) subgroupId

  let blocksPerRow := config.inDim / 256
  let totalWeightU32 := config.outDim * blocksPerRow * 36
  let q8BlocksPerRow := config.inDim / 32
  let q8InputU32Size := q8BlocksPerRow * 9

  let _weightsGate ← ShaderM.declareReadOnlyBuffer "weights_gate" (.array (.scalar .u32) totalWeightU32)
  let _weightsUp   ← ShaderM.declareReadOnlyBuffer "weights_up"   (.array (.scalar .u32) totalWeightU32)
  let _input       ← ShaderM.declareReadOnlyBuffer "input_q8"     (.array (.scalar .u32) q8InputU32Size)
  let _output      ← ShaderM.declareOutputBuffer "output"         (.array (.scalar .f32) config.outDim)

  -- Cooperative smem staging of Q8_1 input: 128 threads load q8InputU32Size
  -- words total, ~6 per lane for Gemma 4's 2560-wide hidden state.
  ShaderM.sharedNamed "s_input_q8" (.array (.scalar .u32) q8InputU32Size)
  let perThread : Nat := (q8InputU32Size + 127) / 128
  for i in [0 : perThread] do
    let idx := Exp.add tid (Exp.litU32 (i * 128))
    ShaderM.if_ (Exp.lt idx (Exp.litU32 q8InputU32Size)) (do
      let v ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" idx
      ShaderM.writeWorkgroup (ty := .scalar .u32) "s_input_q8" idx v
    ) (pure ())
  ShaderM.barrier

  let inBounds := Exp.lt outIdx (Exp.litU32 config.outDim)

  ShaderM.varNamed "accG" (.scalar .f32) (Exp.litF32 0.0)
  ShaderM.varNamed "accU" (.scalar .f32) (Exp.litF32 0.0)
  let accG : Exp (.scalar .f32) := Exp.var "accG"
  let accU : Exp (.scalar .f32) := Exp.var "accU"

  let rowBaseU32 := Exp.mul outIdx (Exp.litU32 (blocksPerRow * 36))

  -- Lane decomposition uses laneId (not tid) — same as 1-row kernel.
  let laneLow := Exp.bitAnd laneId (Exp.litU32 15)
  let pairIdx := Exp.div laneLow (Exp.litU32 4)
  let elemOff := Exp.sub laneLow (Exp.mul pairIdx (Exp.litU32 4))
  let bq8Off := Exp.mul pairIdx (Exp.litU32 2)

  ShaderM.loop (Exp.litU32 0) (Exp.litU32 blocksPerRow) (Exp.litU32 1) fun blockIdx => do
    let blockU32Base := Exp.add rowBaseU32 (Exp.mul blockIdx (Exp.litU32 36))

    let processWeight (which : String) (acc : Exp (.scalar .f32))
        (u0 u1 u2 u3 : Exp (.scalar .u32)) (d8A d8B : Exp (.scalar .f32))
        : ShaderM (Exp (.scalar .f32)) := do
      let dmU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) which blockU32Base
      let dF := fp16ToF32 (Exp.bitAnd dmU32 (Exp.litU32 0xFFFF))
      let dminF := fp16ToF32 (Exp.shiftRight dmU32 (Exp.litU32 16))

      let sc0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) which (Exp.add blockU32Base (Exp.litU32 1))
      let sc1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) which (Exp.add blockU32Base (Exp.litU32 2))
      let sc2 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) which (Exp.add blockU32Base (Exp.litU32 3))

      let extractScaleMin (is : Exp (.scalar .u32)) : Exp (.scalar .f32) × Exp (.scalar .f32) :=
        let isLow := Exp.lt is (Exp.litU32 4)
        let shift4 := Exp.mul is (Exp.litU32 8)
        let scaleLow := Exp.bitAnd (Exp.shiftRight sc0 shift4) (Exp.litU32 0x3F)
        let minLow   := Exp.bitAnd (Exp.shiftRight sc1 shift4) (Exp.litU32 0x3F)
        let isHi := Exp.sub is (Exp.litU32 4)
        let shiftHi := Exp.mul isHi (Exp.litU32 8)
        let scaleHiLo := Exp.bitAnd (Exp.shiftRight sc2 shiftHi) (Exp.litU32 0x0F)
        let scaleHiHi := Exp.shiftLeft
          (Exp.bitAnd (Exp.shiftRight sc0 (Exp.add shiftHi (Exp.litU32 6))) (Exp.litU32 0x03))
          (Exp.litU32 4)
        let scaleHigh := Exp.bitOr scaleHiLo scaleHiHi
        let minHiLo := Exp.bitAnd (Exp.shiftRight sc2 (Exp.add shiftHi (Exp.litU32 4))) (Exp.litU32 0x0F)
        let minHiHi := Exp.shiftLeft
          (Exp.bitAnd (Exp.shiftRight sc1 (Exp.add shiftHi (Exp.litU32 6))) (Exp.litU32 0x03))
          (Exp.litU32 4)
        let minHigh := Exp.bitOr minHiLo minHiHi
        let scaleU := Exp.select isLow scaleLow scaleHigh
        let minU   := Exp.select isLow minLow   minHigh
        (Exp.toF32 scaleU, Exp.toF32 minU)

      let (scA, mA) := extractScaleMin bq8Off
      let (scB, mB) := extractScaleMin (Exp.add bq8Off (Exp.litU32 1))

      let q4BaseIdx := Exp.add blockU32Base
        (Exp.add (Exp.litU32 4) (Exp.add (Exp.mul bq8Off (Exp.litU32 4)) elemOff))
      let v0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) which q4BaseIdx
      let v1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) which (Exp.add q4BaseIdx (Exp.litU32 4))

      let v0i0 := Exp.bitAnd v0 (Exp.litU32 0x0F0F0F0F)
      let v1i0 := Exp.bitAnd v1 (Exp.litU32 0x0F0F0F0F)
      let acc0 := Exp.dot4I8Packed v0i0 u0
      let dot1_0 := Exp.dot4I8Packed v1i0 u1
      let dot1_0Combined := Exp.add acc0 dot1_0
      let sumU_0 := Exp.add (Exp.dot4I8Packed (Exp.litU32 0x01010101) u0)
                            (Exp.dot4I8Packed (Exp.litU32 0x01010101) u1)
      let sumfD_0 := Exp.mul d8A (Exp.mul (Exp.toF32 dot1_0Combined) scA)
      let sumfM_0 := Exp.mul d8A (Exp.mul (Exp.toF32 sumU_0) mA)

      let v0i1 := Exp.bitAnd (Exp.shiftRight v0 (Exp.litU32 4)) (Exp.litU32 0x0F0F0F0F)
      let v1i1 := Exp.bitAnd (Exp.shiftRight v1 (Exp.litU32 4)) (Exp.litU32 0x0F0F0F0F)
      let acc1 := Exp.dot4I8Packed v0i1 u2
      let dot1_1 := Exp.dot4I8Packed v1i1 u3
      let dot1_1Combined := Exp.add acc1 dot1_1
      let sumU_1 := Exp.add (Exp.dot4I8Packed (Exp.litU32 0x01010101) u2)
                            (Exp.dot4I8Packed (Exp.litU32 0x01010101) u3)
      let sumfD_1 := Exp.mul d8B (Exp.mul (Exp.toF32 dot1_1Combined) scB)
      let sumfM_1 := Exp.mul d8B (Exp.mul (Exp.toF32 sumU_1) mB)

      let blockSumfD := Exp.add sumfD_0 sumfD_1
      let blockSumfM := Exp.add sumfM_0 sumfM_1
      let blockContrib := Exp.sub (Exp.mul dF blockSumfD) (Exp.mul dminF blockSumfM)
      pure (Exp.add acc blockContrib)

    -- Q8_1 input from smem (shared across 4 warps in this WG).
    let q8Sub0Base := Exp.add (Exp.mul blockIdx (Exp.litU32 (8 * 9))) (Exp.mul bq8Off (Exp.litU32 9))
    let q8Sub1Base := Exp.add q8Sub0Base (Exp.litU32 9)
    let u0 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := q8InputU32Size) "s_input_q8" (Exp.add q8Sub0Base (Exp.add (Exp.litU32 1) elemOff))
    let u1 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := q8InputU32Size) "s_input_q8" (Exp.add q8Sub0Base (Exp.add (Exp.litU32 5) elemOff))
    let u2 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := q8InputU32Size) "s_input_q8" (Exp.add q8Sub1Base (Exp.add (Exp.litU32 1) elemOff))
    let u3 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := q8InputU32Size) "s_input_q8" (Exp.add q8Sub1Base (Exp.add (Exp.litU32 5) elemOff))
    let q8Hdr0 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := q8InputU32Size) "s_input_q8" q8Sub0Base
    let q8Hdr1 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := q8InputU32Size) "s_input_q8" q8Sub1Base
    let d8A : Exp (.scalar .f32) := Exp.bitcast q8Hdr0
    let d8B : Exp (.scalar .f32) := Exp.bitcast q8Hdr1

    let newAccG ← processWeight "weights_gate" accG u0 u1 u2 u3 d8A d8B
    ShaderM.assign "accG" newAccG
    let newAccU ← processWeight "weights_up" accU u0 u1 u2 u3 d8A d8B
    ShaderM.assign "accU" newAccU

  -- Per-warp subgroup reductions (duplicate-work correction ×0.5).
  ShaderM.varNamed "totalG" (.scalar .f32)
    (Exp.mul (Exp.subgroupAdd accG) (Exp.litF32 0.5))
  ShaderM.varNamed "totalU" (.scalar .f32)
    (Exp.mul (Exp.subgroupAdd accU) (Exp.litF32 0.5))
  let totalG : Exp (.scalar .f32) := Exp.var "totalG"
  let totalU : Exp (.scalar .f32) := Exp.var "totalU"

  -- Each warp's lane 0 writes its own row.
  ShaderM.if_ (Exp.and (Exp.eq laneId (Exp.litU32 0)) inBounds) (do
    let sqrt2OverPi := Exp.litF32 0.7978845608028654
    let z := totalG
    let z3 := Exp.mul (Exp.mul z z) z
    let inner := Exp.mul sqrt2OverPi (Exp.add z (Exp.mul (Exp.litF32 0.044715) z3))
    let gelu := Exp.mul (Exp.mul (Exp.litF32 0.5) z) (Exp.add (Exp.litF32 1.0) (Exp.tanh inner))
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx (Exp.mul gelu totalU)
  ) (pure ())

/-! ## Q4_K × Q8_1 dp4a, 2 rows per workgroup (high occupancy for ffnDown) -/

/-- 2-rows-per-workgroup variant of `fusedQ4KMLinearDP4AKernel`.

    Motivation: ffnDown (10240→2560) launches only 2560 workgroups × 32
    threads = 0.89 waves on RTX 4070 Ti (40 SM × ~1536 resident threads
    ≈ 92K capacity). Less than one wave means the SM scheduler can't
    swap in another warp when the current one stalls on memory.

    This variant uses 64 threads (2 subgroups of 32) per workgroup, with
    each subgroup computing one of two adjacent rows. Both subgroups
    execute the same dp4a matmul logic on different weight rows;
    intra-subgroup reduction is independent (subgroup-local subgroupAdd),
    so no cross-warp barrier is needed.

    Dispatch: (outDim/2, 1, 1) workgroups × 64 threads.
    Requires outDim % 2 == 0 (always true for Gemma 4 shapes). -/
def fusedQ4KMLinearDP4A2RowKernel (config : Config) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let pairIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid

  -- Decompose 64 threads into 2 subgroups of 32:
  --   subgroupId = tid / 32   (0 or 1, picks which row)
  --   laneId     = tid % 32   (0..31, the dp4a lane within row)
  let subgroupId := Exp.shiftRight tid (Exp.litU32 5)
  let laneId := Exp.bitAnd tid (Exp.litU32 31)
  let outIdx := Exp.add (Exp.mul pairIdx (Exp.litU32 2)) subgroupId

  let blocksPerRow := config.inDim / 256
  let totalWeightU32 := config.outDim * blocksPerRow * 36
  let q8BlocksPerRow := config.inDim / 32
  let q8InputU32Size := q8BlocksPerRow * 9

  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareReadOnlyBuffer "input_q8" (.array (.scalar .u32) q8InputU32Size)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) config.outDim)

  let inBounds := Exp.lt outIdx (Exp.litU32 config.outDim)

  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"

  -- Use laneId (not tid) for intra-row work distribution; matches single-row kernel.
  let laneLow := Exp.bitAnd laneId (Exp.litU32 15)
  let pairIdxInRow := Exp.div laneLow (Exp.litU32 4)
  let elemOff := Exp.sub laneLow (Exp.mul pairIdxInRow (Exp.litU32 4))
  let bq8Off := Exp.mul pairIdxInRow (Exp.litU32 2)

  let rowBaseU32 := Exp.mul outIdx (Exp.litU32 (blocksPerRow * 36))

  ShaderM.loop (Exp.litU32 0) (Exp.litU32 blocksPerRow) (Exp.litU32 1) fun blockIdx => do
    let blockU32Base := Exp.add rowBaseU32 (Exp.mul blockIdx (Exp.litU32 36))
    let dmU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" blockU32Base
    let dF := fp16ToF32 (Exp.bitAnd dmU32 (Exp.litU32 0xFFFF))
    let dminF := fp16ToF32 (Exp.shiftRight dmU32 (Exp.litU32 16))

    let sc0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add blockU32Base (Exp.litU32 1))
    let sc1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add blockU32Base (Exp.litU32 2))
    let sc2 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add blockU32Base (Exp.litU32 3))

    let extractScaleMin (is : Exp (.scalar .u32)) : Exp (.scalar .f32) × Exp (.scalar .f32) :=
      let isLow := Exp.lt is (Exp.litU32 4)
      let shift4 := Exp.mul is (Exp.litU32 8)
      let scaleLow := Exp.bitAnd (Exp.shiftRight sc0 shift4) (Exp.litU32 0x3F)
      let minLow   := Exp.bitAnd (Exp.shiftRight sc1 shift4) (Exp.litU32 0x3F)
      let isHi := Exp.sub is (Exp.litU32 4)
      let shiftHi := Exp.mul isHi (Exp.litU32 8)
      let scaleHiLo := Exp.bitAnd (Exp.shiftRight sc2 shiftHi) (Exp.litU32 0x0F)
      let scaleHiHi := Exp.shiftLeft
        (Exp.bitAnd (Exp.shiftRight sc0 (Exp.add shiftHi (Exp.litU32 6))) (Exp.litU32 0x03))
        (Exp.litU32 4)
      let scaleHigh := Exp.bitOr scaleHiLo scaleHiHi
      let minHiLo := Exp.bitAnd (Exp.shiftRight sc2 (Exp.add shiftHi (Exp.litU32 4))) (Exp.litU32 0x0F)
      let minHiHi := Exp.shiftLeft
        (Exp.bitAnd (Exp.shiftRight sc1 (Exp.add shiftHi (Exp.litU32 6))) (Exp.litU32 0x03))
        (Exp.litU32 4)
      let minHigh := Exp.bitOr minHiLo minHiHi
      let scaleU := Exp.select isLow scaleLow scaleHigh
      let minU   := Exp.select isLow minLow   minHigh
      (Exp.toF32 scaleU, Exp.toF32 minU)

    let (scA, mA) := extractScaleMin bq8Off
    let (scB, mB) := extractScaleMin (Exp.add bq8Off (Exp.litU32 1))

    let q4BaseIdx := Exp.add blockU32Base
      (Exp.add (Exp.litU32 4) (Exp.add (Exp.mul bq8Off (Exp.litU32 4)) elemOff))
    let v0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" q4BaseIdx
    let v1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add q4BaseIdx (Exp.litU32 4))

    let q8Sub0Base := Exp.add (Exp.mul blockIdx (Exp.litU32 (8 * 9))) (Exp.mul bq8Off (Exp.litU32 9))
    let q8Sub1Base := Exp.add q8Sub0Base (Exp.litU32 9)

    let u0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" (Exp.add q8Sub0Base (Exp.add (Exp.litU32 1) elemOff))
    let u1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" (Exp.add q8Sub0Base (Exp.add (Exp.litU32 5) elemOff))
    let u2 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" (Exp.add q8Sub1Base (Exp.add (Exp.litU32 1) elemOff))
    let u3 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" (Exp.add q8Sub1Base (Exp.add (Exp.litU32 5) elemOff))

    let q8Hdr0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" q8Sub0Base
    let q8Hdr1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" q8Sub1Base
    let d8A : Exp (.scalar .f32) := Exp.bitcast q8Hdr0
    let d8B : Exp (.scalar .f32) := Exp.bitcast q8Hdr1

    -- i=0: vl & 0x0F
    let v0i0 := Exp.bitAnd v0 (Exp.litU32 0x0F0F0F0F)
    let v1i0 := Exp.bitAnd v1 (Exp.litU32 0x0F0F0F0F)
    let acc0 := Exp.dot4I8Packed v0i0 u0
    let dot1_0 := Exp.dot4I8Packed v1i0 u1
    let dot1_0Combined := Exp.add acc0 dot1_0
    let sumU_0 := Exp.add (Exp.dot4I8Packed (Exp.litU32 0x01010101) u0)
                          (Exp.dot4I8Packed (Exp.litU32 0x01010101) u1)
    let sumfD_0 := Exp.mul d8A (Exp.mul (Exp.toF32 dot1_0Combined) scA)
    let sumfM_0 := Exp.mul d8A (Exp.mul (Exp.toF32 sumU_0) mA)

    -- i=1: vl >> 4
    let v0i1 := Exp.bitAnd (Exp.shiftRight v0 (Exp.litU32 4)) (Exp.litU32 0x0F0F0F0F)
    let v1i1 := Exp.bitAnd (Exp.shiftRight v1 (Exp.litU32 4)) (Exp.litU32 0x0F0F0F0F)
    let acc1 := Exp.dot4I8Packed v0i1 u2
    let dot1_1 := Exp.dot4I8Packed v1i1 u3
    let dot1_1Combined := Exp.add acc1 dot1_1
    let sumU_1 := Exp.add (Exp.dot4I8Packed (Exp.litU32 0x01010101) u2)
                          (Exp.dot4I8Packed (Exp.litU32 0x01010101) u3)
    let sumfD_1 := Exp.mul d8B (Exp.mul (Exp.toF32 dot1_1Combined) scB)
    let sumfM_1 := Exp.mul d8B (Exp.mul (Exp.toF32 sumU_1) mB)

    let blockSumfD := Exp.add sumfD_0 sumfD_1
    let blockSumfM := Exp.add sumfM_0 sumfM_1
    let blockContrib := Exp.sub (Exp.mul dF blockSumfD) (Exp.mul dminF blockSumfM)
    ShaderM.assign "acc" (Exp.add acc blockContrib)

  -- Each subgroup reduces independently. *0.5 because lanes 0..15 and 16..31
  -- of each subgroup compute duplicate work (matches single-row kernel).
  ShaderM.varNamed "total" (.scalar .f32)
    (Exp.mul (Exp.subgroupAdd acc) (Exp.litF32 0.5))
  let total : Exp (.scalar .f32) := Exp.var "total"

  -- Lane 0 of each subgroup writes its row.
  ShaderM.if_ (Exp.and (Exp.eq laneId (Exp.litU32 0)) inBounds) (do
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx total
  ) (pure ())

/-! ## Q6_K × Q8_1 dp4a MatVec (lmHead) -/

/-- Q6_K × Q8_1 mat-vec using dp4a.

    llama.cpp's `vec_dot_q6_K_q8_1_impl_mmvq` algorithm.
    Each lane (of 32 in the subgroup) processes iqs = lane*1 = lane (since VDR_Q6_K_Q8_1_MMVQ=1).
    Only lanes 0..15 do useful work (QI6_K/VDR = 16), lanes 16..31 duplicate.

    Q6_K block layout (210 bytes per 256 elements):
      bytes [0, 128): ql[128] — 4-bit lower quants (2 elements per byte)
      bytes [128, 192): qh[64] — 2-bit upper quants (4 elements per byte)
      bytes [192, 208): scales[16] — signed int8 scales
      bytes [208, 210): d — fp16 super-block scale

    Algorithm per lane (iqs = 0..15):
      bq8_offset = 4*(iqs/8) + (iqs%8)/4         ∈ {0, 1, 4, 5}
      scale_offset = 4*(iqs/8) + (iqs%8)/2       ∈ {0, 1, 2, 3, 4, 5, 6, 7}
      vh_shift = 2 * ((iqs%8)/4)                 ∈ {0, 2}

      vl = ql as int[] @ index iqs                 (4 bytes of packed lower 4-bit nibbles)
      vh = qh as int[] @ index (2*(iqs/8) + (iqs%4)), shifted right by vh_shift
      u[0] = q8[bq8_offset].qs   @ iqs%8
      u[1] = q8[bq8_offset+2].qs @ iqs%8
      d8[0] = q8[bq8_offset].d,  d8[1] = q8[bq8_offset+2].d
      scale[0] = scales[scale_offset + 0]
      scale[1] = scales[scale_offset + 4]
      For i ∈ {0, 1}:
        vil = (vl >> (4*i)) & 0x0F0F0F0F
        vih = ((vh >> (4*i)) << 4) & 0x30303030
        vi  = vsub_s8(vil | vih, 0x20202020)        (signed subtraction: subtract 32 per byte)
        sumf += d8[i] * scale[i] * dp4a(vi, u[i], 0)
      lane_result = d * sumf
    Subgroup reduce → output[outIdx].

    @param inDim Input dimension
    @param outDim Output dimension (often 262144 for Gemma 4 lmHead)
    @param gridX Grid X dimension for 2D grid (≤ 65535 WebGPU limit); 0 for 1D. -/
def fusedQ6KLinearDP4AKernel (inDim outDim : Nat) (gridX : Nat := 0) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx :=
    if gridX == 0 then Exp.vec3X wid
    else Exp.add (Exp.vec3X wid) (Exp.mul (Exp.vec3Y wid) (Exp.litU32 gridX))
  let tid := Exp.vec3X lid

  let blocksPerRow := inDim / 256
  let blockSizeBytes : Nat := 210
  let totalWeightBytes := outDim * blocksPerRow * blockSizeBytes
  let totalWeightU32 := (totalWeightBytes + 3) / 4
  let q8BlocksPerRow := inDim / 32
  let q8InputU32Size := q8BlocksPerRow * 9

  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareReadOnlyBuffer "input_q8" (.array (.scalar .u32) q8InputU32Size)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) outDim)

  let inBounds := Exp.lt outIdx (Exp.litU32 outDim)

  -- QI6_K = QK_K / (4*QR6_K) = 256 / 8 = 32.
  -- iqs ∈ [0, 32) — each lane processes a unique (kbx, kqs) combo; llama.cpp
  -- uses all 32 lanes for Q6_K (VDR=1, qi=32, tid%(qi/vdr)=tid%32).
  let iqs : Exp (.scalar .u32) := tid
  -- bq8_offset = 2*QR6_K*(iqs / (QI6_K/2)) + (iqs%(QI6_K/2)) / (QI6_K/4)
  --            = 4*(iqs/16) + (iqs%16)/8
  let iqsDiv16 := Exp.shiftRight iqs (Exp.litU32 4)
  let iqsMod16 := Exp.bitAnd iqs (Exp.litU32 15)
  let bq8Off := Exp.add (Exp.mul iqsDiv16 (Exp.litU32 4)) (Exp.shiftRight iqsMod16 (Exp.litU32 3))
  -- scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs%(QI6_K/2)) / (QI6_K/8)
  --              = 8*(iqs/16) + (iqs%16)/4
  let scaleOff := Exp.add (Exp.mul iqsDiv16 (Exp.litU32 8)) (Exp.shiftRight iqsMod16 (Exp.litU32 2))
  -- vh_shift = 2 * ((iqs%(QI6_K/2)) / (QI6_K/4)) = 2 * ((iqs%16)/8)
  let vhShift := Exp.mul (Exp.shiftRight iqsMod16 (Exp.litU32 3)) (Exp.litU32 2)
  -- vh_idx = (QI6_K/4) * (iqs/(QI6_K/2)) + iqs%(QI6_K/4)
  --        = 8*(iqs/16) + iqs%8
  let iqsMod8 := Exp.bitAnd iqs (Exp.litU32 7)
  let vhIdx := Exp.add (Exp.mul iqsDiv16 (Exp.litU32 8)) iqsMod8
  -- q8 element offset within sub-block = iqs % QI8_1 = iqs % 8
  let q8ElemOff := iqsMod8

  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"

  -- Bind `outIdx * 2100` to a DSL variable so it materialises once in PTX
  -- (ShaderM re-evaluates Exp structures every read otherwise → ~50 copies
  -- of `mov.u32 %r, 2100; mul.lo.u32 %r2, %outIdx, %r` in the generated PTX).
  ShaderM.varNamed "rowByteBase" (.scalar .u32)
    (Exp.mul outIdx (Exp.litU32 (blocksPerRow * blockSizeBytes)))
  let rowByteBase : Exp (.scalar .u32) := Exp.var "rowByteBase"

  -- Byte-read helper (Q6_K block is 210 bytes, not a multiple of 4).
  let readByte (blockBase : Exp (.scalar .u32)) (offset : Exp (.scalar .u32)) : ShaderM (Exp (.scalar .u32)) := do
    let byteIdx := Exp.add blockBase offset
    let u32Idx := Exp.shiftRight byteIdx (Exp.litU32 2)
    let byteShift := Exp.mul (Exp.bitAnd byteIdx (Exp.litU32 3)) (Exp.litU32 8)
    let u32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" u32Idx
    pure (Exp.bitAnd (Exp.shiftRight u32 byteShift) (Exp.litU32 0xFF))

  -- Read 4 consecutive bytes starting at `base + offset` and pack into one u32
  -- (little-endian: byte[base+offset] in lowest 8 bits).
  --
  -- Q6_K blocks are 210 bytes (not a multiple of 4), so `blockBase + offset`
  -- may cross a u32 boundary.  We load at most 2 u32 words and stitch them,
  -- instead of the old 4 × readByte (4 u32 loads + 16 shift/mask ops).
  --
  --   byteIdx = blockBase + offset
  --   u32Idx  = byteIdx >> 2
  --   byteOff = byteIdx & 3          (0..3)
  --   lo = weights[u32Idx] >> (byteOff * 8)
  --   hi = weights[u32Idx + 1] << ((4 - byteOff) * 8)   -- only when byteOff ≠ 0
  --   result = lo | hi
  --
  -- When byteOff == 0 the load is aligned and hi must be zero; the `select`
  -- guards against an undefined shl-by-32.
  let read4Bytes (blockBase : Exp (.scalar .u32)) (offset : Exp (.scalar .u32)) : ShaderM (Exp (.scalar .u32)) := do
    let byteIdx := Exp.add blockBase offset
    let u32Idx := Exp.shiftRight byteIdx (Exp.litU32 2)
    let byteOff := Exp.bitAnd byteIdx (Exp.litU32 3)
    let shiftLo := Exp.mul byteOff (Exp.litU32 8)
    let shiftHi := Exp.mul (Exp.sub (Exp.litU32 4) byteOff) (Exp.litU32 8)
    let w0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" u32Idx
    let w1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights"
              (Exp.add u32Idx (Exp.litU32 1))
    let lo := Exp.shiftRight w0 shiftLo
    let hi := Exp.select (Exp.eq byteOff (Exp.litU32 0))
                (Exp.litU32 0) (Exp.shiftLeft w1 shiftHi)
    pure (Exp.bitOr lo hi)

  -- Per-byte signed subtract-32: emits the per-byte (x - 32) without cross-byte
  -- borrow. Bit-parallel trick:
  --   - split x into high-bit (0x80 per byte) and low 7 bits (0x7F per byte)
  --   - if we bias both operands up by 0x80 per byte, sub wraps cleanly within each byte
  --   - (x ^ 0x80) - 0xA0  = (x + 0x80 - 0x20 - 0x80) mod 256 per byte = x - 0x20 per byte
  --     ^ 0x80 at end re-centers.
  -- Equivalent identity:  (x | 0x80) - 0x20   produces correct two's-complement
  -- byte values for x ∈ [0, 63], because the borrow from bit 7 never propagates
  -- (bit 7 is forced ON before the sub, so bit 7 - bit 5 borrows into bit 7 only,
  -- never into the next byte).
  -- Then flip bit 7 back:  result = ((x | 0x80) - 0x20) ^ 0x80.
  -- Verification:
  --   x = 0x00: (0x80) - 0x20 = 0x60; ^ 0x80 = 0xE0 = -32 ✓
  --   x = 0x1F: (0x9F) - 0x20 = 0x7F; ^ 0x80 = 0xFF = -1  ✓
  --   x = 0x20: (0xA0) - 0x20 = 0x80; ^ 0x80 = 0x00 = 0   ✓
  --   x = 0x3F: (0xBF) - 0x20 = 0x9F; ^ 0x80 = 0x1F = 31  ✓
  let sub32PerByte (x : Exp (.scalar .u32)) : Exp (.scalar .u32) :=
    Exp.bitXor
      (Exp.sub (Exp.bitOr x (Exp.litU32 0x80808080)) (Exp.litU32 0x20202020))
      (Exp.litU32 0x80808080)

  ShaderM.loop (Exp.litU32 0) (Exp.litU32 blocksPerRow) (Exp.litU32 1) fun blockIdx => do
    let blockByteBase := Exp.add rowByteBase (Exp.mul blockIdx (Exp.litU32 blockSizeBytes))

    -- Read d (fp16 at byte offset 208)
    let dLo ← readByte blockByteBase (Exp.litU32 208)
    let dHi ← readByte blockByteBase (Exp.litU32 209)
    let dBits := Exp.bitOr dLo (Exp.shiftLeft dHi (Exp.litU32 8))
    let d := fp16ToF32 dBits

    -- Read vl (4 bytes of ql at byte offset 4*iqs)
    let vlOffset := Exp.mul iqs (Exp.litU32 4)
    let vl ← read4Bytes blockByteBase vlOffset
    -- Read vh_raw (4 bytes of qh at byte offset 128 + 4*vhIdx), shift right by vh_shift
    let vhOffset := Exp.add (Exp.litU32 128) (Exp.mul vhIdx (Exp.litU32 4))
    let vhRaw ← read4Bytes blockByteBase vhOffset
    let vh := Exp.shiftRight vhRaw vhShift

    -- Read 2 scales: scales[scale_offset], scales[scale_offset + 4]
    -- (scales start at byte 192, each is 1 signed byte).
    -- Keep scales as i32 (sign-extended from i8) throughout the inner loop;
    -- defer the f32 cast until after the int×int multiply with dot_0.
    -- This matches llama.cpp's `d8[i] * (dp4a × sc)` pattern — 1 FFMA per
    -- iter instead of 2 (the old `d8 * (f32(dot) * f32(sc))` required two
    -- f32 conversions and two FFMAs).  SASS confirms: llama has 3 FFMA
    -- vs hesper's 24 in the old version.
    let sc0Byte ← readByte blockByteBase (Exp.add (Exp.litU32 192) scaleOff)
    let sc1Byte ← readByte blockByteBase (Exp.add (Exp.litU32 192) (Exp.add scaleOff (Exp.litU32 4)))
    -- Sign-extend i8 → i32 via "or 0xFFFFFF00 when byte ≥ 128" trick; bit
    -- pattern is identical to two's-complement signed extension, so reinterpret
    -- the u32 as i32 afterwards for the signed multiply with dot_0 (i32).
    let signExtI8 (b : Exp (.scalar .u32)) : Exp (.scalar .u32) :=
      Exp.select (Exp.ge b (Exp.litU32 128))
        (Exp.bitOr b (Exp.litU32 0xFFFFFF00)) b
    let sc0I : Exp (.scalar .i32) := Exp.toI32 (signExtI8 sc0Byte)
    let sc1I : Exp (.scalar .i32) := Exp.toI32 (signExtI8 sc1Byte)

    let q8BlockIdx i :=
      Exp.add (Exp.mul blockIdx (Exp.litU32 (8 * 9)))
              (Exp.mul (Exp.add bq8Off (Exp.mul (Exp.litU32 i) (Exp.litU32 2))) (Exp.litU32 9))
    let q8Sub0 := q8BlockIdx 0
    let q8Sub1 := q8BlockIdx 1
    let u0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8"
              (Exp.add q8Sub0 (Exp.add (Exp.litU32 1) q8ElemOff))
    let u1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8"
              (Exp.add q8Sub1 (Exp.add (Exp.litU32 1) q8ElemOff))
    let q8Hdr0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" q8Sub0
    let q8Hdr1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" q8Sub1
    let d8A : Exp (.scalar .f32) := Exp.bitcast q8Hdr0
    let d8B : Exp (.scalar .f32) := Exp.bitcast q8Hdr1

    -- QR6_K=2 iterations. dot × sc done in i32; single FFMA d8 × f32(dot*sc).
    let vil_0 := Exp.bitAnd vl (Exp.litU32 0x0F0F0F0F)
    let vih_0 := Exp.bitAnd (Exp.shiftLeft vh (Exp.litU32 4)) (Exp.litU32 0x30303030)
    let vi_0 := sub32PerByte (Exp.bitOr vil_0 vih_0)
    let dot_0 := Exp.dot4I8Packed vi_0 u0
    let dotSc_0 := Exp.mul dot_0 sc0I
    let sumf_0 := Exp.mul d8A (Exp.toF32 dotSc_0)

    let vil_1 := Exp.bitAnd (Exp.shiftRight vl (Exp.litU32 4)) (Exp.litU32 0x0F0F0F0F)
    let vih_1 := Exp.bitAnd (Exp.shiftLeft (Exp.shiftRight vh (Exp.litU32 4)) (Exp.litU32 4))
                            (Exp.litU32 0x30303030)
    let vi_1 := sub32PerByte (Exp.bitOr vil_1 vih_1)
    let dot_1 := Exp.dot4I8Packed vi_1 u1
    let dotSc_1 := Exp.mul dot_1 sc1I
    let sumf_1 := Exp.mul d8B (Exp.toF32 dotSc_1)

    ShaderM.assign "acc" (Exp.add acc (Exp.mul d (Exp.add sumf_0 sumf_1)))

  -- All 32 lanes contribute unique partials (iqs=tid). Standard subgroupAdd.
  ShaderM.varNamed "total" (.scalar .f32) (Exp.subgroupAdd acc)
  let total : Exp (.scalar .f32) := Exp.var "total"

  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx total
  ) (pure ())

/-- 2-rows-per-workgroup variant of `fusedQ6KLinearDP4AKernel` with a
    shared-memory cooperative load of the Q8_1 input.

    Motivation: the 1-row-per-WG variant has every output row re-read the
    full Q8_1 input from global memory.  At 262,144 rows (Gemma 4
    lm_head), that's `q8InputU32Size × 262144 = 2880 × 262144 ≈ 756 MB`
    of redundant input traffic.  With 2 rows per WG + a shared staging
    buffer, the input is loaded **once** per WG and reused by both warps,
    cutting that traffic in half.  The load itself is amortised across
    64 threads (~12 u32/lane for a 2560-wide hidden state).

    Dispatch: `⌈outDim / 2⌉` workgroups × 64 threads. Each warp (32
    lanes) independently computes its own row's dot product and writes
    one output element.

    @param inDim Input dimension
    @param outDim Output dimension (even — use the 1-row variant for odd)
    @param gridX Grid X dimension for 2D grid; 0 for 1D. -/
def fusedQ6KLinearDP4A2RowKernel (inDim outDim : Nat) (gridX : Nat := 0) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let pairIdx :=
    if gridX == 0 then Exp.vec3X wid
    else Exp.add (Exp.vec3X wid) (Exp.mul (Exp.vec3Y wid) (Exp.litU32 gridX))
  let tid := Exp.vec3X lid                           -- 0..63
  let subgroupId := Exp.shiftRight tid (Exp.litU32 5) -- 0 or 1 (which row)
  let laneId := Exp.bitAnd tid (Exp.litU32 31)        -- 0..31 (dp4a lane)
  let outIdx := Exp.add (Exp.mul pairIdx (Exp.litU32 2)) subgroupId

  let blocksPerRow := inDim / 256
  let blockSizeBytes : Nat := 210
  let totalWeightBytes := outDim * blocksPerRow * blockSizeBytes
  let totalWeightU32 := (totalWeightBytes + 3) / 4
  let q8BlocksPerRow := inDim / 32
  let q8InputU32Size := q8BlocksPerRow * 9

  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareReadOnlyBuffer "input_q8" (.array (.scalar .u32) q8InputU32Size)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) outDim)

  -- Stage the Q8_1 input in shared memory so the two warps in this WG
  -- re-read from smem instead of issuing two independent global-memory
  -- streams.  64 threads load q8InputU32Size words between them.
  ShaderM.sharedNamed "s_input_q8" (.array (.scalar .u32) q8InputU32Size)
  let perThread : Nat := (q8InputU32Size + 63) / 64
  for i in [0 : perThread] do
    let idx := Exp.add tid (Exp.litU32 (i * 64))
    ShaderM.if_ (Exp.lt idx (Exp.litU32 q8InputU32Size)) (do
      let v ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" idx
      ShaderM.writeWorkgroup (ty := .scalar .u32) "s_input_q8" idx v
    ) (pure ())
  ShaderM.barrier

  let inBounds := Exp.lt outIdx (Exp.litU32 outDim)

  -- Use laneId (not tid) for intra-row dp4a lane assignment.
  let iqs : Exp (.scalar .u32) := laneId
  let iqsDiv16 := Exp.shiftRight iqs (Exp.litU32 4)
  let iqsMod16 := Exp.bitAnd iqs (Exp.litU32 15)
  let bq8Off := Exp.add (Exp.mul iqsDiv16 (Exp.litU32 4)) (Exp.shiftRight iqsMod16 (Exp.litU32 3))
  let scaleOff := Exp.add (Exp.mul iqsDiv16 (Exp.litU32 8)) (Exp.shiftRight iqsMod16 (Exp.litU32 2))
  let vhShift := Exp.mul (Exp.shiftRight iqsMod16 (Exp.litU32 3)) (Exp.litU32 2)
  let iqsMod8 := Exp.bitAnd iqs (Exp.litU32 7)
  let vhIdx := Exp.add (Exp.mul iqsDiv16 (Exp.litU32 8)) iqsMod8
  let q8ElemOff := iqsMod8

  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"

  ShaderM.varNamed "rowByteBase" (.scalar .u32)
    (Exp.mul outIdx (Exp.litU32 (blocksPerRow * blockSizeBytes)))
  let rowByteBase : Exp (.scalar .u32) := Exp.var "rowByteBase"

  let readByte (blockBase : Exp (.scalar .u32)) (offset : Exp (.scalar .u32)) : ShaderM (Exp (.scalar .u32)) := do
    let byteIdx := Exp.add blockBase offset
    let u32Idx := Exp.shiftRight byteIdx (Exp.litU32 2)
    let byteShift := Exp.mul (Exp.bitAnd byteIdx (Exp.litU32 3)) (Exp.litU32 8)
    let u32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" u32Idx
    pure (Exp.bitAnd (Exp.shiftRight u32 byteShift) (Exp.litU32 0xFF))

  let read4Bytes (blockBase : Exp (.scalar .u32)) (offset : Exp (.scalar .u32)) : ShaderM (Exp (.scalar .u32)) := do
    let byteIdx := Exp.add blockBase offset
    let u32Idx := Exp.shiftRight byteIdx (Exp.litU32 2)
    let byteOff := Exp.bitAnd byteIdx (Exp.litU32 3)
    let shiftLo := Exp.mul byteOff (Exp.litU32 8)
    let shiftHi := Exp.mul (Exp.sub (Exp.litU32 4) byteOff) (Exp.litU32 8)
    let w0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" u32Idx
    let w1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights"
              (Exp.add u32Idx (Exp.litU32 1))
    let lo := Exp.shiftRight w0 shiftLo
    let hi := Exp.select (Exp.eq byteOff (Exp.litU32 0))
                (Exp.litU32 0) (Exp.shiftLeft w1 shiftHi)
    pure (Exp.bitOr lo hi)

  let sub32PerByte (x : Exp (.scalar .u32)) : Exp (.scalar .u32) :=
    Exp.bitXor
      (Exp.sub (Exp.bitOr x (Exp.litU32 0x80808080)) (Exp.litU32 0x20202020))
      (Exp.litU32 0x80808080)

  ShaderM.loop (Exp.litU32 0) (Exp.litU32 blocksPerRow) (Exp.litU32 1) fun blockIdx => do
    let blockByteBase := Exp.add rowByteBase (Exp.mul blockIdx (Exp.litU32 blockSizeBytes))

    let dLo ← readByte blockByteBase (Exp.litU32 208)
    let dHi ← readByte blockByteBase (Exp.litU32 209)
    let dBits := Exp.bitOr dLo (Exp.shiftLeft dHi (Exp.litU32 8))
    let d := fp16ToF32 dBits

    let vlOffset := Exp.mul iqs (Exp.litU32 4)
    let vl ← read4Bytes blockByteBase vlOffset
    let vhOffset := Exp.add (Exp.litU32 128) (Exp.mul vhIdx (Exp.litU32 4))
    let vhRaw ← read4Bytes blockByteBase vhOffset
    let vh := Exp.shiftRight vhRaw vhShift

    let sc0Byte ← readByte blockByteBase (Exp.add (Exp.litU32 192) scaleOff)
    let sc1Byte ← readByte blockByteBase (Exp.add (Exp.litU32 192) (Exp.add scaleOff (Exp.litU32 4)))
    let scaleToF32 (b : Exp (.scalar .u32)) : Exp (.scalar .f32) :=
      Exp.select (Exp.ge b (Exp.litU32 128))
        (Exp.sub (Exp.toF32 b) (Exp.litF32 256.0))
        (Exp.toF32 b)
    let sc0 := scaleToF32 sc0Byte
    let sc1 := scaleToF32 sc1Byte

    let q8BlockIdx i :=
      Exp.add (Exp.mul blockIdx (Exp.litU32 (8 * 9)))
              (Exp.mul (Exp.add bq8Off (Exp.mul (Exp.litU32 i) (Exp.litU32 2))) (Exp.litU32 9))
    let q8Sub0 := q8BlockIdx 0
    let q8Sub1 := q8BlockIdx 1
    let u0 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := q8InputU32Size) "s_input_q8"
              (Exp.add q8Sub0 (Exp.add (Exp.litU32 1) q8ElemOff))
    let u1 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := q8InputU32Size) "s_input_q8"
              (Exp.add q8Sub1 (Exp.add (Exp.litU32 1) q8ElemOff))
    let q8Hdr0 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := q8InputU32Size) "s_input_q8" q8Sub0
    let q8Hdr1 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := q8InputU32Size) "s_input_q8" q8Sub1
    let d8A : Exp (.scalar .f32) := Exp.bitcast q8Hdr0
    let d8B : Exp (.scalar .f32) := Exp.bitcast q8Hdr1

    let vil_0 := Exp.bitAnd vl (Exp.litU32 0x0F0F0F0F)
    let vih_0 := Exp.bitAnd (Exp.shiftLeft vh (Exp.litU32 4)) (Exp.litU32 0x30303030)
    let vi_0 := sub32PerByte (Exp.bitOr vil_0 vih_0)
    let dot_0 := Exp.dot4I8Packed vi_0 u0
    let sumf_0 := Exp.mul d8A (Exp.mul (Exp.toF32 dot_0) sc0)

    let vil_1 := Exp.bitAnd (Exp.shiftRight vl (Exp.litU32 4)) (Exp.litU32 0x0F0F0F0F)
    let vih_1 := Exp.bitAnd (Exp.shiftLeft (Exp.shiftRight vh (Exp.litU32 4)) (Exp.litU32 4))
                            (Exp.litU32 0x30303030)
    let vi_1 := sub32PerByte (Exp.bitOr vil_1 vih_1)
    let dot_1 := Exp.dot4I8Packed vi_1 u1
    let sumf_1 := Exp.mul d8B (Exp.mul (Exp.toF32 dot_1) sc1)

    ShaderM.assign "acc" (Exp.add acc (Exp.mul d (Exp.add sumf_0 sumf_1)))

  -- Each warp reduces its 32 lanes' accumulators independently; lane 0
  -- of each warp writes one output row.
  ShaderM.varNamed "total" (.scalar .f32) (Exp.subgroupAdd acc)
  let total : Exp (.scalar .f32) := Exp.var "total"
  ShaderM.if_ (Exp.and (Exp.eq laneId (Exp.litU32 0)) inBounds) (do
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx total
  ) (pure ())

/-- 4-warp cooperative variant of `fusedQ6KLinearDP4AKernel`.

    Same idea as the 2-row kernel — load the Q8_1 input once into shared
    memory and reuse it across N output rows — but scaled up to **4 warps
    per workgroup** (128 threads, 4 output rows). This matches
    llama.cpp's `mul_mat_vec_q<…, nwarps=4>` scheme.

    Why 4 warps: the Q8_1 input at hiddenSize=2560 is `q8InputU32Size =
    720 u32 = 2880 B` per row.  With 128 threads cooperating the load is
    `ceil(720/128) = 6 u32/lane`, and the staged buffer is re-read by
    4 independent warps each computing a different row's dp4a.  The
    global→smem traffic per row drops 4× vs the 1-row kernel.

    Dispatch: `⌈outDim / 4⌉` workgroups × 128 threads. Requires
    subgroup support (used per-warp for the final reduction).

    @param inDim  Input dimension (must be multiple of 256 for Q6_K blocks)
    @param outDim Output dimension (should be multiple of 4 for
                  perfect packing; otherwise the tail rows are masked
                  out by the `inBounds` check and do no output write)
    @param gridX  Grid X dimension for 2D grid; 0 for 1D. -/
def fusedQ6KLinearDP4A4RowKernel (inDim outDim : Nat) (gridX : Nat := 0) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let quadIdx :=
    if gridX == 0 then Exp.vec3X wid
    else Exp.add (Exp.vec3X wid) (Exp.mul (Exp.vec3Y wid) (Exp.litU32 gridX))
  let tid := Exp.vec3X lid                           -- 0..127
  let subgroupId := Exp.shiftRight tid (Exp.litU32 5) -- 0..3 (which row in the quad)
  let laneId := Exp.bitAnd tid (Exp.litU32 31)        -- 0..31 (dp4a lane)
  let outIdx := Exp.add (Exp.mul quadIdx (Exp.litU32 4)) subgroupId

  let blocksPerRow := inDim / 256
  let blockSizeBytes : Nat := 210
  let totalWeightBytes := outDim * blocksPerRow * blockSizeBytes
  let totalWeightU32 := (totalWeightBytes + 3) / 4
  let q8BlocksPerRow := inDim / 32
  let q8InputU32Size := q8BlocksPerRow * 9

  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareReadOnlyBuffer "input_q8" (.array (.scalar .u32) q8InputU32Size)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) outDim)

  -- Cooperative smem load: 128 threads stage q8InputU32Size words.
  ShaderM.sharedNamed "s_input_q8" (.array (.scalar .u32) q8InputU32Size)
  let perThread : Nat := (q8InputU32Size + 127) / 128
  for i in [0 : perThread] do
    let idx := Exp.add tid (Exp.litU32 (i * 128))
    ShaderM.if_ (Exp.lt idx (Exp.litU32 q8InputU32Size)) (do
      let v ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" idx
      ShaderM.writeWorkgroup (ty := .scalar .u32) "s_input_q8" idx v
    ) (pure ())
  ShaderM.barrier

  let inBounds := Exp.lt outIdx (Exp.litU32 outDim)

  -- Per-warp dp4a lane assignment; identical to the 1-row / 2-row variants.
  let iqs : Exp (.scalar .u32) := laneId
  let iqsDiv16 := Exp.shiftRight iqs (Exp.litU32 4)
  let iqsMod16 := Exp.bitAnd iqs (Exp.litU32 15)
  let bq8Off := Exp.add (Exp.mul iqsDiv16 (Exp.litU32 4)) (Exp.shiftRight iqsMod16 (Exp.litU32 3))
  let scaleOff := Exp.add (Exp.mul iqsDiv16 (Exp.litU32 8)) (Exp.shiftRight iqsMod16 (Exp.litU32 2))
  let vhShift := Exp.mul (Exp.shiftRight iqsMod16 (Exp.litU32 3)) (Exp.litU32 2)
  let iqsMod8 := Exp.bitAnd iqs (Exp.litU32 7)
  let vhIdx := Exp.add (Exp.mul iqsDiv16 (Exp.litU32 8)) iqsMod8
  let q8ElemOff := iqsMod8

  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"

  ShaderM.varNamed "rowByteBase" (.scalar .u32)
    (Exp.mul outIdx (Exp.litU32 (blocksPerRow * blockSizeBytes)))
  let rowByteBase : Exp (.scalar .u32) := Exp.var "rowByteBase"

  let readByte (blockBase : Exp (.scalar .u32)) (offset : Exp (.scalar .u32)) : ShaderM (Exp (.scalar .u32)) := do
    let byteIdx := Exp.add blockBase offset
    let u32Idx := Exp.shiftRight byteIdx (Exp.litU32 2)
    let byteShift := Exp.mul (Exp.bitAnd byteIdx (Exp.litU32 3)) (Exp.litU32 8)
    let u32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" u32Idx
    pure (Exp.bitAnd (Exp.shiftRight u32 byteShift) (Exp.litU32 0xFF))

  let read4Bytes (blockBase : Exp (.scalar .u32)) (offset : Exp (.scalar .u32)) : ShaderM (Exp (.scalar .u32)) := do
    let byteIdx := Exp.add blockBase offset
    let u32Idx := Exp.shiftRight byteIdx (Exp.litU32 2)
    let byteOff := Exp.bitAnd byteIdx (Exp.litU32 3)
    let shiftLo := Exp.mul byteOff (Exp.litU32 8)
    let shiftHi := Exp.mul (Exp.sub (Exp.litU32 4) byteOff) (Exp.litU32 8)
    let w0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" u32Idx
    let w1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights"
              (Exp.add u32Idx (Exp.litU32 1))
    let lo := Exp.shiftRight w0 shiftLo
    let hi := Exp.select (Exp.eq byteOff (Exp.litU32 0))
                (Exp.litU32 0) (Exp.shiftLeft w1 shiftHi)
    pure (Exp.bitOr lo hi)

  let sub32PerByte (x : Exp (.scalar .u32)) : Exp (.scalar .u32) :=
    Exp.bitXor
      (Exp.sub (Exp.bitOr x (Exp.litU32 0x80808080)) (Exp.litU32 0x20202020))
      (Exp.litU32 0x80808080)

  ShaderM.loop (Exp.litU32 0) (Exp.litU32 blocksPerRow) (Exp.litU32 1) fun blockIdx => do
    let blockByteBase := Exp.add rowByteBase (Exp.mul blockIdx (Exp.litU32 blockSizeBytes))

    let dLo ← readByte blockByteBase (Exp.litU32 208)
    let dHi ← readByte blockByteBase (Exp.litU32 209)
    let dBits := Exp.bitOr dLo (Exp.shiftLeft dHi (Exp.litU32 8))
    let d := fp16ToF32 dBits

    let vlOffset := Exp.mul iqs (Exp.litU32 4)
    let vl ← read4Bytes blockByteBase vlOffset
    let vhOffset := Exp.add (Exp.litU32 128) (Exp.mul vhIdx (Exp.litU32 4))
    let vhRaw ← read4Bytes blockByteBase vhOffset
    let vh := Exp.shiftRight vhRaw vhShift

    let sc0Byte ← readByte blockByteBase (Exp.add (Exp.litU32 192) scaleOff)
    let sc1Byte ← readByte blockByteBase (Exp.add (Exp.litU32 192) (Exp.add scaleOff (Exp.litU32 4)))
    let scaleToF32 (b : Exp (.scalar .u32)) : Exp (.scalar .f32) :=
      Exp.select (Exp.ge b (Exp.litU32 128))
        (Exp.sub (Exp.toF32 b) (Exp.litF32 256.0))
        (Exp.toF32 b)
    let sc0 := scaleToF32 sc0Byte
    let sc1 := scaleToF32 sc1Byte

    let q8BlockIdx i :=
      Exp.add (Exp.mul blockIdx (Exp.litU32 (8 * 9)))
              (Exp.mul (Exp.add bq8Off (Exp.mul (Exp.litU32 i) (Exp.litU32 2))) (Exp.litU32 9))
    let q8Sub0 := q8BlockIdx 0
    let q8Sub1 := q8BlockIdx 1
    let u0 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := q8InputU32Size) "s_input_q8"
              (Exp.add q8Sub0 (Exp.add (Exp.litU32 1) q8ElemOff))
    let u1 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := q8InputU32Size) "s_input_q8"
              (Exp.add q8Sub1 (Exp.add (Exp.litU32 1) q8ElemOff))
    let q8Hdr0 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := q8InputU32Size) "s_input_q8" q8Sub0
    let q8Hdr1 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := q8InputU32Size) "s_input_q8" q8Sub1
    let d8A : Exp (.scalar .f32) := Exp.bitcast q8Hdr0
    let d8B : Exp (.scalar .f32) := Exp.bitcast q8Hdr1

    let vil_0 := Exp.bitAnd vl (Exp.litU32 0x0F0F0F0F)
    let vih_0 := Exp.bitAnd (Exp.shiftLeft vh (Exp.litU32 4)) (Exp.litU32 0x30303030)
    let vi_0 := sub32PerByte (Exp.bitOr vil_0 vih_0)
    let dot_0 := Exp.dot4I8Packed vi_0 u0
    let sumf_0 := Exp.mul d8A (Exp.mul (Exp.toF32 dot_0) sc0)

    let vil_1 := Exp.bitAnd (Exp.shiftRight vl (Exp.litU32 4)) (Exp.litU32 0x0F0F0F0F)
    let vih_1 := Exp.bitAnd (Exp.shiftLeft (Exp.shiftRight vh (Exp.litU32 4)) (Exp.litU32 4))
                            (Exp.litU32 0x30303030)
    let vi_1 := sub32PerByte (Exp.bitOr vil_1 vih_1)
    let dot_1 := Exp.dot4I8Packed vi_1 u1
    let sumf_1 := Exp.mul d8B (Exp.mul (Exp.toF32 dot_1) sc1)

    ShaderM.assign "acc" (Exp.add acc (Exp.mul d (Exp.add sumf_0 sumf_1)))

  -- Per-warp reduction: each of the 4 warps independently reduces its
  -- 32 lanes.  Lane 0 of each warp writes one output row.
  ShaderM.varNamed "total" (.scalar .f32) (Exp.subgroupAdd acc)
  let total : Exp (.scalar .f32) := Exp.var "total"
  ShaderM.if_ (Exp.and (Exp.eq laneId (Exp.litU32 0)) inBounds) (do
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx total
  ) (pure ())

/-! ## Fused RMSNorm + Q4_K_M MatVec -/

/-- Fused RMSNorm + Q4_K_M mat-vec kernel.

    Computes `y[i] = dot(W_q4k[i], RMSNorm(x))` in a single dispatch by
    computing the RMS normalization on-the-fly within each workgroup.

    Each WG (32 threads = 1 subgroup) handles one output row:
    1. **RMS pass**: all 32 lanes cooperatively read the full input vector
       (strided), accumulate x², subgroupAdd → total sum of squares.
       Compute `rms_inv = rsqrt(mean(x²) + eps)`.
    2. **Q4_K dot-product**: same block-coop structure as
       `fusedQ4KMLinearBlockCoopKernel`, but each input element is
       normalized inline: `normed = x[i] * rms_inv * scale[i]`.
       No intermediate buffer, no extra dispatch.

    The input reads for RMS are served from L2 cache (shared across 2560+
    WGs all reading the same 2560-element vector). The extra per-WG cost
    is one warp reduction (~5 shuffles) + 2560/32 = 80 strided reads for
    the norm scale buffer.

    Buffers: weights (Q4_K), input (raw, un-normed), norm_scale (RMSNorm
    weight), output (f32).

    Dispatch: `outDim` workgroups × 32 threads. Requires subgroup support.

    @param config Layer dimensions
    @param eps RMSNorm epsilon -/
def fusedRMSNormQ4KMLinearKernel (config : Config) (eps : Float) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid

  let blocksPerRow := config.inDim / 256
  let totalWeightU32 := config.outDim * blocksPerRow * 36

  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareReadOnlyBuffer "input" (.array (.scalar .f32) config.inDim)
  let _normScale ← ShaderM.declareReadOnlyBuffer "norm_scale" (.array (.scalar .f32) config.inDim)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) config.outDim)

  let inBounds := Exp.lt outIdx (Exp.litU32 config.outDim)

  -- ═══ Phase 1: Compute RMS inverse ═══
  -- Each lane accumulates x² for its stride of the input vector.
  ShaderM.varNamed "sqSum" (.scalar .f32) (Exp.litF32 0.0)
  let sqSum : Exp (.scalar .f32) := Exp.var "sqSum"

  ShaderM.loop tid (Exp.litU32 config.inDim) (Exp.litU32 32) fun idx => do
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.inDim) "input" idx
    ShaderM.assign "sqSum" (Exp.add sqSum (Exp.mul val val))

  -- subgroupAdd gives total sum of squares across all 32 lanes
  ShaderM.varNamed "totalSqSum" (.scalar .f32) (Exp.subgroupAdd sqSum)
  let totalSqSum : Exp (.scalar .f32) := Exp.var "totalSqSum"
  -- rms_inv = rsqrt(mean(x²) + eps)
  let mean := Exp.div totalSqSum (Exp.litF32 config.inDim.toFloat)
  ShaderM.varNamed "rmsInv" (.scalar .f32)
    (Exp.div (Exp.litF32 1.0) (Exp.sqrt (Exp.add mean (Exp.litF32 eps))))
  let rmsInv : Exp (.scalar .f32) := Exp.var "rmsInv"

  -- ═══ Phase 2: Q4_K dot product with inline normalization ═══
  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"

  let rowBaseU32 := Exp.mul outIdx (Exp.litU32 (blocksPerRow * 36))

  let cLane := Exp.div tid (Exp.litU32 8)
  let l32Lane := Exp.sub tid (Exp.mul cLane (Exp.litU32 8))

  ShaderM.loop (Exp.litU32 0) (Exp.litU32 blocksPerRow) (Exp.litU32 1) fun blockIdx => do
    let blockU32Base := Exp.add rowBaseU32 (Exp.mul blockIdx (Exp.litU32 36))
    let elemBase := Exp.mul blockIdx (Exp.litU32 256)

    let dmU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" blockU32Base
    let d := fp16ToF32 (Exp.bitAnd dmU32 (Exp.litU32 0xFFFF))
    let dmin := fp16ToF32 (Exp.shiftRight dmU32 (Exp.litU32 16))
    let sc0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add blockU32Base (Exp.litU32 1))
    let sc1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add blockU32Base (Exp.litU32 2))
    let sc2 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add blockU32Base (Exp.litU32 3))

    let is0 := Exp.mul cLane (Exp.litU32 2)
    let is1 := Exp.add is0 (Exp.litU32 1)

    let extractScaleMin (is : Exp (.scalar .u32)) : Exp (.scalar .f32) × Exp (.scalar .f32) :=
      let isLow := Exp.lt is (Exp.litU32 4)
      let shift4 := Exp.mul is (Exp.litU32 8)
      let scaleLow := Exp.bitAnd (Exp.shiftRight sc0 shift4) (Exp.litU32 0x3F)
      let minLow   := Exp.bitAnd (Exp.shiftRight sc1 shift4) (Exp.litU32 0x3F)
      let isHi := Exp.sub is (Exp.litU32 4)
      let shiftHi := Exp.mul isHi (Exp.litU32 8)
      let scaleHiLo := Exp.bitAnd (Exp.shiftRight sc2 shiftHi) (Exp.litU32 0x0F)
      let scaleHiHi := Exp.shiftLeft
        (Exp.bitAnd (Exp.shiftRight sc0 (Exp.add shiftHi (Exp.litU32 6))) (Exp.litU32 0x03))
        (Exp.litU32 4)
      let scaleHigh := Exp.bitOr scaleHiLo scaleHiHi
      let minHiLo := Exp.bitAnd (Exp.shiftRight sc2 (Exp.add shiftHi (Exp.litU32 4))) (Exp.litU32 0x0F)
      let minHiHi := Exp.shiftLeft
        (Exp.bitAnd (Exp.shiftRight sc1 (Exp.add shiftHi (Exp.litU32 6))) (Exp.litU32 0x03))
        (Exp.litU32 4)
      let minHigh := Exp.bitOr minHiLo minHiHi
      let scaleU := Exp.select isLow scaleLow scaleHigh
      let minU   := Exp.select isLow minLow   minHigh
      (Exp.toF32 scaleU, Exp.toF32 minU)

    let (scaleA, minA) := extractScaleMin is0
    let (scaleB, minB) := extractScaleMin is1
    let d1 := Exp.mul d scaleA
    let m1 := Exp.mul dmin minA
    let d2 := Exp.mul d scaleB
    let m2 := Exp.mul dmin minB

    let qsLaneIdx := Exp.add blockU32Base
                     (Exp.add (Exp.litU32 4)
                       (Exp.add (Exp.mul cLane (Exp.litU32 8)) l32Lane))
    let qsU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" qsLaneIdx

    let elemOffset := Exp.add (Exp.mul cLane (Exp.litU32 64))
                      (Exp.mul l32Lane (Exp.litU32 4))
    let elemBaseAbs := Exp.add elemBase elemOffset

    for b in [0:4] do
      let byte := Exp.bitAnd (Exp.shiftRight qsU32 (Exp.litU32 (b * 8))) (Exp.litU32 0xFF)
      let qLow := Exp.bitAnd byte (Exp.litU32 0xF)
      let qHigh := Exp.shiftRight byte (Exp.litU32 4)
      let elemIdxLow := Exp.add elemBaseAbs (Exp.litU32 b)
      let elemIdxHigh := Exp.add elemBaseAbs (Exp.litU32 (32 + b))
      -- Read raw input and normalize inline: normed = x * rmsInv * normScale
      let rawLow ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.inDim) "input" elemIdxLow
      let nsLow ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.inDim) "norm_scale" elemIdxLow
      let inLow := Exp.mul (Exp.mul rawLow rmsInv) nsLow
      let rawHigh ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.inDim) "input" elemIdxHigh
      let nsHigh ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.inDim) "norm_scale" elemIdxHigh
      let inHigh := Exp.mul (Exp.mul rawHigh rmsInv) nsHigh
      let wLow := Exp.sub (Exp.mul d1 (Exp.toF32 qLow)) m1
      let wHigh := Exp.sub (Exp.mul d2 (Exp.toF32 qHigh)) m2
      ShaderM.assign "acc" (Exp.add acc (Exp.add (Exp.mul wLow inLow) (Exp.mul wHigh inHigh)))

  -- Subgroup reduction
  ShaderM.varNamed "totalSum" (.scalar .f32) (Exp.subgroupAdd acc)
  let totalSum : Exp (.scalar .f32) := Exp.var "totalSum"

  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx totalSum
  ) (pure ())

/-! ## Split-K Q4_K_M MatVec (for ffnDown-shape low-wave-count linears) -/

/-- Split-K Q4_K_M mat-vec kernel. Each workgroup computes a partial
    dot-product over a **contiguous slice** of the input-dim blocks for
    one output row, and writes the partial sum into a scratch buffer at
    `partial[outIdx * splits + splitIdx]`. A second small kernel
    (`splitKReduceKernel`) sums the `splits` partials into the final
    `output[outIdx]`.

    Motivation: for `ffnDown` (outDim=2560, inDim=10240), the 1-WG-per-row
    scheme only launches `2560 × 32 = 81920` threads, filling ~0.9 waves
    on an RTX 4070 Ti (60 SM × 1536 resident threads = 92160 capacity).
    With less than one wave of latent Warps on each SM, the hardware
    scheduler has no second Warp to swap in when the current one stalls
    on a DRAM load, and memory latency directly serialises the kernel.
    Splitting the K dim (inDim) by a factor of `splits` multiplies the
    WG count by `splits`, raising wave occupancy to `splits × 0.9`
    and restoring inter-Warp latency hiding.

    The per-block loop is the same software-pipelined block-coop body
    as `fusedQ4KMLinearBlockCoopKernel`, just constrained to
    `[splitStart, splitEnd)` of the row's blocks. `blocksPerSplit` must
    divide `blocksPerRow` evenly.

    Dispatch grid: `(outDim * splits, 1, 1)` workgroups × 32 threads.
    Requires subgroup support and `blocksPerRow % splits == 0`. -/
def fusedQ4KMLinearSplitKKernel (config : Config) (splits : Nat) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let gwid := Exp.vec3X wid
  let tid := Exp.vec3X lid

  let blocksPerRow := config.inDim / 256
  let blocksPerSplit := blocksPerRow / splits
  let totalWeightU32 := config.outDim * blocksPerRow * 36

  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareReadOnlyBuffer "input" (.array (.scalar .f32) config.inDim)
  -- Partial-sums scratch: outDim * splits f32s, row-major
  -- [outIdx * splits + splitIdx].
  let _partial ← ShaderM.declareOutputBuffer "partial"
    (.array (.scalar .f32) (config.outDim * splits))

  -- Decompose flat WG id: outIdx = gwid / splits, splitIdx = gwid % splits.
  let outIdx := Exp.div gwid (Exp.litU32 splits)
  let splitIdx := Exp.sub gwid (Exp.mul outIdx (Exp.litU32 splits))
  let inBounds := Exp.lt outIdx (Exp.litU32 config.outDim)

  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"

  -- Row base + split starting block offset (runtime).
  let rowBaseU32 := Exp.mul outIdx (Exp.litU32 (blocksPerRow * 36))
  let splitStartBlock := Exp.mul splitIdx (Exp.litU32 blocksPerSplit)
  let splitBlockBaseU32 := Exp.add rowBaseU32 (Exp.mul splitStartBlock (Exp.litU32 36))
  -- Starting element (relative to row) for the split's first block.
  let splitElemBase := Exp.mul splitStartBlock (Exp.litU32 256)

  let cLane := Exp.div tid (Exp.litU32 8)
  let l32Lane := Exp.sub tid (Exp.mul cLane (Exp.litU32 8))
  let qsOffsetInBlock := Exp.add (Exp.litU32 4)
                         (Exp.add (Exp.mul cLane (Exp.litU32 8)) l32Lane)

  -- Pre-loop depth-1 prefetch of the split's first block.
  let init0Dm  ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" splitBlockBaseU32
  let init0Sc0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add splitBlockBaseU32 (Exp.litU32 1))
  let init0Sc1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add splitBlockBaseU32 (Exp.litU32 2))
  let init0Sc2 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add splitBlockBaseU32 (Exp.litU32 3))
  let init0Qs  ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add splitBlockBaseU32 qsOffsetInBlock)
  ShaderM.varNamed "nextDm"  (.scalar .u32) init0Dm
  ShaderM.varNamed "nextSc0" (.scalar .u32) init0Sc0
  ShaderM.varNamed "nextSc1" (.scalar .u32) init0Sc1
  ShaderM.varNamed "nextSc2" (.scalar .u32) init0Sc2
  ShaderM.varNamed "nextQs"  (.scalar .u32) init0Qs

  for localBlock in [0:blocksPerSplit] do
    -- elemBase: element offset relative to row start = splitElemBase + localBlock*256.
    let elemBase := Exp.add splitElemBase (Exp.litU32 (localBlock * 256))

    let cDm  := s!"currDm_{localBlock}"
    let cSc0 := s!"currSc0_{localBlock}"
    let cSc1 := s!"currSc1_{localBlock}"
    let cSc2 := s!"currSc2_{localBlock}"
    let cQs  := s!"currQs_{localBlock}"
    ShaderM.varNamed cDm  (.scalar .u32) (Exp.var "nextDm")
    ShaderM.varNamed cSc0 (.scalar .u32) (Exp.var "nextSc0")
    ShaderM.varNamed cSc1 (.scalar .u32) (Exp.var "nextSc1")
    ShaderM.varNamed cSc2 (.scalar .u32) (Exp.var "nextSc2")
    ShaderM.varNamed cQs  (.scalar .u32) (Exp.var "nextQs")
    let dmU32 : Exp (.scalar .u32) := Exp.var cDm
    let sc0   : Exp (.scalar .u32) := Exp.var cSc0
    let sc1   : Exp (.scalar .u32) := Exp.var cSc1
    let sc2   : Exp (.scalar .u32) := Exp.var cSc2
    let qsU32 : Exp (.scalar .u32) := Exp.var cQs
    let d := fp16ToF32 (Exp.bitAnd dmU32 (Exp.litU32 0xFFFF))
    let dmin := fp16ToF32 (Exp.shiftRight dmU32 (Exp.litU32 16))

    if localBlock + 1 < blocksPerSplit then
      let nbBaseNext := Exp.add splitBlockBaseU32 (Exp.litU32 ((localBlock + 1) * 36))
      ShaderM.assign "nextDm"  (← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" nbBaseNext)
      ShaderM.assign "nextSc0" (← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add nbBaseNext (Exp.litU32 1)))
      ShaderM.assign "nextSc1" (← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add nbBaseNext (Exp.litU32 2)))
      ShaderM.assign "nextSc2" (← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add nbBaseNext (Exp.litU32 3)))
      ShaderM.assign "nextQs"  (← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add nbBaseNext qsOffsetInBlock))

    let is0 := Exp.mul cLane (Exp.litU32 2)
    let is1 := Exp.add is0 (Exp.litU32 1)

    let extractScaleMin (is : Exp (.scalar .u32)) : Exp (.scalar .f32) × Exp (.scalar .f32) :=
      let isLow := Exp.lt is (Exp.litU32 4)
      let shift4 := Exp.mul is (Exp.litU32 8)
      let scaleLow := Exp.bitAnd (Exp.shiftRight sc0 shift4) (Exp.litU32 0x3F)
      let minLow   := Exp.bitAnd (Exp.shiftRight sc1 shift4) (Exp.litU32 0x3F)
      let isHi := Exp.sub is (Exp.litU32 4)
      let shiftHi := Exp.mul isHi (Exp.litU32 8)
      let scaleHiLo := Exp.bitAnd (Exp.shiftRight sc2 shiftHi) (Exp.litU32 0x0F)
      let scaleHiHi := Exp.shiftLeft
        (Exp.bitAnd (Exp.shiftRight sc0 (Exp.add shiftHi (Exp.litU32 6))) (Exp.litU32 0x03))
        (Exp.litU32 4)
      let scaleHigh := Exp.bitOr scaleHiLo scaleHiHi
      let minHiLo := Exp.bitAnd (Exp.shiftRight sc2 (Exp.add shiftHi (Exp.litU32 4))) (Exp.litU32 0x0F)
      let minHiHi := Exp.shiftLeft
        (Exp.bitAnd (Exp.shiftRight sc1 (Exp.add shiftHi (Exp.litU32 6))) (Exp.litU32 0x03))
        (Exp.litU32 4)
      let minHigh := Exp.bitOr minHiLo minHiHi
      let scaleU := Exp.select isLow scaleLow scaleHigh
      let minU   := Exp.select isLow minLow   minHigh
      (Exp.toF32 scaleU, Exp.toF32 minU)

    let (scaleA, minA) := extractScaleMin is0
    let (scaleB, minB) := extractScaleMin is1
    let d1 := Exp.mul d scaleA
    let m1 := Exp.mul dmin minA
    let d2 := Exp.mul d scaleB
    let m2 := Exp.mul dmin minB

    let elemOffset := Exp.add (Exp.mul cLane (Exp.litU32 64))
                      (Exp.mul l32Lane (Exp.litU32 4))
    let elemBaseAbs := Exp.add elemBase elemOffset

    for b in [0:4] do
      let byte := Exp.bitAnd (Exp.shiftRight qsU32 (Exp.litU32 (b * 8))) (Exp.litU32 0xFF)
      let qLow := Exp.bitAnd byte (Exp.litU32 0xF)
      let qHigh := Exp.shiftRight byte (Exp.litU32 4)
      let elemIdxLow := Exp.add elemBaseAbs (Exp.litU32 b)
      let elemIdxHigh := Exp.add elemBaseAbs (Exp.litU32 (32 + b))
      let inLow ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.inDim) "input" elemIdxLow
      let inHigh ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.inDim) "input" elemIdxHigh
      let wLow := Exp.sub (Exp.mul d1 (Exp.toF32 qLow)) m1
      let wHigh := Exp.sub (Exp.mul d2 (Exp.toF32 qHigh)) m2
      ShaderM.assign "acc" (Exp.add acc (Exp.add (Exp.mul wLow inLow) (Exp.mul wHigh inHigh)))

  ShaderM.varNamed "totalSum" (.scalar .f32) (Exp.subgroupAdd acc)
  let totalSum : Exp (.scalar .f32) := Exp.var "totalSum"

  -- Lane 0 writes the partial sum to partial[outIdx * splits + splitIdx].
  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    let writeIdx := Exp.add (Exp.mul outIdx (Exp.litU32 splits)) splitIdx
    ShaderM.writeBuffer (ty := .scalar .f32) "partial" writeIdx totalSum
  ) (pure ())

/-- Reduce kernel for split-K: sum the `splits` partials for each
    output row into `output[outIdx]`. Dispatched as a flat 1D grid of
    `outDim` threads (one thread per output element); the caller picks
    `numWorkgroups = outDim / 256`, `workgroupSize = 256`. -/
def splitKReduceKernel (outDim splits : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let outIdx := Exp.vec3X gid

  let _partial ← ShaderM.declareReadOnlyBuffer "partial" (.array (.scalar .f32) (outDim * splits))
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) outDim)

  let inBounds := Exp.lt outIdx (Exp.litU32 outDim)

  ShaderM.if_ inBounds (do
    ShaderM.varNamed "sum" (.scalar .f32) (Exp.litF32 0.0)
    let sum : Exp (.scalar .f32) := Exp.var "sum"
    let base := Exp.mul outIdx (Exp.litU32 splits)
    for s in [0:splits] do
      let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := outDim * splits) "partial"
                (Exp.add base (Exp.litU32 s))
      ShaderM.assign "sum" (Exp.add sum v)
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx sum
  ) (pure ())

/-- Decide split-K factor based on shape. Currently returns 1
    unconditionally — the infrastructure is in place and correct, but
    on RTX 4070 Ti / Tint it yields essentially no speedup on ffnDown
    (splits=2,4,8 all tested, all within 1% of the depth-1 pipelined
    baseline). The wave-count hypothesis does not hold here; see the
    commit message for `b1288bf`-followup analysis.

    Kept as a toggle so future investigation on other hardware or with
    a different reduce strategy can re-enable it by returning >1. -/
def splitKFactorFor (_cfg : Config) : Nat := 1

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
  let _weightsGate ← ShaderM.declareReadOnlyBuffer "weights_gate" (.array (.scalar .u32) totalWeightU32)
  let _weightsUp   ← ShaderM.declareReadOnlyBuffer "weights_up"   (.array (.scalar .u32) totalWeightU32)
  let _input       ← ShaderM.declareReadOnlyBuffer "input"        (.array (.scalar .f32) config.inDim)
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
structure LinearLayer (BufT : Type) (CacheT : Type := Unit) where
  config : Config
  weightBuf : BufT    -- Raw packed weights on GPU
  quantFormat : QuantFormat  -- Which dequant kernel to use
  prepared : IO.Ref (Option CacheT)
  -- Split-K partial-sums workspace buffer (`outDim * splits` f32), lazily
  -- allocated on the first call when split-K is eligible (see
  -- `splitKFactorFor` below). Nil until then.
  splitKBuf : IO.Ref (Option BufT)
  -- Prepared dispatch for the split-K partial kernel and the reduce kernel.
  splitKPartialPrepared : IO.Ref (Option CacheT)
  splitKReducePrepared : IO.Ref (Option CacheT)
  -- dp4a path: Q8_1 quantized input scratch (inDim/32 * 9 u32 bytes), lazy.
  dp4aQ8Buf : IO.Ref (Option BufT)
  -- Prepared dispatches for (quantize, matmul) of the dp4a pipeline.
  dp4aQuantizePrepared : IO.Ref (Option CacheT)
  dp4aMatmulPrepared : IO.Ref (Option CacheT)

/-- Q8_1 quantize input + Q4_K/Q6_K dp4a matmul (2 dispatches).
    Uses lazily-allocated per-layer Q8_1 scratch buffer and cache refs.
    Dispatches to the appropriate dp4a matmul kernel by `layer.quantFormat`. -/
def forwardDP4A [GPUBackend β] (ctx : β)
    (layer : LinearLayer (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (inputBuf outputBuf : GPUBackend.Buf β)
    : IO Unit := do
  if layer.quantFormat != .Q4_K && layer.quantFormat != .Q6_K then
    throw (IO.userError s!"forwardDP4A: must be Q4_K or Q6_K, got {repr layer.quantFormat}")

  let profiling ← profilingRef.get
  let startNs ← if profiling then IO.monoNanosNow else pure 0

  let nQ8Blocks := layer.config.inDim / 32
  let q8BufBytes : USize := (nQ8Blocks * 9 * 4).toUSize

  -- Lazily allocate Q8_1 scratch buffer on first call.
  let q8Buf ← match ← layer.dp4aQ8Buf.get with
    | some b => pure b
    | none =>
      let b ← GPUBackend.allocBuffer ctx q8BufBytes
      layer.dp4aQ8Buf.set (some b)
      pure b

  -- Step 1: Quantize input f32 → Q8_1
  GPUBackend.executeWithConfigCached ctx
    (quantizeQ8_1Kernel layer.config.inDim)
    [("input", inputBuf), ("output", q8Buf)]
    { numWorkgroups := (nQ8Blocks, 1, 1), workgroupSize := { x := 32, y := 1, z := 1 } }
    (hash ("q8_1-quantize", layer.config.inDim))
    layer.dp4aQuantizePrepared

  -- Step 2: Weights × Q8_1 matmul via dp4a.
  -- Select kernel by quant format and shape:
  --   Q4_K + small outDim → 2-row variant (high occupancy)
  --   Q4_K + large outDim → 1-row variant
  --   Q6_K + outDim % 4 == 0 → 4-warp cooperative (smem input reuse)
  --   Q6_K + outDim % 2 == 0 → 2-warp cooperative
  --   Q6_K fallback         → 1-row (single-warp)
  if layer.quantFormat == .Q4_K then
    let useTwoRow := layer.config.outDim ≤ 5120 && layer.config.outDim % 2 == 0
    if useTwoRow then
      GPUBackend.executeWithConfigCached ctx
        (fusedQ4KMLinearDP4A2RowKernel layer.config)
        [("weights", layer.weightBuf), ("input_q8", q8Buf), ("output", outputBuf)]
        { numWorkgroups := (layer.config.outDim / 2, 1, 1)
          workgroupSize := { x := 64, y := 1, z := 1 } }
        (hash ("q4k-dp4a-matmul-2row", layer.config.inDim, layer.config.outDim))
        layer.dp4aMatmulPrepared
    else
      GPUBackend.executeWithConfigCached ctx
        (fusedQ4KMLinearDP4AKernel layer.config)
        [("weights", layer.weightBuf), ("input_q8", q8Buf), ("output", outputBuf)]
        { numWorkgroups := (layer.config.outDim, 1, 1), workgroupSize := { x := 32, y := 1, z := 1 } }
        (hash ("q4k-dp4a-matmul", layer.config.inDim, layer.config.outDim))
        layer.dp4aMatmulPrepared
  else  -- Q6_K
    if layer.config.outDim % 4 == 0 then
      GPUBackend.executeWithConfigCached ctx
        (fusedQ6KLinearDP4A4RowKernel layer.config.inDim layer.config.outDim)
        [("weights", layer.weightBuf), ("input_q8", q8Buf), ("output", outputBuf)]
        { numWorkgroups := (layer.config.outDim / 4, 1, 1)
          workgroupSize := { x := 128, y := 1, z := 1 } }
        (hash ("q6k-dp4a-matmul-4row", layer.config.inDim, layer.config.outDim))
        layer.dp4aMatmulPrepared
    else if layer.config.outDim % 2 == 0 then
      GPUBackend.executeWithConfigCached ctx
        (fusedQ6KLinearDP4A2RowKernel layer.config.inDim layer.config.outDim)
        [("weights", layer.weightBuf), ("input_q8", q8Buf), ("output", outputBuf)]
        { numWorkgroups := (layer.config.outDim / 2, 1, 1)
          workgroupSize := { x := 64, y := 1, z := 1 } }
        (hash ("q6k-dp4a-matmul-2row", layer.config.inDim, layer.config.outDim))
        layer.dp4aMatmulPrepared
    else
      GPUBackend.executeWithConfigCached ctx
        (fusedQ6KLinearDP4AKernel layer.config.inDim layer.config.outDim)
        [("weights", layer.weightBuf), ("input_q8", q8Buf), ("output", outputBuf)]
        { numWorkgroups := (layer.config.outDim, 1, 1), workgroupSize := { x := 32, y := 1, z := 1 } }
        (hash ("q6k-dp4a-matmul", layer.config.inDim, layer.config.outDim))
        layer.dp4aMatmulPrepared

  if profiling then
    let endNs ← IO.monoNanosNow
    let delta := (endNs - startNs).toUInt64
    totalNanosRef.modify (· + delta)
    callCountRef.modify (· + 1)
    perShapeAdd layer.config.inDim layer.config.outDim delta

/-- Execute the linear layer: output = input @ weights^T

    Fast path: after the first call, the prepared dispatch is cached in
    `layer.prepared` and subsequent calls bypass WGSL regeneration / hash
    lookup entirely via `replayPreparedDispatch`. -/
def LinearLayer.forward [GPUBackend β] (ctx : β) (layer : LinearLayer (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (inputBuf outputBuf : GPUBackend.Buf β) : IO Unit := do
  let profiling ← profilingRef.get
  let startNs ← if profiling then IO.monoNanosNow else pure 0

  let useSubgroups ← GPUBackend.hasSubgroupSupport ctx

  -- dp4a fast path: Q8_1 quantize + INT8 SIMD matmul for Q4_K and Q6_K linears.
  -- Requires subgroups (used for warp-reduce) and inDim divisible by 32.
  -- Q6_K additionally requires inDim % 256 == 0 (super-block size).
  let dp4aQ4K := layer.quantFormat == .Q4_K
  let dp4aQ6K := layer.quantFormat == .Q6_K && layer.config.inDim % 256 == 0
  if (← dp4aEnabled.get) && (dp4aQ4K || dp4aQ6K) && useSubgroups
     && layer.config.inDim % 32 == 0 then
    forwardDP4A ctx layer inputBuf outputBuf
    return

  let splits := if layer.quantFormat == .Q4_K && useSubgroups
                then splitKFactorFor layer.config else 1

  if splits > 1 then
    -- ## Split-K fast path
    -- Two dispatches: partial compute → reduce. Both have their own
    -- PreparedDispatch caches on the layer.
    if let (some pPart, some pRed) ← do
        let a ← layer.splitKPartialPrepared.get
        let b ← layer.splitKReducePrepared.get
        pure (a, b) then
      GPUBackend.replayCached ctx pPart (layer.config.outDim * splits, 1, 1)
      let reduceWGs := (layer.config.outDim + 255) / 256
      GPUBackend.replayCached ctx pRed (reduceWGs, 1, 1)
      if profiling then
        let endNs ← IO.monoNanosNow
        let delta := (endNs - startNs).toUInt64
        totalNanosRef.modify (· + delta)
        callCountRef.modify (· + 1)
        perShapeAdd layer.config.inDim layer.config.outDim delta
      return

    -- Slow path: lazily alloc the partial-sums buffer, then run both
    -- dispatches in the non-fast form so they get cached in the
    -- respective prepared refs.
    let partialBuf ← (do
      match ← layer.splitKBuf.get with
      | some b => pure b
      | none =>
        let sizeBytes : USize := (layer.config.outDim * splits * 4).toUSize
        let b ← GPUBackend.allocBuffer ctx sizeBytes
        layer.splitKBuf.set (some b)
        pure b)

    let partialBuffers := [
      ("weights", layer.weightBuf),
      ("input", inputBuf),
      ("partial", partialBuf)
    ]
    let partialCfg : Hesper.ExecConfig := {
      numWorkgroups := (layer.config.outDim * splits, 1, 1)
      workgroupSize := { x := 32, y := 1, z := 1 }
    }
    let partialCacheKey : UInt64 :=
      hash ("q4k-lin-splitk-partial", layer.config.inDim, layer.config.outDim, splits)
    GPUBackend.executeWithConfigCached ctx
      (fusedQ4KMLinearSplitKKernel layer.config splits)
      partialBuffers partialCfg partialCacheKey layer.splitKPartialPrepared

    let reduceBuffers := [
      ("partial", partialBuf),
      ("output", outputBuf)
    ]
    let reduceWGs := (layer.config.outDim + 255) / 256
    let reduceCfg : Hesper.ExecConfig := {
      numWorkgroups := (reduceWGs, 1, 1)
      workgroupSize := { x := 256, y := 1, z := 1 }
    }
    let reduceCacheKey : UInt64 :=
      hash ("q4k-lin-splitk-reduce", layer.config.outDim, splits)
    GPUBackend.executeWithConfigCached ctx
      (splitKReduceKernel layer.config.outDim splits)
      reduceBuffers reduceCfg reduceCacheKey layer.splitKReducePrepared

    if profiling then
      let endNs ← IO.monoNanosNow
      let delta := (endNs - startNs).toUInt64
      totalNanosRef.modify (· + delta)
      callCountRef.modify (· + 1)
      perShapeAdd layer.config.inDim layer.config.outDim delta
    return

  -- Fast path: instant replay if we already have a prepared dispatch and
  -- the input/output buffers match the prepared binding. The prepared
  -- dispatch binds specific buffer handles, so if the caller ever passes
  -- different buffers, we need to fall through to the slow path and
  -- re-prepare. In practice for Gemma 4 every layer gets its own
  -- state.* buffers that are stable across forward passes, so the fast
  -- path hits after the first call.
  if let some p ← layer.prepared.get then
    -- Use the same workgroup count that was used at prepare time.
    GPUBackend.replayCached ctx p (layer.config.outDim, 1, 1)
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
  -- Kernel selection strategy:
  --   Q4_K + subgroups: 4-subgroup kernel (128 threads) for high occupancy.
  --     llama.cpp uses 4 warps per row for Q4_K; this matches that strategy.
  --   Q4_K - subgroups: fallback shared-memory tree-reduction (256 threads).
  --   Q6_K: block-coop subgroup (32 threads) or tree-reduction (256 threads).
  let numSubgroups := 4
  let (wgSize, shader, cacheKey) := match layer.quantFormat, useSubgroups with
    | .Q4_K, true  =>
      (32,
       fusedQ4KMLinearBlockCoopKernel layer.config,
       hash ("q4k-lin-blockcoop-swpipe", layer.config.inDim, layer.config.outDim, true))
    | .Q4_K, false =>
      (256,
       fusedQ4KMLinearKernel layer.config,
       hash ("q4k-lin-blockcoop-swpipe", layer.config.inDim, layer.config.outDim, false))
    | .Q6_K, true  =>
      (32,
       Hesper.Quantization.Q6_K.fusedQ6KLinearBlockCoopKernel layer.config.inDim layer.config.outDim,
       hash ("q6k-lin-blockcoop-swpipe", layer.config.inDim, layer.config.outDim, true))
    | .Q6_K, false =>
      (256,
       Hesper.Quantization.Q6_K.fusedQ6KLinearKernel layer.config.inDim layer.config.outDim,
       hash ("q6k-lin-blockcoop-swpipe", layer.config.inDim, layer.config.outDim, false))
  let execConfig : Hesper.ExecConfig := {
    numWorkgroups := (layer.config.outDim, 1, 1)
    workgroupSize := { x := wgSize, y := 1, z := 1 }
  }
  GPUBackend.executeWithConfigCached ctx shader namedBuffers execConfig cacheKey layer.prepared
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
def forwardFusedGateUp [GPUBackend β] (ctx : β)
    (gate up : LinearLayer (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (inputBuf outputBuf : GPUBackend.Buf β)
    (preparedRef : IO.Ref (Option (GPUBackend.CachedDispatch β)))
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

  -- dp4a fast path when enabled: quantize input once, then fused dp4a
  -- gate+up+GELU×mul in a single matmul kernel.  Reuses the layer's
  -- own Q8_1 scratch buffer (keyed to `gate` so the same buffer is
  -- re-used across decode steps).
  let useDP4A ← do
    let on ← dp4aEnabled.get
    pure (on && gate.config.inDim % 32 == 0)
  if useDP4A then
    let nQ8Blocks := gate.config.inDim / 32
    let q8BufBytes : USize := (nQ8Blocks * 9 * 4).toUSize
    let q8Buf ← match ← gate.dp4aQ8Buf.get with
      | some b => pure b
      | none =>
        let b ← GPUBackend.allocBuffer ctx q8BufBytes
        gate.dp4aQ8Buf.set (some b)
        pure b
    -- Step 1: Q8_1 quantize.
    GPUBackend.executeWithConfigCached ctx
      (quantizeQ8_1Kernel gate.config.inDim)
      [("input", inputBuf), ("output", q8Buf)]
      { numWorkgroups := (nQ8Blocks, 1, 1), workgroupSize := { x := 32, y := 1, z := 1 } }
      (hash ("q8_1-quantize", gate.config.inDim))
      gate.dp4aQuantizePrepared
    -- Step 2: dp4a gate+up+GELU×mul.  Select 4-row variant when outDim
    -- is a multiple of 4 (smem-shared input across 4 warps, halves the
    -- effective input-memory traffic).
    if gate.config.outDim % 4 == 0 then
      GPUBackend.executeWithConfigCached ctx
        (fusedQ4KMGateUpDP4A4RowKernel gate.config)
        [("weights_gate", gate.weightBuf),
         ("weights_up",   up.weightBuf),
         ("input_q8",     q8Buf),
         ("output",       outputBuf)]
        { numWorkgroups := (gate.config.outDim / 4, 1, 1)
          workgroupSize := { x := 128, y := 1, z := 1 } }
        (hash ("q4k-gate-up-dp4a-4row", gate.config.inDim, gate.config.outDim))
        preparedRef
    else
      GPUBackend.executeWithConfigCached ctx
        (fusedQ4KMGateUpDP4AKernel gate.config)
        [("weights_gate", gate.weightBuf),
         ("weights_up",   up.weightBuf),
         ("input_q8",     q8Buf),
         ("output",       outputBuf)]
        { numWorkgroups := (gate.config.outDim, 1, 1), workgroupSize := { x := 32, y := 1, z := 1 } }
        (hash ("q4k-gate-up-dp4a", gate.config.inDim, gate.config.outDim))
        preparedRef
    if profiling then
      let endNs ← IO.monoNanosNow
      let delta := (endNs - startNs).toUInt64
      totalNanosRef.modify (· + delta)
      callCountRef.modify (· + 2)
      perShapeAdd gate.config.inDim gate.config.outDim delta
    return

  -- Fast path: instant replay if prepared (non-dp4a path).
  if let some p ← preparedRef.get then
    GPUBackend.replayCached ctx p (gate.config.outDim, 1, 1)
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
  let execConfig : Hesper.ExecConfig := {
    numWorkgroups := (gate.config.outDim, 1, 1)
    workgroupSize := { x := 32, y := 1, z := 1 }
  }
  let cacheKey : UInt64 :=
    hash ("q4k-gate-up", gate.config.inDim, gate.config.outDim)
  GPUBackend.executeWithConfigCached ctx
    (fusedQ4KMGateUpSubgroupKernel gate.config)
    namedBuffers execConfig cacheKey preparedRef
  if profiling then
    let endNs ← IO.monoNanosNow
    let delta := (endNs - startNs).toUInt64
    totalNanosRef.modify (· + delta)
    callCountRef.modify (· + 2)
    perShapeAdd gate.config.inDim gate.config.outDim delta

/-- Fused Q/K/V projection forward pass.

    Runs the Q8_1 quantize ONCE on the shared input, then three dp4a
    matmuls that reuse the same Q8_1 buffer: wQ (outDim_q), wK (outDim_kv),
    wV (outDim_kv).  Because wK and wV have identical shape, they're
    combined into one fused kernel (`fusedQ4KMKVDP4AKernel`); wQ runs as
    a separate matmul.  Overall: 3 projection matmuls + 1 quantize =
    4 dispatches, vs naive 3 + 3 = 6.

    Requires all three layers to be Q4_K.  wQ may have a different
    outDim than wK/wV (the typical Gemma-4 case 2048 vs 256). -/
def forwardFusedQKV [GPUBackend β] (ctx : β)
    (wQ wK wV : LinearLayer (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (inputBuf qBuf kBuf vBuf : GPUBackend.Buf β)
    (kvPreparedRef : IO.Ref (Option (GPUBackend.CachedDispatch β)))
    : IO Unit := do
  if wQ.quantFormat != .Q4_K || wK.quantFormat != .Q4_K || wV.quantFormat != .Q4_K then
    throw (IO.userError "forwardFusedQKV: all three layers must be Q4_K")
  if wQ.config.inDim != wK.config.inDim || wK.config.inDim != wV.config.inDim then
    throw (IO.userError "forwardFusedQKV: inDim mismatch between Q/K/V")
  if wK.config.outDim != wV.config.outDim then
    throw (IO.userError s!"forwardFusedQKV: wK/wV outDim mismatch {wK.config.outDim} vs {wV.config.outDim}")

  let profiling ← profilingRef.get
  let startNs ← if profiling then IO.monoNanosNow else pure 0

  let useDP4A ← do
    let on ← dp4aEnabled.get
    pure (on && wQ.config.inDim % 32 == 0)
  if !useDP4A then
    LinearLayer.forward ctx wQ inputBuf qBuf
    LinearLayer.forward ctx wK inputBuf kBuf
    LinearLayer.forward ctx wV inputBuf vBuf
    return

  let nQ8Blocks := wQ.config.inDim / 32
  let q8BufBytes : USize := (nQ8Blocks * 9 * 4).toUSize
  -- Reuse wQ's scratch as the canonical shared Q8_1 buffer.
  let q8Buf ← match ← wQ.dp4aQ8Buf.get with
    | some b => pure b
    | none =>
      let b ← GPUBackend.allocBuffer ctx q8BufBytes
      wQ.dp4aQ8Buf.set (some b)
      pure b

  -- Step 1: quantize input once.  Keyed on wQ so subsequent Q/K/V reuse.
  GPUBackend.executeWithConfigCached ctx
    (quantizeQ8_1Kernel wQ.config.inDim)
    [("input", inputBuf), ("output", q8Buf)]
    { numWorkgroups := (nQ8Blocks, 1, 1), workgroupSize := { x := 32, y := 1, z := 1 } }
    (hash ("q8_1-quantize", wQ.config.inDim))
    wQ.dp4aQuantizePrepared

  -- Step 2: wQ matmul (separate kernel since its outDim differs).  Pick
  -- 2-row when outDim is even (typical, matches forwardDP4A's heuristic).
  let qIs2Row := wQ.config.outDim ≤ 5120 && wQ.config.outDim % 2 == 0
  if qIs2Row then
    GPUBackend.executeWithConfigCached ctx
      (fusedQ4KMLinearDP4A2RowKernel wQ.config)
      [("weights", wQ.weightBuf), ("input_q8", q8Buf), ("output", qBuf)]
      { numWorkgroups := (wQ.config.outDim / 2, 1, 1)
        workgroupSize := { x := 64, y := 1, z := 1 } }
      (hash ("q4k-dp4a-matmul-2row", wQ.config.inDim, wQ.config.outDim))
      wQ.dp4aMatmulPrepared
  else
    GPUBackend.executeWithConfigCached ctx
      (fusedQ4KMLinearDP4AKernel wQ.config)
      [("weights", wQ.weightBuf), ("input_q8", q8Buf), ("output", qBuf)]
      { numWorkgroups := (wQ.config.outDim, 1, 1), workgroupSize := { x := 32, y := 1, z := 1 } }
      (hash ("q4k-dp4a-matmul", wQ.config.inDim, wQ.config.outDim))
      wQ.dp4aMatmulPrepared

  -- Step 3: fused wK + wV matmul sharing the same Q8_1 buffer.
  GPUBackend.executeWithConfigCached ctx
    (fusedQ4KMKVDP4AKernel wK.config)
    [("weights_k", wK.weightBuf),
     ("weights_v", wV.weightBuf),
     ("input_q8",  q8Buf),
     ("output_k",  kBuf),
     ("output_v",  vBuf)]
    { numWorkgroups := (wK.config.outDim, 1, 1), workgroupSize := { x := 32, y := 1, z := 1 } }
    (hash ("q4k-kv-dp4a", wK.config.inDim, wK.config.outDim))
    kvPreparedRef

  if profiling then
    let endNs ← IO.monoNanosNow
    let delta := (endNs - startNs).toUInt64
    totalNanosRef.modify (· + delta)
    callCountRef.modify (· + 3)
    perShapeAdd wQ.config.inDim wQ.config.outDim delta

/-- PLE `ple.inpGate` + `ple.geluGateMul` fused into one dispatch.

    Standard flow is
      `inpGate_matmul → [plGateBuf] → GELU(_) * perLayerInput[plOffset+i] → outputBuf`
    (2 dispatches).  This wrapper runs the matmul with a GELU + slice-
    multiply epilogue baked in — saves one dispatch per PLE site per
    token.  Requires Q4_K inpGate + dp4a + subgroups + inDim % 256 == 0.
    Caller falls back to the 2-dispatch path otherwise. -/
def forwardFusedPLInpGate [GPUBackend β] (ctx : β)
    (inpGate : LinearLayer (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (inputBuf perLayerInputBuf outputBuf : GPUBackend.Buf β)
    (plTotalSize plOffset : Nat)
    (preparedRef : IO.Ref (Option (GPUBackend.CachedDispatch β)))
    : IO Unit := do
  if inpGate.quantFormat != .Q4_K then
    throw (IO.userError s!"forwardFusedPLInpGate: inpGate must be Q4_K, got {repr inpGate.quantFormat}")
  let useDP4A ← do
    let on ← dp4aEnabled.get
    pure (on && inpGate.config.inDim % 256 == 0)
  if !useDP4A then
    throw (IO.userError "forwardFusedPLInpGate: dp4a precondition failed; caller should fall back")

  -- Reuse this layer's own Q8_1 scratch buffer (same pattern as the
  -- non-fused matmul path).
  let nQ8Blocks := inpGate.config.inDim / 32
  let q8BufBytes : USize := (nQ8Blocks * 9 * 4).toUSize
  let q8Buf ← match ← inpGate.dp4aQ8Buf.get with
    | some b => pure b
    | none =>
      let b ← GPUBackend.allocBuffer ctx q8BufBytes
      inpGate.dp4aQ8Buf.set (some b)
      pure b

  -- Step 1: Q8_1 quantize (the f32 `inputBuf` is the hidden state after
  -- the attention block — NOT already RMSNormed, just per the PLE
  -- semantics).  Shared prepared ref so every PLE site caches.
  GPUBackend.executeWithConfigCached ctx
    (quantizeQ8_1Kernel inpGate.config.inDim)
    [("input", inputBuf), ("output", q8Buf)]
    { numWorkgroups := (nQ8Blocks, 1, 1), workgroupSize := { x := 32, y := 1, z := 1 } }
    (hash ("q8_1-quantize", inpGate.config.inDim))
    inpGate.dp4aQuantizePrepared

  -- Step 2: fused matmul + GELU-slice-mul epilogue.  `plOffset` is
  -- baked into the kernel's PTX; keep it in the cache key so each
  -- layer's variant is cached separately.
  GPUBackend.executeWithConfigCached ctx
    (fusedQ4KMLinearDP4AGeluSliceKernel inpGate.config plTotalSize plOffset)
    [("weights", inpGate.weightBuf), ("input_q8", q8Buf),
     ("per_layer_input", perLayerInputBuf), ("output", outputBuf)]
    { numWorkgroups := (inpGate.config.outDim, 1, 1),
      workgroupSize := { x := 32, y := 1, z := 1 } }
    (hash ("q4k-dp4a-matmul-gelu-slice", inpGate.config.inDim,
           inpGate.config.outDim, plOffset))
    preparedRef

/-- Fused [RMSNorm + Q8_1 quantize] → fused gate+up GeGLU.

    Replaces the standard 3-dispatch sequence
      `RMSNorm.forward → quantizeQ8_1Kernel → fused gate+up dp4a`
    with a 2-dispatch sequence by collapsing the first two kernels
    into the single `fusedRMSNormQ8_1Kernel`.

    Mirrors `forwardFusedNormQKV` for the FFN side: normalised input
    is computed in-register and quantised straight into the Q8_1
    buffer, never round-tripping to VRAM.

    Requires Q4_K weights, dp4a, inDim divisible by 256.  Caller
    must fall back to the unfused path otherwise. -/
def forwardFusedNormGateUp [GPUBackend β] (ctx : β)
    (norm : RMSNorm.RMSNorm (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (gate up : LinearLayer (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (inputBuf outputBuf : GPUBackend.Buf β)
    (preparedRef : IO.Ref (Option (GPUBackend.CachedDispatch β)))
    : IO Unit := do
  if gate.quantFormat != .Q4_K then
    throw (IO.userError s!"forwardFusedNormGateUp: gate must be Q4_K, got {repr gate.quantFormat}")
  if up.quantFormat != .Q4_K then
    throw (IO.userError s!"forwardFusedNormGateUp: up must be Q4_K, got {repr up.quantFormat}")
  if gate.config.inDim != up.config.inDim || gate.config.outDim != up.config.outDim then
    throw (IO.userError s!"forwardFusedNormGateUp: shape mismatch")
  if gate.config.inDim != norm.config.dim then
    throw (IO.userError s!"forwardFusedNormGateUp: dim mismatch norm={norm.config.dim} gate.in={gate.config.inDim}")
  let useDP4A ← do
    let on ← dp4aEnabled.get
    pure (on && gate.config.inDim % 32 == 0 && gate.config.inDim % 256 == 0)
  if !useDP4A then
    throw (IO.userError "forwardFusedNormGateUp: dp4a precondition failed; caller should fall back")

  let nQ8Blocks := gate.config.inDim / 32
  let q8BufBytes : USize := (nQ8Blocks * 9 * 4).toUSize
  let q8Buf ← match ← gate.dp4aQ8Buf.get with
    | some b => pure b
    | none =>
      let b ← GPUBackend.allocBuffer ctx q8BufBytes
      gate.dp4aQ8Buf.set (some b)
      pure b

  -- Step 1: fused RMSNorm + Q8_1 quantize.  No f32 normedBuf.
  GPUBackend.executeWithConfigCached ctx
    (Hesper.Layers.RMSNorm.fusedRMSNormQ8_1Kernel norm.config)
    [("input", inputBuf), ("scale", norm.scale), ("output", q8Buf)]
    { numWorkgroups := (1, 1, 1), workgroupSize := { x := 256, y := 1, z := 1 } }
    (hash ("fused-rmsnorm-q8_1", norm.config.dim))
    gate.dp4aQuantizePrepared

  -- Step 2: dp4a fused gate+up+GELU×mul, picking 4-row variant when
  -- outDim is a multiple of 4 (smem-shared input across 4 warps).
  if gate.config.outDim % 4 == 0 then
    GPUBackend.executeWithConfigCached ctx
      (fusedQ4KMGateUpDP4A4RowKernel gate.config)
      [("weights_gate", gate.weightBuf),
       ("weights_up",   up.weightBuf),
       ("input_q8",     q8Buf),
       ("output",       outputBuf)]
      { numWorkgroups := (gate.config.outDim / 4, 1, 1)
        workgroupSize := { x := 128, y := 1, z := 1 } }
      (hash ("q4k-gate-up-dp4a-4row", gate.config.inDim, gate.config.outDim))
      preparedRef
  else
    GPUBackend.executeWithConfigCached ctx
      (fusedQ4KMGateUpDP4AKernel gate.config)
      [("weights_gate", gate.weightBuf),
       ("weights_up",   up.weightBuf),
       ("input_q8",     q8Buf),
       ("output",       outputBuf)]
      { numWorkgroups := (gate.config.outDim, 1, 1), workgroupSize := { x := 32, y := 1, z := 1 } }
      (hash ("q4k-gate-up-dp4a", gate.config.inDim, gate.config.outDim))
      preparedRef

/-- Fused [RMSNorm + Q8_1 quantize] → wQ → wK+wV.

    Replaces the standard 4-dispatch sequence
      `RMSNorm.forward → quantizeQ8_1Kernel → wQ_matmul → wK+wV matmul`
    with a 3-dispatch sequence by collapsing the first two kernels
    into the single `fusedRMSNormQ8_1Kernel`.  The pre-norm input
    flows directly through to a Q8_1 packed buffer in one pass —
    eliminating the f32 normedBuf round-trip to VRAM (~10 KB per layer
    per token).

    Requires Q4_K weights, dp4a + subgroups, inDim divisible by 32.
    Falls back to the unfused `forwardFusedQKV` path if any precondition
    fails. -/
def forwardFusedNormQKV [GPUBackend β] (ctx : β)
    (norm : RMSNorm.RMSNorm (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (wQ wK wV : LinearLayer (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (inputBuf qBuf kBuf vBuf : GPUBackend.Buf β)
    (kvPreparedRef : IO.Ref (Option (GPUBackend.CachedDispatch β)))
    : IO Unit := do
  if wQ.quantFormat != .Q4_K || wK.quantFormat != .Q4_K || wV.quantFormat != .Q4_K then
    throw (IO.userError "forwardFusedNormQKV: all three projections must be Q4_K")
  if wQ.config.inDim != norm.config.dim then
    throw (IO.userError s!"forwardFusedNormQKV: dim mismatch norm={norm.config.dim} wQ.in={wQ.config.inDim}")
  if wK.config.outDim != wV.config.outDim then
    throw (IO.userError s!"forwardFusedNormQKV: wK/wV outDim mismatch {wK.config.outDim} vs {wV.config.outDim}")
  let useDP4A ← do
    let on ← dp4aEnabled.get
    pure (on && wQ.config.inDim % 32 == 0 && wQ.config.inDim % 256 == 0)
  if !useDP4A then
    -- Cannot use the fused path; fall back to RMSNorm.forward + forwardFusedQKV
    -- (the caller's responsibility to do the RMSNorm separately if it gets here).
    throw (IO.userError "forwardFusedNormQKV: dp4a precondition failed; caller should fall back")

  let nQ8Blocks := wQ.config.inDim / 32
  let q8BufBytes : USize := (nQ8Blocks * 9 * 4).toUSize
  let q8Buf ← match ← wQ.dp4aQ8Buf.get with
    | some b => pure b
    | none =>
      let b ← GPUBackend.allocBuffer ctx q8BufBytes
      wQ.dp4aQ8Buf.set (some b)
      pure b

  -- Step 1: fused RMSNorm + Q8_1 quantize — single dispatch, no
  -- intermediate f32 normedBuf round-trip.
  GPUBackend.executeWithConfigCached ctx
    (Hesper.Layers.RMSNorm.fusedRMSNormQ8_1Kernel norm.config)
    [("input", inputBuf), ("scale", norm.scale), ("output", q8Buf)]
    { numWorkgroups := (1, 1, 1), workgroupSize := { x := 256, y := 1, z := 1 } }
    (hash ("fused-rmsnorm-q8_1", norm.config.dim))
    wQ.dp4aQuantizePrepared

  -- Step 2: wQ matmul (separate kernel — outDim differs from KV).
  let qIs2Row := wQ.config.outDim ≤ 5120 && wQ.config.outDim % 2 == 0
  if qIs2Row then
    GPUBackend.executeWithConfigCached ctx
      (fusedQ4KMLinearDP4A2RowKernel wQ.config)
      [("weights", wQ.weightBuf), ("input_q8", q8Buf), ("output", qBuf)]
      { numWorkgroups := (wQ.config.outDim / 2, 1, 1)
        workgroupSize := { x := 64, y := 1, z := 1 } }
      (hash ("q4k-dp4a-matmul-2row", wQ.config.inDim, wQ.config.outDim))
      wQ.dp4aMatmulPrepared
  else
    GPUBackend.executeWithConfigCached ctx
      (fusedQ4KMLinearDP4AKernel wQ.config)
      [("weights", wQ.weightBuf), ("input_q8", q8Buf), ("output", qBuf)]
      { numWorkgroups := (wQ.config.outDim, 1, 1), workgroupSize := { x := 32, y := 1, z := 1 } }
      (hash ("q4k-dp4a-matmul", wQ.config.inDim, wQ.config.outDim))
      wQ.dp4aMatmulPrepared

  -- Step 3: fused wK + wV matmul sharing the Q8_1 buffer.
  GPUBackend.executeWithConfigCached ctx
    (fusedQ4KMKVDP4AKernel wK.config)
    [("weights_k", wK.weightBuf),
     ("weights_v", wV.weightBuf),
     ("input_q8",  q8Buf),
     ("output_k",  kBuf),
     ("output_v",  vBuf)]
    { numWorkgroups := (wK.config.outDim, 1, 1), workgroupSize := { x := 32, y := 1, z := 1 } }
    (hash ("q4k-kv-dp4a", wK.config.inDim, wK.config.outDim))
    kvPreparedRef

/-- Fused Q4_K dp4a wK + wV forward pass (attention KV projection).

    Runs `quantizeQ8_1` once on the shared input, then the fused dp4a
    wK+wV kernel that writes both K and V outputs in a single dispatch.
    Halves the per-layer attention-projection dispatch count (2 → 1).

    Requires both wK and wV to be Q4_K with matching shape and subgroup
    support.  Uses `wK`'s Q8_1 scratch buffer + shared `preparedRef`. -/
def forwardFusedKV [GPUBackend β] (ctx : β)
    (wK wV : LinearLayer (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (inputBuf kBuf vBuf : GPUBackend.Buf β)
    (preparedRef : IO.Ref (Option (GPUBackend.CachedDispatch β)))
    : IO Unit := do
  if wK.quantFormat != .Q4_K then
    throw (IO.userError s!"forwardFusedKV: wK must be Q4_K, got {repr wK.quantFormat}")
  if wV.quantFormat != .Q4_K then
    throw (IO.userError s!"forwardFusedKV: wV must be Q4_K, got {repr wV.quantFormat}")
  if wK.config.inDim != wV.config.inDim || wK.config.outDim != wV.config.outDim then
    throw (IO.userError s!"forwardFusedKV: shape mismatch wK={wK.config.inDim}→{wK.config.outDim} wV={wV.config.inDim}→{wV.config.outDim}")

  let profiling ← profilingRef.get
  let startNs ← if profiling then IO.monoNanosNow else pure 0

  let useDP4A ← do
    let on ← dp4aEnabled.get
    pure (on && wK.config.inDim % 32 == 0)
  if !useDP4A then
    -- Fallback: call each projection individually.
    LinearLayer.forward ctx wK inputBuf kBuf
    LinearLayer.forward ctx wV inputBuf vBuf
    return

  let nQ8Blocks := wK.config.inDim / 32
  let q8BufBytes : USize := (nQ8Blocks * 9 * 4).toUSize
  let q8Buf ← match ← wK.dp4aQ8Buf.get with
    | some b => pure b
    | none =>
      let b ← GPUBackend.allocBuffer ctx q8BufBytes
      wK.dp4aQ8Buf.set (some b)
      pure b

  -- Step 1: Q8_1 quantize (shared input).
  GPUBackend.executeWithConfigCached ctx
    (quantizeQ8_1Kernel wK.config.inDim)
    [("input", inputBuf), ("output", q8Buf)]
    { numWorkgroups := (nQ8Blocks, 1, 1), workgroupSize := { x := 32, y := 1, z := 1 } }
    (hash ("q8_1-quantize", wK.config.inDim))
    wK.dp4aQuantizePrepared

  -- Step 2: fused wK + wV dp4a matmul.  Single-row variant (outDim is small
  -- — 256 for Gemma 4's K/V — so multi-row cooperation isn't useful).
  GPUBackend.executeWithConfigCached ctx
    (fusedQ4KMKVDP4AKernel wK.config)
    [("weights_k", wK.weightBuf),
     ("weights_v", wV.weightBuf),
     ("input_q8",  q8Buf),
     ("output_k",  kBuf),
     ("output_v",  vBuf)]
    { numWorkgroups := (wK.config.outDim, 1, 1), workgroupSize := { x := 32, y := 1, z := 1 } }
    (hash ("q4k-kv-dp4a", wK.config.inDim, wK.config.outDim))
    preparedRef

  if profiling then
    let endNs ← IO.monoNanosNow
    let delta := (endNs - startNs).toUInt64
    totalNanosRef.modify (· + delta)
    callCountRef.modify (· + 2)
    perShapeAdd wK.config.inDim wK.config.outDim delta

/-- Fused RMSNorm + Q4_K linear forward pass.

    Computes `output[i] = dot(W[i], RMSNorm(input))` in a single dispatch.
    Eliminates the separate RMSNorm dispatch and the intermediate normed buffer
    write/read round-trip through global memory.

    The RMS normalization is computed redundantly by each of the `outDim`
    workgroups, but the cost is just one warp reduction (5 shuffles) + strided
    reads of the input and norm scale buffers, which are hot in L2 cache.

    @param layer The Q4_K linear layer
    @param normScaleBuf RMSNorm weight/scale buffer
    @param inputBuf Raw (un-normalized) input buffer
    @param outputBuf Output buffer
    @param eps RMSNorm epsilon
    @param preparedRef Cache ref for the fused dispatch -/
def forwardFusedRMSNormLinear [GPUBackend β] (ctx : β)
    (layer : LinearLayer (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (normScaleBuf inputBuf outputBuf : GPUBackend.Buf β)
    (eps : Float)
    (preparedRef : IO.Ref (Option (GPUBackend.CachedDispatch β)))
    : IO Unit := do
  if layer.quantFormat != .Q4_K then
    throw (IO.userError s!"forwardFusedRMSNormLinear: must be Q4_K, got {repr layer.quantFormat}")

  let profiling ← profilingRef.get
  let startNs ← if profiling then IO.monoNanosNow else pure 0

  -- Fast path
  if let some p ← preparedRef.get then
    GPUBackend.replayCached ctx p (layer.config.outDim, 1, 1)
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
    ("norm_scale", normScaleBuf),
    ("output", outputBuf)
  ]
  let execConfig : Hesper.ExecConfig := {
    numWorkgroups := (layer.config.outDim, 1, 1)
    workgroupSize := { x := 32, y := 1, z := 1 }
  }
  let cacheKey : UInt64 :=
    hash ("q4k-rmsnorm-lin", layer.config.inDim, layer.config.outDim)
  GPUBackend.executeWithConfigCached ctx
    (fusedRMSNormQ4KMLinearKernel layer.config eps)
    namedBuffers execConfig cacheKey preparedRef
  if profiling then
    let endNs ← IO.monoNanosNow
    let delta := (endNs - startNs).toUInt64
    totalNanosRef.modify (· + delta)
    callCountRef.modify (· + 1)
    perShapeAdd layer.config.inDim layer.config.outDim delta

end Hesper.Layers.Linear
