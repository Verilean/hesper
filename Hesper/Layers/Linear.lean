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

/-- Phase-0 hybrid override.  When set, `forwardDP4A` skips hesper's Q8_1 quantize
    + dp4a matmul and calls this function instead with raw device pointers:
      `(inputF32Ptr, weightPtr, outputF32Ptr, inDim, outDim, quantFormatTag) : IO Bool`
    where `quantFormatTag = 0` for Q4_K and `1` for Q6_K.  Return `true` if the
    override dispatched the work; `false` to fall through to hesper's kernels.
    Installed by `Hesper.LlamaCppPTX.installOverride` (CUDA-only). -/
initialize llamaCppDp4aOverride :
  IO.Ref (Option (USize → USize → USize → Nat → Nat → Nat → IO Bool)) ← IO.mkRef none
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
    let d := Exp.vecX (Exp.unpack2x16float dmU32)
    let dmin := Exp.vecY (Exp.unpack2x16float dmU32)

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

/-- Batched Q4_K matmul: input `[M, inDim]` (row-major) × dequant(weights `[outDim, inDim]`)
    → output `[M, outDim]`. Dispatch `(outDim, M, 1)` workgroups; workgroup `(outIdx, row)`
    computes `output[row*outDim + outIdx]`. Same dequant as `fusedQ4KMLinearKernel`,
    weights shared across rows. Metal-native (no DP4A). -/
def fusedQ4KMBatchKernel (config : Config) (M : Nat) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let row := Exp.vec3Y wid
  let tid := Exp.vec3X lid
  let blocksPerRow := config.inDim / 256
  let totalWeightU32 := config.outDim * blocksPerRow * 36
  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareReadOnlyBuffer "input" (.array (.scalar .f32) (M * config.inDim))
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) (M * config.outDim))
  ShaderM.sharedNamed "shared_partial" (.array (.scalar .f32) workgroupSize)
  let inBounds := Exp.lt outIdx (Exp.litU32 config.outDim)
  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"
  let inRowBase := Exp.mul row (Exp.litU32 config.inDim)
  let rowBaseU32 := Exp.mul outIdx (Exp.litU32 (blocksPerRow * 36))
  ShaderM.loop tid (Exp.litU32 blocksPerRow) (Exp.litU32 workgroupSize) fun blockLocalIdx => do
    let blockU32Base := Exp.add rowBaseU32 (Exp.mul blockLocalIdx (Exp.litU32 36))
    let elemBase := Exp.mul blockLocalIdx (Exp.litU32 256)
    let dmU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" blockU32Base
    let d := Exp.vecX (Exp.unpack2x16float dmU32)
    let dmin := Exp.vecY (Exp.unpack2x16float dmU32)
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
          let elemIdxLow := Exp.add inRowBase (Exp.add elemBase (Exp.litU32 (c * 64 + l32 * 4 + b)))
          let elemIdxHigh := Exp.add inRowBase (Exp.add elemBase (Exp.litU32 (c * 64 + 32 + l32 * 4 + b)))
          let inLow ← ShaderM.readBuffer (ty := .scalar .f32) (n := M * config.inDim) "input" elemIdxLow
          let inHigh ← ShaderM.readBuffer (ty := .scalar .f32) (n := M * config.inDim) "input" elemIdxHigh
          let wLow := Exp.sub (Exp.mul d1 (Exp.toF32 qLow)) m1
          let wHigh := Exp.sub (Exp.mul d2 (Exp.toF32 qHigh)) m2
          ShaderM.assign "acc" (Exp.add acc (Exp.add (Exp.mul wLow inLow) (Exp.mul wHigh inHigh)))
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
  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    let totalSum ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_partial" (Exp.litU32 0)
    ShaderM.writeBuffer (ty := .scalar .f32) "output" (Exp.add (Exp.mul row (Exp.litU32 config.outDim)) outIdx) totalSum
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
    let d := Exp.vecX (Exp.unpack2x16float dmU32)
    let dmin := Exp.vecY (Exp.unpack2x16float dmU32)

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
    let d0 := Exp.vecX (Exp.unpack2x16float dm0U32)
    let dmin0 := Exp.vecY (Exp.unpack2x16float dm0U32)
    let sc0_0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add b0 (Exp.litU32 1))
    let sc0_1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add b0 (Exp.litU32 2))
    let sc0_2 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add b0 (Exp.litU32 3))

    -- Row1 block header
    let b1 := Exp.add row1BaseU32 blockOffsetU32
    let dm1U32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" b1
    let d1 := Exp.vecX (Exp.unpack2x16float dm1U32)
    let dmin1 := Exp.vecY (Exp.unpack2x16float dm1U32)
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
    let d := Exp.vecX (Exp.unpack2x16float dmU32)
    let dmin := Exp.vecY (Exp.unpack2x16float dmU32)

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
      (Exp.toF32U scaleU, Exp.toF32U minU)

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
    let d := Exp.vecX (Exp.unpack2x16float dmU32)
    let dmin := Exp.vecY (Exp.unpack2x16float dmU32)
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
      (Exp.toF32U scaleU, Exp.toF32U minU)

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
    let d := Exp.vecX (Exp.unpack2x16float dmU32)
    let dmin := Exp.vecY (Exp.unpack2x16float dmU32)
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
      (Exp.toF32U scaleU, Exp.toF32U minU)

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

  -- 1b. Sum of input (not of quantised ints): `ds.y` in llama.cpp's
  -- Q8_1 block header.
  ShaderM.varNamed "sumX" (.scalar .f32) (Exp.subgroupAdd x)
  let sumX : Exp (.scalar .f32) := Exp.var "sumX"

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

  -- 4. Thread 0 writes header: half2(d, sumX) packed into a single u32,
  -- matching llama.cpp's `block_q8_1.ds`.  Allows shared consumption by
  -- hesper's dp4a matmul AND llama.cpp's PTX matmul.
  let hdrOff := Exp.mul blockIdx (Exp.litU32 9)
  ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
    let packed : Exp (.scalar .u32) :=
      Exp.pack2x16float (Exp.vec2 d sumX)
    ShaderM.writeBuffer (ty := .scalar .u32) "output" hdrOff packed
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

/-- Batched Q8_1 quantize: processes `seqLen` rows of `inDim` floats.
    Input layout: `input[col * inDim + i]` for col ∈ [0, seqLen), i ∈ [0, inDim).
    Output layout: `output[col * nBlocks * 9 + block * 9 + ...]` (Q8_1 blocks per column).
    Grid: `(nBlocks, seqLen, 1) × 32 threads`.  Same algorithm as `quantizeQ8_1Kernel`
    but with a column offset derived from `blockIdx.y`. -/
def quantizeQ8_1BatchKernel (inDim seqLen : Nat) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let blockIdx := Exp.vec3X wid     -- block within row
  let colIdx := Exp.vec3Y wid       -- which column (sequence position)
  let tid := Exp.vec3X lid

  let nBlocks := inDim / 32
  let totalInputSize := inDim * seqLen
  let totalOutputU32 := nBlocks * 9 * seqLen

  let _input ← ShaderM.declareReadOnlyBuffer "input" (.array (.scalar .f32) totalInputSize)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .u32) totalOutputU32)

  let colInputBase := Exp.mul colIdx (Exp.litU32 inDim)
  let elemIdx := Exp.add colInputBase (Exp.add (Exp.mul blockIdx (Exp.litU32 32)) tid)
  let x ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalInputSize) "input" elemIdx

  let absX := Exp.select (Exp.lt x (Exp.litF32 0.0)) (Exp.sub (Exp.litF32 0.0) x) x
  ShaderM.varNamed "amax" (.scalar .f32) (Exp.subgroupMax absX)
  let amax : Exp (.scalar .f32) := Exp.var "amax"
  ShaderM.varNamed "sumX" (.scalar .f32) (Exp.subgroupAdd x)
  let sumX : Exp (.scalar .f32) := Exp.var "sumX"

  ShaderM.varNamed "d_q8" (.scalar .f32) (Exp.div amax (Exp.litF32 127.0))
  let d : Exp (.scalar .f32) := Exp.var "d_q8"

  ShaderM.varNamed "qF32" (.scalar .f32)
    (Exp.select (Exp.eq d (Exp.litF32 0.0)) (Exp.litF32 0.0) (Exp.div x d))
  let qF32 : Exp (.scalar .f32) := Exp.var "qF32"
  ShaderM.varNamed "qByte" (.scalar .u32)
    (Exp.bitAnd (Exp.roundToI32 qF32) (Exp.litU32 0xFF))
  let qByte : Exp (.scalar .u32) := Exp.var "qByte"

  let colOutputBase := Exp.mul colIdx (Exp.litU32 (nBlocks * 9))
  let hdrOff := Exp.add colOutputBase (Exp.mul blockIdx (Exp.litU32 9))
  ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
    let packed : Exp (.scalar .u32) :=
      Exp.pack2x16float (Exp.vec2 d sumX)
    ShaderM.writeBuffer (ty := .scalar .u32) "output" hdrOff packed
  ) (pure ())

  ShaderM.sharedNamed "shared_q" (.array (.scalar .u32) 32)
  ShaderM.writeWorkgroup (ty := .scalar .u32) "shared_q" tid qByte
  ShaderM.barrier

  let laneQuarter := Exp.div tid (Exp.litU32 4)
  let isQuarterLane := Exp.eq (Exp.sub tid (Exp.mul laneQuarter (Exp.litU32 4))) (Exp.litU32 0)
  ShaderM.if_ isQuarterLane (do
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

/-- Batched Q4_K × Q8_1 matmul: `[outDim, inDim] × [inDim, seqLen] → [outDim, seqLen]`.
    Grid: `(outDim, seqLen, 1) × 32 threads`.  Each WG computes one output element.
    Column-major output: `output[col * outDim + row]`.
    Q8_1 input is column-sliced: `q8[col * nQ8Blocks * 9 + ...]`. -/
def q4kMatmulBatchKernel (config : Config) (seqLen : Nat) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid   -- output row
  let colIdx := Exp.vec3Y wid   -- sequence position
  let tid := Exp.vec3X lid

  let blocksPerRow := config.inDim / 256
  let totalWeightU32 := config.outDim * blocksPerRow * 36
  let q8BlocksPerRow := config.inDim / 32
  let q8InputU32Size := q8BlocksPerRow * 9 * seqLen
  let totalOutputSize := config.outDim * seqLen

  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareReadOnlyBuffer "input_q8" (.array (.scalar .u32) q8InputU32Size)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) totalOutputSize)

  let inBounds := Exp.lt outIdx (Exp.litU32 config.outDim)

  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"

  let rowBaseU32 := Exp.mul outIdx (Exp.litU32 (blocksPerRow * 36))
  let q8ColBase := Exp.mul colIdx (Exp.litU32 (q8BlocksPerRow * 9))

  -- Top-half lanes (tid>>4 == 1) pick up the next block per iteration —
  -- halves trip count and removes the `*0.5` duplicate-work correction.
  let laneLow := Exp.bitAnd tid (Exp.litU32 15)
  let blockOff := Exp.shiftRight tid (Exp.litU32 4)  -- 0 or 1
  let pairIdx := Exp.div laneLow (Exp.litU32 4)
  let elemOff := Exp.sub laneLow (Exp.mul pairIdx (Exp.litU32 4))
  let bq8Off := Exp.mul pairIdx (Exp.litU32 2)

  let halvedTrip := (blocksPerRow + 1) / 2
  ShaderM.loop (Exp.litU32 0) (Exp.litU32 halvedTrip) (Exp.litU32 1) fun iter => do
    let blockIdx := Exp.add (Exp.mul iter (Exp.litU32 2)) blockOff
    let blockInRange := Exp.lt blockIdx (Exp.litU32 blocksPerRow)
    let safeBlockIdx := Exp.select blockInRange blockIdx (Exp.litU32 0)
    let blockU32Base := Exp.add rowBaseU32 (Exp.mul safeBlockIdx (Exp.litU32 36))
    let dmU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" blockU32Base
    let dF := Exp.vecX (Exp.unpack2x16float dmU32)
    let dminF := Exp.vecY (Exp.unpack2x16float dmU32)

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
      (Exp.toF32U scaleU, Exp.toF32U minU)

    let (scA, mA) := extractScaleMin bq8Off
    let (scB, mB) := extractScaleMin (Exp.add bq8Off (Exp.litU32 1))

    let q4BaseIdx := Exp.add blockU32Base
      (Exp.add (Exp.litU32 4) (Exp.add (Exp.mul bq8Off (Exp.litU32 4)) elemOff))
    let v0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" q4BaseIdx
    let v1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add q4BaseIdx (Exp.litU32 4))

    -- Q8_1 reads: offset by colIdx into the batched Q8_1 buffer
    let q8Sub0Base := Exp.add q8ColBase (Exp.add (Exp.mul safeBlockIdx (Exp.litU32 (8 * 9))) (Exp.mul bq8Off (Exp.litU32 9)))
    let q8Sub1Base := Exp.add q8Sub0Base (Exp.litU32 9)
    let u0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" (Exp.add q8Sub0Base (Exp.add (Exp.litU32 1) elemOff))
    let u1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" (Exp.add q8Sub0Base (Exp.add (Exp.litU32 5) elemOff))
    let u2 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" (Exp.add q8Sub1Base (Exp.add (Exp.litU32 1) elemOff))
    let u3 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" (Exp.add q8Sub1Base (Exp.add (Exp.litU32 5) elemOff))
    let q8Hdr0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" q8Sub0Base
    let q8Hdr1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" q8Sub1Base
    -- Q8_1 header is now half2(d, sum) packed in a u32.  Extract `d` via
    -- the low f16 (sum lives in the high f16 — currently unused by hesper's
    -- matmul, but matches llama.cpp layout).
    -- Hoist the f16→f32 conversion: each d8A/d8B is referenced 2-4× below.
    -- Without a ShaderM.var binding, CSE still has to re-emit the
    -- `mov.b32 {lo,hi}, r; cvt.f32.f16` pair for every reference.  The
    -- explicit bind + `Exp.var` forces a single register reuse.
    let d8AName ← ShaderM.var (.scalar .f32) (Exp.vecX (Exp.unpack2x16float q8Hdr0))
    let d8BName ← ShaderM.var (.scalar .f32) (Exp.vecX (Exp.unpack2x16float q8Hdr1))
    let d8A : Exp (.scalar .f32) := Exp.var d8AName
    let d8B : Exp (.scalar .f32) := Exp.var d8BName

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
    let gatedContrib := Exp.select blockInRange blockContrib (Exp.litF32 0.0)
    ShaderM.assign "acc" (Exp.add acc gatedContrib)

  -- Lanes 0..15 and 16..31 now cover distinct blocks, so the full
  -- subgroup sum is the exact row dot — no `*0.5` correction.
  ShaderM.varNamed "total" (.scalar .f32) (Exp.subgroupAdd acc)
  let total : Exp (.scalar .f32) := Exp.var "total"

  -- Column-major output: output[col * outDim + row]
  let outOff := Exp.add (Exp.mul colIdx (Exp.litU32 config.outDim)) outIdx
  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outOff total
  ) (pure ())

/-- MMQ Phase 2: smem-staged X tile for actual perf win.

    Same output shape as Phase 1c (32 rows × 8 cols per WG, 256 threads),
    but loads X (Q4_K block) into shared memory once per super-block and
    reuses across all 8 column threads. This is the actual MMQ optimisation
    that llama.cpp's `mul_mat_q_kernel` exploits.

    Smem layout per super-block load (one Q4_K block per row):
      x_block: 32 rows × 36 ints = 1152 u32 = 4.6 KB
        layout: row-major, 36 ints per row [dm, sc0, sc1, sc2, qs0..qs31]
        no padding (36 != 32 + power-of-2, but bank conflicts are minor for
        our access pattern — each thread reads contiguous ints from one row)

    K-loop:
      for kbx0 in 0..blocksPerRow:
        # cooperative load: 256 threads, 32 rows × 36 ints = 1152 ints
        # → 1152/256 = 4.5 ints/thread, do 5 strided iters
        for it in 0..5:
          idx = it*256 + tid
          if idx < 1152:
            x_block[idx] = weights[i_blk*32 * blocks*36 + (idx/36)*blocks*36 + kbx0*36 + (idx%36)]
        barrier
        # per-thread dot using smem reads instead of global
        for pairIdx in 0..4:
          for elemOff in 0..4:
            v0, v1 from x_block[laneId * 36 + offsets]
            u0..u3 from global Y (per warp/col, no sharing)
            dp4a + accumulate
        barrier (before next super-block load)

    Wired behind HESPER_PREFILL_MMQ2=1.
-/
def q4kMatmulBatchMMQ2Kernel (config : Config) (seqLen : Nat) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let i_blk := Exp.vec3X wid
  let j_blk := Exp.vec3Y wid
  let tid := Exp.vec3X lid
  let warpId := Exp.shiftRight tid (Exp.litU32 5)
  let laneId := Exp.bitAnd tid (Exp.litU32 31)

  let blocksPerRow := config.inDim / 256
  let q8BlocksPerRow := config.inDim / 32
  let totalWeightU32 := config.outDim * blocksPerRow * 36
  let q8InputU32Size := q8BlocksPerRow * 9 * seqLen
  let totalOutputSize := config.outDim * seqLen

  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareReadOnlyBuffer "input_q8" (.array (.scalar .u32) q8InputU32Size)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) totalOutputSize)

  -- Smem: 32 rows × 36 ints. Stride=37 to break 4-way bank conflicts on
  -- the qs reads (most accesses are at offset 4 + 4*bq8Off + elemOff which
  -- has variable stride per-pair).
  let xStride : Nat := 37
  ShaderM.sharedNamed "x_block" (.array (.scalar .u32) (32 * xStride))

  let i_row := Exp.add (Exp.mul i_blk (Exp.litU32 32)) laneId
  let j_col := Exp.add (Exp.mul j_blk (Exp.litU32 8)) warpId
  let i_in := Exp.lt i_row (Exp.litU32 config.outDim)
  let j_in := Exp.lt j_col (Exp.litU32 seqLen)
  let validThread := Exp.and i_in j_in

  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"

  let q8ColBase := Exp.mul j_col (Exp.litU32 (q8BlocksPerRow * 9))

  ShaderM.loop (Exp.litU32 0) (Exp.litU32 blocksPerRow) (Exp.litU32 1) fun kbx0 => do
    -- Phase A: cooperative X load. 256 threads, 32 rows × 36 ints = 1152 ints.
    -- Each thread does 5 strided loads (256 * 5 = 1280 ≥ 1152), masked.
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 5) (Exp.litU32 1) fun it => do
      let flatIdx := Exp.add (Exp.mul it (Exp.litU32 256)) tid
      ShaderM.if_ (Exp.lt flatIdx (Exp.litU32 (32 * 36))) (do
        let rowIdx := Exp.div flatIdx (Exp.litU32 36)
        let intInRow := Exp.sub flatIdx (Exp.mul rowIdx (Exp.litU32 36))
        let global_row := Exp.add (Exp.mul i_blk (Exp.litU32 32)) rowIdx
        let inBoundsLoad := Exp.lt global_row (Exp.litU32 config.outDim)
        ShaderM.if_ inBoundsLoad (do
          let blockBase := Exp.add (Exp.mul global_row (Exp.litU32 (blocksPerRow * 36)))
                                    (Exp.mul kbx0 (Exp.litU32 36))
          let w ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32)
                    "weights" (Exp.add blockBase intInRow)
          ShaderM.assignIndex (ty := .scalar .u32) "x_block"
            (Exp.add (Exp.mul rowIdx (Exp.litU32 xStride)) intInRow) w
        ) (pure ())
      ) (pure ())
    ShaderM.barrier

    -- Phase B: per-thread dot using smem-staged X.
    ShaderM.if_ validThread (do
      let xRowBase := Exp.mul laneId (Exp.litU32 xStride)
      -- Read block metadata once per super-block from smem.
      let dmU32 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := 32 * xStride)
                    "x_block" xRowBase
      let dF := Exp.vecX (Exp.unpack2x16float dmU32)
      let dminF := Exp.vecY (Exp.unpack2x16float dmU32)
      let sc0 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := 32 * xStride)
                  "x_block" (Exp.add xRowBase (Exp.litU32 1))
      let sc1 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := 32 * xStride)
                  "x_block" (Exp.add xRowBase (Exp.litU32 2))
      let sc2 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := 32 * xStride)
                  "x_block" (Exp.add xRowBase (Exp.litU32 3))

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
        (Exp.toF32U scaleU, Exp.toF32U minU)

      ShaderM.varNamed "blockSumfD" (.scalar .f32) (Exp.litF32 0.0)
      ShaderM.varNamed "blockSumfM" (.scalar .f32) (Exp.litF32 0.0)
      let blockSumfD : Exp (.scalar .f32) := Exp.var "blockSumfD"
      let blockSumfM : Exp (.scalar .f32) := Exp.var "blockSumfM"

      ShaderM.loop (Exp.litU32 0) (Exp.litU32 4) (Exp.litU32 1) fun pairIdx => do
        let bq8Off := Exp.mul pairIdx (Exp.litU32 2)
        let bq8OffP1 := Exp.add bq8Off (Exp.litU32 1)
        let (scA, mA) := extractScaleMin bq8Off
        let (scB, mB) := extractScaleMin bq8OffP1

        let q8SuperBase := Exp.add q8ColBase (Exp.mul kbx0 (Exp.litU32 (8 * 9)))
        let q8Sub0Base := Exp.add q8SuperBase (Exp.mul bq8Off (Exp.litU32 9))
        let q8Sub1Base := Exp.add q8SuperBase (Exp.mul bq8OffP1 (Exp.litU32 9))

        ShaderM.loop (Exp.litU32 0) (Exp.litU32 4) (Exp.litU32 1) fun elemOff => do
          -- v0, v1: read from smem (qs at offset 4 + bq8Off*4 + elemOff in row stride)
          let qsOff := Exp.add (Exp.litU32 4)
            (Exp.add (Exp.mul bq8Off (Exp.litU32 4)) elemOff)
          let v0 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := 32 * xStride)
                     "x_block" (Exp.add xRowBase qsOff)
          let v1 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := 32 * xStride)
                     "x_block" (Exp.add xRowBase (Exp.add qsOff (Exp.litU32 4)))

          -- u0..u3 from global Y (small, no smem benefit)
          let off1e := Exp.add (Exp.litU32 1) elemOff
          let off5e := Exp.add (Exp.litU32 5) elemOff
          let u0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size)
                     "input_q8" (Exp.add q8Sub0Base off1e)
          let u1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size)
                     "input_q8" (Exp.add q8Sub0Base off5e)
          let u2 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size)
                     "input_q8" (Exp.add q8Sub1Base off1e)
          let u3 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size)
                     "input_q8" (Exp.add q8Sub1Base off5e)
          let q8Hdr0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size)
                         "input_q8" q8Sub0Base
          let q8Hdr1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size)
                         "input_q8" q8Sub1Base
          let d8A := Exp.vecX (Exp.unpack2x16float q8Hdr0)
          let d8B := Exp.vecX (Exp.unpack2x16float q8Hdr1)

          let v0i0 := Exp.bitAnd v0 (Exp.litU32 0x0F0F0F0F)
          let v1i0 := Exp.bitAnd v1 (Exp.litU32 0x0F0F0F0F)
          let acc0 := Exp.dot4I8Packed v0i0 u0
          let dot1_0 := Exp.dot4I8Packed v1i0 u1
          let dot1_0Comb := Exp.add acc0 dot1_0
          let sumU_0 := Exp.add (Exp.dot4I8Packed (Exp.litU32 0x01010101) u0)
                                 (Exp.dot4I8Packed (Exp.litU32 0x01010101) u1)
          let sumfD_0 := Exp.mul d8A (Exp.mul (Exp.toF32 dot1_0Comb) scA)
          let sumfM_0 := Exp.mul d8A (Exp.mul (Exp.toF32 sumU_0) mA)

          let v0i1 := Exp.bitAnd (Exp.shiftRight v0 (Exp.litU32 4)) (Exp.litU32 0x0F0F0F0F)
          let v1i1 := Exp.bitAnd (Exp.shiftRight v1 (Exp.litU32 4)) (Exp.litU32 0x0F0F0F0F)
          let acc1 := Exp.dot4I8Packed v0i1 u2
          let dot1_1 := Exp.dot4I8Packed v1i1 u3
          let dot1_1Comb := Exp.add acc1 dot1_1
          let sumU_1 := Exp.add (Exp.dot4I8Packed (Exp.litU32 0x01010101) u2)
                                 (Exp.dot4I8Packed (Exp.litU32 0x01010101) u3)
          let sumfD_1 := Exp.mul d8B (Exp.mul (Exp.toF32 dot1_1Comb) scB)
          let sumfM_1 := Exp.mul d8B (Exp.mul (Exp.toF32 sumU_1) mB)

          ShaderM.assign "blockSumfD" (Exp.add blockSumfD (Exp.add sumfD_0 sumfD_1))
          ShaderM.assign "blockSumfM" (Exp.add blockSumfM (Exp.add sumfM_0 sumfM_1))

      let blockContrib := Exp.sub (Exp.mul dF blockSumfD) (Exp.mul dminF blockSumfM)
      ShaderM.assign "acc" (Exp.add acc blockContrib)
    ) (pure ())
    ShaderM.barrier  -- before next super-block load

  ShaderM.if_ validThread (do
    let outOff := Exp.add (Exp.mul j_col (Exp.litU32 config.outDim)) i_row
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outOff acc
  ) (pure ())


/-- MMQ Phase 5 (rev. 2026-05-02): tile-shrunk Q4_K matmul.

    Reduced from llama.cpp's `mul_mat_q<Q4_K, mmq_x=64>` to **half tile**
    after PTX diff (docs/prefill-stubs/comparison.md) showed hesper's
    full-shape MMQ5 emits **25× larger PTX** than llama (16,473 vs 640
    lines) due to Lean `for in [0:N] do` static unroll generating
    32× duplicated dp4a/load chains.

    **Shape**:
      - mmq_y = 64 rows / WG   (was 128 — 4 row groups × 16 lanes' subset)
      - mmq_x = 32 cols / WG   (was 64 — 4 col groups × 8 warpId)
      - nwarps = 8, warp_size = 32, 256 threads / WG
      - **One WG produces 2048 outputs** (vs old 8192)
      - Per-thread accumulators: `mmq_y/warp_size × mmq_x/nwarps = 2×4 = 8`
        outputs (rows {laneId, 32+laneId} × cols {warpId, 8+warpId,
        16+warpId, 24+warpId}).

    Why half: hesper's Lean static unroll multiplies the inner-dot code
    by `iIter*jIter`. Going from 4×8=32 to 2×4=8 cuts PTX 4× and brings
    accumulator count to 8 (still register-resident, no smem accumulator
    needed).

    **Smem** (~17 KB / WG):
      - `x_block[64 × 37]`  (2368 u32, ~9.5 KB): 64 row tile, 36 ints + pad
      - `y_block[32 × 73]`  (2336 u32, ~9.1 KB): 32 col tile, 72 ints + pad

    Cooperative loads:
      - X: 64 × 36 = 2304 ints, 256 threads × 9 strided iters = 2304 ✓
      - Y: 32 × 72 = 2304 ints, 256 threads × 9 strided iters = 2304 ✓

    Inner dot: 4 (pairIdx) × 4 (elemOff) × per (j_iter, i_iter) ≈
      16 × 8 = 128 dp4a / thread / super-block. With 10 super-blocks
      at K=2560, that's 1280 dp4a / thread total. (Was 5120 / thread.)

    Wired behind HESPER_PREFILL_MMQ5=1 (and auto-selected when seqLen >= 32).
    Old full-128/64 MMQ5 retired with this revision per smem 48-KB budget. -/
def q4kMatmulBatchMMQ5Kernel (config : Config) (seqLen : Nat) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let i_blk := Exp.vec3X wid       -- block of 64 rows
  let j_blk := Exp.vec3Y wid       -- block of 32 cols
  let tid := Exp.vec3X lid         -- 0..255
  let warpId := Exp.shiftRight tid (Exp.litU32 5)  -- 0..7
  let laneId := Exp.bitAnd tid (Exp.litU32 31)     -- 0..31

  let blocksPerRow := config.inDim / 256
  let q8BlocksPerRow := config.inDim / 32
  let totalWeightU32 := config.outDim * blocksPerRow * 36
  let q8InputU32Size := q8BlocksPerRow * 9 * seqLen
  let totalOutputSize := config.outDim * seqLen

  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareReadOnlyBuffer "input_q8" (.array (.scalar .u32) q8InputU32Size)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) totalOutputSize)

  -- launch_bounds(256, 2): force ptxas to fit ≥ 2 blocks/SM for occupancy.
  -- ncu measured MMQ5 at 75 reg/thread → 1 block/SM (17.8% occupancy).
  -- Target 128 reg/thread max → 2 block/SM = 35% occupancy. RTX 4070 Ti has
  -- 100 KB smem/SM so 2 × 18.8 KB = 37.6 KB fits.
  ShaderM.setMaxnreg 128
  ShaderM.setMinnctapersm 2

  -- X tile: 64 rows × 37 stride (36 packed Q4_K ints + 1 pad).
  let xStride : Nat := 37
  ShaderM.sharedNamed "x_block" (.array (.scalar .u32) (64 * xStride))
  -- Y tile: 32 cols × 73 stride (72 std-layout Q8_1 ints + 1 pad).
  let yStride : Nat := 73
  ShaderM.sharedNamed "y_block" (.array (.scalar .u32) (32 * yStride))

  -- Initialize 8 accumulators (acc_j0_i0 .. acc_j3_i1).
  for jIter in [0:4] do
    for iIter in [0:2] do
      ShaderM.varNamed s!"acc_{jIter}_{iIter}" (.scalar .f32) (Exp.litF32 0.0)

  ShaderM.loop (Exp.litU32 0) (Exp.litU32 blocksPerRow) (Exp.litU32 1) fun kbx0 => do
    -- Phase A: cooperative X load. 64 rows × 36 ints = 2304 ints, 256 threads × 9 iters.
    -- 256*9 = 2304 = 64*36 exactly, flatIdx never overshoots.
    -- Gemma 4 has all matrix dims (2048/2560/5120/8192/16384) divisible by 64,
    -- so global_row < outDim always holds → no inner bound check.
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 9) (Exp.litU32 1) fun it => do
      let flatIdx := Exp.add (Exp.mul it (Exp.litU32 256)) tid
      let rowIdx := Exp.div flatIdx (Exp.litU32 36)
      let intInRow := Exp.sub flatIdx (Exp.mul rowIdx (Exp.litU32 36))
      let global_row := Exp.add (Exp.mul i_blk (Exp.litU32 64)) rowIdx
      let blockBase := Exp.add (Exp.mul global_row (Exp.litU32 (blocksPerRow * 36)))
                                (Exp.mul kbx0 (Exp.litU32 36))
      let w ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32)
                "weights" (Exp.add blockBase intInRow)
      ShaderM.assignIndex (ty := .scalar .u32) "x_block"
        (Exp.add (Exp.mul rowIdx (Exp.litU32 xStride)) intInRow) w

    -- Phase A': cooperative Y load. 32 cols × 72 ints = 2304 ints, 256 × 9 iters.
    -- 256*9 == 32*72 exactly. Inner col bound stays since seqLen need not be
    -- a multiple of 32.
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 9) (Exp.litU32 1) fun it => do
      let flatIdx := Exp.add (Exp.mul it (Exp.litU32 256)) tid
      let yColIdx := Exp.div flatIdx (Exp.litU32 72)
      let intInY := Exp.sub flatIdx (Exp.mul yColIdx (Exp.litU32 72))
      let global_col := Exp.add (Exp.mul j_blk (Exp.litU32 32)) yColIdx
      let colInBounds := Exp.lt global_col (Exp.litU32 seqLen)
      ShaderM.if_ colInBounds (do
        let yColBase := Exp.add (Exp.mul global_col (Exp.litU32 (q8BlocksPerRow * 9)))
                                 (Exp.mul kbx0 (Exp.litU32 (8 * 9)))
        let v ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size)
                  "input_q8" (Exp.add yColBase intInY)
        ShaderM.assignIndex (ty := .scalar .u32) "y_block"
          (Exp.add (Exp.mul yColIdx (Exp.litU32 yStride)) intInY) v
      ) (pure ())
    ShaderM.barrier

    -- Phase B: per-thread dot, half-tile shape (2 i_iter × 4 j_iter).
    --
    -- For each i_iter we read X-row metadata (dm + 3 sc ints) once and
    -- iterate over all 4 j_iter (col groups) in the inner loop.
    -- iIter ∈ [0,2) processes rows {laneId, 32+laneId}.
    -- Gemma 4 outDim is always a multiple of 64 (= mmq_y), so all 2 rows
    -- per thread are in bounds → no `i_in` branch in the hot loop.
    for iIter in [0:2] do
      let i_local := Exp.add (Exp.mul (Exp.litU32 iIter) (Exp.litU32 32)) laneId
      let i_global := Exp.add (Exp.mul i_blk (Exp.litU32 64)) i_local
      let _ := i_global  -- unused once i_in is gone, but kept for clarity below
      do
        let xRowBase := Exp.mul i_local (Exp.litU32 xStride)
        let dmU32 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := 64 * xStride)
                      "x_block" xRowBase
        let dF := Exp.vecX (Exp.unpack2x16float dmU32)
        let dminF := Exp.vecY (Exp.unpack2x16float dmU32)
        let sc0 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := 64 * xStride)
                    "x_block" (Exp.add xRowBase (Exp.litU32 1))
        let sc1 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := 64 * xStride)
                    "x_block" (Exp.add xRowBase (Exp.litU32 2))
        let sc2 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := 64 * xStride)
                    "x_block" (Exp.add xRowBase (Exp.litU32 3))

        let extractScaleMin (is : Exp (.scalar .u32))
            : Exp (.scalar .f32) × Exp (.scalar .f32) :=
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
          (Exp.toF32U scaleU, Exp.toF32U minU)

        -- For this i_iter, iterate over (pairIdx, elemOff) once and
        -- multiply against all 8 j_iter Y-cols. X-side qs reads happen
        -- once per (pairIdx, elemOff) and reused across all j_iter.
        ShaderM.loop (Exp.litU32 0) (Exp.litU32 4) (Exp.litU32 1) fun pairIdx => do
          let bq8Off := Exp.mul pairIdx (Exp.litU32 2)
          let bq8OffP1 := Exp.add bq8Off (Exp.litU32 1)
          let (scA, mA) := extractScaleMin bq8Off
          let (scB, mB) := extractScaleMin bq8OffP1

          ShaderM.loop (Exp.litU32 0) (Exp.litU32 4) (Exp.litU32 1) fun elemOff => do
            let qsOff := Exp.add (Exp.litU32 4)
              (Exp.add (Exp.mul bq8Off (Exp.litU32 4)) elemOff)
            let v0 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := 64 * xStride)
                       "x_block" (Exp.add xRowBase qsOff)
            let v1 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := 64 * xStride)
                       "x_block" (Exp.add xRowBase (Exp.add qsOff (Exp.litU32 4)))
            let v0i0 := Exp.bitAnd v0 (Exp.litU32 0x0F0F0F0F)
            let v1i0 := Exp.bitAnd v1 (Exp.litU32 0x0F0F0F0F)
            let v0i1 := Exp.bitAnd (Exp.shiftRight v0 (Exp.litU32 4)) (Exp.litU32 0x0F0F0F0F)
            let v1i1 := Exp.bitAnd (Exp.shiftRight v1 (Exp.litU32 4)) (Exp.litU32 0x0F0F0F0F)

            let off1e := Exp.add (Exp.litU32 1) elemOff
            let off5e := Exp.add (Exp.litU32 5) elemOff

            -- Inner loop over 4 j_iter (col groups). The `j_in` bound check
            -- is invariant of (kbx0, pairIdx, elemOff) so we don't repeat it
            -- per (kbx0, pairIdx, elemOff). Out-of-range j_iter just reads
            -- and discards smem (smem loads are already padded; OOB threads
            -- write garbage to a dead accumulator that's never written back).
            for jIter in [0:4] do
              let j_local : Exp (.scalar .u32) :=
                Exp.add (Exp.mul (Exp.litU32 jIter) (Exp.litU32 8)) warpId
              let yColBaseSmem := Exp.mul j_local (Exp.litU32 yStride)
              let q8Sub0Smem := Exp.add yColBaseSmem (Exp.mul bq8Off (Exp.litU32 9))
              let q8Sub1Smem := Exp.add yColBaseSmem (Exp.mul bq8OffP1 (Exp.litU32 9))
              let q8Hdr0 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := 32 * yStride)
                             "y_block" q8Sub0Smem
              let q8Hdr1 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := 32 * yStride)
                             "y_block" q8Sub1Smem
              let d8A := Exp.vecX (Exp.unpack2x16float q8Hdr0)
              let d8B := Exp.vecX (Exp.unpack2x16float q8Hdr1)

              let u0 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := 32 * yStride)
                         "y_block" (Exp.add q8Sub0Smem off1e)
              let u1 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := 32 * yStride)
                         "y_block" (Exp.add q8Sub0Smem off5e)
              let u2 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := 32 * yStride)
                         "y_block" (Exp.add q8Sub1Smem off1e)
              let u3 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := 32 * yStride)
                         "y_block" (Exp.add q8Sub1Smem off5e)

              let acc0' := Exp.dot4I8Packed v0i0 u0
              let dot1_0 := Exp.dot4I8Packed v1i0 u1
              let dot1_0Comb := Exp.add acc0' dot1_0
              let sumU_0 := Exp.add (Exp.dot4I8Packed (Exp.litU32 0x01010101) u0)
                                     (Exp.dot4I8Packed (Exp.litU32 0x01010101) u1)
              let sumfD_0 := Exp.mul d8A (Exp.mul (Exp.toF32 dot1_0Comb) scA)
              let sumfM_0 := Exp.mul d8A (Exp.mul (Exp.toF32 sumU_0) mA)

              let acc1' := Exp.dot4I8Packed v0i1 u2
              let dot1_1 := Exp.dot4I8Packed v1i1 u3
              let dot1_1Comb := Exp.add acc1' dot1_1
              let sumU_1 := Exp.add (Exp.dot4I8Packed (Exp.litU32 0x01010101) u2)
                                     (Exp.dot4I8Packed (Exp.litU32 0x01010101) u3)
              let sumfD_1 := Exp.mul d8B (Exp.mul (Exp.toF32 dot1_1Comb) scB)
              let sumfM_1 := Exp.mul d8B (Exp.mul (Exp.toF32 sumU_1) mB)

              let pairD := Exp.add sumfD_0 sumfD_1
              let pairM := Exp.add sumfM_0 sumfM_1
              let contrib := Exp.sub (Exp.mul dF pairD) (Exp.mul dminF pairM)
              let accName := s!"acc_{jIter}_{iIter}"
              let accExp : Exp (.scalar .f32) := Exp.var accName
              ShaderM.assign accName (Exp.add accExp contrib)
    ShaderM.barrier

  -- Write back 8 outputs / thread. i_global is always in-bounds (outDim % 64 == 0
  -- for Gemma 4); j_global may exceed seqLen at the right edge of the seq tile.
  for jIter in [0:4] do
    for iIter in [0:2] do
      let i_local := Exp.add (Exp.mul (Exp.litU32 iIter) (Exp.litU32 32)) laneId
      let i_global := Exp.add (Exp.mul i_blk (Exp.litU32 64)) i_local
      let j_local : Exp (.scalar .u32) :=
        Exp.add (Exp.mul (Exp.litU32 jIter) (Exp.litU32 8)) warpId
      let j_global := Exp.add (Exp.mul j_blk (Exp.litU32 32)) j_local
      let j_in := Exp.lt j_global (Exp.litU32 seqLen)
      ShaderM.if_ j_in (do
        let outOff := Exp.add (Exp.mul j_global (Exp.litU32 config.outDim)) i_global
        let accExp : Exp (.scalar .f32) := Exp.var s!"acc_{jIter}_{iIter}"
        ShaderM.writeBuffer (ty := .scalar .f32) "output" outOff accExp
      ) (pure ())

/-- Q4_K × Q8_1 mat-vec body emitter (dp4a).

    llama.cppの `vec_dot_q4_K_q8_1_impl_vmmq` と同じアルゴリズム。
    各32スレッド subgroup が1出力要素 (1行) を計算。

    Thread layout: lane `t` ∈ [0,32) processes 2 Q8_1 sub-blocks
    (sub-block pair `t/4`, element offset `t%4`).  Grid is
    `(outDim, 1, 1) × 32`.

    Declares `weights` and `input_q8` buffers, runs the subgroup
    accumulation, and returns `(outIdx, tid, inBounds, total)` where
    `total` is the f32 dot-product result (valid on lane 0 of each
    warp).  The caller is responsible for declaring the output buffer
    and writing `total` (or a function of it) back.

    Reused by:
      * `fusedQ4KMLinearDP4AKernel` — identity epilogue (writes `total`).
      * `lowerMatmulQ4KWithEpilogueKernel` — evaluates a `ScalarExp`
        tail over `total` and caller-provided side buffers. -/
def emitQ4KMLinearDP4ABody (config : Config)
    : ShaderM (Exp (.scalar .u32) × Exp (.scalar .u32) × Exp (.scalar .bool) × Exp (.scalar .f32)) := do
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
    let dF := Exp.vecX (Exp.unpack2x16float dmU32)
    let dminF := Exp.vecY (Exp.unpack2x16float dmU32)

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
      (Exp.toF32U scaleU, Exp.toF32U minU)

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
    -- Q8_1 header is now half2(d, sum) packed in a u32.  Extract `d` via
    -- the low f16 (sum lives in the high f16 — currently unused by hesper's
    -- matmul, but matches llama.cpp layout).
    -- Hoist the f16→f32 conversion: each d8A/d8B is referenced 2-4× below.
    -- Without a ShaderM.var binding, CSE still has to re-emit the
    -- `mov.b32 {lo,hi}, r; cvt.f32.f16` pair for every reference.  The
    -- explicit bind + `Exp.var` forces a single register reuse.
    let d8AName ← ShaderM.var (.scalar .f32) (Exp.vecX (Exp.unpack2x16float q8Hdr0))
    let d8BName ← ShaderM.var (.scalar .f32) (Exp.vecX (Exp.unpack2x16float q8Hdr1))
    let d8A : Exp (.scalar .f32) := Exp.var d8AName
    let d8B : Exp (.scalar .f32) := Exp.var d8BName

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
  return (outIdx, tid, inBounds, total)

/-- The plain Q4_K dp4a matmul kernel: runs `emitQ4KMLinearDP4ABody`
    and writes the scalar result to `output` on lane 0.  Epilogue-less
    dispatch path; `Prim.matmulQ4K` lowers to this. -/
def fusedQ4KMLinearDP4AKernel (config : Config) : ShaderM Unit := do
  let (outIdx, tid, inBounds, total) ← emitQ4KMLinearDP4ABody config
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) config.outDim)
  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx total
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
      let dF := Exp.vecX (Exp.unpack2x16float dmU32)
      let dminF := Exp.vecY (Exp.unpack2x16float dmU32)

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
        (Exp.toF32U scaleU, Exp.toF32U minU)

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
    -- Q8_1 header is now half2(d, sum) packed in a u32.  Extract `d` via
    -- the low f16 (sum lives in the high f16 — currently unused by hesper's
    -- matmul, but matches llama.cpp layout).
    -- Hoist the f16→f32 conversion: each d8A/d8B is referenced 2-4× below.
    -- Without a ShaderM.var binding, CSE still has to re-emit the
    -- `mov.b32 {lo,hi}, r; cvt.f32.f16` pair for every reference.  The
    -- explicit bind + `Exp.var` forces a single register reuse.
    let d8AName ← ShaderM.var (.scalar .f32) (Exp.vecX (Exp.unpack2x16float q8Hdr0))
    let d8BName ← ShaderM.var (.scalar .f32) (Exp.vecX (Exp.unpack2x16float q8Hdr1))
    let d8A : Exp (.scalar .f32) := Exp.var d8AName
    let d8B : Exp (.scalar .f32) := Exp.var d8BName

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

  -- Top-half lanes pick up the next block so a 32-lane warp covers 2
  -- blocks per outer iteration — halves the trip count and eliminates
  -- the duplicate-work correction.
  let laneLow := Exp.bitAnd tid (Exp.litU32 15)
  let blockOff := Exp.shiftRight tid (Exp.litU32 4)
  let pairIdx := Exp.div laneLow (Exp.litU32 4)
  let elemOff := Exp.sub laneLow (Exp.mul pairIdx (Exp.litU32 4))
  let bq8Off := Exp.mul pairIdx (Exp.litU32 2)

  let halvedTrip := (blocksPerRow + 1) / 2
  ShaderM.loop (Exp.litU32 0) (Exp.litU32 halvedTrip) (Exp.litU32 1) fun iter => do
    let blockIdx := Exp.add (Exp.mul iter (Exp.litU32 2)) blockOff
    let blockInRange := Exp.lt blockIdx (Exp.litU32 blocksPerRow)
    let safeBlockIdx := Exp.select blockInRange blockIdx (Exp.litU32 0)
    let blockU32Base := Exp.add rowBaseU32 (Exp.mul safeBlockIdx (Exp.litU32 36))

    let processWeight (which : String) (acc : Exp (.scalar .f32))
        (u0 u1 u2 u3 : Exp (.scalar .u32)) (d8A d8B : Exp (.scalar .f32))
        : ShaderM (Exp (.scalar .f32)) := do
      let dmU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) which blockU32Base
      let dF := Exp.vecX (Exp.unpack2x16float dmU32)
      let dminF := Exp.vecY (Exp.unpack2x16float dmU32)

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
        (Exp.toF32U scaleU, Exp.toF32U minU)

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
      let gatedContrib := Exp.select blockInRange blockContrib (Exp.litF32 0.0)
      pure (Exp.add acc gatedContrib)

    -- Q8_1 input shared between K and V.
    let q8Sub0Base := Exp.add (Exp.mul safeBlockIdx (Exp.litU32 (8 * 9))) (Exp.mul bq8Off (Exp.litU32 9))
    let q8Sub1Base := Exp.add q8Sub0Base (Exp.litU32 9)
    let u0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" (Exp.add q8Sub0Base (Exp.add (Exp.litU32 1) elemOff))
    let u1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" (Exp.add q8Sub0Base (Exp.add (Exp.litU32 5) elemOff))
    let u2 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" (Exp.add q8Sub1Base (Exp.add (Exp.litU32 1) elemOff))
    let u3 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" (Exp.add q8Sub1Base (Exp.add (Exp.litU32 5) elemOff))
    let q8Hdr0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" q8Sub0Base
    let q8Hdr1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" q8Sub1Base
    -- Q8_1 header is now half2(d, sum) packed in a u32.  Extract `d` via
    -- the low f16 (sum lives in the high f16 — currently unused by hesper's
    -- matmul, but matches llama.cpp layout).
    -- Hoist the f16→f32 conversion: each d8A/d8B is referenced 2-4× below.
    -- Without a ShaderM.var binding, CSE still has to re-emit the
    -- `mov.b32 {lo,hi}, r; cvt.f32.f16` pair for every reference.  The
    -- explicit bind + `Exp.var` forces a single register reuse.
    let d8AName ← ShaderM.var (.scalar .f32) (Exp.vecX (Exp.unpack2x16float q8Hdr0))
    let d8BName ← ShaderM.var (.scalar .f32) (Exp.vecX (Exp.unpack2x16float q8Hdr1))
    let d8A : Exp (.scalar .f32) := Exp.var d8AName
    let d8B : Exp (.scalar .f32) := Exp.var d8BName

    let newAccK ← processWeight "weights_k" accK u0 u1 u2 u3 d8A d8B
    ShaderM.assign "accK" newAccK
    let newAccV ← processWeight "weights_v" accV u0 u1 u2 u3 d8A d8B
    ShaderM.assign "accV" newAccV

  -- Subgroup reduction — lanes 0..15 and 16..31 now cover distinct blocks
  -- so the full warp sum is the exact row dot; no ×0.5 correction needed.
  ShaderM.varNamed "totalK" (.scalar .f32) (Exp.subgroupAdd accK)
  ShaderM.varNamed "totalV" (.scalar .f32) (Exp.subgroupAdd accV)
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

  -- Lane decomposition uses laneId (not tid).  Top-half lanes (16..31) pick
  -- up the next block so that a 32-lane warp handles two blocks per outer
  -- iteration — halves outer trip count and eliminates the duplicate-work
  -- ×0.5 correction present in earlier versions of this kernel.
  let laneLow := Exp.bitAnd laneId (Exp.litU32 15)
  let blockOff := Exp.shiftRight laneId (Exp.litU32 4)  -- 0 or 1
  let pairIdx := Exp.div laneLow (Exp.litU32 4)
  let elemOff := Exp.sub laneLow (Exp.mul pairIdx (Exp.litU32 4))
  let bq8Off := Exp.mul pairIdx (Exp.litU32 2)

  let halvedTrip := (blocksPerRow + 1) / 2
  -- Unroll the outer loop at Lean codegen time.  halvedTrip is a
  -- compile-time constant (5 for Gemma 4's inDim=2560); emitting the
  -- iterations linearly lets the PTX compiler schedule loads across
  -- the whole body, hide ~4 rounds of global-memory latency, and
  -- fold the OOB guard entirely when blocksPerRow is even.
  for iterNat in [0 : halvedTrip] do
    let iterU32 := Exp.litU32 iterNat
    let blockIdx := Exp.add (Exp.mul iterU32 (Exp.litU32 2)) blockOff
    let blockInRange :=
      if 2 * iterNat + 1 < blocksPerRow then
        -- Both halves are always in range for this iteration.
        Exp.litBool true
      else
        Exp.lt blockIdx (Exp.litU32 blocksPerRow)
    -- Clamp OOB reads to block 0 so they hit valid memory; their
    -- contribution is gated below via `blockInRange`.
    let safeBlockIdx := Exp.select blockInRange blockIdx (Exp.litU32 0)
    let blockU32Base := Exp.add rowBaseU32 (Exp.mul safeBlockIdx (Exp.litU32 36))

    let processWeight (which : String) (acc : Exp (.scalar .f32))
        (u0 u1 u2 u3 : Exp (.scalar .u32)) (d8A d8B : Exp (.scalar .f32))
        : ShaderM (Exp (.scalar .f32)) := do
      let dmU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) which blockU32Base
      let dF := Exp.vecX (Exp.unpack2x16float dmU32)
      let dminF := Exp.vecY (Exp.unpack2x16float dmU32)

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
        (Exp.toF32U scaleU, Exp.toF32U minU)

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
      -- Gate OOB iterations (only possible when blocksPerRow is odd).
      let gatedContrib := Exp.select blockInRange blockContrib (Exp.litF32 0.0)
      pure (Exp.add acc gatedContrib)

    -- Q8_1 input from smem (shared across 4 warps in this WG).
    let q8Sub0Base := Exp.add (Exp.mul safeBlockIdx (Exp.litU32 (8 * 9))) (Exp.mul bq8Off (Exp.litU32 9))
    let q8Sub1Base := Exp.add q8Sub0Base (Exp.litU32 9)
    let u0 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := q8InputU32Size) "s_input_q8" (Exp.add q8Sub0Base (Exp.add (Exp.litU32 1) elemOff))
    let u1 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := q8InputU32Size) "s_input_q8" (Exp.add q8Sub0Base (Exp.add (Exp.litU32 5) elemOff))
    let u2 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := q8InputU32Size) "s_input_q8" (Exp.add q8Sub1Base (Exp.add (Exp.litU32 1) elemOff))
    let u3 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := q8InputU32Size) "s_input_q8" (Exp.add q8Sub1Base (Exp.add (Exp.litU32 5) elemOff))
    let q8Hdr0 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := q8InputU32Size) "s_input_q8" q8Sub0Base
    let q8Hdr1 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := q8InputU32Size) "s_input_q8" q8Sub1Base
    -- Q8_1 header is now half2(d, sum) packed in a u32.  Extract `d` via
    -- the low f16 (sum lives in the high f16 — currently unused by hesper's
    -- matmul, but matches llama.cpp layout).
    -- Hoist the f16→f32 conversion: each d8A/d8B is referenced 2-4× below.
    -- Without a ShaderM.var binding, CSE still has to re-emit the
    -- `mov.b32 {lo,hi}, r; cvt.f32.f16` pair for every reference.  The
    -- explicit bind + `Exp.var` forces a single register reuse.
    let d8AName ← ShaderM.var (.scalar .f32) (Exp.vecX (Exp.unpack2x16float q8Hdr0))
    let d8BName ← ShaderM.var (.scalar .f32) (Exp.vecX (Exp.unpack2x16float q8Hdr1))
    let d8A : Exp (.scalar .f32) := Exp.var d8AName
    let d8B : Exp (.scalar .f32) := Exp.var d8BName

    let newAccG ← processWeight "weights_gate" accG u0 u1 u2 u3 d8A d8B
    ShaderM.assign "accG" newAccG
    let newAccU ← processWeight "weights_up" accU u0 u1 u2 u3 d8A d8B
    ShaderM.assign "accU" newAccU

  -- Per-warp subgroup reductions.  Each of lanes 0..15 and 16..31 contributes
  -- distinct blocks, so the full subgroup sum is the exact row dot product
  -- — no duplicate-work correction needed.
  ShaderM.varNamed "totalG" (.scalar .f32) (Exp.subgroupAdd accG)
  ShaderM.varNamed "totalU" (.scalar .f32) (Exp.subgroupAdd accU)
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

/-- Single-Linear 4-row Q4_K dp4a kernel.

    Same workgroup layout as `fusedQ4KMGateUpDP4A4RowKernel` (4 rows per
    WG, 1 warp per row, 128 threads, smem-shared Q8_1 input) but with a
    single weight buffer and no GELU/Mul epilogue.  Used for wO,
    postLinear, PLE inpGate/proj, perLayer projection — every Q4_K
    Linear that is NOT the fused gate+up FFN.

    Compared to `fusedQ4KMLinearDP4A4WarpKernel` (the previous default,
    which uses 4 warps cooperatively on a SINGLE row):
      - 4× fewer workgroups dispatched (outDim/4 vs outDim)
      - 4× fewer threads launched (same 128/WG but 4 useful rows each)
      - input Q8_1 reused 4× across the 4 rows in smem (instead of 4×
        global re-reads)
      - matches llama.cpp's `mul_mat_vec_q<Q4_K, 1, nwarps=4>` pattern.

    Dispatch: (⌈outDim / 4⌉, 1, 1) workgroups × 128 threads. -/
def fusedQ4KMLinearDP4A4RowKernel (config : Config) : ShaderM Unit := do
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

  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input   ← ShaderM.declareReadOnlyBuffer "input_q8" (.array (.scalar .u32) q8InputU32Size)
  let _output  ← ShaderM.declareOutputBuffer "output"     (.array (.scalar .f32) config.outDim)

  -- Cooperative smem staging of Q8_1 input — same pattern as the
  -- gate+up 4-row kernel.  All 4 rows in the WG read the same input,
  -- so loading once into smem cuts global input traffic by 4×.
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

  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"

  let rowBaseU32 := Exp.mul outIdx (Exp.litU32 (blocksPerRow * 36))

  -- Lane decomposition (lanes 0..15 + 16..31 process 2 blocks per outer iter).
  let laneLow := Exp.bitAnd laneId (Exp.litU32 15)
  let blockOff := Exp.shiftRight laneId (Exp.litU32 4)  -- 0 or 1
  let pairIdx := Exp.div laneLow (Exp.litU32 4)
  let elemOff := Exp.sub laneLow (Exp.mul pairIdx (Exp.litU32 4))
  let bq8Off := Exp.mul pairIdx (Exp.litU32 2)

  let halvedTrip := (blocksPerRow + 1) / 2
  for iterNat in [0 : halvedTrip] do
    let iterU32 := Exp.litU32 iterNat
    let blockIdx := Exp.add (Exp.mul iterU32 (Exp.litU32 2)) blockOff
    let blockInRange :=
      if 2 * iterNat + 1 < blocksPerRow then
        Exp.litBool true
      else
        Exp.lt blockIdx (Exp.litU32 blocksPerRow)
    let safeBlockIdx := Exp.select blockInRange blockIdx (Exp.litU32 0)
    let blockU32Base := Exp.add rowBaseU32 (Exp.mul safeBlockIdx (Exp.litU32 36))

    let dmU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" blockU32Base
    let dF := Exp.vecX (Exp.unpack2x16float dmU32)
    let dminF := Exp.vecY (Exp.unpack2x16float dmU32)

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
      (Exp.toF32U scaleU, Exp.toF32U minU)

    let (scA, mA) := extractScaleMin bq8Off
    let (scB, mB) := extractScaleMin (Exp.add bq8Off (Exp.litU32 1))

    let q4BaseIdx := Exp.add blockU32Base
      (Exp.add (Exp.litU32 4) (Exp.add (Exp.mul bq8Off (Exp.litU32 4)) elemOff))
    let v0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" q4BaseIdx
    let v1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add q4BaseIdx (Exp.litU32 4))

    -- Q8_1 input from smem.
    let q8Sub0Base := Exp.add (Exp.mul safeBlockIdx (Exp.litU32 (8 * 9))) (Exp.mul bq8Off (Exp.litU32 9))
    let q8Sub1Base := Exp.add q8Sub0Base (Exp.litU32 9)
    let u0 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := q8InputU32Size) "s_input_q8" (Exp.add q8Sub0Base (Exp.add (Exp.litU32 1) elemOff))
    let u1 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := q8InputU32Size) "s_input_q8" (Exp.add q8Sub0Base (Exp.add (Exp.litU32 5) elemOff))
    let u2 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := q8InputU32Size) "s_input_q8" (Exp.add q8Sub1Base (Exp.add (Exp.litU32 1) elemOff))
    let u3 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := q8InputU32Size) "s_input_q8" (Exp.add q8Sub1Base (Exp.add (Exp.litU32 5) elemOff))
    let q8Hdr0 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := q8InputU32Size) "s_input_q8" q8Sub0Base
    let q8Hdr1 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := q8InputU32Size) "s_input_q8" q8Sub1Base
    let d8AName ← ShaderM.var (.scalar .f32) (Exp.vecX (Exp.unpack2x16float q8Hdr0))
    let d8BName ← ShaderM.var (.scalar .f32) (Exp.vecX (Exp.unpack2x16float q8Hdr1))
    let d8A : Exp (.scalar .f32) := Exp.var d8AName
    let d8B : Exp (.scalar .f32) := Exp.var d8BName

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
    let gatedContrib := Exp.select blockInRange blockContrib (Exp.litF32 0.0)
    ShaderM.assign "acc" (Exp.add acc gatedContrib)

  -- Per-warp subgroup reduction (no cross-warp smem needed: each warp owns 1 row).
  ShaderM.varNamed "total" (.scalar .f32) (Exp.subgroupAdd acc)
  let total : Exp (.scalar .f32) := Exp.var "total"

  ShaderM.if_ (Exp.and (Exp.eq laneId (Exp.litU32 0)) inBounds) (do
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx total
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

  -- Use laneId (not tid) for intra-row work distribution.  Top-half lanes
  -- handle the next block (halved outer trip count, no duplicated work).
  let laneLow := Exp.bitAnd laneId (Exp.litU32 15)
  let blockOff := Exp.shiftRight laneId (Exp.litU32 4)  -- 0 or 1
  let pairIdxInRow := Exp.div laneLow (Exp.litU32 4)
  let elemOff := Exp.sub laneLow (Exp.mul pairIdxInRow (Exp.litU32 4))
  let bq8Off := Exp.mul pairIdxInRow (Exp.litU32 2)

  let rowBaseU32 := Exp.mul outIdx (Exp.litU32 (blocksPerRow * 36))

  let halvedTrip := (blocksPerRow + 1) / 2
  ShaderM.loop (Exp.litU32 0) (Exp.litU32 halvedTrip) (Exp.litU32 1) fun iter => do
    let blockIdx := Exp.add (Exp.mul iter (Exp.litU32 2)) blockOff
    let blockInRange := Exp.lt blockIdx (Exp.litU32 blocksPerRow)
    let safeBlockIdx := Exp.select blockInRange blockIdx (Exp.litU32 0)
    let blockU32Base := Exp.add rowBaseU32 (Exp.mul safeBlockIdx (Exp.litU32 36))
    let dmU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" blockU32Base
    let dF := Exp.vecX (Exp.unpack2x16float dmU32)
    let dminF := Exp.vecY (Exp.unpack2x16float dmU32)

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
      (Exp.toF32U scaleU, Exp.toF32U minU)

    let (scA, mA) := extractScaleMin bq8Off
    let (scB, mB) := extractScaleMin (Exp.add bq8Off (Exp.litU32 1))

    let q4BaseIdx := Exp.add blockU32Base
      (Exp.add (Exp.litU32 4) (Exp.add (Exp.mul bq8Off (Exp.litU32 4)) elemOff))
    let v0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" q4BaseIdx
    let v1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add q4BaseIdx (Exp.litU32 4))

    let q8Sub0Base := Exp.add (Exp.mul safeBlockIdx (Exp.litU32 (8 * 9))) (Exp.mul bq8Off (Exp.litU32 9))
    let q8Sub1Base := Exp.add q8Sub0Base (Exp.litU32 9)

    let u0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" (Exp.add q8Sub0Base (Exp.add (Exp.litU32 1) elemOff))
    let u1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" (Exp.add q8Sub0Base (Exp.add (Exp.litU32 5) elemOff))
    let u2 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" (Exp.add q8Sub1Base (Exp.add (Exp.litU32 1) elemOff))
    let u3 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" (Exp.add q8Sub1Base (Exp.add (Exp.litU32 5) elemOff))

    let q8Hdr0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" q8Sub0Base
    let q8Hdr1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" q8Sub1Base
    -- Q8_1 header is now half2(d, sum) packed in a u32.  Extract `d` via
    -- the low f16 (sum lives in the high f16 — currently unused by hesper's
    -- matmul, but matches llama.cpp layout).
    -- Hoist the f16→f32 conversion: each d8A/d8B is referenced 2-4× below.
    -- Without a ShaderM.var binding, CSE still has to re-emit the
    -- `mov.b32 {lo,hi}, r; cvt.f32.f16` pair for every reference.  The
    -- explicit bind + `Exp.var` forces a single register reuse.
    let d8AName ← ShaderM.var (.scalar .f32) (Exp.vecX (Exp.unpack2x16float q8Hdr0))
    let d8BName ← ShaderM.var (.scalar .f32) (Exp.vecX (Exp.unpack2x16float q8Hdr1))
    let d8A : Exp (.scalar .f32) := Exp.var d8AName
    let d8B : Exp (.scalar .f32) := Exp.var d8BName

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
    let gatedContrib := Exp.select blockInRange blockContrib (Exp.litF32 0.0)
    ShaderM.assign "acc" (Exp.add acc gatedContrib)

  -- Each subgroup reduces independently.  Bottom and top halves cover
  -- distinct blocks, so the full subgroup sum is the exact partial — no
  -- ×0.5 correction.
  ShaderM.varNamed "total" (.scalar .f32) (Exp.subgroupAdd acc)
  let total : Exp (.scalar .f32) := Exp.var "total"

  -- Lane 0 of each subgroup writes its row.
  ShaderM.if_ (Exp.and (Exp.eq laneId (Exp.litU32 0)) inBounds) (do
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx total
  ) (pure ())

/-- Q4_K × Q8_1 dp4a matmul, **4-warp cooperative 1-row-per-WG** — the
    pattern llama.cpp's `mul_mat_vec_q<Q4_K, ncols_dst=1>` uses on sm_89.

    Thread layout mirrors llama.cpp's mul_mat_vec_q (mmvq.cu:389+, see
    docs/llama-fusion-analysis/02-mmvq-epilogue.md):

      tid = warpId*32 + laneId        (0..127)
      kbxStart = tid / (qi/vdr) = tid / 16   (0..7)
      iqs      = vdr * (tid % 16)    = 2 * (tid & 15)  (0,2,4,...,30)

    Each of the 16 threads sharing `kbxStart` covers one `iqs` slot of
    that block; together they compute all 16 (bq8Off, elemOff) pairs of
    the block exactly once.  The outer loop strides by
    `blocks_per_iter = vdr * nwarps * warp_size / qi = 2*4*32/32 = 8`.

    Dispatch: `(outDim, 1, 1)` workgroups × **128 threads**.  Works for
    any `blocksPerRow`: for blocksPerRow=10 (Gemma 4 inDim=2560),
    threads 0..31 (warp 0) handle blocks {0,1,8,9}, warps 1–3 handle
    blocks {2..7} — every block is covered exactly once, no duplicated
    work, no `*0.5` correction. -/
def fusedQ4KMLinearDP4A4WarpKernel (config : Config) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid            -- 0..127
  let warpId := Exp.shiftRight tid (Exp.litU32 5)  -- 0..3
  let laneId := Exp.bitAnd tid (Exp.litU32 31)     -- 0..31

  let blocksPerRow := config.inDim / 256
  let totalWeightU32 := config.outDim * blocksPerRow * 36
  let q8BlocksPerRow := config.inDim / 32
  let q8InputU32Size := q8BlocksPerRow * 9

  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareReadOnlyBuffer "input_q8" (.array (.scalar .u32) q8InputU32Size)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) config.outDim)

  -- smem for cross-warp partial sums only (4 floats = 16 B).
  ShaderM.sharedNamed "s_partial" (.array (.scalar .f32) 4)

  let inBounds := Exp.lt outIdx (Exp.litU32 config.outDim)

  -- llama.cpp mul_mat_vec_q<Q4_K, 1, nwarps=4> thread layout (sm_89 decode):
  --   qk=256, qi=32, vdr=2, warp_size=32, nwarps=4, total threads=128.
  --   blocks_per_iter = vdr * nwarps * warp_size / qi = 2*4*32/32 = 8
  --   kbx = tid / (qi/vdr) = tid / 16  → starting block index (0..7)
  --   kqs = vdr * (tid % (qi/vdr)) = 2 * (tid & 15)  → iqs in 0..30
  -- Each of the 16 threads sharing a kbxStart uniquely indexes (bq8Off, elemOff)
  -- via (tid & 15), so together they cover all 16 sub-slots of one block.
  -- No duplicated work — every thread contributes distinct products.
  -- Hoist thread-invariant subexpressions into explicit registers.  Lean's
  -- `Exp` lowering re-emits every use, and ptxas doesn't always CSE them
  -- back (PTX dump of this kernel showed `shr.u32 tid, 5` appearing 51×
  -- and `bfe.u32 tid, 0, 4` appearing 42× inside the outer loop body).
  -- Binding them via `ShaderM.var` forces one materialisation.
  let kbxStartName ← ShaderM.var (.scalar .u32) (Exp.shiftRight tid (Exp.litU32 4))
  let laneLowName  ← ShaderM.var (.scalar .u32) (Exp.bitAnd tid (Exp.litU32 15))
  let kbxStart : Exp (.scalar .u32) := Exp.var kbxStartName         -- 0..7
  let laneLow  : Exp (.scalar .u32) := Exp.var laneLowName
  let pairIdxInRowName ← ShaderM.var (.scalar .u32) (Exp.shiftRight laneLow (Exp.litU32 2))
  let elemOffName      ← ShaderM.var (.scalar .u32) (Exp.bitAnd laneLow (Exp.litU32 3))
  let pairIdxInRow : Exp (.scalar .u32) := Exp.var pairIdxInRowName  -- 0..3
  let elemOff      : Exp (.scalar .u32) := Exp.var elemOffName       -- 0..3
  let bq8OffName ← ShaderM.var (.scalar .u32) (Exp.shiftLeft pairIdxInRow (Exp.litU32 1))
  let bq8Off : Exp (.scalar .u32) := Exp.var bq8OffName              -- 0,2,4,6

  let rowBaseU32Name ← ShaderM.var (.scalar .u32)
    (Exp.mul outIdx (Exp.litU32 (blocksPerRow * 36)))
  let rowBaseU32 : Exp (.scalar .u32) := Exp.var rowBaseU32Name

  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"

  -- Outer loop: `for kbx = kbxStart; kbx < blocksPerRow; kbx += 8`.
  -- Encoded as a bounded iter loop.  On llama.cpp the `for (; kbx <
  -- blocksPerRow; kbx += 8)` exits early when OOB; hesper can't break
  -- so we wrap the loop body in `if blockInRange` — threads whose
  -- slot is past the end skip ALL global reads + dp4a work for that
  -- iteration instead of doing the work and masking the result.
  --
  -- Impact on the hot shapes: for inDim=2560 matmuls (wQ, wK, wV, wO,
  -- ffn_gate, ffn_up, PLE inp_gate/proj) blocksPerRow=10, so threads
  -- with kbxStart=3..7 do 1 useful iter + 1 skipped iter; without
  -- the guard they'd do 2 full iters with the 2nd masked to 0 — about
  -- 30 % of the lane-iterations on those kernels.
  let maxIter := (blocksPerRow + 7) / 8
  ShaderM.loop (Exp.litU32 0) (Exp.litU32 maxIter) (Exp.litU32 1) fun iter => do
    let blockIdxName ← ShaderM.var (.scalar .u32)
      (Exp.add kbxStart (Exp.mul iter (Exp.litU32 8)))
    let blockIdx : Exp (.scalar .u32) := Exp.var blockIdxName
    let blockInRange := Exp.lt blockIdx (Exp.litU32 blocksPerRow)
    ShaderM.if_ blockInRange (do
    -- Bind the per-iter base addresses as explicit registers so the 12 ld.global
    -- calls inside the iter all share the same address-chain materialisation.
    let blockU32BaseName ← ShaderM.var (.scalar .u32)
      (Exp.add rowBaseU32 (Exp.mul blockIdx (Exp.litU32 36)))
    let blockU32Base : Exp (.scalar .u32) := Exp.var blockU32BaseName

    -- Coalesce the 4 contiguous u32 loads (dmU32 + sc0..sc2 at offsets 0..3)
    -- into a single 128-bit ld.global.nc.v4.u32.  Q4_K block layout starts
    -- with 4 u32 of metadata (dm + 3× scale words) at byte offset 0, so it
    -- is naturally 16-byte aligned (block stride 36 u32 = 144 B).  Each
    -- thread still reads its own block (no smem broadcast — Option B
    -- avoids the divergent-control-flow hang that Option A hit), but
    -- ptxas can issue one MIO op for the 16 bytes instead of four
    -- separate scalar LDG.E sectors.
    let (dmU32, sc0, sc1, sc2) ← ShaderM.readBufferU32x4 "weights" blockU32Base
    -- f16x2-packed (d, dmin) → 2× cvt.f32.f16 instead of 15-instruction
    -- arithmetic decode that fp16ToF32 produced.  Bind via ShaderM.var so
    -- ptxas doesn't re-emit the cvt for every reference inside the loop.
    let dFName ← ShaderM.var (.scalar .f32) (Exp.vecX (Exp.unpack2x16float dmU32))
    let dminFName ← ShaderM.var (.scalar .f32) (Exp.vecY (Exp.unpack2x16float dmU32))
    let dF : Exp (.scalar .f32) := Exp.var dFName
    let dminF : Exp (.scalar .f32) := Exp.var dminFName

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
      -- Q4_K scale/min are 6-bit unsigned values (0..63).  toF32U emits
      -- cvt.rn.f32.u32 → I2FP.F32.U32 SASS; cvt.rn.f32.s32 (default toF32)
      -- causes ptxas to insert SGXT.U32 sign-extension before the cvt.
      -- Confirmed safe: scale/min always non-negative.
      (Exp.toF32U scaleU, Exp.toF32U minU)

    let (scA, mA) := extractScaleMin bq8Off
    let (scB, mB) := extractScaleMin (Exp.add bq8Off (Exp.litU32 1))

    -- Bind q4BaseIdx into a register so the 2× ld.global for v0/v1 share the
    -- same address-chain materialisation (otherwise codegen re-expands the
    -- 3-add chain `blockU32Base + (4 + bq8Off*4 + elemOff)` for both reads).
    let q4BaseIdxName ← ShaderM.var (.scalar .u32)
      (Exp.add blockU32Base
        (Exp.add (Exp.litU32 4) (Exp.add (Exp.mul bq8Off (Exp.litU32 4)) elemOff)))
    let q4BaseIdx : Exp (.scalar .u32) := Exp.var q4BaseIdxName
    let v0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" q4BaseIdx
    let v1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add q4BaseIdx (Exp.litU32 4))

    -- Q8_1 reads from global memory (no smem staging — maximizes occupancy).
    -- Hoist sub-block bases AND the per-load addresses (q8Sub0Base + 1 +
    -- elemOff) into explicit registers — codegen otherwise re-emits the
    -- 2-add chain for u0/u1/u2/u3 individually.
    let q8Sub0BaseName ← ShaderM.var (.scalar .u32)
      (Exp.add (Exp.mul blockIdx (Exp.litU32 (8 * 9))) (Exp.mul bq8Off (Exp.litU32 9)))
    let q8Sub0Base : Exp (.scalar .u32) := Exp.var q8Sub0BaseName
    let q8Sub1BaseName ← ShaderM.var (.scalar .u32) (Exp.add q8Sub0Base (Exp.litU32 9))
    let q8Sub1Base : Exp (.scalar .u32) := Exp.var q8Sub1BaseName
    -- Bind `1 + elemOff` and `5 + elemOff` once each (lane-invariant within
    -- iter); the (q8Sub_Base + offset) addition still varies per sub-block
    -- but the offset itself doesn't.
    let q8Off1Name ← ShaderM.var (.scalar .u32) (Exp.add (Exp.litU32 1) elemOff)
    let q8Off5Name ← ShaderM.var (.scalar .u32) (Exp.add (Exp.litU32 5) elemOff)
    let q8Off1 : Exp (.scalar .u32) := Exp.var q8Off1Name
    let q8Off5 : Exp (.scalar .u32) := Exp.var q8Off5Name
    let u0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" (Exp.add q8Sub0Base q8Off1)
    let u1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" (Exp.add q8Sub0Base q8Off5)
    let u2 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" (Exp.add q8Sub1Base q8Off1)
    let u3 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" (Exp.add q8Sub1Base q8Off5)
    let q8Hdr0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" q8Sub0Base
    let q8Hdr1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" q8Sub1Base
    -- Q8_1 header is now half2(d, sum) packed in a u32.  Extract `d` via
    -- the low f16 (sum lives in the high f16 — currently unused by hesper's
    -- matmul, but matches llama.cpp layout).
    -- Hoist the f16→f32 conversion: each d8A/d8B is referenced 2-4× below.
    -- Without a ShaderM.var binding, CSE still has to re-emit the
    -- `mov.b32 {lo,hi}, r; cvt.f32.f16` pair for every reference.  The
    -- explicit bind + `Exp.var` forces a single register reuse.
    let d8AName ← ShaderM.var (.scalar .f32) (Exp.vecX (Exp.unpack2x16float q8Hdr0))
    let d8BName ← ShaderM.var (.scalar .f32) (Exp.vecX (Exp.unpack2x16float q8Hdr1))
    let d8A : Exp (.scalar .f32) := Exp.var d8AName
    let d8B : Exp (.scalar .f32) := Exp.var d8BName

    let v0i0 := Exp.bitAnd v0 (Exp.litU32 0x0F0F0F0F)
    let v1i0 := Exp.bitAnd v1 (Exp.litU32 0x0F0F0F0F)
    let acc0 := Exp.dot4I8Packed v0i0 u0
    let dot1_0 := Exp.dot4I8Packed v1i0 u1
    let dot1_0Combined := Exp.add acc0 dot1_0
    let sumU_0 := Exp.add (Exp.dot4I8Packed (Exp.litU32 0x01010101) u0)
                          (Exp.dot4I8Packed (Exp.litU32 0x01010101) u1)

    let v0i1 := Exp.bitAnd (Exp.shiftRight v0 (Exp.litU32 4)) (Exp.litU32 0x0F0F0F0F)
    let v1i1 := Exp.bitAnd (Exp.shiftRight v1 (Exp.litU32 4)) (Exp.litU32 0x0F0F0F0F)
    let acc1 := Exp.dot4I8Packed v0i1 u2
    let dot1_1 := Exp.dot4I8Packed v1i1 u3
    let dot1_1Combined := Exp.add acc1 dot1_1
    let sumU_1 := Exp.add (Exp.dot4I8Packed (Exp.litU32 0x01010101) u2)
                          (Exp.dot4I8Packed (Exp.litU32 0x01010101) u3)

    -- FFMA-folded epilogue.  Original chain:
    --   sumfD_i = d8_i * (dot_i.toF32 * sc_i)
    --   sumfM_i = d8_i * (sumU_i.toF32 * m_i)
    --   blockSumfD = sumfD_0 + sumfD_1
    --   blockSumfM = sumfM_0 + sumfM_1
    --   acc' = fma(dF, blockSumfD, fma(-dminF, blockSumfM, acc))
    -- = 8 mul + 2 add + 2 fma per iter.
    --
    -- Pre-multiply scale into d8 (loop-invariant per pair) gives:
    --   blockSumfD = fma(dot_0F, d8AscA, dot_1F * d8BscB)
    --   blockSumfM = fma(sumU_0F, d8AmA,  sumU_1F * d8BmB)
    --   acc' = fma(dF, blockSumfD, fma(-dminF, blockSumfM, acc))
    -- = 4 mul + 4 fma per iter (-1 mul, +2 fma).  ptxas can FFMA the rest.
    let dot0F := Exp.toF32 dot1_0Combined
    let dot1F := Exp.toF32 dot1_1Combined
    let sumU0F := Exp.toF32 sumU_0
    let sumU1F := Exp.toF32 sumU_1
    let d8AscA := Exp.mul d8A scA
    let d8BscB := Exp.mul d8B scB
    let d8AmA  := Exp.mul d8A mA
    let d8BmB  := Exp.mul d8B mB
    let blockSumfD := Exp.fma dot0F d8AscA (Exp.mul dot1F d8BscB)
    let blockSumfM := Exp.fma sumU0F d8AmA  (Exp.mul sumU1F d8BmB)
    let accPrime := Exp.fma dF blockSumfD (Exp.fma (Exp.neg dminF) blockSumfM acc)
    ShaderM.assign "acc" accPrime
    ) (pure ())

  -- Intra-warp reduce: all 32 lanes contribute distinct products (matches
  -- llama.cpp's mul_mat_vec_q where each thread handles a unique
  -- (kbx, iqs) sub-block slot).  The full subgroup sum is the warp's
  -- partial — no `*0.5` correction.
  ShaderM.varNamed "warpTotal" (.scalar .f32) (Exp.subgroupAdd acc)
  let warpTotal : Exp (.scalar .f32) := Exp.var "warpTotal"

  -- Lane 0 of warps 1..3 writes to smem; warp 0 keeps its value in register.
  ShaderM.if_ (Exp.and (Exp.eq laneId (Exp.litU32 0))
                       (Exp.gt warpId (Exp.litU32 0))) (do
    ShaderM.writeWorkgroup (ty := .scalar .f32) "s_partial" warpId warpTotal
  ) (pure ())
  ShaderM.barrier

  -- Warp 0, lane 0: read back the 3 partials and produce final sum.
  ShaderM.if_ (Exp.and (Exp.and (Exp.eq warpId (Exp.litU32 0))
                                (Exp.eq laneId (Exp.litU32 0)))
                       inBounds) (do
    let p1 ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 4) "s_partial" (Exp.litU32 1)
    let p2 ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 4) "s_partial" (Exp.litU32 2)
    let p3 ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 4) "s_partial" (Exp.litU32 3)
    let total := Exp.add warpTotal (Exp.add p1 (Exp.add p2 p3))
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx total
  ) (pure ())

/-- Gate+up variant of `fusedQ4KMLinearDP4A4WarpKernel`: 1 row × 4 warps
    cooperative on K, but with TWO weight tensors (gate, up) processed in
    parallel from the same Q8_1 input.  Matches llama.cpp's ggml graph
    shape where ffn_gate and ffn_up are launched as adjacent
    `mul_mat_vec_q<Q4_K, ncols_dst=1, has_fusion=true>` dispatches.

    Same launch shape as the single-Linear 4-warp kernel:
    `(outDim, 1, 1)` workgroups × **128 threads** = 4 warps coop on K.

    vs `fusedQ4KMGateUpDP4A4RowKernel` (4 rows × 1 warp/row):
      - 4× more workgroups dispatched (outDim vs outDim/4) — better wave
        utilisation at outDim=10240 (Gemma 4 ffn).
      - Each warp now does K/4 work (=640 iterations at K=2560) vs the
        4-row variant's full K (=2560) per warp — 4× per-row issue
        parallelism.
      - 2× input register pressure (gate + up acc per thread) vs the
        4-row's smem-based input sharing.

    Trade-off: 4-row variant wins on smaller outDim (input reuse via smem);
    this 4-warp variant wins on larger outDim (wave count). For Gemma 4
    ffn outDim=10240 the 4-warp shape matches llama.cpp at ~16 µs/call
    (vs 4-row's ~68 µs/call combined gate+up). -/
def fusedQ4KMGateUpDP4A4WarpKernel (config : Config) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid            -- 0..127
  let warpId := Exp.shiftRight tid (Exp.litU32 5)  -- 0..3
  let laneId := Exp.bitAnd tid (Exp.litU32 31)     -- 0..31

  let blocksPerRow := config.inDim / 256
  let totalWeightU32 := config.outDim * blocksPerRow * 36
  let q8BlocksPerRow := config.inDim / 32
  let q8InputU32Size := q8BlocksPerRow * 9

  let _weightsGate ← ShaderM.declareReadOnlyBuffer "weights_gate" (.array (.scalar .u32) totalWeightU32)
  let _weightsUp   ← ShaderM.declareReadOnlyBuffer "weights_up"   (.array (.scalar .u32) totalWeightU32)
  let _input       ← ShaderM.declareReadOnlyBuffer "input_q8"     (.array (.scalar .u32) q8InputU32Size)
  let _output      ← ShaderM.declareOutputBuffer "output"         (.array (.scalar .f32) config.outDim)

  -- Cross-warp partial sums: 4 floats for gate + 4 for up.
  ShaderM.sharedNamed "s_partial" (.array (.scalar .f32) 8)

  let inBounds := Exp.lt outIdx (Exp.litU32 config.outDim)

  let kbxStartName ← ShaderM.var (.scalar .u32) (Exp.shiftRight tid (Exp.litU32 4))
  let laneLowName  ← ShaderM.var (.scalar .u32) (Exp.bitAnd tid (Exp.litU32 15))
  let kbxStart : Exp (.scalar .u32) := Exp.var kbxStartName
  let laneLow  : Exp (.scalar .u32) := Exp.var laneLowName
  let pairIdxInRowName ← ShaderM.var (.scalar .u32) (Exp.shiftRight laneLow (Exp.litU32 2))
  let elemOffName      ← ShaderM.var (.scalar .u32) (Exp.bitAnd laneLow (Exp.litU32 3))
  let pairIdxInRow : Exp (.scalar .u32) := Exp.var pairIdxInRowName
  let elemOff      : Exp (.scalar .u32) := Exp.var elemOffName
  let bq8OffName ← ShaderM.var (.scalar .u32) (Exp.shiftLeft pairIdxInRow (Exp.litU32 1))
  let bq8Off : Exp (.scalar .u32) := Exp.var bq8OffName

  let rowBaseU32Name ← ShaderM.var (.scalar .u32)
    (Exp.mul outIdx (Exp.litU32 (blocksPerRow * 36)))
  let rowBaseU32 : Exp (.scalar .u32) := Exp.var rowBaseU32Name

  ShaderM.varNamed "accG" (.scalar .f32) (Exp.litF32 0.0)
  ShaderM.varNamed "accU" (.scalar .f32) (Exp.litF32 0.0)
  let accG : Exp (.scalar .f32) := Exp.var "accG"
  let accU : Exp (.scalar .f32) := Exp.var "accU"

  let maxIter := (blocksPerRow + 7) / 8
  ShaderM.loop (Exp.litU32 0) (Exp.litU32 maxIter) (Exp.litU32 1) fun iter => do
    let blockIdxName ← ShaderM.var (.scalar .u32)
      (Exp.add kbxStart (Exp.mul iter (Exp.litU32 8)))
    let blockIdx : Exp (.scalar .u32) := Exp.var blockIdxName
    let blockInRange := Exp.lt blockIdx (Exp.litU32 blocksPerRow)
    ShaderM.if_ blockInRange (do
    let blockU32BaseName ← ShaderM.var (.scalar .u32)
      (Exp.add rowBaseU32 (Exp.mul blockIdx (Exp.litU32 36)))
    let blockU32Base : Exp (.scalar .u32) := Exp.var blockU32BaseName

    -- Q8_1 input: load once, reuse for gate AND up.  This is the whole
    -- point of the fused kernel — halves global Q8_1 bandwidth vs two
    -- separate Linear dispatches.
    let q8Sub0BaseName ← ShaderM.var (.scalar .u32)
      (Exp.add (Exp.mul blockIdx (Exp.litU32 (8 * 9))) (Exp.mul bq8Off (Exp.litU32 9)))
    let q8Sub0Base : Exp (.scalar .u32) := Exp.var q8Sub0BaseName
    let q8Sub1BaseName ← ShaderM.var (.scalar .u32) (Exp.add q8Sub0Base (Exp.litU32 9))
    let q8Sub1Base : Exp (.scalar .u32) := Exp.var q8Sub1BaseName
    let u0Name ← ShaderM.var (.scalar .u32) (← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" (Exp.add q8Sub0Base (Exp.add (Exp.litU32 1) elemOff)))
    let u1Name ← ShaderM.var (.scalar .u32) (← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" (Exp.add q8Sub0Base (Exp.add (Exp.litU32 5) elemOff)))
    let u2Name ← ShaderM.var (.scalar .u32) (← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" (Exp.add q8Sub1Base (Exp.add (Exp.litU32 1) elemOff)))
    let u3Name ← ShaderM.var (.scalar .u32) (← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" (Exp.add q8Sub1Base (Exp.add (Exp.litU32 5) elemOff)))
    let u0 : Exp (.scalar .u32) := Exp.var u0Name
    let u1 : Exp (.scalar .u32) := Exp.var u1Name
    let u2 : Exp (.scalar .u32) := Exp.var u2Name
    let u3 : Exp (.scalar .u32) := Exp.var u3Name
    let q8Hdr0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" q8Sub0Base
    let q8Hdr1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" q8Sub1Base
    let d8AName ← ShaderM.var (.scalar .f32) (Exp.vecX (Exp.unpack2x16float q8Hdr0))
    let d8BName ← ShaderM.var (.scalar .f32) (Exp.vecX (Exp.unpack2x16float q8Hdr1))
    let d8A : Exp (.scalar .f32) := Exp.var d8AName
    let d8B : Exp (.scalar .f32) := Exp.var d8BName

    -- Pre-compute Q8_1 sums (independent of weight choice — used by both
    -- gate and up via the dminF * sumU correction term).
    let sumU_0Name ← ShaderM.var (.scalar .i32)
      (Exp.add (Exp.dot4I8Packed (Exp.litU32 0x01010101) u0)
               (Exp.dot4I8Packed (Exp.litU32 0x01010101) u1))
    let sumU_1Name ← ShaderM.var (.scalar .i32)
      (Exp.add (Exp.dot4I8Packed (Exp.litU32 0x01010101) u2)
               (Exp.dot4I8Packed (Exp.litU32 0x01010101) u3))
    let sumU_0 : Exp (.scalar .i32) := Exp.var sumU_0Name
    let sumU_1 : Exp (.scalar .i32) := Exp.var sumU_1Name
    let sumU_0FName ← ShaderM.var (.scalar .f32) (Exp.toF32 sumU_0)
    let sumU_1FName ← ShaderM.var (.scalar .f32) (Exp.toF32 sumU_1)
    let sumU_0F : Exp (.scalar .f32) := Exp.var sumU_0FName
    let sumU_1F : Exp (.scalar .f32) := Exp.var sumU_1FName

    -- Process one weight tensor (gate or up).  Reads weights from `which`,
    -- accumulates dp4a result into `acc`, returns new acc.
    let processWeight (which : String) (acc : Exp (.scalar .f32))
        : ShaderM (Exp (.scalar .f32)) := do
      let (dmU32, sc0, sc1, sc2) ← ShaderM.readBufferU32x4 which blockU32Base
      let dF := Exp.vecX (Exp.unpack2x16float dmU32)
      let dminF := Exp.vecY (Exp.unpack2x16float dmU32)

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
        (Exp.toF32U scaleU, Exp.toF32U minU)

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
      let sumfD_0 := Exp.mul d8A (Exp.mul (Exp.toF32 dot1_0Combined) scA)
      let sumfM_0 := Exp.mul d8A (Exp.mul sumU_0F mA)

      let v0i1 := Exp.bitAnd (Exp.shiftRight v0 (Exp.litU32 4)) (Exp.litU32 0x0F0F0F0F)
      let v1i1 := Exp.bitAnd (Exp.shiftRight v1 (Exp.litU32 4)) (Exp.litU32 0x0F0F0F0F)
      let acc1 := Exp.dot4I8Packed v0i1 u2
      let dot1_1 := Exp.dot4I8Packed v1i1 u3
      let dot1_1Combined := Exp.add acc1 dot1_1
      let sumfD_1 := Exp.mul d8B (Exp.mul (Exp.toF32 dot1_1Combined) scB)
      let sumfM_1 := Exp.mul d8B (Exp.mul sumU_1F mB)

      let blockSumfD := Exp.add sumfD_0 sumfD_1
      let blockSumfM := Exp.add sumfM_0 sumfM_1
      pure (Exp.fma dF blockSumfD (Exp.fma (Exp.neg dminF) blockSumfM acc))

    let newAccG ← processWeight "weights_gate" accG
    ShaderM.assign "accG" newAccG
    let newAccU ← processWeight "weights_up" accU
    ShaderM.assign "accU" newAccU
    ) (pure ())

  -- Intra-warp reduce for both gate and up.
  ShaderM.varNamed "warpTotalG" (.scalar .f32) (Exp.subgroupAdd accG)
  ShaderM.varNamed "warpTotalU" (.scalar .f32) (Exp.subgroupAdd accU)
  let warpTotalG : Exp (.scalar .f32) := Exp.var "warpTotalG"
  let warpTotalU : Exp (.scalar .f32) := Exp.var "warpTotalU"

  -- Lane 0 of warps 1..3 writes both partials to smem.  Slot layout:
  --   s_partial[0..3] = gate partials per warp
  --   s_partial[4..7] = up partials per warp
  ShaderM.if_ (Exp.and (Exp.eq laneId (Exp.litU32 0))
                       (Exp.gt warpId (Exp.litU32 0))) (do
    ShaderM.writeWorkgroup (ty := .scalar .f32) "s_partial" warpId warpTotalG
    ShaderM.writeWorkgroup (ty := .scalar .f32) "s_partial"
      (Exp.add warpId (Exp.litU32 4)) warpTotalU
  ) (pure ())
  ShaderM.barrier

  -- Warp 0, lane 0: read partials, compute gelu(gate) * up, write output.
  ShaderM.if_ (Exp.and (Exp.and (Exp.eq warpId (Exp.litU32 0))
                                (Exp.eq laneId (Exp.litU32 0)))
                       inBounds) (do
    let pG1 ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 8) "s_partial" (Exp.litU32 1)
    let pG2 ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 8) "s_partial" (Exp.litU32 2)
    let pG3 ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 8) "s_partial" (Exp.litU32 3)
    let pU1 ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 8) "s_partial" (Exp.litU32 5)
    let pU2 ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 8) "s_partial" (Exp.litU32 6)
    let pU3 ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 8) "s_partial" (Exp.litU32 7)
    let totalG := Exp.add warpTotalG (Exp.add pG1 (Exp.add pG2 pG3))
    let totalU := Exp.add warpTotalU (Exp.add pU1 (Exp.add pU2 pU3))
    -- GELU(gate) * up — same epilogue as fusedQ4KMGateUpDP4A4RowKernel.
    let sqrt2OverPi := Exp.litF32 0.7978845608028654
    let z := totalG
    let z3 := Exp.mul (Exp.mul z z) z
    let inner := Exp.mul sqrt2OverPi (Exp.add z (Exp.mul (Exp.litF32 0.044715) z3))
    let gelu := Exp.mul (Exp.mul (Exp.litF32 0.5) z) (Exp.add (Exp.litF32 1.0) (Exp.tanh inner))
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx (Exp.mul gelu totalU)
  ) (pure ())

/-- Inline-quantize variant of `fusedQ4KMLinearDP4A4WarpKernel`.

    Takes an f32 `input` (not pre-quantized Q8_1).  Each workgroup first
    cooperates to quantize its own copy of the input into shared memory
    (4 warps × 32 threads tile-quantize `blocksPerRow` Q8_1 blocks), then
    runs the same dp4a matmul inner loop, reading from smem instead of
    the separate `input_q8` global buffer.

    Benefit: eliminates the standalone `quantizeQ8_1Kernel` dispatch
    that previously fed this kernel — one dispatch per matmul instead of
    two.  Matches llama.cpp's `mul_mat_vec_q<Q4_K, 1, 4>` pattern.

    Cost: duplicated quantize work across workgroups (every output row
    re-quantizes the same input).  Worth it for decode matmul (N=1 row
    amortizes quantize over blocksPerRow·outDim reads of each Q8_1
    block).  For prefill/batch use the original 2-dispatch path. -/
def fusedQ4KMLinearDP4A4WarpInlineQuantKernel (config : Config) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid            -- 0..127
  let warpId := Exp.shiftRight tid (Exp.litU32 5)  -- 0..3
  let laneId := Exp.bitAnd tid (Exp.litU32 31)     -- 0..31

  let blocksPerRow := config.inDim / 256
  let totalWeightU32 := config.outDim * blocksPerRow * 36
  let q8BlocksPerRow := config.inDim / 32
  let q8SmemSizeU32 := q8BlocksPerRow * 9

  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareReadOnlyBuffer "input" (.array (.scalar .f32) config.inDim)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) config.outDim)

  -- Quantize scratch: blocksPerRow * 8 sub-blocks * 9 u32 = q8SmemSizeU32.
  ShaderM.sharedNamed "s_q8" (.array (.scalar .u32) q8SmemSizeU32)
  ShaderM.sharedNamed "s_partial" (.array (.scalar .f32) 4)

  -- =========================
  --   Phase 1: Quantize input
  -- =========================
  -- Each of 128 threads handles one f32 element of a 32-element block.
  -- Warp i (i ∈ 0..3) owns block i initially; outer loop iterates by 4
  -- until all `q8BlocksPerRow` blocks are quantized.
  -- (For inDim=2560, q8BlocksPerRow=80, outer iter count = 20.)
  let qLoopIters := (q8BlocksPerRow + 3) / 4
  ShaderM.loop (Exp.litU32 0) (Exp.litU32 qLoopIters) (Exp.litU32 1) fun iter => do
    let blockIdx := Exp.add warpId (Exp.mul iter (Exp.litU32 4))
    let blockInRange := Exp.lt blockIdx (Exp.litU32 q8BlocksPerRow)
    let safeBlockIdx := Exp.select blockInRange blockIdx (Exp.litU32 0)
    let elemIdx := Exp.add (Exp.mul safeBlockIdx (Exp.litU32 32)) laneId
    let x ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.inDim) "input" elemIdx
    let absX := Exp.select (Exp.lt x (Exp.litF32 0.0)) (Exp.sub (Exp.litF32 0.0) x) x
    ShaderM.varNamed "amax" (.scalar .f32) (Exp.subgroupMax absX)
    let amax : Exp (.scalar .f32) := Exp.var "amax"
    ShaderM.varNamed "sumX" (.scalar .f32) (Exp.subgroupAdd x)
    let sumX : Exp (.scalar .f32) := Exp.var "sumX"
    ShaderM.varNamed "d_q8" (.scalar .f32) (Exp.div amax (Exp.litF32 127.0))
    let d : Exp (.scalar .f32) := Exp.var "d_q8"
    ShaderM.varNamed "qF32" (.scalar .f32)
      (Exp.select (Exp.eq d (Exp.litF32 0.0)) (Exp.litF32 0.0) (Exp.div x d))
    let qF32 : Exp (.scalar .f32) := Exp.var "qF32"
    ShaderM.varNamed "qByte" (.scalar .u32)
      (Exp.bitAnd (Exp.roundToI32 qF32) (Exp.litU32 0xFF))
    let qByte : Exp (.scalar .u32) := Exp.var "qByte"
    let hdrOff := Exp.mul safeBlockIdx (Exp.litU32 9)
    -- Header half2(d, sumX) at slot 0.
    ShaderM.if_ (Exp.and blockInRange (Exp.eq laneId (Exp.litU32 0))) (do
      let packed : Exp (.scalar .u32) := Exp.pack2x16float (Exp.vec2 d sumX)
      ShaderM.writeWorkgroup (ty := .scalar .u32) "s_q8" hdrOff packed
    ) (pure ())
    -- Lanes 0,4,8,..,28 pack 4 consecutive quants into one u32.  Use
    -- shuffle to grab the other 3 lanes' qByte; avoids a second smem
    -- round-trip (we already have all 32 qBytes in registers).
    let isQuarter := Exp.eq (Exp.bitAnd laneId (Exp.litU32 3)) (Exp.litU32 0)
    ShaderM.if_ (Exp.and blockInRange isQuarter) (do
      let b0 := qByte
      let b1 := Exp.subgroupShuffle qByte (Exp.add laneId (Exp.litU32 1))
      let b2 := Exp.subgroupShuffle qByte (Exp.add laneId (Exp.litU32 2))
      let b3 := Exp.subgroupShuffle qByte (Exp.add laneId (Exp.litU32 3))
      let packed := Exp.bitOr (Exp.bitOr b0 (Exp.shiftLeft b1 (Exp.litU32 8)))
                              (Exp.bitOr (Exp.shiftLeft b2 (Exp.litU32 16))
                                         (Exp.shiftLeft b3 (Exp.litU32 24)))
      let quartOff := Exp.add hdrOff
        (Exp.add (Exp.litU32 1) (Exp.shiftRight laneId (Exp.litU32 2)))
      ShaderM.writeWorkgroup (ty := .scalar .u32) "s_q8" quartOff packed
    ) (pure ())
  ShaderM.barrier

  -- =========================
  --   Phase 2: dp4a matmul
  -- =========================
  -- Identical to `fusedQ4KMLinearDP4A4WarpKernel` but reads Q8_1 data
  -- from `s_q8` instead of a global `input_q8` binding.
  let inBounds := Exp.lt outIdx (Exp.litU32 config.outDim)
  let kbxStart := Exp.shiftRight tid (Exp.litU32 4)  -- 0..7
  let laneLow := Exp.bitAnd tid (Exp.litU32 15)
  let pairIdxInRow := Exp.shiftRight laneLow (Exp.litU32 2)  -- 0..3
  let elemOff := Exp.bitAnd laneLow (Exp.litU32 3)           -- 0..3
  let bq8Off := Exp.shiftLeft pairIdxInRow (Exp.litU32 1)    -- 0,2,4,6
  let rowBaseU32 := Exp.mul outIdx (Exp.litU32 (blocksPerRow * 36))

  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"

  let maxIter := (blocksPerRow + 7) / 8
  ShaderM.loop (Exp.litU32 0) (Exp.litU32 maxIter) (Exp.litU32 1) fun iter => do
    let blockIdx := Exp.add kbxStart (Exp.mul iter (Exp.litU32 8))
    let blockInRange := Exp.lt blockIdx (Exp.litU32 blocksPerRow)
    let safeBlockIdx := Exp.select blockInRange blockIdx (Exp.litU32 0)
    let blockU32Base := Exp.add rowBaseU32 (Exp.mul safeBlockIdx (Exp.litU32 36))

    let dmU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" blockU32Base
    let dF := Exp.vecX (Exp.unpack2x16float dmU32)
    let dminF := Exp.vecY (Exp.unpack2x16float dmU32)
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
      (Exp.toF32U scaleU, Exp.toF32U minU)
    let (scA, mA) := extractScaleMin bq8Off
    let (scB, mB) := extractScaleMin (Exp.add bq8Off (Exp.litU32 1))
    let q4BaseIdx := Exp.add blockU32Base
      (Exp.add (Exp.litU32 4) (Exp.add (Exp.mul bq8Off (Exp.litU32 4)) elemOff))
    let v0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" q4BaseIdx
    let v1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add q4BaseIdx (Exp.litU32 4))
    -- Q8_1 reads from smem instead of global.
    let q8Sub0Base := Exp.add (Exp.mul safeBlockIdx (Exp.litU32 (8 * 9))) (Exp.mul bq8Off (Exp.litU32 9))
    let q8Sub1Base := Exp.add q8Sub0Base (Exp.litU32 9)
    let u0 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := q8SmemSizeU32) "s_q8" (Exp.add q8Sub0Base (Exp.add (Exp.litU32 1) elemOff))
    let u1 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := q8SmemSizeU32) "s_q8" (Exp.add q8Sub0Base (Exp.add (Exp.litU32 5) elemOff))
    let u2 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := q8SmemSizeU32) "s_q8" (Exp.add q8Sub1Base (Exp.add (Exp.litU32 1) elemOff))
    let u3 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := q8SmemSizeU32) "s_q8" (Exp.add q8Sub1Base (Exp.add (Exp.litU32 5) elemOff))
    let q8Hdr0 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := q8SmemSizeU32) "s_q8" q8Sub0Base
    let q8Hdr1 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := q8SmemSizeU32) "s_q8" q8Sub1Base
    let d8AName ← ShaderM.var (.scalar .f32) (Exp.vecX (Exp.unpack2x16float q8Hdr0))
    let d8BName ← ShaderM.var (.scalar .f32) (Exp.vecX (Exp.unpack2x16float q8Hdr1))
    let d8A : Exp (.scalar .f32) := Exp.var d8AName
    let d8B : Exp (.scalar .f32) := Exp.var d8BName
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
    let gatedContrib := Exp.select blockInRange blockContrib (Exp.litF32 0.0)
    ShaderM.assign "acc" (Exp.add acc gatedContrib)

  ShaderM.varNamed "warpTotal" (.scalar .f32) (Exp.subgroupAdd acc)
  let warpTotal : Exp (.scalar .f32) := Exp.var "warpTotal"
  ShaderM.if_ (Exp.and (Exp.eq laneId (Exp.litU32 0))
                       (Exp.gt warpId (Exp.litU32 0))) (do
    ShaderM.writeWorkgroup (ty := .scalar .f32) "s_partial" warpId warpTotal
  ) (pure ())
  ShaderM.barrier
  ShaderM.if_ (Exp.and (Exp.and (Exp.eq warpId (Exp.litU32 0))
                                (Exp.eq laneId (Exp.litU32 0)))
                       inBounds) (do
    let p1 ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 4) "s_partial" (Exp.litU32 1)
    let p2 ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 4) "s_partial" (Exp.litU32 2)
    let p3 ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 4) "s_partial" (Exp.litU32 3)
    let total := Exp.add warpTotal (Exp.add p1 (Exp.add p2 p3))
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx total
  ) (pure ())

/-- **Phase 2b (2026-04-20)** — Multi-layer variant of
    `fusedQ4KMLinearDP4A4WarpKernel`.  Identical compute per workgroup, but
    the 3 bindings (`weights`, `input_q8`, `output`) are `bufferArray`s of
    length `numLayers`.  Grid is `(outDim, numLayers, 1)` → **one dispatch
    replaces `numLayers` dispatches**.

    Every workgroup computes `output[layerIdx][outRow]`, selecting its layer
    via `wid.y`.  Pointer-table indirection is done by the PTX lowering of
    `readBufferArray` / `writeBufferArray`.

    Caller owns the N per-layer buffers; host-side pointer table is
    auto-managed by `executeWithConfigCachedArrays`. -/
def fusedQ4KMLinearDP4A4WarpMultiLayerKernel
    (config : Config) (numLayers : Nat) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let layerIdx := Exp.vec3Y wid
  let tid := Exp.vec3X lid
  let warpId := Exp.shiftRight tid (Exp.litU32 5)
  let laneId := Exp.bitAnd tid (Exp.litU32 31)

  let blocksPerRow := config.inDim / 256
  let q8BlocksPerRow := config.inDim / 32
  let _q8InputU32Size := q8BlocksPerRow * 9

  -- bufferArray bindings: one buffer per layer, indexed at runtime via wid.y.
  let _weights ← ShaderM.declareInputBufferArray "weights" (.scalar .u32) numLayers
  let _input ← ShaderM.declareInputBufferArray "input_q8" (.scalar .u32) numLayers
  let _output ← ShaderM.declareInputBufferArray "output" (.scalar .f32) numLayers

  ShaderM.sharedNamed "s_partial" (.array (.scalar .f32) 4)

  let inBounds := Exp.lt outIdx (Exp.litU32 config.outDim)

  let kbxStart := Exp.shiftRight tid (Exp.litU32 4)
  let laneLow := Exp.bitAnd tid (Exp.litU32 15)
  let pairIdxInRow := Exp.shiftRight laneLow (Exp.litU32 2)
  let elemOff := Exp.bitAnd laneLow (Exp.litU32 3)
  let bq8Off := Exp.shiftLeft pairIdxInRow (Exp.litU32 1)

  let rowBaseU32 := Exp.mul outIdx (Exp.litU32 (blocksPerRow * 36))

  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"

  let maxIter := (blocksPerRow + 7) / 8
  ShaderM.loop (Exp.litU32 0) (Exp.litU32 maxIter) (Exp.litU32 1) fun iter => do
    let blockIdx := Exp.add kbxStart (Exp.mul iter (Exp.litU32 8))
    let blockInRange := Exp.lt blockIdx (Exp.litU32 blocksPerRow)
    let safeBlockIdx := Exp.select blockInRange blockIdx (Exp.litU32 0)
    let blockU32Base := Exp.add rowBaseU32 (Exp.mul safeBlockIdx (Exp.litU32 36))

    let dmU32 ← ShaderM.readBufferArray (elemTy := .scalar .u32) (n := numLayers)
                  "weights" layerIdx blockU32Base
    let dF := Exp.vecX (Exp.unpack2x16float dmU32)
    let dminF := Exp.vecY (Exp.unpack2x16float dmU32)

    let sc0 ← ShaderM.readBufferArray (elemTy := .scalar .u32) (n := numLayers)
                  "weights" layerIdx (Exp.add blockU32Base (Exp.litU32 1))
    let sc1 ← ShaderM.readBufferArray (elemTy := .scalar .u32) (n := numLayers)
                  "weights" layerIdx (Exp.add blockU32Base (Exp.litU32 2))
    let sc2 ← ShaderM.readBufferArray (elemTy := .scalar .u32) (n := numLayers)
                  "weights" layerIdx (Exp.add blockU32Base (Exp.litU32 3))

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
      (Exp.toF32U scaleU, Exp.toF32U minU)

    let (scA, mA) := extractScaleMin bq8Off
    let (scB, mB) := extractScaleMin (Exp.add bq8Off (Exp.litU32 1))

    let q4BaseIdx := Exp.add blockU32Base
      (Exp.add (Exp.litU32 4) (Exp.add (Exp.mul bq8Off (Exp.litU32 4)) elemOff))
    let v0 ← ShaderM.readBufferArray (elemTy := .scalar .u32) (n := numLayers)
                "weights" layerIdx q4BaseIdx
    let v1 ← ShaderM.readBufferArray (elemTy := .scalar .u32) (n := numLayers)
                "weights" layerIdx (Exp.add q4BaseIdx (Exp.litU32 4))

    let q8Sub0Base := Exp.add (Exp.mul safeBlockIdx (Exp.litU32 (8 * 9))) (Exp.mul bq8Off (Exp.litU32 9))
    let q8Sub1Base := Exp.add q8Sub0Base (Exp.litU32 9)
    let u0 ← ShaderM.readBufferArray (elemTy := .scalar .u32) (n := numLayers)
                "input_q8" layerIdx (Exp.add q8Sub0Base (Exp.add (Exp.litU32 1) elemOff))
    let u1 ← ShaderM.readBufferArray (elemTy := .scalar .u32) (n := numLayers)
                "input_q8" layerIdx (Exp.add q8Sub0Base (Exp.add (Exp.litU32 5) elemOff))
    let u2 ← ShaderM.readBufferArray (elemTy := .scalar .u32) (n := numLayers)
                "input_q8" layerIdx (Exp.add q8Sub1Base (Exp.add (Exp.litU32 1) elemOff))
    let u3 ← ShaderM.readBufferArray (elemTy := .scalar .u32) (n := numLayers)
                "input_q8" layerIdx (Exp.add q8Sub1Base (Exp.add (Exp.litU32 5) elemOff))
    let q8Hdr0 ← ShaderM.readBufferArray (elemTy := .scalar .u32) (n := numLayers)
                "input_q8" layerIdx q8Sub0Base
    let q8Hdr1 ← ShaderM.readBufferArray (elemTy := .scalar .u32) (n := numLayers)
                "input_q8" layerIdx q8Sub1Base
    let d8AName ← ShaderM.var (.scalar .f32) (Exp.vecX (Exp.unpack2x16float q8Hdr0))
    let d8BName ← ShaderM.var (.scalar .f32) (Exp.vecX (Exp.unpack2x16float q8Hdr1))
    let d8A : Exp (.scalar .f32) := Exp.var d8AName
    let d8B : Exp (.scalar .f32) := Exp.var d8BName

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
    let gatedContrib := Exp.select blockInRange blockContrib (Exp.litF32 0.0)
    ShaderM.assign "acc" (Exp.add acc gatedContrib)

  ShaderM.varNamed "warpTotal" (.scalar .f32) (Exp.subgroupAdd acc)
  let warpTotal : Exp (.scalar .f32) := Exp.var "warpTotal"

  ShaderM.if_ (Exp.and (Exp.eq laneId (Exp.litU32 0))
                       (Exp.gt warpId (Exp.litU32 0))) (do
    ShaderM.writeWorkgroup (ty := .scalar .f32) "s_partial" warpId warpTotal
  ) (pure ())
  ShaderM.barrier

  ShaderM.if_ (Exp.and (Exp.and (Exp.eq warpId (Exp.litU32 0))
                                (Exp.eq laneId (Exp.litU32 0)))
                       inBounds) (do
    let p1 ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 4) "s_partial" (Exp.litU32 1)
    let p2 ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 4) "s_partial" (Exp.litU32 2)
    let p3 ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 4) "s_partial" (Exp.litU32 3)
    let total := Exp.add warpTotal (Exp.add p1 (Exp.add p2 p3))
    ShaderM.writeBufferArray (ty := .scalar .f32) "output" layerIdx outIdx total
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
  -- Native u8 load — one `ld.global.nc.u8` on CUDA (vs 1 u32 load + shift +
  -- mask previously).  See docs/llama-fusion-analysis/41.md for the PTX
  -- diff that motivated this primitive.
  let readByte (blockBase : Exp (.scalar .u32)) (offset : Exp (.scalar .u32)) : ShaderM (Exp (.scalar .u32)) := do
    let byteIdx := Exp.add blockBase offset
    ShaderM.readBufferByte (n := totalWeightU32) "weights" byteIdx

  -- Native u16 load — one `ld.global.nc.u16` on CUDA.  Used to read the
  -- 2-byte fp16 block scale `d` in a single instruction.
  let readU16 (blockBase : Exp (.scalar .u32)) (offset : Exp (.scalar .u32)) : ShaderM (Exp (.scalar .u32)) := do
    let byteIdx := Exp.add blockBase offset
    ShaderM.readBufferU16 (n := totalWeightU32) "weights" byteIdx

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
    -- Bind byteIdx / u32Idx / byteOff so they materialise once in PTX.
    -- Without these, each downstream use re-emits the `blockBase+offset`
    -- chain (6× add.u32 + 6× shl.b32 per read4Bytes call in earlier PTX).
    let byteIdxName ← ShaderM.var (.scalar .u32) (Exp.add blockBase offset)
    let byteIdx : Exp (.scalar .u32) := Exp.var byteIdxName
    let u32IdxName ← ShaderM.var (.scalar .u32) (Exp.shiftRight byteIdx (Exp.litU32 2))
    let u32Idx : Exp (.scalar .u32) := Exp.var u32IdxName
    let byteOffName ← ShaderM.var (.scalar .u32) (Exp.bitAnd byteIdx (Exp.litU32 3))
    let byteOff : Exp (.scalar .u32) := Exp.var byteOffName
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
    -- Single ld.global.nc.u16 for the fp16 block scale, converted to f32
    -- via `cvt.f32.f16` (hardware instruction, 1 op) instead of the
    -- `fp16ToF32` arithmetic soft-impl which PTX-expands to 15+ ops
    -- (selp/div/ex2/sub for sign/mantissa/exponent).  We reinterpret the
    -- u16 as the low half of a packed half2 and extract the low f16 — the
    -- codegen path for `vecX (unpack2x16float _)` lowers to exactly
    -- `mov.b32 {lo,hi}, r; cvt.f32.f16 f, lo`.
    let dBitsName ← ShaderM.var (.scalar .u32) (← readU16 blockByteBase (Exp.litU32 208))
    let dBits : Exp (.scalar .u32) := Exp.var dBitsName
    let d := Exp.vecX (Exp.unpack2x16float dBits)

    -- Read vl (4 bytes of ql at byte offset 4*iqs).  Bind so the 2 u32
    -- reads inside read4Bytes happen once — vl is referenced twice below
    -- (vil_0 via `vl & 0x0F…`, vil_1 via `(vl >> 4) & 0x0F…`).
    let vlOffset := Exp.mul iqs (Exp.litU32 4)
    let vlName ← ShaderM.var (.scalar .u32) (← read4Bytes blockByteBase vlOffset)
    let vl : Exp (.scalar .u32) := Exp.var vlName
    -- Read vh_raw (4 bytes of qh at byte offset 128 + 4*vhIdx), shift right by vh_shift.
    -- Bind `vh` (not just vhRaw) since `vh` is referenced in vih_0 and vih_1.
    let vhOffset := Exp.add (Exp.litU32 128) (Exp.mul vhIdx (Exp.litU32 4))
    let vhRaw ← read4Bytes blockByteBase vhOffset
    let vhName ← ShaderM.var (.scalar .u32) (Exp.shiftRight vhRaw vhShift)
    let vh : Exp (.scalar .u32) := Exp.var vhName

    -- Read 2 scales: scales[scale_offset], scales[scale_offset + 4]
    -- (scales start at byte 192, each is 1 signed byte).
    -- Keep scales as i32 (sign-extended from i8) throughout the inner loop;
    -- defer the f32 cast until after the int×int multiply with dot_0.
    -- This matches llama.cpp's `d8[i] * (dp4a × sc)` pattern — 1 FFMA per
    -- iter instead of 2 (the old `d8 * (f32(dot) * f32(sc))` required two
    -- f32 conversions and two FFMAs).  SASS confirms: llama has 3 FFMA
    -- vs hesper's 24 in the old version.
    -- Bind each scale byte so it's loaded exactly once — signExtI8 below
    -- references `b` three times, and without the bind the readByte call
    -- inlines to 3× ld.global.u8 per scale (6 total per iter) even though
    -- the HW would cache them.  Makes PTX match llama.cpp's 2 ld.u8/iter.
    let sc0ByteName ← ShaderM.var (.scalar .u32)
      (← readByte blockByteBase (Exp.add (Exp.litU32 192) scaleOff))
    let sc1ByteName ← ShaderM.var (.scalar .u32)
      (← readByte blockByteBase (Exp.add (Exp.litU32 192) (Exp.add scaleOff (Exp.litU32 4))))
    let sc0Byte : Exp (.scalar .u32) := Exp.var sc0ByteName
    let sc1Byte : Exp (.scalar .u32) := Exp.var sc1ByteName
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
    -- Q8_1 header is now half2(d, sum) packed in a u32.  Extract `d` via
    -- the low f16 (sum lives in the high f16 — currently unused by hesper's
    -- matmul, but matches llama.cpp layout).
    -- Hoist the f16→f32 conversion: each d8A/d8B is referenced 2-4× below.
    -- Without a ShaderM.var binding, CSE still has to re-emit the
    -- `mov.b32 {lo,hi}, r; cvt.f32.f16` pair for every reference.  The
    -- explicit bind + `Exp.var` forces a single register reuse.
    let d8AName ← ShaderM.var (.scalar .f32) (Exp.vecX (Exp.unpack2x16float q8Hdr0))
    let d8BName ← ShaderM.var (.scalar .f32) (Exp.vecX (Exp.unpack2x16float q8Hdr1))
    let d8A : Exp (.scalar .f32) := Exp.var d8AName
    let d8B : Exp (.scalar .f32) := Exp.var d8BName

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

    -- acc = d * (sumf_0 + sumf_1) + acc → one fma.rn.f32 instead of mul + add.
    ShaderM.assign "acc" (Exp.fma d (Exp.add sumf_0 sumf_1) acc)

  -- All 32 lanes contribute unique partials (iqs=tid). Standard subgroupAdd.
  ShaderM.varNamed "total" (.scalar .f32) (Exp.subgroupAdd acc)
  let total : Exp (.scalar .f32) := Exp.var "total"

  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx total
  ) (pure ())

/-- Batched Q6_K × Q8_1 matmul: `[outDim, inDim] × [inDim, seqLen] → [outDim, seqLen]`.

    Mirror of `q4kMatmulBatchKernel` for Q6_K. Grid: `(outDim, seqLen, 1) × 32`.
    Each WG (32 threads / 1 warp) computes one output element using the same
    DP4A inner loop as `fusedQ6KLinearDP4AKernel`, with the Q8_1 base pointer
    shifted by `colIdx * (q8BlocksPerRow * 9)` and the output written to
    `output[colIdx * outDim + outIdx]`.

    Replaces the per-column extract+matmul+insert loop in `forwardBatchDP4A`
    that was firing the Q6_K ffn_down kernel `seqLen × n_layers` times during
    prefill (1239 calls/prefill at seqLen=59, ~30 ms total).

    seqLen=1 is supported for unification with the decode path; a real prefill
    has seqLen ≥ 2. -/
def q6kMatmulBatchKernel (inDim outDim : Nat) (seqLen : Nat)
    (rowOffset : Nat := 0) (weightRows : Nat := 0) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid     -- output row (chunk-local)
  let colIdx := Exp.vec3Y wid     -- sequence position
  let tid := Exp.vec3X lid

  let blocksPerRow := inDim / 256
  let blockSizeBytes : Nat := 210
  let wRows := if weightRows == 0 then outDim else weightRows
  let totalWeightBytes := wRows * blocksPerRow * blockSizeBytes
  let totalWeightU32 := (totalWeightBytes + 3) / 4
  let q8BlocksPerRow := inDim / 32
  let q8InputU32Size := q8BlocksPerRow * 9 * seqLen
  let totalOutputSize := outDim * seqLen

  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareReadOnlyBuffer "input_q8" (.array (.scalar .u32) q8InputU32Size)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) totalOutputSize)

  let inBounds := Exp.lt outIdx (Exp.litU32 outDim)

  let iqs : Exp (.scalar .u32) := tid
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
    (Exp.mul (Exp.add outIdx (Exp.litU32 rowOffset)) (Exp.litU32 (blocksPerRow * blockSizeBytes)))
  let rowByteBase : Exp (.scalar .u32) := Exp.var "rowByteBase"

  -- Per-column Q8_1 base offset: column `colIdx` lives at
  --   q8ColBase = colIdx * (q8BlocksPerRow * 9)
  ShaderM.varNamed "q8ColBase" (.scalar .u32)
    (Exp.mul colIdx (Exp.litU32 (q8BlocksPerRow * 9)))
  let q8ColBase : Exp (.scalar .u32) := Exp.var "q8ColBase"

  let readByte (blockBase : Exp (.scalar .u32)) (offset : Exp (.scalar .u32)) : ShaderM (Exp (.scalar .u32)) := do
    let byteIdx := Exp.add blockBase offset
    ShaderM.readBufferByte (n := totalWeightU32) "weights" byteIdx

  let readU16 (blockBase : Exp (.scalar .u32)) (offset : Exp (.scalar .u32)) : ShaderM (Exp (.scalar .u32)) := do
    let byteIdx := Exp.add blockBase offset
    ShaderM.readBufferU16 (n := totalWeightU32) "weights" byteIdx

  let read4Bytes (blockBase : Exp (.scalar .u32)) (offset : Exp (.scalar .u32)) : ShaderM (Exp (.scalar .u32)) := do
    let byteIdxName ← ShaderM.var (.scalar .u32) (Exp.add blockBase offset)
    let byteIdx : Exp (.scalar .u32) := Exp.var byteIdxName
    let u32IdxName ← ShaderM.var (.scalar .u32) (Exp.shiftRight byteIdx (Exp.litU32 2))
    let u32Idx : Exp (.scalar .u32) := Exp.var u32IdxName
    let byteOffName ← ShaderM.var (.scalar .u32) (Exp.bitAnd byteIdx (Exp.litU32 3))
    let byteOff : Exp (.scalar .u32) := Exp.var byteOffName
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

    let dBitsName ← ShaderM.var (.scalar .u32) (← readU16 blockByteBase (Exp.litU32 208))
    let dBits : Exp (.scalar .u32) := Exp.var dBitsName
    let d := Exp.vecX (Exp.unpack2x16float dBits)

    let vlOffset := Exp.mul iqs (Exp.litU32 4)
    let vlName ← ShaderM.var (.scalar .u32) (← read4Bytes blockByteBase vlOffset)
    let vl : Exp (.scalar .u32) := Exp.var vlName
    let vhOffset := Exp.add (Exp.litU32 128) (Exp.mul vhIdx (Exp.litU32 4))
    let vhRaw ← read4Bytes blockByteBase vhOffset
    let vhName ← ShaderM.var (.scalar .u32) (Exp.shiftRight vhRaw vhShift)
    let vh : Exp (.scalar .u32) := Exp.var vhName

    let sc0ByteName ← ShaderM.var (.scalar .u32)
      (← readByte blockByteBase (Exp.add (Exp.litU32 192) scaleOff))
    let sc1ByteName ← ShaderM.var (.scalar .u32)
      (← readByte blockByteBase (Exp.add (Exp.litU32 192) (Exp.add scaleOff (Exp.litU32 4))))
    let sc0Byte : Exp (.scalar .u32) := Exp.var sc0ByteName
    let sc1Byte : Exp (.scalar .u32) := Exp.var sc1ByteName
    let signExtI8 (b : Exp (.scalar .u32)) : Exp (.scalar .u32) :=
      Exp.select (Exp.ge b (Exp.litU32 128))
        (Exp.bitOr b (Exp.litU32 0xFFFFFF00)) b
    let sc0I : Exp (.scalar .i32) := Exp.toI32 (signExtI8 sc0Byte)
    let sc1I : Exp (.scalar .i32) := Exp.toI32 (signExtI8 sc1Byte)

    -- Q8_1 reads use the column-shifted base.
    let q8BlockIdx i :=
      Exp.add q8ColBase
        (Exp.add (Exp.mul blockIdx (Exp.litU32 (8 * 9)))
                 (Exp.mul (Exp.add bq8Off (Exp.mul (Exp.litU32 i) (Exp.litU32 2))) (Exp.litU32 9)))
    let q8Sub0 := q8BlockIdx 0
    let q8Sub1 := q8BlockIdx 1
    let u0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8"
              (Exp.add q8Sub0 (Exp.add (Exp.litU32 1) q8ElemOff))
    let u1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8"
              (Exp.add q8Sub1 (Exp.add (Exp.litU32 1) q8ElemOff))
    let q8Hdr0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" q8Sub0
    let q8Hdr1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" q8Sub1
    let d8AName ← ShaderM.var (.scalar .f32) (Exp.vecX (Exp.unpack2x16float q8Hdr0))
    let d8BName ← ShaderM.var (.scalar .f32) (Exp.vecX (Exp.unpack2x16float q8Hdr1))
    let d8A : Exp (.scalar .f32) := Exp.var d8AName
    let d8B : Exp (.scalar .f32) := Exp.var d8BName

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

    ShaderM.assign "acc" (Exp.fma d (Exp.add sumf_0 sumf_1) acc)

  ShaderM.varNamed "total" (.scalar .f32) (Exp.subgroupAdd acc)
  let total : Exp (.scalar .f32) := Exp.var "total"

  let colInBounds := Exp.lt colIdx (Exp.litU32 seqLen)
  let valid := Exp.and (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) colInBounds
  ShaderM.if_ valid (do
    let outOff := Exp.add (Exp.mul colIdx (Exp.litU32 outDim)) outIdx
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outOff total
  ) (pure ())

/-- Batched Q6_K × **f32** matmul — same 32-lane-warp / subgroupAdd structure and Q6_K
    weight decode as `q6kMatmulBatchKernel`, but reads the RAW f32 activation (no Q8_1
    quant) and does an f32 dot. For the argmax-sensitive lm_head: keeps full f32 precision
    (so the decode output is unchanged) while getting the warp occupancy/barrier win over
    the block-parallel `fusedQ6KBatchKernel` (which uses only 11/256 threads).
    The f32 element index matching q8 sub-block `i` is `colIdx*inDim + blockIdx*256 +
    (bq8Off + 2*i)*32 + q8ElemOff*4 + k` (the Q8_1 was quantized in element order). -/
def fusedQ6KBatchF32WarpKernel (inDim outDim seqLen : Nat)
    (rowOffset : Nat := 0) (weightRows : Nat := 0) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let colIdx := Exp.vec3Y wid
  let tid := Exp.vec3X lid
  let blocksPerRow := inDim / 256
  let blockSizeBytes : Nat := 210
  let wRows := if weightRows == 0 then outDim else weightRows
  let totalWeightBytes := wRows * blocksPerRow * blockSizeBytes
  let totalWeightU32 := (totalWeightBytes + 3) / 4
  let totalInputSize := seqLen * inDim
  let totalOutputSize := outDim * seqLen
  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareReadOnlyBuffer "input" (.array (.scalar .f32) totalInputSize)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) totalOutputSize)
  let inBounds := Exp.lt outIdx (Exp.litU32 outDim)
  let iqs : Exp (.scalar .u32) := tid
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
    (Exp.mul (Exp.add outIdx (Exp.litU32 rowOffset)) (Exp.litU32 (blocksPerRow * blockSizeBytes)))
  let rowByteBase : Exp (.scalar .u32) := Exp.var "rowByteBase"
  let inColBase := Exp.mul colIdx (Exp.litU32 inDim)
  let readByte (blockBase : Exp (.scalar .u32)) (offset : Exp (.scalar .u32)) : ShaderM (Exp (.scalar .u32)) := do
    ShaderM.readBufferByte (n := totalWeightU32) "weights" (Exp.add blockBase offset)
  let readU16 (blockBase : Exp (.scalar .u32)) (offset : Exp (.scalar .u32)) : ShaderM (Exp (.scalar .u32)) := do
    ShaderM.readBufferU16 (n := totalWeightU32) "weights" (Exp.add blockBase offset)
  let read4Bytes (blockBase : Exp (.scalar .u32)) (offset : Exp (.scalar .u32)) : ShaderM (Exp (.scalar .u32)) := do
    let byteIdxName ← ShaderM.var (.scalar .u32) (Exp.add blockBase offset)
    let byteIdx : Exp (.scalar .u32) := Exp.var byteIdxName
    let u32IdxName ← ShaderM.var (.scalar .u32) (Exp.shiftRight byteIdx (Exp.litU32 2))
    let u32Idx : Exp (.scalar .u32) := Exp.var u32IdxName
    let byteOffName ← ShaderM.var (.scalar .u32) (Exp.bitAnd byteIdx (Exp.litU32 3))
    let byteOff : Exp (.scalar .u32) := Exp.var byteOffName
    let w0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" u32Idx
    let w1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add u32Idx (Exp.litU32 1))
    let lo := Exp.shiftRight w0 (Exp.mul byteOff (Exp.litU32 8))
    let hi := Exp.select (Exp.eq byteOff (Exp.litU32 0)) (Exp.litU32 0)
                (Exp.shiftLeft w1 (Exp.mul (Exp.sub (Exp.litU32 4) byteOff) (Exp.litU32 8)))
    pure (Exp.bitOr lo hi)
  let sub32PerByte (x : Exp (.scalar .u32)) : Exp (.scalar .u32) :=
    Exp.bitXor (Exp.sub (Exp.bitOr x (Exp.litU32 0x80808080)) (Exp.litU32 0x20202020)) (Exp.litU32 0x80808080)
  let signByteF32 (x : Exp (.scalar .u32)) (sh : Nat) : Exp (.scalar .f32) :=
    let bb := Exp.bitAnd (Exp.shiftRight x (Exp.litU32 sh)) (Exp.litU32 0xFF)
    Exp.select (Exp.ge bb (Exp.litU32 128)) (Exp.sub (Exp.toF32 bb) (Exp.litF32 256.0)) (Exp.toF32 bb)
  let signExtI8 (b : Exp (.scalar .u32)) : Exp (.scalar .u32) :=
    Exp.select (Exp.ge b (Exp.litU32 128)) (Exp.bitOr b (Exp.litU32 0xFFFFFF00)) b
  ShaderM.loop (Exp.litU32 0) (Exp.litU32 blocksPerRow) (Exp.litU32 1) fun blockIdx => do
    let blockByteBase := Exp.add rowByteBase (Exp.mul blockIdx (Exp.litU32 blockSizeBytes))
    let dBitsName ← ShaderM.var (.scalar .u32) (← readU16 blockByteBase (Exp.litU32 208))
    let d := Exp.vecX (Exp.unpack2x16float (Exp.var dBitsName))
    let vlName ← ShaderM.var (.scalar .u32) (← read4Bytes blockByteBase (Exp.mul iqs (Exp.litU32 4)))
    let vl : Exp (.scalar .u32) := Exp.var vlName
    let vhRaw ← read4Bytes blockByteBase (Exp.add (Exp.litU32 128) (Exp.mul vhIdx (Exp.litU32 4)))
    let vhName ← ShaderM.var (.scalar .u32) (Exp.shiftRight vhRaw vhShift)
    let vh : Exp (.scalar .u32) := Exp.var vhName
    let sc0Byte ← readByte blockByteBase (Exp.add (Exp.litU32 192) scaleOff)
    let sc1Byte ← readByte blockByteBase (Exp.add (Exp.litU32 192) (Exp.add scaleOff (Exp.litU32 4)))
    let scA := Exp.toF32 (Exp.toI32 (signExtI8 sc0Byte))
    let scB := Exp.toF32 (Exp.toI32 (signExtI8 sc1Byte))
    let inBase0 := Exp.add inColBase (Exp.add (Exp.mul blockIdx (Exp.litU32 256))
      (Exp.add (Exp.mul bq8Off (Exp.litU32 32)) (Exp.mul q8ElemOff (Exp.litU32 4))))
    let inBase1 := Exp.add inBase0 (Exp.litU32 64)
    let rd (base : Exp (.scalar .u32)) (o : Nat) : ShaderM (Exp (.scalar .f32)) :=
      ShaderM.readBuffer (ty := .scalar .f32) (n := totalInputSize) "input" (Exp.add base (Exp.litU32 o))
    let i00 ← rd inBase0 0; let i01 ← rd inBase0 1; let i02 ← rd inBase0 2; let i03 ← rd inBase0 3
    let i10 ← rd inBase1 0; let i11 ← rd inBase1 1; let i12 ← rd inBase1 2; let i13 ← rd inBase1 3
    let vi_0 := sub32PerByte (Exp.bitOr (Exp.bitAnd vl (Exp.litU32 0x0F0F0F0F))
      (Exp.bitAnd (Exp.shiftLeft vh (Exp.litU32 4)) (Exp.litU32 0x30303030)))
    let dot_0 := Exp.add (Exp.add (Exp.mul (signByteF32 vi_0 0) i00) (Exp.mul (signByteF32 vi_0 8) i01))
                         (Exp.add (Exp.mul (signByteF32 vi_0 16) i02) (Exp.mul (signByteF32 vi_0 24) i03))
    let vi_1 := sub32PerByte (Exp.bitOr (Exp.bitAnd (Exp.shiftRight vl (Exp.litU32 4)) (Exp.litU32 0x0F0F0F0F))
      (Exp.bitAnd (Exp.shiftLeft (Exp.shiftRight vh (Exp.litU32 4)) (Exp.litU32 4)) (Exp.litU32 0x30303030)))
    let dot_1 := Exp.add (Exp.add (Exp.mul (signByteF32 vi_1 0) i10) (Exp.mul (signByteF32 vi_1 8) i11))
                         (Exp.add (Exp.mul (signByteF32 vi_1 16) i12) (Exp.mul (signByteF32 vi_1 24) i13))
    ShaderM.assign "acc" (Exp.fma d (Exp.add (Exp.mul scA dot_0) (Exp.mul scB dot_1)) acc)
  ShaderM.varNamed "total" (.scalar .f32) (Exp.subgroupAdd acc)
  let total : Exp (.scalar .f32) := Exp.var "total"
  let valid := Exp.and (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (Exp.lt colIdx (Exp.litU32 seqLen))
  ShaderM.if_ valid (do
    ShaderM.writeBuffer (ty := .scalar .f32) "output" (Exp.add (Exp.mul colIdx (Exp.litU32 outDim)) outIdx) total
  ) (pure ())

/-- 4-warp coop K batched Q6_K matmul: `[outDim, inDim] × [inDim, seqLen]`.

    Mirror of `fusedQ6KLinearDP4A4WarpKernel` with a column axis. Each WG
    has 128 threads (4 warps × 32 lanes). 4 warps cooperate on K — warp `w`
    covers blocks `{w, w+4, w+8, ...}`. Cross-warp sum via smem then
    final warp-0 reduction.

    Grid: `(outDim, seqLen, 1) × {x: 32, y: 4, z: 1}`. Each WG computes one
    `output[colIdx * outDim + outIdx]`.

    Targets ~2× speedup over the 1-warp `q6kMatmulBatchKernel` for the
    prefill ffn_down kernel (1285 µs/call → ~640 µs/call expected). -/
def q6kMatmulBatch4WarpKernel (inDim outDim : Nat) (seqLen : Nat) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let colIdx := Exp.vec3Y wid
  -- block = (32, 4, 1): localId.x = lane (0..31), localId.y = warpId (0..3).
  let laneId  := Exp.vec3X lid
  let warpId  := Exp.vec3Y lid

  let blocksPerRow := inDim / 256
  let blockSizeBytes : Nat := 210
  let totalWeightBytes := outDim * blocksPerRow * blockSizeBytes
  let totalWeightU32 := (totalWeightBytes + 3) / 4
  let q8BlocksPerRow := inDim / 32
  let q8InputU32Size := q8BlocksPerRow * 9 * seqLen
  let totalOutputSize := outDim * seqLen

  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareReadOnlyBuffer "input_q8" (.array (.scalar .u32) q8InputU32Size)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) totalOutputSize)

  let inBounds := Exp.lt outIdx (Exp.litU32 outDim)
  let colInBounds := Exp.lt colIdx (Exp.litU32 seqLen)

  let iqsName       ← ShaderM.var (.scalar .u32) laneId
  let iqs           : Exp (.scalar .u32) := Exp.var iqsName
  let iqsDiv16Name  ← ShaderM.var (.scalar .u32) (Exp.shiftRight iqs (Exp.litU32 4))
  let iqsMod16Name  ← ShaderM.var (.scalar .u32) (Exp.bitAnd iqs (Exp.litU32 15))
  let iqsMod8Name   ← ShaderM.var (.scalar .u32) (Exp.bitAnd iqs (Exp.litU32 7))
  let iqsDiv16      : Exp (.scalar .u32) := Exp.var iqsDiv16Name
  let iqsMod16      : Exp (.scalar .u32) := Exp.var iqsMod16Name
  let iqsMod8       : Exp (.scalar .u32) := Exp.var iqsMod8Name
  let bq8OffName    ← ShaderM.var (.scalar .u32)
    (Exp.add (Exp.mul iqsDiv16 (Exp.litU32 4)) (Exp.shiftRight iqsMod16 (Exp.litU32 3)))
  let scaleOffName  ← ShaderM.var (.scalar .u32)
    (Exp.add (Exp.mul iqsDiv16 (Exp.litU32 8)) (Exp.shiftRight iqsMod16 (Exp.litU32 2)))
  let vhShiftName   ← ShaderM.var (.scalar .u32)
    (Exp.mul (Exp.shiftRight iqsMod16 (Exp.litU32 3)) (Exp.litU32 2))
  let vhIdxName     ← ShaderM.var (.scalar .u32)
    (Exp.add (Exp.mul iqsDiv16 (Exp.litU32 8)) iqsMod8)
  let bq8Off    : Exp (.scalar .u32) := Exp.var bq8OffName
  let scaleOff  : Exp (.scalar .u32) := Exp.var scaleOffName
  let vhShift   : Exp (.scalar .u32) := Exp.var vhShiftName
  let vhIdx     : Exp (.scalar .u32) := Exp.var vhIdxName
  let q8ElemOff : Exp (.scalar .u32) := iqsMod8

  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"

  ShaderM.varNamed "rowByteBase" (.scalar .u32)
    (Exp.mul outIdx (Exp.litU32 (blocksPerRow * blockSizeBytes)))
  let rowByteBase : Exp (.scalar .u32) := Exp.var "rowByteBase"

  -- Per-column Q8_1 base offset.
  ShaderM.varNamed "q8ColBase" (.scalar .u32)
    (Exp.mul colIdx (Exp.litU32 (q8BlocksPerRow * 9)))
  let q8ColBase : Exp (.scalar .u32) := Exp.var "q8ColBase"

  let readByte (blockBase : Exp (.scalar .u32)) (offset : Exp (.scalar .u32)) : ShaderM (Exp (.scalar .u32)) := do
    let byteIdx := Exp.add blockBase offset
    ShaderM.readBufferByte (n := totalWeightU32) "weights" byteIdx

  let readU16 (blockBase : Exp (.scalar .u32)) (offset : Exp (.scalar .u32)) : ShaderM (Exp (.scalar .u32)) := do
    let byteIdx := Exp.add blockBase offset
    ShaderM.readBufferU16 (n := totalWeightU32) "weights" byteIdx

  let read4Bytes (blockBase : Exp (.scalar .u32)) (offset : Exp (.scalar .u32)) : ShaderM (Exp (.scalar .u32)) := do
    let byteIdxName ← ShaderM.var (.scalar .u32) (Exp.add blockBase offset)
    let byteIdx : Exp (.scalar .u32) := Exp.var byteIdxName
    let u32IdxName ← ShaderM.var (.scalar .u32) (Exp.shiftRight byteIdx (Exp.litU32 2))
    let u32Idx : Exp (.scalar .u32) := Exp.var u32IdxName
    let byteOffName ← ShaderM.var (.scalar .u32) (Exp.bitAnd byteIdx (Exp.litU32 3))
    let byteOff : Exp (.scalar .u32) := Exp.var byteOffName
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

  -- Per-warp K-stride loop: warpId 0..3 covers blocks {warpId, warpId+4, ...}.
  ShaderM.loop warpId (Exp.litU32 blocksPerRow) (Exp.litU32 4) fun blockIdx => do
    let blockByteBaseName ← ShaderM.var (.scalar .u32)
      (Exp.add rowByteBase (Exp.mul blockIdx (Exp.litU32 blockSizeBytes)))
    let blockByteBase : Exp (.scalar .u32) := Exp.var blockByteBaseName

    let dBitsName ← ShaderM.var (.scalar .u32) (← readU16 blockByteBase (Exp.litU32 208))
    let dBits : Exp (.scalar .u32) := Exp.var dBitsName
    let d := Exp.vecX (Exp.unpack2x16float dBits)

    let vlOffset := Exp.mul iqs (Exp.litU32 4)
    let vlName ← ShaderM.var (.scalar .u32) (← read4Bytes blockByteBase vlOffset)
    let vl : Exp (.scalar .u32) := Exp.var vlName
    let vhOffset := Exp.add (Exp.litU32 128) (Exp.mul vhIdx (Exp.litU32 4))
    let vhRaw ← read4Bytes blockByteBase vhOffset
    let vhName ← ShaderM.var (.scalar .u32) (Exp.shiftRight vhRaw vhShift)
    let vh : Exp (.scalar .u32) := Exp.var vhName

    let sc0ByteName ← ShaderM.var (.scalar .u32)
      (← readByte blockByteBase (Exp.add (Exp.litU32 192) scaleOff))
    let sc1ByteName ← ShaderM.var (.scalar .u32)
      (← readByte blockByteBase (Exp.add (Exp.litU32 192) (Exp.add scaleOff (Exp.litU32 4))))
    let sc0Byte : Exp (.scalar .u32) := Exp.var sc0ByteName
    let sc1Byte : Exp (.scalar .u32) := Exp.var sc1ByteName
    let signExtI8 (b : Exp (.scalar .u32)) : Exp (.scalar .u32) :=
      Exp.select (Exp.ge b (Exp.litU32 128))
        (Exp.bitOr b (Exp.litU32 0xFFFFFF00)) b
    let sc0I : Exp (.scalar .i32) := Exp.toI32 (signExtI8 sc0Byte)
    let sc1I : Exp (.scalar .i32) := Exp.toI32 (signExtI8 sc1Byte)

    -- Q8_1 reads use the column-shifted base.
    let q8BlockIdx i :=
      Exp.add q8ColBase
        (Exp.add (Exp.mul blockIdx (Exp.litU32 (8 * 9)))
                 (Exp.mul (Exp.add bq8Off (Exp.mul (Exp.litU32 i) (Exp.litU32 2))) (Exp.litU32 9)))
    let q8Sub0 := q8BlockIdx 0
    let q8Sub1 := q8BlockIdx 1
    let u0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8"
              (Exp.add q8Sub0 (Exp.add (Exp.litU32 1) q8ElemOff))
    let u1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8"
              (Exp.add q8Sub1 (Exp.add (Exp.litU32 1) q8ElemOff))
    let q8Hdr0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" q8Sub0
    let q8Hdr1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := q8InputU32Size) "input_q8" q8Sub1
    let d8AName ← ShaderM.var (.scalar .f32) (Exp.vecX (Exp.unpack2x16float q8Hdr0))
    let d8BName ← ShaderM.var (.scalar .f32) (Exp.vecX (Exp.unpack2x16float q8Hdr1))
    let d8A : Exp (.scalar .f32) := Exp.var d8AName
    let d8B : Exp (.scalar .f32) := Exp.var d8BName

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

    ShaderM.assign "acc" (Exp.fma d (Exp.add sumf_0 sumf_1) acc)

  -- Cross-warp reduction (same scheme as fusedQ6KLinearDP4A4WarpKernel).
  let nwarpsMinus1 : Nat := 3
  let warpSize    : Nat := 32
  let smemSize    : Nat := nwarpsMinus1 * warpSize
  ShaderM.sharedNamed "s_warp_partials" (.array (.scalar .f32) smemSize)

  ShaderM.if_ (Exp.gt warpId (Exp.litU32 0)) (do
    let warpIdM1 := Exp.sub warpId (Exp.litU32 1)
    let smemIdx := Exp.add (Exp.mul warpIdM1 (Exp.litU32 32)) laneId
    ShaderM.writeWorkgroup (ty := .scalar .f32) "s_warp_partials" smemIdx acc
  ) (pure ())
  ShaderM.barrier

  let p1 ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := smemSize) "s_warp_partials"
            (Exp.add (Exp.litU32 0) laneId)
  let p2 ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := smemSize) "s_warp_partials"
            (Exp.add (Exp.litU32 32) laneId)
  let p3 ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := smemSize) "s_warp_partials"
            (Exp.add (Exp.litU32 64) laneId)
  ShaderM.assign "acc" (acc + p1)
  ShaderM.assign "acc" ((Exp.var "acc" : Exp (.scalar .f32)) + p2)
  ShaderM.assign "acc" ((Exp.var "acc" : Exp (.scalar .f32)) + p3)
  ShaderM.varNamed "total" (.scalar .f32)
    (Exp.subgroupAdd (Exp.var "acc" : Exp (.scalar .f32)))
  let total : Exp (.scalar .f32) := Exp.var "total"
  ShaderM.if_ (Exp.and (Exp.and (Exp.and (Exp.eq warpId (Exp.litU32 0))
                                         (Exp.eq laneId (Exp.litU32 0)))
                                inBounds) colInBounds) (do
    let outOff := Exp.add (Exp.mul colIdx (Exp.litU32 outDim)) outIdx
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outOff total
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

  -- Native u8 load — one `ld.global.nc.u8` on CUDA (vs 1 u32 load + shift +
  -- mask previously).  See docs/llama-fusion-analysis/41.md for the PTX
  -- diff that motivated this primitive.
  let readByte (blockBase : Exp (.scalar .u32)) (offset : Exp (.scalar .u32)) : ShaderM (Exp (.scalar .u32)) := do
    let byteIdx := Exp.add blockBase offset
    ShaderM.readBufferByte (n := totalWeightU32) "weights" byteIdx

  -- Native u16 load — one `ld.global.nc.u16` on CUDA.  Used to read the
  -- 2-byte fp16 block scale `d` in a single instruction.
  let readU16 (blockBase : Exp (.scalar .u32)) (offset : Exp (.scalar .u32)) : ShaderM (Exp (.scalar .u32)) := do
    let byteIdx := Exp.add blockBase offset
    ShaderM.readBufferU16 (n := totalWeightU32) "weights" byteIdx

  let read4Bytes (blockBase : Exp (.scalar .u32)) (offset : Exp (.scalar .u32)) : ShaderM (Exp (.scalar .u32)) := do
    -- Bind byteIdx / u32Idx / byteOff so they materialise once in PTX.
    -- Without these, each downstream use re-emits the `blockBase+offset`
    -- chain (6× add.u32 + 6× shl.b32 per read4Bytes call in earlier PTX).
    let byteIdxName ← ShaderM.var (.scalar .u32) (Exp.add blockBase offset)
    let byteIdx : Exp (.scalar .u32) := Exp.var byteIdxName
    let u32IdxName ← ShaderM.var (.scalar .u32) (Exp.shiftRight byteIdx (Exp.litU32 2))
    let u32Idx : Exp (.scalar .u32) := Exp.var u32IdxName
    let byteOffName ← ShaderM.var (.scalar .u32) (Exp.bitAnd byteIdx (Exp.litU32 3))
    let byteOff : Exp (.scalar .u32) := Exp.var byteOffName
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
    -- Hoist per-iter base so all readByte/read4Bytes calls below share
    -- one address-chain register (roughly a dozen global-memory reads
    -- per iter — without this the base re-materialises for each one).
    let blockByteBaseName ← ShaderM.var (.scalar .u32)
      (Exp.add rowByteBase (Exp.mul blockIdx (Exp.litU32 blockSizeBytes)))
    let blockByteBase : Exp (.scalar .u32) := Exp.var blockByteBaseName

    -- Single ld.global.nc.u16 for the fp16 block scale, converted to f32
    -- via `cvt.f32.f16` (hardware instruction, 1 op) instead of the
    -- `fp16ToF32` arithmetic soft-impl which PTX-expands to 15+ ops
    -- (selp/div/ex2/sub for sign/mantissa/exponent).  We reinterpret the
    -- u16 as the low half of a packed half2 and extract the low f16 — the
    -- codegen path for `vecX (unpack2x16float _)` lowers to exactly
    -- `mov.b32 {lo,hi}, r; cvt.f32.f16 f, lo`.
    let dBitsName ← ShaderM.var (.scalar .u32) (← readU16 blockByteBase (Exp.litU32 208))
    let dBits : Exp (.scalar .u32) := Exp.var dBitsName
    let d := Exp.vecX (Exp.unpack2x16float dBits)

    let vlOffset := Exp.mul iqs (Exp.litU32 4)
    let vl ← read4Bytes blockByteBase vlOffset
    let vhOffset := Exp.add (Exp.litU32 128) (Exp.mul vhIdx (Exp.litU32 4))
    let vhRaw ← read4Bytes blockByteBase vhOffset
    let vh := Exp.shiftRight vhRaw vhShift

    -- Bind each scale byte so it's loaded exactly once — signExtI8 below
    -- references `b` three times, and without the bind the readByte call
    -- inlines to 3× ld.global.u8 per scale (6 total per iter) even though
    -- the HW would cache them.  Makes PTX match llama.cpp's 2 ld.u8/iter.
    let sc0ByteName ← ShaderM.var (.scalar .u32)
      (← readByte blockByteBase (Exp.add (Exp.litU32 192) scaleOff))
    let sc1ByteName ← ShaderM.var (.scalar .u32)
      (← readByte blockByteBase (Exp.add (Exp.litU32 192) (Exp.add scaleOff (Exp.litU32 4))))
    let sc0Byte : Exp (.scalar .u32) := Exp.var sc0ByteName
    let sc1Byte : Exp (.scalar .u32) := Exp.var sc1ByteName
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
    -- Q8_1 header is now half2(d, sum) packed in a u32.  Extract `d` via
    -- the low f16 (sum lives in the high f16 — currently unused by hesper's
    -- matmul, but matches llama.cpp layout).
    -- Hoist the f16→f32 conversion: each d8A/d8B is referenced 2-4× below.
    -- Without a ShaderM.var binding, CSE still has to re-emit the
    -- `mov.b32 {lo,hi}, r; cvt.f32.f16` pair for every reference.  The
    -- explicit bind + `Exp.var` forces a single register reuse.
    let d8AName ← ShaderM.var (.scalar .f32) (Exp.vecX (Exp.unpack2x16float q8Hdr0))
    let d8BName ← ShaderM.var (.scalar .f32) (Exp.vecX (Exp.unpack2x16float q8Hdr1))
    let d8A : Exp (.scalar .f32) := Exp.var d8AName
    let d8B : Exp (.scalar .f32) := Exp.var d8BName

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

    -- acc = d * (sumf_0 + sumf_1) + acc → one fma.rn.f32 instead of mul + add.
    ShaderM.assign "acc" (Exp.fma d (Exp.add sumf_0 sumf_1) acc)

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
  -- Hoist every thread-invariant sub-expression into an explicit register.
  -- Without this `ShaderM.var` lift, the lazy `Exp` representation re-emits
  -- each use at its call site; ptxas often CSEs them back, but the
  -- verbose PTX hides register pressure and forced the same fix that
  -- landed on the Q4_K 4-warp kernel (commit 9d43d81).
  let iqsName       ← ShaderM.var (.scalar .u32) laneId
  let iqs           : Exp (.scalar .u32) := Exp.var iqsName
  let iqsDiv16Name  ← ShaderM.var (.scalar .u32) (Exp.shiftRight iqs (Exp.litU32 4))
  let iqsMod16Name  ← ShaderM.var (.scalar .u32) (Exp.bitAnd iqs (Exp.litU32 15))
  let iqsMod8Name   ← ShaderM.var (.scalar .u32) (Exp.bitAnd iqs (Exp.litU32 7))
  let iqsDiv16      : Exp (.scalar .u32) := Exp.var iqsDiv16Name
  let iqsMod16      : Exp (.scalar .u32) := Exp.var iqsMod16Name
  let iqsMod8       : Exp (.scalar .u32) := Exp.var iqsMod8Name
  let bq8OffName    ← ShaderM.var (.scalar .u32)
    (Exp.add (Exp.mul iqsDiv16 (Exp.litU32 4)) (Exp.shiftRight iqsMod16 (Exp.litU32 3)))
  let scaleOffName  ← ShaderM.var (.scalar .u32)
    (Exp.add (Exp.mul iqsDiv16 (Exp.litU32 8)) (Exp.shiftRight iqsMod16 (Exp.litU32 2)))
  let vhShiftName   ← ShaderM.var (.scalar .u32)
    (Exp.mul (Exp.shiftRight iqsMod16 (Exp.litU32 3)) (Exp.litU32 2))
  let vhIdxName     ← ShaderM.var (.scalar .u32)
    (Exp.add (Exp.mul iqsDiv16 (Exp.litU32 8)) iqsMod8)
  let bq8Off    : Exp (.scalar .u32) := Exp.var bq8OffName
  let scaleOff  : Exp (.scalar .u32) := Exp.var scaleOffName
  let vhShift   : Exp (.scalar .u32) := Exp.var vhShiftName
  let vhIdx     : Exp (.scalar .u32) := Exp.var vhIdxName
  let q8ElemOff : Exp (.scalar .u32) := iqsMod8  -- alias; already in register

  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"

  ShaderM.varNamed "rowByteBase" (.scalar .u32)
    (Exp.mul outIdx (Exp.litU32 (blocksPerRow * blockSizeBytes)))
  let rowByteBase : Exp (.scalar .u32) := Exp.var "rowByteBase"

  -- Native u8 load — one `ld.global.nc.u8` on CUDA (vs 1 u32 load + shift +
  -- mask previously).  See docs/llama-fusion-analysis/41.md for the PTX
  -- diff that motivated this primitive.
  let readByte (blockBase : Exp (.scalar .u32)) (offset : Exp (.scalar .u32)) : ShaderM (Exp (.scalar .u32)) := do
    let byteIdx := Exp.add blockBase offset
    ShaderM.readBufferByte (n := totalWeightU32) "weights" byteIdx

  -- Native u16 load — one `ld.global.nc.u16` on CUDA.  Used to read the
  -- 2-byte fp16 block scale `d` in a single instruction.
  let readU16 (blockBase : Exp (.scalar .u32)) (offset : Exp (.scalar .u32)) : ShaderM (Exp (.scalar .u32)) := do
    let byteIdx := Exp.add blockBase offset
    ShaderM.readBufferU16 (n := totalWeightU32) "weights" byteIdx

  let read4Bytes (blockBase : Exp (.scalar .u32)) (offset : Exp (.scalar .u32)) : ShaderM (Exp (.scalar .u32)) := do
    -- Bind byteIdx / u32Idx / byteOff so they materialise once in PTX.
    -- Without these, each downstream use re-emits the `blockBase+offset`
    -- chain (6× add.u32 + 6× shl.b32 per read4Bytes call in earlier PTX).
    let byteIdxName ← ShaderM.var (.scalar .u32) (Exp.add blockBase offset)
    let byteIdx : Exp (.scalar .u32) := Exp.var byteIdxName
    let u32IdxName ← ShaderM.var (.scalar .u32) (Exp.shiftRight byteIdx (Exp.litU32 2))
    let u32Idx : Exp (.scalar .u32) := Exp.var u32IdxName
    let byteOffName ← ShaderM.var (.scalar .u32) (Exp.bitAnd byteIdx (Exp.litU32 3))
    let byteOff : Exp (.scalar .u32) := Exp.var byteOffName
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
    -- Hoist per-iter base so all readByte/read4Bytes calls below share
    -- one address-chain register (roughly a dozen global-memory reads
    -- per iter — without this the base re-materialises for each one).
    let blockByteBaseName ← ShaderM.var (.scalar .u32)
      (Exp.add rowByteBase (Exp.mul blockIdx (Exp.litU32 blockSizeBytes)))
    let blockByteBase : Exp (.scalar .u32) := Exp.var blockByteBaseName

    -- Single ld.global.nc.u16 for the fp16 block scale, converted to f32
    -- via `cvt.f32.f16` (hardware instruction, 1 op) instead of the
    -- `fp16ToF32` arithmetic soft-impl which PTX-expands to 15+ ops
    -- (selp/div/ex2/sub for sign/mantissa/exponent).  We reinterpret the
    -- u16 as the low half of a packed half2 and extract the low f16 — the
    -- codegen path for `vecX (unpack2x16float _)` lowers to exactly
    -- `mov.b32 {lo,hi}, r; cvt.f32.f16 f, lo`.
    let dBitsName ← ShaderM.var (.scalar .u32) (← readU16 blockByteBase (Exp.litU32 208))
    let dBits : Exp (.scalar .u32) := Exp.var dBitsName
    let d := Exp.vecX (Exp.unpack2x16float dBits)

    let vlOffset := Exp.mul iqs (Exp.litU32 4)
    let vl ← read4Bytes blockByteBase vlOffset
    let vhOffset := Exp.add (Exp.litU32 128) (Exp.mul vhIdx (Exp.litU32 4))
    let vhRaw ← read4Bytes blockByteBase vhOffset
    let vh := Exp.shiftRight vhRaw vhShift

    -- Bind each scale byte so it's loaded exactly once — signExtI8 below
    -- references `b` three times, and without the bind the readByte call
    -- inlines to 3× ld.global.u8 per scale (6 total per iter) even though
    -- the HW would cache them.  Makes PTX match llama.cpp's 2 ld.u8/iter.
    let sc0ByteName ← ShaderM.var (.scalar .u32)
      (← readByte blockByteBase (Exp.add (Exp.litU32 192) scaleOff))
    let sc1ByteName ← ShaderM.var (.scalar .u32)
      (← readByte blockByteBase (Exp.add (Exp.litU32 192) (Exp.add scaleOff (Exp.litU32 4))))
    let sc0Byte : Exp (.scalar .u32) := Exp.var sc0ByteName
    let sc1Byte : Exp (.scalar .u32) := Exp.var sc1ByteName
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
    -- Q8_1 header is now half2(d, sum) packed in a u32.  Extract `d` via
    -- the low f16 (sum lives in the high f16 — currently unused by hesper's
    -- matmul, but matches llama.cpp layout).
    -- Hoist the f16→f32 conversion: each d8A/d8B is referenced 2-4× below.
    -- Without a ShaderM.var binding, CSE still has to re-emit the
    -- `mov.b32 {lo,hi}, r; cvt.f32.f16` pair for every reference.  The
    -- explicit bind + `Exp.var` forces a single register reuse.
    let d8AName ← ShaderM.var (.scalar .f32) (Exp.vecX (Exp.unpack2x16float q8Hdr0))
    let d8BName ← ShaderM.var (.scalar .f32) (Exp.vecX (Exp.unpack2x16float q8Hdr1))
    let d8A : Exp (.scalar .f32) := Exp.var d8AName
    let d8B : Exp (.scalar .f32) := Exp.var d8BName

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

    -- acc = d * (sumf_0 + sumf_1) + acc → one fma.rn.f32 instead of mul + add.
    ShaderM.assign "acc" (Exp.fma d (Exp.add sumf_0 sumf_1) acc)

  -- Per-warp reduction: each of the 4 warps independently reduces its
  -- 32 lanes.  Lane 0 of each warp writes one output row.
  ShaderM.varNamed "total" (.scalar .f32) (Exp.subgroupAdd acc)
  let total : Exp (.scalar .f32) := Exp.var "total"
  ShaderM.if_ (Exp.and (Exp.eq laneId (Exp.litU32 0)) inBounds) (do
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx total
  ) (pure ())

/-- 4-warp / **1 row** cooperative variant of `fusedQ6KLinearDP4AKernel`.

    Matches llama.cpp's `mul_mat_vec_q<Q6_K, ncols_dst=1>` exactly:
    `block = (32, 4, 1)` = 128 thread = 4 warps; `rows_per_cuda_block = 1`;
    `blocks_per_iter = vdr * nwarps * warp_size / qi = 1 * 4 * 32 / 32 = 4`.

    Each warp covers 1/4 of the K blocks (kbx = warpId, warpId+4, warpId+8,
    ...).  At K=10240 (40 blocks/row) each thread runs **10 iter** vs the
    1-warp variant's 40 — 4× per-row issue parallelism, faster DRAM
    saturation.  Per-warp partial sum reduces via `subgroupAdd`; warps
    1..3 publish to smem and warp 0 merges + writes the single output.

    Different from `fusedQ6KLinearDP4A4RowKernel` (4 rows / 1 warp each):
    that one trades cooperative parallelism for output count.  This one
    takes the llama.cpp shape (1 row, 4 warps cooperative on K).

    Dispatch: `outDim` workgroups × 128 threads. Same `inBounds` mask.

    @param inDim  Input dimension (must be multiple of 256 for Q6_K blocks)
    @param outDim Output dimension
    @param gridX  Grid X dimension for 2D grid; 0 for 1D. -/
def fusedQ6KLinearDP4A4WarpKernel (inDim outDim : Nat) (gridX : Nat := 0) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx :=
    if gridX == 0 then Exp.vec3X wid
    else Exp.add (Exp.vec3X wid) (Exp.mul (Exp.vec3Y wid) (Exp.litU32 gridX))
  -- block = (32, 4, 1): localId.x = lane (0..31), localId.y = warpId (0..3).
  let laneId  := Exp.vec3X lid
  let warpId  := Exp.vec3Y lid

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

  -- iqs is the lane-only address (0..31). Same DP4A indexing as the 1-row
  -- kernel; warpId is *not* part of `iqs` — warpId only shifts the kbx
  -- starting point along K.
  let iqsName       ← ShaderM.var (.scalar .u32) laneId
  let iqs           : Exp (.scalar .u32) := Exp.var iqsName
  let iqsDiv16Name  ← ShaderM.var (.scalar .u32) (Exp.shiftRight iqs (Exp.litU32 4))
  let iqsMod16Name  ← ShaderM.var (.scalar .u32) (Exp.bitAnd iqs (Exp.litU32 15))
  let iqsMod8Name   ← ShaderM.var (.scalar .u32) (Exp.bitAnd iqs (Exp.litU32 7))
  let iqsDiv16      : Exp (.scalar .u32) := Exp.var iqsDiv16Name
  let iqsMod16      : Exp (.scalar .u32) := Exp.var iqsMod16Name
  let iqsMod8       : Exp (.scalar .u32) := Exp.var iqsMod8Name
  let bq8OffName    ← ShaderM.var (.scalar .u32)
    (Exp.add (Exp.mul iqsDiv16 (Exp.litU32 4)) (Exp.shiftRight iqsMod16 (Exp.litU32 3)))
  let scaleOffName  ← ShaderM.var (.scalar .u32)
    (Exp.add (Exp.mul iqsDiv16 (Exp.litU32 8)) (Exp.shiftRight iqsMod16 (Exp.litU32 2)))
  let vhShiftName   ← ShaderM.var (.scalar .u32)
    (Exp.mul (Exp.shiftRight iqsMod16 (Exp.litU32 3)) (Exp.litU32 2))
  let vhIdxName     ← ShaderM.var (.scalar .u32)
    (Exp.add (Exp.mul iqsDiv16 (Exp.litU32 8)) iqsMod8)
  let bq8Off    : Exp (.scalar .u32) := Exp.var bq8OffName
  let scaleOff  : Exp (.scalar .u32) := Exp.var scaleOffName
  let vhShift   : Exp (.scalar .u32) := Exp.var vhShiftName
  let vhIdx     : Exp (.scalar .u32) := Exp.var vhIdxName
  let q8ElemOff : Exp (.scalar .u32) := iqsMod8

  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"

  ShaderM.varNamed "rowByteBase" (.scalar .u32)
    (Exp.mul outIdx (Exp.litU32 (blocksPerRow * blockSizeBytes)))
  let rowByteBase : Exp (.scalar .u32) := Exp.var "rowByteBase"

  -- Same byte/u16/u32 readers as the 1-row variant.
  let readByte (blockBase : Exp (.scalar .u32)) (offset : Exp (.scalar .u32)) : ShaderM (Exp (.scalar .u32)) := do
    let byteIdx := Exp.add blockBase offset
    ShaderM.readBufferByte (n := totalWeightU32) "weights" byteIdx

  let readU16 (blockBase : Exp (.scalar .u32)) (offset : Exp (.scalar .u32)) : ShaderM (Exp (.scalar .u32)) := do
    let byteIdx := Exp.add blockBase offset
    ShaderM.readBufferU16 (n := totalWeightU32) "weights" byteIdx

  let read4Bytes (blockBase : Exp (.scalar .u32)) (offset : Exp (.scalar .u32)) : ShaderM (Exp (.scalar .u32)) := do
    let byteIdxName ← ShaderM.var (.scalar .u32) (Exp.add blockBase offset)
    let byteIdx : Exp (.scalar .u32) := Exp.var byteIdxName
    let u32IdxName ← ShaderM.var (.scalar .u32) (Exp.shiftRight byteIdx (Exp.litU32 2))
    let u32Idx : Exp (.scalar .u32) := Exp.var u32IdxName
    let byteOffName ← ShaderM.var (.scalar .u32) (Exp.bitAnd byteIdx (Exp.litU32 3))
    let byteOff : Exp (.scalar .u32) := Exp.var byteOffName
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

  -- Per-warp K-stride loop: warpId 0..3 covers blocks {warpId, warpId+4, ...}.
  -- Inner loop count = blocksPerRow / 4 (10 at K=10240) — 4× shorter chain
  -- than the 1-warp variant's blocksPerRow.
  --
  -- We use a regular `ShaderM.loop` with a starting offset = warpId.  ptxas
  -- unrolls reliably when blocksPerRow is a compile-time multiple of 4.
  ShaderM.loop warpId (Exp.litU32 blocksPerRow) (Exp.litU32 4) fun blockIdx => do
    let blockByteBaseName ← ShaderM.var (.scalar .u32)
      (Exp.add rowByteBase (Exp.mul blockIdx (Exp.litU32 blockSizeBytes)))
    let blockByteBase : Exp (.scalar .u32) := Exp.var blockByteBaseName

    let dBitsName ← ShaderM.var (.scalar .u32) (← readU16 blockByteBase (Exp.litU32 208))
    let dBits : Exp (.scalar .u32) := Exp.var dBitsName
    let d := Exp.vecX (Exp.unpack2x16float dBits)

    let vlOffset := Exp.mul iqs (Exp.litU32 4)
    let vlName ← ShaderM.var (.scalar .u32) (← read4Bytes blockByteBase vlOffset)
    let vl : Exp (.scalar .u32) := Exp.var vlName
    let vhOffset := Exp.add (Exp.litU32 128) (Exp.mul vhIdx (Exp.litU32 4))
    let vhRaw ← read4Bytes blockByteBase vhOffset
    let vhName ← ShaderM.var (.scalar .u32) (Exp.shiftRight vhRaw vhShift)
    let vh : Exp (.scalar .u32) := Exp.var vhName

    let sc0ByteName ← ShaderM.var (.scalar .u32)
      (← readByte blockByteBase (Exp.add (Exp.litU32 192) scaleOff))
    let sc1ByteName ← ShaderM.var (.scalar .u32)
      (← readByte blockByteBase (Exp.add (Exp.litU32 192) (Exp.add scaleOff (Exp.litU32 4))))
    let sc0Byte : Exp (.scalar .u32) := Exp.var sc0ByteName
    let sc1Byte : Exp (.scalar .u32) := Exp.var sc1ByteName
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
    let d8AName ← ShaderM.var (.scalar .f32) (Exp.vecX (Exp.unpack2x16float q8Hdr0))
    let d8BName ← ShaderM.var (.scalar .f32) (Exp.vecX (Exp.unpack2x16float q8Hdr1))
    let d8A : Exp (.scalar .f32) := Exp.var d8AName
    let d8B : Exp (.scalar .f32) := Exp.var d8BName

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

    ShaderM.assign "acc" (Exp.fma d (Exp.add sumf_0 sumf_1) acc)

  -- Cross-warp reduction — match llama.cpp's `mul_mat_vec_q` exactly
  -- (mmvq.cu:503-549).  Key insight: do NOT collapse each warp to a
  -- scalar before publishing.  Instead, each lane of warps 1..3 stores
  -- its OWN per-lane partial to smem; warp 0 then reads back per-lane
  -- partials and accumulates them lane-parallel BEFORE the final
  -- `warp_reduce_sum`.  This preserves the same f32 summation order as
  -- the 1-warp kernel (binary-tree reduce of 32 partials), avoiding
  -- the round-off drift that scalar-merge introduced.
  --
  -- Smem: nwarps-1 = 3 entries × warp_size = 96 floats (384 bytes).
  let nwarpsMinus1 : Nat := 3
  let warpSize    : Nat := 32
  let smemSize    : Nat := nwarpsMinus1 * warpSize  -- 96
  ShaderM.sharedNamed "s_warp_partials" (.array (.scalar .f32) smemSize)

  -- Warps 1..3 publish (no per-warp reduce yet).  Layout:
  --   s_warp_partials[(warpId-1)*32 + laneId] = acc
  ShaderM.if_ (Exp.gt warpId (Exp.litU32 0)) (do
    let warpIdM1 := Exp.sub warpId (Exp.litU32 1)
    let smemIdx := Exp.add (Exp.mul warpIdM1 (Exp.litU32 32)) laneId
    ShaderM.writeWorkgroup (ty := .scalar .f32) "s_warp_partials" smemIdx acc
  ) (pure ())
  ShaderM.barrier

  -- Warps 1..3 done.  Warp 0 (all 32 lanes) accumulates lane-parallel.
  -- Note: `subgroupAdd` is a warp-collective op — every lane in the
  -- warp must reach it.  We can't put it inside an `if (warpId == 0)`
  -- guard because warps 1..3 also need to "execute" the same lowered
  -- shfl.bfly (a no-op for them since they returned).  Solution: do
  -- the smem reads + `subgroupAdd` for **all** warps; only warp 0
  -- writes the result.  Warps 1..3 read garbage and reduce it but
  -- their reduction result is thrown away by the warpId==0 guard
  -- on the final write.
  let p1 ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := smemSize) "s_warp_partials"
            (Exp.add (Exp.litU32 0) laneId)
  let p2 ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := smemSize) "s_warp_partials"
            (Exp.add (Exp.litU32 32) laneId)
  let p3 ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := smemSize) "s_warp_partials"
            (Exp.add (Exp.litU32 64) laneId)
  -- llama.cpp loop order: tmp += tmp_shared[0]; tmp += [1]; tmp += [2].
  ShaderM.assign "acc" (acc + p1)
  ShaderM.assign "acc" ((Exp.var "acc" : Exp (.scalar .f32)) + p2)
  ShaderM.assign "acc" ((Exp.var "acc" : Exp (.scalar .f32)) + p3)
  -- Warp-collective reduce.  All 4 warps participate (warps 1-3's
  -- result is unused).  Same binary-tree order as 1-warp kernel.
  ShaderM.varNamed "total" (.scalar .f32)
    (Exp.subgroupAdd (Exp.var "acc" : Exp (.scalar .f32)))
  let total : Exp (.scalar .f32) := Exp.var "total"
  ShaderM.if_ (Exp.and (Exp.and (Exp.eq warpId (Exp.litU32 0))
                                (Exp.eq laneId (Exp.litU32 0)))
                       inBounds) (do
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
    let d := Exp.vecX (Exp.unpack2x16float dmU32)
    let dmin := Exp.vecY (Exp.unpack2x16float dmU32)
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
      (Exp.toF32U scaleU, Exp.toF32U minU)

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
    let d := Exp.vecX (Exp.unpack2x16float dmU32)
    let dmin := Exp.vecY (Exp.unpack2x16float dmU32)

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
      (Exp.toF32U scaleU, Exp.toF32U minU)

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
    let d := Exp.vecX (Exp.unpack2x16float dmU32)
    let dmin := Exp.vecY (Exp.unpack2x16float dmU32)
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
  | Q8_0   -- Q8_0: 34 bytes per 32 elements (f16 scale + 32×int8)
  | Q5_0   -- Q5_0: 22 bytes per 32 elements (f16 scale + 4B qh + 16B qs)
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
  -- Separate prepared refs for the BATCHED (prefill) path.  The batch
  -- kernel takes seqLen as a compile-time parameter, so sharing the
  -- decode refs above would clobber the decode dispatch when prefill
  -- runs.  Without caching these, every one of the 42 layers'
  -- prefill-time Q4_K matmuls invokes `cuModuleLoadDataEx` (~350 µs)
  -- → ~50 ms of pure JIT overhead per prefill.
  dp4aBatchQuantizePrepared : IO.Ref (Option CacheT)
  dp4aBatchMatmulPrepared : IO.Ref (Option CacheT)

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

  -- Phase-0 hybrid override: if installed and backend exposes raw device
  -- pointers, dispatch to externally-JIT'd llama.cpp PTX instead.
  if let some fn ← llamaCppDp4aOverride.get then
    if let (some inPtr, some wPtr, some outPtr) ← (do
        let a ← GPUBackend.rawDevicePtr ctx inputBuf
        let b ← GPUBackend.rawDevicePtr ctx layer.weightBuf
        let c ← GPUBackend.rawDevicePtr ctx outputBuf
        pure (a, b, c)) then
      let tag := if layer.quantFormat == .Q4_K then 0 else 1
      let handled ← fn inPtr wPtr outPtr layer.config.inDim layer.config.outDim tag
      if handled then
        if profiling then
          let endNs ← IO.monoNanosNow
          let delta := (endNs - startNs).toUInt64
          totalNanosRef.modify (· + delta)
          callCountRef.modify (· + 1)
          perShapeAdd layer.config.inDim layer.config.outDim delta
        return

  let nQ8Blocks := layer.config.inDim / 32
  let q8BufBytes : USize := (nQ8Blocks * 9 * 4).toUSize

  -- Inline-quantize fast path: one dispatch for quantize+matmul.  Only
  -- wired for Q4_K 4-warp config (inDim%256==0 && outDim ≤ 5120) since
  -- those are the decode-hot matmuls.  Toggle with HESPER_INLINE_QUANT=1;
  -- default off pending correctness + perf validation.
  let inlineQuant := (match ← IO.getEnv "HESPER_INLINE_QUANT" with
                       | some "1" => true
                       | _        => false)
  let inlineEligible := layer.quantFormat == .Q4_K
                        && layer.config.inDim % 256 == 0
                        && layer.config.outDim ≤ 5120
  if inlineQuant && inlineEligible then
    GPUBackend.executeWithConfigCached ctx
      (fusedQ4KMLinearDP4A4WarpInlineQuantKernel layer.config)
      [("weights", layer.weightBuf), ("input", inputBuf), ("output", outputBuf)]
      { numWorkgroups := (layer.config.outDim, 1, 1)
        workgroupSize := { x := 128, y := 1, z := 1 } }
      (hash ("q4k-dp4a-matmul-4warp-inlineq", layer.config.inDim, layer.config.outDim))
      layer.dp4aMatmulPrepared
    if profiling then
      let endNs ← IO.monoNanosNow
      let delta := (endNs - startNs).toUInt64
      totalNanosRef.modify (· + delta)
      callCountRef.modify (· + 1)
      perShapeAdd layer.config.inDim layer.config.outDim delta
    return

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
    -- Kernel selection heuristic (llama.cpp parity):
    --   4-warp 1-row: outDim ≤ ~5120 (few WGs otherwise → poor latency hiding)
    --                 AND HESPER_Q4K_4WARP != "0" (opt-out for regression test)
    --   2-row:       outDim ≤ 5120, fallback path
    --   1-row:       outDim > 5120 (2-row would double sub-block smem pressure)
    let allow4Warp := (match ← IO.getEnv "HESPER_Q4K_4WARP" with
                       | some "0" => false
                       | _        => true)
    -- The earlier `outDim ≤ 5120` gate was carried over from the
    -- 2-row kernel's register-pressure concerns; the 4-warp kernel
    -- does NOT have those (each warp only keeps one row's running
    -- accumulator, summed across warps via smem at the end).  At
    -- outDim=10240 (ffn_gate / ffn_up) the 1-warp variant was
    -- ~9 ms/decode across 42 layers — enabling 4-warp here is the
    -- single biggest Q4_K throughput lever.  Opt-out via
    -- HESPER_Q4K_4WARP_WIDE=0 for regression comparison.
    let allow4WarpWide := (match ← IO.getEnv "HESPER_Q4K_4WARP_WIDE" with
                            | some "0" => false
                            | _        => true)
    let use4Warp := allow4Warp && (layer.config.outDim ≤ 5120 || allow4WarpWide)
    -- 4-row kernel exists (`fusedQ4KMLinearDP4A4RowKernel`, opt-in via
    -- HESPER_Q4K_4ROW=1) but A/B testing showed -3 to -5 TPS at the hot
    -- production shape (outDim=2560).  Reason: cutting dispatched WGs
    -- 4× also cuts the wave count (2560 WGs → 640 WGs = ~10 waves on
    -- RTX 4070 Ti's 60 SMs vs ~42 waves for the 4-warp variant), which
    -- starves the SM scheduler of warps to swap when memory-stalled.
    -- The smem input-sharing benefit only pays off at outDim ≥ ~10000.
    -- The 4-row kernel is kept for opt-in testing / future tuning at
    -- larger outDim shapes.
    let allow4Row := (match ← IO.getEnv "HESPER_Q4K_4ROW" with
                      | some "1" => true
                      | _        => false)
    let use4Row := allow4Row && layer.config.outDim % 4 == 0
    if use4Row then
      GPUBackend.executeWithConfigCached ctx
        (fusedQ4KMLinearDP4A4RowKernel layer.config)
        [("weights", layer.weightBuf), ("input_q8", q8Buf), ("output", outputBuf)]
        { numWorkgroups := (layer.config.outDim / 4, 1, 1)
          workgroupSize := { x := 128, y := 1, z := 1 }
          extensions := ["subgroups"] }
        (hash ("q4k-dp4a-matmul-4row", layer.config.inDim, layer.config.outDim))
        layer.dp4aMatmulPrepared
    else if use4Warp then
      GPUBackend.executeWithConfigCached ctx
        (fusedQ4KMLinearDP4A4WarpKernel layer.config)
        [("weights", layer.weightBuf), ("input_q8", q8Buf), ("output", outputBuf)]
        { numWorkgroups := (layer.config.outDim, 1, 1)
          workgroupSize := { x := 128, y := 1, z := 1 } }
        (hash ("q4k-dp4a-matmul-4warp", layer.config.inDim, layer.config.outDim))
        layer.dp4aMatmulPrepared
    else if layer.config.outDim % 2 == 0 && layer.config.outDim ≤ 5120 then
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
    -- Kernel-variant gate.  Default is "1row" after A/B (docs 45-47):
    -- ncu showed the 4-row kernel suffers a 50% tail-wave penalty on
    -- the RTX 4070 Ti (grid 640 = 480 full + 160 partial wave); the
    -- 1-row kernel (grid = outDim = 2560, 5× more blocks, negligible
    -- tail) measured 1.322 → 1.246 ms/dec for ffn_down on the
    -- canonical 10-decode run.  2-row was actually worse than either
    -- (1.401 ms/dec) — smem sharing of Q8_1 input did not pay for the
    -- occupancy hit at 4070 Ti's SM count.
    -- Default: 4-warp variant matching llama.cpp's
    -- `mul_mat_vec_q<Q6_K, nwarps=4>` shape exactly — **per-lane** cross-
    -- warp partials in smem (96 floats), warp 0 lanes accumulate own
    -- partial then warp-reduce.  Same f32 summation order as the
    -- 1-warp kernel, so no accumulator drift on short prompts.
    -- +14% TPS at graphs ON (60.18 → 68.66 TPS).
    -- Override with HESPER_Q6K_KERNEL={1row,2row,4row} for regression.
    let q6kVariant := (← IO.getEnv "HESPER_Q6K_KERNEL").getD "4warp"
    let force1Row  := q6kVariant == "1row"
    let force2Row  := q6kVariant == "2row"
    let force4Warp := q6kVariant == "4warp"
    if force4Warp then
      GPUBackend.executeWithConfigCached ctx
        (fusedQ6KLinearDP4A4WarpKernel layer.config.inDim layer.config.outDim)
        [("weights", layer.weightBuf), ("input_q8", q8Buf), ("output", outputBuf)]
        { numWorkgroups := (layer.config.outDim, 1, 1)
          workgroupSize := { x := 32, y := 4, z := 1 }
          -- preHash is keyed on (funcName, stmts.length, buffer-count, wg).
          -- The ShaderM AST embeds `inDim`/`outDim` as literals but they
          -- don't change `stmts.length`, so two distinct dispatches (e.g.
          -- ffn_down at outDim=2560 and lm_head at outDim=262144) collide
          -- in the cache and the second hits the first's compiled cubin.
          -- Distinguish via funcName.
          funcName := s!"q6k_dp4a_4warp_{layer.config.inDim}_{layer.config.outDim}" }
        (hash ("q6k-dp4a-matmul-4warp", layer.config.inDim, layer.config.outDim))
        layer.dp4aMatmulPrepared
    else if force1Row || !(layer.config.outDim % 2 == 0) then
      GPUBackend.executeWithConfigCached ctx
        (fusedQ6KLinearDP4AKernel layer.config.inDim layer.config.outDim)
        [("weights", layer.weightBuf), ("input_q8", q8Buf), ("output", outputBuf)]
        { numWorkgroups := (layer.config.outDim, 1, 1), workgroupSize := { x := 32, y := 1, z := 1 } }
        (hash ("q6k-dp4a-matmul", layer.config.inDim, layer.config.outDim))
        layer.dp4aMatmulPrepared
    else if !force2Row && layer.config.outDim % 4 == 0 then
      GPUBackend.executeWithConfigCached ctx
        (fusedQ6KLinearDP4A4RowKernel layer.config.inDim layer.config.outDim)
        [("weights", layer.weightBuf), ("input_q8", q8Buf), ("output", outputBuf)]
        { numWorkgroups := (layer.config.outDim / 4, 1, 1)
          workgroupSize := { x := 128, y := 1, z := 1 } }
        (hash ("q6k-dp4a-matmul-4row", layer.config.inDim, layer.config.outDim))
        layer.dp4aMatmulPrepared
    else  -- outDim % 2 == 0
      GPUBackend.executeWithConfigCached ctx
        (fusedQ6KLinearDP4A2RowKernel layer.config.inDim layer.config.outDim)
        [("weights", layer.weightBuf), ("input_q8", q8Buf), ("output", outputBuf)]
        { numWorkgroups := (layer.config.outDim / 2, 1, 1)
          workgroupSize := { x := 64, y := 1, z := 1 } }
        (hash ("q6k-dp4a-matmul-2row", layer.config.inDim, layer.config.outDim))
        layer.dp4aMatmulPrepared

  if profiling then
    let endNs ← IO.monoNanosNow
    let delta := (endNs - startNs).toUInt64
    totalNanosRef.modify (· + delta)
    callCountRef.modify (· + 1)
    perShapeAdd layer.config.inDim layer.config.outDim delta

/-- Batched Q4_K forward: quantize + matmul for `seqLen` input columns.
    Input: `[inDim * seqLen]` f32 (column-major: `input[col * inDim + i]`).
    Output: `[outDim * seqLen]` f32 (column-major: `output[col * outDim + row]`).
    For seqLen=1, falls through to single-token `forwardDP4A`. -/
def forwardBatchDP4A [GPUBackend β] (ctx : β)
    (layer : LinearLayer (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (inputBuf outputBuf : GPUBackend.Buf β)
    (seqLen : Nat) : IO Unit := do
  if seqLen <= 1 then
    forwardDP4A ctx layer inputBuf outputBuf
    return

  if layer.quantFormat == .Q4_K then
    -- Q4_K batch path: 2 dispatches (batch quantize + batch matmul)
    let nQ8Blocks := layer.config.inDim / 32
    let q8BufBytes : USize := (nQ8Blocks * 9 * seqLen * 4).toUSize
    let q8Buf ← GPUBackend.allocBuffer ctx q8BufBytes

    GPUBackend.executeWithConfigCached ctx
      (quantizeQ8_1BatchKernel layer.config.inDim seqLen)
      [("input", inputBuf), ("output", q8Buf)]
      { numWorkgroups := (nQ8Blocks, seqLen, 1), workgroupSize := { x := 32, y := 1, z := 1 } }
      (hash ("q8_1-quantize-batch", layer.config.inDim, seqLen))
      layer.dp4aBatchQuantizePrepared

    -- Same MMQ2-default selection as forwardBatchDP4A_fromQ8 below.
    let mmq2Off := (← IO.getEnv "HESPER_PREFILL_MMQ2_OFF").isSome
    let mmq5Forced := (← IO.getEnv "HESPER_PREFILL_MMQ5").isSome
    -- MMQ5 (mmq_y=64, mmq_x=32, half-tile rev. 2026-05-02) wins at
    -- seqLen ≥ 32 where the full tile is well-utilized. For shorter
    -- prompts, MMQ2 (mmq_y=32, mmq_x=8) is faster.
    let useMMQ5 := mmq5Forced ∨ seqLen ≥ 32
    let useMMQ2Default := !mmq2Off && !useMMQ5
    -- seqLen does NOT have to be divisible by 8: the MMQ kernels already
    -- mask out-of-range j-columns (j_in check). The grid rounds up via
    -- ceil(seqLen/N) and OOB threads no-op. seqLen >= 2 means batched.
    if useMMQ5 && layer.config.inDim % 256 == 0
       && layer.config.outDim % 64 == 0 && seqLen >= 2 then
      let nTileCols := (seqLen + 31) / 32
      GPUBackend.executeWithConfigCached ctx
        (q4kMatmulBatchMMQ5Kernel layer.config seqLen)
        [("weights", layer.weightBuf), ("input_q8", q8Buf), ("output", outputBuf)]
        { numWorkgroups := (layer.config.outDim / 64, nTileCols, 1),
          workgroupSize := { x := 256, y := 1, z := 1 } }
        (hash ("q4k-batch-matmul-mmq5", layer.config.inDim, layer.config.outDim, seqLen))
        layer.dp4aBatchMatmulPrepared
    else if useMMQ2Default && layer.config.inDim % 256 == 0
       && layer.config.outDim % 32 == 0 && seqLen >= 2 then
      let nTileCols := (seqLen + 7) / 8
      GPUBackend.executeWithConfigCached ctx
        (q4kMatmulBatchMMQ2Kernel layer.config seqLen)
        [("weights", layer.weightBuf), ("input_q8", q8Buf), ("output", outputBuf)]
        { numWorkgroups := (layer.config.outDim / 32, nTileCols, 1),
          workgroupSize := { x := 256, y := 1, z := 1 } }
        (hash ("q4k-batch-matmul-mmq2", layer.config.inDim, layer.config.outDim, seqLen))
        layer.dp4aBatchMatmulPrepared
    else
      -- 1-warp baseline fallback (HESPER_PREFILL_MMQ2_OFF=1 or shape doesn't match).
      GPUBackend.executeWithConfigCached ctx
        (q4kMatmulBatchKernel layer.config seqLen)
        [("weights", layer.weightBuf), ("input_q8", q8Buf), ("output", outputBuf)]
        { numWorkgroups := (layer.config.outDim, seqLen, 1), workgroupSize := { x := 32, y := 1, z := 1 } }
        (hash ("q4k-batch-matmul", layer.config.inDim, layer.config.outDim, seqLen))
        layer.dp4aBatchMatmulPrepared

    GPUBackend.freeBuffer ctx q8Buf
  else if layer.quantFormat == .Q6_K && layer.config.inDim % 256 == 0 then
    -- Q6_K batched path: 2 dispatches (batch quantize + batch matmul).
    -- Replaces the per-column loop that fired the matmul `seqLen × n_layers`
    -- times during prefill.
    let nQ8Blocks := layer.config.inDim / 32
    let q8BufBytes : USize := (nQ8Blocks * 9 * seqLen * 4).toUSize
    let q8Buf ← GPUBackend.allocBuffer ctx q8BufBytes

    GPUBackend.executeWithConfigCached ctx
      (quantizeQ8_1BatchKernel layer.config.inDim seqLen)
      [("input", inputBuf), ("output", q8Buf)]
      { numWorkgroups := (nQ8Blocks, seqLen, 1), workgroupSize := { x := 32, y := 1, z := 1 } }
      (hash ("q8_1-quantize-batch", layer.config.inDim, seqLen))
      layer.dp4aBatchQuantizePrepared

    let q6k4WarpOff := (← IO.getEnv "HESPER_PREFILL_Q6K_4WARP_OFF").isSome
    if !q6k4WarpOff then
      -- 4-warp coop K (mirrors fusedQ6KLinearDP4A4WarpKernel). Each WG = 128
      -- threads, 4 warps share K. ~2× faster than 1-warp baseline.
      GPUBackend.executeWithConfigCached ctx
        (q6kMatmulBatch4WarpKernel layer.config.inDim layer.config.outDim seqLen)
        [("weights", layer.weightBuf), ("input_q8", q8Buf), ("output", outputBuf)]
        { numWorkgroups := (layer.config.outDim, seqLen, 1),
          workgroupSize := { x := 32, y := 4, z := 1 } }
        (hash ("q6k-batch-matmul-4warp", layer.config.inDim, layer.config.outDim, seqLen))
        layer.dp4aBatchMatmulPrepared
    else
      -- 1-warp baseline (HESPER_PREFILL_Q6K_4WARP_OFF=1 for diagnostics).
      GPUBackend.executeWithConfigCached ctx
        (q6kMatmulBatchKernel layer.config.inDim layer.config.outDim seqLen)
        [("weights", layer.weightBuf), ("input_q8", q8Buf), ("output", outputBuf)]
        { numWorkgroups := (layer.config.outDim, seqLen, 1),
          workgroupSize := { x := 32, y := 1, z := 1 } }
        (hash ("q6k-batch-matmul", layer.config.inDim, layer.config.outDim, seqLen))
        layer.dp4aBatchMatmulPrepared

    GPUBackend.freeBuffer ctx q8Buf
  else
    -- Other formats: per-column loop with GPU-side column extract/insert.
    let inDim := layer.config.inDim
    let outDim := layer.config.outDim
    let inColBuf ← GPUBackend.allocBuffer ctx (inDim * 4).toUSize
    let outColBuf ← GPUBackend.allocBuffer ctx (outDim * 4).toUSize
    for col in [0:seqLen] do
      -- GPU-side column extract: inColBuf[i] = inputBuf[col * inDim + i]
      GPUBackend.executeWithConfig ctx
        (do let _src ← ShaderM.declareReadOnlyBuffer "src" (.array (.scalar .f32) (inDim * seqLen))
            let _dst ← ShaderM.declareOutputBuffer "dst" (.array (.scalar .f32) inDim)
            let lid ← ShaderM.localId; let wid ← ShaderM.workgroupId
            let i := Exp.add (Exp.mul (Exp.vec3X wid) (Exp.litU32 256)) (Exp.vec3X lid)
            ShaderM.if_ (Exp.lt i (Exp.litU32 inDim)) (do
              let srcIdx := Exp.add (Exp.litU32 (col * inDim)) i
              let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := inDim * seqLen) "src" srcIdx
              ShaderM.writeBuffer (ty := .scalar .f32) "dst" i v) (pure ()))
        [("src", inputBuf), ("dst", inColBuf)]
        (.dispatch1D inDim 256)
      forwardDP4A ctx layer inColBuf outColBuf
      -- GPU-side column insert: outputBuf[col * outDim + i] = outColBuf[i]
      GPUBackend.executeWithConfig ctx
        (do let _src ← ShaderM.declareReadOnlyBuffer "src" (.array (.scalar .f32) outDim)
            let _dst ← ShaderM.declareOutputBuffer "dst" (.array (.scalar .f32) (outDim * seqLen))
            let lid ← ShaderM.localId; let wid ← ShaderM.workgroupId
            let i := Exp.add (Exp.mul (Exp.vec3X wid) (Exp.litU32 256)) (Exp.vec3X lid)
            ShaderM.if_ (Exp.lt i (Exp.litU32 outDim)) (do
              let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := outDim) "src" i
              let dstIdx := Exp.add (Exp.litU32 (col * outDim)) i
              ShaderM.writeBuffer (ty := .scalar .f32) "dst" dstIdx v) (pure ()))
        [("src", outColBuf), ("dst", outputBuf)]
        (.dispatch1D outDim 256)
    GPUBackend.freeBuffer ctx inColBuf
    GPUBackend.freeBuffer ctx outColBuf

/-- Batched Q4_K matmul from pre-quantized Q8_1 input (skip quantize step).
    Used when the caller has already produced Q8_1 via fusedRMSNormQ8_1Kernel
    to guarantee numerical parity with the fused decode path.
    `q8Buf` layout: `[nQ8Blocks * 9 * seqLen]` u32, column-major. -/
def forwardBatchDP4A_fromQ8 [GPUBackend β] (ctx : β)
    (layer : LinearLayer (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (q8Buf outputBuf : GPUBackend.Buf β)
    (seqLen : Nat)
    (refOverride : Option (IO.Ref (Option (GPUBackend.CachedDispatch β))) := none)
    : IO Unit := do
  if layer.quantFormat != .Q4_K then
    throw (IO.userError s!"forwardBatchDP4A_fromQ8: only Q4_K, got {repr layer.quantFormat}")
  -- Cache the (q4kMatmulBatchKernel, seqLen) dispatch so the 2nd+
  -- forwardPrefillBatch call replays it instead of re-running generatePTX
  -- + cuModuleLoad (~350µs × 126 calls = ~44ms/token of pure JIT overhead
  -- that was eating half the decode budget).
  --
  -- `layer.dp4aBatchMatmulPrepared` is seqLen-agnostic, so sharing it
  -- between prefill (seqLen=N) and unified-decode (seqLen=1) would make
  -- the 2nd path replay a wrong-shape dispatch.  Caller passes a
  -- seqLen-specific `refOverride` in that case (typically a kcr-backed
  -- ref keyed on seqLen); absent an override we fall back to the shared
  -- layer ref.
  let ref := refOverride.getD layer.dp4aBatchMatmulPrepared
  -- MMQ2 (smem-staged X tile) is the DEFAULT for shapes that meet
  -- preconditions: outDim % 32 == 0, inDim % 256 == 0, seqLen >= 2.
  -- Set HESPER_PREFILL_MMQ2_OFF=1 to fall through to the 1-warp baseline.
  -- Set HESPER_PREFILL_MMQ5=1 to use the full llama-shape kernel
  -- (mmq_y=128, mmq_x=64, X+Y both smem-staged) for shapes where
  -- outDim % 128 == 0 and seqLen >= 2.
  let mmq2Off := (← IO.getEnv "HESPER_PREFILL_MMQ2_OFF").isSome
  let mmq5Forced := (← IO.getEnv "HESPER_PREFILL_MMQ5").isSome
  -- See comment at forwardBatchDP4A — MMQ5 wins only at seqLen >= 32.
  let useMMQ5 := mmq5Forced ∨ seqLen ≥ 32
  let useMMQ2Default := !mmq2Off && !useMMQ5
  if useMMQ5 && layer.config.inDim % 256 == 0
     && layer.config.outDim % 64 == 0 && seqLen >= 2 then
    let nTileCols := (seqLen + 31) / 32
    let cacheKey : UInt64 := hash ("q4k-batch-matmul-q8-mmq5",
      layer.config.inDim, layer.config.outDim, seqLen)
    GPUBackend.executeWithConfigCached ctx
      (q4kMatmulBatchMMQ5Kernel layer.config seqLen)
      [("weights", layer.weightBuf), ("input_q8", q8Buf), ("output", outputBuf)]
      { numWorkgroups := (layer.config.outDim / 64, nTileCols, 1),
        workgroupSize := { x := 256, y := 1, z := 1 } }
      cacheKey ref
  else if useMMQ2Default && layer.config.inDim % 256 == 0
     && layer.config.outDim % 32 == 0 && seqLen >= 2 then
    let nTileCols := (seqLen + 7) / 8
    let cacheKey : UInt64 := hash ("q4k-batch-matmul-q8-mmq2",
      layer.config.inDim, layer.config.outDim, seqLen)
    GPUBackend.executeWithConfigCached ctx
      (q4kMatmulBatchMMQ2Kernel layer.config seqLen)
      [("weights", layer.weightBuf), ("input_q8", q8Buf), ("output", outputBuf)]
      { numWorkgroups := (layer.config.outDim / 32, nTileCols, 1),
        workgroupSize := { x := 256, y := 1, z := 1 } }
      cacheKey ref
  else
    let cacheKey : UInt64 := hash ("q4k-batch-matmul-q8",
      layer.config.inDim, layer.config.outDim, seqLen)
    GPUBackend.executeWithConfigCached ctx
      (q4kMatmulBatchKernel layer.config seqLen)
      [("weights", layer.weightBuf), ("input_q8", q8Buf), ("output", outputBuf)]
      { numWorkgroups := (layer.config.outDim, seqLen, 1), workgroupSize := { x := 32, y := 1, z := 1 } }
      cacheKey ref

/-- Read one byte (0..255) at a dynamic byte offset within a u32-array buffer. -/
private def q8ReadByte (bufName : String) (n : Nat) (byteOff : Exp (.scalar .u32)) :
    ShaderM (Exp (.scalar .u32)) := do
  let u32idx := Exp.shiftRight byteOff (Exp.litU32 2)
  let shift := Exp.mul (Exp.bitAnd byteOff (Exp.litU32 3)) (Exp.litU32 8)
  let word ← ShaderM.readBuffer (ty := .scalar .u32) (n := n) bufName u32idx
  pure (Exp.bitAnd (Exp.shiftRight word shift) (Exp.litU32 255))

/-- Q8_0 fused matvec: `out[r] = Σ_c W[r][c]·x[c]`, dequantizing the packed
    Q8_0 weights INLINE (f16 scale + 32×int8 per 32-elem block, 34 bytes) —
    weights stay quantized in VRAM (no F32 expansion → preserves the
    memory-bandwidth win).  One workgroup per output row, strided blocks,
    tree reduction.  Mirrors `fusedQ4KMLinearKernel`. -/
def fusedQ8_0LinearKernel (config : Config) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid
  let blocksPerRow := config.inDim / 32
  let totalBytes := config.outDim * blocksPerRow * 34
  let totalWeightU32 := (totalBytes + 3) / 4
  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareReadOnlyBuffer "input" (.array (.scalar .f32) config.inDim)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) config.outDim)
  ShaderM.sharedNamed "shared_partial" (.array (.scalar .f32) workgroupSize)
  let inBounds := Exp.lt outIdx (Exp.litU32 config.outDim)
  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"
  let rowBaseBytes := Exp.mul outIdx (Exp.litU32 (blocksPerRow * 34))
  ShaderM.loop tid (Exp.litU32 blocksPerRow) (Exp.litU32 workgroupSize) fun blk => do
    let blockByteBase := Exp.add rowBaseBytes (Exp.mul blk (Exp.litU32 34))
    let elemBase := Exp.mul blk (Exp.litU32 32)
    -- f16 scale (bytes 0..1 of the block)
    let loByte ← q8ReadByte "weights" totalWeightU32 blockByteBase
    let hiByte ← q8ReadByte "weights" totalWeightU32 (Exp.add blockByteBase (Exp.litU32 1))
    let f16bits := Exp.add loByte (Exp.mul hiByte (Exp.litU32 256))
    let d := Exp.vecX (Exp.unpack2x16float f16bits)
    -- 32 int8 quants (bytes 2..33), dequant inline: w = d * signed(q)
    for i in [0:32] do
      let qb ← q8ReadByte "weights" totalWeightU32 (Exp.add blockByteBase (Exp.litU32 (2 + i)))
      let sign := Exp.shiftRight qb (Exp.litU32 7)
      let qSigned := Exp.sub (Exp.toF32 qb) (Exp.mul (Exp.litF32 256.0) (Exp.toF32 sign))
      let w := Exp.mul d qSigned
      let inVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.inDim) "input" (Exp.add elemBase (Exp.litU32 i))
      ShaderM.assign "acc" (Exp.add acc (Exp.mul w inVal))
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
  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    let totalSum ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_partial" (Exp.litU32 0)
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx totalSum
  ) (pure ())

/-- Batched Q8_0 matmul: input `[M, inDim]` × dequant(weights `[outDim, inDim]`) → `[M, outDim]`.
    Dispatch `(outDim, M, 1)`; workgroup `(outIdx, row)` → `output[row*outDim+outIdx]`. Metal-native. -/
def fusedQ8_0BatchKernel (config : Config) (M : Nat) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let row := Exp.vec3Y wid
  let tid := Exp.vec3X lid
  let blocksPerRow := config.inDim / 32
  let totalBytes := config.outDim * blocksPerRow * 34
  let totalWeightU32 := (totalBytes + 3) / 4
  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareReadOnlyBuffer "input" (.array (.scalar .f32) (M * config.inDim))
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) (M * config.outDim))
  ShaderM.sharedNamed "shared_partial" (.array (.scalar .f32) workgroupSize)
  let inBounds := Exp.lt outIdx (Exp.litU32 config.outDim)
  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"
  let inRowBase := Exp.mul row (Exp.litU32 config.inDim)
  let rowBaseBytes := Exp.mul outIdx (Exp.litU32 (blocksPerRow * 34))
  ShaderM.loop tid (Exp.litU32 blocksPerRow) (Exp.litU32 workgroupSize) fun blk => do
    let blockByteBase := Exp.add rowBaseBytes (Exp.mul blk (Exp.litU32 34))
    let elemBase := Exp.mul blk (Exp.litU32 32)
    let loByte ← q8ReadByte "weights" totalWeightU32 blockByteBase
    let hiByte ← q8ReadByte "weights" totalWeightU32 (Exp.add blockByteBase (Exp.litU32 1))
    let f16bits := Exp.add loByte (Exp.mul hiByte (Exp.litU32 256))
    let d := Exp.vecX (Exp.unpack2x16float f16bits)
    for i in [0:32] do
      let qb ← q8ReadByte "weights" totalWeightU32 (Exp.add blockByteBase (Exp.litU32 (2 + i)))
      let sign := Exp.shiftRight qb (Exp.litU32 7)
      let qSigned := Exp.sub (Exp.toF32 qb) (Exp.mul (Exp.litF32 256.0) (Exp.toF32 sign))
      let w := Exp.mul d qSigned
      let inVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := M * config.inDim) "input" (Exp.add inRowBase (Exp.add elemBase (Exp.litU32 i)))
      ShaderM.assign "acc" (Exp.add acc (Exp.mul w inVal))
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
  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    let totalSum ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_partial" (Exp.litU32 0)
    ShaderM.writeBuffer (ty := .scalar .f32) "output" (Exp.add (Exp.mul row (Exp.litU32 config.outDim)) outIdx) totalSum
  ) (pure ())

/-- Expert-indexed Q8_0 matvec for an MoE expert tensor `[inDim, outDim, nExpert]`
    (DiffusionGemma ffn_down_exps).  Expert `e`'s weight = a Q8_0 `[inDim,outDim]`
    at byte base `e * outDim * (inDim/32) * 34`.  Validated-base Q8_0 math +
    expert offset; packed weights stay in VRAM. -/
def fusedQ8_0ExpertKernel (config : Config) (nExpert : Nat) (paramsLen : Nat := 1) (slot : Nat := 0) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid
  let blocksPerRow := config.inDim / 32
  let perExpertBytes := config.outDim * blocksPerRow * 34
  let totalBytes := nExpert * perExpertBytes
  let totalWeightU32 := (totalBytes + 3) / 4
  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareReadOnlyBuffer "input" (.array (.scalar .f32) config.inDim)
  let _params ← ShaderM.declareReadOnlyBuffer "params" (.array (.scalar .u32) paramsLen)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) config.outDim)
  ShaderM.sharedNamed "shared_partial" (.array (.scalar .f32) workgroupSize)
  let inBounds := Exp.lt outIdx (Exp.litU32 config.outDim)
  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"
  -- expert index read dynamically (one shader for all experts)
  let expertIdx ← ShaderM.readBuffer (ty := .scalar .u32) (n := paramsLen) "params" (Exp.litU32 slot)
  let rowBaseBytes := Exp.add (Exp.mul expertIdx (Exp.litU32 perExpertBytes)) (Exp.mul outIdx (Exp.litU32 (blocksPerRow * 34)))
  ShaderM.loop tid (Exp.litU32 blocksPerRow) (Exp.litU32 workgroupSize) fun blk => do
    let blockByteBase := Exp.add rowBaseBytes (Exp.mul blk (Exp.litU32 34))
    let elemBase := Exp.mul blk (Exp.litU32 32)
    let loByte ← q8ReadByte "weights" totalWeightU32 blockByteBase
    let hiByte ← q8ReadByte "weights" totalWeightU32 (Exp.add blockByteBase (Exp.litU32 1))
    let f16bits := Exp.add loByte (Exp.mul hiByte (Exp.litU32 256))
    let d := Exp.vecX (Exp.unpack2x16float f16bits)
    for i in [0:32] do
      let qb ← q8ReadByte "weights" totalWeightU32 (Exp.add blockByteBase (Exp.litU32 (2 + i)))
      let sign := Exp.shiftRight qb (Exp.litU32 7)
      let qSigned := Exp.sub (Exp.toF32 qb) (Exp.mul (Exp.litF32 256.0) (Exp.toF32 sign))
      let w := Exp.mul d qSigned
      let inVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.inDim) "input" (Exp.add elemBase (Exp.litU32 i))
      ShaderM.assign "acc" (Exp.add acc (Exp.mul w inVal))
  ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_partial" tid acc
  ShaderM.barrier
  let mut stride := workgroupSize / 2
  while stride > 0 do
    ShaderM.if_ (Exp.lt tid (Exp.litU32 stride)) (do
      let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_partial" tid
      let bb ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_partial" (Exp.add tid (Exp.litU32 stride))
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_partial" tid (Exp.add a bb)
    ) (pure ())
    ShaderM.barrier
    stride := stride / 2
  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    let totalSum ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_partial" (Exp.litU32 0)
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx totalSum
  ) (pure ())


/-- Batched expert matmul: row dim + per-row expertIdx from idxs[row*nUsed+slot]. -/
def fusedQ8_0BatchExpertKernel (config : Config) (nExpert N nUsed slot : Nat) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let row := Exp.vec3Y wid
  let tid := Exp.vec3X lid
  let blocksPerRow := config.inDim / 32
  let perExpertBytes := config.outDim * blocksPerRow * 34
  let totalBytes := nExpert * perExpertBytes
  let totalWeightU32 := (totalBytes + 3) / 4
  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareReadOnlyBuffer "input" (.array (.scalar .f32) (N * config.inDim))
  let _idxs ← ShaderM.declareReadOnlyBuffer "idxs" (.array (.scalar .u32) (N * nUsed))
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) (N * config.outDim))
  ShaderM.sharedNamed "shared_partial" (.array (.scalar .f32) workgroupSize)
  let inBounds := Exp.lt outIdx (Exp.litU32 config.outDim)
  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"
  -- expert index read dynamically (one shader for all experts)
  let expertIdx ← ShaderM.readBuffer (ty := .scalar .u32) (n := N * nUsed) "idxs" (Exp.add (Exp.mul row (Exp.litU32 nUsed)) (Exp.litU32 slot))
  let rowBaseBytes := Exp.add (Exp.mul expertIdx (Exp.litU32 perExpertBytes)) (Exp.mul outIdx (Exp.litU32 (blocksPerRow * 34)))
  ShaderM.loop tid (Exp.litU32 blocksPerRow) (Exp.litU32 workgroupSize) fun blk => do
    let blockByteBase := Exp.add rowBaseBytes (Exp.mul blk (Exp.litU32 34))
    let elemBase := Exp.mul blk (Exp.litU32 32)
    let loByte ← q8ReadByte "weights" totalWeightU32 blockByteBase
    let hiByte ← q8ReadByte "weights" totalWeightU32 (Exp.add blockByteBase (Exp.litU32 1))
    let f16bits := Exp.add loByte (Exp.mul hiByte (Exp.litU32 256))
    let d := Exp.vecX (Exp.unpack2x16float f16bits)
    for i in [0:32] do
      let qb ← q8ReadByte "weights" totalWeightU32 (Exp.add blockByteBase (Exp.litU32 (2 + i)))
      let sign := Exp.shiftRight qb (Exp.litU32 7)
      let qSigned := Exp.sub (Exp.toF32 qb) (Exp.mul (Exp.litF32 256.0) (Exp.toF32 sign))
      let w := Exp.mul d qSigned
      let inVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := N * config.inDim) "input" (Exp.add (Exp.mul row (Exp.litU32 config.inDim)) (Exp.add elemBase (Exp.litU32 i)))
      ShaderM.assign "acc" (Exp.add acc (Exp.mul w inVal))
  ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_partial" tid acc
  ShaderM.barrier
  let mut stride := workgroupSize / 2
  while stride > 0 do
    ShaderM.if_ (Exp.lt tid (Exp.litU32 stride)) (do
      let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_partial" tid
      let bb ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_partial" (Exp.add tid (Exp.litU32 stride))
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_partial" tid (Exp.add a bb)
    ) (pure ())
    ShaderM.barrier
    stride := stride / 2
  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    let totalSum ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_partial" (Exp.litU32 0)
    ShaderM.writeBuffer (ty := .scalar .f32) "output" (Exp.add (Exp.mul row (Exp.litU32 config.outDim)) outIdx) totalSum
  ) (pure ())

/-- Dequant 32 Q5_0 weights of one block into the accumulator.
    Q5_0 block = 22 bytes: d(f16, bytes 0-1) + qh(u32, bytes 2-5) + qs(16 bytes 6-21).
    weight k: nibble = (k<16 ? qs[k]&0xF : qs[k-16]>>4); hbit=(qh>>k)&1; x=((nibble|hbit<<4)-16)*d. -/
private def q50AccumBlock (bufName : String) (totalWeightU32 : Nat) (blockByteBase : Exp (.scalar .u32))
    (inBufName : String) (inN : Nat) (inElemBase : Exp (.scalar .u32)) (accName : String) : ShaderM Unit := do
  let acc : Exp (.scalar .f32) := Exp.var accName
  let loByte ← q8ReadByte bufName totalWeightU32 blockByteBase
  let hiByte ← q8ReadByte bufName totalWeightU32 (Exp.add blockByteBase (Exp.litU32 1))
  let d := Exp.vecX (Exp.unpack2x16float (Exp.add loByte (Exp.mul hiByte (Exp.litU32 256))))
  let qh0 ← q8ReadByte bufName totalWeightU32 (Exp.add blockByteBase (Exp.litU32 2))
  let qh1 ← q8ReadByte bufName totalWeightU32 (Exp.add blockByteBase (Exp.litU32 3))
  let qh2 ← q8ReadByte bufName totalWeightU32 (Exp.add blockByteBase (Exp.litU32 4))
  let qh3 ← q8ReadByte bufName totalWeightU32 (Exp.add blockByteBase (Exp.litU32 5))
  let qhName ← ShaderM.var (.scalar .u32)
    (Exp.add qh0 (Exp.add (Exp.mul qh1 (Exp.litU32 256))
      (Exp.add (Exp.mul qh2 (Exp.litU32 65536)) (Exp.mul qh3 (Exp.litU32 16777216)))))
  let qh : Exp (.scalar .u32) := Exp.var qhName
  for i in [0:32] do
    let qsByteIdx := if i < 16 then i else i - 16
    let qb ← q8ReadByte bufName totalWeightU32 (Exp.add blockByteBase (Exp.litU32 (6 + qsByteIdx)))
    let nibble := if i < 16 then Exp.bitAnd qb (Exp.litU32 15)
                  else Exp.bitAnd (Exp.shiftRight qb (Exp.litU32 4)) (Exp.litU32 15)
    let hbit := Exp.bitAnd (Exp.shiftRight qh (Exp.litU32 i)) (Exp.litU32 1)
    let q5 := Exp.add nibble (Exp.mul hbit (Exp.litU32 16))
    let w := Exp.mul d (Exp.sub (Exp.toF32 q5) (Exp.litF32 16.0))
    let inVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := inN) inBufName (Exp.add inElemBase (Exp.litU32 i))
    ShaderM.assign accName (Exp.add acc (Exp.mul w inVal))

/-- Q5_0 batched matvec (dense), 22-byte blocks. Mirrors `fusedQ8_0BatchKernel`. -/
def fusedQ5_0BatchKernel (config : Config) (M : Nat) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let row := Exp.vec3Y wid
  let tid := Exp.vec3X lid
  let blocksPerRow := config.inDim / 32
  let totalBytes := config.outDim * blocksPerRow * 22
  let totalWeightU32 := (totalBytes + 3) / 4
  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareReadOnlyBuffer "input" (.array (.scalar .f32) (M * config.inDim))
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) (M * config.outDim))
  ShaderM.sharedNamed "shared_partial" (.array (.scalar .f32) workgroupSize)
  let inBounds := Exp.lt outIdx (Exp.litU32 config.outDim)
  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"
  let inRowBase := Exp.mul row (Exp.litU32 config.inDim)
  let rowBaseBytes := Exp.mul outIdx (Exp.litU32 (blocksPerRow * 22))
  ShaderM.loop tid (Exp.litU32 blocksPerRow) (Exp.litU32 workgroupSize) fun blk => do
    let blockByteBase := Exp.add rowBaseBytes (Exp.mul blk (Exp.litU32 22))
    let elemBase := Exp.add inRowBase (Exp.mul blk (Exp.litU32 32))
    q50AccumBlock "weights" totalWeightU32 blockByteBase "input" (M * config.inDim) elemBase "acc"
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
  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    let totalSum ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_partial" (Exp.litU32 0)
    ShaderM.writeBuffer (ty := .scalar .f32) "output" (Exp.add (Exp.mul row (Exp.litU32 config.outDim)) outIdx) totalSum
  ) (pure ())

/-- Q5_0 expert-indexed batched matvec, 22-byte blocks. Mirrors `fusedQ8_0BatchExpertKernel`. -/
def fusedQ5_0BatchExpertKernel (config : Config) (nExpert N nUsed slot : Nat) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let row := Exp.vec3Y wid
  let tid := Exp.vec3X lid
  let blocksPerRow := config.inDim / 32
  let perExpertBytes := config.outDim * blocksPerRow * 22
  let totalBytes := nExpert * perExpertBytes
  let totalWeightU32 := (totalBytes + 3) / 4
  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareReadOnlyBuffer "input" (.array (.scalar .f32) (N * config.inDim))
  let _idxs ← ShaderM.declareReadOnlyBuffer "idxs" (.array (.scalar .u32) (N * nUsed))
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) (N * config.outDim))
  ShaderM.sharedNamed "shared_partial" (.array (.scalar .f32) workgroupSize)
  let inBounds := Exp.lt outIdx (Exp.litU32 config.outDim)
  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"
  let expertIdx ← ShaderM.readBuffer (ty := .scalar .u32) (n := N * nUsed) "idxs" (Exp.add (Exp.mul row (Exp.litU32 nUsed)) (Exp.litU32 slot))
  let rowBaseBytes := Exp.add (Exp.mul expertIdx (Exp.litU32 perExpertBytes)) (Exp.mul outIdx (Exp.litU32 (blocksPerRow * 22)))
  let inRowBase := Exp.mul row (Exp.litU32 config.inDim)
  ShaderM.loop tid (Exp.litU32 blocksPerRow) (Exp.litU32 workgroupSize) fun blk => do
    let blockByteBase := Exp.add rowBaseBytes (Exp.mul blk (Exp.litU32 22))
    let elemBase := Exp.add inRowBase (Exp.mul blk (Exp.litU32 32))
    q50AccumBlock "weights" totalWeightU32 blockByteBase "input" (N * config.inDim) elemBase "acc"
  ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_partial" tid acc
  ShaderM.barrier
  let mut stride := workgroupSize / 2
  while stride > 0 do
    ShaderM.if_ (Exp.lt tid (Exp.litU32 stride)) (do
      let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_partial" tid
      let bb ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_partial" (Exp.add tid (Exp.litU32 stride))
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_partial" tid (Exp.add a bb)
    ) (pure ())
    ShaderM.barrier
    stride := stride / 2
  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    let totalSum ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_partial" (Exp.litU32 0)
    ShaderM.writeBuffer (ty := .scalar .f32) "output" (Exp.add (Exp.mul row (Exp.litU32 config.outDim)) outIdx) totalSum
  ) (pure ())


/-- f32-warp MoE expert down for Q8_0 — drop-in for `fusedQ8_0BatchExpertKernel` but 32-lane /
    subgroupAdd instead of 256-thread tree reduction (the block-parallel kernel uses only
    inDim/32 of 256 threads). Lane `l` handles weight `l` of every block (qs[l]); subgroupAdd
    sums the 32 lanes. Same f32 input/output, just faster. -/
def fusedQ8_0BatchExpertF32WarpKernel (config : Config) (nExpert N nUsed slot : Nat) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let row := Exp.vec3Y wid
  let tid := Exp.vec3X lid
  let blocksPerRow := config.inDim / 32
  let perExpertBytes := config.outDim * blocksPerRow * 34
  let totalWeightU32 := (nExpert * perExpertBytes + 3) / 4
  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareReadOnlyBuffer "input" (.array (.scalar .f32) (N * config.inDim))
  let _idxs ← ShaderM.declareReadOnlyBuffer "idxs" (.array (.scalar .u32) (N * nUsed))
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) (N * config.outDim))
  let inBounds := Exp.lt outIdx (Exp.litU32 config.outDim)
  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"
  let expertIdx ← ShaderM.readBuffer (ty := .scalar .u32) (n := N * nUsed) "idxs" (Exp.add (Exp.mul row (Exp.litU32 nUsed)) (Exp.litU32 slot))
  let rowBaseBytes := Exp.add (Exp.mul expertIdx (Exp.litU32 perExpertBytes)) (Exp.mul outIdx (Exp.litU32 (blocksPerRow * 34)))
  let inRowBase := Exp.mul row (Exp.litU32 config.inDim)
  ShaderM.loop (Exp.litU32 0) (Exp.litU32 blocksPerRow) (Exp.litU32 1) fun blk => do
    let blockByteBase := Exp.add rowBaseBytes (Exp.mul blk (Exp.litU32 34))
    let loByte ← q8ReadByte "weights" totalWeightU32 blockByteBase
    let hiByte ← q8ReadByte "weights" totalWeightU32 (Exp.add blockByteBase (Exp.litU32 1))
    let d := Exp.vecX (Exp.unpack2x16float (Exp.add loByte (Exp.mul hiByte (Exp.litU32 256))))
    let qb ← q8ReadByte "weights" totalWeightU32 (Exp.add blockByteBase (Exp.add (Exp.litU32 2) tid))
    let sign := Exp.shiftRight qb (Exp.litU32 7)
    let qSigned := Exp.sub (Exp.toF32 qb) (Exp.mul (Exp.litF32 256.0) (Exp.toF32 sign))
    let inVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := N * config.inDim) "input" (Exp.add inRowBase (Exp.add (Exp.mul blk (Exp.litU32 32)) tid))
    ShaderM.assign "acc" (Exp.add acc (Exp.mul (Exp.mul d qSigned) inVal))
  ShaderM.varNamed "total" (.scalar .f32) (Exp.subgroupAdd acc)
  let total : Exp (.scalar .f32) := Exp.var "total"
  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    ShaderM.writeBuffer (ty := .scalar .f32) "output" (Exp.add (Exp.mul row (Exp.litU32 config.outDim)) outIdx) total
  ) (pure ())

/-- f32-warp MoE expert down for Q5_0 (22-byte blocks) — drop-in for `fusedQ5_0BatchExpertKernel`.
    Lane `l` handles weight `l`: nibble = (l<16 ? qs[l]&0xF : qs[l-16]>>4), hbit=(qh>>l)&1,
    x=((nibble|hbit<<4)-16)*d; subgroupAdd over the 32 lanes. -/
def fusedQ5_0BatchExpertF32WarpKernel (config : Config) (nExpert N nUsed slot : Nat) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let row := Exp.vec3Y wid
  let tid := Exp.vec3X lid
  let blocksPerRow := config.inDim / 32
  let perExpertBytes := config.outDim * blocksPerRow * 22
  let totalWeightU32 := (nExpert * perExpertBytes + 3) / 4
  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareReadOnlyBuffer "input" (.array (.scalar .f32) (N * config.inDim))
  let _idxs ← ShaderM.declareReadOnlyBuffer "idxs" (.array (.scalar .u32) (N * nUsed))
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) (N * config.outDim))
  let inBounds := Exp.lt outIdx (Exp.litU32 config.outDim)
  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"
  let expertIdx ← ShaderM.readBuffer (ty := .scalar .u32) (n := N * nUsed) "idxs" (Exp.add (Exp.mul row (Exp.litU32 nUsed)) (Exp.litU32 slot))
  let rowBaseBytes := Exp.add (Exp.mul expertIdx (Exp.litU32 perExpertBytes)) (Exp.mul outIdx (Exp.litU32 (blocksPerRow * 22)))
  let inRowBase := Exp.mul row (Exp.litU32 config.inDim)
  ShaderM.loop (Exp.litU32 0) (Exp.litU32 blocksPerRow) (Exp.litU32 1) fun blk => do
    let blockByteBase := Exp.add rowBaseBytes (Exp.mul blk (Exp.litU32 22))
    let loByte ← q8ReadByte "weights" totalWeightU32 blockByteBase
    let hiByte ← q8ReadByte "weights" totalWeightU32 (Exp.add blockByteBase (Exp.litU32 1))
    let d := Exp.vecX (Exp.unpack2x16float (Exp.add loByte (Exp.mul hiByte (Exp.litU32 256))))
    let qh0 ← q8ReadByte "weights" totalWeightU32 (Exp.add blockByteBase (Exp.litU32 2))
    let qh1 ← q8ReadByte "weights" totalWeightU32 (Exp.add blockByteBase (Exp.litU32 3))
    let qh2 ← q8ReadByte "weights" totalWeightU32 (Exp.add blockByteBase (Exp.litU32 4))
    let qh3 ← q8ReadByte "weights" totalWeightU32 (Exp.add blockByteBase (Exp.litU32 5))
    let qh := Exp.add qh0 (Exp.add (Exp.mul qh1 (Exp.litU32 256)) (Exp.add (Exp.mul qh2 (Exp.litU32 65536)) (Exp.mul qh3 (Exp.litU32 16777216))))
    let qsByteIdx := Exp.select (Exp.lt tid (Exp.litU32 16)) tid (Exp.sub tid (Exp.litU32 16))
    let qb ← q8ReadByte "weights" totalWeightU32 (Exp.add blockByteBase (Exp.add (Exp.litU32 6) qsByteIdx))
    let nibble := Exp.select (Exp.lt tid (Exp.litU32 16)) (Exp.bitAnd qb (Exp.litU32 15))
                    (Exp.bitAnd (Exp.shiftRight qb (Exp.litU32 4)) (Exp.litU32 15))
    let hbit := Exp.bitAnd (Exp.shiftRight qh tid) (Exp.litU32 1)
    let q5 := Exp.add nibble (Exp.mul hbit (Exp.litU32 16))
    let w := Exp.mul d (Exp.sub (Exp.toF32 q5) (Exp.litF32 16.0))
    let inVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := N * config.inDim) "input" (Exp.add inRowBase (Exp.add (Exp.mul blk (Exp.litU32 32)) tid))
    ShaderM.assign "acc" (Exp.add acc (Exp.mul w inVal))
  ShaderM.varNamed "total" (.scalar .f32) (Exp.subgroupAdd acc)
  let total : Exp (.scalar .f32) := Exp.var "total"
  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    ShaderM.writeBuffer (ty := .scalar .f32) "output" (Exp.add (Exp.mul row (Exp.litU32 config.outDim)) outIdx) total
  ) (pure ())

/-- Expert-indexed Q4_K matvec for an MoE expert tensor `[inDim, outDim, nExpert]`
    (ne0=inDim quantized, ne1=outDim, ne2=expert).  Expert `e`'s weight is a
    standard Q4_K `[inDim,outDim]` matrix at base offset
    `e * outDim * (inDim/256) * 36` u32.  Identical math to
    `fusedQ4KMLinearKernel` (validated) — only the row base adds the expert
    offset.  `weights` stays packed in VRAM (inline dequant). -/
def fusedQ4KMExpertKernel (config : Config) (nExpert : Nat) (paramsLen : Nat := 1) (slot : Nat := 0) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid
  let blocksPerRow := config.inDim / 256
  let perExpertU32 := config.outDim * blocksPerRow * 36
  let totalWeightU32 := nExpert * perExpertU32
  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareReadOnlyBuffer "input" (.array (.scalar .f32) config.inDim)
  let _params ← ShaderM.declareReadOnlyBuffer "params" (.array (.scalar .u32) paramsLen)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) config.outDim)
  ShaderM.sharedNamed "shared_partial" (.array (.scalar .f32) workgroupSize)
  let inBounds := Exp.lt outIdx (Exp.litU32 config.outDim)
  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"
  -- expert index read dynamically (one shader for all experts → no pipeline explosion)
  let expertIdx ← ShaderM.readBuffer (ty := .scalar .u32) (n := paramsLen) "params" (Exp.litU32 slot)
  let rowBaseU32 := Exp.add (Exp.mul expertIdx (Exp.litU32 perExpertU32)) (Exp.mul outIdx (Exp.litU32 (blocksPerRow * 36)))
  ShaderM.loop tid (Exp.litU32 blocksPerRow) (Exp.litU32 workgroupSize) fun blockLocalIdx => do
    let blockU32Base := Exp.add rowBaseU32 (Exp.mul blockLocalIdx (Exp.litU32 36))
    let elemBase := Exp.mul blockLocalIdx (Exp.litU32 256)
    let dmU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" blockU32Base
    let d := Exp.vecX (Exp.unpack2x16float dmU32)
    let dmin := Exp.vecY (Exp.unpack2x16float dmU32)
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
  ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_partial" tid acc
  ShaderM.barrier
  let mut stride := workgroupSize / 2
  while stride > 0 do
    ShaderM.if_ (Exp.lt tid (Exp.litU32 stride)) (do
      let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_partial" tid
      let bb ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_partial" (Exp.add tid (Exp.litU32 stride))
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_partial" tid (Exp.add a bb)
    ) (pure ())
    ShaderM.barrier
    stride := stride / 2
  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    let totalSum ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_partial" (Exp.litU32 0)
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx totalSum
  ) (pure ())


/-- Batched expert matmul: row dim + per-row expertIdx from idxs[row*nUsed+slot]. -/
def fusedQ4KMBatchExpertKernel (config : Config) (nExpert N nUsed slot : Nat) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let row := Exp.vec3Y wid
  let tid := Exp.vec3X lid
  let blocksPerRow := config.inDim / 256
  let perExpertU32 := config.outDim * blocksPerRow * 36
  let totalWeightU32 := nExpert * perExpertU32
  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareReadOnlyBuffer "input" (.array (.scalar .f32) (N * config.inDim))
  let _idxs ← ShaderM.declareReadOnlyBuffer "idxs" (.array (.scalar .u32) (N * nUsed))
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) (N * config.outDim))
  ShaderM.sharedNamed "shared_partial" (.array (.scalar .f32) workgroupSize)
  let inBounds := Exp.lt outIdx (Exp.litU32 config.outDim)
  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"
  -- expert index read dynamically (one shader for all experts → no pipeline explosion)
  let expertIdx ← ShaderM.readBuffer (ty := .scalar .u32) (n := N * nUsed) "idxs" (Exp.add (Exp.mul row (Exp.litU32 nUsed)) (Exp.litU32 slot))
  let rowBaseU32 := Exp.add (Exp.mul expertIdx (Exp.litU32 perExpertU32)) (Exp.mul outIdx (Exp.litU32 (blocksPerRow * 36)))
  ShaderM.loop tid (Exp.litU32 blocksPerRow) (Exp.litU32 workgroupSize) fun blockLocalIdx => do
    let blockU32Base := Exp.add rowBaseU32 (Exp.mul blockLocalIdx (Exp.litU32 36))
    let elemBase := Exp.mul blockLocalIdx (Exp.litU32 256)
    let dmU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" blockU32Base
    let d := Exp.vecX (Exp.unpack2x16float dmU32)
    let dmin := Exp.vecY (Exp.unpack2x16float dmU32)
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
          let elemIdxLow := Exp.add (Exp.mul row (Exp.litU32 config.inDim)) (Exp.add elemBase (Exp.litU32 (c * 64 + l32 * 4 + b)))
          let elemIdxHigh := Exp.add (Exp.mul row (Exp.litU32 config.inDim)) (Exp.add elemBase (Exp.litU32 (c * 64 + 32 + l32 * 4 + b)))
          let inLow ← ShaderM.readBuffer (ty := .scalar .f32) (n := N * config.inDim) "input" elemIdxLow
          let inHigh ← ShaderM.readBuffer (ty := .scalar .f32) (n := N * config.inDim) "input" elemIdxHigh
          let wLow := Exp.sub (Exp.mul d1 (Exp.toF32 qLow)) m1
          let wHigh := Exp.sub (Exp.mul d2 (Exp.toF32 qHigh)) m2
          ShaderM.assign "acc" (Exp.add acc (Exp.add (Exp.mul wLow inLow) (Exp.mul wHigh inHigh)))
  ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_partial" tid acc
  ShaderM.barrier
  let mut stride := workgroupSize / 2
  while stride > 0 do
    ShaderM.if_ (Exp.lt tid (Exp.litU32 stride)) (do
      let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_partial" tid
      let bb ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_partial" (Exp.add tid (Exp.litU32 stride))
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_partial" tid (Exp.add a bb)
    ) (pure ())
    ShaderM.barrier
    stride := stride / 2
  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    let totalSum ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_partial" (Exp.litU32 0)
    ShaderM.writeBuffer (ty := .scalar .f32) "output" (Exp.add (Exp.mul row (Exp.litU32 config.outDim)) outIdx) totalSum
  ) (pure ())



/-- Q4_K expert matmul via dp4a (int8 dot products), per-position expert.  Mirrors
    `emitQ4KMLinearDP4ABody`'s Q4_K×Q8_1 dot but with (a) the per-position expert offset
    in `rowBaseU32` and (b) the per-position Q8_1 input base `q8RowBase`.  One 32-lane
    warp per (outIdx=wid.x, position=wid.y).  Input must be pre-quantized to Q8_1
    (`quantizeQ8_1BatchKernel`).  Far cheaper than the f32-dequant expert kernel. -/
def fusedQ4KMBatchExpertDP4AKernel (config : Config) (nExpert N nUsed slot : Nat) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let row := Exp.vec3Y wid
  let tid := Exp.vec3X lid
  let blocksPerRow := config.inDim / 256
  let perExpertU32 := config.outDim * blocksPerRow * 36
  let totalWeightU32 := nExpert * perExpertU32
  let q8BlocksPerRow := config.inDim / 32
  let q8InputU32Size := q8BlocksPerRow * 9
  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareReadOnlyBuffer "input_q8" (.array (.scalar .u32) (N * q8InputU32Size))
  let _idxs ← ShaderM.declareReadOnlyBuffer "idxs" (.array (.scalar .u32) (N * nUsed))
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) (N * config.outDim))
  let inBounds := Exp.lt outIdx (Exp.litU32 config.outDim)
  let expertIdx ← ShaderM.readBuffer (ty := .scalar .u32) (n := N*nUsed) "idxs" (Exp.add (Exp.mul row (Exp.litU32 nUsed)) (Exp.litU32 slot))
  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"
  let rowBaseU32 := Exp.add (Exp.mul expertIdx (Exp.litU32 perExpertU32)) (Exp.mul outIdx (Exp.litU32 (blocksPerRow * 36)))
  let q8RowBase := Exp.mul row (Exp.litU32 q8InputU32Size)
  let laneLow := Exp.bitAnd tid (Exp.litU32 15)
  let pairIdx := Exp.div laneLow (Exp.litU32 4)
  let elemOff := Exp.sub laneLow (Exp.mul pairIdx (Exp.litU32 4))
  let bq8Off := Exp.mul pairIdx (Exp.litU32 2)
  ShaderM.loop (Exp.litU32 0) (Exp.litU32 blocksPerRow) (Exp.litU32 1) fun blockIdx => do
    let blockU32Base := Exp.add rowBaseU32 (Exp.mul blockIdx (Exp.litU32 36))
    let dmU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" blockU32Base
    let dF := Exp.vecX (Exp.unpack2x16float dmU32)
    let dminF := Exp.vecY (Exp.unpack2x16float dmU32)
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
      let scaleHiHi := Exp.shiftLeft (Exp.bitAnd (Exp.shiftRight sc0 (Exp.add shiftHi (Exp.litU32 6))) (Exp.litU32 0x03)) (Exp.litU32 4)
      let scaleHigh := Exp.bitOr scaleHiLo scaleHiHi
      let minHiLo := Exp.bitAnd (Exp.shiftRight sc2 (Exp.add shiftHi (Exp.litU32 4))) (Exp.litU32 0x0F)
      let minHiHi := Exp.shiftLeft (Exp.bitAnd (Exp.shiftRight sc1 (Exp.add shiftHi (Exp.litU32 6))) (Exp.litU32 0x03)) (Exp.litU32 4)
      let minHigh := Exp.bitOr minHiLo minHiHi
      let scaleU := Exp.select isLow scaleLow scaleHigh
      let minU   := Exp.select isLow minLow   minHigh
      (Exp.toF32U scaleU, Exp.toF32U minU)
    let (scA, mA) := extractScaleMin bq8Off
    let (scB, mB) := extractScaleMin (Exp.add bq8Off (Exp.litU32 1))
    let q4BaseIdx := Exp.add blockU32Base (Exp.add (Exp.litU32 4) (Exp.add (Exp.mul bq8Off (Exp.litU32 4)) elemOff))
    let v0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" q4BaseIdx
    let v1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add q4BaseIdx (Exp.litU32 4))
    let q8Sub0Base := Exp.add q8RowBase (Exp.add (Exp.mul blockIdx (Exp.litU32 (8 * 9))) (Exp.mul bq8Off (Exp.litU32 9)))
    let q8Sub1Base := Exp.add q8Sub0Base (Exp.litU32 9)
    let nInQ8 := N * q8InputU32Size
    let u0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := nInQ8) "input_q8" (Exp.add q8Sub0Base (Exp.add (Exp.litU32 1) elemOff))
    let u1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := nInQ8) "input_q8" (Exp.add q8Sub0Base (Exp.add (Exp.litU32 5) elemOff))
    let u2 ← ShaderM.readBuffer (ty := .scalar .u32) (n := nInQ8) "input_q8" (Exp.add q8Sub1Base (Exp.add (Exp.litU32 1) elemOff))
    let u3 ← ShaderM.readBuffer (ty := .scalar .u32) (n := nInQ8) "input_q8" (Exp.add q8Sub1Base (Exp.add (Exp.litU32 5) elemOff))
    let q8Hdr0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := nInQ8) "input_q8" q8Sub0Base
    let q8Hdr1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := nInQ8) "input_q8" q8Sub1Base
    let d8AName ← ShaderM.var (.scalar .f32) (Exp.vecX (Exp.unpack2x16float q8Hdr0))
    let d8BName ← ShaderM.var (.scalar .f32) (Exp.vecX (Exp.unpack2x16float q8Hdr1))
    let d8A : Exp (.scalar .f32) := Exp.var d8AName
    let d8B : Exp (.scalar .f32) := Exp.var d8BName
    let v0i0 := Exp.bitAnd v0 (Exp.litU32 0x0F0F0F0F)
    let v1i0 := Exp.bitAnd v1 (Exp.litU32 0x0F0F0F0F)
    let dot1_0Combined := Exp.add (Exp.dot4I8Packed v0i0 u0) (Exp.dot4I8Packed v1i0 u1)
    let sumU_0 := Exp.add (Exp.dot4I8Packed (Exp.litU32 0x01010101) u0) (Exp.dot4I8Packed (Exp.litU32 0x01010101) u1)
    let sumfD_0 := Exp.mul d8A (Exp.mul (Exp.toF32 dot1_0Combined) scA)
    let sumfM_0 := Exp.mul d8A (Exp.mul (Exp.toF32 sumU_0) mA)
    let v0i1 := Exp.bitAnd (Exp.shiftRight v0 (Exp.litU32 4)) (Exp.litU32 0x0F0F0F0F)
    let v1i1 := Exp.bitAnd (Exp.shiftRight v1 (Exp.litU32 4)) (Exp.litU32 0x0F0F0F0F)
    let dot1_1Combined := Exp.add (Exp.dot4I8Packed v0i1 u2) (Exp.dot4I8Packed v1i1 u3)
    let sumU_1 := Exp.add (Exp.dot4I8Packed (Exp.litU32 0x01010101) u2) (Exp.dot4I8Packed (Exp.litU32 0x01010101) u3)
    let sumfD_1 := Exp.mul d8B (Exp.mul (Exp.toF32 dot1_1Combined) scB)
    let sumfM_1 := Exp.mul d8B (Exp.mul (Exp.toF32 sumU_1) mB)
    let blockContrib := Exp.sub (Exp.mul dF (Exp.add sumfD_0 sumfD_1)) (Exp.mul dminF (Exp.add sumfM_0 sumfM_1))
    ShaderM.assign "acc" (Exp.add acc blockContrib)
  ShaderM.varNamed "total" (.scalar .f32) (Exp.mul (Exp.subgroupAdd acc) (Exp.litF32 0.5))
  let total : Exp (.scalar .f32) := Exp.var "total"
  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    ShaderM.writeBuffer (ty := .scalar .f32) "output" (Exp.add (Exp.mul row (Exp.litU32 config.outDim)) outIdx) total) (pure ())

/-- Q4_K expert matmul via ggml-order INTEGER dot: recover q8=round(a/d_q8) from the already-Q8'd
    activation, accumulate Σ q4·q8 (exact small-int sums) per sub-block, then combine with weight
    scales as ggml does: acc += d_q8·(d·Σ scale·sumi − dmin·Σ min·bsum). Bit-closer than f32 dequant. -/
def fusedQ4KMBatchExpertKernelInt (config : Config) (nExpert N nUsed slot : Nat) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let row := Exp.vec3Y wid
  let tid := Exp.vec3X lid
  let blocksPerRow := config.inDim / 256
  let perExpertU32 := config.outDim * blocksPerRow * 36
  let totalWeightU32 := nExpert * perExpertU32
  let _weights ← ShaderM.declareReadOnlyBuffer "weights" (.array (.scalar .u32) totalWeightU32)
  let _input ← ShaderM.declareReadOnlyBuffer "input" (.array (.scalar .f32) (N * config.inDim))
  let _idxs ← ShaderM.declareReadOnlyBuffer "idxs" (.array (.scalar .u32) (N * nUsed))
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) (N * config.outDim))
  ShaderM.sharedNamed "shared_partial" (.array (.scalar .f32) workgroupSize)
  let inBounds := Exp.lt outIdx (Exp.litU32 config.outDim)
  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"
  let expertIdx ← ShaderM.readBuffer (ty := .scalar .u32) (n := N * nUsed) "idxs" (Exp.add (Exp.mul row (Exp.litU32 nUsed)) (Exp.litU32 slot))
  let rowBaseU32 := Exp.add (Exp.mul expertIdx (Exp.litU32 perExpertU32)) (Exp.mul outIdx (Exp.litU32 (blocksPerRow * 36)))
  ShaderM.loop tid (Exp.litU32 blocksPerRow) (Exp.litU32 workgroupSize) fun blockLocalIdx => do
    let blockU32Base := Exp.add rowBaseU32 (Exp.mul blockLocalIdx (Exp.litU32 36))
    let baseAct := Exp.add (Exp.mul row (Exp.litU32 config.inDim)) (Exp.mul blockLocalIdx (Exp.litU32 256))
    -- pass 1: recover Q8_K block scale d_q8 from the already-Q8'd activation (signed max-abs / -127)
    let amax ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    let vmax ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 256) (Exp.litU32 1) fun dd => do
      let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := N*config.inDim) "input" (Exp.add baseAct dd)
      ShaderM.if_ (Exp.gt (Exp.abs v) (Exp.var amax)) (do
        ShaderM.assign amax (Exp.abs v); ShaderM.assign vmax v) (pure ())
    let dq8 := Exp.select (Exp.gt (Exp.var amax) (Exp.litF32 0.0)) (Exp.div (Exp.var vmax) (Exp.litF32 (-127.0))) (Exp.litF32 1.0)
    let dmU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" blockU32Base
    let d := Exp.vecX (Exp.unpack2x16float dmU32)
    let dmin := Exp.vecY (Exp.unpack2x16float dmU32)
    let sc0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add blockU32Base (Exp.litU32 1))
    let sc1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add blockU32Base (Exp.litU32 2))
    let sc2 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add blockU32Base (Exp.litU32 3))
    let innerD ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    let innerM ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    for c in [0:4] do
      let is0 := c * 2
      let is1 := c * 2 + 1
      let (scaleA, minA) := getScaleMin is0 sc0 sc1 sc2
      let (scaleB, minB) := getScaleMin is1 sc0 sc1 sc2
      let sumiA ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
      let bsumA ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
      let sumiB ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
      let bsumB ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
      let qsU32Base := Exp.add blockU32Base (Exp.litU32 (4 + c * 8))
      for l32 in [0:8] do
        let qsU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalWeightU32) "weights" (Exp.add qsU32Base (Exp.litU32 l32))
        for b in [0:4] do
          let byte := Exp.bitAnd (Exp.shiftRight qsU32 (Exp.litU32 (b * 8))) (Exp.litU32 0xFF)
          let qLow := Exp.toF32 (Exp.bitAnd byte (Exp.litU32 0xF))
          let qHigh := Exp.toF32 (Exp.shiftRight byte (Exp.litU32 4))
          let aLow ← ShaderM.readBuffer (ty := .scalar .f32) (n := N * config.inDim) "input" (Exp.add baseAct (Exp.litU32 (c * 64 + l32 * 4 + b)))
          let aHigh ← ShaderM.readBuffer (ty := .scalar .f32) (n := N * config.inDim) "input" (Exp.add baseAct (Exp.litU32 (c * 64 + 32 + l32 * 4 + b)))
          let q8Low := Exp.round (Exp.div aLow dq8)
          let q8High := Exp.round (Exp.div aHigh dq8)
          ShaderM.assign sumiA (Exp.add (Exp.var sumiA) (Exp.mul qLow q8Low))
          ShaderM.assign bsumA (Exp.add (Exp.var bsumA) q8Low)
          ShaderM.assign sumiB (Exp.add (Exp.var sumiB) (Exp.mul qHigh q8High))
          ShaderM.assign bsumB (Exp.add (Exp.var bsumB) q8High)
      ShaderM.assign innerD (Exp.add (Exp.var innerD) (Exp.add (Exp.mul scaleA (Exp.var sumiA)) (Exp.mul scaleB (Exp.var sumiB))))
      ShaderM.assign innerM (Exp.add (Exp.var innerM) (Exp.add (Exp.mul minA (Exp.var bsumA)) (Exp.mul minB (Exp.var bsumB))))
    ShaderM.assign "acc" (Exp.add acc (Exp.mul dq8 (Exp.sub (Exp.mul d (Exp.var innerD)) (Exp.mul dmin (Exp.var innerM)))))
  ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_partial" tid acc
  ShaderM.barrier
  let mut stride := workgroupSize / 2
  while stride > 0 do
    ShaderM.if_ (Exp.lt tid (Exp.litU32 stride)) (do
      let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_partial" tid
      let bb ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_partial" (Exp.add tid (Exp.litU32 stride))
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_partial" tid (Exp.add a bb)) (pure ())
    ShaderM.barrier
    stride := stride / 2
  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    let totalSum ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_partial" (Exp.litU32 0)
    ShaderM.writeBuffer (ty := .scalar .f32) "output" (Exp.add (Exp.mul row (Exp.litU32 config.outDim)) outIdx) totalSum
  ) (pure ())

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
    | .Q8_0, _ =>
      (256,
       fusedQ8_0LinearKernel layer.config,
       hash ("q8_0-lin", layer.config.inDim, layer.config.outDim))
    | .Q5_0, _ =>
      (256,
       fusedQ5_0BatchKernel layer.config 1,
       hash ("q5_0-lin", layer.config.inDim, layer.config.outDim))
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

  -- Step 2: dp4a fused gate+up+GELU×mul.  Variant selected by HESPER_Q4K_GATEUP:
  --   "4warp" → 1-row × 4-warp coop K (matches llama.cpp shape; measured TPS-neutral
  --             vs 4-row at outDim=10240, see memo project_q4k_gateup_4warp_no_win)
  --   default → 4-row × 1-warp/row + smem-shared input
  let variant := (← IO.getEnv "HESPER_Q4K_GATEUP").getD "4row"
  if variant == "4warp" then
    GPUBackend.executeWithConfigCached ctx
      (fusedQ4KMGateUpDP4A4WarpKernel gate.config)
      [("weights_gate", gate.weightBuf),
       ("weights_up",   up.weightBuf),
       ("input_q8",     q8Buf),
       ("output",       outputBuf)]
      { numWorkgroups := (gate.config.outDim, 1, 1)
        workgroupSize := { x := 128, y := 1, z := 1 } }
      (hash ("q4k-gate-up-dp4a-4warp", gate.config.inDim, gate.config.outDim))
      preparedRef
  else if gate.config.outDim % 4 == 0 then
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

/-- Fused `RMSNorm → Q8_1 quantize → wQ dp4a matmul` for shared-KV layers
    (Gemma 4's half of the transformer stack where K/V are reused from an
    earlier layer).  Mirrors `forwardFusedNormQKV` but only produces Q —
    no wK/wV projection.  Replaces the `circuitRMSNorm + runCached wQ`
    pair (2 dispatches + a circuit-DSL pass) with 2 dispatches (fused
    norm+quantize, then dp4a matmul) through a stable pair of
    prepared-dispatch refs.

    Rationale: Gemma 4 E4B has 21 shared-KV layers out of 42.  Eliminating
    one dispatch per layer saves 21 × cuLaunchKernel (~40 µs/tok) and a
    single D2H-less kernel launch on the replay path of CUDA Graphs. -/
def forwardFusedNormWQ [GPUBackend β] (ctx : β)
    (norm : RMSNorm.RMSNorm (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (wQ : LinearLayer (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (inputBuf qBuf : GPUBackend.Buf β)
    : IO Unit := do
  if wQ.quantFormat != .Q4_K then
    throw (IO.userError "forwardFusedNormWQ: wQ must be Q4_K")
  if wQ.config.inDim != norm.config.dim then
    throw (IO.userError s!"forwardFusedNormWQ: dim mismatch norm={norm.config.dim} wQ.in={wQ.config.inDim}")
  let useDP4A ← do
    let on ← dp4aEnabled.get
    pure (on && wQ.config.inDim % 32 == 0 && wQ.config.inDim % 256 == 0)
  if !useDP4A then
    throw (IO.userError "forwardFusedNormWQ: dp4a precondition failed; caller should fall back")
  let nQ8Blocks := wQ.config.inDim / 32
  let q8BufBytes : USize := (nQ8Blocks * 9 * 4).toUSize
  let q8Buf ← match ← wQ.dp4aQ8Buf.get with
    | some b => pure b
    | none =>
      let b ← GPUBackend.allocBuffer ctx q8BufBytes
      wQ.dp4aQ8Buf.set (some b)
      pure b
  -- Fused RMSNorm + Q8_1 quantize.
  GPUBackend.executeWithConfigCached ctx
    (Hesper.Layers.RMSNorm.fusedRMSNormQ8_1Kernel norm.config)
    [("input", inputBuf), ("scale", norm.scale), ("output", q8Buf)]
    { numWorkgroups := (1, 1, 1), workgroupSize := { x := 256, y := 1, z := 1 } }
    (hash ("fused-rmsnorm-q8_1", norm.config.dim))
    wQ.dp4aQuantizePrepared
  -- Q matmul.  Pick 2-row when outDim is even (typical).
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

/-- Fused [RMSNorm + Q8_1 quantize] → wQ → wK (Q4_K) → wV (Q6_K).

    Variant of `forwardFusedNormQKV` for own-KV layers whose wV is Q6_K
    (~half of Gemma 4's full-attention layers).  The standard Q4_K KV
    fusion requires wV to also be Q4_K so both share `fusedQ4KMKVDP4AKernel`;
    when wV is Q6_K we instead emit:
      1. fusedRMSNormQ8_1   (1 dispatch, replaces separate norm + quantize)
      2. wQ matmul           (Q4_K dp4a)
      3. wK matmul           (Q4_K dp4a)
      4. wV matmul           (Q6_K dp4a)
    = 4 dispatches vs the unfused fallback's 5 (norm + quantize + Q + K + V),
    saving 1 dispatch per affected layer.  Gemma 4 E4B has 11 such layers. -/
def forwardFusedNormQKQ4KVQ6K [GPUBackend β] (ctx : β)
    (norm : RMSNorm.RMSNorm (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (wQ wK wV : LinearLayer (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (inputBuf qBuf kBuf vBuf : GPUBackend.Buf β)
    : IO Unit := do
  if wQ.quantFormat != .Q4_K then
    throw (IO.userError "forwardFusedNormQKQ4KVQ6K: wQ must be Q4_K")
  if wK.quantFormat != .Q4_K then
    throw (IO.userError "forwardFusedNormQKQ4KVQ6K: wK must be Q4_K")
  if wV.quantFormat != .Q6_K then
    throw (IO.userError "forwardFusedNormQKQ4KVQ6K: wV must be Q6_K")
  if wQ.config.inDim != norm.config.dim then
    throw (IO.userError s!"forwardFusedNormQKQ4KVQ6K: dim mismatch")
  let useDP4A ← do
    let on ← dp4aEnabled.get
    pure (on && wQ.config.inDim % 32 == 0 && wQ.config.inDim % 256 == 0)
  if !useDP4A then
    throw (IO.userError "forwardFusedNormQKQ4KVQ6K: dp4a precondition failed")
  let nQ8Blocks := wQ.config.inDim / 32
  let q8BufBytes : USize := (nQ8Blocks * 9 * 4).toUSize
  let q8Buf ← match ← wQ.dp4aQ8Buf.get with
    | some b => pure b
    | none =>
      let b ← GPUBackend.allocBuffer ctx q8BufBytes
      wQ.dp4aQ8Buf.set (some b)
      pure b

  -- Step 1: fused RMSNorm + Q8_1 quantize.
  GPUBackend.executeWithConfigCached ctx
    (Hesper.Layers.RMSNorm.fusedRMSNormQ8_1Kernel norm.config)
    [("input", inputBuf), ("scale", norm.scale), ("output", q8Buf)]
    { numWorkgroups := (1, 1, 1), workgroupSize := { x := 256, y := 1, z := 1 } }
    (hash ("fused-rmsnorm-q8_1", norm.config.dim))
    wQ.dp4aQuantizePrepared

  -- Step 2: wQ matmul (Q4_K dp4a).  Pick 2-row when outDim is even.
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

  -- Step 3: wK matmul (Q4_K dp4a, single output).
  let kIs2Row := wK.config.outDim ≤ 5120 && wK.config.outDim % 2 == 0
  if kIs2Row then
    GPUBackend.executeWithConfigCached ctx
      (fusedQ4KMLinearDP4A2RowKernel wK.config)
      [("weights", wK.weightBuf), ("input_q8", q8Buf), ("output", kBuf)]
      { numWorkgroups := (wK.config.outDim / 2, 1, 1)
        workgroupSize := { x := 64, y := 1, z := 1 } }
      (hash ("q4k-dp4a-matmul-2row", wK.config.inDim, wK.config.outDim))
      wK.dp4aMatmulPrepared
  else
    GPUBackend.executeWithConfigCached ctx
      (fusedQ4KMLinearDP4AKernel wK.config)
      [("weights", wK.weightBuf), ("input_q8", q8Buf), ("output", kBuf)]
      { numWorkgroups := (wK.config.outDim, 1, 1), workgroupSize := { x := 32, y := 1, z := 1 } }
      (hash ("q4k-dp4a-matmul", wK.config.inDim, wK.config.outDim))
      wK.dp4aMatmulPrepared

  -- Step 4: wV matmul (Q6_K dp4a).  Single-row 1-warp variant for small
  -- outDim (Gemma 4: outDim ∈ {512, 1024}).
  GPUBackend.executeWithConfigCached ctx
    (fusedQ6KLinearDP4AKernel wV.config.inDim wV.config.outDim)
    [("weights", wV.weightBuf), ("input_q8", q8Buf), ("output", vBuf)]
    { numWorkgroups := (wV.config.outDim, 1, 1), workgroupSize := { x := 32, y := 1, z := 1 } }
    (hash ("q6k-dp4a-matmul-vq6k", wV.config.inDim, wV.config.outDim))
    wV.dp4aMatmulPrepared

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
