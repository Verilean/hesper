import Hesper.WGSL.Monad
import Hesper.WGSL.Execute
import Hesper.WGSL.Exp
import Hesper.Backend
import Hesper.Logging

/-!
# Matrix Multiplication Kernels

Implements various matrix multiplication strategies for GPU execution.

## Matrix Multiply Variants

### 1. Naive MatMul
- Simple triple-loop implementation
- Each thread computes one output element
- Good for small matrices, easy to understand

### 2. Tiled MatMul
- Block-based computation using workgroup shared memory
- Reduces global memory access by ~N/tile_size
- Optimal tile size: 16x16 or 32x32 (depends on GPU)

### 3. Attention-Specific MatMul
- Q @ K^T: [batch, heads, seq, d] @ [batch, heads, d, seq] → [batch, heads, seq, seq]
- attn @ V: [batch, heads, seq, seq] @ [batch, heads, seq, d] → [batch, heads, seq, d]
- Often includes scaling factor (1/sqrt(d_k))

## Performance Considerations

**Memory bandwidth** (typically the bottleneck):
```
Naive: Each output element reads full row + column
  Operations: M×N×K
  Memory reads: M×N×K (A) + M×N×K (B) = 2×M×N×K
  Arithmetic intensity: 0.5 FLOP/byte (very low!)

Tiled: Reuse data in shared memory
  Memory reads: M×N×K / tile_size
  Arithmetic intensity: tile_size × 0.5 FLOP/byte (much better!)
```

**For attention** (seq_len=2048, d=80):
```
Q @ K^T: [2048, 80] @ [80, 2048]
  Naive: 2048² × 80 × 2 = 671 MB reads
  Tiled (16×16): 671 / 16 = 42 MB reads (16x reduction!)
```

## References
- CUDA GEMM optimization guide: https://docs.nvidia.com/cuda/cublas/
- Attention implementation: llama.cpp/ggml-cuda/attention.cu
-/

namespace Hesper.WGSL.MatMul

open Hesper.WGSL
open Hesper.WGSL.Monad
open Hesper
open Hesper.Logging (logVerbose)

/-! ## Configuration -/

/-- Matrix multiplication configuration -/
structure Config where
  M : Nat  -- Rows of A (and output)
  N : Nat  -- Columns of B (and output)
  K : Nat  -- Columns of A / Rows of B (inner dimension)
  deriving Repr

/-! ## Naive Matrix Multiply -/

/-- Naive matrix multiply: C = A @ B

    A: [M, K]
    B: [K, N]
    C: [M, N]

    Each thread computes one output element C[i,j]:
    C[i,j] = Σₖ A[i,k] × B[k,j]

    @param config Matrix dimensions
-/
def naiveMatMulKernel (config : Config) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let totalElements := config.M * config.N
  let inBounds := Exp.lt idx (Exp.litU32 totalElements)

  -- Declare buffers
  let _a ← ShaderM.declareInputBuffer "a" (.array (.scalar .f32) (config.M * config.K))
  let _b ← ShaderM.declareInputBuffer "b" (.array (.scalar .f32) (config.K * config.N))
  let _c ← ShaderM.declareOutputBuffer "c" (.array (.scalar .f32) totalElements)

  -- Decompose output index: row and column
  let row := Exp.div idx (Exp.litU32 config.N)
  let col := Exp.mod idx (Exp.litU32 config.N)

  -- Accumulate dot product using runtime WGSL for-loop
  let (sumName, sum) ← ShaderM.varRef (.scalar .f32) (Exp.litF32 0.0)

  ShaderM.loop (Exp.litU32 0) (Exp.litU32 config.K) (Exp.litU32 1) fun k => do
    let aIdx := Exp.add (Exp.mul row (Exp.litU32 config.K)) k
    let bIdx := Exp.add (Exp.mul k (Exp.litU32 config.N)) col

    let aVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.M * config.K) "a" aIdx
    let bVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.K * config.N) "b" bIdx

    let prod := Exp.mul aVal bVal
    ShaderM.assign sumName (Exp.add sum prod)

  -- Write result
  let result := Exp.select inBounds sum (Exp.litF32 0.0)
  ShaderM.writeBuffer (ty := .scalar .f32) "c" idx result

/-! ## Transposed Matrix Multiply -/

/-- Matrix multiply with B transposed: C = A @ B^T

    A: [M, K]
    B: [N, K]  (note: same K dimension, not transposed in memory)
    C: [M, N]

    This is more efficient when B is already stored in row-major format.
    Common in attention: Q @ K^T where K is [seq, d]

    @param config Matrix dimensions (K is shared dimension)
-/
def matMulTransposeBKernel (config : Config) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let totalElements := config.M * config.N
  let inBounds := Exp.lt idx (Exp.litU32 totalElements)

  let _a ← ShaderM.declareInputBuffer "a" (.array (.scalar .f32) (config.M * config.K))
  let _b ← ShaderM.declareInputBuffer "b" (.array (.scalar .f32) (config.N * config.K))
  let _c ← ShaderM.declareOutputBuffer "c" (.array (.scalar .f32) totalElements)

  let row := Exp.div idx (Exp.litU32 config.N)
  let col := Exp.mod idx (Exp.litU32 config.N)

  let (sumName, sum) ← ShaderM.varRef (.scalar .f32) (Exp.litF32 0.0)

  ShaderM.loop (Exp.litU32 0) (Exp.litU32 config.K) (Exp.litU32 1) fun k => do
    -- A[row, k]
    let aIdx := Exp.add (Exp.mul row (Exp.litU32 config.K)) k
    -- B^T[col, k] = B[col, k] (since B is stored as [N, K])
    let bIdx := Exp.add (Exp.mul col (Exp.litU32 config.K)) k

    let aVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.M * config.K) "a" aIdx
    let bVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.N * config.K) "b" bIdx

    let prod := Exp.mul aVal bVal
    ShaderM.assign sumName (Exp.add sum prod)

  let result := Exp.select inBounds sum (Exp.litF32 0.0)
  ShaderM.writeBuffer (ty := .scalar .f32) "c" idx result

/-! ## F16 Transposed Matrix Multiply (for LM Head) -/

/-- Matrix multiply with B transposed, B stored as packed F16: C = A @ B^T

    A: [M, K] in F32
    B: [N, K] stored as packed F16 (each u32 = 2 F16 values via pack2x16float)
    C: [M, N] in F32

    Uses hardware `unpack2x16float` to convert F16→F32 during computation.
    Processes 2 K-elements per loop iteration for 2x bandwidth reduction on B.

    Primary use: LM head projection where B is the F16 embedding table (656 MB vs 1.3 GB F32).

    @param config Matrix dimensions (K must be even)
-/
def matMulTransposeF16Kernel (config : Config) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let totalElements := config.M * config.N
  let packedK := config.K / 2  -- K/2 u32 values per row

  -- A: [M, K] in F32
  let _a ← ShaderM.declareInputBuffer "a" (.array (.scalar .f32) (config.M * config.K))
  -- B: [N, K] stored as packed F16 → [N, K/2] u32 values
  let _b ← ShaderM.declareInputBuffer "b" (.array (.scalar .u32) (config.N * packedK))
  let _c ← ShaderM.declareOutputBuffer "c" (.array (.scalar .f32) totalElements)

  let row := Exp.div idx (Exp.litU32 config.N)
  let col := Exp.mod idx (Exp.litU32 config.N)

  let (accName, acc) ← ShaderM.varRef (.scalar .f32) (Exp.litF32 0.0)

  let aBase := Exp.mul row (Exp.litU32 config.K)
  let bBase := Exp.mul col (Exp.litU32 packedK)

  -- Loop over K/2 packed F16 pairs
  ShaderM.loop (Exp.litU32 0) (Exp.litU32 packedK) (Exp.litU32 1) fun kPair => do
    -- Read packed u32 containing two F16 values
    let packed ← ShaderM.readBuffer (ty := .scalar .u32) (n := config.N * packedK) "b" (Exp.add bBase kPair)

    -- Hardware unpack: u32 → vec2<f32>
    let unpacked := Exp.unpack2x16float packed
    let b0 := Exp.vecX unpacked
    let b1 := Exp.vecY unpacked

    -- Read corresponding A values
    let kIdx0 := Exp.mul kPair (Exp.litU32 2)
    let kIdx1 := Exp.add kIdx0 (Exp.litU32 1)
    let a0 ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.M * config.K) "a" (Exp.add aBase kIdx0)
    let a1 ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.M * config.K) "a" (Exp.add aBase kIdx1)

    -- Accumulate: acc += a0*b0 + a1*b1
    ShaderM.assign accName (Exp.add acc (Exp.add (Exp.mul a0 b0) (Exp.mul a1 b1)))

  let inBounds := Exp.lt idx (Exp.litU32 totalElements)
  let result := Exp.select inBounds acc (Exp.litF32 0.0)
  ShaderM.writeBuffer (ty := .scalar .f32) "c" idx result

/-! ## Block-Cooperative + SW-Pipelined F16 Transposed MatMul (M=1 decode) -/

/-- Block-cooperative F16 matrix-vector multiply with B transposed and
    stored as packed F16: `C = A @ B^T`, specialised to **M = 1**
    (single-row A). Mirrors the Q4_K / Q6_K block-coop pattern:

      * 1 workgroup per output element (N workgroups), 32 threads each
      * The K dim is tiled into `kTilesPerRow = K / 64` tiles of 64
        F16s = 32 u32s. Each lane `tid ∈ [0, 32)` reads one u32 per
        tile (contiguous across lanes → one coalesced 128-byte load),
        unpacks it into two F16→F32 values, pairs them with the
        matching two A f32s, and accumulates two FMAs.
      * Depth-1 software pipelining: each iteration, the u32 for tile
        N+1 is prefetched into a mutable `nextU32` var BEFORE the
        dequant+FMA chain that consumes the current snapshot, so
        memory load latency overlaps with FMA latency (same trick
        that cut Q4_K ffnGate/Up to 0.059 ms at 335 GB/s).

    Constraints: `M = 1` (the inner loop keeps the single A row in
    registers), `K % 64 == 0` (per-lane tile of 64 F16 values needs 32
    u32s cleanly), subgroup support required.

    Weight layout: row-major `[N, K/2]` packed F16 (same as
    `matMulTransposeF16Kernel`). Row `n` starts at u32 index
    `n * (K/2)`.
-/
def matMulTransposeF16BlockCoopKernel (config : Config) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid

  let packedK := config.K / 2        -- u32s per row of B
  let kTilesPerRow := config.K / 64  -- tiles of 64 F16s (= 32 u32s) per row

  let _a ← ShaderM.declareInputBuffer "a" (.array (.scalar .f32) config.K)
  let _b ← ShaderM.declareInputBuffer "b" (.array (.scalar .u32) (config.N * packedK))
  let _c ← ShaderM.declareOutputBuffer "c" (.array (.scalar .f32) config.N)

  let inBounds := Exp.lt outIdx (Exp.litU32 config.N)

  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"

  -- Row base in u32 units (runtime; depends on outIdx).
  let rowBaseU32 := Exp.mul outIdx (Exp.litU32 packedK)

  -- ## Pre-loop prefetch of tile 0's per-lane u32.
  --   lane tid reads u32 at row_base + 0*32 + tid.
  let init0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := config.N * packedK) "b"
                (Exp.add rowBaseU32 tid)
  ShaderM.varNamed "nextU32" (.scalar .u32) init0

  for tile in [0:kTilesPerRow] do
    -- Snapshot into per-iteration `curr` var.
    let cName := s!"currU32_{tile}"
    ShaderM.varNamed cName (.scalar .u32) (Exp.var "nextU32" : Exp (.scalar .u32))
    let currU32 : Exp (.scalar .u32) := Exp.var cName

    -- Issue next tile's load BEFORE dequant+FMA. Independent of curr*.
    if tile + 1 < kTilesPerRow then
      let nextBase := Exp.add rowBaseU32 (Exp.litU32 ((tile + 1) * 32))
      ShaderM.assign "nextU32"
        (← ShaderM.readBuffer (ty := .scalar .u32) (n := config.N * packedK) "b"
              (Exp.add nextBase tid))

    -- Unpack: u32 → vec2<f32>. Each lane's u32 holds two consecutive
    -- F16 values for K-positions (tile*64 + tid*2, tile*64 + tid*2 + 1).
    let unpacked := Exp.unpack2x16float currU32
    let b0 := Exp.vecX unpacked
    let b1 := Exp.vecY unpacked

    -- Corresponding A reads. M=1 so row is 0; A is flat length K.
    let kIdx0 := Exp.add (Exp.litU32 (tile * 64)) (Exp.mul tid (Exp.litU32 2))
    let kIdx1 := Exp.add kIdx0 (Exp.litU32 1)
    let a0 ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.K) "a" kIdx0
    let a1 ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.K) "a" kIdx1

    -- Two chained fmas: acc = a0 * b0 + acc; acc = a1 * b1 + acc.
    -- Maps to two `fma.rn.f32` PTX inst (vs the previous mul + add chain
    -- that emitted 4 inst when ptxas didn't fold them).  Matches
    -- llama.cpp's `mul_mat_vec_f<half,float>` inner loop (mmvf.cu:175-177)
    -- which uses `ggml_cuda_mad` = `__fmaf_rn` for both lanes.
    ShaderM.assign "acc" (Exp.fma a0 b0 acc)
    ShaderM.assign "acc" (Exp.fma a1 b1 (Exp.var "acc" : Exp (.scalar .f32)))

  ShaderM.varNamed "totalSum" (.scalar .f32) (Exp.subgroupAdd acc)
  let totalSum : Exp (.scalar .f32) := Exp.var "totalSum"

  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
    ShaderM.writeBuffer (ty := .scalar .f32) "c" outIdx totalSum
  ) (pure ())

/-- Execute `matMulTransposeF16BlockCoopKernel`. See the kernel doc for
    preconditions (`M = 1`, `K % 64 == 0`, subgroup support). -/
def executeMatMulTransposeF16BlockCoop [GPUBackend β] (ctx : β)
                                        (aBuf bF16Buf cBuf : GPUBackend.Buf β)
                                        (config : Config) : IO Unit := do
  let namedBuffers := [
    ("a", aBuf),
    ("b", bF16Buf),
    ("c", cBuf)
  ]
  let execConfig : Hesper.ExecConfig := {
    numWorkgroups := (config.N, 1, 1)
    workgroupSize := { x := 32, y := 1, z := 1 }
    extensions := ["subgroups"]
  }
  let cacheKey : UInt64 := hash ("mmf16-blockcoop", config.M, config.N, config.K)
  GPUBackend.executeWithConfigCached ctx
    (matMulTransposeF16BlockCoopKernel config)
    namedBuffers execConfig cacheKey (← IO.mkRef none)

/-! ## Optimized F16 Transposed MatMul with Shared Memory -/

/-- Optimized matrix multiply with B transposed, B stored as packed F16: C = A @ B^T

    Key optimizations over `matMulTransposeF16Kernel`:
    1. A vector loaded into workgroup shared memory (eliminates redundant global reads)
    2. Inner loop processes 4 u32 (= 8 F16 values) per iteration (4x fewer loop iterations)

    For M=1 LM head (1×2560 @ 128256×2560):
    - Old: each of 128K threads reads all 2560 A values from global memory = 1.3 GB total A reads
    - New: each workgroup (256 threads) loads A once into shared memory = 5 MB total A reads

    Requirements: K must be divisible by 8

    @param config Matrix dimensions (K must be divisible by 8)
-/
def matMulTransposeF16SharedKernel (config : Config) : ShaderM Unit := do
  let workgroupSize := 256
  let gid ← ShaderM.globalId
  let lid ← ShaderM.localId
  let idx := Exp.vec3X gid
  let tid := Exp.vec3X lid

  let totalElements := config.M * config.N
  let packedK := config.K / 2  -- K/2 u32 values per row in B
  let loopCount := packedK / 4 -- process 4 u32 (8 F16) per iteration

  -- Shared memory for A row (e.g., 2560 × 4B = 10 KB for BitNet)
  ShaderM.sharedNamed "shared_a" (.array (.scalar .f32) config.K)

  -- Buffers
  let _a ← ShaderM.declareInputBuffer "a" (.array (.scalar .f32) (config.M * config.K))
  let _b ← ShaderM.declareInputBuffer "b" (.array (.scalar .u32) (config.N * packedK))
  let _c ← ShaderM.declareOutputBuffer "c" (.array (.scalar .f32) totalElements)

  let row := Exp.div idx (Exp.litU32 config.N)
  let col := Exp.mod idx (Exp.litU32 config.N)

  -- Step 1: Cooperatively load A[row, 0..K-1] into shared memory
  let aBase := Exp.mul row (Exp.litU32 config.K)
  ShaderM.loop tid (Exp.litU32 config.K) (Exp.litU32 workgroupSize) fun i => do
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.M * config.K) "a" (Exp.add aBase i)
    ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_a" i val
  ShaderM.barrier

  -- Step 2: Compute dot product using shared A and packed F16 B
  let (accName, acc) ← ShaderM.varRef (.scalar .f32) (Exp.litF32 0.0)
  let bBase := Exp.mul col (Exp.litU32 packedK)

  -- Main loop: process 4 u32 (= 8 F16 values) per iteration
  ShaderM.loop (Exp.litU32 0) (Exp.litU32 loopCount) (Exp.litU32 1) fun k4 => do
    let baseU32 := Exp.mul k4 (Exp.litU32 4)
    let kBase := Exp.mul k4 (Exp.litU32 8)

    -- Compile-time unroll: read 4 packed u32, unpack 8 F16, do 8 FMAs
    for i in [0:4] do
      let packed ← ShaderM.readBuffer (ty := .scalar .u32) (n := config.N * packedK) "b"
        (Exp.add bBase (Exp.add baseU32 (Exp.litU32 i)))
      let unpacked := Exp.unpack2x16float packed
      let k0 := Exp.add kBase (Exp.litU32 (i * 2))
      let k1 := Exp.add kBase (Exp.litU32 (i * 2 + 1))
      let a0 ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := config.K) "shared_a" k0
      let a1 ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := config.K) "shared_a" k1
      ShaderM.assign accName (Exp.add acc (Exp.add (Exp.mul (Exp.vecX unpacked) a0) (Exp.mul (Exp.vecY unpacked) a1)))

  -- Step 3: Write result
  let inBounds := Exp.lt idx (Exp.litU32 totalElements)
  let result := Exp.select inBounds acc (Exp.litF32 0.0)
  ShaderM.writeBuffer (ty := .scalar .f32) "c" idx result

/-! ## Cooperative-Matrix (WMMA) F16 Transposed MatMul -/

/-- `C = A @ B^T` using NVIDIA / Chromium subgroup matrix operations.

    * `A`: `[M, K]` f32 — cast to f16 on load into workgroup memory
    * `B`: `[N, K]` f16 (packed as u32, two halves per u32) — the same
      layout used by `matMulTransposeF16Kernel` / `-SharedKernel`. The
      kernel reads B in its native f16 form and never materialises a
      f32 copy.
    * `C`: `[M, N]` f32 — cooperative matrix accumulator, written back
      unscaled.

    The kernel targets the NVIDIA Ada/Ampere WMMA config
    `(in=f16, out=f32, M=N=K=16)`, which is exposed by Dawn when both
    `ShaderF16` and `ChromiumExperimentalSubgroupMatrix` are enabled.

    Workgroup layout: 1 subgroup (32 threads) per 16×16 output tile.
    Dispatch `(N/16, M/16, 1)` workgroups.

    Constraints (enforced by the caller):
      * `config.M % 16 == 0`
      * `config.N % 16 == 0`
      * `config.K % 16 == 0`

    Note: this kernel is **generic** (unlike `fusedBitLinearSubgroupMatrixKernel`
    it does no ternary dequant) and is intended to back the Gemma 4
    `Q4_K`/`Q6_K` linears and the LM head once those paths are wired up.
    The pre-dequant-to-f16 variant is what makes it reusable.
-/
def matMulTransposeF16WMMAKernel (config : Config) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let wgN := Exp.vec3X wid    -- output column tile (N direction)
  let wgM := Exp.vec3Y wid    -- output row    tile (M direction)
  let tid := Exp.vec3X lid    -- lane within subgroup, 0..31

  let packedK := config.K / 2  -- K/2 u32 values per row in B
  let totalOut := config.M * config.N

  let _a ← ShaderM.declareInputBuffer "a" (.array (.scalar .f32) (config.M * config.K))
  let _b ← ShaderM.declareInputBuffer "b" (.array (.scalar .u32) (config.N * packedK))
  let _c ← ShaderM.declareOutputBuffer "c" (.array (.scalar .f32) totalOut)

  -- Workgroup-private tile staging.
  ShaderM.sharedNamed "shared_A" (.array (.scalar .f16) 256)
  ShaderM.sharedNamed "shared_B" (.array (.scalar .f16) 256)
  ShaderM.sharedNamed "shared_C" (.array (.scalar .f32) 256)

  ShaderM.declareMatrixLeftArray  "Ax" .f16 16 16 1 Exp.subgroupMatrixZeroLeft
  ShaderM.declareMatrixRightArray "Bx" .f16 16 16 1 Exp.subgroupMatrixZeroRight
  ShaderM.declareMatrixResultArray "Cx" .f32 16 16 1 Exp.subgroupMatrixZeroResult

  let rowBase := Exp.mul wgM (Exp.litU32 16)
  let colBase := Exp.mul wgN (Exp.litU32 16)

  -- Load a 16×16 f16 A tile from A[rowBase..+16, kBase..+16].
  let loadTileA (kBase : Exp (.scalar .u32)) : ShaderM Unit := do
    -- 32 threads, 256 elements → 8 per thread.
    for s in [0:8] do
      let e := Exp.add tid (Exp.litU32 (s * 32))
      let mi := Exp.div e (Exp.litU32 16)
      let ki := Exp.mod e (Exp.litU32 16)
      let row := Exp.add rowBase mi
      let col := Exp.add kBase ki
      let inIdx := Exp.add (Exp.mul row (Exp.litU32 config.K)) col
      let xf32 ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.M * config.K) "a" inIdx
      ShaderM.writeWorkgroup (ty := .scalar .f16) "shared_A" e (Exp.toF16 xf32)

  -- Load a 16×16 f16 B tile from B[colBase..+16, kBase..+16] (B is
  -- stored in [N, K] order; we want B^T[kBase..+16, colBase..+16] for
  -- the subgroup_matrix_right, so element [ki, ni] of the tile corresponds
  -- to B[colBase + ni, kBase + ki]).
  --
  -- B is packed as u32 with two f16 halves per u32, so one u32 holds
  -- two consecutive K entries for a given N row.
  let loadTileB (kBase : Exp (.scalar .u32)) : ShaderM Unit := do
    -- 32 threads, 256 elements → 8 per thread. But it's cheaper to load
    -- 2 elements per u32 fetch, so use 16 threads × 8 u32s = 128 u32 fetches
    -- (each yielding 2 f16s = 256 total elements).
    for s in [0:8] do
      let u32LaneIdx := Exp.add tid (Exp.litU32 (s * 32))
      -- Each u32 holds 2 adjacent K entries for one N row.
      let pairIdx := Exp.mod u32LaneIdx (Exp.litU32 8)    -- which pair within the N row (0..7)
      let ni := Exp.div u32LaneIdx (Exp.litU32 8)         -- which N row (0..15)
      -- Guard in case of overflow (only first 128 lanes are meaningful).
      ShaderM.if_ (Exp.lt u32LaneIdx (Exp.litU32 128)) (do
        let row := Exp.add colBase ni
        let kLocal := Exp.mul pairIdx (Exp.litU32 2)      -- 0, 2, 4, ..., 14
        let kGlobal := Exp.add kBase kLocal
        let kPairGlobal := Exp.div kGlobal (Exp.litU32 2)
        let bIdx := Exp.add (Exp.mul row (Exp.litU32 packedK)) kPairGlobal
        let packed ← ShaderM.readBuffer (ty := .scalar .u32) (n := config.N * packedK) "b" bIdx
        let unpacked := Exp.unpack2x16float packed
        let b0 := Exp.vecX unpacked
        let b1 := Exp.vecY unpacked
        -- B tile is row-major [K=16, N=16]; element [ki, ni] at index ki*16 + ni.
        let idx0 := Exp.add (Exp.mul kLocal (Exp.litU32 16)) ni
        let idx1 := Exp.add (Exp.mul (Exp.add kLocal (Exp.litU32 1)) (Exp.litU32 16)) ni
        ShaderM.writeWorkgroup (ty := .scalar .f16) "shared_B" idx0 (Exp.toF16 b0)
        ShaderM.writeWorkgroup (ty := .scalar .f16) "shared_B" idx1 (Exp.toF16 b1)
      ) (pure ())

  -- Main K loop — 16-block steps over inner dim.
  let numKTiles := config.K / 16
  for kTile in [0:numKTiles] do
    let kBase := Exp.litU32 (kTile * 16)
    loadTileA kBase
    loadTileB kBase
    ShaderM.barrier
    ShaderM.loadMatrixLeft (st := .f16) (m := 16) (k := 16)
      "Ax" 0 "shared_A" (Exp.litU32 0) (Exp.litU32 16)
    ShaderM.loadMatrixRight (st := .f16) (k := 16) (n := 16)
      "Bx" 0 "shared_B" (Exp.litU32 0) (Exp.litU32 16)
    ShaderM.matrixMultiplyAccumulateMixed
      (inSt := .f16) (outSt := .f32) (m := 16) (k := 16) (n := 16)
      "Cx" 0 "Ax" 0 "Bx" 0
    ShaderM.barrier

  -- Store the f32 accumulator tile to shared memory, then copy to C.
  ShaderM.storeMatrixResult (st := .f32) (m := 16) (n := 16)
    "Cx" 0 "shared_C" (Exp.litU32 0) (Exp.litU32 16)
  ShaderM.barrier

  for s in [0:8] do
    let e := Exp.add tid (Exp.litU32 (s * 32))
    let mi := Exp.div e (Exp.litU32 16)
    let ni := Exp.mod e (Exp.litU32 16)
    let row := Exp.add rowBase mi
    let col := Exp.add colBase ni
    let v ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 256) "shared_C" e
    let outIdx := Exp.add (Exp.mul row (Exp.litU32 config.N)) col
    ShaderM.writeBuffer (ty := .scalar .f32) "c" outIdx v

/-- 8×8×8 variant of `matMulTransposeF16WMMAKernel` for hardware that only supports 8×8 subgroup
    matrices (Apple M-series / Metal: probe reports M=N=K=8 only; the 16×16 kernel crashes at
    pipeline creation). A: f32 [M,K]; B: u32-packed f16 [N, K/2] (i.e. B^T); C: f32 [M,N].
    One subgroup (32 lanes) per 8×8 output tile. Requires M,N,K multiples of 8. -/
def matMulTransposeF16WMMA8x8Kernel (config : Config) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let wgN := Exp.vec3X wid
  let wgM := Exp.vec3Y wid
  let tid := Exp.vec3X lid
  let packedK := config.K / 2
  let totalOut := config.M * config.N
  let _a ← ShaderM.declareInputBuffer "a" (.array (.scalar .f32) (config.M * config.K))
  let _b ← ShaderM.declareInputBuffer "b" (.array (.scalar .u32) (config.N * packedK))
  let _c ← ShaderM.declareOutputBuffer "c" (.array (.scalar .f32) totalOut)
  ShaderM.sharedNamed "shared_A" (.array (.scalar .f16) 64)
  ShaderM.sharedNamed "shared_B" (.array (.scalar .f16) 64)
  ShaderM.sharedNamed "shared_C" (.array (.scalar .f32) 64)
  ShaderM.declareMatrixLeftArray  "Ax" .f16 8 8 1 Exp.subgroupMatrixZeroLeft
  ShaderM.declareMatrixRightArray "Bx" .f16 8 8 1 Exp.subgroupMatrixZeroRight
  ShaderM.declareMatrixResultArray "Cx" .f32 8 8 1 Exp.subgroupMatrixZeroResult
  let rowBase := Exp.mul wgM (Exp.litU32 8)
  let colBase := Exp.mul wgN (Exp.litU32 8)
  -- Load 8×8 f16 A tile (64 elems, 32 lanes → 2 each).
  let loadTileA (kBase : Exp (.scalar .u32)) : ShaderM Unit := do
    for s in [0:2] do
      let e := Exp.add tid (Exp.litU32 (s * 32))
      let mi := Exp.div e (Exp.litU32 8)
      let ki := Exp.mod e (Exp.litU32 8)
      let row := Exp.add rowBase mi
      let col := Exp.add kBase ki
      let inIdx := Exp.add (Exp.mul row (Exp.litU32 config.K)) col
      let xf32 ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.M * config.K) "a" inIdx
      ShaderM.writeWorkgroup (ty := .scalar .f16) "shared_A" e (Exp.toF16 xf32)
  -- Load 8×8 f16 B tile (64 elems = 32 u32, one u32 per lane = 2 K entries for one N row).
  let loadTileB (kBase : Exp (.scalar .u32)) : ShaderM Unit := do
    let pairIdx := Exp.mod tid (Exp.litU32 4)   -- K-pair 0..3
    let ni := Exp.div tid (Exp.litU32 4)         -- N row 0..7
    let row := Exp.add colBase ni
    let kLocal := Exp.mul pairIdx (Exp.litU32 2)
    let kGlobal := Exp.add kBase kLocal
    let kPairGlobal := Exp.div kGlobal (Exp.litU32 2)
    let bIdx := Exp.add (Exp.mul row (Exp.litU32 packedK)) kPairGlobal
    let packed ← ShaderM.readBuffer (ty := .scalar .u32) (n := config.N * packedK) "b" bIdx
    let unpacked := Exp.unpack2x16float packed
    let idx0 := Exp.add (Exp.mul kLocal (Exp.litU32 8)) ni
    let idx1 := Exp.add (Exp.mul (Exp.add kLocal (Exp.litU32 1)) (Exp.litU32 8)) ni
    ShaderM.writeWorkgroup (ty := .scalar .f16) "shared_B" idx0 (Exp.toF16 (Exp.vecX unpacked))
    ShaderM.writeWorkgroup (ty := .scalar .f16) "shared_B" idx1 (Exp.toF16 (Exp.vecY unpacked))
  let numKTiles := config.K / 8
  for kTile in [0:numKTiles] do
    let kBase := Exp.litU32 (kTile * 8)
    loadTileA kBase
    loadTileB kBase
    ShaderM.barrier
    ShaderM.loadMatrixLeft (st := .f16) (m := 8) (k := 8) "Ax" 0 "shared_A" (Exp.litU32 0) (Exp.litU32 8)
    ShaderM.loadMatrixRight (st := .f16) (k := 8) (n := 8) "Bx" 0 "shared_B" (Exp.litU32 0) (Exp.litU32 8)
    ShaderM.matrixMultiplyAccumulateMixed (inSt := .f16) (outSt := .f32) (m := 8) (k := 8) (n := 8) "Cx" 0 "Ax" 0 "Bx" 0
    ShaderM.barrier
  ShaderM.storeMatrixResult (st := .f32) (m := 8) (n := 8) "Cx" 0 "shared_C" (Exp.litU32 0) (Exp.litU32 8)
  ShaderM.barrier
  for s in [0:2] do
    let e := Exp.add tid (Exp.litU32 (s * 32))
    let mi := Exp.div e (Exp.litU32 8)
    let ni := Exp.mod e (Exp.litU32 8)
    let row := Exp.add rowBase mi
    let col := Exp.add colBase ni
    let v ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 64) "shared_C" e
    let outIdx := Exp.add (Exp.mul row (Exp.litU32 config.N)) col
    ShaderM.writeBuffer (ty := .scalar .f32) "c" outIdx v

/-- Register-blocked 8×8 WMMA: one workgroup computes a TM×TN grid of 8×8 output tiles
    (= 8·TM × 8·TN output), keeping TM·TN f32 result matrices in registers across the whole
    K-loop and reusing each loaded A/B 8×8 tile TN/TM ways. Amortizes the shared-memory loads
    and barriers TM·TN×. A: f32 [M,K]; B: u32-packed f16 [N,K/2]; C: f32 [M,N].
    Requires M % (8·TM) = 0, N % (8·TN) = 0, K % 8 = 0. Grid: (N/(8·TN), M/(8·TM)) × 32. -/
def matMulTransposeF16WMMA8x8RegKernel (config : Config) (TM TN : Nat) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let wgN := Exp.vec3X wid
  let wgM := Exp.vec3Y wid
  let tid := Exp.vec3X lid
  let packedK := config.K / 2
  let totalOut := config.M * config.N
  let _a ← ShaderM.declareInputBuffer "a" (.array (.scalar .f32) (config.M * config.K))
  let _b ← ShaderM.declareInputBuffer "b" (.array (.scalar .u32) (config.N * packedK))
  let _c ← ShaderM.declareOutputBuffer "c" (.array (.scalar .f32) totalOut)
  ShaderM.sharedNamed "shared_A" (.array (.scalar .f16) (TM * 64))
  ShaderM.sharedNamed "shared_B" (.array (.scalar .f16) (TN * 64))
  ShaderM.sharedNamed "shared_C" (.array (.scalar .f32) (TM * TN * 64))
  ShaderM.declareMatrixLeftArray  "Ax" .f16 8 8 TM Exp.subgroupMatrixZeroLeft
  ShaderM.declareMatrixRightArray "Bx" .f16 8 8 TN Exp.subgroupMatrixZeroRight
  ShaderM.declareMatrixResultArray "Cx" .f32 8 8 (TM * TN) Exp.subgroupMatrixZeroResult
  let rowBase := Exp.mul wgM (Exp.litU32 (8 * TM))
  let colBase := Exp.mul wgN (Exp.litU32 (8 * TN))
  let numKTiles := config.K / 8
  for kTile in [0:numKTiles] do
    let kBase := Exp.litU32 (kTile * 8)
    -- load TM A-tiles (64 f16 each, 2/lane)
    for ti in [0:TM] do
      for s in [0:2] do
        let e := Exp.add tid (Exp.litU32 (s * 32))
        let mi := Exp.div e (Exp.litU32 8)
        let ki := Exp.mod e (Exp.litU32 8)
        let row := Exp.add (Exp.add rowBase (Exp.litU32 (ti * 8))) mi
        let col := Exp.add kBase ki
        let xf32 ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.M * config.K) "a"
          (Exp.add (Exp.mul row (Exp.litU32 config.K)) col)
        ShaderM.writeWorkgroup (ty := .scalar .f16) "shared_A" (Exp.add (Exp.litU32 (ti * 64)) e) (Exp.toF16 xf32)
    -- load TN B-tiles (64 f16 = 32 u32 each, 1 u32/lane)
    for tj in [0:TN] do
      let pairIdx := Exp.mod tid (Exp.litU32 4)
      let ni := Exp.div tid (Exp.litU32 4)
      let row := Exp.add (Exp.add colBase (Exp.litU32 (tj * 8))) ni
      let kLocal := Exp.mul pairIdx (Exp.litU32 2)
      let kPairGlobal := Exp.div (Exp.add kBase kLocal) (Exp.litU32 2)
      let packed ← ShaderM.readBuffer (ty := .scalar .u32) (n := config.N * packedK) "b"
        (Exp.add (Exp.mul row (Exp.litU32 packedK)) kPairGlobal)
      let unpacked := Exp.unpack2x16float packed
      let base := Exp.litU32 (tj * 64)
      ShaderM.writeWorkgroup (ty := .scalar .f16) "shared_B"
        (Exp.add base (Exp.add (Exp.mul kLocal (Exp.litU32 8)) ni)) (Exp.toF16 (Exp.vecX unpacked))
      ShaderM.writeWorkgroup (ty := .scalar .f16) "shared_B"
        (Exp.add base (Exp.add (Exp.mul (Exp.add kLocal (Exp.litU32 1)) (Exp.litU32 8)) ni)) (Exp.toF16 (Exp.vecY unpacked))
    ShaderM.barrier
    for ti in [0:TM] do
      ShaderM.loadMatrixLeft (st := .f16) (m := 8) (k := 8) "Ax" ti "shared_A" (Exp.litU32 (ti * 64)) (Exp.litU32 8)
    for tj in [0:TN] do
      ShaderM.loadMatrixRight (st := .f16) (k := 8) (n := 8) "Bx" tj "shared_B" (Exp.litU32 (tj * 64)) (Exp.litU32 8)
    for ti in [0:TM] do
      for tj in [0:TN] do
        ShaderM.matrixMultiplyAccumulateMixed (inSt := .f16) (outSt := .f32) (m := 8) (k := 8) (n := 8)
          "Cx" (ti * TN + tj) "Ax" ti "Bx" tj
    ShaderM.barrier
  -- store all TM·TN result tiles to shared, then copy to C
  for ti in [0:TM] do
    for tj in [0:TN] do
      ShaderM.storeMatrixResult (st := .f32) (m := 8) (n := 8) "Cx" (ti * TN + tj) "shared_C"
        (Exp.litU32 ((ti * TN + tj) * 64)) (Exp.litU32 8)
  ShaderM.barrier
  for ti in [0:TM] do
    for tj in [0:TN] do
      for s in [0:2] do
        let e := Exp.add tid (Exp.litU32 (s * 32))
        let mi := Exp.div e (Exp.litU32 8)
        let ni := Exp.mod e (Exp.litU32 8)
        let row := Exp.add (Exp.add rowBase (Exp.litU32 (ti * 8))) mi
        let col := Exp.add (Exp.add colBase (Exp.litU32 (tj * 8))) ni
        let v ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := TM * TN * 64) "shared_C"
          (Exp.add (Exp.litU32 ((ti * TN + tj) * 64)) e)
        ShaderM.writeBuffer (ty := .scalar .f32) "c" (Exp.add (Exp.mul row (Exp.litU32 config.N)) col) v

/-- SUBGROUP-blocked 8×8 WMMA: one workgroup runs TM·TN subgroups (32 lanes each), each computing
    ONE 8×8 output tile (1 result matrix → no register spill), but the workgroup loads TM A-tiles +
    TN B-tiles into shared memory ONCE per K-step and all subgroups reuse them. Cuts the A-row/B-col
    re-read (TN/TM×) and the total workgroup/barrier count (TM·TN×) without the matrix-register
    pressure that sinks register-blocking on Apple GPUs. workgroupSize = TM·TN·32 (S subgroups).
    Grid: (N/(8·TN), M/(8·TM)) × (TM·TN·32). -/
def matMulTransposeF16WMMA8x8SgKernel (config : Config) (TM TN : Nat) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let wgN := Exp.vec3X wid
  let wgM := Exp.vec3Y wid
  let tid := Exp.vec3X lid
  let nThreads := TM * TN * 32
  let packedK := config.K / 2
  let _a ← ShaderM.declareInputBuffer "a" (.array (.scalar .f32) (config.M * config.K))
  let _b ← ShaderM.declareInputBuffer "b" (.array (.scalar .u32) (config.N * packedK))
  let _c ← ShaderM.declareOutputBuffer "c" (.array (.scalar .f32) (config.M * config.N))
  ShaderM.sharedNamed "shared_A" (.array (.scalar .f16) (TM * 64))
  ShaderM.sharedNamed "shared_B" (.array (.scalar .f16) (TN * 64))
  ShaderM.sharedNamed "shared_C" (.array (.scalar .f32) (TM * TN * 64))
  ShaderM.declareMatrixLeftArray  "Ax" .f16 8 8 1 Exp.subgroupMatrixZeroLeft
  ShaderM.declareMatrixRightArray "Bx" .f16 8 8 1 Exp.subgroupMatrixZeroRight
  ShaderM.declareMatrixResultArray "Cx" .f32 8 8 1 Exp.subgroupMatrixZeroResult
  let rowBase := Exp.mul wgM (Exp.litU32 (8 * TM))
  let colBase := Exp.mul wgN (Exp.litU32 (8 * TN))
  let sg := Exp.div tid (Exp.litU32 32)          -- subgroup id 0..TM·TN-1
  let myTi := Exp.div sg (Exp.litU32 TN)
  let myTj := Exp.mod sg (Exp.litU32 TN)
  let aIters := (TM * 64 + nThreads - 1) / nThreads
  let bIters := (TN * 32 + nThreads - 1) / nThreads
  let numKTiles := config.K / 8
  for kTile in [0:numKTiles] do
    let kBase := Exp.litU32 (kTile * 8)
    -- cooperative A load (TM·64 f16)
    for s in [0:aIters] do
      let flat := Exp.add tid (Exp.litU32 (s * nThreads))
      ShaderM.if_ (Exp.lt flat (Exp.litU32 (TM * 64))) (do
        let tile := Exp.div flat (Exp.litU32 64)
        let within := Exp.mod flat (Exp.litU32 64)
        let row := Exp.add (Exp.add rowBase (Exp.mul tile (Exp.litU32 8))) (Exp.div within (Exp.litU32 8))
        let col := Exp.add kBase (Exp.mod within (Exp.litU32 8))
        let xf32 ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.M * config.K) "a"
          (Exp.add (Exp.mul row (Exp.litU32 config.K)) col)
        ShaderM.writeWorkgroup (ty := .scalar .f16) "shared_A" flat (Exp.toF16 xf32)) (pure ())
    -- cooperative B load (TN·32 u32 → TN·64 f16)
    for s in [0:bIters] do
      let flatU := Exp.add tid (Exp.litU32 (s * nThreads))
      ShaderM.if_ (Exp.lt flatU (Exp.litU32 (TN * 32))) (do
        let tile := Exp.div flatU (Exp.litU32 32)
        let withinU := Exp.mod flatU (Exp.litU32 32)
        let ni := Exp.div withinU (Exp.litU32 4)
        let kLocal := Exp.mul (Exp.mod withinU (Exp.litU32 4)) (Exp.litU32 2)
        let row := Exp.add (Exp.add colBase (Exp.mul tile (Exp.litU32 8))) ni
        let kPairGlobal := Exp.div (Exp.add kBase kLocal) (Exp.litU32 2)
        let packed ← ShaderM.readBuffer (ty := .scalar .u32) (n := config.N * packedK) "b"
          (Exp.add (Exp.mul row (Exp.litU32 packedK)) kPairGlobal)
        let unpacked := Exp.unpack2x16float packed
        let base := Exp.mul tile (Exp.litU32 64)
        ShaderM.writeWorkgroup (ty := .scalar .f16) "shared_B"
          (Exp.add base (Exp.add (Exp.mul kLocal (Exp.litU32 8)) ni)) (Exp.toF16 (Exp.vecX unpacked))
        ShaderM.writeWorkgroup (ty := .scalar .f16) "shared_B"
          (Exp.add base (Exp.add (Exp.mul (Exp.add kLocal (Exp.litU32 1)) (Exp.litU32 8)) ni)) (Exp.toF16 (Exp.vecY unpacked))) (pure ())
    ShaderM.barrier
    -- each subgroup: load its A row-tile + B col-tile from shared, one MMA into its own Cx
    ShaderM.loadMatrixLeft (st := .f16) (m := 8) (k := 8) "Ax" 0 "shared_A" (Exp.mul myTi (Exp.litU32 64)) (Exp.litU32 8)
    ShaderM.loadMatrixRight (st := .f16) (k := 8) (n := 8) "Bx" 0 "shared_B" (Exp.mul myTj (Exp.litU32 64)) (Exp.litU32 8)
    ShaderM.matrixMultiplyAccumulateMixed (inSt := .f16) (outSt := .f32) (m := 8) (k := 8) (n := 8) "Cx" 0 "Ax" 0 "Bx" 0
    ShaderM.barrier
  ShaderM.storeMatrixResult (st := .f32) (m := 8) (n := 8) "Cx" 0 "shared_C" (Exp.mul sg (Exp.litU32 64)) (Exp.litU32 8)
  ShaderM.barrier
  -- cooperative copy shared_C → C
  let cIters := (TM * TN * 64 + nThreads - 1) / nThreads
  for s in [0:cIters] do
    let flat := Exp.add tid (Exp.litU32 (s * nThreads))
    ShaderM.if_ (Exp.lt flat (Exp.litU32 (TM * TN * 64))) (do
      let tileC := Exp.div flat (Exp.litU32 64)
      let within := Exp.mod flat (Exp.litU32 64)
      let ti := Exp.div tileC (Exp.litU32 TN)
      let tj := Exp.mod tileC (Exp.litU32 TN)
      let row := Exp.add (Exp.add rowBase (Exp.mul ti (Exp.litU32 8))) (Exp.div within (Exp.litU32 8))
      let col := Exp.add (Exp.add colBase (Exp.mul tj (Exp.litU32 8))) (Exp.mod within (Exp.litU32 8))
      let v ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := TM * TN * 64) "shared_C" flat
      ShaderM.writeBuffer (ty := .scalar .f32) "c" (Exp.add (Exp.mul row (Exp.litU32 config.N)) col) v) (pure ())

/-- K-batched 8×8 WMMA: like the naive 8×8 kernel (one 32-lane subgroup, ONE 8×8 output tile, ONE
    result matrix — no register/subgroup pressure) but loads BK K-tiles into shared per barrier and
    does BK back-to-back MMAs, cutting the barrier count BK×. A: f32 [M,K]; B: u32 f16 [N,K/2];
    C: f32 [M,N]. Requires K % (8·BK) = 0. Grid: (N/8, M/8) × 32. -/
def matMulTransposeF16WMMA8x8KbKernel (config : Config) (BK : Nat) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let wgN := Exp.vec3X wid
  let wgM := Exp.vec3Y wid
  let tid := Exp.vec3X lid
  let packedK := config.K / 2
  let _a ← ShaderM.declareInputBuffer "a" (.array (.scalar .f32) (config.M * config.K))
  let _b ← ShaderM.declareInputBuffer "b" (.array (.scalar .u32) (config.N * packedK))
  let _c ← ShaderM.declareOutputBuffer "c" (.array (.scalar .f32) (config.M * config.N))
  ShaderM.sharedNamed "shared_A" (.array (.scalar .f16) (BK * 64))
  ShaderM.sharedNamed "shared_B" (.array (.scalar .f16) (BK * 64))
  ShaderM.sharedNamed "shared_C" (.array (.scalar .f32) 64)
  ShaderM.declareMatrixLeftArray  "Ax" .f16 8 8 1 Exp.subgroupMatrixZeroLeft
  ShaderM.declareMatrixRightArray "Bx" .f16 8 8 1 Exp.subgroupMatrixZeroRight
  ShaderM.declareMatrixResultArray "Cx" .f32 8 8 1 Exp.subgroupMatrixZeroResult
  let rowBase := Exp.mul wgM (Exp.litU32 8)
  let colBase := Exp.mul wgN (Exp.litU32 8)
  let numBatches := config.K / (8 * BK)
  for batch in [0:numBatches] do
    let kBatchBase := batch * 8 * BK
    -- load BK A-tiles + BK B-tiles into shared
    for kk in [0:BK] do
      let kBase := Exp.litU32 (kBatchBase + kk * 8)
      for s in [0:2] do
        let e := Exp.add tid (Exp.litU32 (s * 32))
        let row := Exp.add rowBase (Exp.div e (Exp.litU32 8))
        let col := Exp.add kBase (Exp.mod e (Exp.litU32 8))
        let xf32 ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.M * config.K) "a"
          (Exp.add (Exp.mul row (Exp.litU32 config.K)) col)
        ShaderM.writeWorkgroup (ty := .scalar .f16) "shared_A" (Exp.add (Exp.litU32 (kk * 64)) e) (Exp.toF16 xf32)
      let pairIdx := Exp.mod tid (Exp.litU32 4)
      let ni := Exp.div tid (Exp.litU32 4)
      let row := Exp.add colBase ni
      let kLocal := Exp.mul pairIdx (Exp.litU32 2)
      let packed ← ShaderM.readBuffer (ty := .scalar .u32) (n := config.N * packedK) "b"
        (Exp.add (Exp.mul row (Exp.litU32 packedK)) (Exp.div (Exp.add kBase kLocal) (Exp.litU32 2)))
      let unpacked := Exp.unpack2x16float packed
      let base := Exp.litU32 (kk * 64)
      ShaderM.writeWorkgroup (ty := .scalar .f16) "shared_B"
        (Exp.add base (Exp.add (Exp.mul kLocal (Exp.litU32 8)) ni)) (Exp.toF16 (Exp.vecX unpacked))
      ShaderM.writeWorkgroup (ty := .scalar .f16) "shared_B"
        (Exp.add base (Exp.add (Exp.mul (Exp.add kLocal (Exp.litU32 1)) (Exp.litU32 8)) ni)) (Exp.toF16 (Exp.vecY unpacked))
    ShaderM.barrier
    for kk in [0:BK] do
      ShaderM.loadMatrixLeft (st := .f16) (m := 8) (k := 8) "Ax" 0 "shared_A" (Exp.litU32 (kk * 64)) (Exp.litU32 8)
      ShaderM.loadMatrixRight (st := .f16) (k := 8) (n := 8) "Bx" 0 "shared_B" (Exp.litU32 (kk * 64)) (Exp.litU32 8)
      ShaderM.matrixMultiplyAccumulateMixed (inSt := .f16) (outSt := .f32) (m := 8) (k := 8) (n := 8) "Cx" 0 "Ax" 0 "Bx" 0
    ShaderM.barrier
  ShaderM.storeMatrixResult (st := .f32) (m := 8) (n := 8) "Cx" 0 "shared_C" (Exp.litU32 0) (Exp.litU32 8)
  ShaderM.barrier
  for s in [0:2] do
    let e := Exp.add tid (Exp.litU32 (s * 32))
    let row := Exp.add rowBase (Exp.div e (Exp.litU32 8))
    let col := Exp.add colBase (Exp.mod e (Exp.litU32 8))
    let v ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 64) "shared_C" e
    ShaderM.writeBuffer (ty := .scalar .f32) "c" (Exp.add (Exp.mul row (Exp.litU32 config.N)) col) v

/-! ## Scaled Matrix Multiply (for Attention) -/

/-- Scaled matrix multiply: C = (A @ B) / scale

    Used in attention: scores = (Q @ K^T) / sqrt(d_k)

    @param config Matrix dimensions
    @param scale Scaling factor (e.g., 1/sqrt(d_k))
-/
def scaledMatMulTransposeKernel (config : Config) (scale : Float) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let totalElements := config.M * config.N
  let inBounds := Exp.lt idx (Exp.litU32 totalElements)

  let _a ← ShaderM.declareInputBuffer "a" (.array (.scalar .f32) (config.M * config.K))
  let _b ← ShaderM.declareInputBuffer "b" (.array (.scalar .f32) (config.N * config.K))
  let _c ← ShaderM.declareOutputBuffer "c" (.array (.scalar .f32) totalElements)

  let row := Exp.div idx (Exp.litU32 config.N)
  let col := Exp.mod idx (Exp.litU32 config.N)

  let (sumName, sum) ← ShaderM.varRef (.scalar .f32) (Exp.litF32 0.0)

  ShaderM.loop (Exp.litU32 0) (Exp.litU32 config.K) (Exp.litU32 1) fun k => do
    let aIdx := Exp.add (Exp.mul row (Exp.litU32 config.K)) k
    let bIdx := Exp.add (Exp.mul col (Exp.litU32 config.K)) k

    let aVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.M * config.K) "a" aIdx
    let bVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.N * config.K) "b" bIdx

    let prod := Exp.mul aVal bVal
    ShaderM.assign sumName (Exp.add sum prod)

  -- Apply scaling
  let scaled := Exp.mul sum (Exp.litF32 scale)

  let result := Exp.select inBounds scaled (Exp.litF32 0.0)
  ShaderM.writeBuffer (ty := .scalar .f32) "c" idx result

/-! ## Batched Matrix Multiply -/

/-- Batched matrix multiply: C[b] = A[b] @ B[b] for each batch b

    A: [batch, M, K]
    B: [batch, K, N]
    C: [batch, M, N]

    Common in multi-head attention where batch includes both
    actual batch dimension and number of heads.

    @param config Matrix dimensions (per batch)
    @param batchSize Number of batches
-/
def batchedMatMulKernel (config : Config) (batchSize : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let elementsPerBatch := config.M * config.N
  let totalElements := batchSize * elementsPerBatch
  let inBounds := Exp.lt idx (Exp.litU32 totalElements)

  let _a ← ShaderM.declareInputBuffer "a" (.array (.scalar .f32) (batchSize * config.M * config.K))
  let _b ← ShaderM.declareInputBuffer "b" (.array (.scalar .f32) (batchSize * config.K * config.N))
  let _c ← ShaderM.declareOutputBuffer "c" (.array (.scalar .f32) totalElements)

  -- Decompose: batch, row, col
  let batch := Exp.div idx (Exp.litU32 elementsPerBatch)
  let localIdx := Exp.mod idx (Exp.litU32 elementsPerBatch)
  let row := Exp.div localIdx (Exp.litU32 config.N)
  let col := Exp.mod localIdx (Exp.litU32 config.N)

  let (sumName, sum) ← ShaderM.varRef (.scalar .f32) (Exp.litF32 0.0)

  ShaderM.loop (Exp.litU32 0) (Exp.litU32 config.K) (Exp.litU32 1) fun k => do
    -- A[batch, row, k]
    let aBase := Exp.mul batch (Exp.litU32 (config.M * config.K))
    let aOffset := Exp.add (Exp.mul row (Exp.litU32 config.K)) k
    let aIdx := Exp.add aBase aOffset

    -- B[batch, k, col]
    let bBase := Exp.mul batch (Exp.litU32 (config.K * config.N))
    let bOffset := Exp.add (Exp.mul k (Exp.litU32 config.N)) col
    let bIdx := Exp.add bBase bOffset

    let aVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := batchSize * config.M * config.K) "a" aIdx
    let bVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := batchSize * config.K * config.N) "b" bIdx

    let prod := Exp.mul aVal bVal
    ShaderM.assign sumName (Exp.add sum prod)

  let result := Exp.select inBounds sum (Exp.litF32 0.0)
  ShaderM.writeBuffer (ty := .scalar .f32) "c" idx result

/-- Batched scaled transposed matmul: C[b] = (A[b] @ B[b]^T) * scale

    A: [batch, M, K]
    B: [batch, N, K]  (transposed: B is stored row-major as [N, K])
    C: [batch, M, N]

    Used for attention scores: scores = (Q @ K^T) / sqrt(d_k)
-/
def batchedScaledMatMulTransposeKernel (config : Config) (batchSize : Nat) (scale : Float) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let elementsPerBatch := config.M * config.N
  let totalElements := batchSize * elementsPerBatch
  let inBounds := Exp.lt idx (Exp.litU32 totalElements)

  let _a ← ShaderM.declareInputBuffer "a" (.array (.scalar .f32) (batchSize * config.M * config.K))
  let _b ← ShaderM.declareInputBuffer "b" (.array (.scalar .f32) (batchSize * config.N * config.K))
  let _c ← ShaderM.declareOutputBuffer "c" (.array (.scalar .f32) totalElements)

  let batch := Exp.div idx (Exp.litU32 elementsPerBatch)
  let localIdx := Exp.mod idx (Exp.litU32 elementsPerBatch)
  let row := Exp.div localIdx (Exp.litU32 config.N)
  let col := Exp.mod localIdx (Exp.litU32 config.N)

  let (sumName, sum) ← ShaderM.varRef (.scalar .f32) (Exp.litF32 0.0)

  ShaderM.loop (Exp.litU32 0) (Exp.litU32 config.K) (Exp.litU32 1) fun k => do
    -- A[batch, row, k]
    let aBase := Exp.mul batch (Exp.litU32 (config.M * config.K))
    let aOffset := Exp.add (Exp.mul row (Exp.litU32 config.K)) k
    let aIdx := Exp.add aBase aOffset

    -- B^T: B[batch, col, k] (B stored as [batch, N, K])
    let bBase := Exp.mul batch (Exp.litU32 (config.N * config.K))
    let bOffset := Exp.add (Exp.mul col (Exp.litU32 config.K)) k
    let bIdx := Exp.add bBase bOffset

    let aVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := batchSize * config.M * config.K) "a" aIdx
    let bVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := batchSize * config.N * config.K) "b" bIdx

    let prod := Exp.mul aVal bVal
    ShaderM.assign sumName (Exp.add sum prod)

  let scaled := Exp.mul sum (Exp.litF32 scale)
  let result := Exp.select inBounds scaled (Exp.litF32 0.0)
  ShaderM.writeBuffer (ty := .scalar .f32) "c" idx result

/-! ## High-Level API -/

/-- Execute naive matrix multiply: C = A @ B

    @param device WebGPU device
    @param aBuf Matrix A [M, K]
    @param bBuf Matrix B [K, N]
    @param cBuf Output matrix C [M, N]
    @param config Matrix dimensions
-/
def executeMatMul [GPUBackend β] (ctx : β)
                  (aBuf bBuf cBuf : GPUBackend.Buf β)
                  (config : Config) : IO Unit := do
  logVerbose s!"[MatMul] Computing {config.M}×{config.K} @ {config.K}×{config.N}..."

  let shader := naiveMatMulKernel config
  let namedBuffers := [
    ("a", aBuf),
    ("b", bBuf),
    ("c", cBuf)
  ]

  let totalElements := config.M * config.N
  let execConfig := Hesper.ExecConfig.dispatch1D
    totalElements
    256

  GPUBackend.execute ctx shader namedBuffers execConfig
  logVerbose "[MatMul] ✓ Complete"

/-- Execute transposed matrix multiply: C = A @ B^T

    @param device WebGPU device
    @param aBuf Matrix A [M, K]
    @param bBuf Matrix B [N, K] (will be transposed)
    @param cBuf Output matrix C [M, N]
    @param config Matrix dimensions
-/
def executeMatMulTranspose [GPUBackend β] (ctx : β)
                           (aBuf bBuf cBuf : GPUBackend.Buf β)
                           (config : Config) : IO Unit := do
  logVerbose s!"[MatMul] Computing {config.M}×{config.K} @ ({config.N}×{config.K})^T..."

  let shader := matMulTransposeBKernel config
  let namedBuffers := [
    ("a", aBuf),
    ("b", bBuf),
    ("c", cBuf)
  ]

  let totalElements := config.M * config.N
  let execConfig := Hesper.ExecConfig.dispatch1D
    totalElements
    256

  GPUBackend.execute ctx shader namedBuffers execConfig
  logVerbose "[MatMul] ✓ Complete"

/-- Execute transposed matrix multiply with B in packed F16: C = A @ B^T

    B is stored as packed F16 (each u32 = 2 F16 values).
    Uses hardware unpack2x16float for 2x bandwidth reduction.
    Primary use: LM head with F16 embedding table.

    @param device WebGPU device
    @param aBuf Matrix A [M, K] in F32
    @param bF16Buf Matrix B [N, K] stored as packed F16 ([N, K/2] u32 values)
    @param cBuf Output matrix C [M, N] in F32
    @param config Matrix dimensions (K must be even)
-/
def executeMatMulTransposeF16 [GPUBackend β] (ctx : β)
                               (aBuf bF16Buf cBuf : GPUBackend.Buf β)
                               (config : Config) : IO Unit := do
  logVerbose s!"[MatMul] Computing F16: {config.M}×{config.K} @ ({config.N}×{config.K})^T..."

  let shader := matMulTransposeF16Kernel config
  let namedBuffers := [
    ("a", aBuf),
    ("b", bF16Buf),
    ("c", cBuf)
  ]

  let totalElements := config.M * config.N
  let execConfig := Hesper.ExecConfig.dispatch1D
    totalElements
    256

  let cacheKey : UInt64 := hash ("mmf16", config.M, config.N, config.K)
  GPUBackend.executeWithConfigCached ctx shader namedBuffers execConfig cacheKey (← IO.mkRef none)
  logVerbose "[MatMul] ✓ Complete (F16)"

/-- Execute optimized F16 transposed matmul with shared memory: C = A @ B^T

    Uses shared memory for A vector to eliminate redundant global reads.
    Significantly faster than `executeMatMulTransposeF16` for small M (especially M=1).

    Requirements: K must be divisible by 8

    @param device WebGPU device
    @param aBuf Matrix A [M, K] in F32
    @param bF16Buf Matrix B [N, K] stored as packed F16 ([N, K/2] u32 values)
    @param cBuf Output matrix C [M, N] in F32
    @param config Matrix dimensions (K must be divisible by 8)
-/
def executeMatMulTransposeF16Shared [GPUBackend β] (ctx : β)
                                     (aBuf bF16Buf cBuf : GPUBackend.Buf β)
                                     (config : Config) : IO Unit := do
  logVerbose s!"[MatMul] Computing F16-Shared: {config.M}×{config.K} @ ({config.N}×{config.K})^T..."

  let shader := matMulTransposeF16SharedKernel config
  let namedBuffers := [
    ("a", aBuf),
    ("b", bF16Buf),
    ("c", cBuf)
  ]

  let totalElements := config.M * config.N
  let execConfig := Hesper.ExecConfig.dispatch1D
    totalElements
    256

  let cacheKey : UInt64 := hash ("mmf16s", config.M, config.N, config.K)
  GPUBackend.executeWithConfigCached ctx shader namedBuffers execConfig cacheKey (← IO.mkRef none)
  logVerbose "[MatMul] ✓ Complete (F16-Shared)"

/-- Execute scaled transposed matrix multiply: C = (A @ B^T) / scale

    Used for attention scores: scores = (Q @ K^T) / sqrt(d_k)

    @param device WebGPU device
    @param aBuf Matrix A [M, K]
    @param bBuf Matrix B [N, K]
    @param cBuf Output matrix C [M, N]
    @param config Matrix dimensions
    @param scale Scaling factor
-/
def executeScaledMatMulTranspose [GPUBackend β] (ctx : β)
                                 (aBuf bBuf cBuf : GPUBackend.Buf β)
                                 (config : Config)
                                 (scale : Float) : IO Unit := do
  logVerbose s!"[MatMul] Computing ({config.M}×{config.K} @ {config.N}×{config.K}^T) / {scale}..."

  let shader := scaledMatMulTransposeKernel config scale
  let namedBuffers := [
    ("a", aBuf),
    ("b", bBuf),
    ("c", cBuf)
  ]

  let totalElements := config.M * config.N
  let execConfig := Hesper.ExecConfig.dispatch1D
    totalElements
    256

  GPUBackend.execute ctx shader namedBuffers execConfig
  logVerbose "[MatMul] ✓ Complete"

/-- Execute batched matrix multiply: C[b] = A[b] @ B[b]

    @param device WebGPU device
    @param aBuf Matrix A [batch, M, K]
    @param bBuf Matrix B [batch, K, N]
    @param cBuf Output matrix C [batch, M, N]
    @param config Matrix dimensions (per batch)
    @param batchSize Number of batches
-/
def executeBatchedMatMul [GPUBackend β] (ctx : β)
                         (aBuf bBuf cBuf : GPUBackend.Buf β)
                         (config : Config)
                         (batchSize : Nat) : IO Unit := do
  logVerbose s!"[MatMul] Batched multiply: {batchSize} × ({config.M}×{config.K} @ {config.K}×{config.N})..."

  let shader := batchedMatMulKernel config batchSize
  let namedBuffers := [
    ("a", aBuf),
    ("b", bBuf),
    ("c", cBuf)
  ]

  let totalElements := batchSize * config.M * config.N
  let execConfig := Hesper.ExecConfig.dispatch1D
    totalElements
    256

  GPUBackend.execute ctx shader namedBuffers execConfig
  logVerbose "[MatMul] ✓ Complete"

/-- Execute batched scaled transposed matmul: C[b] = (A[b] @ B[b]^T) * scale

    A: [batch, M, K], B: [batch, N, K] → C: [batch, M, N]
    Used for attention: scores = (Q @ K^T) / sqrt(d_k)
-/
def executeBatchedScaledMatMulTranspose [GPUBackend β] (ctx : β)
                                        (aBuf bBuf cBuf : GPUBackend.Buf β)
                                        (config : Config)
                                        (batchSize : Nat)
                                        (scale : Float) : IO Unit := do
  logVerbose s!"[MatMul] Batched scaled transpose: {batchSize} × ({config.M}×{config.K} @ ({config.N}×{config.K})^T) * {scale}..."

  let shader := batchedScaledMatMulTransposeKernel config batchSize scale
  let namedBuffers := [
    ("a", aBuf),
    ("b", bBuf),
    ("c", cBuf)
  ]

  let totalElements := batchSize * config.M * config.N
  let execConfig := Hesper.ExecConfig.dispatch1D
    totalElements
    256

  GPUBackend.execute ctx shader namedBuffers execConfig
  logVerbose "[MatMul] ✓ Complete"

end Hesper.WGSL.MatMul
