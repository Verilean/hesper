import Hesper.WGSL.Monad
import Hesper.WGSL.Execute
import Hesper.WGSL.Exp
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
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
open Hesper.WebGPU
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
def executeMatMul (device : Device)
                  (aBuf bBuf cBuf : Buffer)
                  (config : Config) : IO Unit := do
  logVerbose s!"[MatMul] Computing {config.M}×{config.K} @ {config.K}×{config.N}..."

  let shader := naiveMatMulKernel config
  let namedBuffers := [
    ("a", aBuf),
    ("b", bBuf),
    ("c", cBuf)
  ]

  let totalElements := config.M * config.N
  let execConfig := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D
    totalElements
    256

  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig
  logVerbose "[MatMul] ✓ Complete"

/-- Execute transposed matrix multiply: C = A @ B^T

    @param device WebGPU device
    @param aBuf Matrix A [M, K]
    @param bBuf Matrix B [N, K] (will be transposed)
    @param cBuf Output matrix C [M, N]
    @param config Matrix dimensions
-/
def executeMatMulTranspose (device : Device)
                           (aBuf bBuf cBuf : Buffer)
                           (config : Config) : IO Unit := do
  logVerbose s!"[MatMul] Computing {config.M}×{config.K} @ ({config.N}×{config.K})^T..."

  let shader := matMulTransposeBKernel config
  let namedBuffers := [
    ("a", aBuf),
    ("b", bBuf),
    ("c", cBuf)
  ]

  let totalElements := config.M * config.N
  let execConfig := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D
    totalElements
    256

  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig
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
def executeMatMulTransposeF16 (device : Device)
                               (aBuf bF16Buf cBuf : Buffer)
                               (config : Config) : IO Unit := do
  logVerbose s!"[MatMul] Computing F16: {config.M}×{config.K} @ ({config.N}×{config.K})^T..."

  let shader := matMulTransposeF16Kernel config
  let namedBuffers := [
    ("a", aBuf),
    ("b", bF16Buf),
    ("c", cBuf)
  ]

  let totalElements := config.M * config.N
  let execConfig := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D
    totalElements
    256

  let cacheKey : UInt64 := hash ("mmf16", config.M, config.N, config.K)
  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig (some cacheKey)
  logVerbose "[MatMul] ✓ Complete (F16)"

/-- Execute scaled transposed matrix multiply: C = (A @ B^T) / scale

    Used for attention scores: scores = (Q @ K^T) / sqrt(d_k)

    @param device WebGPU device
    @param aBuf Matrix A [M, K]
    @param bBuf Matrix B [N, K]
    @param cBuf Output matrix C [M, N]
    @param config Matrix dimensions
    @param scale Scaling factor
-/
def executeScaledMatMulTranspose (device : Device)
                                 (aBuf bBuf cBuf : Buffer)
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
  let execConfig := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D
    totalElements
    256

  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig
  logVerbose "[MatMul] ✓ Complete"

/-- Execute batched matrix multiply: C[b] = A[b] @ B[b]

    @param device WebGPU device
    @param aBuf Matrix A [batch, M, K]
    @param bBuf Matrix B [batch, K, N]
    @param cBuf Output matrix C [batch, M, N]
    @param config Matrix dimensions (per batch)
    @param batchSize Number of batches
-/
def executeBatchedMatMul (device : Device)
                         (aBuf bBuf cBuf : Buffer)
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
  let execConfig := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D
    totalElements
    256

  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig
  logVerbose "[MatMul] ✓ Complete"

/-- Execute batched scaled transposed matmul: C[b] = (A[b] @ B[b]^T) * scale

    A: [batch, M, K], B: [batch, N, K] → C: [batch, M, N]
    Used for attention: scores = (Q @ K^T) / sqrt(d_k)
-/
def executeBatchedScaledMatMulTranspose (device : Device)
                                        (aBuf bBuf cBuf : Buffer)
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
  let execConfig := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D
    totalElements
    256

  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig
  logVerbose "[MatMul] ✓ Complete"

end Hesper.WGSL.MatMul
