import Hesper.WGSL.Monad
import Hesper.WGSL.Execute
import Hesper.WGSL.Exp
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.Logging

/-!
# RoPE - Rotary Position Embeddings

Implements Rotary Position Embeddings (RoPE) as used in LLaMA, GPT-NeoX, and BitNet.

## Mathematical Definition

RoPE encodes position information by rotating query/key vectors in complex space:

```
NeoX split-half style (used by BitNet b1.58):
For position m and dimension i (0 ≤ i < headDim/2):
θᵢ = m × base^(-2i/headDim)  where base = 500000.0 for BitNet

Dimension pairs: (x[i], x[i + headDim/2])

Applied as:
x'[i]             = x[i] × cos(θ) - x[i + headDim/2] × sin(θ)
x'[i + headDim/2] = x[i] × sin(θ) + x[i + headDim/2] × cos(θ)
```

## Why RoPE?

**Traditional positional encoding** (sinusoidal):
- Absolute positions → fixed vectors added to embeddings
- No relative position information in attention scores

**RoPE advantages**:
1. **Relative position encoding**: Attention scores depend on relative positions
2. **Efficient**: No learned parameters, just rotation
3. **Extrapolation**: Can handle longer sequences than training length
4. **Mathematically elegant**: Complex number rotation in frequency space

## Attention Score Property

For queries Q and keys K at positions m and n:
```
(RoPE(Q, m) · RoPE(K, n)) = f(Q, K, m-n)
```
The dot product depends only on relative position (m-n), not absolute positions!

## Implementation Strategy

**Precomputed approach** (llama.cpp):
```
1. Precompute cos(θ) and sin(θ) for all positions and dimensions
2. Store in lookup tables: cos_cache[pos][dim], sin_cache[pos][dim]
3. During forward pass: read from cache and apply rotation
```

**On-the-fly computation** (this implementation):
```
1. Compute θ = pos × base^(-2i/d) in shader
2. Compute cos(θ) and sin(θ) using GPU intrinsics
3. Apply rotation to adjacent dimension pairs
```

## References
- RoFormer paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
- LLaMA: https://github.com/facebookresearch/llama (rope_forward)
- llama.cpp: ggml/src/ggml.c (ggml_rope_impl)
-/

namespace Hesper.Layers.RoPE

open Hesper.WGSL
open Hesper.WGSL.Monad
open Hesper.WebGPU
open Hesper.Logging (logVerbose)

/-! ## Layer Configuration -/

/-- RoPE configuration -/
structure Config where
  dim : Nat               -- Embedding dimension (must be even)
  maxSeqLen : Nat         -- Maximum sequence length
  base : Float := 10000.0 -- Frequency base (theta_base)
  deriving Repr

/-! ## Helper Functions -/

/-- Compute rotation angle θ for position and dimension pair
    θᵢ = pos × base^(-2i/d)
-/
def computeTheta (pos : Exp (.scalar .f32)) (dimPair : Exp (.scalar .u32))
                 (totalDim : Nat) (base : Float) : Exp (.scalar .f32) :=
  -- dimPair is i/2 (since we process pairs)
  -- Compute: pos × base^(-2×dimPair/totalDim)
  let dimPairF32 := Exp.toF32 dimPair
  let exponent := Exp.div (Exp.mul (Exp.litF32 2.0) dimPairF32) (Exp.litF32 totalDim.toFloat)
  let freqInv := Exp.pow (Exp.litF32 base) (Exp.neg exponent)
  Exp.mul pos freqInv

/-! ## GPU Kernel Implementation -/

/-- RoPE kernel: Apply rotary embeddings to query or key tensor

    **Input shape**: [batch, seq_len, num_heads, head_dim]
    **Output shape**: [batch, seq_len, num_heads, head_dim]

    **Algorithm**:
    ```
    for each position pos in sequence:
      for each dimension pair (2i, 2i+1):
        θ = pos × base^(-2i/d)
        x'[2i]   = x[2i]   × cos(θ) - x[2i+1] × sin(θ)
        x'[2i+1] = x[2i]   × sin(θ) + x[2i+1] × cos(θ)
    ```

    **Workgroup strategy**: Each thread handles one dimension pair for one token

    @param config RoPE configuration
    @param batchSize Batch size
    @param seqLen Current sequence length
    @param numHeads Number of attention heads
-/
def ropeKernel (config : Config) (batchSize seqLen numHeads : Nat) (headDimOverride : Nat := 0) (posOffset : Nat := 0) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  -- Flatten index to: [batch, seq, head, dim_pair]
  let headDim := if headDimOverride > 0 then headDimOverride else config.dim / numHeads
  let dimPairs := headDim / 2  -- Process pairs
  let totalElements := batchSize * seqLen * numHeads * dimPairs

  -- Bounds check
  let inBounds := Exp.lt idx (Exp.litU32 totalElements)

  -- Declare buffers
  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) (batchSize * seqLen * numHeads * headDim))
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) (batchSize * seqLen * numHeads * headDim))

  -- Decompose flattened index
  let dimPair := Exp.mod idx (Exp.litU32 dimPairs)
  let tmp1 := Exp.div idx (Exp.litU32 dimPairs)
  let head := Exp.mod tmp1 (Exp.litU32 numHeads)
  let tmp2 := Exp.div tmp1 (Exp.litU32 numHeads)
  let pos := Exp.mod tmp2 (Exp.litU32 seqLen)
  let batch := Exp.div tmp2 (Exp.litU32 seqLen)

  -- Compute theta for this position and dimension (with offset for KV cache)
  let posF32 := Exp.toF32 (Exp.add pos (Exp.litU32 posOffset))
  let theta := computeTheta posF32 dimPair headDim config.base

  -- Compute cos and sin
  let cosTheta := Exp.cos theta
  let sinTheta := Exp.sin theta

  -- Calculate input indices for NeoX split-half dimension pair (i, i + halfDim)
  -- BitNet uses LLAMA_ROPE_TYPE_NEOX: pairs are (x[i], x[i + headDim/2])
  let halfDim := headDim / 2
  let dim0 := dimPair
  let dim1 := Exp.add dimPair (Exp.litU32 halfDim)

  -- Flatten input index: batch * (seqLen * numHeads * headDim) + seq * (numHeads * headDim) + head * headDim + dim
  let batchOffset := Exp.mul batch (Exp.litU32 (seqLen * numHeads * headDim))
  let seqOffset := Exp.mul pos (Exp.litU32 (numHeads * headDim))
  let headOffset := Exp.mul head (Exp.litU32 headDim)
  let baseIdx := Exp.add (Exp.add batchOffset seqOffset) headOffset

  let idx0 := Exp.add baseIdx dim0
  let idx1 := Exp.add baseIdx dim1

  -- Read input values
  let x0 ← ShaderM.readBuffer (ty := .scalar .f32) (n := batchSize * seqLen * numHeads * headDim) "input" idx0
  let x1 ← ShaderM.readBuffer (ty := .scalar .f32) (n := batchSize * seqLen * numHeads * headDim) "input" idx1

  -- Apply rotation
  let x0_new := Exp.sub (Exp.mul x0 cosTheta) (Exp.mul x1 sinTheta)
  let x1_new := Exp.add (Exp.mul x0 sinTheta) (Exp.mul x1 cosTheta)

  -- Write output (conditional on bounds)
  let result0 := Exp.select inBounds x0_new (Exp.litF32 0.0)
  let result1 := Exp.select inBounds x1_new (Exp.litF32 0.0)

  ShaderM.writeBuffer (ty := .scalar .f32) "output" idx0 result0
  ShaderM.writeBuffer (ty := .scalar .f32) "output" idx1 result1

/-- RoPE kernel with dynamic posOffset from params buffer.
    Produces identical WGSL regardless of position, enabling pipeline caching.
    Params buffer layout: [posOffset: u32] (4 bytes minimum, but shares attention params buffer)
    For single-token inference: batchSize=1, seqLen=1, reads posOffset from params[0].
-/
def ropeKernelDynamic (config : Config) (batchSize seqLen numHeads : Nat) (headDimOverride : Nat := 0) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let headDim := if headDimOverride > 0 then headDimOverride else config.dim / numHeads
  let dimPairs := headDim / 2
  let totalElements := batchSize * seqLen * numHeads * dimPairs

  let inBounds := Exp.lt idx (Exp.litU32 totalElements)

  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) (batchSize * seqLen * numHeads * headDim))
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) (batchSize * seqLen * numHeads * headDim))
  let _params ← ShaderM.declareInputBuffer "params" (.array (.scalar .u32) 2)

  -- Read posOffset from params buffer (params[0] = pos)
  let posOffset ← ShaderM.readBuffer (ty := .scalar .u32) (n := 2) "params" (Exp.litU32 0)

  let dimPair := Exp.mod idx (Exp.litU32 dimPairs)
  let tmp1 := Exp.div idx (Exp.litU32 dimPairs)
  let head := Exp.mod tmp1 (Exp.litU32 numHeads)
  let tmp2 := Exp.div tmp1 (Exp.litU32 numHeads)
  let pos := Exp.mod tmp2 (Exp.litU32 seqLen)
  let batch := Exp.div tmp2 (Exp.litU32 seqLen)

  let posF32 := Exp.toF32 (Exp.add pos posOffset)
  let theta := computeTheta posF32 dimPair headDim config.base

  let cosTheta := Exp.cos theta
  let sinTheta := Exp.sin theta

  let halfDim := headDim / 2
  let dim0 := dimPair
  let dim1 := Exp.add dimPair (Exp.litU32 halfDim)

  let batchOffset := Exp.mul batch (Exp.litU32 (seqLen * numHeads * headDim))
  let seqOffset := Exp.mul pos (Exp.litU32 (numHeads * headDim))
  let headOffset := Exp.mul head (Exp.litU32 headDim)
  let baseIdx := Exp.add (Exp.add batchOffset seqOffset) headOffset

  let idx0 := Exp.add baseIdx dim0
  let idx1 := Exp.add baseIdx dim1

  let x0 ← ShaderM.readBuffer (ty := .scalar .f32) (n := batchSize * seqLen * numHeads * headDim) "input" idx0
  let x1 ← ShaderM.readBuffer (ty := .scalar .f32) (n := batchSize * seqLen * numHeads * headDim) "input" idx1

  let x0_new := Exp.sub (Exp.mul x0 cosTheta) (Exp.mul x1 sinTheta)
  let x1_new := Exp.add (Exp.mul x0 sinTheta) (Exp.mul x1 cosTheta)

  let result0 := Exp.select inBounds x0_new (Exp.litF32 0.0)
  let result1 := Exp.select inBounds x1_new (Exp.litF32 0.0)

  ShaderM.writeBuffer (ty := .scalar .f32) "output" idx0 result0
  ShaderM.writeBuffer (ty := .scalar .f32) "output" idx1 result1

/-! ## Optimized: Cached RoPE (Future) -/

/-- Precomputed RoPE: Use cached cos/sin values

    This is more efficient for long sequences where we repeatedly
    apply RoPE to the same positions. Common in inference with KV caching.

    **Approach**:
    1. Precompute cos(θ) and sin(θ) for all positions and dimensions
    2. Store in buffers: cos_cache[max_seq_len][head_dim/2], sin_cache[...]
    3. During forward: lookup instead of compute

    @param config RoPE configuration
    @param batchSize Batch size
    @param seqLen Current sequence length
    @param numHeads Number of attention heads
-/
def ropeCachedKernel (config : Config) (batchSize seqLen numHeads : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let headDim := config.dim / numHeads
  let dimPairs := headDim / 2
  let totalElements := batchSize * seqLen * numHeads * dimPairs

  let inBounds := Exp.lt idx (Exp.litU32 totalElements)

  -- Declare buffers
  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) (batchSize * seqLen * numHeads * headDim))
  let _cosCache ← ShaderM.declareInputBuffer "cos_cache" (.array (.scalar .f32) (config.maxSeqLen * dimPairs))
  let _sinCache ← ShaderM.declareInputBuffer "sin_cache" (.array (.scalar .f32) (config.maxSeqLen * dimPairs))
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) (batchSize * seqLen * numHeads * headDim))

  -- Decompose index
  let dimPair := Exp.mod idx (Exp.litU32 dimPairs)
  let tmp1 := Exp.div idx (Exp.litU32 dimPairs)
  let head := Exp.mod tmp1 (Exp.litU32 numHeads)
  let tmp2 := Exp.div tmp1 (Exp.litU32 numHeads)
  let pos := Exp.mod tmp2 (Exp.litU32 seqLen)
  let batch := Exp.div tmp2 (Exp.litU32 seqLen)

  -- Lookup cached cos/sin
  let cacheIdx := Exp.add (Exp.mul pos (Exp.litU32 dimPairs)) dimPair
  let cosTheta ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.maxSeqLen * dimPairs) "cos_cache" cacheIdx
  let sinTheta ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.maxSeqLen * dimPairs) "sin_cache" cacheIdx

  -- Calculate input indices
  let dim0 := Exp.mul dimPair (Exp.litU32 2)
  let dim1 := Exp.add dim0 (Exp.litU32 1)

  let batchOffset := Exp.mul batch (Exp.litU32 (seqLen * numHeads * headDim))
  let seqOffset := Exp.mul pos (Exp.litU32 (numHeads * headDim))
  let headOffset := Exp.mul head (Exp.litU32 headDim)
  let baseIdx := Exp.add (Exp.add batchOffset seqOffset) headOffset

  let idx0 := Exp.add baseIdx dim0
  let idx1 := Exp.add baseIdx dim1

  -- Read and rotate
  let x0 ← ShaderM.readBuffer (ty := .scalar .f32) (n := batchSize * seqLen * numHeads * headDim) "input" idx0
  let x1 ← ShaderM.readBuffer (ty := .scalar .f32) (n := batchSize * seqLen * numHeads * headDim) "input" idx1

  let x0_new := Exp.sub (Exp.mul x0 cosTheta) (Exp.mul x1 sinTheta)
  let x1_new := Exp.add (Exp.mul x0 sinTheta) (Exp.mul x1 cosTheta)

  -- Write output
  let result0 := Exp.select inBounds x0_new (Exp.litF32 0.0)
  let result1 := Exp.select inBounds x1_new (Exp.litF32 0.0)

  ShaderM.writeBuffer (ty := .scalar .f32) "output" idx0 result0
  ShaderM.writeBuffer (ty := .scalar .f32) "output" idx1 result1

/-! ## High-Level API -/

/-- RoPE layer structure -/
structure RoPE where
  config : Config

/-- Create RoPE layer (no learned parameters)

    @param config RoPE configuration
-/
def create (config : Config) : IO RoPE := do
  if config.dim % 2 ≠ 0 then
    throw $ IO.userError s!"RoPE dim must be even, got {config.dim}"

  logVerbose s!"[RoPE] Creating layer: dim={config.dim}, max_seq_len={config.maxSeqLen}, base={config.base}"
  pure { config }

/-- Apply RoPE to query or key tensor

    @param device WebGPU device
    @param layer RoPE layer
    @param inputBuf GPU buffer [batch, seq_len, num_heads, head_dim]
    @param outputBuf GPU buffer for output (same shape)
    @param batchSize Batch size
    @param seqLen Current sequence length
    @param numHeads Number of attention heads
-/
def forward (device : Device) (layer : RoPE)
            (inputBuf outputBuf : Buffer)
            (batchSize seqLen numHeads : Nat) (headDim : Nat := 0) (posOffset : Nat := 0) : IO Unit := do
  -- If headDim is 0, derive from config.dim / numHeads
  let effectiveHeadDim := if headDim > 0 then headDim else layer.config.dim / numHeads
  logVerbose s!"[RoPE] Applying to batch={batchSize}, seq_len={seqLen}, heads={numHeads}, headDim={effectiveHeadDim}, posOffset={posOffset}"

  let dimPairs := effectiveHeadDim / 2
  let totalElements := batchSize * seqLen * numHeads * dimPairs

  let shader := ropeKernel layer.config batchSize seqLen numHeads effectiveHeadDim posOffset
  let namedBuffers := [
    ("input", inputBuf),
    ("output", outputBuf)
  ]

  let execConfig := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D
    totalElements
    256  -- Workgroup size

  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig
  logVerbose "[RoPE] ✓ Forward pass complete"

/-- Apply RoPE with dynamic posOffset from a params buffer.
    The params buffer must contain [posOffset: u32, ...] (posOffset at index 0).
    Produces identical WGSL across tokens → enables pipeline + bind group caching.
-/
def forwardDynamic (device : Device) (layer : RoPE)
            (inputBuf outputBuf paramsBuf : Buffer)
            (batchSize seqLen numHeads : Nat) (headDim : Nat := 0) : IO Unit := do
  let effectiveHeadDim := if headDim > 0 then headDim else layer.config.dim / numHeads
  logVerbose s!"[RoPE] Applying dynamic to batch={batchSize}, seq_len={seqLen}, heads={numHeads}, headDim={effectiveHeadDim}"

  let dimPairs := effectiveHeadDim / 2
  let totalElements := batchSize * seqLen * numHeads * dimPairs

  let shader := ropeKernelDynamic layer.config batchSize seqLen numHeads effectiveHeadDim
  let namedBuffers := [
    ("input", inputBuf),
    ("output", outputBuf),
    ("params", paramsBuf)
  ]

  let execConfig := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D
    totalElements
    256

  let cacheKey : UInt64 := hash ("rope_dyn", batchSize, seqLen, numHeads, effectiveHeadDim, layer.config.base.toBits)
  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig (some cacheKey)
  logVerbose "[RoPE] ✓ Dynamic forward pass complete"

/-- Apply cached RoPE (requires precomputed cos/sin buffers)

    @param device WebGPU device
    @param layer RoPE layer
    @param inputBuf GPU buffer [batch, seq_len, num_heads, head_dim]
    @param cosCacheBuf Precomputed cos values [max_seq_len, head_dim/2]
    @param sinCacheBuf Precomputed sin values [max_seq_len, head_dim/2]
    @param outputBuf GPU buffer for output
    @param batchSize Batch size
    @param seqLen Current sequence length
    @param numHeads Number of attention heads
-/
def forwardCached (device : Device) (layer : RoPE)
                  (inputBuf cosCacheBuf sinCacheBuf outputBuf : Buffer)
                  (batchSize seqLen numHeads : Nat) : IO Unit := do
  logVerbose s!"[RoPE] Applying cached RoPE: batch={batchSize}, seq_len={seqLen}"

  let headDim := layer.config.dim / numHeads
  let dimPairs := headDim / 2
  let totalElements := batchSize * seqLen * numHeads * dimPairs

  let shader := ropeCachedKernel layer.config batchSize seqLen numHeads
  let namedBuffers := [
    ("input", inputBuf),
    ("cos_cache", cosCacheBuf),
    ("sin_cache", sinCacheBuf),
    ("output", outputBuf)
  ]

  let execConfig := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D
    totalElements
    256

  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig
  logVerbose "[RoPE] ✓ Cached forward pass complete"

/-! ## Cache Generation Utilities -/

/-- Generate cos/sin cache on CPU for later GPU upload

    This is typically done once during model initialization.

    @param config RoPE configuration
    @param numHeads Number of attention heads
    @return (cos_cache, sin_cache) as ByteArrays
-/
def generateCache (config : Config) (numHeads : Nat) : IO (ByteArray × ByteArray) := do
  let headDim := config.dim / numHeads
  let dimPairs := headDim / 2

  IO.println s!"[RoPE] Generating cache: max_seq_len={config.maxSeqLen}, dim_pairs={dimPairs}"

  let mut cosCache := ByteArray.empty
  let mut sinCache := ByteArray.empty

  for pos in [0:config.maxSeqLen] do
    for dimPair in [0:dimPairs] do
      -- Compute theta: pos × base^(-2×dimPair/headDim)
      let exponent := -(2.0 * dimPair.toFloat / headDim.toFloat)
      let freqInv := Float.pow config.base exponent
      let theta := pos.toFloat * freqInv

      -- Compute cos and sin
      let cosVal := theta.cos
      let sinVal := theta.sin

      -- Convert to bytes (Float = 4 bytes, little-endian)
      -- Simplified: store as-is (proper implementation would use bit manipulation)
      -- For now, use placeholder
      cosCache := cosCache.append (ByteArray.mk #[0, 0, 0, 0])  -- TODO: proper Float32 encoding
      sinCache := sinCache.append (ByteArray.mk #[0, 0, 0, 0])

  IO.println "[RoPE] ✓ Cache generated"
  pure (cosCache, sinCache)

end Hesper.Layers.RoPE
