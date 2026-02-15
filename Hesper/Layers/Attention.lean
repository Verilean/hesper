import Hesper.WGSL.Monad
import Hesper.WGSL.Execute
import Hesper.WGSL.Exp
import Hesper.WGSL.MatMul
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.Layers.BitLinear
import Hesper.Layers.RoPE
import Hesper.Layers.Softmax
import Hesper.Layers.RMSNorm
import Hesper.Logging

/-!
# Multi-Head Self-Attention

Implements the attention mechanism used in BitNet transformers.

## Mathematical Definition

Self-attention allows each position to attend to all positions in the input:

```
Q = input @ W_q   # Query projection
K = input @ W_k   # Key projection
V = input @ W_v   # Value projection

scores = (Q @ K^T) / sqrt(d_k)
attn = softmax(scores)  # Attention weights
output = attn @ V
result = output @ W_o   # Output projection
```

## Multi-Head Attention

Instead of single attention, split into H heads:

```
For each head h:
  Q_h = Q[:, h*d_h : (h+1)*d_h]
  K_h = K[:, h*d_h : (h+1)*d_h]
  V_h = V[:, h*d_h : (h+1)*d_h]

  attn_h = softmax(Q_h @ K_h^T / sqrt(d_h)) @ V_h

output = concat(attn_1, ..., attn_H) @ W_o
```

**Why multiple heads?**
- Each head learns different attention patterns
- Head 1: Local syntax (nearby words)
- Head 2: Long-range dependencies
- Head 3: Semantic relationships
- Etc.

## BitNet Optimization

**Standard Transformer**:
- W_q, W_k, W_v, W_o: Float32 weights
- Memory: 4 × d² × 4 bytes per layer

**BitNet**:
- W_q, W_k, W_v, W_o: Ternary {-1, 0, 1} weights (TQ2_0)
- Memory: 4 × d² × 0.25 bytes per layer
- **16x memory savings**
- **Faster compute**: additions instead of multiplications

## Causal Masking

For autoregressive generation, prevent attending to future tokens:

```
mask[i,j] = 0   if j ≤ i  (can attend to past)
mask[i,j] = -∞  if j > i  (cannot attend to future)

attn = softmax(scores + mask)
```

After softmax, masked positions → 0 probability.

## Performance

For BitNet-3B (hidden=2560, heads=32, seq=2048):
```
Per layer:
- Q,K,V projections: 3 × BitLinear(2560→2560) = 52.4 GFLOPS
- Attention scores: Q @ K^T = 107.4 GFLOPS (2048×80 @ 80×2048)
- Softmax: 0.1 GFLOPS (negligible)
- Attention output: attn @ V = 107.4 GFLOPS
- Output projection: BitLinear(2560→2560) = 17.5 GFLOPS

Total: ~285 GFLOPS per attention layer
```

## References
- Attention is All You Need (Vaswani et al., 2017)
- LLaMA architecture: https://github.com/facebookresearch/llama
- llama.cpp: ggml-cuda/attention.cu
- Flash Attention: https://arxiv.org/abs/2205.14135
-/

namespace Hesper.Layers.Attention

open Hesper.WGSL
open Hesper.WGSL.Monad
open Hesper.WGSL.MatMul
open Hesper.WebGPU
open Hesper.Layers
open Hesper.Logging (logVerbose)

/-! ## Configuration -/

/-- Multi-head attention configuration -/
structure Config where
  dim : Nat           -- Model dimension (hidden size)
  numHeads : Nat      -- Number of query heads
  numKVHeads : Nat := 0  -- Number of KV heads (0 = same as numHeads, i.e. MHA)
  headDim : Nat := 0     -- Per-head dimension (0 = dim / numHeads)
  maxSeqLen : Nat     -- Maximum sequence length
  useCausalMask : Bool := true  -- Enable causal masking for autoregressive
  deriving Repr

/-- Effective KV head count -/
def Config.effectiveKVHeads (c : Config) : Nat :=
  if c.numKVHeads == 0 then c.numHeads else c.numKVHeads

/-- Effective head dimension -/
def Config.effectiveHeadDim (c : Config) : Nat :=
  if c.headDim == 0 then c.dim / c.numHeads else c.headDim

/-- KV dimension (numKVHeads * headDim) -/
def Config.kvDim (c : Config) : Nat :=
  c.effectiveKVHeads * c.effectiveHeadDim

/-- Validate configuration -/
def Config.validate (config : Config) : Except String Unit := do
  if config.numHeads = 0 then
    .error "numHeads must be > 0"
  if config.dim = 0 then
    .error "dim must be > 0"
  .ok ()

/-! ## Reshape Kernels for Multi-Head Attention -/

/-- Reshape [batch, seq, inputHeads, headDim] → [batch, outputHeads, seq, headDim]

    For Q: inputHeads = outputHeads = numHeads (simple transpose)
    For K/V with GQA: inputHeads = numKVHeads, outputHeads = numHeads (repeat KV heads)

    Each output element at [b, oh, s, d]:
      kv_head = oh / headsPerKVHead
      input_offset = b * seq * inputHeads * headDim + s * inputHeads * headDim + kv_head * headDim + d
-/
def reshapeToHeadsKernel (batchSize seqLen inputHeads outputHeads headDim : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let totalOut := batchSize * outputHeads * seqLen * headDim
  let totalIn := batchSize * seqLen * inputHeads * headDim
  let inBounds := Exp.lt idx (Exp.litU32 totalOut)

  let _inp ← ShaderM.declareInputBuffer "inp" (.array (.scalar .f32) totalIn)
  let _out ← ShaderM.declareOutputBuffer "out" (.array (.scalar .f32) totalOut)

  let headsPerKVHead := outputHeads / inputHeads

  -- Decompose output index: [batch, head, seq, dim]
  let dimIdx := Exp.mod idx (Exp.litU32 headDim)
  let seqIdx := Exp.mod (Exp.div idx (Exp.litU32 headDim)) (Exp.litU32 seqLen)
  let headIdx := Exp.mod (Exp.div idx (Exp.litU32 (seqLen * headDim))) (Exp.litU32 outputHeads)
  let batchIdx := Exp.div idx (Exp.litU32 (outputHeads * seqLen * headDim))

  -- Map output head to input (KV) head
  let kvHead := Exp.div headIdx (Exp.litU32 headsPerKVHead)

  -- Input: [batch, seq, inputHeads, headDim]
  let inOffset := Exp.add
    (Exp.mul batchIdx (Exp.litU32 (seqLen * inputHeads * headDim)))
    (Exp.add
      (Exp.mul seqIdx (Exp.litU32 (inputHeads * headDim)))
      (Exp.add (Exp.mul kvHead (Exp.litU32 headDim)) dimIdx))

  let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalIn) "inp" inOffset
  let result := Exp.select inBounds val (Exp.litF32 0.0)
  ShaderM.writeBuffer (ty := .scalar .f32) "out" idx result

/-- Reshape [batch, heads, seq, headDim] → [batch, seq, heads, headDim]

    Reverse of reshapeToHeads. Used after attention output, before output projection.
-/
def reshapeFromHeadsKernel (batchSize seqLen numHeads headDim : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let total := batchSize * seqLen * numHeads * headDim
  let inBounds := Exp.lt idx (Exp.litU32 total)

  let _inp ← ShaderM.declareInputBuffer "inp" (.array (.scalar .f32) total)
  let _out ← ShaderM.declareOutputBuffer "out" (.array (.scalar .f32) total)

  -- Decompose output index: [batch, seq, head, dim]
  let dimIdx := Exp.mod idx (Exp.litU32 headDim)
  let headIdx := Exp.mod (Exp.div idx (Exp.litU32 headDim)) (Exp.litU32 numHeads)
  let seqIdx := Exp.mod (Exp.div idx (Exp.litU32 (numHeads * headDim))) (Exp.litU32 seqLen)
  let batchIdx := Exp.div idx (Exp.litU32 (seqLen * numHeads * headDim))

  -- Input: [batch, heads, seq, headDim]
  let inOffset := Exp.add
    (Exp.mul batchIdx (Exp.litU32 (numHeads * seqLen * headDim)))
    (Exp.add
      (Exp.mul headIdx (Exp.litU32 (seqLen * headDim)))
      (Exp.add (Exp.mul seqIdx (Exp.litU32 headDim)) dimIdx))

  let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := total) "inp" inOffset
  let result := Exp.select inBounds val (Exp.litF32 0.0)
  ShaderM.writeBuffer (ty := .scalar .f32) "out" idx result

/-- Execute reshape to multi-head layout -/
def executeReshapeToHeads (device : Device) (inBuf outBuf : Buffer)
    (batchSize seqLen inputHeads outputHeads headDim : Nat) : IO Unit := do
  let shader := reshapeToHeadsKernel batchSize seqLen inputHeads outputHeads headDim
  let namedBuffers := [("inp", inBuf), ("out", outBuf)]
  let totalElements := batchSize * outputHeads * seqLen * headDim
  let execConfig := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D totalElements 256
  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig

/-- Execute reshape from multi-head layout -/
def executeReshapeFromHeads (device : Device) (inBuf outBuf : Buffer)
    (batchSize seqLen numHeads headDim : Nat) : IO Unit := do
  let shader := reshapeFromHeadsKernel batchSize seqLen numHeads headDim
  let namedBuffers := [("inp", inBuf), ("out", outBuf)]
  let totalElements := batchSize * seqLen * numHeads * headDim
  let execConfig := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D totalElements 256
  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig

/-! ## Layer Structure -/

/-- Multi-head self-attention layer -/
structure Attention where
  config : Config
  -- Q, K, V projection weights (TQ2_0 packed)
  wQ : BitLinear.BitLinear
  wK : BitLinear.BitLinear
  wV : BitLinear.BitLinear
  -- Output projection weight
  wO : BitLinear.BitLinear
  -- RoPE layer (no learned parameters)
  rope : RoPE.RoPE
  -- Softmax layer (no learned parameters)
  softmax : Softmax.Softmax

/-! ## Layer Creation -/

/-- Create attention layer from GGUF tensors

    @param device WebGPU device
    @param config Attention configuration
    @param wqData Q projection weights (TQ2_0 packed)
    @param wkData K projection weights (TQ2_0 packed)
    @param wvData V projection weights (TQ2_0 packed)
    @param woData Output projection weights (TQ2_0 packed)
    @param qScales Q projection scales (FP16)
    @param kScales K projection scales (FP16)
    @param vScales V projection scales (FP16)
    @param oScales Output projection scales (FP16)
-/
def create (device : Device) (config : Config)
           (wqData wkData wvData woData : ByteArray)
           (qScale kScale vScale oScale : Float) : IO Attention := do
  -- Validate configuration
  match config.validate with
  | .error msg => throw $ IO.userError s!"Invalid attention config: {msg}"
  | .ok _ => pure ()

  IO.println s!"[Attention] Creating layer: dim={config.dim}, heads={config.numHeads}, max_seq={config.maxSeqLen}"

  let headDim := config.dim / config.numHeads
  IO.println s!"  Head dimension: {headDim}"

  -- Create BitLinear layers for projections
  let bitlinearConfig : BitLinear.Config := {
    inDim := config.dim,
    outDim := config.dim,
    batchSize := 1  -- Will be adjusted during forward pass
  }

  let wQ ← BitLinear.create device bitlinearConfig wqData qScale
  let wK ← BitLinear.create device bitlinearConfig wkData kScale
  let wV ← BitLinear.create device bitlinearConfig wvData vScale
  let wO ← BitLinear.create device bitlinearConfig woData oScale

  -- Create RoPE layer
  let ropeConfig : RoPE.Config := {
    dim := config.dim,
    maxSeqLen := config.maxSeqLen,
    base := 10000.0
  }
  let rope ← RoPE.create ropeConfig

  -- Create Softmax layer (will be configured per forward pass)
  -- Placeholder config (updated in forward)
  let softmaxConfig : Softmax.Config := {
    rowSize := config.maxSeqLen,
    numRows := config.numHeads,
    useMask := config.useCausalMask
  }
  let softmax ← Softmax.create softmaxConfig

  IO.println "[Attention] ✓ Layer created"
  pure { config, wQ, wK, wV, wO, rope, softmax }

/-! ## Forward Pass -/

/-- Execute attention forward pass

    **Algorithm**:
    ```
    1. Project input to Q, K, V:
       Q = input @ W_q
       K = input @ W_k
       V = input @ W_v

    2. Reshape to multi-head format:
       Q, K, V: [batch, seq, dim] → [batch, seq, heads, head_dim]
                                  → [batch, heads, seq, head_dim]

    3. Apply RoPE to Q and K:
       Q_rot = RoPE(Q)
       K_rot = RoPE(K)

    4. Compute attention scores:
       scores = (Q_rot @ K_rot^T) / sqrt(head_dim)
       Shape: [batch, heads, seq, seq]

    5. Apply softmax (with causal mask if enabled):
       attn_weights = softmax(scores)

    6. Apply attention to values:
       attn_output = attn_weights @ V
       Shape: [batch, heads, seq, head_dim]

    7. Reshape and project output:
       output = concat_heads(attn_output)
       result = output @ W_o
    ```

    @param device WebGPU device
    @param layer Attention layer
    @param inputBuf Input tensor [batch, seq_len, dim]
    @param outputBuf Output tensor [batch, seq_len, dim]
    @param batchSize Batch size
    @param seqLen Sequence length
-/
def forward (device : Device) (layer : Attention)
            (inputBuf outputBuf : Buffer)
            (batchSize seqLen : Nat)
            (subNorm : Option RMSNorm.RMSNorm := none) : IO Unit := do
  let headDim := layer.config.effectiveHeadDim
  let numKVHeads := layer.config.effectiveKVHeads
  let kvDim := layer.config.kvDim
  logVerbose s!"[Attention] Forward pass: batch={batchSize}, seq_len={seqLen}, heads={layer.config.numHeads}, kv_heads={numKVHeads}, head_dim={headDim}"

  -- Allocate temporary buffers
  let qSize := (batchSize * seqLen * layer.config.dim * 4).toUSize  -- Float32
  let kvSize := (batchSize * seqLen * kvDim * 4).toUSize  -- KV may be smaller (GQA)
  let scoresSize := (batchSize * layer.config.numHeads * seqLen * seqLen * 4).toUSize
  let attnSize := scoresSize

  let qBuf ← createBuffer device { size := qSize, usage := [.storage], mappedAtCreation := false }
  let kBuf ← createBuffer device { size := kvSize, usage := [.storage], mappedAtCreation := false }
  let vBuf ← createBuffer device { size := kvSize, usage := [.storage], mappedAtCreation := false }
  let scoresBuf ← createBuffer device { size := scoresSize, usage := [.storage], mappedAtCreation := false }
  let attnBuf ← createBuffer device { size := attnSize, usage := [.storage], mappedAtCreation := false }

  -- Step 1: Project to Q, K, V using BitLinear
  let numRows := batchSize * seqLen
  logVerbose s!"  [1/7] Projecting to Q, K, V ({numRows} rows)..."
  BitLinear.forward device layer.wQ inputBuf qBuf numRows
  BitLinear.forward device layer.wK inputBuf kBuf numRows
  BitLinear.forward device layer.wV inputBuf vBuf numRows

  -- Step 2: Apply RoPE to Q and K (need temp buffer since WebGPU disallows aliased writable bindings)
  logVerbose "  [2/7] Applying RoPE to Q and K..."
  let qTmpBuf ← createBuffer device { size := qSize, usage := [.storage], mappedAtCreation := false }
  let kvTmpBuf ← createBuffer device { size := kvSize, usage := [.storage], mappedAtCreation := false }
  RoPE.forward device layer.rope qBuf qTmpBuf batchSize seqLen layer.config.numHeads headDim
  RoPE.forward device layer.rope kBuf kvTmpBuf batchSize seqLen numKVHeads headDim
  -- Swap: use rotated buffers going forward
  let qBuf := qTmpBuf
  let kBuf := kvTmpBuf

  -- Step 2.5: Reshape for multi-head attention
  -- Q: [batch, seq, numHeads*headDim] → [batch, numHeads, seq, headDim]
  -- K: [batch, seq, numKVHeads*headDim] → [batch, numHeads, seq, headDim] (with GQA repeat)
  -- V: [batch, seq, numKVHeads*headDim] → [batch, numHeads, seq, headDim] (with GQA repeat)
  logVerbose "  [2.5/7] Reshaping for multi-head attention..."
  let headBufSize := (batchSize * layer.config.numHeads * seqLen * headDim * 4).toUSize
  let qHeadBuf ← createBuffer device { size := headBufSize, usage := [.storage], mappedAtCreation := false }
  let kHeadBuf ← createBuffer device { size := headBufSize, usage := [.storage], mappedAtCreation := false }
  let vHeadBuf ← createBuffer device { size := headBufSize, usage := [.storage], mappedAtCreation := false }

  -- Q: simple transpose (inputHeads = outputHeads = numHeads)
  executeReshapeToHeads device qBuf qHeadBuf batchSize seqLen layer.config.numHeads layer.config.numHeads headDim
  -- K/V: transpose + GQA expansion (inputHeads = numKVHeads, outputHeads = numHeads)
  executeReshapeToHeads device kBuf kHeadBuf batchSize seqLen numKVHeads layer.config.numHeads headDim
  executeReshapeToHeads device vBuf vHeadBuf batchSize seqLen numKVHeads layer.config.numHeads headDim

  -- Step 3: Compute attention scores: Q @ K^T / sqrt(head_dim)
  logVerbose "  [3/7] Computing attention scores..."
  let scale := 1.0 / headDim.toFloat.sqrt

  let attnScoreConfig : MatMul.Config := {
    M := seqLen,
    N := seqLen,
    K := headDim
  }
  let batchedSize := batchSize * layer.config.numHeads

  MatMul.executeBatchedScaledMatMulTranspose device qHeadBuf kHeadBuf scoresBuf attnScoreConfig batchedSize scale

  -- Step 4: Apply softmax with optional causal mask
  logVerbose "  [4/7] Applying softmax..."
  let softmaxConfig : Softmax.Config := {
    rowSize := seqLen,
    numRows := batchedSize * seqLen,
    useMask := layer.config.useCausalMask
  }
  let softmaxLayer ← Softmax.create softmaxConfig
  Softmax.forward device softmaxLayer scoresBuf attnBuf

  -- Step 5: Apply attention to values: attn @ V
  -- attn: [batch*heads, seq, seq], V: [batch*heads, seq, headDim] → [batch*heads, seq, headDim]
  logVerbose "  [5/7] Applying attention to values..."
  let attnVConfig : MatMul.Config := {
    M := seqLen,
    N := headDim,
    K := seqLen
  }
  -- Reuse qHeadBuf for output (same size: [batch, numHeads, seq, headDim])
  MatMul.executeBatchedMatMul device attnBuf vHeadBuf qHeadBuf attnVConfig batchedSize

  -- Step 5.5: Reshape back: [batch, numHeads, seq, headDim] → [batch, seq, numHeads*headDim]
  logVerbose "  [5.5/7] Reshaping attention output..."
  let attnOutBuf ← createBuffer device { size := qSize, usage := [.storage], mappedAtCreation := false }
  executeReshapeFromHeads device qHeadBuf attnOutBuf batchSize seqLen layer.config.numHeads headDim

  -- Step 5.7: Apply attention sub-norm (if provided) - BitNet specific
  -- This normalizes the attention output before the O projection
  let attnOutForO ← match subNorm with
    | some norm => do
      logVerbose "  [5.7/7] Applying attention sub-norm..."
      let normedBuf ← createBuffer device { size := qSize, usage := [.storage], mappedAtCreation := false }
      RMSNorm.forward device norm attnOutBuf normedBuf numRows
      pure normedBuf
    | none => pure attnOutBuf

  -- Step 6: Output projection
  logVerbose s!"  [6/7] Output projection ({numRows} rows)..."
  BitLinear.forward device layer.wO attnOutForO outputBuf numRows

  logVerbose "[Attention] ✓ Forward pass complete"

/-! ## KV Caching (for Inference) -/

/-- Attention with KV caching for autoregressive generation

    During inference, we cache K and V from previous tokens to avoid recomputation.

    **Without caching** (generating 100 tokens):
    - Token 1: Compute K,V for position 0
    - Token 2: Compute K,V for positions 0,1 (recomputes position 0!)
    - Token 100: Compute K,V for positions 0-99 (recomputes everything!)
    - Total: O(N²) computation

    **With caching**:
    - Token 1: Compute K,V for position 0, cache it
    - Token 2: Compute K,V for position 1, append to cache
    - Token 100: Compute K,V for position 99, append to cache
    - Total: O(N) computation

    @param device WebGPU device
    @param layer Attention layer
    @param inputBuf Input tensor [batch, 1, dim] (single new token)
    @param kvCacheBuf Cached K,V from previous tokens
    @param outputBuf Output tensor [batch, 1, dim]
    @param batchSize Batch size
    @param cacheLen Number of cached positions
-/
def forwardWithCache (device : Device) (layer : Attention)
                     (inputBuf kvCacheBuf outputBuf : Buffer)
                     (batchSize cacheLen : Nat) : IO Unit := do
  -- TODO: Implement KV caching
  -- This requires:
  -- 1. Separate K and V cache buffers
  -- 2. Append operation to extend cache
  -- 3. Attention over cached + new token
  IO.println "[Attention] KV caching not yet implemented"
  throw $ IO.userError "forwardWithCache not implemented"

/-! ## Integration with GGUF -/

/-- Create attention layer from GGUF file

    Loads weight tensors for a specific transformer layer.

    Example tensor names:
    - `blk.0.attn_q.weight` - Q projection
    - `blk.0.attn_k.weight` - K projection
    - `blk.0.attn_v.weight` - V projection
    - `blk.0.attn_output.weight` - Output projection

    @param device WebGPU device
    @param gguf Loaded GGUF file
    @param layerIdx Layer index (0-31 for BitNet-3B)
    @param config Attention configuration
-/
def fromGGUF (device : Device) (gguf : α) (layerIdx : Nat) (config : Config) : IO Attention := do
  -- Placeholder - actual implementation would:
  -- 1. Find tensors: gguf.findTensor s!"blk.{layerIdx}.attn_q.weight"
  -- 2. Extract TQ2_0 packed data + scales
  -- 3. Call create() with extracted data
  IO.println s!"[Attention] Loading from GGUF layer {layerIdx}"
  throw $ IO.userError "fromGGUF not yet implemented - use create() directly"

end Hesper.Layers.Attention
