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

/-! ## Pre-allocated Buffers -/

/-- Pre-allocated buffers for attention forward pass.
    Avoids ~12 GPU buffer allocations per layer per token. -/
structure AttentionBuffers where
  qBuf : Buffer
  kBuf : Buffer
  vBuf : Buffer
  scoresBuf : Buffer
  attnBuf : Buffer
  qRotBuf : Buffer
  kRotBuf : Buffer
  qHeadBuf : Buffer
  kHeadBuf : Buffer
  vHeadBuf : Buffer
  reshapedOutBuf : Buffer
  subNormBuf : Buffer
  rmsTempBuf : Buffer

/-- Create pre-allocated attention buffers for given dimensions -/
def createAttentionBuffers (device : Device) (config : Config) (batchSize seqLen : Nat) : IO AttentionBuffers := do
  let headDim := config.effectiveHeadDim
  let kvDim := config.kvDim
  let numRows := batchSize * seqLen
  let qSize := (batchSize * seqLen * config.dim * 4).toUSize
  let kvSize := (batchSize * seqLen * kvDim * 4).toUSize
  let scoresSize := (batchSize * config.numHeads * seqLen * seqLen * 4).toUSize
  let headBufSize := (batchSize * config.numHeads * seqLen * headDim * 4).toUSize
  let mkBuf := fun size => createBuffer device { size := size, usage := [.storage], mappedAtCreation := false }
  pure {
    qBuf := ← mkBuf qSize
    kBuf := ← mkBuf kvSize
    vBuf := ← mkBuf kvSize
    scoresBuf := ← mkBuf scoresSize
    attnBuf := ← mkBuf scoresSize
    qRotBuf := ← mkBuf qSize
    kRotBuf := ← mkBuf kvSize
    qHeadBuf := ← mkBuf headBufSize
    kHeadBuf := ← mkBuf headBufSize
    vHeadBuf := ← mkBuf headBufSize
    reshapedOutBuf := ← mkBuf qSize
    subNormBuf := ← mkBuf qSize
    rmsTempBuf := ← mkBuf (numRows * 4).toUSize
  }

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

  logVerbose s!"[Attention] Creating layer: dim={config.dim}, heads={config.numHeads}, max_seq={config.maxSeqLen}"

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

  logVerbose "[Attention] ✓ Layer created"
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
            (subNorm : Option RMSNorm.RMSNorm := none)
            (preAllocBufs : Option AttentionBuffers := none)
            (residualBuf : Option Buffer := none) : IO Unit := do
  let headDim := layer.config.effectiveHeadDim
  let numKVHeads := layer.config.effectiveKVHeads
  let kvDim := layer.config.kvDim
  logVerbose s!"[Attention] Forward pass: batch={batchSize}, seq_len={seqLen}, heads={layer.config.numHeads}, kv_heads={numKVHeads}, head_dim={headDim}"

  -- Use pre-allocated or allocate temporary buffers
  let bufs ← match preAllocBufs with
    | some b => pure b
    | none => createAttentionBuffers device layer.config batchSize seqLen

  -- Step 1: Project to Q, K, V using BitLinear
  let numRows := batchSize * seqLen
  logVerbose s!"  [1/7] Projecting to Q, K, V ({numRows} rows)..."
  BitLinear.forward device layer.wQ inputBuf bufs.qBuf numRows
  BitLinear.forward device layer.wK inputBuf bufs.kBuf numRows
  BitLinear.forward device layer.wV inputBuf bufs.vBuf numRows

  -- Step 2: Apply RoPE to Q and K (need temp buffer since WebGPU disallows aliased writable bindings)
  logVerbose "  [2/7] Applying RoPE to Q and K..."
  RoPE.forward device layer.rope bufs.qBuf bufs.qRotBuf batchSize seqLen layer.config.numHeads headDim
  RoPE.forward device layer.rope bufs.kBuf bufs.kRotBuf batchSize seqLen numKVHeads headDim

  -- Step 2.5: Reshape for multi-head attention
  -- Q: [batch, seq, numHeads*headDim] → [batch, numHeads, seq, headDim]
  -- K: [batch, seq, numKVHeads*headDim] → [batch, numHeads, seq, headDim] (with GQA repeat)
  -- V: [batch, seq, numKVHeads*headDim] → [batch, numHeads, seq, headDim] (with GQA repeat)
  logVerbose "  [2.5/7] Reshaping for multi-head attention..."

  -- Q: simple transpose (inputHeads = outputHeads = numHeads)
  executeReshapeToHeads device bufs.qRotBuf bufs.qHeadBuf batchSize seqLen layer.config.numHeads layer.config.numHeads headDim
  -- K/V: transpose + GQA expansion (inputHeads = numKVHeads, outputHeads = numHeads)
  executeReshapeToHeads device bufs.kRotBuf bufs.kHeadBuf batchSize seqLen numKVHeads layer.config.numHeads headDim
  executeReshapeToHeads device bufs.vBuf bufs.vHeadBuf batchSize seqLen numKVHeads layer.config.numHeads headDim

  -- Step 3: Compute attention scores: Q @ K^T / sqrt(head_dim)
  logVerbose "  [3/7] Computing attention scores..."
  let scale := 1.0 / headDim.toFloat.sqrt

  let attnScoreConfig : MatMul.Config := {
    M := seqLen,
    N := seqLen,
    K := headDim
  }
  let batchedSize := batchSize * layer.config.numHeads

  MatMul.executeBatchedScaledMatMulTranspose device bufs.qHeadBuf bufs.kHeadBuf bufs.scoresBuf attnScoreConfig batchedSize scale

  -- Step 4: Apply softmax with optional causal mask
  logVerbose "  [4/7] Applying softmax..."
  let softmaxConfig : Softmax.Config := {
    rowSize := seqLen,
    numRows := batchedSize * seqLen,
    useMask := layer.config.useCausalMask
  }
  let softmaxLayer ← Softmax.create softmaxConfig
  Softmax.forward device softmaxLayer bufs.scoresBuf bufs.attnBuf

  -- Step 5: Apply attention to values: attn @ V
  -- attn: [batch*heads, seq, seq], V: [batch*heads, seq, headDim] → [batch*heads, seq, headDim]
  logVerbose "  [5/7] Applying attention to values..."
  let attnVConfig : MatMul.Config := {
    M := seqLen,
    N := headDim,
    K := seqLen
  }
  -- Reuse qHeadBuf for output (same size: [batch, numHeads, seq, headDim])
  MatMul.executeBatchedMatMul device bufs.attnBuf bufs.vHeadBuf bufs.qHeadBuf attnVConfig batchedSize

  -- Step 5.5: Reshape back: [batch, numHeads, seq, headDim] → [batch, seq, numHeads*headDim]
  logVerbose "  [5.5/7] Reshaping attention output..."
  executeReshapeFromHeads device bufs.qHeadBuf bufs.reshapedOutBuf batchSize seqLen layer.config.numHeads headDim

  -- Step 5.7: Apply attention sub-norm (if provided) - BitNet specific
  -- This normalizes the attention output before the O projection
  let attnOutForO ← match subNorm with
    | some norm => do
      logVerbose "  [5.7/7] Applying attention sub-norm..."
      RMSNorm.forward device norm bufs.reshapedOutBuf bufs.subNormBuf numRows 256 (some bufs.rmsTempBuf)
      pure bufs.subNormBuf
    | none => pure bufs.reshapedOutBuf

  -- Step 6: Output projection (with optional fused residual add)
  logVerbose s!"  [6/7] Output projection ({numRows} rows)..."
  match residualBuf with
  | some resBuf =>
    BitLinear.forwardWithResidual device layer.wO attnOutForO resBuf outputBuf numRows
  | none =>
    BitLinear.forward device layer.wO attnOutForO outputBuf numRows

  logVerbose "[Attention] ✓ Forward pass complete"

/-! ## KV Cache Structures -/

/-- KV cache for a single attention layer.
    Stores key and value vectors for all past positions. -/
structure KVCache where
  kBuf : Buffer    -- [numKVHeads, maxSeqLen, headDim]
  vBuf : Buffer    -- [numKVHeads, maxSeqLen, headDim]
  -- Per-layer PreparedDispatch for kernels that bind kvCache buffers
  preparedCacheWriteK : IO.Ref (Option Hesper.WGSL.Execute.PreparedDispatch)
  preparedCacheWriteV : IO.Ref (Option Hesper.WGSL.Execute.PreparedDispatch)
  preparedCacheWriteKV : IO.Ref (Option Hesper.WGSL.Execute.PreparedDispatch)
  preparedScores : IO.Ref (Option Hesper.WGSL.Execute.PreparedDispatch)
  preparedApply : IO.Ref (Option Hesper.WGSL.Execute.PreparedDispatch)

/-- Create KV cache for one attention layer -/
def createKVCache (device : Device) (config : Config) : IO KVCache := do
  let headDim := config.effectiveHeadDim
  let numKVHeads := config.effectiveKVHeads
  let cacheSize := (numKVHeads * config.maxSeqLen * headDim * 4).toUSize
  let mkBuf := fun () => createBuffer device { size := cacheSize, usage := [.storage], mappedAtCreation := false }
  pure {
    kBuf := ← mkBuf (), vBuf := ← mkBuf ()
    preparedCacheWriteK := ← IO.mkRef none
    preparedCacheWriteV := ← IO.mkRef none
    preparedCacheWriteKV := ← IO.mkRef none
    preparedScores := ← IO.mkRef none
    preparedApply := ← IO.mkRef none
  }

/-- Pre-allocated buffers for cached single-token attention -/
structure CachedAttentionBuffers where
  qBuf : Buffer        -- [dim]
  kNewBuf : Buffer     -- [kvDim]
  vNewBuf : Buffer     -- [kvDim]
  qRotBuf : Buffer     -- [dim]
  kRotBuf : Buffer     -- [kvDim]
  scoresBuf : Buffer   -- [numHeads * maxSeqLen]
  attnBuf : Buffer     -- [numHeads * maxSeqLen]
  subNormBuf : Buffer  -- [dim]
  rmsTempBuf : Buffer  -- small
  paramsBuf : Buffer   -- [2 × u32]: pos, cacheLen
  -- PreparedDispatch refs for instant replay (shared-buffer-safe only)
  preparedSoftmax : IO.Ref (Option Hesper.WGSL.Execute.PreparedDispatch)
  preparedRopeQ : IO.Ref (Option Hesper.WGSL.Execute.PreparedDispatch)
  preparedRopeK : IO.Ref (Option Hesper.WGSL.Execute.PreparedDispatch)

/-- Create pre-allocated buffers for cached attention -/
def createCachedAttentionBuffers (device : Device) (config : Config) : IO CachedAttentionBuffers := do
  let kvDim := config.kvDim
  let mkBuf := fun size => createBuffer device { size := size, usage := [.storage], mappedAtCreation := false }
  let mkCopyBuf := fun size => createBuffer device { size := size, usage := [.storage, .copyDst], mappedAtCreation := false }
  pure {
    qBuf := ← mkBuf (config.dim * 4).toUSize
    kNewBuf := ← mkBuf (kvDim * 4).toUSize
    vNewBuf := ← mkBuf (kvDim * 4).toUSize
    qRotBuf := ← mkBuf (config.dim * 4).toUSize
    kRotBuf := ← mkBuf (kvDim * 4).toUSize
    scoresBuf := ← mkBuf (config.numHeads * config.maxSeqLen * 4).toUSize
    attnBuf := ← mkBuf (config.numHeads * config.maxSeqLen * 4).toUSize
    subNormBuf := ← mkBuf (config.dim * 4).toUSize
    rmsTempBuf := ← mkBuf 4
    paramsBuf := ← mkCopyBuf 8  -- 2 × u32 = 8 bytes
    preparedSoftmax := ← IO.mkRef none
    preparedRopeQ := ← IO.mkRef none
    preparedRopeK := ← IO.mkRef none
  }

/-! ## KV Cache Kernels

All cached attention kernels read `pos` and `cacheLen` from a `params` buffer
instead of baking them as WGSL literals. This produces **identical WGSL** across
tokens, enabling pipeline + bind group caching (97%+ hit rate).

Params buffer layout: `[pos: u32, cacheLen: u32]` (8 bytes)
-/

/-- Write new K or V data into cache at position read from params[0].
    Input: [kvDim] = [numKVHeads * headDim] (flat)
    Cache: [numKVHeads, maxSeqLen, headDim]
    Params: [pos: u32, cacheLen: u32]
-/
def cacheWriteKernel (numKVHeads maxSeqLen headDim kvDim : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let _newData ← ShaderM.declareInputBuffer "new_data" (.array (.scalar .f32) kvDim)
  let _cache ← ShaderM.declareOutputBuffer "cache" (.array (.scalar .f32) (numKVHeads * maxSeqLen * headDim))
  let _params ← ShaderM.declareInputBuffer "params" (.array (.scalar .u32) 2)

  let inBounds := Exp.lt idx (Exp.litU32 kvDim)

  -- Read pos from params buffer
  let pos ← ShaderM.readBuffer (ty := .scalar .u32) (n := 2) "params" (Exp.litU32 0)

  -- Decompose: kvHead = idx / headDim, d = idx % headDim
  let kvHead := Exp.div idx (Exp.litU32 headDim)
  let d := Exp.mod idx (Exp.litU32 headDim)

  -- Cache index: kvHead * maxSeqLen * headDim + pos * headDim + d
  let cacheIdx := Exp.add
    (Exp.mul kvHead (Exp.litU32 (maxSeqLen * headDim)))
    (Exp.add (Exp.mul pos (Exp.litU32 headDim)) d)

  let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := kvDim) "new_data" idx
  ShaderM.if_ inBounds (do
    ShaderM.writeBuffer (ty := .scalar .f32) "cache" cacheIdx val
  ) (pure ())

/-- Fused write K and V data into cache at position read from params[0].
    Processes both K and V in a single dispatch (saves 1 dispatch per layer).
    Input: new_k [kvDim], new_v [kvDim]
    Cache: k_cache [numKVHeads, maxSeqLen, headDim], v_cache [numKVHeads, maxSeqLen, headDim]
    Params: [pos: u32, cacheLen: u32]
-/
def fusedCacheWriteKVKernel (numKVHeads maxSeqLen headDim kvDim : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let _newK ← ShaderM.declareInputBuffer "new_k" (.array (.scalar .f32) kvDim)
  let _newV ← ShaderM.declareInputBuffer "new_v" (.array (.scalar .f32) kvDim)
  let _kCache ← ShaderM.declareOutputBuffer "k_cache" (.array (.scalar .f32) (numKVHeads * maxSeqLen * headDim))
  let _vCache ← ShaderM.declareOutputBuffer "v_cache" (.array (.scalar .f32) (numKVHeads * maxSeqLen * headDim))
  let _params ← ShaderM.declareInputBuffer "params" (.array (.scalar .u32) 2)

  let inBounds := Exp.lt idx (Exp.litU32 kvDim)

  -- Read pos from params buffer
  let pos ← ShaderM.readBuffer (ty := .scalar .u32) (n := 2) "params" (Exp.litU32 0)

  -- Decompose: kvHead = idx / headDim, d = idx % headDim
  let kvHead := Exp.div idx (Exp.litU32 headDim)
  let d := Exp.mod idx (Exp.litU32 headDim)

  -- Cache index: kvHead * maxSeqLen * headDim + pos * headDim + d
  let cacheIdx := Exp.add
    (Exp.mul kvHead (Exp.litU32 (maxSeqLen * headDim)))
    (Exp.add (Exp.mul pos (Exp.litU32 headDim)) d)

  let kVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := kvDim) "new_k" idx
  let vVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := kvDim) "new_v" idx
  ShaderM.if_ inBounds (do
    ShaderM.writeBuffer (ty := .scalar .f32) "k_cache" cacheIdx kVal
    ShaderM.writeBuffer (ty := .scalar .f32) "v_cache" cacheIdx vVal
  ) (pure ())

/-- Attention scores with dynamic cacheLen from params buffer.
    Q: [numHeads * headDim]
    K_cache: [numKVHeads, maxSeqLen, headDim]
    Scores: [numHeads * maxSeqLen] (only first cacheLen per head used)
    Params: [pos: u32, cacheLen: u32]
-/
def cachedScoresKernel (numHeads numKVHeads maxSeqLen headDim : Nat) (scale : Float) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let headsPerKVHead := numHeads / numKVHeads

  let _q ← ShaderM.declareInputBuffer "q" (.array (.scalar .f32) (numHeads * headDim))
  let _kCache ← ShaderM.declareInputBuffer "k_cache" (.array (.scalar .f32) (numKVHeads * maxSeqLen * headDim))
  let _scores ← ShaderM.declareOutputBuffer "scores" (.array (.scalar .f32) (numHeads * maxSeqLen))
  let _params ← ShaderM.declareInputBuffer "params" (.array (.scalar .u32) 2)

  -- Read cacheLen from params buffer
  let cacheLen ← ShaderM.readBuffer (ty := .scalar .u32) (n := 2) "params" (Exp.litU32 1)
  let totalOutputs := Exp.mul (Exp.litU32 numHeads) cacheLen
  let inBounds := Exp.lt idx totalOutputs

  -- Decompose: head = idx / cacheLen, s = idx % cacheLen
  let head := Exp.div idx cacheLen
  let s := Exp.mod idx cacheLen

  -- GQA: map query head to KV head
  let kvHead := Exp.div head (Exp.litU32 headsPerKVHead)

  -- Dot product Q[head, :] · K_cache[kvHead, s, :]
  ShaderM.varNamed "dot" (.scalar .f32) (Exp.litF32 0.0)
  let dot : Exp (.scalar .f32) := Exp.var "dot"

  ShaderM.loop (Exp.litU32 0) (Exp.litU32 headDim) (Exp.litU32 1) fun d => do
    let qIdx := Exp.add (Exp.mul head (Exp.litU32 headDim)) d
    let kIdx := Exp.add
      (Exp.mul kvHead (Exp.litU32 (maxSeqLen * headDim)))
      (Exp.add (Exp.mul s (Exp.litU32 headDim)) d)
    let qVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numHeads * headDim) "q" qIdx
    let kVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numKVHeads * maxSeqLen * headDim) "k_cache" kIdx
    ShaderM.assign "dot" (Exp.add dot (Exp.mul qVal kVal))

  let scaled := Exp.mul dot (Exp.litF32 scale)
  let result := Exp.select inBounds scaled (Exp.litF32 0.0)
  ShaderM.writeBuffer (ty := .scalar .f32) "scores" idx result

/-- Softmax with dynamic row size (cacheLen) from params buffer.
    Input/Output: [numRows * maxSeqLen] (only first cacheLen per row used)
    Params: [pos: u32, cacheLen: u32]
-/
def cachedSoftmaxKernel (numRows maxSeqLen : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) (numRows * maxSeqLen))
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) (numRows * maxSeqLen))
  let _params ← ShaderM.declareInputBuffer "params" (.array (.scalar .u32) 2)

  -- Read cacheLen (= rowSize) from params buffer
  let rowSize ← ShaderM.readBuffer (ty := .scalar .u32) (n := 2) "params" (Exp.litU32 1)
  let totalElements := Exp.mul (Exp.litU32 numRows) rowSize
  let inBounds := Exp.lt idx totalElements

  let row := Exp.div idx rowSize
  let rowStart := Exp.mul row rowSize

  -- Find max in row
  ShaderM.varNamed "max_val" (.scalar .f32) (Exp.litF32 (-3.4e38))
  let maxVal : Exp (.scalar .f32) := Exp.var "max_val"
  ShaderM.loop (Exp.litU32 0) rowSize (Exp.litU32 1) fun i => do
    let elemIdx := Exp.add rowStart i
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := numRows * maxSeqLen) "input" elemIdx
    ShaderM.assign "max_val" (Exp.max maxVal val)

  -- exp(x - max) for this element
  let inputVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numRows * maxSeqLen) "input" idx
  let shifted := Exp.sub inputVal maxVal
  let expVal := Exp.exp shifted

  -- Sum exp values in row
  ShaderM.varNamed "sum_exp" (.scalar .f32) (Exp.litF32 0.0)
  let sumExp : Exp (.scalar .f32) := Exp.var "sum_exp"
  ShaderM.loop (Exp.litU32 0) rowSize (Exp.litU32 1) fun i => do
    let elemIdx := Exp.add rowStart i
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := numRows * maxSeqLen) "input" elemIdx
    let sv := Exp.sub val maxVal
    let ev := Exp.exp sv
    ShaderM.assign "sum_exp" (Exp.add sumExp ev)

  -- Normalize
  let result := Exp.div expVal sumExp
  let finalResult := Exp.select inBounds result (Exp.litF32 0.0)
  ShaderM.writeBuffer (ty := .scalar .f32) "output" idx finalResult

/-- Apply attention weights to V_cache with dynamic cacheLen from params buffer.
    attn_weights: [numHeads * maxSeqLen] (only first cacheLen per head used)
    V_cache: [numKVHeads, maxSeqLen, headDim]
    Output: [numHeads * headDim] = [dim] (single token)
    Params: [pos: u32, cacheLen: u32]
-/
def cachedApplyKernel (numHeads numKVHeads maxSeqLen headDim : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let totalOutputs := numHeads * headDim
  let headsPerKVHead := numHeads / numKVHeads

  let _attn ← ShaderM.declareInputBuffer "attn" (.array (.scalar .f32) (numHeads * maxSeqLen))
  let _vCache ← ShaderM.declareInputBuffer "v_cache" (.array (.scalar .f32) (numKVHeads * maxSeqLen * headDim))
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) totalOutputs)
  let _params ← ShaderM.declareInputBuffer "params" (.array (.scalar .u32) 2)

  let inBounds := Exp.lt idx (Exp.litU32 totalOutputs)

  -- Read cacheLen from params buffer
  let cacheLen ← ShaderM.readBuffer (ty := .scalar .u32) (n := 2) "params" (Exp.litU32 1)

  -- Decompose: head = idx / headDim, d = idx % headDim
  let head := Exp.div idx (Exp.litU32 headDim)
  let d := Exp.mod idx (Exp.litU32 headDim)

  -- GQA mapping
  let kvHead := Exp.div head (Exp.litU32 headsPerKVHead)

  -- Weighted sum: out[head, d] = sum_s attn[head, s] * V[kvHead, s, d]
  ShaderM.varNamed "wsum" (.scalar .f32) (Exp.litF32 0.0)
  let wsum : Exp (.scalar .f32) := Exp.var "wsum"

  ShaderM.loop (Exp.litU32 0) cacheLen (Exp.litU32 1) fun s => do
    let attnIdx := Exp.add (Exp.mul head cacheLen) s
    let vIdx := Exp.add
      (Exp.mul kvHead (Exp.litU32 (maxSeqLen * headDim)))
      (Exp.add (Exp.mul s (Exp.litU32 headDim)) d)
    let attnVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numHeads * maxSeqLen) "attn" attnIdx
    let vVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numKVHeads * maxSeqLen * headDim) "v_cache" vIdx
    ShaderM.assign "wsum" (Exp.add wsum (Exp.mul attnVal vVal))

  let result := Exp.select inBounds wsum (Exp.litF32 0.0)
  ShaderM.writeBuffer (ty := .scalar .f32) "output" idx result

/-! ## Forward Pass with KV Cache -/

/-- Execute attention forward pass with KV cache (single-token inference).

    For each new token:
    1. Project Q, K_new, V_new (single row)
    2. Apply RoPE at position `pos`
    3. Append K_new, V_new to cache
    4. Compute attention scores Q @ K_cache^T * scale (GQA)
    5. Softmax
    6. Apply attention to V_cache (GQA)
    7. Sub-norm + O projection

    @param pos Position of the new token (0-indexed)
-/
def forwardWithCache (device : Device) (layer : Attention)
                     (inputBuf outputBuf : Buffer)
                     (kvCache : KVCache) (pos : Nat)
                     (subNorm : Option RMSNorm.RMSNorm := none)
                     (preAllocBufs : Option CachedAttentionBuffers := none)
                     (residualBuf : Option Buffer := none) : IO Unit := do
  let headDim := layer.config.effectiveHeadDim
  let numKVHeads := layer.config.effectiveKVHeads
  let kvDim := layer.config.kvDim
  let numHeads := layer.config.numHeads
  let maxSeqLen := layer.config.maxSeqLen
  let cacheLen := pos + 1  -- After appending, cache has pos+1 entries

  logVerbose s!"[Attention] Cached forward: pos={pos}, cacheLen={cacheLen}"

  let bufs ← match preAllocBufs with
    | some b => pure b
    | none => createCachedAttentionBuffers device layer.config

  -- Step 1: Project Q, K_new, V_new (single row)
  BitLinear.forward device layer.wQ inputBuf bufs.qBuf 1
  BitLinear.forward device layer.wK inputBuf bufs.kNewBuf 1
  BitLinear.forward device layer.wV inputBuf bufs.vNewBuf 1

  -- Write params buffer: [pos: u32, cacheLen: u32]
  -- Done BEFORE RoPE so the dynamic kernel can read posOffset from params[0]
  let paramsBytes := ByteArray.empty
    |>.push (pos.toUInt32 &&& 0xFF).toUInt8
    |>.push ((pos.toUInt32 >>> 8) &&& 0xFF).toUInt8
    |>.push ((pos.toUInt32 >>> 16) &&& 0xFF).toUInt8
    |>.push ((pos.toUInt32 >>> 24) &&& 0xFF).toUInt8
    |>.push (cacheLen.toUInt32 &&& 0xFF).toUInt8
    |>.push ((cacheLen.toUInt32 >>> 8) &&& 0xFF).toUInt8
    |>.push ((cacheLen.toUInt32 >>> 16) &&& 0xFF).toUInt8
    |>.push ((cacheLen.toUInt32 >>> 24) &&& 0xFF).toUInt8
  writeBuffer device bufs.paramsBuf 0 paramsBytes

  -- Step 2: Apply RoPE at position `pos` (reads posOffset from params[0])
  RoPE.forwardDynamic device layer.rope bufs.qBuf bufs.qRotBuf bufs.paramsBuf 1 1 numHeads headDim (some bufs.preparedRopeQ)
  RoPE.forwardDynamic device layer.rope bufs.kNewBuf bufs.kRotBuf bufs.paramsBuf 1 1 numKVHeads headDim (some bufs.preparedRopeK)

  -- Step 3: Append K (after RoPE) and V to cache at position `pos` (fused single dispatch)
  let cwWx := (kvDim + 255) / 256
  if let some p ← kvCache.preparedCacheWriteKV.get then
    Hesper.WGSL.Execute.replayPreparedDispatch device p cwWx 1 1
  else
    let writeShader := fusedCacheWriteKVKernel numKVHeads maxSeqLen headDim kvDim
    let writeCacheKey : UInt64 := hash ("cwkv", numKVHeads, maxSeqLen, headDim, kvDim)
    Hesper.WGSL.Execute.executeShaderNamed device writeShader
      [("new_k", bufs.kRotBuf), ("new_v", bufs.vNewBuf),
       ("k_cache", kvCache.kBuf), ("v_cache", kvCache.vBuf),
       ("params", bufs.paramsBuf)]
      (Hesper.WGSL.Execute.ExecutionConfig.dispatch1D kvDim 256)
      (some writeCacheKey) (some kvCache.preparedCacheWriteKV)

  -- Step 4: Attention scores with GQA (dispatch size varies with cacheLen)
  let scoresWx := (numHeads * cacheLen + 255) / 256
  if let some p ← kvCache.preparedScores.get then
    Hesper.WGSL.Execute.replayPreparedDispatch device p scoresWx 1 1
  else
    let scale := 1.0 / headDim.toFloat.sqrt
    let scoresShader := cachedScoresKernel numHeads numKVHeads maxSeqLen headDim scale
    let scoresCacheKey : UInt64 := hash ("cs", numHeads, numKVHeads, maxSeqLen, headDim)
    Hesper.WGSL.Execute.executeShaderNamed device scoresShader
      [("q", bufs.qRotBuf), ("k_cache", kvCache.kBuf), ("scores", bufs.scoresBuf), ("params", bufs.paramsBuf)]
      (Hesper.WGSL.Execute.ExecutionConfig.dispatch1D (numHeads * cacheLen) 256)
      (some scoresCacheKey) (some kvCache.preparedScores)

  -- Step 5: Softmax (shared buffers only → shared PreparedDispatch)
  let softmaxWx := (numHeads * cacheLen + 255) / 256
  if let some p ← bufs.preparedSoftmax.get then
    Hesper.WGSL.Execute.replayPreparedDispatch device p softmaxWx 1 1
  else
    let softmaxShader := cachedSoftmaxKernel numHeads maxSeqLen
    let softmaxCacheKey : UInt64 := hash ("sm", numHeads, maxSeqLen)
    Hesper.WGSL.Execute.executeShaderNamed device softmaxShader
      [("input", bufs.scoresBuf), ("output", bufs.attnBuf), ("params", bufs.paramsBuf)]
      (Hesper.WGSL.Execute.ExecutionConfig.dispatch1D (numHeads * cacheLen) 256)
      (some softmaxCacheKey) (some bufs.preparedSoftmax)

  -- Step 6: Apply attention to V cache (uses kvCache.vBuf → per-layer)
  let applyWx := (numHeads * headDim + 255) / 256
  if let some p ← kvCache.preparedApply.get then
    Hesper.WGSL.Execute.replayPreparedDispatch device p applyWx 1 1
  else
    let applyShader := cachedApplyKernel numHeads numKVHeads maxSeqLen headDim
    let applyCacheKey : UInt64 := hash ("ca", numHeads, numKVHeads, maxSeqLen, headDim)
    Hesper.WGSL.Execute.executeShaderNamed device applyShader
      [("attn", bufs.attnBuf), ("v_cache", kvCache.vBuf), ("output", bufs.qRotBuf), ("params", bufs.paramsBuf)]
      (Hesper.WGSL.Execute.ExecutionConfig.dispatch1D (numHeads * headDim) 256)
      (some applyCacheKey) (some kvCache.preparedApply)

  -- Step 7: Sub-norm (if provided)
  let attnOutForO ← match subNorm with
    | some norm => do
      RMSNorm.forward device norm bufs.qRotBuf bufs.subNormBuf 1 256 (some bufs.rmsTempBuf)
      pure bufs.subNormBuf
    | none => pure bufs.qRotBuf

  -- Step 8: O projection (with optional fused residual add)
  match residualBuf with
  | some resBuf =>
    BitLinear.forwardWithResidual device layer.wO attnOutForO resBuf outputBuf 1
  | none =>
    BitLinear.forward device layer.wO attnOutForO outputBuf 1

  logVerbose "[Attention] ✓ Cached forward complete"

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
