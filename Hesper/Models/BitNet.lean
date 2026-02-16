import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WebGPU.BufferOps
import Hesper.Layers.Embedding
import Hesper.Layers.TransformerBlock
import Hesper.Layers.RMSNorm
import Hesper.Layers.BitLinear
import Hesper.Layers.Attention
import Hesper.Layers.Softmax
import Hesper.GGUF.Parser
import Hesper.GGUF.Loader
import Hesper.GGUF.Reader
import Hesper.WGSL.MatMul
import Hesper.WGSL.Monad
import Hesper.WGSL.Exp
import Hesper.Inference.Sampling
import Hesper.Logging

/-!
# BitNet Model - Complete Inference Pipeline

Implements the complete BitNet transformer model for text generation.

## Architecture

```
Input: token_ids [batch, seq_len]
  │
  ▼
[Embedding Layer]  ← TQ2_0 quantized
  │ [batch, seq_len, dim]
  │
  ├─► [Transformer Block 0]
  │       ├─► Attention + FFN
  │       └─► Residual connections
  │
  ├─► [Transformer Block 1]
  │       ...
  │
  ├─► [Transformer Block 31]
  │
  ▼ [batch, seq_len, dim]
[Final RMSNorm]
  │
  ▼
[LM Head] ← Project to vocabulary
  │ [batch, seq_len, vocab_size]
  │
  ▼
[Logits] → Sampling → Next token
```

## Model Configurations

### BitNet-3B
```
Vocabulary: 50,000 tokens
Embedding: 2560 dimensions
Layers: 32 transformer blocks
Heads: 32 attention heads
FFN hidden: 10,240 (4× expansion)
Context: 2048 tokens max
```

### BitNet-1.3B (smaller variant)
```
Vocabulary: 50,000 tokens
Embedding: 2048 dimensions
Layers: 24 transformer blocks
Heads: 16 attention heads
FFN hidden: 8,192
Context: 2048 tokens max
```

## Memory Footprint (BitNet-3B with TQ2_0)

**Model weights**:
```
Embedding:     50000 × 2560 × 0.25 = 32 MB
32 layers:     32 × 26 MB = 832 MB
Final norm:    2560 × 4 = 10 KB
LM head:       2560 × 50000 × 0.25 = 32 MB
─────────────────────────────────────────
Total:         ~896 MB

Compare to Float32: ~12 GB (13.4x savings!)
```

**Activations** (seq_len=2048):
```
Per layer peak: ~512 MB (attention scores)
With buffer reuse: ~512 MB total
```

**Total inference memory**: ~1.4 GB

## Performance (BitNet-3B on A100)

**Single token latency**:
```
Embedding:          0.01 ms
32 transformer layers: 870 ms (27 ms/layer avg)
Final norm:         0.01 ms
LM head:            13 ms
Sampling:           0.1 ms
─────────────────────────────────────────
Total:              ~883 ms/token

Throughput:         1.13 tokens/sec (base)
```

**With optimizations**:
```
Flash Attention:    440 ms (2× speedup)
Tiled matmul:       290 ms (1.5× additional)
Kernel fusion:      223 ms (1.3× additional)
─────────────────────────────────────────
Target:             ~200-250 ms/token (4-5 tokens/sec)
```

## Text Generation

### Greedy Decoding
```
while len(tokens) < max_tokens:
  logits = model(tokens)
  next_token = argmax(logits[-1])
  tokens.append(next_token)
```

### Top-k Sampling
```
logits = model(tokens)
top_k_logits, top_k_indices = topk(logits, k)
probs = softmax(top_k_logits / temperature)
next_token = sample(top_k_indices, probs)
```

### Nucleus (Top-p) Sampling
```
logits = model(tokens)
sorted_logits, sorted_indices = sort(logits, descending=True)
cumulative_probs = cumsum(softmax(sorted_logits))
nucleus = sorted_indices[cumulative_probs <= p]
next_token = sample(nucleus)
```

## References
- BitNet: https://arxiv.org/abs/2402.17764
- LLaMA: https://github.com/facebookresearch/llama
- llama.cpp: Main inference loop in llama.cpp
-/

namespace Hesper.Models.BitNet

open Hesper.WebGPU
open Hesper.Layers
open Hesper.GGUF
open Hesper.WGSL
open Hesper.WGSL.Monad
open Hesper.Logging (logVerbose isVerbose)

/-! ## Configuration -/

/-- Model configuration -/
structure Config where
  -- Architecture
  vocabSize : Nat := 128256
  dim : Nat := 2560
  numLayers : Nat := 30
  numHeads : Nat := 20       -- Query heads
  numKVHeads : Nat := 5      -- KV heads (GQA: each KV head serves 4 query heads)
  headDim : Nat := 128       -- Per-head dimension
  ffnDim : Nat := 6912
  maxSeqLen : Nat := 2048
  ropeBase : Float := 500000.0  -- RoPE frequency base
  -- Generation
  temperature : Float := 1.0
  topK : Nat := 40
  topP : Float := 0.9
  deriving Repr

/-- KV dimension (numKVHeads * headDim) -/
def Config.kvDim (cfg : Config) : Nat := cfg.numKVHeads * cfg.headDim

/-- Predefined configurations -/
def Config.bitnet2B : Config := {
  vocabSize := 128256
  dim := 2560
  numLayers := 30
  numHeads := 20
  numKVHeads := 5
  headDim := 128
  ffnDim := 6912
  maxSeqLen := 2048
  ropeBase := 500000.0
}

/-- Legacy alias -/
def Config.bitnet3B : Config := Config.bitnet2B

def Config.bitnet1_3B : Config := {
  vocabSize := 50000
  dim := 2048
  numLayers := 24
  numHeads := 16
  numKVHeads := 16
  headDim := 128
  ffnDim := 8192
  maxSeqLen := 2048
}

/-! ## Model Structure -/

/-- Complete BitNet model -/
structure BitNetModel where
  config : Config
  -- Layers
  embedding : Embedding.Embedding
  layers : Array TransformerBlock.TransformerBlock
  finalNorm : RMSNorm.RMSNorm
  -- LM head uses weight tying: reuses embedding table as [vocabSize, dim] matrix
  -- output = input @ embedding^T

/-! ## Model Creation -/

/-- Create BitNet model (placeholder - needs GGUF integration)

    @param device WebGPU device
    @param config Model configuration
-/
def create (device : Device) (config : Config) : IO BitNetModel := do
  IO.println "═══════════════════════════════════════════════"
  IO.println "   BitNet Model Initialization"
  IO.println "═══════════════════════════════════════════════"
  IO.println s!"Configuration: {config.numLayers} layers, {config.dim} dim, {config.numHeads} heads"
  IO.println ""

  -- This is a placeholder - actual implementation needs:
  -- 1. Load weights from GGUF file
  -- 2. Create embedding layer
  -- 3. Create all transformer blocks
  -- 4. Create final norm + LM head

  throw $ IO.userError "create() not yet implemented - use fromGGUF() to load a trained model"

/-! ## Forward Pass -/

/-- Execute forward pass through entire model

    @param device WebGPU device
    @param model BitNet model
    @param tokenIdsBuf Input token IDs [batch, seq_len]
    @param outputBuf Output logits [batch, seq_len, vocab_size]
    @param batchSize Batch size
    @param seqLen Sequence length
-/
def forward (device : Device) (model : BitNetModel)
            (tokenIdsBuf outputBuf : Buffer)
            (batchSize seqLen : Nat) : IO Unit := do
  logVerbose "═══════════════════════════════════════════════"
  logVerbose s!"  Forward Pass: batch={batchSize}, seq_len={seqLen}"
  logVerbose "═══════════════════════════════════════════════"

  let numElements := batchSize * seqLen * model.config.dim
  let tempSize := (numElements * 4).toUSize

  -- Allocate ping-pong buffers for layer outputs
  let buf1 ← createBuffer device { size := tempSize, usage := [.storage], mappedAtCreation := false }
  let buf2 ← createBuffer device { size := tempSize, usage := [.storage], mappedAtCreation := false }

  -- Pre-allocate layer buffers ONCE (reused across all 30 layers)
  let attnConfig : Attention.Config := {
    dim := model.config.dim
    numHeads := model.config.numHeads
    numKVHeads := model.config.numKVHeads
    headDim := model.config.headDim
    maxSeqLen := model.config.maxSeqLen
    useCausalMask := true
  }
  let layerBufs ← TransformerBlock.createLayerBuffers device
    model.config.dim model.config.ffnDim attnConfig batchSize seqLen

  let verbose ← isVerbose

  -- Step 1: Embedding lookup (outside batch - needs sync for debug)
  logVerbose "[1/4] Embedding lookup..."
  Embedding.forward device model.embedding tokenIdsBuf buf1 batchSize seqLen

  -- Debug: Check embedding output (needs GPU readback, so before batch)
  if verbose then
    let embDbg ← BufferOps.downloadFloatArray device buf1 (min 10 numElements)
    logVerbose s!"  [DEBUG] Embedding output (first 10): {embDbg.toList.take 10}"

  -- === BEGIN BATCHED EXECUTION ===
  -- All transformer layers + final norm + LM head recorded into single command buffer
  Hesper.WGSL.Execute.beginBatch device

  -- Step 2: Pass through all transformer layers
  logVerbose s!"[2/4] Transformer layers (×{model.config.numLayers})..."
  let mut currentBuf := buf1
  let mut nextBuf := buf2

  for layer in model.layers do
    TransformerBlock.forward device layer currentBuf nextBuf batchSize seqLen (some layerBufs)

    -- Ping-pong buffers
    let temp := currentBuf
    currentBuf := nextBuf
    nextBuf := temp

  -- Step 3: Final normalization
  logVerbose "[3/4] Final RMSNorm..."
  RMSNorm.forward device model.finalNorm currentBuf nextBuf (batchSize * seqLen) 256 (some layerBufs.rmsTempBuf)

  -- Step 4: LM head projection to vocabulary (weight tying with embedding)
  logVerbose "[4/4] LM head projection (weight-tied)..."
  let lmHeadConfig : Hesper.WGSL.MatMul.Config := {
    M := batchSize * seqLen,
    N := model.config.vocabSize,
    K := model.config.dim
  }
  match model.embedding.f16Table with
  | some f16Buf =>
    if model.config.dim % 8 == 0 then
      Hesper.WGSL.MatMul.executeMatMulTransposeF16Shared device nextBuf f16Buf outputBuf lmHeadConfig
    else
      Hesper.WGSL.MatMul.executeMatMulTransposeF16 device nextBuf f16Buf outputBuf lmHeadConfig
  | none =>
    Hesper.WGSL.MatMul.executeMatMulTranspose device nextBuf model.embedding.embeddingTable outputBuf lmHeadConfig

  -- === END BATCHED EXECUTION ===
  -- Submit all recorded dispatches and wait once
  Hesper.WGSL.Execute.endBatch device

  logVerbose "═══════════════════════════════════════════════"
  logVerbose "  ✓ Forward pass complete"
  logVerbose "═══════════════════════════════════════════════"

/-! ## KV Cache State -/

/-- Full KV cache state for incremental inference -/
structure KVCacheState where
  kvCaches : Array Attention.KVCache   -- Per-layer KV caches
  fusedRefs : Array TransformerBlock.FusedLayerRefs  -- Per-layer fused PreparedDispatch refs
  layerBufs : TransformerBlock.CachedLayerBuffers  -- Shared temp buffers (single-token)
  buf1 : Buffer        -- [dim] ping-pong buffer
  buf2 : Buffer        -- [dim] ping-pong buffer
  logitsBuf : Buffer   -- [vocabSize]
  argmaxBuf : Buffer   -- [1] u32 for GPU-side argmax result
  tokenBuf : Buffer    -- [1] u32 for single-token upload (reusable)

/-- Create KV cache state for the model -/
def createKVCacheState (device : Device) (model : BitNetModel) : IO KVCacheState := do
  let cfg := model.config
  let attnConfig : Attention.Config := {
    dim := cfg.dim, numHeads := cfg.numHeads, numKVHeads := cfg.numKVHeads,
    headDim := cfg.headDim, maxSeqLen := cfg.maxSeqLen, useCausalMask := true
  }
  -- Create per-layer KV caches and fused PreparedDispatch refs
  let mut kvCaches := Array.mkEmpty cfg.numLayers
  let mut fusedRefs := Array.mkEmpty cfg.numLayers
  for _ in [0:cfg.numLayers] do
    kvCaches := kvCaches.push (← Attention.createKVCache device attnConfig)
    fusedRefs := fusedRefs.push (← TransformerBlock.createFusedLayerRefs)
  -- Create shared layer buffers
  let layerBufs ← TransformerBlock.createCachedLayerBuffers device cfg.dim cfg.ffnDim attnConfig
  let mkBuf := fun size => createBuffer device { size := size, usage := [.storage], mappedAtCreation := false }
  let mkBufRW := fun size => createBuffer device { size := size, usage := [.storage, .copySrc, .copyDst], mappedAtCreation := false }
  pure {
    kvCaches := kvCaches
    fusedRefs := fusedRefs
    layerBufs := layerBufs
    buf1 := ← mkBuf (cfg.dim * 4).toUSize
    buf2 := ← mkBuf (cfg.dim * 4).toUSize
    logitsBuf := ← mkBuf (cfg.vocabSize * 4).toUSize
    argmaxBuf := ← mkBufRW 4  -- Single u32 for argmax result
    tokenBuf := ← mkBufRW 4   -- Single u32 for token upload
  }

/-- Run single-token forward pass with KV cache.
    Processes one token at position `pos`, using cached K/V from past tokens.
    Returns logits in `cacheState.logitsBuf`. -/
def forwardSingleToken (device : Device) (model : BitNetModel)
                       (tokenId : Nat) (pos : Nat) (cacheState : KVCacheState) : IO Unit := do
  logVerbose s!"[SingleToken] pos={pos}, tokenId={tokenId}"

  -- Step 1: Embedding lookup (single token) - reuse pre-allocated tokenBuf
  let tokenBytes := Hesper.WebGPU.BufferOps.uint32ToBytes tokenId.toUInt32
  writeBuffer device cacheState.tokenBuf 0 tokenBytes
  Embedding.forward device model.embedding cacheState.tokenBuf cacheState.buf1 1 1

  -- === BEGIN BATCHED EXECUTION ===
  Hesper.WGSL.Execute.beginBatch device

  -- Step 2: Pass through all transformer layers with KV cache
  let mut currentBuf := cacheState.buf1
  let mut nextBuf := cacheState.buf2
  let mut layerIdx := 0

  for layer in model.layers do
    if h : layerIdx < cacheState.kvCaches.size then
      let kvCache := cacheState.kvCaches[layerIdx]
      let fusedRef := if h2 : layerIdx < cacheState.fusedRefs.size then
        some cacheState.fusedRefs[layerIdx]
      else none
      TransformerBlock.forwardWithCache device layer currentBuf nextBuf pos kvCache (some cacheState.layerBufs) fusedRef
      let temp := currentBuf; currentBuf := nextBuf; nextBuf := temp
    layerIdx := layerIdx + 1

  -- Step 3: Final normalization (single token)
  RMSNorm.forward device model.finalNorm currentBuf nextBuf 1 256

  -- Step 4: LM head (1×dim @ dim×vocab)
  let lmHeadConfig : Hesper.WGSL.MatMul.Config := {
    M := 1, N := model.config.vocabSize, K := model.config.dim
  }
  match model.embedding.f16Table with
  | some f16Buf =>
    -- Use shared memory kernel when K is divisible by 8 (true for BitNet: K=2560)
    if model.config.dim % 8 == 0 then
      Hesper.WGSL.MatMul.executeMatMulTransposeF16Shared device nextBuf f16Buf cacheState.logitsBuf lmHeadConfig
    else
      Hesper.WGSL.MatMul.executeMatMulTransposeF16 device nextBuf f16Buf cacheState.logitsBuf lmHeadConfig
  | none =>
    Hesper.WGSL.MatMul.executeMatMulTranspose device nextBuf model.embedding.embeddingTable cacheState.logitsBuf lmHeadConfig

  -- === END BATCHED EXECUTION ===
  Hesper.WGSL.Execute.endBatch device

/-! ## GPU Argmax -/

/-- GPU argmax kernel: find index of maximum value using parallel reduction.
    Uses 1 workgroup of `workgroupSize` threads. Each thread scans a strided
    portion of the input, then shared memory reduction finds the global max.
    Output: single u32 (token index of max value). -/
def argmaxKernel (vocabSize : Nat) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let tid ← ShaderM.localId
  let tid := Exp.vec3X tid

  -- Shared memory: values and indices
  ShaderM.sharedNamed "shared_vals" (.array (.scalar .f32) workgroupSize)
  ShaderM.sharedNamed "shared_idxs" (.array (.scalar .u32) workgroupSize)

  -- Buffers
  let _logits ← ShaderM.declareInputBuffer "logits" (.array (.scalar .f32) vocabSize)
  let _result ← ShaderM.declareOutputBuffer "result" (.array (.scalar .u32) 1)

  -- Phase 1: Each thread finds local max over its strided portion
  ShaderM.varNamed "local_max" (.scalar .f32) (Exp.litF32 (-1.0e38))
  ShaderM.varNamed "local_idx" (.scalar .u32) (Exp.litU32 0)
  let localMax : Exp (.scalar .f32) := Exp.var "local_max"
  let localIdx : Exp (.scalar .u32) := Exp.var "local_idx"

  ShaderM.loop tid (Exp.litU32 vocabSize) (Exp.litU32 workgroupSize) fun i => do
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := vocabSize) "logits" i
    ShaderM.if_ (Exp.gt val localMax) (do
      ShaderM.assign "local_max" val
      ShaderM.assign "local_idx" i
    ) (pure ())

  -- Write local results to shared memory
  ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_vals" tid localMax
  ShaderM.writeWorkgroup (ty := .scalar .u32) "shared_idxs" tid localIdx
  ShaderM.barrier

  -- Phase 2: Tree reduction (compare max values, keep winner's index)
  let mut stride := workgroupSize / 2
  while stride > 0 do
    ShaderM.if_ (Exp.lt tid (Exp.litU32 stride)) (do
      let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_vals" tid
      let b ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_vals" (Exp.add tid (Exp.litU32 stride))
      ShaderM.if_ (Exp.gt b a) (do
        ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_vals" tid b
        let bIdx ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := workgroupSize) "shared_idxs" (Exp.add tid (Exp.litU32 stride))
        ShaderM.writeWorkgroup (ty := .scalar .u32) "shared_idxs" tid bIdx
      ) (pure ())
    ) (pure ())
    ShaderM.barrier
    stride := stride / 2

  -- Thread 0 writes the final argmax index
  ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
    let maxIdx ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := workgroupSize) "shared_idxs" (Exp.litU32 0)
    ShaderM.writeBuffer (ty := .scalar .u32) "result" (Exp.litU32 0) maxIdx
  ) (pure ())

/-- Execute GPU argmax on logits buffer.
    Returns the token index with maximum logit value.
    Downloads only 4 bytes instead of vocabSize × 4 bytes. -/
def gpuArgmax (device : Device) (logitsBuf argmaxBuf : Buffer) (vocabSize : Nat) : IO Nat := do
  let shader := argmaxKernel vocabSize
  let namedBuffers := [("logits", logitsBuf), ("result", argmaxBuf)]
  let execConfig : Hesper.WGSL.Execute.ExecutionConfig := {
    workgroupSize := { x := 256, y := 1, z := 1 }
    numWorkgroups := (1, 1, 1)
  }
  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig
  -- Download single u32
  let bytes ← mapBufferRead device argmaxBuf 0 4
  let tokenId := Hesper.WebGPU.BufferOps.bytesToUInt32 bytes 0
  pure tokenId.toNat

/-! ## Text Generation -/

/-- Generate text using greedy decoding

    @param device WebGPU device
    @param model BitNet model
    @param promptTokens Initial prompt tokens
    @param maxTokens Maximum tokens to generate
    @return Generated token sequence
-/
def generate (device : Device) (model : BitNetModel)
             (promptTokens : Array Nat) (maxTokens : Nat)
             (strategy : Hesper.Inference.Sampling.Strategy := .Greedy)
             (eosToken : Option Nat := none)
             (showStats : Bool := false)
    : IO (Array Nat) := do
  logVerbose "═══════════════════════════════════════════════"
  logVerbose "  Text Generation (KV Cache)"
  logVerbose "═══════════════════════════════════════════════"
  logVerbose s!"Prompt length: {promptTokens.size} tokens"
  logVerbose s!"Generating up to {maxTokens} new tokens..."
  logVerbose s!"Strategy: {strategy}"
  logVerbose ""

  -- Create KV cache state (pre-allocated buffers for all layers)
  let cacheState ← createKVCacheState device model

  let mut tokens := promptTokens
  let mut rng := Hesper.Inference.Sampling.RNG.create (some 42)

  -- Phase 1: Process prompt tokens one at a time (populating KV cache)
  IO.println s!"[Prefill] Processing {promptTokens.size} prompt tokens..."
  let prefillStart ← IO.monoNanosNow
  for i in [0:promptTokens.size] do
    if i >= model.config.maxSeqLen then break
    forwardSingleToken device model promptTokens[i]! i cacheState
  let prefillEnd ← IO.monoNanosNow
  let prefillMs := (prefillEnd - prefillStart).toFloat / 1_000_000.0
  IO.println s!"[Prefill] Done in {prefillMs} ms ({prefillMs / promptTokens.size.toFloat} ms/token)"

  -- Phase 2: Generate new tokens using KV cache
  let isGreedy := match strategy with
    | .Greedy => true
    | _ => false
  let genStart ← IO.monoNanosNow
  let mut genTokenCount : Nat := 0
  for step in [0:maxTokens] do
    if tokens.size >= model.config.maxSeqLen then
      IO.println s!"Reached max sequence length ({model.config.maxSeqLen})"
      break

    -- Sample next token
    let mut nextToken := 0
    if isGreedy then
      -- GPU-side argmax: download 4 bytes instead of 512KB
      nextToken ← gpuArgmax device cacheState.logitsBuf cacheState.argmaxBuf model.config.vocabSize
    else
      -- Non-greedy: download full logits for CPU sampling
      let logits ← Hesper.WebGPU.BufferOps.downloadFloatArray device cacheState.logitsBuf model.config.vocabSize
      let (tok, newRng) := Hesper.Inference.Sampling.sampleWithRNG logits strategy rng
      rng := newRng
      nextToken := tok

    logVerbose s!"Step {step+1}/{maxTokens}: token={nextToken}"

    tokens := tokens.push nextToken
    genTokenCount := genTokenCount + 1

    -- Early stopping on EOS
    match eosToken with
    | some eos =>
      if nextToken == eos then
        IO.println "  EOS token, stopping"
        break
    | none => pure ()

    -- Run forward pass for the new token
    let newPos := tokens.size - 1
    if newPos < model.config.maxSeqLen then
      forwardSingleToken device model nextToken newPos cacheState

  let genEnd ← IO.monoNanosNow
  let genMs := (genEnd - genStart).toFloat / 1_000_000.0
  let msPerToken := if genTokenCount > 0 then genMs / genTokenCount.toFloat else 0.0
  let tps := if msPerToken > 0 then 1000.0 / msPerToken else 0.0
  IO.println ""
  IO.println s!"Generated {genTokenCount} tokens in {genMs} ms"
  IO.println s!"  {msPerToken} ms/token = {tps} tokens/sec"

  if showStats then
    let blHits ← Hesper.Layers.BitLinear.preparedHitsRef.get
    let blMisses ← Hesper.Layers.BitLinear.preparedMissesRef.get
    let rmsHits ← Hesper.Layers.RMSNorm.preparedHitsRef.get
    let rmsMisses ← Hesper.Layers.RMSNorm.preparedMissesRef.get
    IO.println s!"  BitLinear PreparedDispatch: {blHits} hits, {blMisses} misses"
    IO.println s!"  RMSNorm PreparedDispatch: {rmsHits} hits, {rmsMisses} misses"
    let (plHits, plMisses) ← Hesper.WGSL.Execute.getPipelineCacheStats
    let (bgHits, bgMisses) ← Hesper.WGSL.Execute.getBindGroupCacheStats
    IO.println s!"  Pipeline cache: {plHits} hits, {plMisses} misses"
    IO.println s!"  BindGroup cache: {bgHits} hits, {bgMisses} misses"

  pure tokens

/-- Generate text WITHOUT KV cache (naive quadratic approach).
    Kept for validation/comparison with cached generate. -/
def generateNaive (device : Device) (model : BitNetModel)
             (promptTokens : Array Nat) (maxTokens : Nat)
             (strategy : Hesper.Inference.Sampling.Strategy := .Greedy)
             (eosToken : Option Nat := none)
    : IO (Array Nat) := do
  let mut tokens := promptTokens
  let mut rng := Hesper.Inference.Sampling.RNG.create (some 42)

  for step in [0:maxTokens] do
    if tokens.size >= model.config.maxSeqLen then break
    let seqLen := tokens.size
    let tokenIdsBuf ← Hesper.WebGPU.BufferOps.uploadTokens device tokens
    let logitsBuf ← Hesper.WebGPU.BufferOps.createLogitsBuffer device 1 seqLen model.config.vocabSize
    forward device model tokenIdsBuf logitsBuf 1 seqLen
    let lastLogits ← Hesper.WebGPU.BufferOps.downloadLastLogits device logitsBuf 1 seqLen model.config.vocabSize
    let (nextToken, newRng) := Hesper.Inference.Sampling.sampleWithRNG lastLogits strategy rng
    rng := newRng
    tokens := tokens.push nextToken
    match eosToken with
    | some eos => if nextToken == eos then break
    | none => pure ()

  pure tokens

/-! ## Sampling Strategies -/

/-- Greedy sampling: select token with highest probability

    @param logits Logit scores [vocab_size]
    @return Selected token ID
-/
def sampleGreedy (logits : Array Float) : Nat :=
  Hesper.Inference.Sampling.sampleGreedy logits

/-- Top-k sampling: sample from k tokens with highest probability

    @param logits Logit scores [vocab_size]
    @param k Number of top candidates
    @param temperature Sampling temperature (higher = more random)
    @param rng Random number generator
    @return (Selected token ID, new RNG)
-/
def sampleTopK (logits : Array Float) (k : Nat) (temperature : Float)
              (rng : Hesper.Inference.Sampling.RNG)
    : Nat × Hesper.Inference.Sampling.RNG :=
  let strategy := Hesper.Inference.Sampling.Strategy.TopK k temperature
  Hesper.Inference.Sampling.sampleWithRNG logits strategy rng

/-- Nucleus (top-p) sampling: sample from smallest set with cumulative probability >= p

    @param logits Logit scores [vocab_size]
    @param p Cumulative probability threshold (typically 0.9)
    @param temperature Sampling temperature
    @param rng Random number generator
    @return (Selected token ID, new RNG)
-/
def sampleNucleus (logits : Array Float) (p : Float) (temperature : Float)
                  (rng : Hesper.Inference.Sampling.RNG)
    : Nat × Hesper.Inference.Sampling.RNG :=
  let strategy := Hesper.Inference.Sampling.Strategy.Nucleus p temperature
  Hesper.Inference.Sampling.sampleWithRNG logits strategy rng

/-! ## GGUF Integration -/

/-- Extract model configuration from GGUF metadata

    Reads metadata keys to determine model architecture.

    @param gguf Parsed GGUF file
    @return Model configuration
-/
def extractConfig (gguf : Hesper.GGUF.GGUFFile) : IO Config := do
  -- For now, use default config for BitNet-b1.58-2B-4T
  -- TODO: Parse GGUF metadata keys to determine architecture
  IO.println "Using default BitNet-2B configuration"
  return Config.bitnet2B

/-- Load BitNet model from pre-loaded GGUF object

    @param device WebGPU device
    @param gguf Already parsed GGUF file object
    @param config Optional configuration (uses defaults if not provided)
    @return Loaded model

    NOTE: This function prevents premature GC of the GGUF object during loading
-/
def fromGGUFObject (device : Device) (gguf : Hesper.GGUF.GGUFFile) (config : Option Config := none) : IO BitNetModel := do
  IO.println "═══════════════════════════════════════════════"
  IO.println s!"  Loading BitNet Model from GGUF Object"
  IO.println "═══════════════════════════════════════════════"
  IO.println s!"  Using pre-loaded GGUF with {gguf.tensors.size} tensors"
  IO.println ""

  -- Step 2: Extract or use provided configuration
  IO.println "[2/5] Extracting model configuration..."
  let cfg ← match config with
    | some c => pure c
    | none => extractConfig gguf

  IO.println s!"  ✓ Model: {cfg.numLayers} layers, {cfg.dim} dim, {cfg.numHeads} heads"

  -- Step 3: Load embedding layer
  IO.println "[3/5] Loading embedding layer..."

  -- Check tensor type (F16 or TQ2_0)
  let embTensorInfo ← match Hesper.GGUF.Loader.findTensor gguf "token_embd.weight" with
    | .ok info => pure info
    | .error e => throw $ IO.userError e

  let embConfig : Embedding.Config := {
    vocabSize := cfg.vocabSize,
    dim := cfg.dim
  }

  let embedding ← match embTensorInfo.ggmlType with
    | .F16 =>
      -- BitNet models use F16 for embeddings
      IO.println "  Using F16 embeddings"
      let embData ← Hesper.GGUF.Loader.extractF16Tensor gguf "token_embd.weight"
      Embedding.createFromF16 device embConfig embData
    | .IQ2_XXS =>
      -- TQ2_0 ternary embeddings
      IO.println "  Using TQ2_0 ternary embeddings"
      let (embPackedData, embScalesData) ← Hesper.GGUF.Loader.extractTQ2_0Tensor gguf "token_embd.weight"
      Embedding.create device embConfig embPackedData embScalesData
    | _ =>
      throw $ IO.userError s!"Unsupported embedding tensor type: {toString embTensorInfo.ggmlType}"

  -- Step 4: Load all transformer layers
  IO.println s!"[4/5] Loading {cfg.numLayers} transformer blocks..."
  let mut layers := Array.mkEmpty cfg.numLayers

  for layerIdx in [0:cfg.numLayers] do
    IO.println s!"  Loading layer {layerIdx}..."

    -- Load RMSNorm scales (Float32)
    let attnNormData ← Hesper.GGUF.Loader.extractFloat32Tensor gguf s!"blk.{layerIdx}.attn_norm.weight"
    let attnSubNormData ← Hesper.GGUF.Loader.extractFloat32Tensor gguf s!"blk.{layerIdx}.attn_sub_norm.weight"
    let ffnNormData ← Hesper.GGUF.Loader.extractFloat32Tensor gguf s!"blk.{layerIdx}.ffn_norm.weight"
    let ffnSubNormData ← Hesper.GGUF.Loader.extractFloat32Tensor gguf s!"blk.{layerIdx}.ffn_sub_norm.weight"

    -- Load attention projections (i2_s ternary format - raw packed to GPU)
    let qConfig : BitLinear.Config := { inDim := cfg.dim, outDim := cfg.dim, batchSize := 1 }
    let wQ ← match Hesper.GGUF.Loader.getI2_S_Tensor gguf s!"blk.{layerIdx}.attn_q.weight" with
      | .ok (packedData, scale, _numElements) =>
        BitLinear.create device qConfig packedData scale
      | .error e => throw $ IO.userError e

    let kConfig : BitLinear.Config := { inDim := cfg.dim, outDim := cfg.kvDim, batchSize := 1 }
    let wK ← match Hesper.GGUF.Loader.getI2_S_Tensor gguf s!"blk.{layerIdx}.attn_k.weight" with
      | .ok (packedData, scale, _numElements) =>
        BitLinear.create device kConfig packedData scale
      | .error e => throw $ IO.userError e

    let vConfig : BitLinear.Config := { inDim := cfg.dim, outDim := cfg.kvDim, batchSize := 1 }
    let wV ← match Hesper.GGUF.Loader.getI2_S_Tensor gguf s!"blk.{layerIdx}.attn_v.weight" with
      | .ok (packedData, scale, _numElements) =>
        BitLinear.create device vConfig packedData scale
      | .error e => throw $ IO.userError e

    let oConfig : BitLinear.Config := { inDim := cfg.dim, outDim := cfg.dim, batchSize := 1 }
    let wO ← match Hesper.GGUF.Loader.getI2_S_Tensor gguf s!"blk.{layerIdx}.attn_output.weight" with
      | .ok (packedData, scale, _numElements) =>
        BitLinear.create device oConfig packedData scale
      | .error e => throw $ IO.userError e

    -- Create RoPE layer
    let ropeConfig : Hesper.Layers.RoPE.Config := {
      dim := cfg.dim,
      maxSeqLen := cfg.maxSeqLen,
      base := cfg.ropeBase
    }
    let rope ← Hesper.Layers.RoPE.create ropeConfig

    -- Create Softmax layer
    let headDim := cfg.headDim
    let softmaxConfig : Hesper.Layers.Softmax.Config := {
      numRows := cfg.numHeads,
      rowSize := cfg.maxSeqLen,
      useMask := true
    }
    let softmax ← Hesper.Layers.Softmax.create softmaxConfig

    -- Create attention layer
    let attnConfig : Attention.Config := {
      dim := cfg.dim,
      numHeads := cfg.numHeads,
      numKVHeads := cfg.numKVHeads,
      headDim := cfg.headDim,
      maxSeqLen := cfg.maxSeqLen,
      useCausalMask := true
    }
    let attention : Attention.Attention := {
      config := attnConfig,
      wQ := wQ,
      wK := wK,
      wV := wV,
      wO := wO,
      rope := rope,
      softmax := softmax
    }

    -- Load FFN projections (i2_s ternary format - raw packed to GPU)
    let gateConfig : BitLinear.Config := { inDim := cfg.dim, outDim := cfg.ffnDim, batchSize := 1 }
    let ffnGate ← match Hesper.GGUF.Loader.getI2_S_Tensor gguf s!"blk.{layerIdx}.ffn_gate.weight" with
      | .ok (packedData, scale, _numElements) =>
        BitLinear.create device gateConfig packedData scale
      | .error e => throw $ IO.userError e

    let upConfig : BitLinear.Config := { inDim := cfg.dim, outDim := cfg.ffnDim, batchSize := 1 }
    let ffnUp ← match Hesper.GGUF.Loader.getI2_S_Tensor gguf s!"blk.{layerIdx}.ffn_up.weight" with
      | .ok (packedData, scale, _numElements) =>
        BitLinear.create device upConfig packedData scale
      | .error e => throw $ IO.userError e

    let downConfig : BitLinear.Config := { inDim := cfg.ffnDim, outDim := cfg.dim, batchSize := 1 }
    let ffnDown ← match Hesper.GGUF.Loader.getI2_S_Tensor gguf s!"blk.{layerIdx}.ffn_down.weight" with
      | .ok (packedData, scale, _numElements) =>
        BitLinear.create device downConfig packedData scale
      | .error e => throw $ IO.userError e

    -- Create transformer block
    let blockConfig : TransformerBlock.Config := {
      layerIdx := layerIdx,
      dim := cfg.dim,
      ffnDim := cfg.ffnDim,
      numHeads := cfg.numHeads,
      maxSeqLen := cfg.maxSeqLen
    }

    let block ← TransformerBlock.createWithLayers device blockConfig
      attnNormData attnSubNormData ffnNormData ffnSubNormData
      attention
      ffnGate ffnUp ffnDown

    layers := layers.push block

  -- Step 5: Load final norm and LM head
  IO.println "[5/5] Loading final normalization and LM head..."
  let finalNormData ← Hesper.GGUF.Loader.extractFloat32Tensor gguf "output_norm.weight"
  let finalNormConfig : RMSNorm.Config := { dim := cfg.dim }
  let finalNorm ← RMSNorm.create device finalNormConfig finalNormData

  -- LM head: Uses weight tying with embedding table (no separate weights needed)
  IO.println "[LM Head] Using weight tying with embedding layer (no extra weights)"

  IO.println ""
  IO.println "═══════════════════════════════════════════════"
  IO.println "  ✓ Model loaded successfully!"
  IO.println "═══════════════════════════════════════════════"

  return {
    config := cfg,
    embedding := embedding,
    layers := layers,
    finalNorm := finalNorm,
  }

/-! ## Utilities -/

/-- Load BitNet model from GGUF file path (convenience wrapper)

    @param device WebGPU device
    @param ggufPath Path to GGUF file
    @param config Optional configuration (uses defaults if not provided)
    @return Loaded model

    NOTE: This loads the GGUF file and calls fromGGUFObject
-/
def fromGGUF (device : Device) (ggufPath : String) (config : Option Config := none) : IO BitNetModel := do
  -- Load GGUF file
  let gguf ← loadGGUF ggufPath
  -- Call the object-based loader
  fromGGUFObject device gguf config

/-- Print model statistics

    @param model BitNet model
-/
def printStats (model : BitNetModel) : IO Unit := do
  IO.println "═══════════════════════════════════════════════"
  IO.println "  Model Statistics"
  IO.println "═══════════════════════════════════════════════"
  IO.println s!"Vocabulary size:     {model.config.vocabSize}"
  IO.println s!"Embedding dimension: {model.config.dim}"
  IO.println s!"Number of layers:    {model.config.numLayers}"
  IO.println s!"Attention heads:     {model.config.numHeads}"
  IO.println s!"FFN hidden size:     {model.config.ffnDim}"
  IO.println s!"Max sequence length: {model.config.maxSeqLen}"
  IO.println ""

  -- Estimate parameter count
  let embParams := model.config.vocabSize * model.config.dim
  let attnParams := 4 * model.config.dim * model.config.dim  -- Q, K, V, O
  let ffnParams := 3 * model.config.dim * model.config.ffnDim  -- Gate, Up, Down
  let layerParams := attnParams + ffnParams
  let totalParams := embParams + (model.config.numLayers * layerParams) + embParams  -- +emb for LM head

  let paramsB := totalParams.toFloat / 1e9

  IO.println s!"Total parameters:    {paramsB}B"
  IO.println s!"  Embedding:         {embParams.toFloat / 1e6}M"
  IO.println s!"  Transformers:      {(model.config.numLayers * layerParams).toFloat / 1e9}B"
  IO.println s!"  LM Head:           {embParams.toFloat / 1e6}M"
  IO.println ""

  -- Estimate memory with TQ2_0 quantization
  let memoryMB := (totalParams.toFloat * 0.25) / 1e6  -- 0.25 bytes per param

  IO.println s!"Memory (TQ2_0):      {memoryMB} MB"
  IO.println s!"Memory (Float32):    {totalParams.toFloat * 4 / 1e6} MB"
  IO.println s!"Compression ratio:   16× savings"
  IO.println "═══════════════════════════════════════════════"

end Hesper.Models.BitNet
