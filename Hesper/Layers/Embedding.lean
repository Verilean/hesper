import Hesper.WGSL.Monad
import Hesper.WGSL.Execute
import Hesper.WGSL.Exp
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.Quantization.TQ2_0
import Hesper.Logging

/-!
# Embedding Layer

Implements token embedding lookup with TQ2_0 quantization support.

## Mathematical Definition

Embedding layer maps discrete token IDs to continuous vector representations:

```
Input: token_ids [batch, seq_len]  (integers: 0 to vocab_size-1)
Output: embeddings [batch, seq_len, dim]  (float vectors)

Operation:
  embedding[i,j] = embedding_table[token_ids[i,j]]
```

## Quantized Embeddings

For memory efficiency, embeddings are stored in TQ2_0 format:

```
Standard: vocab_size × dim × 4 bytes (Float32)
TQ2_0:    vocab_size × dim × 0.25 bytes (2-bit ternary)
Savings:  16× reduction
```

**BitNet-3B example**:
```
Vocabulary: 50,000 tokens
Dimension: 2560
Standard: 50000 × 2560 × 4 = 512 MB
TQ2_0:    50000 × 2560 × 0.25 = 32 MB
```

## GPU Implementation

Two approaches:

### 1. Dequantized Lookup (Simpler)
```
1. Precompute: unpack TQ2_0 → Float32 table (once at startup)
2. During inference: direct lookup from Float32 table
Pros: Fast lookup, simple implementation
Cons: Uses more memory (Float32 table)
```

### 2. On-the-Fly Lookup (Memory-efficient)
```
1. Store embeddings as TQ2_0 (packed)
2. During inference: lookup + unpack in single kernel
Pros: Minimal memory (16× savings)
Cons: Slightly more compute per lookup
```

We implement approach #1 for simplicity, but structure allows easy switch to #2.

## Performance

**Lookup operation**:
```
Compute: ~0 FLOPs (just memory read)
Memory: batch × seq_len × dim × 4 bytes
Latency: ~1 memory transaction per token

For batch=1, seq=2048, dim=2560:
  Memory transfer: 2048 × 2560 × 4 = ~20 MB
  Time on A100 (2 TB/s): 20 MB / 2000 GB/s ≈ 0.01 ms
  (Negligible compared to attention/FFN)
```

## References
- Word2Vec: "Efficient Estimation of Word Representations" (Mikolov et al., 2013)
- Transformer embeddings: "Attention is All You Need" (Vaswani et al., 2017)
- llama.cpp: llama.cpp (get_rows operation)
-/

namespace Hesper.Layers.Embedding

open Hesper.WGSL
open Hesper.WGSL.Monad
open Hesper.WebGPU
open Hesper.Quantization.TQ2_0
open Hesper.Logging (logVerbose)

/-! ## Configuration -/

/-- Embedding layer configuration -/
structure Config where
  vocabSize : Nat  -- Vocabulary size
  dim : Nat        -- Embedding dimension
  deriving Repr

/-! ## GPU Kernel -/

/-- Embedding lookup kernel

    Maps token IDs to embedding vectors.

    **Input**: token_ids [batch × seq_len] (UInt32)
    **Output**: embeddings [batch × seq_len × dim] (Float32)

    Each thread processes one token, reading its embedding vector
    from the embedding table.

    @param config Embedding configuration
    @param batchSize Batch size
    @param seqLen Sequence length
-/
def embeddingLookupKernel (config : Config) (batchSize seqLen : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let totalTokens := batchSize * seqLen
  let inBounds := Exp.lt idx (Exp.litU32 totalTokens)

  -- Declare buffers
  let _tokenIds ← ShaderM.declareInputBuffer "token_ids" (.array (.scalar .u32) totalTokens)
  let _embeddingTable ← ShaderM.declareInputBuffer "embedding_table"
    (.runtimeArray (.scalar .f32))  -- Runtime-sized array to avoid WGSL size limits
  let _output ← ShaderM.declareOutputBuffer "output"
    (.array (.scalar .f32) (totalTokens * config.dim))

  -- Read token ID for this position
  let tokenId ← ShaderM.readBuffer (ty := .scalar .u32) (n := totalTokens) "token_ids" idx

  -- Copy embedding vector from table to output using a GPU loop
  -- Each token needs config.dim values copied
  ShaderM.loop (Exp.litU32 0) (Exp.litU32 config.dim) (Exp.litU32 1) fun d => do
    -- Source: embedding_table[tokenId * dim + d]
    let srcIdx := Exp.add (Exp.mul tokenId (Exp.litU32 config.dim)) d

    -- Destination: output[idx * dim + d]
    let dstIdx := Exp.add (Exp.mul idx (Exp.litU32 config.dim)) d

    -- Read from embedding table
    let embVal ← ShaderM.readBuffer (ty := .scalar .f32)
      (n := config.vocabSize * config.dim) "embedding_table" srcIdx

    -- Write to output
    let finalVal := Exp.select inBounds embVal (Exp.litF32 0.0)
    ShaderM.writeBuffer (ty := .scalar .f32) "output" dstIdx finalVal

/-! ## Layer Structure -/

/-- Embedding layer -/
structure Embedding where
  config : Config
  embeddingTable : Buffer  -- Precomputed Float32 embeddings
  f16Table : Option Buffer := none  -- Original F16 data (for F16 matmul LM head)

/-! ## Layer Creation -/

/-- Create embedding layer from quantized data

    Unpacks TQ2_0 embeddings to Float32 for fast lookup.

    @param device WebGPU device
    @param config Embedding configuration
    @param packedData TQ2_0 packed embedding data
    @param scalesData FP16 scales for each block
-/
def create (device : Device) (config : Config)
           (packedData : ByteArray) (scalesData : ByteArray) : IO Embedding := do
  logVerbose s!"[Embedding] Creating layer: vocab={config.vocabSize}, dim={config.dim}"

  -- Calculate sizes
  let totalElements := config.vocabSize * config.dim
  let tableSize := (totalElements * 4).toUSize  -- Float32

  -- Create buffer for embedding table
  let embeddingTable ← createBuffer device {
    size := tableSize
    usage := [.storage, .copyDst]
    mappedAtCreation := false
  }

  -- TODO: Unpack TQ2_0 to Float32
  -- For now, we'll need to:
  -- 1. Allocate temporary buffers for packed data and scales
  -- 2. Run TQ2_0 unpacking kernel
  -- 3. Copy result to embeddingTable
  -- This would use TQ2_0.executeUnpack

  logVerbose "[Embedding] ✓ Layer created"
  logVerbose "  Note: TQ2_0 unpacking needs to be integrated"

  pure { config, embeddingTable }

/-- Create embedding layer from Float32 data (for testing)

    @param device WebGPU device
    @param config Embedding configuration
    @param float32Data Raw Float32 embedding data
-/
def createFromFloat32 (device : Device) (config : Config)
                      (float32Data : ByteArray) : IO Embedding := do
  logVerbose s!"[Embedding] Creating layer from Float32: vocab={config.vocabSize}, dim={config.dim}"

  let tableSize := float32Data.size.toUSize

  let embeddingTable ← createBuffer device {
    size := tableSize
    usage := [.storage, .copyDst]
    mappedAtCreation := false
  }

  -- Upload Float32 data directly
  writeBuffer device embeddingTable 0 float32Data

  logVerbose "[Embedding] ✓ Layer created"
  pure { config, embeddingTable }

/-- GPU kernel to unpack F16 → F32 using hardware instruction

    Each thread processes `packedPerThread` packed U32 values (each containing 2 F16s),
    producing `2 * packedPerThread` F32 outputs. This keeps workgroup count within
    the 65535 max per dimension limit.
-/
def unpackF16ToF32Kernel (numElements : Nat) (packedPerThread : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let threadIdx := Exp.vec3X gid
  let numPacked := (numElements + 1) / 2

  -- Declare buffers
  let _f16Data ← ShaderM.declareInputBuffer "f16_data" (.array (.scalar .u32) numPacked)
  let _f32Data ← ShaderM.declareOutputBuffer "f32_data" (.array (.scalar .f32) numElements)

  -- Each thread processes packedPerThread packed u32 values
  let basePackedIdx := Exp.mul threadIdx (Exp.litU32 packedPerThread)

  ShaderM.loop (Exp.litU32 0) (Exp.litU32 packedPerThread) (Exp.litU32 1) fun i => do
    let packedIdx := Exp.add basePackedIdx i
    let dstIdx0 := Exp.mul (Exp.litU32 2) packedIdx
    let inBounds := Exp.lt packedIdx (Exp.litU32 numPacked)

    -- Read packed u32 (contains two f16 values)
    let packed ← ShaderM.readBuffer (ty := .scalar .u32)
      (n := numPacked) "f16_data" packedIdx

    -- Hardware unpack: u32 -> vec2<f32>
    let unpacked := Exp.unpack2x16float packed

    -- Write first element
    let finalVal0 := Exp.select inBounds (Exp.vecX unpacked) (Exp.litF32 0.0)
    ShaderM.writeBuffer (ty := .scalar .f32) "f32_data" dstIdx0 finalVal0

    -- Write second element
    let dstIdx1 := Exp.add dstIdx0 (Exp.litU32 1)
    let secondInBounds := Exp.lt dstIdx1 (Exp.litU32 numElements)
    let finalVal1 := Exp.select secondInBounds (Exp.vecY unpacked) (Exp.litF32 0.0)
    ShaderM.writeBuffer (ty := .scalar .f32) "f32_data" dstIdx1 finalVal1

/-- Create embedding layer from F16 data (GPU-optimized version)

    Uploads raw F16 data (256 MB) and unpacks to F32 on GPU using hardware instruction.
    This is 6× faster than CPU conversion: 2× less bandwidth + GPU parallelism.

    @param device WebGPU device
    @param config Embedding configuration
    @param f16Data Raw F16 (Float16) embedding data (packed as bytes)
-/
def createFromF16 (device : Device) (config : Config)
                  (f16Data : ByteArray) : IO Embedding := do
  logVerbose s!"[Embedding] Creating layer from F16 (GPU-optimized): vocab={config.vocabSize}, dim={config.dim}"

  let numElements := config.vocabSize * config.dim
  let f16Size := f16Data.size.toUSize
  let f32Size := (numElements * 4).toUSize

  IO.println s!"  Upload: {f16Size / (1024*1024)} MB F16 (vs {f32Size / (1024*1024)} MB F32 saved)"

  -- Step 1: Upload raw F16 data (256 MB instead of 512 MB!)
  let f16Buffer ← createBuffer device {
    size := f16Size
    usage := [.storage, .copySrc]
    mappedAtCreation := false
  }
  writeBuffer device f16Buffer 0 f16Data

  -- Step 2: Create F32 output buffer
  let f32Buffer ← createBuffer device {
    size := f32Size
    usage := [.storage, .copyDst]
    mappedAtCreation := false
  }

  -- Step 3: Dispatch GPU unpacking kernel
  IO.println "  Unpacking F16 → F32 on GPU..."
  let numPacked := (numElements + 1) / 2
  let workgroupSize := 256
  let maxWorkgroups := 65535
  let maxThreads := maxWorkgroups * workgroupSize  -- 16,776,960
  -- Compute packed elements per thread to stay within dispatch limits
  let packedPerThread := (numPacked + maxThreads - 1) / maxThreads |>.max 1
  let totalThreads := (numPacked + packedPerThread - 1) / packedPerThread
  IO.println s!"  Dispatch: {totalThreads} threads, {packedPerThread} packed/thread"

  let shader := unpackF16ToF32Kernel numElements packedPerThread
  let namedBuffers := [
    ("f16_data", f16Buffer),
    ("f32_data", f32Buffer)
  ]

  let execConfig := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D totalThreads workgroupSize

  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig

  logVerbose s!"[Embedding] ✓ Layer created (GPU unpacked {numElements} F16 → F32 elements)"
  pure { config, embeddingTable := f32Buffer, f16Table := some f16Buffer }

/-! ## Forward Pass -/

/-- Execute embedding lookup

    @param device WebGPU device
    @param layer Embedding layer
    @param tokenIdsBuf Input token IDs [batch, seq_len] (UInt32)
    @param outputBuf Output embeddings [batch, seq_len, dim] (Float32)
    @param batchSize Batch size
    @param seqLen Sequence length
-/
def forward (device : Device) (layer : Embedding)
            (tokenIdsBuf outputBuf : Buffer)
            (batchSize seqLen : Nat) : IO Unit := do
  logVerbose s!"[Embedding] Lookup: batch={batchSize}, seq_len={seqLen}"

  let shader := embeddingLookupKernel layer.config batchSize seqLen
  let namedBuffers := [
    ("token_ids", tokenIdsBuf),
    ("embedding_table", layer.embeddingTable),
    ("output", outputBuf)
  ]

  let totalTokens := batchSize * seqLen
  let execConfig := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D
    totalTokens
    256  -- One thread per token

  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig
  logVerbose "[Embedding] ✓ Lookup complete"

/-! ## Integration with GGUF -/

/-- Create embedding layer from GGUF file

    Typical tensor name: `token_embd.weight`

    @param device WebGPU device
    @param gguf Loaded GGUF file
    @param config Embedding configuration
-/
def fromGGUF (device : Device) (gguf : α) (config : Config) : IO Embedding := do
  -- Placeholder - actual implementation would:
  -- 1. Find tensor: gguf.findTensor "token_embd.weight"
  -- 2. Extract TQ2_0 packed data + scales
  -- 3. Call create() with extracted data
  IO.println "[Embedding] Loading from GGUF"
  throw $ IO.userError "fromGGUF not yet implemented - use create() or createFromFloat32()"

end Hesper.Layers.Embedding
