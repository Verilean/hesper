import Hesper.Backend
import Hesper.Backend.WebGPU
import Hesper.Backend.CUDA
import Hesper.Layers.Linear
import Hesper.Layers.RMSNorm
import Hesper.Layers.RoPE
import Hesper.Layers.Embedding
import Hesper.Layers.Softmax
import Hesper.Layers.MoE
import Hesper.Layers.PerLayerEmbedding
import Hesper.Layers.Attention
import Hesper.Quantization.Q4_K_M
import Hesper.GGUF.Parser
import Hesper.GGUF.Loader
import Hesper.GGUF.Reader
import Hesper.Basic
import Hesper.Logging
import Hesper.Models.Gemma4.Config
import Hesper.Models.Gemma4.Types

/-!
# Gemma 4 GGUF loader

Everything needed to parse a Gemma 4 GGUF file and upload tensors to
the GPU: CPU-side Q6_K dequant helper, GGUF metadata parser,
`Config.fromGGUF`, and `Gemma4Model.fromGGUF{,Data}`.

Kept separate from the forward-pass code so iterating on inference
does not recompile the loader, and vice versa.
-/

namespace Hesper.Models.Gemma4

open Hesper.Layers
open Hesper.Logging (logVerbose)

/-! ## CPU Q6_K Row Dequant -/

/-- Dequantize a single Q6_K row to F32 ByteArray on CPU.
    Used to extract a per-token slice of large Q6_K tables that don't fit in
    a single WebGPU storage buffer binding (e.g. per_layer_token_embd).

    Each block is 256 elements, 210 bytes:
    - bytes [0..128):  ql (low 4 bits, 2 vals/byte)
    - bytes [128..192): qh (high 2 bits, 4 vals/byte)
    - bytes [192..208): scales[16] (int8)
    - bytes [208..210): d (FP16)

    @param data Source ByteArray (e.g. full per_layer_token_embd)
    @param rowOffset Byte offset of the row start within data
    @param numElements Total elements per row (must be multiple of 256)
    @return Float array of length numElements
-/
def dequantQ6KRowCPU (data : ByteArray) (rowOffset numElements : Nat) : Array Float := Id.run do
  let numBlocks := numElements / 256
  let blockBytes := 210
  let mut out : Array Float := (List.replicate numElements (0.0 : Float)).toArray
  for bi in [0:numBlocks] do
    let blockBase := rowOffset + bi * blockBytes
    let outBase := bi * 256
    -- Read d (FP16) at offset 208
    let dLo := (data.get! (blockBase + 208)).toUInt32
    let dHi := (data.get! (blockBase + 209)).toUInt32
    let dBits := dLo ||| (dHi <<< 8)
    -- FP16 → F32 (with subnormal support)
    let sign := (dBits >>> 15) &&& 1
    let exp5 := (dBits >>> 10) &&& 0x1F
    let mant := dBits &&& 0x3FF
    let signF : Float := if sign == 1 then -1.0 else 1.0
    let d : Float :=
      if exp5 == 0 then
        -- Subnormal: (mant / 1024) * 2^(-14)
        signF * (mant.toNat.toFloat / 1024.0) * 6.103515625e-5
      else
        -- Normal: (1 + mant/1024) * 2^(exp - 15)
        signF * (1.0 + mant.toNat.toFloat / 1024.0) * (2.0 ^ (exp5.toNat.toFloat - 15.0))
    -- Process 2 chunks of 128 elements (n = 0, 128 in llama.cpp)
    for chunk in [0:2] do
      let qlBase := blockBase + chunk * 64       -- ql offset: chunk*64 within block
      let qhBase := blockBase + 128 + chunk * 32 -- qh offset: 128 + chunk*32
      let scBase := blockBase + 192 + chunk * 8  -- scales offset: 192 + chunk*8
      let chunkOutBase := outBase + chunk * 128
      for l in [0:32] do
        let isIdx := l / 16  -- 0 or 1, picks scale within sub-block
        let qlByte0 := (data.get! (qlBase + l)).toNat
        let qlByte1 := (data.get! (qlBase + 32 + l)).toNat
        let qhByte := (data.get! (qhBase + l)).toNat
        -- Sign-extend int8 → Float manually
        let sc0Raw := (data.get! (scBase + isIdx)).toNat
        let sc2Raw := (data.get! (scBase + isIdx + 2)).toNat
        let sc4Raw := (data.get! (scBase + isIdx + 4)).toNat
        let sc6Raw := (data.get! (scBase + isIdx + 6)).toNat
        let sc0F : Float := if sc0Raw >= 128 then (sc0Raw.toFloat - 256.0) else sc0Raw.toFloat
        let sc2F : Float := if sc2Raw >= 128 then (sc2Raw.toFloat - 256.0) else sc2Raw.toFloat
        let sc4F : Float := if sc4Raw >= 128 then (sc4Raw.toFloat - 256.0) else sc4Raw.toFloat
        let sc6F : Float := if sc6Raw >= 128 then (sc6Raw.toFloat - 256.0) else sc6Raw.toFloat
        -- q1..q4 are 6-bit unsigned (0..63), then offset by -32 to get [-32..31]
        -- Compute as Nat, then convert to Float, then subtract 32.0
        let q1Raw := (qlByte0 &&& 0xF) ||| (((qhByte >>> 0) &&& 3) <<< 4)
        let q2Raw := (qlByte1 &&& 0xF) ||| (((qhByte >>> 2) &&& 3) <<< 4)
        let q3Raw := (qlByte0 >>> 4) ||| (((qhByte >>> 4) &&& 3) <<< 4)
        let q4Raw := (qlByte1 >>> 4) ||| (((qhByte >>> 6) &&& 3) <<< 4)
        let q1F : Float := q1Raw.toFloat - 32.0
        let q2F : Float := q2Raw.toFloat - 32.0
        let q3F : Float := q3Raw.toFloat - 32.0
        let q4F : Float := q4Raw.toFloat - 32.0
        out := out.set! (chunkOutBase + l)      (d * sc0F * q1F)
        out := out.set! (chunkOutBase + l + 32) (d * sc2F * q2F)
        out := out.set! (chunkOutBase + l + 64) (d * sc4F * q3F)
        out := out.set! (chunkOutBase + l + 96) (d * sc6F * q4F)
  return out

/-- Convert Float Array to F32 ByteArray (little-endian) -/
def floatArrayToBytes (arr : Array Float) : IO ByteArray := do
  let mut bytes := ByteArray.empty
  for f in arr do
    let fb ← Hesper.Basic.floatToBytes f
    bytes := bytes ++ fb
  return bytes

/-! ## GGUF Metadata Parsing -/

/-- Helper: read a u32 value from GGUF metadata raw bytes -/
private def readMetadataU32 (mv : Hesper.GGUF.MetadataValue) : Option Nat :=
  if mv.valueType == .MUInt32 && mv.data.size >= 4 then
    let b0 := mv.data.get! 0 |>.toNat
    let b1 := mv.data.get! 1 |>.toNat
    let b2 := mv.data.get! 2 |>.toNat
    let b3 := mv.data.get! 3 |>.toNat
    some (b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24))
  else none

/-- Helper: read a bool array from GGUF metadata raw bytes.
    The Hesper parser strips the type+length header, so mv.data is just the raw element bytes.
    For bool arrays, each byte is one bool. -/
private def readMetadataBoolArray (mv : Hesper.GGUF.MetadataValue) : Option (Array Bool) :=
  if mv.valueType == .MArray then
    some (Id.run do
      let mut arr : Array Bool := #[]
      for i in [0:mv.data.size] do
        arr := arr.push ((mv.data.get! i) != 0)
      return arr)
  else none

/-- Helper: read a f32 value from GGUF metadata raw bytes -/
private def readMetadataF32 (mv : Hesper.GGUF.MetadataValue) : Option Float :=
  if mv.valueType == .MFloat32 && mv.data.size >= 4 then
    let b0 := mv.data.get! 0 |>.toUInt32
    let b1 := mv.data.get! 1 |>.toUInt32
    let b2 := mv.data.get! 2 |>.toUInt32
    let b3 := mv.data.get! 3 |>.toUInt32
    let bits := b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24)
    some (Hesper.Basic.float32BitsToFloat64 bits)
  else none

/-- Parse Gemma 4 config from GGUF metadata -/
def Config.fromGGUF (gguf : Hesper.GGUF.GGUFFile) : Except String Config := do
  -- Helper to find metadata value
  let findMeta (key : String) : Option Hesper.GGUF.MetadataValue :=
    gguf.metadata.find? (·.1 == key) |>.map (·.2)

  let findU32 (key : String) : Except String Nat :=
    match findMeta key with
    | some mv => match readMetadataU32 mv with
      | some v => .ok v
      | none => .error s!"Metadata key '{key}' is not uint32"
    | none => .error s!"Metadata key '{key}' not found"

  let findU32Either (key1 key2 : String) : Except String Nat :=
    match findU32 key1 with
    | .ok v => .ok v
    | .error _ => findU32 key2

  let findF32Default (key : String) (default : Float) : Float :=
    match findMeta key with
    | some mv => (readMetadataF32 mv).getD default
    | none => default

  let findF32DefaultEither (key1 key2 : String) (default : Float) : Float :=
    match findMeta key1 with
    | some mv => (readMetadataF32 mv).getD default
    | none => findF32Default key2 default

  -- Support both "gemma4." and "llama." prefixes
  let vocabSize := (findU32 "general.vocab_size").toOption.getD 262144
  let hiddenSize ← findU32Either "gemma4.embedding_length" "llama.embedding_length"
  let intermediateSize ← findU32Either "gemma4.feed_forward_length" "llama.feed_forward_length"
  let numLayers ← findU32Either "gemma4.block_count" "llama.block_count"
  let numHeads ← findU32Either "gemma4.attention.head_count" "llama.attention.head_count"

  -- Layer types: parse sliding_window_pattern bool array
  -- True = SWA, False = full attention (per gemma4-iswa.cpp convention)
  let layerTypes : Array LayerType :=
    match findMeta "gemma4.attention.sliding_window_pattern" <|>
          findMeta "llama.attention.sliding_window_pattern" with
    | some mv =>
      match readMetadataBoolArray mv with
      | some bools => bools.map (fun b => if b then LayerType.swa else LayerType.full)
      | none => (List.replicate numLayers LayerType.full).toArray
    | none => (List.replicate numLayers LayerType.full).toArray

  let rmsNormEps := findF32DefaultEither "gemma4.attention.layer_norm_rms_epsilon" "llama.attention.layer_norm_rms_epsilon" 1e-6
  let ropeTheta := findF32DefaultEither "gemma4.rope.freq_base" "llama.rope.freq_base" 1000000.0
  let ropeThetaSWA := findF32DefaultEither "gemma4.rope.freq_base_swa" "llama.rope.freq_base_swa" 10000.0

  return {
    vocabSize
    hiddenSize
    intermediateSize
    numHiddenLayers := numLayers
    numAttentionHeads := numHeads
    numKeyValueHeadsFull := (findU32Either "gemma4.attention.head_count_kv" "llama.attention.head_count_kv").toOption.getD 8
    numKeyValueHeadsSWA := (findU32Either "gemma4.attention.head_count_kv" "llama.attention.head_count_kv").toOption.getD 8
    headDimFull := (findU32Either "gemma4.attention.key_length" "llama.attention.key_length").toOption.getD 128
    headDimSWA := (findU32Either "gemma4.attention.key_length_swa" "llama.attention.key_length_swa").toOption.getD 128
    slidingWindowSize := (findU32Either "gemma4.attention.sliding_window" "llama.attention.sliding_window").toOption.getD 512
    rmsNormEps
    ropeTheta
    ropeThetaSWA
    partialRotaryFactorSWA := 0.5  -- TODO: read from metadata
    layerTypes
    logitSoftcapScale := findF32DefaultEither "gemma4.final_logit_softcapping" "llama.logit_softcapping" 30.0
    -- Cap maxSeqLen at 4096 to keep KV cache buffers under WebGPU's 256 MiB binding limit.
    -- Full layers: 8 KV heads × maxSeqLen × 512 head_dim × 4 bytes
    --   At maxSeqLen=4096: 8 × 4096 × 512 × 4 = 64 MiB per layer ✓
    -- Real context_length is read but capped here.
    maxSeqLen := min 4096 ((findU32Either "gemma4.context_length" "llama.context_length").toOption.getD 4096)
    numExperts := (findU32Either "gemma4.expert_count" "llama.expert_count").toOption.getD 0
    numExpertsUsed := (findU32Either "gemma4.expert_used_count" "llama.expert_used_count").toOption.getD 0
    expertFFSize := (findU32Either "gemma4.expert_feed_forward_length" "llama.expert_feed_forward_length").toOption.getD 0
    embdPerLayer := (findU32Either "gemma4.embedding_length_per_layer_input" "llama.embedding_length_per_layer_input").toOption.getD 0
    numKVSharedLayers := (findU32Either "gemma4.attention.shared_kv_layers" "llama.attention.shared_kv_layers").toOption.getD 0
  }

/-! ## Helper: Create GPU Buffer from ByteArray -/

private def uploadBuffer [GPUBackend β] (ctx : β) (data : ByteArray) : IO (GPUBackend.Buf β) := do
  let bufSize := if data.size == 0 then 4 else data.size
  let buf ← GPUBackend.allocBuffer ctx bufSize.toUSize
  if data.size > 0 then
    GPUBackend.writeBuffer ctx buf data
  return buf

/-- Upload a tensor body to GPU by name.  Routes through mmap
    direct-copy when the GGUFFile was loaded with `loadGGUFMmap`
    (i.e. `.mmap.isSome`), avoiding the 5GB ByteArray allocation +
    extract that the classic `uploadBuffer` requires.  Falls back to
    ByteArray path otherwise. -/
private def uploadTensor [GPUBackend β] (ctx : β) (gguf : Hesper.GGUF.GGUFFile)
    (name : String) : IO (GPUBackend.Buf β) := do
  -- Use Loader's findTensor which gives us `(offset, size)` in the
  -- data section (Nat, not UInt64).
  let info ← match Hesper.GGUF.Loader.findTensor gguf name with
    | .ok i => pure i
    | .error e => throw (IO.userError e)
  let bytes := info.size
  let bufSize : USize := if bytes == 0 then 4 else bytes.toUSize
  let buf ← GPUBackend.allocBuffer ctx bufSize
  if bytes > 0 then
    match gguf.mmap with
    | some mmap =>
      match ← GPUBackend.rawDevicePtr ctx buf with
      | some devPtr =>
        let absOffset : USize := gguf.dataSectionOffset + info.offset.toUSize
        -- Async when HESPER_MMAP_ASYNC=1 + cudaDefaultStream set.  Mmap
        -- lifetime is tied to `gguf` held by the caller, so in-flight
        -- async copies don't race with unmap.
        let useAsync := (← IO.getEnv "HESPER_MMAP_ASYNC").isSome
        let stream : USize ← if useAsync then
          match ← Hesper.cudaDefaultStream.get with
          | some s => pure s
          | none   => pure (0 : USize)
        else pure (0 : USize)
        Hesper.CUDA.cuMemcpyHtoDFromMmap devPtr mmap absOffset bytes.toUSize stream
      | none =>
        let sliceBytes ← Hesper.CUDA.mmapSliceToBytesPersistent mmap
          (gguf.dataSectionOffset + info.offset.toUSize) bytes.toUSize
        GPUBackend.writeBuffer ctx buf sliceBytes
    | none =>
      match Hesper.GGUF.Loader.getTensorData gguf name with
      | .ok (_, data) => GPUBackend.writeBuffer ctx buf data
      | .error e => throw (IO.userError e)
  return buf

/-! ## GGUF Model Loading -/

/-- Load a single quantized linear layer from GGUF tensor.
    Detects quant format (Q4_K vs Q6_K) and selects the appropriate fused kernel. -/
private def loadLinear [GPUBackend β] (ctx : β) (gguf : Hesper.GGUF.GGUFFile)
    (name : String) (inDim outDim : Nat) : IO (Linear.LinearLayer (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)) := do
  -- Detect quant format from tensor type
  let tensorInfo ← match Hesper.GGUF.Loader.findTensor gguf name with
    | .ok ti => pure ti
    | .error e => throw $ IO.userError e
  let quantFormat : Linear.QuantFormat := match tensorInfo.ggmlType with
    | .Q6_K => .Q6_K
    | _ => .Q4_K
  let weightBuf ← uploadTensor ctx gguf name
  let prepared ← GPUBackend.newCacheRef (β := β)
  let splitKBuf ← IO.mkRef none
  let splitKPartialPrepared ← GPUBackend.newCacheRef (β := β)
  let splitKReducePrepared ← GPUBackend.newCacheRef (β := β)
  let dp4aQ8Buf ← IO.mkRef none
  let dp4aQuantizePrepared ← GPUBackend.newCacheRef (β := β)
  let dp4aMatmulPrepared ← GPUBackend.newCacheRef (β := β)
  let dp4aBatchQuantizePrepared ← GPUBackend.newCacheRef (β := β)
  let dp4aBatchMatmulPrepared ← GPUBackend.newCacheRef (β := β)
  return {
    config := { inDim, outDim }
    weightBuf
    quantFormat
    prepared
    splitKBuf
    splitKPartialPrepared
    splitKReducePrepared
    dp4aQ8Buf
    dp4aQuantizePrepared
    dp4aMatmulPrepared
    dp4aBatchQuantizePrepared
    dp4aBatchMatmulPrepared
  }

/-- Load Gemma 4 model from a pre-parsed GGUF (used by both the
    ByteArray path and the mmap path).  The rest of the loading
    logic runs identically; the mmap vs ByteArray split happens
    inside `uploadTensor`. -/
def Gemma4Model.fromGGUFDataWithGguf [GPUBackend β] (ctx : β)
    (gguf : Hesper.GGUF.GGUFFile)
    (configOverride : Option Config := none) : IO (Gemma4Model (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)) := do
  IO.println s!"  ✓ GGUF parsed: {gguf.tensors.size} tensors, dataBlob size: {gguf.dataBlob.size} bytes (mmap: {gguf.mmap.isSome})"

  -- Step 2: Extract configuration
  let cfg ← match configOverride with
    | some c => pure c
    | none => match Config.fromGGUF gguf with
      | .ok c => pure c
      | .error e => throw $ IO.userError s!"Config parse error: {e}"

  IO.println s!"  Model: {cfg.numHiddenLayers} layers, {cfg.hiddenSize} dim, {cfg.numAttentionHeads} heads"
  IO.println s!"  Vocab: {cfg.vocabSize}, FFN: {cfg.intermediateSize}, Experts: {cfg.numExperts}"

  -- Step 3: Load embedding
  IO.println "[Gemma4] Loading embedding..."
  let embConfig : Embedding.Config := {
    vocabSize := cfg.vocabSize
    dim := cfg.hiddenSize
  }
  -- Gemma 4 embeddings: Q6_K, Q4_K, or F16
  let embTensor ← match Hesper.GGUF.Loader.findTensor gguf "token_embd.weight" with
    | .ok ti => pure ti
    | .error e => throw $ IO.userError e
  let (embedding, embdFormat) ← match embTensor.ggmlType with
    | .F16 =>
      IO.println "  Using F16 embeddings"
      let embData ← Hesper.GGUF.Loader.extractF16Tensor gguf "token_embd.weight"
      let e ← Embedding.createFromF16 ctx embConfig embData
      pure (e, EmbdFormat.F16)
    | .Q6_K =>
      IO.println "  Using Q6_K embeddings (GPU on-the-fly dequant)"
      let buf ← uploadTensor ctx gguf "token_embd.weight"
      pure ({ config := embConfig, embeddingTable := buf, f16Table := none : Embedding.Embedding (GPUBackend.Buf β) }, EmbdFormat.Q6_K)
    | .Q4_K =>
      IO.println "  Using Q4_K embeddings (GPU on-the-fly dequant)"
      let buf ← uploadTensor ctx gguf "token_embd.weight"
      pure ({ config := embConfig, embeddingTable := buf, f16Table := none : Embedding.Embedding (GPUBackend.Buf β) }, EmbdFormat.Q4_K)
    | other =>
      IO.println s!"  Embedding type: {other} — loading as raw bytes (F32 fallback)"
      let buf ← uploadTensor ctx gguf "token_embd.weight"
      pure ({ config := embConfig, embeddingTable := buf, f16Table := none : Embedding.Embedding (GPUBackend.Buf β) }, EmbdFormat.F32)

  -- Step 4: Load transformer blocks
  IO.println s!"[Gemma4] Loading {cfg.numHiddenLayers} transformer blocks..."
  let mut blocks : Array (Gemma4Block (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)) := #[]

  for layerIdx in [0:cfg.numHiddenLayers] do
    if layerIdx % 10 == 0 then
      IO.println s!"  Loading layer {layerIdx}/{cfg.numHiddenLayers}..."
    let li := layerIdx

    let layerType := if li < cfg.layerTypes.size then cfg.layerTypes[li]! else .full
    let headDim := cfg.headDim li
    let numKVHeads := cfg.numKVHeads li

    -- Load norms (Float32)
    let attnNormData ← Hesper.GGUF.Loader.extractFloat32Tensor gguf s!"blk.{li}.attn_norm.weight"
    let postAttnNormData ← Hesper.GGUF.Loader.extractFloat32Tensor gguf s!"blk.{li}.post_attention_norm.weight"
    let ffnNormData ← Hesper.GGUF.Loader.extractFloat32Tensor gguf s!"blk.{li}.ffn_norm.weight"
    let postFFNNormData ← Hesper.GGUF.Loader.extractFloat32Tensor gguf s!"blk.{li}.post_ffw_norm.weight"

    let normConfig : RMSNorm.Config := { dim := cfg.hiddenSize, eps := cfg.rmsNormEps }
    let attnNorm ← RMSNorm.create ctx normConfig attnNormData
    let postAttnNorm ← RMSNorm.create ctx normConfig postAttnNormData
    let ffnNorm ← RMSNorm.create ctx normConfig ffnNormData
    let postFFNNorm ← RMSNorm.create ctx normConfig postFFNNormData

    -- Load attention projections (Q4_K)
    let qDim := cfg.numAttentionHeads * headDim
    let kvDim := numKVHeads * headDim
    let wQ ← loadLinear ctx gguf s!"blk.{li}.attn_q.weight" cfg.hiddenSize qDim
    let wK ← loadLinear ctx gguf s!"blk.{li}.attn_k.weight" cfg.hiddenSize kvDim
    let wV ← loadLinear ctx gguf s!"blk.{li}.attn_v.weight" cfg.hiddenSize kvDim
    let wO ← loadLinear ctx gguf s!"blk.{li}.attn_output.weight" qDim cfg.hiddenSize

    -- Load Q/K norm weights (Float32, per-head dimension)
    let qNormData ← Hesper.GGUF.Loader.extractFloat32Tensor gguf s!"blk.{li}.attn_q_norm.weight"
    let kNormData ← Hesper.GGUF.Loader.extractFloat32Tensor gguf s!"blk.{li}.attn_k_norm.weight"
    let qNormBuf ← uploadBuffer ctx qNormData
    let kNormBuf ← uploadBuffer ctx kNormData

    let fusedNormQPrepared ← GPUBackend.newCacheRef (β := β)
    let fusedNormKPrepared ← GPUBackend.newCacheRef (β := β)
    let fusedNormVPrepared ← GPUBackend.newCacheRef (β := β)
    let attention : Gemma4Attention (GPUBackend.Buf β) (GPUBackend.CachedDispatch β) := {
      wQ, wK, wV, wO
      qNormWeight := qNormBuf
      kNormWeight := kNormBuf
      fusedNormQPrepared, fusedNormKPrepared, fusedNormVPrepared
    }

    -- Load FFN projections (Q4_K)
    let ffnGate ← loadLinear ctx gguf s!"blk.{li}.ffn_gate.weight" cfg.hiddenSize cfg.intermediateSize
    let ffnUp ← loadLinear ctx gguf s!"blk.{li}.ffn_up.weight" cfg.hiddenSize cfg.intermediateSize
    let ffnDown ← loadLinear ctx gguf s!"blk.{li}.ffn_down.weight" cfg.intermediateSize cfg.hiddenSize

    let fusedGateUpPrepared ← GPUBackend.newCacheRef (β := β)
    let fusedNormGatePrepared ← GPUBackend.newCacheRef (β := β)
    let fusedNormUpPrepared ← GPUBackend.newCacheRef (β := β)
    let ffn : Gemma4FFN (GPUBackend.Buf β) (GPUBackend.CachedDispatch β) :=
      { gate := ffnGate, up := ffnUp, down := ffnDown, fusedGateUpPrepared,
        fusedNormGatePrepared, fusedNormUpPrepared }

    -- Load optional RoPE frequency factors
    -- In the E4B model, rope_freqs is a global tensor, not per-layer
    let ropeFreqFactors ← if cfg.isFullAttention li then
      match Hesper.GGUF.Loader.findTensor gguf "rope_freqs.weight" with
      | .ok _ => pure (some (← uploadTensor ctx gguf "rope_freqs.weight"))
      | .error _ =>
        match Hesper.GGUF.Loader.findTensor gguf s!"blk.{li}.rope_freqs.weight" with
        | .ok _ => pure (some (← uploadTensor ctx gguf s!"blk.{li}.rope_freqs.weight"))
        | .error _ => pure none
    else pure none

    -- Load optional layer output scale
    let outScale ← match Hesper.GGUF.Loader.findTensor gguf s!"blk.{li}.layer_output_scale.weight" with
      | .ok _ => pure (some (← uploadTensor ctx gguf s!"blk.{li}.layer_output_scale.weight"))
      | .error _ => pure none

    -- Load MoE weights (if present for this layer)
    let isMoE := match Hesper.GGUF.Loader.findTensor gguf s!"blk.{li}.ffn_gate_inp.weight" with
      | .ok _ => true | .error _ => false
    let (moeRouterWeight, moeRouterScale, moeGateUpExps, moeDownExps, moePreNorm2, moePostNorm1, moePostNorm2) ←
      if isMoE then do
        IO.println s!"    Layer {li}: MoE layer"
        let routerW ← match Hesper.GGUF.Loader.findTensor gguf s!"blk.{li}.ffn_gate_inp.weight" with
          | .ok _ => pure (some (← uploadTensor ctx gguf s!"blk.{li}.ffn_gate_inp.weight"))
          | .error _ => pure none
        let routerS ← match Hesper.GGUF.Loader.findTensor gguf s!"blk.{li}.ffn_gate_inp.scale" with
          | .ok _ => pure (some (← uploadTensor ctx gguf s!"blk.{li}.ffn_gate_inp.scale"))
          | .error _ => pure none
        let gateUpE ← match Hesper.GGUF.Loader.findTensor gguf s!"blk.{li}.ffn_gate_up_exps.weight" with
          | .ok _ => pure (some (← uploadTensor ctx gguf s!"blk.{li}.ffn_gate_up_exps.weight"))
          | .error _ => pure none
        let downE ← match Hesper.GGUF.Loader.findTensor gguf s!"blk.{li}.ffn_down_exps.weight" with
          | .ok _ => pure (some (← uploadTensor ctx gguf s!"blk.{li}.ffn_down_exps.weight"))
          | .error _ => pure none
        let preN2 ← match Hesper.GGUF.Loader.findTensor gguf s!"blk.{li}.ffn_pre_norm_2.weight" with
          | .ok _ =>
            let d ← Hesper.GGUF.Loader.extractFloat32Tensor gguf s!"blk.{li}.ffn_pre_norm_2.weight"
            pure (some (← RMSNorm.create ctx normConfig d))
          | .error _ => pure none
        let postN1 ← match Hesper.GGUF.Loader.findTensor gguf s!"blk.{li}.ffn_post_norm_1.weight" with
          | .ok _ =>
            let d ← Hesper.GGUF.Loader.extractFloat32Tensor gguf s!"blk.{li}.ffn_post_norm_1.weight"
            pure (some (← RMSNorm.create ctx normConfig d))
          | .error _ => pure none
        let postN2 ← match Hesper.GGUF.Loader.findTensor gguf s!"blk.{li}.ffn_post_norm_2.weight" with
          | .ok _ =>
            let d ← Hesper.GGUF.Loader.extractFloat32Tensor gguf s!"blk.{li}.ffn_post_norm_2.weight"
            pure (some (← RMSNorm.create ctx normConfig d))
          | .error _ => pure none
        pure (routerW, routerS, gateUpE, downE, preN2, postN1, postN2)
      else
        pure (none, none, none, none, none, none, none)

    blocks := blocks.push {
      layerIdx := li
      layerType
      attnNorm, postAttnNorm, ffnNorm, postFFNNorm
      attention, ffn
      isMoE
      moeRouterWeight, moeRouterScale, moeGateUpExps, moeDownExps
      moePreNorm2, moePostNorm1, moePostNorm2
      ropeFreqFactors, outScale
    }

  -- Step 5: Final norm
  IO.println "[Gemma4] Loading final norm and LM head..."
  let finalNormData ← Hesper.GGUF.Loader.extractFloat32Tensor gguf "output_norm.weight"
  let finalNormConfig : RMSNorm.Config := { dim := cfg.hiddenSize, eps := cfg.rmsNormEps }
  let finalNorm ← RMSNorm.create ctx finalNormConfig finalNormData

  -- Step 6: LM head (output.weight or weight-tied with embedding)
  let outputWeight ← match Hesper.GGUF.Loader.findTensor gguf "output.weight" with
    | .ok _ =>
      IO.println "  Using separate LM head weights"
      uploadTensor ctx gguf "output.weight"
    | .error _ =>
      IO.println "  Using weight-tied LM head (reusing embedding)"
      pure embedding.embeddingTable

  -- Step 7: Load per-layer embeddings (optional)
  -- Two paths for per_layer_token_embd (the 2.2 GiB Q6_K table):
  --   (a) Default: upload the full table to VRAM. Wastes 2.2 GB VRAM
  --       but no per-token H2D copy.
  --   (b) HESPER_USE_MMAP=1 + this path: keep the table in CPU mmap
  --       and DMA only the active row (~45 KB) per token. Matches
  --       llama.cpp's CPU_Mapped buffer pattern. Saves 2.2 GB VRAM.
  let (perLayerEmbdMmap, perLayerEmbdTableGPU, perLayerEmbdRowBytes,
       perLayerModelProj, perLayerProjNorm) ←
    if cfg.hasPerLayerEmbeddings then do
      IO.println "[Gemma4] Loading per-layer embeddings..."
      let blocksPerRow := (cfg.embdPerLayer * cfg.numHiddenLayers) / 256
      let rowBytes := blocksPerRow * 210  -- Q6_K block size
      let (mmapHandle, tableGpu) ← match Hesper.GGUF.Loader.findTensor gguf "per_layer_token_embd.weight" with
        | .ok info =>
          match gguf.mmap with
          | some mh =>
            -- On-demand path: only allocate one row (~45 KB) in VRAM.
            IO.println s!"  per_layer_token_embd: {info.size} bytes → kept in CPU mmap, GPU row buffer = {rowBytes} bytes"
            let rowBuf ← GPUBackend.allocBuffer ctx rowBytes.toUSize
            let dataSecOff := gguf.dataSectionOffset
            let tensorOff := info.offset.toUSize
            pure (some (mh, dataSecOff, tensorOff), some rowBuf)
          | none =>
            -- Legacy path: upload the whole table to VRAM.
            IO.println s!"  per_layer_token_embd: {info.size} bytes → GPU ({info.size / 1024 / 1024} MB)"
            pure (none, some (← uploadTensor ctx gguf "per_layer_token_embd.weight"))
        | .error _ => pure (none, none)
      let proj ← match Hesper.GGUF.Loader.findTensor gguf "per_layer_model_proj.weight" with
        | .ok _ => pure (some (← uploadTensor ctx gguf "per_layer_model_proj.weight"))
        | .error _ => pure none
      let projNorm ← match Hesper.GGUF.Loader.findTensor gguf "per_layer_proj_norm.weight" with
        | .ok _ =>
          let d ← Hesper.GGUF.Loader.extractFloat32Tensor gguf "per_layer_proj_norm.weight"
          let plNormConfig : RMSNorm.Config := { dim := cfg.embdPerLayer, eps := cfg.rmsNormEps }
          pure (some (← RMSNorm.create ctx plNormConfig d))
        | .error _ => pure none
      pure (mmapHandle, tableGpu, rowBytes, proj, projNorm)
    else
      pure (none, none, 0, none, none)

  -- Per-layer gate/proj/norm per block
  let mut perLayerBlocks : Array (Option (Gemma4PerLayerEmbd (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))) := #[]
  for li in [0:cfg.numHiddenLayers] do
    if cfg.hasPerLayerEmbeddings then
      -- Load Q4_K linear layers for inp_gate and proj
      let inpGate ← loadLinear ctx gguf s!"blk.{li}.inp_gate.weight" cfg.hiddenSize cfg.embdPerLayer
      let proj ← loadLinear ctx gguf s!"blk.{li}.proj.weight" cfg.embdPerLayer cfg.hiddenSize
      let normConfig : RMSNorm.Config := { dim := cfg.hiddenSize, eps := cfg.rmsNormEps }
      let postNorm ← match Hesper.GGUF.Loader.findTensor gguf s!"blk.{li}.post_norm.weight" with
        | .ok _ =>
          let d ← Hesper.GGUF.Loader.extractFloat32Tensor gguf s!"blk.{li}.post_norm.weight"
          RMSNorm.create ctx normConfig d
        | .error _ =>
          -- Fallback: create with all-ones weights
          let dummyData : ByteArray ← do
            let mut bytes := ByteArray.empty
            for _ in [0:cfg.hiddenSize] do
              -- Float 1.0 = 0x3f800000 LE: 00 00 80 3f
              bytes := bytes.push 0
              bytes := bytes.push 0
              bytes := bytes.push 0x80
              bytes := bytes.push 0x3f
            pure bytes
          RMSNorm.create ctx normConfig dummyData
      perLayerBlocks := perLayerBlocks.push (some { inpGate, proj, postNorm : Gemma4PerLayerEmbd (GPUBackend.Buf β) (GPUBackend.CachedDispatch β) })
    else
      perLayerBlocks := perLayerBlocks.push none

  IO.println s!"[Gemma4] ✓ Model loaded: {blocks.size} blocks"

  return {
    config := cfg
    embedding
    embdFormat
    blocks
    finalNorm
    outputWeight
    perLayerEmbdMmap
    perLayerEmbdTableGPU
    perLayerEmbdRowBytes
    perLayerModelProj
    perLayerProjNorm
    perLayerBlocks
  }

/-! ## GGUF File Loading Helper -/

/-- Load GGUF file from disk -/
def loadGGUF (path : String) : IO Hesper.GGUF.GGUFFile := do
  let data ← IO.FS.readBinFile path
  match Hesper.GGUF.Parser.parseGGUF data with
  | .ok gf => pure gf
  | .error e => throw $ IO.userError s!"GGUF parse error: {e}"

/-- Parse GGUF from ByteArray and load the model (backward-compat). -/
def Gemma4Model.fromGGUFData [GPUBackend β] (ctx : β) (ggufData : ByteArray)
    (configOverride : Option Config := none) : IO (Gemma4Model (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)) := do
  IO.println s!"[Gemma4] Parsing GGUF ({ggufData.size} bytes)..."
  let gguf ← match Hesper.GGUF.Parser.parseGGUF ggufData with
    | .ok gf => pure gf
    | .error e => throw $ IO.userError s!"GGUF parse error: {e}"
  Gemma4Model.fromGGUFDataWithGguf ctx gguf configOverride

/-- Load model from GGUF file path.  When HESPER_USE_MMAP=1,
    mmaps the file and streams tensors directly to GPU (no Lean
    ByteArray copy).  Otherwise falls back to IO.FS.readBinFile. -/
def Gemma4Model.fromGGUF [GPUBackend β] (ctx : β) (ggufPath : String)
    (configOverride : Option Config := none) : IO (Gemma4Model (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)) := do
  IO.println s!"[Gemma4] Loading model from {ggufPath}..."
  let useMmap := (← IO.getEnv "HESPER_USE_MMAP").isSome
  if useMmap then
    IO.println s!"  path: mmap (HESPER_USE_MMAP=1)"
    let gguf ← Hesper.GGUF.loadGGUFMmap ggufPath
    Gemma4Model.fromGGUFDataWithGguf ctx gguf configOverride
  else
    let ggufData ← IO.FS.readBinFile ggufPath
    Gemma4Model.fromGGUFData ctx ggufData configOverride

end Hesper.Models.Gemma4
