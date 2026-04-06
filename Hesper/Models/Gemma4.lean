import Hesper.WGSL.Monad
import Hesper.WGSL.Execute
import Hesper.WGSL.Exp
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.Layers.Linear
import Hesper.Layers.RMSNorm
import Hesper.Layers.RoPE
import Hesper.Layers.Embedding
import Hesper.Layers.Softmax
import Hesper.Quantization.Q4_K_M
import Hesper.Layers.MoE
import Hesper.Layers.PerLayerEmbedding
import Hesper.GGUF.Parser
import Hesper.GGUF.Loader
import Hesper.Basic
import Hesper.Logging
import Hesper.WGSL.MatMul
import Hesper.WebGPU.BufferOps
import Hesper.Inference.Sampling
import Hesper.WGSL.FlashAttention
import Hesper.Layers.Attention

/-!
# Gemma 4 Model Implementation

Implements the Gemma 4 transformer with:
- ISWA (Interleaved Sliding Window Attention)
- Hybrid MoE + dense FFN (MoE deferred to Phase 3)
- Per-layer embeddings (deferred to Phase 4)
- Q/K normalization
- Logit softcapping
- KV cache sharing (deferred to Phase 4)

Reference: llama.cpp/src/models/gemma4-iswa.cpp

## Architecture

```
embed * sqrt(hiddenSize)
for each layer:
  attnNorm -> Q/K/V projections -> Q-norm, K-norm, V-norm -> RoPE -> attention -> postAttnNorm -> + residual
  ffnNorm -> GeGLU FFN -> postFFNNorm -> + residual
  [per_layer_embedding (Phase 4)]
  [layer_scale]
finalNorm -> lm_head -> logit_softcap
```
-/

namespace Hesper.Models.Gemma4

open Hesper.WGSL
open Hesper.WGSL.Monad
open Hesper.WebGPU
open Hesper.Layers
open Hesper.Logging (logVerbose)

/-! ## Configuration -/

/-- Attention layer type: full context or sliding window -/
inductive LayerType where
  | full   -- Full attention (global context)
  | swa    -- Sliding Window Attention
  deriving Repr, BEq, Inhabited

/-- Gemma 4 model configuration -/
structure Config where
  vocabSize : Nat              -- 262144 for 31B
  hiddenSize : Nat             -- 3840 for 31B
  intermediateSize : Nat       -- GeGLU FFN hidden size
  numHiddenLayers : Nat        -- 62 for 31B
  numAttentionHeads : Nat      -- 32
  numKeyValueHeadsFull : Nat   -- KV heads for full attention layers
  numKeyValueHeadsSWA : Nat    -- KV heads for SWA layers
  headDimFull : Nat            -- 128 (global_head_dim)
  headDimSWA : Nat             -- 128 (head_dim)
  slidingWindowSize : Nat      -- 512
  rmsNormEps : Float           -- 1e-6
  ropeTheta : Float            -- 1000000.0
  partialRotaryFactorSWA : Float -- e.g. 0.5
  layerTypes : Array LayerType -- per-layer: full or SWA
  logitSoftcapScale : Float    -- 30.0
  maxSeqLen : Nat              -- 131072
  -- MoE config
  numExperts : Nat             -- 0 for dense-only models
  numExpertsUsed : Nat         -- top-K routing (e.g., 2)
  expertFFSize : Nat           -- expert intermediate size
  -- Per-layer embeddings
  embdPerLayer : Nat           -- 0 = disabled
  -- KV cache sharing
  numKVSharedLayers : Nat      -- last N layers reuse earlier KV cache
  deriving Repr

/-- Get number of KV heads for a given layer -/
def Config.numKVHeads (c : Config) (layerIdx : Nat) : Nat :=
  if layerIdx < c.layerTypes.size then
    match c.layerTypes[layerIdx]! with
    | .full => c.numKeyValueHeadsFull
    | .swa => c.numKeyValueHeadsSWA
  else c.numKeyValueHeadsSWA

/-- Get head dimension for a given layer -/
def Config.headDim (c : Config) (layerIdx : Nat) : Nat :=
  if layerIdx < c.layerTypes.size then
    match c.layerTypes[layerIdx]! with
    | .full => c.headDimFull
    | .swa => c.headDimSWA
  else c.headDimSWA

/-- Check if a layer uses full attention -/
def Config.isFullAttention (c : Config) (layerIdx : Nat) : Bool :=
  if layerIdx < c.layerTypes.size then
    c.layerTypes[layerIdx]! == .full
  else false

/-- Check if a layer has its own KV cache (not shared) -/
def Config.hasKV (c : Config) (layerIdx : Nat) : Bool :=
  layerIdx < c.numHiddenLayers - c.numKVSharedLayers

/-- Check if per-layer embeddings are enabled -/
def Config.hasPerLayerEmbeddings (c : Config) : Bool :=
  c.embdPerLayer > 0

/-! ## GeGLU FFN Kernel -/

/-- Fused GeGLU FFN kernel: hidden = GELU(x @ W_gate) * (x @ W_up)
    This is the elementwise GELU + multiply step after the gate and up projections.

    @param size Number of elements in the hidden dimension
-/
def geluMulKernel (size : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let _gate ← ShaderM.declareInputBuffer "gate" (.array (.scalar .f32) size)
  let _up ← ShaderM.declareInputBuffer "up" (.array (.scalar .f32) size)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) size)

  ShaderM.if_ (Exp.lt idx (Exp.litU32 size)) (do
    let gateVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := size) "gate" idx
    let upVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := size) "up" idx
    -- GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let sqrt2OverPi := Exp.litF32 0.7978845608028654
    let x3 := Exp.mul (Exp.mul gateVal gateVal) gateVal
    let inner := Exp.mul sqrt2OverPi (Exp.add gateVal (Exp.mul (Exp.litF32 0.044715) x3))
    let gelu := Exp.mul (Exp.mul (Exp.litF32 0.5) gateVal) (Exp.add (Exp.litF32 1.0) (Exp.tanh inner))
    -- output = GELU(gate) * up
    let result := Exp.mul gelu upVal
    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx result
  ) (pure ())

/-! ## Logit Softcapping Kernel -/

/-- Logit softcapping: y = scale * tanh(x / scale)
    @param size Number of elements
    @param scale Softcap scale (30.0 for Gemma 4)
-/
def logitSoftcapKernel (size : Nat) (scale : Float) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) size)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) size)

  ShaderM.if_ (Exp.lt idx (Exp.litU32 size)) (do
    let x ← ShaderM.readBuffer (ty := .scalar .f32) (n := size) "input" idx
    let scaled := Exp.div x (Exp.litF32 scale)
    let result := Exp.mul (Exp.litF32 scale) (Exp.tanh scaled)
    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx result
  ) (pure ())

/-! ## Elementwise Kernels -/

/-- Embedding scale kernel: y = x * sqrt(hiddenSize)
    Applied after token embedding lookup.
-/
def embeddingScaleKernel (size : Nat) (hiddenSize : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) size)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) size)

  ShaderM.if_ (Exp.lt idx (Exp.litU32 size)) (do
    let x ← ShaderM.readBuffer (ty := .scalar .f32) (n := size) "input" idx
    let result := Exp.mul x (Exp.litF32 (Float.sqrt hiddenSize.toFloat))
    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx result
  ) (pure ())

/-- Residual add kernel: y = a + b -/
def residualAddKernel (size : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let _a ← ShaderM.declareInputBuffer "a" (.array (.scalar .f32) size)
  let _b ← ShaderM.declareInputBuffer "b" (.array (.scalar .f32) size)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) size)

  ShaderM.if_ (Exp.lt idx (Exp.litU32 size)) (do
    let aVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := size) "a" idx
    let bVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := size) "b" idx
    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx (Exp.add aVal bVal)
  ) (pure ())

/-! ## RoPE with Frequency Factors -/

/-- RoPE kernel with per-dimension frequency factors.
    Used by Gemma 4 full-attention layers for "proportional" RoPE.
    Frequency factors modulate the base frequency per dimension pair:
    θ[i] = pos * (base^(-2i/d)) / freqFactor[i]
    Setting freqFactor[i] = 1e30 effectively disables rotation for that dimension.

    @param headDim Per-head dimension
    @param numHeads Number of attention heads
    @param ropeBase RoPE frequency base
-/
def ropeWithFreqFactorsKernel (headDim numHeads : Nat) (ropeBase : Float) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let dimPairs := headDim / 2
  let totalElements := numHeads * dimPairs

  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) (numHeads * headDim))
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) (numHeads * headDim))
  let _params ← ShaderM.declareInputBuffer "params" (.array (.scalar .u32) 2)
  let _freqFactors ← ShaderM.declareInputBuffer "freq_factors" (.array (.scalar .f32) dimPairs)

  ShaderM.if_ (Exp.lt idx (Exp.litU32 totalElements)) (do
    let dimPair := Exp.mod idx (Exp.litU32 dimPairs)
    let head := Exp.div idx (Exp.litU32 dimPairs)

    -- Read position from params buffer
    let pos ← ShaderM.readBuffer (ty := .scalar .u32) (n := 2) "params" (Exp.litU32 0)
    let posF32 := Exp.toF32 pos

    -- Read frequency factor for this dimension pair
    let freqFactor ← ShaderM.readBuffer (ty := .scalar .f32) (n := dimPairs) "freq_factors" dimPair

    -- Compute theta with frequency factor
    let dimPairF32 := Exp.toF32 dimPair
    let exponent := Exp.div (Exp.mul (Exp.litF32 2.0) dimPairF32) (Exp.litF32 headDim.toFloat)
    let freqInv := Exp.pow (Exp.litF32 ropeBase) (Exp.neg exponent)
    let theta := Exp.div (Exp.mul posF32 freqInv) freqFactor

    let cosTheta := Exp.cos theta
    let sinTheta := Exp.sin theta

    -- NeoX split-half: pairs are (x[i], x[i + headDim/2])
    let halfDim := headDim / 2
    let headOffset := Exp.mul head (Exp.litU32 headDim)
    let idx0 := Exp.add headOffset dimPair
    let idx1 := Exp.add headOffset (Exp.add dimPair (Exp.litU32 halfDim))

    let x0 ← ShaderM.readBuffer (ty := .scalar .f32) (n := numHeads * headDim) "input" idx0
    let x1 ← ShaderM.readBuffer (ty := .scalar .f32) (n := numHeads * headDim) "input" idx1

    let x0_new := Exp.sub (Exp.mul x0 cosTheta) (Exp.mul x1 sinTheta)
    let x1_new := Exp.add (Exp.mul x0 sinTheta) (Exp.mul x1 cosTheta)

    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx0 x0_new
    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx1 x1_new
  ) (pure ())

/-! ## Per-Head RMSNorm Kernel -/

/-- Per-head RMSNorm: normalize each head independently.
    Used for Q-norm, K-norm, V-norm in Gemma 4.
    Input shape: [numHeads, headDim]
    Each workgroup processes one head.

    @param numHeads Number of heads
    @param headDim Dimension per head
    @param eps RMSNorm epsilon
-/
def perHeadRMSNormKernel (numHeads headDim : Nat) (eps : Float) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let headIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid

  let totalElements := numHeads * headDim

  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) totalElements)
  let _weight ← ShaderM.declareInputBuffer "weight" (.array (.scalar .f32) headDim)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) totalElements)

  -- Shared memory for sum of squares reduction
  let wgSize := if headDim < 256 then headDim else 256
  ShaderM.sharedNamed "shared_sum" (.array (.scalar .f32) wgSize)

  let headBase := Exp.mul headIdx (Exp.litU32 headDim)

  -- Step 1: Compute sum of squares (cooperative)
  ShaderM.varNamed "local_sum" (.scalar .f32) (Exp.litF32 0.0)
  let localSum : Exp (.scalar .f32) := Exp.var "local_sum"

  ShaderM.loop tid (Exp.litU32 headDim) (Exp.litU32 wgSize) fun i => do
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalElements) "input" (Exp.add headBase i)
    ShaderM.assign "local_sum" (Exp.add localSum (Exp.mul val val))

  ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_sum" tid localSum
  ShaderM.barrier

  -- Tree reduction
  let mut stride := wgSize / 2
  while stride > 0 do
    ShaderM.if_ (Exp.lt tid (Exp.litU32 stride)) (do
      let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := wgSize) "shared_sum" tid
      let b ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := wgSize) "shared_sum" (Exp.add tid (Exp.litU32 stride))
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_sum" tid (Exp.add a b)
    ) (pure ())
    ShaderM.barrier
    stride := stride / 2

  -- Step 2: Normalize and write output
  let sumSq ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := wgSize) "shared_sum" (Exp.litU32 0)
  let rms := Exp.inverseSqrt (Exp.add (Exp.div sumSq (Exp.litF32 headDim.toFloat)) (Exp.litF32 eps))

  ShaderM.loop tid (Exp.litU32 headDim) (Exp.litU32 wgSize) fun i => do
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalElements) "input" (Exp.add headBase i)
    let w ← ShaderM.readBuffer (ty := .scalar .f32) (n := headDim) "weight" i
    let normed := Exp.mul (Exp.mul val rms) w
    ShaderM.writeBuffer (ty := .scalar .f32) "output" (Exp.add headBase i) normed

/-! ## Bare RMSNorm Kernel (no learned weights) -/

/-- RMSNorm without learned scale weights.
    Used for V-norm in Gemma 4: Vcur = rms_norm(Vcur, eps)
    Just normalizes by RMS, no γ multiplication.
    One workgroup per vector (for single-token: one workgroup total).

    @param dim Vector dimension
    @param eps Epsilon for numerical stability
-/
def bareRMSNormKernel (dim : Nat) (eps : Float) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let _rowIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid

  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) dim)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) dim)

  ShaderM.sharedNamed "shared_sum" (.array (.scalar .f32) workgroupSize)

  -- Step 1: Compute sum of squares
  ShaderM.varNamed "local_sum" (.scalar .f32) (Exp.litF32 0.0)
  let localSum : Exp (.scalar .f32) := Exp.var "local_sum"

  ShaderM.loop tid (Exp.litU32 dim) (Exp.litU32 workgroupSize) fun i => do
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := dim) "input" i
    ShaderM.assign "local_sum" (Exp.add localSum (Exp.mul val val))

  ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_sum" tid localSum
  ShaderM.barrier

  -- Tree reduction
  let mut stride := workgroupSize / 2
  while stride > 0 do
    ShaderM.if_ (Exp.lt tid (Exp.litU32 stride)) (do
      let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_sum" tid
      let b ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_sum" (Exp.add tid (Exp.litU32 stride))
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_sum" tid (Exp.add a b)
    ) (pure ())
    ShaderM.barrier
    stride := stride / 2

  -- Step 2: Normalize (no weight multiplication)
  let sumSq ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_sum" (Exp.litU32 0)
  let rms := Exp.inverseSqrt (Exp.add (Exp.div sumSq (Exp.litF32 dim.toFloat)) (Exp.litF32 eps))

  ShaderM.loop tid (Exp.litU32 dim) (Exp.litU32 workgroupSize) fun i => do
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := dim) "input" i
    ShaderM.writeBuffer (ty := .scalar .f32) "output" i (Exp.mul val rms)

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

  -- Layer types: default all full attention for now
  -- TODO: parse llama.attention.sliding_window_pattern (bool array in metadata)
  let layerTypes : Array LayerType := (List.replicate numLayers LayerType.full).toArray

  let rmsNormEps := findF32DefaultEither "gemma4.attention.layer_norm_rms_epsilon" "llama.attention.layer_norm_rms_epsilon" 1e-6
  let ropeTheta := findF32DefaultEither "gemma4.rope.freq_base" "llama.rope.freq_base" 1000000.0

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
    partialRotaryFactorSWA := 0.5  -- TODO: read from metadata
    layerTypes
    logitSoftcapScale := findF32DefaultEither "gemma4.final_logit_softcapping" "llama.logit_softcapping" 30.0
    maxSeqLen := (findU32Either "gemma4.context_length" "llama.context_length").toOption.getD 131072
    numExperts := (findU32Either "gemma4.expert_count" "llama.expert_count").toOption.getD 0
    numExpertsUsed := (findU32Either "gemma4.expert_used_count" "llama.expert_used_count").toOption.getD 0
    expertFFSize := (findU32Either "gemma4.expert_feed_forward_length" "llama.expert_feed_forward_length").toOption.getD 0
    embdPerLayer := (findU32Either "gemma4.embedding_length_per_layer_input" "llama.embedding_length_per_layer_input").toOption.getD 0
    numKVSharedLayers := (findU32Either "gemma4.attention.shared_kv_layers" "llama.attention.shared_kv_layers").toOption.getD 0
  }

/-! ## Layer Structures -/

/-- Gemma 4 attention layer (single layer) -/
structure Gemma4Attention where
  wQ : Linear.LinearLayer         -- Q projection [hiddenSize → numHeads * headDim]
  wK : Linear.LinearLayer         -- K projection [hiddenSize → numKVHeads * headDim]
  wV : Linear.LinearLayer         -- V projection [hiddenSize → numKVHeads * headDim]
  wO : Linear.LinearLayer         -- Output projection [numHeads * headDim → hiddenSize]
  qNormWeight : Buffer            -- Per-head Q norm [headDim]
  kNormWeight : Buffer            -- Per-head K norm [headDim]

/-- Gemma 4 dense FFN layer -/
structure Gemma4FFN where
  gate : Linear.LinearLayer       -- Gate projection [hiddenSize → intermediateSize]
  up : Linear.LinearLayer         -- Up projection [hiddenSize → intermediateSize]
  down : Linear.LinearLayer       -- Down projection [intermediateSize → hiddenSize]

/-- Gemma 4 transformer block (single layer) -/
structure Gemma4Block where
  layerIdx : Nat
  layerType : LayerType
  -- Norms
  attnNorm : RMSNorm.RMSNorm
  postAttnNorm : RMSNorm.RMSNorm
  ffnNorm : RMSNorm.RMSNorm
  postFFNNorm : RMSNorm.RMSNorm
  -- Attention
  attention : Gemma4Attention
  -- FFN (shared/dense expert)
  ffn : Gemma4FFN
  -- MoE (optional: present only for MoE layers)
  isMoE : Bool                        -- true if this is a MoE layer
  moeRouterWeight : Option Buffer     -- ffn_gate_inp [numExperts, hiddenSize]
  moeRouterScale : Option Buffer      -- ffn_gate_inp.scale [hiddenSize]
  moeGateUpExps : Option Buffer       -- ffn_gate_up_exps [numExperts, 2*expertFFSize, hiddenSize]
  moeDownExps : Option Buffer         -- ffn_down_exps [numExperts, hiddenSize, expertFFSize]
  moePreNorm2 : Option RMSNorm.RMSNorm  -- ffn_pre_norm_2
  moePostNorm1 : Option RMSNorm.RMSNorm -- ffn_post_norm_1 (after shared expert)
  moePostNorm2 : Option RMSNorm.RMSNorm -- ffn_post_norm_2 (after routed experts)
  -- Optional: RoPE frequency factors (full attention layers only)
  ropeFreqFactors : Option Buffer
  -- Optional: layer output scale
  outScale : Option Buffer

/-- Per-layer embedding weights for a single block -/
structure Gemma4PerLayerEmbd where
  inpGateWeight : Buffer          -- per_layer_inp_gate [hiddenSize, embdPerLayer]
  projWeight : Buffer             -- per_layer_proj [embdPerLayer, hiddenSize]
  postNorm : RMSNorm.RMSNorm     -- per_layer_post_norm

/-- Complete Gemma 4 model -/
structure Gemma4Model where
  config : Config
  embedding : Embedding.Embedding
  blocks : Array Gemma4Block
  finalNorm : RMSNorm.RMSNorm
  outputWeight : Buffer           -- LM head [vocabSize, hiddenSize]
  -- Per-layer embeddings (optional)
  perLayerEmbdTable : Option Buffer         -- tok_embd_per_layer [vocabSize, embdPerLayer * numLayers]
  perLayerModelProj : Option Buffer         -- per_layer_model_proj [embdPerLayer * numLayers, hiddenSize]
  perLayerProjNorm : Option RMSNorm.RMSNorm -- per_layer_proj_norm
  perLayerBlocks : Array (Option Gemma4PerLayerEmbd)  -- per-layer gate/proj/norm

/-! ## Helper: Create GPU Buffer from ByteArray -/

private def uploadBuffer (device : Device) (data : ByteArray) (usage : List WebGPU.BufferUsage := [.storage, .copyDst]) : IO Buffer := do
  let bufSize := if data.size == 0 then 4 else data.size
  let buf ← createBuffer device {
    size := bufSize.toUSize
    usage := usage
    mappedAtCreation := false
  }
  if data.size > 0 then
    writeBuffer device buf 0 data
  return buf

/-! ## GGUF Model Loading -/

/-- Load a single quantized linear layer from GGUF tensor.
    Detects quant format (Q4_K vs Q6_K) and selects the appropriate fused kernel. -/
private def loadLinear (device : Device) (gguf : Hesper.GGUF.GGUFFile)
    (name : String) (inDim outDim : Nat) : IO Linear.LinearLayer := do
  -- Detect quant format from tensor type
  let tensorInfo ← match Hesper.GGUF.Loader.findTensor gguf name with
    | .ok ti => pure ti
    | .error e => throw $ IO.userError e
  let quantFormat : Linear.QuantFormat := match tensorInfo.ggmlType with
    | .Q6_K => .Q6_K
    | _ => .Q4_K
  let (_, data) ← match Hesper.GGUF.Loader.getTensorData gguf name with
    | .ok r => pure r
    | .error e => throw $ IO.userError e
  let weightBuf ← uploadBuffer device data
  let prepared ← IO.mkRef none
  return {
    config := { inDim, outDim }
    weightBuf
    quantFormat
    prepared
  }

/-- Load Gemma 4 model from GGUF file -/
def Gemma4Model.fromGGUF (device : Device) (ggufPath : String)
    (configOverride : Option Config := none) : IO Gemma4Model := do
  IO.println s!"[Gemma4] Loading model from {ggufPath}..."

  -- Step 1: Parse GGUF file
  let ggufData ← IO.FS.readBinFile ggufPath
  let gguf ← match Hesper.GGUF.Parser.parseGGUF ggufData with
    | .ok gf => pure gf
    | .error e => throw $ IO.userError s!"GGUF parse error: {e}"

  IO.println s!"  ✓ GGUF parsed: {gguf.tensors.size} tensors, {gguf.dataBlob.size} bytes data"

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
  -- Gemma 4 embeddings are typically Q4_K or F16
  let embTensor ← match Hesper.GGUF.Loader.findTensor gguf "token_embd.weight" with
    | .ok ti => pure ti
    | .error e => throw $ IO.userError e
  let embedding ← match embTensor.ggmlType with
    | .F16 =>
      IO.println "  Using F16 embeddings"
      let embData ← Hesper.GGUF.Loader.extractF16Tensor gguf "token_embd.weight"
      Embedding.createFromF16 device embConfig embData
    | other =>
      IO.println s!"  Embedding type: {other} — loading as raw bytes"
      -- Fallback: load raw tensor data
      let (_, data) ← match Hesper.GGUF.Loader.getTensorData gguf "token_embd.weight" with
        | .ok r => pure r
        | .error e => throw $ IO.userError e
      let buf ← uploadBuffer device data
      -- Create a minimal embedding structure
      pure { config := embConfig, embeddingTable := buf, f16Table := none }

  -- Step 4: Load transformer blocks
  IO.println s!"[Gemma4] Loading {cfg.numHiddenLayers} transformer blocks..."
  let mut blocks : Array Gemma4Block := #[]

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
    let attnNorm ← RMSNorm.create device normConfig attnNormData
    let postAttnNorm ← RMSNorm.create device normConfig postAttnNormData
    let ffnNorm ← RMSNorm.create device normConfig ffnNormData
    let postFFNNorm ← RMSNorm.create device normConfig postFFNNormData

    -- Load attention projections (Q4_K)
    let qDim := cfg.numAttentionHeads * headDim
    let kvDim := numKVHeads * headDim
    let wQ ← loadLinear device gguf s!"blk.{li}.attn_q.weight" cfg.hiddenSize qDim
    let wK ← loadLinear device gguf s!"blk.{li}.attn_k.weight" cfg.hiddenSize kvDim
    let wV ← loadLinear device gguf s!"blk.{li}.attn_v.weight" cfg.hiddenSize kvDim
    let wO ← loadLinear device gguf s!"blk.{li}.attn_output.weight" qDim cfg.hiddenSize

    -- Load Q/K norm weights (Float32, per-head dimension)
    let qNormData ← Hesper.GGUF.Loader.extractFloat32Tensor gguf s!"blk.{li}.attn_q_norm.weight"
    let kNormData ← Hesper.GGUF.Loader.extractFloat32Tensor gguf s!"blk.{li}.attn_k_norm.weight"
    let qNormBuf ← uploadBuffer device qNormData
    let kNormBuf ← uploadBuffer device kNormData

    let attention : Gemma4Attention := {
      wQ, wK, wV, wO
      qNormWeight := qNormBuf
      kNormWeight := kNormBuf
    }

    -- Load FFN projections (Q4_K)
    let ffnGate ← loadLinear device gguf s!"blk.{li}.ffn_gate.weight" cfg.hiddenSize cfg.intermediateSize
    let ffnUp ← loadLinear device gguf s!"blk.{li}.ffn_up.weight" cfg.hiddenSize cfg.intermediateSize
    let ffnDown ← loadLinear device gguf s!"blk.{li}.ffn_down.weight" cfg.intermediateSize cfg.hiddenSize

    let ffn : Gemma4FFN := { gate := ffnGate, up := ffnUp, down := ffnDown }

    -- Load optional RoPE frequency factors
    -- In the E4B model, rope_freqs is a global tensor, not per-layer
    let ropeFreqFactors ← if cfg.isFullAttention li then
      match Hesper.GGUF.Loader.getTensorData gguf "rope_freqs.weight" with
      | .ok (_, data) =>
        let buf ← uploadBuffer device data
        pure (some buf)
      | .error _ =>
        -- Try per-layer
        match Hesper.GGUF.Loader.getTensorData gguf s!"blk.{li}.rope_freqs.weight" with
        | .ok (_, data) =>
          let buf ← uploadBuffer device data
          pure (some buf)
        | .error _ => pure none
    else pure none

    -- Load optional layer output scale
    let outScale ← match Hesper.GGUF.Loader.getTensorData gguf s!"blk.{li}.layer_output_scale.weight" with
      | .ok (_, data) =>
        let buf ← uploadBuffer device data
        pure (some buf)
      | .error _ => pure none

    -- Load MoE weights (if present for this layer)
    let isMoE := match Hesper.GGUF.Loader.findTensor gguf s!"blk.{li}.ffn_gate_inp.weight" with
      | .ok _ => true | .error _ => false
    let (moeRouterWeight, moeRouterScale, moeGateUpExps, moeDownExps, moePreNorm2, moePostNorm1, moePostNorm2) ←
      if isMoE then do
        IO.println s!"    Layer {li}: MoE layer"
        let routerW ← match Hesper.GGUF.Loader.getTensorData gguf s!"blk.{li}.ffn_gate_inp.weight" with
          | .ok (_, data) => pure (some (← uploadBuffer device data))
          | .error _ => pure none
        let routerS ← match Hesper.GGUF.Loader.getTensorData gguf s!"blk.{li}.ffn_gate_inp.scale" with
          | .ok (_, data) => pure (some (← uploadBuffer device data))
          | .error _ => pure none
        let gateUpE ← match Hesper.GGUF.Loader.getTensorData gguf s!"blk.{li}.ffn_gate_up_exps.weight" with
          | .ok (_, data) => pure (some (← uploadBuffer device data))
          | .error _ => pure none
        let downE ← match Hesper.GGUF.Loader.getTensorData gguf s!"blk.{li}.ffn_down_exps.weight" with
          | .ok (_, data) => pure (some (← uploadBuffer device data))
          | .error _ => pure none
        let preN2 ← match Hesper.GGUF.Loader.findTensor gguf s!"blk.{li}.ffn_pre_norm_2.weight" with
          | .ok _ =>
            let d ← Hesper.GGUF.Loader.extractFloat32Tensor gguf s!"blk.{li}.ffn_pre_norm_2.weight"
            pure (some (← RMSNorm.create device normConfig d))
          | .error _ => pure none
        let postN1 ← match Hesper.GGUF.Loader.findTensor gguf s!"blk.{li}.ffn_post_norm_1.weight" with
          | .ok _ =>
            let d ← Hesper.GGUF.Loader.extractFloat32Tensor gguf s!"blk.{li}.ffn_post_norm_1.weight"
            pure (some (← RMSNorm.create device normConfig d))
          | .error _ => pure none
        let postN2 ← match Hesper.GGUF.Loader.findTensor gguf s!"blk.{li}.ffn_post_norm_2.weight" with
          | .ok _ =>
            let d ← Hesper.GGUF.Loader.extractFloat32Tensor gguf s!"blk.{li}.ffn_post_norm_2.weight"
            pure (some (← RMSNorm.create device normConfig d))
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
  let finalNorm ← RMSNorm.create device finalNormConfig finalNormData

  -- Step 6: LM head (output.weight or weight-tied with embedding)
  let outputWeight ← match Hesper.GGUF.Loader.getTensorData gguf "output.weight" with
    | .ok (_, data) =>
      IO.println "  Using separate LM head weights"
      uploadBuffer device data
    | .error _ =>
      IO.println "  Using weight-tied LM head (reusing embedding)"
      pure embedding.embeddingTable

  -- Step 7: Load per-layer embeddings (optional)
  let (perLayerEmbdTable, perLayerModelProj, perLayerProjNorm) ← if cfg.hasPerLayerEmbeddings then do
    IO.println "[Gemma4] Loading per-layer embeddings..."
    let table ← match Hesper.GGUF.Loader.getTensorData gguf "per_layer_token_embd.weight" with
      | .ok (_, data) => pure (some (← uploadBuffer device data))
      | .error _ => pure none
    let proj ← match Hesper.GGUF.Loader.getTensorData gguf "per_layer_model_proj.weight" with
      | .ok (_, data) => pure (some (← uploadBuffer device data))
      | .error _ => pure none
    let projNorm ← match Hesper.GGUF.Loader.findTensor gguf "per_layer_proj_norm.weight" with
      | .ok _ =>
        let d ← Hesper.GGUF.Loader.extractFloat32Tensor gguf "per_layer_proj_norm.weight"
        let plNormConfig : RMSNorm.Config := { dim := cfg.embdPerLayer * cfg.numHiddenLayers, eps := cfg.rmsNormEps }
        pure (some (← RMSNorm.create device plNormConfig d))
      | .error _ => pure none
    pure (table, proj, projNorm)
  else
    pure (none, none, none)

  -- Per-layer gate/proj/norm per block
  let mut perLayerBlocks : Array (Option Gemma4PerLayerEmbd) := #[]
  for li in [0:cfg.numHiddenLayers] do
    if cfg.hasPerLayerEmbeddings then
      let plEmbd ← do
        let gateW ← match Hesper.GGUF.Loader.getTensorData gguf s!"blk.{li}.inp_gate.weight" with
          | .ok (_, data) => pure (← uploadBuffer device data)
          | .error _ => pure (← uploadBuffer device ByteArray.empty)
        let projW ← match Hesper.GGUF.Loader.getTensorData gguf s!"blk.{li}.proj.weight" with
          | .ok (_, data) => pure (← uploadBuffer device data)
          | .error _ => pure (← uploadBuffer device ByteArray.empty)
        let postNorm ← do
          let normConfig : RMSNorm.Config := { dim := cfg.hiddenSize, eps := cfg.rmsNormEps }
          match Hesper.GGUF.Loader.findTensor gguf s!"blk.{li}.post_norm.weight" with
          | .ok _ =>
            let d ← Hesper.GGUF.Loader.extractFloat32Tensor gguf s!"blk.{li}.post_norm.weight"
            RMSNorm.create device normConfig d
          | .error _ =>
            -- Fallback: create with dummy weights
            RMSNorm.create device normConfig (ByteArray.mk (Array.mk (List.replicate (cfg.hiddenSize * 4) 0)))
        pure (some { inpGateWeight := gateW, projWeight := projW, postNorm := postNorm : Gemma4PerLayerEmbd })
      perLayerBlocks := perLayerBlocks.push plEmbd
    else
      perLayerBlocks := perLayerBlocks.push none

  IO.println s!"[Gemma4] ✓ Model loaded: {blocks.size} blocks"

  return {
    config := cfg
    embedding
    blocks
    finalNorm
    outputWeight
    perLayerEmbdTable
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

/-! ## KV Cache State -/

/-- Per-layer KV cache for Gemma 4 -/
structure Gemma4KVCache where
  kBuf : Buffer    -- [numKVHeads, maxSeqLen, headDim]
  vBuf : Buffer    -- [numKVHeads, maxSeqLen, headDim]

/-- Full inference state -/
structure InferenceState where
  kvCaches : Array Gemma4KVCache
  buf1 : Buffer          -- [hiddenSize] ping-pong
  buf2 : Buffer          -- [hiddenSize] ping-pong
  qBuf : Buffer          -- [numHeads * headDim] Q projection output
  kBuf : Buffer          -- [numKVHeads * headDim] K projection output
  vBuf : Buffer          -- [numKVHeads * headDim] V projection output
  attnOutBuf : Buffer    -- [numHeads * headDim] attention output
  gateBuf : Buffer       -- [intermediateSize] FFN gate output
  upBuf : Buffer         -- [intermediateSize] FFN up output
  geluBuf : Buffer       -- [intermediateSize] GELU*up output
  ffnOutBuf : Buffer     -- [hiddenSize] FFN down output
  normedBuf : Buffer     -- [hiddenSize] normalized output
  logitsBuf : Buffer     -- [vocabSize]
  tokenBuf : Buffer      -- [1] u32 for single token
  paramsBuf : Buffer     -- [2] u32: (pos, cacheLen) for RoPE
  -- MoE buffers
  moeRouterOutBuf : Buffer    -- [hiddenSize] router preprocessed input
  moeLogitsBuf : Buffer       -- [numExperts] router logits
  moeIndicesBuf : Buffer      -- [numExpertsUsed] selected expert indices
  moeWeightsBuf : Buffer      -- [numExpertsUsed] expert weights
  moeExpertOutBuf : Buffer    -- [hiddenSize] combined expert output
  moeExpertGateBuf : Buffer   -- [expertFFSize] expert gate projection output
  moeExpertUpBuf : Buffer     -- [expertFFSize] expert up projection output
  moeExpertGeluBuf : Buffer   -- [expertFFSize] expert GELU*up output
  moeExpertDownBuf : Buffer   -- [hiddenSize] single expert down output
  moeNormedBuf : Buffer       -- [hiddenSize] pre_norm_2 output for routed experts
  -- Per-layer embedding buffers
  plGateBuf : Buffer          -- [embdPerLayer] per-layer gate output
  plProjBuf : Buffer          -- [hiddenSize] per-layer projected output

/-- Create inference state with pre-allocated buffers -/
def createInferenceState (device : Device) (cfg : Config) : IO InferenceState := do
  let mkBuf := fun (size : Nat) => createBuffer device {
    size := (size * 4).toUSize  -- f32 = 4 bytes
    usage := [.storage, .copySrc, .copyDst]
    mappedAtCreation := false
  }
  let maxHeadDim := max cfg.headDimFull cfg.headDimSWA
  let maxQDim := cfg.numAttentionHeads * maxHeadDim
  let maxKVDim := (max cfg.numKeyValueHeadsFull cfg.numKeyValueHeadsSWA) * maxHeadDim

  -- Create per-layer KV caches
  let mut kvCaches : Array Gemma4KVCache := #[]
  for li in [0:cfg.numHiddenLayers] do
    let numKVHeads := cfg.numKVHeads li
    let headDim := cfg.headDim li
    let cacheSize := numKVHeads * cfg.maxSeqLen * headDim
    let kBuf ← mkBuf cacheSize
    let vBuf ← mkBuf cacheSize
    kvCaches := kvCaches.push { kBuf, vBuf }

  return {
    kvCaches
    buf1 := ← mkBuf cfg.hiddenSize
    buf2 := ← mkBuf cfg.hiddenSize
    qBuf := ← mkBuf maxQDim
    kBuf := ← mkBuf maxKVDim
    vBuf := ← mkBuf maxKVDim
    attnOutBuf := ← mkBuf maxQDim
    gateBuf := ← mkBuf cfg.intermediateSize
    upBuf := ← mkBuf cfg.intermediateSize
    geluBuf := ← mkBuf cfg.intermediateSize
    ffnOutBuf := ← mkBuf cfg.hiddenSize
    normedBuf := ← mkBuf cfg.hiddenSize
    logitsBuf := ← mkBuf cfg.vocabSize
    tokenBuf := ← createBuffer device { size := 4, usage := [.storage, .copySrc, .copyDst], mappedAtCreation := false }
    paramsBuf := ← createBuffer device { size := 8, usage := [.storage, .copySrc, .copyDst], mappedAtCreation := false }
    moeRouterOutBuf := ← mkBuf cfg.hiddenSize
    moeLogitsBuf := ← mkBuf (max cfg.numExperts 1)
    moeIndicesBuf := ← createBuffer device { size := (max cfg.numExpertsUsed 1 * 4).toUSize, usage := [.storage, .copySrc, .copyDst], mappedAtCreation := false }
    moeWeightsBuf := ← mkBuf (max cfg.numExpertsUsed 1)
    moeExpertOutBuf := ← mkBuf cfg.hiddenSize
    moeExpertGateBuf := ← mkBuf (max cfg.expertFFSize 1)
    moeExpertUpBuf := ← mkBuf (max cfg.expertFFSize 1)
    moeExpertGeluBuf := ← mkBuf (max cfg.expertFFSize 1)
    moeExpertDownBuf := ← mkBuf cfg.hiddenSize
    moeNormedBuf := ← mkBuf cfg.hiddenSize
    plGateBuf := ← mkBuf (max cfg.embdPerLayer 1)
    plProjBuf := ← mkBuf cfg.hiddenSize
  }

/-! ## Single-Token Forward Pass -/

/-- Run single-token forward pass through one transformer block.

    Flow (from gemma4-iswa.cpp):
    1. attnNorm(input) → Q/K/V projections → Q-norm, K-norm → attention → postAttnNorm → + residual
    2. ffnNorm(attn_out) → GeGLU FFN → postFFNNorm → + residual
-/
def forwardBlock (device : Device) (block : Gemma4Block) (cfg : Config)
    (inputBuf outputBuf : Buffer) (state : InferenceState) (pos : Nat)
    (perLayerEmbd : Option Gemma4PerLayerEmbd := none)
    (perLayerInput : Option Buffer := none) : IO Unit := do
  let li := block.layerIdx
  let headDim := cfg.headDim li

  -- Step 1: Attention pre-norm
  RMSNorm.forward device block.attnNorm inputBuf state.normedBuf

  -- Step 2: Q/K/V projections
  Linear.LinearLayer.forward device block.attention.wQ state.normedBuf state.qBuf
  if cfg.hasKV li then
    Linear.LinearLayer.forward device block.attention.wK state.normedBuf state.kBuf
    Linear.LinearLayer.forward device block.attention.wV state.normedBuf state.vBuf

  -- Step 3: Q-norm, K-norm (per-head RMSNorm)
  let numHeads := cfg.numAttentionHeads
  let numKVHeads := cfg.numKVHeads li
  Hesper.WGSL.Execute.executeShaderNamed device
    (perHeadRMSNormKernel numHeads headDim cfg.rmsNormEps)
    [("input", state.qBuf), ("weight", block.attention.qNormWeight), ("output", state.qBuf)]
    (.dispatch1D numHeads headDim)

  if cfg.hasKV li then
    Hesper.WGSL.Execute.executeShaderNamed device
      (perHeadRMSNormKernel numKVHeads headDim cfg.rmsNormEps)
      [("input", state.kBuf), ("weight", block.attention.kNormWeight), ("output", state.kBuf)]
      (.dispatch1D numKVHeads headDim)

    -- V-norm: bare RMSNorm on V (no learned weights)
    -- From gemma4-iswa.cpp:82: Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps)
    let vDim := numKVHeads * headDim
    Hesper.WGSL.Execute.executeShaderNamed device
      (bareRMSNormKernel vDim cfg.rmsNormEps)
      [("input", state.vBuf), ("output", state.vBuf)]
      (Execute.ExecutionConfig.default (1, 1, 1))

  -- Step 4: RoPE on Q and K
  -- Upload position to params buffer
  let posBytes := Hesper.WebGPU.BufferOps.uint32ToBytes pos.toUInt32
  writeBuffer device state.paramsBuf 0 posBytes

  match block.ropeFreqFactors with
  | some freqFactors =>
    -- Full attention: RoPE with frequency factors
    Hesper.WGSL.Execute.executeShaderNamed device
      (ropeWithFreqFactorsKernel headDim numHeads cfg.ropeTheta)
      [("input", state.qBuf), ("output", state.qBuf), ("params", state.paramsBuf), ("freq_factors", freqFactors)]
      (.dispatch1D (numHeads * headDim / 2))
  | none =>
    -- SWA: standard RoPE (use existing dynamic kernel)
    let ropeConfig : RoPE.Config := { dim := numHeads * headDim, maxSeqLen := cfg.maxSeqLen, base := cfg.ropeTheta }
    Hesper.WGSL.Execute.executeShaderNamed device
      (RoPE.ropeKernelDynamic ropeConfig 1 1 numHeads headDim)
      [("input", state.qBuf), ("output", state.qBuf), ("params", state.paramsBuf)]
      (.dispatch1D (numHeads * headDim / 2))

  if cfg.hasKV li then
    let ropeConfig : RoPE.Config := { dim := numKVHeads * headDim, maxSeqLen := cfg.maxSeqLen, base := cfg.ropeTheta }
    Hesper.WGSL.Execute.executeShaderNamed device
      (RoPE.ropeKernelDynamic ropeConfig 1 1 numKVHeads headDim)
      [("input", state.kBuf), ("output", state.kBuf), ("params", state.paramsBuf)]
      (.dispatch1D (numKVHeads * headDim / 2))

  -- Step 5: Write K/V to cache and compute flash attention
  if h : li < state.kvCaches.size then
    let kvCache := state.kvCaches[li]
    let kvDim := numKVHeads * headDim
    let cacheLen := pos + 1  -- number of cached positions including current

    -- Write K and V to cache at current position (fused kernel)
    if cfg.hasKV li then
      Hesper.WGSL.Execute.executeShaderNamed device
        (Attention.fusedCacheWriteKVKernel numKVHeads cfg.maxSeqLen headDim kvDim)
        [("new_k", state.kBuf), ("new_v", state.vBuf),
         ("k_cache", kvCache.kBuf), ("v_cache", kvCache.vBuf),
         ("params", state.paramsBuf)]
        (.dispatch1D kvDim)

    -- Flash attention: Q @ K_cache^T → softmax → @ V_cache → output
    let scale := 1.0 / Float.sqrt headDim.toFloat
    match block.layerType with
    | .swa =>
      -- Sliding window attention: only attend within windowSize
      Hesper.WGSL.Execute.executeShaderNamed device
        (FlashAttention.flashAttentionSWAKernel numHeads numKVHeads cfg.maxSeqLen headDim cacheLen cfg.slidingWindowSize pos scale)
        [("q", state.qBuf), ("k_cache", kvCache.kBuf), ("v_cache", kvCache.vBuf), ("output", state.attnOutBuf)]
        (Execute.ExecutionConfig.default (numHeads, 1, 1))
    | .full =>
      -- Full attention: attend to all cached positions
      Hesper.WGSL.Execute.executeShaderNamed device
        (FlashAttention.flashAttentionDynamicKernel numHeads numKVHeads cfg.maxSeqLen headDim cacheLen scale)
        [("q", state.qBuf), ("k_cache", kvCache.kBuf), ("v_cache", kvCache.vBuf), ("output", state.attnOutBuf)]
        (Execute.ExecutionConfig.default (numHeads, 1, 1))

    -- Output projection: attnOut [numHeads * headDim] → normedBuf [hiddenSize]
    Linear.LinearLayer.forward device block.attention.wO state.attnOutBuf state.normedBuf
  else
    -- Fallback: skip attention (shouldn't happen)
    Linear.LinearLayer.forward device block.attention.wO state.qBuf state.normedBuf

  -- Step 6: Post-attention norm + residual
  RMSNorm.forward device block.postAttnNorm state.normedBuf state.normedBuf
  Hesper.WGSL.Execute.executeShaderNamed device
    (residualAddKernel cfg.hiddenSize)
    [("a", state.normedBuf), ("b", inputBuf), ("output", state.buf1)]
    (.dispatch1D cfg.hiddenSize)

  -- Step 7: FFN (dense or MoE)
  if block.isMoE then do
    -- MoE layer (from gemma4-iswa.cpp:117-169):
    -- 1. Shared expert: ffn_norm → GeGLU FFN → post_norm_1
    RMSNorm.forward device block.ffnNorm state.buf1 state.normedBuf
    Linear.LinearLayer.forward device block.ffn.gate state.normedBuf state.gateBuf
    Linear.LinearLayer.forward device block.ffn.up state.normedBuf state.upBuf
    Hesper.WGSL.Execute.executeShaderNamed device
      (geluMulKernel cfg.intermediateSize)
      [("gate", state.gateBuf), ("up", state.upBuf), ("output", state.geluBuf)]
      (.dispatch1D cfg.intermediateSize)
    Linear.LinearLayer.forward device block.ffn.down state.geluBuf state.ffnOutBuf

    -- Apply post_norm_1 to shared expert output
    match block.moePostNorm1 with
    | some norm => RMSNorm.forward device norm state.ffnOutBuf state.ffnOutBuf
    | none => pure ()

    -- 2. Router: rms_norm(attn_out) * (1/sqrt(n_embd)) * router_scale → logits → softmax → top-K
    match block.moeRouterWeight, block.moeRouterScale with
    | some routerW, some routerS =>
      Hesper.WGSL.Execute.executeShaderNamed device
        (MoE.routerPreprocessKernel cfg.hiddenSize cfg.rmsNormEps)
        [("input", state.buf1), ("router_scale", routerS), ("output", state.moeRouterOutBuf)]
        (Execute.ExecutionConfig.default (1, 1, 1))
      -- Router matmul: moeRouterOutBuf [hiddenSize] @ routerW^T → moeLogitsBuf [numExperts]
      let routerMatmulConfig : Hesper.WGSL.MatMul.Config := {
        M := 1, N := cfg.numExperts, K := cfg.hiddenSize
      }
      Hesper.WGSL.MatMul.executeMatMulTranspose device state.moeRouterOutBuf routerW state.moeLogitsBuf routerMatmulConfig
      -- Top-K selection
      Hesper.WGSL.Execute.executeShaderNamed device
        (MoE.softmaxTopKKernel cfg.numExperts cfg.numExpertsUsed)
        [("logits", state.moeLogitsBuf), ("indices", state.moeIndicesBuf), ("weights", state.moeWeightsBuf)]
        (.dispatch1D 1)
    | _, _ => pure ()

    -- 3. Routed experts: ffn_pre_norm_2 → expert GeGLU FFN → weighted sum
    match block.moeGateUpExps, block.moeDownExps, block.moePreNorm2, block.moePostNorm2 with
    | some gateUpExps, some downExps, some preNorm2, some postNorm2 =>
      -- Pre-norm for routed expert input
      RMSNorm.forward device preNorm2 state.buf1 state.moeNormedBuf

      -- Zero the accumulator
      Hesper.WGSL.Execute.executeShaderNamed device
        (residualAddKernel cfg.hiddenSize)  -- hack: 0 + 0 = 0 (both inputs are same zeroed buf)
        [("a", state.moeExpertOutBuf), ("b", state.moeExpertOutBuf), ("output", state.moeExpertOutBuf)]
        (.dispatch1D cfg.hiddenSize)
      -- Actually zero it properly
      Hesper.WGSL.Execute.executeShaderNamed device
        (embeddingScaleKernel cfg.hiddenSize 0)  -- scale by 0 to zero
        [("input", state.moeExpertOutBuf), ("output", state.moeExpertOutBuf)]
        (.dispatch1D cfg.hiddenSize)

      let moeConfig : MoE.Config := {
        hiddenSize := cfg.hiddenSize
        expertFFSize := cfg.expertFFSize
        numExperts := cfg.numExperts
        numExpertsUsed := cfg.numExpertsUsed
        rmsNormEps := cfg.rmsNormEps
      }

      -- For each selected expert: gate+up → GELU*up → down → weighted accumulate
      for k in [0:cfg.numExpertsUsed] do
        -- Gate projection
        Hesper.WGSL.Execute.executeShaderNamed device
          (MoE.expertGateUpKernel moeConfig k true)
          [("input", state.moeNormedBuf), ("gate_up_weights", gateUpExps),
           ("expert_indices", state.moeIndicesBuf), ("output", state.moeExpertGateBuf)]
          (Execute.ExecutionConfig.default (cfg.expertFFSize, 1, 1))
        -- Up projection
        Hesper.WGSL.Execute.executeShaderNamed device
          (MoE.expertGateUpKernel moeConfig k false)
          [("input", state.moeNormedBuf), ("gate_up_weights", gateUpExps),
           ("expert_indices", state.moeIndicesBuf), ("output", state.moeExpertUpBuf)]
          (Execute.ExecutionConfig.default (cfg.expertFFSize, 1, 1))
        -- GELU * up
        Hesper.WGSL.Execute.executeShaderNamed device
          (MoE.expertGeluMulKernel cfg.expertFFSize)
          [("gate", state.moeExpertGateBuf), ("up", state.moeExpertUpBuf), ("output", state.moeExpertGeluBuf)]
          (.dispatch1D cfg.expertFFSize)
        -- Down projection
        Hesper.WGSL.Execute.executeShaderNamed device
          (MoE.expertDownKernel moeConfig k)
          [("input", state.moeExpertGeluBuf), ("down_weights", downExps),
           ("expert_indices", state.moeIndicesBuf), ("output", state.moeExpertDownBuf)]
          (Execute.ExecutionConfig.default (cfg.hiddenSize, 1, 1))
        -- Weighted accumulate: moeExpertOutBuf += weight[k] * expertDownBuf
        Hesper.WGSL.Execute.executeShaderNamed device
          (MoE.weightedAccumulateKernel cfg.hiddenSize cfg.numExpertsUsed k)
          [("accumulator", state.moeExpertOutBuf), ("expert_output", state.moeExpertDownBuf),
           ("weights", state.moeWeightsBuf)]
          (.dispatch1D cfg.hiddenSize)

      -- post_norm_2 on routed expert output
      RMSNorm.forward device postNorm2 state.moeExpertOutBuf state.moeExpertOutBuf

      -- 4. Combined: shared_expert + routed_experts
      Hesper.WGSL.Execute.executeShaderNamed device
        (residualAddKernel cfg.hiddenSize)
        [("a", state.ffnOutBuf), ("b", state.moeExpertOutBuf), ("output", state.ffnOutBuf)]
        (.dispatch1D cfg.hiddenSize)
    | _, _, _, _ => pure ()  -- No MoE weights: shared expert only

    -- Post-FFN norm + residual
    RMSNorm.forward device block.postFFNNorm state.ffnOutBuf state.ffnOutBuf
    Hesper.WGSL.Execute.executeShaderNamed device
      (residualAddKernel cfg.hiddenSize)
      [("a", state.ffnOutBuf), ("b", state.buf1), ("output", outputBuf)]
      (.dispatch1D cfg.hiddenSize)
  else do
    -- Dense FFN path (GeGLU)
    RMSNorm.forward device block.ffnNorm state.buf1 state.normedBuf
    Linear.LinearLayer.forward device block.ffn.gate state.normedBuf state.gateBuf
    Linear.LinearLayer.forward device block.ffn.up state.normedBuf state.upBuf
    Hesper.WGSL.Execute.executeShaderNamed device
      (geluMulKernel cfg.intermediateSize)
      [("gate", state.gateBuf), ("up", state.upBuf), ("output", state.geluBuf)]
      (.dispatch1D cfg.intermediateSize)
    Linear.LinearLayer.forward device block.ffn.down state.geluBuf state.ffnOutBuf

    -- Post-FFN norm + residual
    RMSNorm.forward device block.postFFNNorm state.ffnOutBuf state.ffnOutBuf
    Hesper.WGSL.Execute.executeShaderNamed device
      (residualAddKernel cfg.hiddenSize)
      [("a", state.ffnOutBuf), ("b", state.buf1), ("output", outputBuf)]
      (.dispatch1D cfg.hiddenSize)

  -- Step 8: Per-layer embedding (optional, from gemma4-iswa.cpp:192-213)
  -- gate = GELU(per_layer_inp_gate @ cur)
  -- gate * per_layer_input[layerIdx] → project → normalize → + residual
  match perLayerEmbd, perLayerInput with
  | some plEmbd, some plInput =>
    -- per_layer_inp_gate @ outputBuf → plGateBuf [embdPerLayer]
    let plGateConfig : Hesper.WGSL.MatMul.Config := {
      M := 1, N := cfg.embdPerLayer, K := cfg.hiddenSize
    }
    Hesper.WGSL.MatMul.executeMatMulTranspose device outputBuf plEmbd.inpGateWeight state.plGateBuf plGateConfig

    -- GELU(gate) * per_layer_input → plGateBuf
    Hesper.WGSL.Execute.executeShaderNamed device
      (PerLayerEmbedding.geluGateMulKernel cfg.embdPerLayer)
      [("gate", state.plGateBuf), ("per_layer_input", plInput), ("output", state.plGateBuf)]
      (.dispatch1D cfg.embdPerLayer)

    -- per_layer_proj @ plGateBuf → plProjBuf [hiddenSize]
    let plProjConfig : Hesper.WGSL.MatMul.Config := {
      M := 1, N := cfg.hiddenSize, K := cfg.embdPerLayer
    }
    Hesper.WGSL.MatMul.executeMatMulTranspose device state.plGateBuf plEmbd.projWeight state.plProjBuf plProjConfig

    -- per_layer_post_norm
    RMSNorm.forward device plEmbd.postNorm state.plProjBuf state.plProjBuf

    -- residual: outputBuf += plProjBuf
    Hesper.WGSL.Execute.executeShaderNamed device
      (residualAddKernel cfg.hiddenSize)
      [("a", outputBuf), ("b", state.plProjBuf), ("output", outputBuf)]
      (.dispatch1D cfg.hiddenSize)
  | _, _ => pure ()

  -- Step 9: Layer output scale (optional)
  match block.outScale with
  | some scale =>
    Hesper.WGSL.Execute.executeShaderNamed device
      (PerLayerEmbedding.layerScaleKernel cfg.hiddenSize)
      [("input", outputBuf), ("scale", scale), ("output", outputBuf)]
      (.dispatch1D cfg.hiddenSize)
  | none => pure ()

/-- Run full single-token forward pass through the model.
    Returns logits in state.logitsBuf. -/
def forwardSingleToken (device : Device) (model : Gemma4Model)
    (tokenId : Nat) (pos : Nat) (state : InferenceState) : IO Unit := do
  -- Step 1: Embedding lookup + scale
  let tokenBytes := Hesper.WebGPU.BufferOps.uint32ToBytes tokenId.toUInt32
  writeBuffer device state.tokenBuf 0 tokenBytes
  Embedding.forward device model.embedding state.tokenBuf state.buf1 1 1

  -- Scale embeddings by sqrt(hiddenSize)
  Hesper.WGSL.Execute.executeShaderNamed device
    (embeddingScaleKernel model.config.hiddenSize model.config.hiddenSize)
    [("input", state.buf1), ("output", state.buf1)]
    (.dispatch1D model.config.hiddenSize)

  -- Step 2: Process all transformer blocks
  Hesper.WGSL.Execute.beginBatch device

  let mut currentBuf := state.buf1
  let mut nextBuf := state.buf2

  let mut blockIdx := 0
  for block in model.blocks do
    -- Get per-layer embedding for this block (if available)
    let plEmbd := if blockIdx < model.perLayerBlocks.size then
      model.perLayerBlocks[blockIdx]!
    else none
    -- TODO: per-layer input (pre-projected per-layer embeddings) needs to be
    -- precomputed once at the start of forwardSingleToken. For now, skip.
    forwardBlock device block model.config currentBuf nextBuf state pos plEmbd none
    let temp := currentBuf; currentBuf := nextBuf; nextBuf := temp
    blockIdx := blockIdx + 1

  -- Step 3: Final norm
  RMSNorm.forward device model.finalNorm currentBuf nextBuf

  -- Step 4: LM head matmul (1 × hiddenSize @ hiddenSize × vocabSize)
  let lmHeadConfig : Hesper.WGSL.MatMul.Config := {
    M := 1, N := model.config.vocabSize, K := model.config.hiddenSize
  }
  Hesper.WGSL.MatMul.executeMatMulTranspose device nextBuf model.outputWeight state.logitsBuf lmHeadConfig

  -- Step 5: Logit softcapping
  if model.config.logitSoftcapScale > 0.0 then
    Hesper.WGSL.Execute.executeShaderNamed device
      (logitSoftcapKernel model.config.vocabSize model.config.logitSoftcapScale)
      [("input", state.logitsBuf), ("output", state.logitsBuf)]
      (.dispatch1D model.config.vocabSize)

  Hesper.WGSL.Execute.endBatch device

/-! ## Text Generation -/

/-- Generate tokens from a Gemma 4 model.

    @param device WebGPU device
    @param model Loaded Gemma 4 model
    @param promptTokens Input token IDs
    @param maxTokens Maximum new tokens to generate
    @param eosToken Optional EOS token ID for early stopping
-/
def generate (device : Device) (model : Gemma4Model)
    (promptTokens : Array Nat) (maxTokens : Nat)
    (eosToken : Option Nat := none) : IO (Array Nat) := do
  IO.println s!"[Gemma4] Generating: {promptTokens.size} prompt tokens, max {maxTokens} new tokens"

  -- Create inference state
  let state ← createInferenceState device model.config

  let mut tokens := promptTokens

  -- Phase 1: Prefill (process prompt tokens)
  IO.println s!"[Prefill] Processing {promptTokens.size} prompt tokens..."
  let prefillStart ← IO.monoNanosNow
  for i in [0:promptTokens.size] do
    if i >= model.config.maxSeqLen then break
    forwardSingleToken device model promptTokens[i]! i state
  let prefillEnd ← IO.monoNanosNow
  let prefillMs := (prefillEnd - prefillStart).toFloat / 1_000_000.0
  IO.println s!"[Prefill] Done in {prefillMs} ms"

  -- Phase 2: Decode (generate new tokens)
  let genStart ← IO.monoNanosNow
  let mut genCount : Nat := 0

  for _ in [0:maxTokens] do
    if tokens.size >= model.config.maxSeqLen then break

    -- Sample: greedy argmax (download logits to CPU)
    let logits ← Hesper.WebGPU.BufferOps.downloadFloatArray device state.logitsBuf model.config.vocabSize
    let nextToken := Hesper.Inference.Sampling.argmax logits

    tokens := tokens.push nextToken
    genCount := genCount + 1

    -- Check EOS
    match eosToken with
    | some eos => if nextToken == eos then break
    | none => pure ()

    -- Forward pass for next token
    let newPos := tokens.size - 1
    if newPos < model.config.maxSeqLen then
      forwardSingleToken device model nextToken newPos state

  let genEnd ← IO.monoNanosNow
  let genMs := (genEnd - genStart).toFloat / 1_000_000.0
  let msPerToken := if genCount > 0 then genMs / genCount.toFloat else 0.0
  let tps := if msPerToken > 0 then 1000.0 / msPerToken else 0.0
  IO.println s!"[Gemma4] Generated {genCount} tokens in {genMs} ms ({tps} tokens/sec)"

  return tokens

end Hesper.Models.Gemma4
