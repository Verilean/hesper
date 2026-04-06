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

  let findF32Default (key : String) (default : Float) : Float :=
    match findMeta key with
    | some mv => (readMetadataF32 mv).getD default
    | none => default

  let vocabSize := (findU32 "general.vocab_size").toOption.getD 262144
  let hiddenSize ← findU32 "llama.embedding_length"
  let intermediateSize ← findU32 "llama.feed_forward_length"
  let numLayers ← findU32 "llama.block_count"
  let numHeads ← findU32 "llama.attention.head_count"

  -- Layer types: default all full attention for now
  -- TODO: parse llama.attention.sliding_window_pattern (bool array in metadata)
  let layerTypes : Array LayerType := (List.replicate numLayers LayerType.full).toArray

  let rmsNormEps := findF32Default "llama.attention.layer_norm_rms_epsilon" 1e-6
  let ropeTheta := findF32Default "llama.rope.freq_base" 1000000.0

  return {
    vocabSize
    hiddenSize
    intermediateSize
    numHiddenLayers := numLayers
    numAttentionHeads := numHeads
    numKeyValueHeadsFull := (findU32 "llama.attention.head_count_kv").toOption.getD 8
    numKeyValueHeadsSWA := (findU32 "llama.attention.head_count_kv").toOption.getD 8
    headDimFull := (findU32 "llama.attention.key_length").toOption.getD 128
    headDimSWA := (findU32 "llama.attention.key_length_swa").toOption.getD 128
    slidingWindowSize := (findU32 "llama.attention.sliding_window").toOption.getD 512
    rmsNormEps
    ropeTheta
    partialRotaryFactorSWA := 0.5  -- TODO: read from metadata
    layerTypes
    logitSoftcapScale := findF32Default "llama.logit_softcapping" 30.0
    maxSeqLen := 131072
    numExperts := (findU32 "llama.expert_count").toOption.getD 0
    numExpertsUsed := (findU32 "llama.expert_used_count").toOption.getD 0
    expertFFSize := (findU32 "llama.expert_feed_forward_length").toOption.getD 0
    embdPerLayer := (findU32 "llama.embedding_length_per_layer_input").toOption.getD 0
    numKVSharedLayers := (findU32 "llama.attention.shared_kv_layers").toOption.getD 0
  }

end Hesper.Models.Gemma4
