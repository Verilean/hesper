/-!
# Gemma 4 model configuration

Extracted from the main Gemma4 module.  Contains only the `Config`
struct and its small pure helpers.  No IO, no GGUF, no kernels.
-/

namespace Hesper.Models.Gemma4

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
  ropeTheta : Float            -- Full-attn freq_base: 1000000.0
  ropeThetaSWA : Float         -- SWA freq_base: 10000.0  (gemma4.rope.freq_base_swa)
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

/-- RoPE freq_base for a given layer.  Gemma 4 uses a different rope
    base for SWA layers (10000) vs full-attn layers (1000000).  Per
    `llama.cpp/src/models/gemma4-iswa.cpp:37`:
      freq_base_l = model.get_rope_freq_base(cparams, il)
    which resolves to `freq_base_swa` for SWA layers and `freq_base`
    for full-attn layers.  -/
def Config.ropeBase (c : Config) (layerIdx : Nat) : Float :=
  if c.isFullAttention layerIdx then c.ropeTheta else c.ropeThetaSWA

/-- Check if a layer has its own KV cache (not shared) -/
def Config.hasKV (c : Config) (layerIdx : Nat) : Bool :=
  layerIdx < c.numHiddenLayers - c.numKVSharedLayers

/-- For KV-shared layers, return the index of the earlier layer whose KV cache is reused.
    Mirrors llama.cpp's Gemma 4 layer_reuse_cb (see llama-model.cpp:8355):
      reuse(il) = n_layer_kv_from_start - (is_swa(il) ? 2 : 1)    if il >= n_layer_kv_from_start
                = il                                              otherwise
    The reused layer is always in [0, n_layer_kv_from_start), i.e. it has its own KV cache. -/
def Config.kvCacheLayer (c : Config) (layerIdx : Nat) : Nat :=
  if c.hasKV layerIdx then layerIdx
  else
    let firstShared := c.numHiddenLayers - c.numKVSharedLayers
    if c.isFullAttention layerIdx then firstShared - 1 else firstShared - 2

/-- Check if per-layer embeddings are enabled -/
def Config.hasPerLayerEmbeddings (c : Config) : Bool :=
  c.embdPerLayer > 0

end Hesper.Models.Gemma4
