import Hesper.Backend
import Hesper.Backend.WebGPU
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
open Hesper.WGSL.Execute (PreparedDispatch CompiledKernel)
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

/-- Fused post-norm + residual-add kernel, in place on the residual
    buffer. Replaces the 3-dispatch chain used by the per-layer
    embedding:

      norm[i] = plProj[i] * rsqrt(mean(plProj²) + eps) * weight[i]
      residual[i] = residual[i] + norm[i]    -- written in place

    Implementation: 1 workgroup of 256 threads (= 8 subgroups of 32
    each on NVIDIA). Phase 1 accumulates `sum(x²)` per lane then
    reduces via one `subgroupAdd` (intra-subgroup) + one shared-mem
    stash + one barrier + a final cross-subgroup sum on thread 0 —
    total 1 barrier, vs 8 barriers in a full tree reduction. Phase 2
    every thread re-reads its slice, normalises, multiplies by weight,
    and adds into the residual buffer (bound as read_write) in place.

    Dispatch: `(1, 1, 1)` workgroups × 256 threads. -/
def fusedPerLayerPostKernel (hiddenSize : Nat) (eps : Float) : ShaderM Unit := do
  let wgSize := 256
  let numSubgroups := wgSize / 32
  let lid ← ShaderM.localId
  let tid := Exp.vec3X lid

  let _proj ← ShaderM.declareInputBuffer "proj" (.array (.scalar .f32) hiddenSize)
  let _weight ← ShaderM.declareInputBuffer "weight" (.array (.scalar .f32) hiddenSize)
  let _residual ← ShaderM.declareOutputBuffer "residual" (.array (.scalar .f32) hiddenSize)

  -- Shared workspace: `numSubgroups` per-subgroup partials + 1 slot
  -- for the final broadcast `invRms`.
  ShaderM.sharedNamed "shared_sg" (.array (.scalar .f32) (numSubgroups + 1))

  -- Phase 1: per-thread sum of squares over stride-wgSize slice.
  ShaderM.varNamed "partialSq" (.scalar .f32) (Exp.litF32 0.0)
  let partialSq : Exp (.scalar .f32) := Exp.var "partialSq"
  ShaderM.loop tid (Exp.litU32 hiddenSize) (Exp.litU32 wgSize) fun d => do
    let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := hiddenSize) "proj" d
    ShaderM.assign "partialSq" (Exp.add partialSq (Exp.mul v v))

  -- Intra-subgroup reduction: every lane of each subgroup now holds
  -- the same 32-way sum.
  ShaderM.varNamed "sgSum" (.scalar .f32) (Exp.subgroupAdd partialSq)
  let sgSum : Exp (.scalar .f32) := Exp.var "sgSum"

  -- Lane 0 of each subgroup writes its partial into shared memory.
  let subgroupId := Exp.div tid (Exp.litU32 32)
  let laneId := Exp.sub tid (Exp.mul subgroupId (Exp.litU32 32))
  ShaderM.if_ (Exp.eq laneId (Exp.litU32 0)) (do
    ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_sg" subgroupId sgSum
  ) (pure ())
  ShaderM.barrier

  -- Thread 0 sums the per-subgroup partials, computes invRms, and
  -- stashes it in shared_sg[numSubgroups] for broadcast.
  ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
    ShaderM.varNamed "totalSq" (.scalar .f32) (Exp.litF32 0.0)
    let totalSq : Exp (.scalar .f32) := Exp.var "totalSq"
    for sg in [0:numSubgroups] do
      let part ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := numSubgroups + 1) "shared_sg" (Exp.litU32 sg)
      ShaderM.assign "totalSq" (Exp.add totalSq part)
    let invRms := Exp.div (Exp.litF32 1.0)
      (Exp.sqrt (Exp.add (Exp.div totalSq (Exp.litF32 hiddenSize.toFloat))
                         (Exp.litF32 eps)))
    ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_sg" (Exp.litU32 numSubgroups) invRms
  ) (pure ())
  ShaderM.barrier

  -- Every thread grabs invRms from the shared slot.
  ShaderM.varNamed "invRms" (.scalar .f32)
    (← ShaderM.readWorkgroup (ty := .scalar .f32) (n := numSubgroups + 1)
          "shared_sg" (Exp.litU32 numSubgroups))
  let invRms : Exp (.scalar .f32) := Exp.var "invRms"

  -- Phase 2: normalise, apply weight, add into residual in place.
  ShaderM.loop tid (Exp.litU32 hiddenSize) (Exp.litU32 wgSize) fun d => do
    let pVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := hiddenSize) "proj" d
    let wVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := hiddenSize) "weight" d
    let rVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := hiddenSize) "residual" d
    let normed := Exp.mul (Exp.mul pVal invRms) wVal
    ShaderM.writeBuffer (ty := .scalar .f32) "residual" d (Exp.add rVal normed)

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

/-! ## Bare RMSNorm Kernels (no learned weights) -/

/-- Per-head bare RMSNorm: normalize each head independently (no learned weights).
    Used for V-norm in Gemma 4: each KV head's `headDim` elements are normalized
    by their own RMS. Total input size is `numHeads * headDim`.

    One workgroup per head (numHeads workgroups). Within each workgroup,
    threads cooperate on a tree reduction over `headDim` elements. -/
def perHeadBareRMSNormKernel (numHeads headDim : Nat) (eps : Float) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let headIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid

  let totalElements := numHeads * headDim

  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) totalElements)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) totalElements)

  ShaderM.sharedNamed "shared_sum" (.array (.scalar .f32) workgroupSize)

  let headBase := Exp.mul headIdx (Exp.litU32 headDim)

  -- Step 1: sum of squares for this head
  ShaderM.varNamed "local_sum" (.scalar .f32) (Exp.litF32 0.0)
  let localSum : Exp (.scalar .f32) := Exp.var "local_sum"

  ShaderM.loop tid (Exp.litU32 headDim) (Exp.litU32 workgroupSize) fun i => do
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalElements) "input" (Exp.add headBase i)
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

  -- Step 2: normalize (no weight multiplication)
  let sumSq ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_sum" (Exp.litU32 0)
  let rms := Exp.inverseSqrt (Exp.add (Exp.div sumSq (Exp.litF32 headDim.toFloat)) (Exp.litF32 eps))

  ShaderM.loop tid (Exp.litU32 headDim) (Exp.litU32 workgroupSize) fun i => do
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalElements) "input" (Exp.add headBase i)
    ShaderM.writeBuffer (ty := .scalar .f32) "output" (Exp.add headBase i) (Exp.mul val rms)

/-- Legacy: bare RMSNorm over a single vector of size `dim`. Kept for backward
    compatibility; for Gemma 4 V-norm use perHeadBareRMSNormKernel instead. -/
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

/-! ## Per-Layer Embedding Helpers -/

/-- Scaled add: output = (a + b) * scale (avoids aliasing if a/b/output distinct) -/
def scaledAddKernel (size : Nat) (scale : Float) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid
  let _a ← ShaderM.declareInputBuffer "a" (.array (.scalar .f32) size)
  let _b ← ShaderM.declareInputBuffer "b" (.array (.scalar .f32) size)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) size)
  ShaderM.if_ (Exp.lt idx (Exp.litU32 size)) (do
    let aVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := size) "a" idx
    let bVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := size) "b" idx
    let result := Exp.mul (Exp.add aVal bVal) (Exp.litF32 scale)
    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx result
  ) (pure ())

/-- Per-layer RMSNorm: normalize each `chunkDim`-sized chunk independently with shared weights.
    Used for per_layer_proj_norm which normalizes [embdPerLayer * numLayers] in chunks of embdPerLayer.

    One workgroup per chunk. Each workgroup computes RMS over its own chunk.

    @param chunkDim Size of each chunk (e.g. embdPerLayer = 256)
    @param numChunks Number of chunks (e.g. numLayers = 42)
    @param eps RMSNorm epsilon
-/
def chunkedRMSNormKernel (chunkDim numChunks : Nat) (eps : Float) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let chunkIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid

  let totalElements := chunkDim * numChunks
  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) totalElements)
  let _weight ← ShaderM.declareInputBuffer "weight" (.array (.scalar .f32) chunkDim)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) totalElements)

  ShaderM.sharedNamed "shared_sum" (.array (.scalar .f32) workgroupSize)

  let chunkBase := Exp.mul chunkIdx (Exp.litU32 chunkDim)

  -- Step 1: sum of squares
  ShaderM.varNamed "local_sum" (.scalar .f32) (Exp.litF32 0.0)
  let localSum : Exp (.scalar .f32) := Exp.var "local_sum"

  ShaderM.loop tid (Exp.litU32 chunkDim) (Exp.litU32 workgroupSize) fun i => do
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalElements) "input" (Exp.add chunkBase i)
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

  -- Step 2: normalize and apply weight
  let sumSq ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_sum" (Exp.litU32 0)
  let rms := Exp.inverseSqrt (Exp.add (Exp.div sumSq (Exp.litF32 chunkDim.toFloat)) (Exp.litF32 eps))

  ShaderM.loop tid (Exp.litU32 chunkDim) (Exp.litU32 workgroupSize) fun i => do
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalElements) "input" (Exp.add chunkBase i)
    let w ← ShaderM.readBuffer (ty := .scalar .f32) (n := chunkDim) "weight" i
    let result := Exp.mul (Exp.mul val rms) w
    ShaderM.writeBuffer (ty := .scalar .f32) "output" (Exp.add chunkBase i) result

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

/-! ## Layer Structures -/

/-- Gemma 4 attention layer (single layer) -/
structure Gemma4Attention (BufT CacheT : Type) where
  wQ : Linear.LinearLayer BufT CacheT         -- Q projection [hiddenSize → numHeads * headDim]
  wK : Linear.LinearLayer BufT CacheT         -- K projection [hiddenSize → numKVHeads * headDim]
  wV : Linear.LinearLayer BufT CacheT         -- V projection [hiddenSize → numKVHeads * headDim]
  wO : Linear.LinearLayer BufT CacheT         -- Output projection [numHeads * headDim → hiddenSize]
  qNormWeight : BufT            -- Per-head Q norm [headDim]
  kNormWeight : BufT            -- Per-head K norm [headDim]

/-- Gemma 4 dense FFN layer -/
structure Gemma4FFN (BufT CacheT : Type) where
  gate : Linear.LinearLayer BufT CacheT
  up : Linear.LinearLayer BufT CacheT
  down : Linear.LinearLayer BufT CacheT
  fusedGateUpPrepared : IO.Ref (Option CacheT)

/-- Gemma 4 transformer block (single layer) -/
structure Gemma4Block (BufT CacheT : Type) where
  layerIdx : Nat
  layerType : LayerType
  -- Norms
  attnNorm : RMSNorm.RMSNorm BufT CacheT
  postAttnNorm : RMSNorm.RMSNorm BufT CacheT
  ffnNorm : RMSNorm.RMSNorm BufT CacheT
  postFFNNorm : RMSNorm.RMSNorm BufT CacheT
  -- Attention
  attention : Gemma4Attention BufT CacheT
  -- FFN (shared/dense expert)
  ffn : Gemma4FFN BufT CacheT
  -- MoE (optional: present only for MoE layers)
  isMoE : Bool
  moeRouterWeight : Option BufT
  moeRouterScale : Option BufT
  moeGateUpExps : Option BufT
  moeDownExps : Option BufT
  moePreNorm2 : Option (RMSNorm.RMSNorm BufT CacheT)
  moePostNorm1 : Option (RMSNorm.RMSNorm BufT CacheT)
  moePostNorm2 : Option (RMSNorm.RMSNorm BufT CacheT)
  -- Optional: RoPE frequency factors (full attention layers only)
  ropeFreqFactors : Option BufT
  -- Optional: layer output scale
  outScale : Option BufT

/-- Per-layer embedding weights for a single block -/
structure Gemma4PerLayerEmbd (BufT CacheT : Type) where
  inpGate : Linear.LinearLayer BufT CacheT
  proj : Linear.LinearLayer BufT CacheT
  postNorm : RMSNorm.RMSNorm BufT CacheT

/-- Embedding format for token embedding table -/
inductive EmbdFormat where
  | F32   -- Pre-dequantized F32 (via Embedding.forward)
  | F16   -- F16 (via Embedding.forward with f16Table)
  | Q6_K  -- Q6_K packed (GPU dequant lookup + LM head matmul)
  | Q4_K  -- Q4_K packed
  deriving Repr, BEq, Inhabited

/-- Complete Gemma 4 model -/
structure Gemma4Model (BufT CacheT : Type) where
  config : Config
  embedding : Embedding.Embedding BufT
  embdFormat : EmbdFormat
  blocks : Array (Gemma4Block BufT CacheT)
  finalNorm : RMSNorm.RMSNorm BufT CacheT
  outputWeight : BufT
  perLayerEmbdTableCPU : Option ByteArray
  perLayerEmbdRowBytes : Nat
  perLayerModelProj : Option BufT
  perLayerProjNorm : Option (RMSNorm.RMSNorm BufT CacheT)
  perLayerBlocks : Array (Option (Gemma4PerLayerEmbd BufT CacheT))

/-! ## Helper: Create GPU Buffer from ByteArray -/

private def uploadBuffer [GPUBackend β] (ctx : β) (data : ByteArray) : IO (GPUBackend.Buf β) := do
  let bufSize := if data.size == 0 then 4 else data.size
  let buf ← GPUBackend.allocBuffer ctx bufSize.toUSize
  if data.size > 0 then
    GPUBackend.writeBuffer ctx buf data
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
  let (_, data) ← match Hesper.GGUF.Loader.getTensorData gguf name with
    | .ok r => pure r
    | .error e => throw $ IO.userError e
  let weightBuf ← uploadBuffer ctx data
  let prepared ← GPUBackend.newCacheRef (β := β)
  let splitKBuf ← IO.mkRef none
  let splitKPartialPrepared ← GPUBackend.newCacheRef (β := β)
  let splitKReducePrepared ← GPUBackend.newCacheRef (β := β)
  return {
    config := { inDim, outDim }
    weightBuf
    quantFormat
    prepared
    splitKBuf
    splitKPartialPrepared
    splitKReducePrepared
  }

/-- Load Gemma 4 model from GGUF file -/
def Gemma4Model.fromGGUF [GPUBackend β] (ctx : β) (ggufPath : String)
    (configOverride : Option Config := none) : IO (Gemma4Model (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)) := do
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
      let (_, data) ← match Hesper.GGUF.Loader.getTensorData gguf "token_embd.weight" with
        | .ok r => pure r
        | .error e => throw $ IO.userError e
      let buf ← uploadBuffer ctx data
      pure ({ config := embConfig, embeddingTable := buf, f16Table := none : Embedding.Embedding (GPUBackend.Buf β) }, EmbdFormat.Q6_K)
    | .Q4_K =>
      IO.println "  Using Q4_K embeddings (GPU on-the-fly dequant)"
      let (_, data) ← match Hesper.GGUF.Loader.getTensorData gguf "token_embd.weight" with
        | .ok r => pure r
        | .error e => throw $ IO.userError e
      let buf ← uploadBuffer ctx data
      pure ({ config := embConfig, embeddingTable := buf, f16Table := none : Embedding.Embedding (GPUBackend.Buf β) }, EmbdFormat.Q4_K)
    | other =>
      IO.println s!"  Embedding type: {other} — loading as raw bytes (F32 fallback)"
      let (_, data) ← match Hesper.GGUF.Loader.getTensorData gguf "token_embd.weight" with
        | .ok r => pure r
        | .error e => throw $ IO.userError e
      let buf ← uploadBuffer ctx data
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

    let attention : Gemma4Attention (GPUBackend.Buf β) (GPUBackend.CachedDispatch β) := {
      wQ, wK, wV, wO
      qNormWeight := qNormBuf
      kNormWeight := kNormBuf
    }

    -- Load FFN projections (Q4_K)
    let ffnGate ← loadLinear ctx gguf s!"blk.{li}.ffn_gate.weight" cfg.hiddenSize cfg.intermediateSize
    let ffnUp ← loadLinear ctx gguf s!"blk.{li}.ffn_up.weight" cfg.hiddenSize cfg.intermediateSize
    let ffnDown ← loadLinear ctx gguf s!"blk.{li}.ffn_down.weight" cfg.intermediateSize cfg.hiddenSize

    let fusedGateUpPrepared ← GPUBackend.newCacheRef (β := β)
    let ffn : Gemma4FFN (GPUBackend.Buf β) (GPUBackend.CachedDispatch β) :=
      { gate := ffnGate, up := ffnUp, down := ffnDown, fusedGateUpPrepared }

    -- Load optional RoPE frequency factors
    -- In the E4B model, rope_freqs is a global tensor, not per-layer
    let ropeFreqFactors ← if cfg.isFullAttention li then
      match Hesper.GGUF.Loader.getTensorData gguf "rope_freqs.weight" with
      | .ok (_, data) =>
        let buf ← uploadBuffer ctx data
        pure (some buf)
      | .error _ =>
        -- Try per-layer
        match Hesper.GGUF.Loader.getTensorData gguf s!"blk.{li}.rope_freqs.weight" with
        | .ok (_, data) =>
          let buf ← uploadBuffer ctx data
          pure (some buf)
        | .error _ => pure none
    else pure none

    -- Load optional layer output scale
    let outScale ← match Hesper.GGUF.Loader.getTensorData gguf s!"blk.{li}.layer_output_scale.weight" with
      | .ok (_, data) =>
        let buf ← uploadBuffer ctx data
        pure (some buf)
      | .error _ => pure none

    -- Load MoE weights (if present for this layer)
    let isMoE := match Hesper.GGUF.Loader.findTensor gguf s!"blk.{li}.ffn_gate_inp.weight" with
      | .ok _ => true | .error _ => false
    let (moeRouterWeight, moeRouterScale, moeGateUpExps, moeDownExps, moePreNorm2, moePostNorm1, moePostNorm2) ←
      if isMoE then do
        IO.println s!"    Layer {li}: MoE layer"
        let routerW ← match Hesper.GGUF.Loader.getTensorData gguf s!"blk.{li}.ffn_gate_inp.weight" with
          | .ok (_, data) => pure (some (← uploadBuffer ctx data))
          | .error _ => pure none
        let routerS ← match Hesper.GGUF.Loader.getTensorData gguf s!"blk.{li}.ffn_gate_inp.scale" with
          | .ok (_, data) => pure (some (← uploadBuffer ctx data))
          | .error _ => pure none
        let gateUpE ← match Hesper.GGUF.Loader.getTensorData gguf s!"blk.{li}.ffn_gate_up_exps.weight" with
          | .ok (_, data) => pure (some (← uploadBuffer ctx data))
          | .error _ => pure none
        let downE ← match Hesper.GGUF.Loader.getTensorData gguf s!"blk.{li}.ffn_down_exps.weight" with
          | .ok (_, data) => pure (some (← uploadBuffer ctx data))
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
  let outputWeight ← match Hesper.GGUF.Loader.getTensorData gguf "output.weight" with
    | .ok (_, data) =>
      IO.println "  Using separate LM head weights"
      uploadBuffer ctx data
    | .error _ =>
      IO.println "  Using weight-tied LM head (reusing embedding)"
      pure embedding.embeddingTable

  -- Step 7: Load per-layer embeddings (optional)
  -- per_layer_token_embd kept on CPU (table > WebGPU 256 MB binding limit; we
  -- dequant just the input token's row at inference time)
  let (perLayerEmbdTableCPU, perLayerEmbdRowBytes, perLayerModelProj, perLayerProjNorm) ←
    if cfg.hasPerLayerEmbeddings then do
      IO.println "[Gemma4] Loading per-layer embeddings..."
      let blocksPerRow := (cfg.embdPerLayer * cfg.numHiddenLayers) / 256
      let rowBytes := blocksPerRow * 210  -- Q6_K block size
      let tableData ← match Hesper.GGUF.Loader.getTensorData gguf "per_layer_token_embd.weight" with
        | .ok (_, data) =>
          IO.println s!"  per_layer_token_embd: {data.size} bytes (kept on CPU, row size {rowBytes} bytes)"
          pure (some data)
        | .error _ => pure none
      let proj ← match Hesper.GGUF.Loader.getTensorData gguf "per_layer_model_proj.weight" with
        | .ok (_, data) => pure (some (← uploadBuffer ctx data))
        | .error _ => pure none
      let projNorm ← match Hesper.GGUF.Loader.findTensor gguf "per_layer_proj_norm.weight" with
        | .ok _ =>
          let d ← Hesper.GGUF.Loader.extractFloat32Tensor gguf "per_layer_proj_norm.weight"
          let plNormConfig : RMSNorm.Config := { dim := cfg.embdPerLayer, eps := cfg.rmsNormEps }
          pure (some (← RMSNorm.create ctx plNormConfig d))
        | .error _ => pure none
      pure (tableData, rowBytes, proj, projNorm)
    else
      pure (none, 0, none, none)

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
    perLayerEmbdTableCPU
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

/-! ## KV Cache State -/

/-- Per-layer KV cache for Gemma 4 -/
structure Gemma4KVCache (BufT : Type) where
  kBuf : BufT    -- [numKVHeads, maxSeqLen, headDim]
  vBuf : BufT    -- [numKVHeads, maxSeqLen, headDim]

/-- Full inference state -/
structure InferenceState (BufT CacheT : Type) where
  kvCaches : Array (Gemma4KVCache BufT)
  buf1 : BufT          -- [hiddenSize] ping-pong
  buf2 : BufT          -- [hiddenSize] ping-pong
  qBuf : BufT          -- [numHeads * headDim] Q projection output
  kBuf : BufT          -- [numKVHeads * headDim] K projection output
  vBuf : BufT          -- [numKVHeads * headDim] V projection output
  attnOutBuf : BufT    -- [numHeads * headDim] attention output
  gateBuf : BufT       -- [intermediateSize] FFN gate output
  upBuf : BufT         -- [intermediateSize] FFN up output
  geluBuf : BufT       -- [intermediateSize] GELU*up output
  ffnOutBuf : BufT     -- [hiddenSize] FFN down output
  normedBuf : BufT     -- [hiddenSize] normalized output
  attnResidualBuf : BufT  -- [hiddenSize] attn output + residual (between attn and FFN)
  qBuf2 : BufT            -- [maxQDim] alternate Q buffer (for in-place ops)
  kBuf2 : BufT            -- [maxKVDim] alternate K buffer
  vBuf2 : BufT            -- [maxKVDim] alternate V buffer
  normedBuf2 : BufT       -- [hiddenSize] alternate normed buffer
  logitsBuf : BufT     -- [vocabSize]
  logitsBuf2 : BufT    -- [vocabSize] scratch for logit softcap (no aliasing)
  tokenBuf : BufT      -- [1] u32 for single token
  paramsBuf : BufT     -- [2] u32: (pos, cacheLen) for RoPE
  -- MoE buffers
  moeRouterOutBuf : BufT    -- [hiddenSize] router preprocessed input
  moeLogitsBuf : BufT       -- [numExperts] router logits
  moeIndicesBuf : BufT      -- [numExpertsUsed] selected expert indices
  moeWeightsBuf : BufT      -- [numExpertsUsed] expert weights
  moeExpertOutBuf : BufT    -- [hiddenSize] combined expert output
  moeExpertGateBuf : BufT   -- [expertFFSize] expert gate projection output
  moeExpertUpBuf : BufT     -- [expertFFSize] expert up projection output
  moeExpertGeluBuf : BufT   -- [expertFFSize] expert GELU*up output
  moeExpertDownBuf : BufT   -- [hiddenSize] single expert down output
  moeNormedBuf : BufT       -- [hiddenSize] pre_norm_2 output for routed experts
  -- Per-layer embedding buffers
  plGateBuf : BufT          -- [embdPerLayer] per-layer gate output
  plProjBuf : BufT          -- [hiddenSize] per-layer projected output
  -- Per-layer input precomputation (computed once per token, used by all layers)
  plTokenSelected : BufT    -- [embdPerLayer * numLayers] tok_embd_per_layer[token] dequantized
  plModelProj : BufT        -- [embdPerLayer * numLayers] per_layer_model_proj @ scaled_embed
  plInputAll : BufT         -- [embdPerLayer * numLayers] final per-layer input (sum, normed, scaled)
  -- Partial buffer for tiled (split-K) flash attention. Pre-allocated
  -- at createInferenceState with size for the maximum tile count.
  flashPartialBuf : BufT
  -- Small GPU-side scratch for the raw Q6_K bytes of one per-layer
  -- embedding row (~33 KB for Gemma 4 e4b). The full per-layer
  -- embedding table lives on CPU (> WebGPU single-buffer limit); at
  -- decode time we slice the needed row out of the CPU ByteArray,
  -- upload it here, and dequant on-GPU via `q6kSingleRowDequantScaleKernel`.
  -- Sized to the maximum row bytes seen at load time; left as a small
  -- placeholder when per-layer embeddings are absent.
  plRawRowBuf : BufT
  -- Optional: pre-softcap logits buffer for TTT surprise sensor.
  -- When `some`, forwardSingleToken copies logitsBuf here BEFORE
  -- applying logit softcap. When `none` (default), no copy is done
  -- and there is zero performance impact.
  preSoftcapBuf : Option BufT := none
  argmaxBuf : BufT              -- [1] u32 for GPU-side argmax result

/-- Dynamic cache ref store. Lazily creates IO.Ref per unique cacheKey. -/
structure KernelCacheRefs (CacheT : Type) where
  store : IO.Ref (Array (UInt64 × IO.Ref (Option CacheT)))

def KernelCacheRefs.getRef (kcr : KernelCacheRefs CacheT) (key : UInt64) : IO (IO.Ref (Option CacheT)) := do
  let arr ← kcr.store.get
  match arr.find? (fun (k, _) => k == key) with
  | some (_, r) => pure r
  | none =>
    let r ← IO.mkRef none
    kcr.store.modify (·.push (key, r))
    pure r

def createKernelCacheRefs [GPUBackend β] : IO (KernelCacheRefs (GPUBackend.CachedDispatch β)) := do
  pure { store := ← IO.mkRef #[] }

/-- Create inference state with pre-allocated buffers -/
def createInferenceState [GPUBackend β] (ctx : β) (cfg : Config) : IO (InferenceState (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)) := do
  let mkBuf := fun (size : Nat) => GPUBackend.allocBuffer ctx (size * 4).toUSize
  let maxHeadDim := max cfg.headDimFull cfg.headDimSWA
  let maxQDim := cfg.numAttentionHeads * maxHeadDim
  let maxKVDim := (max cfg.numKeyValueHeadsFull cfg.numKeyValueHeadsSWA) * maxHeadDim

  -- Create per-layer KV caches
  let mut kvCaches : Array (Gemma4KVCache (GPUBackend.Buf β)) := #[]
  for li in [0:cfg.numHiddenLayers] do
    let numKVHeads := cfg.numKVHeads li
    let headDim := cfg.headDim li
    let cacheSize := numKVHeads * cfg.maxSeqLen * headDim
    let kBuf ← mkBuf cacheSize
    let vBuf ← mkBuf cacheSize
    kvCaches := kvCaches.push ({ kBuf, vBuf } : Gemma4KVCache (GPUBackend.Buf β))

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
    attnResidualBuf := ← mkBuf cfg.hiddenSize
    qBuf2 := ← mkBuf maxQDim
    kBuf2 := ← mkBuf maxKVDim
    vBuf2 := ← mkBuf maxKVDim
    normedBuf2 := ← mkBuf cfg.hiddenSize
    logitsBuf := ← mkBuf cfg.vocabSize
    logitsBuf2 := ← mkBuf cfg.vocabSize
    tokenBuf := ← GPUBackend.allocBuffer ctx (4 : USize)
    paramsBuf := ← GPUBackend.allocBuffer ctx (8 : USize)
    moeRouterOutBuf := ← mkBuf cfg.hiddenSize
    moeLogitsBuf := ← mkBuf (max cfg.numExperts 1)
    moeIndicesBuf := ← GPUBackend.allocBuffer ctx (max cfg.numExpertsUsed 1 * 4).toUSize
    moeWeightsBuf := ← mkBuf (max cfg.numExpertsUsed 1)
    moeExpertOutBuf := ← mkBuf cfg.hiddenSize
    moeExpertGateBuf := ← mkBuf (max cfg.expertFFSize 1)
    moeExpertUpBuf := ← mkBuf (max cfg.expertFFSize 1)
    moeExpertGeluBuf := ← mkBuf (max cfg.expertFFSize 1)
    moeExpertDownBuf := ← mkBuf cfg.hiddenSize
    moeNormedBuf := ← mkBuf cfg.hiddenSize
    plGateBuf := ← mkBuf (max cfg.embdPerLayer 1)
    plProjBuf := ← mkBuf cfg.hiddenSize
    plTokenSelected := ← mkBuf (max (cfg.embdPerLayer * cfg.numHiddenLayers) 1)
    plModelProj := ← mkBuf (max (cfg.embdPerLayer * cfg.numHiddenLayers) 1)
    plInputAll := ← mkBuf (max (cfg.embdPerLayer * cfg.numHiddenLayers) 1)
    flashPartialBuf := ← FlashAttention.createFlashPartialBuffer ctx
                          cfg.numAttentionHeads cfg.maxSeqLen (max cfg.headDimFull cfg.headDimSWA)
    plRawRowBuf := ← do
      -- Raw Q6_K bytes for one per-layer-embd row:
      --   blocksPerRow = ceil((embdPerLayer * numLayers) / 256)
      --   rowBytes     = blocksPerRow * 210
      let totalPL := cfg.embdPerLayer * cfg.numHiddenLayers
      let blocksPerRow := (totalPL + 255) / 256
      let rowBytes := blocksPerRow * 210
      GPUBackend.allocBuffer ctx (max rowBytes 4).toUSize
    argmaxBuf := ← GPUBackend.allocBuffer ctx (4 : USize)
  }

/-! ## Single-Token Forward Pass -/

/-- Run single-token forward pass through one transformer block.

    Flow (from gemma4-iswa.cpp):
    1. attnNorm(input) → Q/K/V projections → Q-norm, K-norm → attention → postAttnNorm → + residual
    2. ffnNorm(attn_out) → GeGLU FFN → postFFNNorm → + residual
-/
def forwardBlock [GPUBackend β] (ctx : β)
    (block : Gemma4Block (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)) (cfg : Config)
    (inputBuf outputBuf : GPUBackend.Buf β)
    (state : InferenceState (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)) (pos : Nat)
    (kcr : Option (KernelCacheRefs (GPUBackend.CachedDispatch β)) := none)
    (perLayerEmbd : Option (Gemma4PerLayerEmbd (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)) := none)
    (perLayerInput : Option (GPUBackend.Buf β) := none) : IO Unit := do
  let li := block.layerIdx
  let headDim := cfg.headDim li

  -- Helper: cached execute with named cache key. On 2nd+ call for same
  -- kernel, cacheRef hit skips generatePTX entirely (90-330μs → 0μs).
  let ce := fun (name : String) (shader : ShaderM Unit)
      (namedBufs : List (String × GPUBackend.Buf β)) (config : Hesper.ExecConfig) => do
    -- Key includes name + config (numWorkgroups, workgroupSize) to distinguish
    -- same-named kernels with different parameters (e.g., full vs SWA attention)
    let key := hash (name, config.numWorkgroups, config.workgroupSize.x, config.workgroupSize.y)
    match kcr with
    | some k =>
      let ref ← k.getRef key
      GPUBackend.executeWithConfigCached ctx shader namedBufs config key ref
    | none => GPUBackend.execute ctx shader namedBufs config

  -- Step 1: Attention pre-norm
  Hesper.WGSL.Execute.withSection "attnNorm" do
    RMSNorm.forward ctx block.attnNorm inputBuf state.normedBuf

  -- Step 2: Q/K/V projections
  Hesper.WGSL.Execute.withSection "qkvProj" do
    Linear.LinearLayer.forward ctx block.attention.wQ state.normedBuf state.qBuf
    if cfg.hasKV li then
      Linear.LinearLayer.forward ctx block.attention.wK state.normedBuf state.kBuf
      Linear.LinearLayer.forward ctx block.attention.wV state.normedBuf state.vBuf

  -- Step 3: Q-norm, K-norm (per-head RMSNorm)
  let numHeads := cfg.numAttentionHeads
  let numKVHeads := cfg.numKVHeads li
  let wgSize := min headDim 256
  let mkNormConfig := fun (nHeads : Nat) => ({
    numWorkgroups := (nHeads, 1, 1)
    workgroupSize := { x := wgSize, y := 1, z := 1 }
  } : Hesper.ExecConfig)
  let isFull := cfg.isFullAttention li
  Hesper.WGSL.Execute.withSection "qkvNorm" do
    -- Q-norm: qBuf → qBuf2
    ce (if isFull then "qNormFull" else "qNormSWA")
      (perHeadRMSNormKernel numHeads headDim cfg.rmsNormEps)
      [("input", state.qBuf), ("weight", block.attention.qNormWeight), ("output", state.qBuf2)]
      (mkNormConfig numHeads)

    if cfg.hasKV li then
      -- K-norm: kBuf → kBuf2
      ce (if isFull then "kNormFull" else "kNormSWA")
        (perHeadRMSNormKernel numKVHeads headDim cfg.rmsNormEps)
        [("input", state.kBuf), ("weight", block.attention.kNormWeight), ("output", state.kBuf2)]
        (mkNormConfig numKVHeads)

      -- V-norm: bare per-head RMSNorm on V (no learned weights)
      ce (if isFull then "vNormFull" else "vNormSWA")
        (perHeadBareRMSNormKernel numKVHeads headDim cfg.rmsNormEps)
        [("input", state.vBuf), ("output", state.vBuf2)]
        { numWorkgroups := (numKVHeads, 1, 1), workgroupSize := { x := min headDim 256, y := 1, z := 1 } : Hesper.ExecConfig }

  -- Step 4: RoPE on Q and K
  -- Upload position to params buffer
  let posBytes := Hesper.WebGPU.BufferOps.uint32ToBytes pos.toUInt32
  GPUBackend.writeBufferOffset ctx state.paramsBuf 0 posBytes

  Hesper.WGSL.Execute.withSection "rope" do
    -- RoPE on Q: qBuf2 → qBuf
    match block.ropeFreqFactors with
    | some freqFactors =>
      -- Full attention: RoPE with frequency factors
      ce "k1"
        (ropeWithFreqFactorsKernel headDim numHeads cfg.ropeTheta)
        [("input", state.qBuf2), ("output", state.qBuf), ("params", state.paramsBuf), ("freq_factors", freqFactors)]
        (.dispatch1D (numHeads * headDim / 2))
    | none =>
      -- SWA: standard RoPE (use existing dynamic kernel)
      let ropeConfig : RoPE.Config := { dim := numHeads * headDim, maxSeqLen := cfg.maxSeqLen, base := cfg.ropeTheta }
      ce "k2"
        (RoPE.ropeKernelDynamic ropeConfig 1 1 numHeads headDim)
        [("input", state.qBuf2), ("output", state.qBuf), ("params", state.paramsBuf)]
        (.dispatch1D (numHeads * headDim / 2))

    if cfg.hasKV li then
      -- RoPE on K: kBuf2 → kBuf
      -- Must mirror Q's RoPE branch: full attention layers use freq_factors,
      -- otherwise Q and K end up rotated with different frequencies (mismatch).
      match block.ropeFreqFactors with
      | some freqFactors =>
        ce "k3"
          (ropeWithFreqFactorsKernel headDim numKVHeads cfg.ropeTheta)
          [("input", state.kBuf2), ("output", state.kBuf), ("params", state.paramsBuf), ("freq_factors", freqFactors)]
          (.dispatch1D (numKVHeads * headDim / 2))
      | none =>
        let ropeConfig : RoPE.Config := { dim := numKVHeads * headDim, maxSeqLen := cfg.maxSeqLen, base := cfg.ropeTheta }
        ce "k4"
          (RoPE.ropeKernelDynamic ropeConfig 1 1 numKVHeads headDim)
          [("input", state.kBuf2), ("output", state.kBuf), ("params", state.paramsBuf)]
          (.dispatch1D (numKVHeads * headDim / 2))

  -- Step 5: Write K/V to cache and compute flash attention
  -- KV-shared layers reuse an earlier layer's cache (see Config.kvCacheLayer).
  let kvLi := cfg.kvCacheLayer li
  if h : kvLi < state.kvCaches.size then
    let kvCache := state.kvCaches[kvLi]
    let kvDim := numKVHeads * headDim
    let cacheLen := pos + 1  -- number of cached positions including current

    -- Write K and V to cache at current position (fused kernel)
    -- K is now in kBuf (after RoPE), V is in vBuf2 (after V-norm)
    if cfg.hasKV li then
      Hesper.WGSL.Execute.withSection "kvWrite" do
        ce "k5"
          (Attention.fusedCacheWriteKVKernel numKVHeads cfg.maxSeqLen headDim kvDim)
          [("new_k", state.kBuf), ("new_v", state.vBuf2),
           ("k_cache", kvCache.kBuf), ("v_cache", kvCache.vBuf),
           ("params", state.paramsBuf)]
          (.dispatch1D kvDim)

    -- Flash attention: Q @ K_cache^T → softmax → @ V_cache → output
    -- Gemma 4 uses hparams.f_attention_scale = 1.0 (NOT the usual 1/sqrt(headDim)),
    -- because the Q-norm RMSNorm already normalizes each head, so the dot product
    -- magnitudes are bounded without the 1/sqrt(headDim) temperature.
    -- See llama.cpp llama-model.cpp:1272 and gemma4-iswa.cpp:94.
    let scale : Float := 1.0
    Hesper.WGSL.Execute.withSection "flashAttn" do
      match block.layerType with
      | .swa =>
        -- Sliding window attention: only attend within windowSize.
        -- SWA layers have short effective cacheLen (≤ windowSize), so
        -- the 1-WG-per-head dispatch is fine — no split-K needed.
        ce "k6"
          (FlashAttention.flashAttentionSWAKernel numHeads numKVHeads cfg.maxSeqLen headDim cacheLen cfg.slidingWindowSize pos scale)
          [("q", state.qBuf), ("k_cache", kvCache.kBuf), ("v_cache", kvCache.vBuf), ("output", state.attnOutBuf)]
          ({ numWorkgroups := (numHeads, 1, 1) : Hesper.ExecConfig })
      | .full =>
        -- Full attention: use tiled (split-K) flash attention when
        -- cacheLen is large enough to benefit from multiple tiles.
        -- For small cacheLen (≤ tileSize=32), the single-WG-per-head
        -- kernel is faster (1 dispatch vs 2 + intermediate buffer).
        if cacheLen > 32 then
          FlashAttention.executeFlashAttentionTiled ctx
            state.qBuf kvCache.kBuf kvCache.vBuf state.attnOutBuf
            numHeads numKVHeads cfg.maxSeqLen headDim cacheLen scale
            (partialBuf := some state.flashPartialBuf)
        else
          ce "k7"
            (FlashAttention.flashAttentionDynamicKernel numHeads numKVHeads cfg.maxSeqLen headDim cacheLen scale)
            [("q", state.qBuf), ("k_cache", kvCache.kBuf), ("v_cache", kvCache.vBuf), ("output", state.attnOutBuf)]
            ({ numWorkgroups := (numHeads, 1, 1) : Hesper.ExecConfig })

    -- Output projection: attnOut [numHeads * headDim] → normedBuf [hiddenSize]
    Hesper.WGSL.Execute.withSection "oProj" do
      Linear.LinearLayer.forward ctx block.attention.wO state.attnOutBuf state.normedBuf
  else
    -- Fallback: skip attention (shouldn't happen)
    Linear.LinearLayer.forward ctx block.attention.wO state.qBuf state.normedBuf

  -- Step 6: Post-attention norm + residual
  -- normedBuf → normedBuf2 (avoid aliasing)
  Hesper.WGSL.Execute.withSection "postAttnNorm" do
    RMSNorm.forward ctx block.postAttnNorm state.normedBuf state.normedBuf2
    ce "residAdd_postAttn"
      (residualAddKernel cfg.hiddenSize)
      [("a", state.normedBuf2), ("b", inputBuf), ("output", state.attnResidualBuf)]
      (.dispatch1D cfg.hiddenSize)

  -- Step 7: FFN (dense or MoE)
  if block.isMoE then do
    -- MoE layer (from gemma4-iswa.cpp:117-169):
    -- 1. Shared expert: ffn_norm → GeGLU FFN → post_norm_1
    RMSNorm.forward ctx block.ffnNorm state.attnResidualBuf state.normedBuf
    Linear.LinearLayer.forward ctx block.ffn.gate state.normedBuf state.gateBuf
    Linear.LinearLayer.forward ctx block.ffn.up state.normedBuf state.upBuf
    ce "k8"
      (geluMulKernel cfg.intermediateSize)
      [("gate", state.gateBuf), ("up", state.upBuf), ("output", state.geluBuf)]
      (.dispatch1D cfg.intermediateSize)
    Linear.LinearLayer.forward ctx block.ffn.down state.geluBuf state.ffnOutBuf

    -- Apply post_norm_1 to shared expert output (avoid aliasing)
    match block.moePostNorm1 with
    | some norm =>
      RMSNorm.forward ctx norm state.ffnOutBuf state.normedBuf2
      -- Copy back: normedBuf2 → ffnOutBuf
      ce "k9"
        (PerLayerEmbedding.scaleKernel cfg.hiddenSize 1.0)
        [("input", state.normedBuf2), ("output", state.ffnOutBuf)]
        (.dispatch1D cfg.hiddenSize)
    | none => pure ()

    -- 2. Router: rms_norm(attn_out) * (1/sqrt(n_embd)) * router_scale → logits → softmax → top-K
    match block.moeRouterWeight, block.moeRouterScale with
    | some routerW, some routerS =>
      ce "k10"
        (MoE.routerPreprocessKernel cfg.hiddenSize cfg.rmsNormEps)
        [("input", state.attnResidualBuf), ("router_scale", routerS), ("output", state.moeRouterOutBuf)]
        ({ numWorkgroups := (1, 1, 1) : Hesper.ExecConfig })
      -- Router matmul: moeRouterOutBuf [hiddenSize] @ routerW^T → moeLogitsBuf [numExperts]
      let routerMatmulConfig : Hesper.WGSL.MatMul.Config := {
        M := 1, N := cfg.numExperts, K := cfg.hiddenSize
      }
      Hesper.WGSL.MatMul.executeMatMulTranspose ctx state.moeRouterOutBuf routerW state.moeLogitsBuf routerMatmulConfig
      -- Top-K selection
      ce "k11"
        (MoE.softmaxTopKKernel cfg.numExperts cfg.numExpertsUsed)
        [("logits", state.moeLogitsBuf), ("indices", state.moeIndicesBuf), ("weights", state.moeWeightsBuf)]
        (.dispatch1D 1)
    | _, _ => pure ()

    -- 3. Routed experts: ffn_pre_norm_2 → expert GeGLU FFN → weighted sum
    match block.moeGateUpExps, block.moeDownExps, block.moePreNorm2, block.moePostNorm2 with
    | some gateUpExps, some downExps, some preNorm2, some postNorm2 =>
      -- Pre-norm for routed expert input
      RMSNorm.forward ctx preNorm2 state.attnResidualBuf state.moeNormedBuf

      -- Zero the accumulator
      ce "k12"
        (residualAddKernel cfg.hiddenSize)  -- hack: 0 + 0 = 0 (both inputs are same zeroed buf)
        [("a", state.moeExpertOutBuf), ("b", state.moeExpertOutBuf), ("output", state.moeExpertOutBuf)]
        (.dispatch1D cfg.hiddenSize)
      -- Actually zero it properly
      ce "k13"
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
        ce "k14"
          (MoE.expertGateUpKernel moeConfig k true)
          [("input", state.moeNormedBuf), ("gate_up_weights", gateUpExps),
           ("expert_indices", state.moeIndicesBuf), ("output", state.moeExpertGateBuf)]
          ({ numWorkgroups := (cfg.expertFFSize, 1, 1) : Hesper.ExecConfig })
        -- Up projection
        ce "k15"
          (MoE.expertGateUpKernel moeConfig k false)
          [("input", state.moeNormedBuf), ("gate_up_weights", gateUpExps),
           ("expert_indices", state.moeIndicesBuf), ("output", state.moeExpertUpBuf)]
          ({ numWorkgroups := (cfg.expertFFSize, 1, 1) : Hesper.ExecConfig })
        -- GELU * up
        ce "k16"
          (MoE.expertGeluMulKernel cfg.expertFFSize)
          [("gate", state.moeExpertGateBuf), ("up", state.moeExpertUpBuf), ("output", state.moeExpertGeluBuf)]
          (.dispatch1D cfg.expertFFSize)
        -- Down projection
        ce "k17"
          (MoE.expertDownKernel moeConfig k)
          [("input", state.moeExpertGeluBuf), ("down_weights", downExps),
           ("expert_indices", state.moeIndicesBuf), ("output", state.moeExpertDownBuf)]
          ({ numWorkgroups := (cfg.hiddenSize, 1, 1) : Hesper.ExecConfig })
        -- Weighted accumulate: moeExpertOutBuf += weight[k] * expertDownBuf
        ce "k18"
          (MoE.weightedAccumulateKernel cfg.hiddenSize cfg.numExpertsUsed k)
          [("accumulator", state.moeExpertOutBuf), ("expert_output", state.moeExpertDownBuf),
           ("weights", state.moeWeightsBuf)]
          (.dispatch1D cfg.hiddenSize)

      -- post_norm_2 on routed expert output
      -- Avoid aliasing: moeExpertOutBuf → normedBuf2 → moeExpertOutBuf
      RMSNorm.forward ctx postNorm2 state.moeExpertOutBuf state.normedBuf2
      ce "k19"
        (PerLayerEmbedding.scaleKernel cfg.hiddenSize 1.0)
        [("input", state.normedBuf2), ("output", state.moeExpertOutBuf)]
        (.dispatch1D cfg.hiddenSize)

      -- 4. Combined: shared_expert + routed_experts
      ce "k20"
        (residualAddKernel cfg.hiddenSize)
        [("a", state.ffnOutBuf), ("b", state.moeExpertOutBuf), ("output", state.ffnOutBuf)]
        (.dispatch1D cfg.hiddenSize)
    | _, _, _, _ => pure ()  -- No MoE weights: shared expert only

    -- Post-FFN norm + residual (avoid aliasing: ffnOutBuf → normedBuf2 → outputBuf)
    RMSNorm.forward ctx block.postFFNNorm state.ffnOutBuf state.normedBuf2
    ce "k21"
      (residualAddKernel cfg.hiddenSize)
      [("a", state.normedBuf2), ("b", state.attnResidualBuf), ("output", outputBuf)]
      (.dispatch1D cfg.hiddenSize)
  else do
    -- Dense FFN path (GeGLU).
    Hesper.WGSL.Execute.withSection "ffnNorm" do
      RMSNorm.forward ctx block.ffnNorm state.attnResidualBuf state.normedBuf
    Hesper.WGSL.Execute.withSection "ffnGateUp" do
      Linear.LinearLayer.forward ctx block.ffn.gate state.normedBuf state.gateBuf
      Linear.LinearLayer.forward ctx block.ffn.up state.normedBuf state.upBuf
    Hesper.WGSL.Execute.withSection "ffnGeluMul" do
      ce "k22"
        (geluMulKernel cfg.intermediateSize)
        [("gate", state.gateBuf), ("up", state.upBuf), ("output", state.geluBuf)]
        (.dispatch1D cfg.intermediateSize)
    Hesper.WGSL.Execute.withSection "ffnDown" do
      Linear.LinearLayer.forward ctx block.ffn.down state.geluBuf state.ffnOutBuf

    -- Post-FFN norm + residual (avoid aliasing: ffnOutBuf → normedBuf2 → outputBuf)
    Hesper.WGSL.Execute.withSection "postFFNNorm" do
      RMSNorm.forward ctx block.postFFNNorm state.ffnOutBuf state.normedBuf2
      ce "k23"
        (residualAddKernel cfg.hiddenSize)
        [("a", state.normedBuf2), ("b", state.attnResidualBuf), ("output", outputBuf)]
        (.dispatch1D cfg.hiddenSize)

  -- Step 8: Per-layer embedding (optional, from gemma4-iswa.cpp:192-213)
  -- pe_in = cur (= outputBuf at this point)
  -- gate = GELU(per_layer_inp_gate @ cur)
  -- cur = gate * per_layer_input[layerIdx]
  -- cur = per_layer_proj @ cur
  -- cur = per_layer_post_norm(cur)
  -- output = pe_in + cur
  Hesper.WGSL.Execute.withSection "perLayerEmbd" do
    match perLayerEmbd, perLayerInput with
    | some plEmbd, some plInputAll =>
      -- per_layer_inp_gate @ outputBuf → plGateBuf [embdPerLayer]
      Hesper.WGSL.Execute.withSection "ple.inpGate" do
        Linear.LinearLayer.forward ctx plEmbd.inpGate outputBuf state.plGateBuf
      -- GELU(gate) * per_layer_input[layerIdx] → moeRouterOutBuf (slice via offset kernel)
      let plOffset := li * cfg.embdPerLayer
      let plTotalSize := cfg.embdPerLayer * cfg.numHiddenLayers
      Hesper.WGSL.Execute.withSection "ple.geluGateMul" do
        ce "k24"
          (PerLayerEmbedding.geluGateMulSliceKernel cfg.embdPerLayer plTotalSize plOffset)
          [("gate", state.plGateBuf), ("per_layer_input", plInputAll), ("output", state.moeRouterOutBuf)]
          (.dispatch1D cfg.embdPerLayer)
      -- per_layer_proj @ moeRouterOutBuf → plProjBuf [hiddenSize]
      Hesper.WGSL.Execute.withSection "ple.proj" do
        Linear.LinearLayer.forward ctx plEmbd.proj state.moeRouterOutBuf state.plProjBuf
      -- Fused post-norm + residual-add, in place on outputBuf.
      -- Replaces three dispatches (postNorm, copyBack, residAdd) with
      -- one: `outputBuf[i] += rmsNorm(plProjBuf)[i] * postNorm.scale[i]`.
      Hesper.WGSL.Execute.withSection "ple.postNormAdd" do
        ce "k25"
          (fusedPerLayerPostKernel cfg.hiddenSize cfg.rmsNormEps)
          [("proj", state.plProjBuf), ("weight", plEmbd.postNorm.scale), ("residual", outputBuf)]
          { numWorkgroups := (1, 1, 1)
            workgroupSize := { x := 256, y := 1, z := 1 }
            extensions := ["subgroups"]
            : Hesper.ExecConfig }
    | _, _ => pure ()

  -- Step 9: Layer output scale (optional)
  -- Use normedBuf2 as temp to avoid input/output aliasing
  match block.outScale with
  | some scale =>
    ce "k26"
      (PerLayerEmbedding.layerScaleKernel cfg.hiddenSize)
      [("input", outputBuf), ("scale", scale), ("output", state.normedBuf2)]
      (.dispatch1D cfg.hiddenSize)
    -- Copy back: normedBuf2 → outputBuf
    ce "k27"
      (PerLayerEmbedding.scaleKernel cfg.hiddenSize 1.0)
      [("input", state.normedBuf2), ("output", outputBuf)]
      (.dispatch1D cfg.hiddenSize)
  | none => pure ()

/-- Run full single-token forward pass through the model.
    Returns logits in state.logitsBuf. -/
def forwardSingleToken [GPUBackend β] (ctx : β)
    (model : Gemma4Model (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (tokenId : Nat) (pos : Nat)
    (state : InferenceState (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (kcr : Option (KernelCacheRefs (GPUBackend.CachedDispatch β)) := none) : IO Unit := do
  -- Step 1: Embedding lookup (format-dependent)
  -- Cached execute helper (same as forwardBlock's ce)
  let ce := fun (name : String) (shader : ShaderM Unit)
      (namedBufs : List (String × GPUBackend.Buf β)) (config : Hesper.ExecConfig) => do
    -- Key includes name + config (numWorkgroups, workgroupSize) to distinguish
    -- same-named kernels with different parameters (e.g., full vs SWA attention)
    let key := hash (name, config.numWorkgroups, config.workgroupSize.x, config.workgroupSize.y)
    match kcr with
    | some k =>
      let ref ← k.getRef key
      GPUBackend.executeWithConfigCached ctx shader namedBufs config key ref
    | none => GPUBackend.execute ctx shader namedBufs config
  let tokenBytes := Hesper.WebGPU.BufferOps.uint32ToBytes tokenId.toUInt32
  GPUBackend.writeBufferOffset ctx state.tokenBuf 0 tokenBytes
  Hesper.WGSL.Execute.withSection "embedLookup" do
    match model.embdFormat with
    | .Q6_K =>
      -- Q6_K on-the-fly dequant lookup
      ce "k28"
        (Hesper.Quantization.Q6_K.q6kEmbeddingLookupKernel model.config.vocabSize model.config.hiddenSize)
        [("token_ids", state.tokenBuf), ("embedding_table", model.embedding.embeddingTable), ("output", state.buf1)]
        (.dispatch1D model.config.hiddenSize)
    | _ =>
      -- F32 / F16 / Q4_K: use existing Embedding.forward (assumes F32 interpretation)
      Embedding.forward ctx model.embedding state.tokenBuf state.buf1 1 1

  -- Scale embeddings by sqrt(hiddenSize)
  -- Cannot alias input/output in WebGPU, so output to buf2
  Hesper.WGSL.Execute.withSection "embedScale" do
    ce "k29"
      (embeddingScaleKernel model.config.hiddenSize model.config.hiddenSize)
      [("input", state.buf1), ("output", state.buf2)]
      (.dispatch1D model.config.hiddenSize)

  -- Step 1b: Per-layer input precomputation (gemma4-iswa.cpp:258-311)
  -- The per_layer_token_embd table is too large (>2 GB) for a single GPU buffer
  -- with the current Dawn limits, so we dequant just the input token's row on
  -- CPU and upload (~43 KB).
  Hesper.WGSL.Execute.withSection "perLayerInputPre" do
    match model.perLayerEmbdTableCPU, model.perLayerModelProj, model.perLayerProjNorm with
    | some embdTableCPU, some modelProj, some projNorm =>
      let embdPL := model.config.embdPerLayer
      let nLayers := model.config.numHiddenLayers
      let totalPL := embdPL * nLayers

      -- 1) Dequant the `tokenId` row of the Q6_K per-layer embedding
      --    table on the GPU. The table is too big to keep on the GPU
      --    (> single-buffer limit), so we slice the 33 KB of raw Q6_K
      --    bytes for this row out of the CPU ByteArray, upload it to a
      --    small scratch buffer, and run `q6kSingleRowDequantScaleKernel`
      --    to dequant + scale(sqrt(embdPL)) into plModelProj in one
      --    pass. This replaces a 10 ms/token CPU dequant+upload loop.
      Hesper.WGSL.Execute.withSection "plPre.rowUpload" do
        let rowOffset := tokenId * model.perLayerEmbdRowBytes
        let rowBytes := embdTableCPU.extract rowOffset (rowOffset + model.perLayerEmbdRowBytes)
        GPUBackend.writeBufferOffset ctx state.plRawRowBuf 0 rowBytes
      Hesper.WGSL.Execute.withSection "plPre.gpuDequant" do
        let scaleFactor : Float := Float.sqrt embdPL.toFloat
        ce "k30"
          (Hesper.Quantization.Q6_K.q6kSingleRowDequantScaleKernel totalPL scaleFactor)
          [("row", state.plRawRowBuf), ("output", state.plModelProj)]
          (.dispatch1D totalPL)

      -- 2) per_layer_model_proj @ buf2 → plTokenSelected
      let projConfig : Hesper.WGSL.MatMul.Config := {
        M := 1, N := totalPL, K := model.config.hiddenSize
      }
      Hesper.WGSL.Execute.withSection "plPre.f16Matmul" do
        if projConfig.K % 64 == 0 then
          Hesper.WGSL.MatMul.executeMatMulTransposeF16BlockCoop ctx state.buf2 modelProj state.plTokenSelected projConfig
        else
          Hesper.WGSL.MatMul.executeMatMulTransposeF16 ctx state.buf2 modelProj state.plTokenSelected projConfig

      Hesper.WGSL.Execute.withSection "plPre.scale" do
        ce "k31"
          (PerLayerEmbedding.scaleKernel totalPL (1.0 / Float.sqrt model.config.hiddenSize.toFloat))
          [("input", state.plTokenSelected), ("output", state.plInputAll)]
          (.dispatch1D totalPL)

      Hesper.WGSL.Execute.withSection "plPre.chunkedNorm" do
        ce "k32"
          (chunkedRMSNormKernel embdPL nLayers model.config.rmsNormEps)
          [("input", state.plInputAll), ("weight", projNorm.scale), ("output", state.plTokenSelected)]
          { numWorkgroups := (nLayers, 1, 1), workgroupSize := { x := min embdPL 256, y := 1, z := 1 } : Hesper.ExecConfig }

      Hesper.WGSL.Execute.withSection "plPre.scaledAdd" do
        ce "k33"
          (scaledAddKernel totalPL (1.0 / Float.sqrt 2.0))
          [("a", state.plTokenSelected), ("b", state.plModelProj), ("output", state.plInputAll)]
          (.dispatch1D totalPL)
    | _, _, _ => pure ()

  -- Step 2: Process all transformer blocks (starting from buf2 as current).
  -- If the caller has already started a batch (e.g. a batched prefill
  -- wrapper), we nest inside it instead of starting a new one. Callers that
  -- want per-dispatch GPU timing (e.g. `gemma4-profile`) can flip
  -- `Hesper.Layers.Linear.profilingRef` to `true`, in which case we also
  -- skip our own begin/endBatch so each dispatch auto-syncs.
  let profiling ← Hesper.Layers.Linear.profilingRef.get
  let alreadyBatching ← Hesper.WGSL.Execute.isBatching
  let ownBatch := !profiling && !alreadyBatching
  if ownBatch then GPUBackend.beginBatch ctx

  let mut currentBuf := state.buf2
  let mut nextBuf := state.buf1

  let mut blockIdx := 0
  let plInputBuf := if model.config.hasPerLayerEmbeddings then some state.plInputAll else none
  for block in model.blocks do
    let plEmbd := if blockIdx < model.perLayerBlocks.size then
      model.perLayerBlocks[blockIdx]!
    else none
    forwardBlock ctx block model.config currentBuf nextBuf state pos (kcr := kcr) (perLayerEmbd := plEmbd) (perLayerInput := plInputBuf)
    let oldCb := currentBuf
    currentBuf := nextBuf
    nextBuf := oldCb
    blockIdx := blockIdx + 1

  -- Step 3: Final norm
  Hesper.WGSL.Execute.withSection "finalNorm" do
    RMSNorm.forward ctx model.finalNorm currentBuf nextBuf

  -- Step 4: LM head matmul (1 × hiddenSize @ hiddenSize × vocabSize)
  Hesper.WGSL.Execute.withSection "lmHead" do
    match model.embdFormat with
    | .Q6_K =>
      -- 2D dispatch because vocabSize (262144) exceeds the 65535 per-dimension limit.
      let gridX : Nat := 4096
      let gridY : Nat := (model.config.vocabSize + gridX - 1) / gridX
      -- Prefer the subgroup variant when available — 32-thread workgroups
      -- with hardware `subgroupAdd` instead of 256-thread tree reduction.
      let useSubgroups ← GPUBackend.hasSubgroupSupport ctx
      let shader := if useSubgroups then
          Hesper.Quantization.Q6_K.fusedQ6KLinearBlockCoopKernel
            model.config.hiddenSize model.config.vocabSize gridX
        else
          Hesper.Quantization.Q6_K.fusedQ6KLinearKernel
            model.config.hiddenSize model.config.vocabSize 256 gridX
      let wgSize := if useSubgroups then 32 else 256
      let lmBufs : List (String × GPUBackend.Buf β) :=
        [("weights", model.outputWeight), ("input", nextBuf), ("output", state.logitsBuf)]
      ce "k34"
        shader
        lmBufs
        { numWorkgroups := (gridX, gridY, 1)
          workgroupSize := { x := wgSize, y := 1, z := 1 }
          extensions := if useSubgroups then ["subgroups"] else []
          : Hesper.ExecConfig }
    | _ =>
      let lmHeadConfig : Hesper.WGSL.MatMul.Config := {
        M := 1, N := model.config.vocabSize, K := model.config.hiddenSize
      }
      Hesper.WGSL.MatMul.executeMatMulTranspose ctx nextBuf model.outputWeight state.logitsBuf lmHeadConfig

  -- Optional: save pre-softcap logits for TTT surprise sensor.
  -- Only runs when preSoftcapBuf is set (zero cost otherwise).
  match state.preSoftcapBuf with
  | some psBuf =>
    ce "k35"
      (PerLayerEmbedding.scaleKernel model.config.vocabSize 1.0)
      [("input", state.logitsBuf), ("output", psBuf)]
      (.dispatch1D model.config.vocabSize)
  | none => pure ()

  -- Step 5: Logit softcapping (y = scale * tanh(x / scale))
  Hesper.WGSL.Execute.withSection "logitSoftcap" do
    if model.config.logitSoftcapScale > 0.0 then
      ce "k36"
        (logitSoftcapKernel model.config.vocabSize model.config.logitSoftcapScale)
        [("input", state.logitsBuf), ("output", state.logitsBuf2)]
        (.dispatch1D model.config.vocabSize)
      ce "k37"
        (PerLayerEmbedding.scaleKernel model.config.vocabSize 1.0)
        [("input", state.logitsBuf2), ("output", state.logitsBuf)]
        (.dispatch1D model.config.vocabSize)

  if ownBatch then GPUBackend.endBatch ctx

/-! ## Text Generation -/

/-- GPU argmax: parallel reduction to find token with highest logit. -/
private def argmaxKernel (vocabSize : Nat) : ShaderM Unit := do
  let tid ← ShaderM.localId
  let tid := Exp.vec3X tid
  ShaderM.sharedNamed "shared_vals" (.array (.scalar .f32) 256)
  ShaderM.sharedNamed "shared_idxs" (.array (.scalar .u32) 256)
  let _logits ← ShaderM.declareInputBuffer "logits" (.array (.scalar .f32) vocabSize)
  let _result ← ShaderM.declareOutputBuffer "result" (.array (.scalar .u32) 1)
  ShaderM.varNamed "local_max" (.scalar .f32) (Exp.litF32 (-1.0e38))
  ShaderM.varNamed "local_idx" (.scalar .u32) (Exp.litU32 0)
  ShaderM.loop tid (Exp.litU32 vocabSize) (Exp.litU32 256) fun i => do
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := vocabSize) "logits" i
    ShaderM.if_ (Exp.gt val (Exp.var "local_max")) (do
      ShaderM.assign "local_max" val
      ShaderM.assign "local_idx" i
    ) (pure ())
  ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_vals" tid (Exp.var "local_max")
  ShaderM.writeWorkgroup (ty := .scalar .u32) "shared_idxs" tid (Exp.var "local_idx")
  ShaderM.barrier
  let mut stride := 128
  while stride > 0 do
    ShaderM.if_ (Exp.lt tid (Exp.litU32 stride)) (do
      let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 256) "shared_vals" tid
      let b ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 256) "shared_vals" (Exp.add tid (Exp.litU32 stride))
      ShaderM.if_ (Exp.gt b a) (do
        ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_vals" tid b
        let bIdx ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := 256) "shared_idxs" (Exp.add tid (Exp.litU32 stride))
        ShaderM.writeWorkgroup (ty := .scalar .u32) "shared_idxs" tid bIdx
      ) (pure ())
    ) (pure ())
    ShaderM.barrier
    stride := stride / 2
  ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
    let maxIdx ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := 256) "shared_idxs" (Exp.litU32 0)
    ShaderM.writeBuffer (ty := .scalar .u32) "result" (Exp.litU32 0) maxIdx
  ) (pure ())

private def gpuArgmax [GPUBackend β] (ctx : β) (logitsBuf argmaxBuf : GPUBackend.Buf β) (vocabSize : Nat) : IO Nat := do
  GPUBackend.execute ctx (argmaxKernel vocabSize)
    [("logits", logitsBuf), ("result", argmaxBuf)]
    { workgroupSize := { x := 256 }, numWorkgroups := (1, 1, 1) }
  let bytes ← GPUBackend.readBuffer ctx argmaxBuf (4 : USize)
  return (Hesper.Basic.bytesToUInt32 bytes 0).toNat

/-- Generate tokens from a Gemma 4 model.

    @param device WebGPU device
    @param model Loaded Gemma 4 model
    @param promptTokens Input token IDs
    @param maxTokens Maximum new tokens to generate
    @param eosToken Optional EOS token ID for early stopping
-/
def generate [GPUBackend β] (ctx : β) (model : Gemma4Model (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (promptTokens : Array Nat) (maxTokens : Nat)
    (eosToken : Option Nat := none)
    (extraEosTokens : Array Nat := #[]) : IO (Array Nat) := do
  IO.println s!"[Gemma4] Generating: {promptTokens.size} prompt tokens, max {maxTokens} new tokens"

  -- Create inference state + kernel cache refs
  let state ← createInferenceState ctx model.config
  let kcr ← createKernelCacheRefs (β := β)

  let mut tokens := promptTokens

  -- Phase 1: Prefill (process prompt tokens)
  IO.println s!"[Prefill] Processing {promptTokens.size} prompt tokens..."
  let prefillStart ← IO.monoNanosNow
  for i in [0:promptTokens.size] do
    if i >= model.config.maxSeqLen then break
    forwardSingleToken ctx model promptTokens[i]! i state (kcr := none)
  let prefillEnd ← IO.monoNanosNow
  let prefillMs := (prefillEnd - prefillStart).toFloat / 1_000_000.0
  IO.println s!"[Prefill] Done in {prefillMs} ms"

  -- Phase 2: Decode (generate new tokens)
  let genStart ← IO.monoNanosNow
  let mut genCount : Nat := 0

  for _ in [0:maxTokens] do
    if tokens.size >= model.config.maxSeqLen then break

    -- Sample: GPU-side greedy argmax (download 4 bytes instead of 1 MB)
    let nextToken ← gpuArgmax ctx state.logitsBuf state.argmaxBuf model.config.vocabSize

    tokens := tokens.push nextToken
    genCount := genCount + 1

    -- Check EOS (primary + extras, e.g. Gemma 4's <end_of_turn> = 106)
    let mut stop := false
    match eosToken with
    | some eos => if nextToken == eos then stop := true
    | none => pure ()
    if extraEosTokens.any (· == nextToken) then stop := true
    if stop then break

    -- Forward pass for next token
    let newPos := tokens.size - 1
    if newPos < model.config.maxSeqLen then
      forwardSingleToken ctx model nextToken newPos state (kcr := none)

  let genEnd ← IO.monoNanosNow
  let genMs := (genEnd - genStart).toFloat / 1_000_000.0
  let msPerToken := if genCount > 0 then genMs / genCount.toFloat else 0.0
  let tps := if msPerToken > 0 then 1000.0 / msPerToken else 0.0
  IO.println s!"[Gemma4] Generated {genCount} tokens in {genMs} ms ({tps} tokens/sec)"

  return tokens

end Hesper.Models.Gemma4
