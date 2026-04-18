import Hesper.Backend
import Hesper.Backend.WebGPU
import Hesper.Backend.CUDA
import Hesper.Circuit.IR
import Hesper.Circuit.Lowering
import Hesper.Circuit.Passes
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

/-- Batched RoPE-Q with frequency factors: rotates Q for all `seqLen` query
    tokens.  Q layout: [seqLen, numHeads, headDim] column-major
    (`q[col * (numHeads * headDim) + h * headDim + d]`).

    `params[0]` = startPos.  Token at column `col` uses pos = startPos + col.
    Grid: dispatch1D(numHeads * dimPairs * seqLen). -/
def ropeWithFreqFactorsBatchKernel (headDim numHeads seqLen : Nat) (ropeBase : Float) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let dimPairs := headDim / 2
  let perTokenElems := numHeads * dimPairs
  let totalElements := perTokenElems * seqLen

  let qDim := numHeads * headDim
  let _input  ← ShaderM.declareInputBuffer "input"  (.array (.scalar .f32) (qDim * seqLen))
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) (qDim * seqLen))
  let _params ← ShaderM.declareInputBuffer "params" (.array (.scalar .u32) 1)
  let _freqFactors ← ShaderM.declareInputBuffer "freq_factors" (.array (.scalar .f32) dimPairs)

  ShaderM.if_ (Exp.lt idx (Exp.litU32 totalElements)) (do
    let col := Exp.div idx (Exp.litU32 perTokenElems)
    let withinTok := Exp.sub idx (Exp.mul col (Exp.litU32 perTokenElems))
    let dimPair := Exp.mod withinTok (Exp.litU32 dimPairs)
    let head    := Exp.div withinTok (Exp.litU32 dimPairs)

    let startPos ← ShaderM.readBuffer (ty := .scalar .u32) (n := 1) "params" (Exp.litU32 0)
    let pos := Exp.add startPos col
    let posF32 := Exp.toF32 pos

    let freqFactor ← ShaderM.readBuffer (ty := .scalar .f32) (n := dimPairs) "freq_factors" dimPair

    let dimPairF32 := Exp.toF32 dimPair
    let exponent := Exp.div (Exp.mul (Exp.litF32 2.0) dimPairF32) (Exp.litF32 headDim.toFloat)
    let freqInv := Exp.pow (Exp.litF32 ropeBase) (Exp.neg exponent)
    let theta := Exp.div (Exp.mul posF32 freqInv) freqFactor
    let cosTheta := Exp.cos theta
    let sinTheta := Exp.sin theta

    let halfDim := headDim / 2
    let colBase    := Exp.mul col (Exp.litU32 qDim)
    let headOffset := Exp.mul head (Exp.litU32 headDim)
    let idx0 := Exp.add (Exp.add colBase headOffset) dimPair
    let idx1 := Exp.add (Exp.add colBase headOffset) (Exp.add dimPair (Exp.litU32 halfDim))

    let x0 ← ShaderM.readBuffer (ty := .scalar .f32) (n := qDim * seqLen) "input" idx0
    let x1 ← ShaderM.readBuffer (ty := .scalar .f32) (n := qDim * seqLen) "input" idx1

    let x0_new := Exp.sub (Exp.mul x0 cosTheta) (Exp.mul x1 sinTheta)
    let x1_new := Exp.add (Exp.mul x0 sinTheta) (Exp.mul x1 cosTheta)

    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx0 x0_new
    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx1 x1_new
  ) (pure ())

/-- Batched RoPE-K + KV cache write: for all `seqLen` tokens, rotate the K row,
    write rotated K and unmodified V into the KV cache at slot `startPos + col`.

    K/V input layout: [seqLen, numKVHeads, headDim] column-major
    (`new_k[col * kvDim + kvH * headDim + d]`, kvDim = numKVHeads * headDim).
    KV cache layout: [numKVHeads, maxSeqLen, headDim] (same as single-token).

    `params[0]` = startPos.  Token at column `col` writes K/V to cache slot
    `startPos + col`.  Grid: dispatch1D(numKVHeads * dimPairs * seqLen).  Each
    thread processes one (col, kvHead, dimPair) — emits both rotated K
    components AND the corresponding V at idx0/idx1. -/
def fusedRopeKAndCacheWriteBatchKernel (numKVHeads maxSeqLen headDim seqLen : Nat)
    (ropeBase : Float) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let dimPairs := headDim / 2
  let perTokenElems := numKVHeads * dimPairs
  let totalElements := perTokenElems * seqLen
  let kvDim := numKVHeads * headDim

  let _newK   ← ShaderM.declareInputBuffer "new_k" (.array (.scalar .f32) (kvDim * seqLen))
  let _newV   ← ShaderM.declareInputBuffer "new_v" (.array (.scalar .f32) (kvDim * seqLen))
  let _kCache ← ShaderM.declareOutputBuffer "k_cache" (.array (.scalar .f32) (numKVHeads * maxSeqLen * headDim))
  let _vCache ← ShaderM.declareOutputBuffer "v_cache" (.array (.scalar .f32) (numKVHeads * maxSeqLen * headDim))
  let _params ← ShaderM.declareInputBuffer "params" (.array (.scalar .u32) 1)
  let _freqFactors ← ShaderM.declareInputBuffer "freq_factors" (.array (.scalar .f32) dimPairs)

  ShaderM.if_ (Exp.lt idx (Exp.litU32 totalElements)) (do
    let col := Exp.div idx (Exp.litU32 perTokenElems)
    let withinTok := Exp.sub idx (Exp.mul col (Exp.litU32 perTokenElems))
    let dimPair := Exp.mod withinTok (Exp.litU32 dimPairs)
    let kvHead  := Exp.div withinTok (Exp.litU32 dimPairs)

    let startPos ← ShaderM.readBuffer (ty := .scalar .u32) (n := 1) "params" (Exp.litU32 0)
    let pos := Exp.add startPos col
    let posF32 := Exp.toF32 pos

    let freqFactor ← ShaderM.readBuffer (ty := .scalar .f32) (n := dimPairs) "freq_factors" dimPair

    let dimPairF32 := Exp.toF32 dimPair
    let exponent := Exp.div (Exp.mul (Exp.litF32 2.0) dimPairF32) (Exp.litF32 headDim.toFloat)
    let freqInv := Exp.pow (Exp.litF32 ropeBase) (Exp.neg exponent)
    let theta := Exp.div (Exp.mul posF32 freqInv) freqFactor
    let cosTheta := Exp.cos theta
    let sinTheta := Exp.sin theta

    let halfDim := headDim / 2
    let colBase  := Exp.mul col (Exp.litU32 kvDim)
    let kvHeadOffset := Exp.mul kvHead (Exp.litU32 headDim)
    let inIdx0 := Exp.add (Exp.add colBase kvHeadOffset) dimPair
    let inIdx1 := Exp.add (Exp.add colBase kvHeadOffset) (Exp.add dimPair (Exp.litU32 halfDim))

    let k0 ← ShaderM.readBuffer (ty := .scalar .f32) (n := kvDim * seqLen) "new_k" inIdx0
    let k1 ← ShaderM.readBuffer (ty := .scalar .f32) (n := kvDim * seqLen) "new_k" inIdx1
    let v0 ← ShaderM.readBuffer (ty := .scalar .f32) (n := kvDim * seqLen) "new_v" inIdx0
    let v1 ← ShaderM.readBuffer (ty := .scalar .f32) (n := kvDim * seqLen) "new_v" inIdx1

    let k0_new := Exp.sub (Exp.mul k0 cosTheta) (Exp.mul k1 sinTheta)
    let k1_new := Exp.add (Exp.mul k0 sinTheta) (Exp.mul k1 cosTheta)

    -- KV cache slot for this token = startPos + col
    let cacheBase := Exp.add (Exp.mul (Exp.mul kvHead (Exp.litU32 maxSeqLen)) (Exp.litU32 headDim))
                              (Exp.mul pos (Exp.litU32 headDim))
    let cIdx0 := Exp.add cacheBase dimPair
    let cIdx1 := Exp.add cacheBase (Exp.add dimPair (Exp.litU32 halfDim))

    ShaderM.writeBuffer (ty := .scalar .f32) "k_cache" cIdx0 k0_new
    ShaderM.writeBuffer (ty := .scalar .f32) "k_cache" cIdx1 k1_new
    ShaderM.writeBuffer (ty := .scalar .f32) "v_cache" cIdx0 v0
    ShaderM.writeBuffer (ty := .scalar .f32) "v_cache" cIdx1 v1
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

/-- Fused per-head qNorm + kNorm + vNorm, 3 separate dispatches collapsed
    into one.

    Dispatch: `(numHeads, 3, 1)`, workgroup size `min headDim 256`.
    `wg_id.y` multiplexes:
      0 = qNorm  (learned scale; active for `wg_id.x < numHeads`)
      1 = kNorm  (learned scale; active for `wg_id.x < numKVHeads`)
      2 = vNorm  (NO scale;      active for `wg_id.x < numKVHeads`)

    Each WG handles exactly one head's `[headDim]` slice: cooperative
    sum-of-squares + tree reduction + per-lane normalise-and-write.
    Identical math to `perHeadRMSNormKernel` / `perHeadBareRMSNormKernel`
    — just multiplexed by the grid's y dimension.

    WGs outside the valid head range (`wg_id.y in {1,2}` and
    `wg_id.x >= numKVHeads`) early-return BEFORE touching shared memory
    or barriers.  That's safe because `wg_id.x/y` are workgroup-uniform:
    the whole WG takes the same branch so no lane ever reaches a
    barrier its siblings skipped.

    Saves 2 dispatches per `cfg.hasKV` layer per token (3 → 1). -/
def fusedPerHeadQKVNormKernel
    (numHeads numKVHeads headDim : Nat) (eps : Float) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let headIdx := Exp.vec3X wid
  let yIdx := Exp.vec3Y wid
  let tid := Exp.vec3X lid
  let qTotal := numHeads * headDim
  let kvTotal := numKVHeads * headDim
  -- Buffer declarations.  All six (q in/scale/out, k in/scale/out, v
  -- in/out) are bound; the branch picks which ones this WG reads.
  let _qIn    ← ShaderM.declareInputBuffer  "q_in"    (.array (.scalar .f32) qTotal)
  let _qScale ← ShaderM.declareInputBuffer  "q_scale" (.array (.scalar .f32) headDim)
  let _qOut   ← ShaderM.declareOutputBuffer "q_out"   (.array (.scalar .f32) qTotal)
  let _kIn    ← ShaderM.declareInputBuffer  "k_in"    (.array (.scalar .f32) kvTotal)
  let _kScale ← ShaderM.declareInputBuffer  "k_scale" (.array (.scalar .f32) headDim)
  let _kOut   ← ShaderM.declareOutputBuffer "k_out"   (.array (.scalar .f32) kvTotal)
  let _vIn    ← ShaderM.declareInputBuffer  "v_in"    (.array (.scalar .f32) kvTotal)
  let _vOut   ← ShaderM.declareOutputBuffer "v_out"   (.array (.scalar .f32) kvTotal)
  let wgSize := if headDim < 256 then headDim else 256
  ShaderM.sharedNamed "shared_sum" (.array (.scalar .f32) wgSize)
  -- Early-return for K/V lanes beyond numKVHeads.  All lanes of the WG
  -- share `headIdx` and `yIdx`, so the branch is WG-uniform — safe to
  -- skip the entire body (including barriers).
  let invalidKV : Exp (.scalar .bool) :=
    Exp.and (Exp.ne yIdx (Exp.litU32 0))
            (Exp.ge headIdx (Exp.litU32 numKVHeads))
  ShaderM.if_ invalidKV (pure ()) (do
    -- Pick input and output buffer pair per y.  Reads from unused
    -- buffers are gated by the `ShaderM.if_` tree below so we never
    -- issue a load against a mismatched size.
    let headBase := Exp.mul headIdx (Exp.litU32 headDim)
    ShaderM.varNamed "local_sum" (.scalar .f32) (Exp.litF32 0.0)
    let localSum : Exp (.scalar .f32) := Exp.var "local_sum"
    -- ── Phase 1: cooperative sum-of-squares for this head/variant ──
    ShaderM.loop tid (Exp.litU32 headDim) (Exp.litU32 wgSize) fun i => do
      let elemIdx := Exp.add headBase i
      ShaderM.if_ (Exp.eq yIdx (Exp.litU32 0)) (do
        let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := qTotal) "q_in" elemIdx
        ShaderM.assign "local_sum" (Exp.add localSum (Exp.mul v v))
      ) (do
        ShaderM.if_ (Exp.eq yIdx (Exp.litU32 1)) (do
          let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := kvTotal) "k_in" elemIdx
          ShaderM.assign "local_sum" (Exp.add localSum (Exp.mul v v))
        ) (do
          let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := kvTotal) "v_in" elemIdx
          ShaderM.assign "local_sum" (Exp.add localSum (Exp.mul v v)))
      )
    ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_sum" tid localSum
    ShaderM.barrier
    -- Tree reduction.
    let mut stride := wgSize / 2
    while stride > 0 do
      ShaderM.if_ (Exp.lt tid (Exp.litU32 stride)) (do
        let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := wgSize) "shared_sum" tid
        let b ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := wgSize) "shared_sum"
                  (Exp.add tid (Exp.litU32 stride))
        ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_sum" tid (Exp.add a b)
      ) (pure ())
      ShaderM.barrier
      stride := stride / 2
    -- ── Phase 2: normalise and write output ──
    let sumSq ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := wgSize) "shared_sum" (Exp.litU32 0)
    let rms := Exp.inverseSqrt
                 (Exp.add (Exp.div sumSq (Exp.litF32 headDim.toFloat))
                          (Exp.litF32 eps))
    let rmsName ← ShaderM.var (.scalar .f32) rms
    let rmsRef : Exp (.scalar .f32) := Exp.var rmsName
    ShaderM.loop tid (Exp.litU32 headDim) (Exp.litU32 wgSize) fun i => do
      let elemIdx := Exp.add headBase i
      ShaderM.if_ (Exp.eq yIdx (Exp.litU32 0)) (do
        let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := qTotal) "q_in" elemIdx
        let w ← ShaderM.readBuffer (ty := .scalar .f32) (n := headDim) "q_scale" i
        let normed := Exp.mul (Exp.mul v rmsRef) w
        ShaderM.writeBuffer (ty := .scalar .f32) "q_out" elemIdx normed
      ) (do
        ShaderM.if_ (Exp.eq yIdx (Exp.litU32 1)) (do
          let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := kvTotal) "k_in" elemIdx
          let w ← ShaderM.readBuffer (ty := .scalar .f32) (n := headDim) "k_scale" i
          let normed := Exp.mul (Exp.mul v rmsRef) w
          ShaderM.writeBuffer (ty := .scalar .f32) "k_out" elemIdx normed
        ) (do
          let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := kvTotal) "v_in" elemIdx
          ShaderM.writeBuffer (ty := .scalar .f32) "v_out" elemIdx (Exp.mul v rmsRef))
      )
  )

/-- Batched fused per-head Q/K/V RMSNorm.  Same algorithm as
    `fusedPerHeadQKVNormKernel` but processes `seqLen` query tokens in a
    single dispatch.

    Q/K/V column-major batch layout:
    - q[col * (numHeads * headDim) + h * headDim + d]
    - k[col * (numKVHeads * headDim) + kh * headDim + d]
    - v[col * (numKVHeads * headDim) + kh * headDim + d]

    Grid: (numHeads * seqLen, 3, 1) — wgid.x packs (col * numHeads + head),
    wgid.y = which variant (0=Q, 1=K, 2=V).  Per-WG decomposes:
        col  = wgid.x / numHeads
        head = wgid.x % numHeads
    (vec3Z is not exposed in Hesper.WGSL.Exp; pack into x instead.) -/
def fusedPerHeadQKVNormBatchKernel
    (numHeads numKVHeads headDim seqLen : Nat) (eps : Float) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let packed  := Exp.vec3X wid
  let yIdx    := Exp.vec3Y wid
  let colIdx  := Exp.div packed (Exp.litU32 numHeads)
  let headIdx := Exp.sub packed (Exp.mul colIdx (Exp.litU32 numHeads))
  let tid     := Exp.vec3X lid
  let qTotal  := numHeads * headDim
  let kvTotal := numKVHeads * headDim
  let _qIn    ← ShaderM.declareInputBuffer  "q_in"    (.array (.scalar .f32) (qTotal * seqLen))
  let _qScale ← ShaderM.declareInputBuffer  "q_scale" (.array (.scalar .f32) headDim)
  let _qOut   ← ShaderM.declareOutputBuffer "q_out"   (.array (.scalar .f32) (qTotal * seqLen))
  let _kIn    ← ShaderM.declareInputBuffer  "k_in"    (.array (.scalar .f32) (kvTotal * seqLen))
  let _kScale ← ShaderM.declareInputBuffer  "k_scale" (.array (.scalar .f32) headDim)
  let _kOut   ← ShaderM.declareOutputBuffer "k_out"   (.array (.scalar .f32) (kvTotal * seqLen))
  let _vIn    ← ShaderM.declareInputBuffer  "v_in"    (.array (.scalar .f32) (kvTotal * seqLen))
  let _vOut   ← ShaderM.declareOutputBuffer "v_out"   (.array (.scalar .f32) (kvTotal * seqLen))
  let wgSize := if headDim < 256 then headDim else 256
  ShaderM.sharedNamed "shared_sum" (.array (.scalar .f32) wgSize)
  let invalidKV : Exp (.scalar .bool) :=
    Exp.and (Exp.ne yIdx (Exp.litU32 0))
            (Exp.ge headIdx (Exp.litU32 numKVHeads))
  ShaderM.if_ invalidKV (pure ()) (do
    let qColBase  := Exp.add (Exp.mul colIdx (Exp.litU32 qTotal))
                              (Exp.mul headIdx (Exp.litU32 headDim))
    let kvColBase := Exp.add (Exp.mul colIdx (Exp.litU32 kvTotal))
                              (Exp.mul headIdx (Exp.litU32 headDim))
    ShaderM.varNamed "local_sum" (.scalar .f32) (Exp.litF32 0.0)
    let localSum : Exp (.scalar .f32) := Exp.var "local_sum"
    ShaderM.loop tid (Exp.litU32 headDim) (Exp.litU32 wgSize) fun i => do
      ShaderM.if_ (Exp.eq yIdx (Exp.litU32 0)) (do
        let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := qTotal * seqLen) "q_in"
                  (Exp.add qColBase i)
        ShaderM.assign "local_sum" (Exp.add localSum (Exp.mul v v))
      ) (do
        ShaderM.if_ (Exp.eq yIdx (Exp.litU32 1)) (do
          let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := kvTotal * seqLen) "k_in"
                    (Exp.add kvColBase i)
          ShaderM.assign "local_sum" (Exp.add localSum (Exp.mul v v))
        ) (do
          let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := kvTotal * seqLen) "v_in"
                    (Exp.add kvColBase i)
          ShaderM.assign "local_sum" (Exp.add localSum (Exp.mul v v)))
      )
    ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_sum" tid localSum
    ShaderM.barrier
    let mut stride := wgSize / 2
    while stride > 0 do
      ShaderM.if_ (Exp.lt tid (Exp.litU32 stride)) (do
        let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := wgSize) "shared_sum" tid
        let b ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := wgSize) "shared_sum"
                  (Exp.add tid (Exp.litU32 stride))
        ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_sum" tid (Exp.add a b)
      ) (pure ())
      ShaderM.barrier
      stride := stride / 2
    let sumSq ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := wgSize) "shared_sum" (Exp.litU32 0)
    let rms := Exp.inverseSqrt
                 (Exp.add (Exp.div sumSq (Exp.litF32 headDim.toFloat))
                          (Exp.litF32 eps))
    let rmsName ← ShaderM.var (.scalar .f32) rms
    let rmsRef : Exp (.scalar .f32) := Exp.var rmsName
    ShaderM.loop tid (Exp.litU32 headDim) (Exp.litU32 wgSize) fun i => do
      ShaderM.if_ (Exp.eq yIdx (Exp.litU32 0)) (do
        let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := qTotal * seqLen) "q_in"
                  (Exp.add qColBase i)
        let w ← ShaderM.readBuffer (ty := .scalar .f32) (n := headDim) "q_scale" i
        ShaderM.writeBuffer (ty := .scalar .f32) "q_out" (Exp.add qColBase i)
          (Exp.mul (Exp.mul v rmsRef) w)
      ) (do
        ShaderM.if_ (Exp.eq yIdx (Exp.litU32 1)) (do
          let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := kvTotal * seqLen) "k_in"
                    (Exp.add kvColBase i)
          let w ← ShaderM.readBuffer (ty := .scalar .f32) (n := headDim) "k_scale" i
          ShaderM.writeBuffer (ty := .scalar .f32) "k_out" (Exp.add kvColBase i)
            (Exp.mul (Exp.mul v rmsRef) w)
        ) (do
          let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := kvTotal * seqLen) "v_in"
                    (Exp.add kvColBase i)
          ShaderM.writeBuffer (ty := .scalar .f32) "v_out" (Exp.add kvColBase i)
            (Exp.mul v rmsRef)))
  )

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

/-- Fused (pre-scale + chunked RMSNorm + weight + residual-add + post-scale)
    kernel — single dispatch over `chunkDim × numChunks` elements.

    Equivalent to the sequence:
      scaled[i]  = input[i] * preScale
      normed[i]  = rmsNorm(scaled, chunk)[i] * weight[i % chunkDim]
      output[i]  = (normed[i] + residual[i]) * addScale

    Replaces (pleScalePL + chunkedRMSNorm + scaledAdd) — three dispatches
    over `totalPL = embdPL × numHiddenLayers` elements — with one.

    Math: rmsNorm(x * C) = (x * C) / sqrt(mean((x*C)²) + ε)
                       = (x * C) / sqrt(C² * mean(x²) + ε)
    so the kernel computes `meanSq = mean(input²)` then
    `invRms = 1 / sqrt(C² * meanSq + ε)` and applies
    `normed[i] = input[i] * C * invRms * weight[i % chunk]`.
    Exact equivalent to running the 3 kernels in sequence (no
    approximation in the ε term). -/
def chunkedRMSNormAddScaledKernel (chunkDim numChunks : Nat)
    (eps preScale addScale : Float)
    (workgroupSize : Nat := 256) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let chunkIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid

  let totalElements := chunkDim * numChunks
  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) totalElements)
  let _weight ← ShaderM.declareInputBuffer "weight" (.array (.scalar .f32) chunkDim)
  let _residual ← ShaderM.declareInputBuffer "residual" (.array (.scalar .f32) totalElements)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) totalElements)

  ShaderM.sharedNamed "shared_sum" (.array (.scalar .f32) workgroupSize)

  let chunkBase := Exp.mul chunkIdx (Exp.litU32 chunkDim)

  ShaderM.varNamed "local_sum" (.scalar .f32) (Exp.litF32 0.0)
  let localSum : Exp (.scalar .f32) := Exp.var "local_sum"

  ShaderM.loop tid (Exp.litU32 chunkDim) (Exp.litU32 workgroupSize) fun i => do
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalElements) "input" (Exp.add chunkBase i)
    ShaderM.assign "local_sum" (Exp.add localSum (Exp.mul val val))

  ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_sum" tid localSum
  ShaderM.barrier

  let mut stride := workgroupSize / 2
  while stride > 0 do
    ShaderM.if_ (Exp.lt tid (Exp.litU32 stride)) (do
      let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_sum" tid
      let b ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_sum" (Exp.add tid (Exp.litU32 stride))
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_sum" tid (Exp.add a b)
    ) (pure ())
    ShaderM.barrier
    stride := stride / 2

  let sumSq ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_sum" (Exp.litU32 0)
  -- meanSq(input * preScale) = preScale² * meanSq(input).
  let preScaleSq : Float := preScale * preScale
  let scaledMeanSq := Exp.mul (Exp.litF32 preScaleSq)
                               (Exp.div sumSq (Exp.litF32 chunkDim.toFloat))
  let rms := Exp.inverseSqrt (Exp.add scaledMeanSq (Exp.litF32 eps))

  ShaderM.loop tid (Exp.litU32 chunkDim) (Exp.litU32 workgroupSize) fun i => do
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalElements) "input" (Exp.add chunkBase i)
    let w ← ShaderM.readBuffer (ty := .scalar .f32) (n := chunkDim) "weight" i
    let r ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalElements) "residual" (Exp.add chunkBase i)
    let scaled := Exp.mul val (Exp.litF32 preScale)
    let normed := Exp.mul (Exp.mul scaled rms) w
    let result := Exp.mul (Exp.add normed r) (Exp.litF32 addScale)
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
  -- Fused RMSNorm+Linear cache refs (attnNorm fused into Q/K/V projections)
  fusedNormQPrepared : IO.Ref (Option CacheT)
  fusedNormKPrepared : IO.Ref (Option CacheT)
  fusedNormVPrepared : IO.Ref (Option CacheT)

/-- Gemma 4 dense FFN layer -/
structure Gemma4FFN (BufT CacheT : Type) where
  gate : Linear.LinearLayer BufT CacheT
  up : Linear.LinearLayer BufT CacheT
  down : Linear.LinearLayer BufT CacheT
  fusedGateUpPrepared : IO.Ref (Option CacheT)
  -- Fused RMSNorm+Linear cache refs (ffnNorm fused into gate/up)
  fusedNormGatePrepared : IO.Ref (Option CacheT)
  fusedNormUpPrepared : IO.Ref (Option CacheT)

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
  perLayerEmbdTableGPU : Option BufT
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
  let dp4aQ8Buf ← IO.mkRef none
  let dp4aQuantizePrepared ← GPUBackend.newCacheRef (β := β)
  let dp4aMatmulPrepared ← GPUBackend.newCacheRef (β := β)
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
  }

/-- Load Gemma 4 model from GGUF file -/
def Gemma4Model.fromGGUFData [GPUBackend β] (ctx : β) (ggufData : ByteArray)
    (configOverride : Option Config := none) : IO (Gemma4Model (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)) := do
  IO.println s!"[Gemma4] Parsing GGUF ({ggufData.size} bytes)..."
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
  -- per_layer_token_embd: full table uploaded to GPU (like llama.cpp).
  -- No per-token CPU→GPU transfer needed.
  let (perLayerEmbdTableGPU, perLayerEmbdRowBytes, perLayerModelProj, perLayerProjNorm) ←
    if cfg.hasPerLayerEmbeddings then do
      IO.println "[Gemma4] Loading per-layer embeddings..."
      let blocksPerRow := (cfg.embdPerLayer * cfg.numHiddenLayers) / 256
      let rowBytes := blocksPerRow * 210  -- Q6_K block size
      let tableData ← match Hesper.GGUF.Loader.getTensorData gguf "per_layer_token_embd.weight" with
        | .ok (_, data) =>
          IO.println s!"  per_layer_token_embd: {data.size} bytes → GPU ({data.size / 1024 / 1024} MB)"
          let buf ← GPUBackend.allocBuffer ctx data.size.toUSize
          GPUBackend.writeBuffer ctx buf data
          pure (some buf)
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

/-- Load model from GGUF file path (reads file with IO.FS.readBinFile). -/
def Gemma4Model.fromGGUF [GPUBackend β] (ctx : β) (ggufPath : String)
    (configOverride : Option Config := none) : IO (Gemma4Model (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)) := do
  IO.println s!"[Gemma4] Loading model from {ggufPath}..."
  let ggufData ← IO.FS.readBinFile ggufPath
  Gemma4Model.fromGGUFData ctx ggufData configOverride

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
  posF32Buf : BufT     -- [1] f32: pos as f32 (for Circuit DSL dynamic offsets)
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
  -- Scratch buffer for Q8_1 quantized lmHead input (hiddenSize/32 * 9 u32),
  -- lazily allocated on first dp4a-enabled lmHead call.
  lmHeadQ8Buf : IO.Ref (Option BufT)
  lmHeadQuantizePrepared : IO.Ref (Option CacheT)
  lmHeadDP4APrepared : IO.Ref (Option CacheT)
  /-- Pinned host pointer (4 bytes).  See §CUDA Graph notes in the
      InferenceState doc header. -/
  stagingTokenPtr   : USize := 0
  /-- Pinned host pointer (8 bytes — pos @0, cacheLen @4). -/
  stagingParamsPtr  : USize := 0
  /-- Pinned host pointer (4 bytes) for per-layer-embedding row index. -/
  stagingPLRowPtr   : USize := 0
  /-- Pinned host pointer (4 bytes) for the batch-prefill column index. -/
  stagingColIdxPtr  : USize := 0
  /-- Pinned host pointer (4 bytes) for state.posF32Buf (pos as f32 —
      needed by the Circuit DSL scatter addrExpr that writes to the
      KV cache inside fusedRopeKAndCacheWrite). -/
  stagingPosF32Ptr  : USize := 0

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
    posF32Buf := ← GPUBackend.allocBuffer ctx (4 : USize)
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
    lmHeadQ8Buf := ← IO.mkRef none
    lmHeadQuantizePrepared := ← GPUBackend.newCacheRef (β := β)
    lmHeadDP4APrepared := ← GPUBackend.newCacheRef (β := β)
    -- Pinned-host staging (CUDA only; zero on other backends).  Using
    -- IO.getEnv as a crude backend check for now — a typeclass method
    -- `allocPinnedHost` on GPUBackend is the proper extension point.
    stagingTokenPtr  := ← match ← IO.getEnv "HESPER_CUDA_GRAPHS" with
                         | some _ => Hesper.CUDA.cuMemAllocHost 4
                         | none   => pure 0
    stagingParamsPtr := ← match ← IO.getEnv "HESPER_CUDA_GRAPHS" with
                         | some _ => Hesper.CUDA.cuMemAllocHost 8
                         | none   => pure 0
    stagingPLRowPtr  := ← match ← IO.getEnv "HESPER_CUDA_GRAPHS" with
                         | some _ => Hesper.CUDA.cuMemAllocHost 4
                         | none   => pure 0
    stagingColIdxPtr := ← match ← IO.getEnv "HESPER_CUDA_GRAPHS" with
                         | some _ => Hesper.CUDA.cuMemAllocHost 4
                         | none   => pure 0
    stagingPosF32Ptr := ← match ← IO.getEnv "HESPER_CUDA_GRAPHS" with
                         | some _ => Hesper.CUDA.cuMemAllocHost 4
                         | none   => pure 0
  }

/-- Dump a buffer to a file when HESPER_DUMP_DIR env is set.
    Caller passes the byte count; suffix identifies the checkpoint.
    Flushes any pending CUDA batch queue first so the buffer reflects all
    queued launches.  If there was an active batch, reopens it afterwards. -/
def dumpBuf [GPUBackend β] (ctx : β) (buf : GPUBackend.Buf β) (bytes : USize) (suffix : String) : IO Unit := do
  match ← IO.getEnv "HESPER_DUMP_DIR" with
  | none => pure ()
  | some dir =>
    -- Probe CUDA batch state: if currently batching, queue sync returns
    -- some; we flush + reopen only in that case.  Safe regardless of backend
    -- because endBatch on `none` is a no-op.
    let wasBatching ← Hesper.Backend.isCudaBatching
    GPUBackend.endBatch ctx
    let data ← GPUBackend.readBuffer ctx buf bytes
    IO.FS.writeBinFile s!"{dir}/{suffix}.bin" data
    if wasBatching then
      GPUBackend.beginBatch ctx

/-- Write a small scalar (≤8 bytes) to a device buffer via a pinned-host
    staging slot.  Safe inside CUDA Graph capture: the resulting memcpy
    node holds a stable host pointer.  Outside capture it is identical
    in effect to `writeBufferOffset` (just uses pinned host memory as
    the source).

    * `ctx`        — backend context
    * `dstBuf`     — device buffer
    * `dstOffset`  — byte offset into `dstBuf`
    * `staging`    — `USize` pinned-host pointer allocated at state init
    * `stOffset`   — byte offset inside the staging slot (usually 0)
    * `data`       — the bytes to write
 -/
def writeScalarViaStaging [GPUBackend β] (ctx : β)
    (dstBuf : GPUBackend.Buf β) (dstOffset : USize)
    (staging : USize) (stOffset : USize)
    (data : ByteArray) : IO Unit := do
  if staging == 0 then
    -- No pinned slot (e.g. WebGPU, or CUDA_GRAPHS disabled) — fall back.
    GPUBackend.writeBufferOffset ctx dstBuf dstOffset data
  else
    Hesper.CUDA.cuWritePinned staging stOffset data data.size.toUSize
    match ← Hesper.cudaCaptureStream.get with
    | some s =>
      -- Pull the raw device pointer via the backend extension.
      match ← GPUBackend.rawDevicePtr ctx dstBuf with
      | some ptr =>
        Hesper.CUDA.cuMemcpyHtoDFromPinned
          (ptr + dstOffset) staging stOffset data.size.toUSize s
      | none =>
        -- Backend can't expose a raw ptr; fall back to the normal path.
        GPUBackend.writeBufferOffset ctx dstBuf dstOffset data
    | none =>
      -- Not capturing — just plain writeBufferOffset is fine.
      GPUBackend.writeBufferOffset ctx dstBuf dstOffset data

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
    match kcr with
    | some k =>
      let key := hash ("gemma4_ce", name, config.numWorkgroups, config.workgroupSize.x, config.workgroupSize.y, config.workgroupSize.z)
      let ref ← k.getRef key
      GPUBackend.executeWithConfigCached ctx shader namedBufs config key ref
    | none => GPUBackend.execute ctx shader namedBufs config

  -- Step 1+2: Fused attnNorm + Q/K/V projections
  -- Fuses RMSNorm into each matmul: each WG computes RMS on-the-fly (redundant but cheap).
  -- Eliminates the normedBuf global memory write/read round-trip and the attnNorm dispatch.
  -- Local helper: a Gemma RMSNorm via the Circuit DSL.  Builds 4 ops
  -- (reduce + 3 pointwise) which fuseReduceEpilogue collapses to one
  -- dispatch, matching the hand-written `RMSNorm.forward` baseline
  -- but with the kernel generated from ScalarExp instead of being a
  -- hand-maintained ShaderM.  Reuse for any 1D-row RMSNorm site.
  let circuitRMSNorm := fun (tag : String)
      (norm : RMSNorm.RMSNorm (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
      (inB outB : GPUBackend.Buf β) => do
    let key := hash ("circuitRMSNorm-cuda", tag, norm.config.dim, li)
    let ccRef ← Hesper.Circuit.getGlobalCircuitRef (β := β) key
    Hesper.Circuit.runCachedFused ctx ccRef
      (do
        let xT ← Hesper.Circuit.CircuitM.registerExternal
          (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
          inB #[norm.config.dim] .f32 .Global
        let sT ← Hesper.Circuit.CircuitM.registerExternal
          (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
          norm.scale #[norm.config.dim] .f32 .Global
        let _y ← Hesper.Circuit.CircuitM.rmsNorm xT sT norm.config.eps
        pure ())
      [(0, inB), (1, norm.scale), (5, outB)]

  -- Step 1+2 (combined): Attention pre-norm + Q/K/V projections.
  --
  -- The fused path collapses the standalone RMSNorm dispatch INTO the
  -- Q8_1 quantize step of the QKV pipeline (`forwardFusedNormQKV`).
  -- That eliminates the f32 normedBuf round-trip to VRAM (~10 KB/layer)
  -- AND saves one dispatch per layer (4 → 3).  Preconditions: all
  -- three Q/K/V projections Q4_K + inDim divisible by 256 (for dp4a).
  --
  -- Falls back to the prior 4-dispatch sequence otherwise: standalone
  -- RMSNorm via Circuit DSL, then `forwardFusedQKV` reading normedBuf.
  let useFusedQKV := cfg.hasKV li
                  && block.attention.wQ.quantFormat == .Q4_K
                  && block.attention.wK.quantFormat == .Q4_K
                  && block.attention.wV.quantFormat == .Q4_K
                  && block.attention.wK.config.inDim == block.attention.wQ.config.inDim
                  && block.attention.wV.config.inDim == block.attention.wQ.config.inDim
                  && block.attention.wK.config.outDim == block.attention.wV.config.outDim
  let useFusedNormQKV := useFusedQKV
                      && block.attention.wQ.config.inDim == block.attnNorm.config.dim
                      && block.attention.wQ.config.inDim % 256 == 0
  if useFusedNormQKV then
    Hesper.WGSL.Execute.withSection "attnNormQKV" do
      let key := hash ("qkvFusedNormDP4A",
        block.attention.wQ.config.inDim, block.attention.wQ.config.outDim,
        block.attention.wK.config.outDim)
      let kvRef ← match kcr with
        | some k => k.getRef key
        | none => IO.mkRef none
      Linear.forwardFusedNormQKV ctx block.attnNorm
        block.attention.wQ block.attention.wK block.attention.wV
        inputBuf state.qBuf state.kBuf state.vBuf kvRef
  else do
    -- Standalone attnNorm via Circuit DSL.
    Hesper.WGSL.Execute.withSection "attnNorm" do
      circuitRMSNorm "attnNorm" block.attnNorm inputBuf state.normedBuf
    Hesper.WGSL.Execute.withSection "qkvProj" do
      if cfg.hasKV li then
        if useFusedQKV then
          let key := hash ("qkvFusedDP4A",
            block.attention.wQ.config.inDim, block.attention.wQ.config.outDim,
            block.attention.wK.config.outDim)
          let kvRef ← match kcr with
            | some k => k.getRef key
            | none => IO.mkRef none
          Linear.forwardFusedQKV ctx block.attention.wQ block.attention.wK block.attention.wV
            state.normedBuf state.qBuf state.kBuf state.vBuf kvRef
        else
          -- Circuit-DSL: three Q4_K matmuls sharing one input.  Built once,
          -- then `Hesper.Circuit.runCachedFused` runs the Stage 2
          -- `mergeSameDispatch` pass before lowering.  The pass detects
          -- the [matmul wK; matmul wV] pair (same input + same shape) and
          -- merges them into one fusedKV op, mechanically reproducing what
          -- our hand-written `forwardFusedKV` does.
          let key3 := hash ("circuitQKV-fused-cuda", block.attention.wQ.config.inDim,
                            block.attention.wQ.config.outDim,
                            block.attention.wK.config.outDim, li)
          let ccRef3 ← Hesper.Circuit.getGlobalCircuitRef (β := β) key3
          Hesper.Circuit.runCachedFused ctx ccRef3
            (do
              let normed ← Hesper.Circuit.CircuitM.registerExternal
                (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
                state.normedBuf #[cfg.hiddenSize] .f32 .Global
              let _q ← Hesper.Circuit.CircuitM.matmulQ4K normed block.attention.wQ
              let _k ← Hesper.Circuit.CircuitM.matmulQ4K normed block.attention.wK
              let _v ← Hesper.Circuit.CircuitM.matmulQ4K normed block.attention.wV
              pure ())
            [(0, state.normedBuf), (1, state.qBuf), (2, state.kBuf), (3, state.vBuf)]
      else
        -- No-KV layer: wQ only, via Circuit DSL.  `runCached` builds
        -- the Circuit ONCE per (inDim, outDim, backend), caches the
        -- compiled artifact, and replays.
        let key := hash ("circuitWQ-cuda", block.attention.wQ.config.inDim, block.attention.wQ.config.outDim, li)
        let ccRef ← Hesper.Circuit.getGlobalCircuitRef (β := β) key
        Hesper.Circuit.runCached ctx ccRef
          (do
            let normed ← Hesper.Circuit.CircuitM.registerExternal
              (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
              state.normedBuf #[cfg.hiddenSize] .f32 .Global
            let _q ← Hesper.Circuit.CircuitM.matmulQ4K normed block.attention.wQ
            pure ())
          [(0, state.normedBuf), (1, state.qBuf)]

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
    -- When this layer has its own KV, fuse the three per-head norms
    -- (qNorm, kNorm, vNorm) into a single dispatch.  Grid is
    -- `(numHeads, 3, 1)`; `wg_id.y` picks Q/K/V; WGs with
    -- `wg_id.y > 0 && wg_id.x >= numKVHeads` early-return.  Saves 2
    -- dispatches per layer per token.
    --
    -- When the layer shares KV with an earlier block, only qNorm runs
    -- — keep the existing single-dispatch path for that case.
    if cfg.hasKV li then
      ce (if isFull then "qkvNormFull" else "qkvNormSWA")
        (fusedPerHeadQKVNormKernel numHeads numKVHeads headDim cfg.rmsNormEps)
        [("q_in", state.qBuf), ("q_scale", block.attention.qNormWeight), ("q_out", state.qBuf2),
         ("k_in", state.kBuf), ("k_scale", block.attention.kNormWeight), ("k_out", state.kBuf2),
         ("v_in", state.vBuf),                                              ("v_out", state.vBuf2)]
        { numWorkgroups := (numHeads, 3, 1),
          workgroupSize := { x := wgSize, y := 1, z := 1 } : Hesper.ExecConfig }
    else
      ce (if isFull then "qNormFull" else "qNormSWA")
        (perHeadRMSNormKernel numHeads headDim cfg.rmsNormEps)
        [("input", state.qBuf), ("weight", block.attention.qNormWeight), ("output", state.qBuf2)]
        (mkNormConfig numHeads)

  -- Step 4: RoPE on Q and K
  -- Upload position to params buffer (u32 for hand-coded kernels)
  let posBytes := Hesper.WebGPU.BufferOps.uint32ToBytes pos.toUInt32
  writeScalarViaStaging ctx state.paramsBuf 0 state.stagingParamsPtr 0 posBytes
  -- Also upload pos as f32 for Circuit DSL scatter addrExpr.  Routed
  -- through a pinned host slot so CUDA Graph replay picks up the
  -- current pos (not a stale captured value).
  let posF32Bytes ← Hesper.Basic.floatToBytes pos.toFloat
  writeScalarViaStaging ctx state.posF32Buf 0 state.stagingPosF32Ptr 0 posF32Bytes

  Hesper.WGSL.Execute.withSection "rope" do
    -- RoPE on Q: qBuf2 → qBuf
    match block.ropeFreqFactors with
    | some freqFactors =>
      ce s!"ropeFreqQ_{headDim}"
        (ropeWithFreqFactorsKernel headDim numHeads cfg.ropeTheta)
        [("input", state.qBuf2), ("output", state.qBuf), ("params", state.paramsBuf), ("freq_factors", freqFactors)]
        (.dispatch1D (numHeads * headDim / 2))
    | none =>
      let ropeConfig : RoPE.Config := { dim := numHeads * headDim, maxSeqLen := cfg.maxSeqLen, base := cfg.ropeTheta }
      ce s!"ropeDynQ_{headDim}"
        (RoPE.ropeKernelDynamic ropeConfig 1 1 numHeads headDim)
        [("input", state.qBuf2), ("output", state.qBuf), ("params", state.paramsBuf)]
        (.dispatch1D (numHeads * headDim / 2))

    -- ropeK is fused with KV cache write below (when ropeFreqFactors are available
    -- and we have a KV cache).  When freq factors aren't present we fall back to
    -- the legacy two-kernel path.
    if cfg.hasKV li && block.ropeFreqFactors.isNone then
      let ropeConfig : RoPE.Config := { dim := numKVHeads * headDim, maxSeqLen := cfg.maxSeqLen, base := cfg.ropeTheta }
      ce s!"ropeDynK_{headDim}_{numKVHeads}"
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

    -- Write K and V to cache at current position.  When ropeFreqFactors are
    -- available (Gemma 4 default), we use the fused RoPE-K + KV-write kernel
    -- that takes K *before* RoPE (kBuf2) and applies the rotation in-kernel,
    -- saving the separate ropeK dispatch above.
    if cfg.hasKV li then
      Hesper.WGSL.Execute.withSection "kvWrite" do
        let useScatter ← match ← IO.getEnv "HESPER_SCATTER_KV" with
                        | some "1" => pure true
                        | _        => pure false
        match block.ropeFreqFactors, useScatter with
        | some freqFactors, false =>
          -- Default path: single fused hand-coded RoPE-K + KV-write kernel.
          ce s!"ropeKAndKvWrite_{headDim}_{numKVHeads}"
            (Attention.fusedRopeKAndCacheWriteKernel numKVHeads cfg.maxSeqLen headDim kvDim cfg.ropeTheta)
            [("new_k", state.kBuf2), ("new_v", state.vBuf2),
             ("k_cache", kvCache.kBuf), ("v_cache", kvCache.vBuf),
             ("params", state.paramsBuf), ("freq_factors", freqFactors)]
            (.dispatch1D kvDim)
        | some freqFactors, true =>
          -- Circuit DSL path: ONE scatterMulti dispatch writes K (with NeoX RoPE)
          -- AND V (plain copy) into the cache.  Same semantics and same kernel
          -- count as the hand-coded fusedRopeKAndCacheWriteKernel.
          --
          -- Inputs (4 data + 2 dst):
          --   inputs[0] = state.kBuf2   (lane-local, [kvDim])
          --   inputs[1] = freqFactors    (gather-only, [halfDim])
          --   inputs[2] = state.posF32Buf (broadcast scalar, [1])
          --   inputs[3] = state.vBuf2   (lane-local, [kvDim])
          --   inputs[4] = kvCache.kBuf  (dst 0)
          --   inputs[5] = kvCache.vBuf  (dst 1)
          let halfDim := headDim / 2
          let cacheSize := numKVHeads * cfg.maxSeqLen * headDim
          let key := hash ("gemma4-scatter-multi-kv",
                            li, numKVHeads, cfg.maxSeqLen, headDim)
          let ccRef ← Hesper.Circuit.getGlobalCircuitRef (β := β) key
          Hesper.Circuit.runCachedFused ctx ccRef
            (do
              let kT ← Hesper.Circuit.CircuitM.registerExternal
                         (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
                         state.kBuf2 #[kvDim] .f32 .Global
              let ffT ← Hesper.Circuit.CircuitM.registerExternal
                          (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
                          freqFactors #[halfDim] .f32 .Global
              let posT ← Hesper.Circuit.CircuitM.registerExternal
                           (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
                           state.posF32Buf #[1] .f32 .Global
              let vT ← Hesper.Circuit.CircuitM.registerExternal
                         (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
                         state.vBuf2 #[kvDim] .f32 .Global
              let kDst ← Hesper.Circuit.CircuitM.registerExternal
                           (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
                           kvCache.kBuf #[cacheSize] .f32 .Global
              let vDst ← Hesper.Circuit.CircuitM.registerExternal
                           (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
                           kvCache.vBuf #[cacheSize] .f32 .Global
              -- Shared addr arithmetic.
              let i        : Hesper.Circuit.ScalarExp := .laneIdx
              let headSE   : Hesper.Circuit.ScalarExp := .idiv i (.const headDim.toFloat)
              let d        : Hesper.Circuit.ScalarExp := .mod  i (.const headDim.toFloat)
              let posSE    : Hesper.Circuit.ScalarExp := .input 2
              let addrExpr : Hesper.Circuit.ScalarExp :=
                headSE * .const (cfg.maxSeqLen * headDim).toFloat
                + posSE * .const headDim.toFloat
                + d
              -- K value: NeoX RoPE.
              let dLow     : Hesper.Circuit.ScalarExp := .lt d (.const halfDim.toFloat)
              let pairD    : Hesper.Circuit.ScalarExp :=
                .select dLow (d + .const halfDim.toFloat) (d - .const halfDim.toFloat)
              let pairIdx  : Hesper.Circuit.ScalarExp :=
                headSE * .const headDim.toFloat + pairD
              let xSelf    : Hesper.Circuit.ScalarExp := .input 0
              let xPair    : Hesper.Circuit.ScalarExp := .indexed 0 pairIdx
              let dimPair  : Hesper.Circuit.ScalarExp :=
                .select dLow d (d - .const halfDim.toFloat)
              let freqFac  : Hesper.Circuit.ScalarExp := .indexed 1 dimPair
              let exponent : Hesper.Circuit.ScalarExp :=
                .const 2.0 * dimPair / .const headDim.toFloat
              let freqInv  : Hesper.Circuit.ScalarExp := .pow (.const cfg.ropeTheta) (.neg exponent)
              let theta    : Hesper.Circuit.ScalarExp := posSE * freqInv / freqFac
              let cosT     : Hesper.Circuit.ScalarExp := .cos theta
              let sinT     : Hesper.Circuit.ScalarExp := .sin theta
              let x0       : Hesper.Circuit.ScalarExp := .select dLow xSelf xPair
              let x1       : Hesper.Circuit.ScalarExp := .select dLow xPair xSelf
              let x0new    : Hesper.Circuit.ScalarExp := x0 * cosT - x1 * sinT
              let x1new    : Hesper.Circuit.ScalarExp := x0 * sinT + x1 * cosT
              let kValue   : Hesper.Circuit.ScalarExp := .select dLow x0new x1new
              -- V value: plain copy from inputs[3].
              let vValue   : Hesper.Circuit.ScalarExp := .input 3
              let _ ← Hesper.Circuit.CircuitM.scatterMulti #[kvDim]
                        #[kT, ffT, posT, vT]
                        #[kDst, vDst]
                        #[(kValue, addrExpr), (vValue, addrExpr)]
              pure ())
            [(0, state.kBuf2), (1, freqFactors), (2, state.posF32Buf),
             (3, state.vBuf2), (4, kvCache.kBuf), (5, kvCache.vBuf)]
        | none, _ =>
          -- No freqFactors: fall back to the plain K+V copy hand-coded kernel.
          ce s!"kvWrite_{headDim}_{numKVHeads}"
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
    -- Write cacheLen to params buffer for FlashAttention (params = [pos, cacheLen])
    let cacheLenBytes := Hesper.WebGPU.BufferOps.uint32ToBytes cacheLen.toUInt32
    writeScalarViaStaging ctx state.paramsBuf 4 state.stagingParamsPtr 4 cacheLenBytes
    Hesper.WGSL.Execute.withSection "flashAttn" do
      -- Two kernels:
      --   cacheLen > 32: tiled split-K (phase1 + phase2) — parallel
      --     accumulation across KV dimension wins for long caches.
      --   cacheLen ≤ 32: subgroup params kernel (HESPER_FA_SUBGROUP=1)
      --     or the 256-thread tree-reduce `Dynamic` kernel (default).
      --     The subgroup kernel uses 32 threads with a single
      --     `subgroupAdd` per position — no shared memory, no
      --     barriers — and reads cacheLen from `params` so the PTX is
      --     cacheable (works correctly under CUDA Graph capture +
      --     replay past the initial ≤32 capture boundary).
      --
      -- Measured on Gemma 4 E4B Q4_K_M + RTX 4070 Ti: both ≤32 kernels
      -- land within noise for typical 100-tok decode.  Keeping the
      -- older default to avoid touching the long-context path; opt-in
      -- via env for benchmarking / future tuning.
      --
      -- SWA masking isn't needed: cacheLen is already clamped to
      -- ≤ windowSize for SWA layers upstream.
      if cacheLen > 32 then
        FlashAttention.executeFlashAttentionTiled ctx
          state.qBuf kvCache.kBuf kvCache.vBuf state.attnOutBuf
          numHeads numKVHeads cfg.maxSeqLen headDim cacheLen scale
          (partialBuf := some state.flashPartialBuf)
      else
        let useSubgroupFA := (match ← IO.getEnv "HESPER_FA_SUBGROUP" with
                             | some "1" => true
                             | _        => false)
        if useSubgroupFA then
          ce s!"flashAttnS_{headDim}_{numKVHeads}"
            (FlashAttention.flashAttentionSubgroupParamsKernel numHeads numKVHeads cfg.maxSeqLen headDim scale)
            [("q", state.qBuf), ("k_cache", kvCache.kBuf), ("v_cache", kvCache.vBuf),
             ("output", state.attnOutBuf), ("params", state.paramsBuf)]
            ({ numWorkgroups := (numHeads, 1, 1), workgroupSize := { x := 32, y := 1, z := 1 }
               extensions := ["subgroups"] : Hesper.ExecConfig })
        else
          ce s!"flashAttnP_{headDim}_{numKVHeads}"
            (FlashAttention.flashAttentionDynamicParamsKernel numHeads numKVHeads cfg.maxSeqLen headDim scale)
            [("q", state.qBuf), ("k_cache", kvCache.kBuf), ("v_cache", kvCache.vBuf),
             ("output", state.attnOutBuf), ("params", state.paramsBuf)]
            ({ numWorkgroups := (numHeads, 1, 1) : Hesper.ExecConfig })

    -- Output projection: attnOut [numHeads * headDim] → normedBuf [hiddenSize]
    -- Circuit-DSL: single matmulQ4K op via runCached (build once, replay).
    -- Equivalent to direct LinearLayer.forward; sets up the IR for later
    -- fusion with the post-attn norm chain.
    Hesper.WGSL.Execute.withSection "oProj" do
      let keyO := hash ("circuitWO-cuda", block.attention.wO.config.inDim,
                        block.attention.wO.config.outDim, li)
      let ccRefO ← Hesper.Circuit.getGlobalCircuitRef (β := β) keyO
      Hesper.Circuit.runCached ctx ccRefO
        (do
          let attnOut ← Hesper.Circuit.CircuitM.registerExternal
            (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
            state.attnOutBuf #[block.attention.wO.config.inDim] .f32 .Global
          let _o ← Hesper.Circuit.CircuitM.matmulQ4K attnOut block.attention.wO
          pure ())
        [(0, state.attnOutBuf), (1, state.normedBuf)]
  else
    -- Fallback: skip attention (shouldn't happen)
    Linear.LinearLayer.forward ctx block.attention.wO state.qBuf state.normedBuf

  -- Step 6: Post-attention norm + residual.  Gemma 4 is post-norm:
  -- `attnResidualBuf = RMSNorm(attn_out) * scale + inputBuf`.  Fused into
  -- a single kernel via forwardNormThenAdd to save one dispatch.
  Hesper.WGSL.Execute.withSection "postAttnNorm" do
    let key := hash ("postAttnNormAdd", cfg.hiddenSize)
    let ref ← match kcr with
      | some k => k.getRef key
      | none => IO.mkRef none
    RMSNorm.forwardNormThenAdd ctx block.postAttnNorm
      state.normedBuf inputBuf state.attnResidualBuf ref
  dumpBuf ctx state.attnResidualBuf (cfg.hiddenSize * 4).toUSize s!"single_p{pos}_postAttn_L{li}"

  -- Step 7: FFN (dense or MoE)
  if block.isMoE then do
    -- MoE layer (from gemma4-iswa.cpp:117-169):
    -- 1. Shared expert: ffn_norm → GeGLU FFN → post_norm_1
    RMSNorm.forward ctx block.ffnNorm state.attnResidualBuf state.normedBuf
    Linear.LinearLayer.forward ctx block.ffn.gate state.normedBuf state.gateBuf
    Linear.LinearLayer.forward ctx block.ffn.up state.normedBuf state.upBuf
    ce "geluMul"
      (geluMulKernel cfg.intermediateSize)
      [("gate", state.gateBuf), ("up", state.upBuf), ("output", state.geluBuf)]
      (.dispatch1D cfg.intermediateSize)
    Linear.LinearLayer.forward ctx block.ffn.down state.geluBuf state.ffnOutBuf

    -- Apply post_norm_1 to shared expert output (avoid aliasing)
    match block.moePostNorm1 with
    | some norm =>
      RMSNorm.forward ctx norm state.ffnOutBuf state.normedBuf2
      -- Copy back: normedBuf2 → ffnOutBuf
      ce "pleScale1"
        (PerLayerEmbedding.scaleKernel cfg.hiddenSize 1.0)
        [("input", state.normedBuf2), ("output", state.ffnOutBuf)]
        (.dispatch1D cfg.hiddenSize)
    | none => pure ()

    -- 2. Router: rms_norm(attn_out) * (1/sqrt(n_embd)) * router_scale → logits → softmax → top-K
    match block.moeRouterWeight, block.moeRouterScale with
    | some routerW, some routerS =>
      ce "moeRouterPre"
        (MoE.routerPreprocessKernel cfg.hiddenSize cfg.rmsNormEps)
        [("input", state.attnResidualBuf), ("router_scale", routerS), ("output", state.moeRouterOutBuf)]
        ({ numWorkgroups := (1, 1, 1) : Hesper.ExecConfig })
      -- Router matmul: moeRouterOutBuf [hiddenSize] @ routerW^T → moeLogitsBuf [numExperts]
      let routerMatmulConfig : Hesper.WGSL.MatMul.Config := {
        M := 1, N := cfg.numExperts, K := cfg.hiddenSize
      }
      Hesper.WGSL.MatMul.executeMatMulTranspose ctx state.moeRouterOutBuf routerW state.moeLogitsBuf routerMatmulConfig
      -- Top-K selection
      ce "moeSoftmaxTopK"
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
      ce "residAddZero"
        (residualAddKernel cfg.hiddenSize)  -- hack: 0 + 0 = 0 (both inputs are same zeroed buf)
        [("a", state.moeExpertOutBuf), ("b", state.moeExpertOutBuf), ("output", state.moeExpertOutBuf)]
        (.dispatch1D cfg.hiddenSize)
      -- Actually zero it properly
      ce "embedScaleZero"
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
        ce s!"moeGate_{k}"
          (MoE.expertGateUpKernel moeConfig k true)
          [("input", state.moeNormedBuf), ("gate_up_weights", gateUpExps),
           ("expert_indices", state.moeIndicesBuf), ("output", state.moeExpertGateBuf)]
          ({ numWorkgroups := (cfg.expertFFSize, 1, 1) : Hesper.ExecConfig })
        ce s!"moeUp_{k}"
          (MoE.expertGateUpKernel moeConfig k false)
          [("input", state.moeNormedBuf), ("gate_up_weights", gateUpExps),
           ("expert_indices", state.moeIndicesBuf), ("output", state.moeExpertUpBuf)]
          ({ numWorkgroups := (cfg.expertFFSize, 1, 1) : Hesper.ExecConfig })
        ce "moeExpertGelu"
          (MoE.expertGeluMulKernel cfg.expertFFSize)
          [("gate", state.moeExpertGateBuf), ("up", state.moeExpertUpBuf), ("output", state.moeExpertGeluBuf)]
          (.dispatch1D cfg.expertFFSize)
        ce s!"moeDown_{k}"
          (MoE.expertDownKernel moeConfig k)
          [("input", state.moeExpertGeluBuf), ("down_weights", downExps),
           ("expert_indices", state.moeIndicesBuf), ("output", state.moeExpertDownBuf)]
          ({ numWorkgroups := (cfg.hiddenSize, 1, 1) : Hesper.ExecConfig })
        ce s!"moeAccum_{k}"
          (MoE.weightedAccumulateKernel cfg.hiddenSize cfg.numExpertsUsed k)
          [("accumulator", state.moeExpertOutBuf), ("expert_output", state.moeExpertDownBuf),
           ("weights", state.moeWeightsBuf)]
          (.dispatch1D cfg.hiddenSize)

      -- post_norm_2 on routed expert output
      -- Avoid aliasing: moeExpertOutBuf → normedBuf2 → moeExpertOutBuf
      RMSNorm.forward ctx postNorm2 state.moeExpertOutBuf state.normedBuf2
      ce "pleScale2"
        (PerLayerEmbedding.scaleKernel cfg.hiddenSize 1.0)
        [("input", state.normedBuf2), ("output", state.moeExpertOutBuf)]
        (.dispatch1D cfg.hiddenSize)

      -- 4. Combined: shared_expert + routed_experts
      ce "residAddMoePost"
        (residualAddKernel cfg.hiddenSize)
        [("a", state.ffnOutBuf), ("b", state.moeExpertOutBuf), ("output", state.ffnOutBuf)]
        (.dispatch1D cfg.hiddenSize)
    | _, _, _, _ => pure ()  -- No MoE weights: shared expert only

    -- Post-FFN norm + residual, fused: output = RMSNorm(ffn_out) * scale + attn_residual.
    let keyFFN := hash ("postFFNNormAdd", cfg.hiddenSize)
    let refFFN ← match kcr with
      | some k => k.getRef keyFFN
      | none => IO.mkRef none
    RMSNorm.forwardNormThenAdd ctx block.postFFNNorm
      state.ffnOutBuf state.attnResidualBuf outputBuf refFFN
  else do
    -- Dense FFN path (GeGLU).
    -- The fused-norm path collapses ffnNorm + Q8_1 + gate+up into 2
    -- dispatches (vs unfused 3): one fused norm-quantize, one fused
    -- gate+up GeGLU matmul that consumes the Q8_1 buffer.  Eliminates
    -- the f32 normedBuf round-trip AND the standalone ffnNorm dispatch.
    -- A/B confirmed 2026-04-16: fused path is 5.6 TPS faster than the
    -- unfused 2-matmul+geluMul alternative (148 µs/layer vs 122 µs/layer
    -- for the heavy kernel).
    let useFused := block.ffn.gate.quantFormat == .Q4_K
                  && block.ffn.up.quantFormat == .Q4_K
                  && block.ffn.gate.config.inDim == block.ffn.up.config.inDim
                  && block.ffn.gate.config.outDim == block.ffn.up.config.outDim
    let useFusedNorm := useFused
                     && block.ffn.gate.config.inDim == block.ffnNorm.config.dim
                     && block.ffn.gate.config.inDim % 256 == 0
    if useFusedNorm then
      Hesper.WGSL.Execute.withSection "ffnNormGateUp" do
        let key := hash ("ffnGateUpFusedNormDP4A",
          block.ffn.gate.config.inDim, block.ffn.gate.config.outDim)
        let ref ← match kcr with
          | some k => k.getRef key
          | none => IO.mkRef none
        Linear.forwardFusedNormGateUp ctx block.ffnNorm
          block.ffn.gate block.ffn.up
          state.attnResidualBuf state.geluBuf ref
    else do
      Hesper.WGSL.Execute.withSection "ffnNorm" do
        circuitRMSNorm "ffnNorm" block.ffnNorm state.attnResidualBuf state.normedBuf
      Hesper.WGSL.Execute.withSection "ffnGateUpMul" do
        if useFused then
          let key := hash ("ffnGateUpDP4A", block.ffn.gate.config.inDim, block.ffn.gate.config.outDim)
          let ref ← match kcr with
            | some k => k.getRef key
            | none => IO.mkRef none
          Linear.forwardFusedGateUp ctx block.ffn.gate block.ffn.up
            state.normedBuf state.geluBuf ref
        else
          Linear.LinearLayer.forward ctx block.ffn.gate state.normedBuf state.gateBuf
          Linear.LinearLayer.forward ctx block.ffn.up state.normedBuf state.upBuf
          ce "geluMul2"
            (geluMulKernel cfg.intermediateSize)
            [("gate", state.gateBuf), ("up", state.upBuf), ("output", state.geluBuf)]
            (.dispatch1D cfg.intermediateSize)
    -- ffn.down: gelu*up [intermediateSize] → ffnOut [hiddenSize].  Same
    -- Circuit DSL pattern as wO above.
    Hesper.WGSL.Execute.withSection "ffnDown" do
      let keyFD := hash ("circuitFFNDown-cuda", block.ffn.down.config.inDim,
                         block.ffn.down.config.outDim, li)
      let ccRefFD ← Hesper.Circuit.getGlobalCircuitRef (β := β) keyFD
      Hesper.Circuit.runCached ctx ccRefFD
        (do
          let gelu ← Hesper.Circuit.CircuitM.registerExternal
            (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
            state.geluBuf #[block.ffn.down.config.inDim] .f32 .Global
          let _o ← Hesper.Circuit.CircuitM.matmulQ4K gelu block.ffn.down
          pure ())
        [(0, state.geluBuf), (1, state.ffnOutBuf)]

    -- Post-FFN norm + residual, fused.  output = RMSNorm(ffn_out) * scale + attn_residual.
    Hesper.WGSL.Execute.withSection "postFFNNorm" do
      let keyFFN2 := hash ("postFFNNormAdd", cfg.hiddenSize)
      let refFFN2 ← match kcr with
        | some k => k.getRef keyFFN2
        | none => IO.mkRef none
      RMSNorm.forwardNormThenAdd ctx block.postFFNNorm
        state.ffnOutBuf state.attnResidualBuf outputBuf refFFN2
  dumpBuf ctx outputBuf (cfg.hiddenSize * 4).toUSize s!"single_p{pos}_postFFN_L{li}"

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
      let plOffset := li * cfg.embdPerLayer
      let plTotalSize := cfg.embdPerLayer * cfg.numHiddenLayers
      -- Fuse `ple.inpGate` matmul + `ple.geluGateMul` into one dispatch
      -- pair (Q8_1 quantize + fused matmul-with-GELU-slice-mul epilogue).
      -- Saves 1 dispatch per PLE site.  Falls back to the 2-step path
      -- when preconditions fail.
      let useFusedPLGate :=
        plEmbd.inpGate.quantFormat == .Q4_K &&
        plEmbd.inpGate.config.inDim % 256 == 0
      if useFusedPLGate then
        -- Circuit-DSL: one generic `Prim.matmulQ4KWithEpilogue` node
        -- carries the PLE matmul + GELU + slice-mul tail.  Lowering
        -- emits (Q8_1 quantize dispatch) + (fused matmul-epilogue
        -- kernel) — same two dispatches as the prior hand-composed
        -- `forwardFusedPLInpGate`, but from the IR rather than a
        -- duplicated ShaderM kernel.
        --
        -- Epilogue body: `gelu(input 0) * input 1` where
        --   input 0 = matmul dot product (per-row)
        --   input 1 = per_layer_input[plOffset + outIdx]
        Hesper.WGSL.Execute.withSection "ple.inpGateGeluSlice" do
          let key := hash ("circuitPLEInpGateGeluSlice",
            plEmbd.inpGate.config.inDim, plEmbd.inpGate.config.outDim, plOffset)
          let ccRef ← Hesper.Circuit.getGlobalCircuitRef (β := β) key
          Hesper.Circuit.runCached ctx ccRef
            (do
              let x ← Hesper.Circuit.CircuitM.registerExternal
                (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
                outputBuf #[plEmbd.inpGate.config.inDim] .f32 .Global
              let plAll ← Hesper.Circuit.CircuitM.registerExternal
                (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
                plInputAll #[plTotalSize] .f32 .Global
              -- gelu(input 0) * input 1  via tanh approximation.
              let x0 : Hesper.Circuit.ScalarExp := .input 0
              let x3 := .mul (.mul x0 x0) x0
              let inner :=
                .mul (.const 0.7978845608028654)  -- sqrt(2/π)
                     (.add x0 (.mul (.const 0.044715) x3))
              let gelu :=
                .mul (.mul (.const 0.5) x0)
                     (.add (.const 1.0) (.tanh inner))
              let body : Hesper.Circuit.ScalarExp :=
                .mul gelu (.input 1)
              let _out ← Hesper.Circuit.CircuitM.matmulQ4KWithEpilogue
                x plEmbd.inpGate #[plAll] body (epiReadOffsets := #[plOffset])
              pure ())
            -- Tensor ids: 0 = outputBuf external, 1 = plInputAll external,
            -- 2 = matmul-epi output (caller-facing).
            [(0, outputBuf), (1, plInputAll), (2, state.moeRouterOutBuf)]
      else do
        Hesper.WGSL.Execute.withSection "ple.inpGate" do
          Linear.LinearLayer.forward ctx plEmbd.inpGate outputBuf state.plGateBuf
        Hesper.WGSL.Execute.withSection "ple.geluGateMul" do
          ce s!"pleGeluGateMul_{plOffset}"
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
        ce "fusedPLPost"
          (fusedPerLayerPostKernel cfg.hiddenSize cfg.rmsNormEps)
          [("proj", state.plProjBuf), ("weight", plEmbd.postNorm.scale), ("residual", outputBuf)]
          { numWorkgroups := (1, 1, 1)
            workgroupSize := { x := 256, y := 1, z := 1 }
            extensions := ["subgroups"]
            : Hesper.ExecConfig }
    | _, _ => pure ()
  dumpBuf ctx outputBuf (cfg.hiddenSize * 4).toUSize s!"single_p{pos}_postPLE_L{li}"

  -- Step 9: Layer output scale (optional).  Was two dispatches:
  --   layerScale: normedBuf2[i] = outputBuf[i] * scale[0]  (broadcast)
  --   pleScale3:  outputBuf[i]  = normedBuf2[i] * 1.0       (copy-back)
  -- Now lowered through the Circuit DSL: fusePointwise collapses the
  -- chain into a single dispatch whose body is `(outputBuf[i] * scale[0]) * 1`.
  -- The normedBuf2 round-trip is gone; outputBuf is consumed + written
  -- in one kernel (safe because every lane's read precedes its write).
  let skipOutScaleSingle := (← IO.getEnv "HESPER_SKIP_OUTSCALE").isSome
  match if skipOutScaleSingle then none else block.outScale with
  | some scale =>
    Hesper.WGSL.Execute.withSection "layerOutScale" do
      let key := hash ("circuitLayerOutScale-cuda", cfg.hiddenSize, li)
      let ccRef ← Hesper.Circuit.getGlobalCircuitRef (β := β) key
      Hesper.Circuit.runCachedFused ctx ccRef
        (do
          let x ← Hesper.Circuit.CircuitM.registerExternal
                    (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
                    outputBuf #[cfg.hiddenSize] .f32 .Global
          let s ← Hesper.Circuit.CircuitM.registerExternal
                    (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
                    scale #[1] .f32 .Global
          let scaled ← Hesper.Circuit.CircuitM.scaleByBroadcast x s
          let _out   ← Hesper.Circuit.CircuitM.map scaled
                         (.mul (.input 0) (.const 1.0))
          pure ())
        -- ids: 0=x (outputBuf), 1=s (scale), 2=scaled (fused away),
        -- 3=final (written back to outputBuf).
        [(0, outputBuf), (1, scale), (3, outputBuf)]
  | none => pure ()

/-! ## Column-major helper kernels for batched prefill -/

/-- GPU-side u32 copy: `dst[dstIdx] = src[params[0]]`.
    srcIdx is read at runtime from `params[0]`.  dstIdx is compile-time
    (always 0 or 1 — only 2 unique kernels per (srcSize, dstSize, dstIdx)). -/
private def copyU32Kernel (srcSize : Nat) (dstSize : Nat) (dstIdx : Nat) : ShaderM Unit := do
  let _src    ← ShaderM.declareInputBuffer "src" (.array (.scalar .u32) srcSize)
  let _params ← ShaderM.declareInputBuffer "params" (.array (.scalar .u32) 1)
  let _dst    ← ShaderM.declareOutputBuffer "dst" (.array (.scalar .u32) dstSize)
  let srcIdx ← ShaderM.readBuffer (ty := .scalar .u32) (n := 1) "params" (Exp.litU32 0)
  let v ← ShaderM.readBuffer (ty := .scalar .u32) (n := srcSize) "src" srcIdx
  ShaderM.writeBuffer (ty := .scalar .u32) "dst" (Exp.litU32 dstIdx) v

/-- Copy column from a column-major batch buffer into a contiguous single-row
    buffer.  Column index is read at runtime from `params[0]` (u32).
    `batch[params[0] * dim + i] → out[i]` for i in [0, dim).
    One kernel JIT'd per (dim, seqLen) pair; colIdx changes via params only. -/
private def columnExtractKernel (dim : Nat) (seqLen : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let i := Exp.vec3X gid
  let totalBatch := dim * seqLen
  let _batch  ← ShaderM.declareInputBuffer "batch" (.array (.scalar .f32) totalBatch)
  let _params ← ShaderM.declareInputBuffer "params" (.array (.scalar .u32) 1)
  let _out    ← ShaderM.declareOutputBuffer "out" (.array (.scalar .f32) dim)
  ShaderM.if_ (Exp.lt i (Exp.litU32 dim)) (do
    let colIdx ← ShaderM.readBuffer (ty := .scalar .u32) (n := 1) "params" (Exp.litU32 0)
    let srcIdx := Exp.add (Exp.mul colIdx (Exp.litU32 dim)) i
    let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalBatch) "batch" srcIdx
    ShaderM.writeBuffer (ty := .scalar .f32) "out" i v
  ) (pure ())

/-- Copy a contiguous single-row buffer into a column of a column-major batch
    buffer.  Column index is read at runtime from `params[0]` (u32).
    `src[i] → batch[params[0] * dim + i]` for i in [0, dim). -/
private def columnInsertKernel (dim : Nat) (seqLen : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let i := Exp.vec3X gid
  let totalBatch := dim * seqLen
  let _src    ← ShaderM.declareInputBuffer "src" (.array (.scalar .f32) dim)
  let _params ← ShaderM.declareInputBuffer "params" (.array (.scalar .u32) 1)
  let _batch  ← ShaderM.declareOutputBuffer "batch" (.array (.scalar .f32) totalBatch)
  ShaderM.if_ (Exp.lt i (Exp.litU32 dim)) (do
    let colIdx ← ShaderM.readBuffer (ty := .scalar .u32) (n := 1) "params" (Exp.litU32 0)
    let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := dim) "src" i
    let dstIdx := Exp.add (Exp.mul colIdx (Exp.litU32 dim)) i
    ShaderM.writeBuffer (ty := .scalar .f32) "batch" dstIdx v
  ) (pure ())

/-! ## Batched Prefill -/

/-- Process all prompt tokens through the model in batch.
    Uses `forwardBatchDP4A` for Q4_K matmuls and `RMSNorm.forward` with
    `numRows` for batch RMSNorm.  Attention remains per-token (extract
    column, run single-token attention, write KV cache).

    Populates the KV caches for all prompt positions and leaves the last
    token's logits in `state.logitsBuf` so that decode can continue from
    position `promptTokens.size`. -/
def forwardPrefillBatch [GPUBackend β] (ctx : β)
    (model : Gemma4Model (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (promptTokens : Array Nat)
    (state : InferenceState (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (kcr : Option (KernelCacheRefs (GPUBackend.CachedDispatch β)) := none) : IO Unit := do
  let seqLen := promptTokens.size
  if seqLen == 0 then return
  -- NOTE: for seqLen == 1, callers should use `forwardSingleToken` directly.
  -- This function always takes the batch path (forwardBatchDP4A handles
  -- the seqLen=1 case internally by falling through to single-token dp4a).

  let cfg := model.config
  let dim := cfg.hiddenSize
  let interSize := cfg.intermediateSize
  let mkBuf := fun (n : Nat) => GPUBackend.allocBuffer ctx (n * 4).toUSize

  -- Cached execute helper (same pattern as forwardBlock / forwardSingleToken).
  let ce := fun (name : String) (shader : ShaderM Unit)
      (namedBufs : List (String × GPUBackend.Buf β)) (config : Hesper.ExecConfig) => do
    match kcr with
    | some k =>
      let key := hash ("gemma4_prefill_ce", name, config.numWorkgroups,
                        config.workgroupSize.x, config.workgroupSize.y, config.workgroupSize.z)
      let ref ← k.getRef key
      GPUBackend.executeWithConfigCached ctx shader namedBufs config key ref
    | none => GPUBackend.execute ctx shader namedBufs config

  -- ── Allocate prefill-sized batch buffers (column-major) ──────────────
  let batchBuf1 ← mkBuf (dim * seqLen)       -- ping-pong A
  let batchBuf2 ← mkBuf (dim * seqLen)       -- ping-pong B
  let batchNormedBuf ← mkBuf (dim * seqLen)  -- after attnNorm / ffnNorm
  let maxHeadDim := max cfg.headDimFull cfg.headDimSWA
  let maxQDim := cfg.numAttentionHeads * maxHeadDim
  let maxKVDim := (max cfg.numKeyValueHeadsFull cfg.numKeyValueHeadsSWA) * maxHeadDim
  let batchQBuf ← mkBuf (maxQDim * seqLen)
  let batchKBuf ← mkBuf (maxKVDim * seqLen)
  let batchVBuf ← mkBuf (maxKVDim * seqLen)
  let batchAttnOutBuf ← mkBuf (maxQDim * seqLen)
  let batchOProjBuf ← mkBuf (dim * seqLen)
  let batchAttnResidBuf ← mkBuf (dim * seqLen)
  let batchGateBuf ← mkBuf (interSize * seqLen)
  let batchUpBuf ← mkBuf (interSize * seqLen)
  let batchGeluBuf ← mkBuf (interSize * seqLen)
  let batchFFNOutBuf ← mkBuf (dim * seqLen)
  -- Scaled embedding cache: PLE input uses the embedding (not per-layer output)
  let batchScaledEmbdBuf ← mkBuf (dim * seqLen)
  let colIdxBuf ← GPUBackend.allocBuffer ctx (4 : USize)

  -- NOTE: no beginBatch here — each dispatch fires immediately.
  -- Batching would defer all launches until endBatch, but the per-token
  -- attention loop reads batch matmul outputs mid-stream, requiring
  -- them to be complete.  Individual kernel launches on the default
  -- stream are serialized by CUDA, so this is correct.

  -- ── Pre-upload: token IDs and positions to GPU ──────────────────────
  -- Upload all token IDs and position indices to GPU buffers BEFORE any
  -- kernel dispatch.  This eliminates per-token host→device transfers
  -- inside the per-token attention loop, enabling batch dispatch.
  let tokenIdsBuf ← GPUBackend.allocBuffer ctx (seqLen * 4).toUSize
  let posBuf ← GPUBackend.allocBuffer ctx (seqLen * 4).toUSize
  let mut tokBytes : ByteArray := ByteArray.empty
  let mut posBytes : ByteArray := ByteArray.empty
  for i in [0:seqLen] do
    let tokenId := promptTokens[i]!
    tokBytes := tokBytes ++ Hesper.WebGPU.BufferOps.uint32ToBytes tokenId.toUInt32
    posBytes := posBytes ++ Hesper.WebGPU.BufferOps.uint32ToBytes i.toUInt32
  -- cacheLenBuf[i] = i + 1 (number of KV cache entries after writing token i)
  let cacheLenBuf ← GPUBackend.allocBuffer ctx (seqLen * 4).toUSize
  let mut clBytes : ByteArray := ByteArray.empty
  for i in [0:seqLen] do
    clBytes := clBytes ++ Hesper.WebGPU.BufferOps.uint32ToBytes (i + 1).toUInt32
  GPUBackend.writeBuffer ctx tokenIdsBuf tokBytes
  GPUBackend.writeBuffer ctx posBuf posBytes
  GPUBackend.writeBuffer ctx cacheLenBuf clBytes

  -- ── Step 1: Embedding lookup — per token into batch buffer ──────────
  for i in [0:seqLen] do
    -- GPU-side: copy tokenIdsBuf[i] → state.tokenBuf[0]
    let idxBytes := Hesper.WebGPU.BufferOps.uint32ToBytes i.toUInt32
    GPUBackend.writeBufferOffset ctx colIdxBuf 0 idxBytes
    GPUBackend.execute ctx
      (copyU32Kernel seqLen 1 0)
      [("src", tokenIdsBuf), ("params", colIdxBuf), ("dst", state.tokenBuf)]
      { numWorkgroups := (1, 1, 1), workgroupSize := { x := 1, y := 1, z := 1 } }
    match model.embdFormat with
    | .Q6_K =>
      ce "q6kEmbLookup"
        (Hesper.Quantization.Q6_K.q6kEmbeddingLookupKernel model.config.vocabSize dim)
        [("token_ids", state.tokenBuf), ("embedding_table", model.embedding.embeddingTable), ("output", state.buf1)]
        (.dispatch1D dim)
    | _ =>
      Embedding.forward ctx model.embedding state.tokenBuf state.buf1 1 1
    -- Copy state.buf1 → batchBuf1 column i
    GPUBackend.writeBufferOffset ctx colIdxBuf 0 idxBytes
    GPUBackend.execute ctx
      (columnInsertKernel dim seqLen)
      [("src", state.buf1), ("params", colIdxBuf), ("batch", batchBuf1)]
      (.dispatch1D dim)

  -- ── Step 1b: Scale embeddings by sqrt(hiddenSize) — batch-wide ──────
  -- embeddingScaleKernel takes a `size` param — we pass dim * seqLen so
  -- it covers all columns in one dispatch.
  let totalHidden := dim * seqLen
  ce "embedScaleBatch"
    (embeddingScaleKernel totalHidden dim)
    [("input", batchBuf1), ("output", batchBuf2)]
    (.dispatch1D totalHidden)

  -- Cache the scaled embedding (pre-layer state) for PLE usage inside the block loop.
  -- Single-token path precomputes plInputAll ONCE from the scaled embedding and reuses
  -- across layers; batch path recomputes per token per layer, and MUST use the scaled
  -- embedding (not the current layer output) as the PLE matmul input.
  do
    let totalScaled := dim * seqLen
    let shader : ShaderM Unit := do
      let gid ← ShaderM.globalId
      let i := Exp.vec3X gid
      let _src ← ShaderM.declareInputBuffer "src" (.array (.scalar .f32) totalScaled)
      let _dst ← ShaderM.declareOutputBuffer "dst" (.array (.scalar .f32) totalScaled)
      ShaderM.if_ (Exp.lt i (Exp.litU32 totalScaled)) (do
        let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalScaled) "src" i
        ShaderM.writeBuffer (ty := .scalar .f32) "dst" i v) (pure ())
    ce "batchScaledEmbdCopy" shader
      [("src", batchBuf2), ("dst", batchScaledEmbdBuf)]
      (.dispatch1D totalScaled)

  -- Step 1b: Per-layer input precomputation (BATCHED across all prompt
  -- tokens).  Gemma 4 E4B's per_layer_token_embd needs a plInputAll
  -- vector per token; the value depends only on tokenId (via the
  -- scaled embedding) and NOT on the layer index, so it's safe to
  -- compute once per token before the block loop and reuse for all
  -- 42 layers.  Previously this loop ran per-token-per-layer inside
  -- the block loop (42× redundant recompute).  Now the result is
  -- stored in a `batchPLInputAll : [seqLen × totalPL]` scratch
  -- buffer, and `forwardBlock` reads column `i` from it per token.
  let mut batchPLInputAllOpt : Option (GPUBackend.Buf β) := none
  match model.perLayerEmbdTableGPU, model.perLayerModelProj, model.perLayerProjNorm with
  | some embdTableGPU, some modelProj, some projNorm =>
    let embdPL := model.config.embdPerLayer
    let nLayers := model.config.numHiddenLayers
    let totalPL := embdPL * nLayers
    -- Allocate the batched scratch: one totalPL-vector per prompt token.
    let batchPLInputAll ← GPUBackend.allocBuffer ctx (seqLen * totalPL * 4).toUSize
    batchPLInputAllOpt := some batchPLInputAll
    for i in [0:seqLen] do
      let tokenId := promptTokens[i]!
      let tokenIdBytes := Hesper.WebGPU.BufferOps.uint32ToBytes tokenId.toUInt32
      GPUBackend.writeBufferOffset ctx state.plRawRowBuf 0 tokenIdBytes
      let scaleFactor : Float := Float.sqrt embdPL.toFloat
      ce "q6kDequantScale_pf"
        (Hesper.Quantization.Q6_K.q6kTableRowDequantScaleKernel totalPL scaleFactor
          cfg.vocabSize)
        [("table", embdTableGPU), ("params", state.plRawRowBuf), ("output", state.plModelProj)]
        (.dispatch1D totalPL)
      let colIdxBytes := Hesper.WebGPU.BufferOps.uint32ToBytes i.toUInt32
      GPUBackend.writeBufferOffset ctx colIdxBuf 0 colIdxBytes
      GPUBackend.execute ctx
        (columnExtractKernel dim seqLen)
        [("batch", batchBuf2), ("params", colIdxBuf), ("out", state.buf1)]
        (.dispatch1D dim)
      let projConfig : Hesper.WGSL.MatMul.Config := {
        M := 1, N := totalPL, K := dim
      }
      if projConfig.K % 64 == 0 then
        Hesper.WGSL.MatMul.executeMatMulTransposeF16BlockCoop ctx state.buf1 modelProj state.plTokenSelected projConfig
      else
        Hesper.WGSL.MatMul.executeMatMulTransposeF16 ctx state.buf1 modelProj state.plTokenSelected projConfig
      ce "pleScalePL_pf"
        (PerLayerEmbedding.scaleKernel totalPL (1.0 / Float.sqrt dim.toFloat))
        [("input", state.plTokenSelected), ("output", state.plInputAll)]
        (.dispatch1D totalPL)
      ce "chunkedRMSNorm_pf"
        (chunkedRMSNormKernel embdPL nLayers model.config.rmsNormEps)
        [("input", state.plInputAll), ("weight", projNorm.scale), ("output", state.plTokenSelected)]
        { numWorkgroups := (nLayers, 1, 1), workgroupSize := { x := min embdPL 256, y := 1, z := 1 } : Hesper.ExecConfig }
      ce "scaledAdd_pf"
        (scaledAddKernel totalPL (1.0 / Float.sqrt 2.0))
        [("a", state.plTokenSelected), ("b", state.plModelProj), ("output", state.plInputAll)]
        (.dispatch1D totalPL)
      -- Copy the per-token plInputAll into column `i` of the batched
      -- buffer via a params-indexed kernel so the PTX cache key does
      -- NOT embed `i` (otherwise every iteration is a JIT miss).
      let colIdxBytes2 := Hesper.WebGPU.BufferOps.uint32ToBytes i.toUInt32
      GPUBackend.writeBufferOffset ctx colIdxBuf 0 colIdxBytes2
      let copyShader : ShaderM Unit := do
        let gid ← ShaderM.globalId
        let k := Exp.vec3X gid
        let _src ← ShaderM.declareInputBuffer "src" (.array (.scalar .f32) totalPL)
        let _params ← ShaderM.declareInputBuffer "params" (.array (.scalar .u32) 1)
        let _dst ← ShaderM.declareOutputBuffer "dst" (.array (.scalar .f32) (seqLen * totalPL))
        ShaderM.if_ (Exp.lt k (Exp.litU32 totalPL)) (do
          let colId ← ShaderM.readBuffer (ty := .scalar .u32) (n := 1) "params" (Exp.litU32 0)
          let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalPL) "src" k
          let dstIdx := Exp.add (Exp.mul colId (Exp.litU32 totalPL)) k
          ShaderM.writeBuffer (ty := .scalar .f32) "dst" dstIdx v) (pure ())
      ce "batchPLInputAllCopy" copyShader
        [("src", state.plInputAll), ("params", colIdxBuf), ("dst", batchPLInputAll)]
        (.dispatch1D totalPL)
  | _, _, _ => pure ()

  -- ── Step 2: Process transformer blocks ──────────────────────────────
  let mut currentBuf := batchBuf2
  let mut nextBuf := batchBuf1

  -- Dump post-PLE state for each token (extract each column to state.buf1 and dump)
  for i in [0:seqLen] do
    let idxBytes := Hesper.WebGPU.BufferOps.uint32ToBytes i.toUInt32
    GPUBackend.writeBufferOffset ctx colIdxBuf 0 idxBytes
    GPUBackend.execute ctx
      (columnExtractKernel dim seqLen)
      [("batch", currentBuf), ("params", colIdxBuf), ("out", state.buf1)]
      (.dispatch1D dim)
    dumpBuf ctx state.buf1 (dim * 4).toUSize s!"batch_t{i}_postPLE"

  let nBlocksToRun := match ← IO.getEnv "HESPER_PREFILL_LAYERS" with
    | some s => s.toNat!
    | none => model.blocks.size
  for block in model.blocks.extract 0 nBlocksToRun do
    let li := block.layerIdx
    let headDim := cfg.headDim li
    let numHeads := cfg.numAttentionHeads
    let numKVHeads := cfg.numKVHeads li
    let kvDim := numKVHeads * headDim
    let qDim := numHeads * headDim

    -- ── 2a: RMSNorm + Q/K/V projections (batch) ────────────────────────
    -- Fast path (all Q4_K): fused RMSNorm+Q8_1 quantize → Q8_1 batch matmul.
    -- Fallback (any Q6_K in Q/K/V): standalone RMSNorm → f32 batch matmul.
    let nQ8Blocks := dim / 32
    let allQ4K := block.attention.wQ.quantFormat == .Q4_K
                && (!cfg.hasKV li ||
                    (block.attention.wK.quantFormat == .Q4_K
                     && block.attention.wV.quantFormat == .Q4_K))
    if allQ4K then
      let batchQ8Bytes : USize := (nQ8Blocks * 9 * seqLen * 4).toUSize
      let batchQ8Buf ← GPUBackend.allocBuffer ctx batchQ8Bytes
      GPUBackend.executeWithConfig ctx
        (RMSNorm.fusedRMSNormQ8_1Kernel block.attnNorm.config seqLen 256)
        [("input", currentBuf), ("scale", block.attnNorm.scale), ("output", batchQ8Buf)]
        { workgroupSize := { x := 256 }, numWorkgroups := (seqLen, 1, 1) }
      Linear.forwardBatchDP4A_fromQ8 ctx block.attention.wQ batchQ8Buf batchQBuf seqLen
      if cfg.hasKV li then
        Linear.forwardBatchDP4A_fromQ8 ctx block.attention.wK batchQ8Buf batchKBuf seqLen
        Linear.forwardBatchDP4A_fromQ8 ctx block.attention.wV batchQ8Buf batchVBuf seqLen
      GPUBackend.freeBuffer ctx batchQ8Buf
    else
      -- Q6_K fallback: RMSNorm into batchNormedBuf as scratch (CANNOT use
      -- batchBuf1 since it may alias currentBuf during ping-pong).  Then
      -- batch matmul in f32.
      RMSNorm.forward ctx block.attnNorm currentBuf batchNormedBuf seqLen
      Linear.forwardBatchDP4A ctx block.attention.wQ batchNormedBuf batchQBuf seqLen
      if cfg.hasKV li then
        Linear.forwardBatchDP4A ctx block.attention.wK batchNormedBuf batchKBuf seqLen
        Linear.forwardBatchDP4A ctx block.attention.wV batchNormedBuf batchVBuf seqLen

    -- ── 2c: Attention (batched when possible, per-token fallback) ─────
    let wgSize := min headDim 256
    let isFull := cfg.isFullAttention li
    let kvLi := cfg.kvCacheLayer li
    -- Batched path requires:
    --   * cfg.hasKV li (this layer has its own KV cache)
    --   * block.ropeFreqFactors = some (full-attention layers only)
    --   * a valid KV cache slot
    let mut handledByBatched := false
    if hKV : kvLi < state.kvCaches.size then
      match (if cfg.hasKV li then block.ropeFreqFactors else none) with
      | some freqFactors =>
        handledByBatched := true
        let kvCache := state.kvCaches[kvLi]

        -- startPos = 0 (prefill from scratch — KV cache is empty for this
        -- layer before this batch).  Write to paramsBuf[0]; the kernel
        -- treats wgid.y/z as the per-token offset.
        GPUBackend.writeBufferOffset ctx state.paramsBuf 0
          (Hesper.WebGPU.BufferOps.uint32ToBytes 0)

        -- Batched fused QKV norm: grid (numHeads*seqLen, 3, 1).
        ce (if isFull then "qkvNormFullBatch" else "qkvNormSWABatch")
          (fusedPerHeadQKVNormBatchKernel numHeads numKVHeads headDim seqLen cfg.rmsNormEps)
          [("q_in", batchQBuf), ("q_scale", block.attention.qNormWeight), ("q_out", batchQBuf),
           ("k_in", batchKBuf), ("k_scale", block.attention.kNormWeight), ("k_out", batchKBuf),
           ("v_in", batchVBuf),                                            ("v_out", batchVBuf)]
          { numWorkgroups := (numHeads * seqLen, 3, 1),
            workgroupSize := { x := wgSize, y := 1, z := 1 } : Hesper.ExecConfig }

        -- Batched RoPE-Q (in place).
        ce s!"ropeFreqQBatch_{headDim}"
          (ropeWithFreqFactorsBatchKernel headDim numHeads seqLen cfg.ropeTheta)
          [("input", batchQBuf), ("output", batchQBuf),
           ("params", state.paramsBuf), ("freq_factors", freqFactors)]
          (.dispatch1D (numHeads * headDim / 2 * seqLen))

        -- Batched RoPE-K + KV cache write.
        ce s!"ropeKKvWBatch_{headDim}_{numKVHeads}"
          (fusedRopeKAndCacheWriteBatchKernel numKVHeads cfg.maxSeqLen headDim seqLen cfg.ropeTheta)
          [("new_k", batchKBuf), ("new_v", batchVBuf),
           ("k_cache", kvCache.kBuf), ("v_cache", kvCache.vBuf),
           ("params", state.paramsBuf), ("freq_factors", freqFactors)]
          (.dispatch1D (numKVHeads * headDim / 2 * seqLen))

        -- Batched flash-attention.  Grid (numHeads, seqLen, 1).
        let scale : Float := 1.0
        ce s!"flashAttnBatch_{headDim}_{numKVHeads}"
          (FlashAttention.flashAttentionBatchKernel numHeads numKVHeads cfg.maxSeqLen headDim seqLen scale)
          [("q", batchQBuf), ("k_cache", kvCache.kBuf), ("v_cache", kvCache.vBuf),
           ("output", batchAttnOutBuf), ("params", state.paramsBuf)]
          ({ numWorkgroups := (numHeads, seqLen, 1) : Hesper.ExecConfig })
      | none => pure ()

    if !handledByBatched then
    for i in [0:seqLen] do
      let pos := i

      -- Extract Q column i → state.qBuf
      let colIdxBytes := Hesper.WebGPU.BufferOps.uint32ToBytes i.toUInt32
      GPUBackend.writeBufferOffset ctx colIdxBuf 0 colIdxBytes
      GPUBackend.execute ctx
        (columnExtractKernel qDim seqLen)
        [("batch", batchQBuf), ("params", colIdxBuf), ("out", state.qBuf)]
        (.dispatch1D qDim)
      -- Extract K, V columns (if this layer has its own KV)
      if cfg.hasKV li then
        GPUBackend.execute ctx
          (columnExtractKernel kvDim seqLen)
          [("batch", batchKBuf), ("params", colIdxBuf), ("out", state.kBuf)]
          (.dispatch1D kvDim)
        GPUBackend.execute ctx
          (columnExtractKernel kvDim seqLen)
          [("batch", batchVBuf), ("params", colIdxBuf), ("out", state.vBuf)]
          (.dispatch1D kvDim)

      -- Per-head QKV norms (single token)
      if cfg.hasKV li then
        ce (if isFull then "qkvNormFull_pf" else "qkvNormSWA_pf")
          (fusedPerHeadQKVNormKernel numHeads numKVHeads headDim cfg.rmsNormEps)
          [("q_in", state.qBuf), ("q_scale", block.attention.qNormWeight), ("q_out", state.qBuf2),
           ("k_in", state.kBuf), ("k_scale", block.attention.kNormWeight), ("k_out", state.kBuf2),
           ("v_in", state.vBuf),                                              ("v_out", state.vBuf2)]
          { numWorkgroups := (numHeads, 3, 1),
            workgroupSize := { x := wgSize, y := 1, z := 1 } : Hesper.ExecConfig }
      else
        ce (if isFull then "qNormFull_pf" else "qNormSWA_pf")
          (perHeadRMSNormKernel numHeads headDim cfg.rmsNormEps)
          [("input", state.qBuf), ("weight", block.attention.qNormWeight), ("output", state.qBuf2)]
          { numWorkgroups := (numHeads, 1, 1),
            workgroupSize := { x := wgSize, y := 1, z := 1 } : Hesper.ExecConfig }

      -- RoPE on Q: qBuf2 → qBuf
      -- GPU-side: posBuf[i] → paramsBuf[0]
      GPUBackend.writeBufferOffset ctx colIdxBuf 0 colIdxBytes  -- i already written above
      GPUBackend.execute ctx
        (copyU32Kernel seqLen 2 0)
        [("src", posBuf), ("params", colIdxBuf), ("dst", state.paramsBuf)]
        { numWorkgroups := (1, 1, 1), workgroupSize := { x := 1, y := 1, z := 1 } }
      match block.ropeFreqFactors with
      | some freqFactors =>
        ce s!"ropeFreqQ_pf_{headDim}"
          (ropeWithFreqFactorsKernel headDim numHeads cfg.ropeTheta)
          [("input", state.qBuf2), ("output", state.qBuf), ("params", state.paramsBuf), ("freq_factors", freqFactors)]
          (.dispatch1D (numHeads * headDim / 2))
      | none =>
        let ropeConfig : RoPE.Config := { dim := numHeads * headDim, maxSeqLen := cfg.maxSeqLen, base := cfg.ropeTheta }
        ce s!"ropeDynQ_pf_{headDim}"
          (RoPE.ropeKernelDynamic ropeConfig 1 1 numHeads headDim)
          [("input", state.qBuf2), ("output", state.qBuf), ("params", state.paramsBuf)]
          (.dispatch1D (numHeads * headDim / 2))

      -- RoPE on K + KV cache write + flash attention
      if h : kvLi < state.kvCaches.size then
        let kvCache := state.kvCaches[kvLi]
        let cacheLen := pos + 1

        if cfg.hasKV li then
          -- RoPE-K + KV cache write
          match block.ropeFreqFactors with
          | some freqFactors =>
            ce s!"ropeKKvW_pf_{headDim}_{numKVHeads}"
              (Attention.fusedRopeKAndCacheWriteKernel numKVHeads cfg.maxSeqLen headDim kvDim cfg.ropeTheta)
              [("new_k", state.kBuf2), ("new_v", state.vBuf2),
               ("k_cache", kvCache.kBuf), ("v_cache", kvCache.vBuf),
               ("params", state.paramsBuf), ("freq_factors", freqFactors)]
              (.dispatch1D kvDim)
          | none =>
            -- Separate RoPE-K then KV write
            let ropeConfig : RoPE.Config := { dim := kvDim, maxSeqLen := cfg.maxSeqLen, base := cfg.ropeTheta }
            ce s!"ropeDynK_pf_{headDim}_{numKVHeads}"
              (RoPE.ropeKernelDynamic ropeConfig 1 1 numKVHeads headDim)
              [("input", state.kBuf2), ("output", state.kBuf), ("params", state.paramsBuf)]
              (.dispatch1D (numKVHeads * headDim / 2))
            ce s!"kvWrite_pf_{headDim}_{numKVHeads}"
              (Attention.fusedCacheWriteKVKernel numKVHeads cfg.maxSeqLen headDim kvDim)
              [("new_k", state.kBuf), ("new_v", state.vBuf2),
               ("k_cache", kvCache.kBuf), ("v_cache", kvCache.vBuf),
               ("params", state.paramsBuf)]
              (.dispatch1D kvDim)

        -- Flash attention
        let scale : Float := 1.0
        -- GPU-side: cacheLenBuf[i] → paramsBuf[1] (offset 4 bytes = u32 index 1)
        GPUBackend.execute ctx
          (copyU32Kernel seqLen 2 1)
          [("src", cacheLenBuf), ("params", colIdxBuf), ("dst", state.paramsBuf)]
          { numWorkgroups := (1, 1, 1), workgroupSize := { x := 1, y := 1, z := 1 } }
        if cacheLen > 32 then
          FlashAttention.executeFlashAttentionTiled ctx
            state.qBuf kvCache.kBuf kvCache.vBuf state.attnOutBuf
            numHeads numKVHeads cfg.maxSeqLen headDim cacheLen scale
            (partialBuf := some state.flashPartialBuf)
        else
          ce s!"flashAttnP_pf_{headDim}_{numKVHeads}"
            (FlashAttention.flashAttentionDynamicParamsKernel numHeads numKVHeads cfg.maxSeqLen headDim scale)
            [("q", state.qBuf), ("k_cache", kvCache.kBuf), ("v_cache", kvCache.vBuf),
             ("output", state.attnOutBuf), ("params", state.paramsBuf)]
            ({ numWorkgroups := (numHeads, 1, 1) : Hesper.ExecConfig })

        -- Insert attnOut into batch buffer for later O-projection
        GPUBackend.writeBufferOffset ctx colIdxBuf 0 colIdxBytes
        GPUBackend.execute ctx
          (columnInsertKernel qDim seqLen)
          [("src", state.attnOutBuf), ("params", colIdxBuf), ("batch", batchAttnOutBuf)]
          (.dispatch1D qDim)

    -- ── 2d: O projection (batch matmul) ──────────────────────────────
    Linear.forwardBatchDP4A ctx block.attention.wO batchAttnOutBuf batchOProjBuf seqLen

    -- ── 2e: Post-attention norm + residual (per-token) ───────────────
    -- `attnResid[i] = RMSNorm(oProj[i]) * scale + current[i]`
    -- forwardNormThenAdd is hardcoded to 1 row, so we loop per token.
    for i in [0:seqLen] do
      -- Extract oProj column i → state.normedBuf
      let colIdxBytes := Hesper.WebGPU.BufferOps.uint32ToBytes i.toUInt32
      GPUBackend.writeBufferOffset ctx colIdxBuf 0 colIdxBytes
      GPUBackend.execute ctx
        (columnExtractKernel dim seqLen)
        [("batch", batchOProjBuf), ("params", colIdxBuf), ("out", state.normedBuf)]
        (.dispatch1D dim)
      -- Extract current (input) column i → state.buf1
      GPUBackend.execute ctx
        (columnExtractKernel dim seqLen)
        [("batch", currentBuf), ("params", colIdxBuf), ("out", state.buf1)]
        (.dispatch1D dim)
      -- Fused post-attn norm + residual add
      let ref ← IO.mkRef none
      RMSNorm.forwardNormThenAdd ctx block.postAttnNorm
        state.normedBuf state.buf1 state.attnResidualBuf ref
      -- Insert result into batchAttnResidBuf column i
      GPUBackend.execute ctx
        (columnInsertKernel dim seqLen)
        [("src", state.attnResidualBuf), ("params", colIdxBuf), ("batch", batchAttnResidBuf)]
        (.dispatch1D dim)

    -- Diagnostic: dump currentBuf col 0 at L1 after attention inner loop / after O proj
    if li = 1 then
      GPUBackend.writeBufferOffset ctx colIdxBuf 0 (Hesper.WebGPU.BufferOps.uint32ToBytes 0)
      GPUBackend.execute ctx
        (columnExtractKernel dim seqLen)
        [("batch", currentBuf), ("params", colIdxBuf), ("out", state.buf1)]
        (.dispatch1D dim)
      dumpBuf ctx state.buf1 (dim * 4).toUSize s!"batch_t0_currBufL1afterAttnOProj"

    -- Dump post-attn residual for each token (batch diagnostic) — only L0/L1 for brevity
    if li ≤ 2 then for i in [0:seqLen] do
      let idxBytes := Hesper.WebGPU.BufferOps.uint32ToBytes i.toUInt32
      GPUBackend.writeBufferOffset ctx colIdxBuf 0 idxBytes
      GPUBackend.execute ctx
        (columnExtractKernel dim seqLen)
        [("batch", batchAttnResidBuf), ("params", colIdxBuf), ("out", state.buf1)]
        (.dispatch1D dim)
      dumpBuf ctx state.buf1 (dim * 4).toUSize s!"batch_t{i}_postAttn_L{li}"

    -- ── 2f: FFN ──────────────────────────────────────────────────────
    -- Skip MoE — use dense FFN path only.

    -- FFN norm + Gate/Up projections (batch)
    -- Fast path (both Q4_K): fused RMSNorm+Q8_1 quantize → Q8_1 batch matmul.
    -- Fallback: standalone RMSNorm → f32 batch matmul.
    let ffnAllQ4K := block.ffn.gate.quantFormat == .Q4_K
                  && block.ffn.up.quantFormat == .Q4_K
    if ffnAllQ4K then
      let ffnQ8Bytes : USize := (nQ8Blocks * 9 * seqLen * 4).toUSize
      let ffnBatchQ8Buf ← GPUBackend.allocBuffer ctx ffnQ8Bytes
      GPUBackend.executeWithConfig ctx
        (RMSNorm.fusedRMSNormQ8_1Kernel block.ffnNorm.config seqLen 256)
        [("input", batchAttnResidBuf), ("scale", block.ffnNorm.scale), ("output", ffnBatchQ8Buf)]
        { workgroupSize := { x := 256 }, numWorkgroups := (seqLen, 1, 1) }
      Linear.forwardBatchDP4A_fromQ8 ctx block.ffn.gate ffnBatchQ8Buf batchGateBuf seqLen
      Linear.forwardBatchDP4A_fromQ8 ctx block.ffn.up ffnBatchQ8Buf batchUpBuf seqLen
      GPUBackend.freeBuffer ctx ffnBatchQ8Buf
    else
      -- FFN Q6_K fallback: normedBuf can't be batchBuf1 (ping-pong alias).
      RMSNorm.forward ctx block.ffnNorm batchAttnResidBuf batchNormedBuf seqLen
      Linear.forwardBatchDP4A ctx block.ffn.gate batchNormedBuf batchGateBuf seqLen
      Linear.forwardBatchDP4A ctx block.ffn.up batchNormedBuf batchUpBuf seqLen

    -- GELU * up (batch pointwise — dispatch with totalElements = interSize * seqLen)
    let totalInter := interSize * seqLen
    ce s!"geluMulBatch_{li}"
      (geluMulKernel totalInter)
      [("gate", batchGateBuf), ("up", batchUpBuf), ("output", batchGeluBuf)]
      (.dispatch1D totalInter)

    -- Down projection (batch matmul)
    Linear.forwardBatchDP4A ctx block.ffn.down batchGeluBuf batchFFNOutBuf seqLen

    -- ── 2g: Post-FFN norm + residual (per-token) ─────────────────────
    -- `output[i] = RMSNorm(ffnOut[i]) * scale + attnResid[i]`
    for i in [0:seqLen] do
      let colIdxBytes := Hesper.WebGPU.BufferOps.uint32ToBytes i.toUInt32
      GPUBackend.writeBufferOffset ctx colIdxBuf 0 colIdxBytes
      GPUBackend.execute ctx
        (columnExtractKernel dim seqLen)
        [("batch", batchFFNOutBuf), ("params", colIdxBuf), ("out", state.ffnOutBuf)]
        (.dispatch1D dim)
      GPUBackend.execute ctx
        (columnExtractKernel dim seqLen)
        [("batch", batchAttnResidBuf), ("params", colIdxBuf), ("out", state.attnResidualBuf)]
        (.dispatch1D dim)
      let ref ← IO.mkRef none
      RMSNorm.forwardNormThenAdd ctx block.postFFNNorm
        state.ffnOutBuf state.attnResidualBuf state.buf2 ref
      GPUBackend.execute ctx
        (columnInsertKernel dim seqLen)
        [("src", state.buf2), ("params", colIdxBuf), ("batch", nextBuf)]
        (.dispatch1D dim)

    -- Dump post-FFN (pre-PLE) state
    for i in [0:seqLen] do
      let idxBytes := Hesper.WebGPU.BufferOps.uint32ToBytes i.toUInt32
      GPUBackend.writeBufferOffset ctx colIdxBuf 0 idxBytes
      GPUBackend.execute ctx
        (columnExtractKernel dim seqLen)
        [("batch", nextBuf), ("params", colIdxBuf), ("out", state.buf1)]
        (.dispatch1D dim)
      dumpBuf ctx state.buf1 (dim * 4).toUSize s!"batch_t{i}_postFFN_L{li}"

    -- ── 2h: Per-layer embedding + layer output scale (per-token) ──────
    -- Gemma 4 E4B uses per_layer_token_embd: each layer's output gets an
    -- additive embedding that depends on both the layer index and the
    -- token's per-layer input (precomputed in Step 1b).  We also apply
    -- the layer output scale (a single scalar multiply per layer).
    let blockIdx := li
    let skipPLE := (← IO.getEnv "HESPER_SKIP_PLE").isSome
    let plEmbd := if blockIdx < model.perLayerBlocks.size && !skipPLE then
      model.perLayerBlocks[blockIdx]!
    else none
    match plEmbd with
    | some ple =>
      -- Per-layer embedding for each token position.  plInputAll for
      -- each prompt token was precomputed once before the block loop
      -- (see `batchPLInputAllOpt` above) — here we just extract the
      -- relevant column from the batched scratch into `state.plInputAll`,
      -- saving 42× redundant PLE recompute per layer.
      for i in [0:seqLen] do
        let colIdxBytes := Hesper.WebGPU.BufferOps.uint32ToBytes i.toUInt32
        match batchPLInputAllOpt with
        | some batchPL =>
          let embdPL := cfg.embdPerLayer
          let nLayers := cfg.numHiddenLayers
          let totalPL := embdPL * nLayers
          GPUBackend.writeBufferOffset ctx colIdxBuf 0 colIdxBytes
          -- Copy column i of batchPLInputAll (contiguous at offset
          -- i*totalPL) into the single-token state.plInputAll buffer.
          let extractShader : ShaderM Unit := do
            let gid ← ShaderM.globalId
            let k := Exp.vec3X gid
            let _src ← ShaderM.declareInputBuffer "src" (.array (.scalar .f32) (seqLen * totalPL))
            let _params ← ShaderM.declareInputBuffer "params" (.array (.scalar .u32) 1)
            let _dst ← ShaderM.declareOutputBuffer "dst" (.array (.scalar .f32) totalPL)
            ShaderM.if_ (Exp.lt k (Exp.litU32 totalPL)) (do
              let colId ← ShaderM.readBuffer (ty := .scalar .u32) (n := 1) "params" (Exp.litU32 0)
              let srcOff := Exp.add (Exp.mul colId (Exp.litU32 totalPL)) k
              let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := seqLen * totalPL) "src" srcOff
              ShaderM.writeBuffer (ty := .scalar .f32) "dst" k v) (pure ())
          ce "plInputAllExtract"
            extractShader
            [("src", batchPL), ("params", colIdxBuf), ("dst", state.plInputAll)]
            (.dispatch1D totalPL)
        | none => pure ()
        -- Extract this token's output column → state.buf2 (input to PLE inpGate)
        GPUBackend.writeBufferOffset ctx colIdxBuf 0 colIdxBytes
        GPUBackend.execute ctx
          (columnExtractKernel dim seqLen)
          [("batch", nextBuf), ("params", colIdxBuf), ("out", state.buf2)]
          (.dispatch1D dim)
        -- Now run the actual per-layer embedding ops on state.buf2
        -- (inpGate matmul → gelu*gate*slice → proj → postNorm+add)
        let plOffset := li * cfg.embdPerLayer
        let plTotalSize := cfg.embdPerLayer * cfg.numHiddenLayers
        Linear.LinearLayer.forward ctx ple.inpGate state.buf2 state.plGateBuf
        GPUBackend.execute ctx
          (PerLayerEmbedding.geluGateMulSliceKernel cfg.embdPerLayer plTotalSize plOffset)
          [("gate", state.plGateBuf), ("per_layer_input", state.plInputAll), ("output", state.moeRouterOutBuf)]
          (.dispatch1D cfg.embdPerLayer)
        Linear.LinearLayer.forward ctx ple.proj state.moeRouterOutBuf state.plProjBuf
        GPUBackend.execute ctx
          (fusedPerLayerPostKernel cfg.hiddenSize cfg.rmsNormEps)
          [("proj", state.plProjBuf), ("weight", ple.postNorm.scale), ("residual", state.buf2)]
          { numWorkgroups := (1, 1, 1), workgroupSize := { x := 256, y := 1, z := 1 }
            extensions := ["subgroups"] : Hesper.ExecConfig }
        -- Insert modified output back into batch buffer
        GPUBackend.writeBufferOffset ctx colIdxBuf 0 colIdxBytes
        GPUBackend.execute ctx
          (columnInsertKernel dim seqLen)
          [("src", state.buf2), ("params", colIdxBuf), ("batch", nextBuf)]
          (.dispatch1D dim)
    | none => pure ()

    -- Dump post-PLE (pre-outScale) state
    for i in [0:seqLen] do
      let idxBytes := Hesper.WebGPU.BufferOps.uint32ToBytes i.toUInt32
      GPUBackend.writeBufferOffset ctx colIdxBuf 0 idxBytes
      GPUBackend.execute ctx
        (columnExtractKernel dim seqLen)
        [("batch", nextBuf), ("params", colIdxBuf), ("out", state.buf1)]
        (.dispatch1D dim)
      dumpBuf ctx state.buf1 (dim * 4).toUSize s!"batch_t{i}_postPLE_L{li}"

    -- Layer output scale (per-token) — uses Circuit DSL scalar broadcast multiply
    let skipOutScale := (← IO.getEnv "HESPER_SKIP_OUTSCALE").isSome
    match if skipOutScale then none else block.outScale with
    | some scale =>
      for i in [0:seqLen] do
        let colIdxBytes := Hesper.WebGPU.BufferOps.uint32ToBytes i.toUInt32
        GPUBackend.writeBufferOffset ctx colIdxBuf 0 colIdxBytes
        GPUBackend.execute ctx
          (columnExtractKernel dim seqLen)
          [("batch", nextBuf), ("params", colIdxBuf), ("out", state.buf2)]
          (.dispatch1D dim)
        -- output[j] = buf2[j] * scale[0] via Circuit DSL runCachedFused
        let key := hash ("batchPrefillOutScale", dim, li)
        let ccRef ← Hesper.Circuit.getGlobalCircuitRef (β := β) key
        Hesper.Circuit.runCachedFused ctx ccRef
          (do
            let x ← Hesper.Circuit.CircuitM.registerExternal
                      (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
                      state.buf2 #[dim] .f32 .Global
            let s ← Hesper.Circuit.CircuitM.registerExternal
                      (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
                      scale #[1] .f32 .Global
            let scaled ← Hesper.Circuit.CircuitM.scaleByBroadcast x s
            let _out   ← Hesper.Circuit.CircuitM.map scaled
                           (.mul (.input 0) (.const 1.0))
            pure ())
          -- ids: 0=x (state.buf2), 1=s (scale), 3=final map output written back.
          [(0, state.buf2), (1, scale), (3, state.buf2)]
        GPUBackend.execute ctx
          (columnInsertKernel dim seqLen)
          [("src", state.buf2), ("params", colIdxBuf), ("batch", nextBuf)]
          (.dispatch1D dim)
    | none => pure ()

    -- Per-layer batch Q8_1 buffers are freed inside their respective
    -- Q4_K fast-path branches above.

    -- Swap ping-pong buffers
    let oldCur := currentBuf
    currentBuf := nextBuf
    nextBuf := oldCur

    -- Dump post-layer state for each token
    for i in [0:seqLen] do
      let idxBytes := Hesper.WebGPU.BufferOps.uint32ToBytes i.toUInt32
      GPUBackend.writeBufferOffset ctx colIdxBuf 0 idxBytes
      GPUBackend.execute ctx
        (columnExtractKernel dim seqLen)
        [("batch", currentBuf), ("params", colIdxBuf), ("out", state.buf1)]
        (.dispatch1D dim)
      dumpBuf ctx state.buf1 (dim * 4).toUSize s!"batch_t{i}_afterL{li}"
      -- Also dump previous-buffer (before outScale if it ran) to localize
      GPUBackend.execute ctx
        (columnExtractKernel dim seqLen)
        [("batch", batchAttnResidBuf), ("params", colIdxBuf), ("out", state.buf1)]
        (.dispatch1D dim)
      dumpBuf ctx state.buf1 (dim * 4).toUSize s!"batch_t{i}_attnResidL{li}"

  -- ── Step 3: Extract last token → final norm → lm head ─────────────
  -- Copy last column of currentBuf → state.buf2 (single-token)
  let lastCol := seqLen - 1
  let lastColBytes := Hesper.WebGPU.BufferOps.uint32ToBytes lastCol.toUInt32
  GPUBackend.writeBufferOffset ctx colIdxBuf 0 lastColBytes
  GPUBackend.execute ctx
    (columnExtractKernel dim seqLen)
    [("batch", currentBuf), ("params", colIdxBuf), ("out", state.buf2)]
    (.dispatch1D dim)

  -- LM head.
  --
  -- For the Q6_K dp4a path, fuse the final RMSNorm directly into the Q8_1
  -- quantize step — identical pattern to `forwardFusedNormGateUp`.  Saves
  -- one dispatch per token (the standalone RMSNorm) and one ~2560-float
  -- VRAM round-trip (the f32 normed hidden state).  For the fallback
  -- paths (f32 matmul etc.) still run the standalone RMSNorm because
  -- they don't consume Q8_1.
  match model.embdFormat with
  | .Q6_K =>
    let useSubgroups ← GPUBackend.hasSubgroupSupport ctx
    let dp4aOn ← do
      let a ← Hesper.Layers.Linear.dp4aEnabled.get
      let b ← Hesper.Layers.Linear.dp4aQ6KEnabled.get
      pure (a && b)
    let gridX : Nat := 4096
    let gridY : Nat := (cfg.vocabSize + gridX - 1) / gridX
    if dp4aOn && useSubgroups && cfg.hiddenSize % 32 == 0 then
      let nQ8Blocks := cfg.hiddenSize / 32
      let q8BufBytes : USize := (nQ8Blocks * 9 * 4).toUSize
      let q8Buf ← match ← state.lmHeadQ8Buf.get with
        | some b => pure b
        | none =>
          let b ← GPUBackend.allocBuffer ctx q8BufBytes
          state.lmHeadQ8Buf.set (some b)
          pure b
      -- Fused finalNorm + Q8_1 quantize (one dispatch, no f32 normed buf).
      GPUBackend.executeWithConfigCached ctx
        (Hesper.Layers.RMSNorm.fusedRMSNormQ8_1Kernel model.finalNorm.config)
        [("input", state.buf2), ("scale", model.finalNorm.scale), ("output", q8Buf)]
        { numWorkgroups := (1, 1, 1), workgroupSize := { x := 256, y := 1, z := 1 }
          extensions := ["subgroups"] : Hesper.ExecConfig }
        (hash ("fused-rmsnorm-q8_1-lmhead", cfg.hiddenSize))
        state.lmHeadQuantizePrepared
      let quadCount := (cfg.vocabSize + 3) / 4
      let gridX4 : Nat := 4096
      let gridY4 : Nat := (quadCount + gridX4 - 1) / gridX4
      GPUBackend.executeWithConfigCached ctx
        (Hesper.Layers.Linear.fusedQ6KLinearDP4A4RowKernel
          cfg.hiddenSize cfg.vocabSize gridX4)
        [("weights", model.outputWeight), ("input_q8", q8Buf), ("output", state.logitsBuf)]
        { numWorkgroups := (gridX4, gridY4, 1), workgroupSize := { x := 128, y := 1, z := 1 }
          extensions := ["subgroups"] : Hesper.ExecConfig }
        (hash ("q6k-dp4a-lmhead-4row", cfg.hiddenSize, cfg.vocabSize))
        state.lmHeadDP4APrepared
    else
      -- Fallback: f32 Q6_K kernel.  Needs standalone RMSNorm since the
      -- f32 matmul can't consume Q8_1.
      RMSNorm.forward ctx model.finalNorm state.buf2 state.buf1
      let shaderF32 := if useSubgroups then
          Hesper.Quantization.Q6_K.fusedQ6KLinearBlockCoopKernel
            cfg.hiddenSize cfg.vocabSize gridX
        else
          Hesper.Quantization.Q6_K.fusedQ6KLinearKernel
            cfg.hiddenSize cfg.vocabSize 256 gridX
      let wgSize := if useSubgroups then 32 else 256
      GPUBackend.execute ctx shaderF32
        [("weights", model.outputWeight), ("input", state.buf1), ("output", state.logitsBuf)]
        { numWorkgroups := (gridX, gridY, 1), workgroupSize := { x := wgSize, y := 1, z := 1 }
          extensions := if useSubgroups then ["subgroups"] else []
          : Hesper.ExecConfig }
  | _ =>
    -- Non-Q6_K fallback: F32 matmul transpose.  Needs standalone RMSNorm.
    RMSNorm.forward ctx model.finalNorm state.buf2 state.buf1
    let lmHeadConfig : Hesper.WGSL.MatMul.Config := {
      M := 1, N := cfg.vocabSize, K := cfg.hiddenSize
    }
    Hesper.WGSL.MatMul.executeMatMulTranspose ctx state.buf1 model.outputWeight state.logitsBuf lmHeadConfig

  -- ── Free prefill batch buffers ─────────────────────────────────────
  GPUBackend.freeBuffer ctx batchBuf1
  GPUBackend.freeBuffer ctx batchBuf2
  GPUBackend.freeBuffer ctx batchNormedBuf
  GPUBackend.freeBuffer ctx batchQBuf
  GPUBackend.freeBuffer ctx batchKBuf
  GPUBackend.freeBuffer ctx batchVBuf
  GPUBackend.freeBuffer ctx batchAttnOutBuf
  GPUBackend.freeBuffer ctx batchOProjBuf
  GPUBackend.freeBuffer ctx batchAttnResidBuf
  GPUBackend.freeBuffer ctx batchGateBuf
  GPUBackend.freeBuffer ctx batchUpBuf
  GPUBackend.freeBuffer ctx batchGeluBuf
  GPUBackend.freeBuffer ctx batchFFNOutBuf
  GPUBackend.freeBuffer ctx colIdxBuf
  match batchPLInputAllOpt with
  | some b => GPUBackend.freeBuffer ctx b
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
    match kcr with
    | some k =>
      let key := hash ("gemma4_ce", name, config.numWorkgroups, config.workgroupSize.x, config.workgroupSize.y, config.workgroupSize.z)
      let ref ← k.getRef key
      GPUBackend.executeWithConfigCached ctx shader namedBufs config key ref
    | none => GPUBackend.execute ctx shader namedBufs config
  let tokenBytes := Hesper.WebGPU.BufferOps.uint32ToBytes tokenId.toUInt32
  writeScalarViaStaging ctx state.tokenBuf 0 state.stagingTokenPtr 0 tokenBytes
  Hesper.WGSL.Execute.withSection "embedLookup" do
    match model.embdFormat with
    | .Q6_K =>
      -- Q6_K on-the-fly dequant lookup
      ce "q6kEmbLookup"
        (Hesper.Quantization.Q6_K.q6kEmbeddingLookupKernel model.config.vocabSize model.config.hiddenSize)
        [("token_ids", state.tokenBuf), ("embedding_table", model.embedding.embeddingTable), ("output", state.buf1)]
        (.dispatch1D model.config.hiddenSize)
    | _ =>
      -- F32 / F16 / Q4_K: use existing Embedding.forward (assumes F32 interpretation)
      Embedding.forward ctx model.embedding state.tokenBuf state.buf1 1 1

  -- Scale embeddings by sqrt(hiddenSize)
  -- Cannot alias input/output in WebGPU, so output to buf2
  Hesper.WGSL.Execute.withSection "embedScale" do
    ce "embedScale"
      (embeddingScaleKernel model.config.hiddenSize model.config.hiddenSize)
      [("input", state.buf1), ("output", state.buf2)]
      (.dispatch1D model.config.hiddenSize)

  -- Step 1b: Per-layer input precomputation (gemma4-iswa.cpp:258-311)
  -- The per_layer_token_embd table is too large (>2 GB) for a single GPU buffer
  -- with the current Dawn limits, so we dequant just the input token's row on
  -- CPU and upload (~43 KB).
  Hesper.WGSL.Execute.withSection "perLayerInputPre" do
    match model.perLayerEmbdTableGPU, model.perLayerModelProj, model.perLayerProjNorm with
    | some embdTableGPU, some modelProj, some projNorm =>
      let embdPL := model.config.embdPerLayer
      let nLayers := model.config.numHiddenLayers
      let totalPL := embdPL * nLayers

      -- 1) Dequant the `tokenId` row from GPU-resident Q6_K table.
      --    Full table on GPU — no CPU→GPU transfer per token.
      Hesper.WGSL.Execute.withSection "plPre.gpuDequant" do
        let tokenIdBytes := Hesper.WebGPU.BufferOps.uint32ToBytes tokenId.toUInt32
        writeScalarViaStaging ctx state.plRawRowBuf 0 state.stagingPLRowPtr 0 tokenIdBytes
        let scaleFactor : Float := Float.sqrt embdPL.toFloat
        ce "q6kDequantScale"
          (Hesper.Quantization.Q6_K.q6kTableRowDequantScaleKernel totalPL scaleFactor
            model.config.vocabSize)
          [("table", embdTableGPU), ("params", state.plRawRowBuf), ("output", state.plModelProj)]
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

      -- Fused (scale + chunked RMSNorm + scaledAdd) — one dispatch over
      -- totalPL elements.  The kernel reads `plTokenSelected` (f16Matmul
      -- output, un-scaled), absorbs the (1/√hidden) pre-norm scale via
      -- `scaleSq` in the variance computation, applies the norm + weight,
      -- then adds `plModelProj` scaled by 1/√2.  Replaces three separate
      -- dispatches (pleScalePL, chunkedRMSNorm, scaledAdd) with one.
      Hesper.WGSL.Execute.withSection "plPre.fusedNormAdd" do
        let preScale : Float := 1.0 / Float.sqrt model.config.hiddenSize.toFloat
        let addScale : Float := 1.0 / Float.sqrt 2.0
        ce "plFusedNormAdd"
          (chunkedRMSNormAddScaledKernel embdPL nLayers model.config.rmsNormEps preScale addScale)
          [("input", state.plTokenSelected), ("weight", projNorm.scale),
           ("residual", state.plModelProj), ("output", state.plInputAll)]
          { numWorkgroups := (nLayers, 1, 1), workgroupSize := { x := min embdPL 256, y := 1, z := 1 } : Hesper.ExecConfig }
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

  dumpBuf ctx currentBuf (model.config.hiddenSize * 4).toUSize s!"single_p{pos}_postPLE"

  let mut blockIdx := 0
  let skipPLE ← do pure (← IO.getEnv "HESPER_SKIP_PLE").isSome
  let plInputBuf := if model.config.hasPerLayerEmbeddings then some state.plInputAll else none
  for block in model.blocks do
    let plEmbd := if blockIdx < model.perLayerBlocks.size && !skipPLE then
      model.perLayerBlocks[blockIdx]!
    else none
    forwardBlock ctx block model.config currentBuf nextBuf state pos (kcr := kcr) (perLayerEmbd := plEmbd) (perLayerInput := plInputBuf)
    let oldCb := currentBuf
    currentBuf := nextBuf
    nextBuf := oldCb
    dumpBuf ctx currentBuf (model.config.hiddenSize * 4).toUSize s!"single_p{pos}_afterL{blockIdx}"
    blockIdx := blockIdx + 1

  -- Step 3: Final norm.  When the Q6_K dp4a lm_head path is available
  -- (Gemma 4's default for embdFormat=Q6_K with dp4a enabled), we defer
  -- emission until lm_head so we can fuse finalNorm + Q8_1 quantize
  -- into one kernel (`fusedRMSNormQ8_1Kernel`).  Otherwise emit the
  -- standalone Circuit-DSL norm so the f32 matmul fallback has a
  -- valid `nextBuf` to read.
  let useFusedNormLmHead ← do
    match model.embdFormat with
    | .Q6_K =>
      let useSubgroups ← GPUBackend.hasSubgroupSupport ctx
      let a ← Hesper.Layers.Linear.dp4aEnabled.get
      let b ← Hesper.Layers.Linear.dp4aQ6KEnabled.get
      pure (a && b && useSubgroups && model.config.hiddenSize % 32 == 0)
    | _ => pure false
  if !useFusedNormLmHead then
    Hesper.WGSL.Execute.withSection "finalNorm" do
      let key := hash ("circuitFinalNorm-cuda", model.finalNorm.config.dim)
      let ccRef ← Hesper.Circuit.getGlobalCircuitRef (β := β) key
      Hesper.Circuit.runCachedFused ctx ccRef
        (do
          let xT ← Hesper.Circuit.CircuitM.registerExternal
            (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
            currentBuf #[model.finalNorm.config.dim] .f32 .Global
          let sT ← Hesper.Circuit.CircuitM.registerExternal
            (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
            model.finalNorm.scale #[model.finalNorm.config.dim] .f32 .Global
          let _y ← Hesper.Circuit.CircuitM.rmsNorm xT sT model.finalNorm.config.eps
          pure ())
        [(0, currentBuf), (1, model.finalNorm.scale), (5, nextBuf)]

  -- Step 4: LM head matmul (1 × hiddenSize @ hiddenSize × vocabSize)
  Hesper.WGSL.Execute.withSection "lmHead" do
    match model.embdFormat with
    | .Q6_K =>
      let useSubgroups ← GPUBackend.hasSubgroupSupport ctx
      let dp4aOn ← do
        let a ← Hesper.Layers.Linear.dp4aEnabled.get
        let b ← Hesper.Layers.Linear.dp4aQ6KEnabled.get
        pure (a && b)
      let gridX : Nat := 4096
      let gridY : Nat := (model.config.vocabSize + gridX - 1) / gridX
      -- DEBUG: when HESPER_DP4A_Q6K_DEBUG is set, run BOTH the f32 and
      -- the dp4a kernel and dump first 8 logits of each for comparison.
      let debugMode ← do
        match ← IO.getEnv "HESPER_DP4A_Q6K_DEBUG" with
        | some "1" => pure true
        | _ => pure false
      if debugMode && dp4aOn then
        IO.println "[Q6K_DEBUG] Running f32 lmHead for comparison..."
        let shaderF32 := if useSubgroups then
            Hesper.Quantization.Q6_K.fusedQ6KLinearBlockCoopKernel
              model.config.hiddenSize model.config.vocabSize gridX
          else
            Hesper.Quantization.Q6_K.fusedQ6KLinearKernel
              model.config.hiddenSize model.config.vocabSize 256 gridX
        let wgSize := if useSubgroups then 32 else 256
        GPUBackend.execute ctx shaderF32
          [("weights", model.outputWeight), ("input", nextBuf), ("output", state.logitsBuf)]
          { numWorkgroups := (gridX, gridY, 1), workgroupSize := { x := wgSize, y := 1, z := 1 }
            extensions := if useSubgroups then ["subgroups"] else []
            : Hesper.ExecConfig }
        let logitsBytes ← GPUBackend.readBuffer ctx state.logitsBuf (132 * 4 : USize)
        IO.print "[Q6K_DEBUG] f32  logits[5000..5007]: "
        for i in [100:108] do
          let o := i * 4
          let bits := (logitsBytes.get! o).toUInt32 |||
                      ((logitsBytes.get! (o+1)).toUInt32 <<< 8) |||
                      ((logitsBytes.get! (o+2)).toUInt32 <<< 16) |||
                      ((logitsBytes.get! (o+3)).toUInt32 <<< 24)
          let e := (bits >>> 23) &&& 0xFF
          let m := bits &&& (0x7FFFFF : UInt32)
          let s := bits >>> 31
          let v := if e == 0 then 0.0 else
            (1.0 + m.toNat.toFloat / 8388608.0) * Float.pow 2.0 (e.toNat.toFloat - 127.0)
          let v := if s == 1 then -v else v
          IO.print s!"{v} "
        IO.println ""

      let _ := debugMode
      if dp4aOn && useSubgroups && model.config.hiddenSize % 32 == 0 then
        -- dp4a path: fused finalNorm+Q8_1 quantize, then Q6_K × Q8_1 matmul.
        -- The fused kernel consumes the pre-norm hidden state directly
        -- from `currentBuf`, so we deliberately skipped the standalone
        -- finalNorm above (see `useFusedNormLmHead`).  Saves 1 dispatch
        -- and the f32 normed round-trip through `nextBuf`.
        let nQ8Blocks := model.config.hiddenSize / 32
        let q8BufBytes : USize := (nQ8Blocks * 9 * 4).toUSize
        let q8Buf ← match ← state.lmHeadQ8Buf.get with
          | some b => pure b
          | none =>
            let b ← GPUBackend.allocBuffer ctx q8BufBytes
            state.lmHeadQ8Buf.set (some b)
            pure b
        -- Fused finalNorm + Q8_1 quantize.
        GPUBackend.executeWithConfigCached ctx
          (Hesper.Layers.RMSNorm.fusedRMSNormQ8_1Kernel model.finalNorm.config)
          [("input", currentBuf), ("scale", model.finalNorm.scale), ("output", q8Buf)]
          { numWorkgroups := (1, 1, 1), workgroupSize := { x := 256, y := 1, z := 1 }
            extensions := ["subgroups"] : Hesper.ExecConfig }
          (hash ("fused-rmsnorm-q8_1-lmhead", model.config.hiddenSize))
          state.lmHeadQuantizePrepared
        -- Q6_K dp4a matmul (2D grid for vocabSize > 65535).
        -- Default is the 4-warp cooperative variant (smem input reuse across
        -- 4 output rows) — fastest on all shapes tested. Override via env:
        --   HESPER_DP4A_Q6K_2ROW=1  → 2-warp variant  (64 threads, 2 rows/WG)
        --   HESPER_DP4A_Q6K_1ROW=1  → single-warp     (32 threads, 1 row/WG)
        let variant ← do
          match ← IO.getEnv "HESPER_DP4A_Q6K_1ROW" with
          | some "1" => pure "1row"
          | _ =>
            match ← IO.getEnv "HESPER_DP4A_Q6K_2ROW" with
            | some "1" => pure "2row"
            | _ => pure "4row"
        match variant with
        | "4row" =>
          let quadCount := (model.config.vocabSize + 3) / 4
          let gridX4 : Nat := 4096
          let gridY4 : Nat := (quadCount + gridX4 - 1) / gridX4
          GPUBackend.executeWithConfigCached ctx
            (Hesper.Layers.Linear.fusedQ6KLinearDP4A4RowKernel
              model.config.hiddenSize model.config.vocabSize gridX4)
            [("weights", model.outputWeight), ("input_q8", q8Buf), ("output", state.logitsBuf)]
            { numWorkgroups := (gridX4, gridY4, 1), workgroupSize := { x := 128, y := 1, z := 1 }
              extensions := ["subgroups"] : Hesper.ExecConfig }
            (hash ("q6k-dp4a-lmhead-4row", model.config.hiddenSize, model.config.vocabSize))
            state.lmHeadDP4APrepared
        | "2row" =>
          let pairCount := (model.config.vocabSize + 1) / 2
          let gridX2 : Nat := 4096
          let gridY2 : Nat := (pairCount + gridX2 - 1) / gridX2
          GPUBackend.executeWithConfigCached ctx
            (Hesper.Layers.Linear.fusedQ6KLinearDP4A2RowKernel
              model.config.hiddenSize model.config.vocabSize gridX2)
            [("weights", model.outputWeight), ("input_q8", q8Buf), ("output", state.logitsBuf)]
            { numWorkgroups := (gridX2, gridY2, 1), workgroupSize := { x := 64, y := 1, z := 1 }
              extensions := ["subgroups"] : Hesper.ExecConfig }
            (hash ("q6k-dp4a-lmhead-2row", model.config.hiddenSize, model.config.vocabSize))
            state.lmHeadDP4APrepared
        | _ =>
          GPUBackend.executeWithConfigCached ctx
            (Hesper.Layers.Linear.fusedQ6KLinearDP4AKernel
              model.config.hiddenSize model.config.vocabSize gridX)
            [("weights", model.outputWeight), ("input_q8", q8Buf), ("output", state.logitsBuf)]
            { numWorkgroups := (gridX, gridY, 1), workgroupSize := { x := 32, y := 1, z := 1 }
              extensions := ["subgroups"] : Hesper.ExecConfig }
            (hash ("q6k-dp4a-lmhead", model.config.hiddenSize, model.config.vocabSize))
            state.lmHeadDP4APrepared
        if debugMode then
          -- Read logits[5000..5007] (skip reserved tokens which are often 0)
          let logitsBytes ← GPUBackend.readBuffer ctx state.logitsBuf (5008 * 4 : USize)
          IO.print "[Q6K_DEBUG] dp4a logits[5000..5007]: "
          for i in [5000:5008] do
            let o := i * 4
            let bits := (logitsBytes.get! o).toUInt32 |||
                        ((logitsBytes.get! (o+1)).toUInt32 <<< 8) |||
                        ((logitsBytes.get! (o+2)).toUInt32 <<< 16) |||
                        ((logitsBytes.get! (o+3)).toUInt32 <<< 24)
            let e := (bits >>> 23) &&& 0xFF
            let m := bits &&& (0x7FFFFF : UInt32)
            let s := bits >>> 31
            let v := if e == 0 then 0.0 else
              (1.0 + m.toNat.toFloat / 8388608.0) * Float.pow 2.0 (e.toNat.toFloat - 127.0)
            let v := if s == 1 then -v else v
            IO.print s!"{v} "
          IO.println ""
      else
        -- Original f32 path (block-coop with subgroups or 256-thread tree reduction).
        let shader := if useSubgroups then
            Hesper.Quantization.Q6_K.fusedQ6KLinearBlockCoopKernel
              model.config.hiddenSize model.config.vocabSize gridX
          else
            Hesper.Quantization.Q6_K.fusedQ6KLinearKernel
              model.config.hiddenSize model.config.vocabSize 256 gridX
        let wgSize := if useSubgroups then 32 else 256
        let lmBufs : List (String × GPUBackend.Buf β) :=
          [("weights", model.outputWeight), ("input", state.buf1), ("output", state.logitsBuf)]
        ce "lmHead"
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
    ce "pleScaleVocab"
      (PerLayerEmbedding.scaleKernel model.config.vocabSize 1.0)
      [("input", state.logitsBuf), ("output", psBuf)]
      (.dispatch1D model.config.vocabSize)
  | none => pure ()

  -- Step 5: Logit softcapping (y = scale * tanh(x / scale))
  Hesper.WGSL.Execute.withSection "logitSoftcap" do
    if model.config.logitSoftcapScale > 0.0 then
      ce "logitSoftcap"
        (logitSoftcapKernel model.config.vocabSize model.config.logitSoftcapScale)
        [("input", state.logitsBuf), ("output", state.logitsBuf2)]
        (.dispatch1D model.config.vocabSize)
      ce "pleScaleVocab2"
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
  let useBatch := promptTokens.size > 1
    && (match ← IO.getEnv "HESPER_BATCH_PREFILL" with | some "0" => false | _ => true)
  if useBatch then
    IO.println s!"[Prefill] Batched path (seqLen={promptTokens.size})"
    forwardPrefillBatch ctx model promptTokens state (kcr := some kcr)
  else
    for i in [0:promptTokens.size] do
      if i >= model.config.maxSeqLen then break
      forwardSingleToken ctx model promptTokens[i]! i state (kcr := some kcr)
  let prefillEnd ← IO.monoNanosNow
  let prefillMs := (prefillEnd - prefillStart).toFloat / 1_000_000.0
  IO.println s!"[Prefill] Done in {prefillMs} ms"

  -- Phase 2: Decode (generate new tokens)
  let genStart ← IO.monoNanosNow
  let mut genCount : Nat := 0

  -- CUDA Graph capture+replay: env-gated experimental path.
  -- When HESPER_CUDA_GRAPHS=1, we run the FIRST decode forward pass
  -- inside a relaxed stream capture, harvest the resulting graph, and
  -- replay it on subsequent tokens.  Host-side writes (tokenId, pos)
  -- happen BEFORE each replay via default-stream cuMemcpyHtoD — those
  -- are sync to host and effectively "bake in" at execute time.
  --
  -- IMPORTANT: this relies on the decode forward being shape-stable —
  -- true for Gemma 4's single-token path since all kernels have
  -- compile-time-fixed dispatch shapes (one per-layer set).
  let useCudaGraphs := (← IO.getEnv "HESPER_CUDA_GRAPHS").isSome

  let mut graphExecOpt : Option (Hesper.CUDA.CUgraphExec × Hesper.CUDA.CUstream) := none

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
      match graphExecOpt with
      | some (exec, stream) =>
        -- Replay path.  The captured graph contains memcpy nodes whose
        -- host sources are the pinned buffers at state.stagingTokenPtr /
        -- stagingParamsPtr / stagingPLRowPtr (see writeScalarViaStaging).
        -- Update those slots with the new token/pos/cacheLen BEFORE
        -- launching the graph so the memcpy nodes pick up fresh values.
        let tokenBytes := Hesper.WebGPU.BufferOps.uint32ToBytes nextToken.toUInt32
        let posBytes := Hesper.WebGPU.BufferOps.uint32ToBytes newPos.toUInt32
        let cacheLenBytes := Hesper.WebGPU.BufferOps.uint32ToBytes (newPos + 1).toUInt32
        let plRowBytes := tokenBytes
        let posF32Bytes ← Hesper.Basic.floatToBytes newPos.toFloat
        Hesper.CUDA.cuWritePinned state.stagingTokenPtr  0 tokenBytes 4
        Hesper.CUDA.cuWritePinned state.stagingParamsPtr 0 posBytes 4
        Hesper.CUDA.cuWritePinned state.stagingParamsPtr 4 cacheLenBytes 4
        Hesper.CUDA.cuWritePinned state.stagingPLRowPtr  0 plRowBytes 4
        Hesper.CUDA.cuWritePinned state.stagingPosF32Ptr 0 posF32Bytes 4
        Hesper.CUDA.cuGraphLaunch exec stream
        Hesper.CUDA.cuStreamSynchronize stream
      | none =>
        if useCudaGraphs && genCount == 1 then
          -- Capture path: run forwardSingleToken on a capture stream.
          let stream ← Hesper.CUDA.cuStreamCreate
          Hesper.cudaCaptureStream.set (some stream)
          Hesper.CUDA.cuStreamBeginCapture stream
          forwardSingleToken ctx model nextToken newPos state (kcr := some kcr)
          let graph ← Hesper.CUDA.cuStreamEndCapture stream
          Hesper.cudaCaptureStream.set none
          let exec ← Hesper.CUDA.cuGraphInstantiate graph
          Hesper.CUDA.cuGraphDestroy graph
          Hesper.CUDA.cuStreamSynchronize stream
          graphExecOpt := some (exec, stream)
          IO.println s!"[Graph] captured decode graph; subsequent tokens will replay"
        else
          forwardSingleToken ctx model nextToken newPos state (kcr := some kcr)

  -- Clean up graph resources.
  match graphExecOpt with
  | some (exec, stream) =>
    Hesper.CUDA.cuGraphExecDestroy exec
    Hesper.CUDA.cuStreamDestroy stream
  | none => pure ()

  let genEnd ← IO.monoNanosNow
  let genMs := (genEnd - genStart).toFloat / 1_000_000.0
  let msPerToken := if genCount > 0 then genMs / genCount.toFloat else 0.0
  let tps := if msPerToken > 0 then 1000.0 / msPerToken else 0.0
  IO.println s!"[Gemma4] Generated {genCount} tokens in {genMs} ms ({tps} tokens/sec)"

  return tokens

end Hesper.Models.Gemma4
