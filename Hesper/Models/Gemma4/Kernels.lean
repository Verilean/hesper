import Hesper.WGSL.Monad
import Hesper.WGSL.Exp
import Hesper.Models.Gemma4.Config

/-!
# Gemma 4 shader kernels

Pure ShaderM kernel definitions extracted from the main Gemma4 module so
that runtime-heavy code does not get recompiled every time we edit a
kernel.  All kernels here are pointwise / elementwise / RoPE / norm
helpers.  Fused matmul kernels live in `Hesper.Layers.Linear`.
-/

namespace Hesper.Models.Gemma4

open Hesper.WGSL
open Hesper.WGSL.Monad

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

/-- In-place batched broadcast multiply: `buf[i] *= scale[0]` for all
    `i in [0, total)`.  Replaces the per-column
    `extract → runCachedFused(scaleByBroadcast) → insert` chain used by
    Gemma 4's per-layer output-scale step.  One dispatch over the whole
    `[dim, seqLen]` batch tensor; the scale is a single f32. -/
def batchBroadcastScaleInPlaceKernel (total : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid
  let _buf ← ShaderM.declareStorageBuffer "buf"
    (.array (.scalar .f32) total) .readWrite
  let _scale ← ShaderM.declareInputBuffer "scale" (.array (.scalar .f32) 1)
  ShaderM.if_ (Exp.lt idx (Exp.litU32 total)) (do
    let s ← ShaderM.readBuffer (ty := .scalar .f32) (n := 1) "scale" (Exp.litU32 0)
    let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := total) "buf" idx
    ShaderM.writeBuffer (ty := .scalar .f32) "buf" idx (Exp.mul v s)
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

    Dispatch: `(1, 1, 1)` workgroups × `wgSize` threads. -/
def fusedPerLayerPostKernel (hiddenSize : Nat) (eps : Float)
    (wgSize : Nat := 256) : ShaderM Unit := do
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

/-- `fusedPerLayerPostKernel` with a tail broadcast scalar multiply:
    `residual[i] = (residual[i] + rmsNorm(proj)[i] * weight[i]) * outScale[0]`.

    Folds the `layerOutScale` dispatch into `ple.postNormAdd` when the
    block has a per-layer output scale, saving one dispatch per decode
    layer (-42 for a 42-layer model). -/
def fusedPerLayerPostThenScaleKernel (hiddenSize : Nat) (eps : Float)
    (wgSize : Nat := 256) : ShaderM Unit := do
  let numSubgroups := wgSize / 32
  let lid ← ShaderM.localId
  let tid := Exp.vec3X lid

  let _proj ← ShaderM.declareInputBuffer "proj" (.array (.scalar .f32) hiddenSize)
  let _weight ← ShaderM.declareInputBuffer "weight" (.array (.scalar .f32) hiddenSize)
  let _outScale ← ShaderM.declareInputBuffer "out_scale" (.array (.scalar .f32) 1)
  let _residual ← ShaderM.declareOutputBuffer "residual" (.array (.scalar .f32) hiddenSize)

  ShaderM.sharedNamed "shared_sg" (.array (.scalar .f32) (numSubgroups + 1))

  ShaderM.varNamed "partialSq" (.scalar .f32) (Exp.litF32 0.0)
  let partialSq : Exp (.scalar .f32) := Exp.var "partialSq"
  ShaderM.loop tid (Exp.litU32 hiddenSize) (Exp.litU32 wgSize) fun d => do
    let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := hiddenSize) "proj" d
    ShaderM.assign "partialSq" (Exp.add partialSq (Exp.mul v v))
  ShaderM.varNamed "sgSum" (.scalar .f32) (Exp.subgroupAdd partialSq)
  let sgSum : Exp (.scalar .f32) := Exp.var "sgSum"
  let subgroupId := Exp.div tid (Exp.litU32 32)
  let laneId := Exp.sub tid (Exp.mul subgroupId (Exp.litU32 32))
  ShaderM.if_ (Exp.eq laneId (Exp.litU32 0)) (do
    ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_sg" subgroupId sgSum
  ) (pure ())
  ShaderM.barrier
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

  ShaderM.varNamed "invRms" (.scalar .f32)
    (← ShaderM.readWorkgroup (ty := .scalar .f32) (n := numSubgroups + 1)
          "shared_sg" (Exp.litU32 numSubgroups))
  let invRms : Exp (.scalar .f32) := Exp.var "invRms"
  let outSc ← ShaderM.readBuffer (ty := .scalar .f32) (n := 1) "out_scale" (Exp.litU32 0)

  ShaderM.loop tid (Exp.litU32 hiddenSize) (Exp.litU32 wgSize) fun d => do
    let pVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := hiddenSize) "proj" d
    let wVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := hiddenSize) "weight" d
    let rVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := hiddenSize) "residual" d
    let normed := Exp.mul (Exp.mul pVal invRms) wVal
    let combined := Exp.add rVal normed
    ShaderM.writeBuffer (ty := .scalar .f32) "residual" d (Exp.mul combined outSc)

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

/-- Batched per-head RMSNorm: same math as `perHeadRMSNormKernel`, but
    processes all `seqLen` tokens in one dispatch.

    Input layout: `[numHeads * headDim, seqLen]` column-major.
    Output layout: same.  Each workgroup handles one (head, token)
    pair: `wid.x = head`, `wid.y = token`.  In-place is safe (each
    thread reads and writes its own element). -/
def perHeadRMSNormBatchKernel (numHeads headDim seqLen : Nat) (eps : Float) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let headIdx := Exp.vec3X wid
  let tokIdx  := Exp.vec3Y wid
  let tid := Exp.vec3X lid

  let qDim := numHeads * headDim
  let totalElements := qDim * seqLen

  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) totalElements)
  let _weight ← ShaderM.declareInputBuffer "weight" (.array (.scalar .f32) headDim)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) totalElements)

  let wgSize := if headDim < 256 then headDim else 256
  ShaderM.sharedNamed "shared_sum" (.array (.scalar .f32) wgSize)

  -- Offset into the batch buffer for this (token, head): col-major
  -- column stride = qDim; within the column, head starts at head*headDim.
  let colBase := Exp.mul tokIdx (Exp.litU32 qDim)
  let headBase := Exp.add colBase (Exp.mul headIdx (Exp.litU32 headDim))

  ShaderM.varNamed "local_sum" (.scalar .f32) (Exp.litF32 0.0)
  let localSum : Exp (.scalar .f32) := Exp.var "local_sum"

  ShaderM.loop tid (Exp.litU32 headDim) (Exp.litU32 wgSize) fun i => do
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalElements) "input" (Exp.add headBase i)
    ShaderM.assign "local_sum" (Exp.add localSum (Exp.mul val val))

  ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_sum" tid localSum
  ShaderM.barrier

  let mut stride := wgSize / 2
  while stride > 0 do
    ShaderM.if_ (Exp.lt tid (Exp.litU32 stride)) (do
      let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := wgSize) "shared_sum" tid
      let b ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := wgSize) "shared_sum" (Exp.add tid (Exp.litU32 stride))
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_sum" tid (Exp.add a b)
    ) (pure ())
    ShaderM.barrier
    stride := stride / 2

  let sumSq ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := wgSize) "shared_sum" (Exp.litU32 0)
  let rms := Exp.inverseSqrt (Exp.add (Exp.div sumSq (Exp.litF32 headDim.toFloat)) (Exp.litF32 eps))

  ShaderM.loop tid (Exp.litU32 headDim) (Exp.litU32 wgSize) fun i => do
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalElements) "input" (Exp.add headBase i)
    let w ← ShaderM.readBuffer (ty := .scalar .f32) (n := headDim) "weight" i
    let normed := Exp.mul (Exp.mul val rms) w
    ShaderM.writeBuffer (ty := .scalar .f32) "output" (Exp.add headBase i) normed

/-! ## Bare RMSNorm Kernels (no learned weights) -/

/-- IN-PLACE variant of perHeadRMSNormBatchKernel (single read_write q_io binding): binding the
    same buffer as separate input/output entries is writable-storage aliasing — a WebGPU
    validation error. Per-element read→scale→write same-thread same-index, so in-place is safe. -/
def perHeadRMSNormBatchInPlaceKernel (numHeads headDim seqLen : Nat) (eps : Float) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let headIdx := Exp.vec3X wid
  let tokIdx  := Exp.vec3Y wid
  let tid := Exp.vec3X lid

  let qDim := numHeads * headDim
  let totalElements := qDim * seqLen

  let _weight ← ShaderM.declareInputBuffer "weight" (.array (.scalar .f32) headDim)
  let _output ← ShaderM.declareOutputBuffer "q_io" (.array (.scalar .f32) totalElements)

  let wgSize := if headDim < 256 then headDim else 256
  ShaderM.sharedNamed "shared_sum" (.array (.scalar .f32) wgSize)

  -- Offset into the batch buffer for this (token, head): col-major
  -- column stride = qDim; within the column, head starts at head*headDim.
  let colBase := Exp.mul tokIdx (Exp.litU32 qDim)
  let headBase := Exp.add colBase (Exp.mul headIdx (Exp.litU32 headDim))

  ShaderM.varNamed "local_sum" (.scalar .f32) (Exp.litF32 0.0)
  let localSum : Exp (.scalar .f32) := Exp.var "local_sum"

  ShaderM.loop tid (Exp.litU32 headDim) (Exp.litU32 wgSize) fun i => do
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalElements) "q_io" (Exp.add headBase i)
    ShaderM.assign "local_sum" (Exp.add localSum (Exp.mul val val))

  ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_sum" tid localSum
  ShaderM.barrier

  let mut stride := wgSize / 2
  while stride > 0 do
    ShaderM.if_ (Exp.lt tid (Exp.litU32 stride)) (do
      let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := wgSize) "shared_sum" tid
      let b ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := wgSize) "shared_sum" (Exp.add tid (Exp.litU32 stride))
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_sum" tid (Exp.add a b)
    ) (pure ())
    ShaderM.barrier
    stride := stride / 2

  let sumSq ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := wgSize) "shared_sum" (Exp.litU32 0)
  let rms := Exp.inverseSqrt (Exp.add (Exp.div sumSq (Exp.litF32 headDim.toFloat)) (Exp.litF32 eps))

  ShaderM.loop tid (Exp.litU32 headDim) (Exp.litU32 wgSize) fun i => do
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalElements) "q_io" (Exp.add headBase i)
    let w ← ShaderM.readBuffer (ty := .scalar .f32) (n := headDim) "weight" i
    let normed := Exp.mul (Exp.mul val rms) w
    ShaderM.writeBuffer (ty := .scalar .f32) "q_io" (Exp.add headBase i) normed

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
  let _qScale ← ShaderM.declareInputBuffer  "q_scale" (.array (.scalar .f32) headDim)
  let _qOut   ← ShaderM.declareOutputBuffer "q_io"   (.array (.scalar .f32) (qTotal * seqLen))
  let _kScale ← ShaderM.declareInputBuffer  "k_scale" (.array (.scalar .f32) headDim)
  let _kOut   ← ShaderM.declareOutputBuffer "k_io"   (.array (.scalar .f32) (kvTotal * seqLen))
  let _vOut   ← ShaderM.declareOutputBuffer "v_io"   (.array (.scalar .f32) (kvTotal * seqLen))
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
        let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := qTotal * seqLen) "q_io"
                  (Exp.add qColBase i)
        ShaderM.assign "local_sum" (Exp.add localSum (Exp.mul v v))
      ) (do
        ShaderM.if_ (Exp.eq yIdx (Exp.litU32 1)) (do
          let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := kvTotal * seqLen) "k_io"
                    (Exp.add kvColBase i)
          ShaderM.assign "local_sum" (Exp.add localSum (Exp.mul v v))
        ) (do
          let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := kvTotal * seqLen) "v_io"
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
        let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := qTotal * seqLen) "q_io"
                  (Exp.add qColBase i)
        let w ← ShaderM.readBuffer (ty := .scalar .f32) (n := headDim) "q_scale" i
        ShaderM.writeBuffer (ty := .scalar .f32) "q_io" (Exp.add qColBase i)
          (Exp.mul (Exp.mul v rmsRef) w)
      ) (do
        ShaderM.if_ (Exp.eq yIdx (Exp.litU32 1)) (do
          let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := kvTotal * seqLen) "k_io"
                    (Exp.add kvColBase i)
          let w ← ShaderM.readBuffer (ty := .scalar .f32) (n := headDim) "k_scale" i
          ShaderM.writeBuffer (ty := .scalar .f32) "k_io" (Exp.add kvColBase i)
            (Exp.mul (Exp.mul v rmsRef) w)
        ) (do
          let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := kvTotal * seqLen) "v_io"
                    (Exp.add kvColBase i)
          ShaderM.writeBuffer (ty := .scalar .f32) "v_io" (Exp.add kvColBase i)
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

end Hesper.Models.Gemma4
