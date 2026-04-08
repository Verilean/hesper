import Hesper.WGSL.Monad
import Hesper.WGSL.Execute
import Hesper.WGSL.Exp
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.Training.VerifiedBackward

/-!
# Flash Attention (Fused Tiled Attention)

Computes attention in a single kernel by tiling over the sequence dimension,
using shared memory for intermediate results (scores, softmax, weighted sum).

## Equivalence to Standard Attention

Standard attention (3 separate kernels):
```
scores[h,s] = scale * Σ_d Q[h,d] * K[kvHead,s,d]     -- score kernel
attn[h,s] = softmax(scores[h,:])                       -- softmax kernel
output[h,d] = Σ_s attn[h,s] * V[kvHead,s,d]           -- apply kernel
```

Flash attention (1 fused kernel):
Same computation, but scores and attn stay in shared memory.
No global memory write for intermediate scores/attn.

## Proof of Equivalence

The CPU spec functions `scaledDotForward`, `softmaxForward`, and
`attentionForward` are composed. Flash attention computes the same
composition but without materializing intermediates:

```
flashAttention(Q, K, V, scale) = standard_attention(Q, K, V, scale)
```

This is verified numerically by `verifyFlashEquivalence`.

## Memory Savings

Standard: O(numHeads × seqLen) global memory for scores + attn
Flash: O(workgroupSize) shared memory only
For seqLen=2048, numHeads=20: 160KB → ~4KB (40x reduction)
-/

namespace Hesper.WGSL.FlashAttention

open Hesper.WGSL
open Hesper.WGSL.Monad
open Hesper.WebGPU

/-! ## CPU Spec (for equivalence proof) -/

/-- Standard attention: score → softmax → apply (3 steps) -/
def standardAttention (q : Array Float) (kCache vCache : Array (Array Float))
    (scale : Float) : Array Float :=
  Hesper.Training.VerifiedBackward.attentionForward q kCache vCache scale

/-- Flash attention CPU spec: same result, computed differently.
    This is intentionally written to show the tiled computation pattern. -/
def flashAttentionSpec (q : Array Float) (kCache vCache : Array (Array Float))
    (scale : Float) : Array Float :=
  let headDim := q.size
  let seqLen := kCache.size
  -- Online softmax: process one K/V at a time, maintaining running max and sum
  let init := (Array.replicate headDim 0.0, -1e30, 0.0)  -- (acc, maxScore, sumExp)
  let (acc, _maxScore, sumExp) := Id.run do
    let mut acc := Array.replicate headDim 0.0
    let mut maxScore := -1e30
    let mut sumExp := 0.0
    for s in [:seqLen] do
      let k := kCache.getD s #[]
      let v := vCache.getD s #[]
      -- Compute score for position s
      let mut score := 0.0
      for d in [:headDim] do
        score := score + q.getD d 0.0 * k.getD d 0.0
      score := score * scale
      -- Online softmax update
      let newMax := max maxScore score
      let expOld := Float.exp (maxScore - newMax)
      let expNew := Float.exp (score - newMax)
      let newSum := sumExp * expOld + expNew
      -- Rescale accumulated output and add new contribution
      for d in [:headDim] do
        let oldAcc := acc.getD d 0.0
        acc := acc.set! d (oldAcc * (sumExp * expOld / newSum) + v.getD d 0.0 * (expNew / newSum))
      maxScore := newMax
      sumExp := newSum
    pure (acc, maxScore, sumExp)
  acc

/-- Verify flash attention produces same output as standard attention -/
def verifyFlashEquivalence (tol : Float := 1e-4) : Bool := Id.run do
  -- Test case: 4-dim head, 3 sequence positions
  let q := #[1.0, 0.5, -0.3, 0.8]
  let kCache := #[#[0.5, 1.0, 0.2, -0.5], #[-0.3, 0.8, 1.0, 0.1], #[0.7, -0.2, 0.5, 0.9]]
  let vCache := #[#[1.0, 0.0, 0.5, -0.3], #[0.2, 1.0, -0.5, 0.8], #[-0.1, 0.5, 1.0, 0.2]]
  let scale := 0.5

  let standard := standardAttention q kCache vCache scale
  let flash := flashAttentionSpec q kCache vCache scale

  let mut maxErr := 0.0
  for i in [:standard.size] do
    let s := standard.getD i 0.0
    let f := flash.getD i 0.0
    let diff := if s - f < 0.0 then f - s else s - f
    let denom := (if s < 0.0 then -s else s) + (if f < 0.0 then -f else f)
    let err := if denom < 1e-10 then diff else diff / denom
    if err > maxErr then maxErr := err

  return maxErr < tol

/-! ## GPU Kernel: Flash Attention Forward (single-token KV cache) -/

/-- Flash attention forward kernel for single-token query with KV cache.
    One workgroup per head. Each workgroup:
    1. Loads Q for this head from global memory
    2. Iterates over cached K/V positions, computing online softmax
    3. Writes final output to global memory

    No intermediate scores/attn buffers needed.

    @param numHeads Number of query heads
    @param numKVHeads Number of KV heads (GQA)
    @param cacheLen Number of positions in KV cache
    @param headDim Dimension per head
    @param scale 1/sqrt(headDim) -/
def flashAttentionDynamicKernel (numHeads numKVHeads maxSeqLen headDim cacheLen : Nat)
    (scale : Float) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let wgid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let head := Exp.vec3X wgid   -- head index
  let tid := Exp.vec3X lid      -- thread within workgroup

  let headsPerKV := numHeads / numKVHeads
  let kvHead := Exp.div head (Exp.litU32 headsPerKV)

  let _q ← ShaderM.declareInputBuffer "q" (.array (.scalar .f32) (numHeads * headDim))
  let _kCache ← ShaderM.declareInputBuffer "k_cache" (.array (.scalar .f32) (numKVHeads * maxSeqLen * headDim))
  let _vCache ← ShaderM.declareInputBuffer "v_cache" (.array (.scalar .f32) (numKVHeads * maxSeqLen * headDim))
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) (numHeads * headDim))

  -- Shared memory for partial score reduction, Q cache, and per-dim output accumulator
  -- shared_out lets us support headDim > workgroupSize: each thread handles
  -- multiple output dims via a strided loop (d = tid, tid+wgSize, tid+2*wgSize, ...).
  ShaderM.sharedNamed "shared_q" (.array (.scalar .f32) headDim)
  ShaderM.sharedNamed "shared_reduce" (.array (.scalar .f32) workgroupSize)
  ShaderM.sharedNamed "shared_out" (.array (.scalar .f32) headDim)

  -- No bounds check needed: numWorkgroups == numHeads, all workgroups are valid
  -- (Removing if_ avoids WGSL "barrier in non-uniform control flow" error)
  do
    -- Step 1: Load Q for this head into shared memory and zero out the output accumulator
    let qBase := Exp.mul head (Exp.litU32 headDim)
    ShaderM.loop tid (Exp.litU32 headDim) (Exp.litU32 workgroupSize) fun d => do
      let qVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numHeads * headDim) "q" (Exp.add qBase d)
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_q" d qVal
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_out" d (Exp.litF32 0.0)
    ShaderM.barrier

    -- Online softmax state (per-thread, but identical across threads after the
    -- score reduction since every thread reads the same scoreFromShared).
    ShaderM.varNamed "max_score" (.scalar .f32) (Exp.litF32 (-1.0e30))
    ShaderM.varNamed "sum_exp" (.scalar .f32) (Exp.litF32 0.0)
    let maxScore := Exp.var "max_score"
    let sumExp := Exp.var "sum_exp"

    -- Iterate over ALL positions up to maxSeqLen (uniform loop bound)
    -- cacheLen is compile-time constant (shader recompiled per position, cached by pipeline cache)
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 cacheLen) (Exp.litU32 1) fun s => do
      let kBase := Exp.add (Exp.mul (Exp.mul kvHead (Exp.litU32 maxSeqLen)) (Exp.litU32 headDim))
                            (Exp.mul s (Exp.litU32 headDim))

      -- Score = Q · K (dot product across full headDim)
      let partialVar ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
      ShaderM.loop tid (Exp.litU32 headDim) (Exp.litU32 workgroupSize) fun d => do
        let qVal ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := headDim) "shared_q" d
        let kVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numKVHeads * maxSeqLen * headDim) "k_cache" (Exp.add kBase d)
        ShaderM.assign partialVar (Exp.add (Exp.var partialVar) (Exp.mul qVal kVal))

      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_reduce" tid (Exp.var partialVar)
      ShaderM.barrier

      let numSteps := Nat.log2 workgroupSize
      ShaderM.staticLoop numSteps fun step => do
        let stride := workgroupSize >>> (step + 1)
        ShaderM.if_ (Exp.lt tid (Exp.litU32 stride)) (do
          let other ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_reduce" (Exp.add tid (Exp.litU32 stride))
          let cur ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_reduce" tid
          ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_reduce" tid (Exp.add cur other)
        ) (pure ())
        ShaderM.barrier

      let scoreFromShared ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_reduce" (Exp.litU32 0)
      let scaledScore := Exp.mul (Exp.litF32 scale) scoreFromShared

      let oldMaxVar ← ShaderM.var (.scalar .f32) maxScore
      let oldSumVar ← ShaderM.var (.scalar .f32) sumExp
      let oldMax := Exp.var oldMaxVar
      let oldSum := Exp.var oldSumVar

      let newMax := Exp.max oldMax scaledScore
      let expOld := Exp.exp (Exp.sub oldMax newMax)
      let expNew := Exp.exp (Exp.sub scaledScore newMax)
      let newSum := Exp.add (Exp.mul oldSum expOld) expNew

      let rescaleFactor := Exp.div (Exp.mul oldSum expOld) newSum
      let contribFactor := Exp.div expNew newSum

      -- Update output accumulator across ALL dims via strided loop.
      -- All threads agree on rescaleFactor / contribFactor since the reduction
      -- broadcasts the same scoreFromShared to every thread.
      ShaderM.loop tid (Exp.litU32 headDim) (Exp.litU32 workgroupSize) fun d => do
        let vVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numKVHeads * maxSeqLen * headDim) "v_cache" (Exp.add kBase d)
        let prev ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := headDim) "shared_out" d
        let updated := Exp.add (Exp.mul prev rescaleFactor) (Exp.mul vVal contribFactor)
        ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_out" d updated

      ShaderM.assign "max_score" newMax
      ShaderM.assign "sum_exp" newSum
      ShaderM.barrier

    -- Step 3: Write output (strided loop covers all headDim elements)
    ShaderM.loop tid (Exp.litU32 headDim) (Exp.litU32 workgroupSize) fun d => do
      let v ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := headDim) "shared_out" d
      let outIdx := Exp.add (Exp.mul head (Exp.litU32 headDim)) d
      ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx v

/-! ## Dynamic Flash Attention with params buffer (production) -/

/-- Flash attention with dynamic cacheLen from params buffer.
    Same as in-place kernel but reads cacheLen from params[1] (u32).
    Uses diagnostic(off, derivative_uniformity) to allow barrier. -/
def flashAttentionParamsKernel (numHeads numKVHeads maxSeqLen headDim : Nat)
    (scale : Float) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let wgid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let head := Exp.vec3X wgid
  let tid := Exp.vec3X lid

  let headsPerKV := numHeads / numKVHeads
  let kvHead := Exp.div head (Exp.litU32 headsPerKV)

  let _qOutput ← ShaderM.declareOutputBuffer "q_output" (.array (.scalar .f32) (numHeads * headDim))
  let _kCache ← ShaderM.declareStorageBuffer "k_cache" (.array (.scalar .f32) (numKVHeads * maxSeqLen * headDim)) .read
  let _vCache ← ShaderM.declareStorageBuffer "v_cache" (.array (.scalar .f32) (numKVHeads * maxSeqLen * headDim)) .read
  -- params must be read-only storage for WGSL uniformity analysis
  -- (read_write storage is considered non-uniform; read is uniform)
  let _params ← ShaderM.declareStorageBuffer "params" (.array (.scalar .u32) 2) .read

  ShaderM.sharedNamed "shared_q" (.array (.scalar .f32) headDim)
  ShaderM.sharedNamed "shared_reduce" (.array (.scalar .f32) workgroupSize)

  -- Read dynamic cacheLen from params
  let cacheLen ← ShaderM.readBuffer (ty := .scalar .u32) (n := 2) "params" (Exp.litU32 1)

  -- Load Q into shared memory
  let qBase := Exp.mul head (Exp.litU32 headDim)
  ShaderM.loop tid (Exp.litU32 headDim) (Exp.litU32 workgroupSize) fun d => do
    let qVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numHeads * headDim) "q_output" (Exp.add qBase d)
    ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_q" d qVal
  ShaderM.barrier

  ShaderM.varNamed "max_score" (.scalar .f32) (Exp.litF32 (-1.0e30))
  ShaderM.varNamed "sum_exp" (.scalar .f32) (Exp.litF32 0.0)
  ShaderM.varNamed "out_acc" (.scalar .f32) (Exp.litF32 0.0)
  let maxScore := Exp.var "max_score"
  let sumExp := Exp.var "sum_exp"
  let outAcc := Exp.var "out_acc"

  -- Dynamic loop over cacheLen (diagnostic off for uniformity)
  ShaderM.loop (Exp.litU32 0) cacheLen (Exp.litU32 1) fun s => do
    let kBase := Exp.add (Exp.mul (Exp.mul kvHead (Exp.litU32 maxSeqLen)) (Exp.litU32 headDim))
                          (Exp.mul s (Exp.litU32 headDim))

    let partialVar ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    ShaderM.loop tid (Exp.litU32 headDim) (Exp.litU32 workgroupSize) fun d => do
      let qVal ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := headDim) "shared_q" d
      let kVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numKVHeads * maxSeqLen * headDim) "k_cache" (Exp.add kBase d)
      ShaderM.assign partialVar (Exp.add (Exp.var partialVar) (Exp.mul qVal kVal))

    ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_reduce" tid (Exp.var partialVar)
    ShaderM.barrier

    let numSteps := Nat.log2 workgroupSize
    ShaderM.staticLoop numSteps fun step => do
      let stride := workgroupSize >>> (step + 1)
      ShaderM.if_ (Exp.lt tid (Exp.litU32 stride)) (do
        let other ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_reduce" (Exp.add tid (Exp.litU32 stride))
        let cur ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_reduce" tid
        ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_reduce" tid (Exp.add cur other)
      ) (pure ())
      ShaderM.barrier

    let scoreFromShared ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_reduce" (Exp.litU32 0)
    let scaledScore := Exp.mul (Exp.litF32 scale) scoreFromShared

    let oldMaxVar ← ShaderM.var (.scalar .f32) maxScore
    let oldSumVar ← ShaderM.var (.scalar .f32) sumExp
    let oldMax := Exp.var oldMaxVar
    let oldSum := Exp.var oldSumVar

    let newMax := Exp.max oldMax scaledScore
    let expOld := Exp.exp (Exp.sub oldMax newMax)
    let expNew := Exp.exp (Exp.sub scaledScore newMax)
    let newSum := Exp.add (Exp.mul oldSum expOld) expNew

    ShaderM.if_ (Exp.lt tid (Exp.litU32 headDim)) (do
      let vVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numKVHeads * maxSeqLen * headDim) "v_cache" (Exp.add kBase tid)
      let rescaled := Exp.mul outAcc (Exp.div (Exp.mul oldSum expOld) newSum)
      let newContrib := Exp.mul vVal (Exp.div expNew newSum)
      ShaderM.assign "out_acc" (Exp.add rescaled newContrib)
    ) (pure ())

    ShaderM.assign "max_score" newMax
    ShaderM.assign "sum_exp" newSum
    ShaderM.barrier

  -- Write output (overwrites Q in same buffer)
  ShaderM.if_ (Exp.lt tid (Exp.litU32 headDim)) (do
    let outIdx := Exp.add (Exp.mul head (Exp.litU32 headDim)) tid
    ShaderM.writeBuffer (ty := .scalar .f32) "q_output" outIdx outAcc
  ) (pure ())

/-- Execute flash attention with params buffer (dynamic cacheLen, 1 dispatch).
    Same WGSL source for all cacheLen → 100% pipeline cache hit rate. -/
def executeFlashAttentionWithParams (device : Device)
    (qBuf kCacheBuf vCacheBuf paramsBuf outputBuf : Buffer)
    (numHeads numKVHeads maxSeqLen headDim : Nat) (scale : Float) : IO Unit := do
  let workgroupSize := min 256 (max headDim 32)
  let shader := flashAttentionParamsKernel numHeads numKVHeads maxSeqLen headDim scale workgroupSize
  let namedBuffers := [("q_output", outputBuf), ("k_cache", kCacheBuf), ("v_cache", vCacheBuf), ("params", paramsBuf)]
  -- Static cache key: same WGSL for all cacheLen (cacheLen is read from params buffer)
  let cacheKey : UInt64 := hash ("flashP", numHeads, numKVHeads, maxSeqLen, headDim)
  let execConfig : Hesper.WGSL.Execute.ExecutionConfig := {
    workgroupSize := {x := workgroupSize, y := 1, z := 1}
    numWorkgroups := (numHeads, 1, 1)
    -- No diagnostic needed: params is var<storage, read> which is uniform
  }
  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig (some cacheKey)

/-! ## In-Place Flash Attention (single tile, no merge) -/

/-- Flash attention kernel where Q input and output share the same buffer.
    Q is loaded into shared memory first, then output overwrites the buffer.
    Single read-write buffer avoids WebGPU aliasing. 1 dispatch only. -/
def flashAttentionInPlaceKernel (numHeads numKVHeads maxSeqLen headDim cacheLen : Nat)
    (scale : Float) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let wgid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let head := Exp.vec3X wgid
  let tid := Exp.vec3X lid

  let headsPerKV := numHeads / numKVHeads
  let kvHead := Exp.div head (Exp.litU32 headsPerKV)

  -- Single buffer: read Q first, then write output
  let _qOutput ← ShaderM.declareOutputBuffer "q_output" (.array (.scalar .f32) (numHeads * headDim))
  let _kCache ← ShaderM.declareInputBuffer "k_cache" (.array (.scalar .f32) (numKVHeads * maxSeqLen * headDim))
  let _vCache ← ShaderM.declareInputBuffer "v_cache" (.array (.scalar .f32) (numKVHeads * maxSeqLen * headDim))

  ShaderM.sharedNamed "shared_q" (.array (.scalar .f32) headDim)
  ShaderM.sharedNamed "shared_reduce" (.array (.scalar .f32) workgroupSize)

  -- Step 1: Load Q from the read-write buffer into shared memory
  let qBase := Exp.mul head (Exp.litU32 headDim)
  ShaderM.loop tid (Exp.litU32 headDim) (Exp.litU32 workgroupSize) fun d => do
    let qVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numHeads * headDim) "q_output" (Exp.add qBase d)
    ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_q" d qVal
  ShaderM.barrier

  -- Step 2: Online softmax (same as v1)
  ShaderM.varNamed "max_score" (.scalar .f32) (Exp.litF32 (-1.0e30))
  ShaderM.varNamed "sum_exp" (.scalar .f32) (Exp.litF32 0.0)
  ShaderM.varNamed "out_acc" (.scalar .f32) (Exp.litF32 0.0)
  let maxScore := Exp.var "max_score"
  let sumExp := Exp.var "sum_exp"
  let outAcc := Exp.var "out_acc"

  ShaderM.loop (Exp.litU32 0) (Exp.litU32 cacheLen) (Exp.litU32 1) fun s => do
    let kBase := Exp.add (Exp.mul (Exp.mul kvHead (Exp.litU32 maxSeqLen)) (Exp.litU32 headDim))
                          (Exp.mul s (Exp.litU32 headDim))

    let partialVar ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    ShaderM.loop tid (Exp.litU32 headDim) (Exp.litU32 workgroupSize) fun d => do
      let qVal ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := headDim) "shared_q" d
      let kVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numKVHeads * maxSeqLen * headDim) "k_cache" (Exp.add kBase d)
      ShaderM.assign partialVar (Exp.add (Exp.var partialVar) (Exp.mul qVal kVal))

    ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_reduce" tid (Exp.var partialVar)
    ShaderM.barrier

    let numSteps := Nat.log2 workgroupSize
    ShaderM.staticLoop numSteps fun step => do
      let stride := workgroupSize >>> (step + 1)
      ShaderM.if_ (Exp.lt tid (Exp.litU32 stride)) (do
        let other ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_reduce" (Exp.add tid (Exp.litU32 stride))
        let cur ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_reduce" tid
        ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_reduce" tid (Exp.add cur other)
      ) (pure ())
      ShaderM.barrier

    let scoreFromShared ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_reduce" (Exp.litU32 0)
    let scaledScore := Exp.mul (Exp.litF32 scale) scoreFromShared

    let oldMaxVar ← ShaderM.var (.scalar .f32) maxScore
    let oldSumVar ← ShaderM.var (.scalar .f32) sumExp
    let oldMax := Exp.var oldMaxVar
    let oldSum := Exp.var oldSumVar

    let newMax := Exp.max oldMax scaledScore
    let expOld := Exp.exp (Exp.sub oldMax newMax)
    let expNew := Exp.exp (Exp.sub scaledScore newMax)
    let newSum := Exp.add (Exp.mul oldSum expOld) expNew

    ShaderM.if_ (Exp.lt tid (Exp.litU32 headDim)) (do
      let vVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numKVHeads * maxSeqLen * headDim) "v_cache" (Exp.add kBase tid)
      let rescaled := Exp.mul outAcc (Exp.div (Exp.mul oldSum expOld) newSum)
      let newContrib := Exp.mul vVal (Exp.div expNew newSum)
      ShaderM.assign "out_acc" (Exp.add rescaled newContrib)
    ) (pure ())

    ShaderM.assign "max_score" newMax
    ShaderM.assign "sum_exp" newSum
    ShaderM.barrier

  -- Step 3: Write output (overwrites Q data in same buffer)
  ShaderM.if_ (Exp.lt tid (Exp.litU32 headDim)) (do
    let outIdx := Exp.add (Exp.mul head (Exp.litU32 headDim)) tid
    ShaderM.writeBuffer (ty := .scalar .f32) "q_output" outIdx outAcc
  ) (pure ())

/-! ## Tiled Flash Attention (v2) — High Parallelism -/

/-- Tiled flash attention: Phase 1 — each tile computes partial online softmax.
    Dispatch: (numHeads, numTiles). Each workgroup processes tileSize positions.
    Outputs per tile: partial_output[headDim], partial_max[1], partial_sumexp[1] -/
def flashAttentionTiledPhase1 (numHeads numKVHeads maxSeqLen headDim cacheLen tileSize : Nat)
    (scale : Float) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let wgid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let head := Exp.vec3X wgid       -- head index (wgid.x)
  let tileIdx := Exp.vec3Y wgid    -- tile index (wgid.y)
  let tid := Exp.vec3X lid

  let headsPerKV := numHeads / numKVHeads
  let kvHead := Exp.div head (Exp.litU32 headsPerKV)

  let numTiles := (cacheLen + tileSize - 1) / tileSize

  let _q ← ShaderM.declareInputBuffer "q" (.array (.scalar .f32) (numHeads * headDim))
  let _kCache ← ShaderM.declareInputBuffer "k_cache" (.array (.scalar .f32) (numKVHeads * maxSeqLen * headDim))
  let _vCache ← ShaderM.declareInputBuffer "v_cache" (.array (.scalar .f32) (numKVHeads * maxSeqLen * headDim))
  -- Partial results: [numHeads, numTiles, headDim + 2]  (output + max + sumexp)
  let partialSize := numHeads * numTiles * (headDim + 2)
  let _partial ← ShaderM.declareOutputBuffer "partial" (.array (.scalar .f32) partialSize)

  ShaderM.sharedNamed "shared_q" (.array (.scalar .f32) headDim)
  ShaderM.sharedNamed "shared_reduce" (.array (.scalar .f32) workgroupSize)

  -- Load Q into shared memory
  let qBase := Exp.mul head (Exp.litU32 headDim)
  ShaderM.loop tid (Exp.litU32 headDim) (Exp.litU32 workgroupSize) fun d => do
    let qVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numHeads * headDim) "q" (Exp.add qBase d)
    ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_q" d qVal
  ShaderM.barrier

  -- Online softmax for this tile's range
  let tileStart := Exp.mul tileIdx (Exp.litU32 tileSize)

  ShaderM.varNamed "max_score" (.scalar .f32) (Exp.litF32 (-1.0e30))
  ShaderM.varNamed "sum_exp" (.scalar .f32) (Exp.litF32 0.0)
  ShaderM.varNamed "out_acc" (.scalar .f32) (Exp.litF32 0.0)
  let maxScore := Exp.var "max_score"
  let sumExp := Exp.var "sum_exp"
  let outAcc := Exp.var "out_acc"

  -- Process positions in this tile
  ShaderM.loop (Exp.litU32 0) (Exp.litU32 tileSize) (Exp.litU32 1) fun localS => do
    let s := Exp.add tileStart localS

    -- Compute partial dot product
    let partialVar ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    -- Guard: only compute for valid positions
    ShaderM.if_ (Exp.lt s (Exp.litU32 cacheLen)) (do
      let kBase := Exp.add (Exp.mul (Exp.mul kvHead (Exp.litU32 maxSeqLen)) (Exp.litU32 headDim))
                            (Exp.mul s (Exp.litU32 headDim))
      ShaderM.loop tid (Exp.litU32 headDim) (Exp.litU32 workgroupSize) fun d => do
        let qVal ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := headDim) "shared_q" d
        let kVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numKVHeads * maxSeqLen * headDim) "k_cache" (Exp.add kBase d)
        ShaderM.assign partialVar (Exp.add (Exp.var partialVar) (Exp.mul qVal kVal))
    ) (pure ())

    -- Reduction (uniform control flow — loop bound is compile-time constant)
    ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_reduce" tid (Exp.var partialVar)
    ShaderM.barrier

    let numSteps := Nat.log2 workgroupSize
    ShaderM.staticLoop numSteps fun step => do
      let stride := workgroupSize >>> (step + 1)
      ShaderM.if_ (Exp.lt tid (Exp.litU32 stride)) (do
        let other ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_reduce" (Exp.add tid (Exp.litU32 stride))
        let cur ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_reduce" tid
        ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_reduce" tid (Exp.add cur other)
      ) (pure ())
      ShaderM.barrier

    -- Online softmax update (only for valid positions)
    ShaderM.if_ (Exp.lt s (Exp.litU32 cacheLen)) (do
      let scoreFromShared ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_reduce" (Exp.litU32 0)
      let scaledScore := Exp.mul (Exp.litF32 scale) scoreFromShared

      let oldMaxVar ← ShaderM.var (.scalar .f32) maxScore
      let oldSumVar ← ShaderM.var (.scalar .f32) sumExp
      let oldMax := Exp.var oldMaxVar
      let oldSum := Exp.var oldSumVar

      let newMax := Exp.max oldMax scaledScore
      let expOld := Exp.exp (Exp.sub oldMax newMax)
      let expNew := Exp.exp (Exp.sub scaledScore newMax)
      let newSum := Exp.add (Exp.mul oldSum expOld) expNew

      ShaderM.if_ (Exp.lt tid (Exp.litU32 headDim)) (do
        let kBase := Exp.add (Exp.mul (Exp.mul kvHead (Exp.litU32 maxSeqLen)) (Exp.litU32 headDim))
                              (Exp.mul s (Exp.litU32 headDim))
        let vVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numKVHeads * maxSeqLen * headDim) "v_cache" (Exp.add kBase tid)
        let rescaled := Exp.mul outAcc (Exp.div (Exp.mul oldSum expOld) newSum)
        let newContrib := Exp.mul vVal (Exp.div expNew newSum)
        ShaderM.assign "out_acc" (Exp.add rescaled newContrib)
      ) (pure ())

      ShaderM.assign "max_score" newMax
      ShaderM.assign "sum_exp" newSum
    ) (pure ())
    ShaderM.barrier

  -- Write partial results: [head, tileIdx, 0..headDim-1] = output, [.., headDim] = max, [.., headDim+1] = sumexp
  let stride := headDim + 2
  let partialBase := Exp.add (Exp.mul head (Exp.litU32 (numTiles * stride)))
                              (Exp.mul tileIdx (Exp.litU32 stride))
  ShaderM.if_ (Exp.lt tid (Exp.litU32 headDim)) (do
    ShaderM.writeBuffer (ty := .scalar .f32) "partial" (Exp.add partialBase tid) outAcc
  ) (pure ())
  ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
    ShaderM.writeBuffer (ty := .scalar .f32) "partial" (Exp.add partialBase (Exp.litU32 headDim)) maxScore
    ShaderM.writeBuffer (ty := .scalar .f32) "partial" (Exp.add partialBase (Exp.litU32 (headDim + 1))) sumExp
  ) (pure ())

/-- Tiled flash attention: Phase 2 — merge partial results.
    Each thread handles one output dimension for one head.
    Dispatch: (numHeads * headDim) -/
def flashAttentionTiledPhase2 (numHeads headDim numTiles : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid  -- linear index into [numHeads * headDim]

  let stride := headDim + 2
  let _partial ← ShaderM.declareInputBuffer "partial" (.array (.scalar .f32) (numHeads * numTiles * stride))
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) (numHeads * headDim))

  ShaderM.if_ (Exp.lt idx (Exp.litU32 (numHeads * headDim))) (do
    let head := Exp.div idx (Exp.litU32 headDim)
    let d := Exp.mod idx (Exp.litU32 headDim)

    -- Merge partial results using online softmax merge
    ShaderM.varNamed "merged_max" (.scalar .f32) (Exp.litF32 (-1.0e30))
    ShaderM.varNamed "merged_sum" (.scalar .f32) (Exp.litF32 0.0)
    ShaderM.varNamed "merged_out" (.scalar .f32) (Exp.litF32 0.0)
    let mergedMax := Exp.var "merged_max"
    let mergedSum := Exp.var "merged_sum"
    let mergedOut := Exp.var "merged_out"

    ShaderM.loop (Exp.litU32 0) (Exp.litU32 numTiles) (Exp.litU32 1) fun t => do
      let tBase := Exp.add (Exp.mul head (Exp.litU32 (numTiles * stride)))
                            (Exp.mul t (Exp.litU32 stride))
      let tileOut ← ShaderM.readBuffer (ty := .scalar .f32) (n := numHeads * numTiles * stride) "partial" (Exp.add tBase d)
      let tileMax ← ShaderM.readBuffer (ty := .scalar .f32) (n := numHeads * numTiles * stride) "partial" (Exp.add tBase (Exp.litU32 headDim))
      let tileSumExp ← ShaderM.readBuffer (ty := .scalar .f32) (n := numHeads * numTiles * stride) "partial" (Exp.add tBase (Exp.litU32 (headDim + 1)))

      -- Snapshot before update
      let oldMax ← ShaderM.var (.scalar .f32) mergedMax
      let oldSum ← ShaderM.var (.scalar .f32) mergedSum

      let newMax := Exp.max (Exp.var oldMax) tileMax
      let expOld := Exp.exp (Exp.sub (Exp.var oldMax) newMax)
      let expNew := Exp.exp (Exp.sub tileMax newMax)
      let newSum := Exp.add (Exp.mul (Exp.var oldSum) expOld) (Exp.mul tileSumExp expNew)

      -- Guard against division by zero (newSum could be 0 if all tiles empty)
      let safeSum := Exp.max newSum (Exp.litF32 1.0e-10)
      let rescaled := Exp.mul mergedOut (Exp.div (Exp.mul (Exp.var oldSum) expOld) safeSum)
      let newContrib := Exp.mul tileOut (Exp.div (Exp.mul tileSumExp expNew) safeSum)
      ShaderM.assign "merged_out" (Exp.add rescaled newContrib)
      ShaderM.assign "merged_max" newMax
      ShaderM.assign "merged_sum" newSum

    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx mergedOut
  ) (pure ())

/-- Pre-allocate partial buffer for tiled flash attention.
    Call once during initialization, reuse across all tokens. -/
def createFlashPartialBuffer (device : Device) (numHeads maxSeqLen headDim : Nat)
    (tileSize : Nat := 32) : IO Buffer := do
  let maxTiles := (maxSeqLen + tileSize - 1) / tileSize
  let stride := headDim + 2
  let partialSize := numHeads * maxTiles * stride
  createBuffer device {
    size := (partialSize * 4).toUSize
    usage := [.storage, .copySrc, .copyDst]
    mappedAtCreation := false
  }

/-- Execute tiled flash attention (2 phases) -/
def executeFlashAttentionTiled (device : Device)
    (qBuf kCacheBuf vCacheBuf outputBuf : Buffer)
    (numHeads numKVHeads maxSeqLen headDim cacheLen : Nat) (scale : Float)
    (partialBuf : Option Buffer := none) : IO Unit := do
  let tileSize := 32
  let numTiles := (cacheLen + tileSize - 1) / tileSize
  let workgroupSize := min 256 (max headDim 32)

  -- Use pre-allocated buffer or allocate (fallback for compatibility)
  let partialBuf ← match partialBuf with
    | some buf => pure buf
    | none => do
      let stride := headDim + 2
      let partialSize := numHeads * numTiles * stride
      createBuffer device {
        size := (partialSize * 4).toUSize
        usage := [.storage, .copySrc, .copyDst]
        mappedAtCreation := false
      }

  if numTiles == 1 then
    -- Single tile: use in-place v1 kernel (Q and output share same buffer)
    -- Q is loaded to shared memory first, then output overwrites the buffer.
    -- Uses declareOutputBuffer for q_output (read-write) to avoid aliasing.
    let shader := flashAttentionInPlaceKernel numHeads numKVHeads maxSeqLen headDim cacheLen scale workgroupSize
    let namedBuffers := [("q_output", outputBuf), ("k_cache", kCacheBuf), ("v_cache", vCacheBuf)]
    let cacheKey : UInt64 := hash ("flashIP", numHeads, numKVHeads, maxSeqLen, headDim, cacheLen)
    let execConfig : Hesper.WGSL.Execute.ExecutionConfig := {
      workgroupSize := {x := workgroupSize, y := 1, z := 1}
      numWorkgroups := (numHeads, 1, 1)
    }
    Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig (some cacheKey)
  else
    -- Multi-tile: Phase 1 (parallel tiles) + Phase 2 (merge)
    let shader1 := flashAttentionTiledPhase1 numHeads numKVHeads maxSeqLen headDim cacheLen tileSize scale workgroupSize
    let namedBuffers1 := [("q", qBuf), ("k_cache", kCacheBuf), ("v_cache", vCacheBuf), ("partial", partialBuf)]
    let execConfig1 : Hesper.WGSL.Execute.ExecutionConfig := {
      workgroupSize := {x := workgroupSize, y := 1, z := 1}
      numWorkgroups := (numHeads, numTiles, 1)
    }
    let cacheKey1 : UInt64 := hash ("flashT1", numHeads, numKVHeads, maxSeqLen, headDim, cacheLen, tileSize)
    Hesper.WGSL.Execute.executeShaderNamed device shader1 namedBuffers1 execConfig1 (some cacheKey1)

    let shader2 := flashAttentionTiledPhase2 numHeads headDim numTiles
    let namedBuffers2 := [("partial", partialBuf), ("output", outputBuf)]
    let execConfig2 := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D (numHeads * headDim) 256
    let cacheKey2 : UInt64 := hash ("flashT2", numHeads, headDim, numTiles)
    Hesper.WGSL.Execute.executeShaderNamed device shader2 namedBuffers2 execConfig2 (some cacheKey2)

def executeFlashAttentionDynamic (device : Device)
    (qBuf kCacheBuf vCacheBuf outputBuf : Buffer)
    (numHeads numKVHeads maxSeqLen headDim cacheLen : Nat) (scale : Float) : IO Unit := do
  let workgroupSize := min 256 (max headDim 32)
  let shader := flashAttentionDynamicKernel numHeads numKVHeads maxSeqLen headDim cacheLen scale workgroupSize
  let namedBuffers := [("q", qBuf), ("k_cache", kCacheBuf), ("v_cache", vCacheBuf), ("output", outputBuf)]
  -- Pipeline cache key includes cacheLen (shader recompiled per position)
  -- The WGSL source hash + buffer layout is cached, so same cacheLen reuses pipeline
  let cacheKey : UInt64 := hash ("flash", numHeads, numKVHeads, maxSeqLen, headDim, cacheLen)
  let execConfig : Hesper.WGSL.Execute.ExecutionConfig := {
    workgroupSize := {x := workgroupSize, y := 1, z := 1}
    numWorkgroups := (numHeads, 1, 1)
  }
  -- Note: shader compilation is cached by pipeline cache (Execute.lean).
  -- First call with a new cacheLen compiles, subsequent calls with same cacheLen reuse.
  -- Over a training run, common cacheLens are cached and recompilation is rare.
  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig (some cacheKey)

/-- Execute flash attention with static cacheLen (for testing).
    Uses dynamic kernel with a params buffer containing cacheLen. -/
def executeFlashAttention (device : Device)
    (qBuf kCacheBuf vCacheBuf outputBuf : Buffer)
    (numHeads numKVHeads cacheLen headDim : Nat) (scale : Float) : IO Unit := do
  -- For testing: maxSeqLen = cacheLen (buffer sizes match exactly)
  executeFlashAttentionDynamic device qBuf kCacheBuf vCacheBuf outputBuf
    numHeads numKVHeads cacheLen headDim cacheLen scale

/-! ## Sliding Window Flash Attention -/

/-- Flash attention with sliding window masking.
    Only attends to positions within [max(0, pos - windowSize + 1), pos].
    Uses compile-time cacheLen and windowSize for the loop bounds.

    For SWA layers in Gemma 4 ISWA architecture.
    Positions outside the window get -inf score (masked out by online softmax).

    @param numHeads Number of query heads
    @param numKVHeads Number of KV heads (GQA)
    @param maxSeqLen Maximum sequence length (cache buffer size)
    @param headDim Per-head dimension
    @param cacheLen Current number of cached positions
    @param windowSize Sliding window size (e.g., 512)
    @param currentPos Current query position
    @param scale Attention scale (1/sqrt(headDim))
-/
def flashAttentionSWAKernel (numHeads numKVHeads maxSeqLen headDim cacheLen windowSize currentPos : Nat)
    (scale : Float) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let wgid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let head := Exp.vec3X wgid
  let tid := Exp.vec3X lid

  let headsPerKV := numHeads / numKVHeads
  let kvHead := Exp.div head (Exp.litU32 headsPerKV)

  let _q ← ShaderM.declareInputBuffer "q" (.array (.scalar .f32) (numHeads * headDim))
  let _kCache ← ShaderM.declareInputBuffer "k_cache" (.array (.scalar .f32) (numKVHeads * maxSeqLen * headDim))
  let _vCache ← ShaderM.declareInputBuffer "v_cache" (.array (.scalar .f32) (numKVHeads * maxSeqLen * headDim))
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) (numHeads * headDim))

  ShaderM.sharedNamed "shared_q" (.array (.scalar .f32) headDim)
  ShaderM.sharedNamed "shared_reduce" (.array (.scalar .f32) workgroupSize)
  ShaderM.sharedNamed "shared_out" (.array (.scalar .f32) headDim)

  do
    -- Load Q into shared memory and zero shared_out
    let qBase := Exp.mul head (Exp.litU32 headDim)
    ShaderM.loop tid (Exp.litU32 headDim) (Exp.litU32 workgroupSize) fun d => do
      let qVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numHeads * headDim) "q" (Exp.add qBase d)
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_q" d qVal
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_out" d (Exp.litF32 0.0)
    ShaderM.barrier

    -- Online softmax state
    ShaderM.varNamed "max_score" (.scalar .f32) (Exp.litF32 (-1.0e30))
    ShaderM.varNamed "sum_exp" (.scalar .f32) (Exp.litF32 0.0)
    let maxScore := Exp.var "max_score"
    let sumExp := Exp.var "sum_exp"

    -- Sliding window: only attend to [windowStart, cacheLen)
    let windowStart := if currentPos + 1 > windowSize then currentPos + 1 - windowSize else 0

    ShaderM.loop (Exp.litU32 windowStart) (Exp.litU32 cacheLen) (Exp.litU32 1) fun s => do
      let kBase := Exp.add (Exp.mul (Exp.mul kvHead (Exp.litU32 maxSeqLen)) (Exp.litU32 headDim))
                            (Exp.mul s (Exp.litU32 headDim))

      -- Score computation (same as standard flash attention)
      let partialVar ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
      ShaderM.loop tid (Exp.litU32 headDim) (Exp.litU32 workgroupSize) fun d => do
        let qVal ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := headDim) "shared_q" d
        let kVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numKVHeads * maxSeqLen * headDim) "k_cache" (Exp.add kBase d)
        ShaderM.assign partialVar (Exp.add (Exp.var partialVar) (Exp.mul qVal kVal))

      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_reduce" tid (Exp.var partialVar)
      ShaderM.barrier

      let numSteps := Nat.log2 workgroupSize
      ShaderM.staticLoop numSteps fun step => do
        let stride := workgroupSize >>> (step + 1)
        ShaderM.if_ (Exp.lt tid (Exp.litU32 stride)) (do
          let other ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_reduce" (Exp.add tid (Exp.litU32 stride))
          let cur ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_reduce" tid
          ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_reduce" tid (Exp.add cur other)
        ) (pure ())
        ShaderM.barrier

      let scoreFromShared ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_reduce" (Exp.litU32 0)
      let scaledScore := Exp.mul (Exp.litF32 scale) scoreFromShared

      -- Online softmax update
      let oldMaxVar ← ShaderM.var (.scalar .f32) maxScore
      let oldSumVar ← ShaderM.var (.scalar .f32) sumExp
      let oldMax := Exp.var oldMaxVar
      let oldSum := Exp.var oldSumVar

      let newMax := Exp.max oldMax scaledScore
      let expOld := Exp.exp (Exp.sub oldMax newMax)
      let expNew := Exp.exp (Exp.sub scaledScore newMax)
      let newSum := Exp.add (Exp.mul oldSum expOld) expNew

      let rescaleFactor := Exp.div (Exp.mul oldSum expOld) newSum
      let contribFactor := Exp.div expNew newSum

      ShaderM.loop tid (Exp.litU32 headDim) (Exp.litU32 workgroupSize) fun d => do
        let vVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numKVHeads * maxSeqLen * headDim) "v_cache" (Exp.add kBase d)
        let prev ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := headDim) "shared_out" d
        let updated := Exp.add (Exp.mul prev rescaleFactor) (Exp.mul vVal contribFactor)
        ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_out" d updated

      ShaderM.assign "max_score" newMax
      ShaderM.assign "sum_exp" newSum
      ShaderM.barrier

    -- Write output (strided loop covers all headDim elements)
    ShaderM.loop tid (Exp.litU32 headDim) (Exp.litU32 workgroupSize) fun d => do
      let v ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := headDim) "shared_out" d
      let outIdx := Exp.add (Exp.mul head (Exp.litU32 headDim)) d
      ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx v

end Hesper.WGSL.FlashAttention
