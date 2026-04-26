import Hesper.WGSL.Monad
import Hesper.WGSL.Execute
import Hesper.WGSL.Exp
import Hesper.Backend
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
open Hesper

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

/-- Like flashAttentionDynamicKernel but reads cacheLen from params buffer.
    PTX is fixed regardless of cacheLen → fully cacheable.
    Supports headDim > workgroupSize via strided loops (shared_out). -/
def flashAttentionDynamicParamsKernel (numHeads numKVHeads maxSeqLen headDim : Nat)
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
  let _params ← ShaderM.declareStorageBuffer "params" (.array (.scalar .u32) 2) .read

  ShaderM.sharedNamed "shared_q" (.array (.scalar .f32) headDim)
  ShaderM.sharedNamed "shared_reduce" (.array (.scalar .f32) workgroupSize)
  ShaderM.sharedNamed "shared_out" (.array (.scalar .f32) headDim)

  do
    let qBase := Exp.mul head (Exp.litU32 headDim)
    ShaderM.loop tid (Exp.litU32 headDim) (Exp.litU32 workgroupSize) fun d => do
      let qVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numHeads * headDim) "q" (Exp.add qBase d)
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_q" d qVal
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_out" d (Exp.litF32 0.0)
    ShaderM.barrier

    ShaderM.varNamed "max_score" (.scalar .f32) (Exp.litF32 (-1.0e30))
    ShaderM.varNamed "sum_exp" (.scalar .f32) (Exp.litF32 0.0)
    let maxScore := Exp.var "max_score"
    let sumExp := Exp.var "sum_exp"

    -- Read cacheLen from params buffer (runtime, not compile-time)
    let cacheLen ← ShaderM.readBuffer (ty := .scalar .u32) (n := 2) "params" (Exp.litU32 1)

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

    ShaderM.loop tid (Exp.litU32 headDim) (Exp.litU32 workgroupSize) fun d => do
      let v ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := headDim) "shared_out" d
      let outIdx := Exp.add (Exp.mul head (Exp.litU32 headDim)) d
      ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx v

/-- doc 60 Session 1: warp-shuffle vec kernel.

    Like flashAttentionDynamicParamsKernel but reduces the q·K[s] dot
    product via `subgroupAdd` (warp shuffle) instead of a shared-memory
    tree reduce.  workgroupSize is fixed at 128 (4 warps × 32 lanes).

    ## why this is faster

    The legacy 256-thread tree reduce inside the cacheLen-step loop has
    log2(256)=8 barriers per K position.  For Gemma 4 (head_dim=256,
    cacheLen up to ~200) that is **8 × cacheLen ≈ 1600 barriers per head
    per token**.  Replacing the tree with subgroupAdd cuts each K-step
    reduce to:

      warp-shuffle sum    : 5 shfl.bfly, 0 barriers
      lane-0 → smem write : 1 barrier
      warp-0 cross-warp   : 1 subgroupAdd over numWarps=4 lanes (cheap)
      barrier             : 1
      ──────────────────────
      per-K total         : 2 barriers (vs the old 8)

    so we expect ~4× fewer barriers in the inner loop, plus subgroupAdd
    is a register-level shuffle that does not touch shared memory.

    ## algorithm (unchanged from the legacy dynamic kernel)

    Same online softmax over a serial K loop; workgroups are still per
    head (gridX = numHeads, gridY = 1).  This is the *conservative*
    Option B (doc 60): we keep cacheLen-serial, just switch the reduce
    primitive.  A future revision can re-shape to llama.cpp's K-parallel
    layout (multiple warps each processing a distinct K position) for
    the remaining gap. -/
def flashAttentionVecParamsKernel (numHeads numKVHeads maxSeqLen headDim : Nat)
    (scale : Float) : ShaderM Unit := do
  let workgroupSize : Nat := 128
  let numWarps : Nat := workgroupSize / 32

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
  let _params ← ShaderM.declareStorageBuffer "params" (.array (.scalar .u32) 2) .read

  ShaderM.sharedNamed "shared_q" (.array (.scalar .f32) headDim)
  -- One slot per warp for warpBlock-style sum gather.  Sized to numWarps
  -- (4) but rounded up to 32 so warp 0's cross-warp subgroupAdd has a
  -- fixed-shape array to read.
  ShaderM.sharedNamed "shared_warp_sums" (.array (.scalar .f32) 32)
  -- Online softmax accumulators live in shared memory because the
  -- per-K rescale must be visible to every thread that contributes to
  -- the per-dim out_acc array (the 256 threads each own a portion of
  -- shared_out below).
  ShaderM.sharedNamed "shared_out" (.array (.scalar .f32) headDim)

  -- Staged Q load
  let qBase := Exp.mul head (Exp.litU32 headDim)
  ShaderM.loop tid (Exp.litU32 headDim) (Exp.litU32 workgroupSize) fun d => do
    let qVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numHeads * headDim) "q" (Exp.add qBase d)
    ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_q" d qVal
    ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_out" d (Exp.litF32 0.0)
  ShaderM.barrier

  ShaderM.varNamed "max_score" (.scalar .f32) (Exp.litF32 (-1.0e30))
  ShaderM.varNamed "sum_exp" (.scalar .f32) (Exp.litF32 0.0)
  let maxScore := Exp.var "max_score"
  let sumExp := Exp.var "sum_exp"

  let cacheLen ← ShaderM.readBuffer (ty := .scalar .u32) (n := 2) "params" (Exp.litU32 1)

  let laneId := Exp.bitAnd tid (Exp.litU32 31)
  let warpId := Exp.shiftRight tid (Exp.litU32 5)

  ShaderM.loop (Exp.litU32 0) cacheLen (Exp.litU32 1) fun s => do
    let kBase := Exp.add (Exp.mul (Exp.mul kvHead (Exp.litU32 maxSeqLen)) (Exp.litU32 headDim))
                          (Exp.mul s (Exp.litU32 headDim))

    -- Each thread accumulates the partial dot product across its slice
    -- of the head dimension.  For workgroupSize=128 and headDim=256 each
    -- thread covers 2 dims.
    let partialVar ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    ShaderM.loop tid (Exp.litU32 headDim) (Exp.litU32 workgroupSize) fun d => do
      let qVal ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := headDim) "shared_q" d
      let kVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numKVHeads * maxSeqLen * headDim) "k_cache" (Exp.add kBase d)
      ShaderM.assign partialVar (Exp.add (Exp.var partialVar) (Exp.mul qVal kVal))

    -- Warp-level reduce (5 shfl.bfly, 0 barriers).  Materialise via
    -- ShaderM.var so the shfls are emitted before the subsequent
    -- predicated smem write — same trick as warpBlockSumReduce.
    let warpSumName ← ShaderM.var (.scalar .f32) (Exp.subgroupAdd (Exp.var partialVar))
    let warpSum : Exp (.scalar .f32) := Exp.var warpSumName

    -- Lane 0 of each warp publishes its warp's sum.
    ShaderM.if_ (Exp.eq laneId (Exp.litU32 0)) (do
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_warp_sums" warpId warpSum
    ) (pure ())
    ShaderM.barrier

    -- Warp-0 lane-i reads slot i.  Lanes >= numWarps read 0 so the
    -- subsequent subgroupAdd doesn't pick up uninitialised values.
    -- We let *all* 32 lanes in warp 0 participate (mask = 0xFFFFFFFF)
    -- to keep shfl.sync from deadlocking.
    let slotVal ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 32) "shared_warp_sums" laneId
    let guardedName ← ShaderM.var (.scalar .f32)
      (Exp.select (Exp.lt laneId (Exp.litU32 numWarps)) slotVal (Exp.litF32 0.0))
    let guarded : Exp (.scalar .f32) := Exp.var guardedName
    let totalWarpName ← ShaderM.var (.scalar .f32) (Exp.subgroupAdd guarded)
    let totalScoreLane0 : Exp (.scalar .f32) := Exp.var totalWarpName
    -- Broadcast lane 0's value into a single shared slot so every thread
    -- can read the same final score without needing per-thread shfls
    -- across warp boundaries.
    ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_warp_sums" (Exp.litU32 0) totalScoreLane0
    ) (pure ())
    ShaderM.barrier

    let scoreFromShared ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 32) "shared_warp_sums" (Exp.litU32 0)
    let scaledScore := Exp.mul (Exp.litF32 scale) scoreFromShared

    -- Capture the old max/sum *before* `assign` overwrites them.  Using
    -- `Exp.var` directly would re-read the shared name on each use and
    -- pick up the partially-updated value mid-iteration (matches the
    -- existing flashAttentionDynamicParamsKernel pattern).
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

    -- All threads scan their slice of headDim and update shared_out.
    ShaderM.loop tid (Exp.litU32 headDim) (Exp.litU32 workgroupSize) fun d => do
      let vVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numKVHeads * maxSeqLen * headDim) "v_cache" (Exp.add kBase d)
      let prev ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := headDim) "shared_out" d
      let updated := Exp.add (Exp.mul prev rescaleFactor) (Exp.mul vVal contribFactor)
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_out" d updated

    ShaderM.assign "max_score" newMax
    ShaderM.assign "sum_exp" newSum
    ShaderM.barrier

  ShaderM.loop tid (Exp.litU32 headDim) (Exp.litU32 workgroupSize) fun d => do
    let v ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := headDim) "shared_out" d
    let outIdx := Exp.add (Exp.mul head (Exp.litU32 headDim)) d
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx v

/-- Vec-params V2 (Session 2′, doc 60): keeps Q + VKQ accumulator in
    per-thread registers (PTX local vars) instead of round-tripping
    through shared memory.  Targets the 30× shared-memory traffic gap
    identified by ncu vs llama.cpp.

    Layout:
      block_dim = (128, 1, 1) = 4 warps
      grid      = (numHeads, 1, 1)
      headDim must be a multiple of 128 (=workgroupSize).  For Gemma 4
      D=256, each thread owns elemsPerThread=2 contiguous dims.

    Differences vs V1:
      - Q is read once per thread into per-thread registers (no shared_q).
      - VKQ accumulator is per-thread registers (no shared_out).
      - Cross-warp KQ reduce: 1 smem write per warp + 1 broadcast slot
        + 1 barrier — total ≤ 8 smem ops/iter (vs V1's 256+).
      - V is read directly from gmem and applied to per-thread VKQ
        without an intermediate smem stage.
    Pre-condition: headDim % workgroupSize == 0. -/
def flashAttentionVecParamsKernelV2
    (numHeads numKVHeads maxSeqLen headDim : Nat) (scale : Float) :
    ShaderM Unit := do
  let workgroupSize : Nat := 128
  let _numWarps : Nat := workgroupSize / 32
  let elemsPerThread : Nat := headDim / workgroupSize
  -- contiguous-per-thread layout: thread t owns dims [t*epT .. (t+1)*epT)

  let wgid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let head := Exp.vec3X wgid
  let tid := Exp.vec3X lid
  let headsPerKV := numHeads / numKVHeads
  let kvHead := Exp.div head (Exp.litU32 headsPerKV)

  let _q ← ShaderM.declareInputBuffer "q" (.array (.scalar .f32) (numHeads * headDim))
  let _kCache ← ShaderM.declareInputBuffer "k_cache"
                  (.array (.scalar .f32) (numKVHeads * maxSeqLen * headDim))
  let _vCache ← ShaderM.declareInputBuffer "v_cache"
                  (.array (.scalar .f32) (numKVHeads * maxSeqLen * headDim))
  let _output ← ShaderM.declareOutputBuffer "output"
                  (.array (.scalar .f32) (numHeads * headDim))
  let _params ← ShaderM.declareStorageBuffer "params" (.array (.scalar .u32) 2) .read

  -- Only smem: 32 slots for cross-warp reduce.  Sized to 32 so warp 0's
  -- subgroupAdd-over-lane has a fixed-shape array.
  ShaderM.sharedNamed "shared_warp_sums" (.array (.scalar .f32) 32)

  -- Per-thread Q (preloaded once) and VKQ accumulator (updated per K).
  -- Lean static loop unrolls these into elemsPerThread independent vars.
  let qBase := Exp.mul head (Exp.litU32 headDim)
  let dBase := Exp.mul tid (Exp.litU32 elemsPerThread)

  let mut qVars : Array String := #[]
  let mut vkqVars : Array String := #[]
  for ep in [0:elemsPerThread] do
    let d := Exp.add dBase (Exp.litU32 ep)
    let qVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numHeads * headDim)
                 "q" (Exp.add qBase d)
    let qName ← ShaderM.var (.scalar .f32) qVal
    let vkqName ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    qVars := qVars.push qName
    vkqVars := vkqVars.push vkqName

  ShaderM.varNamed "max_score" (.scalar .f32) (Exp.litF32 (-1.0e30))
  ShaderM.varNamed "sum_exp" (.scalar .f32) (Exp.litF32 0.0)
  let maxScore := Exp.var "max_score"
  let sumExp := Exp.var "sum_exp"

  let cacheLen ← ShaderM.readBuffer (ty := .scalar .u32) (n := 2)
                   "params" (Exp.litU32 1)
  let laneId := Exp.bitAnd tid (Exp.litU32 31)
  let warpId := Exp.shiftRight tid (Exp.litU32 5)

  ShaderM.loop (Exp.litU32 0) cacheLen (Exp.litU32 1) fun s => do
    -- 1) Per-thread partial dot product over the elemsPerThread dims.
    let partialVar ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    let kBase := Exp.add (Exp.mul (Exp.mul kvHead (Exp.litU32 maxSeqLen))
                                  (Exp.litU32 headDim))
                          (Exp.mul s (Exp.litU32 headDim))
    for ep in [0:elemsPerThread] do
      let d := Exp.add dBase (Exp.litU32 ep)
      let qExp : Exp (.scalar .f32) := Exp.var qVars[ep]!
      let kVal ← ShaderM.readBuffer (ty := .scalar .f32)
                   (n := numKVHeads * maxSeqLen * headDim) "k_cache"
                   (Exp.add kBase d)
      ShaderM.assign partialVar
        (Exp.add (Exp.var partialVar) (Exp.mul qExp kVal))

    -- 2) Warp reduce (5 shfl, no barrier).
    let warpSumName ← ShaderM.var (.scalar .f32)
                        (Exp.subgroupAdd (Exp.var partialVar))
    let warpSum : Exp (.scalar .f32) := Exp.var warpSumName

    -- 3) Cross-warp reduce: 4 smem writes (lane 0 of each warp), 1 barrier,
    -- then warp 0 does subgroupAdd over the 4 slots and broadcasts via
    -- slot 0.  Total smem ops: 4 writes + 32 reads + 1 broadcast write
    -- + 128 reads of slot 0 = ~165 ops/iter (vs V1's >256 per smem array).
    ShaderM.if_ (Exp.eq laneId (Exp.litU32 0)) (do
      ShaderM.writeWorkgroup (ty := .scalar .f32)
        "shared_warp_sums" warpId warpSum
    ) (pure ())
    ShaderM.barrier

    -- Warp 0 collapses the per-warp slots, lane 0 broadcasts back.
    let slotVal ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 32)
                    "shared_warp_sums" laneId
    let guardedName ← ShaderM.var (.scalar .f32)
      (Exp.select (Exp.lt laneId (Exp.litU32 _numWarps))
                  slotVal (Exp.litF32 0.0))
    let totalName ← ShaderM.var (.scalar .f32)
                      (Exp.subgroupAdd (Exp.var guardedName))
    -- Lane 0 of warp 0 broadcasts the total back via slot 0.
    ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
      ShaderM.writeWorkgroup (ty := .scalar .f32)
        "shared_warp_sums" (Exp.litU32 0) (Exp.var totalName)
    ) (pure ())
    ShaderM.barrier

    let scoreFromShared ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 32)
                            "shared_warp_sums" (Exp.litU32 0)
    let scaledScore := Exp.mul (Exp.litF32 scale) scoreFromShared

    -- 4) Online softmax update (per-thread, no smem).
    let oldMaxVar ← ShaderM.var (.scalar .f32) maxScore
    let oldSumVar ← ShaderM.var (.scalar .f32) sumExp
    let oldMax := Exp.var oldMaxVar
    let oldSum := Exp.var oldSumVar
    let newMax := Exp.max oldMax scaledScore
    let expOld := Exp.exp (Exp.sub oldMax newMax)
    let expNew := Exp.exp (Exp.sub scaledScore newMax)
    let newSum := Exp.add (Exp.mul oldSum expOld) expNew
    let rescaleVar ← ShaderM.var (.scalar .f32)
                       (Exp.div (Exp.mul oldSum expOld) newSum)
    let contribVar ← ShaderM.var (.scalar .f32)
                       (Exp.div expNew newSum)

    -- 5) Update per-thread VKQ accumulator from gmem V.  No smem.
    for ep in [0:elemsPerThread] do
      let d := Exp.add dBase (Exp.litU32 ep)
      let vVal ← ShaderM.readBuffer (ty := .scalar .f32)
                   (n := numKVHeads * maxSeqLen * headDim) "v_cache"
                   (Exp.add kBase d)
      let prevExp : Exp (.scalar .f32) := Exp.var vkqVars[ep]!
      ShaderM.assign vkqVars[ep]!
        (Exp.add (Exp.mul prevExp (Exp.var rescaleVar))
                 (Exp.mul vVal   (Exp.var contribVar)))

    ShaderM.assign "max_score" newMax
    ShaderM.assign "sum_exp" newSum

  -- Write per-thread VKQ to global output.  No final smem read.
  for ep in [0:elemsPerThread] do
    let d := Exp.add dBase (Exp.litU32 ep)
    let outIdx := Exp.add qBase d
    ShaderM.writeBuffer (ty := .scalar .f32)
      "output" outIdx (Exp.var vkqVars[ep]!)

/-- Vec-params V3 (Session 3, doc 60): V2 + K cache stored as f16.
    V cache stays f32 (Session 4 will switch).

    K buffer is declared as `array<u32, N/2>` where each u32 holds 2
    packed f16 values (low half = even index, high half = odd index).
    Layout: K[kvHead × maxSeqLen × headDim + s × headDim + d] is the
    f16 value at the same logical (kvHead, s, d) coordinate as the V2
    f32 layout.  Strides match: f16 row stride = headDim × 2 bytes.

    Pre-condition: headDim must be even (Gemma 4 D=256 ✓) and the
    per-thread elemsPerThread must be even (epT=2 for ws=128 ✓), so
    each thread reads exactly 1 u32 = 2 f16 = 2 dims per K iter. -/
def flashAttentionVecParamsKernelV3
    (numHeads numKVHeads maxSeqLen headDim : Nat) (scale : Float) :
    ShaderM Unit := do
  let workgroupSize : Nat := 128
  let _numWarps : Nat := workgroupSize / 32
  let elemsPerThread : Nat := headDim / workgroupSize

  let wgid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let head := Exp.vec3X wgid
  let tid := Exp.vec3X lid
  let headsPerKV := numHeads / numKVHeads
  let kvHead := Exp.div head (Exp.litU32 headsPerKV)

  let _q ← ShaderM.declareInputBuffer "q" (.array (.scalar .f32) (numHeads * headDim))
  -- K is f16: total f16 elements = numKVHeads × maxSeqLen × headDim,
  -- packed as half that many u32 words.
  let kF16Words : Nat := (numKVHeads * maxSeqLen * headDim) / 2
  let _kCache ← ShaderM.declareInputBuffer "k_cache_f16"
                  (.array (.scalar .u32) kF16Words)
  let _vCache ← ShaderM.declareInputBuffer "v_cache"
                  (.array (.scalar .f32) (numKVHeads * maxSeqLen * headDim))
  let _output ← ShaderM.declareOutputBuffer "output"
                  (.array (.scalar .f32) (numHeads * headDim))
  let _params ← ShaderM.declareStorageBuffer "params" (.array (.scalar .u32) 2) .read

  ShaderM.sharedNamed "shared_warp_sums" (.array (.scalar .f32) 32)

  let qBase := Exp.mul head (Exp.litU32 headDim)
  let dBase := Exp.mul tid (Exp.litU32 elemsPerThread)

  let mut qVars : Array String := #[]
  let mut vkqVars : Array String := #[]
  for ep in [0:elemsPerThread] do
    let d := Exp.add dBase (Exp.litU32 ep)
    let qVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numHeads * headDim)
                 "q" (Exp.add qBase d)
    let qName ← ShaderM.var (.scalar .f32) qVal
    let vkqName ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    qVars := qVars.push qName
    vkqVars := vkqVars.push vkqName

  ShaderM.varNamed "max_score" (.scalar .f32) (Exp.litF32 (-1.0e30))
  ShaderM.varNamed "sum_exp" (.scalar .f32) (Exp.litF32 0.0)
  let maxScore := Exp.var "max_score"
  let sumExp := Exp.var "sum_exp"

  let cacheLen ← ShaderM.readBuffer (ty := .scalar .u32) (n := 2)
                   "params" (Exp.litU32 1)
  let laneId := Exp.bitAnd tid (Exp.litU32 31)
  let warpId := Exp.shiftRight tid (Exp.litU32 5)

  -- Pre-compute the per-thread *u32 word* offset within a K row.  Each
  -- thread owns elemsPerThread=2 dims, packed in 1 u32.  Word index is
  -- floor(d / 2).  Since dBase = tid * 2, dBase/2 = tid, so the word
  -- offset within a K row is just `tid`.
  let dBaseWordU32 := tid
  -- K row stride in u32 words: headDim / 2.
  let kRowStrideU32 := Exp.litU32 (headDim / 2)
  -- K base for this kvHead row: kvHead * maxSeqLen * (headDim/2) words
  let kHeadBaseU32 := Exp.mul kvHead
                        (Exp.mul (Exp.litU32 maxSeqLen) kRowStrideU32)

  ShaderM.loop (Exp.litU32 0) cacheLen (Exp.litU32 1) fun s => do
    -- 1) Per-thread partial dot product: 1 u32 read = 2 f16 = 2 FMAs.
    let partialVar ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    let kRowBaseU32 := Exp.add kHeadBaseU32 (Exp.mul s kRowStrideU32)
    let kWordIdx := Exp.add kRowBaseU32 dBaseWordU32
    let packedName ← ShaderM.var (.scalar .u32)
                       (← ShaderM.readBuffer (ty := .scalar .u32)
                            (n := kF16Words) "k_cache_f16" kWordIdx)
    -- unpack2x16float gives a vec2<f32>: (low f16 → x, high f16 → y).
    -- Layout: word holds f16[d_even] in low bits, f16[d_odd] in high bits.
    let unpacked := Exp.unpack2x16float (Exp.var packedName)
    let kVal0Var ← ShaderM.var (.scalar .f32) (Exp.vecX unpacked)
    let kVal1Var ← ShaderM.var (.scalar .f32) (Exp.vecY unpacked)
    let q0 : Exp (.scalar .f32) := Exp.var qVars[0]!
    let q1 : Exp (.scalar .f32) := Exp.var qVars[1]!
    -- Two FMAs per thread per K iter.
    ShaderM.assign partialVar
      (Exp.add (Exp.var partialVar) (Exp.mul q0 (Exp.var kVal0Var)))
    ShaderM.assign partialVar
      (Exp.add (Exp.var partialVar) (Exp.mul q1 (Exp.var kVal1Var)))

    -- 2) Warp reduce.
    let warpSumName ← ShaderM.var (.scalar .f32)
                        (Exp.subgroupAdd (Exp.var partialVar))
    let warpSum : Exp (.scalar .f32) := Exp.var warpSumName

    -- 3) Cross-warp reduce.
    ShaderM.if_ (Exp.eq laneId (Exp.litU32 0)) (do
      ShaderM.writeWorkgroup (ty := .scalar .f32)
        "shared_warp_sums" warpId warpSum
    ) (pure ())
    ShaderM.barrier

    let slotVal ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 32)
                    "shared_warp_sums" laneId
    let guardedName ← ShaderM.var (.scalar .f32)
      (Exp.select (Exp.lt laneId (Exp.litU32 _numWarps))
                  slotVal (Exp.litF32 0.0))
    let totalName ← ShaderM.var (.scalar .f32)
                      (Exp.subgroupAdd (Exp.var guardedName))
    ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
      ShaderM.writeWorkgroup (ty := .scalar .f32)
        "shared_warp_sums" (Exp.litU32 0) (Exp.var totalName)
    ) (pure ())
    ShaderM.barrier

    let scoreFromShared ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 32)
                            "shared_warp_sums" (Exp.litU32 0)
    let scaledScore := Exp.mul (Exp.litF32 scale) scoreFromShared

    -- 4) Online softmax update.
    let oldMaxVar ← ShaderM.var (.scalar .f32) maxScore
    let oldSumVar ← ShaderM.var (.scalar .f32) sumExp
    let oldMax := Exp.var oldMaxVar
    let oldSum := Exp.var oldSumVar
    let newMax := Exp.max oldMax scaledScore
    let expOld := Exp.exp (Exp.sub oldMax newMax)
    let expNew := Exp.exp (Exp.sub scaledScore newMax)
    let newSum := Exp.add (Exp.mul oldSum expOld) expNew
    let rescaleVar ← ShaderM.var (.scalar .f32)
                       (Exp.div (Exp.mul oldSum expOld) newSum)
    let contribVar ← ShaderM.var (.scalar .f32)
                       (Exp.div expNew newSum)

    -- 5) Update VKQ from f32 V cache (Session 4 will switch this to f16 too).
    let vRowBase := Exp.add (Exp.mul (Exp.mul kvHead (Exp.litU32 maxSeqLen))
                                     (Exp.litU32 headDim))
                            (Exp.mul s (Exp.litU32 headDim))
    for ep in [0:elemsPerThread] do
      let d := Exp.add dBase (Exp.litU32 ep)
      let vVal ← ShaderM.readBuffer (ty := .scalar .f32)
                   (n := numKVHeads * maxSeqLen * headDim) "v_cache"
                   (Exp.add vRowBase d)
      let prevExp : Exp (.scalar .f32) := Exp.var vkqVars[ep]!
      ShaderM.assign vkqVars[ep]!
        (Exp.add (Exp.mul prevExp (Exp.var rescaleVar))
                 (Exp.mul vVal   (Exp.var contribVar)))

    ShaderM.assign "max_score" newMax
    ShaderM.assign "sum_exp" newSum

  for ep in [0:elemsPerThread] do
    let d := Exp.add dBase (Exp.litU32 ep)
    let outIdx := Exp.add qBase d
    ShaderM.writeBuffer (ty := .scalar .f32)
      "output" outIdx (Exp.var vkqVars[ep]!)

/-- Vec-params V6 (doc 60): K-parallel inner loop matching llama.cpp.

    Thread axis assignment (the critical change vs V2):
      - laneId (0..31) = K-position offset within warp's K-tile
      - warpId (0..3)  = K-tile index within nthreads-wide K-batch
      - inside each KQ dot: laneId also re-used as D-axis slice

    Loop structure mirrors fattn-vec.cuh:256-279 exactly:
      for k_VKQ_0 in [0, cacheLen) step nthreads (=128):
        for i_KQ_0 in 0..32:                  // 32 K positions per warp
          i_KQ = warpId * 32 + i_KQ_0
          k_pos = k_VKQ_0 + i_KQ
          sum = my-lane's slice of (Q · K[k_pos])
          sum = warp_reduce(sum)              // sums across D-axis
          if laneId == i_KQ_0: KQ_reg = sum   // store on owning lane

    Per-thread state (registers):
      Q_reg[D/32]   -- 8 elements when D=256, lane owns d = k*32 + laneId
      VKQ[D/32]     -- 8 output dims, same lane mapping as Q
      KQ_max, KQ_sum  -- scalar
      KQ_reg          -- per-K KQ score on owning lane

    For Q=K dot product, each lane covers D/32 dims; warp_reduce collapses
    those to a single scalar = full Q·K_pos.  This is the lever that turns
    cacheLen=128 into "outer loop runs 1 time" instead of "runs 128 times".

    Pre-condition: D % 32 == 0 (Gemma 4 D=256 ✓), workgroupSize=128. -/
def flashAttentionVecParamsKernelV6
    (numHeads numKVHeads maxSeqLen headDim : Nat) (scale : Float) :
    ShaderM Unit := do
  let workgroupSize : Nat := 128
  let numWarps : Nat := workgroupSize / 32
  let dPerLane : Nat := headDim / 32  -- 8 for D=256

  let wgid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let head := Exp.vec3X wgid
  let tid := Exp.vec3X lid
  let headsPerKV := numHeads / numKVHeads
  let kvHead := Exp.div head (Exp.litU32 headsPerKV)

  let _q ← ShaderM.declareInputBuffer "q" (.array (.scalar .f32) (numHeads * headDim))
  let _kCache ← ShaderM.declareInputBuffer "k_cache"
                  (.array (.scalar .f32) (numKVHeads * maxSeqLen * headDim))
  let _vCache ← ShaderM.declareInputBuffer "v_cache"
                  (.array (.scalar .f32) (numKVHeads * maxSeqLen * headDim))
  let _output ← ShaderM.declareOutputBuffer "output"
                  (.array (.scalar .f32) (numHeads * headDim))
  let _params ← ShaderM.declareStorageBuffer "params" (.array (.scalar .u32) 2) .read

  -- Smem KQ tile: per-warp 32 KQ scores, used to broadcast scores to all
  -- lanes during VKQ accumulation.  4 warps × 32 = 128 entries.
  ShaderM.sharedNamed "shared_kq" (.array (.scalar .f32) workgroupSize)
  -- Cross-warp VKQ reduce uses 4 warps × headDim entries (D=256 → 1024).
  ShaderM.sharedNamed "shared_vkq" (.array (.scalar .f32) (numWarps * headDim))
  -- Per-warp (kq_max, kq_sum) for cross-warp softmax merge.  Slot 2*w+0
  -- holds warp w's max, slot 2*w+1 holds warp w's sum.  numWarps × 2 = 8.
  ShaderM.sharedNamed "shared_warp_meta" (.array (.scalar .f32) (numWarps * 2))

  let laneId := Exp.bitAnd tid (Exp.litU32 31)
  let warpId := Exp.shiftRight tid (Exp.litU32 5)

  -- Pre-load Q (already scaled) into per-lane registers.  Lane owns dims
  -- d = k * 32 + laneId for k = 0..dPerLane.
  let qBase := Exp.mul head (Exp.litU32 headDim)
  let mut qVars : Array String := #[]
  let mut vkqVars : Array String := #[]
  for k in [0:dPerLane] do
    let d := Exp.add (Exp.litU32 (k * 32)) laneId
    let qVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numHeads * headDim)
                 "q" (Exp.add qBase d)
    let qScaled := Exp.mul qVal (Exp.litF32 scale)
    let qName ← ShaderM.var (.scalar .f32) qScaled
    let vkqName ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    qVars := qVars.push qName
    vkqVars := vkqVars.push vkqName

  ShaderM.varNamed "kq_max" (.scalar .f32) (Exp.litF32 (-1.0e30))
  ShaderM.varNamed "kq_sum" (.scalar .f32) (Exp.litF32 0.0)
  let kqMax := Exp.var "kq_max"
  let kqSum := Exp.var "kq_sum"

  let cacheLen ← ShaderM.readBuffer (ty := .scalar .u32) (n := 2)
                   "params" (Exp.litU32 1)
  let kRowStride := Exp.litU32 headDim
  let kHeadBase := Exp.mul kvHead (Exp.mul (Exp.litU32 maxSeqLen) kRowStride)

  -- Outer K loop in chunks of workgroupSize (=128).
  ShaderM.loop (Exp.litU32 0) cacheLen (Exp.litU32 workgroupSize) fun kVKQ0 => do
    -- Per-thread KQ score slot — only the lane matching i_KQ_0 holds it.
    let kqRegVar ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    let kqMaxNewVar ← ShaderM.var (.scalar .f32) kqMax

    -- Phase 1: 32 K positions per warp, runtime loop (32 iter).  Each iter
    -- handles ONE K position via warp_reduce of D-axis.  Owning lane
    -- (iKQ0 == laneId) stores the score for the position it owns.
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 32) (Exp.litU32 1) fun iKQ0 => do
      let iKQ := Exp.add (Exp.mul warpId (Exp.litU32 32)) iKQ0
      let kPos := Exp.add kVKQ0 iKQ
      let inBounds := Exp.lt kPos cacheLen
      let kBase := Exp.add kHeadBase (Exp.mul kPos kRowStride)
      -- Per-lane partial dot: D/32 dims owned by this lane.  Compute
      -- regardless of inBounds — out-of-bounds reads OK (zero-fill or
      -- well-formed at this point assuming kPos*headDim < buffer size,
      -- which holds when maxSeqLen >= cacheLen — caller guarantees).
      let partialVar ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
      for k in [0:dPerLane] do
        let d := Exp.add (Exp.litU32 (k * 32)) laneId
        let kVal ← ShaderM.readBuffer (ty := .scalar .f32)
                     (n := numKVHeads * maxSeqLen * headDim) "k_cache"
                     (Exp.add kBase d)
        let qExp : Exp (.scalar .f32) := Exp.var qVars[k]!
        ShaderM.assign partialVar
          (Exp.add (Exp.var partialVar) (Exp.mul qExp kVal))
      -- Warp reduce: collapses D-axis across the 32 lanes.
      let sumWarpName ← ShaderM.var (.scalar .f32)
                          (Exp.subgroupAdd (Exp.var partialVar))
      let sumWarp : Exp (.scalar .f32) := Exp.var sumWarpName
      -- Out-of-bounds → -inf so it doesn't pollute max / sum.
      let scoreGated := Exp.select inBounds sumWarp (Exp.litF32 (-1.0e30))
      ShaderM.assign kqMaxNewVar
        (Exp.max (Exp.var kqMaxNewVar) scoreGated)
      -- Owning lane stores its score (gated -inf for out-of-bounds).
      ShaderM.if_ (Exp.eq laneId iKQ0) (do
        ShaderM.assign kqRegVar scoreGated
      ) (pure ())

    -- Phase 2: per-thread softmax-online update.
    let kqMaxNew := Exp.var kqMaxNewVar
    let kqMaxScaleVar ← ShaderM.var (.scalar .f32)
                          (Exp.exp (Exp.sub kqMax kqMaxNew))
    let kqMaxScale : Exp (.scalar .f32) := Exp.var kqMaxScaleVar
    ShaderM.assign "kq_max" kqMaxNew
    -- kqRegVar holds either the lane's own KQ score (if it owns an
    -- in-bounds K position in this warp's chunk) or -inf (if OOB).
    -- exp(-inf - finite) = 0 so OOB naturally contributes nothing.
    ShaderM.assign kqRegVar
      (Exp.exp (Exp.sub (Exp.var kqRegVar) kqMaxNew))
    -- Update kq_sum: per-lane contribution is its own kqReg (= 0 for OOB).
    ShaderM.assign "kq_sum"
      (Exp.add (Exp.mul kqSum kqMaxScale) (Exp.var kqRegVar))
    -- Rescale my-lane's VKQ accumulator dims by kqMaxScale.
    for k in [0:dPerLane] do
      let prev : Exp (.scalar .f32) := Exp.var vkqVars[k]!
      ShaderM.assign vkqVars[k]! (Exp.mul prev kqMaxScale)
    -- Publish my lane's KQ score into smem (OOB lanes write 0 naturally).
    let kqSlot := Exp.add (Exp.mul warpId (Exp.litU32 32)) laneId
    ShaderM.writeWorkgroup (ty := .scalar .f32)
      "shared_kq" kqSlot (Exp.var kqRegVar)
    ShaderM.barrier

    -- Phase 3: VKQ accumulation.  Runtime loop over 32 K positions in
    -- this warp's chunk.  Out-of-bounds slots have kqScore=0 (set by
    -- gated write in Phase 1+2), so we can skip the bounds check.
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 32) (Exp.litU32 1) fun kOff => do
      let kPos := Exp.add kVKQ0 (Exp.add (Exp.mul warpId (Exp.litU32 32)) kOff)
      let kqScoreSlot := Exp.add (Exp.mul warpId (Exp.litU32 32)) kOff
      let kqScoreRaw ← ShaderM.readWorkgroup (ty := .scalar .f32)
                        (n := workgroupSize) "shared_kq" kqScoreSlot
      -- Out-of-bounds kqScore is the exp(-inf - max) = 0, so it's safe.
      -- Gate V read by inBounds anyway to avoid OOB global access when
      -- kPos × headDim exceeds the V cache buffer.
      let inBounds := Exp.lt kPos cacheLen
      let kPosSafe := Exp.select inBounds kPos (Exp.litU32 0)
      let vBase := Exp.add kHeadBase (Exp.mul kPosSafe kRowStride)
      let kqScore := Exp.select inBounds kqScoreRaw (Exp.litF32 0.0)
      for k in [0:dPerLane] do
        let d := Exp.add (Exp.litU32 (k * 32)) laneId
        let vVal ← ShaderM.readBuffer (ty := .scalar .f32)
                     (n := numKVHeads * maxSeqLen * headDim) "v_cache"
                     (Exp.add vBase d)
        let prev : Exp (.scalar .f32) := Exp.var vkqVars[k]!
        ShaderM.assign vkqVars[k]!
          (Exp.add prev (Exp.mul vVal kqScore))
    ShaderM.barrier

  -- ============================================================
  -- Cross-warp softmax merge (the missing piece in V6 wip2)
  -- ============================================================
  --
  -- Each warp processed a disjoint K range and now holds:
  --   kq_max    — max score over its K range
  --   kq_sum    — Σ exp(score_k - kq_max) over its K range
  --   vkq[k]    — Σ V_k * exp(score_k - kq_max) over its K range
  --              (per-lane, owns dims k*32 + laneId for k in [0,dPerLane))
  --
  -- To merge into a global softmax over all K positions:
  --   global_max = max over warps of warp's kq_max
  --   weight_w = exp(warp_w.kq_max - global_max)
  --   global_sum = Σ_w warp_w.kq_sum * weight_w
  --   output = (Σ_w warp_w.vkq * weight_w) / global_sum
  --
  -- Note: kq_sum differs across lanes within a warp (each lane owned a
  -- different K position over the outer loop), so first warp_reduce
  -- across lanes to get the warp-level sum.

  -- Step 1: each warp's lane 0 writes (kq_max, warp_kq_sum) to smem.
  let warpKqSumName ← ShaderM.var (.scalar .f32) (Exp.subgroupAdd kqSum)
  ShaderM.if_ (Exp.eq laneId (Exp.litU32 0)) (do
    let metaIdx := Exp.mul warpId (Exp.litU32 2)
    ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_warp_meta"
      metaIdx kqMax
    ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_warp_meta"
      (Exp.add metaIdx (Exp.litU32 1)) (Exp.var warpKqSumName)
  ) (pure ())

  -- Step 2: each warp also writes its per-lane VKQ slice to shared_vkq
  -- (warpId * D + d slot).  Indexed by my-lane's owned dims.
  for k in [0:dPerLane] do
    let d := Exp.add (Exp.litU32 (k * 32)) laneId
    let slot := Exp.add (Exp.mul warpId (Exp.litU32 headDim)) d
    ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_vkq" slot
      (Exp.var vkqVars[k]!)
  ShaderM.barrier

  -- Step 3: every thread reads all 4 warps' (max, sum) from smem and
  -- computes global_max, global_sum, and per-warp weights.  Cheap to
  -- replicate across all threads — only 8 smem reads.
  let globalMaxName ← ShaderM.var (.scalar .f32) (Exp.litF32 (-1.0e30))
  for w in [0:numWarps] do
    let m ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := numWarps * 2)
              "shared_warp_meta" (Exp.litU32 (w * 2))
    ShaderM.assign globalMaxName (Exp.max (Exp.var globalMaxName) m)
  let globalMax : Exp (.scalar .f32) := Exp.var globalMaxName

  let globalSumName ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
  let mut weightVars : Array String := #[]
  for w in [0:numWarps] do
    let m ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := numWarps * 2)
              "shared_warp_meta" (Exp.litU32 (w * 2))
    let s ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := numWarps * 2)
              "shared_warp_meta" (Exp.litU32 (w * 2 + 1))
    let weight := Exp.exp (Exp.sub m globalMax)
    let weightName ← ShaderM.var (.scalar .f32) weight
    weightVars := weightVars.push weightName
    ShaderM.assign globalSumName
      (Exp.add (Exp.var globalSumName) (Exp.mul s (Exp.var weightName)))
  let globalSum : Exp (.scalar .f32) := Exp.var globalSumName

  -- Step 4: each thread combines all 4 warps' VKQ slices for its owned dims,
  -- weighted by per-warp weight, divided by globalSum.  Single-pass write
  -- to global output.  This pattern lets ALL threads contribute to the
  -- final write (not just warp 0), keeping bandwidth higher.
  for k in [0:dPerLane] do
    let d := Exp.add (Exp.litU32 (k * 32)) laneId
    let accVar ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    for w in [0:numWarps] do
      let slot := Exp.add (Exp.litU32 (w * headDim)) d
      let v ← ShaderM.readWorkgroup (ty := .scalar .f32)
                (n := numWarps * headDim) "shared_vkq" slot
      ShaderM.assign accVar
        (Exp.add (Exp.var accVar) (Exp.mul v (Exp.var weightVars[w]!)))
    let outIdx := Exp.add qBase d
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx
      (Exp.div (Exp.var accVar) globalSum)

/-- Vec-params V7 (doc 60): V6 K-parallel + K cache f16 + V cache f16.

    Per-thread layout differs from V6 because we read K/V as packed half2
    (1 u32 = 2 f16):
      lane laneId owns dim *pairs* (2*laneId, 2*laneId + 1) shifted by
      pairBase = pair_k * 64 for pair_k in [0, dPerLanePair).
      With D=256, ws=128: dPerLanePair = (D/2) / 32 = 4 pairs = 8 dims.

    K read: 1 u32 read per K row per pair → 2 f16 → unpack to 2 f32
    → 2 fmas into partialVar (D-axis dot product).

    V read: same — 1 u32 → 2 f32 → 2 weighted accums into vkq[2*pair_k]
    and vkq[2*pair_k + 1].

    Buffers (different from V6):
      k_cache_f16: array<u32, numKV * maxSeq * D / 2>  (each u32 = 2 f16)
      v_cache_f16: array<u32, numKV * maxSeq * D / 2>  (each u32 = 2 f16)

    Pre-condition: D % 64 == 0 (D=256 ✓), ws=128, headDim/2 % 32 == 0. -/
def flashAttentionVecParamsKernelV7
    (numHeads numKVHeads maxSeqLen headDim : Nat) (scale : Float) :
    ShaderM Unit := do
  let workgroupSize : Nat := 128
  let numWarps : Nat := workgroupSize / 32
  let dPerLanePair : Nat := (headDim / 2) / 32   -- 4 pairs/lane for D=256

  let wgid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let head := Exp.vec3X wgid
  let tid := Exp.vec3X lid
  let headsPerKV := numHeads / numKVHeads
  let kvHead := Exp.div head (Exp.litU32 headsPerKV)

  let _q ← ShaderM.declareInputBuffer "q" (.array (.scalar .f32) (numHeads * headDim))
  let kvWords : Nat := (numKVHeads * maxSeqLen * headDim) / 2
  let _kCache ← ShaderM.declareInputBuffer "k_cache_f16"
                  (.array (.scalar .u32) kvWords)
  let _vCache ← ShaderM.declareInputBuffer "v_cache_f16"
                  (.array (.scalar .u32) kvWords)
  let _output ← ShaderM.declareOutputBuffer "output"
                  (.array (.scalar .f32) (numHeads * headDim))
  let _params ← ShaderM.declareStorageBuffer "params" (.array (.scalar .u32) 2) .read

  ShaderM.sharedNamed "shared_kq" (.array (.scalar .f32) workgroupSize)
  ShaderM.sharedNamed "shared_vkq" (.array (.scalar .f32) (numWarps * headDim))
  ShaderM.sharedNamed "shared_warp_meta" (.array (.scalar .f32) (numWarps * 2))

  let laneId := Exp.bitAnd tid (Exp.litU32 31)
  let warpId := Exp.shiftRight tid (Exp.litU32 5)

  -- Pre-load Q (already scaled).  Q stays f32 — only K/V are f16.
  -- Lane owns dim pairs (2*laneId + 64*k, 2*laneId + 1 + 64*k).
  let qBase := Exp.mul head (Exp.litU32 headDim)
  let mut q0Vars : Array String := #[]  -- Q at even dim of pair
  let mut q1Vars : Array String := #[]  -- Q at odd dim of pair
  let mut vkq0Vars : Array String := #[]
  let mut vkq1Vars : Array String := #[]
  for pk in [0:dPerLanePair] do
    let d0 := Exp.add (Exp.litU32 (pk * 64)) (Exp.mul laneId (Exp.litU32 2))
    let d1 := Exp.add d0 (Exp.litU32 1)
    let q0Val ← ShaderM.readBuffer (ty := .scalar .f32) (n := numHeads * headDim)
                  "q" (Exp.add qBase d0)
    let q1Val ← ShaderM.readBuffer (ty := .scalar .f32) (n := numHeads * headDim)
                  "q" (Exp.add qBase d1)
    let q0Name ← ShaderM.var (.scalar .f32) (Exp.mul q0Val (Exp.litF32 scale))
    let q1Name ← ShaderM.var (.scalar .f32) (Exp.mul q1Val (Exp.litF32 scale))
    let vkq0Name ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    let vkq1Name ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    q0Vars := q0Vars.push q0Name
    q1Vars := q1Vars.push q1Name
    vkq0Vars := vkq0Vars.push vkq0Name
    vkq1Vars := vkq1Vars.push vkq1Name

  ShaderM.varNamed "kq_max" (.scalar .f32) (Exp.litF32 (-1.0e30))
  ShaderM.varNamed "kq_sum" (.scalar .f32) (Exp.litF32 0.0)
  let kqMax := Exp.var "kq_max"
  let kqSum := Exp.var "kq_sum"

  let cacheLen ← ShaderM.readBuffer (ty := .scalar .u32) (n := 2)
                   "params" (Exp.litU32 1)
  -- K row stride in u32 words = headDim / 2.
  let kRowStrideU32 := Exp.litU32 (headDim / 2)
  let kHeadBaseU32 := Exp.mul kvHead
                        (Exp.mul (Exp.litU32 maxSeqLen) kRowStrideU32)

  ShaderM.loop (Exp.litU32 0) cacheLen (Exp.litU32 workgroupSize) fun kVKQ0 => do
    let kqRegVar ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    let kqMaxNewVar ← ShaderM.var (.scalar .f32) kqMax

    -- Phase 1: 32 K positions per warp, runtime loop.
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 32) (Exp.litU32 1) fun iKQ0 => do
      let iKQ := Exp.add (Exp.mul warpId (Exp.litU32 32)) iKQ0
      let kPos := Exp.add kVKQ0 iKQ
      let inBounds := Exp.lt kPos cacheLen
      -- Per-K-row word base in K f16 buffer (rounded to in-bounds row).
      let kPosSafe := Exp.select inBounds kPos (Exp.litU32 0)
      let kRowBaseU32 := Exp.add kHeadBaseU32 (Exp.mul kPosSafe kRowStrideU32)
      let partialVar ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
      for pk in [0:dPerLanePair] do
        -- Word offset within K row: pair index = pk*32 + laneId (since
        -- 2*laneId / 2 = laneId; pk*64 / 2 = pk*32).
        let wordOff := Exp.add (Exp.litU32 (pk * 32)) laneId
        let kWordIdx := Exp.add kRowBaseU32 wordOff
        let kPacked ← ShaderM.readBuffer (ty := .scalar .u32) (n := kvWords)
                        "k_cache_f16" kWordIdx
        let unpacked := Exp.unpack2x16float kPacked
        let k0 := Exp.vecX unpacked
        let k1 := Exp.vecY unpacked
        let q0Exp : Exp (.scalar .f32) := Exp.var q0Vars[pk]!
        let q1Exp : Exp (.scalar .f32) := Exp.var q1Vars[pk]!
        ShaderM.assign partialVar
          (Exp.add (Exp.var partialVar) (Exp.mul q0Exp k0))
        ShaderM.assign partialVar
          (Exp.add (Exp.var partialVar) (Exp.mul q1Exp k1))

      let sumWarpName ← ShaderM.var (.scalar .f32)
                          (Exp.subgroupAdd (Exp.var partialVar))
      let sumWarp : Exp (.scalar .f32) := Exp.var sumWarpName
      let scoreGated := Exp.select inBounds sumWarp (Exp.litF32 (-1.0e30))
      ShaderM.assign kqMaxNewVar
        (Exp.max (Exp.var kqMaxNewVar) scoreGated)
      ShaderM.if_ (Exp.eq laneId iKQ0) (do
        ShaderM.assign kqRegVar scoreGated
      ) (pure ())

    -- Phase 2: per-thread softmax-online update.  Q is already scaled
    -- so kqReg here = Q·K (unscaled in V6 but scaled here since Q is pre-scaled).
    let kqMaxNew := Exp.var kqMaxNewVar
    let kqMaxScaleVar ← ShaderM.var (.scalar .f32)
                          (Exp.exp (Exp.sub kqMax kqMaxNew))
    let kqMaxScale : Exp (.scalar .f32) := Exp.var kqMaxScaleVar
    ShaderM.assign "kq_max" kqMaxNew
    ShaderM.assign kqRegVar
      (Exp.exp (Exp.sub (Exp.var kqRegVar) kqMaxNew))
    ShaderM.assign "kq_sum"
      (Exp.add (Exp.mul kqSum kqMaxScale) (Exp.var kqRegVar))
    for pk in [0:dPerLanePair] do
      let prev0 : Exp (.scalar .f32) := Exp.var vkq0Vars[pk]!
      let prev1 : Exp (.scalar .f32) := Exp.var vkq1Vars[pk]!
      ShaderM.assign vkq0Vars[pk]! (Exp.mul prev0 kqMaxScale)
      ShaderM.assign vkq1Vars[pk]! (Exp.mul prev1 kqMaxScale)
    let kqSlot := Exp.add (Exp.mul warpId (Exp.litU32 32)) laneId
    ShaderM.writeWorkgroup (ty := .scalar .f32)
      "shared_kq" kqSlot (Exp.var kqRegVar)
    ShaderM.barrier

    -- Phase 3: VKQ accumulation.  Read V as packed f16, unpack, accumulate.
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 32) (Exp.litU32 1) fun kOff => do
      let kPos := Exp.add kVKQ0 (Exp.add (Exp.mul warpId (Exp.litU32 32)) kOff)
      let kqScoreSlot := Exp.add (Exp.mul warpId (Exp.litU32 32)) kOff
      let kqScoreRaw ← ShaderM.readWorkgroup (ty := .scalar .f32)
                        (n := workgroupSize) "shared_kq" kqScoreSlot
      let inBounds := Exp.lt kPos cacheLen
      let kPosSafe := Exp.select inBounds kPos (Exp.litU32 0)
      let kqScore := Exp.select inBounds kqScoreRaw (Exp.litF32 0.0)
      let vRowBaseU32 := Exp.add kHeadBaseU32 (Exp.mul kPosSafe kRowStrideU32)
      for pk in [0:dPerLanePair] do
        let wordOff := Exp.add (Exp.litU32 (pk * 32)) laneId
        let vWordIdx := Exp.add vRowBaseU32 wordOff
        let vPacked ← ShaderM.readBuffer (ty := .scalar .u32) (n := kvWords)
                        "v_cache_f16" vWordIdx
        let unpacked := Exp.unpack2x16float vPacked
        let v0 := Exp.vecX unpacked
        let v1 := Exp.vecY unpacked
        let prev0 : Exp (.scalar .f32) := Exp.var vkq0Vars[pk]!
        let prev1 : Exp (.scalar .f32) := Exp.var vkq1Vars[pk]!
        ShaderM.assign vkq0Vars[pk]! (Exp.add prev0 (Exp.mul v0 kqScore))
        ShaderM.assign vkq1Vars[pk]! (Exp.add prev1 (Exp.mul v1 kqScore))
    ShaderM.barrier

  -- Cross-warp softmax merge (same as V6, adapted for paired vkq).
  let warpKqSumName ← ShaderM.var (.scalar .f32) (Exp.subgroupAdd kqSum)
  ShaderM.if_ (Exp.eq laneId (Exp.litU32 0)) (do
    let metaIdx := Exp.mul warpId (Exp.litU32 2)
    ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_warp_meta"
      metaIdx kqMax
    ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_warp_meta"
      (Exp.add metaIdx (Exp.litU32 1)) (Exp.var warpKqSumName)
  ) (pure ())

  -- Each warp writes its per-lane VKQ slice (paired) to shared_vkq.
  for pk in [0:dPerLanePair] do
    let d0 := Exp.add (Exp.litU32 (pk * 64)) (Exp.mul laneId (Exp.litU32 2))
    let d1 := Exp.add d0 (Exp.litU32 1)
    let slot0 := Exp.add (Exp.mul warpId (Exp.litU32 headDim)) d0
    let slot1 := Exp.add (Exp.mul warpId (Exp.litU32 headDim)) d1
    ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_vkq" slot0
      (Exp.var vkq0Vars[pk]!)
    ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_vkq" slot1
      (Exp.var vkq1Vars[pk]!)
  ShaderM.barrier

  -- Compute global max/sum across warps.
  let globalMaxName ← ShaderM.var (.scalar .f32) (Exp.litF32 (-1.0e30))
  for w in [0:numWarps] do
    let m ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := numWarps * 2)
              "shared_warp_meta" (Exp.litU32 (w * 2))
    ShaderM.assign globalMaxName (Exp.max (Exp.var globalMaxName) m)
  let globalMax : Exp (.scalar .f32) := Exp.var globalMaxName

  let globalSumName ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
  let mut weightVars : Array String := #[]
  for w in [0:numWarps] do
    let m ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := numWarps * 2)
              "shared_warp_meta" (Exp.litU32 (w * 2))
    let s ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := numWarps * 2)
              "shared_warp_meta" (Exp.litU32 (w * 2 + 1))
    let weight := Exp.exp (Exp.sub m globalMax)
    let weightName ← ShaderM.var (.scalar .f32) weight
    weightVars := weightVars.push weightName
    ShaderM.assign globalSumName
      (Exp.add (Exp.var globalSumName) (Exp.mul s (Exp.var weightName)))
  let globalSum : Exp (.scalar .f32) := Exp.var globalSumName

  -- Final write: sum across warps, divide, write paired dims.
  for pk in [0:dPerLanePair] do
    let d0 := Exp.add (Exp.litU32 (pk * 64)) (Exp.mul laneId (Exp.litU32 2))
    let d1 := Exp.add d0 (Exp.litU32 1)
    let acc0Var ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    let acc1Var ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    for w in [0:numWarps] do
      let slot0 := Exp.add (Exp.litU32 (w * headDim)) d0
      let slot1 := Exp.add (Exp.litU32 (w * headDim)) d1
      let v0 ← ShaderM.readWorkgroup (ty := .scalar .f32)
                (n := numWarps * headDim) "shared_vkq" slot0
      let v1 ← ShaderM.readWorkgroup (ty := .scalar .f32)
                (n := numWarps * headDim) "shared_vkq" slot1
      ShaderM.assign acc0Var
        (Exp.add (Exp.var acc0Var) (Exp.mul v0 (Exp.var weightVars[w]!)))
      ShaderM.assign acc1Var
        (Exp.add (Exp.var acc1Var) (Exp.mul v1 (Exp.var weightVars[w]!)))
    let outIdx0 := Exp.add qBase d0
    let outIdx1 := Exp.add qBase d1
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx0
      (Exp.div (Exp.var acc0Var) globalSum)
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx1
      (Exp.div (Exp.var acc1Var) globalSum)

/-- Vec-params V5 (Session 5, doc 60): V2 + split-K parallelism.

    Splits the K range across `numSplits` blocks per head.  Grid is
    `(numHeads, numSplits, 1)` so all `numHeads * numSplits` blocks
    can run concurrently on different SMs.  Each block computes a
    partial (sum, max) softmax + partial VKQ over its K-slice.

    A separate combine kernel (flashAttentionVecCombineKernel) reduces
    the partial outputs across splits using online softmax.

    Output layout (intermediate):
      partial_out[head × numSplits × headDim + split × headDim + d]
        = the un-normalised SUM term for this split (rescaled later)
      partial_meta[head × numSplits + split]
        = packed (max, sum) — stored as 2 consecutive f32 in
          partial_meta[head * numSplits * 2 + split * 2 + {0,1}]

    Pre-condition: cacheLen >= numSplits (else some splits get 0 work
    and produce -inf max → handled by combine).
    headDim % 128 == 0.
-/
def flashAttentionVecParamsKernelV5
    (numHeads numKVHeads maxSeqLen headDim numSplits : Nat) (scale : Float) :
    ShaderM Unit := do
  let workgroupSize : Nat := 128
  let _numWarps : Nat := workgroupSize / 32
  let elemsPerThread : Nat := headDim / workgroupSize

  let wgid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let head := Exp.vec3X wgid
  let splitIdx := Exp.vec3Y wgid
  let tid := Exp.vec3X lid
  let headsPerKV := numHeads / numKVHeads
  let kvHead := Exp.div head (Exp.litU32 headsPerKV)

  let _q ← ShaderM.declareInputBuffer "q" (.array (.scalar .f32) (numHeads * headDim))
  let _kCache ← ShaderM.declareInputBuffer "k_cache"
                  (.array (.scalar .f32) (numKVHeads * maxSeqLen * headDim))
  let _vCache ← ShaderM.declareInputBuffer "v_cache"
                  (.array (.scalar .f32) (numKVHeads * maxSeqLen * headDim))
  let _partialOut ← ShaderM.declareOutputBuffer "partial_out"
                      (.array (.scalar .f32) (numHeads * numSplits * headDim))
  let _partialMeta ← ShaderM.declareOutputBuffer "partial_meta"
                       (.array (.scalar .f32) (numHeads * numSplits * 2))
  let _params ← ShaderM.declareStorageBuffer "params" (.array (.scalar .u32) 2) .read

  ShaderM.sharedNamed "shared_warp_sums" (.array (.scalar .f32) 32)

  let qBase := Exp.mul head (Exp.litU32 headDim)
  let dBase := Exp.mul tid (Exp.litU32 elemsPerThread)

  let mut qVars : Array String := #[]
  let mut vkqVars : Array String := #[]
  for ep in [0:elemsPerThread] do
    let d := Exp.add dBase (Exp.litU32 ep)
    let qVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numHeads * headDim)
                 "q" (Exp.add qBase d)
    let qName ← ShaderM.var (.scalar .f32) qVal
    let vkqName ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    qVars := qVars.push qName
    vkqVars := vkqVars.push vkqName

  ShaderM.varNamed "max_score" (.scalar .f32) (Exp.litF32 (-1.0e30))
  ShaderM.varNamed "sum_exp" (.scalar .f32) (Exp.litF32 0.0)
  let maxScore := Exp.var "max_score"
  let sumExp := Exp.var "sum_exp"

  let cacheLen ← ShaderM.readBuffer (ty := .scalar .u32) (n := 2)
                   "params" (Exp.litU32 1)
  let laneId := Exp.bitAnd tid (Exp.litU32 31)
  let warpId := Exp.shiftRight tid (Exp.litU32 5)

  -- Compute this split's K range: [splitStart, splitEnd).
  -- splitStart = splitIdx * cacheLen / numSplits
  -- splitEnd   = (splitIdx+1) * cacheLen / numSplits
  let splitStart := Exp.div (Exp.mul splitIdx cacheLen) (Exp.litU32 numSplits)
  let splitEnd := Exp.div (Exp.mul (Exp.add splitIdx (Exp.litU32 1)) cacheLen)
                          (Exp.litU32 numSplits)

  ShaderM.loop splitStart splitEnd (Exp.litU32 1) fun s => do
    let partialVar ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    let kBase := Exp.add (Exp.mul (Exp.mul kvHead (Exp.litU32 maxSeqLen))
                                  (Exp.litU32 headDim))
                          (Exp.mul s (Exp.litU32 headDim))
    for ep in [0:elemsPerThread] do
      let d := Exp.add dBase (Exp.litU32 ep)
      let qExp : Exp (.scalar .f32) := Exp.var qVars[ep]!
      let kVal ← ShaderM.readBuffer (ty := .scalar .f32)
                   (n := numKVHeads * maxSeqLen * headDim) "k_cache"
                   (Exp.add kBase d)
      ShaderM.assign partialVar
        (Exp.add (Exp.var partialVar) (Exp.mul qExp kVal))

    let warpSumName ← ShaderM.var (.scalar .f32)
                        (Exp.subgroupAdd (Exp.var partialVar))
    let warpSum : Exp (.scalar .f32) := Exp.var warpSumName

    ShaderM.if_ (Exp.eq laneId (Exp.litU32 0)) (do
      ShaderM.writeWorkgroup (ty := .scalar .f32)
        "shared_warp_sums" warpId warpSum
    ) (pure ())
    ShaderM.barrier

    let slotVal ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 32)
                    "shared_warp_sums" laneId
    let guardedName ← ShaderM.var (.scalar .f32)
      (Exp.select (Exp.lt laneId (Exp.litU32 _numWarps))
                  slotVal (Exp.litF32 0.0))
    let totalName ← ShaderM.var (.scalar .f32)
                      (Exp.subgroupAdd (Exp.var guardedName))
    ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
      ShaderM.writeWorkgroup (ty := .scalar .f32)
        "shared_warp_sums" (Exp.litU32 0) (Exp.var totalName)
    ) (pure ())
    ShaderM.barrier

    let scoreFromShared ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 32)
                            "shared_warp_sums" (Exp.litU32 0)
    let scaledScore := Exp.mul (Exp.litF32 scale) scoreFromShared

    let oldMaxVar ← ShaderM.var (.scalar .f32) maxScore
    let oldSumVar ← ShaderM.var (.scalar .f32) sumExp
    let oldMax := Exp.var oldMaxVar
    let oldSum := Exp.var oldSumVar
    let newMax := Exp.max oldMax scaledScore
    let expOld := Exp.exp (Exp.sub oldMax newMax)
    let expNew := Exp.exp (Exp.sub scaledScore newMax)
    let newSum := Exp.add (Exp.mul oldSum expOld) expNew
    let rescaleVar ← ShaderM.var (.scalar .f32)
                       (Exp.div (Exp.mul oldSum expOld) newSum)
    let contribVar ← ShaderM.var (.scalar .f32)
                       (Exp.div expNew newSum)

    for ep in [0:elemsPerThread] do
      let d := Exp.add dBase (Exp.litU32 ep)
      let vVal ← ShaderM.readBuffer (ty := .scalar .f32)
                   (n := numKVHeads * maxSeqLen * headDim) "v_cache"
                   (Exp.add kBase d)
      let prevExp : Exp (.scalar .f32) := Exp.var vkqVars[ep]!
      ShaderM.assign vkqVars[ep]!
        (Exp.add (Exp.mul prevExp (Exp.var rescaleVar))
                 (Exp.mul vVal   (Exp.var contribVar)))

    ShaderM.assign "max_score" newMax
    ShaderM.assign "sum_exp" newSum

  -- Write per-thread VKQ to partial output.  No final-normalise — combine
  -- kernel does that.  Write the (max, sum) meta from thread 0.
  let partialBase := Exp.add (Exp.mul head
                                       (Exp.mul (Exp.litU32 numSplits) (Exp.litU32 headDim)))
                              (Exp.mul splitIdx (Exp.litU32 headDim))
  for ep in [0:elemsPerThread] do
    let d := Exp.add dBase (Exp.litU32 ep)
    let outIdx := Exp.add partialBase d
    -- VKQ in this kernel is normalised (rescale/contrib divides by
    -- per-split newSum). To allow the combine kernel to reweight by
    -- the right global softmax denominator, we re-multiply by per-split
    -- sum_exp here so partial_out[d] = un-normalised numerator.
    let vkqExp : Exp (.scalar .f32) := Exp.var vkqVars[ep]!
    ShaderM.writeBuffer (ty := .scalar .f32) "partial_out" outIdx
      (Exp.mul vkqExp sumExp)

  ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
    let metaBase := Exp.mul head (Exp.mul (Exp.litU32 numSplits) (Exp.litU32 2))
    let metaIdx := Exp.add metaBase (Exp.mul splitIdx (Exp.litU32 2))
    ShaderM.writeBuffer (ty := .scalar .f32) "partial_meta" metaIdx maxScore
    ShaderM.writeBuffer (ty := .scalar .f32) "partial_meta"
      (Exp.add metaIdx (Exp.litU32 1)) sumExp
  ) (pure ())

/-- Combine kernel for V5 split-K: reduces `numSplits` partial outputs
    per head into one final output via online softmax over the (max, sum)
    metadata.

    Grid: (numHeads, 1, 1).  Block: 128 threads.  Each thread owns
    `headDim/128 = 2` output dims. -/
def flashAttentionVecCombineKernel
    (numHeads headDim numSplits : Nat) : ShaderM Unit := do
  let workgroupSize : Nat := 128
  let elemsPerThread : Nat := headDim / workgroupSize

  let wgid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let head := Exp.vec3X wgid
  let tid := Exp.vec3X lid

  let _partialOut ← ShaderM.declareInputBuffer "partial_out"
                      (.array (.scalar .f32) (numHeads * numSplits * headDim))
  let _partialMeta ← ShaderM.declareInputBuffer "partial_meta"
                       (.array (.scalar .f32) (numHeads * numSplits * 2))
  let _output ← ShaderM.declareOutputBuffer "output"
                  (.array (.scalar .f32) (numHeads * headDim))

  let dBase := Exp.mul tid (Exp.litU32 elemsPerThread)
  let outBase := Exp.mul head (Exp.litU32 headDim)
  let partBase := Exp.mul head (Exp.mul (Exp.litU32 numSplits) (Exp.litU32 headDim))
  let metaBase := Exp.mul head (Exp.mul (Exp.litU32 numSplits) (Exp.litU32 2))

  -- Pass 1: find global max across splits.
  let globalMaxVar ← ShaderM.var (.scalar .f32) (Exp.litF32 (-1.0e30))
  for split in [0:numSplits] do
    let m ← ShaderM.readBuffer (ty := .scalar .f32) (n := numHeads * numSplits * 2)
              "partial_meta"
              (Exp.add metaBase (Exp.mul (Exp.litU32 split) (Exp.litU32 2)))
    ShaderM.assign globalMaxVar (Exp.max (Exp.var globalMaxVar) m)
  let globalMax : Exp (.scalar .f32) := Exp.var globalMaxVar

  -- Pass 2: compute denom = sum over splits of (sum_split * exp(max_split - global_max))
  let denomVar ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
  for split in [0:numSplits] do
    let mIdx := Exp.add metaBase (Exp.mul (Exp.litU32 split) (Exp.litU32 2))
    let m ← ShaderM.readBuffer (ty := .scalar .f32) (n := numHeads * numSplits * 2)
              "partial_meta" mIdx
    let s ← ShaderM.readBuffer (ty := .scalar .f32) (n := numHeads * numSplits * 2)
              "partial_meta" (Exp.add mIdx (Exp.litU32 1))
    let w := Exp.exp (Exp.sub m globalMax)
    ShaderM.assign denomVar (Exp.add (Exp.var denomVar) (Exp.mul s w))
  let denom : Exp (.scalar .f32) := Exp.var denomVar

  -- Pass 3: per-dim weighted sum over splits, divide by denom.
  for ep in [0:elemsPerThread] do
    let d := Exp.add dBase (Exp.litU32 ep)
    let accVar ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    for split in [0:numSplits] do
      let mIdx := Exp.add metaBase (Exp.mul (Exp.litU32 split) (Exp.litU32 2))
      let m ← ShaderM.readBuffer (ty := .scalar .f32) (n := numHeads * numSplits * 2)
                "partial_meta" mIdx
      let pIdx := Exp.add (Exp.add partBase
                            (Exp.mul (Exp.litU32 split) (Exp.litU32 headDim))) d
      let p ← ShaderM.readBuffer (ty := .scalar .f32)
                (n := numHeads * numSplits * headDim) "partial_out" pIdx
      let w := Exp.exp (Exp.sub m globalMax)
      -- p was un-normalised numerator (partial_vkq * partial_sum)
      ShaderM.assign accVar (Exp.add (Exp.var accVar) (Exp.mul p w))
    let outIdx := Exp.add outBase d
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx
      (Exp.div (Exp.var accVar) denom)

/-- Batched flash-attention for prefill: processes `seqLen` query tokens in
    one dispatch, attending each over its own causal prefix of the K/V cache.

    Layout (column-major Q/output, KV cache as in single-token kernel):
    - q[col * (numHeads * headDim) + h * headDim + d]            -- col = query token index
    - output[col * (numHeads * headDim) + h * headDim + d]
    - k_cache, v_cache: [numKVHeads, maxSeqLen, headDim] (same as single-token)

    Grid: (numHeads, seqLen, 1).  Each WG owns one (head, query token).
    cacheLen for query token col = startPos + col + 1 (causal).

    `params[0] = startPos` — KV cache length BEFORE this batch was written.
    For prefill from scratch, startPos = 0 → cacheLen = col+1 per token.

    PTX is fixed for given (numHeads, numKVHeads, maxSeqLen, headDim, seqLen). -/
def flashAttentionBatchKernel (numHeads numKVHeads maxSeqLen headDim seqLen : Nat)
    (scale : Float) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let wgid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let head := Exp.vec3X wgid
  let col  := Exp.vec3Y wgid           -- query token index
  let tid  := Exp.vec3X lid

  let headsPerKV := numHeads / numKVHeads
  let kvHead := Exp.div head (Exp.litU32 headsPerKV)
  let qDim := numHeads * headDim

  let _q ← ShaderM.declareInputBuffer "q" (.array (.scalar .f32) (qDim * seqLen))
  let _kCache ← ShaderM.declareInputBuffer "k_cache" (.array (.scalar .f32) (numKVHeads * maxSeqLen * headDim))
  let _vCache ← ShaderM.declareInputBuffer "v_cache" (.array (.scalar .f32) (numKVHeads * maxSeqLen * headDim))
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) (qDim * seqLen))
  let _params ← ShaderM.declareStorageBuffer "params" (.array (.scalar .u32) 1) .read

  ShaderM.sharedNamed "shared_q" (.array (.scalar .f32) headDim)
  ShaderM.sharedNamed "shared_reduce" (.array (.scalar .f32) workgroupSize)
  ShaderM.sharedNamed "shared_out" (.array (.scalar .f32) headDim)

  do
    -- Load Q row for (col, head) into shared memory; zero accumulator.
    let qBase := Exp.add (Exp.mul col (Exp.litU32 qDim)) (Exp.mul head (Exp.litU32 headDim))
    ShaderM.loop tid (Exp.litU32 headDim) (Exp.litU32 workgroupSize) fun d => do
      let qVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := qDim * seqLen) "q" (Exp.add qBase d)
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_q" d qVal
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_out" d (Exp.litF32 0.0)
    ShaderM.barrier

    ShaderM.varNamed "max_score" (.scalar .f32) (Exp.litF32 (-1.0e30))
    ShaderM.varNamed "sum_exp"   (.scalar .f32) (Exp.litF32 0.0)
    let maxScore := Exp.var "max_score"
    let sumExp   := Exp.var "sum_exp"

    -- cacheLen for this query token = startPos + col + 1 (causal)
    let startPos ← ShaderM.readBuffer (ty := .scalar .u32) (n := 1) "params" (Exp.litU32 0)
    let cacheLen := Exp.add startPos (Exp.add col (Exp.litU32 1))

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

      let rescaleFactor := Exp.div (Exp.mul oldSum expOld) newSum
      let contribFactor := Exp.div expNew newSum

      ShaderM.loop tid (Exp.litU32 headDim) (Exp.litU32 workgroupSize) fun d => do
        let vVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numKVHeads * maxSeqLen * headDim) "v_cache" (Exp.add kBase d)
        let prev ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := headDim) "shared_out" d
        let updated := Exp.add (Exp.mul prev rescaleFactor) (Exp.mul vVal contribFactor)
        ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_out" d updated

      ShaderM.assign "max_score" newMax
      ShaderM.assign "sum_exp"   newSum
      ShaderM.barrier

    -- Write attn output for this (col, head) row.
    let outBase := Exp.add (Exp.mul col (Exp.litU32 qDim)) (Exp.mul head (Exp.litU32 headDim))
    ShaderM.loop tid (Exp.litU32 headDim) (Exp.litU32 workgroupSize) fun d => do
      let v ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := headDim) "shared_out" d
      ShaderM.writeBuffer (ty := .scalar .f32) "output" (Exp.add outBase d) v

/-! ## Subgroup Flash Attention (M=1 decode, no barriers, no shared mem) -/

/-- Subgroup-based M=1 flash attention kernel. Replaces
    `flashAttentionDynamicKernel`'s 256-thread-tree-reduce +
    10-barriers-per-position scheme with a 32-thread (1 hardware
    subgroup) design:

      * 1 workgroup per attention head, 32 threads
      * headDim is partitioned across lanes with stride 32 — lane
        `tid` owns dims `{tid, tid+32, tid+64, ...}`. Gemma 4 has
        headDim=128 so that's exactly 4 dims per lane, fully unrolled.
      * Q is read once per head and held in per-lane registers
        (`q0..q3`), no shared memory.
      * Output accumulator is also per-lane registers (`o0..o3`).
      * Per cached position:
          - lane reads its 4 K values, computes 4 FMAs against q*,
            then ONE `subgroupAdd` → score broadcast to every lane
          - scalar online softmax update in every lane (identical
            results because the subgroupAdd broadcasts the same dot
            product to all 32 lanes)
          - lane reads its 4 V values and applies the rescale/contrib
            factors to o0..o3
      * No shared memory, no barriers, no tree reduction.

    Constraint: `headDim % 32 == 0 && headDim / 32 ≤ 8`. Gemma 4's
    headDim=128 → 4 regs per lane, well within register budget. Also
    requires subgroup support (every desktop Vulkan/NVIDIA driver).
-/
def flashAttentionSubgroupKernel (numHeads numKVHeads maxSeqLen headDim cacheLen : Nat)
    (scale : Float) : ShaderM Unit := do
  let wgid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let head := Exp.vec3X wgid
  let tid := Exp.vec3X lid

  let headsPerKV := numHeads / numKVHeads
  let kvHead := Exp.div head (Exp.litU32 headsPerKV)
  -- Number of headDim slices each lane handles (4 for Gemma 4 headDim=128).
  let dimsPerLane := headDim / 32

  let _q ← ShaderM.declareInputBuffer "q" (.array (.scalar .f32) (numHeads * headDim))
  let _kCache ← ShaderM.declareInputBuffer "k_cache" (.array (.scalar .f32) (numKVHeads * maxSeqLen * headDim))
  let _vCache ← ShaderM.declareInputBuffer "v_cache" (.array (.scalar .f32) (numKVHeads * maxSeqLen * headDim))
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) (numHeads * headDim))

  let qHeadBase := Exp.mul head (Exp.litU32 headDim)
  let kvHeadBase := Exp.mul (Exp.mul kvHead (Exp.litU32 maxSeqLen)) (Exp.litU32 headDim)

  -- Load Q into per-lane registers. Lane tid owns dims {tid, tid+32, ...}.
  for slice in [0:dimsPerLane] do
    let sliceOff := Exp.add tid (Exp.litU32 (slice * 32))
    let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := numHeads * headDim) "q"
              (Exp.add qHeadBase sliceOff)
    ShaderM.varNamed s!"q{slice}" (.scalar .f32) v

  -- Output accumulators, one per owned dim, zero-initialised.
  for slice in [0:dimsPerLane] do
    ShaderM.varNamed s!"o{slice}" (.scalar .f32) (Exp.litF32 0.0)

  -- Online-softmax running state. Every lane holds the same values
  -- because `subgroupAdd` broadcasts the score to all lanes.
  ShaderM.varNamed "max_score" (.scalar .f32) (Exp.litF32 (-1.0e30))
  ShaderM.varNamed "sum_exp" (.scalar .f32) (Exp.litF32 0.0)
  let maxScore : Exp (.scalar .f32) := Exp.var "max_score"
  let sumExp : Exp (.scalar .f32) := Exp.var "sum_exp"

  -- Main loop over cached positions. cacheLen is a Lean Nat (compile
  -- time for the decode path), so we keep it as a runtime ShaderM.loop
  -- to avoid exploding the WGSL source size.
  ShaderM.loop (Exp.litU32 0) (Exp.litU32 cacheLen) (Exp.litU32 1) fun s => do
    let posBase := Exp.add kvHeadBase (Exp.mul s (Exp.litU32 headDim))

    -- Per-position dot product: 4 FMAs per lane then one subgroupAdd.
    ShaderM.varNamed "partialDot" (.scalar .f32) (Exp.litF32 0.0)
    let partialDot : Exp (.scalar .f32) := Exp.var "partialDot"
    for slice in [0:dimsPerLane] do
      let sliceOff := Exp.add tid (Exp.litU32 (slice * 32))
      let kVal ← ShaderM.readBuffer (ty := .scalar .f32)
                  (n := numKVHeads * maxSeqLen * headDim) "k_cache"
                  (Exp.add posBase sliceOff)
      let qReg : Exp (.scalar .f32) := Exp.var s!"q{slice}"
      ShaderM.assign "partialDot" (Exp.add partialDot (Exp.mul qReg kVal))

    -- Broadcast-sum across all 32 lanes. **Critical**: we materialise
    -- the subgroupAdd result and all softmax intermediates into named
    -- `var`s rather than building Exp trees — otherwise each Lean-level
    -- `let` re-inlines the whole subexpression, and subgroupAdd would
    -- end up being emitted dozens of times per loop iteration.
    ShaderM.varNamed "dot"    (.scalar .f32) (Exp.subgroupAdd partialDot)
    let dot : Exp (.scalar .f32) := Exp.var "dot"
    ShaderM.varNamed "score"  (.scalar .f32) (Exp.mul (Exp.litF32 scale) dot)
    let score : Exp (.scalar .f32) := Exp.var "score"

    ShaderM.varNamed "newMax" (.scalar .f32) (Exp.max maxScore score)
    let newMax : Exp (.scalar .f32) := Exp.var "newMax"
    ShaderM.varNamed "expOld" (.scalar .f32) (Exp.exp (Exp.sub maxScore newMax))
    let expOld : Exp (.scalar .f32) := Exp.var "expOld"
    ShaderM.varNamed "expNew" (.scalar .f32) (Exp.exp (Exp.sub score newMax))
    let expNew : Exp (.scalar .f32) := Exp.var "expNew"
    ShaderM.varNamed "newSum" (.scalar .f32) (Exp.add (Exp.mul sumExp expOld) expNew)
    let newSum : Exp (.scalar .f32) := Exp.var "newSum"
    ShaderM.varNamed "rescaleFactor" (.scalar .f32) (Exp.div (Exp.mul sumExp expOld) newSum)
    let rescaleFactor : Exp (.scalar .f32) := Exp.var "rescaleFactor"
    ShaderM.varNamed "contribFactor" (.scalar .f32) (Exp.div expNew newSum)
    let contribFactor : Exp (.scalar .f32) := Exp.var "contribFactor"

    -- Rescale the running output accumulators and add this position's
    -- V contribution. Each lane touches only its owned dims; V is
    -- loaded lane-wise with the same stride pattern as K.
    for slice in [0:dimsPerLane] do
      let sliceOff := Exp.add tid (Exp.litU32 (slice * 32))
      let vVal ← ShaderM.readBuffer (ty := .scalar .f32)
                  (n := numKVHeads * maxSeqLen * headDim) "v_cache"
                  (Exp.add posBase sliceOff)
      let oReg : Exp (.scalar .f32) := Exp.var s!"o{slice}"
      ShaderM.assign s!"o{slice}"
        (Exp.add (Exp.mul oReg rescaleFactor) (Exp.mul vVal contribFactor))

    ShaderM.assign "max_score" newMax
    ShaderM.assign "sum_exp" newSum

  -- Write per-lane output slices back to global memory.
  for slice in [0:dimsPerLane] do
    let sliceOff := Exp.add tid (Exp.litU32 (slice * 32))
    let outIdx := Exp.add qHeadBase sliceOff
    let oReg : Exp (.scalar .f32) := Exp.var s!"o{slice}"
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx oReg

/-- Params-buffer variant of `flashAttentionSubgroupKernel`: reads
    `cacheLen` from a 2-u32 params buffer (position 1 = cacheLen, matching
    the layout used by `flashAttentionDynamicParamsKernel`).  This keeps
    the PTX shape-stable across decode positions — exactly what CUDA
    Graph capture needs to replay correctly past the initial
    cacheLen-at-capture boundary.

    All other properties are identical to `flashAttentionSubgroupKernel`:
    32-thread workgroup, per-lane headDim slicing, single `subgroupAdd`
    per position, online softmax in registers, no shared memory, no
    barriers. -/
def flashAttentionSubgroupParamsKernel (numHeads numKVHeads maxSeqLen headDim : Nat)
    (scale : Float) : ShaderM Unit := do
  let wgid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let head := Exp.vec3X wgid
  let tid := Exp.vec3X lid

  let headsPerKV := numHeads / numKVHeads
  let kvHead := Exp.div head (Exp.litU32 headsPerKV)
  let dimsPerLane := headDim / 32

  let _q ← ShaderM.declareInputBuffer "q" (.array (.scalar .f32) (numHeads * headDim))
  let _kCache ← ShaderM.declareInputBuffer "k_cache" (.array (.scalar .f32) (numKVHeads * maxSeqLen * headDim))
  let _vCache ← ShaderM.declareInputBuffer "v_cache" (.array (.scalar .f32) (numKVHeads * maxSeqLen * headDim))
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) (numHeads * headDim))
  let _params ← ShaderM.declareStorageBuffer "params" (.array (.scalar .u32) 2) .read

  let qHeadBase := Exp.mul head (Exp.litU32 headDim)
  let kvHeadBase := Exp.mul (Exp.mul kvHead (Exp.litU32 maxSeqLen)) (Exp.litU32 headDim)

  for slice in [0:dimsPerLane] do
    let sliceOff := Exp.add tid (Exp.litU32 (slice * 32))
    let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := numHeads * headDim) "q"
              (Exp.add qHeadBase sliceOff)
    ShaderM.varNamed s!"q{slice}" (.scalar .f32) v

  for slice in [0:dimsPerLane] do
    ShaderM.varNamed s!"o{slice}" (.scalar .f32) (Exp.litF32 0.0)

  ShaderM.varNamed "max_score" (.scalar .f32) (Exp.litF32 (-1.0e30))
  ShaderM.varNamed "sum_exp" (.scalar .f32) (Exp.litF32 0.0)
  let maxScore : Exp (.scalar .f32) := Exp.var "max_score"
  let sumExp : Exp (.scalar .f32) := Exp.var "sum_exp"

  -- Read cacheLen from params[1].
  let cacheLen ← ShaderM.readBuffer (ty := .scalar .u32) (n := 2) "params" (Exp.litU32 1)

  ShaderM.loop (Exp.litU32 0) cacheLen (Exp.litU32 1) fun s => do
    let posBase := Exp.add kvHeadBase (Exp.mul s (Exp.litU32 headDim))

    ShaderM.varNamed "partialDot" (.scalar .f32) (Exp.litF32 0.0)
    let partialDot : Exp (.scalar .f32) := Exp.var "partialDot"
    for slice in [0:dimsPerLane] do
      let sliceOff := Exp.add tid (Exp.litU32 (slice * 32))
      let kVal ← ShaderM.readBuffer (ty := .scalar .f32)
                  (n := numKVHeads * maxSeqLen * headDim) "k_cache"
                  (Exp.add posBase sliceOff)
      let qReg : Exp (.scalar .f32) := Exp.var s!"q{slice}"
      ShaderM.assign "partialDot" (Exp.add partialDot (Exp.mul qReg kVal))

    ShaderM.varNamed "dot" (.scalar .f32) (Exp.subgroupAdd partialDot)
    let dot : Exp (.scalar .f32) := Exp.var "dot"
    ShaderM.varNamed "score" (.scalar .f32) (Exp.mul (Exp.litF32 scale) dot)
    let score : Exp (.scalar .f32) := Exp.var "score"

    ShaderM.varNamed "newMax" (.scalar .f32) (Exp.max maxScore score)
    let newMax : Exp (.scalar .f32) := Exp.var "newMax"
    ShaderM.varNamed "expOld" (.scalar .f32) (Exp.exp (Exp.sub maxScore newMax))
    let expOld : Exp (.scalar .f32) := Exp.var "expOld"
    ShaderM.varNamed "expNew" (.scalar .f32) (Exp.exp (Exp.sub score newMax))
    let expNew : Exp (.scalar .f32) := Exp.var "expNew"
    ShaderM.varNamed "newSum" (.scalar .f32) (Exp.add (Exp.mul sumExp expOld) expNew)
    let newSum : Exp (.scalar .f32) := Exp.var "newSum"
    ShaderM.varNamed "rescaleFactor" (.scalar .f32) (Exp.div (Exp.mul sumExp expOld) newSum)
    let rescaleFactor : Exp (.scalar .f32) := Exp.var "rescaleFactor"
    ShaderM.varNamed "contribFactor" (.scalar .f32) (Exp.div expNew newSum)
    let contribFactor : Exp (.scalar .f32) := Exp.var "contribFactor"

    for slice in [0:dimsPerLane] do
      let sliceOff := Exp.add tid (Exp.litU32 (slice * 32))
      let vVal ← ShaderM.readBuffer (ty := .scalar .f32)
                  (n := numKVHeads * maxSeqLen * headDim) "v_cache"
                  (Exp.add posBase sliceOff)
      let oReg : Exp (.scalar .f32) := Exp.var s!"o{slice}"
      ShaderM.assign s!"o{slice}"
        (Exp.add (Exp.mul oReg rescaleFactor) (Exp.mul vVal contribFactor))

    ShaderM.assign "max_score" newMax
    ShaderM.assign "sum_exp" newSum

  for slice in [0:dimsPerLane] do
    let sliceOff := Exp.add tid (Exp.litU32 (slice * 32))
    let outIdx := Exp.add qHeadBase sliceOff
    let oReg : Exp (.scalar .f32) := Exp.var s!"o{slice}"
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx oReg

/-! ## SWA variant: subgroup flash attention with sliding window -/

/-- SWA (Sliding Window Attention) variant of `flashAttentionSubgroupKernel`.
    Identical layout, but only positions in `[currentPos - windowSize + 1,
    currentPos]` contribute to the softmax. We implement that by masking:
    positions outside the window still loop but get `score = -inf` so
    their softmax weight is zero.

    Simpler than llama.cpp's "only loop over window" optimisation (that
    would need a dynamic loop start and break the compile-time cacheLen
    specialisation), but the wasted positions are just a few scalar
    ops per masked-out index per lane, which is negligible vs. the
    K/V memory loads we skip. -/
def flashAttentionSWASubgroupKernel (numHeads numKVHeads maxSeqLen headDim
    cacheLen windowSize currentPos : Nat) (scale : Float) : ShaderM Unit := do
  let wgid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let head := Exp.vec3X wgid
  let tid := Exp.vec3X lid

  let headsPerKV := numHeads / numKVHeads
  let kvHead := Exp.div head (Exp.litU32 headsPerKV)
  let dimsPerLane := headDim / 32

  let _q ← ShaderM.declareInputBuffer "q" (.array (.scalar .f32) (numHeads * headDim))
  let _kCache ← ShaderM.declareInputBuffer "k_cache" (.array (.scalar .f32) (numKVHeads * maxSeqLen * headDim))
  let _vCache ← ShaderM.declareInputBuffer "v_cache" (.array (.scalar .f32) (numKVHeads * maxSeqLen * headDim))
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) (numHeads * headDim))

  let qHeadBase := Exp.mul head (Exp.litU32 headDim)
  let kvHeadBase := Exp.mul (Exp.mul kvHead (Exp.litU32 maxSeqLen)) (Exp.litU32 headDim)

  for slice in [0:dimsPerLane] do
    let sliceOff := Exp.add tid (Exp.litU32 (slice * 32))
    let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := numHeads * headDim) "q"
              (Exp.add qHeadBase sliceOff)
    ShaderM.varNamed s!"q{slice}" (.scalar .f32) v

  for slice in [0:dimsPerLane] do
    ShaderM.varNamed s!"o{slice}" (.scalar .f32) (Exp.litF32 0.0)

  ShaderM.varNamed "max_score" (.scalar .f32) (Exp.litF32 (-1.0e30))
  ShaderM.varNamed "sum_exp" (.scalar .f32) (Exp.litF32 0.0)
  let maxScore : Exp (.scalar .f32) := Exp.var "max_score"
  let sumExp : Exp (.scalar .f32) := Exp.var "sum_exp"

  -- Sliding window: only positions [windowStart, currentPos] contribute.
  let windowStart := if currentPos + 1 ≥ windowSize then currentPos + 1 - windowSize else 0

  ShaderM.loop (Exp.litU32 0) (Exp.litU32 cacheLen) (Exp.litU32 1) fun s => do
    let posBase := Exp.add kvHeadBase (Exp.mul s (Exp.litU32 headDim))
    let inWindow := Exp.ge s (Exp.litU32 windowStart)

    ShaderM.varNamed "partialDot" (.scalar .f32) (Exp.litF32 0.0)
    let partialDot : Exp (.scalar .f32) := Exp.var "partialDot"
    for slice in [0:dimsPerLane] do
      let sliceOff := Exp.add tid (Exp.litU32 (slice * 32))
      let kVal ← ShaderM.readBuffer (ty := .scalar .f32)
                  (n := numKVHeads * maxSeqLen * headDim) "k_cache"
                  (Exp.add posBase sliceOff)
      let qReg : Exp (.scalar .f32) := Exp.var s!"q{slice}"
      ShaderM.assign "partialDot" (Exp.add partialDot (Exp.mul qReg kVal))

    -- Same materialisation trick as the full-attention kernel: name
    -- every intermediate so the DSL emits each exactly once.
    ShaderM.varNamed "dot" (.scalar .f32) (Exp.subgroupAdd partialDot)
    let dot : Exp (.scalar .f32) := Exp.var "dot"
    ShaderM.varNamed "score" (.scalar .f32)
      (Exp.select inWindow (Exp.mul (Exp.litF32 scale) dot) (Exp.litF32 (-1.0e30)))
    let score : Exp (.scalar .f32) := Exp.var "score"

    ShaderM.varNamed "newMax" (.scalar .f32) (Exp.max maxScore score)
    let newMax : Exp (.scalar .f32) := Exp.var "newMax"
    ShaderM.varNamed "expOld" (.scalar .f32) (Exp.exp (Exp.sub maxScore newMax))
    let expOld : Exp (.scalar .f32) := Exp.var "expOld"
    ShaderM.varNamed "expNew" (.scalar .f32) (Exp.exp (Exp.sub score newMax))
    let expNew : Exp (.scalar .f32) := Exp.var "expNew"
    ShaderM.varNamed "newSum" (.scalar .f32) (Exp.add (Exp.mul sumExp expOld) expNew)
    let newSum : Exp (.scalar .f32) := Exp.var "newSum"
    ShaderM.varNamed "rescaleFactor" (.scalar .f32) (Exp.div (Exp.mul sumExp expOld) newSum)
    let rescaleFactor : Exp (.scalar .f32) := Exp.var "rescaleFactor"
    ShaderM.varNamed "contribFactor" (.scalar .f32) (Exp.div expNew newSum)
    let contribFactor : Exp (.scalar .f32) := Exp.var "contribFactor"

    for slice in [0:dimsPerLane] do
      let sliceOff := Exp.add tid (Exp.litU32 (slice * 32))
      let vVal ← ShaderM.readBuffer (ty := .scalar .f32)
                  (n := numKVHeads * maxSeqLen * headDim) "v_cache"
                  (Exp.add posBase sliceOff)
      let oReg : Exp (.scalar .f32) := Exp.var s!"o{slice}"
      ShaderM.assign s!"o{slice}"
        (Exp.add (Exp.mul oReg rescaleFactor) (Exp.mul vVal contribFactor))

    ShaderM.assign "max_score" newMax
    ShaderM.assign "sum_exp" newSum

  for slice in [0:dimsPerLane] do
    let sliceOff := Exp.add tid (Exp.litU32 (slice * 32))
    let outIdx := Exp.add qHeadBase sliceOff
    let oReg : Exp (.scalar .f32) := Exp.var s!"o{slice}"
    ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx oReg

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
    Same WGSL source for all cacheLen → 100% pipeline cache hit rate.
    Takes an optional `cacheRef` so callers can share a CachedDispatch
    across decode steps; when `none`, falls back to the throwaway-ref
    anti-pattern (for first-call/prefill sites). -/
def executeFlashAttentionWithParams [GPUBackend β] (ctx : β)
    (qBuf kCacheBuf vCacheBuf paramsBuf outputBuf : GPUBackend.Buf β)
    (numHeads numKVHeads maxSeqLen headDim : Nat) (scale : Float)
    (cacheRef : Option (IO.Ref (Option (GPUBackend.CachedDispatch β))) := none)
    : IO Unit := do
  let workgroupSize := min 256 (max headDim 32)
  let shader := flashAttentionParamsKernel numHeads numKVHeads maxSeqLen headDim scale workgroupSize
  let namedBuffers := [("q_output", outputBuf), ("k_cache", kCacheBuf), ("v_cache", vCacheBuf), ("params", paramsBuf)]
  -- Static cache key: same WGSL for all cacheLen (cacheLen is read from params buffer)
  let cacheKey : UInt64 := hash ("flashP", numHeads, numKVHeads, maxSeqLen, headDim)
  let execConfig : Hesper.ExecConfig := {
    workgroupSize := {x := workgroupSize, y := 1, z := 1}
    numWorkgroups := (numHeads, 1, 1)
    extensions := ["subgroups"]
    funcName := "flashAttentionWithParams"
    -- No diagnostic needed: params is var<storage, read> which is uniform
  }
  let ref ← match cacheRef with
    | some r => pure r
    | none   => IO.mkRef none
  GPUBackend.executeWithConfigCached ctx shader namedBuffers execConfig cacheKey ref

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
def createFlashPartialBuffer [GPUBackend β] (ctx : β) (numHeads maxSeqLen headDim : Nat)
    (tileSize : Nat := 32) : IO (GPUBackend.Buf β) := do
  let maxTiles := (maxSeqLen + tileSize - 1) / tileSize
  let stride := headDim + 2
  let partialSize := numHeads * maxTiles * stride
  GPUBackend.allocBuffer ctx (partialSize * 4).toUSize

/-- Execute tiled flash attention (2 phases).

    The `phase1Ref` / `phase2Ref` parameters are the per-call dispatch
    cache slots.  Earlier code synthesised them via `IO.mkRef none`
    inside the function body when callers passed `none` — that turned
    every invocation into a cold cudaExecuteImpl miss because the ref
    was thrown away at end of call.  Doc 57 §3b.4 traced the 4 ms
    graphs-OFF host-time gap to that pattern.

    Now, when callers pass `none`, we provide a `kcrLookup` callback
    that resolves a (cacheKey → IO.Ref) for refs *that survive across
    calls* — typically routed through `KernelCacheRefs` from the model
    forward.  As long as the same `cacheLen` recurs across forward
    invocations (and within a forward, all layers share one `cacheLen`)
    the dispatches hit the cache.  Pass `kcrLookup := none` only if you
    want the legacy throwaway behaviour (e.g. a one-shot test). -/
def executeFlashAttentionTiled [GPUBackend β] (ctx : β)
    (qBuf kCacheBuf vCacheBuf outputBuf : GPUBackend.Buf β)
    (numHeads numKVHeads maxSeqLen headDim cacheLen : Nat) (scale : Float)
    (partialBuf : Option (GPUBackend.Buf β) := none)
    (phase1Ref phase2Ref :
       Option (IO.Ref (Option (GPUBackend.CachedDispatch β))) := none)
    (kcrLookup :
       Option (UInt64 → IO (IO.Ref (Option (GPUBackend.CachedDispatch β)))) := none)
    : IO Unit := do
  let tileSize := 32
  let numTiles := (cacheLen + tileSize - 1) / tileSize
  let workgroupSize := min 256 (max headDim 32)

  -- Use pre-allocated buffer or allocate (fallback for compatibility)
  let partialBuf ← match partialBuf with
    | some buf => pure buf
    | none => do
      let stride := headDim + 2
      let partialSize := numHeads * numTiles * stride
      GPUBackend.allocBuffer ctx (partialSize * 4).toUSize

  -- Always use Phase 1 + Phase 2. The old `numTiles == 1` in-place
  -- shortcut assumed Q and output shared the same buffer, which is NOT
  -- the case in Gemma 4 (Q is in qBuf, output is in attnOutBuf).
  do
    -- Multi-tile: Phase 1 (parallel tiles) + Phase 2 (merge)
    let shader1 := flashAttentionTiledPhase1 numHeads numKVHeads maxSeqLen headDim cacheLen tileSize scale workgroupSize
    let namedBuffers1 := [("q", qBuf), ("k_cache", kCacheBuf), ("v_cache", vCacheBuf), ("partial", partialBuf)]
    let execConfig1 : Hesper.ExecConfig := {
      workgroupSize := {x := workgroupSize, y := 1, z := 1}
      numWorkgroups := (numHeads, numTiles, 1)
      extensions := ["subgroups"]
    }
    let cacheKey1 : UInt64 := hash ("flashT1", numHeads, numKVHeads, maxSeqLen, headDim, cacheLen, tileSize)
    let ref1 ← match phase1Ref with
      | some r => pure r
      | none   =>
        match kcrLookup with
        | some lk => lk cacheKey1
        | none    => IO.mkRef none  -- legacy throwaway path; opt-out only
    GPUBackend.executeWithConfigCached ctx shader1 namedBuffers1 execConfig1 cacheKey1 ref1

    let shader2 := flashAttentionTiledPhase2 numHeads headDim numTiles
    let namedBuffers2 := [("partial", partialBuf), ("output", outputBuf)]
    let execConfig2 := Hesper.ExecConfig.dispatch1D (numHeads * headDim) 256
    let cacheKey2 : UInt64 := hash ("flashT2", numHeads, headDim, numTiles)
    let ref2 ← match phase2Ref with
      | some r => pure r
      | none   =>
        match kcrLookup with
        | some lk => lk cacheKey2
        | none    => IO.mkRef none
    GPUBackend.executeWithConfigCached ctx shader2 namedBuffers2 execConfig2 cacheKey2 ref2

/-- doc 60 Session 1: launcher for flashAttentionVecParamsKernel.

    Same call shape as `executeFlashAttentionDynamic` — single-token Q,
    cacheLen lives in `state.paramsBuf[1]` (so the PTX is fully cacheable
    and CUDA-Graph friendly).  The only difference vs the legacy launcher
    is the workgroup size (128 instead of `min 256 headDim`) and the
    use of the new shader. -/
def executeFlashAttentionVecParams [GPUBackend β] (ctx : β)
    (qBuf kCacheBuf vCacheBuf outputBuf paramsBuf : GPUBackend.Buf β)
    (numHeads numKVHeads maxSeqLen headDim : Nat) (scale : Float)
    (kcrLookup :
       Option (UInt64 → IO (IO.Ref (Option (GPUBackend.CachedDispatch β)))) := none)
    : IO Unit := do
  let workgroupSize := 128
  let shader := flashAttentionVecParamsKernel numHeads numKVHeads maxSeqLen headDim scale
  let namedBuffers :=
    [ ("q",       qBuf)
    , ("k_cache", kCacheBuf)
    , ("v_cache", vCacheBuf)
    , ("output",  outputBuf)
    , ("params",  paramsBuf) ]
  let cacheKey : UInt64 := hash ("flashVecParams", numHeads, numKVHeads, maxSeqLen, headDim)
  let execConfig : Hesper.ExecConfig := {
    workgroupSize := { x := workgroupSize, y := 1, z := 1 }
    numWorkgroups := (numHeads, 1, 1)
    extensions := ["subgroups"]
  }
  let ref ← match kcrLookup with
    | some lk => lk cacheKey
    | none    => IO.mkRef none
  GPUBackend.executeWithConfigCached ctx shader namedBuffers execConfig cacheKey ref

def executeFlashAttentionDynamic [GPUBackend β] (ctx : β)
    (qBuf kCacheBuf vCacheBuf outputBuf : GPUBackend.Buf β)
    (numHeads numKVHeads maxSeqLen headDim cacheLen : Nat) (scale : Float)
    (kcrLookup :
       Option (UInt64 → IO (IO.Ref (Option (GPUBackend.CachedDispatch β)))) := none)
    : IO Unit := do
  let workgroupSize := min 256 (max headDim 32)
  let shader := flashAttentionDynamicKernel numHeads numKVHeads maxSeqLen headDim cacheLen scale workgroupSize
  let namedBuffers := [("q", qBuf), ("k_cache", kCacheBuf), ("v_cache", vCacheBuf), ("output", outputBuf)]
  -- Pipeline cache key includes cacheLen (shader recompiled per position)
  -- The WGSL source hash + buffer layout is cached, so same cacheLen reuses pipeline
  let cacheKey : UInt64 := hash ("flash", numHeads, numKVHeads, maxSeqLen, headDim, cacheLen)
  let execConfig : Hesper.ExecConfig := {
    workgroupSize := {x := workgroupSize, y := 1, z := 1}
    numWorkgroups := (numHeads, 1, 1)
    extensions := ["subgroups"]
  }
  -- Note: shader compilation is cached by pipeline cache (Execute.lean).
  -- First call with a new cacheLen compiles, subsequent calls with same cacheLen reuse.
  -- Over a training run, common cacheLens are cached and recompilation is rare.
  let ref ← match kcrLookup with
    | some lk => lk cacheKey
    | none    => IO.mkRef none
  GPUBackend.executeWithConfigCached ctx shader namedBuffers execConfig cacheKey ref

/-- Execute flash attention with static cacheLen (for testing).
    Uses dynamic kernel with a params buffer containing cacheLen. -/
def executeFlashAttention [GPUBackend β] (ctx : β)
    (qBuf kCacheBuf vCacheBuf outputBuf : GPUBackend.Buf β)
    (numHeads numKVHeads cacheLen headDim : Nat) (scale : Float) : IO Unit := do
  -- For testing: maxSeqLen = cacheLen (buffer sizes match exactly)
  executeFlashAttentionDynamic ctx qBuf kCacheBuf vCacheBuf outputBuf
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
