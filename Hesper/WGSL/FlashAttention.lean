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
def flashAttentionKernel (numHeads numKVHeads cacheLen headDim : Nat)
    (scale : Float) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let wgid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let head := Exp.vec3X wgid   -- head index
  let tid := Exp.vec3X lid      -- thread within workgroup

  let headsPerKV := numHeads / numKVHeads
  let kvHead := Exp.div head (Exp.litU32 headsPerKV)

  let _q ← ShaderM.declareInputBuffer "q" (.array (.scalar .f32) (numHeads * headDim))
  let _kCache ← ShaderM.declareInputBuffer "k_cache" (.array (.scalar .f32) (numKVHeads * cacheLen * headDim))
  let _vCache ← ShaderM.declareInputBuffer "v_cache" (.array (.scalar .f32) (numKVHeads * cacheLen * headDim))
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) (numHeads * headDim))

  -- Shared memory for partial score reduction and Q cache
  ShaderM.sharedNamed "shared_q" (.array (.scalar .f32) headDim)
  ShaderM.sharedNamed "shared_reduce" (.array (.scalar .f32) workgroupSize)

  ShaderM.if_ (Exp.lt head (Exp.litU32 numHeads)) (do
    -- Step 1: Load Q for this head into shared memory
    let qBase := Exp.mul head (Exp.litU32 headDim)
    ShaderM.loop tid (Exp.litU32 headDim) (Exp.litU32 workgroupSize) fun d => do
      let qVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numHeads * headDim) "q" (Exp.add qBase d)
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_q" d qVal
    ShaderM.barrier

    -- Step 2: Online softmax over cached positions
    -- Each thread maintains partial accumulator for a subset of headDim
    -- Thread tid accumulates output[tid], output[tid+workgroupSize], etc.

    -- Online softmax state (per-thread, but shared via reduction for score computation)
    ShaderM.varNamed "max_score" (.scalar .f32) (Exp.litF32 (-1.0e30))
    ShaderM.varNamed "sum_exp" (.scalar .f32) (Exp.litF32 0.0)
    let maxScore := Exp.var "max_score"
    let sumExp := Exp.var "sum_exp"

    -- Output accumulator (per-thread dimension elements)
    -- Each thread handles dimensions tid, tid+workgroupSize, ...
    -- For simplicity with headDim <= workgroupSize, each thread handles 1 dim
    ShaderM.varNamed "out_acc" (.scalar .f32) (Exp.litF32 0.0)
    let outAcc := Exp.var "out_acc"

    -- Iterate over cached positions
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 cacheLen) (Exp.litU32 1) fun s => do
      -- Compute score = scale * Q · K[s]
      -- Each thread computes partial dot product, then reduce
      let kBase := Exp.add (Exp.mul (Exp.mul kvHead (Exp.litU32 cacheLen)) (Exp.litU32 headDim))
                            (Exp.mul s (Exp.litU32 headDim))

      let partialVar ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
      ShaderM.loop tid (Exp.litU32 headDim) (Exp.litU32 workgroupSize) fun d => do
        let qVal ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := headDim) "shared_q" d
        let kVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numKVHeads * cacheLen * headDim) "k_cache" (Exp.add kBase d)
        ShaderM.assign partialVar (Exp.add (Exp.var partialVar) (Exp.mul qVal kVal))

      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_reduce" tid (Exp.var partialVar)
      ShaderM.barrier

      -- Tree reduction for score
      let numSteps := Nat.log2 workgroupSize
      ShaderM.staticLoop numSteps fun step => do
        let stride := workgroupSize >>> (step + 1)
        ShaderM.if_ (Exp.lt tid (Exp.litU32 stride)) (do
          let other ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_reduce" (Exp.add tid (Exp.litU32 stride))
          let cur ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_reduce" tid
          ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_reduce" tid (Exp.add cur other)
        ) (pure ())
        ShaderM.barrier

      -- Thread 0 broadcasts score to shared memory slot 0
      -- All threads read the score
      -- All threads read the reduced score from shared memory
      let scoreFromShared ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_reduce" (Exp.litU32 0)
      let scaledScore := Exp.mul (Exp.litF32 scale) scoreFromShared
      let newMax := Exp.max maxScore scaledScore
      let expOld := Exp.exp (Exp.sub maxScore newMax)
      let expNew := Exp.exp (Exp.sub scaledScore newMax)
      let newSum := Exp.add (Exp.mul sumExp expOld) expNew

      -- Update output accumulator for this thread's dimension(s)
      ShaderM.if_ (Exp.lt tid (Exp.litU32 headDim)) (do
        let vIdx := Exp.add kBase tid  -- V uses same layout as K
        let vVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numKVHeads * cacheLen * headDim) "v_cache" vIdx
        -- Rescale old accumulator and add new weighted V
        let rescaled := Exp.mul outAcc (Exp.div (Exp.mul sumExp expOld) newSum)
        let newContrib := Exp.mul vVal (Exp.div expNew newSum)
        ShaderM.assign "out_acc" (Exp.add rescaled newContrib)
      ) (pure ())

      ShaderM.assign "max_score" newMax
      ShaderM.assign "sum_exp" newSum
      ShaderM.barrier

    -- Step 3: Write output
    ShaderM.if_ (Exp.lt tid (Exp.litU32 headDim)) (do
      let outIdx := Exp.add (Exp.mul head (Exp.litU32 headDim)) tid
      ShaderM.writeBuffer (ty := .scalar .f32) "output" outIdx outAcc
    ) (pure ())
  ) (pure ())

/-- Execute flash attention forward for single-token KV cache query -/
def executeFlashAttention (device : Device)
    (qBuf kCacheBuf vCacheBuf outputBuf : Buffer)
    (numHeads numKVHeads cacheLen headDim : Nat) (scale : Float) : IO Unit := do
  let workgroupSize := min 256 (max headDim 32)  -- at least headDim threads
  let shader := flashAttentionKernel numHeads numKVHeads cacheLen headDim scale workgroupSize
  let namedBuffers := [("q", qBuf), ("k_cache", kCacheBuf), ("v_cache", vCacheBuf), ("output", outputBuf)]
  let execConfig : Hesper.WGSL.Execute.ExecutionConfig := {
    workgroupSize := {x := workgroupSize, y := 1, z := 1}
    numWorkgroups := (numHeads, 1, 1)  -- 1 workgroup per head
  }
  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig

end Hesper.WGSL.FlashAttention
