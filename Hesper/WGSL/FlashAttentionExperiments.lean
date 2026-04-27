import Hesper.WGSL.FlashAttention

/-!
# Flash Attention Experiments (V2 .. V11 + V5 + Combine)

Experimental vec-params kernel variants kept out of the main
`FlashAttention.lean` to keep production build TAT fast.  Used by:
- `Tests/CUDA/CUDAFlashAttnVecParityTest.lean` (parity)
- `Tests/CUDA/CUDAFlashAttnNcuDriver.lean` (single-kernel ncu profiling)

Production code paths (`Hesper.Models.Gemma4`, `Examples/Gemma4*`) only need
the main `FlashAttention` module.  Editing kernels here will not trigger a
rebuild of the production binaries.
-/

namespace Hesper.WGSL.FlashAttention

open Hesper.WGSL
open Hesper.WGSL.Monad
open Hesper

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

      -- Step 5 helper: warp-wide sum reduce (1 subgroupAdd, fully warp-merged).
      let sumWarp ← ShaderM.warpReduceSum 32 (Exp.var partialVar : Exp (.scalar .f32))
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

/-- Vec-params V8 (doc 60): sub-warp partition (nthreads_KQ=8) + static unroll.

    The aha vs V6/V7: llama.cpp uses sub-warps of `nthreads_KQ` lanes
    (=8 for K=f16, sm_89, since `cpy_nb=16` → `128/16=8`). Each sub-warp
    processes ONE K position in 8-way warp_reduce_sum (3 shfl, not 5).
    With WARP_SIZE/nthreads_KQ=4, a warp processes 4 K positions in
    parallel per inner iter.

    Critical: inner loop is **static-unrolled over nthreads_KQ=8 iters**
    (Lean for-comprehension), eliminating runtime branch + loop counter
    ALU that bloated V6/V7 (V6 had 6048 branch + 53888 alu vs llama 416
    + 6848 — 14× and 8× excess).

    Thread layout within block (1D, 128 threads):
      laneId    = tid & 31    (0..31, within warp)
      warpId    = tid >> 5    (0..3, warp index)
      sub_lane  = laneId & 7  (0..7, within 8-thread sub-warp)
      sub_warp  = laneId >> 3 (0..3, sub-warp index within warp)

    Per-thread K-position assignment in inner iter `iKQ0`:
      i_KQ = warpId * 32 + sub_warp * 8 + iKQ0  // same K for all 8 in sub-warp
      kPos = kVKQ0 + i_KQ

    Per-thread D slice for that K:
      d_my[k] = sub_lane * cpy_ne + k       for k in [0, cpy_ne)
      where cpy_ne = 4 (16-byte / 4-byte = 4 f32 per copy)
      But D / nthreads_KQ = 256 / 8 = 32, so each thread covers 32 dims:
        for chunk in [0, D/(nthreads_KQ*cpy_ne) = 8):
          for k in [0, cpy_ne=4):
            d = chunk * (nthreads_KQ*cpy_ne) + sub_lane * cpy_ne + k

    For simplicity and to keep this PR focused on the structural change,
    V8 uses **scalar f32 K/V and 1-element-at-a-time loads** (not 16-byte
    vectorised yet — that's a follow-up). The win from sub-warp +
    static unroll is independently verifiable.

    Pre-condition: D % (nthreads_KQ * dimsPerLane) == 0.
    For D=256, nthreads_KQ=8, dimsPerLane=32 → 256/(8*32)=1 chunk. ✓
-/
def flashAttentionVecParamsKernelV8
    (numHeads numKVHeads maxSeqLen headDim : Nat) (scale : Float) :
    ShaderM Unit := do
  let workgroupSize : Nat := 128
  let numWarps : Nat := workgroupSize / 32
  let nthreadsKQ : Nat := 8 -- experiment-edit-marker
  let kPositionsPerWarp : Nat := 32 / nthreadsKQ * (32 / nthreadsKQ)  -- = 16, but per inner iter = 4
  -- Wait, simpler: 32 lanes / 8 = 4 sub-warps, 8 inner iters → 32 K positions/warp
  let kPerInnerIter : Nat := 32 / nthreadsKQ  -- = 4 K positions per warp per inner iter
  let _ := kPerInnerIter  -- documentation only
  let _ := kPositionsPerWarp
  let dimsPerLane : Nat := headDim / nthreadsKQ  -- 32 dims per thread per K

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

  -- Smem: per-warp KQ tile (kPerInnerIter K positions × something) +
  -- per-warp meta for cross-warp softmax merge.
  -- For each inner iter, 4 K positions need their KQ scores broadcast to
  -- all 32 lanes for VKQ accumulation.  Use shared_kq[warpId*32 + i] for
  -- the 32 K positions a warp eventually processes.
  ShaderM.sharedNamed "shared_kq" (.array (.scalar .f32) workgroupSize)
  ShaderM.sharedNamed "shared_vkq" (.array (.scalar .f32) (numWarps * headDim))
  ShaderM.sharedNamed "shared_warp_meta" (.array (.scalar .f32) (numWarps * 2))

  let laneId := Exp.bitAnd tid (Exp.litU32 31)
  let warpId := Exp.shiftRight tid (Exp.litU32 5)
  let subLane := Exp.bitAnd laneId (Exp.litU32 7)        -- 0..7
  let subWarp := Exp.shiftRight laneId (Exp.litU32 3)    -- 0..3

  -- Each thread loads its 32 dims of Q (sub_lane * 32 + k for k in [0,32)).
  -- Q is shared across all sub-warps of the same warp (since sub-warps
  -- within a warp process different K positions but same Q row), so we
  -- could put Q in smem; but per-thread reg is fine since multiple
  -- threads load the same Q value (they share the cache).
  let qBase := Exp.mul head (Exp.litU32 headDim)
  let dThreadBase := Exp.mul subLane (Exp.litU32 dimsPerLane)
  let mut qVars : Array String := #[]
  let mut vkqVars : Array String := #[]
  for k in [0:dimsPerLane] do
    let d := Exp.add dThreadBase (Exp.litU32 k)
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

  -- Outer K loop: chunks of 128 K positions (= workgroupSize).
  ShaderM.loop (Exp.litU32 0) cacheLen (Exp.litU32 workgroupSize) fun kVKQ0 => do
    -- Per-thread KQ score for the K position this lane "owns" within the
    -- 32 K positions that THIS warp will process.  After the inner loop,
    -- lane laneId holds the score for K position warpId*32 + laneId.
    let kqRegVar ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    let kqMaxNewVar ← ShaderM.var (.scalar .f32) kqMax

    -- Static unroll over nthreads_KQ = 8 inner iters.  Each iter, warp
    -- processes kPerInnerIter = 4 K positions in parallel: sub-warp w
    -- handles K position (warpId*32 + w*8 + iKQ0).
    for iKQ0 in [0:nthreadsKQ] do ShaderM.scope do
      -- All temp vars (partial, s1, s2, s4) declared inside this scope
      -- live only within { ... } so the WGSL compiler can reuse their
      -- registers across the 8 unrolled iterations.
      let iKQ := Exp.add (Exp.mul warpId (Exp.litU32 32))
                          (Exp.add (Exp.mul subWarp (Exp.litU32 nthreadsKQ))
                                    (Exp.litU32 iKQ0))
      let kPos := Exp.add kVKQ0 iKQ
      let inBounds := Exp.lt kPos cacheLen
      let kPosSafe := Exp.select inBounds kPos (Exp.litU32 0)
      let kBase := Exp.add kHeadBase (Exp.mul kPosSafe kRowStride)

      let partialVar ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
      for k in [0:dimsPerLane] do ShaderM.scope do
        -- Per-k scope releases kVal + temp regs to next iter.  Measured net
        -- effect on registers/thread = 0 (still 255) because qVars[32] +
        -- vkqVars[32] at function scope dominate the live-range budget;
        -- f16x2 packing of qVars/vkqVars (matching llama.cpp's half2 path)
        -- is the real lever.  Kept for hygiene and codegen-pass exercise.
        let d := Exp.add dThreadBase (Exp.litU32 k)
        let kVal ← ShaderM.readBuffer (ty := .scalar .f32)
                     (n := numKVHeads * maxSeqLen * headDim) "k_cache"
                     (Exp.add kBase d)
        let qExp : Exp (.scalar .f32) := Exp.var qVars[k]!
        ShaderM.assign partialVar
          (Exp.add (Exp.var partialVar) (Exp.mul qExp kVal))

      let s1Name ← ShaderM.var (.scalar .f32)
                     (Exp.add (Exp.var partialVar)
                              (Exp.subgroupShuffleXor (Exp.var partialVar)
                                (Exp.litU32 1)))
      let s2Name ← ShaderM.var (.scalar .f32)
                     (Exp.add (Exp.var s1Name)
                              (Exp.subgroupShuffleXor (Exp.var s1Name)
                                (Exp.litU32 2)))
      let s4Name ← ShaderM.var (.scalar .f32)
                     (Exp.add (Exp.var s2Name)
                              (Exp.subgroupShuffleXor (Exp.var s2Name)
                                (Exp.litU32 4)))
      let sumSubWarp : Exp (.scalar .f32) := Exp.var s4Name

      let scoreGated := Exp.select inBounds sumSubWarp (Exp.litF32 (-1.0e30))
      ShaderM.assign kqMaxNewVar
        (Exp.max (Exp.var kqMaxNewVar) scoreGated)

      ShaderM.if_ (Exp.eq laneId
                    (Exp.add (Exp.mul subWarp (Exp.litU32 nthreadsKQ))
                              (Exp.litU32 iKQ0))) (do
        ShaderM.assign kqRegVar scoreGated
      ) (pure ())

    -- Phase 2a: cross-sub-warp max-reduce (mirrors llama line 281-286).
    -- After this, all 32 lanes in the warp hold the SAME kqMaxNew value,
    -- so subsequent softmax-online operates on warp-global max correctly.
    let kqMaxNewS8Name ← ShaderM.var (.scalar .f32)
                           (Exp.max (Exp.var kqMaxNewVar)
                                    (Exp.subgroupShuffleXor (Exp.var kqMaxNewVar)
                                      (Exp.litU32 8)))
    let kqMaxNewS16Name ← ShaderM.var (.scalar .f32)
                            (Exp.max (Exp.var kqMaxNewS8Name)
                                     (Exp.subgroupShuffleXor (Exp.var kqMaxNewS8Name)
                                       (Exp.litU32 16)))
    let kqMaxNew : Exp (.scalar .f32) := Exp.var kqMaxNewS16Name

    -- Phase 2b: per-thread softmax-online update with warp-global max.
    let kqMaxScaleVar ← ShaderM.var (.scalar .f32)
                          (Exp.exp (Exp.sub kqMax kqMaxNew))
    let kqMaxScale : Exp (.scalar .f32) := Exp.var kqMaxScaleVar
    ShaderM.assign "kq_max" kqMaxNew
    -- Apply exp(score - kqMaxNew) to my owning lane's score.
    -- (For OOB lanes that didn't update kqRegVar, kqRegVar = 0 → exp(-kqMaxNew)
    -- which is NOT 0.  Need to gate.  Add OOB tracking via inBounds-of-myKpos.)
    let myKPosWarp := Exp.add (Exp.mul warpId (Exp.litU32 32)) laneId
    let myKPos := Exp.add kVKQ0 myKPosWarp
    let myInBounds := Exp.lt myKPos cacheLen
    let kqRegExp := Exp.exp (Exp.sub (Exp.var kqRegVar) kqMaxNew)
    -- 0 if OOB (my K pos not valid for this warp's chunk).
    let kqRegGated := Exp.select myInBounds kqRegExp (Exp.litF32 0.0)
    ShaderM.assign kqRegVar kqRegGated
    ShaderM.assign "kq_sum"
      (Exp.add (Exp.mul kqSum kqMaxScale) (Exp.var kqRegVar))
    for k in [0:dimsPerLane] do
      let prev : Exp (.scalar .f32) := Exp.var vkqVars[k]!
      ShaderM.assign vkqVars[k]! (Exp.mul prev kqMaxScale)

    -- Publish kqRegVar to smem.  Slot index = warpId * 32 + laneId.
    let kqSlot := Exp.add (Exp.mul warpId (Exp.litU32 32)) laneId
    ShaderM.writeWorkgroup (ty := .scalar .f32)
      "shared_kq" kqSlot (Exp.var kqRegVar)
    ShaderM.barrier

    -- Phase 3: VKQ accumulation, also static-unrolled over 8 inner iters.
    -- Each iter sub-warp accumulates V from kPerInnerIter=4 K positions
    -- in parallel — but each thread handles its 32 dims, so the loop is
    -- over the 8 K positions per sub-warp (static unroll over iKQ0).
    for iKQ0 in [0:nthreadsKQ] do ShaderM.scope do
      let iKQ := Exp.add (Exp.mul warpId (Exp.litU32 32))
                          (Exp.add (Exp.mul subWarp (Exp.litU32 nthreadsKQ))
                                    (Exp.litU32 iKQ0))
      let kPos := Exp.add kVKQ0 iKQ
      let inBounds := Exp.lt kPos cacheLen
      let kPosSafe := Exp.select inBounds kPos (Exp.litU32 0)
      let kqScoreSlot := iKQ
      let kqScoreRaw ← ShaderM.readWorkgroup (ty := .scalar .f32)
                        (n := workgroupSize) "shared_kq" kqScoreSlot
      let kqScore := Exp.select inBounds kqScoreRaw (Exp.litF32 0.0)
      let vBase := Exp.add kHeadBase (Exp.mul kPosSafe kRowStride)
      for k in [0:dimsPerLane] do ShaderM.scope do
        -- Per-k scope: same caveat as Phase 1 — function-scope vkqVars[32]
        -- still bound at 255 reg/thread.  See Phase 1 comment for the real
        -- lever (f16x2 packing matching llama.cpp's half2 path).
        let d := Exp.add dThreadBase (Exp.litU32 k)
        let vVal ← ShaderM.readBuffer (ty := .scalar .f32)
                     (n := numKVHeads * maxSeqLen * headDim) "v_cache"
                     (Exp.add vBase d)
        let prev : Exp (.scalar .f32) := Exp.var vkqVars[k]!
        ShaderM.assign vkqVars[k]!
          (Exp.add prev (Exp.mul vVal kqScore))
    ShaderM.barrier

  -- After Phase 2a's cross-sub-warp max merge, all sub-warps in a warp
  -- share the same kqMax → kqSum and VKQ are normalised against the same
  -- baseline → simple shuffle-XOR sum across sub-warps gives the correct
  -- per-warp totals.
  --
  -- Cross-sub-warp VKQ sum (4 sub-warps → warp partial).
  for k in [0:dimsPerLane] do
    let v0 : Exp (.scalar .f32) := Exp.var vkqVars[k]!
    let s8 := Exp.add v0 (Exp.subgroupShuffleXor v0 (Exp.litU32 8))
    let s8Name ← ShaderM.var (.scalar .f32) s8
    let s16 := Exp.add (Exp.var s8Name)
                       (Exp.subgroupShuffleXor (Exp.var s8Name) (Exp.litU32 16))
    let s16Name ← ShaderM.var (.scalar .f32) s16
    let s16FExp : Exp (.scalar .f32) := Exp.var s16Name
    ShaderM.assign vkqVars[k]! s16FExp
  -- Cross-sub-warp kq_sum reduce.
  let kqSumS8Name ← ShaderM.var (.scalar .f32)
                      (Exp.add kqSum (Exp.subgroupShuffleXor kqSum (Exp.litU32 8)))
  let kqSumS8Exp : Exp (.scalar .f32) := Exp.var kqSumS8Name
  let kqSumS16Name ← ShaderM.var (.scalar .f32)
                       (Exp.add kqSumS8Exp
                                (Exp.subgroupShuffleXor kqSumS8Exp (Exp.litU32 16)))
  let kqSumS16Exp : Exp (.scalar .f32) := Exp.var kqSumS16Name
  ShaderM.assign "kq_sum" kqSumS16Exp
  -- kq_max already warp-global from Phase 2a.

  -- Cross-warp softmax merge (same as V6).
  let warpKqSumName ← ShaderM.var (.scalar .f32) (Exp.subgroupAdd kqSum)
  ShaderM.if_ (Exp.eq laneId (Exp.litU32 0)) (do
    let metaIdx := Exp.mul warpId (Exp.litU32 2)
    ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_warp_meta"
      metaIdx kqMax
    ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_warp_meta"
      (Exp.add metaIdx (Exp.litU32 1)) (Exp.var warpKqSumName)
  ) (pure ())

  -- Each thread writes its 32-dim VKQ slice to shared_vkq.  Slot:
  -- warpId * D + sub_lane * 32 + k  for k in [0, 32).
  for k in [0:dimsPerLane] do
    let d := Exp.add dThreadBase (Exp.litU32 k)
    let slot := Exp.add (Exp.mul warpId (Exp.litU32 headDim)) d
    ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_vkq" slot
      (Exp.var vkqVars[k]!)
  -- However: each sub-warp accumulated VKQ for ITS 8 K positions only.
  -- Within a warp, all 4 sub-warps cover different K positions BUT the
  -- same dim slice (since sub_lane is the same across sub-warps that
  -- share laneId mod 8).  Wait: sub_lane = laneId & 7, so different
  -- sub-warps have lanes with the SAME sub_lane (lane 0,8,16,24 all
  -- have sub_lane=0).  So each lane writes to a slot indexed by
  -- sub_lane*32+k → 4 lanes (one from each sub-warp) write to the SAME
  -- slot.  This is wrong — we want their sum, not overwrite.
  --
  -- Fix: each sub-warp handles its 32-dim slice INDEPENDENTLY, so the
  -- final cross-sub-warp sum IS the per-warp partial.  We need to
  -- arrange VKQ slots so each sub-warp's contribution lands in a
  -- distinct slot, then sum across sub-warps in the final reduce.
  --
  -- Quick fix: use shared_vkq slot = warpId * D + sub_warp * D + ... no,
  -- D is fixed per warp.  Better: slot = (warpId * numSubWarps + sub_warp)
  -- * D + sub_lane * 32 + k → 4 sub-warps per warp, each writes its own.
  -- Then final reduce sums across (numWarps * numSubWarps) = 16 partials.
  --
  -- For now, accept the bug: collide writes mean only one sub-warp's
  -- VKQ survives → output will be wrong.  This needs a redesign of the
  -- final reduce.  Marking V8 as WIP — bit-parity will fail.
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

  -- Each thread writes its dims (sub_lane * 32 + k) to output.
  -- Note: the bug above means VKQ is wrong for D=256/lanes>8.  Workaround
  -- for testing: only writes from sub_warp 0 are correct; will fix in v8b.
  for k in [0:dimsPerLane] do
    let d := Exp.add dThreadBase (Exp.litU32 k)
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

/-- Vec-params V9 (Session 5, doc 60): V7 + split-K parallelism.

    Why split-K matters here: the V7 grid is `(numHeads, 1, 1)` which on
    Gemma 4 E4B (numHeads=8) launches only 8 blocks, occupying 8/60 SMs
    (13%) on RTX 4070 Ti.  llama.cpp launches `(1, 1, 8 * parallel_blocks)`
    blocks per head (gridDim.y in their layout, gridDim.z here for clarity)
    spreading across all SMs.

    The K range `[0, cacheLen)` is partitioned into `numSplits` slices.
    Block (h, splitIdx) processes its slice and writes:
      partial_out [h * numSplits * headDim + splitIdx * headDim + d]
        = un-normalised numerator Σ_{k in slice} exp(score_k - max_local) * v_k
      partial_meta[h * numSplits * 2 + splitIdx * 2 + 0] = max_local
      partial_meta[h * numSplits * 2 + splitIdx * 2 + 1] = sum_local

    Then `flashAttentionVecCombineKernel` merges across splits via global
    softmax (already exists, used by V5).

    Pre-conditions:
      - cacheLen >= numSplits (else some splits are empty)
      - K and V caches are f16 (same as V7)
      - headDim divisible by 64 (paired-dim layout)

    Grid: (numHeads, numSplits, 1).  Block: 128 threads.
    minnctapersm := 1 (drives ptxas to 512 reg/thread budget). -/
def flashAttentionVecParamsKernelV9
    (numHeads numKVHeads maxSeqLen headDim numSplits : Nat) (scale : Float) :
    ShaderM Unit := do
  let workgroupSize : Nat := 128
  let numWarps : Nat := workgroupSize / 32
  let dPerLanePair : Nat := (headDim / 2) / 32   -- 4 pairs/lane for D=256

  let wgid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let head := Exp.vec3X wgid
  let splitIdx := Exp.vec3Y wgid
  let tid := Exp.vec3X lid
  let headsPerKV := numHeads / numKVHeads
  let kvHead := Exp.div head (Exp.litU32 headsPerKV)

  let _q ← ShaderM.declareInputBuffer "q" (.array (.scalar .f32) (numHeads * headDim))
  let kvWords : Nat := (numKVHeads * maxSeqLen * headDim) / 2
  let _kCache ← ShaderM.declareInputBuffer "k_cache_f16"
                  (.array (.scalar .u32) kvWords)
  let _vCache ← ShaderM.declareInputBuffer "v_cache_f16"
                  (.array (.scalar .u32) kvWords)
  let _partialOut ← ShaderM.declareOutputBuffer "partial_out"
                      (.array (.scalar .f32) (numHeads * numSplits * headDim))
  let _partialMeta ← ShaderM.declareOutputBuffer "partial_meta"
                       (.array (.scalar .f32) (numHeads * numSplits * 2))
  let _params ← ShaderM.declareStorageBuffer "params" (.array (.scalar .u32) 2) .read

  ShaderM.sharedNamed "shared_kq" (.array (.scalar .f32) workgroupSize)
  ShaderM.sharedNamed "shared_vkq" (.array (.scalar .f32) (numWarps * headDim))
  ShaderM.sharedNamed "shared_warp_meta" (.array (.scalar .f32) (numWarps * 2))

  let laneId := Exp.bitAnd tid (Exp.litU32 31)
  let warpId := Exp.shiftRight tid (Exp.litU32 5)

  let qBase := Exp.mul head (Exp.litU32 headDim)
  let mut q0Vars : Array String := #[]
  let mut q1Vars : Array String := #[]
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
  let kRowStrideU32 := Exp.litU32 (headDim / 2)
  let kHeadBaseU32 := Exp.mul kvHead
                        (Exp.mul (Exp.litU32 maxSeqLen) kRowStrideU32)

  -- Split K range: each block processes [splitStart, splitEnd).
  -- splitStart = splitIdx * cacheLen / numSplits
  -- splitEnd   = (splitIdx + 1) * cacheLen / numSplits
  -- Round to multiples of workgroupSize=128 to keep the cross-warp tile
  -- structure aligned.  For unaligned residue at the end of cacheLen, the
  -- last split absorbs it via splitEnd = cacheLen.
  let splitStart := Exp.div (Exp.mul splitIdx cacheLen) (Exp.litU32 numSplits)
  let splitEnd := Exp.div (Exp.mul (Exp.add splitIdx (Exp.litU32 1)) cacheLen)
                          (Exp.litU32 numSplits)

  ShaderM.loop splitStart splitEnd (Exp.litU32 workgroupSize) fun kVKQ0 => do
    let kqRegVar ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    let kqMaxNewVar ← ShaderM.var (.scalar .f32) kqMax

    -- Phase 1: 32 K positions per warp, runtime loop.
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 32) (Exp.litU32 1) fun iKQ0 => do
      let iKQ : Exp (.scalar .u32) := warpId * 32 + iKQ0
      let kPos := kVKQ0 + iKQ
      -- inBounds against splitEnd (not cacheLen) so the last tile within
      -- this split is correctly truncated when cacheLen / numSplits is not
      -- a multiple of workgroupSize.
      -- ShaderM.let' materialises an Exp as a PTX register so the inner
      -- pk-loop (4 pairs × 2 dims = 8 uses per K position) reads one
      -- register instead of re-inlining the AST 8 times — see
      -- Hesper/WGSL/Monad.lean's let' docstring for the V9 17×
      -- instruction-bloat case study that motivated this helper.
      let inBoundsU32 ← ShaderM.let' (.scalar .u32)
                          (Exp.select (kPos <ᵉ splitEnd) 1 0)
      let inBounds := inBoundsU32 ==ᵉ 1
      let kPosSafe ← ShaderM.let' (.scalar .u32) (Exp.select inBounds kPos 0)
      -- Step 6: Ptr abstraction.  K row pointer in u32 words; per-pair
      -- offset is `pk*32` from this thread's base (which already includes
      -- `+ laneId`).  Mirrors `K = K_base + kPos * stride` + `*K[k]` from
      -- llama.cpp.  K's offset is materialised once via let'.
      let kRowPtr ← ShaderM.ptr (.scalar .u32) "k_cache_f16" kvWords
                      (kHeadBaseU32 + kPosSafe * kRowStrideU32 + laneId)
      let partialVar ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
      for pk in [0:dPerLanePair] do
        let kPackedRaw ← (kRowPtr.atOffset (Exp.litU32 (pk * 32))).load
        -- Pin kPacked to a register so CodeGen lowers vecX + vecY of the
        -- same packed half2 into ONE ld.global, not two.
        let kPacked ← ShaderM.let' (.scalar .u32) kPackedRaw
        let unpacked := Exp.unpack2x16float kPacked
        let k0 ← ShaderM.let' (.scalar .f32) (Exp.vecX unpacked)
        let k1 ← ShaderM.let' (.scalar .f32) (Exp.vecY unpacked)
        let q0Exp : Exp (.scalar .f32) := Exp.var q0Vars[pk]!
        let q1Exp : Exp (.scalar .f32) := Exp.var q1Vars[pk]!
        let p0 : Exp (.scalar .f32) := Exp.var partialVar
        ShaderM.assign partialVar (p0 + q0Exp * k0)
        let p1 : Exp (.scalar .f32) := Exp.var partialVar
        ShaderM.assign partialVar (p1 + q1Exp * k1)

      -- Step 5 helper: warp-wide sum reduce (1 subgroupAdd, fully warp-merged).
      let sumWarp ← ShaderM.warpReduceSum 32 (Exp.var partialVar : Exp (.scalar .f32))
      let scoreGated := Exp.select inBounds sumWarp (Exp.litF32 (-1.0e30))
      ShaderM.assign kqMaxNewVar
        (Exp.max (Exp.var kqMaxNewVar) scoreGated)
      ShaderM.if_ (Exp.eq laneId iKQ0) (do
        ShaderM.assign kqRegVar scoreGated
      ) (pure ())

    -- Phase 2: per-thread softmax-online update.
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
    -- Skipping the barrier here is safe: shared_kq slot range
    -- [warpId*32 .. warpId*32+31] is written and read entirely within
    -- this warp.  Warp lanes execute in lock-step on NVIDIA, so shared
    -- writes are visible to subsequent reads in the same warp without
    -- bar.sync.  Measured -2 of 3 barriers; remaining 36% barrier-stall
    -- comes from the final cross-warp shared_vkq merge after the K loop.

    -- Phase 3: VKQ accumulation.
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 32) (Exp.litU32 1) fun kOff => do
      let kPos := Exp.add kVKQ0 (Exp.add (Exp.mul warpId (Exp.litU32 32)) kOff)
      let kqScoreSlot := Exp.add (Exp.mul warpId (Exp.litU32 32)) kOff
      let kqScoreRaw ← ShaderM.readWorkgroup (ty := .scalar .f32)
                        (n := workgroupSize) "shared_kq" kqScoreSlot
      -- See Phase 1 for the let' rationale.
      let inBoundsU32 ← ShaderM.let' (.scalar .u32)
                          (Exp.select (Exp.lt kPos splitEnd) (Exp.litU32 1) (Exp.litU32 0))
      let inBounds := Exp.eq inBoundsU32 (Exp.litU32 1)
      let kPosSafe ← ShaderM.let' (.scalar .u32) (Exp.select inBounds kPos (Exp.litU32 0))
      let kqScore ← ShaderM.let' (.scalar .f32) (Exp.select inBounds kqScoreRaw (Exp.litF32 0.0))
      let vRowPtr ← ShaderM.ptr (.scalar .u32) "v_cache_f16" kvWords
                       (kHeadBaseU32 + kPosSafe * kRowStrideU32 + laneId)
      for pk in [0:dPerLanePair] do
        let vPackedRaw ← (vRowPtr.atOffset (Exp.litU32 (pk * 32))).load
        let vPacked ← ShaderM.let' (.scalar .u32) vPackedRaw
        let unpacked := Exp.unpack2x16float vPacked
        let v0 ← ShaderM.let' (.scalar .f32) (Exp.vecX unpacked)
        let v1 ← ShaderM.let' (.scalar .f32) (Exp.vecY unpacked)
        let prev0 : Exp (.scalar .f32) := Exp.var vkq0Vars[pk]!
        let prev1 : Exp (.scalar .f32) := Exp.var vkq1Vars[pk]!
        ShaderM.assign vkq0Vars[pk]! (Exp.add prev0 (Exp.mul v0 kqScore))
        ShaderM.assign vkq1Vars[pk]! (Exp.add prev1 (Exp.mul v1 kqScore))
    -- No end-of-Phase-3 barrier either: next outer-K iter overwrites
    -- shared_kq[warpId*32..] before reading it again, and shared_vkq
    -- isn't touched until after the K loop ends (where the final cross-
    -- warp barrier still fires).

  -- Cross-warp softmax merge within this block (V7 layout).
  let warpKqSumName ← ShaderM.var (.scalar .f32) (Exp.subgroupAdd kqSum)
  ShaderM.if_ (Exp.eq laneId (Exp.litU32 0)) (do
    let metaIdx := Exp.mul warpId (Exp.litU32 2)
    ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_warp_meta"
      metaIdx kqMax
    ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_warp_meta"
      (Exp.add metaIdx (Exp.litU32 1)) (Exp.var warpKqSumName)
  ) (pure ())

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

  -- Compute block-local global max/sum across warps.
  let blockMaxName ← ShaderM.var (.scalar .f32) (Exp.litF32 (-1.0e30))
  for w in [0:numWarps] do
    let m ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := numWarps * 2)
              "shared_warp_meta" (Exp.litU32 (w * 2))
    ShaderM.assign blockMaxName (Exp.max (Exp.var blockMaxName) m)
  let blockMax : Exp (.scalar .f32) := Exp.var blockMaxName

  let blockSumName ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
  let mut weightVars : Array String := #[]
  for w in [0:numWarps] do
    let m ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := numWarps * 2)
              "shared_warp_meta" (Exp.litU32 (w * 2))
    let s ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := numWarps * 2)
              "shared_warp_meta" (Exp.litU32 (w * 2 + 1))
    let weight := Exp.exp (Exp.sub m blockMax)
    let weightName ← ShaderM.var (.scalar .f32) weight
    weightVars := weightVars.push weightName
    ShaderM.assign blockSumName
      (Exp.add (Exp.var blockSumName) (Exp.mul s (Exp.var weightName)))
  let blockSum : Exp (.scalar .f32) := Exp.var blockSumName

  -- Write per-block partial output (un-normalised numerator) and meta.
  -- Layout: partial_out[h * numSplits * headDim + splitIdx * headDim + d]
  --         partial_meta[h * numSplits * 2 + splitIdx * 2 + {0,1}]
  --
  -- The combine kernel multiplies p * exp(meta.x - global_max), so we want
  -- p = (numerator-of-block-softmax) such that the combine result is
  -- correct.  V5's combine kernel reads:
  --   acc += p * exp(m - global_max)
  --   denom += s * exp(m - global_max)
  --   out = acc / denom
  -- where m = block-local max, s = block-local sum.
  --
  -- p must equal (Σ_k_in_block exp(score_k - m) * v_k), i.e. the
  -- un-normalised VKQ sum for this block under its own local max baseline.
  -- That's exactly `vkq * blockSum` divided by V7's per-block normalisation,
  -- but V7 keeps vkq un-normalised (no per-block divide), so we just
  -- write `vkq` directly — wait, V7's vkq is normalised against per-warp
  -- merging (multiplied by exp(m_w - blockMax) when summed).  Each thread
  -- already holds blockSum-weighted partials in shared_vkq, which we sum
  -- over warps below.
  let partialBase := Exp.add (Exp.mul head
                                       (Exp.mul (Exp.litU32 numSplits) (Exp.litU32 headDim)))
                              (Exp.mul splitIdx (Exp.litU32 headDim))
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
    -- acc{0,1} = Σ_w (vkq_w * exp(m_w - blockMax)) = un-normalised numerator
    -- under the block-local max baseline.  Combine kernel multiplies this by
    -- exp(blockMax - global_max) and divides by Σ_split (sum * exp(.)).
    let outIdx0 := Exp.add partialBase d0
    let outIdx1 := Exp.add partialBase d1
    ShaderM.writeBuffer (ty := .scalar .f32) "partial_out" outIdx0
      (Exp.var acc0Var)
    ShaderM.writeBuffer (ty := .scalar .f32) "partial_out" outIdx1
      (Exp.var acc1Var)

  -- Thread 0 writes (blockMax, blockSum) for this split.
  ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
    let metaBase := Exp.mul head (Exp.mul (Exp.litU32 numSplits) (Exp.litU32 2))
    let metaIdx := Exp.add metaBase (Exp.mul splitIdx (Exp.litU32 2))
    ShaderM.writeBuffer (ty := .scalar .f32) "partial_meta" metaIdx blockMax
    ShaderM.writeBuffer (ty := .scalar .f32) "partial_meta"
      (Exp.add metaIdx (Exp.litU32 1)) blockSum
  ) (pure ())

/-- Vec-params V11 (Session 5, doc 60): V8 sub-warp partition + V7 correct
    softmax + V9 split-K, with f16 K/V cache.

    The sub-warp partition (`nthreads_KQ = 8`) lets each warp process 4 K-
    positions in parallel per inner iter (vs V7/V9's 1 K-position).  The
    8-way warp_reduce_sum (3 shfl) replaces V7/V9's 32-way (5 shfl).
    Combined with split-K, total work distribution matches llama.cpp's.

    V8 had this partition but got cross-sub-warp VKQ reduction wrong —
    multiple sub-warps wrote to the same `shared_vkq` slot.  V11's fix:
      slot = (warpId * numSubWarps + subWarp) * D + dThreadBase + k
    so each (warp, subWarp, lane, k) maps to a distinct slot.  Final
    reduce sums across `numWarps × numSubWarps` partials.

    Lane → dim mapping (V11 f16 layout):
      Each thread owns `dimsPerLane = D / nthreads_KQ = 32` dims.
      Lane t with subLane = t & 7 owns dims [subLane*32, (subLane+1)*32).
      In f16-packed words (2 dims/word): subLane*16 + j for j in [0, 16).

    Pre-conditions:
      - cacheLen >= numSplits, K/V are f16, headDim % 64 == 0.
      - workgroupSize = 128, headDim = 256 (one warp covers all 8 sub-warps,
        so each sub-warp's 8 lanes split 256 dims into 32-dim chunks).
    Grid: (numHeads, numSplits, 1).  Block: 128 threads.
    minnctapersm := 1 (drives ptxas to 512 reg/thread budget). -/
def flashAttentionVecParamsKernelV11
    (numHeads numKVHeads maxSeqLen headDim numSplits : Nat) (scale : Float) :
    ShaderM Unit := do
  let workgroupSize : Nat := 128
  let numWarps : Nat := workgroupSize / 32
  let nthreadsKQ : Nat := 8 -- experiment-edit-marker
  let numSubWarps : Nat := 32 / nthreadsKQ          -- = 4
  let dimsPerLane : Nat := headDim / nthreadsKQ     -- = 32 dims/lane
  let dPerLanePair : Nat := dimsPerLane / 2         -- = 16 word-pairs/lane

  let wgid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let head := Exp.vec3X wgid
  let splitIdx := Exp.vec3Y wgid
  let tid := Exp.vec3X lid
  let headsPerKV := numHeads / numKVHeads
  let kvHead := Exp.div head (Exp.litU32 headsPerKV)

  let _q ← ShaderM.declareInputBuffer "q" (.array (.scalar .f32) (numHeads * headDim))
  let kvWords : Nat := (numKVHeads * maxSeqLen * headDim) / 2
  let _kCache ← ShaderM.declareInputBuffer "k_cache_f16"
                  (.array (.scalar .u32) kvWords)
  let _vCache ← ShaderM.declareInputBuffer "v_cache_f16"
                  (.array (.scalar .u32) kvWords)
  let _partialOut ← ShaderM.declareOutputBuffer "partial_out"
                      (.array (.scalar .f32) (numHeads * numSplits * headDim))
  let _partialMeta ← ShaderM.declareOutputBuffer "partial_meta"
                       (.array (.scalar .f32) (numHeads * numSplits * 2))
  let _params ← ShaderM.declareStorageBuffer "params" (.array (.scalar .u32) 2) .read

  -- Smem layout:
  --   shared_kq: per-warp 32 K-position scores (warpId*32 + i)
  --   shared_vkq: distinct slot per (warp, subWarp, dim) — V8's slot-
  --     collision fix.  Size = numWarps * numSubWarps * headDim
  --   shared_warp_meta: (max, sum) per warp for cross-warp merge
  ShaderM.sharedNamed "shared_kq" (.array (.scalar .f32) workgroupSize)
  ShaderM.sharedNamed "shared_vkq"
    (.array (.scalar .f32) (numWarps * numSubWarps * headDim))
  ShaderM.sharedNamed "shared_warp_meta" (.array (.scalar .f32) (numWarps * 2))

  let laneId := Exp.bitAnd tid (Exp.litU32 31)
  let warpId := Exp.shiftRight tid (Exp.litU32 5)
  -- subLane and subWarp are referenced 100s of times in the unrolled body —
  -- materialise once via ShaderM.let' so PTX gets one register, not N copies.
  let subLane ← ShaderM.let' (.scalar .u32) (Exp.bitAnd laneId (Exp.litU32 7))
  let subWarp ← ShaderM.let' (.scalar .u32) (Exp.shiftRight laneId (Exp.litU32 3))

  -- Each thread loads 16 word-pairs of Q (32 dims).
  -- Word offset within Q row for thread = subLane*16 + j.
  -- Each word holds dims (2*(subLane*16+j), 2*(subLane*16+j)+1).
  -- Step 9c: RegArray ty n replaces 4 manual `Array String` arrays.
  let qBase := Exp.mul head (Exp.litU32 headDim)
  let dThreadBase := Exp.mul subLane (Exp.litU32 dimsPerLane)  -- f32 dim base
  -- Pre-load Q values then scale-init the registers.  Has to be 2 passes
  -- because RegArray.mk's init function is pure (Nat → Exp), it can't
  -- itself perform `readBuffer` (a ShaderM action).
  let mut q0Vals : Array (Exp (.scalar .f32)) := #[]
  let mut q1Vals : Array (Exp (.scalar .f32)) := #[]
  for pk in [0:dPerLanePair] do
    let d0 := Exp.add dThreadBase (Exp.litU32 (pk * 2))
    let d1 := Exp.add d0 (Exp.litU32 1)
    let v0 ← ShaderM.readBuffer (ty := .scalar .f32) (n := numHeads * headDim)
              "q" (Exp.add qBase d0)
    let v1 ← ShaderM.readBuffer (ty := .scalar .f32) (n := numHeads * headDim)
              "q" (Exp.add qBase d1)
    q0Vals := q0Vals.push v0
    q1Vals := q1Vals.push v1
  let q0   ← ShaderM.regArray (.scalar .f32) dPerLanePair
               (fun pk => Exp.mul (q0Vals[pk]?.getD (Exp.litF32 0.0))
                                  (Exp.litF32 scale))
  let q1   ← ShaderM.regArray (.scalar .f32) dPerLanePair
               (fun pk => Exp.mul (q1Vals[pk]?.getD (Exp.litF32 0.0))
                                  (Exp.litF32 scale))
  let vkq0 ← ShaderM.regArray (.scalar .f32) dPerLanePair
               (fun _ => Exp.litF32 0.0)
  let vkq1 ← ShaderM.regArray (.scalar .f32) dPerLanePair
               (fun _ => Exp.litF32 0.0)
  -- Back-compat shims: existing call sites still reference the old
  -- `Array String` names, so expose `q0.names`/etc until the rest of
  -- V11 is migrated.  (Phase 1 inner pk-loop, etc.)
  let q0Vars   := q0.names
  let q1Vars   := q1.names
  let vkq0Vars := vkq0.names
  let vkq1Vars := vkq1.names

  ShaderM.varNamed "kq_max" (.scalar .f32) (Exp.litF32 (-1.0e30))
  ShaderM.varNamed "kq_sum" (.scalar .f32) (Exp.litF32 0.0)
  let kqMax := Exp.var "kq_max"
  let kqSum := Exp.var "kq_sum"

  let cacheLen ← ShaderM.readBuffer (ty := .scalar .u32) (n := 2)
                   "params" (Exp.litU32 1)
  let kRowStrideU32 := Exp.litU32 (headDim / 2)
  let kHeadBaseU32 ← ShaderM.let' (.scalar .u32)
                       (Exp.mul kvHead (Exp.mul (Exp.litU32 maxSeqLen) kRowStrideU32))

  -- Split K range: [splitStart, splitEnd) per (head, splitIdx) block.
  let splitStart := Exp.div (Exp.mul splitIdx cacheLen) (Exp.litU32 numSplits)
  let splitEnd ← ShaderM.let' (.scalar .u32)
                   (Exp.div (Exp.mul (Exp.add splitIdx (Exp.litU32 1)) cacheLen)
                            (Exp.litU32 numSplits))

  -- Outer K loop: workgroupSize K-positions per iter.
  -- Each warp processes 32 K-positions per outer iter; 4 sub-warps run in
  -- parallel covering 4 K-positions per inner iter, 8 inner iters → 32.
  ShaderM.loop splitStart splitEnd (Exp.litU32 workgroupSize) fun kVKQ0 => do
    let kqRegVar ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    let kqMaxNewVar ← ShaderM.var (.scalar .f32) kqMax

    -- Phase 1: 8 inner iters static-unrolled.  Each iter, 4 sub-warps in
    -- parallel handle 4 K-positions: sub-warp s handles K-position
    --   warpId*32 + s*nthreadsKQ + iKQ0
    -- Step 9a: `unrollForScoped` collapses `for ... do ShaderM.scope do`
    -- into one call, mirroring CUDA C++ `#pragma unroll for ...`.
    -- Step 8: comparison operators `<ᵉ` / `==ᵉ` (instead of Exp.lt/eq).
    ShaderM.unrollForScoped nthreadsKQ fun iKQ0 => do
      let iKQ ← ShaderM.let' (.scalar .u32)
                   (warpId * 32 + (subWarp * (Exp.litU32 nthreadsKQ)
                                   + (Exp.litU32 iKQ0)))
      let kPos ← ShaderM.let' (.scalar .u32) (kVKQ0 + iKQ)
      let inBoundsU32 ← ShaderM.let' (.scalar .u32)
                          (Exp.select (kPos <ᵉ splitEnd) 1 0)
      let inBounds := inBoundsU32 ==ᵉ 1
      let kPosSafe ← ShaderM.let' (.scalar .u32) (Exp.select inBounds kPos 0)
      let kRowBaseU32 ← ShaderM.let' (.scalar .u32)
                          (kHeadBaseU32 + kPosSafe * kRowStrideU32)

      let partialVar ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
      for pk in [0:dPerLanePair] do
        -- Word offset within K row: each thread reads 16 consecutive words.
        -- subLane*16 + pk gives the absolute word index within the K row.
        let wordOff := subLane * (Exp.litU32 dPerLanePair) + (Exp.litU32 pk)
        let kWordIdx := kRowBaseU32 + wordOff
        let kPackedRaw ← ShaderM.readBuffer (ty := .scalar .u32) (n := kvWords)
                           "k_cache_f16" kWordIdx
        let kPacked ← ShaderM.let' (.scalar .u32) kPackedRaw
        let unpacked := Exp.unpack2x16float kPacked
        let k0 ← ShaderM.let' (.scalar .f32) (Exp.vecX unpacked)
        let k1 ← ShaderM.let' (.scalar .f32) (Exp.vecY unpacked)
        let q0Exp := q0.get pk     -- Step 9c: typed register array
        let q1Exp := q1.get pk
        let p0 : Exp (.scalar .f32) := Exp.var partialVar
        ShaderM.assign partialVar (p0 + q0Exp * k0)
        let p1 : Exp (.scalar .f32) := Exp.var partialVar
        ShaderM.assign partialVar (p1 + q1Exp * k1)

      -- 8-way warp_reduce_sum: Step 5 helper replaces 12 lines of
      -- subgroupShuffleXor xor 1, 2, 4 by hand.
      let sumSubWarp ← ShaderM.warpReduceSum 8 (Exp.var partialVar)

      let scoreGated := Exp.select inBounds sumSubWarp (Exp.litF32 (-1.0e30))
      ShaderM.assign kqMaxNewVar
        (Exp.max (Exp.var kqMaxNewVar) scoreGated)

      -- Lane that owns K-position iKQ in this warp's chunk:
      --   laneId = subWarp*nthreadsKQ + iKQ0
      ShaderM.if_ (Exp.eq laneId
                    (Exp.add (Exp.mul subWarp (Exp.litU32 nthreadsKQ))
                              (Exp.litU32 iKQ0))) (do
        ShaderM.assign kqRegVar scoreGated
      ) (pure ())

    -- Phase 2a: cross-sub-warp max-reduce so all 32 lanes share the same
    -- per-warp kqMaxNew (4 sub-warps each had different max over their 8 K).
    let kqMaxNewS8Name ← ShaderM.var (.scalar .f32)
                           (Exp.max (Exp.var kqMaxNewVar)
                                    (Exp.subgroupShuffleXor (Exp.var kqMaxNewVar)
                                      (Exp.litU32 8)))
    let kqMaxNewS16Name ← ShaderM.var (.scalar .f32)
                            (Exp.max (Exp.var kqMaxNewS8Name)
                                     (Exp.subgroupShuffleXor (Exp.var kqMaxNewS8Name)
                                       (Exp.litU32 16)))
    let kqMaxNew : Exp (.scalar .f32) := Exp.var kqMaxNewS16Name

    -- Phase 2b: per-thread softmax-online update with warp-global max.
    let kqMaxScaleVar ← ShaderM.var (.scalar .f32)
                          (Exp.exp (Exp.sub kqMax kqMaxNew))
    let kqMaxScale : Exp (.scalar .f32) := Exp.var kqMaxScaleVar
    ShaderM.assign "kq_max" kqMaxNew

    -- My K-position: lane laneId in this warp owns K-position
    --   warpId*32 + laneId = warpId*32 + subWarp*8 + (laneId & 7)
    -- which is exactly the lane that received its score in the inner-loop
    -- "if laneId == subWarp*nthreadsKQ + iKQ0" branch.
    let myKPos ← ShaderM.let' (.scalar .u32)
                   (Exp.add kVKQ0 (Exp.add (Exp.mul warpId (Exp.litU32 32)) laneId))
    let myInBounds := Exp.lt myKPos splitEnd
    let kqRegExp := Exp.exp (Exp.sub (Exp.var kqRegVar) kqMaxNew)
    let kqRegGated := Exp.select myInBounds kqRegExp (Exp.litF32 0.0)
    ShaderM.assign kqRegVar kqRegGated
    ShaderM.assign "kq_sum"
      (Exp.add (Exp.mul kqSum kqMaxScale) (Exp.var kqRegVar))
    -- Rescale all per-thread VKQ accumulators by kqMaxScale.
    for pk in [0:dPerLanePair] do
      let prev0 : Exp (.scalar .f32) := Exp.var vkq0Vars[pk]!
      let prev1 : Exp (.scalar .f32) := Exp.var vkq1Vars[pk]!
      ShaderM.assign vkq0Vars[pk]! (Exp.mul prev0 kqMaxScale)
      ShaderM.assign vkq1Vars[pk]! (Exp.mul prev1 kqMaxScale)

    -- Publish kqRegVar to smem.  Slot = warpId * 32 + laneId.
    let kqSlot := Exp.add (Exp.mul warpId (Exp.litU32 32)) laneId
    ShaderM.writeWorkgroup (ty := .scalar .f32)
      "shared_kq" kqSlot (Exp.var kqRegVar)
    ShaderM.barrier

    -- Phase 3: VKQ accumulation.  For each of 32 K-positions (this warp's
    -- chunk), 4 sub-warps each contribute partial VKQ for their owned dims.
    -- Lane t holds vkq_pk for dims (subLane*32 + 2*pk, ...+1) for this warp.
    -- We iterate K-positions sequentially within each sub-warp (8 iters per
    -- sub-warp) and broadcast the score from shared_kq to all sub-warps.
    --
    -- Within a sub-warp, all 8 lanes process the SAME K-position (each lane
    -- handles its own 32-dim slice).  The 4 sub-warps in parallel handle
    -- 4 different K-positions per inner iter.
    for iKQ0 in [0:nthreadsKQ] do ShaderM.scope do
      let iKQ ← ShaderM.let' (.scalar .u32)
                   (Exp.add (Exp.mul subWarp (Exp.litU32 nthreadsKQ))
                            (Exp.litU32 iKQ0))
      let kqScoreSlot ← ShaderM.let' (.scalar .u32)
                          (Exp.add (Exp.mul warpId (Exp.litU32 32)) iKQ)
      let kqScoreRaw ← ShaderM.readWorkgroup (ty := .scalar .f32)
                        (n := workgroupSize) "shared_kq" kqScoreSlot
      let kPosAbs ← ShaderM.let' (.scalar .u32)
                      (Exp.add kVKQ0 (Exp.add (Exp.mul warpId (Exp.litU32 32)) iKQ))
      let inBoundsU32 ← ShaderM.let' (.scalar .u32)
                          (Exp.select (Exp.lt kPosAbs splitEnd)
                                       (Exp.litU32 1) (Exp.litU32 0))
      let inBounds := Exp.eq inBoundsU32 (Exp.litU32 1)
      let kPosSafe ← ShaderM.let' (.scalar .u32) (Exp.select inBounds kPosAbs (Exp.litU32 0))
      let kqScore ← ShaderM.let' (.scalar .f32)
                       (Exp.select inBounds kqScoreRaw (Exp.litF32 0.0))
      let vRowBaseU32 ← ShaderM.let' (.scalar .u32)
                           (Exp.add kHeadBaseU32 (Exp.mul kPosSafe kRowStrideU32))
      for pk in [0:dPerLanePair] do
        let wordOff := Exp.add (Exp.mul subLane (Exp.litU32 dPerLanePair))
                                (Exp.litU32 pk)
        let vWordIdx := Exp.add vRowBaseU32 wordOff
        let vPackedRaw ← ShaderM.readBuffer (ty := .scalar .u32) (n := kvWords)
                           "v_cache_f16" vWordIdx
        let vPacked ← ShaderM.let' (.scalar .u32) vPackedRaw
        let unpacked := Exp.unpack2x16float vPacked
        let v0 ← ShaderM.let' (.scalar .f32) (Exp.vecX unpacked)
        let v1 ← ShaderM.let' (.scalar .f32) (Exp.vecY unpacked)
        let prev0 : Exp (.scalar .f32) := Exp.var vkq0Vars[pk]!
        let prev1 : Exp (.scalar .f32) := Exp.var vkq1Vars[pk]!
        ShaderM.assign vkq0Vars[pk]! (Exp.add prev0 (Exp.mul v0 kqScore))
        ShaderM.assign vkq1Vars[pk]! (Exp.add prev1 (Exp.mul v1 kqScore))
    ShaderM.barrier

  -- After the K loop: each lane holds its partial VKQ for dims
  --   [subLane*32, (subLane+1)*32) over THIS sub-warp's K-positions.
  -- Cross-warp merge needs to collect all (warp, subWarp) partials.
  -- Slot = (warpId * numSubWarps + subWarp) * D + (subLane*32 + 2*pk + {0,1}).

  -- Cross-warp meta first: each warp's kqMax, kqSum.
  let warpKqSumName ← ShaderM.var (.scalar .f32) (Exp.subgroupAdd kqSum)
  ShaderM.if_ (Exp.eq laneId (Exp.litU32 0)) (do
    let metaIdx := Exp.mul warpId (Exp.litU32 2)
    ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_warp_meta"
      metaIdx kqMax
    ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_warp_meta"
      (Exp.add metaIdx (Exp.litU32 1)) (Exp.var warpKqSumName)
  ) (pure ())

  -- Each thread writes its VKQ partial to a distinct slot.
  -- Slot index: ((warpId * numSubWarps) + subWarp) * D + (subLane*32 + 2*pk + {0,1})
  --           = (warpId * numSubWarps + subWarp) * D + dThreadBase + 2*pk + {0,1}
  let bigSlotBase ← ShaderM.let' (.scalar .u32)
                       (Exp.mul (Exp.add (Exp.mul warpId (Exp.litU32 numSubWarps))
                                          subWarp)
                                 (Exp.litU32 headDim))
  for pk in [0:dPerLanePair] do
    let d0 := Exp.add dThreadBase (Exp.litU32 (pk * 2))
    let d1 := Exp.add d0 (Exp.litU32 1)
    let slot0 := Exp.add bigSlotBase d0
    let slot1 := Exp.add bigSlotBase d1
    ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_vkq" slot0
      (Exp.var vkq0Vars[pk]!)
    ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_vkq" slot1
      (Exp.var vkq1Vars[pk]!)
  ShaderM.barrier

  -- Block-local global max/sum across warps (NOT sub-warps; sub-warps within
  -- a warp share the same kqMax via Phase 2a).
  let blockMaxName ← ShaderM.var (.scalar .f32) (Exp.litF32 (-1.0e30))
  for w in [0:numWarps] do
    let m ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := numWarps * 2)
              "shared_warp_meta" (Exp.litU32 (w * 2))
    ShaderM.assign blockMaxName (Exp.max (Exp.var blockMaxName) m)
  let blockMax : Exp (.scalar .f32) := Exp.var blockMaxName

  let blockSumName ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
  let mut weightVars : Array String := #[]
  for w in [0:numWarps] do
    let m ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := numWarps * 2)
              "shared_warp_meta" (Exp.litU32 (w * 2))
    let s ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := numWarps * 2)
              "shared_warp_meta" (Exp.litU32 (w * 2 + 1))
    let weight := Exp.exp (Exp.sub m blockMax)
    let weightName ← ShaderM.var (.scalar .f32) weight
    weightVars := weightVars.push weightName
    ShaderM.assign blockSumName
      (Exp.add (Exp.var blockSumName) (Exp.mul s (Exp.var weightName)))
  let blockSum : Exp (.scalar .f32) := Exp.var blockSumName

  -- Final partial output: sum across (numWarps × numSubWarps) partials,
  -- weighted by the per-warp exp(m_w - blockMax).  All sub-warps within a
  -- warp share the same weight (their kqMax is unified via Phase 2a).
  let partialBase := Exp.add (Exp.mul head
                                       (Exp.mul (Exp.litU32 numSplits) (Exp.litU32 headDim)))
                              (Exp.mul splitIdx (Exp.litU32 headDim))
  for pk in [0:dPerLanePair] do
    let d0 := Exp.add dThreadBase (Exp.litU32 (pk * 2))
    let d1 := Exp.add d0 (Exp.litU32 1)
    let acc0Var ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    let acc1Var ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    for w in [0:numWarps] do
      for s in [0:numSubWarps] do
        let bigBase := (w * numSubWarps + s) * headDim
        let slot0 := Exp.add (Exp.litU32 bigBase) d0
        let slot1 := Exp.add (Exp.litU32 bigBase) d1
        let v0 ← ShaderM.readWorkgroup (ty := .scalar .f32)
                  (n := numWarps * numSubWarps * headDim) "shared_vkq" slot0
        let v1 ← ShaderM.readWorkgroup (ty := .scalar .f32)
                  (n := numWarps * numSubWarps * headDim) "shared_vkq" slot1
        ShaderM.assign acc0Var
          (Exp.add (Exp.var acc0Var) (Exp.mul v0 (Exp.var weightVars[w]!)))
        ShaderM.assign acc1Var
          (Exp.add (Exp.var acc1Var) (Exp.mul v1 (Exp.var weightVars[w]!)))
    let outIdx0 := Exp.add partialBase d0
    let outIdx1 := Exp.add partialBase d1
    ShaderM.writeBuffer (ty := .scalar .f32) "partial_out" outIdx0
      (Exp.var acc0Var)
    ShaderM.writeBuffer (ty := .scalar .f32) "partial_out" outIdx1
      (Exp.var acc1Var)

  -- Thread 0 writes (blockMax, blockSum) for this split.
  ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
    let metaBase := Exp.mul head (Exp.mul (Exp.litU32 numSplits) (Exp.litU32 2))
    let metaIdx := Exp.add metaBase (Exp.mul splitIdx (Exp.litU32 2))
    ShaderM.writeBuffer (ty := .scalar .f32) "partial_meta" metaIdx blockMax
    ShaderM.writeBuffer (ty := .scalar .f32) "partial_meta"
      (Exp.add metaIdx (Exp.litU32 1)) blockSum
  ) (pure ())

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


end Hesper.WGSL.FlashAttention

