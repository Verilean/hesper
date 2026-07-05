import Hesper.WGSL.FlashAttention

/-!
# Flash Attention V11 (split-K + sub-warp partition + f16 K/V cache)

Production V11 path (used via `executeFlashAttentionV11`).  Only V11 +
its combine kernel are kept here; earlier experimental variants
(V2/V3/V5/V6/V7/V8/V9/V10) were removed since they were only used by
ncu profiling drivers and the cross-kernel parity test, neither of
which is part of the production decode path.
-/

namespace Hesper.WGSL.FlashAttention

open Hesper.WGSL
open Hesper.WGSL.Monad
open Hesper


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
  let nthreadsKQ : Nat := 16 -- experiment-edit-marker (was 8; C-path)
  let numSubWarps : Nat := 32 / nthreadsKQ          -- = 2 (was 4)
  let dimsPerLane : Nat := headDim / nthreadsKQ     -- = 16 dims/lane (was 32)
  let dPerLanePair : Nat := dimsPerLane / 2         -- = 8 word-pairs/lane (was 16)
  -- Mask = nthreadsKQ - 1 (assumes power-of-2): selects within-sub-warp lane.
  let subLaneMask : Nat := nthreadsKQ - 1
  -- Shift = log2(nthreadsKQ) for subWarp extraction.
  let subWarpShift : Nat :=
    if nthreadsKQ == 8 then 3
    else if nthreadsKQ == 16 then 4
    else 5  -- nthreadsKQ == 32 (no sub-warp partition)

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
  -- B-path: shared_vkq holds per-warp partials only (numSubWarps fan-in
  -- collapsed via warp shuffle).  Size = numWarps * headDim, 4× smaller
  -- than V11's numWarps * numSubWarps * headDim.  Cuts the cross-warp
  -- aggregation LDS+FFMA count from 256/thread → 64/thread.
  ShaderM.sharedNamed "shared_vkq"
    (.array (.scalar .f32) (numWarps * headDim))
  ShaderM.sharedNamed "shared_warp_meta" (.array (.scalar .f32) (numWarps * 2))

  let laneId := Exp.bitAnd tid (Exp.litU32 31)
  let warpId := Exp.shiftRight tid (Exp.litU32 5)
  -- subLane and subWarp are referenced 100s of times in the unrolled body —
  -- materialise once via ShaderM.let' so PTX gets one register, not N copies.
  let subLane ← ShaderM.let' (.scalar .u32) (Exp.bitAnd laneId (Exp.litU32 subLaneMask))
  let subWarp ← ShaderM.let' (.scalar .u32) (Exp.shiftRight laneId (Exp.litU32 subWarpShift))

  -- Each thread loads 16 word-pairs of Q (32 dims).
  -- Word offset within Q row for thread = subLane*16 + j.
  -- Each word holds dims (2*(subLane*16+j), 2*(subLane*16+j)+1).
  -- Step 9c: RegArray ty n replaces 4 manual `Array String` arrays.
  let qBase := Exp.mul head (Exp.litU32 headDim)
  -- Materialise dThreadBase via let' so it's a single PTX register, not
  -- recomputed at each of the ~50 use sites in the unrolled body.  This
  -- alone removes ~50 mul.lo.u32 / shl.b32 instructions.
  let dThreadBase ← ShaderM.let' (.scalar .u32)
                       (Exp.mul subLane (Exp.litU32 dimsPerLane))  -- f32 dim base
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

  -- Step 9f: named sentinels — same numeric values as before, but the
  -- intent ("min-finite-score" / "zero accumulator") is explicit.
  ShaderM.varNamed "kq_max" (.scalar .f32) Exp.negInf30
  ShaderM.varNamed "kq_sum" (.scalar .f32) Exp.f32Zero
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
      -- Vec4 load: each thread reads 16 consecutive u32 words in groups of 4
      -- via one ld.global.nc.v4.u32 (LDG.E.128) per group.  Reduces global
      -- load instruction count 4× → relieves MIO-pipe saturation that ncu
      -- identified as V11's #1 stall (mio_throttle).
      for grp in [0:dPerLanePair / 4] do
        let wordOff := subLane * (Exp.litU32 dPerLanePair) +
                       (Exp.litU32 (grp * 4))
        let baseIdx := kRowBaseU32 + wordOff
        let (kw0, kw1, kw2, kw3) ← ShaderM.readBufferU32x4 "k_cache_f16" baseIdx
        for (j, kPacked) in [(0, kw0), (1, kw1), (2, kw2), (3, kw3)] do
          let pk := grp * 4 + j
          let unpacked := Exp.unpack2x16float kPacked
          let k0 ← ShaderM.let' (.scalar .f32) (Exp.vecX unpacked)
          let k1 ← ShaderM.let' (.scalar .f32) (Exp.vecY unpacked)
          let q0Exp := q0.get pk
          let q1Exp := q1.get pk
          let p0 : Exp (.scalar .f32) := Exp.var partialVar
          ShaderM.assign partialVar (p0 + q0Exp * k0)
          let p1 : Exp (.scalar .f32) := Exp.var partialVar
          ShaderM.assign partialVar (p1 + q1Exp * k1)

      -- nthreadsKQ-way warp_reduce_sum (sub-warp dot product reduce).
      -- For nthreadsKQ=8: butterfly xor 1, 2, 4 (3 steps).
      -- For nthreadsKQ=16: butterfly xor 1, 2, 4, 8 (4 steps).
      -- For nthreadsKQ=32: full warp reduce (xor 1..16).
      let sumSubWarp ← ShaderM.warpReduceSum nthreadsKQ (Exp.var partialVar)

      let scoreGated := Exp.select inBounds sumSubWarp (Exp.negInf30)
      ShaderM.assign kqMaxNewVar
        (Exp.max (Exp.var kqMaxNewVar) scoreGated)

      -- Lane that owns K-position iKQ in this warp's chunk:
      --   laneId = subWarp*nthreadsKQ + iKQ0
      ShaderM.if_ (Exp.eq laneId
                    (Exp.add (Exp.mul subWarp (Exp.litU32 nthreadsKQ))
                              (Exp.litU32 iKQ0))) (do
        ShaderM.assign kqRegVar scoreGated
      ) (pure ())

    -- Step 9g: warp-only barrier between Phase 1 (per-sub-warp `kqMaxNew`
    -- writes via the `if laneId == ...` branch) and Phase 2a's
    -- cross-sub-warp shuffle.  PTX emits `bar.warp.sync 0xFFFFFFFF`,
    -- much cheaper than a block barrier; WGSL falls back to block
    -- barrier (correct, slightly over-syncs).
    ShaderM.warpBarrier
    -- Phase 2a: cross-sub-warp max-reduce so all 32 lanes share the same
    -- per-warp kqMaxNew (numSubWarps each had different max over their
    -- nthreadsKQ K).  For numSubWarps=4 (nthreadsKQ=8): xor 8, then xor 16.
    -- For numSubWarps=2 (nthreadsKQ=16): xor 16 only.
    -- For numSubWarps=1 (nthreadsKQ=32): no cross-sub-warp shuffle needed.
    let kqMaxNew : Exp (.scalar .f32) ← do
      if numSubWarps = 1 then
        pure (Exp.var kqMaxNewVar)
      else if numSubWarps = 2 then
        let n ← ShaderM.var (.scalar .f32)
                  (Exp.max (Exp.var kqMaxNewVar)
                           (Exp.subgroupShuffleXor (Exp.var kqMaxNewVar)
                             (Exp.litU32 nthreadsKQ)))
        pure (Exp.var n)
      else
        -- numSubWarps = 4
        let n8 ← ShaderM.var (.scalar .f32)
                   (Exp.max (Exp.var kqMaxNewVar)
                            (Exp.subgroupShuffleXor (Exp.var kqMaxNewVar)
                              (Exp.litU32 nthreadsKQ)))
        let n16 ← ShaderM.var (.scalar .f32)
                    (Exp.max (Exp.var n8)
                             (Exp.subgroupShuffleXor (Exp.var n8)
                               (Exp.litU32 (2 * nthreadsKQ))))
        pure (Exp.var n16)

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
      -- Vec4 load: 16 V-words per thread → 4 ld.global.nc.v4.u32 instructions
      -- instead of 16 scalar ld.global.u32 (same MIO-pipe relief as Phase 1).
      for grp in [0:dPerLanePair / 4] do
        let wordOff := Exp.add (Exp.mul subLane (Exp.litU32 dPerLanePair))
                                (Exp.litU32 (grp * 4))
        let baseIdx := Exp.add vRowBaseU32 wordOff
        let (vw0, vw1, vw2, vw3) ← ShaderM.readBufferU32x4 "v_cache_f16" baseIdx
        for (j, vPacked) in [(0, vw0), (1, vw1), (2, vw2), (3, vw3)] do
          let pk := grp * 4 + j
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

  -- B-path Phase 4a: cross-sub-warp aggregation via warp shuffle.
  -- Each thread's vkq[pk] is a partial over its sub-warp's nthreadsKQ K-pos.
  -- numSubWarps sub-warps within a warp contribute partials for the SAME
  -- dim slice; sum them with butterfly shuffles.
  --   numSubWarps=4: xor nthreadsKQ, xor 2*nthreadsKQ
  --   numSubWarps=2: xor nthreadsKQ only
  --   numSubWarps=1: no shuffle
  if numSubWarps != 1 then
    for pk in [0:dPerLanePair] do
      let v0 : Exp (.scalar .f32) := Exp.var vkq0Vars[pk]!
      let v1 : Exp (.scalar .f32) := Exp.var vkq1Vars[pk]!
      let v0a := Exp.add v0 (Exp.subgroupShuffleXor v0 (Exp.litU32 nthreadsKQ))
      let v1a := Exp.add v1 (Exp.subgroupShuffleXor v1 (Exp.litU32 nthreadsKQ))
      if numSubWarps = 4 then
        let v0b := Exp.add v0a (Exp.subgroupShuffleXor v0a (Exp.litU32 (2 * nthreadsKQ)))
        let v1b := Exp.add v1a (Exp.subgroupShuffleXor v1a (Exp.litU32 (2 * nthreadsKQ)))
        ShaderM.assign vkq0Vars[pk]! v0b
        ShaderM.assign vkq1Vars[pk]! v1b
      else
        ShaderM.assign vkq0Vars[pk]! v0a
        ShaderM.assign vkq1Vars[pk]! v1a

  -- Phase 4b: write the per-warp totals to smem.  All 4 sub-warps in a warp
  -- now hold the SAME value, so only sub-warp 0 needs to write (saves
  -- 3× redundant smem writes).  Layout: shared_vkq[warpId * D + d].
  let warpSlotBase ← ShaderM.let' (.scalar .u32)
                       (Exp.mul warpId (Exp.litU32 headDim))
  ShaderM.if_ (Exp.eq subWarp (Exp.litU32 0)) (do
    for pk in [0:dPerLanePair] do
      let d0 := Exp.add dThreadBase (Exp.litU32 (pk * 2))
      let d1 := Exp.add d0 (Exp.litU32 1)
      let slot0 := Exp.add warpSlotBase d0
      let slot1 := Exp.add warpSlotBase d1
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_vkq" slot0
        (Exp.var vkq0Vars[pk]!)
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_vkq" slot1
        (Exp.var vkq1Vars[pk]!)
  ) (pure ())
  ShaderM.barrier

  -- Block-local global max/sum across warps (NOT sub-warps; sub-warps within
  -- a warp share the same kqMax via Phase 2a).
  let blockMaxName ← ShaderM.var (.scalar .f32) (Exp.negInf30)
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
  -- B-path: cross-warp aggregation reads only numWarps partials per dim
  -- (sub-warp dimension already collapsed via shuffle).  LDS count per
  -- thread: numWarps * numQuads = 4 * 8 = 32 LDS.128 = 8× fewer than V11.
  --
  -- Slot layout for shared_vkq (B):
  --   warpId * D + (subLane*32 + 4*quad + j)
  let numQuads := dPerLanePair / 2  -- = 8 (16 pk pairs / 2 = 8 quads of 4 dims)
  for quad in [0:numQuads] do
    let dBase := Exp.add dThreadBase (Exp.litU32 (quad * 4))
    let acc0Var ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    let acc1Var ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    let acc2Var ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    let acc3Var ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    for w in [0:numWarps] do
      let warpBase := w * headDim
      let slotBase := Exp.add (Exp.litU32 warpBase) dBase
      let (v0, v1, v2, v3) ← ShaderM.readWorkgroupF32x4 "shared_vkq" slotBase
      let weight := Exp.var weightVars[w]!
      ShaderM.assign acc0Var (Exp.add (Exp.var acc0Var) (Exp.mul v0 weight))
      ShaderM.assign acc1Var (Exp.add (Exp.var acc1Var) (Exp.mul v1 weight))
      ShaderM.assign acc2Var (Exp.add (Exp.var acc2Var) (Exp.mul v2 weight))
      ShaderM.assign acc3Var (Exp.add (Exp.var acc3Var) (Exp.mul v3 weight))
    let outBase := Exp.add partialBase dBase
    ShaderM.writeBuffer (ty := .scalar .f32) "partial_out"
      outBase (Exp.var acc0Var)
    ShaderM.writeBuffer (ty := .scalar .f32) "partial_out"
      (Exp.add outBase (Exp.litU32 1)) (Exp.var acc1Var)
    ShaderM.writeBuffer (ty := .scalar .f32) "partial_out"
      (Exp.add outBase (Exp.litU32 2)) (Exp.var acc2Var)
    ShaderM.writeBuffer (ty := .scalar .f32) "partial_out"
      (Exp.add outBase (Exp.litU32 3)) (Exp.var acc3Var)

  -- Thread 0 writes (blockMax, blockSum) for this split.
  ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
    let metaBase := Exp.mul head (Exp.mul (Exp.litU32 numSplits) (Exp.litU32 2))
    let metaIdx := Exp.add metaBase (Exp.mul splitIdx (Exp.litU32 2))
    ShaderM.writeBuffer (ty := .scalar .f32) "partial_meta" metaIdx blockMax
    ShaderM.writeBuffer (ty := .scalar .f32) "partial_meta"
      (Exp.add metaIdx (Exp.litU32 1)) blockSum
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
  let globalMaxVar ← ShaderM.var (.scalar .f32) (Exp.negInf30)
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


/-- Production launcher for V11 FlashAttention (split-K + sub-warp partition,
    f16 K/V cache).  Two kernels:
      1. `flashAttentionVecParamsKernelV11` — partial: per-split (max, sum, vkq).
      2. `flashAttentionVecCombineKernel`   — combine: cross-split softmax merge.

    Inputs:
    - `qBuf`           — f32, [numHeads * headDim]
    - `kCacheF16Buf`   — u32 (packed half2), [numKVHeads * maxSeqLen * headDim / 2]
    - `vCacheF16Buf`   — u32 (packed half2), [numKVHeads * maxSeqLen * headDim / 2]
    - `paramsBuf`      — u32[2] = [pos, cacheLen]
    - `partialOutBuf`  — f32, [numHeads * numSplits * headDim]   (state.flashPartialOutV11)
    - `partialMetaBuf` — f32, [numHeads * numSplits * 2]         (state.flashPartialMetaV11)
    - `outputBuf`      — f32, [numHeads * headDim]               (state.attnOutBuf)
-/
def executeFlashAttentionV11 [GPUBackend β] (ctx : β)
    (qBuf kCacheF16Buf vCacheF16Buf paramsBuf
     partialOutBuf partialMetaBuf outputBuf : GPUBackend.Buf β)
    (numHeads numKVHeads maxSeqLen headDim : Nat) (scale : Float)
    (kcrLookup :
       Option (UInt64 → IO (IO.Ref (Option (GPUBackend.CachedDispatch β)))) := none)
    (cacheLen : Nat := 8) : IO Unit := do
  -- numSplits must not exceed cacheLen: the partial kernel's split range
  -- [i·L/8, (i+1)·L/8) is EMPTY when L < 8, and an empty split skips the whole
  -- K loop — the epilogue then aggregates UNINITIALIZED threadgroup memory into
  -- partial_out/partial_meta (nondeterministic bit-wobble; usually near-benign
  -- because the leftover threadgroup contents are a previous dispatch's data —
  -- found via a 5-process md5 determinism probe at pos=6/cacheLen=7).
  -- For L ≥ 8 every split has ≥1 K position (⌊(i+1)L/8⌋−⌊iL/8⌋ ≥ ⌊L/8⌋ ≥ 1).
  let numSplits : Nat := min 8 (max cacheLen 1)
  let workgroupSize : Nat := 128

  -- Partial kernel: gridDim = (numHeads, numSplits, 1)
  let shaderP := flashAttentionVecParamsKernelV11
                   numHeads numKVHeads maxSeqLen headDim numSplits scale
  let namedBuffersP :=
    [ ("q",            qBuf)
    , ("k_cache_f16",  kCacheF16Buf)
    , ("v_cache_f16",  vCacheF16Buf)
    , ("partial_out",  partialOutBuf)
    , ("partial_meta", partialMetaBuf)
    , ("params",       paramsBuf) ]
  let execConfigP : Hesper.ExecConfig := {
    workgroupSize := { x := workgroupSize, y := 1, z := 1 }
    numWorkgroups := (numHeads, numSplits, 1)
    extensions := ["subgroups"]
  }
  let cacheKeyP : UInt64 :=
    hash ("flashV11Partial", numHeads, numKVHeads, maxSeqLen, headDim, numSplits)
  let refP ← match kcrLookup with
    | some lk => lk cacheKeyP
    | none    => IO.mkRef none
  GPUBackend.executeWithConfigCached ctx shaderP namedBuffersP execConfigP cacheKeyP refP

  -- Combine kernel: gridDim = (numHeads, 1, 1), workgroupSize = headDim
  -- (combine kernel assumes elemsPerThread = headDim / workgroupSize, with
  -- workgroupSize=128 and headDim=256 → 2 elems/thread).
  let shaderC := flashAttentionVecCombineKernel numHeads headDim numSplits
  let namedBuffersC :=
    [ ("partial_out",  partialOutBuf)
    , ("partial_meta", partialMetaBuf)
    , ("output",       outputBuf) ]
  let execConfigC : Hesper.ExecConfig := {
    workgroupSize := { x := workgroupSize, y := 1, z := 1 }
    numWorkgroups := (numHeads, 1, 1)
    extensions := ["subgroups"]
  }
  let cacheKeyC : UInt64 :=
    hash ("flashV11Combine", numHeads, headDim, numSplits)
  let refC ← match kcrLookup with
    | some lk => lk cacheKeyC
    | none    => IO.mkRef none
  GPUBackend.executeWithConfigCached ctx shaderC namedBuffersC execConfigC cacheKeyC refC


end Hesper.WGSL.FlashAttention

