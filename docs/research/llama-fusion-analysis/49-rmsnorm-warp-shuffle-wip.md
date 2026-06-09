# 49 — RMSNorm warp-shuffle reduction: LANDED (parity + -13%)

*Written 2026-04-24.*

## TL;DR (FINAL — parity with llama.cpp achieved)

After 4 waves of optimisation, RMSNorm hits llama.cpp parity:

| stage | ms/dec | µs/call |
|---|---:|---:|
| start (tree reduction, block=256) | 1.07 | 5.6 |
| + warp-shuffle + rsqrt (block=256) | 0.93 | 4.87 |
| + rsqrt HW | 0.93 | 4.86 |
| **+ block=1024** (this session) | **0.46** | **2.43** |
| llama.cpp fusion OFF | 0.48 | 1.58 |

**hesper 0.46 ms/dec vs llama.cpp 0.48 ms/dec — hesper wins overall**,
even though per-call is still 1.54× slower.  Cumulative **−57%**
RMSNorm decode time.  Token parity preserved across all waves.

### Block size was the dominant factor

llama.cpp uses block=1024 (32 warps = full warp-slot occupancy on
1 SM) for 211 of its 302 RMSNorm calls; block=256 for the rest.
hesper had been hardcoded to block=256 (8 warps, 12.5% occupancy).

Switching skeleton dispatch sites to wgSize=1024 + relaxing
`shared_sum` back to `workgroupSize` slots (from the earlier
attempted 33-slot shrink that broke bounds for other code paths):
**2× improvement, parity with llama.cpp**.

User's pushback on "fusion was the action item" unblocked this:
the LlamaPath v2 stub experiment already showed call-count
reduction doesn't move TPS much.  Occupancy (warps per SM) was
the real lever.

## Roofline analysis

Both kernels are **launch-overhead bound, not HBM-bound**:

| | per-call | effective BW | % of 504 GB/s peak |
|---|---:|---:|---:|
| hesper | 4.89 µs | 4.1 GB/s | 0.8% |
| llama.cpp | 1.58 µs | 12.7 GB/s | 2.5% |

For a single-row 2560-d RMSNorm with `grid=(1,1,1) block=(256)` on
60-SM RTX 4070 Ti, only 1 SM runs, and each kernel does ~20KB work.
A well-tuned kernel takes ~1 µs just for launch overhead.  The
remaining 0.5-4 µs is compute time — hesper has ~4 µs of "extra" per
call but the PTX already matches llama.cpp structurally.

Suspicion: hesper's launch path (cuLaunchKernel arg packing, kernel
cache lookup, etc.) adds overhead per invocation.  With 190 calls
per decode × 3.3 µs extra = 627 µs/decode eaten by launch overhead.

## Critical correction from LlamaPath v2 stub experiment

User pointed out: "but stub 実験で ~200 dispatches was target, didn't
help TPS much".  If launch overhead isn't the dominant cost per
call, something else is.

**Revised finding: llama.cpp uses TWO RMSNorm variants**

From nsys on llama.cpp fusion OFF run:

| variant | calls/decode | µs/call |
|---|---:|---:|
| `rms_norm_f32<1024, 0, 0>` | 211 | 1.78 |
| `rms_norm_f32<256, 0, 0>` | 91 | 1.10 |

**llama.cpp uses block=1024 for the majority of its RMSNorms**.
hesper uses block=256 fixed.  Key consequence: block=1024 =
**32 warps** = **1 SM fully occupied** (33% theoretical occ).
hesper's block=256 = 8 warps = only 12.5% of warp slots.  When
`grid=(1,1,1)`, only 1 SM runs; with more warps on that SM it gets
more work done in the same time.

**Attempted fix in this session**: switch skeleton dispatch sites to
workgroupSize=1024.  Ran into `CUDA_ERROR_ILLEGAL_ADDRESS` because
`shared_sum` allocation was set to `max 32 (numWarps+1)` = 33 slots
(matching our warp-shuffle idiom), but some code paths still read
slot up to 512 from the old tree-reduction pattern.  **Requires
more careful smem sizing based on chosen reduction strategy**.

Reverted to wgSize=256 for this session.  Followup in task #225.

## Why porting llama.cpp's rms_norm via LlamaCppPTX is lower ROI here

Unlike Q6_K (where override gave us the critical 1.53× → 1.10×
finding from doc 41), RMSNorm override would tell us:
- kernel body is identical speed → **confirms host overhead**
- kernel body is faster → **invalidates PTX diff**, but doc 49
  shows PTX structurally matches llama.cpp

Porting the 27-parameter `rms_norm_f32<256, true, false>` ABI takes
1-2 hours and the expected answer is "launch overhead" which leads
to the same action anyway (reduce call count via fusion / CUDA
graphs, not kernel-body tuning).

**Recommended next step**: kernel fusion (`rmsNormThenAddKernel`
wider adoption) + CUDA Graphs (task #127 already in progress).
Both reduce per-decode RMSNorm launch count, which the roofline
shows is the actual bottleneck.

## Goal

Replace hesper's smem tree-reduction (log2(N)=8 bar.sync for N=256) with
llama.cpp's warp-shuffle + 1-barrier pattern.  PTX matches what we
want:

| metric | hesper before | hesper shuffle | llama.cpp |
|---|---:|---:|---:|
| PTX lines | 239 | **136** | 275 |
| shfl.sync.bfly | 0 | **10** | 10 |
| bar.sync | 9 | **2** | 1 |

Also made a concurrent win:

- Changed `div rms = sqrt(...)` + `x/rms` → `rsqrt + mul` for the
  normalize step (llama.cpp pattern).
- **-4% ms/dec for RMSNorm even with tree reduction retained**
  (1.07 → 1.03 ms).

## Status: shuffle variant hangs

The warp-shuffle `warpBlockSumReduce` helper in
`Hesper/Layers/RMSNorm.lean` generates the right PTX shape but
**hangs the GPU at runtime**.  Likely cause: warp-divergence with
`shfl.sync` inside predicated ifs.

First attempt had lane 0 of warp 0 enter an `if localIdx == 0`
around the second subgroupAdd — only 1 lane participates, the shfl
waits forever for the other 31.  Fixed to let all lanes run the
second reduce.

Second attempt still hangs.  Suspected remaining issue: the
`subgroupAdd` inside the helper is triggered for all 256 threads, but
the readWorkgroup at slot `localIdx` for lanes 32-255 reads
uninitialised slots (we only wrote slots 0..numWarps-1).  Select-
guard makes the value safe (zeroed), but the read itself might cause
issues with smem allocation size (shared_sum declared as `[256]` but
we only use 8 slots for warp sums + 1 broadcast slot).

Next debug steps:
1. Dump PTX and confirm the emit sequence matches the expected
   shuffle idiom exactly.
2. Run a tiny standalone test with just the RMSNorm kernel, not the
   full Gemma 4 decode, to isolate whether the hang is in RMSNorm or
   downstream.
3. Try `__syncwarp` (PTX bar.warp.sync) between steps if needed.

## What shipped

Kept the conservative tree-reduction path in `rmsNormFusedKernel` and
applied only the rsqrt+mul fix.  Token parity preserved.  The
`warpBlockSumReduce` helper exists but is not wired in.

Files:
- `Hesper/Layers/RMSNorm.lean` — added `warpBlockSumReduce` helper
  (not used by default), kept tree-reduction, switched to rsqrt+mul.
- `docs/llama-fusion-analysis/48-rmsnorm-investigation.md` — earlier
  analysis (updated after fusion-OFF experiment).
- `docs/llama-fusion-analysis/49-rmsnorm-warp-shuffle-wip.md` — this.

## How llama.cpp avoids the deadlock (key finding)

Looked at llama.cpp's PTX (`/tmp/lc_norm.ptx` around line 1320).
Pattern:

```
# ALL 32 lanes in every warp do the first reduce (no predication)
shfl.sync.bfly ... 16
shfl.sync.bfly ... 8
...

# lane != 0 skips smem write (SAFE — no shfl inside)
setp.ne.s32 %p11, %r11, 0
@%p11 bra $L_9
    st.shared.f32 ...
$L_9:
bar.sync 0

# lane > 7 loads zero (SAFE — just predicated smem load)
setp.gt.u32 %p12, %r11, 7
mov.f32 %f63, 0.0
@%p12 bra $L_11
    ld.shared.f32 %f63, ...
$L_11:

# ALL 256 threads (not just warp 0!) do the second reduce
# — the result for warps 1..7 is garbage but unused.
shfl.sync.bfly ... 16
shfl.sync.bfly ... 8
...
```

**Core insight**: *shfl is always unconditional*.  Divergence happens
only around smem R/W, never around shfl.  Warps 1-7 redundantly run
the second warp-reduce; result is thrown away but the shfls don't
hang.

## Root cause of the hang (fixed)

Fresh PTX dump revealed:

```
L1:
  bfe.u32 %r12, %r1, 0, 5      # laneId = tid & 31
  setp.eq.u32 %p1, %r12, 0     # lane == 0?
  @!%p1 bra L2                  # skip entire block if lane != 0
    shfl.sync.bfly.b32 %f7, %f6, 16, 31, 0xFFFFFFFF
    shfl.sync.bfly.b32 %f8, ...
    ...
    st.shared.f32 [slot], %f6
L2:
  bar.sync 0
```

**The shfls were inside `@!%p1 bra L2`** — only lane 0 of each warp
entered, so shfl.sync.bfly deadlocked waiting for the other 31 lanes.

**Reason**: `Exp.subgroupAdd partialSum` is a pure Exp.  Even though
Lean sees it at `let warpSum := Exp.subgroupAdd partialSum` outside
the if, PTX codegen inlines the shfls *at the use site* (inside the
`if laneId == 0` block).  Lean's `let` is not a ShaderM state write.

**Fix**: bind via `ShaderM.var` — that creates a named state entry
forcing materialisation at the declaration point:

```lean
-- Before (buggy)
let warpSum := Exp.subgroupAdd partialSum       -- pure Exp, inlined at use site
ShaderM.if_ (laneId == 0) do
  writeWorkgroup "shared_sum" warpId warpSum    -- shfl ends up here

-- After (correct)
let warpSumName ← ShaderM.var (.scalar .f32) (Exp.subgroupAdd partialSum)
let warpSum : Exp (.scalar .f32) := Exp.var warpSumName
ShaderM.if_ (laneId == 0) do
  writeWorkgroup "shared_sum" warpId warpSum    -- only the var lookup here
```

The same idiom fixed two other CSE-leakage issues in the same
kernel: `val = input[i]` was emitting 2 ld.global (once for `val*val`),
and `rsqrtRms` was re-computing div+sqrt+rcp inside the normalize
loop body.  Both needed explicit `ShaderM.var` binds.

## Remaining 3.1× per-call gap

| config | ms/dec | calls | µs/call |
|---|---:|---:|---:|
| hesper (this landing) | 0.93 | 190 | 4.87 |
| llama.cpp fusion OFF | 0.48 | 302 | **1.58** |

Possible remaining causes (not ncu-verified in this session):

1. **Loop unrolling**: hesper's strided loop iterates 10 times (dim
   2560 / block 256).  llama.cpp may have ptxas unroll further.
2. **Vectorised loads**: hesper uses `ld.global.f32` (4 B);
   llama.cpp rms_norm might use `ld.global.v4.f32` (16 B).
3. **`ntid` SASS tricks**: ptxas for 256-thread kernels may emit
   different SASS than our current path.
4. **Input is read twice**: once for sum-of-squares, once in
   normalize loop.  Could cache in smem (trade smem for global re-read).

Any of these is follow-up work.  The −13% win from this session ships.
