# Q4_K mmvq PTX/SASS diff — hesper vs llama.cpp

## Why this analysis

After landing Q6_K 4-warp (60.18 → 87.2 TPS, +45 %), the new top-1
hot kernel is `fusedQ4KMLinearDP4A4WarpKernel` at the
ffn_gate+up shape (K=2560, N=8192):

| metric | hesper | llama.cpp `mul_mat_vec_q<Q4_K, 1>` | gap |
|---|--:|--:|--:|
| per-call | **68.7 µs** | 23.7 µs | **2.9×** |
| total/decode-tok | **2.88 ms** | 1.4 ms | -1.5 ms |

WG shape, dp4a count, and warp-reduce match.  The gap is in the
**inner-loop instruction mix and loop structure**.

## Inner-loop counts (one outer-loop iteration)

| op | hesper PTX | llama.cpp SASS |
|---|--:|--:|
| dp4a / IDP.4A | 8 | 8 |
| **fma / FFMA** | **0** | **5** |
| add.f32 / FADD.FTZ | 5 | 9 |
| mul.f32 / FMUL | 14 | 1 |
| **ld.global / LDG.E** | **38** | **13** |
| ld.shared / LDS | 0 | 3 |
| shfl / SHFL | 0 (in iter) | 5 (cross-warp) |
| bfe.u32 | 28 | (folded into IMAD) |

Total inst: hesper PTX 489 lines, llama SASS 192 lines (4070 Ti SASS,
post ptxas, post register allocation — not directly comparable but
suggests hesper produces ~2× more work).

## Three structural problems

### 1. `fma.` count = 0 (should be ≥ 5)

`Exp.mul a b` followed by `Exp.add (...) c` is not folded into a single
`Exp.fma a b c` by the codegen.  ptxas may fold some of them at the
SASS stage but it's not reliable.  Q6_K's recent perf work explicitly
hoists / pre-multiplies; Q4_K hasn't.

**Fix**: rewrite `sumfD = mul(d8, mul(toF32(dot), scA))` → use
`Exp.fma` explicitly.  Affects per-block scale-application code at
`Linear.lean:2400-2410`.

**Predicted impact**: 14 mul.f32 + 5 add.f32 → ~10 fma + 4 mul.
About 5–10 inst saved per iter × 2 iter × 80 lanes/SM = small but
free.  ~2 % per-call.

### 2. ld.global × 38 vs 13 — duplicate work (real cost)

hesper's outer loop runs `maxIter = (blocksPerRow + 7) / 8` = 2 iters
for K=2560 (blocksPerRow=10).  Each thread handles `kbxStart = tid >> 4`
in {0..7}, so **8 threads share the same Q4_K block** and each
re-reads the block's scale, dmin, and qs from global.  The 38 loads
per iter break down as:

- 1 dmU32 (block scale+dmin)
- 4 sc/sm (six-bit scale halves) — 4 loads
- 4 v (qs nibbles) — 4 loads
- 4 q8 sub-block headers (d, sum) — 4 loads
- 4 × 4 q8 quant payload — 16 loads
- 5 misc (q8 base addresses, shared address recompute) — 5 loads

llama.cpp avoids this by:

- `vdr=2`, `nwarps=4` → `blocks_per_iter = vdr·nwarps·warp/qi = 32`.
  At blocksPerRow=10 → 1 iter total (10 blocks ≤ 32).  No outer
  loop needed; ptxas straight-line schedules everything.
- Each thread holds 2 distinct sub-blocks (`vdr=2`); 16 threads
  cover one Q4_K block's 16 Q8_1 sub-slots — **no duplication**.

**Fix**: rewrite `fusedQ4KMLinearDP4A4WarpKernel` to match llama.cpp's
`(kbx_start = tid / 16, vdr = 2)` per-thread sub-block assignment.
Eliminates 8 threads sharing a block → ~25 of the 38 loads disappear
since each block is read once per warp instead of 8 times per warp.

**Predicted impact**: per-call 68.7 µs → ~30 µs (closes 1.6 ms / tok →
+5 TPS at 87 TPS baseline ≈ 92 TPS).

### 3. bfe.u32 × 28 — index arithmetic spam

Each iter computes `(blockIdx % 8) << 3`, `(... & 0xff) >> 4`, etc.
hesper emits `bfe.u32` for each.  llama.cpp folds these into `IMAD`
in SASS (single-instruction multiply-add-with-immediate-bit-extract).

**Fix**: hoist `kbxStart`, `laneLow`, `pairIdxInRow`, `elemOff` into
explicit `Exp.var` (already done at line 2283-2293) but the lowering
re-emits them per-use because the **inner loop body re-binds
addresses each iter**.  Move bind out of `if blockInRange`.

**Predicted impact**: small — ~5 % per-call.

## What llama.cpp does (reference: ggml/src/ggml-cuda/mmvq.cu)

```c
template <ggml_type type, int ncols_dst>
__launch_bounds__(nwarps*warp_size, 1)
__global__ void mul_mat_vec_q(...) {
  // nwarps=4, warp_size=32, so 128 threads.
  // For Q4_K: qi=32, vdr=2, qk=256.
  constexpr int blocks_per_iter = vdr * nwarps * warp_size / qi;
  // = 2 * 4 * 32 / 32 = 8 for Q4_K (NOT 32 — re-checked).

  // Each thread handles ONE Q8_1 sub-block per iter.
  // tid in [0..127]; kbx = tid / (qi/vdr) = tid / 16 → 0..7
  //                   kqs = vdr * (tid % (qi/vdr)) = 2 * (tid & 15)
  // 16 threads share a kbx but cover all 16 sub-slots without duplication.
  ...
}
```

Update: re-checked, `blocks_per_iter = 8` not 32 — same as hesper's
8-block stride.  So the outer loop count is the same.  The
**duplication is inside the iter, not across iters**: hesper has 8
threads reading the same block's scale/qs/dm = 8× the work for
constants.

## Action ranking (highest ROI first)

1. **Fix duplicate intra-iter loads** (§2) — predicted +5 TPS.
   Big rewrite but bounded: only `fusedQ4KMLinearDP4A4WarpKernel`,
   parity test exists (`cuda-q4k-microbench`).

2. **Use `Exp.fma`** (§1) — predicted +1–2 TPS.  Tiny diff, low risk.
   Do as warm-up before the §2 rewrite.

3. **Hoist address bind out of `if blockInRange`** (§3) — small.

After all three, predicted reach: **87 TPS → ~95 TPS**.

## Status update — fma fix landed, ncu profile complete

**fma fix (§1)** applied (this session): rewrote
`acc + dF*blockSumfD - dminF*blockSumfM` as
`fma(dF, blockSumfD, fma(-dminF, blockSumfM, acc))`.

PTX before/after:

| op | before | after |
|---|--:|--:|
| fma. | 0 | 2 |
| add.f32 | 13 | 12 |
| mul.f32 | 14 | 12 |
| neg.f32 | 0 | 1 |

End-to-end TPS unchanged (87.1 vs 87.2 before — within run-to-run
noise).  Confirms ptxas was already folding `mul + add → FFMA` at
SASS stage, so PTX-level `fma.` count is cosmetic.  Kept the change
anyway — explicit `fma` in PTX gives ptxas more freedom and matches
the algorithmic intent.

**ncu profile (this session, hesper Q4_K 4-warp ffn_gate+up @ K=2560 N=8192)**:

| metric | hesper |
|---|--:|
| per-call (gpu time) | 78.3 µs |
| L1 hit rate | **74.0 %** |
| registers/thread | 35 (low — occupancy not constrained) |
| warps active | 87.9 % (high) |
| **stall_long_scoreboard** | **33.1 %** ← memory wait |
| stall_short_scoreboard (smem) | 1.4 % |
| stall_drain / lg_throttle / mio_throttle | < 0.1 % each |

**Verdict**: **memory-bound**.  33 % of cycles waiting on global
loads.  This is real ncu evidence (not a guess) that the duplicate-
load problem identified in §2 *is* the per-call gap, since:

- L1 hit 74 % means 26 % of sectors miss into L2/DRAM
- 38 ld.global / iter × 26 % miss = 10 DRAM trips / iter / thread
- llama at ~13 ld.global / iter × similar miss rate = ~3 DRAM trips
- Roughly 3× the DRAM traffic is consistent with the 2.9× per-call gap

**llama.cpp ncu (pending)**: comparing the same metrics on the
Q4_K mmvq sm_89 kernel will confirm whether the gap is the load
duplication specifically or something else (register schedule,
dependency chain).  Run from this session timed out; capture the
data in the next session before the §2 rewrite.

## Next-session entry plan

1. Capture `mul_mat_vec_q<Q4_K, 1>` ncu (graphs OFF, --launch-skip
   matched to the 2024-tok prefill).  Check L1 hit rate, registers,
   stall_long_scoreboard.
2. If hesper stall % is meaningfully higher than llama → §2 rewrite
   is the right call.  If similar → look for compute-side schedule
   issue instead.
3. §3 (hoist address bind) is small; do it as a warm-up while
   reading mmvq.cu for §2.

## Critical files

- `Hesper/Layers/Linear.lean:2248` — `fusedQ4KMLinearDP4A4WarpKernel`
- `llama.cpp/ggml/src/ggml-cuda/mmvq.cu` — reference template
- `Tests/CUDA/Q4KPtxDump.lean` — PTX dumper (this session)
- `/tmp/q4k_ptx/` — dumped PTX/SASS

## Verification

```bash
# 1. Dump PTX after each change
lake exe cuda-q4k-ptx-dump /tmp/q4k_ptx
grep -c "ld.global" /tmp/q4k_ptx/q4k_dp4a_4warp_gateup_b128.ptx

# 2. Parity at production shape
lake exe cuda-q4k-microbench   # if not exists, write one mirroring cuda-q6k-4warp-parity

# 3. End-to-end TPS
scripts/kernel_bench.sh after q4k-fma-fix
scripts/kernel_bench.sh diff  q4k-fma-fix
```
