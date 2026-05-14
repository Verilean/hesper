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

## DRAM + SASS evidence (2026-04-28)

Added DRAM throughput and full SASS dump comparison for hesper vs
llama Q4_K mmvq.  Both confirm memory-bound + structural waste:

**hesper Q4_K 4-warp gateup ncu** (production cubin
`fc0e594e5c785155.cubin`, kernel `k1387739930045770`):

| metric | hesper |
|---|---:|
| per-call (gpu time) | 78.8 µs |
| **dram bytes read / call** | **29.55 MB** |
| **L2 hit rate** | **6.4 %** |
| L1 hit rate | 74.0 % |
| sm throughput | 71.7 % of peak |
| stall_long_scoreboard | 33.1 % |
| registers/thread | 35 |
| warps active | 87.9 % |

Theoretical minimum DRAM = 1 row of Q4_K weights × N=8192 outputs =
**11.8 MB / call**.  hesper actually pulls 29.55 MB = **2.5× the
minimum**.  L2 hit only 6.4 % means the duplicated reads from the 16
threads sharing a Q4_K block are NOT being absorbed by L2 — they
hit DRAM.

**SASS diff (full kernel, sm_89, post-ptxas)**:

| op | hesper | llama | ratio |
|---|--:|--:|--:|
| total instructions | 3384 | 192 | **17.6×** |
| LDG.E (loads) | 116 | 13 | 8.9× |
| IDP.4A (dp4a) | 80 | 8 | 10× |
| FFMA | 243 | 5 | 49× |
| IMAD | 606 | 41 | 14.8× |
| LDS | 90 | 3 | 30× |
| IADD | 286 | 12 | 24× |

llama's 192-instruction kernel runs a small `for` loop ~10 times to
cover K=2560.  hesper's 3384 instructions for *2 outer iters*
implies the inner-iter body is mostly straight-line (BSSY/BSYNC
guards × 20+ `if blockInRange` enclosures, each with the full body
inlined).  ptxas has no loop body to compress; the duplicated loads
become 116 separate LDG.E instructions instead of llama's tight
loop with reuse.

**Verdict updated**: the gap is two compounding factors:

1. **Duplicate scale/dmin/qs reads** — 16 threads sharing a block
   each emit independent LDG.E (PTX 38 / iter, SASS 116 total).
   At only 6.4 % L2 hit, these miss into DRAM repeatedly.

2. **Lack of loop unrolling control** — hesper unrolls the
   if-guarded body into one large basic block; llama leaves it as a
   tight loop that ptxas schedules into 192 instructions reusing
   registers.  This compounds with the duplicate loads since each
   "duplicate" gets a fresh register and a fresh LDG.E.

## Implementation options for next session

**Option A**: smem broadcast of scale/dmin/qs (highest ROI).  Half-
warp lane 0 reads dmU32/sc0/sc1/sc2 (4 loads), writes to smem,
barrier, all 16 threads in the half-warp read from smem.  Eliminates
~25 of the 38 ld.global per iter.  Predicted: dram bytes 29.55 →
~13 MB, per-call 78 → ~35 µs, **+5–7 TPS**.

**Option A — attempted, reverted 2026-04-28**:
PTX dump confirmed the change worked at the codegen level:
ld.global 38 → 18 (-53 %), ld.shared 3 → 27, st.shared 1 → 5,
bar.sync 1 → 2.  But the kernel hung at runtime — `gemma4-cuda
"Hello" 30` never progressed past `[Prefill] Batched path`, even
with a 5-minute timeout and a fresh cubin cache.  263 cubins were
written (PTX JIT completed) so the hang is at GPU execution, not
JIT.

Likely cause: the `if (laneLow == 0)` block contains 4×
`readBuffer` + 4× `writeWorkgroup`, and ShaderM's lowering of those
inside a divergent `if_` may not be safe — the equivalent guarded
path was tested only with `subgroupAdd` or pure smem stores in
prior kernels (Q6_K), never with global loads.  The PTX looked
plausible but ptxas may have scheduled a load on a lane where the
predicate later returns false, leaving the dest register
uninitialised — and the subsequent `ShaderM.barrier` hangs because
some lanes never reach it (control flow divergence).

Reverted to baseline (87.2 TPS).  Re-attempt with Option B (vec4
LDG.E.128 — no divergent control flow) or write a microbench-only
kernel to isolate the if-guarded readBuffer behaviour first.

**Option B**: vec4.u32 ld.global (LDG.E.128).  hesper currently
emits 4× ld.global.u32 where llama emits 1× LDG.E.128.  Requires
vec4-typed readBuffer in ShaderM + CodeGen support.  Preliminary
check: V11 FlashAttn already uses `Exp.unpack4xU8` style vec4 — but
unclear if available for plain weight reads.  Predicted: similar
~5 TPS but more invasive.

**Option C**: rewrite as a tight K loop instead of fully-inlined
2-iter unroll (mirror llama's `for kbx = …; kbx < blocksPerRow;
kbx += 8`).  Requires ShaderM.loop with body preserving
register reuse — current implementation flattens it.  Riskier but
addresses the structural 17.6× SASS bloat.

A is the lowest-risk highest-evidence option.  Start there.

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
