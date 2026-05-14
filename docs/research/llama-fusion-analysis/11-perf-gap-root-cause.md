---
title: "11 — Performance gap root cause analysis (hesper 48 TPS vs llama.cpp 112 TPS)"
date: 2026-04-17
status: confirmed
---

# Root cause of the 2.3× TPS gap

## Summary

The gap is **NOT** caused by a single kernel-level deficiency.  ptxas
produces equally efficient machine code for both hesper and llama.cpp
(same register count, zero spill, same instruction ratios).  The gap is
a system-level effect: **more dispatches × slightly more instructions per
dispatch × host overhead**.

## Evidence

### Per-kernel efficiency: nearly identical

| Metric | hesper (gate+up fused) | llama.cpp Q4_K |
|---|---|---|
| Physical registers | 34 | 39 |
| Register spill | 0 | 0 |
| SASS instruction ratio (select/pred) | 6% | 4% |
| SASS instruction ratio (compute+mem) | 41% | 42% |
| SASS instruction ratio (addr/mov) | 53% | 54% |
| SASS per unfused matmul | ~404 | ~192 |
| Overhead factor per matmul | 1.45× | 1.0× |

The 1.45× is primarily from hesper's u32-indexed buffer reads requiring
more address calculation than llama.cpp's pointer arithmetic.

### Dispatch count: hesper 5× more

| | hesper | llama.cpp (estimated) |
|---|---|---|
| Kernel dispatches / token | **~1040** | ~200 |
| Unique kernel types | 88 | ~30 |

Many of hesper's 1040 dispatches are small pointwise ops (residual add,
softmax components, RoPE, etc.) that llama.cpp fuses into larger kernels
or handles with fewer operations.

### Time breakdown (5-token decode, RTX 4070 Ti)

| Component | Time | % of wall |
|---|---|---|
| GPU kernel execution | 81.4 ms (16.3 ms/tok) | 79% |
| cuLaunchKernel overhead | 6.9 ms (1.4 ms/tok) | 7% |
| Host processing (Lean IO) | ~10 ms (2 ms/tok) | 10% |
| cuCtxSynchronize (token readback) | ~5 ms (1 ms/tok) | 5% |
| **Total** | **~103 ms (20.5 ms/tok → 48.7 TPS)** | |

Target: 8.9 ms/tok (112 TPS).  Need to cut 11.6 ms/tok.

### Smem staging: was a problem, now fixed

The 4-warp kernel was using 11.5 KB smem → 67% occupancy.  Fixed by
removing smem staging (16 B → 100% occupancy).  No TPS improvement
because wO/ffn_down are only 22% of total kernel time.

### Scale unpack branchless rewrite: attempted, regressed

Replacing per-sub-block `Exp.select` with llama.cpp's `uint16_t`-based
pattern regressed TPS by -2.7%.  The 3-way `Exp.select` in the new
readScales16 helper was worse than the original.  Root cause: hesper
reads weights as u32 (not u16) so cannot avoid select for uint16 extraction.

## Conclusions

1. **Individual kernel optimization has diminishing returns.**
   The dp4a inner loop, scale unpack, and warp reduction are already
   efficient.  ptxas handles register allocation optimally.

2. **The gap is systemic: too many dispatches × per-dispatch overhead.**
   1040 dispatches/tok vs llama.cpp's ~200.  Each dispatch has ~1µs GPU
   gap + ~1.2µs host overhead = ~2.3 ms/tok wasted.

3. **Highest ROI path: reduce dispatch count to ~300/tok** via:
   - Fuse more pointwise ops into matmul epilogues (Circuit DSL already does some)
   - Fuse RMSNorm → Q8_1 quantize → matmul into single dispatch
   - Eliminate redundant barriers / sync points
   - Combine small pointwise kernels (residual add, etc.) into preceding kernel

4. **Host overhead (4 ms/tok) is secondary** but non-trivial.
   Lean's IO monad allocates ByteArray per cuLaunchKernel arg packing;
   this could be optimized with pre-allocated arg buffers.

## Next steps (priority order)

1. Profile llama.cpp for apple-to-apple dispatch count comparison
2. Identify the top-20 most-dispatched hesper kernels and categorize which
   can be fused into adjacent ops
3. Extend Circuit DSL fusion passes to cover more patterns
4. Consider CUDA Graphs for the decode loop (eliminates per-dispatch host overhead entirely)
