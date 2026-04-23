# 37 — First pass at Q4_K throughput: widen 4-warp, find real bottleneck

*Written 2026-04-23.  Started on priority #1 from doc 36 (Q4_K matmul
throughput).  Landed one small win and identified why the remaining
kernel-level tuning won't close the gap — it's dispatch count, not
per-kernel speed.*

## What changed

`Hesper/Layers/Linear.lean` — removed the `outDim ≤ 5120` gate on the
4-warp Q4_K kernel path.  ffn_gate and ffn_up (outDim=10240) now use
the 4-warp kernel instead of 1-warp.  Opt-out flag `HESPER_Q4K_4WARP_WIDE=0`.

Measured impact on the canonical 10-token workload:

| Metric                        | Before      | After       |
|-------------------------------|------------:|------------:|
| Total GPU kernel time (nsys)  | 151 ms      | **142 ms**  |
| ffn_gate/up kernel/call       | ~73 µs      | **~60 µs**  |
| Steady-state decode wall time | 58.9 ms     | 57.7 ms     |
| 20-token TPS                  | 11.20       | 10.90       |

Per-kernel time on ffn_gate/up moved correctly, but the wall-time
improvement is only ~1.2 ms/decode out of the ~50 ms/decode gap to
llama.cpp.  Why?

## The real bottleneck is dispatch count, not per-kernel throughput

Per-decode time budget (steady-state replay, 58 ms wall):

| Bucket                              | Time       |
|-------------------------------------|-----------:|
| Sum of all kernel active durations  | ~11–13 ms  |
| Host-side API calls                 | ~0.3 ms    |
| GPU-side scheduling gaps (idle)     | **~45 ms** |

That 45 ms is the cost of GPU-side kernel launch overhead *inside a
CUDA graph*: even a zero-work kernel costs ~30 µs from launch to
another launch-ready state on SM.  Hesper emits **1491 dispatches per
decode**; llama.cpp emits **89**.

    1491 × 30 µs = 45 ms    ✓ matches observed gap
      89 × 30 µs =  3 ms    → what llama.cpp "pays"

So even if we cut per-kernel GPU compute to zero, we'd still be stuck
at ~45 ms/decode because of dispatch count alone.  The 11–13 ms of
actual compute could hypothetically drop to match llama.cpp's
~8.3 ms, giving us ~50 ms/decode = 20 TPS.  Useful but not 115 TPS.

## Consequence for the priority list in doc 36

Doc 36 estimated that three per-kernel wins (Q4_K, FlashAttn,
RMSNorm) would drop per-decode to ~25.5 ms = 39 TPS.  That estimate
**over-counted** because it double-spent the gap: per-kernel
improvements only help the ~11 ms of active compute; they don't
shrink the 45 ms of idle gaps.

Realistic cap from pure per-kernel work: **~39–50 ms/decode = 20–26
TPS**, not 39 TPS.

## Revised plan

Two tracks, both needed:

1. **Dispatch reduction** (new top priority): fuse adjacent
   pointwise+norm+matmul kernels into single dispatches.  Targets:
   - fused RMSNorm + quantize_q8_1 + matmul (attn_norm→wQKV and
     ffn_norm→gate/up): -3 kernels per layer × 42 = -126 dispatches
   - fused post-norm + residual add: already partially done for
     postAttnNorm; apply to postFFNNorm and postPLENorm too
   - fused GLU (gate × gelu(up)) directly inside the next matmul's
     quantize step
   Estimated landing point: ~1000 dispatches/decode → ~30 ms gaps →
   ~40 ms/decode = 25 TPS.

2. **Per-kernel throughput** (continue where it helps):
   - The widened 4-warp Q4_K is committed (already landed).
   - FlashAttention vec-f16 decode port: 2.5 ms recoverable
   - Keep an eye on `main` kernel time — nsys collapses many kernels
     into that name; need HESPER_KERNEL_TRACE to resolve which rows
     are the hot stubs

Combined target: ~800 dispatches × 30 µs gap + 8 ms compute = ~32
ms/decode ≈ 31 TPS.  Still short of 115 but a clear direction.

## To reach 115 TPS eventually

Needs **graph-level kernel fusion / macro-kernels**: a single "layer
body" kernel that does RMSNorm → wQKV → RoPE → attention → wO →
postnorm → resid as one launch.  That's what llama.cpp's graph
batcher accomplishes at graph build time — its 89 ops/token represent
the *maximum* fusion it could extract with block-level rewriting.
We'd need the equivalent in hesper: a BlockGraph-level pass that
recognises kernel sequences and emits one `ShaderM` function per
block.

## What the canonical re-measurement should show after this commit

`docs/llama-fusion-analysis/35-measurement-recipe.md §A.2` — expect:

```
[decode 3] ... ~57 ms, 0 dispatches
...
tokens / sec (20-token): ~10.9
```

and total GPU kernel time from nsys §B.3 around 142 ms.
