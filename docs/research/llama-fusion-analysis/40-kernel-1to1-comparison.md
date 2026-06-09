# 40 — Per-kernel-class 1:1 comparison (after PTX rename + lm_head swap)

*Written 2026-04-23.  Post commits `35da713` (PTX symbol rename) and
`8bd1bfb` (lm_head 4-row swap).  This is the first apples-to-apples
per-decode breakdown with real nsys numbers on both sides.*

## Setup

- hesper: `HESPER_DP4A=1 HESPER_LLAMA_GRAPHS=0` — graphs-OFF, 1 prefill
  seqLen=5 + 10 decodes.  Decode numbers = total / 10.
- llama.cpp: `GGML_CUDA_DISABLE_GRAPHS=1 llama-bench -p 0 -n 20 -r 1`
  — graphs-OFF, 20 decodes.  Decode numbers = total / 20.

**Decode wall time per token**: hesper **13.58 ms** (graphs OFF) vs
llama.cpp **8.38 ms** (graphs OFF) = **1.62× gap** now.

Gap was ~4.6× before this session (Q4_K widen + lm_head swap).

## Per-class comparison

| Kernel class              | hesper ms/decode | llama.cpp ms/decode | Ratio  | hs inst | lc inst |
|---------------------------|-----------------:|--------------------:|-------:|--------:|--------:|
| **Q4_K matmul**           |        **9.006** |           **4.404** | **2.0×** |   301 |   268 |
| **RMSNorm**               |        **2.724** |           **0.733** | **3.7×** |   812 |   302 |
| Q6_K lm_head              |           1.288  |              2.149  | 0.6×   |     1 |    33 |
| quantize_q8_1             |           0.312  |              0.298  | 1.0×   |   287 |   301 |
| residual/pointwise        |           0.150  |              0.104  | 1.4×   |    47 |    86 |
| **FlashAttention**        |       *(bundled)*|           **0.427** |   ?    |     ? |    85 |
| RoPE                      |       *(bundled)*|              0.070  |   ?    |     ? |    66 |
| KV cache write            |       *(bundled)*|              0.060  |   ?    |     ? |    48 |
| softmax                   |       *(bundled)*|              0.059  |   ?    |     ? |    42 |
| GELU/unary                |       *(bundled)*|              0.036  |   ?    |     ? |    42 |

## Headline findings

### 1. lm_head is now FASTER than llama.cpp (0.6×)

After the 4-row Q6_K swap, hesper runs lm_head in **1.29 ms** vs
llama.cpp **2.15 ms**.  llama.cpp dispatches it as many more small
kernels (33 inst vs hesper's 1).  This line item is closed.

### 2. Q4_K matmul is the biggest remaining gap: **2× slower (9.0 vs 4.4 ms)**

Most of hesper's Q4_K time is:
- **outDim=10240 (ffn_gate/up): 45 ms total / 10 = 4.5 ms/decode** — largest single bucket
- outDim=640 (wK+wV fused): 2.1 ms/decode
- outDim=2560 (wO/ffn_down): 1.5 ms/decode
- outDim=2048 (wQ): 0.45 ms/decode
- Various PLE: ~0.4 ms/decode

All already use `fusedQ4KMLinearDP4A4WarpKernel` (1 row × 4 warps).
The 2× gap is per-call latency: 58-60 µs hesper vs ~16 µs llama.cpp
at comparable shapes.  Candidates to close:
- Port the inline Q8_1 quantize into the matmul (removes the separate
  quantize dispatch and its smem round-trip)
- Try multi-row (2 or 4 rows per workgroup) for outDim ≤ 5120 — weight
  reuse within a workgroup
- ncu to find the actual stall reason

### 3. RMSNorm is 3.7× slower (2.72 vs 0.73 ms)

Hesper emits **812 RMSNorm-looking dispatches per decode** (under the
`b=256` classification = block=256 workgroups).  llama.cpp emits 302.
Even per-dispatch time is close (3.35 µs vs 2.43 µs), but the sheer
count is the problem.  Likely causes:
- Hesper emits per-head RMSNorm kernels + full-row RMSNorm + various
  "stub" kernels that got counted here
- llama.cpp fuses RMSNorm with `k_bin_bcast` (residual add) into one
  launch via its graph batcher
- Some of hesper's "RMSNorm" classification is probably pointwise
  stubs that got miscategorised

Need the per-grid classification to be tighter before attacking.
`gx=1` (12.7 ms over 3694 inst) is suspicious — those are 256-thread
single-workgroup kernels, probably stubs / scalars, not RMSNorm.

### 4. hesper is missing the FlashAttn / RoPE / KV-write / softmax classes

Those are bundled inside hesper's stub kernels and haven't been
separated out by classification yet.  Sum of those llama.cpp classes
= **0.65 ms/decode**, which fits inside the 2.7 ms "RMSNorm bucket"
ambiguity.

## Dispatch count comparison

| | hesper | llama.cpp | Ratio |
|---|---:|---:|---:|
| Total kernel dispatches / decode | ~1492 | ~1301 | 1.15× |

Dispatch count is already close (15% over).  The remaining gap is
per-kernel throughput, not launch count.

## Remaining gap decomposition (target: close 13.58 → ~8.4 ms)

| Lever | Est. savings | Notes |
|---|---:|---|
| **Q4_K matmul 2× → 1×** | **-4.5 ms** | Biggest single lever.  Inline-quant or multi-row. |
| **RMSNorm 3.7× → 1.5×** | **-1.6 ms** | Mostly dispatch-count reduction (fuse residual-add, per-head fuse) |
| Classify the bundled kernels | — | Pre-requisite to know what else to attack |

If we close just the Q4_K gap: **13.58 - 4.5 = 9.1 ms/decode → ~110
TPS projected** (at 1 graph launch overhead ~0.3 ms).

## What to work on next

**Q4_K matmul** is now the clear #1 — both in time (9 ms / decode,
66% of the remaining gap) AND actionable (single kernel, per-call
latency is 3.7× behind per-identical-shape).

Sub-priorities:
1. Measure actual per-call breakdown with ncu: memory-BW vs compute
   vs occupancy bound?
2. Port the `has_fusion=true` path from llama.cpp's `mmvq.cu`: inline
   Q8_1 quantize.  That also removes `quantize_q8_1` as a separate
   dispatch (~287/decode → 0), helping the bundled count.
3. Consider multi-row for outDim=10240.
