---
title: "14 — Fresh nsys comparison: hesper vs llama.cpp after Q4_K fixes"
date: 2026-04-18
status: data
---

# nsys comparison at 70.9 TPS

After the full sequence of 2026-04-18 optimizations (Q4_K duplicate-work
removal, llama.cpp per-thread layout port, CUDA Graphs wiring,
flashAttention subgroup kernel, PLE pre-layer fusion, lm_head fusion)
hesper is at **70.9 TPS** on Gemma 4 E4B Q4_K_M / RTX 4070 Ti decode.
llama.cpp reference: **112 TPS** (same model, same hardware, `llama-cli
-ngl 99 -dev CUDA0`).

## Command

```bash
# hesper
HESPER_CUDA_GRAPHS=1 HESPER_DP4A=1 nsys profile -t cuda -s none \
  --cuda-memory-usage=false -o /tmp/hesper_current.nsys-rep --force-overwrite=true \
  ./.lake/build/bin/gemma4-cuda data/gemma-4-e4b-it-Q4_K_M.gguf "Hello world how are you" 30

# llama.cpp
nsys profile -t cuda -s none --cuda-memory-usage=false \
  -o /tmp/llamacpp.nsys-rep --force-overwrite=true \
  ./llama.cpp/build/bin/llama-cli -m data/gemma-4-e4b-it-Q4_K_M.gguf \
  -p "Hello world how are you" -n 30 -ngl 99 -dev CUDA0 -no-cnv
```

## Totals (30 decode tokens + 9-token prefill)

|                        | hesper       | llama.cpp    | ratio |
|------------------------|-------------:|-------------:|------:|
| Total GPU kernel time  | **114.07 ms**| **16.80 ms** | 6.8×  |
| Kernel instances       | 14,758       | 2,602        | 5.7×  |
| Avg kernel time        | 7.7 µs       | 6.5 µs       | 1.2×  |

## hesper top kernels

| kernel                  | total (ms) | instances | avg (µs) |
|-------------------------|-----------:|----------:|---------:|
| `main` (many tiny)      | 53.30      | 9,595     | 5.6      |
| `k_5827556345714019`    | 35.83      |   351     | **102.1**|
| `k_7031743127946451`    | 10.42      |   189     | 55.1     |
| `k_1061309516933780`    |  2.95      |   342     | 8.6      |
| `k_1790517551769375`    |  2.93      |   684     | 4.3      |

`k_5827556345714019` has grid=(10752, 1, 1), block=(32, 1, 1), 40 registers.
10752 = 42 × 256 (numLayers × embdPerLayer).  Likely the batched
`q6kTableRowDequantScaleKernel` or one of the PLE pointwise kernels
dispatched with grid=(totalPL, 1, 1).  At 102 µs × 351 instances it's
**the single biggest GPU cost in hesper decode** and should be the
first target for further investigation.

## llama.cpp top kernels

| kernel                                 | total (ms) | instances | avg (µs) |
|----------------------------------------|-----------:|----------:|---------:|
| `mul_mat_vec_q<Q4_K, 1, has_fusion=1>` | 5.39       |  84       | 64.2     |
| `mul_mat_vec_q<Q6_K, 1, 0>` (lm_head)  | 4.34       |  66       | 65.8     |
| `mul_mat_vec_q<Q4_K, 1, 0>`            | 3.21       | 368       | 8.7      |
| `rms_norm_f32<1024,1,1>` (fused scale) | 0.76       | 250       | 3.0      |
| `quantize_q8_1`                        | 0.59       | 602       | 1.0      |
| `rms_norm_f32<1024,1,0>`               | 0.45       | 172       | 2.6      |

## Blind spots in earlier analyses

Previous docs (`00-summary.md`, `11-perf-gap-root-cause.md`) stated the
gap was driven by:
1. "1040 dispatches/tok vs 200" — dispatch overhead
2. "Low occupancy / register pressure" — kernel internal inefficiency

After the recent work the dispatch count is still high (14,758
instances for the same run), but the **GPU kernel time ratio (6.8×) is
far worse than the dispatch-count ratio (5.7×)**.  So removing
dispatches alone won't close the gap — per-kernel GPU time itself is
~20% higher on average, and a few dominant kernels (the 102-µs
`k_5827556345714019` in particular) are much slower than anything in
llama.cpp's top-of-list.

## Actionable next targets

**UPDATE after identifying `k_5827556345714019`:** grid=(10752,1,1),
block=(32,1,1) matches `matMulTransposeF16BlockCoopKernel` dispatched
with `config.N = totalPL = 42 × 256 = 10752`.  The three call sites
(`Gemma4.lean:2780 / 3173 / 3449`) all compute a per-token F16 matmul
of shape `[1] × [hiddenSize] × [totalPL]` inside the PLE precompute.

The 351 instances come mostly from the **prefill path**, which loops
over `seqLen` prompt tokens inside each layer block and recomputes the
PLE matmul per (layer, token) pair:

  `forwardBatchPrefill / for li in blocks / for i in promptTokens`
      → 42 × 9 = 378 dispatches of the same f16 matmul

plus ~30 dispatches from 30 decode tokens, totalling ~408 → 351 observed.
So **~93% of the `k_5827556345714019` time is PREFILL cost**, not
decode.  The decode-path PLE f16 matmul runs only once per token and
contributes ~30 × 102µs = 3 ms to decode (not the 35.8 ms bulk).

### Revised decode-time breakdown

Total GPU kernel time: 114 ms over 30 decode tokens + 9-token prefill.
Pulling out the prefill-bound F16 matmul (~30-35 ms) leaves ~80 ms of
kernel work spread across decode.  80 ms / 30 tokens = **2.7 ms
kernel/tok**.  Wall time is 15.2 ms/tok, so **~82% of decode wall time
is host/driver/graph-replay overhead, not GPU kernel execution.**

### Targets

1. **Batch the PLE precompute across prompt tokens** (prefill only).
   The layer loop contains `for i in promptTokens` dispatching 9×
   single-column F16 matmuls.  Lifting PLE precomputation OUT of the
   block loop (compute once per token before the loop) reduces 42 × 9
   = 378 dispatches to 9 dispatches.  Better: batch across all 9 prompt
   tokens at once (M=9 instead of M=1), giving 1 dispatch.  Either way
   this is a prefill speedup not decode.

2. **Decode is dominated by non-kernel cost** (~12 ms/tok host/driver
   vs ~3 ms kernel).  Further GPU-level optimisation won't move
   decode TPS much; need to look at host-side graph launch overhead,
   Lean IO.Ref access during the decode loop, or fundamentally
   different pipelining.

3. **Reduce `main`-branded non-cached kernels** (9,595 instances × 5.6
   µs = 53 ms).  Even tiny per-call cost adds up.  Most are likely
   pointwise (GELU, scales, residual adds) that could be fused into
   neighbouring matmul epilogues.  But given point 2 above, this is
   probably not the biggest lever for decode TPS either.
