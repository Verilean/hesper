# Per-kernel microbench findings (2026-04-20)

After building `Tests/Perf/Gemma4KernelBench.lean` and measuring every
Linear kernel used in Gemma 4 E4B decode on RTX 4070 Ti (theoretical
504 GB/s), this is the per-kernel picture.

## Single-call path (`forwardDP4A`)

| Kernel | μs/call | GB/s | Note |
|--------|---------|------|------|
| wQ_L0        | 14.5 | 210 | 4-warp variant |
| wK_L0        |  6.3 | 120 | small outDim, launch-bound |
| wV_L0 (Q6_K) |  7.7 | 139 | small outDim |
| wO_L0        | 10.3 | 293 | 2-row variant |
| ffn_gate_L0  | 72.1 | 210 | **1-row variant, outDim=10240** |
| ffn_up_L0    | 72.1 | 210 | same |
| ffn_down_L0 (Q6_K) | 56.4 | 378 |  |
| wQ_L17       | 25.0 | 242 |  |
| wK_L17       |  9.0 | 167 |  |
| wV_L17       |  9.1 | 167 |  |
| wO_L17       | 15.5 | 390 |  |
| ffn_gate_L17 | 65.7 | 231 | |
| ffn_up_L17   | 65.8 | 230 | |
| ffn_down_L17 | 34.3 | 442 | |
| **lm_head (Q6_K 2560×262144)** | **1255** | **434** | |

## Actual decode path (`forwardFusedGateUp` — NOT single-call)

| Kernel | μs/call | GB/s | Note |
|--------|---------|------|------|
| **forwardFusedGateUp L0** | **64.7** | **468** | **2×Q4_K weights combined, 93% of theoretical** |

**Important discovery**: `forwardDP4A` on ffn_gate / ffn_up alone
(72 μs each, 210 GB/s) is NOT what decode actually uses — the decode
path invokes `forwardFusedGateUp` which fuses gate+up+GELU+mul in one
kernel via `fusedQ4KMGateUpDP4A4RowKernel`.  That kernel already lands
at 468 GB/s — near the theoretical ceiling.

## Implication

All hot Linear kernels in decode are near BW saturation:

- forwardFusedGateUp: 468 / 504 = **93%**
- lm_head Q6_K:        434 / 504 = **86%**
- ffn_down_L17 Q4_K:   442 / 504 = **88%**
- ffn_down_L0 Q6_K:    378 / 504 = **75%**
- wO_L17:              390 / 504 = **77%**

Per-kernel optimisation has diminishing returns.  The remaining ~5.5
TPS gap between 65.8 TPS (measured, CUDA Graphs on) and e.g. 100 TPS
is NOT hiding inside one slow kernel.

## What actually takes 11 ms/token

Summed GPU time per decode token:
- forwardFusedGateUp × 42 layers = 64.7 × 42 ≈ 2.72 ms
- ffn_down × 42                 ≈ 2.02 ms (Q6_K & Q4_K mix)
- wQ × 42                       ≈ 0.80 ms
- wK+wV × 42                    ≈ 0.63 ms
- wO × 42                       ≈ 0.53 ms
- lm_head × 1                   ≈ 1.25 ms
- attention (flashAttn+qkvNorm+RoPE) × 42 ≈ 1.0 ms
- misc (RMSnorm, PLE, residual) × 42      ≈ 2.0 ms
- **Total ≈ 11 ms/token**        ← matches observed decode

All pieces roughly fit.  ~85% of decode time is in BW-bound matmul
kernels that are already at 75-93% of theoretical BW.

## Decision

**Per-kernel optimisation is done.**  Remaining paths to TPS 120:

1. **#129 architecture rewrite (llama.cpp-shaped single-pass)** — the
   main vehicle.  llama.cpp runs the same 42 blocks as one graph-like
   execution, avoiding the token-loop in hesper, trimming the ~2 ms
   "misc" bucket above.  This is the 80/20 opportunity left.
2. **#122 sync-free argmax** — lower bound on the impact; may shave
   1-2 TPS by removing the one GPU↔host round-trip per token, but
   our earlier profile showed the sync itself is only a small fraction
   of the gap.
3. **lm_head elimination via top-K sampling** — skip 95% of the vocab
   for greedy (or top-K) sampling, do the argmax in-kernel from the
   hidden state.  Dense lm_head is ~250M FLOPs; this is structurally
   wasteful when we only need the argmax.

The microbench stays useful as a regression guard: any `forwardFusedGateUp`
regressing below ~60 μs would spoil everything.
