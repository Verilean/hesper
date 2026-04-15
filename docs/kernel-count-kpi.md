# KPI: kernels/token

Tracked metric: **GPU kernel dispatches per output token**, measured via
`nsys stats --report cuda_gpu_kern_sum` on a 30-token decode of
`data/gemma-4-e4b-it-Q4_K_M.gguf` with prompt "Hello world".

Why this KPI: per-token walltime on single-token decode is dominated by
kernel launch + L2 interference overhead, not per-kernel compute.
Empirically, each kernel launch costs ~10 µs of wall-clock
(launch + minimum GPU residency); 1000 extra kernels/token ≈ 10 ms/token
≈ 50 TPS worth of headroom.

## Measured

| engine | date | total (30 tok) | **per-token** | vs llama.cpp |
|---|---|---:|---:|---:|
| llama.cpp CUDA | 2026-04-15 | 5,615 | **187** | 1.00× (target) |
| hesper (session start) | 2026-04-15 | 43,987 | 1,466 | 7.8× |
| hesper (Q6_K ffn_down dp4a) | " | — | ~1,400 est. | 7.5× |
| hesper (fused gate+up) | " | — | ~1,400 | 7.5× |
| hesper (fused KV) | " | 40,097 | 1,336 | 7.1× |
| hesper (fused Q/K/V share q8_1) | " | ~38,800 est. | ~1,295 | 6.9× |
| **hesper (fused post-norm)** | " | **36,827** | **1,227** | **6.6×** |

## Corresponding TPS

| engine | TPS | wall-clock 30 tok |
|---|---:|---:|
| llama.cpp CUDA | 115 | ~260 ms |
| llama.cpp Vulkan | 109 | ~275 ms |
| hesper (session start) | 31.6 | 949 ms |
| hesper (current) | 46.3 | 648 ms |

## Kernel count breakdown (llama.cpp, 30 tok, 187/token)

- `mul_mat_q<Q4_K, ncols=16>` × 306: prefill matmuls, 16 tokens per kernel
- `mul_mat_vec_q<Q4_K, ncols=4>` × 306: 4-token prefill
- `mul_mat_vec_q<Q4_K, ncols=2>` × 306: 2-token prefill
- `mul_mat_vec_q<Q4_K, ncols=1>` × 186: **single-token decode** (~6/token)
- `mul_mat_vec_q<Q6_K, ncols=1>` × 37: Q6_K (ffn_down + lm_head)
- `rms_norm_f32` × 498+86+67: **~21/token** RMSNorms
- `rope_neox` × 55: <2/token
- `flash_attn_ext` × 35+35+35+35: <5/token (one per layer?)
- `quantize_q8_1` × 301+306: input quantize (~20/token)
- Various k_bin_bcast, k_set_rows, soft_max, gelu, etc.

llama.cpp's **187/token** is mostly: 6 matmul-decode + 21 RMSNorms +
<5 FA + ~40 misc ≈ 70-80 per-layer-per-token operations in flight at
once, with layer pipelining.

## Remaining gap for hesper: 1227 → 187 = 1040 to go

Candidate sources still in hesper's 1227/token:
- Attention: Q/K/V + O = still ~4 dispatches × 42 layers = **168**
  - Possible: fuse O-projection into flash attention output write
- FFN: 1 fused gate+up + 1 ffn_down = **~84**
- RMSNorm (attnNorm, qNorm, kNorm, vNorm, ffnNorm) = ~**210**
  - Fuse attnNorm into fusedQKV quantize path (biggest remaining win)
- Per-layer embedding ops = ~**168** (per_layer_input_gate, proj, etc)
- KV cache write = **84**
  - Already fused K+V in hesper (1 kernel per layer)
- RoPE = **84**
- FlashAttention = ~**42**
- Residual add + norm = already fused to **~84** (from 168)
- Misc quantize/scale/copy = rest

**Fattest remaining blocks** to fuse:
1. **attnNorm + fusedQKV quantize** — move the pre-attn RMSNorm inside
   the Q8_1 quantize pass. Saves 42/token.
2. **qNorm + kNorm + vNorm (per-head RMSNorm)** — 3 kernels × 42 = 126
   currently. Could fold into a single "per-head qkvNorm" kernel that
   handles all three in parallel per layer. Saves 84/token.
3. **Attention output → residual → next-layer inputBuf** — the output
   of attention feeds directly into `forwardNormThenAdd`. There's room
   to fuse the attention output projection with the residual add.
4. **RoPE + KV cache write** — both operate on the same Q/K buffers
   in sequence; could be one kernel.

Even aggressive fusion unlikely to get below ~400/token without CUDA
graphs + persistent kernel architecture. llama.cpp's 187 benefits from
much-larger kernels (mul_mat_q does whole prefill rows in one call) and
from CUDA graphs in recent builds.

Last updated: 2026-04-15.
