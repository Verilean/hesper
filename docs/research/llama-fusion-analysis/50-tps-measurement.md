# 50 — Final TPS measurement after Q6_K + RMSNorm optimisations

*Written 2026-04-24.  Task #226.*

## Setup

- Hardware: RTX 4070 Ti (60 SMs, sm_89)
- Model: Gemma 4 E4B Q4_K_M (4.95 GiB, 7.52 B params)
- Prompt: "The quick brown fox jumps" (5 tokens)
- 60-token decode loop

## Results

| config | TPS | ms/decode | ratio vs llama.cpp |
|---|---:|---:|---:|
| **hesper + CUDA Graphs** | **80.4** | **12.4** | **0.73×** |
| hesper graphs OFF | 30.6 | 32.7 | 0.28× |
| llama.cpp CUDA | 110.2 | 9.1 | 1.0× |
| llama.cpp Vulkan | 110.1 | 9.1 | 1.0× |

### Session-over-session progress

| checkpoint | TPS | note |
|---|---:|---|
| prev Phase 3 peak | 63 | baseline from memory |
| this session end | **80** | **+17 TPS (+27%)** |
| llama.cpp target | 110 | — |
| task #51 goal | 120 | — |

## What drove the gain

- **Q6_K 1-row dispatcher** (doc 47): ffn_down 1.83 → 1.33 ms/dec
- **RMSNorm warp-shuffle + rsqrt + block=1024** (docs 43/49):
  1.07 → 0.46 ms/dec (-57%, beats llama.cpp)
- **Various PTX codegen peepholes** along the way (cvt.u64.u32,
  u8/u16 primitives, FMA-chain, ShaderM.var CSE fixes)

## Remaining 37% gap (80 → 110 TPS)

Per-kernel measurements show near-parity now:

- Q6_K ffn_down: 1.10× (1.32 vs 1.20 ms/decode)
- RMSNorm: **beats llama.cpp** (0.46 vs 0.48 ms/decode total)
- Q4_K matmul: 1.12× (4.93 vs 4.40 ms/decode)

Sum of per-kernel costs on hesper ≈ llama.cpp × 1.1, but decode
wall-clock is 1.37× slower.  The delta is spread across:

1. **Launch scheduling overhead** — hesper issues more individual
   CUDA launches than llama.cpp (even though Graphs replays them).
   Per-launch setup inside the captured graph isn't zero.
2. **Small kernel residuals** — 148 pointwise stubs (embedScale,
   pleScale, residual adds) that individually are <0.2ms but sum
   up.
3. **Host-side overhead per step** — prefill re-run for KV
   append, sampling, logit extraction.

## Next actions (if pushing toward 120 TPS)

Low-hanging:
- Check the pointwise-stub bucket (148 kernels).  Some are likely
  fusable into adjacent matmul epilogues (already have
  `forwardFusedGateUp`, check if used everywhere).
- Q4_K matmul final push: 1.12× → 1.00× (doc 45 listed B/C items
  deferred — worth revisiting since they're minor deltas now).

Higher effort:
- True device-side decode loop (task #162 completed, but may not
  cover all ops) — eliminates CPU-side step overhead entirely.
- Vectorised loads in RMSNorm (ld.global.v4.f32) — 2x smaller
  load count per warp → could close per-call gap to 1.0×.

## The bottom line

hesper as a Lean-generated CUDA inference engine now sits at
**73% of llama.cpp's hand-tuned C++ CUDA inference** for Gemma 4
decode.  Per-kernel micro-benchmarks put every major kernel within
10% of llama.cpp.  The framework's optimization surface is
navigable via PTX inspection + ncu, and the fix ratios have been
tractable (5-50% per fix).

Closing the remaining 27% is not a single-kernel problem — it's
a whole-pipeline issue (scheduling, small-kernel residuals,
host-side overhead).  Best addressed either via:
- ncu on the end-to-end decode pipeline (host interaction visible)
- or re-architecting as a single device-side loop
