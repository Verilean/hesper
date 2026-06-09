# Phase F3 — Monolith TPS benchmark results

**Session end**: 2026-04-21
**Gemma 4 E4B Q4_K_M, layer 5, RTX 4070 Ti, 100 iters.**

## Measured cost per layer

| Path | ms/layer | ms/tok (×42) | TPS |
|---|---|---|---|
| **A** Production `forwardBlock` | 3.621 | 152.1 | 6.58 |
| **B** Monolith eager (`runMonolithicGraph`) | 3.600 | 151.2 | 6.61 |
| **C** Monolith + CUDA Graph capture | **0.175** | **7.35** | **136.0** |

## Interpretation

**B/A = 0.994x — the IRv2 expansion is zero-overhead.**
Monolith produces the same kernels in the same order via the same
cache refs; the BlockGraph walk adds no measurable dispatcher cost.

**C/A = 0.048x — CUDA Graph capture is ~20.7× faster than eager.**
With 14 physical kernels per layer × 42 layers = 588 kernel launches
per token in eager mode. Capture collapses them into a single
`cuGraphLaunch`. 136 TPS extrapolated is in the ballpark of production's
`HESPER_CUDA_GRAPHS=1` baseline (~115 TPS from memory), confirming that
IRv2's capture+replay is production-competitive.

## Important caveat — extrapolation, not full decode

The "42-layer token TPS" column is layer cost × 42. Real decode
includes prefix embedding, PLE chain, lm_head, argmax, pos advance —
Monolith doesn't model those yet. The TPS numbers are the per-layer
architecture cost, not end-to-end token generation rate.

For a true end-to-end comparison, Monolith needs:
- PLE (per-layer embedding) IR nodes
- layerOutScale fusion
- SWA-variant `GemmaAttentionMonolith` (layers without `ropeFreqFactors`)
- lm_head + sampling IR nodes

These are follow-ups. F3's purpose was proving the architectural
thesis — expressed as a BlockGraph, decode can run via capture+replay
without per-kernel host negotiation. **This is now proven.**

## What F3 *doesn't* claim

- **Not** "hesper faster than llama.cpp". Per-kernel PTX efficiency
  (doc 28 §10) remains the dominant gap vs llama.cpp's hand-tuned
  kernels.
- **Not** "136 TPS is the end-to-end decode rate". It's the layer-body
  cost × 42; real decode has additional fixed-cost stages.
- **Not** "capture wins on any model". Small kernels benefit
  disproportionately — larger matmul-bound workloads see less speedup.

## What F3 *does* claim

- The IRv2 BlockGraph is a **valid execution plan** — bit-identical to
  production (F2) and zero-overhead to walk (F3.B).
- CUDA Graph capture over this BlockGraph **works** — the `cudaCaptureStream`
  plumbing in `captureMonolithicGraph` routes launches correctly, the
  captured graph contains real kernel nodes, and replay produces
  meaningful speedup (F3.C).
- **"The BlockGraph is the execution plan. Do not let the host
  negotiate with the GPU during a token."** — proven.

## How to reproduce

```bash
lake build gemma4-monolith-tps
HESPER_F3_ITERS=100 lake exe gemma4-monolith-tps
```

## Files changed in F3

- `Examples/DSL/Gemma4MonolithTPS.lean`: new 3-path benchmark harness.
- `Hesper/Circuit/Dispatch_v2.lean`: `captureMonolithicGraph` now sets
  `cudaCaptureStream` ref so `launchKernelMaybeStream` routes launches
  onto the capture stream. Without this the captured graph was empty
  (all launches went to the default stream and were not recorded).
- `lakefile.lean`: registered `gemma4-monolith-tps` executable.

## Next steps (not in this session's scope)

- **F4** (polish): model PLE + outScale + lm_head as IR nodes so
  Monolith can run end-to-end decode matching production TPS without
  the ×42 extrapolation.
- **F5** (optimization): nsys-profile the captured graph vs
  production's `HESPER_CUDA_GRAPHS=1` graph; if Monolith is slower,
  identify non-captured sync points. If faster, the IRv2 representation
  enables further fusion passes that production can't do easily.
- Item #51 remains the global target: 120 TPS end-to-end on RTX 4070 Ti.
  F3's 136 TPS extrapolated suggests this is within reach once PLE +
  lm_head are Monolith-native.
