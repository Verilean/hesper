---
title: Circuit DSL — Open TODOs
date: 2026-04-16
---

# Circuit DSL — Open TODOs

Recorded so future work has a clear list. None of these are blocking
production decode today.

## Performance / closing the gap to llama.cpp CUDA

### TODO-P1: Per-kernel optimization of the top 3 hot kernels
- **Targets**: gate+up (122 µs), wO (95 µs), ffn_down (85 µs)
- **Bottlenecks identified by ncu**:
  - gate+up: 75% SM throughput, 89% occupancy — near saturation, marginal gains
  - wO: 61% occupancy, 0.89 waves/SM — memory dependency stall, low warp count for latency hiding
  - ffn_down: 50% No-Eligible, 0.95 eligible warps/scheduler — memory latency bound
- **Likely fix**: Level-4 hand-coded kernel with `cp.async` + double buffering.  Beyond what current DSL exposes.
- **Estimated effort**: 1-2 weeks per kernel, requires ShaderM extensions for `cp.async`.

### TODO-P2: Q4_K × f32 direct path (skip Q8_1 pre-quantize)
- llama.cpp Vulkan reads f32 input directly into Q4_K matmul; hesper still pre-quantizes to Q8_1
- Saves ~45 quantize dispatches per token + Q8_1-buffer round trip
- **Estimated effort**: 1 week (new shader template + wiring)

### TODO-P3: Gate-GLU Prim (parallel reduce fusion, llama.cpp Pattern B/C)
- Currently the FFN gate+up is a single hand-coded kernel
- Lifting to a `Prim.parallelMatmulQ4KGLU` would let DSL fusion handle it generically (so other models could reuse)
- Now feasible because Level-2 `warpSum` exists
- **Estimated saving**: −84 kernels/tok, but production gate+up is already optimal so wall-time gain is small
- **Estimated effort**: 3-5 days

### TODO-P4: Per-kernel autotuning DSL ("TVM-lite")
- Schedule parameters (rows-per-WG, threads-per-WG, smem size, unroll)
  baked into hand-coded kernels via template parameters; run once at
  build time to pick the best, cache result as JSON
- **Defer until**: multi-model support (Llama / Qwen / Mistral) or
  multi-hardware support (H100, MI300X) makes the schedule space
  large enough to justify the framework
- **Estimated effort**: 1-2 weeks (case A), 3-5 days (case B "minimal cache")

## DSL extensions

### TODO-D1: Async memory copy primitives
- `cp.async` (Ampere+) for prefetch/double-buffering inside kernels
- Needed for ffn_down memory-latency fix (TODO-P1)
- Affects ScalarExp / Lowering / Level-4 ShaderM API
- **Estimated effort**: 1 week (new ShaderM ops + 1 Prim variant)

### TODO-D2: Tensor core operations (`mma.sync`)
- 16×8×16 tile matmul as a single instruction
- Needed for prefill speedup (current task #19 still pending)
- Hard to fit into per-lane ScalarExp; needs a "tile group" abstraction
- **Estimated effort**: 2-3 weeks (new IR concept)

### TODO-D3: Online softmax / per-block mutable state
- Required to express FlashAttention in DSL (currently Level-4 only)
- Needs per-warp/per-block state passing across loop iterations
- **Estimated effort**: 2 weeks (new IR concept), low priority unless flash attention needs algorithmic changes

### TODO-D4: Multi-output for `reduceScatterEpilogue`
- Today: 1 reduction → N writes (single-output)
- Could extend to N parallel reductions or multi-output epilogues
- **Estimated effort**: 1 week, low priority (no current use case)

### TODO-D5: 2D / 3D dispatch grid in scatter
- Today's scatter is 1D (`outShape.numel`)
- Some patterns (per-head per-token batches) are more naturally 2D
- **Estimated effort**: 3-5 days

## Code quality / housekeeping

### TODO-C1: Dead code: `Prim.fusedKV` (legacy, only used by `mergeSameDispatch`)
- Now subsumed by general fusion patterns
- Could remove the special-case Prim and the corresponding pass
- **Estimated effort**: 1 day

### TODO-C2: DP4A chaining in PTX emitter
- ptxas-level inspection found `mov.u32 %rN, 0; dp4a a, b, 0` patterns
  that could be chained into `dp4a a, b, prev_acc`
- Would let llama.cpp's chained-dp4a pattern emerge from hesper's
  inner loops automatically
- **Estimated effort**: 3 days, marginal perf gain (per-instruction)

### TODO-C3: Single-consumer guard explicit in fusion passes
- llama.cpp's `ggml_can_fuse_ext` always checks "intermediate has
  exactly one consumer"; hesper passes rely on `protectedIds` to
  cover externals but the explicit count guard is missing
- Probably correct in practice, but defensive checking helps
- **Estimated effort**: 1 day

## Documentation

### TODO-DOC1: Add autotune section to tutorial
- Once TODO-P4 lands, document how users opt in/out
- Until then, leave as-is

### TODO-DOC2: Worked-out example: implementing a new model layer
- Tutorial covers primitives but not "here's how I'd write attention
  for a new architecture from scratch"
- **Estimated effort**: 2-3 days

## Out of scope (won't do without strong reason)

- **CUDA Graphs**: investigated; gain is < +1 TPS due to end-of-token sync
  *(2026-04-29 update: graphs ON by default since #252, +20 TPS for decode)*
- **Full TVM-style autotuning framework**: see TODO-P4 reasoning

## Active 2026-04-29

### TODO-MMQ: Q4_K MMQ kernel for prefill batched matmul
- Phase 1 skeleton landed; correctness pending. Targets 17.6× kernel
  speedup over 1-warp baseline at seqLen ≥ 8.
- See `docs/llama-fusion-analysis/31-mmq-port-plan.md` and
  `memory/project_mmq_phase1_parity_blocker.md`.
- **Tensor cores in DSL** (was deferred): now relevant — prefill matters.
  llama.cpp's MMQ has a `TURING_MMA_AVAILABLE` branch using `mma.sync` PTX.
  hesper would need `Inst.mma_sync_m16n8k16` or similar primitive.
  Not blocking Phase 1c (DP4A path is enough), but unlocks the next 2×
  on prefill once parity holds.

### TODO-DSL-BlockLayout: typed quantized buffer views
- Discovered as a real DSL gap during MMQ port (see
  `docs/shaderm-cuda-mapping.md` Section 3 + 6).
- Lean structures `Q8_1View`, `Q8_1MMQView`, `Q4_KView` with named fields
  (`.dsWord`, `.qs k`, `.scaleByteFor sb`) that lower to correct offsets.
- Prevents the silent layout-mismatch bug we hit in MMQ Phase 1.
- **Estimated effort**: 2-3 days (new types + 5-6 use-site refactors).
