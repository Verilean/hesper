---
title: "16 — Phase 1 shape audit: hesper forward kernels classified A/B/C"
date: 2026-04-19
status: architecture-plan
---

# Phase 1 shape audit

Classification of every GPU kernel dispatched on hesper's Gemma 4
forward path, per the plan in doc 15.  Goal: identify exactly which
kernels block unifying `forwardSingleToken` with `forwardPrefillBatch`.

Legend:
- **A** — already N-aware: accepts seqLen as a runtime/grid dim and
  produces correct output for N>1 without being called in a loop.
- **B** — trivially extendable: 1D dispatch today but data is
  contiguous and an extra `y = seqLen` grid axis would suffice.
- **C** — needs refactor: has single-token assumptions baked in
  (shape-hardcoded ShaderM body, implicit scratch-buffer reuse,
  shared-memory preloads tied to N=1, or a 1-WG=1-token design).

Line numbers are against `Hesper/Models/Gemma4.lean` on `main` after
commit `8e2e888` (redundant-HtoD cleanup).

## Embedding

| Kernel | Called at | Shape now | Buffer | Class | Batch variant? | Notes |
|---|---|---|---|---|---|---|
| `embeddingScaleKernel` | 2730 | `dispatch1D(dim*seqLen)` | `[dim*seqLen]` flat | **A** | itself | Scales all tokens in one dispatch. |
| `columnInsertKernel` | 2721 | `dispatch1D(dim)` × seqLen | `[dim]` → `[dim, seqLen]` col-major | **B** | — | Add y=seqLen and decode `(col, i)`. |

## QKV projection + norm (batched fast path)

| Kernel | Called at | Shape now | Buffer | Class | Batch variant? | Notes |
|---|---|---|---|---|---|---|
| `forwardBatchDP4A` (wQ / wK / wV) | 2866/2868/2869 | 2D `(seqLen, outDim/64)` | `[dim*seqLen]` batch | **A** | itself | |
| `fusedRMSNormQ8_1Kernel` | 2863 | `(seqLen, 1, 1)`×256 | `[dim*seqLen]` | **A** | itself | |
| `fusedPerHeadQKVNormBatchKernel` | 2904 | `(nHeads*seqLen, 3, 1)` | `[qDim*seqLen]`, `[kvDim*seqLen]` | **A** | itself | Grid y=3 covers Q/K/V simultaneously. |

## Attention — batched fast path (when `freq_factors` + full-attn)

| Kernel | Called at | Shape now | Buffer | Class | Batch variant? | Notes |
|---|---|---|---|---|---|---|
| `ropeWithFreqFactorsBatchKernel` | 2913 | `dispatch1D(nHeads*dimPairs*seqLen)` | `[qDim*seqLen]` | **A** | itself | In-place rotation. |
| `fusedRopeKAndCacheWriteBatchKernel` | 2920 | `dispatch1D(nKVHeads*dimPairs*seqLen)` | `[kvDim*seqLen]` + cache scatter | **A** | itself | K rotation + K/V cache write in one dispatch. |
| `flashAttentionBatchKernel` | 2929 | `(nHeads, seqLen, 1)` | `[qDim*seqLen]` + cache | **A** | itself | Processes all prompt tokens. |

## Attention — single-token fallback (SWA or no freq_factors)

This whole block lives inside the `for i in [0:seqLen]` loop at
Gemma4.lean:2936-3046.  Every kernel here is **C**-in-context even if
the kernel itself is fine — the *loop* is the problem.

| Kernel | Called at | Shape now | Buffer | Class | Batch variant? | Notes |
|---|---|---|---|---|---|---|
| `columnExtractKernel` (Q/K/V) | 2943/2950/2953 | `dispatch1D(qDim)` or `(kvDim)` | col from `[*·seqLen]` → `[*]` | **B** | — | Delete instead of batching — downstream kernels can read `[*·seqLen]` directly. |
| `fusedPerHeadQKVNormKernel` | 2960 | `(nHeads, 3, 1)` | `[headDim]` single-token | **C** | *batch form exists at 2904* | Route everything through the batched variant; delete this call site. |
| `perHeadRMSNormKernel` | 2968 | `(nHeads, 1, 1)` | `[nHeads*headDim]` | **C** | none | Needed only when layer has no K (Qcur-only). Add batch form OR share the QKV-batch kernel by passing null K/V. |
| `ropeWithFreqFactorsKernel` / `ropeKernelDynamic` | 2985/2991 | `dispatch1D(nHeads*dimPairs)` | `[qDim]` single-token | **C** | *batch form exists at 2913* | Use batched RoPE unconditionally once positions are an array. |
| `fusedRopeKAndCacheWriteKernel` | 3005 | `dispatch1D(kvDim)` | `[kvDim]` + cache | **C** | *batch form exists at 2920* | Batch form already handles per-token cache offsets via `inp_pos[y]`. |
| `fusedCacheWriteKVKernel` | 3018 | `dispatch1D(kvDim)` | `[kvDim]` + cache | **C** | none | No-freq_factors path; needs batch form or merge with batched RoPE-K path. |
| `copyU32Kernel` (pos/cacheLen → params) | 2977, 3026 | `(1,1,1)×1` | 4B | **C** | — | Pure dispatch overhead.  Eliminate by passing a `[seqLen]` pos/cacheLen array to the batched kernels directly. |
| `flashAttentionDynamicParamsKernel` (cacheLen ≤ 32) | 3035 | `(nHeads, 1, 1)` | `[qDim]` single-token | **C** | *batch form exists at 2929* | Remove cacheLen-32 gate or make the short-KV case fall through to the batched kernel. |
| `executeFlashAttentionTiled` (cacheLen > 32) | 3030 | varies | `[qDim]` | **C** | needs confirmation | Open question from doc 15: does this already accept Q with an `nQueries > 1` dim? |
| `columnInsertKernel` (attnOut) | 3044 | `dispatch1D(qDim)` | `[qDim]` → `[qDim, seqLen]` | **B** | — | Same fix as the Q/K/V extract: make downstream read batch buf directly. |

**Net**: if we route every layer through the batched-attention fast
path, this entire block (10 kernels × 9 tokens = 90 dispatches/layer)
collapses to 3 dispatches/layer.

## O projection

| Kernel | Called at | Shape now | Buffer | Class | Batch variant? | Notes |
|---|---|---|---|---|---|---|
| `forwardBatchDP4A` (wO) | 3049 | 2D `(seqLen, dim/64)` | `[qDim*seqLen]` | **A** | itself | |

## Post-attention norm + residual (per-token loop — hot)

The `for i in [0:seqLen]` loop at Gemma4.lean:3056 does extract → norm → insert, 3 dispatches × seqLen × 42 layers = 1134 dispatches/prefill.

| Kernel | Called at | Shape now | Buffer | Class | Batch variant? | Notes |
|---|---|---|---|---|---|---|
| `columnExtractKernel` (oProj, current) | 3060, 3065 | `dispatch1D(dim)` | col extract | **B** | — | Delete if batched norm reads `[dim*seqLen]` directly. |
| `RMSNorm.forwardNormThenAdd` | 3071 | hardcoded 1 WG = 1 row | `[dim]` | **C** | **none** | **Highest priority new kernel**: batched `RMSNorm+residual` taking `[dim, seqLen]`, 1 WG per row.  This unblocks both the post-attn and post-FFN loops. |
| `columnInsertKernel` (result) | 3074 | `dispatch1D(dim)` | single-col | **B** | — | Same — delete if the batched norm writes to `[dim*seqLen]` directly. |

## FFN

| Kernel | Called at | Shape now | Buffer | Class | Batch variant? | Notes |
|---|---|---|---|---|---|---|
| `fusedRMSNormQ8_1Kernel` (ffn_norm + quantize) | 3110 | `(seqLen, 1, 1)×256` | `[dim*seqLen]` | **A** | itself | |
| `forwardBatchDP4A_fromQ8` (gate, up) | 3113/3114 | 2D `(seqLen, interSize/64)` | `[interSize*seqLen]` | **A** | itself | |
| `geluMulKernel` | 3125 | `dispatch1D(interSize*seqLen)` | flat | **A** | itself | Pointwise; already batched. |
| `forwardBatchDP4A` (wDown) | 3130 | 2D `(seqLen, dim/64)` | `[dim*seqLen]` | **A** | itself | |

## Post-FFN norm + residual (per-token loop — hot)

Mirrors post-attn: 1134 dispatches/prefill today.  Unblocked by the
same batched `RMSNorm+residual` kernel.

| Kernel | Called at | Shape now | Buffer | Class | Batch variant? | Notes |
|---|---|---|---|---|---|---|
| `columnExtractKernel` (ffnOut, attnResid) | 3142, 3149 | `dispatch1D(dim)` | col extract | **B** | — | Delete once batched norm ships. |
| `RMSNorm.forwardNormThenAdd` | 3154 | 1 WG = 1 row | `[dim]` | **C** | **none** | Same kernel as post-attn. |
| `columnInsertKernel` (nextBuf) | 3157 | `dispatch1D(dim)` | single-col | **B** | — | |

## Per-layer embedding (PLE) — per-token loop

The loop at Gemma4.lean:3188 runs `for i in [0:seqLen]` and dispatches
several single-token kernels per token.  Of everything, this is the
second-biggest structural C after the RMSNorm+residual.

| Kernel | Called at | Shape now | Buffer | Class | Batch variant? | Notes |
|---|---|---|---|---|---|---|
| `plInputAllExtract` (column of `batchPLInputAll`) | 3195 | `dispatch1D(totalPL)` | `[seqLen*totalPL]` → `[totalPL]` | **B** | — | Add y-dim = seqLen; eliminate. |
| `columnExtractKernel` (nextBuf) | 3201 | `dispatch1D(dim)` | col extract | **B** | — | |
| `LinearLayer.forward` (per_layer_inp_gate) | ~3216 | single-token matmul `[dim] → [embdPerLayer]` | single-token | **C** | — | Use `forwardBatchDP4A` on `[dim*seqLen]` input — already exists. Just rewire. |
| `geluGateMulSliceKernel` | ~3218 | `dispatch1D(embdPerLayer)` | `[embdPerLayer*nLayers]` slice+mul | **B** | — | Pointwise; add y=seqLen. |
| `LinearLayer.forward` (per_layer_proj) | similar | single-token | single-token | **C** | — | Same: rewire to batched matmul. |
| `fusedPerLayerPostKernel` (norm+residual inside PLE) | ~3223 | `(1,1,1)×wgSize` | `[hiddenSize]` single-token | **C** | **none** | Subgroup-fused; hard-coded 1 WG = 1 token.  Either add y=seqLen (1 WG per row) or reuse the new batched RMSNorm+residual kernel with a different epilogue. |
| `columnInsertKernel` (back into nextBuf) | ~3222 | `dispatch1D(dim)` | col | **B** | — | |

## Layer output scale (per-token loop)

| Kernel | Called at | Shape now | Buffer | Class | Batch variant? | Notes |
|---|---|---|---|---|---|---|
| `columnExtractKernel` + Circuit DSL scale + `columnInsertKernel` | 3243-3268 | Circuit DSL per-token | `[dim]` | **C** | — | Two options: (1) batch the Circuit DSL fusion with a seqLen broadcast; (2) fuse the scale into the *preceding* matmul epilogue (the Circuit `fuseMatmulEpilogue` pass already supports this pattern). Option (2) is the cleaner long-term fix. |

## Final norm + lm_head + `ggml_get_rows` equivalent

| Kernel | Called at | Shape now | Buffer | Class | Batch variant? | Notes |
|---|---|---|---|---|---|---|
| "Extract last column" + single-token final norm + single-token lm_head | ~3300-end | single-token | `[dim]` | **C**-in-context | — | This is already "correct" — it models llama.cpp's `ggml_get_rows` + final norm + lm_head on just one row. But because the buffer layout upstream is `[dim*seqLen]`, we need a **`selectRow(batch, i)` kernel** — conceptually identical to the existing `columnExtractKernel`.  Run once per forward, so not a hot path. |

## Summary — what actually needs writing

The audit reduces to **three** real refactor items.  Everything else is
either already batched (A) or a trivial grid-extension (B) that falls
out of the refactor.

1. **Batched `RMSNorm + residual-add` kernel** — shape `[dim, seqLen]`,
   1 WG per row.  Unblocks both post-attn (Gemma4.lean:3056) and
   post-FFN (3134) hot loops.  Estimate: ~100 LoC ShaderM + wiring.
   Deletes ~2268 dispatches/prefill.

2. **Unify the two attention paths** — route every layer through the
   existing batched attention fast path (batch QKV-norm + batched RoPE
   + batched RoPE-K+cache-write + `flashAttentionBatchKernel`).  The
   single-token SWA / no-freq_factors branches need batch variants OR
   the SWA mask has to be folded into the batched path.  Estimate:
   mostly plumbing, 1 new batched-RoPE variant if SWA really needs a
   separate kernel.  Deletes ~3780 dispatches/prefill.

3. **Batch the PLE inner loop** — `per_layer_inp_gate` and
   `per_layer_proj` are already batch-capable through
   `forwardBatchDP4A`; the PLE-post kernel (`fusedPerLayerPostKernel`)
   is the only true C in this section.  Either add y=seqLen to it or
   route through the new batched RMSNorm+residual.  Deletes ~378
   dispatches/prefill.

Plus housekeeping (all B → delete-and-simplify):

4. Drop all `columnExtractKernel` / `columnInsertKernel` calls that
   exist only to bridge batched buffers into single-token kernels.
   Once (1)–(3) are done, nothing consumes the single-token views.

5. `copyU32Kernel` dispatches for pos/cacheLen — pass arrays directly
   to the batched RoPE/FA kernels (they already index `inp_pos[y]`).

6. Layer output scale — fuse into the preceding matmul epilogue via
   the existing `Circuit.fuseMatmulEpilogue` pass.

## Projected dispatch counts after Phase 2

For `seqLen = 9, nLayers = 42`:

| Component | Today | After rewrite | Source |
|---|---:|---:|---|
| Embedding | 9 (+ scale) | 1 | B→A |
| Attention inner loop (fallback) | ~90/layer → 3780 | 3/layer → 126 | Phase 2 item 2 |
| Post-attn RMSNorm+residual | 27/layer → 1134 | 1/layer → 42 | Phase 2 item 1 |
| FFN | ~5/layer → 210 | ~5/layer → 210 | already A |
| Post-FFN RMSNorm+residual | 27/layer → 1134 | 1/layer → 42 | Phase 2 item 1 |
| PLE inner loop | ~8·9/layer → 3024 | ~5/layer → 210 | Phase 2 item 3 |
| Output scale | 3/layer → 126 | 0 (fused) | housekeeping 6 |
| Final norm + lm_head | ~5 | ~5 | unchanged |
| **Total (prefill)** | **≈ 9420** | **≈ 640** | 15× fewer |

llama.cpp hits ~20 dispatches/layer → 840 for the same shape.  Our
post-rewrite projection of ~640 is comparable (hesper has fewer ops
per-layer because of the PLE-precompute hoist already done, and no
`ggml_get_rows` bookkeeping).

## Next step — Phase 2

Start with item 1 (batched RMSNorm+residual) — it's the smallest new
kernel, unlocks the biggest single-dispatch drop (1134 each for
post-attn/post-FFN = 2268 total), and is a useful test of whether the
refactor moves the multi-token correctness needle before we tackle the
thornier attention-path unification.
