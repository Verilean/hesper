---
title: "16 — Phase 1 shape audit: hesper forward kernels classified A/B/C"
date: 2026-04-19
status: architecture-plan (in progress)
---

# Phase 1 shape audit

Classification of every GPU kernel dispatched on hesper's Gemma 4
forward path, per the plan in doc 15.  Goal: identify exactly which
kernels block unifying `forwardSingleToken` with `forwardPrefillBatch`.

## Progress tracker

| Phase 2 item | Status | Commit | Measured impact |
|---|---|---|---|
| **1. Batched RMSNorm+residual** (post-attn + post-FFN) | ✅ done | `848c14f` | −2268 dispatches, prefill 290 → 250 ms |
| **2a. Bit-parity harness + RoPE-Q in-place RWW fix** | ✅ done | `4678951`, `c92e377` | Fixed the batched-vs-fallback Qroped divergence (was in-place RWW bug in `ropeWithFreqFactorsBatchKernel`); all 42 layers now BIT-IDENTICAL. |
| **2b. Unify SWA layers via 1.0-freq_factors** | ✅ done | `caa5c7d`, `04a5ef2` | After 2a unblocked, SWA layers now take the batched fast path by default.  Prefill 236 → 220 ms (−7%). 24/42 layers batched (was 4/42).  |
| **2c. Unify shared-KV layers** | ✅ done | `4e70a71` | Added `perHeadRMSNormBatchKernel` + shared-KV match arm.  All 42 layers now batched; fallback is dead code.  Prefill 220 → 216 ms. |
| **3. Batched PLE inner loop** | ✅ done | `1f85284` | −2150 dispatches, prefill 247 → 213 ms |
| 4. Drop `columnExtract` / `columnInsert` around batched paths | ⏳ partial | in items 1 & 3 | Deleted from post-attn/post-FFN + PLE.  Still present in per-token attn fallback (item 2). |
| 5. Pass pos/cacheLen arrays directly (skip `copyU32Kernel`) | ⏳ pending | — | Blocked on item 2 (only per-token loop needs this). |
| 6. Fuse layer output scale into preceding matmul | ⏳ pending | — | Independent; easy once items 1-3 stabilize. |

**Net so far**: −4418 dispatches/prefill, wall time 290 → 216 ms (−26 %).
All 42 layers now take the batched attention path (was 4/42).  Direct
A/B: default 216 ms vs `HESPER_FORCE_FALLBACK=1` 233 ms → 7% faster
on seqLen=9, scales with seqLen on longer prompts.  Multi-token
correctness still broken ("ucucuc.") — a different root cause than
the attention math (see doc 18 § "Correctness diagnostic"). 

## Current state (2026-04-19, post-`1f85284`)

**Measured on Gemma 4 E4B Q4_K_M, RTX 4070 Ti, `"Hello world how are
you"` (seqLen=9):**

| Metric | Original | Current | Change |
|---|---|---|---|
| Prefill wall | 290 ms | **213 ms** | −27 % |
| Decode TPS (single-token path) | 85 | **85** | unchanged |
| Multi-token correctness | `"are are…"` | `"stst{}}<tool_call|>…"` | still wrong (different wrong) |

**Live architecture ground truth (via new `HESPER_LAYER_PROFILE=1` probe,
commit `848c14f`):**

```
Gemma 4 E4B: 42 layers = 7 full-attn + 35 SWA + 18 shared-KV
             ropeFreq_layers = 7
```

Only **17 %** of layers (the 7 full-attention ones) currently enter the
batched attention fast path.  The other **83 %** — all 35 SWA layers and
all 18 shared-KV layers — still go through the per-token fallback at
Gemma4.lean:2936-3046.  That is why items 1 and 3 only brought prefill
from 290 ms to 213 ms: they optimized the things that happen *around*
the fallback, not the fallback itself.  The fallback is item 2.

**Correctness diagnostic** (see doc 17 for details):

- With `HESPER_SKIP_PLE=1`, both pre- and post-item-3 versions produce
  bit-identical garbage output — the pre-existing `"are are…"` bug is
  NOT in the code we've already refactored.  It lives in the
  interaction between the PLE path and the per-token attention
  fallback.  Item 2 is likely to move the needle on correctness as
  well as dispatch count.
- Naive attempt at item 2 (feed `freq_factors=1.0` to SWA layers so
  they share the existing batched kernel) was reverted: regressed both
  perf (250 → 4 222 ms prefill) and correctness (output shifted but
  did not recover).  Revised plan in doc 17 requires a bit-parity
  test harness first.


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
| ~~`RMSNorm.forwardNormThenAdd`~~ → `forwardNormThenAddBatch` | 3071 | `(seqLen, 1, 1)×256` | `[dim*seqLen]` batch | ✅ **A** | `848c14f` | **Done** (Phase 2 item 1).  Batched kernel `rmsNormThenAddBatchKernel` takes `[dim, seqLen]` col-major, 1 WG per row.  Deleted extract/insert wrappers. |
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
| ~~`RMSNorm.forwardNormThenAdd`~~ → `forwardNormThenAddBatch` | 3154 | `(seqLen, 1, 1)×256` | `[dim*seqLen]` batch | ✅ **A** | `848c14f` | **Done** (Phase 2 item 1).  Shared kernel with post-attn.  Both extract/insert wrappers deleted. |
| `columnInsertKernel` (nextBuf) | 3157 | `dispatch1D(dim)` | single-col | **B** | — | |

## Per-layer embedding (PLE) — per-token loop

The loop at Gemma4.lean:3188 runs `for i in [0:seqLen]` and dispatches
several single-token kernels per token.  Of everything, this was the
second-biggest structural C after the RMSNorm+residual.  **✅ Resolved
by Phase 2 item 3 (commit `1f85284`).**  The entire `for i in
[0:seqLen]` loop was replaced with 4 batched dispatches: `inpGate`
matmul (batched) → `geluGateMulSliceBatchKernel` → `proj` matmul
(batched) → `forwardNormThenAddBatch`.  All the column-extract /
column-insert wrappers in this section are gone.

| Kernel (old) | Replacement | Class | Notes |
|---|---|---|---|
| ~~`plInputAllExtract`~~ | folded into `geluGateMulSliceBatchKernel` (reads `per_layer_input[col*plTotalSize + plOffset + d]` directly) | ✅ **A** | Deleted. |
| ~~`columnExtractKernel` (nextBuf)~~ | none — batched matmul reads `nextBuf [dim, seqLen]` directly | ✅ **A** | Deleted. |
| ~~`LinearLayer.forward` (inp_gate)~~ | `forwardBatchDP4A(ple.inpGate, nextBuf, plGateBatchBuf, seqLen)` | ✅ **A** | `1f85284` |
| ~~`geluGateMulSliceKernel`~~ | new `geluGateMulSliceBatchKernel` (1D dispatch over `embdPerLayer*seqLen`) | ✅ **A** | `1f85284` |
| ~~`LinearLayer.forward` (proj)~~ | `forwardBatchDP4A(ple.proj, plMoeOutBatchBuf, plProjBatchBuf, seqLen)` | ✅ **A** | `1f85284` |
| ~~`fusedPerLayerPostKernel`~~ | `forwardNormThenAddBatch(ple.postNorm, plProj, nextBuf, nextBuf, seqLen)` | ✅ **A** | Shares the Phase 2 item 1 kernel. |
| ~~`columnInsertKernel` (back into nextBuf)~~ | none — `forwardNormThenAddBatch` writes directly to `nextBuf [dim, seqLen]` | ✅ **A** | Deleted. |

## Layer output scale (per-token loop)

| Kernel | Called at | Shape now | Buffer | Class | Batch variant? | Notes |
|---|---|---|---|---|---|---|
| `columnExtractKernel` + Circuit DSL scale + `columnInsertKernel` | 3243-3268 | Circuit DSL per-token | `[dim]` | **C** | — | Two options: (1) batch the Circuit DSL fusion with a seqLen broadcast; (2) fuse the scale into the *preceding* matmul epilogue (the Circuit `fuseMatmulEpilogue` pass already supports this pattern). Option (2) is the cleaner long-term fix. |

## Final norm + lm_head + `ggml_get_rows` equivalent

| Kernel | Called at | Shape now | Buffer | Class | Batch variant? | Notes |
|---|---|---|---|---|---|---|
| "Extract last column" + single-token final norm + single-token lm_head | ~3300-end | single-token | `[dim]` | **C**-in-context | — | This is already "correct" — it models llama.cpp's `ggml_get_rows` + final norm + lm_head on just one row. But because the buffer layout upstream is `[dim*seqLen]`, we need a **`selectRow(batch, i)` kernel** — conceptually identical to the existing `columnExtractKernel`.  Run once per forward, so not a hot path. |

## Summary — what actually needs writing

The audit reduced to **three** real refactor items.  Status after
commits `848c14f` (item 1), `1f85284` (item 3), and doc 17 (item 2
revised plan):

1. ✅ **Batched `RMSNorm + residual-add` kernel** — shape `[dim, seqLen]`,
   1 WG per row.  Unblocked both post-attn (Gemma4.lean:3056) and
   post-FFN (3134) hot loops.  **Commit `848c14f`.  −2268 dispatches,
   prefill 290 → 250 ms.**

2. 🟥 **Unify the two attention paths** — first attempt (feed
   `freq_factors=1.0` to SWA layers to share the existing batched
   kernel) regressed both perf (250 → 4,222 ms prefill) and
   correctness (`"are are"` → `"ATP"`), and was reverted.  Root cause
   analysis in doc 17: the SWA batched path does NOT have bit-parity
   with the SWA single-token fallback.  Revised plan: build a
   bit-parity test harness first, then write genuine batched-RoPE-
   no-freq and batched-Q-only kernels (not fake
   `freq_factors=1.0`), THEN delete the fallback.  Expected savings
   still −3780 dispatches.  **Still pending — biggest remaining lever.**

3. ✅ **Batch the PLE inner loop** — `per_layer_inp_gate` and
   `per_layer_proj` rewired to `forwardBatchDP4A`; new
   `geluGateMulSliceBatchKernel`; the PLE-post step reuses the kernel
   from item 1.  **Commit `1f85284`.  −2150 dispatches, prefill 247
   → 213 ms.**

Plus housekeeping (all B → delete-and-simplify):

4. ⏳ Drop all `columnExtractKernel` / `columnInsertKernel` calls that
   exist only to bridge batched buffers into single-token kernels.
   **Partially done** — removed around post-attn/post-FFN norm (item 1)
   and PLE (item 3).  Still present in the per-token attention
   fallback; deleted when item 2 lands.

5. ⏳ `copyU32Kernel` dispatches for pos/cacheLen — pass arrays directly
   to the batched RoPE/FA kernels (they already index `inp_pos[y]`).
   Blocked on item 2 (only per-token loop uses them).

6. ⏳ Layer output scale — fuse into the preceding matmul epilogue via
   the existing `Circuit.fuseMatmulEpilogue` pass.  Independent of
   item 2; easy follow-up.

## Projected dispatch counts after Phase 2

For `seqLen = 9, nLayers = 42`:

| Component | Original | Current (`1f85284`) | After item 2 | Source |
|---|---:|---:|---:|---|
| Embedding | 9 (+ scale) | 9 | 1 | B→A (housekeeping) |
| Attention inner loop (fallback) | ~90/layer → 3780 | 3780 | 3/layer → 126 | item 2 |
| Post-attn RMSNorm+residual | 27/layer → 1134 | **42** | 42 | ✅ item 1 |
| FFN | ~5/layer → 210 | 210 | 210 | already A |
| Post-FFN RMSNorm+residual | 27/layer → 1134 | **42** | 42 | ✅ item 1 |
| PLE inner loop | ~8·9/layer → 3024 | **~210** | ~210 | ✅ item 3 |
| Output scale | 3/layer → 126 | 126 | 0 (fused) | housekeeping 6 |
| Final norm + lm_head | ~5 | 5 | 5 | unchanged |
| **Total (prefill)** | **≈ 9420** | **≈ 4425** | **≈ 640** | items 1+3 done; item 2 pending |

**Progress: 53 % of dispatch reduction achieved** (9420 → 4425 so far;
target 640).  The remaining 3785 dispatches are almost entirely the
per-token attention fallback loop — i.e. exactly item 2.

llama.cpp hits ~20 dispatches/layer → 840 for the same shape.  Our
post-rewrite projection of ~640 is comparable (hesper has fewer ops
per-layer because of the PLE-precompute hoist already done, and no
`ggml_get_rows` bookkeeping).

## Next step — Phase 2

~~Start with item 1 (batched RMSNorm+residual)...~~  ✅ Done.
~~Follow up with item 3 (batched PLE)...~~  ✅ Done.

**Item 2 (attention path unification) is the remaining work.**  Per
doc 17, the naive "feed `freq_factors=1.0`" trick failed — SWA and
shared-KV layers have subtly different math than the existing batched
full-attention path.

### Concrete re-start plan (Phase 2 item 2)

Open these on next session:

- `Hesper/Models/Gemma4.lean:2880-3048` — entire attention block; batched
  fast path at 2891-2932, per-token fallback at 2935-3048.
- `docs/llama-fusion-analysis/17-phase2-item2-findings.md` — revised plan
  and the three correctness hypotheses from the last attempt.
- `docs/llama-fusion-analysis/15-llama-single-path.md` — the llama.cpp
  reference behaviour the unified path must match.

**Step 1 — Bit-parity harness** (smallest standalone deliverable).

Add an env-gated dump to `forwardPrefillBatch`:
- `HESPER_ATTN_DUMP=fallback` → before the per-token loop writes
  `columnInsertKernel` output at 3046, copy `state.attnOutBuf` to a
  host file named `attn_L{li}_t{i}_fallback.bin`.
- `HESPER_ATTN_DUMP=batched` → same dump point in the batched branch
  (after `flashAttentionBatchKernel` at 2929), column-extract each
  token's slice of `batchAttnOutBuf` into a scratch buf, dump with the
  same filename shape.
- Diff script (5 lines of Python or a small Lean test): `numpy.allclose`
  on every `attn_L{li}_t{i}_*.bin` pair for `li ∈ full-attn layers` —
  these should already match bit-identically (same kernels, same data).
  If they don't, our diff infrastructure itself is wrong and item 2's
  main work is blocked.

**Step 2 — Batched-RoPE-no-freq kernel** (once harness works for full-
attn).

In `Hesper/Models/Gemma4.lean`:
- Add `ropeBatchKernel (headDim numHeads seqLen : Nat) (ropeBase :
  Float)` — structural mirror of `ropeWithFreqFactorsBatchKernel` at
  line 369 but drop the `/ freqFactor` division.  Math should match
  `RoPE.ropeKernelDynamic` (single-token, full-layer variant).
- Add `ropeKAndCacheWriteBatchKernel_noFreq` — mirror of
  `fusedRopeKAndCacheWriteBatchKernel` at line 429, drop freq factor
  division.  Math matches `Attention.fusedCacheWriteKVKernel` chained
  after `RoPE.ropeKernelDynamic`.
- Wire SWA layers (`hasKV=true, ropeFreqFactors=none`) into the batched
  branch using these new kernels.  Gate behind `HESPER_UNIFY_ATTN=swa`
  env var so the fallback path is still the default until the harness
  greenlights it.
- Run the Step 1 harness with `HESPER_UNIFY_ATTN=swa` + one SWA layer.
  Expect bit-identical output.

**Step 3 — Flash-attention `nQueries > 1` audit** (open question from
doc 15).

Read `Hesper/WGSL/FlashAttention.lean` — does
`executeFlashAttentionTiled` (called at line 3032 for cacheLen > 32)
already handle a Q with shape `[head_dim, n_heads, n_tokens]` as in
llama.cpp's MMA-F16 prefill tile?  If yes, reuse as-is.  If no, the
batched `flashAttentionBatchKernel` at 2929 already dispatches grid
`(nHeads, seqLen, 1)` — audit whether it hits the same cacheLen > 32
branch correctly.  For `seqLen=9` prefill, `cacheLen` hits 9 max, so
the ≤32 branch covers it.

**Step 4 — Shared-KV layers**.

`hasKV=false` layers compute only Q, then FA against an earlier
layer's KV cache.  In the batched path:
- Skip K/V batch matmul and QKV-batch norm (use Q-only).  Need a
  `perHeadRMSNormBatchKernel` (mirror of `perHeadRMSNormKernel` at line
  502) — grid `(numHeads, seqLen, 1)`, one WG per (head, token).
- Dispatch batched RoPE-Q (already exists).
- Dispatch `flashAttentionBatchKernel` with `kvCache=state.kvCaches[kvLi]`
  where `kvLi = cfg.kvCacheLayer li` points at the earlier full-KV
  layer.
- Gate behind `HESPER_UNIFY_ATTN=swa,shared` (cumulative flag).

**Step 5 — Flip the default, delete the fallback**.

Once all three layer types (full / SWA / shared-KV) pass bit-parity
AND multi-token end-to-end produces sane output for `"Hello world how
are you"`, remove:
- The `if !handledByBatched then for i in [0:seqLen] do …` block at
  Gemma4.lean:2935-3048.
- All `columnExtractKernel` / `columnInsertKernel` calls that only
  existed to bridge batch buffers into that fallback.
- The two `copyU32Kernel` calls for `posBuf[i]` / `cacheLenBuf[i]` →
  `paramsBuf` at 2978-2981 and 3027-3030; the batched kernels already
  index `inp_pos[y]` directly.

### Success criteria

- `lake exe gemma4-cuda data/gemma-4-e4b-it-Q4_K_M.gguf "Hello world
  how are you" 30` produces a sensible continuation (any non-degenerate
  English), not `"are are…"` or `"stst{}}…"`.
- Prefill wall ≤ 100 ms (llama.cpp-parity target from doc 12).
- Decode TPS ≥ 85 (no regression on the decode path).
- `HESPER_LAYER_PROFILE=1` still reports 42 layers but the batched
  path now handles all 42, not just 7.

### Estimated effort

One working session — faster than the first attempt because the
harness from Step 1 tells us immediately which layer type broke.
Expected gain on success: −3780 dispatches/prefill; likely
restoration of multi-token correctness.

### Also unblocked once item 2 lands

- Housekeeping 4 (column extract/insert deletion in the attention path).
- Housekeeping 5 (`copyU32Kernel` deletion).
- PLE numerics investigation: the `"are are…"` vs `"stst{}}…"` divergence
  after item 3 will collapse into a single consistent output once the
  attention path is stable, making any residual PLE bug easier to
  isolate.
