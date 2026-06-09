---
title: "17 — Phase 2 item 2 first attempt: findings and revised plan"
date: 2026-04-19
status: working-notes
---

# Phase 2 item 2 — attention path unification — first attempt

Per doc 16 the plan was: provide an all-1.0 `freq_factors` buffer so SWA
layers can enter the existing batched attention fast path, then delete
the per-token fallback block (Gemma4.lean:2936-3046).  That's broken
and reverted.  Findings below; revised plan at the bottom.

## Ground truth: Gemma 4 E4B layer mix (measured)

Added `HESPER_LAYER_PROFILE=1` instrumentation.  For the shipped model:

```
total=42  full=7  swa=35  kvShared=18  ropeFreq_layers=7
```

Only **7 of 42 layers** (~17 %) currently take the batched attention
fast path.  **35 layers are SWA**, **18 of the last layers share KV
cache with earlier layers**.  So the per-token fallback is not an edge
case — it is the **dominant path** for prefill.

This also explains why prefill has stayed at ~250 ms post-RMSNorm-batch
despite "the batched path should handle most of it": it doesn't.  It
handles 7/42 layers.

## What went wrong with the all-ones `freq_factors` trick

Switching all `hasKV=true` layers (full + SWA) to the batched path by
feeding an all-1.0 `freq_factors` buffer produced:

- Prefill wall jumped 250 ms → 4,222 ms (16× slower).
- Decode dropped 66 TPS → 8 TPS (8× slower).
- Output changed from `"are are…"` to `"ATP ATP…"` — different wrong.

Both perf and correctness regressed.  The match-to-`none` branch still
handles the 18 `hasKV=false` layers via the per-token loop — that
didn't change — so the per-token loop still runs.  But *something else*
in the batched path for SWA layers is broken.

## Why the perf cliff

Didn't fully root-cause, but the most likely contributor is cache
replay: decode afterwards re-enters `forwardSingleToken`, but the
cached KV cache slots for the first 7 full-attn layers are **different**
when we also ran the batched RoPE-K+cache-write path on SWA layers
(different kernel variant, potentially different cache layout
assumptions).  The 8 TPS number suggests host-side resync storms more
than a real GPU slowdown — an 8× drop on a 120 ms/token baseline looks
like ~12 ms of extra `cuStreamSynchronize` per decode, consistent with
an invalidated PTX JIT cache forcing re-compile on every decode token.

That's conjecture.  Not worth chasing — the plan below avoids this
path entirely.

## Why the correctness regressed differently

Pre-existing bug outputs `"are are…"` — a specific token.  Post-change
outputs `"ATP ATP…"` — a different specific token.  Both are
degenerate collapses.  The fact they differ means the SWA batched
path math *is* different from the SWA fallback path math — but neither
is right.  Possibilities:

1. The batched `ropeWithFreqFactorsBatchKernel` divides theta by
   `freq_factor`.  When SWA layers' original `ropeKernelDynamic`
   produces theta with **no division**, feeding `1.0` should be
   equivalent.  But llama.cpp's `rope.cu` has separate code paths for
   `n_ctx_orig_used_yarn` vs non-yarn RoPE.  Our full-attn kernel may
   bake in the yarn correction implicitly; the SWA kernel may use a
   different base phase.  That would explain a phase shift but not the
   collapse.

2. The batched `fusedRopeKAndCacheWriteBatchKernel` writes KV cache at
   slots `startPos + col`.  Hesper's prefill uses the cache across
   consecutive `forward` calls; decode's starting pos has to pick up
   where prefill left off.  If the SWA cache layout the batched kernel
   writes doesn't match what the SWA decode-time FA reads, every
   subsequent decode step sees wrong cache.

3. `fusedRopeKAndCacheWriteBatchKernel` computes K at
   `startPos + col` (correct for prefill starting at 0) — but does it
   also compute V at the same slot, with the same layout as the SWA
   fallback's `fusedCacheWriteKVKernel`?  These two kernels were
   written independently.  Layout mismatch here would mean correct K
   reads but wrong V reads.

Conclusion: we don't have bit-parity between the SWA fallback and the
SWA batched path.  Before merging them, that parity has to be
established.

## Why the pre-existing bug ("are are")

The pre-existing fallback should, on paper, work.  It extracts Q from
`batchQBuf`, K/V from `batchKBuf`/`batchVBuf`, does per-token
qkvNorm → RoPE → KV-write → FA → insert.  No obvious stale-scratch
reuse.  So the bug is *not* the one doc 15 conjectured ("decode-shaped
scratches reused across tokens").  It's something subtler in the
single-token kernels' math or their interaction with the pos/cacheLen
params.

Likely suspects:

1. `copyU32Kernel seqLen 2 0` writing `posBuf[colIdxBuf[0]]` to
   `paramsBuf[0]` — `posBuf` was pre-uploaded with `posBuf[i] = i`,
   `cacheLenBuf[i] = i+1`.  If the KV cache was not zeroed before
   prefill and a previous session left state, token 0's FA would see
   garbage.  But hesper normally zero-initialises caches; need to
   verify.

2. The `cfg.hasKV li = true` branch at line 2958 computes QKV via the
   fused kernel.  For SWA layers the batched QKV norm produced one
   result already (discarded since fallback rewrites `state.qBuf2`
   etc).  So we do the QKV-norm twice — which is not wrong, just
   wasted — unless the batched version already wrote K/V into buffers
   that the fallback also writes to.  Buffer naming check needed.

3. The single-token `perHeadRMSNormKernel` at 2968 is called for
   `hasKV=false` layers.  It takes `state.qBuf` (populated by the
   column-extract).  For SWA shared-KV layers, the FA reads from an
   earlier layer's cache.  If the earlier layer was also SWA and its
   KV writes were in the still-buggy fallback path... cascading
   corruption.

## Revised plan

The doc-15 "single forward path" rewrite is the right north star but
needs a more careful approach than my first attempt.

**Step 1: Bit-parity test harness.** Before any restructuring, write
a standalone test that:

1. Runs one SWA layer's attention via the fallback path with seqLen=9,
   dumps `batchAttnOutBuf` to disk.
2. Runs the same layer's attention via the (proposed) batched path
   with the same inputs, dumps `batchAttnOutBuf`.
3. `numerical_diff` the two.

Without this, we cannot tell whether the unified path is
numerically correct, let alone whether it matches llama.cpp.

**Step 2: Isolate the pre-existing correctness bug.** Current
`"are are"` output means SOMETHING in the fallback is already wrong
for multi-token prefill.  Dump intermediate tensors from layer 0
(first token's Q after norm, first token's K after RoPE, first token's
KV cache slot) and compare against a golden single-token run that we
know works.  This is separate from item 2 and may need its own pass.

**Step 3: Implement a genuinely batched SWA path.** Not by faking
`freq_factors=1.0` — instead by writing:

- `ropeBatchKernel` (no-freq-factors variant) — a 30-line mirror of
  `ropeWithFreqFactorsBatchKernel` that drops the `/ freqFactor`
  division.
- Same for K+cache-write.

3× the new code but keeps bit-parity with the existing SWA
single-token path (which uses `ropeKernelDynamic` — same math).

**Step 4: Batch the shared-KV layers separately.**  These layers have
`hasKV=false`, so no K/V projections and no cache writes — just Q.
They can take a Q-only batched norm + batched RoPE-Q (already have
that) + batched FA against the earlier layer's KV cache.  Need a
batched FA that's compatible with `state.kvCaches[kvLi]` shape — the
existing `flashAttentionBatchKernel` might already work; need to
audit.

**Step 5: Only after 3 & 4 pass bit-parity**, delete the per-token
fallback.

## What's committed vs what's reverted

- **Committed (kept)**: `2f635d2` — batched RMSNorm+residual kernel.
  That's correct and a real -14 % prefill win with no regression.
- **Reverted**: the SWA-batched-path + `onesBuf` change.  Not
  committed.

## Takeaway

Phase 2 item 2 is not a one-commit fix.  It needs a bit-parity
test harness first, then three batched kernel variants, then deletion
of the fallback.  Approximate effort: a full working session, not an
hour.

For the next short session I'll switch to Phase 2 **item 3** (PLE loop
batching), which is contained — `per_layer_inp_gate` /
`per_layer_proj` matmuls just need to be called via `forwardBatchDP4A`
instead of `LinearLayer.forward`; the PLE-post kernel can reuse the
new `rmsNormThenAddBatchKernel`.
