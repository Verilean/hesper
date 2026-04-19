---
title: "19 — Phase 3 plan: locate the remaining multi-token correctness bug"
date: 2026-04-19
status: working-notes
---

# Phase 3 plan — where is "ucucuc." coming from?

Phase 2 (attention unification) is complete.  All 42 Gemma-4-E4B
layers now take the batched attention path, and the bit-parity harness
shows all 42 layers produce BIT-IDENTICAL `batchAttnOutBuf` between
batched and pure-fallback (see doc 18, commit sequence ending at
`4e70a71`).  Yet multi-token prefill output for `"Hello world how
are you"` remains `"ucucuc."` (repeating), same under both paths.

So the remaining bug is NOT in attention.  It's somewhere in:

- PLE (per-layer embedding) — the chain `inpGate → gelu*slice → proj →
  postNorm+residual`.
- Layer output scale (final per-token `y *= scale[0]`).
- Final RMSNorm + lm_head at end of prefill.
- State handoff to decode (decode then reads from prefill-populated
  KV caches).

## What we know

- `HESPER_FORCE_FALLBACK=1` also produces `"ucucuc."` → the bug is NOT
  in the batched kernels introduced in Phase 2 items 2a/2b/2c.  The
  pre-existing fallback pipeline itself gives this.  (This bug has
  been here since the original multi-token implementation —
  previously it surfaced as `"are are…"` or `"inteinteps…"` etc.
  depending on the state of the pipeline.)
- `HESPER_SKIP_PLE=1` produces completely different garbage
  (`"란aren好像…"`).  Not English but not degenerate.  So PLE is
  contributing a specific kind of wrongness.
- Single-token decode on a 1-token prompt produces `"HelloHello…"` —
  a separate pre-existing bug.

## Hypotheses (most likely first)

### H1. KV cache contents are wrong for tokens 1..8

Our bit-parity test only verifies `batchAttnOutBuf` matches between
batched and fallback.  That is, the *output of one attention layer*
agrees.  It does NOT verify the KV cache contents as read from disk
agree with a reference (llama.cpp).

If prefill writes wrong K/V for tokens 1..8 (perhaps due to an
off-by-one in pos-to-slot mapping, or a RoPE-K bug affecting the
cache layout), then:
- Prefill's forward pass through later layers reads these wrong K/V.
- Decode, which uses `forwardSingleToken` starting at pos=seqLen, ALSO
  reads these wrong K/V for positions 0..seqLen-1.  So even if the
  decode code itself is right, it's computing attention over
  garbage-keys.

### H2. PLE output residual chain is wrong

`forwardNormThenAddBatch` computes `output[bi] = (layer_out[bi] / rms)
* scale[i] + residual[bi]` with `layer_out = plProjBatchBuf`,
`residual = nextBuf`, `output = nextBuf`.  Single-token
`fusedPerLayerPostKernel` does `residual[d] += rmsNorm(proj)[d] *
weight[d]`, same math in principle.  Both should match; if they
don't, the math sign / order of operations differs subtly.

### H3. PLE input batching reads wrong offset

My batched `geluGateMulSliceBatchKernel` reads
`per_layer_input[col * plTotalSize + plOffset + d]`.  The buffer layout
for `batchPLInputAll` is `[seqLen * totalPL]` (token-major, since
`batchPLInputAll[i * totalPL + k] = plInputAll_tok_i[k]` per the
precompute kernel).  That matches.  But if the precompute itself
writes a different layout, the batched read would pull wrong bytes.

### H4. Layer output scale (outScale) has a batched bug

Applied via Circuit DSL per-token loop at Gemma4.lean:3392.  For
seqLen > 1, the same cached Circuit runs 9 times with the SAME (li,
dim) cache key but different `state.buf2` slice.  Possibly the
cached dispatch bindings get stale.

## Test plan

**Prerequisite**: free GPU memory (kill the `llama-cli` zombie at pid
2712157).  Then:

1. **Step 1 — Compare hesper's prefill KV cache to llama.cpp's for the
   same prompt.**
   - Dump `kvCaches[0..41].kBuf` at end of prefill via a new
     `HESPER_KVCACHE_DUMP=1` gate.
   - Use `llama.cpp/llama-cli` with `-dkvc` or a small C shim to dump
     the same tensors.
   - Diff.  If K/V for tokens 1..8 differs at ANY layer → H1.

2. **Step 2 — Stage-dump the PLE chain's intermediate tensors.**
   - Extend the existing `HESPER_ATTN_STAGE_LAYER` harness to dump
     `plGateBatchBuf`, `plMoeOutBatchBuf`, `plProjBatchBuf`, and
     `nextBuf` after each PLE step, for both batched path and the
     old single-token path (which we can reconstruct by running with
     `HESPER_PLE_FORCE_PERTOKEN=1` once we reintroduce the loop
     behind a flag).
   - Diff per-stage to localise any PLE divergence → H2/H3.

3. **Step 3 — A/B test outScale.**
   - Run once with the outScale loop.  Run once with it skipped
     (`HESPER_SKIP_OUTSCALE=1`).
   - If output is identical, outScale isn't the bug.  If it changes,
     outScale is suspect → H4.

4. **Step 4 — End-to-end parity against llama.cpp logits.**
   - For the final token of prefill, dump the pre-lm_head hidden
     state.
   - Compare against llama.cpp's same-position hidden state (via its
     `-ndebug` or similar).
   - If matches → bug is in lm_head or beyond (unlikely since lm_head
     is simple matmul).  If differs → bug is earlier; bisect by
     layer index of first divergence.

## Why this isn't just Phase 2 left over

Phase 2 proved the 42 per-layer attention outputs match between
batched and fallback.  But both paths could be WRONG in the same way
— the bit-parity test only shows equivalence, not correctness.  A
cross-check against llama.cpp is needed.  That's step 1.

## Next session kickoff

Exact sequence once GPU memory is available:

1. `git log -1` — confirm we're on `4e70a71` or later.
2. Add the `HESPER_KVCACHE_DUMP=1` gate in
   `forwardPrefillBatch` — dump all 42 `kvCaches[li].kBuf` and
   `vBuf` to `/tmp/kvdump/k_L{li}.bin` and `v_L{li}.bin`.
3. Build llama.cpp with a similar dump hook (or use a shim).
4. Run both on the same prompt.
5. Python diff to find the first diverging (layer, token, head)
   tuple.
6. That identifies the kernel at fault.

If step 5 shows no diff → bug is post-attention.  Step 2 then
bisects PLE / outScale / final-norm.

## Status summary for resumption

- Performance: prefill 220 ms (−54 % from 488 ms original), decode
  65 TPS stable.
- Correctness: wrong but harness-ready.
- Harness: `HESPER_ATTN_DUMP`, `HESPER_FORCE_FALLBACK`,
  `HESPER_ATTN_STAGE_LAYER`, `HESPER_ATTN_STAGE_TOKEN`,
  `HESPER_LAYER_PROFILE`, `HESPER_SKIP_PLE`, `HESPER_SKIP_OUTSCALE`
  all already wired.
- Dead code to clean up eventually: the per-token fallback loop at
  `Gemma4.lean:3079-3240` (not needed for any production layer; only
  runs under `HESPER_FORCE_FALLBACK=1`).
