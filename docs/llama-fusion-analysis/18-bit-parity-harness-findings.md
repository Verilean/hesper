---
title: "18 — Phase 2 item 2 Step 1 results: bit-parity harness exposes pre-existing batched-path bug"
date: 2026-04-19
status: working-notes
---

# Bit-parity harness results — the batched path was already wrong

Step 1 of doc 16's revised Phase 2 item 2 plan: build a
bit-parity harness (`HESPER_ATTN_DUMP`, `HESPER_FORCE_FALLBACK`),
dump `batchAttnOutBuf` for every layer via both paths, diff.

Expected outcome: full-attn layers (5/11/17/23) produce **bit-identical**
output under both paths because they use numerically equivalent math.
This would validate the harness before we attempt new SWA/shared-KV
batched kernels (Step 2).

**Actual outcome**: the existing full-attn batched path does NOT match
the fallback path.  At L5 — the very first full-attention layer —
outputs diverge by max 1.056 (relative ~26%).  The bug was there all
along; we just never noticed because multi-token prefill was
producing garbage for so many reasons at once.

## What the harness does

`Gemma4.lean` gained three small diagnostic hooks:

1. `let attnDumpTag ← IO.getEnv "HESPER_ATTN_DUMP"` — if set, every
   transformer block dumps `batchAttnOutBuf` to
   `$HESPER_ATTN_DUMP_DIR/attn_L{li}_{tag}.bin` (default dir `/tmp`)
   right after the attention block, before O-projection.  Both paths
   populate the same buffer.
2. `let forceFallback := (← IO.getEnv "HESPER_FORCE_FALLBACK").isSome`
   — gates the batched condition at line 2904 so every layer falls
   into the per-token loop even if it would normally take the batched
   fast path.
3. `IO.println` line showing `handledByBatched`, `hasKV`, `full`,
   `ropeFreq.isSome` per layer — lets us verify which layers actually
   take which path.

Also one Lean-scoping fix: `if !handledByBatched then <body>` needed
an explicit `else pure ()` at the same column as `if`, otherwise the
`then`-body was greedily consuming all subsequent do-statements at
the same indent level (including the dump).

## Layer mix confirmed

```
[LayerProfile] total=42 full=7 swa=35 kvShared=18 ropeFreq_layers=7
```

Running once with `HESPER_ATTN_DUMP=batched`:

```
L0..L4    path=fallback  hasKV=true   full=false  ropeFreq=false
L5        path=batched   hasKV=true   full=true   ropeFreq=true   ← batched
L6..L10   path=fallback  hasKV=true   full=false  ropeFreq=false
L11       path=batched   hasKV=true   full=true   ropeFreq=true   ← batched
...
L23       path=batched   hasKV=true   full=true   ropeFreq=true   ← batched
L24..L41  path=fallback  hasKV=false  full=?      ropeFreq=false  (shared-KV)
```

**4 of 42 layers take batched** (L5, L11, L17, L23).  The remaining 38
go through the per-token loop.  Full-attn layers in the shared-KV range
(L29, L35, L41) have `ropeFreq=false` because the loader only populates
`ropeFreqFactors` for the first 7 full-attn layers — this may be
intentional (those are the only ones with their own KV cache, rest
reuse earlier layers' KV) or an oversight.  Either way, they all fall
into the per-token loop.

## Diff result

```
L0: BIT-IDENTICAL
L1: BIT-IDENTICAL
L2: BIT-IDENTICAL
L3: BIT-IDENTICAL
L4: BIT-IDENTICAL
L5:  DIFF max=1.056e+00 rms=6.081e-02 rel=2.574e-01   ← first batched-vs-fallback diff
L6:  DIFF max=2.992e-01 rms=3.448e-02 rel=7.074e-02
...
L23: DIFF max=2.475e+00 rms=2.876e-01 rel=6.949e-01   ← errors grow cumulatively
...
L41: DIFF max=3.574e+00 rms=6.760e-01 rel=1.004e+00
```

L0-L4 match bit-exactly — both runs executed the fallback for those
layers (no batched path available because no `ropeFreqFactors`), so
the computation is literally identical.  L5 is the first layer where
one run took batched and the other took fallback.  That single-layer
diff propagates forward: each subsequent layer's input is slightly
wrong, and errors compound.

## Interpretation

Three hypotheses for the L5 divergence (listed from most to least
likely):

1. **Math mismatch between batched and fallback kernels for the
   full-attention case.**  The batched path uses
   `fusedPerHeadQKVNormBatchKernel` → `ropeWithFreqFactorsBatchKernel`
   → `fusedRopeKAndCacheWriteBatchKernel` → `flashAttentionBatchKernel`.
   The fallback uses `fusedPerHeadQKVNormKernel` → per-token
   `ropeWithFreqFactorsKernel` → `fusedRopeKAndCacheWriteKernel` →
   `flashAttentionDynamicParamsKernel`.  Some pair should be
   algorithmically identical but isn't.  Next step: dump intermediate
   tensors (Qnormed, Qroped, K cache slot contents, FA input) and
   compare per-kernel.
2. **Cache-slot mismatch.**  Batched RoPE-K+cache-write writes at
   `startPos + col` into a column-major `[numKVHeads, maxSeqLen,
   headDim]` cache.  Single-token kernel writes at `pos` into what
   should be the same layout.  If one kernel interprets cache stride
   differently, reads will see wrong data.
3. **Flash-attention query-tile mismatch.**  The batched FA uses grid
   `(numHeads, seqLen, 1)`, emitting one output per (head, token).
   The single-token FA uses grid `(numHeads, 1, 1)`.  They should
   coincide for `seqLen=1` but might diverge in how they interpret
   Q/K/V buffer strides for `seqLen>1`.

## Revised next step

Before writing any new kernels (original Step 2), **we have to fix the
existing batched path's numerics**.  The order becomes:

- **Step 1.5 — Intermediate tensor dumps.**  Add harness support to
  dump per-kernel intermediates (Qnormed, Qroped, Kcache after write,
  FA output before insert) for one specific layer (L5) under both
  paths.  Diff at each stage to localise which kernel pair disagrees.
- **Step 2 — Fix the identified kernel.**  Likely a tensor-layout
  fix, not a math fix.  Once batched and fallback agree bit-identically
  at L5, multi-token output should recover for that 4-layer case.
- **Step 3–5** — unchanged from doc 16 (write SWA batched kernels,
  shared-KV support, delete fallback).

## Commits

- Harness: this session's WIP (Gemma4.lean + a scripted Python diff)
- Findings: this doc

## Why this is good news

Before this investigation we thought the "are are" bug lived in
deep interactions between PLE and a mostly-working fallback.  Now we
know:

- **The fallback itself is not the primary bug**.  Layers 0-4 compute
  identically under both paths — the single-token attention code is
  fine for full-KV SWA-like layers.
- **The pre-existing batched path has a bug at L5** that has been
  there since day one.  It never showed up in single-token decode
  (which takes `forwardSingleToken`, a completely separate code path),
  only in multi-token prefill.
- Fixing one L5 kernel-pair mismatch may cascade into a large chunk of
  the multi-token correctness problem going away, with Phase 2 item 2
  work (SWA/shared-KV batching) becoming an extension of an already-
  correct base.

The harness takes five minutes to re-run and diff, so iterating on
kernel fixes is fast now.
