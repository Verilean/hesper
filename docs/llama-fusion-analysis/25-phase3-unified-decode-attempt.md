# Phase 3 attempt log — unify decode via `forwardPrefillBatch(N=1)`

Date: 2026-04-20

## What was done

Added a `startPos : Nat := 0` parameter to `forwardPrefillBatch` and
wired `HESPER_UNIFIED_DECODE=1` so the decode loop dispatches via:

  forwardPrefillBatch ctx model #[nextToken] state
    (kcr := some kcr) (startPos := newPos)

instead of the dedicated `forwardSingleToken`.  `startPos` threads
into:
- `posBuf[i] = startPos + i`  (RoPE positions)
- `cacheLenBuf[i] = startPos + i + 1`
- `state.paramsBuf[0] = startPos.toUInt32` (for the batched attention
  kernel's absolute-position computation)

Commit: `ef134e3` (HESPER_UNIFIED_DECODE off by default, so not a
regression).

## Observed behaviour

`HESPER_DP4A=1 HESPER_CHAT=1 gemma4-cuda "What is 2+2?" -n 20`:

  non-unified: "The value of 2+2 is 4.<turn|>"   (12 tokens, correct)
  unified:     "The<turn|>"                       (2 tokens, broken)

So the first generated token ("The") is correct, but the second
(`<turn|>`, EOS) is wrong — real answer continues with " value of 2+2
is 4.".

This means **prefill logit on last prompt token is right**, but the
**first decode step (startPos = prompt_len, N = 1) produces logits
that put `<turn|>` on top**.

## Suspects

1. `forwardPrefillBatch` allocates the entire set of batch buffers
   (`batchBuf1`, `batchBuf2`, `batchQBuf`, etc.) at every call, then
   frees them at the end.  Decode calls this 100+ times.  Aside from
   the perf hit, repeated `cuMemAlloc`/`cuMemFree` of 4-32 MB buffers
   has been known to scramble launch ordering vs the previous call's
   outstanding work.
2. The tail of `forwardPrefillBatch` (final norm + lm_head) extracts
   `seqLen - 1` column from `currentBuf` via `columnExtractKernel`.
   For `seqLen = 1` that's column 0 — fine in principle, but the
   cached PTX variant keys include `seqLen` so cache hits / misses
   differ from the prefill path.  Any buffer reused across prefill
   (seqLen=N) and decode (seqLen=1) could trip a cache-key mismatch.
3. KV cache offset under `startPos > 0`: audited the batched RoPE-K +
   KV-write kernel — it uses `params[0] + col`, and we set `params[0]
   = startPos.toUInt32`, so slot `startPos + 0 = startPos`.  That is
   the correct absolute slot for a decode step.  Nothing obviously
   wrong here.
4. PLE / `out_scale` / `build_cvec` dumps inside `forwardPrefillBatch`
   may reference `seqLen * dim` buffers but only the first `dim`
   entries are valid for N=1.  If any kernel in that tail reads past
   index 0 it would consume uninitialised memory.

## Additional diagnostic data (2026-04-20 second pass)

Added a `HESPER_DECODE_TRACE=1` IO.println that prints
`(genCount, tokens.size before push, nextToken)` and
`HESPER_BATCH_PREFILL_FORCE=1` that allows using
`forwardPrefillBatch` for N=1 prompts.

Findings:

1. **forwardPrefillBatch is correct for N=1 with startPos=0**.
   Running `"H"` (1 prompt token) with and without
   `HESPER_BATCH_PREFILL_FORCE=1` produces the exact same output
   (`"PHHPH"`).  So the batched path at seqLen=1, startPos=0 reaches
   a bit-identical endpoint as the single-token path.

2. **First decode step (N=1, startPos=promptLen) is also correct**.
   Under `HESPER_UNIFIED_DECODE=1` for prompt "What is 2+2?" (18
   tokens after chat-wrap):

      genCount=0 nextToken=818    'The'  ← correct, matches non-unified

3. **Second decode step is where it breaks**.

      non-unified  step1: 1550 ' value'   ← correct
      unified      step1:  106 '<turn|>'  ← wrong (early EOS)

So the bug manifests specifically when `forwardPrefillBatch` is called
a *second* time with the same InferenceState and an incremented
`startPos`.  Something in the first call's tail either:
- leaves shared `state.xxx` buffers in a state that the batched path
  assumes is fresh (e.g. `state.paramsBuf`, `state.plModelProj`,
  `state.buf1`/`buf2`), OR
- releases a batch-scratch buffer that the second call still reads,
  OR
- trips a PTX cache key reuse between different call sites that
  happen to hash the same way.

## 2026-04-20 further narrowing: magnitude ≈ 1/2 of correct

Added logit dumps at step 1 under both paths and compared:

  unified step1 top-1:  <turn|>   logit = 21.62   (wrong argmax)
  single  step1 top-1:  ▁value    logit = 26.04   (correct)

  unified logits L2 norm: 2534.59
  single  logits L2 norm: 4710.03

**Unified produces logits ≈ half the magnitude of the single path.**
This is not noise — the distribution is structurally different, with
<turn|> ranked 1st in unified but below position 10 in single (where
▁value is 1st).

Implication: something is scaling the final hidden state by 0.5, or
missing a residual add.  Each block in Gemma 4 has two residual adds
(post-attention, post-FFN).  If one residual is missing across all
42 layers, the accumulated output would lose ~half its magnitude.

Also ruled out earlier:
- `forwardPrefillBatch` IS deterministic — calling it twice with same
  args yields bit-identical logits (`HESPER_DOUBLE_CALL=1` test).
  So no state-dirtying by the function itself.
- `kcr := none` doesn't fix the bug — the PTX cache is not the cause.

**Root cause is almost certainly a missing residual add or wrong
scale in forwardPrefillBatch's block loop, activated specifically when
startPos > 0.**  Maybe the PLE precompute path skips the residual,
or the attnResidualBuf / postAttnNorm ordering differs.

Next minimal test (for next session):
1. dumpGolden after attn-residual, after ffn-residual, and after
   post-PLE residual in forwardPrefillBatch, under HESPER_UNIFIED_DECODE.
2. Compare against same points in forwardSingleToken.
3. First divergence identifies the missing scale/residual.

## 2026-04-20 initial narrowing

With `HESPER_BATCH_PREFILL_FORCE=1 HESPER_UNIFIED_DECODE=1` for a
single-char prompt ("H") — wrapped by HESPER_CHAT into 11 tokens —
both prefill and decode use `forwardPrefillBatch`.  Result:

  [decode] genCount=0 nextToken=236777  'I'      ← first call, correct
  [decode] genCount=1 nextToken=106    '<turn|>' ← second call, wrong
  [Result] Decoded: I<turn|>

So the bug is:

  **The second call to `forwardPrefillBatch` on the same
  InferenceState produces wrong logits, independent of whether the
  first call was for N=11 or N=1.**

This is NOT specific to startPos vs 0 — it triggers as soon as
forwardPrefillBatch runs twice in a row on shared state.

## Root-cause candidates (ranked by likelihood)

(a) **Some `state.xxx` buffer is written in a way that's only correct
    on first call**.  The most likely suspects (shared between prefill
    tail and decode input preparation):
    - `state.plModelProj`, `state.plInputAll`, `state.plTokenSelected`
      — re-used per-token inside the batched PLE precompute loop
    - `state.buf1`, `state.buf2` — used by the last-token extract +
      final norm + lm_head tail
    - `state.tokenBuf` — used by both embedding lookup paths
    - `state.paramsBuf` — both paths write offset 0; single-token also
      writes offset 4 (cacheLen) that batched path doesn't clear.

(b) **Cached dispatch buffer-pointer pinning**: the CUDA backend's
    `executeWithConfigCached` resolves `args` fresh on every call
    (line 191 in `Hesper/Backend/CUDA.lean`), so buffer pointers
    changing between calls should be fine.  Confirmed NOT the cause.

(c) **Two caches colliding**: `kcr.getRef key` is a single shared map
    with `hash("gemma4_prefill_ce", name, config…)` as key.  If
    another call site (e.g. forwardBlock inside the ce) creates a
    different ref using the same key, cached refs could be wired to
    the wrong compiled dispatch.  Worth grepping for collisions.

(d) **KV cache corruption**: first call's RoPE-K+KV-write targets
    slot `startPos`.  If the write went to the wrong slot (e.g.
    scattered across multiple slots), the second call reads garbage
    from slot 18 via attention.  BUT: we verified KV cache write was
    correct in the unit tests, and the first call's *output* is
    correct.  If the KV write were wrong, the first call's lm_head
    logit would be wrong too.  Probably not the cause.

## Minimal next debug step

Dump `state.paramsBuf`, `state.buf1`, `state.buf2` at the *start* of
the second call and compare against their values at the *start* of
the first call.  The first difference localises the corruption.

Better still: compare the **intermediate tensors** (attn_out-0,
ffn_out-0, ..., result_norm) between call 1 and call 2.  Since the
input to call 2 is structurally the same as call 1 (one token, just
at different startPos), seeing which layer first diverges from its
expected value would pinpoint the corruption.

## Recommendation for next iteration

- **Move batch-buffer allocation to state**: add
  `InferenceState.batchBuf1 / batchBuf2 / batchQBuf / …` sized to
  `max(prefillSeqLen, 1)`.  At init, allocate the largest expected
  batch.  In `forwardPrefillBatch`, reuse instead of alloc.  This
  eliminates suspect (1) and makes unified decode viable
  perf-wise.
- **Audit the prefill tail (after the 42-block loop) for seqLen=1
  assumptions**: walk from "Step 3: Extract last token" down through
  PLE / out_scale / lm_head.  Trace every `state.xxx` write and make
  sure the slice bounds are correct for N=1.
- **Isolate which step produces the wrong logits**.  Add a golden
  dump at each stage (attn_out, ffn_out, per-layer embd, out_scale,
  result_norm, result_output) — we already have the dumpGolden
  infrastructure from the correctness campaign (commit 315c0c7 and
  friends) — and diff the first decode step's intermediates against
  llama.cpp's dump for a 6-token prompt + 1 generated token.

Once that lands, `HESPER_UNIFIED_DECODE=1` should match the current
`forwardSingleToken` output bit-for-bit.  From there the kernel-count
reduction benefit of Phase 3 can be measured (goal: 975 → ~500
kernels/token, which translates to ~100+ TPS at decode).

## Why stopping here is the right call

Forwarding Phase 3 now with the correctness regression open would
mean every subsequent change has to juggle two competing code paths.
The infrastructure commit (`ef134e3`) is small (46 lines) and off by
default, so future work can iterate on it without touching the default
path or risking the correctness tests.

Perf target 120 TPS remains open.  Current decode: 65.8 TPS with
`HESPER_CUDA_GRAPHS=1` (doc 23).  Phase 3, when correct, should move
this past 100 TPS in one step.
