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

## 2026-04-20 🎉 FIRST FIX: missing startPos in shared-KV path

Running with `HESPER_PREFILL_TWICE=1` (call forwardPrefillBatch twice
with the SAME args) produced correct output: "The value of 2+2 is 4."
This ruled out state corruption (state is fine when args are the same).

So the bug was in how the 2nd call's DIFFERENT args (startPos > 0) were
being propagated.  Audit found that line 3147 — the shared-KV batched
path (not the regular full-attn batched path) — was still writing
`state.paramsBuf[0] = 0` instead of `startPos.toUInt32`:

```lean
| some freqFactors, false =>
    ...
    GPUBackend.writeBufferOffset ctx state.paramsBuf 0
      (Hesper.WebGPU.BufferOps.uint32ToBytes 0)  ← was 0, fixed to startPos.toUInt32
```

Since Gemma 4 has full-attn and shared-KV layers interleaved, this meant
shared-KV layers attended with startPos=0 while full-attn layers used
startPos=18 — a layer-by-layer inconsistency.

### Immediate effect after fix:

| Prompt | Before | After |
|--------|--------|-------|
| "The capital of France is" | "The<turn\|>" | **"The capital of France is Paris."** ✓ |
| "What is 2+2?" | "The<turn\|>" | "The answer is the answer is the answer..." (looping) |
| "Write a poem about autumn." | — | "The amber, the amber, the amber..." (looping) |
| "Hello, how are you?" | — | "I'm doing well, thanks! I'm doing well, thanks! ..." (looping) |

The magnitude-halving is GONE — no more early EOS.  But prompts that
require multi-token coherent answers still loop, suggesting a similar
missing-startPos bug elsewhere (most likely another kernel reading
cached state from prefill time).

TPS under HESPER_UNIFIED_DECODE=1 is 9 TPS — much slower than single
path (39 TPS).  The slowdown is from `mkBuf` calls per
forwardPrefillBatch invocation (~16 × 2560-float bufs = ~80 MB of
alloc/free per token).  Once correctness is done, moving batch buffers
to state will recover TPS.

### Remaining bug hypothesis: 3-token decode cycle "answer is the"

The cycle suggests attention is aligning each new query with a fixed
earlier position instead of the most-recent one.  Possible causes:
- Some cache-write path still uses startPos=0, so slot 18 is not
  being written.
- The `posF32Buf` (used by Circuit DSL scatter) may also need a
  startPos-aware update.
- PLE precompute may be reading from the wrong column of
  `batchPLInputAll` across decode steps.

Worth checking next:
- All uses of `state.posF32Buf` in forwardPrefillBatch.
- The Circuit DSL scatter path for RoPE-K+KV-write (line 2180+ in
  forwardBlock, invoked when HESPER_SCATTER_KV=1).
- Whether the cache writes at line 3097 (ropeKKvWBatch) actually use
  absolute position or just col offset.

## 2026-04-20 EVEN FINAL-ER: all quant kernels bypassed, still broken

Tested combinations:
- `HESPER_DP4A_Q6K=0` (bypass fused RMSNorm+Q8_1 + Q6_K lm_head): STILL broken
- `HESPER_DP4A=0` (disable entire dp4a path — uses split-K F32 fallback): STILL broken

So the bug is NOT in any of:
- fusedRMSNormQ8_1 (final norm)
- fusedQ6KLinearDP4A4RowKernel (lm_head dp4a)
- The entire dp4a code path

The only remaining place for the bug is:
- forwardPrefillBatch's first 42-block loop producing different hidden
  states on 2nd call than 1st call
- Even though state.buf2 at PREFILL END is bit-identical, the
  DECODE-STEP-1 buf2 is subtly wrong.

Strongest evidence now: **the 2nd `forwardPrefillBatch` call, regardless
of all kernel-level configurations, produces half-magnitude logits**.

Given we've bypassed all known-shared kernel optimisations and the bug
persists, the likely remaining culprit is at the **InferenceState
level**: a buffer that gets reused between calls and carries stale
state across the forwardPrefillBatch→forwardPrefillBatch boundary.

## 2026-04-20 FINAL: bypass fused kernel doesn't fix — bug is upstream

Test: HESPER_DP4A_Q6K=0 forces the f32 fallback path
(standalone RMSNorm.forward + f32 matmul, bypassing fusedRMSNormQ8_1
and fusedQ6KLinearDP4A4RowKernel):

  Single  (DP4A_Q6K=0):  "The value of 2+2 is 4."   ✓ correct
  Unified (DP4A_Q6K=0):  "The<turn|>"                ✗ broken

So bypassing the fused norm+Q8_1 kernel DOES NOT fix the bug.  This
means the root cause is NOT in fusedRMSNormQ8_1.  It must be in buf2
itself (whose content we cannot directly validate without adding a
dump in the single-token path), or in how buf2 is being populated.

Also verified: state.buf2 at PREFILL END is bit-identical between
paths (byte-diff = 0).  So the two paths don't diverge at prefill.

What this means: the divergence happens inside the *second* call to
forwardPrefillBatch (= decode step 1 of unified path).  That call:
  1. Re-allocates batchBuf1/2 from scratch (uninitialised memory!)
  2. Writes embedding of token 818 ("The") to batchBuf1
  3. Scales to batchBuf2
  4. Runs the 42-block loop
  5. Extracts last column to state.buf2
  6. Runs final norm + lm_head

Layer-by-layer L2 on decode step 1 vs prefill ratios were all
near 1.0 (doc above), suggesting the block loop itself runs.  But
since single-token decode step 1 is NOT dumped via golden pipeline,
we can't directly diff layer outputs of the two paths to find where
they start to differ.

**Next session minimal steps (REVISED, prioritized)**:
  1. Add golden dumps to forwardSingleToken (e.g. put `dumpGolden` at
     each block output mirroring the forwardPrefillBatch hooks).
     Requires editing forwardBlock to accept a "decode dump prefix" or
     similar.  Once both paths can write comparable golden files, diff
     every layer to find the first divergence.
  2. Alternatively: instrument forwardSingleToken to write
     `state.buf2` content to disk after its column-equivalent step,
     so we can at least verify whether single and unified produce
     the same buf2 at decode step 1.
  3. Inspect batchBuf1/batchBuf2 allocation: maybe they are NOT being
     zeroed between calls and some kernel reads past seqLen=1
     columns thinking they're valid.

The bug is still NOT in fusedRMSNormQ8_1, and NOT simply PLE (since
HESPER_SKIP_PLE=1 on unified doesn't crash but produces gibberish,
same as single with PLE off).



Added `dumpGolden "prefill_buf2_lastcol_seqLen{N}"` right after the
column-extract in forwardPrefillBatch so we can see the input to the
final norm + lm_head.  Ran both paths on "What is 2+2?":

  Single  prefill end    buf2 (seqLen=18): L2 = 54.0526
  Unified prefill end    buf2 (seqLen=18): L2 = 54.0526  (diff = 0!)
  Unified decode step 1  buf2 (seqLen=1):  L2 = 49.2858  (reasonable)

So `state.buf2` after column-extract is CORRECT in both paths.  The
halving happens between buf2 and the final logits.

Chain between buf2 and logits:
  1. fusedRMSNormQ8_1Kernel (finalNorm+Q8_1 quantize): state.buf2 → q8Buf
  2. fusedQ6KLinearDP4A4RowKernel (lm_head): q8Buf → logitsBuf

Single path runs these exact same two kernels (line 3891+ of Gemma4.lean
forwardSingleToken uses them).  So why would unified decode's run of
the same two kernels produce half-magnitude logits?

Suspect: CachedDispatch prefill run's instance has stale bindings of
args somehow interacting with the Q8_1 scratch buffer.  Specifically,
`state.lmHeadQ8Buf` is *lazily* allocated on first call (line 3601).
If prefill populates it with seqLen=18-worth of data, then decode step
1 reads that same buffer expecting the single-row version...

BUT: fusedRMSNormQ8_1Kernel writes the full q8Buf on every call (it
always runs numRows=1).  Stale reads should not happen.

Another unusual finding: HESPER_SKIP_PLE=1 on both paths produces
gibberish 5-token outputs (expected since PLE is required), but
unified decode DOESN'T emit early EOS.  i.e. **The early-EOS at decode
step 1 is SPECIFIC to the PLE chain being active**.

Chain of suspicion for PLE + unified decode:
  - Prefill's PLE precompute writes `batchPLInputAll[i*totalPL..]` for
    each of seqLen=18 prompt tokens.
  - Decode step 1's PLE precompute writes for seqLen=1 token only.
  - But each `forwardPrefillBatch` call allocates a FRESH
    `batchPLInputAll` of its own size, so no cross-pollination.

I'm running out of time/context for this session.  This is the
narrowest the bug has gotten:

  **Bug lives in fusedRMSNormQ8_1 → Q6_K lm_head chain when invoked
  from a second `forwardPrefillBatch` call, AND the bug is
  PLE-dependent (disappears with HESPER_SKIP_PLE=1 in the sense that
  decode no longer emits early EOS).**

Next session minimal steps (prioritized):
  1. Add a HESPER_DUMP_Q8_UNIFIED env hook: dump the q8Buf written by
     fusedRMSNormQ8_1 in unified path, compare against single path.
     If q8Buf diverges despite buf2 being equal, the fused kernel is
     misbehaving on re-entry.
  2. Bypass the fused kernel by setting HESPER_DP4A_Q6K=0 (forces the
     fallback path that uses a standalone RMSNorm then f32 matmul).
     If unified becomes correct under that flag, the bug is in
     fusedRMSNormQ8_1 (reuse/caching).
  3. Inspect cudaAutoCache — is the PTX for fusedRMSNormQ8_1 at
     numRows=1 getting reused across calls with different state.buf2
     pointers?  (Logically shouldn't matter.)

## 2026-04-20 even narrower: bug lives AFTER `l_out-41`

Dumped `l_out-<li>` (every layer block output) under both paths with
`HESPER_GOLDEN_DUMP_DIR`.  Compared per-layer L2 magnitudes between
single-path (prefill last column, i.e. token 17 = "?") and unified
(decode step 1 col 0, i.e. token 18 = "The"):

  layer   single L2   unified L2   ratio
      0     48.12       55.16      1.15
      5    106.33      122.63      1.15
     17     95.38       83.81      0.88
     30    129.07      129.01      1.00
     40    104.45       96.24      0.92
     41     54.05       49.29      0.91

All layer outputs are within ~10% of each other (NOT halved).

BUT final logits:
  Prefill (single)  logits L2 = 4710
  Decode  (unified) logits L2 = 2534
  Ratio = 0.54  (≈ half)

**So the magnitude halving happens AFTER `l_out-41`, i.e. inside the
final norm + lm_head + softcap chain.**

Path is identical between the two in principle:
  1. columnExtract(currentBuf, col=seqLen-1) → state.buf2
  2. fusedRMSNormQ8_1Kernel (finalNorm + Q8_1 quant)
  3. fusedQ6KLinearDP4A4RowKernel (Q6_K lm_head)
  4. (single path only) logitSoftcapKernel

Note: the single-path dump includes the softcap step (it runs in
forwardSingleToken via generate()'s post-hook dump), but `forwardPrefillBatch`
does NOT apply softcap.  With tanh-based softcap and values up to ~21,
softcap COMPRESSES magnitudes (y = 30*tanh(x/30), for |x|=21 this is
≈20.1, only 5% smaller).  Softcap would make `single` logits smaller
not larger.  Yet `single` is larger — meaning the single path produces
larger logits upstream and softcap then slightly compresses them.

So the real suspect is **fusedRMSNormQ8_1 → Q6_K matmul between single
and unified**.  The PTX kernel is the same, but the input (state.buf2)
might be different.

Most likely concrete bugs to audit next:
  1. columnExtract on seqLen=1: when forwardPrefillBatch runs with
     seqLen=1, the PTX baked with `totalBatch = dim * 1` MUST be
     generated — not cached from the prefill run (seqLen=18).  If the
     cudaAutoCache collides based on PTX hash, and the PTX text differs
     only by a numeric literal, we'd get a stale kernel.  TO CHECK.
  2. PTX generated for seqLen=1 might have a different implicit assumption
     (e.g. only 1 col, so no need to multiply by colIdx) that's wrong.
  3. state.buf2 after extraction: is it actually the last-column's data,
     or half-filled with prev-call data?  A GPU-side memcpy of state.buf2
     right after columnExtract would answer this.

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

---

## Update 2026-04-20 PM: bug found and fixed

**Root cause:** `RMSNorm.forward` has a `numRows == 1` fast path that
replays a cached dispatch from `layer.prepared` via
`GPUBackend.replayCached`.  `replayCached` reuses `cached.args` — the
buffer pointers captured at first call.

Sequence that triggered the bug:
1. Prefill: `forwardPrefillBatch` calls `RMSNorm.forward` with
   `numRows=18` → misses the fast path → executes and **writes the
   numRows=18 dispatch to `layer.prepared`** (args = prefill batch
   buffers).
2. First decode (unified, seqLen=1): `forwardPrefillBatch` calls
   `RMSNorm.forward` with `numRows=1` → hits the fast path →
   `replayCached` with the numRows=18 dispatch and its stale args
   (prefill batch buffers, freed at end of step 1).

The stale dispatch launches against freed/reused memory, which
manifested as all-zero `attn_norm`, `Qcur`, `Kcur`, `Vcur` at L0 and
a collapsed decode output.

Non-unified decode didn't trip this because `forwardSingleToken` uses
persistent `state.*` buffers — the cached args stay valid.

**Fix** (commit pending): added an optional `refOverride :
Option (IO.Ref (Option CachedDispatch))` parameter to
`RMSNorm.forward`.  When caller passes a throwaway ref, the fast path
is bypassed (it only replays when the shared `layer.prepared` is
used).  Applied at the two `forwardPrefillBatch` fallback call sites
(attnNorm and ffnNorm) so transient batch buffers don't pollute the
layer-level cache.

**Verification:**
- `HESPER_UNIFIED_DECODE=1 ... "What is 2+2?"` → "The value of 2+2 is 4" ✓
- `HESPER_UNIFIED_DECODE=1 ... "The capital of France is"` → "The capital of France is **Paris**." ✓
- All 16 golden unit tests still pass.
- Non-unified decode unchanged (36 TPS; no regression).

**Remaining perf work:** unified decode currently at ~9 TPS because
`forwardPrefillBatch` allocates and frees ~15 batch buffers per call.
Moving these to persistent `InferenceState` slots is the next step;
the kernel-count reduction this enables is what makes Phase 3 a
measurable win.

---

## Update 2026-04-20 evening: unified decode at 62 TPS

Two more perf fixes stacked on the RMSNorm shape-guard:

### Fix 3: `unifiedKcr` separate from prefill `kcr`

`forwardPrefillBatch` takes an optional `kcr : KernelCacheRefs`; decode had
been passing `none` (a debug leftover).  That bypassed `ce`-level cache
lookup entirely, re-running `generatePTX + cuModuleLoad` for every
dispatch on every token.  Fix: allocate a decode-local `unifiedKcr` in
the generate loop and pass `some unifiedKcr` to the unified-decode call.
Must be a *separate* cache from the prefill `kcr` because some `ce`
cacheKeys inside `forwardPrefillBatch` don't include `seqLen` — a
cached prefill (seqLen=18) dispatch would otherwise fire at decode
(seqLen=1) shape.

Measured: 10 → 12 TPS (+22%).  Small because most of the kernel cost
actually came from `forwardBatchDP4A_fromQ8`, which goes through
`GPUBackend.executeWithConfig` directly and not `ce`.

### Fix 4: cache `forwardBatchDP4A_fromQ8` per (layer, seqLen)

The real jump.  `Linear.forwardBatchDP4A_fromQ8` was calling
`executeWithConfig` (non-cached), so the 210 Q4_K batched matmuls per
decode token (wQ/wK/wV/gate/up × 42 layers) were each regenerating PTX
→ ~70 ms/token of pure JIT overhead.  The weirdest part: there *was*
a dedicated `layer.dp4aBatchMatmulPrepared` ref — it just wasn't
wired into the function.

Fix: add `refOverride : Option (IO.Ref ...)` parameter.  Default to
`layer.dp4aBatchMatmulPrepared`; caller in `forwardPrefillBatch` passes
kcr-backed per-(layer, role, seqLen) refs so prefill and unified decode
each get their own ref pool.

Measured:
- 12-token "What is 2+2?": 12.4 → 22.3 TPS (+80%)
- 80-token "Write a short poem about cats": 12.2 → **62.2 TPS (+410%)**

### Pool batch scratch buffers

Separate fix, same commit train: the 5 small per-call allocBuffers
(`tokenIdsBuf`, `posBuf`, `cacheLenBuf`, `colIdxBuf`, `batchPLInputAll`)
now live on `state.prefill*Ref` with lazy-allocate / grow-only policy.
No measurable TPS impact on its own; eliminates 5 cudaMalloc+cudaFree
per decode token and is a prerequisite for any future CUDA Graph
capture (where in-capture allocation is illegal).

### Current status (2026-04-20)

| Path | TPS | vs llama.cpp 115-120 |
|------|-----|---------------------|
| single-token decode | 35.5 | 30% |
| single-token + CUDA Graphs | 29.8 | 25% (capture overhead pays off at longer gen) |
| **unified decode (no Graphs)** | **62.2** | **54%** |
| unified decode + CUDA Graphs | SEGV | — |

### Remaining 62 → 115 TPS gap

Per-kernel profile (doc 24) shows GPU matmuls already at 75-93% of
theoretical BW.  The gap is host-side `cuLaunchKernel` overhead
(~2µs × ~500 dispatches = 1ms/token).  Only CUDA Graphs can amortize
that — `cuGraphLaunch` submits the whole graph as one API call.

The Graphs capture crashes at the first decode iteration because
`forwardPrefillBatch` contains 18 `writeBufferOffset` call sites whose
Lean `ByteArray` host source gets baked into the graph by address.
Between replays Lean's GC can relocate that memory → replay reads
garbage.  Fix is mechanical but invasive: route every such write
through `writeScalarViaStaging` (pinned host) — 18 sites × one pinned
slot per site (can't share because a single capture contains multiple
distinct values written to the same slot).

That's the one remaining bottleneck to close the gap to llama.cpp.
Tracked as #142 (pending) and #127 (Phase C2).

---

## Update 2026-04-20 late evening: single-token + Graphs = 70 TPS

After the forwardBatchDP4A_fromQ8 cache fix landed (doc section above),
we returned to the single-token + CUDA Graphs path as the fastest
working config.  Two small additions:

### Fix: capture flow must launch graph once

`cuStreamBeginCapture` intercepts kernel launches — they are recorded
but do NOT execute.  The capture branch originally only captured-then-
instantiated; the first decoded token's forward pass never actually
ran, so next-iteration argmax saw stale prefill logits → the first
token was duplicated ("TheThe" / "SoftSoft" bug).  Fix: explicit
`cuGraphLaunch` after instantiate.

### Fold argmax into the captured graph

Argmax ran as a separate `cuLaunchKernel + D2H read` per token
outside the graph.  Now appended to the captured graph; replay just
reads `argmaxBuf[0]` after the graph sync.  Marginal TPS gain (70.6
vs 69.6) but cleaner control flow.

### forwardFusedNormWQ for shared-KV layers

21 of Gemma 4 E4B's 42 attention layers share KV with an earlier
full-attn layer.  These only need wQ, not wK/wV.  The old path
invoked `circuitRMSNorm` (Circuit DSL) + `runCached wQ matmul`
as two separate `executeWithConfigCached` call chains, each with its
own cache ref.  `forwardFusedNormWQ` replaces them with the same
2-dispatch `fusedRMSNormQ8_1Kernel + dp4a matmul` pair that
`forwardFusedNormQKV` uses on its Q path, on the layer's existing
`dp4aQuantizePrepared` / `dp4aMatmulPrepared` refs.  No dispatch
count change; moves shared-KV onto the same stable prepared-ref
pattern as full-attn layers.

### Current status (2026-04-20 late)

| Path | TPS | vs llama.cpp 115-120 |
|------|-----|---------------------|
| single-token no-Graphs | 35 | 30% |
| single-token + Graphs | 25 (short) / **70 (long)** | **61%** |
| unified no-Graphs | 62 | 54% |
| unified + Graphs | SEGV | — |

Session delta: 10.2 → 70 TPS on the best config (6.9×).

### Remaining 70 → 115 TPS gap

Per doc 24 per-kernel microbench, hesper's Q4_K matmuls are already at
75-93% of theoretical BW.  The 45-TPS gap is mostly at the kernel
level, not host-launch overhead (Graphs already fixed that):
- llama.cpp kernels run with low occupancy + 66 regs/thread (ILP)
- hesper kernels run with high occupancy + 34-36 regs/thread
- Per-kernel wall-time differs ~3x on Q4_K matmul (doc 00 table)

Closing this needs either:
1. Rewrite Q4_K dp4a kernel to llama.cpp pattern (#47, #119)
2. TurboQuant for Q4_K (#50 — not yet verified if llama.cpp uses it)
3. Larger fusions reducing memory traffic (already at 93% BW on main
   kernels so ceiling is tight)

Session stops here.  Next major moves tracked as #47, #119, #127.
