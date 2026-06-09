---
title: "20 — Phase 3 root finding: the bug is NOT multi-token-specific"
date: 2026-04-19
status: critical-finding
---

# The "ucucuc" bug is a pre-existing general correctness problem

## Earlier assumption (wrong)

I had been characterizing the `"ucucuc."` / `"are are…"` output as a
"multi-token prefill correctness bug" — something that appeared only
when the prompt has multiple tokens, and that single-token decode
handled correctly.

## Actual state (ground-truth re-verified 2026-04-19)

With a fresh GPU (after killing the zombie `llama-cli` at pid 2712157),
I ran three tests:

**llama.cpp reference output** (via `llama-cli`):
```
> Hello world how are you
I'm doing great, thank you for asking! I'm here
[ Prompt: 236.0 t/s | Generation: 84.8 t/s ]
```
So llama.cpp gives coherent English.

**hesper multi-token**:
```
$ HESPER_DP4A=1 lake exe gemma4-cuda ... "Hello world how are you" 15
[Result] Decoded: ucr.
```
Garbage.

**hesper single-token** (different prompts):
```
$ "The" 15
[Result] Decoded: TheTheTheTheTheTheTheTheTheTheTheTheTheTheThe

$ "Hello" 10
[Result] Decoded: HelloHelloHelloHelloHelloHelloHelloHelloHelloHello
```
Degenerate: just repeats the prompt word.

**hesper at older commit b38572f** (well before any Phase 2 work):
```
[Result] Decoded: TheTheTheTheTheTheTheTheTheTheTheTheTheTheThe
```
Same pre-existing problem.

## What this tells us

- The `"XxxxxXxxx…"` output is NOT a multi-token-specific bug.
- It is NOT caused by any of Phase 2's batched-attention work.
- It has existed at least since commit `b38572f` (pre-Phase-2).
- The pattern "just repeat the prompt word" suggests the decoded
  next-token prediction is **consistently picking the prompt token** —
  a very specific degeneracy, not random noise.

## Hypothesis

After prefill, the final hidden state + lm_head is producing logits
that peak at the **just-consumed token** rather than the natural
next-token. Possible root causes:

1. **lm_head weight layout bug** — the Q6_K or Q4_K lm_head matmul
   is reading weights with a wrong stride, so the top-1 always
   corresponds to the input.
2. **Final RMSNorm bug** — the hidden state is normalized wrong,
   and in the malformed state the matmul happens to align with the
   prompt token's embedding.
3. **Attention is not attending** — if the attention output is
   always the same as the input (identity), then each layer is
   roughly a no-op, and lm_head sees something close to just the
   embedding of the last prompt token.  The lm_head matmul against
   its own embedding (if the lm_head shares weights with
   token_embd, which Gemma 4 does) would peak at that token's
   index — giving the "HelloHello…" pattern.
4. **Attention scale / masking bug** — attention output is badly
   scaled, masking the real signal.

Hypothesis 3 is the most likely.  Gemma 4 typically ties
`lm_head.weight == token_embd.weight` (transposed).  If attention
produces (approximately) the same vector as the input embedding,
then `lm_head @ hidden ≈ <Embed, Embed_i>` which is maximal at `i`.
The model output would then be "HelloHelloHello…" — **exactly what
we observe**.

## Next steps (revised Phase 3)

Forget the bit-parity harness for now.  The core question is:
**does hesper's attention do anything at all?**

Quick diagnostics (in order):

1. **Test identity vs. attention**: Dump the output of attention
   (post-O-projection, pre-post-norm) for the final layer and compare
   its magnitude to the input embedding's magnitude.  If attention
   adds ~0 to the residual, attention is broken.  If it adds
   significantly, the breakage is elsewhere.

2. **Shuffle prompt tokens test**: Run hesper on "Hello" and on
   "world".  Both should produce the same wrong pattern if the
   last token's embedding is what dominates.  If they both produce
   repetitions of their own prompt, that's hypothesis 3 confirmed.

3. **Check lm_head weight tying**: Is `model.outputWeight`
   actually the same as the token embedding, or loaded separately?
   If tied, reading through the lm_head path is effectively
   computing `<Embed(prompt), Embed(prompt)>` when attention is
   broken.

4. **Simple math check**: Run without attention at all
   (e.g. zero out attnOut).  If the output is similar degeneracy,
   the attention truly contributes nothing.

## Revised priority

**Phase 3 is the single most important thing hesper needs**.  All the
Phase 2 prefill speed-ups are moot if the model is outputting garbage.
Until this is fixed, every performance number is meaningless (the
model is "fast at being wrong").

Options:

- (a) Git-bisect through the whole hesper history to find the commit
  that broke the model.  If there's a clean point in history where
  hesper produced English, that reveals what's wrong.
- (b) Static inspection: compare the hesper forward-pass to
  `models/gemma4-iswa.cpp` op-by-op; look for a missing step
  (maybe the attention scale, or lm_head bias, or logit-cap).
- (c) Single-layer forward parity: run a 1-layer-only forward pass
  (HESPER_PREFILL_LAYERS=1) and compare the pre-lm_head hidden state
  to a Python reference using the GGUF weights.

Option (a) is slow; (b) is quickest; (c) is most rigorous.  Start
with (b).

## Status summary

- Phase 2 (attention unification): complete, no regressions beyond
  what existed pre-Phase-2.
- Phase 3 (correctness): problem scope expanded — the bug is
  general, not multi-token-specific.  Hypotheses above.
- Performance: 216 ms prefill / 85 TPS decode (unchanged).
