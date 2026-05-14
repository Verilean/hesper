# Issue: Gemma 4 multi-token generation collapses into a 2-token loop

**Status:** ✅ RESOLVED (2026-04-08)
**Root cause:** Flash attention used `scale = 1/sqrt(headDim)` instead of Gemma 4's `f_attention_scale = 1.0`.
**Affected:** `Hesper.Models.Gemma4.forwardBlock`, line 1346
**Single-token correctness before fix:** ✅ verified (e2e cosine 0.99991, argmax matches) — but the bug was latent because softmax over 1 element is identity regardless of scale.

## Symptom

Running `lake exe gemma4-inference data/gemma-4-e4b-it-Q4_K_M.gguf "Hello" 10` produces:

```
[Gemma4] Generated 10 tokens in 12718.137909 ms (0.786279 tokens/sec)
[Result] Generated tokens: #[2, 366, 236772, 1018, 236772, 1018, 236772, 1018, 236772, 1018, 236772]
```

- Token 2 = BOS
- Token 366 = the model's first prediction after the prompt (plausible)
- Tokens 236772 and 1018 then alternate forever (8 of 10 generated tokens)

Expected: a coherent continuation of "Hello" (e.g. llama.cpp on the same model produces natural English).

## What we know works (pos = 0)

After fixing the bugs from the previous debugging round (flash attention `headDim > workgroupSize`,
KV-shared layer cache reuse, LM-head 2D dispatch, logit softcap), the **single-token forward pass at
pos = 0 is numerically correct**:

| Stage | Cosine vs llama.cpp | Notes |
|---|---|---|
| All 42 layer outputs | > 0.999 | `scripts/compare_all_layers.py` |
| `result_norm` | 0.994 | Final RMSNorm output |
| `result_output` (logits) | 0.99891 | After Q6_K LM head |
| Final argmax | 9259 ✓ | Matches llama.cpp golden |
| End-to-end vs golden | 0.99991 | `lake exe gemma4-validation` |

So the entire forward graph at pos = 0 is correct down to floating-point drift.

## What's broken (pos > 0)

The bug surfaces only when generation steps past pos = 0. The fact that the loop stabilizes on
exactly two alternating tokens suggests the model has lost all positional information beyond pos = 0
and is producing a deterministic 2-cycle attractor.

## Likely root causes (ranked)

### 1. KV cache write index doesn't advance with `pos`

The most suspicious area. `forwardBlock` calls `Attention.fusedCacheWriteKVKernel` and reads `pos`
from `state.paramsBuf`, but we should verify:

- Is the cache written at offset `pos * headDim` per KV head, or always at offset 0?
- After token 0 is generated, when token 1 runs the forward pass, does the K/V from token 0 still
  exist in the cache, or has it been overwritten?
- Reference: `Hesper/Layers/Attention.lean :: fusedCacheWriteKVKernel` and the call sites in
  `Hesper/Models/Gemma4.lean` around line 1289-1295.

### 2. RoPE at pos > 0 is wrong

At pos = 0, `cos(0) = 1, sin(0) = 0`, so RoPE is the identity. **All our existing tests are at pos
= 0**, so a buggy RoPE wouldn't have been caught yet.

Things to verify:
- Does `ropeWithFreqFactorsKernel` (full attention) read `pos` from `params[0]` correctly?
- Does the SWA branch's `RoPE.ropeKernelDynamic` read pos correctly?
- Are the `params` writes (`writeBuffer device state.paramsBuf 0 posBytes`) happening before each
  forward pass in `forwardSingleToken`, not just in the layer-5-dump example?

### 3. Flash attention `cacheLen` is stale

In `forwardBlock`:
```lean
let cacheLen := pos + 1
... flashAttentionDynamicKernel numHeads numKVHeads cfg.maxSeqLen headDim cacheLen scale ...
```
`cacheLen` is **baked into the kernel as a compile-time constant** (the kernel name encodes it).
This means at pos = 5, we recompile the shader with `cacheLen = 6`. Verify:

- Is the recompiled shader actually being dispatched, or is a cached pos=0 pipeline being reused?
- Each shader compile in the inference run takes time — the 0.79 tps figure with 1114 dispatches
  per token is consistent with full recompilation, but we should confirm the pipeline cache key
  includes `cacheLen`.

### 4. KV-shared layers (24-41) and pos > 0

The fix landed in this debug round used `Config.kvCacheLayer li` to map shared layers to layers
22 (SWA) or 23 (full). At pos = 0, all caches happen to be empty when first read (since the
reused layer was just written), so the bug would not have shown. At pos > 0, the order of
operations matters: when layer 24 runs, has layer 23's cache write for the *current* token
already happened in the same forward pass? If `forwardBlock` for li=24 runs after li=23 in the
same `forwardSingleToken`, it's fine — but verify the ordering and that the read happens after
the write across the GPU command buffer / batch boundary.

### 5. KV-shared layers don't write the new token's K/V anywhere

A subtler concern: shared layers re-use *another* layer's K/V cache, so they neither produce nor
store K/V for the current token. But **at the next position**, do they expect the cache to
contain the current-token K/V? It must be the *reused* layer's cache that grows — verify that
shared layers aren't somehow writing to or reading from a cache that doesn't get updated.

## Speed (secondary issue)

0.79 tokens/sec, 1114 dispatches per token. Two contributors:

1. Every layer-step is a separate dispatch (no kernel fusion across the residual stream).
2. Flash attention shader is recompiled every position because `cacheLen` is in the kernel name.
   Switch to `flashAttentionParamsKernel` (which already exists in `Hesper/WGSL/FlashAttention.lean`)
   that reads `cacheLen` from a uniform/storage buffer — single shader, dispatched many times.

This is a follow-up after correctness is restored.

## Reproduction

```bash
# Single-token correctness (currently passes):
lake exe gemma4-validation
# → Cosine 0.99991, argmax 9259 ✓

# Multi-token generation (currently broken):
lake exe gemma4-inference data/gemma-4-e4b-it-Q4_K_M.gguf "Hello" 10
# → tokens loop on 236772 / 1018
```

## Proposed debugging plan

The same layer-by-layer diff approach that worked for the single-token bugs should work here,
just with `pos = 1`:

1. Patch `llama.cpp` (or use existing `llama-eval-callback`) to dump `l_out-N`, `Qcur-N`,
   `Kcur-N`, `Vcur-N`, `Qcur_pos-N`, `__fattn__-N`, etc. at **pos = 1** for the same prompt.
   Need a 2-token prompt or `-n 2` so the second forward pass dumps too. Files should land in
   `/tmp/llama_dump_pos1/` (separate dir to avoid clobbering pos=0 dumps).
2. Add `Examples/Gemma4Pos1Dump.lean` that runs `forwardSingleToken` twice (pos=0 then pos=1)
   and dumps the layer outputs after the second call.
3. Run `scripts/compare_all_layers.py` (parameterized by dump dir) on the pos=1 dumps.
4. Find the first layer where divergence appears. The shape of the divergence will narrow the
   cause:
   - **Diverges at layer 0 in Q/K/V**: RoPE bug (wrong pos read)
   - **Diverges at layer 0 in `__fattn__`**: KV cache offset / cacheLen bug
   - **All layers 0-23 fine, layer 24 diverges**: KV-shared cache ordering bug
5. Fix and re-run.

## Root cause (what the bug was)

Hesper's `forwardBlock` used the standard transformer attention scale
`scale = 1.0 / sqrt(headDim)`, which is correct for most architectures. But
Gemma 4 sets `hparams.f_attention_scale = 1.0f` (see `llama.cpp/src/llama-model.cpp:1272`
and the `build_attn` call in `gemma4-iswa.cpp:94`). That is: Gemma 4 feeds the raw
`Q · K` dot product into the softmax, without the usual 1/√d temperature.

The reason this is architecturally sound in Gemma 4: **the Q-norm RMSNorm
normalizes each head independently**, which already bounds the dot-product
magnitudes. Re-scaling by 1/√d would over-smooth the softmax.

Why the bug didn't surface in the single-token test:

- At `pos = 0` there is exactly **one** cached K/V position, so the softmax
  degenerates to `softmax([s]) = 1.0` regardless of what `s` is (or what the
  scale is). The attention output at `pos = 0` is always exactly `V[0]`.
- Every existing test (validation exe, all-layers dump, per-layer comparisons)
  only exercises `pos = 0`, so the wrong scale was never distinguishable.

At `pos = 1` with `cacheLen = 2`, the softmax has to actually decide between
two positions. With `scale = 1/16`, `softmax([13.78 / 16, 11.49 / 16])` gave
roughly `[0.54, 0.46]` — effectively averaging V[0] and V[1] together. Llama
with `scale = 1` gives `softmax([13.78, 11.49]) ≈ [0.91, 0.09]` — strongly
preferring V[0] (the current "Hello" context). Hesper's averaged output had
a significantly smaller norm and failed to carry the context forward.

After the first generated token ("366", which was plausible because pos = 0
attention is correct), every subsequent token saw an attention pattern that
blurred all past positions together, which collapsed into a fixed-point
2-cycle on `236772 ↔ 1018`.

## Fix

One line change:

```lean
-- Hesper/Models/Gemma4.lean :: forwardBlock
-- let scale := 1.0 / Float.sqrt headDim.toFloat     -- WRONG
let scale : Float := 1.0                              -- right for Gemma 4
```

After the fix:

- `lake exe gemma4-validation` still passes: cosine **0.99991**, argmax matches ✓
- `lake exe gemma4-inference "Hello" 10` now produces varied tokens:
  `[2, 366, 1390, 568, 818, 22323, 9885, 236764, 3788, 16625, 5645]` — no loop.

## Debugging trail that led to this

1. `scripts/compare_pos1.py` showed layer-0 output at `pos = 1` was already
   broken (cosine 0.92), ruling out KV-shared layer bugs. The divergence was
   inside layer 0 itself.
2. Added temporary dumps of `attn_norm`, `Qcur`, `Kcur`, `Vcur`, `Qcur_pos`,
   `Kcur_pos`, and `__fattn__` inside `forwardBlock` gated on `li == 0 && pos > 0`.
   All pre-attention intermediates matched llama (>0.9994), but `__fattn__`
   dropped to cosine 0.86. So the bug lived inside the flash attention kernel
   itself (or its inputs beyond Q/K/V).
3. Also dumped `kvCache.kBuf` and `kvCache.vBuf` contents at positions 0 and 1:
   both positions matched llama's `Kcur_pos` and `Vcur_normed` exactly. So the
   cache was being populated correctly.
4. Wrote a CPU reference of the softmax attention using exactly Hesper's K/V
   cache contents and Hesper's Q. The CPU reference **matched Hesper's
   `__fattn__` exactly** (cosine 1.000) but disagreed with llama (cosine 0.86).
   This proved the flash attention *kernel* was correct, and the bug was in how
   attention was *parameterised*.
5. Tried the same CPU reference with different scales. `scale = 1.0` gave
   cosine **0.9998** against llama — pointing directly at the scale constant.
6. Grepped `llama.cpp/src/models/gemma4-iswa.cpp` for the scale passed into
   `build_attn` → `hparams.f_attention_scale` → set to `1.0f` in
   `llama-model.cpp:1272` for Gemma 4.

The whole chase took one `Gemma4Pos1Dump.lean` exe, one `compare_pos1.py`, a
few temporary `dumpBuf` calls in `forwardBlock`, and one CPU Python loop. No
GPU-side blind experimentation needed once the comparison harness was in place.

## Files involved

- `Hesper/Models/Gemma4.lean` — `forwardBlock`, `forwardSingleToken`, `generate`
- `Hesper/Layers/Attention.lean` — `fusedCacheWriteKVKernel`
- `Hesper/WGSL/FlashAttention.lean` — `flashAttentionDynamicKernel`, `flashAttentionSWAKernel`,
  `flashAttentionParamsKernel`
- `Examples/Gemma4Inference.lean` — generation loop
- `Examples/Gemma4Validation.lean` — single-token reference

## Related

- Previous debug round (single-token correctness): see commit history and
  `~/.claude/projects/-home-junji-hashimoto-git-hesper-gemma4/memory/project_gemma4.md`
- All four single-token bugs fixed are listed there.
