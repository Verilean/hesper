---
title: "21 — Development flow: llama.cpp-first, golden-value-driven"
date: 2026-04-19
status: process
---

# Development flow — lessons from the "ucucuc" session

This session demonstrated the cost of the old approach: ~6 hours of
Phase 2 attention-unification work with per-layer bit-parity harness,
all while the model's end-to-end output was degenerate
(`"TheTheThe..."`, `"WorldWorldWorld..."`).  We were optimizing a
broken pipeline.

The only reason we caught it at all was running llama.cpp for the
same prompt and noticing its output is coherent English while
hesper's is the prompt token repeated.  That comparison should be
**the first thing**, not the last.

## Core principle: llama.cpp is the oracle

llama.cpp is the reference implementation for every GGUF model we
target.  Before any optimization work starts, every intermediate
tensor must match llama.cpp's tensor at the same name, on the same
prompt, for the same token, to within a documented tolerance.

**No optimization work lands without a golden-value test that runs
end-to-end and compares at least the final logits.**  If you are
touching attention, also compare per-layer `l_out-<li>` and the
attention intermediates (`Qcur`, `Kcur`, `Vcur`, `Qcur_normed`,
`Qcur_pos`, `attn_out`, `ffn_post_norm`).

## The flow

### Step 0. Build llama.cpp with `eval-callback` patched to dump tensors

Already in tree.  `llama.cpp/common/debug.cpp:161` has the HESPER PATCH
that writes every F32 tensor matching a whitelist (any `l_out-*`,
`attn_out-*`, `ffn_post_norm-*`, `pe_in-*`, all `*-0`/`*-5`/`*-24`
intermediates, plus globals) to `/tmp/llama_dump/<name>.bin`.

```
llama.cpp/build/bin/llama-eval-callback -m <model.gguf> \
    -p "Hello world how are you" -ngl 42 > /tmp/llama_dump.out 2>&1
```

Produces a named blob per tensor.  For "Hello world how are you" this
is ~243 files.

### Step 1. Port a new hesper feature behind `HESPER_GOLDEN=1`

In `Gemma4.lean`, add one call per intermediate tensor:

```
if goldenEnabled then
  ce_dump_tensor ctx buf (dim * seqLen).toUSize s!"<name>-<li>"
```

Named the same as the llama.cpp tensor (e.g. `attn_norm-0`,
`attn_out-5`, `l_out-10`).  Dumps go under
`$HESPER_GOLDEN_DUMP_DIR`.

### Step 2. Diff script

`docs/scripts/compare_golden.py` (to be written) walks both dump
directories, per-tensor:

- Shape match? (assume llama.cpp column-major, hesper column-major;
  already consistent now per the KV-cache investigation in doc 19.)
- Bit-identical? Record `BIT-IDENTICAL`.
- Max abs diff, relative to tensor magnitude?  Classify as
  - `TIGHT` (max diff < 1e-5 × magnitude)
  - `LOOSE` (max diff < 1e-2 × magnitude) — acceptable for Q4K
    requantize paths, but note it
  - `BROKEN` (>1e-2 relative).

The script prints the FIRST tensor that hits BROKEN, with
(name, layer, max_diff, magnitude).  That is the bug.

### Step 3. Every PR runs this test

Before review, the author runs:

```
./scripts/run_golden.sh "Hello world how are you"
# PASS if no BROKEN, and final logits argmax matches llama.cpp
```

If a PR would break correctness, the golden test catches it before
any performance numbers are measured.

### Step 4. Only after golden passes → optimize

Optimization PRs start with a baseline golden run (should still
pass), then the optimization, then golden again.  Any BROKEN means
the optimization introduced a bug.

The bit-parity harness from doc 18 (`HESPER_ATTN_DUMP=batched` vs
`HESPER_FORCE_FALLBACK=1`) is still useful, but it's a **lower-
priority** check — it only proves two hesper paths agree, not that
either is correct.

## Structural match with llama.cpp

The in-repo `llama.cpp/src/models/gemma4-iswa.cpp` is ~230 lines.
Port it structurally:

| llama.cpp line | hesper equivalent | Test |
|---|---|---|
| 10: `inpL = build_inp_embd(tok_embd)` | `Embedding.forward` → `state.embdBuf` | `inp_scaled` bit-match |
| 13: `inpL = ggml_scale(inpL, sqrt(n_embd))` | `embeddingScaleKernel` | `inp_scaled` bit-match |
| 25: `inp_per_layer = project_per_layer_inputs(...)` | `perLayerBlocks` precompute | `inp_per_layer` bit-match |
| 42 (per layer): `cur = build_norm(attn_norm)` | `RMSNorm.forward` | `attn_norm-<li>` bit-match |
| 55: `Qcur = build_lora_mm(wq, cur)` | `Linear.forwardBatchDP4A` | `Qcur-<li>` tight |
| 60: `Qcur = build_norm(attn_q_norm)` | `fusedPerHeadQKVNormBatchKernel` | `Qcur_normed-<li>` tight |
| 63: `Qcur = ggml_rope_ext(...)` | `ropeWithFreqFactorsBatchKernel` | `Qcur_pos-<li>` tight |
| 92: `cur = build_attn(wo, Q, K, V)` | `flashAttentionBatchKernel` + `wO` matmul | `attn_out-<li>` tight |
| 107-112: post-attn-norm + residual | `rmsNormThenAddBatchKernel` | `ffn_post_norm-<li>` tight |
| 176-181: FFN | `forwardFusedGateUp` + `ffn_down` | `ffn_post_norm-<li>` |
| 190: `cur = ggml_add(cur, attn_out)` | `forwardNormThenAddBatch` (post-FFN) | `l_out-<li>` |
| 193-213: PLE | `PerLayerEmbedding.*` | `per_layer_embd_out-<li>` |
| 216-218: `cur = cur * out_scale` | Circuit DSL scale | `out_scaled-<li>` |
| 222: `cur = build_cvec(cur, il)` | (no-op for this model) | — |

Each row is a bit-parity checkpoint.

## Why structural match matters

The "ucucuc" bug is almost certainly **a missing or wrong step**
that's been hiding all along.  Given hesper's 42-layer transformer
appears to be a near-identity (doc 20), the likely candidates are:

1. **Embedding scale is applied twice or not at all** (line 13's
   `sqrt(n_embd)` factor) — Gemma-family quirk.
2. **Residual doubled or missing** at the post-attn or post-FFN
   add.
3. **out_scale or cvec multiplied incorrectly** making layer output
   equal input.
4. **build_cvec is a stub for control-vector injection** — we
   probably do nothing there, which is correct.
5. **Attention softmax divides by wrong scale** (`f_attention_scale`
   in llama-model.cpp:4262 is 1.0 for Gemma 4, unlike Gemma 2; if
   hesper uses `1/sqrt(headDim)` it would scale attention output by
   ~1/16 and effectively zero it).

Without the golden harness we can't tell which; with it, we see the
first diverging tensor and immediately know the culprit.

## Reconfirmed tokenizer mismatch (concrete example of why structural match matters)

On "Hello world how are you":

- llama.cpp:  5 tokens `[9259, 1902, 1217, 659, 611]`
- hesper:     9 tokens

Already a divergence at step 0 — before any inference happens —
invalidating ALL subsequent tensor comparisons unless we feed
hesper the **same** token IDs llama.cpp uses.

## What to do for the current "ucucuc" bug

1. Fix the tokenizer mismatch first (force hesper to use
   llama.cpp's token IDs via an env var, or fix hesper's
   SentencePiece encoder).
2. Run golden harness: dump every intermediate with identical
   names, diff.
3. The first diverging tensor identifies the kernel at fault.

## Circuit DSL v2

Not mandatory this session, but when we rewrite a hot path, prefer
to lower through v2 because:

- v2 Prims are shape-polymorphic; a `Prim.matmul` with N=1 or N=9
  lowers to different PTX from a single source.  llama.cpp's
  strength is the one-graph path; v2 gives us the same leverage.
- v2 fusion passes let us collapse "RMSNorm+Q8_1 quantize", "gate +
  GELU slice + up", etc., without writing hand-fused kernels.
- Each fusion pass is testable against golden values: `v2 IR →
  lower to kernels → run → diff against llama.cpp`.

When v2 isn't practical (quick experiment, one-off kernel), keep
using the raw ShaderM API but still gate behind golden.

## Summary

1. **llama.cpp first**: dump golden tensors before touching hesper.
2. **Port by matching tensor names**: each `l_out-<li>` in hesper
   must bit/near-match llama.cpp's.
3. **Golden test mandatory in every PR**.
4. **Structural optimization after correctness**.
5. Circuit DSL v2 preferred for new rewrites but not required.

The attention unification work in Phase 2 was correct under its own
bit-parity test, but the test was insufficient: two paths agreeing
with each other does not imply either is correct.  The golden test
above is the correctness oracle we should have started with.
