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

## 2026-04-19 addendum — correctness by construction

After 2 weeks of gemma4 work and a PLE cache bug that took 6 hours to
localise with E2E tests, the pattern is clear: **"start light, fix
later" is a trap**.  Every time we batched, fused, or refactored a
kernel without a per-kernel correctness oracle, a subtle bug
(off-by-one layer offset, RWW hazard, missing +1 in Gemma convention,
wrong residual semantics) silently entered the tree and was only
caught days later by coherence-of-English inspection.

Going forward: **correctness by construction**.  No kernel change
lands — not one line — until (a) a reference implementation exists
that matches llama.cpp's dump at rel < 1e-5, and (b) an LSpec unit
test against that reference exists and fails on the proposed
implementation for the right reason.  Then the implementation
lands, test goes green, and only then we measure TPS.

Also insufficient until now: **reference-architecture fidelity
check**.  We didn't verify, e.g., that Gemma 4 uses raw weights
(not `1 + w`) in its norm by reading the llama.cpp source until
a bug forced us to.  Always grep `llama.cpp/src/models/gemma4-iswa.cpp`
and `llama.cpp/src/llama-graph.cpp` for the exact formula and
ordering the kernel implements.

### Rule: no kernel change without a passing unit test

The E2E golden harness caught the catastrophic failure ("aga"
output), but it was too coarse to localise anything.  Hours were
spent bisecting a 42-layer stack before the `pleGeluGateMulSliceBatch`
PTX cache bug was found.  And even after the fix, l_out still has
~5-30% rel diff per layer — which is **not OK** just because the
output produces English prefixes.  Same architecture means same
math means same output to f32 numerical floor (≤ 1e-5 rel).

Every kernel that hesper's forward chain uses in prefill or decode
must have a LSpec unit test that:

1. **Inputs**: GGUF weights + a llama.cpp-dumped f32 input tensor.
2. **Output**: run *exactly that kernel* (not a chain), read back.
3. **Compare**: the matching llama.cpp-dumped f32 output tensor
   — or, when llama.cpp doesn't dump the intermediate, a Python
   f64 reference (see "Two-layer oracle" below).
4. **Threshold**: rel diff < `1e-5` (f32 numerical floor), NOT
   "Q4_K quant noise" etc.  If the kernel is itself a quantized
   matmul, the reference must also be that quantized matmul on the
   same weights — so the answer is still `~0` rel diff.
5. **All tests must pass before any TPS measurement.**

Do this per-unit — rushing to batched prefill, then measuring
end-to-end rel diff of 50%+, then trying to localise, is what cost
the ~6 hours.  Batch/fuse/unify rewrites are big changes and
demand per-kernel unit tests first.

### Two-layer oracle: Python f64 ref → LSpec GPU test

Not every kernel output is dumped by llama.cpp's eval-callback.
For example, the PLE intermediate `ple_moe_out` (post-GELU gate ×
per-layer-input slice) is not a llama.cpp-named tensor — llama.cpp
computes it inside `build_ffn` and immediately consumes it.  In
these cases we need our own reference.

**Layer 1 — Python f64 oracle**: for each kernel, write a numpy
(f64) reference in `Tests/golden-unit/python_refs/` that:

- Loads the llama.cpp-dumped input (or, for intermediates, the
  llama.cpp-dumped output of the preceding kernel).
- Reconstructs the kernel's math exactly as stated in
  `llama.cpp/src/models/gemma4-iswa.cpp` and
  `llama.cpp/src/llama-graph.cpp`.  Cite the line numbers in a
  comment.
- **Sanity step**: when llama.cpp does dump the matching output,
  assert the numpy f64 ref matches it at rel < 1e-6.  This
  validates the formula is correct (see
  `test_rmsnorm_attn_l0.py` for the template).
- Writes the reference output to `tmp_<name>_f64ref.bin` for the
  Lean test to consume.

**Layer 2 — LSpec GPU test**: in `Tests/golden-unit/<Kernel>.lean`,
invoke the hesper kernel on the same inputs, read back, and
compare against the f64 ref (or directly against the llama.cpp
dump if available).  Threshold rel < 1e-5 for f32 kernels; rel
< 1e-4 for Q4_K matmuls where the reference is also Q4_K.

### Reference-architecture fidelity checklist

Before writing a new kernel (or batching/fusing an existing one),
open the llama.cpp source for the model family and confirm:

- [ ] **Exact formula**: RMSNorm `y = x/rms * w` vs `y = x/rms * (1+w)`?
      Check `llama.cpp/src/llama-graph.cpp::build_norm` and the
      model's `convert_hf_to_gguf.py::norm_shift`.
- [ ] **Operator order**: residual before or after norm?  Scale
      before or after activation?
- [ ] **Per-layer overrides**: head_dim, num_kv_heads, sliding-
      window flag — does the model change these per-layer?  Gemma 4
      does (key_length vs key_length_swa).
- [ ] **Quantization format**: Q4_K vs Q6_K vs f16 — which format
      does the reference use for *this* tensor?  Must match.
- [ ] **Scale factors**: any `ggml_scale` in the graph?  Their
      constants must match (e.g. `1/sqrt(n_embd)`,
      `1/sqrt(head_dim)`, embedding scale, attention logit cap).
- [ ] **Optional tensors**: is the tensor `TENSOR_NOT_REQUIRED` in
      llama-model.cpp?  If so, branch on its existence, not assume
      it's always present.

Write the checklist answer in a comment above the Lean kernel
call site.  When the kernel gets batched/fused later, the comment
is the reference to re-check against.

### Writing a unit test (LSpec + CUDA)

**Single-exe policy**: *all* unit tests are bundled into one LSpec
exe `gemma4-unit-tests` to avoid per-kernel lakefile entries.
Tests are organised as LSpec groups:

```
Tests/golden-unit/
  Common.lean        — helpers: loadF32Bin, relDiff, GGUF weight extract
  RMSNorm.lean       — attn_norm, ffn_norm, post_attn_norm, post_ffn_norm
  Linear.lean        — wQ/wK/wV/wO, ffn gate/up/down (Q4_K)
  Attention.lean     — perHeadRMSNorm (q/k/v), RoPE, FlashAttention
  PLE.lean           — inpGate, geluGateMulSlice, proj
  All.lean           — aggregator: List (String × List TestSeq)
  Main.lean          — `lake exe gemma4-unit-tests` entry
```

Each test module exports `allTests : IO (List (String × List TestSeq))`
following the pattern of `Tests/BufferTests.lean`.  `Main.lean`
concatenates them and runs `LSpec.lspecIO`.

**Temporary files**: any scratch artefacts that a test creates must
be prefixed `tmp_` (e.g. `/tmp/hesper_unit_tmp_rmsnorm.bin`) so they
are obviously disposable.  The per-run llama.cpp golden dumps under
`/tmp/llama_dump/` are *inputs* — tests read them but don't
recreate them.

Test template (per-kernel, inside one of the modules above):

```
def testRMSNormAttnL0 : IO TestSeq := do
  let ctx ← CUDAContext.init
  let input  ← loadF32Bin "/tmp/llama_dump/inp_scaled.bin" (dim := 2560) (lastToken := true)
  let weight ← extractF32FromGGUF "data/gemma-4-e4b-it-Q4_K_M.gguf"
                  "blk.0.attn_norm.weight"
  let expected ← loadF32Bin "/tmp/llama_dump/attn_norm-0.bin" (dim := 2560)
                  (lastToken := true)
  let actual ← runRMSNormOnGPU ctx weight input
  let rel := relDiff actual expected
  pure $ test s!"RMSNorm(attn_norm) L0 last-token, rel={rel}" (rel < 1e-5)
```

`allTests` per module:

```
def allTests : IO (List (String × List TestSeq)) := do
  let t1 ← testRMSNormAttnL0
  let t2 ← testRMSNormFfnL0
  ...
  pure [ ("RMSNorm attn L0", [t1]), ("RMSNorm ffn L0", [t2]), ... ]
```

`Main.lean`:

```
def main : IO UInt32 := do
  let g1 ← RMSNorm.allTests
  let g2 ← Linear.allTests
  let g3 ← Attention.allTests
  let g4 ← PLE.allTests
  LSpec.lspecIO (.ofList (g1 ++ g2 ++ g3 ++ g4)) ([] : List String)
```

**Prerequisite**: `/tmp/llama_dump/` must be populated by
`scripts/run_golden_dump.sh` (or manually via
`llama-eval-callback -p "Hello world how are you" -n 1`).  The test
fails with a clear error if any dump is missing.

### Kernels that currently need unit tests (prefill-batched path)

Order: same as gemma4-iswa.cpp control flow.  Input = llama.cpp
f32 dump; output compared to llama.cpp f32 dump at the matching
name.  For quantized matmuls, compare against llama.cpp's same-
quantized matmul output (which is what llama.cpp dumps).

| Stage | Hesper kernel | golden input | golden output | threshold |
|---|---|---|---|---|
| 1 | `Embedding.forward` + `embeddingScaleKernel` | token ids | `inp_scaled` | 0 (bit-exact) |
| 2 | `RMSNorm.forward` (attn_norm) | `inp_scaled` | `attn_norm-0` | 1e-5 |
| 3 | `Linear.forwardBatchDP4A` (wQ) | `attn_norm-0` | `Qcur-0` | 1e-4 |
| 4 | `Linear.forwardBatchDP4A` (wK) | `attn_norm-0` | `Kcur-0` | 1e-4 |
| 5 | `Linear.forwardBatchDP4A` (wV) | `attn_norm-0` | `Vcur-0` | 1e-4 |
| 6 | `perHeadRMSNormBatchKernel` (q_norm) | `Qcur-0` | `Qcur_normed-0` | 1e-5 |
| 7 | (same for k_norm, v_norm) | ... | ... | 1e-5 |
| 8 | `ropeWithFreqFactorsBatchKernel` (Q) | `Qcur_normed-0` | `Qcur_pos-0` | 1e-5 |
| 9 | `fusedRopeKAndCacheWriteBatchKernel` (K) | `Kcur_normed-0` | `Kcur_pos-0` | 1e-5 |
| 10 | `flashAttentionBatchKernel` | Q_pos, K cache, V cache | `__fattn__-0` | 1e-4 |
| 11 | `Linear.forwardBatchDP4A` (wO) + residual | `__fattn__-0` | `attn_out-0` | 1e-4 |
| 12 | `rmsNormThenAddBatchKernel` (post-attn) | — | — | — (NOT used this block; fused into attn_norm path) |
| 13 | `fusedRMSNormQ8_1Kernel` (ffn_norm) | `attn_out-0` | `ffn_norm-0` (via reverse dequant) | 1e-5 |
| 14 | Linear gate (Q4_K) | `ffn_norm-0` | `ffn_gate-0` | 1e-4 |
| 15 | Linear up (Q4_K) | `ffn_norm-0` | `ffn_up-0` | 1e-4 |
| 16 | `geluMulKernel` | gate, up | `ffn_geglu-0` | 1e-5 |
| 17 | Linear down (Q4_K) | `ffn_geglu-0` | `ffn_out-0` | 1e-4 |
| 18 | `forwardNormThenAddBatch` (post-FFN) | ffn_out, attn_out | `ffn_post_norm-0` | 1e-5 |
| 19 | PLE `Linear.inpGate` (Q4_K) | `ffn_post_norm-0` | `ple_gate-0` | 1e-4 |
| 20 | `geluGateMulSliceBatchKernel` | ple_gate, inp_per_layer slice | `ple_moe_out-0` | 1e-5 |
| 21 | PLE `Linear.proj` (Q4_K) | ple_moe_out | `ple_proj-0` | 1e-4 |
| 22 | PLE `forwardNormThenAddBatch` | ple_proj, pe_in | `per_layer_embd_out + pe_in` | 1e-5 |
| 23 | `out_scale` mul | (pe_in + PLE) | `l_out-0` | 1e-5 |

Unit tests not yet written for any of these.  Write one at a time,
keep each <150 lines, land it passing, then move to the next.

### Why unit tests matter for big refactors

Small patches (e.g. renaming a variable) don't need a new unit
test.  But anything that rewires dispatch — batching, fusion,
cache-key changes — is a **big change** that can silently
introduce PTX-cache/layout/indexing bugs that E2E tests can't
easily localise.  For these, write the unit test FIRST, watch it
fail on the naive port, fix until green, then measure TPS.

The PLE cache-key bug (commit c3de2f7) is the concrete example:
it had been in the tree for days, invisible to E2E ("aga" output
looked like any other garbage), and took a full session of
layer-by-layer golden diffs to pin down.  A unit test around
`geluGateMulSliceBatchKernel` with L0 AND L4 inputs (same code,
different plOffset) would have caught it the moment the batched
PLE was introduced.
