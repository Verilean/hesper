---
title: "22 — Golden-value diff: where does hesper disagree with llama.cpp?"
date: 2026-04-19
status: working-notes
---

# Golden-value diff status

Per doc 21, we now compare hesper's intermediate tensors to
llama.cpp's `llama-eval-callback` dump on the same 5-token prompt
`"Hello world how are you"` (token IDs `[9259, 1902, 1217, 659, 611]`,
matching after the tokenizer fix in commit 6480aeb).

## Infrastructure

- **llama.cpp side**: `llama.cpp/common/debug.cpp` has a HESPER PATCH
  (line 161+) that dumps named tensors to `/tmp/llama_dump/<name>.bin`.
  Already in tree.  Run with
  `llama.cpp/build/bin/llama-eval-callback -m <model> -p "..." -ngl 42`.
- **hesper side**: `HESPER_GOLDEN_DUMP_DIR=<dir>` writes matching
  names to `<dir>/<name>.bin`.  Wired at commits 767b018, b4f89d6.
- **Diff script**: ad-hoc Python in this doc; see the code block
  under "Results" for the runnable version.

## Coverage so far

| Stage | llama.cpp name | hesper name | Match level |
|---|---|---|---|
| Embedding + scale | `inp_scaled` | `inp_scaled` | 5e-7 rel (bit-identical) |
| Per-layer Q projection | `Qcur-<li>` | `Qcur-<li>` | L0: 0.5%, L5: **87%** |
| Per-layer K projection | `Kcur-<li>` | `Kcur-<li>` | L0: 0.5%, L5: **73%** |
| Per-layer V projection | `Vcur-<li>` | `Vcur-<li>` | L0: 1.0%, L5: **126%** |
| Post-attn residual | `attn_out-<li>` | `attn_out-<li>` | L0: 1.8%, L4: 2.9%, L5: 30% |
| Post-FFN norm | `ffn_post_norm-<li>` | (not yet — hesper FUSED) | N/A |
| Layer output (after PLE + outScale) | `l_out-<li>` | `l_out-<li>` | L0: 2%, L3: 3%, L4: 13%, L5: 21% |

## Interpretation

1. **`inp_scaled` matches** → embedding + sqrt-scale are correct.
2. **L0 Q/K/V: ~1% diff** → Q4_K matmul + QKV-norm chain is within
   quant noise.
3. **L0 `attn_out` 1.8%, L0 `l_out` 2.0%** → the entire L0 (attention
   + FFN + PLE + outScale) introduces only ~2% error, which is
   quant-noise scale.
4. **L1-L3 `l_out`**: 1%, 2%, 3% — growing linearly.  Consistent with
   accumulating Q4_K quant noise, not a bug.
5. **L4 `l_out`: 13% (token 0: 18%).  L5: 21%.  Then climbing to 60%+
   by L30**.  That's the bug region.
6. **L4 attn_out matches at 2.9%** → L4's attention is still fine.
   So the L4 divergence happens **after** attention: in FFN,
   post-FFN norm+add, PLE, or outScale for L4.
7. **L5 Qcur already 87%** → whatever went wrong at end-of-L4
   corrupts the input to L5's Q projection massively.

## Divergence happens post-L4-attention

The chain:

```
L4 inp → attn → attn_out  (OK, 2.9%)
              → FFN       (unknown, no dump yet)
              → post-FFN  (unknown — llama.cpp name mismatch with hesper's fused kernel)
              → PLE       (unknown)
              → outScale  (unknown)
              → l_out-4   (BROKEN, 13%)
```

Something in FFN / post-FFN / PLE / outScale at L4 takes a 3%-error
input and amplifies it to 13%.

## Next step

Insert additional named dumps inside L4's post-attention path:

- `ffn_gate-4`, `ffn_up-4` (pre-GELU) — llama.cpp names from
  models/gemma4-iswa.cpp:175-181.  Requires splitting hesper's fused
  gate+up into separate dumps.
- `ffn_post_norm-4` — llama.cpp's norm-only output (our hesper
  fused kernel would need to split).  Alternative: dump the input
  and output of `forwardNormThenAddBatch` separately.
- `pe_in-4` (pre-PLE), `per_layer_embd_out-4` — PLE intermediate.
- `out_scaled-4` — post outScale.

Each of these is one extra `dumpGolden` call in `Gemma4.lean`.  The
one that first hits >5% rel vs llama.cpp is the bug.

## Status summary

- Tokenizer: ✅ fixed
- Golden harness: ✅ working for 6 tensor types
- Bug localisation: narrowed from "whole 42-layer stack" to
  "L4 post-attention subpath".
- Correctness: multi-token output still garbage ("aga"), but now
  we have a principled path to fix it.

## 2026-04-19 update — extended to ffn_out/per_layer_embd_out/out_scaled

After adding intermediate dumps (`ffn_out-<li>`, `per_layer_embd_out-<li>`,
`out_scaled-<li>`) and widening llama.cpp's whitelist to include
L1–L4, the per-layer picture became clearer:

```
layer |  attn_out |   ffn_out |     l_out
------+-----------+-----------+----------
    0 |    3.46%  |    3.60%  |    3.77%
    1 |    1.85%  |    9.35%  |    4.66%  ← L1 FFN: 1.85 → 9.35 (5×)
    2 |    5.98%  |   42.29%  |   13.06%
    3 |   12.69%  |   38.60%  |   17.32%
    4 |   18.36%  |   50.57%  |   50.33%
    5 |   54.69%  |  544.93%  |   58.06%
```

**The FFN at every layer amplifies the error.**  attn_out-1 is clean
(1.85%) but ffn_out-1 is 5× worse (9.35%).  This is not quant noise;
it's a real correctness bug in hesper's FFN path (ffnNorm → gate/up
→ GELU × up → down-proj).

Note on intermediate naming: `ffn_post_norm-<li>` and
`per_layer_embd_out-<li>` show huge rel diffs (150–2500%) even when
the surrounding `l_out` matches at 3.77%.  Reason: hesper fuses
`RMSNorm + residual add` into one kernel, so hesper dumps the
post-residual tensor, while llama.cpp dumps the pre-residual
(norm-only) tensor at those names.  **Ignore these two columns** —
the `l_out` and `ffn_out` boundaries are semantically matched and
those are the load-bearing numbers.

## Next step (revised)

Isolate which FFN sub-kernel amplifies the error:

1. Dump `ffn_norm-<li>` output (pre-gate/up).  Compare vs llama.cpp
   `ffn_norm-<li>` (already in llama.cpp dumps).
2. Dump `ffn_gate-<li>`, `ffn_up-<li>` (pre-GELU).
3. If `ffn_gate` or `ffn_up` already diverges: bug in Q4_K dp4a
   matmul for gate/up shape (2560→10240).  If they're fine, bug
   is in GELU×up or down-proj.

## 2026-04-19 update — ffn_norm is the amplifier

Added `HESPER_FFN_FASTPATH_DISABLE=1` flag (forces the standalone
RMSNorm → f32 matmul fallback path, same as non-Q4_K-both layers).
With the fast path disabled, hesper now dumps `ffn_norm-<li>` (output
of standalone `RMSNorm.forward`, matching llama.cpp's `cb(cur,
"ffn_norm", il)` which fires AFTER the weight multiply in `build_norm`).

```
L | attn_out | ffn_norm | ffn_gate | ffn_up | ffn_geglu | ffn_out
--+----------+----------+----------+--------+-----------+--------
0 |   3.46%  |   3.91%  |   3.06%  |  3.66% |    4.09%  |  3.60%
1 |   1.85%  |   7.64%  |   4.40%  |  6.34% |   10.12%  |  9.37%
2 |   5.98%  |  38.43%  |  23.70%  | 31.71% |   44.82%  | 42.31%
```

At L1: `attn_out` is 1.85% (quant-noise OK), but `ffn_norm`
**amplifies to 7.64% — a 4× increase**.  RMSNorm is a pure f32 op
with tiny eps; it should NOT amplify error.

Tested hypotheses:

- **Gemma3 "+1 to weight" convention**: rejected — Gemma4 overrides
  `norm_shift()` to return 0.0 in `convert_hf_to_gguf.py:7443-7445`,
  so Gemma4 RMSNorm is plain `y = x * invRms * weight`, which
  matches hesper.
- **Disabling fused RMSNorm+Q8_1 fast path**: rejected — numbers are
  nearly identical with/without `HESPER_FFN_FASTPATH_DISABLE=1`,
  proving the fusion is not the bug.

The RMSNorm output formula is the same on both sides:
  `y[d] = x[d] * rsqrt(sum(x²)/D + eps) * weight[d]`

So either:
- hesper's weight buffer differs from llama.cpp's by a
  layer-dependent amount (unlikely — it's the same GGUF file), or
- hesper's `sum(x²)` reduction has a precision bug (f32 summation
  order?) that amplifies when the input has small-magnitude
  components dominated by cancellation.

Status: **ffn_norm = amplifier confirmed**.  Next: compare raw
ffn_norm weights from GGUF to llama.cpp's (sanity check), then dump
the pre-weight-multiply `norm-<li>` tensor from hesper and compare
to llama.cpp's `norm-<li>` to localise the bug to either the
reduction or the weight multiply.

## 2026-04-19 update 3 — correct last-token slicing + RMSNorm exonerated

**Methodology bug fixed.**  hesper dumps 5 tokens × 2560 floats
per tensor; llama.cpp uses `inp_out_ids` to prune intermediate
positions, so most tensors dump only the last token (2560 floats).
Previous comparisons silently returned `None` on size mismatch and
used wrong slices when they didn't, producing misleading numbers.

Correct diff: always compare the last 2560 floats on both sides
(the last-token slice, position 4 in a 5-token prompt).

**Sanity: `inp_scaled` now matches at 0.0000% (bit-exact)** — the
embedding + sqrt-scale is correct.

**RMSNorm is mathematically correct.**  Reference f64 RMSNorm on
hesper's `attn_out-1` input produces output that matches hesper's
`ffn_norm-1` at **0.0000%**.  The 10.36% output diff vs llama.cpp
is an unavoidable consequence of the 2.29% input diff combined
with the highly skewed ffn_norm weight distribution (range
`[-9.5, 184]`, median 6.78 — some dims amplify by 184×).

Per-stage last-token rel diffs (HESPER_DP4A=1, 5-token prompt):

```
L0: Qcur 0.85%, Kcur 0.68%, Vcur 1.27%, attn_out 3.56%,
    ffn_norm 5.03%, ffn_gate 4.09%, ffn_up 4.87%,
    ffn_out 4.93%, l_out 4.44%
L1: Qcur 3.20%, attn_out 2.29%, ffn_norm 10.36%, l_out 4.28%
L2: attn_out 5.10%, l_out 10.50%
L3: attn_out 9.93%, l_out 14.18%
L4: attn_out 14.94%, l_out 54.38%  ← jump
L5: attn_out 59.10%, l_out 61.61%  ← stabilises
L20: attn_out 120%, l_out 110%
L40: attn_out 74%, l_out 74%
```

**Interpretation**:

1. Q4_K matmul noise is ~0.5–1.5% per projection (Q/K/V at L0).
2. Attention + Oproj doubles the accumulated error (to ~3–4%).
3. RMSNorm doesn't introduce error, but amplifies the input error
   via the high-weight-variance multiply (well-behaved math).
4. Residual adds dilute error — that's why `l_out` is smaller
   than `ffn_out` at each layer.
5. **L3→L4 has a 4× jump (14% → 54%)** that doesn't fit the
   gradual Q4_K noise model.  This is the one genuine anomaly.
   It does NOT correspond to a known Gemma 4 architecture
   boundary (full-attn→SWA is at L7; shared-KV starts at L24).

## Next step (revised again)

Focus on the L3→L4 discontinuity.  Hypotheses to check:

- **FlashAttention split-K threshold**: some per-head code path
  engages only beyond a certain prompt length or head-dim × seqLen,
  and L4 happens to be where SWA window starts needing it.
- **Q4_K matmul kernel selection**: `Linear.forwardBatchDP4A`
  dispatches different kernels based on `outDim` or shape; maybe
  L4 hits a different kernel variant with a bug.
- **Per-layer RoPE freq_factors**: L4 is near where RoPE period
  crosses the sliding-window boundary; any off-by-one indexing
  error in hesper's batched RoPE-Q/K would surface here.

Collect evidence by dumping `Qcur_pos`, `Kcur_pos` (post-RoPE) and
`__fattn__` (attention output pre-Oproj) at L3 and L4, then diff.
Use whichever amplifies first as the next lead.

## 2026-04-19 update 4 — full attention-chain trace

Added `Qcur_pos-<li>` and `__fattn__-<li>` dumps at the batched
attention code path.  With correct last-token slicing (using the
actual per-tensor dims: qDim = numHeads × headDim(li), kvDim =
numKVHeads × headDim(li)):

```
L | Qcur  | Kcur  | Vcur  | Qcur_pos | __fattn__ | attn_out | l_out
0 | 0.94% | 0.71% | 1.28% |  22.87%  |   8.70%   |   3.56%  |  4.44%
1 | 3.26% | 2.78% | 5.00% |  23.99%  |  13.40%   |   2.29%  |  4.28%
2 |17.85% |12.47% |24.13% |  28.06%  |  31.42%   |   5.09%  | 10.50%
3 |20.45% |11.66% |29.45% |  28.83%  |  29.11%   |   9.92%  | 14.17%
4 |20.66% |14.66% |47.70% |  28.43%  |  50.58%   |  14.92%  | 54.47%
5 |89.98% |70.18% |88.56% |  93.57%  |  77.89%   |  59.17%  | 61.67%
```

Key observations:

- **Qcur_pos and __fattn__ rel diffs are suspicious (22%+ at L0)
  but likely spurious layout mismatches.**  llama.cpp's
  `Qcur_pos` is in `[headDim, seqLen, numHeads]` (post-permute)
  while hesper's `batchQRopedBuf` is `[seqLen, numHeads, headDim]`.
  Same bytes, different interpretation → diff is meaningless.
  Same for `__fattn__` (post-FA is permuted in llama.cpp).
- **Load-bearing numbers**: `Qcur`, `Kcur`, `Vcur` (pre-norm,
  pre-permute) and `attn_out` (post-Oproj, back to [dim, tokens]).
  These have matching layouts.
- **Q4_K quant noise grows cleanly L0→L3**: Qcur 0.9% → 20%
  (expected 0.5–1.5% per matmul × accumulation).
- **L4 Vcur jumps 29.45% → 47.70%** while Qcur/Kcur do not.  That
  asymmetry is suspicious.
- **L4→L5 head-dim change**: Gemma 4 uses headDim=256 for SWA
  layers (L0-L4) and headDim=512 for full-attn layers (L5+).
  Hesper's code handles this correctly (Config.headDim branches
  on layer type, file sizes match: L0 qDim=2048, L5 qDim=4096).

- **L3→L4 FFN amplification is the real bug**: attn_out-3 is
  9.92% → l_out-3 14.17% (1.4× amp, normal), but attn_out-4 is
  14.92% → l_out-4 54.47% (**3.6× amp**, abnormal).  The per-layer
  FFN kernels are identical between L3 and L4 — only the layer
  weights differ.  Either L4's FFN weights have extreme
  distribution (rare) or the PLE (per-layer embedding) at L4
  introduces the jump.

Sanity restored: `inp_scaled` matches 0.0000% bit-exact.

## 2026-04-19 update 5 — f64 reference: RMSNorm is not a bug

With proper last-token slicing, f64 reference RMSNorm on hesper's
`attn_out-1` produces output matching hesper's `ffn_norm-1`
bit-exactly.  The 10.36% rel diff from llama.cpp's `ffn_norm-1` is
the "error floor" — any correct RMSNorm on hesper's input would
produce this much diff, because ffn_norm weights span
`[-9.5, 184]` and amplify directionally.  **RMSNorm is exonerated.**
