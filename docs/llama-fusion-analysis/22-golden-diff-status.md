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
