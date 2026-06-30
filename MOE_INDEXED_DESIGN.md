# Indexed MoE (mul_mat_id-style) — design

## Why our current grouping is pure overhead

Our MoE path: `counting-sort → PHYSICAL gather (copy input→sGathered) → matmul → PHYSICAL scatter
(copy sGatheredDown→sDownAll) → wacc`. The physical gather/scatter COPY ~17.5M elements each, the path
pads to `maxPadded` (~3× the real rows), and the multi-pass chain is what Dawn's flush-at-scale RACES on.

The grouping exists to **amortize weight reads** (read each expert's weight once for its M≈16 tokens). But
the roofline harness shows we are **COMPUTE-bound** (matmuls run at ~1.5× the compute floor, memory ~6%).
So the memory-amortization benefit is moot — only the overhead remains. Measured: ~408 ms/step of our
~1315 ms is NON-matmul (element-wise + grouping + attention-compute); llama.cpp's is ~0. That overhead is
HALF the 2.5× gap, and the matmul-only harness never saw it.

## What llama.cpp `mul_mat_id` does

It still groups tokens by expert for M>1, but via an **index list**, not a physical copy:
- `ids[e]` = the token indices routed to expert `e` (built by an argsort — like our counting-sort, but it
  keeps INDICES; the token data stays in place).
- The matmul kernel, per expert tile, reads `input[ids[e][i]]` (INDEXED A-load, in place) and writes
  `output[ids[e][i]]` (indexed store). **No separate gather/scatter buffers or passes. No padding** (it
  walks the actual `ids[e]` range per expert).

## Our indexed design (WGSL)

1. **Keep** the counting-sort, but its product is `ids` (sorted token index per grouped row) + per-expert
   `offsets` (we already compute `sSortedPos`/`sSortedSlot`/`sTileExpert` — `sSortedPos` IS the token index;
   so the data we need already exists, we just stop physically gathering/scattering).
2. **Indexed gate/up kernel** (replaces grouped-MMQ5 + gather + scatter): grid over (expert-tiles × out-col
   tiles). For a 32-row tile, expert `e = tileExpert[tile]`; for each row, `tok = sSortedPos[tileStart+row]`;
   A-load reads `input[tok, k]` (indexed, in-place — input is N×dim ≈ 3 MB, L2-resident, so scattered token
   reads are cheap); matmul vs `W_gateup[e]`; **fuse geglu** on the result; store to the per-slot output
   `out[slot, tok]` (indexed, where `slot = sSortedSlot[tileStart+row]`). One kernel, no gather/scatter pass.
3. **Indexed down kernel** (replaces geglu+q80+down+scatter): same shape; A-load reads the fused geglu output
   indexed; matmul vs `W_down[e]`; store `down_out[slot, tok]` indexed.
4. **wacc**: still a separate pass (`sMoeAcc[tok] += wts[tok,slot]·down_out[slot,tok]`, cross-slot — no WGSL
   f32 atomic to fuse it). BUT it now reads `down_out[slot, tok]` which is written in-place (no scatter
   between), so the chain is `gate/up → down → wacc` — 3 passes vs the old 6, and NO padding.

## What this eliminates (the 3 blockers of the fused-kernel pass, all at once)

- **Grouping overhead** (~100 ms): no physical gather/scatter copies; counting-sort stays (cheap).
- **Padding** (~3× rows): no `maxPadded`; process the actual `ids` per expert.
- **Dawn race**: the gather/scatter passes (the long racy chain) are gone; far fewer producer→consumer
  boundaries → the no-wait flush is far less likely to be dropped (and only `down→wacc` remains, possibly
  one cheap flush).

## The hard parts / open questions

- **Indexed A-load coalescing**: rows in a tile have different `tok` → scattered input reads. Mitigated by
  the input being L2-resident (3 MB); llama.cpp confirms this is fine. Validate with the harness.
- **wacc**: still cross-slot; if it still races, one flush (short chain now) or a tiny restructure. The hope
  is the short chain (3 passes) keeps the flush reliable.
- **Tile spanning experts**: a 32-row tile must not cross an expert boundary (or it reads two experts). The
  counting-sort already pads each expert to a 32-multiple (`sTileExpert` per 32-tile) — reuse that; the
  per-expert padding rows are sentinel-skipped (slot≥nUsed) at the indexed store. (This is a SMALL pad — to
  the 32-tile, not the huge `maxPadded` row-count blowup.)

## Validation (per PERF_AUTOTUNE_LOOP.md)

Golden at real dims + high experts; decode "Paris" AND "Jupiter" diffed vs the per-slot default; measure
emb+fwd; the harness (now with non-matmul benches) confirms the grouping/element-wise overhead dropped.

## Fallback: the shader replacer (metal_replacer)

If the WGSL indexed kernel can't match `mul_mat_id` (coalescing, occupancy we can't control through Tint),
use **github.com/junjihashimoto/metal_replacer** to swap llama.cpp's ACTUAL `mul_mat_id` Metal kernel in for
our generated one at the Metal level — we get its efficiency directly, bypassing the WGSL→Tint→Metal limits.
This is also the pragmatic answer to the whole 2.5×: replace the few hot kernels (mul_mm, mul_mat_id,
flash-attn) with llama.cpp's tuned Metal, keep our WGSL DSL for everything else + the verification/portability
story. Measure each replaced kernel with the harness to confirm the win.
