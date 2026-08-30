# DG→JS port: the trace package (M0/M1-data, 2026-07-15)

Produce with:
```
DG_NOMSL=1 DG_TRACE_JS=<dir> DG_TRACE_JS_DUMP=1 \
  ./.lake/build/bin/diffusiongemma-decode ./diffusiongemma-26B-A4B-it-Q4_K_M.gguf "<prompt>"
```
(**DG_NOMSL=1 is essential**: the default MoE gate/up runs a hand-MSL native
dispatch that bypasses Dawn — invisible to any WGSL trace AND not portable.
The all-WGSL path is what a browser replayer can execute; its gate/up is the
known 1.61× regression to win back by iterating WGSL in Chrome.)

Contents of `<dir>` (one decode step, ~4.9GB):
- `k<hash>.wgsl` — every template-expanded kernel (layer constants baked in,
  so ~885 distinct kernels dispatch once each per step).
- `ops.jsonl` — events between the `step-begin`/`step-end` markers:
  `d` dispatch {k, n, g:[x,y,z], b:[[bindingName, uid]…]},
  `w` writeBuffer {u, o, s, hex? (≤64KB)}, `r` readback {u, o, s},
  `f` queue flush (encoder split — replayer: submit boundary), `m` marker.
- `buffers.json` — uid → byte size (allocate these; activations need nothing
  else).
- `tensors.json` — uid → GGUF tensor name (31 raw-bound weights: 30
  ffn_gate_up_exps + token_embd; the replayer loads them from the GGUF).
- `b<uid>.bin` — 586 derived buffers (5.2GB: lm_head f16 predequant 1.48GB,
  per-layer f16 predequants, Q8 repacks, rope tables …) — load directly;
  replace with in-browser transforms later if GGUF-only distribution matters.

Replay loop per step: update the `w`-event buffers whose contents change per
step (canvas tokens, SC prob/temp, params), re-issue the `d` sequence with
`f` as submit boundaries, service `r` readbacks (logits → the JS commit
scheduler — port of the Lean commit/template logic).

Validated: trace run is non-perturbing (France text unchanged); referenced
buffers = 14.4GB ≈ the full working set; zero untraced dispatches.
