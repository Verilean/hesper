---
title: "15 — llama.cpp's single-path architecture and what hesper must change"
date: 2026-04-19
status: architecture-plan
---

# The core insight: llama.cpp has **one** forward path

llama.cpp does **not** have separate "prefill" and "decode" code paths.
It has exactly one path: `llama_decode(batch)` where `batch.n_tokens` is
`N` (prefill) or `1` (decode).  The same ggml graph, the same CUDA
kernels, the same dispatch topology — only the size of one dimension
changes.

This is the opposite of hesper today, which has:

- `forwardSingleToken` — decode path, hand-written for N=1.
- `forwardPrefillBatch` — prefill path, a separate implementation that
  tries to reuse single-token helpers by looping `for i in [0:seqLen]`.

The two paths diverge in subtle ways (buffer layouts, kernel variants,
loop ordering), and multi-token prefill currently produces garbage
output (`"are are are…"` for `"Hello world how are you"`).  Local fixes
won't close this: **we need to make hesper look like llama.cpp**.

## Evidence: llama.cpp's path is shape-polymorphic

### Entry point: `llama_context::decode` → `process_ubatch`

`llama-context.cpp:1688` — **one** call, always with `LLM_GRAPH_TYPE_DECODER`:

```cpp
const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_DECODER,
                                   mctx.get(), status);
```

`ubatch.n_tokens` is `N` for prefill and `1` for decode.  There is no
`if (is_prefill) {...} else {...}` branching here.  Prefill is just
"decode with a bigger batch".

### Model graph: n_tokens is a dimension, not a loop

`llama.cpp/src/models/gemma4-iswa.cpp:30-226` is the **entire** Gemma 4
forward graph.  The only explicit loop is over the 42 transformer
layers.  Every tensor carries `n_tokens` as a dimension:

```cpp
// line 58
Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);

// line 78
Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

// line 87 — RoPE processes all tokens in one kernel dispatch
Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, freq_factors, ...);

// line 92 — attention processes all tokens in one FA kernel dispatch
cur = build_attn(inp_attn, wo, nullptr, Qcur, Kcur, Vcur, ...);

// line 197 — per-layer embedding gate: n_tokens-wide matmul
cur = build_lora_mm(model.layers[il].per_layer_inp_gate, cur);
// [n_embd_per_layer, n_tokens]
```

No `for (token = 0; token < n_tokens; ++token)` anywhere.  Every kernel
takes `n_tokens` as an implicit dimension (usually a batch axis over
matmul rows, or a sequence axis over the KV cache view).

### How decode gets its single row of logits out

The trick is `ggml_get_rows` at **the very end** of the last layer
(`gemma4-iswa.cpp:103-106`):

```cpp
if (il == n_layer - 1 && inp_out_ids) {
    cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
    inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
}
```

- Prefill (generate logits only for the last prompt token):
  `inp_out_ids = [N-1]` → the final tensor becomes `[n_embd, 1]`.
- Decode: `inp_out_ids = [0]` (or `[0, 1, …, N-1]` if `logits_all`).
- Fully-pooled: `inp_out_ids` covers every output position.

So from embedding through layer `n_layer - 1` the tensor is always
`[n_embd, n_tokens]`.  Only the **last step** collapses to the
`[n_embd, n_outputs]` we actually care about.  The lm_head matmul runs
on that reduced tensor — not on the full batch.

### KV cache: batch-aware scatter, not per-token write

Both prefill and decode use `ggml_set_rows` (`llama-kv-cache.cpp:1216,
1272`):

```cpp
ggml_set_rows(k_view, k_cur, k_idxs);  // k_cur: [n_embd_k_gqa, n_tokens]
```

`k_idxs` is a small u32 tensor of length `n_tokens` telling the kernel
where each token's KV slice goes.  On the CUDA backend this is
(optionally) fused with the preceding RoPE so that RoPE+set_rows becomes
**one** kernel dispatch regardless of `n_tokens`.

### FlashAttention: one call, any n_tokens

`ggml_flash_attn_ext(Q, K_view, V_view, mask)` where
`Q: [head_dim, n_heads, n_tokens]`.  The MMA-F16 kernel
(`fattn-mma-f16.cuh`) picks `ncols = 8` for prefill (N=8 queries per
block) and `ncols = 1` for decode — same kernel template, different
instance.  The graph doesn't know which; `ggml-cuda/fattn.cu:299-498`
selects at dispatch time.

## hesper today: a loop-nest mismatch

In `Hesper/Models/Gemma4.lean` the batched prefill path (starting at
`forwardPrefillBatch`, line 2620) follows the **opposite** pattern:

```
for block in blocks:                 -- 42 layers (OK — same as llama)
    for i in [0:seqLen]:             -- WRONG — llama doesn't loop here
        columnExtract(batch, i) → state.qBuf
        qkvNorm(state.qBuf)
        rope(state.qBuf)
        flashAttn(state.qBuf, kvCache[li])
        columnInsert(state.attnOutBuf, i) → batchAttnOutBuf
```

Every kernel is the single-token kernel.  Per token we pay:

- 3× column-extract dispatch (Q, K, V)
- 1× QKV-norm dispatch
- 1× RoPE-Q dispatch
- 1× RoPE-K + KV-write dispatch
- 1× FlashAttention dispatch
- 1× column-insert dispatch

`≈ 10 kernels × 9 tokens × 42 layers = 3,780` dispatches per prefill
from this single loop.  llama.cpp pays **one dispatch per op** for the
whole batch, so its equivalent is `10 × 42 = 420`.  Ratio 9:1,
matching seqLen exactly — it's pure loop-over-token overhead.

### Why this also breaks correctness

The per-token column-extract + single-token kernel path is reusing the
**decode-shaped** `state.qBuf`, `state.kBuf`, `state.vBuf`,
`state.normedBuf`, `state.attnResidualBuf` scratches.  Data flows:

```
batchQBuf[col i]   ──┐
batchKBuf[col i]    │
batchVBuf[col i]    │→  state.qBuf,kBuf,vBuf  →  (single-token pipeline)
                                                  →  state.attnOutBuf
                                                  ──── columnInsert ──→ batchAttnOutBuf[col i]
```

Any RMSNorm, RoPE, or attention ops operating on these scratches see
only a single column at a time.  If the single-token kernel implicitly
depends on state from "the previous token" (RMSNorm statistics, RoPE
position cache, shared-memory preloads tuned for N=1), that state is
now garbage: we've reset it between tokens, or worse, stale.  Hence
the known multi-token correctness bug.

## What llama.cpp's CUDA kernels look like from hesper's perspective

For each op the llama.cpp kernel is shape-parameterized, not loop-bound.
Concretely, the fix for each of hesper's per-token ops is:

| ggml op | current hesper | llama.cpp shape | what hesper needs |
|---------|---------------|-----------------|-------------------|
| `ggml_reshape_3d` on Qcur | `state.qBuf [qDim]` | `Qcur [head_dim, n_heads, n_tokens]` | single contiguous `[qDim * n_tokens]` buffer; kill the per-token extract |
| RMS_NORM (q_norm, k_norm) | `perHeadRMSNormKernel` dispatched N times | same kernel, dispatched once with `{ numWorkgroups := (n_heads, n_tokens, 1), … }` | add a y-axis = n_tokens; read/write `base[y*rowStride + x]` |
| RoPE_EXT | `ropeKernelDynamic` called N times per layer | one call over `[head_dim, n_heads, n_tokens]` | 2D or 3D dispatch; read `inp_pos[y]` |
| SET_ROWS (KV write) | single-token cache write called N times per layer | one call with `k_idxs [n_tokens]` | extend `fusedRopeKAndCacheWriteKernel` to loop over `y = n_tokens` internally; positions read from `inp_pos[y]`, cache offset = `cacheStart + inp_pos[y]` |
| FLASH_ATTN_EXT | `flashAttentionDynamicParamsKernel` called N times per layer | one call; kernel selects ncols=8 (pref) / 1 (decode) tile | add a query-batch dim to Q input; tile over ncols queries per block |
| `ggml_get_rows` (end of last layer) | no analogue | reduces `[n_embd, n_tokens]` → `[n_embd, 1]` | a `selectLastToken` copy, runs once per forward, replaces the current "extract last column then run single-token postprocessing" dance |

Everything else in the model — embedding lookup, Linear/Q4_K matmul,
per-layer embedding gate, FFN, residual adds, output scale, final norm,
lm_head — is already naturally batch-aware in llama.cpp (matmul has
an M dimension, pointwise ops broadcast across the batch axis).  hesper
has batched variants of most of these already; we just need to stop
un-batching them for the sake of the per-token attention loop.

## Proposed hesper rewrite: one `forward(N)` function

Replace both `forwardSingleToken` and `forwardPrefillBatch` with a
single:

```lean
def forward (ctx : β)
    (model : …) (state : …)
    (tokens : Array Nat)    -- N tokens; N = 1 for decode, N > 1 for prefill
    (positions : Array Nat) -- starting KV position per token
    (outIds : Array Nat)    -- which output rows we actually need logits for
    : IO Unit
```

Inside: one layer loop, all kernels take `N` as a dimension, no
per-token inner loop.  At the end, `selectRows(cur, outIds)` collapses
to just the requested output rows, and the final norm + lm_head run on
that.

For decode `N = 1`, `outIds = [0]` — functionally identical to the
old `forwardSingleToken`, but runs the batched kernels.  We can then
delete the single-token kernels whose only role was the per-token
prefill loop.

## Why this fixes both correctness and performance at once

1. **Correctness**: multi-token prefill no longer reuses decode-shaped
   scratches.  Each op sees its full batch in one go, so there's no
   hidden between-token state leakage.  If the batched kernels
   themselves are right, multi-token output becomes correct.
2. **Dispatch count**: attention inner loop goes from `10 × N × 42` to
   `10 × 42` dispatches (N× fewer).  For N=9, prefill drops from
   ≈ 3,780 to ≈ 420 dispatches → closer to llama.cpp's numbers.
3. **Future CUDA Graphs fit**: once `forward(N=1)` is the decode path
   AND has identical shape on every token, a single captured graph
   replays for all decode steps without re-capture.
4. **Unblocks further fusions**: llama.cpp's `rope_fused`,
   `rms_norm_fused_add`, and `mmvq has_fusion` all assume shape-stable
   tensors.  Porting them on top of hesper's per-token-loop path is
   more work than porting them on top of a batched path.

## Execution plan

### Phase 1 — Shape audit (no code change)

For each kernel touched by `forwardSingleToken`, classify:
- **Already N-aware**: accepts a sequence dim (e.g. `forwardBatchDP4A`).
- **Trivially extendable**: 1D dispatch that can become 2D by adding
  `y = n_tokens`.  Data layout is already contiguous.
- **Needs refactor**: depends on `[dim]` 1D buffers, shared-memory
  preloads tied to single-token, or implicit state.

Output: a checklist in a follow-up doc.

### Phase 2 — Batched attention inner loop

The highest-leverage target: the per-token loop at
`Gemma4.lean:2936-3046`.

1. Replace per-token QKV-norm dispatch with `(numHeads, n_tokens, 3)`
   grid.
2. Replace per-token RoPE with a batched RoPE that reads
   `inp_pos[y]`.
3. Extend `fusedRopeKAndCacheWriteKernel` to loop over `y = n_tokens`;
   cache offsets use `inp_pos[y]`.
4. Replace the per-token `flashAttentionDynamicParamsKernel` call with
   a single call whose Q has shape `[head_dim, n_heads, n_tokens]`.
   This is the only actually-new kernel we need — everything else is
   adding a dispatch dimension.
5. Drop all column-extract / column-insert dispatches around this
   block — Q/K/V are read straight from the batch matmul outputs.

### Phase 3 — Unify single-token and batched forward

Once Phase 2 is in, `forwardSingleToken` can be deleted and its callers
routed to `forwardPrefillBatch(tokens := [t], positions := [p],
outIds := [0])`.  Decode TPS should be ~the same (we're paying one
batched-kernel dispatch per op instead of one single-token dispatch
per op, and those are roughly the same cost at N=1).

### Phase 4 — Numerical validation, then perf

Only once the batched path produces correct output for `"Hello world
how are you"` (i.e. a sane completion, not `"are are are…"`) should we
start re-measuring TPS and re-attacking fusion.  Correctness first.

## What to discard after this lands

- `forwardSingleToken` in `Gemma4.lean`.
- Single-token `columnExtractKernel` / `columnInsertKernel` helpers
  (only needed to bridge batch-major scratches into single-token
  kernels).
- Debug-dump column-extract loops gated on `HESPER_DUMP_DIR` — once
  the forward path works on the native batched layout, `dumpBuf` can
  read slices directly.
- The ping-pong `currentBuf` / `nextBuf` bookkeeping specific to the
  batched path.

## Open questions

- **FlashAttention with ncols=8 tile**: hesper's current FA is decode-
  tuned (1 query per block).  Does the existing
  `executeFlashAttentionTiled` already support `nQueries > 1`, or is
  this a real new kernel?  Answer determines whether Phase 2 step 4 is
  "wire existing code" or "write new kernel".  Needs 10 min grep.
- **KV cache layout**: doc 12 notes we might want to switch to
  `[numKVHeads, max_seq, head_dim]` F16 to match llama.cpp.  Does
  the current hesper KV layout allow per-token `inp_pos[y]`
  scatter writes, or does it implicitly rely on "positions are
  contiguous and start at cacheLen"?
- **Output-ids trick**: hesper currently runs the final RMSNorm + lm_head
  only on the last prompt token (correctly mimicking `ggml_get_rows`).
  Is the row-select already a single kernel, or does hesper still do
  "extract last column → single-token norm → single-token lm_head"?
  The latter is fine as long as it's only done once per forward.

## Summary

Stop trying to beat llama.cpp with a different architecture.  Adopt
llama.cpp's architecture first — one forward path, shape-polymorphic in
N — then compete on kernel quality.  The local optimizations we've
been doing (caching JIT dispatches, gating debug dumps, eliminating
redundant HtoDs) each save ~1 ms in isolation; the architecture
mismatch is costing ~10× more dispatches than llama.cpp on prefill and
is the source of the multi-token correctness bug.  Fix the structure
and both gaps close.
