# Kernel trace diff: llama.cpp vs hesper (prompt="Hello world how are you")

Captured 2026-04-20 via:
- `LLAMA_KERNEL_TRACE=1 llama-cli -p "Hello world how are you" -n 2 -st`
- `HESPER_KERNEL_TRACE=1 gemma4-cuda "Hello world how are you" 1`

Raw traces: `trace_lc.txt` (llama.cpp), `trace_hs.txt` (hesper).

## Totals

Raw line counts differ in coverage (llama.cpp ran further before the
prompt ended; hesper crashed at first decode FFN). **Do not compare
2670 vs 1677 directly** — the ranges are different.

Aligned on the **prefill-only** segment (5 tokens):
- **llama.cpp**: 690 ops (one ne[1]=5 batched forward pass)
- **hesper**: 1661 ops (embedLookup etc. still loop per-token, then
  batched path for the block body)

**Ratio 2.4×** on identical workload. This is the real gap.

## Per-decode per-layer dispatches (layer 0 attention, before crash)

### llama.cpp (layer 1, archetypal, 9 ops for attention half)
```
MUL_MAT  Qcur    ne=[2048,1]   # wQ
ROPE     Qcur_pos
MUL_MAT  Kcur    ne=[512,1]    # wK
ROPE     Kcur_pos
MUL_MAT  Vcur    ne=[512,1]    # wV
RMS_NORM Vcur_normed
SET_ROWS cache_k
SET_ROWS cache_v
FLASH_ATTN_EXT
MUL_MAT  attn_out
```

### hesper (layer 0 attention, 11 dispatches for attention half)
```
attnNorm       1 dispatch       # NOT fused into qkvProj epilogue
qkvProj        3 dispatches     # Q, K, V still separate (fallback path)
qkvNorm        1 dispatch
rope           2 dispatches     # Q + K
kvWrite        1 dispatch       # combined vs llama.cpp's 2 SET_ROWS
flashAttn      1 dispatch
oProj          1 dispatch
postAttnNorm   1 dispatch       # NOT fused into oProj or FFN
```

### Differences (attention half only)

| Op | llama.cpp | hesper | Gap |
|---|---:|---:|---|
| Norm before QKV | 0 (fused in epilogue) | 1 (attnNorm) | **+1** |
| QKV matmul | 3 | 3 | 0 |
| Norm on Q/K | fused via RMS_NORM Vcur (1) | 1 (qkvNorm) | 0 |
| RoPE | 2 | 2 | 0 |
| KV cache write | 2 (SET_ROWS) | 1 (kvWrite) | **-1** (hesper wins) |
| FlashAttn | 1 | 1 | 0 |
| O proj | 1 | 1 | 0 |
| Norm after O | 0 (fused) | 1 (postAttnNorm) | **+1** |

**Net attention gap**: hesper has **+1** attn-half op per layer (attnNorm + postAttnNorm − kvWrite fusion). Over 42 layers that's 42 extra ops.

## Key structural finding — CORRECTION

Initial reading said "126 RMSNorms are reducible". **This is wrong.**
Closer inspection of `Hesper/Models/Gemma4.lean` shows hesper already fuses:
- `forwardFusedNormQKV` — attnNorm + Q+K+V matmul = 1 kernel (13 KV-fused layers)
- `forwardFusedNormWQ` — attnNorm + wQ (18 shared-KV layers)
- `forwardFusedNormGateUp` — ffnNorm + gate + up = 1 kernel
- `forwardNormThenAdd` — postAttnNorm+add / postFFNNorm+add = 1 kernel each
- `fusedRMSNormQ8_1Kernel` — attnNorm + Q8_1 quantize = 1 kernel

What the trace shows as "attnNorm section" with 1 dispatch IS the fused
kernel — not a standalone RMSNorm. The section name is just the logical
label; the underlying launch already includes the matmul/quantize.

**Therefore the "norm dispatches" are NOT reducible — they are already
fused.** My earlier claim was a misreading of the section labels.

The remaining residual differences (hesper 1661 vs llama.cpp 690 at
prefill) come from:
- Embedding lookup ×5 vs ×1 (per-token loop still present)
- plPre chain inside layer loop (doc 10) rather than hoisted
- qkvProj fallback for 11 layers × 3 dispatches (wV=Q6_K blocks K+V fusion)
- rope Q+K separate (2 per layer)

Real structural gap is in **loop structure**, not **norm fusion**.

## Prefill batching

- llama.cpp: `ne=[hidden, 5, ...]` throughout — **one forward call, 5 tokens batched**
- hesper: initial token-by-token for embedding lookup (5× embedLookup-style dispatches), then batched path kicks in for most ops

This is partially addressed (see project_llama_single_path.md) but embedding and early plPre steps still loop per-token.

## Decode crash: forwardFusedNormGateUp dp4a precondition

After prefill succeeds, first decode fails at `ffnNormGateUp` with:
```
uncaught exception: forwardFusedNormGateUp: dp4a precondition failed; caller should fall back
```

The fast path has a shape/alignment precondition that fails for decode (N=1) but not prefill (N=5). Fallback logic is missing at this call site — unrelated to the trace work, but blocks decode comparison.

## Next concrete steps

1. Fix `forwardFusedNormGateUp` fallback so decode completes. Then re-run
   to get full-decode hesper trace.
2. Fold `attnNorm`, `postAttnNorm`, `postFFNNorm` into matmul epilogues
   (one per norm eliminated = 42 dispatches saved). This is the largest
   single reducible item visible from the trace.
3. Split `qkvProj` fallback (3 dispatches) into fused K+V or K+V+Q
   wherever weights are same-format Q4_K (already done for 13 layers; 11
   blocked by wV Q6_K).
