# llama.cpp kernel trace analysis — prompt="Hi", -n 2

Captured via `LLAMA_KERNEL_TRACE=1 llama-cli -p "Hi" -n 2`.
Raw trace: `trace_lc.txt` (2672 `[lc]` events).

## Top-level shape: ne[1] = 2

llama.cpp batches prefill "Hi" (1 tok) + first decode (1 tok) through one
shape-polymorphic graph — `ne=[hidden,2,...]` throughout layer ops. All
later decode steps reuse the same graph with `ne[1]=1`.

This is the fundamental difference with hesper: hesper loops over tokens
with separate forward passes; llama.cpp treats batch dim as a runtime
parameter threaded through one graph.

## Per-layer op list (layer 1, archetypal dense layer, 19 ops)

| # | op | shape | notes |
|---|----|-------|-------|
| 1 | MUL_MAT | Qcur ne=[2048,2,1,1] | wQ @ normed_x |
| 2 | ROPE | Qcur_pos ne=[256,8,2,1] | |
| 3 | MUL_MAT | Kcur ne=[512,2,1,1] | wK @ normed_x |
| 4 | ROPE | Kcur_pos ne=[256,2,2,1] | |
| 5 | MUL_MAT | Vcur ne=[512,2,1,1] | wV @ normed_x |
| 6 | RMS_NORM | Vcur_normed ne=[256,2,2,1] | Gemma 4 post-V RMSNorm |
| 7 | SET_ROWS | cache_k_l1 ne=[512,1024,1,1] | KV cache write |
| 8 | SET_ROWS | cache_v_l1 ne=[512,1024,1,1] | |
| 9 | FLASH_ATTN_EXT | __fattn__-1 ne=[256,8,2,1] | |
| 10 | MUL_MAT | wO post-attn ne=[2560,2,1,1] | |
| 11 | MUL_MAT | ffn_gate ne=[10240,2,1,1] | |
| 12 | MUL_MAT | ffn_up ne=[10240,2,1,1] | |
| 13 | GLU | ffn_geglu ne=[10240,2,1,1] | swiGLU(gate, up) |
| 14 | MUL_MAT | ffn_out ne=[2560,2,1,1] | |
| 15 | MUL_MAT | PLE_inp ne=[256,2,1,1] | |
| 16 | UNARY | GELU ne=[256,2,1,1] | |
| 17 | MUL | gelu*slice ne=[256,2,1,1] | |
| 18 | MUL_MAT | PLE_proj ne=[2560,2,1,1] | |
| 19 | MUL | l_out ne=[2560,2,1,1] | |

Total: **19 ops/layer**. No RMSNorm before attn / before FFN in the trace —
they get fused into the `mul_mat_vec_q` epilogue (see doc 02).

## Per-layer op **count** table (aggregate)

| op | count across trace | per-layer avg |
|----|-------:|-------:|
| MUL_MAT | 1282 | ~30 |
| MUL | 334 | ~8 |
| ROPE | 264 | 6.3 |
| SET_ROWS | 192 | 4.6 |
| FLASH_ATTN_EXT | 168 | 4 |
| UNARY | 167 | 4 |
| GLU | 123 | 2.9 |
| RMS_NORM | 96 | 2.3 |
| SCALE | 16 | — |
| GET_ROWS | 9 | — |
| CPY | 8 | — |
| ADD | 7 | — |
| CONT | 4 | — |

(Covers prefill + 1 decode + lm_head, then trace was truncated by interactive
hang — still enough for structural analysis.)

## Layer 0 extras (26 ops vs 19 elsewhere)

Layer 0 uniquely includes: SCALE (inp_scaled), CPY (kq_mask), SET_ROWS×2
+ MUL_MAT for PLE_inp_per_layer setup (ADD/SCALE/CONT), and ends with a
`MUL name=node_58` that looks like residual hook-up. These 7 extras are
per-forward-pass overhead, not per-layer.

## What's missing that hesper emits

No explicit RMS_NORM op for pre-attn or pre-FFN — **fused into the next
MUL_MAT's epilogue**. hesper currently emits:
- `attnNorm` standalone (11 fallback layers × 42)
- `postAttnNorm` standalone (42)
- `postFFNNorm` standalone (42)
- `ffnNormGateUp` — fused on some layers but separate dispatch

**Gap**: llama.cpp emits ~19 ops/layer; hesper internal counter shows
~22/layer (see project_dispatch_measurement.md). After accounting for
hesper's quantize+matmul 2-dispatch pattern (which llama.cpp also has),
the structural difference is that hesper counts several norm dispatches
llama.cpp folds into matmul epilogues.

## FUSED traces observed

Zero. `[lc] FUSED rope+view+set_rows` and `[lc] FUSED add_chain` never
fired in this trace — either my instrumentation isn't reaching the right
code path, or these fusion sites don't activate for Gemma 4 / this batch
size. The MUL_MAT-level fusion (RMSNorm+quantize+matmul+add) happens
inside the kernel, not at dispatch time, so it wouldn't show a FUSED line.

## Next step: hesper trace

Run `HESPER_KERNEL_TRACE=1 gemma4-cuda "Hi"` for same prompt, diff
against this trace. Key questions:
1. How many ops/layer does hesper emit?
2. Which ops have no llama.cpp counterpart (pure overhead)?
3. Which llama.cpp ops take >1 hesper dispatch (fusion candidates)?
