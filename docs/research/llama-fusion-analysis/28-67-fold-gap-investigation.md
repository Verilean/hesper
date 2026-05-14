# hesper vs llama.cpp — 5.7× kernel-instance gap: investigation report

**Date**: 2026-04-21
**Status**: source-of-truth list + measurement methodology, NOT conclusions

This document compiles every file you need to read to understand where
the 5.7× nsys instance ratio comes from.  Earlier interpretations (the
"-126 from inline quant fixes everything", the "matmul-with-norm-fused
kernel will close the gap", etc.) were **partial** — they explain
kilobytes of dispatch reduction without addressing the 9 595 tiny
`main` kernels that dominate the actual GPU time.

## 1. The number itself: where does 5.7× come from?

### Origin

`docs/llama-fusion-analysis/14-nsys-fresh-comparison.md` (full text below)

### Reproduce command

```bash
# hesper
HESPER_CUDA_GRAPHS=1 HESPER_DP4A=1 nsys profile -t cuda -s none \
  --cuda-memory-usage=false -o /tmp/hesper.nsys-rep --force-overwrite=true \
  ./.lake/build/bin/gemma4-cuda data/gemma-4-e4b-it-Q4_K_M.gguf "Hello world how are you" 30

# llama.cpp
nsys profile -t cuda -s none --cuda-memory-usage=false \
  -o /tmp/llamacpp.nsys-rep --force-overwrite=true \
  ./llama.cpp/build/bin/llama-cli -m data/gemma-4-e4b-it-Q4_K_M.gguf \
  -p "Hello world how are you" -n 30 -ngl 99 -dev CUDA0 -no-cnv

# Then for both:
nsys stats --report cuda_gpu_kern_sum /tmp/{hesper,llamacpp}.nsys-rep
```

### What you get (workload: prefill 9 tokens + decode 30 tokens)

|                       | hesper     | llama.cpp  | ratio |
|-----------------------|-----------:|-----------:|------:|
| Total GPU kernel time | 114.07 ms  | 16.80 ms   | 6.8×  |
| Kernel instances      | 14 758     | 2 602      | 5.7×  |
| Avg kernel time       | 7.7 µs     | 6.5 µs     | 1.2×  |

So 5.7× = **(hesper kernel instance count) / (llama.cpp kernel
instance count)** measured by `nsys cuda_gpu_kern_sum`, same workload,
same hardware, same nsys version.

This is NOT the same metric as:
- `HESPER_DISPATCH_COUNT=1` host-side counter (reports 920/token, see §6)
- llama.cpp trace `[lc] op=` line count (reports 577/token, see §7)

Different counters → different absolute numbers.  The **ratio** under
the same metric is what matters.

## 2. Top kernels — where the time actually lives

### hesper top-5 (kernel name, total ms, instances, avg µs)

```
main                                    53.30  9 595   5.6     ← biggest cost
k_5827556345714019                      35.83    351  102.1    ← single hottest
k_7031743127946451                      10.42    189   55.1
k_1061309516933780                       2.95    342    8.6
k_1790517551769375                       2.93    684    4.3
```

**`main`** is hesper's default kernel name when the caller doesn't
override it.  9 595 instances × 5.6 µs = 53 ms — **47% of GPU time**.
These are tiny grid=(1,1,1) or grid=(N,1,1) helper kernels that hesper
emits for: pleScale, embedScale, layerOutScale, residualAdd,
columnExtract/Insert, advancePos, copyU32, etc.  Each is fast in
isolation but the launch overhead dominates.

**`k_5827556345714019`** (grid=(10752,1,1), block=(32,1,1)) = 102 µs ×
351 instances = 36 ms.  10752 = 42 × 256 = numLayers × embdPerLayer →
batched PLE pre-layer Q6_K dequant or similar.

### llama.cpp top-5

```
mul_mat_vec_q<Q4_K, 1, has_fusion=1>     5.39   84   64.2
mul_mat_vec_q<Q6_K, 1, 0> (lm_head)      4.34   66   65.8
mul_mat_vec_q<Q4_K, 1, 0>                3.21  368    8.7
rms_norm_f32<1024,1,1> (fused scale)     0.76  250    3.0
quantize_q8_1                            0.59  602    1.0
```

llama.cpp's **biggest cost is the matmul itself** (5.4 ms).  Helper
kernels (rms_norm + quantize) are ~1.4 ms total.  No "tiny main" pile.

### What the gap actually is

```
hesper time   = 53 (main pile) + 36 (one PLE kernel) + 24 (everything else)
llama.cpp time = 14 (matmuls)  +  2.8 (helpers)
```

So:
- **The "main pile" alone (53 ms) is 3× llama.cpp's TOTAL (16.8 ms).**
- "Fix the matmul fusion" can shave matmul time but won't touch the
  main pile.

## 3. The two real fusion gaps llama.cpp has that hesper doesn't

### Gap A — quantize is shared, not per-matmul

llama.cpp: 602 quantize_q8_1 instances / 30 tokens = **20/token**.

hesper estimated: 5 quantize/layer × 42 layers + lmHead = **~210/token**.

Source: in llama.cpp every `mul_mat_vec_q` cluster shares its Q8_1
input.  When QKV are computed from the same normedBuf,
`quantize_q8_1` runs **once**, then 3 mul_mat_vec_q reads from it.
hesper does this for fused_norm_qkv etc., but NOT for the standalone
matmuls (oProj, ffnDown, ple.proj) where each matmul re-quantizes its
own input.

### Gap B — `mul_mat_vec_q<has_fusion=1>` does gate+up+GLU in one kernel

llama.cpp: 84 fused matmul / 30 tokens = **~2.8/layer**, perfect match
for FFN's gate+up matmul fused with GeGLU.

hesper: `forwardFusedNormGateUp` does the same thing.  This gap is
already closed for FFN, NOT for any other site (PLE inpGate+geluMul is
2 dispatches, etc.).

## 4. The `main` pile — the 47% of GPU time nobody is talking about

`main` is the kernel name hesper uses when:
- The caller of `executeWithConfig` doesn't override `funcName`
- The cacheKey in `executeWithConfigCached` is 0

What ends up named `main` in the nsys output is the **vast majority of
small helper kernels**:
- `embedScale`, `layerOutScale`, `pleScale1`, `pleScale3`
- `residualAdd`
- `columnExtract`, `columnInsert`
- `copyU32`, `writeColIdxU32`
- `copyBuffer`
- The token-graph `argmaxKernel`, `advancePosKernel` (if HESPER_TOKEN_GRAPH)

Some are unavoidable (argmax MUST run).  Many are not — they exist
because the hesper code path issues them as separate IO actions
instead of folding them into the next "real" kernel.

To attack this you must:
1. Run the trace (`HESPER_KERNEL_TRACE=1`), filter `[hs] main grid=...`
2. Cross-reference each grid signature with the `withSection` markers
3. For each, decide: can it be folded into the next kernel?  fused
   pointwise pass?  precomputed constant?

There is no single "kill the 9 595 main kernels" trick.  It is 9 595
opportunities at maybe 1-5 µs each, mostly under-occupied.

## 5. Files to read — hesper side

| Purpose | Path |
|---|---|
| Forward block (single token) | `Hesper/Models/Gemma4.lean` ~line 348 `def forwardBlock` |
| Forward batch (prefill / unified) | `Hesper/Models/Gemma4.lean` ~line 1144 `def forwardPrefillBatch` |
| Token loop + CUDA Graph | `Hesper/Models/Gemma4.lean` ~line 2643 `def generate` |
| Linear matmul dispatch (Q4_K/Q6_K dp4a) | `Hesper/Layers/Linear.lean` ~line 3490 `def forwardDP4A` |
| Inline quant kernel (WIP) | `Hesper/Layers/Linear.lean` ~line 2089 `fusedQ4KMLinearDP4A4WarpInlineQuantKernel` |
| Norm + add fused kernels | `Hesper/Layers/RMSNorm.lean` ~line 216 `rmsNormThenAddKernel` |
| PLE post-norm + add (just landed: + scale) | `Hesper/Models/Gemma4/Kernels.lean` ~line 118 `fusedPerLayerPostKernel`, +`fusedPerLayerPostThenScaleKernel` |
| Per-section dispatch counter | `Hesper/Backend/CUDA.lean` ~line 86 `dispatchCounter`; `Hesper/WGSL/Execute.lean` `withSection` |
| Token-graph capture | `Hesper/Models/Gemma4.lean` ~line 2786 `if tokenGraph && useCudaGraphs` |
| Loader (model weights) | `Hesper/Models/Gemma4/Loader.lean` |

## 6. Files to read — llama.cpp side

| Purpose | Path |
|---|---|
| Top-level dispatch + graph builder | `llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu` ~line 2486 `ggml_cuda_compute_forward` |
| Graph fusion driver: detects `{RMS_NORM, MUL, ADD}` etc. | `llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu` ~line 3380 (rms-norm chain) and ~line 4005 (`ggml_cuda_can_fuse`) |
| Add-chain fusion (`{ADD, ADD, ...}` → 1 kernel) | `llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu` ~line 3802 `ggml_cuda_op_fused_add` |
| The fused matmul kernel itself | `llama.cpp/ggml/src/ggml-cuda/mmvq.cu` ~line 389 `template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false> ... mul_mat_vec_q` |
| Q8_1 quantize | `llama.cpp/ggml/src/ggml-cuda/quantize.cu` |
| RMSNorm + fused variants | `llama.cpp/ggml/src/ggml-cuda/norm.cu` |
| FlashAttention | `llama.cpp/ggml/src/ggml-cuda/fattn-vec.cuh` (decode), `fattn-mma-f16.cuh` (prefill) |

## 7. Earlier hesper docs that informed this report

| Doc | Key claim |
|---|---|
| `docs/llama-fusion-analysis/14-nsys-fresh-comparison.md` | Source of the 5.7× number |
| `docs/llama-fusion-analysis/26-llama-kernel-trace-analysis.md` | 19 ggml ops/layer breakdown for llama.cpp |
| `docs/llama-fusion-analysis/27-kernel-trace-diff.md` | 920 vs 1661 *prefill* trace count diff (different metric, do not confuse with §1) |
| `memory/project_dispatch_measurement.md` | hesper per-section dispatch breakdown via `HESPER_DISPATCH_COUNT=1` |
| `memory/project_kernel_trace_comparison.md` | How to capture both traces |
| `memory/project_decode_stall_truth.md` | Earlier hypothesis: "fuse quantize_q8_1 into Q4_K matmul (-126 graph nodes/tok)" |
| `memory/project_inline_quant_kernel.md` | Code for that fusion landed under `HESPER_INLINE_QUANT=1`; PTX JIT >10 min, runtime not yet validated |

## 8. Methodology checklist for the next investigator

To make claims about "fusing X closes the gap", you must:

1. Re-run the §1 nsys command on a **clean rebuild**
2. Confirm the 14 758 / 2 602 baseline (or note drift)
3. Apply your change with an env flag toggle
4. Re-run §1 with the flag ON, get new instance count and total GPU time
5. Show: instance delta, GPU time delta, and which top-5 row moved
6. Cross-check: did `main` count drop?  did the dominant kernel
   (k_5827556345714019) drop?  if neither, the change isn't on the
   critical path

Skipping step 6 is how earlier docs ended up claiming that
norm-into-matmul fusion would close the gap.  It won't, because the
gap isn't in the matmul; it's in the 9 595 main kernels and the one
102-µs PLE kernel.

## 9. What would actually close the gap (rough estimate)

| Action | Time saved (est.) | Mechanism |
|---|---:|---|
| Fold half of the 9 595 `main` kernels into adjacent ops | 26 ms | reduce 53 → 27 |
| Replace `k_5827556345714019` with the per-token PLE path | 30 ms | 36 → 6 |
| Inline quant fusion (#163, currently JIT-stuck) | ~3 ms | quantize 0.6 ms × ~5 sites |
| Per-matmul shared quantize (gap A in §3) | ~1 ms | small |
| **Sum** | **~60 ms saved** | hesper 114 → 54 ms ≈ llama.cpp + 3× |

Even with all four, hesper would still be ~3× llama.cpp.  The
remaining 3× is per-kernel speed (Avg 7.7 µs vs 6.5 µs on the matmuls
themselves) — a PTX-quality / register-pressure problem, not a
fusion-count problem.

## 10. Honest summary

The 5.7× ratio is real and measured under one consistent metric.  The
naive interpretation ("close the fusion gap and we win") is wrong:
fusion alone gets us maybe 2× back.  The other ~3× lives in:
- per-kernel GPU efficiency (PTX codegen quality)
- the 9 595 tiny helper kernels that exist because hesper builds the
  forward pass as a sequence of small Lean IO actions instead of
  fewer, bigger ggml-style ops

To be honest about progress: every dispatch-reduction work item
landed so far (E layerOutScale fusion, fusedPerHeadQKVNorm, batched
RMSNorm+add, etc.) attacks the *count*, not the *time-dominant*
kernels.  The next high-ROI work is investigating what
`k_5827556345714019` actually is and whether the "main pile" can be
batched.

## Appendix A — full text of doc 14

(verbatim from `14-nsys-fresh-comparison.md`)

Read the file directly — too long to inline here, but the §2 top-5
tables above are the load-bearing rows.

## Appendix B — example trace lines

From `docs/llama-fusion-analysis/trace_lc.txt` (llama.cpp, layer 1
attention, decode token, ne[1]=1):

```
[lc] op=MUL_MAT name=Qcur-1 ne=[2048,1,1,1]
[lc] op=ROPE name=Qcur_pos-1 ne=[256,8,1,1]
[lc] op=MUL_MAT name=Kcur-1 ne=[512,1,1,1]
[lc] op=ROPE name=Kcur_pos-1 ne=[256,2,1,1]
[lc] op=MUL_MAT name=Vcur-1 ne=[512,1,1,1]
[lc] op=RMS_NORM name=Vcur_normed-1 ne=[256,2,1,1]
[lc] op=SET_ROWS name=cache_k_l1 (view) ne=[512,1024,1,1]
[lc] op=SET_ROWS name=cache_v_l1 (view) ne=[512,1024,1,1]
[lc] op=FLASH_ATTN_EXT name=__fattn__-1 ne=[256,8,1,1]
[lc] op=MUL_MAT name=node_91 ne=[2560,1,1,1]
```

That's 10 ggml ops for one layer's attention half.  Each MUL_MAT
becomes 1 quantize_q8_1 + 1 mul_mat_vec_q internally → 4 more kernels
→ 14 actual GPU kernels.  Per layer × 42 = 588.  Add FFN (6 ggml ops →
~10 GPU kernels) and the per-layer total is ~25 GPU kernels.

42 layers × 25 = 1 050.  Per token (with prefill share over 30 +
prefill) ≈ 87/token.  This matches the nsys count.  hesper runs ~492/
token.  Ratio 5.7×.
