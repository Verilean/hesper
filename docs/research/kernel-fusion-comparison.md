# Kernel Structure & Fusion Comparison: hesper vs llama.cpp

For Gemma 4 E4B Q4_K_M single-token decode. Companion to
`llamacpp-kernel-analysis.md` (which has TPS numbers).

This document focuses on **what each kernel contains** and **what is
bundled together** (kernel fusion). Optimization gain estimates appear
in the rightmost column.

---

## Per-layer attention block

| # | Op | llama.cpp kernel(s) | hesper kernel(s) | Fusion gap | Est. gain if hesper matches |
|---|---|---|---|---|---|
| 1 | attn_norm (RMSNorm) | `rms_norm_f32<1024,true,true>` | `RMSNorm.forward` (rmsNormKernel) | both separate | — |
| 2 | Q proj (Q4_K matmul) | `mul_mat_vec_q<Q4_K, ncols=1>` (4-warp, 64µs avg) | `LinearLayer.forward` → dp4a kernel (1-warp, 75µs) | 4× warps in llama.cpp | (covered in matmul section) |
| 3 | K proj | same `mul_mat_vec_q<Q4_K>` | same | same | — |
| 4 | V proj | same | same | same | — |
| 5 | Q-norm/K-norm/V-norm (per-head RMSNorm) | (small `rms_norm_f32<256>` calls) | `perHeadRMSNormKernel` × 3 | hesper has **3 separate dispatches**; could fuse with prev matmul output | +0.5 TPS |
| 6 | RoPE (Q and K) | `rope_neox<true,false,float>` × 2 | `ropeWithFreqFactorsKernel` × 2 | comparable | — |
| 7 | KV cache write | `k_set_rows` × 2 (K and V separate, ~1.2µs each) | `fusedCacheWriteKVKernel` (K+V in 1) | **hesper advantage** (already fused) | (already saved) |
| 8 | FlashAttention | highly tuned single fused kernel | `flashAttentionDynamicParamsKernel` or split-K tiled | hesper ~3× slower per head; tuning gap | +1-2 TPS |
| 9 | O proj | `mul_mat_vec_q<Q4_K>` | `LinearLayer.forward` (dp4a, 2-row variant) | comparable per-call but lacks 4-warp | +0.5 TPS |
| 10 | post_attn_norm + residual add | (likely fused/optimized) | `RMSNorm.forward` + `residualAddKernel` (2 dispatches) | hesper splits; could fuse | +1 TPS |

## Per-layer FFN block (dense path)

| # | Op | llama.cpp | hesper | Fusion gap | Est. gain |
|---|---|---|---|---|---|
| 11 | ffn_norm | `rms_norm_f32` | `RMSNorm.forward` | both separate | — |
| 12-14 | **gate proj + up proj + GEGLU** | **1 fused kernel**: `mul_mat_vec_q<Q4_K, has_fusion=true>` (computes gate × W_gate, up × W_up, GEGLU(gate)*up internally; 64µs total) | **3 separate dispatches**: `LinearLayer.forward` for gate (75µs) + same for up (75µs) + `geluMulKernel` (5µs) ≈ 155µs | **biggest single fusion gap** | **+5-10 TPS** |
| 15 | down proj | `mul_mat_vec_q<Q4_K>` | dp4a kernel (386µs!! — high gap) | 4-warp + small_k optimizations | +6-15 TPS |
| 16 | post_ffn_norm + residual | (likely fused) | `RMSNorm.forward` + `residualAddKernel` | hesper splits | +1 TPS |

## Per-layer-embedding ops (Gemma 4 specific)

| # | Op | llama.cpp | hesper | Fusion gap |
|---|---|---|---|---|
| 17 | per_layer_input precompute | (in graph build, possibly fused) | dequant + matmul (2 dispatches) | comparable |
| 18 | per_layer_inp_gate matmul | small matmul kernel | `LinearLayer.forward` (small Q4_K) | comparable |
| 19 | per_layer GELU × multiply | small kernel or fused | `geluGateMulSliceKernel` | comparable |
| 20 | per_layer_proj matmul | small matmul | `LinearLayer.forward` | comparable |
| 21 | per_layer post-norm + residual | (likely separate) | **`fusedPerLayerPostKernel`** (norm+scale+residual fused) | **hesper advantage** |

## Final stages

| # | Op | llama.cpp | hesper | Fusion gap |
|---|---|---|---|---|
| 22 | final_norm | `rms_norm_f32` | `RMSNorm.forward` | both separate |
| 23 | lm_head (Q6_K matmul, vocab=262144) | `mul_mat_vec_q<Q6_K>` (~130µs total per token) | `fusedQ6KLinearDP4AKernel` (1742µs!!) | **13× slower**; needs occupancy fix |
| 24 | logit softcap | inline | inline | no gap |

---

## What this tells us

### Bundled in hesper that llama.cpp splits
- **KV cache K+V write** in one kernel (Attention.lean:620) — small win, already done
- **per_layer_post: norm+scale+residual** in one kernel (PerLayerEmbedding.lean:196) — small win, already done

### Bundled in llama.cpp that hesper splits (the actionable gaps)

1. **Gate + Up + GEGLU** — single Q4_K matmul kernel with `has_fusion=true`
   - llama.cpp: 1 kernel, 64µs/layer
   - hesper: 3 dispatches, 155µs/layer
   - Per layer per token: 91µs savings × 42 layers × 30 tokens = **115ms / 30tok = 3.8 ms/tok**
   - **Already have skeleton**: `Linear.lean:2227` `forwardFusedGateUp` exists but **NOT WIRED** into `Gemma4.lean:forwardBlock` (line 1644-1645 has separate calls)
   - **Gain: +5-10 TPS, cost: 1 day** (just wire the existing function)

2. **post_attn_norm + residual_add** — likely 1 op in llama.cpp's graph
   - hesper: 2 dispatches (norm then add)
   - Per layer per token: ~30µs × 42 × 30 = 38ms / 30tok = 1.3 ms/tok
   - **Gain: +1-2 TPS, cost: 0.5 day** (write fusedNormAddKernel)

3. **post_ffn_norm + residual_add** — same pattern, second instance per layer
   - **Gain: +1-2 TPS, cost: 0.5 day** (same kernel as above, just call site change)

4. **Q-norm/K-norm/V-norm (3 per-head RMSNorms)** — could fuse since they take Q/K/V outputs of the same parent matmul
   - Per layer per token: ~3 × 17µs × 42 × 30 = 64ms / 30tok = 2.1 ms/tok (sum of qkvNorm in profile)
   - **Gain: +1 TPS, cost: 1 day** (write fusedQKVNormKernel)

### Per-call performance gaps (NOT a fusion issue, but a within-kernel issue)

Even when fused similarly, individual kernel calls are slower in hesper. These are separate from fusion — they need within-kernel work (cooperativeWarps, ld.global.nc, etc.):

| Op | llama.cpp µs/call | hesper µs/call | gap |
|---|---|---|---|
| Q4_K matmul (single proj) | ~64 | ~75 | 1.2× (dp4a 1-warp vs 4-warp) |
| ffnDown Q4_K | ~10-65 | 386 | 6-44× (worst case) |
| Q6_K lmHead | ~130 | 1742 | 13× |
| Q8_1 quantize | ~1 | ~5.5 | 5× (vector load missing) |

These need:
- **4-warp cooperation** for Q4_K matmul → +15-25 TPS
- **`ld.global.nc`** read-only L1 hint → +5-10 TPS
- **Vector loads** in Q8_1 quantize → +1-2 TPS

---

## Recommended fusion roadmap

**Phase F1 — Wire the existing fused kernel (1 day, +5-10 TPS)**

Just call `Linear.forwardFusedGateUp` from `Gemma4.lean:forwardBlock`
where the dense FFN dispatches gate+up+GEGLU. Already implemented in
`Linear.lean:2227`. Need to verify it works with the dp4a path (may
need a dp4a variant of forwardFusedGateUp).

**Phase F2 — Norm+residual fusion (1 day, +2-3 TPS)**

Add `fusedNormResidualKernel(norm_input, residual, scale, eps)`:
```
y[i] = (x[i] / rms(x) * scale[i]) + residual[i]
```
Apply at `post_attn_norm` (Gemma4.lean:1515-1520) and `post_ffn_norm`
(Gemma4.lean:1655-1660).

**Phase F3 — qkv-norm fusion (1 day, +1 TPS)**

Combine 3 per-head RMSNorms (Q, K, V) into 1 kernel with 3 buffer
outputs. May need separate dispatch per (Q,K,V) due to head-count
differences (Gemma 4 GQA: numHeads=8 for Q, numKVHeads=2 for K/V).
Alternative: fuse Q-norm with the Q matmul output processing.

**Phase F4 — Q8_1 quantize+matmul fusion (2-3 days, +3-5 TPS)**

Inline the Q8_1 quantization into the Q4_K matmul kernel. Each
workgroup quantizes its own slice of input into shared memory before
the matmul loop. Eliminates ~170 small dispatches/tok and the
intermediate q8 buffer.

**Cumulative Phase F1-F4: +11-19 TPS**, hesper 28 → 39-47 TPS.

After fusion is exhausted, the next big chunk requires per-call
optimization (cooperativeWarps, ld.nc) — that's Phase A/B in the
performance roadmap.

---

## Source code citations

### llama.cpp (CUDA backend, ggml/src/ggml-cuda/)
- `mmvq.cu:391-589` — `mul_mat_vec_q` template kernel with `has_fusion`
- `mmvq.cu:565-580` — GLU activation switch (SWIGLU/GEGLU/SWIGLU_OAI)
- `common.cuh:1458-1463` — `ggml_cuda_mm_fusion_args_device` struct
- `vecdotq.cuh:502-524` — chained dp4a in `vec_dot_q4_K_q8_1_impl_vmmq`
- `quantize.cu:175-271` — vectorized `quantize_mmq_q8_1`

### hesper (Lean source)
- `Hesper/Models/Gemma4.lean:1366-1710` — `forwardBlock`
- `Hesper/Models/Gemma4.lean:1644-1648` — separate gate/up/geluMul dispatches (the gap)
- `Hesper/Layers/Linear.lean:2227-2295` — `forwardFusedGateUp` (defined but not used)
- `Hesper/Layers/Linear.lean:2006-2050` — `forwardDP4A` (single matmul path)
- `Hesper/Layers/Linear.lean:1010-1163` — `fusedQ4KMLinearDP4AKernel`
- `Hesper/Layers/RMSNorm.lean:70+` — `rmsNormKernel`
- `Hesper/Layers/Attention.lean:620` — `fusedCacheWriteKVKernel` (already fused — hesper advantage)
- `Hesper/Layers/PerLayerEmbedding.lean:196` — `fusedPerLayerPostKernel` (already fused — hesper advantage)

---

Last updated: 2026-04-15.
