# True per-call top-5 vs llama.cpp (correction to doc 62)

Doc 62's "Q4_K matmul 1.30x slower" was misleading — it averaged
hesper's wO (heavy, 28-68µs) with llama.cpp's mix of all Q4_K shapes
(8.8µs lightweight + 64µs heavy + 3µs trivial).

Re-reading the raw nsys cuda_gpu_kern_sum on **same kernel-shape
pairs** gives a very different picture.

## hesper top 10 (graphs OFF, 60 tok decode)

| # | hesper kernel        | inst | µs/call |
|---|----------------------|-----:|--------:|
| 1 | `k1387` Q4_K wO matmul (2560×2560)         | 2509 |  **68.3** |
| 2 | `k7345` Q6_K ffn_down 1-row (10240→2560)  | 1756 |  43.0 |
| 3 | `k1031` Q6_K lm_head (262144 vocab)        |   84 | **869**  |
| 4 | `k6102` Q4_K perLayer matmul (prefill)     |   60 | 1140 (prefill, runs once) |
| 5 | `k1257` Q4_K post-attn matmul (2560×?)     | 1257 |  34.7 |
| 6 | `k1301` small reduce (gx=1 block=256)      | 4360 |   6.7 |
| 7 | `k1790` small reduce (gx=1 block=256)      | 5019 |   5.3 |
| 8 | `k1484` Q4_K small matmul                  | 2092 |   8.8 |

## llama.cpp top 10

| # | llama.cpp kernel                 | inst   | µs/call |
|---|----------------------------------|-------:|--------:|
| 1 | `mul_mat_vec_q<Q4_K>` (heavy)    |  2480  |  **64.2** |
| 2 | `mul_mat_vec_q<Q6_K>`            |  1951  |  66.2 |
| 3 | `mul_mat_vec_q<Q4_K>` (small)    | 10858  |   8.8 |
| 4 | `rms_norm_f32<1024>`             |  7748  |   3.1 |
| 5 | `quantize_q8_1`                  | 18443  |   1.0 |
| 6 | `flash_attn_ext_vec`             |  2065  |   6.5 |
| 7 | `mul_mat_vec_q<Q4_K>` (tiny)     |  2480  |   3.0 |
| 8 | **`mul_mat_vec_f<half,half>` (lm_head)** |    59 |  **114** |
| 9 | `k_bin_bcast<mul>`               |  5206  |   1.3 |
| 10 | `rope_neox`                      |  3410  |   1.1 |

## Direct same-role comparison

| role                     | hesper µs/call | llama.cpp µs/call | ratio | hs total (60 tok) | gap |
|--------------------------|----------------|-------------------|-------|-------------------|-----|
| **Q4_K wO matmul**       | 68.3           | 64.2              | **1.06x**  | 171 ms            | -10 ms |
| **Q6_K ffn_down**        | 43.0           | 66.2              | **0.65x**  | 75 ms             | (we win 35 ms) |
| **Q6_K lm_head**         | **869**        | **114**           | **7.6x**   | 73 ms             | **+62 ms gap** |
| Q4_K post-attn matmul    | 34.7           | 8.8 (small bucket) | (different shape) |              |     |
| small reduces × 2        | 6.0            | 1.3 (binary bcast) | 4.6x (per call) | 56 ms      |     |
| FlashAttn (V11)          | 6.6 partial+combine | 6.5 single | (matched per-call) | 67 ms (×10k inst) | inst count problem |

## Top 5 levers (corrected)

### 1. Q6_K lm_head — hesper 869 µs vs llama.cpp 114 µs (7.6× slower!)

llama.cpp uses `mul_mat_vec_f<__half, __half>` — f16 dequantized
weights × f16 input. hesper does a full **Q6_K × f32** matmul on the
fly.  60 ms of GPU time wasted on a one-shot decode op.

**Quick fix**: dequantize the Q6_K embedding to f16 once at load,
keep it in VRAM, use a tight f16 matmul kernel for lm_head.  Memory
cost: vocab × hiddenSize × 2 = 262144 × 2560 × 2 = 1.34 GB for f16
weights.  Already occupied by Q6_K weights at ~830 MB; this adds 0.5
GB.  At 11.9 GB total VRAM, we have headroom.

**Or**: read llama.cpp's lm_head dispatch — it might be using `MMV`
because llama.cpp keeps the embedding weights pre-dequantized to f16.
We can do the same.

**Estimated win**: -55 ms over 60 tok = -0.9 ms/tok = +7 TPS at
graphs OFF (65 → 72), or ~+5 TPS at graphs ON.

### 2. Small-reduce dispatches `k1301` + `k1790` (156 calls/token!)

Two tiny kernels, grid=(1,1,1), block=(256), 5-7 µs each.  **156
dispatches per decode token = 56 ms total.**

These are the per-block layerOutScale / postNormAdd / scale-and-add
wrappers that fire between major ops.  llama.cpp folds them into
`k_bin_bcast<op_mul>` (1.27 µs avg, 5206 inst) — same total work but
fewer launches.

**Quick fix**: identify the two specific wrappers (likely
`layerOutScale`, `postNormAddScale`, `embedScale`).  Either fuse them
into the preceding matmul's epilogue (Circuit DSL already supports
this), or batch them per-block.

**Estimated win**: 156 → 60 calls = -50% dispatch cost = -0.4 ms/tok
= +3 TPS.

### 3. Q4_K post-attn matmul — 34.7 µs/call × 1257 inst = 44 ms

`k1257` `(2560,1,128)` 1257 inst = 21 layers × 60 tok.  This is
likely the **post-attention output projection** (RMSNorm output then
postLinear matmul).  Per-call only 35 µs vs the wO's 68 µs, but
called less.

**Why slower than wO's 68/2 = 34**: wait, this IS roughly half wO's
time — same shape but maybe smaller inDim (1280 vs 2560?). Acceptable.

### 4. FlashAttn dispatch count

V11 partial+combine = 2 dispatches × 84 calls/tok = 168 dispatches.
llama.cpp does **1 dispatch per attention** (cacheLen ≤ ~256), so
~84.  hesper has **2× the dispatch overhead**.

**Fix**: when cacheLen < numSplits (8), skip split-K; use V7 1-shot.
That's already a small win for early decode tokens.

### 5. Q4_K wO ratio is actually 1.06x — already fine

The doc 62 "1.30x" was misleading.  Doc-62-rate: 354 ms total / 272
ms = 1.30x.  But that ratio is dominated by INSTANCE-COUNT
differences (hesper has more "wO-like" calls because we don't fuse
post-attn).  **Per-call time is matched**.

## Action plan (revised top 3)

1. **Q6_K lm_head: f16-cache the embedding weights**.  +5-7 TPS.
2. **Fuse small-reduce dispatches**: identify k1301+k1790,
   epilogue-fuse into preceding matmul.  +3 TPS.
3. **FlashAttn: dynamic split-K** (skip combine when cacheLen ≤ 8 or
   per-tok chunk is small).  +2 TPS.

Total estimated: **+10-12 TPS at graphs OFF** (65 → 75), **+8 TPS at
graphs ON** (94 → 100+).

Item 1 is the biggest single lever and the easiest — it's a
one-time-at-load preprocessing change, no kernel rewrite needed.
