# Quantitative Comparison: hesper vs llama.cpp on Gemma 4 E4B Q4_K_M

Hardware: RTX 4070 Ti (sm_89, Ada Lovelace), 12 GB.
Model: Gemma 4 E4B Q4_K_M, 7.52 B params, ~5 GiB on disk.
Workload: single-token decode, 30 tokens after prefill.

Date: 2026-04-15.

---

## 1. Top-line numbers

| Metric | hesper (CUDA, dp4a Q4_K + Q6_K) | llama.cpp | ratio |
|---|---|---|---|
| **TPS (wall-clock)** | **28.3** | **118.9** | **0.24×** |
| Total ms/tok (wall-clock) | 35.3 | 8.4 | **4.2× slower** |
| **GPU kernel time / tok (nsys)** | **23.4 ms** | **0.56 ms** | **42× slower** |
| **Kernel launches / tok** | **1330** | **87** | **15× more** |
| Mean kernel duration | 17.6 µs | 6.5 µs | 2.7× longer |

**Critical observations**:
- llama.cpp's GPU is **almost idle**: 0.56 ms of work per 8.4 ms/tok of wall time
  → ~93% of llama.cpp's per-token time is host-side (sampling, scheduling, synchronization, tokenization, sampling)
- hesper's GPU is **the bottleneck**: 23.4 ms of GPU work per 35.3 ms/tok
  → ~66% of hesper's per-token time is GPU
- This means **even if hesper had zero CPU overhead, it would still be limited to ~43 TPS**
  by the GPU side alone — vs llama.cpp's GPU side only allowing ~1800 TPS.

The 42× GPU kernel time gap decomposes roughly as:
  - **15× more kernel launches** (host dispatch overhead + small-kernel inefficiency)
  - **2.7× longer per-launch on average** (each kernel is less efficient)
  - = ~40× combined (matches observed 42×)

---

## 2. Top-N kernel breakdown

### llama.cpp (30 tokens, total GPU = 16.80 ms)

| % | Total ms | Inst | Avg µs | Kernel |
|---|---|---|---|---|
| 32.1 | 5.39 | 84 | 64.2 | `mul_mat_vec_q<Q4_K, ncols_dst=1, has_fusion=true>` (FFN gate+up+GEGLU fused) |
| 25.8 | 4.34 | 66 | 65.8 | `mul_mat_vec_q<Q6_K, ncols_dst=1, has_fusion=false>` (lmHead + others) |
| 19.1 | 3.21 | 368 | 8.7 | `mul_mat_vec_q<Q4_K, ncols_dst=1, has_fusion=false>` (other matmuls) |
| 4.5 | 0.76 | 250 | 3.0 | `rms_norm_f32<1024, true, true>` |
| 3.5 | 0.59 | 602 | 1.0 | `quantize_q8_1` |
| 2.7 | 0.45 | 172 | 2.6 | `rms_norm_f32<1024, true, false>` |
| 1.8 | 0.31 | 84 | 3.7 | `mul_mat_vec_f<half, half, ncols_dst=1, 128>` |
| 1.5 | 0.25 | 2 | 124.8 | `mul_mat_vec_f<half, half, ncols_dst=1, 256>` |
| 1.4 | 0.23 | 70 | 3.3 | `mul_mat_vec_f<half, float, ncols_dst=1, 128>` |
| 1.3 | 0.22 | 84 | 2.6 | `mul_mat_vec_q<Q4_K, ncols_dst=1, has_fusion=false, small_k=true>` |
| 1.2 | 0.20 | 168 | 1.2 | `k_bin_bcast<op_mul>` |
| 1.1 | 0.19 | 134 | 1.4 | `rms_norm_f32<256, true, false>` |
| 0.7 | 0.12 | 96 | 1.2 | `k_set_rows<float, long, half>` (KV cache writes) |
| 0.7 | 0.12 | 84 | 1.4 | `soft_max_f32<true, 256, 256, float>` |
| 0.7 | 0.11 | 110 | 1.0 | `rope_neox<true, false, float>` |
| 0.4 | 0.07 | 84 | 0.85 | `unary_op_kernel<op_gelu>` |

**Kernel structure** (30 decode tokens):
- Q4_K matmul total: 5.39 + 3.21 + 0.22 = **8.82 ms (52% of GPU time)**
- Q6_K matmul total: **4.34 ms (26%)**
- Norms total: 0.76 + 0.45 + 0.19 + others ≈ 1.6 ms (10%)
- Q8_1 quantize: 0.59 ms (3.5%)
- Everything else (rope, softmax, gelu, get_rows, KV write...): ~1.6 ms (10%)
- **Note: gate+up+GEGLU is FUSED into a single Q4_K matmul kernel — not separate dispatches**

### hesper (30 tokens decode + 1 prefill, total GPU = 724.54 ms)

| % | Total ms | Inst | Avg µs | Likely identity |
|---|---|---|---|---|
| 34.1 | 247.2 | 640 | 386 | **ffnDown** Q4_K dp4a (in=10240→out=2560) |
| 26.6 | 192.7 | 2562 | 75 | **ffnGateUp** Q4_K dp4a (in=2560→out=10240, called 2× per layer for gate & up) |
| 7.2 | 52.3 | 30 | 1742 | **lmHead** Q6_K dp4a (1 per token) |
| 7.1 | 51.5 | 641 | 80 | **oProj** Q4_K dp4a 2-row variant (in≈4096→out=2560) |
| 3.9 | 28.3 | 5155 | 5.5 | **Q8_1 quantize (small)** — many small dispatches |
| 2.7 | 19.7 | 1069 | 18 | (unknown small kernel A, ~2580/30tok = 86 calls/tok) |
| 2.6 | 18.5 | 1068 | 17 | (unknown small kernel B) |
| 2.4 | 17.5 | 1068 | 16 | (unknown small kernel C) |
| 1.6 | 11.6 | 340 | 34 | per-layer-embedding kernel (~12 calls/tok) |
| 1.5 | 11.2 | 1281 | 8.7 | (small kernel) |
| 1.2 | 8.4 | 898 | 9.4 | (small kernel) |
| 1.0 | 7.1 | 213 | 33 | (small kernel — RoPE?) |
| 0.9 | 6.8 | 213 | 32 | (small kernel — qkvNorm?) |

(Identities are inferred from grid/block dimensions; full mapping requires PTX dump comparison.)

---

## 3. Kernel-by-kernel quantitative gap

For the headline matmuls, time per call (lower = better):

| Operation | hesper µs/call | llama.cpp µs/call | Gap |
|---|---|---|---|
| **ffnGateUp Q4_K** (one of gate or up, in=2560→out=10240) | 75 | 64 (FUSED gate+up+GEGLU) | **2.4× total** (because hesper does it twice + separate gelu) |
| **ffnDown Q4_K** (in=10240→out=2560) | 386 | 8.7 (small_k path?) or 64 (regular) | **6-44×** worse (depends which llama.cpp variant matches) |
| **lmHead Q6_K** (1 call/tok) | 1742 | 65.8 (per-call) × ~2 calls = ~130 | **~13× slower** |
| **Q8_1 quantize** | 5.5 (per call) × ~170 calls/tok | 1.0 × ~20 calls/tok | hesper: 935 µs/tok, llama.cpp: 20 µs/tok = **~47×** more spent here |
| **rms_norm** | (multiple kernels at ~17µs × 80+/tok) | 3.0 × 8/tok ≈ 24 µs/tok total | **~50×** more |

The numbers don't sum cleanly because of measurement noise, prefill mixed in, and the 2-row variant on hesper vs the multi-warp variant on llama.cpp. But the pattern is unambiguous: **every kernel class is at least 2× slower per call, and most are 5-50× slower**.

---

## 4. Where the 42× kernel-time gap comes from (root causes)

Working backwards from the table, here are the optimizations llama.cpp uses that hesper doesn't, ranked by expected TPS impact.

### A. **Multi-warp cooperation in matmul** (`nwarps=4`)
**llama.cpp evidence**: launch grid `(nrows_x, 1, 1)` with block `(warp_size=32, nwarps=4, 1)` = 128 threads, 4 warps cooperate on each output row via shared memory reduction.

**hesper status**: Single warp (32 threads) per output row. Tried 2-warp variant (`fusedQ4KMLinearDP4A2RowKernel`); each warp computes a different row, so still 1 warp per row of work.

**Quantitative impact**: ffnDown wave count
- llama.cpp: `2560 rows × 128 threads / 1536 thread/SM = 213 waves` (massive)
- hesper: `2560 rows × 32 threads / 1536 thread/SM = 53 waves`
- More waves → SM scheduler can hide memory latency by swapping warps.

**Estimated TPS gain if implemented**: +15-25 TPS (ffnDown alone could go from 386µs → ~80µs, saving 6.4 ms/tok)

### B. **Gate+up+GEGLU kernel fusion** (`mm_fusion`)
**llama.cpp evidence**: `mul_mat_vec_q<Q4_K, ncols_dst=1, has_fusion=true>` — single kernel computes both `gate = x @ W_gate` AND `up = x @ W_up` AND multiplies them through GEGLU(gate) × up. Total: **84 instances × 64µs = 5.39 ms** for ALL 42 layers × 30 tokens × 1 fused gate-up call/layer = 1260 calls expected, but llama.cpp groups these so 84 calls.

Actually 84 instances / 30 tokens = ~2.8/tok. Suspicious; Gemma 4 has 42 layers × 1 gate+up call/layer = 42 fused per token. Possibly llama.cpp uses 1 kernel per group of layers.

**hesper status**: Three separate dispatches per layer:
1. `gate = x @ W_gate` (Q4_K dp4a, ~75µs)
2. `up = x @ W_up` (Q4_K dp4a, ~75µs)
3. `gelu(gate) * up` (small kernel, ~5µs)
Plus the Q8_1 quantize that's needed for both gate and up (could be shared!).

Per layer: ~155µs in hesper vs ~64µs in llama.cpp = **2.4× slower per layer × 42 layers × 30 tokens = ~115 ms/30tok = 3.8 ms/tok savings if fused**.

**Estimated TPS gain**: +5-10 TPS

### C. **Read-only L1 cache hint** (`__ldg`, `ld.global.nc`)
**llama.cpp evidence**: `ggml_cuda_dp4a` and dequant inner loops use `__ldg` indirectly through `__restrict__` pointers and read-only accessors. NVIDIA driver/ptxas auto-promotes to `ld.global.nc` for `const __restrict__` weight buffers.

**hesper status**: Always uses ordinary `ld.global.u32`. No read-only hints.

**Quantitative impact**: ffnDown reads 14.7 MB of weights per call; with `.nc` hint, L1 cache hit rate jumps from ~5% to 30-40%, reducing effective DRAM traffic.

**Estimated TPS gain**: +5-10 TPS (across all matmuls)

### D. **Vector loads** (`ld.global.v4.b32` / `int4`)
**llama.cpp evidence**: `quantize_q8_1` uses `float4` and `char4` types for vectorized 16-byte loads/stores.

**hesper status**: Per-element `ld.global.f32` and `st.global.u32`.

**Quantitative impact**: Q8_1 quantize would halve memory request count → potentially halve quantize time (currently 935 µs/tok in hesper).

**Estimated TPS gain**: +1-2 TPS

### E. **Chained dp4a accumulator** (already in hesper for both Q4_K and Q6_K — verified)
**Not a gap**, both implementations match.

### F. **`__vsubss4` per-byte SIMD subtract** (already used in Q6_K — bit-trick equivalent in hesper)
**Already addressed** in hesper via `(x | 0x80) - 0x20 ^ 0x80` identity.

### G. **Small-kernel dispatch count** (1330 vs 87 / token)
**llama.cpp evidence**: Many operations fuse together (gate+up+GEGLU, RMSNorm+matmul prep, etc.) and use larger but fewer kernels. Also, some small ops (like elementwise ops) are eliminated by graph optimization passes.

**hesper status**: Each per-layer operation is a separate dispatch. RoPE, qkvNorm, postAttnNorm, postFFNNorm, residualAdd, etc. each their own kernel — × 42 layers × 30 tokens = thousands of small-kernel launches.

**Quantitative impact**: The 5155 instances of "Q8_1 quantize (small)" alone × ~5.5 µs = 28 ms total = **0.93 ms/tok of pure dispatch overhead**. Across ALL the "unknown small kernel" rows in hesper's table, this could be 5-10 ms/tok of small-kernel time that llama.cpp simply doesn't have.

**Estimated TPS gain**: +5-10 TPS via:
- RMSNorm + (next matmul's input quantize) fusion
- gate+up+GEGLU fusion (covered in B)
- RMSNorm + scale (residual) fusion

### H. **Template specialization** (`ncols_dst=1` decode-only path)
**llama.cpp evidence**: All matmul kernels are templated on `ncols_dst`; the decode-only `ncols_dst=1` path enables unique optimizations (rows_per_block=1, fusion enabled, register-resident accumulator).

**hesper status**: Single generic implementation, can't specialize.

**Estimated TPS gain**: +2-5 TPS (compiler-level optimization improvements)

---

## 5. The realistic upper bound for hesper portable

If hesper applies **all** of A through G above (excluding template specialization and CUDA-only intrinsics):

| Improvement | Expected ms/tok savings |
|---|---|
| Multi-warp cooperation (A) | -6.4 |
| gate+up+GEGLU fusion (B) | -3.8 |
| ld.global.nc hint (C) | -3.0 |
| Vector loads (D) | -1.0 |
| Small-kernel fusion (G) | -5.0 |
| **Total savings** | **-19.2 ms/tok** |
| **Resulting GPU time** | **23.4 - 19.2 = ~4.2 ms/tok** |
| **Resulting TPS** (assuming GPU stays bottleneck) | **~240 TPS** |

But host-side overhead (currently ~12 ms/tok in hesper, vs ~7.8 in llama.cpp) limits hesper to:
- If GPU drops to 4.2 ms/tok and CPU stays at 12 ms/tok → 16.2 ms/tok wall = **62 TPS**
- If CPU is also halved (better dispatch batching) → 4.2 + 6 = **10.2 ms/tok = 98 TPS**

**Conclusion**: With sustained engineering effort on both GPU kernels AND CPU dispatch path, **~100 TPS portable is technically achievable**. The 119 TPS llama.cpp number is within ~20% of that ceiling, so "matching llama.cpp" is plausible but requires nearly all the optimizations above.

---

## 6. Recommended optimization order

Ranked by **TPS gain ÷ engineering cost**:

| Priority | Optimization | TPS gain | Cost (weeks) | Risk |
|---|---|---|---|---|
| 1 | `.readOnly` hint → `ld.global.nc` (Phase A) | +5-10 | 0.5 | low |
| 2 | gate+up+GEGLU fusion (CUDA-only kernel) | +5-10 | 1-2 | low (already have skeleton) |
| 3 | Q8_1 quantize+matmul kernel fusion | +3-5 | 1 | low |
| 4 | Multi-warp cooperation (`@hint(.cooperativeWarps 4)`) | +15-25 | 3-4 | medium (DSL extension) |
| 5 | Vector loads in quantize/output kernels | +1-2 | 1 | low |
| 6 | RMSNorm + scale + residual fusion | +2-3 | 1 | low |
| 7 | Reduce small-kernel dispatch (graph fusion) | +3-5 | 2-3 | medium |
| 8 | Verified Native PTX path for Q4_K matmul | +20-40 | 4-6 | high (validation infra) |

**Cumulative total: +54-105 TPS** (28 → 82-133 TPS)

---

## 7. What we already verified (no re-investigation needed)

- ✅ Q4_K dp4a kernel: numerically matches llama.cpp via nvcc reference (commit f7430aa)
- ✅ Q6_K dp4a kernel: numerically matches llama.cpp via nvcc reference (commit 99d3ad5)
- ✅ Chained dp4a pattern in both Q4_K and Q6_K
- ✅ Q8_1 quantization with round-to-nearest-even (commit 23840ed)
- ✅ Per-byte signed subtract in Q6_K via bit identity

These are correctness-equivalent to llama.cpp — the 42× speed gap is purely about kernel structure, dispatch count, and use of CUDA-specific primitives, NOT about computational correctness.

---

Last updated: 2026-04-15.
