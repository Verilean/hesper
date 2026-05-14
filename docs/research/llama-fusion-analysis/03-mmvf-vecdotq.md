---
title: "03 — mmvf.cu + vecdotq.cuh: f16 mat-vec and Q4_K/Q6_K dp4a"
date: 2026-04-16
source:
  - llama.cpp/ggml/src/ggml-cuda/mmvf.cu (862 lines)
  - llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh (1,269 lines)
---

# mmvf.cu + vecdotq.cuh

## Part A: mmvf.cu — f16/f32 matrix-vector multiply

### Template signature (`mmvf.cu:7–13`)

```cuda
template <typename T, typename type_acc, int ncols_dst, int block_size,
          bool has_fusion = false, bool is_multi_token_id = false>
static __global__ void mul_mat_vec_f(
    const T * __restrict__ x, const float * __restrict__ y,
    const int32_t * __restrict__ ids,
    const ggml_cuda_mm_fusion_args_device fusion,
    float * __restrict__ dst, ...)
```

- **Same `fusion_args_device` as `mmvq.cu`** ⇒ same epilogue (bias / gate / GLU)
- `has_fusion` only fires when `ncols_dst == 1` (`mmvf.cu:386, 396`)

### Shared memory (`mmvf.cu:95–100`)

```cuda
extern __shared__ char data_mmv[];
float * buf_iw = (float *) data_mmv;
float * buf_iw_gate = nullptr;
if constexpr (has_fusion) {
    buf_iw_gate = (float *) (data_mmv + warp_size*sizeof(float));
}
```

Just a warp-reduction buffer, plus a second one for the gate path when
fusion is on.

### Block-size autotuning (`mmvf.cu:426–497`)

```cuda
int64_t block_size_best = warp_size;  // 32
for (block_size = 2*warp_size; block_size <= 256; block_size += warp_size) {
    niter = (ncols + 2*block_size - 1) / (2*block_size);
    if (niter < niter_best) block_size_best = block_size;
}
```

Supported: 32, 64, 96, 128, 160, 192, 224, 256. Picked to minimise the
per-thread iteration count over the K dimension.

### Relationship to hesper

hesper's F16 kernel (`perLayerInputPre`) is block-cooperative.
llama.cpp's `mmvf` uses per-thread loops with register accumulation —
ILP-focused.

---

## Part B: vecdotq.cuh — Q4_K / Q6_K inner products

**This is where hesper's per-kernel speed gap originates.**

### Q4_K: `vec_dot_q4_K_q8_1_impl_vmmq` (`vecdotq.cuh:502–524`)

```cuda
static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_vmmq(
    const int * __restrict__ v,
    const int * __restrict__ u,
    const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m,
    const half2 & dm4,
    const float * __restrict__ d8)
{
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {           // QR4_K = 2
        const int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;

        // ★ DP4A chaining: two sequential dp4a calls keep the ALU pipeline busy
        const int dot1 = ggml_cuda_dp4a(v1i, u[2*i+1],
                         ggml_cuda_dp4a(v0i, u[2*i+0], 0));

        // ★ 0x01010101 mask: sums all 4 elements with one DP4A
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2*i+1],
                         ggml_cuda_dp4a(0x01010101, u[2*i+0], 0));

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }

    const float2 dm4f = __half22float2(dm4);  // ★ deferred f16 → f32
    return dm4f.x*sumf_d - dm4f.y*sumf_m;
}
```

**4 DP4A per iteration × 2 iterations = 8 DP4A per block-pair.**

### Q4_K wrapper (`vecdotq.cuh:816–860`)

```cuda
static __device__ __forceinline__ float vec_dot_q4_K_q8_1(
    const void * __restrict__ vbq,
    const block_q8_1 * __restrict__ bq8_1,
    const int & kbx, const int & iqs)
{
    const block_q4_K * bq4_K = (const block_q4_K *) vbq + kbx;

    int v[2];
    int u[2*QR4_K];   // u[4]
    float d8[QR4_K];

    // Scale unpacking: branchy bitwise
    const uint16_t * scales = (const uint16_t *)bq4_K->scales;
    uint16_t aux[2];
    const int j = bq8_offset/2;
    if (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f)
               | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f)
               | ((scales[j-0] & 0xc0c0) >> 2);
    }

    // q4 data: int-cast loads (expects ld.global.nc)
    const int * q4 = (const int *)(bq4_K->qs + ...);
    v[0] = q4[0];
    v[1] = q4[4];

    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int * q8 = (const int *)bq8i->qs + ...;
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);
}
```

### Q6_K: `vec_dot_q6_K_q8_1_impl_mmvq` (`vecdotq.cuh:621–641`)

```cuda
static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmvq(
    const int & vl, const int & vh,
    const int * __restrict__ u,
    const int8_t * __restrict__ scales,
    const float & d, const float * __restrict__ d8)
{
    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {    // QR6_K = 3
        const int sc = scales[4*i];
        const int vil = (vl >> (4*i)) & 0x0F0F0F0F;
        const int vih = ((vh >> (4*i)) << 4) & 0x30303030;

        // ★ __vsubss4: SIMD 4-element signed-saturation subtract,
        //   one instruction subtracts -32 from four bytes at once.
        const int vi = __vsubss4((vil | vih), 0x20202020);

        sumf += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * sc);
    }

    return d*sumf;
}
```

**1 DP4A per iteration × 3 iterations = 3 DP4A per block-pair.**

### Q6_K wrapper (`vecdotq.cuh:908–932`)

```cuda
// High 2 bits
const int vh = get_int_b2(bq6_K->qh, ...) >> vh_shift;

// Low 4 bits
const int vl = get_int_b2(bq6_K->ql, iqs);
```

## 3. Warp reduction (shared)

`mmvf.cu:303–334` and inside `mmvq.cu`:

```cuda
sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);

if (block_size > warp_size) {
    buf_iw[tid/warp_size] = sumf[j];
    __syncthreads();
    if (tid < warp_size) {
        sumf[j] = buf_iw[tid];
        sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);
    }
}
```

`warp_reduce_sum` (in `common.cuh`):

```cuda
template <int warp_size>
__device__ float warp_reduce_sum(float v) {
    for (int offset = warp_size/2; offset > 0; offset /= 2)
        v += __shfl_xor_sync(0xffffffff, v, offset);
    return v;
}
```

## 4. Memory access patterns

| Item | Implementation | Effect |
|---|---|---|
| Weight reads | `(const int *)bq4_K->qs` — int-cast | ld.global.nc (const cache) |
| Q8_1 input | `(const int *)bq8i->qs` | same |
| Scale | bitwise unpack on registers | ALU only, no memory misses |
| dm (delta-min) | `__half22float2(bq4_K->dm)` | one f16→f32 instruction |

**Software pipelining / double buffering**: not used in `vecdotq.cuh`.
The MMQ kernel (`mmq.cuh`) has its own pipelining — out of scope here.

## 5. ncu comparison

### hesper wO (Q4_K 2-row) vs llama.cpp Q4_K

| Metric | hesper | llama.cpp | Delta |
|---|---:|---:|---|
| Duration | 95 µs | 7–36 µs | 3–14× |
| Regs/thread | 36 | **66** | llama.cpp: ILP-focused |
| Occupancy | 61% | 43–54% | hesper higher (warp-parallel) |
| DRAM BW util | 31% | 42–83% | llama.cpp leverages BW better |
| Waves/SM | 0.89 | — | hesper has tail-wave problem |
| L1 hit | 94% | 87–89% | hesper is L2-resident |

**Root difference.** llama.cpp pursues **high register count × low
occupancy × maximised ILP**; hesper pursues **low register count × high
occupancy × warp parallelism**. For small matrices (the 2560×2560 wO),
llama.cpp's strategy wins.

## 6. Five things hesper kernels probably miss

1. **DP4A chaining**: `dp4a(a,b, dp4a(c,d, 0))`. Need to confirm
   hesper's PTX emitter generates this pattern (check
   `Hesper/Backend/CUDA/ShaderM.lean`'s dp4a lowering).

2. **`0x01010101` SIMD row-sum**: one DP4A computes the sum-of-four
   used in Q4_K's `min*bias` term. hesper may be expanding it as
   `for (j in 0..3) minBias += u8[j]`.

3. **`__vsubss4` (Q6_K -32 offset)**: one SIMD instruction subtracts -32
   from four bytes simultaneously. Need to check whether `vsub` is
   exposed in hesper's ShaderM.

4. **Deferred `__half22float2`**: keep the scale as half until the very
   end. hesper may convert eagerly and waste registers.

5. **Const-cache hint (`__ldg` / `ld.global.nc`)**: weight reads should
   go through the const cache. hesper has Step 55 (`BufferHint.readOnly`)
   for this — verify it's applied to per-block weight reads in Q4_K
   inner products. See [`Hesper/Layers/Linear.lean`](../../../Hesper/Layers/Linear.lean).
