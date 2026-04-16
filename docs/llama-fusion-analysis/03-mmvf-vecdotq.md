---
title: "03 — mmvf.cu + vecdotq.cuh: f16 Mat-Vec と Q4_K/Q6_K dp4a 内積"
date: 2026-04-16
source:
  - llama.cpp/ggml/src/ggml-cuda/mmvf.cu (862行)
  - llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh (1,269行)
---

# mmvf.cu + vecdotq.cuh

## Part A: mmvf.cu — f16/f32 Matrix-Vector Multiply

### テンプレートシグネチャ (mmvf.cu:7–13)

```cuda
template <typename T, typename type_acc, int ncols_dst, int block_size,
          bool has_fusion = false, bool is_multi_token_id = false>
static __global__ void mul_mat_vec_f(
    const T * __restrict__ x, const float * __restrict__ y,
    const int32_t * __restrict__ ids,
    const ggml_cuda_mm_fusion_args_device fusion,
    float * __restrict__ dst, ...)
```

- **mmvq.cu と同じ `fusion_args_device` を消費** → 同じ epilogue (bias/gate/GLU)
- `has_fusion` は `ncols_dst == 1` のみ有効 (mmvf.cu:386, 396)

### 共有メモリ (mmvf.cu:95–100)

```cuda
extern __shared__ char data_mmv[];
float * buf_iw = (float *) data_mmv;
float * buf_iw_gate = nullptr;
if constexpr (has_fusion) {
    buf_iw_gate = (float *) (data_mmv + warp_size*sizeof(float));
}
```

warp-reduction buffer のみ。gate 用に追加バッファ。

### Block size 自動選択 (mmvf.cu:426–497)

```cuda
int64_t block_size_best = warp_size;  // 32
for (block_size = 2*warp_size; block_size <= 256; block_size += warp_size) {
    niter = (ncols + 2*block_size - 1) / (2*block_size);
    if (niter < niter_best) block_size_best = block_size;
}
```

サポート: 32, 64, 96, 128, 160, 192, 224, 256。
ncols (K 次元) に対してスレッドあたり反復回数を最小化。

### hesper との関係

hesper の F16 kernel (`perLayerInputPre`) は block-coop 型。
llama.cpp mmvf は per-thread ループで register accumulate → ILP 重視。

---

## Part B: vecdotq.cuh — Q4_K / Q6_K 内積カーネル

**hesper の per-kernel 速度差の鍵はここにある。**

### Q4_K: `vec_dot_q4_K_q8_1_impl_vmmq` (vecdotq.cuh:502–524)

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

        // ★ DP4A chaining: 2回の dp4a を直列に繋げ ILP を稼ぐ
        const int dot1 = ggml_cuda_dp4a(v1i, u[2*i+1],
                         ggml_cuda_dp4a(v0i, u[2*i+0], 0));

        // ★ 0x01010101 mask: 4要素の合計を 1回の DP4A で計算
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2*i+1],
                         ggml_cuda_dp4a(0x01010101, u[2*i+0], 0));

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }

    const float2 dm4f = __half22float2(dm4);  // ★ 遅延 f16→f32 変換
    return dm4f.x*sumf_d - dm4f.y*sumf_m;
}
```

**DP4A 使用: 2×2 = 4回/iteration × 2 iteration = 8 DP4A/block-pair**

### Q4_K ラッパー (vecdotq.cuh:816–860)

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

    // スケール展開: 条件分岐 + bitwise
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

    // q4 data: int キャスト読み (→ ld.global.nc を期待)
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

### Q6_K: `vec_dot_q6_K_q8_1_impl_mmvq` (vecdotq.cuh:621–641)

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

        // ★ __vsubss4: SIMD 4-element signed saturation subtract
        // 1命令で 4値同時に -32 オフセット
        const int vi = __vsubss4((vil | vih), 0x20202020);

        sumf += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * sc);
    }

    return d*sumf;
}
```

**DP4A 使用: 1回/iteration × 3 iteration = 3 DP4A/block-pair**

### Q6_K ラッパー (vecdotq.cuh:908–932)

```cuda
// 高 2 ビット展開
const int vh = get_int_b2(bq6_K->qh, ...) >> vh_shift;

// 低 4 ビット
const int vl = get_int_b2(bq6_K->ql, iqs);
```

## 3. Warp Reduction (共通)

`mmvf.cu:303–334` / `mmvq.cu` 内:

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

`warp_reduce_sum` (common.cuh):
```cuda
template <int warp_size>
__device__ float warp_reduce_sum(float v) {
    for (int offset = warp_size/2; offset > 0; offset /= 2)
        v += __shfl_xor_sync(0xffffffff, v, offset);
    return v;
}
```

## 4. メモリアクセスパターン

| 項目 | 実装 | 効果 |
|---|---|---|
| Weight reads | `(const int *)bq4_K->qs` — int cast | ld.global.nc (const-cache) |
| Q8_1 input | `(const int *)bq8i->qs` | 同上 |
| Scale | レジスタ上の bitwise unpack | ALU のみ、memory miss 回避 |
| dm (delta-min) | `__half22float2(bq4_K->dm)` | 1 命令 f16→f32 |

**Software pipelining / double-buffering**: **使用なし** (vecdotq.cuh 内)。
MMQ kernel (`mmq.cuh`) では別途 pipeline 使用あり。

## 5. ncu 計測比較

### hesper wO (Q4_K 2-row) vs llama.cpp Q4_K

| Metric | hesper | llama.cpp | 差 |
|---|---:|---:|---|
| Duration | 95 µs | 7–36 µs | 3–14× |
| Regs/thread | 36 | **66** | llama.cpp は ILP 重視 |
| Occupancy | 61% | 43–54% | hesper のほうが高い → warp-parallel 戦略 |
| DRAM BW util | 31% | 42–83% | llama.cpp は memory BW を活用 |
| Waves/SM | 0.89 | — | hesper は tail-wave 問題 |
| L1 Hit | 94% | 87–89% | hesper は L2 cache 恩恵 |

**根本的な差**: llama.cpp は**高レジスタ × 低 occupancy × ILP 最大化**戦略。
hesper は **低レジスタ × 高 occupancy × warp 並列** 戦略。
小 matrix (2560×2560 wO) では llama.cpp 戦略が勝つ。

## 6. hesper カーネルが「おそらく足りないもの」5 点

1. **DP4A chaining**: `dp4a(a,b, dp4a(c,d, 0))` — hesper の PTX emitter がこの
   パターンを生成しているか要確認 (`Hesper/Backend/CUDA/ShaderM.lean` の dp4a lowering)

2. **`0x01010101` SIMD row-sum**: Q4_K の min-bias 項を 1 DP4A で sum。
   hesper は `for (j in 0..3) minBias += u8[j]` のような展開かもしれない

3. **`__vsubss4` (Q6_K -32 offset)**: SIMD 1 命令で 4 値に対して saturating subtract。
   hesper の ShaderM に VALU の `vsub` が expose されているか要確認

4. **Half2 遅延変換 (`__half22float2`)**: scale を half のまま保持して最後に f32 化。
   hesper は early conversion で register を浪費している可能性あり

5. **Const-cache hint (`__ldg`/`ld.global.nc`)**: weight reads を const-cache に
   載せる。hesper は Step 55 (`BufferHint.readOnly`) で一部対応済みだが、
   Q4_K 内積の per-block weight reads に適用されているか要確認
   → [`Hesper/Layers/Linear.lean`](../../../Hesper/Layers/Linear.lean) の dp4a kernel 参照
