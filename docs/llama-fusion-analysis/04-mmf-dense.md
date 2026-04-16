---
title: "04 — mmf.cuh: Dense GEMM (Prefill / Batch)"
date: 2026-04-16
source: llama.cpp/ggml/src/ggml-cuda/mmf.cuh (908行)
---

# mmf.cuh: Dense GEMM (Prefill / Batch)

## 1. 概要

`mmf.cuh` は **行列-行列 f16/f32 乗算** — prefill / バッチ推論で使用。
**decode (ncols_dst=1) では使われない**。

| 特徴 | mmvq / mmvf (mat-vec) | mmf (mat-mat) |
|---|---|---|
| 出力列数 | 1–8 (decode) | 1–16 (prefill/batch) |
| テンソルコア | 不使用 | **WMMA 使用** |
| Epilogue | bias/gate/GLU fusion あり | **融合なし** (純 GEMM) |
| MoE routing | ids あり | has_ids テンプレート |

## 2. テンプレートシグネチャ (mmf.cuh:48–50)

```cpp
template <typename T, int rows_per_block, int cols_per_block,
          int nwarps, bool has_ids>
static __global__ void mul_mat_f(...)
```

- `T`: `half2`, `nv_bfloat162`, `float`
- `rows_per_block`: 32 (Turing+) or 64 (CDNA)
- `cols_per_block`: 1–16 (dispatch 時に決定)
- `nwarps`: 1–8 (auto-tuned, K 次元分割)
- `has_ids`: MoE expert routing

## 3. テンソルコア (mmf.cuh:57–80)

```cpp
// Turing+ (CC >= 75):
using tile_A = tile<16, 8, T>;      // m16 × k8
using tile_B = tile<16, 8, T>;      // k16 × n8 (transposed)
using tile_C = tile<16, 16, float>; // m16 × n16 accum

// Volta (CC 70):
using tile_A = tile<32, 4, T>;      // m32 × k4
using tile_B = tile<8,  4, T>;      // k8 × n4
using tile_C = tile<32, 8, float>;  // m32 × n8

// Generic fallback (CC < 70):
using tile_A = tile<16, 8, T>;
using tile_B = tile<8,  8, T>;
using tile_C = tile<16, 8, float>;
```

Turing+ で **m16n8k16** 相当の `mma()` 呼出 (mmf.cuh:221, 437, 491)。

## 4. Fusion **非対応**

**重要**: mmf.cuh は `ggml_cuda_mm_fusion_args_device` を**消費しない**。

入力シグネチャ (mmf.cuh:50–55):
```cpp
const T * __restrict__ x,
const float * __restrict__ y,
const int32_t * __restrict__ ids,
float * __restrict__ dst,      // ← fusion struct なし
...
```

出力 (mmf.cuh:270–274):
```cuda
// 純 f32 直接書き込み、epilogue なし
dst[j*stride_col_dst + row0 + i0*warp_size + threadIdx.x] = sum[i0];
```

**理由推測**: prefill は decode に比べ kernel 数/tok が少なく
(matmul が ncols 分を 1 call で処理)、fusion の ROI が低いため。

## 5. K 次元 multi-warp reduction (mmf.cuh:228–284)

```cuda
// 各 warp が K 次元の一部を担当
// 結果を shared memory buf_iw に書き込み
buf_iw[threadIdx.x + warp_size * (itA * tb_C::I + itB)
       + (threadIdx.x/tb_C::J) * warp_size
       * (tb_C::I * nwarps_dst_rows - 1)] = ...;

__syncthreads();

// warp 0 が全 warp 分を合計
if (threadIdx.x < rows_per_block) {
    for (int iw = 0; iw < nwarps; ++iw) {
        sum[i0] += buf_iw[...];
    }
}
```

## 6. MoE Expert Routing (mmf.cuh:299–320, 404–410)

`has_ids=true` 時、`ids_src_compact` / `ids_dst_compact` / `expert_bounds_dev`
で入出力行を expert ごとに remapping。これは **memory-write phase** に統合
されており、別途 gather/scatter kernel 不要。

## 7. hesper との関係

| 項目 | llama.cpp mmf | hesper |
|---|---|---|
| prefill GEMM | WMMA テンソルコア | Task #19 (pending): WMMA matmul wire 待ち |
| decode GEMM | mmvq/mmvf を使用 | 本ファイルは無関係 |
| MoE | has_ids で routing | Gemma 4 は non-MoE → 不要 |
| Fusion | なし | — |

**結論**: mmf.cuh は **decode 最適化には直接影響しない**。prefill を高速化する
場合に再訪。hesper の既存 prefill kernel (`Hesper/Layers/Linear.lean` の batched
forward) との比較は別タスク。
