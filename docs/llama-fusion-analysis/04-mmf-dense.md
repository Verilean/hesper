---
title: "04 — mmf.cuh: dense GEMM (prefill / batch)"
date: 2026-04-16
source: llama.cpp/ggml/src/ggml-cuda/mmf.cuh (908 lines)
---

# mmf.cuh: dense GEMM (prefill / batch)

## 1. Overview

`mmf.cuh` is **matrix-matrix f16/f32 multiplication** — used during
prefill or batch inference, **not during single-token decode**.

| Feature | mmvq / mmvf (mat-vec) | mmf (mat-mat) |
|---|---|---|
| Output columns | 1–8 (decode) | 1–16 (prefill / batch) |
| Tensor cores | not used | **WMMA used** |
| Epilogue | bias / gate / GLU fusion | **no fusion** (pure GEMM) |
| MoE routing | ids supported | `has_ids` template parameter |

## 2. Template signature (`mmf.cuh:48–50`)

```cpp
template <typename T, int rows_per_block, int cols_per_block,
          int nwarps, bool has_ids>
static __global__ void mul_mat_f(...)
```

- `T`: `half2`, `nv_bfloat162`, `float`
- `rows_per_block`: 32 (Turing+) or 64 (CDNA)
- `cols_per_block`: 1–16 (chosen at dispatch)
- `nwarps`: 1–8 (auto-tuned, splits the K dimension)
- `has_ids`: MoE expert routing

## 3. Tensor cores (`mmf.cuh:57–80`)

```cpp
// Turing+ (CC >= 75):
using tile_A = tile<16, 8, T>;      // m16 × k8
using tile_B = tile<16, 8, T>;      // k16 × n8 (transposed)
using tile_C = tile<16, 16, float>; // m16 × n16 accumulator

// Volta (CC 70):
using tile_A = tile<32, 4, T>;      // m32 × k4
using tile_B = tile<8,  4, T>;      // k8 × n4
using tile_C = tile<32, 8, float>;  // m32 × n8

// Generic fallback (CC < 70):
using tile_A = tile<16, 8, T>;
using tile_B = tile<8,  8, T>;
using tile_C = tile<16, 8, float>;
```

Turing+ effectively uses **m16n8k16** `mma()` calls (`mmf.cuh:221, 437, 491`).

## 4. **No fusion**

Important: `mmf.cuh` **does not consume `ggml_cuda_mm_fusion_args_device`**.

Input signature (`mmf.cuh:50–55`):
```cpp
const T * __restrict__ x,
const float * __restrict__ y,
const int32_t * __restrict__ ids,
float * __restrict__ dst,      // ← no fusion struct
...
```

Output (`mmf.cuh:270–274`):
```cuda
// Pure f32 store, no epilogue
dst[j*stride_col_dst + row0 + i0*warp_size + threadIdx.x] = sum[i0];
```

Likely reason: prefill issues fewer kernels per token than decode (one
matmul covers many ncols), so the fusion ROI is too small to justify
specialisation.

## 5. K-dimension multi-warp reduction (`mmf.cuh:228–284`)

```cuda
// Each warp owns a slice of K; results land in shared memory.
buf_iw[threadIdx.x + warp_size * (itA * tb_C::I + itB)
       + (threadIdx.x/tb_C::J) * warp_size
       * (tb_C::I * nwarps_dst_rows - 1)] = ...;

__syncthreads();

// Warp 0 sums across all warps.
if (threadIdx.x < rows_per_block) {
    for (int iw = 0; iw < nwarps; ++iw) {
        sum[i0] += buf_iw[...];
    }
}
```

## 6. MoE expert routing (`mmf.cuh:299–320, 404–410`)

When `has_ids = true`, the kernel uses `ids_src_compact` /
`ids_dst_compact` / `expert_bounds_dev` to remap input/output rows per
expert. Routing is folded into the **memory-write phase** — no separate
gather/scatter kernel.

## 7. Relationship to hesper

| Item | llama.cpp mmf | hesper |
|---|---|---|
| Prefill GEMM | WMMA tensor cores | Task #19 (pending): wire WMMA |
| Decode GEMM | Uses mmvq / mmvf | This file is irrelevant for decode |
| MoE | `has_ids` routing | Gemma 4 is non-MoE → not needed |
| Fusion | None | — |

**Conclusion.** `mmf.cuh` is **not in the decode-optimisation
critical path**. Revisit when prefill becomes the bottleneck. Comparison
with hesper's existing prefill kernels (the batched `forward` path in
`Hesper/Layers/Linear.lean`) is a separate task.
