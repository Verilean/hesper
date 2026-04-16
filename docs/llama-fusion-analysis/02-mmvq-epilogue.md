---
title: "02 — mmvq.cu: Quantized Mat-Vec + Epilogue Template"
date: 2026-04-16
source: llama.cpp/ggml/src/ggml-cuda/mmvq.cu (1,150行)
---

# mmvq.cu: Quantized Mat-Vec + Epilogue

`mul_mat_vec_q` は llama.cpp の **decode 時メイン matmul kernel**。
1 トークン推論 (`ncols_dst=1`) 時のみ epilogue fusion が有効。

## 1. テンプレートシグネチャ

**mmvq.cu:389–390**

```cuda
template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false>
__launch_bounds__(calc_nwarps(type, ncols_dst, ...) * warp_size, 1)
static __global__ void mul_mat_vec_q(
    const void * __restrict__ vx,        // quantized weights (Q4_K/Q6_K等)
    const void * __restrict__ vy,        // Q8_1 pre-quantized input
    const int32_t * __restrict__ ids,    // MoE routing (optional)
    const ggml_cuda_mm_fusion_args_device fusion,  // epilogue payload
    float * __restrict__ dst,            // output (f32)
    ...)
```

| パラメータ | 型 | 意味 |
|---|---|---|
| `type` | `ggml_type` | Q4_0, Q4_K, Q6_K, IQ2_XXS 等 20+ |
| `ncols_dst` | `int` | 出力バッチ幅 1–8 (compile-time) |
| `has_fusion` | `bool` | true: epilogue code 有効化 |
| `small_k` | `bool` | K 次元が小さい場合の最適化 |

## 2. Fusion 構造体

**`ggml_cuda_mm_fusion_args_device`** (common.cuh, mmvq.cu:1047)

```c
struct ggml_cuda_mm_fusion_args_device {
    const void * x_bias      = nullptr;  // up-path bias (f32 vector)
    const void * gate         = nullptr;  // gate weight (同じ quant type)
    const void * gate_bias    = nullptr;  // gate-path bias (f32 vector)
    ggml_glu_op  glu_op;                 // SWIGLU / GEGLU / REGLU / SWIGLU_OAI
};
```

Host 側で `ggml_tensor*` → `void*` (device pointer) に変換:

```c
// mmvq.cu:1047-1070
ggml_cuda_mm_fusion_args_device fusion_local{};
if (fusion) {
    if (fusion->x_bias)    fusion_local.x_bias    = fusion->x_bias->data;
    if (fusion->gate)      fusion_local.gate       = fusion->gate->data;
    if (fusion->gate_bias) fusion_local.gate_bias  = fusion->gate_bias->data;
    fusion_local.glu_op = fusion->glu_op;
}
```

## 3. Epilogue 実装

### 条件フラグ展開 (mmvq.cu:435–443)

```cuda
if constexpr (has_fusion) {
    use_gate      = fusion.gate      != nullptr;
    use_bias      = fusion.x_bias    != nullptr;
    use_gate_bias = fusion.gate_bias != nullptr && use_gate;
}
```

`constexpr` → dead code elimination で fusion 無しビルドは epilogue が完全除去。

### Gate matmul の並列計算 (mmvq.cu:494–497)

**Inner loop 内で main matmul と gate matmul を同時計算**:

```cuda
// main path
tmp[j][i] += vec_dot_q_cuda(vx, &y[j*stride_col_y + kby],
                            kbx_offset + i*stride_row_x + kbx, kqs);

// gate path (same iteration)
if constexpr (has_fusion) {
    if (use_gate) {
        tmp_gate[j][i] += vec_dot_q_cuda(
            vgate, &y[j*stride_col_y + kby],
            kbx_offset + i*stride_row_x + kbx, kqs);
    }
}
```

**重要**: gate matmul は **同じ Q8_1 input** (`y`) を共有。重み (`vgate`) だけ異なる。

### Warp reduction → epilogue (mmvq.cu:533–582)

```cuda
// reduction 後、threadIdx.x == 0 のスレッドのみ:
float result = /* reduced sum */;

// Step 1: bias
if (use_bias) {
    result += x_biases[j];                    // mmvq.cu:557-558
}

// Step 2: gate bias
float gate_value = /* reduced gate sum */;
if (use_gate_bias) {
    gate_value += gate_biases[j];             // mmvq.cu:562-563
}

// Step 3: GLU activation
switch (active_glu) {
    case GGML_GLU_OP_SWIGLU:
        result *= ggml_cuda_op_silu_single(gate_value);      // mmvq.cu:566
        break;
    case GGML_GLU_OP_GEGLU:
        result *= ggml_cuda_op_gelu_single(gate_value);      // mmvq.cu:569
        break;
    case GGML_GLU_OP_SWIGLU_OAI:
        result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
        break;
    default:
        result = result * gate_value;                        // generic GLU
        break;
}

// Step 4: f32 書き込み
dst[j*stride_col_dst + threadIdx.x] = result;               // mmvq.cu:582
```

### 活性化関数 (unary.cuh)

```cuda
// SiLU: x * σ(x)
__device__ float ggml_cuda_op_silu_single(float x) {
    return x / (1.0f + expf(-x));
}

// GELU (tanh 近似)
__device__ float ggml_cuda_op_gelu_single(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845f * x * (1.0f + 0.044715f * x * x)));
}
```

## 4. Dispatch ロジック (3 段階)

### Level 1: `mul_mat_vec_q_switch_type` (mmvq.cu:880–1019)

`type_x` (Q4_0, Q4_K, Q6_K 等) を switch → Level 2 呼出。

### Level 2: `mul_mat_vec_q_switch_ncols_dst` (mmvq.cu:717–879)

`ncols_dst` (1–8) を switch。fusion チェック:

```c
// mmvq.cu:738
const bool has_fusion =
    fusion.gate != nullptr ||
    fusion.x_bias != nullptr ||
    fusion.gate_bias != nullptr;

if (ncols_dst == 1 && has_fusion) {
    // → Level 3 with has_fusion=true
} else {
    GGML_ASSERT(!has_fusion && "fusion only for ncols_dst=1");
}
```

### Level 3: `mul_mat_vec_q_switch_fusion` (mmvq.cu:667–693)

```c
if constexpr (has_fusion) {
    mul_mat_vec_q<type, 1, true, small_k><<<grid, block>>>(...)
} else {
    mul_mat_vec_q<type, ncols_dst, false, small_k><<<grid, block>>>(...)
}
```

## 5. 入力形式

- **src0 (weights)**: 量子化済み (`type_x` = Q4_K 等)
- **src1 (input)**: **f32 → Q8_1 に事前量子化** (mmvq.cu:1083–1090)
  ```c
  quantize_row_q8_1_cuda(src1_d, ..., src1_q8_1.get(), ...);
  ```
- Q8_1 quantize は **別 kernel** — matmul には含まれない

## 6. 出力形式

- **常に f32** (mmvq.cu:582)
- post-quantize 無し — 後続 kernel に委託

## 7. 制約

| 制約 | 詳細 |
|---|---|
| fusion は ncols_dst == 1 のみ | バッチ/prefill は fusion 無効 |
| gate weight は main weight と同じ quant type | 異なる型は不可 |
| 活性化は 1 種のみ同時適用 | SiLU + GELU 等の chain は不可 |
| bias は up-path のみ (x_bias) | down-proj の bias は別パターン |

## 8. hesper への含意

1. **gate matmul を inner loop 内で並列計算** する設計 → `ScalarExp` (pointwise-only)
   では表現不能 → **新 Prim `matmulQ4KGateGLU` が必要**
2. **同じ Q8_1 input を共有** → hesper の gate+up fused kernel
   (`fusedQ4KMLinearDP4AGeluSliceKernel`) は既に同パターン。
   内部構造を gate-matmul + epilogue に分解すれば IR 化可能
3. **`constexpr` による dead code elimination** → hesper は PTX JIT なので
   同等の特殊化はコスト 0 で可能（ShaderM → PTX lowering で分岐を静的に解決）
