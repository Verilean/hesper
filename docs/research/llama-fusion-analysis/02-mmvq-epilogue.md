---
title: "02 — mmvq.cu: quantised mat-vec + epilogue template"
date: 2026-04-16
source: llama.cpp/ggml/src/ggml-cuda/mmvq.cu (1,150 lines)
---

# mmvq.cu: quantised mat-vec + epilogue

`mul_mat_vec_q` is llama.cpp's main matmul kernel during decode. Epilogue
fusion is enabled only when `ncols_dst = 1` (single-token decode).

## 1. Template signature

**`mmvq.cu:389–390`**

```cuda
template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false>
__launch_bounds__(calc_nwarps(type, ncols_dst, ...) * warp_size, 1)
static __global__ void mul_mat_vec_q(
    const void * __restrict__ vx,        // quantised weights (Q4_K/Q6_K etc.)
    const void * __restrict__ vy,        // Q8_1 pre-quantised input
    const int32_t * __restrict__ ids,    // MoE routing (optional)
    const ggml_cuda_mm_fusion_args_device fusion,  // epilogue payload
    float * __restrict__ dst,            // output (f32)
    ...)
```

| Parameter | Type | Meaning |
|---|---|---|
| `type` | `ggml_type` | Q4_0, Q4_K, Q6_K, IQ2_XXS etc. (20+ variants) |
| `ncols_dst` | `int` | Output batch width 1–8 (compile-time) |
| `has_fusion` | `bool` | True ⇒ epilogue code is enabled |
| `small_k` | `bool` | Optimisation flag for small K dimension |

## 2. The fusion struct

**`ggml_cuda_mm_fusion_args_device`** (declared in `common.cuh`, used at `mmvq.cu:1047`)

```c
struct ggml_cuda_mm_fusion_args_device {
    const void * x_bias      = nullptr;  // up-path bias (f32 vector)
    const void * gate         = nullptr;  // gate weight (same quant type)
    const void * gate_bias    = nullptr;  // gate-path bias (f32 vector)
    ggml_glu_op  glu_op;                 // SWIGLU / GEGLU / REGLU / SWIGLU_OAI
};
```

Host-side conversion of `ggml_tensor*` → device pointer (`mmvq.cu:1047–1070`):

```c
ggml_cuda_mm_fusion_args_device fusion_local{};
if (fusion) {
    if (fusion->x_bias)    fusion_local.x_bias    = fusion->x_bias->data;
    if (fusion->gate)      fusion_local.gate       = fusion->gate->data;
    if (fusion->gate_bias) fusion_local.gate_bias  = fusion->gate_bias->data;
    fusion_local.glu_op = fusion->glu_op;
}
```

## 3. Epilogue implementation

### Conditional flag setup (`mmvq.cu:435–443`)

```cuda
if constexpr (has_fusion) {
    use_gate      = fusion.gate      != nullptr;
    use_bias      = fusion.x_bias    != nullptr;
    use_gate_bias = fusion.gate_bias != nullptr && use_gate;
}
```

`if constexpr` means the dead-code branches vanish at compile time when
`has_fusion = false`.

### Parallel gate matmul (`mmvq.cu:494–497`)

The main and gate matmuls are computed **in the same inner loop**:

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

Both paths share the same Q8_1 input `y`; only the weight pointer
(`vx` vs `vgate`) differs.

### Warp reduction → epilogue (`mmvq.cu:533–582`)

```cuda
// After warp reduction, only thread 0 of each warp:
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

// Step 4: f32 store
dst[j*stride_col_dst + threadIdx.x] = result;               // mmvq.cu:582
```

### Activation primitives (in `unary.cuh`)

```cuda
// SiLU: x * σ(x)
__device__ float ggml_cuda_op_silu_single(float x) {
    return x / (1.0f + expf(-x));
}

// GELU (tanh approximation)
__device__ float ggml_cuda_op_gelu_single(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845f * x * (1.0f + 0.044715f * x * x)));
}
```

## 4. Three-stage dispatcher

### Level 1: `mul_mat_vec_q_switch_type` (`mmvq.cu:880–1019`)

Switch on `type_x` (Q4_0, Q4_K, Q6_K, …) and call Level 2.

### Level 2: `mul_mat_vec_q_switch_ncols_dst` (`mmvq.cu:717–879`)

Switch on `ncols_dst` (1–8). Fusion check:

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

### Level 3: `mul_mat_vec_q_switch_fusion` (`mmvq.cu:667–693`)

```c
if constexpr (has_fusion) {
    mul_mat_vec_q<type, 1, true, small_k><<<grid, block>>>(...)
} else {
    mul_mat_vec_q<type, ncols_dst, false, small_k><<<grid, block>>>(...)
}
```

## 5. Input format

- **src0 (weights)**: pre-quantised (`type_x` = Q4_K etc.)
- **src1 (input)**: f32 → **Q8_1 by a separate quantize kernel** (`mmvq.cu:1083–1090`)
  ```c
  quantize_row_q8_1_cuda(src1_d, ..., src1_q8_1.get(), ...);
  ```
- The Q8_1 quantize is a **distinct dispatch**, not part of the matmul kernel.

## 6. Output format

- **Always f32** (`mmvq.cu:582`).
- No post-quantise; downstream kernels do their own quantisation if needed.

## 7. Constraints

| Constraint | Detail |
|---|---|
| Fusion only for `ncols_dst == 1` | Batch / prefill paths skip fusion entirely |
| Gate weight must match main quant type | Mixed types are not supported |
| One activation at a time | Cannot chain SiLU + GELU etc. |
| Bias is up-path only (`x_bias`) | down-proj bias uses a different pattern |

## 8. Implications for hesper

1. **Gate matmul is computed inside the main kernel's inner loop.**
   `ScalarExp` (pointwise-only) cannot represent that — gate matmul has
   its own reduction. **A new `Prim.matmulQ4KGateGLU` is required.**

2. **Gate and main share the same Q8_1 input.** hesper's existing
   `fusedQ4KMLinearDP4AGeluSliceKernel` follows the same pattern; the
   internal structure can be lifted into IR with the right Prim.

3. **`if constexpr` dead-code elimination.** hesper compiles to PTX via
   JIT, so the same specialisation has zero cost — the lowering can
   resolve branches statically based on the `Prim` arguments.
