---
title: "00 — Summary: llama.cpp CUDA fusion vs hesper Circuit DSL"
date: 2026-04-16
---

# Summary: llama.cpp CUDA fusion vs hesper Circuit DSL

## 1. llama.cpp CUDA fusion is a two-layer system

### Layer 1: graph-level pattern matcher

- Location: [`llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:3540–4024`](../../../llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu)
- Walks the ggml op DAG and collapses recognised op sequences into one kernel
- **Eight patterns** total (see [`01-graph-fusion.md`](01-graph-fusion.md))
- Guards: intermediate tensor must be single-consumer, shapes must match, no memory overlap
- CUDA Graphs (`cudaStreamBeginCapture`) sit on top of this layer, eliminating launch overhead

### Layer 2: kernel-level template specialisation

- Location: [`llama.cpp/ggml/src/ggml-cuda/mmvq.cu:389–582`](../../../llama.cpp/ggml/src/ggml-cuda/mmvq.cu)
- Template `<ggml_type, ncols_dst, has_fusion, small_k>` drives a 4-stage dispatch
- A `ggml_cuda_mm_fusion_args_device` struct is passed as a kernel argument and controls the epilogue
- **Epilogues are strictly pointwise** (no cross-lane reductions): bias / gate-matmul / GLU activation
- **Output is always f32**

## 2. The eight recognised fusion patterns

| # | Pattern | Representative source | Ops → 1 | Gemma 4? | hesper coverage |
|---|---|---|---:|:---:|---|
| A | `MUL_MAT + ADD` (bias) | `ggml-cuda.cu:3933` | 2→1 | ✅ | ✅ existing (`fuseMatmulEpilogue`) |
| B | `MUL_MAT + ADD + MUL_MAT + ADD + GLU` | `ggml-cuda.cu:3810` | 5→1 | ✅ | 🟡 **new Prim needed** |
| C | `MUL_MAT + MUL_MAT + GLU` (no bias) | `ggml-cuda.cu:3887` | 3→1 | ✅ | 🟡 **new Prim needed** |
| D | `RMS_NORM + MUL + ADD` | `ggml-cuda.cu:3994` | 3→1 | ✅ | ✅ existing (`reduceWithEpilogue`) |
| E | `ROPE + VIEW + SET_ROWS` | `ggml-cuda.cu:3762` | 3→1 | ✅ | ✅ hand-coded / now also DSL |
| F | Chained `ADD` (up to 8) | `ggml-cuda.cu:3771` | N→1 | ✅ | 🟢 pass extension |
| G | `SSM_CONV + SILU` | `ggml-cuda.cu:4006` | 2→1 | ❌ | N/A |
| H | `UNARY(SILU) + MUL` | `ggml-cuda.cu:4012` | 2→1 | ✅ | ✅ existing (`fusePointwise`) |
| I | TopK-MOE (11 ops) | `ggml-cuda.cu:3689` | 11→1 | ❌ | N/A |

Of these, **A/B/C/D/E/F/H (seven)** apply to Gemma 4. **B/C are the only
patterns that need truly new machinery** at the IR level.

## 3. The kernel-level fusion struct

### `ggml_cuda_mm_fusion_args_device`

Declared in `common.cuh`; used at `mmvq.cu:1047`:

```c
struct ggml_cuda_mm_fusion_args_device {
    const void * x_bias;       // up-path bias (f32)
    const void * gate;         // gate weight tensor (same quant type)
    const void * gate_bias;    // gate-path bias (f32)
    ggml_glu_op  glu_op;       // SWIGLU / GEGLU / REGLU / ...
};
```

### Epilogue execution order (`mmvq.cu:556–582`)

```
1. matmul done → warp reduce → partial sum
2. result += x_bias[j]              (if use_bias)
3. gate_value += gate_biases[j]     (if use_gate_bias)
4. gate_value = silu/gelu(gate_value)
5. result *= gate_value              (GLU)
6. dst[j] = result                   (f32 write)
```

## 4. ncu comparison: hesper vs llama.cpp CUDA (Q4_K matmul)

| Metric | hesper wO (2-row) | hesper gate+up (4-row) | llama.cpp Q4_K (ncols=2) |
|---|---:|---:|---:|
| Duration | 95 µs | 148 µs | 7–36 µs |
| SM throughput | 62% | 75% | 23–47% |
| DRAM throughput | 31% | 40% | 42–83% |
| L1 hit rate | 94% | 74% | 87–89% |
| Occupancy | 61% | 89% | 43–54% |
| Regs/thread | 36 | 34 | **66** |
| No-eligible | 35% | — | 53–71% |
| Waves/SM | **0.89** | ~22 | — |

**Key observation.** llama.cpp runs with **low occupancy + high register
count**, exploiting ILP per warp. hesper runs with **high occupancy +
low register count**, relying on warp parallelism. Per-kernel speed
differs by 3–10×.

Details in [`03-mmvf-vecdotq.md`](03-mmvf-vecdotq.md).

## 5. Per-kernel micro-optimisations (`vecdotq.cuh`)

Techniques visible in llama.cpp's Q4_K / Q6_K inner loop:

| Technique | Source | hesper status |
|---|---|---|
| DP4A chaining (`dp4a(a,b, dp4a(c,d,0))`) | `vecdotq.cuh:514–518` | unverified |
| SIMD row-sum (`0x01010101` mask) | `vecdotq.cuh:516` | not used |
| `__vsubss4` (Q6_K -32 offset) | `vecdotq.cuh:637` | not exposed in ShaderM |
| Deferred `__half22float2` | `vecdotq.cuh:523` | unverified |

## 6. Kernels-per-token reduction roadmap

| Step | Work | Saves | Running total | Basis |
|---:|---|---:|---:|---|
| 0 | Starting point | — | 975 | measured |
| 1 | Wire bias `ADD` into `fuseMatmulEpilogue` | −42 | 933 | Pattern A |
| 2 | Extend `reduceWithEpilogue` to `RMS_NORM+MUL+ADD` | −42 | 891 | Pattern D |
| 3 | **Add `matmulQ4KGateGLU` Prim** | −84 | 807 | Patterns B/C |
| 4 | N-ary `ADD` chain fusion | −30 | 777 | Pattern F |
| 5 | Bring the PLE path into Circuit DSL | −150 | 627 | Gemma 4 specific |
| 6 | Wire residual-add as matmul epilogue | −84 | 543 | Pattern A extension |

**Step 3 is the single highest-ROI commit**: −84/tok in one coherent change.

## 7. Source file map

### llama.cpp (subject files)

| File | Lines | Role |
|---|---:|---|
| `llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu` | 5,316 | graph fusion dispatcher |
| `llama.cpp/ggml/src/ggml-cuda/mmvq.cu` | 1,150 | quantised mat-vec + epilogue |
| `llama.cpp/ggml/src/ggml-cuda/mmvf.cu` | 862 | f16/f32 mat-vec |
| `llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh` | 1,269 | Q4_K/Q6_K dp4a inner products |
| `llama.cpp/ggml/src/ggml-cuda/mmf.cuh` | 908 | dense GEMM (prefill) |
| `llama.cpp/ggml/src/ggml-cuda/common.cuh` | 1,463 | shared types, fusion struct |

### llama.cpp Vulkan (prior analysis)

| File | Role |
|---|---|
| `llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q4_k.comp` | Vulkan Q4_K shader |
| `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp:14196+` | Vulkan fusion patterns |

### hesper Circuit DSL (targets of change)

| File | Lines | Role |
|---|---:|---|
| [`Hesper/Circuit/IR.lean`](../../../Hesper/Circuit/IR.lean) | 414 | `Prim`, `ScalarExp`, `TensorRef` |
| [`Hesper/Circuit/Passes.lean`](../../../Hesper/Circuit/Passes.lean) | 782 | fusion passes (`fusePointwise`, `fuseReduceEpilogue`, `fuseMatmulEpilogue`) |
| [`Hesper/Circuit/Lowering.lean`](../../../Hesper/Circuit/Lowering.lean) | 712 | Prim → ShaderM/PTX |
| [`Tests/Circuit/FuseMatmulEpilogueTest.lean`](../../../Tests/Circuit/FuseMatmulEpilogueTest.lean) | 102 | IR fusion unit test |
| [`Hesper/Layers/Linear.lean`](../../../Hesper/Layers/Linear.lean) | ~3,300 | all matmul kernels |
| [`Hesper/Models/Gemma4.lean`](../../../Hesper/Models/Gemma4.lean) | ~2,000 | model forward (Circuit wiring) |
