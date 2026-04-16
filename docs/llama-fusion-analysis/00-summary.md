---
title: "00 — 統合サマリー: llama.cpp CUDA fusion vs hesper Circuit DSL"
date: 2026-04-16
---

# 統合サマリー: llama.cpp CUDA fusion vs hesper Circuit DSL

## 1. llama.cpp CUDA fusion の 2 層構造

### Layer 1: Graph-level pattern matcher

- **場所**: [`llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:3540–4024`](../../../llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu)
- ggml op DAG を走査し、連続する op パターンを認識して 1 kernel に畳む
- **8 種類** のパターン（詳細 → [`01-graph-fusion.md`](01-graph-fusion.md)）
- 制約: 中間テンソルは single-consumer、shape 一致、memory overlap 無し
- CUDA Graphs (`cudaStreamBeginCapture`) はこの上に乗り、dispatch overhead も除去

### Layer 2: Kernel-level template specialization

- **場所**: [`llama.cpp/ggml/src/ggml-cuda/mmvq.cu:389–582`](../../../llama.cpp/ggml/src/ggml-cuda/mmvq.cu)
- テンプレート `<ggml_type, ncols_dst, has_fusion, small_k>` で 4 段階 dispatch
- `ggml_cuda_mm_fusion_args_device` 構造体を kernel 引数で渡し epilogue を制御
- **epilogue は純 pointwise のみ**（reduce 無し）— bias / gate-matmul / GLU activation
- **f32 出力固定**

## 2. 認識される 8 種の fusion パターン

| # | パターン | 代表ソース行 | op数→1 | Gemma4 対象 | hesper対応 |
|---|---|---|---:|:---:|---|
| A | `MUL_MAT + ADD` (bias) | `ggml-cuda.cu:3933` | 2→1 | ✅ | ✅ 既存 (`fuseMatmulEpilogue`) |
| B | `MUL_MAT + ADD + MUL_MAT + ADD + GLU` | `ggml-cuda.cu:3810` | 5→1 | ✅ | 🟡 **新Prim必要** |
| C | `MUL_MAT + MUL_MAT + GLU` (no bias) | `ggml-cuda.cu:3887` | 3→1 | ✅ | 🟡 **新Prim必要** |
| D | `RMS_NORM + MUL + ADD` | `ggml-cuda.cu:3994` | 3→1 | ✅ | ✅ 既存 (`reduceWithEpilogue`) |
| E | `ROPE + VIEW + SET_ROWS` | `ggml-cuda.cu:3762` | 3→1 | ✅ | ✅ hand-coded済 |
| F | 連続 `ADD` (max 8) | `ggml-cuda.cu:3771` | N→1 | ✅ | 🟢 pass拡張で可 |
| G | `SSM_CONV + SILU` | `ggml-cuda.cu:4006` | 2→1 | ❌ | N/A |
| H | `UNARY(SILU) + MUL` | `ggml-cuda.cu:4012` | 2→1 | ✅ | ✅ 既存 (`fusePointwise`) |
| I | TopK-MOE (11 op) | `ggml-cuda.cu:3689` | 11→1 | ❌ | N/A |

Gemma 4 で**効くのは A/B/C/D/E/F/H の 7 種**。**B/C だけが新機能必須**。

## 3. Kernel-level fusion 構造体

### `ggml_cuda_mm_fusion_args_device`

**定義**: `common.cuh` / 使用: `mmvq.cu:1047`

```c
struct ggml_cuda_mm_fusion_args_device {
    const void * x_bias;       // up-path bias (f32)
    const void * gate;         // gate weight tensor (same quant type)
    const void * gate_bias;    // gate-path bias (f32)
    ggml_glu_op  glu_op;       // SWIGLU / GEGLU / REGLU / ...
};
```

### Epilogue 実行順序 (`mmvq.cu:556–582`)

```
1. matmul 完了 → warp reduce → 部分和
2. result += x_bias[j]              (if use_bias)
3. gate_value += gate_biases[j]     (if use_gate_bias)
4. gate_value = silu/gelu(gate_value)
5. result *= gate_value              (GLU)
6. dst[j] = result                   (f32 書き込み)
```

## 4. ncu 比較: hesper vs llama.cpp CUDA (Q4_K matmul)

| Metric | hesper wO (2-row) | hesper gate+up (4-row) | llama.cpp Q4_K (ncols=2) |
|---|---:|---:|---:|
| Duration | 95 µs | 148 µs | 7–36 µs |
| SM Throughput | 62% | 75% | 23–47% |
| DRAM Throughput | 31% | 40% | 42–83% |
| L1 Hit Rate | 94% | 74% | 87–89% |
| Occupancy | 61% | 89% | 43–54% |
| Regs/thread | 36 | 34 | **66** |
| No Eligible | 35% | — | 53–71% |
| Waves/SM | **0.89** | ~22 | — |

**重要な所見**: llama.cpp は **低オキュパンシー・高レジスタ** で ILP を稼ぐ戦略。
hesper は warp 数で並列化している。per-kernel speed は llama.cpp の 3–10×。

詳細 → [`03-mmvf-vecdotq.md`](03-mmvf-vecdotq.md)

## 5. per-kernel 微最適化 (vecdotq.cuh)

llama.cpp の Q4_K / Q6_K inner loop で使われている手法：

| 手法 | ソース | hesper 状況 |
|---|---|---|
| DP4A chaining (`dp4a(a,b, dp4a(c,d,0))`) | `vecdotq.cuh:514–518` | 未確認 |
| SIMD row-sum (`0x01010101` mask) | `vecdotq.cuh:516` | 未使用 |
| `__vsubss4` (Q6_K -32 offset) | `vecdotq.cuh:637` | ShaderM 未 expose |
| `__half22float2` 遅延変換 | `vecdotq.cuh:523` | 未確認 |

## 6. kernels/tok 削減ロードマップ

| Step | 作業 | 削減 | 累計 | 根拠 |
|---:|---|---:|---:|---|
| 0 | 現状 | — | 975 | 測定値 |
| 1 | bias ADD を `fuseMatmulEpilogue` wire | −42 | 933 | パターンA |
| 2 | `RMS_NORM+MUL+ADD` を `reduceWithEpilogue` 拡張 | −42 | 891 | パターンD |
| 3 | **matmulQ4KGateGLU Prim 追加** | −84 | 807 | パターンB/C |
| 4 | 連続 ADD n-ary fusion | −30 | 777 | パターンF |
| 5 | PLE 経路を Circuit 化 | −150 | 627 | Gemma4 特有 |
| 6 | residual-add epilogue 化 | −84 | 543 | パターンA 拡張 |

**Step 3 が single-commit で最大 ROI**: −84/tok、既存 IR と同系統の設計。

## 7. ソースファイル・パス一覧

### llama.cpp (調査対象)

| ファイル | 行数 | 概要 |
|---|---:|---|
| `llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu` | 5,316 | graph fusion dispatcher |
| `llama.cpp/ggml/src/ggml-cuda/mmvq.cu` | 1,150 | quantized mat-vec + epilogue |
| `llama.cpp/ggml/src/ggml-cuda/mmvf.cu` | 862 | f16/f32 mat-vec |
| `llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh` | 1,269 | Q4_K/Q6_K dp4a 内積 |
| `llama.cpp/ggml/src/ggml-cuda/mmf.cuh` | 908 | dense GEMM (prefill) |
| `llama.cpp/ggml/src/ggml-cuda/common.cuh` | 1,463 | 共通型定義・fusion struct |

### llama.cpp Vulkan (既調査)

| ファイル | 概要 |
|---|---|
| `llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q4_k.comp` | Vulkan Q4_K shader |
| `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp:14196+` | Vulkan fusion patterns |

### hesper Circuit DSL (変更対象)

| ファイル | 行数 | 概要 |
|---|---:|---|
| [`Hesper/Circuit/IR.lean`](../../../Hesper/Circuit/IR.lean) | 414 | Prim 定義・ScalarExp・TensorRef |
| [`Hesper/Circuit/Passes.lean`](../../../Hesper/Circuit/Passes.lean) | 782 | fusion passes (fusePointwise, fuseReduceEpilogue, fuseMatmulEpilogue) |
| [`Hesper/Circuit/Lowering.lean`](../../../Hesper/Circuit/Lowering.lean) | 712 | Prim → ShaderM/PTX lowering |
| [`Tests/Circuit/FuseMatmulEpilogueTest.lean`](../../../Tests/Circuit/FuseMatmulEpilogueTest.lean) | 102 | IR fusion unit test |
| [`Hesper/Layers/Linear.lean`](../../../Hesper/Layers/Linear.lean) | ~3,300 | 全 matmul kernel 実装 |
| [`Hesper/Models/Gemma4.lean`](../../../Hesper/Models/Gemma4.lean) | ~2,000 | モデル forward (Circuit wiring) |
