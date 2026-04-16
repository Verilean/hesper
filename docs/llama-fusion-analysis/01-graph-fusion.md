---
title: "01 — ggml-cuda.cu Graph-Level Fusion Logic"
date: 2026-04-16
source: llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu (5,316行)
---

# ggml-cuda.cu Graph-Level Fusion Logic

## 1. Fusion Entry Point

**`ggml_cuda_graph_evaluate_and_capture()`** (`ggml-cuda.cu:3540`)

ggml cgraph の全ノードを `for (int i = 0; i < cgraph->n_nodes; i++)` で走査
(`ggml-cuda.cu:3638`)。各ノードで以下の fusion パターンを**順次チェック**し、
最初にマッチしたパターンで dispatch → `i` を融合分だけスキップ。

disable ゲート (`ggml-cuda.cu:3687`):
```c
static bool disable_fusion = (getenv("GGML_CUDA_DISABLE_FUSION") != nullptr);
```

## 2. Fusability Predicate

### Core: `ggml_can_fuse_ext()` (`ggml-impl.h:663–690`)

```c
static inline bool ggml_can_fuse_ext(
    const struct ggml_cgraph * cgraph,
    const int * node_idxs,
    const enum ggml_op * ops,
    int num_ops)
{
    for (int i = 0; i < num_ops; ++i) {
        // 1. ノードが graph 内に存在
        if (node_idxs[i] >= cgraph->n_nodes) return false;

        struct ggml_tensor * node = cgraph->nodes[node_idxs[i]];

        // 2. op type がパターンと一致
        if (node->op != ops[i]) return false;

        // 3. COMPUTE flag (dead code 除外)
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) return false;

        // 4. 中間テンソルは single-consumer (最後の op は複数 consumer 可)
        if (i < num_ops - 1 &&
            !ggml_node_has_n_uses(cgraph, node_idxs[i], 1))
            return false;

        // 5. data dependency (sequential chain)
        if (i > 0) {
            struct ggml_tensor * prev = cgraph->nodes[node_idxs[i - 1]];
            if (node->src[0] != prev && node->src[1] != prev)
                return false;

            // 6. 同一 shape
            if (!ggml_are_same_shape(node, prev))
                return false;
        }
    }
    return true;
}
```

### Memory overlap check (`ggml-cuda.cu:3477–3538`)

```c
static bool ggml_cuda_check_fusion_memory_ranges(
    ggml_cgraph * cgraph, ...) {
    // nrows > 1 の場合: output が中間入力を上書きしないか検証
    // nrows == 1 は vectorized write で race-free → always safe
}
```

### hesper Circuit DSL との対応

| llama.cpp の predicate | hesper の実装 |
|---|---|
| single-consumer | `protectedIds` で「最終出力と外部入力のみ保護」→ 暗黙的に single-consumer |
| shape 一致 | `fusePointwise` で `op.outShape == consumer.inShapes[slot]` チェック |
| memory overlap | 未実装（IR は SSA なので in-place hazard は lowering 側で対処） |

## 3. 全 Fusion パターン詳細

### Pattern A: `MUL_MAT + ADD` (bias epilogue)

**ggml-cuda.cu:3933–3987**

```c
if (!ggml_can_fuse(cgraph, i, { op, bias_op })) continue;

ggml_cuda_mm_fusion_args_host fusion_data{};
fusion_data.x_bias = bias_tensor;

if (ggml_cuda_should_fuse_mul_mat_vec_f(mm_node)) {
    ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
    fused_node_count = 2;  // 2 ops → 1 kernel
    break;
}
```

**hesper 対応**: `Prim.matmulQ4KWithEpilogue` + `fuseMatmulEpilogue` pass
(`Passes.lean:485–580`)。bias を `ScalarExp.add (input 0) (input 1)` で表現可能。

---

### Pattern B: `MUL_MAT + ADD + MUL_MAT + ADD + GLU` (FFN gate+up fused)

**ggml-cuda.cu:3810–3886** — 最大 ROI パターン

```c
if (ggml_cuda_can_fuse(cgraph, i,
    { op, bias_op, op, bias_op, GGML_OP_GLU }, {})) {
    fusion_data.gate      = gate_n->src[0];
    fusion_data.x_bias    = up_bias_tensor;
    fusion_data.gate_bias = gate_bias_tensor;
    fusion_data.glu_op    = ggml_get_glu_op(glu);

    ggml_cuda_mul_mat_vec_f(*cuda_ctx, ...);
    fused_node_count = 5;  // 5 ops → 1 kernel
    break;
}
```

Gate matmul は**メインの matmul inner loop の中で並列計算**される
(`mmvq.cu:494–497`):
```c
if (use_gate) {
    tmp_gate[j][i] += vec_dot_q_cuda(
        vgate, &y[j*stride_col_y + kby],
        kbx_offset + i*stride_row_x + kbx, kqs);
}
```

**hesper 対応**: `ScalarExp` は pointwise のみ → **gate matmul は reduce を含む**
ため `matmulQ4KWithEpilogue` の body では表現不能。**新 Prim が必要**。

→ 提案: `Prim.matmulQ4KGateGLU` ([05-hesper-dsl-plan.md](05-hesper-dsl-plan.md) 参照)

---

### Pattern C: `MUL_MAT + MUL_MAT + GLU` (no bias variant)

**ggml-cuda.cu:3887–3922** — Pattern B の bias 無しバージョン。

```c
else if (ggml_cuda_can_fuse(cgraph, i, { op, op, GGML_OP_GLU }, {})) {
    fused_node_count = 3;
}
```

**hesper 対応**: Pattern B の `xBias = none, gateBias = none` で包含。

---

### Pattern D: `RMS_NORM + MUL (+ ADD)`

**ggml-cuda.cu:3994–4004**

```c
if (ggml_cuda_can_fuse(cgraph, i,
    { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, {})) {
    ggml_cuda_op_rms_norm_fused_add(*cuda_ctx, node, ...);
    i += 2;
    continue;
}

if (ggml_cuda_can_fuse(cgraph, i,
    { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
    ggml_cuda_op_rms_norm_fused(*cuda_ctx, node, ...);
    i++;
    continue;
}
```

**hesper 対応**: Step 6 `fuseReduceEpilogue` (`Passes.lean:376–451`) で
`reduceLastAxis → pointwise(mul)` 融合済み。`+ ADD` 拡張は `ScalarExp` body を
2-input → 3-input にするだけ。

---

### Pattern E: `ROPE + VIEW + SET_ROWS`

**ggml-cuda.cu:3762–3769**

```c
if (ggml_cuda_can_fuse(cgraph, i,
    { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
    ggml_cuda_op_rope_fused(*cuda_ctx, rope, set_rows);
    i += 2;
}
```

**hesper 対応**: `Gemma4.lean` に hand-coded の fused RoPE-K+KVwrite kernel で
対応済み。将来的に `Prim.ropeWithKVWrite` として IR 化すれば汎用性向上。

---

### Pattern F: 連続 `ADD` (max 8)

**ggml-cuda.cu:3771–3802**

```c
for (; n_fuse <= 6; ++n_fuse) {
    if (!ggml_can_fuse(cgraph, i + n_fuse, ops + n_fuse, 2))
        break;
    if (cgraph->nodes[i + n_fuse] !=
        cgraph->nodes[i + n_fuse + 1]->src[0])
        break;  // chain dependency
}
if (n_fuse > 1) {
    ggml_cuda_op_fused_add(*cuda_ctx, &fused_add_node, n_fuse);
}
```

**hesper 対応**: `fusePointwise` が 2-input zip → N-input zip に拡張可能。
現在は binary → binary の連鎖でしか対応していない。

---

### Pattern G: `SSM_CONV + SILU` — Gemma4 非対象

### Pattern H: `UNARY(SILU/SIGMOID) + MUL`

**ggml-cuda.cu:4012–4018**

```c
if (ggml_cuda_can_fuse(cgraph, i,
    { GGML_OP_UNARY, GGML_OP_MUL },
    { GGML_UNARY_OP_SILU })) {
    ggml_cuda_op_unary_mul(*cuda_ctx, node, cgraph->nodes[i+1]);
    i++;
}
```

**hesper 対応**: `fusePointwise` で `ScalarExp.mul (ScalarExp.silu (input 0)) (input 1)`
として自動融合可能。

---

### Pattern I: TopK-MOE — Gemma4 非対象

## 4. CUDA Graph Capture

**Entry**: `ggml_backend_cuda_graph_compute()` (`ggml-cuda.cu:4105–4162`)

```
1. graph_key = hash(cgraph) で前回 capture と比較
2. 初回: 直接実行 (warmup), capture しない
3. 2回目: properties 安定 → cudaStreamBeginCapture
4. fusion dispatch loop 実行 → 全 kernel が graph に record
5. cudaStreamEndCapture → cudaGraphInstantiate
6. 以後: cudaGraphLaunch で 1-call dispatch
```

**Gating**: CC < 80 (pre-Ampere) は disable (`ggml-cuda.cu:4089`)。

**hesper への影響**: Investigation C で gap histogram を測定済み。
per-token 4.6 ms の end-of-token sync が dominant → **CUDA Graphs の実効利得
< +1 TPS**。見送り。

## 5. 環境変数

| 変数 | 用途 | 場所 |
|---|---|---|
| `GGML_CUDA_DISABLE_FUSION` | 全 fusion 無効化 | `ggml-cuda.cu:3687` |
| `GGML_CUDA_GRAPH_OPT` | graph 最適化有効化 | `ggml-cuda.cu:4201` |
