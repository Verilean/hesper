---
title: "01 — ggml-cuda.cu graph-level fusion"
date: 2026-04-16
source: llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu (5,316 lines)
---

# ggml-cuda.cu graph-level fusion

## 1. Entry point

**`ggml_cuda_graph_evaluate_and_capture()`** at `ggml-cuda.cu:3540`.

It walks every node of the ggml cgraph with `for (int i = 0; i < cgraph->n_nodes; i++)`
(`ggml-cuda.cu:3638`). At each node it tries the fusion patterns
**in order**; the first one that matches dispatches a fused kernel and
advances `i` past the absorbed ops.

Kill switch at `ggml-cuda.cu:3687`:
```c
static bool disable_fusion = (getenv("GGML_CUDA_DISABLE_FUSION") != nullptr);
```

## 2. Fusability predicate

### Core: `ggml_can_fuse_ext()` (`ggml-impl.h:663–690`)

```c
static inline bool ggml_can_fuse_ext(
    const struct ggml_cgraph * cgraph,
    const int * node_idxs,
    const enum ggml_op * ops,
    int num_ops)
{
    for (int i = 0; i < num_ops; ++i) {
        // 1. Node exists in graph
        if (node_idxs[i] >= cgraph->n_nodes) return false;

        struct ggml_tensor * node = cgraph->nodes[node_idxs[i]];

        // 2. Op type matches expected pattern
        if (node->op != ops[i]) return false;

        // 3. COMPUTE flag set (skip dead code)
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) return false;

        // 4. Intermediate tensor must have exactly one consumer
        //    (final op may have multiple consumers)
        if (i < num_ops - 1 &&
            !ggml_node_has_n_uses(cgraph, node_idxs[i], 1))
            return false;

        // 5. Data dependency: sequential ops must chain
        if (i > 0) {
            struct ggml_tensor * prev = cgraph->nodes[node_idxs[i - 1]];
            if (node->src[0] != prev && node->src[1] != prev)
                return false;

            // 6. Same-shape requirement
            if (!ggml_are_same_shape(node, prev))
                return false;
        }
    }
    return true;
}
```

### Memory-overlap check (`ggml-cuda.cu:3477–3538`)

```c
static bool ggml_cuda_check_fusion_memory_ranges(
    ggml_cgraph * cgraph, ...) {
    // For nrows > 1, the output must not overwrite an intermediate input
    // before it is read.  nrows == 1 is always safe (vectorised writes
    // don't race reads within a wavefront).
}
```

### Correspondence with hesper Circuit DSL

| llama.cpp predicate | hesper equivalent |
|---|---|
| single-consumer | `protectedIds` marks externals + final outputs; anything else is implicitly at-most-single-consumer for fusion |
| same-shape | `fusePointwise` checks `op.outShape == consumer.inShapes[slot]` |
| memory overlap | not yet implemented (IR is SSA; in-place hazards handled in lowering) |

## 3. Pattern catalogue

### Pattern A: `MUL_MAT + ADD` (bias epilogue)

**`ggml-cuda.cu:3933–3987`**

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

**hesper**: covered by `Prim.matmulQ4KWithEpilogue` + `fuseMatmulEpilogue`
(`Passes.lean:485–580`). Bias is expressed as
`ScalarExp.add (input 0) (input 1)`.

---

### Pattern B: `MUL_MAT + ADD + MUL_MAT + ADD + GLU` (FFN gate+up fused)

**`ggml-cuda.cu:3810–3886`** — highest-ROI pattern.

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

The gate matmul is computed **inside the main matmul's inner loop**
(`mmvq.cu:494–497`):

```c
if (use_gate) {
    tmp_gate[j][i] += vec_dot_q_cuda(
        vgate, &y[j*stride_col_y + kby],
        kbx_offset + i*stride_row_x + kbx, kqs);
}
```

**hesper**: `ScalarExp` is pointwise-only; the gate matmul requires a
reduction, so the current `matmulQ4KWithEpilogue` body can't express it.
A **new Prim is required** — see [`05-hesper-dsl-plan.md`](05-hesper-dsl-plan.md)
for `Prim.matmulQ4KGateGLU`.

---

### Pattern C: `MUL_MAT + MUL_MAT + GLU` (no-bias variant)

**`ggml-cuda.cu:3887–3922`** — same as B without the ADD steps.

```c
else if (ggml_cuda_can_fuse(cgraph, i, { op, op, GGML_OP_GLU }, {})) {
    fused_node_count = 3;
}
```

**hesper**: covered by the same new Prim, with `xBias = none, gateBias = none`.

---

### Pattern D: `RMS_NORM + MUL (+ ADD)`

**`ggml-cuda.cu:3994–4004`**

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

**hesper**: Step 6's `fuseReduceEpilogue` (`Passes.lean:376–451`)
already fuses `reduceLastAxis → pointwise(mul)`. Adding `+ ADD` only
requires the body to become a 3-input `ScalarExp`.

---

### Pattern E: `ROPE + VIEW + SET_ROWS`

**`ggml-cuda.cu:3762–3769`**

```c
if (ggml_cuda_can_fuse(cgraph, i,
    { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
    ggml_cuda_op_rope_fused(*cuda_ctx, rope, set_rows);
    i += 2;
}
```

**hesper**: previously handled by hand-coded fused RoPE-K+KVwrite kernel
in `Gemma4.lean`. As of commit `b515e13` this is **also expressible via
`Prim.scatter`** with a `.indexed` gather and dynamic `addrExpr` — see
[`08-scatter-impl-notes.md`](08-scatter-impl-notes.md).

---

### Pattern F: Chained `ADD` (up to 8)

**`ggml-cuda.cu:3771–3802`**

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

**hesper**: `fusePointwise` can be extended from binary-chain to n-ary.
Currently the pass collapses pairs of binary adds; it doesn't group them
into a single n-input `Prim.scatter`.

---

### Pattern G: `SSM_CONV + SILU` — not applicable to Gemma 4.

### Pattern H: `UNARY(SILU/SIGMOID) + MUL`

**`ggml-cuda.cu:4012–4018`**

```c
if (ggml_cuda_can_fuse(cgraph, i,
    { GGML_OP_UNARY, GGML_OP_MUL },
    { GGML_UNARY_OP_SILU })) {
    ggml_cuda_op_unary_mul(*cuda_ctx, node, cgraph->nodes[i+1]);
    i++;
}
```

**hesper**: `fusePointwise` already handles it: the composed body
`ScalarExp.mul (ScalarExp.silu (input 0)) (input 1)` is a single
Map scatter after fusion.

---

### Pattern I: TopK-MOE — not applicable to Gemma 4.

## 4. CUDA Graph capture

**Entry**: `ggml_backend_cuda_graph_compute()` at `ggml-cuda.cu:4105–4162`.

```
1. Compute graph_key = hash(cgraph) and look up in cache
2. First call: execute directly (warmup, no capture)
3. Second call: properties stable → cudaStreamBeginCapture
4. Run the fusion dispatch loop — each kernel gets recorded into the graph
5. cudaStreamEndCapture → cudaGraphInstantiate
6. Subsequent tokens: cudaGraphLaunch dispatches the whole graph in one call
```

**Gate**: compute capability < 80 (pre-Ampere) disables Graphs
(`ggml-cuda.cu:4089`).

**Impact on hesper**: Investigation C measured the gap histogram and
found the ~10 ms/tok GPU-idle time sits in a single ~4.6 ms
`cuCtxSynchronize` per token (end-of-token logits readback — unavoidable
for autoregressive decode). Realistic CUDA Graphs gain is **< +1 TPS**,
so we skip it.

## 5. Environment variables

| Variable | Purpose | Source |
|---|---|---|
| `GGML_CUDA_DISABLE_FUSION` | Disable all fusion patterns | `ggml-cuda.cu:3687` |
| `GGML_CUDA_GRAPH_OPT` | Enable graph-level optimisations | `ggml-cuda.cu:4201` |
