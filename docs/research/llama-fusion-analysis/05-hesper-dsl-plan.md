---
title: "05 — hesper Circuit DSL extension plan"
date: 2026-04-16
---

# hesper Circuit DSL extension plan

Concrete changes to close the gap identified in the llama.cpp analysis.

> **Note** (2026-04-16): some of these items have since been implemented;
> the unification of write-side primitives is recorded in
> [`08-scatter-impl-notes.md`](08-scatter-impl-notes.md). The plan
> below is the original action list.

## Existing DSL architecture

```
[IR]      Prim → Op → TensorRef → Circuit state
[Passes]  fusePointwise / fuseReduceEpilogue / fuseMatmulEpilogue
[Lower]   Prim → ShaderM → PTX JIT → cuLaunchKernel
```

### Existing Prim catalogue (IR.lean:163–213)

| Prim | Purpose | Line |
|---|---|---|
| `pointwise` | N-input → 1-output element-wise | `IR.lean:163` |
| `matmulQ4K` | Q4_K mat-vec | `IR.lean:172` |
| `matmulQ4KWithEpilogue` | Q4_K mat-vec with pointwise tail | `IR.lean:176` |
| `reduceLastAxis` | sum / sumOfSquares | `IR.lean:191` |
| `reduceLastAxisWithEpilogue` | reduce + pointwise tail | `IR.lean:211` |

### Existing fusion passes (Passes.lean)

| Pass | Purpose | Line |
|---|---|---|
| `fusePointwise` | Compress pointwise chains | `Passes.lean:278` |
| `fuseReduceEpilogue` | Fuse reduce → pointwise | `Passes.lean:451` |
| `fuseMatmulEpilogue` | Fuse matmulQ4K → pointwise | `Passes.lean:575` |
| `mergeSameDispatch` | Merge identical dispatches | `Passes.lean:53` |

---

## Change 1: `Prim.matmulQ4KGateGLU` (new Prim)

### Motivation

Pattern B/C from llama.cpp: `MUL_MAT + [ADD +] MUL_MAT + [ADD +] GLU` →
1 kernel. Gemma 4's FFN gate+up pair matches this exactly. Today
hesper has `fusedQ4KMLinearDP4AGeluSliceKernel` hand-coded but never
exposed at the IR level, so Circuit DSL fusion passes can't reach it.

### Design

```lean
-- Hesper/Circuit/IR.lean
inductive GLUOp where
  | silu      -- result *= silu(gate)
  | gelu      -- result *= gelu(gate)
  | reglu     -- result *= gate
  | swigluOai -- swiglu OpenAI variant

inductive Prim where
  ...
  | matmulQ4KGateGLU
      (mainLayer : LinearLayer BufT CacheT)
      (gateLayer : LinearLayer BufT CacheT)
      (xBias     : Option TensorRef)
      (gateBias  : Option TensorRef)
      (gluOp     : GLUOp)
      : Prim
```

### Lowering

In [`Lowering.lean`](../../../Hesper/Circuit/Lowering.lean), emit the
existing `fusedQ4KMLinearDP4AGeluSliceKernel` shape with two parallel
accumulators — same pattern as `mmvq.cu:494–497`:

```
Prim.matmulQ4KGateGLU main gate xBias gateBias gluOp
  → emit a PTX kernel that computes `main · x` and `gate · x` in the
    same inner loop, applies optional biases on lane 0 after warp
    reduction, applies gluOp, and writes the f32 product.
```

### Estimated savings

- FFN gate+up: **−42/tok** (gate+up fold to one kernel)
- Plus the geluMul pointwise gets absorbed: **another −42/tok**
- **Total −84/tok** (975 → 891)

### Edits

| File | Change |
|---|---|
| `Hesper/Circuit/IR.lean` | Add `GLUOp` enum + `Prim.matmulQ4KGateGLU` |
| `Hesper/Circuit/Lowering.lean` | New `lowerMatmulQ4KGateGLU` |
| `Hesper/Layers/Linear.lean` | Templatise the existing fused kernel |
| `Hesper/Models/Gemma4.lean` | FFN's wGate/wUp emit through CircuitM |
| `Tests/Circuit/FuseGateGLUTest.lean` | New IR fusion test |

---

## Change 2: `fuseGateMatmulEpilogue` pass (new pass)

### Motivation

Walk the IR for Pattern B/C and rewrite into `matmulQ4KGateGLU`. Same
shape as `fuseMatmulEpilogue` (Passes.lean:575).

### Pattern detected

```
op_i:    matmulQ4K(x, wUp)         → up     (single consumer)
op_j:    matmulQ4K(x, wGate)       → gate   (single consumer)
op_k:    pointwise(up, gate, body)  → out
  body = mul (input 0) (silu (input 1))   -- or gelu / reglu
```

Conditions:
- `op_i` and `op_j` consume the **same input `x`**
- `up` and `gate` are each **single-consumer** of `op_k`
- Neither `up` nor `gate` is in `protectedIds`

### Rewrite

```
op_new: matmulQ4KGateGLU(x, wUp, wGate, none, none, .silu) → out
```

### Edits

| File | Change |
|---|---|
| `Hesper/Circuit/Passes.lean` | `fuseGateMatmulEpilogue` / `fuseGateMatmulEpilogueStep` |
| `Tests/Circuit/FuseGateGLUTest.lean` | Pattern-match unit test |

---

## Change 3: extend `reduceWithEpilogue` to absorb ADD

### Motivation

Pattern D from llama.cpp: `RMS_NORM + MUL + ADD` → 1 kernel.
Today `fuseReduceEpilogue` covers `reduce → pointwise(mul)`.
Adding ADD just means accepting a 3-input `ScalarExp` body.

### Design

Change `fuseReduceEpilogueStep` (Passes.lean:376) **only** in pattern
matching:

```
Now:    reduce → pointwise(2-input mul)  → reduceWithEpilogue(mul body)
Want:   reduce → pointwise(2-input mul) → pointwise(2-input add)
        → reduceWithEpilogue(mul+add body, 3 inputs)
```

### Estimated savings

- attnNorm / ffnNorm sites where MUL+ADD is chained: **−42/tok**

### Edits

| File | Change |
|---|---|
| `Hesper/Circuit/Passes.lean` | Pattern extension in `fuseReduceEpilogueStep` |
| `Hesper/Circuit/Lowering.lean` | 3-input reduce-with-epilogue kernel |

---

## Change 4: n-ary ADD fusion

### Motivation

Pattern F: `ADD → ADD → ... → ADD` (max 8) → 1 kernel. hesper's
residual-add chain matches.

### Design

Extend `fusePointwise` (Passes.lean:278) to compress chained binary ADDs
into an n-ary pointwise:

```
add(add(add(a, b), c), d) → pointwise(a, b, c, d, body=input0+input1+input2+input3)
```

Today `fusePointwise` only fuses 2-input → 1-output chains.
n-input `ScalarExp` is already representable via `input : Nat → ScalarExp`.

### Estimated savings

- residual-add chains: **−30/tok**

### Edits

| File | Change |
|---|---|
| `Hesper/Circuit/Passes.lean` | Multi-stage ADD-chain detection in `fusePointwiseStep` |

---

## Change 5: explicit single-consumer safety check

### Motivation

llama.cpp's `ggml_can_fuse_ext` always checks "intermediate tensor has
exactly 1 consumer" (`ggml-impl.h:680–682`). hesper's fusion passes
protect externally-visible tensors via `protectedIds`, but **two
non-protected consumers of the same intermediate** may not be guarded
explicitly.

### Design

Add a `countConsumers` guard at the start of each fusion pass requiring
"intermediate id is not in protectedIds AND consumer count == 1".

### Edits

| File | Change |
|---|---|
| `Hesper/Circuit/Passes.lean` | `countConsumers` helper + guards |

---

## Priority order

| Priority | Change | Saves | Difficulty | Depends on |
|---:|---|---:|---|---|
| 1 | **`matmulQ4KGateGLU`** + **`fuseGateMatmulEpilogue`** | −84 | Medium (3–5 days) | none |
| 2 | `reduceWithEpilogue` ADD extension | −42 | Small (1 day) | none |
| 3 | n-ary ADD fusion | −30 | Small (1 day) | none |
| 4 | single-consumer guard | 0 (correctness) | Small (0.5 day) | none |
| 5 | PLE path → Circuit | −150 | Large (1–2 weeks) | 1–4 |

**Step 1 has the highest single-commit ROI.** Steps 2–4 are independent
and parallelisable. Step 5 builds on 1–4.

---

## Long-term roadmap: kernels-per-token outlook

```
          975 ←── starting point
    Step 1: −84  (matmulQ4KGateGLU)
    Step 2: −42  (reduce+MUL+ADD)
    Step 3: −30  (n-ary ADD)
    ────────────
          819
    Step 5: −150 (PLE → Circuit)
    Step 6: −84  (residual-add as matmul epilogue)
    ────────────
          585
    ────────────
    Theoretical floor: ~400/tok (without per-layer persistent kernel)
    llama.cpp CUDA:   187/tok (CUDA Graphs + multi-op-per-kernel)
```

Reaching 187/tok requires **per-layer persistent kernels** (one kernel
per layer per token). llama.cpp gets there indirectly via CUDA Graphs;
the same can be achieved with custom kernel design (deferred).
