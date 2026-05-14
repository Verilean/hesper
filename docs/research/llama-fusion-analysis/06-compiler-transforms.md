---
title: "06 — Fusion as compiler transforms: the reduce wall and how to generalise"
date: 2026-04-16
---

# Fusion as compiler transforms

## 1. Why reduce is a fusion wall

Today's Circuit DSL fusion passes:

| Pass | Producer Prim | Consumer Prim | Result |
|---|---|---|---|
| `fusePointwise` | `pointwise` | `pointwise` | inlined `pointwise` |
| `fuseReduceEpilogue` | `reduceLastAxis` | `pointwise` chain | `reduceLastAxisWithEpilogue` |
| `fuseMatmulEpilogue` | `matmulQ4K` | `pointwise` | `matmulQ4KWithEpilogue` |

**Every pass requires producer/consumer to be a specific Prim pair.**

When a reduce sits between two ops:
- `fusePointwise`: needs producer **and** consumer to be `pointwise`
  (`Passes.lean:154,175`); a `reduceLastAxis` breaks the chain.
- `fuseMatmulEpilogue`: consumer must be `pointwise` (`Passes.lean:485+`);
  matmul → reduce → pointwise can't be fused because the reduce sits
  immediately after the matmul.
- `fuseReduceEpilogue`: producer must be `reduceLastAxis`
  (`Passes.lean:383`), so pointwise → reduce or reduce → reduce don't fuse.

Net: **reduce acts as a wall in the fusion DAG.**

## 2. Algebraic transforms across reduce

Mathematically valid moves we'd like a compiler to do automatically:

```
a × Σ x_i = Σ (a × x_i)            -- pull a scalar inside the reduce
(Σ x_i) + b = Σ x_i + b             -- pull a bias outside the reduce
f(Σ x_i)  where f is pointwise      -- apply pointwise after the reduce
```

The third case is what `fuseReduceEpilogue` already does.
The first case (push pointwise into the reduce) is **not implemented**.

## 3. Classifying llama.cpp's eight patterns as compiler transforms

| # | Pattern | Compiler-transform name | Notes |
|---|---|---|---|
| A | matmul + bias | **epilogue absorption** | absorb pointwise after reduce |
| B/C | matmul×2 + GLU | **parallel reduce fusion** | merge two reduces sharing input |
| D | RMSNorm + mul + add | **reduce-epilogue chain extension** | extend the epilogue chain |
| E | RoPE + view + set_rows | **output destination fusion** | redirect the write |
| F | add×N | **n-ary pointwise collapse** | collapse multi-stage pointwise |
| H | silu + mul | **pointwise fusion** | existing `fusePointwise` |

**B/C and E are the only patterns that need genuinely new transforms.**
The rest are extensions of existing passes.

## 4. Parallel reduce fusion (Patterns B/C)

### Pattern

```
r1 = reduce(x, weights1)    -- matmul = weighted sum = reduce
r2 = reduce(x, weights2)    -- same input
out = pointwise(r1, r2)     -- silu(r2) * r1 (GLU)
```

The two Σ traverse the same `j`, so we can accumulate both in one loop.

### IR design: `parallelMatmulWithEpilogue`

```lean
| parallelMatmulWithEpilogue
    (layers : Array (LinearLayer BufT CacheT))  -- N weight tensors
    (epiBody : ScalarExp)
    -- body: input 0 = layers[0] output, input 1 = layers[1] output, ...
    --       input N.. = additional side inputs
```

- N=1, body=identity → equivalent to `matmulQ4K`
- N=1, body=add(input 0)(input 1) → equivalent to `matmulQ4KWithEpilogue` (bias)
- **N=2, body=mul(input 0)(silu(input 1))** → llama.cpp B/C

### Lowering

Place N parallel accumulators in the inner loop (same shape as
`mmvq.cu:494–497`):

```cuda
tmp[j][i]      += vec_dot_q(vx_main, &y[...], ...);  // main
tmp_gate[j][i] += vec_dot_q(vx_gate, &y[...], ...);  // gate (same y)
```

### Pass extension

Extend `fuseMatmulEpilogue` to detect "N matmuls sharing an input,
joined at a pointwise consumer".

### Source links

- llama.cpp impl: [`mmvq.cu:494–497`](../../../llama.cpp/ggml/src/ggml-cuda/mmvq.cu)
- llama.cpp graph fusion: [`ggml-cuda.cu:3810–3886`](../../../llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu)
- hesper today: [`Hesper/Circuit/IR.lean:176`](../../../Hesper/Circuit/IR.lean)
  (`matmulQ4KWithEpilogue` — N=1)
- hesper pass: [`Hesper/Circuit/Passes.lean:485–580`](../../../Hesper/Circuit/Passes.lean)
  (`fuseMatmulEpilogue` — 1 matmul + 1 pointwise)

## 5. Output destination fusion (Pattern E)

### Pattern

```
1. result = compute(inputs)            -- RoPE / matmul / pointwise
2. view(result)                        -- zero-cost shape reinterpret
3. set_rows(cache, view_out, pos)      -- write at a position in an existing buffer
```

VIEW is zero-cost. Effectively:

```
compute(inputs) → write directly at cache[pos]
```

### Why old DSL couldn't express it

1. **No alias concept.** `TensorRef` was `(id, shape, dtype, scope)`
   with id as unique identity. Viewing a buffer at a different shape
   meant "new id = new buffer = copy".
2. **Outputs were always fresh allocations.** `emitOp` (IR.lean:289–294)
   called `allocTensor`, so "write at slot pos of existing buffer" was
   inexpressible.
3. **No way for a fusion pass to redirect the output.**

### Resolution (commit b515e13)

`Prim.scatter` with explicit `addrExpr : ScalarExp` covers this directly.
See [`08-scatter-impl-notes.md`](08-scatter-impl-notes.md). For the
record, the original design proposal:

```lean
-- Add alias info to TensorRef
structure TensorRef where
  id     : Nat
  shape  : Shape
  dtype  : DType
  scope  : Scope
  base   : Option Nat := none   -- source buffer id (when this is an alias)
  offset : Nat := 0             -- element offset within source

-- New Prims
inductive Prim where
  ...
  | view (newShape : Shape)             -- zero-cost reshape / alias
  | writeSlice (dstOffset : Nat)        -- write to dst[offset..]
```

The actual implementation went further and unified all writes under one
`Prim.scatter`, eliminating the need for a separate `view` and the
alias fields on `TensorRef`. The source links stay valid:

- llama.cpp graph fusion: [`ggml-cuda.cu:3762–3769`](../../../llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu)
- llama.cpp predicate: [`ggml-cuda.cu:3357–3365`](../../../llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu)
- hesper hand-coded: [`Hesper/Models/Gemma4.lean`](../../../Hesper/Models/Gemma4.lean)
  (fused RoPE-K + KVwrite)

## 6. Pattern × DSL coverage

| # | Pattern | Compiler transform | DSL change needed | Saves |
|---|---|---|---|---:|
| A | matmul+bias | epilogue absorption | none (existing) | — |
| B/C | matmul×2+GLU | parallel reduce fusion | `layers : Array` generalisation | −84/tok |
| D | RMSNorm+mul+add | reduce-epilogue chain | pass extension | −42/tok |
| **E** | **RoPE+view+set_rows** | **output destination fusion** | **`Prim.scatter` (DONE)** | **−42/tok** |
| F | add×N | n-ary pointwise | pass extension | −30/tok |
| H | silu+mul | pointwise fusion | none (existing) | — |

## 7. Implementation order (recommended)

1. **E (output destination fusion)** — biggest leap in DSL expressiveness.
   `view` / `writeSlice` (or now `scatter`) become the foundation that
   B/C also rest on. Reusable across models.
2. **B/C (parallel reduce fusion)** — biggest TPS win (−84/tok). Easier
   once `view` is in place.
3. **D (reduce-epilogue chain)** — pass extension only.
4. **F (n-ary pointwise)** — pass extension only.
