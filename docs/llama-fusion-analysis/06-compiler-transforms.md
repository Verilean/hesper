---
title: "06 — コンパイラ変換として見た fusion: reduce の壁と汎用化戦略"
date: 2026-04-16
---

# コンパイラ変換として見た fusion

## 1. reduce が fusion の壁になる理由

現在の Circuit DSL の fusion pass 構造：

| Pass | producer Prim | consumer Prim | 結果 |
|---|---|---|---|
| `fusePointwise` | `pointwise` | `pointwise` | `pointwise` (body inlined) |
| `fuseReduceEpilogue` | `reduceLastAxis` | `pointwise` chain | `reduceLastAxisWithEpilogue` |
| `fuseMatmulEpilogue` | `matmulQ4K` | `pointwise` | `matmulQ4KWithEpilogue` |

**全 pass が「producer と consumer が特定の Prim 種族ペア」でないとスキップする。**

reduce（sum）が間に入ると：
- `fusePointwise`: producer/consumer とも `pointwise` 必須 → `reduceLastAxis` はスキップ
- `fuseMatmulEpilogue`: consumer が `pointwise` 必須 → matmul→reduce→pointwise のうち
  matmul の直後が reduce なので融合不可
- `fuseReduceEpilogue`: producer が `reduceLastAxis` 必須 → pointwise→reduce は不可

→ **reduce は fusion chain を切断する壁として機能**

## 2. reduce をまたぐ代数的変換

数学的に可換な変換は自動化可能：

```
a × Σ x_i = Σ (a × x_i)     -- scalar を reduce の中に吸収
(Σ x_i) + b = Σ x_i + b     -- bias は reduce の外
f(Σ x_i)  where f is pointwise -- reduce の後に pointwise 適用
```

3番目は **`fuseReduceEpilogue` が既にやっていること**。
1番目（reduce の前に pointwise を吸収）は未実装。

## 3. llama.cpp 8 パターンのコンパイラ変換分類

| # | パターン | コンパイラ変換名 | 概要 |
|---|---|---|---|
| A | matmul + bias | **epilogue absorption** | reduce 後の pointwise を吸収 |
| B/C | matmul×2 + GLU | **parallel reduce fusion** | 同入力の 2 reduce を 1 kernel に |
| D | RMSNorm + mul + add | **reduce-epilogue chain extension** | epilogue chain を延長 |
| E | RoPE + view + set_rows | **output destination fusion** | 書き込み先をリダイレクト |
| F | add×N | **n-ary pointwise collapse** | 多段 pointwise を 1 op に圧縮 |
| H | silu + mul | **pointwise fusion** | 既存 `fusePointwise` |

**B/C と E のみが新しい変換**。他は既存 pass の拡張。

## 4. Parallel Reduce Fusion (B/C パターン)

### パターン

```
r1 = reduce(x, weights1)    -- matmul = weighted sum = reduce
r2 = reduce(x, weights2)    -- 同じ x を消費
out = pointwise(r1, r2)     -- silu(r2) * r1 (GLU)
```

2 つの Σ が同じ j を走査するので、1 ループで同時に累積可能。

### IR 設計: `parallelMatmulWithEpilogue`

```lean
| parallelMatmulWithEpilogue
    (layers : Array (LinearLayer BufT CacheT))  -- N 個の weight
    (epiBody : ScalarExp)
    -- body で input 0 = layers[0] の結果
    --       input 1 = layers[1] の結果
    --       input N.. = 追加の side inputs
```

- N=1, body=identity → 現在の `matmulQ4K`
- N=1, body=add(input 0)(input 1) → 現在の `matmulQ4KWithEpilogue` (bias)
- **N=2, body=mul(input 0)(silu(input 1))** → llama.cpp Pattern B/C

### Lowering

inner loop に N 本の accumulator を並列配置（`mmvq.cu:494–497` と同パターン）：

```cuda
tmp[j][i]      += vec_dot_q(vx_main, &y[...], ...);  // main
tmp_gate[j][i] += vec_dot_q(vx_gate, &y[...], ...);  // gate (same y)
```

### Fusion Pass 拡張

`fuseMatmulEpilogue` を「同一入力を消費する N 個の matmul + 合流 pointwise」パターンに拡張。

### 対応ソース

- llama.cpp 実装: [`mmvq.cu:494–497`](../../../llama.cpp/ggml/src/ggml-cuda/mmvq.cu)
  (gate matmul の inner loop 並列計算)
- llama.cpp graph fusion: [`ggml-cuda.cu:3810–3886`](../../../llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu)
  (Pattern B: 5 ops → 1 kernel)
- hesper 現行: [`Hesper/Circuit/IR.lean:176`](../../../Hesper/Circuit/IR.lean)
  (`matmulQ4KWithEpilogue` — N=1 のみ)
- hesper fusion pass: [`Hesper/Circuit/Passes.lean:485–580`](../../../Hesper/Circuit/Passes.lean)
  (`fuseMatmulEpilogue` — 1 matmul + 1 pointwise のみ)

## 5. Output Destination Fusion (E パターン)

### パターン

```
1. result = compute(inputs)            -- RoPE, matmul, pointwise 等
2. view(result)                        -- zero-cost shape reinterpret
3. set_rows(cache, view_out, pos)      -- 既存バッファの特定位置に書き込み
```

VIEW は zero-cost（メモリコピーなし）。実質：

```
compute(inputs) → cache[pos] に直接書き込み
```

### 現行 DSL で表現できない理由

1. **alias の概念がない**: `TensorRef` は `(id, shape, dtype, scope)` で id が
   unique identity。同じバッファを別 shape で見る view は「新 id = 新バッファ = コピー」

2. **output は常に fresh allocation**: `emitOp` (IR.lean:289–294) が `allocTensor`
   で新バッファを作る。「既存バッファの pos 番目に書く」は表現不能。

3. **fusion pass が output 先をリダイレクトする仕組みがない**

### IR 設計案

```lean
-- TensorRef に alias 情報を追加
structure TensorRef where
  id     : Nat
  shape  : Shape
  dtype  : DType
  scope  : Scope
  base   : Option Nat := none   -- 元バッファの id (alias の場合)
  offset : Nat := 0             -- 元バッファ内のオフセット

-- 新 Prim
inductive Prim where
  ...
  | view (newShape : Shape)             -- zero-cost reshape / alias 作成
  | writeSlice (dstOffset : Nat)        -- output を dst[offset..] に書く
```

### Fusion Pass: `fuseWriteDestination`

```
Rule: fuse-write-destination

Match:
  op_a: any_compute(inputs) → r       (single consumer = op_b)
  op_b: writeSlice(cache, r, offset)  (r を cache[offset] に書く)

Rewrite:
  op_a の output address を cache[offset] に差し替え
  op_b を削除
```

**任意の compute op に適用可能** — RoPE 固有ではない：

```
matmul(x, w)    → result → writeSlice(cache, result, pos)   -- wO → residual 直接書き
pointwise(a, b) → result → writeSlice(buf, result, offset)  -- PLE scale 直接書き
```

GCC の **store sinking / destination propagation** に相当する汎用最適化。

### 必要な変更

| 変更 | ファイル | 難易度 |
|---|---|---|
| `TensorRef` に `base`/`offset` 追加 | [`IR.lean:55`](../../../Hesper/Circuit/IR.lean) | 小 |
| `Prim.view` 追加 | [`IR.lean:151+`](../../../Hesper/Circuit/IR.lean) | 小 |
| `Prim.writeSlice` 追加 | [`IR.lean:238+`](../../../Hesper/Circuit/IR.lean) | 小 |
| `fuseWriteDestination` pass | [`Passes.lean`](../../../Hesper/Circuit/Passes.lean) (新規) | 中 |
| Lowering: output address を offset 付きで emit | [`Lowering.lean`](../../../Hesper/Circuit/Lowering.lean) | 中 |
| `CircuitM.writeSlice` builder sugar | [`IR.lean:263+`](../../../Hesper/Circuit/IR.lean) | 小 |

### 対応ソース

- llama.cpp graph fusion: [`ggml-cuda.cu:3762–3769`](../../../llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu)
  (ROPE + VIEW + SET_ROWS → 1 kernel)
- llama.cpp predicate: [`ggml-cuda.cu:3357–3365`](../../../llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu)
  (`ggml_cuda_can_fuse` for rope_set_rows_ops)
- hesper 現行 hand-coded: [`Hesper/Models/Gemma4.lean`](../../../Hesper/Models/Gemma4.lean)
  (fused RoPE-K+KVwrite kernel — hand-composed, 非 Circuit)

## 6. 全パターンの DSL 対応まとめ

| # | パターン | コンパイラ変換 | 必要な DSL 改変 | 削減 |
|---|---|---|---|---:|
| A | matmul+bias | epilogue absorption | なし (既存) | — |
| B/C | matmul×2+GLU | parallel reduce fusion | `layers: Array` 汎化 | −84/tok |
| D | RMSNorm+mul+add | reduce-epilogue chain | pass 拡張のみ | −42/tok |
| **E** | **RoPE+view+set_rows** | **output destination fusion** | **view + writeSlice Prim + pass** | **−42/tok** |
| F | add×N | n-ary pointwise | pass 拡張のみ | −30/tok |
| H | silu+mul | pointwise fusion | なし (既存) | — |

## 7. 実装優先順位

1. **E (output destination fusion)** — DSL の表現力向上が最大。view/writeSlice は
   B/C にも将来使える汎用基盤。他のモデルにも再利用可能。
2. **B/C (parallel reduce fusion)** — TPS 効果が最大 (−84/tok)。E で view が入った後なら
   fusion chain が繋がりやすい。
3. **D (reduce-epilogue chain)** — pass 拡張のみ。
4. **F (n-ary pointwise)** — pass 拡張のみ。
