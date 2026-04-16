---
title: "05 — hesper Circuit DSL 拡張計画"
date: 2026-04-16
---

# hesper Circuit DSL 拡張計画

llama.cpp CUDA fusion 調査で判明した gap を Circuit DSL で埋めるための
具体的な変更計画。

## 前提: 既存 DSL アーキテクチャ

```
[IR 層]   Prim → Op → TensorRef → Circuit state
[Pass 層] fusePointwise / fuseReduceEpilogue / fuseMatmulEpilogue
[Lowering] Prim → ShaderM → PTX JIT → cuLaunchKernel
```

### 既存 Prim 一覧 (IR.lean:163–213)

| Prim | 概要 | 行番号 |
|---|---|---|
| `pointwise` | N-input → 1-output element-wise | `IR.lean:163` |
| `matmulQ4K` | Q4_K mat-vec | `IR.lean:172` |
| `matmulQ4KWithEpilogue` | Q4_K mat-vec + pointwise tail | `IR.lean:176` |
| `reduceLastAxis` | sum / sumOfSquares | `IR.lean:191` |
| `reduceLastAxisWithEpilogue` | reduce + pointwise tail | `IR.lean:211` |

### 既存 Fusion Pass 一覧 (Passes.lean)

| Pass | 概要 | 行番号 |
|---|---|---|
| `fusePointwise` | pointwise チェーン圧縮 | `Passes.lean:278` |
| `fuseReduceEpilogue` | reduce → pointwise 融合 | `Passes.lean:451` |
| `fuseMatmulEpilogue` | matmulQ4K → pointwise 融合 | `Passes.lean:575` |
| `mergeSameDispatch` | 同一 dispatch 統合 | `Passes.lean:53` |

---

## 変更 1: `Prim.matmulQ4KGateGLU` (新規 Prim)

### 動機

llama.cpp Pattern B/C: `MUL_MAT + [ADD +] MUL_MAT + [ADD +] GLU` → 1 kernel。
Gemma 4 FFN の gate+up 構造がこのパターンに合致。現状 hesper は
`fusedQ4KMLinearDP4AGeluSliceKernel` として手書きだが、IR 化されていないため
Circuit DSL の fusion pass から到達不可能。

### 設計

```lean
-- Hesper/Circuit/IR.lean に追加
inductive GLUOp where
  | silu     -- result *= silu(gate)
  | gelu     -- result *= gelu(gate)
  | reglu    -- result *= gate
  | swigluOai -- swiglu-openai variant

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

[`Lowering.lean`](../../../Hesper/Circuit/Lowering.lean) に追加:

```
Prim.matmulQ4KGateGLU main gate xBias gateBias gluOp
  → 既存 fusedQ4KMLinearDP4AGeluSliceKernel と同構造の PTX kernel を emit
  → inner loop: main dot-product + gate dot-product を並列計算
     (mmvq.cu:494-497 と同パターン)
  → epilogue: bias add → gluOp 適用 → f32 output
```

### 推定削減

- FFN gate+up fusion: **−42/tok** (gate+up が 1 kernel に)
- 加えて、geluMul pointwise を吸収: **追加 −42/tok**
- **合計 −84/tok** (975 → 891)

### 実装場所

| ファイル | 変更内容 |
|---|---|
| `Hesper/Circuit/IR.lean` | `GLUOp` enum + `Prim.matmulQ4KGateGLU` 追加 |
| `Hesper/Circuit/Lowering.lean` | `lowerMatmulQ4KGateGLU` lowering 関数 |
| `Hesper/Layers/Linear.lean` | 既存 `fusedQ4KMLinearDP4AGeluSliceKernel` を template 化 |
| `Hesper/Models/Gemma4.lean` | FFN の wGate/wUp を CircuitM 経由で emit |
| `Tests/Circuit/FuseGateGLUTest.lean` | IR fusion テスト (新規) |

---

## 変更 2: `fuseGateMatmulEpilogue` pass (新規 Pass)

### 動機

IR を走査して Pattern B/C を自動検出 → `matmulQ4KGateGLU` に置換。
`fuseMatmulEpilogue` (Passes.lean:575) と同系統の AST walker。

### 検出パターン

```
op_i:    matmulQ4K(x, wUp)         → up     (single consumer)
op_j:    matmulQ4K(x, wGate)       → gate   (single consumer)
op_k:    pointwise(up, gate, body)  → out
  body = mul (input 0) (silu (input 1))   -- or gelu / reglu
```

条件:
- `op_i` と `op_j` が **同一入力 x** を消費
- `up` と `gate` が **単一 consumer** (`op_k`)
- `up`, `gate` がどちらも `protectedIds` に含まれない

### 置換結果

```
op_new: matmulQ4KGateGLU(x, wUp, wGate, none, none, .silu) → out
```

### 実装場所

| ファイル | 変更内容 |
|---|---|
| `Hesper/Circuit/Passes.lean` | `fuseGateMatmulEpilogue` / `fuseGateMatmulEpilogueStep` |
| `Tests/Circuit/FuseGateGLUTest.lean` | パターンマッチテスト |

---

## 変更 3: `reduceWithEpilogue` の ADD 拡張

### 動機

llama.cpp Pattern D: `RMS_NORM + MUL + ADD` → 1 kernel。
現在の `fuseReduceEpilogue` は `reduce → pointwise(mul)` まで。
`ADD` を追加するには epilogue の `ScalarExp` body を 3-input に拡張。

### 設計

変更箇所は `fuseReduceEpilogueStep` (Passes.lean:376) の**パターンマッチ拡張**のみ:

```
現在: reduce → pointwise(2-input mul)  → reduceWithEpilogue(mul body)
拡張: reduce → pointwise(2-input mul) → pointwise(2-input add)
      → reduceWithEpilogue(mul+add body, 3 inputs)
```

### 推定削減

- attnNorm / ffnNorm で MUL+ADD がチェーン化されている箇所: **−42/tok**

### 実装場所

| ファイル | 変更内容 |
|---|---|
| `Hesper/Circuit/Passes.lean` | `fuseReduceEpilogueStep` のパターン拡張 |
| `Hesper/Circuit/Lowering.lean` | 3-input reduce-with-epilogue kernel emit |

---

## 変更 4: 連続 ADD の n-ary fusion

### 動機

llama.cpp Pattern F: `ADD → ADD → ... → ADD` (max 8) → 1 kernel。
hesper の residual 加算チェーンがこれに該当。

### 設計

`fusePointwise` (Passes.lean:278) を拡張して、連続する binary ADD を n-ary
pointwise に圧縮:

```
add(add(add(a, b), c), d) → pointwise(a, b, c, d, body=input0+input1+input2+input3)
```

現在の `fusePointwise` は 2-input → 1-output の chain fusion のみ。
n-input `ScalarExp` は IR.lean の `input : Nat → ScalarExp` で既に表現可能。

### 推定削減

- residual add chain: **−30/tok**

### 実装場所

| ファイル | 変更内容 |
|---|---|
| `Hesper/Circuit/Passes.lean` | `fusePointwiseStep` の多段 ADD chain 検出 |

---

## 変更 5: Single-consumer safety check 明文化

### 動機

llama.cpp `ggml_can_fuse_ext` は「中間テンソルが exactly 1 consumer」を必ず
チェック (ggml-impl.h:680–682)。hesper の fusion pass は `protectedIds` で
外部参照を保護しているが、**2 つの non-protected consumer が同じ中間テンソルを
消費** する場合に安全性が保証されているか未検証。

### 設計

各 fusion pass の冒頭で「中間テンソルが protectedIds 外かつ
consumer 数 == 1」を明示的に検証する guard を追加。

### 実装場所

| ファイル | 変更内容 |
|---|---|
| `Hesper/Circuit/Passes.lean` | `countConsumers` ヘルパー + guard |

---

## 優先順位

| 順位 | 変更 | 削減 | 難易度 | 依存 |
|---:|---|---:|---|---|
| 1 | **matmulQ4KGateGLU** + **fuseGateMatmulEpilogue** | −84 | 中 (3–5日) | なし |
| 2 | reduceWithEpilogue ADD 拡張 | −42 | 小 (1日) | なし |
| 3 | 連続 ADD n-ary fusion | −30 | 小 (1日) | なし |
| 4 | single-consumer guard | 0 (正しさ) | 小 (0.5日) | なし |
| 5 | PLE 経路 Circuit 化 | −150 | 大 (1–2週) | 1–4 |

**Step 1 が最大 ROI**。Step 2–4 は並行可能。Step 5 は Step 1–4 完了後。

---

## 長期ロードマップ: kernels/tok 見込み

```
          975 ←── 現状
    Step 1: −84  (matmulQ4KGateGLU)
    Step 2: −42  (reduce+MUL+ADD)
    Step 3: −30  (n-ary ADD)
    ────────────
          819
    Step 5: −150 (PLE Circuit化)
    Step 6: −84  (residual-add epilogue)
    ────────────
          585
    ────────────
    理論下限: ~400/tok (per-layer persistent kernel なし)
    llama.cpp CUDA: 187/tok (CUDA Graphs + multi-op-per-kernel)
```

187/tok に到達するには **per-layer persistent kernel** (1 token = 1 kernel/layer)
が必要。これは llama.cpp が CUDA Graphs で間接的に達成しているものの、
個別 kernel 設計でも可能（将来検討）。
