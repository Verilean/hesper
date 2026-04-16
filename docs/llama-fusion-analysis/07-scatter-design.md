---
title: "07 — Dynamic Offset / Scatter の設計課題"
date: 2026-04-16
status: open — next session に持ち越し
---

# Dynamic Offset / Scatter の設計課題

## 今日の発見

Pattern E (`ROPE + VIEW + SET_ROWS`) を hesper の Circuit DSL に取り込む
過程で、**純粋な data-parallel Map** と **動的アドレス計算を伴う Scatter**
の意味論的ギャップが浮上した。

具体例: KV cache write の書き込み先アドレスは

```
out[kvHead * maxSeqLen * headDim + pos * headDim + d] = ...
```

で、`pos` は **runtime parameter buffer から読み出す動的値**。
現状の `Prim.pointwise` / `Prim.writeSlice` ではこのアドレス計算を表現
できない。

## 現状の DSL セマンティクス

### Map パターン（現状サポート）

```
Prim.pointwise outShape inShapes body
  thread i:
    slots[k] = inputs[k][i]   -- broadcast if inShapes[k] == #[1] else [i]
    out[i]   = body(slots)
```

**暗黙の前提**: 出力インデックス `i` と入力インデックス `i` が**同一** (identity map)。
`ScalarExp` の `.input k` は「同じ lane の k 番目 input」を読むだけ。

### 今日追加した `Prim.writeSlice`

```
Prim.writeSlice dstShape dstOffset srcShape
  thread i:
    out[dstOffset + i] = src[i]
```

`dstOffset : ScalarExp` に汎化したが、この `ScalarExp` の `.input k`
**何を参照するのか未定義**。現状は slot array が空なので参照不能。

## 意味論ギャップの本質

| 操作 | 出力アドレス | 入力アドレス | 現状 |
|---|---|---|---|
| Map (pointwise) | `i` (identity) | `i` (identity) | ✅ |
| Gather | `i` | `f(i, params)` | ❌ 未対応 |
| **Scatter** | `f(i, params)` | `i` | ❌ **Pattern E はこれ** |
| General (MoE 的) | `g(i, params)` | `f(i, params)` | ❌ |

Scatter を表現するには **「値を計算する式」と「書き込み先アドレスを
計算する式」を別々に、両方に runtime parameter を渡せる** 必要がある。

## 現状の機能で足りる/足りないポイント

### ✅ 今日の改良で獲得したもの

- `ScalarExp` に 9 演算子追加: `cos`, `sin`, `pow`, `lt`, `select`,
  `mod`, `idiv`, `toFloat`, `laneIdx`
- `laneIdx` で thread index を式として取得可能
- `+`, `-`, `*`, `/`, `neg`, `OfNat`, `OfScientific`, `BEq` オーバーロード
- `writeSlice` / `pointwiseToSlice` の offset が `ScalarExp`（static const は OK）
- `fuseWriteDestination` pass が動作

### ❌ まだ不足

1. **アドレス計算から parameter buffer を参照できない**
   - 現状: `ScalarExp` の `.input k` は **pointwise body の slot** を指す
   - 必要: offset 計算からも同じ方式で parameter buffer を読める枠組み

2. **値計算 (value) とアドレス計算 (index) が分離されていない**
   - 現状: `Prim.pointwise.body` が唯一の `ScalarExp` で、暗黙に output index は thread id
   - 必要: 「値はこう、書き込み先はこう」を明示的に書ける Prim

## 提案: `Prim.scatter`

次セッションで検討する設計案：

```lean
/-- Scatter primitive: compute value + destination address per lane.

    thread i:
      valueSlots[k]    = valueInputs[k][i]     (broadcast if [1])
      addrSlots[k]     = addrInputs[k][i]      (broadcast if [1])
      out[addrExpr]    = valueExpr             (evaluated with above slots)

    - `valueExpr`  : the value to write (uses `.input k` for valueSlots)
    - `addrExpr`   : the destination index (uses `.input k` for addrSlots)
    - `.laneIdx`   : thread's global id, available in both exprs
    - `outShape`   : dispatch grid shape (one thread per output element to write)
    - `dstShape`   : destination buffer shape (addrExpr must be in [0, dstShape.numel))

    Covers Map (identity addrExpr = laneIdx), writeSlice (addrExpr = laneIdx + const),
    and Scatter (addrExpr uses parameter buffers + laneIdx arithmetic). -/
| scatter
    (outShape    : Shape)             -- dispatch grid
    (dstShape    : Shape)              -- destination buffer shape
    (valueInputs : Array Shape)        -- shapes of the value-path inputs
    (addrInputs  : Array Shape)        -- shapes of the addr-path inputs
    (valueExpr   : ScalarExp)
    (addrExpr    : ScalarExp)
```

呼び出し規約:
- `inputs` array layout: `[dst, valueInputs..., addrInputs...]`
- `valueExpr` の `.input k` → `valueInputs[k]`
- `addrExpr` の `.input k` → `addrInputs[k]`
- 両方で `.laneIdx` が使える

### 既存の Prim との関係

| 既存 Prim | `scatter` による表現 |
|---|---|
| `pointwise outShape inShapes body` | `scatter outShape outShape inShapes #[] body .laneIdx` |
| `writeSlice dstShape dstOffset srcShape` | `scatter srcShape dstShape #[srcShape] #[] (.input 0) (.laneIdx + dstOffset)` |
| `pointwiseToSlice` | 上記の合成（`valueExpr` に任意 body、`addrExpr = laneIdx + offset`） |

つまり `scatter` は**これら全部を包含する一般化**。既存の Prim は
lowering の特殊化（`addrExpr = .laneIdx` なら offset 計算 code を省略）と
捉えられる。

## KV cache write が `scatter` でどう書けるか

RoPE 無しの V cache write を例に：

```lean
-- external inputs
let vNew    ← registerExternal newVBuf       #[kvDim]                .f32 .Global
let vCache  ← registerExternal kvCacheV      #[numKVHeads * maxSeqLen * headDim] .f32 .Global
let params  ← registerExternal paramsBuf     #[1]                    .f32 .Global  -- stores `pos`

-- scatter: v_cache[kvHead * maxSeqLen * headDim + pos * headDim + d] = v_new[i]
--   where i = laneIdx, kvHead = i / headDim, d = i % headDim
open ScalarExp in
let i     := laneIdx
let d     := mod i (.const headDim.toFloat)
let kvH   := idiv i (.const headDim.toFloat)
let pos   := .input 0  -- addrInputs[0] = params
let addr  := kvH * .const (maxSeqLen * headDim).toFloat
           + pos * .const headDim.toFloat
           + d
let _out ← CircuitM.scatter
  (outShape := #[kvDim])
  (dst := vCache)
  (valueInputs := #[vNew])
  (addrInputs  := #[params])
  (valueExpr := .input 0)
  (addrExpr := addr)
```

RoPE K cache write は `valueExpr` に RoPE 計算 (`cos`/`sin`/`pow`) を
書くだけで同じ枠組みに収まる。

## Lowering 方針

- `valueInputs` と `addrInputs` は別々に buffer read
  - value path は lane indexing か broadcast
  - addr path は常に broadcast（[1]-shape か同じ broadcast ルール）
- `addrExpr` は f32 で計算して最後に `Exp.toU32` で integer に
- output address bounds check は optional（debug only）

## Fusion 側への影響

`fuseWriteDestination` pass も拡張が必要:
- `[A: pointwise] → [B: writeSlice with dynamic offset]` を検出
- B の addr inputs を A の inputs に merge して 1 つの `scatter` を生成

`writeSlice` を `scatter` の syntactic sugar として再定義すれば、既存
fusion pass はほぼそのまま動くはず。

## 次セッションのタスク

1. `Prim.scatter` を IR に追加（または `Prim.writeSlice` / `pointwiseToSlice`
   を置き換え）
2. Lowering を実装
3. `fuseWriteDestination` pass を新 Prim に対応
4. IR ユニットテスト（static offset / dynamic offset 両方）
5. Gemma4.lean の KV cache write を Circuit DSL 経由に置換
   - まず V cache (plain copy + dynamic offset) で動作確認
   - 次に K cache (RoPE + plain scatter) — ここで `valueExpr` の複雑さも検証
6. kernels/tok と decode bit-identical を確認

## 関連ファイル

- 現状の IR: [`Hesper/Circuit/IR.lean`](../../../Hesper/Circuit/IR.lean)
- 現状の Passes: [`Hesper/Circuit/Passes.lean`](../../../Hesper/Circuit/Passes.lean)
- 現状の Lowering: [`Hesper/Circuit/Lowering.lean`](../../../Hesper/Circuit/Lowering.lean)
- 既存テスト: [`Tests/Circuit/FuseWriteDestinationTest.lean`](../../../Tests/Circuit/FuseWriteDestinationTest.lean)
- llama.cpp Pattern E 実装: [`llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:3762`](../../../llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu)
- 置換対象: [`Hesper/Models/Gemma4.lean:1682–1697`](../../../Hesper/Models/Gemma4.lean)
  (RoPE-K + KV write の hand-coded kernel)

## 教訓

今回の設計課題は、**DSL の第一原則**を再確認させてくれた:

> **データ並列の意味論は「どこに書くか」と「何を書くか」が独立している**

hesper の現状 DSL はこの独立性を暗黙に identity にしてしまっていた。
`scatter` を導入すれば、Map / Scatter / さらに将来的な Gather や
atomic accumulator まで同じ枠組みで扱える。

hand-coded kernel を急いで置き換えず、**DSL の表現力不足を発見した
時点で立ち止まって設計し直す**という判断自体が、このプロジェクトの
健全性を担保している。
