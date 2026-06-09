# V11 vs llama.cpp `flash_attn_ext_vec` 認知ギャップ分析

Step 1-8 の ShaderM ergonomics 強化後に **残っている** 認知ギャップを、
V11 (= llama.cpp の `nthreads_KQ=8` sub-warp partition 版に対応) の
コード A/B 比較で抽出する。各項目に Step 9 候補の改善提案を付ける。

参照:
- llama.cpp: `llama.cpp/ggml/src/ggml-cuda/fattn-vec.cuh` (≈600 行)
- hesper:   `Hesper/WGSL/FlashAttentionExperiments.lean` `flashAttentionVecParamsKernelV11` (1513-1853)

## 0. 結論サマリ

| カテゴリ | 当初のギャップ | 現状 |
|---|---|---|
| 1 行レベル算術 | 小 | ✅ Step 1/8 |
| 1 行レベル制御 | 小 | ✅ Step 4/5 |
| ループ構造 | 中 | ✅ Step 9a `unrollForScoped` |
| メモリ参照 | 大 | ✅ Step 9b `MutPtr.advance` (`Ptr.atOffset` Step 6 と併用) |
| 配列レジスタ | 大 | ✅ Step 9c `RegArray ty n` |
| smem reuse | 中 | 未着手 (Step 9d、ROI 低) |
| sentinel化 | 中 | ✅ Step 9f `Exp.negInf30` 他 |
| online softmax | 中 | ✅ Step 9e `softmaxOnlineUpdate` |
| `__syncwarp()` | 中 | ✅ Step 9g `warpBarrier` (PTX `bar.warp.sync`) |

### 残ってるもの

- **S9d** smem alias view: 1 つの smem を quantize tmp と KQ score tmp で alias 再利用するパターン。 V11 / fattn-vec のみで価値があるので、本格的にこの kernel を最適化したくなったタイミングで着手で十分。
- **lane-id pattern match sugar**: `if laneOwns iKQ subWarp ...` のような helper。優先度低、必要になったら追加。
- **DSL 用 `for` macro**: CUDA の `for (init; cond; step1, step2, step3)` 多進行 update を Lean で 1 行表現する macro。Lean の構文制約で根本的に必要なときに検討。

---

## 1. 1 行レベル: 既に縮まったギャップ

### 1.1 算術 (Step 1, OK)

```cpp
// llama.cpp 261-262
float sum = vec_dot_KQ(K + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);
sum = warp_reduce_sum<nthreads_KQ>(sum);
```
```lean
-- hesper V11 1640-1643 (per-pair の inline 版)
ShaderM.assign partialVar (Exp.add (Exp.var partialVar) (Exp.mul q0Exp k0))
```

(MutVar 化していないのでまだ verbose。Step 2 の `+↦` を使えば
`partialVar +↦ q0Exp * k0` で 1 行になる。)

**Step 9 提案**: V11 を MutVar 化して Step 2 を使い切る。

### 1.2 比較 (Step 8, OK)

```cpp
if (kPos < splitEnd) ...
```
```lean
-- before Step 8: Exp.lt kPos splitEnd
-- after  Step 8: kPos <ᵉ splitEnd
```
完全に 1:1。

### 1.3 lane / warp (Step 4, OK)

```cpp
const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
const int laneId = tid & 31;
```
```lean
let tid := Exp.vec3X lid
let laneId := Exp.bitAnd tid (Exp.litU32 31)
-- もしくは: laneId ← ShaderM.laneId
```
hesper の現 V11 はまだ手書き。`ShaderM.laneId` に統一するのは即効性のある clean-up。

### 1.4 warp reduce (Step 5, OK)

```cpp
sum = warp_reduce_sum<nthreads_KQ>(sum);  // 8-way: 3 shfl_xor 1,2,4
```
```lean
-- V11 1646-1657 (手書き 12 行):
let s1Name ← ShaderM.var (.scalar .f32) (... shuffleXor ... 1)
let s2Name ← ShaderM.var (.scalar .f32) (... shuffleXor ... 2)
let s4Name ← ShaderM.var (.scalar .f32) (... shuffleXor ... 4)
-- Step 5 提供済の 1 行:
let sum ← ShaderM.warpReduceSum 8 (Exp.var partialVar)
```

**Step 9 提案**: V11 を `ShaderM.warpReduceSum 8 ...` に切り替える。
12 行 → 1 行。バグ余地が消える。

---

## 2. 構造レベル: 残っているギャップ

### 2.1 `#pragma unroll for` vs `for in [0:n] do ShaderM.scope do`

```cpp
#pragma unroll
for (int i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
    ...
}
```
```lean
for iKQ0 in [0:nthreadsKQ] do ShaderM.scope do
    ...
```

CUDA では `#pragma unroll` が「iter ごとに register reuse のヒント」と
「meta-time 展開」の両方を兼ねる。hesper では:
- meta-time 展開 = `for ... in [0:n] do`
- register scope = `ShaderM.scope do` (Step 269-270)

の **2 つの concept に分かれている**。混在の認知コストが残る。

**Step 9 提案 (S9a)**: `unrollFor` を `ShaderM.scope` 込みの版で再定義する。

```lean
def unrollForScoped (n : Nat) (body : Nat → ShaderM Unit) : ShaderM Unit :=
  (List.range n).forM fun i => ShaderM.scope (body i)
```

これで V11 は

```lean
ShaderM.unrollForScoped nthreadsKQ fun iKQ0 => do
  -- inner body
```

の 1 行で済む。register pressure 抑制と意図が同時表現される。

### 2.2 ポインタ進行 (`K += stride`) vs 絶対 word index

llama.cpp の outer K loop は **ポインタ自体を進める**:

```cpp
K += blockIdx.y*nthreads * nb11;  // 初期 offset
for (int k_VKQ_0 = blockIdx.y*nthreads; k_VKQ_0 < k_VKQ_max;
     k_VKQ_0 += gridDim.y*nthreads,
     K += gridDim.y*nthreads*nb11,    // 進行 1
     V += gridDim.y*nthreads*nb21,    // 進行 2
     maskh += gridDim.y*nthreads) {   // 進行 3
    ...
    float sum = vec_dot_KQ(K + i_KQ*nb11, ...);  // 進行後 K に i_KQ*nb11 を足す
```

hesper の現 V11 は **絶対 word index 計算** を毎 iter 再構築:

```lean
let kRowBaseU32 ← ShaderM.let' (.scalar .u32)
                    (Exp.add kHeadBaseU32 (Exp.mul kPosSafe kRowStrideU32))
let wordOff := Exp.add (Exp.mul subLane (Exp.litU32 dPerLanePair))
                       (Exp.litU32 pk)
let kWordIdx := Exp.add kRowBaseU32 wordOff
```

**ギャップ**: llama.cpp は 1 つの `K` ポインタが 4 つの bookkeeping (head 初期 / split 初期 / outer iter 進行 / inner i_KQ オフセット) を全部担う。
hesper は毎回名前を変えて `kHeadBase + kPosSafe*stride + subLane*16 + pk` を再合成する。

Step 6 で `Ptr ty` は導入済だが、現状 `Ptr.atOffset` は **加算オフセットのみ**。
`K += stride` 形式の **mutable ポインタ進行** がない。

**Step 9 提案 (S9b)**: `MutPtr ty` を追加。

```lean
structure MutPtr (ty : WGSLType) where
  buf : String
  offsetVar : String  -- mutable u32 register holding current offset
  bufLen : Nat
def MutPtr.advance (p : MutPtr ty) (delta : Exp (.scalar .u32)) : ShaderM Unit
def MutPtr.load    (p : MutPtr ty) : ShaderM (Exp ty)
```

それでもなお hesper の outer K loop と CUDA の outer K loop の構造ズレ
(CUDA は 4 進行を update 句で並べる、Lean の `do` は 4 行に並ぶ) は残る。
これは Lean 構文の制約で、根本解消には DSL 用の独自 `for` macro が必要。

### 2.3 配列レジスタ `Q_reg[ncols][D/2/nthreads_KQ]`

llama.cpp:

```cpp
half2  Q_reg[ncols][(D/2)/nthreads_KQ];        // 1 行 declare
Q_reg[j][i] = ...;                              // 直接 index
... Q_reg[j], Q_i32[j], Q_ds[j] ...             // 関数渡しもできる
```

hesper V11:

```lean
let mut q0Vars : Array String := #[]   -- 偶 dim
let mut q1Vars : Array String := #[]   -- 奇 dim
let mut vkq0Vars : Array String := #[]
let mut vkq1Vars : Array String := #[]
for pk in [0:dPerLanePair] do
  let q0Name ← ShaderM.var (.scalar .f32) (Exp.mul q0Val (Exp.litF32 scale))
  ...
  q0Vars := q0Vars.push q0Name
  q1Vars := q1Vars.push q1Name
-- 後で参照:
let q0Exp : Exp (.scalar .f32) := Exp.var q0Vars[pk]!
```

これは **大きな** 認知ギャップ。理由:
- `q0Vars` は Array String (Lean meta) なのに値は Exp (target) → 二段管理
- index `[pk]!` で `Exp.var name` を取り出す boilerplate が毎回必要
- pair (even/odd) 化すると `q0Vars` `q1Vars` の 2 並列管理になる

**Step 9 提案 (S9c)**: `RegArray ty n` 抽象を導入。

```lean
structure RegArray (ty : WGSLType) (n : Nat) where
  names : Array String  -- length = n
  hasN : names.size = n
def RegArray.mk (ty : WGSLType) (n : Nat)
    (init : Nat → Exp ty) : ShaderM (RegArray ty n) := do
  let names ← (List.range n).mapM fun i => ShaderM.var ty (init i)
  return ⟨names.toArray, by simp [List.length_range]⟩
def RegArray.get (a : RegArray ty n) (i : Nat) : Exp ty :=
  Exp.var a.names[i]!
def RegArray.set (a : RegArray ty n) (i : Nat) (v : Exp ty) : ShaderM Unit :=
  ShaderM.assign a.names[i]! v
```

V11 の宣言部は

```lean
let q0 ← RegArray.mk (.scalar .f32) dPerLanePair fun pk =>
  Exp.mul q0Vals[pk] (Exp.litF32 scale)
let vkq0 ← RegArray.mk (.scalar .f32) dPerLanePair (fun _ => Exp.litF32 0)
```

参照は `q0.get pk` / `vkq0.set pk newVal`。Array String の手動管理 ×4 が消える。

### 2.4 smem alias re-use

llama.cpp 117-150:

```cpp
__shared__ half KQ[ne_KQ > ne_combine ? ne_KQ : ne_combine];
// Phase 1 で:
int    * tmp_q_i32 = (int    *) &KQ[j*D];     // KQ smem を Q quantize tmp として再利用
float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));
// 後で:
KQ[j*nthreads + tid] = KQ_reg[j];              // 同じ smem を本来の用途で使う
```

**hesper にはこの alias 再利用パターンを表現する syntax がない**。
V11 は `shared_kq` `shared_vkq` `shared_warp_meta` の 3 つを別々に declare し、
最大用途分の memory を予約する。

**Step 9 提案 (S9d)**: `ShaderM.sharedAliased` で「同じ memory を異なる
寿命の 2 つ以上の view として使う」を declarative に書ける構文を提供。

```lean
let smemBytes := max (ne_KQ * 4) (ne_combine * 4)
let smemRaw ← ShaderM.sharedRawBytes "kq_combine_smem" smemBytes
let kqAsScore := smemRaw.viewAs (.array (.scalar .f32) ne_KQ)
let kqAsCombine := smemRaw.viewAs (.array (.scalar .f32) ne_combine)
```

ただしこの抽象は重く、kernel が小さいうちは hesper 流の「分離 declare」で
まあいい。**優先度は中**。

### 2.5 softmax-online update が 7-8 行に散らばる

CUDA:

```cpp
const float KQ_max_scale = expf(KQ_max[j] - KQ_max_new[j]);
KQ_max[j] = KQ_max_new[j];
KQ_reg[j] = expf(KQ_reg[j] - KQ_max[j]);
KQ_sum[j] = KQ_sum[j]*KQ_max_scale + KQ_reg[j];
```

hesper V11 1684-1701:

```lean
let kqMaxScaleVar ← ShaderM.var (.scalar .f32) (Exp.exp (Exp.sub kqMax kqMaxNew))
let kqMaxScale : Exp (.scalar .f32) := Exp.var kqMaxScaleVar
ShaderM.assign "kq_max" kqMaxNew
let myKPos ← ShaderM.let' (.scalar .u32) ...
let myInBounds := Exp.lt myKPos splitEnd
let kqRegExp := Exp.exp (Exp.sub (Exp.var kqRegVar) kqMaxNew)
let kqRegGated := Exp.select myInBounds kqRegExp (Exp.litF32 0.0)
ShaderM.assign kqRegVar kqRegGated
ShaderM.assign "kq_sum" (Exp.add (Exp.mul kqSum kqMaxScale) (Exp.var kqRegVar))
```

CUDA 4 行 ↔ hesper 9 行。差分は:
- gating (`myInBounds`) — CUDA は K_max の sentinel 値 `-FLT_MAX/2` で吸収
- name の重複 (`kqMaxScaleVar` vs `Exp.var kqMaxScaleVar`)

**Step 9 提案 (S9e)**: 高水準 helper

```lean
def SoftmaxOnline.update
  (max sum reg : MutVar (.scalar .f32)) (newMax : Exp (.scalar .f32))
  (gate : Exp (.scalar .bool)) : ShaderM Unit
```

を提供すれば 9 行 → 1 行になる。ただしこれは **DSL の domain knowledge を
DSL に焼き込む** 方向で、CUDA の autonomy (各行を見たい) とは反対方向の改善。
**汎用性は低い**ので最後に検討。

---

## 3. ギャップに「乗らない」ところ

### 3.1 sentinel 値

CUDA: `-FLT_MAX/2.0f` (= -inf 近似) で OOB を gating
hesper: `Exp.litF32 (-1.0e30)` を毎回手書き

これは ergonomics 問題ではなく **慣用句の自前定義の有無**:

```lean
def Exp.negInfHalf : Exp (.scalar .f32) := Exp.litF32 (-3.4e37)
def Exp.f32Zero : Exp (.scalar .f32) := Exp.litF32 0.0
```

を `Hesper/WGSL/Exp/Sentinels.lean` に集めるだけで V11 の認知負荷は下がる。
**Step 9 候補に入れる価値はあるが、優先度は低い**。

### 3.2 `__syncwarp()` vs block barrier

CUDA: `__syncwarp()` (warp 内 31 lane を同期、block-level barrier より軽い)
hesper: `ShaderM.barrier` のみ。警告を緩和した版がない。

PTX 直叩きすれば `bar.warp.sync` を出せるはずなので、**Step 9 候補**:

```lean
def ShaderM.warpBarrier : ShaderM Unit
```

WGSL backend では block barrier フォールバック、PTX backend では `bar.warp.sync`。
V11 では Phase 1→2a の間に warpBarrier を入れる場所があるはず。

### 3.3 配列レジスタの half2 packing

llama.cpp `V_DOT2_F32_F16_AVAILABLE` 分岐で `half2 VKQ[ncols][D/2/nthreads_V]`
を使う。1 reg に 2 dim 詰む。hesper は f32 で 1 reg = 1 dim。
**これは ergonomics ではなく PTX backend の packing primitive 問題** で、
S9 で扱う話ではない (Session #266 で `Exp.fmaF16x2` 導入済 → kernel level の自前 wiring が必要)。

---

## 4. 提案 Step 9: 優先度順 (完了状況込み)

| # | 名前 | 効果範囲 | 状態 | 実装サイズ | ROI |
|---|---|---|---|---|---|
| **S9a** | `unrollForScoped` | V8/V9/V11 全部 | ✅ 完了 | 5 行 | **高** |
| **S9b** | `MutPtr` (ポインタ進行) | outer K loop, weight 移動系 | ✅ 完了 | 50 行 | **中** |
| **S9c** | `RegArray ty n` | V8/V11 の `q0Vars` 系 | ✅ 完了 | 80 行 | **高** |
| S9d | smem alias view | flash-attn-vec 系のみ | 未着手 | 150 行 | 低 |
| S9e | `softmaxOnlineUpdate` | flash-attn-vec 系のみ | ✅ 完了 | 25 行 | 低 |
| S9f | `Exp.negInfHalf` 定数集 | 全部 | ✅ 完了 | 20 行 | 低 |
| S9g | `ShaderM.warpBarrier` | flash-attn 系 | ✅ 完了 | 30 行 | 中 |

**最初に着手すべきは S9c (RegArray) と S9a (unrollForScoped)**。
両方とも V11 のコード量を 30% 削れて、認知ギャップの「数の暴力」(同じ
パターンを 4-8 回手書きする) が解消される。

S9b (MutPtr) は CUDA `K += stride` への寄せだが、hesper の現 V11 は絶対
index 計算がそれなりに動いていて、コード量よりは「概念の理解」の話。
S9c/S9a の後に余裕があれば。

## 5. 数値での見積もり

V11 (1513-1853 = **341 行**) を Step 9a+c 適用後に re-port した場合の
予想行数:

| Phase | 現在 | S9a+c 後 | 削減 |
|---|---|---|---|
| 宣言 (Q_reg, VKQ accum) 1565-1583 | 19 行 | 6 行 | -13 |
| Phase 1 inner pk loop 1626-1643 | 18 行 | 8 行 | -10 |
| warp reduce 1645-1657 | 13 行 | 1 行 | -12 |
| Phase 2b 1684-1707 | 24 行 | 14 行 | -10 |
| Phase 3 inner pk loop 1743-1756 | 14 行 | 6 行 | -8 |
| 末尾 acc weight loop 1821-1844 | 24 行 | 12 行 | -12 |
| **合計** | **112** | **47** | **-65** |

行数で 60% 削減。**かつ V11 の parity bug の原因に近づける可能性がある**
(配列 index 計算の typo は手動 Array String 管理が温床)。

---

## 6. このドキュメントの位置づけ

Step 1-8 は「行レベル ergonomics」を解決した。
**残るギャップは「集約レベル抽象」(配列レジスタ / ポインタ進行 / scope) と
「DSL に domain knowledge を焼くか」(softmax-online helper) の 2 軸**。

前者 (S9a/c) は他の kernel にも転用効くので価値が高い。後者は kernel 個別
の話なので保留。
