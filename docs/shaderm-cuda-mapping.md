# ShaderM ⇔ CUDA C++ 構文対応表 + 認知ギャップ縮小ロードマップ

llama.cpp などの CUDA C++ カーネルを ShaderM に port するときの対応表と、
認知ギャップを縮める DSL 改善 TODO。

## 1. 対応表 (現状: 2026-04-27 段階)

| 用途 | CUDA C++ | ShaderM (現状) | ShaderM (目標) |
|---|---|---|---|
| 算術 | `q + k * scale` | ✅ `q + k * scale` (Step 1) | 同じ |
| u32 リテラル | `(uint32_t)5` | ✅ `(5 : Exp _)` (OfNat) | 同じ |
| ビット | `x & 0xFF`, `x \| y` | ✅ `x &&& 0xFF`, `x ||| y` | 同じ |
| シフト | `x >> 3` | ✅ `x >>> 3` | 同じ |
| 比較 | `x < y` | ✅ `x <ᵉ y` (Step 8) | 同じ |
| 等号 | `x == y` | ✅ `x ==ᵉ y` (Step 8) | 同じ |
| 三項 | `c ? a : b` | `Exp.select c a b` | 同じ (Lean は ternary なし) |
| `if (cond)` | `if (...) { ... }` | `ShaderM.if_ cond then_ else_` | `whenE cond do ...` (← TODO) |
| 変数宣言 (mut) | `float acc = 0;` | `let acc ← ShaderM.var (.scalar .f32) 0` | ✅ `let acc ← ShaderM.mutVar (.scalar .f32) 0` (Step 2) |
| 変数読み | `acc` | `Exp.var name : Exp (.scalar .f32)` | ✅ `acc.read` (Step 2) |
| 変数書き | `acc = x` | `ShaderM.assign name x` | ✅ `acc ↦= x` (Step 2) |
| 累積 (+=) | `acc += x` | `ShaderM.assign acc (Exp.var acc + x)` | ✅ `acc +↦ x` (Step 2) |
| 累積 (*=) | `acc *= x` | (手書き) | ✅ `acc *↦ x` (Step 2) |
| **unroll for** | `#pragma unroll for (int i=0;i<8;++i)` | `for i in [0:8] do` | ✅ `ShaderM.unrollFor 8 fun i => ...` (Step 3) |
| **runtime for** | `for (int i=0; i<n; ++i)` | `ShaderM.loop 0 n 1 fun i => ...` | ✅ `ShaderM.runtimeFor 0 n 1 fun i => ...` (Step 3) |
| `threadIdx.x` | `threadIdx.x` | `← ShaderM.localId` の `.x` | ✅ `← ShaderM.tidX` (Step 4) |
| `blockIdx.x` | `blockIdx.x` | `← ShaderM.workgroupId` の `.x` | ✅ `← ShaderM.bidX` (Step 4) |
| **lane index** | `threadIdx.x & 31` | `tid & 31` (手書き) | ✅ `← ShaderM.laneId` (Step 4) |
| **warp index** | `threadIdx.x >> 5` | `tid >>> 5` (手書き) | ✅ `← ShaderM.warpId` (Step 4) |
| sub-warp split | `(tid & ~7, tid & 7)` | 手書き | ✅ `← ShaderM.subWarpSplit 8` (Step 4) |
| `__syncthreads()` | `__syncthreads();` | ✅ `ShaderM.barrier` | 同じ |
| `__syncwarp()` | `__syncwarp();` | ✅ `ShaderM.warpBarrier` (Step 9g) | 同じ |
| `__shfl_xor_sync(.., off, 32)` | `__shfl_xor_sync(0xFFFFFFFF, x, off, 32)` | `Exp.subgroupShuffleXor x off` | 同じ |
| `warp_reduce_sum<32>` | `__reduce_add_sync(...)` or 5-shfl loop | `Exp.subgroupAdd x` | ✅ `← ShaderM.warpReduceSum 32 x` (Step 5) |
| `warp_reduce_sum<8>` | 3-shfl xor 1,2,4 ループ | 手書き 3 行 | ✅ `← ShaderM.warpReduceSum 8 x` (Step 5) |
| `warp_reduce_max<n>` | shfl-xor max butterfly | 手書き | ✅ `← ShaderM.warpReduceMax n x` (Step 5) |
| `ld.global.u32` | `*K_ptr` (after offset) | `← ShaderM.readBuffer ... idx` | 同じ |
| **vec4 load** | `ggml_cuda_memcpy_1<16>(&dst, &K)` | (まだなし — `ld_v4_u32` PTX infra のみ) | `← ShaderM.readBufferVec4 ...` (← TODO) |
| `__half2half2(x)` | `__half2half2(x)` | (手書き or pack2x16float) | `Exp.broadcastH2 x` (← TODO) |
| `fma.rn.f16x2` | `__hfma2(a, b, c)` | ✅ `Exp.fmaF16x2 a b c` | 同じ |
| **f32 FMA** | `fmaf(a, b, c)` | (なし — mul + add 別命令) | `Exp.fma a b c` (← TODO) |
| ポインタ進める | `K += stride;` | ✅ `MutPtr.advance stride` (Step 9b) | 同じ |

## 2. 完了したステップ (Step 1-7)

### Step 1: Operator overloading on Exp (commit c82c81b)
- `HAdd/HSub/HMul/HDiv/HMod/AndOp/OrOp/HShiftLeft/HShiftRight/OfNat`
- V9 inner loop ported, parity preserved
- **caveat**: `Exp.var name + x` は `(Exp.var name : Exp _)` の型注釈が必要

### Step 2: Typed mutable variables (`MutVar`) (commit e00d222)
- `structure MutVar (ty : WGSLType)` + `read` / `write` / `addAssign` / `mulAssign` / `subAssign`
- 演算子: `v +↦ x`, `v *↦ x`, `v ↦= x`
- **これで Step 1 の caveat 解消**: `acc +↦ q * k` で型注釈不要

### Step 3: `unrollFor` / `runtimeFor` (commit e00d222)
- `unrollFor n body`: meta-time, `body : Nat → ShaderM Unit`, n 個 inline 展開
- `runtimeFor start end_ step body`: runtime, `body : Exp (.scalar .u32) → ShaderM Unit`
- 既存の `for i in [0:n]` と `ShaderM.loop` も使える、新名前で意図を明示

### Step 4: Lane / warp helpers (commit 4b648fe)
- `ShaderM.tidX` / `tidY` / `tidZ`, `bidX` / `bidY` / `bidZ`
- `ShaderM.laneId` (`tid & 31`), `ShaderM.warpId` (`tid >>> 5`)
- `ShaderM.subWarpSplit n` returns `(subWarp, subLane)`

### Step 5: Warp reductions (commit 53c06ac)
- `ShaderM.warpReduceSum n x`, `ShaderM.warpReduceMax n x`
- Butterfly via `subgroupShuffleXor` for n ∈ {2, 4, 8, 16, 32}

### Step 6: `Ptr ty` abstraction (commit 7b1a3e7)
- `structure Ptr (ty : WGSLType)` + `load` / `store` / `advance` / `atOffset`
- `ShaderM.ptr ty buf bufLen offset` — material once, advance per use
- llama.cpp の `K += stride; *K[k]` パターンに対応

### Step 7: Loop Invariant Code Motion (LICM) — opt-in
- `ShaderM.loopWithLICM` automatically hoists `varDecl`s whose init expression is loop-invariant
- 実装: `Exp.toWGSL` でシリアライズして名前ベースで free-var 検査
- 安全条件: 同名 reassign / loop-var 依存 / 内部 mutation がある場合は hoist しない
- 既定の `ShaderM.loop` は変更なし — LICM は意図して呼ぶときのみ
- 4-test ユニット (`Tests/ShaderMonadTests.lean`): hoist invariant / keep loop-variant / keep reassigned / default loop unchanged

### Step 8: 比較演算子 `<ᵉ` `==ᵉ` 他
- `<ᵉ` `≤ᵉ` `>ᵉ` `≥ᵉ` `==ᵉ` `!=ᵉ` を `Exp.lt/le/gt/ge/eq/ne` の infix として定義
- Lean 既定の `<` / `==` は `Bool` を返すため再利用不可 → unicode 接尾辞 `ᵉ` で区別
- V9 の `Exp.lt kPos splitEnd` → `kPos <ᵉ splitEnd` で `Exp.select` 句が CUDA に近づく

### Step 9a: `unrollForScoped` (集約版 unroll)
- `for ... in [0:n] do ShaderM.scope do` を `ShaderM.unrollForScoped n` 1 行に
- meta 展開＋per-iter register scope を 1 つのコンセプトとして扱える
- CUDA `#pragma unroll for ...` に直接対応

### Step 9b: `MutPtr` (mutable pointer advance)
- `Ptr` の `offset` フィールドを mutable u32 register 化、`advance` で更新可
- CUDA `K += stride; *K[k]` 慣用句に対応 (outer loop で進行 / inner で `loadAt`)
- `Ptr.atOffset` (純粋オフセット) と組み合わせて 2 段ポインタ操作が書ける

### Step 9c: `RegArray ty n` (typed register array)
- `Array String` + `Exp.var` の手動管理を撲滅
- `arr.get pk` / `arr.set pk v` で typed access
- `ShaderM.regArray ty n init` で n 個の `var` を一括宣言

### Step 9f: 標準 sentinel 定数
- `Exp.negInf30` (= `-1.0e30`、hesper 既定) / `Exp.negInfHalf` (= `-FLT_MAX/2`、llama.cpp 互換)
- `Exp.f32Zero`, `Exp.f32One`, `Exp.u32Zero`, `Exp.u32One`
- 重複リテラルの撲滅、意図 (out-of-bounds gating か単なる zero か) が明示される

### Step 9g: `ShaderM.warpBarrier`
- CUDA `__syncwarp()` 相当
- PTX backend: `bar.warp.sync 0xFFFFFFFF` (block barrier より大幅に軽い)
- WGSL backend: `workgroupBarrier()` フォールバック (仕方なく block barrier)
- flash-attn vec の Phase 1→2a 間で使用

### Step 9e: `softmaxOnlineUpdate` (online softmax helper)
- `softmaxOnlineUpdate kqMaxName kqSumName kqScore` で 1 行
- 7 行の Milakov & Gimelshein online softmax (max更新 + scale計算 + exp + sum更新) を内部に隠蔽
- 戻り値 `(kqMaxNew, scale, kqExp)` で VKQ 累積側の rescale が書ける
- domain-specific だが flash-attn 系全部で同パターンなので価値あり

## 3. 残り TODO (優先度順)

### 高 ROI
- `← ShaderM.warpId` (`tid >>> 5`)
- `← ShaderM.subWarpSplit 8` returns `(subWarp, subLane)` という pair
- 効果: V11 のような sub-warp partition kernel を書くときの index 計算ミスを排除

**Step 5: 比較演算子 `<ᵉ` `=ᵉ` `≤ᵉ`**
- `Exp.lt`, `Exp.eq` を演算子化 (Bool との衝突を避けるため `<ᵉ` のような unicode)
- もしくは `Decidable` インスタンスで `==` を直接使う
- `cond` 句が CUDA に近づく

**Step 6: 高水準プリミティブ**
- `ShaderM.warpReduceSum (n := 8) x` — 3-shfl もしくは subgroupAdd を選択
- `ShaderM.warpReduceMax (n := 32) x`
- `ShaderM.broadcastH2 x` — half2 broadcast
- `ShaderM.readBufferVec4 ...` — `Inst.ld_v4_u32` を front-end で使えるように
- 効果: V11 の Phase 1 実装の半分は warp_reduce で書ける

**Step 7: `Ptr ty` 抽象 (ポインタ進め)**
- `structure Ptr (ty : WGSLType) where buf : String; offset : Exp (.scalar .u32)`
- `Ptr.load`, `Ptr.advance`, `Ptr.storeAt`
- llama.cpp の `K += stride; *K` パターンが直接書ける

### 中期 (Step 8+)

- **マクロ `share% v := e`**: 複数参照される `let` を自動的に `let'` 化
- **比較で出てきた warning fix**: `select` の Bool 引数 vs Exp Bool の混同
- **`ShaderM.warpBarrier`**: PTX backend では `bar.warp.sync`、WGSL ではフォールバックで block barrier
- **Lane の暗黙化**: ShaderM 自体に lane context を入れて `tid` 自動アクセス (やりすぎかも、要検討)

### 長期: アルゴリズム DSL

- `Attention.kqDotProduct ...` など高水準ビルディングブロック
- llama.cpp 的に「sub-warp partition」を選択肢として宣言、低レベル実装は backend 選択
- 多分 hesper の Circuit DSL v2 と統合する話

## 4. 認知ギャップ縮小の効果測定 (V9 inner loop の例)

### Before all steps (元コード)

```lean
ShaderM.loop (Exp.litU32 0) (Exp.litU32 32) (Exp.litU32 1) fun iKQ0 => do
  let iKQ := Exp.add (Exp.mul warpId (Exp.litU32 32)) iKQ0
  let kPos := Exp.add kVKQ0 iKQ
  let inBoundsU32 ← ShaderM.let' (.scalar .u32)
                      (Exp.select (Exp.lt kPos splitEnd) (Exp.litU32 1) (Exp.litU32 0))
  let kRowBaseU32 ← ShaderM.let' (.scalar .u32)
                      (Exp.add kHeadBaseU32 (Exp.mul kPosSafe kRowStrideU32))
  ...
  ShaderM.assign partialVar
    (Exp.add (Exp.var partialVar) (Exp.mul q0Exp k0))
```

### After Step 1 (operators)

```lean
ShaderM.loop 0 32 1 fun iKQ0 => do
  let iKQ : Exp (.scalar .u32) := warpId * 32 + iKQ0
  let kPos := kVKQ0 + iKQ
  let inBoundsU32 ← ShaderM.let' (.scalar .u32)
                      (Exp.select (Exp.lt kPos splitEnd) 1 0)
  let kRowBaseU32 ← ShaderM.let' (.scalar .u32)
                      (kHeadBaseU32 + kPosSafe * kRowStrideU32)
  ...
  let p : Exp (.scalar .f32) := Exp.var partialVar  -- まだ型注釈必要
  ShaderM.assign partialVar (p + q0Exp * k0)
```

### After Step 2 (MutVar, hypothetical port)

```lean
ShaderM.loop 0 32 1 fun iKQ0 => do
  let iKQ : Exp (.scalar .u32) := warpId * 32 + iKQ0
  let kPos := kVKQ0 + iKQ
  let inBoundsU32 ← ShaderM.let' (.scalar .u32)
                      (Exp.select (Exp.lt kPos splitEnd) 1 0)
  let kRowBaseU32 ← ShaderM.let' (.scalar .u32)
                      (kHeadBaseU32 + kPosSafe * kRowStrideU32)
  ...
  partialVar +↦ q0Exp * k0     -- partialVar : MutVar (.scalar .f32)
```

### After Step 3 + Step 4-6 (hypothetical)

```lean
ShaderM.runtimeFor 0 32 1 fun iKQ0 => do
  let iKQ := warp * 32 + iKQ0  -- warp ← ShaderM.warpId
  let kPos := kVKQ0 + iKQ
  let inBounds ← ShaderM.let' _ (kPos <ᵉ splitEnd)  -- 比較演算子
  let kPosSafe := kPos.maskBy inBounds 0
  let kRowBase ← ShaderM.let' _ (kHeadBase + kPosSafe * kRowStride)
  ...
  partialVar +↦ q0Exp * k0
  -- 全体的に CUDA C++ と 1:1 で対応
```

## 5. なぜここまでの距離が必要だったか

ShaderM は元々 WGSL を生成するための DSL だった。WGSL 自体は CUDA より制約的:
- `for` は runtime のみ (unroll は実装定義)
- ポインタなし、smem は array 経由
- subgroup は extension

だから DSL は CUDA より WGSL 寄りに設計された。CUDA backend を後付けしたとき、
**意味論的には CUDA で書ける** のに **記法的には WGSL のまま** だった。
このギャップが port のミスを増やしていた。

Step 1-3 は **Lean の機能 (typeclass, structure, naming) で記法を CUDA 寄りに**
寄せた。Step 4+ も同方向で進める。
