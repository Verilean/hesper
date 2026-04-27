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
| 比較 | `x < y` | `Exp.lt x y` | `x <ᵉ y` (← TODO) |
| 等号 | `x == y` | `Exp.eq x y` | `x =ᵉ y` (← TODO) |
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
| `__syncwarp()` | `__syncwarp();` | (なし — block barrier で代用) | `ShaderM.warpBarrier` (← TODO Step 6) |
| `__shfl_xor_sync(.., off, 32)` | `__shfl_xor_sync(0xFFFFFFFF, x, off, 32)` | `Exp.subgroupShuffleXor x off` | 同じ |
| `warp_reduce_sum<32>` | `__reduce_add_sync(...)` or 5-shfl loop | `Exp.subgroupAdd x` | ✅ `← ShaderM.warpReduceSum 32 x` (Step 5) |
| `warp_reduce_sum<8>` | 3-shfl xor 1,2,4 ループ | 手書き 3 行 | ✅ `← ShaderM.warpReduceSum 8 x` (Step 5) |
| `warp_reduce_max<n>` | shfl-xor max butterfly | 手書き | ✅ `← ShaderM.warpReduceMax n x` (Step 5) |
| `ld.global.u32` | `*K_ptr` (after offset) | `← ShaderM.readBuffer ... idx` | 同じ |
| **vec4 load** | `ggml_cuda_memcpy_1<16>(&dst, &K)` | (まだなし — `ld_v4_u32` PTX infra のみ) | `← ShaderM.readBufferVec4 ...` (← TODO) |
| `__half2half2(x)` | `__half2half2(x)` | (手書き or pack2x16float) | `Exp.broadcastH2 x` (← TODO) |
| `fma.rn.f16x2` | `__hfma2(a, b, c)` | ✅ `Exp.fmaF16x2 a b c` | 同じ |
| **f32 FMA** | `fmaf(a, b, c)` | (なし — mul + add 別命令) | `Exp.fma a b c` (← TODO) |
| ポインタ進める | `K += stride;` | (なし — index 計算で代用) | `ptr.advance stride` (← TODO) |

## 2. 完了したステップ (Step 1-3)

### Step 1: Operator overloading on Exp (commit c82c81b)
- `HAdd/HSub/HMul/HDiv/HMod/AndOp/OrOp/HShiftLeft/HShiftRight/OfNat`
- V9 inner loop ported, parity preserved
- **caveat**: `Exp.var name + x` は `(Exp.var name : Exp _)` の型注釈が必要

### Step 2: Typed mutable variables (`MutVar`)
- `structure MutVar (ty : WGSLType)` + `read` / `write` / `addAssign` / `mulAssign` / `subAssign`
- 演算子: `v +↦ x`, `v *↦ x`, `v ↦= x`
- **これで Step 1 の caveat 解消**: `acc +↦ q * k` で型注釈不要

### Step 3: `unrollFor` / `runtimeFor`
- `unrollFor n body`: meta-time, `body : Nat → ShaderM Unit`, n 個 inline 展開
- `runtimeFor start end_ step body`: runtime, `body : Exp (.scalar .u32) → ShaderM Unit`
- 既存の `for i in [0:n]` と `ShaderM.loop` も使える、新名前で意図を明示

## 3. 残り TODO (優先度順)

### 高 ROI (Step 4-7 候補)

**Step 4: Lane / warp 慣用句**
- `← ShaderM.tidX` `← ShaderM.bidX` (vec3X workgroupId/localId のヘルパ)
- `← ShaderM.laneId` (`tid & 31`)
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
