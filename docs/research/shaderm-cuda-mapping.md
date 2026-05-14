# ShaderM ⇔ CUDA C++ 構文対応表 + 認知ギャップ縮小ロードマップ

llama.cpp などの CUDA C++ カーネルを ShaderM に port するときの対応表と、
認知ギャップを縮める DSL 改善 TODO。

> **Last refresh:** 2026-04-29 (after Q4_K MMQ Phase 1 port attempt).
>
> **TL;DR for newcomers porting CUDA → ShaderM:**
> 1. Read Section 1 (mapping table) first — it covers 95% of CUDA constructs.
> 2. **Before claiming a feature is missing, grep `Hesper/WGSL/Monad.lean`.**
>    Most `← TODO` lines below are *stale*; features were added but the doc
>    drifted. Real status verified 2026-04-29.
> 3. For complex multi-step kernels (e.g. MMQ), draw the K-loop and indexing
>    on paper *before* writing ShaderM. The DSL faithfully transcribes what
>    you write — it doesn't catch missing iteration ranges or off-by-N
>    sub-block bugs (Section 6 has a worked example).

## 1. 対応表 (現状: 2026-04-29 段階)

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
| **vec4 (4×u32) load** | `*reinterpret_cast<int4*>(&p[i])` | ✅ `← ShaderM.readBufferU32x4 buf base` returns `(u32, u32, u32, u32)` tuple | 同じ |
| `__half2half2(x)` | `__half2half2(x)` | (手書き or `pack2x16float`) | `Exp.broadcastH2 x` (← still TODO) |
| `fma.rn.f16x2` | `__hfma2(a, b, c)` | ✅ `Exp.fmaF16x2 a b c` | 同じ |
| **f32 FMA** | `fmaf(a, b, c)` | ✅ `Exp.fma a b c` (Exp.lean:156) | 同じ |
| ポインタ進める | `K += stride;` | ✅ `MutPtr.advance stride` (Step 9b) | 同じ |
| **per-thread reg array** | `float tmp[N] = {0}` | ✅ `← ShaderM.regArray ty N init` (Step 9c) — `tmp.get i`, `tmp.set i v` | 同じ |
| **smem array** | `__shared__ int x_qs[mmq_y * 33]` | ✅ `ShaderM.sharedNamed name (.array ty size)` | 同じ |
| `unpack_scales_q45_K` | inline u8 bit ops | (existing `extractScaleMin` helper — see Linear.lean:2343) | reusable, just inline |
| **block-cooperative load** | `for (i=0;i<N;i+=nwarps*ws) { ... }` | manual `unrollFor 4 fun loadIter => ...` | 既存の `unrollFor` で十分 |
| **multi-row × multi-col output tile** | `float tmp[ncols_dst][nrows_dst]` | `RegArray (.scalar .f32) (ncols * nrows)` + 2D index helper | 2D の薄いラッパが欲しい (Section 3) |

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

## 3. 残り TODO (refreshed 2026-04-29 after Q4_K MMQ port)

> **Note:** Steps 1–9g are all landed. Items previously marked `← TODO` in
> Section 1 that pointed at warpId/subWarpSplit/warpReduceSum/Ptr/RegArray
> are **all complete** — the doc was stale.

### 高 ROI (open)

**`Exp.broadcastH2 x`** — half2 broadcast of a single half scalar
- CUDA: `__half2half2(x)`
- Currently written manually with `pack2x16float`
- Shows up in flash attn V12 and any per-iter half2 widening
- 5–10 line addition in `Hesper/WGSL/Exp.lean` + lowering

**`BlockLayout` typed buffer view** — *new TODO from MMQ port*
- Problem: every kernel reads quantized buffers via raw `Exp.add base off` and
  comments like `-- ds_word at sub*9 + 0, qs at sub*9 + 1..8` that drift over
  time. MMQ Phase 1 had a parity bug *because the layout was wrong* and the
  type system didn't catch it.
- Sketch: a Lean structure `Q8_1View { col, sub : Exp u32 }` with methods
  `.dsWord`, `.qs (k : Nat)` that lowers to the right offset. Unifies
  knowledge of layout in one place. Other layouts (`Q4_K`, `block_q8_1_mmq`)
  get parallel views.
- ROI: prevents the class of bug we hit in MMQ Phase 1; quantized matmul
  parity becomes much faster to debug.

**`ShaderM.regArray2D ty (rows : Nat) (cols : Nat)`** — *new from MMQ port*
- 2D acc tile `float tmp[ncols_dst][nrows_dst]` shows up in MMQ, MMVQ-with-ncols,
  and TTT inner loops. Currently encoded with `RegArray ty (rows*cols)` plus a
  hand-written `idx (i, j) = i*cols + j` helper.
- Tiny wrapper over `RegArray` — `.get (i j : Nat)` / `.set (i j) v` —
  mostly for documentation and intent.

**`Exp.bufPtr "name" base`** — typed buffer pointer (read side of `MutPtr`)
- `MutPtr` exists but is mutable. Most use cases (matmul Y reads from inside
  a callee helper) want an *immutable* pointer with `.load (off : Nat)`.
  Closely related to `BlockLayout` above.

### 中期

- **`ShaderM.staticFor n body`**: compile-time fully-unrolled loop where the
  index becomes a `Nat` literal (vs `unrollFor` which is meta but still emits
  `Exp.litU32` per iter). For ports of `#pragma unroll for (int i=0;i<8;++i)`
  with constant `i` used in switch-like structures.
- **`whenE cond do ...`**: `ShaderM.if_` with no else, sugar (50% of `if_`
  call sites have `pure ()` else).
- **共通 quantized-matmul helper**: `Q4K.dotPair v u sc m dm ds8 → Exp f32`
  that wraps `vec_dot_q4_K_q8_1_impl_mmq` exactly. Hand-rolled in 4 places now;
  consolidating prevents drift.

### 長期: アルゴリズム DSL

- `Attention.kqDotProduct ...` など高水準ビルディングブロック
- `Matmul.qNK ncols_dst nrows_dst` — declarative matmul shape, backend picks
  MMVQ / MMQ / WMMA at lower level
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

## 6. Worked example: porting llama.cpp `mul_mat_q_kernel<Q4_K>` (MMQ)

This is the case study from the 2026-04-29 Phase 1 attempt
(`q4kMatmulBatchMMQKernel` in Linear.lean, see also
`memory/project_mmq_phase1_parity_blocker.md`). Outcome: the kernel
**compiled and ran 17.6× faster** than the 1-warp baseline but produced
NaN output. Root causes were not DSL gaps — they were design errors that
the DSL faithfully transcribed.

> **Source-of-truth links** (open these alongside the recipe in §6.3):
>
> - **DSL feature index** → §1 mapping table above. Grep `Hesper/WGSL/Monad.lean`
>   for the exact API.
> - **Existing kernels to copy from** (read these before writing a new one):
>   - 1-warp Q4_K dp4a baseline: `Hesper/Layers/Linear.lean::q4kMatmulBatchKernel`
>     (currently around line 1142). Reference algorithm; never break parity vs this.
>   - 4-warp coop K variant: `Hesper/Layers/Linear.lean::fusedQ4KMLinearDP4A4WarpKernel`.
>   - The shared `extractScaleMin` 6+6+4+2-bit Q4_K scale unpack appears verbatim
>     in both — search for `extractScaleMin` in Linear.lean and **copy it, don't reinvent**.
> - **Parity-test scaffold** → `Examples/DSL/Gemma4Q4KMMQParity.lean` and the
>   surrounding `Examples/DSL/Gemma4*Parity.lean` family. They all follow the
>   same load-real-GGUF-weights-then-diff pattern.
> - **Buffer-layout primer**:
>   - Q4_K block layout: `llama.cpp/ggml/src/ggml-common.h` (search `block_q4_K`).
>     32 q4 ints (= 128 bytes) per super-block, sub-blocks paired (k, k+1) via
>     low/high nibble of the same q4 ints.
>   - Q8_1 standard: `[ds_half2 (1 int) | qs (8 ints)]` per 32-K sub-block.
>     Hesper's `quantizeQ8_1BatchKernel` writes this layout. Used by hesper's
>     1-warp baseline.
>   - Q8_1 MMQ-packed: `block_q8_1_mmq` in `llama.cpp/ggml/src/ggml-cuda/mmq.cuh`
>     line 28. **Different from standard**: 4 ds at the front of each 128-K
>     group, then 32 qs ints contiguous. llama.cpp generates this via a
>     dedicated `quantize_mmq_q8_1` kernel; hesper has nothing equivalent yet.

### 6.1 What I got wrong

| Error | What happened | What would have caught it (DSL → §1 row) |
|---|---|---|
| Inner dot iterated `bq8Off ∈ {0,1}` only, so each output saw 2/8 sub-blocks | Output magnitude wildly off (~50M instead of ~0.5) | Explicit `static_for sub in 0..8` over Q4_K sub-blocks would have made coverage visible. See §1 "**unroll for**" row + §3 *staticFor* TODO. |
| Sub-block pairing for Q4_K nibbles assumed (k, k+4); actually it's (2g, 2g+1) sharing 8 q4 ints | Wrong-magnitude per-element output | Would *not* have been caught by DSL; needs reading `q4kMatmulBatchKernel` (Linear.lean) line-by-line. **Recipe step 5 below.** |
| Y tile read with offsets `{col*72 + sb*9 + 0..8}` (standard layout) but matmul expected MMQ-packed layout (4 ds at front, 32 qs after) | NaN at column 1+ from misaligned `ds` | `BlockLayout` typed view (§3 *DSL-BlockLayout* TODO) would have made layout choice explicit at the kernel signature. |
| `unpack_scales_q45_K` stubbed with `bitAnd sc_word 0xFF` (first byte) | Scale × 0..255 → magnitude ~10x too large per element | Reuse the existing `extractScaleMin` helper (search `extractScaleMin` in `Hesper/Layers/Linear.lean`) instead of inlining a stub. **Recipe step 5.** |

### 6.2 What the DSL handled correctly

| DSL feature used (§1 row) | Mirror of CUDA construct |
|---|---|
| `ShaderM.varNamed`, `Exp.var name` (also `MutVar` / Step 2) | `float acc = 0; ...; acc += x;` |
| `ShaderM.unrollFor 4 fun loadIter => ...` (§1 "unroll for", Step 3) | `#pragma unroll for (int i = 0; i < 4; ++i)` |
| `Exp.dot4I8Packed v u` | inline `__dp4a(v, u, 0)` |
| `Exp.unpack2x16float dmU32` | `__half22float2(dm)` |
| `ShaderM.sharedNamed "x_qs" (.array ...)` | `__shared__ int x_qs[N]` |
| `Exp.toF32`, `Exp.toF32U` | `(float)x` (signed) / `__uint2float_rn(x)` (unsigned) |
| `ShaderM.if_ cond ... ` (§1 `if_`) | `if (cond) { ... }` |

The "DSL was hard" feeling was 90% me missing what was already there
(`RegArray`, `Ptr`, `MutPtr`, `Exp.fma` — all present, see §1 mapping +
`Hesper/WGSL/Monad.lean`) and 10% real (no `BlockLayout`, no `staticFor`
with literal index — see §3 TODOs).

### 6.3 Recipe for next port

1. **Read llama.cpp source twice** before opening Linear.lean. Sketch the
   K-loop tree on paper:
   ```
   for kb0 in super_blocks:           ← outer (Q4_K = 256 K each)
     load X tile (32 ints into smem)
     for k01 in {0, 16}:              ← within a 32-int tile
       for nib in {0, 1}:             ← QR4_K
         for j in 0..7:               ← QI8_1
           dp4a(v[j] >> 4*nib, u[nib*8+j])
   ```
   Confirm the iteration count multiplies out to the right number of
   K-elements (here: super_blocks × 2 × 2 × 8 × 4 = K_total).

   *DSL pieces you'll need:* §1 "unroll for" / "runtime for" rows for the
   loops; §1 "shared array" + "per-thread reg array" rows for the smem /
   register tiles; §1 "lane index" / "warp index" rows for thread→tile
   index calculations.

2. **Pick the target Y layout up-front.** llama.cpp's MMVQ uses standard
   `block_q8_1`; MMQ uses `block_q8_1_mmq`. Decide which one your kernel
   reads and **comment it at the kernel signature**. If you change later,
   update the comment first.

   *Cross-ref:* See the layout primer at the top of §6 for the byte-level
   diff between the two Q8_1 variants. Once `BlockLayout` (§3) lands this
   choice will be in the type signature; until then, comment + assert.

3. **Write the parity test before the kernel body.** Use the pattern in
   `Examples/DSL/Gemma4Q4KMMQParity.lean`: load real GGUF weights, feed
   identical input through the new kernel and an existing reference
   kernel, diff. The test runs in <2 s once weights are cached → very
   tight feedback loop.

   *Test pattern:* `lake build gemma4-q4k-mmq-parity && lake exe ...`.
   Diff threshold: `1e-3` for "near bit-parity" (Q4_K dp4a is integer-
   accumulated so equal inputs *should* be bit-identical; floats only
   diverge through the dF/dminF f16 conversions).

4. **Bisect parity bottom-up:**
   - Start with kernel writing zeros → confirm grid/block geometry
     reaches every output (count writes per-output via atomic).
   - Add load X → confirm one row matches by reading back x_qs to host.
   - Add ONE sub-block dot → confirm magnitude ratio matches (expect
     1/8 of full output, then × 8).
   - Add full sub-block iteration → expect magnitude match.
   - Add scale extraction → expect bit-exact parity (since reference
     and candidate use the same `extractScaleMin`).

   *DSL pieces:* `ShaderM.if_`, `Exp.var`/`Exp.assign`, atomic helpers
   if you take the per-output-write-count approach.

5. **Reuse, don't reimplement.** `extractScaleMin` is already correct
   in `q4kMatmulBatch4WarpKernel` and the 1-warp variant. Copy-paste it
   into your new kernel verbatim before optimizing. The 6+6 bit pack
   layout is too easy to get wrong from scratch.

   *Find it:* grep `extractScaleMin` in `Hesper/Layers/Linear.lean`.
   Three sites currently — the inline-`let` form is most port-friendly.

6. **Compare emitted PTX against the reference kernel for one tiny
   shape** before benchmarking. `HESPER_PTX_DUMP=/tmp/ptx lake exe ...`
   writes per-kernel `.ptx` files. Diff the inner-loop region against the
   reference kernel's PTX; the K-loop trip count and `dp4a` instruction
   count should match within a small constant.

   *DSL pieces:* PTX dumping is built into `Hesper/Backend/CUDA.lean`
   (`HESPER_PTX_DUMP` env). See `docs/sass-q6k-comparison.md` for an
   example diff write-up.
