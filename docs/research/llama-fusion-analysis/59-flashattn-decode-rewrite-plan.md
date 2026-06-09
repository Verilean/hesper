# Decode flashAttention rewrite — match llama.cpp's vec kernel

2026-04-26.

## 計測 (graphs ON, decode)

| | hesper flshttnP_256_2 | llama.cpp flash_attn_ext_vec |
|---|---:|---:|
| call/tok | ~34 | ~34 |
| avg µs/call | **32.6** | **6.1** |
| ms/tok | **1.12** | **0.21** |
| ratio | **5.4×** | 1.0× |

decode の **GPU kernel time の 14% (1.12 / 7.92 ms)** がここに集中。閉じれば
+1 ms/tok = TPS 81.9 → ~92.

## 根本構造の違い

### llama.cpp `flash_attn_ext_vec<D=256, ncols=1, K=f16, V=f16>`

`llama.cpp/ggml/src/ggml-cuda/fattn-vec.cuh` および
`fattn-common.cuh:launch_fattn`:

- block_dim = `(warp_size=32, nwarps=4, 1)` ⇒ **2D thread block 32×4 = 128
  threads**
- gridX = **1 block per (head × tile)**, gridY = **nwaves of K positions
  (= ceil(cacheLen / nthreads_KQ))**
- `nthreads_KQ = 32` for D=256/4=64 (capped at 32)
- Q を shared memory に **1 回だけ** load
- 各反復で **複数 K 位置を warp 内 thread が並列に分担**:
  ```
  for (k_VKQ_0 = blockIdx.y*nthreads; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*nthreads)
      // 各 thread が異なる k 位置を担当
  ```
- 1 thread が 1 k 位置の `q·K[k]` を計算し、warp shuffle で reduce
- gridY による **K 軸 split** (split-K decode!)

### hesper `flashAttentionDynamicParamsKernel`

`Hesper/WGSL/FlashAttention.lean:233-321`:

- workgroupSize = **256 threads in 1D** (8 warps)
- gridX = `numHeads` ⇒ 1 workgroup per head
- gridY = 1 (split-K なし)
- 内側ループ: `for s in 0..cacheLen { ... }` — **1 K 位置 / 反復で逐次**
  - 各反復で 256 threads が `q·K[s]` の D=256 次元方向に並列 (256 thread = headDim dimension)
  - 内側 reduce + barrier × 2-3 回
- cacheLen=200 なら **200 回 barrier**
- split-K も batch-K もなし

## 並列度の差

llama.cpp:
- D=256, threads=128, ncols=1
- 1 dispatch あたり: `(headDim=256 / nthreads_KQ=32) = 8` 次元分担 × 32 lanes × 4 warps = **256 threads / 4 warps が異なる k 位置担当**
- 例えば cacheLen=200 → gridY ≈ 200/(nthreads_KQ_per_warp×nwarps) 〜 数 wave で完了
- **K 位置を warp 軸で並列化**

hesper:
- D=256, threads=256, K loop は逐次
- すべての thread が同じ k 位置を処理 (D 次元方向に分散)
- **K 位置を逐次に処理 → cacheLen 回ループ**

**並列性の違い**: llama.cpp は K 軸並列 (cacheLen 個を warp で分担)、hesper は
K 軸逐次 (1 個ずつ全 thread で処理)。Gemma 4 head dim=256 では D 次元並列
だけだと cacheLen×barrier 数が劇的に増える。

## 修正案

ref: 既に `executeFlashAttentionTiled` (split-K 版) が実装されている
が、prefill / large cacheLen 用で decode の small cacheLen では使われ
ていない。

### Option A: 既存 tiled kernel を全 cacheLen 範囲で使う

**検証 → 没**: dispatcher 見たら既に `cacheLen > 32 → tiled` を使ってお
り、measure された 32.6 µs/call はほぼ tiled の per-call time だった。
さらに tiled `flashAttentionTiledPhase1` の inner loop も **K position
を逐次 (tileSize=32 回ループ)** で各回 D 軸並列 + tree reduce + 2
barrier。tile **内部の並列化が llama.cpp と違う**。閾値変更だけでは速
くならない。Option B 必須。

### Option B: llama.cpp 風 vec kernel を新規実装

新 kernel `flashAttentionVecKernel` を `Hesper/WGSL/FlashAttention.lean`
に追加:

- gridX = numHeads, gridY = `ceil(cacheLen / Kpar)` (Kpar=32 など)
- workgroupSize = `(32, 4, 1)` = 4 warps  ※ ShaderM が 2D thread block
  をサポートしてれば
- 各 warp が異なる k 位置 chunk を担当
- warp 内 shuffle reduce で `q·K[k]` をまとめる
- 各 wave の partial output を gridY 軸で **softmax-aware combine** (split-K
  combine kernel が必要 — `executeFlashAttentionTiled` の combine 部を流用?)

ShaderM の制限を調査 → 2D thread block 可能なら option B、不可なら
option A から始める。

### Option C: K/V cache を f16 に下げる

llama.cpp は **K/V 共に f16 がデフォルト**
(`llama-context.cpp:2905-2906`: `type_k = GGML_TYPE_F16, type_v =
GGML_TYPE_F16`).  flash_attn_ext_vec の template も
`<D=256, ncols=1, K_type=GGML_TYPE_F16, V_type=GGML_TYPE_F16>` で確認。

hesper は **K/V cache を f32** で持っている (`Hesper/Models/Gemma4.lean`
の `kvCaches[].kBuf, vBuf` allocator サイズ参照).

flash attn は memory-bound なので f32 → f16 で **bandwidth が 2×** に
なる。Gemma 4 num_kv_heads × maxSeqLen × headDim = 1 × 4096 × 256 =
1 MB/cache-pair, 42 layer ⇒ KV cache 全体 84 MB → 42 MB に半減。decode
時の touch 量も半減なので flash attn の memory-bound セクション 2×
速くなる可能性。

**影響範囲**:
- KV cache buffer 確保サイズ (allocBuffer の size を半分に)
- KV cache write kernel (RoPE-K + scatter): output f32 → f16 (`cvt.f16.f32`
  挿入)
- flash attn kernel: K/V buffer の readBuffer 型を `.f16`、計算前に f32
  に拡張 (これも 1 命令)
- bit-parity: f16 で持つと精度 7 bit ロス → 厳密な bit-parity 不可。
  `cosine_similarity > 0.999` 程度の許容パリティで合格判定する必要。

**規模**: 中規模. ShaderM の f16 read/cvt サポートが必要 → 既存の Q6_K
embedding lookup などで `fp16ToF32` がある (Linear.lean) ので使い回し可。

**期待効果**: flash attn 1.12 → 0.5-0.7 ms/tok 程度 (memory-bound 部分が
2× 速く、reduce + softmax の計算は変わらず). +5 TPS 程度.

Option B と C は **直交** で組み合わせ可能 (両方やれば +10 TPS 期待).
B (algorithmic) のほうが効果大、C (bandwidth) のほうが実装単純。

## 期待効果

A or B が成功すれば: hesper flashAttn 1.12 ms/tok → 0.3-0.5 ms/tok 程度
(完全 parity でなくとも 2× は固い)。+5-7 TPS。

## 検証

1. 既存 `Tests/CUDA/MultiLayerQ4KTest.lean` 系の per-kernel
   bit-parity test がもし FlashAttn にあれば走らせる
2. 60 token decode で baseline (現行) vs new で:
   - kernel_compare_graphs.py で flashAttn の avg µs/call が変わるか
   - 全体 TPS の変化
3. `Hello! How can I help you today? 😊` 出力が一致するか

## 次にやること

1. ShaderM の 2D thread block サポート確認
2. `executeFlashAttentionTiled` を全範囲で使ってみる (Option A)
3. 効果がイマイチなら Option B
