# FlashAttention 高速化マルチセッション計画

2026-04-26 起案. doc 59 の Option B + C を複数セッションに分けて完遂する計画.

## ゴール

decode flashAttn kernel time を hesper 1.12 ms/tok → 0.21 ms/tok (llama.cpp parity).
graphs ON での全体 TPS を 81.9 → ≥ 95 (理論上限近く).

## 計測条件 (frozen)

```bash
HESPER_DP4A=1 HESPER_USE_MMAP=1 HESPER_IGNORE_EOS=1 HESPER_CUDA_GRAPHS=1 \
  lake exe gemma4-cuda data/gemma-4-e4b-it-Q4_K_M.gguf "Hello world how are you" 60
```
- 指標 1: TPS (tokens/sec)
- 指標 2: scripts/kernel_compare_graphs.py の **flashAttn 行 µs/call**
- 指標 3: bit/cosine parity vs baseline output
- 各セッション末尾で 3 指標を doc に追記

## セッション構成

### Session 1: Option B Phase 1 — vec kernel skeleton + dispatcher 切替

**目標**: llama.cpp 風の "1 thread per K position, warp-shuffle reduce" 構造を 1 つだけ実装し、HESPER_FA_VEC=1 で opt-in 可能にする. f32 KV のまま. bit-parity 確保.

**実装**:
1. ShaderM 側調査:
   - 2D thread block (block_dim.x=warp_size, block_dim.y=nwarps) サポート → ない場合 1D で 32 lane × 4 warp manual emulation
   - `subgroupAdd` (warp shuffle reduce) 使用箇所をサンプル抜粋
2. 新 kernel `flashAttentionVecKernel` を `Hesper/WGSL/FlashAttention.lean` に追加:
   - gridX = numHeads, gridY = 1 (split-K は Phase 2 で), gridZ = 1
   - workgroupSize = 128 (4 warps = 4 × 32)
   - 各 thread が異なる k position を分担して q·K[k] dot product (warp-内 D 軸並列で 32 thread shuffle reduce)
   - online softmax は warp-private のまま、最後に warp 間で combine
   - V·attn は同様
3. dispatcher gate: `if (← IO.getEnv "HESPER_FA_VEC").isSome` で新 kernel に切替, default は既存 tiled
4. bit-parity test (HESPER_FA_VEC=0 vs =1 で同じ token 列を出すか)
5. trace で flashAttn 行の µs/call を測る

**Definition of Done**: bit-parity ✓ + flashAttn µs/call が 32.6 → < 15 (2× 以上) になっていれば Phase 2 へ. それ未満なら kernel 内ボトルネックを ncu で見て調整.

### Session 2: Option B Phase 2 — split-K (gridY > 1) + combine kernel

**目標**: cacheLen が大きいほど gridY を増やして K 軸並列度を上げる. partial を combine kernel で集約.

**実装**:
1. flashAttentionVecKernel を gridY 軸対応に変更 (start = blockIdx.y * nthreads, stride = gridDim.y * nthreads)
2. partial buffer + combine kernel (既存 phase2 を流用 or 新規) で gridY 個の partial を softmax-aware merge
3. dispatcher で `gridY = ceil(cacheLen / 128)` を計算
4. trace で flashAttn 0.5 ms/tok 以下を狙う

**Definition of Done**: bit-parity ✓ + flashAttn ≤ 0.5 ms/tok + TPS 90+

### Session 3: Option C Phase 1 — K cache f16 化

**目標**: K cache を f32 → f16. V cache は f32 のまま (段階的移行で blast radius 制御).

**実装**:
1. `Gemma4KVCache` 構造体 / state allocation を f16 サイズ (×0.5) に. 互換性のため `kBufF16Mode : Bool` flag をつけて opt-in
2. RoPE-K + scatter kernel (現在 f32 出力) を f16 出力版で別途用意, dispatcher で切替
3. flash attn kernel (Vec, Tiled 両方) で `k_cache` を f16 として読み (`fp16ToF32` cvt 挿入)
4. KV cache の読み書きの整合性を再確認 (forwardPrefillBatch 経路含む)
5. **bit parity → cosine similarity > 0.999 へ判定基準変更**: テスト用に `HESPER_KV_F16=1` で 60 token 出してログを diff. 出力 token 列が **数 token 違っても correctness OK** とする (greedy decode の small numerical drift)
6. trace で flashAttn ms/tok の動きを記録

**Definition of Done**: cosine sim > 0.999 (or 出力 ≥ 95% 一致) + KV cache VRAM 半減 + flashAttn ms/tok 改善が観測

### Session 4: Option C Phase 2 — V cache f16 化

**目標**: V cache も f16 へ. flash attn kernel に対するメモリ帯域を 50% に.

**実装**:
1. forwardBlock の Vcur scatter (現在 f32) を f16 出力版に (RMSNorm-V の出力を直接 f16 へ)
2. flash attn kernel `v_cache` を f16 として load
3. forwardPrefillBatch 側も同様
4. 全テスト (mono layer parity, e2e decode)

**Definition of Done**: cosine sim > 0.999 + V cache VRAM 半減 + flashAttn が memory-bound 部分 2× の効果

### Session 5: 最終測定 + ドキュメント

1. baseline / Option B-only / Option C-only / Option B+C の 4 way 測定
2. nsys + kernel_compare_graphs.py で 1:1 比較
3. doc 59 / 60 に最終結果追記
4. memory `project_flashattn_decode_5x_gap.md` を CLOSED に更新
5. CLAUDE.md 風に summary を残す

## 段階的にコミットする方針

各 Session の Definition of Done を満たした時点で commit. opt-in env flag を残すので revert は不要 — 旧経路は env 未設定で残る. flashAttn 系の env: `HESPER_FA_VEC=1`, `HESPER_KV_F16=1`, `HESPER_KV_F16_V=1`.

## 失敗時の Rollback

- Session 1 で bit-parity 取れない → 中断, Phase 0 (1-warp 1-tile, full sequential) で完成度を下げて再構築
- Session 2 で gridY > 1 で精度 drift → split-K の combine 関数を再検証 (online softmax の rescale)
- Session 3-4 で cosine sim < 0.999 → cvt path のバグ; 1 layer のみ f16 化して bisect

## 各セッション開始時にやること

1. このドキュメント読む
2. baseline 測定 (上の frozen コマンド)
3. memory 関連メモを読む (`project_flashattn_decode_5x_gap.md`)
4. 当該 Session の "実装" を順に
5. Definition of Done 判定 + 数値追記

## このセッション (kickoff) でやること

Session 1 の Step 1 (ShaderM 側調査) のみ実施し doc に追記する.
それ以上は次セッション以降.

## kickoff session 結果 (2026-04-26)

### ShaderM 側調査結果

**1. 2D thread block サポート**: ✅ あり
- `Hesper/WGSL/Shader.lean:52`: `WorkgroupSize { x : Nat, y : Nat := 1, z : Nat := 1 }`
- `ExecConfig.workgroupSize := { x := 32, y := 4 }` で llama.cpp 風
  `block_dim(warp_size, nwarps)` 構造が表現可能
- `ShaderM.localId` は `Exp (.vec3 .u32)` を返すので `vec3X` (lane id within warp)
  と `vec3Y` (warp id within block) を独立に取れる

**2. subgroup primitives**: ✅ あり
- `Hesper/WGSL/Exp.lean:265`: `subgroupAdd : Exp t → Exp t` (warp-内 sum reduction)
- `Hesper/WGSL/Exp.lean:259-260`: `subgroupBroadcast`, `subgroupBroadcastFirst`
- 既存の RMSNorm warp-shuffle で実証済み (doc 49)
- needsSubgroups flag が `Hesper/WGSL/Monad.lean:48` にあって自動的に WGSL
  の `enable subgroups;` が emit される

**3. 既存 Q6_K dp4a kernel に warp-内 partial reduction が既にある**:
- `Hesper/Layers/Linear.lean` の fusedQ4K kernels に `subgroupAdd` 使用例
- 同パターンを flash-attn 用にコピー流用可

### Session 1 開始時の必要なもの (準備完了)

- ShaderM API: 2D wg + subgroupAdd (両方✅)
- 比較対象: `llama.cpp/ggml/src/ggml-cuda/fattn-vec.cuh` (template instantiation:
  `<D=256, ncols=1, K_type=GGML_TYPE_F16, V_type=GGML_TYPE_F16, false>`)
  ※ ただし f32 KV のまま実装. f16 化は Session 3-4.
- 検証ツール: `scripts/kernel_compare_graphs.py` (graphs ON 1:1 比較)
- 既存テスト: per-layer flash attn parity test がもしあれば bisect に使う

### Session 1 で書く新コード見積

- 新 kernel `flashAttentionVecKernelF32` (~150 lines, ShaderM)
- dispatcher 切り替え (HESPER_FA_VEC=1) (~10 lines)
- Lean 側 `executeFlashAttentionVecF32` 関数 (~30 lines)
- 計 200 行程度

### Session 1 進捗 (2026-04-26)

**WIP commit `0076815`**: skeleton landed but **NOT correct yet**.

- TPS regression: 82.1 → 70.1 (HESPER_FA_VEC=1) — algorithmic bug
- Output diverges from baseline ⇒ silent correctness regression
- Suspected root causes (still TBD, no parity test yet to bisect):
  - `shared_warp_sums` not zeroed between K iterations; cross-warp
    subgroupAdd may pull stale lanes >= numWarps
  - cross-warp broadcast: lane 0 of warp 0 produces canonical sum but
    we publish via `tid==0` write — confirm warp 0 of all heads agrees

### Lesson from Session 1: gemma4 TAT is too long for bisect

End-to-end gemma4 60-token decode is ~6 min per iteration (build 5 min
+ measure 1 min).  Bisecting kernel correctness against the legacy
kernel via gemma4 is impractical.

**Pivot**: build a **CUDA unit test** that compares
`flashAttentionVecParamsKernel` against `flashAttentionDynamicParamsKernel`
on the same Q/K/V/params inputs at multiple sizes (e.g. cacheLen ∈
{1, 8, 32, 64, 128}, headDim=256, numHeads=2 ish).  Existing pattern:
`Tests/CUDA/CUDAFlashAttnTest.lean` does this for sub-kernels.

DoD update: **bit-parity unit test passes first, then enable in gemma4**.
That keeps the iteration cycle <30s instead of 6min.

### Next concrete action (Session 1 continued)

1. Add a new test file `Tests/CUDA/CUDAFlashAttnVecParityTest.lean`
   that runs both kernels at small sizes and asserts max abs diff <
   1e-5
2. Run it to bisect — fix the warp_sums clear / cross-warp issue
3. Re-enable in gemma4 with confidence
