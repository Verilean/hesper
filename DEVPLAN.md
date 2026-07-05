# DEVPLAN — Hesper 再構築: 実測 JIT (autotune) + システム化開発

**これが開発を駆動する単一の生きたドキュメント。** 全セッションはここから始まりここで終わる:

```
セッションの循環（毎回）:
  開始: この DEVPLAN.md を読む → 担当モデルが合うマイルストーン(M)を選ぶ
  作業: §1 の原則ゲートに従う（sweep → golden → 統合 → eval）
  終了: §3 の状態と §4 の決定ログを更新 → コード + DEVPLAN.md を同一 commit
  ★関門: 成果物をユーザーがレビューしてから次の M へ
```

---

## §1. 開発原則（不変）

1. **予測せず、測る。** 性能はソースの静的性質ではない。仮説は sweep/実測で判定（ヤマカン one-off 実験の禁止）。
2. **speed + resource を常に両方記録。** ms/GFLOPS/%floor に加えて occupancy(maxThreads)/execWidth/tgMem を必須列に。省略禁止。
3. **golden bit-exact ゲート。** codegen/パラメータ変更で maxDiff が動いたらそれはバグ。variant は自動失格。
4. **単体勝ち ≠ 統合勝ち。** 採否は必ず e2e（batched decode 内 per-step、または decode tok/s）で判定。単体 bench だけで採用しない。
5. **測定衛生**: pgrep（decode/bench の stray）== 0、cool box、`kill -9` 厳禁（Metal が wedge する）。timeout/SIGTERM のみ。
6. **負の結果も記録。** 棄却した variant/仮説と根拠を §4 決定ログと tune log に残す。同じ実験を二度導出しない。
7. **リファレンス駆動。** 新 kernel は llama.cpp（refs/llama.cpp-diffusiongemma/）の構造を精読してから。tile 幅だけの模倣は失敗する（WIDE64 の実測教訓）— 構造ごと忠実に。
8. **fusion は 2 層探索。** 外側 = 分割/fusion（llama.cpp 一致表 + 節約 bytes ランキングでガイド）、内側 = パラメータ sweep。fusion が変われば内側を回し直す。

**確定済みの技術前提**（再調査しない）:
- Metal では WGSL/Tint と手書き Metal の決定差は小さい（let' 修正後 kernel 1.15×）。MSL hybrid 拡張は棄却済み（batch/bubble で回収不能）。
- Metal が回収する最適化（strength reduction、SROA、copy-prop）を WGSL 側で先回りしても null。回収しないのは大域の重複部分式 CSE（→ `ShaderM.let'`）。詳細 `WGSL_CODEGEN_OPTIMIZATION.md`。
- kernel は Lean 実行時に WGSL を生成する関数 → **パラメータ sweep に Lean 再ビルド不要**（1 プロセスで variant 生成→compile→実測）。
- pipeline cache は WGSL ソース hash キー → 100 variants 問題なし。workgroupSize は kernel 定義に焼かれる（dispatch パラメータではない）。
- **TAT 実測済み（M0, 2026-07-05, TATPROBE=1）**: **0.118 s/variant**（gen+compile ~80ms + bench100 ~40ms、deployed reg kernel @ QKV-KV 実 shape）→ **100 variants ≈ 12 秒、1000 ≈ 2 分**。律速は sweep でなく Lean 再ビルド（構造変更 ~1-3 分）と decode 統合（~8 分 + eval）。
- **sweep 基盤は runtime-K-loop 生成器必須**: Lean レベルで K を全展開する生成器（`matMulTransposeF16WMMA8x8RegKernel`）は K=2816 で 1 variant のコンパイルが 5 分超（Tint/Metal 爆発）→ sweep 不能。M2/M4 は deployed 型（`ShaderM.loop`）で作る。
- M2 golden の設計注意: 実 shape の CPU 参照 matmul は遅すぎる → golden は小 shape（64×32×64 級）で variant ごとに実施し、実 shape は timing のみ。

---

## §2. マイルストーン表

**モデル割当**: 設計/FFI/kernel 構造/根本原因究明 = **Fable**（GPU correctness はレース混入リスク高）。sweep 実行・記録・統合・eval の反復 = **安いモデル（Sonnet 級）**。迷ったら格上げし、理由を §4 に記録。

| # | 小ゴール | 成果物 | 合否基準 | モデル | レビュー |
|---|---|---|---|---|---|
| M0 | DEVPLAN 確定 + TAT 実測 | この文書 + TAT 表（実測値） | variant/秒・100 variants 壁時計が見積り±2×（見積り: ~2-3s/variant、100 で ≈5-15分） | Fable | **★**（TAT 許容可否） |
| M1 | occupancy probe + dump 捕獲 | bridge.cpp（g_last_dumped_msl + HESPER_DUMP_MSL_QUIET + FFI）、metal_replace.mm（`lean_hesper_msl_occupancy_probe`: newLibraryWithSource → maxTotalThreadsPerThreadgroup/threadExecutionWidth/staticThreadgroupMemoryLength）、stub、Device.lean externs | macOS で数値が出る + CI（非-Apple）緑 | Fable（FFI。StringView null 終端の前科あり） | —（自動ゲート） |
| M2 | sweep ランナー Tier 1 | MatmulBench `SWEEP=1`: grid（TM×TN×wg、`matMulTransposeF16WMMA8x8RegKernel` + 現行 64×32 baseline）→ 必須列記録 → tune log CSV 追記 → ランク表。golden 自動判定内蔵 | QKV-KV [262,1024,2816] / dense g/up [262,2112,2816] で全 variant golden 合格 + occupancy 列出力 | 設計: Fable / 実行: 低 | **★**（最初の sweep 表 = フロー機能の証拠） |
| M3 | リファレンス精読 | llama.cpp 構造選択表（mul_mm: 64×128, sg 2×2, BK, staging, load 幅, double-buffer / mul_mv 閾値 / flash_attn）+ **カーネル分割一致表**（1 layer の dispatch 列: ours vs theirs、どこで fuse しているか） | 表が揃い、Tier 2/3 軸と fusion 候補が導出済み | Fable | **★**（この表で kernel 方針を承認） |
| M4 | 汎用パラメータ化 kernel | `matMulTransposeF16WMMARegKernelGen(TMsg,TNsg,sgRows,sgCols,BK)` — 現行 64×32 の sg 配置/load 回数/K-loop 定数を式に置換（~40-60 行） | golden bit-exact 全 variant + sweep が現行 64×32 を再発見 or 超え | Fable（レース危険域） | sweep 結果で判定 |
| M5 | diffusion 統合 | 勝者 variant を env flag で DiffusionGemmaDecode に統合 | dg_eval 8/8 + per-step 改善（cool） | 中 | **★**（per-step 数字） |
| M6 | E2B bring-up | Gemma 4 E2B decode が正しく動く。資産: E4B generate path（Gemma4.generate + KV cache）、M=1 Q4_K/Q6_K kernels、PLE 実装済（Phase-4 gate）、KV 共有 config、chat template。ギャップ: loader E2B metadata (S) / PLE gate 解除 (S) / chat token 検証 (S) / dense-FFN 経路検証 (M)。見積り 1-2 日。**最初の一歩 = 既存 E4B 例の動作確認 + 現状 tok/s** | greedy が llama.cpp と一致 or 妥当テキスト | Fable（アーキ）/ 低（実行） | **★**（初の正しい decode + 現状 tok/s） |
| M7 | E2B 最適化 = フロー証明 | 「1 コマンド ≤30 分」autotune で **≥200 tps**（天井 ~200-250: 1.4-1.5GB/token ÷ ~300-350GB/s）+ 採用ピッチ文書（bring-up 日数・TAT ログ・llama.cpp E2B 比・webml 250tps 比） | ≥200 tps（cool）+ 最適化壁時計 ≤30 分 | **低〜中**（安いモデルで達成できること自体が証明） | **★★**（最終成果物 = 採用ピッチ） |

依存: M0-M2 と M3 は並行可。M4 は M2+M3 後。M6 は M2 後いつでも。**Phase A'（M6→M7）を最初のフル適用にする**（採用動機の成果物を先に出す）。

後続フェーズ（M7 後に本表へ展開）:
- **Phase B**: diffusion 側 e2e 再測定、llama.cpp 64 tok/s 比の更新、残ギャップの roofline 再分解
- **Phase C**: 量子化 layout の indexed type 化 + bounds 証明 → disable_robustness default ON（検証が性能を買う）。README "verified"→"typed" 降格。Hespera 別リポ分離
- **Phase D**: 判断点 — novel 機能（実行時学習等）を llama.cpp に移植 vs Hesper 続行を実測で再判定

---

## §3. 現在の状態

- **完了**: M0（★承認済み）、M1（probe、3/3 クリーン）、**M2+M4 前倒し（★レビュー待ち、汎用化+再現性検証済み）**:
  - **汎用 autotune core = `Hesper/WGSL/Autotune.lean`（library）**: Family 契約（space/feasible/gen/cfg/golden 宣言 ~50 行）に対し、engine が sweep/golden/probe-prune/occupancy/tune log/resume/**refine(top-10 × 300 iters × 3 反復)**/winners.csv/**実行時 `best` lookup** を共有提供。matmul は最初の instantiation（`regGenFamily`）。E2B mul_mv family が「matmul 専用でない」の証明担当（M6/M7）
  - **再現性（ユーザー指摘で検証）**: 独立 2 回の全 sweep → dense 勝者は完全一致、QKV-KV は同着プレート。ただし**単発 sweep 行には最大 ~3× の過渡外れ値**（同一設定が 0.267/0.767ms）→ **refine 段を新設**（top-10 を 300×3 で再計測、min-of-reps で確定。反復内ばらつき ±2%）
  - **正直な勝ち幅（300×3 同一ハーネス・cool、当初の 2.7×/1.7× は外れ値 baseline による誇張と判明・訂正)**: QKV-KV 0.283ms vs deployed 0.417ms = **1.45×**、dense-gu 0.463 vs 0.580 = **1.25×**。floor 距離 QKV-KV 2.9×、dense 2.3×
  - **汎用 kernel の忠実性確認**: Gen(deployed 設定) 0.413/0.577 ≈ 実 deployed kernel 0.417/0.580（1% 以内）= M4 合否「64×32 を再発見」クリア
  - 勝ちパターン: **BK16 + 小中 tile + sgC 多め（wg128）**、全行 spill 無し
- **M5 判定（★レビュー待ち、正直な結論 = e2e NEUTRAL）**: 実 decode 8 shape (M=nP=320) を sweep+refine → `DG_TUNED=1`（`Autotune.best` lookup 駆動、数字手書き無し、winners.csv 差し替えで decode 再ビルド不要）。**初回は熱で腐った勝者が +39ms の regression → incumbent ガード（deployed 設定が refine に常時参戦、勝てなければ winner 無し=構造的に regression 不可）+ cool refine で修正 → ABAB: TUNED 642/611 vs default 639/607 = ノイズ内**。事後算: 勝った 6 shape の節約 ~13ms/step vs ノイズ ±30ms — **tune 対象の WGSL matmul 群は step の ~10% しか占めず、そこの 20% は雑音に沈む**。DG_TUNED は opt-in のまま（shipped path 不変）。**autotune フロー自体は e2e 検証完了**（sweep→refine→lookup→統合→ゲートが全部機能し、regression を 2 回捕捉）。残り時間の主 = MoE(MSL 凍結)・battnB attention（WGSL、次の family 候補）・elementwise
- **進行中**: M5 ★レビュー → 次は M3（llama.cpp 精読表）or M6（E2B bring-up — matvec が step 時間の全てを占める土俵で、フロー証明の本命）
- **直近の実測**（2026-07-05, M4 Max, feat/diffusiongemma）:
  - TAT probe: 0.118 s/variant（cold Metal cache）。**再 sweep は Metal ディスク shader cache で compile 80ms→4ms** → 0.046 s/variant
  - occupancy probe 動作: deployed reg kernel = maxThreads=1024（spill 無し）, execWidth=32
  - DiffusionGemma 26B-A4B: llama.cpp 比 54-96%（平均 ~73%）、per-step ~800ms vs 363ms
  - roofline 上位: QKV-KV 4.9× above floor、dense g/up 3.2×（M2 の初弾対象）
- **次の一手（TODO）**:
  1. [x] M0: TAT 実測 → ★承認済み
  2. [x] M1: probe FFI（CI 確認のみ残）
  3. [ ] M2: runtime-K-loop パラメータ化 kernel（M4 前倒し）+ SWEEP=1 + 小 shape golden + tune log → QKV-KV/dense ランク表 ★
  3b. [ ] **TAT 対策（ユーザー指摘 2026-07-05）: decode 統合 build が ~8-10 分と長すぎる** — DiffusionGemmaDecode.lean が 2000 行超の単一ファイルで 1 行の変更が全再 elaborate を誘発。対策候補: (a) decode を複数モジュールに分割（forward/schedule/main）、(b) 滅多に使わない Examples/exe を default deps から外す・別ディレクトリに move、(c) 統合面を薄く保つ（lookup 駆動にしたのはその一歩）。M5 完了後に着手
  4. [ ] M3: llama.cpp 精読表 ★（M2 と並行）
  5. [x] **M6 完了: E2B greedy decode が llama.cpp と一致** — "The capital of France is" → **"Paris." + EOT** ✓ / "The largest planet…" → **"Jupiter." + EOT** ✓ / 64-token 詩も coherent。**現状 8.05 t/s**（graphs-off・correctness-first 経路）vs llama.cpp **156.5 t/s** = M7 の出発点 19.4×。★レビュー待ち。llama.cpp 参照確立: E2B 生成 **156.5 t/s**（M7 の現実的な的; 私の 200-250 BW 見積りは楽観だった）。
     踏んだ修正（1 日で 21 回の実行反復、全て engine 一般益）: GGUF u32 配列 metadata（E 系）/ MatFormer per-layer dims（tensor shape 優先, E2B L0 ffn=6144≠12288）/ PLE 1.6GB 表 → row-staging 64-slot（binding 上限+robustness clamp+batch 順序ハザード 3 連対応）/ binding 4 倍数 round / CreateBindGroup error scope（作成時理由の可視化）/ 書込 aliasing×3 を in-place kernel 化（qkvNorm・qNorm・normThenAdd — CUDA 合法/WebGPU 違法）/ ce 名 sanitize（Float 埋込→WGSL 不正識別子）/ **maxComputeWorkgroupSize 1024 要求**（wg-512/1024 kernel 解禁 = autotune の sg4x4 も解禁）/ Q4_K embd lookup kernel 新設（dequantQ4KElementAt 抽出）/ Q4_K lm-head chunked f16 前 dequant（f32 中間 1.6GB 回避）/ BlockCoop 2D grid（vocab 262144 > 65535）/ dp4a enable（WebGPU、subgroups 時）。
     **数値 parity bisect（進行中、大きく前進）**: llama-eval-callback を参照に層別比較を確立。
     - **✅ 消去済み**: Q4_K embedding lookup（CPU golden 16/16 一致）/ inp_scaled（llama と厳密一致）/ Q6_K down matmul（正しかった）/ normThenAdd kernel（CPU golden 一致）
     - **✅ 大修正: Metal relaxed-math の fast::tanh は |x|≳44 で NaN** — E2B の GELU inner=58 で発火（E4B は振幅不足で潜伏）。DSL codegen で `tanh(clamp(x,±20))` に（engine-wide、意味論不変）。修正後 **ffn_out-0 が llama と parity**（-6.2511 vs -6.2759）
     - **層別スキャン: layer 0 完全一致（attn 0.02 / ffn 0.006）→ 乖離は layer 0 末尾の PLE 適用チェーン**（llama: proj@inp_scaled → SCALE → per-256 RMS_NORM×projNormW → +selected → SCALE(1/√2) → gelu(inp_gate@pe_in)×inp_per_layer → proj → norm×post_norm → +pe_in → ×layer_output_scale）。our per_layer_embd_out ≠ llama → **PLE 式のどこかの SCALE 定数 or 順序が違う**
     - 学び: llama の tensor 名は add 前後でズレる（ffn_post_norm=add 前、pe_in=add 後）— 比較時は OP チェーンで確認
     - 訂正: 「batched prefill の staging 順序ハザード」理論は誤り（prefill は非バッチ、各 dispatch 即時実行）。slot 化は無害だが不要だった
     - **最終ラウンド（2026-07-05、4 バグで decode 完全一致）**: PLE 定数は全部正しかった — 犯人は**テンソル型の思い込み**×3 + **prepared-cache の buffer 捕捉**×1:
       1. **`per_layer_token_embd` は E2B では Q5_K（type 13）**（E4B は Q6_K）— Q6_K として dequant していた。CPU golden（llama node_1 GET_ROWS と bit 一致）で確定 → `Hesper/Quantization/Q5_K.lean` 新設（176B/block は 4-aligned なので u32-indexed、scale unpack は Q4_K と同一で `getScaleMin` 再利用）+ loader 型検出 + 両 call site 分岐
       2. **`per_layer_model_proj` は BF16（type 30）** — 生 bytes を f16 として matmul に食わせていた（指数幅違い→ゴミ）。GPU BF16→F16 repack kernel（算術変換、bitcast 不要）を load 時に 1 回。GGUF Loader.GGMLType に BF16 追加
       3. **`blk.*.inp_gate/proj` は E2B では F32**（E4B は Q4_K）— loadLinear が F32 を Q4_K として解釈（ple_gate が ±10⁴）。`Linear.QuantFormat.F16` 新設: load 時 GPU pack f32→f16、forward/forwardBatchDP4A に F16 早期分岐（executeMatMulTransposeF16）
       4. **decode 常時 196228 の真因 = finalNorm の prepared-cache が prefill の buffer binding (buf2→buf1) を捕捉**、decode の ping-pong は **35 層（奇数）で buf1→buf2 に反転** → replay が古い scratch を読む。E4B は 42 層（偶数）で偶然一致し潜伏。`state.finalNormDecodePrepared`（decode 専用 ref）で分離。**教訓: prepared/cached dispatch は buffer binding を捕捉する — 呼び出し毎に buffer が変わる call site で shared ref を使ってはならない（層数の偶奇でしか発火しない罠）**
     - 検証: decode hidden vs 7-token prefill hidden relL2 0.014（L34）= KV-cache/全層 decode 経路も parity。層別スキャン l_out-0/34 とも llama と一致
     - 診断手法の資産化: `scripts/llama_parity/scan_layers.py`（llama-eval-callback text vs golden .bin 層別比較）、decode 側 `single_p{pos}_finalNorm/logits` dump 追加、dumpBuf を WebGPU batch 下で安全化（isBatching probe + 再 open）
     - CPU lm-head 切り分け（hidden 正しい/GPU logits 誤り→犯人は norm→lmHead チェーン）が決定打だった — bisect は「GPU dump → CPU で続きを計算」が最速
  6. [~] **M7 進行中（2026-07-05）: 8.05 → 68.0 t/s（8.4×）— まだ kernel チューニング前、全部オーバーヘッド除去**。llama.cpp 156.5 の 43%。
     - **ラダー**: 8.05 → **39.9**（WebGPU executeWithConfigCached が cacheKey を捨て毎 dispatch WGSL 全再生成 ~180µs×600 → CUDA 同様 key を権威化、collision 検証 = KEYED=0 と全 1151 golden ビット一致）→ **64.1**（pipeline/bindgroup/kcr の 3 cache が Array 線形スキャン ~450k probe/token → Std.HashMap）→ **68.0**（Q4_K dp4a lm-head 新設: f16 800MB read → Q4_K 226MB、HESPER_LMHEAD_F16 不要化 + VRAM -786MB）
     - **既知問題（棄却済み lever）**: bridge の single-compute-pass 化（HESPER_SINGLE_PASS=1 opt-in、default OFF）— decode L1-attention から bit-divergence（mode 毎に決定的、prefill は不変）= intra-pass hazard がどこかで漏れる。利得 +0.6ms のみと判明（pass 切替は想定より安い）→ hunt は保留。再現: poem 24tok の md5 が SINGLE_PASS=0/1 で異なる
     - **現状内訳** (~14.7ms/token): GPU ~8ms（帯域 floor 見積 ~4ms、FFN matvec 838MB/token が支配）+ host record ~2-3ms + argmax readback 0.6ms + prePLE 0.8ms
     - **次**: mul_mv Family を Autotune に実装（FFN gate/up/down K=1536 N=6144/12288 dp4a matvec）= フレームワーク汎用性証明の本命 + GPU 2.2× ギャップの主対象。その後 argmax/readback、prePLE
     - 単体診断ツール: gemma4-profile が argv でモデル指定可に（E2B 対応）

**M1 で確定した実装制約**（原則に準ずる）:
- **Dawn の SetLoggingCallback は CAPTURELESS lambda 限定**: capture 付き functor は寿命が繰返し発火をカバーせず use-after-free（確率的に quiet 無視 + SIGTRAP/SIGABRT を実測）。状態は global で渡す。
- **Tint MSL の staticThreadgroupMemoryLength は常に 0**（threadgroup メモリは dispatch 時動的割当）→ shared サイズ列は Lean 側で計算して出す。
- Metal ディスク shader cache: 同一ソースの再 compile は ~4ms → 再 sweep はほぼ無料。

## §4. 決定ログ

| 日付 | 決定 | 根拠 |
|---|---|---|
| 2026-07-05 | MSL hybrid 拡張を棄却 | batch/bubble で回収失敗（DG_MSLONESTREAM/FUSEDOWN neutral 実測）。Metal では WGSL との決定差小 |
| 2026-07-05 | 強度削減/scalar-accumulator 系の WGSL 書き換えを棄却 | Metal が回収（null 実測）。効くのは let' CSE のみ（1.37→1.15×） |
| 2026-07-05 | tile 幅の単独模倣を棄却 | WIDE64 実測で逆効果。リファレンスは構造ごと写す（原則 7） |
| 2026-07-05 | JAX 移行を棄却 | jax-metal 未成熟 / Pallas に Metal 無し / block-quant 一級サポート無し |
| 2026-07-05 | E2B をフロー証明のターゲットに採用 | 帯域律速で Tint 税が消える土俵 + 公開比較（webml 250tps / llama.cpp）+ E4B 資産で 1-2 日圏 |
| 2026-07-05 | **M0 合格**: sweep TAT = 0.118s/variant 実測（設計成立） | TATPROBE=1。当初 8x8Reg で probe → K 全展開で compile 爆発 → deployed 型（runtime K loop）に切替えて成功。**sweep 基盤 = runtime-K-loop 必須**を §1 に昇格 |
| 2026-07-05 | M0 ★承認（ユーザー）+ M2 設計変更承認: M4（runtime-K-loop パラメータ化 kernel）を M2 に前倒し | full-unroll 制約により 8x8Reg が sweep 基盤に使えないため |
| 2026-07-05 | **M1 合格**: occupancy probe 動作（maxThreads/execWidth）。callback は captureless 限定 | capture 付き lambda で use-after-free（quiet 確率無視 + SIGTRAP）を実測 → global 状態 + captureless で 3/3 クリーン |
| 2026-07-05 | sweep は resumable（CSV skip + SWEEP_LIMIT）に | 300 variant 一気は SIGSEGV。真因 = **wg=512 が maxComputeWorkgroupSizeX=256 を超え pipeline が null → execute 経路が null 参照で SIGSEGV**（engine 堅牢性の課題、别途）。feasibility に wg≤256 を追加して解消 |
| 2026-07-05 | **M2 合格（524 variants、golden 0 fail）**: QKV-KV 0.733→0.267ms、dense 0.833→0.500ms（同条件）。勝ち = BK16+小 tile+wg64-128 | 「4.9×/3.2× above floor は未チューニングが原因」仮説が実証された。深 tile(BK32/TM4)は M=262 では padding 浪費+occupancy 損 |
| 2026-07-05 | 手動 tune ループ→**汎用 Autotune library に抽出**（Family 契約 + 共有 engine + 実行時 lookup）。ユーザー指摘（matmul 専用/手動 tune/パラメータ非外出し）が正しかった | Triton の @autotune と同じ境界: kernel 作者は宣言のみ、探索/記録/消費は機構 |
| 2026-07-05 | **勝ち幅を 2.7×/1.7× → 1.45×/1.25× に訂正**（refine 300×3 実測）。probe-prune + refine 段を新設 | 単発 sweep 行に ~3× 過渡外れ値（0.267/0.767 同一設定）→ sweep=安く順位付け、refine=確実に決定、の 2 段が必須。ユーザーの再現性指摘が的中 |
| 2026-07-05 | **incumbent ガード新設**（deployed 設定が refine に常時参戦、勝てなければ winner 無し） | 8-shape 連続 sweep の後半が熱で腐り、deployed より遅い設定が「勝者」になり decode で +39ms regression（ABA で捕捉）。ガード+cool refine 後は Q-full/attnO-SWA で incumbent が正しく防衛 |
| 2026-07-05 | **M5 = e2e NEUTRAL、DG_TUNED は opt-in 維持** | 勝った 6 shape の合計節約 ~13ms/step はノイズ ±30ms 未満（tune 対象は step の ~10%）。原則 4 のゲートが機能。フロー自体は検証完了 — 効かせるには step 時間を支配する対象（E2B matvec / battnB）に向ける |
| 2026-07-05 | **M6 合格: E2B greedy = llama.cpp 一致（"Paris."✓ "Jupiter."✓）、8.05 t/s** | 最終 4 バグ = E2B 固有テンソル型（PLE 表 Q5_K / proj BF16 / inp_gate·proj F32）×3 + finalNorm prepared-cache の buffer 捕捉（35 層奇数で ping-pong 反転、E4B 42 層は偶然潜伏）。詳細 §3-5 |
| 2026-07-05 | 量子化型は tensor 毎に GGUF から読む（思い込み禁止）を原則運用に | E2B は同名 tensor が E4B と別型（Q6_K→Q5_K, Q4_K→F32, F16→BF16）。3/4 バグがこのクラス。loadLinear/loader は型検出 + 明示 throw（未対応型）に |
| 2026-07-05 | prepared/cached dispatch の shared ref は「buffer が呼び出し毎に不変」の site 限定 | finalNorm bug の教訓。ping-pong buffer を渡す site は専用 ref か throwaway。層数の偶奇でしか発火しない罠として決定ログに固定 |
| 2026-07-05 | **WebGPU cacheKey を権威化（CUDA と同契約）** — 8.05→39.9 t/s | 毎 dispatch の WGSL 再生成 ~180µs×600 が decode の 88% だった。collision 検証: HESPER_KEYED_PIPELINES=0 と全 1151 golden ビット一致。契約 = key は生成 WGSL を一意に識別（baked param は key に含める） |
| 2026-07-05 | 3 つの dispatch cache を HashMap 化 — 39.9→64.1 t/s | pipeline/bindgroup/kcr が Array 線形スキャン（~450k probe/token）。ホスト記録 5.3→1.1ms |
| 2026-07-05 | single-compute-pass 化は default OFF で保留 | decode L1-attention から bit-divergence（intra-pass hazard 漏れ、mode 毎決定的・prefill 不変）。利得 +0.6ms のみ。reproducer 記録済（poem 24tok md5） |
| 2026-07-05 | Q4_K dp4a lm-head を default に — 64.1→68.0 t/s、HESPER_LMHEAD_F16 廃止（opt-in 化） | f16 事前 dequant 800MB read → Q4_K 226MB 直読。VRAM -786MB。prefill/decode 両方 |
