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

- **進行中**: M0（この文書の確定 + TAT 実測）— 新 Fable セッションで開始すること
- **直近の実測**（2026-07-05, M4 Max, feat/diffusiongemma）:
  - DiffusionGemma 26B-A4B: llama.cpp 比 54-96%（平均 ~73%）、per-step ~800ms vs 363ms
  - roofline 上位: QKV-KV 4.9× above floor、dense g/up 3.2×（未チューニング = M2 の初弾対象）
  - WGSL-vs-hand-MSL kernel 床: 1.15×（H1 let' 後）
- **次の一手（TODO）**:
  1. [ ] M0: TAT 表の実測（sweep 骨格ができた時点で variant/秒を測る → §1 に確定値を記入）★
  2. [ ] M1: probe FFI
  3. [ ] M2: SWEEP=1 Tier 1 → QKV-KV/dense のランク表 ★
  4. [ ] M3: llama.cpp 精読表 ★（M2 と並行）
  5. [ ] M6: E4B 例の現状確認 → E2B bring-up ★
  6. [ ] M7: E2B ≤30分 autotune で ≥200tps ★★

## §4. 決定ログ

| 日付 | 決定 | 根拠 |
|---|---|---|
| 2026-07-05 | MSL hybrid 拡張を棄却 | batch/bubble で回収失敗（DG_MSLONESTREAM/FUSEDOWN neutral 実測）。Metal では WGSL との決定差小 |
| 2026-07-05 | 強度削減/scalar-accumulator 系の WGSL 書き換えを棄却 | Metal が回収（null 実測）。効くのは let' CSE のみ（1.37→1.15×） |
| 2026-07-05 | tile 幅の単独模倣を棄却 | WIDE64 実測で逆効果。リファレンスは構造ごと写す（原則 7） |
| 2026-07-05 | JAX 移行を棄却 | jax-metal 未成熟 / Pallas に Metal 無し / block-quant 一級サポート無し |
| 2026-07-05 | E2B をフロー証明のターゲットに採用 | 帯域律速で Tint 税が消える土俵 + 公開比較（webml 250tps / llama.cpp）+ E4B 資産で 1-2 日圏 |
