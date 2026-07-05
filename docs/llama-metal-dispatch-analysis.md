# M3: llama.cpp Metal 精読 — E2B decode の 2.2× ギャップの正体（2026-07-05）

## 結論（1 行）

**ギャップはカーネルでも dispatch 数でもなく「dispatch 並列性」**: llama.cpp は
`MTLDispatchTypeConcurrent` + hazard 時のみ barrier、Dawn は `MTLDispatchTypeSerial`
ハードコード（crbug.com/425987598）で全 dispatch 直列 = ~14µs/dispatch の床。

## 1. dispatch/encoding 構造の一致表

| | llama.cpp Metal | hesper (Dawn-Metal) |
|---|---|---|
| encoder | 1 command buffer/eval、`computeCommandEncoderWithDispatchType:` **Concurrent** (`ggml-metal-device.m:469`) | 1 command buffer/token ✓ だが Dawn が **`MTLDispatchTypeSerial` をハードコード** (`CommandBufferMTL.mm:165`, crbug.com/425987598) |
| 同期 | `ggml_mem_ranges`（src/dst のバッファ区間追跡, `ggml-metal-common.cpp`）で **真の hazard の時だけ** `memoryBarrierWithScope:MTLBarrierScopeBuffers` → 独立 op はオーバーラップ | pass 内も pass 間も**全 dispatch ペアで直列**（single-pass 化しても submit+wait 9.5ms 不変を実測 — barrier コスト ≒ pass 切替コスト） |
| 実 op 数/token | **~1033**（MUL 297 + MUL_MAT 277 + RMS_NORM 242 + ADD 106 + ROPE 50 + FA 35 + GEGLU/GELU 70 + SCALE 6; VIEW/RESHAPE/PERMUTE 238 は no-op）≈ 30/層 | **684** ≈ 19.5/層（fused norm+matmul, norm+add 等で既に llama より融合が進んでいる!） |
| 结果 | 6.4 ms/token (156.5 t/s) | GPU 9.5ms + host ~4ms = 14.4ms (69.5 t/s) |

**含意**: 我々は llama.cpp より 34% 少ない dispatch 数で 1.5× 遅い GPU 時間。
オーバーラップが無いことがすべて。fusion（684→400）は ~-4ms 止まり（~100 t/s 天井）。

## 2. mul_mv カーネル構造の一致表（Q4_K, M=1）

| | llama.cpp `kernel_mul_mv_q4_K_f32` | hesper 勝者 (mulMvQ4K sweep) |
|---|---|---|
| 構成 | N_R0=2 rows/simdgroup × NSG=2 sg/TG = 64 threads, 4 rows/TG | R=1, W=1 = 32 threads, 1 row/WG（R2W1 と同着圏） |
| K 分割 | simdgroup 内 ix=tiisg/8 で 4 block 並列 | kbxStart=tid/16 で 2 block 並列 |
| 判定 | 同クラス。**カーネルはギャップの主因ではない**（deployed R1W4 だけが Apple で負け筋だった） |

## 3. 取れる道

| 案 | 内容 | 期待 | コスト/リスク |
|---|---|---|---|
| A. Dawn パッチ | vendored Dawn の `CommandBufferMTL.mm` を Concurrent + hazard-only barrier に（Dawn は per-dispatch の SyncScopeUsageTracker を既に持つ） | GPU 9.5→~4-5ms | Dawn fork の保守; Dawn 内部の正しさリスク; upstream が将来対応するかも |
| B. native dispatch transport | decode の実行だけ metal_replace 経路: 各 pipeline の Tint-MSL（HESPER_DUMP_MSL で捕獲済みの機構）→ MTLComputePipelineState、Dawn buffer → MTLBuffer（bridge 検証済み）、**自前 Concurrent encoder + Lean 側で ggml 式 mem-range hazard 判定**（依存は buffer リストから自明） | 同上 + 制御自在 | macOS 専用 fast path（WebGPU 経路は default で維持）; metal_replacer の「DEBUG/REFERENCE」スコープ決定の変更 = ★ユーザー承認要; 記録/再生層 ~数百行 |
| C. fusion のみ | WebGPU 内で 684→<400 | ~90-100 t/s 天井 | 可搬・低リスクだが llama.cpp に届かない |

**重要な整理**: 以前棄却した「MSL hybrid」は**手書き MSL カーネル**の話。B は**カーネルを一切書かない**
（全カーネルは DSL→WGSL→Tint 生成のまま、dispatch の運搬層だけ native）。「Tint 税はチューニングで
消える」の結論は保持される — 消えないのは **Dawn の Serial dispatch 税**で、これはカーネル品質と無関係。

## 検証手順（案 B/A 共通）

1. 正しさ: Paris/Jupiter ゲート + 全 golden dump ビット比較（barrier 判定の誤りは即 bit-divergence として出る — single-pass hunt で確立した手法）
2. 性能: HESPER_GPUBUSY 相当で GPU 時間を直接測定、ABAB
3. hazard 判定は保守側（怪しければ barrier）から始めて緩める
