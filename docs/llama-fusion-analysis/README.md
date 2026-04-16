---
title: llama.cpp CUDA fusion 戦略 調査レポート
date: 2026-04-16
scope: hesper-gemma4 Circuit DSL 拡張方針の根拠資料
---

# llama.cpp CUDA fusion 戦略 調査レポート

このディレクトリは、hesper-gemma4 の Circuit DSL を **llama.cpp CUDA 並みの
kernel 数 (187/tok)** まで圧縮するための調査資料です。

**調査日**: 2026-04-16
**対象**: `llama.cpp` 最新 master（調査時点、コミットは `llama.cpp/.git` 参照）
**測定環境**: RTX 4070 Ti (CC 8.9), CUDA 12.8, nsight-compute 2025.1.1

## ファイル構成

| ファイル | 内容 | 対応する llama.cpp ソース |
|---|---|---|
| [`00-summary.md`](00-summary.md) | **統合サマリー**（まず読む） | — |
| [`01-graph-fusion.md`](01-graph-fusion.md) | op graph パターンマッチャ | `ggml-cuda.cu` (5,316行) |
| [`02-mmvq-epilogue.md`](02-mmvq-epilogue.md) | `mul_mat_vec_q` テンプレート＆epilogue | `mmvq.cu` (1,150行) |
| [`03-mmvf-vecdotq.md`](03-mmvf-vecdotq.md) | f16/f32 matmul と dp4a 内積 | `mmvf.cu` (862行), `vecdotq.cuh` (1,269行) |
| [`04-mmf-dense.md`](04-mmf-dense.md) | 密 GEMM (prefill系) | `mmf.cuh` (908行) |
| [`05-hesper-dsl-plan.md`](05-hesper-dsl-plan.md) | **hesper 側の対応計画** | `Hesper/Circuit/*.lean` |
| [`06-compiler-transforms.md`](06-compiler-transforms.md) | コンパイラ変換分類と reduce の壁 | — |
| [`07-scatter-design.md`](07-scatter-design.md) | **Dynamic Offset / Scatter 設計課題 (open)** | `Hesper/Circuit/IR.lean` |

## 測定データ

- `/tmp/ncu_top3.ncu-rep` — hesper top-3 kernel の ncu 詳細
- `/tmp/ncu_llama_cuda.ncu-rep` — llama.cpp CUDA `mul_mat_vec_q<Q4_K>` の ncu
- `/tmp/step10.sqlite` — hesper 30tok decode の nsys
- `/tmp/llamacpp.sqlite` — llama.cpp CUDA 30tok decode の nsys

## 重要な前提

1. **対象は decode (ncols_dst=1) のみ**。prefill / batch では llama.cpp の fusion は
   大半が無効化される (`ncols_dst==1` ガード)。hesper も decode 最適化を優先する限り
   この制約は問題ない。
2. **CUDA Graphs は Investigation C で見送り確定**（ncu gap histogram で
   end-of-token sync が dominant と判明、<+1 TPS）。このレポートも
   CUDA Graphs を前提とした改善は扱わない。
3. **MoE / SSM 系は Gemma 4 に無関係** なので TopK-MOE fusion (#I) /
   SSM_CONV+SILU (#G) は対象外。

## 現在位置 (2026-04-16)

| backend | kernels/tok | TPS |
|---|---:|---:|
| llama.cpp CUDA | 187 | 115 |
| llama.cpp Vulkan | 1,186 | 98 |
| **hesper (current)** | **975** | **49** |

hesper は Vulkan 並みの dispatch 数まで絞れているが、TPS は半分。
このレポートの目的は **さらに dispatch を減らし、かつ per-kernel を速くする
道筋** を明文化することです。
