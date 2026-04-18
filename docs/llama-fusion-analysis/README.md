---
title: llama.cpp CUDA fusion strategy — research report
date: 2026-04-16
scope: basis for hesper-gemma4 Circuit DSL extension plan
---

# llama.cpp CUDA fusion strategy — research report

This directory collects the analysis that underpinned hesper-gemma4's
move to compress kernel count toward **llama.cpp CUDA's 187 kernels/tok**.

**Date**: 2026-04-16
**Subject**: `llama.cpp` master as of this date (see `llama.cpp/.git` for commit)
**Test machine**: RTX 4070 Ti (CC 8.9), CUDA 12.8, nsight-compute 2025.1.1

## Files

| File | Contents | Matching llama.cpp source |
|---|---|---|
| [`00-summary.md`](00-summary.md) | **Top-level summary** — read this first | — |
| [`01-graph-fusion.md`](01-graph-fusion.md) | Op-graph pattern matcher | `ggml-cuda.cu` (5,316 lines) |
| [`02-mmvq-epilogue.md`](02-mmvq-epilogue.md) | `mul_mat_vec_q` template + epilogues | `mmvq.cu` (1,150 lines) |
| [`03-mmvf-vecdotq.md`](03-mmvf-vecdotq.md) | f16/f32 mat-vec and dp4a inner products | `mmvf.cu` (862), `vecdotq.cuh` (1,269) |
| [`04-mmf-dense.md`](04-mmf-dense.md) | Dense GEMM (prefill path) | `mmf.cuh` (908 lines) |
| [`05-hesper-dsl-plan.md`](05-hesper-dsl-plan.md) | **hesper-side action plan** | `Hesper/Circuit/*.lean` |
| [`06-compiler-transforms.md`](06-compiler-transforms.md) | Fusion viewed as compiler passes; the "reduce wall" | — |
| [`07-scatter-design.md`](07-scatter-design.md) | Dynamic-offset / scatter design discussion (**implemented**) | `Hesper/Circuit/IR.lean` |
| [`08-scatter-impl-notes.md`](08-scatter-impl-notes.md) | **Scatter unification: impl + verification** | commit `b515e13` |
| [`12-complete-cuda-flow.md`](12-complete-cuda-flow.md) | **Complete CUDA flow** for Gemma 4 E4B | `fattn*`, `mmvq`, `mmq`, `rope`, `norm` |
| [`14-nsys-fresh-comparison.md`](14-nsys-fresh-comparison.md) | nsys hesper vs llama.cpp (post-Q4K-fixes) | nsys captures |
| [`15-llama-single-path.md`](15-llama-single-path.md) | **Architecture plan**: adopt llama.cpp's single-forward shape-polymorphic path | `llama-context.cpp`, `gemma4-iswa.cpp` |
| [`16-shape-audit-checklist.md`](16-shape-audit-checklist.md) | **Phase 1 shape audit**: every forward kernel classified A/B/C; reduces rewrite to 3 items | `Hesper/Models/Gemma4.lean`, `Hesper/Layers/*.lean` |
| [`17-phase2-item2-findings.md`](17-phase2-item2-findings.md) | Phase 2 item 2 first attempt: why the naive SWA→batched-path change regressed; revised plan requires bit-parity harness + separate batched SWA kernels | — |

## Measurement artefacts

- `/tmp/ncu_top3.ncu-rep` — ncu detail for hesper's top-3 kernels
- `/tmp/ncu_llama_cuda.ncu-rep` — ncu for llama.cpp CUDA `mul_mat_vec_q<Q4_K>`
- `/tmp/step10.sqlite` — nsys capture of hesper 30-token decode
- `/tmp/llamacpp.sqlite` — nsys capture of llama.cpp CUDA 30-token decode

## Scope notes

1. **Decode-only (`ncols_dst = 1`).** llama.cpp disables most of its
   fusion in prefill/batch mode (`ncols_dst == 1` guard). hesper's
   optimisation target is decode too, so these match.
2. **CUDA Graphs are not on the table.** Investigation C found the
   end-of-token `cuCtxSynchronize` is dominant; CUDA Graphs would
   yield < +1 TPS. This report doesn't pursue Graphs-based wins.
3. **MoE / SSM variants are out of scope.** Gemma 4 is non-MoE and
   non-Mamba, so TopK-MOE (#I) and SSM_CONV+SILU (#G) don't apply.

## Starting point (2026-04-16)

| Backend | Kernels/tok | TPS |
|---|---:|---:|
| llama.cpp CUDA | 187 | 115 |
| llama.cpp Vulkan | 1,186 | 98 |
| **hesper (starting point)** | **975** | **49** |

hesper already dispatches fewer kernels than Vulkan but runs at half the
TPS. This report's purpose is to locate the remaining gap and lay out a
path forward.
