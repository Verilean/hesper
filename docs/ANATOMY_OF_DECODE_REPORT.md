# Where the Milliseconds Go: a Measured Anatomy of LLM Decode on Apple Silicon across Three Engines

*Verilean / Hesper post-mortem technical report — 2026-07-06.*
*All measurements: M4 Max (546 GB/s peak DRAM BW), gemma-4-E2B-it, greedy decode.*
*Methods and raw logs: `DEVPLAN.md` §8–§13, `DEVLOG.md`, `tools/replay/README.md`.*

## Abstract

We built Hesper, a verified-inference engine in Lean 4 targeting WebGPU (Dawn/Metal),
and optimized its Gemma-4-E2B decode from 8.05 to ~119 tok/s. To explain the remaining
gap to llama.cpp (~147 tok/s) and a hand-written WebGPU engine (webml, ~250 tok/s), we
developed a **bare-Metal replay harness**: capture one decode token's full dispatch
sequence from any engine — ours via a Lean hook, llama.cpp via its Metal encoder
wrappers, webml via monkey-patched WebGPU APIs in headless Chrome — and re-execute it on
a minimal Metal loop with identical methodology. The replays, together with llama.cpp's
own ablation toggles and pre-registered predictions, falsified every "systems" theory of
the gap in turn: dispatch transport (worth ~0.7 ms), concurrent scheduling (~0.1–0.6 ms
in *every* engine — decode dataflows are ~86–96 % dependency-barriered), op count
(llama runs 1.5× our ops faster), and small-op fusion (−28 dispatches = ±0 ms).
What remains is arithmetic: **kernels + graph set the floor; every mature runtime — a
JS loop in Chrome, C++ ggml, our Lean loop after a CUDA-Graphs-style frozen replay — is
a ±0.5 ms rounding term on top.** webml's 1.7× lead over llama.cpp is kernel craft
(fat epilogue-fused kernels, int4+int8-SRQ operands), not runtime or model format
(the QAT model accounts for only ~10–15 % of bytes). We further show the classic
isolated kernel benchmark overstates matvec bandwidth by 13–26 % (warm-SLC artifact),
and that re-autotuning against a cold-stream objective recovers ≤3 %: per-op bandwidth
is a function of op size, not of tuning parameters. We conclude with an honest account
of what a typed/verified DSL and an autotune framework did and did not buy, and why
jax-metal exhibits the same disease in a worse stage.

## 1. Introduction

The project's original bet: browsers + WebGPU as a distribution channel, with formal
verification (Lean 4) as differentiation — "proof-carrying kernels as fast as unproven
hand-written ones." Two premises failed in the market before any of the engineering
failed: the browser lane is already served by hand-written engines (webml) and
compiler stacks (MLC/web-llm), and no near-term LLM buyer demands verified inference
(the verdict of record: `DEVPLAN.md` §4). The bet was retired on 2026-07-06.

What remained worth doing — and what this report documents — is the measurement
program that the retirement decision demanded: *if our conclusions about why we were
slower are right, removing the abstraction layers must produce a near-fastest engine.*
Executing that falsification test produced a decode anatomy we believe is more broadly
useful than the engine itself.

## 2. The engines

| | Hesper | llama.cpp (Metal) | webml gemma-4-webgpu |
|---|---|---|---|
| language / stack | Lean 4 → WGSL DSL → Dawn → Tint → Metal | C++ → hand MSL → Metal | JS → hand WGSL (templated) → Chrome/Dawn → Tint → Metal |
| model format | GGUF Q4_K_M (1.26 GB/token read) | GGUF Q4_K_M (1.26 GB) | QAT int4 "mobile" (~1.12 GB) |
| scheduling | Dawn serial (hardcoded, crbug.com/425987598) | MTLDispatchTypeConcurrent + ggml_mem_ranges | Dawn serial |
| notable kernel tech | llama-port f32 matvecs, fused QKV/gate-up, autotuned | many thin tuned kernels, N_R0×NSG matvecs | few fat kernels: epilogue-fused norms/adds, int8-SRQ activations, subgroup-matrix |

Also referenced: jax-metal (closed PJRT plugin → closed MPSGraph; no kernel authoring
escape hatch) as the terminal case of the abstraction disease discussed in §7.

## 3. Methodology

**Bare-Metal replay (the core instrument).** Capture ONE steady decode token's complete
dispatch sequence — pipelines/MSL, buffers with binding order, grids, threadgroup
memory — and re-execute it on a ~100-line Metal loop, timing `GPUStartTime→GPUEndTime`
(20 iters, min & avg). Same loop for all engines ⇒ directly comparable numbers.
Capture adapters: (a) Hesper: a hook in the single dispatch choke point, MSL obtained by
running the *pinned* tint CLI on the exact WGSL; (b) llama.cpp: a patch on its eight
Metal-encoder wrapper functions (local fork only); (c) webml: WebGPU prototype
monkey-patching in headless Chrome around the unmodified app, including a content
mirror of ≤64 KiB buffers (uniform values steer loop trip counts — zeros would fake
timing). Full HOWTO: `tools/replay/README.md`.

**Protocol.** Predictions registered in DEVPLAN before measuring (they were wrong
twice, which is the point); token-sequence equality as the correctness gate for any
change that touches execution; cool box, no stray processes; negative results recorded.

**Hazard analysis.** To separate legitimate concurrency from what we call the *race
mirage*, the replay supports automatic hazard barriers (whole-buffer granularity,
llama's `ggml_mem_ranges` semantics): a barrier before any op that reads a
written-since-barrier buffer or writes a read/written one.

## 4. Results

### 4.1 The Hesper optimization ladder (what the 15× was made of)

| step | tok/s | cause |
|---|---|---|
| correct bring-up | 8.05 | 4 model-specific bugs (tensor-type assumptions ×3, prepared-dispatch buffer capture) |
| authoritative pipeline keys | 39.9 | engine re-generated full WGSL per dispatch just to hash it |
| HashMap caches | 64.1 | three linear-scan caches, ~450 k probes/token |
| direct-quantized lm-head | 68.0 | 800 → 226 MB/token read |
| llama-port f32 matvecs | 74.8 | dp4a abandoned (`dot4I8Packed` is *emulated* on Apple; f32 port 1.4–1.7× faster, 90–97 % BW isolated) |
| PLE fusion + 1-dispatch attention | 89.6–96 (steady) | 684 → 567 dispatches |
| **native transport + frozen token replay** | **~119 (8.4 ms)** | §4.3: skip Dawn *and* the per-token host walk; CUDA-Graphs analogue |

Note the shape of the ladder: 8→90 was almost entirely **engine tax removal** (host
work per dispatch), not kernel speed. The kernels' contribution (llama-port matvecs)
was +8 % e2e despite 1.4–1.7× isolated wins — foreshadowing §4.4.

### 4.2 The cross-engine anatomy (the central table)

One decode token, same bare-Metal harness, serial:

| engine | real total | ops | replayed GPU (serial) | µs/op | **host = total − GPU** | eff. BW |
|---|---|---|---|---|---|---|
| webml (Chrome, JS) | ~4.0 ms (~250 t/s) | 316 | **3.90 ms** | 12.3 | **~0.1–0.3 ms** | 287 GB/s |
| llama.cpp (C++) | 6.78 ms (147.6 ± 3.1 t/s, tg64)¹ | 854 | **6.31 ms** | 7.4 | **~0.5–0.6 ms** | 200 GB/s |
| Hesper (Lean, Dawn path) | ~9.8 ms | 572 | **6.78 ms** | 11.9 | **~2.5–3.0 ms**² | 186 GB/s |
| Hesper (Lean, frozen-native) | **8.4 ms** | 572 | 7.5 ms (in-decode) | — | **~0.9 ms** | — |

¹ An earlier llama-cli measurement gave 156.5 t/s (different tool/day/box state).
² Dawn encoder overhead ~0.7 ms + Lean per-token walk ~1.3 ms + argmax sync 0.55 ms.

Readings:
- **The GPU column ranks the engines; the host column is a rounding term** for every
  mature runtime. The JS host is the *thinnest* (Chrome encodes via IPC to the GPU
  process, overlapped; bind groups prebuilt).
- **Op count predicts nothing** (llama: most ops, second-fastest; webml: fattest
  per-op, fastest total). Total = bytes moved × kernel efficiency.
- Normalizing webml to our byte count (×1.26/1.12) gives ≈4.4 ms: **≈2.4 ms of our
  6.78 is kernel-shape/fusion quality on equal bytes** — the largest single factor in
  the entire anatomy. The QAT model itself is only the remaining ~0.5 ms (~10–15 %
  fewer bytes); llama.cpp running the QAT weights as q4_0 would land ≈177 t/s, not 250.

**Would webml's kernels inside llama.cpp beat webml? No — ≈wash or slightly slower**:
3.90 ms floor + llama's ~0.6 ms host ≈ 4.4–4.5 ms vs webml's ~4.0. Concurrency cannot
help (see §4.3). The runtime choice matters for bring-up and iteration speed, not for
steady-state tokens.

### 4.3 The falsified systems theories

| theory | test | result |
|---|---|---|
| "Dawn's serialized dispatch costs ~4 ms" | native replay, serial: 6.78 vs Dawn 7.7 ms | transport worth **~0.7 ms** |
| "concurrency + mem_ranges is llama's edge" | llama's own `GGML_METAL_CONCURRENCY_DISABLE` etc. | serial llama = 145.4 vs 147.6 t/s; reorder ≈ 0; their op-fusion ≈ 0.5 ms. **All scheduling ≲0.6 ms** |
| the 3.43 ms concurrent-no-barrier replay | automatic hazard analysis | **race mirage**: 547/572 of our ops (721/854 of llama's, i.e. *everyone's*) are true dependencies; honest concurrency ≈ serial |
| "op count is the lever (−10 µs/op)" | fusion round 2: fold ropeQ into the norm kernel | −28 dispatches, **±0 ms** (the deleted round-trip is 0.2 MB against a 1.26 GB token) |
| "kernels are at 90–97 % BW" (isolated bench) | cold-stream bench (rotate weight clones past the SLC) | warm figures inflated **13–26 %/shape**; true cold BW 304–465 GB/s, matching the in-token 340 GB/s average |
| "so re-tune for the cold objective" | sweep+refine, cold | winners shift at 3/6 shapes by **≤1–3 %**: per-op BW is a function of **op size** (8 µs over 3.9 MB cannot hide DRAM latency: 304 GB/s; the 226 MB lm-head streams at 465 GB/s) — physics, not parameters |

### 4.4 The remaining per-op budget (per-class profiler)

Replaying each kernel class of our token in isolation (class-sum 6.69 vs whole-token
6.78 ms — consistent attribution):

| bucket | ops | GPU ms | diagnosis |
|---|---|---|---|
| big matvecs | ~146 | ~3.7 | 340 GB/s in-token (62 % peak); recoverable only by *overlapping the next op's weight stream* (§6) — not by tuning |
| attention (grid 8×1×1) | 35 | 0.77 | 8 workgroups on a ~40-core GPU: launch-bound at short context, every engine pays it |
| 1-workgroup norm/add tail | 141 | ~0.70 | GPU essentially idle per op; webml's cure is riding fat kernels' occupancy (epilogue fusion), not fewer boundaries |
| other small ops | ~250 | ~1.6 | same under-occupancy class |

## 5. What the DSL and autotuning actually bought

**Autotune framework** (Family contract ~50 lines/kernel family + shared engine:
sweep, occupancy probe, golden gate, native cold/warm timing, top-K refine with an
incumbent guard, winners.csv runtime lookup — no rebuild to deploy). Delivered:
0.118 s/variant; a 6-shape family sweeps+refines in ~12 s; caught two would-be
regressions (a thermally-skewed winner; a stale ranking) before they shipped; and it
returned honest nulls three times (integrated matmul tuning: noise-level; matvec
family: e2e neutral; cold retune: ≤3 %). **Its limit surfaced quickly and is
structural: parameter search can only recover parameter-shaped losses.** On this
workload the losses are op-size physics, graph shape, and kernel craft — outside the
search space. The framework generalizes (it is salvage), but as a *product thesis*
("autotuning closes the gap to hand-written") it is falsified for M=1 decode.

**Verification/typed-quantization.** The demand side never materialized. The supply
side is ironic: this program paid days to exactly the bug classes a typed layer could
eliminate by construction — 11 clamp-write races (`select`-to-zero does not guard a
write), a writable-storage aliasing bug that batch mode *silently* swallowed,
tensor-dtype assumptions (Q5_K vs Q6_K, BF16-as-F16, F32-as-Q4_K), out-of-bounds
grid-roundup writes. The value is real but it is *development cost*, not a product
moat — and the races happened inside the DSL, i.e., the safety was available in
principle and not enforced in practice.

**The abstraction ledger.** An abstraction layer pays only with ① ownership/
understanding of every layer beneath, ② observability to the bottom, ③ escape
hatches. Hesper built ③ (MSL dump, native bench, the replay harness) and partial ②;
① failed permanently at Tint and Dawn — both foreign, and both bit us (a Tint MSL
printer bug emitting silently-wrong entry points for a 9-binding fused kernel; Dawn's
serial hardcode; Dawn swallowing validation errors in batches). Authoring in an
expression-tree DSL was ~3–5× slower than raw WGSL for kernel work. **jax-metal is the
same disease at a terminal stage**: closed PJRT plugin onto closed MPSGraph, no Pallas
escape hatch — zero of the three conditions, and no way to even build the diagnostic
ladder we used here.

## 6. Open problems

1. **Weight-prefetch overlap** (the legitimate residue of the race mirage): op N+1's
   weight bytes are independent of op N's output; a write-free prefetch dispatch is
   hazard-free and may overlap legally. Prototype cost is low in the replay harness.
   This is the only identified path to the ~1.1 ms cold-stream deficit.
2. Argmax deferral (0.55 ms) — blocked on a pre-existing device-fed-loop bug.
3. Engine: propagate Dawn validation errors in batch mode (silent-drop cost a full
   debugging day); Tint printer bug minimal repro → upstream; Dawn pin is 9 months old.
4. External validity: one box (M4 Max), one model family (E2B), M=1 greedy. Prefill,
   batch>1, and long-context attention change the anatomy (the attention bucket grows
   from launch-bound to bandwidth-bound).

## 7. Conclusion

For M=1 LLM decode on Apple Silicon, the engine is nearly irrelevant and the
scheduler is entirely so: kernels + graph set the floor, dependency chains nullify
concurrency, and every mature host loop costs ≲0.6 ms. A hand-written engine wins by
kernel craft — fat, epilogue-fused, low-precision-operand kernels sized to occupy the
machine — and by nothing else we could measure. A verified DSL could in principle have
eliminated the bug classes that consumed most of our debugging time, but nobody is
buying that in this market, and it does not make tokens faster. The transferable
outputs are the replay methodology (three capture adapters + one comparable harness),
the falsification protocol that repeatedly outperformed our own expert intuition, and
this anatomy itself.
