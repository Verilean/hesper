# DEVPLAN — Hesper measured-JIT rebuild: premises, hypotheses, evidence, verdict

**Status: the product bet is RETIRED (Phase D verdict, 2026-07-06). This document is now
the post-mortem of record.** The verbatim Japanese working log (per-milestone state +
dated decision log, the diary this document distills) is preserved in **`DEVLOG.md`**;
§6 below reconstructs the causal chain from it. Development followed the cycle: read this
file → work a milestone under the principles → update state + decision log → commit code
and DEVPLAN together; ★ items required user review.

---

## §0. Operating principles (kept — they are part of the salvage)

1. **Measure, don't predict.** Performance is not a static property of source; hypotheses
   are settled by sweeps and A/B measurement, never by one-off hunch experiments.
2. **Record speed AND resources** (ms / GFLOPS / %-of-floor plus occupancy / execWidth /
   threadgroup memory).
3. **Golden gates.** A codegen or parameter change that moves maxDiff is a bug; variants
   auto-disqualify.
4. **Isolated win ≠ integrated win.** Adoption requires e2e measurement (decode tok/s),
   never an isolated bench alone.
5. **Measurement hygiene:** no stray processes (`pgrep` == 0), cool box, never `kill -9` a
   GPU process (it wedges Metal); timeout/SIGTERM only.
6. **Record negative results** so no experiment is derived twice.
7. **Reference-driven:** read llama.cpp/webml structure BEFORE designing kernels.
   Imitating a single parameter without the structure fails (measured: WIDE64).
8. **Fusion is a two-level search:** outer loop = op splitting/fusion choice, inner loop =
   parameter sweep; a fusion change invalidates the inner winners.

---

## §1. Premises (as assumed at plan time, 2026-07-05) — and how they fared

| # | Premise | Fate |
|---|---|---|
| P1 | On Metal, WGSL-via-Tint vs hand-written MSL has no decisive per-kernel gap once codegen quality is handled (`let'` CSE); the "Tint tax" was an excuse for unswept tuning knobs. | **HELD.** The llama.cpp `mul_mv_q4_K` port, written in the DSL, reached 90–97 % of memory bandwidth. |
| P2 | Kernels are runtime-generated Lean functions, so parameter sweeps need no rebuild → measured-JIT autotuning is cheap. | **HELD.** 0.118 s/variant; a 6-shape family sweeps + refines in ~12 s with one command. |
| P3 | M=1 decode is bandwidth-bound, so per-kernel BW parity ⇒ e2e parity. | **FALSIFIED.** The binding constraint is the dispatch layer (567 serialized launches through Dawn ≈ 5.7 ms/token), not kernel throughput. Per-kernel parity produced an e2e NEUTRAL. |
| P4 | The market niche is verified WebGPU inference + browser distribution. | **RETIRED** (§4-C1). The browser lane is occupied (webml, MLC/web-llm); verification has no near-term LLM buyer. |
| P5 | The realistic target is llama.cpp on the same box (156.5 t/s measured), not a paper BW estimate. | HELD; used as the M7 yardstick throughout. |

---

## §2. Improvement hypotheses and their fates

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| H1 | A generic autotune framework (Family contract + shared engine + runtime winner lookup) beats per-kernel hand tuning. | **CONFIRMED** | Two families proven (`regGen` WMMA matmul, `mulMvQ4K(f32)` matvec); caught 2 regressions before shipping (thermal false winner, incumbent guard); honest nulls recorded in winners.csv. |
| H2 | Kernel quality is the E2B gap → tune to parity and the gap closes. | **HALF-CONFIRMED** | Isolated parity achieved: llama-port kernels at 90–97 % BW; dp4a was 1.4–1.7 × worse because `dot4I8Packed` is emulated on Apple (no dp4a hardware). But e2e moved only ~+8 % — kernels were not the constraint (see P3). |
| H3 | Fewer dispatches (fusion) closes the remaining gap. | **PARTIALLY CONFIRMED** | Each −35 dispatches ≈ −0.4 ms ≈ +3–4 t/s. PLE-gate fusion and 1-dispatch direct-output attention landed (684 → 567 dispatches, GPU 9.4 → 7.7 ms). Extrapolated ceiling of this line ≈ 105–115 t/s. |
| H4 | llama.cpp's edge is concurrent dispatch → copy it. | **REJECTED for us** | True for them (`MTLDispatchTypeConcurrent` + hazard-only barriers via ggml_mem_ranges), but Dawn hardcodes `MTLDispatchTypeSerial` (crbug.com/425987598). Measured probe: same kernel 13.5 µs/dispatch serial vs 6.9 µs concurrent. Patching Dawn / native transport parked as unnecessary — see H5. |
| H5 | webml's 250 t/s proves the ceiling is reachable **within** Dawn serial dispatch. | **CONFIRMED** | Their decode is ~390 dispatches/token with every norm/add/quantize riding matmul epilogues (`OprojNorm`, `DownNormAdd`, `NormAddNorm`, `GateUpNorm`, SRQ produced by the upstream op, 1-dispatch `DecodeAttention`). Same Dawn, same WGSL — structure, not stack. |
| H6 | The remaining webml gap is mostly their different model (QAT int4 vs our Q4_K_M GGUF). | **REJECTED — the model is the SMALL factor.** Effective read throughput: ours 122 GB/s, llama.cpp 197 GB/s, webml ~279 GB/s, against a 546 GB/s peak — nobody is bandwidth-saturated, so bytes-per-token is not what separates us. QAT int4 reads only ~10–15 % fewer bytes than Q4_K_M; llama.cpp running the same QAT model as q4_0 would gain roughly that (+13 % ≈ 177 t/s), nowhere near 250. QAT's real gift is *quality at low bit-width* (train-time adaptation lets int4 match bf16 quality), not speed. The 250 t/s comes from dispatch count × per-dispatch cost × host-loop leanness. |
| H7 | The DSL itself blocks webml-style optimization ("the abstraction is the bottleneck"). | **NUANCED** (§4-C3) | Expressiveness: no — every webml/llama.cpp trick proved expressible (the ports exist and hit 90–97 % BW). Velocity and visibility: yes — expression-tree authoring is ~3–5 × slower than raw WGSL, and two foreign layers (Tint, Dawn) hide behavior until you drop a level (we had to build MSL dumping and native timing to see anything). |

---

## §3. Evidence ledger (M4 Max, gemma-4-E2B-it Q4_K_M, greedy decode)

**The optimization ladder** — correctness gates ("Paris.", "Jupiter…") held exactly at
every step:

| Step | t/s | What changed |
|---|---|---|
| M6 bring-up done | 8.05 | Correct decode. 4 E2B-specific bugs: PLE token table is Q5_K (not E4B's Q6_K); per_layer_model_proj is BF16; inp_gate/proj are F32; finalNorm prepared-ref captured prefill-sized buffers (the odd layer count exposed it). |
| Authoritative pipeline keys | 39.9 | The WebGPU layer re-generated the full WGSL string per dispatch (~180 µs × 684) just to hash it for the cache. Caller-supplied cacheKey, CUDA-contract mirror. |
| HashMap caches | 64.1 | Pipeline/bindgroup/KCR caches were linear Array scans (~450 k probes/token). |
| Q4_K dp4a lm-head | 68.0 | f16 pre-dequant read 800 MB/token; direct quantized read is 226 MB (and −786 MB VRAM). |
| f32 llama-port matvec winners | 74.8 * | `mul_mv_q4_K_f32` port (f32 input, no quantize dispatch) beat the dp4a path 1.4–1.7 ×; quantize dispatches deleted at attnO/ffn-down/lm-head. |
| PLE-gate fusion + 1-dispatch attention | **89.6–96 steady** | 684 → 567 dispatches; GPU 9.4 → 7.7 ms/token. |

\* 48-token averages were warmup-diluted (first-token pipeline compiles); steady state is
the honest metric — 111-token run: 89.6 t/s, per-token trace 10.1–10.4 ms.

**Per-token decomposition at end state (~10.3 ms):** GPU 7.7 ms ≈ 2 ms of kernel work at
90–97 % BW + ~5.7 ms of serialized dispatch latency (567 × ~10 µs); host loop ~2.0 ms;
argmax readback sync 0.55 ms.

**The three-way comparison that settles H6** (per-token, same box):

| Engine | Bytes read/token | Wall | Effective read BW | t/s |
|---|---|---|---|---|
| Hesper (end state) | ~1.26 GB (Q4_K_M) | 10.3 ms | **122 GB/s** | ~90–96 |
| llama.cpp Metal | ~1.26 GB (Q4_K_M) | 6.4 ms | **197 GB/s** | 156.5 |
| webml | ~1.12 GB (QAT int4) | ~4.0 ms | **279 GB/s** | ~250 |
| (peak, M4 Max) | | | 546 GB/s | |

llama.cpp on the QAT model (as q4_0): ~1.12 GB / 197 GB/s ≈ 5.7 ms ≈ **177 t/s** — the
model swap alone explains ~13 %, not the 250.

**Correctness findings (all fixed; decode is now bit-deterministic across processes,
verified by 5-run md5):**
- Flash-attention split-K with cacheLen < numSplits produced an **empty split** whose
  epilogue aggregated uninitialized threadgroup memory (`40b07ba`).
- **Clamp-write races in 11 kernels:** the pattern `select(inBounds, x, 0.0)` followed by
  an unguarded write — the out-of-bounds *index* clamps onto the buffer's last element and
  races its rightful owner. Smoking gun: ropeDynK (128 pairs in a 256-thread workgroup)
  flipped the K-cache's last word per process (`1e5a145`). Lesson: select-to-zero guards
  the value, not the write.
- **Tint MSL printer bug (open, upstreamable):** a valid-WGSL fused kernel (9 bindings +
  threadgroup arrays) compiles to MSL whose entry point shadows a buffer parameter
  (`v_52`) with a local struct const and drops parameters/freq_factors/threadgroup
  arguments from the inner function call → silent zeros in q heads 4–7 on headDim-512
  layers. Repro: `HESPER_QKNR=1`. Minimal repro + upstream report pending.
- Legacy `executeMatMulTranspose` (f32 path) drops the last output element — documented at
  the golden that caught it; avoid as a reference oracle (use CPU dot).

**Diagnostic kit built (reusable beyond this project):** `HESPER_DUMP_MSL` (per-pipeline
Tint-MSL capture), `mslBenchSerial` (native-Metal kernel timing — Dawn adds ~35 µs/dispatch
that drowns small kernels in a Dawn-side bench), `HESPER_GPUBUSY` (per-token dispatch count
+ GPU wall), `HESPER_DECODE_NOBATCH` + cross-process md5 (determinism probe), `MULMV_DET`
(kernel-isolation determinism), layer-wise golden parity vs llama.cpp eval-callback
(`scripts/llama_parity/scan_layers.py`).

---

## §4. Conclusions

**C1 — The product bet (verified WebGPU LLM inference) is retired, on our own evidence.**
Browser distribution is occupied: webml-community ships a hand-written, single-model,
QAT-int4 engine at 250 t/s, and MLC/web-llm covers the general case. We proved the same
ceiling is *technically reachable* on the same Dawn/WGSL stack — which is exactly why a
latecomer has no edge there. Verified inference has no near-term LLM buyer; its
demonstrated value (robustness-off ≈ 10 %, and killing by construction the bug class we
paid days for — the 11 clamp-write races) is a development-cost story, not a moat.

**C2 — Where the remaining ~2.5 × to webml actually lives** (settling the "is it the
model?" question): dispatch count (567 vs ~390 — epilogue fusion everywhere), per-dispatch
cost (~12 µs vs ~9 µs — runtime leanness above the same Dawn), host loop + argmax sync
(~2.5 ms vs minimal), and only ~10–15 % from the QAT model format. Nothing is
language-level: same WGSL, same Dawn, same GPU.

**C3 — The abstraction verdict.** An abstraction layer earns its cost only with all three:
① ownership/understanding of every layer beneath, ② observability to the bottom,
③ escape hatches. This project built ③ and partial ②; ① fails permanently at Tint and
Dawn (foreign codebases — and both bit us: the MSL printer bug, the serial-dispatch
hardcode). webml and llama.cpp avoid the problem by having no layers they don't see
through. jax-metal fails ①②③ outright (closed PJRT plugin onto closed MPSGraph, no
Pallas-on-Metal) — that is our diagnosis of its slowness, same disease worse stage. A DSL
that is merely a verbose way to write WGSL is strictly worse than WGSL; it pays only via
bug-class elimination by construction (unrealized here — the 11 races happened *inside*
the DSL), composition (epilogue combinators — used at 2 sites), and machine sweepability
(realized — the autotuner).

**C4 — Routing for inference-time learning (TTT) work.** Research belongs in PyTorch, with
fake-quant evaluation from day one (a bf16-validated learning signal may die on a Q4
frozen base). Deployment: server → vLLM (it *is* PyTorch under the hood, and per-session
fast weights map naturally onto its multi-LoRA serving); local/Mac → a llama.cpp fork (a
fixed TTT recipe needs only ~4–6 hand kernels: transposed-quant matvec for dL/dx, adapter
outer products, optimizer step). Nothing about TTT requires Hesper.

---

## §5. Salvage — assets that outlive the bet

1. **The autotune framework** (`Hesper/WGSL/Autotune.lean` + `tune/winners.csv`): a Family
   contract (~50 lines per kernel family) + shared engine (sweep, occupancy probe-prune,
   golden gate, native timing, top-K refine with incumbent guard, persistent winners,
   runtime lookup). Generic GPU-tuning value independent of LLMs.
2. **The GPU-correctness methodology:** the diagnostic kit in §3 plus the hunt patterns —
   bisect by per-layer/per-stage dumps, CPU-continuation of GPU state, cross-process
   determinism as a race detector, the grid-roundup / clamp-write audit checklist.
3. **Upstreamable findings:** the Tint MSL printer bug (minimal repro pending); measured
   serial-vs-concurrent data for Dawn's dispatch limitation (crbug.com/425987598); the
   clamp-write race pattern as a lint rule for any WGSL codebase.
4. **The DEVPLAN method itself** — principles, ★ review gates, negative-result log; the
   process that made this post-mortem cheap and honest to write.

---

## §6. Chronology — the causal chain to the verdict

How each conclusion was actually reached, in order. Each step names its trigger and what
it caused next; dated primary entries are in `DEVLOG.md` §4. All on 2026-07-05/06.

1. **M0 — TAT probe.** Measured 0.118 s/variant → the autotune design is viable.
   Side-discovery: the full-unroll kernel generator explodes Tint/Metal compile at
   K=2816 → the sweep substrate must use runtime K-loops (promoted to a premise).
2. **M2+M4 — first sweeps.** 524 variants, golden 0 fail; claimed 2.7×/1.7× wins.
   **User challenged reproducibility** → re-measurement found ~3× transient outliers in
   single sweep rows → the refine stage (top-10 × 300 iters × 3 reps) was created and the
   wins corrected to **1.45×/1.25×**. A thermally-rotten winner then caused a +39 ms
   decode regression → the **incumbent guard** was created. (Two framework features exist
   *because* of two caught mistakes.)
3. **M5 — first integration: e2e NEUTRAL.** The tuned WGSL matmuls were only ~10 % of the
   diffusion step; a 20 % win there sinks below ±30 ms noise. Principle 4's gate fired as
   designed → redirect to a target where matvec dominates the step: **E2B (M6/M7)**.
4. **M6 — E2B bring-up to correctness, 8.05 t/s.** Four bugs, three of them
   "assumed the tensor type instead of reading it" (PLE table Q5_K, proj BF16,
   inp_gate/proj F32) → new operating rule: read every tensor's type from GGUF, throw on
   unhandled. Fourth: prepared-dispatch captured prefill buffers, exposed only by E2B's
   odd layer count.
5. **M7 — overhead removals, 8.05 → 68.0 t/s.** cacheKey authority (WGSL was re-generated
   per dispatch just to hash it) → 39.9; HashMap caches (linear scans, ~450 k
   probes/token) → 64.1; Q4_K dp4a lm-head (800 → 226 MB/token) → 68.0. No kernel was
   made faster yet — this was all engine tax.
6. **mulMv family sweep → e2e NEUTRAL again → the real constraint found.** All-shape
   winner R1W1 didn't move decode. `HESPER_GPUBUSY` (built for this) measured **684
   dispatches × ~14 µs serialized = 9.5 ms GPU floor** — the constraint is the dispatch
   layer, not kernel interiors. **This falsified premise P3.**
7. **Violation #1 → course correction.** I proposed fusion ("fewer dispatches") *before*
   the M3 reference reading. **User: "did you analyze llama.cpp's kernels?"** M3 reading
   then showed llama.cpp runs *more* ops (1033) faster via `MTLDispatchTypeConcurrent` +
   hazard-only barriers, while Dawn hardcodes Serial → fusion demoted, native-dispatch
   plan B proposed; CONCPROBE derisked it (13.5 → 6.9 µs/dispatch).
8. **Violation #2 → kernel-first ordering.** **User: "the kernels are bad and autotune
   isn't done — why native dispatch?"** Native-encoder timing (Dawn overhead removed)
   showed the tuned R1W1 at only **43 % BW** — the "kernels are same-class" claim had been
   structural, not measured. Order fixed: kernels first.
9. **The principle-7-compliant port.** llama.cpp `kernel_mul_mv_q4_K_f32` ported
   *structure-whole* into the DSL → **90–97 % BW**, beating the dp4a family 1.4–1.7×
   (`dot4I8Packed` is emulated on Apple). Deployed: +8 %, 74.8 t/s. Honest decomposition:
   kernels now ~2 ms; the ~6.5 ms dispatch tax dominates → back to the dispatch layer,
   but now with clean hands.
10. **Violation #3 → fusion reinstated, plan B parked.** **User: "webml is WebGPU — why
    250 t/s?"** (webml was required reading in the original plan's A-0.) Reading it:
    ~390 dispatches/token via epilogue fusion, *inside the same Dawn serial dispatch* →
    H5 confirmed, native transport unnecessary, fusion is the road.
11. **Fusion batches + the race hunt.** PLE-gate fusion (−35) landed; qkNormRope fusion
    hit the **Tint MSL printer bug** (valid WGSL → broken MSL, parked). Its gating
    required bit-determinism, which was broken — hunting that exposed two *pre-existing*
    bugs: the empty split-K aggregating uninitialized threadgroup memory, and **11
    clamp-write races**. 1-dispatch attention → 567 dispatches, GPU 7.7 ms.
12. **Metric correction.** The "~75 t/s" figures were 48-token averages diluted by
    first-token pipeline compiles; steady state = **89.6–96 t/s**.
13. **The closing decomposition → verdict.** User asked whether webml's edge is the QAT
    model. Effective-BW table (§3): 122/197/279 GB/s; llama.cpp on QAT-q4_0 ≈ 177 t/s →
    H6 rejected; the gap is dispatch structure + runtime leanness. With P3 falsified, P4's
    market premise examined, and the ceiling shown to belong to hand-written incumbents,
    **user verdict: retire the product bet** → this post-mortem.

## §7. Process lessons (the cost accounting)

The core hypothesis was falsified *cleanly*: the 8.05 → 90 t/s ladder isolated "not the
kernels, not the language — the dispatch structure" one variable at a time. But the same
conclusion was reachable weeks earlier: **principle 7 was violated three times in one
session** — a fusion plan proposed before reading llama.cpp's kernels; a native-transport
plan proposed before kernel work was finished; and webml (listed in the original plan's
A-0 as required reading) actually read only after the user asked "why is webml at
250 t/s?". Reference reading is cheaper than every experiment it replaces. That, plus
metric hygiene (steady-state vs warmup-diluted averages), is what the next project
inherits on day one.
