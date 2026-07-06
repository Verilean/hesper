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

---

## §8. Post-verdict validation: Experiment 2 (native Metal transport), 2026-07-06

The user's challenge after the verdict: *"if the conclusion is right, removing the
abstraction layer must produce a near-fastest engine."* Implemented and measured
(commits `ff2286a`, `7798b56`); the test both **validated the direction and corrected
C2's magnitude**.

**Phase A (timing replay).** All 572 dispatches of one decode token captured (Tint-CLI
MSL, Dawn-backed MTLBuffers, 0 misses) and replayed natively, identical kernels:
serial 6.78 ms (≈ Dawn's 7.7 — sanity), concurrent no-barrier 3.43 ms, per-layer
barriers 3.92 ms.

**Phase B (real execution) — and the correction.** Automatic hazard analysis
(llama-style, whole-buffer) inserts **547 barriers over 572 ops**: the decode dataflow
is a **width-1 dependency chain** — our own fusion (fused QKV, fused gate/up) removed
the q/k/v & gate/up parallelism llama.cpp still has. So the 3.43 ms was a **race
mirage**; honest concurrency ≈ serial. `HESPER_NATIVE_DECODE=1` runs every decode
token natively on Dawn's own MTLQueue (key-indexed PSO + cached binding permutation →
steady-state record cost is a u64 probe): **token-sequence-identical to the Dawn path**
over 32 tokens, total 9.15 ms/token vs Dawn 9.8 ms (+7 %).

**Revised decomposition (native serial, steady):** GPU 7.5 ms = ~2 ms kernel work +
**~5.5 ms of inter-dispatch gaps (~10 µs/op even natively)**. The dispatch layer's
recoverable waste was ~0.7 ms (Dawn encoder overhead), not ~3.8 ms; the dominant term
is the **op count itself** — every op removed ≈ −10 µs. C2 stands in direction
(nothing is language-level; transport + structure), but the actionable lever is
**fusion** (572 → ~390 ops ≈ −1.8 ms) plus host trim and argmax deferral — the webml
lesson re-derived from our own hardware counters. llama.cpp's 6.4 ms *total* remains
the yardstick; the road there is op-count, not concurrency.

Side finds: gemma4-inference dies at startup on WebGPU unless `HESPER_CUDA_GRAPHS=0`
(graphs default-on hits the CUDA stub); FFI error objects must be
`lean_mk_io_user_error`-wrapped or the top-level error printer segfaults; Tint renames
entry points (`main` → `v`) in MSL.

---

## §9. Controlled cross-engine experiment: PROVE the "concurrency + mem_ranges" claim
### (predictions registered 2026-07-06 BEFORE measurement — principle 1)

§8 left the llama.cpp side of the story asserted, not proven ("their 6.4 ms comes from
kernels + Concurrent + range-granularity hazard tracking hiding the ~10 µs/op gaps on a
width-2–3 graph"). llama.cpp ships its own ablation toggles
(`GGML_METAL_CONCURRENCY_DISABLE`, `GGML_METAL_FUSION_DISABLE`,
`GGML_METAL_GRAPH_OPTIMIZE_DISABLE`), so the claim is testable on their engine with
zero instrumentation. Combined with our Phase B numbers this completes a 2×2:
graph {ours width-1 fused, theirs wide} × schedule {serial, concurrent+hazard}.

**Predictions (E2B Q4_K_M, M4 Max, greedy decode steady state):**

| # | Config | Prediction | Reasoning |
|---|---|---|---|
| P1 | llama default (conc+fusion+reorder) | ~156 t/s (≈6.4 ms) | prior measurement reproduces |
| P2 | llama + CONCURRENCY_DISABLE | **90–115 t/s (8.5–11 ms)** — the key test | serial exposes ~10 µs/op × their op count; drops BELOW our native-serial 9.15 ms because they run more, finer ops |
| P3 | llama + GRAPH_OPTIMIZE_DISABLE | 140–155 t/s (small hit) | reordering only widens the overlap window; ranges still hide most gaps |
| P4 | llama + FUSION_DISABLE | 120–145 t/s | more ops, partially hidden by concurrency |
| P5 | (ours, measured §8) our graph serial vs hazard-concurrent | 7.5 vs 7.0–7.7 ms — already ≈ equal | width-1 chain: nothing to overlap |

If P2 lands (a large serial penalty on THEIR graph) while P5 stands (no concurrency
penalty/gain on OURS), the mechanism is proven: **llama.cpp's speed = per-op gaps
hidden by concurrent overlap on a graph that kept its width; ours kept nothing to
overlap because we fused the width away.** Then the fusion-lever estimate (−10 µs/op)
inherits that proof, and fusion round 2 becomes the justified last step.

Follow-up (webml quadrant): replay webml's 44 WGSL kernels (already extracted,
refs/webml-gemma4/wgsl/) with their ~390-dispatch structure through the same harness —
prediction: serial replay lands ≈ 390 × ~10 µs + kernel work ≈ **~4.5–5.5 ms**,
matching their ~4 ms/token and proving the op-count lever independently.

### §9 RESULTS (2026-07-06, llama-bench tg64, r=3, same box) — the theory is FALSIFIED

| # | Config | Predicted | Measured | Verdict |
|---|---|---|---|---|
| P1 | default | ~156 t/s | **147.6 ± 3.1** (6.78 ms) | ≈ reproduced (tool/box delta vs the older 156.5) |
| P2 | CONCURRENCY_DISABLE | 90–115 t/s | **145.4 ± 2.8** (6.88 ms) | **FALSIFIED** — concurrency buys ~0.1 ms, not 2–4 ms |
| P3 | GRAPH_OPTIMIZE_DISABLE | 140–155 | **145.4 ± 3.0** | held, trivially (effect ≈ 0) |
| P4 | FUSION_DISABLE | 120–145 | **137.6 ± 2.8** (7.27 ms) | held — their op-fusion is worth ~0.5 ms |
| P4b | both off | — | **141.7 ± 3.0** | ≈ P4 within noise |

**What this kills and what it establishes:**

1. **"llama.cpp's speed = Concurrent + mem_ranges" is DEAD.** Their entire scheduling
   apparatus (concurrency, reordering, op-fusion) sums to ≲0.6 ms. Fully-serial
   llama.cpp still does ~142–145 t/s.
2. Symmetric with our P5 (our serial ≈ our hazard-concurrent): **on this hardware,
   LLM-decode concurrency is irrelevant in BOTH engines.** Phase A's 3.43 ms
   no-barrier number was doubly a mirage — it raced dependencies AND measured a
   schedule no correct engine can use.
3. Apples-to-apples serial totals: **llama ~6.9 ms vs our native 9.15 ms — and they
   run ~2× our op count** (~1033 vs 572). The "~10 µs/op universal gap" model of §8
   is WRONG: their per-op average ≈ 6.5 µs including kernel work; ours ≈ 13 µs.
   Effective decode read-BW: ours 1.26 GB / 7.5 ms ≈ 168 GB/s vs theirs ≈ 194 GB/s.
4. ⇒ The remaining deficit is NOT transport (settled §8: ~0.7 ms), NOT scheduling
   (this table: ~0), NOT op count per se (they have more) — it is **average per-op
   efficiency**: kernel execution + per-op tail effects (grid/threadgroup shapes,
   drain), plus our ~1.1 ms host and 0.55 ms argmax.
5. **Fusion re-priced before round 2** (user gate: "only if proven"): llama's own
   op-fusion = 0.5 ms over ~1000 ops; our one measured marginal (PLE fusion, −35
   dispatches) = −0.4 ms ≈ 11 µs/op. A −80-op round 2 projects ≈ −0.9 ms — real but
   not decisive alone.

**Next diagnostics (before touching fusion):** (a) bisect our 7.5 ms serial GPU by op
class (grid-size filter in the replay harness) — find where the 13 µs/op average
lives: the big matvecs (are they really 90–97 % BW *in-token*, streaming cold weights?)
vs the small-op tail; (b) the webml quadrant: replay their 44 kernels / ~390-dispatch
token — if it lands ≈ 4 ms serial, fat-kernel efficiency (not count) is what they prove.

### §9b. The cross-replay (user-directed): llama.cpp's kernels in a bare harness

The user cut through: *"just replay llama.cpp's kernels in our harness."* Implemented
INSIDE the local fork (ggml-metal-device.m: capture the encoder command stream of the
GGML_METAL_REPLAY=<n>-th encoder, re-encode 20× on a private queue, GPU-time it —
same methodology as our Phase A; local fork only, never upstream).

llama.cpp E2B decode token = 2 encoders (84 + 770 = 854 dispatches):

| graph+kernels | ops | serial GPU (min) | as-recorded concurrent (min) | barriers |
|---|---|---|---|---|
| llama.cpp | 854 | **6.31 ms** (5.82 + 0.48) | **5.92 ms** | 665+56 of 854 |
| Hesper (§8, same method) | 572 | **6.78 ms** | ~7.0 (mode-3 hazard) | 547 of 572 |

**Findings:**
1. llama's decode is ALSO ~86 %-barriered under their own mem_ranges — everyone's
   decode is a dependency chain; their concurrency is worth ~0.4 ms (6 %), matching
   the §9 ablation. The "wide graph they kept" story is dead too: nobody has
   meaningful width.
2. Same-harness kernel+graph comparison: **854 ops in 6.31 ms (7.4 µs/op) vs our
   572 ops in 6.78 ms (11.9 µs/op)** — their per-op average is ~1.6× better; more,
   smaller ops that each cost less. The GPU-side deficit of kernels+graph is
   **~0.5 ms (serial) to ~0.9 ms (their concurrent vs our serial)** — real but far
   smaller than any previous theory claimed.
3. Reconciling totals (llama ~6.9 ms/token vs our native ~9.15): GPU kernels+graph
   ≈ 0.5–1.3 ms (above, plus our steady-vs-min gap), **host machinery ≈ 1.0–1.2 ms**
   (our per-dispatch record path + argmax readback 0.55 vs their lean C++ loop).
   After transport (§8), scheduling (§9), and now kernels+graph (§9b) are all
   measured, the single biggest UNexplored lever on our side is the host loop and
   argmax, followed by per-op kernel efficiency parity.

Instrumentation kept in the fork (env-gated, off by default):
`GGML_METAL_REPLAY_LOG=1` (per-encoder dispatch counts), `GGML_METAL_REPLAY=<n>`
(capture+replay encoder n). Deadlock note: llama pre-enqueues its CBs, so the replay
must commit on a PRIVATE MTLCommandQueue.

### §9c. The webml quadrant (user-directed): their real token replayed, prediction HELD

Method (new, reusable): mirror the HF space locally, monkey-patch the WebGPU API in a
driver page (createShaderModule/Pipeline/BindGroup/Buffer, setPipeline/BindGroup,
dispatchWorkgroups, PLUS a ≤64 KiB buffer-content mirror via writeBuffer/mapped-range
hooks — uniform values drive loop trip counts, zeros would fake the timing), run the
real app in headless Chrome (`--headless=new --use-angle=metal`, WebGPU works; model
2.34 GB QAT int4 cached in the profile), trace exactly one steady decode token, POST
the trace to a local collector; then tint each pipeline (`--overrides` for constants;
all 53 compiled — including subgroup-matrix kernels) and replay in a ~150-line
standalone Metal tool with synthesized buffers + real uniform contents.
Tooling + full HOWTO for all three replay methods (Hesper env-gated harness,
llama.cpp fork patch, browser WebGPU trace): **`tools/replay/README.md`**.

**Result (2026-07-06):** webml decode token = **316 ops, 53 pipelines, serial GPU
3.90 ms min (4.47 avg)** — §9's pre-registered prediction (4.5–5.5 ms ≈ their real
~4.0 ms/token) HELD at the fast edge. Concurrent-no-barrier 1.85 ms (same race-mirage
caveat as ever).

**The completed three-engine table (same bare-Metal harness, serial, one decode token):**

| engine | ops | serial GPU | µs/op | bytes/token | effective BW |
|---|---|---|---|---|---|
| webml | 316 | **3.90 ms** | 12.3 | ~1.12 GB | **287 GB/s** |
| llama.cpp | 854 | 6.31 ms | 7.4 | ~1.26 GB | 200 GB/s |
| Hesper | 572 | 6.78 ms | 11.9 | ~1.26 GB | 186 GB/s |

**Reading:** op count per se predicts nothing (llama: most ops, second-fastest;
webml: fattest per-op cost, fastest total). What separates the engines is **total
kernel work = bytes moved × kernel efficiency**. Normalizing webml to our byte count
(×1.26/1.12) gives ≈ 4.4 ms — so ≈ **2.4 ms of our 6.78 is kernel-shape/fusion
quality on equal bytes**, the single largest measured lever left. webml earns it with
epilogue fusion (intermediates never round-trip through memory) and int8-SRQ
activations; llama earns its 200 GB/s with many thin, highly-tuned kernels.

**Fusion verdict (user gate satisfied):** fusion round 2 is JUSTIFIED — not as
"fewer dispatch boundaries" (§9 killed that) but as **removing intermediate memory
round-trips**, webml-style. Expected first-tranche win −0.5…−1.5 ms of the 2.4 ms
pool (Q6K-V merge, argmax deferral, factors-less qkNormRope), plus kernel-efficiency
parity work toward llama's 200 GB/s.

---

## §10. Host-cost kill (a): frozen-token decode — 9.8 → 8.4 ms/token, token-exact

Insight: from cacheLen ≥ 8 the decode dispatch sequence is **token-invariant** —
identical buffers, grids, kernels; only params-buffer CONTENT changes. So the native
list recorded once replays verbatim every token (`HESPER_NATIVE_FROZEN=<m>`, the
CUDA-Graphs analogue), skipping both Dawn and the Lean walk of `forwardSingleToken`.

Two instructive bugs on the way (both found by the token-exactness gate,
commit `315c8e0`):
1. **Dawn's queue.writeBuffer is staged, not submitted** — the frozen seeds
   (pos/cacheLen/posF32/token) sat host-side while the native CB ran. Fix: replay_exec
   issues an empty `Queue::Submit` first; same-queue FIFO orders the staging copies
   ahead of the CB. (The walk had masked this via its own batch submits.)
2. **`gpuArgmax` writes plRawRowBuf = tokenId every iteration** (deviceFed plumbing);
   the walk silently overwrote it with 0, which the row-staged PLE dequant requires.
   The frozen path must restore the 0. Divergence was deterministic and started
   exactly at the first frozen token — the gate localized it in two bisections.

Per-token host work is now: 5 small seed writes + the token's 6160 B PLE row + the
argmax readback. **Ladder: Dawn 9.8 → nativeExec 9.15 → frozen ~8.4 ms/token
(~119 t/s), token-sequence-identical.** Remaining (a)-item: argmax readback 0.55 ms
(deferral machinery exists but HESPER_DEVICE_FED is independently broken — zeros from
its 9th token even on the plain Dawn path; needs its own fix first).

Answering "is webml's loop path also fast?": yes — their real total (~4.0 ms) minus
replayed GPU (3.90 ms) leaves ≲0.3 ms of JS host per token; Chrome encodes via IPC to
the GPU process (overlapped), bind groups are prebuilt, and their loop does almost
nothing else. Ours was slow because of the per-token Lean walk (~1.3 ms) + record
path — which frozen mode now eliminates.

---

## §11. Fusion round 2, first tranche (c): factors-less qkvNorm+ropeQ — NEUTRAL

Target chosen by census (572 ops: rope 35 + ropeTail ~12 were the largest single-op
cluster): fold rope-Q into the per-head norm kernel on the 28 factors-less (SWA)
layers. The 8-binding variant (no freq_factors) dodges the Tint MSL printer bug that
parked the 9-binding FULL variant. Commit `f02b207`, opt-in `HESPER_QKNR_SWA=1`.

**Result: −28 dispatches, GPU time UNCHANGED (frozen-native 7.51 ms vs ~7.5 ms).**
§9c's −0.5…−1.5 ms projection for this tranche is FALSIFIED for small fusions: the
deleted qBuf2 round-trip is 0.2 MB/token against a 1.26 GB token — op count and small
round-trips are not the lever; only byte-heavy intermediates would pay (and our
biggest intermediates are already fused). The variant's rope rounding also flips
near-tie tokens (poem diverges at token 24; "Paris." holds), so default stays OFF for
output reproducibility. Remaining fusion candidates (Q6K-V merge ≈ 17 ops of the same
small class) are now expected NEUTRAL by the same arithmetic — deprioritized.

**Two real finds shipped with it:**
1. **Writable-storage-aliasing latent bug**: the KV-shared placeholder bound
   qNormWeight to BOTH q_scale and k_scale. The DSL declares every storage buffer
   `read_write`, so Dawn rejects the dispatch (writable aliasing). The pre-existing
   freq variant (HESPER_QKNR) had the same bug since it landed. Fixed in both.
2. **Batched decode SWALLOWS dispatch validation errors** — the invalid dispatch is
   silently dropped and decode degenerates with no message (this cost the whole hunt:
   layer-bisect → "on == RAW input, the kernel never ran" → `HESPER_DECODE_NOBATCH=1`
   finally surfaced the validation text). Engine TODO: propagate Dawn validation
   errors in batch mode.

**Where this leaves the ladder:** Dawn 9.8 → native 9.15 → frozen 8.4 ms/token
(~119 t/s), fusion r2 neutral. Remaining honest levers, in measured-size order:
per-op kernel efficiency toward llama's ~6.5 µs/op average (our small-op tail),
argmax deferral (0.55 ms, blocked on the deviceFed fix), then structural ideas.

---

## §12. The per-op efficiency residual, decomposed (per-class GPU profiler)

New instrument (`a99ab47`): `HESPER_NATIVE_PROFILE=1` replays every kernel class of the
recorded token in isolation (serial, min-of-30) — a complete GPU-time budget by class.
Class-sum 6.69 ms vs whole-token serial 6.78 ms → the attribution is consistent.

**The 6.8 ms serial GPU budget (572 ops), aggregated:**

| bucket | ops | GPU ms | note |
|---|---|---|---|
| big matvecs (QKV/oProj/gate-up/down/lm-head) | ~146 | **~3.7** | effective ~340 GB/s = **62 % of peak** in-token — NOT the 90–97 % of the isolated bench (which re-reads warm weights 300×; a real token streams every weight cold, once) |
| attention (grid 8×1×1!) | 35 | **0.77** | 22 µs/layer with EIGHT workgroups on a ~40-core GPU — launch-bound and >80 % idle at short context; a short-ctx tax every engine pays |
| single-workgroup norm/add tail (grid 1×1×1) | 141 | **~0.70** | 2.7–5.4 µs each, one workgroup = GPU essentially idle; postAttn/postFFN norms, PLE post-adds |
| rope/kv/PLE-gate/misc small ops | ~250 | **~1.6** | same underoccupancy class |

**The residual, named.** After transport (§8), scheduling (§9), op count (§9b/§11) all
measured ≈ null, "per-op efficiency" resolves into three concrete items:
1. **In-token matvec streaming at 62 % of peak** (~1.1 ms recoverable toward 90 %):
   the isolated-bench 90–97 % figure was warm-cache optimistic — the honest target is
   cold-stream BW. This is the single biggest kernel-side item and is autotune-able
   (the harness must stream cold weights to measure it, i.e. rotate weight buffers).
2. **~400 small ops that can't occupy the GPU** (~2.3 ms total): each op is
   individually "fine" but 1-to-8-workgroup dispatches leave the machine idle.
   webml's answer is not fewer dispatch *boundaries* (§11: neutral) but making these
   ride the fat kernels' occupancy (epilogue fusion into ops that already fill the
   GPU) — worth it ONLY where the small op runs between two big ones.
3. **Attention at decode length is launch-bound** (0.77 ms): grid = numHeads. A
   cross-layer or Q-batched attention shape would be needed to occupy; llama.cpp pays
   the same class of cost here.

**Answer to "would webml's kernels inside llama.cpp beat webml?" — NO, ≈ wash or
slightly slower.** The bare-harness floor of webml's kernels+graph is 3.90 ms (§9c).
A runtime adds only its host cost on top: webml's JS loop adds ~0.1–0.3 ms (real
in-browser ~4.0 ms), llama.cpp's C++ loop adds ~0.6 ms (6.9 total − 6.31 GPU; ggml
graph build/schedule per token) — so webml-kernels-in-llama ≈ 4.4–4.5 ms ≈ 220 t/s vs
webml's ~250. llama's concurrency machinery can't help (the webml graph is 316 chained
fat ops — nothing to overlap, §9). The general law all four experiments converge on:
**kernels+graph set the floor; every mature runtime (JS-in-Chrome, C++ ggml, our
Lean+frozen-native at 0.9 ms) is a ±0.5 ms rounding term on top.** Where the runtime
DOES matter is bring-up and iteration speed, not steady-state tokens.
