# Canonical document: hesper vs llama.cpp host-overhead breakdown

**This is the single source of truth.** Earlier docs (51, 53, 54, 55, 56)
hold partial views and conflicting conclusions. **They are superseded by
this file.**

The investigation has been restarted from zero too many times because
the same numbers were re-interpreted in incompatible ways. The rules
below freeze the methodology; the next time someone asks "where is the
8 ms gap?", the answer is in §3 "facts and what they mean".

---

## 0. Things we keep saying that are NOT the answer

Every previous session has ended with one of these claims, and every
one has been refuted by measurement at least once. Do not propose them
again without first reading §1 "measurement protocol".

| Claim | Why it's wrong |
|---|---|
| "Need more kernel fusion" | Per-layer kernel **count** already matches llama.cpp. Doc 56 phase 1 done. Adding fusion doesn't shrink the wall any more — kernel-count parity reached. |
| "Per-kernel GPU time is the gap" | False at our current state. nsys shows hesper kernel total = **10.9 ms/tok**, llama-cli = **8.4 ms/tok**. Difference 2.5 ms accounts for *part* of the wall gap (8 ms total) but **not the dominant part**. |
| "Lean GC / refcount tail latency causes 5 ms" | Lean is reference-counted, not GC'd. There are no tail-latency stalls of multi-ms scale unless something allocates a multi-MB object every token. perf cycles do show `lean_dec_ref_cold + lean_copy_expand_array` at ~14 % of CPU cycles, but a CPU-cycle share is not the same as a wall-time share — see §3. |
| "DtoH 9.8 ms is a copy cost we can remove" | False. Tested with `HESPER_DEVICE_ARGMAX=1`: the 9.8 ms moves from `cuMemcpyDtoH` to `cuStreamSynchronize` byte-for-byte. The wait is a **stream drain**, the API row name is the only thing that changes. |
| "Drain just renames itself, kernel speed is the only lever" | False — see §3. The drain wait is bounded by kernel time, but **on top of** the drain, hesper has a measurable host CPU window per token (5 ms steady-state) that is sequential with the drain instead of overlapping it. THIS is the actual remaining lever. |

If the next thing you want to type is one of the above, **stop and
re-read §3 first.**

## 1. Measurement protocol (freeze this)

To compare host overhead vs llama-cli we need the same prompt, same
decode count, same flags, same metric on both engines. The script
`scripts/perf_compare.sh` is the only sanctioned way:

```bash
rm -rf /dev/shm/perf_compare
PROMPT_LONG="The capital of France is and the population is" \
P_LONG_TOK=11 N_TOKENS=300 \
HESPER_USE_MMAP=1 GRAPHS=off \
NSYS_TIMEOUT=180 PERF_TIMEOUT=180 \
bash scripts/perf_compare.sh both
```

What it does (do NOT diverge from this):
- Prompt: 11 tokens, decode: 300 tokens, deterministic greedy.
- hesper: `HESPER_IGNORE_EOS=1 HESPER_DP4A=1` (so decode runs the full N).
- llama-cli: `--single-turn --no-warmup --temp 0 -ngl 99 -c 4096`.
  **Do NOT add `--ignore-eos` or `--jinja`** — they re-enable
  interactive mode and produce 1.9 GB of `> ` lines (doc 56).
- nsys: `--trace=cuda --sample=none --stats=false`. Output to `/dev/shm`
  (tmpfs) so post-process doesn't fsync to disk.
- nsys orphan agents are killed by an EXIT trap. Failures to do this
  in past sessions caused 6× CPU contamination.
- Single-case (pLgN only) — not a 3-case linear regression. The 1-token
  cases are post-process-bound, not GPU-bound, so the regression slope
  was noise.

Per-token instrumentation:

```bash
HESPER_DECODE_SECT_TRACE=1 HESPER_IGNORE_EOS=1 \
HESPER_DP4A=1 HESPER_USE_MMAP=1 \
lake exe gemma4-cuda data/gemma-4-e4b-it-Q4_K_M.gguf "..." 30
```

prints `[sect] tok=N argmax=… forward=… total=…` lines. **Steady state
is tok ≥ 4** — the first few tokens have one-time JIT warmup costs
(modload, fused-norm cache miss).

## 2. Architecture map (re-derive once, cache here)

```
 ┌── decode iteration: ──────────────────────────────────────────────┐
 │                                                                   │
 │   ▼ host: read previous-token logits / argmax result              │
 │   "argmax" section (HESPER_DECODE_SECT_TRACE)                     │
 │     - cuStreamSynchronize: drain any pending GPU work             │
 │     - read 4 bytes from argmax buf (or host-mapped slot)          │
 │                                                                   │
 │   ▼ host: prepare next token forward                              │
 │   "forward" section                                               │
 │     - writeScalarViaStaging: tokenBuf, paramsBuf, posF32Buf       │
 │     - issue all kernel launches for the forward (→ stream)        │
 │     - return immediately; kernels run on GPU                      │
 │                                                                   │
 │   end of iteration; next iteration's "argmax" section drains      │
 │                                                                   │
 └───────────────────────────────────────────────────────────────────┘
```

llama-cli arrangement is the same shape *except* it pipelines: by the
time `cudaStreamSynchronize` is called, the host has *already* started
queueing the next forward's launches into the same stream — they sit
behind the in-flight kernels and run as soon as the GPU finishes the
previous forward. So "forward host work" overlaps with "previous
forward GPU work".

## 3. Facts (numbers and what they mean)

### 3a. Per-token wall budget — measured 2026-04-26

| section / metric | hesper | llama-cli | comment |
|---|---:|---:|---|
| **Wall / token** | 15 ms | 9 ms | hesper 66 TPS, llama-cli 110 TPS |
| GPU kernel time (sum) | 10.9 ms | 8.4 ms | nsys cuda_gpu_kern_sum |
| GPU memcpy time (sum) | 0.7 ms | 0.7 ms | nsys cuda_gpu_mem_time_sum |
| GPU busy total | **11.6 ms** | **9.0 ms** | kernel + memcpy |
| `cuStreamSynchronize` (host wall in API) | 9.8 ms | 6.8 ms | the drain wait |
| `cuMemcpyDtoH` (host wall in API) | 9.8 ms (→ 0 with HESPER_DEVICE_ARGMAX=1) | 0 ms | implicit drain via DtoH |
| `cuLaunchKernel` (host wall in API) | 1.1 ms | 1.8 ms | hesper actually does *fewer* host launch calls |
| `HESPER_DECODE_SECT_TRACE` argmax | ~10 ms | (not measured) | = drain wait on previous forward |
| `HESPER_DECODE_SECT_TRACE` forward | **~5 ms** | (not measured directly; inferred ≤ 0 over kernel) | host launch path |

### 3b. Decomposition of the 8 ms wall gap

```
hesper wall  = 15 ms = 10.9 ms (GPU kernel)        +  4.1 ms (other)
llama wall   =  9 ms =  8.4 ms (GPU kernel)        +  0.6 ms (other)
gap          =  6 ms =  2.5 ms (kernel speed gap)  +  3.5 ms (host overlap gap)
```

The "other" component on hesper splits in two:
- ≈ 0.6 ms tail of memcpy + per-launch driver overhead that can't be
  shaved further (matches llama.cpp's residual);
- ≈ **3.5 ms host CPU window** that should run during the drain but
  doesn't, because the host issues `forward (5 ms)` work *after* the
  drain finishes instead of before.

### 3b.1. Forward subsection breakdown (HESPER_FWD_SUBSECT_TRACE=1, 2026-04-26)

Inside the 5 ms `forward` section (steady-state, tok ≥ 4):

| sub-section | host CPU / token | comment |
|---|---:|---|
| `prePLE` (embed lookup + embed scale + perLayerInputPre) | **~0.06 ms** | negligible |
| `blocks` (42 transformer blocks, `forwardBlock`) | **~4.13 ms** | dominant |
| `post` (finalNorm + lmHead + softcap) | **~0.78 ms** | small but real |
| total | **~4.96 ms** | |

4.13 ms ÷ 42 blocks ≈ 98 µs / block of host CPU per `forwardBlock`.
A block emits ~20 dispatches, so a naive split gives "5 µs / dispatch".

**That naive split was wrong.** Dispatch counts on hesper and llama.cpp
are within a few percent of each other (per doc 240); if "5 µs / dispatch"
were the dispatch-rate cost, llama.cpp would also pay 5 ms / token of
forward host time and we would not see it run graphs-OFF at >100 TPS.
It does. So the per-dispatch share is similar between the two engines —
something **hesper-specific**, not present (or not as large) in llama.cpp,
sits inside the 4.13 ms of `blocks`. Candidates worth measuring before
acting on any of them:

- `Hesper.Circuit.CompiledCircuit.replay` builds/expands an args Array
  per dispatch even on cache hits; the pattern is hesper-specific.
- `forwardBlock` does per-layer `kcr.getRef` (hash + IO.Ref lookup) in
  ~20 places; the cumulative cost per block is plausibly several µs and
  has no llama.cpp analogue.
- `withSection` push/pop bookkeeping adds string-keyed work per section.
- Lean refcount slow paths (`lean_dec_ref_cold`) hit when shared `Array`
  values cross IO boundaries; perf shows ~7.6 % of *cycles* but the
  question is which fraction of *wall* lands inside `blocks` vs the
  overlapping argmax drain.

**Implication for H4**: do NOT decompose the 5 ms again as "X dispatches
× Y µs each" without first taking the same `forward host time / token`
on llama-cli for comparison. Comparable per-dispatch counts force the
conclusion that the gap is per-block work that hesper does and llama.cpp
doesn't.

### 3b.2. Graphs ON measurement (H4a verdict, 2026-04-26)

`HESPER_CUDA_GRAPHS=1` captures the entire forward into a single CUDA
graph and replays it as one `cuGraphLaunch` call per token. This collapses
all per-dispatch host overhead (the 5 µs × 880 dispatches we just measured)
into a single driver call, AND lets the host return immediately while the
GPU works through the graph — which is the exact "host work overlaps with
GPU drain" structure llama-cli uses.

| metric (steady-state, tok ≥ 4) | graphs OFF | graphs ON | delta |
|---|---:|---:|---:|
| `argmax` section wall | ~10 ms | **0.005 ms** | -9.99 ms |
| `forward` section wall | ~5 ms | **10 ms** | +5 ms |
| `total` section wall | ~15 ms | **10 ms** | -5 ms |
| TPS (80-token, prompt "Explain ...") | 59 TPS | **80 TPS** | +21 TPS |
| TPS (steady-state extrapolation) | ~67 TPS | **~100 TPS** | +33 TPS |

What changed:
- The `forward` section now contains the cuGraphLaunch + drain wait
  (cuStreamSynchronize) — these two were separate sections in graphs-OFF.
- The `argmax` section is just a 5 µs `cuReadPinnedU32` from the
  host-mapped slot (no driver call), because by the time we reach it the
  drain has already happened inside `forward`.
- Total wall ≈ GPU kernel time (10.9 ms), as it should be when host work
  is fully overlapped with the GPU.

**This invalidates `project_cuda_graphs_status.md` ("0 % speedup on CUDA
backend")** — that conclusion was either from a buggy capture path or a
flawed measurement. Doc 57's protocol gives the +20-30 TPS we expect from
H4 in the canonical decomposition.

### 3b.3. Stub-forward bench result (H4b verdict, 2026-04-26)

`Examples/Gemma4StubDecodeBench.lean` runs `forwardTokenStubPerLayer` —
714 dispatches/token of **no-op stub kernels** (1×1×1 workgroup with a
trivial body), in a 200-token loop. The shape (number of dispatches per
token) matches the real `forwardSingleToken` to within 20 %, but the
GPU side contributes essentially zero. The wall is therefore the
**host-side floor** for hesper's current dispatch path.

| run                                        | dispatches / tok | wall / tok | "TPS" |
|---|---:|---:|---:|
| stub forward, **no-kcr** (cold path each time) | 714 | **5.33 ms** | 188 |
| stub forward, **kcr** (cache-hit path)      | 714 | **1.24 ms** | 808 |
| real forwardSingleToken (HESPER_DECODE_SECT_TRACE 'forward') | 880 | ~5.0 ms | (~60 TPS overall) |

Two facts the bench exposes:

1. **The host floor with the cache hit (1.24 ms) is ~4× lower than the
   real forward's 5 ms.** The real forward has the same 880 dispatches
   on the same machine, yet pays cold-path cost. **Something in the real
   path is missing the cache.** That ~3.8 ms is recoverable without
   any GPU work.

2. **The cold-path cost (5.33 ms) is essentially identical to the real
   forward's 5 ms.** This eliminates dispatch *count* as the cause —
   the per-dispatch host work itself is comparable when not cached.
   The gap to llama.cpp graphs-OFF is therefore not "dispatch rate"
   but "we're not actually using our cache".

Predicted budget for fixing this:
- If the real forward's `cacheRef` lookups miss for any reason
  (cacheKey collision, ref reset between calls, IR rebuild on every
  iteration), bringing it to the kcr-cached path reduces forward host
  time from ~5 ms → ~1.2 ms ⇒ wall 15 ms → 11 ms ⇒ ~90 TPS.
- This is independent of the kernel-speed gap (#47, ~+18 TPS) and the
  graphs-ON workaround (#252).

### 3b.4. Cache-miss localisation (#253 instrumentation, 2026-04-26)

`HESPER_ALLOC_TRACE=1` enables the existing throwaway-ref detector +
miss histogram in `Hesper/Backend/CUDA.lean`.  Run on 30 decode tokens:

```
[execImpl] cudaExecuteImpl calls=1599, total=33.4 ms
[cacheMiss] executeWithConfigCached cacheRef miss histogram (top 10):
  [throwaway?] key=11590943693958497111 × 455
  [throwaway?] key=3871721250815197703  ×  91
  [throwaway?] key=13014778072997566101 ×  73
  [throwaway?] key=8447589405719129130  ×  64
  ...
[cacheMissFirst] main-nwg=(8,2,1) wg=(256,1,1) (k_xxx) bufs=#[q, k_cache, v_cache, partial]   (×42 distinct keys)
[throwaway-ref] cacheKey=…  seen twice with none-Ref — caller uses IO.mkRef none
```

53 cache misses per token (1599 / 30) attributed to a single call
shape: `bufs=#[q, k_cache, v_cache, partial]`.  That is the
**FlashAttention tiled phase-1 kernel**.  42 layers × 2 phases
(phase-1 + phase-2) = 84 per token would be the worst case, and we
see ~53 — consistent with FlashAttn dominating the misses.

Source confirms it: `Hesper/WGSL/FlashAttention.lean::executeFlashAttentionTiled`
(line 1150) takes optional `phase1Ref / phase2Ref` parameters but the
two callers in `Hesper/Models/Gemma4.lean::forwardBlock` (lines 797 and
2191) do not pass them.  When omitted, the function constructs a fresh
`IO.mkRef none` *inside the function body* and uses it as the
`cacheRef` for `executeWithConfigCached` (line 1183, 1192).  That ref
is dropped at the end of the call ⇒ every iteration is a cold miss.

`executeFlashAttentionDynamic` (line 1212) has the same problem
hard-coded: `(← IO.mkRef none)` is the cacheRef argument with no way to
override it.

**Fix scope** (#253):
- Thread `phase1Ref / phase2Ref` from the call sites in
  `forwardBlock` through `executeFlashAttentionTiled`. The refs should
  live on the model's `KernelCacheRefs` (one pair per layer), so the
  dispatch is stable per (layer, cacheLen-bucket) across decode tokens.
- Drop the `(← IO.mkRef none)` in `executeFlashAttentionDynamic` for an
  optional ref param (or drop the function if forwardBlock no longer
  uses it).
- Re-run with `HESPER_ALLOC_TRACE=1`; expect the FlashAttn rows to
  vanish from the throwaway histogram.
- Re-run perf_compare: `forward` section should drop from ~5 ms to
  ~1.5 ms, wall ~15 ms → ~11 ms, TPS 60 → ~90.

This is consistent with H4b's predicted -3.8 ms wall delta in §3b.3.
The next session attacks the fix; #253 will be reopened with these
findings.

### 3c. Where the host CPU spends those 5 ms

`perf record -e cycles:u --call-graph=fp` during steady-state decode:

| symbol | self % | location |
|---|---:|---|
| `0x...001f44e1` (libcuda)         | ~14 % | driver internal — kernel launch path |
| `__memmove_avx_unaligned_erms`    | ~13 % | called from `lean_copy_expand_array` |
| `[k] kernel-space PMU sample`     | ~10 % | scheduler / IRQ |
| `lean_dec_ref_cold`               | ~ 8 % | refcount drop slow path |
| `lean_copy_expand_array`          | ~ 7 % | Array COW expand |

**Stack trace** (consistently): the lean_dec/lean_copy callers are all
`Hesper.Circuit.CompiledCircuit.replay → Array.forIn_loop → ...`. The
args Array per dispatch is being expand+drop'd every iteration. **Bound:**
14 % of 5 ms ≈ 0.7 ms wall. Removing the Array expand entirely is worth
< +5 TPS by itself.

The much larger lever is structural: **make the host work overlap with
the GPU drain**. The 5 ms forward block is sequential with the 10 ms
drain → 15 ms total. If forward overlaps drain, total → 10 ms (= drain)
+ residual = ~10.5 ms = 95 TPS.

## 4. Hypotheses & current verdicts

(loop: hypothesis → predicted ΔTPS → measurement → verdict)

| H# | Hypothesis | Predicted | Measured | Verdict |
|---:|---|---:|---:|---|
| H1 | Need more kernel fusion (more ops per launch) | +20 TPS | already at parity | ✗ DEAD |
| H2 | DtoH(4 byte) of argmax is a copy cost; replace with host-mapped slot | -9.8 ms → +30 TPS | -0.0 ms (renamed to cuStreamSync) | ✗ DEAD (#250) |
| H3 | Per-kernel GPU time | +18 TPS | not yet attempted | active (#47, on hold) |
| H4 | Forward host work runs sequentially with the drain; pipeline it across iterations | +30 TPS | partially measured (§3b.1) | **active** — root cause of the 4 ms gap is NOT yet identified; need llama.cpp graphs-OFF comparison |
| H4a | Existing CUDA-graph capture path (HESPER_CUDA_GRAPHS=1) already pipelines host launch via cuGraphLaunch | +20 TPS / steady state +30 TPS | **measured 59 → 80 TPS at 80-token, ~100 TPS steady-state** | **WORKAROUND only** — see §3b.2; hides the host work but does not explain why hesper graphs-OFF needs 4× more host work than llama.cpp graphs-OFF |
| H4b | hesper-specific per-block overhead (Circuit.replay args Array expand, kcr.getRef hash, withSection bookkeeping) is the real source of the 4 ms gap | varies | **measured via stub bench** — stub+kcr=1.24ms / stub-no-kcr=5.33ms / real=5ms ⇒ real path runs cache-cold-equivalent work despite kcr being available | ✓ CONFIRMED — see §3b.3; see #253 for the fix |
| H5 | Lean Array expand in replay loop adds ms | +5 TPS | bounded ≤ 0.7 ms wall | ✗ marginal, deferred |
| H6 | cuMemAlloc per token (driver alloc) | +5 TPS | already at 1 alloc/tok (pool); driver work is minimal | ✗ DEAD (#246) |
| H7 | mmap PLE on-demand H2D | -3 TPS regression noted | confirmed | trade-off accepted (#247-248) |
| H8 | host-mapped UVA for PLE table (skip register) | -824 MiB VRAM, neutral TPS | not yet attempted | parked (#249) |

## 5. Loop discipline

For every new hypothesis:

1. **Predict the wall-time delta in ms**, not TPS — TPS is a derived
   ratio that hides the size of the change.
2. **Identify the section that should change** (argmax / forward /
   total) per `HESPER_DECODE_SECT_TRACE`.
3. **Predict the API-table delta** (which row in nsys' cuda_api_sum
   should move, and by how much).
4. Implement behind an env flag (so A/B can run back-to-back without
   rebuild).
5. Measure with the protocol in §1. Compare predicted vs measured.
6. **Update §4 with the verdict.** If predicted ≠ measured, **the model
   is wrong**, not the implementation. Stop and re-read §3.

Every previous session has skipped step 1 or step 6. Don't.

## 6. Active workstreams (priority order)

1. **#253 (next, biggest lever)** — find the cache-miss in the real
   forwardSingleToken / forwardBlock dispatch path that makes it pay
   cold-path host cost (5 ms) when stub-bench at the same dispatch
   shape with kcr=ON pays only 1.24 ms. Predicted -3.8 ms wall =
   60 → 90 TPS for graphs-OFF. Approach:
   - Add `kcr` hit/miss counters in cudaExecuteImpl + Circuit.replay.
   - Run gemma4-cuda with `HESPER_KCR_TRACE=1`; expect to see hits but
     they may be on the wrong key, or the ref may be resetting.
   - Fix the cache-key / ref ownership; re-bench.

2. **#47 (kernel speed, paused)** — Q4_K matmul to llama.cpp parity.
   Predicted -2.5 ms wall. Independent of #1; combined will close most
   of the remaining gap.

3. **#252 (workaround) — default to CUDA Graphs ON** as a stop-gap so
   users see 80-100 TPS today while #1 is being investigated. Marked
   workaround because it hides H4b instead of fixing it; it does not
   close the *graphs-OFF* gap to llama.cpp.

4. **#235 follow-on (deferred)** — Lean Array expand in replay loop.
   May overlap with #1 once we know which Array expand is hot.

5. **#249 (deferred)** — PLE register-less UVA. Pure VRAM, no TPS.

## 7. What 'good enough' looks like

`scripts/perf_compare.sh both` shows:
- hesper `cuStreamSynchronize` per-token median ≤ 9 ms (= GPU kernel
  time + tiny tail), AND
- hesper `HESPER_DECODE_SECT_TRACE forward` median ≤ 1 ms (proves the
  host work is overlapped), AND
- wall ≤ 11 ms/token (≥ 90 TPS).

When all three hit, we're done with host-side; remaining gap to
llama-cli's 110 TPS is pure kernel speed (#47).
