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

### 3b.5. H4b refutation — fixing the cache miss did not move wall (2026-04-26)

`Hesper/WGSL/FlashAttention.lean::executeFlashAttentionTiled` was rewired
to receive an optional `kcrLookup` callback resolving (cacheKey →
IO.Ref) through the model's `KernelCacheRefs`, replacing the inline
`(← IO.mkRef none)` throwaway. The `forwardBlock` callers at
`Gemma4.lean:797` and `Gemma4.lean:2191` now pass `kcr.map (...)`.

Result on graphs-OFF, 80-token decode of "Explain the fall of the
Roman Empire":

| metric                | before | after  | delta |
|---|---:|---:|---:|
| `cudaExecuteImpl calls` (30-tok) | 1599 | **535** | -1064 (-66 %) |
| `cudaExecuteImpl total` (30-tok) | 33.4 ms | 31.4 ms | -2 ms (-6 %) |
| TPS (80-tok)          | 59.4 | 60.4 | +1 (noise) |

The cache misses **were** real (counts dropped by 2/3), but the wall
**did not move**. cudaExecuteImpl's miss path costs only ~60 µs on
average and adds up to ~1 ms / token total. Fixing it lifts a 1 ms
floor that we never saw in `forward` host time anyway, because PTX
module caching at the driver level was already absorbing the
re-generation cost. The Lean side of the miss path (ShaderM build +
hash + ref bookkeeping) is microseconds, not milliseconds.

**That kills H4b**: cache-key bookkeeping is not the source of the
4 ms forward host-time gap.

The stub-bench delta (no-kcr 5.5 ms vs kcr=on 1.2 ms = -4.3 ms) is
real, but it measures a *different* cost than what the real forward
pays: `forwardTokenStubPerLayer` deliberately routes through the cold
path of `cudaExecuteImpl` (no kcr arg) so every dispatch re-runs
ShaderM + hashes; the real forward avoids that on the *non-FlashAttn*
path because most call sites correctly pass `kcr` to `executeWithConfigCached`.

The 4 ms gap therefore lives somewhere in the real forward's
**non-cudaExecuteImpl** host work. Candidates not yet measured:
- per-block IO.Ref reads on `state.fused*Prepared` slots (6+ refs/block)
- Lean Array build-up in `Hesper.Layers.Linear.forward` (the dp4a path
  builds up new buffer-name lists per call)
- `Hesper.WGSL.Execute.withSection` push/pop (string allocation)
- `Hesper.Layers.Linear.dp4aEnabled.get` etc. (env-time refs read every
  layer × every call)
- `Hesper.Backend.beginBatch` overhead — currently a no-op on CUDA but
  enters via typeclass dispatch

Next step: `perf record` with `-c 100000 --call-graph=fp` on the **real
forward only** (sandwich `IO.monoNanosNow` markers around it), filter
to kernel-time-excluding samples, and look for hot Lean symbols that do
not show up in the stub bench. Without this we are guessing again.

### 3b.6. Section profile (HESPER_SECTION_PROFILE=1, 2026-04-26)

`Hesper/WGSL/Execute.lean::sectionProfilingRef` was already wired to
accumulate per-section host wall via the `withSection` instrumentation;
plumbing it through `Examples/Gemma4CUDA.lean` behind `HESPER_SECTION_PROFILE=1`
gives a per-section ms breakdown over the full decode. Run on 30 tokens,
graphs OFF:

| section                | total ms (30 tok) | calls | avg µs/call | per-token ms |
|---|---:|---:|---:|---:|
| **perLayerEmbd**       | 51.0 | 1260 (=30×42) | **40** | **1.70** |
| ple.inpGateGeluSlice    | 27.4 | 1260 | 22 | 0.91 |
| ffnNormGateUp           | 17.3 | 1260 | 14 | 0.58 |
| oProj                   | 14.9 | 1260 | 12 | 0.50 |
| flashAttn               | 14.8 | 1260 | 12 | 0.49 |
| ffnDown                 | 14.1 | 1260 | 11 | 0.47 |
| ple.proj                | 12.6 | 1260 | 10 | 0.42 |
| rope                    |  8.8 | 1260 |  7 | 0.29 |
| qkvNorm                 |  6.9 | 1260 |  5 | 0.23 |
| postAttnNorm/postFFNNorm/ple.postNormAdd | ~6.3 each | 1260 | 5 | 0.21 |
| kvWrite                 |  2.8 |  720 |  4 | 0.09 |
| (smaller sections sum)  |  ~10 | various | | ~0.3 |
| **total approx (host wall, includes nested)** | ~196 | | | **~6.5** |

(Sections nest: `perLayerEmbd` includes `ple.inpGateGeluSlice` +
`ple.proj` + `ple.postNormAdd`. Subtracting nested children, the
*direct* `perLayerEmbd` body is ~5 ms over 30 tokens = 4 µs/call /
0.17 ms/tok — small. The big number is dominated by its children.)

### 3b.7. Comparison with stub-bench (H4c)

`Examples/Gemma4StubDecodeBench.lean` with kcr=on hits 1.7 µs / dispatch.
The real forward's hottest sections sit at **10-22 µs / call** even
though they go through the same `executeWithConfigCached` path —
**4-12× the stub floor**. The extra cost is *between* Lean entering
the section and `executeWithConfigCached` actually running:

- `Hesper.Circuit.runCached` / `runCachedFused` is in every hot
  section (forwardBlock lines 448, 549, 568, 696, 834, 1032, 1085,
  1160, 1300, 1361). Each call hits `CompiledCircuit.replay`, which
  iterates `cc.ops` and rebuilds the (id → buffer) lookup list with
  `buffers ++ cc.baseBuffers` and `combined.find? ...` per op.
- `getGlobalCircuitRef` resolves the cacheRef by hashing a key on
  every call (cheap individually, but 7+ times per layer × 42 layers
  is ~3000 hash+lookup operations / token).
- `CircuitM.run` does NOT run on cached path (good), but the lookup
  list build inside `replay` *does* run every iteration.

**H4c (new active hypothesis)**: the real forward's per-section host
overhead is concentrated in `Hesper.Circuit.CompiledCircuit.replay`,
specifically in (a) the `buffers ++ cc.baseBuffers` list concat per
call and (b) `combined.find?` linear scan per op. Predicted budget:
collapsing the 4 µs/call gap (real 10-22 µs - stub 1.7 µs ≈ 8-20 µs)
times ~300 calls / token ≈ 2-5 ms / token. Same order as the gap.

Next experiment (no code change yet): re-run with two of the hottest
Circuit-using call sites bypassed (replace `runCached` with the
manual `ce` path) and see whether `perLayerEmbd` host wall drops by
the predicted amount. If yes → the fix is to specialise `replay` for
the small-fixed-N op case (no list concat, no `find?`).

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
| H4b | FlashAttention's `IO.mkRef none` throwaway is the source of the 4 ms gap | -3.8 ms | **measured: cudaExecuteImpl calls 1599→535 / wall +0 ms** (60.4 vs 59.4 TPS = noise) | ✗ FALSE — fixing the cache misses removed 1064 misses but did not move wall. The cudaExecuteImpl miss path is cheap (~60 µs avg, 1 ms/tok total), so eliminating it has no wall effect. The 4 ms gap is somewhere else; see §3b.5 |
| H4c | `Hesper.Circuit.CompiledCircuit.replay` (buffers list concat + linear `find?` per op) inflates each Circuit-DSL section to 10-22 µs vs stub's 1.7 µs | -2 to -5 ms | **A/B run on `oProj` 2026-04-25**: `runCached` 11.79 µs → bypass 9.90 µs (-1.9 µs/call ≈ -2.4 ms total, but TPS 55.0→55.4 = noise). Bypass does NOT reach stub-floor 1.7 µs. | ✗ **mostly DEAD** — replay overhead is real but small (~16% of section host time); the remaining ~10 µs/section is shared with the bypass path (Linear.LinearLayer.forward), so it lives in `cudaExecuteImpl` / dispatch building, not in Circuit replay. Refactoring replay would save ≤ 2 ms/tok at most. |
| H4d | `HESPER_SECTION_PROFILE=1` instrumentation itself inflates the steady-state forward by 3 ms/tok | -3 ms wall | **measured 2026-04-25**: profile-OFF forward steady-state 4.93 ms vs profile-ON forward ~8 ms → ~3 ms is the profile cost. Profile-OFF TPS 56.8 vs profile-ON 55.0 (+0.6 TPS / -0.6 ms wall once argmax overlap is accounted for). | ✓ confirmed — **all section-profile per-call numbers are inflated by ~3 µs/section** (`withSection` wraps each call in `IO.monoNanosNow` × 2 + Std.HashMap update). Real per-section host floor is closer to 7-8 µs/call, not 12 µs. Profile is still a useful relative-ordering tool but not an absolute timing source. |
| H4e | Real forward 4.93 ms / token = 880 dispatches × 5.6 µs/dispatch (kcr-cached, no profiling) — 3× the stub-bench 1.7 µs/dispatch floor — is dominated by per-launch FFI / cuLaunchKernel cost on real (non-stub) kernel arg lists | -2 ms wall | partial: H4e(b) buffer-resolution lockstep fast-path tried 2026-04-25, **forward 4.93→5.02 ms (no improvement)** | active — `executeWithConfigCached`'s buffer-resolution scan is NOT the dominant per-dispatch cost. Remaining suspects: tryDescLaunch path, PendingLaunch struct alloc, args-Array USize copy on cache-hit |
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

### Re-measurement 2026-04-26 (graphs OFF, both sides, decode steady state)

`scripts/perf_compare.sh both` with `GRAPHS=off`, `N_TOKENS=60`.
Decode steady-state isolated by taking deciles 2-10 of the kernel
timeline (skipping first 10% to drop prefill warmup) — the timeline
is uniform throughout the run since hesper TPS 62.4 / llama TPS
111.6 means kernel emission rates are steady.

| metric | hesper | llama-cli | 比 (h/l) |
|---|---:|---:|---:|
| kernel calls | 49953 | 68006 | 0.73× (hesper fewer) |
| **kernel sum time** | **585.0 ms** | **469.9 ms** | **1.245×** |
| wall time of decode window | 869.2 ms | 523.2 ms | 1.66× |
| **GPU busy ratio** (kernel/wall) | **67.3%** | **89.8%** | — |
| **GPU idle (wall − kernel)** | **284.2 ms** | **53.3 ms** | **5.33×** |
| top kernel per-call (hesper k_1387 / llama mul_mat_vec_q12) | 68.8 µs | 64.0 µs | 1.07× |

Per-token decomposition (decode window / 60 tok):

| | hesper ms/tok | llama ms/tok | gap/tok |
|---|---:|---:|---:|
| total wall | 14.5 | 8.7 | 5.77 |
| ↳ kernel time | 9.75 | 7.83 | 1.92 |
| ↳ GPU idle | 4.74 | 0.89 | **3.85** |

**This settles the prior contradiction.** The 5.77 ms/tok wall gap
splits as:

- **Per-kernel speed: 1.92 ms/tok (33%)** — real but bounded.
  Top-kernel per-call is 1.07× (hesper 68.8 vs llama 64.0 µs on
  the busiest matmul); kernel mix difference accounts for the
  rest of the 1.245× total kernel-time delta. `project_final_tps_80.md`
  "90% parity" was about hand-tuned individual kernels and still
  holds; the long tail of pointwise stubs etc. is where the +25%
  comes from.
- **GPU idle bubbles: 3.85 ms/tok (67%)** — the dominant lever.
  hesper's GPU sits idle ~33% of decode wall vs llama.cpp's 10%.
  This is exactly H4 (forward host work runs sequential with
  drain instead of overlapping it).

Lever sizing — closing each fully would yield:

- closing kernel-speed gap (1.92 ms/tok recovered): TPS 62 → 71
- closing idle-bubble gap (3.85 ms/tok recovered): TPS 62 → 92
- closing both: TPS → ~115 (≈ llama.cpp parity)

So **the dominant lever is the idle-bubble side, by ~2×**. Earlier
"#47 kernel speed at top priority" claim was wrong: kernel speed
recovers at most 1.92 ms/tok, not 6.7 ms/tok.

Priority by lever size (from the decomposition above):

1. **GPU idle bubble (3.85 ms/tok lever, 67% of gap)**. hesper GPU
   busy 67.3% vs llama 89.8% during decode. The 284 ms idle in
   the 869 ms decode window means host work is delaying kernel
   submission. Same bucket as H4e but framed as a wall-time
   target: drive decode-window GPU busy from 67% → 90% to recover
   ~30 TPS. Concrete experiments still untried:
   (a) `perf record --call-graph=fp` on the decode loop, attribute
       cycles to specific Lean callees.
   (b) Compare hesper trace to llama.cpp trace at the inter-launch
       gap distribution (we did this for p50/p90/p99 in
       `project_lean_tail_latency_vs_llama.md`; rerun on the
       current binary).

2. **#252 (workaround) — default to CUDA Graphs ON**. Stops the
   bubble (recapture, host overlaps replay). 80-100 TPS today.

3. **Per-kernel speed (1.92 ms/tok lever, 33% of gap)**. Top
   kernels at 1.07× parity; long tail (pointwise stubs, plus 25%
   on non-top kernels) accounts for the rest. Closing this gives
   +9 TPS at most. Q6_K (#216) and Q4_K (#47) are partial
   contributors.

4. **#249 (deferred)** — PLE register-less UVA. Pure VRAM, no TPS.

## 7. What 'good enough' looks like

`scripts/perf_compare.sh both` shows:
- hesper `cuStreamSynchronize` per-token median ≤ 9 ms (= GPU kernel
  time + tiny tail), AND
- hesper `HESPER_DECODE_SECT_TRACE forward` median ≤ 1 ms (proves the
  host work is overlapped), AND
- wall ≤ 11 ms/token (≥ 90 TPS).

When all three hit, we're done with host-side; remaining gap to
llama-cli's 110 TPS is pure kernel speed (#47).
