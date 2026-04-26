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
| H4 | Forward host work runs sequentially with the drain; pipeline it across iterations | +30 TPS | not yet attempted | **active (#251) ← next** |
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

1. **#251 (active)** — pipeline forward host work over the drain.
   Predicted -3.5 ms wall (66 → 95 TPS). Plan: have the host queue the
   *next* forward's kernels onto the same stream *before* calling
   cuStreamSynchronize. Token-dependent buffers (`tokenBuf`, `posBuf`)
   need to be set after argmax read; everything else (the kernels'
   captured args) can be queued ahead.
2. **#47 (paused)** — kernel speed. Predicted -2.5 ms wall (66 → 80 TPS
   independent of #251; combined with #251, would target 110 TPS).
   Re-prioritise after #251 measurement.
3. **#235 follow-on (deferred)** — Lean Array expand in replay loop.
   Bounded benefit ≤ 0.7 ms.
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
