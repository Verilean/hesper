**SUPERSEDED by 57-host-overhead-canonical.md** — left for history; do not
follow conclusions here without cross-checking §3 of doc 57.

# 51 — Lean runtime tail latency is the steady-state TPS gap vs llama.cpp

*Written 2026-04-24. Session end: hesper graphs OFF 60 TPS vs llama.cpp 107 TPS.*

## TL;DR

After ruling out per-launch FFI cost, cuMemAlloc, argmax DtoH, cacheRef misses,
and cuModuleLoadData one by one, the remaining **4.5 ms per-token steady-state
gap vs llama.cpp** lives in Lean runtime tail latency between CUDA API calls.

- hesper's median inter-API host time is **109 ns** — *faster* than llama.cpp's
  402 ns.
- hesper's p90 is **12 µs** vs llama.cpp's 785 ns (**15×**).
- hesper's p99 is **268 µs** vs llama.cpp's 1.9 µs (**140×**).

Across 877 decode-token launches, the p90+ tail alone contributes **4-9 ms per
token** — exactly the steady-state gap we've been hunting.

Lean runtime is faster on the median but has heavy-tailed stalls (likely GC,
refcount cycle collection, or `Array.push` reallocation), and each stall is a
direct GPU pipeline bubble.

## Session trajectory

| state | graphs-OFF TPS |
|---|---:|
| start of session | 43.0 |
| after preHash fix | **60.0** (+40%) |
| pipelined decode | 56.6 (worse, kept as infrastructure only) |
| target (llama.cpp) | 107 |

### preHash fix (landed, Hesper/Backend/CUDA.lean)

`cudaExecuteImpl` was unconditionally running `generatePTX` +
`fastStringHash` on every call, even when the module cache would hit. Added a
cheap preHash (funcName + workgroupSize + `state.stmts.length` +
`declaredBuffers.length`) probed against the module cache *before* the
expensive PTX generation. Only on preHash miss do we run the full PTX
generation and use its `sourceHash` as the authoritative key; the preHash is
also registered so subsequent calls hit it directly.

Impact: first-token stall 105 → 44 ms, cudaExecuteImpl total wall 95 → 29 ms,
TPS 43 → 60.

## Hypotheses tested and falsified

Each of these was previously suspected; each has been ruled out by direct
measurement in this session.

### Per-launch FFI wrapper cost (1 ns, as expected)

Microbench (`Tests/CUDA/CUDALaunchBench.lean`) measures raw
`cuLaunchKernel` = 1.116 µs/call, `executeWithConfigCached` = 1.117 µs/call.
Lean wrapper overhead is ~1 ns in a tight loop.

### Per-token argmax DtoH (not the bottleneck)

`HESPER_PIPELINED_DECODE=1` keeps the 4-byte argmax result on device and
reads it once at loop end. nsys confirms the per-token `cuMemcpyDtoH` is
gone (11 → 1 across 11 tokens). But TPS gets *worse*: 60 → 57.

The "28% gpuArgmax" in `perf report` children-sort was inclusive GPU-wait
time within the blocking DtoH call, not actual memcpy cost. Removing the
sync doesn't free up useful parallel work because the host has nothing
else to do at that point.

### cuMemAlloc during decode (1.2 ms total)

`HESPER_ALLOC_TRACE=1` instrumentation records 242 decode-only allocs
across 11 tokens, 5 unique sizes, total wall time **1.2 ms** (0.11 ms/tok).
Not the 4.5 ms/tok gap.

### cacheRef miss (saturated after prefill)

Instrumented `executeWithConfigCached` to count misses by funcName. 477
of 486 cudaExecuteImpl calls fire on prefill/first-token, only ~1 per
subsequent token. Steady-state cacheRef hit rate is effectively 100%.

### cuModuleLoadData (only fires first-token)

Timed with `recordModuleLoad`: 78 calls total, all during token 1 JIT
warmup, 5.9 ms total. Zero in tokens 2+.

## nsys inter-API gap distribution (the smoking gun)

Command:
```
nsys profile -t cuda,nvtx,osrt --sample=process-tree --cpuctxsw=process-tree ...
```

Gap = time between end of one CUDA API call and start of next CUDA API call
on the same host thread. This is "time the host spent running *our* code
instead of CUDA API code."

| percentile | hesper | llama.cpp | ratio |
|---|---:|---:|---:|
| p50  | **109 ns** | 402 ns | **0.27× (hesper faster!)** |
| p90  | **12 µs** | 785 ns | **15×** |
| p99  | **268 µs** | 1.9 µs | **140×** |
| total sum across run | 5.7 s | 1.1 s | 5× |

hesper is *faster* than llama.cpp at the median (Lean bytecode is tight).
But its distribution is catastrophically heavy-tailed. Roughly:

- ~90% of launches: gap < 12 µs, comparable to llama.cpp
- ~10% of launches: gap 12 µs – 268 µs, stalls the GPU pipeline
- ~1% of launches: gap > 268 µs, single-stall-dominates-step

Per-token budget: 877 × (10% × ~50-100 µs) ≈ **4-9 ms/token** of tail stalls.
This matches the **4.45 ms per-token gap** seen in nsys between `k_1506` (last
kernel of previous forward) and `k_1563` (first kernel of next forward).

## Interpretation

The CUDA API layer is a reliable per-call fixed cost (~1-3 µs) that both
hesper and llama.cpp pay equally. The difference is in what runs *between*
the API calls:

- **llama.cpp C++**: tight, predictable, 400 ns – 2 µs range. No heap allocation
  per call, stable refcount, no GC.
- **hesper Lean**: nanosecond-fast in the common case, but heavy-tailed with
  10-100× spikes that look like periodic runtime work.

Signatures consistent with the observed tail:

- **GC mark-and-sweep**: Lean runtime runs a minor/major collector periodically,
  tens to hundreds of µs.
- **Refcount cycle breaking**: `lean_dec_ref_cold` shows up at 8% in perf self
  time — this is the slow path for reference-counted objects with complex
  sharing graphs.
- **`Array.push` reallocation**: amortized O(1) but every 2^N pushes triggers
  a copy of the entire array. Appears as 5% `lean_copy_expand_array` in perf.
- **String interpolation (`s!"..."`)**: each call allocates a ByteArray, a
  closure capturing the captured variables, and returns a new String object —
  all refcount churn.
- **Possibly mmap / page fault**: cold model weight pages faulting in on the
  first few tokens (less likely after token 2, but not ruled out).

## Implications for the 115 TPS target

The gap is, in principle, closable:

1. **Reduce host-side tail events**. Audit forward path for:
   - `s!"..."` interpolation — replace with pre-computed strings or sectioned
     tag enums.
   - `List (String × Buf)` passing — switch to `Array` with pre-resolved
     indices (attempted in this session, but Lean elaboration of
     `Examples/Gemma4CUDA.lean` hit the 200K-heartbeat limit; needs a smaller-
     footprint redesign).
   - `ByteArray.empty |>.push ... |>.push ...` chains building params
     per call — pre-allocate.
   - `Array.push` growth — start with `Array.mkEmpty n` at known sizes.

2. **Bypass Lean between launches entirely**. The architectural fix:
   - `HESPER_CUDA_GRAPHS=1` already captures the submission sequence and
     replays it without returning to Lean between kernels. Gets ~80 TPS.
   - A native pre-assembled launch sequence (flat array of `(CUfunction,
     Array USize)` pairs with an outer `foldlM` in a tight `@[inline]` loop)
     would deliver the same benefit without CUDA Graph capture.

3. **Force GC before critical sections**. If the tail is dominated by GC,
   calling `GC.forceMinor` at the start of each decode step could move the
   cost outside the measurement window. Easy to try.

## What was also attempted and reverted

**Indexed `namedBufferIndices` cache** on `CUDACachedDispatch`. Added a field
to store the resolved buffer positions on first call, with a fast path on
subsequent calls that skips `List.find?` entirely. The CUDA backend change
compiled cleanly, but `Examples/Gemma4CUDA.lean` elaboration exceeded 200K
heartbeats. This is a heavy-elaboration warning sign — the change probably
needs to be redesigned to keep the record-update path trivial. Reverted.

## Next-session recipe

1. `nsys profile -t cuda,nvtx,osrt --sample=process-tree
   --cpuctxsw=process-tree` both hesper and llama.cpp runs — already captured
   this session: `/tmp/nsys_hesper_cpu.nsys-rep`, `/tmp/nsys_llama_cpu.nsys-rep`.
2. Inspect the actual stacks during the p90+ gap window using
   `nsys stats --report osrt_sum` and the `nsys-ui` timeline.
3. Implement the `s!"..."` and Array.push audit (cheap, bounded blast radius).
4. Try `Lean.PrivateName.GC.forceMinor` / `IO.Ref`-backed pool for ByteArray
   params.
5. If (3) and (4) together close < 2 ms/token of the 4.5 ms gap, commit to
   architectural route (CUDA Graphs wider adoption, or native flat launch list).

## 調査方法 (Methodology — how to reproduce / re-run)

The root-cause hunt went through several measurement layers. Each layer
answered a different question. All of these are reusable in future sessions.

### 1. Per-launch microbench (isolates FFI wrapper cost)

File: `Tests/CUDA/CUDALaunchBench.lean`

Purpose: measure the absolute cost of launching a trivial kernel through
each layer (raw FFI, `replayCached`, `executeWithConfigCached`) with no
other work. Rules out "the Lean wrapper itself is slow."

Command:
```
lake build cuda-launch-bench
./.lake/build/bin/cuda-launch-bench
```

Reads out 3 per-call timings. Any number >>1 µs here indicates wrapper
bloat; we saw 1.12 µs for all three — conclusion: FFI is fine.

### 2. HESPER_ALLOC_TRACE=1 instrumentation (per-category host wall)

Landed in `Hesper/Backend/CUDA.lean`. Env-flag-gated so it's cheap to leave
in the tree. Counters:

- `cudaAllocCounter` — (sizeBytes, totalWallNs) per `allocBuffer` size
  bucket. Detects uncached per-call allocations.
- `cudaModuleLoadWallNs` / `cudaModuleLoadCount` — time spent in
  `cuModuleLoadData` (PTX JIT). Detects decode-time JIT firing.
- `cudaExecuteImplWallNs` / `cudaExecuteImplCount` — time spent inside
  `cudaExecuteImpl` (= generatePTX + fastStringHash + module cache
  lookup + possibly cuModuleLoadData).
- `cudaCacheMissByName` — (funcName, count) histogram of
  `executeWithConfigCached` cacheRef misses.

Reset hooks fire at the start of the decode loop in `Gemma4.lean` so
prefill is excluded. Printed at end of `generate`.

Command:
```
HESPER_DP4A=1 HESPER_CHAT=1 HESPER_ALLOC_TRACE=1 \
  ./.lake/build/bin/gemma4-cuda data/gemma-4-e4b-it-Q4_K_M.gguf "Hello" 30
```

Output includes `[tok N] wall=... modLoad=...` per iteration, `[alloc]`
histogram, `[modload]` stats, `[execImpl]` stats, `[cacheMiss]` histogram.

### 3. perf record + call-graph (CPU-side blame)

Command:
```
perf record -F 4999 --call-graph dwarf,16384 -g -o /tmp/perf_decode.data -- \
  env HESPER_DP4A=1 HESPER_CHAT=1 \
  ./.lake/build/bin/gemma4-cuda data/gemma-4-e4b-it-Q4_K_M.gguf \
    "Write a long story" 200 > /tmp/perf_out.log 2>&1

# Self-time top:
perf report -i /tmp/perf_decode.data --stdio --no-children \
  --sort overhead,symbol --percent-limit 0.5

# Children-sort (inclusive):
perf report -i /tmp/perf_decode.data --stdio --sort overhead,symbol \
  --percent-limit 0.3 | head -50

# Callers/callees of a specific symbol:
perf report -i /tmp/perf_decode.data --stdio -g graph,2,caller \
  --symbol-filter=expToPTX
```

Paranoid trap: children-sort attributes *inclusive* time, so a blocking
call like `cuMemcpyDtoH_v2` can appear at 28% but most of it is GPU-wait
inside the call, not actual CPU work. Use self-time (`--no-children`) to
find real CPU hotspots.

200-token prompt + no-EOS story makes decode time dominate over model
load time in the sample mix. Smaller max_tokens leaves model load as the
dominant symbol bucket (memmove 27% is mostly mmap → parse GGUF).

### 4. nsys with CUDA + OSRT + CPU sampling (inter-API gap distribution)

This is what finally pinned the root cause.

```
# hesper
HESPER_DP4A=1 HESPER_CHAT=1 \
  nsys profile -o /tmp/nsys_hesper_cpu --force-overwrite=true --stats=false \
  -t cuda,nvtx,osrt --sample=process-tree --cpuctxsw=process-tree \
  ./.lake/build/bin/gemma4-cuda data/gemma-4-e4b-it-Q4_K_M.gguf "Hello" 30

# llama.cpp (graphs OFF for apples-to-apples)
GGML_CUDA_DISABLE_GRAPHS=1 \
  nsys profile -o /tmp/nsys_llama_cpu --force-overwrite=true --stats=false \
  -t cuda,nvtx,osrt --sample=process-tree --cpuctxsw=process-tree \
  llama.cpp/build/bin/llama-cli -m data/gemma-4-e4b-it-Q4_K_M.gguf \
    -p "Hello" -n 30 -ngl 99 --no-warmup -st
```

Use `-st` (single-turn) on llama.cpp to avoid hanging at stdin. Both
runs need `-ngl 99` to keep the full model on GPU, and must be run when
VRAM is free of other consumers.

#### Inter-API gap extraction

```
# Export cuda_api_trace to CSV, compute gaps between consecutive events,
# sort, and derive percentiles.
nsys stats --force-export=true --report cuda_api_trace --format csv \
    /tmp/nsys_hesper_cpu.nsys-rep 2>/dev/null \
  | grep -v "^NOTICE\|It is\|Consider\|^$\|Processing" \
  > /tmp/hesper_api.csv

awk -F, 'NR>1 {
    start=$1; dur=$2;
    if (prev_end > 0) gap = start - prev_end;
    if (gap > 0) print gap;
    if (start+dur > prev_end) prev_end = start+dur
}' /tmp/hesper_api.csv | sort -n > /tmp/hesper_gaps.txt

awk 'BEGIN{c=0}{c++;a[c]=$1;s+=$1}END{
  printf "sum=%.2f ms mean=%.0f ns p50=%d p90=%d p99=%d max=%d\n",
    s/1e6, s/c, a[int(c*0.5)], a[int(c*0.9)], a[int(c*0.99)], a[c]
}' /tmp/hesper_gaps.txt
```

Same commands on `nsys_llama_cpu.nsys-rep` for comparison.

### 5. nsys GPU gap analysis (where the idle lives on the GPU)

Steady-state window extraction (skip prefill + first-token stall):

```
# Extract kernel events (exclude memset/memcpy)
nsys stats --force-export=true --report cuda_gpu_trace --format csv \
    /tmp/nsys_hesper_cpu.nsys-rep 2>/dev/null \
  | grep -v "^NOTICE\|It is\|Consider\|^$\|Processing\|^Start" \
  > /tmp/gpu_trace.csv
awk -F, '$NF !~ /memset|memcpy/ { print $1","$2","$NF }' /tmp/gpu_trace.csv \
  | sort -t, -k1,1n > /tmp/gpu_kern.csv

# Pick a 140ms slice that spans 9 steady-state tokens (after prefill and
# after the first-token stall).
SS_START=...  # from timeline
SS_END=$((SS_START + 140000000))
awk -F, -v s=$SS_START -v e=$SS_END '$1 >= s && $1 < e' /tmp/gpu_kern.csv \
  > /tmp/ss_kern.csv

# Busy/idle breakdown on that slice:
awk -F, 'BEGIN{prev_end=0;busy=0;idle=0;n=0;first=0;last=0}
{ start=$1; dur=$2;
  if (first==0) first=start;
  if (prev_end > 0 && start > prev_end) idle += start - prev_end;
  busy += dur;
  if (start+dur > prev_end) prev_end = start+dur;
  if (start+dur > last) last = start+dur;
  n++
}
END {
  span = last - first;
  printf "kernels=%d span=%.2fms busy=%.2fms(%.1f%%) idle=%.2fms\n",
    n, span/1e6, busy/1e6, 100*busy/span, idle/1e6
}' /tmp/ss_kern.csv
```

For finding which kernel pairs bracket the big gaps:

```
awk -F, 'BEGIN{prev_end=0}
{ start=$1; dur=$2; name=$3;
  if (prev_end > 0) {
    gap = start - prev_end;
    if (gap > 1000000)
      printf "gap=%.2fms at %.2fms, before=%s, after=%s\n",
        gap/1e6, start/1e6, prev_name, name
  }
  if (start+dur > prev_end) { prev_end = start+dur; prev_name = name }
}' /tmp/ss_kern.csv
```

### 6. nsys CUDA API zoom (what host was doing inside a specific gap)

Once a 4.45 ms GPU-idle gap is located at timestamp T, extract the CUDA
API events in the window [T-1ms, T+5ms] and count by API name:

```
nsys stats --force-export=true --report cuda_api_trace --format csv \
    /tmp/nsys_hesper_cpu.nsys-rep 2>/dev/null > /tmp/api.csv

START=$((T_NS - 1000000))
END=$((T_NS + 5000000))
awk -F, -v s=$START -v e=$END '$1 >= s && $1 < e' /tmp/api.csv \
  | awk -F, '{print $3}' | sort | uniq -c | sort -rn
```

This is what revealed "128 cuMemcpyHtoD + 216 cuLaunchKernel during the
gap" and led to — correctly ruling out — the sync-write hypothesis
(switching to async on default stream didn't help).

### 7. Kernel launch-to-exec latency (AAvg / QAvg / KAvg)

```
nsys stats --force-export=true --report cuda_kern_exec_sum --format csv \
    /tmp/nsys_hesper_cpu.nsys-rep 2>/dev/null | head -20
```

Columns of interest:
- `AAvg` — host API time (cuLaunchKernel call duration)
- `QAvg` — queue time (submit → GPU start)
- `KAvg` — kernel execution time on GPU

Compare these between hesper and llama.cpp per-kernel. `AAvg` being
equal across both (1-3 µs) ruled out "driver is slower on us." `QAvg`
being larger on hesper reflects the tail-latency submission gaps
(kernels queue behind still-submitting kernels).

## Path forward: Lean-only vs C shim vs C++ rewrite (AD considered)

hesper is not inference-only — it carries a verified automatic
differentiation story (see `docs/VERIFIED_AD.md`,
`docs/BACKWARD_COMPLETENESS.md`). The AD pipeline generates backward
kernels by transforming the Lean forward expressions (ShaderM / Exp /
Prim). Any option that moves forward logic out of Lean must be judged
against whether it preserves AD-reachability of the forward graph.

### Option A: rewrite the decode loop in C++ (`extern "C"`)

- Labor: high.
- TPS ceiling: parity with llama.cpp (~110 TPS).
- **AD impact: BREAKS IT.** If forward is implemented in C++, the
  Lean transformer passes cannot see it; pullback generation loses
  its input. Inference and training would fork into two implementations,
  creating numerical and kernel-set divergence. Violates the
  framework's verified-AD premise.
- **Rejected**.

### Option B: narrow C shim for the `cuLaunchKernel` hot path

- Labor: ~1 day.
- TPS ceiling: estimated 95-110 TPS (tail stalls eliminated in the
  one place that matters per-launch).
- **AD impact: NONE.** The forward expression graph, ShaderM, PTX
  generation, and pullback derivation all stay in Lean. The shim
  replaces only the Lean code between "we have a CUfunction + args
  Array USize" and "cuLaunchKernel has returned" — by then the AD
  pipeline has already done its work on the forward graph. Backward
  kernels use the same shim.
- Why this kills the tail: a C function that does not touch the Lean
  heap cannot trigger GC / refcount cycle collection / Array realloc
  between launches. The p90 12 µs stall is a Lean-runtime event, and
  a C function call is invisible to it.
- Shim signature sketch:
  ```c
  // Takes a pre-resolved func + arg-pointer table and launches,
  // bumping the dispatch counter.  No Lean allocations.
  void hesper_launch_fast(CUfunction f, CUstream s,
                           uint32_t gx, uint32_t gy, uint32_t gz,
                           uint32_t bx, uint32_t by, uint32_t bz,
                           size_t* args, size_t nargs);
  ```
  Lean just builds `args` once (or reuses cached args) and calls
  this. All the PendingLaunch / batch queue / trace logic on the
  hot path moves into C.
- Applies to WebGPU backend too (Dawn has an equivalent C API).
- **Preferred.**

### Option C: AOT-precompile the launch sequence via Lean macros

- Labor: medium. CUDA-only (WebGPU still needs runtime WGSL string
  handoff).
- TPS ceiling: similar to B (removes the same allocations).
- **AD impact: PARTIAL.** Inference's forward path can be statically
  precompiled because the op order is fixed. Backward pullback is also
  a function of the forward graph, but *training* loops can have
  input-dependent shapes (variable seq length, masking, etc.), so
  precompiling a fixed launch sequence may not cover every AD
  scenario. Would need dual-path support (precompiled for inference,
  dynamic for training) to preserve AD.
- Known elaboration risk: the earlier indexed-buffer resolve PoC
  (just a record-update + mutable loop) hit the 200K-heartbeat limit
  in `Examples/Gemma4CUDA.lean`. Precompiling 500+ kernel launches
  at elaboration time is likely to blow past that budget unless
  carefully staged.
- **Viable as a later polish, not the first move.**

### Summary

| option | TPS ceiling | AD preserved | labor |
|---|---:|---|---|
| A: C++ forward | ~110 | ❌ | high |
| **B: C shim for launch** | **95-110** | **✅** | **~1 day** |
| C: AOT macro | ~95 | ⚠️ partial | medium |
| Lean-only tuning | ~75 | ✅ | ~1 week |

The recommended path is **B** (narrow C shim), possibly preceded by
the low-risk Lean-side allocation audit (`s!"..."`, `List` → `Array`,
`ByteArray` pre-alloc) to capture the "easy" 1-2 ms/token before
deciding whether the remaining gap justifies the C shim.

AD-aware note: the training loop hits the same `cuLaunchKernel` hot
path with its backward kernels, so option B benefits training as
well. Forward and backward share the shim; neither is coupled to
Lean's GC cadence at launch boundaries.

## Artifacts produced this session

- `Hesper/Backend/CUDA.lean` — preHash logic, allocCounter, moduleLoadTimer,
  executeImplTimer, cacheMissTracker instrumentation (all gated on
  `HESPER_ALLOC_TRACE=1`).
- `Hesper/Models/Gemma4.lean` — per-iter `[tok N] wall=... modLoad=...`
  logging, `HESPER_PIPELINED_DECODE=1` flag (infrastructure only).
- `Tests/CUDA/CUDALaunchBench.lean` — per-launch 3-layer microbench.
- `/tmp/nsys_hesper_cpu.nsys-rep`, `/tmp/nsys_llama_cpu.nsys-rep` —
  CPU-sampled nsys traces for apples-to-apples inter-API gap analysis.
- `/tmp/hesper_gaps.txt`, `/tmp/llama_gaps.txt` — sorted inter-API gap
  distributions.
