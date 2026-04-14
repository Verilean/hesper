# Hesper Profiling & Analysis Guide

Standard workflow for measuring and analyzing inference kernels across
WebGPU and CUDA PTX backends. Covers the full path from wall-clock TPS
down to individual PTX instructions.

---

## 1. Top-down Flow

```
[ Benchmark: TPS measurement ]       ← start here
        │
        ├── Section profile (WebGPU: gemma4-profile)
        │   "how many ms/tok does each section consume?"
        │
        ├── Kernel profile (CUDA: nsys)
        │   "how many ns × how many invocations per kernel?"
        │
        └── Kernel-internals (CUDA: ncu)
            "BW efficiency / occupancy / warp stall reasons"

[ Identify suspect kernel ]
        │
        ├── PTX diff (nvcc reference vs hesper-generated)
        │   "is the instruction stream correct at the bit level?"
        │
        └── Known-answer unit test
            "does a small controlled input produce the expected output?"
```

Always work top-down. Do not dive into PTX before narrowing the
bottleneck with nsys; do not dive into nsys before identifying the
slow section with `gemma4-profile`.

---

## 2. Wall-clock TPS benchmark

```bash
# f32 baseline (dp4a disabled)
for i in 1 2 3; do
  ./.lake/build/bin/gemma4-cuda data/gemma-4-e4b-it-Q4_K_M.gguf "Hello" 50 \
    2>&1 | grep 'tokens/sec'
done

# dp4a enabled
for i in 1 2 3; do
  HESPER_DP4A=1 ./.lake/build/bin/gemma4-cuda \
    data/gemma-4-e4b-it-Q4_K_M.gguf "Hello" 50 2>&1 | grep 'tokens/sec'
done

# llama.cpp reference (the target)
./llama.cpp/build/bin/llama-bench \
  -m data/gemma-4-e4b-it-Q4_K_M.gguf -n 50 -p 1 -ngl 999
```

Take at least 3 runs and average. Discard the first run (JIT compile
and pipeline-cache warmup cost).

---

## 2.5 How each profiler measures time (internals)

Three distinct measurement mechanisms are used. Know which one you are
looking at, because each one measures something different.

### 2.5.1 Wall-clock TPS (end-to-end)

Simplest possible mechanism — `IO.monoNanosNow` before and after the
generation loop:

```lean
let start ← IO.monoNanosNow
for i in [0:nTokens] do
  forwardSingleToken device model tokenId i state
let stop  ← IO.monoNanosNow
let tps := nTokens.toFloat * 1e9 / (stop - start).toFloat
```

`forwardSingleToken` in *batch mode* (the default for
`gemma4-cuda` / `gemma4-inference`) queues a whole forward pass'
dispatches into one command encoder and flushes at the end — one
host/device sync per token. So wall-clock includes:

- PTX JIT on the very first call (excluded by warmup)
- all kernel launches, queued then flushed
- H2D upload of small parameters (pos, cacheLen) per token
- the final `cuCtxSynchronize` (or WebGPU fence) that makes
  `logitsBuf` readable on the host for the next tokenizer step

This is the number that matters for "is the model faster?". Everything
below is attribution, not truth.

### 2.5.2 Per-section CPU timing (`withSection`)

`Hesper/WGSL/Execute.lean` exposes:

```lean
initialize sectionProfilingRef : IO.Ref Bool
initialize sectionTotalsRef    : IO.Ref (Array (String × UInt64 × Nat))

@[inline] def withSection (name : String) (act : IO α) : IO α := do
  if ← sectionProfilingRef.get then
    let t0 ← IO.monoNanosNow
    let r  ← act
    let t1 ← IO.monoNanosNow
    addSectionSample name (t1 - t0).toUInt64
    pure r
  else act
```

And `forwardSingleToken` is peppered with `withSection` calls:

```lean
Hesper.WGSL.Execute.withSection "ffnDown" do
  Linear.LinearLayer.forward ctx block.ffn.down state.geluBuf state.ffnOutBuf
```

**Crucial constraint**: each `executeWithConfig` in *unbatched* mode
does `deviceWait future` immediately after `dispatchCompute` (see
`WGSL/Execute.lean`). That means `t1 - t0` really reflects the time
that dispatch spent on the GPU (plus a bit of host overhead), not the
queue-submit time. **In batch mode the numbers would be garbage**, so
`gemma4-profile` explicitly disables batching (`profilingRef.set true`
forces per-call deviceWait).

Aggregation is a linear scan over `sectionTotalsRef`:

```lean
private def addSectionSample (name : String) (ns : UInt64) : IO Unit := do
  let arr ← sectionTotalsRef.get
  match arr.findIdx? (fun e => e.1 == name) with
  | some i => ... accumulate
  | none   => ... push new entry
```

Nested sections work and the outer section includes the nested time.

### 2.5.3 Per-shape LinearLayer timing

`Hesper/Layers/Linear.lean` has a **second, independent** bookkeeping
layer for `LinearLayer.forward`:

```lean
initialize profilingRef  : IO.Ref Bool
initialize totalNanosRef : IO.Ref UInt64
initialize callCountRef  : IO.Ref Nat
-- (inDim, outDim, totalNanos, callCount)
initialize perShapeRef   : IO.Ref (Array (Nat × Nat × UInt64 × Nat))
```

Every `LinearLayer.forward` / `forwardDP4A` / `forwardFusedGateUp` etc
has a `startNs = monoNanosNow` at the top and at the end:

```lean
if profiling then
  let endNs ← IO.monoNanosNow
  let delta := (endNs - startNs).toUInt64
  totalNanosRef.modify (· + delta)
  callCountRef.modify  (· + 1)
  perShapeAdd layer.config.inDim layer.config.outDim delta
```

`perShapeAdd` groups samples by `(inDim, outDim)` so you can see
"ffnDown-shape" (10240×2560) separately from "qkvProj-shape"
(2560×2048).

Same constraint as sections: the CPU-side timestamp only reflects GPU
time because the underlying dispatch is unbatched when profiling.

### 2.5.4 nsys / ncu GPU-side measurements

Section profiling measures *wall time on the CPU between the
`dispatch` call and the fence return*. That is close to GPU time on a
well-behaved driver, but not perfect — driver overhead, event
polling, scheduling hiccups all creep in.

nsys / ncu use **NVIDIA's profiling API** (CUPTI underneath). That
gives actual on-device start/end timestamps via hardware counters, plus
per-kernel launch bookkeeping (grid/block/register count). This is the
ground truth for "what did the GPU actually do?".

Three cost layers to keep straight when reading nsys output:

```
┌────────────────────── wall time (TPS) ──────────────────────┐
│                                                              │
│  host overhead   queue sync   ┌── GPU active time ──┐ fence  │
│                                                     │        │
│                               nsys kernel durations          │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

When `withSection "ffnDown"` reports 5.5 ms/tok and nsys reports the
same kernel group summing to 4.8 ms, the 0.7 ms gap is host-side
overhead per dispatch — visible in nsys as gaps between kernel runs
on the timeline.

### 2.5.5 How to measure dispatch overhead concretely

Dispatch overhead = (host CPU time to issue a kernel launch) + (kernel
launch latency until GPU starts executing it). Two complementary ways
to measure it:

#### A. nsys timeline: gaps between kernels

The nsys GPU trace CSV lists every kernel launch with a `Start (ns)`
and `Duration (ns)`. The **difference between one kernel's end and
the next kernel's start** is dispatch + driver overhead.

```bash
nsys stats --report cuda_gpu_trace --format csv /tmp/hesper.nsys-rep \
  | awk -F',' 'NR>11 && $21 !~ /memset|memcpy/' > /tmp/trace.csv

# Compute inter-kernel gaps
awk -F',' '
  NR==1 { prev_end = $1 + $2; next }
  { gap = $1 - prev_end; print gap, "ns"; prev_end = $1 + $2 }
' /tmp/trace.csv | sort -n | uniq -c | tail -20
```

Typical numbers we have observed on RTX 4070 Ti:

| Mode | Median inter-kernel gap |
|---|---|
| Unbatched (`deviceWait` per dispatch) | **50 – 100 μs** |
| Batched (single encoder flush per token) | **0.5 – 2 μs** |
| Batched + `executeWithConfigCached` (PTX cached) | **0.3 – 1 μs** |

So a forward pass with ~170 dispatches/token at 50 μs gap = **8.5 ms
pure overhead**. Batching cuts this to ~0.2 ms.

#### B. Microbenchmark: empty dispatches

`Benchmarks/` has a "GPU fixed-cost microbenchmark" (`#28` in the
task list) that does N trivial dispatches and times them:

```lean
for _ in [0:1000] do
  GPUBackend.execute ctx noOpKernel [] (.dispatch1D 1)
```

Dividing wall-clock by N gives the per-launch fixed cost. On our
machine it is around **60 μs/dispatch (unbatched, CUDA)** and
**< 1 μs/dispatch (batched)**. For WebGPU the unbatched cost is
higher (~80 μs) because of an extra Dawn/Tint round-trip.

#### C. `withSection` vs nsys cross-check (how we caught the problem)

When dp4a was first wired in and TPS rose from 16 → 23:

```
withSection "ffnDown"   →  2.3 ms/tok
nsys kernel time sum    →  1.6 ms/tok
gap (= dispatch / sync) →  0.7 ms/tok
```

The 0.7 ms/tok gap was the direct evidence that dispatch overhead was
now comparable to actual compute — motivating the "batch all 170
dispatches per token" path that lives in `forwardSingleToken`.

**Action checklist when you see large gaps**:
1. Confirm `beginBatch` / `endBatch` are wrapping the hot loop.
2. Confirm `executeWithConfigCached` is being used (not
   `executeWithConfig`), so PTX generation doesn't re-run.
3. Confirm the `cacheRef` is actually populated on the second call
   (cached → `replayCached` fast path).
4. If all three hold and gaps are still large, the bottleneck is
   NVIDIA driver overhead itself, and the only remedy is **fewer
   dispatches** (fuse kernels).

### 2.5.6 Putting it together

| Tool | Granularity | Measures | When to use |
|---|---|---|---|
| Wall-clock loop | whole forward pass | GPU + host + sync | Final "is it faster?" check |
| `withSection` | code sections | GPU time + small host overhead (unbatched only) | "Which code section is the bottleneck?" |
| `LinearLayer` counters | per-shape LinearLayer calls | same as withSection, filtered | "Is this matmul or some other kernel?" |
| nsys kernel aggregate | per PTX entry point | true GPU kernel duration | "Is the kernel itself slow, or is it dispatch overhead?" |
| ncu metrics | inside one kernel | BW %, occupancy, stall reasons | "Why is this one kernel slow?" |

Always cross-check: if `withSection` says 5 ms but nsys says the
kernels only totalled 3 ms, the remaining 2 ms is host overhead
(dispatch latency, synchronization, state tracking). That's
actionable — batch more or reduce dispatch count.

---

## 3. Section profile (WebGPU)

`Examples/Gemma4Profile.lean` runs on the WebGPU backend and prints
per-section cumulative time using `withSection` labels.

```bash
lake exe gemma4-profile | tee /tmp/profile.txt
```

Example output:
```
Section breakdown (sorted by ms/tok)
  section          calls/tok   total_ms   ms/tok
  ffnDown   calls/tok=42  total=54.88 ms  ms/tok=5.49
  ffnGateUp calls/tok=42  total=53.46 ms  ms/tok=5.35
  ...
```

**How to read it**:
- `calls/tok=42` → called once per transformer layer → per-call cost
  is `ms/tok / 42`
- `calls/tok=1` → one-shot kernel (prefill, final-stage, lmHead, etc.)
- Rank by `ms/tok` to find the biggest targets

### Adding a section

```lean
Hesper.WGSL.Execute.withSection "myKernel" do
  ...
```

**Caveat**: `withSection` inserts a `deviceWait` after each dispatch,
so the run is serialized (not batched). Use it for attribution, not
for TPS measurement.

---

## 4. CUDA kernel-level profile (nsys)

### Installation

Add to `shell.nix` and re-enter `nix-shell`:

```nix
cudaPackages.nsight_systems     # nsys
cudaPackages.nsight_compute     # ncu
```

`nvprof` is **unsupported on compute capability ≥ 8.0** (Ampere / Ada /
Hopper). Always use `nsys` / `ncu`.

### Timeline collection

```bash
HESPER_DP4A=1 nsys profile \
  -t cuda -s none --cuda-memory-usage=false \
  -o /tmp/hesper.nsys-rep --force-overwrite=true \
  ./.lake/build/bin/gemma4-cuda data/gemma-4-e4b-it-Q4_K_M.gguf "Hello" 30

# Kernel aggregate (how much time each kernel spent)
nsys stats --report cuda_gpu_kern_sum --format csv /tmp/hesper.nsys-rep | head -30

# GPU trace (grid/block/duration of each launch)
nsys stats --report cuda_gpu_trace --format csv /tmp/hesper.nsys-rep \
  | awk -F',' 'NR>11 && $21 !~ /memset|memcpy/' | head -20
```

### Distinguishing kernels by name

The Hesper CUDA backend derives a unique PTX entry-point name
`k_<cacheKey>` from the `cacheKey` passed to `executeWithConfigCached`
(see `Backend/CUDA.lean`). Without this, every kernel would show up as
the generic `main`.

Example nsys output:
```
Time(%)  Total(ns)   Instances  Name
28.2%   264,534,943   30        k_1925851598382315  ← Q6_K lmHead (1/tok × 8.8ms)
26.3%   246,853,675  642        k_5262965136986155  ← Q4_K dp4a matmul (21.4/tok × 385μs)
20.5%   192,750,638 2574        k_1122526822368282  ← Q4_K dp4a matmul (85.8/tok × 75μs)
```

To reverse-lookup a `cacheKey` back to source, grep `Linear.lean` and
`Gemma4.lean` for the `hash (...)` expression. Common keys:

| hash tuple | kernel |
|---|---|
| `("q4k-dp4a-matmul", inDim, outDim)` | Q4_K × Q8_1 dp4a matmul |
| `("q8_1-quantize", inDim)` | Q8_1 quantization |
| `("q4k-lin-blockcoop-swpipe", …)` | f32 block-coop Q4_K matmul |
| `("gemma4_ce", name, config)` | small inline model kernels |

### Kernel internals (ncu)

When you need BW efficiency / occupancy for a single kernel:

```bash
# Profile only the first N launches (full profiling is very slow).
HESPER_DP4A=1 ncu --set basic --launch-count 20 \
  --target-processes all \
  ./.lake/build/bin/gemma4-cuda data/gemma-4-e4b-it-Q4_K_M.gguf "Hello" 5 \
  2>&1 | tee /tmp/ncu.txt

# Or filter by kernel name
ncu --kernel-name "k_5262965136986155" --launch-count 10 \
  --metrics gpu__time_duration.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed \
  ./.lake/build/bin/gemma4-cuda ...
```

**Key metrics**:
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` — effective
  DRAM bandwidth (%). Target ≥ 80%.
- `sm__throughput.avg.pct_of_peak_sustained_elapsed` — compute
  utilization (%).
- `smsp__warps_launched.avg.per_cycle_active` — warp occupancy.
- `smsp__warp_issue_stalled_*` — why warps stall (memory vs compute).

Rule of thumb:
- Low BW % (< 50%) → memory latency bound. Redesign data access.
- Low compute % AND low BW % → occupancy / scheduling problem. Try
  more warps, split-K, or software pipelining.

---

## 5. PTX-level analysis

### Dump the hesper-generated PTX

Inside a test, call `generatePTX` and write the string to a file:

```lean
let ptx := Hesper.CUDA.CodeGen.generatePTX
  "main" { x := 32, y := 1, z := 1 } myKernel
IO.FS.writeFile "/tmp/my_kernel.ptx" ptx
```

See `Tests/CUDA/CUDADP4ATest.lean` for a working example.

### Generate an nvcc reference PTX

Write the "correct" algorithm in C++ / CUDA and compile with nvcc so
you can diff against hesper's output:

```bash
cat > /tmp/ref_kernel.cu <<'EOF'
#include <cuda_runtime.h>
#include <cstdint>

extern "C" __global__ __launch_bounds__(32)
void ref_my_kernel(const float* x, uint32_t* y, int n) {
  // reference implementation
  ...
}
EOF

# PTX only, no executable
nvcc -arch=sm_89 -ptx -O2 /tmp/ref_kernel.cu -o /tmp/ref.ptx
```

### PTX diff

```bash
# Instruction count / structure comparison
grep -c 'dp4a\|shfl\|ld.global\|st.global' \
  /tmp/hesper_kernel.ptx /tmp/ref.ptx

# Specific-instruction context
grep -B 3 'dp4a.s32' /tmp/ref.ptx
grep -B 3 'dp4a.s32' /tmp/hesper_kernel.ptx
```

**Common differences you will see** (usually benign, but verify):
- `ld.global.nc.f32` (reference; uses read-only cache) vs
  `ld.global.f32` (hesper) — semantics identical, L1 efficiency differs.
- `cvta.to.global.u64` (reference; explicit address-space conversion)
  vs raw pointer — no functional difference on current drivers.
- `shfl.sync.bfly.b32 %r17|%p5` (reference; writes predicate) vs
  `shfl.sync.bfly.b32 %r17` (hesper) — ABI difference, same result.
- hesper PTX being much longer (e.g. 1000 lines vs 200) — codegen is
  more verbose. Correctness-wise OK, but worth investigating for perf.

### Pitfalls observed in practice

| Symptom | Root cause | Fix |
|---|---|---|
| Output is all zeros | `Exp.toI32` had no CodeGen branch → fell through to `u32(0)` | Add the case in `expToPTX` |
| Output has ±0.5 LSB systematic error | `cvt.rzi.u32.f32` (truncate) used for rounding | Switch to `cvt.rni.s32.f32` (round-to-nearest-even) |
| Negative matmul results become huge | `cvt.rn.f32.u32` (unsigned conversion of signed result) | Switch to `cvt.rn.f32.s32` |
| Wrong result with real Gemma 4 weights on Q4_K dp4a, right with synthetic ones | Over-allocated lanes (`pairIdx = tid/4 ∈ 0..7` reading non-existent sub-blocks 8..14) | `laneLow = tid & 15` + final `×0.5` reduce |
| Test harness gives IEEE 754 mismatches in f32 | f64 → f32 conversion was truncating the mantissa | Proper round-to-nearest-even conversion |

---

## 6. Known-answer testing

Scenario: verify a new kernel against a trusted reference.

1. **Generate synthetic input** (see `Tests/CUDA/CUDADP4ATest.lean`).
2. **Run the hesper kernel** and dump inputs/outputs to binary files
   (e.g. `/tmp/test_weights.bin`, `/tmp/test_q8input.bin`).
3. **Run an nvcc reference kernel** on the same binary files:

   ```cpp
   // /tmp/ref_run.cu skeleton
   int main(int argc, char** argv) {
     block_q4_K w;    fread(&w, ..., fopen(argv[1], "rb"));
     block_q8_1 x[8]; fread(x, ..., fopen(argv[2], "rb"));
     // allocate on device, H2D copy, launch ref_kernel<<<1,32>>>
     // D2H copy result
     printf("reference output: %f\n", out);
   }
   ```

   Build:
   ```bash
   nvcc -arch=sm_89 -O2 /tmp/ref_run.cu -o /tmp/ref_run \
     -lcudart_static -lcuda
   ```

4. **Compare**: hesper output and reference output should match within
   f32 ULP (typical tolerance: relative error < 1% to absorb Q8_1
   quantization noise).

---

## 7. Prioritizing optimization targets

From the nsys aggregate:

| Pattern | Diagnosis | Next action |
|---|---|---|
| Many calls × short time | Dispatch overhead dominates | Kernel fusion / batching |
| Few calls × long time | Single kernel is heavy | Improve BW efficiency / dp4a |
| High BW % (≥ 80%) | Memory-bandwidth saturated | Algorithm change (lower-bit quant, fewer ops) |
| Low BW % (< 30%) | Occupancy or latency problem | More warps, split-K, software pipelining |

**Concrete cases from Gemma 4**:
- `ffnDown` was 5.5 ms/tok at 16% BW efficiency → split-K tried (no
  improvement) → dp4a brought it to 40% and halved the time.
- `lmHead` (Q6_K, 1 call/tok, 4.5 ms) → f32 dequant bottleneck → dp4a
  is the next big win.

---

## 8. WebGPU-specific notes

### Dawn/Tint codegen surprises

Through `WGSL → SPIR-V → Vulkan → NVIDIA driver`, these can happen:

- **Compile-time `for [0:N]` unrolls**: with large N (e.g. 10 blocks)
  register pressure explodes, causing spill. Switch to runtime
  `ShaderM.loop`.
- **Large chains of inline `bitAnd/shiftRight`**: the compiler emits
  redundant ops. Materialize intermediates with `ShaderM.varNamed`.
- **`subgroupAdd` requires uniform control flow**: never call a
  subgroup op inside an `if_` body whose condition varies per lane.

### WebGPU profile does not show kernel internals

Chrome DevTools / dawn-node can time individual dispatches but cannot
report BW efficiency or warp stall reasons. For that you must port the
kernel to CUDA and run `ncu`.

### WebGPU limits

- Each grid dim ≤ 65535 workgroups → for `vocabSize = 262144` (Gemma 4
  lmHead) you need a 2D grid.
- A single `storage` binding is limited to 256 MiB → watch KV cache
  sizing vs `maxSeqLen`.

---

## 9. What to learn from `git log`

Past commit messages are a fast tour of prior optimizations:

```
fix(cuda): use round-to-nearest-even for Q8_1 quantization (not truncation)
test(cuda): dp4a Q4_K matmul VERIFIED against nvcc reference kernel
feat(cuda): dp4a (packed 4×int8 dot product) infrastructure + Q8_1/Q4_K kernels
perf(q4k): runtime block loop — eliminates compile-time unroll register spill
perf(cuda): use cuModuleLoadDataEx with optimization level 4
feat: mmap file I/O FFI + fromGGUFData for zero-redundant-read loading
```

Each commit should state the **observable symptom**, the **root cause**,
and the **fix**. That is the accountability trail for future readers.

---

## 10. Quick reference

| What you want | Command |
|---|---|
| TPS benchmark | `./.lake/build/bin/gemma4-cuda ... \| grep tokens/sec` |
| Section breakdown (WebGPU) | `lake exe gemma4-profile` |
| Kernel aggregate (CUDA) | `nsys profile … && nsys stats --report cuda_gpu_kern_sum` |
| Kernel internals (BW, occupancy) | `ncu --set basic --launch-count N` |
| Dump hesper PTX | Call `Hesper.CUDA.CodeGen.generatePTX` inside a test |
| nvcc reference PTX | `nvcc -arch=sm_89 -ptx -O2 ref.cu -o ref.ptx` |
| End-to-end correctness | Binary dump + nvcc reference run + value comparison |

---

Last updated: 2026-04-14 (right after wiring Q4_K dp4a into Gemma 4).
