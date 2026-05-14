# 35 — Measurement recipe: hesper stub decode vs llama.cpp

*Written 2026-04-23.  Run this whenever you want to re-measure the TPS
gap and the per-tax breakdown, so successive sessions are comparing
apples to apples.*

## Environment

- Hardware: RTX 4070 Ti (12 GB VRAM).
- Model: `data/gemma-4-e4b-it-Q4_K_M.gguf` (Gemma 4 E4B, Q4_K_M quant).
- Driver stack: CUDA 12.8, Nsight Systems 2024.6.2 (`nsys` on PATH).
- Canonical prompt: `"Hello world how are you"` (5 SPM tokens).
  Tokenises to `[9259, 1902, 1217, 659, 611]`.  Pick this one — the
  short-prompt correctness bug (issue #136) still breaks single-word
  prompts, so for honest TPS numbers always use ≥3 tokens.

All hesper runs assume `HESPER_DP4A=1` (CUDA dp4a Q4_K path).

## A. Quick TPS numbers (no profiler)

### A.1 — hesper stub, eager (no graphs, baseline)

```bash
HESPER_DP4A=1 lake exe gemma4-llama-prefill-skeleton \
  data/gemma-4-e4b-it-Q4_K_M.gguf "Hello world how are you" 10
```

Expected `tokens / sec` line: **~6.7 TPS**.  Look at the final
`───────────── Result ─────────────` block.  Per-decode wall time
prints on the `[decode N] ... ms` lines — should be ~68–70 ms/decode
in steady state.

### A.2 — hesper stub with CUDA Graphs

```bash
HESPER_DP4A=1 HESPER_LLAMA_GRAPHS=1 \
  lake exe gemma4-llama-prefill-skeleton \
  data/gemma-4-e4b-it-Q4_K_M.gguf "Hello world how are you" 10
```

Expected: **~9 TPS on 10-token, ~11 TPS on 20-token**.  The longer the
run the better graphs amortise the one-time capture cost.  Confirm
graphs actually kicked in:
- `[Graph] captured decode step (step=2)` appears once.
- `[decode N]` lines for N ≥ 3 report `0 dispatches` (one
  `cuGraphLaunch` replaces ~1500 individual launches).
- Token IDs match the `HESPER_LLAMA_GRAPHS=0` run decode-for-decode.

### A.3 — llama.cpp reference

```bash
llama.cpp/build/bin/llama-cli \
  -m data/gemma-4-e4b-it-Q4_K_M.gguf \
  -p "Hello world how are you" \
  -n 10 --no-warmup -ngl 99 --seed 0 --single-turn
```

Expected: **`[ Prompt: ~500 t/s | Generation: ~116 t/s ]`** at the tail.

> **Trap**: using `-no-cnv` (or no chat-mode flag) can drop into an
> interactive loop that waits on stdin.  `--single-turn` is the flag
> that cleanly produces one completion and exits.

## B. Full per-tax breakdown with nsys

### B.1 — capture an nsys report

```bash
mkdir -p /tmp/nsys-graphs
HESPER_DP4A=1 HESPER_LLAMA_GRAPHS=1 \
  nsys profile -t cuda,nvtx --stats=false \
    -o /tmp/nsys-graphs/hesper_graphs -f true \
  lake exe gemma4-llama-prefill-skeleton \
    data/gemma-4-e4b-it-Q4_K_M.gguf "Hello world how are you" 10
```

Produces `/tmp/nsys-graphs/hesper_graphs.nsys-rep`.  The nsys overhead
adds ~1–2 s to wall time but doesn't distort the per-call averages.

### B.2 — CUDA API summary (host-side tax)

```bash
nsys stats --report cuda_api_sum --format csv \
  /tmp/nsys-graphs/hesper_graphs.nsys-rep | head -25
```

Rows to watch:
| API                    | What it means                                     |
|------------------------|---------------------------------------------------|
| `cuStreamSynchronize`  | GPU-completion wait (mostly just GPU compute time)|
| `cuMemAlloc_v2`        | ScratchPool should keep this tiny (~0 steady-state)|
| `cuMemFree_v2`         | Same                                              |
| `cuMemcpyHtoD_v2`      | Weight load (one-time) + redundant const writes   |
| `cuMemcpyHtoDAsync_v2` | Per-step pinned→device (should be ~5 µs each)     |
| `cuLaunchKernel`       | Eager path; graphs path should show ~0            |
| `cuGraphLaunch`        | Graph replay — expect ~8 calls (1/decode + capture)|

Sanity checks after graphs are on:
- `cuGraphLaunch` count == number of decode steps that replayed.
- `cuLaunchKernel` ≈ prefill + step-1 warm-up + step-2 capture only,
  NOT ~1500/decode.
- `cuStreamSynchronize` Total Time divided by its call count gives the
  per-decode wall-time floor.

### B.3 — GPU kernel summary

```bash
nsys stats --report cuda_gpu_kern_sum --format csv \
  /tmp/nsys-graphs/hesper_graphs.nsys-rep | head -15
```

Sum the `Total Time (ns)` column (skip the CSV header) to get total
GPU kernel time across the run:

```bash
nsys stats --report cuda_gpu_kern_sum --format csv \
  /tmp/nsys-graphs/hesper_graphs.nsys-rep 2>/dev/null \
  | awk -F',' 'NR>1 && $2+0>0 {s+=$2} END {print "total GPU kernel ms=", s/1e6}'
```

Divide by number of forwards (1 prefill + N decode = N+1) to estimate
per-forward kernel time.

### B.4 — full GPU trace (for kernel-by-kernel latency)

```bash
nsys stats --report cuda_gpu_trace --format csv \
  /tmp/nsys-graphs/hesper_graphs.nsys-rep 2>/dev/null \
  | awk -F',' 'NR>1 && $2+0>0 {s+=$2} END {print "total GPU time ms=", s/1e6}'
```

The trace also covers memcpys / memsets — subtract kernel sum from
trace sum to see how much of GPU time is non-compute.

## C. Deriving the headline numbers

From B.2 and B.3, per-decode wall-time decomposes as:

```
wall_ms  = cuGraphLaunch_avg        ~0.15 ms
         + cuMemcpyHtoDAsync × 2    ~0.01 ms   (pinned slots)
         + cuMemcpyDtoH             ~0.13 ms   (logits read, 1 MB)
         + cuStreamSynchronize_rest = rest
```

`cuStreamSynchronize` average = TotalTime / calls.  That figure IS
essentially the GPU-work-plus-stall for that decode, so "non-kernel
host tax" is the first three rows summed — currently ~0.3 ms/decode.

## D. Comparison table to update

After each session, refresh the final row of the table in
`docs/llama-fusion-analysis/34-host-overhead-vs-llamacpp.md` with the
new numbers.  Columns:

| Tokens | Eager TPS | Graphs TPS | ms/decode | GPU kernel ms | Host tax ms |
|-------:|----------:|-----------:|----------:|--------------:|------------:|

Last measured (2026-04-23, commit `0cc5f87`):

|   5    |    5.72   |    6.26    |     —     |      —        |    ~0.3     |
|  10    |    6.72   |    9.02    |   58.9    |   ~13 (?)     |    ~0.3     |
|  20    |    7.14   |   11.20    |   58.9    |      —        |    ~0.3     |

llama.cpp reference: 116.2 TPS, 8.6 ms/token, host tax ~0.3 ms, GPU
kernel ~8.3 ms.

## E. Correctness spot-check

Before trusting a TPS number, confirm the model still produces the
right tokens.  Compare the `[decode N] startPos=P next=T` lines
between eager and graphs runs — token IDs must match 1:1.  The
canonical 10-token output for `"Hello world how are you"` is:

```
[decode 1] next=108
[decode 2] next=9259 "Hello"
[decode 3] next=1902 "world"
[decode 4] next=236888 "!"
[decode 5] next=108
[decode 6] next=9259 "Hello"
... (repeats)
```

If graphs path diverges, you broke capture — roll back before trusting
any TPS delta.

## F. What NOT to do

- **Don't** use `-no-cnv` with llama.cpp — it hangs on stdin.
- **Don't** measure with profiler disabled on the first run after a
  rebuild; PTX JIT on first kernel load adds 50–100 ms and skews TPS.
  Warm the binary once, then measure.
- **Don't** compare prefill throughput across different prompt
  lengths — prefill kernels are shape-specialised, so 5-token vs
  50-token prefill numbers aren't interchangeable.
- **Don't** trust `cuStreamSynchronize` as a "stall" measurement —
  it's mostly just the GPU completing its work.  The real host tax is
  the sum of the *other* cu* calls.
