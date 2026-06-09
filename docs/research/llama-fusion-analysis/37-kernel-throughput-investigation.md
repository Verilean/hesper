# 37 — Q4_K widen + retracted detour (second correction)

*Written 2026-04-23.  Small Q4_K win landed; then I mis-reasoned the
rest of this doc TWICE.  Second correction leaves us back where doc 36
started: per-kernel throughput is the right lever.*

## What actually landed

`Hesper/Layers/Linear.lean` — removed the `outDim ≤ 5120` gate on the
4-warp Q4_K kernel path.  ffn_gate and ffn_up (outDim=10240) now use
the 4-warp kernel instead of 1-warp.  Opt-out flag
`HESPER_Q4K_4WARP_WIDE=0`.

Measured impact on the canonical 10-token workload:

| Metric                        | Before      | After       |
|-------------------------------|------------:|------------:|
| Total GPU kernel time (nsys)  | 151 ms      | **142 ms**  |
| ffn_gate/up kernel/call       | ~73 µs      | **~60 µs**  |

Small but clean — keeps.

## What I got wrong, version 1 (already withdrawn)

Claimed "llama.cpp = 89 ops/token".  That's ggml's logical op count in
`trace_lc.txt`, not CUDA kernel launches.

## What I got wrong, version 2 (this correction)

After version 1, I re-measured llama.cpp (graphs ON) and claimed
"~510 dispatches/decode" for llama.cpp.  That came from naive
division of nsys's total kernel instances (5615) by my assumed
forward count (11).  Both the numerator and the denominator were
wrong:

- With CUDA Graphs ON, llama.cpp emits 16 `cudaGraphLaunch` calls
  that each replay ~hundreds of kernel nodes; kernel instance count
  isn't the same as launch count.
- Internal forward count for `llama-cli … -n 10` isn't 11 — there are
  additional preparatory forwards, tokenize paths, etc.

The apples-to-apples measurement uses `llama-bench` with
`GGML_CUDA_DISABLE_GRAPHS=1` (same flags Doc 33 used):

| Config                                | dispatches |
|---------------------------------------|-----------:|
| llama.cpp  prefill seqLen=50, n=0     |    2 016   |
| llama.cpp  decode only, n=20          |   26 020   |
| → per decode                          |    **1 301** |
| hesper stub decode (graphs OFF)       |    1 491   |
| → ratio                               |   **1.15×** |

**Hesper is within 15% of llama.cpp on dispatch count**, exactly as
doc 33's plan predicted and previous measurements recorded.  Kernel
fusion isn't the missing lever.

## Corrected per-decode attribution

At steady-state decode with hesper graphs ON (58 ms/decode wall):

| Bucket                              | hesper   | llama.cpp | Gap    |
|-------------------------------------|---------:|----------:|-------:|
| Host API                            |  ~0.3 ms |   ~0.3 ms | ~0     |
| GPU kernel active (sum of durations)| **~13 ms** | ~3 ms  | 10 ms  |
| Schedule/driver overhead (residual) |  ~44 ms  |  ~5 ms    | 39 ms  |

That 39 ms residual is real and still unexplained — but with
dispatch counts comparable, it can't be framed as a
"dispatch-gap-arithmetic" story.  Candidate explanations (need
verification, not more hand-waving):

1. **`cudaMallocHost` / host-buffer cost**: llama.cpp's nsys showed
   `cudaMallocHost` at 94.8 ms total — prefill one-shot.  Does
   hesper have an equivalent cost per-decode somewhere?
2. **CUDA Graph quality on the hesper side**: our capture-then-replay
   works but the graph may be serialising across kernels that CUDA
   normally pipelines.  Need to look at the graph's dep structure in
   nsys timeline view, not just the summary.
3. **Per-kernel launch-to-start latency on the GPU**: nsys's "Total
   Time" for kernels is occupancy time, not wall time.  If our
   kernels stall waiting for scoreboard clears between launches,
   that doesn't show up in the kernel sum.

## Revised plan (back to doc 36's priority list)

1. **Q4_K matmul per-kernel throughput** — the 4-warp widen landed
   (-9 ms GPU).  Next: port `mul_mat_vec_q` inline-q8 path (removes
   the standalone quantize dispatch, ~42 fewer launches per decode),
   try multi-row variants for outDim ≤ 5120.
2. **FlashAttention vec-f16 decode path** — currently the FLASH_ATTN
   kernel is ~76 µs, llama.cpp's is ~15-25 µs.  Port fattn-vec-f16.
3. **Instrument the 39 ms residual** alongside #1 and #2 — don't
   pivot to fusion without evidence.

## Lesson learned, logged

When a dispatch count looks wildly off between us and llama.cpp,
re-run with **identical flags** (same graphs setting, same prompt,
same n tokens) and use nsys kernel instance counts — NOT op-row
counts from trace files.  Both failures above came from comparing
different measurement setups and calling the diff a "gap".
