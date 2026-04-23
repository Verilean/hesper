# 37 — Q4_K widen + mis-measured dispatch-count detour (retracted)

*Written 2026-04-23.  Small Q4_K win landed; then I mis-reasoned the
rest of this doc and withdrew the conclusion in revision below.*

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

## What I got wrong (retracted)

The first revision of this doc claimed:
- "hesper emits 1491 dispatches/decode, llama.cpp emits 89"
- "per-decode 45 ms are GPU-side scheduling gaps from 1491 × 30 µs"
- "per-kernel tuning has a hard ceiling at ~20–26 TPS"

All three are wrong.  Recovered reality after re-measuring with
nsys on llama.cpp directly (doc 35 §A.3 workload):

| Metric                           | hesper stub | llama.cpp |
|----------------------------------|------------:|----------:|
| kernel instances / 11 forwards   | ~13 500     | **5 615** |
| per-decode kernel dispatches     | ~1 491      | **~510**  |
| per-decode GPU kernel time       | ~11 ms      | ~3 ms     |
| per-decode wall time             | 58 ms       | 8.6 ms    |

The "89" figure I used for llama.cpp came from a count of **op rows in
`trace_lc.txt`**, which is ggml's *logical* op count.  ggml's graph
batcher collapses several of those ops into each `cudaLaunchKernel`
(multi-row matmul + epilogue etc.), but each logical op is NOT one
kernel launch.  And conversely, a single ggml op can expand to
multiple kernels (quantize_q8_1 + mul_mat_vec_q + combine, etc.).

Re-measured with nsys: llama.cpp actually issues ~510
`cudaLaunchKernel`/decode.  That's 2.9× fewer than hesper's 1491,
not 17×.  dispatch count matters but it's not the 45 ms fire-alarm I
painted it as.

And the "30 µs per-kernel gap" is wrong too — if it were real,
llama.cpp's 510 × 30 µs = 15 ms would already exceed its total
8.6 ms/decode budget.  Inside a CUDA graph, adjacent-kernel gaps are
much smaller (~1–3 µs median), and wall time is dominated by
kernel-active time, not gaps.

## Where the gap actually sits (corrected)

Per decode:

| Bucket                | hesper   | llama.cpp | Gap     |
|-----------------------|---------:|----------:|--------:|
| Host API (pre-graph)  |  ~0.3 ms |   ~0.3 ms | ≈0      |
| GPU kernel active     |   ~11 ms |    ~3 ms  | **8 ms**|
| Kernel-count overhead |  ~2–3 ms |   ~1 ms   | ~2 ms   |
| Other (scheduling)    | ~43 ms   |   ~4 ms   | ??      |

That **~43 ms "other"** is the real unknown.  It's still way larger
than CUDA's theoretical scheduling cost for 1491 kernels.  Candidate
explanations to rule in or out:
- Hidden serialisation inside the graph (missing barriers forcing
  scoreboard stalls).
- FFI / driver per-launch overhead that nsys doesn't attribute.
- `main` kernel in nsys sum is collapsing many actual kernels whose
  individual latencies nsys is under-reporting.

Without an answer here, I can't honestly predict what per-kernel
improvements buy.  Doc 36's priority list should stand, but with the
caveat that each win should be re-measured *end-to-end TPS* — not
just "this kernel got faster by X ms".

## Revised plan

1. **Finish doc 36 priority 1** — continue Q4_K matmul work beyond
   the 4-warp widen (inline Q8_1 quantize into the matmul, multi-row
   tiling for ≤5120 outDim, ncu to find the actual stall reason).
2. **Instrument the 43 ms "other" bucket** before pivoting.  Options:
   enable nsys GPU metric export, or run a small test with KernelTrace
   that measures wall - kernel_active and attributes it.
3. **Only then** decide if dispatch-count work (kernel fusion) is the
   right next lever.  The earlier "fusion is mandatory" claim is
   withdrawn.

## Lesson

When a number looks suspiciously round (89 ops/token?), re-measure
before building a plan on top of it.  I let a stale figure from an
old doc propagate into a false conclusion.
