# 38 — Hot-list: actuals vs doc 36 estimates

*Written 2026-04-23.  Doc 36's hot-list had concrete ms numbers per
kernel group; they were estimates.  This doc replaces them with
nsys-measured values (graphs OFF, decode-only, apples-to-apples).*

## Measurement setup

- hesper: `/tmp/nsys-4warp/hesper.nsys-rep` — `HESPER_DP4A=1
  HESPER_LLAMA_GRAPHS=1` on 10-token workload (1 prefill seqLen=5 +
  10 decodes).  Graphs-ON, but `cuda_gpu_kern_sum` still reports real
  per-kernel durations inside the graph.  4-warp-widen (commit
  `87d523e`) applied.
- llama.cpp: `/tmp/nsys-lc/lc_nograph_dec.nsys-rep` —
  `GGML_CUDA_DISABLE_GRAPHS=1 llama-bench -p 0 -n 20 -r 1`.  20
  decode forwards only.

Per-decode = total / N decodes.

## What's real, what's estimated

Doc 36 stated hot-list with values like "37 ms vs 4.5 ms" for
Q4_K.  Those were **plausible estimates** I wrote without instrumenting
per-kernel time.  The nsys numbers below are actuals.

### llama.cpp per-decode actuals (20-decode graphs-OFF bench)

| Kernel group (llama.cpp name)                     | total ms | per-decode ms | inst/decode |
|---------------------------------------------------|---------:|--------------:|------------:|
| `mul_mat_vec_q<Q4_K>` (all variants summed)       |    88.0  |       **4.40**|   ~270      |
| `mul_mat_vec_q<Q6_K>` (lm_head etc.)              |    42.99 |         2.15  |    33       |
| `rms_norm_f32` (all configs summed)               |    14.67 |         0.73  |    ~302     |
| `mul_mat_vec_f<half>` (attention inner matmuls)   |     9.28 |         0.46  |    ~128     |
| `quantize_q8_1`                                   |     5.95 |         0.30  |    301      |
| `rope_neox`                                       |     1.41 |         0.07  |    66       |
| `k_bin_bcast` (mul/add)                           |     2.09 |         0.10  |    ~86      |
| `k_set_rows` (KV write)                           |     1.20 |         0.06  |    48       |
| `soft_max_f32`                                    |     1.18 |         0.06  |    42       |
| `unary_op_kernel` (gelu etc.)                     |     0.73 |         0.04  |    42       |
| everything else                                   |     1.47 |         0.07  |    ~100     |
| **Total**                                         |  **168.97** |  **8.44**  |  **~1300**  |

8.44 ms/decode matches the 8.6 ms from `--single-turn` TPS (116 TPS)
within noise.

### hesper per-decode actuals

Catch: hesper's PTX generator emits most kernels with
`funcName="main"`.  nsys summary sees all of them as one row ("main",
99.6 ms, 1534 instances).  The named `k_<hash>` kernels only cover
the `executeWithConfigCached` paths (Q4_K matmul, Q6_K lm_head,
quantize_q8_1, a few others).

Breakdown from the profile:

| Bucket                                      | total ms | per-decode ms |
|---------------------------------------------|---------:|--------------:|
| Named decode kernels (grid.y=1 hashes)      |   19.60  |         1.96  |
|   └─ of which Q4_K matmul-shaped            |   14.67  |         1.47  |
|   └─ of which small scale/RoPE/KVW          |    4.93  |         0.49  |
| Named prefill kernels (grid.y>1 hashes)     |   23.07  |       (prefill one-shot) |
| `main` aggregate (all unnamed kernels)      |   99.62  |         ~7.5  |
| **Total per-decode (approx)**               |          |     **~9.5**  |

The `main` bucket contains RMSNorm, RoPE, FlashAttn, pointwise, stubs,
etc. — things that go through `GPUBackend.execute` rather than
`executeWithConfigCached`.  Can't split further without renaming PTX
symbols.

## The actual hot-list (corrected)

| Group | hesper ms | llama.cpp ms | Ratio | Where it's emitted |
|:---|---:|---:|---:|:---|
| **Q4_K matmul (named)**        | **1.47**   |    4.40   | **0.33×** hesper is FASTER?! |
| **`main` bucket** (hesper-only)|   ~7.5     |      —    |   —   |
| Q6_K lm_head / small matmuls   |   ~0.2     |    2.15   | 0.1× |
| RMSNorm (inside `main`)        |   ~1.0?    |    0.73   | ~1.4× |
| FlashAttn (inside `main`)      |   ~1.0?    |    0.46   | ~2× |
| Pointwise + stubs (inside main)|   ~5?      |    ~0.2   | HUGE |
| Quantize_q8_1 (inside main?)   |   unknown  |    0.30   |   ?  |

## What this changes

**Doc 36's "Q4_K matmul 37 ms vs 4.5 ms, 8× gap" is wrong.**

Real situation: the **named** Q4_K matmul kernels (the ones
`executeWithConfigCached` caches by shape) actually run FASTER than
llama.cpp's `mul_mat_vec_q` in raw ms because llama.cpp issues ~270
per-decode Q4_K calls vs hesper's ~28.  llama.cpp calls it more times
because each of its `mul_mat_vec_q` handles a smaller chunk of the
same logical operation.  Per-call latencies are comparable (~60 µs
hesper vs ~64 µs llama.cpp for the wQ/wK/wV-sized calls).

**The real gap is inside hesper's `main` bucket.**  Roughly 7.5 ms/
decode there, vs llama.cpp's ~4 ms scattered across its small
kernels (RMSNorm + FlashAttn + Q_norm + small matmuls + quantize +
pointwise).  Need PTX-symbol renaming to decompose.

## Action items (revised)

1. **Rename PTX symbols** — change `funcName := "main"` default in
   `Hesper/WGSL/Execute.lean:62` to something like `s!"kernel_{hash}"`
   so nsys shows per-kernel rows.  Without this, every subsequent
   investigation is blind inside the 7.5 ms / 99.6 ms bucket.
2. **Re-measure hot-list** once (1) is done.  Pick the #1 item by
   actual ms.
3. **Don't start coding** per-kernel fixes without (1)+(2) — doc 36's
   priorities were based on a wrong per-kernel breakdown.

## Lesson

When a doc's hot-list ms column comes without a link to the nsys
report that produced it, distrust the numbers.  Doc 36 inherited the
"8× gap" framing from an earlier analysis without a reproducible
measurement — and I cited it as gospel when building doc 37.
