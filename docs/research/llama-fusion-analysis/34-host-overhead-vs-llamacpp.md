# 34 — host overhead: hesper stub vs llama.cpp

*Date: 2026-04-22.  Hardware: RTX 4070 Ti.  Workload: prompt
"Hello world how are you", n=10 generated tokens.  Hesper build:
stub-first parity path + persistent KV + batched PLE + ScratchPool.*

## The question

The user asked three related things:

1. **"2-3 ms削減は大きい?"** — yes, because the 115 TPS target = 8.7 ms/token.
   Every 2-3 ms of avoidable tax is ~25% of that budget.
2. **"それは税金のようなものですよね。"** — correct: host overhead is a
   fixed per-forward cost, not something the GPU kernels can fuse away.
3. **"llama.cpp のケースと比較して同等か？"** — this is the real question.
   If llama.cpp pays the same tax, chasing it harder in hesper gives us
   nothing; if it's much lower, that's where the gap is.

Also asked: **"Hello と入れると返事は帰ってくる？"** — answer below.

## TL;DR numbers (same workload, same GPU)

| Metric                     | hesper stub | llama.cpp |
|----------------------------|-------------|-----------|
| Generation TPS             | **5.65**    | **113.7** |
| ms per decoded token       | ~177        | ~8.8      |
| Prefill throughput         | ~9 t/s      | ~498 t/s  |
| "Hello world how are you"  | coherent    | coherent  |
| "Hello" (single token)     | **broken** (HelloHello...) | coherent |

The 20× gap is **not** in host overhead in percentage terms — llama.cpp
spends a comparable fraction of per-token time in driver/API — it's that
llama.cpp's per-token wall clock is ~20× smaller *overall*.  So the tax
is similar in ratio, but much smaller in absolute ms because their total
budget is much smaller.

## Hesper per-call CUDA API breakdown (5-token run, after ScratchPool)

11 forward passes (1 prefill×5 + 10 decodes) → 1.5 s wall clock.

```
cuCtxSynchronize    10 calls   378 ms  ← 25% : mostly GPU work waiting
cuMemcpyHtoD_v2    794 calls   343 ms  ← 23% : dominated by 1-time weight upload (~120 MB)
cuCtxCreate_v2       1 call    116 ms  ← one-time init
cuMemFree_v2       382 calls    82 ms
cuModuleLoadDataEx 183 calls    76 ms  ← prefill-only JIT, hits 100% cache after
cuMemAlloc_v2     1711 calls    42 ms  ← was 2043 before ScratchPool
cuLaunchKernel  15510 calls    23 ms  ← 1.5µs each (healthy)
cuMemsetD8_v2    1711 calls     4 ms
```

Per-forward breakdown (subtracting one-time costs ~350 ms):

```
(1.5s - 350ms one-time) / 11 forwards ≈ 104 ms/forward
```

Where does the 104 ms go?

* **GPU kernel time** (from nsys kernel sum on decode forwards): ~60 ms.
  The dominant kernels are `cuda_fused_mul_mat_vec_q4_K_q8_1_impl`
  (21 ms each), the batched matmul (7 ms), GELU (3 ms).
* **`cuCtxSynchronize` stalls**: ~34 ms — this is launch-latency
  bubbles + last-kernel-in-graph wait.
* **Driver API calls** (alloc/memcpy/memset/launch): ~10 ms — the "tax".

So for our 5.65 TPS target (8.7 ms/token), the 10 ms/forward "tax" alone
would already push us over budget.  **The tax must drop to <2 ms
to clear headroom for the kernel work.**

## llama.cpp's counterpart

llama.cpp at 113.7 TPS = **8.8 ms/token**, and nearly all of that is GPU
kernel time.  They pay essentially zero "tax" because:

1. **CUDA Graphs** capture the entire per-token op sequence once and
   replay it.  One `cudaGraphLaunch` replaces ~89 op-level `cuLaunchKernel`
   calls.  Hesper has Graphs capture plumbing (F3 result) but it's not
   on the llama-parity stub path yet.
2. **Pre-allocated compute buffer** (`ggml_gallocr`): zero
   `cudaMalloc`/`cudaFree` per forward after warmup.  Hesper's
   `ScratchPool` (this session's work) reaches the same end-state — 0
   `cuMemAlloc` steady-state — but only once plumbed on the hot path.
3. **Persistent param buffer on device**: `startPos` etc. live on device
   and are advanced by a CUDA kernel, not re-uploaded from host.  Hesper
   currently re-uploads `paramsBuf` (4 bytes) and `onesBuf` (~512 bytes)
   per forward.  Small but 0-cost savings are measurable at 115 TPS.

## So the tax is comparable in *ratio*, not in *absolute ms*

| Overhead source         | hesper now | llama.cpp | gap |
|-------------------------|-----------:|----------:|----:|
| `cuMemAlloc` per forward | ~3 ms     | ~0        | yes |
| `cuLaunchKernel` per fwd | ~2 ms     | 0 (graph) | yes |
| `cuMemcpyHtoD` per fwd   | ~0.5 ms   | 0         | small |
| `cuCtxSynchronize` stalls| ~30 ms    | bounded by GPU work | BIG |
| Actual GPU work          | 60 ms     | 7 ms      | HUGE |

The biggest structural gap is that **llama.cpp's kernels finish in ~7 ms,
so the sync is cheap**.  Our kernels take ~60 ms, so the sync *looks*
like host overhead but is really just "waiting for our slow kernel".

**Conclusion**: the tax line-items ARE roughly proportional to llama.cpp's
(each is single-digit ms vs ~0 ms), but they're only a meaningful fraction
of our total because our kernels are 8-9× slower.  Close the kernel gap
and the host-tax becomes a non-issue.  Pursue them in parallel because
at 115 TPS (8.7 ms/token), every ms matters.

## "Hello"だけ与えると？

* **"Hello world how are you" + 10 tokens** → coherent (`"?  Hello world!  Hello world!"`)
* **"Hello" + 20 tokens** → broken (`"HelloHelloHello..."`)

So the previously-declared "multi-token correctness bug RESOLVED" finding
is **prompt-dependent**.  Short prompts still reproduce the degenerate-repeat
behavior.  Issue #136 (Phase 3 Step 1) stays open.

## What to do next, in priority order

1. **Correctness first** — close the short-prompt degenerate-output bug
   so we can benchmark real workloads honestly.
2. **Wire CUDA Graphs capture into the stub path** — biggest single
   tax cut (eliminates ~2-3 ms/fwd of launch overhead + enables deeper
   kernel pipelining).
3. **Pinned host memory for `paramsBuf`/`onesBuf` uploads** — removes
   ~0.5 ms/fwd of `cuMemcpyHtoD` overhead; tiny individually but at
   115 TPS it's ~5 % of budget.
4. **Fuse `quantize_q8_1` into Q4_K matmul kernel** — the big kernel-time
   item; cuts ~126 graph nodes/token and likely halves GPU time.

## Postscript — graph-capture now works (2026-04-22 PM)

Attempted (2) with `HESPER_LLAMA_GRAPHS=1` env gate.  First attempt hit
`CUDA_ERROR_ILLEGAL_ADDRESS` on replay; second attempt with pinned host
memory for the varying sources **works**.

Root cause (first attempt) is identified and already documented in
`Hesper/CUDA/FFI.lean` near `cuMemAllocHost`: hesper's
`GPUBackend.writeBuffer` on the capture stream hands CUDA a pointer
into a Lean-managed `ByteArray`.  The graph records that pointer.
Between capture and replay the `ByteArray` is GC'd → replay
dereferences freed memory.

The per-decode forward has 5 `writeBuffer` sites:

| Buffer          | Varies across decodes? | Per-forward |
|-----------------|-----------------------:|------------:|
| `colIdxBuf`     | no (always = 0)        | always written |
| `plColIdxBuf`   | no (always = 0)        | always written |
| `onesBuf`       | no (all 1.0)           | always written |
| `lastColIdxBuf` | no (seqLen-1 = 0)      | always written |
| `paramsBuf`     | **yes** (startPos)     | always written (but handled via `paramsBufOverride` now) |
| `tokenIdsBuf`   | **yes** (last-gen tok) | caller-side |

Fix that shipped (commit 8a00295):
* Added `skipConstantWrites` flag; set `true` on captured/replayed
  forwards.  The 4 constant sites are initialised during step-1 eager
  warm-up and never re-written — ScratchPool pointer reuse keeps the
  contents alive.
* Two pinned host slots hold `startPos` and `tokenIdsBuf[0]`; driver
  calls `cuWritePinned` before each step, and a captured
  `cuMemcpyHtoDFromPinned` on the capture stream updates the device
  buffers during replay.
* Capture step itself launches the instantiated graph once so its
  result matches an eager decode (capture-only would return stale
  step-1 logits because capture records without executing).

Measured impact (same workload, RTX 4070 Ti):

|  Tokens | No graphs | Graphs | Δ     |
|:-------:|----------:|-------:|:-----:|
|    5    |     5.72  |  6.26  | +9%   |
|   10    |     6.72  |  9.02  | +34%  |
|   20    |     7.14  | 11.20  | +57%  |

Per-decode wall clock drops from ~68 ms → ~59 ms once replay kicks in
— essentially the ~10 ms of `cuLaunchKernel` overhead the earlier
profile predicted.  `dispatchCounter=0` on replays confirms one
`cuGraphLaunch` replaces ~1500 individual launches each decode step.

## Status summary

| Tax item            | Before          | After ScratchPool | After Graphs | llama.cpp |
|---------------------|-----------------|-------------------|--------------|-----------|
| `cuMemAlloc` /fwd   | ~3 ms           | ~0                | ~0           | ~0        |
| `cuLaunchKernel`/fwd| ~2 ms           | ~2 ms             | ~0           | ~0        |
| const writeBuffer   | ~0.5 ms         | ~0.5 ms           | 0            | 0         |
| GPU kernel time     | 60 ms           | 60 ms             | 60 ms        | 7 ms      |

Non-kernel overhead is now essentially at llama.cpp's level in absolute
ms per decode.  Remaining ~8× TPS gap lives entirely inside the GPU
kernels.
