# Host overhead breakdown — hesper vs llama-cli (2026-04-25)

## Setup

- prompt: `"The capital of France is and the population is"` (P=11 tokens)
- decode: 300 tokens (`-n 300`, `--single-turn` for llama-cli, `HESPER_IGNORE_EOS=1` for hesper)
- graphs OFF on both engines
- model: `gemma-4-e4b-it-Q4_K_M.gguf`
- GPU: RTX 4070 Ti (12 GiB)
- script: `scripts/perf_compare.sh both`, single-case mode (pLgN only)
- output dir: `/dev/shm/perf_compare` (tmpfs — sqlite is 35-120 MB and was the
  dominant nsys post-process cost on ext4)

## Per-token breakdown

|                                | hesper    | llama-cli | delta      |
|--------------------------------|----------:|----------:|-----------:|
| GPU kernel time                | 10.9 ms   | 8.4 ms    | +2.5 ms    |
| GPU memcpy time                |  0.7 ms   | 0.7 ms    |  =         |
| **GPU busy (kernel+memcpy)**   | **11.6 ms** | **9.0 ms** | **+2.5 ms** |
| Host: cudaLaunchKernel         |  1.12 ms  | 1.83 ms   | -0.7 ms    |
| Host: HtoD (write logits)      |  0.67 ms  | 0.68 ms   |  =         |
| **Host: DtoH (argmax read)**   | **9.80 ms** | **0.00 ms** | +9.8 ms |
| **Host: cuStreamSync (drain)** | **0.00 ms** | **6.79 ms** | -6.8 ms |
| Host total in CUDA API         | 11.6 ms   | 9.3 ms    | +2.3 ms    |
| **Wall / token (1/TPS)**       | ~17 ms    | ~9 ms     | +8 ms      |
| **Decode TPS**                 | ~60       | ~110      |            |

`cudaLaunchKernel` count: hesper 924/tok, llama-cli 1220/tok — hesper actually
launches **fewer** kernels per token.

## The "9.8 ms host overhead" myth

Earlier sessions wrote that hesper had ~5-6 ms of "Lean tail latency" and
attributed it to GC / large-object refcount. That was wrong. The truth is:

- hesper's `cuMemcpyDtoH_v2` of the 4-byte argmax result is implicitly
  synchronous on stream 0, so the **9.80 ms attributed to DtoH** is actually
  **GPU drain wait** — the host is blocked inside the driver until the last
  kernel of the forward finishes.
- llama-cli does the same blocking, but it issues an explicit
  `cudaStreamSynchronize` first (6.79 ms in the table), then a non-blocking
  DtoH that takes essentially zero time. The attributions differ; the
  semantics are identical.
- Of hesper's 9.80 ms DtoH, the part that is *truly* extra wall time over
  llama-cli is `9.80 + 0.00 - 0.00 - 6.79 = 3.01 ms`. That 3 ms is just the
  **same difference as the kernel-time gap** (10.9 - 8.4 ≈ 2.5 ms): hesper's
  GPU finishes 2.5 ms later, so the drain takes ~2.5 ms longer.

So the dominant wall-clock gap is **kernel speed**, not host code.

## Lean tail latency contribution

`perf record cycles:u --no-children` on hesper's pLgN run shows:

| Symbol                                   | Self % | What it is |
|------------------------------------------|-------:|------------|
| `0x00000000001f44e1` (libcuda)           | 14.5 % | driver sync wait |
| `__memmove_avx_unaligned_erms`           | 13.4 % | Lean Array COW expand |
| `[k] 0xffffffff952012f0` (kernel)        | 10.4 % | sched / IRQ |
| `lean_dec_ref_cold`                      |  7.6 % | refcount drop slow path |
| `lean_copy_expand_array`                 |  6.6 % | Array realloc for IR replay |

`lean_dec_ref_cold + lean_copy_expand_array = 14.2 %` of CPU cycles. But the
absolute wall those cycles consume is **bounded by the time the host is NOT
blocked in the driver**. Host total in CUDA API is 11.6 ms / 17 ms wall =
68 %, leaving ~5.4 ms of unblocked host time per token. 14 % of 5.4 ms ≈
**0.75 ms** of wall actually goes to those Lean refcount/expand paths.

In other words, even if every Lean refcount and Array COW disappeared, wall
would only drop ~0.75 ms (60 → 63 TPS). The remaining +5.5 ms gap to
llama-cli is **all in kernel speed + drain wait**.

## Where to spend effort

In priority order to close the 60 → 110 TPS gap:

1. **Kernel speed** (+2.5 ms / +18 TPS budget) — task #47, Q4_K matmul.
   Same level of effort that brought RMSNorm and Q6_K to llama.cpp parity.
2. **Eliminate the 9.8 ms DtoH drain bubble** by doing on-device argmax +
   token feedback (graphs-OFF pipelined decode). This is task #229 attempt
   redux — was reverted earlier because the test methodology conflated
   wall-clock with TPS denominator. Worth retrying with this measurement
   harness.
3. Lean Array overhead in `Hesper.Circuit.CompiledCircuit.replay` —
   skip the Array allocation in the replay loop entirely (the dispatch
   list is fixed per-cache-key). Bounded benefit ~0.75 ms.

## Methodology notes — things that broke during this run

Recording these because they wasted hours:

- **`llama-cli -no-cnv` does not actually disable interactive mode** when
  stdin is `/dev/null`. EOS triggers a re-prompt loop that produces 1.9 GB
  of `> ` output and runs nsys for 30+ minutes. **`--single-turn` works**.
- **`--ignore-eos` *implies* interactive mode** in this build of llama-cli
  (per `--help`). Using it together with `-no-cnv` does not help; it
  re-enables the chat loop.
- **nsys leaves `--start-agent` daemons running after `timeout --signal=KILL`
  of the launcher**. Six orphan agents from previous failed runs were each
  consuming 90 % CPU when this investigation started, contaminating every
  measurement. Trap installed in `perf_compare.sh` to kill them on exit.
- **`nsys profile --stats=true` (the default) post-processes traces into
  sqlite synchronously**, so a 100k-CUDA-event run can hang the host for
  60 seconds after the workload finishes. `--stats=false` skips it; we
  call `nsys stats` only when we actually need the table.
- **`--export=none` breaks `nsys stats --force-export=true` later** because
  some nsys versions need the original sqlite to derive new reports.
  Leave the auto-export on, just skip the stats pass.
- **Default `OUT_DIR=/dev/shm/perf_compare`** (tmpfs) eliminates fsync
  latency on the 35-120 MB sqlite files; on ext4 those writes were the
  dominant cost of the post-process pass.
- **3-case linear regression** for load / prefill / decode separation was
  fragile — short-decode cases (1 token) are dominated by post-process
  overhead, not GPU work, so the regression slope was noise. Single-case
  pLgN with N=300 is ~4 % biased by prefill-in-decode but reliable.
