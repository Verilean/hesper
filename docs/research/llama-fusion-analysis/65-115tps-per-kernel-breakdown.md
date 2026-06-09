---
name: 115 TPS root cause = cuStreamSynchronize 11 ms/tok
description: nsys cuda_api_sum on hesper graphs ON shows 670 ms / 740 ms = 90% wall is in cuStreamSynchronize (60 calls × 11.16 ms). This is the GPU drain wait between cuGraphLaunch and the host reading argmax. NOT graph launch overhead, NOT DtoH itself, NOT kernel speed. The fix is host/GPU pipelining (next-token host work || current-token GPU drain).
type: project
originSessionId: b07a5aea-3af6-4635-9116-063496d475d5
---
## What nsys says (hesper graphs ON, "Hello world how are you" 60 tok, 740 ms wall)

```
Time%  Total ms   Calls  Avg/call    Name
59.6    669.9     60     11.16 ms    cuStreamSynchronize          ← 90 % of wall
28.1    315.5    798      395 µs     cuMemcpyHtoDAsync_v2
 6.7     75.4      1     75.5 ms     cuCtxCreate_v2 (one-shot init)
 2.8     31.1    238      131 µs     cuModuleLoadData (one-shot per kernel)
 1.8     19.9   1241       16 µs     cuMemAlloc_v2
 0.3      3.1   1471        2.2 µs   cuMemsetD8_v2
 0.3      3.1   2583        1.2 µs   cuLaunchKernel (eager, before/after capture)
 0.2      1.98     1     1.98 ms     cuGraphInstantiateWithFlags
 0.2      1.98    60       33 µs     cuGraphLaunch                ← graph submit cost
 0.1      0.71    60       12 µs     cuMemcpyDtoH_v2 (argmax)     ← actual DtoH cost
```

## The 11 ms breakdown

Each decode token loops:

1. Build forward inputs on host (Lean) — small.
2. `cuGraphLaunch` (33 µs) — submits captured graph.
3. `cuMemcpyDtoH(argmax 4 byte)` (12 µs as a launched op) — async on stream.
4. **`cuStreamSynchronize` (11.16 ms)** — host blocks until **every kernel
   in the graph + the DtoH** has completed.
5. Read argmax host-side, advance pos, build next token's inputs.

The 11.16 ms is **how long it takes the GPU to finish all kernels of one
forward pass**.  At 60 tokens × 11.16 ms = 669.9 ms.  Plus host work
between syncs (≈ 70 ms total) = 740 ms wall.

## Why this isn't a graph launch problem

`cuGraphLaunch` itself is 33 µs/call × 60 = 2 ms.  Negligible.

Earlier hypothesis "graphs ON helps because per-launch overhead is
amortised" is right for *kernel launch* cost, but the wall is now
dominated by **GPU drain time (kernel work)**, not host launch time.

## Why this isn't a DtoH cost problem (but is a sync problem)

`cuMemcpyDtoH_v2` itself = 12 µs × 60 = 0.7 ms.  Tiny.

But the **argmax result is needed on host before the next token's
graph can launch**.  So the host issues `cuStreamSynchronize` and waits
for the GPU to drain.  During that 11 ms, host does nothing.

This is the same pattern as project_argmax_dtoh_is_28pct.md (graphs OFF
case) but transposed — the work is now hidden inside the graph but
still serialised by the host's need for the argmax result.

## Why hesper kernel time = 11 ms/tok and llama.cpp = 8.4 ms/tok

llama.cpp from kernel_compare.py: 8.4 ms/tok kernel-only at graphs OFF.
hesper measured GPU drain = 11.16 ms/tok = **2.7 ms/tok kernel-time slower**.

Where does the 2.7 ms come from?  Comparing per-kernel from the nsys
top-12 (memory: project_115tps_roadmap_2026_04_28.md):

| | hesper | llama.cpp (kernel_compare.py) | delta |
|---|--:|--:|--:|
| Q4_K matmul | ~17.8 ms / 60 tok = 0.30 ms/tok-equiv (instances 84 + 21 spread) | 4.4 ms/tok | -4.1 ms/tok ?! |
| Q6_K matmul | ~5.6 ms / 60 tok = 0.09 ms/tok-equiv | 2.2 ms/tok | -2.1 ms/tok ?! |

The math doesn't reconcile — hesper top-12 named kernels appear *faster*
than llama.cpp, yet hesper's GPU drain is 11 ms/tok vs llama.cpp 8.4
ms/tok.  Possibilities:

a) nsys graphs ON only shows one representative replay of each kernel,
   not 60 replays.  So the named-kernel sum should be **multiplied** by
   the per-token instance count.  Hesper Q4_K with 84 instances over 1
   capture × 60 replays really runs **84 × 60 = 5040 times** during
   decode, totalling 0.95 s (15.9 ms × 60) — wait, that's 953 ms, even
   more than the 670 ms drain.  Still doesn't reconcile.

b) The named-kernel total in graphs ON nsys IS the sum across all 60
   replays already (graph replay does emit per-kernel events).  Then
   sum = 30 ms vs drain 670 ms.  670 - 30 = 640 ms of GPU time is in
   *other* kernels not in the top-12 (long tail of small kernels) or
   in inter-kernel gap inside the graph.

c) Inter-kernel gap inside the graph itself — graph replay schedules
   kernels back-to-back, but each one's launch on the SM takes ~1-2 µs,
   and with ~150 kernels per token at 2 µs gap = 0.3 ms/tok of "GPU
   bubble" that's not in any kernel's reported duration.

(b) is the most likely: long tail of tiny kernels each contributing
microseconds.  llama.cpp shows the same in its 502 ms total — most of
the time is in many small kernels.

## 11 ms/tok breakdown (CUPTI_ACTIVITY_KIND_GRAPH_TRACE)

`graph_replay duration table` shows 60 events × **11.161 ms each**.  Each
replay is a pure GPU run (host is in `cuStreamSynchronize`).

To get the per-kernel breakdown, query the *graph capture* phase
(timestamps 7348.5 - 7360.0 ms in the rep), where each kernel ran
exactly once and is recorded by CUPTI.  A capture replay = same kernels
in the same order, so capture-time totals = replay-time totals.

```
1 decode token (11.5 ms window in capture):
─────────────────────────────────────────────────────────────
338 kernel launches
  kernel busy : 9.446 ms (82%)
  GPU idle    : 2.054 ms (18% — inter-kernel gap inside graph)
─────────────────────────────────────────────────────────────
top per-kernel (µs/tok, sorted by total time):
  k6234881094363609 Q4_K gate+up 4-row dp4a   18×189 =  3410
  k1636655464751297 lm_head f16 matmul         1×2743=  2743
  k7345968540891897 Q6_K ffn_down 1-row       35× 30 =  1058
  k1114518596643478 Q4_K wO/postLinear         3×192 =   576
  k1093310491457399 Q6_K (PLE) 2048-out        7× 41 =   284
  k1489805646318020 Q6_K (?)                   7× 41 =   284
  k8624103759895339 small reduce              29×5.5 =   160
  k1544409311798721                            2×78  =   157
  k9862780927254350                            2×78  =   156
  k1043146023426832                           10×13  =   130
  ...300+ tiny kernels..                     ≈300×3 =   500
```

## Component diff vs llama.cpp

llama.cpp ground truth (from kernel_compare.py output):

| component | hesper ms/tok | llama.cpp ms/tok | hesper-llama |
|---|--:|--:|--:|
| Q4_K matmul (gate+up + wO + post)        | 3.99 | 4.40 | **-0.41** |
| Q6_K matmul (ffn_down + lm_head + PLE)   | 4.36 | 2.15 | **+2.21** ← 主犯 |
| RMSNorm + Q8_1 quantize fused            | 0.13 | 1.03 | -0.90 |
| FlashAttn (V11)                          | 0.06 | 0.43 | -0.37 |
| RoPE / KV write / softmax / GELU / pad   | 0.40 | 0.30 | +0.10 |
| 〈smaller kernels long tail (400+)〉     | 0.51 | 0.10 | +0.41 |
| GPU idle inter-kernel                    | 2.05 | ~0   | +2.05 |
| **TOTAL kernel + idle**                  | 11.5 | 8.4  | **+3.1** |

(hesper の RMSNorm / FlashAttn が llama より速いのは、fused-Q8_1 RMSNorm と
 V11 split-K の最適化が効いているから。)

## Where the 3.1 ms gap comes from — actionable

1. **Q6_K lm_head 2.74 ms** — single biggest kernel.  llama.cpp's全
   Q6_K total が 2.15 ms.  lm_head だけでそれを超えている。
   - DRAM-bound (project_lmhead_dram_peak_dead_lever.md) で **per-call
     time は更には縮まない**.
   - 唯一の lever は **lm_head を skip / 削減**:
     - **早期 EOS 検出** で argmax のみ計算 → vocab/N サイズ小さく
       (Top-K matmul など)
     - **speculative decoding**: 軽 model で N tokens まとめ
       検証時のみ lm_head 1回
   - これは architectural change なので保留, 残り 0.4 + 2.0 = 2.4 ms 取り組む
2. **GPU idle 2.05 ms inter-kernel** — 338 kernels × 6 µs gap
   - graph 内でも CUDA scheduler は kernel boundary で数 µs idle 入れる
     (synchronization, register write-back).
   - 改善: **kernel fusion** で 338 → 200 dispatches → idle 半減 (-1 ms)
   - 候補: ffnNorm + Q8_1 quantize は既に fused だが、その後 Q4_K matmul
     が separate. Circuit DSL で **RMSNorm+Q8_1+Q4_K matmul** 一体化可能か?
3. **400+ small kernels の long tail (≈ 0.5 ms)** — 1-3 µs/kernel が
   400 個 = 0.5 ms.  llama.cpp は同等の small kernels を **k_bin_bcast**
   などで集約 (kernel_compare.py で binary bcast = 0.105 ms).
   - hesper の `glMlBtch_*` 系 (10個 × 1.1 µs = 11 µs/tok) や
     `plGlGtMlSlcBtch_*` (10個 × 1.1 µs) が llama では 1 kernel に
     集約されている.

## Concrete path to 115 TPS

### Option 1: device-side argmax + skip cuStreamSynchronize

If argmax is computed on device and feedback lookup is also device-side
(token table indexed by device argmax), then the host doesn't need the
result for the next iteration.  Pipeline:

```
token i:    GPU [ k1, k2, ..., kN, argmax → tokenBuf ]
                                                   ↓
token i+1:  GPU [ k1', k2', ... ]      (already queued, no sync needed)
host:       (idle, just runs cuGraphLaunch in tight loop)
```

Memory: project_pipelined_decode_attempt.md tried this for graphs OFF
(HESPER_PIPELINED_DECODE=1) and saw a *regression* (43 → 41 TPS).
Reasons cited: (1) the test was flawed (maxTokens > EOS inflated denom),
(2) graphs OFF scheduling allowed host work in parallel anyway.

For graphs ON, the situation is different: host *can't* do useful work
between graph launches because the next graph's input is the argmax.
Eliminating the sync would let the host immediately queue the next
graph launch.  Expected gain: **closes the 11 ms drain → ~0 ms** if
fully pipelined, but realistically ~50% overlap = saves 5 ms/tok →
+15 TPS = **~96 TPS**.

Risk: argmax-as-device-feedback is already implemented for graphs OFF
(HESPER_PIPELINED_DECODE / HESPER_DEVICE_FED).  Need to verify it's
wired into the graphs ON capture too.

### Option 2: longer captured graph (multi-token)

Capture N consecutive tokens into one graph, replay once.  Saves N-1
of the 11-ms syncs.  For N=10: 60 / 10 = 6 syncs = 67 ms vs current
670 ms → **+15-20 TPS ⇒ ~100 TPS**.

Tricky because each token's input depends on previous token's argmax
output.  Need device-side feedback (Option 1 prerequisite) plus
parametric graph (token slot, kv cache offset).

### Option 3: NVIDIA cudaGraphCondNode (CUDA 12.4+)

Conditional graph nodes can implement the "EOS check + token feedback"
loop entirely on device.  One graph instantiation, runs until EOS.
Most surgical but also most exotic.

### Realistic order

1. **#252 graphs OFF default → ON**: already +20 TPS.  Already covered
   in this commit.
2. **Verify HESPER_DEVICE_FED is active in graphs ON**: free if it is,
   moderate work otherwise.
3. **Single-graph multi-token replay (Option 2)** for 10 token at a
   time: +15 TPS → 100 TPS.
4. **For final 15 TPS to 115**: shave per-kernel time of the long tail
   (small reduces, RMSNorm internals).

## Diagnostic for next session

Re-run hesper graphs ON with HESPER_DEVICE_FED=1 (memory: project_device_fed_step_a.md)
and compare:

```
nsys stats --report cuda_api_sum hesper_device_fed.nsys-rep
```

Expect: `cuStreamSynchronize` calls drop from 60 → 1 (one final sync at end of run).

If it drops, TPS should jump from 80.9 → 100+.

If it doesn't drop, then HESPER_DEVICE_FED isn't being honored under
graphs ON — need to extend the graph capture to include the device-side
argmax feedback loop.
