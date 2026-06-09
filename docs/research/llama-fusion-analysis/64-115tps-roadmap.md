# Path to 115 TPS — kernel breakdown vs llama.cpp

**2026-04-28** · Gemma 4 E4B Q4_K_M / RTX 4070 Ti / "Hello world how are you" 60 tok decode.

## Current state

| | hesper graphs OFF | hesper graphs ON | llama.cpp graphs OFF |
|---|---:|---:|---:|
| Decode TPS | 58.7 | 80.9 | ~110 |
| Wall / 60 tok | 1024 ms | 740 ms | 545 ms |
| GPU kernel total / 60 tok | TBD | ≈ 700 ms | **502 ms** (8.4 ms/tok × 60) |

llama.cpp ground truth from `scripts/kernel_compare.py` (graphs OFF, 30 decodes, 502/30 = 16.7 ms/tok with prefill mixed in; pure decode = 8.4 ms/tok):

```
category                            hs ms      lc ms    ratio     lc inst
Q4_K matmul                         ?          4.405    ?         268      → 4-5×Q4K (wQKV+wO+gate+up+down each layer ×42)
Q6_K matmul                         ?          2.151    ?          33      → ffn_down + lm_head
RMSNorm                             ?          0.733    ?         302      → ~5/layer × 42 + finalNorm
FlashAttn (mul_mat_vec_f)           ?          0.426    ?          85      → 2/layer × 42
quantize_q8_1                       ?          0.297    ?         301      → 1/projection (RMSNorm fused?)
binary bcast (add/mul)              ?          0.105    ?          86
RoPE                                ?          0.070    ?          66
KV cache write                      ?          0.060    ?          48
softmax                             ?          0.058    ?          42
GELU/unary                          ?          0.036    ?          42
pad/scale/get_rows/softcap          ?          0.034    ?          28
TOTAL (kernel only)                 ?          8.375                       → llama 119 TPS theoretical
```

hesper kernel side measured at task #300 / nsys (graphs ON, 60 tok):

| hesper kernel (key) | grid | block | inst | total ms | avg µs | role guess |
|--|--|--|--:|--:|--:|--|
| `k6234881094363609` | (10240, 5, 1) | 32 | 84 | **15.9** | 189 | Q4_K gate+up 4-row dp4a (ffn) |
| `k1114518596643478` | (2560, 5, 1)  | 32 | 21 | **4.04** | 192 | Q4_K wO/postLinear |
| `k1636655464751297` | (262144, 1, 1)| 32 | 1  | **2.74** | 2743| **lm_head f16 matmul** |
| `k7345968540891897` | (2560, 1, 1)  | 32 | 105 | 3.18 | 30  | Q6_K ffn_down 1-row |
| `k1489805646318020` | (2560, 5, 1)  | 32 | 35 | 1.43 | 40  | Q6_K |
| `k1093310491457399` | (2048, 5, 1)  | 32 | 24 | 0.97 | 40  | Q6_K (PLE perLayer) |
| `k8624103759895339` | (5, 1, 1)     | 256 | 126 | 0.69 | 5.5 | small reduce |
| `k5827556345714019` | (10240, 1, 1) | 32 | 5 | 0.59 | 118 | Q4_K (single-instance variant?) |
| `k1544409311798721` | ? | ? | 7 | 0.55 | 79 | TBD |
| `k9862780927254350` | ? | ? | 7 | 0.54 | 78 | TBD |
| ffnNrmQ8_1 / attnNrmQ8_1 | (1,1,1) | 256? | 73 | 0.52 | 7.1 | RMSNorm + Q8_1 fused |
| flashAttn V11 | ? | ? | 35 | 0.21 | 6.1 | FA decode |

hesper full kernel total ≈ **31 ms** (rough sum of top), much less than wall.  Where does the gap go?

## Gap analysis: kernel ≠ wall

graphs ON wall = 740 ms, kernel = 31 ms (ish for the top dozens) → roughly:

- **GPU idle / inter-kernel gaps**: The remaining ≈ 700 ms of wall is not in
  the kernels themselves but in **CUDA Graph replay overhead + GPU drain
  between graph launches + argmax DtoH sync**.
- llama.cpp graphs OFF wall (545 ms) ≈ kernel (502 ms) + 43 ms host =
  **kernel-bound**.
- hesper graphs ON wall (740 ms) ≫ kernel (~30 ms × 60 = 1.8 s if all kernels
  ran 60×, but the numbers in the table are *full-run totals* so per-token
  is 31/60 = 0.5 ms/tok!).  This means hesper is *enormously* idle — wall =
  740 ms but kernel = 31 ms = 4 % busy.

WAIT.  The nsys table shows total over the entire 60-tok run.
- hesper graphs ON: full-run total kernels in top-12 ≈ 31 ms.
- 60 tok wall = 740 ms.
- **GPU busy fraction ≈ 31 / 740 = 4 %**?

That can't be right.  Either nsys is missing kernels (likely — graphs ON
captures one representative instance per kernel and the rest run inside
graph replay which doesn't always emit per-kernel events), or the
remaining time really is GPU-idle.

The earlier graphs ON measurement (project_d_tps_measurement_2026_04_27.md)
showed similar: kernels visible to nsys ≪ wall.  This is a known nsys
artefact for CUDA Graph replay paths.

## So what's *really* slow?

Two scenarios, ordered by likelihood:

### Scenario A: graph replay launch latency (most likely)

Each cuda graph re-instantiation per token has a per-launch cost.  hesper
captures one graph per token and replays it 60 times.  If the replay
overhead is ~5 ms/tok, 60 × 5 = 300 ms of the wall isn't in any kernel.

Test: read `cuda_api_sum` for graphs ON to see `cuGraphLaunch` time.
Expected: 60 × ~5 ms = 300 ms in `cuGraphLaunch`.

### Scenario B: argmax DtoH sync per token

Token N's argmax result must arrive on host before token N+1 can be
constructed.  If this is `cuMemcpyDtoH` (4 byte) → drains the GPU pipeline
each token = ~5 ms/tok × 60 = 300 ms.  Memory project_argmax_dtoh_is_28pct.md
saw 28% of wall in `cuMemcpyDtoH` for graphs OFF before the
device-side-argmax landed.  Worth re-checking for graphs ON.

## Path to 115 TPS

### Where llama.cpp's 110 TPS comes from

- 8.4 ms/tok kernel time at graphs OFF (no graph overhead, host bubble small)
- ≈ 500 ms / 60 tok = 110 TPS.

### What hesper needs

To reach 115 TPS = 522 ms / 60 tok ≈ 8.7 ms/tok wall:

- **Kernel time can't grow much**.  Already at task #300 / IR fix landed.
  Sum of named kernels in nsys ≈ 31 ms (graphs ON), but real cumulative GPU
  busy time over 60 token decode = ~700 ms (mostly inside graph replay).
  Kernel-budget for 115 TPS = ~500 ms.  We have ~200 ms room.
- **The gap must come from removing graph-replay overhead or DtoH sync**.

### Concrete top-5 levers (re-prioritised)

| # | Lever | est gain | source |
|---|---|---|---|
| **1** | Match llama.cpp's **graph-less GPU kernel saturation** (= eliminate idle bubbles) | +15-20 TPS | Re-run hesper with on-device argmax + async pipeline (PIPELINED_DECODE landed but TPS regressed; needs re-investigation post-f16-lm_head). |
| **2** | **lm_head 2.74 ms/tok** vs llama Q6_K 2.15 ms/tok = -0.6 ms/tok | +3 TPS | f16 weight is 1.34 GiB vs Q6_K 0.83 GiB; switching back to **on-the-fly Q6_K dp4a + smaller weight** could save 0.6 ms/tok if compute fits. |
| 3 | Q4_K matmul kernel time: hesper top-2 (15.9 + 4.04 = 20 ms) vs llama 4.4 ms × 60 = 264 ms | hesper already faster | actually hesper *kernel-time* per matmul looks already 1.5× *better* than llama's per-token total. |
| 4 | Reduce 60 token-loop graph replays into a single multi-token graph | +5 TPS | task #162 partially landed (token-graph for fixed N); need to verify it's active. |
| 5 | RMSNorm / Q8_1 quantize fusion already comparable to llama (project_rmsnorm_warp_shuffle.md) | done | no further ROI |

### Recommended next step

1. **`cuda_api_sum` for hesper graphs ON** to identify whether
   `cuGraphLaunch` per-iter cost dominates the 700 ms gap.
2. **device-side argmax + pipelined decode regression test** —
   project_pipelined_decode_attempt.md said it didn't help with Q6_K dp4a
   lm_head but f16 lm_head changes the GPU-drain timing; might help now.
3. If graph replay isn't the root cause, **single-graph multi-token capture**
   (task #162 / forwardSingleTokenDeviceFed style) is the architecture lever.

## Memory references

- project_q6k_lmhead_f16_landed.md (this commit's win)
- project_lmhead_dram_peak_dead_lever.md (lm_head DRAM-bound, no kernel-time lever)
- project_lmhead_ir_diff.md (PTX/SASS diff)
- project_decode_top5_kernels.md (older graphs OFF top 5)
- project_argmax_dtoh_is_28pct.md (DtoH sync analysis)
- project_pipelined_decode_attempt.md (TPS regression history)

## Note on the script

`scripts/kernel_compare.py` was designed for `gemma4-llama-prefill-skeleton`
(stub kernels, decode-only structural comparison).  It outputs all-zero
hesper times because the stub exe doesn't do the real matmul.  To reuse
the classifier for *real* hesper data, the script needs:
- option to call `gemma4-cuda` instead of the stub exe.
- graphs OFF/ON env switch (currently hard-coded `HESPER_LLAMA_GRAPHS=0`).
- merge of hesper grid-based classification with the `k...` mangled names
  emitted by graphs ON path.

Pending task: extend `kernel_compare.py` so the table at the top of this
doc can be auto-regenerated.
