# SASS-Level Comparison: Q6_K dp4a — hesper vs llama.cpp

Instruction-level comparison of the Q6_K × Q8_1 dp4a matmul kernel
between hesper's generated PTX (compiled to SASS) and llama.cpp's
`mul_mat_vec_q<GGML_TYPE_Q6_K, ncols_dst=1, has_fusion=0, use_stream_k=0>`.
Target: sm_89 (RTX 4070 Ti). Gemma 4 E4B lm_head shape (hiddenSize=2560,
vocabSize=262144).

## How to reproduce

```bash
# hesper → PTX → SASS
HESPER_DP4A=1 HESPER_PTX_DUMP=/tmp/hesper_ptx ./.lake/build/bin/gemma4-cuda \
    data/gemma-4-e4b-it-Q4_K_M.gguf "Hi" 2
ptxas -arch=sm_89 /tmp/hesper_ptx/k_1498334141806601.ptx -o /tmp/hesper_q6k.cubin
cuobjdump --dump-sass /tmp/hesper_q6k.cubin > /tmp/hesper_q6k_sass.txt

# llama.cpp → SASS (single-variant slice)
cuobjdump --dump-sass llama.cpp/build/bin/libggml-cuda.so.0 \
    | awk '/mul_mat_vec_qIL9ggml_type14ELi1ELb0ELb0E/{found=1} found' \
    | awk '/Function :/{n++; if(n==2)exit} n==1' \
    > /tmp/llama_q6k_sass_single.txt
```

Both files should decode the same mathematical operation: one output row
of `logits = Q6_K_weights · Q8_1_input`.

## Wall-clock baseline (nsys)

| Kernel | Time per call | Calls per token |
|---|---|---|
| hesper `fusedQ6KLinearDP4AKernel` | **1567 µs** | 2 (col-split) |
| llama.cpp `mul_mat_vec_q<Q6_K,…>` | **46 µs (median)** | ~1 |

Ratio: **34× slower** on hesper (after the rowByteBase hoist / read4Bytes
u32 merge / readOnly patches that brought us from 1809 µs).

## Instruction counts (single-variant SASS)

| Op category | hesper | llama.cpp | Δ | Notes |
|---|---:|---:|---:|---|
| **IDP.4A.S8.S8** | 2 | 2 | 0 | The actual dp4a work — **identical** |
| LDG (global) | 16 | 12 | +4 | hesper reads 4 more global operands |
| **LDS (shared)** | **0** | **3** | **−3** | llama.cpp reads input from smem |
| STS (shared store) | 0 | 1 | −1 | llama.cpp writes once to smem |
| BAR.SYNC | 0 | 1 | −1 | llama.cpp has one barrier |
| SHFL | 5 | 5 | 0 | Warp-reduction identical |
| IMAD | 71 | 56 | +15 | +27% address computation |
| IADD3 | 22 | 14 | +8 | +57% three-way adds |
| **LOP3.LUT** | **52** | **17** | **+35** | **3× bit manipulation** |
| **FFMA** | **24** | **3** | **+21** | **8× f32 fused multiply-add** |
| PRMT | 3 | 8 | −5 | llama.cpp uses byte-permute more |
| ISETP | 17 | 7 | +10 | 2.4× predicate generation |
| SEL | 4 | 0 | +4 | hesper's `select` in f32 scale-sign-extend |
| SASS lines | 695 | 389 | +306 | hesper is **1.8× larger** |

Both kernels do the same 2 dp4a operations. hesper's excess comes from
per-instruction inefficiency, not extra work.

## The four real gaps

### 1. No shared-memory input reuse (LDS 0 vs 3, STS 0 vs 1, BAR 0 vs 1)

llama.cpp issues:
```sass
STS [R16.X4+-0x80], R25        // store 1 input word to smem
BAR.SYNC.DEFER_BLOCKING 0x0    // sync warps within WG
LDS R0,  [R15.X4]              // read input from smem
LDS R5,  [R15.X4+0x80]
LDS R7,  [R15.X4+0x100]
```

This is the `nwarps=4` cooperative-load pattern: four warps per workgroup
stage the Q8_1 input into shared memory, then every warp reads from smem
during the dp4a loop. The smem traffic amortises over 4 warps' worth of
work.

hesper's kernel is `workgroupSize=32` (1 warp per WG, one row per WG), so
smem re-use doesn't apply — every dp4a load goes straight to the constant
read-only cache via `ld.global.nc`. That's fine for a batch=1 decode with
no warp-to-warp sharing, but it caps the theoretical bandwidth at
`hiddenSize × outDim × 1 B/row × nWG` and leaves 3/4 of the SM's
scheduler slots idle.

**Fix: 4-warp cooperative kernel (1 WG = 128 threads, each warp covers
1/4 of the K dimension, smem holds the input).** Expected gain: 2–4×
on this kernel, ≥ +5 TPS at the system level.

### 2. 3× LOP3 (52 vs 17) — `sub32PerByte` bloat

hesper uses the bit-trick `((x | 0x80) - 0x20) ^ 0x80` to subtract 0x20
from each packed byte without cross-byte borrow. This is 3 `LOP3.LUT`
ops per byte-pack (OR, XOR, XOR). llama.cpp replaces this with
`__vsubss4` which compiles to a single `IADD3` (saturating byte sub) on
Ampere+:

```c
// llama.cpp vecdotq.cuh:513
const int vi = __vsubss4(vil | vih, 0x20202020);
```

**Fix: add `Exp.vsubss4` (per-byte saturated sub) intrinsic to the
CUDA code-path.** On WGSL this can fall back to the LOP3 sequence.
Expected: −35 LOP3 → −15 LOP3 ≈ 15% fewer ALU ops.

### 3. 8× FFMA (24 vs 3) — f32 dequant in hot loop

hesper dequantises scales to f32 every sub-block:
```
sc0 = select(sc0Byte ≥ 128, f32(sc0Byte) - 256, f32(sc0Byte))
// → 1 I2FP + 1 FSUB + 1 SEL + 2 FFMA (scale_times_dotprod)
```

llama.cpp keeps the scale as `int` (`sc = scales[4*i]`) and does a
**single** `FFMA` per sub-block when combining `d * d8 * (dp4a_result *
sc)`.  It defers float conversion until the outermost multiplication.

**Fix: keep `sc0/sc1` as signed i32 in the inner loop, do the
`d * d8 * int_dot * int_scale` chain as one f32 FFMA at the end.** This
removes ~20 FFMAs.

### 4. +27% IMAD (71 vs 56) — address computation redundancy

hesper regenerates addresses for every `readByte` call; the ShaderM DSL
has no common-subexpression elimination across expressions. A partial
mitigation landed in `perf(cuda): hoist Q6_K rowByteBase via
ShaderM.varNamed` — it's why the count is already down from 225 to 71.

**Remaining fix: bind `blockByteBase = rowByteBase + blockIdx * 210`
once per loop iteration** in the DSL. Currently it appears to be
inlined into each `read4Bytes` call. Expected: −10–15 IMAD.

## Concrete next steps (ranked by expected TPS)

| # | Fix | Effort | Expected gain | Justification |
|---|---|---|---|---|
| 1 | 4-warp cooperative kernel | 1 week | **+5–15 TPS** | Occupancy 1→4, matches llama.cpp structure |
| 2 | `Exp.vsubss4` intrinsic | 2 days | +1–2 TPS | −35 LOP3 |
| 3 | Defer f32 dequant | 2 days | +1–2 TPS | −20 FFMA |
| 4 | Block-byte-base hoist | 1 day | +0–1 TPS | −15 IMAD, mostly ptxas already optimises |

(1) is the only **architectural** win. (2)–(4) are peephole
improvements that we can expect ptxas to partly take care of anyway.

## Notes

- hesper's `ld.global.nc` emission (from `declareReadOnlyBuffer`) does
  show up as `LDG.E.CONSTANT` in the SASS — the hint is plumbed through.
  But for batch=1 decode with `nwarps=1`, there is no cross-warp reuse
  of the constant cache, so this reads as "informs the scheduler" rather
  than "cuts latency".
- The 2 `IDP.4A.S8.S8` instructions per kernel match Q6_K's
  `QR6_K = 2` sub-block iterations. This is correct — hesper is not
  missing any dp4a work, it's spending 8× more surrounding
  instructions per dp4a.

## Artifacts

- `/tmp/hesper_q6k_sass.txt` (695 lines, 1 variant)
- `/tmp/llama_q6k_sass_single.txt` (389 lines, 1 variant)
- `/tmp/hesper_ptx/k_1498334141806601.ptx` (source PTX, 1154 lines)

Last updated: 2026-04-15.
