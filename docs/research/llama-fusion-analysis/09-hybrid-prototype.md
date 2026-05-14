---
title: "09 — Phase 0: Hybrid hesper + llama.cpp PTX prototype"
date: 2026-04-16
status: in progress
---

# Hybrid prototype: load llama.cpp's PTX into hesper

## Goal

Determine whether **115 TPS is achievable** on RTX 4070 Ti by swapping
hesper's hottest matmul kernels (`gate+up`, `wO`, `ffn_down`) for
llama.cpp's pre-compiled PTX.  This is a feasibility check before
investing 8-10 weeks in DSL extensions + per-kernel rewrites.

## Step 0.1: Baseline confirmation

| Backend | Gemma 4 E4B Q4_K_M, RTX 4070 Ti | TPS |
|---|---:|---:|
| llama.cpp CUDA | `llama-cli -p Hello -n 30 -ngl 99 -dev CUDA0 -no-cnv` | **112.6** |
| hesper (current) | `gemma4-cuda data/...gguf "Hello" 30` | 48.5 |
| Gap | | **2.32×** |

Confirms 115 TPS is the right target.

## Step 0.2: PTX extraction

Compiled `llama.cpp/ggml/src/ggml-cuda/mmvq.cu` to PTX for `sm_89`:

```bash
cd llama.cpp/build && nvcc -ptx \
  -DGGML_BACKEND_BUILD -DGGML_BACKEND_SHARED -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 \
  -DGGML_CUDA_USE_GRAPHS -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED \
  -D_GNU_SOURCE -D_XOPEN_SOURCE=600 \
  -I"../ggml/src/ggml-cuda/.." -I"../ggml/include" \
  -O3 -DNDEBUG -std=c++17 \
  --generate-code=arch=compute_89,code=compute_89 \
  -use_fast_math -extended-lambda -Xcompiler=-fPIC \
  -x cu -c .../mmvq.cu -o /tmp/llamacpp_ptx/mmvq.ptx
```

Output: 10 MB PTX, 252 entry points.

## Step 0.3: ABI for `mul_mat_vec_q`

For decode (`ncols_dst=1, has_fusion=false, small_k=false`):

- Q4_K kernel: `_Z13mul_mat_vec_qIL9ggml_type12ELi1ELb0ELb0EEv...`
- Q6_K kernel: `_Z13mul_mat_vec_qIL9ggml_type14ELi1ELb0ELb0EEv...`

Source signature (`mmvq.cu`):

```cuda
template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false>
static __global__ void mul_mat_vec_q(
    const void * vx,                              // weights  (Q4_K blocks)
    const void * vy,                              // input    (Q8_1 blocks)
    const int32_t * ids,                          // MoE routing (NULL for non-MoE)
    const ggml_cuda_mm_fusion_args_device fusion, // 32-byte struct, all-zero for no fusion
    float * dst,                                  // output (f32)
    const uint32_t ncols_x,                       // K-dim (= inDim)
    const uint3 nchannels_y,                      // (1, 0, 1) for fastdiv divisor=1
    const uint32_t stride_row_x,                  // inDim/256 * 144 (Q4_K block stride in u32)
    const uint32_t stride_col_y,                  // inDim/32 * 36 (Q8_1 block stride)
    const uint32_t stride_col_dst,                // outDim
    const uint3 channel_ratio,                    // (0, 0, 1)
    const uint32_t stride_channel_x,              // 0
    const uint32_t stride_channel_y,              // 0
    const uint32_t stride_channel_dst,            // 0
    const uint3 sample_ratio,                     // (0, 0, 1)
    const uint32_t stride_sample_x,               // 0
    const uint32_t stride_sample_y,               // 0
    const uint32_t stride_sample_dst,             // 0
    const uint32_t ids_stride);                    // 0
```

PTX layout (19 params, all sizes match):

| # | C++ type | PTX entry |
|---|---|---|
| 0 | `void *` | `.param .u64 ... param_0` |
| 1 | `void *` | `.param .u64 ... param_1` |
| 2 | `int32_t *` | `.param .u64 ... param_2` |
| 3 | `fusion_args_device` | `.param .align 8 .b8 ... param_3[32]` |
| 4 | `float *` | `.param .u64 ... param_4` |
| 5 | `uint32_t` | `.param .u32 ... param_5` |
| 6 | `uint3` | `.param .align 4 .b8 ... param_6[12]` |
| 7-9 | `uint32_t` × 3 | `.param .u32` |
| 10 | `uint3` | `.param .align 4 .b8 ... param_10[12]` |
| 11-13 | `uint32_t` × 3 | `.param .u32` |
| 14 | `uint3` | `.param .align 4 .b8 ... param_14[12]` |
| 15-18 | `uint32_t` × 4 | `.param .u32` |

### `fastdiv` (uint3) for divisor=1

`init_fastdiv_values(1) = (mp=0, L=0, divisor=1)` → `uint3 (0, 0, 1)`.

For all our use-cases (no MoE, single channel, single sample), all
fastdiv values are `(0, 0, 1)`.

### Block / grid shape

From `mmvq.cu` host launcher:
```cpp
const dim3 block_dims(warp_size, nwarps, 1);
const dim3 block_nums(rows_per_iter, nchannels_dst, nsamples_dst);
```

For Q4_K decode (`ncols_dst=1, has_fusion=false`):
- `nwarps = calc_nwarps(Q4_K, 1, table)` → likely 4 (Ada)
- `warp_size = 32`
- `rows_per_cuda_block = calc_rows_per_block(1, table, false, 4)` → likely 4

So `block_dims = (32, 4, 1)`, `block_nums = (outDim/4, 1, 1)`.

For wO (outDim=2560): grid = (640, 1, 1), block = (32, 4, 1) = 128 thread/WG.

### Q8_1 input format

llama.cpp pre-quantizes `vy` to Q8_1 blocks of 32 elements each:

```cpp
struct block_q8_1 {
    half2  ds;          // delta + sum-of-quants (each as half)
    int8_t qs[32];      // 32 quantized values
};                       // total 36 bytes per block
```

hesper currently has its own Q8_1 quantize path (`fusedNormQ8` etc.).
Need to verify the **byte layout matches** between hesper and
llama.cpp.  If it doesn't, we either:
- (a) Borrow llama.cpp's `quantize_row_q8_1` PTX as well
- (b) Adjust hesper's Q8_1 emit to match llama.cpp's layout

### Q4_K weight format

```cpp
struct block_q4_K {
    half2 dm;           // delta + min (each as half)
    uint8_t scales[12];  // packed 6-bit scales+mins for 8 sub-blocks
    uint8_t qs[128];     // 256 4-bit quantized weights
};                       // total 144 bytes per block (= 36 u32)
```

This matches hesper's Q4_K layout (verified from `Hesper/Quants.lean`).

## Step 0.4: Integration plan

### 4.1 Load PTX in hesper

Add to `Hesper/Backend/CUDA.lean`:
```lean
def loadLlamacppKernel (modulePath : String) (kernelName : String)
    : IO CUfunction := do
  let bytes ← IO.FS.readBinFile modulePath
  let mod ← cuModuleLoadData bytes
  cuModuleGetFunction mod kernelName
```

Mangled names:
- Q4_K decode: `_Z13mul_mat_vec_qIL9ggml_type12ELi1ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj`
- Q6_K decode: `_Z13mul_mat_vec_qIL9ggml_type14ELi1ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj`

### 4.2 Argument packing

19 args of mixed types.  Use `cuLaunchKernel` argument array (each
element is `void *` pointing at the corresponding parameter value):

```c
uint3 fastdiv_one = {0, 0, 1};
struct fusion_args { void *xb, *gate, *gb; uint32_t glu_op; } fusion = {0};
uint32_t ncols_x = inDim, stride_row_x = inDim/256*144/4, ..., ids_stride = 0;
void *kernel_args[19] = {
  &weight_buf, &q8_buf, &null_ids,
  &fusion, &out_buf,
  &ncols_x, &fastdiv_one, &stride_row_x, &stride_col_y, &stride_col_dst,
  &fastdiv_one, &z, &z, &z, &fastdiv_one, &z, &z, &z, &z};
cuLaunchKernel(func, outDim/4, 1, 1, 32, 4, 1, 0, stream, kernel_args, NULL);
```

### 4.3 Q8_1 quantize bridge — **ABI MISMATCH FOUND**

**Critical**: hesper's Q8_1 layout differs from llama.cpp's.

| | llama.cpp | hesper |
|---|---|---|
| Per-block size | 36 bytes (= 9 u32) | 36 bytes (= 9 u32) ✅ same |
| Header | `(d : f16, s : f16)` (4 bytes) | `(d : f32 bitcast u32)` (4 bytes) ❌ |
| Quants | `int8_t qs[32]` (32 bytes) | packed `u32 qs[8]` (32 bytes) ✅ same |
| `s = d * Σ qs[i]` | computed and stored | **not computed** |

While `vec_dot_q4_K_q8_1` only reads `d8[i] = __low2float(bq8i->ds)`
(LOW half of the half2, i.e. `d`), the **format of those low 2 bytes
differs** (f16 in llama.cpp, low 2 bytes of an f32 in hesper).

Reading hesper's f32 bytes as f16 will produce nonsense.

**Decision**: borrow llama.cpp's `quantize_row_q8_1_cuda` PTX as well.
This adds 1 dispatch per matmul (or shared per layer) but guarantees
correct ABI.

#### `quantize_q8_1` PTX ABI

Extracted symbol: `_Z13quantize_q8_1PKfPvlllllj5uint3`

9 params:

| # | C++ type | PTX entry | Decode value |
|---|---|---|---:|
| 0 | `const float *` | `.param .u64 param_0` | input x |
| 1 | `void *` | `.param .u64 param_1` | output Q8_1 |
| 2 | `int64_t ne00` | `.param .u64 param_2` | inDim |
| 3 | `int64_t s01` | `.param .u64 param_3` | 0 |
| 4 | `int64_t s02` | `.param .u64 param_4` | 0 |
| 5 | `int64_t s03` | `.param .u64 param_5` | 0 |
| 6 | `int64_t ne0` | `.param .u64 param_6` | inDim |
| 7 | `uint32_t ne1` | `.param .u32 param_7` | 1 |
| 8 | `uint3 ne2_fastdiv` | `.param .b8 param_8[12]` | (0,0,1) |

Launch: `block=(256,1,1)`, `grid=(ceil(inDim/256), 1, 1)`.

### 4.4 Replace `Linear.forwardDP4A` for Q4_K

In `Hesper/Layers/Linear.lean`, add a flag (`HESPER_USE_LLAMACPP_PTX=1`)
that swaps the matmul dispatch for the Q4_K case:

```lean
if useLLamaCppPTX && layer.quantFormat == .Q4_K then
  Hesper.Backend.CUDA.launchLLamaCppQ4KMatmul ctx layer.weightBuf q8Buf outputBuf
    layer.config.inDim layer.config.outDim
else
  -- existing hesper kernel ...
```

## Step 0.5: Validation plan

1. Single layer test: replace ONLY wO for one layer, verify output is
   bit-equivalent (within float-noise) to hesper's existing kernel.
2. All layers: enable `HESPER_USE_LLAMACPP_PTX=1`, run decode 30 tokens,
   verify output text is sensible (perplexity check).
3. Benchmark: nsys + per-kernel time → TPS.

## Step 0.6: Decision

- **TPS ≥ 100**: target reachable, proceed to Phase 1 (DSL extensions)
- **TPS < 80**: bottleneck is elsewhere, re-examine
- **TPS in 80-100**: marginal — investigate which kernel is limiting,
  decide whether to push DSL or accept limit

## Status (2026-04-16)

- [x] 0.1 Baseline confirmed (112.6 TPS llama.cpp)
- [x] 0.2 PTX extracted (`/tmp/llamacpp_ptx/{mmvq,quantize}.ptx`)
- [x] 0.3 ABI documented (mmvq + quantize_q8_1)
- [x] 0.4b PTX loader + launch helpers (`Hesper/Backend/LlamaCppPTX.lean`)
- [x] 0.4c `cuLaunchKernelRaw` FFI for mixed-type args (uint3 / fusion_args struct)
- [x] 0.4-verify: all 3 kernels JIT successfully on driver 565.77
  (note: CUDA 12.8 emits PTX 8.7 but driver only supports up to 8.6;
   workaround = post-process PTX to replace `.version 8.7` → `.version 8.6`.
   sm_89 target, dp4a instructions all present in both ISA versions.)
- [x] 0.4d wire `HESPER_USE_LLAMACPP_PTX=1` via `llamaCppDp4aOverride` Ref +
  new `GPUBackend.rawDevicePtr` method (CUDA: buf.ptr; WebGPU: none)
- [x] 0.4e end-to-end runs without crashes at **70 TPS** (baseline 48)
  BUT output is gibberish — ABI close but not exact.

### Remaining ABI debug work

Evidence:
- No kernel crashes / no illegal address accesses (CUDA doesn't error)
- TPS +45%, suggesting the kernel is actually doing dp4a work
- Output garbage in every position (not just late tokens) → matmul result is
  systematically wrong, not noise

Likely suspects, in order:
1. **Q4_K block field order**: hesper and llama.cpp both define `block_q4_K =
   {ggml_half2 dm; uint8_t scales[12]; uint8_t qs[128]}` (144 B).  But
   hesper's Q4_K was loaded from GGUF — need to verify the on-disk layout
   matches what llama.cpp's kernel expects (it loads from GGUF too, so this
   should be automatic, but verify byte-for-byte in `Hesper/Quants.lean`).
2. **Scales packing**: the 12-byte `scales[]` packs 8×(6-bit scale, 6-bit min)
   in a specific bit-interleaved format.  If hesper's dequantize interprets
   this differently, llama.cpp's kernel will read the right bytes but assume
   the wrong bit layout.  This is how the bug could produce garbage without
   crashes.
3. **Strides**: `stride_row_x = inDim/256` (verified against mmvq.cu host).
   `stride_col_y = inDim/32` — verified for single-row.  Unlikely the bug.
4. **Launch config**: `block=(32,4,1)`, `grid=(outDim,1,1)` with rpb=1 —
   matches host code for `small_k=false`.  Unlikely the bug.

Next debug step: write a standalone micro-test in `Tests/LlamaCppPTX/` that
takes a known Q4_K-quantized row, runs both hesper's forwardDP4A and the
llama.cpp PTX path, and compares outputs element-by-element for the first
dispatch.  That isolates the ABI issue from Gemma 4's full pipeline.

Profiling: with Q4_K override on and Q6_K off, TPS was 63.8 — the ~6 TPS
gap between 63.8 and 70 comes from lm_head on Q6_K being different.
Unchanged baseline gap (48 → 70) is +45%, consistent with Q4_K kernels
being the hottest component.
- [ ] 0.5 Validation
- [ ] 0.6 TPS measurement + decision
