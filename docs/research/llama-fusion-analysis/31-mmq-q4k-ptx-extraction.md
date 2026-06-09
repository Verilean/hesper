# llama.cpp mmq Q4_K PTX extraction (sm_89) — recipe + ABI

Built on top of #212 (existing mmvq PTX direct-execution infrastructure).
Parallel goal: time the *real* llama.cpp prefill matmul kernel against
hesper's MMQ5 to ground-truth the 9.2× prefill gap
(`project_prefill_9x_gap_2026_04_29.md`).

## Recipe

```bash
NVCC=/nix/store/hhaw47wywchphvsvpf1jmz17clihzi9j-cuda-merged-12.8/bin/nvcc
cd llama.cpp/build
$NVCC -ptx \
  -DGGML_BACKEND_BUILD -DGGML_BACKEND_SHARED -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 \
  -DGGML_CUDA_USE_GRAPHS -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED \
  -D_GNU_SOURCE -D_XOPEN_SOURCE=600 \
  -I"../ggml/src/ggml-cuda/.." -I"../ggml/include" \
  -O3 -DNDEBUG -std=c++17 \
  --generate-code=arch=compute_89,code=compute_89 \
  -use_fast_math -extended-lambda -Xcompiler=-fPIC \
  -x cu -c ../ggml/src/ggml-cuda/template-instances/mmq-instance-q4_k.cu \
  -o /tmp/llamacpp_ptx/mmq_q4k.ptx
```

Output: 6.4 MB PTX, 64 .entry symbols (16 mmq_x sizes × 2 need_check
flags × 2 entry kinds — the kernel and its stream-k fixup pass).

## Symbols

The Q4_K mul_mat_q kernel symbol family is:

```
_Z9mul_mat_qIL9ggml_type{TYPE}ELi{MMQ_X}ELb{NEED_CHECK}EE...
```

For Q4_K the `{TYPE}` is `12` (= `GGML_TYPE_Q4_K` enum value). `{MMQ_X}`
ranges over `8, 16, 24, ..., 128` (16 sizes, the per-block tile width).
`{NEED_CHECK}` is `0` or `1` — `0` means `nrows_x % mmq_y == 0` (the
fast path with no bounds checking inside the kernel), `1` adds bounds
checks.

Most representative for hesper's MMQ5 comparison (mmq_x=64 matches the
hesper tile shape, need_check=0 = bound-aligned fast path):

```
_Z9mul_mat_qIL9ggml_type12ELi64ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiiiiiii
```

## Kernel signature (`mmq.cuh:3538`)

```cpp
static __global__ void mul_mat_q(
    const char * __restrict__ x,            // Q4_K weight blocks (packed, super-block layout)
    const int  * __restrict__ y,            // Q8_1 input tiles (32 ints/super-block)
    const int32_t * __restrict__ ids_dst,    // MoE row indices — nullptr for regular matmul
    const int32_t * __restrict__ expert_bounds, // MoE — nullptr
    float       * __restrict__ dst,         // f32 output
    float       * __restrict__ tmp_fixup,   // stream-K only — nullptr otherwise
    const int ncols_x,           // = K (input dimension)
    const int nrows_x,           // = outDim
    const int ncols_dst,         // = seqLen (or batch)
    const int stride_row_x,      // = K / 32  (for Q4_K: 32 ints per super-block)
    const int ncols_y,           // = seqLen
    const int stride_col_dst,    // = nrows_x = outDim
    const int channel_ratio,     // = 1 (single-channel)
    const int nchannels_y,       // = 1
    const int stride_channel_x,  // = 0
    const int stride_channel_y,  // = 0
    const int stride_channel_dst,// = 0
    const int sample_ratio,      // = 1
    const int nsamples_y,        // = 1
    const int stride_sample_x,   // = 0
    const int stride_sample_y,   // = 0
    const int stride_sample_dst, // = 0
    const int ncols_max);        // = seqLen
```

For our microbench (no MoE, single channel/sample, no stream-K):
- `ids_dst = expert_bounds = tmp_fixup = nullptr`
- All channel/sample strides = 0, ratios = 1
- `nchannels_y = nsamples_y = 1`
- `ncols_max = ncols_dst = ncols_y = seqLen`

## Launch config (`mmq.cuh:3960` `launch_mul_mat_q`)

```cpp
const dim3 block_dims(warp_size, nwarps, 1);  // (32, nwarps, 1)
const int nty  = (nrows_x + mmq_y - 1) / mmq_y;
const int ntx  = (ncols_max + mmq_x - 1) / mmq_x;
const int ntzw = nchannels_y * nsamples_y;     // = 1 for us
const dim3 block_nums_xy_tiling(nty, ntx, ntzw);
```

Where on RTX 4070 Ti (sm_89, Ada/Ampere class):
- `warp_size = 32`
- `nwarps = mmq_get_nwarps_host(cc, 32)` — see `mmq.cuh` for exact value;
  for sm_89 with TURING_MMA_AVAILABLE this returns 8.
- `mmq_y = get_mmq_y_host(cc)` — returns 128 for sm ≥ 7.5 (matches hesper MMQ5).
- `mmq_x = 64` (chosen via heuristic in `mul_mat_q_case`).

So the launch becomes:
- `block_dims = (32, 8, 1)` — 256 threads/block
- `block_nums = (outDim/128, seqLen/64, 1)`

This **exactly matches hesper's MMQ5 grid shape**, confirming the WG/loop
structure was correctly ported per #321.

Required dynamic shared memory: `mmq_get_nbytes_shared<Q4_K>(64, 128, sm89, 32, 8)`.
Computed at runtime in launch_mul_mat_q. Empirically ~46-50 KB. Must call
`cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize, N)` to
unlock above 48 KB.

## Q4_K input layout (PKc = const char*)

llama.cpp Q4_K weight super-block (144 bytes / 256 elements):
```
struct block_q4_K {
    union {
        struct {
            ggml_half d;   // super-block scale  (2 bytes)
            ggml_half dmin; // super-block min   (2 bytes)
        };
        uint16_t scales_h[2];  // alt view (4 bytes)
    };
    uint8_t scales[12];        // sub-block scale/min packed (12 bytes)
    uint8_t qs[QK_K/2];        // 4-bit quants            (128 bytes)
};
```

This matches hesper's wQ.weightBuf layout used in MMQ5
(`Hesper/Layers/Linear.lean:1455` blocksPerRow = inDim/256, 36 u32/block
= 144 bytes — IDENTICAL).

## Q8_1 input layout (PKi = const int*)

Per-super-block (288 bytes = 72 ints / 256 elements):

8 sub-blocks of 36 bytes each. For each sub-block:
- `int qs[8]` — 32 int8 quants packed (32 bytes)
- `half2 ds`  — (scale, sum) packed (4 bytes)

This **matches hesper's standard Q8_1 layout** (post-#146 port).

## Comparison plan (next session)

1. **Add to `Hesper/Backend/LlamaCppPTX.lean`** a `mmqQ4KSymbol` constant
   and `loadKernels` extension that loads `/tmp/llamacpp_ptx/mmq_q4k.ptx`.
2. **New microbench `Tests/CUDA/MmqQ4KBench.lean`**: load Gemma 4 wO
   weights and a Q8_1-quantized fp32 input, then time both kernels at
   shape (outDim=2560, K=2560, seqLen=64) with stream-bracketed batches:
   - `q4kMatmulBatchMMQ5Kernel` (hesper-native dp4a)
   - `_Z9mul_mat_qIL9ggml_type12ELi64ELb0EE...` (llama.cpp from PTX)
   Output: ms/call, parity check (max |err|).
3. **If llama.cpp wins by >1.5×**: continue to int8 mma.sync Inst (#338)
   so we can transpile/match llama.cpp's compute path.
4. **If margin <1.2×**: the gap is launch overhead / dispatcher, not
   kernel — pivot to host-side investigation.

## Resolved: PTX 8.7 vs driver 565.77 — cubin loader now in place

The first extraction emits `.version 8.7` (not 8.6 like mmvq.ptx, both
built with the same nvcc 12.8). Difference: mmq.cuh has 4
`NO_DEVICE_CODE` calls that expand to `__device__ printf(...)` for
unreachable architectures. Even though the printf is dead code at
runtime, nvcc emits `.extern .func vprintf` in the PTX header, and the
vararg-extension semantics force PTX 8.7. Driver 565.77 (CUDA 12.7)
rejects this with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` even though
ptxas (12.8) accepts it offline.

**Resolution: cubin loader landed**. We added a
`cuModuleLoadDataBytes : ByteArray → IO CUmodule` FFI variant
(`Hesper/CUDA/FFI.lean` + `native/cuda_bridge.cpp`), and a cubin path
in `LlamaCppPTX.loadKernels` that prefers cubin over PTX when present.
Cubin is sm-specific binary which skips driver JIT and so dodges the
PTX-version check entirely.

**Cubin extraction** (preferred path):
```bash
NVCC=/nix/store/.../bin/nvcc
cd llama.cpp/build
$NVCC -cubin \
  -DGGML_BACKEND_BUILD -DGGML_BACKEND_SHARED -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 \
  -DGGML_CUDA_USE_GRAPHS -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED \
  -D_GNU_SOURCE -D_XOPEN_SOURCE=600 \
  -I"../ggml/src/ggml-cuda/.." -I"../ggml/include" \
  -O3 -DNDEBUG -std=c++17 \
  --generate-code=arch=compute_89,code=sm_89 \
  -use_fast_math -extended-lambda -Xcompiler=-fPIC \
  -x cu -c ../ggml/src/ggml-cuda/template-instances/mmq-instance-q4_k.cu \
  -o /tmp/llamacpp_ptx/mmq_q4k.cubin
```

Output: 2.5 MB cubin. Loader will pick it up automatically; verify with
`lake exe llamacpp-ptx-load-test` showing
`✓ mmq Q4_K (mmq_x=64) @ 0xXXXX`.

## Status (after this session, 2026-04-30)

Landed:
- ✓ Cubin extraction recipe (this doc)
- ✓ `cuModuleLoadDataBytes` FFI for binary cubin loading (commit `0ef465d`)
- ✓ `cuFuncSetMaxDynamicSmem` FFI for raising smem cap above 48 KB
- ✓ `LlamaCppPTX.launchMmqQ4K` — packs 23 args (6 ptrs + 17 ints) and
  launches with grid=(outDim/128, seqLen/64, 1), block=(32,8,1)
  (commit `578b5f2`)
- ✓ `Tests/LlamaCppPTX/MmqLaunchTest.lean` — confirms ABI is correct:
  loads cubin, raises smem to 96 KB, launches, syncs, reads zero
  output for zero inputs

Pending for #343 (perf microbench):
- llama.cpp's mmq kernel reads Q8_1 in the **`block_q8_1_mmq` layout**:
  144 B / 128 elements = 4 sub-blocks of 32 i8 quants packed + 4 half2
  (d, s) scale-sum pairs at the tail (per-pair-of-sub-blocks, with 16 B
  padding to avoid smem bank conflicts).
  This is **different from hesper's standard Q8_1 layout** (post-#146
  port, 36 B / 32 elements). Two options to match:
  1. Use llama.cpp's `quantize_q8_1_mmq` kernel (also in mmq.cuh /
     quantize.cu) which we'd extract similarly to mmq Q4_K.
  2. Write a Lean-side reformat from standard Q8_1 to mmq layout.
- Once Q8_1 inputs are in the right format, the microbench is:
  hesper `q4kMatmulBatchMMQ5Kernel` vs `launchMmqQ4K`, same shape,
  cuStreamSync-bracketed batches, ms/call comparison + parity check.

## Why this is worth doing

The 9.2× prefill gap is the single largest perf delta vs llama.cpp
(`project_prefill_9x_gap_2026_04_29.md`). Without an apples-to-apples
kernel-time number we've been guessing whether it's:
- (a) llama.cpp's mma.sync int8 Tensor Core throughput, or
- (b) hesper's launch dispatcher overhead, or
- (c) WG-shape / loop-structure differences.

Past audits (#321) confirmed hesper's MMQ5 matches llama.cpp's
WG=(32,8,1), grid=(outDim/128, seqLen/64, 1), mmq_x=64, mmq_y=128.
That eliminates (c). The kernel-time microbench separates (a) from (b).
