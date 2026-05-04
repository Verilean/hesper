# Video / vision tower implementation plan: im2col + conv2d

Date: 2026-05-04

After cp.async / launch_bounds exhausted Q4_K-shape MMQ optimization
levers (compute-bound, see `project_mmq5_warm_bottleneck.md`), shifted
focus to multi-modal (video / audio).

Stub-extractor output for video in `docs/video-stubs/overview.md`:
10 kernels in `llama.cpp/ggml/src/ggml-cuda/{im2col,conv2d,conv2d-dw,conv2d-transpose,upscale,pad}.cu`.
All have `smem=0`, `sync=0`, `loops=0` — i.e. **simple per-thread kernels**.

## Why video first (vs audio)

| | audio (SSM/Mamba/RWKV) | video (CLIP-ViT, conv2d) |
|---|---|---|
| algorithm family | recursive state-space | im2col + matmul |
| reuse of existing hesper | low (new family) | **high** (matmul = MMQ5/MMVQ) |
| smem use | heavy (3 decl, 4 sync) | **none** |
| llama.cpp implementation size | medium-large | **small** (~50 LoC each) |
| target model | Mamba-2.8B etc. | LLaVA / SigLIP / CLIP |

Video tower for LLaVA-class models is the most concrete entry point.
**im2col is the gateway kernel** — once it works, conv2d reduces to
matmul (which hesper already has via Linear).

## Per-kernel triage

| kernel | role | priority | impl plan |
|---|---|---|---|
| **im2col_kernel** | conv → matmul shim | P0 | NEW — port |
| im2col_3d_kernel | same, 3D variant | P2 | port if needed |
| conv2d_kernel | direct conv (no im2col) | P3 | likely skipped — im2col path works |
| conv2d_dw_kernel | depthwise conv | P3 | only for MobileNet-style (not used by SigLIP/CLIP) |
| conv2d_transpose_kernel | upsample conv | P3 | not on hot path for vision encoders |
| upscale_f32 / _bilinear / _antialias / _bicubic | image resize | P1 | bilinear is small + needed |
| pad_f32 | pad image | P1 | needed (small) |

P0 = im2col is the foundation for everything else.

## im2col algorithm (from llama.cpp/ggml-cuda/im2col.cu)

```
template <typename T>
__global__ void im2col_kernel(
    const float * x,      // input [N, IC, IH, IW]
    T * dst,              // output [N, OH, OW, IC*KH*KW]
    int64_t IC, IW, IH, OH, OW, KW, KH,
    int64_t IC_IH_IW, IH_IW, N_OH, KH_KW, IC_KH_KW,
    int s0, s1,           // stride
    int p0, p1,           // padding
    int d0, d1            // dilation
) {
    const int64_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= IC_KH_KW) return;

    const int64_t iic = i / KH_KW;
    const int64_t rem = i - iic * KH_KW;
    const int64_t ikh = rem / KW;
    const int64_t ikw = rem - ikh * KW;

    const int64_t iow = blockIdx.y;
    for (int64_t iz = blockIdx.z; iz < N_OH; iz += MAX_GRIDDIM_Z) {
        const int64_t in = iz / OH;
        const int64_t ioh = iz - in * OH;

        const int64_t iiw = iow * s0 + ikw * d0 - p0;
        const int64_t iih = ioh * s1 + ikh * d1 - p1;

        const int64_t offset_dst =
            ((in * OH + ioh) * OW + iow) * IC_KH_KW + iic * KH_KW + ikh * KW + ikw;

        if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
            dst[offset_dst] = 0;
        } else {
            const int64_t offset_src = iic * IC_IH_IW + in * IH_IW;
            dst[offset_dst] = x[offset_src + iih * IW + iiw];
        }
    }
}
```

Grid:
- `block.x = MIN(IC*KH*KW, 256)` (CUDA_IM2COL_BLOCK_SIZE = 256)
- `grid.x = ceil(IC*KH*KW / 256)`
- `grid.y = OW`
- `grid.z = MIN(N*OH, 65535)`

i.e. each thread emits one output element to `dst[(N, OH, OW, KH*KW*IC)]`.
The `for iz` loop is essentially 1 iteration (the `MAX_GRIDDIM_Z=65535`
is for safety on huge batches; vision tower seqlen × OH typically <<
65535 — e.g. SigLIP-base patches 224x224 / 16 patch_size = 14×14 = 196).

## Hesper port plan

### Phase A: f32 → f32 im2col, single-shot bound check

```lean
-- src: input image (f32) [N, IC, IH, IW] flat
-- dst: output (f32) [N, OH, OW, IC*KH*KW] flat
-- All shape params come in via uniform (or kernel args)
def im2colF32Kernel
    (IC IW IH OH OW KW KH : Nat)
    (s0 s1 p0 p1 d0 d1 : Int)  -- stride/pad/dilation as i32 since
                                 -- pad subtracts and goes negative
    : ShaderM Unit := do
  let lid ← ShaderM.localId
  let wid ← ShaderM.workgroupId
  let _src ← ShaderM.declareReadOnlyBuffer "src" (.array (.scalar .f32) (...))
  let _dst ← ShaderM.declareOutputBuffer "dst"  (.array (.scalar .f32) (...))

  let tid := Exp.vec3X lid             -- thread within block
  let bx  := Exp.vec3X wid             -- IC*KH*KW chunk
  let iow := Exp.vec3Y wid             -- output col
  let iz  := Exp.vec3Z wid             -- N * ioh (batch * output row)

  let i := tid + bx * 256
  let inBoundsI := i <ᵉ Exp.litU32 (IC * KH * KW)
  if_ inBoundsI (do
    let iic := i / (KH * KW)
    ...
    -- compute iiw, iih, do bound check, write to dst
  )
```

**Bound check**: `iiw ≥ 0` and `iih ≥ 0` need signed comparisons since
`pad` subtracts. hesper has signed comparison via Exp.lt with .scalar .i32.
Use Exp.toI32 + signed cmp + Exp.if_ (or Exp.select).

**Skipped initially**: `for iz step MAX_GRIDDIM_Z` outer loop — assume
`N * OH < 65535`. Add later if needed (most LLM vision towers don't hit).

### Phase B: f16 dst variant

`Exp.pack2x16float` would be needed to write f16. If hesper doesn't
have it, add as new Exp constructor lowering to PTX `cvt.rn.f16.f32`
+ pack into u32 via existing `Exp.bitOr` of two `cvt.rn.f16.f32` values.
(Inverse of existing `Exp.unpack2x16float`.)

But for first parity test, **f32 → f32 is enough**.

### Phase C: parity test

Tests/CUDA/CUDAIm2colTest.lean:
- Generate small synthetic image (e.g. N=1, IC=3, IH=8, IW=8)
- Generate kernel params (KH=3, KW=3, stride=1, pad=1, dil=1)
- Compute reference im2col on CPU (~30 lines of Lean)
- Run hesper kernel; compare element-wise.

### Phase D: wire conv2d via im2col + Linear

`conv2d(src, w) ≡ Linear(im2col(src), reshape(w))`
- im2col: [N,IC,IH,IW] → [N*OH*OW, IC*KH*KW]
- reshape weights: [OC, IC, KH, KW] → [OC, IC*KH*KW]
- matmul: [N*OH*OW, IC*KH*KW] × [IC*KH*KW, OC] → [N*OH*OW, OC]
- reshape output: [N*OH*OW, OC] → [N, OH, OW, OC] (or NCHW)

This uses hesper's existing `Linear.forward` for the matmul, just need
shape params.

## What's NOT in scope this round

- Q4_K-quantized conv weights (vision tower weights are usually f16/f32,
  not Q4_K, so skip).
- conv2d_dw (depthwise) — not used by SigLIP/CLIP/ViT.
- 3D conv (im2col_3d_kernel) — not used by 2D vision towers.
- conv2d_transpose — used in some upsample but not vision encoders.
- upscale + pad — needed for image preprocessing, but those are tiny
  (<1% of total time); can do them inline in CPU or hesper later.

## Estimated effort

| phase | LoC | sessions |
|---|---:|---:|
| A: f32 im2col + parity test | 200 | 1 |
| B: f16 variant (if needed) | 50 | 0.5 |
| C: f32 parity passing | (covered in A) | — |
| D: conv2d via Linear | 100 | 0.5 |
| Pre-conditions: Exp.toI32 if not present | check | low |

Total: 1-2 sessions for working f32 conv2d via im2col + Linear. End
result: hesper can run any 2D conv-only frontend (simple CNNs, basic
ViTs without their full attention block).

## Out of scope (next sessions)

- Full ViT (needs attention, MLP, layer norm — these already exist in
  hesper for LLM use, just need ViT-shape wiring)
- LLaVA tower (CLIP-ViT-L/14 + projector)
- SigLIP (similar but different normalization)
