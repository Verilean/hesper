import Hesper.WGSL.Monad
import Hesper.WGSL.Exp
import Hesper.Backend

/-!
# Vision tower kernels (im2col, conv2d helpers)

Foundation for running 2D-conv-based vision encoders (CLIP-ViT, SigLIP,
generic ViTs with patch embedding) on hesper. Mirrors llama.cpp's
`ggml/src/ggml-cuda/im2col.cu` algorithm.

## im2col

Input:  `src : f32 [N, IC, IH, IW]`
Output: `dst : f32 [N, OH, OW, IC*KH*KW]`

For each output position `(in, ioh, iow, iic*KH*KW + ikh*KW + ikw)`:
- Compute input position `(iiw, iih)` from output via stride/dilation/pad.
- If `(iiw, iih)` is in `[0, IW) × [0, IH)`, copy the corresponding
  input pixel; otherwise write zero (zero-pad).

The output layout `[N, OH, OW, IC*KH*KW]` lets the subsequent matmul
do `[N*OH*OW, IC*KH*KW] × [IC*KH*KW, OC] → [N*OH*OW, OC]` which is
exactly conv2d.

## Bound-check trick

`iiw = iow * s0 + ikw * d0 - p0` can go **negative** when padding
extends past the image boundary. PTX `setp` only supports unsigned
in hesper's CodeGen (no `.s32` variant). Workaround: compute as u32,
which wraps negative values to ≥ 2^31. Since `IW < 2^31` for any
realistic image, the unsigned compare `iiw_u < IW` correctly rejects
both `iiw < 0` (wraps to ≥ 2^31) and `iiw ≥ IW`.
-/

namespace Hesper.Layers.Vision

open Hesper
open Hesper.WGSL.Monad (ShaderM)
open Hesper.WGSL (Exp)

/-- im2col kernel for f32 input, f32 output.

  Static parameters (compile-time):
  - `IC` : input channels
  - `IH IW` : input height/width
  - `OH OW` : output height/width
  - `KH KW` : kernel height/width
  - `s0 s1` : stride (col, row)
  - `p0 p1` : padding (col, row)
  - `d0 d1` : dilation (col, row)
  - `N` : batch size

  Dispatch grid (caller responsibility):
  - `block.x = min(IC*KH*KW, 256)`
  - `grid.x = ceil(IC*KH*KW / 256)`
  - `grid.y = OW`
  - `grid.z = N * OH`  (assumes < 65535; otherwise wrap is needed,
    but vision encoders rarely hit this — e.g. SigLIP-224/16 has
    N*OH = 1*14 = 14)

  Each thread emits exactly one f32 output element. -/
def im2colF32Kernel
    (IC IW IH OW OH KW KH : Nat)
    (s0 s1 p0 p1 d0 d1 : Nat)  -- stride/pad/dilation; padding subtracted via wrap
    (N : Nat)
    : ShaderM Unit := do
  let IC_KH_KW := IC * KH * KW
  let KH_KW := KH * KW
  -- llama.cpp naming: src has shape [N, IC, IH, IW], row-major.
  -- - srcChannelStride = IH*IW  (stride between adjacent IC slots)
  --   → llama.cpp calls this `IC_IH_IW` (legacy name from `src1->nb[2]/4`).
  -- - srcBatchStride = IC*IH*IW (stride between adjacent N slots)
  --   → llama.cpp calls this `IH_IW` (legacy name from `src1->nb[3]/4`).
  let srcChannelStride := IH * IW            -- llama: IC_IH_IW
  let srcBatchStride := IC * srcChannelStride -- llama: IH_IW
  let srcSize := N * srcBatchStride
  let dstSize := N * OH * OW * IC_KH_KW

  let _src ← ShaderM.declareReadOnlyBuffer "src" (.array (.scalar .f32) srcSize)
  let _dst ← ShaderM.declareOutputBuffer  "dst" (.array (.scalar .f32) dstSize)

  let lid ← ShaderM.localId
  let wid ← ShaderM.workgroupId
  let tid := Exp.vec3X lid              -- 0..255
  let bx  := Exp.vec3X wid              -- IC*KH*KW chunk index
  let iow := Exp.vec3Y wid              -- output column
  let iz  := Exp.vec3Z wid              -- in*OH + ioh

  -- i = thread index within IC*KH*KW dimension. Skip if out of range.
  let i := Exp.add (Exp.mul bx (Exp.litU32 256)) tid
  let iInBounds := Exp.lt i (Exp.litU32 IC_KH_KW)
  ShaderM.if_ iInBounds (do
    let iic := Exp.div i (Exp.litU32 KH_KW)
    let rem := Exp.sub i (Exp.mul iic (Exp.litU32 KH_KW))
    let ikh := Exp.div rem (Exp.litU32 KW)
    let ikw := Exp.sub rem (Exp.mul ikh (Exp.litU32 KW))

    -- N_OH = N * OH. Single iteration of the outer loop (assume iz < 65535).
    let in_ := Exp.div iz (Exp.litU32 OH)
    let ioh := Exp.sub iz (Exp.mul in_ (Exp.litU32 OH))

    -- iiw = iow * s0 + ikw * d0 - p0. Compute as u32; negative wraps to
    -- > 2^31, which fails `iiw < IW` (unsigned), so bound-check works
    -- without a signed setp.
    let iiw := Exp.sub (Exp.add (Exp.mul iow (Exp.litU32 s0))
                                 (Exp.mul ikw (Exp.litU32 d0)))
                        (Exp.litU32 p0)
    let iih := Exp.sub (Exp.add (Exp.mul ioh (Exp.litU32 s1))
                                 (Exp.mul ikh (Exp.litU32 d1)))
                        (Exp.litU32 p1)

    let iiwInBounds := Exp.lt iiw (Exp.litU32 IW)
    let iihInBounds := Exp.lt iih (Exp.litU32 IH)
    -- Combine two bool predicates: select(iiwInBounds, iihInBounds-as-u32, 0)
    -- gives 1 iff both are true. Avoids needing bool→u32 cast.
    let iihAsU32 := Exp.select iihInBounds (Exp.litU32 1) (Exp.litU32 0)
    let combined := Exp.select iiwInBounds iihAsU32 (Exp.litU32 0)
    let inBoundsBool := Exp.ne combined (Exp.litU32 0)

    -- Output offset: ((in*OH+ioh)*OW+iow)*IC_KH_KW + iic*KH_KW + ikh*KW + ikw
    let outRow := Exp.add (Exp.mul in_ (Exp.litU32 OH)) ioh
    let outRowOW := Exp.mul outRow (Exp.litU32 OW)
    let outRowOWiow := Exp.add outRowOW iow
    let outRowChunk := Exp.mul outRowOWiow (Exp.litU32 IC_KH_KW)
    let dstOff := Exp.add outRowChunk
      (Exp.add (Exp.mul iic (Exp.litU32 KH_KW))
        (Exp.add (Exp.mul ikh (Exp.litU32 KW)) ikw))

    ShaderM.if_ inBoundsBool (do
      -- src[in * (IC*IH*IW) + iic * (IH*IW) + iih * IW + iiw]
      let srcOff := Exp.add
        (Exp.add (Exp.mul in_ (Exp.litU32 srcBatchStride))
                 (Exp.mul iic (Exp.litU32 srcChannelStride)))
        (Exp.add (Exp.mul iih (Exp.litU32 IW)) iiw)
      let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := srcSize) "src" srcOff
      ShaderM.writeBuffer (ty := .scalar .f32) "dst" dstOff v
    ) (do
      ShaderM.writeBuffer (ty := .scalar .f32) "dst" dstOff (Exp.litF32 0.0)
    )
  ) (pure ())

/-- Naive f32 matmul kernel: `dst[m, n] = Σ_k a[m, k] * b[n, k]`.
    Note: `b` is stored row-major as `[N, K]` so `b[n, k]` is at
    `n * K + k`. This matches the conv2d use case where the weight
    tensor `[OC, IC*KH*KW]` is contiguous along the IC*KH*KW dimension.

    Grid: `(N, M, 1)` workgroups, `(1, 1, 1)` block. Each thread emits
    one output element. Slow (no smem, no fma chain) but parity-correct.

    Used as the matmul stage of conv2d via im2col. Not for hot-path use. -/
def matmulF32NaiveKernel
    (M N K : Nat) : ShaderM Unit := do
  let _a   ← ShaderM.declareReadOnlyBuffer "a"   (.array (.scalar .f32) (M * K))
  let _b   ← ShaderM.declareReadOnlyBuffer "b"   (.array (.scalar .f32) (N * K))
  let _dst ← ShaderM.declareOutputBuffer  "dst" (.array (.scalar .f32) (M * N))

  let wid ← ShaderM.workgroupId
  let n := Exp.vec3X wid               -- output column ∈ [0, N)
  let m := Exp.vec3Y wid               -- output row    ∈ [0, M)

  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  ShaderM.loop (Exp.litU32 0) (Exp.litU32 K) (Exp.litU32 1) fun k => do
    let aIdx := Exp.add (Exp.mul m (Exp.litU32 K)) k
    let bIdx := Exp.add (Exp.mul n (Exp.litU32 K)) k
    let av ← ShaderM.readBuffer (ty := .scalar .f32) (n := M * K) "a" aIdx
    let bv ← ShaderM.readBuffer (ty := .scalar .f32) (n := N * K) "b" bIdx
    let accExp : Exp (.scalar .f32) := Exp.var "acc"
    ShaderM.assign "acc" (Exp.add accExp (Exp.mul av bv))

  let dstIdx := Exp.add (Exp.mul m (Exp.litU32 N)) n
  let accExp : Exp (.scalar .f32) := Exp.var "acc"
  ShaderM.writeBuffer (ty := .scalar .f32) "dst" dstIdx accExp

end Hesper.Layers.Vision
