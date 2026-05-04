import Hesper.WGSL.Monad
import Hesper.WGSL.Exp
import Hesper.Backend

/-!
# Audio model kernels (1D conv-transpose, ...)

Foundation for running 1D-conv-based audio decoders (Mamba/SSM-adjacent
ops, small TTS upsamplers, audio codec heads). Mirrors llama.cpp's
`ggml/src/ggml-cuda/conv-transpose-1d.cu` algorithm.

## conv_transpose_1d

Input: `src1 : f32 [IC, IL]` (input length × in_channels)
Weights: `src0 : f32 [IC, OC, KW]` (kernel × out_channels × in_channels)
Output: `dst : f32 [OC, OL]` (output length × out_channels)

Each thread emits one output element at flat index `g = threadIdx.x +
blockIdx.x * blockDim.x` < `output_size = OC * OL`. The output channel
is `g / OL` and the output position `idx = g % OL`. The kernel sums

  dst[c_out, idx] = Σ_{c_in} Σ_{i: idx - i*s0 ∈ [0, KW)}
                      src1[c_in, i] * src0[c_in, c_out, idx - i*s0]

llama.cpp passes the strides as `srcN_neK` parameters (logical shape
along axis K). hesper's port hardcodes them as compile-time `Nat`s
because the audio decoder shape is small and known at dispatch time.

## Why no early-continue

hesper's `ShaderM.if_` has no early-`continue` form. Instead the inner
loop body is wrapped in `if (in-range) acc += ...`. For non-overlapping
strides (`s0 >= KW`) at most one `i` matches per `(idx, c_in)` pair, so
the cost is identical to llama.cpp's branch. For overlapping strides
the behavior is identical too — both implementations sum every match.

## Bound-check trick

`(idx - i*s0)` can go negative when `i*s0 > idx`. Compute as u32; the
result wraps to ≥ 2^31, which fails the unsigned compare against `KW`
(< 2^31 for any realistic kernel). Same trick used in im2col.
-/

namespace Hesper.Layers.Audio

open Hesper
open Hesper.WGSL.Monad (ShaderM)
open Hesper.WGSL (Exp)

/-- 1D transpose convolution kernel for f32 input/weights/output.

  Static parameters (compile-time):
  - `IC` : input channels (`src0_ne2` and `src1_ne1` in llama.cpp)
  - `OC` : output channels (`src0_ne1` = `dst_ne1`)
  - `KW` : kernel width (`src0_ne0`)
  - `IL` : input length (`src1_ne0`)
  - `OL` : output length (`dst_ne0`)
  - `s0` : stride

  Dispatch grid (caller responsibility):
  - block.x = 256
  - grid.x  = ceil(OC * OL / 256)

  Each thread emits exactly one f32 output element. -/
def convTranspose1dF32Kernel
    (IC OC KW IL OL : Nat) (s0 : Nat)
    : ShaderM Unit := do
  let outSize := OC * OL
  let srcSize := IC * IL                    -- src1 = input
  let wSize := IC * OC * KW                 -- src0 = weights

  let _src ← ShaderM.declareReadOnlyBuffer "src" (.array (.scalar .f32) srcSize)
  let _w   ← ShaderM.declareReadOnlyBuffer "w"   (.array (.scalar .f32) wSize)
  let _dst ← ShaderM.declareOutputBuffer  "dst" (.array (.scalar .f32) outSize)

  let lid ← ShaderM.localId
  let wid ← ShaderM.workgroupId
  let tid := Exp.vec3X lid
  let bx  := Exp.vec3X wid
  let g := Exp.add (Exp.mul bx (Exp.litU32 256)) tid
  let inBounds := Exp.lt g (Exp.litU32 outSize)
  ShaderM.if_ inBounds (do
    -- out_index = g / OL; idx = g % OL
    let outChan := Exp.div g (Exp.litU32 OL)
    let idx     := Exp.sub g (Exp.mul outChan (Exp.litU32 OL))

    -- Per-thread accumulator initialised to zero.
    ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)

    -- Outer loop over IC, inner loop over KW.
    -- llama.cpp matches `for c in [0, IC); for i in [0, KW); …`.
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 IC) (Exp.litU32 1) fun c => do
      ShaderM.loop (Exp.litU32 0) (Exp.litU32 IL) (Exp.litU32 1) fun i => do
        -- weight_idx = idx - i * s0; valid iff < KW (unsigned).
        let iMulS := Exp.mul i (Exp.litU32 s0)
        let wIdx := Exp.sub idx iMulS
        let wIdxOk := Exp.lt wIdx (Exp.litU32 KW)
        ShaderM.if_ wIdxOk (do
          -- src0 layout: [IC, OC, KW] row-major → c * (OC*KW) + outChan * KW + wIdx
          let wOff := Exp.add
            (Exp.add (Exp.mul c (Exp.litU32 (OC * KW)))
                     (Exp.mul outChan (Exp.litU32 KW)))
            wIdx
          -- src1 layout: [IC, IL] row-major → c * IL + i
          let sOff := Exp.add (Exp.mul c (Exp.litU32 IL)) i
          let kw ← ShaderM.readBuffer (ty := .scalar .f32) (n := wSize) "w" wOff
          let iv ← ShaderM.readBuffer (ty := .scalar .f32) (n := srcSize) "src" sOff
          let accExp : Exp (.scalar .f32) := Exp.var "acc"
          ShaderM.assign "acc" (Exp.add accExp (Exp.mul kw iv))
        ) (pure ())

    -- Store: dst layout [OC, OL] row-major → outChan * OL + idx = g.
    let accExp : Exp (.scalar .f32) := Exp.var "acc"
    ShaderM.writeBuffer (ty := .scalar .f32) "dst" g accExp
  ) (pure ())

end Hesper.Layers.Audio
