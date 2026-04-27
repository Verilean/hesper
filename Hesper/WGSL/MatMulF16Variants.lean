import Hesper.WGSL.Monad
import Hesper.WGSL.MatMul

/-!
# F16 lm_head matmul block-size variants for microbenchmarking

Multiple workgroup-size variants of the M=1 packed-half2 matmul,
sharing one body so block-size effects (occupancy / wave count /
intra-warp parallelism) can be isolated.

The body is the same shape as `matMulTransposeF16BlockCoopKernel`
(`Hesper/WGSL/MatMul.lean:246`) — one workgroup per output row,
each thread covers an even subset of the K dimension via stride
loop, partial sums reduced through a smem tree (or warp-shuffle
when `bs=32`).

llama.cpp's `mul_mat_vec_f<half,float,1,256>` uses bs=256 = 8 warps
per output row; that's the head-to-head baseline.
-/

namespace Hesper.WGSL.MatMul

open Hesper.WGSL
open Hesper.WGSL.Monad

/-- Generic packed-half2 lm_head matmul, parameterised by block size
    `bs` (must be 32 / 64 / 128 / 256).  One row per workgroup.

    For `bs > 32`, partial sums are reduced through a smem tree
    (warp-level subgroupAdd then cross-warp reduction).

    Layout matches `matMulTransposeF16BlockCoopKernel`:
    * `a`: f32 K-vector (length `K`)
    * `b`: u32 packed half2 weights (length `N * K/2`), row-major
    * `c`: f32 output (length `N`)

    Each thread reads `K/(2*bs)` u32 (= 2 half2 lanes) per stride iter.
    Total per-thread iters: `K / (2*bs)`.  At K=2560, bs=256 → 5 iters
    (matches llama.cpp niter_best).  At bs=32 → 40 iters.
-/
def matMulTransposeF16RowBlockKernel (config : Config) (bs : Nat) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let outIdx := Exp.vec3X wid
  let tid := Exp.vec3X lid

  let packedK := config.K / 2

  let _a ← ShaderM.declareInputBuffer "a" (.array (.scalar .f32) config.K)
  let _b ← ShaderM.declareInputBuffer "b" (.array (.scalar .u32) (config.N * packedK))
  let _c ← ShaderM.declareOutputBuffer "c" (.array (.scalar .f32) config.N)

  let inBounds := Exp.lt outIdx (Exp.litU32 config.N)

  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"

  let rowBaseU32 := Exp.mul outIdx (Exp.litU32 packedK)

  -- Strided per-thread loop: each thread covers K/(2*bs) u32 entries.
  ShaderM.loop tid (Exp.litU32 packedK) (Exp.litU32 bs) fun k => do
    let packed ← ShaderM.readBuffer (ty := .scalar .u32) (n := config.N * packedK) "b"
                  (Exp.add rowBaseU32 k)
    let unpacked := Exp.unpack2x16float packed
    let b0 := Exp.vecX unpacked
    let b1 := Exp.vecY unpacked
    let kf := Exp.mul k (Exp.litU32 2)
    let a0 ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.K) "a" kf
    let a1 ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.K) "a" (Exp.add kf (Exp.litU32 1))
    ShaderM.assign "acc" (Exp.add acc (Exp.add (Exp.mul a0 b0) (Exp.mul a1 b1)))

  -- Reduction.  For bs=32, a single subgroupAdd; for larger bs, fold
  -- across warps via shared memory.  llama.cpp uses warp-shuffle for
  -- intra-warp + smem fan-in across `nwarps`.
  if bs ≤ 32 then
    ShaderM.varNamed "total" (.scalar .f32) (Exp.subgroupAdd acc)
    let total : Exp (.scalar .f32) := Exp.var "total"
    ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) inBounds) (do
      ShaderM.writeBuffer (ty := .scalar .f32) "c" outIdx total
    ) (pure ())
  else
    let nwarps := bs / 32
    -- intra-warp reduce
    let warpSum := Exp.subgroupAdd acc
    let laneId := Exp.bitAnd tid (Exp.litU32 31)
    let warpId := Exp.shiftRight tid (Exp.litU32 5)
    -- smem [nwarps]
    ShaderM.sharedNamed "warp_partial" (.array (.scalar .f32) nwarps)
    ShaderM.if_ (Exp.eq laneId (Exp.litU32 0)) (do
      ShaderM.writeWorkgroup (ty := .scalar .f32) "warp_partial" warpId warpSum
    ) (pure ())
    ShaderM.barrier
    -- warp 0 reduces the nwarps partials via subgroupAdd over a
    -- pred-loaded scalar (zero-pad lanes >= nwarps).
    ShaderM.if_ (Exp.eq warpId (Exp.litU32 0)) (do
      ShaderM.varNamed "wp" (.scalar .f32) (Exp.litF32 0.0)
      ShaderM.if_ (Exp.lt laneId (Exp.litU32 nwarps)) (do
        let v ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := nwarps) "warp_partial" laneId
        ShaderM.assign "wp" v
      ) (pure ())
      let final := Exp.subgroupAdd (Exp.var "wp" : Exp (.scalar .f32))
      ShaderM.if_ (Exp.and (Exp.eq laneId (Exp.litU32 0)) inBounds) (do
        ShaderM.writeBuffer (ty := .scalar .f32) "c" outIdx final
      ) (pure ())
    ) (pure ())

end Hesper.WGSL.MatMul
