import Hesper.WGSL.Monad
import Hesper.WGSL.Exp
import Hesper.Backend
import Hesper.Core.Differentiable

/-!
# Mean Squared Error (MSE) Loss — verified op + GPU kernels

`MSEOp` is the verified-op tag for `L = (1/N) Σᵢ (predᵢ − targetᵢ)²`.
It carries:

* a `Differentiable MSEOp (Array Float × Array Float) Float` instance
  (CPU spec: forward = the loss, backward = `(2/N)(pred − target)` w.r.t. `pred`,
   and `−(2/N)(pred − target)` w.r.t. `target` — symmetric pair),
* a forward GPU kernel that writes the scalar loss into a 1-element buffer,
* a backward GPU kernel that writes `dPred[i] = (2/N)(pred[i] − target[i])`,
* thin `executeMSEForward` / `executeMSEBackward` wrappers in the same
  shape as `Hesper.Training.Loss.executeCrossEntropy{Forward,Backward}`.

Used end-to-end in `Examples/Tutorial/CaliforniaHousingGPU.lean` for
linear-regression training on California Housing.
-/

namespace Hesper.Training.MSE

open Hesper.WGSL
open Hesper.WGSL.Monad
open Hesper.Core

/-! ## Verified-op tag + CPU `Differentiable` spec -/

/-- Verified-op tag for mean-squared-error loss.  Carrying nothing —
    `forward` and `backward` are functions of the input arrays only. -/
structure MSEOp deriving Inhabited

/-- CPU specification: `L = (1/N) Σᵢ (pᵢ − yᵢ)²`.

    `forward` returns the scalar mean.  `backward` returns the
    vector-Jacobian product w.r.t. the upstream gradient `v` (1.0 for
    a scalar loss head): `dL/dPred[i] = (2/N)(p − y) · v`, and
    `dL/dTarget` is its mirror image.  Empty inputs return zeros so
    the spec is total. -/
instance : Differentiable MSEOp (Array Float × Array Float) Float where
  forward _op input :=
    let preds   := input.fst
    let targets := input.snd
    let n := preds.size
    if n = 0 then 0.0
    else Id.run do
      let mut s : Float := 0.0
      for i in [0:n] do
        let d := preds[i]! - targets[i]!
        s := s + d * d
      return s / n.toFloat
  backward _op input v :=
    let preds   := input.fst
    let targets := input.snd
    let n := preds.size
    if n = 0 then (#[], #[])
    else
      let scale := 2.0 * v / n.toFloat
      let dPred : Array Float := Array.ofFn (n := n) fun i =>
        scale * (preds[i.val]! - targets[i.val]!)
      let dTarget := dPred.map (· * (-1.0))
      (dPred, dTarget)

/-! ## Forward GPU kernel: scalar loss = (1/N) Σᵢ (predᵢ − yᵢ)² -/

/-- One workgroup of `workgroupSize` threads strided-sums the squared
    residuals across `nRows` and writes the mean into `loss[0]`.  Same
    tree-reduction shape as `Hesper.Training.Loss.crossEntropyForwardKernel`. -/
def forwardKernel (nRows : Nat) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let lid ← ShaderM.localId
  let tidX := Exp.vec3X lid

  let _pred ← ShaderM.declareInputBuffer "pred"   (.array (.scalar .f32) nRows)
  let _y    ← ShaderM.declareInputBuffer "tgt" (.array (.scalar .f32) nRows)
  let _loss ← ShaderM.declareOutputBuffer "loss"  (.array (.scalar .f32) 1)
  ShaderM.sharedNamed "ssum" (.array (.scalar .f32) workgroupSize)

  -- Phase 1: per-thread strided sum of (p − y)².
  let (sumName, sumVal) ← ShaderM.varRef (.scalar .f32) (Exp.litF32 0.0)
  ShaderM.loop tidX (Exp.litU32 nRows) (Exp.litU32 workgroupSize) fun i => do
    let p ← ShaderM.readBuffer (ty := .scalar .f32) (n := nRows) "pred"   i
    let t ← ShaderM.readBuffer (ty := .scalar .f32) (n := nRows) "tgt" i
    let d := Exp.sub p t
    ShaderM.assign sumName (Exp.add sumVal (Exp.mul d d))
  ShaderM.writeWorkgroup (ty := .scalar .f32) "ssum" tidX sumVal
  ShaderM.barrier

  -- Phase 2: tree reduction in shared memory.
  let numSteps := Nat.log2 workgroupSize
  ShaderM.staticLoop numSteps fun step => do
    let s := workgroupSize >>> (step + 1)
    ShaderM.if_ (Exp.lt tidX (Exp.litU32 s)) (do
      let other ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "ssum"
                    (Exp.add tidX (Exp.litU32 s))
      let cur   ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "ssum" tidX
      ShaderM.writeWorkgroup (ty := .scalar .f32) "ssum" tidX (Exp.add cur other)
    ) (pure ())
    ShaderM.barrier

  -- Phase 3: thread 0 publishes the mean.
  ShaderM.if_ (Exp.eq tidX (Exp.litU32 0)) (do
    let total ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "ssum" (Exp.litU32 0)
    ShaderM.writeBuffer (ty := .scalar .f32) "loss" (Exp.litU32 0)
      (Exp.div total (Exp.litF32 nRows.toFloat))
  ) (pure ())

/-- Execute MSE forward.  Reads `pred`/`target` (length `nRows`),
    writes the scalar mean loss into `loss[0]`. -/
@[inline]
def executeMSEForward [Hesper.GPUBackend β] (ctx : β)
    (predBuf targetBuf lossBuf : Hesper.GPUBackend.Buf β)
    (nRows : Nat) : IO Unit := do
  let workgroupSize := 256
  Hesper.GPUBackend.execute ctx
    (forwardKernel nRows workgroupSize)
    [("pred", predBuf), ("tgt", targetBuf), ("loss", lossBuf)]
    { workgroupSize := { x := workgroupSize }, numWorkgroups := (1, 1, 1) }

/-! ## Backward GPU kernel: `dPred[i] = (2/N) (predᵢ − yᵢ)` -/

/-- One thread per row writes the per-element gradient.  The `2/N`
    factor is baked into the kernel as a literal so the SGD step
    downstream only needs to multiply by `lr`. -/
def backwardKernel (nRows : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let i := Exp.vec3X gid
  let _pred ← ShaderM.declareInputBuffer  "pred"   (.array (.scalar .f32) nRows)
  let _y    ← ShaderM.declareInputBuffer  "tgt" (.array (.scalar .f32) nRows)
  let _dp   ← ShaderM.declareOutputBuffer "dPred"  (.array (.scalar .f32) nRows)
  let scale : Float := 2.0 / nRows.toFloat
  ShaderM.if_ (Exp.lt i (Exp.litU32 nRows)) (do
    let p ← ShaderM.readBuffer (ty := .scalar .f32) (n := nRows) "pred"   i
    let t ← ShaderM.readBuffer (ty := .scalar .f32) (n := nRows) "tgt" i
    ShaderM.writeBuffer (ty := .scalar .f32) "dPred" i
      (Exp.mul (Exp.litF32 scale) (Exp.sub p t))
  ) (pure ())

/-- Execute MSE backward.  Writes `dPred[i] = (2/N)(predᵢ − yᵢ)` into
    `dPredBuf` (length `nRows`). -/
@[inline]
def executeMSEBackward [Hesper.GPUBackend β] (ctx : β)
    (predBuf targetBuf dPredBuf : Hesper.GPUBackend.Buf β)
    (nRows : Nat) : IO Unit :=
  Hesper.GPUBackend.execute ctx
    (backwardKernel nRows)
    [("pred", predBuf), ("tgt", targetBuf), ("dPred", dPredBuf)]
    (Hesper.ExecConfig.dispatch1D nRows 256)

end Hesper.Training.MSE
