import Hesper.LoRA.Types
import Hesper.WGSL.Monad
import Hesper.WGSL.Execute
import Hesper.WGSL.Exp
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer

/-!
# Gradient Clipping and Scaling

GPU kernels for:
1. **Global gradient norm** — L2 norm across all LoRA parameter gradients
2. **Gradient clipping** — scale gradients if norm exceeds threshold
3. **Gradient scaling** — divide gradients by token count (loss normalization)

## Standard Values (matches PyTorch defaults)
- max_grad_norm = 1.0
- Loss normalization: divide gradients by number of output tokens
-/

namespace Hesper.Optimizer.GradientClip

open Hesper.WGSL
open Hesper.WGSL.Monad
open Hesper.WebGPU

/-- Buffers needed for gradient clipping -/
structure ClipBuffers where
  /-- Accumulator for sum of squared gradients [1] -/
  normSqBuf : Buffer
  /-- Temporary for per-buffer partial sums [1] -/
  partialBuf : Buffer

/-- Create clip buffers -/
def createClipBuffers (device : Device) : IO ClipBuffers := do
  let mkBuf := fun (n : Nat) =>
    createBuffer device { size := (n * 4).toUSize, usage := [.storage, .copySrc, .copyDst, .mapRead], mappedAtCreation := false }
  pure { normSqBuf := ← mkBuf 1, partialBuf := ← mkBuf 1 }

/-! ## Sum of Squares Kernel -/

/-- Compute sum of squares of a buffer, write result to accumulator (ADD to existing value).
    Uses single workgroup with shared memory reduction. -/
def sumSquaredKernel (numElements : Nat) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let lid ← ShaderM.localId
  let tid := Exp.vec3X lid

  let _grad ← ShaderM.declareInputBuffer "grad" (.array (.scalar .f32) numElements)
  let _accum ← ShaderM.declareOutputBuffer "accum" (.array (.scalar .f32) 1)

  ShaderM.sharedNamed "shared_sum" (.array (.scalar .f32) workgroupSize)

  -- Strided accumulation
  let localVar ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
  ShaderM.loop tid (Exp.litU32 numElements) (Exp.litU32 workgroupSize) fun i => do
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := numElements) "grad" i
    ShaderM.assign localVar (Exp.add (Exp.var localVar) (Exp.mul val val))

  ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_sum" tid (Exp.var localVar)
  ShaderM.barrier

  let numSteps := Nat.log2 workgroupSize
  ShaderM.staticLoop numSteps fun step => do
    let s := workgroupSize >>> (step + 1)
    ShaderM.if_ (Exp.lt tid (Exp.litU32 s)) (do
      let other ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_sum" (Exp.add tid (Exp.litU32 s))
      let cur ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_sum" tid
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_sum" tid (Exp.add cur other)
    ) (pure ())
    ShaderM.barrier

  -- Thread 0: add to accumulator
  ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
    let localSum ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_sum" (Exp.litU32 0)
    let oldAccum ← ShaderM.readBuffer (ty := .scalar .f32) (n := 1) "accum" (Exp.litU32 0)
    ShaderM.writeBuffer (ty := .scalar .f32) "accum" (Exp.litU32 0) (Exp.add oldAccum localSum)
  ) (pure ())

/-- Execute sum of squares and add to accumulator -/
def executeSumSquared (device : Device) (gradBuf accumBuf : Buffer) (numElements : Nat) : IO Unit := do
  let workgroupSize := 256
  let shader := sumSquaredKernel numElements workgroupSize
  let namedBuffers := [("grad", gradBuf), ("accum", accumBuf)]
  let execConfig : Hesper.WGSL.Execute.ExecutionConfig := {
    workgroupSize := {x := workgroupSize, y := 1, z := 1}
    numWorkgroups := (1, 1, 1)
  }
  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig

/-! ## Gradient Clip Kernel -/

/-- Scale gradient buffer by clip_factor = maxNorm / globalNorm (if norm > maxNorm).
    Reads globalNormSq[0], computes norm = sqrt(normSq), clips if needed. -/
def clipKernel (numElements : Nat) (maxNorm : Float) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let i := Exp.vec3X gid

  let _grad ← ShaderM.declareOutputBuffer "grad" (.array (.scalar .f32) numElements)
  let _normSq ← ShaderM.declareInputBuffer "normSq" (.array (.scalar .f32) 1)

  ShaderM.if_ (Exp.lt i (Exp.litU32 numElements)) (do
    let normSqVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := 1) "normSq" (Exp.litU32 0)
    let norm := Exp.sqrt (Exp.max normSqVal (Exp.litF32 1e-12))
    let clipFactor := Exp.div (Exp.litF32 maxNorm) (Exp.max norm (Exp.litF32 maxNorm))
    -- clipFactor = min(1.0, maxNorm / norm)
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := numElements) "grad" i
    ShaderM.writeBuffer (ty := .scalar .f32) "grad" i (Exp.mul val clipFactor)
  ) (pure ())

/-- Execute gradient clipping on a single buffer -/
def executeClip (device : Device) (gradBuf normSqBuf : Buffer) (numElements : Nat) (maxNorm : Float) : IO Unit := do
  let shader := clipKernel numElements maxNorm
  let namedBuffers := [("grad", gradBuf), ("normSq", normSqBuf)]
  let execConfig := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D numElements 256
  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig

/-! ## In-place Gradient Scale Kernel -/

/-- Scale gradient in-place: grad[i] *= scaleFactor -/
def scaleKernel (numElements : Nat) (scaleFactor : Float) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let i := Exp.vec3X gid

  let _grad ← ShaderM.declareOutputBuffer "grad" (.array (.scalar .f32) numElements)

  ShaderM.if_ (Exp.lt i (Exp.litU32 numElements)) (do
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := numElements) "grad" i
    ShaderM.writeBuffer (ty := .scalar .f32) "grad" i (Exp.mul val (Exp.litF32 scaleFactor))
  ) (pure ())

def executeScale (device : Device) (gradBuf : Buffer) (numElements : Nat) (scaleFactor : Float) : IO Unit := do
  let shader := scaleKernel numElements scaleFactor
  let namedBuffers := [("grad", gradBuf)]
  let execConfig := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D numElements 256
  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig

/-! ## High-Level API -/

/-- Clip gradients of all LoRA parameters to maxNorm (global L2 norm).
    Returns the gradient norm before clipping (for logging). -/
def clipGradNorm (device : Device) (adapter : Hesper.LoRA.Adapter)
    (grads : Hesper.LoRA.AdapterGrad) (maxNorm : Float)
    (clipBufs : ClipBuffers) : IO Float := do
  -- Zero the norm accumulator
  let zeroBytes := ByteArray.mk #[0, 0, 0, 0]
  writeBuffer device clipBufs.normSqBuf 0 zeroBytes

  -- Phase 1: Accumulate sum of squares across ALL gradient buffers
  for i in [:adapter.layers.size] do
    if h1 : i < adapter.layers.size then
      if h2 : i < grads.layers.size then
        let layer := adapter.layers[i]
        let grad := grads.layers[i]
        executeSumSquared device grad.gradQ.dA clipBufs.normSqBuf (layer.loraQ.rank * layer.loraQ.inDim)
        executeSumSquared device grad.gradQ.dB clipBufs.normSqBuf (layer.loraQ.outDim * layer.loraQ.rank)
        executeSumSquared device grad.gradV.dA clipBufs.normSqBuf (layer.loraV.rank * layer.loraV.inDim)
        executeSumSquared device grad.gradV.dB clipBufs.normSqBuf (layer.loraV.outDim * layer.loraV.rank)

  -- Phase 2: Clip all gradient buffers
  for i in [:adapter.layers.size] do
    if h1 : i < adapter.layers.size then
      if h2 : i < grads.layers.size then
        let layer := adapter.layers[i]
        let grad := grads.layers[i]
        executeClip device grad.gradQ.dA clipBufs.normSqBuf (layer.loraQ.rank * layer.loraQ.inDim) maxNorm
        executeClip device grad.gradQ.dB clipBufs.normSqBuf (layer.loraQ.outDim * layer.loraQ.rank) maxNorm
        executeClip device grad.gradV.dA clipBufs.normSqBuf (layer.loraV.rank * layer.loraV.inDim) maxNorm
        executeClip device grad.gradV.dB clipBufs.normSqBuf (layer.loraV.outDim * layer.loraV.rank) maxNorm

  -- Read back norm for logging
  let normBytes ← mapBufferRead device clipBufs.normSqBuf 0 4
  let b0 := normBytes.get! 0 |>.toUInt32
  let b1 := normBytes.get! 1 |>.toUInt32
  let b2 := normBytes.get! 2 |>.toUInt32
  let b3 := normBytes.get! 3 |>.toUInt32
  let bits := b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24)
  let normSq := Hesper.Basic.float32BitsToFloat64 bits
  pure (Float.sqrt normSq)

/-- Scale all gradients by a factor (e.g., 1/numTokens for loss normalization) -/
def scaleGrads (device : Device) (adapter : Hesper.LoRA.Adapter)
    (grads : Hesper.LoRA.AdapterGrad) (scaleFactor : Float) : IO Unit := do
  for i in [:adapter.layers.size] do
    if h1 : i < adapter.layers.size then
      if h2 : i < grads.layers.size then
        let layer := adapter.layers[i]
        let grad := grads.layers[i]
        executeScale device grad.gradQ.dA (layer.loraQ.rank * layer.loraQ.inDim) scaleFactor
        executeScale device grad.gradQ.dB (layer.loraQ.outDim * layer.loraQ.rank) scaleFactor
        executeScale device grad.gradV.dA (layer.loraV.rank * layer.loraV.inDim) scaleFactor
        executeScale device grad.gradV.dB (layer.loraV.outDim * layer.loraV.rank) scaleFactor

end Hesper.Optimizer.GradientClip
