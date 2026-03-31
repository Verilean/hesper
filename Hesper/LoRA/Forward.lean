import Hesper.LoRA.Types
import Hesper.WGSL.Monad
import Hesper.WGSL.Execute
import Hesper.WGSL.Exp
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.Logging

/-!
# LoRA Forward Pass GPU Kernels

Implements the LoRA forward computation on GPU:

```
output = BitLinear(x) + (alpha / rank) * B @ (A @ x)
```

Decomposed into three GPU operations:
1. **loraProjectA**: h = A @ x  ([rank] = [rank, inDim] @ [inDim])
2. **loraProjectB**: y = B @ h  ([outDim] = [outDim, rank] @ [rank])
3. **loraFusedAdd**: output[i] += scale * y[i]

For single-token training (rank=8, dim=2560), these are very small matmuls.
-/

namespace Hesper.LoRA.Forward

open Hesper.WGSL
open Hesper.WGSL.Monad
open Hesper.WebGPU

/-! ## GPU Kernels -/

/-- Kernel: h = A @ x
    A is [rank, inDim] row-major, x is [inDim], h is [rank].
    Each thread computes one element of h (one dot product over inDim). -/
def loraProjectAKernel (rank inDim : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let r := Exp.vec3X gid  -- row index in A (0..rank-1)

  let _a ← ShaderM.declareInputBuffer "a" (.array (.scalar .f32) (rank * inDim))
  let _x ← ShaderM.declareInputBuffer "x" (.array (.scalar .f32) inDim)
  let _h ← ShaderM.declareOutputBuffer "h" (.array (.scalar .f32) rank)

  ShaderM.if_ (Exp.lt r (Exp.litU32 rank)) (do
    let (accName, acc) ← ShaderM.varRef (.scalar .f32) (Exp.litF32 0.0)
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 inDim) (Exp.litU32 1) fun j => do
      let aIdx := Exp.add (Exp.mul r (Exp.litU32 inDim)) j
      let aVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := rank * inDim) "a" aIdx
      let xVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := inDim) "x" j
      ShaderM.assign accName (Exp.add acc (Exp.mul aVal xVal))
    ShaderM.writeBuffer (ty := .scalar .f32) "h" r acc
  ) (pure ())

/-- Kernel: y = B @ h
    B is [outDim, rank] row-major, h is [rank], y is [outDim].
    Each thread computes one element of y. -/
def loraProjectBKernel (outDim rank : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let i := Exp.vec3X gid  -- row index in B (0..outDim-1)

  let _b ← ShaderM.declareInputBuffer "b" (.array (.scalar .f32) (outDim * rank))
  let _h ← ShaderM.declareInputBuffer "h" (.array (.scalar .f32) rank)
  let _y ← ShaderM.declareOutputBuffer "y" (.array (.scalar .f32) outDim)

  ShaderM.if_ (Exp.lt i (Exp.litU32 outDim)) (do
    let (accName, acc) ← ShaderM.varRef (.scalar .f32) (Exp.litF32 0.0)
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 rank) (Exp.litU32 1) fun r => do
      let bIdx := Exp.add (Exp.mul i (Exp.litU32 rank)) r
      let bVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := outDim * rank) "b" bIdx
      let hVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := rank) "h" r
      ShaderM.assign accName (Exp.add acc (Exp.mul bVal hVal))
    ShaderM.writeBuffer (ty := .scalar .f32) "y" i acc
  ) (pure ())

/-- Kernel: output[i] += scale * y[i]
    Adds the LoRA contribution to the base BitLinear output in-place.
    `output` is read-write (already contains base output). -/
def loraAddScaledKernel (numElements : Nat) (scale : Float) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let i := Exp.vec3X gid

  let _y ← ShaderM.declareInputBuffer "y" (.array (.scalar .f32) numElements)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) numElements)

  ShaderM.if_ (Exp.lt i (Exp.litU32 numElements)) (do
    let outVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numElements) "output" i
    let yVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numElements) "y" i
    ShaderM.writeBuffer (ty := .scalar .f32) "output" i (Exp.add outVal (Exp.mul (Exp.litF32 scale) yVal))
  ) (pure ())

/-! ## Execution Functions -/

/-- Execute LoRA A projection: h = A @ x -/
def executeProjectA (device : Device) (weight : Hesper.LoRA.Weight)
    (xBuf hBuf : Buffer) : IO Unit := do
  let shader := loraProjectAKernel weight.rank weight.inDim
  let namedBuffers := [("a", weight.a), ("x", xBuf), ("h", hBuf)]
  let execConfig := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D weight.rank 64
  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig

/-- Execute LoRA B projection: y = B @ h -/
def executeProjectB (device : Device) (weight : Hesper.LoRA.Weight)
    (hBuf yBuf : Buffer) : IO Unit := do
  let shader := loraProjectBKernel weight.outDim weight.rank
  let namedBuffers := [("b", weight.b), ("h", hBuf), ("y", yBuf)]
  let execConfig := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D weight.outDim 256
  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig

/-- Execute LoRA add: output += scale * y -/
def executeAddScaled (device : Device) (yBuf outputBuf : Buffer)
    (numElements : Nat) (scale : Float) : IO Unit := do
  let shader := loraAddScaledKernel numElements scale
  let namedBuffers := [("y", yBuf), ("output", outputBuf)]
  let execConfig := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D numElements 256
  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig

/-- Full LoRA forward pass for a single projection.
    Computes: outputBuf += (alpha/rank) * B @ (A @ inputBuf)

    @param device GPU device
    @param weight LoRA weight pair (A, B)
    @param scale The alpha/rank scaling factor
    @param inputBuf Input buffer [inDim] (shared with base BitLinear input)
    @param outputBuf Output buffer [outDim] (already contains base BitLinear output)
    @param hBuf Temporary buffer [rank] for intermediate h = A @ x
    @param yBuf Temporary buffer [outDim] for y = B @ h -/
def executeLoRAForward (device : Device) (weight : Hesper.LoRA.Weight) (scale : Float)
    (inputBuf outputBuf hBuf yBuf : Buffer) : IO Unit := do
  -- Step 1: h = A @ x
  executeProjectA device weight inputBuf hBuf
  -- Step 2: y = B @ h
  executeProjectB device weight hBuf yBuf
  -- Step 3: output += scale * y
  executeAddScaled device yBuf outputBuf weight.outDim scale

/-- Save input activation for backward pass (copy inputBuf to savedBuf) -/
def saveActivation (device : Device) (srcBuf dstBuf : Buffer) (numElements : Nat) : IO Unit := do
  -- Use a simple copy kernel
  let shader : ShaderM Unit := do
    let gid ← ShaderM.globalId
    let i := Exp.vec3X gid
    let _src ← ShaderM.declareInputBuffer "src" (.array (.scalar .f32) numElements)
    let _dst ← ShaderM.declareOutputBuffer "dst" (.array (.scalar .f32) numElements)
    ShaderM.if_ (Exp.lt i (Exp.litU32 numElements)) (do
      let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := numElements) "src" i
      ShaderM.writeBuffer (ty := .scalar .f32) "dst" i val
    ) (pure ())
  let namedBuffers := [("src", srcBuf), ("dst", dstBuf)]
  let execConfig := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D numElements 256
  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig

end Hesper.LoRA.Forward
