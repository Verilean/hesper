import Hesper.LoRA.Types
import Hesper.WGSL.Monad
import Hesper.WGSL.Execute
import Hesper.WGSL.Exp
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer

/-!
# LoRA Backward Pass GPU Kernels

Given upstream gradient dOutput [outDim], computes:

1. **dB** = scale * outer(dOutput, h)   where h = A @ x (saved from forward)
2. **dA** = scale * outer(B^T @ dOutput, x)  where x is saved from forward
3. **dInput** += A^T @ (B^T @ dOutput) * scale  (gradient to residual stream)

All operations are small due to low rank (4-16).
-/

namespace Hesper.LoRA.Backward

open Hesper.WGSL
open Hesper.WGSL.Monad
open Hesper.WebGPU

/-! ## GPU Kernels -/

/-- Kernel: dB[i, r] += scale * dOutput[i] * h[r]
    Outer product of dOutput [outDim] and h [rank].
    Each thread computes one element of the [outDim, rank] gradient matrix. -/
def gradBKernel (outDim rank : Nat) (scale : Float) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid  -- linear index into [outDim * rank]

  let _dOutput ← ShaderM.declareInputBuffer "dOutput" (.array (.scalar .f32) outDim)
  let _h ← ShaderM.declareInputBuffer "h" (.array (.scalar .f32) rank)
  let _dB ← ShaderM.declareOutputBuffer "dB" (.array (.scalar .f32) (outDim * rank))

  let totalElements := outDim * rank
  let inBounds := Exp.lt idx (Exp.litU32 totalElements)

  -- Decompose linear index: i = idx / rank, r = idx % rank
  let i := Exp.div idx (Exp.litU32 rank)
  let r := Exp.mod idx (Exp.litU32 rank)

  let dOutVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := outDim) "dOutput" i
  let hVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := rank) "h" r

  -- dB[i,r] += scale * dOutput[i] * h[r]
  let oldDB ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalElements) "dB" idx
  let grad := Exp.mul (Exp.litF32 scale) (Exp.mul dOutVal hVal)
  let result := Exp.add oldDB grad
  let finalResult := Exp.select inBounds result (Exp.litF32 0.0)

  ShaderM.writeBuffer (ty := .scalar .f32) "dB" idx finalResult

/-- Kernel: dh[r] = sum_i B[i, r] * dOutput[i]
    Computes B^T @ dOutput. Each thread computes one element of dh [rank]. -/
def gradDhKernel (outDim rank : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let r := Exp.vec3X gid  -- index into [rank]

  let _b ← ShaderM.declareInputBuffer "b" (.array (.scalar .f32) (outDim * rank))
  let _dOutput ← ShaderM.declareInputBuffer "dOutput" (.array (.scalar .f32) outDim)
  let _dh ← ShaderM.declareOutputBuffer "dh" (.array (.scalar .f32) rank)

  let inBounds := Exp.lt r (Exp.litU32 rank)

  -- dh[r] = sum_i B[i, r] * dOutput[i]
  let (accName, acc) ← ShaderM.varRef (.scalar .f32) (Exp.litF32 0.0)
  ShaderM.loop (Exp.litU32 0) (Exp.litU32 outDim) (Exp.litU32 1) fun i => do
    let bIdx := Exp.add (Exp.mul i (Exp.litU32 rank)) r
    let bVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := outDim * rank) "b" bIdx
    let dOutVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := outDim) "dOutput" i
    ShaderM.assign accName (Exp.add acc (Exp.mul bVal dOutVal))

  let result := Exp.select inBounds acc (Exp.litF32 0.0)
  ShaderM.writeBuffer (ty := .scalar .f32) "dh" r result

/-- Kernel: dA[r, j] += scale * dh[r] * x[j]
    Outer product of dh [rank] and x [inDim].
    Each thread computes one element of the [rank, inDim] gradient matrix. -/
def gradAKernel (rank inDim : Nat) (scale : Float) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid  -- linear index into [rank * inDim]

  let _dh ← ShaderM.declareInputBuffer "dh" (.array (.scalar .f32) rank)
  let _x ← ShaderM.declareInputBuffer "x" (.array (.scalar .f32) inDim)
  let _dA ← ShaderM.declareOutputBuffer "dA" (.array (.scalar .f32) (rank * inDim))

  let totalElements := rank * inDim
  let inBounds := Exp.lt idx (Exp.litU32 totalElements)

  -- Decompose: r = idx / inDim, j = idx % inDim
  let r := Exp.div idx (Exp.litU32 inDim)
  let j := Exp.mod idx (Exp.litU32 inDim)

  let dhVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := rank) "dh" r
  let xVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := inDim) "x" j

  -- dA[r,j] += scale * dh[r] * x[j]
  let oldDA ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalElements) "dA" idx
  let grad := Exp.mul (Exp.litF32 scale) (Exp.mul dhVal xVal)
  let result := Exp.add oldDA grad
  let finalResult := Exp.select inBounds result (Exp.litF32 0.0)

  ShaderM.writeBuffer (ty := .scalar .f32) "dA" idx finalResult

/-- Kernel: dInput[j] += scale * sum_r A[r, j] * dh[r]
    Propagates gradient back through LoRA to the residual stream.
    Each thread computes one element of dInput [inDim]. -/
def inputGradKernel (rank inDim : Nat) (scale : Float) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let j := Exp.vec3X gid  -- index into [inDim]

  let _a ← ShaderM.declareInputBuffer "a" (.array (.scalar .f32) (rank * inDim))
  let _dh ← ShaderM.declareInputBuffer "dh" (.array (.scalar .f32) rank)
  let _dInput ← ShaderM.declareOutputBuffer "dInput" (.array (.scalar .f32) inDim)

  let inBounds := Exp.lt j (Exp.litU32 inDim)

  -- dInput[j] += scale * sum_r A[r, j] * dh[r]
  let oldDInput ← ShaderM.readBuffer (ty := .scalar .f32) (n := inDim) "dInput" j
  let (accName, acc) ← ShaderM.varRef (.scalar .f32) (Exp.litF32 0.0)
  ShaderM.loop (Exp.litU32 0) (Exp.litU32 rank) (Exp.litU32 1) fun r => do
    let aIdx := Exp.add (Exp.mul r (Exp.litU32 inDim)) j
    let aVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := rank * inDim) "a" aIdx
    let dhVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := rank) "dh" r
    ShaderM.assign accName (Exp.add acc (Exp.mul aVal dhVal))

  let grad := Exp.mul (Exp.litF32 scale) acc
  let result := Exp.add oldDInput grad
  let finalResult := Exp.select inBounds result (Exp.litF32 0.0)

  ShaderM.writeBuffer (ty := .scalar .f32) "dInput" j finalResult

/-! ## Execution Functions -/

/-- Execute gradient computation for B: dB += scale * outer(dOutput, h) -/
def executeGradB (device : Device) (dOutputBuf hBuf dBBuf : Buffer)
    (outDim rank : Nat) (scale : Float) : IO Unit := do
  let shader := gradBKernel outDim rank scale
  let namedBuffers := [("dOutput", dOutputBuf), ("h", hBuf), ("dB", dBBuf)]
  let execConfig := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D (outDim * rank) 256
  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig

/-- Execute B^T @ dOutput to get dh [rank] -/
def executeGradDh (device : Device) (bBuf dOutputBuf dhBuf : Buffer)
    (outDim rank : Nat) : IO Unit := do
  let shader := gradDhKernel outDim rank
  let namedBuffers := [("b", bBuf), ("dOutput", dOutputBuf), ("dh", dhBuf)]
  let execConfig := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D rank 64
  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig

/-- Execute gradient computation for A: dA += scale * outer(dh, x) -/
def executeGradA (device : Device) (dhBuf xBuf dABuf : Buffer)
    (rank inDim : Nat) (scale : Float) : IO Unit := do
  let shader := gradAKernel rank inDim scale
  let namedBuffers := [("dh", dhBuf), ("x", xBuf), ("dA", dABuf)]
  let execConfig := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D (rank * inDim) 256
  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig

/-- Execute input gradient propagation: dInput += scale * A^T @ dh -/
def executeInputGrad (device : Device) (aBuf dhBuf dInputBuf : Buffer)
    (rank inDim : Nat) (scale : Float) : IO Unit := do
  let shader := inputGradKernel rank inDim scale
  let namedBuffers := [("a", aBuf), ("dh", dhBuf), ("dInput", dInputBuf)]
  let execConfig := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D inDim 256
  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig

/-- Full LoRA backward pass for a single projection.
    Computes dA, dB gradients and propagates dInput.

    @param device GPU device
    @param weight LoRA weight (A, B matrices)
    @param grad Gradient buffers to accumulate into
    @param scale alpha/rank scaling factor
    @param dOutputBuf Upstream gradient [outDim]
    @param savedX Saved input from forward pass [inDim]
    @param savedH Saved intermediate h = A @ x from forward [rank]
    @param dInputBuf Buffer to accumulate input gradient into [inDim]
    @param dhBuf Temporary buffer [rank] for dh = B^T @ dOutput -/
def executeLoRABackward (device : Device) (weight : Hesper.LoRA.Weight)
    (grad : Hesper.LoRA.WeightGrad) (scale : Float)
    (dOutputBuf savedX savedH dInputBuf dhBuf : Buffer) : IO Unit := do
  -- Step 1: dB += scale * outer(dOutput, h)
  executeGradB device dOutputBuf savedH grad.dB weight.outDim weight.rank scale
  -- Step 2: dh = B^T @ dOutput
  executeGradDh device weight.b dOutputBuf dhBuf weight.outDim weight.rank
  -- Step 3: dA += scale * outer(dh, x)
  executeGradA device dhBuf savedX grad.dA weight.rank weight.inDim scale
  -- Step 4: dInput += scale * A^T @ dh
  executeInputGrad device weight.a dhBuf dInputBuf weight.rank weight.inDim scale

end Hesper.LoRA.Backward
