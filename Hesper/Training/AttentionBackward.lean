import Hesper.WGSL.Monad
import Hesper.WGSL.Execute
import Hesper.WGSL.Exp
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer

/-!
# Attention Backward GPU Kernels

GPU kernels implementing the backward pass through attention for LoRA training.
Each kernel corresponds to a verified CPU spec in `VerifiedBackward.lean`.

## Gradient Flow (reverse order of forward)

```
dOutput [dim]
  ↓ O projection backward (BitLinear transpose)
dAttnOut [dim]
  ↓ RMSNorm backward (sub-norm)
dAttnWeighted [numHeads * headDim]
  ↓ Attention apply backward
dAttn [numHeads * cacheLen] + dV [kvDim] (not needed for LoRA Q)
  ↓ Softmax backward
dScores [numHeads * cacheLen]
  ↓ Score backward (Q @ K^T)
dQ [numHeads * headDim]
  ↓ RoPE backward (inverse rotation)
dQpre [numHeads * headDim]  ← This is ∂L/∂(BitLinear_Q output) = LoRA Q gradient signal
```
-/

namespace Hesper.Training.AttentionBackward

open Hesper.WGSL
open Hesper.WGSL.Monad
open Hesper.WebGPU

/-! ## Softmax Backward -/

/-- Softmax backward kernel:
    dScores[h, s] = attn[h, s] * (dAttn[h, s] - Σ_s' attn[h, s'] * dAttn[h, s'])

    One thread per (head, seq_pos) pair.
    Uses shared memory for the dot product reduction per head. -/
def softmaxBackwardKernel (numHeads cacheLen : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid  -- linear index into [numHeads * cacheLen]

  let _attn ← ShaderM.declareInputBuffer "attn" (.array (.scalar .f32) (numHeads * cacheLen))
  let _dAttn ← ShaderM.declareInputBuffer "dAttn" (.array (.scalar .f32) (numHeads * cacheLen))
  let _dScores ← ShaderM.declareOutputBuffer "dScores" (.array (.scalar .f32) (numHeads * cacheLen))

  let total := numHeads * cacheLen
  ShaderM.if_ (Exp.lt idx (Exp.litU32 total)) (do
    let head := Exp.div idx (Exp.litU32 cacheLen)
    let _s := Exp.mod idx (Exp.litU32 cacheLen)

    -- Compute dot = Σ_s' attn[h, s'] * dAttn[h, s']
    let dotVar ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 cacheLen) (Exp.litU32 1) fun s' => do
      let aIdx := Exp.add (Exp.mul head (Exp.litU32 cacheLen)) s'
      let aVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := total) "attn" aIdx
      let dVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := total) "dAttn" aIdx
      ShaderM.assign dotVar (Exp.add (Exp.var dotVar) (Exp.mul aVal dVal))

    let attnVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := total) "attn" idx
    let dAttnVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := total) "dAttn" idx
    -- dScores[idx] = attn[idx] * (dAttn[idx] - dot)
    let result := Exp.mul attnVal (Exp.sub dAttnVal (Exp.var dotVar))
    ShaderM.writeBuffer (ty := .scalar .f32) "dScores" idx result
  ) (pure ())

def executeSoftmaxBackward (device : Device) (attnBuf dAttnBuf dScoresBuf : Buffer)
    (numHeads cacheLen : Nat) : IO Unit := do
  let shader := softmaxBackwardKernel numHeads cacheLen
  let namedBuffers := [("attn", attnBuf), ("dAttn", dAttnBuf), ("dScores", dScoresBuf)]
  let execConfig := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D (numHeads * cacheLen) 256
  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig

/-! ## Attention Score Backward (dQ from dScores) -/

/-- Score backward kernel for Q:
    dQ[h, d] = scale * Σ_s dScores[h, s] * K_cache[kvHead(h), s, d]

    One thread per (head, dim) pair.
    GQA: multiple heads map to the same KV head. -/
def scoreBackwardQKernel (numHeads numKVHeads cacheLen headDim : Nat) (scale : Float) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid  -- linear index into [numHeads * headDim]

  let _dScores ← ShaderM.declareInputBuffer "dScores" (.array (.scalar .f32) (numHeads * cacheLen))
  let _kCache ← ShaderM.declareInputBuffer "kCache" (.array (.scalar .f32) (numKVHeads * cacheLen * headDim))
  let _dQ ← ShaderM.declareOutputBuffer "dQ" (.array (.scalar .f32) (numHeads * headDim))

  let total := numHeads * headDim
  ShaderM.if_ (Exp.lt idx (Exp.litU32 total)) (do
    let head := Exp.div idx (Exp.litU32 headDim)
    let d := Exp.mod idx (Exp.litU32 headDim)
    -- GQA mapping: kvHead = head / headsPerKVHead
    let headsPerKV := numHeads / numKVHeads
    let kvHead := Exp.div head (Exp.litU32 headsPerKV)

    let accVar ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 cacheLen) (Exp.litU32 1) fun s => do
      let dsIdx := Exp.add (Exp.mul head (Exp.litU32 cacheLen)) s
      let dsVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numHeads * cacheLen) "dScores" dsIdx
      -- K_cache[kvHead, s, d] at linear index: kvHead * cacheLen * headDim + s * headDim + d
      let kIdx := Exp.add (Exp.add (Exp.mul (Exp.mul kvHead (Exp.litU32 cacheLen)) (Exp.litU32 headDim))
                                    (Exp.mul s (Exp.litU32 headDim))) d
      let kVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numKVHeads * cacheLen * headDim) "kCache" kIdx
      ShaderM.assign accVar (Exp.add (Exp.var accVar) (Exp.mul dsVal kVal))

    ShaderM.writeBuffer (ty := .scalar .f32) "dQ" idx (Exp.mul (Exp.litF32 scale) (Exp.var accVar))
  ) (pure ())

def executeScoreBackwardQ (device : Device) (dScoresBuf kCacheBuf dQBuf : Buffer)
    (numHeads numKVHeads cacheLen headDim : Nat) (scale : Float) : IO Unit := do
  let shader := scoreBackwardQKernel numHeads numKVHeads cacheLen headDim scale
  let namedBuffers := [("dScores", dScoresBuf), ("kCache", kCacheBuf), ("dQ", dQBuf)]
  let execConfig := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D (numHeads * headDim) 256
  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig

/-! ## Attention Apply Backward (dAttn from dOutput @ V^T) -/

/-- Attention apply backward kernel:
    dAttn[h, s] = Σ_d dOutput[h, d] * V_cache[kvHead(h), s, d]

    One thread per (head, seq_pos) pair. -/
def applyBackwardKernel (numHeads numKVHeads cacheLen headDim : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid  -- [numHeads * cacheLen]

  let _dOutput ← ShaderM.declareInputBuffer "dOutput" (.array (.scalar .f32) (numHeads * headDim))
  let _vCache ← ShaderM.declareInputBuffer "vCache" (.array (.scalar .f32) (numKVHeads * cacheLen * headDim))
  let _dAttn ← ShaderM.declareOutputBuffer "dAttn" (.array (.scalar .f32) (numHeads * cacheLen))

  let total := numHeads * cacheLen
  ShaderM.if_ (Exp.lt idx (Exp.litU32 total)) (do
    let head := Exp.div idx (Exp.litU32 cacheLen)
    let s := Exp.mod idx (Exp.litU32 cacheLen)
    let headsPerKV := numHeads / numKVHeads
    let kvHead := Exp.div head (Exp.litU32 headsPerKV)

    let accVar ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 headDim) (Exp.litU32 1) fun d => do
      let dOutIdx := Exp.add (Exp.mul head (Exp.litU32 headDim)) d
      let dOutVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numHeads * headDim) "dOutput" dOutIdx
      let vIdx := Exp.add (Exp.add (Exp.mul (Exp.mul kvHead (Exp.litU32 cacheLen)) (Exp.litU32 headDim))
                                    (Exp.mul s (Exp.litU32 headDim))) d
      let vVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numKVHeads * cacheLen * headDim) "vCache" vIdx
      ShaderM.assign accVar (Exp.add (Exp.var accVar) (Exp.mul dOutVal vVal))

    ShaderM.writeBuffer (ty := .scalar .f32) "dAttn" idx (Exp.var accVar)
  ) (pure ())

def executeApplyBackward (device : Device) (dOutputBuf vCacheBuf dAttnBuf : Buffer)
    (numHeads numKVHeads cacheLen headDim : Nat) : IO Unit := do
  let shader := applyBackwardKernel numHeads numKVHeads cacheLen headDim
  let namedBuffers := [("dOutput", dOutputBuf), ("vCache", vCacheBuf), ("dAttn", dAttnBuf)]
  let execConfig := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D (numHeads * cacheLen) 256
  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig

/-! ## RoPE Backward (inverse rotation) -/

/-- RoPE backward kernel: apply inverse rotation R(-θ) to gradient.
    For NeoX split-half layout:
    dx[h, d]         = dy[h, d] * cos(θ) + dy[h, d+half] * sin(θ)
    dx[h, d+half]    = -dy[h, d] * sin(θ) + dy[h, d+half] * cos(θ)

    where θ = pos * base^(-2d/headDim), same as forward. -/
def ropeBackwardKernel (numHeads headDim : Nat) (ropeBase : Float) (pos : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid  -- [numHeads * headDim/2] (one thread per dimension pair)

  let _dOut ← ShaderM.declareInputBuffer "dOut" (.array (.scalar .f32) (numHeads * headDim))
  let _dIn ← ShaderM.declareOutputBuffer "dIn" (.array (.scalar .f32) (numHeads * headDim))

  let halfDim := headDim / 2
  let total := numHeads * halfDim
  ShaderM.if_ (Exp.lt idx (Exp.litU32 total)) (do
    let head := Exp.div idx (Exp.litU32 halfDim)
    let d := Exp.mod idx (Exp.litU32 halfDim)
    let baseOffset := Exp.mul head (Exp.litU32 headDim)

    -- Compute theta = pos * base^(-2d/headDim)
    -- We use the same formula as forward RoPE
    -- For GPU: theta = pos * exp(-2d/headDim * log(base))
    let dFloat := Exp.toF32 d
    let logBase := Exp.litF32 (Float.log ropeBase)
    let exponent := Exp.mul (Exp.litF32 (-2.0 / headDim.toFloat)) (Exp.mul dFloat logBase)
    let freqScale := Exp.exp exponent
    let theta := Exp.mul (Exp.litF32 pos.toFloat) freqScale

    let cosTheta := Exp.cos theta
    let sinTheta := Exp.sin theta

    -- Read dOut pair
    let idx0 := Exp.add baseOffset d
    let idx1 := Exp.add baseOffset (Exp.add d (Exp.litU32 halfDim))
    let dy0 ← ShaderM.readBuffer (ty := .scalar .f32) (n := numHeads * headDim) "dOut" idx0
    let dy1 ← ShaderM.readBuffer (ty := .scalar .f32) (n := numHeads * headDim) "dOut" idx1

    -- R(-θ)ᵀ @ [dy0, dy1] = [dy0*cos + dy1*sin, -dy0*sin + dy1*cos]
    let dx0 := Exp.add (Exp.mul dy0 cosTheta) (Exp.mul dy1 sinTheta)
    let dx1 := Exp.add (Exp.mul (Exp.litF32 (-1.0)) (Exp.mul dy0 sinTheta)) (Exp.mul dy1 cosTheta)

    ShaderM.writeBuffer (ty := .scalar .f32) "dIn" idx0 dx0
    ShaderM.writeBuffer (ty := .scalar .f32) "dIn" idx1 dx1
  ) (pure ())

def executeRopeBackward (device : Device) (dOutBuf dInBuf : Buffer)
    (numHeads headDim : Nat) (ropeBase : Float) (pos : Nat) : IO Unit := do
  let shader := ropeBackwardKernel numHeads headDim ropeBase pos
  let namedBuffers := [("dOut", dOutBuf), ("dIn", dInBuf)]
  let execConfig := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D (numHeads * headDim / 2) 256
  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig

/-! ## RMSNorm Backward -/

/-- RMSNorm backward kernel (single workgroup, shared memory reduction):
    dx[i] = (1/rms) * (dy[i]*γ[i] - x[i] * dot(dy*γ, x) / (n * rms²))

    Uses the same workgroup reduction pattern as forward RMSNorm. -/
def rmsNormBackwardKernel (dim : Nat) (eps : Float) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let lid ← ShaderM.localId
  let tid := Exp.vec3X lid

  let _x ← ShaderM.declareInputBuffer "x" (.array (.scalar .f32) dim)
  let _gamma ← ShaderM.declareInputBuffer "gamma" (.array (.scalar .f32) dim)
  let _dOut ← ShaderM.declareInputBuffer "dOut" (.array (.scalar .f32) dim)
  let _dIn ← ShaderM.declareOutputBuffer "dIn" (.array (.scalar .f32) dim)

  ShaderM.sharedNamed "shared_sum" (.array (.scalar .f32) workgroupSize)

  -- Phase 1: Compute sum(x²) via parallel reduction
  let sqSumVar ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
  ShaderM.loop tid (Exp.litU32 dim) (Exp.litU32 workgroupSize) fun i => do
    let xi ← ShaderM.readBuffer (ty := .scalar .f32) (n := dim) "x" i
    ShaderM.assign sqSumVar (Exp.add (Exp.var sqSumVar) (Exp.mul xi xi))
  ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_sum" tid (Exp.var sqSumVar)
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

  -- Save sumSq to a local variable BEFORE Phase 2 overwrites shared_sum
  let sumSqFromShared ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_sum" (Exp.litU32 0)
  let sumSqVar ← ShaderM.var (.scalar .f32) sumSqFromShared
  let sumSq := Exp.var sumSqVar
  let rms2 := Exp.add (Exp.div sumSq (Exp.litF32 dim.toFloat)) (Exp.litF32 eps)
  let rms := Exp.sqrt rms2

  -- Phase 2: Compute dot = Σ(dy*γ*x) via parallel reduction
  let dotVar ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
  ShaderM.loop tid (Exp.litU32 dim) (Exp.litU32 workgroupSize) fun i => do
    let xi ← ShaderM.readBuffer (ty := .scalar .f32) (n := dim) "x" i
    let gi ← ShaderM.readBuffer (ty := .scalar .f32) (n := dim) "gamma" i
    let di ← ShaderM.readBuffer (ty := .scalar .f32) (n := dim) "dOut" i
    ShaderM.assign dotVar (Exp.add (Exp.var dotVar) (Exp.mul (Exp.mul di gi) xi))
  ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_sum" tid (Exp.var dotVar)
  ShaderM.barrier

  ShaderM.staticLoop numSteps fun step => do
    let s := workgroupSize >>> (step + 1)
    ShaderM.if_ (Exp.lt tid (Exp.litU32 s)) (do
      let other ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_sum" (Exp.add tid (Exp.litU32 s))
      let cur ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_sum" tid
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_sum" tid (Exp.add cur other)
    ) (pure ())
    ShaderM.barrier

  let dot ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_sum" (Exp.litU32 0)

  -- Phase 3: Compute dx[i] = (1/rms) * (dy[i]*γ[i] - x[i] * dot / (n * rms²))
  ShaderM.loop tid (Exp.litU32 dim) (Exp.litU32 workgroupSize) fun i => do
    let xi ← ShaderM.readBuffer (ty := .scalar .f32) (n := dim) "x" i
    let gi ← ShaderM.readBuffer (ty := .scalar .f32) (n := dim) "gamma" i
    let di ← ShaderM.readBuffer (ty := .scalar .f32) (n := dim) "dOut" i
    let dyGamma := Exp.mul di gi
    let correction := Exp.div (Exp.mul xi dot) (Exp.mul (Exp.litF32 dim.toFloat) rms2)
    let result := Exp.mul (Exp.div (Exp.litF32 1.0) rms) (Exp.sub dyGamma correction)
    ShaderM.writeBuffer (ty := .scalar .f32) "dIn" i result

def executeRmsNormBackward (device : Device) (xBuf gammaBuf dOutBuf dInBuf : Buffer)
    (dim : Nat) (eps : Float := 1e-6) : IO Unit := do
  let workgroupSize := 256
  let shader := rmsNormBackwardKernel dim eps workgroupSize
  let namedBuffers := [("x", xBuf), ("gamma", gammaBuf), ("dOut", dOutBuf), ("dIn", dInBuf)]
  let execConfig : Hesper.WGSL.Execute.ExecutionConfig := {
    workgroupSize := {x := workgroupSize, y := 1, z := 1}
    numWorkgroups := (1, 1, 1)
  }
  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig

end Hesper.Training.AttentionBackward
