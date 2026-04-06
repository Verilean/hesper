import Hesper.LoRA.Types
import Hesper.WGSL.Monad
import Hesper.WGSL.Execute
import Hesper.WGSL.Exp
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer

/-!
# GPU-Accelerated Adam Optimizer

Implements the Adam optimizer (Kingma & Ba, 2014) as a GPU compute kernel
for efficient parameter updates on LoRA weights.

```
m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)
θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)
```

All updates happen in-place on GPU buffers (param, m, v, grad).

## Reference
CPU implementation: `Hesper/Optimizer/Adam.lean`
-/

namespace Hesper.Optimizer.AdamGPU

open Hesper.WGSL
open Hesper.WGSL.Monad
open Hesper.WebGPU

/-- AdamW hyperparameters (matches PyTorch defaults) -/
structure Config where
  lr : Float := 2e-4
  beta1 : Float := 0.9
  beta2 : Float := 0.999
  eps : Float := 1e-7      -- 1e-7 for FP32 stability (PyTorch uses 1e-8 for FP64)
  weightDecay : Float := 0.01  -- Decoupled weight decay (AdamW)
  deriving Repr

/-- GPU kernel: Adam parameter update.

    For each element i:
      m[i] = beta1 * m[i] + (1 - beta1) * grad[i]
      v[i] = beta2 * v[i] + (1 - beta2) * grad[i]^2
      m_hat = m[i] / (1 - beta1^step)
      v_hat = v[i] / (1 - beta2^step)
      param[i] -= lr * m_hat / (sqrt(v_hat) + eps)
      grad[i] = 0  (zero gradient for next step)

    Buffers: param, grad, m, v (all read-write, [numElements] FP32) -/
def adamUpdateKernel (numElements : Nat) (lr beta1 beta2 eps weightDecay : Float)
    (biasCorrection1 biasCorrection2 : Float) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let i := Exp.vec3X gid

  let _param ← ShaderM.declareOutputBuffer "param" (.array (.scalar .f32) numElements)
  let _grad ← ShaderM.declareOutputBuffer "grad" (.array (.scalar .f32) numElements)
  let _m ← ShaderM.declareOutputBuffer "m" (.array (.scalar .f32) numElements)
  let _v ← ShaderM.declareOutputBuffer "v" (.array (.scalar .f32) numElements)

  let inBounds := Exp.lt i (Exp.litU32 numElements)

  ShaderM.if_ inBounds (do
    let paramVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numElements) "param" i
    let gradVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numElements) "grad" i
    let mVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numElements) "m" i
    let vVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := numElements) "v" i

    -- AdamW: decoupled weight decay FIRST (before moment updates)
    let paramDecayed := Exp.sub paramVal (Exp.mul (Exp.litF32 (lr * weightDecay)) paramVal)

    -- Update first moment: m = beta1 * m + (1 - beta1) * grad
    let newM := Exp.add
      (Exp.mul (Exp.litF32 beta1) mVal)
      (Exp.mul (Exp.litF32 (1.0 - beta1)) gradVal)

    -- Update second moment: v = beta2 * v + (1 - beta2) * grad^2
    let newV := Exp.add
      (Exp.mul (Exp.litF32 beta2) vVal)
      (Exp.mul (Exp.litF32 (1.0 - beta2)) (Exp.mul gradVal gradVal))

    -- Bias-corrected estimates
    let mHat := Exp.div newM (Exp.litF32 biasCorrection1)
    let vHat := Exp.div newV (Exp.litF32 biasCorrection2)

    -- Update parameter: param -= lr * mHat / (sqrt(max(vHat, 0)) + eps)
    let update := Exp.div
      (Exp.mul (Exp.litF32 lr) mHat)
      (Exp.add (Exp.sqrt (Exp.max vHat (Exp.litF32 0.0))) (Exp.litF32 eps))
    let newParam := Exp.sub paramDecayed update

    -- Write back
    ShaderM.writeBuffer (ty := .scalar .f32) "param" i newParam
    ShaderM.writeBuffer (ty := .scalar .f32) "m" i newM
    ShaderM.writeBuffer (ty := .scalar .f32) "v" i newV
    -- Zero gradient for next step
    ShaderM.writeBuffer (ty := .scalar .f32) "grad" i (Exp.litF32 0.0)
  ) (pure ())

/-- Execute Adam update on a single parameter buffer -/
def executeAdamUpdate (device : Device) (paramBuf gradBuf mBuf vBuf : Buffer)
    (numElements : Nat) (config : Config) (step : Nat) : IO Unit := do
  -- Compute bias correction terms: (1 - beta^step)
  let biasCorrection1 := 1.0 - Float.pow config.beta1 step.toFloat
  let biasCorrection2 := 1.0 - Float.pow config.beta2 step.toFloat

  let shader := adamUpdateKernel numElements config.lr config.beta1 config.beta2 config.eps config.weightDecay
    biasCorrection1 biasCorrection2
  let namedBuffers := [("param", paramBuf), ("grad", gradBuf), ("m", mBuf), ("v", vBuf)]
  let execConfig := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D numElements 256
  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig

/-- Execute Adam update on all LoRA parameters in the adapter -/
def updateLoRAAdapter (device : Device) (adapter : Hesper.LoRA.Adapter)
    (grads : Hesper.LoRA.AdapterGrad)
    (adamState : Hesper.LoRA.AdapterAdamState)
    (config : Config) : IO Hesper.LoRA.AdapterAdamState := do
  let step := adamState.step + 1

  for i in [:adapter.layers.size] do
    if h1 : i < adapter.layers.size then
      if h2 : i < grads.layers.size then
        if h3 : i < adamState.layers.size then
          let layer := adapter.layers[i]
          let grad := grads.layers[i]
          let state := adamState.layers[i]

          -- Update Q projection LoRA weights
          let numA_Q := layer.loraQ.rank * layer.loraQ.inDim
          let numB_Q := layer.loraQ.outDim * layer.loraQ.rank
          executeAdamUpdate device layer.loraQ.a grad.gradQ.dA state.stateQ.mA state.stateQ.vA numA_Q config step
          executeAdamUpdate device layer.loraQ.b grad.gradQ.dB state.stateQ.mB state.stateQ.vB numB_Q config step

          -- Update V projection LoRA weights
          let numA_V := layer.loraV.rank * layer.loraV.inDim
          let numB_V := layer.loraV.outDim * layer.loraV.rank
          executeAdamUpdate device layer.loraV.a grad.gradV.dA state.stateV.mA state.stateV.vA numA_V config step
          executeAdamUpdate device layer.loraV.b grad.gradV.dB state.stateV.mB state.stateV.vB numB_V config step

  pure { adamState with step }

end Hesper.Optimizer.AdamGPU
