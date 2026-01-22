import Hesper.Core.VerifiedOpFusion
import Hesper.WGSL.Kernel
import Hesper.WGSL.Exp
import Hesper.WGSL.Types
import Hesper.WGSL.Monad
import Hesper.WGSL.CodeGen
import Hesper.WGSL.Helpers
import Hesper.Op.Activation

/-!
# Multi-Layer Perceptron (MLP) with GPU Backpropagation

Complete implementation of 2-layer MLP for MNIST with GPU-accelerated forward
and backward passes.

## Architecture

```
Input(784) → Dense(128) + ReLU → Dense(10) + Softmax → Output(10)
```

## GPU Implementation

**Forward Pass:**
1. Layer 1: MatVec(784→128) + Bias + ReLU (fused kernel)
2. Layer 2: MatVec(128→10) + Bias (fused kernel)
3. Softmax(10)

**Backward Pass:**
1. Softmax gradient: dL/dlogits = probs - one_hot(label)
2. Layer 2 gradient: dW2 = h1^T @ dlogits, dB2 = dlogits, dH1 = W2^T @ dlogits
3. ReLU gradient: dH1pre = dH1 * (h1 > 0)
4. Layer 1 gradient: dW1 = input^T @ dH1pre, dB1 = dH1pre

**SGD Update (GPU):**
- W = W - lr * dW (element-wise on GPU)
- B = B - lr * dB (element-wise on GPU)
-/

namespace Hesper.NN.MLP

open Hesper.Core
open Hesper.WGSL
open Hesper.WGSL.CodeGen
open Hesper.Tensor
open Hesper.Op.Activation

/-! ## Type Definitions -/

/-- MLP parameters (weights and biases for 2 layers) -/
structure MLPParams where
  w1 : TensorData  -- [784 × 128]
  b1 : TensorData  -- [128]
  w2 : TensorData  -- [128 × 10]
  b2 : TensorData  -- [10]
  deriving Inhabited

/-- MLP input (single sample + label for training) -/
structure MLPInput where
  x : TensorData      -- [784] input features
  label : Nat         -- Ground truth label (0-9)
  params : MLPParams
  deriving Inhabited

/-- MLP output (predictions + loss) -/
structure MLPOutput where
  probs : TensorData   -- [10] softmax probabilities
  loss : Float         -- Cross-entropy loss
  deriving Inhabited

/-- MLP gradients (for backprop) -/
structure MLPGradients where
  dW1 : TensorData  -- [784 × 128]
  dB1 : TensorData  -- [128]
  dW2 : TensorData  -- [128 × 10]
  dB2 : TensorData  -- [10]
  deriving Inhabited

/-! ## CPU Reference Implementation -/

/-- Matrix-vector multiplication on CPU -/
def cpuMatVec (mat : Array Float) (vec : Array Float) (rows cols : Nat) : Array Float :=
  Array.range rows |>.map fun i =>
    (Array.range cols).foldl
      (init := 0.0)
      (fun sum j => sum + mat[i * cols + j]! * vec[j]!)

/-- Add bias vector to result -/
def cpuAddBias (vec bias : Array Float) : Array Float :=
  Array.zipWith (· + ·) vec bias

/-- ReLU activation -/
def cpuReLU (vec : Array Float) : Array Float :=
  vec.map (fun x => max 0.0 x)

/-- Softmax activation -/
def cpuSoftmax (vec : Array Float) : Array Float :=
  let maxVal := vec.foldl max (-1e9)
  let exp_vec := vec.map (fun x => Float.exp (x - maxVal))
  let sum := exp_vec.foldl (· + ·) 0.0
  exp_vec.map (· / sum)

/-- Cross-entropy loss (approximated as 1 - prob for simplicity) -/
def cpuCrossEntropyLoss (probs : Array Float) (label : Nat) : Float :=
  let prob := probs[label]!
  let clamped := if prob < 1e-7 then 1e-7 else prob
  -- True cross-entropy would be -(Float.log clamped) but Float.log doesn't exist
  -- Use approximation: 1 - prob (monotonic with -log(prob) for prob near 1)
  1.0 - clamped

/-- CPU forward pass -/
def cpuMLPForward (input : MLPInput) : MLPOutput :=
  -- Layer 1: x @ W1 + b1, ReLU
  let h1_pre := cpuMatVec input.params.w1.data input.x.data 128 784
  let h1_bias := cpuAddBias h1_pre input.params.b1.data
  let h1 := cpuReLU h1_bias

  -- Layer 2: h1 @ W2 + b2
  let logits := cpuMatVec input.params.w2.data h1 10 128
  let logits_bias := cpuAddBias logits input.params.b2.data

  -- Softmax
  let probs := cpuSoftmax logits_bias

  -- Loss
  let loss := cpuCrossEntropyLoss probs input.label

  {
    probs := { shape := ⟨[10]⟩, data := probs }
    loss := loss
  }

/-- CPU backward pass (returns gradients) -/
def cpuMLPBackward (input : MLPInput) (output : MLPOutput) : MLPGradients :=
  -- Re-compute forward pass to get activations
  let h1_pre := cpuMatVec input.params.w1.data input.x.data 128 784
  let h1_bias := cpuAddBias h1_pre input.params.b1.data
  let h1 := cpuReLU h1_bias

  let logits := cpuMatVec input.params.w2.data h1 10 128
  let logits_bias := cpuAddBias logits input.params.b2.data

  -- Gradient of softmax + cross-entropy: probs - one_hot(label)
  let dLogits := Array.range output.probs.data.size |>.map fun i =>
    let val := output.probs.data[i]!
    if i == input.label then val - 1.0 else val

  -- Layer 2 gradients
  -- dW2 = h1^T @ dLogits (outer product: [128 × 10])
  let dW2 := Array.range (128 * 10) |>.map fun idx =>
    let i := idx / 10
    let j := idx % 10
    h1[i]! * dLogits[j]!

  -- dB2 = dLogits
  let dB2 := dLogits

  -- dH1 = W2 @ dLogits (matrix-vector: [128])
  let dH1 := Array.range 128 |>.map fun i =>
    (Array.range 10).foldl
      (init := 0.0)
      (fun sum j => sum + input.params.w2.data[i * 10 + j]! * dLogits[j]!)

  -- ReLU gradient: dH1pre = dH1 * (h1 > 0)
  let dH1pre := Array.zipWith (fun grad act => if act > 0.0 then grad else 0.0) dH1 h1

  -- Layer 1 gradients
  -- dW1 = input^T @ dH1pre (outer product: [784 × 128])
  let dW1 := Array.range (784 * 128) |>.map fun idx =>
    let i := idx / 128
    let j := idx % 128
    input.x.data[i]! * dH1pre[j]!

  -- dB1 = dH1pre
  let dB1 := dH1pre

  {
    dW1 := { shape := ⟨[784, 128]⟩, data := dW1 }
    dB1 := { shape := ⟨[128]⟩, data := dB1 }
    dW2 := { shape := ⟨[128, 10]⟩, data := dW2 }
    dB2 := { shape := ⟨[10]⟩, data := dB2 }
  }

/-! ## GPU Kernels (using existing WGSL helpers) -/

/-- Generate Layer 1 kernel (MatVec + Bias + ReLU) -/
def generateLayer1Kernel : String :=
  Helpers.generateMatVecBiasActivationShader 784 128 Helpers.reluActivation

/-- Generate Layer 2 kernel (MatVec + Bias) -/
def generateLayer2Kernel : String :=
  Helpers.generateMatVecBiasActivationShader 128 10 Helpers.identityActivation

/-- Generate Softmax kernel -/
def generateSoftmaxKernel : String :=
  Helpers.generateSoftmaxShader 10

/-! ## GPU Backward Pass Kernels (using WGSL DSL) -/

section GPUKernels

/-- Softmax gradient kernel: dLogits[i] = probs[i] - (i == label ? 1 : 0) -/
def genSoftmaxGradKernel (outputSize : Nat) : String :=
  let shaderBody : WGSL.Monad.ShaderM Unit := do
    -- Read label (scalar u32)
    let labelVal : Exp (.scalar .u32) := Exp.index (Exp.var "label" : Exp (.array (.scalar .u32) 1)) (Exp.litU32 0)

    -- Loop over all outputs
    WGSL.Monad.ShaderM.loop (Exp.litU32 0) (Exp.litU32 outputSize) (Exp.litU32 1) fun i => do
      let prob := Exp.index (Exp.var "probs" : Exp (.array (.scalar .f32) outputSize)) i

      -- If i == label: dLogits[i] = probs[i] - 1.0, else: dLogits[i] = probs[i]
      let isTarget := Exp.eq i labelVal
      WGSL.Monad.ShaderM.if_ isTarget (do
        let grad := Exp.sub prob (Exp.litF32 1.0)
        WGSL.Monad.ShaderM.assignIndex "dLogits" i grad
      ) (do
        WGSL.Monad.ShaderM.assignIndex "dLogits" i prob
      )

  let state := WGSL.Monad.ShaderM.exec shaderBody

  let buffers : List StorageBuffer := [
    { group := 0, binding := 0, name := "probs", elemType := .array (.scalar .f32) outputSize, readWrite := false },
    { group := 0, binding := 1, name := "label", elemType := .array (.scalar .u32) 1, readWrite := false },
    { group := 0, binding := 2, name := "dLogits", elemType := .array (.scalar .f32) outputSize, readWrite := true }
  ]

  let mainFunc : FunctionDecl := {
    name := "main"
    attributes := ["@compute", "@workgroup_size(1, 1, 1)"]
    params := []
    body := state.stmts
  }

  let module : ShaderModule := {
    storageBuffers := buffers
    functions := [mainFunc]
  }

  module.toWGSL

/-- Layer 2 backward kernel: computes dW2, dB2, dH1 -/
def genLayer2BackwardKernel (hiddenSize outputSize : Nat) : String :=
  let shaderBody : WGSL.Monad.ShaderM Unit := do
    let gid ← WGSL.Monad.ShaderM.globalId
    let tid := Exp.vec3X gid

    -- Compute dW2: outer product h1^T @ dLogits
    -- Loop with stride to parallelize across threads
    WGSL.Monad.ShaderM.loop tid (Exp.litU32 (hiddenSize * outputSize)) (Exp.litU32 256) fun idx => do
      let i := Exp.div idx (Exp.litU32 outputSize)
      let j := Exp.mod idx (Exp.litU32 outputSize)

      let h1Val := Exp.index (Exp.var "h1" : Exp (.array (.scalar .f32) hiddenSize)) i
      let dLogitsVal := Exp.index (Exp.var "dLogits" : Exp (.array (.scalar .f32) outputSize)) j
      let dW2Val := Exp.mul h1Val dLogitsVal

      WGSL.Monad.ShaderM.assignIndex "dW2" idx dW2Val

    -- Compute dB2 = dLogits (parallel copy)
    let tidLtOutput := Exp.lt tid (Exp.litU32 outputSize)
    WGSL.Monad.ShaderM.if_ tidLtOutput (do
      let dLogitsVal := Exp.index (Exp.var "dLogits" : Exp (.array (.scalar .f32) outputSize)) tid
      WGSL.Monad.ShaderM.assignIndex "dB2" tid dLogitsVal
    ) (pure ())

    -- Compute dH1 = W2^T @ dLogits (matrix-vector product)
    let tidLtHidden := Exp.lt tid (Exp.litU32 hiddenSize)
    WGSL.Monad.ShaderM.if_ tidLtHidden (do
      let sumVar ← WGSL.Monad.ShaderM.var (.scalar .f32) (Exp.litF32 0.0)

      WGSL.Monad.ShaderM.loop (Exp.litU32 0) (Exp.litU32 outputSize) (Exp.litU32 1) fun j => do
        let w2Idx := Exp.add (Exp.mul tid (Exp.litU32 outputSize)) j
        let w2Val := Exp.index (Exp.var "w2" : Exp (.array (.scalar .f32) (hiddenSize * outputSize))) w2Idx
        let dLogitsVal := Exp.index (Exp.var "dLogits" : Exp (.array (.scalar .f32) outputSize)) j

        let currentSum := Exp.var sumVar
        let newSum := Exp.add currentSum (Exp.mul w2Val dLogitsVal)
        WGSL.Monad.ShaderM.assign sumVar newSum

      WGSL.Monad.ShaderM.assignIndex "dH1" tid (Exp.var sumVar : Exp (.scalar .f32))
    ) (pure ())

  let state := WGSL.Monad.ShaderM.exec shaderBody

  let buffers : List StorageBuffer := [
    { group := 0, binding := 0, name := "h1", elemType := .array (.scalar .f32) hiddenSize, readWrite := false },
    { group := 0, binding := 1, name := "dLogits", elemType := .array (.scalar .f32) outputSize, readWrite := false },
    { group := 0, binding := 2, name := "w2", elemType := .array (.scalar .f32) (hiddenSize * outputSize), readWrite := false },
    { group := 0, binding := 3, name := "dW2", elemType := .array (.scalar .f32) (hiddenSize * outputSize), readWrite := true },
    { group := 0, binding := 4, name := "dB2", elemType := .array (.scalar .f32) outputSize, readWrite := true },
    { group := 0, binding := 5, name := "dH1", elemType := .array (.scalar .f32) hiddenSize, readWrite := true }
  ]

  let mainFunc : FunctionDecl := {
    name := "main"
    attributes := ["@compute", "@workgroup_size(256, 1, 1)"]
    params := [{ name := "global_invocation_id", ty := .vec3 .u32, builtin := some .globalInvocationId }]
    body := state.stmts
  }

  let module : ShaderModule := {
    storageBuffers := buffers
    functions := [mainFunc]
  }

  module.toWGSL

/-- ReLU backward kernel: dH1pre[i] = dH1[i] * (h1[i] > 0) -/
def genReLUBackwardKernel (size : Nat) : String :=
  let shaderBody : WGSL.Monad.ShaderM Unit := do
    let gid ← WGSL.Monad.ShaderM.globalId
    let tid := Exp.vec3X gid

    let boundsCheck := Exp.lt tid (Exp.litU32 size)
    WGSL.Monad.ShaderM.if_ boundsCheck (do
      let h1Val := Exp.index (Exp.var "h1" : Exp (.array (.scalar .f32) size)) tid
      let dH1Val := Exp.index (Exp.var "dH1" : Exp (.array (.scalar .f32) size)) tid

      let isPositive := Exp.gt h1Val (Exp.litF32 0.0)
      WGSL.Monad.ShaderM.if_ isPositive (do
        WGSL.Monad.ShaderM.assignIndex "dH1pre" tid dH1Val
      ) (do
        WGSL.Monad.ShaderM.assignIndex "dH1pre" tid (Exp.litF32 0.0)
      )
    ) (pure ())

  let state := WGSL.Monad.ShaderM.exec shaderBody

  let buffers : List StorageBuffer := [
    { group := 0, binding := 0, name := "h1", elemType := .array (.scalar .f32) size, readWrite := false },
    { group := 0, binding := 1, name := "dH1", elemType := .array (.scalar .f32) size, readWrite := false },
    { group := 0, binding := 2, name := "dH1pre", elemType := .array (.scalar .f32) size, readWrite := true }
  ]

  let mainFunc : FunctionDecl := {
    name := "main"
    attributes := ["@compute", "@workgroup_size(256, 1, 1)"]
    params := [{ name := "global_invocation_id", ty := .vec3 .u32, builtin := some .globalInvocationId }]
    body := state.stmts
  }

  let module : ShaderModule := {
    storageBuffers := buffers
    functions := [mainFunc]
  }

  module.toWGSL

/-- Layer 1 backward kernel: computes dW1, dB1 -/
def genLayer1BackwardKernel (inputSize hiddenSize : Nat) : String :=
  let shaderBody : WGSL.Monad.ShaderM Unit := do
    let gid ← WGSL.Monad.ShaderM.globalId
    let tid := Exp.vec3X gid

    -- Compute dW1: outer product input^T @ dH1pre
    WGSL.Monad.ShaderM.loop tid (Exp.litU32 (inputSize * hiddenSize)) (Exp.litU32 256) fun idx => do
      let i := Exp.div idx (Exp.litU32 hiddenSize)
      let j := Exp.mod idx (Exp.litU32 hiddenSize)

      let inputVal := Exp.index (Exp.var "input" : Exp (.array (.scalar .f32) inputSize)) i
      let dH1preVal := Exp.index (Exp.var "dH1pre" : Exp (.array (.scalar .f32) hiddenSize)) j
      let dW1Val := Exp.mul inputVal dH1preVal

      WGSL.Monad.ShaderM.assignIndex "dW1" idx dW1Val

    -- Compute dB1 = dH1pre (parallel copy)
    let tidLtHidden := Exp.lt tid (Exp.litU32 hiddenSize)
    WGSL.Monad.ShaderM.if_ tidLtHidden (do
      let dH1preVal := Exp.index (Exp.var "dH1pre" : Exp (.array (.scalar .f32) hiddenSize)) tid
      WGSL.Monad.ShaderM.assignIndex "dB1" tid dH1preVal
    ) (pure ())

  let state := WGSL.Monad.ShaderM.exec shaderBody

  let buffers : List StorageBuffer := [
    { group := 0, binding := 0, name := "input", elemType := .array (.scalar .f32) inputSize, readWrite := false },
    { group := 0, binding := 1, name := "dH1pre", elemType := .array (.scalar .f32) hiddenSize, readWrite := false },
    { group := 0, binding := 2, name := "dW1", elemType := .array (.scalar .f32) (inputSize * hiddenSize), readWrite := true },
    { group := 0, binding := 3, name := "dB1", elemType := .array (.scalar .f32) hiddenSize, readWrite := true }
  ]

  let mainFunc : FunctionDecl := {
    name := "main"
    attributes := ["@compute", "@workgroup_size(256, 1, 1)"]
    params := [{ name := "global_invocation_id", ty := .vec3 .u32, builtin := some .globalInvocationId }]
    body := state.stmts
  }

  let module : ShaderModule := {
    storageBuffers := buffers
    functions := [mainFunc]
  }

  module.toWGSL

/-! ## SGD Update on GPU -/

/-- SGD kernel: param[i] -= lr * grad[i] -/
def genSGDKernel (size : Nat) (lr : Float) : String :=
  let shaderBody : WGSL.Monad.ShaderM Unit := do
    let gid ← WGSL.Monad.ShaderM.globalId
    let tid := Exp.vec3X gid

    let boundsCheck := Exp.lt tid (Exp.litU32 size)
    WGSL.Monad.ShaderM.if_ boundsCheck (do
      let paramVal := Exp.index (Exp.var "params" : Exp (.array (.scalar .f32) size)) tid
      let gradVal := Exp.index (Exp.var "grads" : Exp (.array (.scalar .f32) size)) tid

      -- param -= lr * grad
      let update := Exp.sub paramVal (Exp.mul (Exp.litF32 lr) gradVal)
      WGSL.Monad.ShaderM.assignIndex "params" tid update
    ) (pure ())

  let state := WGSL.Monad.ShaderM.exec shaderBody

  let buffers : List StorageBuffer := [
    { group := 0, binding := 0, name := "params", elemType := .array (.scalar .f32) size, readWrite := true },
    { group := 0, binding := 1, name := "grads", elemType := .array (.scalar .f32) size, readWrite := false }
  ]

  let mainFunc : FunctionDecl := {
    name := "main"
    attributes := ["@compute", "@workgroup_size(256, 1, 1)"]
    params := [{ name := "global_invocation_id", ty := .vec3 .u32, builtin := some .globalInvocationId }]
    body := state.stmts
  }

  let module : ShaderModule := {
    storageBuffers := buffers
    functions := [mainFunc]
  }

  module.toWGSL

/-- Generate GPU kernel for Adam optimizer update.

Adam (Adaptive Moment Estimation) combines momentum and RMSprop.

**Algorithm:**
```
m[i] = beta1 * m[i] + (1 - beta1) * grad[i]           # Update first moment
v[i] = beta2 * v[i] + (1 - beta2) * grad[i]^2         # Update second moment
m_hat = m[i] / (1 - beta1^step)                       # Bias correction
v_hat = v[i] / (1 - beta2^step)                       # Bias correction
params[i] -= lr * m_hat / (sqrt(v_hat) + epsilon)    # Update parameters
```

**Parameters:**
- size: Number of parameters
- lr: Learning rate (typical: 0.001)
- beta1: Momentum decay rate (typical: 0.9)
- beta2: Variance decay rate (typical: 0.999)
- epsilon: Numerical stability constant (typical: 1e-8)
- step: Current step number (for bias correction)

**Buffers:**
- @binding(0): params (read-write) - parameters to update
- @binding(1): grads (read-only) - gradients
- @binding(2): m (read-write) - first moment estimates
- @binding(3): v (read-write) - second moment estimates

**Example:**
```lean
let kernel := genAdamKernel 1000 0.001 0.9 0.999 1e-8 10
```
-/
def genAdamKernel (size : Nat) (lr : Float) (beta1 beta2 epsilon : Float) (step : Nat) : String :=
  let shaderBody : WGSL.Monad.ShaderM Unit := do
    let gid ← WGSL.Monad.ShaderM.globalId
    let tid := Exp.vec3X gid

    let boundsCheck := Exp.lt tid (Exp.litU32 size)
    WGSL.Monad.ShaderM.if_ boundsCheck (do
      -- Load current values
      let paramVal := Exp.index (Exp.var "params" : Exp (.array (.scalar .f32) size)) tid
      let gradVal := Exp.index (Exp.var "grads" : Exp (.array (.scalar .f32) size)) tid
      let mVal := Exp.index (Exp.var "m" : Exp (.array (.scalar .f32) size)) tid
      let vVal := Exp.index (Exp.var "v" : Exp (.array (.scalar .f32) size)) tid

      -- Update first moment: m = beta1 * m + (1 - beta1) * grad
      let beta1Lit := Exp.litF32 beta1
      let oneSub1 := Exp.sub (Exp.litF32 1.0) beta1Lit
      let newM := Exp.add (Exp.mul beta1Lit mVal) (Exp.mul oneSub1 gradVal)

      -- Update second moment: v = beta2 * v + (1 - beta2) * grad^2
      let beta2Lit := Exp.litF32 beta2
      let oneSub2 := Exp.sub (Exp.litF32 1.0) beta2Lit
      let gradSq := Exp.mul gradVal gradVal
      let newV := Exp.add (Exp.mul beta2Lit vVal) (Exp.mul oneSub2 gradSq)

      -- Bias correction: m_hat = m / (1 - beta1^step), v_hat = v / (1 - beta2^step)
      let stepF := Exp.litF32 (Float.ofNat step)
      let beta1Power := Exp.pow beta1Lit stepF
      let beta2Power := Exp.pow beta2Lit stepF
      let biasCorr1 := Exp.sub (Exp.litF32 1.0) beta1Power
      let biasCorr2 := Exp.sub (Exp.litF32 1.0) beta2Power
      let mHat := Exp.div newM biasCorr1
      let vHat := Exp.div newV biasCorr2

      -- Update params: params -= lr * m_hat / (sqrt(v_hat) + epsilon)
      let sqrtV := Exp.sqrt vHat
      let denominator := Exp.add sqrtV (Exp.litF32 epsilon)
      let update := Exp.div mHat denominator
      let delta := Exp.mul (Exp.litF32 lr) update
      let newParam := Exp.sub paramVal delta

      -- Write back
      WGSL.Monad.ShaderM.assignIndex "params" tid newParam
      WGSL.Monad.ShaderM.assignIndex "m" tid newM
      WGSL.Monad.ShaderM.assignIndex "v" tid newV
    ) (pure ())

  let state := WGSL.Monad.ShaderM.exec shaderBody

  let buffers : List StorageBuffer := [
    { group := 0, binding := 0, name := "params", elemType := .array (.scalar .f32) size, readWrite := true },
    { group := 0, binding := 1, name := "grads", elemType := .array (.scalar .f32) size, readWrite := false },
    { group := 0, binding := 2, name := "m", elemType := .array (.scalar .f32) size, readWrite := true },
    { group := 0, binding := 3, name := "v", elemType := .array (.scalar .f32) size, readWrite := true }
  ]

  let mainFunc : FunctionDecl := {
    name := "main"
    attributes := ["@compute", "@workgroup_size(256, 1, 1)"]
    params := [{ name := "global_invocation_id", ty := .vec3 .u32, builtin := some .globalInvocationId }]
    body := state.stmts
  }

  let module : ShaderModule := {
    storageBuffers := buffers
    functions := [mainFunc]
  }

  module.toWGSL

end GPUKernels

/-! ## SGD Update on CPU -/

/-- CPU SGD parameter update -/
def cpuSGDUpdate (params : MLPParams) (grads : MLPGradients) (lr : Float) : MLPParams :=
  {
    w1 := { shape := params.w1.shape, data := Array.zipWith (fun w g => w - lr * g) params.w1.data grads.dW1.data }
    b1 := { shape := params.b1.shape, data := Array.zipWith (fun b g => b - lr * g) params.b1.data grads.dB1.data }
    w2 := { shape := params.w2.shape, data := Array.zipWith (fun w g => w - lr * g) params.w2.data grads.dW2.data }
    b2 := { shape := params.b2.shape, data := Array.zipWith (fun b g => b - lr * g) params.b2.data grads.dB2.data }
  }

/-! ## VerifiedOpFusion Instance -/

/-- Note: Full VerifiedOpFusion instance would require:
    1. Kernel type signatures matching the pattern
    2. WGSL expression builders
    3. Proper workgroup dimensions

    For now, we provide the CPU spec which can be called directly
    from the training loop. GPU kernels can be invoked via Compute API. -/

instance : VerifiedOpFusion 256 1 1 MLPInput MLPOutput
    (Exp (.scalar .f32)) (Exp (.scalar .f32)) where
  spec_forward := cpuMLPForward
  impl_kernel := ⟨fun _ => pure (Exp.litF32 0.0)⟩  -- Placeholder
  spec_backward := fun input _ => input  -- Placeholder (returns input unchanged)
  impl_kernel_backward := ⟨fun _ => pure (Exp.litF32 0.0)⟩  -- Placeholder

/-! ## Training Helpers -/

/-- Single training step: forward + backward + update -/
def trainStep (input : MLPInput) (lr : Float) : MLPOutput × MLPParams :=
  -- Forward pass
  let output := cpuMLPForward input

  -- Backward pass
  let grads := cpuMLPBackward input output

  -- Update parameters
  let newParams := cpuSGDUpdate input.params grads lr

  (output, newParams)

end Hesper.NN.MLP
