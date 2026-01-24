import Hesper.Tensor.Types
import Hesper.WGSL.Monad
import Hesper.WGSL.CodeGen

/-!
# Neural Network Activation Functions

GPU kernels for common activation functions using the ShaderM monad.
All operations are element-wise on tensors.
-/

namespace Hesper.NN.Activation

open Hesper.Tensor
open Hesper.WGSL
open Hesper.WGSL.Monad (ShaderM)
open Hesper.WGSL.Monad.ShaderM
open Hesper.WGSL.CodeGen

/-- Configuration for activation function kernels -/
structure ActivationConfig where
  /-- Number of elements to process -/
  size : Nat
  /-- Workgroup size (threads per workgroup) -/
  workgroupSize : Nat := 256
  deriving Inhabited, Repr

namespace ActivationConfig

  /-- Number of workgroups needed -/
  def numWorkgroups (config : ActivationConfig) : Nat :=
    (config.size + config.workgroupSize - 1) / config.workgroupSize

end ActivationConfig

/-- ReLU activation kernel: f(x) = max(0, x) -/
def reluKernel (config : ActivationConfig) : ShaderM Unit := do
  let inputBuf ← declareInputBuffer "input" (.scalar .f32)
  let outputBuf ← declareOutputBuffer "output" (.scalar .f32)

  let gid ← globalId
  let idx := Exp.vec3X gid

  if_ (Exp.lt idx (litU config.size)) (do
    let x ← readBuffer (ty := .scalar .f32) (n := config.size) inputBuf idx
    let result := Exp.max (litF 0.0) x
    writeBuffer (ty := .scalar .f32) outputBuf idx result
  ) (pure ())

/-- Leaky ReLU kernel: f(x) = max(alpha * x, x) -/
def leakyReluKernel (config : ActivationConfig) (alpha : Float := 0.01) : ShaderM Unit := do
  let inputBuf ← declareInputBuffer "input" (.scalar .f32)
  let outputBuf ← declareOutputBuffer "output" (.scalar .f32)

  let gid ← globalId
  let idx := Exp.vec3X gid

  if_ (Exp.lt idx (litU config.size)) (do
    let x ← readBuffer (ty := .scalar .f32) (n := config.size) inputBuf idx
    -- Leaky ReLU: alpha * x for negative values
    let alphaX := Exp.mul (litF alpha) x
    let result := Exp.select (Exp.gt x (litF 0.0)) x alphaX
    writeBuffer (ty := .scalar .f32) outputBuf idx result
  ) (pure ())

/-- GELU activation kernel (approximate) -/
def geluKernel (config : ActivationConfig) : ShaderM Unit := do
  let inputBuf ← declareInputBuffer "input" (.scalar .f32)
  let outputBuf ← declareOutputBuffer "output" (.scalar .f32)

  let gid ← globalId
  let idx := Exp.vec3X gid

  if_ (Exp.lt idx (litU config.size)) (do
    let x ← readBuffer (ty := .scalar .f32) (n := config.size) inputBuf idx
    -- GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let sqrt2OverPi := litF 0.7978845608028654
    let x3 := Exp.mul (Exp.mul x x) x  -- x^3
    let inner := Exp.mul sqrt2OverPi (Exp.add x (Exp.mul (litF 0.044715) x3))
    let tanhInner := Exp.tanh inner
    let result := Exp.mul (Exp.mul (litF 0.5) x) (Exp.add (litF 1.0) tanhInner)
    writeBuffer (ty := .scalar .f32) outputBuf idx result
  ) (pure ())

/-- Sigmoid activation kernel: f(x) = 1 / (1 + exp(-x)) -/
def sigmoidKernel (config : ActivationConfig) : ShaderM Unit := do
  let inputBuf ← declareInputBuffer "input" (.scalar .f32)
  let outputBuf ← declareOutputBuffer "output" (.scalar .f32)

  let gid ← globalId
  let idx := Exp.vec3X gid

  if_ (Exp.lt idx (litU config.size)) (do
    let x ← readBuffer (ty := .scalar .f32) (n := config.size) inputBuf idx
    -- Sigmoid: 1 / (1 + exp(-x))
    let negX := Exp.neg x
    let expNegX := Exp.exp negX
    let result := Exp.div (litF 1.0) (Exp.add (litF 1.0) expNegX)
    writeBuffer (ty := .scalar .f32) outputBuf idx result
  ) (pure ())

/-- Tanh activation kernel: f(x) = tanh(x) -/
def tanhKernel (config : ActivationConfig) : ShaderM Unit := do
  let inputBuf ← declareInputBuffer "input" (.scalar .f32)
  let outputBuf ← declareOutputBuffer "output" (.scalar .f32)

  let gid ← globalId
  let idx := Exp.vec3X gid

  if_ (Exp.lt idx (litU config.size)) (do
    let x ← readBuffer (ty := .scalar .f32) (n := config.size) inputBuf idx
    let result := Exp.tanh x
    writeBuffer (ty := .scalar .f32) outputBuf idx result
  ) (pure ())

/-- Swish/SiLU activation kernel: f(x) = x * sigmoid(x) -/
def swishKernel (config : ActivationConfig) : ShaderM Unit := do
  let inputBuf ← declareInputBuffer "input" (.scalar .f32)
  let outputBuf ← declareOutputBuffer "output" (.scalar .f32)

  let gid ← globalId
  let idx := Exp.vec3X gid

  if_ (Exp.lt idx (litU config.size)) (do
    let x ← readBuffer (ty := .scalar .f32) (n := config.size) inputBuf idx
    -- Swish/SiLU: x * sigmoid(x)
    let negX := Exp.neg x
    let expNegX := Exp.exp negX
    let sigmoid := Exp.div (litF 1.0) (Exp.add (litF 1.0) expNegX)
    let result := Exp.mul x sigmoid
    writeBuffer (ty := .scalar .f32) outputBuf idx result
  ) (pure ())

/-- ELU activation kernel: f(x) = x if x > 0 else alpha * (exp(x) - 1) -/
def eluKernel (config : ActivationConfig) (alpha : Float := 1.0) : ShaderM Unit := do
  let inputBuf ← declareInputBuffer "input" (.scalar .f32)
  let outputBuf ← declareOutputBuffer "output" (.scalar .f32)

  let gid ← globalId
  let idx := Exp.vec3X gid

  if_ (Exp.lt idx (litU config.size)) (do
    let x ← readBuffer (ty := .scalar .f32) (n := config.size) inputBuf idx
    -- ELU: x if x > 0 else alpha * (exp(x) - 1)
    let expX := Exp.exp x
    let negative := Exp.mul (litF alpha) (Exp.sub expX (litF 1.0))
    let result := Exp.select (Exp.gt x (litF 0.0)) x negative
    writeBuffer (ty := .scalar .f32) outputBuf idx result
  ) (pure ())

/-- Softplus activation kernel: f(x) = log(1 + exp(x)) -/
def softplusKernel (config : ActivationConfig) : ShaderM Unit := do
  let inputBuf ← declareInputBuffer "input" (.scalar .f32)
  let outputBuf ← declareOutputBuffer "output" (.scalar .f32)

  let gid ← globalId
  let idx := Exp.vec3X gid

  if_ (Exp.lt idx (litU config.size)) (do
    let x ← readBuffer (ty := .scalar .f32) (n := config.size) inputBuf idx
    -- Softplus: log(1 + exp(x)), numerically stable
    let expX := Exp.exp x
    let softplus := Exp.log (Exp.add (litF 1.0) expX)
    -- Numerically stable: use x when x > 20
    let result := Exp.select (Exp.gt x (litF 20.0)) x softplus
    writeBuffer (ty := .scalar .f32) outputBuf idx result
  ) (pure ())

/-- Mish activation kernel: f(x) = x * tanh(softplus(x)) -/
def mishKernel (config : ActivationConfig) : ShaderM Unit := do
  let inputBuf ← declareInputBuffer "input" (.scalar .f32)
  let outputBuf ← declareOutputBuffer "output" (.scalar .f32)

  let gid ← globalId
  let idx := Exp.vec3X gid

  if_ (Exp.lt idx (litU config.size)) (do
    let x ← readBuffer (ty := .scalar .f32) (n := config.size) inputBuf idx
    -- Mish: x * tanh(softplus(x))
    let expX := Exp.exp x
    let softplus := Exp.log (Exp.add (litF 1.0) expX)
    let stableSoftplus := Exp.select (Exp.gt x (litF 20.0)) x softplus
    let tanhSoftplus := Exp.tanh stableSoftplus
    let result := Exp.mul x tanhSoftplus
    writeBuffer (ty := .scalar .f32) outputBuf idx result
  ) (pure ())

/-- Softmax activation kernel (simplified version for demo)
    Note: This is a simplified implementation. Production softmax requires
    multi-pass reduction for finding max and sum across the entire array.
-/
def softmaxKernel (config : ActivationConfig) : ShaderM Unit := do
  let inputBuf ← declareInputBuffer "input" (.scalar .f32)
  let outputBuf ← declareOutputBuffer "output" (.scalar .f32)

  let gid ← globalId
  let idx := Exp.vec3X gid

  if_ (Exp.lt idx (litU config.size)) (do
    -- For demo: simplified softmax that assumes max has been pre-subtracted
    -- and this just does exp normalization
    let x ← readBuffer (ty := .scalar .f32) (n := config.size) inputBuf idx
    let expX := Exp.exp x
    writeBuffer (ty := .scalar .f32) outputBuf idx expX
  ) (pure ())

/-- Generate WGSL shader for ReLU -/
def generateReLUShader (config : ActivationConfig) : String :=
  generateWGSL "main" {x := config.workgroupSize, y := 1, z := 1} [] [] (reluKernel config)

/-- Generate WGSL shader for Leaky ReLU -/
def generateLeakyReLUShader (config : ActivationConfig) (alpha : Float := 0.01) : String :=
  generateWGSL "main" {x := config.workgroupSize, y := 1, z := 1} [] [] (leakyReluKernel config alpha)

/-- Generate WGSL shader for GELU -/
def generateGELUShader (config : ActivationConfig) : String :=
  generateWGSL "main" {x := config.workgroupSize, y := 1, z := 1} [] [] (geluKernel config)

/-- Generate WGSL shader for Sigmoid -/
def generateSigmoidShader (config : ActivationConfig) : String :=
  generateWGSL "main" {x := config.workgroupSize, y := 1, z := 1} [] [] (sigmoidKernel config)

/-- Generate WGSL shader for Tanh -/
def generateTanhShader (config : ActivationConfig) : String :=
  generateWGSL "main" {x := config.workgroupSize, y := 1, z := 1} [] [] (tanhKernel config)

/-- Generate WGSL shader for Swish/SiLU -/
def generateSwishShader (config : ActivationConfig) : String :=
  generateWGSL "main" {x := config.workgroupSize, y := 1, z := 1} [] [] (swishKernel config)

/-- Generate WGSL shader for ELU -/
def generateELUShader (config : ActivationConfig) (alpha : Float := 1.0) : String :=
  generateWGSL "main" {x := config.workgroupSize, y := 1, z := 1} [] [] (eluKernel config alpha)

/-- Generate WGSL shader for Softplus -/
def generateSoftplusShader (config : ActivationConfig) : String :=
  generateWGSL "main" {x := config.workgroupSize, y := 1, z := 1} [] [] (softplusKernel config)

/-- Generate WGSL shader for Mish -/
def generateMishShader (config : ActivationConfig) : String :=
  generateWGSL "main" {x := config.workgroupSize, y := 1, z := 1} [] [] (mishKernel config)

/-- Generate WGSL shader for Softmax (simplified version) -/
def generateSoftmaxShader (config : ActivationConfig) : String :=
  generateWGSL "main" {x := config.workgroupSize, y := 1, z := 1} [] [] (softmaxKernel config)

end Hesper.NN.Activation
