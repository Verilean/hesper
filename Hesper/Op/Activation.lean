import Hesper.Core.VerifiedOpFusion
import Hesper.WGSL.Kernel
import Hesper.WGSL.Exp
import Hesper.WGSL.Types

/-!
# Activation Functions as Verified Operators

Implements element-wise activation functions (ReLU, Sigmoid, Tanh) as verified operators
with kernel fusion support.

These are perfect examples of fusable operations because they:
1. Are element-wise (no cross-element dependencies)
2. Compose nicely: `MatMul |> ReLU |> Softmax`
3. Can be fused into preceding operations

## Mathematical Definitions

**ReLU**: `f(x) = max(0, x)`
- Forward: `y = max(0, x)`
- Backward: `dy/dx = 1 if x > 0, else 0`

**Sigmoid**: `f(x) = 1 / (1 + exp(-x))`
- Forward: `y = sigmoid(x)`
- Backward: `dy/dx = y * (1 - y)`

**Tanh**: `f(x) = tanh(x)`
- Forward: `y = tanh(x)`
- Backward: `dy/dx = 1 - y²`
-/

namespace Hesper.Op.Activation

open Hesper.Core
open Hesper.WGSL
open Hesper.Tensor

/-- Activation function type -/
inductive ActivationType
  | Identity
  | ReLU
  | Sigmoid
  | Gelu
  deriving Inhabited, Repr, BEq

/-! ## ReLU Activation -/

/-- ReLU input: single tensor -/
structure ReLUInput where
  x : TensorData
  deriving Inhabited

/-- ReLU output: single tensor -/
structure ReLUOutput where
  y : TensorData
  deriving Inhabited

/-- CPU ReLU forward pass: element-wise max(0, x) -/
def cpuReLU (input : ReLUInput) : ReLUOutput :=
  let y_data := input.x.data.map fun val => max 0.0 val
  { y := { shape := input.x.shape, data := y_data } }

/-- CPU ReLU backward pass: gradient is 1 if x > 0, else 0 -/
def cpuReLUBackward (input : ReLUInput) (grad_output : ReLUOutput) : ReLUInput :=
  let grad_x := Array.range input.x.data.size |>.map fun i =>
    let x_val := input.x.data[i]!
    let grad_y := grad_output.y.data[i]!
    if x_val > 0.0 then grad_y else 0.0
  { x := { shape := input.x.shape, data := grad_x } }

/-! ## GPU ReLU Kernel -/

/-- ReLU as a WGSL expression transformation.
    This is a pure function that can be fused into other kernels. -/
def reluExp (x : Exp (.scalar .f32)) : Exp (.scalar .f32) :=
  Exp.max x (Exp.litF32 0.0)

/-- GPU ReLU forward kernel.
    Element-wise operation that can be fused with other kernels.

    Example usage:
    ```lean
    let fused = matmul_kernel |> relu_kernel  -- Fuses matmul + relu
    ```
-/
def gpuReLUKernel {N : Nat} : Kernel 256 1 1 (Exp (.scalar .f32)) (Exp (.scalar .f32)) :=
  mapK reluExp

/-- GPU ReLU backward kernel.
    Gradient: 1 if x > 0, else 0

    Takes (input, grad_output) and returns grad_input.
    For ReLU: grad_input = grad_output if input > 0, else 0 -/
def gpuReLUBackwardKernel
    : Kernel 256 1 1 (Exp (.scalar .f32) × Exp (.scalar .f32)) (Exp (.scalar .f32)) :=
  ⟨fun (x, grad_y) => do
    -- if x > 0 then grad_y else 0
    let zero := Exp.litF32 0.0
    let mask := Exp.gt x zero
    -- Use select: select(condition, if_true, if_false)
    -- If x > 0, return grad_y, else return 0
    let result := Exp.mul grad_y (Exp.select mask (Exp.litF32 1.0) zero)
    return result⟩

/-! ## VerifiedOpFusion Instance for ReLU -/

instance : VerifiedOpFusion 256 1 1 ReLUInput ReLUOutput
    (Exp (.scalar .f32)) (Exp (.scalar .f32)) where
  spec_forward := cpuReLU
  impl_kernel := gpuReLUKernel (N := 256)
  spec_backward := cpuReLUBackward
  impl_kernel_backward := gpuReLUBackwardKernel

  -- Optional: provide immediate execution wrapper
  run_forward := fun input => do
    -- Placeholder: would compile and execute kernel
    return input  -- placeholder

  verify_consistency := fun input tolerance => do
    -- Placeholder: would run both CPU and GPU and compare
    return true

/-! ## Helper Functions -/

/-- Create ReLU input from TensorData -/
def mkReLUInput (data : TensorData) : ReLUInput :=
  { x := data }

/-- Extract TensorData from ReLU output -/
def getReLUOutput (output : ReLUOutput) : TensorData :=
  output.y

/-! ## Sigmoid Activation (TODO) -/

/-- Sigmoid as WGSL expression: 1 / (1 + exp(-x)) -/
def sigmoidExp (x : Exp (.scalar .f32)) : Exp (.scalar .f32) :=
  let negX := Exp.neg x
  let expNegX := Exp.exp negX
  let onePlusExp := Exp.add (Exp.litF32 1.0) expNegX
  Exp.div (Exp.litF32 1.0) onePlusExp

/-! ## Tanh Activation (TODO) -/

/-- Tanh as WGSL expression -/
def tanhExp (x : Exp (.scalar .f32)) : Exp (.scalar .f32) :=
  -- (exp(x) - exp(-x)) / (exp(x) + exp(-x))
  let expX := Exp.exp x
  let expNegX := Exp.exp (Exp.neg x)
  let numer := Exp.sub expX expNegX
  let denom := Exp.add expX expNegX
  Exp.div numer denom

end Hesper.Op.Activation
