import Hesper.Tensor.Types

namespace Hesper.Core

/--
# Differentiable Interface

The core type class for all differentiable operations in Hesper.
Unifies scalar CPU operations with tensor GPU kernels.

Type parameters:
- `I`: Input type (e.g., Float, TensorData, or a product of types)
- `O`: Output type (result of the forward pass)
-/
class Differentiable (Op : Type) (I : Type) (O : Type) where
  /-- Primal execution (Forward pass).
      Maps input to output. -/
  forward : Op → I → O

  /-- Adjoint computation (Backward pass).
      Given the original input and the gradient of the loss with respect to the output (v),
      computes the gradient of the loss with respect to the input (Jᵀv).

      This is also known as the Vector-Jacobian Product (VJP). -/
  backward : Op → I → O → I

export Differentiable (forward backward)

end Hesper.Core
