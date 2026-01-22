import Hesper.Tensor.Typed

/-!
# RMSNorm (Root Mean Square Normalization)
-/

namespace Hesper.NN.Normalization

open Hesper.Tensor

/-- Typed RMSNorm Layer -/
structure RMSNorm (dim : Nat) (dtype : DType) where
  weight : TypedTensor (Shape.vector dim) dtype
  eps : Float := 1e-6

namespace RMSNorm

  def forward {Batch Seq : Nat} {dim : Nat} {dt : DType}
    (layer : RMSNorm dim dt)
    (input : TypedTensor (Shape.tensor3D Batch Seq dim) dt)
    : TypedTensor (Shape.tensor3D Batch Seq dim) dt :=
    -- Placeholder
    input

end RMSNorm

end Hesper.NN.Normalization
