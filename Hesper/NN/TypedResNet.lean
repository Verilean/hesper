import Hesper.Tensor.Typed
import Hesper.NN.ResNet

namespace Hesper.NN.TypedResNet

open Hesper.Tensor

/-!
# Typed ResNet Implementation
Demonstrates using TypedTensors for verifying ResNet architecture constraints.
-/

/-- Helper to calculate output dimension -/
def convDim (dim : Nat) (padding : Nat) (kernel : Nat) (stride : Nat) : Nat :=
  (dim + 2 * padding - kernel) / stride + 1

/-- Typed Convolution Layer -/
structure TypedConvLayer (inC outC kSize stride padding : Nat) (dtype : DType) where
  weights : TypedTensor (Shape.tensor4D outC inC kSize kSize) dtype
  bias    : TypedTensor (Shape.vector outC) dtype

namespace TypedConvLayer

  /--
  Typed Forward Pass for Conv2D.
   Input: [N, H, W, inC]
   Output: [N, OH, OW, outC]
  -/
  def forward {N H W inC outC kSize stride padding : Nat} {dt : DType}
    (layer : TypedConvLayer inC outC kSize stride padding dt)
    (input : TypedTensor (Shape.tensor4D N H W inC) dt)
    : TypedTensor (Shape.tensor4D N
        (convDim H padding kSize stride)
        (convDim W padding kSize stride)
        outC) dt :=
    -- Placeholder logic (implementation would loop loops)
    let OH := convDim H padding kSize stride
    let OW := convDim W padding kSize stride
    TypedTensor.zeros (Shape.tensor4D N OH OW outC) dt

end TypedConvLayer

/--
Typed Residual Block.
Constraint: Input shape must match Output shape of Conv block for addition.
This implies Stride=1 and Padding preserves size, OR projection is needed.
Here we verify the "Identity Skip" case requires shape preservation.
-/
structure TypedResidualBlock (channels : Nat) (dtype : DType) where
  -- We fix kernel=3, stride=1, padding=1 for "same" convolution behavior
  conv1 : TypedConvLayer channels channels 3 1 1 dtype
  conv2 : TypedConvLayer channels channels 3 1 1 dtype

namespace TypedResidualBlock

  /--
  Forward pass.
  Note: Logic automatically verifies that shapes match for addition!
  convDim H 1 3 1 = (H + 2 - 3)/1 + 1 = (H - 1) + 1 = H
  So the type signature proves dimensions are preserved.
  -/
  def forward {N H W : Nat} {dt : DType} {C : Nat}
    (block : TypedResidualBlock C dt)
    (input : TypedTensor (Shape.tensor4D N H W C) dt)
    : TypedTensor (Shape.tensor4D N H W C) dt :=

    -- 1. Conv 1
    -- Type inference automatically calculates output shape is [N, H, W, C]
    let out1 := block.conv1.forward input

    -- 2. ReLU (Map) - shape preserved
    let act1 := out1.map (fun x => x) -- Placeholder for relu

    -- 3. Conv 2
    let out2 := block.conv2.forward act1

    -- 4. Skip Connection Add
    -- This compiles ONLY if out2 shape == input shape
    let added := out2.add input

    -- 5. Final Activation
    added.map (fun x => x)

end TypedResidualBlock

/-!
## Typed ResNet Verification
-/

def demoTypedResNet : TypedTensor (Shape.tensor4D 1 32 32 64) .f16 :=
  let tInput := TypedTensor.zeros (Shape.tensor4D 1 32 32 64) .f16

  -- Create block with 64 channels
  let block : TypedResidualBlock 64 .f16 := {
    conv1 := { weights := TypedTensor.zeros _ _, bias := TypedTensor.zeros _ _ }
    conv2 := { weights := TypedTensor.zeros _ _, bias := TypedTensor.zeros _ _ }
  }

  -- Run forward pass
  block.forward tInput

-- Demonstration of Safety:
-- If we tried to use a block expecting 32 channels on 64 channel input,
-- it would fail at `block.conv1.forward input` because `inC` mismatches.

end Hesper.NN.TypedResNet
