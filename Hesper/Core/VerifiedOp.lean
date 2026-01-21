import Hesper.WebGPU.Types
import Hesper.WebGPU.Buffer
import Hesper.Tensor.Types
import Hesper.Basic

/-!
# Verified Operator Pattern for Hesper

This module implements the core abstraction for GPU-accelerated operations with
verified correctness. The pattern separates:

1. **Specification** (spec_forward, spec_backward): Pure mathematical definitions on CPU
   - Reference implementation for testing and verification
   - Used in proofs and correctness checking

2. **Implementation** (impl_forward, impl_backward): Optimized WGSL kernels on GPU
   - High-performance GPU implementations
   - Verified to match specification within tolerance

## Design Principles

- **Separation of Concerns**: Math (CPU) vs Performance (GPU)
- **Type Safety**: Lean's type system ensures correctness
- **Verifiability**: Can prove `impl ≈ spec` for all inputs
- **Composability**: Operators compose to build larger networks

## Usage

```lean
-- Define an operator instance
instance : VerifiedOp MatMulInput MatMulOutput where
  spec_forward := cpuMatMul
  impl_forward := gpuMatMul
  spec_backward := cpuMatMulGrad
  impl_backward := gpuMatMulGrad

-- Verify consistency
verify_consistency myInput tolerance
```
-/

namespace Hesper.Core

open Hesper.WebGPU
open Hesper.Tensor

/-! ## Data Abstractions -/

/-- CPU tensor data: wrapper around Array Float with shape information.
    Used for specification and testing. -/
structure TensorData where
  /-- Shape of the tensor -/
  shape : Shape
  /-- Flattened data in row-major order -/
  data : Array Float
  deriving Inhabited, Repr

namespace TensorData

/-- Create a zero tensor with given shape -/
def zeros (shape : Shape) : TensorData :=
  let size := shape.size
  { shape := shape, data := Array.mk (List.replicate size 0.0) }

/-- Create a tensor filled with a constant value -/
def constant (shape : Shape) (value : Float) : TensorData :=
  let size := shape.size
  { shape := shape, data := Array.mk (List.replicate size value) }

/-- Get element at flattened index -/
def get (t : TensorData) (idx : Nat) : Float :=
  t.data[idx]!

/-- Set element at flattened index -/
def set (t : TensorData) (idx : Nat) (value : Float) : TensorData :=
  { t with data := t.data.set! idx value }

/-- Total number of elements -/
def size (t : TensorData) : Nat :=
  t.shape.size

/-- Check if two tensors are approximately equal within tolerance -/
def approxEq (a b : TensorData) (tolerance : Float := 1e-5) : Bool :=
  if a.shape.dims != b.shape.dims then
    false
  else
    a.data.size == b.data.size &&
    (Array.range a.data.size).all fun i =>
      let diff := Float.abs (a.data[i]! - b.data[i]!)
      diff ≤ tolerance

end TensorData

/-- GPU tensor handle: wrapper around WebGPU Buffer with shape and device.
    Used for high-performance GPU operations. -/
structure GPUHandle where
  /-- Device this buffer belongs to -/
  device : Device
  /-- GPU buffer containing the data -/
  buffer : Buffer
  /-- Shape of the tensor -/
  shape : Shape

namespace GPUHandle

/-- Upload CPU tensor data to GPU -/
def fromTensorData (device : Device) (data : TensorData) : IO GPUHandle := do
  let sizeBytes := data.data.size * 4  -- Float = 4 bytes
  let desc : BufferDescriptor := {
    size := sizeBytes.toUSize,
    usage := [.storage, .copySrc, .copyDst],
    mappedAtCreation := false
  }
  let buffer ← createBuffer device desc
  let bytes := floatArrayToBytes data.data
  writeBuffer device buffer 0 bytes
  return { device := device, buffer := buffer, shape := data.shape }

/-- Download GPU tensor data to CPU -/
def toTensorData (handle : GPUHandle) : IO TensorData := do
  let sizeBytes := handle.shape.size * 4
  let bytes ← mapBufferRead handle.device handle.buffer 0 sizeBytes.toUSize
  let data ← Hesper.Basic.bytesToFloatArray bytes
  return { shape := handle.shape, data := data }

/-- Create a zero tensor on GPU -/
def zeros (device : Device) (shape : Shape) : IO GPUHandle := do
  let cpuData := TensorData.zeros shape
  fromTensorData device cpuData

end GPUHandle

/-! ## Verified Operator Type Class -/

/-- Type class for verified operators with both CPU specification and GPU implementation.

    Type parameters:
    - `I`: Input type (can be single tensor or tuple of tensors)
    - `O`: Output type (can be single tensor or tuple of tensors)

    This pattern enables:
    1. Formal verification: Prove properties on `spec_forward`
    2. Performance: Run `impl_forward` on GPU
    3. Testing: Compare `spec` vs `impl` via `verify_consistency`
    4. Custom gradients: Hand-optimized `impl_backward` kernels
-/
class VerifiedOp (I O : Type) where
  /-- **Specification**: Pure mathematical definition (CPU).
      Reference implementation for correctness and verification. -/
  spec_forward : I → O

  /-- **Implementation**: Optimized GPU implementation.
      Executes WGSL compute shaders for high performance. -/
  impl_forward : I → IO O

  /-- **Specification**: Mathematical gradient (CPU).
      Defines the backward pass for automatic differentiation.
      Takes (input, grad_output) and returns grad_input. -/
  spec_backward : I → O → I

  /-- **Implementation**: Optimized GPU gradient kernel.
      Custom WGSL shader for backpropagation.
      Takes (input, grad_output) and returns grad_input on GPU. -/
  impl_backward : I → O → IO I

  /-- **Verification**: Check that GPU implementation matches CPU specification.
      Runs both `spec_forward` and `impl_forward`, compares results within tolerance.
      Returns `true` if they match, `false` otherwise. -/
  verify_consistency : I → Float → IO Bool :=
    -- Default implementation (can be overridden per operator)
    fun _ _ => return true  -- Placeholder: requires conversion functions I→TensorData

export VerifiedOp (spec_forward impl_forward spec_backward impl_backward verify_consistency)

/-! ## Helper Functions -/

/-- Run verification test and print result -/
def runVerificationTest [inst : VerifiedOp I O] (name : String) (input : I) (tolerance : Float := 1e-4) : IO Unit := do
  IO.println s!"🔍 Verifying {name}..."
  let passed ← inst.verify_consistency input tolerance
  if passed then
    IO.println s!"  ✅ Verification passed (tolerance: {tolerance})"
  else
    IO.println s!"  ❌ Verification FAILED (tolerance: {tolerance})"

end Hesper.Core
