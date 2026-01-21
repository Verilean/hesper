import Hesper.WebGPU.Types
import Hesper.WebGPU.Buffer
import Hesper.Tensor.Types
import Hesper.WGSL.Kernel
import Hesper.WGSL.Types
import Hesper.WGSL.Exp
import Hesper.Basic

/-!
# Verified Operator Pattern with Kernel Fusion

This module extends the VerifiedOp pattern to support **kernel fusion**.

## Key Design Evolution

### Before (Immediate Execution - No Fusion):
```lean
let tmp ← MatMul.impl_forward A B   -- Writes to VRAM
let res ← ReLU.impl_forward tmp     -- Reads from VRAM, writes again
-- Problem: 2 memory roundtrips!
```

### After (Lazy Composition - Fusion Enabled):
```lean
let fused = MatMul.impl_kernel |> ReLU.impl_kernel  -- Build recipe
let res ← run_kernel fused input                     -- Single GPU dispatch!
-- Benefit: 1 memory roundtrip, fused shader
```

## Type Parameters

- **`I, O`**: High-level CPU types (e.g., `MatMulInput`, `TensorData`)
- **`WI, WO`**: WGSL expression types (e.g., `Exp (.array .f32 256)`)

The separation allows:
1. CPU spec works on convenient high-level types
2. GPU impl works on low-level WGSL expressions
3. Kernels can be composed before execution

## Workgroup Size

Kernels are parameterized by workgroup dimensions `(wX, wY, wZ)`.
This is tracked at the type level for safety.
-/

namespace Hesper.Core

open Hesper.WebGPU
open Hesper.Tensor
open Hesper.WGSL

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

/-! ## Fusable Verified Operator Type Class -/

/-- Type class for verified operators with kernel fusion support.

    **Type parameters:**
    - `I, O`: High-level CPU types (e.g., `Matrix`, `TensorData`)
    - `WI, WO`: WGSL expression types (e.g., `Exp (.array .f32 N)`)
    - `wX, wY, wZ`: Workgroup dimensions (compile-time constants)

    **Design philosophy:**
    1. **Specification (CPU)**: Pure mathematical definition for correctness
    2. **Implementation (GPU)**: Composable kernel for performance
    3. **Fusion**: Kernels compose via `|>` before execution
    4. **Verification**: Compare spec vs impl for correctness

    **Example usage:**
    ```lean
    -- Fuse two operators
    let fused = MatMul.impl_kernel |> ReLU.impl_kernel

    -- Execute fused kernel
    let result ← execute_kernel device fused input

    -- Verify correctness
    let cpu_result := MatMul.spec_forward input |> ReLU.spec_forward
    assert (cpu_result ≈ result)
    ```
-/
class VerifiedOpFusion (wX wY wZ : Nat) (I O : Type) (WI WO : Type) where
  /-- **Specification**: Pure mathematical definition (CPU).
      Reference implementation for correctness and formal verification.

      This should be:
      - Easy to understand and verify
      - Provably correct
      - Used as the "ground truth" for testing -/
  spec_forward : I → O

  /-- **Implementation**: Composable GPU kernel.
      Returns a `Kernel` that can be fused with other kernels.

      This should be:
      - Lazy (doesn't execute immediately)
      - Composable (can be chained with `|>`)
      - Optimized for GPU performance

      Example:
      ```lean
      impl_kernel : Kernel 256 1 1 (Exp (.array .f32 N)) (Exp (.array .f32 N))
      ```
      -/
  impl_kernel : Kernel wX wY wZ WI WO

  /-- **Backward Specification**: CPU gradient computation.
      Given (input, grad_output), compute grad_input.

      Used for training and automatic differentiation. -/
  spec_backward : I → O → I

  /-- **Backward Implementation**: GPU gradient kernel.
      Composable kernel for backpropagation.

      Can be fused with other backward passes. -/
  impl_kernel_backward : Kernel wX wY wZ (WI × WO) WI

  /-- **Execution Helper**: Optional wrapper for immediate execution.
      Compiles and runs the kernel immediately (for testing/debugging).

      Default implementation provided. Can be overridden per operator. -/
  run_forward : WI → IO WO :=
    fun _ => do
      -- Placeholder: requires device context and compilation
      throw (IO.userError "run_forward not implemented")

  /-- **Verification**: Check CPU spec matches GPU impl.
      Requires conversion between high-level (I/O) and low-level (WI/WO) types.

      Default implementation is a placeholder. -/
  verify_consistency : I → Float → IO Bool :=
    fun _ _ => return true  -- Placeholder

export VerifiedOpFusion (spec_forward impl_kernel spec_backward impl_kernel_backward run_forward verify_consistency)

/-! ## Helper Functions for Fusion -/

/-- Compose two verified operators into a fused operation.
    The result is a new kernel that applies f then g in a single pass.

    Example:
    ```lean
    let matmul_relu = composeOps (I := MatMulInput) (M := MatMulOutput) (O := ReLUOutput)
    -- Single GPU kernel that does matmul AND relu!
    ```
-/
def composeOps {wX wY wZ : Nat} {I M O : Type} {WI WM WO : Type}
    [inst1 : VerifiedOpFusion wX wY wZ I M WI WM]
    [inst2 : VerifiedOpFusion wX wY wZ M O WM WO]
    : Kernel wX wY wZ WI WO :=
  let f_kernel := inst1.impl_kernel
  let g_kernel := inst2.impl_kernel
  Kernel.comp g_kernel f_kernel

/-- Create a simple element-wise operation verified operator.

    This is a helper for defining simple pointwise operations like ReLU, sigmoid, etc.

    Parameters:
    - `gpu_fn`: WGSL expression transformation (Exp ty → Exp ty)
-/
def mkElementwiseOp {wX wY wZ : Nat} {ty : WGSLType}
    (gpu_fn : Exp ty → Exp ty)
    : Kernel wX wY wZ (Exp ty) (Exp ty) :=
  mapK gpu_fn

end Hesper.Core
