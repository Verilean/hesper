/-!
# BFloat16Array - Brain Floating Point (Opaque Type Implementation)

High-performance BFloat16 array optimized for **deep learning** and **neural network training**.

## What is BFloat16?

BFloat16 (Brain Float 16) is a 16-bit floating point format designed by Google Brain:

- **16 bits total:** 1 sign + 8 exponent + 7 mantissa
- **Same exponent range as Float32:** 8 bits (can represent same magnitude range)
- **Reduced precision:** 7-bit mantissa vs 23-bit (Float32) or 10-bit (Float16)

## Why BFloat16 for ML?

1. **Gradient Stability:** Same exponent range as FP32 prevents gradient underflow/overflow
2. **Memory Bandwidth:** 2x savings vs FP32 (critical for large models)
3. **Hardware Support:** TPUs, some GPUs, and modern CPUs support BF16
4. **Easy Conversion:** Truncate FP32's mantissa (simpler than FP16 conversion)

## Hardware Support

**x86_64:** AVX-512 BF16 extensions (Intel Cooper Lake+, AMD Zen4+)
**ARM64:** BFloat16 arithmetic (ARMv8.6-A+, Apple M2+)
**GPU:** NVIDIA Ampere+, AMD RDNA3+, Intel Arc+

**Fallback:** Software emulation available (truncate FP32 mantissa to 7 bits)

## Key Advantages

1. **2x Memory Savings:** vs Float32 (2 bytes vs 4 bytes)
2. **Stable Training:** No gradient underflow issues unlike FP16
3. **Fast Conversion:** FP32 ↔ BF16 is simpler than FP32 ↔ FP16
4. **Zero Copy GPU:** Direct pointer passing to WebGPU
5. **In-Place Updates:** Mutable operations (no copying)

## Usage in Mixed-Precision Training

```lean
-- Forward pass: BF16 (memory efficient)
let activations ← BFloat16Array.create 1000000
-- ... compute forward pass in BF16 ...

-- Backward pass: Convert gradients to FP32 for accumulation
let gradients ← activations.toFloat32Array
-- ... accumulate in FP32 for numerical stability ...

-- Update weights: Convert back to BF16 for storage
let weights ← BFloat16Array.fromFloat32Array updatedWeights
```

## Implementation Pattern

- **Lean Side:** Opaque type (contents invisible)
- **C++ Side:** `std::vector<uint16_t>` (BF16 stored as raw bits)
- **Conversion:** C++ handles f32 ↔ bf16 via bit manipulation or hardware intrinsics
- **GPU Interop:** Pass raw pointer directly
-/

namespace Hesper.Core

/-! ## Type Definition -/

/-- Opaque type representing a C++ std::vector<uint16_t> storing BF16 values.
    Contents are invisible to Lean - all operations go through FFI. -/
opaque BFloat16ArrayPointed : NonemptyType

/-- BFloat16 array type (opaque to Lean, managed by C++) -/
def BFloat16Array := BFloat16ArrayPointed.type

instance : Nonempty BFloat16Array := BFloat16ArrayPointed.property

/-! ## Hardware Detection -/

/-- Check if BFloat16 hardware support is available.

    **Returns:**
    - `true`: AVX-512 BF16 or ARMv8.6 BF16 available
    - `false`: Will use software emulation (slower but functional)

    **Note:** Unlike FP16, BF16 has efficient software fallback. -/
@[extern "lean_bf16_hardware_check"]
opaque BFloat16Array.hasHardwareSupport : IO Bool

/-! ## Creation and Destruction -/

/-- Create a new BFloat16Array with the specified number of elements.
    All elements are initialized to 0.0 (BF16 zero).

    **C++ Implementation:** `new std::vector<uint16_t>(size, 0)` -/
@[extern "lean_bf16_array_create"]
opaque BFloat16Array.create (size : @& USize) : IO BFloat16Array

/-! ## Element Access -/

/-- Set a value at the specified index (in-place mutation).
    Value is converted from Float64 → Float32 → BFloat16.

    **Conversion:** Round to nearest BF16 value (RNE mode).
    **Precision:** 7-bit mantissa provides ~3 decimal digits accuracy.

    **Safety:** Out-of-bounds access returns error.
    **Performance:** O(1), in-place update. -/
@[extern "lean_bf16_array_set"]
opaque BFloat16Array.set (self : @& BFloat16Array) (index : @& USize) (value : Float) : IO Unit

/-- Get a value at the specified index.
    Value is converted from BFloat16 → Float32 → Float64.

    **Conversion:** Exact (zero-extend mantissa).
    **Safety:** Out-of-bounds access returns error.
    **Performance:** O(1). -/
@[extern "lean_bf16_array_get"]
opaque BFloat16Array.get (self : @& BFloat16Array) (index : @& USize) : IO Float

/-! ## Array Properties -/

/-- Get the number of BFloat16 elements in the array. -/
@[extern "lean_bf16_array_size"]
opaque BFloat16Array.size (self : @& BFloat16Array) : USize

/-- Get the size in bytes (for GPU buffer allocation).

    Returns: `size * sizeof(bf16)` = `size * 2` -/
@[extern "lean_bf16_array_byte_size"]
opaque BFloat16Array.byteSize (self : @& BFloat16Array) : IO USize

/-! ## GPU Interop (Zero-Copy Pointer Access) -/

/-- Get the raw pointer to the underlying data for GPU operations.

    **UNSAFE:** Returns raw pointer (size_t) to `vec.data()`.

    **Use Case:** Zero-copy uploads for ML training on GPU:
    ```lean
    let ptr ← weights.ptr
    let size ← weights.byteSize
    -- C++: wgpuQueueWriteBuffer(queue, weightBuffer, 0, (void*)ptr, size)
    ```

    **Safety:**
    1. Keep BFloat16Array alive during GPU operation
    2. No concurrent modifications
    -/
@[extern "lean_bf16_array_ptr"]
opaque BFloat16Array.ptr (self : @& BFloat16Array) : IO USize

/-! ## Conversions -/

/-- Create BFloat16Array from Lean's native Array Float.

    **Conversion:** Float64 → Float32 → BFloat16
    **Performance:** O(n), uses hardware BF16 if available (TODO). -/
@[extern "lean_bf16_array_from_float_array"]
opaque BFloat16Array.fromFloatArray (arr : @& Array Float) : IO BFloat16Array

/-- Convert BFloat16Array to Lean's native Array Float.

    **Conversion:** BFloat16 → Float32 → Float64 (exact)
    **Performance:** O(n) (TODO: implementation pending). -/
@[extern "lean_bf16_array_to_float_array"]
opaque BFloat16Array.toFloatArray (self : @& BFloat16Array) : IO (Array Float)

/-- Create BFloat16Array from Float32Array.

    **Most common ML use case:** Store activations/weights in BF16.
    **Conversion:** Truncate FP32 mantissa from 23 bits to 7 bits.
    **Performance:** O(n), very fast (bit shift operation). -/
@[extern "lean_bf16_array_from_f32_array"]
opaque BFloat16Array.fromFloat32Array (arr : @& Float32Array) : IO BFloat16Array

/-- Convert BFloat16Array to Float32Array.

    **Most common ML use case:** Convert to FP32 for gradient accumulation.
    **Conversion:** Zero-extend mantissa from 7 bits to 23 bits (exact).
    **Performance:** O(n), very fast (bit shift operation). -/
@[extern "lean_bf16_array_to_f32_array"]
opaque BFloat16Array.toFloat32Array (self : @& BFloat16Array) : IO Float32Array

/-! ## SIMD Operations -/

/-- SIMD element-wise addition: result[i] = a[i] + b[i]

    **Hardware Acceleration:**
    - x86_64: AVX-512 BF16 (Intel Cooper Lake+)
    - ARM64: BF16 arithmetic (ARMv8.6-A+)
    - Fallback: Software emulation via FP32

    **Safety:** Arrays must have same size.
    **Performance:** Hardware ~16x faster than scalar, software ~4x. -/
@[extern "lean_bf16_array_simd_add"]
opaque BFloat16Array.simdAdd (a : @& BFloat16Array) (b : @& BFloat16Array) : IO BFloat16Array

/-- SIMD element-wise multiplication: result[i] = a[i] * b[i] -/
@[extern "lean_bf16_array_simd_mul"]
opaque BFloat16Array.simdMul (a : @& BFloat16Array) (b : @& BFloat16Array) : IO BFloat16Array

/-! ## ML-Specific Operations -/

/-- Fused multiply-add: result[i] = a[i] * b[i] + c[i]

    **Use Case:** Core operation in matrix multiplication and convolution.
    **Hardware:** Single instruction on modern CPUs/GPUs with BF16 support.
    **Performance:** 2x faster than separate mul + add. -/
@[extern "lean_bf16_array_fma"]
opaque BFloat16Array.fma (a : @& BFloat16Array) (b : @& BFloat16Array) (c : @& BFloat16Array) : IO BFloat16Array

/-! ## Utilities -/

/-- String representation for debugging -/
def BFloat16Array.toString (arr : BFloat16Array) : IO String := do
  let size := arr.size
  let n := min size.toNat 8
  let mut elements := []
  for i in [0:n] do
    let val ← arr.get i.toUSize
    elements := elements ++ [s!"{val}"]
  let rest := if size.toNat > 8 then s!", ... ({size.toNat - 8} more)" else ""
  return s!"BFloat16Array[{size}]: [{String.intercalate ", " elements}{rest}]"

instance : ToString BFloat16Array where
  toString arr := s!"BFloat16Array[size=?] (use arr.toString for details)"

end Hesper.Core
