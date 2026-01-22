/-!
# Float16Array - Opaque Type Implementation

High-performance Float16 (half precision) array with **hardware-accelerated SIMD** and **GPU interop**.

## Hardware Requirements

**x86_64:** F16C extension (Intel Ivy Bridge+ / AMD Bulldozer+)
**ARM64:** ARMv8.2-A with FP16 vector arithmetic (Apple M1+, AWS Graviton2+)

**Fallback:** If hardware FP16 is unavailable, operations return errors.
Use `hasHardwareSupport` to check before using.

## Key Advantages

1. **4x Memory Savings:** vs Float64, 2x vs Float32 (2 bytes/element)
2. **Massive SIMD Width:** NEON 8 halfs/op, AVX2+F16C 8 halfs/op
3. **GPU Tensor Bandwidth:** Modern GPUs prefer FP16 for bandwidth-limited ops
4. **Zero Copy:** Direct pointer passing to WebGPU
5. **In-Place Updates:** Mutable operations (no array copying)

## Implementation Pattern

- **Lean Side:** Opaque type (contents invisible)
- **C++ Side:** `std::vector<uint16_t>` (FP16 stored as raw bits)
- **Conversion:** C++ handles f64 ↔ f16 conversion via `_cvtss_sh`/`_cvtsh_ss` (x86) or `vcvt` (ARM)
- **GPU Interop:** Pass raw pointer directly

## Usage

```lean
-- Check hardware support first!
let hasF16 ← Float16Array.hasHardwareSupport
if hasF16 then
  let arr ← Float16Array.create 1000
  arr.set 0 3.14
  let val ← arr.get 0

  -- Pass to GPU
  let ptr ← arr.ptr
  let size ← arr.byteSize
else
  -- Fall back to Float32
  IO.println "FP16 not supported, using Float32"
```
-/

namespace Hesper.Core

/-! ## Type Definition -/

/-- Opaque type representing a C++ std::vector<uint16_t> storing FP16 values.
    Contents are invisible to Lean - all operations go through FFI. -/
opaque Float16ArrayPointed : NonemptyType

/-- Float16 array type (opaque to Lean, managed by C++) -/
def Float16Array := Float16ArrayPointed.type

instance : Nonempty Float16Array := Float16ArrayPointed.property

/-! ## Hardware Detection -/

/-- Check if FP16 hardware support is available.

    **Returns:**
    - `true`: F16C (x86) or FP16 (ARM) available
    - `false`: No hardware support, Float16 operations will fail

    **Detection Method:**
    - x86_64: Check CPUID for F16C bit
    - ARM64: Check HWCAP for FPHP/ASIMDHP flags -/
@[extern "lean_f16_hardware_check"]
opaque Float16Array.hasHardwareSupport : IO Bool

/-! ## Creation and Destruction -/

/-- Create a new Float16Array with the specified number of elements.
    All elements are initialized to 0.0 (FP16 zero).

    **C++ Implementation:** `new std::vector<uint16_t>(size, 0)`
    **Requires:** FP16 hardware support (returns error if unavailable) -/
@[extern "lean_f16_array_create"]
opaque Float16Array.create (size : @& USize) : IO Float16Array

/-! ## Element Access -/

/-- Set a value at the specified index (in-place mutation).
    Value is converted from Float64 to Float16 (may lose precision).

    **Precision Loss:** F64 has 53-bit mantissa, F16 has 10-bit mantissa.
    Large values or high precision values will be rounded.

    **Safety:** Out-of-bounds access returns error.
    **Performance:** O(1), in-place update (no array copy). -/
@[extern "lean_f16_array_set"]
opaque Float16Array.set (self : @& Float16Array) (index : @& USize) (value : Float) : IO Unit

/-- Get a value at the specified index.
    Value is converted from Float16 to Float64 (exact conversion).

    **Safety:** Out-of-bounds access returns error.
    **Performance:** O(1). -/
@[extern "lean_f16_array_get"]
opaque Float16Array.get (self : @& Float16Array) (index : @& USize) : IO Float

/-! ## Array Properties -/

/-- Get the number of Float16 elements in the array. -/
@[extern "lean_f16_array_size"]
opaque Float16Array.size (self : @& Float16Array) : USize

/-- Get the size in bytes (for GPU buffer allocation).

    Returns: `size * sizeof(f16)` = `size * 2` -/
@[extern "lean_f16_array_byte_size"]
opaque Float16Array.byteSize (self : @& Float16Array) : IO USize

/-! ## GPU Interop (Zero-Copy Pointer Access) -/

/-- Get the raw pointer to the underlying data for GPU operations.

    **UNSAFE:** Returns raw pointer (size_t) to `vec.data()`.
    Valid only while Float16Array object is alive.

    **Use Case:** Zero-copy GPU tensor uploads:
    ```lean
    let ptr ← arr.ptr
    let size ← arr.byteSize
    -- C++: wgpuQueueWriteBuffer(queue, buffer, 0, (void*)ptr, size)
    ```

    **Safety:**
    1. Keep Float16Array alive during GPU operation
    2. No concurrent modifications while GPU reads
    -/
@[extern "lean_f16_array_ptr"]
opaque Float16Array.ptr (self : @& Float16Array) : IO USize

/-! ## Conversions -/

/-- Create Float16Array from Lean's native Array Float.

    **Precision Loss:** Each Float64 → Float16 conversion may lose precision.
    **Performance:** O(n), uses software FP16 conversion (hardware acceleration TODO). -/
@[extern "lean_f16_array_from_float_array"]
opaque Float16Array.fromFloatArray (arr : @& Array Float) : IO Float16Array

/-- Convert Float16Array to Lean's native Array Float.

    **Precision:** Float16 → Float64 is exact (no loss).
    **Performance:** O(n), uses software FP16 conversion (hardware acceleration TODO). -/
@[extern "lean_f16_array_to_float_array"]
opaque Float16Array.toFloatArray (self : @& Float16Array) : IO (Array Float)

/-- Create Float16Array from Float32Array.

    **Use Case:** GPU tensor pipelines often prefer FP16 for bandwidth.
    **Performance:** O(n), hardware-accelerated conversion. -/
@[extern "lean_f16_array_from_f32_array"]
opaque Float16Array.fromFloat32Array (arr : @& Float32Array) : IO Float16Array

/-- Convert Float16Array to Float32Array.

    **Use Case:** Mixed-precision training (FP16 forward, FP32 backward).
    **Performance:** O(n), hardware-accelerated conversion. -/
@[extern "lean_f16_array_to_f32_array"]
opaque Float16Array.toFloat32Array (self : @& Float16Array) : IO Float32Array

/-! ## SIMD Operations -/

/-- SIMD element-wise addition: result[i] = a[i] + b[i]

    **Hardware Acceleration:**
    - x86_64: AVX2 + F16C (8 halfs/operation)
    - ARM64: NEON FP16 (8 halfs/operation)

    **Safety:** Arrays must have same size.
    **Performance:** ~8x faster than scalar. -/
@[extern "lean_f16_array_simd_add"]
opaque Float16Array.simdAdd (a : @& Float16Array) (b : @& Float16Array) : IO Float16Array

/-- SIMD element-wise multiplication: result[i] = a[i] * b[i] -/
@[extern "lean_f16_array_simd_mul"]
opaque Float16Array.simdMul (a : @& Float16Array) (b : @& Float16Array) : IO Float16Array

/-! ## Utilities -/

/-- String representation for debugging -/
def Float16Array.toString (arr : Float16Array) : IO String := do
  let size := arr.size
  let n := min size.toNat 8
  let mut elements := []
  for i in [0:n] do
    let val ← arr.get i.toUSize
    elements := elements ++ [s!"{val}"]
  let rest := if size.toNat > 8 then s!", ... ({size.toNat - 8} more)" else ""
  return s!"Float16Array[{size}]: [{String.intercalate ", " elements}{rest}]"

instance : ToString Float16Array where
  toString arr := s!"Float16Array[size=?] (use arr.toString for details)"

end Hesper.Core
