/-!
# Float32Array - Opaque Type Implementation

High-performance Float32 array with **zero-copy GPU interop** and **in-place mutations**.

## Key Advantages

1. **Zero Copy:** Direct pointer passing to WebGPU (no conversion overhead)
2. **Memory Efficient:** Native C++ `std::vector<float>`, not boxed Lean objects
3. **In-Place Updates:** Mutable operations in IO monad (no array copying)
4. **2x Memory Savings:** vs Float64 (4 bytes/element vs 8 bytes/element)
5. **Better SIMD:** AVX2 8 floats/op vs 4 doubles/op, NEON 4 vs 2

## Implementation Pattern

- **Lean Side:** Opaque type (contents invisible to Lean)
- **C++ Side:** `std::vector<float>` allocated on heap
- **FFI Boundary:** Lean holds pointer via `Lean.External` with finalizer
- **GPU Interop:** Pass raw pointer via `ptr()` for `wgpuQueueWriteBuffer`

## Usage

```lean
-- Create array of 1000 elements (allocated in C++ heap)
let arr ← Float32Array.create 1000

-- Set values (in-place mutation, no copying!)
arr.set 0 3.14
arr.set 1 2.71

-- Get values (converts f32 → f64 only when reading)
let val ← arr.get 0  -- returns Float (f64)

-- Pass to GPU (zero-copy!)
let ptr ← arr.ptr
let size ← arr.byteSize
-- Use ptr with wgpuQueueWriteBuffer(queue, buffer, 0, ptr, size)
```
-/

namespace Hesper.Core

/-! ## Type Definition -/

/-- Opaque type representing a C++ std::vector<float>.
    Contents are invisible to Lean - all operations go through FFI. -/
opaque Float32ArrayPointed : NonemptyType

/-- Float32 array type (opaque to Lean, managed by C++) -/
def Float32Array := Float32ArrayPointed.type

instance : Nonempty Float32Array := Float32ArrayPointed.property

/-! ## Creation and Destruction -/

/-- Create a new Float32Array with the specified number of elements.
    All elements are initialized to 0.0f.

    **C++ Implementation:** `new std::vector<float>(size, 0.0f)` -/
@[extern "lean_f32_array_create"]
opaque Float32Array.create (size : @& USize) : IO Float32Array

/-! ## Element Access -/

/-- Set a value at the specified index (in-place mutation).
    Value is converted from Float64 to Float32.

    **Safety:** Out-of-bounds access will return error.
    **Performance:** O(1), in-place update (no array copy). -/
@[extern "lean_f32_array_set"]
opaque Float32Array.set (self : @& Float32Array) (index : @& USize) (value : Float) : IO Unit

/-- Get a value at the specified index.
    Value is converted from Float32 to Float64.

    **Safety:** Out-of-bounds access will return error.
    **Performance:** O(1). -/
@[extern "lean_f32_array_get"]
opaque Float32Array.get (self : @& Float32Array) (index : @& USize) : IO Float

/-! ## Array Properties -/

/-- Get the number of Float32 elements in the array.

    **Performance:** O(1) - calls `vec.size()`. -/
@[extern "lean_f32_array_size"]
opaque Float32Array.size (self : @& Float32Array) : USize

/-- Get the size in bytes (for GPU buffer allocation).

    Returns: `size * sizeof(float)` = `size * 4`
    **Performance:** O(1). -/
@[extern "lean_f32_array_byte_size"]
opaque Float32Array.byteSize (self : @& Float32Array) : IO USize

/-! ## GPU Interop (Zero-Copy Pointer Access) -/

/-- Get the raw pointer to the underlying data for GPU operations.

    **UNSAFE:** This returns a raw pointer (size_t) to `vec.data()`.
    The pointer is only valid while the Float32Array object is alive.

    **Use Case:** Pass to `wgpuQueueWriteBuffer` for zero-copy uploads:
    ```lean
    let ptr ← arr.ptr
    let size ← arr.byteSize
    -- C++: wgpuQueueWriteBuffer(queue, buffer, 0, (void*)ptr, size)
    ```

    **Safety:** Caller must ensure:
    1. Float32Array stays alive during GPU operation
    2. No concurrent modifications while GPU reads
    -/
@[extern "lean_f32_array_ptr"]
opaque Float32Array.ptr (self : @& Float32Array) : IO USize

/-! ## Conversions -/

/-- Create Float32Array from Lean's native Array Float.

    **Performance:** O(n) - each Float64 is converted to Float32.
    **Use Case:** Initial data preparation from Lean computations. -/
@[extern "lean_f32_array_from_float_array"]
opaque Float32Array.fromFloatArray (arr : @& Array Float) : IO Float32Array

/-- Convert Float32Array to Lean's native Array Float.

    **Performance:** O(n) - each Float32 is converted to Float64.
    **Use Case:** Reading results back into Lean for verification/analysis. -/
@[extern "lean_f32_array_to_float_array"]
opaque Float32Array.toFloatArray (self : @& Float32Array) : IO (Array Float)

/-! ## SIMD Operations -/

/-- SIMD element-wise addition: result[i] = a[i] + b[i]

    **Hardware Acceleration:**
    - x86_64: AVX2 (8 floats/operation)
    - ARM64: NEON (4 floats/operation)

    **Safety:** Arrays must have same size, returns error otherwise.
    **Performance:** ~8x faster than scalar on AVX2, ~4x on NEON. -/
@[extern "lean_f32_array_simd_add"]
opaque Float32Array.simdAdd (a : @& Float32Array) (b : @& Float32Array) : IO Float32Array

/-- SIMD element-wise multiplication: result[i] = a[i] * b[i] -/
@[extern "lean_f32_array_simd_mul"]
opaque Float32Array.simdMul (a : @& Float32Array) (b : @& Float32Array) : IO Float32Array

/-! ## Utilities -/

/-- String representation for debugging (shows first 8 elements) -/
def Float32Array.toString (arr : Float32Array) : IO String := do
  let size := arr.size
  let n := min size.toNat 8
  let mut elements := []
  for i in [0:n] do
    let val ← arr.get i.toUSize
    elements := elements ++ [s!"{val}"]
  let rest := if size.toNat > 8 then s!", ... ({size.toNat - 8} more)" else ""
  return s!"Float32Array[{size}]: [{String.intercalate ", " elements}{rest}]"

instance : ToString Float32Array where
  toString _arr :=
    -- Note: Can't use IO in pure ToString, so we show a placeholder
    s!"Float32Array[size=?] (use arr.toString for details)"

end Hesper.Core
