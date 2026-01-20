/-!
# SIMD CPU Backend for Hesper

High-performance CPU vector operations using SIMD intrinsics (AVX2/NEON).

This module provides FFI bindings to optimized C++ implementations that leverage:
- **AVX2** on x86_64 (8 floats per operation)
- **NEON** on ARM64 (4 floats per operation)
- **Scalar fallback** on other architectures

## Example

```lean
let a := FloatArray.mk #[1.0, 2.0, 3.0, 4.0]
let b := FloatArray.mk #[5.0, 6.0, 7.0, 8.0]
let c := simdAdd a b  -- [6.0, 8.0, 10.0, 12.0]
```
-/

namespace Hesper.Simd

/-- Get information about the SIMD backend being used -/
@[extern "lean_simd_backend_info"]
opaque backendInfo : IO String

/--
SIMD-optimized vector addition (FFI binding).

**Warning:** This is the raw FFI function. Use `simdAdd` instead for safety checks.
-/
@[extern "lean_simd_add_f64"]
private opaque simdAddF64Unsafe (a b : @& FloatArray) : FloatArray

/--
SIMD-optimized Float64 vector addition: c = a + b

Adds two float arrays element-wise using SIMD instructions when available.

**Safety:** Returns empty array if input sizes don't match.

**Performance:**
- AVX2 (x86_64): 4 doubles/operation
- NEON (ARM64): 2 doubles/operation
- OpenMP multithreading for arrays ≥10K elements
-/
def simdAdd (a b : FloatArray) : FloatArray :=
  if a.size != b.size then
    -- Safety: return empty array on size mismatch
    FloatArray.mk #[]
  else
    simdAddF64Unsafe a b

/--
Naive (non-SIMD) vector addition for benchmarking comparison.

Implemented in pure Lean using Array.mapIdx.
-/
def naiveAdd (a b : FloatArray) : FloatArray :=
  if a.size != b.size then
    FloatArray.mk #[]
  else
    let result := a.data.mapIdx fun i x => x + b.data[i]!
    FloatArray.mk result

/--
Verify two float arrays are equal within epsilon tolerance.

Used for correctness verification of SIMD operations.
-/
def verifyEqual (a b : FloatArray) (epsilon : Float := 1e-6) : Bool :=
  if a.size != b.size then
    false
  else
    (Array.range a.size).all fun i =>
      let diff := (a.data[i]! - b.data[i]!).abs
      diff < epsilon

end Hesper.Simd
