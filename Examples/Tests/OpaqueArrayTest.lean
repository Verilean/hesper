import Hesper.Core.Float32Array
import Hesper.Core.Float16Array

/-!
# Opaque Array Type Test

Demonstrates zero-copy, in-place mutable arrays for efficient GPU interop.

This test validates:
1. Creating opaque Float32Array and Float16Array (allocated in C++ heap)
2. In-place mutations (no array copying!)
3. Getting values with precision conversion
4. Zero-copy pointer access for GPU operations
5. SIMD operations
6. Format conversions (Float32 ↔ Float16)
-/

namespace Examples.Tests.OpaqueArrayTest

open Hesper.Core

def main : IO Unit := do
  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║  Opaque Float32Array Test                    ║"
  IO.println "╚══════════════════════════════════════════════╝\n"

  -- Test 1: Create and modify array
  IO.println "Test 1: Create and in-place modification"
  IO.println "─────────────────────────────────────────────"

  let arr ← Float32Array.create 10
  IO.println s!"✓ Created Float32Array with {arr.size} elements"

  -- Set some values (in-place mutation, no copying!)
  arr.set 0 3.14159
  arr.set 1 2.71828
  arr.set 2 1.41421
  arr.set 3 1.73205
  IO.println "✓ Set values using in-place mutation (no array copy)"

  -- Get values back
  let v0 ← arr.get 0
  let v1 ← arr.get 1
  let v2 ← arr.get 2
  let v3 ← arr.get 3

  IO.println s!"  arr[0] = {v0} (expected: ~3.14159)"
  IO.println s!"  arr[1] = {v1} (expected: ~2.71828)"
  IO.println s!"  arr[2] = {v2} (expected: ~1.41421)"
  IO.println s!"  arr[3] = {v3} (expected: ~1.73205)"

  -- Test 2: GPU interop (zero-copy pointer access)
  IO.println "\nTest 2: Zero-copy GPU interop"
  IO.println "─────────────────────────────────────────────"

  let ptr ← arr.ptr
  let byteSize ← arr.byteSize
  IO.println s!"✓ Raw pointer: 0x{Nat.toDigits 16 ptr.toNat}"
  IO.println s!"✓ Byte size: {byteSize} bytes ({arr.size} * 4 bytes/float)"
  IO.println "  This pointer can be passed directly to wgpuQueueWriteBuffer!"
  IO.println "  Zero copy - no data conversion overhead"

  -- Test 3: Conversions
  IO.println "\nTest 3: Conversions to/from FloatArray"
  IO.println "─────────────────────────────────────────────"

  let floatArr := #[1.0, 2.0, 3.0, 4.0, 5.0]
  let arr2 ← Float32Array.fromFloatArray floatArr
  IO.println s!"✓ Created Float32Array from FloatArray of size {arr2.size}"

  let back ← arr2.toFloatArray
  IO.println s!"✓ Converted back to FloatArray: {back}"

  -- Test 4: SIMD operations
  IO.println "\nTest 4: SIMD element-wise operations"
  IO.println "─────────────────────────────────────────────"

  let a ← Float32Array.fromFloatArray #[1.0, 2.0, 3.0, 4.0]
  let b ← Float32Array.fromFloatArray #[10.0, 20.0, 30.0, 40.0]

  -- SIMD addition
  let sum ← Float32Array.simdAdd a b
  let sumArr ← sum.toFloatArray
  IO.println s!"✓ SIMD Add: [1,2,3,4] + [10,20,30,40] = {sumArr}"
  IO.println "  Expected: #[11.0, 22.0, 33.0, 44.0]"

  -- SIMD multiplication
  let prod ← Float32Array.simdMul a b
  let prodArr ← prod.toFloatArray
  IO.println s!"✓ SIMD Mul: [1,2,3,4] * [10,20,30,40] = {prodArr}"
  IO.println "  Expected: #[10.0, 40.0, 90.0, 160.0]"

  -- Test 5: toString
  IO.println "\nTest 5: String representation"
  IO.println "─────────────────────────────────────────────"
  let str ← arr.toString
  IO.println s!"✓ {str}"

  -- Test 6: Float16Array
  IO.println "\n╔══════════════════════════════════════════════╗"
  IO.println "║  Float16Array Test (FP16)                    ║"
  IO.println "╚══════════════════════════════════════════════╝\n"

  IO.println "Test 6.1: Float16Array creation and access"
  IO.println "─────────────────────────────────────────────"

  let f16arr ← Float16Array.create 10
  IO.println s!"✓ Created Float16Array with {f16arr.size} elements"

  -- Set values (note: FP16 has reduced precision ~3 decimal digits)
  f16arr.set 0 3.14159  -- Will be rounded to ~3.14
  f16arr.set 1 2.71828  -- Will be rounded to ~2.72
  f16arr.set 2 1000.5   -- Large values work
  f16arr.set 3 0.00001  -- Small values may underflow

  let v0 ← f16arr.get 0
  let v1 ← f16arr.get 1
  let v2 ← f16arr.get 2
  let v3 ← f16arr.get 3

  IO.println s!"  f16arr[0] = {v0} (input: 3.14159, FP16 precision ~3 digits)"
  IO.println s!"  f16arr[1] = {v1} (input: 2.71828)"
  IO.println s!"  f16arr[2] = {v2} (input: 1000.5)"
  IO.println s!"  f16arr[3] = {v3} (input: 0.00001, may underflow)"

  IO.println "\nTest 6.2: Float16 memory efficiency"
  IO.println "─────────────────────────────────────────────"
  let f16size ← f16arr.byteSize
  let f32size ← arr.byteSize  -- From Test 1
  IO.println s!"✓ Float16 byte size: {f16size} bytes (10 * 2 bytes/half)"
  IO.println s!"✓ Float32 byte size: {f32size} bytes (10 * 4 bytes/float)"
  IO.println s!"✓ Memory savings: {f32size.toNat / f16size.toNat}x smaller than Float32"

  IO.println "\nTest 6.3: Float32 ↔ Float16 conversions"
  IO.println "─────────────────────────────────────────────"
  let f32test ← Float32Array.fromFloatArray #[1.0, 2.0, 3.0, 4.0]
  let f16converted ← Float16Array.fromFloat32Array f32test
  let f32back : Float32Array ← f16converted.toFloat32Array
  let values ← Float32Array.toFloatArray f32back
  IO.println s!"✓ Float32 → Float16 → Float32: {values}"
  IO.println "  (Minor rounding differences expected due to FP16 precision)"

  IO.println "\nTest 6.4: Float16 SIMD operations"
  IO.println "─────────────────────────────────────────────"
  let a16 ← Float16Array.fromFloatArray #[10.0, 20.0, 30.0, 40.0]
  let b16 ← Float16Array.fromFloatArray #[1.0, 2.0, 3.0, 4.0]
  let sum16 ← Float16Array.simdAdd a16 b16
  let sumArr16 ← sum16.toFloatArray
  IO.println s!"✓ FP16 SIMD Add: [10,20,30,40] + [1,2,3,4] = {sumArr16}"

  IO.println "\n═════════════════════════════════════════════"
  IO.println "✅ All opaque array tests complete!"
  IO.println "═════════════════════════════════════════════"
  IO.println "\nKey Benefits Demonstrated:"
  IO.println "  1. Zero-copy GPU interop (raw pointer access)"
  IO.println "  2. In-place mutations (no array copying)"
  IO.println "  3. Memory efficient (native C++ std::vector)"
  IO.println "  4. SIMD operations (hardware accelerated)"
  IO.println "  5. Seamless Lean ↔ C++ data flow"
  IO.println "  6. Multiple precision formats (FP32, FP16)"
  IO.println "  7. Easy format conversions for mixed-precision workflows"

end Examples.Tests.OpaqueArrayTest

def main : IO Unit := Examples.Tests.OpaqueArrayTest.main
