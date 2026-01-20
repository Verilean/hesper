/-!
# Float16 (Half Precision) Array Support

Direct Float16 support using raw byte arrays.
**No automatic conversions** - all conversions must be explicit.

**Hardware Requirements:**
- ARM: Requires ARMv8.2-A with FP16 vector arithmetic
- x86_64: Requires F16C extension (Ivy Bridge+)
- **Returns error if hardware support is unavailable**

**Performance Benefits:**
- 4x memory savings vs Float64, 2x vs Float32
- SIMD width: NEON 8 halfs/op, AVX2+F16C 8 halfs/op
- Direct GPU tensor interop without conversion
-/

namespace Hesper.Float16

/--
Float16 array stored as raw bytes (2 bytes per element).
Works directly with C FFI and GPU without type conversion overhead.

**IMPORTANT:** We store `numElements` explicitly because Lean cannot properly
access `ByteArray.size` when the ByteArray is wrapped in a struct from C++.
-/
structure Float16Array where
  /-- Raw byte representation (2 bytes per float16) -/
  data : ByteArray
  /-- Number of Float16 elements (NOT bytes) -/
  numElements : Nat
  deriving Inhabited

/-- Get number of Float16 elements -/
def size (arr : Float16Array) : Nat :=
  arr.numElements

/-- Create empty Float16Array -/
def empty : Float16Array :=
  { data := ByteArray.empty, numElements := 0 }

/-- Create Float16Array from raw ByteArray (must be 2-byte aligned) -/
def fromBytes (bytes : ByteArray) : Option Float16Array :=
  if bytes.size % 2 == 0 then
    some { data := bytes, numElements := bytes.size / 2 }
  else
    none

/-- Check if FP16 hardware support is available -/
@[extern "lean_f16_hw_check"]
opaque hasHardwareSupport : IO Bool

/-- SIMD addition for Float16 arrays (requires hardware support) -/
@[extern "lean_simd_add_f16"]
private opaque simdAddFFI (a b : @& ByteArray) : IO ByteArray

def simdAdd (a b : Float16Array) : IO Float16Array := do
  if size a != size b then
    throw (IO.userError "Float16 array size mismatch")

  let result ← simdAddFFI a.data b.data
  return { data := result, numElements := a.numElements }

/-- ONLY for explicit conversion when needed - converts Float64 to Float16 -/
@[extern "lean_f16_from_f64_array"]
opaque fromFloatArray (arr : @& FloatArray) : IO Float16Array

/-- ONLY for explicit conversion when needed - converts Float16 to Float64 -/
@[extern "lean_f16_to_f64_array"]
opaque toFloatArray (arr : @& Float16Array) : IO FloatArray

/-- Get single element (converts to Float64 only for access) -/
@[extern "lean_f16_get"]
opaque get (arr : @& Float16Array) (i : @& USize) : IO Float

/-- Set single element (converts from Float64) -/
@[extern "lean_f16_set"]
opaque set (arr : Float16Array) (i : USize) (val : Float) : IO Float16Array

def toString (arr : Float16Array) : IO String := do
  let n := min (size arr) 8
  let mut elements := []
  for i in [0:n] do
    let val ← get arr i.toUSize
    elements := elements ++ [s!"{val}"]
  let rest := if size arr > 8 then s!", ... ({size arr - 8} more)" else ""
  return s!"Float16[{size arr}]: [{String.intercalate ", " elements}{rest}]"

end Hesper.Float16
