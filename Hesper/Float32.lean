/-!
# Float32 Array Support

Direct Float32 (single precision) support using raw byte arrays.
**No automatic conversions** - all conversions must be explicit.

**Performance Benefits:**
- 2x memory savings vs Float64
- SIMD width: AVX2 8 vs 4, NEON 4 vs 2
- Direct GPU/FFI interop without conversion overhead
-/

namespace Hesper.Float32

/--
Float32 array stored as raw bytes (4 bytes per element).
Works directly with C FFI without type conversion overhead.

**IMPORTANT:** We store `numElements` explicitly because Lean cannot properly
access `ByteArray.size` when the ByteArray is wrapped in a struct from C++.
-/
structure Float32Array where
  /-- Raw byte representation (4 bytes per float32) -/
  data : ByteArray
  /-- Number of Float32 elements (NOT bytes) -/
  numElements : Nat
  deriving Inhabited

/-- Get number of Float32 elements -/
def size (arr : Float32Array) : Nat :=
  arr.numElements

/-- Create empty Float32Array -/
def empty : Float32Array :=
  { data := ByteArray.empty, numElements := 0 }

/-- Create Float32Array from raw ByteArray (must be 4-byte aligned) -/
def fromBytes (bytes : ByteArray) : Option Float32Array :=
  if bytes.size % 4 == 0 then
    some { data := bytes, numElements := bytes.size / 4 }
  else
    none

/-- SIMD addition for Float32 arrays -/
@[extern "lean_simd_add_f32"]
private opaque simdAddFFI (a b : @& ByteArray) : ByteArray

def simdAdd (a b : Float32Array) : Float32Array :=
  if size a != size b then
    empty
  else
    { data := simdAddFFI a.data b.data, numElements := a.numElements }

/-- ONLY for explicit conversion when needed - converts Float64 to Float32 -/
@[extern "lean_f32_from_f64_array"]
opaque fromFloatArray (arr : @& FloatArray) : Float32Array

/-- ONLY for explicit conversion when needed - converts Float32 to Float64 -/
@[extern "lean_f32_to_f64_array"]
opaque toFloatArray (arr : @& Float32Array) : FloatArray

/-- Get single element (converts to Float64 only for access) -/
@[extern "lean_f32_get"]
opaque get (arr : @& Float32Array) (i : @& USize) : Float

/-- Set single element (converts from Float64) -/
@[extern "lean_f32_set"]
opaque set (arr : Float32Array) (i : USize) (val : Float) : Float32Array

instance : ToString Float32Array where
  toString arr :=
    let n := min (size arr) 8
    let elements := (List.range n).map fun i => s!"{get arr i.toUSize}"
    let rest := if size arr > 8 then s!", ... ({size arr - 8} more)" else ""
    s!"Float32[{size arr}]: [{String.intercalate ", " elements}{rest}]"

end Hesper.Float32
