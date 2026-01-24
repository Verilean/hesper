/-!
# Basic Utilities

Utility functions and helpers for the Hesper library.
-/

namespace Hesper.Basic

/-- Convert Float64 (Lean Float) to 4 bytes (little-endian f32)
    Uses FFI to properly convert f64→f32→bytes -/
@[extern "lean_hesper_float64_to_bytes"]
opaque floatToBytes (f : @& Float) : IO ByteArray

/-- Convert 4 bytes (little-endian) to Float -/
def bytesToFloat (bytes : ByteArray) (offset : Nat := 0) : Float :=
  if offset + 4 > bytes.size then 0.0
  else
    let b0 := bytes.get! offset |>.toNat
    let b1 := bytes.get! (offset + 1) |>.toNat
    let b2 := bytes.get! (offset + 2) |>.toNat
    let b3 := bytes.get! (offset + 3) |>.toNat
    let bits32 := b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24)
    -- Convert to UInt64 for Float.ofBits
    Float.ofBits (UInt64.ofNat bits32)

/-- Convert an array of floats to a byte array -/
def floatArrayToBytes (arr : Array Float) : IO ByteArray := do
  let mut bytes := ByteArray.empty
  for f in arr do
    let fb ← floatToBytes f
    bytes := bytes ++ fb
  return bytes

/-- Convert 4 bytes (little-endian f32) to Float (f64)
    Uses FFI to properly interpret f32 bits and convert to f64 -/
@[extern "lean_hesper_bytes_to_float64"]
opaque bytesToFloat64FFI (bytes : @& ByteArray) (offset : UInt32) : IO Float

/-- Convert 4 bytes (little-endian f32) to Float (f64)
    Properly handles f32→f64 conversion via FFI -/
def bytesToFloat32 (bytes : ByteArray) (offset : Nat := 0) : IO Float :=
  if offset + 4 > bytes.size then pure 0.0
  else bytesToFloat64FFI bytes offset.toUInt32

/-- Convert a byte array of f32 (GPU 32-bit floats) to Array Float (Lean 64-bit doubles)
    Uses FFI to properly convert each f32 to f64 -/
def bytesToFloatArray (bytes : ByteArray) : IO (Array Float) := do
  let numFloats := bytes.size / 4
  let mut result := Array.mkEmpty numFloats
  for i in [0:numFloats] do
    let f ← bytesToFloat32 bytes (i * 4)
    result := result.push f
  return result

/-- Pure version (fallback implementation using Lean)
    Note: This doesn't properly handle f32→f64 conversion! Use the FFI version above. -/
def bytesToFloatArrayPure (bytes : ByteArray) : Array Float :=
  let numFloats := bytes.size / 4
  Array.range numFloats |>.map fun i => bytesToFloat bytes (i * 4)

/-- Convert UInt32 to 4 bytes (little-endian) -/
def uint32ToBytes (n : UInt32) : ByteArray :=
  let b0 := UInt8.ofNat (n.toNat &&& 0xFF)
  let b1 := UInt8.ofNat ((n.toNat >>> 8) &&& 0xFF)
  let b2 := UInt8.ofNat ((n.toNat >>> 16) &&& 0xFF)
  let b3 := UInt8.ofNat ((n.toNat >>> 24) &&& 0xFF)
  ByteArray.mk #[b0, b1, b2, b3]

/-- Convert 4 bytes (little-endian) to UInt32 -/
def bytesToUInt32 (bytes : ByteArray) (offset : Nat := 0) : UInt32 :=
  if offset + 4 > bytes.size then 0
  else
    let b0 := bytes.get! offset |>.toNat
    let b1 := bytes.get! (offset + 1) |>.toNat
    let b2 := bytes.get! (offset + 2) |>.toNat
    let b3 := bytes.get! (offset + 3) |>.toNat
    let n := b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24)
    UInt32.ofNat n

/-- Get high-precision system time in nanoseconds.
    Used for benchmarking GPU performance. -/
@[extern "lean_hesper_get_time_ns"]
opaque getTimeNs : IO UInt64

end Hesper.Basic
