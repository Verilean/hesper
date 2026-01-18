/-!
# Basic Utilities

Utility functions and helpers for the Hesper library.
-/

namespace Hesper.Basic

/-- Convert a Float to 4 bytes (little-endian) -/
def floatToBytes (f : Float) : ByteArray :=
  -- Use bit representation of float as UInt64 (Lean's Float is 64-bit)
  -- For 32-bit float representation, we take the lower 32 bits
  let bits : UInt64 := f.toBits
  let bits32 := bits.toNat &&& 0xFFFFFFFF
  let b0 := UInt8.ofNat (bits32 &&& 0xFF)
  let b1 := UInt8.ofNat ((bits32 >>> 8) &&& 0xFF)
  let b2 := UInt8.ofNat ((bits32 >>> 16) &&& 0xFF)
  let b3 := UInt8.ofNat ((bits32 >>> 24) &&& 0xFF)
  ByteArray.mk #[b0, b1, b2, b3]

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
def floatArrayToBytes (arr : Array Float) : ByteArray :=
  arr.foldl (fun acc f => acc ++ floatToBytes f) ByteArray.empty

/-- Convert a byte array to an array of floats -/
def bytesToFloatArray (bytes : ByteArray) : Array Float :=
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

end Hesper.Basic
