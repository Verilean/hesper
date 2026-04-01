import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.Basic

/-!
# Safe Buffer Operations

Type-safe buffer read/write utilities that prevent out-of-bounds access
at compile time or with proper runtime checks.

## Design

Instead of `ByteArray.get!` (panics on OOB), we use:
- `readF32` with explicit bounds checking, returning `Option Float`
- `readF32D` with a default value for OOB
- `readU32` similarly

For GPU buffer reads, `safeMapBufferRead` validates the requested
size against the expected size.
-/

namespace Hesper.Training.SafeBuffer

open Hesper.WebGPU

/-- Safely read a UInt32 (4 bytes LE) from a ByteArray.
    Returns 0 if out of bounds. -/
def readU32 (bytes : ByteArray) (offset : Nat) : UInt32 :=
  if offset + 4 <= bytes.size then
    let b0 := bytes.get! offset |>.toUInt32
    let b1 := bytes.get! (offset + 1) |>.toUInt32
    let b2 := bytes.get! (offset + 2) |>.toUInt32
    let b3 := bytes.get! (offset + 3) |>.toUInt32
    b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24)
  else 0

/-- Safely read a Float32 from a ByteArray at the given byte offset.
    Returns 0.0 if out of bounds. -/
def readF32 (bytes : ByteArray) (offset : Nat) : Float :=
  Hesper.Basic.float32BitsToFloat64 (readU32 bytes offset)

/-- Safely read N Float32 values starting at byte offset 0.
    Returns array of exactly N values (0.0 for any OOB reads). -/
def readF32Array (bytes : ByteArray) (n : Nat) : Array Float := Id.run do
  let mut result := #[]
  for i in [:n] do
    result := result.push (readF32 bytes (i * 4))
  return result

/-- Safely read a GPU buffer and return Float32 values.
    Validates that the read size matches expected element count.
    Returns array of floats, or empty array on failure. -/
def safeMapBufferReadF32 (device : Device) (buf : Buffer) (numElements : Nat)
    : IO (Array Float) := do
  let byteSize := (numElements * 4).toUSize
  let bytes ← mapBufferRead device buf 0 byteSize
  if bytes.size < numElements * 4 then
    IO.eprintln s!"[SafeBuffer] WARNING: mapBufferRead returned {bytes.size} bytes, expected {numElements * 4}"
    return #[]
  return readF32Array bytes numElements

/-- Safely read a single Float32 from a GPU buffer at element index.
    Returns 0.0 on failure. -/
def safeReadF32 (device : Device) (buf : Buffer) (elementIdx : Nat := 0)
    : IO Float := do
  let bytes ← mapBufferRead device buf (elementIdx * 4).toUSize 4
  if bytes.size < 4 then return 0.0
  return readF32 bytes 0

/-- Check if a Float32 value is NaN -/
def isNaN (f : Float) : Bool := f != f

/-- Check if any value in a GPU buffer is NaN.
    Reads first N elements and checks each. -/
def hasNaN (device : Device) (buf : Buffer) (numElements : Nat) : IO Bool := do
  let vals ← safeMapBufferReadF32 device buf numElements
  return vals.any isNaN

end Hesper.Training.SafeBuffer
