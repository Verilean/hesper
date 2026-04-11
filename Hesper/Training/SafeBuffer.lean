import Hesper.Backend
import Hesper.Basic

/-!
# Safe Buffer Operations

Type-safe buffer read/write utilities with runtime bounds checking.
Backend-agnostic via `[GPUBackend β]`.
-/

namespace Hesper.Training.SafeBuffer

open Hesper

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

/-- Safely read a Float32 from a ByteArray at the given byte offset. -/
def readF32 (bytes : ByteArray) (offset : Nat) : Float :=
  Hesper.Basic.float32BitsToFloat64 (readU32 bytes offset)

/-- Safely read N Float32 values starting at byte offset 0. -/
def readF32Array (bytes : ByteArray) (n : Nat) : Array Float := Id.run do
  let mut result := #[]
  for i in [:n] do
    result := result.push (readF32 bytes (i * 4))
  return result

/-- Safely read a GPU buffer and return Float32 values. -/
@[inline]
def safeMapBufferReadF32 [GPUBackend β] (ctx : β) (buf : GPUBackend.Buf β)
    (numElements : Nat) : IO (Array Float) := do
  let byteSize := (numElements * 4).toUSize
  let bytes ← GPUBackend.readBuffer ctx buf byteSize
  if bytes.size < numElements * 4 then
    IO.eprintln s!"[SafeBuffer] WARNING: readBuffer returned {bytes.size} bytes, expected {numElements * 4}"
    return #[]
  return readF32Array bytes numElements

/-- Safely read a single Float32 from a GPU buffer at element index. -/
@[inline]
def safeReadF32 [GPUBackend β] (ctx : β) (buf : GPUBackend.Buf β)
    (elementIdx : Nat := 0) : IO Float := do
  let bytes ← GPUBackend.readBuffer ctx buf ((elementIdx + 1) * 4).toUSize
  if bytes.size < (elementIdx + 1) * 4 then return 0.0
  return readF32 bytes (elementIdx * 4)

def isNaN (f : Float) : Bool := f != f

/-- Check if any value in a GPU buffer is NaN. -/
@[inline]
def hasNaN [GPUBackend β] (ctx : β) (buf : GPUBackend.Buf β)
    (numElements : Nat) : IO Bool := do
  let vals ← safeMapBufferReadF32 ctx buf numElements
  return vals.any isNaN

end Hesper.Training.SafeBuffer
