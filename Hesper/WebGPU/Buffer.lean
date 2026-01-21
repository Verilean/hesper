import Hesper.WebGPU.Types

namespace Hesper.WebGPU

/-- Buffer descriptor for creating GPU buffers -/
structure BufferDescriptor where
  size : USize              -- Size in bytes
  usage : List BufferUsage  -- Usage flags
  mappedAtCreation : Bool   -- Whether to map at creation
  deriving Inhabited

/-- Create a GPU buffer.
    Resources are automatically cleaned up by Lean's GC via External finalizers. -/
@[extern "lean_hesper_create_buffer"]
opaque createBufferImpl (device : @& Device) (desc : @& BufferDescriptor) : IO Buffer

/-- Wrapper with debug output -/
def createBuffer (device : @& Device) (desc : @& BufferDescriptor) : IO Buffer := do
  IO.println s!"[Lean] createBuffer: size={desc.size}, usage={desc.usage.length} items, mapped={desc.mappedAtCreation}"
  createBufferImpl device desc

/-- Write data to a buffer from the CPU.
    @param buffer The target buffer
    @param offset Offset in bytes
    @param data Pointer to source data (ByteArray)
-/
@[extern "lean_hesper_write_buffer"]
opaque writeBuffer (device : @& Device) (buffer : @& Buffer) (offset : USize) (data : @& ByteArray) : IO Unit

/-- Map a buffer for reading.
    Returns the mapped data as a ByteArray.
    @param buffer The buffer to map
    @param offset Offset in bytes
    @param size Size in bytes to map
-/
@[extern "lean_hesper_map_buffer_read"]
opaque mapBufferRead (device : @& Device) (buffer : @& Buffer) (offset : USize) (size : USize) : IO ByteArray

/-- Unmap a previously mapped buffer -/
@[extern "lean_hesper_unmap_buffer"]
opaque unmapBuffer (buffer : @& Buffer) : IO Unit

/-- Helper: Convert Float array to ByteArray for buffer upload -/
def floatArrayToBytes (arr : Array Float) : ByteArray :=
  let bytes := ByteArray.empty
  arr.foldl (fun (acc : ByteArray) (f : Float) =>
    -- Convert float to bytes (little-endian)
    let bits : UInt64 := f.toBits
    let b0 := bits.toUInt8
    let b1 := (bits >>> 8).toUInt8
    let b2 := (bits >>> 16).toUInt8
    let b3 := (bits >>> 24).toUInt8
    acc.push b0 |>.push b1 |>.push b2 |>.push b3
  ) bytes

/-- Helper: Convert ByteArray to Float array after buffer readback -/
def bytesToFloatArray (bytes : ByteArray) : Array Float :=
  let numFloats := bytes.size / 4
  Array.range numFloats |>.map fun i =>
    let offset := i * 4
    let b0 := bytes.get! offset
    let b1 := bytes.get! (offset + 1)
    let b2 := bytes.get! (offset + 2)
    let b3 := bytes.get! (offset + 3)
    let bits : UInt64 := b0.toUInt64 ||| (b1.toUInt64 <<< 8) ||| (b2.toUInt64 <<< 16) ||| (b3.toUInt64 <<< 24)
    Float.ofBits bits

end Hesper.WebGPU
