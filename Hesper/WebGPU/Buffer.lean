import Hesper.WebGPU.Types
import Hesper.Basic
import Hesper.Logging
import Hesper.WGSL.JSTrace

namespace Hesper.WebGPU

/-- Buffer descriptor for creating GPU buffers -/
structure BufferDescriptor where
  size : USize              -- Size in bytes
  usage : List BufferUsage  -- Usage flags
  mappedAtCreation : Bool   -- Whether to map at creation
  deriving Inhabited

/-- Early alias of `getBufferId` (declared before first use; same FFI symbol). -/
@[extern "lean_hesper_buffer_id"]
opaque getBufferIdEarly (buffer : @& Buffer) : IO UInt64

/-- JS-trace registry: uid → (buffer, size). Populated only when DG_TRACE_JS
is set (keeps the buffers alive for the post-save dump — acceptable in a
trace run). -/
initialize jsTraceRegistryRef :
    IO.Ref (Std.HashMap UInt64 (Buffer × Nat)) ← IO.mkRef {}

/-- Create a GPU buffer.
    Resources are automatically cleaned up by Lean's GC via External finalizers. -/
@[extern "lean_hesper_create_buffer"]
opaque createBufferImpl (device : @& Device) (desc : @& BufferDescriptor) : IO Buffer

/-- Wrapper with debug output -/
def createBuffer (device : @& Device) (desc : @& BufferDescriptor) : IO Buffer := do
  Hesper.Logging.logVerbose s!"[Lean] createBuffer: size={desc.size}, usage={desc.usage.length} items, mapped={desc.mappedAtCreation}"
  let buf ← createBufferImpl device desc
  if ← Hesper.WGSL.JSTrace.enabled then
    let uid ← getBufferIdEarly buf
    Hesper.WGSL.JSTrace.bufCreated uid desc.size.toNat
    jsTraceRegistryRef.modify (·.insert uid (buf, desc.size.toNat))
  return buf

/-- Write data to a buffer from the CPU.
    @param buffer The target buffer
    @param offset Offset in bytes
    @param data Pointer to source data (ByteArray)
-/
@[extern "lean_hesper_write_buffer"]
opaque writeBufferImpl (device : @& Device) (buffer : @& Buffer) (offset : USize) (data : @& ByteArray) : IO Unit

/-- writeBuffer with an optional JS-trace hook (DG_TRACE_JS): records small
writes (params, canvases) with contents so a replayer can reproduce them. -/
def writeBuffer (device : @& Device) (buffer : @& Buffer) (offset : USize) (data : @& ByteArray) : IO Unit := do
  if ← Hesper.WGSL.JSTrace.armed then
    Hesper.WGSL.JSTrace.write (← getBufferIdEarly buffer) offset.toNat data
  writeBufferImpl device buffer offset data

/-- Map a buffer for reading.
    Returns the mapped data as a ByteArray.
    @param buffer The buffer to map
    @param offset Offset in bytes
    @param size Size in bytes to map
-/
@[extern "lean_hesper_map_buffer_read"]
opaque mapBufferReadImpl (device : @& Device) (buffer : @& Buffer) (offset : USize) (size : USize) : IO ByteArray

/-- mapBufferRead with a JS-trace hook: readbacks are the replayer's sync
points (logits → CPU commit logic). -/
def mapBufferRead (device : @& Device) (buffer : @& Buffer) (offset : USize) (size : USize) : IO ByteArray := do
  if ← Hesper.WGSL.JSTrace.armed then
    Hesper.WGSL.JSTrace.read (← getBufferIdEarly buffer) offset.toNat size.toNat
  mapBufferReadImpl device buffer offset size

/-- Unmap a previously mapped buffer -/
@[extern "lean_hesper_unmap_buffer"]
opaque unmapBuffer (buffer : @& Buffer) : IO Unit

/-- Get a stable unique identifier for a GPU buffer (raw WGPUBuffer handle as UInt64).
    Used for bind group caching — same ID means same underlying GPU buffer. -/
@[extern "lean_hesper_buffer_id"]
opaque getBufferId (buffer : @& Buffer) : IO UInt64

/-- metal_replacer STEP 2: the underlying MTLBuffer of this Dawn buffer (via reinterpret to metal::Buffer
    + GetMTLBuffer). Reports its length/storageMode/contents — proves the buffer bridge for dispatching
    llama.cpp's Metal kernels on our data with no copies. See METAL_REPLACER_INTEGRATION.md. -/
@[extern "lean_hesper_mtl_buffer_probe"]
opaque mtlBufferProbe (buffer : @& Buffer) : IO String

/-- metal_replacer STEP 3: dispatch a CUSTOM Metal kernel (out[i] = in[i]*2) on our Dawn-backed MTLBuffers.
    Validates running a hand-written Metal kernel on our data end-to-end. Caller syncs the `in` write first
    (a mapBufferRead) and reads `out` after. See METAL_REPLACER_INTEGRATION.md. -/
@[extern "lean_hesper_metal_dispatch_mul2"]
opaque metalDispatchMul2 (device : @& Device) (inBuf : @& Buffer) (outBuf : @& Buffer) (n : UInt32) : IO Unit

/-- Hash an array of buffers into a single UInt64 key in one FFI call.
    Avoids N separate `getBufferId` calls per dispatch. -/
@[extern "lean_hesper_hash_buffer_array"]
opaque hashBufferArray (seed : UInt64) (buffers : @& Array Buffer) : IO UInt64

/-- Convert Float64 to Float32 IEEE 754 bits -/
def float64ToFloat32Bits (f : Float) : UInt32 :=
  let bits64 : UInt64 := f.toBits
  let sign64 := (bits64 >>> 63) &&& 1
  let exp64 := (bits64 >>> 52) &&& 0x7FF
  let mant64 := bits64 &&& 0x000FFFFFFFFFFFFF
  if exp64 == 0 then (0 : UInt32)
  else if exp64 == 0x7FF then
    (sign64.toUInt32 <<< 31) ||| ((0xFF : UInt32) <<< 23) ||| ((mant64 >>> 29).toUInt32 &&& (0x7FFFFF : UInt32))
  else
    let exp32val : Int := exp64.toNat - 1023 + 127
    if exp32val <= 0 then (0 : UInt32)
    else if exp32val >= 255 then (sign64.toUInt32 <<< 31) ||| ((0xFF : UInt32) <<< 23)
    else
      (sign64.toUInt32 <<< 31) ||| (exp32val.toNat.toUInt32 <<< 23) ||| ((mant64 >>> 29).toUInt32 &&& (0x7FFFFF : UInt32))

/-- Helper: Convert Float array to ByteArray for buffer upload (Float64 → Float32) -/
def floatArrayToBytes (arr : Array Float) : ByteArray :=
  arr.foldl (fun (acc : ByteArray) (f : Float) =>
    let bits := float64ToFloat32Bits f
    acc.push bits.toUInt8
       |>.push (bits >>> 8).toUInt8
       |>.push (bits >>> 16).toUInt8
       |>.push (bits >>> 24).toUInt8
  ) ByteArray.empty

/-- Helper: Convert ByteArray to Float array after buffer readback -/
def bytesToFloatArray (bytes : ByteArray) : Array Float :=
  let numFloats := bytes.size / 4
  Array.range numFloats |>.map fun i =>
    let offset := i * 4
    let b0 := bytes.get! offset
    let b1 := bytes.get! (offset + 1)
    let b2 := bytes.get! (offset + 2)
    let b3 := bytes.get! (offset + 3)
    let bits : UInt32 := b0.toUInt32 ||| (b1.toUInt32 <<< 8) ||| (b2.toUInt32 <<< 16) ||| (b3.toUInt32 <<< 24)
    Hesper.Basic.float32BitsToFloat64 bits

end Hesper.WebGPU

namespace Hesper.WebGPU

/-- DG_TRACE_JS_DUMP=1: after `JSTrace.save`, dump every referenced buffer
without provenance (derived weights: predequants, repacks — and activations,
harmless) as `b<uid>.bin` so the JS replayer can load them directly. -/
def jsTraceDumpMissing (device : Device) : IO Unit := do
  if (← IO.getEnv "DG_TRACE_JS_DUMP").isNone then return
  let some d ← Hesper.WGSL.JSTrace.outDirGet | return
  let reg ← jsTraceRegistryRef.get
  let miss ← Hesper.WGSL.JSTrace.missingUids
  let mut dumped := 0
  let mut bytes := 0
  for uid in miss do
    match reg[uid]? with
    | some (buf, size) =>
      let data ← mapBufferReadImpl device buf 0 size.toUSize
      unmapBuffer buf
      IO.FS.writeBinFile s!"{d}/b{uid}.bin" data
      dumped := dumped + 1
      bytes := bytes + size
    | none => pure ()
  IO.println s!"[JSTrace] dumped {dumped}/{miss.size} derived buffers ({bytes / 1000000} MB)"

end Hesper.WebGPU
