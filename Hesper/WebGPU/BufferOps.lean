import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.Basic
import Hesper.Logging

/-!
# GPU Buffer Operations

Utilities for transferring data between CPU and GPU.

## Memory Model

WebGPU uses explicit memory transfers:

```
CPU Memory              GPU Memory
┌─────────┐            ┌──────────┐
│ByteArray│  ────────► │  Buffer  │  writeBuffer (CPU → GPU)
└─────────┘            └──────────┘
                            │
                            ▼
                       [Compute Shader]
                            │
                            ▼
┌─────────┐            ┌──────────┐
│ByteArray│  ◄──────── │  Buffer  │  readBuffer  (GPU → CPU)
└─────────┘            └──────────┘
```

## Buffer Usage Flags

Buffers must be created with appropriate usage flags:

- `copyDst`: Can receive data from CPU
- `copySrc`: Can send data to CPU
- `storage`: Can be read/written by compute shaders
- `uniform`: Can be used as uniform buffer

**Example**:
```lean
-- For model weights (upload once)
usage := [.storage, .copyDst]

-- For output logits (download to CPU)
usage := [.storage, .copySrc]

-- For input tokens (upload + read in shader)
usage := [.storage, .copyDst]
```

## Asynchronous Operations

WebGPU operations are asynchronous:
1. Submit compute commands → Returns immediately
2. Use `queue.onSubmittedWorkDone()` to wait
3. Map buffer → Wait for GPU → Read data → Unmap

## References
- WebGPU Spec: https://www.w3.org/TR/webgpu/
- Buffer mapping: https://gpuweb.github.io/gpuweb/#buffer-mapping
-/

namespace Hesper.WebGPU.BufferOps

open Hesper.WebGPU
open Hesper.Logging (logVerbose)

/-! ## Data Type Conversion -/

/-- Convert Float32 to bytes (little-endian)

    @param f Float32 value
    @return 4 bytes representing the float
-/
def floatToBytes (f : Float) : IO ByteArray :=
  -- Use FFI for proper f64→f32→bytes conversion
  Hesper.Basic.floatToBytes f

/-- Alias for backward compatibility -/
def float32BitsToFloat64 := Hesper.Basic.float32BitsToFloat64

/-- Convert bytes to Float32 (little-endian)

    @param bytes ByteArray (must have at least offset+4 bytes)
    @param offset Start offset
    @return Float32 value
-/
def bytesToFloat (bytes : ByteArray) (offset : Nat) : Float :=
  if offset + 4 > bytes.size then
    0.0
  else
    let b0 := bytes[offset]!.toUInt32
    let b1 := bytes[offset + 1]!.toUInt32
    let b2 := bytes[offset + 2]!.toUInt32
    let b3 := bytes[offset + 3]!.toUInt32
    let bits := b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24)
    -- Convert IEEE 754 float32 bits to float64 bits
    float32BitsToFloat64 bits

/-- Convert UInt32 to bytes (little-endian)

    @param n UInt32 value
    @return 4 bytes
-/
def uint32ToBytes (n : UInt32) : ByteArray :=
  let b0 := (n &&& 0xFF).toUInt8
  let b1 := ((n >>> 8) &&& 0xFF).toUInt8
  let b2 := ((n >>> 16) &&& 0xFF).toUInt8
  let b3 := ((n >>> 24) &&& 0xFF).toUInt8
  let arr := ByteArray.empty
  let arr := arr.push b0
  let arr := arr.push b1
  let arr := arr.push b2
  arr.push b3

/-- Convert bytes to UInt32 (little-endian)

    @param bytes ByteArray
    @param offset Start offset
    @return UInt32 value
-/
def bytesToUInt32 (bytes : ByteArray) (offset : Nat) : UInt32 :=
  if offset + 4 > bytes.size then
    0
  else
    let b0 := bytes[offset]!.toUInt32
    let b1 := bytes[offset + 1]!.toUInt32
    let b2 := bytes[offset + 2]!.toUInt32
    let b3 := bytes[offset + 3]!.toUInt32
    b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24)

/-! ## Array Conversions -/

/-- Convert Array Float to ByteArray (Float32 format)

    @param arr Array of floats
    @return Packed ByteArray
-/
def floatArrayToBytes (arr : Array Float) : IO ByteArray := do
  let mut bytes := ByteArray.empty
  for f in arr do
    let fb ← floatToBytes f
    bytes := bytes ++ fb
  return bytes

/-- Convert ByteArray to Array Float

    @param bytes ByteArray (must be multiple of 4)
    @return Array of Float32 values
-/
def bytesToFloatArray (bytes : ByteArray) : Array Float :=
  let numFloats := bytes.size / 4
  let rec loop (idx : Nat) (acc : Array Float) : Array Float :=
    if idx >= numFloats then
      acc
    else
      let f := bytesToFloat bytes (idx * 4)
      loop (idx + 1) (acc.push f)
  loop 0 #[]

/-- Convert Array Nat (token IDs) to ByteArray (UInt32 format)

    @param arr Array of token IDs
    @return Packed ByteArray
-/
def tokenArrayToBytes (arr : Array Nat) : ByteArray :=
  arr.foldl (fun bytes n => bytes ++ uint32ToBytes n.toUInt32) ByteArray.empty

/-- Convert ByteArray to Array Nat (token IDs)

    @param bytes ByteArray (must be multiple of 4)
    @return Array of token IDs
-/
def bytesToTokenArray (bytes : ByteArray) : Array Nat :=
  let numTokens := bytes.size / 4
  let rec loop (idx : Nat) (acc : Array Nat) : Array Nat :=
    if idx >= numTokens then
      acc
    else
      let token := (bytesToUInt32 bytes (idx * 4)).toNat
      loop (idx + 1) (acc.push token)
  loop 0 #[]

/-! ## GPU Upload/Download -/

/-- Upload Float32 array to GPU buffer

    Creates a new buffer and uploads data.

    @param device WebGPU device
    @param data Array of Float32 values
    @return GPU buffer containing the data
-/
def uploadFloatArray (device : Device) (data : Array Float) : IO Buffer := do
  let bytes ← floatArrayToBytes data
  let bufSize := bytes.size.toUSize

  let buffer ← createBuffer device {
    size := bufSize
    usage := [.storage, .copyDst]
    mappedAtCreation := false
  }

  writeBuffer device buffer 0 bytes
  return buffer

/-- Upload token IDs to GPU buffer

    Creates a new buffer and uploads token data.

    @param device WebGPU device
    @param tokens Array of token IDs
    @return GPU buffer containing the tokens
-/
def uploadTokens (device : Device) (tokens : Array Nat) : IO Buffer := do
  let bytes := tokenArrayToBytes tokens
  let bufSize := bytes.size.toUSize

  let buffer ← createBuffer device {
    size := bufSize
    usage := [.storage, .copyDst, .copySrc]
    mappedAtCreation := false
  }

  writeBuffer device buffer 0 bytes
  return buffer

/-- Download Float32 array from GPU buffer

    Reads data from GPU buffer back to CPU.

    @param device WebGPU device
    @param buffer GPU buffer to read from
    @param numElements Number of Float32 elements to read
    @return Array of Float32 values
-/
def downloadFloatArray (device : Device) (buffer : Buffer) (numElements : Nat) : IO (Array Float) := do
  logVerbose s!"[BufferOps] Downloading {numElements} Float32 values from GPU..."

  -- Calculate size in bytes
  let sizeBytes := (numElements * 4).toUSize

  -- Use mapBufferRead to get data from GPU
  -- This creates a staging buffer, copies GPU→staging, maps it, and returns the data
  let bytes ← mapBufferRead device buffer 0 sizeBytes

  logVerbose s!"[BufferOps] Downloaded {bytes.size} bytes"

  -- Convert ByteArray to Array Float
  let floats := bytesToFloatArray bytes

  logVerbose s!"[BufferOps] Converted to {floats.size} Float32 values"

  return floats

/-- Download logits and extract last position

    Helper function for text generation: downloads logits from GPU
    and extracts the logits for the last token position.

    @param device WebGPU device
    @param logitsBuffer GPU buffer containing logits [batch, seq_len, vocab_size]
    @param batchSize Batch size
    @param seqLen Sequence length
    @param vocabSize Vocabulary size
    @return Logits for last position [vocab_size]
-/
def downloadLastLogits (device : Device) (logitsBuffer : Buffer)
                       (batchSize seqLen vocabSize : Nat) : IO (Array Float) := do
  -- Total elements in logits buffer
  let totalElements := batchSize * seqLen * vocabSize

  -- Download entire buffer
  let allLogits ← downloadFloatArray device logitsBuffer totalElements

  -- Extract last position for first batch item
  -- Last position starts at: (seqLen - 1) * vocabSize
  let lastPosStart := (seqLen - 1) * vocabSize
  let lastLogits := allLogits.extract lastPosStart (lastPosStart + vocabSize)

  return lastLogits

/-! ## Utilities -/

/-- Create buffer for model outputs (logits)

    @param device WebGPU device
    @param batchSize Batch size
    @param seqLen Sequence length
    @param vocabSize Vocabulary size
    @return Buffer sized for logits
-/
def createLogitsBuffer (device : Device) (batchSize seqLen vocabSize : Nat) : IO Buffer := do
  let numElements := batchSize * seqLen * vocabSize
  let bufSize := (numElements * 4).toUSize  -- Float32

  createBuffer device {
    size := bufSize
    usage := [.storage, .copySrc]  -- copySrc allows downloading to CPU
    mappedAtCreation := false
  }

/-- Print buffer info for debugging

    @param buffer GPU buffer
    @param name Descriptive name
-/
def printBufferInfo (buffer : Buffer) (name : String) : IO Unit := do
  logVerbose s!"[Buffer] {name}"
  -- Would print size, usage flags, etc. if accessible
  -- For now, just placeholder
  pure ()

end Hesper.WebGPU.BufferOps
