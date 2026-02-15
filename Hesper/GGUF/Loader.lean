import Hesper.GGUF.Parser
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.Basic
import Hesper.Quantization.TQ2_0

/-!
# GGUF Tensor Loader

Loads tensor data from parsed GGUF files and uploads to GPU buffers.

## Architecture

```
GGUF File (on disk)
  │
  ├─► Parser → GGUFFile structure (in memory)
  │            ├─ Header
  │            ├─ Metadata
  │            ├─ Tensor Info (names, shapes, offsets)
  │            └─ Data Blob (raw bytes)
  │
  ├─► Loader → Extract specific tensors by name
  │            ├─ Lookup tensor info
  │            ├─ Extract ByteArray slice
  │            └─ Return (data, shape, quantization type)
  │
  └─► Upload → GPU Buffer
               ├─ If quantized (TQ2_0): Upload packed + scales
               └─ If Float32: Upload directly
```

## Tensor Naming Conventions

GGUF files use specific naming patterns (from llama.cpp):

### Embedding
```
token_embd.weight → [vocab_size, dim]
```

### Transformer Blocks (per layer N)
```
blk.N.attn_norm.weight → [dim]           (RMSNorm scales)
blk.N.attn_q.weight → [dim, dim]         (Q projection)
blk.N.attn_k.weight → [dim, kv_dim]      (K projection)
blk.N.attn_v.weight → [dim, kv_dim]      (V projection)
blk.N.attn_output.weight → [dim, dim]    (Output projection)

blk.N.ffn_norm.weight → [dim]            (RMSNorm scales)
blk.N.ffn_gate.weight → [dim, ffn_dim]   (Gate projection)
blk.N.ffn_up.weight → [dim, ffn_dim]     (Up projection)
blk.N.ffn_down.weight → [ffn_dim, dim]   (Down projection)
```

### Output
```
output_norm.weight → [dim]               (Final RMSNorm)
output.weight → [dim, vocab_size]        (LM head)
```

## Quantization Handling

Different tensors use different quantization:
- **TQ2_0 (ternary)**: Most weight matrices (BitLinear)
- **Float32**: RMSNorm scales, some small tensors
- **FP16**: Sometimes used for scales

We handle this by:
1. Check tensor's `ggml_type` field
2. Extract appropriate data format
3. Upload to GPU with correct interpretation

## Performance

**Zero-copy extraction**: Tensor data is a `ByteArray.extract` slice - no memory copying.

## References
- llama.cpp: `convert_hf_to_gguf.py` (tensor naming)
- llama.cpp: `ggml-quants.h` (quantization types)
- GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
-/

namespace Hesper.GGUF.Loader

open Hesper.GGUF.Parser
open Hesper.WebGPU
open Hesper.Quantization.TQ2_0

/-! ## Tensor Data Types -/

/-- GGML quantization type IDs (from ggml-quants.h) -/
inductive GGMLType where
  | F32     -- 0: Float32
  | F16     -- 1: Float16
  | Q4_0    -- 2: 4-bit quantization
  | Q4_1    -- 3: 4-bit quantization with offset
  | Q5_0    -- 6: 5-bit quantization
  | Q5_1    -- 7: 5-bit quantization with offset
  | Q8_0    -- 8: 8-bit quantization
  | Q8_1    -- 9: 8-bit quantization with offset
  | IQ2_XXS -- 14: 2-bit ternary (BitNet)
  | TQ2_0   -- 35: Ternary 2-bit (BitNet)
  | IQ4_NL_4_4 -- 36: 4-bit non-linear quantization (4x4 blocks)
  | IQ4_NL_4_8 -- 37: 4-bit non-linear quantization (4x8 blocks)
  | IQ4_NL_8_8 -- 38: 4-bit non-linear quantization (8x8 blocks)
  | MXFP4   -- 39: MXFP4 (1 block)
  | Unknown (id : Nat)
  deriving Repr

def GGMLType.fromNat (n : Nat) : GGMLType :=
  match n with
  | 0 => .F32
  | 1 => .F16
  | 2 => .Q4_0
  | 3 => .Q4_1
  | 6 => .Q5_0
  | 7 => .Q5_1
  | 8 => .Q8_0
  | 9 => .Q8_1
  | 14 => .IQ2_XXS  -- TQ2_0 ternary
  | id => .Unknown id

def GGMLType.bytesPerElement (t : GGMLType) : Nat :=
  match t with
  | .F32 => 4
  | .F16 => 2
  | .Q4_0 | .Q4_1 => 0  -- Block-based, varies
  | .Q5_0 | .Q5_1 => 0
  | .Q8_0 | .Q8_1 => 0
  | .IQ2_XXS => 0  -- Block-based: 256 elements per block
  | .TQ2_0 => 0    -- Ternary 2-bit
  | .IQ4_NL_4_4 => 0  -- i2_s format: 2 bits per element + scale
  | .IQ4_NL_4_8 => 0
  | .IQ4_NL_8_8 => 0
  | .MXFP4 => 0
  | .Unknown _ => 0

instance : ToString GGMLType where
  toString t := match t with
    | .F32 => "F32"
    | .F16 => "F16"
    | .Q4_0 => "Q4_0"
    | .Q4_1 => "Q4_1"
    | .Q5_0 => "Q5_0"
    | .Q5_1 => "Q5_1"
    | .Q8_0 => "Q8_0"
    | .Q8_1 => "Q8_1"
    | .IQ2_XXS => "IQ2_XXS"
    | .TQ2_0 => "TQ2_0"
    | .IQ4_NL_4_4 => "IQ4_NL_4_4"
    | .IQ4_NL_4_8 => "IQ4_NL_4_8"
    | .IQ4_NL_8_8 => "IQ4_NL_8_8"
    | .MXFP4 => "MXFP4"
    | .Unknown n => s!"Unknown({n})"

/-! ## Tensor Info -/

/-- Tensor metadata extracted from GGUF -/
structure TensorInfo where
  name : String
  shape : Array Nat  -- Dimensions [d0, d1, ...]
  ggmlType : GGMLType
  offset : Nat       -- Byte offset in data blob
  size : Nat         -- Size in bytes
  deriving Repr

/-! ## Tensor Extraction -/

/-- Find tensor by name in GGUF file

    @param gguf Parsed GGUF file
    @param name Tensor name (e.g., "blk.0.attn_q.weight")
    @return Tensor info if found
-/
-- Convert GGUF.GGMLType to Loader.GGMLType
def convertGGMLType (t : Hesper.GGUF.GGMLType) : GGMLType :=
  match t with
  | .F32 => .F32
  | .F16 => .F16
  | .Q4_0 => .Q4_0
  | .Q4_1 => .Q4_1
  | .Q5_0 => .Q5_0
  | .Q5_1 => .Q5_1
  | .Q8_0 => .Q8_0
  | .Q8_1 => .Q8_1
  | .IQ2_XXS => .IQ2_XXS
  | .TQ2_0 => .TQ2_0
  | .IQ4_NL_4_4 => .IQ4_NL_4_4
  | .IQ4_NL_4_8 => .IQ4_NL_4_8
  | .IQ4_NL_8_8 => .IQ4_NL_8_8
  | .MXFP4 => .MXFP4
  | _ => .Unknown 999

def findTensor (gguf : GGUFFile) (name : String) : Except String TensorInfo := do
  -- Search through tensor info array
  for idx in [:gguf.tensors.size] do
    let ti := gguf.tensors[idx]!
    if ti.name == name then
      let ggmlType := convertGGMLType ti.ggmlType

      -- Calculate size from offset difference for accurate results
      let size : Nat :=
        if idx + 1 < gguf.tensors.size then
          -- Next tensor's offset - current offset
          let nextOffset := gguf.tensors[idx + 1]!.offset.toNat
          let currentOffset := ti.offset.toNat
          nextOffset - currentOffset
        else
          -- Last tensor: use remaining data
          gguf.dataBlob.size - ti.offset.toNat

      return {
        name := ti.name,
        shape := ti.dimensions.map (·.toNat),
        ggmlType := ggmlType,
        offset := ti.offset.toNat,
        size := size
      }

  throw s!"Tensor '{name}' not found in GGUF file"

/-- Extract tensor data by name

    Returns raw bytes for the tensor. Caller must interpret based on ggmlType.

    @param gguf Parsed GGUF file
    @param name Tensor name
    @return (TensorInfo, ByteArray) - metadata and data
-/
def getTensorData (gguf : GGUFFile) (name : String) : Except String (TensorInfo × ByteArray) := do
  let info ← findTensor gguf name

  -- Extract data slice
  let dataStart := info.offset
  let dataEnd := info.offset + info.size

  if dataEnd > gguf.dataBlob.size then
    throw s!"Tensor '{name}' data extends beyond file: {dataEnd} > {gguf.dataBlob.size}"

  let data := gguf.dataBlob.extract dataStart dataEnd
  return (info, data)

/-- Extract Float32 tensor data

    Unpacks F32 data into Float array.

    @param gguf Parsed GGUF file
    @param name Tensor name
    @return Array of Float32 values
-/
-- Helper to create empty arrays (ByteArray.mkEmpty doesn't exist in this Lean version)
def mkEmptyByteArray : ByteArray := ByteArray.empty

def getFloat32Tensor (gguf : GGUFFile) (name : String) : Except String (Array Float) := do
  let (info, data) ← getTensorData gguf name

  match info.ggmlType with
  | .F32 =>
    -- Parse Float32 array
    let numElements := info.size / 4
    let mut result := #[]

    for i in [0:numElements] do
      let offset := i * 4
      if offset + 4 <= data.size then
        -- Read little-endian Float32
        let b0 := data.get! offset
        let b1 := data.get! (offset + 1)
        let b2 := data.get! (offset + 2)
        let b3 := data.get! (offset + 3)

        let bits := b0.toUInt32
                  ||| (b1.toUInt32 <<< 8)
                  ||| (b2.toUInt32 <<< 16)
                  ||| (b3.toUInt32 <<< 24)

        -- Convert IEEE 754 float32 bits to Lean Float (float64)
        let value := Hesper.Basic.float32BitsToFloat64 bits
        result := result.push value

    return result

  | _ => throw s!"Tensor '{name}' is not Float32 (type: {toString info.ggmlType})"

/-- Extract F16 (Float16) tensor data as ByteArray

    Returns raw F16 data as ByteArray (ready for GPU upload).

    @param gguf Parsed GGUF file
    @param name Tensor name
    @return ByteArray of F16 data
-/
def getF16Tensor (gguf : GGUFFile) (name : String) : Except String ByteArray := do
  let (info, data) ← getTensorData gguf name

  match info.ggmlType with
  | .F16 =>
    -- F16 format: 2 bytes per element
    return data
  | _ => throw s!"Tensor '{name}' is not F16 (type: {toString info.ggmlType})"

/-- Extract I2_S (BitNet ternary) tensor data

    Returns packed 2-bit ternary data and scale factor.
    I2_S format: 2 bits per weight {-1, 0, +1}
    Encoding: 00 → -1, 01 → 0, 10 → +1

    @param gguf Parsed GGUF file
    @param name Tensor name
    @return (packed_data: ByteArray, scale: Float, num_elements: Nat)
-/
def getI2_S_Tensor (gguf : GGUFFile) (name : String)
    : Except String (ByteArray × Float × Nat) := do
  let (info, data) ← getTensorData gguf name

  -- i2_s is stored as type 36 (IQ4_NL_4_4) in the model file
  match info.ggmlType with
  | .IQ4_NL_4_4 | .IQ4_NL_4_8 | .IQ4_NL_8_8 =>
    -- Calculate number of elements from shape
    let numElements := info.shape.foldl (· * ·) 1

    -- i2_s format: total blob = packedSize + 32 (where packedSize = numElements / 4)
    -- Scale (float32) is at offset packedSize, followed by 28 bytes of padding
    let packedSize := numElements / 4

    -- Debug: print tensor data size and expected sizes
    dbg_trace s!"  [DEBUG i2_s] tensor='{name}' shape={info.shape.toList} numElements={numElements} dataSize={data.size} packedSize={packedSize} expectedTQ2_0={(numElements/256)*66} ratio={data.size.toFloat/numElements.toFloat}"

    if data.size < packedSize + 4 then
      throw s!"Tensor '{name}' i2_s data too small: {data.size} < {packedSize + 4}"

    -- Extract packed data (first packedSize bytes)
    let packedData := data.extract 0 packedSize

    -- Extract scale at offset packedSize (NOT last 4 bytes - there's 28 bytes padding after)
    let scaleOffset := packedSize
    let b0 := data.get! scaleOffset
    let b1 := data.get! (scaleOffset + 1)
    let b2 := data.get! (scaleOffset + 2)
    let b3 := data.get! (scaleOffset + 3)

    let scaleBits := b0.toUInt32
                  ||| (b1.toUInt32 <<< 8)
                  ||| (b2.toUInt32 <<< 16)
                  ||| (b3.toUInt32 <<< 24)

    -- Convert bits to Float using Float.ofBits
    let scale := Hesper.Basic.float32BitsToFloat64 scaleBits

    -- Debug: print scale bytes and value
    dbg_trace s!"  [DEBUG i2_s] scaleOffset={scaleOffset} rawBytes=[{b0},{b1},{b2},{b3}] scaleBits=0x{String.mk (Nat.toDigits 16 scaleBits.toNat)} scale={scale}"

    return (packedData, scale, numElements)
  | _ => throw s!"Tensor '{name}' is not i2_s format (type: {toString info.ggmlType})"

/-- Extract TQ2_0 quantized tensor data

    Returns packed ternary data and FP16 scales for TQ2_0 tensors.

    @param gguf Parsed GGUF file
    @param name Tensor name
    @return (packed_data, scales_data, num_blocks)
-/
def getTQ2_0Tensor (gguf : GGUFFile) (name : String)
    : Except String (ByteArray × ByteArray × Nat) := do
  let (info, data) ← getTensorData gguf name

  match info.ggmlType with
  | .IQ2_XXS =>
    -- TQ2_0 format: blocks of 256 elements
    -- Each block: 64 bytes packed data + 2 bytes FP16 scale
    let blockSize := 66  -- 64 packed + 2 scale
    let numBlocks := info.size / blockSize

    if info.size % blockSize != 0 then
      throw s!"Tensor '{name}' has invalid TQ2_0 size: {info.size}"

    -- Separate packed data from scales
    let mut packedData := mkEmptyByteArray
    let mut scalesData := mkEmptyByteArray

    for i in [0:numBlocks] do
      let blockOffset := i * blockSize

      -- Extract 64 bytes of packed data
      for j in [0:64] do
        let byte := data.get! (blockOffset + j)
        packedData := packedData.push byte

      -- Extract 2 bytes of FP16 scale
      let scale0 := data.get! (blockOffset + 64)
      let scale1 := data.get! (blockOffset + 65)
      scalesData := scalesData.push scale0
      scalesData := scalesData.push scale1

    return (packedData, scalesData, numBlocks)

  | _ => throw s!"Tensor '{name}' is not TQ2_0 (type: {toString info.ggmlType})"

/-! ## GPU Upload -/

/-- Extract Float32 tensor as ByteArray

    @param gguf Parsed GGUF file
    @param name Tensor name
    @return ByteArray of Float32 data (ready for GPU upload)
-/
def extractFloat32Tensor (gguf : GGUFFile) (name : String) : IO ByteArray := do
  IO.println s!"[Loader] Extracting Float32 tensor: {name}"

  -- Extract raw bytes directly (no need to convert to Float and back)
  match getTensorData gguf name with
  | .error e => throw $ IO.userError e
  | .ok (info, data) =>
    match info.ggmlType with
    | .F32 =>
      let numFloats := data.size / 4
      IO.println s!"  ✓ Extracted {numFloats} Float32 values ({data.size} bytes)"
      return data
    | _ => throw $ IO.userError s!"Tensor '{name}' is not Float32 (type: {toString info.ggmlType})"

/-- Extract F16 tensor as ByteArray

    @param gguf Parsed GGUF file
    @param name Tensor name
    @return ByteArray of F16 data (ready for GPU upload)
-/
def extractF16Tensor (gguf : GGUFFile) (name : String) : IO ByteArray := do
  IO.println s!"[Loader] Extracting F16 tensor: {name}"

  match getF16Tensor gguf name with
  | .error e => throw $ IO.userError e
  | .ok data =>
    IO.println s!"  ✓ Extracted F16 tensor ({data.size} bytes)"
    return data

/-- Extract I2_S tensor and dequantize to Float32

    Dequantizes BitNet ternary weights to F32 for GPU computation.

    @param gguf Parsed GGUF file
    @param name Tensor name
    @return ByteArray of F32 dequantized data
-/
def extractI2_S_Tensor (gguf : GGUFFile) (name : String) : IO ByteArray := do
  IO.println s!"[Loader] Extracting i2_s tensor: {name}"

  match getI2_S_Tensor gguf name with
  | .error e => throw $ IO.userError e
  | .ok (packedData, scale, numElements) =>
    IO.println s!"  ✓ Extracted i2_s tensor: {numElements} elements, scale={scale}"

    -- Dequantize: unpack 2-bit values and convert to F32
    let mut float32Data := ByteArray.empty

    for i in [0:numElements] do
      -- Extract 2-bit value
      let byteIdx := i / 4
      let bitPos := (i % 4) * 2

      if byteIdx < packedData.size then
        let byte := packedData.get! byteIdx
        let val2bit := (byte.toNat >>> bitPos) &&& 0x03

        -- Decode to ternary {-1, 0, +1}
        let ternary : Float :=
          if val2bit == 0 then -1.0      -- 00 → -1
          else if val2bit == 1 then 0.0  -- 01 →  0
          else 1.0                        -- 10 → +1

        -- Apply scale
        let value := ternary * scale

        -- Convert F32 to bytes (little-endian) via FFI (proper f64→f32 conversion)
        let valueBytes ← Hesper.Basic.floatToBytes value
        float32Data := float32Data ++ valueBytes

    IO.println s!"  ✓ Dequantized to {float32Data.size} bytes F32"
    return float32Data

/-- Extract TQ2_0 quantized tensor as ByteArrays

    Returns two ByteArrays: packed data and scales.

    @param gguf Parsed GGUF file
    @param name Tensor name
    @return (packed_data, scales_data)
-/
def extractTQ2_0Tensor (gguf : GGUFFile) (name : String)
    : IO (ByteArray × ByteArray) := do
  IO.println s!"[Loader] Extracting TQ2_0 tensor: {name}"

  match getTQ2_0Tensor gguf name with
  | .error e => throw $ IO.userError e
  | .ok (packedData, scalesData, numBlocks) =>
    IO.println s!"  ✓ Extracted {numBlocks} TQ2_0 blocks ({packedData.size + scalesData.size} bytes)"
    return (packedData, scalesData)

/-! ## Utilities -/

/-- List all tensor names in GGUF file

    Useful for debugging and validation.

    @param gguf Parsed GGUF file
-/
def listTensors (gguf : GGUFFile) : IO Unit := do
  IO.println "═══════════════════════════════════════════════"
  IO.println s!"  Tensors in GGUF file ({gguf.tensors.size} total)"
  IO.println "═══════════════════════════════════════════════"

  for ti in gguf.tensors do
    let ggmlType := convertGGMLType ti.ggmlType
    let shapeStr := String.intercalate " × " ((ti.dimensions.map toString).toList)
    let sizeBytes := Hesper.GGUF.TensorInfo.sizeBytes ti
    IO.println s!"  {ti.name}"
    IO.println s!"    Shape: [{shapeStr}]"
    IO.println s!"    Type: {toString ggmlType}"
    IO.println s!"    Size: {sizeBytes} bytes"
    IO.println ""

/-- Validate tensor exists and has expected shape

    @param gguf Parsed GGUF file
    @param name Tensor name
    @param expectedShape Expected dimensions
    @return true if tensor exists with correct shape
-/
def validateTensor (gguf : GGUFFile) (name : String) (expectedShape : Array Nat)
    : Except String Unit := do
  let info ← findTensor gguf name

  if info.shape.size != expectedShape.size then
    throw s!"Tensor '{name}' has wrong rank: {info.shape.size} vs {expectedShape.size}"

  for i in [0:info.shape.size] do
    if info.shape[i]! != expectedShape[i]! then
      throw s!"Tensor '{name}' has wrong shape at dim {i}: {info.shape[i]!} vs {expectedShape[i]!}"

  return ()

end Hesper.GGUF.Loader
