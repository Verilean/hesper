import Hesper.GGUF.Types

/-!
# GGUF Binary Parser

Binary parser for GGUF v3 format files.
Uses single ByteArray with offset tracking (following gemma.hs pattern for efficiency).

## Design Principles
1. **Single Buffer Allocation**: Pin ByteArray once, pass offsets
2. **Zero-Copy**: Extract sub-arrays without copying when possible
3. **Bounds Checking**: Return Except for all parsing operations
4. **Little-Endian**: All integers are stored in little-endian format

## References
- gemma.hs/src/Gemma/GGUF.hs - Memory pinning strategy
- llama.cpp/gguf-py/gguf/gguf_reader.py - Format specification
-/

namespace Hesper.GGUF.Parser

/-! ## Low-level Binary Reading Utilities -/

/-- Read UInt8 from ByteArray at offset -/
def readUInt8 (data : ByteArray) (offset : Nat) : Except String (UInt8 × Nat) :=
  if offset >= data.size then
    .error s!"readUInt8: offset {offset} out of bounds (size: {data.size})"
  else
    .ok (data.get! offset, offset + 1)

/-- Read UInt16 (little-endian) from ByteArray -/
def readUInt16LE (data : ByteArray) (offset : Nat) : Except String (UInt16 × Nat) :=
  if offset + 2 > data.size then
    .error s!"readUInt16LE: offset {offset} out of bounds"
  else
    let b0 := data.get! offset
    let b1 := data.get! (offset + 1)
    let val := b0.toUInt16 ||| (b1.toUInt16 <<< 8)
    .ok (val, offset + 2)

/-- Read UInt32 (little-endian) from ByteArray -/
def readUInt32LE (data : ByteArray) (offset : Nat) : Except String (UInt32 × Nat) :=
  if offset + 4 > data.size then
    .error s!"readUInt32LE: offset {offset} out of bounds"
  else
    let b0 := data.get! offset
    let b1 := data.get! (offset + 1)
    let b2 := data.get! (offset + 2)
    let b3 := data.get! (offset + 3)
    let val := b0.toUInt32 ||| (b1.toUInt32 <<< 8) |||
               (b2.toUInt32 <<< 16) ||| (b3.toUInt32 <<< 24)
    .ok (val, offset + 4)

/-- Read UInt64 (little-endian) from ByteArray -/
def readUInt64LE (data : ByteArray) (offset : Nat) : Except String (UInt64 × Nat) :=
  if offset + 8 > data.size then
    .error s!"readUInt64LE: offset {offset} out of bounds"
  else
    let b0 := data.get! offset
    let b1 := data.get! (offset + 1)
    let b2 := data.get! (offset + 2)
    let b3 := data.get! (offset + 3)
    let b4 := data.get! (offset + 4)
    let b5 := data.get! (offset + 5)
    let b6 := data.get! (offset + 6)
    let b7 := data.get! (offset + 7)
    let val := b0.toUInt64 ||| (b1.toUInt64 <<< 8) |||
               (b2.toUInt64 <<< 16) ||| (b3.toUInt64 <<< 24) |||
               (b4.toUInt64 <<< 32) ||| (b5.toUInt64 <<< 40) |||
               (b6.toUInt64 <<< 48) ||| (b7.toUInt64 <<< 56)
    .ok (val, offset + 8)

/-- Read Int32 (little-endian) from ByteArray -/
def readInt32LE (data : ByteArray) (offset : Nat) : Except String (Int32 × Nat) := do
  let (uval, newOffset) ← readUInt32LE data offset
  .ok (uval.toInt32, newOffset)

/-- Read Int64 (little-endian) from ByteArray -/
def readInt64LE (data : ByteArray) (offset : Nat) : Except String (Int64 × Nat) := do
  let (uval, newOffset) ← readUInt64LE data offset
  .ok (uval.toInt64, newOffset)

/-- Read Float32 (IEEE 754, little-endian) from ByteArray -/
def readFloat32LE (data : ByteArray) (offset : Nat) : Except String (Float × Nat) := do
  let (bits, newOffset) ← readUInt32LE data offset
  -- Convert UInt32 bits to Float using bit reinterpretation
  -- Simplified: just convert as integer (not proper IEEE 754)
  .ok (bits.toNat.toFloat, newOffset)  -- Placeholder

/-- Read string (length-prefixed) from ByteArray -/
def readString (data : ByteArray) (offset : Nat) : Except String (String × Nat) := do
  let (len, offset1) ← readUInt64LE data offset
  if offset1 + len.toNat > data.size then
    .error s!"readString: string length {len} exceeds bounds"
  else
    let strBytes := data.extract offset1 (offset1 + len.toNat)
    match String.fromUTF8? strBytes with
    | some str => .ok (str, offset1 + len.toNat)
    | none     => .error "readString: invalid UTF-8 encoding"

/-! ## GGUF Header Parsing -/

/-- Parse GGUF header (24 bytes) -/
def parseHeader (data : ByteArray) : Except String (GGUFHeader × Nat) := do
  if data.size < 24 then
    .error s!"File too small for GGUF header: {data.size} bytes"

  let (magic, offset1) ← readUInt32LE data 0
  if magic ≠ GGUF_MAGIC then
    .error s!"Invalid GGUF magic: 0x{magic.toNat.toDigits 16} (expected 0x{GGUF_MAGIC.toNat.toDigits 16})"

  let (version, offset2) ← readUInt32LE data offset1
  if version ≠ GGUF_VERSION then
    .error s!"Unsupported GGUF version: {version} (expected {GGUF_VERSION})"

  let (tensorCount, offset3) ← readUInt64LE data offset2
  let (metadataKVCount, offset4) ← readUInt64LE data offset3

  .ok ({ magic, version, tensorCount, metadataKVCount }, offset4)

/-! ## Metadata Parsing -/

/-- Parse a single metadata value (simplified - stores raw bytes) -/
partial def parseMetadataValue (data : ByteArray) (offset : Nat) : Except String (MetadataValue × Nat) := do
  let (typeId, offset1) ← readUInt32LE data offset
  let valueType ← MetadataValueType.fromNat typeId.toNat

  match valueType with
  | .MUInt8 => do
    let (val, offset2) ← readUInt8 data offset1
    let bytes := ByteArray.mk #[val]
    .ok ({ valueType, data := bytes }, offset2)

  | .MInt8 => do
    let (val, offset2) ← readUInt8 data offset1
    let bytes := ByteArray.mk #[val]
    .ok ({ valueType, data := bytes }, offset2)

  | .MUInt16 => do
    let (_, offset2) ← readUInt16LE data offset1
    let bytes := data.extract offset1 offset2
    .ok ({ valueType, data := bytes }, offset2)

  | .MInt16 => do
    let (_, offset2) ← readUInt16LE data offset1
    let bytes := data.extract offset1 offset2
    .ok ({ valueType, data := bytes }, offset2)

  | .MUInt32 => do
    let (_, offset2) ← readUInt32LE data offset1
    let bytes := data.extract offset1 offset2
    .ok ({ valueType, data := bytes }, offset2)

  | .MInt32 => do
    let (_, offset2) ← readInt32LE data offset1
    let bytes := data.extract offset1 offset2
    .ok ({ valueType, data := bytes }, offset2)

  | .MUInt64 => do
    let (_, offset2) ← readUInt64LE data offset1
    let bytes := data.extract offset1 offset2
    .ok ({ valueType, data := bytes }, offset2)

  | .MInt64 => do
    let (_, offset2) ← readInt64LE data offset1
    let bytes := data.extract offset1 offset2
    .ok ({ valueType, data := bytes }, offset2)

  | .MFloat32 => do
    let (_, offset2) ← readFloat32LE data offset1
    let bytes := data.extract offset1 offset2
    .ok ({ valueType, data := bytes }, offset2)

  | .MFloat64 => do
    -- Read 8 bytes for Float64
    if offset1 + 8 > data.size then
      .error "parseMetadataValue: Float64 out of bounds"
    let bytes := data.extract offset1 (offset1 + 8)
    .ok ({ valueType, data := bytes }, offset1 + 8)

  | .MBool => do
    let (val, offset2) ← readUInt8 data offset1
    let bytes := ByteArray.mk #[val]
    .ok ({ valueType, data := bytes }, offset2)

  | .MString => do
    let (str, offset2) ← readString data offset1
    let bytes := str.toUTF8
    .ok ({ valueType, data := bytes }, offset2)

  | .MArray => do
    -- Array: type (UInt32) + length (UInt64) + elements
    let (arrTypeId, offset2) ← readUInt32LE data offset1
    let (arrLen, offset3) ← readUInt64LE data offset2
    let arrType ← MetadataValueType.fromNat arrTypeId.toNat

    -- Calculate size based on element type and skip over array data
    let elementSize := match arrType with
      | .MUInt8 | .MInt8 | .MBool => 1
      | .MUInt16 | .MInt16 => 2
      | .MUInt32 | .MInt32 | .MFloat32 => 4
      | .MUInt64 | .MInt64 | .MFloat64 => 8
      | .MString => 0  -- Strings are variable length, need to parse each
      | .MArray => 0   -- Nested arrays not supported

    let mut currentOffset := offset3

    -- For string arrays, we need to skip each string
    if arrType == .MString then
      for _ in [0:arrLen.toNat] do
        let (_, newOffset) ← readString data currentOffset
        currentOffset := newOffset
    else if elementSize > 0 then
      -- Fixed-size elements: just skip
      currentOffset := offset3 + (arrLen.toNat * elementSize)
    else
      -- Unsupported array type (nested arrays)
      .error s!"Nested arrays not supported in metadata"

    -- Store array as raw bytes (simplified)
    let arrayBytes := data.extract offset3 currentOffset
    .ok ({ valueType := .MArray, data := arrayBytes }, currentOffset)

/-- Parse single metadata key-value pair -/
def parseMetadataKV (data : ByteArray) (offset : Nat) : Except String ((String × MetadataValue) × Nat) := do
  let (key, offset1) ← readString data offset
  let (value, offset2) ← parseMetadataValue data offset1
  .ok ((key, value), offset2)

/-- Parse all metadata key-value pairs -/
partial def parseAllMetadata (data : ByteArray) (offset : Nat) (count : UInt64) : Except String (Array (String × MetadataValue) × Nat) := do
  let rec loop (offset : Nat) (remaining : Nat) (acc : Array (String × MetadataValue)) : Except String (Array (String × MetadataValue) × Nat) := do
    if remaining = 0 then
      .ok (acc, offset)
    else
      let ((key, val), newOffset) ← parseMetadataKV data offset
      loop newOffset (remaining - 1) (acc.push (key, val))

  loop offset count.toNat #[]

/-! ## Tensor Info Parsing -/

/-- Parse dimensions array -/
def parseDimensions (data : ByteArray) (offset : Nat) (nDims : UInt32) : Except String (Array UInt64 × Nat) := do
  let rec loop (offset : Nat) (remaining : Nat) (acc : Array UInt64) : Except String (Array UInt64 × Nat) := do
    if remaining = 0 then
      .ok (acc, offset)
    else
      let (dim, newOffset) ← readUInt64LE data offset
      loop newOffset (remaining - 1) (acc.push dim)

  loop offset nDims.toNat #[]

/-- Parse single tensor info -/
def parseTensorInfo (data : ByteArray) (offset : Nat) : Except String (TensorInfo × Nat) := do
  let (name, offset1) ← readString data offset
  let (nDims, offset2) ← readUInt32LE data offset1
  let (dimensions, offset3) ← parseDimensions data offset2 nDims
  let (typeId, offset4) ← readUInt32LE data offset3
  let ggmlType ← GGMLType.fromUInt32 typeId
  let (tensorOffset, offset5) ← readUInt64LE data offset4

  .ok ({ name, nDims, dimensions, ggmlType, offset := tensorOffset }, offset5)

/-- Parse all tensor infos -/
partial def parseAllTensorInfos (data : ByteArray) (offset : Nat) (count : UInt64) : Except String (Array TensorInfo × Nat) := do
  let rec loop (offset : Nat) (remaining : Nat) (acc : Array TensorInfo) : Except String (Array TensorInfo × Nat) := do
    if remaining = 0 then
      .ok (acc, offset)
    else
      let (tensorInfo, newOffset) ← parseTensorInfo data offset
      loop newOffset (remaining - 1) (acc.push tensorInfo)

  loop offset count.toNat #[]

/-! ## Complete GGUF File Parsing -/

/-- Calculate alignment padding -/
def alignmentPadding (offset : Nat) (alignment : UInt32) : Nat :=
  let alignNat := alignment.toNat
  let remainder := offset % alignNat
  if remainder = 0 then 0 else alignNat - remainder

/-- Parse complete GGUF file -/
def parseGGUF (data : ByteArray) : Except String GGUFFile := do
  -- Parse header
  let (header, offset1) ← parseHeader data

  -- Parse metadata
  let (metadata, offset2) ← parseAllMetadata data offset1 header.metadataKVCount

  -- Parse tensor infos
  let (tensors, offset3) ← parseAllTensorInfos data offset2 header.tensorCount

  -- Calculate alignment padding
  let alignment := GGUF_DEFAULT_ALIGNMENT
  let padding := alignmentPadding offset3 alignment
  let dataStart := offset3 + padding

  -- Extract data blob (rest of file)
  -- Some GGUF files (like vocab-only) have no tensor data
  let dataBlob := if dataStart > data.size || header.tensorCount == 0 then
    ByteArray.empty
  else
    data.extract dataStart data.size

  .ok { header, metadata, tensors, dataBlob, alignment }

end Hesper.GGUF.Parser
