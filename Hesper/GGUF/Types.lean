/-!
# GGUF Format Type Definitions

GGUF (Generic Graph Universal Format) is a binary format for storing large language models.
This module defines the core data structures for the GGUF v3 format.

## Format Structure
```
[Header]        - Magic number, version, counts
[Metadata]      - Key-value pairs (architecture, hyperparameters)
[Tensor Info]   - Tensor descriptors (name, shape, type, offset)
[Alignment]     - Padding to align tensor data
[Tensor Data]   - Raw binary blobs
```

## Reference
- Specification: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- Implementation: llama.cpp/gguf-py/gguf/gguf_reader.py
-/

namespace Hesper.GGUF

/-- GGUF file magic number: 0x46554747 ("GGUF" in ASCII) -/
def GGUF_MAGIC : UInt32 := 0x46554747

/-- Supported GGUF version -/
def GGUF_VERSION : UInt32 := 3

/-- Default alignment for tensor data (32 bytes) -/
def GGUF_DEFAULT_ALIGNMENT : UInt32 := 32

/-- GGML tensor types (quantization formats) -/
inductive GGMLType where
  | F32     : GGMLType  -- 32-bit float
  | F16     : GGMLType  -- 16-bit float
  | Q4_0    : GGMLType  -- 4-bit quantization (type 0)
  | Q4_1    : GGMLType  -- 4-bit quantization (type 1)
  | Q5_0    : GGMLType  -- 5-bit quantization (type 0)
  | Q5_1    : GGMLType  -- 5-bit quantization (type 1)
  | Q8_0    : GGMLType  -- 8-bit quantization
  | Q8_1    : GGMLType  -- 8-bit quantization (type 1)
  | Q2_K    : GGMLType  -- 2-bit K-quantization
  | Q3_K    : GGMLType  -- 3-bit K-quantization
  | Q4_K    : GGMLType  -- 4-bit K-quantization
  | Q5_K    : GGMLType  -- 5-bit K-quantization
  | Q6_K    : GGMLType  -- 6-bit K-quantization
  | Q8_K    : GGMLType  -- 8-bit K-quantization
  | IQ2_XXS : GGMLType  -- 2-bit quantization (extra extra small)
  | IQ2_XS  : GGMLType  -- 2-bit quantization (extra small)
  | IQ3_XXS : GGMLType  -- 3-bit quantization (extra extra small)
  | IQ1_S   : GGMLType  -- 1-bit quantization (small)
  | IQ4_NL  : GGMLType  -- 4-bit quantization (non-linear)
  | IQ3_S   : GGMLType  -- 3-bit quantization (small)
  | IQ2_S   : GGMLType  -- 2-bit quantization (small)
  | IQ4_XS  : GGMLType  -- 4-bit quantization (extra small)
  | I8      : GGMLType  -- 8-bit integer
  | I16     : GGMLType  -- 16-bit integer
  | I32     : GGMLType  -- 32-bit integer
  | I64     : GGMLType  -- 64-bit integer
  | F64     : GGMLType  -- 64-bit float
  | IQ1_M   : GGMLType  -- 1-bit quantization (medium)
  | BF16    : GGMLType  -- bfloat16
  | Q4_0_4_4 : GGMLType -- 4-bit quantization (4x4 blocks)
  | Q4_0_4_8 : GGMLType -- 4-bit quantization (4x8 blocks)
  | Q4_0_8_8 : GGMLType -- 4-bit quantization (8x8 blocks)
  | TQ1_0   : GGMLType  -- Ternary 1-bit (BitNet) {-1, 0, 1}
  | TQ2_0   : GGMLType  -- Ternary 2-bit (BitNet) {-1, 0, 1}
  | IQ4_NL_4_4 : GGMLType -- 4-bit non-linear quantization (4x4 blocks)
  | IQ4_NL_4_8 : GGMLType -- 4-bit non-linear quantization (4x8 blocks)
  | IQ4_NL_8_8 : GGMLType -- 4-bit non-linear quantization (8x8 blocks)
  | MXFP4   : GGMLType  -- MXFP4 (1 block)
  deriving Repr, BEq, Inhabited

namespace GGMLType

/-- Convert GGMLType to its numeric ID used in GGUF files -/
def toUInt32 : GGMLType → UInt32
  | .F32     => 0
  | .F16     => 1
  | .Q4_0    => 2
  | .Q4_1    => 3
  | .Q5_0    => 6
  | .Q5_1    => 7
  | .Q8_0    => 8
  | .Q8_1    => 9
  | .Q2_K    => 10
  | .Q3_K    => 11
  | .Q4_K    => 12
  | .Q5_K    => 13
  | .Q6_K    => 14
  | .Q8_K    => 15
  | .IQ2_XXS => 16
  | .IQ2_XS  => 17
  | .IQ3_XXS => 18
  | .IQ1_S   => 19
  | .IQ4_NL  => 20
  | .IQ3_S   => 21
  | .IQ2_S   => 22
  | .IQ4_XS  => 23
  | .I8      => 24
  | .I16     => 25
  | .I32     => 26
  | .I64     => 27
  | .F64     => 28
  | .IQ1_M   => 29
  | .BF16    => 30
  | .Q4_0_4_4 => 31
  | .Q4_0_4_8 => 32
  | .Q4_0_8_8 => 33
  | .TQ1_0   => 34
  | .TQ2_0   => 35
  | .IQ4_NL_4_4 => 36
  | .IQ4_NL_4_8 => 37
  | .IQ4_NL_8_8 => 38
  | .MXFP4   => 39

/-- Parse GGMLType from numeric ID -/
def fromUInt32 (id : UInt32) : Except String GGMLType :=
  match id with
  | 0  => .ok .F32
  | 1  => .ok .F16
  | 2  => .ok .Q4_0
  | 3  => .ok .Q4_1
  | 6  => .ok .Q5_0
  | 7  => .ok .Q5_1
  | 8  => .ok .Q8_0
  | 9  => .ok .Q8_1
  | 10 => .ok .Q2_K
  | 11 => .ok .Q3_K
  | 12 => .ok .Q4_K
  | 13 => .ok .Q5_K
  | 14 => .ok .Q6_K
  | 15 => .ok .Q8_K
  | 16 => .ok .IQ2_XXS
  | 17 => .ok .IQ2_XS
  | 18 => .ok .IQ3_XXS
  | 19 => .ok .IQ1_S
  | 20 => .ok .IQ4_NL
  | 21 => .ok .IQ3_S
  | 22 => .ok .IQ2_S
  | 23 => .ok .IQ4_XS
  | 24 => .ok .I8
  | 25 => .ok .I16
  | 26 => .ok .I32
  | 27 => .ok .I64
  | 28 => .ok .F64
  | 29 => .ok .IQ1_M
  | 30 => .ok .BF16
  | 31 => .ok .Q4_0_4_4
  | 32 => .ok .Q4_0_4_8
  | 33 => .ok .Q4_0_8_8
  | 34 => .ok .TQ1_0
  | 35 => .ok .TQ2_0
  | 36 => .ok .IQ4_NL_4_4
  | 37 => .ok .IQ4_NL_4_8
  | 38 => .ok .IQ4_NL_8_8
  | 39 => .ok .MXFP4
  | _  => .error s!"Unknown GGML type ID: {id}"

/-- Check if type is a ternary (BitNet) quantization format -/
def isTernary : GGMLType → Bool
  | .TQ1_0 => true
  | .TQ2_0 => true
  | _     => false

/-- Get the bytes per element for unquantized types -/
def bytesPerElement : GGMLType → Nat
  | .F32  => 4
  | .F16  => 2
  | .BF16 => 2
  | .I8   => 1
  | .I16  => 2
  | .I32  => 4
  | .I64  => 8
  | .F64  => 8
  | _    => 0  -- Quantized types don't have fixed bytes per element

end GGMLType

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
    | .Q2_K => "Q2_K"
    | .Q3_K => "Q3_K"
    | .Q4_K => "Q4_K"
    | .Q5_K => "Q5_K"
    | .Q6_K => "Q6_K"
    | .Q8_K => "Q8_K"
    | .IQ2_XXS => "IQ2_XXS"
    | .IQ2_XS => "IQ2_XS"
    | .IQ3_XXS => "IQ3_XXS"
    | .IQ1_S => "IQ1_S"
    | .IQ4_NL => "IQ4_NL"
    | .IQ3_S => "IQ3_S"
    | .IQ2_S => "IQ2_S"
    | .IQ4_XS => "IQ4_XS"
    | .I8 => "I8"
    | .I16 => "I16"
    | .I32 => "I32"
    | .I64 => "I64"
    | .F64 => "F64"
    | .IQ1_M => "IQ1_M"
    | .BF16 => "BF16"
    | .Q4_0_4_4 => "Q4_0_4_4"
    | .Q4_0_4_8 => "Q4_0_4_8"
    | .Q4_0_8_8 => "Q4_0_8_8"
    | .TQ1_0 => "TQ1_0"
    | .TQ2_0 => "TQ2_0"
    | .IQ4_NL_4_4 => "IQ4_NL_4_4"
    | .IQ4_NL_4_8 => "IQ4_NL_4_8"
    | .IQ4_NL_8_8 => "IQ4_NL_8_8"
    | .MXFP4 => "MXFP4"

/-- Metadata value types in GGUF -/
inductive MetadataValueType where
  | MUInt8   : MetadataValueType
  | MInt8    : MetadataValueType
  | MUInt16  : MetadataValueType
  | MInt16   : MetadataValueType
  | MUInt32  : MetadataValueType
  | MInt32   : MetadataValueType
  | MFloat32 : MetadataValueType
  | MBool    : MetadataValueType
  | MString  : MetadataValueType
  | MArray   : MetadataValueType
  | MUInt64  : MetadataValueType
  | MInt64   : MetadataValueType
  | MFloat64 : MetadataValueType
  deriving Repr, BEq, Inhabited

instance : ToString MetadataValueType where
  toString
    | .MUInt8   => "UInt8"
    | .MInt8    => "Int8"
    | .MUInt16  => "UInt16"
    | .MInt16   => "Int16"
    | .MUInt32  => "UInt32"
    | .MInt32   => "Int32"
    | .MFloat32 => "Float32"
    | .MBool    => "Bool"
    | .MString  => "String"
    | .MArray   => "Array"
    | .MUInt64  => "UInt64"
    | .MInt64   => "Int64"
    | .MFloat64 => "Float64"

/-- Convert MetadataValueType to its numeric ID -/
def MetadataValueType.toNat (t : MetadataValueType) : Nat :=
  match t with
  | .MUInt8   => 0
  | .MInt8    => 1
  | .MUInt16  => 2
  | .MInt16   => 3
  | .MUInt32  => 4
  | .MInt32   => 5
  | .MFloat32 => 6
  | .MBool    => 7
  | .MString  => 8
  | .MArray   => 9
  | .MUInt64  => 10
  | .MInt64   => 11
  | .MFloat64 => 12

/-- Parse MetadataValueType from numeric ID -/
def MetadataValueType.fromNat (id : Nat) : Except String MetadataValueType :=
  match id with
  | 0  => .ok .MUInt8
  | 1  => .ok .MInt8
  | 2  => .ok .MUInt16
  | 3  => .ok .MInt16
  | 4  => .ok .MUInt32
  | 5  => .ok .MInt32
  | 6  => .ok .MFloat32
  | 7  => .ok .MBool
  | 8  => .ok .MString
  | 9  => .ok .MArray
  | 10 => .ok .MUInt64
  | 11 => .ok .MInt64
  | 12 => .ok .MFloat64
  | _  => .error s!"Unknown metadata value type: {id}"

/-- Metadata value (simplified - stores raw bytes for now) -/
structure MetadataValue where
  valueType : MetadataValueType
  data      : ByteArray
  deriving Inhabited

instance : Repr MetadataValue where
  reprPrec mv _ := s!"MetadataValue({mv.valueType}, {mv.data.size} bytes)"

/-- GGUF file header -/
structure GGUFHeader where
  magic            : UInt32  -- Must be GGUF_MAGIC
  version          : UInt32  -- Must be GGUF_VERSION (3)
  tensorCount      : UInt64  -- Number of tensors in file
  metadataKVCount  : UInt64  -- Number of metadata key-value pairs
  deriving Repr, Inhabited

/-- Tensor information (metadata about a single tensor) -/
structure TensorInfo where
  name       : String      -- Tensor name (e.g., "blk.0.attn_q.weight")
  nDims      : UInt32      -- Number of dimensions
  dimensions : Array UInt64 -- Shape of tensor [dim0, dim1, ...]
  ggmlType   : GGMLType    -- Data type (F32, TQ2_0, etc.)
  offset     : UInt64      -- Byte offset in data section
  deriving Repr, Inhabited

namespace TensorInfo

/-- Calculate total number of elements in tensor -/
def numElements (ti : TensorInfo) : UInt64 :=
  ti.dimensions.foldl (· * ·) 1

/-- Calculate size in bytes (for quantized types, this is approximate) -/
def sizeBytes (ti : TensorInfo) : UInt64 :=
  let elems := numElements ti
  match ti.ggmlType with
  | .F32    => elems * 4
  | .F16    => elems * 2
  | .BF16   => elems * 2
  | .I8     => elems
  | .I16    => elems * 2
  | .I32    => elems * 4
  | .I64    => elems * 8
  | .F64    => elems * 8
  | .Q4_0   => (elems * 18) / 32  -- 32 elements per block, 18 bytes per block
  | .Q4_1   => (elems * 20) / 32  -- 32 elements per block, 20 bytes per block
  | .Q8_0   => (elems * 34) / 32  -- 32 elements per block, 34 bytes per block
  | .TQ2_0  => (elems / 4) + ((elems / 256) * 2)  -- 4 elements per byte + scale (FP16) per 256
  | .TQ1_0  => (elems * 5) / 3 + ((elems / 256) * 2)  -- Base-3 encoding + scales
  | _       => elems  -- Fallback (not accurate for all types)

end TensorInfo

/-- Complete GGUF file structure -/
structure GGUFFile where
  header    : GGUFHeader
  metadata  : Array (String × MetadataValue)  -- Key-value pairs
  tensors   : Array TensorInfo
  dataBlob  : ByteArray  -- Raw tensor data (single allocation)
  alignment : UInt32
  deriving Inhabited

instance : Repr GGUFFile where
  reprPrec gf _ := s!"GGUFFile(tensors: {gf.tensors.size}, data: {gf.dataBlob.size} bytes)"

namespace GGUFFile

/-- Find tensor by name -/
def findTensor (gf : GGUFFile) (name : String) : Option TensorInfo :=
  gf.tensors.find? (·.name == name)

/-- Get tensor names -/
def tensorNames (gf : GGUFFile) : Array String :=
  gf.tensors.map (·.name)

/-- Find metadata value by key -/
def findMetadata (gf : GGUFFile) (key : String) : Option MetadataValue :=
  gf.metadata.find? (·.1 == key) |>.map (·.2)

/-- Extract tensor data as ByteArray -/
def getTensorData (gf : GGUFFile) (ti : TensorInfo) : ByteArray :=
  let startOffset := ti.offset.toNat
  let size := ti.sizeBytes.toNat
  gf.dataBlob.extract startOffset (startOffset + size)

end GGUFFile

end Hesper.GGUF
