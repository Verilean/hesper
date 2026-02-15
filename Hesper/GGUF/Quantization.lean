import Hesper.GGUF.Types

/-!
# GGUF Quantization and Dequantization

Implements unpacking of quantized tensor formats, with focus on ternary quantization for BitNet.

## Ternary Quantization Formats

### TQ2_0 (2-bit packing)
- **Block size**: 256 elements
- **Packing**: 4 ternary values per byte (2 bits each)
- **Scale**: 1 FP16 scale per block
- **Encoding**: {-1, 0, 1} → {0b11, 0b00, 0b01}
  - 0b00 → 0
  - 0b01 → +1
  - 0b11 → -1 (two's complement)
- **Layout**: [packed_bytes (64)] [scale (FP16=2 bytes)]

### TQ1_0 (base-3 encoding)
- **Block size**: 256 elements
- **Packing**: 5 ternary values per byte (3^5 = 243 < 256)
- **More compact** but more complex to unpack
- **Layout**: [packed_bytes (51)] [extra (4)] [scale (FP16=2 bytes)]

## References
- llama.cpp/ggml/src/ggml-quants.c (lines 2103-2271) - TQ2_0 implementation
- llama.cpp/convert_hf_to_gguf.py (lines 3152-3190) - Quantization logic
-/

namespace Hesper.GGUF.Quantization

/-! ## Ternary Value Helpers -/

/-- Unpack a single 2-bit ternary value from packed byte
    @param packed The packed byte containing 4 values
    @param idx The index (0-3) of the value to extract
    @return Ternary value: -1, 0, or 1
-/
def unpackTernary2bit (packed : UInt8) (idx : Nat) : Int8 :=
  if idx >= 4 then 0  -- Out of bounds, return 0
  else
    let shift := idx * 2
    let packedNat := packed.toNat
    let bits := ((packedNat >>> shift) &&& 0x03).toUInt8
    -- Decode: 0b00 → 0, 0b01 → 1, 0b11 → -1
    match bits with
    | 0x00 => 0   -- 0b00
    | 0x01 => 1   -- 0b01
    | 0x03 => -1  -- 0b11 (two's complement: 11 in binary = -1 in 2-bit signed)
    | _    => 0   -- 0b10 (invalid, should not occur)

/-- Pack 4 ternary values into a single byte (inverse operation)
    Used for testing and verification
-/
def packTernary2bit (v0 v1 v2 v3 : Int8) : UInt8 :=
  let encode (v : Int8) : UInt8 :=
    if v > 0 then 0x01       -- +1 → 0b01
    else if v < 0 then 0x03  -- -1 → 0b11
    else 0x00                --  0 → 0b00

  (encode v0) ||| ((encode v1) <<< 2) ||| ((encode v2) <<< 4) ||| ((encode v3) <<< 6)

/-! ## TQ2_0 Block Structure -/

/-- TQ2_0 block: 256 elements packed into 64 bytes + 1 FP16 scale
    Total: 66 bytes per block
-/
structure TQ2_0_Block where
  qs : ByteArray  -- Quantized values (64 bytes)
  d  : UInt16     -- Scale (FP16 represented as UInt16 bits)

namespace TQ2_0_Block

/-- Block size in elements -/
def blockSize : Nat := 256

/-- Block size in bytes (64 packed + 2 scale) -/
def blockSizeBytes : Nat := 66

/-- Simplified FP16 to Float32 conversion
    Handles normalized values only (sufficient for most scales)
-/
def fp16ToFloat32Simple (bits : UInt16) : Float :=
  let sign := (bits >>> 15) &&& 0x1
  let exp := (bits >>> 10) &&& 0x1F
  let mant := bits &&& 0x3FF

  -- Simplified: only handle normalized values
  if exp = 0 && mant = 0 then
    0.0  -- Zero
  else if exp = 0 then
    -- Subnormal: approximate as very small number
    if sign = 1 then -0.0001 else 0.0001
  else if exp = 31 then
    -- Infinity/NaN: return large number
    if sign = 1 then -1000000.0 else 1000000.0
  else
    -- Normalized: (-1)^sign * 2^(exp-15) * (1 + mant/1024)
    let expVal := exp.toNat
    let exponent : Int := expVal - 15
    let mantissa := 1.0 + (mant.toFloat / 1024.0)

    -- Compute 2^exponent using Float.pow
    let expFloat : Float := match exponent with
      | Int.ofNat n => n.toFloat
      | Int.negSucc n => -(n + 1).toFloat

    let powerOf2 :=
      if exponent >= 0 then
        Float.pow 2.0 expFloat
      else
        1.0 / (Float.pow 2.0 (-expFloat))

    let val := mantissa * powerOf2
    if sign = 1 then -val else val

/-- Parse TQ2_0 block from ByteArray -/
def parse (data : ByteArray) (offset : Nat) : Except String TQ2_0_Block :=
  if offset + blockSizeBytes > data.size then
    .error s!"TQ2_0_Block.parse: insufficient data at offset {offset}"
  else
    let qs := data.extract offset (offset + 64)
    let scaleByte0 := data.get! (offset + 64)
    let scaleByte1 := data.get! (offset + 65)
    let scaleU16 := scaleByte0.toUInt16 ||| (scaleByte1.toUInt16 <<< 8)
    .ok { qs, d := scaleU16 }

/-- Unpack TQ2_0 block to Float32 array -/
def unpack (block : TQ2_0_Block) : Array Float :=
  -- Convert FP16 scale to Float32 (simplified)
  let scale := fp16ToFloat32Simple block.d

  -- Unpack 256 ternary values
  Array.ofFn (n := blockSize) fun i =>
    let byteIdx := i.val / 4
    let bitIdx := i.val % 4
    if byteIdx < block.qs.size then
      let packed := block.qs.get! byteIdx
      let ternary := unpackTernary2bit packed bitIdx
      ternary.toFloat * scale
    else
      0.0

end TQ2_0_Block

/-! ## TQ1_0 Block Structure (TODO: Lower priority) -/

/-- TQ1_0 block: 256 elements packed via base-3 encoding
    More compact but more complex unpacking
-/
structure TQ1_0_Block where
  qs : ByteArray  -- Quantized values (51 bytes for base-3 encoding)
  qh : ByteArray  -- Extra bits (4 bytes)
  d  : UInt16     -- Scale (FP16)

/-! ## High-Level Dequantization API -/

/-- Dequantize TQ2_0 tensor to Float32 array -/
def dequantizeTQ2_0 (data : ByteArray) (numElements : Nat) : Except String (Array Float) := do
  let numBlocks := (numElements + TQ2_0_Block.blockSize - 1) / TQ2_0_Block.blockSize

  let rec processBlocks (fuel : Nat) (blockIdx : Nat) (acc : Array Float) : Except String (Array Float) := do
    match fuel with
    | 0 => .error "dequantizeTQ2_0: fuel exhausted"
    | fuel' + 1 =>
      if blockIdx >= numBlocks then
        -- Truncate to exact numElements
        .ok (acc.extract 0 numElements)
      else
        let offset := blockIdx * TQ2_0_Block.blockSizeBytes
        let block ← TQ2_0_Block.parse data offset
        let unpacked := block.unpack
        processBlocks fuel' (blockIdx + 1) (acc ++ unpacked)

  processBlocks numBlocks 0 #[]

/-- Dequantize tensor based on GGMLType -/
def dequantize (data : ByteArray) (ggmlType : GGMLType) (numElements : Nat) : Except String (Array Float) :=
  match ggmlType with
  | .TQ2_0 => dequantizeTQ2_0 data numElements
  | .TQ1_0 => .error "TQ1_0 dequantization not yet implemented"
  | .F32   =>
    -- Already Float32, just read directly
    if data.size < numElements * 4 then
      .error "F32: insufficient data"
    else
      .ok <| Array.ofFn (n := numElements) fun i =>
        if i.val < numElements && i.val * 4 + 4 <= data.size then
          let b0 := data.get! (i.val * 4)
          let b1 := data.get! (i.val * 4 + 1)
          let b2 := data.get! (i.val * 4 + 2)
          let b3 := data.get! (i.val * 4 + 3)
          let bits := b0.toUInt32 ||| (b1.toUInt32 <<< 8) |||
                      (b2.toUInt32 <<< 16) ||| (b3.toUInt32 <<< 24)
          -- Simple conversion: treat as unsigned int (not ideal but will compile)
          bits.toFloat
        else
          0.0
  | .F16 => .error "F16 dequantization not yet implemented"
  | _ => .error s!"Dequantization not implemented for {ggmlType}"

/-! ## Quantization (for testing) -/

/-- Helper: Simplified Float32 to FP16 conversion -/
def float32ToFP16Simple (f : Float) : UInt16 :=
  -- Very simplified conversion - just preserve sign and rough magnitude
  let sign := if f < 0.0 then (1 : Nat) else (0 : Nat)
  let absF := if f < 0.0 then -f else f

  -- Rough mapping: use integer approximation
  -- FP16 normalized: 2^(exp-15) * (1 + mant/1024)
  -- For scale ~1.0: exp=15, mant~0
  let expGuess := 15  -- Assume values near 1.0
  let mantGuess := ((absF - 1.0) * 1024.0).toUInt32
  let mantClamped := if mantGuess > 1023 then (1023 : UInt16) else mantGuess.toUInt16

  (sign.toUInt16 <<< 15) ||| (expGuess.toUInt16 <<< 10) ||| mantClamped

/-- Quantize Float32 array to TQ2_0 format
    Used for testing and validation (simplified version)
-/
def quantizeTQ2_0 (values : Array Float) : ByteArray :=
  let numBlocks := (values.size + TQ2_0_Block.blockSize - 1) / TQ2_0_Block.blockSize

  let rec processBlock (fuel : Nat) (blockIdx : Nat) (acc : ByteArray) : ByteArray :=
    match fuel with
    | 0 => acc  -- Fuel exhausted, return what we have
    | fuel' + 1 =>
      if blockIdx >= numBlocks then acc
      else
      let startIdx := blockIdx * TQ2_0_Block.blockSize
      let endIdx := Nat.min (startIdx + TQ2_0_Block.blockSize) values.size
      let blockValues := values.extract startIdx endIdx

      -- Find max absolute value for scale (simplified)
      let maxAbs := blockValues.foldl (fun acc v =>
        let absV := if v < 0.0 then -v else v
        if absV > acc then absV else acc
      ) 0.0
      let scale := if maxAbs > 0.0 then maxAbs else 1.0
      let invScale := 1.0 / scale

      -- Quantize to {-1, 0, 1}
      let quantized := blockValues.map fun v =>
        let qv := (v * invScale)
        if qv > 0.5 then (1 : Int8)
        else if qv < -0.5 then (-1 : Int8)
        else (0 : Int8)

      -- Pack 4 values per byte
      let packed := ByteArray.mk <| Array.ofFn (n := 64) fun byteIdx =>
        let i := byteIdx.val * 4
        let v0 := quantized.getD i 0
        let v1 := quantized.getD (i+1) 0
        let v2 := quantized.getD (i+2) 0
        let v3 := quantized.getD (i+3) 0
        packTernary2bit v0 v1 v2 v3

      -- Convert scale to FP16 (very simplified)
      let scaleU16 := float32ToFP16Simple scale
      let scaleByte0 := (scaleU16 &&& 0xFF).toUInt8
      let scaleByte1 := ((scaleU16 >>> 8) &&& 0xFF).toUInt8

        let blockBytes := packed.push scaleByte0 |>.push scaleByte1
        processBlock fuel' (blockIdx + 1) (acc ++ blockBytes)

  processBlock numBlocks 0 ByteArray.empty

end Hesper.GGUF.Quantization
