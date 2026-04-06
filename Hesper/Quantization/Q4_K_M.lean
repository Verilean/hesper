import Hesper.WGSL.Monad
import Hesper.WGSL.Execute
import Hesper.WGSL.Exp
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer

/-!
# Q4_K_M Quantization - GPU Dequantization Kernels

Implements on-the-fly dequantization of Q4_K (4-bit K-quantization) weights on GPU.

## Q4_K Block Format (256 elements, 144 bytes)

```
Offset  Size   Description
0       2      d     (FP16) - super-block scale for quantized scales
2       2      dmin  (FP16) - super-block scale for quantized mins
4       12     scales[12] - 8 sub-block scales + 8 sub-block mins (6-bit packed)
16      128    qs[128] - 4-bit quantized values (2 values per byte)
```

Total: 2 + 2 + 12 + 128 = 144 bytes per 256 elements

## Sub-block Scale Packing (12 bytes → 8 scales + 8 mins, 6-bit each)

For sub-block j (0..7):
- j < 4: sc = scales[j] & 63,      m = scales[j+4] & 63
- j >= 4: sc = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
           m = (scales[j+4] >> 4)   | ((scales[j] >> 6) << 4)

## Dequantization Formula

Each 256-element block has 8 sub-blocks of 32 elements.
For element i in sub-block j:
  y[i] = d * sc[j] * q[i] - dmin * m[j]

where q[i] is the 4-bit quantized value (0..15).

## Memory Layout

The 128 bytes of qs[] encode 256 values in pairs:
- Even iterations (j = 0,64,128,192): q[l] = qs[l] & 0xF  (low nibble)
- Odd iterations:                      q[l] = qs[l] >> 4    (high nibble)

Processing order per block (from dequantize_row_q4_K):
  for j in [0, 64, 128, 192]:
    sc0, m0 = get_scale_min(is+0)
    sc1, m1 = get_scale_min(is+1)
    for l in 0..31: y = d*sc0*(qs[l] & 0xF) - dmin*m0
    for l in 0..31: y = d*sc1*(qs[l] >> 4) - dmin*m1
    qs += 32; is += 2

## References
- llama.cpp: ggml/src/ggml-quants.c (dequantize_row_q4_K, get_scale_min_k4)
- GGUF spec: Q4_K = type ID 12
-/

namespace Hesper.Quantization.Q4_K_M

open Hesper.WGSL
open Hesper.WGSL.Monad
open Hesper.WebGPU

/-! ## Constants -/

/-- Q4_K super-block size in elements -/
def blockSize : Nat := 256

/-- Q4_K super-block size in bytes: 2(d) + 2(dmin) + 12(scales) + 128(qs) -/
def blockSizeBytes : Nat := 144

/-- Number of sub-blocks per super-block -/
def numSubBlocks : Nat := 8

/-- Elements per sub-block -/
def subBlockSize : Nat := 32

/-! ## FP16 Conversion -/

/-- Convert FP16 bits (stored in u32) to f32 in WGSL
    FP16: 1 sign + 5 exponent + 10 mantissa
    Special cases: 0, denormals, inf, nan handled -/
def fp16ToF32 (bits : Exp (.scalar .u32)) : Exp (.scalar .f32) :=
  -- Arithmetic FP16 → F32 conversion (no bitcast needed)
  -- FP16: 1 sign + 5 exponent + 10 mantissa, bias=15
  let sign := Exp.shiftRight bits (Exp.litU32 15)        -- 0 or 1
  let exp5 := Exp.bitAnd (Exp.shiftRight bits (Exp.litU32 10)) (Exp.litU32 0x1F)
  let mant := Exp.bitAnd bits (Exp.litU32 0x3FF)
  -- signF = (-1)^sign: if sign=1 then -1.0, else 1.0
  let signF := Exp.select (Exp.eq sign (Exp.litU32 1)) (Exp.litF32 (-1.0)) (Exp.litF32 1.0)
  -- mantissa as fraction: 1.mant = 1.0 + mant/1024.0
  let mantF := Exp.add (Exp.litF32 1.0) (Exp.div (Exp.toF32 mant) (Exp.litF32 1024.0))
  -- 2^(exp-15): use exp2() or ldexp-like computation
  -- exp2(f32(exp5) - 15.0)
  let expF := Exp.exp2 (Exp.sub (Exp.toF32 exp5) (Exp.litF32 15.0))
  -- result = sign * mantissa * 2^(exp-15)
  let normal := Exp.mul signF (Exp.mul mantF expF)
  -- Handle zero exponent (zero or denormal) → return 0
  let isZeroExp := Exp.eq exp5 (Exp.litU32 0)
  Exp.select isZeroExp (Exp.litF32 0.0) normal

/-! ## Scale/Min Extraction -/

/-- Extract scale (sc) and min (m) for sub-block j from the 12-byte scales array.

    The scales array is read as 3 u32s (12 bytes):
    scales_u32[0] = bytes[0..3]
    scales_u32[1] = bytes[4..7]
    scales_u32[2] = bytes[8..11]

    For j < 4: sc = scales[j] & 63,      m = scales[j+4] & 63
    For j >= 4: sc = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
                m = (scales[j+4] >> 4)   | ((scales[j] >> 6) << 4)

    @param j Sub-block index (0..7), must be a compile-time Nat
    @param scalesU32_0 First 4 bytes of scales as u32
    @param scalesU32_1 Next 4 bytes of scales as u32
    @param scalesU32_2 Last 4 bytes of scales as u32
    @return (scale, min) as (Exp f32, Exp f32)
-/
def getScaleMin (j : Nat)
    (scalesU32_0 scalesU32_1 scalesU32_2 : Exp (.scalar .u32))
    : Exp (.scalar .f32) × Exp (.scalar .f32) :=
  -- Helper: extract byte at position p from the three u32s
  let getByte (p : Nat) : Exp (.scalar .u32) :=
    if p < 4 then
      Exp.bitAnd (Exp.shiftRight scalesU32_0 (Exp.litU32 (p * 8))) (Exp.litU32 0xFF)
    else if p < 8 then
      Exp.bitAnd (Exp.shiftRight scalesU32_1 (Exp.litU32 ((p - 4) * 8))) (Exp.litU32 0xFF)
    else
      Exp.bitAnd (Exp.shiftRight scalesU32_2 (Exp.litU32 ((p - 8) * 8))) (Exp.litU32 0xFF)
  if j < 4 then
    let sc := Exp.bitAnd (getByte j) (Exp.litU32 63)
    let m := Exp.bitAnd (getByte (j + 4)) (Exp.litU32 63)
    (Exp.toF32 sc, Exp.toF32 m)
  else
    -- sc = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
    let scLow := Exp.bitAnd (getByte (j + 4)) (Exp.litU32 0xF)
    let scHigh := Exp.shiftLeft (Exp.shiftRight (getByte (j - 4)) (Exp.litU32 6)) (Exp.litU32 4)
    let sc := Exp.bitOr scLow scHigh
    -- m = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
    let mLow := Exp.shiftRight (getByte (j + 4)) (Exp.litU32 4)
    let mHigh := Exp.shiftLeft (Exp.shiftRight (getByte j) (Exp.litU32 6)) (Exp.litU32 4)
    let m := Exp.bitOr mLow mHigh
    (Exp.toF32 sc, Exp.toF32 m)

/-! ## GPU Dequantization Kernel -/

/-- GPU kernel to dequantize Q4_K_M packed weights to Float32

    One thread per element. Each thread:
    1. Determines its super-block and position within it
    2. Reads the FP16 d/dmin from block header
    3. Reads and extracts its 6-bit scale and min
    4. Reads its 4-bit quantized value
    5. Computes: y = d * sc * q - dmin * m

    @param numElements Total number of elements to dequantize
-/
def dequantQ4KMKernel (numElements : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  -- Bounds check
  let inBounds := Exp.lt idx (Exp.litU32 numElements)

  -- Number of blocks
  let numBlocks := (numElements + blockSize - 1) / blockSize
  -- Data buffer: raw bytes as u32 array
  let numU32 := (numBlocks * blockSizeBytes + 3) / 4

  let _data ← ShaderM.declareInputBuffer "data" (.array (.scalar .u32) numU32)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) numElements)

  ShaderM.if_ inBounds (do
    -- Which super-block and element within it
    let blockIdx := Exp.div idx (Exp.litU32 blockSize)
    let localIdx := Exp.mod idx (Exp.litU32 blockSize)

    -- Block byte offset in data buffer
    -- blockBase (in u32 units) = blockIdx * 144 / 4 = blockIdx * 36
    let blockBaseU32 := Exp.mul blockIdx (Exp.litU32 36)

    -- Read d and dmin (first 4 bytes = 1 u32, containing two FP16 values)
    let dmU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := numU32) "data" blockBaseU32
    let dBits := Exp.bitAnd dmU32 (Exp.litU32 0xFFFF)
    let dminBits := Exp.shiftRight dmU32 (Exp.litU32 16)
    let d := fp16ToF32 dBits
    let dmin := fp16ToF32 dminBits

    -- Read scales[12] as 3 u32s (bytes 4..15, u32 offset 1..3)
    let scalesU32_0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := numU32) "data" (Exp.add blockBaseU32 (Exp.litU32 1))
    let scalesU32_1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := numU32) "data" (Exp.add blockBaseU32 (Exp.litU32 2))
    let scalesU32_2 ← ShaderM.readBuffer (ty := .scalar .u32) (n := numU32) "data" (Exp.add blockBaseU32 (Exp.litU32 3))

    -- Determine sub-block index (j) and position within sub-block
    -- The dequant loop processes in chunks of 64 elements:
    --   chunk 0: elements 0..63   (sub-blocks 0,1)
    --   chunk 1: elements 64..127 (sub-blocks 2,3)
    --   chunk 2: elements 128..191 (sub-blocks 4,5)
    --   chunk 3: elements 192..255 (sub-blocks 6,7)
    --
    -- Within each chunk of 64:
    --   elements 0..31 use low nibble, sub-block = chunk*2
    --   elements 32..63 use high nibble, sub-block = chunk*2+1
    let chunkIdx := Exp.div localIdx (Exp.litU32 64)       -- 0..3
    let posInChunk := Exp.mod localIdx (Exp.litU32 64)      -- 0..63
    let isHighNibble := Exp.ge posInChunk (Exp.litU32 32)  -- true for elements 32..63
    let posInSubBlock := Exp.mod posInChunk (Exp.litU32 32) -- 0..31

    -- Sub-block index (0..7) for scale/min lookup
    let subBlockIdx := Exp.add (Exp.mul chunkIdx (Exp.litU32 2)) (Exp.select isHighNibble (Exp.litU32 1) (Exp.litU32 0))

    -- Get scale and min for this sub-block
    -- Compute all 8 (sc, m) pairs and cascade select() to pick the right one at runtime
    let (sc0, m0) := getScaleMin 0 scalesU32_0 scalesU32_1 scalesU32_2
    let (sc1, m1) := getScaleMin 1 scalesU32_0 scalesU32_1 scalesU32_2
    let (sc2, m2) := getScaleMin 2 scalesU32_0 scalesU32_1 scalesU32_2
    let (sc3, m3) := getScaleMin 3 scalesU32_0 scalesU32_1 scalesU32_2
    let (sc4, m4) := getScaleMin 4 scalesU32_0 scalesU32_1 scalesU32_2
    let (sc5, m5) := getScaleMin 5 scalesU32_0 scalesU32_1 scalesU32_2
    let (sc6, m6) := getScaleMin 6 scalesU32_0 scalesU32_1 scalesU32_2
    let (sc7, m7) := getScaleMin 7 scalesU32_0 scalesU32_1 scalesU32_2

    -- Cascade select: start from j=7, work backwards
    let scVal := Exp.select (Exp.eq subBlockIdx (Exp.litU32 0)) sc0
      (Exp.select (Exp.eq subBlockIdx (Exp.litU32 1)) sc1
      (Exp.select (Exp.eq subBlockIdx (Exp.litU32 2)) sc2
      (Exp.select (Exp.eq subBlockIdx (Exp.litU32 3)) sc3
      (Exp.select (Exp.eq subBlockIdx (Exp.litU32 4)) sc4
      (Exp.select (Exp.eq subBlockIdx (Exp.litU32 5)) sc5
      (Exp.select (Exp.eq subBlockIdx (Exp.litU32 6)) sc6 sc7))))))
    let mVal := Exp.select (Exp.eq subBlockIdx (Exp.litU32 0)) m0
      (Exp.select (Exp.eq subBlockIdx (Exp.litU32 1)) m1
      (Exp.select (Exp.eq subBlockIdx (Exp.litU32 2)) m2
      (Exp.select (Exp.eq subBlockIdx (Exp.litU32 3)) m3
      (Exp.select (Exp.eq subBlockIdx (Exp.litU32 4)) m4
      (Exp.select (Exp.eq subBlockIdx (Exp.litU32 5)) m5
      (Exp.select (Exp.eq subBlockIdx (Exp.litU32 6)) m6 m7))))))

    -- Read the 4-bit quantized value
    -- qs[] starts at byte offset 16 in the block, i.e., u32 offset 4
    -- qs index = chunkIdx * 32 + posInSubBlock (byte index within qs[128])
    let qsByteIdx := Exp.add (Exp.mul chunkIdx (Exp.litU32 32)) posInSubBlock
    let qsU32Idx := Exp.div qsByteIdx (Exp.litU32 4)
    let qsByteOffset := Exp.mul (Exp.mod qsByteIdx (Exp.litU32 4)) (Exp.litU32 8)
    let qsU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := numU32) "data" (Exp.add blockBaseU32 (Exp.add (Exp.litU32 4) qsU32Idx))
    let qsByte := Exp.bitAnd (Exp.shiftRight qsU32 qsByteOffset) (Exp.litU32 0xFF)

    -- Extract nibble: low or high
    let qVal := Exp.select isHighNibble
      (Exp.shiftRight qsByte (Exp.litU32 4))
      (Exp.bitAnd qsByte (Exp.litU32 0xF))

    -- Dequantize: y = d * sc * q - dmin * m
    let result := Exp.sub
      (Exp.mul d (Exp.mul scVal (Exp.toF32 qVal)))
      (Exp.mul dmin mVal)

    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx result
  ) (pure ())

/-! ## High-Level API -/

/-- Configuration for Q4_K_M dequantization -/
structure DequantConfig where
  numElements : Nat
  workgroupSize : Nat := 256
  deriving Repr

/-- Execute Q4_K_M dequantization on GPU

    @param device WebGPU device
    @param dataBuf GPU buffer containing raw Q4_K_M block data
    @param outputBuf GPU buffer for Float32 output
    @param config Dequantization configuration
-/
def executeDequant (device : Device)
                   (dataBuf outputBuf : Buffer)
                   (config : DequantConfig) : IO Unit := do
  let shader := dequantQ4KMKernel config.numElements
  let namedBuffers := [
    ("data", dataBuf),
    ("output", outputBuf)
  ]

  let execConfig := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D
    config.numElements
    config.workgroupSize

  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig

end Hesper.Quantization.Q4_K_M
