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
  -- Arithmetic FP16 → F32 conversion (supports subnormals)
  -- Normal:    (-1)^s * (1 + mant/1024) * 2^(exp-15)
  -- Subnormal: (-1)^s * (mant/1024) * 2^(-14)
  let sign := Exp.shiftRight bits (Exp.litU32 15)
  let exp5 := Exp.bitAnd (Exp.shiftRight bits (Exp.litU32 10)) (Exp.litU32 0x1F)
  let mant := Exp.bitAnd bits (Exp.litU32 0x3FF)
  let signF := Exp.select (Exp.eq sign (Exp.litU32 1)) (Exp.litF32 (-1.0)) (Exp.litF32 1.0)
  let isSubnormal := Exp.eq exp5 (Exp.litU32 0)
  let mantFNormal := Exp.add (Exp.litF32 1.0) (Exp.div (Exp.toF32 mant) (Exp.litF32 1024.0))
  let mantFSubnormal := Exp.div (Exp.toF32 mant) (Exp.litF32 1024.0)
  let mantF := Exp.select isSubnormal mantFSubnormal mantFNormal
  let expFNormal := Exp.exp2 (Exp.sub (Exp.toF32 exp5) (Exp.litF32 15.0))
  let expFSubnormal := Exp.litF32 6.103515625e-5  -- 2^(-14)
  let expF := Exp.select isSubnormal expFSubnormal expFNormal
  Exp.mul signF (Exp.mul mantF expF)

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
def dequantQ4KMKernel (numElements : Nat) (gridXWidth : Nat := 0) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  -- 1D when gridXWidth=0; else 2D (idx = x + y*gridXWidth) so numElements/256 > the 65535
  -- per-dimension workgroup limit can be covered by a single 2D dispatch.
  let idx := if gridXWidth > 0
    then Exp.add (Exp.vec3X gid) (Exp.mul (Exp.vec3Y gid) (Exp.litU32 gridXWidth))
    else Exp.vec3X gid

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

/-- FUSED Q4_K-dequant grouped reg-matmul (MoE gate/up): reads the expert weights as Q4_K DIRECTLY
    and dequants to f16 inside the B-load (no pre-dequant, no 30GB f16 cache, no extra weight traffic
    — the matrix-unit analogue of MMQ5's in-kernel int8 dequant). A = f32 [M,K] gathered activations;
    B = Q4_K bytes for [nExpert·N, K] as a u32 array (row-major, each row = its own Q4_K vector of K);
    C = f32 [M,N]; tileExpert[M/64] picks each 64-row M-tile's expert. Requires K%256=0; with BK=32
    each K-tile is exactly one 32-elem Q4_K sub-block ⇒ one getScaleMin per tile. Grid (N/32, M/64)×128.
    The A-load / MMA / store are identical to matMulTransposeF16WMMARegKernel; only the B-load differs. -/
def q4kMatmulGroupedRegKernel (M N K nExpert : Nat) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let tid := Exp.vec3X lid
  let rowStrideU32 := (K / 256) * 36       -- u32 per weight row: (K/256) blocks × 144 bytes / 4
  let bRows := nExpert * N
  let bU32 := bRows * rowStrideU32
  let _a ← ShaderM.declareInputBuffer "a" (.array (.scalar .f32) (M * K))
  let _b ← ShaderM.declareInputBuffer "b" (.array (.scalar .u32) bU32)
  let _c ← ShaderM.declareOutputBuffer "c" (.array (.scalar .f32) (M * N))
  let _te ← ShaderM.declareInputBuffer "tileExpert" (.array (.scalar .u32) (M / 32))
  ShaderM.sharedNamed "shared_A" (.array (.scalar .f16) (32 * 32))
  ShaderM.sharedNamed "shared_B" (.array (.scalar .f16) (32 * 32))
  ShaderM.sharedNamed "shared_dq" (.array (.scalar .f32) (32 * 18))   -- per-row: d,dmin,sc0..7,m0..7 for this 256-block
  ShaderM.declareMatrixLeftArray  "Ax" .f16 8 8 2 Exp.subgroupMatrixZeroLeft
  ShaderM.declareMatrixRightArray "Bx" .f16 8 8 2 Exp.subgroupMatrixZeroRight
  ShaderM.declareMatrixResultArray "Cx" .f32 8 8 4 Exp.subgroupMatrixZeroResult
  let rowBase := Exp.mul (Exp.vec3Y wid) (Exp.litU32 32)
  let colBase := Exp.mul (Exp.vec3X wid) (Exp.litU32 32)
  let sgitg := Exp.div tid (Exp.litU32 32)
  let sgRow := Exp.mod sgitg (Exp.litU32 2)
  let sgCol := Exp.div sgitg (Exp.litU32 2)
  let mOff := Exp.mul sgRow (Exp.litU32 16)
  let nOff := Exp.mul sgCol (Exp.litU32 16)
  let teRaw ← ShaderM.readBuffer (ty := .scalar .u32) (n := M / 32) "tileExpert" (Exp.vec3Y wid)
  let e := Exp.select (Exp.lt teRaw (Exp.litU32 nExpert)) teRaw (Exp.litU32 (nExpert - 1))
  let weightRowOffsetE := Exp.mul e (Exp.litU32 N)
  let numBlk := K / 256
  -- BLOCK-AT-A-TIME: outer loop over 256-elem Q4_K blocks. The 8 getScaleMin per row are computed ONCE
  -- per block (cooperatively) into shared_dq; the inner jSub loop then just INDEXES shared_dq[2+jSub]
  -- (no per-K-tile cascade, no per-K-tile header reads — 8× fewer getScaleMin than the per-K-tile path).
  ShaderM.loop (Exp.litU32 0) (Exp.litU32 numBlk) (Exp.litU32 1) fun blockIdx => do
    ShaderM.if_ (Exp.lt tid (Exp.litU32 32)) (do
      let row := Exp.add (Exp.add weightRowOffsetE colBase) tid
      let bb := Exp.add (Exp.mul row (Exp.litU32 rowStrideU32)) (Exp.mul blockIdx (Exp.litU32 36))
      let dmU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := bU32) "b" bb
      let d := fp16ToF32 (Exp.bitAnd dmU32 (Exp.litU32 0xFFFF))
      let dmin := fp16ToF32 (Exp.shiftRight dmU32 (Exp.litU32 16))
      let sca0 ← ShaderM.readBuffer (ty := .scalar .u32) (n := bU32) "b" (Exp.add bb (Exp.litU32 1))
      let sca1 ← ShaderM.readBuffer (ty := .scalar .u32) (n := bU32) "b" (Exp.add bb (Exp.litU32 2))
      let sca2 ← ShaderM.readBuffer (ty := .scalar .u32) (n := bU32) "b" (Exp.add bb (Exp.litU32 3))
      let base := Exp.mul tid (Exp.litU32 18)
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_dq" base d
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_dq" (Exp.add base (Exp.litU32 1)) dmin
      for j in [0:8] do
        let (scj, mmj) := getScaleMin j sca0 sca1 sca2
        ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_dq" (Exp.add base (Exp.litU32 (2+j))) scj
        ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_dq" (Exp.add base (Exp.litU32 (10+j))) mmj) (pure ())
    ShaderM.barrier
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 8) (Exp.litU32 1) fun jSub => do
      let kBaseE := Exp.add (Exp.mul blockIdx (Exp.litU32 256)) (Exp.mul jSub (Exp.litU32 32))
      let chunk := Exp.div jSub (Exp.litU32 2)
      let isHigh := Exp.eq (Exp.mod jSub (Exp.litU32 2)) (Exp.litU32 1)
      -- A-load [32,32] f16
      for s in [0:8] do
        let idx := Exp.add tid (Exp.litU32 (s * 128))
        let m := Exp.div idx (Exp.litU32 32)
        let k := Exp.mod idx (Exp.litU32 32)
        let aIdx := Exp.add (Exp.mul (Exp.add rowBase m) (Exp.litU32 K)) (Exp.add kBaseE k)
        let xf32 ← ShaderM.readBuffer (ty := .scalar .f32) (n := M * K) "a" aIdx
        let blk := Exp.add (Exp.mul (Exp.div m (Exp.litU32 8)) (Exp.litU32 4)) (Exp.div k (Exp.litU32 8))
        let within := Exp.add (Exp.mul (Exp.mod m (Exp.litU32 8)) (Exp.litU32 8)) (Exp.mod k (Exp.litU32 8))
        ShaderM.writeWorkgroup (ty := .scalar .f16) "shared_A" (Exp.add (Exp.mul blk (Exp.litU32 64)) within) (Exp.toF16 xf32)
      -- B-load: read this row's d/dmin + sc[jSub]/m[jSub] from shared_dq (no cascade); fetch the quants.
      for s in [0:4] do
        let u := Exp.add tid (Exp.litU32 (s * 128))
        let n := Exp.div u (Exp.litU32 16)
        let kpair := Exp.mod u (Exp.litU32 16)
        let row := Exp.add (Exp.add weightRowOffsetE colBase) n
        let blockBaseU32 := Exp.add (Exp.mul row (Exp.litU32 rowStrideU32)) (Exp.mul blockIdx (Exp.litU32 36))
        let dqBase := Exp.mul n (Exp.litU32 18)
        let d ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 32*18) "shared_dq" dqBase
        let dmin ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 32*18) "shared_dq" (Exp.add dqBase (Exp.litU32 1))
        let scV ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 32*18) "shared_dq" (Exp.add dqBase (Exp.add (Exp.litU32 2) jSub))
        let mV ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 32*18) "shared_dq" (Exp.add dqBase (Exp.add (Exp.litU32 10) jSub))
        let k0 := Exp.mul kpair (Exp.litU32 2)
        let readQ := fun (p : Exp (.scalar .u32)) => do
          let qsByteIdx := Exp.add (Exp.mul chunk (Exp.litU32 32)) p
          let qsU32Idx := Exp.div qsByteIdx (Exp.litU32 4)
          let qsByteOff := Exp.mul (Exp.mod qsByteIdx (Exp.litU32 4)) (Exp.litU32 8)
          let qsU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := bU32) "b" (Exp.add blockBaseU32 (Exp.add (Exp.litU32 4) qsU32Idx))
          let qsByte := Exp.bitAnd (Exp.shiftRight qsU32 qsByteOff) (Exp.litU32 0xFF)
          pure (Exp.toF32 (Exp.select isHigh (Exp.shiftRight qsByte (Exp.litU32 4)) (Exp.bitAnd qsByte (Exp.litU32 0xF))))
        let q0 ← readQ k0
        let q1 ← readQ (Exp.add k0 (Exp.litU32 1))
        let y0 := Exp.sub (Exp.mul d (Exp.mul scV q0)) (Exp.mul dmin mV)
        let y1 := Exp.sub (Exp.mul d (Exp.mul scV q1)) (Exp.mul dmin mV)
        let ktile := Exp.div k0 (Exp.litU32 8)
        let kr := Exp.mod k0 (Exp.litU32 8)
        let blkB := Exp.add (Exp.mul ktile (Exp.litU32 4)) (Exp.div n (Exp.litU32 8))
        let baseB := Exp.add (Exp.mul blkB (Exp.litU32 64)) (Exp.mod n (Exp.litU32 8))
        ShaderM.writeWorkgroup (ty := .scalar .f16) "shared_B" (Exp.add baseB (Exp.mul kr (Exp.litU32 8))) (Exp.toF16 y0)
        ShaderM.writeWorkgroup (ty := .scalar .f16) "shared_B" (Exp.add baseB (Exp.mul (Exp.add kr (Exp.litU32 1)) (Exp.litU32 8))) (Exp.toF16 y1)
      ShaderM.barrier
      for k8 in [0:4] do
        for mt in [0:2] do
          let mtileG := Exp.add (Exp.mul sgRow (Exp.litU32 2)) (Exp.litU32 mt)
          let blkA := Exp.add (Exp.mul mtileG (Exp.litU32 4)) (Exp.litU32 k8)
          ShaderM.loadMatrixLeft (st := .f16) (m := 8) (k := 8) "Ax" mt "shared_A" (Exp.mul blkA (Exp.litU32 64)) (Exp.litU32 8)
        for nt in [0:2] do
          let ntileG := Exp.add (Exp.mul sgCol (Exp.litU32 2)) (Exp.litU32 nt)
          let blkB := Exp.add (Exp.mul (Exp.litU32 (k8 * 4)) (Exp.litU32 1)) ntileG
          ShaderM.loadMatrixRight (st := .f16) (k := 8) (n := 8) "Bx" nt "shared_B" (Exp.mul blkB (Exp.litU32 64)) (Exp.litU32 8)
        for mt in [0:2] do
          for nt in [0:2] do
            ShaderM.matrixMultiplyAccumulateMixed (inSt := .f16) (outSt := .f32) (m := 8) (k := 8) (n := 8) "Cx" (mt * 2 + nt) "Ax" mt "Bx" nt
      ShaderM.barrier
  for mt in [0:2] do
    for nt in [0:2] do
      let row := Exp.add (Exp.add rowBase mOff) (Exp.litU32 (mt * 8))
      let col := Exp.add (Exp.add colBase nOff) (Exp.litU32 (nt * 8))
      let off := Exp.add (Exp.mul row (Exp.litU32 N)) col
      ShaderM.storeMatrixResult (st := .f32) (m := 8) (n := 8) "Cx" (mt * 2 + nt) "c" off (Exp.litU32 N)

/-- FUSED Q8_0-dequant grouped reg-matmul (MoE DOWN): the matrix-unit analogue of the warp-per-output
    down kernel (which is ~7× less efficient/FLOP than the tiled gate/up — the real MoE bottleneck per
    the DG_SKIP measurement). Reads the expert down weights as Q8_0 bytes DIRECTLY and dequants to f16
    in the B-load. A = f32 [M,K] grouped geglu output; B = Q8_0 bytes for [nExpert·N, K] (34 bytes/block
    = f16 scale + 32 int8); C = f32 [M,N]; tileExpert[M/32] picks the expert. BK=32 = exactly one Q8_0
    block ⇒ one scale per K-tile (no sub-blocks — simpler than Q4_K). A-load/MMA/store identical to the
    Q4_K reg (BM=32); only the B-load differs. Grid (N/32, M/32)×128. -/
def q8MatmulGroupedRegKernel (M N K nExpert : Nat) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let tid := Exp.vec3X lid
  let rowByteStride := (K / 32) * 34       -- bytes per weight row: (K/32) Q8_0 blocks × 34 bytes
  let bRows := nExpert * N
  let bU32 := (bRows * rowByteStride + 3) / 4
  let _a ← ShaderM.declareInputBuffer "a" (.array (.scalar .f32) (M * K))
  let _b ← ShaderM.declareInputBuffer "b" (.array (.scalar .u32) bU32)
  let _c ← ShaderM.declareOutputBuffer "c" (.array (.scalar .f32) (M * N))
  let _te ← ShaderM.declareInputBuffer "tileExpert" (.array (.scalar .u32) (M / 32))
  ShaderM.sharedNamed "shared_A" (.array (.scalar .f16) (32 * 32))
  ShaderM.sharedNamed "shared_B" (.array (.scalar .f16) (32 * 32))
  ShaderM.sharedNamed "shared_d" (.array (.scalar .f32) 32)   -- per-row Q8_0 scale for this K-tile's block
  ShaderM.declareMatrixLeftArray  "Ax" .f16 8 8 2 Exp.subgroupMatrixZeroLeft
  ShaderM.declareMatrixRightArray "Bx" .f16 8 8 2 Exp.subgroupMatrixZeroRight
  ShaderM.declareMatrixResultArray "Cx" .f32 8 8 4 Exp.subgroupMatrixZeroResult
  let rowBase := Exp.mul (Exp.vec3Y wid) (Exp.litU32 32)
  let colBase := Exp.mul (Exp.vec3X wid) (Exp.litU32 32)
  let sgitg := Exp.div tid (Exp.litU32 32)
  let sgRow := Exp.mod sgitg (Exp.litU32 2)
  let sgCol := Exp.div sgitg (Exp.litU32 2)
  let mOff := Exp.mul sgRow (Exp.litU32 16)
  let nOff := Exp.mul sgCol (Exp.litU32 16)
  let teRaw ← ShaderM.readBuffer (ty := .scalar .u32) (n := M / 32) "tileExpert" (Exp.vec3Y wid)
  let isActive := Exp.lt teRaw (Exp.litU32 nExpert)
  let e := Exp.select isActive teRaw (Exp.litU32 (nExpert - 1))
  let weightRowOffsetE := Exp.mul e (Exp.litU32 N)
  let rdByte := fun (bo : Exp (.scalar .u32)) => do
    let w ← ShaderM.readBuffer (ty := .scalar .u32) (n := bU32) "b" (Exp.shiftRight bo (Exp.litU32 2))
    pure (Exp.bitAnd (Exp.shiftRight w (Exp.mul (Exp.bitAnd bo (Exp.litU32 3)) (Exp.litU32 8))) (Exp.litU32 0xFF))
  let numKB := K / 32
  -- NOTE: padding tiles (tileExpert ≥ nExpert) clamp e=nExpert-1 and compute garbage, but the grouped
  -- scatter skips padding rows (only real tokens are scattered) so it's ignored. A kEnd-based skip
  -- breaks WGSL barrier-uniformity (runtime loop bound) — not worth it; the scatter already drops them.
  ShaderM.loop (Exp.litU32 0) (Exp.litU32 numKB) (Exp.litU32 1) fun batchV => do
    let blkByte := Exp.mul batchV (Exp.litU32 34)
    -- COOPERATIVE: 32 threads each load ONE N-tile row's Q8_0 scale for this block → shared_d
    ShaderM.if_ (Exp.lt tid (Exp.litU32 32)) (do
      let row := Exp.add (Exp.add weightRowOffsetE colBase) tid
      let bb := Exp.add (Exp.mul row (Exp.litU32 rowByteStride)) blkByte
      let lo ← rdByte bb
      let hi ← rdByte (Exp.add bb (Exp.litU32 1))
      let d := fp16ToF32 (Exp.add lo (Exp.mul hi (Exp.litU32 256)))
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_d" tid d) (pure ())
    ShaderM.barrier
    -- A-load [32,32] f16
    for s in [0:8] do
      let idx := Exp.add tid (Exp.litU32 (s * 128))
      let m := Exp.div idx (Exp.litU32 32)
      let k := Exp.mod idx (Exp.litU32 32)
      let aIdx := Exp.add (Exp.mul (Exp.add rowBase m) (Exp.litU32 K)) (Exp.add (Exp.mul batchV (Exp.litU32 32)) k)
      let xf32 ← ShaderM.readBuffer (ty := .scalar .f32) (n := M * K) "a" aIdx
      let blk := Exp.add (Exp.mul (Exp.div m (Exp.litU32 8)) (Exp.litU32 4)) (Exp.div k (Exp.litU32 8))
      let within := Exp.add (Exp.mul (Exp.mod m (Exp.litU32 8)) (Exp.litU32 8)) (Exp.mod k (Exp.litU32 8))
      -- sentinel-skip (correctness): padding tiles (tileExpert ≥ nExpert) zero their A-input → C=0×B=0 →
      -- output 0, so the grouped scatter (which DOES scatter padding rows at pos=0/slot=0) writes 0 not
      -- garbage. A data-select, NOT a runtime loop bound, so WGSL barrier-uniformity is preserved.
      ShaderM.writeWorkgroup (ty := .scalar .f16) "shared_A" (Exp.add (Exp.mul blk (Exp.litU32 64)) within) (Exp.toF16 (Exp.select isActive xf32 (Exp.litF32 0.0)))
    -- B-load: Q8_0 dequant (scale from shared_d, two int8 quants per thread)
    for s in [0:4] do
      let u := Exp.add tid (Exp.litU32 (s * 128))
      let n := Exp.div u (Exp.litU32 16)
      let kpair := Exp.mod u (Exp.litU32 16)
      let row := Exp.add (Exp.add weightRowOffsetE colBase) n
      let bb := Exp.add (Exp.mul row (Exp.litU32 rowByteStride)) blkByte
      let d ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 32) "shared_d" n
      let k0 := Exp.mul kpair (Exp.litU32 2)
      let deq := fun (qb : Exp (.scalar .u32)) =>
        Exp.mul d (Exp.sub (Exp.toF32 qb) (Exp.mul (Exp.litF32 256.0) (Exp.toF32 (Exp.shiftRight qb (Exp.litU32 7)))))
      let q0b ← rdByte (Exp.add bb (Exp.add (Exp.litU32 2) k0))
      let q1b ← rdByte (Exp.add bb (Exp.add (Exp.litU32 3) k0))
      let y0 := deq q0b
      let y1 := deq q1b
      let ktile := Exp.div k0 (Exp.litU32 8)
      let kr := Exp.mod k0 (Exp.litU32 8)
      let blkB := Exp.add (Exp.mul ktile (Exp.litU32 4)) (Exp.div n (Exp.litU32 8))
      let baseB := Exp.add (Exp.mul blkB (Exp.litU32 64)) (Exp.mod n (Exp.litU32 8))
      ShaderM.writeWorkgroup (ty := .scalar .f16) "shared_B" (Exp.add baseB (Exp.mul kr (Exp.litU32 8))) (Exp.toF16 y0)
      ShaderM.writeWorkgroup (ty := .scalar .f16) "shared_B" (Exp.add baseB (Exp.mul (Exp.add kr (Exp.litU32 1)) (Exp.litU32 8))) (Exp.toF16 y1)
    ShaderM.barrier
    for k8 in [0:4] do
      for mt in [0:2] do
        let mtileG := Exp.add (Exp.mul sgRow (Exp.litU32 2)) (Exp.litU32 mt)
        let blkA := Exp.add (Exp.mul mtileG (Exp.litU32 4)) (Exp.litU32 k8)
        ShaderM.loadMatrixLeft (st := .f16) (m := 8) (k := 8) "Ax" mt "shared_A" (Exp.mul blkA (Exp.litU32 64)) (Exp.litU32 8)
      for nt in [0:2] do
        let ntileG := Exp.add (Exp.mul sgCol (Exp.litU32 2)) (Exp.litU32 nt)
        let blkB := Exp.add (Exp.mul (Exp.litU32 (k8 * 4)) (Exp.litU32 1)) ntileG
        ShaderM.loadMatrixRight (st := .f16) (k := 8) (n := 8) "Bx" nt "shared_B" (Exp.mul blkB (Exp.litU32 64)) (Exp.litU32 8)
      for mt in [0:2] do
        for nt in [0:2] do
          ShaderM.matrixMultiplyAccumulateMixed (inSt := .f16) (outSt := .f32) (m := 8) (k := 8) (n := 8) "Cx" (mt * 2 + nt) "Ax" mt "Bx" nt
    ShaderM.barrier
  for mt in [0:2] do
    for nt in [0:2] do
      let row := Exp.add (Exp.add rowBase mOff) (Exp.litU32 (mt * 8))
      let col := Exp.add (Exp.add colBase nOff) (Exp.litU32 (nt * 8))
      let off := Exp.add (Exp.mul row (Exp.litU32 N)) col
      ShaderM.storeMatrixResult (st := .f32) (m := 8) (n := 8) "Cx" (mt * 2 + nt) "c" off (Exp.litU32 N)

/-- FUSED geglu + Q8_0 down matmul + scatter in ONE kernel — removes the inter-pass flushes
    (geglu→q80→down→scatter) that Dawn drops at batch scale (the grouped-down RACE). Reads sGatheredGU
    (grouped gate/up [maxPadded, 2·expFF]), computes the tanh-GELU geglu on-the-fly into the A-tile, runs
    the matrix-reg Q8_0 down matmul, stores the 32×32 result tile to shared, then scatters each row to
    sDownAll via pos/slot (padding rows = slot≥nUsed are skipped). Only fused→wacc stays a sync point. -/
def q8FusedGegluDownScatterKernel (maxPadded dim expFF nExpert nUsed nTok : Nat) : ShaderM Unit := do
  let M := maxPadded; let N := dim; let K := expFF
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let tid := Exp.vec3X lid
  let rowByteStride := (K / 32) * 34
  let bU32 := (nExpert * N * rowByteStride + 3) / 4
  let guSize := M * 2 * K
  let _gu ← ShaderM.declareInputBuffer "gu" (.array (.scalar .f32) guSize)
  let _b ← ShaderM.declareInputBuffer "b" (.array (.scalar .u32) bU32)
  let _te ← ShaderM.declareInputBuffer "tileExpert" (.array (.scalar .u32) (M / 32))
  let _pos ← ShaderM.declareInputBuffer "pos" (.array (.scalar .u32) M)
  let _slot ← ShaderM.declareInputBuffer "slot" (.array (.scalar .u32) M)
  let _dst ← ShaderM.declareOutputBuffer "dst" (.array (.scalar .f32) (nUsed * nTok * N))
  ShaderM.sharedNamed "shared_A" (.array (.scalar .f16) (32 * 32))
  ShaderM.sharedNamed "shared_B" (.array (.scalar .f16) (32 * 32))
  ShaderM.sharedNamed "shared_d" (.array (.scalar .f32) 32)
  ShaderM.sharedNamed "shared_C" (.array (.scalar .f32) (32 * 32))
  ShaderM.declareMatrixLeftArray  "Ax" .f16 8 8 2 Exp.subgroupMatrixZeroLeft
  ShaderM.declareMatrixRightArray "Bx" .f16 8 8 2 Exp.subgroupMatrixZeroRight
  ShaderM.declareMatrixResultArray "Cx" .f32 8 8 4 Exp.subgroupMatrixZeroResult
  let rowBase := Exp.mul (Exp.vec3Y wid) (Exp.litU32 32)
  let colBase := Exp.mul (Exp.vec3X wid) (Exp.litU32 32)
  let sgitg := Exp.div tid (Exp.litU32 32)
  let sgRow := Exp.mod sgitg (Exp.litU32 2)
  let sgCol := Exp.div sgitg (Exp.litU32 2)
  let mOff := Exp.mul sgRow (Exp.litU32 16)
  let nOff := Exp.mul sgCol (Exp.litU32 16)
  let teRaw ← ShaderM.readBuffer (ty := .scalar .u32) (n := M / 32) "tileExpert" (Exp.vec3Y wid)
  let e := Exp.select (Exp.lt teRaw (Exp.litU32 nExpert)) teRaw (Exp.litU32 (nExpert - 1))
  let weightRowOffsetE := Exp.mul e (Exp.litU32 N)
  let rdByte := fun (bo : Exp (.scalar .u32)) => do
    let w ← ShaderM.readBuffer (ty := .scalar .u32) (n := bU32) "b" (Exp.shiftRight bo (Exp.litU32 2))
    pure (Exp.bitAnd (Exp.shiftRight w (Exp.mul (Exp.bitAnd bo (Exp.litU32 3)) (Exp.litU32 8))) (Exp.litU32 0xFF))
  let numKB := K / 32
  ShaderM.loop (Exp.litU32 0) (Exp.litU32 numKB) (Exp.litU32 1) fun batchV => do
    let blkByte := Exp.mul batchV (Exp.litU32 34)
    ShaderM.if_ (Exp.lt tid (Exp.litU32 32)) (do
      let row := Exp.add (Exp.add weightRowOffsetE colBase) tid
      let bb := Exp.add (Exp.mul row (Exp.litU32 rowByteStride)) blkByte
      let lo ← rdByte bb
      let hi ← rdByte (Exp.add bb (Exp.litU32 1))
      let d := fp16ToF32 (Exp.add lo (Exp.mul hi (Exp.litU32 256)))
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_d" tid d) (pure ())
    ShaderM.barrier
    -- A-load: geglu(gu) on-the-fly → shared_A (tanh-GELU, matches gegluMergedB)
    for s in [0:8] do
      let idx := Exp.add tid (Exp.litU32 (s * 128))
      let m := Exp.div idx (Exp.litU32 32)
      let k := Exp.mod idx (Exp.litU32 32)
      let row := Exp.add rowBase m
      let kGlobal := Exp.add (Exp.mul batchV (Exp.litU32 32)) k
      let gbase := Exp.mul row (Exp.litU32 (2 * K))
      let gate ← ShaderM.readBuffer (ty := .scalar .f32) (n := guSize) "gu" (Exp.add gbase kGlobal)
      let up ← ShaderM.readBuffer (ty := .scalar .f32) (n := guSize) "gu" (Exp.add (Exp.add gbase kGlobal) (Exp.litU32 K))
      let g3 := Exp.mul gate (Exp.mul gate gate)
      let inner := Exp.mul (Exp.litF32 0.7978845608) (Exp.add gate (Exp.mul (Exp.litF32 0.044715) g3))
      let iC := Exp.max (Exp.litF32 (-10.0)) (Exp.min (Exp.litF32 10.0) inner)
      let gl := Exp.mul (Exp.mul (Exp.litF32 0.5) gate) (Exp.add (Exp.litF32 1.0) (Exp.tanh iC))
      let eh := Exp.mul gl up
      let blk := Exp.add (Exp.mul (Exp.div m (Exp.litU32 8)) (Exp.litU32 4)) (Exp.div k (Exp.litU32 8))
      let within := Exp.add (Exp.mul (Exp.mod m (Exp.litU32 8)) (Exp.litU32 8)) (Exp.mod k (Exp.litU32 8))
      ShaderM.writeWorkgroup (ty := .scalar .f16) "shared_A" (Exp.add (Exp.mul blk (Exp.litU32 64)) within) (Exp.toF16 eh)
    for s in [0:4] do
      let u := Exp.add tid (Exp.litU32 (s * 128))
      let n := Exp.div u (Exp.litU32 16)
      let kpair := Exp.mod u (Exp.litU32 16)
      let row := Exp.add (Exp.add weightRowOffsetE colBase) n
      let bb := Exp.add (Exp.mul row (Exp.litU32 rowByteStride)) blkByte
      let d ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 32) "shared_d" n
      let k0 := Exp.mul kpair (Exp.litU32 2)
      let deq := fun (qb : Exp (.scalar .u32)) =>
        Exp.mul d (Exp.sub (Exp.toF32 qb) (Exp.mul (Exp.litF32 256.0) (Exp.toF32 (Exp.shiftRight qb (Exp.litU32 7)))))
      let q0b ← rdByte (Exp.add bb (Exp.add (Exp.litU32 2) k0))
      let q1b ← rdByte (Exp.add bb (Exp.add (Exp.litU32 3) k0))
      let y0 := deq q0b
      let y1 := deq q1b
      let ktile := Exp.div k0 (Exp.litU32 8)
      let kr := Exp.mod k0 (Exp.litU32 8)
      let blkB := Exp.add (Exp.mul ktile (Exp.litU32 4)) (Exp.div n (Exp.litU32 8))
      let baseB := Exp.add (Exp.mul blkB (Exp.litU32 64)) (Exp.mod n (Exp.litU32 8))
      ShaderM.writeWorkgroup (ty := .scalar .f16) "shared_B" (Exp.add baseB (Exp.mul kr (Exp.litU32 8))) (Exp.toF16 y0)
      ShaderM.writeWorkgroup (ty := .scalar .f16) "shared_B" (Exp.add baseB (Exp.mul (Exp.add kr (Exp.litU32 1)) (Exp.litU32 8))) (Exp.toF16 y1)
    ShaderM.barrier
    for k8 in [0:4] do
      for mt in [0:2] do
        let mtileG := Exp.add (Exp.mul sgRow (Exp.litU32 2)) (Exp.litU32 mt)
        let blkA := Exp.add (Exp.mul mtileG (Exp.litU32 4)) (Exp.litU32 k8)
        ShaderM.loadMatrixLeft (st := .f16) (m := 8) (k := 8) "Ax" mt "shared_A" (Exp.mul blkA (Exp.litU32 64)) (Exp.litU32 8)
      for nt in [0:2] do
        let ntileG := Exp.add (Exp.mul sgCol (Exp.litU32 2)) (Exp.litU32 nt)
        let blkB := Exp.add (Exp.mul (Exp.litU32 (k8 * 4)) (Exp.litU32 1)) ntileG
        ShaderM.loadMatrixRight (st := .f16) (k := 8) (n := 8) "Bx" nt "shared_B" (Exp.mul blkB (Exp.litU32 64)) (Exp.litU32 8)
      for mt in [0:2] do
        for nt in [0:2] do
          ShaderM.matrixMultiplyAccumulateMixed (inSt := .f16) (outSt := .f32) (m := 8) (k := 8) (n := 8) "Cx" (mt * 2 + nt) "Ax" mt "Bx" nt
    ShaderM.barrier
  -- store the 32×32 result tile to shared_C
  for mt in [0:2] do
    for nt in [0:2] do
      let rOff := Exp.add mOff (Exp.litU32 (mt * 8))
      let cOff := Exp.add nOff (Exp.litU32 (nt * 8))
      ShaderM.storeMatrixResult (st := .f32) (m := 8) (n := 8) "Cx" (mt * 2 + nt) "shared_C" (Exp.add (Exp.mul rOff (Exp.litU32 32)) cOff) (Exp.litU32 32)
  ShaderM.barrier
  -- scatter shared_C [32,32] → dst[slot, pos, col]; skip padding (slot ≥ nUsed)
  for s in [0:8] do
    let idx := Exp.add tid (Exp.litU32 (s * 128))
    let r := Exp.div idx (Exp.litU32 32)
    let c := Exp.mod idx (Exp.litU32 32)
    let rowGlobal := Exp.add rowBase r
    let slotR ← ShaderM.readBuffer (ty := .scalar .u32) (n := M) "slot" rowGlobal
    ShaderM.if_ (Exp.lt slotR (Exp.litU32 nUsed)) (do
      let posR ← ShaderM.readBuffer (ty := .scalar .u32) (n := M) "pos" rowGlobal
      let v ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 32 * 32) "shared_C" (Exp.add (Exp.mul r (Exp.litU32 32)) c)
      let colGlobal := Exp.add colBase c
      let dstIdx := Exp.add (Exp.add (Exp.mul slotR (Exp.litU32 (nTok * N))) (Exp.mul posR (Exp.litU32 N))) colGlobal
      ShaderM.writeBuffer (ty := .scalar .f32) "dst" dstIdx v) (pure ())

end Hesper.Quantization.Q4_K_M
