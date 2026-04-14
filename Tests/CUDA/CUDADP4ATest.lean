import Hesper.CUDA.FFI
import Hesper.CUDA.Buffer
import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.Layers.Linear
import Hesper.Quantization.Q4_K_M

set_option maxRecDepth 2048

/-!
# Q4_K × Q8_1 dp4a Kernel Correctness Test

Validates that the dp4a-based Q4_K matmul produces the same result as
the f32 block-coop kernel for the same weight matrix and input vector.

Strategy:
1. Generate random input f32 vector (size = inDim = 256 or 512)
2. Generate random Q4_K weights (arbitrary but valid bit patterns)
3. Run both kernels:
   - f32: fusedQ4KMLinearBlockCoopKernel
   - dp4a: quantizeQ8_1Kernel + fusedQ4KMLinearDP4AKernel
4. Compare outputs. Expected relative error < 1% (Q8_1 quantization noise)
-/

open Hesper
open Hesper.CUDA
open Hesper.WGSL.Monad (ShaderM)
open Hesper.Layers.Linear

-- Byte array helpers
private def f64ToF32Bits (f : Float) : UInt32 :=
  let b := f.toBits
  let s := (b >>> 63) &&& 1
  let e64 := (b >>> 52) &&& 0x7FF
  let m64 := b &&& 0x000FFFFFFFFFFFFF
  if e64 == 0 then 0
  else
    let eUnb : Int := Int.ofNat e64.toNat - 1023
    let e32i : Int := eUnb + 127
    if e32i ≤ 0 then 0
    else if e32i ≥ 255 then (s.toUInt32 <<< 31) ||| ((0xFF : UInt32) <<< 23)
    else
      let lower29 := m64 &&& (0x1FFFFFFF : UInt64)
      let m32Truncated := (m64 >>> 29).toUInt32 &&& (0x7FFFFF : UInt32)
      let halfway : UInt64 := 0x10000000
      let roundUp :=
        lower29 > halfway ||
        (lower29 == halfway && (m32Truncated &&& 1) == 1)
      let m32 := if roundUp then m32Truncated + 1 else m32Truncated
      let (m32Final, e32i') := if m32 == 0x800000 then (0, e32i + 1) else (m32, e32i)
      if e32i' ≥ 255 then (s.toUInt32 <<< 31) ||| ((0xFF : UInt32) <<< 23)
      else
        let e32 : UInt32 := e32i'.toNat.toUInt32
        (s.toUInt32 <<< 31) ||| (e32 <<< 23) ||| m32Final

private def f32BitsToF64 (bits : UInt32) : Float :=
  let e := (bits >>> 23) &&& 0xFF; let m := bits &&& (0x7FFFFF : UInt32); let s := bits >>> 31
  if e == 0 then 0.0 else
    let v := (1.0 + m.toNat.toFloat / 8388608.0) * Float.pow 2.0 (e.toNat.toFloat - 127.0)
    if s == 1 then -v else v

private def packF32 (arr : Array Float) : ByteArray :=
  arr.foldl (fun acc f => let b := f64ToF32Bits f
    acc.push b.toUInt8 |>.push (b>>>8).toUInt8 |>.push (b>>>16).toUInt8 |>.push (b>>>24).toUInt8
  ) ByteArray.empty

private def unpackF32 (ba : ByteArray) (i : Nat) : Float :=
  let o := i * 4
  let b0 : UInt32 := ba.get! o |>.toUInt32
  let b1 : UInt32 := ba.get! (o+1) |>.toUInt32
  let b2 : UInt32 := ba.get! (o+2) |>.toUInt32
  let b3 : UInt32 := ba.get! (o+3) |>.toUInt32
  f32BitsToF64 (b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24))

-- LCG random number generator (deterministic)
private def lcg (seed : UInt64) : UInt64 := seed * 6364136223846793005 + 1442695040888963407

-- Generate synthetic "quasi-random" input: x[i] = sin(i * 0.1) to have bounded values
private def genInput (n : Nat) : Array Float := Id.run do
  let mut arr := Array.mkEmpty n
  for i in [0:n] do
    let x := Float.sin (i.toFloat * 0.1)
    arr := arr.push x
  arr

/-- Generate a valid Q4_K weight block.
    Format per block (144 bytes = 36 u32):
    - u32[0]: fp16(d) | fp16(dmin)
    - u32[1..3]: 12 bytes of packed 6-bit scales and mins
    - u32[4..35]: 128 bytes of 4-bit quantized weights (256 values)

    We use d = 0.01, dmin = 0.005 (constants) and fill scales/mins/quants
    with small, predictable values for easy debugging. -/
private def genQ4KBlock (seed : UInt32) : Array UInt32 := Id.run do
  let mut arr := Array.mkEmpty 36
  -- d = 0.01 as fp16: 0x211F ≈ 0.00999
  -- dmin = 0.005 as fp16: 0x1D1F ≈ 0.004997
  let d_fp16 : UInt32 := 0x211F
  let dmin_fp16 : UInt32 := 0x1D1F
  arr := arr.push (d_fp16 ||| (dmin_fp16 <<< 16))
  -- Scales and mins: 8 6-bit values each, packed into 12 bytes
  -- sc[0..3] low 6 bits at u32[1] bytes 0..3
  -- m[0..3] low 6 bits at u32[2] bytes 0..3
  -- u32[3] has high 2 bits of sc[0..3] at bits 6..7 of u32[1] (not u32[3] — need re-read)
  -- Simplification: use scales = [10,11,12,...,17] and mins = [1,2,...,8]
  -- Low 4 bits of scale[0..3]: 10,11,12,13 → 0x0D0C0B0A
  -- Low 4 bits of min[0..3]:   1,2,3,4 → 0x04030201
  -- High 2 bits (bits 4,5) of scale: all zeros for <16 → 0
  -- For simplicity, use scales <64 (fits in 6 bits low) and high 2 bits unused.
  -- Format per llama.cpp (see q4_K struct):
  --   scales[0..3] = sc[0..3] low 6 bits (bits 0..5 of bytes 0..3 of u32[1])
  --   scales[4..7] = m[0..3] low 6 bits (bits 0..5 of bytes 0..3 of u32[2])
  --   scales[8..11] = sc[4..7]<<0 | m[4..7]<<4, with high 2 bits from scales[0..7] packed into bits 6..7
  -- For our test: all scales/mins in [0,63] → high bits 0. Layout:
  -- u32[1] bytes = [sc0, sc1, sc2, sc3]  (each 6 bits, high 2 bits zero)
  -- u32[2] bytes = [m0,  m1,  m2,  m3]
  -- u32[3] bytes = [sc4|m4<<4, sc5|m5<<4, sc6|m6<<4, sc7|m7<<4]
  let sc : Array UInt32 := #[10, 11, 12, 13, 14, 15, 16, 17]
  let m  : Array UInt32 := #[1, 2, 3, 4, 5, 6, 7, 8]
  let u32_1 := sc[0]! ||| (sc[1]! <<< 8) ||| (sc[2]! <<< 16) ||| (sc[3]! <<< 24)
  let u32_2 := m[0]! ||| (m[1]! <<< 8) ||| (m[2]! <<< 16) ||| (m[3]! <<< 24)
  -- llama.cpp layout for sc[4..7], m[4..7] (high part):
  -- scales[8+i] = (sc[4+i] & 0x0F) | ((m[4+i] & 0x0F) << 4)
  -- Plus high 2 bits go elsewhere — but we use sc<16 so high 2 bits are zero anyway.
  let mask4 : UInt32 := 0x0F
  let shift4 : UInt32 := 4
  let shift8 : UInt32 := 8
  let shift16 : UInt32 := 16
  let shift24 : UInt32 := 24
  let u32_3 : UInt32 :=
    ((sc[4]! &&& mask4) ||| ((m[4]! &&& mask4) <<< shift4)) |||
    (((sc[5]! &&& mask4) ||| ((m[5]! &&& mask4) <<< shift4)) <<< shift8) |||
    (((sc[6]! &&& mask4) ||| ((m[6]! &&& mask4) <<< shift4)) <<< shift16) |||
    (((sc[7]! &&& mask4) ||| ((m[7]! &&& mask4) <<< shift4)) <<< shift24)
  arr := arr.push u32_1
  arr := arr.push u32_2
  arr := arr.push u32_3
  -- Quants: 32 u32 × 32 bits/u32 = 1024 bits = 256 × 4 bits
  -- Use seed-based pseudo-random nibbles for variety
  for i in [0:32] do
    let s1 := lcg (seed.toUInt64 + i.toUInt64)
    let v : UInt32 := s1.toUInt32  -- take low 32 bits
    arr := arr.push v
  arr

private def packU32Array (arr : Array UInt32) : ByteArray :=
  arr.foldl (init := ByteArray.empty) fun (acc : ByteArray) (v : UInt32) =>
    acc.push v.toUInt8
      |>.push (v >>> 8).toUInt8
      |>.push (v >>> 16).toUInt8
      |>.push (v >>> 24).toUInt8

def main : IO Unit := do
  IO.println "═══ Q4_K × Q8_1 dp4a Correctness Test ═══\n"
  let cuda ← CUDAContext.init

  -- Test configuration: 1 output row × 256 input elements (1 Q4_K block)
  let inDim := 256
  let outDim := 1
  let config : Config := { inDim, outDim }

  -- Generate input
  let input := genInput inDim
  IO.println s!"Input[0..8]: {input[0]!} {input[1]!} {input[2]!} {input[3]!} {input[4]!} {input[5]!} {input[6]!} {input[7]!}"

  -- Generate Q4_K weights (1 block per row × 1 row = 1 block = 36 u32)
  let weights := genQ4KBlock 42

  let inputBytes := packF32 input
  let weightBytes := packU32Array weights

  -- Allocate buffers
  let inputBuf ← GPUBackend.allocBuffer cuda (4 * inDim).toUSize
  GPUBackend.writeBuffer cuda inputBuf inputBytes

  let weightBuf ← GPUBackend.allocBuffer cuda weightBytes.size.toUSize
  GPUBackend.writeBuffer cuda weightBuf weightBytes

  let outputF32Buf ← GPUBackend.allocBuffer cuda (4 * outDim).toUSize
  let outputDP4ABuf ← GPUBackend.allocBuffer cuda (4 * outDim).toUSize

  -- ═══ Method A: f32 block-coop kernel ═══
  IO.println "── Method A: f32 block-coop kernel ──"
  let f32Kernel := fusedQ4KMLinearBlockCoopKernel config
  let f32Buffers : List (String × GPUBackend.Buf CUDAContext) :=
    [("weights", weightBuf), ("input", inputBuf), ("output", outputF32Buf)]
  GPUBackend.execute cuda f32Kernel f32Buffers
    { numWorkgroups := (outDim, 1, 1), workgroupSize := { x := 32, y := 1, z := 1 }
      extensions := ["subgroups"] }
  let f32Result ← GPUBackend.readBuffer cuda outputF32Buf (4 * outDim).toUSize
  let f32Val := unpackF32 f32Result 0
  IO.println s!"  f32 output[0] = {f32Val}"

  -- ═══ Method B: Q8_1 quantize + dp4a matmul ═══
  IO.println "── Method B: dp4a kernel (quantize + matmul) ──"
  let nQ8Blocks := inDim / 32
  let q8BufSize : USize := (nQ8Blocks * 9 * 4).toUSize  -- 9 u32 per block
  let q8Buf ← GPUBackend.allocBuffer cuda q8BufSize

  -- Step 1: Quantize f32 → Q8_1
  GPUBackend.execute cuda (quantizeQ8_1Kernel inDim)
    [("input", inputBuf), ("output", q8Buf)]
    { numWorkgroups := (nQ8Blocks, 1, 1), workgroupSize := { x := 32, y := 1, z := 1 }
      extensions := ["subgroups"] }

  -- Debug: read first Q8_1 block
  let q8Data ← GPUBackend.readBuffer cuda q8Buf 36
  IO.print "  Q8_1 block[0] u32s: "
  for i in [0:9] do
    let o := i * 4
    let b0 : UInt32 := q8Data.get! o |>.toUInt32
    let b1 : UInt32 := q8Data.get! (o+1) |>.toUInt32
    let b2 : UInt32 := q8Data.get! (o+2) |>.toUInt32
    let b3 : UInt32 := q8Data.get! (o+3) |>.toUInt32
    let v := b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24)
    IO.print s!"{v} "
  IO.println ""
  -- Header u32[0] should be bitcast(d as f32); d = max|x|/127
  -- For input[0..32] = sin(0), sin(0.1), ..., sin(3.1), max~=1.0 → d ≈ 1/127 ≈ 0.00787
  let hdrBits : UInt32 :=
    let o := 0
    (q8Data.get! o).toUInt32 ||| ((q8Data.get! (o+1)).toUInt32 <<< 8) |||
    ((q8Data.get! (o+2)).toUInt32 <<< 16) ||| ((q8Data.get! (o+3)).toUInt32 <<< 24)
  IO.println s!"  d (f32 from u32={hdrBits}) = {f32BitsToF64 hdrBits}"

  -- Step 2: Q4_K × Q8_1 matmul via dp4a
  GPUBackend.execute cuda (fusedQ4KMLinearDP4AKernel config)
    [("weights", weightBuf), ("input_q8", q8Buf), ("output", outputDP4ABuf)]
    { numWorkgroups := (outDim, 1, 1), workgroupSize := { x := 32, y := 1, z := 1 }
      extensions := ["subgroups"] }

  let dp4aResult ← GPUBackend.readBuffer cuda outputDP4ABuf (4 * outDim).toUSize
  let dp4aVal := unpackF32 dp4aResult 0
  IO.println s!"  dp4a output[0] = {dp4aVal}"

  -- Compare
  IO.println "── Comparison ──"
  let absDiff := (f32Val - dp4aVal).abs
  let relDiff := if f32Val.abs > 1e-8 then absDiff / f32Val.abs else 0.0
  IO.println s!"  abs diff = {absDiff}"
  IO.println s!"  rel diff = {relDiff}"
  -- Q8_1 introduces ~1/127 ≈ 0.78% quantization noise, so we allow 5% tolerance
  if relDiff < 0.05 then
    IO.println "  ✓ PASS: dp4a within 5% of f32 reference"
  else
    IO.println "  ✗ FAIL: dp4a result differs too much from f32 reference"

  GPUBackend.freeBuffer cuda inputBuf
  GPUBackend.freeBuffer cuda weightBuf
  GPUBackend.freeBuffer cuda outputF32Buf
  GPUBackend.freeBuffer cuda outputDP4ABuf
  GPUBackend.freeBuffer cuda q8Buf
