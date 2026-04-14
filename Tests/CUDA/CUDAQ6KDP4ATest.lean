import Hesper.CUDA.FFI
import Hesper.CUDA.Buffer
import Hesper.CUDA.CodeGen
import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.Layers.Linear
import Hesper.Quantization.Q6_K

set_option maxRecDepth 2048

/-!
# Q6_K × Q8_1 dp4a Correctness Test

Compares hesper's `fusedQ6KLinearDP4AKernel` against:
1. The existing f32 `fusedQ6KLinearBlockCoopKernel` (uses raw f32 input)
2. An nvcc-compiled reference (run separately via /tmp/ref_q6k)
   on the SAME synthetic Q6_K block + Q8_1 quantized input.
-/

open Hesper
open Hesper.CUDA
open Hesper.WGSL.Monad (ShaderM)
open Hesper.Layers.Linear

private def toHex (v : UInt32) : String :=
  let hex := "0123456789ABCDEF"
  let h i := (hex.get ⟨Nat.min i.toNat 15⟩).toString
  h ((v >>> 28) &&& 0xF) ++ h ((v >>> 24) &&& 0xF) ++
  h ((v >>> 20) &&& 0xF) ++ h ((v >>> 16) &&& 0xF) ++
  h ((v >>> 12) &&& 0xF) ++ h ((v >>> 8) &&& 0xF) ++
  h ((v >>> 4) &&& 0xF) ++ h (v &&& 0xF)

-- Proper IEEE 754 binary64 → binary32 conversion with round-to-nearest-even.
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
  arr.foldl (init := ByteArray.empty) fun acc f => let b := f64ToF32Bits f
    acc.push b.toUInt8 |>.push (b>>>8).toUInt8 |>.push (b>>>16).toUInt8 |>.push (b>>>24).toUInt8

private def unpackF32 (ba : ByteArray) (i : Nat) : Float :=
  let o := i * 4
  let b0 : UInt32 := ba.get! o |>.toUInt32
  let b1 : UInt32 := ba.get! (o+1) |>.toUInt32
  let b2 : UInt32 := ba.get! (o+2) |>.toUInt32
  let b3 : UInt32 := ba.get! (o+3) |>.toUInt32
  f32BitsToF64 (b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24))

private def lcg (seed : UInt64) : UInt64 := seed * 6364136223846793005 + 1442695040888963407

private def genInput (n : Nat) : Array Float := Id.run do
  let mut arr := Array.mkEmpty n
  for i in [0:n] do
    arr := arr.push (Float.sin (i.toFloat * 0.1))
  arr

/-- Generate a valid Q6_K block (210 bytes).
    Layout:  ql[128] qh[64] scales[16] d(fp16)
    Use deterministic pseudo-random nibbles and small scale values. -/
private def genQ6KBlock : ByteArray := Id.run do
  let mut bytes : ByteArray := ByteArray.empty
  -- ql[128]: 256 nibbles = 4-bit lower quants
  for i in [0:128] do
    let s := lcg (42 + i.toUInt64)
    bytes := bytes.push s.toUInt8
  -- qh[64]: 256 × 2-bit upper quants packed 4-per-byte
  for i in [0:64] do
    let s := lcg (1000 + i.toUInt64)
    bytes := bytes.push s.toUInt8
  -- scales[16]: int8 in [-16, 14] (small magnitudes for sane output).
  -- Stored as unsigned bytes; negative values use two's complement.
  for i in [0:16] do
    let s : Int := (Int.ofNat i - 8) * 2  -- -16, -14, ..., 14
    let asU8 : UInt8 := if s < 0 then ((256 + s).toNat).toUInt8 else s.toNat.toUInt8
    bytes := bytes.push asU8
  -- d (fp16): use d = 0.01.  0.01 fp16 ≈ 0x211F
  bytes := bytes.push 0x1F  -- low byte
  bytes := bytes.push 0x21  -- high byte
  bytes

def main : IO Unit := do
  IO.println "═══ Q6_K × Q8_1 dp4a Correctness Test ═══\n"

  let inDim := 256
  let outDim := 1
  let gridX := 0  -- 1D dispatch for testing

  -- Dump PTX for analysis.
  let ptx := Hesper.CUDA.CodeGen.generatePTX "main" { x := 32, y := 1, z := 1 }
    (fusedQ6KLinearDP4AKernel inDim outDim gridX)
  IO.FS.writeFile "/tmp/hesper_q6k_dp4a.ptx" ptx
  IO.println s!"[PTX] Wrote {ptx.length} chars to /tmp/hesper_q6k_dp4a.ptx"

  let cuda ← CUDAContext.init

  -- Generate input
  let input := genInput inDim
  IO.println s!"Input[0..7]: {input[0]!} {input[1]!} {input[2]!} {input[3]!}"

  let weightBytes := genQ6KBlock
  IO.println s!"Q6_K block: {weightBytes.size} bytes (expected 210)"

  -- Pad weights to next u32 boundary (kernel reads as u32 buffer)
  let padBytes : Nat := (4 - (weightBytes.size % 4)) % 4
  let mut weightBytesPadded := weightBytes
  for _ in [0:padBytes] do
    weightBytesPadded := weightBytesPadded.push 0

  IO.FS.writeBinFile "/tmp/test_q6k_weights.bin" weightBytes
  let inputBytes := packF32 input

  -- Allocate buffers
  let inputBuf ← GPUBackend.allocBuffer cuda (4 * inDim).toUSize
  GPUBackend.writeBuffer cuda inputBuf inputBytes
  let weightBuf ← GPUBackend.allocBuffer cuda weightBytesPadded.size.toUSize
  GPUBackend.writeBuffer cuda weightBuf weightBytesPadded

  let outputF32Buf ← GPUBackend.allocBuffer cuda (4 * outDim).toUSize
  let outputDP4ABuf ← GPUBackend.allocBuffer cuda (4 * outDim).toUSize

  -- ═══ Method A: f32 block-coop kernel (existing, used in production) ═══
  IO.println "\n── Method A: f32 block-coop Q6_K kernel ──"
  let f32Kernel := Hesper.Quantization.Q6_K.fusedQ6KLinearBlockCoopKernel
    inDim outDim gridX
  GPUBackend.execute cuda f32Kernel
    [("weights", weightBuf), ("input", inputBuf), ("output", outputF32Buf)]
    { numWorkgroups := (outDim, 1, 1), workgroupSize := { x := 32, y := 1, z := 1 }
      extensions := ["subgroups"] }
  let f32Result ← GPUBackend.readBuffer cuda outputF32Buf (4 * outDim).toUSize
  let f32Val := unpackF32 f32Result 0
  IO.println s!"  f32 output[0] = {f32Val}"

  -- ═══ Method B: Q8_1 quantize + Q6_K dp4a matmul ═══
  IO.println "\n── Method B: dp4a (quantize Q8_1 + Q6_K matmul) ──"
  let nQ8Blocks := inDim / 32
  let q8BufSize : USize := (nQ8Blocks * 9 * 4).toUSize
  let q8Buf ← GPUBackend.allocBuffer cuda q8BufSize

  -- Step 1: Quantize input → Q8_1
  GPUBackend.execute cuda (quantizeQ8_1Kernel inDim)
    [("input", inputBuf), ("output", q8Buf)]
    { numWorkgroups := (nQ8Blocks, 1, 1), workgroupSize := { x := 32, y := 1, z := 1 }
      extensions := ["subgroups"] }

  -- Dump quantized input for nvcc reference comparison.
  let q8DataForDump ← GPUBackend.readBuffer cuda q8Buf q8BufSize
  IO.FS.writeBinFile "/tmp/test_q6k_q8input.bin" q8DataForDump

  -- Step 2: Q6_K × Q8_1 dp4a
  GPUBackend.execute cuda (fusedQ6KLinearDP4AKernel inDim outDim gridX)
    [("weights", weightBuf), ("input_q8", q8Buf), ("output", outputDP4ABuf)]
    { numWorkgroups := (outDim, 1, 1), workgroupSize := { x := 32, y := 1, z := 1 }
      extensions := ["subgroups"] }

  let dp4aResult ← GPUBackend.readBuffer cuda outputDP4ABuf (4 * outDim).toUSize
  let dp4aVal := unpackF32 dp4aResult 0
  IO.println s!"  dp4a output[0] = {dp4aVal}"

  -- Compare
  IO.println "\n── Comparison ──"
  let absDiff := (f32Val - dp4aVal).abs
  let relDiff := if f32Val.abs > 1e-8 then absDiff / f32Val.abs else 0.0
  IO.println s!"  abs diff = {absDiff}"
  IO.println s!"  rel diff = {relDiff}"
  IO.println s!"\nNext: run /tmp/ref_q6k /tmp/test_q6k_weights.bin /tmp/test_q6k_q8input.bin"
  IO.println "and compare against the dp4a output above."

  GPUBackend.freeBuffer cuda inputBuf
  GPUBackend.freeBuffer cuda weightBuf
  GPUBackend.freeBuffer cuda outputF32Buf
  GPUBackend.freeBuffer cuda outputDP4ABuf
  GPUBackend.freeBuffer cuda q8Buf
