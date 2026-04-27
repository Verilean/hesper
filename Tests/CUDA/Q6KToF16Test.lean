import Hesper.CUDA.FFI
import Hesper.CUDA.Buffer
import Hesper.CUDA.CodeGen
import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.Quantization.Q6_K
import Hesper.Quantization.Q6KDequant
import Hesper.WGSL.MatMul

set_option maxRecDepth 2048

/-!
# Q6_K → f16 dequant kernel parity test

Verifies that the new `q6kToF16Kernel` (one-shot dequantization run at
load time for the LM head) produces an f16 weight buffer that, when
fed through `matMulTransposeF16BlockCoopKernel`, matches the f32
output of `fusedQ6KLinearBlockCoopKernel` (the existing on-the-fly
Q6_K × f32 matmul).

Used to validate the lm_head migration plan: store f16 weights, use
f16 matmul, drop the per-token Q6_K dequant cost (1140 µs → ~114 µs).
-/

open Hesper
open Hesper.CUDA
open Hesper.WGSL.Monad (ShaderM)

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
  let e := (bits >>> 23) &&& 0xFF
  let m := bits &&& (0x7FFFFF : UInt32)
  let s := bits >>> 31
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

/-- Generate one valid Q6_K block (210 bytes) seeded by `seed` so that
    distinct blocks yield distinct content (good for catching off-by-one
    bugs across rows / blocks). -/
private def genQ6KBlock (seed : Nat) : ByteArray := Id.run do
  let mut bytes : ByteArray := ByteArray.empty
  for i in [0:128] do
    let s := lcg ((42 + seed).toUInt64 + i.toUInt64)
    bytes := bytes.push s.toUInt8
  for i in [0:64] do
    let s := lcg ((1000 + seed).toUInt64 + i.toUInt64)
    bytes := bytes.push s.toUInt8
  for i in [0:16] do
    let s : Int := (Int.ofNat (i + (seed % 4)) - 8) * 2
    let asU8 : UInt8 := if s < 0 then ((256 + s).toNat).toUInt8 else s.toNat.toUInt8
    bytes := bytes.push asU8
  -- d (fp16): use 0.01 (= 0x211F) for all blocks
  bytes := bytes.push 0x1F
  bytes := bytes.push 0x21
  bytes

private def runTest (cuda : CUDAContext) (inDim outDim : Nat) : IO Unit := do
  let gridX := 0
  IO.println s!"\n═══ Q6_K → f16 dequant parity (inDim={inDim}, outDim={outDim}) ═══"

  let blocksPerRow := inDim / 256
  let totalBlocks := outDim * blocksPerRow
  let input := genInput inDim
  let mut weightBytes : ByteArray := ByteArray.empty
  for b in [0 : totalBlocks] do
    weightBytes := weightBytes ++ genQ6KBlock b
  let padBytes : Nat := (4 - (weightBytes.size % 4)) % 4
  for _ in [0:padBytes] do
    weightBytes := weightBytes.push 0

  let inputBuf ← GPUBackend.allocBuffer cuda (4 * inDim).toUSize
  GPUBackend.writeBuffer cuda inputBuf (packF32 input)
  let weightBuf ← GPUBackend.allocBuffer cuda weightBytes.size.toUSize
  GPUBackend.writeBuffer cuda weightBuf weightBytes

  -- Method A: existing on-the-fly Q6_K × f32 matmul
  let outputAbuf ← GPUBackend.allocBuffer cuda (4 * outDim).toUSize
  let f32Kernel := Hesper.Quantization.Q6_K.fusedQ6KLinearBlockCoopKernel inDim outDim gridX
  GPUBackend.execute cuda f32Kernel
    [("weights", weightBuf), ("input", inputBuf), ("output", outputAbuf)]
    { numWorkgroups := (outDim, 1, 1), workgroupSize := { x := 32, y := 1, z := 1 }
      extensions := ["subgroups"] }
  let resA ← GPUBackend.readBuffer cuda outputAbuf (4 * outDim).toUSize
  let valA := unpackF32 resA 0

  -- Method B: q6kToF16Kernel (one-shot dequant) + matMulTransposeF16BlockCoop
  let totalOutU32 := outDim * (inDim / 2)
  let f16WeightBuf ← GPUBackend.allocBuffer cuda (4 * totalOutU32).toUSize
  let dequantKernel := Hesper.Quantization.Q6_K.q6kToF16Kernel inDim outDim
  GPUBackend.execute cuda dequantKernel
    [("weights", weightBuf), ("output", f16WeightBuf)]
    { numWorkgroups := (totalBlocks, 1, 1), workgroupSize := { x := 64, y := 1, z := 1 }
      extensions := [] }

  let outputBbuf ← GPUBackend.allocBuffer cuda (4 * outDim).toUSize
  let mmConfig : Hesper.WGSL.MatMul.Config := { M := 1, N := outDim, K := inDim }
  Hesper.WGSL.MatMul.executeMatMulTransposeF16BlockCoop cuda inputBuf f16WeightBuf outputBbuf mmConfig
  let resB ← GPUBackend.readBuffer cuda outputBbuf (4 * outDim).toUSize
  let valB := unpackF32 resB 0

  -- Compare every output element
  let mut maxAbs : Float := 0.0
  let mut maxRel : Float := 0.0
  for i in [0 : outDim] do
    let a := unpackF32 resA i
    let b := unpackF32 resB i
    let ad := (a - b).abs
    let rd := if a.abs > 1e-8 then ad / a.abs else 0.0
    if ad > maxAbs then maxAbs := ad
    if rd > maxRel then maxRel := rd
  IO.println s!"  out[0] A={valA}  B={valB}"
  IO.println s!"  max abs = {maxAbs}, max rel = {maxRel}"
  if maxRel < 0.01 || maxAbs < 0.01 then
    IO.println "  ✓ PASS"
  else
    IO.println "  ✗ FAIL"

  GPUBackend.freeBuffer cuda inputBuf
  GPUBackend.freeBuffer cuda weightBuf
  GPUBackend.freeBuffer cuda outputAbuf
  GPUBackend.freeBuffer cuda outputBbuf
  GPUBackend.freeBuffer cuda f16WeightBuf

def main : IO Unit := do
  IO.println "═══ Q6_K → f16 packed dequant Parity Test ═══"
  let cuda ← CUDAContext.init
  runTest cuda 256 1
  runTest cuda 256 4
  runTest cuda 1280 1
  runTest cuda 2560 1
  runTest cuda 2560 8
  IO.println "\n═══ All tests done ═══"
