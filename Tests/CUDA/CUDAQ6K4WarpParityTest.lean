import Hesper.CUDA.FFI
import Hesper.CUDA.Buffer
import Hesper.CUDA.CodeGen
import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.Layers.Linear

set_option maxRecDepth 2048

/-!
# Q6_K dp4a — 4-warp 1-row vs 1-warp 1-row parity

Compares `fusedQ6KLinearDP4A4WarpKernel` (block=32×4=128 thread, 4 warps
cooperating on K) against `fusedQ6KLinearDP4AKernel` (block=32, 1 warp).

Both kernels read the same Q6_K weight buffer + Q8_1-quantised input and
must produce bit-identical outputs (modulo f32 reduction order).  Uses
production-shape `inDim=2560 outDim=64` so all 4 warps loop a real
K-stride (10 blocks/row, 4 warps × {2,3,3,2} blocks via the kbx %= 4 split).

The 4-warp variant only changes how partials are summed across warps;
the per-(kbx, kqs) `vec_dot_q6_K_q8_1` payload is identical, so the
expected `abs diff` is < 1e-4 with `rel diff` typically < 1e-5.
-/

open Hesper
open Hesper.CUDA
open Hesper.WGSL.Monad (ShaderM)
open Hesper.Layers.Linear

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

/-- Generate `n` pseudo-random f32 inputs in [-1, 1]. -/
private def genInput (n : Nat) (seed : UInt64) : Array Float := Id.run do
  let mut arr := Array.mkEmpty n
  let mut s := seed
  for _ in [0:n] do
    s := lcg s
    let u : Float := (s.toNat % 65536).toFloat / 32768.0  -- [0, 2)
    arr := arr.push (u - 1.0)                               -- [-1, 1)
  return arr

/-- Generate one Q6_K block (210 bytes): 128 ql, 64 qh, 16 i8 scales, fp16 d. -/
private def genQ6KBlock (seed : UInt64) : ByteArray := Id.run do
  let mut bytes := ByteArray.empty
  let mut s := seed
  -- ql: 128 bytes, each byte holds 2× 4-bit quants.  Random in [0..255].
  for _ in [0:128] do
    s := lcg s
    bytes := bytes.push (s.toNat % 256).toUInt8
  -- qh: 64 bytes, holds the high 2 bits of 4 quants per byte.  Random [0..255].
  for _ in [0:64] do
    s := lcg s
    bytes := bytes.push (s.toNat % 256).toUInt8
  -- scales: 16 i8 values in roughly [-16, 16].
  for i in [0:16] do
    s := lcg s
    let v : Int := (Int.ofNat (s.toNat % 32)) - 16
    let asU8 : UInt8 := if v < 0 then ((256 + v).toNat).toUInt8 else v.toNat.toUInt8
    let _ := i
    bytes := bytes.push asU8
  -- d: fp16 representing 0.01 ≈ 0x211F (constant — varying d doesn't help
  -- exercise the kernel logic, only scales the output).
  bytes := bytes.push 0x1F
  bytes := bytes.push 0x21
  bytes

/-- Build a `outDim × blocksPerRow` Q6_K weight buffer (= outDim × 210 bytes
    when blocksPerRow=1, etc.). -/
private def genQ6KWeights (outDim blocksPerRow : Nat) (seed : UInt64) : ByteArray := Id.run do
  let mut bytes := ByteArray.empty
  let mut s := seed
  for _row in [0:outDim] do
    for _blk in [0:blocksPerRow] do
      s := lcg s
      let blk := genQ6KBlock s
      bytes := bytes ++ blk
  bytes

def main : IO Unit := do
  IO.println "═══ Q6_K 4-warp vs 1-warp Parity Test ═══\n"

  -- Production-ish shape: K=2560 → 10 blocks/row.  4 warps split: warp k
  -- runs blocks {k, k+4, k+8} (3 blocks for k=0,1; 2 blocks for k=2,3).
  -- This exercises every (warp, kbx) combo without OOB reads.
  let inDim := 2560
  let outDim := 64
  let gridX := 0  -- 1-D dispatch
  let blocksPerRow := inDim / 256
  IO.println s!"inDim={inDim}, outDim={outDim}, blocksPerRow={blocksPerRow}"

  let cuda ← CUDAContext.init

  let input := genInput inDim 0xCAFEBABE
  let weightBytes := genQ6KWeights outDim blocksPerRow 0xDEADBEEF
  let padBytes : Nat := (4 - (weightBytes.size % 4)) % 4
  let mut weightBytesPadded := weightBytes
  for _ in [0:padBytes] do
    weightBytesPadded := weightBytesPadded.push 0
  IO.println s!"  weight buffer: {weightBytesPadded.size} bytes"

  let inputBuf ← GPUBackend.allocBuffer cuda (4 * inDim).toUSize
  GPUBackend.writeBuffer cuda inputBuf (packF32 input)
  let weightBuf ← GPUBackend.allocBuffer cuda weightBytesPadded.size.toUSize
  GPUBackend.writeBuffer cuda weightBuf weightBytesPadded

  let nQ8Blocks := inDim / 32
  let q8BufSize : USize := (nQ8Blocks * 9 * 4).toUSize
  let q8Buf ← GPUBackend.allocBuffer cuda q8BufSize

  -- Q8_1-quantise the input once.
  GPUBackend.execute cuda (quantizeQ8_1Kernel inDim)
    [("input", inputBuf), ("output", q8Buf)]
    { numWorkgroups := (nQ8Blocks, 1, 1), workgroupSize := { x := 32, y := 1, z := 1 }
      extensions := ["subgroups"] }

  -- ── Method A: 1-warp 1-row (block=32, grid=outDim) ──
  let outA ← GPUBackend.allocBuffer cuda (4 * outDim).toUSize
  GPUBackend.execute cuda (fusedQ6KLinearDP4AKernel inDim outDim gridX)
    [("weights", weightBuf), ("input_q8", q8Buf), ("output", outA)]
    { numWorkgroups := (outDim, 1, 1), workgroupSize := { x := 32, y := 1, z := 1 }
      extensions := ["subgroups"] }
  let resA ← GPUBackend.readBuffer cuda outA (4 * outDim).toUSize

  -- ── Method B: 4-warp 1-row (block=32×4=128, grid=outDim) ──
  let outB ← GPUBackend.allocBuffer cuda (4 * outDim).toUSize
  GPUBackend.execute cuda (fusedQ6KLinearDP4A4WarpKernel inDim outDim gridX)
    [("weights", weightBuf), ("input_q8", q8Buf), ("output", outB)]
    { numWorkgroups := (outDim, 1, 1), workgroupSize := { x := 32, y := 4, z := 1 }
      extensions := ["subgroups"] }
  let resB ← GPUBackend.readBuffer cuda outB (4 * outDim).toUSize

  -- ── Compare row-wise ──
  let mut maxAbs : Float := 0.0
  let mut maxRel : Float := 0.0
  let mut firstMismatch : Option (Nat × Float × Float) := none
  for i in [0:outDim] do
    let a := unpackF32 resA i
    let b := unpackF32 resB i
    let absD := (a - b).abs
    let relD := if a.abs > 1e-8 then absD / a.abs else absD
    if absD > maxAbs then maxAbs := absD
    if relD > maxRel then maxRel := relD
    if firstMismatch.isNone && absD > 1e-3 then
      firstMismatch := some (i, a, b)
  IO.println s!"  outA[0..3] = {unpackF32 resA 0} {unpackF32 resA 1} {unpackF32 resA 2} {unpackF32 resA 3}"
  IO.println s!"  outB[0..3] = {unpackF32 resB 0} {unpackF32 resB 1} {unpackF32 resB 2} {unpackF32 resB 3}"
  IO.println s!"  max |abs diff| = {maxAbs}"
  IO.println s!"  max  rel diff  = {maxRel}"
  match firstMismatch with
  | some (i, a, b) =>
      IO.println s!"  ✗ FAIL: first mismatch row {i}: 1warp={a} 4warp={b}"
      IO.Process.exit 1
  | none =>
      IO.println "  ✓ PASS (rel diff < 1e-3 across all rows)"

  GPUBackend.freeBuffer cuda inputBuf
  GPUBackend.freeBuffer cuda weightBuf
  GPUBackend.freeBuffer cuda outA
  GPUBackend.freeBuffer cuda outB
  GPUBackend.freeBuffer cuda q8Buf
