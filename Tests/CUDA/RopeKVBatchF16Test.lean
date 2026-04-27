import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.Layers.Attention
import Hesper.Models.Gemma4.Kernels

/-!
# fusedRopeKAndCacheWriteBatchKernelF16 parity test

Compares the f32 batched RoPE-K + KV-cache writer
(`fusedRopeKAndCacheWriteBatchKernel` in Models/Gemma4/Kernels.lean) and the
new f16 packed-cache version (`fusedRopeKAndCacheWriteBatchKernelF16`).

Same input new_k / new_v / params / freq_factors → both kernels write the
same logical (kvHead, pos, dim) cells, just to different cache buffers.
The f32 buffer is unpacked plainly; the f16 buffer is unpacked via
unpack2x16float.  Diff is compared element-wise within f16 precision.

Run: `lake exe cuda-rope-kv-batch-f16-parity`
-/

open Hesper
open Hesper.CUDA
open Hesper.WGSL
open Hesper.WGSL.Monad (ShaderM)

private def packFloats (arr : Array Float) : ByteArray :=
  arr.foldl (init := ByteArray.empty) fun (acc : ByteArray) (f : Float) =>
    let bits64 : UInt64 := f.toBits
    let s := (bits64 >>> 63) &&& 1
    let e := (bits64 >>> 52) &&& 0x7FF
    let m := bits64 &&& 0x000FFFFFFFFFFFFF
    let n : UInt32 :=
      if e == 0 then 0
      else
        let e32 : Int := e.toNat - 1023 + 127
        if e32 <= 0 then 0
        else if e32 >= 255 then (s.toUInt32 <<< 31) ||| ((0xFF : UInt32) <<< 23)
        else (s.toUInt32 <<< 31) ||| (e32.toNat.toUInt32 <<< 23) |||
             ((m >>> 29).toUInt32 &&& (0x7FFFFF : UInt32))
    acc.push n.toUInt8 |>.push (n>>>8).toUInt8
       |>.push (n>>>16).toUInt8 |>.push (n>>>24).toUInt8

private def packU32s (arr : Array UInt32) : ByteArray :=
  arr.foldl (init := ByteArray.empty) fun (acc : ByteArray) (n : UInt32) =>
    acc.push n.toUInt8 |>.push (n>>>8).toUInt8
       |>.push (n>>>16).toUInt8 |>.push (n>>>24).toUInt8

private def unpackF32 (ba : ByteArray) (i : Nat) : Float :=
  let o := i * 4
  let b0 : UInt32 := ba.get! o |>.toUInt32
  let b1 : UInt32 := ba.get! (o+1) |>.toUInt32
  let b2 : UInt32 := ba.get! (o+2) |>.toUInt32
  let b3 : UInt32 := ba.get! (o+3) |>.toUInt32
  let bits := b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24)
  let e := (bits >>> 23) &&& 0xFF
  let m := bits &&& (0x7FFFFF : UInt32)
  let s := bits >>> 31
  if e == 0 then 0.0 else
    let v := (1.0 + m.toNat.toFloat / 8388608.0) * Float.pow 2.0 (e.toNat.toFloat - 127.0)
    if s == 1 then -v else v

private def unpackU32 (ba : ByteArray) (i : Nat) : UInt32 :=
  let o := i * 4
  let b0 : UInt32 := ba.get! o |>.toUInt32
  let b1 : UInt32 := ba.get! (o+1) |>.toUInt32
  let b2 : UInt32 := ba.get! (o+2) |>.toUInt32
  let b3 : UInt32 := ba.get! (o+3) |>.toUInt32
  b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24)

private def unpackHalf2 (u : UInt32) : Float × Float :=
  let halfToFloat (h : UInt32) : Float :=
    let s := (h >>> 15) &&& 1
    let e := (h >>> 10) &&& 0x1F
    let m := h &&& 0x3FF
    let absV : Float :=
      if e == 0 then
        if m == 0 then 0.0
        else m.toNat.toFloat * Float.pow 2.0 (-24.0)
      else if e == 31 then 0.0
      else
        (1.0 + m.toNat.toFloat / 1024.0) * Float.pow 2.0 (e.toNat.toFloat - 15.0)
    if s == 1 then -absV else absV
  (halfToFloat (u &&& 0xFFFF), halfToFloat (u >>> 16))

def runRopeKVBatchF16Case (ctx : CUDAContext)
    (numKVHeads maxSeqLen headDim seqLen startPos : Nat) (ropeBase : Float) :
    IO Float := do
  let halfDim := headDim / 2
  let kvDim := numKVHeads * headDim
  let cacheSize := numKVHeads * maxSeqLen * headDim
  let cacheU32Size := cacheSize / 2
  let inSize := (kvDim * seqLen * 4).toUSize

  let newKBuf ← GPUBackend.allocBuffer ctx inSize
  let newVBuf ← GPUBackend.allocBuffer ctx inSize
  let kF32Buf ← GPUBackend.allocBuffer ctx (cacheSize * 4).toUSize
  let vF32Buf ← GPUBackend.allocBuffer ctx (cacheSize * 4).toUSize
  let kF16Buf ← GPUBackend.allocBuffer ctx (cacheU32Size * 4).toUSize
  let vF16Buf ← GPUBackend.allocBuffer ctx (cacheU32Size * 4).toUSize
  let paramsBuf ← GPUBackend.allocBuffer ctx (4 : USize)
  let freqBuf ← GPUBackend.allocBuffer ctx (halfDim * 4).toUSize

  -- Inputs (deterministic).
  let newKData : Array Float := Array.range (kvDim * seqLen) |>.map fun i =>
    0.5 + (i.toFloat / 13.0).sin
  let newVData : Array Float := Array.range (kvDim * seqLen) |>.map fun i =>
    -0.3 + (i.toFloat / 17.0).cos
  let freqFactorsData : Array Float := Array.range halfDim |>.map fun i =>
    1.0 + i.toFloat * 0.05
  GPUBackend.writeBuffer ctx newKBuf (packFloats newKData)
  GPUBackend.writeBuffer ctx newVBuf (packFloats newVData)
  -- Zero caches before each kernel.
  GPUBackend.writeBuffer ctx kF32Buf (packFloats (Array.replicate cacheSize 0.0))
  GPUBackend.writeBuffer ctx vF32Buf (packFloats (Array.replicate cacheSize 0.0))
  GPUBackend.writeBuffer ctx kF16Buf (packU32s (Array.replicate cacheU32Size 0))
  GPUBackend.writeBuffer ctx vF16Buf (packU32s (Array.replicate cacheU32Size 0))
  GPUBackend.writeBuffer ctx paramsBuf (packU32s #[startPos.toUInt32])
  GPUBackend.writeBuffer ctx freqBuf (packFloats freqFactorsData)

  -- f32 reference
  let f32Shader := Hesper.Models.Gemma4.fusedRopeKAndCacheWriteBatchKernel
                     numKVHeads maxSeqLen headDim seqLen ropeBase
  let totalElements := numKVHeads * (headDim / 2) * seqLen
  let wgSize := 256
  let numWG := (totalElements + wgSize - 1) / wgSize
  GPUBackend.execute ctx f32Shader
    [("new_k", newKBuf), ("new_v", newVBuf),
     ("k_cache", kF32Buf), ("v_cache", vF32Buf),
     ("params", paramsBuf), ("freq_factors", freqBuf)]
    { workgroupSize := { x := wgSize }, numWorkgroups := (numWG, 1, 1) }

  -- f16 candidate
  let f16Shader := Hesper.Layers.Attention.fusedRopeKAndCacheWriteBatchKernelF16
                     numKVHeads maxSeqLen headDim seqLen ropeBase
  let totalPairs := numKVHeads * (headDim / 2) * seqLen
  let numWG16 := (totalPairs + wgSize - 1) / wgSize
  GPUBackend.execute ctx f16Shader
    [("new_k", newKBuf), ("new_v", newVBuf),
     ("k_cache_f16", kF16Buf), ("v_cache_f16", vF16Buf),
     ("params", paramsBuf), ("freq_factors", freqBuf)]
    { workgroupSize := { x := wgSize }, numWorkgroups := (numWG16, 1, 1) }

  let kF32Bytes ← GPUBackend.readBuffer ctx kF32Buf (cacheSize * 4).toUSize
  let vF32Bytes ← GPUBackend.readBuffer ctx vF32Buf (cacheSize * 4).toUSize
  let kF16Bytes ← GPUBackend.readBuffer ctx kF16Buf (cacheU32Size * 4).toUSize
  let vF16Bytes ← GPUBackend.readBuffer ctx vF16Buf (cacheU32Size * 4).toUSize

  -- Compare per-position written cells.
  let mut maxAbs : Float := 0.0
  for kv in [0:numKVHeads] do
    for col in [0:seqLen] do
      let pos := startPos + col
      let cacheRowF32 := (kv * maxSeqLen + pos) * headDim
      let cacheRowF16 := (kv * maxSeqLen + pos) * halfDim
      for dPair in [0:halfDim] do
        let kF32Lo := unpackF32 kF32Bytes (cacheRowF32 + 2 * dPair)
        let kF32Hi := unpackF32 kF32Bytes (cacheRowF32 + 2 * dPair + 1)
        let vF32Lo := unpackF32 vF32Bytes (cacheRowF32 + 2 * dPair)
        let vF32Hi := unpackF32 vF32Bytes (cacheRowF32 + 2 * dPair + 1)
        let kU := unpackU32 kF16Bytes (cacheRowF16 + dPair)
        let vU := unpackU32 vF16Bytes (cacheRowF16 + dPair)
        let (kF16Lo, kF16Hi) := unpackHalf2 kU
        let (vF16Lo, vF16Hi) := unpackHalf2 vU
        let dKLo := (kF32Lo - kF16Lo).abs
        let dKHi := (kF32Hi - kF16Hi).abs
        let dVLo := (vF32Lo - vF16Lo).abs
        let dVHi := (vF32Hi - vF16Hi).abs
        if dKLo > maxAbs then maxAbs := dKLo
        if dKHi > maxAbs then maxAbs := dKHi
        if dVLo > maxAbs then maxAbs := dVLo
        if dVHi > maxAbs then maxAbs := dVHi
  return maxAbs

def main : IO UInt32 := do
  let ctx ← CUDAContext.init
  let cases :=
    -- (numKVHeads, maxSeqLen, headDim, seqLen, startPos)
    [ (4, 256, 128, 4, 0)
    , (4, 256, 128, 11, 0)
    , (4, 256, 128, 32, 0)
    , (4, 256, 128, 16, 32) ]
  let ropeBase : Float := 10000.0
  let tol : Float := 0.01
  let mut allPass := true
  for (kvh, msl, hd, sl, sp) in cases do
    let maxAbs ← runRopeKVBatchF16Case ctx kvh msl hd sl sp ropeBase
    let pass := maxAbs < tol
    let mark := if pass then "✓" else "✗"
    IO.println s!"{mark} kvh={kvh} hd={hd} sl={sl} sp={sp}: maxAbsDiff={maxAbs}"
    if !pass then allPass := false
  if allPass then
    IO.println "✓ ALL PASS"
    return 0
  else
    IO.println "✗ FAIL"
    return 1
