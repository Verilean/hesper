import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.WGSL.FlashAttention
import Hesper.WGSL.FlashAttentionExperiments

/-!
# V11 launcher parity test

End-to-end test for `executeFlashAttentionV11`: pack a known f32 K/V
cache into the f16 packed format, run V11 partial+combine, and compare
against `executeFlashAttentionDynamic` (legacy f32 reference) on the
same Q + same K/V data.

Tolerance is f16 precision (~1e-2 relative for typical scores).

Run: `lake exe cuda-fa-v11-parity`
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

/-- Float32 → IEEE half (round-to-nearest-even, no NaN handling). -/
private def f32ToF16Bits (f : Float) : UInt16 :=
  let b32 : UInt64 := f.toBits  -- already f32-rounded by Lean's runtime
  let bits32 : UInt32 :=
    let s := (b32 >>> 63) &&& 1
    let e := (b32 >>> 52) &&& 0x7FF
    let m := b32 &&& 0x000FFFFFFFFFFFFF
    if e == 0 then 0
    else
      let e32 : Int := e.toNat - 1023 + 127
      if e32 <= 0 then 0
      else if e32 >= 255 then (s.toUInt32 <<< 31) ||| ((0xFF : UInt32) <<< 23)
      else (s.toUInt32 <<< 31) ||| (e32.toNat.toUInt32 <<< 23) |||
           ((m >>> 29).toUInt32 &&& (0x7FFFFF : UInt32))
  let s : UInt32 := (bits32 >>> 31) &&& 1
  let e : UInt32 := (bits32 >>> 23) &&& 0xFF
  let m : UInt32 := bits32 &&& 0x7FFFFF
  let result32 : UInt32 :=
    if e == 0 then s <<< 15
    else
      let e16 : Int := e.toNat - 127 + 15
      if e16 <= 0 then s <<< 15
      else if e16 >= 31 then (s <<< 15) ||| ((0x1F : UInt32) <<< 10)
      else (s <<< 15) ||| (e16.toNat.toUInt32 <<< 10) ||| (m >>> 13)
  result32.toUInt16

/-- Pack K/V cache from f32 array (laid out [kvHead, pos, dim]) into the
    u32 half2-packed layout V11 expects:
      cache[kvHead * maxSeqLen * (headDim/2) + pos * (headDim/2) + dPair]
        = pack2x16float(K[kvHead, pos, 2*dPair], K[kvHead, pos, 2*dPair+1])
-/
private def packKVCacheF32ToF16 (kvData : Array Float)
    (numKVHeads maxSeqLen headDim : Nat) : ByteArray := Id.run do
  let halfDim := headDim / 2
  let mut out : ByteArray := ByteArray.empty
  for kv in [0:numKVHeads] do
    for pos in [0:maxSeqLen] do
      for dPair in [0:halfDim] do
        let idxLo := (kv * maxSeqLen + pos) * headDim + 2 * dPair
        let idxHi := idxLo + 1
        let lo := kvData[idxLo]!
        let hi := kvData[idxHi]!
        let loBits : UInt32 := (f32ToF16Bits lo).toUInt32
        let hiBits : UInt32 := (f32ToF16Bits hi).toUInt32
        let packed : UInt32 := loBits ||| (hiBits <<< 16)
        out := out.push packed.toUInt8
                  |>.push (packed >>> 8).toUInt8
                  |>.push (packed >>> 16).toUInt8
                  |>.push (packed >>> 24).toUInt8
  return out

private def unpackFloats (ba : ByteArray) (n : Nat) : Array Float := Id.run do
  let mut arr : Array Float := #[]
  for i in [0:n] do
    let o := i * 4
    let b0 : UInt32 := ba.get! o |>.toUInt32
    let b1 : UInt32 := ba.get! (o+1) |>.toUInt32
    let b2 : UInt32 := ba.get! (o+2) |>.toUInt32
    let b3 : UInt32 := ba.get! (o+3) |>.toUInt32
    let bits := b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24)
    let e := (bits >>> 23) &&& 0xFF
    let m := bits &&& (0x7FFFFF : UInt32)
    let s := bits >>> 31
    let f := if e == 0 then 0.0 else
      let v := (1.0 + m.toNat.toFloat / 8388608.0) * Float.pow 2.0 (e.toNat.toFloat - 127.0)
      if s == 1 then -v else v
    arr := arr.push f
  return arr

/-- Run one parity case at given (numHeads, numKVHeads, headDim, maxSeqLen,
    cacheLen).  Returns (legacyOut, v11Out, maxAbsDiff).
    Pre-condition: cacheLen >= 8 (V11 needs cacheLen ≥ numSplits=8).
                   headDim % 64 == 0 (V11 sub-warp partition assumes this).
-/
def runV11ParityCase (ctx : CUDAContext)
    (numHeads numKVHeads headDim maxSeqLen cacheLen : Nat) (scale : Float) :
    IO (Array Float × Array Float × Float) := do
  let qSize := (numHeads * headDim * 4).toUSize
  let kvF32Size := (numKVHeads * maxSeqLen * headDim * 4).toUSize
  let kvF16Size := (numKVHeads * maxSeqLen * headDim * 2).toUSize
  let outSize := qSize
  let qBuf ← GPUBackend.allocBuffer ctx qSize
  let kF32Buf ← GPUBackend.allocBuffer ctx kvF32Size
  let vF32Buf ← GPUBackend.allocBuffer ctx kvF32Size
  let kF16Buf ← GPUBackend.allocBuffer ctx kvF16Size
  let vF16Buf ← GPUBackend.allocBuffer ctx kvF16Size
  let outLegacyBuf ← GPUBackend.allocBuffer ctx outSize
  let outV11Buf ← GPUBackend.allocBuffer ctx outSize
  let paramsBuf ← GPUBackend.allocBuffer ctx (8 : USize)

  let numSplits : Nat := 8
  let partialOutSize := (numHeads * numSplits * headDim * 4).toUSize
  let partialMetaSize := (numHeads * numSplits * 2 * 4).toUSize
  let partialOutBuf ← GPUBackend.allocBuffer ctx partialOutSize
  let partialMetaBuf ← GPUBackend.allocBuffer ctx partialMetaSize

  -- Deterministic test inputs
  let qData : Array Float := Array.range (numHeads * headDim) |>.map fun i =>
    0.1 + (i.toFloat / 64.0).sin
  let mut kvData : Array Float := Array.replicate (numKVHeads * maxSeqLen * headDim) 0.0
  let mut vData : Array Float := Array.replicate (numKVHeads * maxSeqLen * headDim) 0.0
  for kv in [0:numKVHeads] do
    for pos in [0:cacheLen] do
      for d in [0:headDim] do
        let off := (kv * maxSeqLen + pos) * headDim + d
        kvData := kvData.set! off
          (((kv + 1).toFloat * 0.05) + ((pos + 1).toFloat * 0.013)
           + (d.toFloat / 53.0).cos)
        vData := vData.set! off
          (((kv + 1).toFloat * 0.07) + ((pos + 1).toFloat * 0.011)
           + (d.toFloat / 41.0).sin)

  GPUBackend.writeBuffer ctx qBuf (packFloats qData)
  GPUBackend.writeBuffer ctx kF32Buf (packFloats kvData)
  GPUBackend.writeBuffer ctx vF32Buf (packFloats vData)
  GPUBackend.writeBuffer ctx kF16Buf
    (packKVCacheF32ToF16 kvData numKVHeads maxSeqLen headDim)
  GPUBackend.writeBuffer ctx vF16Buf
    (packKVCacheF32ToF16 vData numKVHeads maxSeqLen headDim)
  GPUBackend.writeBuffer ctx paramsBuf
    (packU32s #[(cacheLen - 1).toUInt32, cacheLen.toUInt32])

  -- Reference: legacy 256-thread tree-reduce kernel reading f32 cache.
  Hesper.WGSL.FlashAttention.executeFlashAttentionDynamic ctx
    qBuf kF32Buf vF32Buf outLegacyBuf
    numHeads numKVHeads maxSeqLen headDim cacheLen scale

  -- V11 production launcher reading f16 packed cache.
  Hesper.WGSL.FlashAttention.executeFlashAttentionV11 ctx
    qBuf kF16Buf vF16Buf paramsBuf
    partialOutBuf partialMetaBuf outV11Buf
    numHeads numKVHeads maxSeqLen headDim scale

  let legacyBytes ← GPUBackend.readBuffer ctx outLegacyBuf outSize
  let v11Bytes    ← GPUBackend.readBuffer ctx outV11Buf outSize
  let legacyOut := unpackFloats legacyBytes (numHeads * headDim)
  let v11Out    := unpackFloats v11Bytes (numHeads * headDim)

  let mut maxAbs : Float := 0.0
  for i in [0:legacyOut.size] do
    let d := (legacyOut[i]! - v11Out[i]!).abs
    if d > maxAbs then maxAbs := d
  return (legacyOut, v11Out, maxAbs)

def main : IO UInt32 := do
  let ctx ← CUDAContext.init
  let cases :=
    -- (numHeads, numKVHeads, headDim, maxSeqLen, cacheLen)
    -- Gemma 4 E4B realistic: 8 heads, 4 KV, headDim=128, cacheLen ≥ 8
    [ (8, 4, 128, 256, 16)
    , (8, 4, 128, 256, 32)
    , (8, 4, 128, 256, 64)
    , (8, 4, 128, 256, 128)
    , (8, 4, 128, 256, 200) ]
  let scale : Float := 1.0
  let mut allPass := true
  let tol : Float := 0.05  -- f16 precision; legacy is f32, V11 is f16, so loose
  for (h, kvh, hd, msl, cl) in cases do
    let (legacy, v11, maxAbs) ← runV11ParityCase ctx h kvh hd msl cl scale
    let pass := maxAbs < tol
    let mark := if pass then "✓" else "✗"
    IO.println s!"{mark} h={h} kvh={kvh} hd={hd} cl={cl}: maxAbsDiff={maxAbs}"
    if !pass then
      allPass := false
      -- Show first 8 elements of each for diagnosis
      IO.println s!"  legacy[0..8]: {legacy.toSubarray 0 8 |>.toArray}"
      IO.println s!"  v11   [0..8]: {v11.toSubarray 0 8 |>.toArray}"
  if allPass then
    IO.println "✓ ALL PASS"
    return 0
  else
    IO.println "✗ SOME FAILED"
    return 1
