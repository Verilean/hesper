import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.WGSL.FlashAttention

/-!
# flashAttentionBatchKernelF16 parity unit test

Compares `flashAttentionBatchKernel` (f32 cache, prefill workhorse)
against `flashAttentionBatchKernelF16` (new f16-cache version).

Both run the same Q + K/V values; the f16 version reads K/V from a
pre-packed half2 cache.  Output is compared within f16 precision
(~1e-2 relative).

Run: `lake exe cuda-fa-batch-f16-parity`
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

private def f32ToF16Bits (f : Float) : UInt16 :=
  let b32 : UInt64 := f.toBits
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

def runBatchAttnF16ParityCase (ctx : CUDAContext)
    (numHeads numKVHeads headDim maxSeqLen seqLen startPos : Nat) (scale : Float) :
    IO Float := do
  let qSize := (numHeads * headDim * seqLen * 4).toUSize
  let kvF32Size := (numKVHeads * maxSeqLen * headDim * 4).toUSize
  let kvF16Size := (numKVHeads * maxSeqLen * headDim * 2).toUSize
  let outSize := qSize
  let qBuf ← GPUBackend.allocBuffer ctx qSize
  let kF32Buf ← GPUBackend.allocBuffer ctx kvF32Size
  let vF32Buf ← GPUBackend.allocBuffer ctx kvF32Size
  let kF16Buf ← GPUBackend.allocBuffer ctx kvF16Size
  let vF16Buf ← GPUBackend.allocBuffer ctx kvF16Size
  let outF32Buf ← GPUBackend.allocBuffer ctx outSize
  let outF16Buf ← GPUBackend.allocBuffer ctx outSize
  let paramsBuf ← GPUBackend.allocBuffer ctx (4 : USize)

  -- Cache must be populated for positions 0 .. startPos+seqLen-1
  let lastPos := startPos + seqLen
  let qData : Array Float := Array.range (numHeads * headDim * seqLen) |>.map fun i =>
    0.1 + (i.toFloat / 64.0).sin
  let mut kvData : Array Float := Array.replicate (numKVHeads * maxSeqLen * headDim) 0.0
  let mut vData : Array Float := Array.replicate (numKVHeads * maxSeqLen * headDim) 0.0
  for kv in [0:numKVHeads] do
    for pos in [0:lastPos] do
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
  GPUBackend.writeBuffer ctx paramsBuf (packU32s #[startPos.toUInt32])

  let workgroupSize := min (max headDim 32) 256

  -- f32 reference
  let f32Shader := Hesper.WGSL.FlashAttention.flashAttentionBatchKernel
                     numHeads numKVHeads maxSeqLen headDim seqLen scale workgroupSize
  GPUBackend.execute ctx f32Shader
    [("q", qBuf), ("k_cache", kF32Buf), ("v_cache", vF32Buf),
     ("output", outF32Buf), ("params", paramsBuf)]
    { workgroupSize := { x := workgroupSize }, numWorkgroups := (numHeads, seqLen, 1),
      extensions := ["subgroups"] }

  -- f16 candidate
  let f16Shader := Hesper.WGSL.FlashAttention.flashAttentionBatchKernelF16
                     numHeads numKVHeads maxSeqLen headDim seqLen scale workgroupSize
  GPUBackend.execute ctx f16Shader
    [("q", qBuf), ("k_cache_f16", kF16Buf), ("v_cache_f16", vF16Buf),
     ("output", outF16Buf), ("params", paramsBuf)]
    { workgroupSize := { x := workgroupSize }, numWorkgroups := (numHeads, seqLen, 1),
      extensions := ["subgroups"] }

  let f32Bytes ← GPUBackend.readBuffer ctx outF32Buf outSize
  let f16Bytes ← GPUBackend.readBuffer ctx outF16Buf outSize
  let f32Out := unpackFloats f32Bytes (numHeads * headDim * seqLen)
  let f16Out := unpackFloats f16Bytes (numHeads * headDim * seqLen)
  let mut maxAbs : Float := 0.0
  for i in [0:f32Out.size] do
    let d := (f32Out[i]! - f16Out[i]!).abs
    if d > maxAbs then maxAbs := d
  return maxAbs

def main : IO UInt32 := do
  let ctx ← CUDAContext.init
  -- (numHeads, numKVHeads, headDim, maxSeqLen, seqLen, startPos)
  let cases :=
    [ (8, 4, 128, 256, 4, 0)    -- short prefill from scratch
    , (8, 4, 128, 256, 11, 0)   -- typical "Hello!" prompt
    , (8, 4, 128, 256, 32, 0)   -- longer prefill
    , (8, 4, 128, 256, 16, 32)  -- continuation prefill (startPos > 0)
    ]
  let scale : Float := 1.0
  let tol : Float := 0.05
  let mut allPass := true
  for (h, kvh, hd, msl, sl, sp) in cases do
    let maxAbs ← runBatchAttnF16ParityCase ctx h kvh hd msl sl sp scale
    let pass := maxAbs < tol
    let mark := if pass then "✓" else "✗"
    IO.println s!"{mark} h={h} kvh={kvh} hd={hd} sl={sl} sp={sp}: maxAbsDiff={maxAbs}"
    if !pass then allPass := false
  if allPass then
    IO.println "✓ ALL PASS"
    return 0
  else
    IO.println "✗ FAIL"
    return 1
