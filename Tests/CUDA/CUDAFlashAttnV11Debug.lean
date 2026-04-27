import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.CUDA.FFI
import Hesper.WGSL.Monad
import Hesper.WGSL.Exp
import Hesper.WGSL.Shader
import Hesper.WGSL.FlashAttention
import Hesper.WGSL.FlashAttentionExperiments
import Hesper

/-!
# V11 vs V9 partial-output diff

Runs V9 and V11 on the same Q/K/V buffers, dumps `partial_out` and
`partial_meta` from each, and prints per-(split, dim) divergences.

This bypasses the combine kernel — the bug is in V11's partial output
itself (since V9 + same combine = pass, V11 + same combine = fail).

Run:
  lake exe cuda-flashattn-v11-debug
-/

open Hesper
open Hesper.WGSL.Monad (ShaderM)

private def f64ToF32Bits (f : Float) : UInt32 :=
  let b := f.toBits
  let s := (b >>> 63) &&& 1
  let e := (b >>> 52) &&& 0x7FF
  let m := b &&& 0x000FFFFFFFFFFFFF
  if e == 0 then 0
  else
    let e32 : Int := e.toNat - 1023 + 127
    if e32 <= 0 then 0
    else if e32 >= 255 then
      (s.toUInt32 <<< 31) ||| ((0xFF : UInt32) <<< 23)
    else
      (s.toUInt32 <<< 31) |||
      (e32.toNat.toUInt32 <<< 23) |||
      ((m >>> 29).toUInt32 &&& (0x7FFFFF : UInt32))

private def packFloats (arr : Array Float) : ByteArray :=
  arr.foldl (fun acc f =>
    let bits := f64ToF32Bits f
    acc.push bits.toUInt8
       |>.push (bits >>> 8).toUInt8
       |>.push (bits >>> 16).toUInt8
       |>.push (bits >>> 24).toUInt8) ByteArray.empty

private def f32ToF16Bits (b : UInt32) : UInt16 :=
  let s : UInt32 := (b >>> 31) &&& 1
  let e32 : UInt32 := (b >>> 23) &&& 0xFF
  let m32 : UInt32 := b &&& 0x7FFFFF
  if e32 == 0 then 0
  else if e32 == 0xFF then
    (s.toUInt16 <<< 15) ||| (0x7C00 : UInt16)
  else
    let e32i : Int := e32.toNat - 127 + 15
    if e32i <= 0 then 0
    else if e32i >= 31 then
      (s.toUInt16 <<< 15) ||| (0x7C00 : UInt16)
    else
      let m16 : UInt32 := m32 >>> 13
      (s.toUInt16 <<< 15) |||
      (e32i.toNat.toUInt16 <<< 10) |||
      m16.toUInt16

private def packHalfs (arr : Array Float) : ByteArray :=
  arr.foldl (fun acc f =>
    let h := f32ToF16Bits (f64ToF32Bits f)
    acc.push h.toUInt8
       |>.push (h >>> 8).toUInt8) ByteArray.empty

private def unpackFloats (ba : ByteArray) (n : Nat) : Array Float := Id.run do
  let mut arr := #[]
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
      let fv := (1.0 + m.toNat.toFloat / 8388608.0)
                * Float.pow 2.0 (e.toNat.toFloat - 127.0)
      if s == 1 then -fv else fv
    arr := arr.push f
  return arr

private def packU32x2 (a b : Nat) : ByteArray :=
  let aU : UInt32 := a.toUInt32
  let bU : UInt32 := b.toUInt32
  ByteArray.empty
    |>.push aU.toUInt8 |>.push (aU >>> 8).toUInt8
    |>.push (aU >>> 16).toUInt8 |>.push (aU >>> 24).toUInt8
    |>.push bU.toUInt8 |>.push (bU >>> 8).toUInt8
    |>.push (bU >>> 16).toUInt8 |>.push (bU >>> 24).toUInt8

unsafe def main : IO Unit := do
  let ctx ← Hesper.CUDAContext.init
  let numHeads := 1
  let numKVHeads := 1
  let headDim := 256
  let maxSeqLen := 64
  let cacheLen := 8  -- a passing case (V11=0.006 ✓)
  let numSplits : Nat := 8
  let scale : Float := 1.0 / headDim.toFloat.sqrt

  let qSize := (numHeads * headDim * 4).toUSize
  let kvSize := (numKVHeads * maxSeqLen * headDim * 4).toUSize
  let kvSizeHalf := (numKVHeads * maxSeqLen * headDim * 2).toUSize
  let partialOutSize := (numHeads * numSplits * headDim * 4).toUSize
  let partialMetaSize := (numHeads * numSplits * 2 * 4).toUSize

  let qBuf ← GPUBackend.allocBuffer ctx qSize
  let kBufF16 ← GPUBackend.allocBuffer ctx kvSizeHalf
  let vBufF16 ← GPUBackend.allocBuffer ctx kvSizeHalf
  let paramsBuf ← GPUBackend.allocBuffer ctx (8 : USize)
  let v9PartialOut ← GPUBackend.allocBuffer ctx partialOutSize
  let v9PartialMeta ← GPUBackend.allocBuffer ctx partialMetaSize
  let v11PartialOut ← GPUBackend.allocBuffer ctx partialOutSize
  let v11PartialMeta ← GPUBackend.allocBuffer ctx partialMetaSize
  -- Pre-fill meta buffers with sentinel so we can detect un-written slots.
  let zeros := Array.replicate (numHeads * numSplits * 2) (0.0 : Float)
  GPUBackend.writeBuffer ctx v9PartialMeta (packFloats zeros)
  GPUBackend.writeBuffer ctx v11PartialMeta (packFloats zeros)

  let qData := Array.range (numHeads * headDim)
                |>.map (fun i => 0.1 + (i.toFloat / 64.0).sin)
  let mut kData : Array Float :=
    Array.replicate (numKVHeads * maxSeqLen * headDim) 0.0
  let mut vData : Array Float :=
    Array.replicate (numKVHeads * maxSeqLen * headDim) 0.0
  for kv in [0:numKVHeads] do
    for pos in [0:cacheLen] do
      for d in [0:headDim] do
        let off := (kv * maxSeqLen + pos) * headDim + d
        kData := kData.set! off (((kv + 1).toFloat * 0.05) +
                                  ((pos + 1).toFloat * 0.013) +
                                  (d.toFloat / 53.0).cos)
        vData := vData.set! off (((kv + 1).toFloat * 0.07) +
                                  ((pos + 1).toFloat * 0.011) +
                                  (d.toFloat / 41.0).sin)

  GPUBackend.writeBuffer ctx qBuf (packFloats qData)
  GPUBackend.writeBuffer ctx kBufF16 (packHalfs kData)
  GPUBackend.writeBuffer ctx vBufF16 (packHalfs vData)
  GPUBackend.writeBuffer ctx paramsBuf (packU32x2 (cacheLen - 1) cacheLen)

  IO.println s!"=== Setup: nH={numHeads} nKV={numKVHeads} D={headDim} maxSeq={maxSeqLen} cacheLen={cacheLen} numSplits={numSplits} ==="

  -- Run V9
  let v9Shader := Hesper.WGSL.FlashAttention.flashAttentionVecParamsKernelV9
                     numHeads numKVHeads maxSeqLen headDim numSplits scale
  let v9Bufs : List (String × GPUBackend.Buf Hesper.CUDAContext) :=
    [ ("q",             qBuf)
    , ("k_cache_f16",   kBufF16)
    , ("v_cache_f16",   vBufF16)
    , ("partial_out",   v9PartialOut)
    , ("partial_meta",  v9PartialMeta)
    , ("params",        paramsBuf) ]
  GPUBackend.execute ctx v9Shader v9Bufs
    { workgroupSize := { x := 128 }, numWorkgroups := (numHeads, numSplits, 1),
      extensions := ["subgroups", "f16"] }

  -- Run V11
  let v11Shader := Hesper.WGSL.FlashAttention.flashAttentionVecParamsKernelV11
                       numHeads numKVHeads maxSeqLen headDim numSplits scale
  let v11Bufs : List (String × GPUBackend.Buf Hesper.CUDAContext) :=
    [ ("q",             qBuf)
    , ("k_cache_f16",   kBufF16)
    , ("v_cache_f16",   vBufF16)
    , ("partial_out",   v11PartialOut)
    , ("partial_meta",  v11PartialMeta)
    , ("params",        paramsBuf) ]
  GPUBackend.execute ctx v11Shader v11Bufs
    { workgroupSize := { x := 128 }, numWorkgroups := (numHeads, numSplits, 1),
      extensions := ["subgroups", "f16"] }

  let v9OutBytes ← GPUBackend.readBuffer ctx v9PartialOut partialOutSize
  let v11OutBytes ← GPUBackend.readBuffer ctx v11PartialOut partialOutSize
  let v9MetaBytes ← GPUBackend.readBuffer ctx v9PartialMeta partialMetaSize
  let v11MetaBytes ← GPUBackend.readBuffer ctx v11PartialMeta partialMetaSize
  let v9Out := unpackFloats v9OutBytes (numHeads * numSplits * headDim)
  let v11Out := unpackFloats v11OutBytes (numHeads * numSplits * headDim)
  let v9Meta := unpackFloats v9MetaBytes (numHeads * numSplits * 2)
  let v11Meta := unpackFloats v11MetaBytes (numHeads * numSplits * 2)

  IO.println ""
  IO.println "=== partial_meta per split (max, sum) ==="
  IO.println "split  V9_max     V9_sum     V11_max    V11_sum    Δmax       Δsum"
  for s in [0:numSplits] do
    let i := s * 2
    let v9m := v9Meta[i]!
    let v9s := v9Meta[i+1]!
    let v11m := v11Meta[i]!
    let v11s := v11Meta[i+1]!
    IO.println s!"  {s}    {v9m}  {v9s}  {v11m}  {v11s}  {v11m-v9m}  {v11s-v9s}"

  IO.println ""
  IO.println "=== partial_out per split, max abs diff over 256 dims ==="
  IO.println "split  max|Δ|     first_div_dim  V9          V11         (V11-V9)"
  for s in [0:numSplits] do
    let mut maxAbs : Float := 0.0
    let mut firstDivDim : Int := -1
    for d in [0:headDim] do
      let off := s * headDim + d
      let v9v := v9Out[off]!
      let v11v := v11Out[off]!
      let δ := (v11v - v9v).abs
      if δ > maxAbs then maxAbs := δ
      if firstDivDim < 0 ∧ δ > 1e-3 then firstDivDim := Int.ofNat d
    if maxAbs > 1e-4 then
      let dim := if firstDivDim < 0 then 0 else firstDivDim.toNat
      let off := s * headDim + dim
      IO.println s!"  {s}    {maxAbs}     {firstDivDim}             {v9Out[off]!}  {v11Out[off]!}  {v11Out[off]! - v9Out[off]!}"
    else
      IO.println s!"  {s}    {maxAbs}     (none)         -           -           -"

  IO.println ""
  IO.println "=== V11 first 8 dims of split 0 ==="
  for d in [0:8] do
    IO.println s!"  dim {d}: V9={v9Out[d]!} V11={v11Out[d]!}"
