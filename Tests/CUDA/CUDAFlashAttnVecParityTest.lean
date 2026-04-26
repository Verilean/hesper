import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.WGSL.Monad
import Hesper.WGSL.Exp
import Hesper.WGSL.Shader
import Hesper.WGSL.FlashAttention
import Hesper

/-!
# FlashAttention vec-params parity test

Compares `flashAttentionVecParamsKernel` (doc 60 Session 1, warp-shuffle
reduce) against `flashAttentionDynamicParamsKernel` (legacy, 256-thread
tree reduce) on identical Q/K/V/cacheLen inputs.  Both kernels share
the same input/output contract — Q from `q`, output to `output`,
cacheLen in `params[1]`.

Pass criterion: `max(|out_vec[i] - out_legacy[i]|) < 1e-4` for all
output positions, across multiple cacheLen sizes.

Run:
  lake exe cuda-flashattn-vec-parity
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

/-- Pack two Nat values as little-endian u32 followed by another u32. -/
private def packU32x2 (a b : Nat) : ByteArray :=
  let aU : UInt32 := a.toUInt32
  let bU : UInt32 := b.toUInt32
  ByteArray.empty
    |>.push aU.toUInt8 |>.push (aU >>> 8).toUInt8
    |>.push (aU >>> 16).toUInt8 |>.push (aU >>> 24).toUInt8
    |>.push bU.toUInt8 |>.push (bU >>> 8).toUInt8
    |>.push (bU >>> 16).toUInt8 |>.push (bU >>> 24).toUInt8

/-- Run one (numHeads, numKVHeads, headDim, maxSeqLen, cacheLen) case
    on both kernels and return (legacyOut, vecOut). -/
def runCase
    (ctx : Hesper.CUDAContext)
    (numHeads numKVHeads headDim maxSeqLen cacheLen : Nat)
    (scale : Float) : IO (Array Float × Array Float) := do
  let qSize := (numHeads * headDim * 4).toUSize
  let kvSize := (numKVHeads * maxSeqLen * headDim * 4).toUSize
  let outSize := qSize
  let qBuf ← GPUBackend.allocBuffer ctx qSize
  let kBuf ← GPUBackend.allocBuffer ctx kvSize
  let vBuf ← GPUBackend.allocBuffer ctx kvSize
  let outBufLegacy ← GPUBackend.allocBuffer ctx outSize
  let outBufVec ← GPUBackend.allocBuffer ctx outSize
  let paramsBuf ← GPUBackend.allocBuffer ctx (8 : USize)

  -- Deterministic test inputs
  let qData := Array.range (numHeads * headDim)
                |>.map (fun i => 0.1 + (i.toFloat / 64.0).sin)
  let mut kData : Array Float := Array.replicate (numKVHeads * maxSeqLen * headDim) 0.0
  let mut vData : Array Float := Array.replicate (numKVHeads * maxSeqLen * headDim) 0.0
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
  GPUBackend.writeBuffer ctx kBuf (packFloats kData)
  GPUBackend.writeBuffer ctx vBuf (packFloats vData)
  -- params = [pos=cacheLen-1, cacheLen]; the kernel only reads slot 1.
  GPUBackend.writeBuffer ctx paramsBuf (packU32x2 (cacheLen - 1) cacheLen)

  -- Legacy: dynamic-params kernel (256-thread tree reduce)
  let legacyShader := Hesper.WGSL.FlashAttention.flashAttentionDynamicParamsKernel
                       numHeads numKVHeads maxSeqLen headDim scale 256
  let legacyBufs : List (String × GPUBackend.Buf Hesper.CUDAContext) :=
    [ ("q",       qBuf)
    , ("k_cache", kBuf)
    , ("v_cache", vBuf)
    , ("output",  outBufLegacy)
    , ("params",  paramsBuf) ]
  GPUBackend.execute ctx legacyShader legacyBufs
    { workgroupSize := { x := 256 }, numWorkgroups := (numHeads, 1, 1),
      extensions := ["subgroups"] }

  -- Vec: doc 60 Session 1 kernel (128-thread, warp-shuffle reduce)
  let vecShader := Hesper.WGSL.FlashAttention.flashAttentionVecParamsKernel
                     numHeads numKVHeads maxSeqLen headDim scale
  let vecBufs : List (String × GPUBackend.Buf Hesper.CUDAContext) :=
    [ ("q",       qBuf)
    , ("k_cache", kBuf)
    , ("v_cache", vBuf)
    , ("output",  outBufVec)
    , ("params",  paramsBuf) ]
  GPUBackend.execute ctx vecShader vecBufs
    { workgroupSize := { x := 128 }, numWorkgroups := (numHeads, 1, 1),
      extensions := ["subgroups"] }

  let legacyResultBytes ← GPUBackend.readBuffer ctx outBufLegacy outSize
  let vecResultBytes    ← GPUBackend.readBuffer ctx outBufVec outSize
  let legacyOut := unpackFloats legacyResultBytes (numHeads * headDim)
  let vecOut    := unpackFloats vecResultBytes (numHeads * headDim)

  GPUBackend.freeBuffer ctx qBuf
  GPUBackend.freeBuffer ctx kBuf
  GPUBackend.freeBuffer ctx vBuf
  GPUBackend.freeBuffer ctx outBufLegacy
  GPUBackend.freeBuffer ctx outBufVec
  GPUBackend.freeBuffer ctx paramsBuf

  return (legacyOut, vecOut)

def compareCase (label : String) (legacyOut vecOut : Array Float)
    (tolerance : Float := 1e-4) : IO Bool := do
  let mut maxAbs := 0.0
  let mut maxRel := 0.0
  let mut firstMismatchIdx : Option Nat := none
  for i in [0:legacyOut.size] do
    let a := legacyOut[i]!
    let b := vecOut[i]!
    let d := (a - b).abs
    if d > maxAbs then maxAbs := d
    let denom := max a.abs b.abs
    if denom > 1e-6 then
      let r := d / denom
      if r > maxRel then maxRel := r
    if d > tolerance && firstMismatchIdx.isNone then
      firstMismatchIdx := some i
  let pass := maxAbs < tolerance
  let mark := if pass then "✓" else "✗"
  IO.println s!"  {mark} {label}: max abs diff = {maxAbs}, max rel diff = {maxRel}"
  match firstMismatchIdx with
  | some i => IO.println s!"      first mismatch at [{i}]: legacy={legacyOut[i]!} vec={vecOut[i]!}"
  | none => pure ()
  return pass

unsafe def main : IO Unit := do
  IO.println "═══ FlashAttention Vec-Params Parity Test ═══"
  let ctx ← Hesper.CUDAContext.init

  -- Cases match Gemma 4 (head_dim=256, num_heads=8, num_kv_heads=1)
  -- but kept small enough that PTX cache + run time stays under a
  -- few seconds.  Bisect by varying cacheLen.
  let cases : List (Nat × Nat × Nat × Nat × Nat) :=
    [ -- (numHeads, numKVHeads, headDim, maxSeqLen, cacheLen)
      (1, 1, 64,  16, 4)
    , (1, 1, 64,  16, 8)
    , (1, 1, 128, 32, 16)
    , (1, 1, 256, 32, 8)
    , (2, 1, 256, 32, 16)
    , (2, 1, 256, 64, 32)
    , (4, 1, 256, 64, 32)
    -- Gemma 4 production geometry
    , (8, 1, 256, 64, 32)
    , (8, 1, 256, 128, 64)
    , (8, 1, 256, 256, 100)
    -- Gemma 4 decode-style cacheLen increments (1..40)
    , (8, 1, 256, 256, 5)
    , (8, 1, 256, 256, 6)
    , (8, 1, 256, 256, 7)
    , (8, 1, 256, 256, 10)
    , (8, 1, 256, 256, 33)
    , (8, 1, 256, 256, 35)
    , (8, 1, 256, 256, 40) ]

  let mut allPassed := true
  for (nh, nkv, hd, msl, cl) in cases do
    let scale : Float := 1.0 / (hd.toFloat.sqrt)
    let label := s!"nH={nh} nKV={nkv} D={hd} maxSeq={msl} cacheLen={cl}"
    try
      let (legacy, vec) ← runCase ctx nh nkv hd msl cl scale
      let ok ← compareCase label legacy vec
      if !ok then allPassed := false
    catch e =>
      IO.println s!"  ✗ {label}: exception {e}"
      allPassed := false

  IO.println ""
  if allPassed then
    IO.println "═══ ALL CASES PASS — vec kernel is bit-parity with legacy ═══"
  else
    IO.println "═══ SOME CASES FAILED — see above ═══"
    IO.Process.exit 1
