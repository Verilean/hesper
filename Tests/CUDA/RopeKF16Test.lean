import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.Layers.Attention

/-!
# RoPE-K + KV-cache f16 write kernel — unit test

Compares `fusedRopeKAndCacheWriteKernel` (f32 cache writer) with
`fusedRopeKAndCacheWriteKernelF16` (new f16 packed writer) at a single
position.  Both are given the same `new_k` input + `params` + freq_factors;
we read back the f32 cache row and the f16 cache row (unpacked), and
verify they match within f16 precision (~1e-3 relative).

Also verifies `packVCacheF32ToF16Kernel`: write a known f32 V row, run
the pack kernel, read back f16, unpack, compare.

Run: `lake exe cuda-rope-k-f16-test`
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

/-- Decode a packed `pack2x16float` u32 to a (lo, hi) f32 pair. -/
private def unpackHalf2 (u : UInt32) : Float × Float :=
  let halfToFloat (h : UInt32) : Float :=
    let s := (h >>> 15) &&& 1
    let e := (h >>> 10) &&& 0x1F
    let m := h &&& 0x3FF
    let absV : Float :=
      if e == 0 then
        if m == 0 then 0.0
        else m.toNat.toFloat * Float.pow 2.0 (-24.0)  -- subnormal
      else if e == 31 then 0.0  -- inf/nan unsupported
      else
        (1.0 + m.toNat.toFloat / 1024.0) * Float.pow 2.0 (e.toNat.toFloat - 15.0)
    if s == 1 then -absV else absV
  let lo := halfToFloat (u &&& 0xFFFF)
  let hi := halfToFloat (u >>> 16)
  (lo, hi)

/-- Run RoPE-K f32 kernel + RoPE-K f16 kernel on the same input, compare
    f32 cache row against unpacked f16 cache row. -/
def runRopeKF16Parity [GPUBackend β] (ctx : β) : IO Bool := do
  let numKVHeads := 1
  let maxSeqLen := 4
  let headDim := 8
  let halfDim := headDim / 2
  let kvDim := numKVHeads * headDim
  let cacheSize := numKVHeads * maxSeqLen * headDim
  let pos : UInt32 := 1  -- write to row pos=1
  let cacheLen : UInt32 := 2
  let ropeBase : Float := 10000.0

  -- Inputs (synthetic K row)
  let newKData : Array Float := Array.range kvDim |>.map fun i =>
    0.5 + (i.toFloat / 10.0) - 0.3
  let freqFactorsData : Array Float := Array.range halfDim |>.map fun i =>
    1.0 + i.toFloat * 0.05
  -- Initial caches zero
  let zeroF32 : Array Float := Array.replicate cacheSize 0.0
  let zeroU32 : Array UInt32 := Array.replicate (cacheSize / 2) 0

  -- f32 kernel
  let newKBuf ← GPUBackend.allocBuffer ctx (kvDim * 4).toUSize
  let kCacheF32 ← GPUBackend.allocBuffer ctx (cacheSize * 4).toUSize
  let vCacheF32 ← GPUBackend.allocBuffer ctx (cacheSize * 4).toUSize
  let newVBuf ← GPUBackend.allocBuffer ctx (kvDim * 4).toUSize  -- unused but declared
  let paramsBuf ← GPUBackend.allocBuffer ctx (8 : USize)
  let freqBuf ← GPUBackend.allocBuffer ctx (halfDim * 4).toUSize

  GPUBackend.writeBuffer ctx newKBuf (packFloats newKData)
  GPUBackend.writeBuffer ctx newVBuf (packFloats (Array.replicate kvDim 0.0))
  GPUBackend.writeBuffer ctx kCacheF32 (packFloats zeroF32)
  GPUBackend.writeBuffer ctx vCacheF32 (packFloats zeroF32)
  GPUBackend.writeBuffer ctx paramsBuf (packU32s #[pos, cacheLen])
  GPUBackend.writeBuffer ctx freqBuf (packFloats freqFactorsData)

  -- Run f32 kernel
  let f32Shader := Hesper.Layers.Attention.fusedRopeKAndCacheWriteKernel
                     numKVHeads maxSeqLen headDim kvDim ropeBase
  GPUBackend.execute ctx f32Shader
    [("new_k", newKBuf), ("new_v", newVBuf),
     ("k_cache", kCacheF32), ("v_cache", vCacheF32),
     ("params", paramsBuf), ("freq_factors", freqBuf)]
    ({ workgroupSize := { x := 32, y := 1, z := 1 },
       numWorkgroups := ((kvDim + 31) / 32, 1, 1) : Hesper.ExecConfig })

  -- Run f16 kernel
  let kCacheF16 ← GPUBackend.allocBuffer ctx (cacheSize * 2).toUSize
  GPUBackend.writeBuffer ctx kCacheF16 (packU32s zeroU32)

  let f16Shader := Hesper.Layers.Attention.fusedRopeKAndCacheWriteKernelF16
                     numKVHeads maxSeqLen headDim kvDim ropeBase
  let halfKvDim := kvDim / 2
  GPUBackend.execute ctx f16Shader
    [("new_k", newKBuf), ("k_cache_f16", kCacheF16),
     ("params", paramsBuf), ("freq_factors", freqBuf)]
    ({ workgroupSize := { x := 32, y := 1, z := 1 },
       numWorkgroups := ((halfKvDim + 31) / 32, 1, 1) : Hesper.ExecConfig })

  -- Read back
  let f32Bytes ← GPUBackend.readBuffer ctx kCacheF32 (cacheSize * 4).toUSize
  let f16Bytes ← GPUBackend.readBuffer ctx kCacheF16 (cacheSize * 2).toUSize

  -- f32 row at pos=1
  let kvHead := 0
  let rowOffset := kvHead * maxSeqLen * headDim + pos.toNat * headDim
  let f32Row : Array Float := Array.range headDim |>.map fun d =>
    unpackF32 f32Bytes (rowOffset + d)
  -- f16 row at pos=1
  let halfRowOffset := kvHead * maxSeqLen * halfDim + pos.toNat * halfDim
  let f16Row : Array Float := (Array.range halfDim).foldl (init := #[]) fun acc dPair =>
    let u := unpackU32 f16Bytes (halfRowOffset + dPair)
    let (lo, hi) := unpackHalf2 u
    acc.push lo |>.push hi

  -- Compare
  IO.println "  f32 row: "
  for i in [0:headDim] do IO.println s!"    [{i}] = {f32Row[i]!}"
  IO.println "  f16 row (unpacked): "
  for i in [0:headDim] do IO.println s!"    [{i}] = {f16Row[i]!}"

  let mut maxAbsDiff := 0.0
  for i in [0:headDim] do
    let d := (f32Row[i]! - f16Row[i]!).abs
    if d > maxAbsDiff then maxAbsDiff := d
  IO.println s!"  max abs diff = {maxAbsDiff}"

  -- f16 precision: ~1e-3 for values near 1.0
  return maxAbsDiff < 0.005

/-- Run f32→f16 V cache pack kernel, verify unpacked f16 ≈ f32 source. -/
def runVPackF16 [GPUBackend β] (ctx : β) : IO Bool := do
  let numKVHeads := 1
  let maxSeqLen := 4
  let headDim := 8
  let halfDim := headDim / 2
  let cacheSize := numKVHeads * maxSeqLen * headDim
  let pos : UInt32 := 2

  -- Synthetic V cache row at pos=2
  let mut vData : Array Float := Array.replicate cacheSize 0.0
  for d in [0:headDim] do
    vData := vData.set! (pos.toNat * headDim + d) (0.1 + d.toFloat * 0.07)

  let vCacheF32 ← GPUBackend.allocBuffer ctx (cacheSize * 4).toUSize
  let vCacheF16 ← GPUBackend.allocBuffer ctx (cacheSize * 2).toUSize
  let paramsBuf ← GPUBackend.allocBuffer ctx (8 : USize)
  GPUBackend.writeBuffer ctx vCacheF32 (packFloats vData)
  GPUBackend.writeBuffer ctx vCacheF16 (packU32s (Array.replicate (cacheSize / 2) 0))
  GPUBackend.writeBuffer ctx paramsBuf (packU32s #[pos, 4])

  let shader := Hesper.Layers.Attention.packVCacheF32ToF16Kernel
                  numKVHeads maxSeqLen headDim
  let halfKvDim := numKVHeads * halfDim
  GPUBackend.execute ctx shader
    [("v_cache", vCacheF32), ("v_cache_f16", vCacheF16), ("params", paramsBuf)]
    ({ workgroupSize := { x := 32, y := 1, z := 1 },
       numWorkgroups := ((halfKvDim + 31) / 32, 1, 1) : Hesper.ExecConfig })

  let f16Bytes ← GPUBackend.readBuffer ctx vCacheF16 (cacheSize * 2).toUSize
  let halfRowOffset := pos.toNat * halfDim
  let unpacked : Array Float := (Array.range halfDim).foldl (init := #[]) fun acc dPair =>
    let u := unpackU32 f16Bytes (halfRowOffset + dPair)
    let (lo, hi) := unpackHalf2 u
    acc.push lo |>.push hi

  let mut maxAbsDiff := 0.0
  IO.println "  V row check (pos=2):"
  for d in [0:headDim] do
    let expected := 0.1 + d.toFloat * 0.07
    let got := unpacked[d]!
    let diff := (expected - got).abs
    IO.println s!"    [{d}] expected = {expected}, got = {got}, diff = {diff}"
    if diff > maxAbsDiff then maxAbsDiff := diff

  return maxAbsDiff < 0.005

def main : IO UInt32 := do
  IO.println "═══ RoPE-K F16 + V pack F16 unit tests ═══"
  let ctx ← CUDAContext.init

  IO.println ""
  IO.println "Test 1: RoPE-K f32 vs f16 parity"
  let ok1 ← runRopeKF16Parity ctx

  IO.println ""
  IO.println "Test 2: V cache f32→f16 pack"
  let ok2 ← runVPackF16 ctx

  if ok1 && ok2 then
    IO.println ""
    IO.println "✓ ALL PASS"
    return 0
  else
    IO.println ""
    IO.println "✗ FAIL"
    return 1
