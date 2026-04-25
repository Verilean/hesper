import Hesper.Circuit.IRv2
import Hesper.Circuit.Lowering
import Hesper.Circuit.Lowering_v2
import Hesper.Circuit.Dispatch_v2
import Hesper.Models.Gemma4_v2
import Hesper.Layers.Attention
import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.Basic

/-!
# Phase B8 PoC: RoPE-K + V combined KV-cache write via IRv2 ScatterMulti

Single IRv2 block:

    [ ScatterMulti
        reads  = [new_k, freq_factors, new_v]
        writes = [k_cache, v_cache]
        ops    = [ (cacheIdx, NeoX-rotated K),
                   (cacheIdx, new_v[laneIdx]) ] ]

This is the IR shape the production code calls `scatterMulti`: one
kernel, two output buffers, shared per-lane math.  Parity target:
`fusedRopeKAndCacheWriteKernel` (writes K with RoPE and V plain in a
single dispatch).  Both cache halves must match bit-for-bit.
-/

open Hesper
open Hesper.Circuit
open Hesper.Circuit.IRv2
open Hesper.Models.Gemma4_v2

def maxAbsDiffMulti (a b : Array Float) : Float := Id.run do
  let mut m : Float := 0.0
  for i in [0:a.size] do
    let d := (a[i]! - b[i]!).abs
    if d > m then m := d
  return m

def readF32BufMulti [GPUBackend β]
    (ctx : β) (buf : GPUBackend.Buf β) (n : Nat) : IO (Array Float) := do
  let bytes ← GPUBackend.readBuffer ctx buf (n * 4).toUSize
  let mut out : Array Float := Array.mkEmpty n
  for i in [0:n] do
    let f ← Hesper.Basic.bytesToFloat32 bytes (i * 4)
    out := out.push f
  return out

def main : IO Unit := do
  IO.println "=== Phase B8 PoC: K+V ScatterMulti parity via IRv2 ==="
  let ctx ← Hesper.CUDAContext.init
  let numKVHeads : Nat   := 4
  let headDim    : Nat   := 128
  let halfDim    : Nat   := headDim / 2
  let maxSeqLen  : Nat   := 128
  let kvDim      := numKVHeads * headDim
  let cacheSize  := numKVHeads * maxSeqLen * headDim
  let pos        : Nat   := 7
  let ropeBase   : Float := 10000.0
  IO.println s!"[Shapes] numKVHeads={numKVHeads} headDim={headDim} halfDim={halfDim}"
  IO.println s!"[Shapes] maxSeqLen={maxSeqLen} kvDim={kvDim} cacheSize={cacheSize}"
  IO.println s!"[RoPE ] pos={pos}  base={ropeBase}"

  -- Distinct deterministic K and V inputs so the test distinguishes
  -- them (a bug that swaps K↔V would not mask).
  let kArr : Array Float :=
    (List.range kvDim).toArray.map (fun i =>
      Float.sin (i.toFloat * 0.017) * 0.5 + (i.toFloat * 0.001))
  let vArr : Array Float :=
    (List.range kvDim).toArray.map (fun i =>
      Float.cos (i.toFloat * 0.021) * 0.3 - (i.toFloat * 0.0007))
  let freqArr : Array Float :=
    (List.range halfDim).toArray.map (fun i =>
      1.0 + 0.05 * Float.sin (i.toFloat * 0.3))
  let kBytes    ← Hesper.Basic.floatArrayToBytes kArr
  let vBytes    ← Hesper.Basic.floatArrayToBytes vArr
  let freqBytes ← Hesper.Basic.floatArrayToBytes freqArr
  let kBufSz    : USize := (kvDim   * 4).toUSize
  let freqBufSz : USize := (halfDim * 4).toUSize
  let cacheSzB  : USize := (cacheSize * 4).toUSize
  let zeroArr : Array Float := Array.replicate cacheSize 0.0
  let zeroBytes ← Hesper.Basic.floatArrayToBytes zeroArr

  -- Shared inputs.
  let kBuf ← GPUBackend.allocBuffer ctx kBufSz
  GPUBackend.writeBuffer ctx kBuf kBytes
  let vBuf ← GPUBackend.allocBuffer ctx kBufSz
  GPUBackend.writeBuffer ctx vBuf vBytes
  let freqBuf ← GPUBackend.allocBuffer ctx freqBufSz
  GPUBackend.writeBuffer ctx freqBuf freqBytes

  -- ================================================================
  -- REFERENCE: fusedRopeKAndCacheWriteKernel writes BOTH halves.
  -- ================================================================
  let kCacheRef ← GPUBackend.allocBuffer ctx cacheSzB
  let vCacheRef ← GPUBackend.allocBuffer ctx cacheSzB
  GPUBackend.writeBuffer ctx kCacheRef zeroBytes
  GPUBackend.writeBuffer ctx vCacheRef zeroBytes
  let paramsBytes : ByteArray := Id.run do
    let mut b := ByteArray.empty
    let pv := pos.toUInt32
    let cv := (pos + 1).toUInt32
    for v in [pv, cv] do
      b := b.push (UInt8.ofNat (v.toNat % 256))
      b := b.push (UInt8.ofNat ((v.toNat / 256) % 256))
      b := b.push (UInt8.ofNat ((v.toNat / 65536) % 256))
      b := b.push (UInt8.ofNat ((v.toNat / 16777216) % 256))
    return b
  let paramsBuf ← GPUBackend.allocBuffer ctx (2 * 4 : Nat).toUSize
  GPUBackend.writeBuffer ctx paramsBuf paramsBytes
  let refShader := Hesper.Layers.Attention.fusedRopeKAndCacheWriteKernel
                     numKVHeads maxSeqLen headDim kvDim ropeBase
  GPUBackend.executeWithConfig ctx refShader
    [("new_k", kBuf), ("new_v", vBuf),
     ("k_cache", kCacheRef), ("v_cache", vCacheRef),
     ("params", paramsBuf), ("freq_factors", freqBuf)]
    { numWorkgroups := ((kvDim + 63) / 64, 1, 1),
      workgroupSize := { x := 64, y := 1, z := 1 } }
  let kRef ← readF32BufMulti ctx kCacheRef cacheSize
  let vRef ← readF32BufMulti ctx vCacheRef cacheSize
  let sliceAt (arr : Array Float) (start len : Nat) : List Float :=
    (List.range len).map (fun k => arr[start + k]!)
  IO.println s!"[Ref ] K-cache[pos*hd+0..3] = {sliceAt kRef (pos * headDim) 4}"
  IO.println s!"[Ref ] V-cache[pos*hd+0..3] = {sliceAt vRef (pos * headDim) 4}"

  -- ================================================================
  -- IRv2: single ScatterMulti block writes both halves.
  -- ================================================================
  let kCacheV2 ← GPUBackend.allocBuffer ctx cacheSzB
  let vCacheV2 ← GPUBackend.allocBuffer ctx cacheSzB
  GPUBackend.writeBuffer ctx kCacheV2 zeroBytes
  GPUBackend.writeBuffer ctx vCacheV2 zeroBytes
  let kNewId    : Nat := 7000
  let freqId    : Nat := 7001
  let vNewId    : Nat := 7002
  let kCacheId  : Nat := 7003
  let vCacheId  : Nat := 7004
  let (_, graph) := runBuilder
    (buildRopeKVWriteLazy kNewId freqId vNewId kCacheId vCacheId
       pos ropeBase numKVHeads maxSeqLen headDim)
  IO.println s!"[IRv2] graph blocks: {graph.blocks.size}, tensors: {graph.tensors.size}"
  if graph.blocks.size != 1 then
    IO.println s!"FAIL: expected 1 block, got {graph.blocks.size}"
    IO.Process.exit 1
  -- Confirm the block body is ScatterMulti with 2 ops.
  match graph.blocks[0]!.body with
  | .ScatterMulti ops =>
    if ops.size != 2 then
      IO.println s!"FAIL: expected 2 ops in ScatterMulti, got {ops.size}"
      IO.Process.exit 1
    IO.println "[IRv2] block body is ScatterMulti with 2 ops (K + V)"
  | _ =>
    IO.println "FAIL: block body is not ScatterMulti"
    IO.Process.exit 1
  Hesper.Circuit.IRv2.runBlockGraph ctx graph
    (externalBufs :=
      [(kNewId,   kBuf),
       (freqId,   freqBuf),
       (vNewId,   vBuf),
       (kCacheId, kCacheV2),
       (vCacheId, vCacheV2)])
    (matmulLayers := [])
    (matmulInputBufs := [])
    (normHandles := [])
  let kV2 ← readF32BufMulti ctx kCacheV2 cacheSize
  let vV2 ← readF32BufMulti ctx vCacheV2 cacheSize
  IO.println s!"[IRv2] K-cache[pos*hd+0..3] = {sliceAt kV2 (pos * headDim) 4}"
  IO.println s!"[IRv2] V-cache[pos*hd+0..3] = {sliceAt vV2 (pos * headDim) 4}"

  let errK := maxAbsDiffMulti kRef kV2
  let errV := maxAbsDiffMulti vRef vV2
  IO.println s!"[Parity] max |errK|={errK}  max |errV|={errV}"

  -- Sanity: non-zero region is exactly the `pos` slot per kvHead.
  let checkStructural (label : String) (arr : Array Float) : IO Unit := do
    let mut wrote := false
    let mut stray := false
    for i in [0:cacheSize] do
      if arr[i]! != 0.0 then
        wrote := true
        let withinHead := i % (maxSeqLen * headDim)
        if withinHead < pos * headDim ∨ withinHead ≥ (pos + 1) * headDim then
          stray := true
    if !wrote then
      IO.println s!"FAIL: {label} wrote no non-zero slots"
      IO.Process.exit 1
    if stray then
      IO.println s!"FAIL: {label} wrote outside expected pos slot"
      IO.Process.exit 1
  checkStructural "reference K-cache" kRef
  checkStructural "reference V-cache" vRef

  if errK == 0.0 ∧ errV == 0.0 then
    IO.println "PASS: IRv2 ScatterMulti is BIT-IDENTICAL to fusedRopeKAndCacheWriteKernel"
  else if errK < 1e-5 ∧ errV < 1e-5 then
    IO.println s!"PASS (≈): K+V match within 1e-5"
  else
    IO.println s!"FAIL: K+V mismatch (errK={errK}, errV={errV})"
    IO.Process.exit 1
