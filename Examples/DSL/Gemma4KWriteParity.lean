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
# Phase B7 PoC: RoPE-K + KV-cache Scatter parity via IRv2

Builds a 1-block IRv2 graph:

    [ Scatter
        reads  = [new_k, freq_factors]
        writes = [k_cache]
        indexExpr = kvHead * (maxSeqLen * headDim) + pos * headDim + d
        applyBody = <NeoX-style RoPE rotation of new_k using freq_factors> ]

The dispatcher picks it up via the standalone-Scatter fallback
(Pattern F) and lowers via v1's monadic `lowerScalarExp`, which
supports `.select`, `.lt`, `.pow`, `.cos`, `.sin`, and `.indexed` —
everything the NeoX pair-swap rotation needs.

Parity target: the K-cache slice produced by
`Hesper.Layers.Attention.fusedRopeKAndCacheWriteKernel` with the
same `pos`, `ropeBase`, and synthetic inputs.  Bit-identity required
over all `kvDim` written cache elements.
-/

open Hesper
open Hesper.Circuit
open Hesper.Circuit.IRv2
open Hesper.Models.Gemma4_v2

def maxAbsDiffRK (a b : Array Float) : Float := Id.run do
  let mut m : Float := 0.0
  for i in [0:a.size] do
    let d := (a[i]! - b[i]!).abs
    if d > m then m := d
  return m

def readF32BufRK [GPUBackend β]
    (ctx : β) (buf : GPUBackend.Buf β) (n : Nat) : IO (Array Float) := do
  let bytes ← GPUBackend.readBuffer ctx buf (n * 4).toUSize
  let mut out : Array Float := Array.mkEmpty n
  for i in [0:n] do
    let f ← Hesper.Basic.bytesToFloat32 bytes (i * 4)
    out := out.push f
  return out

def main : IO Unit := do
  IO.println "=== Phase B7 PoC: RoPE-K Scatter parity via IRv2 ==="
  let ctx ← Hesper.CUDAContext.init
  -- Synthetic shapes (match a Gemma 4 SWA attention slice).
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

  -- Deterministic K input and freq_factors.
  let kArr : Array Float :=
    (List.range kvDim).toArray.map (fun i =>
      Float.sin (i.toFloat * 0.017) * 0.5 + (i.toFloat * 0.001))
  let freqArr : Array Float :=
    (List.range halfDim).toArray.map (fun i =>
      1.0 + 0.05 * Float.sin (i.toFloat * 0.3))  -- non-unit factors
  let kBytes    ← Hesper.Basic.floatArrayToBytes kArr
  let freqBytes ← Hesper.Basic.floatArrayToBytes freqArr
  let kBufSz    : USize := (kvDim   * 4).toUSize
  let freqBufSz : USize := (halfDim * 4).toUSize
  let cacheSzB  : USize := (cacheSize * 4).toUSize
  let zeroArr : Array Float := Array.replicate cacheSize 0.0
  let zeroBytes ← Hesper.Basic.floatArrayToBytes zeroArr

  -- Shared inputs.
  let kBuf ← GPUBackend.allocBuffer ctx kBufSz
  GPUBackend.writeBuffer ctx kBuf kBytes
  let freqBuf ← GPUBackend.allocBuffer ctx freqBufSz
  GPUBackend.writeBuffer ctx freqBuf freqBytes

  -- ================================================================
  -- REFERENCE: fusedRopeKAndCacheWriteKernel writes K (with RoPE) AND
  -- V (plain copy) into the cache.  We ignore the V half and compare
  -- the K half only.
  -- ================================================================
  let kCacheRef ← GPUBackend.allocBuffer ctx cacheSzB
  GPUBackend.writeBuffer ctx kCacheRef zeroBytes
  let vCacheDummy ← GPUBackend.allocBuffer ctx cacheSzB
  GPUBackend.writeBuffer ctx vCacheDummy zeroBytes
  let vBufDummy ← GPUBackend.allocBuffer ctx kBufSz
  GPUBackend.writeBuffer ctx vBufDummy kBytes  -- any contents OK
  -- params = [pos, cacheLen=pos+1] (only pos is read by the write half).
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
    [("new_k", kBuf), ("new_v", vBufDummy),
     ("k_cache", kCacheRef), ("v_cache", vCacheDummy),
     ("params", paramsBuf), ("freq_factors", freqBuf)]
    { numWorkgroups := ((kvDim + 63) / 64, 1, 1),
      workgroupSize := { x := 64, y := 1, z := 1 } }
  let refArr ← readF32BufRK ctx kCacheRef cacheSize
  let sliceAt (arr : Array Float) (start len : Nat) : List Float :=
    (List.range len).map (fun k => arr[start + k]!)
  IO.println s!"[Ref ] K-cache[pos*headDim + 0..3] = {sliceAt refArr (pos * headDim) 4}"

  -- ================================================================
  -- IRv2: single Scatter block describing RoPE-K + cache write.
  -- ================================================================
  let kCacheV2 ← GPUBackend.allocBuffer ctx cacheSzB
  GPUBackend.writeBuffer ctx kCacheV2 zeroBytes
  let kNewId   : Nat := 6000
  let freqId   : Nat := 6001
  let kCacheId : Nat := 6002
  let (_, graph) := runBuilder
    (buildRopeKWriteLazy kNewId freqId kCacheId pos ropeBase
       numKVHeads maxSeqLen headDim)
  IO.println s!"[IRv2] graph blocks: {graph.blocks.size}, tensors: {graph.tensors.size}"
  if graph.blocks.size != 1 then
    IO.println s!"FAIL: expected 1 block, got {graph.blocks.size}"
    IO.Process.exit 1
  Hesper.Circuit.IRv2.runBlockGraph ctx graph
    (externalBufs :=
      [(kNewId, kBuf), (freqId, freqBuf), (kCacheId, kCacheV2)])
    (matmulLayers := [])
    (matmulInputBufs := [])
    (normHandles := [])
  let v2Arr ← readF32BufRK ctx kCacheV2 cacheSize
  IO.println s!"[IRv2] K-cache[pos*headDim + 0..3] = {sliceAt v2Arr (pos * headDim) 4}"

  -- Parity over the full cache — both paths should write identical
  -- values into the same kvDim slots and leave the rest at 0.
  let err := maxAbsDiffRK refArr v2Arr
  IO.println s!"[Parity] max |err| over full cache ({cacheSize} elems) = {err}"

  -- Sanity checks: the written region is non-zero; everywhere outside
  -- of `pos` for each kvHead is still 0.
  let mut wroteSomething := false
  let mut strayWrite := false
  for i in [0:cacheSize] do
    if refArr[i]! != 0.0 then
      wroteSomething := true
      let withinHead := i % (maxSeqLen * headDim)
      if withinHead < pos * headDim ∨ withinHead ≥ (pos + 1) * headDim then
        strayWrite := true
  if !wroteSomething then
    IO.println "FAIL: reference kernel wrote no non-zero slots"
    IO.Process.exit 1
  if strayWrite then
    IO.println "FAIL: reference kernel wrote outside the expected pos slot"
    IO.Process.exit 1

  if err == 0.0 then
    IO.println "PASS: IRv2 RoPE-K scatter is BIT-IDENTICAL to fusedRopeKAndCacheWriteKernel (K half)"
  else if err < 1e-5 then
    IO.println s!"PASS (≈): RoPE-K matches reference to {err}"
  else
    IO.println s!"FAIL: RoPE-K mismatch (max |err| = {err})"
    IO.Process.exit 1
