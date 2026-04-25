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
# Phase B6 PoC: KV cache V-scatter parity via IRv2

Builds a 1-block IRv2 graph:

    [ Scatter
        indexExpr = kvHead * (maxSeqLen * headDim) + pos * headDim + d
        applyBody = .input 0 ]

where `kvHead = laneIdx / headDim`, `d = laneIdx % headDim`, for a
per-token V write to the Gemma 4 KV cache layout
`[numKVHeads, maxSeqLen, headDim]`.

The dispatcher recognises the standalone Scatter block and lowers it
via `lowerBlockGraph` → generic scatter WGSL/PTX.

Parity target: the V-cache slice produced by hesper's hand-tuned
`fusedCacheWriteKVKernel` (same input V, same position, same layout).
Assert per-element bit-identity at all lanes.

Scope: V-only.  Production's `scatterMulti` writes K+V atomically;
this PoC proves the IRv2 Scatter block is correct on the V half.
Adding K would require RoPE rotation (a separate IR-extension PoC).
-/

open Hesper
open Hesper.Circuit
open Hesper.Circuit.IRv2
open Hesper.Models.Gemma4_v2

def readF32BufKV [GPUBackend β]
    (ctx : β) (buf : GPUBackend.Buf β) (n : Nat) : IO (Array Float) := do
  let bytes ← GPUBackend.readBuffer ctx buf (n * 4).toUSize
  let mut out : Array Float := Array.mkEmpty n
  for i in [0:n] do
    let f ← Hesper.Basic.bytesToFloat32 bytes (i * 4)
    out := out.push f
  return out

def maxAbsDiffKV (a b : Array Float) : Float := Id.run do
  let mut m : Float := 0.0
  for i in [0:a.size] do
    let d := (a[i]! - b[i]!).abs
    if d > m then m := d
  return m

def main : IO Unit := do
  IO.println "=== Phase B6 PoC: KV-cache V-scatter parity via IRv2 ==="
  let ctx ← Hesper.CUDAContext.init
  -- Synthetic shapes matching Gemma 4 E4B SWA layers.
  let numKVHeads : Nat := 4
  let headDim    : Nat := 128
  let maxSeqLen  : Nat := 128       -- small for a fast PoC (full 8192 also works)
  let kvDim      := numKVHeads * headDim
  let cacheSize  := numKVHeads * maxSeqLen * headDim
  let pos        : Nat := 7
  IO.println s!"[Shapes] numKVHeads={numKVHeads} headDim={headDim} maxSeqLen={maxSeqLen}"
  IO.println s!"[Shapes] kvDim={kvDim} cacheSize={cacheSize} pos={pos}"

  -- Deterministic V input.
  let vArr : Array Float :=
    (List.range kvDim).toArray.map (fun i =>
      (i.toFloat * 0.013) + Float.sin (i.toFloat * 0.21) * 0.3)
  let vBytes ← Hesper.Basic.floatArrayToBytes vArr
  let vBufSz    : USize := (kvDim * 4).toUSize
  let cacheSzB  : USize := (cacheSize * 4).toUSize

  -- Upload V input once (shared by both paths).
  let vBuf ← GPUBackend.allocBuffer ctx vBufSz
  GPUBackend.writeBuffer ctx vBuf vBytes

  -- ================================================================
  -- REFERENCE: fusedCacheWriteKVKernel with dummy K + params(pos=7).
  -- ================================================================
  -- Zero-initialised reference cache.
  let zeroArr : Array Float := Array.replicate cacheSize 0.0
  let zeroBytes ← Hesper.Basic.floatArrayToBytes zeroArr
  let vCacheRef ← GPUBackend.allocBuffer ctx cacheSzB
  GPUBackend.writeBuffer ctx vCacheRef zeroBytes
  -- Dummy K + K-cache buffers (written but not read).
  let kBufDummy ← GPUBackend.allocBuffer ctx vBufSz
  GPUBackend.writeBuffer ctx kBufDummy vBytes       -- any contents fine
  let kCacheDummy ← GPUBackend.allocBuffer ctx cacheSzB
  GPUBackend.writeBuffer ctx kCacheDummy zeroBytes
  -- params = #[pos, cacheLen]  (only pos is used by the write kernel).
  let paramsBytes : ByteArray := Id.run do
    let mut b := ByteArray.empty
    for v in [pos.toUInt32, pos.toUInt32 + 1] do
      b := b.push (UInt8.ofNat (v.toNat % 256))
      b := b.push (UInt8.ofNat ((v.toNat / 256) % 256))
      b := b.push (UInt8.ofNat ((v.toNat / 65536) % 256))
      b := b.push (UInt8.ofNat ((v.toNat / 16777216) % 256))
    return b
  let paramsBuf ← GPUBackend.allocBuffer ctx (2 * 4 : Nat).toUSize
  GPUBackend.writeBuffer ctx paramsBuf paramsBytes
  -- Run the hand-tuned kernel.
  let refShader := Hesper.Layers.Attention.fusedCacheWriteKVKernel
                     numKVHeads maxSeqLen headDim kvDim
  GPUBackend.executeWithConfig ctx refShader
    [("new_k", kBufDummy), ("new_v", vBuf),
     ("k_cache", kCacheDummy), ("v_cache", vCacheRef),
     ("params", paramsBuf)]
    { numWorkgroups := ((kvDim + 63) / 64, 1, 1),
      workgroupSize := { x := 64, y := 1, z := 1 } }
  let refArr ← readF32BufKV ctx vCacheRef cacheSize
  -- Preview the slots we expect to have been written.
  let sliceAt (arr : Array Float) (start len : Nat) : List Float :=
    (List.range len).map (fun k => arr[start + k]!)
  IO.println s!"[Ref ] V-cache[pos*headDim + 0..3] = {sliceAt refArr (pos * headDim) 4}"

  -- ================================================================
  -- IRv2: single Scatter block.  New cache init to zero; dispatcher
  -- writes only position `pos`.
  -- ================================================================
  let vCacheV2 ← GPUBackend.allocBuffer ctx cacheSzB
  GPUBackend.writeBuffer ctx vCacheV2 zeroBytes
  let vNewId   : Nat := 5000
  let vCacheId : Nat := 5001
  let (_, graph) := runBuilder
    (buildKVWriteLazy vNewId vCacheId pos numKVHeads maxSeqLen headDim)
  IO.println s!"[IRv2] graph blocks: {graph.blocks.size}, tensors: {graph.tensors.size}"
  if graph.blocks.size != 1 then
    IO.println s!"FAIL: expected 1 block, got {graph.blocks.size}"
    IO.Process.exit 1
  Hesper.Circuit.IRv2.runBlockGraph ctx graph
    (externalBufs := [(vNewId, vBuf), (vCacheId, vCacheV2)])
    (matmulLayers := [])
    (matmulInputBufs := [])
    (normHandles := [])
  let v2Arr ← readF32BufKV ctx vCacheV2 cacheSize
  IO.println s!"[IRv2] V-cache[pos*headDim + 0..3] = {sliceAt v2Arr (pos * headDim) 4}"

  -- Parity over the full cache — both should match in written slots
  -- AND leave untouched slots at 0.
  let err := maxAbsDiffKV refArr v2Arr
  IO.println s!"[Parity] max |err| over full cache ({cacheSize} elems) = {err}"

  -- Sanity: confirm the correct region is non-zero and that slots
  -- outside the written `pos` are all zero.  (V input may contain
  -- incidental zeros, so we don't require N == kvDim exactly.)
  let mut wroteSomething := false
  let mut strayWrite := false
  for i in [0:cacheSize] do
    if refArr[i]! != 0.0 then
      wroteSomething := true
      -- i must fall within kvHead's `pos` slot: i mod (maxSeqLen*headDim) ∈ [pos*headDim, pos*headDim+headDim)
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
    IO.println "PASS: IRv2 V-scatter is BIT-IDENTICAL to fusedCacheWriteKVKernel (V half)"
  else if err < 1e-6 then
    IO.println s!"PASS (≈): V-scatter matches reference to {err}"
  else
    IO.println s!"FAIL: V-scatter mismatch (max |err| = {err})"
    IO.Process.exit 1
