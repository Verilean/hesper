import Hesper.Circuit.IRv2
import Hesper.Circuit.Lowering
import Hesper.Circuit.Lowering_v2
import Hesper.Circuit.Dispatch_v2
import Hesper.Models.Gemma4_v2
import Hesper.Models.Gemma4.Kernels
import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.Basic

/-!
# Phase B9 PoC: RoPE-Q (in-place) parity via IRv2

Builds a 1-block IRv2 graph:

    [ Scatter
        reads  = [new_q, freq_factors]
        writes = [q_out]
        indexExpr = .laneIdx                   -- same-position write
        applyBody = NeoX-rotated Q value ]

Parity target: `ropeWithFreqFactorsKernel` from
`Hesper.Models.Gemma4.Kernels`.  The hand-tuned kernel dispatches
`numHeads * (headDim/2)` lanes (each writes 2 paired outputs); the
IRv2 expression dispatches `numHeads * headDim` lanes (each writes 1
element).  Both evaluate the same fp32 formula with the same operand
ordering per output, so parity is bit-identical.
-/

open Hesper
open Hesper.Circuit
open Hesper.Circuit.IRv2
open Hesper.Models.Gemma4_v2

def maxAbsDiffQR (a b : Array Float) : Float := Id.run do
  let mut m : Float := 0.0
  for i in [0:a.size] do
    let d := (a[i]! - b[i]!).abs
    if d > m then m := d
  return m

def readF32BufQR [GPUBackend β]
    (ctx : β) (buf : GPUBackend.Buf β) (n : Nat) : IO (Array Float) := do
  let bytes ← GPUBackend.readBuffer ctx buf (n * 4).toUSize
  let mut out : Array Float := Array.mkEmpty n
  for i in [0:n] do
    let f ← Hesper.Basic.bytesToFloat32 bytes (i * 4)
    out := out.push f
  return out

def main : IO Unit := do
  IO.println "=== Phase B9 PoC: RoPE-Q parity via IRv2 ==="
  let ctx ← Hesper.CUDAContext.init
  let numHeads : Nat   := 8
  let headDim  : Nat   := 128
  let halfDim  : Nat   := headDim / 2
  let qDim     := numHeads * headDim
  let pos      : Nat   := 7
  let ropeBase : Float := 10000.0
  IO.println s!"[Shapes] numHeads={numHeads} headDim={headDim} halfDim={halfDim} qDim={qDim}"
  IO.println s!"[RoPE ] pos={pos} base={ropeBase}"

  -- Deterministic Q input and freq_factors.
  let qArr : Array Float :=
    (List.range qDim).toArray.map (fun i =>
      Float.sin (i.toFloat * 0.019) * 0.4 + (i.toFloat * 0.0008))
  let freqArr : Array Float :=
    (List.range halfDim).toArray.map (fun i =>
      1.0 + 0.05 * Float.sin (i.toFloat * 0.3))
  let qBytes    ← Hesper.Basic.floatArrayToBytes qArr
  let freqBytes ← Hesper.Basic.floatArrayToBytes freqArr
  let qBufSz    : USize := (qDim    * 4).toUSize
  let freqBufSz : USize := (halfDim * 4).toUSize
  -- Pre-zero output buffers so lanes that the dispatcher overshoots
  -- (shouldn't exist here) are visibly unaffected.
  let zeroArr : Array Float := Array.replicate qDim 0.0
  let zeroBytes ← Hesper.Basic.floatArrayToBytes zeroArr

  let qBuf ← GPUBackend.allocBuffer ctx qBufSz
  GPUBackend.writeBuffer ctx qBuf qBytes
  let freqBuf ← GPUBackend.allocBuffer ctx freqBufSz
  GPUBackend.writeBuffer ctx freqBuf freqBytes

  -- ================================================================
  -- REFERENCE: ropeWithFreqFactorsKernel.
  -- ================================================================
  let qOutRef ← GPUBackend.allocBuffer ctx qBufSz
  GPUBackend.writeBuffer ctx qOutRef zeroBytes
  -- params = [pos, cacheLen=pos+1].
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
  let refShader := Hesper.Models.Gemma4.ropeWithFreqFactorsKernel
                     headDim numHeads ropeBase
  -- Reference dispatches `numHeads * halfDim` lanes — each writes 2 elts.
  let refLanes := numHeads * halfDim
  GPUBackend.executeWithConfig ctx refShader
    [("input", qBuf), ("output", qOutRef),
     ("params", paramsBuf), ("freq_factors", freqBuf)]
    { numWorkgroups := ((refLanes + 63) / 64, 1, 1),
      workgroupSize := { x := 64, y := 1, z := 1 } }
  let refArr ← readF32BufQR ctx qOutRef qDim
  IO.println s!"[Ref ] Q[0..4] = {refArr.toList.take 4}"
  IO.println s!"[Ref ] Q[halfDim..halfDim+3] = {refArr.toList.drop halfDim |>.take 4}"

  -- ================================================================
  -- IRv2: single Scatter block (in-place layout, one lane per element).
  -- ================================================================
  let qOutV2 ← GPUBackend.allocBuffer ctx qBufSz
  GPUBackend.writeBuffer ctx qOutV2 zeroBytes
  let qNewId    : Nat := 8000
  let freqId    : Nat := 8001
  let qOutId    : Nat := 8002
  let (_, graph) := runBuilder
    (buildRopeQLazy qNewId freqId qOutId pos ropeBase numHeads headDim)
  IO.println s!"[IRv2] graph blocks: {graph.blocks.size}, tensors: {graph.tensors.size}"
  if graph.blocks.size != 1 then
    IO.println s!"FAIL: expected 1 block, got {graph.blocks.size}"
    IO.Process.exit 1
  match graph.blocks[0]!.body with
  | .Scatter _ _ => IO.println "[IRv2] block body is Scatter (as expected)"
  | _            => do
    IO.println "FAIL: block body is not Scatter"
    IO.Process.exit 1
  Hesper.Circuit.IRv2.runBlockGraph ctx graph
    (externalBufs :=
      [(qNewId, qBuf), (freqId, freqBuf), (qOutId, qOutV2)])
    (matmulLayers := [])
    (matmulInputBufs := [])
    (normHandles := [])
  let v2Arr ← readF32BufQR ctx qOutV2 qDim
  IO.println s!"[IRv2] Q[0..4] = {v2Arr.toList.take 4}"
  IO.println s!"[IRv2] Q[halfDim..halfDim+3] = {v2Arr.toList.drop halfDim |>.take 4}"

  let err := maxAbsDiffQR refArr v2Arr
  IO.println s!"[Parity] max |err| over Q output ({qDim} elems) = {err}"

  -- Sanity: at least some non-zero output (would mask a silent no-op).
  let mut nonZero : Nat := 0
  for i in [0:qDim] do
    if refArr[i]! != 0.0 then nonZero := nonZero + 1
  IO.println s!"[Ref ] {nonZero}/{qDim} output slots are non-zero"
  if nonZero < qDim / 2 then
    IO.println "FAIL: reference kernel produced suspiciously many zeros"
    IO.Process.exit 1

  if err == 0.0 then
    IO.println "PASS: IRv2 RoPE-Q is BIT-IDENTICAL to ropeWithFreqFactorsKernel"
  else if err < 1e-5 then
    IO.println s!"PASS (≈): RoPE-Q matches reference to {err}"
  else
    IO.println s!"FAIL: RoPE-Q mismatch (max |err| = {err})"
    IO.Process.exit 1
