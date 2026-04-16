import Hesper.Backend.CUDA
import Hesper.Circuit.IR
import Hesper.Circuit.Lowering
import Hesper.Circuit.Passes
import Hesper.WebGPU.BufferOps
import Hesper.Basic

/-!
End-to-end GPU test for `CircuitM.scatterInto` with a *dynamic* address
expression — the pattern needed for KV cache writes.

Setup (mimics V cache write for Gemma 4):
  numHeads  = 4
  seqLen    = 8
  headDim   = 16
  (total cache size = 4 * 8 * 16 = 512)

We scatter src[kvDim=64] into dst at per-head position `pos=3`.
Destination address per lane `i`:
    head = i / headDim      (which head group this element belongs to)
    d    = i mod headDim    (offset within head)
    addr = head * seqLen * headDim + pos * headDim + d

So lane 0 writes to dst[3*16] = dst[48], lane 1 → dst[49], ..., lane 15 → dst[63],
lane 16 writes to dst[128+48] = dst[176], ..., lane 31 → dst[191], and so on.

Source values: src[i] = 100 + i.
Destination pre-filled with zeros.

After the dispatch we read dst back and verify:
  - dst[head*seqLen*headDim + pos*headDim + d] = 100 + (head*headDim + d)
  - All other dst slots remain 0.
-/

open Hesper
open Hesper.Circuit

abbrev β := Hesper.CUDAContext

unsafe def main : IO Unit := do
  let numHeads : Nat := 4
  let seqLen   : Nat := 8
  let headDim  : Nat := 16
  let kvDim    : Nat := numHeads * headDim       -- 64 (elements per write)
  let cacheSz  : Nat := numHeads * seqLen * headDim   -- 512 (dst size)
  let pos      : Nat := 3
  let ctx ← Hesper.CUDAContext.init

  -- Source: [kvDim] with src[i] = 100 + i.
  let srcArr : Array Float := Array.ofFn (n := kvDim) fun i => 100.0 + i.val.toFloat
  let srcBytes ← Hesper.WebGPU.BufferOps.floatArrayToBytes srcArr
  let srcBuf ← GPUBackend.allocBuffer (β := β) ctx (4 * kvDim).toUSize
  GPUBackend.writeBuffer (β := β) ctx srcBuf srcBytes

  -- Destination pre-filled with zeros.
  let zeroArr : Array Float := Array.ofFn (n := cacheSz) fun _ => 0.0
  let zeroBytes ← Hesper.WebGPU.BufferOps.floatArrayToBytes zeroArr
  let dstBuf ← GPUBackend.allocBuffer (β := β) ctx (4 * cacheSz).toUSize
  GPUBackend.writeBuffer (β := β) ctx dstBuf zeroBytes

  -- Pos as f32 in a [1]-shape broadcast buffer.
  let posBuf ← GPUBackend.allocBuffer (β := β) ctx (4 : USize)
  let posBytes ← Hesper.Basic.floatToBytes pos.toFloat
  GPUBackend.writeBuffer (β := β) ctx posBuf posBytes

  -- Build: scatterInto dst srcShape=[kvDim] inputs=[src, pos]
  --   value = .input 0  (src[laneIdx])
  --   addr  = (laneIdx / headDim) * (seqLen*headDim)
  --         + .input 1 * headDim
  --         + laneIdx mod headDim
  let ccRef : IO.Ref (Option (CompiledCircuit β)) ← IO.mkRef none
  Hesper.Circuit.runCachedFused ctx ccRef
    (do
      let src ← CircuitM.registerExternal
                  (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
                  srcBuf #[kvDim] .f32 .Global
      let dst ← CircuitM.registerExternal
                  (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
                  dstBuf #[cacheSz] .f32 .Global
      let posT ← CircuitM.registerExternal
                  (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
                  posBuf #[1] .f32 .Global
      let valueExpr : ScalarExp := .input 0
      let i    : ScalarExp := .laneIdx
      let head : ScalarExp := .idiv i (.const headDim.toFloat)
      let d    : ScalarExp := .mod  i (.const headDim.toFloat)
      let addrExpr : ScalarExp :=
        head * .const (seqLen * headDim).toFloat
        + .input 1 * .const headDim.toFloat
        + d
      let _ ← CircuitM.scatterInto dst #[kvDim] #[src, posT] valueExpr addrExpr
      pure ())
    [(0, srcBuf), (1, dstBuf), (2, posBuf)]

  IO.println "Waiting for GPU..."

  -- Read back dst.
  let outBytes ← GPUBackend.readBuffer (β := β) ctx dstBuf (4 * cacheSz).toUSize

  let mut ok   : Nat := 0
  let mut fail : Nat := 0
  let mut firstErr : Option (Nat × Float × Float) := none
  for head in [0:numHeads] do
    for s in [0:seqLen] do
      for d in [0:headDim] do
        let addr := head * seqLen * headDim + s * headDim + d
        let actual ← Hesper.Basic.bytesToFloat32 outBytes (addr * 4)
        let expected : Float :=
          if s == pos then 100.0 + (head * headDim + d).toFloat else 0.0
        if (actual - expected).abs < 1e-6 then
          ok := ok + 1
        else
          fail := fail + 1
          if firstErr.isNone then firstErr := some (addr, actual, expected)

  if fail == 0 then
    IO.println s!"✓ PASS: all {ok} cells match (scatter wrote {kvDim} values into dst[pos={pos}] slot of each head)"
  else
    IO.println s!"✗ FAIL: {fail} mismatches, {ok} matches"
    match firstErr with
    | some (addr, a, e) => IO.println s!"  first mismatch: dst[{addr}] = {a}, expected {e}"
    | none => pure ()
    IO.Process.exit 1
