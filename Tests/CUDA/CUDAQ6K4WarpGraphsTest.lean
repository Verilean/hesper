import Hesper.CUDA.FFI
import Hesper.CUDA.Buffer
import Hesper.CUDA.CodeGen
import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.Layers.Linear

set_option maxRecDepth 2048

/-!
# Q6_K 4-warp + CUDA Graphs capture/replay reproduction test

End-to-end production exhibits a bug: when `fusedQ6KLinearDP4A4WarpKernel`
is dispatched 14 own-KV layers × 1 ffn_down per layer = 14 dispatches per
token, with **CUDA Graphs ON**, the output starts diverging at decode
token 3.  graphs OFF works perfectly.  lm_head (1 dispatch / token,
graphs ON) also works.

This test reproduces the multi-dispatch graphs-ON pattern in isolation:
- Allocate N independent (weight, input, output) buffer triples.
- Run two configurations:
    A. **eager**: launch the 4-warp kernel N times directly.
    B. **graph capture+replay**: capture N dispatches into a CUDA graph,
       then `cuGraphLaunch` to replay.
- Compare every output buffer pairwise.  If A == B → graphs handling
  is fine and the production bug is elsewhere.  If A != B → the bug is
  in our 4-warp + graphs interaction (cubin/smem layout).
-/

open Hesper
open Hesper.CUDA
open Hesper.WGSL.Monad (ShaderM)
open Hesper.Layers.Linear

private def f64ToF32Bits (f : Float) : UInt32 :=
  let b := f.toBits
  let s := (b >>> 63) &&& 1
  let e64 := (b >>> 52) &&& 0x7FF
  let m64 := b &&& 0x000FFFFFFFFFFFFF
  if e64 == 0 then 0
  else
    let eUnb : Int := Int.ofNat e64.toNat - 1023
    let e32i : Int := eUnb + 127
    if e32i ≤ 0 then 0
    else if e32i ≥ 255 then (s.toUInt32 <<< 31) ||| ((0xFF : UInt32) <<< 23)
    else
      let lower29 := m64 &&& (0x1FFFFFFF : UInt64)
      let m32Truncated := (m64 >>> 29).toUInt32 &&& (0x7FFFFF : UInt32)
      let halfway : UInt64 := 0x10000000
      let roundUp :=
        lower29 > halfway ||
        (lower29 == halfway && (m32Truncated &&& 1) == 1)
      let m32 := if roundUp then m32Truncated + 1 else m32Truncated
      let (m32Final, e32i') := if m32 == 0x800000 then (0, e32i + 1) else (m32, e32i)
      if e32i' ≥ 255 then (s.toUInt32 <<< 31) ||| ((0xFF : UInt32) <<< 23)
      else
        let e32 : UInt32 := e32i'.toNat.toUInt32
        (s.toUInt32 <<< 31) ||| (e32 <<< 23) ||| m32Final

private def f32BitsToF64 (bits : UInt32) : Float :=
  let e := (bits >>> 23) &&& 0xFF; let m := bits &&& (0x7FFFFF : UInt32); let s := bits >>> 31
  if e == 0 then 0.0 else
    let v := (1.0 + m.toNat.toFloat / 8388608.0) * Float.pow 2.0 (e.toNat.toFloat - 127.0)
    if s == 1 then -v else v

private def packF32 (arr : Array Float) : ByteArray :=
  arr.foldl (init := ByteArray.empty) fun acc f => let b := f64ToF32Bits f
    acc.push b.toUInt8 |>.push (b>>>8).toUInt8 |>.push (b>>>16).toUInt8 |>.push (b>>>24).toUInt8

private def unpackF32 (ba : ByteArray) (i : Nat) : Float :=
  let o := i * 4
  let b0 : UInt32 := ba.get! o |>.toUInt32
  let b1 : UInt32 := ba.get! (o+1) |>.toUInt32
  let b2 : UInt32 := ba.get! (o+2) |>.toUInt32
  let b3 : UInt32 := ba.get! (o+3) |>.toUInt32
  f32BitsToF64 (b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24))

private def lcg (seed : UInt64) : UInt64 := seed * 6364136223846793005 + 1442695040888963407

private def genInput (n : Nat) (seed : UInt64) : Array Float := Id.run do
  let mut arr := Array.mkEmpty n
  let mut s := seed
  for _ in [0:n] do
    s := lcg s
    let u : Float := (s.toNat % 65536).toFloat / 32768.0
    arr := arr.push (u - 1.0)
  return arr

private def genQ6KBlock (seed : UInt64) : ByteArray := Id.run do
  let mut bytes := ByteArray.empty
  let mut s := seed
  for _ in [0:128] do
    s := lcg s
    bytes := bytes.push (s.toNat % 256).toUInt8
  for _ in [0:64] do
    s := lcg s
    bytes := bytes.push (s.toNat % 256).toUInt8
  for _ in [0:16] do
    s := lcg s
    let v : Int := (Int.ofNat (s.toNat % 32)) - 16
    let asU8 : UInt8 := if v < 0 then ((256 + v).toNat).toUInt8 else v.toNat.toUInt8
    bytes := bytes.push asU8
  bytes := bytes.push 0x1F
  bytes := bytes.push 0x21
  bytes

private def genQ6KWeights (outDim blocksPerRow : Nat) (seed : UInt64) : ByteArray := Id.run do
  let mut bytes := ByteArray.empty
  let mut s := seed
  for _row in [0:outDim] do
    for _blk in [0:blocksPerRow] do
      s := lcg s
      let blk := genQ6KBlock s
      bytes := bytes ++ blk
  bytes

def main : IO Unit := do
  IO.println "═══ Q6_K 4-warp + CUDA Graphs capture/replay reproduction ═══\n"

  -- Production-shape ffn_down: K=10240, N=2560, run 14 layers.
  let inDim     := 10240
  let outDim    := 2560
  let numLayers := 14
  let blocksPerRow := inDim / 256
  IO.println s!"inDim={inDim}, outDim={outDim}, numLayers={numLayers}"

  let cuda ← CUDAContext.init

  -- Allocate one (weight, q8 input, output_eager, output_graph) triple per
  -- layer.  Each layer gets DIFFERENT random weights so we exercise the
  -- per-layer arg substitution that graph capture must record.
  let mut wBufs   : List CUDABuffer := []
  let mut qBufs   : List CUDABuffer := []
  let mut outE    : List CUDABuffer := []
  let mut outG    : List CUDABuffer := []
  let mut prepEager : List (IO.Ref (Option CUDACachedDispatch)) := []
  let mut prepGraph : List (IO.Ref (Option CUDACachedDispatch)) := []
  for layer in [0:numLayers] do
    let inputArr := genInput inDim (0xCAFEBABE * (layer + 1).toUInt64)
    let weightBytes := genQ6KWeights outDim blocksPerRow (0xDEADBEEF * (layer + 1).toUInt64)
    let padBytes : Nat := (4 - (weightBytes.size % 4)) % 4
    let mut wbp := weightBytes
    for _ in [0:padBytes] do wbp := wbp.push 0
    let inBuf ← GPUBackend.allocBuffer cuda (4 * inDim).toUSize
    GPUBackend.writeBuffer cuda inBuf (packF32 inputArr)
    let wBuf ← GPUBackend.allocBuffer cuda wbp.size.toUSize
    GPUBackend.writeBuffer cuda wBuf wbp
    let nQ8 := inDim / 32
    let qSz : USize := (nQ8 * 9 * 4).toUSize
    let qBuf ← GPUBackend.allocBuffer cuda qSz
    -- Q8_1-quantise input.
    GPUBackend.execute cuda (quantizeQ8_1Kernel inDim)
      [("input", inBuf), ("output", qBuf)]
      { numWorkgroups := (nQ8, 1, 1), workgroupSize := { x := 32, y := 1, z := 1 }
        extensions := ["subgroups"] }
    let oE ← GPUBackend.allocBuffer cuda (4 * outDim).toUSize
    let oG ← GPUBackend.allocBuffer cuda (4 * outDim).toUSize
    GPUBackend.freeBuffer cuda inBuf
    wBufs := wBufs ++ [wBuf]
    qBufs := qBufs ++ [qBuf]
    outE  := outE ++ [oE]
    outG  := outG ++ [oG]
    prepEager := prepEager ++ [← GPUBackend.newCacheRef (β := Hesper.CUDAContext)]
    prepGraph := prepGraph ++ [← GPUBackend.newCacheRef (β := Hesper.CUDAContext)]

  -- Pair lists for easy iteration.
  let eagerLayers := wBufs.zip (qBufs.zip (outE.zip prepEager))
  let graphLayers := wBufs.zip (qBufs.zip (outG.zip prepGraph))

  -- ── Phase A: eager (no capture) — N back-to-back dispatches ──
  IO.println "── Phase A: eager dispatches ──"
  for ((w, (q, (o, pr))), idx) in eagerLayers.zipIdx do
    GPUBackend.executeWithConfigCached cuda
      (fusedQ6KLinearDP4A4WarpKernel inDim outDim)
      [("weights", w), ("input_q8", q), ("output", o)]
      { numWorkgroups := (outDim, 1, 1), workgroupSize := { x := 32, y := 4, z := 1 }
        extensions := ["subgroups"]
        funcName := s!"q6k_dp4a_4warp_{inDim}_{outDim}" }
      (hash ("q6k-4warp-eager", idx))
      pr
  -- Sync to make sure eager outputs are visible before graph phase.
  match outE with
  | first :: _ => let _ ← GPUBackend.readBuffer cuda first 4
  | []         => pure ()

  -- ── Phase B: capture N dispatches into a graph, then replay ──
  IO.println "── Phase B: graph capture + replay ──"
  let stream ← Hesper.CUDA.cuStreamCreateDefault
  Hesper.CUDA.cuStreamBeginCapture stream
  Hesper.cudaCaptureStream.set (some stream)
  try
    for ((w, (q, (o, pr))), idx) in graphLayers.zipIdx do
      GPUBackend.executeWithConfigCached cuda
        (fusedQ6KLinearDP4A4WarpKernel inDim outDim)
        [("weights", w), ("input_q8", q), ("output", o)]
        { numWorkgroups := (outDim, 1, 1), workgroupSize := { x := 32, y := 4, z := 1 }
          extensions := ["subgroups"]
          funcName := s!"q6k_dp4a_4warp_{inDim}_{outDim}" }
        (hash ("q6k-4warp-graph", idx))
        pr
  finally
    Hesper.cudaCaptureStream.set none
  let graph ← Hesper.CUDA.cuStreamEndCapture stream
  let exec ← Hesper.CUDA.cuGraphInstantiate graph
  Hesper.CUDA.cuGraphDestroy graph
  -- Launch the captured graph N=10 times.
  for _ in [0:10] do
    Hesper.CUDA.cuGraphLaunch exec stream
    Hesper.CUDA.cuStreamSynchronize stream
  Hesper.CUDA.cuGraphExecDestroy exec
  Hesper.CUDA.cuStreamDestroy stream

  -- ── Compare ──
  IO.println "── Compare eager vs graph ──"
  let mut totalMaxAbs : Float := 0.0
  let mut totalMaxRel : Float := 0.0
  let mut firstMismatch : Option (Nat × Nat × Float × Float) := none
  for ((eBuf, gBuf), layer) in (outE.zip outG).zipIdx do
    let bytesE ← GPUBackend.readBuffer cuda eBuf (4 * outDim).toUSize
    let bytesG ← GPUBackend.readBuffer cuda gBuf (4 * outDim).toUSize
    let mut layerMaxAbs : Float := 0.0
    let mut layerMaxRel : Float := 0.0
    for i in [0:outDim] do
      let e := unpackF32 bytesE i
      let g := unpackF32 bytesG i
      let absD := (e - g).abs
      let relD := if e.abs > 1e-8 then absD / e.abs else absD
      if absD > layerMaxAbs then layerMaxAbs := absD
      if relD > layerMaxRel then layerMaxRel := relD
      if firstMismatch.isNone && absD > 1e-3 then
        firstMismatch := some (layer, i, e, g)
    IO.println s!"  layer {layer}: max |abs|={layerMaxAbs}, max rel={layerMaxRel}, eager[0]={unpackF32 bytesE 0} graph[0]={unpackF32 bytesG 0}"
    if layerMaxAbs > totalMaxAbs then totalMaxAbs := layerMaxAbs
    if layerMaxRel > totalMaxRel then totalMaxRel := layerMaxRel

  IO.println s!"\n  total max |abs diff| = {totalMaxAbs}"
  IO.println s!"  total max  rel diff  = {totalMaxRel}"
  match firstMismatch with
  | some (layer, i, e, g) =>
      IO.println s!"  ✗ FAIL: layer {layer} row {i}: eager={e} graph={g}"
      IO.println "      Reproduction confirms graphs ON breaks 4-warp kernel."
      IO.Process.exit 1
  | none =>
      IO.println "  ✓ PASS — eager and graph match.  Bug is elsewhere."

  -- Cleanup
  for b in wBufs do GPUBackend.freeBuffer cuda b
  for b in qBufs do GPUBackend.freeBuffer cuda b
  for b in outE do GPUBackend.freeBuffer cuda b
  for b in outG do GPUBackend.freeBuffer cuda b
