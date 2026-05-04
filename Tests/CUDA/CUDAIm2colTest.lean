import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.Layers.Vision
import Hesper.WGSL.Monad
import Hesper.WGSL.Exp
import Hesper

/-!
# Parity test for `im2colF32Kernel`

Compares hesper's GPU im2col output against a CPU reference implementation
on a small synthetic input. PASS criterion: every element bit-equal
(max |err| = 0).

Shape (deliberately small to keep CPU reference fast):
  N=2, IC=3, IH=8, IW=8
  KH=3, KW=3
  stride=1, pad=1, dilation=1
  → OH = (8 + 2*1 - 3) / 1 + 1 = 8
  → OW = 8

Output: [N=2, OH=8, OW=8, IC*KH*KW=27] = 3,456 floats.
-/

open Hesper
open Hesper.WGSL.Monad (ShaderM)
open Hesper.WGSL (Exp)
open Hesper.Layers.Vision

private def packF32 (arr : Array Float) : IO ByteArray := do
  Hesper.Basic.floatArrayToBytes arr

private def unpackF32 (ba : ByteArray) (n : Nat) : IO (Array Float) := do
  let mut arr := Array.mkEmpty n
  for i in [0:n] do
    let f ← Hesper.Basic.bytesToFloat32 ba (i * 4)
    arr := arr.push f
  return arr

/-- CPU reference im2col: same algorithm as the GPU kernel.
    src: [N, IC, IH, IW]   (row-major)
    dst: [N, OH, OW, IC*KH*KW] (row-major)
    Returns dst as a flat Array Float. -/
private def cpuIm2col
    (src : Array Float)
    (N IC IH IW OH OW KH KW : Nat)
    (s0 s1 p0 p1 d0 d1 : Int) : Array Float := Id.run do
  let IC_KH_KW := IC * KH * KW
  let dstSize := N * OH * OW * IC_KH_KW
  let mut dst : Array Float := Array.replicate dstSize 0.0
  for in_ in [0:N] do
    for iic in [0:IC] do
      for ioh in [0:OH] do
        for iow in [0:OW] do
          for ikh in [0:KH] do
            for ikw in [0:KW] do
              let iiw : Int := (iow.toInt32.toInt) * s0 + (ikw.toInt32.toInt) * d0 - p0
              let iih : Int := (ioh.toInt32.toInt) * s1 + (ikh.toInt32.toInt) * d1 - p1
              let dstIdx := ((in_ * OH + ioh) * OW + iow) * IC_KH_KW + iic * (KH * KW) + ikh * KW + ikw
              if iiw < 0 ∨ iiw ≥ IW.toInt32.toInt ∨ iih < 0 ∨ iih ≥ IH.toInt32.toInt then
                dst := dst.set! dstIdx 0.0
              else
                let srcIdx := in_ * (IC * IH * IW) + iic * (IH * IW) + iih.toNat * IW + iiw.toNat
                dst := dst.set! dstIdx src[srcIdx]!
  return dst

unsafe def main : IO Unit := do
  IO.println "═══ im2col parity test ═══"

  -- Shape
  let N := 2
  let IC := 3
  let IH := 8
  let IW := 8
  let KH := 3
  let KW := 3
  let s0 := 1
  let s1 := 1
  let p0 := 1
  let p1 := 1
  let d0 := 1
  let d1 := 1
  let OH := (IH + 2*p0 - KH) / s0 + 1   -- 8
  let OW := (IW + 2*p1 - KW) / s1 + 1   -- 8
  let IC_KH_KW := IC * KH * KW          -- 27
  let srcSize := N * IC * IH * IW       -- 384
  let dstSize := N * OH * OW * IC_KH_KW -- 3456
  IO.println s!"  N={N} IC={IC} IH={IH} IW={IW} KH={KH} KW={KW}"
  IO.println s!"  → OH={OH} OW={OW} dst={dstSize} elems"

  -- Deterministic input: src[i] = sin(i * 0.013) * 0.4
  let srcArr : Array Float :=
    (List.range srcSize).toArray.map (fun i => Float.sin (i.toFloat * 0.013) * 0.4)

  -- 1. PTX dump check
  let ptx := Hesper.CUDA.CodeGen.generatePTX "im2col_test"
               { x := IC_KH_KW, y := 1, z := 1 }
               (im2colF32Kernel IC IW IH OW OH KW KH s0 s1 p0 p1 d0 d1 N)
  IO.FS.writeFile "/tmp/im2col.ptx" ptx
  IO.println s!"  (PTX written to /tmp/im2col.ptx, {ptx.length} bytes)"
  if (ptx.splitOn ".entry").length < 2 then
    IO.println "✗ PTX has no .entry"; IO.Process.exit 1
  IO.println "✓ PTX has .entry"

  -- 2. CPU reference
  let cpuOut := cpuIm2col srcArr N IC IH IW OH OW KH KW
                  s0.toInt32.toInt s1.toInt32.toInt p0.toInt32.toInt p1.toInt32.toInt d0.toInt32.toInt d1.toInt32.toInt

  -- 3. GPU run
  let ctx ← Hesper.CUDAContext.init
  let srcBuf ← GPUBackend.allocBuffer ctx ((srcSize * 4).toUSize)
  let dstBuf ← GPUBackend.allocBuffer ctx ((dstSize * 4).toUSize)
  let srcBytes ← packF32 srcArr
  GPUBackend.writeBuffer ctx srcBuf srcBytes

  -- Grid: x = ceil(IC*KH*KW / 256), y = OW, z = N*OH
  let nBlocksX := (IC_KH_KW + 255) / 256
  let bufs : List (String × GPUBackend.Buf Hesper.CUDAContext) :=
    [ ("src", srcBuf), ("dst", dstBuf) ]
  GPUBackend.execute ctx
    (im2colF32Kernel IC IW IH OW OH KW KH s0 s1 p0 p1 d0 d1 N) bufs
    { workgroupSize := { x := IC_KH_KW.min 256, y := 1, z := 1 },
      numWorkgroups := (nBlocksX, OW, N * OH) }

  let resultBytes ← GPUBackend.readBuffer ctx dstBuf ((dstSize * 4).toUSize)
  let gpuOut ← unpackF32 resultBytes dstSize

  -- 4. Compare
  let mut maxErr : Float := 0.0
  let mut firstErrIdx : Int := -1
  for i in [0:dstSize] do
    let d := (gpuOut[i]! - cpuOut[i]!).abs
    if d > maxErr then maxErr := d
    if d > 1e-5 ∧ firstErrIdx == -1 then
      firstErrIdx := i.toInt32.toInt
      IO.println s!"  ✗ first mismatch idx={i} cpu={cpuOut[i]!} gpu={gpuOut[i]!}"

  GPUBackend.freeBuffer ctx srcBuf
  GPUBackend.freeBuffer ctx dstBuf

  IO.println s!"  max |err| = {maxErr}"
  if maxErr < 1.0e-5 then
    IO.println "═══ im2col PARITY PASS ═══"
  else
    IO.println "═══ im2col PARITY FAIL ═══"
    IO.Process.exit 1
