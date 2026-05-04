import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.Layers.Vision
import Hesper.WGSL.Monad
import Hesper.WGSL.Exp
import Hesper

/-!
# Parity test for conv2d via im2col + naive f32 matmul

Compares hesper's GPU output of `conv2d(src, w, bias=0)` (computed
as im2col → matmul) against a CPU reference. PASS criterion: max
|err| < 1e-4 (float-add ordering may shift bits slightly).

Shape (small to keep CPU reference manageable):
  N=2, IC=3, IH=8, IW=8
  OC=4 output channels
  KH=3, KW=3 kernel
  stride=1, pad=1, dilation=1
  → OH=8, OW=8

Matmul stage:
  A = im2col(src) : [N*OH*OW, IC*KH*KW] = [128, 27]
  B = weights    : [OC, IC*KH*KW]       = [4, 27]
  → out         : [N*OH*OW, OC]         = [128, 4]
  Reshape to NHWC: [N, OH, OW, OC] = [2, 8, 8, 4] = 512 floats.
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

/-- CPU reference conv2d.
    src  : [N, IC, IH, IW] (row-major)
    w    : [OC, IC, KH, KW]
    out  : [N, OC, OH, OW] (row-major) — note this layout!
-/
private def cpuConv2d
    (src w : Array Float)
    (N IC IH IW OC OH OW KH KW : Nat)
    (s0 s1 p0 p1 d0 d1 : Int) : Array Float := Id.run do
  let outSize := N * OC * OH * OW
  let mut out : Array Float := Array.replicate outSize 0.0
  for in_ in [0:N] do
    for ioc in [0:OC] do
      for ioh in [0:OH] do
        for iow in [0:OW] do
          let mut acc : Float := 0.0
          for iic in [0:IC] do
            for ikh in [0:KH] do
              for ikw in [0:KW] do
                let iiw : Int := iow.toInt32.toInt * s0 + ikw.toInt32.toInt * d0 - p0
                let iih : Int := ioh.toInt32.toInt * s1 + ikh.toInt32.toInt * d1 - p1
                if iiw ≥ 0 ∧ iiw < IW.toInt32.toInt ∧ iih ≥ 0 ∧ iih < IH.toInt32.toInt then
                  let srcIdx := in_ * (IC * IH * IW) + iic * (IH * IW) + iih.toNat * IW + iiw.toNat
                  let wIdx := ioc * (IC * KH * KW) + iic * (KH * KW) + ikh * KW + ikw
                  acc := acc + src[srcIdx]! * w[wIdx]!
          let outIdx := in_ * (OC * OH * OW) + ioc * (OH * OW) + ioh * OW + iow
          out := out.set! outIdx acc
  return out

unsafe def main : IO Unit := do
  IO.println "═══ conv2d via im2col + matmul parity test ═══"

  let N := 2
  let IC := 3
  let OC := 4
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
  let OH := (IH + 2*p0 - KH) / s0 + 1
  let OW := (IW + 2*p1 - KW) / s1 + 1
  let IC_KH_KW := IC * KH * KW
  let M := N * OH * OW          -- 128
  let K := IC_KH_KW             -- 27
  let Ncol := OC                -- 4
  let srcSize := N * IC * IH * IW   -- 384
  let wSize := OC * IC * KH * KW    -- 108
  let im2colSize := M * K           -- 3456
  let outSize := M * Ncol           -- 512  (NHWC layout)
  IO.println s!"  N={N} IC={IC} OC={OC} IH={IH} IW={IW} KH={KH} KW={KW}"
  IO.println s!"  → OH={OH} OW={OW} M={M} K={K} N={Ncol}"
  IO.println s!"  src={srcSize} w={wSize} im2col={im2colSize} out={outSize}"

  -- Deterministic input + weight tensors.
  let srcArr : Array Float :=
    (List.range srcSize).toArray.map (fun i => Float.sin (i.toFloat * 0.013) * 0.4)
  let wArr : Array Float :=
    (List.range wSize).toArray.map (fun i => Float.cos (i.toFloat * 0.027) * 0.3)

  -- 1. CPU reference (writes NCHW)
  let cpuOutNCHW := cpuConv2d srcArr wArr N IC IH IW OC OH OW KH KW
                      s0.toInt32.toInt s1.toInt32.toInt p0.toInt32.toInt p1.toInt32.toInt
                      d0.toInt32.toInt d1.toInt32.toInt

  -- 2. GPU pipeline: im2col → matmul → output is NHWC
  let ctx ← Hesper.CUDAContext.init

  let srcBuf  ← GPUBackend.allocBuffer ctx (srcSize * 4).toUSize
  let wBuf    ← GPUBackend.allocBuffer ctx (wSize * 4).toUSize
  let im2colBuf ← GPUBackend.allocBuffer ctx (im2colSize * 4).toUSize
  let outBuf  ← GPUBackend.allocBuffer ctx (outSize * 4).toUSize
  GPUBackend.writeBuffer ctx srcBuf  (← packF32 srcArr)
  GPUBackend.writeBuffer ctx wBuf    (← packF32 wArr)

  -- Stage 1: im2col
  let im2colBufs : List (String × GPUBackend.Buf Hesper.CUDAContext) :=
    [ ("src", srcBuf), ("dst", im2colBuf) ]
  let nBlocksX := (IC_KH_KW + 255) / 256
  GPUBackend.execute ctx
    (im2colF32Kernel IC IW IH OW OH KW KH s0 s1 p0 p1 d0 d1 N) im2colBufs
    { workgroupSize := { x := IC_KH_KW.min 256, y := 1, z := 1 },
      numWorkgroups := (nBlocksX, OW, N * OH) }

  -- Stage 2: matmul. Each WG = 1 thread emits 1 output element.
  -- Kernel reads `a` (im2col output, M×K), `b` (weights, N×K).
  let matmulBufs : List (String × GPUBackend.Buf Hesper.CUDAContext) :=
    [ ("a", im2colBuf), ("b", wBuf), ("dst", outBuf) ]
  GPUBackend.execute ctx
    (matmulF32NaiveKernel M Ncol K) matmulBufs
    { workgroupSize := { x := 1, y := 1, z := 1 },
      numWorkgroups := (Ncol, M, 1) }

  let resultBytes ← GPUBackend.readBuffer ctx outBuf (outSize * 4).toUSize
  let gpuOutNHWC ← unpackF32 resultBytes outSize

  -- 3. Reshape GPU NHWC [N, OH, OW, OC] → NCHW [N, OC, OH, OW] for compare
  -- gpuOutNHWC[((in*OH+ioh)*OW+iow)*OC + ioc] should match
  -- cpuOutNCHW [in*OC*OH*OW + ioc*OH*OW + ioh*OW + iow]
  let mut maxErr : Float := 0.0
  let mut firstErrIdx : Int := -1
  for in_ in [0:N] do
    for ioc in [0:OC] do
      for ioh in [0:OH] do
        for iow in [0:OW] do
          let nhwcIdx := ((in_ * OH + ioh) * OW + iow) * OC + ioc
          let nchwIdx := in_ * (OC * OH * OW) + ioc * (OH * OW) + ioh * OW + iow
          let cpuV := cpuOutNCHW[nchwIdx]!
          let gpuV := gpuOutNHWC[nhwcIdx]!
          let d := (cpuV - gpuV).abs
          if d > maxErr then maxErr := d
          if d > 1e-3 ∧ firstErrIdx == -1 then
            firstErrIdx := nhwcIdx.toInt32.toInt
            IO.println s!"  ✗ first mismatch (in={in_} ioc={ioc} ioh={ioh} iow={iow}): cpu={cpuV} gpu={gpuV}"

  GPUBackend.freeBuffer ctx srcBuf
  GPUBackend.freeBuffer ctx wBuf
  GPUBackend.freeBuffer ctx im2colBuf
  GPUBackend.freeBuffer ctx outBuf

  IO.println s!"  max |err| = {maxErr}"
  if maxErr < 1.0e-4 then
    IO.println "═══ conv2d PARITY PASS (im2col + matmul) ═══"
  else
    IO.println "═══ conv2d PARITY FAIL ═══"
    IO.Process.exit 1
