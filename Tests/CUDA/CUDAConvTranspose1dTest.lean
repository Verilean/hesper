import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.Layers.Audio
import Hesper.WGSL.Monad
import Hesper.WGSL.Exp
import Hesper

/-!
# Parity test for `convTranspose1dF32Kernel`

Compares hesper's GPU 1-D conv-transpose output against a CPU reference
on a synthetic input. PASS criterion: max |err| < 1e-4 (float-add
ordering may shift bits slightly).

Shape (small to keep CPU reference manageable):
  IC=4, OC=3, KW=4
  IL=8, stride=2 → OL = (IL-1)*s0 + KW = 18

  weights : [IC, OC, KW] = [4, 3, 4] = 48 floats
  input   : [IC, IL]     = [4, 8]    = 32 floats
  output  : [OC, OL]     = [3, 18]   = 54 floats
-/

open Hesper
open Hesper.WGSL.Monad (ShaderM)
open Hesper.WGSL (Exp)
open Hesper.Layers.Audio

private def packF32 (arr : Array Float) : IO ByteArray := do
  Hesper.Basic.floatArrayToBytes arr

private def unpackF32 (ba : ByteArray) (n : Nat) : IO (Array Float) := do
  let mut arr := Array.mkEmpty n
  for i in [0:n] do
    let f ← Hesper.Basic.bytesToFloat32 ba (i * 4)
    arr := arr.push f
  return arr

/-- CPU reference, mirrors llama.cpp's `conv_transpose_1d_kernel`.
    src1 : [IC, IL]    (row-major)
    src0 : [IC, OC, KW]
    dst  : [OC, OL]
    Returns dst as a flat `Array Float`. -/
private def cpuConvTranspose1d
    (src1 src0 : Array Float)
    (IC OC KW IL OL : Nat) (s0 : Nat) : Array Float := Id.run do
  let outSize := OC * OL
  let mut dst : Array Float := Array.replicate outSize 0.0
  for g in [0:outSize] do
    let outChan := g / OL
    let idx     := g % OL
    let mut acc : Float := 0.0
    for c in [0:IC] do
      for i in [0:IL] do
        let lo := i * s0
        let hi := i * s0 + KW
        if idx ≥ lo ∧ idx < hi then
          let wIdx := idx - i * s0
          let wOff := c * (OC * KW) + outChan * KW + wIdx
          let sOff := c * IL + i
          acc := acc + src0[wOff]! * src1[sOff]!
    dst := dst.set! g acc
  return dst

unsafe def main : IO Unit := do
  IO.println "═══ conv-transpose-1d parity test ═══"

  let IC := 4
  let OC := 3
  let KW := 4
  let IL := 8
  let s0 := 2
  let OL := (IL - 1) * s0 + KW   -- 7*2 + 4 = 18
  let srcSize := IC * IL          -- 32
  let wSize := IC * OC * KW       -- 48
  let outSize := OC * OL          -- 54
  IO.println s!"  IC={IC} OC={OC} KW={KW} IL={IL} OL={OL} s0={s0}"
  IO.println s!"  src={srcSize} w={wSize} out={outSize}"

  -- Deterministic input + weight tensors.
  let srcArr : Array Float :=
    (List.range srcSize).toArray.map (fun i => Float.sin (i.toFloat * 0.027) * 0.4)
  let wArr : Array Float :=
    (List.range wSize).toArray.map (fun i => Float.cos (i.toFloat * 0.039) * 0.3)

  -- 1. CPU reference
  let cpuOut := cpuConvTranspose1d srcArr wArr IC OC KW IL OL s0

  -- 2. GPU run
  let ctx ← Hesper.CUDAContext.init
  let srcBuf ← GPUBackend.allocBuffer ctx (srcSize * 4).toUSize
  let wBuf   ← GPUBackend.allocBuffer ctx (wSize * 4).toUSize
  let dstBuf ← GPUBackend.allocBuffer ctx (outSize * 4).toUSize
  GPUBackend.writeBuffer ctx srcBuf (← packF32 srcArr)
  GPUBackend.writeBuffer ctx wBuf   (← packF32 wArr)

  let nBlocks := (outSize + 255) / 256
  let bufs : List (String × GPUBackend.Buf Hesper.CUDAContext) :=
    [ ("src", srcBuf), ("w", wBuf), ("dst", dstBuf) ]
  GPUBackend.execute ctx
    (convTranspose1dF32Kernel IC OC KW IL OL s0) bufs
    { workgroupSize := { x := 256, y := 1, z := 1 },
      numWorkgroups := (nBlocks, 1, 1) }

  let resultBytes ← GPUBackend.readBuffer ctx dstBuf (outSize * 4).toUSize
  let gpuOut ← unpackF32 resultBytes outSize

  -- 3. Compare
  let mut maxErr : Float := 0.0
  let mut firstErrIdx : Int := -1
  for g in [0:outSize] do
    let d := (gpuOut[g]! - cpuOut[g]!).abs
    if d > maxErr then maxErr := d
    if d > 1e-4 ∧ firstErrIdx == -1 then
      firstErrIdx := g.toInt32.toInt
      IO.println s!"  ✗ first mismatch idx={g} cpu={cpuOut[g]!} gpu={gpuOut[g]!}"

  GPUBackend.freeBuffer ctx srcBuf
  GPUBackend.freeBuffer ctx wBuf
  GPUBackend.freeBuffer ctx dstBuf

  IO.println s!"  max |err| = {maxErr}"
  if maxErr < 1.0e-4 then
    IO.println "═══ conv-transpose-1d PARITY PASS ═══"
  else
    IO.println "═══ conv-transpose-1d PARITY FAIL ═══"
    IO.Process.exit 1
