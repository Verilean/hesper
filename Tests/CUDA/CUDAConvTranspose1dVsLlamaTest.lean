import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.Layers.Audio
import Hesper.WGSL.Monad
import Hesper.WGSL.Exp
import Hesper

/-!
# Parity test: hesper conv_transpose_1d vs llama.cpp ggml_conv_transpose_1d

Reads files dumped by `scripts/llama_parity/dump_conv_transpose_1d_golden`:

  $GOLDEN_DIR/src.bin     — input  f32 [IC=4, IL=8] = 32 elem
  $GOLDEN_DIR/weight.bin  — kernel f32 [IC=4, OC=3, KW=4] = 48 elem (ggml ne order)
  $GOLDEN_DIR/out.bin     — ggml's CPU output f32 [OC=3, OL=18] = 54 elem

PASS: max |err| = 0.
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

private def readBinAsF32 (path : String) (n : Nat) : IO (Array Float) := do
  let bytes ← IO.FS.readBinFile path
  if bytes.size != n * 4 then
    throw <| IO.userError s!"file {path}: expected {n*4} bytes, got {bytes.size}"
  unpackF32 bytes n

unsafe def main : IO Unit := do
  IO.println "═══ conv_transpose_1d PARITY vs llama.cpp golden ═══"

  let goldenDir ← (← IO.getEnv "GOLDEN_DIR").getDM (pure "/tmp/conv_transpose_1d_golden")
  IO.println s!"  golden dir: {goldenDir}"

  let IC := 4
  let OC := 3
  let KW := 4
  let IL := 8
  let s0 := 2
  let OL := (IL - 1) * s0 + KW   -- 18
  let srcSize := IC * IL          -- 32
  let wSize := IC * OC * KW       -- 48
  let outSize := OC * OL          -- 54

  let srcArr ← readBinAsF32 (goldenDir ++ "/src.bin") srcSize
  let wArr   ← readBinAsF32 (goldenDir ++ "/weight.bin") wSize
  let llamaOut ← readBinAsF32 (goldenDir ++ "/out.bin") outSize

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
  let hesperOut ← unpackF32 resultBytes outSize

  let mut maxErr : Float := 0.0
  let mut firstMis : Int := -1
  for i in [0:outSize] do
    let d := (hesperOut[i]! - llamaOut[i]!).abs
    if d > maxErr then maxErr := d
    if d > 0 ∧ firstMis == -1 then
      firstMis := i.toInt32.toInt
      IO.println s!"  ✗ first mismatch idx={i} llama={llamaOut[i]!} hesper={hesperOut[i]!} diff={d}"

  GPUBackend.freeBuffer ctx srcBuf
  GPUBackend.freeBuffer ctx wBuf
  GPUBackend.freeBuffer ctx dstBuf

  IO.println s!"  max |err| = {maxErr}"
  if maxErr == 0.0 then
    IO.println "═══ conv_transpose_1d PARITY vs llama PASS (bit-exact) ═══"
  else if maxErr < 1.0e-5 then
    IO.println s!"═══ conv_transpose_1d PARITY vs llama PASS (within FP noise: {maxErr}) ═══"
  else
    IO.println "═══ conv_transpose_1d PARITY vs llama FAIL ═══"
    IO.Process.exit 1
