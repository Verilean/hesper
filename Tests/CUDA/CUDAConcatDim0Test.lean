import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.Layers.Vision
import Hesper.WGSL.Monad
import Hesper.WGSL.Exp
import Hesper

/-!
# Parity test: hesper concat_dim0 vs llama.cpp ggml_concat

PASS criterion: max |err| = 0 (memory-copy operation, no FP math).
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

private def readBinAsF32 (path : String) (n : Nat) : IO (Array Float) := do
  let bytes ← IO.FS.readBinFile path
  if bytes.size != n * 4 then
    throw <| IO.userError s!"file {path}: expected {n*4} bytes, got {bytes.size}"
  unpackF32 bytes n

unsafe def main : IO Unit := do
  IO.println "═══ concat_dim0 PARITY vs llama.cpp golden ═══"

  let goldenDir ← (← IO.getEnv "GOLDEN_DIR").getDM (pure "/tmp/concat_dim0_golden")
  let ne00 := 32
  let ne10 := 32
  let ne1 := 12
  let ne2 := 64
  let xSize := ne00 * ne1 * ne2
  let ySize := ne10 * ne1 * ne2
  let outSize := (ne00 + ne10) * ne1 * ne2

  let xArr ← readBinAsF32 (goldenDir ++ "/x.bin") xSize
  let yArr ← readBinAsF32 (goldenDir ++ "/y.bin") ySize
  let llamaOut ← readBinAsF32 (goldenDir ++ "/out.bin") outSize

  let ctx ← Hesper.CUDAContext.init
  let xBuf ← GPUBackend.allocBuffer ctx (xSize * 4).toUSize
  let yBuf ← GPUBackend.allocBuffer ctx (ySize * 4).toUSize
  let dstBuf ← GPUBackend.allocBuffer ctx (outSize * 4).toUSize
  GPUBackend.writeBuffer ctx xBuf (← packF32 xArr)
  GPUBackend.writeBuffer ctx yBuf (← packF32 yArr)

  let ne0 := ne00 + ne10
  let nBlocksX := (ne0 + 255) / 256
  let bufs : List (String × GPUBackend.Buf Hesper.CUDAContext) :=
    [ ("x", xBuf), ("y", yBuf), ("dst", dstBuf) ]
  GPUBackend.execute ctx
    (concat_dim0_f32_kernel ne00 ne10 ne1 ne2) bufs
    { workgroupSize := { x := 256, y := 1, z := 1 },
      numWorkgroups := (nBlocksX, ne1, ne2) }

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

  GPUBackend.freeBuffer ctx xBuf
  GPUBackend.freeBuffer ctx yBuf
  GPUBackend.freeBuffer ctx dstBuf

  IO.println s!"  max |err| = {maxErr}"
  if maxErr == 0.0 then
    IO.println "═══ concat_dim0 PARITY PASS (bit-exact) ═══"
  else
    IO.println "═══ concat_dim0 PARITY FAIL ═══"
    IO.Process.exit 1
