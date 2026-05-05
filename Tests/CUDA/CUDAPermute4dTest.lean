import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.Layers.Vision
import Hesper.WGSL.Monad
import Hesper.WGSL.Exp
import Hesper

/-!
# Parity test: hesper permute_4d (axes 0,2,1,3) vs llama.cpp ggml_permute + ggml_cont
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
  IO.println "═══ permute_4d (0,2,1,3) PARITY vs llama.cpp golden ═══"

  let goldenDir ← (← IO.getEnv "GOLDEN_DIR").getDM (pure "/tmp/permute_4d_golden")
  let s0 := 64
  let s1 := 12
  let s2 := 64
  let s3 := 1
  let total := s0 * s1 * s2 * s3

  let srcArr ← readBinAsF32 (goldenDir ++ "/src.bin") total
  let llamaOut ← readBinAsF32 (goldenDir ++ "/out.bin") total

  let ctx ← Hesper.CUDAContext.init
  let srcBuf ← GPUBackend.allocBuffer ctx (total * 4).toUSize
  let dstBuf ← GPUBackend.allocBuffer ctx (total * 4).toUSize
  GPUBackend.writeBuffer ctx srcBuf (← packF32 srcArr)

  let nBlocks := (total + 255) / 256
  let bufs : List (String × GPUBackend.Buf Hesper.CUDAContext) :=
    [ ("src", srcBuf), ("dst", dstBuf) ]
  GPUBackend.execute ctx
    (permute_4d_f32_kernel s0 s1 s2 s3 0 2 1 3) bufs
    { workgroupSize := { x := 256, y := 1, z := 1 },
      numWorkgroups := (nBlocks, 1, 1) }

  let resultBytes ← GPUBackend.readBuffer ctx dstBuf (total * 4).toUSize
  let hesperOut ← unpackF32 resultBytes total

  let mut maxErr : Float := 0.0
  let mut firstMis : Int := -1
  for i in [0:total] do
    let d := (hesperOut[i]! - llamaOut[i]!).abs
    if d > maxErr then maxErr := d
    if d > 0 ∧ firstMis == -1 then
      firstMis := i.toInt32.toInt
      IO.println s!"  ✗ first mismatch idx={i} llama={llamaOut[i]!} hesper={hesperOut[i]!} diff={d}"

  GPUBackend.freeBuffer ctx srcBuf
  GPUBackend.freeBuffer ctx dstBuf

  IO.println s!"  max |err| = {maxErr}"
  if maxErr == 0.0 then
    IO.println "═══ permute_4d PARITY PASS (bit-exact) ═══"
  else
    IO.println "═══ permute_4d PARITY FAIL ═══"
    IO.Process.exit 1
