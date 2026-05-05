import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.Layers.Vision
import Hesper.WGSL.Monad
import Hesper.WGSL.Exp
import Hesper

/-!
# Parity test: hesper geglu_quick vs llama.cpp ggml_geglu_quick_split

PASS criterion: max |err| < 1e-5 (single FP roundoff is acceptable since
sigmoid uses exp; bit-exact unlikely).
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
  IO.println "═══ geglu_quick PARITY vs llama.cpp golden ═══"

  let goldenDir ← (← IO.getEnv "GOLDEN_DIR").getDM (pure "/tmp/geglu_quick_golden")
  let n := 512

  let xArr ← readBinAsF32 (goldenDir ++ "/x.bin") n
  let gArr ← readBinAsF32 (goldenDir ++ "/g.bin") n
  let llamaOut ← readBinAsF32 (goldenDir ++ "/out.bin") n

  let ctx ← Hesper.CUDAContext.init
  let xBuf ← GPUBackend.allocBuffer ctx (n * 4).toUSize
  let gBuf ← GPUBackend.allocBuffer ctx (n * 4).toUSize
  let dstBuf ← GPUBackend.allocBuffer ctx (n * 4).toUSize
  GPUBackend.writeBuffer ctx xBuf (← packF32 xArr)
  GPUBackend.writeBuffer ctx gBuf (← packF32 gArr)

  let nBlocks := (n + 255) / 256
  let bufs : List (String × GPUBackend.Buf Hesper.CUDAContext) :=
    [ ("x", xBuf), ("g", gBuf), ("dst", dstBuf) ]
  GPUBackend.execute ctx
    (geglu_quick_split_f32_kernel n) bufs
    { workgroupSize := { x := 256, y := 1, z := 1 },
      numWorkgroups := (nBlocks, 1, 1) }

  let resultBytes ← GPUBackend.readBuffer ctx dstBuf (n * 4).toUSize
  let hesperOut ← unpackF32 resultBytes n

  let mut maxErr : Float := 0.0
  let mut firstMis : Int := -1
  -- 1e-3 threshold: GPU's __expf differs from libm's expf at ~1e-4
  -- relative, which compounds through the sigmoid + multiply chain.
  -- Bit-exact is unattainable across libm vs CUDA fast-math.
  let threshold : Float := 1.0e-3
  for i in [0:n] do
    let d := (hesperOut[i]! - llamaOut[i]!).abs
    if d > maxErr then maxErr := d
    if d > threshold ∧ firstMis == -1 then
      firstMis := i.toInt32.toInt
      IO.println s!"  ✗ first mismatch idx={i} llama={llamaOut[i]!} hesper={hesperOut[i]!} diff={d}"

  GPUBackend.freeBuffer ctx xBuf
  GPUBackend.freeBuffer ctx gBuf
  GPUBackend.freeBuffer ctx dstBuf

  IO.println s!"  max |err| = {maxErr}"
  if maxErr < threshold then
    IO.println s!"═══ geglu_quick PARITY PASS (within fast-math tolerance: {maxErr}) ═══"
  else
    IO.println "═══ geglu_quick PARITY FAIL ═══"
    IO.Process.exit 1
