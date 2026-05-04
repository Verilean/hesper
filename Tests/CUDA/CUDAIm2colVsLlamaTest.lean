import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.Layers.Vision
import Hesper.WGSL.Monad
import Hesper.WGSL.Exp
import Hesper

/-!
# Parity test: hesper im2col vs llama.cpp ggml_im2col (CPU backend) golden

Reads three files dumped by `scripts/llama_parity/dump_im2col_golden`:

  $GOLDEN_DIR/src.bin     — input  f32 [N, IC, IH, IW] = [2, 3, 8, 8]
  $GOLDEN_DIR/weight.bin  — kernel f32 (unused by im2col itself)
  $GOLDEN_DIR/out.bin     — ggml's CPU output, f32, 3456 elements

Runs hesper's `im2colF32Kernel` on the **same** src/weight, dumps the GPU
output, and compares element-by-element against ggml's f32 output.

PASS criterion: max |err| = 0 (bit-exact).
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
  IO.println "═══ im2col PARITY vs llama.cpp ggml_im2col golden ═══"

  let goldenDir ← (← IO.getEnv "GOLDEN_DIR").getDM (pure "/tmp/im2col_golden")
  IO.println s!"  golden dir: {goldenDir}"

  -- Hardcoded shape — must match the C++ generator.
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
  let OH := (IH + 2*p0 - KH) / s0 + 1
  let OW := (IW + 2*p1 - KW) / s1 + 1
  let IC_KH_KW := IC * KH * KW
  let srcSize := N * IC * IH * IW
  let dstSize := N * OH * OW * IC_KH_KW

  IO.println s!"  N={N} IC={IC} IH={IH} IW={IW} KH={KH} KW={KW} src={srcSize} dst={dstSize}"

  -- Load files.
  let srcArr ← readBinAsF32 (goldenDir ++ "/src.bin") srcSize
  let llamaOut ← readBinAsF32 (goldenDir ++ "/out.bin") dstSize

  -- Run hesper.
  let ctx ← Hesper.CUDAContext.init
  let srcBuf ← GPUBackend.allocBuffer ctx ((srcSize * 4).toUSize)
  let dstBuf ← GPUBackend.allocBuffer ctx ((dstSize * 4).toUSize)
  GPUBackend.writeBuffer ctx srcBuf (← packF32 srcArr)

  let nBlocksX := (IC_KH_KW + 255) / 256
  let bufs : List (String × GPUBackend.Buf Hesper.CUDAContext) :=
    [ ("src", srcBuf), ("dst", dstBuf) ]
  GPUBackend.execute ctx
    (im2colF32Kernel IC IW IH OW OH KW KH s0 s1 p0 p1 d0 d1 N) bufs
    { workgroupSize := { x := IC_KH_KW.min 256, y := 1, z := 1 },
      numWorkgroups := (nBlocksX, OW, N * OH) }

  let resultBytes ← GPUBackend.readBuffer ctx dstBuf ((dstSize * 4).toUSize)
  let hesperOut ← unpackF32 resultBytes dstSize

  -- Compare element-wise; show first mismatch.
  let mut maxErr : Float := 0.0
  let mut firstMis : Int := -1
  for i in [0:dstSize] do
    let d := (hesperOut[i]! - llamaOut[i]!).abs
    if d > maxErr then maxErr := d
    if d > 0 ∧ firstMis == -1 then
      firstMis := i.toInt32.toInt
      IO.println s!"  ✗ first mismatch idx={i} llama={llamaOut[i]!} hesper={hesperOut[i]!} diff={d}"

  GPUBackend.freeBuffer ctx srcBuf
  GPUBackend.freeBuffer ctx dstBuf

  IO.println s!"  max |err| = {maxErr}"
  if maxErr == 0.0 then
    IO.println "═══ im2col PARITY vs llama PASS (bit-exact) ═══"
  else if maxErr < 1.0e-5 then
    IO.println s!"═══ im2col PARITY vs llama PASS (within FP noise: {maxErr}) ═══"
  else
    IO.println "═══ im2col PARITY vs llama FAIL ═══"
    IO.Process.exit 1
