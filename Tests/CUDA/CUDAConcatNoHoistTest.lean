import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.WGSL.Monad
import Hesper.WGSL.Exp
import Hesper

/-!
# Regression sentinel: concat_dim0 with offsets INSIDE branches (no let' hoist)

This is the original natural form of `concat_dim0_f32_kernel` — offsets
computed inside each `if_` branch instead of hoisted to `let'` before
the branch.  Before the codegen scoped-cache fix (commit XXX), this
form produced `CUDA_ERROR_INVALID_ADDRESS_SPACE` because the sreg /
imm / exp caches leaked across branches: a register first emitted
inside `thenBody` was reused inside `elseBody`, but it's only
assigned on threads that took the then path.

After the fix (snapshot/restore all 3 caches around `ifStmt` branches),
this form is bit-exact correct.

Keep this test as a regression sentinel: if it ever fails again, the
codegen-side cache scoping has regressed.
-/

open Hesper
open Hesper.WGSL.Monad (ShaderM)
open Hesper.WGSL (Exp)

private def concatNoHoistKernel (ne00 ne10 ne1 ne2 : Nat) : ShaderM Unit := do
  let ne0 := ne00 + ne10
  let _x ← ShaderM.declareReadOnlyBuffer "x" (.array (.scalar .f32) (ne00 * ne1 * ne2))
  let _y ← ShaderM.declareReadOnlyBuffer "y" (.array (.scalar .f32) (ne10 * ne1 * ne2))
  let _dst ← ShaderM.declareOutputBuffer  "dst" (.array (.scalar .f32) (ne0 * ne1 * ne2))

  let lid ← ShaderM.localId
  let wid ← ShaderM.workgroupId
  let bx := Exp.vec3X wid
  let by_ := Exp.vec3Y wid
  let bz := Exp.vec3Z wid
  let tid := Exp.vec3X lid
  let nidx := Exp.add (Exp.mul bx (Exp.litU32 256)) tid
  let inBounds := Exp.lt nidx (Exp.litU32 ne0)
  ShaderM.if_ inBounds (do
    let offDst :=
      Exp.add (Exp.add nidx (Exp.mul by_ (Exp.litU32 ne0)))
              (Exp.mul bz (Exp.litU32 (ne0 * ne1)))
    let fromX := Exp.lt nidx (Exp.litU32 ne00)
    ShaderM.if_ fromX (do
      let offSrc :=
        Exp.add (Exp.add nidx (Exp.mul by_ (Exp.litU32 ne00)))
                (Exp.mul bz (Exp.litU32 (ne00 * ne1)))
      let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := ne00 * ne1 * ne2) "x" offSrc
      ShaderM.writeBuffer (ty := .scalar .f32) "dst" offDst v
    ) (do
      let nIdxAdj := Exp.sub nidx (Exp.litU32 ne00)
      let offSrc :=
        Exp.add (Exp.add nIdxAdj (Exp.mul by_ (Exp.litU32 ne10)))
                (Exp.mul bz (Exp.litU32 (ne10 * ne1)))
      let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := ne10 * ne1 * ne2) "y" offSrc
      ShaderM.writeBuffer (ty := .scalar .f32) "dst" offDst v
    )
  ) (pure ())

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

/-- Sentinel test: the natural offsets-inside-branches form of concat must
    produce bit-exact output vs llama.cpp.  This relies on the codegen-side
    scoped-cache fix (snapshot/restore sregCache/immCache/expCache around
    ifStmt branches).  If this test fails again, the cache scoping has
    regressed. -/
unsafe def main : IO Unit := do
  IO.println "═══ concat_dim0 no-hoist (sentinel) vs llama.cpp ═══"
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
    (concatNoHoistKernel ne00 ne10 ne1 ne2) bufs
    { workgroupSize := { x := 256, y := 1, z := 1 },
      numWorkgroups := (nBlocksX, ne1, ne2) }

  let resultBytes ← GPUBackend.readBuffer ctx dstBuf (outSize * 4).toUSize
  let hesperOut ← unpackF32 resultBytes outSize

  let mut maxErr : Float := 0.0
  let mut firstMis : Int := -1
  let mut nMis := 0
  for i in [0:outSize] do
    let d := (hesperOut[i]! - llamaOut[i]!).abs
    if d > maxErr then maxErr := d
    if d > 1e-5 then
      nMis := nMis + 1
      if firstMis == -1 then
        firstMis := i.toInt32.toInt
        IO.println s!"  ✗ first mismatch idx={i} llama={llamaOut[i]!} hesper={hesperOut[i]!} diff={d}"

  GPUBackend.freeBuffer ctx xBuf
  GPUBackend.freeBuffer ctx yBuf
  GPUBackend.freeBuffer ctx dstBuf

  IO.println s!"  max |err| = {maxErr}, mismatched = {nMis}/{outSize}"
  if maxErr < 1e-5 then
    IO.println "═══ concat_dim0 no-hoist sentinel PASS ═══"
  else
    IO.println "═══ concat_dim0 no-hoist sentinel FAIL — codegen cache scoping regressed ═══"
    IO.Process.exit 1
