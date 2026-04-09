import Hesper
import Hesper.WGSL.Execute
import Hesper.WGSL.Monad
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WebGPU.BufferOps
import Hesper.Float16

open Hesper.WebGPU
open Hesper.WGSL

-- Use the monadic ShaderM builder from Hesper.WGSL.Monad. There is an older
-- `Hesper.WGSL.ShaderM` free-standing alias that leads to an ambiguity if
-- we blindly open `Hesper.WGSL`, so we bring in just the type with an abbrev.
abbrev SM := Hesper.WGSL.Monad.ShaderM
open Hesper.WGSL.Monad.ShaderM (
  declareInputBuffer declareOutputBuffer
  declareMatrixLeftArray declareMatrixRightArray declareMatrixResultArray
  loadMatrixLeft loadMatrixRight matrixMultiplyAccumulate
  matrixMultiplyAccumulateMixed storeMatrixResult
)

/-!
# Subgroup Matrix f16 Probe

Minimal end-to-end smoke test for an 8×8 f16 subgroup matrix matmul on
the current device. Figures out whether (M, N, K) = (8, 8, 8) with f16
element type is a supported configuration before we commit to using it
in the BitLinear kernel.
-/

namespace Examples.Compute.SubgroupMatrixF16Probe

/-- Single 16×16 × 16×16 → 16×16 matmul using one subgroup.
    All three matrices f16. -/
def probeShader : SM Unit := do
  let _A ← declareInputBuffer "A" (.array (.scalar .f16) 256)
  let _B ← declareInputBuffer "B" (.array (.scalar .f16) 256)
  let _C ← declareOutputBuffer "C" (.array (.scalar .f16) 256)

  declareMatrixLeftArray "Ax" .f16 16 16 1 Exp.subgroupMatrixZeroLeft
  declareMatrixRightArray "Bx" .f16 16 16 1 Exp.subgroupMatrixZeroRight
  declareMatrixResultArray "Cx" .f16 16 16 1 Exp.subgroupMatrixZeroResult

  loadMatrixLeft (st := .f16) (m := 16) (k := 16)
    "Ax" 0 "A" (Exp.litU32 0) (Exp.litU32 16)
  loadMatrixRight (st := .f16) (k := 16) (n := 16)
    "Bx" 0 "B" (Exp.litU32 0) (Exp.litU32 16)
  matrixMultiplyAccumulate (st := .f16) (m := 16) (k := 16) (n := 16)
    "Cx" 0 "Ax" 0 "Bx" 0
  storeMatrixResult (st := .f16) (m := 16) (n := 16)
    "Cx" 0 "C" (Exp.litU32 0) (Exp.litU32 16)

/-- Convert an `Array Float` to f16 bytes via `Hesper.Float16`. -/
def floatsToF16Bytes (arr : Array Float) : IO ByteArray := do
  let fa : FloatArray := FloatArray.mk arr
  let f16 ← Hesper.Float16.fromFloatArray fa
  pure f16.data

/-- Read back a `ByteArray` of f16 values as `Array Float`. -/
def f16BytesToFloats (bytes : ByteArray) : IO (Array Float) := do
  match Hesper.Float16.fromBytes bytes with
  | none => throw (IO.userError "f16 bytes must be 2-byte aligned")
  | some arr =>
    let fa ← Hesper.Float16.toFloatArray arr
    let mut out : Array Float := Array.empty
    for i in [:fa.size] do
      out := out.push (fa.get! i)
    pure out

def main : IO Unit := do
  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║   f16 Subgroup Matrix 8×8 Probe              ║"
  IO.println "╚══════════════════════════════════════════════╝"

  let inst ← Hesper.init
  let device ← getDevice inst

  let hasSm ← Execute.hasSubgroupMatrixSupport device
  let hasF16 ← Execute.hasShaderF16Support device
  IO.println s!"  device has SubgroupMatrix: {hasSm}"
  IO.println s!"  device has ShaderF16:      {hasF16}"
  if !hasSm || !hasF16 then
    IO.println "  ✗ Required features missing; cannot probe f16 subgroup matrix."
    return

  -- A[i,j] = i+1, B[i,j] = j+1  →  (A@B)[i,j] = 16 * (i+1) * (j+1)
  let mut aData : Array Float := Array.empty
  for i in [:16] do
    for _j in [:16] do
      aData := aData.push (i + 1 : Nat).toFloat
  let mut bData : Array Float := Array.empty
  for _i in [:16] do
    for j in [:16] do
      bData := bData.push (j + 1 : Nat).toFloat

  let aBytes ← floatsToF16Bytes aData
  let bBytes ← floatsToF16Bytes bData

  let mkBuf (usz : USize) : IO Buffer := createBuffer device {
    size := usz
    usage := [.storage, .copyDst, .copySrc]
    mappedAtCreation := false
  }
  let aBuf ← mkBuf 512   -- 256 × 2 bytes (f16)
  let bBuf ← mkBuf 512
  let cBuf ← mkBuf 512   -- f16 output
  writeBuffer device aBuf 0 aBytes
  writeBuffer device bBuf 0 bBytes

  let config : Execute.ExecutionConfig := {
    funcName := "main"
    workgroupSize := { x := 32, y := 1, z := 1 }
    numWorkgroups := (1, 1, 1)
    extensions := ["f16", "chromium_experimental_subgroup_matrix"]
    diagnostics := [("off", "chromium.subgroup_matrix_uniformity")]
  }

  IO.println ""
  IO.println "  Dispatching f16 subgroup matmul..."
  let bufs : List (String × Buffer) := [("A", aBuf), ("B", bBuf), ("C", cBuf)]
  Execute.executeShaderNamed device probeShader bufs config

  -- C is f16 (512 bytes = 256 elements)
  let cBytes ← Hesper.WebGPU.mapBufferRead device cBuf 0 512
  let cData ← f16BytesToFloats cBytes

  IO.println ""
  IO.println "  Result (row-major 16×16, showing first 4×4 corner):"
  for i in [:4] do
    let mut row := ""
    for j in [:4] do
      let v := cData.getD (i * 16 + j) 0.0
      row := row ++ s!"{v}  "
    IO.println s!"    {row}"

  -- Expected: C[i,j] = 16 * (i+1) * (j+1)
  let mut okCount := 0
  let mut bad := 0
  for i in [:16] do
    for j in [:16] do
      let expected := (16 * (i + 1) * (j + 1) : Nat).toFloat
      let got := cData.getD (i * 16 + j) 0.0
      let diff := expected - got
      let ad := if diff < 0.0 then -diff else diff
      if ad < 0.5 then
        okCount := okCount + 1
      else
        bad := bad + 1
        if bad ≤ 3 then
          IO.println s!"    ✗ [{i},{j}] expected={expected} got={got}"
  IO.println ""
  IO.println s!"  {okCount}/256 elements match."
  if okCount == 256 then
    IO.println "  ✅ f16 subgroup matrix 16×16 works on this adapter."
  else
    IO.println "  ❌ f16 subgroup matrix 16×16 disagreed with CPU expected."

end Examples.Compute.SubgroupMatrixF16Probe

def main : IO Unit := Examples.Compute.SubgroupMatrixF16Probe.main
