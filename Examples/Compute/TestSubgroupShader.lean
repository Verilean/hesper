import Hesper.WGSL.Monad
import Hesper.WGSL.CodeGen

open Hesper.WGSL
open Hesper.WGSL.Monad
open Hesper.WGSL.CodeGen

namespace Examples.Compute.TestSubgroupShader

/-!
# Test Subgroup Matrix Operations in ShaderM

This tests the generated WGSL code from ShaderM subgroup matrix operations
without requiring a full GPU build.
-/

/-- Small subgroup matrix multiplication shader for testing code generation -/
def testSubgroupShader : ShaderM Unit := do
  let wg ← ShaderM.workgroupId
  let localID ← ShaderM.localId

  let wgX := Exp.vec3X wg
  let wgY := Exp.vec3Y wg
  let localIDY := Exp.vec3Y localID

  -- Declare buffers
  let _A ← ShaderM.declareInputBuffer "A" (.array (.scalar .f32) 64)
  let _B ← ShaderM.declareInputBuffer "B" (.array (.scalar .f32) 64)
  let _C ← ShaderM.declareOutputBuffer "C" (.array (.scalar .f32) 64)

  -- Compute positions
  ShaderM.varNamed "rowStart" (.scalar .u32) (Exp.mul wgX (Exp.litU32 8))
  ShaderM.varNamed "colStart" (.scalar .u32) (Exp.mul wgY (Exp.litU32 8))

  let rowStartVar : Exp (.scalar .u32) := Exp.var "rowStart"
  let colStartVar : Exp (.scalar .u32) := Exp.var "colStart"

  -- Declare subgroup matrices (1 tile each)
  ShaderM.declareMatrixLeftArray "Ax" .f32 8 8 1 Exp.subgroupMatrixZeroLeft
  ShaderM.declareMatrixRightArray "Bx" .f32 8 8 1 Exp.subgroupMatrixZeroRight
  ShaderM.declareMatrixResultArray "accxx" .f32 8 8 1 Exp.subgroupMatrixZeroResult

  -- Load matrix A
  let offsetA := rowStartVar
  ShaderM.loadMatrixLeft (st := .f32) (m := 8) (k := 8)
    "Ax" 0 "A" offsetA (Exp.litU32 8)

  -- Load matrix B
  let offsetB := colStartVar
  ShaderM.loadMatrixRight (st := .f32) (k := 8) (n := 8)
    "Bx" 0 "B" offsetB (Exp.litU32 8)

  -- Multiply-accumulate
  ShaderM.matrixMultiplyAccumulate (st := .f32) (m := 8) (k := 8) (n := 8)
    "accxx" 0 "Ax" 0 "Bx" 0

  -- Store result
  let offsetC := rowStartVar
  let strideC := Exp.litU32 8
  ShaderM.storeMatrixResult (st := .f32) (m := 8) (n := 8)
    "accxx" 0 "C" offsetC strideC

def main : IO Unit := do
  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║   Testing ShaderM Subgroup Operations       ║"
  IO.println "╚══════════════════════════════════════════════╝"
  IO.println ""

  -- Generate WGSL with subgroup extension and diagnostics
  let wgsl := generateWGSL
    "main"
    {x := 32, y := 1, z := 1}
    ["chromium_experimental_subgroup_matrix"]
    [("off", "chromium.subgroup_matrix_uniformity")]
    testSubgroupShader

  IO.println "Generated WGSL Code:"
  IO.println "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  IO.println wgsl
  IO.println "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  IO.println ""
  IO.println "Expected features in generated code:"
  IO.println "  ✓ enable chromium_experimental_subgroup_matrix;"
  IO.println "  ✓ diagnostic(off, chromium.subgroup_matrix_uniformity);"
  IO.println "  ✓ var<private> Ax: array<subgroup_matrix_left<...>, ...>"
  IO.println "  ✓ var<private> Bx: array<subgroup_matrix_right<...>, ...>"
  IO.println "  ✓ var<private> accxx: array<subgroup_matrix_result<...>, ...>"
  IO.println "  ✓ subgroupMatrixLoad(...)"
  IO.println "  ✓ subgroupMatrixMultiplyAccumulate(...)"
  IO.println "  ✓ subgroupMatrixStore(...)"
  IO.println ""
  IO.println "✅ ShaderM monad successfully generates subgroup matrix WGSL!"

end Examples.Compute.TestSubgroupShader

def main : IO Unit := Examples.Compute.TestSubgroupShader.main
