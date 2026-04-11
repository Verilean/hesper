import Hesper.CUDA.CodeGen
import Hesper.WGSL.Monad

/-!
# PTX Code Generation Tests

Verifies that ShaderM kernels produce valid PTX assembly via `generatePTX`.
-/

open Hesper.WGSL
open Hesper.WGSL.Monad (ShaderM)
open Hesper.WGSL.Monad.ShaderM
open Hesper.CUDA.CodeGen

def containsSubstr (s : String) (sub : String) : Bool :=
  (s.splitOn sub).length > 1

/-! Simple vector-double kernel: output[i] = input[i] * 2.0 -/
def simpleKernel : ShaderM Unit := do
  let gid ← globalId
  let idx := Exp.vec3X gid

  let _input ← declareInputBuffer "input" (.array (.scalar .f32) 1024)
  let _output ← declareOutputBuffer "output" (.array (.scalar .f32) 1024)

  let val ← readBuffer (ty := .scalar .f32) (n := 1024) "input" idx
  let result := Exp.mul val (Exp.litF32 2.0)
  writeBuffer (ty := .scalar .f32) "output" idx result

/-! Reduction kernel with shared memory and barrier -/
def reductionKernel : ShaderM Unit := do
  let gid ← globalId
  let lid ← localId
  let globalIdx := Exp.vec3X gid
  let localIdx := Exp.vec3X lid

  let _input ← declareInputBuffer "input" (.array (.scalar .f32) 4096)
  let _output ← declareOutputBuffer "output" (.array (.scalar .f32) 16)

  sharedNamed "sdata" (.array (.scalar .f32) 256)

  let val ← readBuffer (ty := .scalar .f32) (n := 4096) "input" globalIdx
  writeWorkgroup (ty := .scalar .f32) "sdata" localIdx val
  barrier

  let sum := Exp.subgroupAdd val
  if_ (Exp.eq localIdx (Exp.litU32 0))
    (do
      let wgId := Exp.vec3X (← workgroupId)
      writeBuffer (ty := .scalar .f32) "output" wgId sum)
    (pure ())

def main : IO Unit := do
  IO.println "═══ PTX Code Generation Tests ═══"
  IO.println ""

  -- Test 1: Simple kernel
  IO.println "Test 1: Simple Vector Double"
  IO.println "────────────────────────────"
  let ptx1 := generatePTX "vectorDouble" {x := 256} simpleKernel
  IO.println ptx1

  -- Validate key PTX features
  let checks := [
    (".version 8.0", "PTX version header"),
    (".target sm_89", "Target architecture"),
    (".entry vectorDouble", "Entry point name"),
    (".param .u64 param_input", "Input buffer param"),
    (".param .u64 param_output", "Output buffer param"),
    ("ld.param.u64", "Param load"),
    ("ld.global.f32", "Global memory load"),
    ("st.global.f32", "Global memory store (via assignIndex)"),
    ("mul.f32", "Float multiply"),
    ("mov.u32", "Thread ID access"),
    ("ret;", "Return instruction")
  ]

  let mut passed := 0
  let mut failed := 0
  for (needle, desc) in checks do
    if containsSubstr ptx1 needle then
      IO.println s!"  ✓ {desc}: found '{needle}'"
      passed := passed + 1
    else
      IO.println s!"  ✗ {desc}: missing '{needle}'"
      failed := failed + 1

  IO.println ""

  -- Test 2: Reduction with shared memory + subgroupAdd
  IO.println "Test 2: Reduction with Shared Memory + SubgroupAdd"
  IO.println "───────────────────────────────────────────────────"
  let ptx2 := generatePTX "reduce" {x := 256} reductionKernel
  IO.println ptx2

  let checks2 := [
    (".shared .f32 sdata", "Shared memory declaration"),
    ("bar.sync 0", "Workgroup barrier"),
    ("shfl.sync.bfly.b32", "Warp shuffle for subgroupAdd"),
    ("%ctaid.x", "Workgroup ID access"),
    ("%tid.x", "Local thread ID access"),
    ("setp.eq", "Predicate set for if-stmt"),
    ("@!", "Predicated branch")
  ]

  for (needle, desc) in checks2 do
    if containsSubstr ptx2 needle then
      IO.println s!"  ✓ {desc}: found '{needle}'"
      passed := passed + 1
    else
      IO.println s!"  ✗ {desc}: missing '{needle}'"
      failed := failed + 1

  IO.println ""
  IO.println s!"═══ Results: {passed} passed, {failed} failed ═══"
  if failed == 0 then
    IO.println "✓ ALL PTX CODEGEN TESTS PASSED"
  else
    IO.println "✗ SOME TESTS FAILED"
    IO.Process.exit 1
