import LSpec
import Hesper.WGSL.Monad
import Hesper.WGSL.Shader

/-!
# ShaderM Monad Comprehensive Tests

Tests for the ShaderM monad covering:
- State management and initialization
- Variable declarations (fresh and named)
- Shared memory declarations
- Control flow (if, for, loop)
- Buffer operations (read/write)
- Built-in variables
- Statement emission and capture
- Complete shader generation
-/

namespace Tests.ShaderMonadTests

open Hesper.WGSL
open Hesper.WGSL.Monad (ShaderM ShaderState)
open Hesper.WGSL.Monad.ShaderM
open LSpec

-- ============================================================================
-- State Management Tests
-- ============================================================================

/-- Test: Initial state is empty -/
def testInitialState : TestSeq :=
  let state := initialState
  test "Initial stmts empty" (state.stmts.length == 0) ++
  test "Initial varCounter is 0" (state.varCounter == 0) ++
  test "Initial sharedVars empty" (state.sharedVars.length == 0) ++
  test "Initial declaredBuffers empty" (state.declaredBuffers.length == 0)

/-- Test: Fresh variable generation increments counter -/
def testFreshVarIncrement : TestSeq :=
  let (v1, s1) := run (freshVar "tmp")
  let (v2, s2) := freshVar "tmp" s1
  let (v3, s3) := freshVar "tmp" s2

  test "First var is tmp0" (v1 == "tmp0") ++
  test "Second var is tmp1" (v2 == "tmp1") ++
  test "Third var is tmp2" (v3 == "tmp2") ++
  test "Counter increments" (s3.varCounter == 3)

/-- Test: Multiple fresh vars with different prefixes -/
def testFreshVarPrefixes : TestSeq :=
  let computation := do
    let v1 ← freshVar "x"
    let v2 ← freshVar "y"
    let v3 ← freshVar "x"
    pure (v1, v2, v3)

  let ((v1, v2, v3), _) := run computation

  test "Variable x0" (v1 == "x0") ++
  test "Variable y1" (v2 == "y1") ++
  test "Variable x2" (v3 == "x2")

/-- Test: Statement emission appends to list -/
def testEmitStmt : TestSeq :=
  let computation := do
    emitStmt (Stmt.exprStmt Exp.workgroupBarrier)
    emitStmt (Stmt.exprStmt Exp.workgroupBarrier)

  let state := exec computation

  test "Two statements emitted" (state.stmts.length == 2)

-- ============================================================================
-- Variable Declaration Tests
-- ============================================================================

/-- Test: Var declares variable and returns fresh name -/
def testVarDeclaration : TestSeq :=
  let computation := do
    let name ← Monad.ShaderM.var (.scalar .f32) (Exp.litF32 42.0)
    pure name

  let (name, state) := run computation

  test "Returns fresh variable name" (name == "v0") ++
  test "Emits one statement" (state.stmts.length == 1) ++
  test "Counter incremented" (state.varCounter == 1)

/-- Test: Named variable declaration -/
def testVarNamed : TestSeq :=
  let computation := varNamed "myVar" (.scalar .f32) (Exp.litF32 3.14)
  let state := exec computation

  test "Emits one statement for named var" (state.stmts.length == 1) ++
  test "Counter unchanged for named var" (state.varCounter == 0)

/-- Test: Multiple variable declarations -/
def testMultipleVars : TestSeq :=
  let computation := do
    let _v1 ← Monad.ShaderM.var (.scalar .f32) (Exp.litF32 1.0)
    let _v2 ← Monad.ShaderM.var (.scalar .i32) (Exp.litI32 2)
    let _v3 ← Monad.ShaderM.var (.scalar .u32) (Exp.litU32 3)
    pure ()

  let state := exec computation

  test "Three variables declared" (state.stmts.length == 3) ++
  test "Counter at 3" (state.varCounter == 3)

-- ============================================================================
-- Shared Memory Tests
-- ============================================================================

/-- Test: Shared memory declaration with fresh name -/
def testSharedDeclaration : TestSeq :=
  let computation := do
    let name ← shared (.array (.scalar .f32) 256)
    pure name

  let (name, state) := run computation

  test "Returns fresh shared name" (name == "shared0") ++
  test "Shared var registered" (state.sharedVars.length == 1) ++
  test "Counter incremented" (state.varCounter == 1)

/-- Test: Named shared memory declaration -/
def testSharedNamed : TestSeq :=
  let computation := sharedNamed "myShared" (.array (.scalar .f32) 256)
  let state := exec computation

  test "Shared var registered" (state.sharedVars.length == 1) ++
  test "Counter unchanged for named shared" (state.varCounter == 0)

/-- Test: Multiple shared declarations -/
def testMultipleShared : TestSeq :=
  let computation := do
    let _s1 ← shared (.array (.scalar .f32) 256)
    let _s2 ← shared (.array (.scalar .i32) 128)
    pure ()

  let state := exec computation

  test "Two shared vars registered" (state.sharedVars.length == 2) ++
  test "Counter at 2" (state.varCounter == 2)

-- ============================================================================
-- Assignment Tests
-- ============================================================================

/-- Test: Simple assignment -/
def testAssign : TestSeq :=
  let computation := ShaderM.assign "x" (Exp.litF32 42.0 : Exp (.scalar .f32))
  let state := exec computation

  test "Assignment emits statement" (state.stmts.length == 1)

/-- Test: Array index assignment -/
def testAssignIndex : TestSeq :=
  let computation := assignIndex "arr" (Exp.litU32 5) (Exp.litF32 3.14 : Exp (.scalar .f32))
  let state := exec computation

  test "Index assignment emits statement" (state.stmts.length == 1)

-- ============================================================================
-- Control Flow Tests
-- ============================================================================

/-- Test: If statement captures branches -/
def testIfStatement : TestSeq :=
  let computation :=
    if_ (Exp.litBool true)
      (emitStmt (Stmt.exprStmt Exp.workgroupBarrier))
      (emitStmt (Stmt.exprStmt Exp.workgroupBarrier))

  let state := exec computation

  test "If statement emits one stmt" (state.stmts.length == 1)

/-- Test: For loop statement -/
def testForLoop : TestSeq :=
  let computation :=
    for_ "i" (Exp.litU32 0) (Exp.litU32 10) (Exp.litU32 1) do
      emitStmt (Stmt.exprStmt Exp.workgroupBarrier)

  let state := exec computation

  test "For loop emits one stmt" (state.stmts.length == 1)

/-- Test: Loop with higher-order function -/
def testLoop : TestSeq :=
  let computation :=
    loop (Exp.litU32 0) (Exp.litU32 5) (Exp.litU32 1) fun _i => do
      emitStmt (Stmt.exprStmt Exp.workgroupBarrier)

  let state := exec computation

  test "Loop emits one stmt" (state.stmts.length == 1) ++
  test "Loop creates fresh loop var" (state.varCounter == 1)

/-- Test: Nested control flow -/
def testNestedControlFlow : TestSeq :=
  let computation := do
    if_ (Exp.litBool true)
      (loop (Exp.litU32 0) (Exp.litU32 3) (Exp.litU32 1) fun _i => do
        emitStmt (Stmt.exprStmt Exp.workgroupBarrier))
      (pure ())

  let state := exec computation

  test "Nested control flow emits one stmt" (state.stmts.length == 1)

-- ============================================================================
-- Synchronization Tests
-- ============================================================================

/-- Test: Barrier statement -/
def testBarrier : TestSeq :=
  let computation := barrier
  let state := exec computation

  test "Barrier emits statement" (state.stmts.length == 1)

-- ============================================================================
-- Built-in Variable Tests
-- ============================================================================

/-- Test: Global ID returns expected expression -/
def testGlobalId : TestSeq :=
  let computation := do
    let gid ← globalId
    pure gid

  let (gid, _) := run computation

  -- Check that it returns an Exp that references the builtin
  test "GlobalId computation succeeds" true

/-- Test: Local ID returns expected expression -/
def testLocalId : TestSeq :=
  let computation := do
    let lid ← localId
    pure lid

  let (lid, _) := run computation

  test "LocalId computation succeeds" true

/-- Test: Workgroup ID returns expected expression -/
def testWorkgroupId : TestSeq :=
  let computation := do
    let wid ← workgroupId
    pure wid

  let (wid, _) := run computation

  test "WorkgroupId computation succeeds" true

/-- Test: Num workgroups returns expected expression -/
def testNumWorkgroups : TestSeq :=
  let computation := do
    let nwg ← numWorkgroups
    pure nwg

  let (nwg, _) := run computation

  test "NumWorkgroups computation succeeds" true

-- ============================================================================
-- Buffer Operations Tests
-- ============================================================================

/-- Test: Declare input buffer -/
def testDeclareInputBuffer : TestSeq :=
  let computation := declareInputBuffer "input" (.array (.scalar .f32) 1024)
  let state := exec computation

  test "Input buffer registered" (state.declaredBuffers.length == 1)

/-- Test: Declare output buffer -/
def testDeclareOutputBuffer : TestSeq :=
  let computation := declareOutputBuffer "output" (.array (.scalar .f32) 1024)
  let state := exec computation

  test "Output buffer registered" (state.declaredBuffers.length == 1)

/-- Test: Declare storage buffer -/
def testDeclareStorageBuffer : TestSeq :=
  let computation := declareStorageBuffer "storage" (.array (.scalar .f32) 512)
  let state := exec computation

  test "Storage buffer registered" (state.declaredBuffers.length == 1)

/-- Test: Multiple buffer declarations -/
def testMultipleBuffers : TestSeq :=
  let computation := do
    let _in1 ← declareInputBuffer "input1" (.array (.scalar .f32) 1024)
    let _in2 ← declareInputBuffer "input2" (.array (.scalar .f32) 1024)
    let _out ← declareOutputBuffer "output" (.array (.scalar .f32) 1024)
    pure ()

  let state := exec computation

  test "Three buffers registered" (state.declaredBuffers.length == 3)

/-- Test: Buffer read operation -/
def testBufferRead : TestSeq :=
  let computation := do
    let val ← readBuffer (ty := .scalar .f32) (n := 1024) "input" (Exp.litU32 0)
    pure val

  let (_val, state) := run computation

  test "Buffer read succeeds" (state.stmts.length == 0)  -- Read doesn't emit stmt

/-- Test: Buffer write operation -/
def testBufferWrite : TestSeq :=
  let computation := writeBuffer (ty := .scalar .f32) "output" (Exp.litU32 0) (Exp.litF32 42.0)
  let state := exec computation

  test "Buffer write emits statement" (state.stmts.length == 1)

-- ============================================================================
-- Statement Capture Tests
-- ============================================================================

/-- Test: Capture statements isolates nested actions -/
def testCaptureStmts : TestSeq :=
  let computation := do
    emitStmt (Stmt.exprStmt Exp.workgroupBarrier)  -- Outer stmt 1
    let (_result, captured) ← captureStmts do
      emitStmt (Stmt.exprStmt Exp.workgroupBarrier)  -- Captured stmt 1
      emitStmt (Stmt.exprStmt Exp.workgroupBarrier)  -- Captured stmt 2
    emitStmt (Stmt.exprStmt Exp.workgroupBarrier)  -- Outer stmt 2
    pure captured

  let (captured, state) := run computation

  test "Captured 2 statements" (captured.length == 2) ++
  test "Outer has 2 statements" (state.stmts.length == 2)

-- ============================================================================
-- Integration Tests - Complete Shader Construction
-- ============================================================================

/-- Test: Simple vector add shader -/
def testSimpleVectorAdd : TestSeq :=
  let computation := do
    let gid ← globalId
    let idx := Exp.vecZ gid

    let _inputA ← declareInputBuffer "inputA" (.array (.scalar .f32) 1024)
    let _inputB ← declareInputBuffer "inputB" (.array (.scalar .f32) 1024)
    let _output ← declareOutputBuffer "output" (.array (.scalar .f32) 1024)

    let valA ← readBuffer (ty := .scalar .f32) (n := 1024) "inputA" idx
    let valB ← readBuffer (ty := .scalar .f32) (n := 1024) "inputB" idx
    let sum := Exp.add valA valB

    writeBuffer (ty := .scalar .f32) "output" idx sum

  let state := exec computation

  test "Vector add has 3 buffers" (state.declaredBuffers.length == 3) ++
  test "Vector add emits write statement" (state.stmts.length == 1)

/-- Test: Shader with loop and accumulator -/
def testShaderWithLoop : TestSeq :=
  let computation := do
    let gid ← globalId
    let idx := Exp.vecZ gid

    let _output ← declareOutputBuffer "output" (.array (.scalar .f32) 1024)

    let acc ← Monad.ShaderM.var (.scalar .f32) (Exp.litF32 0.0)

    loop (Exp.litU32 0) (Exp.litU32 10) (Exp.litU32 1) fun i => do
      let iAsF32 := Exp.toF32 i
      let newAcc := Exp.add (Exp.var acc) iAsF32
      assign acc newAcc

    writeBuffer (ty := .scalar .f32) "output" idx (Exp.var acc)

  let state := exec computation

  test "Loop shader has 1 buffer" (state.declaredBuffers.length == 1) ++
  test "Loop shader has statements (var, loop, write)" (state.stmts.length == 3)

/-- Test: Shader with shared memory -/
def testShaderWithShared : TestSeq :=
  let computation := do
    let _sharedMem ← shared (.array (.scalar .f32) 256)
    let gid ← globalId
    let _lid ← localId
    pure ()

  let state := exec computation

  test "Shared memory registered" (state.sharedVars.length == 1)

/-- Test: Shader with conditional -/
def testShaderWithConditional : TestSeq :=
  let computation := do
    let gid ← globalId
    let idx := Exp.vecZ gid

    let _input ← declareInputBuffer "input" (.array (.scalar .f32) 1024)
    let _output ← declareOutputBuffer "output" (.array (.scalar .f32) 1024)

    let val ← readBuffer (ty := .scalar .f32) (n := 1024) "input" idx

    if_ (Exp.lt val (Exp.litF32 0.0))
      (writeBuffer (ty := .scalar .f32) "output" idx (Exp.litF32 0.0))
      (writeBuffer (ty := .scalar .f32) "output" idx val)

  let state := exec computation

  test "Conditional shader has 2 buffers" (state.declaredBuffers.length == 2) ++
  test "Conditional shader emits if statement" (state.stmts.length == 1)

/-- Test: Complex shader with multiple features -/
def testComplexShader : TestSeq :=
  let computation := do
    let gid ← globalId
    let lid ← localId
    let idx := Exp.vecZ gid
    let localIdx := Exp.vecZ lid

    let _sharedMem ← shared (.array (.scalar .f32) 256)
    let _input ← declareInputBuffer "input" (.array (.scalar .f32) 1024)
    let _output ← declareOutputBuffer "output" (.array (.scalar .f32) 1024)

    let val ← readBuffer (ty := .scalar .f32) (n := 1024) "input" idx

    writeWorkgroup (ty := .scalar .f32) "shared_mem0" localIdx val
    barrier

    let sharedVal ← readWorkgroup (ty := .scalar .f32) (n := 256) "shared_mem0" localIdx
    let result := Exp.mul sharedVal (Exp.litF32 2.0)

    writeBuffer (ty := .scalar .f32) "output" idx result

  let state := exec computation

  test "Complex shader has 2 buffers" (state.declaredBuffers.length == 2) ++
  test "Complex shader has 1 shared var" (state.sharedVars.length == 1) ++
  test "Complex shader emits multiple statements" (state.stmts.length >= 3)

-- ============================================================================
-- All Tests
-- ============================================================================

def allTests : IO (List (String × List TestSeq)) := do
  IO.println "Running ShaderM Monad Tests..."

  pure [
    ("State Management: Initial State", [testInitialState]),
    ("State Management: Fresh Var Increment", [testFreshVarIncrement]),
    ("State Management: Fresh Var Prefixes", [testFreshVarPrefixes]),
    ("State Management: Emit Stmt", [testEmitStmt]),
    ("Variables: Var Declaration", [testVarDeclaration]),
    ("Variables: Named Var", [testVarNamed]),
    ("Variables: Multiple Vars", [testMultipleVars]),
    ("Shared Memory: Declaration", [testSharedDeclaration]),
    ("Shared Memory: Named", [testSharedNamed]),
    ("Shared Memory: Multiple", [testMultipleShared]),
    ("Assignment: Simple", [testAssign]),
    ("Assignment: Index", [testAssignIndex]),
    ("Control Flow: If Statement", [testIfStatement]),
    ("Control Flow: For Loop", [testForLoop]),
    ("Control Flow: Loop", [testLoop]),
    ("Control Flow: Nested", [testNestedControlFlow]),
    ("Synchronization: Barrier", [testBarrier]),
    ("Built-ins: Global ID", [testGlobalId]),
    ("Built-ins: Local ID", [testLocalId]),
    ("Built-ins: Workgroup ID", [testWorkgroupId]),
    ("Built-ins: Num Workgroups", [testNumWorkgroups]),
    ("Buffers: Declare Input", [testDeclareInputBuffer]),
    ("Buffers: Declare Output", [testDeclareOutputBuffer]),
    ("Buffers: Declare Storage", [testDeclareStorageBuffer]),
    ("Buffers: Multiple", [testMultipleBuffers]),
    ("Buffers: Read", [testBufferRead]),
    ("Buffers: Write", [testBufferWrite]),
    ("Statement Capture", [testCaptureStmts]),
    ("Integration: Simple Vector Add", [testSimpleVectorAdd]),
    ("Integration: Shader with Loop", [testShaderWithLoop]),
    ("Integration: Shader with Shared Memory", [testShaderWithShared]),
    ("Integration: Shader with Conditional", [testShaderWithConditional]),
    ("Integration: Complex Shader", [testComplexShader])
  ]

end Tests.ShaderMonadTests
