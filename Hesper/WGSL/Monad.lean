import Hesper.WGSL.Types
import Hesper.WGSL.Exp
import Hesper.WGSL.Shader

namespace Hesper.WGSL.Monad

open Hesper.WGSL

/-!
# Shader Monad for Imperative Shader Construction

The ShaderM monad provides an imperative interface for building WGSL compute shaders.
It tracks:
- Accumulated statements
- Fresh variable generation
- Shared memory declarations
- Automatic buffer binding

Usage pattern:
```lean
def myShader : ShaderM Unit := do
  let gid ← globalId
  let idx := swizzleX gid

  -- Declare buffers
  input ← declareInputBuffer "input" (.array (.scalar .f32) 1024)
  output ← declareOutputBuffer "output" (.array (.scalar .f32) 1024)

  -- Read, compute, write
  val ← readBuffer input idx
  let result := val * litF32 2.0
  writeBuffer output idx result
```
-/

/-- Buffer access mode for storage buffers -/
inductive BufferAccessMode where
  | read       -- var<storage, read> (read-only)
  | readWrite  -- var<storage, read_write> (read-write)
  deriving Repr, BEq

/-- Shader Construction State -/
structure ShaderState where
  stmts : List Stmt                                    -- Accumulated statements
  varCounter : Nat                                      -- For generating unique variable names
  sharedVars : List (String × WGSLType)                -- Shared memory declarations
  declaredBuffers : List (String × WGSLType × BufferAccessMode)  -- Auto-tracked buffer bindings (name, type, access mode)

/-- The Shader Monad -/
abbrev ShaderM (α : Type) := StateM ShaderState α

namespace ShaderM

/-- Initial state for shader construction -/
def initialState : ShaderState :=
  { stmts := []
    varCounter := 0
    sharedVars := []
    declaredBuffers := [] }

/-- Run a shader computation and extract the result and final state -/
def run (m : ShaderM α) : α × ShaderState :=
  m initialState

/-- Run a shader computation and extract only the final state -/
def exec (m : ShaderM α) : ShaderState :=
  (m initialState).snd

/-- Emit a statement to the shader body -/
def emitStmt (stmt : Stmt) : ShaderM Unit :=
  modify fun s => { s with stmts := s.stmts ++ [stmt] }

/-- Generate a fresh variable name with given prefix -/
def freshVar (pfx : String) : ShaderM String := do
  let s ← get
  let n := s.varCounter
  modify fun s => { s with varCounter := n + 1 }
  return s!"{pfx}{n}"

/-- Capture statements from a monadic action (for control flow) -/
def captureStmts (action : ShaderM α) : ShaderM (α × List Stmt) := do
  let oldState ← get
  -- Clear statements
  set { oldState with stmts := [] }
  let result ← action
  let newState ← get
  let capturedStmts := newState.stmts
  -- Restore old state but keep var counter
  set { oldState with varCounter := newState.varCounter }
  return (result, capturedStmts)

-- ============================================================================
-- Variable Declarations
-- ============================================================================

/-- Declare a private variable with fresh name -/
def var (ty : WGSLType) (init : Exp ty) : ShaderM String := do
  let name ← freshVar "v"
  emitStmt (Stmt.varDecl name ty (some ⟨ty, init⟩))
  return name

/-- Declare a named private variable -/
def varNamed (name : String) (ty : WGSLType) (init : Exp ty) : ShaderM Unit :=
  emitStmt (Stmt.varDecl name ty (some ⟨ty, init⟩))

/-- Declare a mutable variable with fresh name, returning both name and typed expression.
    This avoids manually constructing `Exp.var name` and passing raw string literals. -/
def varRef (ty : WGSLType) (init : Exp ty) : ShaderM (String × Exp ty) := do
  let name ← freshVar "v"
  emitStmt (Stmt.varDecl name ty (some ⟨ty, init⟩))
  return (name, Exp.var name)

/-- Declare shared memory (workgroup-scoped) with fresh name -/
def shared (ty : WGSLType) : ShaderM String := do
  let name ← freshVar "shared"
  modify fun s => { s with sharedVars := s.sharedVars ++ [(name, ty)] }
  return name

/-- Declare named shared memory -/
def sharedNamed (name : String) (ty : WGSLType) : ShaderM Unit :=
  modify fun s => { s with sharedVars := s.sharedVars ++ [(name, ty)] }

-- ============================================================================
-- Assignment
-- ============================================================================

/-- Assign expression to variable -/
def assign {ty : WGSLType} (varName : String) (expr : Exp ty) : ShaderM Unit :=
  emitStmt (Stmt.assign varName ty expr)

/-- Assign to array index -/
def assignIndex {ty : WGSLType} (arrName : String) (idx : Exp (.scalar .u32)) (value : Exp ty) : ShaderM Unit :=
  emitStmt (Stmt.assignIndex arrName idx ty value)

-- ============================================================================
-- Control Flow
-- ============================================================================

/-- If-then-else statement -/
def if_ (cond : Exp (.scalar .bool)) (thenBranch : ShaderM Unit) (elseBranch : ShaderM Unit) : ShaderM Unit := do
  let (_, thenStmts) ← captureStmts thenBranch
  let (_, elseStmts) ← captureStmts elseBranch
  emitStmt (Stmt.ifStmt cond thenStmts elseStmts)

/-- While loop - implemented as for loop with always-true condition -/
def while_ (_cond : Exp (.scalar .bool)) (body : ShaderM Unit) : ShaderM Unit := do
  -- Note: WGSL doesn't have while loops, so we'd need to use a different approach
  -- For now, just capture the body statements
  let (_, bodyStmts) ← captureStmts body
  -- This is a simplified version - real implementation would need break/continue support
  emitStmt (Stmt.block bodyStmts)

/-- For loop (start to end, incrementing by step)
    Builds proper WGSL for loop: for (var i: u32 = start; i < end; i = i + step) -/
def for_ (varName : String) (start : Exp (.scalar .u32)) (end_ : Exp (.scalar .u32)) (step : Exp (.scalar .u32)) (body : ShaderM Unit) : ShaderM Unit := do
  let (_, bodyStmts) ← captureStmts body
  let loopVar : Exp (.scalar .u32) := Exp.var varName
  let condition := Exp.lt loopVar end_
  let update := Exp.add loopVar step
  emitStmt (Stmt.forLoop varName start condition update bodyStmts)

/-- Higher-order loop: pass loop variable as Exp
    Usage: loop start end step fun i => do { ... use i ... } -/
def loop (start : Exp (.scalar .u32)) (end_ : Exp (.scalar .u32)) (step : Exp (.scalar .u32)) (bodyFn : Exp (.scalar .u32) → ShaderM Unit) : ShaderM Unit := do
  let varName ← freshVar "i"
  let (_, bodyStmts) ← captureStmts (bodyFn (Exp.var varName))
  let loopVar : Exp (.scalar .u32) := Exp.var varName
  let condition := Exp.lt loopVar end_
  let update := Exp.add loopVar step
  emitStmt (Stmt.forLoop varName start condition update bodyStmts)

-- ============================================================================
-- Synchronization
-- ============================================================================

/-- Workgroup barrier (synchronization) -/
def barrier : ShaderM Unit :=
  emitStmt (Stmt.exprStmt Exp.workgroupBarrier)

-- ============================================================================
-- Built-in Variables
-- ============================================================================

/-- Global invocation ID (3D) -/
def globalId : ShaderM (Exp (.vec3 .u32)) :=
  return Exp.var "global_invocation_id"

/-- Local invocation ID (3D) -/
def localId : ShaderM (Exp (.vec3 .u32)) :=
  return Exp.var "local_invocation_id"

/-- Workgroup ID (3D) -/
def workgroupId : ShaderM (Exp (.vec3 .u32)) :=
  return Exp.var "workgroup_id"

/-- Number of workgroups (3D) -/
def numWorkgroups : ShaderM (Exp (.vec3 .u32)) :=
  return Exp.var "num_workgroups"

-- ============================================================================
-- Buffer Operations
-- ============================================================================

/-- Read from a global storage buffer at index
    Note: You need to provide the element type explicitly -/
def readBuffer {ty : WGSLType} {n : Nat} (bufferName : String) (idx : Exp (.scalar .u32)) : ShaderM (Exp ty) :=
  return Exp.index (Exp.var bufferName : Exp (.array ty n)) idx

/-- Write to a global storage buffer at index -/
def writeBuffer {ty : WGSLType} (bufferName : String) (idx : Exp (.scalar .u32)) (value : Exp ty) : ShaderM Unit :=
  assignIndex bufferName idx value

/-- Read from workgroup shared memory at index -/
def readWorkgroup {ty : WGSLType} {n : Nat} (sharedName : String) (idx : Exp (.scalar .u32)) : ShaderM (Exp ty) :=
  return Exp.index (Exp.var sharedName : Exp (.array ty n)) idx

/-- Write to workgroup shared memory at index -/
def writeWorkgroup {ty : WGSLType} (sharedName : String) (idx : Exp (.scalar .u32)) (value : Exp ty) : ShaderM Unit :=
  assignIndex sharedName idx value

-- ============================================================================
-- Automatic Binding Management
-- ============================================================================

/-- Declare an input buffer (read-only) with automatic binding assignment -/
def declareInputBuffer (name : String) (ty : WGSLType) : ShaderM String := do
  modify fun s => { s with declaredBuffers := s.declaredBuffers ++ [(name, ty, .readWrite)] }
  return name

/-- Declare an output buffer (read-write) with automatic binding assignment -/
def declareOutputBuffer (name : String) (ty : WGSLType) : ShaderM String := do
  modify fun s => { s with declaredBuffers := s.declaredBuffers ++ [(name, ty, .readWrite)] }
  return name

/-- Declare a storage buffer with explicit access mode -/
def declareStorageBuffer (name : String) (ty : WGSLType)
    (mode : BufferAccessMode := .readWrite) : ShaderM String := do
  modify fun s => { s with declaredBuffers := s.declaredBuffers ++ [(name, ty, mode)] }
  return name

-- ============================================================================
-- High-level Helpers
-- ============================================================================

/-- Compile-time loop for unrolling (Haskell-side loop, not WGSL loop) -/
def staticFor {α β : Type} (xs : List α) (f : α → ShaderM β) : ShaderM Unit :=
  xs.forM (fun x => f x *> pure ())

/-- Float32 literal -/
def litF (x : Float) : Exp (.scalar .f32) :=
  Exp.litF32 x

/-- Int32 literal -/
def litI (x : Int) : Exp (.scalar .i32) :=
  Exp.litI32 x

/-- UInt32 literal -/
def litU (x : Nat) : Exp (.scalar .u32) :=
  Exp.litU32 x

-- ============================================================================
-- Subgroup Matrix Operations (chromium_experimental_subgroup_matrix)
-- ============================================================================

/-- Declare an array of subgroup_matrix_left matrices with initialization -/
def declareMatrixLeftArray
    (name : String)
    (st : ScalarType)
    (m k : Nat)
    (count : Nat)
    (init : Exp (.subgroupMatrixLeft st m k))
    : ShaderM Unit := do
  let matTy := WGSLType.subgroupMatrixLeft st m k
  let arrTy := WGSLType.array matTy count
  emitStmt (Stmt.varDecl name arrTy none)
  -- Initialize all elements
  for i in [0:count] do
    emitStmt (Stmt.assignIndex name (Exp.litU32 i) matTy init)

/-- Declare an array of subgroup_matrix_right matrices with initialization -/
def declareMatrixRightArray
    (name : String)
    (st : ScalarType)
    (k n : Nat)
    (count : Nat)
    (init : Exp (.subgroupMatrixRight st k n))
    : ShaderM Unit := do
  let matTy := WGSLType.subgroupMatrixRight st k n
  let arrTy := WGSLType.array matTy count
  emitStmt (Stmt.varDecl name arrTy none)
  -- Initialize all elements
  for i in [0:count] do
    emitStmt (Stmt.assignIndex name (Exp.litU32 i) matTy init)

/-- Declare an array of subgroup_matrix_result matrices with initialization -/
def declareMatrixResultArray
    (name : String)
    (st : ScalarType)
    (m n : Nat)
    (count : Nat)
    (init : Exp (.subgroupMatrixResult st m n))
    : ShaderM Unit := do
  let matTy := WGSLType.subgroupMatrixResult st m n
  let arrTy := WGSLType.array matTy count
  emitStmt (Stmt.varDecl name arrTy none)
  -- Initialize all elements
  for i in [0:count] do
    emitStmt (Stmt.assignIndex name (Exp.litU32 i) matTy init)

/-- Load subgroup_matrix_left from buffer

    Example: Ax[i] = subgroupMatrixLoad<subgroup_matrix_left<f32,8,8>>(&A, offset, false, stride)
-/
def loadMatrixLeft
    {st : ScalarType} {m k : Nat}
    (arrayName : String)
    (index : Nat)
    (bufferName : String)
    (offset : Exp (.scalar .u32))
    (stride : Exp (.scalar .u32))
    : ShaderM Unit := do
  let matTy := WGSLType.subgroupMatrixLeft st m k
  let loadExpr := Exp.subgroupMatrixLoad bufferName offset (Exp.litBool false) stride
  emitStmt (Stmt.assignIndex arrayName (Exp.litU32 index) matTy loadExpr)

/-- Load subgroup_matrix_right from buffer -/
def loadMatrixRight
    {st : ScalarType} {k n : Nat}
    (arrayName : String)
    (index : Nat)
    (bufferName : String)
    (offset : Exp (.scalar .u32))
    (stride : Exp (.scalar .u32))
    : ShaderM Unit := do
  let matTy := WGSLType.subgroupMatrixRight st k n
  let loadExpr := Exp.subgroupMatrixLoadRight bufferName offset (Exp.litBool false) stride
  emitStmt (Stmt.assignIndex arrayName (Exp.litU32 index) matTy loadExpr)

/-- Perform matrix multiply-accumulate: acc = a * b + acc

    Example: accxx[idx] = subgroupMatrixMultiplyAccumulate(Ax[i], Bx[j], accxx[idx])
-/
def matrixMultiplyAccumulate
    {st : ScalarType} {m k n : Nat}
    (resultArrayName : String)
    (resultIndex : Nat)
    (leftArrayName : String)
    (leftIndex : Nat)
    (rightArrayName : String)
    (rightIndex : Nat)
    : ShaderM Unit := do
  let leftMatTy := WGSLType.subgroupMatrixLeft st m k
  let rightMatTy := WGSLType.subgroupMatrixRight st k n
  let resultMatTy := WGSLType.subgroupMatrixResult st m n

  let leftArr : Exp (.array leftMatTy leftIndex) := Exp.var leftArrayName
  let rightArr : Exp (.array rightMatTy rightIndex) := Exp.var rightArrayName
  let resultArr : Exp (.array resultMatTy resultIndex) := Exp.var resultArrayName

  let leftMat := Exp.index leftArr (Exp.litU32 leftIndex)
  let rightMat := Exp.index rightArr (Exp.litU32 rightIndex)
  let accMat := Exp.index resultArr (Exp.litU32 resultIndex)

  let mulAccExpr := Exp.subgroupMatrixMultiplyAccumulate leftMat rightMat accMat
  emitStmt (Stmt.assignIndex resultArrayName (Exp.litU32 resultIndex) resultMatTy mulAccExpr)

/-- Store subgroup_matrix_result to buffer

    Example: subgroupMatrixStore(&C, offset, accxx[idx], false, stride)
-/
def storeMatrixResult
    {st : ScalarType} {m n : Nat}
    (arrayName : String)
    (index : Nat)
    (bufferName : String)
    (offset : Exp (.scalar .u32))
    (stride : Exp (.scalar .u32))
    : ShaderM Unit := do
  let resultMatTy := WGSLType.subgroupMatrixResult st m n
  let resultArr : Exp (.array resultMatTy index) := Exp.var arrayName
  let mat := Exp.index resultArr (Exp.litU32 index)
  let storeExpr := Exp.subgroupMatrixStore bufferName offset mat (Exp.litBool false) stride
  emitStmt (Stmt.exprStmt storeExpr)

/-- Static loop unrolling for matrix operations

    Executes an action for each index in the range [0, count)
-/
def staticLoop (count : Nat) (body : Nat → ShaderM Unit) : ShaderM Unit := do
  for i in [0:count] do
    body i

/-- Static nested loop for 2D iteration -/
def staticLoop2D (rows cols : Nat) (body : Nat → Nat → ShaderM Unit) : ShaderM Unit := do
  for i in [0:rows] do
    for j in [0:cols] do
      body i j

end ShaderM

end Hesper.WGSL.Monad
