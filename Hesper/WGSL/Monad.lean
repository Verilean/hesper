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

/-- Shader Construction State -/
structure ShaderState where
  stmts : List Stmt                                    -- Accumulated statements
  varCounter : Nat                                      -- For generating unique variable names
  sharedVars : List (String × WGSLType)                -- Shared memory declarations
  declaredBuffers : List (String × WGSLType × String)  -- Auto-tracked buffer bindings (name, type, space)

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

/-- Declare an input buffer with automatic binding assignment -/
def declareInputBuffer (name : String) (ty : WGSLType) : ShaderM String := do
  modify fun s => { s with declaredBuffers := s.declaredBuffers ++ [(name, ty, "storage")] }
  return name

/-- Declare an output buffer with automatic binding assignment -/
def declareOutputBuffer (name : String) (ty : WGSLType) : ShaderM String := do
  modify fun s => { s with declaredBuffers := s.declaredBuffers ++ [(name, ty, "storage")] }
  return name

/-- Declare a storage buffer with automatic binding assignment -/
def declareStorageBuffer (name : String) (ty : WGSLType) : ShaderM String := do
  modify fun s => { s with declaredBuffers := s.declaredBuffers ++ [(name, ty, "storage")] }
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

end ShaderM

end Hesper.WGSL.Monad
