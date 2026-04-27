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
  needsSubgroups : Bool := false                       -- Set automatically when subgroupAdd is used

/-- The Shader Monad -/
abbrev ShaderM (α : Type) := StateM ShaderState α

namespace ShaderM

/-- Initial state for shader construction -/
def initialState : ShaderState :=
  { stmts := []
    varCounter := 0
    sharedVars := []
    declaredBuffers := []
    needsSubgroups := false }

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

/-- Bind a sub-expression to a named PTX register and return an `Exp.var`
    referring to it.  Use this when an `Exp` is referenced multiple times
    inside an unrolled loop body or fan-out: without it, each reference is
    inlined as a fresh AST traversal and the CodeGen emits the same
    arithmetic at every use site (causing the 17× instruction inflation
    seen in V9 vs llama.cpp).  Wrapping the value with `let'` materialises
    it as one PTX register that all uses share.

    Example:
    ```
    let kRowBase ← ShaderM.let' (.scalar .u32) (Exp.add base offset)
    for pk in [0:4] do
      -- `kRowBase` is a single Exp.var, so the four iterations all
      -- read the same register instead of recomputing `base + offset`.
      let idx := Exp.add kRowBase (Exp.litU32 (pk * 32))
      ...
    ```
    Functionally equivalent to `do let n ← var ty e; pure (Exp.var n)`,
    but the intent is clearer at the call site. -/
def let' (ty : WGSLType) (init : Exp ty) : ShaderM (Exp ty) := do
  let name ← freshVar "v"
  emitStmt (Stmt.varDecl name ty (some ⟨ty, init⟩))
  return Exp.var name

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

/-- Compile-time unrolled loop: emits `n` copies of the body inline,
    with `i` bound to a Lean `Nat` (0, 1, ..., n-1).  Use when iteration
    count is statically known and the unroll is desired (eg. dot-product
    over a fixed dim count, sub-warp partition iters).

    The Nat `i` is a meta value — to use it in an Exp context, lift via
    `Exp.litU32 i` or simply `(i : Exp _)` (the Exp OfNat instance).

    Equivalent to writing `for i in [0:n] do body i`, but the helper name
    documents intent: "this is meta-time unrolled, not a runtime loop".
    Mirrors CUDA C++ `#pragma unroll for (int i = 0; i < N; ++i)`.

    For runtime loops use `ShaderM.loop` (or its alias `runtimeFor`). -/
@[inline] def unrollFor (n : Nat) (body : Nat → ShaderM Unit) : ShaderM Unit :=
  (List.range n).forM body

/-- Higher-order loop: pass loop variable as Exp
    Usage: loop start end step fun i => do { ... use i ... } -/
def loop (start : Exp (.scalar .u32)) (end_ : Exp (.scalar .u32)) (step : Exp (.scalar .u32)) (bodyFn : Exp (.scalar .u32) → ShaderM Unit) : ShaderM Unit := do
  let varName ← freshVar "i"
  let (_, bodyStmts) ← captureStmts (bodyFn (Exp.var varName))
  let loopVar : Exp (.scalar .u32) := Exp.var varName
  let condition := Exp.lt loopVar end_
  let update := Exp.add loopVar step
  emitStmt (Stmt.forLoop varName start condition update bodyStmts)

/-- Runtime loop alias for `loop`, named to pair with `unrollFor` so the
    meta-vs-runtime distinction is explicit at the call site:
    ```
    ShaderM.unrollFor 8 fun i => ...               -- meta-time, 8 inline copies
    ShaderM.runtimeFor 0 cacheLen 1 fun i => ...   -- runtime for-loop
    ```
    Mirrors CUDA C++ `for (int i = 0; ...)` (no `#pragma unroll`). -/
@[inline] def runtimeFor
    (start end_ step : Exp (.scalar .u32))
    (bodyFn : Exp (.scalar .u32) → ShaderM Unit) : ShaderM Unit :=
  loop start end_ step bodyFn

/-- Block scope: emits `{ ... body ... }` so var declared inside the body
    is **block-scoped** in WGSL, allowing Naga/Tint/Vulkan-driver register
    allocator to reuse the physical register once the scope exits.

    This is the analog of CUDA `{ ... }` block scope for register-pressure
    reduction. For example, when a temporary is needed only inside a hot
    inner loop iteration, putting it in a `scope` instead of at function
    level lets the compiler reuse its register slot afterwards.

    Usage:
    ```lean
    ShaderM.scope do
      let tmp ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
      -- ... use tmp ...
      -- tmp's register is released at end of scope
    ```

    Note: Lean-side `let` bindings to `tmp` outside the do-block can still
    refer to the var name string, but reading from it via `Exp.var` after
    scope-exit is **undefined behavior** — the WGSL register is gone.
    Discipline: only use `tmp` inside the `scope` body. -/
def scope (body : ShaderM α) : ShaderM α := do
  let (result, bodyStmts) ← captureStmts body
  emitStmt (Stmt.block bodyStmts)
  return result

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

/-! ## Thread index helpers (CUDA-style shortcuts)

These mirror CUDA's `threadIdx.x`, `blockIdx.x`, etc. for the common
1D case so kernel code reads close to the CUDA original.  The vec3
versions (`localId`, `workgroupId`) are still available when 2D/3D
index components are needed.
-/

/-- `threadIdx.x` — local invocation X coordinate (= "tid" in CUDA kernels). -/
@[inline] def tidX : ShaderM (Exp (.scalar .u32)) := do
  return Exp.vec3X (← localId)

/-- `threadIdx.y` -/
@[inline] def tidY : ShaderM (Exp (.scalar .u32)) := do
  return Exp.vec3Y (← localId)

/-- `threadIdx.z` -/
@[inline] def tidZ : ShaderM (Exp (.scalar .u32)) := do
  return Exp.vecZ (← localId)

/-- `blockIdx.x` — workgroup X coordinate. -/
@[inline] def bidX : ShaderM (Exp (.scalar .u32)) := do
  return Exp.vec3X (← workgroupId)

/-- `blockIdx.y` -/
@[inline] def bidY : ShaderM (Exp (.scalar .u32)) := do
  return Exp.vec3Y (← workgroupId)

/-- `blockIdx.z` -/
@[inline] def bidZ : ShaderM (Exp (.scalar .u32)) := do
  return Exp.vecZ (← workgroupId)

/-! ### Warp / sub-warp index decomposition

NVIDIA warps are 32 lanes.  These helpers compute `laneId` (`tid & 31`)
and `warpId` (`tid >>> 5`) once and return them as `Exp` values, so
kernel code that uses them many times shares one PTX register.  The
inner expressions are simple enough that lean codegen folds the
extraction efficiently — no `let'` needed at the use site.

Sub-warp partition (used in V8/V11 attention vec kernels) splits a
warp into N-lane groups via `subWarpSplit n`.
-/

/-- Lane index within the warp (0..31).  Equivalent to CUDA's
    `threadIdx.x & 31` when threads are 1D, or `threadIdx.x % WARP_SIZE`. -/
@[inline] def laneId : ShaderM (Exp (.scalar .u32)) := do
  return (← tidX) &&& (Exp.litU32 31)

/-- Warp index within the workgroup.  Equivalent to CUDA's
    `threadIdx.x / WARP_SIZE` (= `>> 5`).  For 2D blocks where
    `threadIdx.y` selects warps, use `tidY` directly. -/
@[inline] def warpId : ShaderM (Exp (.scalar .u32)) := do
  return (← tidX) >>> (Exp.litU32 5)

/-- Split a warp's lane into `(subWarp, subLane)` for `nthreads_KQ`-style
    sub-warp partitioning where each sub-warp handles a different K
    position in parallel.  Returns `(laneId / n, laneId % n)`.

    Example (V11 sub-warp partition with `n = 8`):
    ```
    let (sw, sl) ← ShaderM.subWarpSplit 8
    -- sw ∈ [0, 4), sl ∈ [0, 8)
    let kPos := warpBase + sw * 8 + iKQ0    -- this sub-warp's K
    let dimBase := sl * 32                  -- this lane's D slice
    ```

    Mirrors llama.cpp's
    `(threadIdx.x & ~(nthreads_KQ-1), threadIdx.x % nthreads_KQ)`. -/
@[inline] def subWarpSplit (n : Nat) :
    ShaderM (Exp (.scalar .u32) × Exp (.scalar .u32)) := do
  let lane ← laneId
  let sw := lane / (Exp.litU32 n)
  let sl := lane &&& (Exp.litU32 (n - 1))   -- assumes n is power of 2
  return (sw, sl)

/-! ### Warp-level reductions

`warpReduceSum n e` reduces `e` across the `n`-lane group within the warp:
- `n = 32` (full warp) → emits a single `subgroupAdd` (5-shfl on PTX, hardware
  wide reduction on WGSL).
- `n = 8` (sub-warp, eg. nthreads_KQ from llama.cpp) → emits 3 shfl-xor
  by 1, 2, 4.  All 8 lanes end with the sum of their 8-lane group.
- Other `n` (must be power of 2, ≤ 32) → log2(n) shfl-xor butterflies.

After the call, every lane in the n-lane group holds the same reduced
value.  Mirrors llama.cpp's `warp_reduce_sum<nthreads>` template.

Example (V11-like sub-warp dot product):
```
let partialVar ← ShaderM.mutVar (.scalar .f32) 0
ShaderM.unrollFor dimsPerLane fun k => partialVar +↦ q[k]! * kVec[k]!
let sum ← ShaderM.warpReduceSum 8 partialVar.read   -- 3 shfl
```
-/

/-- Sum-reduce `e` across an n-lane group within the warp.  Uses
    `Exp.subgroupAdd` when `n = 32`, otherwise emits `log2(n)` butterfly
    shuffles via `Exp.subgroupShuffleXor`.

    Pre: `n` is a power of 2 and `1 ≤ n ≤ 32`.

    Each lane in the n-lane group receives the same sum.  Cost: log2(n)
    shfl + n−1 add per lane (so a sub-warp reduce of 8 lanes is 3 shfl
    + 3 add — half the cost of the full 32-lane reduce, useful when only
    8-lane groups need to coordinate). -/
def warpReduceSum (n : Nat) (e : Exp ty) : ShaderM (Exp ty) := do
  if n = 32 then
    -- Full warp: one subgroup primitive.
    let v ← let' ty (Exp.subgroupAdd e)
    return v
  else
    -- Butterfly via xor 1, 2, 4, ... up to n/2.
    let mut acc ← let' ty e
    let mut step := 1
    while step < n do
      let next ← let' ty (acc + Exp.subgroupShuffleXor acc (Exp.litU32 step))
      acc := next
      step := step * 2
    return acc

/-- Max-reduce variant: `warpReduceMax n e`.  Same shfl-xor butterfly,
    using `Exp.max` instead of `+`.  After the call, every lane in the
    group holds the same max. -/
def warpReduceMax (n : Nat) (e : Exp (.scalar .f32)) :
    ShaderM (Exp (.scalar .f32)) := do
  let mut acc ← let' (.scalar .f32) e
  let mut step := 1
  while step < n do
    let next ← let' (.scalar .f32)
                  (Exp.max acc (Exp.subgroupShuffleXor acc (Exp.litU32 step)))
    acc := next
    step := step * 2
  return acc

-- ============================================================================
-- Buffer Operations
-- ============================================================================

/-- Read from a global storage buffer at index
    Note: You need to provide the element type explicitly -/
def readBuffer {ty : WGSLType} {n : Nat} (bufferName : String) (idx : Exp (.scalar .u32)) : ShaderM (Exp ty) :=
  return Exp.index (Exp.var bufferName : Exp (.array ty n)) idx

/-- Read a single byte (zero-extended to u32) from a buffer declared as
    `array<u32, n>`, addressed by *byte* index.  Lowers to one `ld.global.u8`
    on CUDA; emulated via u32-load+shift+mask on WGSL.  Essential for Q6_K
    scale reads (avoids issuing a full u32 load per byte). -/
def readBufferByte {n : Nat} (bufferName : String) (byteIdx : Exp (.scalar .u32))
    : ShaderM (Exp (.scalar .u32)) :=
  return Exp.loadByteFromU32Buf (n := n) bufferName byteIdx

/-- Read a halfword (16 bits, zero-extended to u32) from a buffer declared
    as `array<u32, n>`, addressed by *byte* index.  Lowers to `ld.global.u16`
    on CUDA.  Use for fp16 block scales. -/
def readBufferU16 {n : Nat} (bufferName : String) (byteIdx : Exp (.scalar .u32))
    : ShaderM (Exp (.scalar .u32)) :=
  return Exp.loadU16FromU32Buf (n := n) bufferName byteIdx

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

/-- Declare a read-only storage buffer.

    Semantically equivalent to `declareInputBuffer` on WGSL, but on CUDA this
    emits `ld.global.nc.*` (the read-only L1/tex cache hint, aka `__ldg`).
    Use for weight matrices and other buffers the kernel never writes to.
    It is a logical error to call `writeBuffer` on the returned name. -/
def declareReadOnlyBuffer (name : String) (ty : WGSLType) : ShaderM String := do
  modify fun s => { s with declaredBuffers := s.declaredBuffers ++ [(name, ty, .read)] }
  return name

/-- Declare a read-only *buffer array* binding: N separate runtime-sized
    storage buffers addressable by runtime layer index.  The host side must
    pass a device-pointer-table — `N × 8 bytes` holding each layer's
    `CUdeviceptr` — as the kernel argument bound to `name`.  Inside the
    kernel, read elements with `readBufferArray name layerIdx elemIdx`.

    This is the primitive that lets a *single* kernel iterate over all 42
    transformer layers, collapsing 42 dispatches into 1.  The existing
    per-layer kernels remain; use `bufferArray` only when a kernel actually
    wants to fuse across layers. -/
def declareInputBufferArray (name : String) (elemTy : WGSLType) (n : Nat) : ShaderM String := do
  modify fun s => { s with declaredBuffers :=
    s.declaredBuffers ++ [(name, .bufferArray elemTy n, .read)] }
  return name

/-- Read element `elemIdx` from the `bufIdx`-th buffer of a `bufferArray`.
    Emits a single-indirection load via the pointer table. -/
def readBufferArray {elemTy : WGSLType} {n : Nat}
    (name : String) (bufIdx elemIdx : Exp (.scalar .u32)) : ShaderM (Exp elemTy) :=
  return Exp.indexBuf (elemTy := elemTy) (n := n) (Exp.var name) bufIdx elemIdx

/-- Write `value` to `arr[bufIdx][elemIdx]` where `arr` is a `bufferArray`. -/
def writeBufferArray {ty : WGSLType}
    (name : String) (bufIdx elemIdx : Exp (.scalar .u32)) (value : Exp ty) : ShaderM Unit :=
  emitStmt (Stmt.assignIndexBuf name bufIdx elemIdx ty value)

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

/-- Mixed-precision variant of `matrixMultiplyAccumulate`: A and B use
    `inSt`, C / D use `outSt`. This matches NVIDIA cooperative matrix's
    native `(f16, f16) → f32` config. -/
def matrixMultiplyAccumulateMixed
    {inSt outSt : ScalarType} {m k n : Nat}
    (resultArrayName : String)
    (resultIndex : Nat)
    (leftArrayName : String)
    (leftIndex : Nat)
    (rightArrayName : String)
    (rightIndex : Nat)
    : ShaderM Unit := do
  let leftMatTy := WGSLType.subgroupMatrixLeft inSt m k
  let rightMatTy := WGSLType.subgroupMatrixRight inSt k n
  let resultMatTy := WGSLType.subgroupMatrixResult outSt m n

  let leftArr : Exp (.array leftMatTy leftIndex) := Exp.var leftArrayName
  let rightArr : Exp (.array rightMatTy rightIndex) := Exp.var rightArrayName
  let resultArr : Exp (.array resultMatTy resultIndex) := Exp.var resultArrayName

  let leftMat := Exp.index leftArr (Exp.litU32 leftIndex)
  let rightMat := Exp.index rightArr (Exp.litU32 rightIndex)
  let accMat := Exp.index resultArr (Exp.litU32 resultIndex)

  let mulAccExpr := Exp.subgroupMatrixMultiplyAccumulateMixed leftMat rightMat accMat
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

/-! ## Typed mutable variables

`MutVar ty` wraps a private (PTX-register / WGSL `var`) variable with its
type carried in Lean.  This closes the second cognitive gap from porting
CUDA C++ — operator overloading in Step 1 still needed `(Exp.var name : Exp _)`
type annotations everywhere.  With `MutVar`, `v.read + x` and `v +↦ x`
type-check directly because `read` returns a typed `Exp ty`.

Idiomatic use:
```
let acc ← ShaderM.mutVar (.scalar .f32) 0
for k in [0:8] do
  acc +↦ qVec[k]! * kVec[k]!     -- in-place +=
let final := acc.read
```
-/

/-- Mutable shader-side variable carrying its type in Lean.  Construct via
    `ShaderM.mutVar` — never by hand (the contained name must be one
    emitted by `ShaderM.var` so PTX/WGSL codegen sees a declaration). -/
structure MutVar (ty : WGSLType) where
  name : String
  deriving Repr

namespace MutVar

/-- Read the current value as a typed `Exp ty`. -/
@[inline] def read (v : MutVar ty) : Exp ty := Exp.var v.name

/-- Overwrite the variable with `x`. -/
@[inline] def write (v : MutVar ty) (x : Exp ty) : ShaderM Unit :=
  ShaderM.assign v.name x

/-- In-place add: `v += x`.  Sugar for `v.write (v.read + x)`. -/
@[inline] def addAssign (v : MutVar ty) [HAdd (Exp ty) (Exp ty) (Exp ty)]
    (x : Exp ty) : ShaderM Unit :=
  ShaderM.assign v.name (v.read + x)

/-- In-place multiply: `v *= x`. -/
@[inline] def mulAssign (v : MutVar ty) [HMul (Exp ty) (Exp ty) (Exp ty)]
    (x : Exp ty) : ShaderM Unit :=
  ShaderM.assign v.name (v.read * x)

/-- In-place subtract: `v -= x`. -/
@[inline] def subAssign (v : MutVar ty) [HSub (Exp ty) (Exp ty) (Exp ty)]
    (x : Exp ty) : ShaderM Unit :=
  ShaderM.assign v.name (v.read - x)

end MutVar

/-- Mutating-add operator.  `v +↦ x` ≡ `MutVar.addAssign v x`. -/
infixl:55 " +↦ " => MutVar.addAssign

/-- Mutating-multiply operator. -/
infixl:60 " *↦ " => MutVar.mulAssign

/-- Mutating-write operator (≡ `MutVar.write`).  `v ↦= x`. -/
infix:55 " ↦= " => MutVar.write

namespace ShaderM

/-- Declare a typed mutable variable.  Use `v.read` / `v.write` /
    `v +↦ x` / `v ↦= x`.  The successor to `ShaderM.var`.

    Example:
    ```
    let acc ← ShaderM.mutVar (.scalar .f32) 0
    for k in [0:dimsPerLane] do
      acc +↦ qVec[k]! * kVec[k]!
    ```
-/
def mutVar (ty : WGSLType) (init : Exp ty) : ShaderM (MutVar ty) := do
  let name ← freshVar "v"
  emitStmt (Stmt.varDecl name ty (some ⟨ty, init⟩))
  return ⟨name⟩

end ShaderM

end Hesper.WGSL.Monad
