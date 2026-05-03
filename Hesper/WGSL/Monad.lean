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

/-! ### LICM helpers (loop invariant code motion)

Approximate "free variable" check: serialise an Exp via `Exp.toWGSL` and
look for the variable name as a whole token (`name` not preceded/followed
by an alphanumeric or underscore).  This is correct for ShaderM-emitted
names because they're unique fresh names like `v3` / `i7` — there's no
risk of accidental match against a substring of another identifier.

Why string-match rather than a structural traversal: `Exp` has 50+
constructors and many of them hold sub-`Exp`s.  A structural
`Exp.containsVar` would be a lot of pattern matching that would have to
stay in sync as new `Exp` cases are added.  `toWGSL` is the existing
single source of truth that walks every sub-Exp; reusing it for the
free-var check costs O(n) extra string work per stmt but is robust to
Exp evolution.  In practice the runtime impact is negligible because
LICM only runs at kernel-emission time, not on the GPU. -/

/-- True if `name` appears as a standalone identifier in `s`.
    Standalone = not surrounded by alphanumeric / underscore chars,
    so `i7` doesn't match `i7x` or `xi7`.

    Implementation note: works on the WGSL serialisation of an Exp
    rather than recursing structurally over `Exp` (which has 50+
    constructors). -/
private def usesNameInString (s : String) (name : String) : Bool := Id.run do
  if name.length = 0 then return false
  -- Walk byte-by-byte using String.find / substring
  let parts := s.splitOn name
  if parts.length ≤ 1 then return false
  -- For each gap between occurrences, check the chars adjacent.
  -- The substring `name` appears at positions where parts joined back
  -- means: prev part's last char must NOT be ident-like, and next
  -- part's first char must NOT be ident-like.
  let isIdent : Char → Bool := fun c => c.isAlphanum || c = '_'
  let mut i := 0
  for part in parts do
    if i + 1 < parts.length then
      let prevOk := part.isEmpty || !isIdent part.back
      let nextPart := parts[i + 1]!
      let nextOk := nextPart.isEmpty || !isIdent nextPart.front
      if prevOk && nextOk then return true
    i := i + 1
  return false

/-- True if `name` appears as a standalone identifier in `e`. -/
private def Exp.usesName {ty} (e : Exp ty) (name : String) : Bool :=
  usesNameInString (Exp.toWGSL e) name

/-- True if a Stmt's "value computation" depends on any name in `defs`.
    For `varDecl`, this is the init expression.  For everything else,
    we conservatively return true (don't hoist).  We only LICM `varDecl`
    nodes — assigns, control flow, and barriers stay inside. -/
private def stmtDependsOnAny (st : Stmt) (defs : List String) : Bool :=
  match st with
  | .varDecl _ _ none => false  -- declaration without init: pure
  | .varDecl _ _ (some ⟨_, init⟩) => defs.any (fun n => Exp.usesName init n)
  | _ => true  -- conservative: keep inside loop

/-- Names of variables assigned-to (post-decl) anywhere in `body`.
    These cannot be hoisted because their declaration is conceptually
    "the latest write before any read", and moving the decl outside the
    loop would expose stale values across iterations. -/
private def collectAssignTargets : List Stmt → List String
  | [] => []
  | st :: rest =>
    let here := match st with
      | .assign name _ _      => [name]
      | .assignIndex _ _ _ _  => []  -- writes go through array, not var
      | .assignIndexBuf _ _ _ _ _ => []
      | .forLoop _ _ _ _ inner    => collectAssignTargets inner
      | .ifStmt _ thenB elseB => collectAssignTargets thenB ++ collectAssignTargets elseB
      | .block inner          => collectAssignTargets inner
      | _ => []
    here ++ collectAssignTargets rest

/-- LICM: split body into (hoist-out, keep-inside) given the loop var.
    A `varDecl` is hoisted iff:
    1. its init doesn't reference the loop var, OR
    2. any variable assigned inside the loop (including names declared
       outside the loop body — they're still loop-variant if reassigned),
       OR
    3. any varDecl declared and kept inside the loop so far.

    Plus: the variable being declared must itself never be assigned-to,
    otherwise its decl must stay where any subsequent `assign` references
    a single, monotonic write site.

    `loopVariant` is the union of (loop var) ∪ (assigned vars across the
    body, recursive into nested loops/ifs). -/
private def licmSplit (loopVar : String) (body : List Stmt) :
    List Stmt × List Stmt := Id.run do
  let mut hoisted : List Stmt := []
  let mut kept : List Stmt := []
  -- Loop-variant set: loop var + everything that's assigned-to anywhere
  -- in the body (so externally-declared mutable vars updated inside the
  -- loop are correctly treated as loop-variant).
  let assignTargets := collectAssignTargets body
  let mut loopVariant : List String := loopVar :: assignTargets
  for st in body do
    match st with
    | .varDecl name _ _ =>
      let isAssignedHere := assignTargets.contains name
      if isAssignedHere || stmtDependsOnAny st loopVariant then
        loopVariant := name :: loopVariant
        kept := st :: kept
      else
        hoisted := st :: hoisted
    | _ =>
      kept := st :: kept
  return (hoisted.reverse, kept.reverse)

/-- Higher-order loop: pass loop variable as Exp.  Now applies LICM:
    `varDecl`s in the body whose init expression doesn't reference the
    loop variable (or any other inner-loop binding) are hoisted before
    the for-loop.  This automates the manual `let'`-outside-the-loop
    pattern that V9 / earlier kernels used by hand.

    Usage: loop start end step fun i => do { ... use i ... } -/
def loop (start : Exp (.scalar .u32)) (end_ : Exp (.scalar .u32)) (step : Exp (.scalar .u32)) (bodyFn : Exp (.scalar .u32) → ShaderM Unit) : ShaderM Unit := do
  let varName ← freshVar "i"
  let (_, bodyStmts) ← captureStmts (bodyFn (Exp.var varName))
  let loopVar : Exp (.scalar .u32) := Exp.var varName
  let condition := Exp.lt loopVar end_
  let update := Exp.add loopVar step
  -- LICM is opt-in via `loopWithLICM` — this default `loop` keeps body
  -- unchanged.  Reason: a sound LICM needs to track shared-memory writes
  -- and inter-iteration carries that are easy to miss; the experimental
  -- pass broke V2/V3 parity.  Available as licmSplit for callers who
  -- understand the constraints (see loopWithLICM below).
  emitStmt (Stmt.forLoop varName start condition update bodyStmts)

/-- Variant of `loop` that applies LICM to hoist `varDecl`s whose init
    is loop-invariant out of the for-loop.  Use only when the body
    contains no shared-memory writes that other iterations read, no
    inter-iteration carries via externally-declared mutable vars beyond
    those captured by `assign`, and only `varDecl` (no `if`/inner
    `for`/`barrier`) sites that you want hoisted.

    Semantics-changing in subtle ways — verify bit-parity before merging
    a kernel that uses this. -/
def loopWithLICM (start end_ step : Exp (.scalar .u32))
    (bodyFn : Exp (.scalar .u32) → ShaderM Unit) : ShaderM Unit := do
  let varName ← freshVar "i"
  let (_, bodyStmts) ← captureStmts (bodyFn (Exp.var varName))
  let loopVar : Exp (.scalar .u32) := Exp.var varName
  let condition := Exp.lt loopVar end_
  let update := Exp.add loopVar step
  let (hoisted, kept) := licmSplit varName bodyStmts
  for st in hoisted do
    emitStmt st
  emitStmt (Stmt.forLoop varName start condition update kept)

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

/-- `unrollFor` + per-iter `scope`. Each of the `n` inline copies of the
    body lives in its own `{ ... }` block, so any `ShaderM.var` declared
    inside is block-scoped and the WGSL→PTX backend can reuse the
    physical register across iters.

    Collapses the V8/V9/V11 idiom

    ```lean
    for i in [0:n] do ShaderM.scope do
      ...
    ```

    into

    ```lean
    ShaderM.unrollForScoped n fun i => do
      ...
    ```

    Mirrors CUDA C++ `#pragma unroll for (int i = ...)` where each
    unrolled body is implicitly its own register-allocation scope. -/
@[inline] def unrollForScoped (n : Nat) (body : Nat → ShaderM Unit) : ShaderM Unit :=
  (List.range n).forM fun i => scope (body i)

-- ============================================================================
-- Synchronization
-- ============================================================================

/-- Workgroup barrier (synchronization) -/
def barrier : ShaderM Unit :=
  emitStmt (Stmt.exprStmt Exp.workgroupBarrier)

/-- Warp-level barrier (CUDA `__syncwarp()`).  PTX backend emits the
    cheap `bar.warp.sync 0xFFFFFFFF`; WGSL backend falls back to
    `workgroupBarrier()`. Use when only intra-warp ordering is needed
    — eg. between Phase 1 (KQ dot) and Phase 2a (cross-sub-warp max
    reduce) in the flash-attn vec kernel where all 32 lanes of a warp
    must finish writing before reading. -/
def warpBarrier : ShaderM Unit :=
  emitStmt (Stmt.exprStmt Exp.warpBarrier)

/-- Pure: raw u64 pointer to element `idx` of a global buffer.
    `elemSize` is the byte size of one element (typically 4 for u32/f32).
    Used as the global-address operand for `cpAsync`. CUDA-only. -/
def bufferAddr (bufName : String) (elemSize : Nat)
               (idx : Exp (.scalar .u32)) : Exp (.scalar .u64) :=
  Exp.bufferAddr bufName elemSize idx

/-- ── cp.async (sm_80+) ──
    Issue an async global→shared memory copy of `bytes` bytes
    (must be 4, 8, or 16). Non-blocking — completion synchronised
    via `cpAsyncCommit` + `cpAsyncWait`. Used by llama.cpp's MMQ
    pipeline to overlap the next K-iteration's load with the
    current K-iteration's compute. WGSL backend has no equivalent. -/
def cpAsync (smemAddr : Exp (.scalar .u32))
            (globalAddr : Exp (.scalar .u64))
            (bytes : Nat) : ShaderM Unit :=
  emitStmt (Stmt.exprStmt (Exp.cpAsyncCgSharedGlobal smemAddr globalAddr bytes))

/-- `cp.async.commit_group` — bundle all preceding `cpAsync` issues
    by this thread into one group for later `cpAsyncWait`. -/
def cpAsyncCommit : ShaderM Unit :=
  emitStmt (Stmt.exprStmt Exp.cpAsyncCommitGroup)

/-- `cp.async.wait_group N` — block this thread until all but the most
    recent N committed groups have completed. `N=0` waits for all. -/
def cpAsyncWait (n : Nat) : ShaderM Unit :=
  emitStmt (Stmt.exprStmt (Exp.cpAsyncWaitGroup n))

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
    -- Full warp: one subgroup primitive.  Don't `let'` — caller can pin
    -- if they want, but for one-shot consumers (eg. `Exp.select inBounds
    -- (warpReduceSum ...) ...`) the extra register hop is wasteful.
    return Exp.subgroupAdd e
  else
    -- Butterfly via xor 1, 2, 4, ... up to n/2.  Each step IS pinned
    -- because the result of step k feeds into step k+1's operands twice
    -- (acc + shfl(acc, mask)) — without let' the AST gets re-traversed.
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

/-- One step of online softmax (Milakov & Gimelshein 2018 / FlashAttention).
    Given the running max `kqMax`, running sum `kqSum`, and a new score
    `kqScore`, computes:

    ```
    kqMaxNew  = max(kqMax, kqScore)
    scale     = exp(kqMax - kqMaxNew)        -- multiplier for old VKQ accumulator
    kqExp     = exp(kqScore - kqMaxNew)      -- per-thread weight for current K
    kqSumNew  = kqSum * scale + kqExp
    kqMax := kqMaxNew                        -- updated in-place
    kqSum := kqSumNew                        -- updated in-place
    ```

    Returns `(kqMaxNew, scale, kqExp)`. The caller rescales their VKQ
    accumulators by `scale` and accumulates `kqExp * V` for this K
    position.

    `kqMaxName`/`kqSumName` are the **names** of the running registers
    so we can `assign` them in place.  The new max is also returned as
    an `Exp` so the caller can use it without an extra `Exp.var` lookup.

    Mirrors the inner-loop accumulation in llama.cpp's
    `flash_attn_ext_vec` (lines 273, 287-291 in fattn-vec.cuh). -/
def softmaxOnlineUpdate
    (kqMaxName kqSumName : String) (kqScore : Exp (.scalar .f32)) :
    ShaderM (Exp (.scalar .f32) × Exp (.scalar .f32) × Exp (.scalar .f32)) := do
  let kqMax : Exp (.scalar .f32) := Exp.var kqMaxName
  let kqSum : Exp (.scalar .f32) := Exp.var kqSumName
  let kqMaxNew ← let' (.scalar .f32) (Exp.max kqMax kqScore)
  let scale    ← let' (.scalar .f32) (Exp.exp (Exp.sub kqMax kqMaxNew))
  let kqExp    ← let' (.scalar .f32) (Exp.exp (Exp.sub kqScore kqMaxNew))
  assign kqMaxName kqMaxNew
  assign kqSumName (Exp.add (Exp.mul kqSum scale) kqExp)
  return (kqMaxNew, scale, kqExp)

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

/-- 128-bit (4× u32) wide read from a buffer declared `array<u32, n>`,
    addressed by *u32* index.  The starting index must be 4-aligned and the
    buffer pointer 16-byte aligned (caller's responsibility — typical
    callers iterate by `pk * 4`).

    On CUDA this lowers to a single `ld.global.nc.v4.u32` instruction
    (one MIO op delivering 16 bytes), 4× more efficient than four scalar
    `ld.global.u32` reads in MIO-pipe-saturated kernels (FlashAttn V11).
    On WGSL it emulates as four scalar reads.

    Returns four `Exp (.scalar .u32)` referring to fresh declared vars.
    Each value can be used independently with `Exp.unpack2x16float` etc. -/
def readBufferU32x4 (bufferName : String) (u32Idx : Exp (.scalar .u32))
    : ShaderM (Exp (.scalar .u32) × Exp (.scalar .u32) ×
               Exp (.scalar .u32) × Exp (.scalar .u32)) := do
  let n0 ← freshVar "v4_0"
  let n1 ← freshVar "v4_1"
  let n2 ← freshVar "v4_2"
  let n3 ← freshVar "v4_3"
  emitStmt (Stmt.varDeclLdV4U32 n0 n1 n2 n3 bufferName u32Idx)
  return (Exp.var n0, Exp.var n1, Exp.var n2, Exp.var n3)

/-- 128-bit (4× f32) wide read from shared memory.  `f32Idx` must be
    4-aligned.  CUDA lowers to one `ld.shared.v4.f32`; WGSL emulates as
    four scalar reads.

    Used to relieve MIO-pipe saturation on shared-memory traffic
    (FlashAttn V11 partial aggregation: 4× LDS instruction count
    reduction). -/
def readWorkgroupF32x4 (sharedName : String) (f32Idx : Exp (.scalar .u32))
    : ShaderM (Exp (.scalar .f32) × Exp (.scalar .f32) ×
               Exp (.scalar .f32) × Exp (.scalar .f32)) := do
  let n0 ← freshVar "lds4_0"
  let n1 ← freshVar "lds4_1"
  let n2 ← freshVar "lds4_2"
  let n3 ← freshVar "lds4_3"
  emitStmt (Stmt.varDeclLdV4F32Shared n0 n1 n2 n3 sharedName f32Idx)
  return (Exp.var n0, Exp.var n1, Exp.var n2, Exp.var n3)

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

/-! ## Fragment-as-local-variable helpers (real-CUDA-style WMMA)

The array-of-fragments helpers above (`loadMatrixLeft`, `matrixMultiplyAccumulate`,
`storeMatrixResult`) match WGSL's `array<subgroup_matrix_left, N>` model —
nice for blocked algorithms but awkward to lower to PTX since fragments live
in *registers*, not memory. The helpers below match the more direct
`nvcuda::wmma::fragment a_frag; load_matrix_sync(a_frag, ...);` style: each
fragment is a single named variable, with no array indirection. This is the
recommended path for new WMMA kernels — the array helpers remain for
backward-compat and WGSL-targeted code. -/

/-- Declare a left fragment (`subgroup_matrix_left<st, m, k>`) bound to a
    name, initialized via a load from `bufferName` at byte-offset `offset`
    with row-major `stride` (in elements). -/
def loadFragmentLeft
    (st : ScalarType) (m k : Nat)
    (name : String)
    (bufferName : String)
    (offset : Exp (.scalar .u32))
    (stride : Exp (.scalar .u32))
    : ShaderM Unit := do
  let matTy := WGSLType.subgroupMatrixLeft st m k
  let loadExpr := Exp.subgroupMatrixLoad bufferName offset (Exp.litBool false) stride
  emitStmt (Stmt.varDecl name matTy (some ⟨matTy, loadExpr⟩))

/-- Declare a right fragment, loaded col-major (the WGSL `subgroupMatrixLoadRight`
    instruction, which lowers to PTX `wmma.load.b...col.f16`). -/
def loadFragmentRight
    (st : ScalarType) (k n : Nat)
    (name : String)
    (bufferName : String)
    (offset : Exp (.scalar .u32))
    (stride : Exp (.scalar .u32))
    : ShaderM Unit := do
  let matTy := WGSLType.subgroupMatrixRight st k n
  let loadExpr := Exp.subgroupMatrixLoadRight bufferName offset (Exp.litBool false) stride
  emitStmt (Stmt.varDecl name matTy (some ⟨matTy, loadExpr⟩))

/-- Declare a result fragment initialized to zero. Maps to PTX
    `mov.f32 %f0..7, 0;` for the f32 case (8 regs). -/
def declareFragmentResultZero
    (st : ScalarType) (m n : Nat)
    (name : String)
    : ShaderM Unit := do
  let matTy := WGSLType.subgroupMatrixResult st m n
  emitStmt (Stmt.varDecl name matTy (some ⟨matTy, Exp.subgroupMatrixZeroResult⟩))

/-- `cName ← cName * (a × b)`. Uses the existing `Exp.subgroupMatrix-
    MultiplyAccumulate`; lowers to one `wmma.mma.sync` instruction and
    rebinds `cName` to the result fragment. -/
def fragmentMultiplyAccumulate
    (st : ScalarType) (m k n : Nat)
    (cName aName bName : String)
    : ShaderM Unit := do
  let aMatTy := WGSLType.subgroupMatrixLeft st m k
  let bMatTy := WGSLType.subgroupMatrixRight st k n
  let cMatTy := WGSLType.subgroupMatrixResult st m n
  let aExp : Exp aMatTy := Exp.var aName
  let bExp : Exp bMatTy := Exp.var bName
  let cExp : Exp cMatTy := Exp.var cName
  let mma := Exp.subgroupMatrixMultiplyAccumulate aExp bExp cExp
  emitStmt (Stmt.assign cName cMatTy mma)

/-- Store a result fragment to a buffer at byte-offset `offset` with
    row-major `stride` (in elements). -/
def storeFragmentResult
    (st : ScalarType) (m n : Nat)
    (cName : String)
    (bufferName : String)
    (offset : Exp (.scalar .u32))
    (stride : Exp (.scalar .u32))
    : ShaderM Unit := do
  let cMatTy := WGSLType.subgroupMatrixResult st m n
  let cExp : Exp cMatTy := Exp.var cName
  let storeExpr := Exp.subgroupMatrixStore bufferName offset cExp (Exp.litBool false) stride
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

/-! ## Buffer pointers (`Ptr ty`)

`Ptr ty` is a `(bufferName, offset)` pair carrying its element type so
the `load` / `store` calls don't need explicit `(ty := ...)` annotations
each time.  Mirrors CUDA C++ `T *p` arithmetic — particularly the common
attention/matmul pattern:

```cpp
const float *K = K_base + kvHead * maxSeq * D;
for (int k = 0; k < cacheLen; ++k) {
    sum += *K * q;
    K += D;
}
```

In ShaderM:
```
let K ← ShaderM.ptr (.scalar .f32) "k_cache" (kvHead * maxSeqLen * D)
ShaderM.runtimeFor 0 cacheLen 1 fun _ => do
  let v ← K.load
  acc +↦ v * q
  K := K.advance D    -- value-level advance (Ptr is immutable)
```

For mutating advance inside a runtime loop, use `MutPtr` (declared
below).  For pure pointer arithmetic at meta-time, use `Ptr.atOffset`.

`Ptr` is intentionally pure (no ShaderM effect).  The constructor
`ShaderM.ptr` only computes a `let'`-bound base offset for safety,
which is the common case.
-/

/-- Buffer pointer: typed (buffer name, current u32 offset, declared
    array size) triple.

    - `buf`: buffer name registered via `declareInputBuffer` etc.
    - `offset`: current element index (NOT byte) into that buffer.
    - `bufLen`: the buffer's declared array length (passed to
      `readBuffer`'s `{n}` for the WGSL `array<ty, n>` type).  Set this
      to the same value used at `declareInputBuffer` time.

    `Ptr` is value-level (no ShaderM effect needed for arithmetic).
    Use `ShaderM.ptr` to construct one with a `let'`-stored base. -/
structure Ptr (ty : WGSLType) where
  buf : String
  offset : Exp (.scalar .u32)
  bufLen : Nat

namespace Ptr

/-- Element-wise read at the current pointer offset.  Mirrors CUDA `*p`. -/
@[inline] def load (p : Ptr ty) : ShaderM (Exp ty) :=
  ShaderM.readBuffer (n := p.bufLen) p.buf p.offset

/-- Element-wise write at the current pointer offset.  Mirrors CUDA `*p = x`. -/
@[inline] def store (p : Ptr ty) (x : Exp ty) : ShaderM Unit :=
  ShaderM.writeBuffer p.buf p.offset x

/-- Advance the pointer by `delta` elements, returning a new `Ptr`.
    Mirrors CUDA `p + delta` (or `p += delta` if you reassign).  Pure
    — no shader-side effect. -/
@[inline] def advance (p : Ptr ty) (delta : Exp (.scalar .u32)) : Ptr ty :=
  ⟨p.buf, p.offset + delta, p.bufLen⟩

/-- Read at `p[k]` without permanently advancing.  Mirrors CUDA `p[k]`.
    Composes well with `unrollFor`:
    ```
    ShaderM.unrollFor n fun k =>
      let v ← (p.atOffset (Exp.litU32 k)).load
      ...
    ```
-/
@[inline] def atOffset (p : Ptr ty) (extra : Exp (.scalar .u32)) : Ptr ty :=
  ⟨p.buf, p.offset + extra, p.bufLen⟩

end Ptr

namespace ShaderM

/-- Construct a `Ptr` with a base offset materialised in a register
    (via `let'`).  Use when the base offset is computed once (eg.
    `kvHead * maxSeqLen * D`) and the resulting pointer is then
    walked many times in inner loops.

    `bufLen` must match the array size declared via
    `declareInputBuffer` for `buf`.

    Example:
    ```
    let K ← ShaderM.ptr (.scalar .u32) "k_cache_f16" kvWords
              (kvHead * maxSeqLen * (D/2))
    -- K.offset is a single PTX register; subsequent advance/atOffset
    -- only emit `add` against this register.
    ```
-/
def ptr (ty : WGSLType) (buf : String) (bufLen : Nat)
    (baseOffset : Exp (.scalar .u32)) : ShaderM (Ptr ty) := do
  let off ← let' (.scalar .u32) baseOffset
  return ⟨buf, off, bufLen⟩

end ShaderM

/-- Mutable pointer — like `Ptr` but `offset` is a `var` (read/write
    register) instead of an `Exp`, so `advance` actually updates the
    register in-place rather than constructing a new value.

    Mirrors the CUDA outer-loop idiom

    ```cpp
    K += blockIdx.y * nthreads * nb11;          // initial
    for (int k = ...; k < kmax; k += step,
         K += step * nb11, V += step * nb21) {  // advance per iter
        sum += vec_dot_KQ(K + i_KQ * nb11, ...);// inner per-iter offset
    }
    ```

    Step 6's `Ptr` only handles the inner per-iter offset; `MutPtr`
    handles the outer per-iter pointer advance. -/
structure MutPtr (ty : WGSLType) where
  buf : String
  /-- Name of the u32 register that holds the current offset. Mutable —
      gets reassigned by `MutPtr.advance`. -/
  offsetVar : String
  bufLen : Nat
deriving Repr

namespace MutPtr

/-- Current offset as an `Exp.var` so it can be used in index
    arithmetic without materialising another register. -/
@[inline] def offset {ty : WGSLType} (p : MutPtr ty) : Exp (.scalar .u32) :=
  Exp.var p.offsetVar

/-- Read at the current offset. -/
@[inline] def load {ty : WGSLType} (p : MutPtr ty) : ShaderM (Exp ty) :=
  ShaderM.readBuffer (ty := ty) (n := p.bufLen) p.buf p.offset

/-- Read at `p.offset + extra` without advancing.  Mirrors CUDA `p[k]`
    inside an unrolled inner loop. -/
@[inline] def loadAt {ty : WGSLType} (p : MutPtr ty)
    (extra : Exp (.scalar .u32)) : ShaderM (Exp ty) :=
  ShaderM.readBuffer (ty := ty) (n := p.bufLen) p.buf (p.offset + extra)

/-- Advance the pointer by `delta`. The next `load` / `loadAt` reads
    from the new offset. Mirrors CUDA `p += delta`. -/
@[inline] def advance {ty : WGSLType} (p : MutPtr ty)
    (delta : Exp (.scalar .u32)) : ShaderM Unit :=
  ShaderM.assign p.offsetVar (p.offset + delta)

/-- Store at the current offset. -/
@[inline] def store {ty : WGSLType} (p : MutPtr ty) (value : Exp ty) :
    ShaderM Unit :=
  ShaderM.writeBuffer (ty := ty) p.buf p.offset value

/-- Snapshot to an immutable `Ptr` at the current offset.  Useful when
    handing the pointer to a helper that doesn't need `advance`. -/
@[inline] def freeze {ty : WGSLType} (p : MutPtr ty) : Ptr ty :=
  ⟨p.buf, p.offset, p.bufLen⟩

end MutPtr

namespace ShaderM

/-- Construct a `MutPtr` whose offset register is initialised from
    `baseOffset`. The resulting pointer can be `advance`d in-place
    inside an outer loop, matching CUDA's `K += stride` idiom.

    Example (V11 outer K loop):
    ```
    let K ← ShaderM.mutPtr (.scalar .u32) "k_cache_f16" kvWords
              (kvHead * maxSeqLen * (D/2) + laneId)
    ShaderM.runtimeFor splitStart splitEnd (Exp.litU32 wgSize) fun _ => do
      ...
      let kPacked ← K.loadAt (Exp.litU32 (pk * 32))
      ...
      K.advance (Exp.litU32 (wgSize * (D/2)))
    ```
-/
def mutPtr (ty : WGSLType) (buf : String) (bufLen : Nat)
    (baseOffset : Exp (.scalar .u32)) : ShaderM (MutPtr ty) := do
  let name ← var (.scalar .u32) baseOffset
  return ⟨buf, name, bufLen⟩

end ShaderM

/-- Typed register array — `n` named ShaderM vars that share a type and
    can be indexed by a meta-time `Nat`. Replaces the V8/V11 idiom

    ```lean
    let mut q0Vars : Array String := #[]
    for pk in [0:n] do
      let v ← ShaderM.var ty (init pk)
      q0Vars := q0Vars.push v
    -- later:
    let q0 : Exp _ := Exp.var q0Vars[pk]!
    ```

    with

    ```lean
    let q0 ← RegArray.mk ty n init
    -- later:
    let q0Exp := q0.get pk
    q0.set pk newVal
    ```

    The `Array String` field stays in Lean meta land — at codegen time
    it materialises as `n` separate `varDecl` stmts (one per slot).
    Mirrors CUDA's `T arr[N];` register array. -/
structure RegArray (ty : WGSLType) (n : Nat) where
  names : Array String
deriving Repr

namespace RegArray

/-- Read slot `i` as an `Exp`. Returns `Exp.litU32 0` if out-of-range
    (shouldn't happen in well-typed code; the bounds check is a safety
    net against meta-time index typos). -/
@[inline] def get {ty : WGSLType} {n : Nat} (a : RegArray ty n) (i : Nat) : Exp ty :=
  Exp.var (a.names[i]?.getD "RegArray.get/oob")

/-- Assign a new value to slot `i`. -/
@[inline] def set {ty : WGSLType} {n : Nat} (a : RegArray ty n) (i : Nat)
    (v : Exp ty) : ShaderM Unit :=
  ShaderM.assign (a.names[i]?.getD "RegArray.set/oob") v

end RegArray

namespace ShaderM

/-- Construct a `RegArray ty n` by emitting `n` `var` declarations.
    `init i` produces the initial value for slot `i`. -/
def regArray (ty : WGSLType) (n : Nat) (init : Nat → Exp ty) :
    ShaderM (RegArray ty n) := do
  let mut names : Array String := Array.empty
  for i in [0:n] do
    let nm ← var ty (init i)
    names := names.push nm
  return ⟨names⟩

end ShaderM

end Hesper.WGSL.Monad
