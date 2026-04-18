import Hesper.Circuit.IR
/-!
# Circuit DSL v2 — IR and type system (Phase B proposal)

This module is a **frontend-only** proposal for a Hesper DSL redesign.  It
introduces type-level memory scopes, ghost `StateToken`s for safe inplace
updates, TensorIR-style `Block`s with explicit `reads`/`writes`, and a
`Scatter` whose indices are proved in-range at the type level.

None of this module emits WGSL or PTX.  The eventual lowering target is
the existing `Hesper.WGSL.Monad.ShaderM`, reusing Circuit v1 passes.

## Design tenets (from the spec)

* **Zero-cost abstraction.**  All of this disappears at lowering time.
  `StateToken` has no runtime representation (no VRAM bytes, no CUDA
  call).  `CircuitTensor` erases to a plain `TensorRef` (an integer ID).
  The scope parameters are `Prop`-like phantom indices — they control
  which primitives typecheck, nothing more.
* **Dependent types.**  Shapes are `List Nat`.  Shape mismatches are
  caught at elaboration time.  `Scatter` indices carry a proof that
  they are `< dim`, either statically or via `Fin`-bounded refinement.
* **Inspiration:**
    - Spatial (memory-hierarchy in the type)
    - TVM TensorIR (`Block` with explicit read/write footprints)
    - Rust linear-ish state tokens (thread safe mutation ordering)

## Layer diagram

```
  User DSL   (this file)
     │
     │ elab / tactic
     ▼
  CircuitAST (inductive; this file)
     │
     │ toShaderM lowering (NOT in this file)
     ▼
  ShaderM / WGSL / PTX           (Circuit v1 lowering — reuse)
```
-/

namespace Hesper.CircuitV2

/-! ## 1. Scopes, dtypes, shapes -/

/-- Where a tensor physically lives on the device.

    | Scope | description                            | lowered to |
    |-------|----------------------------------------|------------|
    | VRAM  | global/device memory (`ggml` buffers)  | f32* / u32* in .global |
    | SRAM  | workgroup-shared memory                | `__shared__` / `var<workgroup>` |
    | Reg   | thread-local registers / variables     | `let x` / local `var` |

    The scope is a type parameter on `CircuitTensor`, so a primitive that
    requires a Reg input (e.g. `matmul` accumulator) won't accept a VRAM
    handle. -/
inductive Scope where
  | VRAM
  | SRAM
  | Reg
  deriving BEq, Repr, DecidableEq, Inhabited

/-- Minimal runtime dtype tag.  Kept as a data constructor (rather than a
    Lean `Type`) because we need to serialise / hash it in lowering. -/
inductive DType where
  | f32
  | f16
  | bf16
  | u32
  | i8
  | q4k | q6k | q8_0 | q8_1
  deriving BEq, Repr, DecidableEq, Inhabited

/-- Compile-time shape.  `List Nat` (not `Array`) so it is easy to pattern
    match in elaboration and still stays decidable.  The last dim is the
    "fast" dim in the usual row-major sense. -/
abbrev Shape := List Nat

/-- Total element count of a shape.  Used in safety side conditions. -/
def Shape.numel : Shape → Nat
  | []      => 1
  | d :: ds => d * Shape.numel ds

/-! ## 2. TensorRef

A `CircuitTensor` is a typed handle to a Circuit-managed tensor.  All
fields but `id` are phantom at runtime — but Lean uses them to rule out
bad programs.

We keep it as a `structure` (not `abbrev`) so that Lean prints the full
type in error messages. -/
structure CircuitTensor (s : Scope) (shape : Shape) (dt : DType) where
  /-- Monotone ID, allocated by the builder.  Two CircuitTensors with
      the same `id` refer to the same underlying buffer — the compiler
      uses this for CSE and aliasing. -/
  id : Nat
  deriving DecidableEq, Repr

/-! ## 3. StateToken

A `StateToken (s : Scope)` is a ghost value used to order **side-effecting**
primitives that mutate a buffer in `scope s`.  It has no runtime footprint.

Rule: any primitive that mutates a VRAM buffer (`Scatter`, `Store` into an
existing handle) consumes an old `StateToken VRAM` and produces a new
one.  Because Lean is a pure language the caller must thread the token
through sequential mutations — the single-use discipline recovers the
happens-before ordering.

At lowering time the tokens are erased and the mutations are emitted in
monadic-DSL order. -/
structure StateToken (s : Scope) where
  /-- Abstract tick count.  The builder stamps a fresh tick on each
      `Scatter`/`Store` output; the runtime relies on the builder having
      emitted operations in tick order. -/
  tick : Nat
  deriving DecidableEq, Repr

/-! ## 4. Safe index (refinement)

For `Scatter` into a tensor of shape `[..., dim]`, the write address
must be a `ScatterIdx dim`, i.e. a `Nat` with a proof it is `< dim`.

Why not `Fin dim`? Two reasons:
  * Shape dims are often large (`262144` for Gemma 4's embedding table);
    `Fin`-arithmetic keeps the bound on every step and makes proof
    obligations annoying.
  * We want to admit *runtime* indices (e.g. a u32 read from a param
    buffer) whose bound is checked by a **clamp kernel** (see below).
    `Fin` would force static-only indices.

So we carry the bound separately as a proposition.  A `.clamp` constructor
lets the builder emit a runtime `min(i, dim - 1)` when the proof is
unavailable. -/
inductive ScatterIdx (dim : Nat) where
  /-- A statically-known index `i` with a proof `i < dim`. -/
  | ofNatLt (i : Nat) (h : i < dim) : ScatterIdx dim
  /-- A runtime index (a scalar expression) that the lowering clamps to
      `[0, dim)` via a `min` at code-gen time.  Unsafe-by-omission of the
      bound; the `.ofNatLt` variant is preferred wherever a static proof
      is available. -/
  | clampRuntime (e : ScatterAddr) : ScatterIdx dim

/-- Scatter addresses are small expressions built from the current lane
    id, a runtime value read from a VRAM buffer (typical for `pos`), and
    affine combinations thereof.  Kept tiny on purpose — anything more
    complex should be materialised into a buffer first. -/
inductive ScatterAddr where
  | laneId
  | const   (v : Nat)
  | readU32 (src : Nat) (offset : Nat)  -- buffer id + offset
  | add     (a b : ScatterAddr)
  | mul     (a b : ScatterAddr)
  deriving Repr

/-! ## 5. Block — reads/writes footprint

A `Block` wraps a body of primitives and declares its memory footprint.
Fusion and scheduling passes rely on the declaration being honest, so
the elaborator should verify that every buffer touched by the body
appears in `reads` or `writes`.  (Checker is a later Pass; for now the
constructor trusts the user.)

A slice is `(tensor, start, length)` over the leading dim.  For a more
general slice (e.g. `[:, 3:9, :]`) callers can emit an inner Block. -/
structure TensorSlice where
  tensorId : Nat
  offset   : Nat
  length   : Nat
  deriving Repr

/-! ## 6. Primitive AST

The set of core ops.  Kept small; pointwise bodies use `ScalarExp`
(reused from Circuit v1).

**Binder convention:** every `Prim` that produces a tensor returns a
`CircuitTensor`.  Every `Prim` that mutates a VRAM buffer *additionally*
threads a `StateToken VRAM` through its type. -/

open Scope

/-- Reduction ops mirror v1. -/
inductive ReduceOp where
  | sum | max | sumOfSquares
  deriving Repr

/-- Type-erased scatter index (stores the underlying addr; the refinement
    is enforced at the *constructor* of the user-facing DSL, see `scatter`
    in §7). -/
structure ScatterAddrK where
  addr     : ScatterAddr
  isStatic : Bool
  deriving Repr

inductive Prim : Type where
  /-- `Load`: copy from a VRAM tensor into a new SRAM or Reg tensor.
      The destination scope is chosen by the user via the return type.
      No mutation — the source is not touched. -/
  | load
      (srcId   : Nat)       -- source CircuitTensor id
      (dstScope : Scope)    -- must be SRAM or Reg
      (dstShape : Shape)
      (dt       : DType)
  /-- `Store`: write a Reg or SRAM tensor back to an existing VRAM
      tensor at a fixed offset.  Requires a `StateToken VRAM` in and
      produces a new one.  For scattered writes use `scatter`. -/
  | store
      (srcId   : Nat)
      (dstId   : Nat)
      (offset  : Nat)
  /-- Pointwise op.  Body is a `Hesper.Circuit.ScalarExp` (v1) — we
      reuse v1's ScalarExp verbatim since the lane-local algebra (add,
      mul, exp, tanh, rsqrt, warpSum, etc.) is already complete.  The
      returned tensor lives in the same scope as the inputs. -/
  | pointwise
      (inputs  : Array Nat)    -- input CircuitTensor ids
      (body    : Hesper.Circuit.ScalarExp)
      (outShape : Shape)
      (outDt    : DType)
      (outScope : Scope)
  /-- Reduction over the last axis (sum, max, sum-of-squares, etc.).
      Lowered to a tree reduction when in SRAM, a simple loop when in
      VRAM.  Reg-scope operands are scalarised. -/
  | reduce
      (input   : Nat)
      (op      : ReduceOp)
      (outDt   : DType)
      (outScope : Scope)
  /-- `Scatter`: `dst[index] <- value`.  Threads a state token.  `index`
      is a `ScatterIdx dim` ensuring in-range by construction.  The
      return is both the new token and the dst handle (whose `id` is
      the same — aliasing is explicit). -/
  | scatter
      (dstId   : Nat)
      (dim     : Nat)       -- outer dim being indexed into
      (indexK  : ScatterAddrK)  -- type-erased index (see below)
      (srcId   : Nat)
  /-- `block`: a named scheduling unit with explicit footprint.  Body
      is a list of nested `Prim`s.  Fusion passes treat a Block as an
      atomic unit unless they can prove the merge preserves semantics
      (via the reads/writes declaration). -/
  | block
      (name   : String)
      (reads  : Array TensorSlice)
      (writes : Array TensorSlice)
      (body   : Array Prim)


/-! ## 7. User-facing builder (`CircuitM`)

The monad threads the fresh-ID counter, the accumulated ops, and — most
importantly — the set of live `StateToken`s.  Because `StateToken` is a
dependent parameter on the token scope, the builder can enforce that
each token is used exactly once (linear-ish) simply by handing the user
a fresh handle on every mutation.

Token-ID arithmetic is purely advisory for now; a full linearity check
would need a custom elaborator or a `monad_linear` plugin. -/

structure BuilderState where
  nextId    : Nat := 0
  nextTick  : Nat := 0
  ops       : Array Prim := #[]
  deriving Inhabited

abbrev CircuitM := StateM BuilderState

namespace CircuitM

def fresh : CircuitM Nat := do
  let s ← get
  set { s with nextId := s.nextId + 1 }
  return s.nextId

def tick : CircuitM Nat := do
  let s ← get
  set { s with nextTick := s.nextTick + 1 }
  return s.nextTick

def emit (p : Prim) : CircuitM Unit := do
  modify fun s => { s with ops := s.ops.push p }

/-- Allocate a new in-scope tensor handle.  No op is emitted yet — the
    handle is live only after a primitive actually produces it. -/
def newTensor (scope : Scope) (shape : Shape) (dt : DType)
    : CircuitM (CircuitTensor scope shape dt) := do
  let id ← fresh
  return ⟨id⟩

/-! ### Memory transitions -/

/-- Copy a VRAM tensor into SRAM.  Returns the SRAM handle. -/
def loadToSram (src : CircuitTensor .VRAM shape dt)
    : CircuitM (CircuitTensor .SRAM shape dt) := do
  let dst ← newTensor .SRAM shape dt
  emit (Prim.load src.id .SRAM shape dt)
  return dst

/-- Copy a VRAM tensor into registers (only safe for small tensors). -/
def loadToReg (src : CircuitTensor .VRAM shape dt)
    : CircuitM (CircuitTensor .Reg shape dt) := do
  let dst ← newTensor .Reg shape dt
  emit (Prim.load src.id .Reg shape dt)
  return dst

/-- Write a Reg/SRAM tensor back to VRAM at a fixed offset.  Requires
    and produces a `StateToken VRAM`.  The token is ghost — Lean threads
    it through so the caller can only issue mutating primitives in a
    well-ordered chain. -/
def storeTo {srcScope : Scope} {shape dstShape : Shape} {dt : DType}
    (src : CircuitTensor srcScope shape dt)
    (dst : CircuitTensor .VRAM dstShape dt)
    (offset : Nat)
    (_tok : StateToken .VRAM)
    : CircuitM (StateToken .VRAM) := do
  emit (Prim.store src.id dst.id offset)
  let t ← tick
  return ⟨t⟩

/-! ### Scatter -/

/-- Scatter write with a **statically** proved index.  `h : i < outerDim`
    is consumed only by the Lean elaborator (it fires on the caller) — at
    the AST level we erase it to a concrete `Nat`. -/
def scatterStatic {outerDim : Nat} {restShape : Shape} {dt : DType}
    (dst : CircuitTensor .VRAM (outerDim :: restShape) dt)
    (src : CircuitTensor .Reg restShape dt)
    (i   : Nat) (_h : i < outerDim)
    (_tok : StateToken .VRAM)
    : CircuitM (StateToken .VRAM) := do
  let k : ScatterAddrK := { addr := ScatterAddr.const i, isStatic := true }
  emit (Prim.scatter dst.id outerDim k src.id)
  let t ← tick
  return ⟨t⟩

/-- Scatter write with a **runtime** index.  The lowering inserts a
    `min(idx, outerDim - 1)` clamp for safety.  Callers who can prove
    the bound should prefer `scatterStatic`. -/
def scatterRuntime {outerDim : Nat} {restShape : Shape} {dt : DType}
    (dst : CircuitTensor .VRAM (outerDim :: restShape) dt)
    (src : CircuitTensor .Reg restShape dt)
    (addr : ScatterAddr)
    (_tok : StateToken .VRAM)
    : CircuitM (StateToken .VRAM) := do
  let k : ScatterAddrK := { addr := addr, isStatic := false }
  emit (Prim.scatter dst.id outerDim k src.id)
  let t ← tick
  return ⟨t⟩

end CircuitM

/-! ## 8. Reference example

KV-cache-write pattern, matching what the Gemma 4 prefill needs.  Shown
for type checking — no lowering yet.

```
-- dst  : KV cache   [maxSeqLen, headDim]  VRAM
-- src  : new_k      [headDim]             Reg (after RoPE)
-- pos  : runtime u32 read from params[0]
-- tok  : initial StateToken VRAM (provided by caller)

-- writes: kv[pos][j] = src[j]  for j in [0, headDim)
```

See `Hesper/CircuitV2/Examples.lean` (forthcoming) for the full round
trip.  -/

end Hesper.CircuitV2
