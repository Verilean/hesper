import Hesper.Backend
import Hesper.Layers.Linear

/-!
# Circuit IR — minimal viable subset

See `docs/circuit-dsl-design.md` for the full design.  This file
contains only the pieces needed for the MVP: run a Q4_K matmul
through the graph path and show zero-overhead dispatch.

Stage 1 scope:
  - `TensorRef` carries shape, dtype, scope — but we start with
    everything in `Scope.Global` (no register chains yet).
  - Exactly one `Prim`: `matmulQ4K`, wrapping the existing
    `Hesper.Layers.Linear.LinearLayer.forward` dispatch.
  - `CircuitM` builder that captures (inputBuf, layer) and the
    implied output buffer; lowering is a direct call.
  - No fusion passes yet.

Once this compiles and runs at parity, we extend with more Prims and
the first fusion passes.
-/

namespace Hesper.Circuit

open Hesper

/-- Element type of a tensor. -/
inductive DType where
  | f32
  | f16
  | u32
  | q4k   -- packed Q4_K (weight storage)
  | q6k   -- packed Q6_K
  | q8_1  -- quantized input for dp4a path
  deriving BEq, Repr, Inhabited

/-- Where the tensor lives in the memory hierarchy.  For the MVP we
    only use `.Global`; the other scopes are reserved for fusion
    passes to promote/demote tensors to. -/
inductive Scope where
  | Register   -- thread-local (fusion target)
  | Lane       -- subgroup-local
  | Shared     -- workgroup-local
  | Global     -- device-wide (buffer-backed)
  deriving BEq, Repr, Inhabited

/-- Compile-time tensor shape.  We keep it as `Array Nat` for now
    (typed-shape `List Nat` is for a later stage). -/
abbrev Shape := Array Nat

/-- A reference to a tensor in the Circuit.  `id` is allocated
    monotonically by the builder and serves as the tensor's only
    identity — the user never sees a string. -/
structure TensorRef where
  id    : Nat
  shape : Shape
  dtype : DType
  scope : Scope
  deriving Inhabited

/-- External tensors supplied by the caller (model weights, input
    activations).  These are not produced by any Op in the circuit. -/
structure ExternalTensor (BufT : Type) where
  buf   : BufT              -- the device buffer
  shape : Shape
  dtype : DType
  deriving Inhabited

/-- A typed description of a primitive operation.  Each Prim has a
    well-defined (input shapes, output shapes, dispatch shape) mapping
    that the lowering pass consumes. -/
inductive Prim (BufT : Type) (CacheT : Type) where
  /-- Matmul against a Q4_K weight layer.  The matmul is a full
      forward through `Hesper.Layers.Linear.LinearLayer.forward` — the Prim owns the
      layer, so it owns the prepared-dispatch cache. -/
  | matmulQ4K
      (layer : Hesper.Layers.Linear.LinearLayer BufT CacheT)
  -- Future primitives will go here: rmsNorm, residualAdd, rope, …

/-- An op in the circuit: a Prim plus the concrete tensor wiring. -/
structure Op (BufT : Type) (CacheT : Type) where
  prim    : Prim BufT CacheT
  inputs  : Array TensorRef
  outputs : Array TensorRef

/-- State of the circuit builder. -/
structure CircuitState (BufT : Type) (CacheT : Type) where
  tensors    : Array TensorRef
  ops        : Array (Op BufT CacheT)
  /-- External tensors registered by the user.  Indexed by tensor id. -/
  externals  : Array (TensorRef × BufT)

/-- Builder monad for constructing a Circuit.  `StateM` enforces the
    DAG invariant by construction — every TensorRef in `inputs` must
    already be in `tensors` when an op is emitted, which is only
    possible if it was produced by an earlier `emitOp` call or
    registered as an external. -/
abbrev CircuitM (BufT : Type) (CacheT : Type) (α : Type) :=
  StateM (CircuitState BufT CacheT) α

namespace CircuitM

variable {BufT CacheT : Type}

/-- Run a builder to produce a final CircuitState.  The caller gets
    back whatever the builder returned (typically the "output"
    TensorRefs) along with the collected state. -/
def run (m : CircuitM BufT CacheT α) : α × CircuitState BufT CacheT :=
  m { tensors := #[], ops := #[], externals := #[] }

/-- Allocate a fresh TensorRef at the given shape/dtype/scope.  The id
    is the next slot in the `tensors` array. -/
def freshTensor (shape : Shape) (dtype : DType) (scope : Scope := .Global)
    : CircuitM BufT CacheT TensorRef := do
  let s ← get
  let id := s.tensors.size
  let tr : TensorRef := { id, shape, dtype, scope }
  set { s with tensors := s.tensors.push tr }
  return tr

/-- Register an externally-supplied buffer as an input TensorRef.
    Typical use: the hidden state coming from the previous layer, or a
    model weight that isn't modelled by a Prim. -/
def registerExternal (buf : BufT) (shape : Shape) (dtype : DType) (scope : Scope := .Global)
    : CircuitM BufT CacheT TensorRef := do
  let tr ← freshTensor shape dtype scope
  modify fun s => { s with externals := s.externals.push (tr, buf) }
  return tr

/-- Emit an op.  Allocates fresh TensorRefs for each output spec and
    returns them.  No dispatch-shape argument is taken; the Prim knows
    its own shape. -/
def emitOp (prim : Prim BufT CacheT) (inputs : Array TensorRef)
    (outSpecs : Array (Shape × DType × Scope))
    : CircuitM BufT CacheT (Array TensorRef) := do
  let outs ← outSpecs.mapM fun (shape, dtype, scope) =>
    freshTensor shape dtype scope
  modify fun s => { s with ops := s.ops.push { prim, inputs, outputs := outs } }
  return outs

/-- Typed wrapper for Q4_K matmul: `input : [inDim] f32 Global` →
    `output : [outDim] f32 Global`.  The Prim carries the layer; the
    CircuitM layer carries the tensor wiring. -/
def matmulQ4K (input : TensorRef) (layer : Hesper.Layers.Linear.LinearLayer BufT CacheT)
    : CircuitM BufT CacheT TensorRef := do
  let outs ← emitOp (Prim.matmulQ4K layer) #[input]
    #[(#[layer.config.outDim], .f32, .Global)]
  return outs[0]!

end CircuitM

end Hesper.Circuit
