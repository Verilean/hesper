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

/-! ## Scalar expression language

`ScalarExp` is the body of a pointwise op.  It is a pure inductive AST,
NOT a Lean closure — the whole point is that we can structurally
inspect, hash, substitute into, and rewrite it from the fusion pass.

`input i` refers to the i-th input tensor of the surrounding
`Prim.pointwise` op, evaluated at the current lane.  `const v` is a
compile-time float literal.  The remaining constructors are standard
arithmetic / transcendentals, added as Gemma 4 demands them. -/
inductive ScalarExp where
  | input  (idx : Nat)
  | const  (v : Float)
  | add    (a b : ScalarExp)
  | sub    (a b : ScalarExp)
  | mul    (a b : ScalarExp)
  | div    (a b : ScalarExp)
  | neg    (a : ScalarExp)
  | rsqrt  (a : ScalarExp)
  | exp    (a : ScalarExp)
  | tanh   (a : ScalarExp)
  | gelu   (a : ScalarExp)
  | silu   (a : ScalarExp)
  deriving Repr, Inhabited

namespace ScalarExp

/-- Shift every `input i` inside `e` by `k` — used when inlining a
    producer into a consumer so that the producer's input indices
    continue to reference the right slots after the consumer's input
    array is rewritten. -/
partial def shiftInputs (k : Nat) : ScalarExp → ScalarExp
  | input i      => input (i + k)
  | const v      => const v
  | add a b      => add (shiftInputs k a) (shiftInputs k b)
  | sub a b      => sub (shiftInputs k a) (shiftInputs k b)
  | mul a b      => mul (shiftInputs k a) (shiftInputs k b)
  | div a b      => div (shiftInputs k a) (shiftInputs k b)
  | neg a        => neg (shiftInputs k a)
  | rsqrt a      => rsqrt (shiftInputs k a)
  | exp a        => exp (shiftInputs k a)
  | tanh a       => tanh (shiftInputs k a)
  | gelu a       => gelu (shiftInputs k a)
  | silu a       => silu (shiftInputs k a)

/-- Substitute `args[i]` for `input i` everywhere in `e`.  Missing
    indices default to `input i` unchanged — callers should ensure
    `args` covers all free `input i`s that appear. -/
partial def subst (args : Array ScalarExp) : ScalarExp → ScalarExp
  | input i      => match args[i]? with | some a => a | none => input i
  | const v      => const v
  | add a b      => add (subst args a) (subst args b)
  | sub a b      => sub (subst args a) (subst args b)
  | mul a b      => mul (subst args a) (subst args b)
  | div a b      => div (subst args a) (subst args b)
  | neg a        => neg (subst args a)
  | rsqrt a      => rsqrt (subst args a)
  | exp a        => exp (subst args a)
  | tanh a       => tanh (subst args a)
  | gelu a       => gelu (subst args a)
  | silu a       => silu (subst args a)

end ScalarExp

/-- Number of elements in a shape. -/
def Shape.numel (s : Shape) : Nat := s.foldl (· * ·) 1

/-- A typed description of a primitive operation.  Each Prim has a
    well-defined (input shapes, output shapes, dispatch shape) mapping
    that the lowering pass consumes. -/
inductive Prim (BufT : Type) (CacheT : Type) where
  /-- Matmul against a Q4_K weight layer.  The matmul is a full
      forward through `Hesper.Layers.Linear.LinearLayer.forward` — the Prim owns the
      layer, so it owns the prepared-dispatch cache. -/
  | matmulQ4K
      (layer : Hesper.Layers.Linear.LinearLayer BufT CacheT)
  /-- Pure pointwise op.  `outShape` is the shape of the output tensor
      AND of the dispatch grid (fixed 1D `(numel+255)/256 × 256`).
      `inShapes[i]` describes the i-th input:
        * equal to `outShape` ⇒ full element-wise, indexed by the
          thread's global id at every lane;
        * `#[1]` (a 1-element tensor) ⇒ **scalar broadcast**; every
          lane reads slot 0.  Used for scalar biases / scales that are
          loaded from GPU buffers rather than baked as `const` in the
          body.
      Any other shape is rejected at build time (for now — general
      NumPy-style broadcast is out of scope).

      `body` is a `ScalarExp` where `input i` refers to `inputs[i]`.
      Fusion is trivially safe: the output shape uniquely determines
      the grid, and broadcast inputs compose (scalar-of-scalar is
      still scalar). -/
  | pointwise
      (outShape : Shape) (inShapes : Array Shape) (body : ScalarExp)
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

/-! ### Pointwise builder sugar

All sugar below lowers to `Prim.pointwise`.  The body uses
`ScalarExp.input i` to refer to the i-th tensor in the `inputs` array
— the builder chooses that layout so the fusion pass can inline
`input` indices by shifting. -/

/-- Generic pointwise: caller supplies the inputs array (each either
    of the output shape or scalar `#[1]`) and a body tree.  Output shape
    is taken from the first full-shape input, or falls back to the
    first input's shape.  For the purely-broadcast degenerate case
    (all inputs scalar) the output is also scalar `#[1]`. -/
def pointwise (inputs : Array TensorRef) (body : ScalarExp)
    : CircuitM BufT CacheT TensorRef := do
  -- Pick the output shape: first non-broadcast input, else any input's shape.
  let outShape : Shape :=
    (inputs.find? (fun tr => tr.shape != #[1])).map (·.shape)
      |>.getD ((inputs[0]?).map (·.shape) |>.getD #[])
  let inShapes : Array Shape := inputs.map (·.shape)
  let outs ← emitOp (Prim.pointwise outShape inShapes body) inputs
    #[(outShape, .f32, .Global)]
  return outs[0]!

/-- Unary map: `out[i] = f(a[i])`.  `f` is a ScalarExp with a single
    free variable `input 0`. -/
def map (a : TensorRef) (f : ScalarExp) : CircuitM BufT CacheT TensorRef :=
  pointwise #[a] f

/-- Binary zip: `out[i] = f(a[i], b[i])`.  `f` has free `input 0`
    (=`a`) and `input 1` (=`b`).  If either input has shape `#[1]` it
    broadcasts across the full shape of the other. -/
def zip2 (a b : TensorRef) (f : ScalarExp)
    : CircuitM BufT CacheT TensorRef :=
  pointwise #[a, b] f

/-- Scalar multiply: `out[i] = a[i] * k`. -/
def scale (a : TensorRef) (k : Float) : CircuitM BufT CacheT TensorRef :=
  map a (.mul (.input 0) (.const k))

/-- Scalar broadcast multiply: `out[i] = a[i] * scale[0]`.  `scale`
    must be a `#[1]`-shape TensorRef. -/
def scaleByBroadcast (a scale : TensorRef) : CircuitM BufT CacheT TensorRef :=
  zip2 a scale (.mul (.input 0) (.input 1))

/-- Elementwise add: `out[i] = a[i] + b[i]`. -/
def addT (a b : TensorRef) : CircuitM BufT CacheT TensorRef :=
  zip2 a b (.add (.input 0) (.input 1))

end CircuitM

end Hesper.Circuit
