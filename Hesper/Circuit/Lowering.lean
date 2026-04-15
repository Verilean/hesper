import Hesper.Circuit.IR
import Hesper.Backend

/-!
# Circuit Lowering — MVP

Takes a `CircuitState` + a resolver from `TensorRef → Buf` and
dispatches each Op through the existing `Linear.LinearLayer.forward`
machinery.  This is the zero-fusion baseline: one Op per dispatch,
same kernels/token as hand-written code — we want to prove the
framework is overhead-free before adding passes.

Subsequent stages will add passes (constFold, mergeSameDispatch,
inlineProducer, …) that rewrite the `ops` list before emit.
-/

namespace Hesper.Circuit

open Hesper

/-- A mapping from TensorRef id to the concrete device buffer.
    Represented as a simple association list; linear lookup is fine
    for MVP circuit sizes (<100 tensors per compile). -/
abbrev BufferMap (BufT : Type) := List (Nat × BufT)

/-- Lower a Circuit: execute each Op in order.  Buffers for produced
    tensors are supplied by the caller via `outputBufs` — the caller
    knows which TensorRef is which (they hold the TensorRef handles
    from the builder). -/
def compile [GPUBackend β]
    (ctx : β)
    (state : CircuitState (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (outputBufs : List (TensorRef × GPUBackend.Buf β))
    : IO Unit := do
  -- Build the initial buffer map from externals + outputBufs.
  let mut bmap : BufferMap (GPUBackend.Buf β) := []
  for (tr, buf) in state.externals.toList do
    bmap := (tr.id, buf) :: bmap
  for (tr, buf) in outputBufs do
    bmap := (tr.id, buf) :: bmap
  let lookup (id : Nat) : Option (GPUBackend.Buf β) :=
    (bmap.find? (fun e => e.1 == id)).map (·.2)
  -- Execute each op.
  for op in state.ops do
    match op.prim with
    | Prim.matmulQ4K layer =>
      let inTr := op.inputs[0]!
      let outTr := op.outputs[0]!
      match lookup inTr.id, lookup outTr.id with
      | some inputBuf, some outputBuf =>
        Hesper.Layers.Linear.LinearLayer.forward ctx layer inputBuf outputBuf
      | _, _ =>
        throw (IO.userError s!"Circuit.compile: missing buffer for matmul op (in={inTr.id}, out={outTr.id})")

end Hesper.Circuit
