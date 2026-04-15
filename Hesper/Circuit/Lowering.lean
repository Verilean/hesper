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

/-! ## Build-once, replay-many — zero-overhead dispatch path

`CompiledCircuit` holds a pre-resolved list of op closures that take
only the caller's per-call buffers (hidden state in, Q out, etc.) and
execute the backing kernels directly.  No CircuitM evaluation, no
buffer-map construction, no pattern-matching on Prim on the hot path.

Usage (cached):
  state.compiledQ : IO.Ref (Option (CompiledCircuit β))
  ...
  let cc ← state.compiledQ.get >>= fun
    | some cc => pure cc
    | none =>
      let cc ← Circuit.compileOnce ctx (buildCircuit ...)
      state.compiledQ.set (some cc)
      pure cc
  cc.replay [(tensorQNormed, state.normedBuf), (tensorQOut, state.qBuf)]
-/

/-- An op compiled down to a single closure.  The closure takes the
    resolver function (input/output id → buffer) and runs the dispatch. -/
structure OpClosure (β : Type) [GPUBackend β] where
  run : (lookup : Nat → Option (GPUBackend.Buf β)) → IO Unit

/-- A circuit compiled into a flat, cache-friendly representation.
    No Lean-side allocation on replay — just a walk over the closures. -/
structure CompiledCircuit (β : Type) [GPUBackend β] where
  ops : Array (OpClosure β)
  /-- TensorRefs the caller needs to supply a buffer for (externals +
      every produced tensor).  Kept so callers know what to pass on
      replay. -/
  externalIds : Array Nat
  producedIds : Array Nat

/-- Compile a CircuitState into a CompiledCircuit that can be replayed
    with minimal overhead.  Runs once per unique circuit. -/
def compileOnce [GPUBackend β]
    (_ctx : β)
    (state : CircuitState (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    : IO (CompiledCircuit β) := do
  -- Convert each Op to an OpClosure that captures its metadata
  -- (layer, tensor ids) but defers buffer resolution to replay time.
  let mkClosure (op : Op (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
      : OpClosure β :=
    match op.prim with
    | Prim.matmulQ4K layer =>
      let inId := op.inputs[0]!.id
      let outId := op.outputs[0]!.id
      { run := fun lookup => do
          match lookup inId, lookup outId with
          | some inputBuf, some outputBuf =>
            Hesper.Layers.Linear.LinearLayer.forward (β := β) _ctx layer inputBuf outputBuf
          | _, _ =>
            throw (IO.userError s!"CompiledCircuit: missing buffer (in={inId} out={outId})")
      }
  let closures := state.ops.map mkClosure
  let externalIds := state.externals.map (fun (tr, _) => tr.id)
  -- producedIds := tensor ids that are produced by some op's outputs
  let producedIds := state.ops.foldl (init := (#[] : Array Nat)) fun acc op =>
    op.outputs.foldl (init := acc) fun acc' tr => acc'.push tr.id
  return { ops := closures, externalIds, producedIds }

/-- Replay a compiled circuit.  `buffers` lists the (tensorId, buffer)
    pairs the caller wants to wire in for this invocation — externals
    AND produced-tensor outputs they want preserved. -/
def CompiledCircuit.replay [GPUBackend β]
    (cc : CompiledCircuit β)
    (buffers : List (Nat × GPUBackend.Buf β))
    : IO Unit := do
  -- Small associative lookup.  For MVP circuits (<5 entries) linear
  -- search is faster than hash construction.
  let lookup (id : Nat) : Option (GPUBackend.Buf β) :=
    (buffers.find? (fun e => e.1 == id)).map (·.2)
  for op in cc.ops do
    op.run lookup

end Hesper.Circuit
