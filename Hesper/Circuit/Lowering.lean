import Hesper.Circuit.IR
import Hesper.Backend
import Hesper.WGSL.Monad
import Hesper.WGSL.Exp

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
open Hesper.WGSL
open Hesper.WGSL.Monad

/-! ## Generic pointwise lowering

`lowerScalarExp` folds a `ScalarExp` tree into a typed `Exp` over the
`f32` tensor element at the current lane.  `lowerPointwise` builds the
entire `ShaderM Unit` from the body + input-slot names.

Key property: the lowering is a **function of the IR alone**.  There is
no pattern registry that maps `Prim.pointwise scaleBody → scaleKernel`;
instead, any `ScalarExp` body — whether hand-written, fusion-produced,
or constant-folded — flows through the same lowering.  That's what
makes this a compiler pass instead of a dispatcher. -/

/-- Lower a `ScalarExp` to a typed WGSL `Exp` of f32.  Per-element
    reads from the input slots are pre-loaded into `slot : Array (Exp f32)`
    so the same `input i` reference inside `body` compiles to a single
    shared `var` read, not N redundant buffer loads. -/
def lowerScalarExp (slot : Array (Exp (.scalar .f32)))
    : ScalarExp → Exp (.scalar .f32)
  | .input i      =>
    match slot[i]? with
    | some e => e
    | none   => Exp.litF32 0.0  -- unreachable: caller shape-checks
  | .const v      => Exp.litF32 v
  | .add a b      => Exp.add (lowerScalarExp slot a) (lowerScalarExp slot b)
  | .sub a b      => Exp.sub (lowerScalarExp slot a) (lowerScalarExp slot b)
  | .mul a b      => Exp.mul (lowerScalarExp slot a) (lowerScalarExp slot b)
  | .div a b      => Exp.div (lowerScalarExp slot a) (lowerScalarExp slot b)
  | .neg a        => Exp.neg (lowerScalarExp slot a)
  | .rsqrt a      => Exp.inverseSqrt (lowerScalarExp slot a)
  | .exp a        => Exp.exp (lowerScalarExp slot a)
  | .tanh a       => Exp.tanh (lowerScalarExp slot a)
  | .gelu a       =>
    -- tanh approximation: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    let x := lowerScalarExp slot a
    let c : Float := 0.7978845608028654
    let k : Float := 0.044715
    let x3 := Exp.mul (Exp.mul x x) x
    let inner := Exp.mul (Exp.litF32 c)
                         (Exp.add x (Exp.mul (Exp.litF32 k) x3))
    Exp.mul (Exp.mul (Exp.litF32 0.5) x)
            (Exp.add (Exp.litF32 1.0) (Exp.tanh inner))
  | .silu a       =>
    -- x / (1 + exp(-x))
    let x := lowerScalarExp slot a
    Exp.div x (Exp.add (Exp.litF32 1.0) (Exp.exp (Exp.neg x)))

/-- Build a complete `ShaderM Unit` that implements a `Prim.pointwise`
    op.  `inputNames[i]` is the buffer binding name for input slot i;
    `inputBroadcast[i] = true` means that input is a 1-element scalar
    broadcast (every lane reads slot 0), false means indexed by thread
    id.  `outputName` is where to write.  `numel` is the total output
    element count.

    Dispatch shape is a fixed 1D `(numel+255)/256 × 256`.  Inside the
    kernel, each thread computes one output element if its global id is
    in range. -/
def lowerPointwise
    (inputNames : Array String) (inputBroadcast : Array Bool) (outputName : String)
    (numel : Nat) (body : ScalarExp) : ShaderM Unit := do
  -- Declare broadcast inputs with length 1; full inputs with length numel.
  for i in [0:inputNames.size] do
    let name := inputNames[i]!
    let bc := inputBroadcast[i]?.getD false
    let len := if bc then 1 else numel
    let _ ← ShaderM.declareInputBuffer name (.array (.scalar .f32) len)
  let _ ← ShaderM.declareOutputBuffer outputName (.array (.scalar .f32) numel)
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid
  ShaderM.if_ (Exp.lt idx (Exp.litU32 numel)) (do
    -- Pre-load every input slot once, bind to a local var.  Broadcast
    -- inputs read at index 0; full inputs at the thread's id.
    let mut slots : Array (Exp (.scalar .f32)) := #[]
    for i in [0:inputNames.size] do
      let name := inputNames[i]!
      let bc := inputBroadcast[i]?.getD false
      let len := if bc then 1 else numel
      let readIdx := if bc then Exp.litU32 0 else idx
      let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := len) name readIdx
      let vName ← ShaderM.var (.scalar .f32) v
      slots := slots.push (Exp.var vName)
    let result := lowerScalarExp slots body
    ShaderM.writeBuffer (ty := .scalar .f32) outputName idx result
  ) (pure ())

/-- A mapping from TensorRef id to the concrete device buffer.
    Represented as a simple association list; linear lookup is fine
    for MVP circuit sizes (<100 tensors per compile). -/
abbrev BufferMap (BufT : Type) := List (Nat × BufT)

/-- Dispatch a `Prim.pointwise` op through `executeWithConfigCached`.
    `numel` is the output element count (= dispatch grid).  `inShapes`
    describes each input: shape `#[1]` ⇒ broadcast; otherwise full. -/
def runPointwiseOp [GPUBackend β]
    (ctx : β) (numel : Nat) (inShapes : Array Shape) (body : ScalarExp)
    (inputBufs : Array (GPUBackend.Buf β))
    (outputBuf : GPUBackend.Buf β)
    (cacheKey : UInt64) (cacheRef : IO.Ref (Option (GPUBackend.CachedDispatch β)))
    : IO Unit := do
  let inputNames : Array String :=
    Array.ofFn (fun (i : Fin inputBufs.size) => s!"in{i.val}")
  let inputBroadcast : Array Bool :=
    inShapes.map (fun s => s == #[1])
  let outputName := "out"
  let shader : ShaderM Unit :=
    lowerPointwise inputNames inputBroadcast outputName numel body
  let mut namedBufs : List (String × GPUBackend.Buf β) := [(outputName, outputBuf)]
  let n := inputBufs.size
  for i in [0:n] do
    match inputBufs[i]? with
    | some buf => namedBufs := (s!"in{i}", buf) :: namedBufs
    | none     => pure ()
  let config : ExecConfig := ExecConfig.dispatch1D numel
  GPUBackend.executeWithConfigCached ctx shader namedBufs config cacheKey cacheRef

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
    | Prim.pointwise outShape inShapes body =>
      let outTr := op.outputs[0]!
      let mut inBufs : Array (GPUBackend.Buf β) := #[]
      for inTr in op.inputs do
        match lookup inTr.id with
        | some b => inBufs := inBufs.push b
        | none   => throw (IO.userError s!"Circuit.compile: missing buffer for pointwise input {inTr.id}")
      match lookup outTr.id with
      | some outBuf =>
        let cacheRef ← IO.mkRef (α := Option (GPUBackend.CachedDispatch β)) none
        runPointwiseOp ctx outShape.numel inShapes body inBufs outBuf 0 cacheRef
      | none => throw (IO.userError s!"Circuit.compile: missing output buffer for pointwise op (out={outTr.id})")

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
  -- Pointwise ops allocate their own cacheRef in IO so the replay
  -- closure can close over it and skip PTX regeneration.
  let mut closures : Array (OpClosure β) := #[]
  for op in state.ops do
    match op.prim with
    | Prim.matmulQ4K layer =>
      let inId := op.inputs[0]!.id
      let outId := op.outputs[0]!.id
      closures := closures.push
        { run := fun lookup => do
            match lookup inId, lookup outId with
            | some inputBuf, some outputBuf =>
              Hesper.Layers.Linear.LinearLayer.forward (β := β) _ctx layer inputBuf outputBuf
            | _, _ =>
              throw (IO.userError s!"CompiledCircuit: missing buffer (in={inId} out={outId})")
        }
    | Prim.pointwise outShape inShapes body =>
      let numel := outShape.numel
      let inIds := op.inputs.map (·.id)
      let outId := op.outputs[0]!.id
      let cacheRef : IO.Ref (Option (GPUBackend.CachedDispatch β)) ← IO.mkRef none
      let cacheKey : UInt64 :=
        hash ("circuit-pointwise", numel, reprStr body,
              reprStr inShapes.toList)
      closures := closures.push
        { run := fun lookup => do
            let mut inBufs : Array (GPUBackend.Buf β) := #[]
            for id in inIds do
              match lookup id with
              | some b => inBufs := inBufs.push b
              | none   => throw (IO.userError s!"CompiledCircuit: missing pointwise input buffer id={id}")
            match lookup outId with
            | some outBuf =>
              runPointwiseOp _ctx numel inShapes body inBufs outBuf cacheKey cacheRef
            | none => throw (IO.userError s!"CompiledCircuit: missing pointwise output buffer id={outId}")
        }
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

/-- Build-or-replay helper.  On first call, runs `build` to produce the
    Circuit, compiles it, and stores it in `cacheRef`.  On subsequent
    calls, replays the cached CompiledCircuit directly.  This is the
    zero-overhead production path: the CircuitM builder runs at most once
    per unique cacheRef. -/
def runCached [GPUBackend β]
    (ctx : β)
    (cacheRef : IO.Ref (Option (CompiledCircuit β)))
    (build : CircuitM (GPUBackend.Buf β) (GPUBackend.CachedDispatch β) Unit)
    (buffers : List (Nat × GPUBackend.Buf β))
    : IO Unit := do
  match ← cacheRef.get with
  | some cc => cc.replay buffers
  | none =>
    let (_, st) := CircuitM.run build
    let cc ← compileOnce ctx st
    cacheRef.set (some cc)
    cc.replay buffers

/-- Per-cacheKey store for CompiledCircuits — sibling of
    `Hesper.Models.Gemma4.KernelCacheRefs` but typed for our IR.  Lazily
    creates IO.Refs on demand. -/
structure CircuitCacheRefs (β : Type) [GPUBackend β] where
  store : IO.Ref (Array (UInt64 × IO.Ref (Option (CompiledCircuit β))))

def CircuitCacheRefs.create [GPUBackend β] : IO (CircuitCacheRefs β) := do
  let r ← IO.mkRef #[]
  return ⟨r⟩

def CircuitCacheRefs.getRef [GPUBackend β]
    (ccr : CircuitCacheRefs β) (key : UInt64)
    : IO (IO.Ref (Option (CompiledCircuit β))) := do
  let arr ← ccr.store.get
  match arr.find? (fun e => e.1 == key) with
  | some (_, r) => return r
  | none =>
    let r ← IO.mkRef none
    ccr.store.modify (·.push (key, r))
    return r

/-! ## Module-level fallback cache

Untyped global cache keyed by hash.  Used when callers can't (yet)
thread a `CircuitCacheRefs` through their signatures.  The stored
pointer is `unsafeCast`-erased; the caller must supply matching `β`
on each lookup (typically by including the backend tag in the key). -/
initialize circuitCacheGlobal : IO.Ref (Array (UInt64 × NonScalar)) ← IO.mkRef #[]

/-- Get-or-create a cache slot.  Returns an `IO.Ref (Option (CompiledCircuit β))`. -/
unsafe def getGlobalCircuitRefUnsafe [GPUBackend β] (key : UInt64)
    : IO (IO.Ref (Option (CompiledCircuit β))) := do
  let arr ← circuitCacheGlobal.get
  match arr.find? (fun e => e.1 == key) with
  | some (_, ptr) => return unsafeCast ptr
  | none =>
    let r : IO.Ref (Option (CompiledCircuit β)) ← IO.mkRef none
    circuitCacheGlobal.modify (·.push (key, unsafeCast r))
    return r

@[implemented_by getGlobalCircuitRefUnsafe]
opaque getGlobalCircuitRef [GPUBackend β] (key : UInt64)
    : IO (IO.Ref (Option (CompiledCircuit β)))

end Hesper.Circuit
