import Hesper.Circuit.IR
import Hesper.Circuit.Lowering
import Hesper.Layers.Linear

/-!
# Circuit fusion passes

Stage 2 entry point.  Each pass takes a `CircuitState` and returns a
rewritten one, ideally with fewer Ops.  A pass MUST preserve the
program semantics (same outputs, modulo numerical reorder) but may
change the kernel-count and dispatch shape.

For the MVP we implement only `mergeSameDispatch`:
  - Find two consecutive `Prim.matmulQ4K` Ops that share the same
    *input* TensorRef and have the same `(inDim, outDim)`.
  - Replace the pair with a single `Prim.fusedKV` that lowers to
    the existing `fusedQ4KMKVDP4AKernel`.

This isn't fully generic — `Prim.fusedKV` is special-cased — but it
proves the loop end-to-end:
  builder writes 2 matmuls → IR has 2 ops → pass detects pair →
  IR has 1 fused op → lowering dispatches 1 kernel.

Stage 3 will replace the special-case `fusedKV` with a generic
inlining pass that synthesises the fused ShaderM from the two
input bodies.  For now we lean on the manually-written kernel.
-/

namespace Hesper.Circuit

open Hesper

/-- Extension to the Prim universe used after fusion passes — these
    don't appear in user-written CircuitMs but are introduced by
    the rewriter. -/
inductive PrimExt (BufT : Type) (CacheT : Type) where
  | base    (p : Prim BufT CacheT)
  /-- Two-output fused KV matmul (wK + wV).  Both matmuls share the
      same input; the lowering emits one dispatch via
      `Linear.forwardFusedKV`. -/
  | fusedKV (wK wV : Hesper.Layers.Linear.LinearLayer BufT CacheT)

/-- Op variant with the extended Prim. -/
structure OpExt (BufT : Type) (CacheT : Type) where
  prim    : PrimExt BufT CacheT
  inputs  : Array TensorRef
  outputs : Array TensorRef

/-- Walk the op list once, looking for `[matmulQ4K wA, matmulQ4K wB]`
    pairs that share the same input TensorRef AND identical
    `(inDim, outDim)`.  When found, replace the pair with a single
    `fusedKV` op.  Other ops pass through unchanged. -/
def mergeSameDispatch {BufT CacheT : Type}
    (ops : Array (Op BufT CacheT)) : Array (OpExt BufT CacheT) := Id.run do
  let mut out : Array (OpExt BufT CacheT) := #[]
  let mut i := 0
  while h : i < ops.size do
    let op := ops[i]
    let nextOpt : Option (Op BufT CacheT) :=
      if h2 : i + 1 < ops.size then some ops[i + 1] else none
    match nextOpt with
    | some next =>
      match op.prim, next.prim with
      | Prim.matmulQ4K wA, Prim.matmulQ4K wB =>
        let sameInput :=
          op.inputs.size == 1 && next.inputs.size == 1 &&
          (match op.inputs[0]?, next.inputs[0]? with
           | some a, some b => a.id == b.id
           | _, _ => false)
        let sameShape := wA.config.inDim == wB.config.inDim
                      && wA.config.outDim == wB.config.outDim
        -- The hand-written `forwardFusedKV` requires both layers to be
        -- Q4_K — Q4_K_M models often have wV as Q6_K, in which case we
        -- bail out and emit two separate matmuls.
        let bothQ4K := wA.quantFormat == .Q4_K && wB.quantFormat == .Q4_K
        let hasOuts :=
          (match op.outputs[0]?, next.outputs[0]? with
           | some _, some _ => true
           | _, _ => false)
        if sameInput && sameShape && bothQ4K && hasOuts then
          match op.outputs[0]?, next.outputs[0]? with
          | some outA, some outB =>
            out := out.push
              { prim := PrimExt.fusedKV wA wB
                inputs := op.inputs
                outputs := #[outA, outB] }
            i := i + 2
          | _, _ =>
            out := out.push { prim := PrimExt.base op.prim, inputs := op.inputs, outputs := op.outputs }
            i := i + 1
        else
          out := out.push { prim := PrimExt.base op.prim, inputs := op.inputs, outputs := op.outputs }
          i := i + 1
      | _, _ =>
        -- Other prim combinations (including Prim.pointwise) pass through unchanged here.
        -- Pointwise fusion is handled by `fusePointwise` upstream of this pass.
        out := out.push { prim := PrimExt.base op.prim, inputs := op.inputs, outputs := op.outputs }
        i := i + 1
    | none =>
      out := out.push { prim := PrimExt.base op.prim, inputs := op.inputs, outputs := op.outputs }
      i := i + 1
  return out

/-! ## Pointwise fusion — β-reduction on pointwise op DAG

`fusePointwise` is the core compiler pass.  It repeatedly picks a
producer `A : pointwise` whose unique consumer `B` is also a
`pointwise`, and inlines A's body into B — replacing the input slot
that referred to A with A's expression, and folding A's input tensors
into B's input array.

Invariants (all automatic from `Prim.pointwise`):
- same element count ⇒ 1:1 thread-to-element mapping
- no inter-thread dependency ⇒ inlining changes nothing observable
- dispatch shape is fixed 1D so merged ops have the same grid

The "unique consumer" guard avoids duplicating producer work across
multiple consumers.  Without it, fusion could degrade performance by
recomputing; refusing is always safe.

After substitution the producer becomes unreachable and is removed by
the trailing filter step. -/

/-- Count how many surviving ops reference `tid` as one of their inputs,
    plus "is `tid` a caller-requested output buffer" (which we can't know
    here; callers supply this via `isExternalOutput` — in the `runCached*`
    path, all tensor ids registered via `registerExternal` are fixed, but
    produced TensorRefs may be wired by the caller too).

    Because `runCached` passes an explicit `buffers` list on replay, we
    conservatively treat *any* TensorRef whose id appears in some
    op's input list as internal-only; the caller's `buffers` list is
    opaque to this pass.  See the `protectedIds` argument. -/
private def countConsumers {BufT CacheT : Type}
    (ops : Array (Op BufT CacheT)) (tid : Nat) : Nat :=
  ops.foldl (init := 0) fun acc op =>
    op.inputs.foldl (init := acc) fun acc' tr =>
      if tr.id == tid then acc' + 1 else acc'

/-- One fusion step: scan for a producer/consumer pair and return
    `some newOps` if we found one to inline; `none` otherwise.  The
    caller iterates until fixpoint. -/
private def fusePointwiseStep {BufT CacheT : Type}
    (ops : Array (Op BufT CacheT)) (protectedIds : Array Nat)
    : Option (Array (Op BufT CacheT)) := Id.run do
  let isProtected (id : Nat) : Bool := protectedIds.any (· == id)
  -- Search for a consumer op B such that one of its pointwise inputs
  -- is produced by an earlier pointwise A with B as its unique
  -- surviving consumer (and A's output is not caller-facing).
  for bIdx in [0 : ops.size] do
    match ops[bIdx]? with
    | none => pure ()
    | some B =>
    let .pointwise bOutShape bInShapes bBody := B.prim | continue
    -- Walk B's inputs looking for a fusable producer.
    for iIdx in [0 : B.inputs.size] do
      let prodRef := B.inputs[iIdx]!
      if isProtected prodRef.id then continue
      -- Refuse to fuse when B consumes this slot as a scalar broadcast:
      -- inlining a non-broadcast producer would violate the "all inputs
      -- are either outShape or #[1]" invariant.
      let consumedAsBroadcast : Bool := (bInShapes[iIdx]?.getD #[]) == #[1]
      if consumedAsBroadcast then continue
      -- Find the (single) op in `ops` whose outputs contain prodRef.id.
      let mut aIdxOpt : Option Nat := none
      for hA : aIdx in [0 : ops.size] do
        match ops[aIdx]? with
        | some A =>
          if A.outputs.any (fun tr => tr.id == prodRef.id) then
            aIdxOpt := some aIdx
            break
        | none => pure ()
      let some aIdx := aIdxOpt | continue
      let some A := ops[aIdx]? | continue
      let .pointwise aOutShape aInShapes aBody := A.prim | continue
      -- Fusion legality: A's output shape must match B's.
      if aOutShape != bOutShape then continue
      -- Uniqueness: prodRef.id is consumed by exactly one op in `ops`.
      if countConsumers ops prodRef.id != 1 then continue
      -- Inline A into B.  B's slot iIdx is replaced by A's body; A's
      -- own `input j` originally referred to aInputs[j], which after
      -- append live at position `bInputs.size + j` — so shift by
      -- `bInputs.size`.
      let bInputs := B.inputs
      let aInputs := A.inputs
      let aShifted := aBody.shiftInputs bInputs.size
      let mut argMap : Array ScalarExp := #[]
      for hJ : j in [0 : bInputs.size] do
        if j == iIdx then
          argMap := argMap.push aShifted
        else
          argMap := argMap.push (.input j)
      let newBody := bBody.subst argMap
      let newInputs := bInputs ++ aInputs
      let newInShapes := bInShapes ++ aInShapes
      let newB : Op BufT CacheT :=
        { prim := Prim.pointwise bOutShape newInShapes newBody
          inputs := newInputs
          outputs := B.outputs }
      -- Produce a new op array: drop A, replace B with newB.  If A is
      -- still needed by any other op (shouldn't happen given the
      -- uniqueness check), we leave it — but here it isn't.
      let mut out : Array (Op BufT CacheT) := #[]
      for hK : k in [0 : ops.size] do
        if k == aIdx then continue
        if k == bIdx then
          out := out.push newB
        else
          match ops[k]? with
          | some opk => out := out.push opk
          | none => pure ()
      return some out
  return none

/-- Collect the set of `input i` indices actually referenced inside a
    ScalarExp.  Used to prune dead input slots after fusion. -/
partial def ScalarExp.usedInputs : ScalarExp → Array Nat
  | .input i      => #[i]
  | .const _      => #[]
  | .add a b      => a.usedInputs ++ b.usedInputs
  | .sub a b      => a.usedInputs ++ b.usedInputs
  | .mul a b      => a.usedInputs ++ b.usedInputs
  | .div a b      => a.usedInputs ++ b.usedInputs
  | .neg a        => a.usedInputs
  | .rsqrt a      => a.usedInputs
  | .exp a        => a.usedInputs
  | .tanh a       => a.usedInputs
  | .gelu a       => a.usedInputs
  | .silu a       => a.usedInputs

/-- Rewrite every `input i` inside `e` using the mapping `remap[i]`. -/
private partial def renumberInputs (remap : Array Nat) : ScalarExp → ScalarExp
  | .input i      => .input (remap[i]!)
  | .const v      => .const v
  | .add a b      => .add (renumberInputs remap a) (renumberInputs remap b)
  | .sub a b      => .sub (renumberInputs remap a) (renumberInputs remap b)
  | .mul a b      => .mul (renumberInputs remap a) (renumberInputs remap b)
  | .div a b      => .div (renumberInputs remap a) (renumberInputs remap b)
  | .neg a        => .neg (renumberInputs remap a)
  | .rsqrt a      => .rsqrt (renumberInputs remap a)
  | .exp a        => .exp (renumberInputs remap a)
  | .tanh a       => .tanh (renumberInputs remap a)
  | .gelu a       => .gelu (renumberInputs remap a)
  | .silu a       => .silu (renumberInputs remap a)

/-- Drop input slots from a pointwise op that the body never references.
    Builds a permutation `old→new` and substitutes `input i` accordingly.
    Safe for correctness: unused slots are by definition irrelevant. -/
private def compactPointwiseInputs {BufT CacheT : Type} (op : Op BufT CacheT)
    : Op BufT CacheT := Id.run do
  match op.prim with
  | .pointwise outShape inShapes body =>
    let used := body.usedInputs
    let mut newSlot : Array Nat := Array.replicate op.inputs.size 0
    let mut newInputs : Array TensorRef := #[]
    let mut newInShapes : Array Shape := #[]
    let mut next : Nat := 0
    for i in [0 : op.inputs.size] do
      let keep := used.any (· == i)
      match keep, op.inputs[i]?, inShapes[i]? with
      | true, some tr, some sh =>
        newSlot := newSlot.set! i next
        newInputs := newInputs.push tr
        newInShapes := newInShapes.push sh
        next := next + 1
      | _, _, _ => pure ()
    return { op with
      prim := .pointwise outShape newInShapes (renumberInputs newSlot body)
      inputs := newInputs }
  | _ => return op

/-- Apply `fusePointwiseStep` until fixpoint, then compact every
    pointwise op's input list.  `protectedIds` lists TensorRef ids the
    caller promises to use externally (the `buffers` argument in
    `runCached`-style APIs): they cannot be fused away.
    Termination: each step reduces op count by 1, bounded by
    `ops.size`. -/
def fusePointwise {BufT CacheT : Type}
    (ops : Array (Op BufT CacheT)) (protectedIds : Array Nat)
    : Array (Op BufT CacheT) := Id.run do
  let mut current := ops
  for _ in [0 : ops.size] do
    match fusePointwiseStep current protectedIds with
    | some next => current := next
    | none      => break
  return current.map compactPointwiseInputs

/-! ## Lowering for OpExt -/

/-- Stage 2 compile: run fusePointwise + mergeSameDispatch, then
    lower each OpExt into an `OpClosure`.  `protectedIds` are the
    tensor ids the caller guarantees to supply a buffer for on replay
    (externals + caller-facing outputs); the fusion pass will not
    collapse them away even if they have a unique internal consumer.

    Allocates per-op IO refs (prepared dispatches, pointwise caches) in
    IO context so the replay closures can close over them. -/
def compileWithPasses [GPUBackend β]
    (ctx : β)
    (state : CircuitState (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (protectedIds : Array Nat := #[])
    : IO (CompiledCircuit β) := do
  -- External tensor ids are always protected — they come from the
  -- caller so we can't legally inline them away.
  let mergedProtected : Array Nat :=
    state.externals.foldl (init := protectedIds) (fun acc (tr, _) => acc.push tr.id)
  let opsFused := fusePointwise state.ops mergedProtected
  let opsExt := mergeSameDispatch opsFused

  -- Allocate intermediate buffers for op outputs the caller didn't
  -- reserve (e.g. the scalar output of a reduceLastAxis that feeds a
  -- fused pointwise tail).  One-shot alloc at compile time; persists
  -- for the life of the CompiledCircuit.
  let callerIds : Array Nat :=
    state.externals.foldl (init := protectedIds) (fun acc (tr, _) => acc.push tr.id)
  let mut baseBuffers : List (Nat × GPUBackend.Buf β) := []
  for op in opsExt do
    for outTr in op.outputs do
      if !(callerIds.any (· == outTr.id)) &&
         !(baseBuffers.any (·.1 == outTr.id)) then
        -- f32 size for now; extend if other dtypes land.
        let bytes : USize := (outTr.shape.numel * 4).toUSize
        let buf ← GPUBackend.allocBuffer (β := β) ctx bytes
        baseBuffers := (outTr.id, buf) :: baseBuffers

  -- For each fusedKV op we need a fresh `IO.Ref` allocated *here*
  -- (in IO context) so the closure can close over it.
  let mut closures : Array (OpClosure β) := #[]
  for op in opsExt do
    match op.prim with
    | PrimExt.fusedKV wK wV =>
      let preparedRef : IO.Ref (Option (GPUBackend.CachedDispatch β)) ← IO.mkRef none
      match op.inputs[0]?, op.outputs[0]?, op.outputs[1]? with
      | some inT, some outK, some outV =>
        let inId := inT.id
        let outKId := outK.id
        let outVId := outV.id
        closures := closures.push
          { run := fun lookup => do
              match lookup inId, lookup outKId, lookup outVId with
              | some inputBuf, some kBuf, some vBuf =>
                Hesper.Layers.Linear.forwardFusedKV (β := β) ctx wK wV inputBuf kBuf vBuf preparedRef
              | _, _, _ =>
                throw (IO.userError s!"compileWithPasses: missing buffer for fusedKV")
          }
      | _, _, _ => throw (IO.userError "compileWithPasses: fusedKV op missing in/out tensor")
    | PrimExt.base (Prim.matmulQ4K layer) =>
      match op.inputs[0]?, op.outputs[0]? with
      | some inT, some outT =>
        let inId := inT.id
        let outId := outT.id
        closures := closures.push
          { run := fun lookup => do
              match lookup inId, lookup outId with
              | some inputBuf, some outputBuf =>
                Hesper.Layers.Linear.LinearLayer.forward (β := β) ctx layer inputBuf outputBuf
              | _, _ =>
                throw (IO.userError s!"compileWithPasses: missing buffer (in={inId}, out={outId})")
          }
      | _, _ => throw (IO.userError "compileWithPasses: matmulQ4K op missing in/out tensor")
    | PrimExt.base (Prim.pointwise outShape inShapes body) =>
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
              | none   => throw (IO.userError s!"compileWithPasses: missing pointwise input id={id}")
            match lookup outId with
            | some outBuf =>
              runPointwiseOp ctx numel inShapes body inBufs outBuf cacheKey cacheRef
            | none => throw (IO.userError s!"compileWithPasses: missing pointwise output id={outId}")
        }
    | PrimExt.base (Prim.reduceLastAxis rop inShape) =>
      let D := inShape.numel
      let inId  := op.inputs[0]!.id
      let outId := op.outputs[0]!.id
      let cacheRef : IO.Ref (Option (GPUBackend.CachedDispatch β)) ← IO.mkRef none
      let cacheKey : UInt64 := hash ("circuit-reduce", reprStr rop, D)
      closures := closures.push
        { run := fun lookup => do
            match lookup inId, lookup outId with
            | some inBuf, some outBuf =>
              runReduceOp ctx rop D inBuf outBuf cacheKey cacheRef
            | _, _ =>
              throw (IO.userError s!"compileWithPasses: missing reduce buffer (in={inId} out={outId})")
        }
  let externalIds := state.externals.map (fun (tr, _) => tr.id)
  let producedIds := state.ops.foldl (init := (#[] : Array Nat)) fun acc op =>
    op.outputs.foldl (init := acc) fun acc' tr => acc'.push tr.id
  return { ops := closures, externalIds, producedIds, baseBuffers }

/-- Build-once-replay-many with fusion passes enabled.  Drop-in for
    `runCached` when the caller wants the Stage 2 fusion behaviour. -/
def runCachedFused [GPUBackend β]
    (ctx : β)
    (cacheRef : IO.Ref (Option (CompiledCircuit β)))
    (build : CircuitM (GPUBackend.Buf β) (GPUBackend.CachedDispatch β) Unit)
    (buffers : List (Nat × GPUBackend.Buf β))
    : IO Unit := do
  match ← cacheRef.get with
  | some cc => cc.replay buffers
  | none =>
    let (_, st) := CircuitM.run build
    -- Every tensor id the caller wires on replay is protected from
    -- pointwise fusion: it must survive as a real buffer.
    let protectedIds : Array Nat := (buffers.map (·.1)).toArray
    let cc ← compileWithPasses ctx st protectedIds
    cacheRef.set (some cc)
    cc.replay buffers

end Hesper.Circuit
