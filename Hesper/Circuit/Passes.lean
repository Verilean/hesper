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
    | none =>
      out := out.push { prim := PrimExt.base op.prim, inputs := op.inputs, outputs := op.outputs }
      i := i + 1
  return out

/-! ## Lowering for OpExt -/

/-- Stage 2 compile: run mergeSameDispatch, then lower each OpExt into
    an `OpClosure`.  Allocates the per-fusedKV `preparedRef`s in IO
    context so they persist across replay calls. -/
def compileWithPasses [GPUBackend β]
    (ctx : β)
    (state : CircuitState (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    : IO (CompiledCircuit β) := do
  let opsExt := mergeSameDispatch state.ops
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
  let externalIds := state.externals.map (fun (tr, _) => tr.id)
  let producedIds := state.ops.foldl (init := (#[] : Array Nat)) fun acc op =>
    op.outputs.foldl (init := acc) fun acc' tr => acc'.push tr.id
  return { ops := closures, externalIds, producedIds }

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
    let cc ← compileWithPasses ctx st
    cacheRef.set (some cc)
    cc.replay buffers

end Hesper.Circuit
