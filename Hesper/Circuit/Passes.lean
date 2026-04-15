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

/-! ## Reduce-with-epilogue fusion

When a `reduceLastAxis` is consumed by a single `pointwise` whose
body uses the reduction's result as a scalar broadcast (slot shape
`#[1]`), we can collapse them into one kernel — `Phase 1`: the
reduction; `Phase 2`: per-lane evaluation of the pointwise body with
the reduction result substituted for `input 0`.  This is the canonical
RMSNorm-style "reduce → use-result-broadcast" pattern.

Output shape of the consumer must equal the reduction's input shape
(the epilogue "expands" back to the reduction's domain).  Other
consumer inputs must be full-shape — no nested broadcasts in the
epilogue tail (they'd require multi-WG handling we don't have yet).

Termination: each step strictly reduces op count by 1. -/

/-- Walk a chain of scalar-only pointwise ops starting from a tensor
    id `startId`.  Returns `(scalarChainExp, finalConsumerIdx,
    finalConsumerSlotIdx)`:

      - `scalarChainExp`: a `ScalarExp` over a single `input 0` (which
        the caller will substitute for the reduction result) that
        represents the composed scalar computation reaching the final
        full-shape consumer.  If `startId` is consumed directly by a
        full-shape op, this is just `.input 0` (identity).
      - `finalConsumerIdx`: index in `ops` of the full-shape pointwise
        op that ultimately consumes the chain.
      - `finalConsumerSlotIdx`: which slot of that op the chain feeds.

    Returns `none` if there's no path to a full-shape consumer (e.g.
    the chain dead-ends, branches, or the consumer is non-pointwise),
    or if the chain has multiple consumers at any step (uniqueness
    required for safe inlining), or if any intermediate op is the
    caller-protected output.

    Each scalar-stage hop must be a pointwise op whose:
      - output shape is `#[1]` (still scalar);
      - inputs are all `#[1]` (purely scalar — no full-shape mixed in);
      - is the unique consumer of its input scalar.
-/
private partial def followScalarChain {BufT CacheT : Type}
    (ops : Array (Op BufT CacheT))
    (protectedIds : Array Nat)
    (startId : Nat)
    (acc : ScalarExp)
    : Option (ScalarExp × Nat × Nat) := Id.run do
  if protectedIds.any (· == startId) then return none
  if countConsumers ops startId != 1 then return none
  -- Find the unique consumer.
  let mut consumerOpt : Option (Nat × { op : Op BufT CacheT // True }) := none
  for cIdx in [0 : ops.size] do
    match ops[cIdx]? with
    | some C =>
      if C.inputs.any (fun tr => tr.id == startId) then
        consumerOpt := some (cIdx, ⟨C, trivial⟩); break
    | none => pure ()
  let some (cIdx, ⟨C, _⟩) := consumerOpt | return none
  let .pointwise cOutShape cInShapes cBody := C.prim | return none
  -- Locate the slot in C that consumes startId.
  let mut slotOpt : Option Nat := none
  for j in [0 : C.inputs.size] do
    match C.inputs[j]? with
    | some tr => if tr.id == startId then slotOpt := some j; break
    | none => pure ()
  let some slot := slotOpt | return none
  if (cInShapes[slot]?.getD #[]) != #[1] then return none
  if cOutShape != #[1] then
    -- Reached the full-shape tail.  All non-startId slots must be
    -- full-shape (no nested broadcasts in the tail).  Return the chain
    -- expression and the consumer position; caller will splice it.
    let mut otherShapesOK := true
    for j in [0 : C.inputs.size] do
      if j != slot then
        if (cInShapes[j]?.getD #[]) != cOutShape then
          otherShapesOK := false
    if !otherShapesOK then return none
    return some (acc, cIdx, slot)
  -- Still scalar-stage.  All inputs must be `#[1]` so the body is
  -- evaluable from `acc` + scalar broadcasts of the OTHER inputs.
  -- For now we only allow chains where the scalar-stage op's only
  -- input is the producer's scalar (no extra broadcast scalars at
  -- intermediate stages — keeps the body composition simple).
  if C.inputs.size != 1 then return none
  -- Compose: substitute acc for `input 0` in cBody.
  let nextAcc := cBody.subst #[acc]
  let cOutId := C.outputs[0]!.id
  followScalarChain ops protectedIds cOutId nextAcc

private def fuseReduceEpilogueStep {BufT CacheT : Type}
    (ops : Array (Op BufT CacheT)) (protectedIds : Array Nat)
    : Option (Array (Op BufT CacheT)) := Id.run do
  for pIdx in [0 : ops.size] do
    match ops[pIdx]? with
    | none => pure ()
    | some P =>
    let .reduceLastAxis rop pInShape := P.prim | continue
    let pOutId := P.outputs[0]!.id
    -- Walk the scalar chain starting from P's output.  acc = `.input 0`
    -- (the reduction's scalar; the caller of the chain function will
    -- bind it to the actual scratch[0] read in the lowering).
    let some (chainExp, qIdx, pSlot) :=
      followScalarChain ops protectedIds pOutId (.input 0) | continue
    let some Q := ops[qIdx]? | continue
    let .pointwise qOutShape qInShapes qBody := Q.prim | continue
    if qOutShape != pInShape then continue
    -- Build the epilogue body: substitute the chain expression at the
    -- pSlot and renumber other slots to 1, 2, …
    let mut argMap : Array ScalarExp := #[]
    let mut nextOther : Nat := 1
    for j in [0 : Q.inputs.size] do
      if j == pSlot then
        argMap := argMap.push chainExp
      else
        argMap := argMap.push (.input nextOther)
        nextOther := nextOther + 1
    let newBody := qBody.subst argMap
    -- Build new op.
    let pIn := P.inputs[0]!
    let mut newInputs : Array TensorRef := #[pIn]
    let mut epiShapes : Array Shape := #[]
    for j in [0 : Q.inputs.size] do
      if j != pSlot then
        match Q.inputs[j]?, qInShapes[j]? with
        | some tr, some sh =>
          newInputs := newInputs.push tr
          epiShapes := epiShapes.push sh
        | _, _ => pure ()
    let newOp : Op BufT CacheT :=
      { prim := Prim.reduceLastAxisWithEpilogue rop pInShape epiShapes newBody
        inputs := newInputs
        outputs := Q.outputs }
    -- Determine which scalar-chain ops to drop: walk the chain from
    -- pOutId again, collect their indices.
    let mut toDrop : Array Nat := #[pIdx, qIdx]
    let mut cursor := pOutId
    while true do
      let mut nextIdx : Option Nat := none
      for kIdx in [0 : ops.size] do
        if kIdx == qIdx then continue  -- Q is the tail, already in toDrop
        match ops[kIdx]? with
        | some opK =>
          if opK.inputs.any (fun tr => tr.id == cursor) then
            nextIdx := some kIdx; break
        | none => pure ()
      match nextIdx with
      | some kIdx =>
        toDrop := toDrop.push kIdx
        match ops[kIdx]? with
        | some opK => cursor := opK.outputs[0]!.id
        | none => break
      | none => break
    -- Splice: drop everything in toDrop, push newOp at qIdx's position.
    let mut out : Array (Op BufT CacheT) := #[]
    for k in [0 : ops.size] do
      if toDrop.any (· == k) then
        if k == qIdx then out := out.push newOp
      else
        match ops[k]? with
        | some opk => out := out.push opk
        | none => pure ()
    return some out
  return none

/-- Iterate `fuseReduceEpilogueStep` until fixpoint. -/
def fuseReduceEpilogue {BufT CacheT : Type}
    (ops : Array (Op BufT CacheT)) (protectedIds : Array Nat)
    : Array (Op BufT CacheT) := Id.run do
  let mut current := ops
  for _ in [0 : ops.size] do
    match fuseReduceEpilogueStep current protectedIds with
    | some next => current := next
    | none      => break
  return current

/-! ## Matmul-epilogue fusion

Detects `[A : Prim.matmulQ4K layer] → [B : Prim.pointwise outShape inShapes body]`
chains where the matmul's output feeds the pointwise body as a **full-shape**
input slot (shape `[layer.outDim]`), and rewrites the pair into a
single `Prim.matmulQ4KWithEpilogue` op.

Refuses to fuse when:
- A's output is caller-protected (appears in the replay `buffers` list);
- A's output has multiple consumers;
- B's output shape ≠ `[layer.outDim]` (shape transformation would
  require a reshape Prim we don't have);
- B consumes A's output via a broadcast slot (shape `[1]`) — the
  current `matmulQ4KWithEpilogue` expects its slot 0 to BE the
  matmul scalar result, not a lane-broadcast of it;
- any OTHER slot of B has shape ≠ outShape (i.e. anything that
  requires a slice offset or a broadcast in the epilogue); the
  matmulQ4KWithEpilogue Prim supports slice offsets but the pass
  doesn't try to infer them — the caller can explicitly use
  `CircuitM.matmulQ4KWithEpilogue` for that case.

Termination: each fire reduces op count by 1. -/

private def fuseMatmulEpilogueStep {BufT CacheT : Type}
    (ops : Array (Op BufT CacheT)) (protectedIds : Array Nat)
    : Option (Array (Op BufT CacheT)) := Id.run do
  let isProtected (id : Nat) : Bool := protectedIds.any (· == id)
  for aIdx in [0 : ops.size] do
    match ops[aIdx]? with
    | none => pure ()
    | some A =>
    let .matmulQ4K layer := A.prim | continue
    let aOutId := A.outputs[0]!.id
    if isProtected aOutId then continue
    if countConsumers ops aOutId != 1 then continue
    -- Find the unique consumer B.
    let mut bIdxOpt : Option Nat := none
    for bIdx in [0 : ops.size] do
      match ops[bIdx]? with
      | some B =>
        if B.inputs.any (fun tr => tr.id == aOutId) then
          bIdxOpt := some bIdx; break
      | none => pure ()
    let some bIdx := bIdxOpt | continue
    let some B := ops[bIdx]? | continue
    let .pointwise bOutShape bInShapes bBody := B.prim | continue
    -- Output shape of B must equal `[outDim]`.  Anything else implies
    -- a reshape the matmul-epilogue kernel can't absorb.
    if bOutShape != #[layer.config.outDim] then continue
    -- Locate the slot in B that consumes A.  Must be full-shape
    -- (broadcast slots would mean "lane-broadcast the matmul scalar",
    -- which doesn't fit the one-scalar-per-row model).
    let mut aSlotOpt : Option Nat := none
    for j in [0 : B.inputs.size] do
      match B.inputs[j]? with
      | some tr =>
        if tr.id == aOutId then aSlotOpt := some j; break
      | none => pure ()
    let some aSlot := aSlotOpt | continue
    if (bInShapes[aSlot]?.getD #[]) != bOutShape then continue
    -- All OTHER slots must be full-shape (== bOutShape) too.  A
    -- future relaxation could derive read offsets from size-mismatched
    -- slots, but that's out of scope.
    let mut otherSlotsValid : Bool := true
    for j in [0 : B.inputs.size] do
      if j != aSlot then
        if (bInShapes[j]?.getD #[]) != bOutShape then
          otherSlotsValid := false
    if !otherSlotsValid then continue
    -- Build the new epilogue body.  Slot mapping:
    --   B's aSlot     →  `input 0` (the matmul scalar)
    --   B's other j   →  `input (kPosition + 1)`
    -- where kPosition is the j-th kept slot in original order
    -- (skipping aSlot).  Collect the same order into newInputs.
    let mut remap : Array Nat := Array.replicate B.inputs.size 0
    let mut nextK : Nat := 1
    for j in [0 : B.inputs.size] do
      if j == aSlot then
        remap := remap.set! j 0
      else
        remap := remap.set! j nextK
        nextK := nextK + 1
    let newBody := renumberInputs remap bBody
    -- Build new op: inputs[0] = A's matmul input;
    -- inputs[1..] = B's other inputs in order.
    let mut newInputs : Array TensorRef := #[A.inputs[0]!]
    let mut epiBufferSizes : Array Nat := #[]
    for j in [0 : B.inputs.size] do
      if j != aSlot then
        match B.inputs[j]?, bInShapes[j]? with
        | some tr, some _ =>
          newInputs := newInputs.push tr
          epiBufferSizes := epiBufferSizes.push layer.config.outDim
        | _, _ => pure ()
    -- All offsets zero: pass only fuses the read-at-outIdx case.
    let epiReadOffsets : Array Nat := Array.replicate epiBufferSizes.size 0
    let newOp : Op BufT CacheT :=
      { prim := Prim.matmulQ4KWithEpilogue layer epiBufferSizes epiReadOffsets newBody
        inputs := newInputs
        outputs := B.outputs }
    -- Splice: drop A (idx aIdx), replace B (idx bIdx) with newOp.
    let mut out : Array (Op BufT CacheT) := #[]
    for k in [0 : ops.size] do
      if k == aIdx then continue
      if k == bIdx then out := out.push newOp
      else
        match ops[k]? with
        | some opk => out := out.push opk
        | none => pure ()
    return some out
  return none

/-- Iterate `fuseMatmulEpilogueStep` to fixpoint. -/
def fuseMatmulEpilogue {BufT CacheT : Type}
    (ops : Array (Op BufT CacheT)) (protectedIds : Array Nat)
    : Array (Op BufT CacheT) := Id.run do
  let mut current := ops
  for _ in [0 : ops.size] do
    match fuseMatmulEpilogueStep current protectedIds with
    | some next => current := next
    | none      => break
  return current

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
  -- Pass order: reduce-with-epilogue first, then pointwise chain
  -- fusion, then matmul same-dispatch merge.  fuseReduceEpilogue
  -- looks for `reduce → (scalar-pointwise chain) → full-shape pointwise`
  -- patterns and collapses them to a single reduce-with-epilogue op;
  -- fusePointwise then handles any remaining same-shape pointwise
  -- chains.
  -- Pass order:
  --   1. fuseMatmulEpilogue: [matmulQ4K → pointwise] → matmulQ4KWithEpilogue
  --   2. fuseReduceEpilogue: [reduceLastAxis → … → pointwise] → reduceLastAxisWithEpilogue
  --   3. fusePointwise: same-shape pointwise chain β-reduction
  --   4. mergeSameDispatch: KV matmul pair → fusedKV
  -- Running matmul-epilogue first lets a following `pointwise` get absorbed
  -- before fusePointwise tries to collapse it into ANOTHER downstream pointwise.
  let opsMmEpi  := fuseMatmulEpilogue state.ops mergedProtected
  let opsRedFused := fuseReduceEpilogue opsMmEpi mergedProtected
  let opsFused := fusePointwise opsRedFused mergedProtected
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
    | PrimExt.base (Prim.matmulQ4KWithEpilogue layer epiBufferSizes epiReadOffsets epiBody) =>
      match op.inputs[0]?, op.outputs[0]? with
      | some inT, some outT =>
        let inId := inT.id
        let outId := outT.id
        let epiIds : Array Nat :=
          (op.inputs.extract 1 op.inputs.size).map (·.id)
        let cacheRef : IO.Ref (Option (GPUBackend.CachedDispatch β)) ← IO.mkRef none
        let cacheKey : UInt64 :=
          hash ("circuit-matmulQ4K-epi",
                layer.config.inDim, layer.config.outDim,
                reprStr epiBody, reprStr epiBufferSizes.toList,
                reprStr epiReadOffsets.toList)
        closures := closures.push
          { run := fun lookup => do
              match lookup inId, lookup outId with
              | some inputBuf, some outputBuf =>
                let mut epiBufs : Array (GPUBackend.Buf β) := #[]
                for id in epiIds do
                  match lookup id with
                  | some b => epiBufs := epiBufs.push b
                  | none => throw (IO.userError s!"compileWithPasses: missing matmul-epi input id={id}")
                runMatmulQ4KWithEpilogueOp ctx layer inputBuf epiBufs
                  epiBufferSizes epiReadOffsets epiBody outputBuf cacheKey cacheRef
              | _, _ =>
                throw (IO.userError s!"compileWithPasses: missing matmul-epi buffer (in={inId}, out={outId})")
          }
      | _, _ => throw (IO.userError "compileWithPasses: matmul-epi op missing in/out tensor")
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
    | PrimExt.base (Prim.reduceLastAxisWithEpilogue rop reduceInShape epiShapes body) =>
      let D := reduceInShape.numel
      let inId := op.inputs[0]!.id
      let outId := op.outputs[0]!.id
      let epiIds : Array Nat :=
        (op.inputs.extract 1 op.inputs.size).map (·.id)
      let cacheRef : IO.Ref (Option (GPUBackend.CachedDispatch β)) ← IO.mkRef none
      let cacheKey : UInt64 :=
        hash ("circuit-reduce-epi", reprStr rop, D, reprStr body,
              reprStr epiShapes.toList)
      closures := closures.push
        { run := fun lookup => do
            match lookup inId, lookup outId with
            | some reduceBuf, some outBuf =>
              let mut epiBufs : Array (GPUBackend.Buf β) := #[]
              for id in epiIds do
                match lookup id with
                | some b => epiBufs := epiBufs.push b
                | none => throw (IO.userError s!"compileWithPasses: missing epi input id={id}")
              runReduceWithEpilogueOp ctx rop D body reduceBuf epiBufs outBuf cacheKey cacheRef
            | _, _ =>
              throw (IO.userError s!"compileWithPasses: missing reduce-epi buffer (in={inId} out={outId})")
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
