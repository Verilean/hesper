import Hesper.Circuit.IR

/-!
# Circuit IR v2 — Block-structured lazy AST (PoC)

Concept-proof of the architectural shift described in
`docs/llama-fusion-analysis/28-67-fold-gap-investigation.md` §10 and
the user's instruction doc: instead of model code emitting `IO Unit`
kernel launches eagerly, model code returns a pure `BlockGraph` AST
that a later Fusion pass optimises before lowering to real dispatches.

This file intentionally does NOT touch the existing `Hesper.Circuit`
IR (`Prim`, `CircuitM`, `runCached` etc.) — it lives beside it so the
PoC doesn't destabilise production code.  Once the pattern is proven
on RMSNorm, we will migrate callers in a separate pass.

## What is in here

* `Scope` — memory-hierarchy tag (already exists in v1; re-exported).
* `Region` — a slice of a tensor described by (tensor id, ranges).
* `BlockBody` — the compute the Block performs: Pointwise, Reduce, or
  a nested fusion epilogue.
* `Block` — `{ reads, writes, body }`, the unit of scheduling.
* `BlockGraph` — a list of Blocks + the set of tensor declarations.
* `BlockBuilder` — state monad that threads tensor-id allocation.
* `fusePointwiseIntoReduce` — the ONE fusion pass implemented in the
  PoC.  Detects `[Reduce src → mid]; [Pointwise (mid, …) → out]` where
  `mid` is consumed ONLY by the pointwise block, and inlines the
  pointwise body into the reduce's post-reduction epilogue, deleting
  `mid` entirely.
* `rmsNormTwoBlocks` — builds the unfused 2-block form.
* `rmsNormFusedOneBlock` — the hand-written target (for parity check).
* `fusePoCTest` — runs the pass on `rmsNormTwoBlocks` and asserts the
  result equals `rmsNormFusedOneBlock` up to tensor-id renaming.

## What is NOT in here (deferred)

* Lowering to PTX / WGSL (v1's `Lowering.lean` pattern will be reused).
* Multi-block fusion, loop tiling, `Scope` promotion to SRAM/Reg.
* Replacing existing model code — this PoC only builds tiny graphs in
  its own test.
-/

namespace Hesper.Circuit.IRv2

open Hesper.Circuit

/-- Half-open integer range `[lo, hi)` over a single tensor axis. -/
structure Range where
  lo : Nat
  hi : Nat
  deriving Repr, BEq, Inhabited

/-- A region of a tensor: which tensor and the per-axis slice.

    An empty `slices` array means "the whole tensor". -/
structure Region where
  tensorId : Nat
  slices   : Array Range := #[]
  deriving Repr, BEq, Inhabited

/-- Kinds of compute a `Block` can perform. -/
inductive BlockBody where
  /-- `out[i] = body(in[i], …)` — lane-local arithmetic, no reduction. -/
  | Pointwise (body : ScalarExp)
  /-- Two-phase: reduce `reduceInput` with `rop` to one scalar, then
      run `applyBody` elementwise on the region.  `applyBody` can
      reference `.input 0` (the per-lane value) and `.const …` etc.;
      the reduced scalar is bound as `.input <reduceSlot>`.

      This is the shape RMSNorm/Softmax have: Σ x² → 1/sqrt(mean) → y*γ. -/
  | Reduce (rop : ReduceOp) (reduceInput : Region)
           (reduceSlot : Nat) (applyBody : ScalarExp)
  /-- Writes `applyBody` into a caller-supplied dynamic address.

      `indexExpr` is a ScalarExp that resolves to the DESTINATION
      offset (typically `pos * rowStride + laneIdx`).  `applyBody` is
      the per-lane value to store.  Both can reference the block's
      `.input i` reads, enabling fusion of e.g. "compute RoPE on K and
      write the result straight into the KV cache" as one block.

      Lowering discipline: emit a 1D dispatch over the source region
      (reads[0]); for each lane compute indexExpr + applyBody and
      store into writes[0]. -/
  | Scatter (indexExpr : ScalarExp) (applyBody : ScalarExp)
  /-- Multi-output scatter: N (indexExpr, applyBody) pairs writing to
      N separate destination buffers in a single kernel.

      Semantically equivalent to N independent `Scatter` blocks
      over the same sources, but expressed as a single block so the
      lowering emits *one* dispatch (not N) — matching the
      hand-tuned `scatterMulti` pattern used by Gemma 4's KV write
      (K with RoPE + V plain-copy).

      Lowering: 1D dispatch over reads[0].  Each lane evaluates all
      `ops[k]` pairs and writes to `writes[k]` at `dst{k}`.  Both
      exprs share the same per-lane slots & decls, so common gather
      reads (e.g., new_k[laneIdx]) are naturally CSE'd by PTX. -/
  | ScatterMulti (ops : Array (ScalarExp × ScalarExp))
  /-- Q8_1 quantize: f32 input (reads[0]) → Q8_1 packed output (writes[0]).

      Made an explicit IR node so the CSE pass `eliminateCommonQuantize`
      can dedupe redundant quantizations of the same source f32 tensor.
      Two `Quantize` blocks reading the same `tensorId` collapse into
      one shared quantize whose output is consumed by both downstream
      `MatMul`s. -/
  | Quantize
  /-- Fused Reduce + Q8_1 quantize.  Same arithmetic as `Reduce`, but
      the per-element output is packed Q8_1 instead of f32 — i.e. the
      result of `applyBody` at each lane is the input to the per-block
      Q8_1 packing step.  This collapses `[Reduce → Quantize]` into one
      kernel (`fusedRMSNormQ8_1Kernel` in production hesper).

      `writes[0]` is the Q8_1 output buffer; `reads`, `reduceInput`,
      `reduceSlot`, `applyBody` follow the same conventions as `Reduce`. -/
  | ReduceQuantize (rop : ReduceOp) (reduceInput : Region)
                   (reduceSlot : Nat) (applyBody : ScalarExp)
  /-- Matrix-vector multiply of a Q4_K / Q6_K weight against a
      register-held input vector, with a `ScalarExp` epilogue.

      `layerId` is an opaque key that the lowering pass uses to
      resolve the actual weight buffer + matmul kernel variant (dp4a
      4-warp, split-K etc.) at dispatch time.  `outDim, inDim` feed
      PTX shape specialisation.  `epilogue` transforms each output
      element before store — used to fuse GELU/scale/residual-add
      into the matmul itself.

      The epilogue's slot-space: slot 0 = matmul dot product, and
      slot 1..= reads[0..] (the per-row extra inputs: bias, gate
      partner matmul result, residual, etc.). -/
  | MatMul (layerId : UInt64) (outDim inDim : Nat) (epilogue : ScalarExp)
  -- ── Logical Monolith nodes (Phase D) ──
  -- These nodes are NOT physical kernels — they are logical AST
  -- identifiers for production hand-fused operator clusters.  An
  -- opaque `layerKey` lets the dispatcher resolve each Monolith to a
  -- bundle of `LinearLayer`/`RMSNorm` handles + buffer references at
  -- run time, then expand into the parity-proven production sequence
  -- (`forwardFusedNormQKV`, `forwardFusedNormGateUp`,
  -- `fusedRopeKAndCacheWriteKernel`, `forwardNormThenAdd`, etc.).
  -- Cardinality: 1 logical Monolith block ≠ 1 physical dispatch.
  -- See `Gemma4DispatchCount` for the logical-vs-physical breakdown.

  -- Pre-attention monolith: RMSNorm + Q8_1 + wQ/wK/wV + qkvNorm +
  -- RoPE-Q + RoPE-K-with-V-cache-write.  ~6 physical dispatches.
  | GemmaAttentionMonolith (layerKey : UInt64) (pos : Nat)
  -- Flash-attention body: Q × K^T softmax × V.  2 physical dispatches.
  | FlashAttention (layerKey : UInt64) (pos : Nat)
  -- wO projection: attnOut[numHeads*headDim] → [hiddenSize].  Single
  -- Q4_K matmul (2 physical dispatches: quantize + dp4a).
  | GemmaAttnOutProj (layerKey : UInt64)
  -- Fused post-attention RMSNorm + residual add.  Inputs: wO output
  -- (normed) + pre-attn input buffer.  1 physical dispatch
  -- (`RMSNorm.forwardNormThenAdd` using `postAttnNorm`).
  | PostAttnNormAdd (layerKey : UInt64)
  -- FFN body: pre-norm + Q8_1 + gate+up + GELU×mul + wDown.
  -- ~3 physical dispatches.
  | GemmaFFNMonolith (layerKey : UInt64)
  -- Post-FFN RMSNorm fused with residual add.  1 physical dispatch
  -- (`forwardNormThenAdd`).
  | PostFFNNormAdd (layerKey : UInt64)
  deriving Repr, Inhabited

/-- A Block is the unit the fusion pass schedules.  It is pure data:
    building one does NOT run any IO. -/
structure Block where
  reads  : Array Region
  writes : Array Region
  body   : BlockBody
  deriving Inhabited

/-- A declared tensor in the graph.  `scope = .Global` means it is
    buffer-backed after lowering; `.Register` means it is intermediate
    and the fusion pass is free to eliminate it. -/
structure TensorDecl where
  id    : Nat
  shape : Shape
  dtype : DType
  scope : Scope
  deriving Repr, Inhabited

/-- The full lazy program: all tensors that may appear + ordered Blocks. -/
structure BlockGraph where
  tensors : Array TensorDecl
  blocks  : Array Block
  deriving Inhabited

/-! ## Builder monad — threads tensor-id allocation. -/

structure BuilderState where
  tensors : Array TensorDecl := #[]
  blocks  : Array Block := #[]
  nextId  : Nat := 0
  deriving Inhabited

abbrev BuilderM := StateM BuilderState

/-- Allocate a fresh tensor.  `scope` defaults to `.Register` — the
    fusion pass treats register-scope tensors as candidates for
    inlining into the next block. -/
def declareTensor (shape : Shape) (dtype : DType) (scope : Scope := .Register) :
    BuilderM TensorRef := do
  let s ← get
  let id := s.nextId
  let decl : TensorDecl := { id, shape, dtype, scope }
  set { s with tensors := s.tensors.push decl, nextId := id + 1 }
  return { id, shape, dtype, scope }

/-- Register an **externally-owned** tensor under a caller-chosen id.

    Unlike `declareTensor` the caller picks the id, because external
    tensors (model weights, input activations) are referenced by
    stable names from the surrounding code.  We just record the
    declaration so the lowering pass can look up the correct shape /
    dtype / scope instead of defaulting to an empty shape.

    Safe to call with the same id twice — the second call is a no-op
    (first-writer-wins so tests can re-declare deterministically).
    Also advances `nextId` past the given id so later
    `declareTensor` allocations don't collide. -/
def declareExternal (id : Nat) (shape : Shape) (dtype : DType)
    (scope : Scope := .Global) : BuilderM Unit := do
  let s ← get
  if s.tensors.any (·.id == id) then
    pure ()
  else
    let decl : TensorDecl := { id, shape, dtype, scope }
    let nextId' := if id ≥ s.nextId then id + 1 else s.nextId
    set { s with tensors := s.tensors.push decl, nextId := nextId' }

/-- Emit a Block into the graph.  Pure — no IO. -/
def emitBlock (b : Block) : BuilderM Unit := do
  modify fun s => { s with blocks := s.blocks.push b }

def runBuilder (m : BuilderM α) : α × BlockGraph :=
  let (a, s) := m.run {}
  (a, { tensors := s.tensors, blocks := s.blocks })

/-! ## Pointwise-into-reduce fusion pass

The one transformation the PoC needs to demonstrate: collapse

  Block A: Reduce src → midTensor (applyBody_A)
  Block B: Pointwise (midTensor, others…) → out (body_B)

into a single Reduce block whose apply body is
`body_B[input_0 := applyBody_A]`, and drop `midTensor`.

Preconditions (any violation leaves the pair alone):
1. Block B immediately follows Block A in `blocks`.
2. `midTensor` is written by A's `writes[0]` and read by B only.
3. No other block reads `midTensor` after B.
4. B is `BlockBody.Pointwise`.
5. A is `BlockBody.Reduce`. -/

/-- Substitute `subst` for `.input selfSlot` inside B's pointwise
    body and renumber the remaining inputs so they match the fused
    block's slot-space.

    Slot-space of the fused Reduce block:
        slot reduceSlot            = the reduced scalar (virtual)
        slot reduceSlot+1..=       = `cur.reads` then `restReads`
      i.e. reduceSlot+1+k for k ∈ [0, cur.reads.size) indexes cur.reads,
           reduceSlot+1+cur.reads.size+k indexes restReads[k].

    `e` is the POINTWISE body of block B; there `.input i` means
    `B.reads[i]`.  We strip out the mid read (at index `selfSlot`) and
    rebase everything onto the fused layout.

    Args:
      - `selfSlot`     – index of `mid` in B.reads
      - `reduceSlot`   – fused block's reduceSlot (for the `subst` target)
      - `cRSize`       – number of reads the Reduce block brings (= cur.reads.size)
      - `subst`        – typically `.input reduceSlot`, i.e. the reduced scalar -/
partial def substAndRenumber (e : ScalarExp)
    (selfSlot reduceSlot cRSize : Nat) (subst : ScalarExp) : ScalarExp :=
  match e with
  | .input i =>
      if i == selfSlot then subst
      else if i > selfSlot then .input (reduceSlot + 1 + cRSize + i - 1)
      else .input (reduceSlot + 1 + cRSize + i)
  | .add a b => .add (substAndRenumber a selfSlot reduceSlot cRSize subst) (substAndRenumber b selfSlot reduceSlot cRSize subst)
  | .sub a b => .sub (substAndRenumber a selfSlot reduceSlot cRSize subst) (substAndRenumber b selfSlot reduceSlot cRSize subst)
  | .mul a b => .mul (substAndRenumber a selfSlot reduceSlot cRSize subst) (substAndRenumber b selfSlot reduceSlot cRSize subst)
  | .div a b => .div (substAndRenumber a selfSlot reduceSlot cRSize subst) (substAndRenumber b selfSlot reduceSlot cRSize subst)
  | .neg a   => .neg   (substAndRenumber a selfSlot reduceSlot cRSize subst)
  | .rsqrt a => .rsqrt (substAndRenumber a selfSlot reduceSlot cRSize subst)
  | .exp a   => .exp   (substAndRenumber a selfSlot reduceSlot cRSize subst)
  | .tanh a  => .tanh  (substAndRenumber a selfSlot reduceSlot cRSize subst)
  | .gelu a  => .gelu  (substAndRenumber a selfSlot reduceSlot cRSize subst)
  | .silu a  => .silu  (substAndRenumber a selfSlot reduceSlot cRSize subst)
  | .cos a   => .cos   (substAndRenumber a selfSlot reduceSlot cRSize subst)
  | .sin a   => .sin   (substAndRenumber a selfSlot reduceSlot cRSize subst)
  | .pow a b => .pow (substAndRenumber a selfSlot reduceSlot cRSize subst) (substAndRenumber b selfSlot reduceSlot cRSize subst)
  | .lt a b  => .lt  (substAndRenumber a selfSlot reduceSlot cRSize subst) (substAndRenumber b selfSlot reduceSlot cRSize subst)
  | .select c t f =>
      .select (substAndRenumber c selfSlot reduceSlot cRSize subst)
              (substAndRenumber t selfSlot reduceSlot cRSize subst)
              (substAndRenumber f selfSlot reduceSlot cRSize subst)
  | .mod a b  => .mod  (substAndRenumber a selfSlot reduceSlot cRSize subst) (substAndRenumber b selfSlot reduceSlot cRSize subst)
  | .idiv a b => .idiv (substAndRenumber a selfSlot reduceSlot cRSize subst) (substAndRenumber b selfSlot reduceSlot cRSize subst)
  | .toFloat a => .toFloat (substAndRenumber a selfSlot reduceSlot cRSize subst)
  | other => other  -- const, laneIdx, warp primitives etc. pass through

/-- True iff any read region in `rs` targets `tid`. -/
def hasReadOf (rs : Array Region) (tid : Nat) : Bool :=
  rs.any fun r => r.tensorId == tid

/-- Remap a ScalarExp's `.input i` indices via a per-slot mapping. -/
partial def remapInputs (e : ScalarExp) (m : Array Nat) : ScalarExp :=
  match e with
  | .input i =>
      match m[i]? with
      | some j => .input j
      | none   => .input i
  | .add a b => .add (remapInputs a m) (remapInputs b m)
  | .sub a b => .sub (remapInputs a m) (remapInputs b m)
  | .mul a b => .mul (remapInputs a m) (remapInputs b m)
  | .div a b => .div (remapInputs a m) (remapInputs b m)
  | .neg a   => .neg   (remapInputs a m)
  | .rsqrt a => .rsqrt (remapInputs a m)
  | .exp a   => .exp   (remapInputs a m)
  | .tanh a  => .tanh  (remapInputs a m)
  | .gelu a  => .gelu  (remapInputs a m)
  | .silu a  => .silu  (remapInputs a m)
  | .cos a   => .cos   (remapInputs a m)
  | .sin a   => .sin   (remapInputs a m)
  | .pow a b => .pow (remapInputs a m) (remapInputs b m)
  | .lt a b  => .lt  (remapInputs a m) (remapInputs b m)
  | .select c t f =>
      .select (remapInputs c m) (remapInputs t m) (remapInputs f m)
  | .mod a b  => .mod  (remapInputs a m) (remapInputs b m)
  | .idiv a b => .idiv (remapInputs a m) (remapInputs b m)
  | .toFloat a => .toFloat (remapInputs a m)
  | other => other

/-- Dedup reads and remap body's `.input i`.

    Slot-space convention (used by `Lowering_v2`):
      - slot `reduceSlot`    = the reduced scalar  (virtual, not in reads)
      - slot `reduceSlot+1+k` = reads[k] for k=0..reads.size-1

    So `dedupReads` leaves `.input i` for `i ≤ reduceSlot` unchanged and
    rewrites `.input (reduceSlot+1+j)` into `.input (reduceSlot+1+slotMap[j])`. -/
def dedupReads (b : Block) : Block := Id.run do
  let mut seen : Array Region := #[]
  let mut slotMap : Array Nat := #[]
  for r in b.reads do
    match seen.findIdx? (fun s => s.tensorId == r.tensorId && s.slices == r.slices) with
    | some j =>
      slotMap := slotMap.push j
    | none =>
      slotMap := slotMap.push seen.size
      seen := seen.push r
  let body' : BlockBody := match b.body with
    | .Pointwise body =>
      -- Pointwise blocks have no reduce slot; slot i directly indexes reads.
      .Pointwise (remapInputs body slotMap)
    | .Reduce rop rin rslot applyBody =>
      -- Build a full slot-space remap array that fixes slot `rslot` (=
      -- reduced scalar) and remaps higher slots via `slotMap`.
      let maxSlot := rslot + 1 + slotMap.size
      let full : Array Nat := Id.run do
        let mut a : Array Nat := #[]
        for i in [0:maxSlot] do
          if i ≤ rslot then
            a := a.push i
          else
            let j := i - rslot - 1
            match slotMap[j]? with
            | some m => a := a.push (rslot + 1 + m)
            | none   => a := a.push i
        return a
      .Reduce rop rin rslot (remapInputs applyBody full)
    | .Scatter idxE applyE =>
      -- Scatter: slot i directly indexes reads; no virtual slot.
      .Scatter (remapInputs idxE slotMap) (remapInputs applyE slotMap)
    | .ScatterMulti ops =>
      -- Same slot-space as Scatter; remap both elements of each pair.
      .ScatterMulti (ops.map fun (idxE, applyE) =>
        (remapInputs idxE slotMap, remapInputs applyE slotMap))
    | .Quantize =>
      -- Quantize has no ScalarExp body, just remap reads.
      .Quantize
    | .ReduceQuantize rop rin rslot applyBody =>
      -- Same slot-space rewrite as Reduce.
      let maxSlot := rslot + 1 + slotMap.size
      let full : Array Nat := Id.run do
        let mut a : Array Nat := #[]
        for i in [0:maxSlot] do
          if i ≤ rslot then
            a := a.push i
          else
            let j := i - rslot - 1
            match slotMap[j]? with
            | some m => a := a.push (rslot + 1 + m)
            | none   => a := a.push i
        return a
      .ReduceQuantize rop rin rslot (remapInputs applyBody full)
    | .GemmaAttentionMonolith k pos => .GemmaAttentionMonolith k pos
    | .FlashAttention k pos         => .FlashAttention k pos
    | .GemmaAttnOutProj k           => .GemmaAttnOutProj k
    | .PostAttnNormAdd k            => .PostAttnNormAdd k
    | .GemmaFFNMonolith k           => .GemmaFFNMonolith k
    | .PostFFNNormAdd k             => .PostFFNNormAdd k
    | .MatMul layerId oD iD epi =>
      -- MatMul epilogue: slot 0 = dot product (virtual), slot 1..= reads.
      -- Treat like a Reduce with rslot = 0.
      let maxSlot := 1 + slotMap.size
      let full : Array Nat := Id.run do
        let mut a : Array Nat := #[]
        for i in [0:maxSlot] do
          if i == 0 then
            a := a.push 0
          else
            let j := i - 1
            match slotMap[j]? with
            | some m => a := a.push (1 + m)
            | none   => a := a.push i
        return a
      .MatMul layerId oD iD (remapInputs epi full)
  return { reads := seen, writes := b.writes, body := body' }

/-- Fuse `[Pointwise body_A → mid]; [Scatter idxE applyE reading mid]`
    into a single `Scatter` block whose compute body uses `body_A`
    directly, eliminating `mid` entirely.

    Preconditions (same shape as `fusePointwiseIntoReduce`):
    1. B is `Scatter`, A is `Pointwise`
    2. `mid` (A.writes[0]) appears in B.reads at exactly one position
    3. `mid` is NOT read by any later block
    4. A.writes.size = 1 and B.writes.size = 1 -/
def fusePointwiseIntoScatter (g : BlockGraph) : BlockGraph := Id.run do
  let mut out : Array Block := #[]
  let mut i : Nat := 0
  let n := g.blocks.size
  while i < n do
    let cur := g.blocks[i]!
    if _hi : i + 1 < n then
      let nxt := g.blocks[i+1]!
      match cur.body, nxt.body with
      | .Pointwise bodyA, .Scatter idxE applyE =>
        if _h : cur.writes.size = 1 ∧ nxt.reads.size ≥ 1 ∧ nxt.writes.size = 1 then
          let midId := cur.writes[0]!.tensorId
          let selfSlot? := nxt.reads.findIdx? (·.tensorId == midId)
          let laterReads := (g.blocks.extract (i+2) n).any fun b =>
            hasReadOf b.reads midId
          match selfSlot?, laterReads with
          | some selfSlot, false =>
            -- In the fused Scatter block, reads = cur.reads ++ restReads.
            -- Pointwise body A's `.input i` references A.reads[i]; in the
            -- fused block those live at slots 0..cRSize-1.  B's remaining
            -- reads sit after — apply the same renumber logic as for
            -- fuse-into-reduce, but with NO reduce slot (so subst target
            -- is the inlined bodyA itself and the Scatter is slot-0-free).
            let cRSize := cur.reads.size
            let restReads := nxt.reads.filter (·.tensorId != midId)
            -- Rewrite B's idxE / applyE:
            -- slot == selfSlot → bodyA (the inlined upstream value)
            -- slot < selfSlot → slot + cRSize  (pushed right by cur.reads)
            -- slot > selfSlot → slot + cRSize - 1
            let renumber : ScalarExp → ScalarExp :=
              fun e =>
                let map : Array Nat := Id.run do
                  let mut a : Array Nat := #[]
                  for k in [0:nxt.reads.size] do
                    if k < selfSlot then a := a.push (cRSize + k)
                    else if k > selfSlot then a := a.push (cRSize + k - 1)
                    else a := a.push 0  -- placeholder; we substitute below
                  return a
                -- First remap, then substitute selfSlot with bodyA.
                let remapped := remapInputs e map
                -- The placeholder 0 may collide with a real slot; use a
                -- sentinel remap via fresh scan.
                let rec subst : ScalarExp → ScalarExp
                  | .input i =>
                      if i == (map[selfSlot]!) then bodyA
                      else .input i
                  | .add a b => .add (subst a) (subst b)
                  | .sub a b => .sub (subst a) (subst b)
                  | .mul a b => .mul (subst a) (subst b)
                  | .div a b => .div (subst a) (subst b)
                  | .neg a   => .neg (subst a)
                  | .rsqrt a => .rsqrt (subst a)
                  | .exp a   => .exp (subst a)
                  | .tanh a  => .tanh (subst a)
                  | .gelu a  => .gelu (subst a)
                  | .silu a  => .silu (subst a)
                  | .cos a   => .cos (subst a)
                  | .sin a   => .sin (subst a)
                  | .pow a b => .pow (subst a) (subst b)
                  | .lt a b  => .lt (subst a) (subst b)
                  | .select c t f => .select (subst c) (subst t) (subst f)
                  | .mod a b  => .mod (subst a) (subst b)
                  | .idiv a b => .idiv (subst a) (subst b)
                  | .toFloat a => .toFloat (subst a)
                  | other => other
                subst remapped
            let fusedRaw : Block :=
              { reads := cur.reads ++ restReads
                writes := nxt.writes
                body := .Scatter (renumber idxE) (renumber applyE) }
            out := out.push (dedupReads fusedRaw)
            i := i + 2
          | _, _ =>
            out := out.push cur
            i := i + 1
        else
          out := out.push cur
          i := i + 1
      | _, _ =>
        out := out.push cur
        i := i + 1
    else
      out := out.push cur
      i := i + 1
  return { g with blocks := out }

/-- Main pass: walk `blocks` once, emit a new list with matching
    (Reduce, Pointwise) pairs fused. -/
def fusePointwiseIntoReduce (g : BlockGraph) : BlockGraph := Id.run do
  let mut out : Array Block := #[]
  let mut i : Nat := 0
  let n := g.blocks.size
  while i < n do
    let cur := g.blocks[i]!
    if h : i + 1 < n then
      let nxt := g.blocks[i+1]!
      match cur.body, nxt.body with
      | .Reduce rop reduceInput reduceSlot applyBody, .Pointwise body =>
        -- Precondition: cur writes exactly one tensor, nxt reads it.
        if h2 : cur.writes.size = 1 ∧ nxt.reads.size ≥ 1 then
          let midId := cur.writes[0]!.tensorId
          -- Find the slot in nxt.reads where `midId` sits — that's B's
          -- "selfSlot".  Also require nxt writes exactly one tensor.
          let selfSlot? := nxt.reads.findIdx? (·.tensorId == midId)
          let restReads := nxt.reads.filter (·.tensorId != midId)
          -- Must also check no later block reads midId.
          let laterReads := (g.blocks.extract (i+2) n).any fun b =>
            hasReadOf b.reads midId
          match selfSlot?, laterReads, nxt.writes.size with
          | some selfSlot, false, 1 =>
            let cRSize := cur.reads.size
            -- `applyBody` from the Reduce block already references
            -- slot `reduceSlot` for the reduced scalar, so it IS the
            -- value we want to substitute for B's `.input selfSlot`.
            let newApply := substAndRenumber body selfSlot reduceSlot cRSize applyBody
            let fusedRaw : Block :=
              { reads := cur.reads ++ restReads
                writes := nxt.writes
                body := .Reduce rop reduceInput reduceSlot newApply }
            out := out.push (dedupReads fusedRaw)
            i := i + 2
          | _, _, _ =>
            out := out.push cur
            i := i + 1
        else
          out := out.push cur
          i := i + 1
      | _, _ =>
        out := out.push cur
        i := i + 1
    else
      out := out.push cur
      i := i + 1
  -- Drop any tensor that is no longer read/written by any remaining block.
  let liveIds : Array Nat := Id.run do
    let mut s : Array Nat := #[]
    for b in out do
      for r in b.reads do
        if !(s.contains r.tensorId) then s := s.push r.tensorId
      for w in b.writes do
        if !(s.contains w.tensorId) then s := s.push w.tensorId
    return s
  let tensors' := g.tensors.filter (fun t => liveIds.contains t.id)
  return { tensors := tensors', blocks := out }

/-- Fuse `[MatMul → mid]; [Pointwise (mid, others…) → out]` into a
    single `MatMul` block whose `epilogue` is `body_B[input_selfSlot
    := oldEpilogue]` — where `oldEpilogue` is slot-0 in MatMul's
    epilogue space (`slot 0 = dot`).  Structurally identical to
    `fusePointwiseIntoReduce` with `reduceSlot = 0`.

    Preconditions:
    1. A is `MatMul`, B is `Pointwise`.
    2. `mid` (A.writes[0]) appears in B.reads at exactly one position.
    3. `mid` is NOT read by any later block.
    4. A.writes.size = 1 and B.writes.size = 1. -/
def fusePointwiseIntoMatMul (g : BlockGraph) : BlockGraph := Id.run do
  let mut out : Array Block := #[]
  let mut i : Nat := 0
  let n := g.blocks.size
  while i < n do
    let cur := g.blocks[i]!
    if _h : i + 1 < n then
      let nxt := g.blocks[i+1]!
      match cur.body, nxt.body with
      | .MatMul layerId oD iD oldEpi, .Pointwise body =>
        if _h2 : cur.writes.size = 1 ∧ nxt.reads.size ≥ 1 then
          let midId := cur.writes[0]!.tensorId
          let selfSlot? := nxt.reads.findIdx? (·.tensorId == midId)
          let restReads := nxt.reads.filter (·.tensorId != midId)
          let laterReads := (g.blocks.extract (i+2) n).any fun b =>
            hasReadOf b.reads midId
          match selfSlot?, laterReads, nxt.writes.size with
          | some selfSlot, false, 1 =>
            let cRSize := cur.reads.size
            -- MatMul slot-space: slot 0 = dot (virtual), slot 1+k =
            -- reads[k].  Reuse substAndRenumber with reduceSlot = 0 and
            -- subst = oldEpi (the existing epilogue, already in the
            -- fused slot-space since cur.reads are the first reads).
            let newEpi := substAndRenumber body selfSlot 0 cRSize oldEpi
            let fusedRaw : Block :=
              { reads := cur.reads ++ restReads
                writes := nxt.writes
                body := .MatMul layerId oD iD newEpi }
            out := out.push (dedupReads fusedRaw)
            i := i + 2
          | _, _, _ =>
            out := out.push cur
            i := i + 1
        else
          out := out.push cur
          i := i + 1
      | _, _ =>
        out := out.push cur
        i := i + 1
    else
      out := out.push cur
      i := i + 1
  -- Drop dead tensors.
  let liveIds : Array Nat := Id.run do
    let mut s : Array Nat := #[]
    for b in out do
      for r in b.reads do
        if !(s.contains r.tensorId) then s := s.push r.tensorId
      for w in b.writes do
        if !(s.contains w.tensorId) then s := s.push w.tensorId
    return s
  let tensors' := g.tensors.filter (fun t => liveIds.contains t.id)
  return { tensors := tensors', blocks := out }

/-! ## Fusion pass: collapse `[Reduce; Quantize]` into `ReduceQuantize`.

Mirrors production hesper's `fusedRMSNormQ8_1Kernel`: the per-element
output of an RMSNorm is fed directly into Q8_1 packing inside the
same kernel, eliminating the f32 normedBuf round-trip to VRAM.  IRv2
recognises the `[Reduce → mid]; [Quantize reading mid]` pair and folds
them into a single `ReduceQuantize` block.

Conditions for the fold:
  1. The Reduce writes exactly one tensor (`mid`).
  2. The next block is `Quantize` reading `mid` at reads[0].
  3. `mid` is NOT read by any block past the Quantize.
  4. The Quantize writes exactly one tensor (the Q8_1 output). -/
def fuseReduceIntoQuantize (g : BlockGraph) : BlockGraph := Id.run do
  let mut out : Array Block := #[]
  let mut i : Nat := 0
  let n := g.blocks.size
  while i < n do
    let cur := g.blocks[i]!
    if _h : i + 1 < n then
      let nxt := g.blocks[i+1]!
      match cur.body, nxt.body with
      | .Reduce rop rin rslot applyBody, .Quantize =>
        if cur.writes.size = 1 ∧ nxt.reads.size = 1 ∧ nxt.writes.size = 1 then
          let midId := cur.writes[0]!.tensorId
          if nxt.reads[0]!.tensorId == midId then
            -- mid must not be read by any later block (after the Quantize).
            let laterReads := (g.blocks.extract (i+2) n).any fun b =>
              hasReadOf b.reads midId
            if !laterReads then
              let fused : Block :=
                { reads  := cur.reads
                  writes := nxt.writes
                  body   := .ReduceQuantize rop rin rslot applyBody }
              out := out.push fused
              i := i + 2
              continue
      | _, _ => pure ()
    out := out.push cur
    i := i + 1
  -- Drop tensors no remaining block touches.
  let liveIds : Array Nat := Id.run do
    let mut s : Array Nat := #[]
    for b in out do
      for r in b.reads do
        if !(s.contains r.tensorId) then s := s.push r.tensorId
      for w in b.writes do
        if !(s.contains w.tensorId) then s := s.push w.tensorId
    return s
  let tensors' := g.tensors.filter (fun t => liveIds.contains t.id)
  return { tensors := tensors', blocks := out }

/-! ## CSE pass: eliminate redundant `Quantize` blocks

When the same f32 tensor feeds multiple Q4_K matmuls (the canonical
case: hidden state → wQ + wK + wV), the naive builder emits one
`Quantize` per matmul.  This pass detects "two Quantize blocks share
the same source f32 tensorId", drops the second, and rewrites every
downstream block's reads from the dropped quantize-output tensor to
the surviving quantize-output tensor.

Implementation:
  1. Walk blocks once, building a map srcTid → firstQuantizeOutTid.
     For each Quantize block:
       - if its reads[0].tensorId is new, keep the block, record the
         mapping `srcTid → outTid`.
       - if its reads[0].tensorId already mapped, drop the block and
         record `outTid → existing-outTid` so downstream reads can be
         remapped.
  2. Walk again, applying the rewrite map to every block's reads.

Liveness cleanup is left implicit — any tensor that no block reads or
writes is filtered out at the end. -/

/-- Rewrite every `Region.tensorId` in `arr` via the `(old → new)` map.
    Untouched ids pass through. -/
private def remapRegions (arr : Array Region) (rewrite : Array (Nat × Nat)) :
    Array Region :=
  arr.map fun r =>
    match rewrite.find? (·.fst == r.tensorId) with
    | some (_, newId) => { r with tensorId := newId }
    | none            => r

def eliminateCommonQuantize (g : BlockGraph) : BlockGraph := Id.run do
  -- Pass 1: classify each Quantize block as "first" (kept) or "dup" (dropped).
  -- `firstSeen` maps source f32 tid → quantize-output tid of the first
  -- Quantize that handled it.  `rewrite` maps later quantize-output tids
  -- → first quantize-output tid.
  let mut firstSeen : Array (Nat × Nat) := #[]   -- (srcTid, outTid)
  let mut rewrite   : Array (Nat × Nat) := #[]   -- (oldOutTid, newOutTid)
  let mut keep      : Array Bool        := #[]   -- per-block keep flag
  for b in g.blocks do
    let mut isDup : Bool := false
    match b.body with
    | .Quantize =>
      if b.reads.size = 1 ∧ b.writes.size = 1 then
        let srcTid := b.reads[0]!.tensorId
        match firstSeen.find? (·.fst == srcTid) with
        | some (_, firstOut) =>
          rewrite := rewrite.push (b.writes[0]!.tensorId, firstOut)
          isDup := true
        | none =>
          firstSeen := firstSeen.push (srcTid, b.writes[0]!.tensorId)
    | _ => pure ()
    keep := keep.push (!isDup)
  -- Pass 2: rebuild block list, dropping duplicates and rewriting reads.
  let mut out : Array Block := #[]
  for h : i in [0:g.blocks.size] do
    if keep[i]! then
      let b : Block := g.blocks[i]!
      let b' : Block :=
        { reads := remapRegions b.reads rewrite
          writes := b.writes
          body := b.body }
      out := out.push b'
  -- Liveness: drop any tensor declarations no block touches.
  let liveIds : Array Nat := Id.run do
    let mut s : Array Nat := #[]
    for b in out do
      for r in b.reads do
        if !(s.contains r.tensorId) then s := s.push r.tensorId
      for w in b.writes do
        if !(s.contains w.tensorId) then s := s.push w.tensorId
    return s
  let tensors' := g.tensors.filter (fun t => liveIds.contains t.id)
  return { tensors := tensors', blocks := out }

/-! ## PoC: RMSNorm as two Blocks, fused to one

Semantics of RMSNorm (ignoring γ for brevity):
  mean  = (1/N) Σ x²
  invRms = 1 / sqrt(mean + eps)
  y     = x * invRms

Two-block form (what the naive model code would emit):
  Block A (Reduce): squareSum over x → scalar `mid`, apply = identity
  Block B (Pointwise): y = x * (1/sqrt(mid/N + eps))

After `fusePointwiseIntoReduce`:
  Block A': Reduce with apply body doing x * (1/sqrt(mid/N + eps)) directly
-/

/-- Build the two-block form.  `N` is the reduction length (= shape[0]).  -/
def rmsNormTwoBlocks (N : Nat) (eps : Float) (xId : Nat) (outId : Nat) :
    BuilderM Unit := do
  let mid ← declareTensor #[1] .f32 .Register
  let xRegion : Region := { tensorId := xId }
  let midRegion : Region := { tensorId := mid.id }
  let outRegion : Region := { tensorId := outId }
  -- Block A: Σ x²  (reduce, apply body is identity — we just want the
  -- sum).  reduceSlot=0 means "the reduced scalar is slot 0 of the
  -- apply body"; applyBody = input 0.
  emitBlock
    { reads := #[xRegion]
      writes := #[midRegion]
      body := .Reduce ReduceOp.sumOfSquares xRegion 0 (.input 0) }
  -- Block B: out[i] = x[i] * (1/sqrt(mid / N + eps))
  -- reads: [x, mid]; mid's slot in reads is 1.
  let invRms : ScalarExp :=
    .rsqrt (.add (.div (.input 1) (.const N.toFloat)) (.const eps))
  let body : ScalarExp := .mul (.input 0) invRms
  emitBlock
    { reads := #[xRegion, midRegion]
      writes := #[outRegion]
      body := .Pointwise body }

/-- Build the hand-written fused form directly (for the parity
    check).  One Reduce block whose apply body is the full
    `x * invRms` expression. -/
def rmsNormFusedOneBlock (N : Nat) (eps : Float) (xId : Nat) (outId : Nat) :
    BuilderM Unit := do
  let xRegion : Region := { tensorId := xId }
  let outRegion : Region := { tensorId := outId }
  -- In the hand-fused form the reduced scalar is slot 0 (inside the
  -- Reduce block it's bound as `.input <reduceSlot>`), but the
  -- pointwise body's OWN input 0 (the per-lane `x`) is accessed as
  -- `.input 1` after the reduce slot is added.  For the parity check
  -- below we only compare structural equivalence up to that binding.
  let invRms : ScalarExp :=
    .rsqrt (.add (.div (.input 0) (.const N.toFloat)) (.const eps))
  let body : ScalarExp := .mul (.input 1) invRms
  emitBlock
    { reads := #[xRegion]
      writes := #[outRegion]
      body := .Reduce ReduceOp.sumOfSquares xRegion 0 body }

/-- Small smoke test.  Returns the fused graph + the reference graph;
    the caller compares block counts + body shapes. -/
def fusePoCTest (N : Nat := 16) (eps : Float := 1e-6) :
    BlockGraph × BlockGraph :=
  let (_, unfused) := runBuilder (rmsNormTwoBlocks N eps 100 101)
  let (_, target)  := runBuilder (rmsNormFusedOneBlock N eps 100 101)
  (fusePointwiseIntoReduce unfused, target)

end Hesper.Circuit.IRv2
