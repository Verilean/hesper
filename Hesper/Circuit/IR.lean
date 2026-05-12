import Hesper.Circuit.IRCore
import Hesper.Backend
import Hesper.Layers.Linear

/-!
# Circuit IR — GPU-flavoured pieces (`Prim`, `Op`, `CircuitState`, `CircuitM`)

The pure-Lean data types (`DType`, `Scope`, `TensorRef`,
`ExternalTensor`, `ScalarExp`, `ReduceOp`) live in
`Hesper.Circuit.IRCore` so that pure tooling (e.g. the
`Hesper.Circuit.Eval` reference evaluator) can depend on them
without pulling in `Hesper.Backend` / `Hesper.Layers.Linear`.

This file re-opens the `Hesper.Circuit` namespace and adds the
constructions that **do** depend on the GPU stack: `Prim`, `Op`,
`CircuitState`, and the `CircuitM` builder.
-/

namespace Hesper.Circuit

/-- A typed description of a primitive operation.  Each Prim has a
    well-defined (input shapes, output shapes, dispatch shape) mapping
    that the lowering pass consumes. -/
inductive Prim (BufT : Type) (CacheT : Type) where
  /-- Matmul against a Q4_K weight layer.  The matmul is a full
      forward through `Hesper.Layers.Linear.LinearLayer.forward` — the Prim owns the
      layer, so it owns the prepared-dispatch cache. -/
  | matmulQ4K
      (layer : Hesper.Layers.Linear.LinearLayer BufT CacheT)
  /-- Q4_K matmul with a lane-local pointwise epilogue.

      `inputs[0]`  = matmul input (f32, pre-Q8_1 quantize)  [shape inDim]
      `inputs[1..]` = epilogue side inputs, each length = epiBufferSizes[k]

      In `epiBody`:
        `input 0`         = the matmul dot product (f32) at outIdx
        `input (k+1)`     = `inputs[k+1][outIdx + epiReadOffsets[k]]`

      The matmul's dispatch shape is unchanged `(outDim, 1, 1) × 32`;
      the epilogue runs on lane 0 after the subgroup reduction — so
      it's purely lane-local.  Cross-lane / cross-WG reductions in
      the epilogue are NOT supported (the whole kernel is one warp
      per output row).

      `epiReadOffsets[k]` lets the caller slice a larger buffer by a
      compile-time offset — used e.g. for PLE where the per-layer
      input table is `plTotalSize = outDim * numLayers` and we read
      at `outIdx + plOffset`.  Default 0 = read at outIdx. -/
  | matmulQ4KWithEpilogue
      (layer          : Hesper.Layers.Linear.LinearLayer BufT CacheT)
      (epiBufferSizes : Array Nat)
      (epiReadOffsets : Array Nat)
      (epiBody        : ScalarExp)
  /-- Reduction along the last axis.  For the MVP we handle only
      `inShape = [D]`; the output is always `#[1]` (one scalar), which
      composes with pointwise ops via the existing broadcast path.
      `D` must fit in a single workgroup's lane count (D ≤ 1024) —
      multi-WG split-reductions are out of scope for this round.

      The dispatch shape for `reduceLastAxis` is fixed:
      `numWorkgroups = 1`, `workgroupSize = min D 256`.  Each lane
      processes `D / wgSize` elements, then the WG does a tree
      reduction in shared memory. -/
  | reduceLastAxis
      (op : ReduceOp) (inShape : Shape)
  /-- Reduce-then-pointwise fused into one kernel.

      `inputs[0]` is the reduction input (shape `reduceInShape`); the
      remaining `inputs[1..]` are the epilogue's full-shape inputs
      whose shapes are listed in `epilogueInShapes`.  The output has
      the SAME shape as the reduction input — the body computes one
      output element per input lane.

      In `body`:
        * `input 0`        = the scalar reduction result
        * `input (1..k)`   = `inputs[1..k][lane_id]`

      Dispatch: 1 WG of `min D 256` lanes.  Phase 1 does the strided
      accumulate + tree reduce just like `Prim.reduceLastAxis`; phase 2
      reads the scalar from shared memory and every lane evaluates the
      epilogue body, writing its slot of the output.  This collapses
      "reduce + use of result via broadcast in a pointwise" into a
      single kernel — the canonical RMSNorm pattern. -/
  | reduceLastAxisWithEpilogue
      (op : ReduceOp)
      (reduceInShape : Shape)
      (epilogueInShapes : Array Shape)
      (epilogueBody : ScalarExp)
  /-- **Level 3 (block-cooperative)**: reduce-then-scatter fused into
      one kernel.

      Like `reduceLastAxisWithEpilogue`, this dispatches a single
      workgroup that:
        Phase 1: tree-reduce `inputs[0]` (shape `reduceInShape`) using
                 shared memory.
        Phase 2: every lane re-reads the scalar reduction result
                 (broadcast through smem), evaluates `valueExpr` and
                 `addrExpr`, then writes `dst[addr] = value`.

      Compared to `reduceLastAxisWithEpilogue`:
        - Output goes to an **external `dst` buffer** of `dstShape`,
          not a fresh allocation of `reduceInShape`.
        - Each lane writes at a computed `addrExpr` (dynamic), not at
          its own lane id.

      In `valueExpr` and `addrExpr`:
        * `.input 0`        = the scalar reduction result (broadcast)
        * `.input (1..k)`   = `inputs[1..k][lane_id]`  (epilogueInShapes[k-1])
        * `.laneIdx`        = the lane within the workgroup

      `inputs` layout in the Op:
        `inputs[0]`    = reduction input (`reduceInShape`)
        `inputs[1..k]` = epilogue side inputs (`epilogueInShapes[0..k-1]`)
        `inputs[k+1]`  = `dst` (the destination buffer, shape `dstShape`)

      The output TensorRef is `dst` itself (in-place scatter into an
      existing buffer), reusing its id like `scatterInto` does.

      Used for: RMSNorm + dynamic-write fusion, where today RMSNorm is
      computed and *then* scattered to a per-position cache slot. -/
  | reduceScatterEpilogue
      (op               : ReduceOp)
      (reduceInShape    : Shape)
      (epilogueInShapes : Array Shape)
      (dstShape         : Shape)
      (valueExpr        : ScalarExp)
      (addrExpr         : ScalarExp)
  /-- Unified Map + Scatter primitive.

      `inputs` layout: there is **one** inputs array, shared by both
      `valueExpr` and `addrExpr`.  Each `.input k` in either expression
      refers to `inputs[k]`.

      Shape semantics: `inShapes[k]` describes `inputs[k]`.
        * equal to `outShape` ⇒ lane-local: `slots[k] = inputs[k][laneIdx]`
        * `#[1]`            ⇒ broadcast: `slots[k] = inputs[k][0]`

      Per-lane evaluation (lane `i` in `[0, outShape.numel)`):
        slots[k] = inputs[k][i] if lane-local, else inputs[k][0]
        value    = valueExpr(slots, laneIdx := i)
        addr     = addrExpr(slots, laneIdx := i) |> toU32
        dst[addr] = value                        -- dst has shape `dstShape`

      Special cases:
        * Map (old pointwise):         `dstShape = outShape`, `addrExpr = .laneIdx`
        * writeSlice (old):             `dstShape` may be larger, `addrExpr = .laneIdx + offset`
        * Dynamic scatter (KV cache):   `addrExpr` uses broadcast inputs (e.g. pos)

      Dispatch grid: `(outShape.numel + 255) / 256 × 256`. -/
  | scatter
      (outShape  : Shape)
      (dstShape  : Shape)
      (inShapes  : Array Shape)
      (valueExpr : ScalarExp)
      (addrExpr  : ScalarExp)
  /-- Multi-output scatter: same dispatch grid (`outShape`) and same
      `inputs` array as `scatter`, but writes to N independent
      destinations.  Each entry of `outputs` is `(dstShape_k,
      valueExpr_k, addrExpr_k)`: per lane, evaluate value and address
      for output k and write `dst_k[addr_k] = value_k`.

      `inputs` layout in the Op: `[data inputs..., dst_0, dst_1, ..., dst_{N-1}]`
      — i.e. the destinations are appended after the data inputs so
      `.input k` in any sub-expr refers to `inputs[k]` (data only),
      identically to the single-output `scatter`.  The destinations are
      *external* TensorRefs supplied by the caller; their ids appear in
      `op.outputs` (one entry per destination).

      Used for K + V cache writes: one dispatch covers both outputs
      with shared work (input loads, broadcast pos, etc.). -/
  | scatterMulti
      (outShape : Shape)
      (inShapes : Array Shape)
      (outputs  : Array (Shape × ScalarExp × ScalarExp))

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

/-- Q4_K matmul with a lane-local pointwise epilogue.

    `epiInputs[k]` is a TensorRef; `epiReadOffsets[k]` is the offset
    at which each epilogue lane reads its k-th side input (defaults to
    zero when the array is shorter than `epiInputs`).

    In `epiBody`, `input 0` refers to the matmul dot product and
    `input (k+1)` to `epiInputs[k][outIdx + epiReadOffsets[k]]`. -/
def matmulQ4KWithEpilogue
    (input : TensorRef)
    (layer : Hesper.Layers.Linear.LinearLayer BufT CacheT)
    (epiInputs : Array TensorRef)
    (epiBody : ScalarExp)
    (epiReadOffsets : Array Nat := #[])
    : CircuitM BufT CacheT TensorRef := do
  let epiBufferSizes : Array Nat := epiInputs.map (fun tr => tr.shape.numel)
  let offsets : Array Nat :=
    Array.ofFn (n := epiInputs.size) (fun i => epiReadOffsets[i.val]?.getD 0)
  let outs ← emitOp
    (Prim.matmulQ4KWithEpilogue layer epiBufferSizes offsets epiBody)
    (#[input] ++ epiInputs)
    #[(#[layer.config.outDim], .f32, .Global)]
  return outs[0]!

/-! ### Pointwise builder sugar

All sugar below lowers to `Prim.pointwise`.  The body uses
`ScalarExp.input i` to refer to the i-th tensor in the `inputs` array
— the builder chooses that layout so the fusion pass can inline
`input` indices by shifting. -/

/-- Generic pointwise (Map): caller supplies the inputs array (each
    either of the output shape or scalar `#[1]`) and a body tree.
    Lowers to `Prim.scatter` with `addrExpr = .laneIdx` (identity
    addressing) and `dstShape = outShape` (fresh output buffer). -/
def pointwise (inputs : Array TensorRef) (body : ScalarExp)
    : CircuitM BufT CacheT TensorRef := do
  let outShape : Shape :=
    (inputs.find? (fun tr => tr.shape != #[1])).map (·.shape)
      |>.getD ((inputs[0]?).map (·.shape) |>.getD #[])
  let inShapes : Array Shape := inputs.map (·.shape)
  let outs ← emitOp
    (Prim.scatter outShape outShape inShapes body .laneIdx)
    inputs
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

/-- Reduction along the last axis: `[D] → [1]`.  Emits a single-WG
    shared-memory reduction.  Downstream consumers can broadcast the
    resulting scalar through the existing `#[1]`-shape pointwise path. -/
def reduceLastAxis (op : ReduceOp) (x : TensorRef)
    : CircuitM BufT CacheT TensorRef := do
  let outs ← emitOp (Prim.reduceLastAxis op x.shape) #[x]
    #[(#[1], .f32, .Global)]
  return outs[0]!

/-- Composite: RMSNorm computed from primitives.

      out[i] = x[i] * rsqrt(sum(x²) / D + eps) * scale[i]

    Emits 5 ops:
      1. reduceLastAxis sumOfSquares x       -- sumSq : [1]
      2. pointwise (sumSq / D + eps)         -- meanSqPlusEps : [1]
      3. pointwise (rsqrt _)                 -- invRms : [1]
      4. pointwise (x * invRms * scale)      -- out : [D]    (the tail)
    fusePointwise collapses ops 2+3 into step 4's producer chain,
    yielding net 2 dispatches (one reduce + one fused pointwise).
    Cross-domain fusion (reduce+epilogue) is Stage 2b and will bring
    this down to 1 dispatch. -/
def rmsNorm (x scale : TensorRef) (eps : Float)
    : CircuitM BufT CacheT TensorRef := do
  let D := x.shape[0]!
  let invD : Float := 1.0 / D.toFloat
  let sumSq ← reduceLastAxis .sumOfSquares x
  let meanSqPlusEps ← map sumSq
    (.add (.mul (.input 0) (.const invD)) (.const eps))
  let invRms ← map meanSqPlusEps (.rsqrt (.input 0))
  -- x[i] * invRms[0] * scale[i] — invRms is #[1] broadcast.
  pointwise #[x, invRms, scale]
    (.mul (.mul (.input 0) (.input 1)) (.input 2))

/-- General Scatter: compute a value per lane and write it to `dst` at
    a computed address.  `dst` is an existing buffer; the resulting
    TensorRef reuses `dst.id` (the write is in-place semantically).

    - `inputs`    : data inputs for both `valueExpr` and `addrExpr`;
                   each either same shape as `outShape` (lane-local)
                   or `#[1]` (broadcast).
    - `outShape`  : dispatch grid shape (= number of writes).
    - `valueExpr` : what to write (uses `.input k` → `inputs[k]`,
                    `.laneIdx` for thread index).
    - `addrExpr`  : where to write (same slot semantics; result is
                    converted to u32 via truncation).

    Examples:
      * Plain copy to offset:  `valueExpr = .input 0`, `addrExpr = .laneIdx + offset`
      * KV cache write:        `addrExpr = .input 1 * kvDim + .laneIdx`
                                (`inputs[1]` is a broadcast `pos` scalar)
      * RoPE into cache:       `valueExpr` = RoPE computation, `addrExpr` = dyn offset -/
def scatterInto (dst : TensorRef) (outShape : Shape) (inputs : Array TensorRef)
    (valueExpr addrExpr : ScalarExp)
    : CircuitM BufT CacheT TensorRef := do
  let inShapes : Array Shape := inputs.map (·.shape)
  -- Reuse dst's id: emit op with outputs := #[dst] directly, bypassing
  -- `emitOp` (which always allocates a fresh TensorRef).
  let newOp : Op BufT CacheT :=
    { prim := Prim.scatter outShape dst.shape inShapes valueExpr addrExpr,
      inputs := inputs,
      outputs := #[dst] }
  modify fun s => { s with ops := s.ops.push newOp }
  return dst

/-- Simple write-slice: copy `src` into `dst` starting at element
    offset `dstOffset`.  Sugar over `scatterInto`. -/
def writeSlice (dst : TensorRef) (src : TensorRef) (dstOffset : ScalarExp)
    : CircuitM BufT CacheT TensorRef :=
  scatterInto dst src.shape #[src] (.input 0) (.add .laneIdx dstOffset)

/-- Multi-output scatter: one dispatch grid (`outShape`) writes into
    N pre-existing destination buffers, each with its own value and
    address expression.

    `inputs` are the shared data inputs (referenced by both `valueExpr_k`
    and `addrExpr_k` via `.input k`).  `dsts[k]` is the k-th destination
    TensorRef; the corresponding `(valueExpr_k, addrExpr_k)` lives at
    `bodies[k]`.

    Returns the array of destination TensorRefs (their ids are reused
    so downstream ops can consume them like any external buffer). -/
def scatterMulti (outShape : Shape) (inputs : Array TensorRef)
    (dsts : Array TensorRef)
    (bodies : Array (ScalarExp × ScalarExp))
    : CircuitM BufT CacheT (Array TensorRef) := do
  let inShapes : Array Shape := inputs.map (·.shape)
  let outs : Array (Shape × ScalarExp × ScalarExp) :=
    Array.ofFn (n := dsts.size) fun i =>
      let dst := dsts[i.val]!
      let (v, a) := bodies[i.val]!
      (dst.shape, v, a)
  let opInputs : Array TensorRef := inputs ++ dsts
  let newOp : Op BufT CacheT :=
    { prim := Prim.scatterMulti outShape inShapes outs,
      inputs := opInputs,
      outputs := dsts }
  modify fun s => { s with ops := s.ops.push newOp }
  return dsts

/-- Block-cooperative reduce + dynamic-address scatter, fused into one
    kernel.  See `Prim.reduceScatterEpilogue` for semantics.

    `reduceIn`     : the tensor being reduced (shape will be queried).
    `epilogueIns`  : side inputs visible in the epilogue at lane id.
    `dst`          : pre-allocated destination buffer (any shape).
    `valueExpr`    : per-lane f32 value to write.  `.input 0` = reduced
                     scalar, `.input (1..k)` = epilogue inputs at laneIdx.
    `addrExpr`     : per-lane address into `dst`.

    Returns `dst` (in-place semantics).  -/
def reduceScatterEpilogue
    (op : ReduceOp) (reduceIn : TensorRef)
    (epilogueIns : Array TensorRef) (dst : TensorRef)
    (valueExpr addrExpr : ScalarExp)
    : CircuitM BufT CacheT TensorRef := do
  let epiShapes : Array Shape := epilogueIns.map (·.shape)
  let inputs : Array TensorRef := #[reduceIn] ++ epilogueIns ++ #[dst]
  let newOp : Op BufT CacheT :=
    { prim := Prim.reduceScatterEpilogue op reduceIn.shape epiShapes
                                          dst.shape valueExpr addrExpr,
      inputs := inputs,
      outputs := #[dst] }
  modify fun s => { s with ops := s.ops.push newOp }
  return dst

end CircuitM

end Hesper.Circuit
