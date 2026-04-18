import Hesper.CircuitV2.IR
import Hesper.Circuit.Lowering

/-!
# Circuit DSL v2 — lowering to ShaderM (Phase C1)

The v2 AST is **lowered** by translating every primitive to a (possibly
shared) `Hesper.WGSL.Monad.ShaderM Unit` whose dispatch shape is
decided by the op.

Phase C1-a covers `Prim.pointwise` only — the smallest meaningful op
to validate the bridge end to end.

## Why we reuse v1's `lowerScalarExp`

The lane-local algebra is identical between v1 and v2; v2 extends v1 at
the *tensor* level (scopes, state tokens, blocks), not at the *lane*
level.  The whole `ScalarExp` tree — `add`, `mul`, `rsqrt`, `warpSum`,
the fastdiv encoding — is already complete in v1.  Copying the whole
lowering function into v2 would only create drift; instead v2 imports
it and reuses it verbatim.

## Output type

`LoweredPrim` pairs a kernel name with the `ShaderM Unit` that builds
it and the dispatch shape.  The runtime then calls
`GPUBackend.executeWithConfig` / `executeWithConfigCached` once per
`LoweredPrim` — Phase C1 does not implement fusion or batching across
prims; that is Phase C2/D work.
-/

namespace Hesper.CircuitV2

open Hesper.WGSL.Monad
open Hesper.WGSL
open Hesper.Circuit (ScalarExp)

/-- Compact id→shape lookup.  Kept as an assoc list (the Hesper code
    base doesn't depend on any HashMap implementation) — lookup is
    linear, which is fine for the handful of tensors a single kernel
    touches. -/
abbrev ShapeEnv := Array (Nat × Shape)

def ShapeEnv.find? (env : ShapeEnv) (id : Nat) : Option Shape :=
  env.findSome? fun (k, s) => if k = id then some s else none

def ShapeEnv.maxId (env : ShapeEnv) : Nat :=
  env.foldl (init := 0) (fun acc (k, _) => max acc k)

/-- Lowering output: a single kernel description that the runtime can
    dispatch.  `dispatch` is `(numWorkgroups, workgroupSize)` chosen by
    the pass.  `binds` lists every named buffer the kernel will read /
    write — bound at execute time from a tensor-id → buffer map. -/
structure LoweredPrim where
  name        : String
  shader      : ShaderM Unit
  numWg       : Nat × Nat × Nat
  wgSize      : Nat
  /-- Tensor ids the kernel expects buffers for, in declaration order.
      The runtime resolves each id to a `GPUBackend.Buf` just before
      `executeWithConfig`. -/
  binds       : Array (String × Nat)
  deriving Inhabited

/-- Lower a single `Prim.pointwise` to a `LoweredPrim`.

    Algorithm:
      1. Declare one input buffer `in_i` per input tensor.
      2. Declare an output buffer `out`.
      3. Load every `input i` once into a per-lane slot and fold the
         `body : ScalarExp` over them via v1's `lowerScalarExp`.
      4. Write the result to `out[laneIdx]`.
      5. Dispatch shape = `numel(outShape)` threads with
         `workgroupSize = min 256 numel`.

    Scope-awareness: `.Reg` / `.SRAM` outputs are handled the same at
    the lowering level — both lower to a shader that writes the lane's
    result.  The difference is *upstream* (producing the `load_to_sram`
    primitive, for instance).  When an op's outScope is `.SRAM` the
    runtime resolves the output bind to a shared-memory scratch buffer
    rather than a VRAM buffer; that promotion is Phase C2 / D work.

    Returns `none` if the prim is not a pointwise. -/
def lowerPointwise
    (inShapes : Array Shape) (inIds : Array Nat)
    (body : ScalarExp) (outShape : Shape) (outId : Nat)
    : LoweredPrim :=
  let numElems := Shape.numel outShape
  let wgSize := min 256 numElems
  let numWg  := ((numElems + wgSize - 1) / wgSize, 1, 1)
  let inputNames : Array String :=
    (Array.range inIds.size).map fun i => s!"in_{i}"
  let outName := "out"
  let shader : ShaderM Unit := do
    -- Declare input / output buffers.  Every shape is lowered to a flat
    -- element count; v1's lowerScalarExp does the per-lane indexing.
    let mut decls : Array Hesper.Circuit.InputDecl := #[]
    for i in [0 : inIds.size] do
      let len :=
        if h : i < inShapes.size then Shape.numel inShapes[i]
        else numElems
      let name := inputNames[i]!
      let _ ← ShaderM.declareInputBuffer name (.array (.scalar .f32) len)
      decls := decls.push { name, len }
    let _ ← ShaderM.declareOutputBuffer outName (.array (.scalar .f32) numElems)
    -- Per-lane load of inputs.  Broadcast (#[1]) inputs degenerate
    -- naturally via index-clamp on the gather path; for this MVP we
    -- assume same-shape inputs.
    let gid ← ShaderM.globalId
    let idx := Exp.vec3X gid
    ShaderM.if_ (Exp.lt idx (Exp.litU32 numElems)) (do
      let mut slots : Array (Exp (.scalar .f32)) := #[]
      for i in [0 : inIds.size] do
        let decl := decls[i]!
        let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := decl.len) decl.name idx
        slots := slots.push v
      let result ← Hesper.Circuit.lowerScalarExp slots (Exp.toF32 idx) decls body
      ShaderM.writeBuffer (ty := .scalar .f32) outName idx result
    ) (pure ())
  let binds : Array (String × Nat) :=
    (Array.range inIds.size).foldl (init := #[]) (fun acc i =>
      acc.push (inputNames[i]!, inIds[i]!))
    |>.push (outName, outId)
  { name := s!"v2_pointwise_n{numElems}"
    shader, numWg, wgSize, binds }

/-! ## Phase C1-b — `Prim.reduce` → ShaderM

v2 supports three reductions (sum, max, sumOfSquares); v1 only has
the first and third.  We emit the reduction directly in v2 rather than
call v1's helper, because:

  * the combine-op differs (add vs max) but the kernel skeleton is
    identical (strided local accumulation → tree reduction → lane-0
    writes `out[0]`);
  * v2 will eventually want output-scope awareness (`.SRAM` output =
    leave the result in shared memory for an epilogue), which v1's
    helper doesn't model.

For Phase C1-b the output scope is treated as VRAM — the kernel writes
one scalar to `out[0]`.  A subsequent pass will promote VRAM→SRAM when
a downstream op in the same block can consume the shared scalar.
-/

/-- Identity element for a reduction op, in the f32 lane-local algebra. -/
def ReduceOp.identity : ReduceOp → Exp (.scalar .f32)
  | .sum          => Exp.litF32 0.0
  | .max          => Exp.litF32 (-1.0e30)
  | .sumOfSquares => Exp.litF32 0.0

/-- Per-lane-per-iteration contribution before the running combine. -/
def ReduceOp.contrib (op : ReduceOp) (v : Exp (.scalar .f32))
    : Exp (.scalar .f32) :=
  match op with
  | .sum          => v
  | .max          => v
  | .sumOfSquares => Exp.mul v v

/-- Associative combine for the running accumulator. -/
def ReduceOp.combine (op : ReduceOp)
    (a b : Exp (.scalar .f32)) : Exp (.scalar .f32) :=
  match op with
  | .sum          => Exp.add a b
  | .max          => Exp.max a b
  | .sumOfSquares => Exp.add a b

/-- Lower `Prim.reduce` to a single-WG ShaderM.  Same structure as
    v1's `lowerReduceLastAxis` but parameterised over the combine op
    so `.max` works.

    Dispatch: `numWg = (1, 1, 1)`, `wgSize = min 256 D`. -/
def lowerReduce
    (inShape : Shape) (inId : Nat) (op : ReduceOp) (outId : Nat)
    : LoweredPrim :=
  let D := Shape.numel inShape
  let wgSize := min 256 (max D 1)
  let inName  := "in0"
  let outName := "out"
  let shader : ShaderM Unit := do
    ShaderM.sharedNamed "scratch" (.array (.scalar .f32) wgSize)
    let _ ← ShaderM.declareInputBuffer  inName  (.array (.scalar .f32) D)
    let _ ← ShaderM.declareOutputBuffer outName (.array (.scalar .f32) 1)
    let lid ← ShaderM.localId
    let localIdx := Exp.vec3X lid
    ShaderM.varNamed "accum" (.scalar .f32) (ReduceOp.identity op)
    let accumE : Exp (.scalar .f32) := Exp.var "accum"
    -- Strided accumulation.
    ShaderM.loop localIdx (Exp.litU32 D) (Exp.litU32 wgSize) fun loopIdx => do
      let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := D) inName loopIdx
      ShaderM.assign "accum" (ReduceOp.combine op accumE (ReduceOp.contrib op v))
    ShaderM.writeWorkgroup (ty := .scalar .f32) "scratch" localIdx accumE
    ShaderM.barrier
    -- Classic power-of-two tree reduction.
    let mut stride := wgSize / 2
    while stride > 0 do
      ShaderM.if_ (Exp.lt localIdx (Exp.litU32 stride)) (do
        let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := wgSize) "scratch" localIdx
        let b ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := wgSize) "scratch"
                  (Exp.add localIdx (Exp.litU32 stride))
        ShaderM.writeWorkgroup (ty := .scalar .f32) "scratch" localIdx
          (ReduceOp.combine op a b)
      ) (pure ())
      ShaderM.barrier
      stride := stride / 2
    -- Lane 0 writes the total.
    ShaderM.if_ (Exp.eq localIdx (Exp.litU32 0)) (do
      let total ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := wgSize) "scratch" (Exp.litU32 0)
      ShaderM.writeBuffer (ty := .scalar .f32) outName (Exp.litU32 0) total
    ) (pure ())
  let opTag := match op with
    | .sum => "sum" | .max => "max" | .sumOfSquares => "sumSq"
  { name   := s!"v2_reduce_{opTag}_D{D}"
    shader
    numWg  := (1, 1, 1)
    wgSize
    binds  := #[(inName, inId), (outName, outId)] }

/-- Top-level lowerer for a finished `BuilderState`.  Handles `pointwise`
    (Phase C1-a) and `reduce` (Phase C1-b); unsupported prims are
    no-ops — incremental bring-up. -/
def lowerAll (st : BuilderState) (idToShape : ShapeEnv)
    : Array LoweredPrim := Id.run do
  let mut out : Array LoweredPrim := #[]
  let mut currentOutId : Nat := idToShape.maxId
  for op in st.ops do
    match op with
    | .pointwise inputs body outShape _outDt _outScope =>
      let inShapes : Array Shape :=
        inputs.map (fun id => (idToShape.find? id).getD [])
      -- Each emitted Prim produces a fresh tensor id in the builder;
      -- we increment our own counter to stay in sync, since we don't
      -- yet thread that back through BuilderState.  A proper
      -- implementation would have Prim carry its output id explicitly.
      currentOutId := currentOutId + 1
      out := out.push (lowerPointwise inShapes inputs body outShape currentOutId)
    | .reduce inputId op _outDt _outScope =>
      let inShape := (idToShape.find? inputId).getD []
      currentOutId := currentOutId + 1
      out := out.push (lowerReduce inShape inputId op currentOutId)
    | _ =>
      -- TODO Phase C1-c: lower scatter / load / store / block
      pure ()
  return out

end Hesper.CircuitV2
