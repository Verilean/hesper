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

/-- Top-level lowerer for a finished `BuilderState`.  For C1-a we only
    handle `Prim.pointwise`; unsupported prims are skipped with a
    warning comment so incremental bring-up is possible. -/
def lowerAll (st : BuilderState) (idToShape : ShapeEnv)
    : Array LoweredPrim := Id.run do
  let mut out : Array LoweredPrim := #[]
  for op in st.ops do
    match op with
    | .pointwise inputs body outShape _outDt _outScope =>
      let inShapes : Array Shape :=
        inputs.map (fun id => (idToShape.find? id).getD [])
      -- Output id is the next fresh id the builder allocated — in v2
      -- it's the emitter's responsibility to pair an op with the id it
      -- produces.  For C1-a we take it from the end of the tensor map.
      -- (A real implementation would pass the map through the builder.)
      let outId := idToShape.maxId
      out := out.push (lowerPointwise inShapes inputs body outShape outId)
    | _ =>
      -- TODO Phase C1-b/c: lower reduce / scatter / load / store / block
      pure ()
  return out

end Hesper.CircuitV2
