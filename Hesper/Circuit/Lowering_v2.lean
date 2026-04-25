import Hesper.Circuit.IRv2
import Hesper.Circuit.Lowering
import Hesper.Layers.Linear
import Hesper.WGSL.Monad
import Hesper.WGSL.Exp

/-!
# Lowering v2 — BlockGraph → ShaderM

Minimum viable backend for `Hesper.Circuit.IRv2`.  Walks a fused
`BlockGraph` and emits equivalent `ShaderM` statements.  Designed to
reproduce the existing hand-tuned `rmsNormKernel` shape from a
`Block { body := Reduce ... }` after `fusePointwiseIntoReduce`.

## Scope responsibilities

| `Scope`     | lowering action |
|-------------|-----------------|
| `.Global`   | declare as input/output buffer, read/write by thread index |
| `.Shared`   | declare `sharedNamed`, use for cross-thread reduction     |
| `.Register` | emit `ShaderM.varNamed`; value lives in the thread         |
| `.Lane`     | (not used in this PoC — reserved for warp-shuffle fusion) |

Block semantics:
- `Pointwise body`: each thread reads its slice, evaluates `body`,
  writes its slice.  No smem, no barrier.
- `Reduce rop inputRegion slot applyBody`: classic 2-pass reduce in
  one kernel — per-thread partial accumulate over strided elements,
  write to smem, tree-reduce, read back, then evaluate `applyBody`
  (with the reduced scalar bound to `input slot`) elementwise over
  the output region.

## Current limits (PoC)

- Single-Block graphs only (one Reduce or one Pointwise).  Multi-block
  orchestration needs dependency scheduling.
- Reduction op: `sum`, `sumOfSquares`.
- Output shape = input shape, 1D (`shape = #[dim]`).
- Workgroup size is a parameter to the lowering; caller picks it at
  dispatch.

These are enough to prove the full `build → fuse → lower → generate
WGSL` chain on RMSNorm, which is what the instruction asks for.
-/

namespace Hesper.Circuit.IRv2

open Hesper.WGSL
open Hesper.WGSL.Monad

/-- Eval a `ScalarExp` whose only free variables are `input i`
    references (resolved via `inputOf i`) and numeric constants.
    Returns a `Exp (.scalar .f32)` suitable for emission inside the
    surrounding ShaderM context.

    Lean's `partial def` needs `Inhabited` on the return, but
    `Exp t` is a GADT and has none.  Use structural recursion; the
    inductive `ScalarExp` already guarantees termination so no
    `partial` is needed. -/
def evalScalar (inputOf : Nat → Exp (.scalar .f32))
    (laneIdxU32 : Exp (.scalar .u32)) :
    ScalarExp → Exp (.scalar .f32)
  | .input i    => inputOf i
  | .const v    => Exp.litF32 v
  | .laneIdx    => Exp.toF32 laneIdxU32
  | .toFloat a  => evalScalar inputOf laneIdxU32 a
  | .add a b    => Exp.add (evalScalar inputOf laneIdxU32 a) (evalScalar inputOf laneIdxU32 b)
  | .sub a b    => Exp.sub (evalScalar inputOf laneIdxU32 a) (evalScalar inputOf laneIdxU32 b)
  | .mul a b    => Exp.mul (evalScalar inputOf laneIdxU32 a) (evalScalar inputOf laneIdxU32 b)
  | .div a b    => Exp.div (evalScalar inputOf laneIdxU32 a) (evalScalar inputOf laneIdxU32 b)
  | .neg a      => Exp.neg (evalScalar inputOf laneIdxU32 a)
  | .rsqrt a    =>
      Exp.div (Exp.litF32 1.0) (Exp.sqrt (evalScalar inputOf laneIdxU32 a))
  | .exp a      => Exp.exp (evalScalar inputOf laneIdxU32 a)
  | .tanh a     => Exp.tanh (evalScalar inputOf laneIdxU32 a)
  | .gelu a     =>
      let x := evalScalar inputOf laneIdxU32 a
      let x3 := Exp.mul (Exp.mul x x) x
      let inner := Exp.mul (Exp.litF32 0.7978845608028654)
                           (Exp.add x (Exp.mul (Exp.litF32 0.044715) x3))
      Exp.mul (Exp.mul (Exp.litF32 0.5) x) (Exp.add (Exp.litF32 1.0) (Exp.tanh inner))
  | .silu a     =>
      let x := evalScalar inputOf laneIdxU32 a
      Exp.div x (Exp.add (Exp.litF32 1.0) (Exp.exp (Exp.neg x)))
  | .idiv a b   =>
      let ea := evalScalar inputOf laneIdxU32 a
      let eb := evalScalar inputOf laneIdxU32 b
      Exp.floor (Exp.div ea eb)
  | .mod a b    =>
      let ea := evalScalar inputOf laneIdxU32 a
      let eb := evalScalar inputOf laneIdxU32 b
      Exp.sub ea (Exp.mul (Exp.floor (Exp.div ea eb)) eb)
  | _           => Exp.litF32 0.0  -- catch-all for ops not yet needed

/-- Total element count of a `Shape`.  (Shape is `Array Nat`.) -/
def shapeSize (s : Shape) : Nat :=
  s.foldl (· * ·) 1

/-- Emit the hand-written RMSNorm-style kernel from a single
    `Block { body := Reduce ... }`.  Assumes:
    - `reads[0]` = the reduce input (primary region),
    - `reads[1..]` = additional pointwise inputs (scale, residual…),
    - `writes[0]` = output region, shape matches reads[0],
    - all regions are `.Global` f32.

    The generated structure mirrors `rmsNormKernel`:
      partial_sum loop → smem barrier → tree reduce → derive reduced
      scalar → emit `applyBody` per-element.  No intermediate tensor
      materialised in global memory. -/
def lowerReduceBlock (b : Block)
    (declShapes : Array Shape) (workgroupSize : Nat := 256) :
    ShaderM Unit := do
  match b.body with
  | .Pointwise _ =>
    -- Standalone Pointwise lowering is not needed for the current PoC;
    -- fusion folds these into the upstream Reduce/Scatter.  Leave a
    -- no-op so exhaustive matching is satisfied.
    pure ()
  | .Reduce rop reduceIn reduceSlot applyBody =>
    let _ := reduceIn   -- shape already captured in `declShapes`
    -- Block structural assumptions
    let dim := match declShapes[0]? with
      | some s => shapeSize s
      | none   => 0
    let extraInputs := b.reads.size - 1    -- inputs beyond the reduce input
    -- 1. Declare buffers — names match what a typical Gemma 4 call
    --    binds: ("input", ...), ("scale", ...), ("output", ...).
    let _ ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) dim)
    for j in [0:extraInputs] do
      let nm := if j == 0 then "scale" else s!"extra{j}"
      let _ ← ShaderM.declareInputBuffer nm (.array (.scalar .f32) dim)
    let _ ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) dim)

    -- 2. Shared memory for cross-thread reduction.
    ShaderM.sharedNamed "shared_sum" (.array (.scalar .f32) workgroupSize)

    let gid ← ShaderM.globalId
    let lid ← ShaderM.localId
    let idx := Exp.vec3X gid
    let localIdx := Exp.vec3X lid

    -- 3. Per-thread partial accumulate over strided elements.
    ShaderM.varNamed "partial_sum" (.scalar .f32) (Exp.litF32 0.0)
    let partialSum : Exp (.scalar .f32) := Exp.var "partial_sum"
    ShaderM.loop localIdx (Exp.litU32 dim) (Exp.litU32 workgroupSize) fun i => do
      let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := dim) "input" i
      let contrib :=
        match rop with
        | .sum           => v
        | .sumOfSquares  => Exp.mul v v
      ShaderM.assign "partial_sum" (Exp.add partialSum contrib)

    -- 4. Write partial sum to smem, tree-reduce.
    ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_sum" localIdx partialSum
    ShaderM.barrier
    let mut stride := workgroupSize / 2
    while stride > 0 do
      ShaderM.if_ (Exp.lt localIdx (Exp.litU32 stride)) (do
        let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_sum" localIdx
        let c ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_sum" (Exp.add localIdx (Exp.litU32 stride))
        ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_sum" localIdx (Exp.add a c)
      ) (pure ())
      ShaderM.barrier
      stride := stride / 2

    -- 5. Broadcast reduced scalar to all threads.
    let totalSum ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_sum" (Exp.litU32 0)
    -- For RMSNorm semantics we expose `totalSum` as `input reduceSlot`
    -- inside `applyBody`.  The builder emitted `applyBody` expecting
    -- the reduced scalar at slot 0 (see `rmsNormFusedOneBlock`), with
    -- the per-lane input at slot 1.  We honour `reduceSlot`.

    -- 6. Per-element apply: read per-lane input (+ extras), evaluate
    --    applyBody, write output.
    let inBounds := Exp.lt idx (Exp.litU32 dim)
    let laneVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := dim) "input" idx
    let extrasArr : Array (Exp (.scalar .f32)) ← do
      let mut acc : Array (Exp (.scalar .f32)) := #[]
      for j in [0:extraInputs] do
        let nm := if j == 0 then "scale" else s!"extra{j}"
        let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := dim) nm idx
        acc := acc.push v
      pure acc

    -- Input resolver: slot `reduceSlot` = totalSum, others index into
    -- `laneVal` (slot reduceSlot+1 originally) then extras.  To match
    -- `rmsNormFusedOneBlock`, laneVal is at slot 1 when reduceSlot = 0.
    let inputOf : Nat → Exp (.scalar .f32) := fun slot =>
      if slot == reduceSlot then totalSum
      else if slot == reduceSlot + 1 then laneVal
      else
        let k := slot - (reduceSlot + 2)
        match extrasArr[k]? with
        | some v => v
        | none   => Exp.litF32 0.0

    let result := evalScalar inputOf idx applyBody
    let gated := Exp.select inBounds result (Exp.litF32 0.0)
    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx gated
  | .Scatter idxE applyE =>
    -- Scatter lowering: 1D dispatch over the primary input (reads[0]).
    -- `applyBody` / `indexExpr` are lowered via v1's monadic
    -- `lowerScalarExp`, which supports the full ScalarExp surface —
    -- `.select`, `.lt`, `.pow`, `.cos`, `.sin`, `.indexed`, etc.
    --
    -- Slot / buffer convention for Scatter:
    --   slot 0 = new_k[laneIdx] (pre-read from reads[0] at the lane idx)
    --   slot i (i ≥ 1) = NOT pre-read; reads[i] is only accessible via
    --                    `.indexed i addr` gathers.
    --   decls[0..] = (reads[i]'s buffer binding, length) — enables
    --                gather reads at caller-computed addresses, which
    --                RoPE-K needs for xPair and freq_factors.
    let dim := match declShapes[0]? with
      | some s => shapeSize s
      | none   => 0
    let extraInputs := b.reads.size - 1
    let primaryName := "input"
    let _ ← ShaderM.declareInputBuffer primaryName (.array (.scalar .f32) dim)
    let extraNames : Array String := Id.run do
      let mut a : Array String := #[]
      for j in [0:extraInputs] do
        let nm := if j == 0 then "scale" else s!"extra{j}"
        a := a.push nm
      return a
    let extraLens : Array Nat := Id.run do
      let mut a : Array Nat := #[]
      for j in [0:extraInputs] do
        -- Look up each extra's declared size.  If absent, default to
        -- `dim` (matches the simple V-Scatter case).
        let sz := match declShapes[j+1]? with
          | some s => shapeSize s
          | none   => dim
        a := a.push sz
      return a
    for j in [0:extraInputs] do
      let nm := extraNames[j]!
      let sz := extraLens[j]!
      let _ ← ShaderM.declareInputBuffer nm (.array (.scalar .f32) sz)
    -- destination buffer: its size comes from the writes[0] tensor id
    -- if declared, else fall back to the placeholder used by the
    -- earlier V-Scatter PoC.
    let dstSize := match b.writes[0]? with
      | some r =>
        match declShapes[b.reads.size]? with  -- writes shape comes right after reads in declShapes on our path
        | some s => shapeSize s
        | none   => dim * 1024
      | none => dim * 1024
    -- The above `declShapes[b.reads.size]?` is best-effort; callers
    -- that need an exact declared dst size should ensure the tensor
    -- declaration is visible to `lowerBlockGraph`.
    let _ := dstSize
    let _ ← ShaderM.declareOutputBuffer "dst" (.array (.scalar .f32) (dim * 1024))
    let gid ← ShaderM.globalId
    let idx := Exp.vec3X gid
    let inBounds := Exp.lt idx (Exp.litU32 dim)
    -- Pre-read the primary input at laneIdx → slot[0].
    let laneVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := dim) primaryName idx
    let slot : Array (Exp (.scalar .f32)) := #[laneVal]
    let laneIdxF32 : Exp (.scalar .f32) := Exp.toF32 idx
    -- `decls` for v1's lowerScalarExp.  Index 0 = primary, 1..= extras.
    let decls : Array Hesper.Circuit.InputDecl :=
      #[{ name := primaryName, len := dim }] ++
      (Array.ofFn (fun (j : Fin extraInputs) =>
        ({ name := extraNames[j.val]!, len := extraLens[j.val]! } : Hesper.Circuit.InputDecl)))
    -- Gate writes inside the inBounds check so scatter stays safe
    -- when dispatch grid overshoots the primary length.
    ShaderM.if_ inBounds (do
      let val ← Hesper.Circuit.lowerScalarExp slot laneIdxF32 decls applyE
      let idxF ← Hesper.Circuit.lowerScalarExp slot laneIdxF32 decls idxE
      let dstIdx := Exp.toU32 idxF
      ShaderM.writeBuffer (ty := .scalar .f32) "dst" dstIdx val
    ) (pure ())
  | .ScatterMulti ops =>
    -- Multi-output Scatter: N `(indexExpr, applyBody)` pairs writing to
    -- N separate destination buffers `dst0`, `dst1`, … — all driven
    -- by the SAME 1D dispatch over reads[0] and sharing slot[]/decls[].
    -- Matches the "K (with RoPE) + V (plain copy)" pattern used by
    -- Gemma 4's production `scatterMulti` KV write.
    let dim := match declShapes[0]? with
      | some s => shapeSize s
      | none   => 0
    let extraInputs := b.reads.size - 1
    let primaryName := "input"
    let _ ← ShaderM.declareInputBuffer primaryName (.array (.scalar .f32) dim)
    let extraNames : Array String := Id.run do
      let mut a : Array String := #[]
      for j in [0:extraInputs] do
        let nm := if j == 0 then "scale" else s!"extra{j}"
        a := a.push nm
      return a
    let extraLens : Array Nat := Id.run do
      let mut a : Array Nat := #[]
      for j in [0:extraInputs] do
        let sz := match declShapes[j+1]? with
          | some s => shapeSize s
          | none   => dim
        a := a.push sz
      return a
    for j in [0:extraInputs] do
      let nm := extraNames[j]!
      let sz := extraLens[j]!
      let _ ← ShaderM.declareInputBuffer nm (.array (.scalar .f32) sz)
    -- Declare N output buffers: dst0, dst1, ….
    for k in [0:ops.size] do
      let _ ← ShaderM.declareOutputBuffer s!"dst{k}" (.array (.scalar .f32) (dim * 1024))
    let gid ← ShaderM.globalId
    let idx := Exp.vec3X gid
    let inBounds := Exp.lt idx (Exp.litU32 dim)
    let laneVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := dim) primaryName idx
    let slot : Array (Exp (.scalar .f32)) := #[laneVal]
    let laneIdxF32 : Exp (.scalar .f32) := Exp.toF32 idx
    let decls : Array Hesper.Circuit.InputDecl :=
      #[{ name := primaryName, len := dim }] ++
      (Array.ofFn (fun (j : Fin extraInputs) =>
        ({ name := extraNames[j.val]!, len := extraLens[j.val]! } : Hesper.Circuit.InputDecl)))
    -- All writes go inside ONE inBounds check — one kernel, N stores.
    ShaderM.if_ inBounds (do
      for k in [0:ops.size] do
        let (idxE, applyE) := ops[k]!
        let val ← Hesper.Circuit.lowerScalarExp slot laneIdxF32 decls applyE
        let idxF ← Hesper.Circuit.lowerScalarExp slot laneIdxF32 decls idxE
        let dstIdx := Exp.toU32 idxF
        ShaderM.writeBuffer (ty := .scalar .f32) s!"dst{k}" dstIdx val
    ) (pure ())
  | .MatMul layerId oD iD epi =>
    -- Route through the existing v1 Q4_K + epilogue kernel.  Slot-space
    -- contract matches: slot 0 = dot product (virtual `__total__`),
    -- slot 1+k = b.reads[k] (bias/residual/gate partner/…) read at
    -- `outIdx` with offset 0.
    let _ := layerId
    let cfg : Hesper.Layers.Linear.Config := { inDim := iD, outDim := oD }
    let epiInputNames : Array String := b.reads.mapIdx fun i _ =>
      s!"epi{i}"
    let epiBufferSizes : Array Nat := Array.replicate b.reads.size oD
    let epiReadOffsets : Array Nat := Array.replicate b.reads.size 0
    Hesper.Circuit.lowerMatmulQ4KWithEpilogueKernel
      cfg epiInputNames epiBufferSizes epiReadOffsets epi
  | .Quantize =>
    -- Standalone Q8_1 quantize.  This Lowering path isn't exercised by
    -- the parity-test harness yet (the dispatcher routes Quantize
    -- blocks to the existing `quantizeQ8_1Kernel` directly); emit a
    -- no-op so exhaustive matching is satisfied.
    pure ()
  | .ReduceQuantize _ _ _ _ =>
    -- Fused Reduce + Q8_1.  Lowering routes through the existing
    -- `fusedRMSNormQ8_1Kernel` at the dispatcher layer, so this path
    -- is currently a no-op for the static analyzer's benefit.
    pure ()
  | .GemmaAttentionMonolith _ _
  | .FlashAttention _ _
  | .GemmaAttnOutProj _
  | .PostAttnNormAdd _
  | .GemmaFFNMonolith _
  | .PostFFNNormAdd _ =>
    -- Logical Monolith nodes — never lowered through this path.  The
    -- dispatcher in `Dispatch_v2.runMonolith` matches them directly
    -- and expands into the production hand-fused sequences.
    pure ()

/-- Entry point: lower a (single-block) fused `BlockGraph` to ShaderM.

    The shape of the read-0 region is looked up by matching its
    `tensorId` against `g.tensors` for the Global-scope input.  If
    that lookup fails the graph is not lowerable as RMSNorm; emit
    nothing and let the caller assert. -/
def lowerBlockGraph (g : BlockGraph) (workgroupSize : Nat := 256) :
    ShaderM Unit := do
  if h : g.blocks.size = 1 then
    let b := g.blocks[0]
    -- Collect shapes for reads in order.  When a read's tensorId is
    -- not declared in g.tensors, it must be the external primary
    -- input; pick the shape from the reduce input's declared shape
    -- if available, else default to 0.
    let declShapes : Array Shape := b.reads.map fun r =>
      match g.tensors.find? (·.id == r.tensorId) with
      | some t => t.shape
      | none   => #[]   -- external tensor — caller binds a real shape
    lowerReduceBlock b declShapes workgroupSize
  else
    pure ()  -- multi-block unsupported in PoC

end Hesper.Circuit.IRv2
