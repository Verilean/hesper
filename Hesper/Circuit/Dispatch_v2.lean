import Hesper.Circuit.IRv2
import Hesper.Circuit.Lowering
import Hesper.Circuit.Lowering_v2
import Hesper.Layers.Linear
import Hesper.Layers.RMSNorm
import Hesper.Layers.Attention
import Hesper.Models.Gemma4.Kernels
import Hesper.WGSL.FlashAttention
import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.CUDA.FFI

/-!
# IRv2 BlockGraph dispatcher (Phase B2)

Walks a `BlockGraph` and executes each block on a real GPU backend.
The dispatcher is deliberately minimal: it recognises only the block
patterns the current PoC needs and errors on anything else.

## Recognised patterns

| Graph fragment                                                                     | Dispatch                                                        |
|------------------------------------------------------------------------------------|-----------------------------------------------------------------|
| `[Reduce(sumSq)]; [MatMul wGate]; [MatMul wUp]; [Pointwise GELU(in0)*in1]; [MatMul wDown]` | `forwardFusedNormGateUp` + `wDown.forward` (2 dispatches) |
| `[Reduce(sumSq)]; [MatMul wQ]; [MatMul wK]; [MatMul wV]` sharing Reduce output     | `forwardFusedNormQKV` (3 dispatches: fused-rmsnorm-q8_1, wQ matmul, fused wK+wV matmul) |
| `[Reduce(sumOfSquares, applyBody with rsqrt+eps)]; [MatMul layerId]`               | `forwardFusedNormWQ` (2 dispatches, fused-rmsnorm-q8_1 + dp4a matmul) |
| `[Reduce(sumSq, applyBody = (x*invRms)*scale + residual)]` alone                   | `forwardNormThenAdd` (1 dispatch)                                |
| `[MatMul layerId]` alone                                                           | `runMatmulQ4KWithEpilogueOp` (plain quantize + dp4a matmul+epi) |
| `[Scatter]` alone                                                                  | `lowerBlockGraph` → `executeWithConfig` (generic scatter WGSL/PTX) |

The Reduce→MatMul pattern is identified structurally (ReduceOp =
sumOfSquares) plus by the caller-supplied `normHandles` lookup that
maps the Reduce block's output tensor id back to its `RMSNorm` layer
handle.  The dispatcher does NOT try to extract `eps` or `N` from the
`applyBody` AST — that would couple the dispatcher to an exact
ScalarExp template.  Instead the RMSNorm handle carries those as
configuration.

## Contracts

- `externalBufs` maps every Global-scope tensor id that appears in
  `g` (via declareExternal) to a real GPU buffer.
- `matmulLayers` maps every `BlockBody.MatMul layerId ...` to a
  `LinearLayer` with matching `config.inDim / outDim`.
- `normHandles` maps a Reduce block's `writes[0].tensorId` (the
  intermediate "reduction scalar" tensor) to the `RMSNorm` that will
  be used when the Reduce feeds a MatMul.
-/

namespace Hesper.Circuit.IRv2

open Hesper.Circuit
open Hesper.Layers
open Hesper.Layers.Linear (LinearLayer)
open Hesper.Layers.RMSNorm (RMSNorm)

/-- Look up a tensor id in a flat `(id, buf)` association. -/
private def findBuf [GPUBackend β]
    (externalBufs : List (Nat × GPUBackend.Buf β)) (tid : Nat)
    : IO (GPUBackend.Buf β) :=
  match externalBufs.find? (·.fst == tid) with
  | some (_, b) => pure b
  | none => throw (IO.userError s!"runBlockGraph: tensor id {tid} not in externalBufs")

/-- Look up a MatMul layerId. -/
private def findLayer [GPUBackend β]
    (matmulLayers : List (UInt64 × LinearLayer (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)))
    (lid : UInt64)
    : IO (LinearLayer (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)) :=
  match matmulLayers.find? (·.fst == lid) with
  | some (_, l) => pure l
  | none => throw (IO.userError s!"runBlockGraph: matmul layerId {lid} not in matmulLayers")

/-- Look up a RMSNorm handle by the Reduce block's output tensor id. -/
private def findNorm [GPUBackend β]
    (normHandles : List (Nat × RMSNorm (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)))
    (tid : Nat)
    : IO (Option (RMSNorm (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))) := do
  match normHandles.find? (·.fst == tid) with
  | some (_, n) => pure (some n)
  | none        => pure none

/-- Dispatch a plain `BlockBody.MatMul` via `runMatmulQ4KWithEpilogueOp`.

    The block's input (the f32 vector the matmul multiplies) is NOT a
    declared read — by our AST convention the MatMul kernel reads its
    f32 input internally (via Q8_1 quantize of a caller-supplied
    buffer).  Callers pass the input buffer id via `matmulInputBufs`.

    The block's `reads` become epilogue side-inputs (bias, residual,
    gate partner etc.), in order. -/
def runMatmulBlock [GPUBackend β]
    (ctx : β) (b : Block)
    (externalBufs : List (Nat × GPUBackend.Buf β))
    (matmulLayers : List (UInt64 × LinearLayer (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)))
    (matmulInputBufs : List (UInt64 × GPUBackend.Buf β))
    : IO Unit := do
  match b.body with
  | .MatMul layerId _oD _iD epi =>
    let layer ← findLayer matmulLayers layerId
    let inBuf ← match matmulInputBufs.find? (·.fst == layerId) with
      | some (_, b) => pure b
      | none => throw (IO.userError s!"runBlockGraph: MatMul {layerId} has no input buffer registered")
    if b.writes.size != 1 then
      throw (IO.userError s!"runBlockGraph: MatMul block must have exactly 1 write")
    let outBuf ← findBuf externalBufs b.writes[0]!.tensorId
    -- Epilogue side inputs (b.reads in order → epi0, epi1, …).
    let mut epiBufs : Array (GPUBackend.Buf β) := #[]
    for r in b.reads do
      let buf ← findBuf externalBufs r.tensorId
      epiBufs := epiBufs.push buf
    let epiSizes := Array.replicate b.reads.size layer.config.outDim
    let epiOffs  := Array.replicate b.reads.size 0
    let cacheRef : IO.Ref (Option (GPUBackend.CachedDispatch β)) ← IO.mkRef none
    Hesper.Circuit.runMatmulQ4KWithEpilogueOp ctx layer inBuf
      epiBufs epiSizes epiOffs epi outBuf
      (hash ("irv2-matmul", layerId, layer.config.inDim, layer.config.outDim))
      cacheRef
  | _ => throw (IO.userError "runMatmulBlock: not a MatMul block")

/-- Execute an entire `BlockGraph`.

    The dispatcher walks blocks in order and, for each, chooses one of
    the recognised patterns above.  It supports single-block MatMul
    directly, and the `[Reduce(RMSNorm); MatMul]` adjacent pair by
    calling `forwardFusedNormWQ` when the Reduce block's output tensor
    id is registered in `normHandles`.

    Not a general-purpose executor: other block patterns (standalone
    Reduce, Scatter, Pointwise) raise `IO.userError`. -/
partial def runBlockGraph [GPUBackend β]
    (ctx : β) (g : BlockGraph)
    (externalBufs : List (Nat × GPUBackend.Buf β))
    (matmulLayers : List (UInt64 × LinearLayer (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)))
    (matmulInputBufs : List (UInt64 × GPUBackend.Buf β))
    (normHandles : List (Nat × RMSNorm (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)))
    : IO Unit := do
  -- Helper: is this MatMul's reads[0] equal to `midId`, and is the
  -- epilogue the identity (slot 0)?  Returns the layerId/outBuf tid if
  -- so, else none.
  let matmulConsuming (b : Block) (midId : Nat) : Option (UInt64 × Nat) :=
    match b.body with
    | .MatMul lid _ _ epi =>
      if epi == (.input 0 : ScalarExp)
         ∧ b.reads.size == 1 ∧ b.reads[0]!.tensorId == midId
         ∧ b.writes.size == 1 then
        some (lid, b.writes[0]!.tensorId)
      else none
    | _ => none
  -- Helper: does this Pointwise block compute `GELU(.input 0) * .input 1`
  -- over exactly [gateTid, upTid] and write to geluTid?
  let isGeluMul (b : Block) (gateTid upTid : Nat) : Option Nat :=
    match b.body with
    | .Pointwise body =>
      let expected : ScalarExp := .mul (.gelu (.input 0)) (.input 1)
      if body == expected
         ∧ b.reads.size == 2 ∧ b.writes.size == 1
         ∧ b.reads[0]!.tensorId == gateTid
         ∧ b.reads[1]!.tensorId == upTid then
        some b.writes[0]!.tensorId
      else none
    | _ => none
  let mut i : Nat := 0
  let n := g.blocks.size
  while i < n do
    let cur := g.blocks[i]!
    -- ===========================================================
    -- Pattern D: [Reduce(sumSq); MatMul wGate; MatMul wUp;
    --            Pointwise GELU(.input 0)*.input 1; MatMul wDown]
    -- Route to forwardFusedNormGateUp (producing geluBuf) + wDown
    -- LinearLayer.forward on geluBuf → outBuf.
    -- ===========================================================
    let try5BlockFFN : IO Bool := do
      if i + 4 ≥ n then return false
      let b0 := g.blocks[i]!
      let b1 := g.blocks[i+1]!
      let b2 := g.blocks[i+2]!
      let b3 := g.blocks[i+3]!
      let b4 := g.blocks[i+4]!
      match b0.body with
      | .Reduce .sumOfSquares _ _ _ =>
        if b0.writes.size != 1 then return false
        let midId := b0.writes[0]!.tensorId
        match (matmulConsuming b1 midId), (matmulConsuming b2 midId) with
        | some (gateKey, gateTid), some (upKey, upTid) =>
          match isGeluMul b3 gateTid upTid with
          | none => return false
          | some geluTid =>
            -- b4 must be MatMul wDown consuming geluTid.
            match matmulConsuming b4 geluTid with
            | none => return false
            | some (downKey, outTid) =>
              -- All preconditions satisfied — dispatch the 2 kernels.
              match ← findNorm normHandles midId with
              | none => return false
              | some norm =>
                let wGate ← findLayer matmulLayers gateKey
                let wUp   ← findLayer matmulLayers upKey
                let wDown ← findLayer matmulLayers downKey
                let inTid := match b0.body with
                  | .Reduce _ rin _ _ => rin.tensorId
                  | _                 => 0
                let inputBuf ← findBuf externalBufs inTid
                let geluBuf ← findBuf externalBufs geluTid
                let outBuf ← findBuf externalBufs outTid
                let geluRef : IO.Ref (Option (GPUBackend.CachedDispatch β)) ← IO.mkRef none
                Hesper.Layers.Linear.forwardFusedNormGateUp ctx norm wGate wUp
                  inputBuf geluBuf geluRef
                Hesper.Layers.Linear.LinearLayer.forward ctx wDown geluBuf outBuf
                return true
        | _, _ => return false
      | _ => return false
    if ← try5BlockFFN then
      i := i + 5
      continue
    -- ===========================================================
    -- Pattern A: [Reduce(sumSq); MatMul wQ; MatMul wK; MatMul wV]
    -- All 3 MatMuls consume the Reduce's output and have identity
    -- epilogues.  Route to forwardFusedNormQKV.
    -- ===========================================================
    let try4Block : IO Bool := do
      if i + 3 ≥ n then return false
      let b0 := g.blocks[i]!
      let b1 := g.blocks[i+1]!
      let b2 := g.blocks[i+2]!
      let b3 := g.blocks[i+3]!
      match b0.body with
      | .Reduce .sumOfSquares _ _ _ =>
        if b0.writes.size != 1 then return false
        let midId := b0.writes[0]!.tensorId
        match (matmulConsuming b1 midId), (matmulConsuming b2 midId),
              (matmulConsuming b3 midId) with
        | some (qKey, qTid), some (kKey, kTid), some (vKey, vTid) =>
          match ← findNorm normHandles midId with
          | none => return false
          | some norm =>
            let wQ ← findLayer matmulLayers qKey
            let wK ← findLayer matmulLayers kKey
            let wV ← findLayer matmulLayers vKey
            let inTid := match b0.body with
              | .Reduce _ rin _ _ => rin.tensorId
              | _                 => 0
            let inputBuf ← findBuf externalBufs inTid
            let qBuf ← findBuf externalBufs qTid
            let kBuf ← findBuf externalBufs kTid
            let vBuf ← findBuf externalBufs vTid
            let kvRef : IO.Ref (Option (GPUBackend.CachedDispatch β)) ← IO.mkRef none
            Hesper.Layers.Linear.forwardFusedNormQKV ctx norm wQ wK wV
              inputBuf qBuf kBuf vBuf kvRef
            return true
        | _, _, _ => return false
      | _ => return false
    if ← try4Block then
      i := i + 4
      continue
    -- ===========================================================
    -- Pattern B: [Reduce(sumSq); MatMul wQ]
    -- Single consumer — route to forwardFusedNormWQ.
    -- ===========================================================
    let try2Block : IO Bool := do
      if i + 1 ≥ n then return false
      let b0 := g.blocks[i]!
      let b1 := g.blocks[i+1]!
      match b0.body, b1.body with
      | .Reduce .sumOfSquares _ _ _, .MatMul lid _ _ epi =>
        if b0.writes.size != 1 ∨ b1.writes.size != 1 then return false
        if epi != (.input 0 : ScalarExp) then return false
        let midId := b0.writes[0]!.tensorId
        match ← findNorm normHandles midId with
        | none => return false
        | some norm =>
          let layer ← findLayer matmulLayers lid
          let inTid := match b0.body with
            | .Reduce _ rin _ _ => rin.tensorId
            | _                 => 0
          let inputBuf ← findBuf externalBufs inTid
          let outBuf ← findBuf externalBufs b1.writes[0]!.tensorId
          Hesper.Layers.Linear.forwardFusedNormWQ ctx norm layer inputBuf outBuf
          return true
      | _, _ => return false
    if ← try2Block then
      i := i + 2
      continue
    -- ===========================================================
    -- Pattern E: standalone [Reduce(sumSq, applyBody = (x*invRms)*scale + residual)]
    -- Post-FFN RMSNorm + residual-add.  Route to forwardNormThenAdd.
    -- ===========================================================
    let tryPostFFN : IO Bool := do
      match cur.body with
      | .Reduce .sumOfSquares _ _ applyBody =>
        if cur.writes.size != 1 ∨ cur.reads.size != 3 then return false
        -- Structural shape check: top-level is `.add (_) (.input 3)`,
        -- i.e. residual is read at slot 3 and combined by an outer add.
        -- That distinguishes this from the RMSNorm-only Reduce used by
        -- the QKV/FFN patterns (which have top-level `.mul`).
        match applyBody with
        | .add _ (.input 3) =>
          let outTid := cur.writes[0]!.tensorId
          match ← findNorm normHandles outTid with
          | none => return false
          | some norm =>
            -- reads[0] = ffnOut, reads[1] = scale, reads[2] = residual.
            -- The scale handled by `forwardNormThenAdd` comes from
            -- `norm.scale`; the externally-declared scale tensor is the
            -- same buffer, so we don't re-bind it here.
            let layerOutBuf ← findBuf externalBufs cur.reads[0]!.tensorId
            let residualBuf ← findBuf externalBufs cur.reads[2]!.tensorId
            let outBuf      ← findBuf externalBufs outTid
            let ref : IO.Ref (Option (GPUBackend.CachedDispatch β)) ← IO.mkRef none
            Hesper.Layers.RMSNorm.forwardNormThenAdd ctx norm
              layerOutBuf residualBuf outBuf ref
            return true
        | _ => return false
      | _ => return false
    if ← tryPostFFN then
      i := i + 1
      continue
    -- ===========================================================
    -- Pattern F: standalone [Scatter] or [ScatterMulti] → generic
    -- lowered dispatch.  Lowers the single-block graph via
    -- `lowerBlockGraph` and executes with (input, extras…, dstK…).
    -- ===========================================================
    let tryScatter : IO Bool := do
      -- `numWrites` is 1 for single Scatter, N for ScatterMulti.
      let numWrites : Nat := match cur.body with
        | .Scatter _ _       => 1
        | .ScatterMulti ops  => ops.size
        | _                  => 0
      if numWrites == 0 then return false
      if cur.reads.size < 1 ∨ cur.writes.size != numWrites then return false
      let inputBuf ← findBuf externalBufs cur.reads[0]!.tensorId
      let dim := match g.tensors.find? (·.id == cur.reads[0]!.tensorId) with
        | some t => t.shape.foldl (· * ·) 1
        | none   => 0
      if dim == 0 then return false
      let wgSize : Nat := if dim < 64 then dim else 64
      let shader := Hesper.Circuit.IRv2.lowerBlockGraph
        { tensors := g.tensors, blocks := #[cur] } wgSize
      -- Build the named-buffer list.  Single Scatter binds "dst";
      -- ScatterMulti binds "dst0", "dst1", ….
      let mut namedBufs : List (String × GPUBackend.Buf β) :=
        [("input", inputBuf)]
      match cur.body with
      | .Scatter _ _ =>
        let dstBuf ← findBuf externalBufs cur.writes[0]!.tensorId
        namedBufs := ("dst", dstBuf) :: namedBufs
      | .ScatterMulti _ =>
        for k in [0:numWrites] do
          let dstBuf ← findBuf externalBufs cur.writes[k]!.tensorId
          namedBufs := (s!"dst{k}", dstBuf) :: namedBufs
      | _ => pure ()
      for j in [0:(cur.reads.size - 1)] do
        let nm := if j == 0 then "scale" else s!"extra{j}"
        let buf ← findBuf externalBufs cur.reads[j+1]!.tensorId
        namedBufs := (nm, buf) :: namedBufs
      let nwg := (dim + wgSize - 1) / wgSize
      GPUBackend.executeWithConfig ctx shader namedBufs
        { numWorkgroups := (nwg, 1, 1),
          workgroupSize := { x := wgSize, y := 1, z := 1 } }
      return true
    if ← tryScatter then
      i := i + 1
      continue
    -- ===========================================================
    -- Pattern C (fallback): single MatMul via plain quantize+epilogue.
    -- ===========================================================
    match cur.body with
    | .MatMul _ _ _ _ =>
      runMatmulBlock ctx cur externalBufs matmulLayers matmulInputBufs
      i := i + 1
    | _ =>
      throw (IO.userError s!"runBlockGraph: block {i} has no dispatcher")

/-! ## Static analysis: dispatch count / pattern breakdown

`analyzeBlockGraph` walks a `BlockGraph` the same way `runBlockGraph`
does, but instead of dispatching it counts per-pattern hits and the
total number of physical GPU kernel launches each pattern emits.  Used
by the Phase-B stocktake to quantify the dispatch-reduction win.
-/

/-- Breakdown of how many physical kernel launches a BlockGraph emits,
    split by dispatcher-recognised pattern. -/
structure DispatchAnalysis where
  blocks            : Nat := 0
  patternD_ffn      : Nat := 0  -- 5-block FFN → 2 dispatches each
  patternA_qkv      : Nat := 0  -- 4-block NormQKV → 3 dispatches each
  patternB_normWQ   : Nat := 0  -- 2-block NormWQ → 2 dispatches each
  patternE_postNorm : Nat := 0  -- 1-block postNormAdd → 1 dispatch each
  patternF_scatter  : Nat := 0  -- 1-block Scatter → 1 dispatch each
  patternF_multi    : Nat := 0  -- 1-block ScatterMulti → 1 dispatch each
  patternC_matmul   : Nat := 0  -- 1-block plain MatMul → 2 dispatches each (quantize+matmul)
  quantizeStandalone : Nat := 0 -- explicit Quantize blocks → 1 dispatch each
  -- ── Logical Monolith counters ──
  monoAttention     : Nat := 0  -- GemmaAttentionMonolith → 6 dispatches each
  monoFFN           : Nat := 0  -- GemmaFFNMonolith → 3 dispatches each
  monoFlashAttn     : Nat := 0  -- FlashAttention → 2 dispatches each
  monoAttnOut       : Nat := 0  -- GemmaAttnOutProj → 2 dispatches each
  monoPostAttn      : Nat := 0  -- PostAttnNormAdd → 1 dispatch each
  monoPostFFN       : Nat := 0  -- PostFFNNormAdd → 1 dispatch each
  placeholder       : Nat := 0  -- Pointwise/Reduce blocks not matched
  deriving Repr, Inhabited

/-- Total physical kernel launches emitted by the dispatcher for `g`.
    Pattern D (FFN) = forwardFusedNormGateUp (2) + wDown via forwardDP4A (2) = 4.
    Pattern A (NormQKV) = forwardFusedNormQKV (fused-rmsnorm-q8_1 + wQ + wK+wV) = 3.
    Pattern B (NormWQ) = forwardFusedNormWQ (fused-rmsnorm-q8_1 + dp4a) = 2.
    Pattern C (plain MatMul) = quantize_q8_1 + dp4a-matmul-epilogue = 2. -/
def DispatchAnalysis.totalLaunches (a : DispatchAnalysis) : Nat :=
  a.patternD_ffn * 4
  + a.patternA_qkv * 3
  + a.patternB_normWQ * 2
  + a.patternE_postNorm * 1
  + a.patternF_scatter * 1
  + a.patternF_multi * 1
  + a.patternC_matmul * 2
  + a.quantizeStandalone * 1
  -- Monolith physical-dispatch costs (matches the production hand-fused
  -- sequence each Monolith expands into):
  + a.monoAttention * 6     -- forwardFusedNormQKV (3) + qkvNorm (1) + ropeFreqQ (1) + scatterMulti (1)
  + a.monoFFN       * 3     -- forwardFusedNormGateUp (2) + wDown.forward (1)
  + a.monoFlashAttn * 2     -- tiled flashAttn (phase 1 + phase 2)
  + a.monoAttnOut   * 2     -- wO: quantize + dp4a matmul
  + a.monoPostAttn  * 1     -- forwardNormThenAdd (postAttnNorm)
  + a.monoPostFFN   * 1     -- forwardNormThenAdd (postFFNNorm)
  + a.placeholder * 1

/-- Logical block count: how many AST nodes the IRv2 graph has.
    For Monoliths this is 1 per node (the whole point — abstraction). -/
def DispatchAnalysis.totalLogicalBlocks (a : DispatchAnalysis) : Nat :=
  a.blocks

/-- Pretty-print the analysis. -/
def DispatchAnalysis.toReport (a : DispatchAnalysis) : String := Id.run do
  let mut s : String := ""
  s := s ++ s!"  total blocks in graph     : {a.blocks}\n"
  s := s ++ s!"  Pattern D (FFN 5-block → 4): {a.patternD_ffn}    = {a.patternD_ffn * 4} dispatches\n"
  s := s ++ s!"  Pattern A (NormQKV → 3)   : {a.patternA_qkv}    = {a.patternA_qkv * 3} dispatches\n"
  s := s ++ s!"  Pattern B (NormWQ → 2)    : {a.patternB_normWQ}    = {a.patternB_normWQ * 2} dispatches\n"
  s := s ++ s!"  Pattern E (postNormAdd)   : {a.patternE_postNorm}    = {a.patternE_postNorm * 1} dispatches\n"
  s := s ++ s!"  Pattern F (Scatter single): {a.patternF_scatter}    = {a.patternF_scatter * 1} dispatches\n"
  s := s ++ s!"  Pattern F (ScatterMulti)  : {a.patternF_multi}    = {a.patternF_multi * 1} dispatches\n"
  s := s ++ s!"  Pattern C (plain MatMul)  : {a.patternC_matmul}    = {a.patternC_matmul * 2} dispatches\n"
  s := s ++ s!"  Quantize (standalone)     : {a.quantizeStandalone}    = {a.quantizeStandalone * 1} dispatches\n"
  s := s ++ s!"  ── Monolith logical nodes ──\n"
  s := s ++ s!"  GemmaAttentionMonolith (→6): {a.monoAttention}    = {a.monoAttention * 6} dispatches\n"
  s := s ++ s!"  FlashAttention         (→2): {a.monoFlashAttn}    = {a.monoFlashAttn * 2} dispatches\n"
  s := s ++ s!"  GemmaAttnOutProj       (→2): {a.monoAttnOut}    = {a.monoAttnOut * 2} dispatches\n"
  s := s ++ s!"  PostAttnNormAdd        (→1): {a.monoPostAttn}    = {a.monoPostAttn * 1} dispatches\n"
  s := s ++ s!"  GemmaFFNMonolith       (→3): {a.monoFFN}    = {a.monoFFN * 3} dispatches\n"
  s := s ++ s!"  PostFFNNormAdd         (→1): {a.monoPostFFN}    = {a.monoPostFFN * 1} dispatches\n"
  s := s ++ s!"  Placeholder (Pointwise/Reduce not matched): {a.placeholder}\n"
  s := s ++ s!"  ---\n"
  s := s ++ s!"  TOTAL physical kernel launches: {a.totalLaunches}\n"
  return s

/-- Walk `g` and classify each block the way `runBlockGraph` would.
    The logic mirrors `runBlockGraph` exactly so the analysis stays
    truthful: if a new pattern is added to the runtime, add it here. -/
def analyzeBlockGraph (g : BlockGraph) : DispatchAnalysis := Id.run do
  let mut a : DispatchAnalysis := { blocks := g.blocks.size }
  let mut i : Nat := 0
  let n := g.blocks.size
  let matmulConsuming (b : Block) (midId : Nat) : Option Unit :=
    match b.body with
    | .MatMul _ _ _ epi =>
      if epi == (.input 0 : ScalarExp)
         ∧ b.reads.size == 1 ∧ b.reads[0]!.tensorId == midId
         ∧ b.writes.size == 1 then some ()
      else none
    | _ => none
  let isGeluMul (b : Block) (gateTid upTid : Nat) : Option Nat :=
    match b.body with
    | .Pointwise body =>
      let expected : ScalarExp := .mul (.gelu (.input 0)) (.input 1)
      if body == expected
         ∧ b.reads.size == 2 ∧ b.writes.size == 1
         ∧ b.reads[0]!.tensorId == gateTid
         ∧ b.reads[1]!.tensorId == upTid then
        some b.writes[0]!.tensorId
      else none
    | _ => none
  -- A "norm-style head" is a Reduce or ReduceQuantize over sumOfSquares
  -- with exactly 1 write — the shape that appears at the head of
  -- Patterns A/B/D.  After `fuseReduceIntoQuantize`, the head will
  -- usually be the ReduceQuantize variant.
  let isNormHead (b : Block) : Option Nat :=
    if b.writes.size != 1 then none
    else
      match b.body with
      | .Reduce .sumOfSquares _ _ _         => some b.writes[0]!.tensorId
      | .ReduceQuantize .sumOfSquares _ _ _ => some b.writes[0]!.tensorId
      | _ => none
  while i < n do
    let cur := g.blocks[i]!
    -- Pattern D: 5-block FFN
    if i + 4 < n then
      let b0 := g.blocks[i]!
      let b1 := g.blocks[i+1]!
      let b2 := g.blocks[i+2]!
      let b3 := g.blocks[i+3]!
      let b4 := g.blocks[i+4]!
      match isNormHead b0 with
      | some midId =>
        match matmulConsuming b1 midId, matmulConsuming b2 midId with
        | some _, some _ =>
          let gateTid := match b1.body with
            | .MatMul _ _ _ _ => b1.writes[0]!.tensorId
            | _               => 0
          let upTid := match b2.body with
            | .MatMul _ _ _ _ => b2.writes[0]!.tensorId
            | _               => 0
          match isGeluMul b3 gateTid upTid with
          | some geluTid =>
            match matmulConsuming b4 geluTid with
            | some _ =>
              a := { a with patternD_ffn := a.patternD_ffn + 1 }
              i := i + 5
              continue
            | none => pure ()
          | none => pure ()
        | _, _ => pure ()
      | none => pure ()
    -- Pattern A: 4-block NormQKV
    if i + 3 < n then
      let b0 := g.blocks[i]!
      let b1 := g.blocks[i+1]!
      let b2 := g.blocks[i+2]!
      let b3 := g.blocks[i+3]!
      match isNormHead b0 with
      | some midId =>
        match matmulConsuming b1 midId, matmulConsuming b2 midId,
              matmulConsuming b3 midId with
        | some _, some _, some _ =>
          a := { a with patternA_qkv := a.patternA_qkv + 1 }
          i := i + 4
          continue
        | _, _, _ => pure ()
      | none => pure ()
    -- Pattern B: 2-block NormWQ
    if i + 1 < n then
      let b0 := g.blocks[i]!
      let b1 := g.blocks[i+1]!
      match isNormHead b0, b1.body with
      | some _, .MatMul _ _ _ epi =>
        if b1.writes.size == 1 ∧ epi == (.input 0 : ScalarExp) then
          a := { a with patternB_normWQ := a.patternB_normWQ + 1 }
          i := i + 2
          continue
      | _, _ => pure ()
    -- Pattern E: post-norm+residual (single Reduce with .add _ (.input 3))
    match cur.body with
    | .Reduce .sumOfSquares _ _ (.add _ (.input 3)) =>
      a := { a with patternE_postNorm := a.patternE_postNorm + 1 }
      i := i + 1
      continue
    | _ => pure ()
    -- Pattern F: standalone Scatter / ScatterMulti
    match cur.body with
    | .Scatter _ _ =>
      a := { a with patternF_scatter := a.patternF_scatter + 1 }
      i := i + 1
      continue
    | .ScatterMulti _ =>
      a := { a with patternF_multi := a.patternF_multi + 1 }
      i := i + 1
      continue
    | _ => pure ()
    -- Pattern C: plain MatMul (quantize+matmul, 2 dispatches)
    match cur.body with
    | .MatMul _ _ _ _ =>
      a := { a with patternC_matmul := a.patternC_matmul + 1 }
      i := i + 1
      continue
    | _ => pure ()
    -- Standalone Quantize: 1 dispatch (quantizeQ8_1Kernel).
    match cur.body with
    | .Quantize =>
      a := { a with quantizeStandalone := a.quantizeStandalone + 1 }
      i := i + 1
      continue
    | _ => pure ()
    -- Logical Monolith blocks.
    match cur.body with
    | .GemmaAttentionMonolith _ _ =>
      a := { a with monoAttention := a.monoAttention + 1 }
      i := i + 1
      continue
    | .FlashAttention _ _ =>
      a := { a with monoFlashAttn := a.monoFlashAttn + 1 }
      i := i + 1
      continue
    | .GemmaAttnOutProj _ =>
      a := { a with monoAttnOut := a.monoAttnOut + 1 }
      i := i + 1
      continue
    | .PostAttnNormAdd _ =>
      a := { a with monoPostAttn := a.monoPostAttn + 1 }
      i := i + 1
      continue
    | .GemmaFFNMonolith _ =>
      a := { a with monoFFN := a.monoFFN + 1 }
      i := i + 1
      continue
    | .PostFFNNormAdd _ =>
      a := { a with monoPostFFN := a.monoPostFFN + 1 }
      i := i + 1
      continue
    | _ => pure ()
    -- Placeholder: Pointwise/Reduce blocks that don't match any pattern.
    -- For the stocktake these count as 1 dispatch each (pessimistic).
    a := { a with placeholder := a.placeholder + 1 }
    i := i + 1
  return a

/-! ## Phase D2: Monolith runtime

`AttnBundle` / `FFNBundle` group all production handles a Monolith
needs (norms + Q4_K weights + RoPE freq buffer + flash-attention
shapes).  `runMonolithicGraph` walks a Monolith-shaped BlockGraph and
expands each Monolith node into the production hand-fused sequence,
firing the **physical** kernels.

Buffer plumbing: the dispatcher exposes the same `externalBufs` map
as the fine-grain runtime, so the same caller wiring code works for
both modes.  The Monolith node itself dictates which buffer goes
where — block.reads[0] = inputBuf, block.writes[i] = output buffers
in monolith-specific order. -/

/-- Bundle of production handles for a single attention site.  Mirrors
    `Hesper.Models.Gemma4.Gemma4Block.attention` + `attnNorm` etc. -/
structure AttnBundle (β : Type) [GPUBackend β] where
  attnNorm    : RMSNorm (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)
  wQ          : LinearLayer (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)
  wK          : LinearLayer (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)
  wV          : LinearLayer (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)
  wO          : LinearLayer (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)
  qNormScale  : GPUBackend.Buf β
  kNormScale  : GPUBackend.Buf β
  postAttnNorm : RMSNorm (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)
  freqFactors : Option (GPUBackend.Buf β)
  paramsBuf   : GPUBackend.Buf β  -- [pos, cacheLen]
  kCacheBuf   : GPUBackend.Buf β
  vCacheBuf   : GPUBackend.Buf β
  -- Post-qkvNorm scratch buffers (production uses qBuf2/kBuf2/vBuf2).
  -- Required because `fusedPerHeadQKVNormKernel` is NOT in-place-safe.
  qBuf2       : GPUBackend.Buf β
  kBuf2       : GPUBackend.Buf β
  vBuf2       : GPUBackend.Buf β
  -- Flash-attention shape parameters (per-layer constants).
  numHeads    : Nat
  numKVHeads  : Nat
  headDim     : Nat
  maxSeqLen   : Nat
  attnScale   : Float
  -- RoPE theta base for this layer (10000 for SWA, 1000000 for full-attn).
  ropeBase    : Float

/-- Bundle of production handles for a single FFN site. -/
structure FFNBundle (β : Type) [GPUBackend β] where
  ffnNorm     : RMSNorm (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)
  wGate       : LinearLayer (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)
  wUp         : LinearLayer (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)
  wDown       : LinearLayer (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)
  postFFNNorm : RMSNorm (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)
  geluScratch : GPUBackend.Buf β  -- intermediate [interDim] for gate*up output

private def findAttnBundle [GPUBackend β]
    (bundles : List (UInt64 × AttnBundle β)) (key : UInt64) : IO (AttnBundle β) :=
  match bundles.find? (·.fst == key) with
  | some (_, b) => pure b
  | none => throw (IO.userError s!"runMonolith: no AttnBundle for key {key.toNat}")

private def findFFNBundle [GPUBackend β]
    (bundles : List (UInt64 × FFNBundle β)) (key : UInt64) : IO (FFNBundle β) :=
  match bundles.find? (·.fst == key) with
  | some (_, b) => pure b
  | none => throw (IO.userError s!"runMonolith: no FFNBundle for key {key.toNat}")

/-- Walk a Monolith-shaped BlockGraph and execute each node by
    expanding into the production hand-fused sequence.

    Block-shape contract (from `Gemma4_v2.forwardLayerLazyMonolith`):
      - `GemmaAttentionMonolith`: reads = [input], writes = [qBuf, kBuf, vBuf]
      - `FlashAttention`        : reads = [qBuf, kBuf, vBuf], writes = [attnOut]
      - `GemmaFFNMonolith`      : reads = [attnOut], writes = [ffnOut]
      - `PostFFNNormAdd`        : reads = [ffnOut, residual], writes = [out]
-/
partial def runMonolithicGraph [GPUBackend β]
    (ctx : β) (g : BlockGraph)
    (externalBufs : List (Nat × GPUBackend.Buf β))
    (attnBundles  : List (UInt64 × AttnBundle β))
    (ffnBundles   : List (UInt64 × FFNBundle β))
    : IO Unit := do
  for h : i in [0:g.blocks.size] do
    let b := g.blocks[i]!
    match b.body with
    | .GemmaAttentionMonolith key pos =>
      if b.reads.size != 1 ∨ b.writes.size != 3 then
        throw (IO.userError "GemmaAttentionMonolith: expected reads=[input], writes=[q,k,v]")
      let bundle ← findAttnBundle attnBundles key
      let inputBuf ← findBuf externalBufs b.reads[0]!.tensorId
      let qBuf     ← findBuf externalBufs b.writes[0]!.tensorId
      let kBuf     ← findBuf externalBufs b.writes[1]!.tensorId
      let vBuf     ← findBuf externalBufs b.writes[2]!.tensorId
      let _ := pos  -- pos is in bundle.paramsBuf for production kernels
      -- 1) RMSNorm + Q8_1 + wQ + wK+wV  (3 dispatches)
      let kvRef : IO.Ref (Option (GPUBackend.CachedDispatch β)) ← IO.mkRef none
      Hesper.Layers.Linear.forwardFusedNormQKV ctx
        bundle.attnNorm bundle.wQ bundle.wK bundle.wV
        inputBuf qBuf kBuf vBuf kvRef
      -- 2) Per-head qkvNorm: qBuf → qBuf2, kBuf → kBuf2, vBuf → vBuf2.
      -- NOT in-place-safe; production uses separate read/write pointers.
      let qkvShader := Hesper.Models.Gemma4.fusedPerHeadQKVNormKernel
        bundle.numHeads bundle.numKVHeads bundle.headDim bundle.attnNorm.config.eps
      let qkvWgSize := if bundle.headDim < 256 then bundle.headDim else 256
      let qkvCfg : Hesper.ExecConfig :=
        { workgroupSize := { x := qkvWgSize, y := 1, z := 1 }
          numWorkgroups := (bundle.numHeads, 3, 1) }
      let qkvKey : UInt64 :=
        hash ("mono-qkvNorm", bundle.numHeads, bundle.numKVHeads, bundle.headDim)
      GPUBackend.executeWithConfigCached ctx qkvShader
        [("q_in", qBuf),  ("q_scale", bundle.qNormScale), ("q_out", bundle.qBuf2),
         ("k_in", kBuf),  ("k_scale", bundle.kNormScale), ("k_out", bundle.kBuf2),
         ("v_in", vBuf),                                   ("v_out", bundle.vBuf2)]
        qkvCfg qkvKey (← IO.mkRef none)
      -- 3/4) RoPE-Q, RoPE-K + KV-cache write.
      -- Two variants matching production's `Gemma4.forwardBlock`:
      --   * freqFactors.isSome: fused RoPE-K+KV-write kernel (one dispatch),
      --                         RoPE-Q is its own dispatch.
      --   * freqFactors.isNone: separate RoPE-Q, RoPE-K, and KV-write dispatches
      --                         (three dispatches, SWA layers).
      let kvDim := bundle.numKVHeads * bundle.headDim
      match bundle.freqFactors with
      | some ff =>
        let ropeShader := Hesper.Models.Gemma4.ropeWithFreqFactorsKernel
          bundle.headDim bundle.numHeads bundle.ropeBase
        let ropeKey : UInt64 := hash ("mono-rope-q-freq", bundle.headDim, bundle.numHeads, bundle.ropeBase.toUInt64)
        GPUBackend.executeWithConfigCached ctx ropeShader
          [("input", bundle.qBuf2), ("output", qBuf),
           ("params", bundle.paramsBuf), ("freq_factors", ff)]
          (Hesper.ExecConfig.dispatch1D (bundle.numHeads * bundle.headDim / 2))
          ropeKey (← IO.mkRef none)
        let kvShader := Hesper.Layers.Attention.fusedRopeKAndCacheWriteKernel
          bundle.numKVHeads bundle.maxSeqLen bundle.headDim kvDim bundle.ropeBase
        let kvKey : UInt64 :=
          hash ("mono-kv-write-freq", bundle.numKVHeads, bundle.maxSeqLen, bundle.headDim, bundle.ropeBase.toUInt64)
        GPUBackend.executeWithConfigCached ctx kvShader
          [("new_k", bundle.kBuf2), ("new_v", bundle.vBuf2),
           ("k_cache", bundle.kCacheBuf), ("v_cache", bundle.vCacheBuf),
           ("params", bundle.paramsBuf), ("freq_factors", ff)]
          (Hesper.ExecConfig.dispatch1D kvDim)
          kvKey (← IO.mkRef none)
      | none =>
        -- SWA variant: plain RoPE (no freq scaling), separate K write.
        let qRopeConfig : Hesper.Layers.RoPE.Config :=
          { dim := bundle.numHeads * bundle.headDim
            maxSeqLen := bundle.maxSeqLen
            base := bundle.ropeBase }
        let qRopeShader := Hesper.Layers.RoPE.ropeKernelDynamic qRopeConfig 1 1 bundle.numHeads bundle.headDim
        let qRopeKey : UInt64 := hash ("mono-rope-q-dyn", bundle.headDim, bundle.numHeads, bundle.ropeBase.toUInt64)
        GPUBackend.executeWithConfigCached ctx qRopeShader
          [("input", bundle.qBuf2), ("output", qBuf),
           ("params", bundle.paramsBuf)]
          (Hesper.ExecConfig.dispatch1D (bundle.numHeads * bundle.headDim / 2))
          qRopeKey (← IO.mkRef none)
        -- RoPE-K: kBuf2 → kBuf (in-place-unsafe, so use kBuf as output).
        let kRopeConfig : Hesper.Layers.RoPE.Config :=
          { dim := kvDim, maxSeqLen := bundle.maxSeqLen, base := bundle.ropeBase }
        let kRopeShader := Hesper.Layers.RoPE.ropeKernelDynamic kRopeConfig 1 1 bundle.numKVHeads bundle.headDim
        let kRopeKey : UInt64 := hash ("mono-rope-k-dyn", bundle.headDim, bundle.numKVHeads, bundle.ropeBase.toUInt64)
        GPUBackend.executeWithConfigCached ctx kRopeShader
          [("input", bundle.kBuf2), ("output", kBuf),
           ("params", bundle.paramsBuf)]
          (Hesper.ExecConfig.dispatch1D (kvDim / 2))
          kRopeKey (← IO.mkRef none)
        -- Plain K+V cache write: post-RoPE kBuf + vBuf2 → kCache, vCache.
        let kvShader := Hesper.Layers.Attention.fusedCacheWriteKVKernel
          bundle.numKVHeads bundle.maxSeqLen bundle.headDim kvDim
        let kvKey : UInt64 :=
          hash ("mono-kv-write-plain", bundle.numKVHeads, bundle.maxSeqLen, bundle.headDim)
        GPUBackend.executeWithConfigCached ctx kvShader
          [("new_k", kBuf), ("new_v", bundle.vBuf2),
           ("k_cache", bundle.kCacheBuf), ("v_cache", bundle.vCacheBuf),
           ("params", bundle.paramsBuf)]
          (Hesper.ExecConfig.dispatch1D kvDim)
          kvKey (← IO.mkRef none)

    | .FlashAttention key pos =>
      if b.reads.size != 3 ∨ b.writes.size != 1 then
        throw (IO.userError "FlashAttention: expected reads=[q,k,v], writes=[attnOut]")
      let bundle ← findAttnBundle attnBundles key
      let qBuf      ← findBuf externalBufs b.reads[0]!.tensorId
      let _kBufLane ← findBuf externalBufs b.reads[1]!.tensorId  -- not used: K from cache
      let _vBufLane ← findBuf externalBufs b.reads[2]!.tensorId  -- not used: V from cache
      let attnOut   ← findBuf externalBufs b.writes[0]!.tensorId
      -- cacheLen = pos+1 (this token + all prior cached tokens).
      let cacheLen := pos + 1
      -- Match production's kernel selection: small cacheLen (≤32) uses the
      -- dynamicParams kernel (reads pos/cacheLen from paramsBuf), large uses
      -- tiled.  This is required for bit-parity — tiled with tiny cacheLen
      -- has off-by-one behavior in the tail.
      if cacheLen > 32 then
        Hesper.WGSL.FlashAttention.executeFlashAttentionTiled ctx
          qBuf bundle.kCacheBuf bundle.vCacheBuf attnOut
          bundle.numHeads bundle.numKVHeads bundle.maxSeqLen bundle.headDim
          cacheLen bundle.attnScale none
      else
        GPUBackend.executeWithConfig ctx
          (Hesper.WGSL.FlashAttention.flashAttentionDynamicParamsKernel
            bundle.numHeads bundle.numKVHeads bundle.maxSeqLen bundle.headDim bundle.attnScale)
          [("q", qBuf), ("k_cache", bundle.kCacheBuf), ("v_cache", bundle.vCacheBuf),
           ("output", attnOut), ("params", bundle.paramsBuf)]
          ({ numWorkgroups := (bundle.numHeads, 1, 1) : Hesper.ExecConfig })

    | .GemmaAttnOutProj key =>
      if b.reads.size != 1 ∨ b.writes.size != 1 then
        throw (IO.userError "GemmaAttnOutProj: expected reads=[attnOut], writes=[wOOut]")
      let bundle ← findAttnBundle attnBundles key
      let inBuf  ← findBuf externalBufs b.reads[0]!.tensorId
      let outBuf ← findBuf externalBufs b.writes[0]!.tensorId
      Hesper.Layers.Linear.LinearLayer.forward ctx bundle.wO inBuf outBuf

    | .PostAttnNormAdd key =>
      if b.reads.size != 2 ∨ b.writes.size != 1 then
        throw (IO.userError "PostAttnNormAdd: expected reads=[wOOut, input], writes=[out]")
      let bundle ← findAttnBundle attnBundles key
      let wOOut    ← findBuf externalBufs b.reads[0]!.tensorId
      let residBuf ← findBuf externalBufs b.reads[1]!.tensorId
      let outBuf   ← findBuf externalBufs b.writes[0]!.tensorId
      let paRef : IO.Ref (Option (GPUBackend.CachedDispatch β)) ← IO.mkRef none
      Hesper.Layers.RMSNorm.forwardNormThenAdd ctx
        bundle.postAttnNorm wOOut residBuf outBuf paRef

    | .GemmaFFNMonolith key =>
      if b.reads.size != 1 ∨ b.writes.size != 1 then
        throw (IO.userError "GemmaFFNMonolith: expected reads=[in], writes=[out]")
      let bundle ← findFFNBundle ffnBundles key
      let inputBuf ← findBuf externalBufs b.reads[0]!.tensorId
      let outBuf   ← findBuf externalBufs b.writes[0]!.tensorId
      -- 1) forwardFusedNormGateUp (2 dispatches)
      let geluRef : IO.Ref (Option (GPUBackend.CachedDispatch β)) ← IO.mkRef none
      Hesper.Layers.Linear.forwardFusedNormGateUp ctx
        bundle.ffnNorm bundle.wGate bundle.wUp
        inputBuf bundle.geluScratch geluRef
      -- 2) wDown via LinearLayer.forward (1-2 dispatches depending on path)
      Hesper.Layers.Linear.LinearLayer.forward ctx
        bundle.wDown bundle.geluScratch outBuf

    | .PostFFNNormAdd key =>
      if b.reads.size != 2 ∨ b.writes.size != 1 then
        throw (IO.userError "PostFFNNormAdd: expected reads=[ffnOut, residual], writes=[out]")
      let bundle ← findFFNBundle ffnBundles key
      let ffnOut    ← findBuf externalBufs b.reads[0]!.tensorId
      let residBuf  ← findBuf externalBufs b.reads[1]!.tensorId
      let outBuf    ← findBuf externalBufs b.writes[0]!.tensorId
      let pfnRef : IO.Ref (Option (GPUBackend.CachedDispatch β)) ← IO.mkRef none
      Hesper.Layers.RMSNorm.forwardNormThenAdd ctx
        bundle.postFFNNorm ffnOut residBuf outBuf pfnRef

    | _ =>
      throw (IO.userError s!"runMonolithicGraph: block {i} is not a Monolith node — use runBlockGraph for fine-grain blocks")

/-! ## Phase E: CUDA Graph capture wrapper

`captureMonolithicGraph` runs the BlockGraph once on a dedicated
CUDA stream while it is in capture mode, producing a `CUgraphExec`
handle.  All ~504 kernel launches per token become **one host-side
`cuGraphLaunch` call** when replayed.

This is the IRv2 expression of the same trick production hesper uses
in `forwardSingleToken` under `HESPER_CUDA_GRAPHS=1`: the graph IS the
execution plan; capture-and-replay collapses launch overhead.

Usage:
```
-- Once at startup (or first decode call):
let exec ← captureMonolithicGraph ctx stream graph externalBufs
            attnBundles ffnBundles

-- Per decode token (assuming positions advance device-side):
Hesper.CUDA.cuGraphLaunch exec stream
Hesper.CUDA.cuStreamSynchronize stream
```

The CUDA backend's `executeWithConfig` family must route launches to
the capture stream while capture is active — this PoC assumes that
plumbing is already wired (production hesper has it; the IRv2
dispatcher inherits it via `GPUBackend.executeWithConfig`). -/

/-- Capture an entire Monolith BlockGraph as a single CUDA graph.
    Returns the executable graph handle for replay. -/
def captureMonolithicGraph
    (ctx : Hesper.CUDAContext) (stream : Hesper.CUDA.CUstream)
    (g : BlockGraph)
    (externalBufs : List (Nat × GPUBackend.Buf Hesper.CUDAContext))
    (attnBundles  : List (UInt64 × AttnBundle Hesper.CUDAContext))
    (ffnBundles   : List (UInt64 × FFNBundle Hesper.CUDAContext))
    : IO Hesper.CUDA.CUgraphExec := do
  Hesper.CUDA.cuStreamBeginCapture stream
  -- Route all launches through the capture stream.  The CUDA backend
  -- reads `cudaCaptureStream` in `launchKernelMaybeStream` and calls
  -- `cuLaunchKernelOnStream` when it's `some s`; otherwise launches
  -- on the default stream (which a different capture ignores).
  cudaCaptureStream.set (some stream)
  try
    runMonolithicGraph ctx g externalBufs attnBundles ffnBundles
  finally
    cudaCaptureStream.set none
  let graph ← Hesper.CUDA.cuStreamEndCapture stream
  let exec  ← Hesper.CUDA.cuGraphInstantiate graph
  Hesper.CUDA.cuGraphDestroy graph
  return exec

/-- Static count of the host-side launches the captured graph collapses
    to.  Always 1 (the single `cuGraphLaunch`).  This is the metric we
    advertise as the "Phase E win" — independent of how many kernel
    instances the graph contains. -/
def hostLaunchesAfterCapture : Nat := 1

end Hesper.Circuit.IRv2
