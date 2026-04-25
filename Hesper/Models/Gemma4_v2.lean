import Hesper.Circuit.IRv2

/-!
# Gemma 4 lazy builder v2 (Phase B foothold)

First deliverable of the Circuit IRv2 → model migration: a `BuilderM`
function that describes the wQ projection of a Gemma 4 attention block
as a `BlockGraph`, so it can be fused + lowered by the IRv2 compiler
pipeline instead of dispatched eagerly.

Scope of this file is deliberately narrow — one MatMul block with an
identity epilogue.  The accompanying parity test
(`Examples/DSL/Gemma4QProjParity.lean`) runs both the existing hand-
tuned path and the IRv2 path against the same input + same Q4_K
weights and asserts bit-identical output.

Once this parity holds, the graph grows block-by-block:
RMSNorm (Reduce block) → wQ MatMul → per-head Q norm → RoPE-Q, each
as a separate sub-PoC with its own assertion.
-/

namespace Hesper.Models.Gemma4_v2

open Hesper.Circuit
open Hesper.Circuit.IRv2

/-- Build the "Q projection" sub-graph — SINGLE MatMul variant.

        input[inDim]  ──────▶  [ MatMul wQ, epi = identity ]  ──▶  qOut[outDim]

    `weightLayerId` is an opaque handle the dispatcher resolves back
    to a `LinearLayer` at dispatch time.  The f32 input is consumed
    out-of-band by the Q8_1 quantize step inside the matmul kernel;
    the dispatcher expects the caller to register the input buffer in
    `matmulInputBufs` keyed by `weightLayerId`.

    Inputs:
      - `inputId`  : external tensor id for the f32 hidden state
      - `outId`    : external tensor id for the f32 Q output
      - `weightLayerId` : opaque key → wQ `LinearLayer`
      - `inDim`, `outDim` : matmul shape -/
def buildQProjLazy
    (inputId outId : Nat) (weightLayerId : UInt64)
    (inDim outDim : Nat) : BuilderM Unit := do
  declareExternal inputId #[inDim]  .f32 .Global
  declareExternal outId   #[outDim] .f32 .Global
  let outR : Region := { tensorId := outId }
  emitBlock
    { reads  := #[]
      writes := #[outR]
      body   := .MatMul weightLayerId outDim inDim (.input 0) }

/-- Build the "pre-norm + Q projection" sub-graph — 2-block variant.

        input[dim] ──▶ [ Reduce sumSq → mid ] ──▶ [ MatMul wQ ] ──▶ qOut[outDim]

    This is the minimal building block of a Gemma 4 attention
    prologue: a pre-attention RMSNorm feeding the wQ projection.  The
    dispatcher pattern-matches the adjacent `[Reduce(sumOfSquares),
    MatMul]` pair and picks the fused `forwardFusedNormWQ` path, which
    uses the hand-tuned `fusedRMSNormQ8_1Kernel` followed by the dp4a
    matmul — collapsing what would otherwise be 3 dispatches (norm,
    quantize, matmul) into 2.

    The Reduce block's `reads = [input, scale]` mirrors the hand-
    tuned kernel's binding order: slot 0 = the reduced scalar
    (virtual), slot 1 = per-lane input, slot 2 = per-lane scale.  Fed
    to `applyBody = (x * scale) * rsqrt(sum/N + eps)` — the RMSNorm
    identity.

    Inputs:
      - `inputId`, `scaleId`, `outId` : external tensor ids
      - `weightLayerId` : opaque key → wQ `LinearLayer`
      - `normKey` : tensor id used as the Reduce's write target; the
        dispatcher uses it to resolve the `RMSNorm` handle
      - `dim`, `outDim` : shapes (dim = norm input/output size, outDim
        = matmul output dim)
      - `eps` : RMSNorm epsilon -/
def buildNormQProjLazy
    (inputId scaleId outId : Nat) (weightLayerId : UInt64) (normKey : Nat)
    (dim outDim : Nat) (eps : Float) : BuilderM Unit := do
  declareExternal inputId #[dim]    .f32 .Global
  declareExternal scaleId #[dim]    .f32 .Global
  declareExternal outId   #[outDim] .f32 .Global
  let inR    : Region := { tensorId := inputId }
  let scaleR : Region := { tensorId := scaleId }
  let outR   : Region := { tensorId := outId }
  -- Intermediate "reduce scalar" — register-scope, dispatcher-visible
  -- only through `normKey` / `writes[0].tensorId`.
  let mid ← declareTensor #[1] .f32 .Register
  let midR : Region := { tensorId := mid.id }
  -- We force the Reduce-block output id to match `normKey` so the
  -- dispatcher's `normHandles` lookup is keyed on a builder-chosen
  -- tensor id; since `declareTensor` assigns fresh ids, we route the
  -- Reduce's write through `normKey` directly instead.
  let _ := midR  -- (kept for readability)
  -- Fused-form Reduce: applyBody = (input * scale) * rsqrt(sum/N + eps)
  --   slot 0 = reduced Σx²  (virtual, at reduceSlot)
  --   slot 1 = per-lane input
  --   slot 2 = per-lane scale
  let invRms : ScalarExp :=
    .rsqrt (.add (.div (.input 0) (.const dim.toFloat)) (.const eps))
  let applyBody : ScalarExp :=
    .mul (.mul (.input 1) (.input 2)) invRms
  emitBlock
    { reads := #[inR, scaleR]
      writes := #[{ tensorId := normKey }]
      body := .Reduce ReduceOp.sumOfSquares inR 0 applyBody }
  -- Block 2: MatMul wQ with identity epilogue.  Its f32 input is the
  -- normed hidden state; the dispatcher plumbs that through
  -- `matmulInputBufs` using `weightLayerId` as the key.
  emitBlock
    { reads  := #[]
      writes := #[outR]
      body   := .MatMul weightLayerId outDim dim (.input 0) }

/-- Build the "pre-norm + QKV projection" sub-graph — 4-block variant.

        input[dim] ─▶ [Reduce sumSq → mid] ─▶ [MatMul wQ] ─▶ qOut[qOutDim]
                                            ├▶ [MatMul wK] ─▶ kOut[kvOutDim]
                                            └▶ [MatMul wV] ─▶ vOut[kvOutDim]

    All three MatMul blocks logically consume the RMSNorm output
    (the Reduce block's `writes[0].tensorId == normKey`).  The
    dispatcher uses that adjacency + shared input to route to the
    hand-tuned `forwardFusedNormQKV` kernel sequence (fused-rmsnorm-
    q8_1, then wQ dp4a matmul, then fused wK+wV dp4a matmul).

    Inputs:
      - `inputId`, `scaleId` : external tensor ids for f32 input + γ
      - `qOutId`, `kOutId`, `vOutId` : external tensor ids for Q/K/V
      - `wQKey`, `wKKey`, `wVKey` : opaque keys → `LinearLayer`s
      - `normKey` : tensor id used as the Reduce's write target
      - `dim` : hidden size (= RMSNorm in/out size, matmul inDim)
      - `qOutDim` : wQ.outDim (typically numAttentionHeads * headDim)
      - `kvOutDim` : wK/wV.outDim (numKVHeads * headDim)
      - `eps` : RMSNorm epsilon -/
def buildNormQKVProjLazy
    (inputId scaleId qOutId kOutId vOutId : Nat)
    (wQKey wKKey wVKey : UInt64) (normKey : Nat)
    (dim qOutDim kvOutDim : Nat) (eps : Float) : BuilderM Unit := do
  declareExternal inputId #[dim]      .f32 .Global
  declareExternal scaleId #[dim]      .f32 .Global
  declareExternal qOutId  #[qOutDim]  .f32 .Global
  declareExternal kOutId  #[kvOutDim] .f32 .Global
  declareExternal vOutId  #[kvOutDim] .f32 .Global
  let inR     : Region := { tensorId := inputId }
  let scaleR  : Region := { tensorId := scaleId }
  let qOutR   : Region := { tensorId := qOutId }
  let kOutR   : Region := { tensorId := kOutId }
  let vOutR   : Region := { tensorId := vOutId }
  let normR   : Region := { tensorId := normKey }
  -- Block 1: Reduce (RMSNorm).  Same applyBody template as the
  -- 2-block form — slots 0=Σx² (virtual), 1=input, 2=scale.
  let invRms : ScalarExp :=
    .rsqrt (.add (.div (.input 0) (.const dim.toFloat)) (.const eps))
  let applyBody : ScalarExp :=
    .mul (.mul (.input 1) (.input 2)) invRms
  emitBlock
    { reads := #[inR, scaleR]
      writes := #[{ tensorId := normKey }]
      body := .Reduce ReduceOp.sumOfSquares inR 0 applyBody }
  -- Blocks 2-4: Q / K / V projections.  Each lists the Reduce output
  -- `normR` in its reads so the dispatcher can confirm they all
  -- consume the same source.  Epilogue is identity (slot 0 = dot).
  emitBlock
    { reads  := #[normR]
      writes := #[qOutR]
      body   := .MatMul wQKey qOutDim dim (.input 0) }
  emitBlock
    { reads  := #[normR]
      writes := #[kOutR]
      body   := .MatMul wKKey kvOutDim dim (.input 0) }
  emitBlock
    { reads  := #[normR]
      writes := #[vOutR]
      body   := .MatMul wVKey kvOutDim dim (.input 0) }

/-- Build the FFN pipeline sub-graph (5 blocks, pre-residual).

        input[dim]
         │
         ├──▶ [Reduce sumSq → mid]
         ├──▶ [MatMul wGate → gateOut[interDim]]   (reads mid)
         ├──▶ [MatMul wUp   → upOut  [interDim]]   (reads mid)
         ├──▶ [Pointwise (GELU(gateOut) * upOut) → geluOut[interDim]]
         └──▶ [MatMul wDown → ffnOut[dim]]        (reads geluOut)

    The dispatcher recognises this exact 5-block shape and routes to
    the hand-tuned 2-dispatch path:
      1. `forwardFusedNormGateUp` — RMSNorm+Q8_1+gate+up+GELU×mul, ONE
         dispatch, produces geluOut.
      2. `LinearLayer.forward` on `wDown` — another dispatch (internal
         Q8_1 quantize + dp4a Q4_K matmul), produces ffnOut.

    Residual-add with the pre-FFN hidden is NOT part of this graph;
    it lives in the post-FFN RMSNorm+add path, which is a separate
    sub-graph.

    Inputs:
      - `inputId`, `scaleId`, `outId`   : external tensor ids
      - `wGateKey`, `wUpKey`, `wDownKey`: opaque keys → `LinearLayer`s
      - `normKey`   : Reduce write target; dispatcher uses it to
                      resolve the `RMSNorm` handle
      - `geluKey`   : Pointwise write target; dispatcher uses it to
                      resolve the intermediate buffer for wDown input
      - `dim`       : hidden size (= RMSNorm in/out, wGate.inDim, wDown.outDim)
      - `interDim`  : FFN intermediate size (= wGate.outDim = wUp.outDim
                      = wDown.inDim)
      - `eps`       : RMSNorm epsilon -/
def buildFFNLazy
    (inputId scaleId outId : Nat)
    (wGateKey wUpKey wDownKey : UInt64)
    (normKey geluKey : Nat)
    (dim interDim : Nat) (eps : Float) : BuilderM Unit := do
  declareExternal inputId #[dim]       .f32 .Global
  declareExternal scaleId #[dim]       .f32 .Global
  declareExternal outId   #[dim]       .f32 .Global
  let inR     : Region := { tensorId := inputId }
  let scaleR  : Region := { tensorId := scaleId }
  let outR    : Region := { tensorId := outId }
  let normR   : Region := { tensorId := normKey }
  let geluR   : Region := { tensorId := geluKey }
  -- Intermediates (allocated by the dispatcher).  We declare them as
  -- register-scope tensors so the dispatcher knows they're not caller-
  -- supplied externals.
  let _gateT ← declareTensor #[interDim] .f32 .Register
  let gateR : Region := { tensorId := _gateT.id }
  let _upT   ← declareTensor #[interDim] .f32 .Register
  let upR   : Region := { tensorId := _upT.id }
  -- Block 1: Reduce (RMSNorm).  Same applyBody template as QKV.
  let invRms : ScalarExp :=
    .rsqrt (.add (.div (.input 0) (.const dim.toFloat)) (.const eps))
  let applyBody : ScalarExp :=
    .mul (.mul (.input 1) (.input 2)) invRms
  emitBlock
    { reads := #[inR, scaleR]
      writes := #[{ tensorId := normKey }]
      body := .Reduce ReduceOp.sumOfSquares inR 0 applyBody }
  -- Block 2: wGate matmul (identity epi).
  emitBlock
    { reads := #[normR]
      writes := #[gateR]
      body := .MatMul wGateKey interDim dim (.input 0) }
  -- Block 3: wUp matmul (identity epi).
  emitBlock
    { reads := #[normR]
      writes := #[upR]
      body := .MatMul wUpKey interDim dim (.input 0) }
  -- Block 4: Pointwise GELU(gate) * up.
  let geluXMul : ScalarExp :=
    .mul (.gelu (.input 0)) (.input 1)
  emitBlock
    { reads := #[gateR, upR]
      writes := #[{ tensorId := geluKey }]
      body := .Pointwise geluXMul }
  -- Block 5: wDown matmul (identity epi).  Reads geluR so the
  -- dispatcher can chain its input to the Pointwise's output buffer.
  emitBlock
    { reads := #[geluR]
      writes := #[outR]
      body := .MatMul wDownKey dim interDim (.input 0) }

/-- Build the post-FFN `RMSNorm(ffnOut) * scale + residual` sub-graph.

        ffnOut[dim], scale[dim], residual[dim]
                    │
                    ▼
        [ Reduce sumOfSquares →
            applyBody = (ffnOut * rsqrt(sum/N + eps)) * scale + residual ]
                    │
                    ▼
                 out[dim]

    A single `Reduce` block.  The Reduce's slot-space convention:
      slot 0 = Σ ffnOut²  (the reduced scalar, virtual at reduceSlot)
      slot 1 = ffnOut[i]  (per-lane input)
      slot 2 = scale[i]   (per-lane extra read)
      slot 3 = residual[i] (per-lane extra read)

    The dispatcher recognises this exact block shape — single Reduce
    with a `... + .input 3` tail — and routes to the hand-tuned
    `forwardNormThenAdd` kernel (one dispatch, no intermediate).

    Inputs:
      - `ffnOutId`, `scaleId`, `residualId`, `outId` : external tensor ids
      - `normKey` : Reduce write target; dispatcher uses it to
                    resolve the `RMSNorm` handle
      - `dim` : hidden size
      - `eps` : RMSNorm epsilon -/
def buildPostFFNLazy
    (ffnOutId scaleId residualId outId : Nat)
    (dim : Nat) (eps : Float) : BuilderM Unit := do
  declareExternal ffnOutId   #[dim] .f32 .Global
  declareExternal scaleId    #[dim] .f32 .Global
  declareExternal residualId #[dim] .f32 .Global
  declareExternal outId      #[dim] .f32 .Global
  let ffnOutR : Region := { tensorId := ffnOutId }
  let scaleR  : Region := { tensorId := scaleId }
  let residR  : Region := { tensorId := residualId }
  -- Fused-form Reduce: applyBody = (ffnOut * rsqrt(sum/N + eps)) * scale + residual
  let invRms : ScalarExp :=
    .rsqrt (.add (.div (.input 0) (.const dim.toFloat)) (.const eps))
  let applyBody : ScalarExp :=
    .add (.mul (.mul (.input 1) invRms) (.input 2)) (.input 3)
  -- The Reduce's writes[0] is the final output tensor id — callers
  -- register `(outId, RMSNorm-handle)` in `normHandles` so the
  -- dispatcher can resolve the layer when it matches this pattern.
  emitBlock
    { reads := #[ffnOutR, scaleR, residR]
      writes := #[{ tensorId := outId }]
      body := .Reduce ReduceOp.sumOfSquares ffnOutR 0 applyBody }

/-- Build the KV-cache write sub-graph for a single token position.

        vNew[kvDim]  ──▶  [ Scatter  indexExpr = cacheIdx(laneIdx)
                                      applyBody = .input 0 ]  ──▶  vCache

    Cache layout (Gemma 4 convention): `[numKVHeads, maxSeqLen, headDim]`,
    flattened as `cache[kvHead][pos][d]`.  For lane `i ∈ [0, kvDim)`:

        kvHead  = i / headDim
        d       = i % headDim
        cacheIdx(i) = kvHead * (maxSeqLen * headDim)
                    + pos     * headDim
                    + d

    `pos` is threaded as a compile-time Nat constant for this PoC (the
    production path reads it from a u32 params buffer; swapping to
    that is a follow-up once we add `ScalarExp.indexed` for the
    scatter lowering).

    Inputs:
      - `vNewId`   : external id of the new V tensor [kvDim]
      - `vCacheId` : external id of the cache buffer
      - `pos`      : current token position (constant for this PoC)
      - `numKVHeads`, `maxSeqLen`, `headDim` : cache layout constants
    `kvDim = numKVHeads * headDim` is derived. -/
def buildKVWriteLazy
    (vNewId vCacheId : Nat) (pos : Nat)
    (numKVHeads maxSeqLen headDim : Nat) : BuilderM Unit := do
  let kvDim := numKVHeads * headDim
  let cacheSize := numKVHeads * maxSeqLen * headDim
  declareExternal vNewId   #[kvDim]      .f32 .Global
  declareExternal vCacheId #[cacheSize]  .f32 .Global
  let vR     : Region := { tensorId := vNewId }
  let cacheR : Region := { tensorId := vCacheId }
  -- kvHead = laneIdx / headDim
  let kvHead : ScalarExp := .idiv (.toFloat .laneIdx) (.const headDim.toFloat)
  -- d = laneIdx % headDim
  let d : ScalarExp := .mod (.toFloat .laneIdx) (.const headDim.toFloat)
  -- cacheIdx = kvHead * (maxSeqLen * headDim) + pos * headDim + d
  let stride : Float := (maxSeqLen * headDim).toFloat
  let posOffset : Float := (pos * headDim).toFloat
  let cacheIdx : ScalarExp :=
    .add (.mul kvHead (.const stride))
         (.add (.const posOffset) d)
  emitBlock
    { reads  := #[vR]
      writes := #[cacheR]
      body   := .Scatter cacheIdx (.input 0) }

/-- Build the RoPE-K + KV-cache write sub-graph for a single token.

    Per-lane math (for lane `i ∈ [0, kvDim)`):
        kvHead  = i / headDim,     d = i % headDim
        halfDim = headDim / 2
        dInLow  = (d < halfDim)
        dPair   = dInLow ? d + halfDim : d - halfDim
        pairIdx = kvHead * headDim + dPair
        dimPair = dInLow ? d : d - halfDim        -- 0..halfDim-1
        freqFactor = freq_factors[dimPair]
        theta = pos * ropeBase^(-2*dimPair/headDim) / freqFactor
        x0 = dInLow ? new_k[i] : new_k[pairIdx]
        x1 = dInLow ? new_k[pairIdx] : new_k[i]
        myNew = dInLow ? (x0*cos θ − x1*sin θ)
                       : (x0*sin θ + x1*cos θ)
        cacheIdx = kvHead * (maxSeqLen * headDim) + pos * headDim + d
        k_cache[cacheIdx] = myNew

    Expressed as a single `Scatter` block with:
      reads  = [new_k, freq_factors]
      slot[0] = new_k[laneIdx]         (xSelf, pre-read)
      .indexed 0 addr  → gather from new_k
      .indexed 1 addr  → gather from freq_factors
    `applyBody` builds the NeoX rotation formula using `.select`.

    Inputs:
      - `kNewId`, `freqFactorsId`, `kCacheId` : external tensor ids
      - `pos`, `ropeBase` : compile-time constants
      - `numKVHeads`, `maxSeqLen`, `headDim` : layout constants -/
def buildRopeKWriteLazy
    (kNewId freqFactorsId kCacheId : Nat)
    (pos : Nat) (ropeBase : Float)
    (numKVHeads maxSeqLen headDim : Nat) : BuilderM Unit := do
  let halfDim   := headDim / 2
  let kvDim     := numKVHeads * headDim
  let cacheSize := numKVHeads * maxSeqLen * headDim
  declareExternal kNewId        #[kvDim]      .f32 .Global
  declareExternal freqFactorsId #[halfDim]    .f32 .Global
  declareExternal kCacheId      #[cacheSize]  .f32 .Global
  let kR     : Region := { tensorId := kNewId }
  let freqR  : Region := { tensorId := freqFactorsId }
  let cacheR : Region := { tensorId := kCacheId }
  -- Per-lane integer decomposition (f32 arithmetic — exact for laneIdx
  -- below 2^23, easily covers any real attention head shape).
  let laneF  : ScalarExp := .toFloat .laneIdx
  let hdF    : ScalarExp := .const headDim.toFloat
  let halfF  : ScalarExp := .const halfDim.toFloat
  let kvHead : ScalarExp := .idiv laneF hdF          -- laneIdx / headDim
  let dim_   : ScalarExp := .mod  laneF hdF          -- laneIdx % headDim
  -- dInLow = (d < halfDim).  For ScalarExp.lt the result is 1.0/0.0 f32.
  let dInLow : ScalarExp := .lt dim_ halfF
  -- dPair = dInLow ? d + halfDim : d - halfDim
  let dPair  : ScalarExp :=
    .select dInLow (.add dim_ halfF) (.sub dim_ halfF)
  -- pairIdx = kvHead * headDim + dPair
  let pairIdx : ScalarExp :=
    .add (.mul kvHead hdF) dPair
  -- dimPair = dInLow ? d : d - halfDim   (0..halfDim-1)
  let dimPair : ScalarExp :=
    .select dInLow dim_ (.sub dim_ halfF)
  -- xSelf = new_k[laneIdx] = .input 0 (pre-read by lowering).
  let xSelf  : ScalarExp := .input 0
  let xPair  : ScalarExp := .indexed 0 pairIdx        -- gather from new_k
  -- x0 = dInLow ? xSelf : xPair ; x1 = dInLow ? xPair : xSelf
  let x0 : ScalarExp := .select dInLow xSelf xPair
  let x1 : ScalarExp := .select dInLow xPair xSelf
  -- freqFactor = freq_factors[dimPair]  (gather from extra slot 1).
  let freqFactor : ScalarExp := .indexed 1 dimPair
  -- exponent = 2 * dimPair / headDim
  let exponent : ScalarExp :=
    .div (.mul (.const 2.0) dimPair) hdF
  -- freqInv = ropeBase^(-exponent)
  let freqInv : ScalarExp :=
    .pow (.const ropeBase) (.neg exponent)
  -- theta = pos * freqInv / freqFactor
  let theta : ScalarExp :=
    .div (.mul (.const pos.toFloat) freqInv) freqFactor
  let cosT : ScalarExp := .cos theta
  let sinT : ScalarExp := .sin theta
  -- x0_new = x0*cos - x1*sin ; x1_new = x0*sin + x1*cos
  let x0New : ScalarExp :=
    .sub (.mul x0 cosT) (.mul x1 sinT)
  let x1New : ScalarExp :=
    .add (.mul x0 sinT) (.mul x1 cosT)
  -- myNew = dInLow ? x0New : x1New
  let myNew : ScalarExp := .select dInLow x0New x1New
  -- cacheIdx = kvHead * (maxSeqLen * headDim) + pos * headDim + d
  let stride    : Float := (maxSeqLen * headDim).toFloat
  let posOffset : Float := (pos * headDim).toFloat
  let cacheIdx : ScalarExp :=
    .add (.mul kvHead (.const stride))
         (.add (.const posOffset) dim_)
  emitBlock
    { reads  := #[kR, freqR]
      writes := #[cacheR]
      body   := .Scatter cacheIdx myNew }

/-- Build the RoPE-K + V combined KV-cache write — ONE ScatterMulti block.

    Extends `buildRopeKWriteLazy` with a second (indexExpr, applyBody)
    op for the plain V copy, both firing from a single 1D dispatch.
    Semantically equivalent to issuing `buildRopeKWriteLazy` and
    `buildKVWriteLazy` back-to-back; physically it's ONE kernel launch.

    Reads layout:
      reads[0] = new_k       (primary; slot[0] = xSelf_k)
      reads[1] = freq_factors (gather via `.indexed 1 dimPair`)
      reads[2] = new_v       (gather via `.indexed 2 laneIdx`)
    Writes:
      writes[0] = k_cache (RoPE-rotated K)
      writes[1] = v_cache (plain V)
    Both writes use the same `cacheIdx` formula. -/
def buildRopeKVWriteLazy
    (kNewId freqFactorsId vNewId kCacheId vCacheId : Nat)
    (pos : Nat) (ropeBase : Float)
    (numKVHeads maxSeqLen headDim : Nat) : BuilderM Unit := do
  let halfDim   := headDim / 2
  let kvDim     := numKVHeads * headDim
  let cacheSize := numKVHeads * maxSeqLen * headDim
  declareExternal kNewId        #[kvDim]     .f32 .Global
  declareExternal freqFactorsId #[halfDim]   .f32 .Global
  declareExternal vNewId        #[kvDim]     .f32 .Global
  declareExternal kCacheId      #[cacheSize] .f32 .Global
  declareExternal vCacheId      #[cacheSize] .f32 .Global
  let kR      : Region := { tensorId := kNewId }
  let freqR   : Region := { tensorId := freqFactorsId }
  let vR      : Region := { tensorId := vNewId }
  let kCacheR : Region := { tensorId := kCacheId }
  let vCacheR : Region := { tensorId := vCacheId }
  -- Per-lane integer decomposition.
  let laneF  : ScalarExp := .toFloat .laneIdx
  let hdF    : ScalarExp := .const headDim.toFloat
  let halfF  : ScalarExp := .const halfDim.toFloat
  let kvHead : ScalarExp := .idiv laneF hdF
  let dim_   : ScalarExp := .mod  laneF hdF
  let dInLow : ScalarExp := .lt dim_ halfF
  let dPair  : ScalarExp :=
    .select dInLow (.add dim_ halfF) (.sub dim_ halfF)
  let pairIdx : ScalarExp :=
    .add (.mul kvHead hdF) dPair
  let dimPair : ScalarExp :=
    .select dInLow dim_ (.sub dim_ halfF)
  let xSelf  : ScalarExp := .input 0                 -- slot 0 = new_k[laneIdx]
  let xPair  : ScalarExp := .indexed 0 pairIdx        -- gather from new_k
  let x0 : ScalarExp := .select dInLow xSelf xPair
  let x1 : ScalarExp := .select dInLow xPair xSelf
  let freqFactor : ScalarExp := .indexed 1 dimPair   -- gather from freq_factors
  let exponent : ScalarExp :=
    .div (.mul (.const 2.0) dimPair) hdF
  let freqInv : ScalarExp :=
    .pow (.const ropeBase) (.neg exponent)
  let theta : ScalarExp :=
    .div (.mul (.const pos.toFloat) freqInv) freqFactor
  let cosT : ScalarExp := .cos theta
  let sinT : ScalarExp := .sin theta
  let x0New : ScalarExp :=
    .sub (.mul x0 cosT) (.mul x1 sinT)
  let x1New : ScalarExp :=
    .add (.mul x0 sinT) (.mul x1 cosT)
  let kNew : ScalarExp := .select dInLow x0New x1New
  -- V value: plain gather from new_v at laneIdx (= reads[2]).
  let vVal : ScalarExp := .indexed 2 laneF
  -- cacheIdx: same formula for K and V.
  let stride    : Float := (maxSeqLen * headDim).toFloat
  let posOffset : Float := (pos * headDim).toFloat
  let cacheIdx : ScalarExp :=
    .add (.mul kvHead (.const stride))
         (.add (.const posOffset) dim_)
  emitBlock
    { reads  := #[kR, freqR, vR]
      writes := #[kCacheR, vCacheR]
      body   := .ScatterMulti #[(cacheIdx, kNew), (cacheIdx, vVal)] }

/-- Build the RoPE-Q (query-side NeoX rotation) sub-graph.

    Parity target: `ropeWithFreqFactorsKernel` from
    `Hesper.Models.Gemma4.Kernels`.  The reference kernel uses a
    half-width dispatch (`numHeads * halfDim` lanes, each writing 2
    paired outputs); the IRv2 expression uses a full-width dispatch
    (`numHeads * headDim` lanes, each writing 1 element).  Both evaluate
    the same per-output formula with the same operand ordering, so
    fp32 parity is bit-identical.

    Single `Scatter` block:

        reads  = [new_q, freq_factors]
        writes = [q_out]
        indexExpr = .laneIdx       -- destination == source position (per-lane)
        applyBody = NeoX-rotated Q using freq_factors (same as B7 K-write,
                    but with q's layout: [numHeads, headDim], no cache stride)

    Caller must allocate a separate output buffer — this is NOT safe to
    run in-place, because the per-lane formula reads both xSelf AND xPair
    from `new_q`; mutating the input mid-dispatch would corrupt the pair
    read. -/
def buildRopeQLazy
    (qNewId freqFactorsId qOutId : Nat)
    (pos : Nat) (ropeBase : Float)
    (numHeads headDim : Nat) : BuilderM Unit := do
  let halfDim := headDim / 2
  let qDim    := numHeads * headDim
  declareExternal qNewId        #[qDim]    .f32 .Global
  declareExternal freqFactorsId #[halfDim] .f32 .Global
  declareExternal qOutId        #[qDim]    .f32 .Global
  let qR     : Region := { tensorId := qNewId }
  let freqR  : Region := { tensorId := freqFactorsId }
  let qOutR  : Region := { tensorId := qOutId }
  -- Per-lane integer decomposition (head/d).
  let laneF  : ScalarExp := .toFloat .laneIdx
  let hdF    : ScalarExp := .const headDim.toFloat
  let halfF  : ScalarExp := .const halfDim.toFloat
  let head   : ScalarExp := .idiv laneF hdF
  let dim_   : ScalarExp := .mod  laneF hdF
  let dInLow : ScalarExp := .lt dim_ halfF
  -- Partner lane id (in the flat Q layout).
  let dPair  : ScalarExp :=
    .select dInLow (.add dim_ halfF) (.sub dim_ halfF)
  let pairIdx : ScalarExp :=
    .add (.mul head hdF) dPair
  let dimPair : ScalarExp :=
    .select dInLow dim_ (.sub dim_ halfF)
  -- xSelf = new_q[laneIdx] (pre-read slot 0); xPair gathered from new_q.
  let xSelf  : ScalarExp := .input 0
  let xPair  : ScalarExp := .indexed 0 pairIdx
  let x0 : ScalarExp := .select dInLow xSelf xPair
  let x1 : ScalarExp := .select dInLow xPair xSelf
  let freqFactor : ScalarExp := .indexed 1 dimPair
  let exponent : ScalarExp :=
    .div (.mul (.const 2.0) dimPair) hdF
  let freqInv : ScalarExp :=
    .pow (.const ropeBase) (.neg exponent)
  let theta : ScalarExp :=
    .div (.mul (.const pos.toFloat) freqInv) freqFactor
  let cosT : ScalarExp := .cos theta
  let sinT : ScalarExp := .sin theta
  let x0New : ScalarExp := .sub (.mul x0 cosT) (.mul x1 sinT)
  let x1New : ScalarExp := .add (.mul x0 sinT) (.mul x1 cosT)
  let qNew  : ScalarExp := .select dInLow x0New x1New
  -- Destination index = laneIdx (in-place Q layout — each lane writes
  -- its own position in the Q output buffer).
  emitBlock
    { reads  := #[qR, freqR]
      writes := #[qOutR]
      body   := .Scatter laneF qNew }

/-! ## Full-layer composition

Combines every sub-builder into one `BuilderM` program for a single
Gemma 4 layer's forward pass.  IR constructs we haven't yet covered
(per-head QKV RMSNorm, Flash Attention) are emitted as placeholder
`Pointwise` blocks — the analyzer counts them as 1 dispatch each,
matching production's physical-kernel count for those ops. -/

/-- Shape parameters for a Gemma 4 decode layer (decode path, fused peak).
    Provide these as `Nat` constants (not a `Config`) so this module
    stays free of model-config dependency. -/
structure LayerShapes where
  dim        : Nat  -- hiddenSize
  numHeads   : Nat
  numKVHeads : Nat
  headDim    : Nat
  maxSeqLen  : Nat
  interDim   : Nat  -- FFN intermediate size
  eps        : Float
  ropeBase   : Float
  deriving Repr, Inhabited

/-- Allocate a fresh Register-scope tensor of `shape` f32. -/
private def tmp (shape : Shape) : BuilderM Nat := do
  let t ← declareTensor shape .f32 .Register
  return t.id

/-- Emit a placeholder Pointwise block to stand in for an op the IR
    doesn't yet express (qkvNorm, Flash Attention).  Counted by the
    analyzer as 1 dispatch; the Lean compiler's BlockGraph walk just
    sees it as an opaque 1-in-1-out transform. -/
private def placeholderPointwise
    (reads : Array Region) (writes : Array Region) : BuilderM Unit := do
  emitBlock
    { reads, writes
      body := .Pointwise (.input 0) }

/-- End-to-end single-layer forward as a `BlockGraph`.

    Block sequence (numbers match the pattern names in the dispatcher):
      1.  [Reduce; MatMul×3]      -- Pattern A  (attnNorm + wQKV)
      2.  [Pointwise]             -- Placeholder: per-head qkvNorm
      3.  [Scatter]               -- Pattern F  (RoPE-Q)
      4.  [ScatterMulti]          -- Pattern F  (K+V cache write with RoPE-K)
      5.  [Pointwise]             -- Placeholder: flash-attention
      6.  [MatMul]                -- Pattern C  (wO)
      7.  [Reduce with +residual] -- Pattern E  (postAttnNorm + add)
      8.  [Reduce; MatMul×2; Pointwise; MatMul]  -- Pattern D  (FFN)
      9.  [Reduce with +residual] -- Pattern E  (postFFNNorm + add)

    Caller wires external tensor ids (input / ffn-residual / layer output / …)
    and `LayerShapes`; buffer resolution happens in the dispatcher. -/
def forwardLayerLazy
    (s : LayerShapes)
    (inputId attnNormScaleId
     qkvMidTmpQId qkvMidTmpKId qkvMidTmpVId
     qkvNormQId qkvNormKId qkvNormVId
     qRotId
     freqFactorsId kCacheId vCacheId
     attnOutId wOOutId postAttnScaleId attnResidualId
     ffnNormScaleId ffnOutId postFFNScaleId layerOutId : Nat)
    (wQKey wKKey wVKey wOKey
     ffnGateKey ffnUpKey ffnDownKey : UInt64)
    (pos : Nat) : BuilderM Unit := do
  let qDim  := s.numHeads   * s.headDim
  let kvDim := s.numKVHeads * s.headDim

  -- (1) attnNorm + wQ/wK/wV — 4 blocks → Pattern A (3 dispatches).
  buildNormQKVProjLazy inputId attnNormScaleId
    qkvMidTmpQId qkvMidTmpKId qkvMidTmpVId
    wQKey wKKey wVKey
    /-normKey-/ 0xDEAD0001  -- synthetic; caller registers in normHandles
    s.dim qDim kvDim s.eps

  -- (2) Per-head QKV norm — placeholder (one Pointwise block).
  placeholderPointwise
    #[{ tensorId := qkvMidTmpQId }, { tensorId := qkvMidTmpKId }, { tensorId := qkvMidTmpVId }]
    #[{ tensorId := qkvNormQId },   { tensorId := qkvNormKId },   { tensorId := qkvNormVId }]

  -- (3) RoPE-Q: 1-block Scatter → 1 dispatch.
  buildRopeQLazy qkvNormQId freqFactorsId qRotId pos s.ropeBase s.numHeads s.headDim

  -- (4) RoPE-K + V cache write: 1-block ScatterMulti → 1 dispatch.
  buildRopeKVWriteLazy qkvNormKId freqFactorsId qkvNormVId kCacheId vCacheId
    pos s.ropeBase s.numKVHeads s.maxSeqLen s.headDim

  -- (5) Flash-attention — placeholder (1 Pointwise).
  placeholderPointwise
    #[{ tensorId := qRotId }, { tensorId := kCacheId }, { tensorId := vCacheId }]
    #[{ tensorId := attnOutId }]

  -- (6) wO projection: single MatMul → Pattern C (2 dispatches).
  buildQProjLazy attnOutId wOOutId wOKey qDim s.dim

  -- (7) Post-attention norm + residual → Pattern E (1 dispatch).
  buildPostFFNLazy wOOutId postAttnScaleId inputId attnResidualId s.dim s.eps

  -- (8) FFN body (RMSNorm + gate+up+GELU×mul + wDown) → Pattern D (2 dispatches).
  buildFFNLazy attnResidualId ffnNormScaleId ffnOutId
    ffnGateKey ffnUpKey ffnDownKey
    /-normKey-/ 0xDEAD0008 /-geluKey-/ 0xDEAD0009
    s.dim s.interDim s.eps

  -- (9) Post-FFN norm + residual → Pattern E (1 dispatch).
  buildPostFFNLazy ffnOutId postFFNScaleId attnResidualId layerOutId s.dim s.eps

/-! ## Naive full-layer composition (pre-fusion ground truth)

`forwardLayerLazyNaive` emits every small step hesper v1 actually
issues as its own block — no hand-fusion pre-applied.  The point is
to show what the IRv2 fusion passes can *automatically* absorb.

Each "main-pile" helper (layerOutScale, pleScale, residualAdd, etc.)
appears as a standalone Pointwise block.  The graph is deliberately
ugly — after running `fusePointwiseIntoReduce` / `fusePointwiseIntoMatMul`
adjacent Pointwise tails get folded into the upstream Reduce or MatMul
block, mirroring what hesper's hand-fused kernels do manually.

Block layout (before fusion):
  1.   Reduce (attnNorm)
  2-4. MatMul wQ, wK, wV                       ← Pattern A absorbs 2-4
  5.   Pointwise placeholder (qkvNorm)         ← not yet fusible
  6.   Scatter (RoPE-Q)                        ← Pattern F
  7.   ScatterMulti (K+V cache)                ← Pattern F (multi)
  8.   Pointwise placeholder (flashAttn)       ← not yet fusible
  9.   MatMul (wO)
  10.  Pointwise (layerOutScale on wO out)     ← absorbable into 9 epilogue
  11.  Reduce (postAttnNorm)
  12.  Pointwise (residualAdd postAttn)        ← absorbable into 11 epilogue
  13.  Reduce (ffnNorm)                        ← Pattern D absorbs 13-16
  14.  MatMul wGate
  15.  MatMul wUp
  16.  Pointwise (GELU*up)
  17.  MatMul wDown
  18.  Pointwise (layerOutScale on ffnOut)     ← absorbable into 17 epilogue
  19.  Reduce (postFFNNorm)
  20.  Pointwise (residualAdd postFFN)         ← absorbable into 19 epilogue
  21.  MatMul (PLE inpGate)                    ← PLE chain
  22.  Pointwise (GELU*pl_input slice)         ← absorbable into 21 epilogue
  23.  MatMul (PLE proj)
  24.  Reduce (PLE postNorm)
  25.  Pointwise (PLE residualAdd)             ← absorbable into 24 epilogue

Total unfused: 25 blocks.  After ideal fusion the block count drops
to ~9 (Pattern A + placeholders + Pattern D + 4×Pattern E + Pattern C). -/
/-- Reserve all internal-tensor ids above `anchorId` by declaring a
    dummy external tensor at `anchorId`.  This forces all subsequent
    `declareTensor` calls to allocate above `anchorId + 1`, eliminating
    collisions with caller-chosen external ids below `anchorId`. -/
def reserveInternalRange (anchorId : Nat) : BuilderM Unit :=
  declareExternal anchorId #[1] .f32 .Register

def forwardLayerLazyNaive
    (s : LayerShapes)
    (inputId attnNormScaleId
     qkvMidTmpQId qkvMidTmpKId qkvMidTmpVId
     qkvNormQId qkvNormKId qkvNormVId
     qRotId
     freqFactorsId kCacheId vCacheId
     attnOutId wOOutPreId wOOutId postAttnScaleId attnResidualId
     ffnNormScaleId ffnOutPreId ffnOutId postFFNScaleId postFFNOutId
     pleInputId pleGateOutId pleProjOutId pleScaleId pleResidOutId layerOutId
     layerScaleId : Nat)
    (wQKey wKKey wVKey wOKey
     ffnGateKey ffnUpKey ffnDownKey
     pleGateKey pleProjKey : UInt64)
    (pos : Nat) : BuilderM Unit := do
  let qDim  := s.numHeads   * s.headDim
  let kvDim := s.numKVHeads * s.headDim
  -- Reserve id range: internal `declareTensor` calls below will start
  -- at the highest external id we touch (layerScaleId).  Bump the
  -- builder's nextId past every caller-supplied id so internal tensors
  -- don't collide with external regions.
  reserveInternalRange (layerScaleId + 1)
  -- (0) attnNorm produces a single f32 vector → attnNormOutId
  declareExternal inputId          #[s.dim]  .f32 .Global
  declareExternal attnNormScaleId  #[s.dim]  .f32 .Global
  let attnNormOutT ← declareTensor #[s.dim]  .f32 .Register
  let invRms_dim : ScalarExp :=
    .rsqrt (.add (.div (.input 0) (.const s.dim.toFloat)) (.const s.eps))
  let normApply_dim : ScalarExp :=
    .mul (.mul (.input 1) (.input 2)) invRms_dim
  emitBlock
    { reads := #[{ tensorId := inputId }, { tensorId := attnNormScaleId }]
      writes := #[{ tensorId := attnNormOutT.id }]
      body := .Reduce ReduceOp.sumOfSquares { tensorId := inputId } 0 normApply_dim }
  -- (1) Quantize attnNorm output (3 explicit Quantize blocks, each
  --     reading the same source — CSE will collapse them to 1).
  let attnQ8a ← declareTensor #[s.dim] .f32 .Register   -- pretend Q8_1 buf
  let attnQ8b ← declareTensor #[s.dim] .f32 .Register
  let attnQ8c ← declareTensor #[s.dim] .f32 .Register
  emitBlock
    { reads := #[{ tensorId := attnNormOutT.id }]
      writes := #[{ tensorId := attnQ8a.id }]
      body := .Quantize }
  emitBlock
    { reads := #[{ tensorId := attnNormOutT.id }]
      writes := #[{ tensorId := attnQ8b.id }]
      body := .Quantize }
  emitBlock
    { reads := #[{ tensorId := attnNormOutT.id }]
      writes := #[{ tensorId := attnQ8c.id }]
      body := .Quantize }
  -- (2-4) wQ/wK/wV reading the (post-CSE shared) Q8_1 buffer.
  declareExternal qkvMidTmpQId #[qDim]  .f32 .Global
  declareExternal qkvMidTmpKId #[kvDim] .f32 .Global
  declareExternal qkvMidTmpVId #[kvDim] .f32 .Global
  emitBlock
    { reads  := #[{ tensorId := attnQ8a.id }]
      writes := #[{ tensorId := qkvMidTmpQId }]
      body   := .MatMul wQKey qDim s.dim (.input 0) }
  emitBlock
    { reads  := #[{ tensorId := attnQ8b.id }]
      writes := #[{ tensorId := qkvMidTmpKId }]
      body   := .MatMul wKKey kvDim s.dim (.input 0) }
  emitBlock
    { reads  := #[{ tensorId := attnQ8c.id }]
      writes := #[{ tensorId := qkvMidTmpVId }]
      body   := .MatMul wVKey kvDim s.dim (.input 0) }
  -- (5) qkvNorm placeholder
  placeholderPointwise
    #[{ tensorId := qkvMidTmpQId }, { tensorId := qkvMidTmpKId }, { tensorId := qkvMidTmpVId }]
    #[{ tensorId := qkvNormQId },   { tensorId := qkvNormKId },   { tensorId := qkvNormVId }]
  -- (6) RoPE-Q = Pattern F
  buildRopeQLazy qkvNormQId freqFactorsId qRotId pos s.ropeBase s.numHeads s.headDim
  -- (7) K+V cache write = Pattern F (multi)
  buildRopeKVWriteLazy qkvNormKId freqFactorsId qkvNormVId kCacheId vCacheId
    pos s.ropeBase s.numKVHeads s.maxSeqLen s.headDim
  -- (8) flashAttn placeholder
  placeholderPointwise
    #[{ tensorId := qRotId }, { tensorId := kCacheId }, { tensorId := vCacheId }]
    #[{ tensorId := attnOutId }]
  -- (9) wO: explicit Quantize on attnOut, then MatMul.
  let attnOutQ8 ← declareTensor #[qDim] .f32 .Register
  emitBlock
    { reads  := #[{ tensorId := attnOutId }]
      writes := #[{ tensorId := attnOutQ8.id }]
      body   := .Quantize }
  emitBlock
    { reads  := #[{ tensorId := attnOutQ8.id }]
      writes := #[{ tensorId := wOOutPreId }]
      body   := .MatMul wOKey s.dim qDim (.input 0) }
  -- (10) layerOutScale on wO output: out = wOPre * scale_attn
  --      Pointwise body reads wOPre (slot 0) and layerScale (slot 1).
  --      After fusion, collapses into block 9's epilogue.
  emitBlock
    { reads  := #[{ tensorId := wOOutPreId }, { tensorId := layerScaleId }]
      writes := #[{ tensorId := wOOutId }]
      body   := .Pointwise (.mul (.input 0) (.input 1)) }
  -- (11) postAttnNorm (the ugly version: separate Reduce producing a
  --     normed buffer — the hand-tuned code fuses this with the add).
  declareExternal postAttnScaleId #[s.dim] .f32 .Global
  declareExternal attnResidualId  #[s.dim] .f32 .Global
  let invRms : ScalarExp :=
    .rsqrt (.add (.div (.input 0) (.const s.dim.toFloat)) (.const s.eps))
  let normApply : ScalarExp :=
    .mul (.mul (.input 1) (.input 2)) invRms
  -- Intermediate normed buffer (pre-residual-add) — register-scope.
  let normedAttn ← declareTensor #[s.dim] .f32 .Register
  emitBlock
    { reads := #[{ tensorId := wOOutId }, { tensorId := postAttnScaleId }]
      writes := #[{ tensorId := normedAttn.id }]
      body := .Reduce ReduceOp.sumOfSquares { tensorId := wOOutId } 0 normApply }
  -- (12) residualAdd: attnResidual = normed + inputBuf
  --     Body: .input 0 + .input 1.  Fusion with block 11's Reduce
  --     would turn it into "Reduce with +residual" (Pattern E shape).
  emitBlock
    { reads  := #[{ tensorId := normedAttn.id }, { tensorId := inputId }]
      writes := #[{ tensorId := attnResidualId }]
      body   := .Pointwise (.add (.input 0) (.input 1)) }
  -- (13-17) FFN expanded by hand so we can insert Quantize blocks
  --     for both gate+up (which share ffnNormOut → CSE collapses them)
  --     and for wDown (independent).
  declareExternal ffnNormScaleId #[s.dim] .f32 .Global
  let ffnNormOutT ← declareTensor #[s.dim] .f32 .Register
  emitBlock
    { reads := #[{ tensorId := attnResidualId }, { tensorId := ffnNormScaleId }]
      writes := #[{ tensorId := ffnNormOutT.id }]
      body := .Reduce ReduceOp.sumOfSquares
                { tensorId := attnResidualId } 0 normApply_dim }
  -- gate+up Quantize ×2 (CSE collapses to 1)
  let ffnQ8gate ← declareTensor #[s.dim] .f32 .Register
  let ffnQ8up   ← declareTensor #[s.dim] .f32 .Register
  emitBlock
    { reads := #[{ tensorId := ffnNormOutT.id }]
      writes := #[{ tensorId := ffnQ8gate.id }]
      body := .Quantize }
  emitBlock
    { reads := #[{ tensorId := ffnNormOutT.id }]
      writes := #[{ tensorId := ffnQ8up.id }]
      body := .Quantize }
  let gateOut ← declareTensor #[s.interDim] .f32 .Register
  let upOut   ← declareTensor #[s.interDim] .f32 .Register
  emitBlock
    { reads  := #[{ tensorId := ffnQ8gate.id }]
      writes := #[{ tensorId := gateOut.id }]
      body   := .MatMul ffnGateKey s.interDim s.dim (.input 0) }
  emitBlock
    { reads  := #[{ tensorId := ffnQ8up.id }]
      writes := #[{ tensorId := upOut.id }]
      body   := .MatMul ffnUpKey s.interDim s.dim (.input 0) }
  -- GELU × up
  let geluOut ← declareTensor #[s.interDim] .f32 .Register
  emitBlock
    { reads  := #[{ tensorId := gateOut.id }, { tensorId := upOut.id }]
      writes := #[{ tensorId := geluOut.id }]
      body   := .Pointwise (.mul (.gelu (.input 0)) (.input 1)) }
  -- wDown: independent Quantize + MatMul
  declareExternal ffnOutPreId #[s.dim] .f32 .Global
  let geluQ8 ← declareTensor #[s.interDim] .f32 .Register
  emitBlock
    { reads := #[{ tensorId := geluOut.id }]
      writes := #[{ tensorId := geluQ8.id }]
      body := .Quantize }
  emitBlock
    { reads  := #[{ tensorId := geluQ8.id }]
      writes := #[{ tensorId := ffnOutPreId }]
      body   := .MatMul ffnDownKey s.dim s.interDim (.input 0) }
  -- (18) layerOutScale on ffnOut
  declareExternal ffnOutId #[s.dim] .f32 .Global
  emitBlock
    { reads  := #[{ tensorId := ffnOutPreId }, { tensorId := layerScaleId }]
      writes := #[{ tensorId := ffnOutId }]
      body   := .Pointwise (.mul (.input 0) (.input 1)) }
  -- (19) postFFNNorm split again into naive Reduce.
  declareExternal postFFNScaleId #[s.dim] .f32 .Global
  declareExternal postFFNOutId   #[s.dim] .f32 .Global
  let normedFFN ← declareTensor #[s.dim] .f32 .Register
  emitBlock
    { reads  := #[{ tensorId := ffnOutId }, { tensorId := postFFNScaleId }]
      writes := #[{ tensorId := normedFFN.id }]
      body   := .Reduce ReduceOp.sumOfSquares { tensorId := ffnOutId } 0 normApply }
  -- (20) residualAdd postFFN
  emitBlock
    { reads  := #[{ tensorId := normedFFN.id }, { tensorId := attnResidualId }]
      writes := #[{ tensorId := postFFNOutId }]
      body   := .Pointwise (.add (.input 0) (.input 1)) }
  -- (21) PLE inpGate: explicit Quantize on the post-FFN output (which
  --      IS the PLE input — they're the same buffer in production).
  --      Reusing `postFFNOutId` instead of a fresh `pleInputId` lets
  --      `fuseReduceIntoQuantize` see the [Reduce(postFFN); Quantize]
  --      adjacency and fold them into one ReduceQuantize.
  declareExternal pleGateOutId  #[s.interDim] .f32 .Global
  let _ := pleInputId  -- alias intent; left as caller-supplied id
  let pleInpQ8 ← declareTensor #[s.dim] .f32 .Register
  emitBlock
    { reads := #[{ tensorId := postFFNOutId }]
      writes := #[{ tensorId := pleInpQ8.id }]
      body := .Quantize }
  emitBlock
    { reads  := #[{ tensorId := pleInpQ8.id }]
      writes := #[{ tensorId := pleGateOutId }]
      body   := .MatMul pleGateKey s.interDim s.dim (.input 0) }
  -- (22) PLE GELU × per_layer_input slice — fusible into 21's epilogue.
  declareExternal pleScaleId   #[s.interDim] .f32 .Global
  declareExternal pleProjOutId #[s.dim]      .f32 .Global
  emitBlock
    { reads  := #[{ tensorId := pleGateOutId }, { tensorId := pleScaleId }]
      writes := #[{ tensorId := pleProjOutId }]
      body   := .Pointwise (.mul (.gelu (.input 0)) (.input 1)) }
  -- (23) PLE proj: explicit Quantize on pleProjOutId (the GELU×slice
  --      output), then MatMul.
  declareExternal pleResidOutId #[s.dim] .f32 .Global
  let plePrjQ8 ← declareTensor #[s.interDim] .f32 .Register
  emitBlock
    { reads := #[{ tensorId := pleProjOutId }]
      writes := #[{ tensorId := plePrjQ8.id }]
      body := .Quantize }
  emitBlock
    { reads  := #[{ tensorId := plePrjQ8.id }]
      writes := #[{ tensorId := pleResidOutId }]
      body   := .MatMul pleProjKey s.dim s.interDim (.input 0) }
  -- (24) PLE postNorm (naive Reduce)
  declareExternal layerOutId #[s.dim] .f32 .Global
  let normedPLE ← declareTensor #[s.dim] .f32 .Register
  emitBlock
    { reads  := #[{ tensorId := pleResidOutId }, { tensorId := postFFNScaleId }]
      writes := #[{ tensorId := normedPLE.id }]
      body   := .Reduce ReduceOp.sumOfSquares
                  { tensorId := pleResidOutId } 0 normApply }
  -- (25) PLE residualAdd (post-PLE output = normed + postFFNOut)
  emitBlock
    { reads  := #[{ tensorId := normedPLE.id }, { tensorId := postFFNOutId }]
      writes := #[{ tensorId := layerOutId }]
      body   := .Pointwise (.add (.input 0) (.input 1)) }

/-! ## Phase D: Monolith-based layer

Drastically simpler than `forwardLayerLazyNaive`: each layer is just
4 logical AST nodes.  The dispatcher (`Dispatch_v2.runMonolith`)
expands them into the production hand-fused sequences (B3/B4/B7/B8/B9
parity-proven kernels).

```
forwardLayerLazyMonolith key pos =
  emit GemmaAttentionMonolith key pos      -- ~6 physical dispatches
  emit FlashAttention         key pos      -- ~2 physical dispatches
  emit GemmaFFNMonolith       key          -- ~3 physical dispatches
  emit PostFFNNormAdd         key          -- ~1 physical dispatch
```

Logical: 4 blocks/layer.  Physical: ~12 dispatches/layer (matches v1). -/
def forwardLayerLazyMonolith (layerKey : UInt64) (pos : Nat)
    (inputId qBufId kBufId vBufId attnOutId wOOutId attnResidId ffnOutId postFFNOutId : Nat) :
    BuilderM Unit := do
  declareExternal inputId      #[1] .f32 .Global
  declareExternal qBufId       #[1] .f32 .Global
  declareExternal kBufId       #[1] .f32 .Global
  declareExternal vBufId       #[1] .f32 .Global
  declareExternal attnOutId    #[1] .f32 .Global
  declareExternal wOOutId      #[1] .f32 .Global
  declareExternal attnResidId  #[1] .f32 .Global
  declareExternal ffnOutId     #[1] .f32 .Global
  declareExternal postFFNOutId #[1] .f32 .Global
  -- 1. attention prologue → qBuf, kBuf, vBuf (post-RoPE)
  emitBlock
    { reads  := #[{ tensorId := inputId }]
      writes := #[{ tensorId := qBufId }, { tensorId := kBufId }, { tensorId := vBufId }]
      body   := .GemmaAttentionMonolith layerKey pos }
  -- 2. flash attention → attnOut
  emitBlock
    { reads  := #[{ tensorId := qBufId }, { tensorId := kBufId }, { tensorId := vBufId }]
      writes := #[{ tensorId := attnOutId }]
      body   := .FlashAttention layerKey pos }
  -- 3. wO projection → wOOut (raw projection before norm)
  emitBlock
    { reads  := #[{ tensorId := attnOutId }]
      writes := #[{ tensorId := wOOutId }]
      body   := .GemmaAttnOutProj layerKey }
  -- 4. postAttnNorm + residual → attnResid (= wOOut_normed + inputBuf)
  emitBlock
    { reads  := #[{ tensorId := wOOutId }, { tensorId := inputId }]
      writes := #[{ tensorId := attnResidId }]
      body   := .PostAttnNormAdd layerKey }
  -- 5. FFN body → ffnOut
  emitBlock
    { reads  := #[{ tensorId := attnResidId }]
      writes := #[{ tensorId := ffnOutId }]
      body   := .GemmaFFNMonolith layerKey }
  -- 6. postFFNNorm + residual (+ attnResid) → postFFNOut
  emitBlock
    { reads  := #[{ tensorId := ffnOutId }, { tensorId := attnResidId }]
      writes := #[{ tensorId := postFFNOutId }]
      body   := .PostFFNNormAdd layerKey }

/-! ## Phase E: whole-token BlockGraph

Builds an entire forward pass (all `numLayers` layers) as one
BlockGraph.  Each layer is the 4-block Monolith produced by
`forwardLayerLazyMonolith`; consecutive layers share buffers
(layer i's output = layer i+1's input).  The dispatcher can replay
the whole graph through CUDA Graphs as ONE host-side launch. -/

/-- Per-layer external tensor ids derived from a base.  Each layer
    needs 7 external tensor ids (input, q, k, v, attnOut, ffnOut,
    postFFNOut).  Layer i uses ids `[base + 7*i .. base + 7*i + 6]`.
    Layer i+1's input == layer i's postFFNOut, achieved by reusing
    the right id explicitly in `forwardTokenLazyMonolith`. -/
def forwardTokenLazyMonolith
    (numLayers : Nat) (firstInputId : Nat) (lastOutputId : Nat)
    (baseTensorId : Nat) (firstLayerKey : UInt64) (pos : Nat) :
    BuilderM Unit := do
  -- 9 ids per layer: q, k, v, attnOut, wOOut, attnResid, ffnOut,
  -- postFFNOut (+1 reserved).
  let perLayer := 9
  reserveInternalRange (baseTensorId + numLayers * perLayer + 16)
  for li in [0:numLayers] do
    let slot := baseTensorId + li * perLayer
    let qId         := slot + 0
    let kId         := slot + 1
    let vId         := slot + 2
    let attnOutId   := slot + 3
    let wOOutId     := slot + 4
    let attnResidId := slot + 5
    let ffnOutId    := slot + 6
    -- Input is the previous layer's postFFNOut (or `firstInputId` for layer 0).
    let inId :=
      if li == 0 then firstInputId
      else baseTensorId + (li - 1) * perLayer + 7
    -- postFFNOut: last layer writes `lastOutputId`, others write slot+7.
    let outId :=
      if li + 1 == numLayers then lastOutputId
      else slot + 7
    let layerKey := firstLayerKey + li.toUInt64
    forwardLayerLazyMonolith layerKey pos
      inId qId kId vId attnOutId wOOutId attnResidId ffnOutId outId

end Hesper.Models.Gemma4_v2
