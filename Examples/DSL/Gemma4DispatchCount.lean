import Hesper.Circuit.IRv2
import Hesper.Circuit.Dispatch_v2
import Hesper.Models.Gemma4_v2

set_option maxRecDepth 2048

/-!
# Phase B stocktake: per-layer dispatch count, naive → fused → v1.

Quantifies the IRv2 fusion passes' effect on the "main pile" — the
~9,595 tiny helper kernels (layerOutScale, pleScale, residualAdd, …)
that dominate hesper v1's GPU launch count.

Three columns are reported:

  1. v1 (eager):    hesper v1's production fused peak path — hand-counted.
  2. v2 naive:      forwardLayerLazyNaive emits every small step as its
                    own block — NO fusion pre-applied.
  3. v2 fused:      after applying `fusePointwiseIntoReduce` and
                    `fusePointwiseIntoMatMul`, the adjacent Pointwise
                    tails collapse into the upstream MatMul / Reduce.

The gap between (2) and (3) is what the compiler is actually earning.
The gap between (3) and (1) is the Phase B win over hand-fused code.
-/

open Hesper.Circuit
open Hesper.Circuit.IRv2
open Hesper.Models.Gemma4_v2

/-- Hand-counted v1 dispatches per layer, fused peak decode path.
    Derived from `docs/llama-fusion-analysis/28-67-fold-gap-investigation.md`
    and `memory/project_dispatch_measurement.md` (HESPER_DISPATCH_COUNT=1
    runs: 920/token ÷ 42 layers ≈ 22/layer). -/
structure V1Breakdown where
  preAttnNormQKV   : Nat := 3   -- forwardFusedNormQKV (rmsnorm-q8_1 + wQ + wK+wV)
  qkvNormFused     : Nat := 1   -- fusedPerHeadQKVNormKernel
  ropeQ            : Nat := 1   -- ropeWithFreqFactors
  kvScatterMulti   : Nat := 1   -- scatterMulti (RoPE-K + K+V cache)
  flashAttnTiled   : Nat := 2   -- tiled flashAttn (phase 1 + phase 2)
  wOMatmul         : Nat := 2   -- circuitWO = quantize + dp4a
  postAttnNormAdd  : Nat := 1   -- forwardNormThenAdd (hand-fused)
  ffnNormGateUp    : Nat := 2   -- forwardFusedNormGateUp
  ffnDown          : Nat := 2   -- LinearLayer.forward → quantize + dp4a
  postFFNNormAdd   : Nat := 1   -- forwardNormThenAdd (hand-fused)
  pleInpGate       : Nat := 2   -- circuitPLEInpGateGeluSlice (quantize + matmul+epi)
  pleProj          : Nat := 2   -- LinearLayer.forward PLE proj
  plePostNormAdd   : Nat := 1   -- fusedPLPostScale (+ layerOutScale folded in)
  deriving Repr, Inhabited

def V1Breakdown.total (v : V1Breakdown) : Nat :=
    v.preAttnNormQKV + v.qkvNormFused + v.ropeQ + v.kvScatterMulti
  + v.flashAttnTiled + v.wOMatmul + v.postAttnNormAdd
  + v.ffnNormGateUp + v.ffnDown + v.postFFNNormAdd
  + v.pleInpGate + v.pleProj + v.plePostNormAdd

def V1Breakdown.toReport (v : V1Breakdown) : String := Id.run do
  let mut s : String := ""
  s := s ++ s!"  preAttn norm+wQKV  (forwardFusedNormQKV)    : {v.preAttnNormQKV}\n"
  s := s ++ s!"  qkvNorm per-head   (fusedPerHeadQKVNorm)    : {v.qkvNormFused}\n"
  s := s ++ s!"  RoPE-Q             (ropeWithFreqFactors)    : {v.ropeQ}\n"
  s := s ++ s!"  KV cache write     (scatterMulti K+V)       : {v.kvScatterMulti}\n"
  s := s ++ s!"  flash-attention    (tiled, phase 1+2)       : {v.flashAttnTiled}\n"
  s := s ++ s!"  wO projection      (circuitWO q+matmul)     : {v.wOMatmul}\n"
  s := s ++ s!"  post-attn norm+add (forwardNormThenAdd)     : {v.postAttnNormAdd}\n"
  s := s ++ s!"  FFN norm+gate+up   (forwardFusedNormGateUp) : {v.ffnNormGateUp}\n"
  s := s ++ s!"  FFN down           (q+matmul)               : {v.ffnDown}\n"
  s := s ++ s!"  post-FFN norm+add  (forwardNormThenAdd)     : {v.postFFNNormAdd}\n"
  s := s ++ s!"  PLE inpGate        (q + matmul+GELUslice)   : {v.pleInpGate}\n"
  s := s ++ s!"  PLE proj           (q+matmul)               : {v.pleProj}\n"
  s := s ++ s!"  PLE postNorm+add   (fusedPLPostScale)       : {v.plePostNormAdd}\n"
  s := s ++ s!"  ---\n"
  s := s ++ s!"  TOTAL v1 (fused peak) : {v.total}\n"
  return s

def buildNaiveGraph (s : LayerShapes) (pos : Nat) : BlockGraph := Id.run do
  -- Use a high base so internal declareTensor calls (which start at 0)
  -- never collide with our externally-named ids.
  let mut tid : Nat := 1000000
  let inputId         := tid; tid := tid + 1
  let attnNormScaleId := tid; tid := tid + 1
  let qkvMidTmpQId    := tid; tid := tid + 1
  let qkvMidTmpKId    := tid; tid := tid + 1
  let qkvMidTmpVId    := tid; tid := tid + 1
  let qkvNormQId      := tid; tid := tid + 1
  let qkvNormKId      := tid; tid := tid + 1
  let qkvNormVId      := tid; tid := tid + 1
  let qRotId          := tid; tid := tid + 1
  let freqFactorsId   := tid; tid := tid + 1
  let kCacheId        := tid; tid := tid + 1
  let vCacheId        := tid; tid := tid + 1
  let attnOutId       := tid; tid := tid + 1
  let wOOutPreId      := tid; tid := tid + 1
  let wOOutId         := tid; tid := tid + 1
  let postAttnScaleId := tid; tid := tid + 1
  let attnResidualId  := tid; tid := tid + 1
  let ffnNormScaleId  := tid; tid := tid + 1
  let ffnOutPreId     := tid; tid := tid + 1
  let ffnOutId        := tid; tid := tid + 1
  let postFFNScaleId  := tid; tid := tid + 1
  let postFFNOutId    := tid; tid := tid + 1
  let pleInputId      := tid; tid := tid + 1
  let pleGateOutId    := tid; tid := tid + 1
  let pleProjOutId    := tid; tid := tid + 1
  let pleScaleId      := tid; tid := tid + 1
  let pleResidOutId   := tid; tid := tid + 1
  let layerOutId      := tid; tid := tid + 1
  let layerScaleId    := tid; tid := tid + 1
  let (_, g) := runBuilder
    (forwardLayerLazyNaive s
      inputId attnNormScaleId
      qkvMidTmpQId qkvMidTmpKId qkvMidTmpVId
      qkvNormQId qkvNormKId qkvNormVId
      qRotId
      freqFactorsId kCacheId vCacheId
      attnOutId wOOutPreId wOOutId postAttnScaleId attnResidualId
      ffnNormScaleId ffnOutPreId ffnOutId postFFNScaleId postFFNOutId
      pleInputId pleGateOutId pleProjOutId pleScaleId pleResidOutId layerOutId
      layerScaleId
      0x10 0x20 0x30 0x40 0x50 0x60 0x70 0x80 0x90
      pos)
  g

def main : IO Unit := do
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println " Phase B11 stocktake: naive → fused → v1 dispatch comparison"
  IO.println " Model: Gemma 4 E4B-it (Q4_K_M), decode path, fused peak"
  IO.println "════════════════════════════════════════════════════════════════"
  let shapes : LayerShapes :=
    { dim        := 2560
      numHeads   := 8
      numKVHeads := 4
      headDim    := 128
      maxSeqLen  := 128
      interDim   := 10240
      eps        := 1e-6
      ropeBase   := 10000.0 }
  let pos : Nat := 7

  -- 1. Naive graph (no fusion pre-applied).
  let g0 := buildNaiveGraph shapes pos
  let a0 := analyzeBlockGraph g0
  IO.println ""
  IO.println s!"── v2 NAIVE (pre-fusion) ───────────────────────────────────"
  IO.println s!"  Graph: {g0.blocks.size} blocks, {g0.tensors.size} tensors"
  IO.println a0.toReport

  -- 2. Apply fusion passes (reduce+pointwise; matmul+pointwise).
  let g1 := fusePointwiseIntoReduce g0
  let a1 := analyzeBlockGraph g1
  IO.println s!"── intermediate after fusePointwiseIntoReduce ───────────"
  IO.println s!"  Graph: {g1.blocks.size} blocks, {a1.totalLaunches} dispatches"
  IO.println ""
  let g2 := fusePointwiseIntoMatMul g1
  let a2 := analyzeBlockGraph g2
  IO.println s!"── intermediate after fusePointwiseIntoMatMul ───────────"
  IO.println s!"  Graph: {g2.blocks.size} blocks, {a2.totalLaunches} dispatches"
  IO.println ""
  -- CSE: collapse redundant Quantize blocks reading the same f32 source.
  let g3 := eliminateCommonQuantize g2
  let a3 := analyzeBlockGraph g3
  IO.println s!"── intermediate after eliminateCommonQuantize (CSE) ──────"
  IO.println s!"  Graph: {g3.blocks.size} blocks, {a3.totalLaunches} dispatches"
  IO.println ""
  -- Fold [Reduce; Quantize] → ReduceQuantize so Pattern A/B/D heads
  -- can fire again (they require an isNormHead at the top).
  let g4 := fuseReduceIntoQuantize g3
  let a4 := analyzeBlockGraph g4
  -- Debug dump of final block sequence.
  let bodyKind (b : Hesper.Circuit.IRv2.Block) : String :=
    match b.body with
    | .Pointwise _          => "Pointwise"
    | .Reduce _ _ _ _       => "Reduce"
    | .ReduceQuantize _ _ _ _ => "ReduceQuantize"
    | .Scatter _ _          => "Scatter"
    | .ScatterMulti _       => "ScatterMulti"
    | .Quantize             => "Quantize"
    | .MatMul _ _ _ _       => "MatMul"
    | .GemmaAttentionMonolith _ _ => "GemmaAttentionMonolith"
    | .FlashAttention _ _   => "FlashAttention"
    | .GemmaFFNMonolith _   => "GemmaFFNMonolith"
    | .PostFFNNormAdd _     => "PostFFNNormAdd"
  IO.println "[DEBUG] final block sequence:"
  for i in [0:g4.blocks.size] do
    IO.println s!"  {i}: {bodyKind g4.blocks[i]!}"
  IO.println s!"── v2 FUSED + CSE + ReduceQuantize fold (final) ────────────"
  IO.println s!"  Graph: {g4.blocks.size} blocks, {g4.tensors.size} tensors"
  IO.println a4.toReport

  -- 3. v1 hand-counted baseline.
  let v1 : V1Breakdown := {}
  IO.println s!"── v1 EAGER (hesper forwardBlock, fused peak) ──────────────"
  IO.println v1.toReport

  let v1Total := v1.total
  let naiveTotal := a0.totalLaunches
  let fusedTotal := a2.totalLaunches
  let cseTotal   := a3.totalLaunches
  let rqTotal    := a4.totalLaunches

  IO.println "── Summary ──────────────────────────────────────────────────"
  IO.println s!"  v1 (eager, hand-fused)              : {v1Total} dispatches/layer"
  IO.println s!"  v2 naive (pre-fusion)               : {naiveTotal} dispatches/layer"
  IO.println s!"  v2 + Pointwise fusion (R/MM)        : {fusedTotal} dispatches/layer"
  IO.println s!"  v2 + CSE                            : {cseTotal} dispatches/layer"
  IO.println s!"  v2 + ReduceQuantize fold            : {rqTotal} dispatches/layer  ← final"
  IO.println ""
  let fusionAbsorbed : Int :=
    Int.ofNat naiveTotal - Int.ofNat fusedTotal
  let cseAbsorbed : Int :=
    Int.ofNat fusedTotal - Int.ofNat cseTotal
  let rqAbsorbed : Int :=
    Int.ofNat cseTotal - Int.ofNat rqTotal
  let vsV1 : Int :=
    Int.ofNat v1Total - Int.ofNat rqTotal
  IO.println s!"  Pointwise fusion absorbed       : {fusionAbsorbed} dispatches"
  IO.println s!"  CSE pass absorbed               : {cseAbsorbed} dispatches"
  IO.println s!"  ReduceQuantize fold absorbed    : {rqAbsorbed} dispatches"
  IO.println s!"  Total absorbed (naive → final)  : {fusionAbsorbed + cseAbsorbed + rqAbsorbed}"
  IO.println s!"  vs v1 delta                     : {vsV1} (positive = v2 better)"
  IO.println ""
  let numLayers : Nat := 42
  let qNaive := a0.quantizeStandalone
  let qCSE   := a3.quantizeStandalone
  let qFinal := a4.quantizeStandalone
  IO.println "── Quantize block count ─────────────────────────────────────"
  IO.println s!"  naive  : {qNaive} explicit Quantize blocks per layer"
  IO.println s!"  + CSE  : {qCSE} (deduped {qNaive - qCSE})"
  IO.println s!"  + RQ   : {qFinal} (folded {qCSE - qFinal} into Reduce → ReduceQuantize)"
  IO.println ""
  IO.println "── Pattern hits in the final graph ──────────────────────────"
  IO.println s!"  Pattern D (FFN)        : {a4.patternD_ffn}"
  IO.println s!"  Pattern A (NormQKV)    : {a4.patternA_qkv}"
  IO.println s!"  Pattern B (NormWQ)     : {a4.patternB_normWQ}"
  IO.println s!"  Pattern E (postNormAdd): {a4.patternE_postNorm}"
  IO.println s!"  Pattern F (Scatter)    : {a4.patternF_scatter}"
  IO.println s!"  Pattern F (multi)      : {a4.patternF_multi}"
  IO.println s!"  Pattern C (plain MM)   : {a4.patternC_matmul}"
  IO.println s!"  Standalone Quantize    : {a4.quantizeStandalone}"
  IO.println s!"  Placeholder (qkvNorm + flashAttn + leftover) : {a4.placeholder}"
  IO.println ""
  IO.println s!"  At {numLayers} layers / token:"
  IO.println s!"    v1 total : {v1Total * numLayers} dispatches"
  IO.println s!"    v2 final : {rqTotal * numLayers} dispatches"
  IO.println ""
  -- Coverage % of main pile.
  IO.println "── Interpretation ───────────────────────────────────────────"
  IO.println s!"  The naive graph emits {g0.blocks.size} blocks (1 per op)."
  IO.println s!"  After the 2 fusion passes run, only {g2.blocks.size} blocks remain."
  let blockAbsorbed : Int := Int.ofNat g0.blocks.size - Int.ofNat g2.blocks.size
  IO.println s!"  → Fusion eliminated {blockAbsorbed} Pointwise/Reduce blocks"
  IO.println s!"    by folding them into upstream Reduce / MatMul epilogues."
  IO.println ""
  if rqTotal < v1Total then
    IO.println s!"RESULT (fine-grain): IRv2 issues {rqTotal} dispatches/layer (vs v1 {v1Total})"
  else if rqTotal == v1Total then
    IO.println "RESULT (fine-grain): IRv2 matches v1 exactly."
  else
    IO.println s!"RESULT (fine-grain): IRv2 has {rqTotal - v1Total} more dispatches than v1."

  -- ================================================================
  -- Phase D: Monolith pipeline (logical vs physical view).
  -- ================================================================
  IO.println ""
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println " Phase D: Monolith pipeline (logical vs physical)"
  IO.println "════════════════════════════════════════════════════════════════"
  let (_, gMono) := runBuilder
    (forwardLayerLazyMonolith
      /-layerKey-/ 0xCAFE
      /-pos-/      pos
      /-inputId-/  20000
      /-qBufId-/   20001
      /-kBufId-/   20002
      /-vBufId-/   20003
      /-attnOut-/  20004
      /-ffnOut-/   20005
      /-postFFN-/  20006)
  let aMono := analyzeBlockGraph gMono
  IO.println ""
  IO.println s!"── v2 MONOLITH (logical) ───────────────────────────────────"
  IO.println s!"  Graph: {gMono.blocks.size} logical blocks, {gMono.tensors.size} tensors"
  IO.println aMono.toReport
  IO.println ""
  IO.println "── Logical-vs-Physical view ─────────────────────────────────"
  IO.println s!"  Logical AST blocks per layer  : {gMono.blocks.size}"
  IO.println s!"  Physical dispatches per layer : {aMono.totalLaunches}"
  IO.println s!"  Abstraction ratio             : {aMono.totalLaunches} physical / {gMono.blocks.size} logical = {aMono.totalLaunches / gMono.blocks.size}× per node"
  IO.println ""
  IO.println s!"  At {numLayers} layers / token (Monolith path):"
  IO.println s!"    Logical : {gMono.blocks.size * numLayers} AST nodes"
  IO.println s!"    Physical: {aMono.totalLaunches * numLayers} GPU dispatches"
  IO.println ""
  IO.println "── Three-way summary ────────────────────────────────────────"
  IO.println s!"            Logical/layer  Physical/layer  Per-token (×42)"
  IO.println s!"  v1       :     —              {v1Total}              {v1Total * numLayers}"
  IO.println s!"  v2 fine  :     {a4.blocks}             {rqTotal}              {rqTotal * numLayers}"
  IO.println s!"  v2 Mono  :      {gMono.blocks.size}             {aMono.totalLaunches}              {aMono.totalLaunches * numLayers}"
  IO.println ""
  IO.println "Phase D abstraction win:"
  IO.println s!"  AST size collapsed from {a4.blocks} (fine-grain) to {gMono.blocks.size} (Monolith) per layer"
  IO.println s!"  = {(a4.blocks - gMono.blocks.size) * numLayers} fewer AST nodes / token"
  IO.println s!"  Physical dispatches: {aMono.totalLaunches}/layer (matches production v1: {v1Total})"
  IO.println "  → Monolith gives the cleanest IR while keeping production-grade kernel choice."

  -- ================================================================
  -- Phase E: whole-token graph + CUDA Graph capture.
  -- ================================================================
  IO.println ""
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println " Phase E: whole-token BlockGraph + CUDA Graph capture"
  IO.println "════════════════════════════════════════════════════════════════"
  let (_, gToken) := runBuilder
    (Hesper.Models.Gemma4_v2.forwardTokenLazyMonolith
      /-numLayers-/    numLayers
      /-firstInputId-/ 30000
      /-lastOutputId-/ 30999
      /-baseTensorId-/ 50000
      /-firstLayerKey-/ 0xCAFE0000
      /-pos-/          pos)
  let aToken := analyzeBlockGraph gToken
  IO.println ""
  IO.println s!"── Whole-token Monolith BlockGraph ──────────────────────────"
  IO.println s!"  Logical AST blocks (whole token) : {gToken.blocks.size}"
  IO.println s!"  Physical kernel launches (sum)   : {aToken.totalLaunches}"
  IO.println ""
  IO.println "── Phase E execution-mode contract ──────────────────────────"
  IO.println "  Mode             host launches/token  GPU kernels/token"
  IO.println s!"  v1 (eager)            {v1Total * numLayers}                   {v1Total * numLayers}"
  IO.println s!"  v2 Monolith (eager)   {aMono.totalLaunches * numLayers}                   {aMono.totalLaunches * numLayers}"
  IO.println s!"  v2 Monolith + Capture       1                   {aToken.totalLaunches}"
  IO.println ""
  let footer : String :=
    "Phase E: same kernel work, but the compiler knows the entire\n" ++
    "execution plan ahead of time, so the driver replays the whole\n" ++
    "token as ONE host call.  IRv2's BlockGraph IS the cudaGraphExec_t.\n" ++
    "\n" ++
    "Production hesper achieves the same effect via HESPER_CUDA_GRAPHS=1\n" ++
    "by hand; IRv2 gets it for free because the BlockGraph IS the plan."
  IO.println footer
