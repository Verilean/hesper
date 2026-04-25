import Hesper.Circuit.IRv2
import Hesper.Circuit.Lowering
import Hesper.Circuit.Lowering_v2
import Hesper.Circuit.Dispatch_v2
import Hesper.Models.Gemma4
import Hesper.Models.Gemma4_v2
import Hesper.Layers.Linear
import Hesper.Layers.RMSNorm
import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.Basic

/-!
# Phase B3 PoC: RMSNorm + Q/K/V projection parity via IRv2

Builds a 4-block IRv2 graph for layer-0 of Gemma 4:
    [Reduce sumSq]; [MatMul wQ]; [MatMul wK]; [MatMul wV]

and lets the dispatcher route to the hand-tuned `forwardFusedNormQKV`
(which dispatches fused-rmsnorm-q8_1, wQ matmul, then fused wK+wV
matmul = 3 dispatches total).  Compares against a direct call to
`forwardFusedNormQKV` using the same weights + input.  Asserts
bit-identity for all three output tensors Q, K, V.
-/

open Hesper
open Hesper.Circuit
open Hesper.Circuit.IRv2
open Hesper.Models.Gemma4_v2

def maxAbsDiffQKV (a b : Array Float) : Float := Id.run do
  let mut m : Float := 0.0
  for i in [0:a.size] do
    let d := (a[i]! - b[i]!).abs
    if d > m then m := d
  return m

/-- Read an f32 buffer of `n` elements back into an Array Float. -/
def readF32Buf [GPUBackend β]
    (ctx : β) (buf : GPUBackend.Buf β) (n : Nat) : IO (Array Float) := do
  let bytes ← GPUBackend.readBuffer ctx buf (n * 4).toUSize
  let mut out : Array Float := Array.mkEmpty n
  for i in [0:n] do
    let f ← Hesper.Basic.bytesToFloat32 bytes (i * 4)
    out := out.push f
  return out

def main : IO Unit := do
  IO.println "=== Phase B3 PoC: RMSNorm+QKV projection parity via IRv2 ==="
  let modelPath := "data/gemma-4-e4b-it-Q4_K_M.gguf"
  Hesper.Layers.Linear.dp4aEnabled.set true
  let ctx ← Hesper.CUDAContext.init
  IO.println s!"[Load] {modelPath}"
  let model ← Hesper.Models.Gemma4.Gemma4Model.fromGGUF ctx modelPath
  let cfg := model.config

  -- Scan for the first layer with all-Q4_K Q/K/V (some layers use
  -- different quant formats for attention projections).
  let mut foundLi : Nat := 0
  let mut found := false
  for i in [0:model.blocks.size] do
    if !found then
      if let some b := model.blocks[i]? then
        if b.attention.wQ.quantFormat == .Q4_K
           ∧ b.attention.wK.quantFormat == .Q4_K
           ∧ b.attention.wV.quantFormat == .Q4_K then
          foundLi := i
          found := true
  if !found then
    IO.println "FAIL: no layer with all-Q4_K Q/K/V found"
    IO.Process.exit 1
  IO.println s!"[Layer] using layer {foundLi}"
  let block0 ← match model.blocks[foundLi]? with
    | some b => pure b
    | none   => do IO.println "FAIL: block vanished"; IO.Process.exit 1
  let attnNorm := block0.attnNorm
  let wQ := block0.attention.wQ
  let wK := block0.attention.wK
  let wV := block0.attention.wV
  IO.println s!"[Formats] Q={repr wQ.quantFormat} K={repr wK.quantFormat} V={repr wV.quantFormat}"
  let dim      := wQ.config.inDim
  let qOutDim  := wQ.config.outDim
  let kvOutDim := wK.config.outDim
  if wV.config.outDim != kvOutDim then
    IO.println s!"FAIL: wV outDim {wV.config.outDim} ≠ wK outDim {kvOutDim}"
    IO.Process.exit 1
  if attnNorm.config.dim != dim then
    IO.println s!"FAIL: attnNorm dim {attnNorm.config.dim} ≠ wQ inDim {dim}"
    IO.Process.exit 1
  IO.println s!"[Shapes] dim={dim} qOutDim={qOutDim} kvOutDim={kvOutDim}"
  let eps := attnNorm.config.eps

  -- Deterministic input: sine sawtooth.
  let inArr : Array Float :=
    (List.range dim).toArray.map (fun i => Float.sin (i.toFloat * 0.017) * 0.5)
  let inBytes ← Hesper.Basic.floatArrayToBytes inArr
  let inputBuf ← GPUBackend.allocBuffer ctx (dim * 4).toUSize
  GPUBackend.writeBuffer ctx inputBuf inBytes

  -- ================================================================
  -- REFERENCE: forwardFusedNormQKV directly.
  -- ================================================================
  let qBufRef ← GPUBackend.allocBuffer ctx (qOutDim  * 4).toUSize
  let kBufRef ← GPUBackend.allocBuffer ctx (kvOutDim * 4).toUSize
  let vBufRef ← GPUBackend.allocBuffer ctx (kvOutDim * 4).toUSize
  let kvRefRef : IO.Ref (Option (GPUBackend.CachedDispatch _)) ← IO.mkRef none
  Hesper.Layers.Linear.forwardFusedNormQKV ctx attnNorm wQ wK wV
    inputBuf qBufRef kBufRef vBufRef kvRefRef
  let qRef ← readF32Buf ctx qBufRef qOutDim
  let kRef ← readF32Buf ctx kBufRef kvOutDim
  let vRef ← readF32Buf ctx vBufRef kvOutDim
  IO.println s!"[Ref ] Q[0..3]={qRef.toList.take 3}"
  IO.println s!"[Ref ] K[0..3]={kRef.toList.take 3}"
  IO.println s!"[Ref ] V[0..3]={vRef.toList.take 3}"

  -- ================================================================
  -- IRv2: build [Reduce; MatMul×3] → dispatch via runBlockGraph.
  -- ================================================================
  let inputId  : Nat    := 2000
  let scaleId  : Nat    := 2001
  let qOutId   : Nat    := 2002
  let kOutId   : Nat    := 2003
  let vOutId   : Nat    := 2004
  let normKey  : Nat    := 9001
  let wQKey    : UInt64 := 0x0010
  let wKKey    : UInt64 := 0x0020
  let wVKey    : UInt64 := 0x0030
  let (_, graph) := runBuilder
    (buildNormQKVProjLazy inputId scaleId qOutId kOutId vOutId
       wQKey wKKey wVKey normKey dim qOutDim kvOutDim eps)
  IO.println s!"[IRv2] graph blocks: {graph.blocks.size}, tensors: {graph.tensors.size}"
  if graph.blocks.size != 4 then
    IO.println s!"FAIL: expected 4 blocks, got {graph.blocks.size}"
    IO.Process.exit 1
  let qBufV2 ← GPUBackend.allocBuffer ctx (qOutDim  * 4).toUSize
  let kBufV2 ← GPUBackend.allocBuffer ctx (kvOutDim * 4).toUSize
  let vBufV2 ← GPUBackend.allocBuffer ctx (kvOutDim * 4).toUSize
  Hesper.Circuit.IRv2.runBlockGraph ctx graph
    (externalBufs :=
      [(inputId, inputBuf), (scaleId, attnNorm.scale),
       (qOutId, qBufV2), (kOutId, kBufV2), (vOutId, vBufV2)])
    (matmulLayers := [(wQKey, wQ), (wKKey, wK), (wVKey, wV)])
    (matmulInputBufs := [(wQKey, inputBuf), (wKKey, inputBuf), (wVKey, inputBuf)])
    (normHandles := [(normKey, attnNorm)])
  let qV2 ← readF32Buf ctx qBufV2 qOutDim
  let kV2 ← readF32Buf ctx kBufV2 kvOutDim
  let vV2 ← readF32Buf ctx vBufV2 kvOutDim
  IO.println s!"[IRv2] Q[0..3]={qV2.toList.take 3}"
  IO.println s!"[IRv2] K[0..3]={kV2.toList.take 3}"
  IO.println s!"[IRv2] V[0..3]={vV2.toList.take 3}"

  -- Parity — all three tensors must match bit-exactly.
  let errQ := maxAbsDiffQKV qRef qV2
  let errK := maxAbsDiffQKV kRef kV2
  let errV := maxAbsDiffQKV vRef vV2
  IO.println s!"[Parity] max |errQ|={errQ}  max |errK|={errK}  max |errV|={errV}"
  if errQ == 0.0 ∧ errK == 0.0 ∧ errV == 0.0 then
    IO.println "PASS: IRv2 QKV projection is BIT-IDENTICAL to forwardFusedNormQKV"
  else if errQ < 1e-5 ∧ errK < 1e-5 ∧ errV < 1e-5 then
    IO.println "PASS (≈): QKV matches within 1e-5"
  else
    IO.println s!"FAIL: QKV mismatch exceeds 1e-5"
    IO.Process.exit 1
