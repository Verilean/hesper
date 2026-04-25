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
# Phase B4 PoC: FFN (RMSNorm+gate+up+GELU×mul+down) parity via IRv2

Builds a 5-block IRv2 graph for a Gemma 4 FFN body:

    [Reduce sumSq]; [MatMul wGate]; [MatMul wUp];
    [Pointwise GELU(gate)*up]; [MatMul wDown]

The dispatcher recognises the pattern and routes to:
    forwardFusedNormGateUp → LinearLayer.forward(wDown)  (2 dispatches)

Parity target is the same hand-tuned sequence invoked directly.
Bit-identity required for all elements of the FFN output buffer.

Scope note: the final residual-add onto the pre-FFN hidden is
deliberately outside this graph — it's merged into the post-FFN
RMSNorm kernel in the production code, which is a separate sub-graph.
-/

open Hesper
open Hesper.Circuit
open Hesper.Circuit.IRv2
open Hesper.Models.Gemma4_v2

def maxAbsDiffFFN (a b : Array Float) : Float := Id.run do
  let mut m : Float := 0.0
  for i in [0:a.size] do
    let d := (a[i]! - b[i]!).abs
    if d > m then m := d
  return m

def readF32Buf [GPUBackend β]
    (ctx : β) (buf : GPUBackend.Buf β) (n : Nat) : IO (Array Float) := do
  let bytes ← GPUBackend.readBuffer ctx buf (n * 4).toUSize
  let mut out : Array Float := Array.mkEmpty n
  for i in [0:n] do
    let f ← Hesper.Basic.bytesToFloat32 bytes (i * 4)
    out := out.push f
  return out

def main : IO Unit := do
  IO.println "=== Phase B4 PoC: FFN parity via IRv2 ==="
  let modelPath := "data/gemma-4-e4b-it-Q4_K_M.gguf"
  Hesper.Layers.Linear.dp4aEnabled.set true
  let ctx ← Hesper.CUDAContext.init
  IO.println s!"[Load] {modelPath}"
  let model ← Hesper.Models.Gemma4.Gemma4Model.fromGGUF ctx modelPath

  -- Scan for the first layer with all-Q4_K gate/up/down.
  let mut foundLi : Nat := 0
  let mut found := false
  for i in [0:model.blocks.size] do
    if !found then
      if let some b := model.blocks[i]? then
        if b.ffn.gate.quantFormat == .Q4_K
           ∧ b.ffn.up.quantFormat == .Q4_K
           ∧ b.ffn.down.quantFormat == .Q4_K then
          foundLi := i
          found := true
  if !found then
    IO.println "FAIL: no layer with all-Q4_K gate/up/down found"
    IO.Process.exit 1
  IO.println s!"[Layer] using layer {foundLi}"
  let block ← match model.blocks[foundLi]? with
    | some b => pure b
    | none   => do IO.println "FAIL"; IO.Process.exit 1
  let ffnNorm := block.ffnNorm
  let wGate := block.ffn.gate
  let wUp   := block.ffn.up
  let wDown := block.ffn.down
  IO.println s!"[Formats] gate={repr wGate.quantFormat} up={repr wUp.quantFormat} down={repr wDown.quantFormat}"
  let dim      := wGate.config.inDim
  let interDim := wGate.config.outDim
  if wUp.config.inDim != dim ∨ wUp.config.outDim != interDim then
    IO.println s!"FAIL: wUp shape {wUp.config.inDim}→{wUp.config.outDim} ≠ {dim}→{interDim}"
    IO.Process.exit 1
  if wDown.config.inDim != interDim ∨ wDown.config.outDim != dim then
    IO.println s!"FAIL: wDown shape {wDown.config.inDim}→{wDown.config.outDim} ≠ {interDim}→{dim}"
    IO.Process.exit 1
  if ffnNorm.config.dim != dim then
    IO.println s!"FAIL: ffnNorm dim {ffnNorm.config.dim} ≠ wGate inDim {dim}"
    IO.Process.exit 1
  IO.println s!"[Shapes] dim={dim} interDim={interDim}"
  let eps := ffnNorm.config.eps

  -- Deterministic input.
  let inArr : Array Float :=
    (List.range dim).toArray.map (fun i => Float.sin (i.toFloat * 0.017) * 0.5)
  let inBytes ← Hesper.Basic.floatArrayToBytes inArr
  let inputBuf ← GPUBackend.allocBuffer ctx (dim * 4).toUSize
  GPUBackend.writeBuffer ctx inputBuf inBytes

  -- ================================================================
  -- REFERENCE: forwardFusedNormGateUp → LinearLayer.forward (wDown).
  -- ================================================================
  let geluBufRef ← GPUBackend.allocBuffer ctx (interDim * 4).toUSize
  let ffnOutRef  ← GPUBackend.allocBuffer ctx (dim * 4).toUSize
  let refRef : IO.Ref (Option (GPUBackend.CachedDispatch _)) ← IO.mkRef none
  Hesper.Layers.Linear.forwardFusedNormGateUp ctx ffnNorm wGate wUp
    inputBuf geluBufRef refRef
  Hesper.Layers.Linear.LinearLayer.forward ctx wDown geluBufRef ffnOutRef
  let refArr ← readF32Buf ctx ffnOutRef dim
  IO.println s!"[Ref ] ffnOut[0..4] = {refArr.toList.take 4}"

  -- ================================================================
  -- IRv2: build 5-block graph → dispatcher auto-routes.
  -- ================================================================
  let inputId  : Nat    := 3000
  let scaleId  : Nat    := 3001
  let outId    : Nat    := 3002
  let normKey  : Nat    := 9101
  let geluKey  : Nat    := 9102
  let wGateKey : UInt64 := 0x0100
  let wUpKey   : UInt64 := 0x0200
  let wDownKey : UInt64 := 0x0300
  let (_, graph) := runBuilder
    (buildFFNLazy inputId scaleId outId wGateKey wUpKey wDownKey
       normKey geluKey dim interDim eps)
  IO.println s!"[IRv2] graph blocks: {graph.blocks.size}, tensors: {graph.tensors.size}"
  if graph.blocks.size != 5 then
    IO.println s!"FAIL: expected 5 blocks, got {graph.blocks.size}"
    IO.Process.exit 1
  -- Dispatcher requires externalBufs for: inputId, scaleId, outId, AND
  -- the intermediate geluKey (because Pattern D dispatcher resolves the
  -- geluBuf from externalBufs for wDown's input).
  let geluBufV2 ← GPUBackend.allocBuffer ctx (interDim * 4).toUSize
  let ffnOutV2  ← GPUBackend.allocBuffer ctx (dim * 4).toUSize
  Hesper.Circuit.IRv2.runBlockGraph ctx graph
    (externalBufs :=
      [(inputId, inputBuf), (scaleId, ffnNorm.scale),
       (outId, ffnOutV2), (geluKey, geluBufV2)])
    (matmulLayers :=
      [(wGateKey, wGate), (wUpKey, wUp), (wDownKey, wDown)])
    (matmulInputBufs :=
      [(wGateKey, inputBuf), (wUpKey, inputBuf), (wDownKey, geluBufV2)])
    (normHandles := [(normKey, ffnNorm)])
  let v2Arr ← readF32Buf ctx ffnOutV2 dim
  IO.println s!"[IRv2] ffnOut[0..4] = {v2Arr.toList.take 4}"

  let err := maxAbsDiffFFN refArr v2Arr
  IO.println s!"[Parity] max |err| = {err}"
  if err == 0.0 then
    IO.println "PASS: IRv2 FFN is BIT-IDENTICAL to the reference path"
  else if err < 1e-5 then
    IO.println s!"PASS (≈): IRv2 FFN matches reference to {err}"
  else
    IO.println s!"FAIL: FFN mismatch exceeds 1e-5"
    IO.Process.exit 1
