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
# Phase B PoC: Q-projection parity via IRv2 BlockGraph

Compares two paths producing the same Q tensor for layer-0 wQ:

  REFERENCE: call `runMatmulQ4KWithEpilogueOp` directly with
             epilogue = identity (slot 0 passthrough).
  IRv2     : build a `BlockGraph` with a single `MatMul` block whose
             epilogue is `.input 0`, extract the epilogue from the
             fused graph, dispatch via the SAME executor.

Both paths use the same Q4_K weights loaded from the real GGUF, the
same f32 input, and the same underlying Q4_K DP4A kernel.  They must
produce bit-identical Q tensors.

This is the minimum-viable "a real inference pipeline is compiled from
IRv2 against real quantized weights" milestone.  Subsequent PoCs will
(a) add a Reduce block for RMSNorm and fuse it into MatMul, and
(b) grow the graph to cover the full attention prologue.
-/

open Hesper
open Hesper.Circuit
open Hesper.Circuit.IRv2
open Hesper.Models.Gemma4_v2

/-- Max absolute difference between two float arrays (caller guarantees
    equal length). -/
def maxAbsDiffQ (a b : Array Float) : Float := Id.run do
  let mut m : Float := 0.0
  for i in [0:a.size] do
    let d := (a[i]! - b[i]!).abs
    if d > m then m := d
  return m

def main : IO Unit := do
  IO.println "=== Phase B PoC: Q-projection parity via IRv2 BlockGraph ==="
  let modelPath := "data/gemma-4-e4b-it-Q4_K_M.gguf"
  -- Enable dp4a (required for the fused RMSNorm+Q8_1+matmul paths).
  Hesper.Layers.Linear.dp4aEnabled.set true
  -- 1. Init CUDA backend.
  let ctx ← Hesper.CUDAContext.init
  -- 2. Load Gemma 4 weights from GGUF (all on CUDA).
  IO.println s!"[Load] {modelPath}"
  let model ← Hesper.Models.Gemma4.Gemma4Model.fromGGUF ctx modelPath
  let cfg := model.config
  IO.println s!"[Model] hiddenSize={cfg.hiddenSize} numAttentionHeads={cfg.numAttentionHeads}"
  let block0 ← match model.blocks[0]? with
    | some b => pure b
    | none   => do
      IO.println "FAIL: model has no blocks"
      IO.Process.exit 1
  let wQ := block0.attention.wQ
  if wQ.quantFormat != .Q4_K then
    IO.println s!"FAIL: layer 0 wQ quant format is {repr wQ.quantFormat}, need Q4_K"
    IO.Process.exit 1
  let inDim  := wQ.config.inDim
  let outDim := wQ.config.outDim
  IO.println s!"[wQ] inDim={inDim} outDim={outDim}"

  -- 3. Deterministic f32 input: a sawtooth pattern so every cell differs.
  let inArr : Array Float :=
    (List.range inDim).toArray.map (fun i =>
      Float.sin (i.toFloat * 0.017) * 0.5)
  let inBytes ← Hesper.Basic.floatArrayToBytes inArr
  let inBufSz  : USize := (inDim  * 4).toUSize
  let outBufSz : USize := (outDim * 4).toUSize

  -- 4. Upload input to GPU.
  let inputBuf ← GPUBackend.allocBuffer ctx inBufSz
  GPUBackend.writeBuffer ctx inputBuf inBytes

  -- Pre-attention RMSNorm (attnNorm) — same scale used by both paths.
  let attnNorm := block0.attnNorm
  if attnNorm.config.dim != inDim then
    IO.println s!"FAIL: attnNorm dim {attnNorm.config.dim} ≠ wQ inDim {inDim}"
    IO.Process.exit 1
  let eps := attnNorm.config.eps

  -- 5. REFERENCE path: call the existing hand-tuned
  --    `forwardFusedNormWQ` (fused RMSNorm+Q8_1 quantize + dp4a matmul).
  let outBufRef ← GPUBackend.allocBuffer ctx outBufSz
  Hesper.Layers.Linear.forwardFusedNormWQ ctx attnNorm wQ inputBuf outBufRef
  let outBytesRef ← GPUBackend.readBuffer ctx outBufRef outBufSz
  let mut outArrRef : Array Float := Array.mkEmpty outDim
  for i in [0:outDim] do
    let f ← Hesper.Basic.bytesToFloat32 outBytesRef (i * 4)
    outArrRef := outArrRef.push f
  IO.println s!"[Ref ] Q[0..4] = {outArrRef.toList.take 4}"

  -- 6. IRv2 path: build a 2-block graph [Reduce(RMSNorm); MatMul(wQ)]
  --    and let the dispatcher recognise the adjacency and route to
  --    `forwardFusedNormWQ` automatically.
  let inputId  : Nat    := 1000
  let scaleId  : Nat    := 1001
  let outId    : Nat    := 1002
  let normKey  : Nat    := 7001
  let layerKey : UInt64 := 0x516a
  let (_, graph) := runBuilder
    (Hesper.Models.Gemma4_v2.buildNormQProjLazy
       inputId scaleId outId layerKey normKey inDim outDim eps)
  IO.println s!"[IRv2] unfused blocks: {graph.blocks.size}, tensors: {graph.tensors.size}"
  -- No Pointwise neighbours in the produced graph, so fusion passes
  -- are pass-through here.  Run them anyway to exercise the roundtrip.
  let graph' := fusePointwiseIntoReduce graph
  let graph' := fusePointwiseIntoMatMul graph'
  IO.println s!"[IRv2] post-fusion blocks: {graph'.blocks.size}, tensors: {graph'.tensors.size}"
  if graph'.blocks.size != 2 then
    IO.println s!"FAIL: expected 2 blocks, got {graph'.blocks.size}"
    IO.Process.exit 1
  -- Dispatch the graph.  The input buffer is fed via two routes:
  --   - as an external tensor (Reduce reads it via its `reads` region),
  --   - as the MatMul's f32 input (plumbed via matmulInputBufs).
  -- For the fused-path the dispatcher uses the external-tensor route;
  -- matmulInputBufs is only consulted on the fallback plain-MatMul
  -- path, which we don't take here.
  let outBufV2 ← GPUBackend.allocBuffer ctx outBufSz
  Hesper.Circuit.IRv2.runBlockGraph ctx graph'
    (externalBufs :=
      [(inputId, inputBuf), (scaleId, attnNorm.scale), (outId, outBufV2)])
    (matmulLayers := [(layerKey, wQ)])
    (matmulInputBufs := [(layerKey, inputBuf)])
    (normHandles := [(normKey, attnNorm)])
  let outBytesV2 ← GPUBackend.readBuffer ctx outBufV2 outBufSz
  let mut outArrV2 : Array Float := Array.mkEmpty outDim
  for i in [0:outDim] do
    let f ← Hesper.Basic.bytesToFloat32 outBytesV2 (i * 4)
    outArrV2 := outArrV2.push f
  IO.println s!"[IRv2] Q[0..4] = {outArrV2.toList.take 4}"

  -- 7. Parity assertion.
  let err := maxAbsDiffQ outArrRef outArrV2
  IO.println s!"[Parity] max |err| = {err}"
  if err == 0.0 then
    IO.println "PASS: IRv2 Q-projection is BIT-IDENTICAL to the reference path"
  else if err < 1.0e-5 then
    IO.println s!"PASS (≈): IRv2 Q-projection matches reference to {err}"
  else
    IO.println s!"FAIL: Q mismatch exceeds 1e-5 (max |err| = {err})"
    IO.Process.exit 1
