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
# Phase B5 PoC: Post-FFN RMSNorm + Residual-add parity via IRv2

Builds a 1-block IRv2 graph:

    [ Reduce sumSq, applyBody = (ffnOut * rsqrt(sum/N + eps)) * scale
                                 + residual ]

The dispatcher recognises the shape `.add (_) (.input 3)` at the
top-level of the applyBody, resolves the `RMSNorm` handle for the
block's writes[0] tensor id, and routes to the hand-tuned
`forwardNormThenAdd` kernel (single dispatch).

Parity target is a direct call to `forwardNormThenAdd` using the
real `postFFNNorm` weight from Gemma 4 layer 5.
-/

open Hesper
open Hesper.Circuit
open Hesper.Circuit.IRv2
open Hesper.Models.Gemma4_v2

def maxAbsDiffPostFFN (a b : Array Float) : Float := Id.run do
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
  IO.println "=== Phase B5 PoC: Post-FFN RMSNorm+Residual parity via IRv2 ==="
  let modelPath := "data/gemma-4-e4b-it-Q4_K_M.gguf"
  let ctx ← Hesper.CUDAContext.init
  IO.println s!"[Load] {modelPath}"
  let model ← Hesper.Models.Gemma4.Gemma4Model.fromGGUF ctx modelPath

  -- Use a fixed layer — postFFNNorm exists on every layer.
  let li : Nat := 5
  let block ← match model.blocks[li]? with
    | some b => pure b
    | none   => do IO.println "FAIL: missing block"; IO.Process.exit 1
  let postNorm := block.postFFNNorm
  let dim := postNorm.config.dim
  let eps := postNorm.config.eps
  IO.println s!"[Layer] {li}  dim={dim}  eps={eps}"

  -- Deterministic inputs: two distinct sawtooth patterns so the
  -- residual-add is observable.
  let ffnOutArr : Array Float :=
    (List.range dim).toArray.map (fun i => Float.sin (i.toFloat * 0.017) * 0.5)
  let residArr : Array Float :=
    (List.range dim).toArray.map (fun i => Float.cos (i.toFloat * 0.023) * 0.3)
  let ffnOutBytes ← Hesper.Basic.floatArrayToBytes ffnOutArr
  let residBytes  ← Hesper.Basic.floatArrayToBytes residArr

  let bufSz := (dim * 4).toUSize
  let ffnOutBuf    ← GPUBackend.allocBuffer ctx bufSz
  let residualBuf  ← GPUBackend.allocBuffer ctx bufSz
  GPUBackend.writeBuffer ctx ffnOutBuf   ffnOutBytes
  GPUBackend.writeBuffer ctx residualBuf residBytes

  -- ================================================================
  -- REFERENCE: direct forwardNormThenAdd.
  -- ================================================================
  let outBufRef ← GPUBackend.allocBuffer ctx bufSz
  let refRef : IO.Ref (Option (GPUBackend.CachedDispatch _)) ← IO.mkRef none
  Hesper.Layers.RMSNorm.forwardNormThenAdd ctx postNorm
    ffnOutBuf residualBuf outBufRef refRef
  let refArr ← readF32Buf ctx outBufRef dim
  IO.println s!"[Ref ] out[0..4] = {refArr.toList.take 4}"

  -- ================================================================
  -- IRv2: build 1-block Reduce → dispatcher routes to forwardNormThenAdd.
  -- ================================================================
  let ffnOutId    : Nat := 4000
  let scaleId     : Nat := 4001
  let residualId  : Nat := 4002
  let outId       : Nat := 4003
  let (_, graph) := runBuilder
    (buildPostFFNLazy ffnOutId scaleId residualId outId dim eps)
  IO.println s!"[IRv2] graph blocks: {graph.blocks.size}, tensors: {graph.tensors.size}"
  if graph.blocks.size != 1 then
    IO.println s!"FAIL: expected 1 block, got {graph.blocks.size}"
    IO.Process.exit 1
  let outBufV2 ← GPUBackend.allocBuffer ctx bufSz
  Hesper.Circuit.IRv2.runBlockGraph ctx graph
    (externalBufs :=
      [(ffnOutId,   ffnOutBuf),
       (scaleId,    postNorm.scale),
       (residualId, residualBuf),
       (outId,      outBufV2)])
    (matmulLayers := [])
    (matmulInputBufs := [])
    (normHandles := [(outId, postNorm)])
  let v2Arr ← readF32Buf ctx outBufV2 dim
  IO.println s!"[IRv2] out[0..4] = {v2Arr.toList.take 4}"

  let err := maxAbsDiffPostFFN refArr v2Arr
  IO.println s!"[Parity] max |err| = {err}"
  if err == 0.0 then
    IO.println "PASS: IRv2 post-FFN is BIT-IDENTICAL to the reference path"
  else if err < 1e-5 then
    IO.println s!"PASS (≈): IRv2 post-FFN matches reference to {err}"
  else
    IO.println s!"FAIL: post-FFN mismatch exceeds 1e-5"
    IO.Process.exit 1
