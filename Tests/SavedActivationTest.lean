import Hesper
import Hesper.Models.BitNet
import Hesper.LoRA.Types
import Hesper.LoRA.Init
import Hesper.LoRA.Inference
import Hesper.Training.SafeBuffer
import Hesper.GGUF.Reader
import Hesper.Tokenizer.SentencePiece

/-!
# Saved Activation Test

Verifies that savedAttnOutput (qRotBuf after attention apply) is correctly
saved during forward pass and contains valid (non-NaN, non-zero) values.

This test diagnoses the root cause of RMSNorm backward NaN.
-/

open Hesper.WebGPU
open Hesper.Models.BitNet
open Hesper.LoRA
open Hesper.GGUF
open Hesper.Training.SafeBuffer

def main (args : List String) : IO Unit := do
  let modelPath := args.getD 0 "data/gguf/ggml-model-i2_s.gguf"

  IO.println "=== Saved Activation Test ==="
  IO.println ""

  -- Initialize
  let inst ← Hesper.init
  let device ← Hesper.WebGPU.getDevice inst
  let gguf ← loadGGUF modelPath
  let model ← fromGGUFObject device gguf none
  let dim := model.config.dim

  IO.println s!"Model loaded: {model.config.numLayers} layers, dim={dim}"

  -- Create LoRA adapter + training state
  let loraConfig : Hesper.LoRA.Config := { rank := 8, alpha := 8.0 }
  let adapter ← createAdapter device loraConfig model.config.numLayers dim model.config.kvDim
  let loraState ← Inference.createLoRATrainingState device adapter
    dim model.config.kvDim model.config.numHeads model.config.headDim
    model.config.maxSeqLen model.config.numLayers

  IO.println s!"LoRA state created: {loraState.savedNormed.size} savedNormed, {loraState.savedAttnOut.size} savedAttnOut"

  -- Create KV cache
  let cacheState ← createKVCacheState device model
  resetPreparedDispatches model

  -- Run forward for 1 token
  IO.println ""
  IO.println "Running forward for token 0 (BOS=128000)..."
  let grads ← createAdapterGrad device adapter
  let trainState ← Hesper.Training.TrainLoop.createTrainState device adapter dim model.config.kvDim
  let lossBuf ← createBuffer device { size := 4, usage := [.storage, .copySrc, .copyDst, .mapRead], mappedAtCreation := false }
  let targetBuf ← createBuffer device { size := 4, usage := [.storage, .copySrc, .copyDst], mappedAtCreation := false }
  let dLogitsBuf ← createBuffer device { size := (model.config.vocabSize * 4).toUSize, usage := [.storage, .copySrc, .copyDst], mappedAtCreation := false }
  let dHiddenBuf ← createBuffer device { size := (dim * 4).toUSize, usage := [.storage, .copySrc, .copyDst], mappedAtCreation := false }

  -- Zero loss buf
  writeBuffer device lossBuf 0 (ByteArray.mk #[0,0,0,0])
  -- Write target token
  writeBuffer device targetBuf 0 (Hesper.WebGPU.BufferOps.uint32ToBytes 42)

  -- Forward with isOutputToken=true to trigger activation saving
  Inference.forwardAndBackwardBatched device model
    128000 0 cacheState adapter loraState
    true targetBuf lossBuf dLogitsBuf dHiddenBuf
    grads trainState 0

  IO.println "Forward complete."
  IO.println ""

  -- Check savedAttnOut for each layer
  IO.println "=== Checking savedAttnOut (attention output, input to sub-norm) ==="
  let mut allOk := true
  for i in [:model.config.numLayers] do
    if h : i < loraState.savedAttnOut.size then
      let vals ← safeMapBufferReadF32 device loraState.savedAttnOut[i] 8
      let hasNan := vals.any isNaN
      let allZero := vals.all (· == 0.0)
      let maxAbs := vals.foldl (init := 0.0) fun acc v =>
        let a := if v < 0.0 then 0.0 - v else v
        if a > acc then a else acc
      let status := if hasNan then "NaN!" else if allZero then "ALL_ZERO" else "OK"
      if i == 0 || i == 14 || i == 28 || i == 29 || hasNan || allZero then
        IO.println s!"  Layer {i}: {status} max_abs={maxAbs} first={vals.getD 0 0.0}"
      if hasNan || allZero then allOk := false

  IO.println ""

  -- Check savedNormed for comparison
  IO.println "=== Checking savedNormed (pre-attention RMSNorm output) ==="
  for i in [:model.config.numLayers] do
    if h : i < loraState.savedNormed.size then
      let vals ← safeMapBufferReadF32 device loraState.savedNormed[i] 8
      let hasNan := vals.any isNaN
      let allZero := vals.all (· == 0.0)
      let maxAbs := vals.foldl (init := 0.0) fun acc v =>
        let a := if v < 0.0 then 0.0 - v else v
        if a > acc then a else acc
      let status := if hasNan then "NaN!" else if allZero then "ALL_ZERO" else "OK"
      if i == 0 || i == 14 || i == 28 || i == 29 || hasNan || allZero then
        IO.println s!"  Layer {i}: {status} max_abs={maxAbs} first={vals.getD 0 0.0}"

  IO.println ""
  if allOk then
    IO.println "✓ All saved activations are valid (no NaN, no all-zero)"
  else
    IO.println "✗ Some saved activations are INVALID — this causes RMSNorm backward NaN"
