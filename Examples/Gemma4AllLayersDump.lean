import Hesper
import Hesper.Models.Gemma4
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WebGPU.BufferOps

/-!
# Gemma 4 — Run forwardBlock for ALL layers, dumping each layer output

For each layer N from 0 to numHiddenLayers-1, dump the output to
/tmp/hesper_dump/forward_layer{N}_out.bin

Then compare against /tmp/llama_dump/l_out-{N}.bin to find where divergence occurs.
-/

open Hesper.WebGPU
open Hesper.Models.Gemma4

def dumpBuf (device : Device) (path : String) (buf : Buffer) (n : Nat) : IO Unit := do
  let data ← Hesper.WebGPU.BufferOps.downloadFloatArray device buf n
  let mut bytes := ByteArray.empty
  for v in data do
    let fb ← Hesper.Basic.floatToBytes v
    bytes := bytes ++ fb
  IO.FS.writeBinFile path bytes

def main : IO Unit := do
  let modelPath := "data/gemma-4-e4b-it-Q4_K_M.gguf"
  IO.FS.createDirAll "/tmp/hesper_dump"

  IO.println "[Init] Creating WebGPU device..."
  let inst ← Hesper.init
  let device ← Hesper.WebGPU.getDevice inst

  IO.println s!"[Load] Loading model..."
  let model ← Gemma4Model.fromGGUF device modelPath
  let cfg := model.config
  let tokenId := 9259  -- "Hello"
  let pos := 0

  let state ← createInferenceState device cfg

  -- Embedding lookup
  let tokenBytes := Hesper.WebGPU.BufferOps.uint32ToBytes tokenId.toUInt32
  writeBuffer device state.tokenBuf 0 tokenBytes
  match model.embdFormat with
  | .Q6_K =>
    let bufs : List (String × Buffer) :=
      [("token_ids", state.tokenBuf), ("embedding_table", model.embedding.embeddingTable), ("output", state.buf1)]
    Hesper.WGSL.Execute.executeShaderNamed device
      (Hesper.Quantization.Q6_K.q6kEmbeddingLookupKernel cfg.vocabSize cfg.hiddenSize)
      bufs (Hesper.WGSL.Execute.ExecutionConfig.dispatch1D cfg.hiddenSize)
  | _ =>
    Hesper.Layers.Embedding.forward device model.embedding state.tokenBuf state.buf1 1 1

  -- Embedding scale: buf1 → buf2
  let scaleBufs : List (String × Buffer) := [("input", state.buf1), ("output", state.buf2)]
  Hesper.WGSL.Execute.executeShaderNamed device
    (Hesper.Models.Gemma4.embeddingScaleKernel cfg.hiddenSize cfg.hiddenSize)
    scaleBufs (Hesper.WGSL.Execute.ExecutionConfig.dispatch1D cfg.hiddenSize)

  -- Per-layer input precompute
  match model.perLayerEmbdTableCPU, model.perLayerModelProj, model.perLayerProjNorm with
  | some embdTableCPU, some modelProj, some projNorm =>
    let embdPL := cfg.embdPerLayer
    let nLayers := cfg.numHiddenLayers
    let totalPL := embdPL * nLayers
    let rowOffset := tokenId * model.perLayerEmbdRowBytes
    let rowFloats := dequantQ6KRowCPU embdTableCPU rowOffset totalPL
    let scaleFactor : Float := Float.sqrt embdPL.toFloat
    let scaledRow := rowFloats.map (· * scaleFactor)
    let rowBytes ← floatArrayToBytes scaledRow
    writeBuffer device state.plModelProj 0 rowBytes
    let projConfig : Hesper.WGSL.MatMul.Config := {
      M := 1, N := totalPL, K := cfg.hiddenSize
    }
    Hesper.WGSL.MatMul.executeMatMulTransposeF16 device state.buf2 modelProj state.plTokenSelected projConfig
    let bufs2 : List (String × Buffer) :=
      [("input", state.plTokenSelected), ("output", state.plInputAll)]
    Hesper.WGSL.Execute.executeShaderNamed device
      (Hesper.Layers.PerLayerEmbedding.scaleKernel totalPL (1.0 / Float.sqrt cfg.hiddenSize.toFloat))
      bufs2 (Hesper.WGSL.Execute.ExecutionConfig.dispatch1D totalPL)
    let chunkBufs : List (String × Buffer) :=
      [("input", state.plInputAll), ("weight", projNorm.scale), ("output", state.plTokenSelected)]
    Hesper.WGSL.Execute.executeShaderNamed device
      (Hesper.Models.Gemma4.chunkedRMSNormKernel embdPL nLayers cfg.rmsNormEps)
      chunkBufs
      { numWorkgroups := (nLayers, 1, 1), workgroupSize := { x := min embdPL 256, y := 1, z := 1 } : Hesper.WGSL.Execute.ExecutionConfig }
    let addBufs : List (String × Buffer) :=
      [("a", state.plTokenSelected), ("b", state.plModelProj), ("output", state.plInputAll)]
    Hesper.WGSL.Execute.executeShaderNamed device
      (Hesper.Models.Gemma4.scaledAddKernel totalPL (1.0 / Float.sqrt 2.0))
      addBufs (Hesper.WGSL.Execute.ExecutionConfig.dispatch1D totalPL)
  | _, _, _ => pure ()

  -- Run all layers, dumping after each
  let mut currentBuf := state.buf2
  let mut nextBuf := state.buf1
  let plInputBuf := if cfg.hasPerLayerEmbeddings then some state.plInputAll else none

  for hi : i in [0:cfg.numHiddenLayers] do
    if hb : i < model.blocks.size then
      let block := model.blocks[i]
      let plEmbd := if hp : i < model.perLayerBlocks.size then model.perLayerBlocks[i] else none
      forwardBlock device block cfg currentBuf nextBuf state pos plEmbd plInputBuf
      -- Dump nextBuf (which now contains layer i's output)
      dumpBuf device s!"/tmp/hesper_dump/forward_layer{i}_out.bin" nextBuf cfg.hiddenSize
      let tmp := currentBuf
      currentBuf := nextBuf
      nextBuf := tmp
      if i % 5 == 0 || i == cfg.numHiddenLayers - 1 then
        IO.println s!"  Dumped layer {i}"

  IO.println "Done! Compare with: python3 scripts/compare_all_layers.py"
