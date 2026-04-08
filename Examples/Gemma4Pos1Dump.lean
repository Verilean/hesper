import Hesper
import Hesper.Models.Gemma4
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WebGPU.BufferOps

/-!
# Gemma 4 — pos=1 layer dump

Reproduces the bug where `pos > 0` breaks generation. The plan:

1. Run a full `forwardSingleToken` at pos=0 with token 9259 ("Hello"). This
   populates every layer's KV cache with the pos=0 entries.
2. Run the transformer blocks manually at pos=1 with token 1902 (" world"),
   dumping each layer's output to `/tmp/hesper_dump/forward_pos1_layerN_out.bin`.

Then `scripts/compare_pos1.py` compares each dump against the second half of
the corresponding llama.cpp `/tmp/llama_dump/l_out-N.bin` (which is shaped
`[hiddenSize, 2]` because llama was invoked with prompt "Hello world" — the
first half is pos=0, the second half is pos=1).
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
  let state ← createInferenceState device cfg

  -- Step 1: run pos=0 with token 9259 ("Hello") to populate the KV caches
  IO.println "[pos=0] Running forwardSingleToken(token=9259, pos=0)"
  forwardSingleToken device model 9259 0 state

  -- Step 2: run pos=1 with token 1902 (" world"), dumping each layer output
  IO.println "[pos=1] Running per-layer forward manually (token=1902, pos=1)"
  let tokenId := 1902
  let pos := 1

  -- Embedding lookup: tokenBuf ← 1902, then buf1 ← embed[tokenBuf]
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

  -- Embedding scale buf1 → buf2
  let scaleBufs : List (String × Buffer) := [("input", state.buf1), ("output", state.buf2)]
  Hesper.WGSL.Execute.executeShaderNamed device
    (Hesper.Models.Gemma4.embeddingScaleKernel cfg.hiddenSize cfg.hiddenSize)
    scaleBufs (Hesper.WGSL.Execute.ExecutionConfig.dispatch1D cfg.hiddenSize)
  dumpBuf device "/tmp/hesper_dump/forward_pos1_inp_scaled.bin" state.buf2 cfg.hiddenSize

  -- Per-layer input precompute (same as forwardSingleToken / AllLayersDump)
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

  -- Run all layers at pos=1, dumping each layer output
  let mut currentBuf := state.buf2
  let mut nextBuf := state.buf1
  let plInputBuf := if cfg.hasPerLayerEmbeddings then some state.plInputAll else none

  for hi : i in [0:cfg.numHiddenLayers] do
    if hb : i < model.blocks.size then
      let block := model.blocks[i]
      let plEmbd := if hp : i < model.perLayerBlocks.size then model.perLayerBlocks[i] else none
      forwardBlock device block cfg currentBuf nextBuf state pos plEmbd plInputBuf
      dumpBuf device s!"/tmp/hesper_dump/forward_pos1_layer{i}_out.bin" nextBuf cfg.hiddenSize
      let tmp := currentBuf
      currentBuf := nextBuf
      nextBuf := tmp

  IO.println "Done! Compare with: python3 scripts/compare_pos1.py"
