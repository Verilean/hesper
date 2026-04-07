import Hesper
import Hesper.Models.Gemma4
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WebGPU.BufferOps

/-!
# Gemma 4 — Run forwardBlock for layer 0 only

This uses the actual `forwardBlock` (not manual kernel calls) to verify that
the production forward pass produces the same result as the manually-stepped
dump. Compares against `l_out-0` from llama.cpp.
-/

open Hesper.WebGPU
open Hesper.Models.Gemma4

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

  -- Set up the embedding (same as forwardSingleToken does)
  let tokenBytes := Hesper.WebGPU.BufferOps.uint32ToBytes tokenId.toUInt32
  writeBuffer device state.tokenBuf 0 tokenBytes
  match model.embdFormat with
  | .Q6_K =>
    let embdBufs : List (String × Buffer) :=
      [("token_ids", state.tokenBuf), ("embedding_table", model.embedding.embeddingTable), ("output", state.buf1)]
    Hesper.WGSL.Execute.executeShaderNamed device
      (Hesper.Quantization.Q6_K.q6kEmbeddingLookupKernel cfg.vocabSize cfg.hiddenSize)
      embdBufs
      (Hesper.WGSL.Execute.ExecutionConfig.dispatch1D cfg.hiddenSize)
  | _ =>
    Hesper.Layers.Embedding.forward device model.embedding state.tokenBuf state.buf1 1 1

  -- Embedding scale: buf1 → buf2
  let scaleBufs : List (String × Buffer) := [("input", state.buf1), ("output", state.buf2)]
  Hesper.WGSL.Execute.executeShaderNamed device
    (Hesper.Models.Gemma4.embeddingScaleKernel cfg.hiddenSize cfg.hiddenSize)
    scaleBufs
    (Hesper.WGSL.Execute.ExecutionConfig.dispatch1D cfg.hiddenSize)

  -- Per-layer input precompute (same as forwardSingleToken)
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
    let scaleBufs2 : List (String × Buffer) :=
      [("input", state.plTokenSelected), ("output", state.plInputAll)]
    Hesper.WGSL.Execute.executeShaderNamed device
      (Hesper.Layers.PerLayerEmbedding.scaleKernel totalPL (1.0 / Float.sqrt cfg.hiddenSize.toFloat))
      scaleBufs2
      (Hesper.WGSL.Execute.ExecutionConfig.dispatch1D totalPL)
    let chunkedNormBufs : List (String × Buffer) :=
      [("input", state.plInputAll), ("weight", projNorm.scale), ("output", state.plTokenSelected)]
    Hesper.WGSL.Execute.executeShaderNamed device
      (Hesper.Models.Gemma4.chunkedRMSNormKernel embdPL nLayers cfg.rmsNormEps)
      chunkedNormBufs
      { numWorkgroups := (nLayers, 1, 1), workgroupSize := { x := min embdPL 256, y := 1, z := 1 } : Hesper.WGSL.Execute.ExecutionConfig }
    let addBufs : List (String × Buffer) :=
      [("a", state.plTokenSelected), ("b", state.plModelProj), ("output", state.plInputAll)]
    Hesper.WGSL.Execute.executeShaderNamed device
      (Hesper.Models.Gemma4.scaledAddKernel totalPL (1.0 / Float.sqrt 2.0))
      addBufs
      (Hesper.WGSL.Execute.ExecutionConfig.dispatch1D totalPL)
  | _, _, _ => pure ()

  -- Now run forwardBlock for layer 0 ONLY (skip the rest)
  if h : 0 < model.blocks.size then
    let block := model.blocks[0]
    let plEmbd := if h2 : 0 < model.perLayerBlocks.size then model.perLayerBlocks[0] else none
    let plInputBuf := if cfg.hasPerLayerEmbeddings then some state.plInputAll else none

    -- forwardBlock: input=buf2 (scaled embed), output=buf1
    forwardBlock device block cfg state.buf2 state.buf1 state pos plEmbd plInputBuf

    -- Dump buf1 (= layer 0 output)
    let path := "/tmp/hesper_dump/forward_layer0_out.bin"
    let data ← Hesper.WebGPU.BufferOps.downloadFloatArray device state.buf1 cfg.hiddenSize
    let mut bytes := ByteArray.empty
    for v in data do
      let fb ← Hesper.Basic.floatToBytes v
      bytes := bytes ++ fb
    IO.FS.writeBinFile path bytes
    IO.println s!"[forward_layer0_out] size={data.size}, first4={data.toList.take 4}"
    IO.println s!"  Saved to {path}"
    IO.println "Compare with: python3 -c 'import struct; a=open(\"/tmp/llama_dump/l_out-0.bin\",\"rb\").read(); b=open(\"/tmp/hesper_dump/forward_layer0_out.bin\",\"rb\").read(); fa=struct.unpack(f\"<{len(a)//4}f\",a); fb=struct.unpack(f\"<{len(b)//4}f\",b); import math; d=sum(x*y for x,y in zip(fa,fb)); na=math.sqrt(sum(x*x for x in fa)); nb=math.sqrt(sum(x*x for x in fb)); print(f\"cosine={d/(na*nb):.6f}\")'"
  else
    IO.println "no blocks"
