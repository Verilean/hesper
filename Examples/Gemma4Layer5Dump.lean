import Hesper
import Hesper.Models.Gemma4
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WebGPU.BufferOps

/-!
# Gemma 4 — Dump layer 5 intermediates (first FULL attention layer)

Loads `l_out-4.bin` (the input to layer 5) from llama.cpp's dumps,
then manually runs each step of layer 5's forwardBlock equivalent,
saving intermediate values to /tmp/hesper_dump/layer5_*.bin.

This isolates layer 5 from accumulated drift in earlier layers,
so any divergence is intrinsic to layer 5's computation.
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
  let first4 := data.toList.take 4
  IO.println s!"  {path}: size={data.size}, first4={first4}"

def loadF32File (path : String) : IO ByteArray := IO.FS.readBinFile path

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

  -- Load layer 4's output from llama.cpp as the input to layer 5
  IO.println "[Setup] Loading l_out-4.bin from llama dump..."
  let layer4Out ← loadF32File "/tmp/llama_dump/l_out-4.bin"
  IO.println s!"  Loaded {layer4Out.size} bytes"

  -- Upload to state.buf2 (which forwardBlock uses as input via the layer input)
  writeBuffer device state.buf2 0 layer4Out
  let layerInput := state.buf2

  -- Pre-populate per-layer input (needed since layer 5 uses it)
  -- Same as forwardSingleToken's per-layer precompute
  let tokenBytes := Hesper.WebGPU.BufferOps.uint32ToBytes tokenId.toUInt32
  writeBuffer device state.tokenBuf 0 tokenBytes

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
    -- We need the embedding-scaled version of the original token to compute
    -- per_layer_model_proj @ embed. Recompute from token_embd.
    let embdBufs : List (String × Buffer) :=
      [("token_ids", state.tokenBuf), ("embedding_table", model.embedding.embeddingTable), ("output", state.buf1)]
    Hesper.WGSL.Execute.executeShaderNamed device
      (Hesper.Quantization.Q6_K.q6kEmbeddingLookupKernel cfg.vocabSize cfg.hiddenSize)
      embdBufs (Hesper.WGSL.Execute.ExecutionConfig.dispatch1D cfg.hiddenSize)
    -- buf1 has raw embed; we need scaled embed for per_layer_model_proj input
    -- Use normedBuf as temp
    let scaleBufs : List (String × Buffer) := [("input", state.buf1), ("output", state.normedBuf)]
    Hesper.WGSL.Execute.executeShaderNamed device
      (Hesper.Models.Gemma4.embeddingScaleKernel cfg.hiddenSize cfg.hiddenSize)
      scaleBufs (Hesper.WGSL.Execute.ExecutionConfig.dispatch1D cfg.hiddenSize)
    let projConfig : Hesper.WGSL.MatMul.Config := {
      M := 1, N := totalPL, K := cfg.hiddenSize
    }
    Hesper.WGSL.MatMul.executeMatMulTransposeF16 device state.normedBuf modelProj state.plTokenSelected projConfig
    let bufs2 : List (String × Buffer) := [("input", state.plTokenSelected), ("output", state.plInputAll)]
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

  -- Now layer 5 setup
  let li := 5
  if hb : li < model.blocks.size then
    let block := model.blocks[li]
    let layerType := block.layerType
    let headDim := cfg.headDim li
    let numHeads := cfg.numAttentionHeads
    let numKVHeads := cfg.numKVHeads li
    let layerTypeStr := match layerType with | .full => "FULL" | .swa => "swa"
    IO.println s!"[Layer 5] type={layerTypeStr}, headDim={headDim}, numHeads={numHeads}, numKVHeads={numKVHeads}"
    let rffStr := if block.ropeFreqFactors.isSome then "present" else "absent"
    IO.println s!"  ropeFreqFactors={rffStr}"
    IO.println s!"  Expected: type=FULL, headDim=512, ropeFreqFactors=present"

    -- Step 10: attn_norm(layer4_output)
    Hesper.Layers.RMSNorm.forward device block.attnNorm layerInput state.normedBuf
    dumpBuf device "/tmp/hesper_dump/layer5_step_10_attn_norm.bin" state.normedBuf cfg.hiddenSize

    -- Step 11: Q projection
    Hesper.Layers.Linear.LinearLayer.forward device block.attention.wQ state.normedBuf state.qBuf
    dumpBuf device "/tmp/hesper_dump/layer5_step_11_q_proj.bin" state.qBuf (numHeads * headDim)

    -- Step 12: Q-norm (per-head RMSNorm)
    let wgSizeNorm := min headDim 256
    let qNormBufs : List (String × Buffer) :=
      [("input", state.qBuf), ("weight", block.attention.qNormWeight), ("output", state.qBuf2)]
    Hesper.WGSL.Execute.executeShaderNamed device
      (Hesper.Models.Gemma4.perHeadRMSNormKernel numHeads headDim cfg.rmsNormEps)
      qNormBufs
      { numWorkgroups := (numHeads, 1, 1), workgroupSize := { x := wgSizeNorm, y := 1, z := 1 } : Hesper.WGSL.Execute.ExecutionConfig }
    dumpBuf device "/tmp/hesper_dump/layer5_step_12_q_norm.bin" state.qBuf2 (numHeads * headDim)

    -- Step 13: RoPE on Q (qBuf2 → qBuf)
    let posBytes := Hesper.WebGPU.BufferOps.uint32ToBytes pos.toUInt32
    writeBuffer device state.paramsBuf 0 posBytes
    match block.ropeFreqFactors with
    | some freqFactors =>
      IO.println "  Using ropeWithFreqFactorsKernel (full attention)"
      let ropeBufs : List (String × Buffer) :=
        [("input", state.qBuf2), ("output", state.qBuf), ("params", state.paramsBuf), ("freq_factors", freqFactors)]
      Hesper.WGSL.Execute.executeShaderNamed device
        (Hesper.Models.Gemma4.ropeWithFreqFactorsKernel headDim numHeads cfg.ropeTheta)
        ropeBufs
        (Hesper.WGSL.Execute.ExecutionConfig.dispatch1D (numHeads * headDim / 2))
    | none =>
      IO.println "  Using standard RoPE (NO freq_factors — unexpected for full layer!)"
      let ropeConfig : Hesper.Layers.RoPE.Config := { dim := numHeads * headDim, maxSeqLen := cfg.maxSeqLen, base := cfg.ropeTheta }
      let ropeBufs : List (String × Buffer) :=
        [("input", state.qBuf2), ("output", state.qBuf), ("params", state.paramsBuf)]
      Hesper.WGSL.Execute.executeShaderNamed device
        (Hesper.Layers.RoPE.ropeKernelDynamic ropeConfig 1 1 numHeads headDim)
        ropeBufs (Hesper.WGSL.Execute.ExecutionConfig.dispatch1D (numHeads * headDim / 2))
    dumpBuf device "/tmp/hesper_dump/layer5_step_13_q_rope.bin" state.qBuf (numHeads * headDim)

    -- Step 14: K projection
    Hesper.Layers.Linear.LinearLayer.forward device block.attention.wK state.normedBuf state.kBuf
    dumpBuf device "/tmp/hesper_dump/layer5_step_14_k_proj.bin" state.kBuf (numKVHeads * headDim)

    -- Step 15: K-norm
    let kNormBufs : List (String × Buffer) :=
      [("input", state.kBuf), ("weight", block.attention.kNormWeight), ("output", state.kBuf2)]
    Hesper.WGSL.Execute.executeShaderNamed device
      (Hesper.Models.Gemma4.perHeadRMSNormKernel numKVHeads headDim cfg.rmsNormEps)
      kNormBufs
      { numWorkgroups := (numKVHeads, 1, 1), workgroupSize := { x := wgSizeNorm, y := 1, z := 1 } : Hesper.WGSL.Execute.ExecutionConfig }
    dumpBuf device "/tmp/hesper_dump/layer5_step_15_k_norm.bin" state.kBuf2 (numKVHeads * headDim)

    -- Step 16: RoPE on K
    -- For full attention, K should also use freq_factors? Actually no — looking at gemma4-iswa.cpp:
    -- Both Q and K use the same freq_factors (line 87: ggml_rope_ext for K with freq_factors).
    match block.ropeFreqFactors with
    | some freqFactors =>
      let ropeBufs : List (String × Buffer) :=
        [("input", state.kBuf2), ("output", state.kBuf), ("params", state.paramsBuf), ("freq_factors", freqFactors)]
      Hesper.WGSL.Execute.executeShaderNamed device
        (Hesper.Models.Gemma4.ropeWithFreqFactorsKernel headDim numKVHeads cfg.ropeTheta)
        ropeBufs (Hesper.WGSL.Execute.ExecutionConfig.dispatch1D (numKVHeads * headDim / 2))
    | none =>
      let ropeConfig : Hesper.Layers.RoPE.Config := { dim := numKVHeads * headDim, maxSeqLen := cfg.maxSeqLen, base := cfg.ropeTheta }
      let ropeBufs : List (String × Buffer) :=
        [("input", state.kBuf2), ("output", state.kBuf), ("params", state.paramsBuf)]
      Hesper.WGSL.Execute.executeShaderNamed device
        (Hesper.Layers.RoPE.ropeKernelDynamic ropeConfig 1 1 numKVHeads headDim)
        ropeBufs (Hesper.WGSL.Execute.ExecutionConfig.dispatch1D (numKVHeads * headDim / 2))
    dumpBuf device "/tmp/hesper_dump/layer5_step_16_k_rope.bin" state.kBuf (numKVHeads * headDim)

    -- Step 17: V projection
    Hesper.Layers.Linear.LinearLayer.forward device block.attention.wV state.normedBuf state.vBuf
    dumpBuf device "/tmp/hesper_dump/layer5_step_17_v_proj.bin" state.vBuf (numKVHeads * headDim)

    -- Step 18+: run the full forwardBlock for layer 5 with l_out-4.bin as input
    -- and dump the final output. This isolates whether attn/ffn steps after V proj
    -- are the source of the divergence.
    let plEmbd := if hp : li < model.perLayerBlocks.size then model.perLayerBlocks[li] else none
    let plInputBuf := if cfg.hasPerLayerEmbeddings then some state.plInputAll else none
    -- Reload l_out-4.bin into a fresh buffer (state.buf2) since steps 10-17 may have used it
    writeBuffer device state.buf2 0 layer4Out
    forwardBlock device block cfg state.buf2 state.buf1 state pos plEmbd plInputBuf
    dumpBuf device "/tmp/hesper_dump/layer5_full_out.bin" state.buf1 cfg.hiddenSize
    IO.println "Done with layer 5 dump (steps 10-17 + full forwardBlock)!"
  else
    IO.println "no layer 5"
