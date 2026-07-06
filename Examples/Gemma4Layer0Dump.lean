import Hesper
import Hesper.Models.Gemma4
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WebGPU.BufferOps

/-!
# Gemma 4 Layer 0 Intermediate Dump

Runs a single-token forward pass and dumps intermediate values from layer 0
to disk for comparison against scripts/layer0_reference.py output.

Usage:
  python3 scripts/layer0_reference.py                    # generates reference
  lake exe gemma4-layer0-dump                            # dumps Hesper values
  python3 scripts/compare_layer0.py                      # compares

Dumps to /tmp/hesper_dump/step_XX_NAME.bin
-/

open Hesper.WebGPU

def dumpBuffer (device : Device) (name : String) (buf : Buffer) (numElements : Nat) : IO Unit := do
  let path := s!"/tmp/hesper_dump/{name}.bin"
  let data ← Hesper.WebGPU.BufferOps.downloadFloatArray device buf numElements
  -- Write as f32 LE
  let mut bytes := ByteArray.empty
  for v in data do
    let fb ← Hesper.Basic.floatToBytes v
    bytes := bytes ++ fb
  IO.FS.writeBinFile path bytes
  let first4 := data.toList.take 4
  IO.println s!"  {name}: size={data.size}, first4={first4}"

open Hesper.Models.Gemma4
open Hesper.WGSL

/-- Run kernels manually to dump layer 0 intermediate values.
    We replicate the steps of forwardSingleToken + forwardBlock for layer 0 only,
    saving each intermediate value to disk. -/
def main (args : List String) : IO Unit := do
  let modelPath := args.getD 0 "data/gemma-4-e4b-it-Q4_K_M.gguf"
  IO.FS.createDirAll "/tmp/hesper_dump"

  IO.println "[Init] Creating WebGPU device..."
  let inst ← Hesper.init
  let device ← Hesper.WebGPU.getDevice inst

  IO.println s!"[Load] Loading model from {modelPath}..."
  let model ← Hesper.Models.Gemma4.Gemma4Model.fromGGUF device modelPath

  let tokenId := 9259
  let pos := 0
  let cfg := model.config

  let state ← Hesper.Models.Gemma4.createInferenceState device model.config

  -- ====================================================================
  -- Step 1: Embedding lookup (Q6_K)
  -- ====================================================================
  let tokenBytes := Hesper.WebGPU.BufferOps.uint32ToBytes tokenId.toUInt32
  writeBuffer device state.tokenBuf 0 tokenBytes
  let embdBufs : List (String × Buffer) :=
    [("token_ids", state.tokenBuf), ("embedding_table", model.embedding.embeddingTable), ("output", state.buf1)]
  Hesper.WGSL.Execute.executeShaderNamed device
    (Hesper.Quantization.Q6_K.q6kEmbeddingLookupKernel cfg.vocabSize cfg.hiddenSize)
    embdBufs
    (Hesper.WGSL.Execute.ExecutionConfig.dispatch1D cfg.hiddenSize)
  dumpBuffer device "step_01_embed" state.buf1 cfg.hiddenSize

  -- Step 2: Embedding scale by sqrt(hiddenSize)
  let scaleBufs : List (String × Buffer) :=
    [("input", state.buf1), ("output", state.buf2)]
  Hesper.WGSL.Execute.executeShaderNamed device
    (Hesper.Models.Gemma4.embeddingScaleKernel cfg.hiddenSize cfg.hiddenSize)
    scaleBufs
    (Hesper.WGSL.Execute.ExecutionConfig.dispatch1D cfg.hiddenSize)
  dumpBuffer device "step_02_embed_scaled" state.buf2 cfg.hiddenSize

  -- Now state.buf2 contains the scaled embedding (the input to layer 0)
  let layerInput := state.buf2

  -- Layer 0 setup
  if h : model.blocks.size = 0 then
    throw $ IO.userError "no blocks loaded"
  else
  let block := model.blocks[0]'(by omega)
  let li := 0
  let layerType := block.layerType
  let headDim := cfg.headDim li
  let numHeads := cfg.numAttentionHeads
  let numKVHeads := cfg.numKVHeads li
  let layerTypeStr := match layerType with | .full => "full" | .swa => "swa"
  IO.println s!"[Layer 0] type={layerTypeStr}, headDim={headDim}, numHeads={numHeads}, numKVHeads={numKVHeads}"

  -- ====================================================================
  -- Step 10: attn_norm
  -- ====================================================================
  Hesper.Layers.RMSNorm.forward device block.attnNorm layerInput state.normedBuf
  dumpBuffer device "step_10_attn_norm" state.normedBuf cfg.hiddenSize

  -- ====================================================================
  -- Step 11: Q projection
  -- ====================================================================
  Hesper.Layers.Linear.LinearLayer.forward device block.attention.wQ state.normedBuf state.qBuf
  dumpBuffer device "step_11_q_proj" state.qBuf (numHeads * headDim)

  -- Step 12: Q-norm (per-head RMSNorm)
  let wgSizeNorm := min headDim 256
  let qNormBufs : List (String × Buffer) :=
    [("input", state.qBuf), ("weight", block.attention.qNormWeight), ("output", state.qBuf2)]
  Hesper.WGSL.Execute.executeShaderNamed device
    (Hesper.Models.Gemma4.perHeadRMSNormKernel numHeads headDim cfg.rmsNormEps)
    qNormBufs
    { numWorkgroups := (numHeads, 1, 1), workgroupSize := { x := wgSizeNorm, y := 1, z := 1 } : Execute.ExecutionConfig }
  dumpBuffer device "step_12_q_norm" state.qBuf2 (numHeads * headDim)

  -- Step 13: RoPE on Q (qBuf2 → qBuf)
  let posBytes := Hesper.WebGPU.BufferOps.uint32ToBytes pos.toUInt32
  writeBuffer device state.paramsBuf 0 posBytes
  match block.ropeFreqFactors with
  | some freqFactors =>
    let ropeBufs : List (String × Buffer) :=
      [("input", state.qBuf2), ("output", state.qBuf), ("params", state.paramsBuf), ("freq_factors", freqFactors)]
    Hesper.WGSL.Execute.executeShaderNamed device
      (Hesper.Models.Gemma4.ropeWithFreqFactorsKernel headDim numHeads cfg.ropeTheta)
      ropeBufs
      (.dispatch1D (numHeads * headDim / 2))
  | none =>
    let ropeConfig : Hesper.Layers.RoPE.Config := { dim := numHeads * headDim, maxSeqLen := cfg.maxSeqLen, base := cfg.ropeTheta }
    let ropeBufs : List (String × Buffer) :=
      [("input", state.qBuf2), ("output", state.qBuf), ("params", state.paramsBuf)]
    Hesper.WGSL.Execute.executeShaderNamed device
      (Hesper.Layers.RoPE.ropeKernelDynamic ropeConfig 1 1 numHeads headDim)
      ropeBufs
      (.dispatch1D (numHeads * headDim / 2))
  dumpBuffer device "step_13_q_rope" state.qBuf (numHeads * headDim)

  -- ====================================================================
  -- Step 14: K projection
  -- ====================================================================
  Hesper.Layers.Linear.LinearLayer.forward device block.attention.wK state.normedBuf state.kBuf
  dumpBuffer device "step_14_k_proj" state.kBuf (numKVHeads * headDim)

  -- Step 15: K-norm
  let kNormBufs : List (String × Buffer) :=
    [("input", state.kBuf), ("weight", block.attention.kNormWeight), ("output", state.kBuf2)]
  Hesper.WGSL.Execute.executeShaderNamed device
    (Hesper.Models.Gemma4.perHeadRMSNormKernel numKVHeads headDim cfg.rmsNormEps)
    kNormBufs
    { numWorkgroups := (numKVHeads, 1, 1), workgroupSize := { x := wgSizeNorm, y := 1, z := 1 } : Execute.ExecutionConfig }
  dumpBuffer device "step_15_k_norm" state.kBuf2 (numKVHeads * headDim)

  -- Step 16: RoPE on K
  let kRopeConfig : Hesper.Layers.RoPE.Config := { dim := numKVHeads * headDim, maxSeqLen := cfg.maxSeqLen, base := cfg.ropeTheta }
  let kRopeBufs : List (String × Buffer) :=
    [("input", state.kBuf2), ("output", state.kBuf), ("params", state.paramsBuf)]
  Hesper.WGSL.Execute.executeShaderNamed device
    (Hesper.Layers.RoPE.ropeKernelDynamic kRopeConfig 1 1 numKVHeads headDim)
    kRopeBufs
    (.dispatch1D (numKVHeads * headDim / 2))
  dumpBuffer device "step_16_k_rope" state.kBuf (numKVHeads * headDim)

  -- ====================================================================
  -- Step 17: V projection
  -- ====================================================================
  Hesper.Layers.Linear.LinearLayer.forward device block.attention.wV state.normedBuf state.vBuf
  dumpBuffer device "step_17_v_proj" state.vBuf (numKVHeads * headDim)

  -- Step 18: V-norm (bare per-head RMSNorm — each head normalized independently)
  let vDim := numKVHeads * headDim
  let vNormBufs : List (String × Buffer) :=
    [("input", state.vBuf), ("output", state.vBuf2)]
  Hesper.WGSL.Execute.executeShaderNamed device
    (Hesper.Models.Gemma4.perHeadBareRMSNormKernel numKVHeads headDim cfg.rmsNormEps)
    vNormBufs
    { numWorkgroups := (numKVHeads, 1, 1), workgroupSize := { x := min headDim 256, y := 1, z := 1 } : Execute.ExecutionConfig }
  dumpBuffer device "step_18_v_norm" state.vBuf2 vDim

  -- ====================================================================
  -- Steps 19-21: KV cache write + Flash Attention
  -- ====================================================================
  -- Write K and V to per-layer cache (using kvCache from state)
  if h2 : state.kvCaches.size = 0 then
    throw $ IO.userError "no kv caches"
  else
  let kvCache := state.kvCaches[0]'(by omega)
  let kvDim := numKVHeads * headDim
  let kvWriteBufs : List (String × Buffer) :=
    [("new_k", state.kBuf), ("new_v", state.vBuf2),
     ("k_cache", kvCache.kBuf), ("v_cache", kvCache.vBuf),
     ("params", state.paramsBuf)]
  Hesper.WGSL.Execute.executeShaderNamed device
    (Hesper.Layers.Attention.fusedCacheWriteKVKernel numKVHeads cfg.maxSeqLen headDim kvDim)
    kvWriteBufs
    (.dispatch1D kvDim)

  -- Flash attention for cacheLen=1
  let scale := 1.0 / Float.sqrt headDim.toFloat
  let attnBufs : List (String × Buffer) :=
    [("q", state.qBuf), ("k_cache", kvCache.kBuf), ("v_cache", kvCache.vBuf), ("output", state.attnOutBuf)]
  Hesper.WGSL.Execute.executeShaderNamed device
    (Hesper.WGSL.FlashAttention.flashAttentionSWAKernel numHeads numKVHeads cfg.maxSeqLen headDim 1 cfg.slidingWindowSize 0 scale)
    attnBufs
    (Hesper.WGSL.Execute.ExecutionConfig.default (numHeads, 1, 1))
  dumpBuffer device "step_21_attn_output" state.attnOutBuf (numHeads * headDim)

  -- Step 22: O projection
  Hesper.Layers.Linear.LinearLayer.forward device block.attention.wO state.attnOutBuf state.normedBuf
  dumpBuffer device "step_22_o_proj" state.normedBuf cfg.hiddenSize

  -- Step 23: post_attention_norm
  Hesper.Layers.RMSNorm.forward device block.postAttnNorm state.normedBuf state.normedBuf2
  dumpBuffer device "step_23_post_attn_norm" state.normedBuf2 cfg.hiddenSize

  -- Step 24: + residual (= attn_out)
  let resAttnBufs : List (String × Buffer) :=
    [("a", state.normedBuf2), ("b", layerInput), ("output", state.attnResidualBuf)]
  Hesper.WGSL.Execute.executeShaderNamed device
    (Hesper.Models.Gemma4.residualAddKernel cfg.hiddenSize)
    resAttnBufs
    (.dispatch1D cfg.hiddenSize)
  dumpBuffer device "step_24_attn_residual" state.attnResidualBuf cfg.hiddenSize

  -- ====================================================================
  -- Steps 30-34: GeGLU FFN
  -- ====================================================================
  Hesper.Layers.RMSNorm.forward device block.ffnNorm state.attnResidualBuf state.normedBuf
  dumpBuffer device "step_30_ffn_norm" state.normedBuf cfg.hiddenSize

  Hesper.Layers.Linear.LinearLayer.forward device block.ffn.gate state.normedBuf state.gateBuf
  dumpBuffer device "step_31_ffn_gate" state.gateBuf cfg.intermediateSize

  Hesper.Layers.Linear.LinearLayer.forward device block.ffn.up state.normedBuf state.upBuf
  dumpBuffer device "step_32_ffn_up" state.upBuf cfg.intermediateSize

  let geluBufs : List (String × Buffer) :=
    [("gate", state.gateBuf), ("up", state.upBuf), ("output", state.geluBuf)]
  Hesper.WGSL.Execute.executeShaderNamed device
    (Hesper.Models.Gemma4.geluMulKernel cfg.intermediateSize)
    geluBufs
    (.dispatch1D cfg.intermediateSize)
  dumpBuffer device "step_33_ffn_gelu" state.geluBuf cfg.intermediateSize

  Hesper.Layers.Linear.LinearLayer.forward device block.ffn.down state.geluBuf state.ffnOutBuf
  dumpBuffer device "step_34_ffn_down" state.ffnOutBuf cfg.hiddenSize

  -- Step 35: post_ffw_norm (ffnOutBuf → normedBuf2 to avoid aliasing)
  Hesper.Layers.RMSNorm.forward device block.postFFNNorm state.ffnOutBuf state.normedBuf2
  dumpBuffer device "step_35_post_ffn_norm" state.normedBuf2 cfg.hiddenSize

  -- Step 36: pe_in = ffn_post_norm + attn_residual → buf1 (we'll use buf1 as pe_in)
  let peInBufs : List (String × Buffer) :=
    [("a", state.normedBuf2), ("b", state.attnResidualBuf), ("output", state.buf1)]
  Hesper.WGSL.Execute.executeShaderNamed device
    (Hesper.Models.Gemma4.residualAddKernel cfg.hiddenSize)
    peInBufs
    (.dispatch1D cfg.hiddenSize)
  dumpBuffer device "step_36_pe_in" state.buf1 cfg.hiddenSize

  -- ====================================================================
  -- Steps 40-44: Per-layer embedding
  -- ====================================================================
  -- Run the per-layer input precompute first to populate state.plInputAll
  -- We need this for the layer 0 slice
  match model.perLayerEmbdTableBytes, model.perLayerModelProj, model.perLayerProjNorm with
  | some embdTableCPU, some modelProj, some projNorm =>
    let embdPL := cfg.embdPerLayer
    let nLayers := cfg.numHiddenLayers
    let totalPL := embdPL * nLayers
    -- 1) CPU dequant of per_layer_token_embd[token]
    let rowOffset := tokenId * model.perLayerEmbdRowBytes
    let rowFloats := Hesper.Models.Gemma4.dequantQ6KRowCPU embdTableCPU rowOffset totalPL
    let scaleFactor : Float := Float.sqrt embdPL.toFloat
    let scaledRow := rowFloats.map (· * scaleFactor)
    let rowBytes ← Hesper.Models.Gemma4.floatArrayToBytes scaledRow
    writeBuffer device state.plModelProj 0 rowBytes
    -- 2) per_layer_model_proj @ buf2 → plTokenSelected
    let projConfig : Hesper.WGSL.MatMul.Config := {
      M := 1, N := totalPL, K := cfg.hiddenSize
    }
    Hesper.WGSL.MatMul.executeMatMulTransposeF16 device state.buf2 modelProj state.plTokenSelected projConfig
    -- 3) Scale by 1/sqrt(hiddenSize) → plInputAll
    let scaleBufs2 : List (String × Buffer) :=
      [("input", state.plTokenSelected), ("output", state.plInputAll)]
    Hesper.WGSL.Execute.executeShaderNamed device
      (Hesper.Layers.PerLayerEmbedding.scaleKernel totalPL (1.0 / Float.sqrt cfg.hiddenSize.toFloat))
      scaleBufs2
      (.dispatch1D totalPL)
    -- 4) chunkedRMSNorm → plTokenSelected
    let chunkedNormBufs : List (String × Buffer) :=
      [("input", state.plInputAll), ("weight", projNorm.scale), ("output", state.plTokenSelected)]
    Hesper.WGSL.Execute.executeShaderNamed device
      (Hesper.Models.Gemma4.chunkedRMSNormKernel embdPL nLayers cfg.rmsNormEps)
      chunkedNormBufs
      { numWorkgroups := (nLayers, 1, 1), workgroupSize := { x := min embdPL 256, y := 1, z := 1 } : Execute.ExecutionConfig }
    -- 5) Scaled add → plInputAll
    let addBufs : List (String × Buffer) :=
      [("a", state.plTokenSelected), ("b", state.plModelProj), ("output", state.plInputAll)]
    Hesper.WGSL.Execute.executeShaderNamed device
      (Hesper.Models.Gemma4.scaledAddKernel totalPL (1.0 / Float.sqrt 2.0))
      addBufs
      (.dispatch1D totalPL)
  | _, _, _ => pure ()

  -- Now perform the per-layer block forward (steps 40-44)
  if h3 : model.perLayerBlocks.size = 0 then
    IO.println "  no per-layer blocks"
  else do
    let plEmbd := model.perLayerBlocks[0]'(by omega)
    match plEmbd with
    | some plEmbd =>
      -- Step 40: inp_gate(pe_in=buf1) → plGateBuf
      Hesper.Layers.Linear.LinearLayer.forward device plEmbd.inpGate state.buf1 state.plGateBuf
      dumpBuffer device "step_40_pl_gate" state.plGateBuf cfg.embdPerLayer

      -- Step 41: GELU(gate) * pl_input[layer 0] → moeRouterOutBuf (reuse as temp)
      let plOffset := 0 * cfg.embdPerLayer  -- layer 0
      let totalPL := cfg.embdPerLayer * cfg.numHiddenLayers
      let pl41Bufs : List (String × Buffer) :=
        [("gate", state.plGateBuf), ("per_layer_input", state.plInputAll), ("output", state.moeRouterOutBuf)]
      Hesper.WGSL.Execute.executeShaderNamed device
        (Hesper.Layers.PerLayerEmbedding.geluGateMulSliceKernel cfg.embdPerLayer totalPL plOffset)
        pl41Bufs
        (.dispatch1D cfg.embdPerLayer)
      dumpBuffer device "step_41_pl_gelu_mul" state.moeRouterOutBuf cfg.embdPerLayer

      -- Step 42: per_layer_proj
      Hesper.Layers.Linear.LinearLayer.forward device plEmbd.proj state.moeRouterOutBuf state.plProjBuf
      dumpBuffer device "step_42_pl_proj" state.plProjBuf cfg.hiddenSize

      -- Step 43: per_layer_post_norm → normedBuf2 (avoid aliasing)
      Hesper.Layers.RMSNorm.forward device plEmbd.postNorm state.plProjBuf state.normedBuf2
      dumpBuffer device "step_43_pl_post_norm" state.normedBuf2 cfg.hiddenSize

      -- Step 44: pe_in (buf1) + per_layer_embd_out (normedBuf2) → ffnOutBuf
      let res44Bufs : List (String × Buffer) :=
        [("a", state.buf1), ("b", state.normedBuf2), ("output", state.ffnOutBuf)]
      Hesper.WGSL.Execute.executeShaderNamed device
        (Hesper.Models.Gemma4.residualAddKernel cfg.hiddenSize)
        res44Bufs
        (.dispatch1D cfg.hiddenSize)
      dumpBuffer device "step_44_pl_residual" state.ffnOutBuf cfg.hiddenSize
    | none => pure ()

  -- Step 50: layer_output_scale
  match block.outScale with
  | some scale =>
    let scaleBufs3 : List (String × Buffer) :=
      [("input", state.ffnOutBuf), ("scale", scale), ("output", state.normedBuf2)]
    Hesper.WGSL.Execute.executeShaderNamed device
      (Hesper.Layers.PerLayerEmbedding.layerScaleKernel cfg.hiddenSize)
      scaleBufs3
      (.dispatch1D cfg.hiddenSize)
    dumpBuffer device "step_50_layer_scale" state.normedBuf2 cfg.hiddenSize
  | none => pure ()

  IO.println "Done with steps 1-50! Run: python3 scripts/compare_layer0.py"
