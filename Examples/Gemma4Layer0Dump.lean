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
def main : IO Unit := do
  let modelPath := "data/gemma-4-e4b-it-Q4_K_M.gguf"
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

  -- Step 18: V-norm (bare RMSNorm)
  let vDim := numKVHeads * headDim
  let vNormBufs : List (String × Buffer) :=
    [("input", state.vBuf), ("output", state.vBuf2)]
  Hesper.WGSL.Execute.executeShaderNamed device
    (Hesper.Models.Gemma4.bareRMSNormKernel vDim cfg.rmsNormEps)
    vNormBufs
    { numWorkgroups := (1, 1, 1), workgroupSize := { x := min vDim 256, y := 1, z := 1 } : Execute.ExecutionConfig }
  dumpBuffer device "step_18_v_norm" state.vBuf2 vDim

  IO.println "Done with steps 1-18! Run: python3 scripts/compare_layer0.py"
