import Hesper.Backend
import Hesper.Backend.WebGPU
import Hesper.Backend.CUDA
import Hesper.Layers.Linear
import Hesper.Layers.RMSNorm
import Hesper.Layers.Embedding
import Hesper.Quantization.Q4_K_M
import Hesper.Quantization.Q6KDequant
import Hesper.GGUF.Parser
import Hesper.GGUF.Loader
import Hesper.GGUF.Reader
import Hesper.Basic
import Hesper.Models.Gemma4.Config
import Hesper.Models.Gemma4.Types
import Hesper.Models.DiffusionGemma.Config

/-!
# DiffusionGemma GPU loader (backend-generic; runs on WebGPU/Metal)

Loads DiffusionGemma weights into the reused `Gemma4Model` structures (the
backbone is identical), so the existing Metal-capable kernels apply.
Differences handled here vs the Gemma 4 loader:
- `attention.head_count_kv` per-layer (via `DiffusionConfig`);
- global (full-attention) layers have **no `attn_v`** → V reuses the K
  projection (`wV := wK`);
- both per-layer scalars: `layer_output_scale` (canvas) +
  `enc_layer_output_scale` (prompt);
- MoE norm tensor names `pre_ffw_norm_2` / `post_ffw_norm_1` /
  `post_ffw_norm_2` (DiffusionGemma naming);
- LM head tied to `token_embd` (no `output.weight`).
-/

namespace Hesper.Models.DiffusionGemma

open Hesper.Layers
open Hesper.Models.Gemma4 (Config Gemma4Model Gemma4Block Gemma4Attention Gemma4FFN EmbdFormat LayerType)

/-- DiffusionGemma model = the reused Gemma4 backbone model + the diffusion
    decode parameters. -/
structure DiffusionGemmaModel (BufT CacheT : Type) where
  inner : Gemma4Model BufT CacheT
  dg : DiffusionConfig

private def uploadBuffer [GPUBackend β] (ctx : β) (data : ByteArray) : IO (GPUBackend.Buf β) := do
  let bufSize := if data.size == 0 then 4 else data.size
  let buf ← GPUBackend.allocBuffer ctx bufSize.toUSize
  if data.size > 0 then GPUBackend.writeBuffer ctx buf data
  return buf

/-- Upload a tensor body to a GPU buffer (WebGPU/Metal path: read the
    tensor's bytes via `getTensorData`).  Mirrors the Gemma 4 loader. -/
private def uploadTensor [GPUBackend β] (ctx : β) (gguf : Hesper.GGUF.GGUFFile)
    (name : String) : IO (GPUBackend.Buf β) := do
  let info ← match Hesper.GGUF.Loader.findTensor gguf name with
    | .ok i => pure i
    | .error e => throw (IO.userError e)
  let bytes := info.size
  let buf ← GPUBackend.allocBuffer ctx (if bytes == 0 then 4 else bytes.toUSize)
  if bytes > 0 then
    match Hesper.GGUF.Loader.getTensorData gguf name with
    | .ok (_, data) => GPUBackend.writeBuffer ctx buf data
    | .error e => throw (IO.userError e)
  return buf

private def loadLinear [GPUBackend β] (ctx : β) (gguf : Hesper.GGUF.GGUFFile)
    (name : String) (inDim outDim : Nat) :
    IO (Linear.LinearLayer (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)) := do
  let ti ← match Hesper.GGUF.Loader.findTensor gguf name with
    | .ok ti => pure ti
    | .error e => throw $ IO.userError e
  let quantFormat : Linear.QuantFormat := match ti.ggmlType with
    | .Q6_K => .Q6_K
    | _ => .Q4_K
  let weightBuf ← uploadTensor ctx gguf name
  let prepared ← GPUBackend.newCacheRef (β := β)
  let splitKBuf ← IO.mkRef none
  let splitKPartialPrepared ← GPUBackend.newCacheRef (β := β)
  let splitKReducePrepared ← GPUBackend.newCacheRef (β := β)
  let dp4aQ8Buf ← IO.mkRef none
  let dp4aQuantizePrepared ← GPUBackend.newCacheRef (β := β)
  let dp4aMatmulPrepared ← GPUBackend.newCacheRef (β := β)
  let dp4aBatchQuantizePrepared ← GPUBackend.newCacheRef (β := β)
  let dp4aBatchMatmulPrepared ← GPUBackend.newCacheRef (β := β)
  return {
    config := { inDim, outDim }, weightBuf, quantFormat, prepared
    splitKBuf, splitKPartialPrepared, splitKReducePrepared
    dp4aQ8Buf, dp4aQuantizePrepared, dp4aMatmulPrepared
    dp4aBatchQuantizePrepared, dp4aBatchMatmulPrepared
  }

private def loadNorm [GPUBackend β] (ctx : β) (gguf : Hesper.GGUF.GGUFFile)
    (name : String) (cfg : RMSNorm.Config) :
    IO (Option (RMSNorm.RMSNorm (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))) := do
  match Hesper.GGUF.Loader.findTensor gguf name with
  | .ok _ =>
    let d ← Hesper.GGUF.Loader.extractFloat32Tensor gguf name
    pure (some (← RMSNorm.create ctx cfg d))
  | .error _ => pure none

/-- Load a DiffusionGemma model from a parsed GGUF onto the GPU backend. -/
def DiffusionGemmaModel.fromGGUFData [GPUBackend β] (ctx : β) (gguf : Hesper.GGUF.GGUFFile) :
    IO (DiffusionGemmaModel (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)) := do
  let dg ← match DiffusionConfig.fromGGUF gguf with
    | .ok c => pure c
    | .error e => throw $ IO.userError s!"DiffusionGemma config parse error: {e}"
  let cfg := dg.base
  IO.println s!"[DiffusionGemma] {cfg.numHiddenLayers}L dim={cfg.hiddenSize} experts={cfg.numExperts}/{cfg.numExpertsUsed} canvas={dg.canvasLength}"

  -- Embedding (Q6_K on-the-fly dequant)
  let embConfig : Embedding.Config := { vocabSize := cfg.vocabSize, dim := cfg.hiddenSize }
  let embTensor ← match Hesper.GGUF.Loader.findTensor gguf "token_embd.weight" with
    | .ok ti => pure ti | .error e => throw $ IO.userError e
  let embBuf ← uploadTensor ctx gguf "token_embd.weight"
  let embedding : Embedding.Embedding (GPUBackend.Buf β) :=
    { config := embConfig, embeddingTable := embBuf, f16Table := none }
  let embdFormat := match embTensor.ggmlType with
    | .Q6_K => EmbdFormat.Q6_K | .Q4_K => EmbdFormat.Q4_K | .F16 => EmbdFormat.F16 | _ => EmbdFormat.F32

  let normCfg : RMSNorm.Config := { dim := cfg.hiddenSize, eps := cfg.rmsNormEps }

  let mut blocks : Array (Gemma4Block (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)) := #[]
  for li in [0:cfg.numHiddenLayers] do
    if li % 6 == 0 then IO.println s!"  loading layer {li}/{cfg.numHiddenLayers}..."
    let headDim := cfg.headDim li
    let numKVHeads := cfg.numKVHeads li
    let isFull := cfg.isFullAttention li
    let qDim := cfg.numAttentionHeads * headDim
    let kvDim := numKVHeads * headDim

    let some attnNorm ← loadNorm ctx gguf s!"blk.{li}.attn_norm.weight" normCfg | throw (IO.userError s!"missing attn_norm {li}")
    let some postAttnNorm ← loadNorm ctx gguf s!"blk.{li}.post_attention_norm.weight" normCfg | throw (IO.userError s!"missing post_attention_norm {li}")
    let some ffnNorm ← loadNorm ctx gguf s!"blk.{li}.ffn_norm.weight" normCfg | throw (IO.userError s!"missing ffn_norm {li}")
    let some postFFNNorm ← loadNorm ctx gguf s!"blk.{li}.post_ffw_norm.weight" normCfg | throw (IO.userError s!"missing post_ffw_norm {li}")

    let wQ ← loadLinear ctx gguf s!"blk.{li}.attn_q.weight" cfg.hiddenSize qDim
    let wK ← loadLinear ctx gguf s!"blk.{li}.attn_k.weight" cfg.hiddenSize kvDim
    -- global layers have no attn_v → V reuses the K projection
    let wV ← match Hesper.GGUF.Loader.findTensor gguf s!"blk.{li}.attn_v.weight" with
      | .ok _ => loadLinear ctx gguf s!"blk.{li}.attn_v.weight" cfg.hiddenSize kvDim
      | .error _ => pure wK
    let wO ← loadLinear ctx gguf s!"blk.{li}.attn_output.weight" qDim cfg.hiddenSize
    let qNormBuf ← uploadBuffer ctx (← Hesper.GGUF.Loader.extractFloat32Tensor gguf s!"blk.{li}.attn_q_norm.weight")
    let kNormBuf ← uploadBuffer ctx (← Hesper.GGUF.Loader.extractFloat32Tensor gguf s!"blk.{li}.attn_k_norm.weight")
    let fQ ← GPUBackend.newCacheRef (β := β); let fK ← GPUBackend.newCacheRef (β := β); let fV ← GPUBackend.newCacheRef (β := β)
    let attention : Gemma4Attention (GPUBackend.Buf β) (GPUBackend.CachedDispatch β) :=
      { wQ, wK, wV, wO, qNormWeight := qNormBuf, kNormWeight := kNormBuf
        fusedNormQPrepared := fQ, fusedNormKPrepared := fK, fusedNormVPrepared := fV }

    let ffnGate ← loadLinear ctx gguf s!"blk.{li}.ffn_gate.weight" cfg.hiddenSize cfg.intermediateSize
    let ffnUp ← loadLinear ctx gguf s!"blk.{li}.ffn_up.weight" cfg.hiddenSize cfg.intermediateSize
    let ffnDown ← loadLinear ctx gguf s!"blk.{li}.ffn_down.weight" cfg.intermediateSize cfg.hiddenSize
    let fGU ← GPUBackend.newCacheRef (β := β); let fNG ← GPUBackend.newCacheRef (β := β); let fNU ← GPUBackend.newCacheRef (β := β)
    let ffn : Gemma4FFN (GPUBackend.Buf β) (GPUBackend.CachedDispatch β) :=
      { gate := ffnGate, up := ffnUp, down := ffnDown, fusedGateUpPrepared := fGU,
        fusedNormGatePrepared := fNG, fusedNormUpPrepared := fNU }

    -- MoE (present on every DiffusionGemma layer)
    let routerW ← uploadTensor ctx gguf s!"blk.{li}.ffn_gate_inp.weight"
    let routerS ← match Hesper.GGUF.Loader.findTensor gguf s!"blk.{li}.ffn_gate_inp.scale" with
      | .ok _ => pure (some (← uploadTensor ctx gguf s!"blk.{li}.ffn_gate_inp.scale")) | .error _ => pure none
    let gateUpE ← uploadTensor ctx gguf s!"blk.{li}.ffn_gate_up_exps.weight"
    let downE ← uploadTensor ctx gguf s!"blk.{li}.ffn_down_exps.weight"
    let preN2 ← loadNorm ctx gguf s!"blk.{li}.pre_ffw_norm_2.weight" normCfg
    let postN1 ← loadNorm ctx gguf s!"blk.{li}.post_ffw_norm_1.weight" normCfg
    let postN2 ← loadNorm ctx gguf s!"blk.{li}.post_ffw_norm_2.weight" normCfg

    -- RoPE freqs (global tensor; full-attention layers use it)
    let ropeFreqFactors ← if isFull then
      match Hesper.GGUF.Loader.findTensor gguf "rope_freqs.weight" with
      | .ok _ => pure (some (← uploadTensor ctx gguf "rope_freqs.weight")) | .error _ => pure none
      else pure none
    let outScale ← match Hesper.GGUF.Loader.findTensor gguf s!"blk.{li}.layer_output_scale.weight" with
      | .ok _ => pure (some (← uploadTensor ctx gguf s!"blk.{li}.layer_output_scale.weight")) | .error _ => pure none
    let encOutScale ← match Hesper.GGUF.Loader.findTensor gguf s!"blk.{li}.enc_layer_output_scale.weight" with
      | .ok _ => pure (some (← uploadTensor ctx gguf s!"blk.{li}.enc_layer_output_scale.weight")) | .error _ => pure none

    blocks := blocks.push {
      layerIdx := li, layerType := if isFull then LayerType.full else LayerType.swa
      attnNorm, postAttnNorm, ffnNorm, postFFNNorm, attention, ffn
      isMoE := true
      moeRouterWeight := some routerW, moeRouterScale := routerS
      moeGateUpExps := some gateUpE, moeDownExps := some downE
      moePreNorm2 := preN2, moePostNorm1 := postN1, moePostNorm2 := postN2
      ropeFreqFactors, outScale, encOutScale
    }

  let some finalNorm ← loadNorm ctx gguf "output_norm.weight" normCfg | throw (IO.userError "missing output_norm")
  -- LM head tied to token_embd (no output.weight)
  let outputWeight := embBuf

  let inner : Gemma4Model (GPUBackend.Buf β) (GPUBackend.CachedDispatch β) := {
    config := cfg, embedding, embdFormat, blocks, finalNorm, outputWeight
    outputWeightF16 := none
    perLayerEmbdMmap := none, perLayerEmbdTableGPU := none, perLayerEmbdRowBytes := 0
    perLayerModelProj := none, perLayerProjNorm := none, perLayerBlocks := #[]
  }
  IO.println s!"[DiffusionGemma] ✓ loaded {blocks.size} blocks"
  return { inner, dg }

/-- Load a DiffusionGemma model from a GGUF file path. -/
def DiffusionGemmaModel.fromGGUF [GPUBackend β] (ctx : β) (path : String) :
    IO (DiffusionGemmaModel (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)) := do
  IO.println s!"[DiffusionGemma] loading {path} (full read; ~16GB)..."
  let gguf ← Hesper.GGUF.loadGGUF path
  DiffusionGemmaModel.fromGGUFData ctx gguf

end Hesper.Models.DiffusionGemma
