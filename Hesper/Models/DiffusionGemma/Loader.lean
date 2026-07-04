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
import Std.Data.HashMap
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
  -- self-conditioning MLP (global): pre_norm → GeGLU(gate,up) → down, fed the
  -- previous step's soft-embedded prediction during the denoise loop.
  scPreNorm : Option (RMSNorm.RMSNorm BufT CacheT) := none
  scGate : Option (Linear.LinearLayer BufT CacheT) := none
  scUp : Option (Linear.LinearLayer BufT CacheT) := none
  scDown : Option (Linear.LinearLayer BufT CacheT) := none

private def uploadBuffer [GPUBackend β] (ctx : β) (data : ByteArray) : IO (GPUBackend.Buf β) := do
  let bufSize := if data.size == 0 then 4 else data.size
  let buf ← GPUBackend.allocBuffer ctx bufSize.toUSize
  if data.size > 0 then GPUBackend.writeBuffer ctx buf data
  return buf

/-! ## Streaming loader (peak memory ≈ file size)

The legacy `loadGGUF` slurps the whole 15.7GB file into a heap ByteArray; that +
the wired GPU upload peaks >48GB → swaps on a 48GB Mac (2× slowdown + memory-
pressure garbage).  Instead: parse only the header prefix (`loadGGUFHeader`),
then stream tensor bodies in ascending file-offset order via a sequential
`IO.FS.Handle` (no mmap — stubbed on macOS; no seek — Lean 4.28 has none),
uploading each big weight to its GPU buffer immediately + keeping only the small
F32 norms as bytes.  Peak ≈ 15.7GB (one GPU copy) + one in-flight tensor. -/

/-- Read exactly `remaining` bytes (a single `Handle.read` may return fewer). -/
private partial def readExact (h : IO.FS.Handle) (remaining : Nat) (acc : ByteArray) : IO ByteArray := do
  if remaining == 0 then return acc
  let chunk ← h.read (min remaining (64 * 1024 * 1024)).toUSize
  if chunk.isEmpty then return acc  -- EOF
  readExact h (remaining - chunk.size) (acc ++ chunk)

/-- Read + discard `remaining` bytes (skip forward without seek). -/
private partial def skipBytes (h : IO.FS.Handle) (remaining : Nat) : IO Unit := do
  if remaining == 0 then return ()
  let chunk ← h.read (min remaining (64 * 1024 * 1024)).toUSize
  if chunk.isEmpty then return ()
  skipBytes h (remaining - chunk.size)

/-- Read to EOF (the last tensor's size = the remaining file bytes). -/
private partial def readToEnd (h : IO.FS.Handle) (acc : ByteArray) : IO ByteArray := do
  let chunk ← h.read (64 * 1024 * 1024)
  if chunk.isEmpty then return acc
  readToEnd h (acc ++ chunk)

/-- Tensors ≥ this upload to a GPU buffer immediately; smaller F32 tensors
    (norms/scales/rope) are kept as bytes and resolved on demand. -/
private def streamThreshold : Nat := 8 * 1024 * 1024

/-- Stream tensor bodies from `path` in file-offset order (sequential reads):
    big weights → GPU bufs, small tensors → kept bytes.  Peak ≈ one GPU copy +
    one in-flight tensor.  Sizes come from consecutive offset deltas — exactly
    what `findTensor`/the legacy upload use, so the bufs are byte-identical. -/
private def streamTensors [GPUBackend β] (ctx : β) (path : String) (gguf : Hesper.GGUF.GGUFFile) :
    IO ((Std.HashMap String (GPUBackend.Buf β)) × (Std.HashMap String ByteArray)) := do
  let h ← IO.FS.Handle.mk path .read
  let dso := gguf.dataSectionOffset.toNat
  skipBytes h dso                 -- advance to the aligned data section
  let ts := gguf.tensors           -- stored in ascending file-offset order
  let n := ts.size
  let mut bufs : Std.HashMap String (GPUBackend.Buf β) := ∅
  let mut f32s : Std.HashMap String ByteArray := ∅
  let mut cursor : Nat := 0         -- bytes consumed within the data section
  for idx in [0:n] do
    let ti := ts[idx]!
    let off := ti.offset.toNat
    if off > cursor then skipBytes h (off - cursor)   -- inter-tensor padding
    let data ← if idx + 1 < n then
        readExact h (ts[idx+1]!.offset.toNat - off) ByteArray.empty
      else
        readToEnd h ByteArray.empty
    if data.size ≥ streamThreshold then
      bufs := bufs.insert ti.name (← uploadBuffer ctx data)
    else
      f32s := f32s.insert ti.name data
    cursor := off + data.size
  return (bufs, f32s)

/-- Legacy A/B path: build the same maps from a fully-read gguf's `dataBlob`.
    Uses ~2× memory (dataBlob + bufs) — for correctness comparison only. -/
private def buildMapsFromBlob [GPUBackend β] (ctx : β) (gguf : Hesper.GGUF.GGUFFile) :
    IO ((Std.HashMap String (GPUBackend.Buf β)) × (Std.HashMap String ByteArray)) := do
  let mut bufs : Std.HashMap String (GPUBackend.Buf β) := ∅
  let mut f32s : Std.HashMap String ByteArray := ∅
  for ti in gguf.tensors do
    let (_, data) ← Hesper.GGUF.Loader.getTensorDataM gguf ti.name
    if data.size ≥ streamThreshold then
      bufs := bufs.insert ti.name (← uploadBuffer ctx data)
    else
      f32s := f32s.insert ti.name data
  return (bufs, f32s)

/-- Resolve a weight tensor to its GPU buffer: pre-uploaded (big) or uploaded
    on demand from the kept bytes (small scales / rope). -/
private def getBuf [GPUBackend β] (ctx : β)
    (bufs : Std.HashMap String (GPUBackend.Buf β)) (f32s : Std.HashMap String ByteArray)
    (name : String) : IO (GPUBackend.Buf β) := do
  match bufs[name]? with
  | some b => pure b
  | none => match f32s[name]? with
    | some data => uploadBuffer ctx data
    | none => throw (IO.userError s!"getBuf: tensor '{name}' not streamed")

/-- Resolve a small F32 tensor's raw bytes (norms). -/
private def getF32 (f32s : Std.HashMap String ByteArray) (name : String) : IO ByteArray := do
  match f32s[name]? with
  | some d => pure d
  | none => throw (IO.userError s!"getF32: F32 tensor '{name}' not streamed")

private def loadLinear [GPUBackend β] (ctx : β) (gguf : Hesper.GGUF.GGUFFile)
    (bufs : Std.HashMap String (GPUBackend.Buf β)) (f32s : Std.HashMap String ByteArray)
    (name : String) (inDim outDim : Nat) :
    IO (Linear.LinearLayer (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)) := do
  let ti ← match Hesper.GGUF.Loader.findTensor gguf name with
    | .ok ti => pure ti
    | .error e => throw $ IO.userError e
  -- NOTE: DiffusionGemma's ffn_down / ffn_down_exps are Q8_0, which
  -- Hesper.Layers.Linear does NOT yet support (only Q4_K/Q6_K).  Loading
  -- them as Q4_K mis-dequantizes → NaN.  Adding a Q8_0 dequant+matmul WGSL
  -- kernel is the gating item for the native forward (see task #5).
  -- ffn_down / ffn_down_exps are Q8_0 (inline-dequant kernel added to
  -- Hesper.Layers.Linear); everything else is Q4_K (or Q6_K for attn_v).
  let quantFormat : Linear.QuantFormat := match ti.ggmlType with
    | .Q6_K => .Q6_K
    | .Q8_0 => .Q8_0
    | .Q5_0 => .Q5_0
    | _ => .Q4_K
  let weightBuf ← getBuf ctx bufs f32s name
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
    (f32s : Std.HashMap String ByteArray)
    (name : String) (cfg : RMSNorm.Config) :
    IO (Option (RMSNorm.RMSNorm (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))) := do
  match Hesper.GGUF.Loader.findTensor gguf name with
  | .ok _ =>
    let d ← getF32 f32s name
    pure (some (← RMSNorm.create ctx cfg d))
  | .error _ => pure none

/-- Load a DiffusionGemma model from a parsed GGUF onto the GPU backend, given
    pre-resolved tensor maps (`bufs` = uploaded weights, `f32s` = small norm
    bytes) built by `streamTensors` (default) or `buildMapsFromBlob` (legacy). -/
def DiffusionGemmaModel.fromGGUFData [GPUBackend β] (ctx : β) (gguf : Hesper.GGUF.GGUFFile)
    (bufs : Std.HashMap String (GPUBackend.Buf β)) (f32s : Std.HashMap String ByteArray) :
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
  let embBuf ← getBuf ctx bufs f32s "token_embd.weight"
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

    let some attnNorm ← loadNorm ctx gguf f32s s!"blk.{li}.attn_norm.weight" normCfg | throw (IO.userError s!"missing attn_norm {li}")
    let some postAttnNorm ← loadNorm ctx gguf f32s s!"blk.{li}.post_attention_norm.weight" normCfg | throw (IO.userError s!"missing post_attention_norm {li}")
    let some ffnNorm ← loadNorm ctx gguf f32s s!"blk.{li}.ffn_norm.weight" normCfg | throw (IO.userError s!"missing ffn_norm {li}")
    let some postFFNNorm ← loadNorm ctx gguf f32s s!"blk.{li}.post_ffw_norm.weight" normCfg | throw (IO.userError s!"missing post_ffw_norm {li}")

    let wQ ← loadLinear ctx gguf bufs f32s s!"blk.{li}.attn_q.weight" cfg.hiddenSize qDim
    let wK ← loadLinear ctx gguf bufs f32s s!"blk.{li}.attn_k.weight" cfg.hiddenSize kvDim
    -- global layers have no attn_v → V reuses the K projection
    let wV ← match Hesper.GGUF.Loader.findTensor gguf s!"blk.{li}.attn_v.weight" with
      | .ok _ => loadLinear ctx gguf bufs f32s s!"blk.{li}.attn_v.weight" cfg.hiddenSize kvDim
      | .error _ => pure wK
    let wO ← loadLinear ctx gguf bufs f32s s!"blk.{li}.attn_output.weight" qDim cfg.hiddenSize
    let qNormBuf ← uploadBuffer ctx (← getF32 f32s s!"blk.{li}.attn_q_norm.weight")
    let kNormBuf ← uploadBuffer ctx (← getF32 f32s s!"blk.{li}.attn_k_norm.weight")
    let fQ ← GPUBackend.newCacheRef (β := β); let fK ← GPUBackend.newCacheRef (β := β); let fV ← GPUBackend.newCacheRef (β := β)
    let attention : Gemma4Attention (GPUBackend.Buf β) (GPUBackend.CachedDispatch β) :=
      { wQ, wK, wV, wO, qNormWeight := qNormBuf, kNormWeight := kNormBuf
        fusedNormQPrepared := fQ, fusedNormKPrepared := fK, fusedNormVPrepared := fV }

    let ffnGate ← loadLinear ctx gguf bufs f32s s!"blk.{li}.ffn_gate.weight" cfg.hiddenSize cfg.intermediateSize
    let ffnUp ← loadLinear ctx gguf bufs f32s s!"blk.{li}.ffn_up.weight" cfg.hiddenSize cfg.intermediateSize
    let ffnDown ← loadLinear ctx gguf bufs f32s s!"blk.{li}.ffn_down.weight" cfg.intermediateSize cfg.hiddenSize
    let fGU ← GPUBackend.newCacheRef (β := β); let fNG ← GPUBackend.newCacheRef (β := β); let fNU ← GPUBackend.newCacheRef (β := β)
    let ffn : Gemma4FFN (GPUBackend.Buf β) (GPUBackend.CachedDispatch β) :=
      { gate := ffnGate, up := ffnUp, down := ffnDown, fusedGateUpPrepared := fGU,
        fusedNormGatePrepared := fNG, fusedNormUpPrepared := fNU }

    -- MoE (present on every DiffusionGemma layer)
    let routerW ← getBuf ctx bufs f32s s!"blk.{li}.ffn_gate_inp.weight"
    let routerS ← match Hesper.GGUF.Loader.findTensor gguf s!"blk.{li}.ffn_gate_inp.scale" with
      | .ok _ => pure (some (← getBuf ctx bufs f32s s!"blk.{li}.ffn_gate_inp.scale")) | .error _ => pure none
    let gateUpE ← getBuf ctx bufs f32s s!"blk.{li}.ffn_gate_up_exps.weight"
    let downE ← getBuf ctx bufs f32s s!"blk.{li}.ffn_down_exps.weight"
    let preN2 ← loadNorm ctx gguf f32s s!"blk.{li}.pre_ffw_norm_2.weight" normCfg
    let postN1 ← loadNorm ctx gguf f32s s!"blk.{li}.post_ffw_norm_1.weight" normCfg
    let postN2 ← loadNorm ctx gguf f32s s!"blk.{li}.post_ffw_norm_2.weight" normCfg

    -- RoPE freqs (global tensor; full-attention layers use it)
    let ropeFreqFactors ← if isFull then
      match Hesper.GGUF.Loader.findTensor gguf "rope_freqs.weight" with
      | .ok _ => pure (some (← getBuf ctx bufs f32s "rope_freqs.weight")) | .error _ => pure none
      else pure none
    let outScale ← match Hesper.GGUF.Loader.findTensor gguf s!"blk.{li}.layer_output_scale.weight" with
      | .ok _ => pure (some (← getBuf ctx bufs f32s s!"blk.{li}.layer_output_scale.weight")) | .error _ => pure none
    let encOutScale ← match Hesper.GGUF.Loader.findTensor gguf s!"blk.{li}.enc_layer_output_scale.weight" with
      | .ok _ => pure (some (← getBuf ctx bufs f32s s!"blk.{li}.enc_layer_output_scale.weight")) | .error _ => pure none

    blocks := blocks.push {
      layerIdx := li, layerType := if isFull then LayerType.full else LayerType.swa
      attnNorm, postAttnNorm, ffnNorm, postFFNNorm, attention, ffn
      isMoE := true
      moeRouterWeight := some routerW, moeRouterScale := routerS
      moeGateUpExps := some gateUpE, moeDownExps := some downE
      moePreNorm2 := preN2, moePostNorm1 := postN1, moePostNorm2 := postN2
      ropeFreqFactors, outScale, encOutScale
    }

  let some finalNorm ← loadNorm ctx gguf f32s "output_norm.weight" normCfg | throw (IO.userError "missing output_norm")
  -- LM head tied to token_embd (no output.weight)
  let outputWeight := embBuf

  let inner : Gemma4Model (GPUBackend.Buf β) (GPUBackend.CachedDispatch β) := {
    config := cfg, embedding, embdFormat, blocks, finalNorm, outputWeight
    outputWeightF16 := none
    perLayerEmbdMmap := none, perLayerEmbdTableGPU := none, perLayerEmbdRowBytes := 0
    perLayerModelProj := none, perLayerProjNorm := none, perLayerBlocks := #[]
  }
  -- self-conditioning MLP (global, optional): pre_norm + gate/up (Q4_K) + down (Q5_0)
  let scPreNorm ← loadNorm ctx gguf f32s "self_cond_pre_norm.weight" normCfg
  let scGate ← (do pure (some (← loadLinear ctx gguf bufs f32s "self_cond_gate.weight" cfg.hiddenSize cfg.intermediateSize))) <|> pure none
  let scUp ← (do pure (some (← loadLinear ctx gguf bufs f32s "self_cond_up.weight" cfg.hiddenSize cfg.intermediateSize))) <|> pure none
  let scDown ← (do pure (some (← loadLinear ctx gguf bufs f32s "self_cond_down.weight" cfg.intermediateSize cfg.hiddenSize))) <|> pure none
  IO.println s!"[DiffusionGemma] ✓ loaded {blocks.size} blocks; SC={scGate.isSome}"
  return { inner, dg, scPreNorm, scGate, scUp, scDown }

/-- Load a DiffusionGemma model from a GGUF file path. -/
def DiffusionGemmaModel.fromGGUF [GPUBackend β] (ctx : β) (path : String) :
    IO (DiffusionGemmaModel (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)) := do
  -- Default = STREAMING loader: parse only the header prefix, then stream tensor
  -- bodies in file-offset order (sequential Handle reads) straight to GPU, keeping
  -- only small F32 norms as bytes.  Peak ≈ file size (one GPU copy) — no 15.7GB
  -- heap ByteArray, so no swap on a 48GB Mac (the swap caused the 2× slowdown +
  -- memory-pressure garbage).  DG_LEGACYLOAD forces the old whole-file read for
  -- A/B correctness comparison (uses ~2× memory → may swap).
  if (← IO.getEnv "DG_LEGACYLOAD").isSome then
    IO.println s!"[DiffusionGemma] loading {path} (LEGACY full read; ~16GB heap + GPU → may swap)..."
    let gguf ← Hesper.GGUF.loadGGUF path
    let (bufs, f32s) ← buildMapsFromBlob ctx gguf
    DiffusionGemmaModel.fromGGUFData ctx gguf bufs f32s
  else
    IO.println s!"[DiffusionGemma] loading {path} (streaming; peak mem ≈ file size)..."
    let gguf ← Hesper.GGUF.loadGGUFHeader path
    let (bufs, f32s) ← streamTensors ctx path gguf
    DiffusionGemmaModel.fromGGUFData ctx gguf bufs f32s

end Hesper.Models.DiffusionGemma
