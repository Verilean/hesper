import Hesper.GGUF.Parser
import Hesper.GGUF.Types
import Hesper.Basic
import Hesper.Models.Gemma4.Config

/-!
# DiffusionGemma model configuration

DiffusionGemma ("diffusion-gemma", e.g. `diffusiongemma-26B-A4B-it`) is a
block text-diffusion MoE model on a Gemma-4 backbone.  The transformer
backbone is *identical* to Gemma 4 (per the llama.cpp reference
`gemma4-common.h`), so we reuse `Gemma4.Config` for the backbone shape and
wrap it with the diffusion-specific knobs.

Key differences from Gemma 4 that this parser handles:
- metadata keys use the `diffusion-gemma.` prefix
- `attention.head_count_kv` is a **per-layer u32 array** (full-attention
  layers use fewer KV heads than the sliding-window layers)
- diffusion decode parameters: `diffusion.canvas_length`, mask token id,
  and `attention.causal = false`

See `refs/diffusiongemma/ARCH_NOTES.md` for the full verified architecture.
-/

namespace Hesper.Models.DiffusionGemma

open Hesper.Models.Gemma4 (Config LayerType)

/-! ## Local GGUF metadata helpers

Small self-contained readers so this parser does not depend on the
`private` helpers inside the Gemma 4 loader and does not modify Gemma 4.
The Hesper GGUF parser stores fixed-size values / array elements as raw
little-endian bytes with the type+length header stripped (see
`Hesper/GGUF/Parser.lean`). -/

private def findMeta (gguf : Hesper.GGUF.GGUFFile) (key : String) :
    Option Hesper.GGUF.MetadataValue :=
  gguf.metadata.find? (·.1 == key) |>.map (·.2)

private def u32At (data : ByteArray) (i : Nat) : Nat :=
  let b0 := (data.get! (i)    ).toNat
  let b1 := (data.get! (i + 1)).toNat
  let b2 := (data.get! (i + 2)).toNat
  let b3 := (data.get! (i + 3)).toNat
  b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24)

private def readU32 (mv : Hesper.GGUF.MetadataValue) : Option Nat :=
  if mv.valueType == .MUInt32 && mv.data.size >= 4 then some (u32At mv.data 0) else none

private def readF32 (mv : Hesper.GGUF.MetadataValue) : Option Float :=
  if mv.valueType == .MFloat32 && mv.data.size >= 4 then
    some (Hesper.Basic.float32BitsToFloat64 (UInt32.ofNat (u32At mv.data 0)))
  else none

/-- Read an array of bools (1 byte each). -/
private def readBoolArray (mv : Hesper.GGUF.MetadataValue) : Option (Array Bool) :=
  if mv.valueType == .MArray then
    some (Id.run do
      let mut arr := #[]
      for i in [0:mv.data.size] do
        arr := arr.push ((mv.data.get! i) != 0)
      return arr)
  else none

/-- Read an array of u32 (4 bytes each, little-endian). -/
private def readU32Array (mv : Hesper.GGUF.MetadataValue) : Option (Array Nat) :=
  if mv.valueType == .MArray then
    some (Id.run do
      let mut arr := #[]
      let n := mv.data.size / 4
      for i in [0:n] do
        arr := arr.push (u32At mv.data (i * 4))
      return arr)
  else none

/-- DiffusionGemma configuration: the Gemma-4 backbone shape plus the
    diffusion decode parameters. -/
structure DiffusionConfig where
  /-- Gemma-4 backbone configuration (reused verbatim for the transformer). -/
  base : Config
  /-- Number of canvas positions denoised in parallel (`diffusion.canvas_length`). -/
  canvasLength : Nat
  /-- `[MASK]` token id used to seed the canvas (`tokenizer.ggml.mask_token_id`). -/
  maskTokenId : Nat
  /-- Whether attention is causal.  Always false for diffusion-gemma. -/
  causal : Bool
  /-- Number of denoising steps for the decode loop (default 128). -/
  denoiseSteps : Nat := 128
  deriving Repr

/-- Architecture string for diffusion-gemma GGUF files. -/
def archName : String := "diffusion-gemma"

/-- Parse a DiffusionGemma config from GGUF metadata.

    Reuses `Gemma4.Config` for the backbone but reads the
    `diffusion-gemma.` key prefix and handles the per-layer
    `head_count_kv` array. -/
def DiffusionConfig.fromGGUF (gguf : Hesper.GGUF.GGUFFile) : Except String DiffusionConfig := do
  let find := findMeta gguf
  let getU32 (key : String) : Except String Nat :=
    match find key with
    | some mv => match readU32 mv with
      | some v => .ok v
      | none => .error s!"Metadata key '{key}' is not uint32"
    | none => .error s!"Metadata key '{key}' not found"
  let getU32D (key : String) (d : Nat) : Nat :=
    (find key).bind readU32 |>.getD d
  let getF32D (key : String) (d : Float) : Float :=
    (find key).bind readF32 |>.getD d
  let p (suffix : String) : String := s!"{archName}.{suffix}"

  let hiddenSize ← getU32 (p "embedding_length")
  let intermediateSize ← getU32 (p "feed_forward_length")
  let numLayers ← getU32 (p "block_count")
  let numHeads ← getU32 (p "attention.head_count")
  let vocabSize := getU32D "general.vocab_size" ((getU32D (p "vocab_size") 262144))

  -- Per-layer attention types from sliding_window_pattern.
  -- Convention (matches Gemma 4 / gemma4-iswa.cpp): True = SWA, False = full.
  let layerTypes : Array LayerType :=
    match (find (p "attention.sliding_window_pattern")).bind readBoolArray with
    | some bools => bools.map (fun b => if b then LayerType.swa else LayerType.full)
    | none => (List.replicate numLayers LayerType.full).toArray

  -- head_count_kv is a per-layer u32 array. Derive the per-type counts by
  -- indexing the array at a representative SWA layer and a full-attn layer.
  let kvArr : Array Nat :=
    match (find (p "attention.head_count_kv")).bind readU32Array with
    | some a => a
    | none => (List.replicate numLayers (getU32D (p "attention.head_count_kv") 8)).toArray
  let kvHeadsForType (lt : LayerType) (default : Nat) : Nat :=
    match (List.range numLayers).find? (fun i =>
            (layerTypes[i]?.getD LayerType.full) == lt) with
    | some i => kvArr[i]?.getD default
    | none => default
  let numKeyValueHeadsSWA := kvHeadsForType LayerType.swa 8
  let numKeyValueHeadsFull := kvHeadsForType LayerType.full numKeyValueHeadsSWA

  let headDimSWA := getU32D (p "attention.key_length_swa") 256
  let headDimFull := getU32D (p "attention.key_length") headDimSWA

  let base : Config := {
    vocabSize
    hiddenSize
    intermediateSize
    numHiddenLayers := numLayers
    numAttentionHeads := numHeads
    numKeyValueHeadsFull
    numKeyValueHeadsSWA
    headDimFull
    headDimSWA
    slidingWindowSize := getU32D (p "attention.sliding_window") 1024
    rmsNormEps := getF32D (p "attention.layer_norm_rms_epsilon") 1e-6
    ropeTheta := getF32D (p "rope.freq_base") 1000000.0
    ropeThetaSWA := getF32D (p "rope.freq_base_swa") 10000.0
    partialRotaryFactorSWA := 0.5
    layerTypes
    logitSoftcapScale := getF32D (p "final_logit_softcapping") 30.0
    -- Diffusion forward is a single no-cache pass over [prompt | canvas];
    -- there is no growing KV cache, so the context only needs to cover
    -- prompt + canvas. Keep a modest cap for buffer sizing.
    maxSeqLen := min 4096 (getU32D (p "context_length") 4096)
    numExperts := getU32D (p "expert_count") 0
    numExpertsUsed := getU32D (p "expert_used_count") 0
    expertFFSize := getU32D (p "expert_feed_forward_length") 0
    embdPerLayer := getU32D (p "embedding_length_per_layer_input") 0
    numKVSharedLayers := getU32D (p "attention.shared_kv_layers") 0
  }

  return {
    base
    canvasLength := getU32D "diffusion.canvas_length" 256
    maskTokenId := getU32D "tokenizer.ggml.mask_token_id" 4
    causal := false
    denoiseSteps := 128
  }

end Hesper.Models.DiffusionGemma
