import Hesper.Layers.Linear
import Hesper.Layers.RMSNorm
import Hesper.Layers.Embedding
import Hesper.Models.Gemma4.Config
import Hesper.CUDA.FFI

/-!
# Gemma 4 layer structure types

Plain data holding Gemma 4's per-layer weights, per-block norms, and
the top-level `Gemma4Model`.  No IO.  Extracted so both the loader and
the runtime can depend on these types without pulling in each other.
-/

namespace Hesper.Models.Gemma4

open Hesper.Layers

/-! ## Layer Structures -/

/-- Gemma 4 attention layer (single layer) -/
structure Gemma4Attention (BufT CacheT : Type) where
  wQ : Linear.LinearLayer BufT CacheT         -- Q projection [hiddenSize → numHeads * headDim]
  wK : Linear.LinearLayer BufT CacheT         -- K projection [hiddenSize → numKVHeads * headDim]
  wV : Linear.LinearLayer BufT CacheT         -- V projection [hiddenSize → numKVHeads * headDim]
  wO : Linear.LinearLayer BufT CacheT         -- Output projection [numHeads * headDim → hiddenSize]
  qNormWeight : BufT            -- Per-head Q norm [headDim]
  kNormWeight : BufT            -- Per-head K norm [headDim]
  -- Fused RMSNorm+Linear cache refs (attnNorm fused into Q/K/V projections)
  fusedNormQPrepared : IO.Ref (Option CacheT)
  fusedNormKPrepared : IO.Ref (Option CacheT)
  fusedNormVPrepared : IO.Ref (Option CacheT)

/-- Gemma 4 dense FFN layer -/
structure Gemma4FFN (BufT CacheT : Type) where
  gate : Linear.LinearLayer BufT CacheT
  up : Linear.LinearLayer BufT CacheT
  down : Linear.LinearLayer BufT CacheT
  fusedGateUpPrepared : IO.Ref (Option CacheT)
  -- Fused RMSNorm+Linear cache refs (ffnNorm fused into gate/up)
  fusedNormGatePrepared : IO.Ref (Option CacheT)
  fusedNormUpPrepared : IO.Ref (Option CacheT)

/-- Gemma 4 transformer block (single layer) -/
structure Gemma4Block (BufT CacheT : Type) where
  layerIdx : Nat
  layerType : LayerType
  -- Norms
  attnNorm : RMSNorm.RMSNorm BufT CacheT
  postAttnNorm : RMSNorm.RMSNorm BufT CacheT
  ffnNorm : RMSNorm.RMSNorm BufT CacheT
  postFFNNorm : RMSNorm.RMSNorm BufT CacheT
  -- Attention
  attention : Gemma4Attention BufT CacheT
  -- FFN (shared/dense expert)
  ffn : Gemma4FFN BufT CacheT
  -- MoE (optional: present only for MoE layers)
  isMoE : Bool
  moeRouterWeight : Option BufT
  moeRouterScale : Option BufT
  moeGateUpExps : Option BufT
  moeDownExps : Option BufT
  moePreNorm2 : Option (RMSNorm.RMSNorm BufT CacheT)
  moePostNorm1 : Option (RMSNorm.RMSNorm BufT CacheT)
  moePostNorm2 : Option (RMSNorm.RMSNorm BufT CacheT)
  -- Optional: RoPE frequency factors (full attention layers only)
  ropeFreqFactors : Option BufT
  -- Optional: layer output scale (DiffusionGemma: canvas/decoder scalar)
  outScale : Option BufT
  -- Optional: encoder layer output scale (DiffusionGemma global/prompt scalar).
  -- Unused by Gemma 4; defaults to none so the Gemma 4 loader is unaffected.
  encOutScale : Option BufT := none

/-- Per-layer embedding weights for a single block -/
structure Gemma4PerLayerEmbd (BufT CacheT : Type) where
  inpGate : Linear.LinearLayer BufT CacheT
  proj : Linear.LinearLayer BufT CacheT
  postNorm : RMSNorm.RMSNorm BufT CacheT

/-- Embedding format for token embedding table -/
inductive EmbdFormat where
  | F32   -- Pre-dequantized F32 (via Embedding.forward)
  | F16   -- F16 (via Embedding.forward with f16Table)
  | Q6_K  -- Q6_K packed (GPU dequant lookup + LM head matmul)
  | Q4_K  -- Q4_K packed
  deriving Repr, BEq, Inhabited

/-- Complete Gemma 4 model -/
structure Gemma4Model (BufT CacheT : Type) where
  config : Config
  embedding : Embedding.Embedding BufT
  embdFormat : EmbdFormat
  blocks : Array (Gemma4Block BufT CacheT)
  finalNorm : RMSNorm.RMSNorm BufT CacheT
  outputWeight : BufT
  /-- When `output.weight` was loaded as Q6_K, we pre-dequantize it to
      a packed half2 (f16) buffer at load time and use the f16 matmul
      kernel for lm_head — matches llama.cpp's CUDA backend, which keeps
      the LM head in f16.  Reduces Q6_K dispatch from 1140 µs/call to
      ~114 µs/call (matches `mul_mat_vec_f<half,half>`). -/
  outputWeightF16 : Option BufT
  /-- When `none`, the full Q6_K table was uploaded to VRAM.
      When `some`, the full table stays in CPU mmap (saving ~2.2 GiB VRAM)
      and the kernel reads it via the unified-VA device pointer.  Tuple is
      `(mmap, dataSecOff, tensorOff, deviceMappedHostPtr)`: the first three
      keep mmap alive and let us compute file offsets; the fourth is the
      `cuMemHostGetDevicePointer` result for `(addr + dataSecOff + tensorOff)`,
      which kernels dereference directly per token.  Same pattern as
      llama.cpp's getrows kernel reading the CPU_Mapped buffer. -/
  perLayerEmbdMmap : Option (Hesper.CUDA.MMappedFile × USize × USize × USize) := none
  perLayerEmbdTableGPU : Option BufT
  /-- Row-staging path (WebGPU / no-mmap): the full Q6_K per-layer table kept on CPU. When set,
      `perLayerEmbdTableGPU` is a ONE-ROW scratch buffer and the active token's row is
      writeBuffer'd into it before each dequant dispatch (kernel tokenId = 0). Binding the full
      1.5-2.2 GiB table is invalid on WebGPU (maxStorageBufferBindingSize) AND wrong (the kernel
      declares a 2-row table, so robustness clamps any real tokenId). -/
  perLayerEmbdTableBytes : Option ByteArray := none
  perLayerEmbdRowBytes : Nat
  perLayerModelProj : Option BufT
  perLayerProjNorm : Option (RMSNorm.RMSNorm BufT CacheT)
  perLayerBlocks : Array (Option (Gemma4PerLayerEmbd BufT CacheT))

end Hesper.Models.Gemma4
