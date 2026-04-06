import Hesper.WGSL.Monad
import Hesper.WGSL.Execute
import Hesper.WGSL.Exp
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.Logging

/-!
# Gemma 4 Vision Encoder (SigLIP)

Implements the SigLIP-based vision encoder for Gemma 4 multimodal.

## Architecture (from gemma4v.cpp)

```
Image [H, W, 3]
  │
  ├─ Scale: patches * 2 - 1
  ├─ Conv2D patch embedding (kernel=stride=patch_size, no bias)
  ├─ Reshape + Transpose → [n_patches, n_embd]
  ├─ 2D Positional embeddings (separate X/Y lookup tables)
  │
  ├─ ViT Transformer blocks (N layers):
  │   ├─ RMSNorm
  │   ├─ Self-attention with RoPE2D (NEOX ordering)
  │   │   First half dims: RoPE with pos_x
  │   │   Second half dims: RoPE with pos_y
  │   ├─ FFN (GELU activation)
  │   └─ Residual connections
  │
  ├─ Vision Pooler:
  │   ├─ Transpose → [n_patches_x, n_patches_y, n_embd]
  │   ├─ Avg pool 2D (kernel=stride=n_merge)
  │   ├─ Reshape → [n_embd, out_x * out_y]
  │   └─ Scale by sqrt(n_embd)
  │
  ├─ Std normalization: (hidden - std_bias) * std_scale
  │
  └─ Multimodal Embedder:
      ├─ Gemma4ClippableLinear (mm_input_proj_w, with input/output clamping)
      └─ Post-projection RMSNorm
```

## GGUF Tensor Names (vision mmproj file)

```
v.patch_embeddings_0.weight          -- Conv2D weights [n_embd, 3, patch_size, patch_size]
v.position_embeddings.weight         -- 2D pos embeddings [2*max_pos, n_embd]
v.blk.N.attn_q.weight               -- Self-attention Q
v.blk.N.attn_k.weight               -- Self-attention K
v.blk.N.attn_v.weight               -- Self-attention V
v.blk.N.attn_out.weight             -- Self-attention output
v.blk.N.attn_q_norm.weight          -- Q normalization
v.blk.N.attn_k_norm.weight          -- K normalization
v.blk.N.ln1.weight                  -- Pre-attention layer norm
v.blk.N.ln2.weight                  -- Pre-FFN layer norm
v.blk.N.ffn_down.weight             -- FFN down projection
v.blk.N.ffn_up.weight               -- FFN up projection
v.blk.N.ffn_gate.weight             -- FFN gate projection
v.std_bias                           -- Std normalization bias
v.std_scale                          -- Std normalization scale
v.mm_input_proj_w.weight             -- Multimodal projector
```

## References
- llama.cpp/tools/mtmd/models/gemma4v.cpp
-/

namespace Hesper.Models.Gemma4Vision

open Hesper.WGSL
open Hesper.WGSL.Monad
open Hesper.WebGPU
open Hesper.Logging (logVerbose)

/-! ## Configuration -/

/-- Vision encoder configuration -/
structure Config where
  imageSize : Nat          -- Input image size (224)
  patchSize : Nat          -- Patch size (14 or 16)
  numChannels : Nat := 3   -- RGB
  hiddenSize : Nat         -- Vision embedding dimension
  numLayers : Nat          -- Number of ViT blocks
  numHeads : Nat           -- Attention heads
  headDim : Nat            -- Per-head dimension
  ffnSize : Nat            -- FFN intermediate size
  mergeSize : Nat          -- Vision pooler merge kernel size
  ropeTheta : Float        -- RoPE base for vision
  rmsNormEps : Float       -- Layer norm epsilon (1e-6)
  textHiddenSize : Nat     -- Text model hidden size (projection target)
  deriving Repr, Inhabited

/-- Number of patches per axis -/
def Config.numPatchesPerAxis (c : Config) : Nat := c.imageSize / c.patchSize

/-- Total number of patches -/
def Config.numPatches (c : Config) : Nat := c.numPatchesPerAxis * c.numPatchesPerAxis

/-- Pooled output patches per axis -/
def Config.pooledPatchesPerAxis (c : Config) : Nat := c.numPatchesPerAxis / c.mergeSize

/-- Total pooled output patches (= number of vision tokens) -/
def Config.numVisionTokens (c : Config) : Nat := c.pooledPatchesPerAxis * c.pooledPatchesPerAxis

/-! ## Patch Preprocessing Kernel -/

/-- Input preprocessing: patches * 2 - 1
    Normalizes pixel values from [0, 1] to [-1, 1].
    @param size Total number of pixel values (H * W * 3)
-/
def patchPreprocessKernel (size : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) size)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) size)

  ShaderM.if_ (Exp.lt idx (Exp.litU32 size)) (do
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := size) "input" idx
    -- patches * 2 - 1
    let result := Exp.sub (Exp.mul val (Exp.litF32 2.0)) (Exp.litF32 1.0)
    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx result
  ) (pure ())

/-! ## 2D Positional Embedding Lookup -/

/-- Add 2D positional embeddings to patch embeddings.

    Position embedding table is stored as [2*maxPos, hiddenSize]:
    - First half [0..maxPos): X position embeddings
    - Second half [maxPos..2*maxPos): Y position embeddings

    For each patch at grid position (px, py):
      output[patch] = input[patch] + posEmbed[px] + posEmbed[maxPos + py]

    @param numPatches Total number of patches
    @param hiddenSize Embedding dimension
    @param patchesPerAxis Patches per axis (for X/Y decomposition)
    @param maxPos Maximum position entries per axis
-/
def addPosEmbedKernel (numPatches hiddenSize patchesPerAxis maxPos : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let totalElements := numPatches * hiddenSize

  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) totalElements)
  let _posEmbed ← ShaderM.declareInputBuffer "pos_embed" (.array (.scalar .f32) (2 * maxPos * hiddenSize))
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) totalElements)

  ShaderM.if_ (Exp.lt idx (Exp.litU32 totalElements)) (do
    let patchIdx := Exp.div idx (Exp.litU32 hiddenSize)
    let dimIdx := Exp.mod idx (Exp.litU32 hiddenSize)

    -- Decompose patch index to grid (px, py)
    let px := Exp.mod patchIdx (Exp.litU32 patchesPerAxis)
    let py := Exp.div patchIdx (Exp.litU32 patchesPerAxis)

    -- Look up X embedding: posEmbed[px * hiddenSize + dimIdx]
    let xEmbIdx := Exp.add (Exp.mul px (Exp.litU32 hiddenSize)) dimIdx
    let xEmb ← ShaderM.readBuffer (ty := .scalar .f32) (n := 2 * maxPos * hiddenSize) "pos_embed" xEmbIdx

    -- Look up Y embedding: posEmbed[(maxPos + py) * hiddenSize + dimIdx]
    let yEmbIdx := Exp.add (Exp.mul (Exp.add (Exp.litU32 maxPos) py) (Exp.litU32 hiddenSize)) dimIdx
    let yEmb ← ShaderM.readBuffer (ty := .scalar .f32) (n := 2 * maxPos * hiddenSize) "pos_embed" yEmbIdx

    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalElements) "input" idx
    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx (Exp.add val (Exp.add xEmb yEmb))
  ) (pure ())

/-! ## RoPE2D Kernel -/

/-- RoPE2D: Apply 2D rotary position embeddings.

    Split head dimension in half:
    - First half: apply RoPE with X positions
    - Second half: apply RoPE with Y positions
    Both use NEOX ordering within their halves.

    Input shape: [numPatches, numHeads, headDim]
    @param numPatches Total patches
    @param numHeads Attention heads
    @param headDim Per-head dimension
    @param ropeTheta RoPE base frequency
    @param patchesPerAxis For X/Y position decomposition
-/
def rope2DKernel (numPatches numHeads headDim : Nat) (ropeTheta : Float) (patchesPerAxis : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let halfDim := headDim / 2
  let quarterDim := halfDim / 2  -- pairs per half
  let totalElements := numPatches * numHeads * quarterDim

  let totalBuf := numPatches * numHeads * headDim

  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) totalBuf)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) totalBuf)

  ShaderM.if_ (Exp.lt idx (Exp.litU32 totalElements)) (do
    -- Decompose: [patch, head, dimPair]
    let dimPair := Exp.mod idx (Exp.litU32 quarterDim)
    let tmp1 := Exp.div idx (Exp.litU32 quarterDim)
    let head := Exp.mod tmp1 (Exp.litU32 numHeads)
    let patch := Exp.div tmp1 (Exp.litU32 numHeads)

    -- Grid position
    let px := Exp.mod patch (Exp.litU32 patchesPerAxis)
    let py := Exp.div patch (Exp.litU32 patchesPerAxis)

    let baseOffset := Exp.add (Exp.mul patch (Exp.litU32 (numHeads * headDim)))
      (Exp.mul head (Exp.litU32 headDim))

    -- Compute theta
    let dimPairF32 := Exp.toF32 dimPair
    let exponent := Exp.div (Exp.mul (Exp.litF32 2.0) dimPairF32) (Exp.litF32 halfDim.toFloat)
    let freqInv := Exp.pow (Exp.litF32 ropeTheta) (Exp.neg exponent)

    -- First half: RoPE with X position (NEOX: pairs at [i, i + quarterDim])
    let thetaX := Exp.mul (Exp.toF32 px) freqInv
    let cosX := Exp.cos thetaX
    let sinX := Exp.sin thetaX

    let idx0x := Exp.add baseOffset dimPair
    let idx1x := Exp.add baseOffset (Exp.add dimPair (Exp.litU32 quarterDim))

    let x0 ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalBuf) "input" idx0x
    let x1 ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalBuf) "input" idx1x
    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx0x (Exp.sub (Exp.mul x0 cosX) (Exp.mul x1 sinX))
    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx1x (Exp.add (Exp.mul x0 sinX) (Exp.mul x1 cosX))

    -- Second half: RoPE with Y position (NEOX: pairs at [halfDim+i, halfDim+i+quarterDim])
    let thetaY := Exp.mul (Exp.toF32 py) freqInv
    let cosY := Exp.cos thetaY
    let sinY := Exp.sin thetaY

    let idx0y := Exp.add baseOffset (Exp.add (Exp.litU32 halfDim) dimPair)
    let idx1y := Exp.add baseOffset (Exp.add (Exp.litU32 halfDim) (Exp.add dimPair (Exp.litU32 quarterDim)))

    let y0 ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalBuf) "input" idx0y
    let y1 ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalBuf) "input" idx1y
    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx0y (Exp.sub (Exp.mul y0 cosY) (Exp.mul y1 sinY))
    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx1y (Exp.add (Exp.mul y0 sinY) (Exp.mul y1 cosY))
  ) (pure ())

/-! ## Vision Pooler Kernel -/

/-- Vision Pooler: 2D average pooling + scale.

    Input: [n_embd, n_patches_x, n_patches_y] (transposed from ViT output)
    Output: [n_embd, out_x * out_y]

    Performs avg pool with kernel=stride=mergeSize, then scales by sqrt(n_embd).

    @param hiddenSize Embedding dimension
    @param patchesX Input patches per X axis
    @param patchesY Input patches per Y axis
    @param mergeSize Pooling kernel/stride size
-/
def visionPoolerKernel (hiddenSize patchesX patchesY mergeSize : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let outX := patchesX / mergeSize
  let outY := patchesY / mergeSize
  let numOutputTokens := outX * outY
  let totalOutputs := hiddenSize * numOutputTokens

  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) (hiddenSize * patchesX * patchesY))
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) totalOutputs)

  ShaderM.if_ (Exp.lt idx (Exp.litU32 totalOutputs)) (do
    let dimIdx := Exp.div idx (Exp.litU32 numOutputTokens)
    let tokenIdx := Exp.mod idx (Exp.litU32 numOutputTokens)
    let outPx := Exp.mod tokenIdx (Exp.litU32 outX)
    let outPy := Exp.div tokenIdx (Exp.litU32 outX)

    -- Average pool over mergeSize x mergeSize window
    ShaderM.varNamed "pool_sum" (.scalar .f32) (Exp.litF32 0.0)
    let poolSum : Exp (.scalar .f32) := Exp.var "pool_sum"

    for dy in [0:mergeSize] do
      for dx in [0:mergeSize] do
        let inPx := Exp.add (Exp.mul outPx (Exp.litU32 mergeSize)) (Exp.litU32 dx)
        let inPy := Exp.add (Exp.mul outPy (Exp.litU32 mergeSize)) (Exp.litU32 dy)
        -- Input index: dimIdx * (patchesX * patchesY) + inPy * patchesX + inPx
        let inIdx := Exp.add (Exp.mul dimIdx (Exp.litU32 (patchesX * patchesY)))
          (Exp.add (Exp.mul inPy (Exp.litU32 patchesX)) inPx)
        let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := hiddenSize * patchesX * patchesY) "input" inIdx
        ShaderM.assign "pool_sum" (Exp.add poolSum val)

    let poolSize := Exp.litF32 (mergeSize * mergeSize).toFloat
    let avgVal := Exp.div poolSum poolSize

    -- Scale by sqrt(hiddenSize)
    let scaledVal := Exp.mul avgVal (Exp.litF32 (Float.sqrt hiddenSize.toFloat))

    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx scaledVal
  ) (pure ())

/-! ## Std Normalization Kernel -/

/-- Standard deviation normalization: y = (x - std_bias) * std_scale

    Applied after vision pooler, before the multimodal projector.
    @param size Total elements (hiddenSize * numVisionTokens)
    @param hiddenSize For per-dimension bias/scale lookup
-/
def stdNormKernel (size hiddenSize : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) size)
  let _bias ← ShaderM.declareInputBuffer "std_bias" (.array (.scalar .f32) hiddenSize)
  let _scale ← ShaderM.declareInputBuffer "std_scale" (.array (.scalar .f32) hiddenSize)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) size)

  ShaderM.if_ (Exp.lt idx (Exp.litU32 size)) (do
    let dimIdx := Exp.mod idx (Exp.litU32 hiddenSize)
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := size) "input" idx
    let b ← ShaderM.readBuffer (ty := .scalar .f32) (n := hiddenSize) "std_bias" dimIdx
    let s ← ShaderM.readBuffer (ty := .scalar .f32) (n := hiddenSize) "std_scale" dimIdx
    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx (Exp.mul (Exp.sub val b) s)
  ) (pure ())

/-! ## Clampable Linear Kernel -/

/-- Gemma4ClippableLinear: linear projection with input/output clamping.

    Used by the multimodal embedder (mm_input_proj_w).
    1. Clamp input to [inp_min, inp_max]
    2. Matmul: y = x @ W^T
    3. Clamp output to [out_min, out_max]

    For now, this is just a placeholder for the clamping logic.
    The actual matmul uses the existing Linear layer or a dedicated F32 matmul.

    @param size Total elements
    @param inpMin Input clamp minimum
    @param inpMax Input clamp maximum
    @param outMin Output clamp minimum
    @param outMax Output clamp maximum
-/
def clampKernel (size : Nat) (clampMin clampMax : Float) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) size)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) size)

  ShaderM.if_ (Exp.lt idx (Exp.litU32 size)) (do
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := size) "input" idx
    let clamped := Exp.clamp val (Exp.litF32 clampMin) (Exp.litF32 clampMax)
    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx clamped
  ) (pure ())

end Hesper.Models.Gemma4Vision
