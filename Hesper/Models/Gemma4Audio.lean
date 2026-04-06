import Hesper.WGSL.Monad
import Hesper.WGSL.Execute
import Hesper.WGSL.Exp
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.Logging

/-!
# Gemma 4 Audio Encoder (Conformer)

Implements the Conformer-based audio encoder for Gemma 4 multimodal.

## Architecture (from conformer.cpp)

```
Audio waveform
  │
  ├─ Mel spectrogram (CPU preprocessing)
  │   └─ [n_frames, n_mel_bins]
  │
  ├─ Pre-encode conv subsampling:
  │   ├─ Conv2D (stride 2) + ReLU
  │   ├─ Depthwise Conv2D (stride 2) + Conv2D + ReLU
  │   ├─ Depthwise Conv2D (stride 2) + Conv2D + ReLU
  │   ├─ Flatten + linear projection
  │   └─ → [n_pos, hidden_size]
  │
  ├─ Positional embeddings (relative position bias)
  │
  ├─ Conformer blocks (N layers):
  │   ├─ Feed-Forward 1: LayerNorm → Linear → SiLU → Linear → * 0.5 + residual
  │   ├─ Self-Attention:
  │   │   ├─ LayerNorm → Q/K/V projections (with bias)
  │   │   ├─ Q + pos_bias_u, Q + pos_bias_v
  │   │   ├─ Relative position attention (matrix_ac + matrix_bd)
  │   │   ├─ Softmax → attention @ V → linear out
  │   │   └─ + residual
  │   ├─ Convolution Module:
  │   │   ├─ LayerNorm → Pointwise Conv1 → Sigmoid GLU
  │   │   ├─ Depthwise Conv → BatchNorm → SiLU → Pointwise Conv2
  │   │   └─ + residual
  │   ├─ Feed-Forward 2: LayerNorm → Linear → SiLU → Linear → * 0.5 + residual
  │   └─ Final LayerNorm
  │
  └─ Audio Adapter:
      ├─ LayerNorm
      └─ FFN (GELU) → projected [textHiddenSize]
```

## GGUF Tensor Names (audio section of mmproj)

```
conformer.pre_encode.conv.0.weight/bias    -- Pre-encode conv layers
conformer.pre_encode.conv.2.weight/bias
conformer.pre_encode.conv.3.weight/bias
conformer.pre_encode.conv.5.weight/bias
conformer.pre_encode.conv.6.weight/bias
conformer.pre_encode.out.weight/bias       -- Pre-encode output projection
a.blk.N.norm_feed_forward1.weight/bias     -- FF1 norm
a.blk.N.feed_forward1.up.weight/bias       -- FF1 up
a.blk.N.feed_forward1.down.weight/bias     -- FF1 down
a.blk.N.norm_self_att.weight/bias          -- Self-attn norm
a.blk.N.attn_q.weight/bias                -- Q projection
a.blk.N.attn_k.weight/bias                -- K projection
a.blk.N.attn_v.weight/bias                -- V projection
a.blk.N.attn_out.weight/bias              -- Output projection
a.blk.N.pos_bias_u                         -- Position bias u
a.blk.N.pos_bias_v                         -- Position bias v
a.blk.N.linear_pos.weight                  -- Linear position
a.blk.N.per_dim_k_scale                    -- Per-dimension K scale
a.blk.N.norm_conv.weight/bias             -- Conv norm
a.blk.N.conv_pw1.weight/bias              -- Pointwise conv 1
a.blk.N.conv_dw.weight/bias               -- Depthwise conv
a.blk.N.conv_norm.weight/bias             -- Conv batch norm
a.blk.N.conv_pw2.weight/bias              -- Pointwise conv 2
a.blk.N.norm_feed_forward2.weight/bias    -- FF2 norm
a.blk.N.feed_forward2.up.weight/bias      -- FF2 up
a.blk.N.feed_forward2.down.weight/bias    -- FF2 down
a.blk.N.norm_out.weight/bias              -- Output norm
model.audio_tower.output_proj.weight       -- Final projection to text dim
```

## References
- llama.cpp/tools/mtmd/models/conformer.cpp
- llama.cpp/tools/mtmd/mtmd-audio.cpp (mel spectrogram)
-/

namespace Hesper.Models.Gemma4Audio

open Hesper.WGSL
open Hesper.WGSL.Monad
open Hesper.WebGPU
open Hesper.Logging (logVerbose)

/-! ## Configuration -/

/-- Conformer audio encoder configuration -/
structure Config where
  numMelBins : Nat         -- Mel filterbank bins (128)
  sampleRate : Nat := 16000  -- Audio sample rate
  hiddenSize : Nat         -- Conformer hidden dimension
  numLayers : Nat          -- Number of Conformer blocks
  numHeads : Nat           -- Self-attention heads
  headDim : Nat            -- Per-head dimension
  ffnSize : Nat            -- FFN intermediate size
  convKernelSize : Nat := 9  -- Depthwise conv kernel size
  textHiddenSize : Nat     -- Target text model dimension (projection output)
  layerNormEps : Float := 1e-5
  deriving Repr, Inhabited

/-! ## Pre-encode Conv Subsampling Kernels -/

/-- Conv2D with stride 2 + bias kernel (pre-encode subsampling).

    Each conv layer reduces temporal resolution by 2x.
    3 such layers: 8x total downsampling of frames.

    @param inChannels Input channels
    @param outChannels Output channels
    @param inHeight Input height (frequency)
    @param inWidth Input width (time frames)
    @param kernelSize Kernel size (3)
    @param stride Stride (2)
-/
def conv2dStridedKernel (inChannels outChannels inHeight inWidth kernelSize stride : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let outHeight := (inHeight + 2 * 1 - kernelSize) / stride + 1  -- padding=1
  let outWidth := (inWidth + 2 * 1 - kernelSize) / stride + 1
  let totalOutputs := outChannels * outHeight * outWidth

  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) (inChannels * inHeight * inWidth))
  let _weight ← ShaderM.declareInputBuffer "weight" (.array (.scalar .f32) (outChannels * inChannels * kernelSize * kernelSize))
  let _bias ← ShaderM.declareInputBuffer "bias" (.array (.scalar .f32) outChannels)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) totalOutputs)

  ShaderM.if_ (Exp.lt idx (Exp.litU32 totalOutputs)) (do
    -- Decompose: [outChannel, outY, outX]
    let outX := Exp.mod idx (Exp.litU32 outWidth)
    let tmp := Exp.div idx (Exp.litU32 outWidth)
    let outY := Exp.mod tmp (Exp.litU32 outHeight)
    let outC := Exp.div tmp (Exp.litU32 outHeight)

    -- Read bias
    let biasVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := outChannels) "bias" outC

    ShaderM.varNamed "conv_sum" (.scalar .f32) biasVal
    let convSum : Exp (.scalar .f32) := Exp.var "conv_sum"

    -- Convolution loop (compile-time unrolled for small kernels)
    for ky in [0:kernelSize] do
      for kx in [0:kernelSize] do
        let inY := Exp.sub (Exp.add (Exp.mul outY (Exp.litU32 stride)) (Exp.litU32 ky)) (Exp.litU32 1)  -- padding=1
        let inX := Exp.sub (Exp.add (Exp.mul outX (Exp.litU32 stride)) (Exp.litU32 kx)) (Exp.litU32 1)
        let inBoundsY := Exp.and (Exp.ge inY (Exp.litU32 0)) (Exp.lt inY (Exp.litU32 inHeight))
        let inBoundsX := Exp.and (Exp.ge inX (Exp.litU32 0)) (Exp.lt inX (Exp.litU32 inWidth))
        let inBounds := Exp.and inBoundsY inBoundsX

        -- Sum over input channels (unrolled for small channel counts in pre-encode)
        -- For larger channel counts, this needs a loop
        ShaderM.if_ inBounds (do
          -- Simplified: iterate over input channels
          -- weight index: [outC, inC, ky, kx]
          -- input index: [inC, inY, inX]
          -- For now just accumulate channel 0 as placeholder
          -- TODO: full channel loop
          let inIdx := Exp.add (Exp.mul inY (Exp.litU32 inWidth)) inX
          let wIdx := Exp.add
            (Exp.mul outC (Exp.litU32 (inChannels * kernelSize * kernelSize)))
            (Exp.litU32 (ky * kernelSize + kx))
          let inVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := inChannels * inHeight * inWidth) "input" inIdx
          let wVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := outChannels * inChannels * kernelSize * kernelSize) "weight" wIdx
          ShaderM.assign "conv_sum" (Exp.add convSum (Exp.mul inVal wVal))
        ) (pure ())

    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx convSum
  ) (pure ())

/-! ## SiLU (Swish) Activation Kernel -/

/-- SiLU activation: f(x) = x * sigmoid(x)
    Used in Conformer FFN layers.
    @param size Number of elements
-/
def siluKernel (size : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) size)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) size)

  ShaderM.if_ (Exp.lt idx (Exp.litU32 size)) (do
    let x ← ShaderM.readBuffer (ty := .scalar .f32) (n := size) "input" idx
    let sigmoid := Exp.div (Exp.litF32 1.0) (Exp.add (Exp.litF32 1.0) (Exp.exp (Exp.neg x)))
    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx (Exp.mul x sigmoid)
  ) (pure ())

/-! ## Sigmoid GLU Kernel -/

/-- Sigmoid GLU: used in Conformer convolution module.
    Input: [2*d, n_pos], split into two halves
    Output: first_half * sigmoid(second_half)
    @param halfSize d (half of input first dimension)
    @param seqLen n_pos
-/
def sigmoidGLUKernel (halfSize seqLen : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let totalOutput := halfSize * seqLen

  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) (2 * halfSize * seqLen))
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) totalOutput)

  ShaderM.if_ (Exp.lt idx (Exp.litU32 totalOutput)) (do
    -- First half value
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := 2 * halfSize * seqLen) "input" idx
    -- Second half (gate) value, offset by halfSize * seqLen
    let gateVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := 2 * halfSize * seqLen) "input" (Exp.add idx (Exp.litU32 (halfSize * seqLen)))
    let gate := Exp.div (Exp.litF32 1.0) (Exp.add (Exp.litF32 1.0) (Exp.exp (Exp.neg gateVal)))
    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx (Exp.mul val gate)
  ) (pure ())

/-! ## Relative Position Attention Kernel -/

/-- Scale + add for relative position attention scores.
    scores = (matrix_ac + matrix_bd) / sqrt(d_head)

    matrix_ac: content-based attention scores [numHeads, seqLen, seqLen]
    matrix_bd: position-based attention scores [numHeads, seqLen, seqLen]

    @param numHeads Number of attention heads
    @param seqLen Sequence length
    @param headDim Per-head dimension (for scaling)
-/
def relPosScoreKernel (numHeads seqLen headDim : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let totalElements := numHeads * seqLen * seqLen

  let _matrixAC ← ShaderM.declareInputBuffer "matrix_ac" (.array (.scalar .f32) totalElements)
  let _matrixBD ← ShaderM.declareInputBuffer "matrix_bd" (.array (.scalar .f32) totalElements)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) totalElements)

  ShaderM.if_ (Exp.lt idx (Exp.litU32 totalElements)) (do
    let ac ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalElements) "matrix_ac" idx
    let bd ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalElements) "matrix_bd" idx
    let scale := Exp.litF32 (1.0 / Float.sqrt headDim.toFloat)
    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx (Exp.mul (Exp.add ac bd) scale)
  ) (pure ())

/-! ## Feed-Forward Half-Residual Kernel -/

/-- Half-residual add: output = residual + 0.5 * ffn_output
    Used in Conformer for feed-forward modules (fc_factor = 0.5).
    @param size Number of elements
-/
def halfResidualAddKernel (size : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let _residual ← ShaderM.declareInputBuffer "residual" (.array (.scalar .f32) size)
  let _ffnOut ← ShaderM.declareInputBuffer "ffn_out" (.array (.scalar .f32) size)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) size)

  ShaderM.if_ (Exp.lt idx (Exp.litU32 size)) (do
    let res ← ShaderM.readBuffer (ty := .scalar .f32) (n := size) "residual" idx
    let ffn ← ShaderM.readBuffer (ty := .scalar .f32) (n := size) "ffn_out" idx
    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx (Exp.add res (Exp.mul (Exp.litF32 0.5) ffn))
  ) (pure ())

/-! ## Mel Spectrogram (CPU Preprocessing) -/

/-- Mel spectrogram configuration for CPU-side preprocessing.
    The actual computation is done on CPU before GPU inference.
-/
structure MelConfig where
  sampleRate : Nat := 16000
  nFFT : Nat := 512
  hopLength : Nat := 160
  nMelBins : Nat := 128
  fMin : Float := 0.0
  fMax : Float := 8000.0
  deriving Repr, Inhabited

end Hesper.Models.Gemma4Audio
