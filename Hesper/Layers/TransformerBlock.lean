import Hesper.WGSL.Monad
import Hesper.WGSL.Elementwise
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WebGPU.BufferOps
import Hesper.Layers.RMSNorm
import Hesper.Layers.Attention
import Hesper.Layers.BitLinear
import Hesper.Logging

/-!
# Transformer Block

Implements a complete transformer layer combining attention and feed-forward network.

## Architecture

```
Input
  │
  ├─────────────────────────┐
  │                         │
  ▼                         │
[RMSNorm]                   │  Pre-attention normalization
  │                         │
  ▼                         │
[Multi-Head Attention]      │  Self-attention mechanism
  │                         │
  ▼                         │
[Add] ←─────────────────────┘  Residual connection
  │
  ├─────────────────────────┐
  │                         │
  ▼                         │
[RMSNorm]                   │  Pre-FFN normalization
  │                         │
  ▼                         │
[FFN: Gate + Up/Down]       │  Feed-forward network
  │                         │
  ▼                         │
[Add] ←─────────────────────┘  Residual connection
  │
  ▼
Output
```

## Feed-Forward Network (FFN)

BitNet uses a gated FFN similar to LLaMA:

```
gate = ReLU²(x @ W_gate)  # Gating signal (BitNet uses relu_sqr, NOT silu)
up = x @ W_up              # Value signal
hidden = gate × up         # Element-wise gating
output = hidden @ W_down   # Down-projection
```

**Why gating?**
- Non-linear mixing of features
- Selective information flow
- Better gradient properties than simple ReLU

**Dimensions** (BitNet-3B):
```
Input: [batch, seq, 2560]
Gate: [2560, 10240] (4× expansion)
Up: [2560, 10240]
Down: [10240, 2560] (back to original)
```

## Memory Layout

**Per-layer weights** (TQ2_0 quantized):
```
Attention:
- W_q, W_k, W_v: 3 × 2560² × 0.25 bytes = 4.9 MB
- W_o: 2560² × 0.25 bytes = 1.6 MB

FFN:
- W_gate, W_up: 2 × (2560 × 10240) × 0.25 bytes = 13.1 MB
- W_down: (10240 × 2560) × 0.25 bytes = 6.5 MB

RMSNorm:
- attn_norm, ffn_norm: 2 × 2560 × 4 bytes = 20 KB

Total: ~26 MB per layer
32 layers: ~832 MB
```

## Performance

**Compute per token** (BitNet-3B):
```
Attention: 285 GFLOPS
FFN:
- Gate projection: 52.4 GFLOPS
- Up projection: 52.4 GFLOPS
- SiLU + multiply: 0.05 GFLOPS
- Down projection: 104.8 GFLOPS
FFN Total: 209.7 GFLOPS

RMSNorm: 0.02 GFLOPS (negligible)

Total per layer: ~495 GFLOPS
```

## References
- LLaMA: https://github.com/facebookresearch/llama
- BitNet: https://arxiv.org/abs/2402.17764
- llama.cpp: llama.cpp (llm_load_tensors, llm_build_graph)
-/

namespace Hesper.Layers.TransformerBlock

open Hesper.WGSL
open Hesper.WGSL.Elementwise
open Hesper.WebGPU
open Hesper.Layers
open Hesper.Logging (logVerbose isVerbose)

/-! ## Configuration -/

/-- Transformer block configuration -/
structure Config where
  layerIdx : Nat      -- Layer index (0-31 for BitNet-3B)
  dim : Nat           -- Model dimension (hidden size)
  ffnDim : Nat        -- FFN intermediate dimension (typically 4× dim)
  numHeads : Nat      -- Number of attention heads
  maxSeqLen : Nat     -- Maximum sequence length
  deriving Repr

/-! ## Layer Structure -/

/-- Complete transformer block -/
structure TransformerBlock where
  config : Config
  -- Pre-attention normalization
  attnNorm : RMSNorm.RMSNorm
  -- Multi-head self-attention
  attention : Attention.Attention
  -- Attention sub-norm (applied after attention, before O projection)
  attnSubNorm : RMSNorm.RMSNorm
  -- Pre-FFN normalization
  ffnNorm : RMSNorm.RMSNorm
  -- FFN projections (all using BitLinear with ternary weights)
  ffnGate : BitLinear.BitLinear
  ffnUp : BitLinear.BitLinear
  -- FFN sub-norm (applied after ReLU²(gate)*up, before down projection)
  ffnSubNorm : RMSNorm.RMSNorm
  ffnDown : BitLinear.BitLinear

/-! ## Layer Creation -/

/-- Create transformer block from GGUF tensors

    @param device WebGPU device
    @param config Block configuration
    @param attnNormData Pre-attention RMSNorm scale parameters
    @param ffnNormData Pre-FFN RMSNorm scale parameters
    @param attnWeights Attention layer weight data (Q, K, V, O + scales)
    @param ffnGateData FFN gate projection weights + scales
    @param ffnUpData FFN up projection weights + scales
    @param ffnDownData FFN down projection weights + scales
-/
def create (device : Device) (config : Config)
           (attnNormData attnSubNormData ffnNormData ffnSubNormData : ByteArray)
           (attnWeights : Attention.Attention)
           (ffnGateData ffnUpData ffnDownData : ByteArray × ByteArray) : IO TransformerBlock := do
  IO.println s!"[TransformerBlock] Creating layer {config.layerIdx}..."

  -- Create RMSNorm layers
  let attnNormConfig : RMSNorm.Config := { dim := config.dim }
  let attnNorm ← RMSNorm.create device attnNormConfig attnNormData

  let attnSubNormConfig : RMSNorm.Config := { dim := config.dim }
  let attnSubNorm ← RMSNorm.create device attnSubNormConfig attnSubNormData

  let ffnNormConfig : RMSNorm.Config := { dim := config.dim }
  let ffnNorm ← RMSNorm.create device ffnNormConfig ffnNormData

  let ffnSubNormConfig : RMSNorm.Config := { dim := config.ffnDim }
  let ffnSubNorm ← RMSNorm.create device ffnSubNormConfig ffnSubNormData

  -- Attention layer is passed in (already created)

  -- Create FFN BitLinear layers
  let ffnGateConfig : BitLinear.Config := {
    inDim := config.dim,
    outDim := config.ffnDim,
    batchSize := 1
  }
  let (ffnGateWeights, ffnGateScales) := ffnGateData
  let ffnGate ← BitLinear.createFromBytes device ffnGateConfig ffnGateWeights ffnGateScales

  let ffnUpConfig : BitLinear.Config := {
    inDim := config.dim,
    outDim := config.ffnDim,
    batchSize := 1
  }
  let (ffnUpWeights, ffnUpScales) := ffnUpData
  let ffnUp ← BitLinear.createFromBytes device ffnUpConfig ffnUpWeights ffnUpScales

  let ffnDownConfig : BitLinear.Config := {
    inDim := config.ffnDim,
    outDim := config.dim,
    batchSize := 1
  }
  let (ffnDownWeights, ffnDownScales) := ffnDownData
  let ffnDown ← BitLinear.createFromBytes device ffnDownConfig ffnDownWeights ffnDownScales

  IO.println s!"[TransformerBlock] ✓ Layer {config.layerIdx} created"
  pure { config, attnNorm, attnSubNorm, attention := attnWeights, ffnNorm,
         ffnGate, ffnUp, ffnSubNorm, ffnDown }

/-- Create transformer block with pre-built BitLinear layers

    @param device WebGPU device
    @param config Block configuration
    @param attnNormData Pre-attention RMSNorm scale parameters
    @param ffnNormData Pre-FFN RMSNorm scale parameters
    @param attnWeights Already-created Attention layer
    @param ffnGateLayer Already-created FFN gate BitLinear
    @param ffnUpLayer Already-created FFN up BitLinear
    @param ffnDownLayer Already-created FFN down BitLinear
-/
def createWithLayers (device : Device) (config : Config)
           (attnNormData attnSubNormData ffnNormData ffnSubNormData : ByteArray)
           (attnWeights : Attention.Attention)
           (ffnGateLayer ffnUpLayer ffnDownLayer : BitLinear.BitLinear) : IO TransformerBlock := do
  IO.println s!"[TransformerBlock] Creating layer {config.layerIdx}..."

  -- Create RMSNorm layers
  let attnNormConfig : RMSNorm.Config := { dim := config.dim }
  let attnNorm ← RMSNorm.create device attnNormConfig attnNormData

  let attnSubNormConfig : RMSNorm.Config := { dim := config.dim }
  let attnSubNorm ← RMSNorm.create device attnSubNormConfig attnSubNormData

  let ffnNormConfig : RMSNorm.Config := { dim := config.dim }
  let ffnNorm ← RMSNorm.create device ffnNormConfig ffnNormData

  let ffnSubNormConfig : RMSNorm.Config := { dim := config.ffnDim }
  let ffnSubNorm ← RMSNorm.create device ffnSubNormConfig ffnSubNormData

  IO.println s!"[TransformerBlock] ✓ Layer {config.layerIdx} created"
  pure { config, attnNorm, attnSubNorm, attention := attnWeights, ffnNorm,
         ffnGate := ffnGateLayer, ffnUp := ffnUpLayer, ffnSubNorm, ffnDown := ffnDownLayer }

/-! ## Forward Pass -/

/-- Execute transformer block forward pass

    **Algorithm**:
    ```
    # Attention sub-layer
    1. normed = RMSNorm(input)
    2. attn_out = MultiHeadAttention(normed)
    3. residual1 = input + attn_out  # Residual connection

    # FFN sub-layer
    4. normed2 = RMSNorm(residual1)
    5. gate = W_gate @ normed2
    6. up = W_up @ normed2
    7. hidden = ReLU²(gate) × up  # Gated activation (BitNet uses relu_sqr)
    8. ffn_out = W_down @ hidden
    9. residual2 = residual1 + ffn_out  # Residual connection

    return residual2
    ```

    @param device WebGPU device
    @param block Transformer block
    @param inputBuf Input tensor [batch, seq_len, dim]
    @param outputBuf Output tensor [batch, seq_len, dim]
    @param batchSize Batch size
    @param seqLen Sequence length
-/
def forward (device : Device) (block : TransformerBlock)
            (inputBuf outputBuf : Buffer)
            (batchSize seqLen : Nat) : IO Unit := do
  logVerbose s!"[Block {block.config.layerIdx}] Forward pass: batch={batchSize}, seq_len={seqLen}"

  let numElements := batchSize * seqLen * block.config.dim
  let ffnElements := batchSize * seqLen * block.config.ffnDim

  -- Allocate temporary buffers
  let tempSize := (numElements * 4).toUSize  -- Float32
  let ffnTempSize := (ffnElements * 4).toUSize

  let normedBuf ← createBuffer device { size := tempSize, usage := [.storage], mappedAtCreation := false }
  let attnOutBuf ← createBuffer device { size := tempSize, usage := [.storage], mappedAtCreation := false }
  let residual1Buf ← createBuffer device { size := tempSize, usage := [.storage], mappedAtCreation := false }
  let normed2Buf ← createBuffer device { size := tempSize, usage := [.storage], mappedAtCreation := false }
  let gateBuf ← createBuffer device { size := ffnTempSize, usage := [.storage], mappedAtCreation := false }
  let upBuf ← createBuffer device { size := ffnTempSize, usage := [.storage], mappedAtCreation := false }
  let hiddenBuf ← createBuffer device { size := ffnTempSize, usage := [.storage], mappedAtCreation := false }
  let ffnOutBuf ← createBuffer device { size := tempSize, usage := [.storage], mappedAtCreation := false }

  -- === ATTENTION SUB-LAYER ===

  -- Step 1: Pre-attention normalization
  let numRows := batchSize * seqLen
  logVerbose "  [1/9] Pre-attention RMSNorm..."
  RMSNorm.forward device block.attnNorm inputBuf normedBuf numRows

  -- Debug: check RMSNorm output (layer 0 only)
  if block.config.layerIdx == 0 && (← isVerbose) then
    let dbg1 ← Hesper.WebGPU.BufferOps.downloadFloatArray device normedBuf (min 5 numElements)
    logVerbose s!"  [DEBUG] RMSNorm output (first 5): {dbg1.toList.take 5}"

  -- Step 2: Multi-head self-attention (with attn sub-norm before O projection)
  logVerbose "  [2/9] Multi-head attention..."
  Attention.forward device block.attention normedBuf attnOutBuf batchSize seqLen (some block.attnSubNorm)

  -- Debug: check attention output (layer 0 only)
  if block.config.layerIdx == 0 && (← isVerbose) then
    let dbg2 ← Hesper.WebGPU.BufferOps.downloadFloatArray device attnOutBuf (min 5 numElements)
    logVerbose s!"  [DEBUG] Attention output (first 5): {dbg2.toList.take 5}"

  -- Step 3: Residual connection
  logVerbose "  [3/9] Attention residual connection..."
  let elemConfig : Elementwise.Config := { numElements }
  executeAdd device inputBuf attnOutBuf residual1Buf elemConfig

  -- === FFN SUB-LAYER ===

  -- Step 4: Pre-FFN normalization
  logVerbose "  [4/9] Pre-FFN RMSNorm..."
  RMSNorm.forward device block.ffnNorm residual1Buf normed2Buf numRows

  -- Step 5: FFN gate projection
  logVerbose "  [5/9] FFN gate projection..."
  BitLinear.forward device block.ffnGate normed2Buf gateBuf numRows

  -- Step 6: FFN up projection
  logVerbose "  [6/9] FFN up projection..."
  BitLinear.forward device block.ffnUp normed2Buf upBuf numRows

  -- Step 7: Gated activation (ReLU² + multiply, BitNet b1.58 uses LLM_FFN_RELU_SQR)
  logVerbose "  [7/10] Gated activation (ReLU²(gate) × up)..."
  let ffnElemConfig : Elementwise.Config := { numElements := ffnElements }
  executeReluSqrMul device gateBuf upBuf hiddenBuf ffnElemConfig

  -- Debug: check FFN intermediate (layer 0 only)
  if block.config.layerIdx == 0 && (← isVerbose) then
    let dbgGate ← Hesper.WebGPU.BufferOps.downloadFloatArray device gateBuf (min 5 ffnElements)
    logVerbose s!"  [DEBUG] Gate output (first 5): {dbgGate.toList.take 5}"
    let dbgUp ← Hesper.WebGPU.BufferOps.downloadFloatArray device upBuf (min 5 ffnElements)
    logVerbose s!"  [DEBUG] Up output (first 5): {dbgUp.toList.take 5}"
    let dbgHidden ← Hesper.WebGPU.BufferOps.downloadFloatArray device hiddenBuf (min 5 ffnElements)
    logVerbose s!"  [DEBUG] ReLU²*Up output (first 5): {dbgHidden.toList.take 5}"

  -- Step 7.5: FFN sub-norm (normalize before down projection - BitNet specific)
  logVerbose "  [7.5/10] FFN sub-norm..."
  let ffnNormedBuf ← createBuffer device { size := ffnTempSize, usage := [.storage], mappedAtCreation := false }
  RMSNorm.forward device block.ffnSubNorm hiddenBuf ffnNormedBuf numRows

  -- Debug: check FFN normed (layer 0 only)
  if block.config.layerIdx == 0 && (← isVerbose) then
    let dbgNormed ← Hesper.WebGPU.BufferOps.downloadFloatArray device ffnNormedBuf (min 5 ffnElements)
    logVerbose s!"  [DEBUG] FFN sub-normed (first 5): {dbgNormed.toList.take 5}"

  -- Step 8: FFN down projection
  logVerbose "  [8/10] FFN down projection..."
  BitLinear.forward device block.ffnDown ffnNormedBuf ffnOutBuf numRows

  -- Debug: check FFN output (layer 0 only)
  if block.config.layerIdx == 0 && (← isVerbose) then
    let dbgDown ← Hesper.WebGPU.BufferOps.downloadFloatArray device ffnOutBuf (min 5 numElements)
    logVerbose s!"  [DEBUG] FFN down output (first 5): {dbgDown.toList.take 5}"
    let dbgRes1 ← Hesper.WebGPU.BufferOps.downloadFloatArray device residual1Buf (min 5 numElements)
    logVerbose s!"  [DEBUG] Residual1 (pre-FFN residual, first 5): {dbgRes1.toList.take 5}"

  -- Step 9: Final residual connection
  logVerbose "  [9/10] FFN residual connection..."
  executeAdd device residual1Buf ffnOutBuf outputBuf elemConfig

  logVerbose s!"[Block {block.config.layerIdx}] ✓ Forward pass complete"

/-! ## Integration with GGUF -/

/-- Create transformer block from GGUF file

    Loads all tensors for a specific transformer layer.

    Example tensor names in GGUF:
    - `blk.{idx}.attn_norm.weight` - Pre-attention RMSNorm
    - `blk.{idx}.attn_q.weight` - Attention Q projection
    - `blk.{idx}.attn_k.weight` - Attention K projection
    - `blk.{idx}.attn_v.weight` - Attention V projection
    - `blk.{idx}.attn_output.weight` - Attention output projection
    - `blk.{idx}.ffn_norm.weight` - Pre-FFN RMSNorm
    - `blk.{idx}.ffn_gate.weight` - FFN gate projection
    - `blk.{idx}.ffn_up.weight` - FFN up projection
    - `blk.{idx}.ffn_down.weight` - FFN down projection

    @param device WebGPU device
    @param gguf Loaded GGUF file
    @param config Block configuration
-/
def fromGGUF (device : Device) (gguf : α) (config : Config) : IO TransformerBlock := do
  -- Placeholder - actual implementation would:
  -- 1. Find all tensors for this layer: gguf.findTensor s!"blk.{config.layerIdx}...."
  -- 2. Extract TQ2_0 packed data + scales for each weight matrix
  -- 3. Extract Float32 data for RMSNorm scales
  -- 4. Call create() with extracted data
  IO.println s!"[TransformerBlock] Loading layer {config.layerIdx} from GGUF"
  throw $ IO.userError "fromGGUF not yet implemented - use create() directly"

end Hesper.Layers.TransformerBlock
