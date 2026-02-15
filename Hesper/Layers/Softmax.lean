import Hesper.WGSL.Monad
import Hesper.WGSL.Execute
import Hesper.WGSL.Exp
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.Logging

/-!
# Softmax - Numerically Stable Implementation

Implements softmax activation for attention scores.

## Mathematical Definition

Standard softmax:
```
softmax(x)ᵢ = exp(xᵢ) / Σⱼ exp(xⱼ)
```

**Problem**: Numerical instability when xᵢ is large (exp overflow)

## Numerically Stable Softmax

```
1. Find maximum: M = max(x)
2. Subtract max:  y = x - M  (shifts values to ≤ 0)
3. Exponentiate:  z = exp(y)
4. Normalize:     softmax(x) = z / sum(z)
```

**Why stable**:
- Largest value becomes exp(0) = 1 (no overflow)
- All other values are exp(negative) ∈ (0, 1)
- Mathematically equivalent due to cancellation:
  ```
  exp(xᵢ - M) / Σⱼ exp(xⱼ - M)
  = [exp(xᵢ) / exp(M)] / [Σⱼ exp(xⱼ) / exp(M)]
  = exp(xᵢ) / Σⱼ exp(xⱼ)
  ```

## Attention Masking

In causal (autoregressive) attention, we mask future positions:
```
mask[i,j] = -∞ if j > i  (can't attend to future)
mask[i,j] = 0   if j ≤ i  (can attend to past)

softmax_scores = softmax(attention_scores + mask)
```

After softmax, masked positions become ≈ 0.

## Implementation Strategy

**Two-pass approach** (memory-efficient):
```
Pass 1: Parallel reduction to find max(x) and sum(exp(x - max))
Pass 2: Normalize each element: xᵢ / sum
```

**Single-pass approach** (simplified):
```
Each workgroup handles one row (one query token's attention to all keys)
Uses shared memory for max reduction and sum reduction
```

## References
- Attention is All You Need (Vaswani et al., 2017)
- llama.cpp: ggml/src/ggml.c (ggml_soft_max_impl)
- Flash Attention: https://arxiv.org/abs/2205.14135
-/

namespace Hesper.Layers.Softmax

open Hesper.WGSL
open Hesper.WGSL.Monad
open Hesper.WebGPU
open Hesper.Logging (logVerbose)

/-! ## Configuration -/

/-- Softmax configuration -/
structure Config where
  rowSize : Nat       -- Size of each row to normalize
  numRows : Nat       -- Number of independent rows
  useMask : Bool := false  -- Whether to apply causal masking
  deriving Repr

/-! ## GPU Kernel Implementation -/

/-- Softmax kernel with numerical stability

    **Input shape**: [num_rows, row_size]
    **Output shape**: [num_rows, row_size]

    **Algorithm**:
    ```
    for each row:
      1. Find max value: M = max(row)
      2. Compute shifted exp: zᵢ = exp(xᵢ - M)
      3. Compute sum: S = Σᵢ zᵢ
      4. Normalize: yᵢ = zᵢ / S
    ```

    **Workgroup strategy**:
    - One workgroup per row (for small sequences)
    - Parallel reduction for max/sum within workgroup
    - Shared memory for intermediate results

    @param config Softmax configuration
-/
def softmaxKernel (config : Config) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  -- Each thread processes one element
  let totalElements := config.numRows * config.rowSize
  let inBounds := Exp.lt idx (Exp.litU32 totalElements)

  -- Declare buffers
  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) totalElements)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) totalElements)

  -- Decompose index: row and column
  let row := Exp.div idx (Exp.litU32 config.rowSize)
  let col := Exp.mod idx (Exp.litU32 config.rowSize)

  -- Read input value
  let inputVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalElements) "input" idx

  -- Step 1: Find max in row (simplified - first thread does it)
  -- In production, use workgroup reduction
  let isFirstInRow := Exp.eq col (Exp.litU32 0)

  let mut maxVal := Exp.litF32 (-3.4e38)  -- -FLT_MAX
  if config.rowSize <= 256 then  -- Simplified for small rows
    for i in [0:min 256 config.rowSize] do
      let rowStart := Exp.mul row (Exp.litU32 config.rowSize)
      let elemIdx := Exp.add rowStart (Exp.litU32 i)
      let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalElements) "input" elemIdx
      maxVal := Exp.max maxVal val

  -- Step 2: Compute exp(x - max)
  let shifted := Exp.sub inputVal maxVal
  let expVal := Exp.exp shifted

  -- Step 3: Compute sum of exp values (simplified - first thread does it)
  let mut sumExp := Exp.litF32 0.0
  if config.rowSize <= 256 then
    for i in [0:min 256 config.rowSize] do
      let rowStart := Exp.mul row (Exp.litU32 config.rowSize)
      let elemIdx := Exp.add rowStart (Exp.litU32 i)
      let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalElements) "input" elemIdx
      let shiftedVal := Exp.sub val maxVal
      let expValTemp := Exp.exp shiftedVal
      sumExp := Exp.add sumExp expValTemp

  -- Step 4: Normalize
  let result := Exp.div expVal sumExp

  -- Write output
  let finalResult := Exp.select inBounds result (Exp.litF32 0.0)
  ShaderM.writeBuffer (ty := .scalar .f32) "output" idx finalResult

/-- Softmax with causal masking

    Applies causal mask before softmax: mask[i,j] = -∞ if j > i

    @param config Softmax configuration (useMask should be true)
-/
def softmaxMaskedKernel (config : Config) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let totalElements := config.numRows * config.rowSize
  let inBounds := Exp.lt idx (Exp.litU32 totalElements)

  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) totalElements)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) totalElements)

  let row := Exp.div idx (Exp.litU32 config.rowSize)
  let col := Exp.mod idx (Exp.litU32 config.rowSize)

  -- Sequence position within each attention head (row % rowSize)
  -- Layout: [batch * numHeads * seqLen, seqLen], position = row % seqLen
  let position := Exp.mod row (Exp.litU32 config.rowSize)

  -- Read input
  let inputVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalElements) "input" idx

  -- Apply causal mask: if col > position, set to -∞
  let isMasked := Exp.gt col position
  let maskedVal := Exp.select isMasked (Exp.litF32 (-3.4e38)) inputVal

  -- Find max (considering masked values)
  let mut maxVal := Exp.litF32 (-3.4e38)
  if config.rowSize <= 256 then
    for i in [0:min 256 config.rowSize] do
      let rowStart := Exp.mul row (Exp.litU32 config.rowSize)
      let elemIdx := Exp.add rowStart (Exp.litU32 i)
      let colIdx := Exp.litU32 i
      let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalElements) "input" elemIdx

      -- Apply mask to this value too
      let isMaskedTemp := Exp.gt colIdx position
      let maskedValTemp := Exp.select isMaskedTemp (Exp.litF32 (-3.4e38)) val

      maxVal := Exp.max maxVal maskedValTemp

  -- Compute exp(x - max) for masked value
  let shifted := Exp.sub maskedVal maxVal
  let expVal := Exp.exp shifted

  -- Sum exp values
  let mut sumExp := Exp.litF32 0.0
  if config.rowSize <= 256 then
    for i in [0:min 256 config.rowSize] do
      let rowStart := Exp.mul row (Exp.litU32 config.rowSize)
      let elemIdx := Exp.add rowStart (Exp.litU32 i)
      let colIdx := Exp.litU32 i
      let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalElements) "input" elemIdx

      let isMaskedTemp := Exp.gt colIdx position
      let maskedValTemp := Exp.select isMaskedTemp (Exp.litF32 (-3.4e38)) val

      let shiftedVal := Exp.sub maskedValTemp maxVal
      let expValTemp := Exp.exp shiftedVal
      sumExp := Exp.add sumExp expValTemp

  -- Normalize
  let result := Exp.div expVal sumExp

  let finalResult := Exp.select inBounds result (Exp.litF32 0.0)
  ShaderM.writeBuffer (ty := .scalar .f32) "output" idx finalResult

/-! ## High-Level API -/

/-- Softmax layer structure -/
structure Softmax where
  config : Config

/-- Create Softmax layer (no learned parameters)

    @param config Softmax configuration
-/
def create (config : Config) : IO Softmax := do
  logVerbose s!"[Softmax] Creating layer: rows={config.numRows}, row_size={config.rowSize}, masked={config.useMask}"
  pure { config }

/-- Apply softmax

    @param device WebGPU device
    @param layer Softmax layer
    @param inputBuf GPU buffer [num_rows, row_size]
    @param outputBuf GPU buffer for output (same shape)
-/
def forward (device : Device) (layer : Softmax)
            (inputBuf outputBuf : Buffer) : IO Unit := do
  logVerbose "[Softmax] Applying softmax..."

  let shader := if layer.config.useMask then
    softmaxMaskedKernel layer.config
  else
    softmaxKernel layer.config

  let namedBuffers := [
    ("input", inputBuf),
    ("output", outputBuf)
  ]

  let totalElements := layer.config.numRows * layer.config.rowSize
  let execConfig := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D
    totalElements
    256  -- Workgroup size

  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig
  logVerbose "[Softmax] ✓ Forward pass complete"

/-! ## Optimized: Flash Softmax (Future) -/

/-- Flash Softmax: Fused softmax + attention matmul

    This is part of Flash Attention optimization, where softmax is fused
    with the attention score computation to reduce memory bandwidth.

    **Key idea**:
    Instead of:
    1. Compute Q @ K^T (write to memory)
    2. Softmax (read from memory, write back)
    3. @ V (read from memory)

    Do:
    1. Compute chunks of Q @ K^T in registers
    2. Apply softmax to chunks
    3. Multiply by V chunk immediately
    4. Accumulate result (never write intermediate scores)

    **Benefits**:
    - 2-4x speedup on long sequences
    - O(N) memory instead of O(N²)
    - Enables much longer context windows

    This is marked for future implementation when integrating with Attention layer.
-/
def flashSoftmaxKernel (config : Config) : ShaderM Unit := do
  -- Placeholder for Flash Attention integration
  -- TODO: Implement fused softmax + matmul for Flash Attention
  pure ()

end Hesper.Layers.Softmax
