import Hesper.WGSL.Monad
import Hesper.WGSL.Execute
import Hesper.WGSL.Exp
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.Quantization.TQ2_0
import Hesper.Basic
import Hesper.Logging

/-!
# BitLinear Layer - Ternary Weight Matrix Multiplication (i2_s format)

Implements BitNet's BitLinear layer with on-the-fly i2_s dequantization on GPU.

## Key Innovation: Fused Kernel
Instead of:
1. Unpack i2_s → Float32 on CPU (slow, memory-intensive)
2. Matrix multiply Float32 × Float32

We do:
1. Upload raw packed i2_s bytes to GPU
2. **Read packed weights + compute matmul in same kernel**

## i2_s Packing Format

Encoding table:
| Ternary | 2-bit code |
|---------|-----------|
| -1      | 0b00 (0)  |
|  0      | 0b01 (1)  |
| +1      | 0b10 (2)  |

Dequantization: `float_value = (code - 1) * scale`

Layout (groups of 128 elements per 32 bytes):
- Elements [0..31]:    bytes[0..31] >> 6 & 3
- Elements [32..63]:   bytes[0..31] >> 4 & 3
- Elements [64..95]:   bytes[0..31] >> 2 & 3
- Elements [96..127]:  bytes[0..31] >> 0 & 3

Scale: single F32 at the END of tensor data.

## BitLinear Mathematics

For ternary weights w in {-1, 0, 1}, the matrix-vector product:

```
y[i] = scale * sum_j( w[i,j] * x[j] )
     = scale * (sum_{w=+1} x[j] - sum_{w=-1} x[j])
```

## References
- BitNet paper: https://arxiv.org/abs/2402.17764
- bitnet.cpp: i2_s format specification
-/

namespace Hesper.Layers.BitLinear

open Hesper.WGSL
open Hesper.WGSL.Monad
open Hesper.WebGPU
open Hesper.Logging (logVerbose)

/-! ## Layer Configuration -/

/-- BitLinear layer configuration -/
structure Config where
  inDim : Nat      -- Input dimension
  outDim : Nat     -- Output dimension
  batchSize : Nat  -- Batch size
  deriving Repr, Inhabited

/-! ## Fused BitLinear Kernel (i2_s format) -/

/-- Fused kernel: i2_s unpack + matrix-vector multiply

    **Algorithm**:
    ```
    for each output element y[out_idx]:
      acc = 0.0
      for each input element x[in_idx]:
        w = unpack_i2s(weights[out_idx * inDim + in_idx])
        ternary = w - 1  // gives {-1, 0, +1}
        acc += f32(ternary) * x[in_idx]
      y[out_idx] = scale * acc
    ```

    **Workgroup strategy**: Each thread computes one output element.
    Uses ShaderM.loop for runtime WGSL for-loop over inDim.

    **i2_s unpacking** (no row permutation):
    GGUF i2_s stores rows in natural order. Decode using group128 descending shift:
    ```
    group128 = elemIdx / 128
    local128 = elemIdx % 128
    byteIdx = group128 * 32 + (local128 % 32)
    shift = 6 - (local128 / 32) * 2
    code = (data[byteIdx] >> shift) & 0x3
    ternary = code - 1
    ```

    @param config BitLinear layer configuration
-/
def fusedBitLinearKernel (config : Config) (numRows : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let globalIdx := Exp.vec3X gid

  let totalOutputs := numRows * config.outDim
  -- Bounds check
  let inBounds := Exp.lt globalIdx (Exp.litU32 totalOutputs)

  -- Determine row and column from global thread index
  let rowIdx := Exp.div globalIdx (Exp.litU32 config.outDim)
  let outIdx := Exp.mod globalIdx (Exp.litU32 config.outDim)

  -- Total weight elements and packed buffer sizes
  let totalWeightElements := config.outDim * config.inDim
  let numPackedBytes := totalWeightElements / 4  -- 4 elements per byte (2 bits each)
  let numPackedU32 := (numPackedBytes + 3) / 4  -- Round up to u32 count
  let totalInputElements := numRows * config.inDim

  -- Declare buffers
  -- weights_packed: raw i2_s packed bytes stored as u32 array
  let _packed ← ShaderM.declareInputBuffer "weights_packed" (.array (.scalar .u32) numPackedU32)
  -- scale: single f32 value stored in 1-element array
  let _scale ← ShaderM.declareInputBuffer "scale" (.array (.scalar .f32) 1)
  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) totalInputElements)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) totalOutputs)

  -- Initialize accumulator
  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"

  -- Loop over input dimension (WGSL runtime for loop)
  ShaderM.loop (Exp.litU32 0) (Exp.litU32 config.inDim) (Exp.litU32 1) fun j => do
    -- No row permutation needed: GGUF i2_s stores rows in natural order.
    -- The group128 packing is within each row, not across rows.

    -- Flat element index using output row directly
    let elemIdx := Exp.add (Exp.mul outIdx (Exp.litU32 config.inDim)) j

    -- i2_s unpacking: group128 descending shift
    -- 128-element blocks, 32 bytes per block
    -- shift pattern: elements [0..31] at shift 6, [32..63] at shift 4, etc.
    let group128 := Exp.div elemIdx (Exp.litU32 128)
    let local128 := Exp.mod elemIdx (Exp.litU32 128)
    let groupIdx := Exp.div local128 (Exp.litU32 32)
    let groupPos := Exp.mod local128 (Exp.litU32 32)
    let byteIdx := Exp.add (Exp.mul group128 (Exp.litU32 32)) groupPos
    let shift := Exp.sub (Exp.litU32 6) (Exp.mul groupIdx (Exp.litU32 2))

    -- Read the u32 containing this byte
    let u32Idx := Exp.div byteIdx (Exp.litU32 4)
    let packedU32 ← ShaderM.readBuffer (ty := .scalar .u32) (n := numPackedU32) "weights_packed" u32Idx

    -- Extract the specific byte from u32 (little-endian)
    let byteInU32 := Exp.mod byteIdx (Exp.litU32 4)
    let byteShift := Exp.mul byteInU32 (Exp.litU32 8)
    let theByte := Exp.bitAnd (Exp.shiftRight packedU32 byteShift) (Exp.litU32 0xFF)

    -- Extract 2-bit code from the byte
    let code := Exp.bitAnd (Exp.shiftRight theByte shift) (Exp.litU32 0x3)

    -- Convert to ternary: code - 1 gives {-1, 0, +1}
    let ternaryF32 := Exp.sub (Exp.toF32 code) (Exp.litF32 1.0)

    -- Read input value from THIS ROW: input[rowIdx * inDim + j]
    let inputIdx := Exp.add (Exp.mul rowIdx (Exp.litU32 config.inDim)) j
    let inputVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalInputElements) "input" inputIdx

    -- Accumulate: acc += ternary * input
    let product := Exp.mul ternaryF32 inputVal
    ShaderM.assign "acc" (Exp.add acc product)

  -- Read scale (single f32 value)
  let scaleVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := 1) "scale" (Exp.litU32 0)

  -- Compute final output: scale * acc
  let result := Exp.mul scaleVal acc

  -- Write output (conditional on bounds)
  let finalResult := Exp.select inBounds result (Exp.litF32 0.0)
  ShaderM.writeBuffer (ty := .scalar .f32) "output" globalIdx finalResult

/-! ## High-Level API -/

/-- BitLinear layer structure -/
structure BitLinear where
  config : Config
  weightsPacked : Buffer   -- i2_s packed weights (raw bytes as u32 array)
  scaleBuf : Buffer        -- Single f32 scale value

/-- Create BitLinear layer from i2_s packed data

    @param device WebGPU device
    @param config Layer configuration
    @param packedWeights Raw i2_s packed byte data from GGUF
    @param scale Float32 scale factor for the ternary weights
-/
def create (device : Device) (config : Config)
           (packedWeights : ByteArray) (scale : Float) : IO BitLinear := do
  IO.println s!"[BitLinear] Creating layer: {config.inDim} -> {config.outDim}, scale={scale}"

  -- Pad packed weights to u32 alignment if needed
  let paddedWeights ← do
    if packedWeights.size % 4 == 0 then pure packedWeights
    else do
      let padding := 4 - (packedWeights.size % 4)
      let mut w := packedWeights
      for _ in [0:padding] do
        w := w.push 0
      pure w

  -- Create GPU buffer for packed weights
  let bufSize := if paddedWeights.size == 0 then 4 else paddedWeights.size
  let weightsBuf ← createBuffer device {
    size := bufSize.toUSize
    usage := [.storage, .copyDst]
    mappedAtCreation := false
  }
  if paddedWeights.size > 0 then
    writeBuffer device weightsBuf 0 paddedWeights

  -- Create GPU buffer for scale (single f32 = 4 bytes)
  let scaleBuf ← createBuffer device {
    size := 4
    usage := [.storage, .copyDst]
    mappedAtCreation := false
  }
  -- Encode scale as f32 bytes (little-endian) via FFI (proper f64→f32 conversion)
  let scaleBytes ← Hesper.Basic.floatToBytes scale
  writeBuffer device scaleBuf 0 scaleBytes

  IO.println s!"[BitLinear] Layer created: packed={paddedWeights.size} bytes"
  pure { config, weightsPacked := weightsBuf, scaleBuf := scaleBuf }

/-- Create BitLinear layer from packed data + scale ByteArrays

    Alternative constructor that takes pre-encoded scale bytes.
    Used when the caller has already extracted the raw data.

    @param device WebGPU device
    @param config Layer configuration
    @param packedWeights Raw i2_s packed byte data
    @param scaleBytes 4-byte little-endian F32 scale
-/
def createFromBytes (device : Device) (config : Config)
                    (packedWeights : ByteArray) (scaleBytes : ByteArray) : IO BitLinear := do
  IO.println s!"[BitLinear] Creating layer from bytes: {config.inDim} -> {config.outDim}"

  -- Pad packed weights to u32 alignment if needed
  let paddedWeights ← do
    if packedWeights.size % 4 == 0 then pure packedWeights
    else do
      let padding := 4 - (packedWeights.size % 4)
      let mut w := packedWeights
      for _ in [0:padding] do
        w := w.push 0
      pure w

  -- Create GPU buffer for packed weights
  let bufSize := if paddedWeights.size == 0 then 4 else paddedWeights.size
  let weightsBuf ← createBuffer device {
    size := bufSize.toUSize
    usage := [.storage, .copyDst]
    mappedAtCreation := false
  }
  if paddedWeights.size > 0 then
    writeBuffer device weightsBuf 0 paddedWeights

  -- Create GPU buffer for scale (single f32 = 4 bytes)
  let scaleBuf ← createBuffer device {
    size := (scaleBytes.size.max 4).toUSize
    usage := [.storage, .copyDst]
    mappedAtCreation := false
  }
  writeBuffer device scaleBuf 0 scaleBytes

  IO.println s!"[BitLinear] Layer created: packed={paddedWeights.size} bytes"
  pure { config, weightsPacked := weightsBuf, scaleBuf := scaleBuf }

/-- Execute forward pass

    @param device WebGPU device
    @param layer BitLinear layer
    @param inputBuf GPU buffer containing input (Float32)
    @param outputBuf GPU buffer for output (Float32)
-/
def forward (device : Device) (layer : BitLinear)
            (inputBuf outputBuf : Buffer) (numRows : Nat := 1) : IO Unit := do
  logVerbose s!"[BitLinear] Executing forward pass ({numRows} rows, {layer.config.inDim}→{layer.config.outDim})..."

  let shader := fusedBitLinearKernel layer.config numRows
  let namedBuffers := [
    ("weights_packed", layer.weightsPacked),
    ("scale", layer.scaleBuf),
    ("input", inputBuf),
    ("output", outputBuf)
  ]

  let totalOutputs := numRows * layer.config.outDim
  let execConfig := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D
    totalOutputs
    256  -- Workgroup size

  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig
  logVerbose "[BitLinear] Forward pass complete"

end Hesper.Layers.BitLinear
