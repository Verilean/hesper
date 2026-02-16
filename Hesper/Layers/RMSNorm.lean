import Hesper.WGSL.Monad
import Hesper.WGSL.Execute
import Hesper.WGSL.Exp
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.Logging

/-!
# RMSNorm Layer - Root Mean Square Normalization

Implements RMSNorm as used in BitNet and Llama architectures.

## Mathematical Definition

RMSNorm normalizes a vector by its root mean square:

```
RMS(x) = sqrt(1/n * Σᵢ xᵢ²)
y = (x / RMS(x)) * γ
```

Where:
- x: input vector (n elements)
- γ: learned scale parameter (n elements)
- ε: small constant for numerical stability (typically 1e-5 or 1e-6)

## Comparison with LayerNorm

**LayerNorm**: `y = γ * (x - μ) / σ + β`
- Requires mean (μ) and variance (σ) calculation
- Has bias term (β)
- More expensive (two passes + bias)

**RMSNorm**: `y = γ * x / RMS(x)`
- No mean subtraction (assumes mean ≈ 0 from prior layers)
- No bias term
- Simpler and faster (single pass)

## Performance Advantages

1. **Simpler computation**: No mean calculation
2. **Better gradients**: Avoids mean-centering instability
3. **Fused implementation**: RMS + scale in single kernel

## References
- Llama: https://github.com/facebookresearch/llama (RMSNorm implementation)
- Paper: "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
- llama.cpp: ggml/src/ggml.c (ggml_rms_norm implementation)
-/

namespace Hesper.Layers.RMSNorm

open Hesper.WGSL
open Hesper.WGSL.Monad
open Hesper.WebGPU
open Hesper.Logging (logVerbose)

/-- Counters for PreparedDispatch fast-path vs slow-path -/
initialize preparedHitsRef : IO.Ref Nat ← IO.mkRef 0
initialize preparedMissesRef : IO.Ref Nat ← IO.mkRef 0

/-! ## Layer Configuration -/

/-- RMSNorm layer configuration -/
structure Config where
  dim : Nat           -- Hidden dimension
  eps : Float := 1e-5 -- Epsilon for numerical stability
  deriving Repr

/-! ## GPU Kernel Implementation -/

/-- RMSNorm kernel using workgroup reduction for RMS calculation

    **Algorithm**:
    ```
    1. Each thread loads one element and computes x²
    2. Parallel reduction to compute sum(x²) in workgroup shared memory
    3. Compute RMS = sqrt(sum / n + ε)
    4. Each thread normalizes: y = (x / RMS) * scale
    ```

    **Workgroup strategy**:
    - Workgroup size: 256 threads (typical for hidden dims like 768, 1024, 2048)
    - For larger dims: Multiple workgroups, each handling 256 elements
    - Reduction uses shared memory for efficiency

    @param config RMSNorm configuration
-/
def rmsNormKernel (config : Config) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let lid ← ShaderM.localId
  let idx := Exp.vec3X gid
  let localIdx := Exp.vec3X lid

  -- Declare shared memory for workgroup reduction
  ShaderM.sharedNamed "shared_sum" (.array (.scalar .f32) workgroupSize)

  -- Declare buffers
  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) config.dim)
  let _scale ← ShaderM.declareInputBuffer "scale" (.array (.scalar .f32) config.dim)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) config.dim)

  -- Each thread accumulates partial sum of x² over strided elements
  ShaderM.varNamed "partial_sum" (.scalar .f32) (Exp.litF32 0.0)
  let partialSum : Exp (.scalar .f32) := Exp.var "partial_sum"

  ShaderM.loop localIdx (Exp.litU32 config.dim) (Exp.litU32 workgroupSize) fun loopIdx => do
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.dim) "input" loopIdx
    ShaderM.assign "partial_sum" (Exp.add partialSum (Exp.mul val val))

  -- Write partial sum to shared memory
  ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_sum" localIdx partialSum
  ShaderM.barrier

  -- Tree reduction in shared memory
  let mut stride := workgroupSize / 2
  while stride > 0 do
    ShaderM.if_ (Exp.lt localIdx (Exp.litU32 stride)) (do
      let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_sum" localIdx
      let b ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_sum" (Exp.add localIdx (Exp.litU32 stride))
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_sum" localIdx (Exp.add a b)
    ) (pure ())
    ShaderM.barrier
    stride := stride / 2

  -- Thread 0 has the total sum; all threads read it from shared[0]
  let totalSum ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_sum" (Exp.litU32 0)

  -- Compute RMS: sqrt(mean(x²) + eps)
  let mean := Exp.div totalSum (Exp.litF32 config.dim.toFloat)
  let rms := Exp.sqrt (Exp.add mean (Exp.litF32 config.eps))

  -- Each thread normalizes its element(s)
  let inBounds := Exp.lt idx (Exp.litU32 config.dim)
  let inputVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.dim) "input" idx
  let scaleVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.dim) "scale" idx
  let normalized := Exp.div inputVal rms
  let result := Exp.mul normalized scaleVal

  let finalResult := Exp.select inBounds result (Exp.litF32 0.0)
  ShaderM.writeBuffer (ty := .scalar .f32) "output" idx finalResult

/-! ## Fused Single-Pass Kernel (multi-row) -/

/-- Fused RMSNorm kernel: compute RMS + apply normalization in one dispatch.
    Each workgroup handles one row. Threads use strided loops for both
    RMS accumulation and normalization, supporting dim >> workgroupSize.

    Reduces dispatch count from 2 to 1 per RMSNorm call.
-/
def rmsNormFusedKernel (config : Config) (numRows : Nat) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let rowIdx := Exp.vec3X wid
  let localIdx := Exp.vec3X lid

  let totalElements := numRows * config.dim

  -- Declare shared memory for workgroup reduction
  ShaderM.sharedNamed "shared_sum" (.array (.scalar .f32) workgroupSize)

  -- Declare buffers
  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) totalElements)
  let _scale ← ShaderM.declareInputBuffer "scale" (.array (.scalar .f32) config.dim)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) totalElements)

  let rowBase := Exp.mul rowIdx (Exp.litU32 config.dim)

  -- Step 1: Accumulate partial sum of x² via strided loop
  ShaderM.varNamed "partial_sum" (.scalar .f32) (Exp.litF32 0.0)
  let partialSum : Exp (.scalar .f32) := Exp.var "partial_sum"

  ShaderM.loop localIdx (Exp.litU32 config.dim) (Exp.litU32 workgroupSize) fun loopIdx => do
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalElements) "input" (Exp.add rowBase loopIdx)
    ShaderM.assign "partial_sum" (Exp.add partialSum (Exp.mul val val))

  -- Write partial sum to shared memory
  ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_sum" localIdx partialSum
  ShaderM.barrier

  -- Step 2: Tree reduction in shared memory
  let mut stride := workgroupSize / 2
  while stride > 0 do
    ShaderM.if_ (Exp.lt localIdx (Exp.litU32 stride)) (do
      let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_sum" localIdx
      let b ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_sum" (Exp.add localIdx (Exp.litU32 stride))
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_sum" localIdx (Exp.add a b)
    ) (pure ())
    ShaderM.barrier
    stride := stride / 2

  -- All threads read the total sum from shared[0]
  let totalSum ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_sum" (Exp.litU32 0)

  -- Compute RMS: sqrt(mean(x²) + eps)
  let mean := Exp.div totalSum (Exp.litF32 config.dim.toFloat)
  let rms := Exp.sqrt (Exp.add mean (Exp.litF32 config.eps))

  -- Step 3: Each thread normalizes its strided elements
  ShaderM.loop localIdx (Exp.litU32 config.dim) (Exp.litU32 workgroupSize) fun dimIdx => do
    let inputVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalElements) "input" (Exp.add rowBase dimIdx)
    let scaleVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.dim) "scale" dimIdx
    let normalized := Exp.div inputVal rms
    let result := Exp.mul normalized scaleVal
    ShaderM.writeBuffer (ty := .scalar .f32) "output" (Exp.add rowBase dimIdx) result

/-! ## Optimized Two-Pass Kernel -/

/-- First pass: Compute RMS value

    This kernel computes the RMS value using parallel reduction.
    Result is a single scalar stored in a 1-element buffer.

    @param config RMSNorm configuration
-/
def rmsComputeKernel (config : Config) (numRows : Nat) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let lid ← ShaderM.localId
  let wid ← ShaderM.workgroupId
  let localIdx := Exp.vec3X lid
  let rowIdx := Exp.vec3X wid  -- Each workgroup handles one row

  -- Declare shared memory for workgroup reduction
  ShaderM.sharedNamed "shared_sum" (.array (.scalar .f32) workgroupSize)

  -- Declare buffers
  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) (numRows * config.dim))
  let _rmsOut ← ShaderM.declareOutputBuffer "rms_output" (.array (.scalar .f32) numRows)

  -- Base offset for this row
  let rowBase := Exp.mul rowIdx (Exp.litU32 config.dim)

  -- Each thread accumulates partial sum of x² over strided elements
  ShaderM.varNamed "partial_sum" (.scalar .f32) (Exp.litF32 0.0)
  let partialSum : Exp (.scalar .f32) := Exp.var "partial_sum"

  ShaderM.loop localIdx (Exp.litU32 config.dim) (Exp.litU32 workgroupSize) fun loopIdx => do
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := numRows * config.dim) "input" (Exp.add rowBase loopIdx)
    ShaderM.assign "partial_sum" (Exp.add partialSum (Exp.mul val val))

  -- Write partial sum to shared memory
  ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_sum" localIdx partialSum
  ShaderM.barrier

  -- Tree reduction in shared memory
  let mut stride := workgroupSize / 2
  while stride > 0 do
    ShaderM.if_ (Exp.lt localIdx (Exp.litU32 stride)) (do
      let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_sum" localIdx
      let b ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_sum" (Exp.add localIdx (Exp.litU32 stride))
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_sum" localIdx (Exp.add a b)
    ) (pure ())
    ShaderM.barrier
    stride := stride / 2

  -- Thread 0 of each workgroup writes RMS for its row
  let totalSum ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_sum" (Exp.litU32 0)
  let mean := Exp.div totalSum (Exp.litF32 config.dim.toFloat)
  let rms := Exp.sqrt (Exp.add mean (Exp.litF32 config.eps))

  ShaderM.if_ (Exp.eq localIdx (Exp.litU32 0)) (do
    ShaderM.writeBuffer (ty := .scalar .f32) "rms_output" rowIdx rms
  ) (pure ())

/-- Second pass: Apply normalization using precomputed RMS

    This kernel normalizes the input using the RMS computed in the first pass.

    @param config RMSNorm configuration
-/
def rmsApplyKernel (config : Config) (numRows : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let totalElements := numRows * config.dim
  -- Bounds check
  let inBounds := Exp.lt idx (Exp.litU32 totalElements)

  -- Declare buffers
  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) totalElements)
  let _rmsIn ← ShaderM.declareInputBuffer "rms_input" (.array (.scalar .f32) numRows)
  let _scale ← ShaderM.declareInputBuffer "scale" (.array (.scalar .f32) config.dim)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) totalElements)

  -- Determine which row this element belongs to
  let rowIdx := Exp.div idx (Exp.litU32 config.dim)
  let dimIdx := Exp.mod idx (Exp.litU32 config.dim)

  -- Read precomputed RMS for this row
  let rms ← ShaderM.readBuffer (ty := .scalar .f32) (n := numRows) "rms_input" rowIdx

  -- Read input and scale (scale is shared across rows)
  let inputVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalElements) "input" idx
  let scaleVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.dim) "scale" dimIdx

  -- Normalize: y = (x / RMS) * scale
  let normalized := Exp.div inputVal rms
  let result := Exp.mul normalized scaleVal

  -- Write output
  let finalResult := Exp.select inBounds result (Exp.litF32 0.0)
  ShaderM.writeBuffer (ty := .scalar .f32) "output" idx finalResult

/-! ## High-Level API -/

/-- RMSNorm layer structure -/
structure RMSNorm where
  config : Config
  scale : Buffer  -- Learned scale parameters (γ)
  prepared : IO.Ref (Option Hesper.WGSL.Execute.PreparedDispatch)  -- Graph capture cache

/-- Create RMSNorm layer from GGUF tensors

    @param device WebGPU device
    @param config Layer configuration
    @param scaleData Raw scale data from GGUF (Float32 or FP16)
-/
def create (device : Device) (config : Config) (scaleData : ByteArray) : IO RMSNorm := do
  IO.println s!"[RMSNorm] Creating layer: dim={config.dim}, eps={config.eps}"

  -- Create GPU buffer for scale parameters
  let scaleBuf ← createBuffer device {
    size := scaleData.size.toUSize
    usage := [.storage, .copyDst]
    mappedAtCreation := false
  }

  -- Upload scale data
  writeBuffer device scaleBuf 0 scaleData

  let prepared ← IO.mkRef none
  IO.println "[RMSNorm] ✓ Layer created on GPU"
  pure { config, scale := scaleBuf, prepared }

/-- Execute forward pass (single-kernel version)

    @param device WebGPU device
    @param layer RMSNorm layer
    @param inputBuf GPU buffer containing input (Float32)
    @param outputBuf GPU buffer for output (Float32)
-/
def forward (device : Device) (layer : RMSNorm)
            (inputBuf outputBuf : Buffer) (numRows : Nat := 1) (workgroupSize : Nat := 256)
            (preAllocRmsBuf : Option Buffer := none) : IO Unit := do
  -- Fast path: replay prepared dispatch (skips ALL Lean processing)
  if numRows == 1 then
    if let some p ← layer.prepared.get then
      preparedHitsRef.modify (· + 1)
      Hesper.WGSL.Execute.replayPreparedDispatch device p numRows 1 1
      return

  preparedMissesRef.modify (· + 1)
  logVerbose s!"[RMSNorm] Executing forward pass ({numRows} rows × {layer.config.dim} dim)..."

  -- Fused single-pass: compute RMS + apply normalization in one dispatch
  -- 1 workgroup per row, each workgroup uses shared memory reduction
  let shader := rmsNormFusedKernel layer.config numRows workgroupSize
  let namedBuffers := [
    ("input", inputBuf),
    ("scale", layer.scale),
    ("output", outputBuf)
  ]
  let execConfig : Hesper.WGSL.Execute.ExecutionConfig := {
    workgroupSize := { x := workgroupSize, y := 1, z := 1 }
    numWorkgroups := (numRows, 1, 1)
  }
  let cacheKey : UInt64 := hash ("rms", layer.config.dim, numRows, workgroupSize)
  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig (some cacheKey) (some layer.prepared)

  logVerbose "[RMSNorm] ✓ Forward pass complete"

/-- Execute forward pass (two-pass optimized version)

    This version uses separate kernels for RMS computation and normalization,
    enabling better workgroup reduction optimization.

    @param device WebGPU device
    @param layer RMSNorm layer
    @param inputBuf GPU buffer containing input (Float32)
    @param outputBuf GPU buffer for output (Float32)
    @param rmsTempBuf Temporary 1-element buffer for RMS value
-/
def forwardTwoPass (device : Device) (layer : RMSNorm)
                   (inputBuf outputBuf rmsTempBuf : Buffer) (numRows : Nat := 1) (workgroupSize : Nat := 256) : IO Unit := do
  logVerbose "[RMSNorm] Executing two-pass forward pass..."

  -- Pass 1: Compute RMS (one workgroup per row)
  let shader1 := rmsComputeKernel layer.config numRows workgroupSize
  let namedBuffers1 := [
    ("input", inputBuf),
    ("rms_output", rmsTempBuf)
  ]
  let execConfig1 : Hesper.WGSL.Execute.ExecutionConfig := {
    workgroupSize := { x := workgroupSize, y := 1, z := 1 }
    numWorkgroups := (numRows, 1, 1)
  }

  Hesper.WGSL.Execute.executeShaderNamed device shader1 namedBuffers1 execConfig1

  -- Pass 2: Apply normalization (one thread per element across all rows)
  let shader2 := rmsApplyKernel layer.config numRows
  let namedBuffers2 := [
    ("input", inputBuf),
    ("rms_input", rmsTempBuf),
    ("scale", layer.scale),
    ("output", outputBuf)
  ]
  let totalElements := numRows * layer.config.dim
  let execConfig2 := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D
    totalElements
    workgroupSize

  Hesper.WGSL.Execute.executeShaderNamed device shader2 namedBuffers2 execConfig2
  logVerbose "[RMSNorm] ✓ Two-pass forward complete"

/-! ## Integration with GGUF Reader -/

/-- Create RMSNorm layer directly from GGUF file

    This extracts the scale (weight) tensor for a specific layer.

    Example tensor names in GGUF:
    - `blk.0.attn_norm.weight` - Pre-attention RMSNorm
    - `blk.0.ffn_norm.weight` - Pre-FFN RMSNorm
    - `output_norm.weight` - Final output RMSNorm

    @param device WebGPU device
    @param gguf Loaded GGUF file
    @param tensorName Name of the scale tensor in GGUF
    @param config Layer configuration
-/
def fromGGUF (device : Device) (gguf : α) (tensorName : String) (config : Config) : IO RMSNorm := do
  -- This is a placeholder - actual implementation would:
  -- 1. Find tensor by name: gguf.findTensor tensorName
  -- 2. Extract raw data: gguf.getTensorRaw tensorInfo
  -- 3. Convert FP16 → Float32 if needed
  -- 4. Call create() with extracted data
  IO.println s!"[RMSNorm] Loading from GGUF tensor: {tensorName}"
  throw $ IO.userError "fromGGUF not yet implemented - use create() directly"

end Hesper.Layers.RMSNorm
