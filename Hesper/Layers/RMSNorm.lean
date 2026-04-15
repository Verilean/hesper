import Hesper.WGSL.Monad
import Hesper.WGSL.Exp
import Hesper.Backend
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
open Hesper
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

/-! ## Fused post-norm (Gemma 4 style): RMSNorm(layer_out) + residual -/

/-- Computes `output = RMSNorm(layer_out) * scale + residual` in one kernel.

    This is Gemma 4's post-norm shape: the normalisation is applied to the
    attention / FFN output FIRST, and only then the pre-block residual is
    added back in.  Replaces the two-dispatch pattern
    `RMSNorm.forward → residualAddKernel` used after attention and FFN.

    Dispatch: 1 workgroup × workgroupSize threads.
-/
def rmsNormThenAddKernel (config : Config) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let lid ← ShaderM.localId
  let localIdx := Exp.vec3X lid
  ShaderM.sharedNamed "shared_sum" (.array (.scalar .f32) workgroupSize)
  let _layerOut ← ShaderM.declareInputBuffer "layer_out" (.array (.scalar .f32) config.dim)
  let _residual ← ShaderM.declareInputBuffer "residual" (.array (.scalar .f32) config.dim)
  let _scale    ← ShaderM.declareInputBuffer "scale" (.array (.scalar .f32) config.dim)
  let _output   ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) config.dim)

  -- Pass 1: sum of squares of layer_out.
  ShaderM.varNamed "partial_sum" (.scalar .f32) (Exp.litF32 0.0)
  let partialSum : Exp (.scalar .f32) := Exp.var "partial_sum"
  ShaderM.loop localIdx (Exp.litU32 config.dim) (Exp.litU32 workgroupSize) fun i => do
    let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.dim) "layer_out" i
    ShaderM.assign "partial_sum" (Exp.add partialSum (Exp.mul v v))
  ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_sum" localIdx partialSum
  ShaderM.barrier
  let mut stride := workgroupSize / 2
  while stride > 0 do
    ShaderM.if_ (Exp.lt localIdx (Exp.litU32 stride)) (do
      let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_sum" localIdx
      let b ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_sum" (Exp.add localIdx (Exp.litU32 stride))
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_sum" localIdx (Exp.add a b)
    ) (pure ())
    ShaderM.barrier
    stride := stride / 2
  let totalSum ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_sum" (Exp.litU32 0)
  let mean := Exp.div totalSum (Exp.litF32 config.dim.toFloat)
  let rms := Exp.sqrt (Exp.add mean (Exp.litF32 config.eps))

  -- Pass 2: normalise × scale, then add residual.
  ShaderM.loop localIdx (Exp.litU32 config.dim) (Exp.litU32 workgroupSize) fun i => do
    let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.dim) "layer_out" i
    let s ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.dim) "scale" i
    let r ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.dim) "residual" i
    let y := Exp.add (Exp.mul (Exp.div v rms) s) r
    ShaderM.writeBuffer (ty := .scalar .f32) "output" i y

/-! ## Fused residual-add + RMSNorm -/

/-- Compute `residualOut = a + b` and `output = RMSNorm(residualOut) * scale`
    in one dispatch.  Replaces the two-kernel pattern `residualAddKernel
    + RMSNorm.forward` used after attention/FFN.

    Dispatch: 1 workgroup × `workgroupSize` threads per row.  Each row
    independently loads `a[row]+b[row]`, reduces to RMS, and writes both
    `residualOut[row]` (for the next residual chain) and `output[row]`
    (the normalised result).
-/
def residualAddRmsNormKernel (config : Config) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let lid ← ShaderM.localId
  let localIdx := Exp.vec3X lid
  -- One workgroup per row.  Single-row case (hidden state = one flat row).
  ShaderM.sharedNamed "shared_sum" (.array (.scalar .f32) workgroupSize)
  let _a     ← ShaderM.declareInputBuffer "a" (.array (.scalar .f32) config.dim)
  let _b     ← ShaderM.declareInputBuffer "b" (.array (.scalar .f32) config.dim)
  let _scale ← ShaderM.declareInputBuffer "scale" (.array (.scalar .f32) config.dim)
  let _resid ← ShaderM.declareOutputBuffer "residualOut" (.array (.scalar .f32) config.dim)
  let _out   ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) config.dim)

  -- Pass 1: compute sum-of-squares over (a + b) via strided loop.  We
  -- also write the sum to residualOut as we go so pass 2 doesn't re-add.
  ShaderM.varNamed "partial_sum" (.scalar .f32) (Exp.litF32 0.0)
  let partialSum : Exp (.scalar .f32) := Exp.var "partial_sum"
  ShaderM.loop localIdx (Exp.litU32 config.dim) (Exp.litU32 workgroupSize) fun i => do
    let va ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.dim) "a" i
    let vb ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.dim) "b" i
    let s := Exp.add va vb
    ShaderM.writeBuffer (ty := .scalar .f32) "residualOut" i s
    ShaderM.assign "partial_sum" (Exp.add partialSum (Exp.mul s s))

  ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_sum" localIdx partialSum
  ShaderM.barrier
  let mut stride := workgroupSize / 2
  while stride > 0 do
    ShaderM.if_ (Exp.lt localIdx (Exp.litU32 stride)) (do
      let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_sum" localIdx
      let b ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_sum" (Exp.add localIdx (Exp.litU32 stride))
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_sum" localIdx (Exp.add a b)
    ) (pure ())
    ShaderM.barrier
    stride := stride / 2

  let totalSum ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_sum" (Exp.litU32 0)
  let mean := Exp.div totalSum (Exp.litF32 config.dim.toFloat)
  let rms := Exp.sqrt (Exp.add mean (Exp.litF32 config.eps))

  -- Pass 2: read residualOut (which we just wrote), normalise, apply scale.
  ShaderM.loop localIdx (Exp.litU32 config.dim) (Exp.litU32 workgroupSize) fun i => do
    let s ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.dim) "residualOut" i
    let scale ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.dim) "scale" i
    let y := Exp.mul (Exp.div s rms) scale
    ShaderM.writeBuffer (ty := .scalar .f32) "output" i y

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

/-! ## Fused RMSNorm + Q8_1 quantize

Eliminates the VRAM round-trip between RMSNorm output and the matmul's
Q8_1 quantize phase by combining both into a single dispatch.

Algorithm:
  Phase 1 — Single-WG cooperative RMSNorm reduction
    - 256 threads stride over D=`config.dim` input elements (each lane
      processes `D / 256` elements via the strided loop).
    - Tree reduction in shared memory yields the sum-of-squares.
    - All threads compute `invRms = rsqrt(sumSq/D + eps)` from
      `scratch[0]`.
  Phase 2 — Per-block Q8_1 quantize (10 strided passes for D=2560)
    Per pass `p`, lane `tid` handles input element `tid + p*256`.
    The 256 lanes split into 8 warps of 32 lanes each.  Warp `w`
    owns the 32-element block at `8*p + w` of the output.

    Inside each warp (lanes l=0..31, owning one Q8_1 block):
      - x_normed[l] = inputBuf[elemIdx] * scale[elemIdx] * invRms
      - amax = subgroupMax(|x_normed[l]|)
      - d = amax / 127
      - q[l] = round(x_normed[l] / d)  (clamped to int8)
      - shared_q[warpId * 32 + l] = q[l]; barrier
      - lane l divisible by 4 packs 4 quants into one u32 → output

    Lane 0 of each warp writes the d|s header (s=0; subsequent
    Q4_K dp4a path doesn't use s).

This stays within Hesper's hand-written-kernel inventory but is wired
through the IR as `Prim.rmsNormQ8_1Quantize`, so callers see it as a
proper compiler op.  The Stage 3 follow-up will replace the body
with auto-generated reduce + double-reduce-epilogue lowering. -/
def fusedRMSNormQ8_1Kernel (config : Config) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let lid ← ShaderM.localId
  let localIdx := Exp.vec3X lid
  let D := config.dim
  let nBlocks := D / 32
  let outU32Size := nBlocks * 9

  ShaderM.sharedNamed "scratch_norm" (.array (.scalar .f32) workgroupSize)
  -- One byte slot per lane; reused across the 10 strided quantize passes.
  ShaderM.sharedNamed "shared_q" (.array (.scalar .u32) workgroupSize)

  let _input  ← ShaderM.declareReadOnlyBuffer "input"  (.array (.scalar .f32) D)
  let _scale  ← ShaderM.declareReadOnlyBuffer "scale"  (.array (.scalar .f32) D)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .u32) outU32Size)

  -- ── Phase 1: cooperative RMSNorm reduction over the input ──
  ShaderM.varNamed "accum" (.scalar .f32) (Exp.litF32 0.0)
  let accumE : Exp (.scalar .f32) := Exp.var "accum"
  ShaderM.loop localIdx (Exp.litU32 D) (Exp.litU32 workgroupSize) fun loopIdx => do
    let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := D) "input" loopIdx
    ShaderM.assign "accum" (Exp.add accumE (Exp.mul v v))
  ShaderM.writeWorkgroup (ty := .scalar .f32) "scratch_norm" localIdx accumE
  ShaderM.barrier
  -- Tree reduction.
  let mut stride := workgroupSize / 2
  while stride > 0 do
    ShaderM.if_ (Exp.lt localIdx (Exp.litU32 stride)) (do
      let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "scratch_norm" localIdx
      let b ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "scratch_norm"
                (Exp.add localIdx (Exp.litU32 stride))
      ShaderM.writeWorkgroup (ty := .scalar .f32) "scratch_norm" localIdx (Exp.add a b)
    ) (pure ())
    ShaderM.barrier
    stride := stride / 2
  -- All lanes read the total sum-of-squares from scratch[0] and compute invRms.
  let sumSq ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "scratch_norm" (Exp.litU32 0)
  ShaderM.varNamed "invRms" (.scalar .f32)
    (Exp.inverseSqrt (Exp.add (Exp.div sumSq (Exp.litF32 D.toFloat)) (Exp.litF32 config.eps)))
  let invRms : Exp (.scalar .f32) := Exp.var "invRms"

  -- ── Phase 2: per-32-element-block Q8_1 quantize, strided over D ──
  -- numPasses = D / wgSize.  Each pass: 256 lanes write their normed
  -- value, then 8 warps each own one Q8_1 block.
  let numPasses := D / workgroupSize
  let warpId   := Exp.div localIdx (Exp.litU32 32)             -- 0..7
  let laneInW  := Exp.sub localIdx (Exp.mul warpId (Exp.litU32 32))  -- 0..31
  let mut p : Nat := 0
  while p < numPasses do
    let elemIdx := Exp.add (Exp.litU32 (p * workgroupSize)) localIdx
    let blockIdx := Exp.add (Exp.litU32 (p * (workgroupSize / 32))) warpId
    let x  ← ShaderM.readBuffer (ty := .scalar .f32) (n := D) "input" elemIdx
    let s  ← ShaderM.readBuffer (ty := .scalar .f32) (n := D) "scale" elemIdx
    let xN := Exp.mul (Exp.mul x invRms) s
    -- Materialise the normalised value before reductions branch on it.
    let xnName ← ShaderM.var (.scalar .f32) xN
    let xNRef : Exp (.scalar .f32) := Exp.var xnName
    -- Per-warp max-abs reduction (subgroupMax operates over the warp).
    let absXN := Exp.select (Exp.lt xNRef (Exp.litF32 0.0))
                            (Exp.sub (Exp.litF32 0.0) xNRef) xNRef
    let amaxName ← ShaderM.var (.scalar .f32) (Exp.subgroupMax absXN)
    let amax : Exp (.scalar .f32) := Exp.var amaxName
    -- Q8_1 scale d = amax / 127.
    let dName ← ShaderM.var (.scalar .f32) (Exp.div amax (Exp.litF32 127.0))
    let d : Exp (.scalar .f32) := Exp.var dName
    -- Quantize: q = round(xN / d), guarded against d==0.
    let qF32 := Exp.select (Exp.eq d (Exp.litF32 0.0))
                           (Exp.litF32 0.0) (Exp.div xNRef d)
    let qByte := Exp.bitAnd (Exp.roundToI32 qF32) (Exp.litU32 0xFF)
    -- Stage byte in shared mem (each lane writes its slot).
    ShaderM.writeWorkgroup (ty := .scalar .u32) "shared_q" localIdx qByte
    ShaderM.barrier
    -- Write the d|s header (s=0; downstream Q4_K dp4a kernel ignores s
    -- — see fusedQ4KMLinearDP4AKernel which reconstructs sums per
    -- block from the int8 quants).  Lane 0 of each warp owns its block.
    let hdrOff := Exp.mul blockIdx (Exp.litU32 9)
    ShaderM.if_ (Exp.eq laneInW (Exp.litU32 0)) (do
      let dBits : Exp (.scalar .u32) := Exp.bitcast d
      ShaderM.writeBuffer (ty := .scalar .u32) "output" hdrOff dBits
    ) (pure ())
    -- Pack 4 bytes per output u32.  Lanes with laneInW % 4 == 0 (that
    -- is, lanes 0,4,8,…,28 within each warp) write one packed u32.
    let laneQuarter := Exp.div laneInW (Exp.litU32 4)
    let isQuarterLane := Exp.eq (Exp.sub laneInW (Exp.mul laneQuarter (Exp.litU32 4))) (Exp.litU32 0)
    ShaderM.if_ isQuarterLane (do
      let warpBase := Exp.mul warpId (Exp.litU32 32)
      let base := Exp.add warpBase (Exp.mul laneQuarter (Exp.litU32 4))
      let b0 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := workgroupSize) "shared_q" base
      let b1 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := workgroupSize) "shared_q" (Exp.add base (Exp.litU32 1))
      let b2 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := workgroupSize) "shared_q" (Exp.add base (Exp.litU32 2))
      let b3 ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := workgroupSize) "shared_q" (Exp.add base (Exp.litU32 3))
      let packed := Exp.bitOr (Exp.bitOr b0 (Exp.shiftLeft b1 (Exp.litU32 8)))
                              (Exp.bitOr (Exp.shiftLeft b2 (Exp.litU32 16)) (Exp.shiftLeft b3 (Exp.litU32 24)))
      let outIdx := Exp.add hdrOff (Exp.add (Exp.litU32 1) laneQuarter)
      ShaderM.writeBuffer (ty := .scalar .u32) "output" outIdx packed
    ) (pure ())
    -- Barrier before the next pass so shared_q is free to overwrite.
    ShaderM.barrier
    p := p + 1

/-! ## High-Level API -/

/-- RMSNorm layer structure -/
structure RMSNorm (BufT : Type) (CacheT : Type := Unit) where
  config : Config
  scale : BufT
  prepared : IO.Ref (Option CacheT)

/-- Create RMSNorm layer from GGUF tensors

    @param device WebGPU device
    @param config Layer configuration
    @param scaleData Raw scale data from GGUF (Float32 or FP16)
-/
def create [GPUBackend β] (ctx : β) (config : Config) (scaleData : ByteArray)
    : IO (RMSNorm (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)) := do
  logVerbose s!"[RMSNorm] Creating layer: dim={config.dim}, eps={config.eps}"
  let scaleBuf ← GPUBackend.allocBuffer ctx scaleData.size.toUSize
  GPUBackend.writeBuffer ctx scaleBuf scaleData
  let prepared ← GPUBackend.newCacheRef (β := β)
  logVerbose "[RMSNorm] ✓ Layer created on GPU"
  pure { config, scale := scaleBuf, prepared }

/-- Execute forward pass (single-kernel version)

    @param device WebGPU device
    @param layer RMSNorm layer
    @param inputBuf GPU buffer containing input (Float32)
    @param outputBuf GPU buffer for output (Float32)
-/
@[inline]
def forward [GPUBackend β] (ctx : β)
            (layer : RMSNorm (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
            (inputBuf outputBuf : GPUBackend.Buf β) (numRows : Nat := 1) (workgroupSize : Nat := 256)
            (preAllocRmsBuf : Option (GPUBackend.Buf β) := none) : IO Unit := do
  -- Fast path: replay cached dispatch
  if numRows == 1 then
    if let some p ← layer.prepared.get then
      preparedHitsRef.modify (· + 1)
      GPUBackend.replayCached ctx p (numRows, 1, 1)
      return

  preparedMissesRef.modify (· + 1)
  logVerbose s!"[RMSNorm] Executing forward pass ({numRows} rows × {layer.config.dim} dim)..."
  let shader := rmsNormFusedKernel layer.config numRows workgroupSize
  let cacheKey : UInt64 := hash ("rms", layer.config.dim, numRows, workgroupSize)
  GPUBackend.executeWithConfigCached ctx shader
    [("input", inputBuf), ("scale", layer.scale), ("output", outputBuf)]
    { workgroupSize := { x := workgroupSize }, numWorkgroups := (numRows, 1, 1) }
    cacheKey layer.prepared
  logVerbose "[RMSNorm] ✓ Forward pass complete"

/-- Fused post-norm: `output = RMSNorm(layer_out) * scale + residual`.
    Gemma 4's post-attention / post-FFN pattern in a single dispatch.  -/
@[inline]
def forwardNormThenAdd [GPUBackend β] (ctx : β)
    (layer : RMSNorm (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (layerOutBuf residualBuf outputBuf : GPUBackend.Buf β)
    (preparedRef : IO.Ref (Option (GPUBackend.CachedDispatch β)))
    (workgroupSize : Nat := 256) : IO Unit := do
  let shader := rmsNormThenAddKernel layer.config workgroupSize
  let cacheKey : UInt64 := hash ("rms-then-add", layer.config.dim, workgroupSize)
  GPUBackend.executeWithConfigCached ctx shader
    [("layer_out", layerOutBuf), ("residual", residualBuf), ("scale", layer.scale),
     ("output", outputBuf)]
    { workgroupSize := { x := workgroupSize }, numWorkgroups := (1, 1, 1) }
    cacheKey preparedRef

/-- Fused residual-add + RMSNorm forward.  Computes
    `residualOut = a + b` and `output = RMSNorm(residualOut) * scale` in
    a single kernel — replaces the two-dispatch pattern
    `residualAddKernel + RMSNorm.forward`. -/
@[inline]
def forwardResidualAdd [GPUBackend β] (ctx : β)
    (layer : RMSNorm (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (aBuf bBuf residualOutBuf outputBuf : GPUBackend.Buf β)
    (preparedRef : IO.Ref (Option (GPUBackend.CachedDispatch β)))
    (workgroupSize : Nat := 256) : IO Unit := do
  let shader := residualAddRmsNormKernel layer.config workgroupSize
  let cacheKey : UInt64 := hash ("rms-resid", layer.config.dim, workgroupSize)
  GPUBackend.executeWithConfigCached ctx shader
    [("a", aBuf), ("b", bBuf), ("scale", layer.scale),
     ("residualOut", residualOutBuf), ("output", outputBuf)]
    { workgroupSize := { x := workgroupSize }, numWorkgroups := (1, 1, 1) }
    cacheKey preparedRef

/-- Execute forward pass (two-pass optimized version)

    This version uses separate kernels for RMS computation and normalization,
    enabling better workgroup reduction optimization.

    @param device WebGPU device
    @param layer RMSNorm layer
    @param inputBuf GPU buffer containing input (Float32)
    @param outputBuf GPU buffer for output (Float32)
    @param rmsTempBuf Temporary 1-element buffer for RMS value
-/
@[inline]
def forwardTwoPass [GPUBackend β] (ctx : β)
                   (layer : RMSNorm (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
                   (inputBuf outputBuf rmsTempBuf : GPUBackend.Buf β) (numRows : Nat := 1) (workgroupSize : Nat := 256) : IO Unit := do
  logVerbose "[RMSNorm] Executing two-pass forward pass..."
  GPUBackend.execute ctx (rmsComputeKernel layer.config numRows workgroupSize)
    [("input", inputBuf), ("rms_output", rmsTempBuf)]
    { workgroupSize := { x := workgroupSize }, numWorkgroups := (numRows, 1, 1) }
  let totalElements := numRows * layer.config.dim
  GPUBackend.execute ctx (rmsApplyKernel layer.config numRows)
    [("input", inputBuf), ("rms_input", rmsTempBuf), ("scale", layer.scale), ("output", outputBuf)]
    (ExecConfig.dispatch1D totalElements workgroupSize)
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
def fromGGUF [GPUBackend β] (ctx : β) (gguf : α) (tensorName : String) (config : Config)
    : IO (RMSNorm (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)) := do
  -- This is a placeholder - actual implementation would:
  -- 1. Find tensor by name: gguf.findTensor tensorName
  -- 2. Extract raw data: gguf.getTensorRaw tensorInfo
  -- 3. Convert FP16 → Float32 if needed
  -- 4. Call create() with extracted data
  IO.println s!"[RMSNorm] Loading from GGUF tensor: {tensorName}"
  throw $ IO.userError "fromGGUF not yet implemented - use create() directly"

end Hesper.Layers.RMSNorm
