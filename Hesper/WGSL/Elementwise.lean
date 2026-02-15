import Hesper.WGSL.Monad
import Hesper.WGSL.Execute
import Hesper.WGSL.Exp
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer

/-!
# Element-wise Operations

Implements element-wise tensor operations for neural network layers.

## Operations

### 1. Add (Residual Connections)
```
c[i] = a[i] + b[i]
```
Used in residual connections throughout transformer blocks.

### 2. Multiply
```
c[i] = a[i] × b[i]
```
Used for gating mechanisms in FFN.

### 3. ReLU² (Squared ReLU) Activation
```
ReLU²(x) = max(0, x)²
```
Used in BitNet b1.58 FFN layers (LLM_FFN_RELU_SQR).

### 4. GELU Activation
```
GELU(x) ≈ 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))
```
Gaussian Error Linear Unit - used in some transformer variants.

### 5. Scale
```
c[i] = a[i] × scalar
```
Used for normalization and attention scaling.

## Performance

Element-wise ops are typically **memory-bound**:
```
Compute: 1 FLOP per element
Memory: 2 reads + 1 write = 3 × 4 bytes = 12 bytes
Arithmetic intensity: 1 / 12 ≈ 0.083 FLOP/byte
```

**On A100**:
- Peak FLOPS: 19.5 TFLOPS
- Memory bandwidth: 2039 GB/s
- Bandwidth limit: 2039 / 12 ≈ 170 GFLOPS
- **Utilization: 170 / 19500 ≈ 0.9%** (memory-bound!)

**Optimization strategies**:
1. **Kernel fusion**: Combine multiple element-wise ops
2. **Vectorization**: Process 4 elements per thread (vec4)
3. **In-place**: Reuse input buffer as output when possible

## References
- ReLU²: Used in BitNet b1.58 (LLM_FFN_RELU_SQR in llama.cpp)
- GELU: "Gaussian Error Linear Units" (Hendrycks & Gimpel, 2016)
-/

namespace Hesper.WGSL.Elementwise

open Hesper.WGSL
open Hesper.WGSL.Monad
open Hesper.WebGPU

/-! ## Configuration -/

/-- Element-wise operation configuration -/
structure Config where
  numElements : Nat
  deriving Repr

/-! ## Addition (Residual Connections) -/

/-- Element-wise addition: c = a + b

    Used for residual connections in transformer blocks.

    @param config Operation configuration
-/
def addKernel (config : Config) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let inBounds := Exp.lt idx (Exp.litU32 config.numElements)

  let _a ← ShaderM.declareInputBuffer "a" (.array (.scalar .f32) config.numElements)
  let _b ← ShaderM.declareInputBuffer "b" (.array (.scalar .f32) config.numElements)
  let _c ← ShaderM.declareOutputBuffer "c" (.array (.scalar .f32) config.numElements)

  let aVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.numElements) "a" idx
  let bVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.numElements) "b" idx

  let result := Exp.add aVal bVal
  let finalResult := Exp.select inBounds result (Exp.litF32 0.0)

  ShaderM.writeBuffer (ty := .scalar .f32) "c" idx finalResult

/-- Execute element-wise addition

    @param device WebGPU device
    @param aBuf Input buffer A
    @param bBuf Input buffer B
    @param cBuf Output buffer C
    @param config Configuration
-/
def executeAdd (device : Device) (aBuf bBuf cBuf : Buffer) (config : Config) : IO Unit := do
  let shader := addKernel config
  let namedBuffers := [("a", aBuf), ("b", bBuf), ("c", cBuf)]
  let execConfig := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D config.numElements 256
  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig

/-! ## Multiplication (Gating) -/

/-- Element-wise multiplication: c = a × b

    Used for gating in FFN layers.

    @param config Operation configuration
-/
def mulKernel (config : Config) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let inBounds := Exp.lt idx (Exp.litU32 config.numElements)

  let _a ← ShaderM.declareInputBuffer "a" (.array (.scalar .f32) config.numElements)
  let _b ← ShaderM.declareInputBuffer "b" (.array (.scalar .f32) config.numElements)
  let _c ← ShaderM.declareOutputBuffer "c" (.array (.scalar .f32) config.numElements)

  let aVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.numElements) "a" idx
  let bVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.numElements) "b" idx

  let result := Exp.mul aVal bVal
  let finalResult := Exp.select inBounds result (Exp.litF32 0.0)

  ShaderM.writeBuffer (ty := .scalar .f32) "c" idx finalResult

/-- Execute element-wise multiplication

    @param device WebGPU device
    @param aBuf Input buffer A
    @param bBuf Input buffer B
    @param cBuf Output buffer C
    @param config Configuration
-/
def executeMul (device : Device) (aBuf bBuf cBuf : Buffer) (config : Config) : IO Unit := do
  let shader := mulKernel config
  let namedBuffers := [("a", aBuf), ("b", bBuf), ("c", cBuf)]
  let execConfig := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D config.numElements 256
  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig

/-! ## ReLU² Activation -/

/-- ReLU² (Squared ReLU) activation: y = max(0, x)²

    Used in BitNet b1.58 FFN layers (LLM_FFN_RELU_SQR in llama.cpp).

    @param config Operation configuration
-/
def reluSqrKernel (config : Config) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let inBounds := Exp.lt idx (Exp.litU32 config.numElements)

  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) config.numElements)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) config.numElements)

  let x ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.numElements) "input" idx

  -- ReLU²(x) = max(0, x)²
  let relu := Exp.max x (Exp.litF32 0.0)
  let result := Exp.mul relu relu

  let finalResult := Exp.select inBounds result (Exp.litF32 0.0)
  ShaderM.writeBuffer (ty := .scalar .f32) "output" idx finalResult

/-- Execute ReLU² activation

    @param device WebGPU device
    @param inputBuf Input buffer
    @param outputBuf Output buffer
    @param config Configuration
-/
def executeReluSqr (device : Device) (inputBuf outputBuf : Buffer) (config : Config) : IO Unit := do
  let shader := reluSqrKernel config
  let namedBuffers := [("input", inputBuf), ("output", outputBuf)]
  let execConfig := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D config.numElements 256
  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig

/-! ## GELU Activation -/

/-- GELU activation (tanh approximation)

    GELU(x) ≈ 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))

    Used in some transformer variants (BERT, GPT-2).

    @param config Operation configuration
-/
def geluKernel (config : Config) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let inBounds := Exp.lt idx (Exp.litU32 config.numElements)

  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) config.numElements)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) config.numElements)

  let x ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.numElements) "input" idx

  -- Constants
  let sqrtTwoDivPi := Exp.litF32 0.7978845608  -- sqrt(2/π)
  let coeff := Exp.litF32 0.044715

  -- Compute: x + 0.044715 × x³
  let xCubed := Exp.mul (Exp.mul x x) x
  let cubedTerm := Exp.mul coeff xCubed
  let inner := Exp.add x cubedTerm

  -- Multiply by sqrt(2/π)
  let scaled := Exp.mul sqrtTwoDivPi inner

  -- Apply tanh
  let tanhVal := Exp.tanh scaled

  -- 1 + tanh(...)
  let onePlusTanh := Exp.add (Exp.litF32 1.0) tanhVal

  -- 0.5 × x × (1 + tanh(...))
  let halfX := Exp.mul (Exp.litF32 0.5) x
  let result := Exp.mul halfX onePlusTanh

  let finalResult := Exp.select inBounds result (Exp.litF32 0.0)
  ShaderM.writeBuffer (ty := .scalar .f32) "output" idx finalResult

/-- Execute GELU activation

    @param device WebGPU device
    @param inputBuf Input buffer
    @param outputBuf Output buffer
    @param config Configuration
-/
def executeGELU (device : Device) (inputBuf outputBuf : Buffer) (config : Config) : IO Unit := do
  let shader := geluKernel config
  let namedBuffers := [("input", inputBuf), ("output", outputBuf)]
  let execConfig := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D config.numElements 256
  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig

/-! ## Scalar Operations -/

/-- Scalar multiplication: c = a × scalar

    Used for scaling operations.

    @param config Operation configuration
    @param scalar Scalar value to multiply by
-/
def scaleKernel (config : Config) (scalar : Float) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let inBounds := Exp.lt idx (Exp.litU32 config.numElements)

  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) config.numElements)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) config.numElements)

  let x ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.numElements) "input" idx

  let result := Exp.mul x (Exp.litF32 scalar)
  let finalResult := Exp.select inBounds result (Exp.litF32 0.0)

  ShaderM.writeBuffer (ty := .scalar .f32) "output" idx finalResult

/-- Execute scalar multiplication

    @param device WebGPU device
    @param inputBuf Input buffer
    @param outputBuf Output buffer
    @param config Configuration
    @param scalar Scalar multiplier
-/
def executeScale (device : Device) (inputBuf outputBuf : Buffer)
                 (config : Config) (scalar : Float) : IO Unit := do
  let shader := scaleKernel config scalar
  let namedBuffers := [("input", inputBuf), ("output", outputBuf)]
  let execConfig := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D config.numElements 256
  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig

/-! ## Fused Operations (Optimization) -/

/-- Fused: ReLU² + Multiply (for BitNet FFN gating)

    Computes: c = ReLU²(a) × b = max(0, a)² × b

    This is the gating mechanism in BitNet b1.58 FFN:
    ```
    gate = ReLU²(W_gate @ x)
    up = W_up @ x
    result = gate × up
    ```

    Fusing ReLU² + multiply saves one global memory roundtrip.

    @param config Operation configuration
-/
def reluSqrMulKernel (config : Config) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let inBounds := Exp.lt idx (Exp.litU32 config.numElements)

  let _a ← ShaderM.declareInputBuffer "a" (.array (.scalar .f32) config.numElements)
  let _b ← ShaderM.declareInputBuffer "b" (.array (.scalar .f32) config.numElements)
  let _c ← ShaderM.declareOutputBuffer "c" (.array (.scalar .f32) config.numElements)

  let aVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.numElements) "a" idx
  let bVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := config.numElements) "b" idx

  -- Compute ReLU²(a) = max(0, a)²
  let relu := Exp.max aVal (Exp.litF32 0.0)
  let reluSqr := Exp.mul relu relu

  -- Multiply by b
  let result := Exp.mul reluSqr bVal

  let finalResult := Exp.select inBounds result (Exp.litF32 0.0)
  ShaderM.writeBuffer (ty := .scalar .f32) "c" idx finalResult

/-- Execute fused ReLU² + multiply

    @param device WebGPU device
    @param aBuf Input buffer A (will be ReLU²'d)
    @param bBuf Input buffer B (multiplier)
    @param cBuf Output buffer C
    @param config Configuration
-/
def executeReluSqrMul (device : Device) (aBuf bBuf cBuf : Buffer) (config : Config) : IO Unit := do
  let shader := reluSqrMulKernel config
  let namedBuffers := [("a", aBuf), ("b", bBuf), ("c", cBuf)]
  let execConfig := Hesper.WGSL.Execute.ExecutionConfig.dispatch1D config.numElements 256
  Hesper.WGSL.Execute.executeShaderNamed device shader namedBuffers execConfig

end Hesper.WGSL.Elementwise
