import Hesper.WGSL.Types
import Hesper.WGSL.Exp
import Hesper.WGSL.Monad
import Hesper.WGSL.CodeGen

/-!
# WGSL DSL Helpers for Common Patterns

Provides high-level helpers for common GPU operations:
- Matrix-vector multiplication
- Element-wise operations with activation functions
- Reductions (sum, max, softmax)
- Fused operations (MatMul + Bias + Activation)
-/

namespace Hesper.WGSL.Helpers

open Hesper.WGSL
open Hesper.WGSL.Monad
open Hesper.WGSL.CodeGen

/-! ## Matrix Operations -/

/-- Matrix-vector multiplication with bias and activation (fused Layer operation)

    Computes: output[i] = activation(sum_j(weights[j, i] * input[j]) + bias[i])

    Parameters:
    - inputName: name of input buffer
    - weightsName: name of weights buffer
    - biasName: name of bias buffer
    - outputName: name of output buffer
    - inputSize: number of input features
    - outputSize: number of output features
    - activation: activation function to apply (or identity)
    - outputIdx: expression for output index (usually from global_invocation_id)
-/
def matVecMulBiasActivation
    (inputName weightsName biasName outputName : String)
    (inputSize outputSize : Nat)
    (activation : Exp (.scalar .f32) → Exp (.scalar .f32))
    (outputIdx : Exp (.scalar .u32)) : ShaderM Unit := do

  -- Check bounds
  let boundsCheck := Exp.lt outputIdx (Exp.litU32 outputSize)
  ShaderM.if_ boundsCheck (do
    -- Initialize accumulator
    let sumVar ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)

    -- Matrix-vector dot product
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 inputSize) (Exp.litU32 1) fun j => do
      -- Load input[j]
      let inputVal := Exp.index (Exp.var inputName : Exp (.array (.scalar .f32) inputSize)) j

      -- Compute weight index: weights[j * outputSize + i]
      let outputIdxU32 := outputIdx
      let weightIdx := Exp.add (Exp.mul j (Exp.litU32 outputSize)) outputIdxU32

      -- Load weight
      let weightVal := Exp.index (Exp.var weightsName : Exp (.array (.scalar .f32) (inputSize * outputSize))) weightIdx

      -- Accumulate: sum += input[j] * weight[j, i]
      let prod := Exp.mul inputVal weightVal
      let currentSum := Exp.var sumVar
      let newSum := Exp.add currentSum prod
      ShaderM.assign sumVar newSum

    -- Add bias
    let biasVal := Exp.index (Exp.var biasName : Exp (.array (.scalar .f32) outputSize)) outputIdx
    let sumWithBias := Exp.add (Exp.var sumVar) biasVal

    -- Apply activation
    let result := activation sumWithBias

    -- Store result
    ShaderM.assignIndex outputName outputIdx result
  ) (do
    -- Out of bounds: do nothing
    pure ()
  )

/-- ReLU activation: max(0, x) -/
def reluActivation (x : Exp (.scalar .f32)) : Exp (.scalar .f32) :=
  Exp.max x (Exp.litF32 0.0)

/-- Identity activation: x -/
def identityActivation (x : Exp (.scalar .f32)) : Exp (.scalar .f32) :=
  x

/-- Tanh activation (using WGSL built-in) -/
def tanhActivation (x : Exp (.scalar .f32)) : Exp (.scalar .f32) :=
  Exp.tanh x

/-! ## Softmax Operation -/

/-- Softmax normalization (single-workgroup version for small arrays)

    Computes: output[i] = exp(input[i] - max) / sum(exp(input - max))

    Note: This is a simplified version for small arrays that fit in one workgroup.
    For production use with large arrays, use multi-pass reduction.

    Parameters:
    - dataName: name of buffer (input/output in-place)
    - size: number of elements
-/
def softmaxInPlace (dataName : String) (size : Nat) : ShaderM Unit := do
  -- Find max (for numerical stability)
  let maxVar ← ShaderM.var (.scalar .f32) (Exp.litF32 (-3.402823466e+38))  -- -FLT_MAX

  ShaderM.loop (Exp.litU32 0) (Exp.litU32 size) (Exp.litU32 1) fun i => do
    let val := Exp.index (Exp.var dataName : Exp (.array (.scalar .f32) size)) i
    let currentMax := Exp.var maxVar
    let newMax := Exp.max currentMax val
    ShaderM.assign maxVar newMax

  -- Compute exp(x - max) in-place
  ShaderM.loop (Exp.litU32 0) (Exp.litU32 size) (Exp.litU32 1) fun i => do
    let val := Exp.index (Exp.var dataName : Exp (.array (.scalar .f32) size)) i
    let shifted := Exp.sub val (Exp.var maxVar)
    let expVal := Exp.exp shifted
    ShaderM.assignIndex dataName i expVal

  -- Compute sum of exp values
  let sumVar ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)

  ShaderM.loop (Exp.litU32 0) (Exp.litU32 size) (Exp.litU32 1) fun i => do
    let expVal := Exp.index (Exp.var dataName : Exp (.array (.scalar .f32) size)) i
    let currentSum := Exp.var sumVar
    let newSum := Exp.add currentSum expVal
    ShaderM.assign sumVar newSum

  -- Normalize: divide by sum
  ShaderM.loop (Exp.litU32 0) (Exp.litU32 size) (Exp.litU32 1) fun i => do
    let val := Exp.index (Exp.var dataName : Exp (.array (.scalar .f32) size)) i
    let normalized := Exp.div val (Exp.var sumVar)
    ShaderM.assignIndex dataName i normalized

/-! ## Shader Generation Helpers -/

/-- Generate complete compute shader for matrix-vector multiply + bias + activation

    Generates a WGSL compute shader with proper bindings for:
    - @binding(0): input array [inputSize]
    - @binding(1): weights array [inputSize * outputSize]
    - @binding(2): bias array [outputSize]
    - @binding(3): output array [outputSize]

    The shader fuses MatMul + Bias + Activation into a single kernel.
-/
def generateMatVecBiasActivationShader
    (inputSize outputSize : Nat)
    (activation : Exp (.scalar .f32) → Exp (.scalar .f32))
    (workgroupSizeX : Nat := 256)
    (workgroupSizeY : Nat := 1)
    (workgroupSizeZ : Nat := 1)
    : String :=

  let shaderBody : ShaderM Unit := do
    -- Get global invocation ID
    let gid ← ShaderM.globalId
    let i := Exp.vec3X gid

    -- Perform fused operation
    matVecMulBiasActivation "input" "weights" "bias" "output"
      inputSize outputSize activation i

  -- Extract statements from monad
  let state := ShaderM.exec shaderBody

  -- Create storage buffer declarations
  let buffers : List StorageBuffer := [
    { group := 0, binding := 0, name := "input", elemType := .array (.scalar .f32) inputSize, readWrite := false },
    { group := 0, binding := 1, name := "weights", elemType := .array (.scalar .f32) (inputSize * outputSize), readWrite := false },
    { group := 0, binding := 2, name := "bias", elemType := .array (.scalar .f32) outputSize, readWrite := false },
    { group := 0, binding := 3, name := "output", elemType := .array (.scalar .f32) outputSize, readWrite := true }
  ]

  -- Create main function
  let mainFunc : FunctionDecl := {
    name := "main"
    attributes := [
      "@compute",
      s!"@workgroup_size({workgroupSizeX}, {workgroupSizeY}, {workgroupSizeZ})"
    ]
    params := [
      { name := "global_invocation_id", ty := .vec3 .u32, builtin := some .globalInvocationId }
    ]
    body := state.stmts
  }

  -- Create complete module
  let module : ShaderModule := {
    storageBuffers := buffers
    workgroupVars := []
    functions := [mainFunc]
  }

  module.toWGSL

/-- Generate complete compute shader for softmax (in-place)

    Generates a WGSL compute shader for softmax normalization:
    - @binding(0): data array [size] (read_write)

    Note: Uses single workgroup, suitable for small arrays (size ≤ 1024)
-/
def generateSoftmaxShader
    (size : Nat)
    (workgroupSizeX : Nat := 1)
    (workgroupSizeY : Nat := 1)
    (workgroupSizeZ : Nat := 1)
    : String :=

  let shaderBody : ShaderM Unit := do
    softmaxInPlace "data" size

  let state := ShaderM.exec shaderBody

  let buffers : List StorageBuffer := [
    { group := 0, binding := 0, name := "data", elemType := .array (.scalar .f32) size, readWrite := true }
  ]

  let mainFunc : FunctionDecl := {
    name := "main"
    attributes := [
      "@compute",
      s!"@workgroup_size({workgroupSizeX}, {workgroupSizeY}, {workgroupSizeZ})"
    ]
    params := []
    body := state.stmts
  }

  let module : ShaderModule := {
    storageBuffers := buffers
    functions := [mainFunc]
  }

  module.toWGSL

end Hesper.WGSL.Helpers
