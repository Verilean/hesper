import Hesper.WGSL.Monad
import Hesper.WGSL.Execute
import Hesper.WGSL.Exp
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer

/-!
# TTT GPU Kernels

Simple WGSL compute kernels for the TTT inner loop:
1. matVec — dense matrix-vector multiply
2. vecAdd — element-wise addition
3. outerProduct — rank-1 outer product
4. sgdUpdate — in-place SGD parameter update
5. copyBuffer — element-wise copy

Cross-entropy forward/backward are reused directly from
`Hesper.Training.Loss` (not re-implemented here).
-/

namespace Hesper.TTT.Kernels

open Hesper.WGSL
open Hesper.WGSL.Monad
open Hesper.WebGPU

/-! ## 1. Matrix-Vector Multiply -/

/-- `output[i] = Σ_j weight[i * inDim + j] * input[j]`
    One thread per output row. -/
def matVecKernel (outDim inDim : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let i := Exp.vec3X gid

  let _weight ← ShaderM.declareInputBuffer "weight" (.array (.scalar .f32) (outDim * inDim))
  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) inDim)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) outDim)

  ShaderM.if_ (Exp.lt i (Exp.litU32 outDim)) (do
    let (accName, acc) ← ShaderM.varRef (.scalar .f32) (Exp.litF32 0.0)
    ShaderM.loop (Exp.litU32 0) (Exp.litU32 inDim) (Exp.litU32 1) fun j => do
      let wIdx := Exp.add (Exp.mul i (Exp.litU32 inDim)) j
      let wVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := outDim * inDim) "weight" wIdx
      let xVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := inDim) "input" j
      ShaderM.assign accName (Exp.add acc (Exp.mul wVal xVal))
    ShaderM.writeBuffer (ty := .scalar .f32) "output" i acc
  ) (pure ())

def executeMatVec (device : Device) (weightBuf inputBuf outputBuf : Buffer)
    (outDim inDim : Nat) : IO Unit :=
  Execute.executeShaderNamed device
    (matVecKernel outDim inDim)
    [("weight", weightBuf), ("input", inputBuf), ("output", outputBuf)]
    (Execute.ExecutionConfig.dispatch1D outDim 256)

/-! ## 2. Vector Addition -/

/-- `output[i] = a[i] + b[i]` -/
def vecAddKernel (n : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let i := Exp.vec3X gid

  let _a ← ShaderM.declareInputBuffer "a" (.array (.scalar .f32) n)
  let _b ← ShaderM.declareInputBuffer "b" (.array (.scalar .f32) n)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) n)

  ShaderM.if_ (Exp.lt i (Exp.litU32 n)) (do
    let aVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := n) "a" i
    let bVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := n) "b" i
    ShaderM.writeBuffer (ty := .scalar .f32) "output" i (Exp.add aVal bVal)
  ) (pure ())

def executeVecAdd (device : Device) (aBuf bBuf outputBuf : Buffer)
    (n : Nat) : IO Unit :=
  Execute.executeShaderNamed device
    (vecAddKernel n)
    [("a", aBuf), ("b", bBuf), ("output", outputBuf)]
    (Execute.ExecutionConfig.dispatch1D n 256)

/-! ## 3. Outer Product -/

/-- `result[i * inDim + j] = vec_a[i] * vec_b[j]`
    Writes fresh (does NOT accumulate). -/
def outerProductKernel (outDim inDim : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid

  let total := outDim * inDim
  let _vecA ← ShaderM.declareInputBuffer "vec_a" (.array (.scalar .f32) outDim)
  let _vecB ← ShaderM.declareInputBuffer "vec_b" (.array (.scalar .f32) inDim)
  let _result ← ShaderM.declareOutputBuffer "result" (.array (.scalar .f32) total)

  ShaderM.if_ (Exp.lt idx (Exp.litU32 total)) (do
    let i := Exp.div idx (Exp.litU32 inDim)
    let j := Exp.sub idx (Exp.mul i (Exp.litU32 inDim))
    let aVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := outDim) "vec_a" i
    let bVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := inDim) "vec_b" j
    ShaderM.writeBuffer (ty := .scalar .f32) "result" idx (Exp.mul aVal bVal)
  ) (pure ())

def executeOuterProduct (device : Device) (vecABuf vecBBuf resultBuf : Buffer)
    (outDim inDim : Nat) : IO Unit :=
  Execute.executeShaderNamed device
    (outerProductKernel outDim inDim)
    [("vec_a", vecABuf), ("vec_b", vecBBuf), ("result", resultBuf)]
    (Execute.ExecutionConfig.dispatch1D (outDim * inDim) 256)

/-! ## 4. SGD Update (in-place) -/

/-- `param[i] = param[i] - lr * grad[i]`
    `param` is read-write (declared as output). -/
def sgdUpdateKernel (n : Nat) (lr : Float) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let i := Exp.vec3X gid

  let _grad ← ShaderM.declareInputBuffer "grad" (.array (.scalar .f32) n)
  let _param ← ShaderM.declareOutputBuffer "param" (.array (.scalar .f32) n)

  ShaderM.if_ (Exp.lt i (Exp.litU32 n)) (do
    let pVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := n) "param" i
    let gVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := n) "grad" i
    ShaderM.writeBuffer (ty := .scalar .f32) "param" i
      (Exp.sub pVal (Exp.mul (Exp.litF32 lr) gVal))
  ) (pure ())

def executeSGDUpdate (device : Device) (paramBuf gradBuf : Buffer)
    (n : Nat) (lr : Float) : IO Unit :=
  Execute.executeShaderNamed device
    (sgdUpdateKernel n lr)
    [("grad", gradBuf), ("param", paramBuf)]
    (Execute.ExecutionConfig.dispatch1D n 256)

/-! ## 5. Buffer Copy -/

/-- `dst[i] = src[i]` -/
def copyBufferKernel (n : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let i := Exp.vec3X gid

  let _src ← ShaderM.declareInputBuffer "src" (.array (.scalar .f32) n)
  let _dst ← ShaderM.declareOutputBuffer "dst" (.array (.scalar .f32) n)

  ShaderM.if_ (Exp.lt i (Exp.litU32 n)) (do
    let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := n) "src" i
    ShaderM.writeBuffer (ty := .scalar .f32) "dst" i v
  ) (pure ())

def executeCopy (device : Device) (srcBuf dstBuf : Buffer) (n : Nat) : IO Unit :=
  Execute.executeShaderNamed device
    (copyBufferKernel n)
    [("src", srcBuf), ("dst", dstBuf)]
    (Execute.ExecutionConfig.dispatch1D n 256)

end Hesper.TTT.Kernels
