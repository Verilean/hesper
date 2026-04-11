import Hesper.WGSL.Monad
import Hesper.WGSL.Exp
import Hesper.Backend

/-!
# TTT GPU Kernels

Backend-agnostic compute kernels for the TTT inner loop.
Kernel definitions (ShaderM) are pure — no backend dependency.
Execute functions use `[GPUBackend β]` typeclass.
-/

namespace Hesper.TTT.Kernels

open Hesper
open Hesper.WGSL
open Hesper.WGSL.Monad

/-! ## 1. Matrix-Vector Multiply -/

/-- `output[i] = Σ_j weight[i * inDim + j] * input[j]` -/
def matVecKernel (outDim inDim : Nat) (gridDimX : Nat := 0) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let i := if gridDimX == 0 then Exp.vec3X gid
    else Exp.add (Exp.vec3X gid) (Exp.mul (Exp.vec3Y gid) (Exp.litU32 gridDimX))

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

@[inline]
def executeMatVec [GPUBackend β] (ctx : β) (weightBuf inputBuf outputBuf : GPUBackend.Buf β)
    (outDim inDim : Nat) : IO Unit := do
  let (cfg, gdx) := smartDispatch outDim
  GPUBackend.execute ctx (matVecKernel outDim inDim gdx)
    [("weight", weightBuf), ("input", inputBuf), ("output", outputBuf)] cfg

/-! ## 2. Vector Addition -/

/-- `output[i] = a[i] + b[i]` -/
def vecAddKernel (n : Nat) (gridDimX : Nat := 0) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let i := if gridDimX == 0 then Exp.vec3X gid
    else Exp.add (Exp.vec3X gid) (Exp.mul (Exp.vec3Y gid) (Exp.litU32 gridDimX))

  let _a ← ShaderM.declareInputBuffer "a" (.array (.scalar .f32) n)
  let _b ← ShaderM.declareInputBuffer "b" (.array (.scalar .f32) n)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) n)

  ShaderM.if_ (Exp.lt i (Exp.litU32 n)) (do
    let aVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := n) "a" i
    let bVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := n) "b" i
    ShaderM.writeBuffer (ty := .scalar .f32) "output" i (Exp.add aVal bVal)
  ) (pure ())

@[inline]
def executeVecAdd [GPUBackend β] (ctx : β) (aBuf bBuf outputBuf : GPUBackend.Buf β)
    (n : Nat) : IO Unit := do
  let (cfg, gdx) := smartDispatch n
  GPUBackend.execute ctx (vecAddKernel n gdx)
    [("a", aBuf), ("b", bBuf), ("output", outputBuf)] cfg

/-! ## 3. Outer Product -/

/-- `result[i * inDim + j] = vec_a[i] * vec_b[j]` -/
def outerProductKernel (outDim inDim : Nat) (gridDimX : Nat := 0) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := if gridDimX == 0 then Exp.vec3X gid
    else Exp.add (Exp.vec3X gid) (Exp.mul (Exp.vec3Y gid) (Exp.litU32 gridDimX))

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

@[inline]
def executeOuterProduct [GPUBackend β] (ctx : β)
    (vecABuf vecBBuf resultBuf : GPUBackend.Buf β)
    (outDim inDim : Nat) : IO Unit := do
  let (cfg, gdx) := smartDispatch (outDim * inDim)
  GPUBackend.execute ctx (outerProductKernel outDim inDim gdx)
    [("vec_a", vecABuf), ("vec_b", vecBBuf), ("result", resultBuf)] cfg

/-! ## 4. SGD Update (in-place) -/

/-- `param[i] = param[i] - lr * grad[i]` -/
def sgdUpdateKernel (n : Nat) (lr : Float) (gridDimX : Nat := 0) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let i := if gridDimX == 0 then Exp.vec3X gid
    else Exp.add (Exp.vec3X gid) (Exp.mul (Exp.vec3Y gid) (Exp.litU32 gridDimX))

  let _grad ← ShaderM.declareInputBuffer "grad" (.array (.scalar .f32) n)
  let _param ← ShaderM.declareOutputBuffer "param" (.array (.scalar .f32) n)

  ShaderM.if_ (Exp.lt i (Exp.litU32 n)) (do
    let pVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := n) "param" i
    let gVal ← ShaderM.readBuffer (ty := .scalar .f32) (n := n) "grad" i
    ShaderM.writeBuffer (ty := .scalar .f32) "param" i
      (Exp.sub pVal (Exp.mul (Exp.litF32 lr) gVal))
  ) (pure ())

@[inline]
def executeSGDUpdate [GPUBackend β] (ctx : β)
    (paramBuf gradBuf : GPUBackend.Buf β)
    (n : Nat) (lr : Float) : IO Unit := do
  let (cfg, gdx) := smartDispatch n
  GPUBackend.execute ctx (sgdUpdateKernel n lr gdx)
    [("grad", gradBuf), ("param", paramBuf)] cfg

/-! ## 5. Buffer Copy -/

/-- `dst[i] = src[i]` -/
def copyBufferKernel (n : Nat) (gridDimX : Nat := 0) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let i := if gridDimX == 0 then Exp.vec3X gid
    else Exp.add (Exp.vec3X gid) (Exp.mul (Exp.vec3Y gid) (Exp.litU32 gridDimX))

  let _src ← ShaderM.declareInputBuffer "src" (.array (.scalar .f32) n)
  let _dst ← ShaderM.declareOutputBuffer "dst" (.array (.scalar .f32) n)

  ShaderM.if_ (Exp.lt i (Exp.litU32 n)) (do
    let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := n) "src" i
    ShaderM.writeBuffer (ty := .scalar .f32) "dst" i v
  ) (pure ())

@[inline]
def executeCopy [GPUBackend β] (ctx : β) (srcBuf dstBuf : GPUBackend.Buf β)
    (n : Nat) : IO Unit := do
  let (cfg, gdx) := smartDispatch n
  GPUBackend.execute ctx (copyBufferKernel n gdx)
    [("src", srcBuf), ("dst", dstBuf)] cfg

/-! ## 6. MSE Residual Loss + Gradient (Hidden-Space TTT) -/

/-- MSE loss with shared memory tree reduction.
    loss = Σ(pred - target)² / dim
    grad[i] = 2 * (pred - target) / dim -/
def mseResidualLossAndGradKernel (dim : Nat) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let lid ← ShaderM.localId
  let tid := Exp.vec3X lid

  let _tttOut  ← ShaderM.declareInputBuffer "ttt_output" (.array (.scalar .f32) dim)
  let _hiddenT ← ShaderM.declareInputBuffer "hidden_t" (.array (.scalar .f32) dim)
  let _hiddenT1 ← ShaderM.declareInputBuffer "hidden_t1" (.array (.scalar .f32) dim)
  let _grad ← ShaderM.declareOutputBuffer "grad" (.array (.scalar .f32) dim)
  let _loss ← ShaderM.declareOutputBuffer "loss" (.array (.scalar .f32) 1)

  ShaderM.sharedNamed "shared_sum" (.array (.scalar .f32) workgroupSize)

  ShaderM.varNamed "partialSq" (.scalar .f32) (Exp.litF32 0.0)
  let partialSq : Exp (.scalar .f32) := Exp.var "partialSq"

  ShaderM.loop tid (Exp.litU32 dim) (Exp.litU32 workgroupSize) fun i => do
    let pred ← ShaderM.readBuffer (ty := .scalar .f32) (n := dim) "ttt_output" i
    let ht  ← ShaderM.readBuffer (ty := .scalar .f32) (n := dim) "hidden_t" i
    let ht1 ← ShaderM.readBuffer (ty := .scalar .f32) (n := dim) "hidden_t1" i
    let target := Exp.sub ht1 ht
    let diff := Exp.sub pred target
    ShaderM.assign "partialSq" (Exp.add partialSq (Exp.mul diff diff))
    let gradVal := Exp.div (Exp.mul (Exp.litF32 2.0) diff) (Exp.litF32 dim.toFloat)
    ShaderM.writeBuffer (ty := .scalar .f32) "grad" i gradVal

  ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_sum" tid partialSq
  ShaderM.barrier

  let mut stride := workgroupSize / 2
  while stride > 0 do
    ShaderM.if_ (Exp.lt tid (Exp.litU32 stride)) (do
      let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_sum" tid
      let b ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_sum"
                (Exp.add tid (Exp.litU32 stride))
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_sum" tid (Exp.add a b)
    ) (pure ())
    ShaderM.barrier
    stride := stride / 2

  ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
    let totalSq ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_sum" (Exp.litU32 0)
    ShaderM.writeBuffer (ty := .scalar .f32) "loss" (Exp.litU32 0)
      (Exp.div totalSq (Exp.litF32 dim.toFloat))
  ) (pure ())

@[inline]
def executeMSEResidualLossAndGrad [GPUBackend β] (ctx : β)
    (tttOutBuf hiddenTBuf hiddenT1Buf gradBuf lossBuf : GPUBackend.Buf β)
    (dim : Nat) : IO Unit := do
  let workgroupSize := min 256 (max dim 32)
  GPUBackend.execute ctx
    (mseResidualLossAndGradKernel dim workgroupSize)
    [("ttt_output", tttOutBuf), ("hidden_t", hiddenTBuf),
     ("hidden_t1", hiddenT1Buf), ("grad", gradBuf), ("loss", lossBuf)]
    { numWorkgroups := (1, 1, 1),
      workgroupSize := { x := workgroupSize, y := 1, z := 1 } }

/-! ## 7. Cosine Surprise Sensor -/

def cosineSurpriseSensorKernel (dim : Nat) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let lid ← ShaderM.localId
  let tid := Exp.vec3X lid

  let _hiddenT  ← ShaderM.declareInputBuffer "hidden_t" (.array (.scalar .f32) dim)
  let _hiddenT1 ← ShaderM.declareInputBuffer "hidden_t1" (.array (.scalar .f32) dim)
  let _loss ← ShaderM.declareOutputBuffer "loss" (.array (.scalar .f32) 1)

  ShaderM.sharedNamed "shared_dot" (.array (.scalar .f32) workgroupSize)
  ShaderM.sharedNamed "shared_norm_t" (.array (.scalar .f32) workgroupSize)
  ShaderM.sharedNamed "shared_norm_t1" (.array (.scalar .f32) workgroupSize)

  -- Phase 1: find max absolute value for scaling
  ShaderM.varNamed "maxAbs" (.scalar .f32) (Exp.litF32 0.0)
  let maxAbs : Exp (.scalar .f32) := Exp.var "maxAbs"

  ShaderM.loop tid (Exp.litU32 dim) (Exp.litU32 workgroupSize) fun i => do
    let ht  ← ShaderM.readBuffer (ty := .scalar .f32) (n := dim) "hidden_t" i
    let ht1 ← ShaderM.readBuffer (ty := .scalar .f32) (n := dim) "hidden_t1" i
    let absHt := Exp.abs ht
    let absHt1 := Exp.abs ht1
    ShaderM.assign "maxAbs" (Exp.max maxAbs (Exp.max absHt absHt1))

  -- Phase 2: compute scaled dot/norms
  ShaderM.varNamed "partDot" (.scalar .f32) (Exp.litF32 0.0)
  ShaderM.varNamed "partNormT" (.scalar .f32) (Exp.litF32 0.0)
  ShaderM.varNamed "partNormT1" (.scalar .f32) (Exp.litF32 0.0)
  let partDot : Exp (.scalar .f32) := Exp.var "partDot"
  let partNormT : Exp (.scalar .f32) := Exp.var "partNormT"
  let partNormT1 : Exp (.scalar .f32) := Exp.var "partNormT1"

  let safeMax := Exp.max maxAbs (Exp.litF32 1.0e-10)
  let invScale := Exp.div (Exp.litF32 1.0) safeMax

  ShaderM.loop tid (Exp.litU32 dim) (Exp.litU32 workgroupSize) fun i => do
    let ht  ← ShaderM.readBuffer (ty := .scalar .f32) (n := dim) "hidden_t" i
    let ht1 ← ShaderM.readBuffer (ty := .scalar .f32) (n := dim) "hidden_t1" i
    let sHt := Exp.mul ht invScale
    let sHt1 := Exp.mul ht1 invScale
    ShaderM.assign "partDot" (Exp.add partDot (Exp.mul sHt sHt1))
    ShaderM.assign "partNormT" (Exp.add partNormT (Exp.mul sHt sHt))
    ShaderM.assign "partNormT1" (Exp.add partNormT1 (Exp.mul sHt1 sHt1))

  -- Phase 3: shared mem reduction
  ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_dot" tid partDot
  ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_norm_t" tid partNormT
  ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_norm_t1" tid partNormT1
  ShaderM.barrier

  let mut stride := workgroupSize / 2
  while stride > 0 do
    ShaderM.if_ (Exp.lt tid (Exp.litU32 stride)) (do
      let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_dot" tid
      let b ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_dot"
                (Exp.add tid (Exp.litU32 stride))
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_dot" tid (Exp.add a b)
      let c ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_norm_t" tid
      let d ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_norm_t"
                (Exp.add tid (Exp.litU32 stride))
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_norm_t" tid (Exp.add c d)
      let e ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_norm_t1" tid
      let f ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_norm_t1"
                (Exp.add tid (Exp.litU32 stride))
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_norm_t1" tid (Exp.add e f)
    ) (pure ())
    ShaderM.barrier
    stride := stride / 2

  ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
    let dot ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_dot" (Exp.litU32 0)
    let nt ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_norm_t" (Exp.litU32 0)
    let nt1 ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_norm_t1" (Exp.litU32 0)
    let normProd := Exp.sqrt (Exp.mul nt nt1)
    let cosSim := Exp.div dot (Exp.max normProd (Exp.litF32 1.0e-10))
    ShaderM.writeBuffer (ty := .scalar .f32) "loss" (Exp.litU32 0) (Exp.sub (Exp.litF32 1.0) cosSim)
  ) (pure ())

@[inline]
def executeCosineSurpriseSensor [GPUBackend β] (ctx : β)
    (hiddenTBuf hiddenT1Buf lossBuf : GPUBackend.Buf β)
    (dim : Nat) : IO Unit := do
  let workgroupSize := min 256 (max dim 32)
  GPUBackend.execute ctx
    (cosineSurpriseSensorKernel dim workgroupSize)
    [("hidden_t", hiddenTBuf), ("hidden_t1", hiddenT1Buf), ("loss", lossBuf)]
    { numWorkgroups := (1, 1, 1),
      workgroupSize := { x := workgroupSize, y := 1, z := 1 } }

/-! ## 8. Zero KV Row -/

def zeroKVRowKernel (numKVHeads maxSeqLen headDim : Nat) (pos : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid
  let kvDim := numKVHeads * headDim

  let _kCache ← ShaderM.declareOutputBuffer "k_cache" (.array (.scalar .f32) (numKVHeads * maxSeqLen * headDim))
  let _vCache ← ShaderM.declareOutputBuffer "v_cache" (.array (.scalar .f32) (numKVHeads * maxSeqLen * headDim))

  ShaderM.if_ (Exp.lt idx (Exp.litU32 kvDim)) (do
    let kvHead := Exp.div idx (Exp.litU32 headDim)
    let d := Exp.sub idx (Exp.mul kvHead (Exp.litU32 headDim))
    let cacheIdx := Exp.add
      (Exp.mul kvHead (Exp.litU32 (maxSeqLen * headDim)))
      (Exp.add (Exp.litU32 (pos * headDim)) d)
    ShaderM.writeBuffer (ty := .scalar .f32) "k_cache" cacheIdx (Exp.litF32 0.0)
    ShaderM.writeBuffer (ty := .scalar .f32) "v_cache" cacheIdx (Exp.litF32 0.0)
  ) (pure ())

@[inline]
def executeZeroKVRow [GPUBackend β] (ctx : β)
    (kCacheBuf vCacheBuf : GPUBackend.Buf β)
    (numKVHeads maxSeqLen headDim pos : Nat) : IO Unit := do
  let kvDim := numKVHeads * headDim
  GPUBackend.execute ctx
    (zeroKVRowKernel numKVHeads maxSeqLen headDim pos)
    [("k_cache", kCacheBuf), ("v_cache", vCacheBuf)]
    (ExecConfig.dispatch1D kvDim)

end Hesper.TTT.Kernels
