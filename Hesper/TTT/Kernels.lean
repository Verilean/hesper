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

/-- Smart dispatch: 1D if fits, 2D otherwise.
    Returns `(config, gridDimX)` where gridDimX=0 for 1D, >0 for 2D. -/
private def smartDispatch (totalThreads : Nat) (wgSize : Nat := 256)
    : Execute.ExecutionConfig × Nat :=
  let wgCount := (totalThreads + wgSize - 1) / wgSize
  if wgCount <= 65535 then
    (Execute.ExecutionConfig.dispatch1D totalThreads wgSize, 0)
  else
    let gridX : Nat := 4096
    let gridY := (wgCount + gridX - 1) / gridX
    ({ numWorkgroups := (gridX, gridY, 1),
       workgroupSize := { x := wgSize, y := 1, z := 1 } }, gridX * wgSize)

/-! ## 1. Matrix-Vector Multiply -/

/-- `output[i] = Σ_j weight[i * inDim + j] * input[j]`
    One thread per output row. Supports 2D dispatch for large outDim
    via `i = globalId.x + globalId.y * numWorkgroups.x * workgroupSize.x`. -/
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

def executeMatVec (device : Device) (weightBuf inputBuf outputBuf : Buffer)
    (outDim inDim : Nat) : IO Unit := do
  let (cfg, gdx) := smartDispatch outDim
  Execute.executeShaderNamed device
    (matVecKernel outDim inDim gdx)
    [("weight", weightBuf), ("input", inputBuf), ("output", outputBuf)]
    cfg

/-! ## 2. Vector Addition -/

/-- `output[i] = a[i] + b[i]`. Supports 2D dispatch via gridDimX. -/
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

def executeVecAdd (device : Device) (aBuf bBuf outputBuf : Buffer)
    (n : Nat) : IO Unit := do
  let (cfg, gdx) := smartDispatch n
  Execute.executeShaderNamed device
    (vecAddKernel n gdx)
    [("a", aBuf), ("b", bBuf), ("output", outputBuf)]
    cfg

/-! ## 3. Outer Product -/

/-- `result[i * inDim + j] = vec_a[i] * vec_b[j]`
    Writes fresh (does NOT accumulate). Supports 2D dispatch. -/
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

def executeOuterProduct (device : Device) (vecABuf vecBBuf resultBuf : Buffer)
    (outDim inDim : Nat) : IO Unit := do
  let (cfg, gdx) := smartDispatch (outDim * inDim)
  Execute.executeShaderNamed device
    (outerProductKernel outDim inDim gdx)
    [("vec_a", vecABuf), ("vec_b", vecBBuf), ("result", resultBuf)]
    cfg

/-! ## 4. SGD Update (in-place) -/

/-- `param[i] = param[i] - lr * grad[i]`
    `param` is read-write. Supports 2D dispatch. -/
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

def executeSGDUpdate (device : Device) (paramBuf gradBuf : Buffer)
    (n : Nat) (lr : Float) : IO Unit := do
  let (cfg, gdx) := smartDispatch n
  Execute.executeShaderNamed device
    (sgdUpdateKernel n lr gdx)
    [("grad", gradBuf), ("param", paramBuf)]
    cfg

/-! ## 5. Buffer Copy -/

/-- `dst[i] = src[i]`. Supports 2D dispatch. -/
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

def executeCopy (device : Device) (srcBuf dstBuf : Buffer) (n : Nat) : IO Unit := do
  let (cfg, gdx) := smartDispatch n
  Execute.executeShaderNamed device
    (copyBufferKernel n gdx)
    [("src", srcBuf), ("dst", dstBuf)]
    cfg

/-! ## 6. MSE Residual Loss + Gradient (Hidden-Space TTT) -/

/-- Fused MSE loss + gradient for hidden-space TTT.

    Given:
      ttt_output[i] = W_ttt @ hidden_t  (the TTT prediction)
      hidden_t[i]   = current hidden state
      hidden_t1[i]  = next token's hidden state
      target_residual[i] = hidden_t1[i] - hidden_t[i]

    Computes:
      loss = (1/dim) * Σ_i (ttt_output[i] - target_residual[i])²
      d_ttt_output[i] = (2/dim) * (ttt_output[i] - target_residual[i])

    Uses a single workgroup with shared-memory reduction for the scalar
    loss, then every thread writes its gradient element.
    Requires dim ≤ workgroupSize (true for typical hidden dims ≤ 4096
    with wgSize=256 when using strided loops). -/
def mseResidualLossAndGradKernel (dim : Nat) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let lid ← ShaderM.localId
  let tid := Exp.vec3X lid

  let _tttOut  ← ShaderM.declareInputBuffer "ttt_output" (.array (.scalar .f32) dim)
  let _hiddenT ← ShaderM.declareInputBuffer "hidden_t" (.array (.scalar .f32) dim)
  let _hiddenT1 ← ShaderM.declareInputBuffer "hidden_t1" (.array (.scalar .f32) dim)
  let _grad ← ShaderM.declareOutputBuffer "grad" (.array (.scalar .f32) dim)
  let _loss ← ShaderM.declareOutputBuffer "loss" (.array (.scalar .f32) 1)

  ShaderM.sharedNamed "shared_sum" (.array (.scalar .f32) workgroupSize)

  -- Phase 1: each thread computes partial sum of squared errors
  ShaderM.varNamed "partialSq" (.scalar .f32) (Exp.litF32 0.0)
  let partialSq : Exp (.scalar .f32) := Exp.var "partialSq"

  ShaderM.loop tid (Exp.litU32 dim) (Exp.litU32 workgroupSize) fun i => do
    let pred ← ShaderM.readBuffer (ty := .scalar .f32) (n := dim) "ttt_output" i
    let ht  ← ShaderM.readBuffer (ty := .scalar .f32) (n := dim) "hidden_t" i
    let ht1 ← ShaderM.readBuffer (ty := .scalar .f32) (n := dim) "hidden_t1" i
    -- target_residual = hidden_t1 - hidden_t
    let target := Exp.sub ht1 ht
    let diff := Exp.sub pred target
    -- Accumulate squared error
    ShaderM.assign "partialSq" (Exp.add partialSq (Exp.mul diff diff))
    -- Write gradient: d_ttt_output[i] = 2 * diff / dim
    let gradVal := Exp.div (Exp.mul (Exp.litF32 2.0) diff) (Exp.litF32 dim.toFloat)
    ShaderM.writeBuffer (ty := .scalar .f32) "grad" i gradVal

  -- Phase 2: reduce partial sums for scalar loss
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

  -- Thread 0 writes loss = total_sq / dim
  ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
    let totalSq ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_sum" (Exp.litU32 0)
    ShaderM.writeBuffer (ty := .scalar .f32) "loss" (Exp.litU32 0)
      (Exp.div totalSq (Exp.litF32 dim.toFloat))
  ) (pure ())

def executeMSEResidualLossAndGrad (device : Device)
    (tttOutBuf hiddenTBuf hiddenT1Buf gradBuf lossBuf : Buffer) (dim : Nat) : IO Unit := do
  let workgroupSize := min 256 (max dim 32)
  Execute.executeShaderNamed device
    (mseResidualLossAndGradKernel dim workgroupSize)
    [("ttt_output", tttOutBuf), ("hidden_t", hiddenTBuf),
     ("hidden_t1", hiddenT1Buf), ("grad", gradBuf), ("loss", lossBuf)]
    { numWorkgroups := (1, 1, 1),
      workgroupSize := { x := workgroupSize, y := 1, z := 1 } }

/-! ## 7. Cosine Surprise Sensor (scale-invariant) -/

/-- Cosine distance sensor: measures how "surprising" the transition
    from hidden_t to hidden_t1 is. Scale-invariant — works regardless
    of embedding scale (no inf/NaN on Gemma 4's sqrt(d) scaling).

    surprise = 1 - cosine_sim(hidden_t, hidden_t1)
    where cosine_sim = dot(h_t, h_t1) / (||h_t|| * ||h_t1||)

    Output: lossBuf[0] = surprise ∈ [0, 2] (0 = identical, 2 = opposite)
    Uses shared memory tree reduction for dot product and norms. -/
def cosineSurpriseSensorKernel (dim : Nat) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let lid ← ShaderM.localId
  let tid := Exp.vec3X lid

  let _hiddenT  ← ShaderM.declareInputBuffer "hidden_t" (.array (.scalar .f32) dim)
  let _hiddenT1 ← ShaderM.declareInputBuffer "hidden_t1" (.array (.scalar .f32) dim)
  let _loss ← ShaderM.declareOutputBuffer "loss" (.array (.scalar .f32) 1)

  -- Three shared arrays for parallel reduction: dot, normT², normT1²
  ShaderM.sharedNamed "shared_dot" (.array (.scalar .f32) workgroupSize)
  ShaderM.sharedNamed "shared_normT" (.array (.scalar .f32) workgroupSize)
  ShaderM.sharedNamed "shared_normT1" (.array (.scalar .f32) workgroupSize)

  -- Phase 1: per-thread partial sums.
  -- To avoid f32 overflow with Gemma 4's huge hidden values (~1e15+),
  -- we first find the max absolute value, then scale all elements by
  -- 1/maxAbs before computing dot products and norms.

  -- Phase 1a: find max absolute value across both vectors
  ShaderM.varNamed "partialMax" (.scalar .f32) (Exp.litF32 0.0)
  let pMax : Exp (.scalar .f32) := Exp.var "partialMax"
  ShaderM.loop tid (Exp.litU32 dim) (Exp.litU32 workgroupSize) fun i => do
    let ht ← ShaderM.readBuffer (ty := .scalar .f32) (n := dim) "hidden_t" i
    let ht1 ← ShaderM.readBuffer (ty := .scalar .f32) (n := dim) "hidden_t1" i
    ShaderM.assign "partialMax" (Exp.max pMax (Exp.max (Exp.abs ht) (Exp.abs ht1)))

  ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_dot" tid pMax
  ShaderM.barrier
  let mut maxStride := workgroupSize / 2
  while maxStride > 0 do
    ShaderM.if_ (Exp.lt tid (Exp.litU32 maxStride)) (do
      let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_dot" tid
      let b ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_dot" (Exp.add tid (Exp.litU32 maxStride))
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_dot" tid (Exp.max a b)
    ) (pure ())
    ShaderM.barrier
    maxStride := maxStride / 2
  -- Broadcast max to all threads via shared[0]
  ShaderM.varNamed "globalMax" (.scalar .f32)
    (← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_dot" (Exp.litU32 0))
  let globalMax : Exp (.scalar .f32) := Exp.var "globalMax"
  let invMax := Exp.div (Exp.litF32 1.0) (Exp.max globalMax (Exp.litF32 1.0e-30))
  ShaderM.barrier

  -- Phase 1b: compute scaled dot product and norms
  ShaderM.varNamed "partialDot" (.scalar .f32) (Exp.litF32 0.0)
  ShaderM.varNamed "partialNormT" (.scalar .f32) (Exp.litF32 0.0)
  ShaderM.varNamed "partialNormT1" (.scalar .f32) (Exp.litF32 0.0)
  let pDot : Exp (.scalar .f32) := Exp.var "partialDot"
  let pNormT : Exp (.scalar .f32) := Exp.var "partialNormT"
  let pNormT1 : Exp (.scalar .f32) := Exp.var "partialNormT1"

  ShaderM.loop tid (Exp.litU32 dim) (Exp.litU32 workgroupSize) fun i => do
    let ht_raw ← ShaderM.readBuffer (ty := .scalar .f32) (n := dim) "hidden_t" i
    let ht1_raw ← ShaderM.readBuffer (ty := .scalar .f32) (n := dim) "hidden_t1" i
    -- Scale by 1/maxAbs to prevent overflow
    let ht := Exp.mul ht_raw invMax
    let ht1 := Exp.mul ht1_raw invMax
    ShaderM.assign "partialDot" (Exp.add pDot (Exp.mul ht ht1))
    ShaderM.assign "partialNormT" (Exp.add pNormT (Exp.mul ht ht))
    ShaderM.assign "partialNormT1" (Exp.add pNormT1 (Exp.mul ht1 ht1))

  -- Phase 2: tree reduction for all three
  ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_dot" tid pDot
  ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_normT" tid pNormT
  ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_normT1" tid pNormT1
  ShaderM.barrier

  let mut stride := workgroupSize / 2
  while stride > 0 do
    ShaderM.if_ (Exp.lt tid (Exp.litU32 stride)) (do
      let d1 ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_dot" tid
      let d2 ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_dot" (Exp.add tid (Exp.litU32 stride))
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_dot" tid (Exp.add d1 d2)
      let n1 ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_normT" tid
      let n2 ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_normT" (Exp.add tid (Exp.litU32 stride))
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_normT" tid (Exp.add n1 n2)
      let m1 ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_normT1" tid
      let m2 ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_normT1" (Exp.add tid (Exp.litU32 stride))
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_normT1" tid (Exp.add m1 m2)
    ) (pure ())
    ShaderM.barrier
    stride := stride / 2

  -- Thread 0: compute cosine similarity and write surprise
  ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
    let dotVal ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_dot" (Exp.litU32 0)
    let normTSq ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_normT" (Exp.litU32 0)
    let normT1Sq ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "shared_normT1" (Exp.litU32 0)
    let normProduct := Exp.mul (Exp.sqrt normTSq) (Exp.sqrt normT1Sq)
    -- Guard against division by zero
    let safeNorm := Exp.max normProduct (Exp.litF32 1.0e-8)
    let cosineSim := Exp.div dotVal safeNorm
    -- Clamp to [-1, 1] for numerical safety
    let clampedSim := Exp.max (Exp.litF32 (-1.0)) (Exp.min (Exp.litF32 1.0) cosineSim)
    let surprise := Exp.sub (Exp.litF32 1.0) clampedSim
    ShaderM.writeBuffer (ty := .scalar .f32) "loss" (Exp.litU32 0) surprise
  ) (pure ())

def executeCosineSurpriseSensor (device : Device)
    (hiddenTBuf hiddenT1Buf lossBuf : Buffer) (dim : Nat) : IO Unit := do
  let workgroupSize := min 256 (max dim 32)
  Execute.executeShaderNamed device
    (cosineSurpriseSensorKernel dim workgroupSize)
    [("hidden_t", hiddenTBuf), ("hidden_t1", hiddenT1Buf), ("loss", lossBuf)]
    { numWorkgroups := (1, 1, 1),
      workgroupSize := { x := workgroupSize, y := 1, z := 1 } }

/-! ## 8. KV Cache Zero-Out (Eviction) -/

/-- Zero out a single KV cache row at position `pos` across one layer.
    Sets K[head, pos, :] = 0 and V[head, pos, :] = 0 for all KV heads.
    Zeroed K forces dot(Q, K) = 0, minimizing attention weight. -/
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

def executeZeroKVRow (device : Device) (kCacheBuf vCacheBuf : Buffer)
    (numKVHeads maxSeqLen headDim pos : Nat) : IO Unit := do
  let kvDim := numKVHeads * headDim
  Execute.executeShaderNamed device
    (zeroKVRowKernel numKVHeads maxSeqLen headDim pos)
    [("k_cache", kCacheBuf), ("v_cache", vCacheBuf)]
    (Execute.ExecutionConfig.dispatch1D kvDim 256)

end Hesper.TTT.Kernels
