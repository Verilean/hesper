import Hesper
import Hesper.Training.AttentionBackward
import Hesper.Training.FFNBackward
import Hesper.Training.VerifiedBackward
import Hesper.Training.SafeBuffer
import Hesper.AD.Verified

/-!
# GPU vs CPU Backward Consistency Test

For each backward GPU kernel, uploads test data, runs the GPU kernel,
downloads the result, and compares it to the CPU spec output.

This ensures the WGSL shader produces the same result as the verified
pure-Lean backward function.
-/

open Hesper.WebGPU
open Hesper.Training.SafeBuffer
open Hesper.Training.VerifiedBackward
open Hesper.AD.Verified

/-- Upload Float array to GPU buffer -/
def uploadFloats (device : Device) (buf : Buffer) (vals : Array Float) : IO Unit :=
  writeBuffer device buf 0 (floatArrayToBytes vals)

/-- Compare GPU result with CPU spec -/
def compareResults (gpuResult cpuResult : Array Float) (name : String) (tol : Float := 1e-3) : IO Bool := do
  let n := min gpuResult.size cpuResult.size
  let mut maxErr := 0.0
  let mut ok := true
  for i in [:n] do
    let g := gpuResult.getD i 0.0
    let c := cpuResult.getD i 0.0
    if isNaN g then
      IO.println s!"  {name}[{i}]: GPU=NaN, CPU={c}"
      ok := false
    else
      let diff := if g - c < 0.0 then c - g else g - c
      let denom := (if g < 0.0 then -g else g) + (if c < 0.0 then -c else c)
      let err := if denom < 1e-10 then diff else diff / denom
      if err > maxErr then maxErr := err
      if err > tol then
        if ok then  -- only print first mismatch
          IO.println s!"  {name}[{i}]: GPU={g}, CPU={c}, err={err}"
        ok := false
  if ok then
    IO.println s!"  ✓ {name}: max_err={maxErr} (n={n})"
  else
    IO.println s!"  ✗ {name}: max_err={maxErr} MISMATCH"
  return ok

def main : IO Unit := do
  IO.println "=== GPU vs CPU Backward Consistency Test ==="
  IO.println ""

  let inst ← Hesper.init
  let device ← Hesper.WebGPU.getDevice inst
  let mut allPassed := true

  -- Helper to create a GPU buffer of N floats
  let mkBuf := fun (n : Nat) =>
    createBuffer device { size := (n * 4).toUSize, usage := [.storage, .copySrc, .copyDst, .mapRead], mappedAtCreation := false }

  -- ============================================================
  -- 1. Softmax Backward
  -- ============================================================
  IO.println "1. Softmax Backward"
  do
    let n := 16  -- small attention: 4 heads × 4 cacheLen
    let numHeads := 4
    let cacheLen := 4

    -- Test data: attention weights (softmax output) and dAttn
    let attnData := softmaxFwd #[1.0, 2.0, 0.5, 1.5, 3.0, 1.0, 2.0, 0.5,
                                  1.5, 2.5, 0.5, 1.0, 2.0, 1.5, 3.0, 0.5]
    let dAttnData : Array Float := #[0.1, -0.2, 0.3, 0.1, -0.1, 0.2, 0.1, -0.3,
                                      0.2, -0.1, 0.1, 0.3, -0.2, 0.1, 0.2, -0.1]

    -- CPU spec: per-row softmax backward
    -- GPU kernel takes pre-computed attn weights (softmax output), not logits.
    -- The backward formula is: dScores[i] = attn[i] * (dAttn[i] - Σ_j attn[j]*dAttn[j])
    -- Apply this directly using attnData as the softmax output.
    let mut smCpuResult := #[]
    for h in [:numHeads] do
      let rowStart := h * cacheLen
      let s := Array.ofFn (n := cacheLen) fun i => attnData.getD (rowStart + i.val) 0.0
      let dy := Array.ofFn (n := cacheLen) fun i => dAttnData.getD (rowStart + i.val) 0.0
      -- dot = Σ s[j] * dy[j]
      let dot := (Array.zipWith (· * ·) s dy).foldl (· + ·) 0.0
      -- dx[i] = s[i] * (dy[i] - dot)
      let dx := Array.zipWith (fun si di => si * (di - dot)) s dy
      for i in [:cacheLen] do
        smCpuResult := smCpuResult.push (dx.getD i 0.0)

    -- GPU
    let attnBuf ← mkBuf n
    let dAttnBuf ← mkBuf n
    let dScoresBuf ← mkBuf n
    uploadFloats device attnBuf attnData
    uploadFloats device dAttnBuf dAttnData
    Hesper.Training.AttentionBackward.executeSoftmaxBackward device attnBuf dAttnBuf dScoresBuf numHeads cacheLen
    let gpuResult ← safeMapBufferReadF32 device dScoresBuf n

    let ok ← compareResults gpuResult smCpuResult "SoftmaxBackward"
    if !ok then allPassed := false

  -- ============================================================
  -- 2. RMSNorm Backward
  -- ============================================================
  IO.println "2. RMSNorm Backward"
  do
    let dim := 8
    let xData := #[1.0, -2.0, 3.0, 0.5, -1.0, 2.0, -0.5, 1.5]
    let gammaData := #[1.0, 0.5, 2.0, 1.5, 1.0, 0.5, 2.0, 1.5]
    let dOutData := #[0.1, -0.3, 0.2, 0.5, -0.1, 0.2, -0.2, 0.3]
    let eps := 1e-6

    -- CPU spec
    let rmsCpuResult := rmsNormBackward xData gammaData dOutData eps

    -- GPU
    let xBuf ← mkBuf dim
    let gammaBuf ← mkBuf dim
    let dOutBuf ← mkBuf dim
    let dInBuf ← mkBuf dim
    uploadFloats device xBuf xData
    uploadFloats device gammaBuf gammaData
    uploadFloats device dOutBuf dOutData
    Hesper.Training.AttentionBackward.executeRmsNormBackward device xBuf gammaBuf dOutBuf dInBuf dim eps
    let gpuResult ← safeMapBufferReadF32 device dInBuf dim

    let ok ← compareResults gpuResult rmsCpuResult "RMSNormBackward"
    if !ok then allPassed := false

  -- ============================================================
  -- 3. RoPE Backward
  -- ============================================================
  IO.println "3. RoPE Backward"
  do
    let numHeads := 2
    let headDim := 4  -- halfDim = 2
    let n := numHeads * headDim  -- 8
    let ropeBase := 10000.0
    let pos := 3

    let dOutData := #[0.1, -0.2, 0.3, 0.4, -0.1, 0.5, -0.3, 0.2]

    -- CPU spec: per-head, per-pair RoPE backward
    let mut ropeCpuResult := Array.replicate n 0.0
    let halfDim := headDim / 2
    for h in [:numHeads] do
      for d in [:halfDim] do
        let theta := pos.toFloat * Float.pow ropeBase (-(2.0 * d.toFloat / headDim.toFloat))
        let idx0 := h * headDim + d
        let idx1 := h * headDim + d + halfDim
        let dy0 := dOutData.getD idx0 0.0
        let dy1 := dOutData.getD idx1 0.0
        let (dx0, dx1) := ropeBackward dy0 dy1 theta
        ropeCpuResult := ropeCpuResult.set! idx0 dx0
        ropeCpuResult := ropeCpuResult.set! idx1 dx1

    -- GPU
    let dOutBuf ← mkBuf n
    let dInBuf ← mkBuf n
    uploadFloats device dOutBuf dOutData
    Hesper.Training.AttentionBackward.executeRopeBackward device dOutBuf dInBuf numHeads headDim ropeBase pos
    let gpuResult ← safeMapBufferReadF32 device dInBuf n

    let ok ← compareResults gpuResult ropeCpuResult "RoPEBackward"
    if !ok then allPassed := false

  -- ============================================================
  -- 4. ReLU²×Mul Backward
  -- ============================================================
  IO.println "4. ReLU²×Mul Backward"
  do
    let n := 4
    let gateData := #[1.0, -0.5, 2.0, 0.3]
    let upData := #[0.5, 1.0, -1.0, 2.0]
    let dHData := #[0.1, -0.2, 0.3, 0.5]

    -- CPU spec
    let cpuInput := gateData ++ upData
    let cpuBwd := reluSqrMulBwd cpuInput dHData
    let cpuDGate := Array.ofFn (n := n) fun i => cpuBwd.getD i.val 0.0
    let cpuDUp := Array.ofFn (n := n) fun i => cpuBwd.getD (i.val + n) 0.0

    -- GPU
    let gateBuf ← mkBuf n
    let upBuf ← mkBuf n
    let dHBuf ← mkBuf n
    let dGateBuf ← mkBuf n
    let dUpBuf ← mkBuf n
    uploadFloats device gateBuf gateData
    uploadFloats device upBuf upData
    uploadFloats device dHBuf dHData
    Hesper.Training.FFNBackward.executeReluSqrMulBackward device gateBuf upBuf dHBuf dGateBuf dUpBuf n
    let gpuDGate ← safeMapBufferReadF32 device dGateBuf n
    let gpuDUp ← safeMapBufferReadF32 device dUpBuf n

    let ok1 ← compareResults gpuDGate cpuDGate "ReLU²×Mul_dGate"
    let ok2 ← compareResults gpuDUp cpuDUp "ReLU²×Mul_dUp"
    if !ok1 || !ok2 then allPassed := false

  -- ============================================================
  -- Summary
  -- ============================================================
  IO.println ""
  if allPassed then
    IO.println "✓ All GPU kernels match CPU specs"
  else
    IO.println "✗ Some GPU kernels DON'T match CPU specs — investigate!"
