import Hesper
import Hesper.Compute
import Hesper.Backend.WebGPU
import Hesper.Data.DataFrame
import Hesper.WGSL.MatMul
import Hesper.Training.MSE

/-!
# Ch11 California Housing — GPU edition

Same dataset, same hyperparameters, same Sabela loss trajectory as
`CaliforniaHousing.lean`, but every per-iteration arithmetic step is
dispatched onto WebGPU.

Layout:
* Bias is folded into the parameter vector by appending a `1.0`
  column to `X`, so `pred = X @ w` is a pure matmul.
* Forward (`X @ w`) uses `Hesper.WGSL.MatMul.executeMatMul`.
* Loss + per-row gradient come from the verified `MSEOp`:
  - `Hesper.Training.MSE.executeMSEForward`  (mean of (pred − y)²)
  - `Hesper.Training.MSE.executeMSEBackward` (dPred = (2/N)(pred − y))
  This is the same pattern as `Hesper.Training.Loss.CrossEntropy`:
  a single `Differentiable` instance carries the CPU spec, two GPU
  kernels carry the device implementation, and `execute*` wrappers
  hide the binding setup.
* `gradMatmulKernel` then projects `dPred` through `Xᵀ` to get the
  parameter gradient `dw`, and `sgdUpdateKernel` does `w -= lr · dw`.

Loop is in Lean; kernel dispatches are persistent (buffers allocated
once, no host↔device traffic per iter except a single scalar loss
readback every 100 iters).
-/

open Hesper
open Hesper.Data
open Hesper.WGSL
open Hesper.WebGPU

private def oceanProximityCode : String → Float
  | "ISLAND"      => 0.0
  | "NEAR OCEAN"  => 1.0
  | "NEAR BAY"    => 2.0
  | "<1H OCEAN"   => 3.0
  | "INLAND"      => 4.0
  | _             => -1.0

/-- Compute `dw[j] = Σᵢ dPred[i] · X[i, j]` for `j ∈ [0, nParams)`.
    One thread per `j`, runtime loop over rows.  This is `Xᵀ @ res`
    expressed in the natural per-column form, sidestepping the
    naive-matmul row/col decomposition that drops the tail output
    when `N` isn't a multiple of the workgroup size. -/
def gradMatmulKernel (nRows nParams : Nat) : Monad.ShaderM Unit := do
  let gid ← Monad.ShaderM.globalId
  let j := Exp.vec3X gid
  let _x    ← Monad.ShaderM.declareInputBuffer "x"     (.array (.scalar .f32) (nRows * nParams))
  let _dp   ← Monad.ShaderM.declareInputBuffer "dPred" (.array (.scalar .f32) nRows)
  let _dw   ← Monad.ShaderM.declareOutputBuffer "dw"   (.array (.scalar .f32) nParams)
  Monad.ShaderM.if_ (Exp.lt j (Exp.litU32 nParams)) (do
    let (sumName, sum) ← Monad.ShaderM.varRef (.scalar .f32) (Exp.litF32 0.0)
    Monad.ShaderM.loop (Exp.litU32 0) (Exp.litU32 nRows) (Exp.litU32 1) fun i => do
      let xIdx := Exp.add (Exp.mul i (Exp.litU32 nParams)) j
      let d ← Monad.ShaderM.readBuffer (ty := .scalar .f32) (n := nRows)           "dPred" i
      let x ← Monad.ShaderM.readBuffer (ty := .scalar .f32) (n := nRows * nParams) "x"     xIdx
      Monad.ShaderM.assign sumName (Exp.add sum (Exp.mul d x))
    Monad.ShaderM.writeBuffer (ty := .scalar .f32) "dw" j sum
  ) (pure ())

/-- SGD update: `w[j] -= (lr * 2 / nRows) * dw[j]`. -/
def sgdUpdateKernel (nParams : Nat) (scale : Float) : Monad.ShaderM Unit := do
  let gid ← Monad.ShaderM.globalId
  let j := Exp.vec3X gid
  let _w  ← Monad.ShaderM.declareStorageBuffer "w"
              (.array (.scalar .f32) nParams) .readWrite
  let _dw ← Monad.ShaderM.declareInputBuffer "dw" (.array (.scalar .f32) nParams)
  Monad.ShaderM.if_ (Exp.lt j (Exp.litU32 nParams)) (do
    let wj  ← Monad.ShaderM.readBuffer (ty := .scalar .f32) (n := nParams) "w"  j
    let dwj ← Monad.ShaderM.readBuffer (ty := .scalar .f32) (n := nParams) "dw" j
    Monad.ShaderM.writeBuffer (ty := .scalar .f32) "w" j
      (Exp.sub wj (Exp.mul (Exp.litF32 scale) dwj))
  ) (pure ())

/-- Append a `1.0` bias column to each row of a row-major `[nRows × nFeat]`
    array, returning the augmented `[nRows × (nFeat+1)]` array. -/
private def appendBiasColumn (xs : Array Float) (nRows nFeat : Nat) : Array Float := Id.run do
  let mut out : Array Float := Array.mkEmpty (nRows * (nFeat + 1))
  for i in [0:nRows] do
    for j in [0:nFeat] do
      out := out.push xs[i * nFeat + j]!
    out := out.push 1.0
  return out

def main : IO Unit := do
  -- ── Data prep (identical to the CPU example) ─────────────────────
  let path : System.FilePath := "Examples/Tutorial/data/housing.csv"
  IO.println s!"Reading {path}..."
  let df ← DataFrame.readCsv path
  let meanBeds := df.meanMaybe "total_bedrooms"
  let cleaned :=
    df
      |>.impute "total_bedrooms" meanBeds
      |>.derive "ocean_proximity_code" (fun row =>
        match row[df.colIndex! "ocean_proximity"]! with
        | .text s => .f64 (oceanProximityCode s)
        | _       => .f64 (-1.0))
      |>.deriveFloat "rooms_per_household" #["total_rooms", "households"]
        (fun xs => xs[0]! / xs[1]!)
  let (train, _test) := cleaned.randomSplit (seed := 42) 0.8
  let trainFeaturesDF :=
    train.exclude #["ocean_proximity", "median_house_value"]
         |>.normalizeFeatures
  let (xFlat, nRows, nFeat) := trainFeaturesDF.toTensor
  let yArr : Array Float :=
    (train.column "median_house_value").filterMap Hesper.Data.Value.toFloat?
  let nParams := nFeat + 1                          -- weights + bias column
  let xAug := appendBiasColumn xFlat nRows nFeat
  IO.println s!"Training (GPU): {nRows} rows × {nFeat} features (+1 bias col, total {nParams} params)"

  -- ── GPU init ─────────────────────────────────────────────────────
  let inst ← Hesper.init
  let device ← Hesper.WebGPU.getDevice inst

  -- ── Persistent buffers ───────────────────────────────────────────
  let mkBuf := fun (n : Nat) =>
    Hesper.WebGPU.createBuffer device
      { size := (n * 4).toUSize
        usage := [.storage, .copyDst, .copySrc]
        mappedAtCreation := false }

  let xBuf     ← mkBuf (nRows * nParams)
  let yBuf     ← mkBuf nRows
  let wBuf     ← mkBuf nParams
  let predBuf  ← mkBuf nRows
  let dPredBuf ← mkBuf nRows
  let dwBuf    ← mkBuf nParams
  let lossBuf  ← mkBuf 1

  -- Upload X, y, w₀ (= zeros) once.
  let upload (buf : Hesper.WebGPU.Buffer) (xs : Array Float) : IO Unit := do
    let bytes ← Hesper.Basic.floatArrayToBytes xs
    Hesper.WebGPU.writeBuffer device buf 0 bytes
  upload xBuf xAug
  upload yBuf yArr
  upload wBuf (Array.replicate nParams 0.0)

  -- ── Hyperparameters (same as Sabela / CPU example) ───────────────
  -- The `2/N` factor lives inside `MSEOp.backward` (`executeMSEBackward`
  -- writes `(2/N)(pred − y)`), so the SGD step here is the bare `lr`.
  let lr     : Float := 0.1
  let nIters : Nat   := 1001

  -- ── Matmul configs (one-off allocation; reused every iter) ───────
  let fwdCfg : Hesper.WGSL.MatMul.Config :=        -- pred = X @ w
    { M := nRows, N := 1, K := nParams }

  IO.println "Starting training..."
  for it in [0:nIters] do
    -- 1. Forward: pred = X @ w.
    Hesper.WGSL.MatMul.executeMatMul device xBuf wBuf predBuf fwdCfg

    -- 2. (Optional) MSE forward — scalar loss readback every 100 iters.
    --    Uses the verified `MSEOp.forward` kernel.
    if it % 100 == 0 then
      Hesper.Training.MSE.executeMSEForward device predBuf yBuf lossBuf nRows
      let lossBytes ← Hesper.WebGPU.mapBufferRead device lossBuf 0 4
      Hesper.WebGPU.unmapBuffer lossBuf
      let lossArr ← Hesper.Basic.bytesToFloatArray lossBytes
      IO.println s!"Iteration: {it} | Loss: {lossArr[0]!}"

    -- 3. MSE backward: dPred[i] = (2/N)(pred[i] − y[i]).
    --    Verified-op pair with the forward above.
    Hesper.Training.MSE.executeMSEBackward device predBuf yBuf dPredBuf nRows

    -- 4. dw[j] = Σᵢ dPred[i] · X[i, j]  via per-column custom kernel.
    --    The naive matmul drops the last output element when N is not
    --    a multiple of the workgroup tail; this kernel parallelises on
    --    `j` and guards the only write with a bounds check.
    let gradBufs := [("x", xBuf), ("dPred", dPredBuf), ("dw", dwBuf)]
    GPUBackend.execute device (gradMatmulKernel nRows nParams) gradBufs
      (Hesper.ExecConfig.dispatch1D nParams 256)

    -- 5. SGD update: w -= lr · dw   (the 2/N factor is already in dw).
    let sgdBufs := [("w", wBuf), ("dw", dwBuf)]
    GPUBackend.execute device (sgdUpdateKernel nParams lr) sgdBufs
      (Hesper.ExecConfig.dispatch1D nParams 256)

  -- ── Final read-back ──────────────────────────────────────────────
  IO.println ""
  IO.println "Trained weights (last is the bias):"
  let wBytes ← Hesper.WebGPU.mapBufferRead device wBuf 0 ((nParams * 4).toUSize)
  Hesper.WebGPU.unmapBuffer wBuf
  let wFinal ← Hesper.Basic.bytesToFloatArray wBytes
  for j in [0:nParams] do
    IO.println s!"  w[{j}] = {wFinal[j]!}"
