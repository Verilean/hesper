import Hesper.Data.DataFrame
import Hesper.Optimizer.SGD
import Hesper.AD.Reverse
import Hesper.AD.ScalarInstances

/-!
# Ch11 California Housing — runnable example

End-to-end DataFrame example.  Reads the housing CSV, cleans + imputes,
splits, prints first rows + dimensions.

Notebook variant lives in `docs/tutorial/md/Ch11_DataAnalysis.md`.  The
shared `Hesper.Data.DataFrame` API and the data-cleaning policy follow
[DataHaskell's `dataframe`](https://github.com/mchav/dataframe) and
[Sabela's port](https://github.com/DataHaskell/sabela); see the
in-module Acknowledgements in `Hesper/Data/DataFrame.lean`.
-/

open Hesper.Data
open Hesper.Data.DataFrame

/-- Numeric encoding for the `ocean_proximity` text column.  Same
    mapping Sabela uses. -/
private def oceanProximityCode : String → Float
  | "ISLAND"      => 0.0
  | "NEAR OCEAN"  => 1.0
  | "NEAR BAY"    => 2.0
  | "<1H OCEAN"   => 3.0
  | "INLAND"      => 4.0
  | _             => -1.0

def main : IO Unit := do
  let path : System.FilePath := "Examples/Tutorial/data/housing.csv"
  IO.println s!"Reading {path}..."
  let df ← DataFrame.readCsv path
  let (nRows, nCols) := df.dimensions
  IO.println s!"Loaded: {nRows} rows × {nCols} columns"
  IO.println ""
  IO.println "=== first 5 rows (raw) ==="
  IO.println (df.toMarkdownHead 5)
  IO.println ""

  -- Impute missing total_bedrooms with the column mean.
  let meanBeds := df.meanMaybe "total_bedrooms"
  IO.println s!"mean(total_bedrooms) = {meanBeds}  (will impute missing cells)"
  let cleaned :=
    df
      |>.impute "total_bedrooms" meanBeds
      |>.derive "ocean_proximity_code" (fun row =>
        match row[df.colIndex! "ocean_proximity"]! with
        | .text s => .f64 (oceanProximityCode s)
        | _       => .f64 (-1.0))
      |>.deriveFloat "rooms_per_household" #["total_rooms", "households"]
        (fun xs => xs[0]! / xs[1]!)

  IO.println ""
  IO.println "=== first 5 rows (cleaned + derived) ==="
  IO.println (cleaned.toMarkdownHead 5)

  let (train, test) := cleaned.randomSplit (seed := 42) 0.8
  IO.println ""
  IO.println s!"train dims = {train.dimensions}, test dims = {test.dimensions}"

  -- For sanity, run the normaliser and confirm the numeric range.
  let normalised :=
    train.exclude #["ocean_proximity"]
         |>.normalizeFeatures
  let mn := normalised.minColumn "median_house_value"
  let mx := normalised.maxColumn "median_house_value"
  IO.println s!"after normalize: median_house_value range = [{mn}, {mx}]"

  -- Demonstrate the GPU-ready tensor export shape (numeric cols only).
  let (xs, r, c) := normalised.exclude #["median_house_value"] |>.toTensor
  IO.println s!"feature tensor: {r} rows × {c} cols  ({xs.size} floats)"

  -- ─── Training: linear regression with hesper's SGD ───────────────
  -- Same recipe as Sabela: normalise features (X) only, keep the
  -- target (median_house_value) on its original scale so the
  -- printed loss matches the Hasktorch original (Iteration 100
  -- ≈ 8.5e9, Iteration 1000 ≈ 5.4e9).
  let trainFeaturesDF :=
    train.exclude #["ocean_proximity", "median_house_value"]
         |>.normalizeFeatures
  let (xFlat, nRows, nCols) := trainFeaturesDF.toTensor
  let yArr : Array Float :=
    (train.column "median_house_value").filterMap Hesper.Data.Value.toFloat?
  IO.println ""
  IO.println s!"Training linear regression: {nRows} rows × {nCols} features"

  -- Linear model used only for displaying predictions at the end.
  let predict (w : Array Float) (b : Float) (x : Array Float) : Float := Id.run do
    let mut acc := b
    for j in [0:nCols] do
      acc := acc + w[j]! * x[j]!
    return acc

  -- ── AD-based loss: build the tape once per iter ──────────────────
  --
  -- Parameter packing convention for `gradNWith`:
  --   inputs[0 .. nCols-1] = w[0..nCols-1]
  --   inputs[nCols]        = b
  --
  -- The function constructs the *whole batch* loss
  --     L(w, b) = (1/N) Σᵢ (Σⱼ x[i,j] * w[j] + b - y[i])²
  -- using `Hesper.AD.Reverse` Dual numbers.  `backprop` then walks the
  -- tape in reverse to recover ∂L/∂wⱼ and ∂L/∂b in one pass.
  --
  -- Why rebuild the tape every iteration?  `lift2` stores *local
  -- gradients* at tape-build time (evaluated against the current
  -- primal of each operand).  Once `w` updates, those stored
  -- gradients are stale, so the tape must be reconstructed.  This is
  -- the same behaviour as PyTorch / JAX autograd.
  let lossFn : Hesper.AD.Reverse.ADContext → Array Hesper.AD.Reverse.Dual
             → Hesper.AD.Reverse.ADContext × Hesper.AD.Reverse.Dual :=
    fun ctx params => Id.run do
      let mut ctx := ctx
      let bDual := params[nCols]!
      -- Sum (pred[i] - y[i])² over all rows.
      let mut acc : Hesper.AD.Reverse.Dual := .const 0.0
      for i in [0:nRows] do
        -- pred[i] = b + Σⱼ x[i,j] * w[j]
        let mut p : Hesper.AD.Reverse.Dual := bDual
        for j in [0:nCols] do
          let xij : Hesper.AD.Reverse.Dual := .const xFlat[i * nCols + j]!
          let (ctx', wj_x) := ctx.mul params[j]! xij
          ctx := ctx'
          let (ctx', p') := ctx.add p wj_x
          ctx := ctx'
          p := p'
        -- (pred - y[i])² via the verified `SquaredErrorOp` — single
        -- tape node whose local gradient comes from the op's
        -- `Differentiable.backward`, not from a hand-unrolled
        -- `sub`+`pow` chain.
        let yi : Hesper.AD.Reverse.Dual := .const yArr[i]!
        let (ctx', r2) := ctx.liftBinaryVerified Hesper.AD.SquaredErrorOp.mk p yi
        ctx := ctx'
        let (ctx', acc') := ctx.add acc r2
        ctx := ctx'
        acc := acc'
      -- Mean
      let nInv : Hesper.AD.Reverse.Dual := .const (1.0 / nRows.toFloat)
      let (ctx', loss) := ctx.mul acc nInv
      ctx := ctx'
      return (ctx, loss)

  -- Initialise parameters and a single-slot SGD state.
  let mut w : Array Float := Array.replicate nCols 0.0
  let mut b : Float := 0.0
  -- Same hyperparameters as Sabela's `runStep state GD loss 0.1`.
  let cfg : Hesper.Optimizer.SGD.SGDConfig :=
    Hesper.Optimizer.SGD.SGDConfig.default
      |>.withLearningRate 0.1
  let mut sgdState := Hesper.Optimizer.SGD.SGDState.init #[nCols]

  -- Training loop — mirrors Sabela's `foldLoop init 1_000 ...`,
  -- but every gradient is derived from the AD tape rather than the
  -- closed-form MSE expression.
  for it in [0:1001] do
    let inputs : Array Float := w.push b
    let (lossVal, grads) := Hesper.AD.Reverse.gradNWith lossFn inputs
    if it % 100 == 0 then
      IO.println s!"Iteration: {it} | Loss: {lossVal}"
    let dw : Array Float := grads.extract 0 nCols
    let db : Float := grads[nCols]!
    let (wNew, sgdNew) :=
      Hesper.Optimizer.SGD.step cfg #[w] #[dw] sgdState
    w := wNew[0]!
    b := b - cfg.learningRate * db
    sgdState := sgdNew

  -- Show first 10 predictions vs ground truth on the test set.
  IO.println ""
  IO.println "=== predictions vs truth (test set, first 10) ==="
  let testFeaturesDF :=
    test.exclude #["ocean_proximity", "median_house_value"]
        |>.normalizeFeatures
  let (xTestFlat, nTest, _) := testFeaturesDF.toTensor
  let yTest : Array Float :=
    (test.column "median_house_value").filterMap Hesper.Data.Value.toFloat?
  for i in [0:Nat.min 10 nTest] do
    let xRow := xTestFlat.extract (i * nCols) ((i + 1) * nCols)
    let p := predict w b xRow
    IO.println s!"  truth = {yTest[i]!}  predicted = {p}"
