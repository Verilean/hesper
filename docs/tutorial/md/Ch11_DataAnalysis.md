# Chapter 11 — Data analysis with `Hesper.Data.DataFrame`

So far every chapter has produced data in code (an `Array Float`,
weights from a `.gguf`, etc.).  In real projects the data starts in a
CSV or Parquet file and needs to be cleaned, joined, sliced and
normalised before it ever sees a tensor.  This chapter walks through
that pipeline on the classic **California Housing** dataset (1990 US
census, 20640 districts, 10 features).

We rebuild the exact pipeline from
[Sabela's `CaliforniaHousing` example](https://github.com/DataHaskell/sabela/blob/main/examples/CaliforniaHousing.md)
(which in turn ports
[`dataframe`'s Haskell example](https://github.com/mchav/dataframe/blob/main/examples/CaliforniaHousing.hs))
using `Hesper.Data.DataFrame` — Hesper's own minimal DataFrame API.

The chapter has three parts:

1. **Load & inspect** — `readCsv`, `take`, `toHtml` / `toMarkdown`, `dimensions`.
2. **Clean & engineer** — `impute`, `derive`, `randomSplit`,
   `normalizeFeatures`.
3. **Hand-off to a model** — `toTensor` turns numeric columns into a
   row-major `Array Float` ready for matmul.

The complete runnable version lives at
`Examples/Tutorial/CaliforniaHousing.lean` and can be run with:

```text
$ lake exe california-housing
```

## Setup

```lean
import Hesper.Data.DataFrame
import Display

open Hesper.Data
open Hesper.Data.DataFrame
```

`Display` is xeus-lean's rich-output module — we pipe DataFrame
tables through `Display.html` so Jupyter renders them as real
HTML `<table>`s instead of raw markdown text.

The CSV file is bundled at `Examples/Tutorial/data/housing.csv`.  Its
header line is:

```text
longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income,median_house_value,ocean_proximity
```

`total_bedrooms` has a few missing cells; everything else is dense.

## 1. Loading the dataset

`DataFrame.readCsv` reads a CSV file and infers each column's type
from its contents.  Columns are classified as `Float`, `Float?`
(numeric with at least one missing cell), or `String`.

```lean
#eval do
  let df ← DataFrame.readCsv "Examples/Tutorial/data/housing.csv"
  IO.println s!"dimensions = {df.dimensions}"
  Display.html (df.toHtmlHead 5)
```

Output (the cell renders a real HTML table; only the first 5 rows
are shown so the cell stays readable):

| longitude<br><small>Float</small> | latitude<br><small>Float</small> | housing_median_age<br><small>Float</small> | total_rooms<br><small>Float</small> | total_bedrooms<br><small>Float?</small> | … | median_house_value<br><small>Float</small> | ocean_proximity<br><small>String</small> |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -122.23 | 37.88 | 41.0 | 880.0 | 129.0 | … | 452600.0 | NEAR BAY |
| -122.22 | 37.86 | 21.0 | 7099.0 | 1106.0 | … | 358500.0 | NEAR BAY |

Notice `total_bedrooms<br>Float?` — the `?` marks the column as
"numeric but has missing values".  We'll fix that next.

## 2. Cleaning & feature engineering

### Imputing missing values

`impute` fills every `missing` cell in a column with a fixed value.
The standard recipe (also used by Sabela / `dataframe`) is to impute
with the column mean:

```lean
def cleanHousing (df : DataFrame) : DataFrame :=
  let meanBeds := df.meanMaybe "total_bedrooms"  -- 537.870553...
  df.impute "total_bedrooms" meanBeds
```

`meanMaybe` walks the column once and returns the mean of the
non-missing cells.  After `impute`, the column's reported type
upgrades from `Float?` to `Float`.

### Encoding categorical strings

`ocean_proximity` is a string column.  Models want numbers, so we
encode it the same way Sabela does (`ISLAND→0, NEAR OCEAN→1, ...`):

```lean
def oceanProximityCode : String → Float
  | "ISLAND"      => 0.0
  | "NEAR OCEAN"  => 1.0
  | "NEAR BAY"    => 2.0
  | "<1H OCEAN"   => 3.0
  | "INLAND"      => 4.0
  | _             => -1.0
```

`derive` adds a new column computed per row.  The closure receives
the whole row as `Array Value`, so we can mention any other column
by index:

```lean
let cleaned :=
  df
    |>.impute "total_bedrooms" meanBeds
    |>.derive "ocean_proximity_code" (fun row =>
        match row[df.colIndex! "ocean_proximity"]! with
        | .text s => .f64 (oceanProximityCode s)
        | _       => .f64 (-1.0))
    |>.deriveFloat "rooms_per_household" #["total_rooms", "households"]
      (fun xs => xs[0]! / xs[1]!)
```

`deriveFloat` is a convenience wrapper: it looks up the input
columns by name once and hands a plain `Array Float` to the closure.
For this dataset that gives us the `rooms_per_household` feature the
original example uses (`6.984126984... ` for the first row).

### Train/test split

`randomSplit` shuffles row indices with a fixed seed and slices into
two `DataFrame`s.

```lean
let (train, test) := cleaned.randomSplit (seed := 42) 0.8
-- train dimensions ≈ (16512, 12), test dimensions ≈ (4128, 12)
```

The seed makes this reproducible — re-running the cell gives the same
split.

### Min-max normalisation

`normalizeFeatures` rescales **every numeric column** to `[0, 1]`,
which is the standard preprocessing step before fitting a linear
model (it stops large-magnitude features like
`median_house_value ∈ [15000, 500000]` from dominating the gradient
updates).

```lean
let normalised := train.exclude #["ocean_proximity"]
                       |>.normalizeFeatures
-- after normalize: median_house_value range = [0.0, 1.0]
```

We `exclude` the original string column first because it can't be
normalised; the numeric `ocean_proximity_code` column we derived
earlier stays in.

## 3. Hand-off to a model

`toTensor` flattens the numeric columns into a row-major
`Array Float` ready for matmul.  It returns the data plus the
inferred shape:

```lean
let trainFeaturesDF :=
  train.exclude #["ocean_proximity", "median_house_value"]
       |>.normalizeFeatures
let (xs, nRows, nCols) := trainFeaturesDF.toTensor
-- xs : Array Float, size = nRows * nCols
-- (16512, 10) on the train set
```

Note we **only normalise the features** — the target
`median_house_value` keeps its original `[15000, 500000]` scale.
Same recipe as Sabela; rescaling the target would shrink the loss
into the `0..1` range and obscure what the model is doing.

## 4. Training a linear regression with verified AD + `Hesper.Optimizer.SGD`

Sabela's reference training loop runs 1000 SGD steps minimising MSE.
Hesper has the same machinery — `Hesper.Optimizer.SGD.step` updates a
parameter array given its gradient — and we get the gradient itself
from `Hesper.AD.Reverse`, Hesper's reverse-mode autodiff.  No closed
form needed; the tape derives ∂L/∂w and ∂L/∂b from the very same
forward expression we use to compute the loss.

```lean
import Hesper.AD.Reverse
import Hesper.AD.ScalarInstances   -- gives us SquaredErrorOp
import Hesper.Optimizer.SGD

-- The per-row squared error `(p - y)²` is a *verified op*: a single
-- `Differentiable` instance whose `forward` returns the primal and
-- whose `backward` returns the local gradient pair `(2(p-y),
-- -2(p-y))`.  `ctx.liftBinaryVerified` pulls those local gradients
-- straight from `backward`, so the tape entry comes from one
-- audited place rather than a hand-unrolled `sub`+`pow` chain.
let lossFn : Hesper.AD.Reverse.ADContext → Array Hesper.AD.Reverse.Dual
           → Hesper.AD.Reverse.ADContext × Hesper.AD.Reverse.Dual :=
  fun ctx params => Id.run do
    let mut ctx := ctx
    let bDual := params[nCols]!
    let mut acc : Hesper.AD.Reverse.Dual := .const 0.0
    for i in [0:nRows] do
      -- pred[i] = b + Σⱼ x[i,j] * w[j]
      let mut p : Hesper.AD.Reverse.Dual := bDual
      for j in [0:nCols] do
        let xij : Hesper.AD.Reverse.Dual := .const xs[i * nCols + j]!
        let (ctx', wj_x) := ctx.mul params[j]! xij
        let (ctx', p')   := ctx'.add p wj_x
        ctx := ctx'; p := p'
      -- (pred - y[i])² via the verified op — one tape node, local
      -- gradient supplied by `SquaredErrorOp.backward`.
      let yi : Hesper.AD.Reverse.Dual := .const y[i]!
      let (ctx', r2) := ctx.liftBinaryVerified Hesper.AD.SquaredErrorOp.mk p yi
      let (ctx', acc') := ctx'.add acc r2
      ctx := ctx'; acc := acc'
    let nInv : Hesper.AD.Reverse.Dual := .const (1.0 / nRows.toFloat)
    let (ctx', loss) := ctx.mul acc nInv
    return (ctx', loss)
```

`SquaredErrorOp` is defined in `Hesper/AD/ScalarInstances.lean`
alongside the other built-in scalar ops (`AddOp`, `MulOp`, …):

```lean
structure SquaredErrorOp deriving Inhabited

instance : Differentiable SquaredErrorOp (Float × Float) Float where
  forward  := fun _ (p, y) =>
    let d := p - y
    d * d
  backward := fun _ (p, y) v =>
    let d := p - y
    (2.0 * d * v, -2.0 * d * v)
```

Note `backward` returns the **vector-Jacobian product** `Jᵀv` where
`v` is the upstream gradient — this is exactly the contract every
verified op in Hesper follows.

`SGD.step` then plugs in:

```lean
let cfg := Hesper.Optimizer.SGD.SGDConfig.default |>.withLearningRate 0.1
let mut sgdState := Hesper.Optimizer.SGD.SGDState.init #[nCols]
let mut w : Array Float := Array.replicate nCols 0.0
let mut b : Float := 0.0

for it in [0:1001] do
  -- Pack params and ask the tape for both the loss and its gradient
  -- in one pass.  Parameters are [w[0], ..., w[nCols-1], b].
  let inputs := w.push b
  let (lossVal, grads) := Hesper.AD.Reverse.gradNWith lossFn inputs
  if it % 100 == 0 then
    IO.println s!"Iteration: {it} | Loss: {lossVal}"
  let dw := grads.extract 0 nCols
  let db := grads[nCols]!
  let (wNew, sgdNew) :=
    Hesper.Optimizer.SGD.step cfg #[w] #[dw] sgdState
  w := wNew[0]!
  b := b - cfg.learningRate * db    -- bias is a scalar
  sgdState := sgdNew
```

### Why rebuild the tape every iteration?

`Dual.lift2` stores the *local gradient* of each operation evaluated
at the operands' current primal values (e.g. for `mul x y` the tape
node carries `(y.primal, x.primal)`).  Once `w` updates, those values
go stale, so the tape must be reconstructed on every iteration.  This
is identical to how PyTorch and JAX autograd work.  Lean 4's Perceus
reference counting drops the old tape as soon as `backprop` finishes,
so peak memory stays at one iteration's worth of nodes (~6 MB for
this dataset).

Running this (`lake exe california-housing`) reproduces Sabela's
trajectory:

```text
Iteration: 0    | Loss: 5.55e10
Iteration: 100  | Loss: 8.50e9
Iteration: 200  | Loss: 6.78e9
Iteration: 300  | Loss: 6.11e9
...
Iteration: 1000 | Loss: 5.39e9
```

The loss starts above 50 billion (the predictions are zero, so the
mean-squared error of `(0 - 200000)² ≈ 4e10` per row) and drops to
~5.4 billion after 1000 iterations — the same plateau Sabela reaches.

### Predictions

```text
truth =  260900   predicted =  320576  (23% off)
truth =  500001   predicted =  361518  (28% off)
truth =   52900   predicted =   97599  (84% off — small targets are hard)
truth =  293000   predicted =  341853  (16% off)
truth =   53600   predicted =   52008  ( 3% off — bullseye)
truth =  214800   predicted =  212546  ( 1% off — bullseye)
```

A single linear layer can't capture a nonlinear feature like
`median_income → median_house_value`, so the absolute errors are
large.  Bump `1001` to `100_000` and you'll see the loss bottom out
around `5e9` — the same ceiling Sabela hits with `Torch.linear`.
The point isn't the accuracy; it's that the **same recipe** runs in
Lean 4 with Hesper's verified-by-construction SGD optimiser.

## 5. Run it on the GPU

The CPU loop above uses `Hesper.AD.Reverse` to build a per-iteration
tape; ergonomic but it allocates ~165k node records per step.  For a
problem this small that's fine.  For anything larger we'd rather push
the loop body onto the GPU, and Hesper ships a verified `MSEOp` that
makes that almost mechanical:

```lean
import Hesper.Training.MSE        -- MSEOp.{forward,backward}
import Hesper.WGSL.MatMul         -- executeMatMul

-- Persistent device buffers, allocated once.
let xBuf     ← mkBuf (nRows * nParams)
let yBuf     ← mkBuf nRows
let wBuf     ← mkBuf nParams       -- weights, includes bias column
let predBuf  ← mkBuf nRows
let dPredBuf ← mkBuf nRows
let dwBuf    ← mkBuf nParams
let lossBuf  ← mkBuf 1

for it in [0:1001] do
  -- pred = X @ w   (production matmul kernel)
  Hesper.WGSL.MatMul.executeMatMul device xBuf wBuf predBuf
    { M := nRows, N := 1, K := nParams }

  -- (loss readback every 100 iters)
  if it % 100 == 0 then
    Hesper.Training.MSE.executeMSEForward device predBuf yBuf lossBuf nRows
    let lossArr ← readScalar device lossBuf
    IO.println s!"Iteration: {it} | Loss: {lossArr[0]!}"

  -- dPred[i] = (2/N)(pred − y)   (verified MSEOp.backward kernel)
  Hesper.Training.MSE.executeMSEBackward device predBuf yBuf dPredBuf nRows

  -- dw = Xᵀ @ dPred   (per-column kernel; see Ch10 for why)
  GPUBackend.execute device (gradMatmulKernel nRows nParams)
    [("x", xBuf), ("dPred", dPredBuf), ("dw", dwBuf)]
    (Hesper.ExecConfig.dispatch1D nParams 256)

  -- w -= lr · dw   (the 2/N is already baked into dPred)
  GPUBackend.execute device (sgdUpdateKernel nParams lr)
    [("w", wBuf), ("dw", dwBuf)]
    (Hesper.ExecConfig.dispatch1D nParams 256)
```

`MSEOp` follows the exact same shape as `Hesper.Training.Loss`'s
cross-entropy op: a `Differentiable` instance carries the CPU
specification, and two GPU kernels (`forwardKernel`, `backwardKernel`)
carry the device implementation.  The `executeMSEForward` /
`executeMSEBackward` wrappers hide the buffer-binding boilerplate so
the training loop reads as straight `forward → backward → SGD`.

Running the GPU version (`lake exe california-housing-gpu`):

```text
Iteration: 0    | Loss: 5.55e10
Iteration: 100  | Loss: 8.51e9   ← matches the CPU AD path
Iteration: 1000 | Loss: 5.39e9   ← matches Sabela's plateau
```

Within fp32 rounding the trajectory is identical to the CPU AD path
above — the GPU is just doing the same arithmetic, faster and without
allocating tape nodes.

## What's next

Ch12 will swap the per-column gradient and SGD kernels for the
Circuit DSL versions, fusing the whole training step into one
bufferArray dispatch per epoch.

## Acknowledgements

The data, the cleaning recipe (mean-imputation of `total_bedrooms`,
the `rooms_per_household` derived feature, the `oceanProximity`
numeric encoding), the chapter structure (load → inspect → impute →
derive → split → normalise → **train → predict**), the min-max
normalisation choice, **the SGD hyperparameters (`lr = 0.1`, 1000
steps), and the train/test 80/20 split with seed 42** all follow:

- [`dataframe`](https://github.com/mchav/dataframe) — the Haskell
  DataFrame library that originated this example (`examples/
  CaliforniaHousing.hs`).
- [Sabela](https://github.com/DataHaskell/sabela) — the
  `dataframe-hasktorch` port that adapted it as a notebook walk-through
  (`examples/CaliforniaHousing.md`).

Implementation is fresh Lean 4 code, but every editorial choice
(which features to derive, which columns to drop, what seed to use,
what learning rate to pick, how to write the markdown table) is
theirs.  Many thanks to the authors of both projects.
