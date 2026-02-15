/-!
# Validation Metrics

Numerical comparison utilities for validating BitNet implementation against reference implementations.

## Purpose

As specified in CLAUDE.md, the critical validation step is:
> "This is the most critical part of the prompt: 'Compare calculation results.'"

This module provides metrics to compare outputs:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Cosine Similarity
- Max Absolute Difference
- Element-wise comparison

## Usage

```lean
-- Compare two float arrays
let mse := computeMSE reference_output hesper_output
let cosine := computeCosineSimilarity reference_output hesper_output

-- Check if arrays match within tolerance
let matches := arraysMatch reference_output hesper_output 1e-5
```

## References
- CLAUDE.md: Section 5 "Implementation Strategy: The Validation Loop"
-/

namespace Hesper.Validation.Metrics

/-! ## Basic Statistics -/

/-- Compute mean of an array -/
def mean (arr : Array Float) : Float :=
  if arr.isEmpty then
    0.0
  else
    let sum := arr.foldl (· + ·) 0.0
    sum / arr.size.toFloat

/-- Compute standard deviation -/
def stdDev (arr : Array Float) : Float :=
  if arr.isEmpty then
    0.0
  else
    let m := mean arr
    let variance := arr.foldl (fun acc x => acc + (x - m) * (x - m)) 0.0 / arr.size.toFloat
    Float.sqrt variance

/-! ## Error Metrics -/

/-- Mean Squared Error (MSE)

    MSE = (1/n) * Σ(y_pred - y_true)²

    Lower is better. MSE = 0 means perfect match.
-/
def computeMSE (reference : Array Float) (predicted : Array Float) : Float :=
  if reference.size != predicted.size then
    1.0e308  -- Invalid comparison (large value instead of inf)
  else if reference.isEmpty then
    0.0
  else
    let squaredErrors := Array.zipWith (fun r p =>
      let diff := p - r
      diff * diff) reference predicted
    mean squaredErrors

/-- Root Mean Squared Error (RMSE)

    RMSE = √MSE

    Same units as the original data.
-/
def computeRMSE (reference : Array Float) (predicted : Array Float) : Float :=
  Float.sqrt (computeMSE reference predicted)

/-- Mean Absolute Error (MAE)

    MAE = (1/n) * Σ|y_pred - y_true|

    More robust to outliers than MSE.
-/
def computeMAE (reference : Array Float) (predicted : Array Float) : Float :=
  if reference.size != predicted.size then
    1.0e308  -- Invalid comparison
  else if reference.isEmpty then
    0.0
  else
    let absoluteErrors := Array.zipWith (fun r p =>
      Float.abs (p - r)) reference predicted
    mean absoluteErrors

/-- Maximum Absolute Error

    Max|y_pred - y_true|

    Worst-case error.
-/
def computeMaxAbsError (reference : Array Float) (predicted : Array Float) : Float :=
  if reference.size != predicted.size then
    1.0e308  -- Invalid comparison
  else if reference.isEmpty then
    0.0
  else
    let absoluteErrors := Array.zipWith (fun r p =>
      Float.abs (p - r)) reference predicted
    absoluteErrors.foldl (fun acc x => if x > acc then x else acc) 0.0

/-! ## Similarity Metrics -/

/-- Dot product of two arrays -/
def dotProduct (a : Array Float) (b : Array Float) : Float :=
  if a.size != b.size then
    0.0
  else
    let products := Array.zipWith (· * ·) a b
    products.foldl (· + ·) 0.0

/-- L2 norm (Euclidean norm) of an array -/
def l2Norm (arr : Array Float) : Float :=
  Float.sqrt (dotProduct arr arr)

/-- Cosine Similarity

    cos_sim = (a · b) / (||a|| * ||b||)

    Range: [-1, 1]
    - 1.0: Identical direction
    - 0.0: Orthogonal
    - -1.0: Opposite direction

    For validation: expect > 0.999 for correct implementation
-/
def computeCosineSimilarity (reference : Array Float) (predicted : Array Float) : Float :=
  if reference.size != predicted.size then
    0.0
  else if reference.isEmpty then
    1.0  -- Empty arrays are considered identical
  else
    let dot := dotProduct reference predicted
    let norm_ref := l2Norm reference
    let norm_pred := l2Norm predicted

    if norm_ref == 0.0 || norm_pred == 0.0 then
      0.0
    else
      dot / (norm_ref * norm_pred)

/-- Relative error

    rel_error = |pred - ref| / |ref|

    Percentage-based error metric.
-/
def computeRelativeError (reference : Array Float) (predicted : Array Float) : Float :=
  if reference.size != predicted.size then
    1.0e308  -- Invalid comparison
  else if reference.isEmpty then
    0.0
  else
    let relativeErrors := Array.zipWith (fun r p =>
      if Float.abs r < 1e-10 then
        Float.abs (p - r)  -- Absolute error for near-zero values
      else
        Float.abs ((p - r) / r)) reference predicted
    mean relativeErrors

/-! ## Comparison -/

/-- Check if two arrays match within tolerance -/
def arraysMatch (reference : Array Float) (predicted : Array Float) (tolerance : Float) : Bool :=
  if reference.size != predicted.size then
    false
  else
    let maxError := computeMaxAbsError reference predicted
    maxError <= tolerance

/-- Count number of elements that differ beyond tolerance -/
def countMismatches (reference : Array Float) (predicted : Array Float) (tolerance : Float) : Nat :=
  if reference.size != predicted.size then
    reference.size + predicted.size
  else
    let errors := Array.zipWith (fun r p => Float.abs (p - r)) reference predicted
    errors.foldl (fun acc err => if err > tolerance then acc + 1 else acc) 0

/-! ## Reporting -/

structure ValidationReport where
  size : Nat
  mse : Float
  rmse : Float
  mae : Float
  maxAbsError : Float
  cosineSimilarity : Float
  relativeError : Float
  matchesWithinTolerance : Bool
  numMismatches : Nat
  deriving Repr, Inhabited

/-- Generate comprehensive validation report -/
def generateReport (reference : Array Float) (predicted : Array Float) (tolerance : Float := 1e-5) : ValidationReport :=
  {
    size := reference.size
    mse := computeMSE reference predicted
    rmse := computeRMSE reference predicted
    mae := computeMAE reference predicted
    maxAbsError := computeMaxAbsError reference predicted
    cosineSimilarity := computeCosineSimilarity reference predicted
    relativeError := computeRelativeError reference predicted
    matchesWithinTolerance := arraysMatch reference predicted tolerance
    numMismatches := countMismatches reference predicted tolerance
  }

/-- Print validation report -/
def printReport (report : ValidationReport) : IO Unit := do
  IO.println "═══════════════════════════════════════════════"
  IO.println "  Validation Report"
  IO.println "═══════════════════════════════════════════════"
  IO.println s!"Array size: {report.size}"
  IO.println ""
  IO.println "Error Metrics:"
  IO.println s!"  MSE:              {report.mse}"
  IO.println s!"  RMSE:             {report.rmse}"
  IO.println s!"  MAE:              {report.mae}"
  IO.println s!"  Max Abs Error:    {report.maxAbsError}"
  IO.println s!"  Relative Error:   {report.relativeError}"
  IO.println ""
  IO.println "Similarity Metrics:"
  IO.println s!"  Cosine Similarity: {report.cosineSimilarity}"
  IO.println ""
  IO.println "Match Status:"
  if report.matchesWithinTolerance then
    IO.println "  ✓ PASS - Arrays match within tolerance"
  else
    IO.println s!"  ✗ FAIL - {report.numMismatches} / {report.size} elements differ"
  IO.println "═══════════════════════════════════════════════"

/-- Print detailed comparison (first N elements) -/
def printDetailedComparison (reference : Array Float) (predicted : Array Float) (numElements : Nat := 10) : IO Unit := do
  IO.println "Detailed Element Comparison:"
  IO.println "  Idx | Reference      | Predicted      | Abs Error"
  IO.println "  ────┼────────────────┼────────────────┼──────────────"

  let n := min numElements (min reference.size predicted.size)
  for i in [0:n] do
    let ref := reference[i]!
    let pred := predicted[i]!
    let err := Float.abs (pred - ref)
    IO.println s!"  {i} | {ref} | {pred} | {err}"

end Hesper.Validation.Metrics
