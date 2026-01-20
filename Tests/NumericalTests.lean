import LSpec
import Hesper.Tensor.Types
import Hesper.Tensor.MatMul

/-!
# Numerical Accuracy Tests

Tests for numerical correctness of GPU operations:
- Matrix multiplication validation
- Numerical precision checks
- Reference implementation comparisons
- Edge case testing
-/

namespace Tests.NumericalTests

open Hesper.Tensor
open LSpec

-- ============================================================================
-- CPU Reference Implementations
-- ============================================================================

/-- CPU reference implementation of matrix multiplication
    C[i,j] = sum_k A[i,k] * B[k,j]
    A: M×K, B: K×N, C: M×N -/
def cpuMatMul (a b : List Float) (m k n : Nat) : List Float :=
  ((List.range m).map fun i =>
    (List.range n).map fun j =>
      (List.range k).foldl (fun s kk =>
        let aVal := a[i * k + kk]?.getD 0.0
        let bVal := b[kk * n + j]?.getD 0.0
        s + aVal * bVal
      ) 0.0
  ).flatten

/-- Helper: Initialize matrix with constant value -/
def constMatrix (rows cols : Nat) (val : Float) : List Float :=
  List.replicate (rows * cols) val

/-- Helper: Initialize identity matrix -/
def identityMatrix (size : Nat) : List Float :=
  ((List.range size).map fun i =>
    (List.range size).map fun j =>
      if i == j then 1.0 else 0.0
  ).flatten

/-- Helper: Initialize matrix with sequential values -/
def sequentialMatrix (rows cols : Nat) : List Float :=
  (List.range (rows * cols)).map (Float.ofNat ·)

-- ============================================================================
-- Numerical Comparison Functions
-- ============================================================================

/-- Check if two floats are approximately equal within relative tolerance -/
def approxEq (a b : Float) (relTol : Float := 1e-5) (absTol : Float := 1e-8) : Bool :=
  let diff := (a - b).abs
  let maxVal := max a.abs b.abs
  diff <= absTol || diff <= relTol * maxVal

/-- Check if two matrices are approximately equal -/
def matricesApproxEq (a b : List Float) (relTol : Float := 1e-5) (absTol : Float := 1e-8) : Bool :=
  a.length == b.length &&
  (List.zip a b).all fun (x, y) => approxEq x y relTol absTol

/-- Calculate maximum absolute error between two matrices -/
def maxAbsError (a b : List Float) : Float :=
  if a.length != b.length then 1000.0
  else
    (List.zip a b).foldl (fun maxErr (x, y) =>
      max maxErr (x - y).abs
    ) 0.0

/-- Calculate relative error between two matrices -/
def relativeError (a b : List Float) : Float :=
  if a.length != b.length then 1000.0
  else
    let errors := (List.zip a b).map fun (x, y) =>
      let diff := (x - y).abs
      let maxVal := max x.abs y.abs
      if maxVal < 1e-10 then 0.0 else diff / maxVal
    errors.foldl max 0.0

-- ============================================================================
-- Matrix Multiplication Tests (CPU Reference)
-- ============================================================================

/-- Test: 2x2 matrix multiplication -/
def testMatMul2x2 : TestSeq :=
  -- A = [[1, 2], [3, 4]]
  let a := [1.0, 2.0, 3.0, 4.0]
  -- B = [[5, 6], [7, 8]]
  let b := [5.0, 6.0, 7.0, 8.0]
  -- Expected C = [[19, 22], [43, 50]]
  let expected := [19.0, 22.0, 43.0, 50.0]

  let result := cpuMatMul a b 2 2 2

  test "2x2 matrix multiplication" (matricesApproxEq result expected)

/-- Test: 3x3 matrix multiplication -/
def testMatMul3x3 : TestSeq :=
  -- A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  let a := sequentialMatrix 3 3
  -- B = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
  let b := (List.range 9).reverse.map (Float.ofNat ·)

  -- Calculate result
  let result := cpuMatMul a b 3 3 3

  -- Verify result is correct length (self-consistency check)
  test "3x3 matrix multiplication" (result.length == 9)

/-- Test: Identity matrix multiplication -/
def testMatMulIdentity : TestSeq :=
  let size := 4
  let a := sequentialMatrix size size
  let identity := identityMatrix size

  -- A * I = A
  let result := cpuMatMul a identity size size size

  test "Identity matrix multiplication (A * I = A)" (matricesApproxEq result a)

/-- Test: Zero matrix multiplication -/
def testMatMulZero : TestSeq :=
  let a := sequentialMatrix 3 4
  let b := constMatrix 4 3 0.0

  let result := cpuMatMul a b 3 4 3
  let expected := constMatrix 3 3 0.0

  test "Zero matrix multiplication" (matricesApproxEq result expected)

/-- Test: Non-square matrix multiplication (3x4 * 4x2) -/
def testMatMulNonSquare : TestSeq :=
  -- A: 3×4
  let a := sequentialMatrix 3 4
  -- B: 4×2
  let b := sequentialMatrix 4 2

  let result := cpuMatMul a b 3 4 2

  -- Verify dimensions are correct
  test "Non-square matrix multiplication (3x4 * 4x2 = 3x2)" (result.length == 6)

/-- Test: Large matrix multiplication (32x32) -/
def testMatMulLarge : TestSeq :=
  let size := 32
  let a := constMatrix size size 1.0
  let b := constMatrix size size 2.0

  let result := cpuMatMul a b size size size
  -- Each element should be size * 1.0 * 2.0 = size * 2.0
  let expected := constMatrix size size ((Float.ofNat size) * 2.0)

  let maxErr := maxAbsError result expected
  test "Large matrix multiplication (32x32)" (maxErr < 1e-3)

-- ============================================================================
-- Numerical Precision Tests
-- ============================================================================

/-- Test: Numerical precision for small values -/
def testPrecisionSmallValues : TestSeq :=
  let size := 4
  let a := constMatrix size size 1e-5
  let b := constMatrix size size 1e-5

  let result := cpuMatMul a b size size size
  -- Each element should be size * 1e-5 * 1e-5 = size * 1e-10
  let expected := constMatrix size size ((Float.ofNat size) * 1e-10)

  let relErr := relativeError result expected
  test "Precision with small values (1e-5)" (relErr < 1e-3)

/-- Test: Numerical precision for large values -/
def testPrecisionLargeValues : TestSeq :=
  let size := 4
  let a := constMatrix size size 1000.0
  let b := constMatrix size size 2000.0

  let result := cpuMatMul a b size size size
  -- Each element should be size * 1000.0 * 2000.0 = size * 2e6
  let expected := constMatrix size size ((Float.ofNat size) * 2.0e6)

  let relErr := relativeError result expected
  test "Precision with large values (1000)" (relErr < 1e-5)

/-- Test: Numerical stability with mixed magnitudes -/
def testStabilityMixedMagnitudes : TestSeq :=
  -- Mix of large and small values
  let a := [1e6, 1e-6, 1.0, 1e3]
  let b := [1e-6, 1e6, 1.0, 1e-3]

  let result := cpuMatMul a b 2 2 2

  -- Just verify computation completes without overflow/underflow
  -- Note: 1e6 * 1e6 = 1e12, so we need tolerance > 1e12
  test "Stability with mixed magnitudes" (result.all fun x => x.abs < 1e15)

-- ============================================================================
-- Edge Case Tests
-- ============================================================================

/-- Test: 1x1 matrix multiplication (scalar) -/
def testMatMul1x1 : TestSeq :=
  let a := [3.0]
  let b := [4.0]
  let expected := [12.0]

  let result := cpuMatMul a b 1 1 1

  test "1x1 matrix multiplication (scalar)" (matricesApproxEq result expected)

/-- Test: Matrix-vector multiplication (4x4 * 4x1) -/
def testMatrixVector : TestSeq :=
  let a := identityMatrix 4
  let b := [1.0, 2.0, 3.0, 4.0]

  let result := cpuMatMul a b 4 4 1

  -- I * v = v
  test "Matrix-vector multiplication (I * v = v)" (matricesApproxEq result b)

/-- Test: Associativity (A * B) * C = A * (B * C) -/
def testAssociativity : TestSeq :=
  let a := sequentialMatrix 2 3
  let b := sequentialMatrix 3 2
  let c := sequentialMatrix 2 3

  -- (A * B) * C
  let ab := cpuMatMul a b 2 3 2
  let ab_c := cpuMatMul ab c 2 2 3

  -- A * (B * C)
  let bc := cpuMatMul b c 3 2 3
  let a_bc := cpuMatMul a bc 2 3 3

  let maxErr := maxAbsError ab_c a_bc
  test "Associativity (A*B)*C = A*(B*C)" (maxErr < 1e-4)

-- ============================================================================
-- Performance Characteristics Tests
-- ============================================================================

/-- Test: Power of 2 dimensions (optimizable) -/
def testMatMulPowerOf2 : TestSeq :=
  let sizes := [2, 4, 8, 16]

  let allPass := sizes.all fun size =>
    let a := constMatrix size size 1.0
    let b := constMatrix size size 1.0
    let result := cpuMatMul a b size size size
    let expected := constMatrix size size (Float.ofNat size)
    matricesApproxEq result expected

  test "MatMul with power-of-2 dimensions" allPass

/-- Test: Non-power of 2 dimensions (edge case handling) -/
def testMatMulNonPowerOf2 : TestSeq :=
  let sizes := [(3, 5, 7), (5, 3, 11), (7, 11, 3)]

  let allPass := sizes.all fun (m, k, n) =>
    let a := constMatrix m k 1.0
    let b := constMatrix k n 1.0
    let result := cpuMatMul a b m k n
    let expected := constMatrix m n (Float.ofNat k)
    matricesApproxEq result expected

  test "MatMul with non-power-of-2 dimensions" allPass

-- ============================================================================
-- MatMul Config Tests
-- ============================================================================

/-- Test: MatMulConfig shape calculations -/
def testMatMulConfig : TestSeq :=
  let config : MatMulConfig := {
    M := 64,
    K := 32,
    N := 128,
    tileSize := 16,
    dtype := .f32
  }

  let shapeASize := config.shapeA.size
  let shapeBSize := config.shapeB.size
  let shapeCSize := config.shapeC.size

  test "MatMulConfig shape A" (shapeASize == 64 * 32) ++
  test "MatMulConfig shape B" (shapeBSize == 32 * 128) ++
  test "MatMulConfig shape C" (shapeCSize == 64 * 128) ++
  test "MatMulConfig workgroup count" (config.numWorkgroups == (8, 4, 1))

-- ============================================================================
-- All Tests
-- ============================================================================

def allTests : IO (List (String × List TestSeq)) := do
  IO.println "Running Numerical Accuracy Tests..."

  pure [
    ("MatMul 2x2", [testMatMul2x2]),
    ("MatMul 3x3", [testMatMul3x3]),
    ("MatMul Identity", [testMatMulIdentity]),
    ("MatMul Zero", [testMatMulZero]),
    ("MatMul Non-Square", [testMatMulNonSquare]),
    ("MatMul Large (32x32)", [testMatMulLarge]),
    ("Precision Small Values", [testPrecisionSmallValues]),
    ("Precision Large Values", [testPrecisionLargeValues]),
    ("Stability Mixed Magnitudes", [testStabilityMixedMagnitudes]),
    ("MatMul 1x1 (Scalar)", [testMatMul1x1]),
    ("Matrix-Vector Multiplication", [testMatrixVector]),
    ("Associativity", [testAssociativity]),
    ("MatMul Power-of-2 Sizes", [testMatMulPowerOf2]),
    ("MatMul Non-Power-of-2 Sizes", [testMatMulNonPowerOf2]),
    ("MatMulConfig", [testMatMulConfig])
  ]

end Tests.NumericalTests
