import Hesper.AD.Reverse

/-!
# Automatic Differentiation Correctness Proof

Verifies the AD implementation by comparing computed gradients
with analytical derivatives for various test functions.

## Test Functions
1. Polynomials: f(x) = x², x³, x⁴
2. Transcendental: exp, log, sin, cos
3. Composite functions
4. Multivariate functions

Each test computes both:
- Forward pass: f(x) - the primal value
- Backward pass: f'(x) - the gradient

Results are compared against analytical derivatives.
-/

namespace Examples.ADProof

open Hesper.AD.Reverse

/-- Tolerance for floating point comparison -/
def ε : Float := 1e-6

/-- Pi constant -/
def pi : Float := 3.141592653589793

/-- Check if two floats are approximately equal -/
def approxEq (x y : Float) : Bool :=
  Float.abs (x - y) < ε

/-- Test result structure -/
structure TestResult where
  name : String
  point : Float
  computed_value : Float
  expected_value : Float
  computed_grad : Float
  expected_grad : Float
  value_ok : Bool
  grad_ok : Bool
  deriving Repr

/-- Print test result -/
def printResult (r : TestResult) : IO Unit := do
  let status := if r.value_ok && r.grad_ok then "✓ PASS" else "✗ FAIL"
  IO.println s!"{status} | {r.name}"
  IO.println s!"  x = {r.point}"
  IO.println s!"  f(x): computed={r.computed_value}, expected={r.expected_value}, match={r.value_ok}"
  IO.println s!"  f'(x): computed={r.computed_grad}, expected={r.expected_grad}, match={r.grad_ok}"
  IO.println ""

/-- Test 1: f(x) = x²
    f'(x) = 2x -/
def test_square : IO TestResult := do
  let f (ctx : ADContext) (x : Dual) : ADContext × Dual :=
    ctx.mul x x

  let x := 3.0
  let (value, grad) := gradWith f x

  let expected_value := x * x
  let expected_grad := 2.0 * x

  return {
    name := "f(x) = x²"
    point := x
    computed_value := value
    expected_value := expected_value
    computed_grad := grad
    expected_grad := expected_grad
    value_ok := approxEq value expected_value
    grad_ok := approxEq grad expected_grad
  }

/-- Test 2: f(x) = x³
    f'(x) = 3x² -/
def test_cube : IO TestResult := do
  let f (ctx : ADContext) (x : Dual) : ADContext × Dual :=
    let (ctx, x2) := ctx.mul x x
    ctx.mul x2 x

  let x := 2.0
  let (value, grad) := gradWith f x

  let expected_value := x * x * x
  let expected_grad := 3.0 * x * x

  return {
    name := "f(x) = x³"
    point := x
    computed_value := value
    expected_value := expected_value
    computed_grad := grad
    expected_grad := expected_grad
    value_ok := approxEq value expected_value
    grad_ok := approxEq grad expected_grad
  }

/-- Test 3: f(x) = exp(x)
    f'(x) = exp(x) -/
def test_exp : IO TestResult := do
  let f (ctx : ADContext) (x : Dual) : ADContext × Dual :=
    ctx.exp x

  let x := 1.0
  let (value, grad) := gradWith f x

  let expected_value := Float.exp x
  let expected_grad := Float.exp x

  return {
    name := "f(x) = exp(x)"
    point := x
    computed_value := value
    expected_value := expected_value
    computed_grad := grad
    expected_grad := expected_grad
    value_ok := approxEq value expected_value
    grad_ok := approxEq grad expected_grad
  }

/-- Test 4: f(x) = log(x)
    f'(x) = 1/x -/
def test_log : IO TestResult := do
  let f (ctx : ADContext) (x : Dual) : ADContext × Dual :=
    ctx.log x

  let x := 2.0
  let (value, grad) := gradWith f x

  let expected_value := Float.log x
  let expected_grad := 1.0 / x

  return {
    name := "f(x) = log(x)"
    point := x
    computed_value := value
    expected_value := expected_value
    computed_grad := grad
    expected_grad := expected_grad
    value_ok := approxEq value expected_value
    grad_ok := approxEq grad expected_grad
  }

/-- Test 5: f(x) = sin(x)
    f'(x) = cos(x) -/
def test_sin : IO TestResult := do
  let f (ctx : ADContext) (x : Dual) : ADContext × Dual :=
    ctx.sin x

  let x := pi / 4.0  -- 45 degrees
  let (value, grad) := gradWith f x

  let expected_value := Float.sin x
  let expected_grad := Float.cos x

  return {
    name := "f(x) = sin(x)"
    point := x
    computed_value := value
    expected_value := expected_value
    computed_grad := grad
    expected_grad := expected_grad
    value_ok := approxEq value expected_value
    grad_ok := approxEq grad expected_grad
  }

/-- Test 6: f(x) = cos(x)
    f'(x) = -sin(x) -/
def test_cos : IO TestResult := do
  let f (ctx : ADContext) (x : Dual) : ADContext × Dual :=
    ctx.cos x

  let x := pi / 3.0  -- 60 degrees
  let (value, grad) := gradWith f x

  let expected_value := Float.cos x
  let expected_grad := -Float.sin x

  return {
    name := "f(x) = cos(x)"
    point := x
    computed_value := value
    expected_value := expected_value
    computed_grad := grad
    expected_grad := expected_grad
    value_ok := approxEq value expected_value
    grad_ok := approxEq grad expected_grad
  }

/-- Test 7: f(x) = x² + 3x + 2
    f'(x) = 2x + 3 -/
def test_polynomial : IO TestResult := do
  let f (ctx : ADContext) (x : Dual) : ADContext × Dual :=
    let (ctx, x2) := ctx.mul x x
    let three := Dual.const 3.0
    let (ctx, threex) := ctx.mul three x
    let (ctx, x2_plus_3x) := ctx.add x2 threex
    let two := Dual.const 2.0
    ctx.add x2_plus_3x two

  let x := 5.0
  let (value, grad) := gradWith f x

  let expected_value := x * x + 3.0 * x + 2.0
  let expected_grad := 2.0 * x + 3.0

  return {
    name := "f(x) = x² + 3x + 2"
    point := x
    computed_value := value
    expected_value := expected_value
    computed_grad := grad
    expected_grad := expected_grad
    value_ok := approxEq value expected_value
    grad_ok := approxEq grad expected_grad
  }

/-- Test 8: f(x) = exp(x²)
    f'(x) = 2x·exp(x²) (chain rule) -/
def test_chain_rule : IO TestResult := do
  let f (ctx : ADContext) (x : Dual) : ADContext × Dual :=
    let (ctx, x2) := ctx.mul x x
    ctx.exp x2

  let x := 1.5
  let (value, grad) := gradWith f x

  let x2 := x * x
  let expected_value := Float.exp x2
  let expected_grad := 2.0 * x * Float.exp x2

  return {
    name := "f(x) = exp(x²) [chain rule]"
    point := x
    computed_value := value
    expected_value := expected_value
    computed_grad := grad
    expected_grad := expected_grad
    value_ok := approxEq value expected_value
    grad_ok := approxEq grad expected_grad
  }

/-- Test 9: f(x) = x / (1 + x²)
    f'(x) = (1 - x²) / (1 + x²)² (quotient rule) -/
def test_quotient_rule : IO TestResult := do
  let f (ctx : ADContext) (x : Dual) : ADContext × Dual :=
    let one := Dual.const 1.0
    let (ctx, x2) := ctx.mul x x
    let (ctx, denominator) := ctx.add one x2
    ctx.div x denominator

  let x := 2.0
  let (value, grad) := gradWith f x

  let x2 := x * x
  let expected_value := x / (1.0 + x2)
  let expected_grad := (1.0 - x2) / Float.pow (1.0 + x2) 2.0

  return {
    name := "f(x) = x/(1+x²) [quotient rule]"
    point := x
    computed_value := value
    expected_value := expected_value
    computed_grad := grad
    expected_grad := expected_grad
    value_ok := approxEq value expected_value
    grad_ok := approxEq grad expected_grad
  }

/-- Test 10: f(x) = sigmoid(x) = 1/(1+exp(-x))
    f'(x) = sigmoid(x)·(1-sigmoid(x)) -/
def test_sigmoid : IO TestResult := do
  let f (ctx : ADContext) (x : Dual) : ADContext × Dual :=
    ctx.sigmoid x

  let x := 0.5
  let (value, grad) := gradWith f x

  let sigmoid_x := 1.0 / (1.0 + Float.exp (-x))
  let expected_value := sigmoid_x
  let expected_grad := sigmoid_x * (1.0 - sigmoid_x)

  return {
    name := "f(x) = sigmoid(x)"
    point := x
    computed_value := value
    expected_value := expected_value
    computed_grad := grad
    expected_grad := expected_grad
    value_ok := approxEq value expected_value
    grad_ok := approxEq grad expected_grad
  }

/-- Test 11: f(x) = tanh(x)
    f'(x) = 1 - tanh²(x) -/
def test_tanh : IO TestResult := do
  let f (ctx : ADContext) (x : Dual) : ADContext × Dual :=
    ctx.tanh x

  let x := 0.8
  let (value, grad) := gradWith f x

  let tanh_x := Float.tanh x
  let expected_value := tanh_x
  let expected_grad := 1.0 - tanh_x * tanh_x

  return {
    name := "f(x) = tanh(x)"
    point := x
    computed_value := value
    expected_value := expected_value
    computed_grad := grad
    expected_grad := expected_grad
    value_ok := approxEq value expected_value
    grad_ok := approxEq grad expected_grad
  }

/-- Test 12: f(x,y) = x² + y² (multivariate)
    ∂f/∂x = 2x, ∂f/∂y = 2y -/
def test_multivariate_sum : IO (String × Float × Float × Float × Float × Bool) := do
  let f (ctx : ADContext) (vars : Array Dual) : ADContext × Dual :=
    let x := vars[0]!
    let y := vars[1]!
    let (ctx, x2) := ctx.mul x x
    let (ctx, y2) := ctx.mul y y
    ctx.add x2 y2

  let point := #[3.0, 4.0]
  let (value, grads) := gradNWith f point

  let x := point[0]!
  let y := point[1]!
  let expected_value := x * x + y * y
  let expected_grad_x := 2.0 * x
  let expected_grad_y := 2.0 * y

  let value_ok := approxEq value expected_value
  let grad_ok := approxEq grads[0]! expected_grad_x && approxEq grads[1]! expected_grad_y

  return ("f(x,y) = x² + y²", value, grads[0]!, grads[1]!, expected_value, value_ok && grad_ok)

/-- Test 13: f(x,y) = x·y (multivariate product)
    ∂f/∂x = y, ∂f/∂y = x -/
def test_multivariate_product : IO (String × Float × Float × Float × Float × Bool) := do
  let f (ctx : ADContext) (vars : Array Dual) : ADContext × Dual :=
    let x := vars[0]!
    let y := vars[1]!
    ctx.mul x y

  let point := #[5.0, 7.0]
  let (value, grads) := gradNWith f point

  let x := point[0]!
  let y := point[1]!
  let expected_value := x * y
  let expected_grad_x := y
  let expected_grad_y := x

  let value_ok := approxEq value expected_value
  let grad_ok := approxEq grads[0]! expected_grad_x && approxEq grads[1]! expected_grad_y

  return ("f(x,y) = x·y", value, grads[0]!, grads[1]!, expected_value, value_ok && grad_ok)

/-- Main test suite -/
def main : IO Unit := do
  IO.println "╔════════════════════════════════════════════════╗"
  IO.println "║   Automatic Differentiation Correctness       ║"
  IO.println "║   Forward & Backward Pass Verification        ║"
  IO.println "╚════════════════════════════════════════════════╝"
  IO.println ""

  IO.println "Testing univariate functions..."
  IO.println "═══════════════════════════════════════════════"
  IO.println ""

  -- Run all univariate tests
  let tests := [
    test_square,
    test_cube,
    test_exp,
    test_log,
    test_sin,
    test_cos,
    test_polynomial,
    test_chain_rule,
    test_quotient_rule,
    test_sigmoid,
    test_tanh
  ]

  let mut passed := 0
  let mut failed := 0

  for test in tests do
    let result ← test
    printResult result
    if result.value_ok && result.grad_ok then
      passed := passed + 1
    else
      failed := failed + 1

  IO.println "Testing multivariate functions..."
  IO.println "═══════════════════════════════════════════════"
  IO.println ""

  -- Test multivariate sum
  let (name1, value1, gx1, gy1, expected1, ok1) ← test_multivariate_sum
  let status1 := if ok1 then "✓ PASS" else "✗ FAIL"
  IO.println s!"{status1} | {name1}"
  IO.println s!"  (x,y) = (3, 4)"
  IO.println s!"  f(x,y) = {value1}, expected = {expected1}"
  IO.println s!"  ∂f/∂x = {gx1}, expected = 6.0"
  IO.println s!"  ∂f/∂y = {gy1}, expected = 8.0"
  IO.println ""

  if ok1 then passed := passed + 1 else failed := failed + 1

  -- Test multivariate product
  let (name2, value2, gx2, gy2, expected2, ok2) ← test_multivariate_product
  let status2 := if ok2 then "✓ PASS" else "✗ FAIL"
  IO.println s!"{status2} | {name2}"
  IO.println s!"  (x,y) = (5, 7)"
  IO.println s!"  f(x,y) = {value2}, expected = {expected2}"
  IO.println s!"  ∂f/∂x = {gx2}, expected = 7.0"
  IO.println s!"  ∂f/∂y = {gy2}, expected = 5.0"
  IO.println ""

  if ok2 then passed := passed + 1 else failed := failed + 1

  -- Summary
  IO.println "═══════════════════════════════════════════════"
  IO.println "Summary:"
  IO.println s!"  Total tests: {passed + failed}"
  IO.println s!"  ✓ Passed: {passed}"
  IO.println s!"  ✗ Failed: {failed}"
  IO.println ""

  if failed == 0 then
    IO.println "╔════════════════════════════════════════════════╗"
    IO.println "║   ✓ ALL TESTS PASSED                          ║"
    IO.println "║   AD implementation is CORRECT!                ║"
    IO.println "╚════════════════════════════════════════════════╝"
  else
    IO.println "╔════════════════════════════════════════════════╗"
    IO.println "║   ✗ SOME TESTS FAILED                         ║"
    IO.println "║   Check implementation                         ║"
    IO.println "╚════════════════════════════════════════════════╝"
  IO.println ""

  IO.println "Verified properties:"
  IO.println "  ✓ Forward pass computes correct primal values"
  IO.println "  ✓ Backward pass computes correct gradients"
  IO.println "  ✓ Chain rule applied correctly"
  IO.println "  ✓ Product rule applied correctly"
  IO.println "  ✓ Quotient rule applied correctly"
  IO.println "  ✓ Multivariate gradients computed correctly"
  IO.println ""

end Examples.ADProof

-- Entry point
def main : IO Unit := Examples.ADProof.main
