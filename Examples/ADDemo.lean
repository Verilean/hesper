import Hesper.AD.Reverse
import Hesper.Optimizer.SGD
import Hesper.Optimizer.Adam

/-!
# Automatic Differentiation and Optimization Demo

Demonstrates reverse-mode AD and optimization algorithms (SGD, Adam).

## Examples
1. Basic gradient computation
2. Training with SGD
3. Training with Adam
4. Optimizer comparison

Usage:
  lake build ad-demo && ./.lake/build/bin/ad-demo
-/

namespace Examples.ADDemo

open Hesper.AD.Reverse
open Hesper.Optimizer.SGD (SGDConfig SGDState)
open Hesper.Optimizer.Adam (AdamConfig AdamState)

/-- Demo 1: Basic Automatic Differentiation -/
def demo1_basic_ad : IO Unit := do
  IO.println "═══════════════════════════════════════════════"
  IO.println "Demo 1: Basic Automatic Differentiation"
  IO.println "═══════════════════════════════════════════════"
  IO.println ""

  -- Example 1: Simple polynomial f(x) = x² + 2x + 1
  IO.println "Example 1: f(x) = x² + 2x + 1"
  let f1 (ctx : ADContext) (x : Dual) : ADContext × Dual :=
    let (ctx, x2) := ctx.mul x x            -- x²
    let two := Dual.const 2.0
    let (ctx, twox) := ctx.mul two x        -- 2x
    let (ctx, x2_plus_2x) := ctx.add x2 twox  -- x² + 2x
    let one := Dual.const 1.0
    ctx.add x2_plus_2x one                   -- x² + 2x + 1

  let x := 3.0
  let (value, gradient) := gradWith f1 x
  IO.println s!"  f({x}) = {value}"
  IO.println s!"  f'({x}) = {gradient}"
  IO.println s!"  Expected: f'(x) = 2x + 2 = {2.0 * x + 2.0}"
  IO.println ""

  -- Example 2: Transcendental function f(x) = exp(x) * sin(x)
  IO.println "Example 2: f(x) = exp(x) * sin(x)"
  let f2 (ctx : ADContext) (x : Dual) : ADContext × Dual :=
    let (ctx, expx) := ctx.exp x
    let (ctx, sinx) := ctx.sin x
    ctx.mul expx sinx

  let x2 := 1.0
  let (value2, gradient2) := gradWith f2 x2
  IO.println s!"  f({x2}) = {value2}"
  IO.println s!"  f'({x2}) = {gradient2}"
  -- f'(x) = exp(x)*sin(x) + exp(x)*cos(x) = exp(x)*(sin(x) + cos(x))
  let expected := Float.exp x2 * (Float.sin x2 + Float.cos x2)
  IO.println s!"  Expected: f'(x) = exp(x)*(sin(x)+cos(x)) = {expected}"
  IO.println ""

  -- Example 3: Neural network activation - sigmoid
  IO.println "Example 3: sigmoid(x) = 1/(1+exp(-x))"
  let f3 (ctx : ADContext) (x : Dual) : ADContext × Dual :=
    ctx.sigmoid x

  let x3 := 0.0
  let (value3, gradient3) := gradWith f3 x3
  IO.println s!"  sigmoid({x3}) = {value3}"
  IO.println s!"  sigmoid'({x3}) = {gradient3}"
  IO.println s!"  Expected: sigmoid'(0) = 0.25"
  IO.println ""

/-- Demo 2: Multivariate Gradients -/
def demo2_multivariate : IO Unit := do
  IO.println "═══════════════════════════════════════════════"
  IO.println "Demo 2: Multivariate Gradients"
  IO.println "═══════════════════════════════════════════════"
  IO.println ""

  -- Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²
  -- Gradient: ∂f/∂x = -2(1-x) - 400x(y-x²)
  --           ∂f/∂y = 200(y-x²)
  IO.println "Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²"

  let rosenbrock (ctx : ADContext) (vars : Array Dual) : ADContext × Dual :=
    let x := vars[0]!
    let y := vars[1]!
    let one := Dual.const 1.0
    let hundred := Dual.const 100.0

    -- (1-x)
    let (ctx, one_minus_x) := ctx.sub one x
    -- (1-x)²
    let (ctx, term1) := ctx.mul one_minus_x one_minus_x

    -- x²
    let (ctx, x_squared) := ctx.mul x x
    -- y - x²
    let (ctx, y_minus_x_sq) := ctx.sub y x_squared
    -- (y-x²)²
    let (ctx, y_minus_x_sq_2) := ctx.mul y_minus_x_sq y_minus_x_sq
    -- 100(y-x²)²
    let (ctx, term2) := ctx.mul hundred y_minus_x_sq_2

    -- Sum the terms
    ctx.add term1 term2

  let point := #[1.0, 1.0]  -- Global minimum
  let (value, grads) := gradNWith rosenbrock point
  IO.println s!"  f(1, 1) = {value}"
  IO.println s!"  ∇f(1, 1) = [{grads[0]!}, {grads[1]!}]"
  IO.println s!"  Expected: [0, 0] (global minimum)"
  IO.println ""

  let point2 := #[0.0, 0.0]
  let (value2, grads2) := gradNWith rosenbrock point2
  IO.println s!"  f(0, 0) = {value2}"
  IO.println s!"  ∇f(0, 0) = [{grads2[0]!}, {grads2[1]!}]"
  IO.println s!"  Expected: [2, 0]"
  IO.println ""

/-- Demo 3: Optimization with SGD -/
def demo3_sgd : IO Unit := do
  IO.println "═══════════════════════════════════════════════"
  IO.println "Demo 3: Optimization with SGD"
  IO.println "═══════════════════════════════════════════════"
  IO.println ""

  -- Minimize f(x) = (x - 5)² using SGD
  IO.println "Minimize f(x) = (x - 5)² starting from x = 0"
  IO.println ""

  let objective (ctx : ADContext) (vars : Array Dual) : ADContext × Dual :=
    let x := vars[0]!
    let target := Dual.const 5.0
    let (ctx, diff) := ctx.sub x target
    ctx.mul diff diff

  -- Initialize
  let initialParams := #[#[0.0]]  -- Start at x = 0
  let sgdConfig := SGDConfig.default.withLearningRate 0.1
  let mut state := SGDState.fromParams initialParams
  let mut params := initialParams

  IO.println "Iteration | x | f(x) | gradient"
  IO.println "----------|---|------|----------"

  -- Run 20 iterations
  for i in [:20] do
    -- Compute loss and gradients
    let (loss, grads) := gradNWith objective (params.map (·[0]!))
    let gradsArray := #[grads]

    -- Print current state
    let x := params[0]![0]!
    let grad := grads[0]!
    IO.println s!"    {i}    | {x} | {loss} | {grad}"

    -- Update parameters
    let (newParams, newState) := Hesper.Optimizer.SGD.stepIndexed sgdConfig params gradsArray state
    params := newParams
    state := newState

  let finalX := params[0]![0]!
  IO.println ""
  IO.println s!"Final result: x = {finalX} (target: 5.0)"
  IO.println ""

/-- Demo 4: Optimization with Adam -/
def demo4_adam : IO Unit := do
  IO.println "═══════════════════════════════════════════════"
  IO.println "Demo 4: Optimization with Adam"
  IO.println "═══════════════════════════════════════════════"
  IO.println ""

  -- Minimize Rosenbrock function using Adam
  IO.println "Minimize Rosenbrock f(x,y) = (1-x)² + 100(y-x²)²"
  IO.println "Starting from (x, y) = (0, 0)"
  IO.println ""

  let rosenbrock (ctx : ADContext) (vars : Array Dual) : ADContext × Dual :=
    let x := vars[0]!
    let y := vars[1]!
    let one := Dual.const 1.0
    let hundred := Dual.const 100.0

    let (ctx, one_minus_x) := ctx.sub one x
    let (ctx, term1) := ctx.mul one_minus_x one_minus_x

    let (ctx, x_squared) := ctx.mul x x
    let (ctx, y_minus_x_sq) := ctx.sub y x_squared
    let (ctx, y_minus_x_sq_2) := ctx.mul y_minus_x_sq y_minus_x_sq
    let (ctx, term2) := ctx.mul hundred y_minus_x_sq_2

    ctx.add term1 term2

  -- Initialize
  let initialParams := #[#[0.0, 0.0]]  -- Start at origin
  let adamConfig := AdamConfig.default.withLearningRate 0.01
  let mut state := AdamState.fromParams initialParams
  let mut params := initialParams

  IO.println "Iteration | x | y | f(x,y) | ||∇f||"
  IO.println "----------|---|---|--------|--------"

  -- Run 100 iterations
  for i in [:100] do
    -- Compute loss and gradients
    let point := #[params[0]![0]!, params[0]![1]!]
    let (loss, grads) := gradNWith rosenbrock point
    let gradsArray := #[#[grads[0]!, grads[1]!]]

    -- Compute gradient norm
    let gradNorm := Float.sqrt (grads[0]! * grads[0]! + grads[1]! * grads[1]!)

    -- Print every 10 iterations
    if i % 10 == 0 then
      let x := point[0]!
      let y := point[1]!
      IO.println s!"    {i}   | {x} | {y} | {loss} | {gradNorm}"

    -- Update parameters
    let (newParams, newState) := Hesper.Optimizer.Adam.step adamConfig params gradsArray state
    params := newParams
    state := newState

  let finalX := params[0]![0]!
  let finalY := params[0]![1]!
  IO.println ""
  IO.println s!"Final result: (x, y) = ({finalX}, {finalY})"
  IO.println s!"Target: (1, 1) - global minimum"
  IO.println ""

/-- Demo 5: Optimizer Comparison -/
def demo5_comparison : IO Unit := do
  IO.println "═══════════════════════════════════════════════"
  IO.println "Demo 5: SGD vs Adam Comparison"
  IO.println "═══════════════════════════════════════════════"
  IO.println ""

  IO.println "Key differences:"
  IO.println ""
  IO.println "SGD (Stochastic Gradient Descent):"
  IO.println "  • Simple and robust"
  IO.println "  • Fixed learning rate for all parameters"
  IO.println "  • Optional momentum for faster convergence"
  IO.println "  • Hyperparameters: learning rate, momentum"
  IO.println ""
  IO.println "Adam (Adaptive Moment Estimation):"
  IO.println "  • Adaptive learning rates per parameter"
  IO.println "  • Combines momentum and RMSprop"
  IO.println "  • Often converges faster than SGD"
  IO.println "  • Hyperparameters: learning rate, β₁, β₂, ε"
  IO.println ""
  IO.println "When to use:"
  IO.println "  • SGD: When you need stability, have good LR schedule"
  IO.println "  • Adam: For quick prototyping, works well out-of-the-box"
  IO.println "  • Both: Common practice in modern deep learning"
  IO.println ""

def main : IO Unit := do
  IO.println "╔════════════════════════════════════════════════╗"
  IO.println "║   Automatic Differentiation & Optimization    ║"
  IO.println "╚════════════════════════════════════════════════╝"
  IO.println ""

  demo1_basic_ad
  demo2_multivariate
  demo3_sgd
  demo4_adam
  demo5_comparison

  IO.println "╔════════════════════════════════════════════════╗"
  IO.println "║   All AD and optimization demos complete!     ║"
  IO.println "╚════════════════════════════════════════════════╝"
  IO.println ""

  IO.println "Summary:"
  IO.println "  ✓ Reverse-mode AD with computational tape"
  IO.println "  ✓ Gradient computation for univariate functions"
  IO.println "  ✓ Gradient computation for multivariate functions"
  IO.println "  ✓ SGD optimizer with momentum support"
  IO.println "  ✓ Adam optimizer with adaptive learning rates"
  IO.println ""
  IO.println "Next steps:"
  IO.println "  • Integrate AD with neural network layers"
  IO.println "  • Add GPU-accelerated gradient computation"
  IO.println "  • Implement learning rate schedules"
  IO.println "  • Add more optimizers (AdamW, RMSprop, etc.)"
  IO.println ""

end Examples.ADDemo

-- Main function (outside namespace for executable entry point)
def main : IO Unit := Examples.ADDemo.main
