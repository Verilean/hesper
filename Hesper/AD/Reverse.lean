/-!
# Reverse-Mode Automatic Differentiation

Implementation of reverse-mode AD (backpropagation) similar to Haskell's `ad` package.
This enables efficient gradient computation for machine learning optimization.

## Features
- Dual numbers with forward value and backward gradient
- Reverse-mode differentiation (backpropagation)
- Support for common math operations
- Efficient gradient computation via computational graph

## Usage Example
```lean
-- Define a function
def f (x : Float) : Float := x * x + 2.0 * x + 1.0

-- Compute gradient at x = 3.0
let grad := diff f 3.0
-- Result: 8.0 (derivative of x² + 2x + 1 at x=3 is 2x + 2 = 8)
```
-/

namespace Hesper.AD.Reverse

/-- Tape entry for reverse-mode AD.
    Stores the operation index, parent indices, and local derivatives.
-/
structure TapeEntry where
  /-- Index of this operation in the tape -/
  idx : Nat
  /-- Parent indices (inputs to this operation) -/
  parents : List Nat
  /-- Local derivatives with respect to each parent -/
  localGrads : List Float
  deriving Inhabited, Repr

/-- Computational tape for reverse-mode AD -/
structure Tape where
  /-- Entries in the tape (reverse chronological order) -/
  entries : Array TapeEntry
  /-- Current index counter -/
  nextIdx : Nat
  deriving Inhabited

namespace Tape

/-- Create an empty tape -/
def empty : Tape := {
  entries := #[]
  nextIdx := 1  -- Start at 1 to avoid aliasing with constants (index 0)
}

/-- Add an entry to the tape -/
def add (tape : Tape) (parents : List Nat) (localGrads : List Float) : Tape × Nat :=
  let idx := tape.nextIdx
  let entry : TapeEntry := {
    idx := idx
    parents := parents
    localGrads := localGrads
  }
  ({ entries := tape.entries.push entry
     nextIdx := idx + 1 }, idx)

end Tape

/-- Dual number for reverse-mode AD.
    Contains the primal value and a reference to the tape.
-/
structure Dual where
  /-- Forward (primal) value -/
  primal : Float
  /-- Index in the computational tape -/
  tapeIdx : Nat
  deriving Inhabited, Repr

namespace Dual

/-- Create a constant (no gradient) -/
def const (x : Float) : Dual := {
  primal := x
  tapeIdx := 0  -- Constants have tape index 0
}

/-- Create a variable (will have gradient) -/
def var (x : Float) (tape : Tape) : Tape × Dual :=
  let (tape', idx) := tape.add [] []
  (tape', { primal := x, tapeIdx := idx })

/-- Lift a unary function to Dual numbers -/
def lift1 (f : Float → Float) (df : Float → Float) (x : Dual) (tape : Tape) : Tape × Dual :=
  let y := f x.primal
  let (tape', idx) := tape.add [x.tapeIdx] [df x.primal]
  (tape', { primal := y, tapeIdx := idx })

/-- Lift a binary function to Dual numbers -/
def lift2 (f : Float → Float → Float)
          (df_dx : Float → Float → Float)
          (df_dy : Float → Float → Float)
          (x y : Dual) (tape : Tape) : Tape × Dual :=
  let z := f x.primal y.primal
  let (tape', idx) := tape.add
    [x.tapeIdx, y.tapeIdx]
    [df_dx x.primal y.primal, df_dy x.primal y.primal]
  (tape', { primal := z, tapeIdx := idx })

end Dual

/-- AD context for managing the computational tape -/
structure ADContext where
  /-- The computational tape -/
  tape : Tape
  deriving Inhabited

namespace ADContext

/-- Create a new AD context -/
def new : ADContext := { tape := Tape.empty }

/-- Create a variable in this context -/
def var (ctx : ADContext) (x : Float) : ADContext × Dual :=
  let (tape', d) := Dual.var x ctx.tape
  ({ tape := tape' }, d)

/-- Addition -/
def add (ctx : ADContext) (x y : Dual) : ADContext × Dual :=
  let (tape', d) := Dual.lift2
    (· + ·)
    (fun _ _ => 1.0)  -- ∂(x+y)/∂x = 1
    (fun _ _ => 1.0)  -- ∂(x+y)/∂y = 1
    x y ctx.tape
  ({ tape := tape' }, d)

/-- Subtraction -/
def sub (ctx : ADContext) (x y : Dual) : ADContext × Dual :=
  let (tape', d) := Dual.lift2
    (· - ·)
    (fun _ _ => 1.0)   -- ∂(x-y)/∂x = 1
    (fun _ _ => -1.0)  -- ∂(x-y)/∂y = -1
    x y ctx.tape
  ({ tape := tape' }, d)

/-- Multiplication -/
def mul (ctx : ADContext) (x y : Dual) : ADContext × Dual :=
  let (tape', d) := Dual.lift2
    (· * ·)
    (fun _ y => y)     -- ∂(x*y)/∂x = y
    (fun x _ => x)     -- ∂(x*y)/∂y = x
    x y ctx.tape
  ({ tape := tape' }, d)

/-- Division -/
def div (ctx : ADContext) (x y : Dual) : ADContext × Dual :=
  let (tape', d) := Dual.lift2
    (· / ·)
    (fun _ y => 1.0 / y)                    -- ∂(x/y)/∂x = 1/y
    (fun x y => -x / (y * y))               -- ∂(x/y)/∂y = -x/y²
    x y ctx.tape
  ({ tape := tape' }, d)

/-- Exponentiation -/
def exp (ctx : ADContext) (x : Dual) : ADContext × Dual :=
  let (tape', d) := Dual.lift1
    Float.exp
    Float.exp  -- ∂(exp(x))/∂x = exp(x)
    x ctx.tape
  ({ tape := tape' }, d)

/-- Natural logarithm -/
def log (ctx : ADContext) (x : Dual) : ADContext × Dual :=
  let (tape', d) := Dual.lift1
    Float.log
    (fun x => 1.0 / x)  -- ∂(log(x))/∂x = 1/x
    x ctx.tape
  ({ tape := tape' }, d)

/-- Square root -/
def sqrt (ctx : ADContext) (x : Dual) : ADContext × Dual :=
  let (tape', d) := Dual.lift1
    Float.sqrt
    (fun x => 0.5 / Float.sqrt x)  -- ∂(√x)/∂x = 1/(2√x)
    x ctx.tape
  ({ tape := tape' }, d)

/-- Sine -/
def sin (ctx : ADContext) (x : Dual) : ADContext × Dual :=
  let (tape', d) := Dual.lift1
    Float.sin
    Float.cos  -- ∂(sin(x))/∂x = cos(x)
    x ctx.tape
  ({ tape := tape' }, d)

/-- Cosine -/
def cos (ctx : ADContext) (x : Dual) : ADContext × Dual :=
  let (tape', d) := Dual.lift1
    Float.cos
    (fun x => -Float.sin x)  -- ∂(cos(x))/∂x = -sin(x)
    x ctx.tape
  ({ tape := tape' }, d)

/-- Power (x^n) -/
def pow (ctx : ADContext) (x : Dual) (n : Float) : ADContext × Dual :=
  let (tape', d) := Dual.lift1
    (fun x => Float.pow x n)
    (fun x => n * Float.pow x (n - 1.0))  -- ∂(x^n)/∂x = n*x^(n-1)
    x ctx.tape
  ({ tape := tape' }, d)

/-- Sigmoid activation -/
def sigmoid (ctx : ADContext) (x : Dual) : ADContext × Dual :=
  let (tape', d) := Dual.lift1
    (fun x => 1.0 / (1.0 + Float.exp (-x)))
    (fun x =>
      let s := 1.0 / (1.0 + Float.exp (-x))
      s * (1.0 - s))  -- ∂(sigmoid(x))/∂x = sigmoid(x) * (1 - sigmoid(x))
    x ctx.tape
  ({ tape := tape' }, d)

/-- Tanh activation -/
def tanh (ctx : ADContext) (x : Dual) : ADContext × Dual :=
  let (tape', d) := Dual.lift1
    (fun x => Float.tanh x)
    (fun x =>
      let t := Float.tanh x
      1.0 - t * t)  -- ∂(tanh(x))/∂x = 1 - tanh²(x)
    x ctx.tape
  ({ tape := tape' }, d)

/-- ReLU activation -/
def relu (ctx : ADContext) (x : Dual) : ADContext × Dual :=
  let (tape', d) := Dual.lift1
    (fun x => max 0.0 x)
    (fun x => if x > 0.0 then 1.0 else 0.0)  -- ∂(ReLU(x))/∂x = 1 if x>0, else 0
    x ctx.tape
  ({ tape := tape' }, d)

end ADContext

/-- Reverse-mode gradient computation via backpropagation -/
def backprop (tape : Tape) (outputIdx : Nat) : Array Float :=
  -- Initialize gradient array (all zeros)
  let grads : Array Float := Array.mk (List.replicate tape.nextIdx 0.0)

  -- Set output gradient to 1.0
  let grads := grads.set! outputIdx 1.0

  -- Backpropagate through tape in reverse order
  tape.entries.foldr (init := grads) fun entry grads =>
    let outGrad := grads[entry.idx]!
    -- Distribute gradient to parents
    entry.parents.zip entry.localGrads |>.foldl (init := grads) fun grads (parentIdx, localGrad) =>
      let currentGrad := grads[parentIdx]!
      grads.set! parentIdx (currentGrad + outGrad * localGrad)

/-- Compute gradient of a univariate function using reverse-mode AD -/
def grad (f : ADContext → Dual → ADContext × Dual) (x : Float) : Float :=
  -- Create context and variable
  let ctx := ADContext.new
  let (ctx, xDual) := ctx.var x

  -- Evaluate function
  let (ctx, y) := f ctx xDual

  -- Backpropagate
  let grads := backprop ctx.tape y.tapeIdx

  -- Return gradient at input
  grads[xDual.tapeIdx]!

/-- Compute value and gradient of a univariate function -/
def gradWith (f : ADContext → Dual → ADContext × Dual) (x : Float) : Float × Float :=
  let ctx := ADContext.new
  let (ctx, xDual) := ctx.var x
  let (ctx, y) := f ctx xDual
  let grads := backprop ctx.tape y.tapeIdx
  (y.primal, grads[xDual.tapeIdx]!)

/-- Compute gradient of a function with respect to multiple variables -/
def gradN (f : ADContext → Array Dual → ADContext × Dual) (xs : Array Float) : Array Float :=
  let ctx := ADContext.new

  -- Create variables for all inputs
  let (ctx, xDuals) := xs.foldl (init := (ctx, #[])) fun (ctx, acc) x =>
    let (ctx', xDual) := ctx.var x
    (ctx', acc.push xDual)

  -- Evaluate function
  let (ctx, y) := f ctx xDuals

  -- Backpropagate
  let grads := backprop ctx.tape y.tapeIdx

  -- Collect gradients for all inputs
  xDuals.map fun xDual => grads[xDual.tapeIdx]!

/-- Compute value and gradient for multiple variables -/
def gradNWith (f : ADContext → Array Dual → ADContext × Dual) (xs : Array Float) : Float × Array Float :=
  let ctx := ADContext.new
  let (ctx, xDuals) := xs.foldl (init := (ctx, #[])) fun (ctx, acc) x =>
    let (ctx', xDual) := ctx.var x
    (ctx', acc.push xDual)
  let (ctx, y) := f ctx xDuals
  let grads := backprop ctx.tape y.tapeIdx
  let gradValues := xDuals.map fun xDual => grads[xDual.tapeIdx]!
  (y.primal, gradValues)

end Hesper.AD.Reverse
