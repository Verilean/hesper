import Hesper.Core.Differentiable

namespace Hesper.AD

open Hesper.Core

-- Op tags for scalar operations
structure AddOp deriving Inhabited
structure MulOp deriving Inhabited
structure ExpOp deriving Inhabited
structure SigmoidOp deriving Inhabited
structure ReLUOp deriving Inhabited

-- Scalar Addition
instance : Differentiable AddOp (Float × Float) Float where
  forward := fun _ (x, y) => x + y
  backward := fun _ (_, _) _ => (1.0, 1.0)

-- Scalar Multiplication
instance : Differentiable MulOp (Float × Float) Float where
  forward := fun _ (x, y) => x * y
  backward := fun _ (x, y) _ => (y, x)

-- Scalar Exp
instance : Differentiable ExpOp Float Float where
  forward := fun _ x => Float.exp x
  backward := fun _ x _ => Float.exp x

-- Scalar Sigmoid
instance : Differentiable SigmoidOp Float Float where
  forward := fun _ x => 1.0 / (1.0 + Float.exp (-x))
  backward := fun _ x _ =>
    let s := 1.0 / (1.0 + Float.exp (-x))
    s * (1.0 - s)

-- Scalar ReLU
instance : Differentiable ReLUOp Float Float where
  forward := fun _ x => max 0.0 x
  backward := fun _ x _ => if x > 0.0 then 1.0 else 0.0

end Hesper.AD
