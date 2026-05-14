# Chapter 01 — Lean 4 for ML Engineers

You don't need to be a Lean expert to use Hesper. This chapter teaches
just enough to read the rest of the tutorial. If you've used Haskell,
OCaml, or even just Python with type hints, most of it will feel
familiar — Lean is a strict, ML-style functional language with a
dependent type system on top.

## Types and definitions

```lean
def square (x : Float) : Float := x * x

#eval square 3.14                 -- 9.8596
#check @square                    -- @square : Float → Float
```

`def name (args) : ReturnType := body`. Type annotations on arguments
and return type are usually optional — Lean infers them — but in
tutorial code we'll write them out for clarity.

## The `Array`, `List`, and `String` types

Lean's standard library gives you the usual collections:

```lean
def xs : Array Nat := #[1, 2, 3, 4]
def ys : List Float := [1.0, 2.0, 3.0]

#eval xs.map (· + 1)              -- #[2, 3, 4, 5]
#eval ys.foldl (· + ·) 0.0        -- 6.0
```

`(· + 1)` is shorthand for `fun x => x + 1`. The dot-notation `xs.map`
desugars to `Array.map xs`.

## Structures and product types

```lean
structure Shape where
  rows : Nat
  cols : Nat
deriving Repr

def s : Shape := { rows := 3, cols := 4 }

#eval s.rows * s.cols             -- 12
```

We'll use small `structure`s a lot — `Tensor`, `Config`, `Layer`, etc.

## The `IO` monad

GPU side-effects and FFI calls live inside `IO`. You write `do`-blocks
just like Haskell or Rust:

```lean
def hello : IO Unit := do
  IO.println "hello GPU"

#eval hello
```

`←` binds the result of an `IO` action; `let x := ...` is pure binding.
The `s!"..."` syntax is string interpolation.

## Dependent types (the part that's new)

Lean lets a type depend on a value. This is what makes the shader DSL
type-safe:

```lean
inductive Kind | f32 | i32 | u32

-- A type that depends on a value of Kind.  We use `abbrev` (not `def`)
-- so that `Carrier .f32` reduces to `Float` transparently — otherwise
-- the elaborator can't see through the alias when it needs to find a
-- `OfNat Float 1` instance below.
abbrev Carrier : Kind → Type
  | .f32 => Float
  | .i32 => Int
  | .u32 => UInt32

def oneOf : (k : Kind) → Carrier k
  | .f32 => 1.0
  | .i32 => 1
  | .u32 => 1
```

In Hesper, `Exp (.scalar .f32)` and `Exp (.scalar .i32)` are *different*
types, so a stray `i32` cannot sneak into an `f32` shader expression.

## Typeclasses

Typeclasses look like Haskell:

```lean
class Approximable (α : Type) where
  approxEq : α → α → Bool

instance : Approximable Float where
  approxEq a b := (a - b).abs < 1e-6

#eval Approximable.approxEq (1.0 : Float) 1.0000001
-- true
```

Hesper uses typeclasses for `GPUBackend` (Ch05), `Differentiable` (Ch03),
and for tensor element types.

## Module imports

A real Lean file starts with all its imports up top, like Python or
Haskell. In the rest of this tutorial each chapter's notebook places
its imports in the *first* code cell:

```text
import Hesper.WGSL.DSL          -- the type-safe shader DSL
import Hesper.Compute           -- high-level compute API
import Hesper.Models.BitNet     -- BitNet inference
```

## What's next

- [Chapter 01b — Your First Hesper Project](Ch01b_YourFirstProject.md):
  set up a fresh package that uses Hesper as a dependency.
- [Chapter 02 — The Shader DSL](Ch02_DSL.md): the type-safe expression
  language we actually compile to WGSL and PTX.
