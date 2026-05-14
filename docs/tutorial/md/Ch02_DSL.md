# Chapter 02 — The Shader DSL (WGSL + ShaderM)

Hesper's core is a small, strongly typed language for writing GPU
shaders inside Lean. There are two layers:

1. **`Exp`** — type-safe shader expressions (the "value" level).
2. **`ShaderM`** — an imperative monad for assembling a full shader
   (declare buffers, write loops, emit statements).

Both layers lower to two backends: WGSL (for WebGPU / Dawn) and PTX (for
CUDA). You write the same DSL once.

## Why this layer exists

Plain WGSL is a string. Anything goes — including type mismatches and
buffer-index off-by-ones that only surface at runtime when the GPU
driver refuses to compile your shader. Hesper makes the elaborator do
the work:

```lean
import Hesper.WGSL.DSL
import Hesper.WGSL.CodeGen

open Hesper.WGSL
open Hesper.WGSL.Monad (ShaderM)
open Hesper.WGSL.Monad.ShaderM
open Hesper.WGSL.CodeGen

-- f32 expressions:
def x_f32 : Exp (.scalar .f32) := var "x"
def y_f32 : Exp (.scalar .f32) := var "y"
def r_f32 := sqrt (x_f32 * x_f32 + y_f32 * y_f32)
```

```lean
-- This is fine and elaborates to a printable expression:
#eval r_f32.toWGSL
-- sqrt(((x * x) + (y * y)))
```

If your shader elaborates, its types are consistent. Try uncommenting
the line below in your own copy — it's a Lean type error, not a
runtime crash:

```text
-- def bad : Exp (.scalar .f32) :=
--   x_f32 + (var "i" : Exp (.scalar .i32))
```

## The `Exp` layer

`Exp` is indexed by a `WGSLType` describing the shape and scalar kind.
Common constructors:

```lean
-- Variables and literals
#check (var "name" : Exp (.scalar .f32))
#check (Exp.litF32 1.0 : Exp (.scalar .f32))
#check (Exp.litU32 0  : Exp (.scalar .u32))

-- Arithmetic (operators come from instances on Exp):
#check fun (a b : Exp (.scalar .f32)) => a + b
#check fun (a b : Exp (.scalar .f32)) => a * b

-- Math wrappers:
#check fun (a : Exp (.scalar .f32)) => sqrt a
```

Vector types work the same way — `Exp (.vec3 .f32)` is a 3-vector of
f32, etc.

## The `ShaderM` monad

A full kernel is a `ShaderM Unit`. Inside it you can:

- declare bound buffers (input / output, read-only / read-write),
- declare shared / workgroup-local arrays,
- write loops and `if` branches,
- emit reads and writes that lower to actual ld/st instructions.

```lean
def addOneKernel : ShaderM Unit := do
  let _input  ← declareInputBuffer  "input"  (.array (.scalar .f32) 1024)
  let _output ← declareOutputBuffer "output" (.array (.scalar .f32) 1024)
  let gid ← globalId
  let idx := Exp.vecZ gid                     -- thread x-coordinate
  let v ← readBuffer (ty := .scalar .f32) (n := 1024) "input" idx
  writeBuffer (ty := .scalar .f32) "output" idx (v + Exp.litF32 1.0)
```

`globalId : ShaderM (Exp (.vec3 .u32))` gives the global thread id;
`Exp.vecZ` projects the x-component. `if_` and `loop` follow the same
pattern (see `Hesper/WGSL/Monad.lean`).

## Compiling to WGSL

```lean
def addOneWGSL : IO Unit := do
  -- Generate the full WGSL module string and print it
  let src := Hesper.WGSL.CodeGen.generateWGSLSimple addOneKernel
  IO.println src
```

`toWGSL` is the deterministic printer — what you write is what runs.
There's no hidden optimizer in the way.

## Compiling to PTX

The same `ShaderM` value goes through a different backend for CUDA.
See `Hesper/CUDA/CodeGen.lean` for the PTX printer; the lowering for
each `Exp` constructor lives next to it. Ch05 covers backend selection
in depth.

## Common patterns

The pre-built examples in `Examples/DSL/` show idiomatic kernels:

- **`Examples/DSL/DSLBasics.lean`** — arithmetic and shader expression
  printing.
- **`Examples/DSL/CodeGenDemo.lean`** — simpleKernel / reductionKernel /
  complexCompute, end-to-end ShaderM + WGSL.
- **`Examples/DSL/AtomicCounter.lean`** — atomic ops with `if_` and
  workgroup barriers.

```bash
lake exe codegen-demo        # prints the three kernels above as WGSL
lake exe dsl-basics          # prints Exp expressions
```

## Where to look next

- **`Hesper/WGSL/Exp.lean`** — every `Exp` constructor with its type.
- **`Hesper/WGSL/Monad.lean`** — `ShaderM` helpers (buffers, loops,
  reductions, warp/lane primitives).
- **`Hesper/WGSL/CodeGen.lean`** — the WGSL printer.
- **`Hesper/CUDA/CodeGen.lean`** — the PTX printer.
- **`docs/circuit-dsl-tutorial.md`** — the fusion DSL layered on top of
  ShaderM for hand-tuned kernels.

## What's next

- [Chapter 03 — Automatic Differentiation](Ch03_AD.md): build differentiable
  ops on top of the DSL.
- [Chapter 10 — Hesper Architecture](Ch10_Architecture.md): how `Exp`,
  `ShaderM`, the WGSL/PTX backends, and Circuit DSL fit together.
