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

-- f32 expressions:
let x : Exp (.scalar .f32) := var "x"
let y : Exp (.scalar .f32) := var "y"
let r := sqrt (x * x + y * y)         -- OK

-- This is a compile error, not a runtime crash:
-- let bad : Exp (.scalar .f32) := x + (var "i" : Exp (.scalar .i32))
```

If your shader elaborates, its types are consistent.

## The `Exp` layer

`Exp` is indexed by a `Ty` describing the shape and scalar kind. Common
constructors:

```lean
-- Variable / parameter:
var "name" : Exp (.scalar .f32)

-- Numeric literals:
const 1.0 : Exp (.scalar .f32)

-- Arithmetic:
(a + b)        -- Exp (.scalar .f32)
(a * b)
(a / b)

-- Math:
sqrt a
log a
exp a
relu a
```

Vector and matrix types work the same way:

```lean
let v : Exp (.vec 4 .f32) := var "v"
let w : Exp (.vec 4 .f32) := var "w"
let dot4 := v.dot w            -- Exp (.scalar .f32)
```

## The `ShaderM` monad

A full kernel is a `ShaderM Unit`. Inside it you can:

- declare bound buffers (input / output, read-only / read-write),
- declare shared / workgroup-local arrays,
- write loops and `if` branches,
- emit reads and writes that lower to actual ld/st instructions.

```lean
import Hesper.WGSL.Monad

def addOneKernel (n : Nat) : ShaderM Unit := do
  let input  ← declareReadOnlyBuffer (.array .f32 n)
  let output ← declareOutputBuffer    (.array .f32 n)
  let gid    ← workgroupX
  if_ (gid <ᵉ .nat n) do
    let x := readBuffer input gid
    writeBuffer output gid (x + const 1.0)
```

`workgroupX` is the global-thread-x index. `<ᵉ` is the lifted `<`
operator on `Exp` (Lean's `<` operates on values, not shader
expressions, hence the suffix).

## Compiling to WGSL

```lean
import Hesper.WGSL.Execute

let wgsl : String := Hesper.WGSL.toString (addOneKernel 1024)
IO.println wgsl
```

The generated WGSL is a faithful translation — every Lean construct
maps to a deterministic WGSL fragment. There's no hidden optimizer in
the way; what you write is what runs.

## Compiling to PTX

The same `ShaderM` value goes through a different backend:

```lean
import Hesper.CUDA.CodeGen

let ptx : String := Hesper.CUDA.toPTX (addOneKernel 1024)
IO.println ptx
```

PTX is NVIDIA's intermediate representation; the driver JITs it into
SASS at load time. Ch05 covers backend selection in depth.

## Common patterns

### Reduction along the last axis

```lean
def sumLastAxis (n : Nat) : ShaderM Unit := do
  let input  ← declareReadOnlyBuffer (.array .f32 n)
  let output ← declareOutputBuffer    (.scalar .f32)
  let mut acc ← varNamed "acc" (const 0.0)
  loop (n.toNat) fun i => do
    acc := acc + readBuffer input (.nat i)
  writeBuffer output (.nat 0) acc
```

### Matmul one-row-per-thread

See `Examples/Compute/MatmulSimple.lean` for a runnable version. The
sketch:

```lean
def matmulRow (m k n : Nat) : ShaderM Unit := do
  let a ← declareReadOnlyBuffer (.array .f32 (m * k))
  let b ← declareReadOnlyBuffer (.array .f32 (k * n))
  let c ← declareOutputBuffer    (.array .f32 (m * n))
  let row ← workgroupX
  let col ← workgroupY
  let mut acc ← varNamed "acc" (const 0.0)
  loop k fun i => do
    acc := acc + readBuffer a (row * .nat k + .nat i)
              * readBuffer b (.nat i * .nat n + col)
  writeBuffer c (row * .nat n + col) acc
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
