# Chapter 01b — Your First Hesper Project

This chapter walks you through creating a new Lean 4 project that uses
Hesper as a library dependency. At the end you'll have a working
executable that runs a small GPU shader expression.

## Create a fresh package

```bash
lake new MyHesperApp
cd MyHesperApp
```

This produces a minimal `MyHesperApp/` tree with `lakefile.lean`,
`Main.lean`, and a stub `MyHesperApp.lean` library.

## Depend on Hesper

Edit `lakefile.lean` to add a `require` clause (this snippet is a Lake
build script — it's not regular Lean code, so we can't execute it in
the notebook, but you can paste it into your project's `lakefile.lean`):

```text
import Lake
open Lake DSL

package «MyHesperApp» where

require Hesper from git
  "https://github.com/Verilean/hesper.git" @ "main"

lean_lib «MyHesperApp» where

@[default_target]
lean_exe «my-hesper-app» where
  root := `Main
```

Run `lake update Hesper` once to fetch the dependency. The first build
will compile Hesper's native bits (Dawn, Highway) — that takes about ten
minutes. Subsequent builds are incremental.

## A minimal Hesper program

Once `Hesper` is available, your `Main.lean` can use it like any other
module. Here we just build a small WGSL expression and pretty-print it:

```lean
import Hesper.WGSL.DSL

open Hesper.WGSL

def shader : Exp (.scalar .f32) :=
  let x : Exp (.scalar .f32) := var "x"
  let y : Exp (.scalar .f32) := var "y"
  sqrt (x * x + y * y)

#eval shader.toWGSL
-- sqrt(((x * x) + (y * y)))
```

In a real app you'd replace `#eval` with a `def main : IO Unit := ...`
that uploads the kernel to the GPU. See Ch02 for the full ShaderM flow,
and the runnable example at `Examples/DSL/DSLBasics.lean`.

## Build and run

```bash
lake build
./.lake/build/bin/my-hesper-app
```

If the build fails complaining about Dawn or X11 headers, see the
[Setup](Ch00_Setup.md) chapter for platform-specific apt/brew packages.

## What's next

- [Chapter 02 — The Shader DSL](Ch02_DSL.md): write the shader expression
  by hand and emit a full kernel.
- [Chapter 04 — High-Level API & Tensors](Ch04_HighLevelApi.md): jump to
  tensor and NN layers if you'd rather not write shaders.
