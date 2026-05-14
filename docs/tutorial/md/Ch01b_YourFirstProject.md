# Chapter 01b — Your First Hesper Project

This chapter walks you through creating a new Lean 4 project that uses
Hesper as a library dependency. At the end you'll have a working
executable that runs a small computation on the GPU.

## Create a fresh package

```bash
lake new MyHesperApp
cd MyHesperApp
```

This produces a minimal `MyHesperApp/` tree with `lakefile.lean`,
`Main.lean`, and a stub `MyHesperApp.lean` library.

## Depend on Hesper

Edit `lakefile.lean` to add a `require` clause:

```lean
import Lake
open Lake DSL

package «MyHesperApp» where
  -- optional: extraDepTargets to build native deps eagerly

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

## A minimal GPU computation

Replace `Main.lean` with:

```lean
import Hesper.Compute
import Hesper.WGSL.DSL

def main : IO Unit := do
  let dev ← Hesper.Device.create
  let n := 1024

  -- Build a buffer of 1024 f32 zeros, run an "add 1" shader, read it
  -- back, and print the first element.
  let input  ← dev.allocBuffer (n * 4)
  let output ← dev.allocBuffer (n * 4)

  -- Use the high-level API for simple elementwise ops — Ch02 shows
  -- how to write the shader by hand.
  dev.fill input 0.0
  dev.elementwise output input (fun x => x + 1.0)

  let host : ByteArray ← dev.readBuffer output
  let firstF32 : Float := host.toFloat32Array.get! 0
  IO.println s!"output[0] = {firstF32}"      -- 1.0
```

`Hesper.Compute` exposes the user-facing API. `Hesper.WGSL.DSL` exports
the type-safe shader-expression layer covered in the next chapter.

## Build and run

```bash
lake build
./.lake/build/bin/my-hesper-app
```

Expected output:

```
output[0] = 1.0
```

If the build fails complaining about Dawn or X11 headers, see the
[Setup](Ch00_Setup.md) chapter for platform-specific apt/brew packages.

## What's next

- [Chapter 02 — The Shader DSL](Ch02_DSL.md): write the shader expression
  by hand instead of using the elementwise helper.
- [Chapter 04 — High-Level API & Tensors](Ch04_HighLevelApi.md): jump to
  tensor and NN layers if you'd rather not write shaders.
