# Chapter 09 — Embedding Hesper in Other Projects

Hesper is published as a Lake package. This chapter shows the three
common ways to depend on it from your own code.

## Pattern 1: pure-Lean dependency

If you only need the DSL and the high-level API (no native customisation),
add Hesper to your `lakefile.lean`:

```lean
import Lake
open Lake DSL

package «MyApp» where

require Hesper from git
  "https://github.com/Verilean/hesper.git" @ "main"

lean_lib «MyApp» where

@[default_target]
lean_exe «my-app» where
  root := `Main
```

Then `lake update Hesper && lake build`. The Hesper package's
`extraDepTargets := #[`nativeDeps]` triggers the Dawn / Highway / CUDA
bridge build automatically the first time you compile.

This is the right pattern for: applications, research scripts, training
loops.

## Pattern 2: pin a specific commit

For reproducibility, pin a tag or commit hash:

```lean
require Hesper from git
  "https://github.com/Verilean/hesper.git" @ "v0.7-gemma4"
```

The `lake-manifest.json` file at the root of your package records the
resolved revision, so a fresh `git clone` of your project always pulls
the same Hesper.

## Pattern 3: local clone (for hacking on Hesper itself)

If you're developing both your app and Hesper, replace the git URL with
a path:

```lean
require Hesper from "/path/to/local/hesper"
```

Saves you a push/pull cycle every time you tweak the library.

## Linking your own native code

If your app has its own C++ FFI on top of Hesper's, you need to extend
the link line. Hesper exposes `stdLinkArgs` and `cudaExeArgs` as `def`s
in its `lakefile.lean`; mirror that pattern in your own:

```lean
def myExtraLinks : Array String := #[
  "-L/usr/local/lib", "-lmyhelper"
]

lean_exe «my-app» where
  root := `Main
  moreLinkArgs := Hesper.stdLinkArgs ++ Hesper.cudaExeArgs ++ myExtraLinks
```

`Hesper.stdLinkArgs` already pulls in Dawn, Highway, and (on Linux) the
CUDA bridge static lib. You usually only add your own libs on top.

## Choosing modules to import

The library is split so you can keep imports minimal:

| Import | What it brings in |
|---|---|
| `Hesper.WGSL.DSL` | The type-safe `Exp` layer (no GPU device) |
| `Hesper.WGSL.Monad` | The `ShaderM` monad |
| `Hesper.Compute` | High-level device + buffer + dispatch API |
| `Hesper.Layers.*` | Pre-built NN layers |
| `Hesper.AD` | Reverse-mode autodiff |
| `Hesper.Models.BitNet` | The full BitNet engine |
| `Hesper.Models.Gemma4` | The full Gemma 4 engine (CUDA only) |
| `Hesper.CUDA.*` | Direct CUDA driver bindings (advanced) |

Lean is good at dead-code elimination across modules, so importing
`Hesper.Compute` doesn't drag every model into your binary.

## Worked example: a custom training loop

```lean
import Hesper.Compute
import Hesper.AD
import Hesper.Layers.Linear

structure Model where
  fc : Linear (inDim := 32) (outDim := 32)

def trainStep (m : Model) (x y : Tensor [.batch 64, .dim 32]) :
    AD.Update Model :=
  AD.gradientStep (lr := 1e-3) m fun m =>
    let pred := m.fc.forward x
    let loss := ((pred - y) * (pred - y)).mean
    loss

def main : IO Unit := do
  let dev ← Hesper.Device.create
  let mut model : Model := { fc := Linear.random dev 32 32 }
  for _step in [0:1000] do
    let (x, y) ← sampleBatch dev
    model := trainStep model x y |>.apply
```

A complete runnable version lives in `Examples/MachineLearning/`.

## Versioning and stability

Hesper is alpha (see the project README). API breakage policy:

- **Stable**: `Exp`, `ShaderM` core, `Tensor`, `Differentiable`.
- **Stabilising**: `Circuit` DSL, `GPUBackend` typeclass.
- **Unstable**: model-specific internals (`Hesper.Models.*` private
  modules), CUDA-specific tuning knobs.

If you need stability today, pin to a tag and check `docs/CHANGELOG.md`
before bumping.

## What's next

- [Chapter 10 — Hesper Architecture](Ch10_Architecture.md): the
  full-architecture chapter that ties everything together.
