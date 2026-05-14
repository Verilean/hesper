# Notebook FFI investigation (2026-05-15)

## Goal

Make `#eval runGpu` work inside a xeus-lean notebook cell — i.e. the
notebook elaborates `parallelForDSL device kernel input` and Lean's
interpreter actually resolves the `@[extern "lean_hesper_*"]` symbols
to call into Dawn / Vulkan.

## What works (committed)

`lean_lib «Hesper»` now sets `precompileModules := true` and
`moreLinkArgs := stdLinkArgs`.  This makes Lake produce
`Hesper_Hesper_<Mod>.so` next to every `.olean`.  Each per-module
`.so` is link-statically-pulled-in to the C bridge artifacts
(`libhesper_native.a`, Dawn, Highway, Vulkan, …) via `stdLinkArgs`,
so the `.so` is self-contained for the `@[extern]` symbols it
declares.

`lake build parallel-demo` succeeds and the resulting binary runs
on the host's NVIDIA Vulkan adapter, exactly like before the change
— the only delta is the extra per-module `.so`s now exist.

## What still does NOT work (deferred)

The xlean Jupyter kernel (xeus-lean) doesn't auto-load these
`.so`s when a notebook does `import Hesper.*`, so `#eval`-ing a
function that calls an `@[extern]` declaration still fails with
"Could not find native implementation of external declaration".

Lake's own `lean` invocations work because Lake writes a per-module
`setup.json` containing the `dynlibs` list and passes
`--setup <file>.setup.json` to `lean`.  xlean doesn't do this — it
just sets `LEAN_PATH` and trusts auto-discovery.

## Three possible fixes (next session)

1. **Patch xlean to read `LEAN_DYNLIB_PATH`** — small (5-line) change
   to `src/XeusKernel.lean`'s `main` to iterate `LEAN_DYNLIB_PATH`
   env-var entries and call `Lean.loadDynlib`.  Requires rebuilding
   xeus-lean (CMake configure → cmake build → lake build xlean),
   which is hard on NixOS host but easy in Docker.
   **Already drafted** at `~/git/xeus-lean/src/XeusKernel.lean` (not
   committed in xeus-lean repo).
2. **Re-link xlean against Hesper at Docker build time** — what
   `docker/tutorial/Dockerfile`'s Stage 5 attempted earlier this
   session.  Works in principle but requires bundling
   `.c.o.export` files into a single archive, which requires
   `lake build` to emit them (only happens when a `lean_exe` links).
3. **Auto-symlink `.so`s into a directory `lean` searches by
   default** — investigate whether putting them under
   `${LEAN_PATH}/.../` makes Lean pick them up.  Not yet tested.

## Verified facts (so they don't get re-discovered)

- `Lean.loadDynlib (path : FilePath) : IO Unit` is the supported API
  (`import Lean.LoadDynlib`).
- `--load-dynlib=PATH.so` flag on the `lean` CLI works for `#eval`.
- `LD_PRELOAD` does NOT work, because Lean resolves `@[extern]` via
  Lean-internal `lp_<mangled>___boxed` trampolines, which only
  exist in the `precompileModules` `.so`.
- A minimal `extern_lib` + `precompileModules` + `moreLinkArgs`
  combination works end-to-end at
  `/tmp/extern-test/` (test returns `42` from `#eval myDouble 21`).
- xeus-lean upstream needs a CMake step before `lake build xlean`
  to produce `libxeus_ffi.a` and the protocol-version patch — see
  `Dockerfile.native-sparkle` Stage 4 for the exact recipe.

## What to do tomorrow

1. Run `docker build` of the tutorial image — Stage 4 builds xeus-
   lean fresh, where my `LEAN_DYNLIB_PATH` patch can be applied via
   a `COPY` overlay (avoids needing to push upstream).
2. After `lake build Hesper` in Stage 5, set
   `LEAN_DYNLIB_PATH=$(echo $.so_files | tr '\n' ':')` in the
   kernelspec `env`.
3. Open Ch04 in jupyter and run the `#eval runGpu` cell.
   Expected: `#[0.0, 1000.0, 2000.0, …, 9000.0]`.
