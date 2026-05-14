# Chapter 00 — Setup

This chapter gets you to a working Hesper environment. After it you can
load any later chapter as a notebook and run its code cells.

## Two ways to set up

1. **Docker (recommended).** A pre-built image ships Lean 4, `xeus-lean`,
   Jupyter Lab, and all 12 tutorial chapters as runnable notebooks:

   ```bash
   docker run --rm -p 8888:8888 ghcr.io/verilean/hesper-tutorial:latest
   ```

   Open http://localhost:8888 in your browser. Pick `ch00-setup.ipynb` to
   verify everything works, then walk the chapter list in order.

2. **Local install.** You'll need:

   - Lean 4 via [`elan`](https://github.com/leanprover/elan)
   - A C++17 toolchain (`clang` or `g++`)
   - `cmake ≥ 3.16`
   - For Gemma 4: NVIDIA driver + CUDA Toolkit ≥ 12.0

   Then:

   ```bash
   git clone https://github.com/Verilean/hesper.git
   cd hesper
   lake build Hesper
   ```

   First build takes 10–15 minutes (it builds Dawn and Highway from
   source). Subsequent builds are incremental.

## A first sanity check

The smallest thing we can run is a Hesper smoke test that prints the
Lean toolchain version and verifies the WGSL DSL elaborates:

```lean
import Hesper.WGSL.DSL

-- Build a tiny shader expression — this fails at elaboration time if
-- the DSL doesn't load, so reaching the end means everything is fine.
#check fun (x : Exp (.scalar .f32)) => sqrt (x * x + var "one")
```

In a notebook this prints the inferred type. From the command line:

```bash
lake env lean --version
lake build Hesper          # rebuild if needed
lake exe dsl-basics        # print a few DSL example shaders
```

## Switching between chapters

The Markdown sources live at `docs/tutorial/md/Ch##_*.md`. Each chapter
is converted into both `.lean` and `.ipynb` by `xlean-convert`:

```bash
bash docs/tutorial/build-from-md.sh
```

The Docker image runs this at build time, so notebooks are already
present when you start Jupyter Lab.

## What's next

- [Chapter 01 — Lean 4 for ML Engineers](Ch01_LeanForMl.md): just enough
  Lean to read everything that follows.
- [Chapter 01b — Your First Hesper Project](Ch01b_YourFirstProject.md):
  a fresh `lake init` package that uses Hesper as a dependency.
