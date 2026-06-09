# Hesper Tutorial

A twelve-chapter walk through Hesper, written as Markdown masters that
double as runnable Jupyter notebooks via [`xeus-lean`](https://github.com/Verilean/xeus-lean).

## How to read it

### In a browser (Docker, recommended)

```bash
docker run --rm -p 8888:8888 ghcr.io/verilean/hesper-tutorial:latest
# → open http://localhost:8888
```

The image ships Lean 4, the xeus-lean kernel, every chapter
pre-converted to `.ipynb`, and Hesper itself pre-built. First run pulls
~2 GB of image; subsequent runs reuse the layer cache.

### From a checkout (local)

```bash
# 1. Install xlean-convert (one-time setup)
git clone https://github.com/Verilean/xeus-lean
cd xeus-lean && lake build xlean-convert && sudo install -m 0755 \
    .lake/build/bin/xlean-convert /usr/local/bin/

# 2. Convert chapters into .lean + .ipynb
cd /path/to/hesper
bash docs/tutorial/build-from-md.sh

# Outputs land under docs/tutorial/Notebooks/Gen/
```

Generated files are git-ignored — they're regenerated on demand.

### Read-only (just the Markdown)

If you only want to read the prose, every chapter renders correctly on
GitHub:

| # | Chapter | Source |
|---|---|---|
| 00 | Setup | [`Ch00_Setup.md`](md/Ch00_Setup.md) |
| 01 | Lean 4 for ML Engineers | [`Ch01_LeanForMl.md`](md/Ch01_LeanForMl.md) |
| 01b | Your First Hesper Project | [`Ch01b_YourFirstProject.md`](md/Ch01b_YourFirstProject.md) |
| 02 | The Shader DSL — WGSL + ShaderM | [`Ch02_DSL.md`](md/Ch02_DSL.md) |
| 03 | Automatic Differentiation & Verified Ops | [`Ch03_AD.md`](md/Ch03_AD.md) |
| 04 | High-Level API & Tensors | [`Ch04_HighLevelApi.md`](md/Ch04_HighLevelApi.md) |
| 05 | Switching Backends — WebGPU / CUDA | [`Ch05_Backends.md`](md/Ch05_Backends.md) |
| 06 | Proofs — Equivalence & Invariants | [`Ch06_Proofs.md`](md/Ch06_Proofs.md) |
| 07 | BitNet End-to-End | [`Ch07_BitNet.md`](md/Ch07_BitNet.md) |
| 08 | Gemma 4 End-to-End | [`Ch08_Gemma4.md`](md/Ch08_Gemma4.md) |
| 09 | Embedding Hesper in Other Projects | [`Ch09_Embedding.md`](md/Ch09_Embedding.md) |
| 10 | Hesper Architecture | [`Ch10_Architecture.md`](md/Ch10_Architecture.md) |

## Authoring rules

- The `.md` file under `md/` is the source of truth.
- One H1 per chapter (`# Chapter ## — Title`).
- Code blocks are fenced with ` ```lean ` so xlean-convert recognises
  them. Other languages (` ```bash `, ` ```text `) render in
  notebooks as Markdown blocks, not executable cells.
- End each chapter with a `## What's next` section.
- Don't reference an `Examples/` path that doesn't exist — chapters are
  expected to pin to runnable code.

## Why this structure

We borrow the tutorial format from
[Verilean/sparkle](https://github.com/Verilean/sparkle). A single
Markdown master gives you (1) GitHub-rendered prose, (2) a `.lean`
chapter you can `lake build`, and (3) a `.ipynb` cell-stream for
interactive exploration — all without forking the source three ways.
