# Hesper

**Write GPU programs in Lean 4. Prove them correct. Run on WebGPU or CUDA.**

[![CI](https://github.com/Verilean/hesper/actions/workflows/ci.yml/badge.svg)](https://github.com/Verilean/hesper/actions/workflows/ci.yml)

> [!IMPORTANT]
> **Alpha software.** APIs, verification features, and the compiler are under
> active development and subject to breaking changes. Current focus: BitNet,
> Gemma 4, and the verified-AD layer.

Hesper is a verified GPU programming framework. You write type-safe shaders
and high-level tensor code in Lean 4, prove correctness with the same
language, and run the result on WebGPU (Metal / Vulkan / D3D12) or CUDA PTX.

```lean
import Hesper.WGSL.DSL

-- Type-safe shader expressions, checked at Lean elaboration time:
let x : Exp (.scalar .f32) := var "x"
let y : Exp (.scalar .f32) := var "y"
let r := sqrt (x * x + y * y)              -- generates: sqrt(x*x + y*y)

-- let wrong := x + (var "i" : Exp (.scalar .i32))  -- ✗ type error
```

## The Hesper Way

| Layer | What it gives you |
|---|---|
| **DSL** | Type-safe WGSL/ShaderM expressions — wrong-type mixes are Lean errors |
| **Verified AD** | Reverse-mode autodiff with `Differentiable` typeclass; gradients are *proven* correct |
| **Multi-backend** | Same DSL lowers to WGSL (Dawn) or PTX (CUDA driver) — pick at compile time |
| **Kernel fusion** | Circuit DSL fuses pointwise + reduce + matmul into one dispatch |
| **Verified ops** | `VerifiedOpFusion` proves fused kernels equal the unfused spec |

## IP Catalog

| Model / IP | Backend | Status | Entry |
|---|---|---|---|
| **BitNet b1.58 (2 B)** | WebGPU | ✅ 125 TPS on M4 Max | `lake exe bitnet-complete` |
| **Gemma 4 (E4B)** | CUDA PTX | ✅ ~100 TPS on RTX 4070 Ti | `lake exe gemma4-cuda` |
| **LoRA fine-tuning** | WebGPU | ✅ Alpaca-style instruction tuning | `lake exe lora-train` |
| **Verified AD** | WebGPU | ✅ 11 executable parity tests | `lake exe ad-demo` |
| **Tetris (graphics)** | WebGPU | ✅ Interactive | `lake exe tetris` |

See [`docs/CHANGELOG.md`](docs/CHANGELOG.md) for the release timeline.

## Quick Start

### Option 1 — Tutorial Docker image (recommended)

A pre-built image with Lean 4, xeus-lean, Jupyter Lab, and all 12 tutorial
chapters as runnable notebooks:

```bash
docker run --rm -p 8888:8888 ghcr.io/verilean/hesper-tutorial:latest
# → open http://localhost:8888 in your browser
```

### Option 2 — Local build

```bash
git clone https://github.com/Verilean/hesper.git
cd hesper
lake build Hesper             # native deps (Dawn, Highway) build on demand
lake exe bitnet-complete      # first inference run
```

Prerequisites: Lean 4 (via [`elan`](https://github.com/leanprover/elan)), a
C++17 toolchain, CMake ≥ 3.16. For Gemma 4 on CUDA: NVIDIA driver + CUDA
Toolkit. Full setup instructions: [`docs/tutorial/md/Ch00_Setup.md`](docs/tutorial/md/Ch00_Setup.md).

## Documentation

The tutorial is the canonical entry point. Each chapter is a Markdown master
that's also rendered as a runnable Jupyter notebook via `xeus-lean`.

| # | Chapter | Source |
|---|---|---|
| 00 | Setup | [`Ch00_Setup.md`](docs/tutorial/md/Ch00_Setup.md) |
| 01 | Lean 4 for ML Engineers | [`Ch01_LeanForMl.md`](docs/tutorial/md/Ch01_LeanForMl.md) |
| 01b | Your First Hesper Project | [`Ch01b_YourFirstProject.md`](docs/tutorial/md/Ch01b_YourFirstProject.md) |
| 02 | The Shader DSL — WGSL + ShaderM | [`Ch02_DSL.md`](docs/tutorial/md/Ch02_DSL.md) |
| 03 | Automatic Differentiation & Verified Ops | [`Ch03_AD.md`](docs/tutorial/md/Ch03_AD.md) |
| 04 | High-Level API & Tensors | [`Ch04_HighLevelApi.md`](docs/tutorial/md/Ch04_HighLevelApi.md) |
| 05 | Switching Backends — WebGPU / CUDA | [`Ch05_Backends.md`](docs/tutorial/md/Ch05_Backends.md) |
| 06 | Proofs — Equivalence & Invariants | [`Ch06_Proofs.md`](docs/tutorial/md/Ch06_Proofs.md) |
| 07 | BitNet End-to-End | [`Ch07_BitNet.md`](docs/tutorial/md/Ch07_BitNet.md) |
| 08 | Gemma 4 End-to-End | [`Ch08_Gemma4.md`](docs/tutorial/md/Ch08_Gemma4.md) |
| 09 | Embedding Hesper in Other Projects | [`Ch09_Embedding.md`](docs/tutorial/md/Ch09_Embedding.md) |
| 10 | Hesper Architecture | [`Ch10_Architecture.md`](docs/tutorial/md/Ch10_Architecture.md) |

Other user-facing docs:

- [`docs/LORA_FINETUNING.md`](docs/LORA_FINETUNING.md) — BitNet LoRA training guide
- [`docs/VERIFIED_AD.md`](docs/VERIFIED_AD.md) — Verified-AD design notes
- [`docs/circuit-dsl-tutorial.md`](docs/circuit-dsl-tutorial.md) — Kernel-fusion DSL deep-dive
- [`docs/research/`](docs/research/) — Archived internal R&D notes (not maintained)

## How It Works

```
┌──────────────────────────────────────────────────────────────┐
│  Lean 4 code                                                  │
│  ─ Type-safe shader DSL  ─ Tensor ops  ─ Formal proofs        │
└────────────────────────────┬──────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  ShaderM IR  (a small imperative shader language in Lean)     │
└──────────────┬────────────────────────────┬───────────────────┘
               ▼                            ▼
┌─────────────────────────┐    ┌──────────────────────────────┐
│  WGSL backend           │    │  CUDA PTX backend             │
│  (Dawn native)          │    │  (libcuda driver API)         │
└──────────┬──────────────┘    └─────────────┬─────────────────┘
           ▼                                 ▼
   Metal / Vulkan / D3D12               NVIDIA GPU
```

The same DSL emits two surface languages. Pick the backend at compile time
(`-Kgpu=cuda` for CUDA exes, default WebGPU otherwise).

## Examples

```bash
lake exe bitnet-complete             # BitNet b1.58 inference end-to-end
lake exe gemma4-cuda data/...gguf "Hello"   # Gemma 4 on CUDA
lake exe tetris                      # Interactive WebGPU game
lake exe dsl-basics                  # Walk through the type-safe DSL
lake exe matmul-simple               # Smallest GPU matmul
```

The `Examples/` directory has ~90 runnable demos organized by topic
(`DSL/`, `Compute/`, `Graphics/`, `MachineLearning/`, …).

## Project Structure

```
Hesper/
├── Hesper/             # Core library (DSL, WebGPU, CUDA, Layers, Models)
├── Examples/           # ~90 runnable demos
├── Tests/              # LSpec-based test suites
├── native/             # C++ FFI bridge (Dawn + CUDA driver shim)
├── c_src/              # Portable SIMD (Google Highway)
├── docs/
│   ├── tutorial/      # User-facing tutorial (Markdown master)
│   └── research/      # Internal R&D archive
└── lakefile.lean       # Lake build configuration
```

## Testing

```bash
lake build test-all && ./.lake/build/bin/test-all
```

The same suite runs in CI across Linux + Vulkan, macOS + Metal, Windows +
D3D12, and a CPU-only fallback path.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). In short: fork, branch, run the
test suite, open a PR. Bug reports and design discussions go to
[GitHub Issues](https://github.com/Verilean/hesper/issues) /
[Discussions](https://github.com/Verilean/hesper/discussions).

## License

Apache-2.0. See [`LICENSE`](LICENSE).

## Acknowledgments

- [Lean 4](https://lean-lang.org/) — the language we're built on.
- [Google Dawn](https://dawn.googlesource.com/dawn) — WebGPU native runtime.
- [Google Highway](https://github.com/google/highway) — portable SIMD.
- [Verilean / sparkle](https://github.com/Verilean/sparkle) — sister project
  bringing the same verification ideas to hardware description.

## Community

- **Discord**: [https://discord.gg/94Xueve8WD](https://discord.gg/94Xueve8WD)
  — design discussion, weekly progress threads, beginner Q&A.
