# Contributing to Hesper

Hesper is part of the **Verilean** organization's effort to bring verified
computing to GPUs. Patches, bug reports, and discussions are welcome.

## Getting started

1. Fork the repository and create a feature branch off `main`.
2. Run the test suite to make sure your environment is healthy:
   ```bash
   lake build test-all
   ./.lake/build/bin/test-all
   ```
3. Make your changes — see "Code organization" below.
4. Add tests for new features (the suites under `Tests/` are LSpec-based).
5. Open a pull request describing what you changed and why.

## Code organization

```
Hesper/
├── Hesper/                 # Core library
│   ├── WGSL/              # Type-safe shader DSL (Types, Exp, DSL, Monad, CodeGen)
│   ├── WebGPU/            # WebGPU bindings (Device, Buffer, Shader, Pipeline)
│   ├── CUDA/              # CUDA PTX backend (Codegen, FFI, Execute)
│   ├── Layers/            # Reusable NN layers (Linear, Attention, RMSNorm, …)
│   ├── Models/            # End-to-end models (BitNet, Gemma 4)
│   ├── Compute.lean       # High-level compute API
│   └── Simd*.lean         # Portable SIMD (Google Highway)
├── Examples/               # Runnable demos, grouped by topic
├── Tests/                  # Unit and integration tests
├── native/                 # C++ FFI bridge (Dawn/WebGPU + CUDA driver shim)
├── c_src/                  # SIMD CPU implementations
├── docs/
│   ├── tutorial/          # User-facing tutorial (Markdown master → .ipynb)
│   └── research/          # Internal R&D notes, archived
└── lakefile.lean           # Lake build configuration
```

## Testing guidelines

- **FFI changes**: run the FFI suite via `lake build test-ffi`.
- **DSL changes**: run shader-generation tests (`Tests.WGSLDSLTests`).
- **GPU kernels**: test on real hardware — compilation alone is not enough.
- **SIMD changes**: ARM64 (NEON) and x86_64 (AVX2) targets, where available.
- **Cross-platform**: macOS (Metal), Linux (Vulkan), Windows (D3D12/Vulkan).

## Reporting issues

- Use [GitHub Issues](https://github.com/Verilean/hesper/issues) for bugs.
- Use [GitHub Discussions](https://github.com/Verilean/hesper/discussions)
  for design questions and broader topics.

## Key files for new contributors

- `native/bridge.cpp` — FFI boundary; all Lean ↔ C++ data conversion.
- `Hesper/WGSL/Monad.lean` — ShaderM monad used to construct kernels.
- `Hesper/WGSL/Execute.lean` — Compiles ShaderM → WGSL and runs on the GPU.
- `Hesper/CUDA/Execute.lean` — Same path for the PTX/CUDA backend.
- `Tests/FFIBoundaryTests.lean` — Reference test for FFI data conversion.

## License

By contributing you agree that your contributions will be licensed under the
same terms as the project (Apache-2.0; see `LICENSE`).
