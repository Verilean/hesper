# Platform support matrix (release prep — REL-D)

## What we ship

This release targets **Linux x86_64 with NVIDIA GPU + CUDA driver**.

| platform | status | notes |
|---|---|---|
| Linux x86_64 + NVIDIA + CUDA driver ≥ 565 | ✅ tier-1 | full Gemma 4 inference path; 43-test regression suite |
| Linux x86_64 without CUDA            | ⚠️ partial | `lake build hesper` (lib) succeeds; `gemma4-cuda` exe will fail at link time without `-lcuda` |
| Linux ARM64 (e.g. Jetson)            | 🚧 untested | should work in principle (CUDA driver API is portable); nothing in the codebase is x86-specific |
| macOS (Intel or Apple Silicon)       | 🚧 docs only | `cuda_bridge.cpp` is excluded from build (per `buildBridgeIfNeeded`), but Lean FFI symbols `lean_hesper_cuda_*` referenced from `Hesper/CUDA/FFI.lean` will be unresolved at link |
| Windows                              | 🚧 untested | no path is currently exercised |

## Why CUDA-less Linux is "lib-only"

`Hesper.lean` re-exports everything from `Hesper.CUDA.FFI`, so any `lean_exe` that imports anything from `Hesper.*` ends up pulling the CUDA FFI symbols into its `.c.o.export`. Those are defined in `native/cuda_bridge.cpp` and link against the CUDA driver library.

The `lakefile.lean` reflects this:

- `buildBridgeIfNeeded` only builds `libhesper_cuda.a` on non-macOS platforms (`!System.Platform.isOSX`).
- `stdLinkArgs` and `cudaExeArgs` evaluate to `#[]` on macOS, so per-exe link commands omit `-lcuda`.

Verifications run for this release:

```bash
lake check-build           # lakefile parses cleanly
lake build hesper          # the library proper builds
lake build gemma4-cuda     # the headline exe builds + links on Linux+CUDA
bash scripts/regression.sh # 43-test suite passes
```

## What "CUDA-less build" would require to *actually link*

Two options exist; neither is implemented in this release.

### Option A: build-time stub library (quickest)
Provide `native/cuda_bridge_stub.cpp` that defines the same `lean_hesper_cuda_*` entry points but each one returns `IO.Error.userError "CUDA backend unavailable on this platform"`. Add a `buildCudaStubIfNoCuda` Lake target that's selected when `CUDA_HOME` is absent. Then `cudaExeArgs` would point at the stub instead of `libhesper_cuda.a`.

Effort: ~150 LoC C++ + ~30 LoC lakefile. Not done for this release because no user has asked for the macOS / no-CUDA path yet.

### Option B: separate `hesper-core` package (cleaner long-term)
Split the lib into:
- `hesper-core` — pure Lean, no FFI, builds anywhere.
- `hesper-cuda` — `hesper-core` + CUDA FFI + native bridge, gated behind a Lake `if cudaAvailable` test.

`gemma4-cuda` exe depends on `hesper-cuda`; pure-Lean DSL/Circuit examples depend on `hesper-core`. Mac users get the DSL/IR side without the executor.

Effort: ~1-2 sessions of refactoring import graphs. Out of scope for this release.

## What changed in REL-D

1. Audited the lakefile: 81 hardcoded `["./.lake/build/native/libhesper_cuda.a", "-lcuda"]` per-exe entries.
2. Centralised them into a single `cudaExeArgs` def that's empty on macOS:
   ```lean
   def cudaExeArgs : Array String :=
     if System.Platform.isOSX then #[]
     else #["./.lake/build/native/libhesper_cuda.a", "-lcuda"]
   ```
3. Replaced all 81 sites via `replace_all`.
4. Verified `lake build hesper` and `lake build gemma4-cuda` still succeed on Linux.

This change is cosmetic on Linux (compiles to the exact same link line). The benefit is that a future macOS / stub-CUDA work item only needs one more line change (point `cudaExeArgs` at the stub) rather than 81.

## Recommendation for v0.X release

- Document the supported platform in `README.md` as **"Linux x86_64 + NVIDIA GPU + CUDA driver"**.
- Don't ship Mac binaries — refer Mac users to "build the lib + use as DSL frontend" once option A or B lands in a future release.
- Don't claim CI-tested macOS support; the lakefile guards exist but the FFI link path has not been exercised.
