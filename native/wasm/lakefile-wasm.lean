-- WASM-only Lake build override for Hesper.
--
-- The production `lakefile.lean` pulls in LSpec, a doc-gen4 dep, the
-- `nativeDeps` target (Dawn / Highway / CUDA bridge), and several
-- `lean_exe` targets that require Dawn/CUDA FFI.  None of that can run
-- inside a WebAssembly host — emdawnwebgpu provides only the `wgpu*`
-- C API forwarded to `navigator.gpu`, and we link the FFI shims via
-- `native/wasm/build-wasm.sh` which emits a separate
-- `libhesper_wasm.a` static archive instead.
--
-- This file is COPIED on top of `lakefile.lean` by `build-wasm.sh`
-- just before `lake build`.  The original lakefile (plus
-- `lakefile.toml` if any) is preserved on disk and restored on exit,
-- including on failure.
--
-- Phase 1 keeps just the pure-Lean WGSL DSL building.  As we widen
-- coverage (CUDA backend, NN layers, etc.) the build-wasm.sh caller
-- will pass additional module targets.

import Lake
open Lake DSL

package «Hesper» where

lean_lib «Hesper» where
  globs := #[.submodules `Hesper]

-- Lib only — no executables on WASM.
