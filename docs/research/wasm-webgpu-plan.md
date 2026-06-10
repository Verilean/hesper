# Hesper Wasm + WebGPU plan (libhesper_wasm.a)

## Status

- xeus-lean side: **wasm + Memory64 runtime, extern dlsym via generated symbol table, hesper-wasm/build-wasm.sh that bakes `Hesper.WGSL.DSL` into a JupyterLite VFS.** All working today.  See `../xeus-lean/WASM_BUILD.md`.
- Hesper side: **only pure-Lean is wasm-ready** so far (WGSL code generation, type system, ShaderM monad).  Anything that calls `lean_hesper_*` extern (init, getDevice, createBuffer, dispatch, …) currently dies.

This document is the design for closing that gap.

## Goal

End-to-end: in a JupyterLite notebook cell open at <https://verilean.github.io/hesper/>,
`#eval` of `runGpu` from Ch04 (`parallelForDSL device scaleByThousand input`) returns
`#[0, 1000, …, 9000]` against the browser's WebGPU.

## Why this is tractable

`native/bridge.cpp` is **already written against Dawn's C `wgpu*` API**, which is the
same API surface emdawnwebgpu exposes inside an Emscripten module — emdawnwebgpu is
specifically Dawn's wasm port that forwards `wgpu*` calls to `navigator.gpu`.  Concretely:

- `wgpuInstanceRequestAdapter`, `wgpuAdapterRequestDevice`,
  `wgpuDeviceCreateBuffer`, `wgpuQueueWriteBuffer`,
  `wgpuDeviceCreateShaderModule`, `wgpuDeviceCreateComputePipeline`,
  `wgpuDeviceCreateBindGroup{,Layout}`,
  `wgpuCommandEncoderBeginComputePass`,
  `wgpuComputePassEncoder{Set{Pipeline,BindGroup},DispatchWorkgroups,End}`,
  `wgpuQueueSubmit`, `wgpuBufferMapAsync`, `wgpuBufferGetConstMappedRange`,
  `wgpuBufferUnmap` — **every one of these resolves to a JS-backed `navigator.gpu`
  call under emdawnwebgpu**.
- We do not need to write a single line of JavaScript.

The pieces we **do** have to handle differently:

1. **GLFW window / surface creation** — only meaningful in `glfw_bridge.cpp` and only
   used by Hesper for the demo windows.  WebGPU compute in `parallelForDSL` does not
   require a surface (no rendering, just dispatch).  Wasm build drops glfw_bridge.cpp
   entirely; surface support arrives later if/when a notebook cell wants to render to
   a canvas.
2. **CUDA bridge** — already stubbed everywhere outside Linux+CUDA.  Wasm uses the
   existing `cuda_bridge_stub.cpp`.
3. **Pthreads / promise-based map** — emdawnwebgpu's `wgpuBufferMapAsync` returns
   asynchronously and is processed by `emscripten_sleep` /
   `wgpuInstanceProcessEvents`.  bridge.cpp's `lean_hesper_map_buffer_read` already
   uses callback-driven mapping (`bufferMapCallback`) and a manual tick loop, so the
   structure is right; we just need `wgpuInstanceProcessEvents` between ticks so
   emdawnwebgpu can drain the JS promise queue.
4. **`Hesper.init`'s `dawnProcSetProcs`** — Dawn's native build uses a function-pointer
   table populated at runtime.  emdawnwebgpu links `wgpu*` symbols *directly* and does
   not have a procs table; under `__EMSCRIPTEN__` we skip the call.  This is the only
   `#ifdef` likely required in bridge.cpp itself.
5. **`@[extern]` discovery** — xeus-lean's `gen_wasm_symtab.sh` already finds every
   `lean_hesper_*` symbol in any `.a` we link.  Putting bridge.cpp's object code into
   `libhesper_wasm.a` and `--whole-archive`-linking it into xlean is enough.

## What we build, and where it lives

```
hesper/
  native/
    CMakeLists.txt         ← add EMSCRIPTEN branch
    bridge.cpp             ← unchanged, except a handful of #ifdef __EMSCRIPTEN__ guards
    glfw_bridge.cpp        ← excluded from wasm build
    cuda_bridge_stub.cpp   ← already linked from non-Linux + non-CUDA paths; reuse for wasm
    wasm/
      build-libhesper-wasm.sh   ← NEW.  Invokes emcmake / emmake.  Mirrors
                                  xeus-lean/hesper-wasm/build-wasm.sh's structure
                                  but produces libhesper_wasm.a instead of .olean.
      README.md                 ← short pointer to this design doc
```

The output is **one static archive** `libhesper_wasm.a` containing:

- `bridge.o`              (Lean ↔ Dawn FFI, used by every per-module Lean dynlib via
                           `@[extern "lean_hesper_*"]`)
- `cuda_bridge_stub.o`    (so the `lean_hesper_cuda_*` extern names that Lean modules
                           may reference still resolve to error-returning stubs)

…linked against emdawnwebgpu's `libwebgpu_dawn.a` (the wasm variant; see
<https://dawn.googlesource.com/dawn/+/refs/heads/main/docs/quickstart-emdawnwebgpu.md>).
The xeus-lean build then `--whole-archive`-links `libhesper_wasm.a` into `xlean`, the
same way it currently does for `libsparkle_wasm.a`.

## Phases

### M0 — sanity (1 session)

- Run `../xeus-lean/hesper-wasm/build-wasm.sh ./xeus-lean/hesper ./xeus-lean/hesper-wasm/build`
  against the current Hesper main.  Confirm the WGSL codegen modules build and the
  patched Lean-4.28 String.Slice fix still applies cleanly.
- Open a JupyterLite cell, `import Hesper.WGSL.DSL`, eval
  `(generateWGSLSimple scaleByThousand).take 300`.  This should already work and
  proves Phase 1 is not regressed by anything we landed in main.

### M1 — `libhesper_wasm.a` skeleton, no GPU yet (1–2 sessions)

- Add `if (EMSCRIPTEN)` branch to `native/CMakeLists.txt`:
  - Drop the Dawn FetchContent / `find_library(COCOA_LIB)` / `vulkan` / `glfw_bridge.cpp`
    paths.
  - Produce `libhesper_wasm.a` containing only `bridge.cpp` + `cuda_bridge_stub.cpp`.
  - Add `-sUSE_WEBGPU=0 -sUSE_EMDAWNWEBGPU=1` (or whatever the current emdawnwebgpu
    integration name is — Dawn's docs change; check at impl time).
- Add `#ifdef __EMSCRIPTEN__` guards around the only two known-incompatible call
  sites in bridge.cpp:
  - `dawnProcSetProcs(...)` in `lean_hesper_init` → skip on wasm.
  - The Linux/macOS `glfw_bridge.cpp` declarations → behind `HESPER_HAS_GLFW`,
    define only on native.
- `hesper/native/wasm/build-libhesper-wasm.sh` does:
  ```
  emcmake cmake -S native -B native/build-wasm -DEMDAWNWEBGPU_DIR=...
  emmake make -C native/build-wasm -j
  ```
- Goal at end of M1: `libhesper_wasm.a` exists, contains the right symbols
  (`llvm-nm | grep lean_hesper_init` succeeds), but we have not yet wired it into
  xeus-lean's xlean.

### M2 — move build inputs onto the hesper side, prep xeus-lean PR (1 session)

Direction change vs. M0/M1 design: rather than have xeus-lean own the build
recipe (lakefile-wasm + patches + builder script under
`xeus-lean/hesper-wasm/`), hesper itself ships them under `native/wasm/`.
xeus-lean then only needs to call one script and link the result.

What landed this milestone (hesper side):

- `native/wasm/lakefile-wasm.lean` — WGSL-only Lake build that drops LSpec /
  nativeDeps / Tests / Examples.
- `native/wasm/build-wasm.sh <out-dir> [target ...]` — single script that
  produces both halves in one staging directory:
  - `<out-dir>/Hesper{,/**}.{olean,olean.server,olean.private,ir,ilean}`
    via `lake build` against the WASM-only lakefile.
  - `<out-dir>/lib/libhesper_wasm.a` via the existing
    `native/CMakeLists.txt` EMSCRIPTEN branch (emcc).
  - Env knobs: `LEAN_TOOLCHAIN_OVERRIDE` (pin Lean version for xeus-lean
    integration), `SKIP_LIB` / `SKIP_OLEAN` (phase-only runs).
- `native/wasm/build-libhesper-wasm.sh` removed — replaced by the unified
  script above.

What's left for the xeus-lean PR (follow-up, separate session):

- Delete `xeus-lean/hesper/` submodule.
- Delete `xeus-lean/hesper-wasm/` entirely (build-wasm.sh +
  lakefile-wasm.lean + patches/). xeus-lean no longer owns hesper build
  recipes.
- Add `-DHESPER_WASM_DIR=<staging-dir>` option to xeus-lean's
  `CMakeLists.txt`. If set, mirror the sparkle `--whole-archive` pattern
  (`CMakeLists.txt:150-240`) for `${HESPER_WASM_DIR}/lib/libhesper_wasm.a`
  and stage `${HESPER_WASM_DIR}/Hesper*` into `olean_assets`.
- Update `xeus-lean`'s CI to run `hesper/native/wasm/build-wasm.sh
  ${runner.temp}/hesper-wasm` (with the kernel's Lean toolchain pinned via
  `LEAN_TOOLCHAIN_OVERRIDE`), then pass `-DHESPER_WASM_DIR=...` to the wasm
  configure step.

Goal at end of M2's xeus-lean PR: in a JupyterLite cell, `import Hesper.WGSL.DSL`
elaborates and `#eval Hesper.init` returns an `IO.Error` from the stub (which
proves the link path is correct end-to-end). The real `wgpu*` API wiring is M3.

### M3 — first `parallelForDSL` end-to-end (1–2 sessions)

- Re-test the Ch04 cell from the tutorial:
  ```lean
  def runGpu : IO (Array Float) := do
    let inst   ← Hesper.init
    let device ← getDevice inst
    parallelForDSL device scaleByThousand input
  #eval runGpu
  ```
- Expected likely failure modes:
  - `wgpuInstanceProcessEvents` not pumped between buffer-map ticks → tweak
    `lean_hesper_map_buffer_read`'s wait loop.
  - emdawnwebgpu's adapter selection differs (no `getMaxLimits` equivalent on
    `Adapter`).  Trim the limits query for wasm.
  - JavaScript-side async timing of `mapAsync` returning before the read-back
    callback fires.  Document in this file what we observe.
- Goal at end of M3: `#eval runGpu` returns `#[0, 1000, …, 9000]` in the browser.

### M4 — Ch11 GPU stage, BitNet (later, scope dependent)

Once M3 lands, the Ch11 California-Housing GPU training and BitNet single-token
forward should "just work" given enough VRAM.  No new infrastructure expected —
only model loading (GGUF file fetch / IndexedDB cache, ~5GB) becomes the next
real engineering problem.  Out of scope for this doc.

## Differentiation we'll be able to claim once M3 lands

Single ShaderM DSL program → 4 backends (Linux Vulkan / macOS Metal / NVIDIA CUDA /
browser WebGPU) with **bit-identical output** verified by the existing parity tests
(extended with a wasm column).  No other Lean-4 framework currently offers any of
the four targets, let alone all four from one source.

See `docs/tutorial/md/` for where this story should appear once M3 is real:

- A new tutorial chapter `Ch12_BrowserBackend.md` (4-backend cross-compile + parity
  table) is the natural home.
