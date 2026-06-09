# Chapter 05 — Switching Backends (WebGPU / CUDA)

Hesper compiles the same `ShaderM` source to two backends:

- **WebGPU** via Google Dawn (Metal on macOS, Vulkan on Linux, D3D12 on
  Windows).
- **CUDA PTX** via the libcuda driver API.

You pick the backend at compile time. Most code stays
backend-agnostic through the `GPUBackend` typeclass.

## The `GPUBackend` typeclass

```lean
import Hesper.Backend

open Hesper

-- The typeclass is parameterised by a context type β (the "device").
-- Every backend supplies a Buf type and the operations we need:
#check @GPUBackend
-- @GPUBackend : (β : Type) → Type 1

-- Some of the methods that backends must implement:
#check @GPUBackend.allocBuffer
-- allocBuffer : ∀ {β} [GPUBackend β], β → USize → IO _
#check @GPUBackend.executeWithConfig
#check @GPUBackend.readBuffer
```

The full interface is in `Hesper/Backend.lean`. Two concrete instances
ship today: `Hesper.Backend.WebGPU` and `Hesper.Backend.CUDA`. They
share the same `ShaderM` input — the kernels you wrote in Ch02 work
unchanged on both.

## Choosing at build time with `-Kgpu`

The lakefile exposes a `gpu` configuration flag:

```bash
lake -Kgpu=cuda build my-app          # CUDA exe; -lcuda is added
lake -Kgpu=cpu  build my-app          # CPU-only (no CUDA, no -lcuda)
lake -Kgpu=auto build my-app          # default: probe; CUDA if available
```

`auto` means: if `cmake` is present *and* (`libcuda.so` or `nvcc`)
exists, configure for CUDA; otherwise fall back to CPU/WebGPU paths.

The flag controls *only* whether `-lcuda` (the NVIDIA driver shim)
links into the final executable. Dawn / WebGPU is always built — it's
how every shader-target backend works on macOS, Linux, and Windows.

## What changes between backends

| Aspect | WebGPU | CUDA PTX |
|---|---|---|
| Surface language | WGSL | PTX 8.7 (NVIDIA) |
| Tensor cores | via `subgroupMatrix*` → WMMA (limited) | `wmma.mma.sync` directly |
| Async memcpy | command-buffer batching | `cuMemcpyHtoD_v2` + streams |
| Capture / graphs | n/a in Dawn | CUDA Graphs |
| Driver | Dawn / Metal / Vulkan / D3D12 | `libcuda.so.1` |

For most layers the difference is invisible; for the perf-critical
kernels (Q4_K matmul, flash attention, RMSNorm) the CUDA backend
generates code tuned for NVIDIA's dp4a, `cp.async`, and `wmma`
instructions, which WGSL doesn't expose.

## The `gemma4-cuda` entry point

The Gemma 4 model is wired to CUDA only — it depends on `wmma.mma.sync`
and `dp4a`. Build and run it like any CUDA executable:

```bash
lake -Kgpu=cuda build gemma4-cuda
HESPER_CHAT=1 \
  ./.lake/build/bin/gemma4-cuda data/gemma-4-e4b-it-Q4_K_M.gguf "Hello" 30
```

Ch08 walks through everything this entry point does end-to-end.

## What's next

- [Chapter 06 — Proofs](Ch06_Proofs.md): what we mean by "verified",
  with the equivalence layer.
- [Chapter 08 — Gemma 4 End-to-End](Ch08_Gemma4.md): the CUDA path
  applied to a 4 B-parameter LLM.
