# Chapter 05 — Switching Backends (WebGPU / CUDA)

Hesper compiles the same `ShaderM` source to two backends:

- **WebGPU** via Google Dawn (Metal on macOS, Vulkan on Linux, D3D12 on
  Windows).
- **CUDA PTX** via the libcuda driver API.

You pick the backend at compile time. Most code is backend-agnostic
through the `GPUBackend` typeclass.

## The `GPUBackend` typeclass

```lean
class GPUBackend (m : Type → Type) where
  Device   : Type
  Buffer   : Type
  Kernel   : Type
  allocBuffer  : Device → Nat → m Buffer
  writeBuffer  : Device → Buffer → ByteArray → m Unit
  readBuffer   : Device → Buffer → m ByteArray
  compileKernel : Device → ShaderM Unit → m Kernel
  dispatchKernel : Device → Kernel → Args → m Unit
```

`Hesper.Compute` parameterises every high-level API over `GPUBackend`,
so writing model code rarely mentions the backend at all.

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

## Backend selection at runtime

Within a CUDA build you can still target either backend per device:

```lean
import Hesper.WebGPU.Device
import Hesper.CUDA.Device

def main : IO Unit := do
  -- CUDA path (only available in a -Kgpu=cuda exe):
  let cudaDev ← Hesper.CUDA.Device.create
  let cudaKernel ← cudaDev.compileKernel myKernel

  -- WebGPU path is always available:
  let wgpuDev ← Hesper.WebGPU.Device.create
  let wgpuKernel ← wgpuDev.compileKernel myKernel
  ...
```

## What changes between backends

| Aspect | WebGPU | CUDA PTX |
|---|---|---|
| Surface language | WGSL | PTX 8.7 (NVIDIA) |
| Tensor cores | via `subgroupMatrix*` → WMMA (limited) | `wmma.mma.sync` directly |
| Async memcpy | `writeBuffer`/`readBuffer` | `cuMemcpyHtoD_v2` + streams |
| Capture / graphs | command-buffer batching | CUDA Graphs |
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
