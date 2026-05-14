# Chapter 10 — Hesper Architecture

This chapter is the wide-angle view. We've gone from the DSL (Ch02) to
real models (Ch07, Ch08); now let's see how the layers stack and where
to look when something goes wrong.

## The five layers

```
┌────────────────────────────────────────────────────────────────┐
│  L5  Models                                                     │
│      Hesper.Models.BitNet, Hesper.Models.Gemma4                 │
│      end-to-end inference + training loops                      │
├────────────────────────────────────────────────────────────────┤
│  L4  High-level API                                             │
│      Hesper.Layers.*, Hesper.AD, Hesper.Tensor                  │
│      composable layers + reverse-mode autodiff + tensors        │
├────────────────────────────────────────────────────────────────┤
│  L3  Circuit DSL (kernel fusion)                                │
│      Hesper.Circuit                                             │
│      Prim.pointwise / Prim.reduce / Prim.matmul / Prim.scatter  │
│      + fusion passes (mergeSameDispatch, fusePointwise, …)      │
├────────────────────────────────────────────────────────────────┤
│  L2  ShaderM IR                                                 │
│      Hesper.WGSL.Monad                                          │
│      imperative kernel construction: buffers, loops, barriers   │
├────────────────────────────────────────────────────────────────┤
│  L1  Exp (typed shader expressions)                             │
│      Hesper.WGSL.{Types, Exp, DSL}                              │
│      type-safe arithmetic, math, vector, matrix                 │
├────────────────────────────────────────────────────────────────┤
│  L0  Backends                                                   │
│      Hesper.WGSL.CodeGen     (→ WGSL → Dawn → Metal/Vulkan/D3D12)│
│      Hesper.CUDA.CodeGen     (→ PTX → libcuda → NVIDIA GPU)     │
└────────────────────────────────────────────────────────────────┘
```

Each layer is the public interface to the one above it. You can drop
down to a lower layer when you need control — model code happily mixes
L4 layers with hand-written L2 kernels.

## The IR triangle

Two intermediate representations exist:

- **`ShaderM`** (L2) is the *imperative* IR. A `ShaderM Unit` is a
  block of statements over typed buffers. It's the natural form to
  hand-write a kernel.
- **`Circuit`** (L3) is the *graph* IR. A `Circuit` is a DAG of `Prim`
  nodes; each node has a known forward semantics and a fusion pass
  rewrites the graph before lowering to ShaderM.

Both lower to the same backend printers (L0). The Circuit DSL lets the
compiler fuse safely; ShaderM lets you write exactly what you want.

## The fusion story

The decode path used to issue ~200 kernel dispatches per token. After
fusion it issues ~25, all captured inside one CUDA Graph. The wins came
from chaining the following passes:

| Pass | What it merges |
|---|---|
| `fusePointwise` | Adjacent pointwise (elementwise) ops |
| `fuseReduceIntoQuantize` | RMSNorm → Q8_1 quantize |
| `fuseMatmulEpilogue` | matmul + pleScale / postNormAdd / activation |
| `mergeSameDispatch` | Identical dispatches scheduled together |
| `fuseScatter` | RoPE-K → KV cache write |
| `fuseWriteDestination` | scatter + view + writeSlice |

Read the passes in `Hesper/Circuit/Passes/*.lean`. Each is < 200 lines
and proves its rewrite preserves semantics.

## The backend split

```
ShaderM Unit
   │
   ├──► WGSL printer ──► WGSL string ──► Dawn ──► driver
   │                                        │
   │                                        ├── Metal (macOS)
   │                                        ├── Vulkan (Linux + Win)
   │                                        └── D3D12 (Windows)
   │
   └──► PTX printer  ──► PTX string  ──► libcuda
                                            │
                                            └── NVIDIA driver → SASS
```

The two backends share `Exp`, the type system, and most of `ShaderM`.
The differences are concentrated in:

- `Hesper/WGSL/CodeGen.lean` vs `Hesper/CUDA/CodeGen.lean` — the printers.
- A few backend-specific `Exp` constructors (`dot4I8Packed`,
  `subgroupMatrix*`, `cp.async`) that are no-ops on the side that doesn't
  support them.

## Verification surfaces

| Layer | What's checked |
|---|---|
| L1 (Exp) | Type system catches scalar/vector/matrix mismatches |
| L2 (ShaderM) | Buffer-shape/array-bound constraints in types |
| L3 (Circuit) | Fusion passes carry equivalence proofs |
| L4 (AD) | Every `Differentiable` instance has a correctness theorem |
| L5 (Models) | Numerical bit-parity tests against `llama.cpp` |

The L5 tests don't replace the higher proofs — they protect against
issues outside the proof scope (driver bugs, JIT differences, race
conditions).

## Where to start reading the code

The fastest tour of the architecture:

1. `Hesper.lean` — the top-level re-exports.
2. `Hesper/WGSL/Exp.lean` — the type-safe expression AST.
3. `Hesper/WGSL/Monad.lean` — `ShaderM` and the builder API.
4. `Hesper/Layers/RMSNorm.lean` — a small, self-contained layer.
5. `Hesper/Layers/FlashAttention.lean` — the production attention kernel.
6. `Hesper/Circuit/Lowering.lean` — Circuit → ShaderM.
7. `Hesper/Models/Gemma4/Gemma4.lean` — full model assembly.

After that, follow imports.

## Architecture decisions worth knowing

- **No hidden optimizer.** The DSL → backend path is deterministic;
  what you write is what runs (modulo backend-side ptxas / SPIR-V
  optimisation, but those are stable across versions).
- **Fusion is opt-in by *not* writing the unfused chain.** If you call
  `gateProj` then `relu` then `upProj`, the Circuit DSL will fuse
  them — and you can verify by dumping the generated WGSL/PTX.
- **No global state.** Devices and buffers are explicit; there's no
  implicit "current device" like PyTorch's `torch.cuda.current_device()`.
- **Lean is the build system.** Lake compiles the library, schedules
  native deps, runs tests, generates docs. No CMake-on-CMake.

## End of the tutorial

That's the tour. From here:

- Skim `Hesper/Models/Gemma4/Gemma4.lean` — it uses every layer above.
- Try modifying `Hesper/Layers/FlashAttention.lean` and re-running
  `lake exe gemma4-cuda` to see how the change propagates.
- Read [`docs/research/`](../../research/) if you want the messy history
  behind each kernel's current shape — every shortcut has a debug log.

Happy hacking.
