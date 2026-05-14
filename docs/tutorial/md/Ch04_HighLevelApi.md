# Chapter 04 — High-Level API & Tensors

Most users don't write shaders directly — they compose tensors and NN
layers. This chapter shows the layer above `ShaderM`: tensor
descriptors, the matmul / RMSNorm / attention helpers under
`Hesper.Layers.*`, and the runtime under `Hesper.Compute`.

## Tensors

The smallest unit of typed shape information is `TensorDesc`:

```lean
import Hesper.Tensor.Types

open Hesper.Tensor

#check @TensorDesc.matrix
-- TensorDesc.matrix : Nat → Nat → optParam DType .f32 → TensorDesc

#eval (TensorDesc.matrix 768 768).sizeBytes
-- 2359296   (768*768*4 bytes for f32)
```

`TensorDesc` carries shape and dtype together so the compute layer can
allocate the right buffer size without separate book-keeping.

## Configuring a matmul

Hesper ships pre-fused matmul kernels for the common quantised formats
(`Q4_K`, `Q6_K`). Each one takes a `Config` describing the (inDim,
outDim) shape and the quantisation parameters:

```text
-- See Hesper/Layers/Linear.lean
let cfg : Linear.Config := {
  inDim  := 2560
  outDim := 2560
  -- + scale / block parameters specific to the quant type
}

-- Then build a ShaderM kernel for this shape:
let kernel : ShaderM Unit := Linear.fusedQ4KMLinearKernel cfg
```

The kernel takes the input as a buffer of Q8_1-quantised activations
and the weight as `block_q4_K` blocks (the GGUF layout). Higher-level
"Tensor" objects are still under construction — for now you wire
buffers up through `ShaderM` and the FFI directly.

## Pre-built layers

`Hesper.Layers.*` contains the standard transformer building blocks:

| Module | Highlights |
|---|---|
| `Linear` | Q4_K / Q6_K dp4a and MMQ tile kernels (decode + prefill) |
| `Attention` | Flash-attention V11 (sub-warp partition, K-parallel, split-K) |
| `RMSNorm` | Fused RMSNorm + Q8_1 quantise (eliminates the round-trip) |
| `Embedding` | Token embedding lookup |
| `Activation` | `gelu`, `geluQuick`, `relu`, `silu` |

Each layer's source has a short docstring at the top explaining when
to use which kernel variant.

## Composing a model

For an end-to-end example see:

- `Hesper/Models/BitNet.lean` — every BitNet b1.58 layer wired together.
- `Hesper/Models/Gemma4.lean` — Gemma 4 E4B forward + decode loop.
- `Examples/BitNetComplete.lean` — driver that runs BitNet inference.
- `Examples/Gemma4CUDA.lean` — driver that runs Gemma 4 inference.

We walk those drivers in Ch07 and Ch08.

## Operator fusion via Circuit DSL

When you compose many layers, the kernel-fusion layer under
`Hesper.Circuit` rewrites the graph before emitting shaders. The
fusion passes (pointwise, reduce-into-quantise, matmul-epilogue,
scatter, etc.) are responsible for the ~10× dispatch-count reduction
documented in the CHANGELOG.

See [`docs/circuit-dsl-tutorial.md`](../../circuit-dsl-tutorial.md)
for a walkthrough of the IR and how each pass works.

## What's next

- [Chapter 05 — Switching Backends](Ch05_Backends.md): the same kernel
  on CUDA instead of WebGPU.
- [Chapter 07 — BitNet End-to-End](Ch07_BitNet.md): every piece in this
  chapter applied to a real inference engine.
