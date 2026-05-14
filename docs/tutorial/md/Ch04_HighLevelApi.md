# Chapter 04 — High-Level API & Tensors

Most users don't write shaders directly — they compose tensors and NN
layers. This chapter shows the layer above `ShaderM`: `Tensor`, the
standard NN modules under `Hesper.Layers.*`, and the `Hesper.Compute`
runtime.

## Tensors

```lean
import Hesper.Tensor

-- A shape is a list of named dimensions:
abbrev Shape := List Dim

def t : Tensor [.batch 4, .seq 16, .dim 768] := ...
```

Shapes are dependent types — operations check at elaboration time that
inputs are compatible:

```lean
def matmul (a : Tensor [.batch B, .row M, .col K])
           (b : Tensor [.batch B, .row K, .col N]) :
           Tensor [.batch B, .row M, .col N] := ...

-- Mismatching K dimensions is a Lean error, not a runtime crash.
```

## Pre-built layers

`Hesper.Layers.*` provides the standard transformer building blocks:

| Module | Layers |
|---|---|
| `Linear` | `Linear`, `Linear.forward`, `Linear.forwardBatch` |
| `Attention` | `Attention.flash`, `Attention.scaledDot` |
| `RMSNorm` | `RMSNorm`, `RMSNorm.fused` |
| `Embedding` | `Embedding`, `embedTokens` |
| `Activation` | `gelu`, `relu`, `silu`, `geluQuick` |

Each layer is `Differentiable`, so composing them into a model gives
you AD for free.

## A small MLP

```lean
import Hesper.Compute
import Hesper.Layers.Linear
import Hesper.Layers.Activation

structure MLP where
  fc1 : Linear (inDim := 256) (outDim := 1024)
  fc2 : Linear (inDim := 1024) (outDim := 256)

def MLP.forward (m : MLP) (x : Tensor [.batch B, .dim 256]) :
    Tensor [.batch B, .dim 256] :=
  m.fc2.forward (relu (m.fc1.forward x))
```

Build it, allocate buffers, run forward:

```lean
def main : IO Unit := do
  let dev ← Hesper.Device.create
  let mlp : MLP := { fc1 := Linear.random ..., fc2 := Linear.random ... }
  let x : Tensor [.batch 8, .dim 256] ← Tensor.randn dev ..
  let y := mlp.forward x
  let host ← dev.readTensor y
  IO.println s!"y[0,:5] = {host.firstRow.take 5}"
```

## Operator fusion via Circuit DSL

When you compose many layers, the high-level API uses a fusion DSL
behind the scenes. For an attention block:

```lean
import Hesper.Circuit

def attentionBlock (h : Tensor [.batch B, .seq S, .dim D]) :
    Tensor [.batch B, .seq S, .dim D] :=
  let (q, k, v) := splitQKV (qkvProj.forward h)
  flashAttention q k v |> projOut.forward
```

The compiler fuses the matmul, scale, online softmax, and apply into a
single shader (flash-attention pattern), and fuses the residual / RMSNorm
into the same dispatch where it can. See
[`docs/circuit-dsl-tutorial.md`](../../circuit-dsl-tutorial.md) for the
mechanics.

You don't have to opt in. Writing layer composition the obvious way
gives you fused kernels; if you need to inspect the result, dump the
generated WGSL/PTX via `Circuit.dump`.

## Optimizers

```lean
import Hesper.NN.Optim

let opt := Adam.create lr := 1e-3
let (newModel, newOpt) := Adam.step opt model grads
```

`Hesper.NN.Optim` ships `SGD`, `Adam`, and `LoRA`-aware variants. They
all consume the `vjp` from Ch03 and update model parameters in place
on the GPU.

## What's next

- [Chapter 05 — Switching Backends](Ch05_Backends.md): the same model
  on CUDA instead of WebGPU.
- [Chapter 07 — BitNet End-to-End](Ch07_BitNet.md): everything in this
  chapter applied to a real inference engine.
