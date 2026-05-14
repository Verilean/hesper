# Chapter 04 — High-Level API & Tensors

Most users don't write shaders directly — they compose tensors and NN
layers. This chapter shows the layer above `ShaderM`: tensor
descriptors, the matmul / RMSNorm / attention helpers under
`Hesper.Layers.*`, and the runtime under `Hesper.Compute`.

## Tensors

The smallest unit of typed shape information is `TensorDesc`:

```lean
import Hesper.Tensor.Types
import Hesper.WGSL.DSL
import Hesper.WGSL.CodeGen

open Hesper.Tensor
open Hesper.WGSL
open Hesper.WGSL.Monad (ShaderM)
open Hesper.WGSL.Monad.ShaderM
open Hesper.WGSL.CodeGen

#check @TensorDesc.matrix
-- TensorDesc.matrix : Nat → Nat → optParam DType .f32 → TensorDesc

#eval (TensorDesc.matrix 768 768).sizeBytes
-- 2359296   (768*768*4 bytes for f32)
```

`TensorDesc` carries shape and dtype together so the compute layer can
allocate the right buffer size without separate book-keeping.

## A worked example: process an `array<f32>` with the DSL

The most common pattern is "one thread per array element, apply a
pointwise function." Here's the full kernel for *out = relu(a + b)*
written in `ShaderM`. It elaborates in this notebook and you can read
the generated WGSL with `#eval`:

```lean
-- (Continuing in the same module; imports are at the top of the chapter.)
def fusedAddReLU (size : Nat) : ShaderM Unit := do
  -- Declare three binding-0/1/2 buffers of size N f32:
  let _a   ← declareInputBuffer  "a"   (.array (.scalar .f32) size)
  let _b   ← declareInputBuffer  "b"   (.array (.scalar .f32) size)
  let _out ← declareOutputBuffer "out" (.array (.scalar .f32) size)

  -- Global thread index → array slot:
  let gid ← globalId
  let idx := Exp.vecZ gid

  -- Bounds guard: the launch may over-allocate workgroups.
  if_ (Exp.lt idx (Exp.litU32 size))
    (do
      let av ← readBuffer (ty := .scalar .f32) (n := size) "a" idx
      let bv ← readBuffer (ty := .scalar .f32) (n := size) "b" idx
      -- Pointwise: out[i] = max(a[i] + b[i], 0.0)
      let sum     := Exp.add av bv
      let relud   := Exp.max sum (Exp.litF32 0.0)
      writeBuffer (ty := .scalar .f32) "out" idx relud)
    (pure ())
```

```lean
-- Print the generated WGSL.  No GPU required — this is just a string.
#eval (generateWGSLSimple (fusedAddReLU 1024)).take 600
```

The pattern is the same for every pointwise op — addition, scaling,
GELU, layer norm. The DSL keeps types straight (`Exp (.scalar .f32)`
mismatches are Lean errors, not runtime crashes), and the lowering is
deterministic: what you write is what the GPU runs.

### Reduce: sum along an axis

A *reduce-then-broadcast* pattern (the core of RMSNorm) needs shared
memory + a barrier. Schematically:

```lean
def sumReduce (n : Nat) : ShaderM Unit := do
  let _input ← declareInputBuffer  "input"  (.array (.scalar .f32) n)
  let _out   ← declareOutputBuffer "out"    (.scalar .f32)

  -- One workgroup of 256 threads cooperates over the array.
  sharedNamed "sdata" (.array (.scalar .f32) 256)

  let lid  ← localId
  let lidx := Exp.vecZ lid
  let gid  ← globalId
  let gidx := Exp.vecZ gid

  -- Stage 1: load + per-thread tile reduction into shared memory.
  let v ← readBuffer (ty := .scalar .f32) (n := n) "input" gidx
  writeWorkgroup (ty := .scalar .f32) "sdata" lidx v
  barrier

  -- Stage 2: thread 0 sums the smem tile and writes the result.
  if_ (Exp.eq lidx (Exp.litU32 0))
    (do
      let s0 ← readWorkgroup (ty := .scalar .f32) (n := 256) "sdata" (Exp.litU32 0)
      writeBuffer (ty := .scalar .f32) "out" (Exp.litU32 0) s0)
    (pure ())
```

```lean
-- The shader compiles to ~1 KB of WGSL:
#eval (generateWGSLSimple (sumReduce 256)).length
```

Real reduce kernels (`Hesper/Layers/RMSNorm.lean`) use a proper
tree-reduction inside the workgroup; this sketch shows the structural
pattern (smem + barrier + thread-0 write-back) that every reduction
follows.

### Running it on the GPU

Compilation is half the story — running the kernel needs a `Device`
and uploaded data. The end-to-end driver lives at
`Examples/Compute/MainSimple.lean` (`lake exe matmul-simple`) and
follows this skeleton:

```text
-- Open WebGPU
let inst   ← Hesper.init
let device ← getDevice inst

-- Allocate three GPU buffers
let aBuf ← createBuffer device { size := (n * 4).toUSize,
                                 usage := [.storage, .copyDst],
                                 mappedAtCreation := false }
let bBuf ← createBuffer device { ... }
let outBuf ← createBuffer device { ... }

-- Upload host data
writeBuffer device aBuf 0 aData
writeBuffer device bBuf 0 bData

-- Dispatch the kernel we just defined
let config := ExecutionConfig.dispatch1D n 64
executeShaderNamed device (fusedAddReLU n)
  [("a", aBuf), ("b", bBuf), ("out", outBuf)] config

-- Read back
let bytes   ← mapBufferRead device outBuf 0 ((n * 4).toUSize)
unmapBuffer outBuf
let results ← Hesper.Basic.bytesToFloatArray bytes
```

Run the full version with:

```bash
lake exe matmul-simple             # vector add (compute pipeline smoke test)
lake exe codegen-demo              # prints WGSL for several DSL kernels
```

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
