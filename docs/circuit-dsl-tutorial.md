---
title: Circuit DSL — Tutorial
date: 2026-04-16
audience: GPU programmers new to hesper's IR layer
---

# Circuit DSL Tutorial

Hesper's Circuit DSL is a **shape-typed graph IR** for expressing GPU
kernels in Lean 4. It compiles via PTX JIT to native CUDA dispatches.
You write the **what** (a graph of typed operations) and the DSL
generates the **how** (PTX, dispatch parameters, fusion).

This tutorial walks through the four-level abstraction stack:

1. **Level 1 — ScalarExp**: per-lane element-wise computation
2. **Level 2 — WarpExp**: warp-level cooperation (reduce, broadcast)
3. **Level 3 — BlockExp**: block-level cooperation (shared memory, barriers)
4. **Level 4 — Raw ShaderM**: full escape hatch for custom kernels

You can mix levels freely.

## 0. Why a DSL at all?

Compared to writing CUDA / WGSL by hand, the DSL gives you:

- **Type-safe shapes**: dispatch grid, input/output buffer sizes, and
  broadcast semantics are tracked in the IR. Mismatches are compile
  errors, not silent crashes.
- **Automatic kernel fusion**: chains of operations collapse to single
  dispatches via fusion passes (`fusePointwise`, `fuseMatmulEpilogue`,
  etc.). You write small composable pieces; the compiler emits one
  optimized kernel.
- **Backend independence**: the same Circuit can lower to CUDA PTX
  (production) or WGSL (debugging / Vulkan).
- **PTX cache**: once compiled, kernels are reused across invocations
  with sub-microsecond dispatch overhead.

Compared to the lowest level (`ShaderM`):

- DSL can express **most** kernels without escaping.
- Hand-written `ShaderM` kernels remain available as `Level 4`
  for things the DSL doesn't yet abstract (tensor cores, async copy,
  flash attention).

## 1. The basic loop

A Circuit is a sequence of operations on `TensorRef`s, built inside a
`CircuitM` monad and then compiled.

```lean
import Hesper.Circuit.IR
import Hesper.Circuit.Lowering
import Hesper.Circuit.Passes
import Hesper.Backend.CUDA

open Hesper Hesper.Circuit

abbrev β := Hesper.CUDAContext

-- 1. Set up the GPU context and some buffers.
let ctx ← Hesper.CUDAContext.init
let n : Nat := 128
let xBuf ← GPUBackend.allocBuffer (β := β) ctx (4 * n).toUSize
-- ... write data into xBuf ...
let outBuf ← GPUBackend.allocBuffer (β := β) ctx (4 * n).toUSize

-- 2. Build a Circuit.  registerExternal binds a host buffer to an IR
--    TensorRef.  The shape (#[n]) and dtype (.f32) are part of the type.
let ccRef : IO.Ref (Option (CompiledCircuit β)) ← IO.mkRef none
Hesper.Circuit.runCachedFused ctx ccRef
  (do
    let x ← CircuitM.registerExternal
              (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
              xBuf #[n] .f32 .Global
    -- 3. Operate on it.  CircuitM.map is shorthand for a pointwise op:
    --    out[i] = body(x[i]).
    let _ ← CircuitM.map x (.mul (.input 0) (.const 2.0))
    pure ())
  -- 4. Replay-time buffer mapping: TensorRef ids → buffers.
  --    id 0 = the first registered tensor (xBuf).
  --    id 1 = the output produced by `map` (outBuf).
  [(0, xBuf), (1, outBuf)]
```

Key concepts:

- **`registerExternal buf shape dtype scope`** binds an existing GPU
  buffer to a TensorRef with id `0, 1, 2, …` (assigned in registration
  order).
- **`runCachedFused`** runs the builder once, applies fusion passes,
  compiles to PTX, caches the compiled kernels in the `IO.Ref`, and
  dispatches. Subsequent calls skip the build/compile and just dispatch.
- The **buffer mapping list** at the end says "for the replay, bind
  TensorRef `0` to `xBuf`, etc." Tensor ids are assigned in the order
  `registerExternal` and `emitOp` appear.

## 2. Level 1 — ScalarExp (per-lane compute)

`ScalarExp` is a pure AST representing a per-lane f32 computation.
The constructors:

```lean
inductive ScalarExp where
  | input  (idx : Nat)            -- inputs[idx] at the current lane
  | const  (v : Float)             -- compile-time constant
  | laneIdx                         -- thread global id as f32
  | indexed (bufIdx : Nat) (addr : ScalarExp)  -- gather: inputs[bufIdx][addr]
  | add | sub | mul | div | neg | rsqrt | exp | tanh | gelu | silu
  | cos | sin | pow | lt | select | mod | idiv | toFloat
  | warpSum | warpBroadcast | warpShuffleXor   -- Level 2 (see below)
  -- ... operator overloading for +, -, *, /, OfNat available
```

You build expressions naturally, thanks to operator overloading:

```lean
-- out[i] = (x[i] * 2.0 + 1.0) ^ 2
let body : ScalarExp := (.input 0 * (.const 2.0) + (.const 1.0)) * (.input 0 * (.const 2.0) + (.const 1.0))

-- Or more idiomatic:
let x      : ScalarExp := .input 0
let scaled : ScalarExp := x * (.const 2.0) + (.const 1.0)
let body            := scaled * scaled
```

The DSL accepts `Nat` literals via `OfNat`:

```lean
let body : ScalarExp := .input 0 * 2 + 1
```

### Map (pointwise / element-wise)

`CircuitM.map`, `CircuitM.zip2`, `CircuitM.pointwise` are sugar that
emit `Prim.scatter` with `addrExpr = .laneIdx` (identity addressing).

```lean
let doubled  ← CircuitM.map x (.input 0 * 2.0)              -- unary
let summed   ← CircuitM.zip2 a b (.input 0 + .input 1)       -- binary
let scaled   ← CircuitM.pointwise #[a, b, c]                 -- N-ary
                  (.input 0 * .input 1 + .input 2)
```

### Broadcast inputs

If an input has shape `#[1]` it broadcasts to every lane:

```lean
-- biasBuf has shape #[1], a single scalar
let bias ← CircuitM.registerExternal biasBuf #[1] .f32 .Global
let result ← CircuitM.zip2 x bias (.input 0 + .input 1)
-- Every lane reads bias[0] (broadcast), full-shape inputs at lane id.
```

### Gather (read at a computed address)

`.indexed bufIdx addr` reads `inputs[bufIdx][toU32(addr)]`. This lets a
lane read another lane's slot or a position in a side buffer:

```lean
-- Pair-element lookup, e.g. NeoX RoPE
let pairIdx : ScalarExp := .laneIdx + .const halfDim.toFloat
let xPair    : ScalarExp := .indexed 0 pairIdx
```

### Scatter (write at a computed address)

`CircuitM.scatterInto` writes to an existing buffer (an `external`)
at a computed address, instead of a fresh allocation:

```lean
-- KV cache write: dst[head*stride + pos*headDim + d] = src[laneIdx]
let i        : ScalarExp := .laneIdx
let head     : ScalarExp := .idiv i (.const headDim.toFloat)
let d        : ScalarExp := .mod  i (.const headDim.toFloat)
let pos      : ScalarExp := .input 1   -- broadcast scalar from posBuf
let addrExpr : ScalarExp :=
  head * .const (maxSeqLen * headDim).toFloat
  + pos * .const headDim.toFloat
  + d
let _ ← CircuitM.scatterInto dstBuf #[kvDim] #[srcBuf, posBuf]
          (.input 0)         -- value: just copy src
          addrExpr
```

This produces a **single dispatch** that does:

```
for laneIdx in [0 .. kvDim):
  value = src[laneIdx]
  addr  = head * stride + pos * headDim + d   (computed per lane)
  dst[addr] = value
```

### Multi-output scatter

`CircuitM.scatterMulti` writes to N destinations from one dispatch:

```lean
-- Both K and V cache writes share the same dispatch grid + inputs
let _ ← CircuitM.scatterMulti #[kvDim]
          #[kSrc, vSrc, posBuf]                -- shared inputs
          #[kCacheBuf, vCacheBuf]               -- two destinations
          #[(kValue, kAddr), (vValue, vAddr)]   -- per-output (value, addr)
```

## 3. Level 2 — WarpExp (warp cooperation)

For warp-level reduction (32 lanes summing to one value), use:

```lean
-- Warp-level dot product: every lane sees Σ(a[lane] * b[lane])
let _ ← CircuitM.zip2 a b
          (.warpSum (.input 0 * .input 1))
```

After `warpSum`, **every lane in the warp holds the same value** (the
sum). You can then continue computing per-lane:

```lean
-- Normalize: value[lane] = a[lane] / Σ a[lane]
let total := .warpSum (.input 0)
let _ ← CircuitM.map a (.input 0 / total)
```

Other Level-2 primitives:

- `warpBroadcast a` — every lane receives `a` evaluated at lane 0
- `warpShuffleXor a mask` — receive value from lane `(self ^ mask)`

These lower to WGSL `subgroupAdd`, `subgroupBroadcastFirst`,
`subgroupShuffleXor`, which on NVIDIA become hardware tree-reduces
(no shared memory needed).

**Important constraint**: warp primitives must be called **uniformly**
by all lanes in the warp. Don't put them inside a divergent branch.

## 4. Level 3 — BlockExp (shared memory + barriers)

For workgroup-level cooperation (multiple warps sharing data via shared
memory), use `Prim.reduceLastAxis` (sum a buffer to one scalar) or the
fused `reduceScatterEpilogue` (reduce + per-lane scatter):

```lean
-- RMSNorm-style fused reduce + dynamic-address write:
--   reduce(x) → total, then per lane: dst[laneIdx + 64] = (x[lane] / total) * 100
let _ ← CircuitM.reduceScatterEpilogue
          .sum                                    -- reduce op
          x                                       -- input being reduced
          #[x]                                    -- epilogue inputs (visible at lane id)
          dstBuf                                  -- destination
          (.input 1 / .input 0 * .const 100)      -- valueExpr (input 0 = reduced scalar)
          (.laneIdx + .const 64)                  -- addrExpr
```

This generates **one workgroup** that:
1. Tree-reduces `x` in shared memory → `total`
2. Each lane reads `x[lane]` and `total`, evaluates `valueExpr`, writes
   `dstBuf[laneIdx + 64] = (x[lane] / total) * 100`

For simpler RMSNorm without dynamic address, the convenience builder
`CircuitM.rmsNorm` already exists:

```lean
let normed ← CircuitM.rmsNorm x scale eps
```

This emits a chain of primitives that the fusion pass collapses to one
dispatch.

## 5. Level 4 — Raw ShaderM escape

When the DSL doesn't abstract what you need (tensor cores, custom
async copy, flash attention's online softmax), drop down to ShaderM:

```lean
import Hesper.WGSL.Monad

def myCustomKernel : ShaderM Unit := do
  let _ ← ShaderM.declareInputBuffer "x" (.array (.scalar .f32) 1024)
  let _ ← ShaderM.declareOutputBuffer "y" (.array (.scalar .f32) 1024)
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid
  ShaderM.if_ (Exp.lt idx (Exp.litU32 1024)) (do
    let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := 1024) "x" idx
    -- ... whatever ...
    ShaderM.writeBuffer (ty := .scalar .f32) "y" idx v
  ) (pure ())

-- Dispatch directly through the backend, bypassing CircuitM.
GPUBackend.executeWithConfigCached ctx myCustomKernel
  [("x", xBuf), ("y", yBuf)]
  (Hesper.ExecConfig.dispatch1D 1024)
  cacheKey cacheRef
```

You lose fusion and shape checking, but you get full control.

## 6. Fusion: write small, get one big kernel

The fusion passes are run automatically by `runCachedFused`. Three
concrete patterns:

### 6.1 Pointwise chain → one kernel

```lean
let doubled ← CircuitM.scale x 2.0
let inc     ← CircuitM.map doubled (.input 0 + .const 1.0)
let final   ← CircuitM.map inc (.input 0 * .input 0)
```

Three `CircuitM.map` calls produce three `Prim.scatter` ops, but
`fusePointwise` collapses them to a single op:

```
final[i] = (x[i] * 2.0 + 1.0)²
```

generated as **one dispatch**.

### 6.2 Map + scatter → one dispatch

```lean
let y ← CircuitM.map x (.input 0 * 2.0)
let _ ← CircuitM.writeSlice dst y 128
```

`fusePointwise` recognises that `y` is a Map-shaped scatter (addrExpr
= laneIdx) feeding a scatter consumer (writeSlice), and inlines it:

```
dst[laneIdx + 128] = x[laneIdx] * 2.0
```

in **one dispatch**.

### 6.3 matmul + epilogue → matmulQ4KWithEpilogue

```lean
let y    ← CircuitM.matmulQ4K x layer       -- Q4_K matmul
let _ ← CircuitM.zip2 y bias                 -- + bias
            (.add (.input 0) (.input 1))
```

`fuseMatmulEpilogue` collapses these into `Prim.matmulQ4KWithEpilogue`,
which emits **one kernel** that does the matmul and adds the bias on
the lane-0 thread of each row right after the warp reduction.

## 7. Operator cheatsheet

```lean
-- Arithmetic
.add a b        -- or:  a + b
.sub a b        -- or:  a - b
.mul a b        -- or:  a * b
.div a b        -- or:  a / b
.neg a          -- or:  -a

-- Constants
.const 3.14
.const 0.5
.const n.toFloat       -- from a Nat
3.14 : ScalarExp        -- via OfScientific
2 : ScalarExp           -- via OfNat

-- Lane index
.laneIdx                -- thread global id as f32

-- Slot reads
.input 0                -- inputs[0] at this lane
.input 1                -- inputs[1] at this lane

-- Gather (computed address)
.indexed 0 (some_addr)  -- inputs[0][toU32(some_addr)]

-- Math
.rsqrt a, .exp a, .tanh a, .gelu a, .silu a
.cos a, .sin a, .pow base exp
.toFloat a              -- (no-op; ScalarExp is already f32)
.idiv a b               -- floor(a/b) via toU32
.mod a b                -- a - floor(a/b)*b

-- Comparison and conditionals
.lt a b                 -- 1.0 if a < b else 0.0
.select cond t f        -- t if cond > 0.5 else f

-- Warp ops (Level 2)
.warpSum a              -- sum across warp; broadcast result
.warpBroadcast a        -- lane 0's value broadcast to all
.warpShuffleXor a mask  -- receive value from lane (self ^ mask)
```

## 8. Common patterns

### RMSNorm

```lean
-- High-level: rmsNorm builder.
let out ← CircuitM.rmsNorm x scale eps

-- Or manually with reduceScatterEpilogue if you need a custom epilogue:
let invD : Float := 1.0 / D.toFloat
let _ ← CircuitM.reduceScatterEpilogue
          .sumOfSquares x #[x, scale] dst
          (.input 1 * .input 2 * .rsqrt (.input 0 * .const invD + .const eps))
          .laneIdx
```

### Q4_K matmul + bias

```lean
let y    ← CircuitM.matmulQ4K x layer
let _    ← CircuitM.zip2 y bias (.input 0 + .input 1)
-- fuseMatmulEpilogue → one fused kernel
```

### Slice-write into a larger buffer

```lean
let y ← CircuitM.map src (.input 0 * scale)
let _ ← CircuitM.writeSlice dst y (.const offset.toFloat)
-- fusePointwise → one dispatch with addrExpr = laneIdx + offset
```

### KV cache write (RoPE-K + scatter)

```lean
let i        := .laneIdx
let head     := .idiv i (.const headDim.toFloat)
let d        := .mod  i (.const headDim.toFloat)
let pairD    := .select (.lt d (.const halfDim.toFloat))
                  (d + .const halfDim.toFloat) (d - .const halfDim.toFloat)
let xSelf    := .input 0
let xPair    := .indexed 0 (head * .const headDim.toFloat + pairD)
let pos      := .input 2  -- broadcast posF32Buf
let theta    := pos * .pow (.const ropeBase) (-(2 * d / .const headDim.toFloat))
let valueExpr := xSelf * .cos theta - xPair * .sin theta
let addrExpr  := head * .const (maxSeqLen * headDim).toFloat
              + pos * .const headDim.toFloat + d
let _ ← CircuitM.scatterInto kvCache #[kvDim]
          #[kSrc, freqFactors, posBuf]
          valueExpr addrExpr
```

## 9. How TensorRef ids are assigned

Each call to `registerExternal` or `emitOp` allocates a new TensorRef
with the next sequential id (starting at 0). When you supply the
`buffers` list to `runCachedFused`, you map ids to buffers:

```lean
[(0, xBuf), (1, biasBuf), (2, outBuf)]
```

If your circuit produces several intermediate tensors that get fused
away, you only need to bind the **caller-facing** ones (externals +
final outputs). The fusion passes inline the rest.

## 10. Debugging

- **`HESPER_PTX_DUMP=/tmp/hesper_ptx`**: writes generated PTX to disk
  (one file per kernel, named by hash).
- **`ptxas --gpu-name=sm_89 -v file.ptx -o /dev/null`**: register count,
  spilling, smem usage.
- **`HESPER_DP4A=1`**: enable Q4_K dp4a path (off by default for
  determinism).

## 11. Files for further reading

- [`Hesper/Circuit/IR.lean`](../Hesper/Circuit/IR.lean) — Prim catalogue,
  ScalarExp, CircuitM builder
- [`Hesper/Circuit/Lowering.lean`](../Hesper/Circuit/Lowering.lean) —
  Prim → ShaderM → PTX
- [`Hesper/Circuit/Passes.lean`](../Hesper/Circuit/Passes.lean) —
  fusion passes (fusePointwise, fuseReduceEpilogue, fuseMatmulEpilogue)
- [`Tests/Circuit/`](../Tests/Circuit/) — concrete examples:
  - `FusePointwiseTest.lean` — pointwise chain fusion
  - `FuseMatmulEpilogueTest.lean` — matmul + epilogue fusion
  - `FuseWriteDestinationTest.lean` — pointwise + writeSlice fusion
  - `ScatterDynamicGPUTest.lean` — scatter with dynamic addr
  - `RopeKScatterGPUTest.lean` — RoPE-K + KV write as one scatter
  - `WarpSumGPUTest.lean` — warp reduction
  - `WarpDotProductGPUTest.lean` — warp-level dot product
  - `ReduceScatterGPUTest.lean` — block-cooperative reduce + scatter
- [`docs/llama-fusion-analysis/`](llama-fusion-analysis/) — design notes
  for the Prim catalogue, motivated by llama.cpp's CUDA fusion

## 12. Glossary

- **TensorRef**: an SSA-style handle to a tensor in the IR
  (`(id, shape, dtype, scope)`).
- **Prim**: an IR primitive (matmulQ4K, scatter, scatter, …).
- **Op**: a Prim plus its concrete inputs and outputs.
- **CircuitState**: the builder's accumulated list of Ops + externals.
- **CompiledCircuit**: a flat list of replay closures, one per Op,
  plus base buffers for fused-away intermediates.
- **PrimExt**: an internal extension Prim used by `mergeSameDispatch`
  (e.g. `fusedKV`).
- **Lowering**: the per-Prim emitter that produces a `ShaderM Unit`.
- **Fusion pass**: a CircuitState → CircuitState rewrite that
  preserves semantics while reducing dispatch count.
