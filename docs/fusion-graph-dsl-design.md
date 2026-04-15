# 2-Layer DSL Design: Graph IR + ShaderM Lowering

## Problem statement

The current `ShaderM` DSL is a flat, string-indexed builder:

```lean
ShaderM = StateM ShaderState
ShaderState = { stmts: List Stmt
              , declaredBuffers: List (String × WGSLType × BufferAccessMode)
              , sharedVars: List (String × WGSLType)
              , varCounter: Nat
              ... }
```

Every buffer reference is a raw `String`. Variable references are raw
`String`. Dispatch shape lives in a separate `ExecConfig` outside the
ShaderM. There is no notion of producer/consumer dependency between
ShaderM kernels — that's all encoded in the **caller** writing `ce
"foo" kernel [("input", srcBuf), ("output", dstBuf)] config` correctly.

This makes kernel-fusion optimisations near-impossible to implement
generically. To fuse two same-dispatch kernels, a fusion pass would
have to:

- Reconcile colliding buffer names (e.g. both kernels declared `input`).
- Reconcile colliding `var` names (e.g. both used `acc`).
- Verify dispatch shapes match (lives in caller, not the ShaderM).
- Detect that kernel B's `input` is kernel A's `output` to elide the
  bridge buffer (currently impossible — the two ShaderMs only share
  `String` buffer names that the caller wired up).

In practice we end up writing **bespoke fused kernels** by hand
(`fusedQ4KMGateUpDP4AKernel`, `fusedQKVNormKernel`, etc.) — every new
fusion is a new ShaderM written from scratch. This is why the
kernels-per-token KPI is stuck at 6.5× llama.cpp despite a session of
fusion work.

## Inspiration: Sparkle's IR/JIT design

[Verilean/Sparkle](https://github.com/Verilean/sparkle) (a Lean 4
hardware HDL DSL → Verilog/C++/JIT compiler) follows a clean
separation:

```
Pure Lean DSL  →  IR (DAG of typed signals)  →  N backends
```

Key invariants Sparkle enforces:

- **The construction monad enforces a strict DAG** — combinational
  loops are impossible at the IR level.
- **Signals carry static type and bit-width** — typed reasoning, no
  string identity.
- **Backends consume the same IR** — Verilog, C++ simulator,
  dlopen-loaded native JIT. Each backend implements lowering rules.
- **IR-level passes (inlining, constant folding, var promotion)** run
  before lowering. A new backend doesn't have to re-implement them.

The pattern transposes naturally onto GPU compute:

```
Pure Lean DSL  →  Graph IR (DAG of typed tensors + ops)  →  ShaderM, WGSL, PTX, …
```

## Proposed design

### Layer 1: Graph IR (new)

`Hesper.Fusion.IR` — the typed, name-free DAG.

```lean
namespace Hesper.Fusion.IR

-- Tensors are named by node-id, NOT by user-supplied string.
structure TensorRef where
  id    : Nat                     -- unique within a Graph
  shape : Shape                   -- e.g. [batch, hidden]
  dtype : DType                   -- f32, f16, u32, q4_k, q8_1, ...
  scope : Scope                   -- Global | Workgroup | RegisterChain

inductive Scope where
  | Global         -- resides in GPU memory; needs a Buffer at lowering
  | RegisterChain  -- intermediate fed only into the next op via register
                    -- (becomes a Lean-level Exp during lowering)

-- Ops are first-order data — no embedded ShaderM.  Each op has:
-- * A list of input TensorRefs (read).
-- * A list of output TensorRefs (write).
-- * A DispatchShape (per-op grid + workgroupSize).
-- * An OpKernel — a Lean-side description of the kernel body (NOT a
--   ShaderM).  See "OpKernel definitions" below.
inductive Op where
  | mk (kernel : OpKernel)
       (inputs  : Array TensorRef)
       (outputs : Array TensorRef)
       (dispatch : DispatchShape)

-- A graph is a list of Ops in topological order, plus the set of
-- TensorRefs that escape (are returned to the caller).
structure Graph where
  ops      : Array Op
  outputs  : Array TensorRef
  -- Internal: a tensor's `id` indexes into a flat allocator.

-- Builder monad — like Sparkle's Signal monad, enforces DAG
-- by construction.
abbrev GraphM (α : Type) := StateM GraphState α

-- Allocate a fresh TensorRef.  ID is monotonic; user never sees a string.
def freshTensor (shape : Shape) (dtype : DType) (scope : Scope := .Global)
    : GraphM TensorRef

-- Append an op to the graph.  Outputs are auto-allocated as
-- `freshTensor`s before the op runs and returned to the caller.
def emitOp (kernel : OpKernel) (inputs : Array TensorRef)
    (outDescs : Array (Shape × DType))
    (dispatch : DispatchShape)
    : GraphM (Array TensorRef)

end Hesper.Fusion.IR
```

### OpKernel — the kernel-body description

Two flavours:

**Wrapped ShaderM** — adapt existing kernels with minimal change:

```lean
-- The ShaderM closure receives a name-resolution function that maps
-- input/output indices to the canonical buffer/var names the lowering
-- pass picked.  The kernel body uses those, not user-supplied strings.
structure WrappedShaderM where
  build : (resolveInput : Nat → String) →
          (resolveOutput : Nat → String) →
          ShaderM Unit

-- Existing fusedQ4KMGateUpDP4AKernel is wrapped as:
--   WrappedShaderM.mk fun ri ro => existingKernel
--     (using ri 0 for "weights_gate", ri 1 for "weights_up",
--          ri 2 for "input_q8", ro 0 for "output")
```

**Native OpDescription** — for ops the framework knows about
intrinsically.  These are the ops the **fusion pass** can reason about:

```lean
inductive OpDescription where
  | matmulQ4K  (config : MatmulConfig)        -- inDim, outDim, qFmt, …
  | rmsNorm    (config : RMSNormConfig)
  | residualAdd (size : Nat)
  | elementwise (f : Exp -> Exp)              -- map over a tensor
  | quantizeQ8_1
  | rope       (config : RoPEConfig)
  | softmax    (config : SoftmaxConfig)
  | dp4aDot    (config : DotConfig)
  | reduce     (op : ReduceOp)                -- sum, max, …
  ...
```

The fusion pass pattern-matches on `OpDescription` to decide whether
adjacent ops can be fused.

### Layer 2: Lowering passes

`Hesper.Fusion.Lowering` — converts a `Graph` into one or more
`(ShaderM Unit, ExecConfig)` pairs.

**Pass 0: validation.**  Check that every TensorRef is produced before
it's consumed (DAG invariant).  Already implied by the GraphM monad
order, but we re-check to catch programming bugs.

**Pass 1: `fuseSameDispatch`.**  Walk the op list.  When N adjacent ops
all have the same dispatch shape AND only feed each other's outputs
through `RegisterChain` scope (no `Global` bridge needed), inline them
into a single ShaderM:

- Allocate fresh internal buffer/var names (e.g. `__t{id}`).
- Concatenate the per-op ShaderM bodies in order.
- Bridge tensors with `scope = RegisterChain` are translated to a
  Lean-level `Exp` (or a `varNamed` / `var` pair) — no global memory
  round-trip.
- Bridge tensors with `scope = Global` stay as separate buffer
  references (still fused into one kernel, but the round trip remains
  — useful when a downstream op needs cross-WG visibility).

**Pass 2: `liftElementwiseChain`.**  When N adjacent `elementwise` ops
operate on the same shape, combine their `Exp -> Exp` functions into
one composite `Exp -> Exp` and emit a single map kernel.

**Pass 3: `inlineProducer`.**  For an op that consumes exactly one
producer and the producer is element-wise local (no reduction across
threads), inline the producer's body into the consumer.

**Pass 4: `commute / hoist`.**  Move ops to enable upstream fusion
(e.g. swap two independent ops to bring same-dispatch ops together).

The result of all passes is the `Graph` rewritten with fewer, larger
`Op`s, then **lowered op-by-op** to existing `ShaderM` infrastructure.

### Stage 1 MVP

Goal: **the existing `forwardBlock` rewritten in `GraphM` produces
identical output to the current implementation**, with no fusion
applied.  Just the structural change.

Concretely:

1. `Hesper.Fusion.IR` module: `TensorRef`, `Op`, `Graph`, `GraphM`.
2. `Hesper.Fusion.Lowering` module: `compileGraph : Graph → β →
   IO Unit` that runs each op as a separate dispatch using the existing
   `executeWithConfigCached` infrastructure.
3. Two `OpDescription` constructors implemented (`matmulQ4K` and
   `elementwise`) plus a generic `WrappedShaderM` for everything else.
4. Rewrite ONE forwardBlock section (e.g. the FFN dense path) to use
   `GraphM`.

At this point, kernels-per-token should be unchanged from current
(1222), but we've established the graph layer.

### Stage 2: `fuseSameDispatch` pass

Implement the rewriting described above.  Once landed, all the existing
small element-wise fusions (`pleScale1` … `pleScale3`, etc.) should
collapse automatically, and we can document the per-token kernel
count drop.

### Stage 3: Producer-consumer register chains

The big win: when op A produces `RegisterChain`-scope tensor `t`, and
op B is the unique consumer of `t` AND has the same dispatch shape,
the lowering inlines A's body into B's prologue, threading `t` as a
Lean-level `Exp` rather than going through global memory.

This is what makes it possible to write the existing
`fusedQ4KMGateUpDP4AKernel` in a few lines:

```lean
let normed ← op (rmsNorm attnNormCfg) inputBuf
let q ← op (matmulQ4K wQConfig) [normed, wQ.weight]   -- can fuse with norm
let k ← op (matmulQ4K wKConfig) [normed, wK.weight]   -- shares `normed`
let v ← op (matmulQ4K wVConfig) [normed, wV.weight]   -- shares `normed`
-- The fusion pass sees: normed is RegisterChain-scope, three consumers,
-- all per-WG-1-row dispatch.  Cannot inline directly (3 consumers), so
-- normed stays as a Global buffer ONCE, but the three matmuls fuse the
-- shared Q8_1 quantize step.
```

## Migration strategy

The 2-layer DSL coexists with the existing `ShaderM`-direct path:

- **Today**: All forwardBlock code uses raw `ShaderM` + `ce` helper.
- **Stage 1**: New `GraphM` builder available.  Pilot one section
  (e.g. dense FFN) ported, others unchanged.
- **Stage 2**: `fuseSameDispatch` pass enabled in the lowering.
  Manually-written fused kernels (`fusedQ4KMGateUpDP4AKernel`, etc.)
  become **fallback / reference**: they're still used as the kernel
  body for the corresponding `Op` until the fusion pass produces a
  better result.
- **Stage 3**: Once register-chain fusion is robust, the manual
  fused kernels are removed in favour of the auto-fused path.

The existing `fusedQ4KMGateUpDP4A4RowKernel` etc. don't disappear —
they become the *target* the framework should be able to reproduce
mechanically by inlining `RMSNorm + matmulQ + matmulK + matmulV +
quantizeQ8_1 + … `.

## Costs and risks

**Implementation cost**: Stage 1 is ~1 week of focused work.  Stages 2
and 3 are ~2 weeks each.  Total commitment for a usable fusion
framework: ~5 weeks.

**Correctness risk**: The lowering pass must preserve the exact
numerical output of each individual op + any inlined sequence.
Mitigations:

- Fall back to per-op dispatch when the fusion pass can't prove
  legality (e.g. shape mismatch, cross-thread dependency).
- Property test: `compile graph` and `compile (manuallyFused graph)`
  produce bit-identical output buffers.
- Existing manually-fused kernels serve as ground-truth references.

**Maintenance risk**: A fusion pass that mostly works but occasionally
mis-fuses is worse than no fusion pass at all.  We'll keep the
pass strictly opt-in (a `Graph` is allowed to declare which ops *may*
fuse) until it has been stable for several months.

## Why not just keep writing manual fused kernels?

We've been doing exactly that for a session.  The result:

- Wrote 4 manual fused kernels (`gate+up`, `wK+wV`, `wQ+wK+wV q8_1
  share`, `RMSNorm + residual add`, `RoPE-K + KV write`).
- Saved ~244 kernels/token (1466 → 1222, -16%).
- Each kernel was 100-200 LoC.

Linearly, hitting llama.cpp's 187/token would require ~30 more
manually-fused kernels, ~5,000-10,000 LoC.  The framework is the
better long-term investment if we want to keep narrowing the gap.

## References

- `Sparkle` — typed Lean DSL → IR → multi-backend, JIT via dlopen.
  https://github.com/Verilean/sparkle
- `XLA` (TensorFlow) — HLO IR with op-fusion pass.
- `Triton` — Python DSL with explicit grid, automatic shared-memory
  staging.
- `MLIR` — multi-dialect IR designed for fusion + multi-backend.

Last updated: 2026-04-15.
