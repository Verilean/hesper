# Circuit DSL for GPU Kernels: Design Spec

> *Supersedes parts of `fusion-graph-dsl-design.md`.  Same framework
> but re-grounded in HDL synthesis concepts.*

## Core thesis

A GPU kernel is a **synchronous circuit clocked by kernel dispatch**:

- **Thread = wire** carrying a value.
- **Register = combinational chain** (no inter-thread sync).
- **Shared memory = local register bank** with an explicit `barrier` as
  a latch.
- **Global memory = board-level bus** visible across kernel boundaries.
- **Kernel dispatch boundary = clock edge** — across it, all state
  must settle into Global (or persisted Shared); within it, we reason
  combinationally.

With that framing, **kernel fusion is just logic synthesis**: merge
combinational chains that live within the same clock domain (dispatch).
The tools the HDL world uses for this — typed signals, DAG-by-construction
builder monads, scope-parametric types, IR passes separated from
back-ends — transpose directly onto GPU code generation.

## Design principles (from Sparkle)

1. **Typed signals, no strings.**  Every tensor is a `TensorRef` with a
   static shape, element type, AND memory scope.  The Lean type system
   catches illegal scope transitions at compile time.
2. **DAG by construction.**  The builder monad only allows appending
   new ops whose inputs are already-produced `TensorRef`s.  Cycles are
   unrepresentable.
3. **IR-centric backend split.**  One IR feeds the CUDA-PTX backend,
   the WebGPU/WGSL backend, and future (SPIRV, C-for-CPU) backends.
   Optimisation passes run on the IR, once.
4. **Synthesis via inlining.**  Passes are small, composable:
   constant fold → scope promotion (e.g. an unused Global write
   becomes a Register chain) → combinational merge → dead-op elim.

## Scope algebra

```lean
inductive Scope where
  | Register   -- thread-local.  Only the lane that produced it reads.
  | Lane       -- subgroup-local; reachable via shuffle.  1 warp-step.
  | Shared     -- workgroup-local.  Requires a barrier for visibility
                -- between lanes within the WG.
  | Global     -- device-wide.  Visible after kernel-dispatch boundary.
  deriving DecidableEq, Ord
```

**Ordering**: `Register < Lane < Shared < Global`.  A value may be
*promoted* (moved to a strictly-higher scope) but not *demoted*
without an explicit sync op:

| Transition | Operation | Cost |
|---|---|---|
| Register → Lane | `shuffle`, `subgroupBroadcast` | 1 warp cycle |
| Lane → Shared | direct store (lane index = smem slot) | 1 store + barrier |
| Shared → Global | `writeBuffer` + dispatch boundary | kernel launch |
| Global → Register | `readBuffer` (implicit in next kernel) | 1 load |

The compiler chooses the lowest scope that satisfies all consumers of a
tensor.  When every consumer is the same thread-lane as the producer,
the tensor stays in Register and no global memory is touched.  **This
is exactly the register-chain fusion we've been writing by hand.**

## IR

### Tensor reference

```lean
structure TensorRef (shape : Shape) (dtype : DType) where
  id    : Nat           -- globally unique within a Circuit
  scope : Scope
```

The `id` is allocated monotonically by the builder; the user never
writes a tensor name as a string.  `shape` and `dtype` are compile-
time.

### Op

```lean
-- Primitive ops are data, not ShaderM closures.  The fusion pass
-- pattern-matches against them.
inductive Prim where
  -- Purely combinational per-thread ops (Register → Register):
  | map     (f : Exp → Exp)                           -- unary elementwise
  | zip2    (f : Exp → Exp → Exp)                     -- binary elementwise
  | bitcast (from_ to_ : DType)

  -- Inter-lane (Register → Lane):
  | subgroupReduce (op : ReduceOp)

  -- Inter-WG or requires shared memory:
  | workgroupReduce (op : ReduceOp)
  | matmulQ4K       (cfg : MatmulQ4KConfig)
  | matmulQ6K       (cfg : MatmulQ6KConfig)
  | dot4I8Packed    (cfg : DotConfig)
  | rmsNormReduce   (dim : Nat)           -- needs WG reduce

  -- Memory:
  | readGlobal      (buf : ExternalBuffer)
  | writeGlobal     (buf : ExternalBuffer)

  -- Escape hatch: wrap an existing ShaderM kernel verbatim.
  | wrappedShaderM  (body : WrappedShaderM)

-- An Op sits in a Circuit.  The builder enforces that `inputs` are
-- already in the circuit before `emit` runs.
structure Op where
  prim     : Prim
  inputs   : Array AnyTensorRef     -- index into the Circuit's tensor table
  outputs  : Array AnyTensorRef
  dispatch : DispatchShape           -- grid × workgroupSize
```

### Circuit

```lean
structure Circuit where
  tensors : Array TensorInfo         -- TensorRefs' metadata, indexed by id
  ops     : Array Op                 -- topological order
  outputs : Array AnyTensorRef       -- the tensors returned to the caller

-- Builder monad.  Enforces DAG — cannot reference a TensorRef before
-- its producing Op has been emitted.
abbrev CircuitM := StateM CircuitState

namespace CircuitM

-- Primitive API (most users compose these via typed wrappers):
def emit : (prim : Prim) → (inputs : Array AnyTensorRef)
         → (dispatch : DispatchShape)
         → (outputSpec : Array (Shape × DType × Scope))
         → CircuitM (Array AnyTensorRef)

-- Typed surface (example):
def rmsNorm : TensorRef [B, H] .f32 .Global
           → TensorRef [H]    .f32 .Global
           → CircuitM (TensorRef [B, H] .f32 .Register)

def matmulQ4K : TensorRef [B, K] .f32     s
             → ExternalBuffer              -- weight tensor
             → (outScope : Scope)
             → CircuitM (TensorRef [B, N] .f32 outScope)

end CircuitM
```

### DispatchShape

```lean
structure DispatchShape where
  grid          : Nat × Nat × Nat
  workgroupSize : Nat × Nat × Nat
  -- Two shapes are "synth-compatible" (fusable without boundary
  -- crossing) when equal on the dimensions that matter.  Specifically:
  --   - identical (grid, wgSize): trivially fusable.
  --   - divides-into relationship: future work.
  deriving DecidableEq
```

## Lowering

### Pass pipeline

Default pipeline (ordered):

1. **validate**: topological order, scope consistency.
2. **constFold**: fold `map` and `zip2` over constant inputs.
3. **promoteScope**: if a tensor's consumers are all in the same
   thread-lane, set its scope to Register; if same WG, Shared; etc.
4. **mergeSameDispatch**: adjacent Ops with identical DispatchShape
   are combined into a single synthesised Op whose body is the
   concatenation of the originals (with fresh internal names).
5. **chainCombinational**: sequences of `map` / `zip2` / `bitcast`
   (Register-scope only) collapse into a single map with a composite
   function.
6. **inlineProducer**: when Op B consumes a Register-scope TensorRef
   produced by Op A as its unique consumer, inline A's body into B's
   prologue so the tensor flows as a Lean `Exp`, never touching
   memory.
7. **deadOpElim**: remove Ops whose outputs are unreachable from the
   Circuit's `outputs` set.
8. **emit**: each surviving Op becomes one `(ShaderM Unit, ExecConfig)`
   pair, compiled and dispatched via the existing
   `executeWithConfigCached` infrastructure.

Passes 4-6 are the *kernel fusion*.  Each is straightforward once the
IR is in place.

### Backend

The `emit` pass targets one of:

- **CUDA backend**: `ShaderM → PTX → cuModuleLoadData`.  Same as today.
- **WGSL backend**: `ShaderM → WGSL → wgpuRenderPipeline`.  Same.
- **Future SPIRV**: `ShaderM → SPIRV` for Vulkan parity.

A back-end specifies how primitive ops lower to its shader language.
For `map`, CUDA emits an inline `Exp.mul`, WGSL emits `<T>` operator —
**the IR doesn't care**.

### Hot path: JIT cache (Sparkle-compatible)

Each `Circuit` is hashed (structural, ignoring TensorRef ids) and
memo-cached on first compile.  Subsequent runs replay the cached
kernels via the existing `executeWithConfigCached`.  This mirrors
Sparkle's dlopen path — the IR → backend work happens once.

## Concrete example: Gemma-4 attention prologue

### Today (manual fusion)

```lean
RMSNorm.forward ctx block.attnNorm inputBuf state.normedBuf
Linear.forwardFusedQKV ctx wQ wK wV state.normedBuf state.qBuf ...
ce "qkvNorm" (perHeadRMSNormKernel ...) ...
```

Four bespoke ShaderMs (`rmsNormFusedKernel`, `fusedQ4KMKVDP4AKernel`,
etc.), wired with raw buffer names.  Adding another fusion means a
fifth kernel.

### With the Circuit DSL

```lean
let h   ← read inputBuf                           -- Global [H]
let n   ← rmsNorm h attnNormScale                 -- Register [H] (fusion candidate)
let qIn ← quantizeQ8_1 n                          -- Register chain
let q   ← matmulQ4K qIn wQ (outScope := .Global) -- Global [Q]
let k   ← matmulQ4K qIn wK (outScope := .Global) -- Global [K]
let v   ← matmulQ4K qIn wV (outScope := .Global) -- Global [V]
return (q, k, v)
```

The compiler sees: `n` and `qIn` are Register-scope with a unique
consumer branch each (but the Q/K/V matmuls all share `qIn` as
input — three consumers).  Decision:

- `qIn` stays in one Shared-scope buffer since it has 3 consumers.
- `rmsNorm` + `quantizeQ8_1` fuse into the matmul WGs' prologue if
  their dispatch shape is subdivision of the matmul's (typically
  yes — matmul is 1 WG per output row, norm is 1 WG over the full
  input).
- If norm / quantize's dispatch differs from matmul, they stay as
  separate prologue kernels, but `quantizeQ8_1`'s output still
  flows via a single Global buffer shared by all three matmuls
  (not three — one quantize → three matmul).

The same source collapses to the same kernel layout today hand-writes
(`forwardFusedQKV`), **without hand-writing any fused kernel**.

## Migration plan

### Stage 0 (complete): hand-written fusion baseline

This is what we shipped this session.  5 bespoke fused kernels,
−244 kernels/token.

### Stage 1: IR + per-op lowering (week 1-2)

- Create `Hesper/Circuit/IR.lean` (TensorRef, Prim, Op, Circuit,
  CircuitM).
- Create `Hesper/Circuit/Lowering.lean` with `validate` + `emit`
  passes only (no fusion).
- Wrap every existing ShaderM kernel as `Prim.wrappedShaderM`.
- Pilot: rewrite **one** forwardBlock section (dense FFN) using
  `CircuitM`.  Demonstrate that kernels/token is unchanged (we just
  moved to the new DSL).

**Acceptance**: bit-identical logits vs current, same TPS, same
kernels/token.

### Stage 2: elementwise fusion + same-dispatch merge (week 3)

- Add `chainCombinational` + `mergeSameDispatch` passes.
- Replace `residualAddKernel` and the scale kernels (pleScale1..3,
  layerScale, etc.) with `Prim.map` / `Prim.zip2`.
- Expect −50-100 kernels/token; the 2M+ adjacent elementwise ops get
  collapsed.

**Acceptance**: bit-identical logits, kernels/token drops, no manual
fusion code removed yet.

### Stage 3: producer-consumer register chains (week 4-5)

- Add `promoteScope` + `inlineProducer` passes.
- Replace manually-fused kernels (`fusedQ4KMGateUpDP4AKernel`,
  `fusedQKVDP4AKernel`, `fusedRopeKAndCacheWriteKernel`,
  `rmsNormThenAddKernel`, `fusedQ4KMKVDP4AKernel`) with their
  Circuit-DSL equivalents.

The existing manually-fused ShaderMs stay as **golden references**:
we property-test the output of the auto-fused kernel against them.

**Acceptance**: each hand-written kernel reproducible from the
Circuit DSL; property test passes for 30+ tokens; kernels/token
matches Stage 2 level within ±5%.

### Stage 4: opportunistic (week 6+)

Additional fusion patterns, guided by the `nsys` KPI:

- Per-head qkvNorm (3 dispatches → 1 with 3D grid).
- attnNorm inline into Q8_1 quantize (currently out of reach).
- FFN down + post-FFN residual+norm fusion.

Each additional pattern is a **few lines of Prim emission**, not a
100-LoC ShaderM.

## Implementation cost vs expected reward

| Stage | Weeks | Expected kernels/token | TPS delta |
|---|---|---|---|
| Stage 1 | 2 | 1222 (unchanged) | 0 |
| Stage 2 | 1 | ~1100 | +3-5 |
| Stage 3 | 2 | ~800 | +10-15 |
| Stage 4 | ongoing | target 500 | +15-20 |

Total: ~5 focused weeks for a framework that lets kernel fusion keep
being added **without growing the handwritten kernel inventory**.

## Risks, mitigations

**Over-engineering risk**: a full IR is a lot of code to maintain if
future fusion opportunities turn out to be few.  Mitigation: Stage 1
is small (~500 LoC) and usable on its own.  We can stop at Stage 1 if
the direction isn't paying off.

**Correctness risk**: fusion passes have subtle bugs (e.g. scope
promotion invalidates a barrier).  Mitigation: the pre-existing
hand-written fused kernels are used as reference implementations.
Property tests check Circuit-output == ShaderM-output for the ops
Stage 3 replaces.

**Numerical stability**: register-chain fusion can change the order of
floating-point ops, producing different but correct-within-tolerance
outputs.  Mitigation: document the tolerance; test that the
downstream argmax agrees.

## Appendix: why this is really a circuit

The Sparkle `Signal` monad enforces a combinational DAG.  Our
`CircuitM` monad enforces exactly the same invariant — producer ops
precede consumer ops in the builder order.

The Sparkle `Signal` crossing a flip-flop becomes a `Reg`.  Our
tensor crossing a dispatch boundary is a `Global`-scope tensor; at
synthesis time, the IR is partitioned into dispatches (clock domains)
and the `Global` tensors become the buses between them.

The Sparkle logic-synthesis passes (inline-on-unique-use, constant
fold, dead wire) are literally the fusion passes described above.
Even the JIT structure (Sparkle: dlopen a generated .so; Hesper:
`cuModuleLoadData` a generated .ptx) is the same.

The GPU is a very wide synchronous circuit with a really weird
packaging story.  Most of HDL's 40 years of synthesis research
applies.

Last updated: 2026-04-15.
