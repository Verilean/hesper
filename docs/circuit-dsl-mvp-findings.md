# Circuit DSL MVP: findings

Companion to `docs/circuit-dsl-design.md`.  Records the outcome of
the first MVP wire-up and what it says about the framework design.

## What we built

`Hesper/Circuit/IR.lean` and `Hesper/Circuit/Lowering.lean` — the
smallest viable framework:

- `TensorRef` with shape + dtype + scope, id-based identity (no
  strings).
- `Prim.matmulQ4K` wrapping an existing `Linear.LinearLayer`.
- `CircuitM` builder monad that enforces DAG-by-construction.
- `Hesper.Circuit.compile` lowering: for each Op, call the existing
  `LinearLayer.forward` on the resolved buffers.

No fusion passes.  No scope promotion.  No Exp-level inlining.  Just
"call the existing kernel via a DAG IR".

## What we tried

Rewrote ONE `LinearLayer.forward` call site in
`Gemma4.forwardBlock` (the non-KV-layer wQ projection) as:

```lean
let (_, st) := Hesper.Circuit.CircuitM.run (do
  let normed ← Hesper.Circuit.CircuitM.registerExternal
    state.normedBuf #[cfg.hiddenSize] .f32 .Global
  let _q ← Hesper.Circuit.CircuitM.matmulQ4K normed block.attention.wQ
  pure ())
let qOutTr := st.tensors.back!
Hesper.Circuit.compile ctx st [(qOutTr, state.qBuf)]
```

vs the original direct call:

```lean
Linear.LinearLayer.forward ctx block.attention.wQ state.normedBuf state.qBuf
```

## Result

| | TPS |
|---|---|
| Baseline (direct call) | **46.0** |
| Circuit DSL (zero-fusion wire-up) | **37-40 (-15%)** |

**Correctness is fine** — bit-identical decoded text.  The slowdown
is pure CPU-side overhead: allocating a fresh `CircuitState`,
pushing tensor metadata, pattern-matching on Prim, and doing a
linear-search buffer lookup, for every `forwardBlock` invocation.

## Analysis

The failure is instructive.  `forwardBlock` is called ~42 layers ×
30 tokens ≈ 1260 times per decode session.  Each call currently
pays:

1. `CircuitM.run` → allocate empty CircuitState (Array constructor)
2. `registerExternal` → push to `tensors` + `externals` (2 Array pushes)
3. `matmulQ4K` → push to `tensors` + `ops` (2 Array pushes)
4. `compile` → build BufferMap (List cons x2), iterate ops, pattern-
   match each Prim, linear-search lookup x2, then the dispatch itself

Rough napkin: 10-20 microseconds of CPU-side overhead per Op, over
1260 calls per session = 20-40 ms.  Observed delta is ~6-9 ms/30-
tokens, which fits this shape.

## What this means

The IR design itself is fine — the design doc calls out exactly
this issue in passing ("CircuitM run cost must be ~zero").  The
MVP just doesn't implement the caching you'd want in production:

- **Build the Circuit once** per unique layer shape, not per call.
  Subsequent calls reuse the precompiled `CircuitState` + the
  cached `prepared` dispatches inside each `LinearLayer`.
- **Use arrays/`Vec` for tensors instead of association-list
  BufferMaps**, so lookup is O(1).
- **Lift the dispatch straight from the IR**, skipping the
  Lean-side `run` entirely on replay.

In Sparkle's terms: the first run is the JIT compile step, the
subsequent runs are the dlopen'd `.so` reuse.  We haven't
implemented the cache yet.

## Decision: keep framework files, do NOT wire into Gemma4 yet

- `Hesper/Circuit/IR.lean` and `Hesper/Circuit/Lowering.lean` land as
  framework-only code.  No production caller uses them.
- The Gemma4 wire-up was reverted (baseline 46.0 TPS restored).
- Stage 1.5 of the design plan needs a **"compile once, replay many"**
  API shape before any caller can adopt the DSL without regressing
  per-call overhead.

## Next step

Before any further fusion work, the overhead-free path must be built:

```lean
-- Stage 1.5: build-once-replay-many.
structure CompiledCircuit (β : Type) where
  -- Pre-resolved Op list with cached dispatches; no allocation on replay.
  replay : β → List (Nat × GPUBackend.Buf β) → IO Unit

def Circuit.compileOnce [GPUBackend β] (c : CircuitM ...) : IO (CompiledCircuit β)

-- Caller holds the CompiledCircuit in its InferenceState and calls
-- `cc.replay ctx buffers` per token — no DSL evaluation on the hot path.
```

This brings the per-call overhead to ~zero (one function call through a
closure, plus the actual kernel dispatch), at which point every later
fusion stage pays for itself.

Last updated: 2026-04-15.
