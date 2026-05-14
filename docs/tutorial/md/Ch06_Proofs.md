# Chapter 06 — Proofs: Equivalence & Invariants

Hesper's proofs fall into two flavours. This chapter shows what each
looks like and where they live.

## Flavour 1: equivalence between two shaders

When kernel fusion merges two ops into one shader, we need to know the
fused output equals the unfused chain. The proof lives next to the
fusion pass:

```lean
-- Hesper/Tensor/VerifiedOpFusion.lean (abridged)

theorem fuseGateUp_correct
    (gateW : Linear) (upW : Linear) (x : Tensor _) :
    fuseGateUpKernel gateW upW x
      = (gateW.forward x) * gelu (upW.forward x) := by
  -- proof via reduction on the kernel's body
  ...
```

Once this theorem typechecks, the fusion pass can swap two dispatches
for one and the rest of the codebase keeps working without further
audit.

## Flavour 2: invariants on AD

For autodiff, every `Differentiable op` instance bundles a theorem
saying `backward op` is the transpose of the linearization of
`forward op`:

```lean
-- Hesper/AD/Primitives/Matmul.lean (abridged)

instance : Differentiable .matmul where
  forward  := matmul
  backward := matmul.bwd
  correct  := by
    intro x dy
    -- prove that matmul.bwd x dy is the transpose-Jacobian-vector
    -- product of matmul at x applied to dy
    ...
```

If you add a new primitive without supplying `correct`, the file
doesn't compile.

## Running the proof suite

```bash
lake build Tests.AD                  # rebuilds and re-checks all AD proofs
lake build Tests.VerifiedOpFusion    # equivalence proofs for fused kernels
```

These run as part of CI. The interesting bit is that Lean *only* lets
you build the library if every theorem closes — there's no "skip" flag.

## Writing your own equivalence proof

A small example: prove that swapping the order of two pointwise
operations is a no-op.

```lean
import Hesper.Circuit.Equivalence

theorem add_then_scale_eq_scale_then_add (x : Tensor s) (a b : Float) :
    pointwise (fun t => t + const a) (pointwise (· * const b) x)
      = pointwise (fun t => t * const b + const a * const b) x := by
  -- one-line proof: pointwise distributes
  funext i
  simp [pointwise, mul_add]
```

The same pattern scales up: the bigger fusion theorems just have more
algebraic rewriting in the middle.

## Where proofs *don't* go

We don't prove:

- **Numerical equality across precisions.** `f16 * f16` is not equal to
  `f32 * f32` cast back to `f16`. We test these with bit-parity suites
  in `Tests/`.
- **Performance contracts.** Whether a kernel hits its TPS budget is
  measured, not proved.
- **Driver behaviour.** What Dawn / libcuda actually do is treated as
  an axiom; we test against `llama.cpp` as a ground truth.

## What to read next

- `Hesper/AD/Primitives/` — every per-op correctness proof.
- `Hesper/Tensor/VerifiedOpFusion.lean` — the equivalence framework
  used by fusion passes.
- `Tests/AD/` — bit-parity test harnesses (CPU reference vs GPU).

## What's next

- [Chapter 07 — BitNet End-to-End](Ch07_BitNet.md): everything we've
  covered, applied to a real inference engine.
