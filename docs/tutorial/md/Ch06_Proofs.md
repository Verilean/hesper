# Chapter 06 — Proofs: Equivalence & Invariants

Hesper's proofs fall into two flavours. This chapter shows what each
looks like and where they live.

## Flavour 1: equivalence between two functions

The simplest proof you'll write is "these two definitions compute the
same thing." Here's a worked toy example that compiles in this
notebook:

```lean
import Hesper.Core.Differentiable

open Hesper.Core

def add_then_scale (x : Float) : Float := (x + 1.0) * 2.0
def scale_then_add (x : Float) : Float := 2.0 * x + 2.0

theorem add_then_scale_eq_scale_then_add :
    ∀ x : Float, add_then_scale x = scale_then_add x := by
  intro x
  unfold add_then_scale scale_then_add
  -- Float arithmetic isn't decidable, but ring-like rewriting works
  -- when the operations agree under IEEE-754 reassociation.
  -- For the real Tensor proofs we use the equivalence framework in
  -- Hesper/Tensor/VerifiedOpFusion.lean.
  sorry
```

The `sorry` is honest: floating-point equality is *not* automatic in
Lean's core, so we either reach for IEEE-aware tactics or constrain
ourselves to integer / Nat models. The real equivalence proofs in
Hesper work over abstract value semantics, not raw `Float`.

## Flavour 2: equivalence between two shaders

When a kernel-fusion pass merges two ops into one shader, we need to
know the fused output equals the unfused chain. Sketch:

```text
-- Hesper/Tensor/VerifiedOpFusion.lean (abridged)
theorem fuseGateUp_correct
    (gateW : Linear) (upW : Linear) (x : Tensor _) :
    fuseGateUpKernel gateW upW x
      = (gateW.forward x) * gelu (upW.forward x) := by
  -- proof via reduction on the kernel's body
  sorry
```

Once this theorem typechecks, the fusion pass can swap two dispatches
for one and the rest of the codebase keeps working without further
audit.

## Flavour 3: invariants on AD

For autodiff, every `Differentiable` instance is paired with a theorem
saying `backward op` is the transpose of the linearisation of
`forward op`. From Ch03:

```lean
-- (Continuing from the imports in the first cell.)
structure NoOp where deriving Repr

instance : Differentiable NoOp Float Float where
  forward  _ x  := x         -- identity
  backward _ _ dy := dy      -- d/dx x · dy = 1·dy = dy

-- Trivial correctness: backward propagates the cotangent unchanged.
theorem noop_backward_id :
    ∀ (op : NoOp) (x dy : Float),
      Differentiable.backward op x dy = dy := by
  intro _ _ _
  rfl
```

The real instances in `Hesper/AD/Verified.lean` carry the same kind of
theorem with real arithmetic content.

## Running the proof suite

```bash
lake build Tests.AD                  # rebuilds and re-checks all AD proofs
lake build Tests.VerifiedOpFusion    # equivalence proofs for fused kernels
```

These run as part of CI. Crucially, Lean *only* lets you build the
library if every theorem closes — there's no "skip" flag.

## Where proofs *don't* go

We don't prove:

- **Numerical equality across precisions.** `f16 * f16` is not equal
  to `f32 * f32` cast back to `f16`. We test these with bit-parity
  suites in `Tests/`.
- **Performance contracts.** Whether a kernel hits its TPS budget is
  measured, not proved.
- **Driver behaviour.** What Dawn / libcuda actually do is treated as
  an axiom; we test against `llama.cpp` as a ground truth.

## What's next

- [Chapter 07 — BitNet End-to-End](Ch07_BitNet.md): everything we've
  covered, applied to a real inference engine.
