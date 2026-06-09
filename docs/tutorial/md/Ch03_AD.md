# Chapter 03 — Automatic Differentiation & Verified Ops

Hesper ships a reverse-mode autodiff layer that's *verified*: every
forward primitive comes with a backward primitive, and the equivalence
between fused and unfused kernels is checked at the type level. You
get gradients without writing them by hand.

For the full design notes see [`docs/VERIFIED_AD.md`](../../VERIFIED_AD.md).
This chapter is the hands-on tour.

## The `Differentiable` typeclass

```lean
import Hesper.Core.Differentiable

open Hesper.Core

-- The shape of the interface — every differentiable op has a forward
-- function and a vector-Jacobian-product backward function.
#check @Differentiable.forward
-- Differentiable.forward : ∀ {Op I O} [self : Differentiable Op I O], Op → I → O

#check @Differentiable.backward
-- Differentiable.backward : ∀ {Op I O} [self : Differentiable Op I O], Op → I → O → I
```

The three type parameters are:

- `Op` — a marker / configuration type for the operation (`AddOp`,
  `MatMulOp`, `ReLUOp`, …).
- `I` — the input shape: `Float` for scalars, `TensorData` for tensors,
  product types for multi-input ops.
- `O` — the output shape.

`backward op x dy` is the vector-Jacobian product: given a cotangent
`dy` at the output, produce a cotangent `dx` at the input.

## A trivial scalar instance

```lean
-- A marker for "multiply by a constant scalar":
structure ScaleByTwo where
  deriving Repr

instance : Differentiable ScaleByTwo Float Float where
  forward  _ x        := 2.0 * x
  backward _ _ dy     := 2.0 * dy        -- d/dx (2x) · dy = 2·dy
```

```lean
def s : ScaleByTwo := {}

#eval (Differentiable.forward  s (3.0 : Float) : Float)        -- 6.0
#eval (Differentiable.backward s (3.0 : Float) (1.0 : Float))  -- 2.0
```

That's the whole interface. Composing op instances (Chain rule, etc.)
lives in `Hesper.AD.Chain`; ready-made instances for the tensor
primitives live in `Hesper.AD.BackwardOps`.

## Where the verification lives

Two kinds of proofs travel with the AD code:

1. **Per-primitive correctness.** `Hesper/AD/Verified.lean` carries the
   theorem that, for each shipped op, `backward` is the transpose of
   the linearization of `forward`.
2. **Fusion equivalence.** When a kernel-fusion pass merges two ops
   into one shader, it discharges a proof that the fused kernel
   computes the same forward and the same backward as the unfused
   chain (see `Hesper/Tensor/VerifiedOpFusion.lean`).

If you add a new primitive, the elaborator forces you to supply the
proof — there's no "trust me" escape hatch.

## Running the AD demos

```bash
lake exe ad-demo               # walk through a 4-layer MLP forward + backward
lake exe verified-op-demo      # show a fused op matching its unfused spec
lake exe unified-ad-demo       # one big example using all features
```

`Examples/MachineLearning/` contains a small Adam optimizer driver and
a LoRA-style fine-tuning loop that all build on this layer.

## What you can rely on

- Gradients of any registered primitive are correct by construction.
- Composite functions inherit correctness from their primitives.
- Fusing two primitives doesn't change the gradient — Lean checks.
- New primitives can't be merged into the AD layer without a proof.

## What this *doesn't* give you

- It doesn't prove numerical stability — `1/x` near zero still
  explodes. Use safe variants where it matters.
- It doesn't replace tests — proofs are about mathematics, not
  hardware. Bit-parity tests against a CPU reference still live in
  `Tests/`.

## What's next

- [Chapter 04 — High-Level API & Tensors](Ch04_HighLevelApi.md): build
  models out of differentiable layers.
- [Chapter 06 — Proofs](Ch06_Proofs.md): how the equivalence and
  invariant proofs in this chapter are actually written.
