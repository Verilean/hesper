# Chapter 03 — Automatic Differentiation & Verified Ops

Hesper ships a reverse-mode autodiff layer that's *verified*: every
forward primitive comes with a proven-correct backward primitive, and
fused kernels are checked against the unfused reference at the type
level. You get gradients without writing them by hand, and without
trusting an opaque framework.

For the full design notes see [`docs/VERIFIED_AD.md`](../../VERIFIED_AD.md).
This chapter is the hands-on tour.

## The `Differentiable` typeclass

Every primitive that participates in AD implements one interface:

```lean
class Differentiable (op : OpKind) where
  forward  : Tensor → Tensor
  backward : Tensor → Tensor → Tensor    -- (input, grad-out) → grad-in
  -- and a proof that backward is correct w.r.t. forward
```

Built-in instances cover the usual list: `add`, `mul`, `relu`, `gelu`,
`softmax`, `rmsNorm`, `matmul`, `embedding`, `flashAttention`. Each
instance also registers a *fused-kernel parity* spec, which kernel-fusion
passes must respect.

## A tiny example

```lean
import Hesper.AD

def f (x : Tensor [.batch 4, .dim 8]) : Tensor [.batch 4, .dim 8] :=
  relu (x * x + x)

-- Forward and backward in one call:
def main : IO Unit := do
  let x : Tensor [.batch 4, .dim 8] ← Tensor.randn ...
  let (y, vjp) := AD.forwardWithBackward f x
  IO.println s!"y = {y}"

  -- Pull back a unit cotangent:
  let dy : Tensor [.batch 4, .dim 8] := Tensor.ones _
  let dx := vjp dy
  IO.println s!"∂y/∂x · 1 = {dx}"
```

`forwardWithBackward` returns the forward result and a `vjp` closure
that applies the transpose of the linearization at `x`. This is the
standard reverse-mode pattern; the difference is that every step is
backed by a Lean proof.

## Where the verification lives

Two kinds of proofs travel with the AD code:

1. **Per-primitive correctness.** For each `Differentiable` instance,
   a theorem in `Hesper/AD/Primitives/<op>.lean` says
   `backward op x (grad-of forward op at x) = grad-of identity`.
2. **Fusion equivalence.** When a kernel-fusion pass merges two ops
   into one shader, it must discharge a proof that the fused kernel
   computes the same forward and the same backward as the unfused
   chain. The proofs live in `Hesper/Tensor/VerifiedOpFusion.lean`.

When you add a new primitive, the elaborator forces you to supply both
proofs — there's no "trust me" escape hatch.

## Running the AD demos

```bash
lake exe ad-demo               # walk through a 4-layer MLP forward + backward
lake exe verified-op-demo      # show a fused op matching its unfused spec
lake exe unified-ad-demo       # one big example using all features
```

`Examples/MachineLearning/` contains a small Adam optimizer and a
LoRA-style fine-tuning loop that all build on this layer.

## What you can rely on

- Gradients of any registered primitive are correct by construction.
- Composite functions inherit correctness from their primitives.
- Fusing two primitives doesn't change the gradient — Lean checks.
- New primitives can't be merged into the AD layer without a proof.

## What this *doesn't* give you

- It doesn't prove numerical stability — `1/x` near zero still
  explodes. Use `safeDiv` / `safeLog` where it matters.
- It doesn't replace tests — proofs are about mathematics, not
  hardware. Bit-parity tests against a CPU reference still live in
  `Tests/AD/`.

## What's next

- [Chapter 04 — High-Level API & Tensors](Ch04_HighLevelApi.md): build
  models out of differentiable layers.
- [Chapter 06 — Proofs](Ch06_Proofs.md): how the equivalence and
  invariant proofs in this chapter are actually written.
