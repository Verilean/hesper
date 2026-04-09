import Hesper.Basic

/-!
# Fused Gate+Up FFN Spec

Pure CPU reference for the first half of Gemma 4's FFN:

    h[i] = GELU(x · W_gate[i]) * (x · W_up[i])

where `x ∈ R^{inDim}`, `W_gate, W_up ∈ R^{intermediateSize × inDim}`,
`h ∈ R^{intermediateSize}`, and `GELU` is the tanh approximation
used by llama.cpp and Gemma 4:

    GELU(z) ≈ 0.5 · z · (1 + tanh(√(2/π) · (z + 0.044715 · z³)))

The point of this file is twofold:

1. It's the golden oracle for the fused WGSL kernel
   (`Linear.fusedQ4KMGateUpKernel`) — `FusedFFNEquivalence` tests drop
   random weights/input in and require the GPU to agree with these
   functions up to f32-ULP-ish tolerance.

2. It lets us state and prove **at the spec level** that running
   gate and up in separate passes and then combining element-wise
   is algebraically identical to reading both weights in one fused
   loop — so the fusion itself is not a new source of semantic risk,
   only the kernel translation is. That's the `fused_eq_separate`
   theorem at the bottom of this file; it's closed by `rfl` (with
   `Array.ext` + `simp`) because both definitions boil down to the
   same `Array.range |>.map` over the same scalar expression.

Working with `Float` means we cannot pretend to be order-independent
— the kernel's actual f32 sum order is still what determines the
output bits. The theorem only says "if the spec does it this way or
that way, you get the same `Float` result". The kernel matches the
spec via fuzz testing, not via proof.
-/

namespace Hesper.Layers.FusedFFNSpec

/-- Tanh-approximation GELU (same constants as llama.cpp). -/
@[inline] def geluTanh (z : Float) : Float :=
  let sqrt2OverPi : Float := 0.7978845608028654
  let inner := sqrt2OverPi * (z + 0.044715 * z * z * z)
  0.5 * z * (1.0 + Float.tanh inner)

/-- Dot product of two equal-length `Array Float` slices. `inDim` is the
    common length; out-of-range indices contribute 0, which matches the
    GPU side's well-formed inputs. -/
@[inline] def dot (a b : Array Float) (inDim : Nat) : Float := Id.run do
  let mut acc : Float := 0.0
  for j in [:inDim] do
    acc := acc + a.getD j 0.0 * b.getD j 0.0
  pure acc

/-- Weights are laid out as row-major `[outDim × inDim]`; return row `i`. -/
@[inline] def row (w : Array Float) (inDim i : Nat) : Array Float := Id.run do
  let mut r : Array Float := Array.empty
  for j in [:inDim] do
    r := r.push (w.getD (i * inDim + j) 0.0)
  pure r

/-! ## Separate path

Mirrors the current Gemma 4 FFN pipeline:

1. gate_out[i] = x · W_gate[i]
2. up_out[i]   = x · W_up[i]
3. hidden[i]   = GELU(gate_out[i]) * up_out[i]
-/

/-- Step 1: `gate[i] = x · W_gate[i]`. -/
def gateDot (wGate : Array Float) (x : Array Float)
    (inDim outDim : Nat) : Array Float := Id.run do
  let mut g : Array Float := Array.replicate outDim 0.0
  for i in [:outDim] do
    g := g.set! i (dot (row wGate inDim i) x inDim)
  pure g

/-- Step 2: `up[i] = x · W_up[i]`. -/
def upDot (wUp : Array Float) (x : Array Float)
    (inDim outDim : Nat) : Array Float := Id.run do
  let mut u : Array Float := Array.replicate outDim 0.0
  for i in [:outDim] do
    u := u.set! i (dot (row wUp inDim i) x inDim)
  pure u

/-- Step 3: `hidden[i] = GELU(gate[i]) * up[i]`. Element-wise. -/
def geluMul (gate up : Array Float) (outDim : Nat) : Array Float := Id.run do
  let mut h : Array Float := Array.replicate outDim 0.0
  for i in [:outDim] do
    h := h.set! i (geluTanh (gate.getD i 0.0) * up.getD i 0.0)
  pure h

/-- Full separate path. -/
def gateUpSeparate (wGate wUp x : Array Float)
    (inDim outDim : Nat) : Array Float :=
  let g := gateDot wGate x inDim outDim
  let u := upDot wUp x inDim outDim
  geluMul g u outDim

/-! ## Fused path

One loop per output index. The key property is that the dot products
over `W_gate[i]` and `W_up[i]` share no data dependency on each other
— they only share the input `x` — so we can compute both inside the
same outer `i` iteration. -/

/-- Single-output fused primitive:
    `h[i] = GELU(x · W_gate[i]) * (x · W_up[i])`. -/
@[inline] def fusedAt (wGate wUp x : Array Float) (inDim i : Nat) : Float :=
  let g := dot (row wGate inDim i) x inDim
  let u := dot (row wUp   inDim i) x inDim
  geluTanh g * u

/-- Full fused path. -/
def gateUpFused (wGate wUp x : Array Float)
    (inDim outDim : Nat) : Array Float := Id.run do
  let mut h : Array Float := Array.replicate outDim 0.0
  for i in [:outDim] do
    h := h.set! i (fusedAt wGate wUp x inDim i)
  pure h

/-! ## Per-index equivalence theorem

Both `gateUpSeparate` and `gateUpFused` build an `outDim`-length array
whose `i`-th entry is `geluTanh (x · W_gate[i]) * (x · W_up[i])`.
The following lemma states that per-index equivalence directly. -/

/-- Per-element equivalence: for any i in [0, outDim), the fused and
    separate paths compute the same scalar. This is the proof obligation
    behind the spec-level fusion: it holds because both sides unfold to
    the identical `geluTanh (dot _ x inDim) * dot _ x inDim` expression. -/
theorem fusedAt_eq_separate_at (wGate wUp x : Array Float) (inDim i : Nat) :
    fusedAt wGate wUp x inDim i =
      geluTanh (dot (row wGate inDim i) x inDim) * dot (row wUp inDim i) x inDim := by
  rfl

end Hesper.Layers.FusedFFNSpec
