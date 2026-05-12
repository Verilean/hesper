import Hesper.Circuit.IRCore

/-!
# Circuit IR — pure-Lean evaluator

A reference semantics for `Hesper.Circuit.ScalarExp` (and a small
slice of `Hesper.Circuit.Prim`) that runs entirely inside Lean,
with no GPU FFI and no WGSL/PTX string codegen.

## Why

The Circuit IR is the canonical mathematical reference for
Hesper's GPU kernels — every WGSL/PTX lowering is supposed to
agree with it. Until now there was no way to *run* a Circuit
program inside Lean to check that, except by emitting a kernel
and executing it on real hardware. This evaluator gives that
missing capability:

  - Reference semantics that downstream proofs can pin against.
  - A way to test Circuit-level optimisations without needing a
    GPU available in CI.
  - An anchor for cross-stack equivalence work
    (e.g. Sparkle HDL ↔ Hesper at the Circuit-IR layer).

## Design

The evaluator interprets `ScalarExp` as a structural-recursive
function that returns a `Float`. Per-lane arguments come from an
`EvalEnv` that bundles the current lane index, the input tensors,
and (for `indexed`) the buffer arena.

Warp-level primitives (`warpSum`, `warpBroadcast`,
`warpShuffleXor`) are evaluated under a **warp-collapsed**
semantics: when an explicit warp environment is provided we
fan out the body across lanes and combine; when it is not (the
single-thread default) we fall back to identity, which is the
contract WGSL itself guarantees for a single-lane subgroup.

`Prim`-level helpers (`evalPointwise`, `evalReduce`) cover the
two non-trivial constructors needed for any matmul / softmax
kernel; the remaining `Prim`s (load/store/scatter/block) are
memory plumbing and don't need a per-element semantics.

The whole evaluator is `partial`-free except for the inner loop
of `ScalarExp.eval`, which is `partial` because `ScalarExp` is
not structurally decreasing under `warpSum`. Lean still reduces
it on closed inputs at compile time, so `native_decide` is
available for sanity-checks.
-/

namespace Hesper.Circuit

/-- Evaluation environment for a single `ScalarExp` evaluation.

  - `inputs[i]`  is read by `ScalarExp.input i` at the current `laneIdx`.
  - `laneIdx`    is the lane this evaluation is for.
  - `buffers[i]` is read by `ScalarExp.indexed i addr` at the
                 (truncated) `addr` slot.
  - `warpSize`   declares the warp width for `warpSum` / `warpBroadcast`
                 / `warpShuffleXor`.
  - `warpLanes`  carries one input array per lane in the warp.  When
                 it is empty we are in single-thread mode and warp
                 ops degrade to identity (the WGSL spec says a
                 size-1 subgroup has no observable warp effect). -/
structure EvalEnv where
  inputs    : Array (Array Float) := #[]
  laneIdx   : Nat                  := 0
  buffers   : Array (Array Float) := #[]
  warpSize  : Nat                  := 32
  warpLanes : Array (Array Float) := #[]

/-! ## Float helpers

WGSL `rsqrt` / `silu` / `gelu` lower to specific approximations.
We use textbook formulae here — close enough for fixture-level
cross-checks. -/

def f32_rsqrt (x : Float) : Float := 1.0 / x.sqrt

def f32_silu (x : Float) : Float := x / (1.0 + (-x).exp)

/-- GELU approximation: `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`. -/
def f32_gelu (x : Float) : Float :=
  let c     : Float := 0.7978845608   -- sqrt(2/π)
  let inner := c * (x + 0.044715 * x * x * x)
  0.5 * x * (1.0 + inner.tanh)

/-! ## ScalarExp.eval

Structural recursion on `ScalarExp`. For warp-level primitives we
re-evaluate the body once per lane, drawing each lane's inputs
from `env.warpLanes`. -/

partial def ScalarExp.eval (env : EvalEnv) : ScalarExp → Float
  | .input i =>
    (env.inputs.getD i #[]).getD env.laneIdx 0.0
  | .const v          => v
  | .laneIdx          => env.laneIdx.toFloat
  | .indexed bufIdx addr =>
    let a := ScalarExp.eval env addr
    let idx := a.toUInt32.toNat
    (env.buffers.getD bufIdx #[]).getD idx 0.0
  | .warpSum a =>
    if env.warpLanes.isEmpty then ScalarExp.eval env a
    else Id.run do
      let mut acc : Float := 0.0
      for lane in [:env.warpSize] do
        let env' := { env with laneIdx := lane, inputs := env.warpLanes }
        acc := acc + ScalarExp.eval env' a
      pure acc
  | .warpBroadcast a =>
    let inputs' :=
      if env.warpLanes.isEmpty then env.inputs else env.warpLanes
    let env' := { env with laneIdx := 0, inputs := inputs' }
    ScalarExp.eval env' a
  | .warpShuffleXor a mask =>
    let target := env.laneIdx ^^^ mask
    let inputs' :=
      if env.warpLanes.isEmpty then env.inputs else env.warpLanes
    let env' := { env with laneIdx := target, inputs := inputs' }
    ScalarExp.eval env' a
  | .add a b          => ScalarExp.eval env a + ScalarExp.eval env b
  | .sub a b          => ScalarExp.eval env a - ScalarExp.eval env b
  | .mul a b          => ScalarExp.eval env a * ScalarExp.eval env b
  | .div a b          => ScalarExp.eval env a / ScalarExp.eval env b
  | .neg a            => -(ScalarExp.eval env a)
  | .rsqrt a          => f32_rsqrt (ScalarExp.eval env a)
  | .exp a            => (ScalarExp.eval env a).exp
  | .tanh a           => (ScalarExp.eval env a).tanh
  | .gelu a           => f32_gelu (ScalarExp.eval env a)
  | .silu a           => f32_silu (ScalarExp.eval env a)
  | .cos a            => (ScalarExp.eval env a).cos
  | .sin a            => (ScalarExp.eval env a).sin
  | .pow a b          => (ScalarExp.eval env a).pow (ScalarExp.eval env b)
  | .lt a b           =>
    if ScalarExp.eval env a < ScalarExp.eval env b then 1.0 else 0.0
  | .select c t f     =>
    if ScalarExp.eval env c != 0.0
    then ScalarExp.eval env t else ScalarExp.eval env f
  | .mod a b          =>
    -- WGSL `%` for f32: `a − b · trunc(a/b)`.
    let a' := ScalarExp.eval env a
    let b' := ScalarExp.eval env b
    a' - b' * (a' / b').toUInt32.toFloat
  | .idiv a b         =>
    let a' := (ScalarExp.eval env a).toUInt32.toNat
    let b' := (ScalarExp.eval env b).toUInt32.toNat
    if b' = 0 then 0.0 else (a' / b').toFloat
  | .fastdiv n _mp _L d =>
    -- Reference semantics ignores the (mp, L) magic constants and
    -- evaluates the abstract `n / d` directly.
    let nv := (ScalarExp.eval env n).toUInt32.toNat
    if d = 0 then 0.0 else (nv / d).toFloat
  | .toFloat a        => ScalarExp.eval env a

/-! ## Prim-level helpers

These are not full `Prim` interpreters; they execute the two
data-dependent ops (`pointwise`, `reduce`) for callers that just
want a Lean-native test harness. Memory ops (`load/store/scatter/
block`) are handled by the surrounding caller. -/

/-- Evaluate a `ScalarExp` body across `n` lanes. -/
def evalPointwise (body : ScalarExp) (inputs : Array (Array Float))
    (n : Nat) (warpSize : Nat := 32) : Array Float := Id.run do
  let mut out : Array Float := Array.replicate n 0.0
  for i in [:n] do
    let env : EvalEnv := {
      inputs := inputs, laneIdx := i, buffers := #[],
      warpSize := warpSize, warpLanes := inputs
    }
    out := out.set! i (ScalarExp.eval env body)
  pure out

/-- Evaluate a `Prim.reduce` along the last axis of `input`. -/
def evalReduce (op : ReduceOp) (input : Array Float) : Float :=
  match op with
  | .sum          => input.foldl (· + ·) 0.0
  | .sumOfSquares => input.foldl (fun acc x => acc + x * x) 0.0

end Hesper.Circuit
