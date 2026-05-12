/-!
# Circuit IR — pure-Lean core (no GPU backend dependency)

Contains the **dependency-free** pieces of the Circuit IR: `DType`,
`Scope`, `Shape`, `TensorRef`, `ScalarExp` (and its convenience
helpers + `fastdiv` magic-constant builder), and `ReduceOp`. None
of these reference `Hesper.Backend` or `Hesper.Layers.Linear` —
they are pure data types Lean can elaborate without any native
dependency.

This file exists so that pure-Lean tooling — most importantly the
`Hesper.Circuit.Eval` reference evaluator — can depend on the IR
without pulling in Dawn / WebGPU / X11 build requirements.

`Hesper.Circuit.IR` continues to host the GPU-flavoured pieces
(`Prim`, `Op`, `CircuitState`, `CircuitM`); it imports this file
so existing code that says `import Hesper.Circuit.IR` keeps
working unchanged.
-/

namespace Hesper.Circuit

/-- Element type of a tensor. -/
inductive DType where
  | f32
  | f16
  | u32
  | q4k   -- packed Q4_K (weight storage)
  | q6k   -- packed Q6_K
  | q8_1  -- quantized input for dp4a path
  deriving BEq, Repr, Inhabited

/-- Where the tensor lives in the memory hierarchy.  For the MVP we
    only use `.Global`; the other scopes are reserved for fusion
    passes to promote/demote tensors to. -/
inductive Scope where
  | Register   -- thread-local (fusion target)
  | Lane       -- subgroup-local
  | Shared     -- workgroup-local
  | Global     -- device-wide (buffer-backed)
  deriving BEq, Repr, Inhabited

/-- Compile-time tensor shape.  We keep it as `Array Nat` for now
    (typed-shape `List Nat` is for a later stage). -/
abbrev Shape := Array Nat

/-- A reference to a tensor in the Circuit.  `id` is allocated
    monotonically by the builder and serves as the tensor's only
    identity — the user never sees a string. -/
structure TensorRef where
  id    : Nat
  shape : Shape
  dtype : DType
  scope : Scope
  deriving Inhabited

/-- External tensors supplied by the caller (model weights, input
    activations).  These are not produced by any Op in the circuit.
    `BufT` is the GPU-backend-specific buffer handle. -/
structure ExternalTensor (BufT : Type) where
  buf   : BufT
  shape : Shape
  dtype : DType
  deriving Inhabited

/-! ## Scalar expression language

`ScalarExp` is the body of a pointwise op.  It is a pure inductive AST,
NOT a Lean closure — the whole point is that we can structurally
inspect, hash, substitute into, and rewrite it from the fusion pass.

See `Hesper.Circuit.Eval` for a reference evaluator. -/
inductive ScalarExp where
  | input  (idx : Nat)
  | const  (v : Float)
  | laneIdx
  | indexed (bufIdx : Nat) (addr : ScalarExp)
  | warpSum (a : ScalarExp)
  | warpBroadcast (a : ScalarExp)
  | warpShuffleXor (a : ScalarExp) (mask : Nat)
  | add    (a b : ScalarExp)
  | sub    (a b : ScalarExp)
  | mul    (a b : ScalarExp)
  | div    (a b : ScalarExp)
  | neg    (a : ScalarExp)
  | rsqrt  (a : ScalarExp)
  | exp    (a : ScalarExp)
  | tanh   (a : ScalarExp)
  | gelu   (a : ScalarExp)
  | silu   (a : ScalarExp)
  | cos    (a : ScalarExp)
  | sin    (a : ScalarExp)
  | pow    (a b : ScalarExp)
  | lt     (a b : ScalarExp)
  | select (cond t f : ScalarExp)
  | mod    (a b : ScalarExp)
  | idiv   (a b : ScalarExp)
  /-- Fast integer division by a constant via Granlund-Montgomery
      magic constants. See `mkFastdiv` for the convenience builder. -/
  | fastdiv (n : ScalarExp) (mp L d : Nat)
  | toFloat (a : ScalarExp)
  deriving Repr, Inhabited, BEq

namespace ScalarExp

/-- Shift every `input i` inside `e` by `k`. -/
partial def shiftInputs (k : Nat) : ScalarExp → ScalarExp
  | input i      => input (i + k)
  | const v      => const v
  | laneIdx      => laneIdx
  | indexed i a  => indexed (i + k) (shiftInputs k a)
  | warpSum a        => warpSum (shiftInputs k a)
  | warpBroadcast a  => warpBroadcast (shiftInputs k a)
  | warpShuffleXor a m => warpShuffleXor (shiftInputs k a) m
  | add a b      => add (shiftInputs k a) (shiftInputs k b)
  | sub a b      => sub (shiftInputs k a) (shiftInputs k b)
  | mul a b      => mul (shiftInputs k a) (shiftInputs k b)
  | div a b      => div (shiftInputs k a) (shiftInputs k b)
  | neg a        => neg (shiftInputs k a)
  | rsqrt a      => rsqrt (shiftInputs k a)
  | exp a        => exp (shiftInputs k a)
  | tanh a       => tanh (shiftInputs k a)
  | gelu a       => gelu (shiftInputs k a)
  | silu a       => silu (shiftInputs k a)
  | cos a        => cos (shiftInputs k a)
  | sin a        => sin (shiftInputs k a)
  | pow a b      => pow (shiftInputs k a) (shiftInputs k b)
  | lt a b       => lt (shiftInputs k a) (shiftInputs k b)
  | select c t f => select (shiftInputs k c) (shiftInputs k t) (shiftInputs k f)
  | mod a b      => mod (shiftInputs k a) (shiftInputs k b)
  | idiv a b     => idiv (shiftInputs k a) (shiftInputs k b)
  | fastdiv n mp L d => fastdiv (shiftInputs k n) mp L d
  | toFloat a    => toFloat (shiftInputs k a)

/-- Substitute `args[i]` for `input i` everywhere in `e`. -/
partial def subst (args : Array ScalarExp) : ScalarExp → ScalarExp
  | input i      => match args[i]? with | some a => a | none => input i
  | const v      => const v
  | laneIdx      => laneIdx
  | indexed i a  => indexed i (subst args a)
  | warpSum a        => warpSum (subst args a)
  | warpBroadcast a  => warpBroadcast (subst args a)
  | warpShuffleXor a m => warpShuffleXor (subst args a) m
  | add a b      => add (subst args a) (subst args b)
  | sub a b      => sub (subst args a) (subst args b)
  | mul a b      => mul (subst args a) (subst args b)
  | div a b      => div (subst args a) (subst args b)
  | neg a        => neg (subst args a)
  | rsqrt a      => rsqrt (subst args a)
  | exp a        => exp (subst args a)
  | tanh a       => tanh (subst args a)
  | gelu a       => gelu (subst args a)
  | silu a       => silu (subst args a)
  | cos a        => cos (subst args a)
  | sin a        => sin (subst args a)
  | pow a b      => pow (subst args a) (subst args b)
  | lt a b       => lt (subst args a) (subst args b)
  | select c t f => select (subst args c) (subst args t) (subst args f)
  | mod a b      => mod (subst args a) (subst args b)
  | idiv a b     => idiv (subst args a) (subst args b)
  | fastdiv n mp L d => fastdiv (subst args n) mp L d
  | toFloat a    => toFloat (subst args a)

instance : Add ScalarExp      := ⟨ScalarExp.add⟩
instance : Sub ScalarExp      := ⟨ScalarExp.sub⟩
instance : Mul ScalarExp      := ⟨ScalarExp.mul⟩
instance : Div ScalarExp      := ⟨ScalarExp.div⟩
instance : Neg ScalarExp      := ⟨ScalarExp.neg⟩
instance : OfScientific ScalarExp :=
  ⟨fun m e dp => ScalarExp.const (OfScientific.ofScientific m e dp)⟩
instance : OfNat ScalarExp n  := ⟨ScalarExp.const n.toFloat⟩

def ge (a b : ScalarExp) : ScalarExp := .lt b a
def le (a b : ScalarExp) : ScalarExp := .lt a b
def ite (c t f : ScalarExp) : ScalarExp := .select c t f

/-- Granlund-Montgomery magic numbers for a constant u32 divisor. -/
def initFastdivValues (d : Nat) : Nat × Nat :=
  if d == 0 then (0, 0) else
    let rec findL (L : Nat) (fuel : Nat) : Nat :=
      match fuel with
      | 0 => L
      | fuel+1 => if (1 <<< L) < d then findL (L+1) fuel else L
    let L := findL 0 32
    let twoL : Nat := 1 <<< L
    let num : Nat := (1 <<< 32) * (twoL - d)
    let mp : Nat := num / d + 1
    let mp32 := mp % (1 <<< 32)
    (mp32, L)

/-- Convenience: build a `fastdiv n d` ScalarExp with constants computed
    from `d`. -/
def mkFastdiv (n : ScalarExp) (d : Nat) : ScalarExp :=
  let (mp, L) := initFastdivValues d
  .fastdiv n mp L d

end ScalarExp

namespace Shape

def numel (s : Shape) : Nat := s.foldl (· * ·) 1

end Shape

/-- Reduction operators supported by `Prim.reduceLastAxis`. -/
inductive ReduceOp where
  | sum
  | sumOfSquares
  deriving Repr, Inhabited, BEq

end Hesper.Circuit
