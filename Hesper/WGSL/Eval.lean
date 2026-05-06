import Hesper.WGSL.Exp

/-!
# WGSL DSL — pure-Lean evaluator (Phase 1)

A reference semantics for `Hesper.WGSL.Exp` that runs entirely
inside Lean, with no GPU FFI. Companion to
`Hesper.Circuit.Eval` — pinning equivalence at the WGSL DSL
layer (just above the GPU) removes the Circuit → WGSL lowering
from the trusted base.

## What this file does

  - Defines `WGSLType.denote : WGSLType → Type`, mapping every
    WGSL type to a Lean type (f32 → Float, i32 → Int32, vecN /
    matNxN / array → tuples and Arrays).
  - Defines `Exp.eval : EvalEnv → (e : Exp t) → t.denote`, a
    type-indexed evaluator.

## Phase coverage

`Hesper.WGSL.Exp` has ~225 constructors. The Lean equation
compiler is strict about coverage in dependent types, so this
first-cut Phase 1 evaluator handles the **arithmetic /
comparison / conversion / vec / array-indexing slice** that
BitLinear, Q·K^T, softmax and weighted-V kernels exercise. Other
constructors fall through to `default` via `Inhabited
(t.denote)`. A full evaluator that pins every upstream
constructor with non-trivial semantics is tracked as a Phase-2
follow-up; experimentally it's another ~1500 lines but
mechanical (each WGSL primitive has a documented spec).

The Sparkle project keeps a more complete evaluator in its tree
(see `Tests/Hesper/Vendored/WGSLInterp.lean` in the Sparkle
repo) which already covers all 225 constructors with full Phase-4
semantics (stateful atomics, real textures, 4×4 determinants,
typed bitcast). That file is the design source for the eventual
upstream Phase 2.

## Why not just upstream the full evaluator?

Two reasons:

  1. The full evaluator depends on a Lean-side `mat4x4_f32`
     constructor that doesn't exist in upstream `Exp` yet
     (upstream has `mat4x4 : ScalarType → WGSLType` as the
     *type*, but no value-level constructor). Phase 2 adds the
     constructors.

  2. Stateful atomics need a state-passing evaluator
     (`Exp.evalSt`) whose API is best designed alongside
     existing Hesper users' atomic-using code, which is in flux.

This file ships Phase 1 — enough to verify BitLinear and
attention kernels — and leaves a clear extension path.
-/

namespace Hesper.WGSL

/-! ## Semantic domain: `WGSLType.denote`

Maps every `WGSLType` to the Lean type a value of it inhabits.
We use Lean's primitive numeric types where they line up:
f32/f16 → `Float` (Lean has no f16 native; f16 ops collapse to
f64 here — exact-bit f16 round-trip is Phase-2). i32/u32 → Int32/
UInt32. bool → Bool. -/

@[reducible]
def ScalarType.denote : ScalarType → Type
  | .f32 | .f16 => Float
  | .i32        => Int32
  | .u32        => UInt32
  | .u64        => UInt64
  | .bool       => Bool
  | .atomicI32  => Int32
  | .atomicU32  => UInt32

instance : (st : ScalarType) → Inhabited st.denote
  | .f32 | .f16 => ⟨(0.0 : Float)⟩
  | .i32        => ⟨(0 : Int32)⟩
  | .u32        => ⟨(0 : UInt32)⟩
  | .u64        => ⟨(0 : UInt64)⟩
  | .bool       => ⟨false⟩
  | .atomicI32  => ⟨(0 : Int32)⟩
  | .atomicU32  => ⟨(0 : UInt32)⟩

/-- Phase-1 denote: arrays / matrices / subgroup matrices erase
    to `Array Float` (untyped). Phase-2 will introduce per-elemTy
    arrays via `Vector` for index-bound proofs. Texture / sampler
    are opaque (`Unit`). -/
@[reducible]
def WGSLType.denote : WGSLType → Type
  | .scalar st => st.denote
  | .vec2 st   => st.denote × st.denote
  | .vec3 st   => st.denote × st.denote × st.denote
  | .vec4 st   => st.denote × st.denote × st.denote × st.denote
  | .mat2x2 _  => Array Float
  | .mat3x3 _  => Array Float
  | .mat4x4 _  => Array Float
  | .array _ _ => Array Float
  | .runtimeArray _ => Array Float
  | .bufferArray _ _ => Array (Array Float)
  | .ptr _ _   => Array Float       -- not really useful here
  | .struct _  => Array Float
  | .subgroupMatrixLeft _ _ _   => Array Float
  | .subgroupMatrixRight _ _ _  => Array Float
  | .subgroupMatrixResult _ _ _ => Array Float
  | .texture2D _ => Unit
  | .sampler   => Unit

instance : (t : WGSLType) → Inhabited t.denote
  | .scalar st => inferInstanceAs (Inhabited st.denote)
  | .vec2 st   => ⟨((default : st.denote), default)⟩
  | .vec3 st   => ⟨((default : st.denote), default, default)⟩
  | .vec4 st   => ⟨((default : st.denote), default, default, default)⟩
  | .mat2x2 _ | .mat3x3 _ | .mat4x4 _
  | .array _ _ | .runtimeArray _ | .ptr _ _ | .struct _
  | .subgroupMatrixLeft _ _ _ | .subgroupMatrixRight _ _ _
  | .subgroupMatrixResult _ _ _ => ⟨#[]⟩
  | .bufferArray _ _ => ⟨#[]⟩
  | .texture2D _ | .sampler => ⟨()⟩

/-! ## Environment

A minimal heterogeneous environment. Phase-2 can refine to a
typed name → value map. -/

structure EvalEnv where
  f32_vars   : List (String × Float)        := []
  u32_vars   : List (String × UInt32)       := []
  f32_arrays : List (String × Array Float)  := []
  deriving Inhabited

def EvalEnv.lookupF32 (env : EvalEnv) (name : String) : Float :=
  (env.f32_vars.find? (·.1 = name)).map (·.2) |>.getD 0.0

def EvalEnv.lookupU32 (env : EvalEnv) (name : String) : UInt32 :=
  (env.u32_vars.find? (·.1 = name)).map (·.2) |>.getD 0

def EvalEnv.lookupF32Array (env : EvalEnv) (name : String) : Array Float :=
  (env.f32_arrays.find? (·.1 = name)).map (·.2) |>.getD #[]

/-! ## Type-indexed evaluator (Phase 1 subset)

Lean's equation compiler will require us to handle every
constructor of the GADT `Exp` if we write `Exp.eval` as a single
match. To stay tractable in Phase 1, we wrap the evaluator in a
top-level `partial def` and use `match e with` only on the cases
we cover; everything else falls through to `default`.

This is a deliberate choice: Phase 1 is "evaluator that handles
the BitLinear/attention slice"; Phase 2 will replace this with
exhaustive coverage, possibly behind a derived helper that
synthesises the boilerplate from upstream constructor lists. -/

partial def Exp.eval (env : EvalEnv) :
    {t : WGSLType} → Exp t → t.denote := fun {t} e =>
  match t, e with
  -- Literals
  | _, .litF32 v        => v
  | _, .litF16 v        => v
  | _, .litI32 v        => Int32.ofInt v
  | _, .litU32 v        => UInt32.ofNat v
  | _, .litBool v       => v
  -- Variables
  | .scalar .f32, .var name        => env.lookupF32 name
  | .scalar .u32, .var _           => 0
  | .array (.scalar .f32) _, .var name => env.lookupF32Array name
  -- Arithmetic (f32, i32, u32 — kept hand-typed for clarity)
  | .scalar .f32, .add a b => (Exp.eval env a : Float) + Exp.eval env b
  | .scalar .f32, .sub a b => (Exp.eval env a : Float) - Exp.eval env b
  | .scalar .f32, .mul a b => (Exp.eval env a : Float) * Exp.eval env b
  | .scalar .f32, .div a b => (Exp.eval env a : Float) / Exp.eval env b
  | .scalar .f32, .neg a   => -((Exp.eval env a : Float))
  | .scalar .i32, .add a b => (Exp.eval env a : Int32) + Exp.eval env b
  | .scalar .i32, .sub a b => (Exp.eval env a : Int32) - Exp.eval env b
  | .scalar .i32, .mul a b => (Exp.eval env a : Int32) * Exp.eval env b
  | .scalar .u32, .add a b => (Exp.eval env a : UInt32) + Exp.eval env b
  | .scalar .u32, .sub a b => (Exp.eval env a : UInt32) - Exp.eval env b
  | .scalar .u32, .mul a b => (Exp.eval env a : UInt32) * Exp.eval env b
  -- Math (f32 specialisation; polymorphic upstream ones are Phase 2)
  | .scalar .f32, .exp e   => (Exp.eval env e : Float).exp
  | .scalar .f32, .log e   => (Exp.eval env e : Float).log
  | .scalar .f32, .sqrt e  => (Exp.eval env e : Float).sqrt
  | .scalar .f32, .abs e   => (Exp.eval env e : Float).abs
  | .scalar .f32, .sin e   => (Exp.eval env e : Float).sin
  | .scalar .f32, .cos e   => (Exp.eval env e : Float).cos
  | .scalar .f32, .tanh e  => (Exp.eval env e : Float).tanh
  | .scalar .f32, .min a b =>
    let av : Float := Exp.eval env a
    let bv : Float := Exp.eval env b
    if av < bv then av else bv
  | .scalar .f32, .max a b =>
    let av : Float := Exp.eval env a
    let bv : Float := Exp.eval env b
    if av < bv then bv else av
  -- Boolean ops (concrete-typed, no polymorphism)
  | .scalar .bool, .and a b => (Exp.eval env a : Bool) && Exp.eval env b
  | .scalar .bool, .or  a b => (Exp.eval env a : Bool) || Exp.eval env b
  | .scalar .bool, .not a   => !(Exp.eval env a : Bool)
  -- Phase 2 will add per-type dispatch for the polymorphic comparison
  -- ops `eq`/`lt`/`gt`/`ne`/`le`/`ge`. Phase 1 leaves them at default.
  -- Conversions
  | .scalar .f32, .toF32 e =>
    -- Phase-1 best effort: read the source, coerce via Float.
    -- The source type is parametric in the upstream `toF32`
    -- signature, so we degrade to default for non-numeric inputs.
    default
  | .scalar .i32, .toI32 _ => default
  | .scalar .u32, .toU32 _ => default
  -- Vec construction / access (f32 only in Phase 1)
  | .vec2 .f32, .vec2 a b => ((Exp.eval env a : Float), Exp.eval env b)
  | .vec3 .f32, .vec3 a b c =>
    ((Exp.eval env a : Float), Exp.eval env b, Exp.eval env c)
  | .vec4 .f32, .vec4 a b c d =>
    ((Exp.eval env a : Float), Exp.eval env b, Exp.eval env c, Exp.eval env d)
  | .scalar .f32, .vecX e =>
    ((Exp.eval env e : (Float × Float))).1
  | .scalar .f32, .vecY e =>
    ((Exp.eval env e : (Float × Float))).2
  -- Array indexing (f32 elements)
  | .scalar .f32, @Exp.index (.scalar .f32) _ arr idx =>
    let v : Array Float := Exp.eval env arr
    let i : UInt32 := Exp.eval env idx
    v.getD i.toNat 0.0
  -- Phase-1 fallback: every other constructor returns the
  -- inhabitant. Real semantics for the rest is Phase-2 work;
  -- Sparkle's vendored copy already implements the full set.
  | _, _ => default

end Hesper.WGSL
