import Hesper.CircuitV2.IR
import Hesper.CircuitV2.Lowering

/-!
# Circuit DSL v2 — Phase B reference examples

These examples exist only to verify that the v2 AST types hold together:

* the scope-parameter makes `Load`/`Store` well-formed
* `StateToken` correctly orders two successive Scatters
* `scatterStatic` demands (and consumes) a `< dim` proof
* the builder monad threads the token so the caller can't accidentally
  drop it

Everything here is **elaboration-only** — no kernel is emitted, no
WGSL/PTX lowering happens.
-/

namespace Hesper.CircuitV2.Examples

open Hesper.CircuitV2 Scope

/-- Round-trip #1: load a VRAM row into SRAM, scatter a Reg tensor into
    a different VRAM slot, then back into VRAM.  Exists so the compiler
    checks that scopes compose the way we expect. -/
def kvCacheWriteLoop (headDim : Nat) (maxSeq : Nat)
    (kCache : CircuitTensor .VRAM [maxSeq, headDim] .f16)
    (newK   : CircuitTensor .Reg [headDim] .f16)
    (pos    : Nat) (hPos : pos < maxSeq)
    (tok0   : StateToken .VRAM)
    : CircuitM (StateToken .VRAM) := do
  -- One scatter: kCache[pos] ← newK
  let tok1 ← CircuitM.scatterStatic kCache newK pos hPos tok0
  return tok1

/-- Round-trip #2: a chain of two scatters proves the StateToken
    threading — the second scatter cannot be issued with `tok0`, it
    must consume `tok1`.  If we accidentally pass `tok0` twice, Lean
    won't complain (tokens are just values) — linearity needs a
    monad-linearity plugin.  For now this doc just shows the intent. -/
def twoWrites (headDim : Nat) (maxSeq : Nat)
    (kCache : CircuitTensor .VRAM [maxSeq, headDim] .f16)
    (k0 k1  : CircuitTensor .Reg [headDim] .f16)
    (pos0 pos1 : Nat)
    (h0 : pos0 < maxSeq) (h1 : pos1 < maxSeq)
    (tok : StateToken .VRAM)
    : CircuitM (StateToken .VRAM) := do
  let tok ← CircuitM.scatterStatic kCache k0 pos0 h0 tok
  let tok ← CircuitM.scatterStatic kCache k1 pos1 h1 tok
  return tok

/-- Round-trip #3: load → store round-trip via SRAM.  Verifies that
    `loadToSram` produces a handle whose scope is `.SRAM` and that
    `storeTo` accepts it into a `.VRAM` destination. -/
def loadStoreRoundTrip (dim : Nat)
    (src : CircuitTensor .VRAM [dim] .f32)
    (dst : CircuitTensor .VRAM [dim] .f32)
    (tok : StateToken .VRAM)
    : CircuitM (StateToken .VRAM) := do
  let s ← CircuitM.loadToSram src
  let tok ← CircuitM.storeTo s dst 0 tok
  return tok

/-! ## Lowering round-trip: pointwise → ShaderM

Emit a single pointwise op (`y[i] = x[i] * 2 + 1`) through the v2
builder and lower it to a `LoweredPrim`.  A compile-time sanity check
that the pipeline wiring is correct. -/

open Hesper.Circuit (ScalarExp)

/-- Build a single pointwise op that doubles its input and adds one. -/
def doubleAddOne (inId outId : Nat) (n : Nat) : CircuitM Unit := do
  let body : ScalarExp :=
    .add (.mul (.input 0) (.const 2.0)) (.const 1.0)
  CircuitM.emit (Prim.pointwise #[inId] body [n] .f32 .VRAM)
  -- consume outId so it's not flagged unused
  let _ := outId
  pure ()

/-- End-to-end: build, lower, inspect.  This definition is used only to
    check that `lowerAll` produces exactly one `LoweredPrim` with the
    expected dispatch shape.  The returned plan is not executed. -/
def pointwiseRoundTrip : Array LoweredPrim :=
  let initSt : BuilderState := { nextId := 0, nextTick := 0, ops := #[] }
  let (_, st) := (doubleAddOne 0 1 128).run initSt
  let env : ShapeEnv := #[(0, [128]), (1, [128])]
  lowerAll st env

#eval pointwiseRoundTrip.size   -- expect: 1
#eval (pointwiseRoundTrip[0]!).name   -- expect: "v2_pointwise_n128"

end Hesper.CircuitV2.Examples
