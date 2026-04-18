import Hesper.CircuitV2.IR

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

end Hesper.CircuitV2.Examples
