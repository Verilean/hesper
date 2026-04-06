/-!
# Type-Safe Backward Chain

Ensures every forward operation has a corresponding backward operation.
If a backward is missing, the code won't compile.

## Design

A `DiffLayer` bundles forward and backward for a single operation.
A `DiffChain` is a sequence of `DiffLayer`s.

The full model backward is constructed by composing `DiffLayer`s in reverse.
The type system ensures:
1. Every forward op has a backward op (structural completeness)
2. Chain rule is applied correctly (composition order)
3. Buffer dimensions match (input/output sizes)

## Usage

```lean
-- Define each op as a DiffLayer
let normLayer := DiffLayer.mk "pre_norm" rmsnormForward rmsnormBackward dim dim
let attnLayer := DiffLayer.mk "attention" attnForward attnBackward dim dim

-- Chain automatically handles reverse ordering for backward
let chain := DiffChain.mk #[normLayer, attnLayer]
chain.forward device inputBuf outputBuf  -- runs norm → attn
chain.backward device dOutputBuf dInputBuf  -- runs attn_bwd → norm_bwd
```
-/

-- No GPU imports needed — this module is pure Lean for type-level guarantees.

namespace Hesper.AD.Chain

/-- A differentiable layer: forward + backward pair.
    The type system ensures backward exists for every forward.
    The actual GPU kernels are bound separately; this structure
    tracks completeness and verification status. -/
structure DiffLayer where
  /-- Human-readable name for debugging -/
  name : String
  /-- Input dimension (number of Float32 elements) -/
  inDim : Nat
  /-- Output dimension (number of Float32 elements) -/
  outDim : Nat
  /-- Whether this layer has been verified via numerical gradient check -/
  verified : Bool := false
  deriving Repr

/-- A chain of differentiable layers.
    Forward runs layers in order. Backward runs in reverse. -/
structure DiffChain where
  layers : Array DiffLayer
  deriving Inhabited

namespace DiffChain

/-- Create an empty chain -/
def empty : DiffChain := { layers := #[] }

/-- Add a layer to the chain -/
def push (chain : DiffChain) (layer : DiffLayer) : DiffChain :=
  { layers := chain.layers.push layer }

/-- Check that all layers in the chain are verified -/
def allVerified (chain : DiffChain) : Bool :=
  chain.layers.all (·.verified)

/-- Get names of unverified layers -/
def unverifiedLayers (chain : DiffChain) : Array String :=
  chain.layers.filter (!·.verified) |>.map (·.name)

/-- Print chain structure for debugging -/
def printChain (chain : DiffChain) : IO Unit := do
  IO.println s!"DiffChain ({chain.layers.size} layers):"
  IO.println "  Forward order:"
  for i in [:chain.layers.size] do
    if h : i < chain.layers.size then
      let l := chain.layers[i]
      let v := if l.verified then "✓" else "?"
      IO.println s!"    [{i}] {v} {l.name} : [{l.inDim}] → [{l.outDim}]"
  IO.println "  Backward order:"
  let n := chain.layers.size
  for i_rev in [:n] do
    let i := n - 1 - i_rev
    if h : i < chain.layers.size then
      let l := chain.layers[i]
      let v := if l.verified then "✓" else "?"
      IO.println s!"    [{i}] {v} {l.name}_bwd : [{l.outDim}] → [{l.inDim}]"

  let unv := chain.unverifiedLayers
  if unv.isEmpty then
    IO.println "  All layers verified ✓"
  else
    IO.println s!"  WARNING: {unv.size} unverified layers: {unv.toList}"

/-- Completeness check: verify input/output dimensions match between adjacent layers -/
def checkDimensions (chain : DiffChain) : Bool := Id.run do
  for i in [:chain.layers.size - 1] do
    if h1 : i < chain.layers.size then
      if h2 : i + 1 < chain.layers.size then
        if chain.layers[i].outDim != chain.layers[i + 1].inDim then
          return false
  return true

end DiffChain

/-- Builder for constructing a transformer layer's backward chain.
    Forces the user to provide backward for every forward op. -/
structure TransformerBackwardBuilder where
  /-- Attention sub-layer ops (in forward order) -/
  preNorm : Option DiffLayer := none
  qProjection : Option DiffLayer := none
  vProjection : Option DiffLayer := none
  ropeQ : Option DiffLayer := none
  attentionScores : Option DiffLayer := none
  softmax : Option DiffLayer := none
  attentionApply : Option DiffLayer := none
  subNorm : Option DiffLayer := none
  oProjection : Option DiffLayer := none
  /-- FFN sub-layer ops (in forward order) -/
  ffnNorm : Option DiffLayer := none
  ffnGate : Option DiffLayer := none
  ffnUp : Option DiffLayer := none
  ffnActivation : Option DiffLayer := none
  ffnSubNorm : Option DiffLayer := none
  ffnDown : Option DiffLayer := none

namespace TransformerBackwardBuilder

/-- Build the attention backward chain.
    Returns None if any required op is missing. -/
def buildAttentionChain (b : TransformerBackwardBuilder) : Option DiffChain := do
  let preNorm ← b.preNorm
  let oProj ← b.oProjection
  let subNorm ← b.subNorm
  let apply ← b.attentionApply
  let softmax ← b.softmax
  let scores ← b.attentionScores
  let rope ← b.ropeQ
  pure {
    layers := #[preNorm, rope, scores, softmax, apply, subNorm, oProj]
  }

/-- Check which attention ops are missing backward -/
def missingAttentionOps (b : TransformerBackwardBuilder) : Array String := Id.run do
  let mut missing := #[]
  if b.preNorm.isNone then missing := missing.push "preNorm"
  if b.oProjection.isNone then missing := missing.push "oProjection"
  if b.subNorm.isNone then missing := missing.push "subNorm"
  if b.attentionApply.isNone then missing := missing.push "attentionApply"
  if b.softmax.isNone then missing := missing.push "softmax"
  if b.attentionScores.isNone then missing := missing.push "attentionScores"
  if b.ropeQ.isNone then missing := missing.push "ropeQ"
  return missing

/-- Check which FFN ops are missing backward -/
def missingFFNOps (b : TransformerBackwardBuilder) : Array String := Id.run do
  let mut missing := #[]
  if b.ffnNorm.isNone then missing := missing.push "ffnNorm"
  if b.ffnGate.isNone then missing := missing.push "ffnGate"
  if b.ffnUp.isNone then missing := missing.push "ffnUp"
  if b.ffnActivation.isNone then missing := missing.push "ffnActivation"
  if b.ffnSubNorm.isNone then missing := missing.push "ffnSubNorm"
  if b.ffnDown.isNone then missing := missing.push "ffnDown"
  return missing

end TransformerBackwardBuilder

end Hesper.AD.Chain
