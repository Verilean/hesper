import Hesper.Circuit.Dispatch_v2
import Hesper.Models.Gemma4
import Hesper.Models.Gemma4_v2

/-!
# Gemma 4 ↔ IRv2 Monolith Bridge (Phase F1)

The production `Gemma4Model` carries per-layer weights, norms, and
scratch buffers in a shape that's convenient for the legacy eager
`forwardBlock` path.  The IRv2 Monolith dispatcher
(`runMonolithicGraph`) instead consumes two lookup tables:

  `List (UInt64 × AttnBundle β)` — one attention bundle per layer
  `List (UInt64 × FFNBundle   β)` — one FFN bundle per layer

This module provides `extractMonolithBundles`, a pure (except for
`IO` to create the `CachedDispatch` refs) mapping that walks the
`Gemma4Model.blocks` array and packs each per-block field set into
its matching bundle.

After this, runtime callers can:
```
let (attnBundles, ffnBundles) ← extractMonolithBundles model state
let (_, graph) := runBuilder (forwardTokenLazyMonolith …)
runMonolithicGraph ctx graph externalBufs attnBundles ffnBundles
```

No GPU execution happens here — success is type-check parity plus
every `layerKey` being resolvable at dispatch time.

Key conventions:
- Layer `li`'s `layerKey` := `firstLayerKey + li` (matches
  `Gemma4_v2.forwardTokenLazyMonolith`).
- Shared-KV layers (`Config.kvCacheLayer li ≠ li`) share the upstream
  layer's KV cache; we resolve that here and store the resolved
  buffer into the bundle, so the dispatcher doesn't need to re-resolve.
- `freqFactors`, `outScale` etc. are `Option` in the Gemma4Block;
  if missing the bundle is still constructed with a defaulted value
  (the Monolith lowering will fail fast on None where appropriate).
-/

namespace Hesper.Models.Gemma4

open Hesper
open Hesper.Circuit.IRv2
open Hesper.Layers

/-- Build per-layer `AttnBundle` + `FFNBundle` lookup tables from a
    loaded `Gemma4Model` and its live `InferenceState` (for KV caches
    and paramsBuf).

    `firstLayerKey` is the opaque key assigned to layer 0.  Layer `li`
    gets `firstLayerKey + li`.

    This function is **pure plumbing** — it pulls already-loaded
    weights into a structure the IRv2 dispatcher can index by key.
    No GPU work, no buffer allocation. -/
def extractMonolithBundles [GPUBackend β]
    (model : Gemma4Model (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (state : InferenceState (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (firstLayerKey : UInt64 := 0xCAFE0000) :
    IO (List (UInt64 × Hesper.Circuit.IRv2.AttnBundle β) ×
        List (UInt64 × Hesper.Circuit.IRv2.FFNBundle β)) := do
  let cfg := model.config
  let mut attnBundles : List (UInt64 × Hesper.Circuit.IRv2.AttnBundle β) := []
  let mut ffnBundles  : List (UInt64 × Hesper.Circuit.IRv2.FFNBundle β)  := []
  for h : li in [0:model.blocks.size] do
    let block := model.blocks[li]
    let layerKey := firstLayerKey + li.toUInt64
    -- Resolve the KV cache for this layer — for shared-KV layers
    -- `kvCacheLayer li` returns the upstream layer's cache slot.
    let kvLi := cfg.kvCacheLayer li
    let kvCache ← if h : kvLi < state.kvCaches.size then
      pure state.kvCaches[kvLi]
    else
      throw (IO.userError s!"extractMonolithBundles: kvCacheLayer {kvLi} out of range for layer {li}")
    -- Attention-side bundle.
    let numHeads   := cfg.numAttentionHeads
    let numKVHeads := cfg.numKVHeads li
    let headDim    := cfg.headDim li
    -- Gemma 4: attention_scale = 1.0 (NOT 1/sqrt(headDim)).
    -- Q-norm RMSNorm pre-normalizes each head, so the temperature
    -- is absorbed.  See Hesper/Models/Gemma4.lean:704-706.
    let attnScale  : Float := 1.0
    let attnBundle : Hesper.Circuit.IRv2.AttnBundle β :=
      { attnNorm     := block.attnNorm
        wQ           := block.attention.wQ
        wK           := block.attention.wK
        wV           := block.attention.wV
        wO           := block.attention.wO
        qNormScale   := block.attention.qNormWeight
        kNormScale   := block.attention.kNormWeight
        postAttnNorm := block.postAttnNorm
        freqFactors  := block.ropeFreqFactors
        paramsBuf    := state.paramsBuf
        kCacheBuf    := kvCache.kBuf
        vCacheBuf    := kvCache.vBuf
        qBuf2        := state.qBuf2
        kBuf2        := state.kBuf2
        vBuf2        := state.vBuf2
        numHeads     := numHeads
        numKVHeads   := numKVHeads
        headDim      := headDim
        maxSeqLen    := cfg.maxSeqLen
        attnScale    := attnScale
        ropeBase     := cfg.ropeBase li }
    attnBundles := (layerKey, attnBundle) :: attnBundles
    -- FFN-side bundle.
    let ffnBundle : Hesper.Circuit.IRv2.FFNBundle β :=
      { ffnNorm     := block.ffnNorm
        wGate       := block.ffn.gate
        wUp         := block.ffn.up
        wDown       := block.ffn.down
        postFFNNorm := block.postFFNNorm
        geluScratch := state.geluBuf }
    ffnBundles := (layerKey, ffnBundle) :: ffnBundles
  return (attnBundles.reverse, ffnBundles.reverse)

/-- One-liner convenience: build the whole-token BlockGraph AND the
    bundle tables from the model/state at a given position.

    Returns `(graph, attnBundles, ffnBundles)` ready to feed into
    `runMonolithicGraph` or `captureMonolithicGraph`. -/
def buildMonolithTokenPlan [GPUBackend β]
    (model : Gemma4Model (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (state : InferenceState (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (pos : Nat)
    (firstInputId lastOutputId baseTensorId : Nat)
    (firstLayerKey : UInt64 := 0xCAFE0000) :
    IO (Hesper.Circuit.IRv2.BlockGraph ×
        List (UInt64 × Hesper.Circuit.IRv2.AttnBundle β) ×
        List (UInt64 × Hesper.Circuit.IRv2.FFNBundle β)) := do
  let (attnBundles, ffnBundles) ← extractMonolithBundles model state firstLayerKey
  let (_, graph) := Hesper.Circuit.IRv2.runBuilder
    (Hesper.Models.Gemma4_v2.forwardTokenLazyMonolith
       model.blocks.size firstInputId lastOutputId baseTensorId firstLayerKey pos)
  return (graph, attnBundles, ffnBundles)

end Hesper.Models.Gemma4
