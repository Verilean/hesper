import Hesper.Circuit.IRv2
import Hesper.Circuit.Lowering_v2

/-!
# KV Cache write v2 — lazy Scatter block

Proof of concept for the second architectural target: represent the
"apply RoPE to K and write the result into the KV cache" pipeline as
a chain of two `Block`s (Pointwise rope → Scatter write), then fuse
them into one Scatter block via `fusePointwiseIntoScatter`.

The generated kernel reads K once, rotates it in-register, and writes
straight into the dynamic cache address — no intermediate VRAM hop.
-/

namespace Hesper.Layers.KVCache_v2

open Hesper.Circuit
open Hesper.Circuit.IRv2

/-- Simplified "rope" stand-in for the PoC: multiplies each K element
    by a per-lane factor read from a freq-factor table.  The real
    RoPE uses cos/sin pairs + position; we keep the structure (one
    pointwise op) and swap in a trivial body so we can exercise the
    Pointwise→Scatter fusion. -/
def ropeAndWriteLazy (dim posRowStride : Nat)
    (kId freqId cacheId : Nat) : BuilderM Unit := do
  declareExternal kId #[dim] .f32 .Global
  declareExternal freqId #[dim] .f32 .Global
  -- cacheId is a large destination buffer; declare a generous shape.
  declareExternal cacheId #[dim * 1024] .f32 .Global

  let kRegion    : Region := { tensorId := kId }
  let freqRegion : Region := { tensorId := freqId }

  -- Block A (Pointwise): rotated[i] = k[i] * freq[i]
  let rotated ← declareTensor #[dim] .f32 .Register
  let rotatedRegion : Region := { tensorId := rotated.id }
  let ropeBody : ScalarExp := .mul (.input 0) (.input 1)
  emitBlock
    { reads  := #[kRegion, freqRegion]
      writes := #[rotatedRegion]
      body   := .Pointwise ropeBody }

  -- Block B (Scatter): cache[pos * posRowStride + i] = rotated[i]
  -- `.input 0` here is the rotated value (only read in B).
  -- `indexExpr` is computed per-lane.  We use `laneIdx` for `i` and
  -- fold `pos * posRowStride` as a constant for simplicity (real code
  -- would read pos from a buffer; see `paramsBuf` in Gemma4).
  let posOffset : Nat := posRowStride   -- placeholder: pos=1 row
  let idxE : ScalarExp :=
    .add (.const posOffset.toFloat) (.toFloat .laneIdx)
  let applyE : ScalarExp := .input 0
  emitBlock
    { reads  := #[rotatedRegion]
      writes := #[{ tensorId := cacheId }]
      body   := .Scatter idxE applyE }

/-- End-to-end: build → fuse pointwise→scatter → lower. -/
def compile (dim posRowStride : Nat)
    (kId freqId cacheId : Nat) (workgroupSize : Nat := 64) :
    BlockGraph × Hesper.WGSL.Monad.ShaderM Unit :=
  let (_, g) := runBuilder (ropeAndWriteLazy dim posRowStride kId freqId cacheId)
  let fused := fusePointwiseIntoScatter g
  (fused, lowerBlockGraph fused workgroupSize)

end Hesper.Layers.KVCache_v2
