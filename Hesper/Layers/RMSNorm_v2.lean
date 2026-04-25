import Hesper.Circuit.IRv2
import Hesper.Circuit.Lowering_v2

/-!
# RMSNorm v2 — lazy forward returning a BlockGraph

Proof-of-concept lazy rewrite of `Hesper.Layers.RMSNorm`'s forward
path.  Where v1 emits `IO Unit` kernel launches eagerly, v2 emits a
pure `BuilderM` computation that adds Block nodes to a BlockGraph.

Running the program is a three-step process the caller drives:

  1. Build the AST:
       `let (_, g) := runBuilder (RMSNorm_v2.forwardLazy N eps inId outId)`

  2. Fuse:
       `let g' := fusePointwiseIntoReduce g`

  3. Lower to ShaderM (for WGSL/PTX codegen):
       `let kernel : ShaderM Unit := lowerBlockGraph g'`

No IO happens anywhere in this file — the `BuilderM` state monad is
pure, and `ShaderM` is also a pure description of the kernel to be
compiled later.
-/

namespace Hesper.Layers.RMSNorm_v2

open Hesper.Circuit
open Hesper.Circuit.IRv2

/-- Lazy RMSNorm forward.  Constructs the unfused 2-block form:
    Reduce (sum of squares) → Pointwise (normalize & scale).

    Inputs:
    - `dim`  – length of the input vector (= shape[0])
    - `eps`  – RMSNorm epsilon
    - `xId`  – tensor id of the pre-declared input (Global scope)
    - `scaleId` – tensor id of the pre-declared scale vector
    - `outId` – tensor id of the pre-declared output (Global scope)

    The fusion pass will collapse these 2 blocks into 1 Reduce block
    whose apply body is `(x / sqrt(Σx²/N + eps)) * scale`. -/
def forwardLazy (dim : Nat) (eps : Float)
    (xId scaleId outId : Nat) : BuilderM Unit := do
  -- Register the externally-owned tensors so lowering can look up
  -- their real shapes instead of defaulting to empty / 1-element.
  declareExternal xId #[dim] .f32 .Global
  declareExternal scaleId #[dim] .f32 .Global
  declareExternal outId #[dim] .f32 .Global

  let xRegion     : Region := { tensorId := xId }
  let scaleRegion : Region := { tensorId := scaleId }
  let outRegion   : Region := { tensorId := outId }

  -- Intermediate scalar holding Σ x² — fusion will eliminate this.
  let mid ← declareTensor #[1] .f32 .Register
  let midRegion : Region := { tensorId := mid.id }

  -- Block 1: Reduce Σx² into `mid`.  applyBody = identity so the
  -- reduced scalar passes straight through.
  emitBlock
    { reads := #[xRegion]
      writes := #[midRegion]
      body := .Reduce ReduceOp.sumOfSquares xRegion 0 (.input 0) }

  -- Block 2: out[i] = x[i] * scale[i] * rsqrt(mid/N + eps)
  -- reads: [x, scale, mid]
  -- input 0 = x[i], input 1 = scale[i], input 2 = mid scalar
  let invRms : ScalarExp :=
    .rsqrt (.add (.div (.input 2) (.const dim.toFloat)) (.const eps))
  let body : ScalarExp :=
    .mul (.mul (.input 0) (.input 1)) invRms
  emitBlock
    { reads := #[xRegion, scaleRegion, midRegion]
      writes := #[outRegion]
      body := .Pointwise body }

/-- End-to-end helper: build → fuse → lower.  Returns the fused
    BlockGraph and the ShaderM kernel program.  Pure — no IO yet. -/
def compile (dim : Nat) (eps : Float)
    (xId scaleId outId : Nat) (workgroupSize : Nat := 256) :
    BlockGraph × Hesper.WGSL.Monad.ShaderM Unit :=
  let (_, g) := runBuilder (forwardLazy dim eps xId scaleId outId)
  let fused := fusePointwiseIntoReduce g
  (fused, lowerBlockGraph fused workgroupSize)

end Hesper.Layers.RMSNorm_v2
