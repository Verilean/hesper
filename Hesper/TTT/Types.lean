import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer

/-!
# TTT Core Types (GPU)

Type definitions for GPU-accelerated Surprise-Gated Residual Test-Time Training.

Architecture:
- Frozen base model provides general knowledge
- Lightweight plastic TTT module (full dense matrix, zero-initialized)
- Surprise gate: update only when base model loss > tau
- Output: `final_logits = base_logits + ttt_logits`
-/

namespace Hesper.TTT

open Hesper.WebGPU

/-- Configuration for TTT layer -/
structure TTTConfig where
  hiddenDim : Nat
  vocabSize : Nat
  innerLR : Float
  tau : Float
  deriving Inhabited, Repr

/-- Pre-allocated GPU buffers for the full TTT pipeline.
    All scratch buffers are created once and reused across sequence steps. -/
structure TTTBuffers where
  -- Weights
  baseWeightBuf : Buffer    -- [vocabSize × hiddenDim] f32, read-only after init
  tttWeightBuf : Buffer     -- [vocabSize × hiddenDim] f32, mutable (zero-init)
  -- Per-step inputs (re-written each step)
  hiddenBuf : Buffer        -- [hiddenDim] f32
  targetBuf : Buffer        -- [1] u32
  -- Scratch: logits
  baseLogitsBuf : Buffer    -- [vocabSize] f32
  tttLogitsBuf : Buffer     -- [vocabSize] f32
  combinedLogitsBuf : Buffer -- [vocabSize] f32
  finalLogitsBuf : Buffer   -- [vocabSize] f32
  -- Scratch: gradients
  dCombinedBuf : Buffer     -- [vocabSize] f32 (CE backward output)
  dWeightBuf : Buffer       -- [vocabSize × hiddenDim] f32 (outer product)
  -- Scratch: loss
  lossBuf : Buffer          -- [1] f32 (scalar loss for gate readback)

/-- Result of a single GPU TTT step, with data read back for validation -/
structure TTTStepResultGPU where
  baseLogits : Array Float     -- [vocabSize]
  baseLoss : Float
  gateOpen : Bool
  finalLogits : Array Float    -- [vocabSize]
  tttWeightsAfter : Array Float -- [vocabSize × hiddenDim]
  deriving Inhabited, Repr

/-- Allocate all GPU buffers for the TTT pipeline.
    tttWeightBuf is zero-initialized; all others are scratch. -/
def createTTTBuffers (device : Device) (config : TTTConfig) : IO TTTBuffers := do
  let mkF32Buf := fun (n : Nat) => createBuffer device {
    size := (n * 4).toUSize  -- f32 = 4 bytes
    usage := [.storage, .copySrc, .copyDst]
    mappedAtCreation := false
  }
  let weightSize := config.vocabSize * config.hiddenDim
  return {
    baseWeightBuf := ← mkF32Buf weightSize
    tttWeightBuf := ← do
      let buf ← mkF32Buf weightSize
      -- Zero-initialize TTT weights so initial ttt_logits = 0
      let mut zeros := ByteArray.empty
      for _ in [0:weightSize * 4] do
        zeros := zeros.push 0
      writeBuffer device buf 0 zeros
      pure buf
    hiddenBuf := ← mkF32Buf config.hiddenDim
    targetBuf := ← createBuffer device {
      size := 4  -- 1 × u32
      usage := [.storage, .copySrc, .copyDst]
      mappedAtCreation := false
    }
    baseLogitsBuf := ← mkF32Buf config.vocabSize
    tttLogitsBuf := ← mkF32Buf config.vocabSize
    combinedLogitsBuf := ← mkF32Buf config.vocabSize
    finalLogitsBuf := ← mkF32Buf config.vocabSize
    dCombinedBuf := ← mkF32Buf config.vocabSize
    dWeightBuf := ← mkF32Buf weightSize
    lossBuf := ← mkF32Buf 1
  }

end Hesper.TTT
