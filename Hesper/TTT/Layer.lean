import Hesper.TTT.Types
import Hesper.TTT.InnerLoop
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WebGPU.BufferOps

/-!
# TTT Layer (GPU)

Sequence-level orchestration: upload per-token data, call tttStepGPU,
readback results for validation.
-/

namespace Hesper.TTT

open Hesper.WebGPU

/-- Upload a hidden state vector to GPU -/
def uploadHiddenState (device : Device) (bufs : TTTBuffers) (hidden : Array Float) : IO Unit := do
  let bytes ← BufferOps.floatArrayToBytes hidden
  writeBuffer device bufs.hiddenBuf 0 bytes

/-- Upload a target token ID (as u32) to GPU -/
def uploadTarget (device : Device) (bufs : TTTBuffers) (target : Nat) : IO Unit := do
  let bytes := BufferOps.uint32ToBytes target.toUInt32
  writeBuffer device bufs.targetBuf 0 bytes

/-- Download a float32 array from GPU -/
def downloadFloatBuffer (device : Device) (buf : Buffer) (n : Nat) : IO (Array Float) :=
  BufferOps.downloadFloatArray device buf n

/-- Run the full TTT sequence on GPU, collecting per-step results for validation.
    Each step uploads the token's hidden state and target, runs tttStepGPU,
    then reads back the relevant buffers. -/
def tttSequenceGPU (device : Device) (config : TTTConfig) (bufs : TTTBuffers)
    (hiddenStates : Array (Array Float)) (targets : Array Nat)
    : IO (Array TTTStepResultGPU) := do
  let seqLen := hiddenStates.size
  let mut results : Array TTTStepResultGPU := #[]

  for t in [0:seqLen] do
    -- Upload this step's input
    uploadHiddenState device bufs hiddenStates[t]!
    uploadTarget device bufs targets[t]!

    -- Run one TTT step
    let (baseLoss, gateOpen) ← tttStepGPU device config bufs

    -- Readback for validation
    let baseLogits ← downloadFloatBuffer device bufs.baseLogitsBuf config.vocabSize
    let finalLogits ← downloadFloatBuffer device bufs.finalLogitsBuf config.vocabSize
    let tttWeights ← downloadFloatBuffer device bufs.tttWeightBuf (config.vocabSize * config.hiddenDim)

    results := results.push {
      baseLogits
      baseLoss
      gateOpen
      finalLogits
      tttWeightsAfter := tttWeights
    }

  return results

end Hesper.TTT
