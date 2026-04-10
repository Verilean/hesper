import Hesper.TTT.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WebGPU.BufferOps
import Hesper.Validation.ReferenceData

/-!
# TTT Initialization

Load golden value data and upload base weights to GPU.
-/

namespace Hesper.TTT

open Hesper.WebGPU
open Hesper.Validation.ReferenceData

/-- Load base_weights.bin and upload to GPU buffer -/
def loadBaseWeights (device : Device) (bufs : TTTBuffers) (path : String) : IO Unit := do
  let data ← loadFloatArrayFromFile path
  let bytes ← Hesper.WebGPU.BufferOps.floatArrayToBytes data
  writeBuffer device bufs.baseWeightBuf 0 bytes

/-- Load hidden_states.bin as array of per-token vectors -/
def loadHiddenStates (path : String) (seqLen hiddenDim : Nat) : IO (Array (Array Float)) := do
  let flat ← loadFloatArrayFromFile path
  let mut result := #[]
  for t in [0:seqLen] do
    let offset := t * hiddenDim
    let vec := (Array.range hiddenDim).map fun j => flat[offset + j]!
    result := result.push vec
  return result

/-- Load targets.bin (stored as float32 for simplicity) -/
def loadTargets (path : String) : IO (Array Nat) := do
  let flat ← loadFloatArrayFromFile path
  return flat.map fun f => f.toUInt64.toNat

end Hesper.TTT
