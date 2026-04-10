import Hesper.TTT.Types
import Hesper.TTT.Kernels
import Hesper.Training.Loss
import Hesper.Training.SafeBuffer
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer

/-!
# TTT Inner Loop (GPU)

Single TTT step on GPU: forward → loss → gate readback → conditional backward + SGD.

For one token position:
1. Compute base logits = baseWeight @ hidden
2. Compute cross-entropy loss of base logits vs target
3. Read loss back to CPU (4 bytes) and check surprise gate
4. If gate open: forward ttt, combine, CE backward, outer product, SGD update
5. Return final_logits = base_logits + ttt_logits(updated)
-/

namespace Hesper.TTT

open Hesper.WebGPU
open Hesper.TTT.Kernels
open Hesper.Training

/-- Execute one TTT step for the current token.
    Assumes `bufs.hiddenBuf` and `bufs.targetBuf` have been written
    for this step. Returns `(baseLoss, gateOpen)`. -/
def tttStepGPU (device : Device) (config : TTTConfig) (bufs : TTTBuffers)
    : IO (Float × Bool) := do
  let v := config.vocabSize
  let h := config.hiddenDim
  let n := v * h

  -- Step 1: Base model forward (frozen weights)
  executeMatVec device bufs.baseWeightBuf bufs.hiddenBuf bufs.baseLogitsBuf v h

  -- Step 2: Cross-entropy loss of base logits vs target
  Loss.executeCrossEntropyForward device bufs.baseLogitsBuf bufs.targetBuf bufs.lossBuf v

  -- Step 3: CPU readback of loss scalar (4 bytes) for gate decision
  let baseLoss ← SafeBuffer.safeReadF32 device bufs.lossBuf

  -- Step 4: Surprise gate
  let gateOpen := baseLoss > config.tau

  if gateOpen then
    -- Gate OPEN: forward TTT, backward, SGD update, recompute final

    -- TTT forward with current weights
    executeMatVec device bufs.tttWeightBuf bufs.hiddenBuf bufs.tttLogitsBuf v h

    -- Combined logits = base + ttt
    executeVecAdd device bufs.baseLogitsBuf bufs.tttLogitsBuf bufs.combinedLogitsBuf v

    -- CE backward: dCombined = softmax(combined) - one_hot(target)
    Loss.executeCrossEntropyBackward device bufs.combinedLogitsBuf bufs.targetBuf bufs.dCombinedBuf v

    -- Weight gradient: dWeight = outer(dCombined, hidden)
    executeOuterProduct device bufs.dCombinedBuf bufs.hiddenBuf bufs.dWeightBuf v h

    -- SGD update: tttWeight -= lr * dWeight
    executeSGDUpdate device bufs.tttWeightBuf bufs.dWeightBuf n config.innerLR

    -- Recompute TTT logits with updated weights
    executeMatVec device bufs.tttWeightBuf bufs.hiddenBuf bufs.tttLogitsBuf v h

    -- Final logits = base + updated ttt
    executeVecAdd device bufs.baseLogitsBuf bufs.tttLogitsBuf bufs.finalLogitsBuf v
  else
    -- Gate CLOSED: no update, final = base + current ttt
    executeMatVec device bufs.tttWeightBuf bufs.hiddenBuf bufs.tttLogitsBuf v h
    executeVecAdd device bufs.baseLogitsBuf bufs.tttLogitsBuf bufs.finalLogitsBuf v

  return (baseLoss, gateOpen)

end Hesper.TTT
