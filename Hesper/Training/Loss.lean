import Hesper.WGSL.Monad
import Hesper.WGSL.Exp
import Hesper.Backend

/-!
# Cross-Entropy Loss for Language Model Training

Implements numerically stable cross-entropy loss for teacher-forcing:

```
loss = -log(softmax(logits)[target])
     = -logits[target] + log(sum(exp(logits - max(logits))))
```

Backward:
```
dLogits[i] = softmax(logits)[i] - (i == target ? 1 : 0)
```

This elegant form means the backward pass is just softmax minus one-hot.
-/

namespace Hesper.Training.Loss

open Hesper.WGSL
open Hesper.WGSL.Monad

/-! ## Forward: Cross-Entropy Loss -/

/-- GPU kernel: Compute cross-entropy loss for a single token.

    Uses two-pass approach for numerical stability:
    1. Find max(logits) via parallel reduction
    2. Compute log-sum-exp and loss

    Input: logits [vocabSize], target [1] (u32 token ID)
    Output: loss [1] (scalar float)

    Uses workgroup shared memory for reductions. -/
def crossEntropyForwardKernel (vocabSize : Nat) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let _gid ← ShaderM.globalId
  let lid ← ShaderM.localId
  let tidX := Exp.vec3X lid
  let numSteps := Nat.log2 workgroupSize

  let _logits ← ShaderM.declareInputBuffer "logits" (.array (.scalar .f32) vocabSize)
  let _target ← ShaderM.declareInputBuffer "target_id" (.array (.scalar .u32) 1)
  let _loss ← ShaderM.declareOutputBuffer "loss" (.array (.scalar .f32) 1)

  -- Shared memory for reductions
  ShaderM.sharedNamed "smax" (.array (.scalar .f32) workgroupSize)
  ShaderM.sharedNamed "ssum" (.array (.scalar .f32) workgroupSize)

  -- Phase 1: Find max(logits) using strided parallel scan
  let (localMaxName, localMax) ← ShaderM.varRef (.scalar .f32) (Exp.litF32 (-1.0e30))
  ShaderM.loop tidX (Exp.litU32 vocabSize) (Exp.litU32 workgroupSize) fun i => do
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := vocabSize) "logits" i
    ShaderM.assign localMaxName (Exp.max localMax val)

  ShaderM.writeWorkgroup (ty := .scalar .f32) "smax" tidX localMax
  ShaderM.barrier

  -- Tree reduction for max (unrolled at Lean meta level)
  ShaderM.staticLoop numSteps fun step => do
    let s := workgroupSize >>> (step + 1)
    ShaderM.if_ (Exp.lt tidX (Exp.litU32 s)) (do
      let other ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "smax" (Exp.add tidX (Exp.litU32 s))
      let cur ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "smax" tidX
      ShaderM.writeWorkgroup (ty := .scalar .f32) "smax" tidX (Exp.max cur other)
    ) (pure ())
    ShaderM.barrier

  let globalMax ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "smax" (Exp.litU32 0)

  -- Phase 2: Compute sum(exp(logits - max))
  let (localSumName, localSum) ← ShaderM.varRef (.scalar .f32) (Exp.litF32 0.0)
  ShaderM.loop tidX (Exp.litU32 vocabSize) (Exp.litU32 workgroupSize) fun i => do
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := vocabSize) "logits" i
    let expVal := Exp.exp (Exp.sub val globalMax)
    ShaderM.assign localSumName (Exp.add localSum expVal)

  ShaderM.writeWorkgroup (ty := .scalar .f32) "ssum" tidX localSum
  ShaderM.barrier

  -- Tree reduction for sum (unrolled at Lean meta level)
  ShaderM.staticLoop numSteps fun step => do
    let s := workgroupSize >>> (step + 1)
    ShaderM.if_ (Exp.lt tidX (Exp.litU32 s)) (do
      let other ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "ssum" (Exp.add tidX (Exp.litU32 s))
      let cur ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "ssum" tidX
      ShaderM.writeWorkgroup (ty := .scalar .f32) "ssum" tidX (Exp.add cur other)
    ) (pure ())
    ShaderM.barrier

  -- Thread 0 computes final loss
  ShaderM.if_ (Exp.eq tidX (Exp.litU32 0)) (do
    let totalSum ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "ssum" (Exp.litU32 0)
    let logSumExp := Exp.add globalMax (Exp.log totalSum)
    let targetId ← ShaderM.readBuffer (ty := .scalar .u32) (n := 1) "target_id" (Exp.litU32 0)
    let targetLogit ← ShaderM.readBuffer (ty := .scalar .f32) (n := vocabSize) "logits" targetId
    -- loss = -targetLogit + logSumExp = logSumExp - targetLogit
    let lossVal := Exp.sub logSumExp targetLogit
    ShaderM.writeBuffer (ty := .scalar .f32) "loss" (Exp.litU32 0) lossVal
  ) (pure ())

/-- Execute cross-entropy loss forward.
    Returns loss value by reading back from GPU. -/
@[inline]
def executeCrossEntropyForward [Hesper.GPUBackend β] (ctx : β)
    (logitsBuf targetBuf lossBuf : Hesper.GPUBackend.Buf β)
    (vocabSize : Nat) : IO Unit := do
  let workgroupSize := 256
  Hesper.GPUBackend.execute ctx
    (crossEntropyForwardKernel vocabSize workgroupSize)
    [("logits", logitsBuf), ("target_id", targetBuf), ("loss", lossBuf)]
    { workgroupSize := {x := workgroupSize}, numWorkgroups := (1, 1, 1) }

/-- Cross-entropy forward with GPU-side loss accumulation.
    Adds the per-token loss to an accumulator buffer instead of overwriting.
    This allows batching all tokens' loss computation without CPU readback. -/
def crossEntropyForwardAccumKernel (vocabSize : Nat) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let lid ← ShaderM.localId
  let tidX := Exp.vec3X lid

  let _logits ← ShaderM.declareInputBuffer "logits" (.array (.scalar .f32) vocabSize)
  let _target ← ShaderM.declareInputBuffer "target_id" (.array (.scalar .u32) 1)
  let _lossAccum ← ShaderM.declareOutputBuffer "loss_accum" (.array (.scalar .f32) 1)

  ShaderM.sharedNamed "smax" (.array (.scalar .f32) workgroupSize)
  ShaderM.sharedNamed "ssum" (.array (.scalar .f32) workgroupSize)

  -- Phase 1: Find max(logits)
  let maxVar ← ShaderM.var (.scalar .f32) (Exp.litF32 (-1.0e30))
  ShaderM.loop tidX (Exp.litU32 vocabSize) (Exp.litU32 workgroupSize) fun i => do
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := vocabSize) "logits" i
    ShaderM.assign maxVar (Exp.max (Exp.var maxVar) val)
  ShaderM.writeWorkgroup (ty := .scalar .f32) "smax" tidX (Exp.var maxVar)
  ShaderM.barrier

  let numSteps := Nat.log2 workgroupSize
  ShaderM.staticLoop numSteps fun step => do
    let s := workgroupSize >>> (step + 1)
    ShaderM.if_ (Exp.lt tidX (Exp.litU32 s)) (do
      let other ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "smax" (Exp.add tidX (Exp.litU32 s))
      let cur ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "smax" tidX
      ShaderM.writeWorkgroup (ty := .scalar .f32) "smax" tidX (Exp.max cur other)
    ) (pure ())
    ShaderM.barrier

  let globalMax ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "smax" (Exp.litU32 0)

  -- Phase 2: sum(exp(logits - max))
  let sumVar ← ShaderM.var (.scalar .f32) (Exp.litF32 0.0)
  ShaderM.loop tidX (Exp.litU32 vocabSize) (Exp.litU32 workgroupSize) fun i => do
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := vocabSize) "logits" i
    ShaderM.assign sumVar (Exp.add (Exp.var sumVar) (Exp.exp (Exp.sub val globalMax)))
  ShaderM.writeWorkgroup (ty := .scalar .f32) "ssum" tidX (Exp.var sumVar)
  ShaderM.barrier

  ShaderM.staticLoop numSteps fun step => do
    let s := workgroupSize >>> (step + 1)
    ShaderM.if_ (Exp.lt tidX (Exp.litU32 s)) (do
      let other ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "ssum" (Exp.add tidX (Exp.litU32 s))
      let cur ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "ssum" tidX
      ShaderM.writeWorkgroup (ty := .scalar .f32) "ssum" tidX (Exp.add cur other)
    ) (pure ())
    ShaderM.barrier

  -- Thread 0: ACCUMULATE loss (add to existing value, not overwrite)
  ShaderM.if_ (Exp.eq tidX (Exp.litU32 0)) (do
    let totalSum ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "ssum" (Exp.litU32 0)
    let logSumExp := Exp.add globalMax (Exp.log totalSum)
    let targetId ← ShaderM.readBuffer (ty := .scalar .u32) (n := 1) "target_id" (Exp.litU32 0)
    let targetLogit ← ShaderM.readBuffer (ty := .scalar .f32) (n := vocabSize) "logits" targetId
    let lossVal := Exp.sub logSumExp targetLogit
    -- Accumulate: loss_accum[0] += lossVal
    let oldLoss ← ShaderM.readBuffer (ty := .scalar .f32) (n := 1) "loss_accum" (Exp.litU32 0)
    ShaderM.writeBuffer (ty := .scalar .f32) "loss_accum" (Exp.litU32 0) (Exp.add oldLoss lossVal)
  ) (pure ())

/-- Execute cross-entropy forward with GPU-side loss accumulation.
    Call this per-token; loss accumulates on GPU. Read once at end of example. -/
@[inline]
def executeCrossEntropyForwardAccum [Hesper.GPUBackend β] (ctx : β)
    (logitsBuf targetBuf lossAccumBuf : Hesper.GPUBackend.Buf β)
    (vocabSize : Nat) : IO Unit := do
  let workgroupSize := 256
  Hesper.GPUBackend.execute ctx
    (crossEntropyForwardAccumKernel vocabSize workgroupSize)
    [("logits", logitsBuf), ("target_id", targetBuf), ("loss_accum", lossAccumBuf)]
    { workgroupSize := {x := workgroupSize}, numWorkgroups := (1, 1, 1) }

/-! ## Backward: dLogits = softmax(logits) - one_hot(target) -/

/-- GPU kernel: Compute gradient of cross-entropy loss w.r.t. logits.

    dLogits[i] = softmax(logits)[i] - (i == target ? 1 : 0)

    Two phases:
    1. Compute max and sum-exp (same as forward) via shared memory
    2. Each thread computes its softmax value and subtracts one-hot

    Input: logits [vocabSize], target [1] (u32)
    Output: dLogits [vocabSize] -/
def crossEntropyBackwardKernel (vocabSize : Nat) (workgroupSize : Nat := 256) : ShaderM Unit := do
  let _gid ← ShaderM.globalId
  let lid ← ShaderM.localId
  let tidX := Exp.vec3X lid
  let numSteps := Nat.log2 workgroupSize

  let _logits ← ShaderM.declareInputBuffer "logits" (.array (.scalar .f32) vocabSize)
  let _target ← ShaderM.declareInputBuffer "target_id" (.array (.scalar .u32) 1)
  let _dLogits ← ShaderM.declareOutputBuffer "dLogits" (.array (.scalar .f32) vocabSize)

  ShaderM.sharedNamed "smax" (.array (.scalar .f32) workgroupSize)
  ShaderM.sharedNamed "ssum" (.array (.scalar .f32) workgroupSize)

  -- Phase 1: Find max(logits)
  let (localMaxName, localMax) ← ShaderM.varRef (.scalar .f32) (Exp.litF32 (-1.0e30))
  ShaderM.loop tidX (Exp.litU32 vocabSize) (Exp.litU32 workgroupSize) fun i => do
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := vocabSize) "logits" i
    ShaderM.assign localMaxName (Exp.max localMax val)

  ShaderM.writeWorkgroup (ty := .scalar .f32) "smax" tidX localMax
  ShaderM.barrier

  ShaderM.staticLoop numSteps fun step => do
    let s := workgroupSize >>> (step + 1)
    ShaderM.if_ (Exp.lt tidX (Exp.litU32 s)) (do
      let other ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "smax" (Exp.add tidX (Exp.litU32 s))
      let cur ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "smax" tidX
      ShaderM.writeWorkgroup (ty := .scalar .f32) "smax" tidX (Exp.max cur other)
    ) (pure ())
    ShaderM.barrier

  let globalMax ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "smax" (Exp.litU32 0)

  -- Phase 2: Compute sum(exp(logits - max))
  let (localSumName, localSum) ← ShaderM.varRef (.scalar .f32) (Exp.litF32 0.0)
  ShaderM.loop tidX (Exp.litU32 vocabSize) (Exp.litU32 workgroupSize) fun i => do
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := vocabSize) "logits" i
    ShaderM.assign localSumName (Exp.add localSum (Exp.exp (Exp.sub val globalMax)))

  ShaderM.writeWorkgroup (ty := .scalar .f32) "ssum" tidX localSum
  ShaderM.barrier

  ShaderM.staticLoop numSteps fun step => do
    let s := workgroupSize >>> (step + 1)
    ShaderM.if_ (Exp.lt tidX (Exp.litU32 s)) (do
      let other ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "ssum" (Exp.add tidX (Exp.litU32 s))
      let cur ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "ssum" tidX
      ShaderM.writeWorkgroup (ty := .scalar .f32) "ssum" tidX (Exp.add cur other)
    ) (pure ())
    ShaderM.barrier

  let totalSum ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := workgroupSize) "ssum" (Exp.litU32 0)
  let targetId ← ShaderM.readBuffer (ty := .scalar .u32) (n := 1) "target_id" (Exp.litU32 0)

  -- Phase 3: Compute dLogits[i] = softmax[i] - one_hot[i]
  ShaderM.loop tidX (Exp.litU32 vocabSize) (Exp.litU32 workgroupSize) fun i => do
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := vocabSize) "logits" i
    let softmaxVal := Exp.div (Exp.exp (Exp.sub val globalMax)) totalSum
    -- Subtract 1.0 if this is the target token
    let isTarget := Exp.eq i targetId
    let oneHot := Exp.select isTarget (Exp.litF32 1.0) (Exp.litF32 0.0)
    let grad := Exp.sub softmaxVal oneHot
    ShaderM.writeBuffer (ty := .scalar .f32) "dLogits" i grad

/-- Execute cross-entropy backward: dLogits = softmax(logits) - one_hot(target) -/
@[inline]
def executeCrossEntropyBackward [Hesper.GPUBackend β] (ctx : β)
    (logitsBuf targetBuf dLogitsBuf : Hesper.GPUBackend.Buf β)
    (vocabSize : Nat) : IO Unit := do
  let workgroupSize := 256
  Hesper.GPUBackend.execute ctx
    (crossEntropyBackwardKernel vocabSize workgroupSize)
    [("logits", logitsBuf), ("target_id", targetBuf), ("dLogits", dLogitsBuf)]
    { workgroupSize := {x := workgroupSize}, numWorkgroups := (1, 1, 1) }

end Hesper.Training.Loss
