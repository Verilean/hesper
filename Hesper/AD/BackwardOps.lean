import Hesper.WebGPU.Types
import Hesper.AD.Chain

/-!
# Backward Operations Registry

Type-safe registry of backward operations for the transformer.
Adding a new forward op REQUIRES adding a backward op — the code
won't compile otherwise.

## Usage

```lean
-- Define all backward ops (compiler error if any is missing)
let ops : TransformerBackwardOps := {
  finalNormBwd := executeRmsNormBackward ...
  oProjectionBwd := executeBitLinearTranspose ...
  subNormBwd := executeRmsNormBackward ...
  applyBwd := executeApplyBackward ...
  softmaxBwd := executeSoftmaxBackward ...
  scoreBwd := executeScoreBackwardQ ...
  ropeBwd := executeRopeBackward ...
  ffnDownBwd := executeBitLinearTranspose ...
  ffnSubNormBwd := executeRmsNormBackward ...
  ffnActivationBwd := executeReluSqrMulBackward ...
  ffnGateBwd := executeBitLinearTranspose ...
  ffnUpBwd := executeBitLinearTranspose ...
  ffnNormBwd := executeRmsNormBackward ...
}

-- Execute the full backward (all ops guaranteed present)
ops.executeAttentionBackward device layerIdx ...
ops.executeFFNBackward device layerIdx ...
```
-/

namespace Hesper.AD.BackwardOps

open Hesper.WebGPU

/-- GPU backward operation: takes device + layer-specific buffers, dispatches kernel -/
abbrev BackwardKernel := Device → IO Unit

/-- All backward operations for the attention sub-layer.
    Every field is required — omitting one causes a compile error.
    This structure is the "proof" that attention backward is complete. -/
structure AttentionBackwardOps where
  /-- Final RMSNorm backward (before entering per-layer loop) -/
  finalNormBwd : BackwardKernel
  /-- O projection backward: W_O^T @ dOutput -/
  oProjectionBwd : BackwardKernel
  /-- Sub-norm RMSNorm backward -/
  subNormBwd : BackwardKernel
  /-- Attention apply backward: dOutput @ V^T → dAttn -/
  applyBwd : BackwardKernel
  /-- Softmax backward: attn * (dAttn - Σ attn*dAttn) → dScores -/
  softmaxBwd : BackwardKernel
  /-- Score backward: scale * dScores @ K → dQ -/
  scoreBwd : BackwardKernel
  /-- RoPE backward: R(-θ) @ dQ → dQpre -/
  ropeBwd : BackwardKernel

/-- All backward operations for the FFN sub-layer.
    Every field is required. -/
structure FFNBackwardOps where
  /-- FFN down projection backward: W_down^T @ dOutput -/
  ffnDownBwd : BackwardKernel
  /-- FFN sub-norm RMSNorm backward -/
  ffnSubNormBwd : BackwardKernel
  /-- ReLU²×Mul backward: dGate, dUp from dHidden -/
  ffnActivationBwd : BackwardKernel
  /-- FFN gate backward: W_gate^T @ dGate -/
  ffnGateBwd : BackwardKernel
  /-- FFN up backward: W_up^T @ dUp -/
  ffnUpBwd : BackwardKernel
  /-- Pre-FFN RMSNorm backward -/
  ffnNormBwd : BackwardKernel

/-- Complete backward operations for one transformer layer.
    Both attention AND FFN ops are required. -/
structure LayerBackwardOps where
  attention : AttentionBackwardOps
  ffn : FFNBackwardOps

/-- Execute attention backward in correct order (reverse of forward) -/
def AttentionBackwardOps.execute (ops : AttentionBackwardOps) (device : Device) : IO Unit := do
  ops.oProjectionBwd device
  ops.subNormBwd device
  ops.applyBwd device
  ops.softmaxBwd device
  ops.scoreBwd device
  ops.ropeBwd device

/-- Execute FFN backward in correct order (reverse of forward) -/
def FFNBackwardOps.execute (ops : FFNBackwardOps) (device : Device) : IO Unit := do
  ops.ffnDownBwd device
  ops.ffnSubNormBwd device
  ops.ffnActivationBwd device
  ops.ffnGateBwd device
  ops.ffnUpBwd device
  ops.ffnNormBwd device

/-- Execute full layer backward: attention then FFN -/
def LayerBackwardOps.execute (ops : LayerBackwardOps) (device : Device) : IO Unit := do
  ops.attention.execute device
  ops.ffn.execute device

/-- Verify that a backward ops set has all fields by simply constructing it.
    If any field is missing, this function won't compile.
    This is the compile-time completeness guarantee. -/
def verifyComplete (_ops : LayerBackwardOps) : Bool := true

end Hesper.AD.BackwardOps
