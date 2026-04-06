import Hesper.WebGPU.Types

/-!
# LoRA (Low-Rank Adaptation) Types

Core data structures for LoRA finetuning of BitNet models.

## Overview

LoRA injects trainable low-rank matrices alongside frozen ternary weights:

```
output = BitLinear(x) + (alpha / rank) * B @ A @ x
```

Where:
- BitLinear(x): frozen ternary base model output
- A: [rank, inDim] FP32 matrix (Kaiming initialized)
- B: [outDim, rank] FP32 matrix (zero initialized)
- alpha: scaling factor (typically equal to rank)

## References
- "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
- Stanford Alpaca: instruction-following finetuning
-/

namespace Hesper.LoRA

open Hesper.WebGPU

/-- LoRA configuration for finetuning -/
structure Config where
  /-- Rank of the low-rank matrices (typical: 4, 8, 16) -/
  rank : Nat := 8
  /-- Scaling factor: output is multiplied by alpha/rank -/
  alpha : Float := 8.0
  /-- Which attention projections to apply LoRA to -/
  targetModules : List String := ["wQ", "wV"]
  deriving Repr

/-- Compute the LoRA scaling factor: alpha / rank -/
def Config.scale (config : Config) : Float :=
  config.alpha / config.rank.toFloat

/-- A single LoRA weight pair (A and B matrices) for one projection.
    Forward: output += scale * B @ (A @ x)
    A is [rank, inDim], B is [outDim, rank] in row-major FP32. -/
structure Weight where
  /-- A matrix: [rank, inDim] FP32, Kaiming initialized -/
  a : Buffer
  /-- B matrix: [outDim, rank] FP32, zero initialized (so LoRA starts as identity) -/
  b : Buffer
  /-- Input dimension -/
  inDim : Nat
  /-- Output dimension -/
  outDim : Nat
  /-- Rank -/
  rank : Nat

/-- Gradient buffers for a single LoRA weight pair -/
structure WeightGrad where
  /-- Gradient for A: [rank, inDim] FP32 -/
  dA : Buffer
  /-- Gradient for B: [outDim, rank] FP32 -/
  dB : Buffer

/-- Adam optimizer state for a single LoRA weight pair -/
structure AdamState where
  /-- First moment for A -/
  mA : Buffer
  /-- Second moment for A -/
  vA : Buffer
  /-- First moment for B -/
  mB : Buffer
  /-- Second moment for B -/
  vB : Buffer

/-- LoRA adapter for a single attention layer (Q and V projections) -/
structure LayerAdapter where
  /-- LoRA weights for Q projection -/
  loraQ : Weight
  /-- LoRA weights for V projection -/
  loraV : Weight

/-- Gradient buffers for a single attention layer -/
structure LayerAdapterGrad where
  gradQ : WeightGrad
  gradV : WeightGrad

/-- Adam state for a single attention layer -/
structure LayerAdapterAdamState where
  stateQ : AdamState
  stateV : AdamState

/-- Full LoRA adapter for the entire model (all transformer layers) -/
structure Adapter where
  config : Config
  /-- Per-layer adapter weights, indexed by layer number -/
  layers : Array LayerAdapter

/-- Full gradient state for the entire model -/
structure AdapterGrad where
  layers : Array LayerAdapterGrad

/-- Full Adam optimizer state for the entire model -/
structure AdapterAdamState where
  layers : Array LayerAdapterAdamState
  /-- Current optimizer step (for bias correction) -/
  step : Nat

/-- Saved activations from forward pass, needed for backward.
    For each LoRA layer, we save the input x and intermediate h = A @ x. -/
structure SavedActivations where
  /-- Per-layer saved activations: (inputToQ, hQ, inputToV, hV) -/
  layers : Array (Buffer × Buffer × Buffer × Buffer)

/-- Training configuration -/
structure TrainConfig where
  /-- Learning rate -/
  lr : Float := 1e-4
  /-- Adam beta1 -/
  beta1 : Float := 0.9
  /-- Adam beta2 -/
  beta2 : Float := 0.999
  /-- Adam epsilon -/
  eps : Float := 1e-8
  /-- Number of training epochs -/
  epochs : Nat := 3
  /-- Log every N steps -/
  logEvery : Nat := 10
  /-- Max sequence length for training -/
  maxSeqLen : Nat := 512
  deriving Repr

end Hesper.LoRA
