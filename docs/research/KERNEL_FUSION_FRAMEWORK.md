# Kernel Fusion Framework Design

## Problem

Training is 9x slower than inference because backward consists of many
small GPU kernel dispatches. Each dispatch has overhead (~0.1ms) even
if the actual computation is tiny.

Current backward: ~600 dispatches per output token
- 30 layers × (7 attention backward + 6 FFN backward + 7 save activations) = 600

## Solution: Automatic Kernel Fusion via ShaderM Composition

ShaderM is a monad that generates WGSL code. Two ShaderM computations
can be composed into one, producing a single WGSL shader that does
both operations in one dispatch.

### Current (unfused):
```lean
-- 2 dispatches, 2 GPU submits in batch
executeRmsNormBackward device xBuf gammaBuf dOutBuf dInBuf dim
executeBitLinearTranspose device wO dInBuf dOutputBuf
```

### Fused:
```lean
-- 1 dispatch, 1 GPU submit
executeFusedRmsNormAndTranspose device xBuf gammaBuf dOutBuf wO dOutputBuf dim
```

## Framework Architecture

```
┌─────────────────────────────────────────────────────┐
│  FusionBuilder                                       │
│                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │ Op A     │───▶│ Op B     │───▶│ Op C     │      │
│  │ ShaderM  │    │ ShaderM  │    │ ShaderM  │      │
│  └──────────┘    └──────────┘    └──────────┘      │
│       │               │               │              │
│       ▼               ▼               ▼              │
│  ┌──────────────────────────────────────────┐       │
│  │  Fused ShaderM (single WGSL shader)      │       │
│  │  - Intermediate buffers eliminated       │       │
│  │  - One dispatch instead of three         │       │
│  └──────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────┘
```

## Fusion Categories

### Category 1: Element-wise Chain Fusion
Operations that are element-wise (each output[i] depends only on input[i])
can always be fused by inlining.

Example: RMSNorm output → scale → clamp
```lean
-- Unfused: 3 dispatches
rmsNormForward ...
scaleKernel ...
clampKernel ...

-- Fused: 1 dispatch
fusedRmsNormScaleClamp ...
```

**How**: Compose ShaderM computations, eliminating intermediate writeBuffer/readBuffer.

### Category 2: Reduction + Element-wise Fusion
A reduction (sum, max) followed by element-wise using the result.
Already done in forward: RMSNorm fuses sum(x²) reduction + normalization.

Example: Softmax backward = reduction (dot product) + element-wise
```lean
-- Already fused in a single kernel:
-- Phase 1: dot = Σ attn[i] * dAttn[i]  (reduction)
-- Phase 2: dScores[i] = attn[i] * (dAttn[i] - dot)  (element-wise)
```

### Category 3: Buffer Copy Elimination
When Op B reads from the buffer that Op A just wrote to,
fuse them to use a local variable instead.

Example: RMSNorm backward → BitLinear transpose
```
-- Unfused: RMSNorm backward writes dAttnOutBuf, transpose reads dAttnOutBuf
-- Fused: RMSNorm backward result stays in register, transpose reads from register
```

This is the most impactful fusion for backward.

### Category 4: Multi-Buffer Copy Fusion
Multiple independent copy operations fused into one kernel.

Example: Save 7 activation buffers per layer
```lean
-- Unfused: 7 copy kernels
saveActivation device normedBuf savedNormed dim
saveActivation device attnBuf savedAttn attnSize
...

-- Fused: 1 kernel with 14 buffer bindings (7 src + 7 dst)
fusedSaveActivations device [normedBuf, attnBuf, ...] [savedNormed, savedAttn, ...] [dim, attnSize, ...]
```

## Implementation Plan

### Phase 1: FusedOp primitive (ShaderM level)

```lean
-- A fusable operation: ShaderM computation + metadata
structure FusableOp where
  name : String
  computation : ShaderM Unit
  inputBuffers : Array (String × Nat)   -- (name, size)
  outputBuffers : Array (String × Nat)

-- Fuse two ops: eliminate intermediate buffer
def fuseSequential (a b : FusableOp) (intermediateBuffer : String) : FusableOp
```

The key insight: if `a` writes to buffer X and `b` reads from buffer X,
we can replace buffer X with a workgroup-shared variable or local variable.

### Phase 2: Backward Fusion Groups

Group backward operations that can be fused:

```
Attention backward fusion groups:
  Group 1: O_transpose + RMSNorm_backward → dAttnWeighted
  Group 2: Apply_backward + Softmax_backward → dScores
  Group 3: Score_backward + RoPE_backward → dQpre

FFN backward fusion groups:
  Group 1: Down_transpose + RMSNorm_backward → dHidden
  Group 2: Gate_transpose + Up_transpose + elementwise_add → dNormed2
```

Each group becomes 1 dispatch instead of 2-3.

### Phase 3: Automatic Fusion Analysis

```lean
-- Analyze a list of ops and find fusable pairs
def findFusionOpportunities (ops : Array FusableOp) : Array (Nat × Nat × String)
  -- Returns: (op_i, op_j, intermediate_buffer_name) for each fusion opportunity
```

### Phase 4: Verified Fusion

```lean
-- Prove that fused kernel produces same output as unfused sequence
def verifyFusion (unfused : Array FusableOp) (fused : FusableOp)
    (testInput : Array Float) (tol : Float) : IO Bool
```

## Expected Speedup

| Optimization | Dispatches saved | Estimated speedup |
|-------------|------------------|-------------------|
| Save activation fusion (7→1 per layer) | 180 | 15% |
| Attention backward fusion (3 groups) | 120 | 10% |
| FFN backward fusion (2 groups) | 60 | 5% |
| BitLinear transpose workgroup 32→256 | 0 (faster kernels) | 20% |
| **Total** | **360 dispatches eliminated** | **~40-50%** |

## Comparison with PyTorch

PyTorch `torch.compile`:
- Traces Python → graph IR → fuses element-wise ops → generates Triton/CUDA
- Cannot fuse custom CUDA kernels (BlackBox)
- Compilation overhead at first run

Hesper ShaderM fusion:
- Composes at Lean level → generates single WGSL shader
- All ops are ShaderM, all are fusable
- Verification via numerical gradient check
- No runtime compilation overhead (WGSL cached by pipeline cache)
