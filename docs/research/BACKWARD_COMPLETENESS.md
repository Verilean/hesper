# Backward Completeness Plan

## Problem

The backward chain has gaps that cause loss to increase instead of decrease.
PyTorch's autograd guarantees completeness automatically. Hesper needs the
same guarantee through a different mechanism.

## Root Cause Analysis

### Issue 1: savedAttnOutput NaN
**What**: RMSNorm backward receives NaN input from `savedAttnOutput`.
**Why**: `qRotBuf` is a shared buffer reused across layers. The save happens
after `forwardWithCache` returns, but within the same GPU batch. The buffer
content should be valid at that point.
**Debug plan**:
1. Add a GPU kernel that writes a known constant to `savedAttnOut[0]` right
   after forward, then read it back and verify.
2. If the constant survives, the issue is in the forward pass corrupting qRotBuf.
3. If it doesn't, the buffer save has a timing/aliasing issue.
**Test**: `Tests/SavedActivationTest.lean` — forward one token, read savedAttnOut,
check for NaN.

### Issue 2: Residual backward incorrect
**What**: `dHidden` is reused unchanged across all 30 layers.
**Why**: In the transformer, each layer adds to the residual stream:
```
x_out = x_in + attention(norm(x_in)) + ffn(norm(x_in + attention(norm(x_in))))
```
The correct backward through residual connections is:
```
dX_in = dX_out + dAttention_sublayer + dFFN_sublayer
```
Currently, `dHidden` (= dX_out from LM head) is passed to every layer unchanged.
**Fix**: After computing LoRA gradients for layer i, update dHidden:
```
dHidden += dLoRA_contribution  (gradient from LoRA's effect on the residual)
```
Actually, for residual connections, dHidden passes through unchanged (this is correct!).
The issue is that LoRA backward's `dInput` should be accumulated into dHidden.
Currently `applyLoRABackward` writes to `dInputBuf` but this isn't added to dHidden.
**Fix plan**: After each layer's LoRA backward, add dInputBuf to dHidden:
```
dHidden[j] += dInput_from_LoRA_Q[j] + dInput_from_LoRA_V[j]
```

### Issue 3: FFN backward missing
**What**: FFN (gate+up+ReLU²×mul+down) backward is not computed.
**Why**: LoRA only applies to attention Q/V, not FFN. So FFN backward
is not needed for LoRA gradient computation per se. However, FFN backward
is needed for correct `dHidden` propagation through the residual stream.
**Impact**: Without FFN backward, the gradient signal from upper layers
doesn't properly propagate to lower layers. For the last layer (29),
the gradient is correct. For layer 28, the gradient should include
the effect of layer 29's FFN — but it doesn't.
**Fix**: Implement FFN backward chain:
  FFN down backward → sub-norm backward → ReLU² backward →
  gate/up backward → pre-FFN norm backward
This is a significant amount of code but follows the same pattern.
**Alternative**: Skip FFN backward but correctly propagate dHidden through
residual connections. Since LoRA only touches Q/V, the FFN contribution to
dHidden is second-order. The residual connection means dHidden passes through.

## Implementation Plan

### Step 1: Fix savedAttnOutput (1 hour)

1. Write `Tests/SavedActivationTest.lean`:
   - Load model, create KV cache, run 1 token forward
   - Read `savedAttnOut[29]` from GPU
   - Check: no NaN, reasonable range (|x| < 100)
   - Check: not all zeros (would indicate save failure)

2. If test fails, fix the save timing:
   - Move `saveActivation` INSIDE `forwardWithCache` (not after)
   - Or use a copy kernel that runs in the same batch

### Step 2: Fix Residual backward (30 min)

After each layer's LoRA backward, accumulate dInput into dHidden:

```lean
-- Current: LoRA backward writes dInputBuf (not used)
-- Fix: dHidden += dInputBuf (elementwise add)
Hesper.WGSL.Elementwise.executeAdd device dInputBuf dHiddenBuf dHiddenBuf dim
```

Wait — `executeAdd` writes to a 3rd buffer. Need in-place add.
Use `executeAddScaled` with scale=1.0:
```lean
Forward.executeAddScaled device trainState.dInputBuf dHiddenBuf dim 1.0
```

This accumulates the LoRA backward's input gradient into the residual gradient.

### Step 3: FFN backward (2-3 hours)

For each layer in reverse:
1. FFN down backward: `dFFNNormed = W_down^T @ dHidden` (BitLinear transpose)
2. FFN sub-norm backward: `dHidden_ffn = RMSNorm_bwd(ffnHidden, gamma, dFFNNormed)`
3. ReLU² backward: `dGate = 2*relu(gate)*sign(gate) * dHidden_ffn * up`
4. Gate/Up backward: `dNormed = W_gate^T @ dGate + W_up^T @ dUp` (BitLinear transpose)
5. Pre-FFN norm backward: `dResidual1 = RMSNorm_bwd(residual1, gamma, dNormed)`
6. `dHidden += dResidual1`

Each step is a verified backward op that can be tested independently.

## Prevention: Type-Safe Backward Chain

### Design: `TransformerLayer` as a `VerifiedOp` composition

```lean
-- Each layer op has verified forward/backward
structure LayerOp where
  name : String
  forward : Device → Buffer → Buffer → IO Unit
  backward : Device → Buffer → Buffer → IO Unit
  -- Proof or test that backward matches forward's derivative
  verified : Bool

-- A transformer layer is a sequence of ops
structure TransformerLayerOps where
  preNorm : LayerOp        -- RMSNorm
  attention : LayerOp      -- Q,K,V projection + scores + softmax + apply
  subNorm : LayerOp        -- RMSNorm
  oProjection : LayerOp    -- BitLinear O
  residualAdd1 : LayerOp   -- x + attention(norm(x))
  ffnNorm : LayerOp        -- RMSNorm
  ffnGateUp : LayerOp      -- gate + up + ReLU²×mul
  ffnSubNorm : LayerOp     -- RMSNorm
  ffnDown : LayerOp        -- BitLinear down
  residualAdd2 : LayerOp   -- x + ffn(norm(x))

-- The backward of the full layer is the reverse composition
-- Type system ensures every forward op has a corresponding backward
def layerBackward (ops : TransformerLayerOps) : TransformerLayerBackward :=
  { residualAdd2_bwd := ops.residualAdd2.backward
  , ffnDown_bwd := ops.ffnDown.backward
  , ffnSubNorm_bwd := ops.ffnSubNorm.backward
  , ffnGateUp_bwd := ops.ffnGateUp.backward
  , ffnNorm_bwd := ops.ffnNorm.backward
  , residualAdd1_bwd := ops.residualAdd1.backward
  , oProjection_bwd := ops.oProjection.backward
  , subNorm_bwd := ops.subNorm.backward
  , attention_bwd := ops.attention.backward
  , preNorm_bwd := ops.preNorm.backward
  }
```

### Completeness Check

```lean
-- This function REQUIRES all backward ops to be provided
-- If any is missing, it won't compile
def fullLayerBackward (fwd : TransformerLayerOps) (bwd : TransformerLayerBackward)
    (dOutput : Buffer) : IO Buffer := do
  -- Every op in fwd must have a corresponding op in bwd
  -- The type signature enforces this
  ...
```

### Automated Registration

When a new forward op is added to the layer, the type system forces
adding a backward op. This is impossible to forget:

```lean
-- Adding a new op to TransformerLayerOps REQUIRES adding to TransformerLayerBackward
-- Otherwise: compilation error
```

### Numerical Verification at Registration

```lean
-- Each LayerOp must pass gradient check before being accepted
def registerOp (name : String) (fwd bwd : ...) : IO LayerOp := do
  let ok := verifyOp { forward := fwd, backward := bwd, ... }
  if !ok then throw "Gradient check failed for {name}"
  pure { name, forward := fwd, backward := bwd, verified := true }
```

## Testing Strategy

### Per-op tests (verified AD)
- Already have: Softmax, RoPE, RMSNorm, ScaledDot
- Need: BitLinear transpose, ReLU², ElementwiseAdd (residual)
- Each tested with `numericalVJP` against CPU spec

### Chain test
- Forward full model, compute loss
- Backward full chain, update weights
- Check: loss decreases monotonically for 10 steps on 1 example
- If loss doesn't decrease: some backward op is wrong

### Completeness test
- Count number of forward dispatches vs backward dispatches
- They should be equal (each forward op has exactly one backward op)
- Log this as a diagnostic

## Summary

| Fix | Effort | Impact | Blocks |
|-----|--------|--------|--------|
| savedAttnOutput NaN | 1h | RMSNorm backward works | Nothing |
| Residual backward | 30min | Correct multi-layer gradient | Nothing |
| FFN backward | 2-3h | Complete backward chain | savedAttnOutput fix |
| Type-safe chain | 2h | Prevents future gaps | Understanding of all ops |
