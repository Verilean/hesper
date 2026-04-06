# LoRA Finetuning for BitNet — Development Guide

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    Training Pipeline                              │
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ Alpaca   │───▶│ Forward  │───▶│  Loss    │───▶│ Backward │  │
│  │ Dataset  │    │ + LoRA   │    │ (CE)     │    │ (AD)     │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                       │                               │          │
│                       ▼                               ▼          │
│              ┌──────────────┐                ┌──────────────┐   │
│              │ GPU Kernels  │                │ Verified AD  │   │
│              │ (ShaderM)    │                │ (DiffOp)     │   │
│              └──────────────┘                └──────────────┘   │
│                                                      │          │
│                                              ┌──────────────┐   │
│                                              │ Numerical    │   │
│                                              │ Gradient     │   │
│                                              │ Check        │   │
│                                              └──────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

## File Structure

```
Hesper/
├── AD/
│   ├── Reverse.lean          # CPU scalar AD (tape-based, existing)
│   └── Verified.lean         # Verified AD framework (DiffOp, numerical VJP)
├── LoRA/
│   ├── Types.lean            # Config, Weight, Adapter, SavedActivations
│   ├── Init.lean             # Kaiming/zero initialization, RNG
│   ├── Forward.lean          # GPU kernels: A@x, B@h, fused add
│   ├── Backward.lean         # GPU kernels: dA, dB, dInput
│   ├── IO.lean               # Binary save/load of LoRA weights
│   └── Inference.lean        # LoRA-aware generate, batched forward+backward
├── Training/
│   ├── Loss.lean             # Cross-entropy forward/backward + GPU accumulation
│   ├── AlpacaDataset.lean    # JSON parser, prompt templating
│   ├── TrainLoop.lean        # Training utilities, gradient management
│   ├── VerifiedBackward.lean # CPU backward specs with numerical checks
│   └── AttentionBackward.lean# GPU attention backward kernels
├── Optimizer/
│   └── AdamGPU.lean          # GPU-accelerated Adam (has NaN issues, use SGD)
├── Layers/
│   ├── Attention.lean        # Modified: optional LoRA injection via loraOpt
│   └── TransformerBlock.lean # Modified: pass-through LoRA to attention
Examples/
├── Training/
│   └── AlpacaFinetune.lean   # End-to-end finetuning CLI
Tests/
├── BackwardVerification.lean # Run backward spec checks
├── VerifiedAD.lean           # Run verified AD checks
└── WrongBackwardTest.lean    # Prove checker detects wrong backwards
docs/
├── VERIFIED_AD.md            # How to add verified operations
└── LORA_FINETUNING.md        # This file
```

## Development Workflow

### 1. Define Spec → 2. Verify → 3. Implement GPU → 4. Test

This workflow ensures correctness at each step.

#### Step 1: CPU Spec (Pure Function)

Define forward and backward as pure Lean functions in `Hesper/AD/Verified.lean`:

```lean
def myOpFwd (x : Array Float) : Array Float := ...
def myOpBwd (x dy : Array Float) : Array Float := ...
```

**Key rule**: These are the **source of truth**. GPU kernels must match them.

#### Step 2: Numerical Verification

Register as `DiffOp` and verify:

```lean
def myOp : DiffOp := {
  name := "MyOp", forward := myOpFwd, backward := myOpBwd,
  testInput := #[...], testGradOutput := #[...]
}
-- In runVerification: verifyOp myOp → should PASS
```

Also test that wrong implementations are detected:

```lean
-- Intentionally wrong backward
def wrongOp : DiffOp := { myOp with backward := fun _ _ => #[0.0, ...] }
-- verifyOp wrongOp → should FAIL
```

#### Step 3: GPU Kernel (ShaderM)

Implement the kernel using `ShaderM` DSL:

```lean
def myOpBackwardKernel (n : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let i := Exp.vec3X gid
  ShaderM.if_ (Exp.lt i (Exp.litU32 n)) (do
    -- ... compute gradient matching CPU spec ...
  ) (pure ())
```

**Critical patterns for GPU kernels**:
- Always use `ShaderM.if_` guard (not `Exp.select` + write) to prevent OOB writes
- Use `ShaderM.var` + `ShaderM.assign` for mutable accumulators in loops
- Use `Exp.var varName` to read a mutable variable (not the initial binding)
- Match buffer names exactly between `declareInputBuffer`/`declareOutputBuffer` and `readBuffer`/`writeBuffer`

#### Step 4: Integration Test

Run GPU kernel and compare output to CPU spec:

```lean
-- Upload test data to GPU
-- Run GPU kernel
-- Download result
-- Compare to CPU spec output
-- Assert match within tolerance
```

## Lessons Learned

### Float64 → Float32 Conversion

Lean's `Float` is Float64. GPU buffers are Float32. Converting via `f.toBits` gives
Float64 bits — you MUST convert to Float32 format manually:

```lean
private def float64ToFloat32Bits (f : Float) : UInt32 := ...
```

The lower 4 bytes of Float64 bits are NOT Float32 bits. This caused astronomically
large initialization values that corrupted training.

### WGSL Reserved Keywords

`target` is a reserved keyword in WGSL. Buffer names must avoid reserved words.
Our cross-entropy kernel originally used `"target"` → renamed to `"target_id"`.

### WebGPU Buffer Aliasing

WebGPU does not allow the same buffer to be bound as both input and output in a
single dispatch. For in-place operations (like gradient clipping), use a single
`declareOutputBuffer` and read/write through it:

```lean
let _data ← ShaderM.declareOutputBuffer "data" (.array (.scalar .f32) n)
let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := n) "data" i
ShaderM.writeBuffer (ty := .scalar .f32) "data" i (clamp val)
```

### Out-of-Bounds GPU Writes

Never use `Exp.select inBounds result (Exp.litF32 0.0)` followed by an unconditional
`writeBuffer`. Out-of-bounds threads will write 0.0 to memory beyond the buffer,
causing NaN corruption. Always use `ShaderM.if_` to guard:

```lean
-- WRONG: writes 0.0 to OOB indices
let result := Exp.select inBounds val (Exp.litF32 0.0)
ShaderM.writeBuffer "buf" i result

-- CORRECT: skips OOB threads entirely
ShaderM.if_ (Exp.lt i (Exp.litU32 n)) (do
  ShaderM.writeBuffer "buf" i val
) (pure ())
```

### Gradient Signal Strength

Without full attention backward, the gradient signal from `dLogits → dHidden` through
the LM head is too weak for effective LoRA training. The attention backward chain
(apply → softmax → scores → RoPE → dQ) amplifies the gradient to the correct
magnitude. Without it, `lr` must be impractically large.

### SavedActivations vs Gradient Checkpointing

The forward pass uses shared buffers (`layerBufs.normedBuf`) that get overwritten
each layer. For backward, you must either:
1. **Save activations** per layer during forward (memory-expensive)
2. **Recompute** activations in backward (compute-expensive but memory-efficient)

Current approach: recompute `h = A @ normedBuf` in backward using the last layer's
normedBuf. This is approximate — only the last layer's gradient is accurate.

### GPU Batching

The biggest performance win is batching GPU dispatches:

```lean
-- SLOW: 20 GPU syncs per token
forwardSingleToken ...    -- 1 sync
crossEntropyForward ...   -- 1 sync
crossEntropyBackward ...  -- 1 sync
...18 more dispatches...  -- 18 syncs

-- FAST: 1 GPU sync per token
beginBatch device
  forwardSingleToken ...       -- recorded
  crossEntropyForward ...      -- recorded
  crossEntropyBackward ...     -- recorded
  ...all dispatches...         -- recorded
endBatch device                -- 1 sync
```

Loss accumulation on GPU avoids per-token `mapBufferRead` (CPU←GPU sync).

## Known Issues

### Adam Optimizer NaN

The GPU Adam kernel (`AdamGPU.lean`) produces NaN when:
- `v_hat` becomes negative due to floating point (impossible mathematically but happens)
- Large gradient × large lr causes overflow

**Workaround**: Use SGD (`param += -lr * grad`) which is stable.

**Fix needed**: Clamp `v_hat` to non-negative before `sqrt`, clip update magnitude.

### Loss Plateau

With only last-layer attention backward, loss decreases very slowly because:
- Only layer 29's LoRA Q gets correct gradients
- Other layers' LoRA weights remain at initialization
- The gradient signal diminishes through many layers

**Fix needed**: Full multi-layer backward with per-layer saved activations.

### Speed

Current: ~4 seconds per example (1 token ≈ 30ms forward + backward).
Bottleneck: `writeBuffer` for token upload is per-token and causes GPU queue flush.

**Fix needed**: Pre-upload all tokens, use token index buffer.

## Running Tests

```bash
# Verified AD (backward correctness)
lake build verified-ad && ./.lake/build/bin/verified-ad

# Backward spec verification
lake build backward-verify && ./.lake/build/bin/backward-verify

# Wrong backward detection
lake build wrong-backward-test && ./.lake/build/bin/wrong-backward-test

# Training (small test)
lake build alpaca-finetune
./.lake/build/bin/alpaca-finetune \
  --model data/gguf/ggml-model-i2_s.gguf \
  --data data/alpaca_test.json \
  --epochs 5 --rank 8 --lr 1e-3

# Inference with LoRA
lake build bitnet-complete
./.lake/build/bin/bitnet-complete \
  data/gguf/ggml-model-i2_s.gguf \
  "Your prompt" 50 --lora lora_weights.bin
```

## Next Steps

### Priority 1: Full Multi-Layer Backward

Currently only the last layer gets correct attention backward gradients.
To fix:
1. Save `normedBuf` per layer during forward (30 × 2560 × 4 = 300KB)
2. In backward, iterate layers in reverse, using saved normedBuf
3. This gives all 30 layers correct gradients

### Priority 2: Differentiable Typeclass Integration

Use `Hesper.Core.Differentiable` to formally link forward/backward:

```lean
instance : Differentiable SoftmaxOp (Array Float) (Array Float) where
  forward _ := softmaxFwd
  backward _ := softmaxBwd
```

Then compose with verified chain rule:

```lean
instance [Differentiable f I M] [Differentiable g M O] :
    Differentiable (g ∘ f) I O where
  forward op x := g.forward op.2 (f.forward op.1 x)
  backward op x dy := f.backward op.1 x (g.backward op.2 (f.forward op.1 x) dy)
```

### Priority 3: GPU ↔ CPU Spec Consistency Test

For each GPU backward kernel, download GPU output and compare to CPU spec:

```lean
def testGPUKernel (gpuResult cpuResult : Array Float) (tol : Float := 1e-4) : Bool :=
  maxRelativeError gpuResult cpuResult < tol
```

### Priority 4: Formal Lean Proofs

Graduate from numerical checks to symbolic proofs:

```lean
theorem softmax_backward_correct (x dy : Vector ℝ n) :
    softmaxBwd x dy = jacobianTranspose (softmaxFwd ·) x dy := by
  ...
```

This requires Mathlib's analysis library but provides absolute correctness.
