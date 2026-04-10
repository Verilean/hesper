# Surprise-Gated Residual TTT — Architecture & Design Decisions

## Overview

Test-Time Training (TTT) adds online learning to a frozen LLM during
inference. The "Surprise Gate" ensures updates happen only when the base
model is wrong (cross-entropy loss > threshold τ), preventing
catastrophic forgetting of correct predictions.

## Architecture Variants Evaluated

### Variant A: Logit-Space TTT (vocabSize × dim)

```
final_logits = base_logits + W_ttt @ hidden
```

- **W_ttt shape**: [vocabSize, dim]
- **Memory**: vocabSize × dim × 4B
  - Golden test (16×32): 2 KB ✓
  - BitNet 2B (128256×2560): **1.3 GB** ✗
  - Gemma 4 e4b (262144×2560): **2.7 GB** ✗
- **What it learns**: Direct per-token probability adjustments
- **Limitation**: Only affects next-token prediction; does not change
  the model's internal representation, so chain-of-thought reasoning
  and multi-step recall are not improved
- **Status**: Implemented, golden-validated (1024/1024 steps pass),
  but impractical for real models due to memory

### Variant A': LoRA Logit-Space TTT (rank × dim + vocabSize × rank)

```
final_logits = base_logits + B @ A @ hidden
```

- **Memory**: rank=16 → ~8 MB for BitNet 2B
- **Pros**: Memory-efficient version of Variant A
- **Cons**: Same limitation — only affects logits, not hidden state
- **Status**: Not implemented (superseded by Variant C)

### Variant B: Top-K Logit Correction

- Correct only the top-K logits (K=1000)
- **Memory**: ~10 MB
- **Limitation**: Strictly worse than LoRA — doesn't cover rare tokens
- **Status**: Not implemented (no advantage over A' or C)

### Variant C: Hidden-Space TTT ⭐ (dim × dim)

```
hidden' = hidden + W_ttt @ hidden
final_logits = LM_head(hidden')
```

- **W_ttt shape**: [dim, dim]
- **Memory**: dim × dim × 4B
  - BitNet 2B (2560×2560): **26 MB** ✓
  - Gemma 4 e4b (2560×2560): **26 MB** ✓
- **What it learns**: Residual correction to the model's internal
  representation. The LM head then maps the corrected hidden state
  to logits, so **all vocab tokens are affected** through the
  model's learned projection.
- **Why it's better**:
  1. **50,000× more memory-efficient** than Variant A for 128K vocab
  2. Hidden state changes propagate through the **entire vocab** via
     the LM head (dim→vocab mapping is learned, not trained by TTT)
  3. Conceptually: TTT learns "how to adjust the model's understanding"
     rather than "which token probabilities to change"
  4. Matrix is [2560, 2560] = tiny for GPU — matVec + outerProduct
     complete in ~15 µs
- **Status**: Implemented (current production version)

## Comparison Table

| Aspect | Logit-Space (A) | LoRA Logit (A') | Hidden-Space (C) |
|--------|-----------------|-----------------|------------------|
| W_ttt shape | [vocab, dim] | [rank, dim]+[vocab, rank] | **[dim, dim]** |
| Memory (128K vocab) | 1.3 GB | 8 MB | **26 MB** |
| Affects logits | Direct | Direct | Via LM head |
| Affects hidden state | No | No | **Yes** |
| Multi-step reasoning | No help | No help | **Helps** |
| Chain-of-thought | No help | No help | **Helps** |
| Implementation | Simple matVec | LoRA A@B | Simple matVec |

## Surprise Gate

```
gate_decision = (CrossEntropy(base_logits, target) > τ)
```

- **τ = 2.0** (default for BitNet 2B): opens for surprising tokens
- When OPEN: compute gradient, SGD update W_ttt
- When CLOSED: skip update, save compute
- Observed learning curve on MQAR task:
  - 1st encounter of rare pair: loss ~14 → OPEN (learning)
  - 2nd encounter: loss ~0.5 → CLOSED (already memorized)
  - 3rd encounter: loss ~0.02 → CLOSED (perfect recall)

## SGD Update (Hidden-Space)

```
combined_logits = LM_head(hidden + W_ttt @ hidden)
loss = CrossEntropy(combined_logits, target)
dLogits = softmax(combined_logits) - one_hot(target)
dHidden = LM_head^T @ dLogits                    -- backprop through LM head
dW_ttt = outer(dHidden, hidden)                   -- gradient of W_ttt
W_ttt -= lr * dW_ttt                              -- SGD step
```

Note: For the hidden-space variant, the gradient must flow back through
the LM head to reach the hidden space. This requires one additional
transpose-matVec (`LM_head^T @ dLogits`).

## Memory Budget on RTX 4070 Ti (12 GB VRAM)

| Component | BitNet 2B | With Hidden TTT |
|-----------|-----------|-----------------|
| Model weights | ~3 GB | ~3 GB |
| KV cache (2048 seq) | ~2 GB | ~2 GB |
| TTT W_ttt | — | 26 MB |
| TTT scratch buffers | — | ~2 MB |
| **Total** | ~5 GB | **~5.03 GB** |
| **Headroom** | 7 GB | **6.97 GB** |

Compared to Logit-Space TTT which would add 1.3 GB → only 5.7 GB
headroom, Hidden-Space TTT is essentially free.

## File Structure

```
Hesper/TTT/
  Types.lean       — TTTConfig, TTTBuffers, TTTStepResultGPU
  Kernels.lean     — matVec, vecAdd, outerProduct, sgdUpdate, copyBuffer
  InnerLoop.lean   — tttStepGPU (single-step orchestration)
  Layer.lean       — tttSequenceGPU (sequence loop + readback)
  Init.lean        — Golden data loading
  BitNetTTT.lean   — BitNet integration (generateWithTTT)

Tests/TTT/
  TTTGoldenGPUMain.lean — Golden value validation (1024 steps)

Examples/
  BitNetTTT_MQAR.lean   — MQAR proof-of-concept
  BitNetTTT_Needle.lean — Needle-in-haystack test
```
