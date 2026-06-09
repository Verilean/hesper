# Hesper Project Status

## Current State (2026-04-05)

No critical issues. All tests pass. Production ready for inference and LoRA finetuning.

## Inference

| Metric | Value |
|--------|-------|
| Model | BitNet b1.58 2B (30 layers, 2560 dim, 128K vocab) |
| Speed | **40.6 TPS** (Flash Attention, RTX 4070 Ti) |
| Pipeline cache | 99.2% hit rate |
| Platform | NixOS + Vulkan (NVIDIA 565.77) |

## LoRA Finetuning

| Metric | Value |
|--------|-------|
| Backward chain | **13/13 ops COMPLETE** (attention 7 + FFN 6) |
| Optimizer | AdamW (PyTorch defaults: lr=2e-4, clip=1.0, warmup=6%) |
| Loss | 4.16 → 3.59 (50 epochs, 10 examples) |
| Output change | Tokyo weather: "sunny, 25°C" (base: "I don't know") |
| Verified AD | 8 ops numerically verified + chain rule composition |
| GPU ↔ CPU | 5 backward kernels match CPU spec (error=0.0) |

## Test Suites

| Suite | Tests | Status |
|-------|-------|--------|
| Verified AD (numerical gradient) | 8 | PASS |
| Backward Verify (CPU specs) | 4 | PASS |
| ParseFloat + floatToWGSL (LSpec) | 33 | PASS |
| RMSNorm GPU kernel | 1 | PASS |
| Wrong Backward Detection | 1 | PASS |
| GPU vs CPU consistency | 5 | PASS |
| Chain Completeness (compile-time) | 13/13 | COMPLETE |
| Flash Attention equivalence | 2 | PASS |

## Architecture

```
Inference:
  Embedding → [30 × TransformerBlock] → FinalNorm → LM Head → Argmax
  Each block: RMSNorm → Attention(+LoRA) → SubNorm → O proj → RMSNorm → FFN → SubNorm → Down

Flash Attention (per layer):
  Q @ K^T → online softmax → weighted V sum  (1 dispatch, shared memory)

Training backward (per output token):
  dLogits → LM head bwd → FinalNorm bwd →
  [30 × reverse]:
    Attention: O bwd → SubNorm bwd → Apply bwd → Softmax bwd → Score bwd → RoPE bwd → LoRA bwd
    FFN: Down bwd → SubNorm bwd → ReLU²×Mul bwd → Gate/Up bwd → Norm bwd
```

## Remaining Tasks

### Priority: Medium

| Task | Description | Effort |
|------|-------------|--------|
| Training speed | 287ms/token. Backward dispatch reduction, FFN backward fusion | 2-3h |
| Flash Attention backward | Fuse score+softmax+apply backward into 1 kernel (like forward) | 2h |
| Exp.var snapshot safety | ShaderM `snapshotVar` primitive to prevent live-reference bugs | 1h |

### Priority: Low

| Task | Description | Effort |
|------|-------------|--------|
| pre-attention RMSNorm backward | Needs layer input saving. Small impact (residual bypass) | 30min |
| `var<storage, read>` generalization | Apply to other read-only buffers for uniformity + perf | 1h |
| Large-scale training test | 1000 Alpaca examples (~17h on current hardware) | 17h run |

### Priority: Future

| Task | Description |
|------|-------------|
| Formal proofs (Mathlib) | Upgrade numerical gradient checks to symbolic proofs |
| GPU tensor AD (autograd) | Replace hand-written backward with automatic differentiation |
| TTT / TurboQuant | Next research directions |

## Key Files

```
Hesper/
  WGSL/FlashAttention.lean    — Flash Attention kernels (v1, v2 tiled, in-place, params)
  WGSL/Fusion.lean            — Kernel fusion framework
  WGSL/Exp.lean               — floatToWGSL (scientific notation for precision)
  LoRA/                        — Types, Init, Forward (fused), Backward, IO, Inference
  Training/                    — Loss, AlpacaDataset, TrainLoop, AttentionBackward,
                                 FFNBackward, BitLinearBackward, VerifiedBackward,
                                 SafeBuffer, ParseFloat, LRScheduler
  AD/Verified.lean             — DiffOp + numerical VJP verification
  AD/Chain.lean                — Type-safe backward chain (DiffChain)
  AD/BackwardOps.lean          — Compile-time completeness guarantee
  Optimizer/AdamGPU.lean       — GPU AdamW optimizer
  Optimizer/GradientClip.lean  — Global L2 norm gradient clipping
Examples/
  Training/AlpacaFinetune.lean — End-to-end finetuning CLI
Tests/
  VerifiedAD.lean              — 8 ops + chain rule verification
  BackwardVerification.lean    — CPU backward spec checks
  ParseFloatSpec.lean          — 33 LSpec tests
  GPUvsCPUBackwardTest.lean    — 5 GPU kernel consistency tests
  FlashAttentionTest.lean      — CPU + GPU equivalence
  ChainCompletenessTest.lean   — 13/13 compile-time check
  SavedActivationTest.lean     — Per-layer activation validity
  RMSNormBackwardGPUTest.lean  — Standalone GPU kernel test
  WrongBackwardTest.lean       — Wrong backward detection
docs/
  VERIFIED_AD.md               — How to add verified operations
  LORA_FINETUNING.md           — Development guide + lessons learned
  BACKWARD_COMPLETENESS.md     — Root cause analysis + type-safe chain design
  KERNEL_FUSION_FRAMEWORK.md   — Fusion categories + expected speedup
  CHANGELOG.md                 — Release history
  STATUS.md                    — This file
```

## Bugs Fixed (Notable)

| Bug | Root Cause | Impact |
|-----|-----------|--------|
| AdamW NaN | `Exp.litF32` truncated 1e-7 to "0.000000" | All training broken |
| lr=0 | `"2e-4".toNat!` returned 0 | No learning |
| RMSNorm backward NaN | `floatArrayToBytes` used Float64 lower bytes | Backward chain broken |
| RMSNorm backward NaN (2) | `sumSq` overwritten by Phase 2 shared memory | Wrong gradients |
| Flash Attention mismatch | `Exp.var` live reference after `ShaderM.assign` | Wrong output |
| WGSL uniformity error | `params` was `read_write` storage (non-uniform) | Flash kernel rejected |
| OOB GPU write | `Exp.select` + unconditional write | NaN corruption |
| WGSL reserved keyword | Buffer named `"target"` | Shader compile fail |
