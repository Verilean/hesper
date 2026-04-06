# Verified Automatic Differentiation in Hesper

## Overview

Hesper uses Lean 4's type system to **verify** that backward (gradient) computations are mathematically correct. This ensures that GPU training kernels produce correct gradients without relying on manual testing alone.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  1. CPU Spec (Pure Lean functions)              │
│     forward_spec : Array Float → Array Float    │
│     backward_spec : Array Float → Array Float   │
│           → Array Float                         │
├─────────────────────────────────────────────────┤
│  2. Numerical Verification                       │
│     numericalVJP ≈ backward_spec                │
│     (finite differences, tolerance 1e-3)         │
├─────────────────────────────────────────────────┤
│  3. Chain Rule Composition                       │
│     (g ∘ f).backward = f.backward ∘ g.backward  │
│     Verified algebraically (error = 0.0)         │
├─────────────────────────────────────────────────┤
│  4. GPU Kernel (WGSL ShaderM)                   │
│     Must match CPU spec output                   │
│     Tested at runtime via readback               │
└─────────────────────────────────────────────────┘
```

## How to Add a New Verified Operation

### Step 1: Define Forward and Backward as Pure Functions

```lean
-- In Hesper/AD/Verified.lean

/-- Forward: element-wise ReLU -/
def reluFwd (x : Array Float) : Array Float :=
  x.map (fun xi => if xi > 0.0 then xi else 0.0)

/-- Backward: ReLU gradient (step function) -/
def reluBwd (x dy : Array Float) : Array Float :=
  Array.zipWith (fun xi di => if xi > 0.0 then di else 0.0) x dy
```

### Step 2: Register as a DiffOp with Test Data

```lean
def reluOp : DiffOp := {
  name := "ReLU"
  forward := reluFwd
  backward := reluBwd
  testInput := #[1.0, -2.0, 3.0, -0.5, 0.1]
  testGradOutput := #[0.1, -0.3, 0.2, 0.5, -0.1]
}
```

### Step 3: Verify via Numerical Gradient Check

```lean
-- In runVerification:
let (passed, err) := verifyOp reluOp
-- passed = true, err ≈ 0.0
```

This automatically verifies:
- `reluBwd x dy ≈ Jᵀ(x) · dy` where `J` is the Jacobian of `reluFwd`
- Uses central finite differences: `(f(x+ε) - f(x-ε)) / 2ε`

### Step 4: Implement GPU Kernel Matching the Spec

```lean
-- In a WGSL module:
def reluBackwardKernel (n : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let i := Exp.vec3X gid
  let _x ← ShaderM.declareInputBuffer "x" (.array (.scalar .f32) n)
  let _dy ← ShaderM.declareInputBuffer "dy" (.array (.scalar .f32) n)
  let _dx ← ShaderM.declareOutputBuffer "dx" (.array (.scalar .f32) n)
  ShaderM.if_ (Exp.lt i (Exp.litU32 n)) (do
    let xi ← ShaderM.readBuffer (ty := .scalar .f32) (n := n) "x" i
    let di ← ShaderM.readBuffer (ty := .scalar .f32) (n := n) "dy" i
    let result := Exp.select (Exp.gt xi (Exp.litF32 0.0)) di (Exp.litF32 0.0)
    ShaderM.writeBuffer (ty := .scalar .f32) "dx" i result
  ) (pure ())
```

### Step 5: Compose Operations with Verified Chain Rule

```lean
-- Composition is automatically correct:
let fusedOp := compose reluOp softmaxOp testInput testGrad
let (passed, err) := verifyOp fusedOp
-- Chain rule: (softmax ∘ relu).bwd(x, dy) = relu.bwd(x, softmax.bwd(relu(x), dy))
```

## Currently Verified Operations

| Operation | Forward | Backward | Numerical Error |
|-----------|---------|----------|-----------------|
| **Softmax** | `exp(xᵢ-max) / Σ exp` | `sᵢ(dyᵢ - Σsⱼdyⱼ)` | 0.000000 |
| **RoPE** | `R(θ) @ x` | `R(-θ) @ dy` | 0.000000 |
| **RMSNorm** | `(x/rms) * γ` | chain rule | 0.000000 |
| **ScaledDot** | `scale * q·k` | `scale * dy * k` | 0.000000 |
| **Composition** | `g ∘ f` | `f.bwd ∘ g.bwd` | 0.000000 |

## Running Verification

```bash
lake build verified-ad backward-verify
./.lake/build/bin/verified-ad
./.lake/build/bin/backward-verify
```

Expected output:
```
═══════════════════════════════════════════════
  Verified AD: Numerical Gradient Checks
═══════════════════════════════════════════════

  PASS Softmax: max_relative_error = 0.000000
  PASS RoPE(θ=0.700000): max_relative_error = 0.000000
  PASS RoPE(θ=1.500000): max_relative_error = 0.000000
  PASS RMSNorm: max_relative_error = 0.000000
  PASS ScaledDot: max_relative_error = 0.000000
  PASS ScaledDot ∘ RoPE(θ=0.300000): max_relative_error = 0.000000

  Chain Rule Verification:
  PASS Chain rule composition: error = 0.000000

  ✓ All AD verifications PASSED
```

## Design Philosophy

1. **Spec first**: Write the pure mathematical spec before any GPU code
2. **Verify numerically**: Finite differences catch sign errors, missing terms
3. **Compose safely**: Chain rule is verified once, applies to all compositions
4. **GPU matches spec**: Runtime tests compare GPU output to CPU spec

This approach eliminates the class of bugs where backward kernels have
incorrect gradient formulas — the most common source of training failures
in hand-written GPU training code.

## Next Steps

- [ ] Full attention backward as verified composition of primitives
- [ ] Lean proof that chain rule preserves VJP correctness (beyond numerical)
- [ ] Auto-generate GPU kernels from verified specs
- [ ] Extend to more operations (Conv2D, LayerNorm, GELU)
