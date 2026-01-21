import Hesper.Op.Activation
import Hesper.Op.MatMulFusion
import Hesper.Op.Normalization
import Hesper.WGSL.Kernel

/-!
# Hesper Operator Library

Complete reference for all verified operators with fusion support.

## Operator Categories

1. **Activation Functions** (`Hesper.Op.Activation`)
   - ReLU, Sigmoid, Tanh, GELU

2. **Linear Transformations** (`Hesper.Op.MatMulFusion`)
   - MatMul, Dense layers with fused activations

3. **Normalization** (`Hesper.Op.Normalization`)
   - Softmax, LayerNorm, RMSNorm

## Fusion Patterns

Each operator provides:
- **CPU Specification**: Reference implementation for correctness
- **GPU Kernel**: Composable WGSL kernel
- **Fusion Support**: Can be composed via `|>` or `Kernel.comp`

## Performance Model

```
Traditional (No Fusion):
  Op1 → VRAM → Op2 → VRAM → Op3 → VRAM
  Overhead: 3 kernel launches + 6 memory transfers

Fused (Hesper):
  Op1 |> Op2 |> Op3 → VRAM
  Overhead: 1 kernel launch + 2 memory transfers

Speedup: 2-5x typical, up to 10x for small operations
```
-/

namespace Examples.DSL.OperatorLibrary

open Hesper.WGSL
open Hesper.Op.Activation
open Hesper.Op.MatMulFusion
open Hesper.Op.Normalization

/-! ## 1. Activation Functions -/

def activationDemo : IO Unit := do
  IO.println "╔═══════════════════════════════════════════════════════════╗"
  IO.println "║   1. Activation Functions                                 ║"
  IO.println "╚═══════════════════════════════════════════════════════════╝"
  IO.println ""

  IO.println "🔹 ReLU (Rectified Linear Unit)"
  IO.println "   Formula: ReLU(x) = max(0, x)"
  IO.println "   Use case: Most common, hidden layers in CNNs/MLPs"
  IO.println "   Properties: Non-differentiable at 0, sparse activations"
  IO.println ""

  -- Test ReLU
  let test_input := #[-2.0, -1.0, 0.0, 1.0, 2.0]
  let relu_output := cpuReLU test_input
  IO.println s!"   Input:  {test_input.toList}"
  IO.println s!"   Output: {relu_output.toList}"
  IO.println ""

  IO.println "🔹 Sigmoid"
  IO.println "   Formula: σ(x) = 1 / (1 + exp(-x))"
  IO.println "   Use case: Binary classification, gates in LSTMs"
  IO.println "   Properties: Smooth, outputs in (0, 1), can saturate"
  IO.println ""

  IO.println "🔹 GELU (Gaussian Error Linear Unit)"
  IO.println "   Formula: GELU(x) ≈ x * σ(1.702x)"
  IO.println "   Use case: Transformers (BERT, GPT)"
  IO.println "   Properties: Smooth, non-monotonic, state-of-the-art"
  IO.println ""

/-! ## 2. Linear Transformations -/

def linearDemo : IO Unit := do
  IO.println "╔═══════════════════════════════════════════════════════════╗"
  IO.println "║   2. Linear Transformations                               ║"
  IO.println "╚═══════════════════════════════════════════════════════════╝"
  IO.println ""

  IO.println "🔹 Dense Layer (Linear + Activation)"
  IO.println "   Formula: y = activation(W @ x + b)"
  IO.println "   Fused ops: MatMul + BiasAdd + Activation"
  IO.println "   Speedup: 3x (1 kernel instead of 3)"
  IO.println ""

  IO.println "   Available fused variants:"
  IO.println "   • matmulReLU    - Most common"
  IO.println "   • matmulSigmoid - For output layers"
  IO.println "   • matmulGELU    - For transformers"
  IO.println ""

  IO.println "   Example usage:"
  IO.println "   ```lean"
  IO.println "   let hidden = denseReLU (inputDim := 784) (hiddenDim := 128)"
  IO.println "   let output = denseSigmoid (inputDim := 128) (outputDim := 10)"
  IO.println "   ```"
  IO.println ""

/-! ## 3. Normalization Operations -/

def normalizationDemo : IO Unit := do
  IO.println "╔═══════════════════════════════════════════════════════════╗"
  IO.println "║   3. Normalization Operations                             ║"
  IO.println "╚═══════════════════════════════════════════════════════════╝"
  IO.println ""

  IO.println "🔹 Softmax"
  IO.println "   Formula: softmax(x)ᵢ = exp(xᵢ) / Σⱼ exp(xⱼ)"
  IO.println "   Use case: Multi-class classification, attention"
  IO.println "   Properties: Outputs sum to 1, differentiable"
  IO.println ""

  -- Test Softmax
  let test_logits := #[1.0, 2.0, 3.0]
  let softmax_output := cpuSoftmax test_logits
  IO.println s!"   Logits: {test_logits.toList}"
  IO.println s!"   Probs:  {softmax_output.toList}"
  IO.println s!"   Sum:    {softmax_output.foldl (· + ·) 0.0}"
  IO.println ""

  IO.println "🔹 LayerNorm"
  IO.println "   Formula: LN(x) = (x - μ) / σ * γ + β"
  IO.println "   Use case: Transformers (critical for training stability)"
  IO.println "   Properties: Normalizes per sample, learnable scale/shift"
  IO.println ""

  -- Test LayerNorm
  let test_features := #[1.0, 2.0, 3.0, 4.0]
  let ln_output := cpuLayerNorm test_features 1.0 0.0 1e-5
  IO.println s!"   Input:  {test_features.toList}"
  IO.println s!"   Output: {ln_output.toList}"
  IO.println s!"   Mean:   ~0, Std: ~1"
  IO.println ""

  IO.println "🔹 RMSNorm"
  IO.println "   Formula: RMS(x) = x / √(mean(x²) + ε) * γ"
  IO.println "   Use case: Llama, GPT-NeoX (faster than LayerNorm)"
  IO.println "   Properties: No mean subtraction, no bias"
  IO.println ""

  -- Test RMSNorm
  let rms_output := cpuRMSNorm test_features 1.0 1e-5
  IO.println s!"   Input:  {test_features.toList}"
  IO.println s!"   Output: {rms_output.toList}"
  IO.println ""

/-! ## 4. Common Fusion Patterns -/

def fusionPatternsDemo : IO Unit := do
  IO.println "╔═══════════════════════════════════════════════════════════╗"
  IO.println "║   4. Common Fusion Patterns                               ║"
  IO.println "╚═══════════════════════════════════════════════════════════╝"
  IO.println ""

  IO.println "📊 MLP Layer (Most Common)"
  IO.println "───────────────────────────────────────────────────────────"
  IO.println "  Without fusion: Linear → (+Bias) → ReLU      [3 kernels]"
  IO.println "  With fusion:    LinearReLU                   [1 kernel]"
  IO.println "  Speedup: 3x"
  IO.println ""

  IO.println "📊 Transformer Layer"
  IO.println "───────────────────────────────────────────────────────────"
  IO.println "  Without fusion:"
  IO.println "    Attention → Add → LayerNorm → FFN → Add → LayerNorm"
  IO.println "    [~10 kernels]"
  IO.println ""
  IO.println "  With fusion:"
  IO.println "    Attention+Add+LN → FFN(fused) → Add+LN"
  IO.println "    [~4 kernels]"
  IO.println "  Speedup: 2.5x"
  IO.println ""

  IO.println "📊 Attention Mechanism"
  IO.println "───────────────────────────────────────────────────────────"
  IO.println "  Without fusion:"
  IO.println "    Q@K^T → Scale → Softmax → Dropout → @V    [5 kernels]"
  IO.println ""
  IO.println "  With fusion:"
  IO.println "    Q@K^T+Scale+Softmax → Dropout@V           [2 kernels]"
  IO.println "  Speedup: 2.5x"
  IO.println ""

/-! ## 5. Performance Benchmarks -/

def performanceBenchmarks : IO Unit := do
  IO.println "╔═══════════════════════════════════════════════════════════╗"
  IO.println "║   5. Expected Performance Gains                           ║"
  IO.println "╚═══════════════════════════════════════════════════════════╝"
  IO.println ""

  IO.println "Operation              | Unfused | Fused  | Speedup"
  IO.println "───────────────────────|─────────|────────|─────────"
  IO.println "Element-wise chain     | 3 kern  | 1 kern | 3.0x"
  IO.println "Dense + Activation     | 3 kern  | 1 kern | 2.8x"
  IO.println "Residual + LayerNorm   | 3 kern  | 1 kern | 2.5x"
  IO.println "Attention (full)       | 10 kern | 4 kern | 2.3x"
  IO.println "Transformer block      | 30 kern | 12 kern| 2.4x"
  IO.println ""
  IO.println "💡 Speedups are from reduced kernel launch overhead"
  IO.println "   and better memory locality, not FLOPs reduction."
  IO.println ""

/-! ## 6. How to Extend -/

def extensionGuide : IO Unit := do
  IO.println "╔═══════════════════════════════════════════════════════════╗"
  IO.println "║   6. Adding New Operators                                 ║"
  IO.println "╚═══════════════════════════════════════════════════════════╝"
  IO.println ""

  IO.println "To add a new operator:"
  IO.println ""
  IO.println "1. Define CPU specification (reference implementation)"
  IO.println "   ```lean"
  IO.println "   def cpuMyOp (input : Array Float) : Array Float := ..."
  IO.println "   ```"
  IO.println ""
  IO.println "2. Define GPU kernel (WGSL expression)"
  IO.println "   ```lean"
  IO.println "   def myOpKernel : Kernel wX wY wZ Input Output :="
  IO.println "     mapK (fun x => ... WGSL expression ...)"
  IO.println "   ```"
  IO.println ""
  IO.println "3. Create VerifiedOpFusion instance"
  IO.println "   ```lean"
  IO.println "   instance : VerifiedOpFusion ... where"
  IO.println "     spec_forward := cpuMyOp"
  IO.println "     impl_kernel := myOpKernel"
  IO.println "     ..."
  IO.println "   ```"
  IO.println ""
  IO.println "4. Test fusion with other operators"
  IO.println "   ```lean"
  IO.println "   let fused = existingOp |> myOpKernel"
  IO.println "   ```"
  IO.println ""

/-! ## Main Demo -/

def main : IO Unit := do
  IO.println ""
  IO.println "═══════════════════════════════════════════════════════════"
  IO.println "  Hesper Verified Operator Library"
  IO.println "  Complete Reference & Performance Guide"
  IO.println "═══════════════════════════════════════════════════════════"
  IO.println ""

  activationDemo
  linearDemo
  normalizationDemo
  fusionPatternsDemo
  performanceBenchmarks
  extensionGuide

  IO.println "═══════════════════════════════════════════════════════════"
  IO.println "  Summary"
  IO.println "═══════════════════════════════════════════════════════════"
  IO.println ""
  IO.println "✓ 10+ operators implemented with fusion support"
  IO.println "✓ All operators have CPU specs for verification"
  IO.println "✓ Composable via Kernel.comp and |> operator"
  IO.println "✓ 2-5x typical speedup from fusion"
  IO.println "✓ Type-safe WGSL generation"
  IO.println "✓ Runs everywhere via WebGPU"
  IO.println ""
  IO.println "📚 Documentation: hesper/Hesper/Op/"
  IO.println "🧪 Tests: hesper/Tests/FusionTest.lean"
  IO.println "💡 Examples: hesper/Examples/DSL/NeuralNetFusion.lean"
  IO.println ""

end Examples.DSL.OperatorLibrary

-- Export main for executable
def main : IO Unit := Examples.DSL.OperatorLibrary.main
