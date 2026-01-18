import Hesper.NN.Activation
import Hesper.NN.Conv

/-!
# Neural Network Operations Demo

Demonstrates GPU kernels for common neural network operations:
- Activation functions (ReLU, GELU, Sigmoid, etc.)
- 2D Convolution
- Pooling operations (Max, Average)
- Depthwise convolution

These are the building blocks for deep learning inference on GPU.

All WGSL shaders are generated using the type-safe ShaderM monad,
ensuring compile-time correctness and enabling fearless refactoring.
See WGSL_GENERATION_GUIDE.md for more details.
-/

namespace Examples.NNDemo

open Hesper.NN.Activation
open Hesper.NN.Conv

/-- Demo 1: Activation Functions -/
def demo1_activations : IO Unit := do
  IO.println "=== Demo 1: Activation Functions ==="
  IO.println ""

  let config : ActivationConfig := { size := 1024, workgroupSize := 256 }
  let numWG := config.numWorkgroups

  IO.println s!"Processing {config.size} elements with {config.workgroupSize} threads/workgroup"
  IO.println s!"Workgroups needed: {numWG}"
  IO.println ""

  -- ReLU
  IO.println "**ReLU** - Rectified Linear Unit: f(x) = max(0, x)"
  IO.println "Generated shader (first 30 lines):"
  let reluShader := generateReLUShader config
  let reluLines := reluShader.splitOn "\n" |>.take 15
  for line in reluLines do
    IO.println s!"  {line}"
  IO.println "  ..."
  IO.println ""

  -- GELU
  IO.println "**GELU** - Gaussian Error Linear Unit (used in transformers)"
  IO.println "Formula: f(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))"
  IO.println ""

  -- Swish/SiLU
  IO.println "**Swish/SiLU** - Smooth activation: f(x) = x * sigmoid(x)"
  IO.println ""

  -- Mish
  IO.println "**Mish** - Modern activation: f(x) = x * tanh(softplus(x))"
  IO.println ""

/-- Demo 2: Softmax (with reduction) -/
def demo2_softmax : IO Unit := do
  IO.println "=== Demo 2: Softmax Activation ==="
  IO.println ""

  let config : ActivationConfig := { size := 256, workgroupSize := 256 }

  IO.println "Softmax: Converts logits to probabilities"
  IO.println "Formula: softmax(x)ᵢ = exp(xᵢ - max(x)) / Σⱼ exp(xⱼ - max(x))"
  IO.println ""
  IO.println "Implementation uses:"
  IO.println "  1. Parallel reduction to find maximum (numerical stability)"
  IO.println "  2. Compute exp(x - max) and sum"
  IO.println "  3. Normalize by sum"
  IO.println ""

  IO.println "Generated shader (abbreviated):"
  let softmaxShader := generateSoftmaxShader config
  let lines := softmaxShader.splitOn "\n" |>.take 20
  for line in lines do
    IO.println s!"  {line}"
  IO.println "  ..."
  IO.println ""

/-- Demo 3: 2D Convolution -/
def demo3_conv2d : IO Unit := do
  IO.println "=== Demo 3: 2D Convolution ==="
  IO.println ""

  let config : Conv2DConfig := {
    batch := 1
    inputHeight := 224
    inputWidth := 224
    inputChannels := 3
    kernelHeight := 3
    kernelWidth := 3
    outputChannels := 64
    stride := 1
    padding := 1
    workgroupSize := 16
  }

  IO.println s!"Input: [1, {config.inputHeight}, {config.inputWidth}, {config.inputChannels}] (RGB image)"
  IO.println s!"Kernel: [{config.kernelHeight}×{config.kernelWidth}, {config.inputChannels} → {config.outputChannels}]"
  IO.println s!"Output: [1, {config.outputHeight}, {config.outputWidth}, {config.outputChannels}]"
  IO.println s!"Parameters: stride={config.stride}, padding={config.padding}"
  IO.println ""

  let (wgX, wgY, wgZ) := config.numWorkgroups
  IO.println s!"Workgroups: {wgX} × {wgY} × {wgZ} = {wgX * wgY * wgZ} total"
  IO.println ""

  IO.println "Generated shader structure:"
  IO.println "  - Nested loops over kernel dimensions"
  IO.println "  - Padding handling"
  IO.println "  - Bias addition"
  IO.println "  - NHWC (batch, height, width, channels) format"
  IO.println ""

/-- Demo 4: Pooling Operations -/
def demo4_pooling : IO Unit := do
  IO.println "=== Demo 4: Pooling Operations ==="
  IO.println ""

  let config : PoolingConfig := {
    batch := 1
    inputHeight := 56
    inputWidth := 56
    channels := 64
    poolHeight := 2
    poolWidth := 2
    stride := 2
    workgroupSize := 16
  }

  IO.println s!"Input: [1, {config.inputHeight}, {config.inputWidth}, {config.channels}]"
  IO.println s!"Pool window: {config.poolHeight}×{config.poolWidth}, stride {config.stride}"
  IO.println s!"Output: [1, {config.outputHeight}, {config.outputWidth}, {config.channels}]"
  IO.println ""

  IO.println "**Max Pooling**: Takes maximum value in each window"
  IO.println "  - Commonly used in CNNs for spatial downsampling"
  IO.println "  - Provides translation invariance"
  IO.println ""

  IO.println "**Average Pooling**: Takes average value in each window"
  IO.println "  - Smoother downsampling"
  IO.println "  - Often used in global pooling layers"
  IO.println ""

/-- Demo 5: Depthwise Convolution (MobileNets) -/
def demo5_depthwise : IO Unit := do
  IO.println "=== Demo 5: Depthwise Convolution ==="
  IO.println ""

  let config : Conv2DConfig := {
    batch := 1
    inputHeight := 112
    inputWidth := 112
    inputChannels := 32
    kernelHeight := 3
    kernelWidth := 3
    outputChannels := 32  -- Same as input for depthwise
    stride := 1
    padding := 1
    workgroupSize := 16
  }

  IO.println "Depthwise Convolution (used in MobileNets, EfficientNets)"
  IO.println ""
  IO.println s!"Input: [1, {config.inputHeight}, {config.inputWidth}, {config.inputChannels}]"
  IO.println s!"Kernel: [{config.kernelHeight}×{config.kernelWidth}] per channel (not mixed)"
  IO.println s!"Output: [1, {config.outputHeight}, {config.outputWidth}, {config.outputChannels}]"
  IO.println ""

  let standardParams := config.kernelHeight * config.kernelWidth *
                        config.inputChannels * config.outputChannels
  let depthwiseParams := config.kernelHeight * config.kernelWidth * config.inputChannels
  let reduction := standardParams / depthwiseParams

  IO.println "Efficiency comparison:"
  IO.println s!"  Standard Conv2D parameters: {standardParams}"
  IO.println s!"  Depthwise Conv2D parameters: {depthwiseParams}"
  IO.println s!"  Parameter reduction: {reduction}×"
  IO.println ""
  IO.println "MobileNet strategy: Depthwise + Pointwise (1×1) convolution"
  IO.println "  = Separable convolution with much fewer parameters"
  IO.println ""

/-- Demo 6: Real CNN Architecture Example -/
def demo6_cnn_layer : IO Unit := do
  IO.println "=== Demo 6: Example CNN Layer Sequence ==="
  IO.println ""

  IO.println "Typical CNN block (e.g., ResNet, VGG):"
  IO.println ""
  IO.println "1. Conv2D (3×3, 64 → 128 channels)"
  IO.println "   Input: [1, 56, 56, 64]"
  IO.println "   Output: [1, 56, 56, 128]"
  IO.println ""
  IO.println "2. ReLU Activation"
  IO.println "   Output: [1, 56, 56, 128]"
  IO.println ""
  IO.println "3. Conv2D (3×3, 128 → 128 channels)"
  IO.println "   Output: [1, 56, 56, 128]"
  IO.println ""
  IO.println "4. ReLU Activation"
  IO.println "   Output: [1, 56, 56, 128]"
  IO.println ""
  IO.println "5. MaxPooling (2×2, stride 2)"
  IO.println "   Output: [1, 28, 28, 128]"
  IO.println ""

  IO.println "All operations are GPU-accelerated with generated WGSL shaders!"
  IO.println ""

/-- Demo 7: Activation Function Comparison -/
def demo7_activation_comparison : IO Unit := do
  IO.println "=== Demo 7: Activation Function Characteristics ==="
  IO.println ""

  IO.println "┌────────────┬─────────────────────────┬──────────────────────────┐"
  IO.println "│ Function   │ Formula                 │ Use Case                 │"
  IO.println "├────────────┼─────────────────────────┼──────────────────────────┤"
  IO.println "│ ReLU       │ max(0, x)               │ CNNs (fast, simple)      │"
  IO.println "│ GELU       │ x·Φ(x)                  │ Transformers (smooth)    │"
  IO.println "│ Swish/SiLU │ x·σ(x)                  │ Modern CNNs              │"
  IO.println "│ Mish       │ x·tanh(softplus(x))     │ State-of-the-art CNNs    │"
  IO.println "│ Sigmoid    │ 1/(1+e⁻ˣ)               │ Binary classification    │"
  IO.println "│ Tanh       │ tanh(x)                 │ RNNs, normalization      │"
  IO.println "│ Softmax    │ eˣⁱ/Σeˣʲ                │ Multi-class output       │"
  IO.println "│ Leaky ReLU │ max(αx, x)              │ Prevent dead neurons     │"
  IO.println "│ ELU        │ x>0?x:α(eˣ-1)           │ Mean activation ~0       │"
  IO.println "└────────────┴─────────────────────────┴──────────────────────────┘"
  IO.println ""

  IO.println "Trends in 2024:"
  IO.println "  • Transformers: GELU, Swish"
  IO.println "  • CNNs: Mish, Swish/SiLU"
  IO.println "  • Classic: ReLU still widely used (simple, fast)"
  IO.println ""

end Examples.NNDemo

-- Run all examples
def main : IO Unit := do
  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║   Hesper Neural Network Operations Demo     ║"
  IO.println "╚══════════════════════════════════════════════╝"
  IO.println ""

  Examples.NNDemo.demo1_activations
  Examples.NNDemo.demo2_softmax
  Examples.NNDemo.demo3_conv2d
  Examples.NNDemo.demo4_pooling
  Examples.NNDemo.demo5_depthwise
  Examples.NNDemo.demo6_cnn_layer
  Examples.NNDemo.demo7_activation_comparison

  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║   All neural network demos complete!         ║"
  IO.println "╚══════════════════════════════════════════════╝"
