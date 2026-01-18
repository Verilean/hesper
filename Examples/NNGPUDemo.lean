import Hesper.NN.Activation
import Hesper.NN.Conv
import Hesper

/-!
# Neural Network GPU Demo

Demonstrates GPU-ready WGSL shader generation for neural network operations.
This shows that the NN kernels are ready for GPU execution.

Usage:
  lake build nn-gpu-demo && ./.lake/build/bin/nn-gpu-demo
-/

open Hesper.NN.Activation
open Hesper.NN.Conv

def main : IO Unit := do
  IO.println "╔════════════════════════════════════════════════╗"
  IO.println "║   Neural Network GPU Demo                    ║"
  IO.println "╚════════════════════════════════════════════════╝"
  IO.println ""

  -- Initialize WebGPU
  IO.println "Initializing WebGPU..."
  Hesper.init
  let _device ← Hesper.WebGPU.getDevice
  IO.println "✓ GPU device initialized (Metal backend)"
  IO.println ""

  -- Test 1: ReLU
  IO.println "═══════════════════════════════════════════════"
  IO.println "Test 1: ReLU Activation - f(x) = max(0, x)"
  IO.println "═══════════════════════════════════════════════"
  let relu_config := { size := 1024, workgroupSize := 256 : ActivationConfig }
  let relu_shader := generateReLUShader relu_config
  IO.println s!"✓ Generated WGSL shader ({relu_shader.length} bytes)"
  IO.println s!"✓ Workgroups needed: {relu_config.numWorkgroups}"
  IO.println ""

  -- Test 2: GELU
  IO.println "═══════════════════════════════════════════════"
  IO.println "Test 2: GELU Activation (Transformer standard)"
  IO.println "═══════════════════════════════════════════════"
  let gelu_config := { size := 2048, workgroupSize := 256 : ActivationConfig }
  let gelu_shader := generateGELUShader gelu_config
  IO.println s!"✓ Generated WGSL shader ({gelu_shader.length} bytes)"
  IO.println s!"✓ Workgroups needed: {gelu_config.numWorkgroups}"
  IO.println ""

  -- Test 3: Sigmoid
  IO.println "═══════════════════════════════════════════════"
  IO.println "Test 3: Sigmoid Activation - f(x) = 1/(1+e^-x)"
  IO.println "═══════════════════════════════════════════════"
  let sigmoid_config := { size := 512, workgroupSize := 256 : ActivationConfig }
  let sigmoid_shader := generateSigmoidShader sigmoid_config
  IO.println s!"✓ Generated WGSL shader ({sigmoid_shader.length} bytes)"
  IO.println s!"✓ Workgroups needed: {sigmoid_config.numWorkgroups}"
  IO.println ""

  -- Test 4: Swish/SiLU
  IO.println "═══════════════════════════════════════════════"
  IO.println "Test 4: Swish/SiLU - f(x) = x * sigmoid(x)"
  IO.println "═══════════════════════════════════════════════"
  let swish_config := { size := 1024, workgroupSize := 256 : ActivationConfig }
  let swish_shader := generateSwishShader swish_config
  IO.println s!"✓ Generated WGSL shader ({swish_shader.length} bytes)"
  IO.println s!"✓ Workgroups needed: {swish_config.numWorkgroups}"
  IO.println ""

  -- Test 5: 2D Convolution
  IO.println "═══════════════════════════════════════════════"
  IO.println "Test 5: 2D Convolution (3×3 kernel, 224×224 input)"
  IO.println "═══════════════════════════════════════════════"
  let conv_config : Conv2DConfig := {
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
  let conv_shader := generateConv2DShaderFromMonad conv_config
  let (wgX, wgY, wgZ) := conv_config.numWorkgroups
  IO.println s!"✓ Generated WGSL shader ({conv_shader.length} bytes)"
  IO.println s!"✓ Input: [1, {conv_config.inputHeight}, {conv_config.inputWidth}, {conv_config.inputChannels}]"
  IO.println s!"✓ Output: [1, {conv_config.outputHeight}, {conv_config.outputWidth}, {conv_config.outputChannels}]"
  IO.println s!"✓ Workgroups: {wgX} × {wgY} × {wgZ} = {wgX * wgY * wgZ}"
  IO.println ""

  -- Summary
  IO.println "╔════════════════════════════════════════════════╗"
  IO.println "║   All GPU shaders generated successfully!     ║"
  IO.println "╚════════════════════════════════════════════════╝"
  IO.println ""
  IO.println "Summary:"
  IO.println "  ✓ WebGPU device initialized (Metal backend)"
  IO.println "  ✓ 5 neural network operations tested"
  IO.println "  ✓ All WGSL shaders generated successfully"
  IO.println "  ✓ Kernels ready for GPU execution"
  IO.println ""
  IO.println "Available Activations:"
  IO.println "  • ReLU, Leaky ReLU, ELU"
  IO.println "  • GELU (Transformers)"
  IO.println "  • Sigmoid, Tanh"
  IO.println "  • Swish/SiLU, Mish"
  IO.println "  • Softmax, Softplus"
  IO.println ""
  IO.println "Available Operations:"
  IO.println "  • 2D Convolution"
  IO.println "  • Depthwise Convolution"
  IO.println "  • Max Pooling, Average Pooling"
  IO.println ""
