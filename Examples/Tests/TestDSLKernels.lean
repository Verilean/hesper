import Hesper.NN.MLP

/-!
# Test DSL Kernel Generation

Verify that WGSL DSL generates correct shader code for backward kernels.
-/

namespace Examples.Tests.TestDSLKernels

open Hesper.NN.MLP

def main : IO Unit := do
  IO.println "╔══════════════════════════════════════════════════════════╗"
  IO.println "║  Testing DSL-Generated WGSL Kernels                     ║"
  IO.println "╚══════════════════════════════════════════════════════════╝"
  IO.println ""

  -- Test 1: Softmax Gradient Kernel
  IO.println "1. Softmax Gradient Kernel (outputSize=10):"
  IO.println "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  let softmaxGrad := genSoftmaxGradKernel 10
  IO.println softmaxGrad
  IO.println ""

  -- Test 2: Layer 2 Backward Kernel
  IO.println "2. Layer 2 Backward Kernel (hidden=128, output=10):"
  IO.println "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  let layer2Back := genLayer2BackwardKernel 128 10
  IO.println (layer2Back.take 800)  -- First 800 chars
  IO.println "..."
  IO.println ""

  -- Test 3: ReLU Backward Kernel
  IO.println "3. ReLU Backward Kernel (size=128):"
  IO.println "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  let reluBack := genReLUBackwardKernel 128
  IO.println reluBack
  IO.println ""

  -- Test 4: Layer 1 Backward Kernel
  IO.println "4. Layer 1 Backward Kernel (input=784, hidden=128):"
  IO.println "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  let layer1Back := genLayer1BackwardKernel 784 128
  IO.println (layer1Back.take 800)  -- First 800 chars
  IO.println "..."
  IO.println ""

  -- Test 5: SGD Kernel
  IO.println "5. SGD Update Kernel (size=128, lr=0.01):"
  IO.println "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  let sgd := genSGDKernel 128 0.01
  IO.println sgd
  IO.println ""

  IO.println "╔══════════════════════════════════════════════════════════╗"
  IO.println "║  Verification Complete                                   ║"
  IO.println "╚══════════════════════════════════════════════════════════╝"
  IO.println ""
  IO.println "✅ All kernels generated successfully"
  IO.println "✅ Check output above to verify WGSL correctness"

end Examples.Tests.TestDSLKernels

def main : IO Unit := Examples.Tests.TestDSLKernels.main
