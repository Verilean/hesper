import Hesper
import Hesper.Compute
import Hesper.NN.MLP
import Examples.MachineLearning.MNISTData

/-!
# Test GPU Backward Pass Execution

Verify that GPU backward kernels produce correct numerical results.
-/

namespace Examples.Tests.TestGPUBackward

open Hesper.WebGPU
open Hesper.Compute
open Hesper.Core (TensorData)
open Hesper.NN.MLP
open Examples.MachineLearning.MNISTData

def main : IO Unit := do
  IO.println "╔══════════════════════════════════════════════════════════╗"
  IO.println "║  Testing GPU Backward Pass Execution                    ║"
  IO.println "╚══════════════════════════════════════════════════════════╝"
  IO.println ""

  -- Initialize WebGPU
  IO.println "🚀 Initializing WebGPU..."
  let inst ← Hesper.init
  let device ← getDevice inst
  IO.println "✅ GPU device ready"
  IO.println ""

  -- Test 1: Softmax Gradient Kernel
  IO.println "Test 1: Softmax Gradient Kernel"
  IO.println "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  -- Create test data: probs = [0.1, 0.2, 0.7], label = 2
  let probs := #[0.1, 0.2, 0.7]
  let label := #[2]  -- u32 array
  let dLogits := #[0.0, 0.0, 0.0]

  -- Expected: dLogits = [0.1, 0.2, -0.3] (probs[i] - (i==label ? 1 : 0))
  IO.println s!"  Input probs: {probs}"
  IO.println s!"  Label: {label[0]!}"
  IO.println "  Expected dLogits: [0.1, 0.2, -0.3]"

  -- Generate and run kernel
  let shader := genSoftmaxGradKernel 3

  -- Create buffers manually for testing
  let probsBytes ← Hesper.Basic.floatArrayToBytes probs
  -- Create label bytes manually (u32 = 4 bytes, little-endian)
  let labelValue := label[0]!.toUInt32
  let labelBytes := ByteArray.mk #[
    (labelValue &&& 0xFF).toUInt8,
    ((labelValue >>> 8) &&& 0xFF).toUInt8,
    ((labelValue >>> 16) &&& 0xFF).toUInt8,
    ((labelValue >>> 24) &&& 0xFF).toUInt8
  ]
  let dLogitsBytes ← Hesper.Basic.floatArrayToBytes dLogits

  -- Create GPU buffers
  let probsBuf ← createBuffer device {
    size := (probs.size * 4).toUSize
    usage := [.storage, .copyDst]
    mappedAtCreation := false
  }
  let labelBuf ← createBuffer device {
    size := 4  -- Single u32
    usage := [.storage, .copyDst]
    mappedAtCreation := false
  }
  let dLogitsBuf ← createBuffer device {
    size := (dLogits.size * 4).toUSize
    usage := [.storage, .copyDst, .copySrc]
    mappedAtCreation := false
  }

  -- Upload data
  writeBuffer device probsBuf 0 probsBytes
  writeBuffer device labelBuf 0 labelBytes
  writeBuffer device dLogitsBuf 0 dLogitsBytes

  -- Compile shader
  let shaderModule ← createShaderModule device shader

  -- Create bind group layout
  let layoutEntries : Array BindGroupLayoutEntry := #[
    { binding := 0, visibility := .compute, bindingType := .buffer true },
    { binding := 1, visibility := .compute, bindingType := .buffer true },
    { binding := 2, visibility := .compute, bindingType := .buffer false }
  ]
  let bindGroupLayout ← createBindGroupLayout device layoutEntries

  -- Create pipeline
  let pipeline ← createComputePipeline device {
    shaderModule := shaderModule
    entryPoint := "main"
    bindGroupLayout := bindGroupLayout
  }

  -- Create bind group
  let bindEntries : Array BindGroupEntry := #[
    { binding := 0, buffer := probsBuf, offset := 0, size := (probs.size * 4).toUSize },
    { binding := 1, buffer := labelBuf, offset := 0, size := 4 },
    { binding := 2, buffer := dLogitsBuf, offset := 0, size := (dLogits.size * 4).toUSize }
  ]
  let bindGroup ← createBindGroup device bindGroupLayout bindEntries

  -- Dispatch
  let future ← dispatchCompute device pipeline bindGroup 1 1 1
  deviceWait future

  -- Read results
  let resultBytes ← mapBufferRead device dLogitsBuf 0 (dLogits.size * 4).toUSize
  unmapBuffer dLogitsBuf
  let result ← Hesper.Basic.bytesToFloatArray resultBytes

  IO.println s!"  Actual dLogits: {result}"

  -- Verify
  let expected := #[0.1, 0.2, -0.3]
  let mut allCorrect := true
  for i in [:3] do
    let diff := (result[i]! - expected[i]!).abs
    if diff > 0.01 then
      allCorrect := false
      IO.println s!"  ❌ Mismatch at index {i}: got {result[i]!}, expected {expected[i]!}"

  if allCorrect then
    IO.println "  ✅ Softmax gradient correct!"
  else
    IO.println "  ❌ Softmax gradient failed!"

  IO.println ""

  -- Test 2: SGD Update Kernel
  IO.println "Test 2: SGD Update Kernel"
  IO.println "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  let params := #[1.0, 2.0, 3.0]
  let grads := #[0.1, 0.2, 0.3]
  let lr := 0.1

  -- Expected: params -= lr * grads = [0.99, 1.98, 2.97]
  IO.println s!"  Input params: {params}"
  IO.println s!"  Input grads: {grads}"
  IO.println s!"  Learning rate: {lr}"
  IO.println "  Expected output: [0.99, 1.98, 2.97]"

  let sgdShader := genSGDKernel 3 lr

  let paramsBuf ← createBuffer device {
    size := (params.size * 4).toUSize
    usage := [.storage, .copyDst, .copySrc]
    mappedAtCreation := false
  }
  let gradsBuf ← createBuffer device {
    size := (grads.size * 4).toUSize
    usage := [.storage, .copyDst]
    mappedAtCreation := false
  }

  let paramsBytes ← Hesper.Basic.floatArrayToBytes params
  let gradsBytes ← Hesper.Basic.floatArrayToBytes grads
  writeBuffer device paramsBuf 0 paramsBytes
  writeBuffer device gradsBuf 0 gradsBytes

  let sgdModule ← createShaderModule device sgdShader
  let sgdLayoutEntries : Array BindGroupLayoutEntry := #[
    { binding := 0, visibility := .compute, bindingType := .buffer false },
    { binding := 1, visibility := .compute, bindingType := .buffer true }
  ]
  let sgdLayout ← createBindGroupLayout device sgdLayoutEntries
  let sgdPipeline ← createComputePipeline device {
    shaderModule := sgdModule
    entryPoint := "main"
    bindGroupLayout := sgdLayout
  }

  let sgdBindEntries : Array BindGroupEntry := #[
    { binding := 0, buffer := paramsBuf, offset := 0, size := (params.size * 4).toUSize },
    { binding := 1, buffer := gradsBuf, offset := 0, size := (grads.size * 4).toUSize }
  ]
  let sgdBindGroup ← createBindGroup device sgdLayout sgdBindEntries

  let future ← dispatchCompute device sgdPipeline sgdBindGroup 1 1 1
  deviceWait future

  let sgdResultBytes ← mapBufferRead device paramsBuf 0 (params.size * 4).toUSize
  unmapBuffer paramsBuf
  let sgdResult ← Hesper.Basic.bytesToFloatArray sgdResultBytes

  IO.println s!"  Actual output: {sgdResult}"

  let sgdExpected := #[0.99, 1.98, 2.97]
  let mut sgdCorrect := true
  for i in [:3] do
    let diff := (sgdResult[i]! - sgdExpected[i]!).abs
    if diff > 0.01 then
      sgdCorrect := false
      IO.println s!"  ❌ Mismatch at index {i}: got {sgdResult[i]!}, expected {sgdExpected[i]!}"

  if sgdCorrect then
    IO.println "  ✅ SGD update correct!"
  else
    IO.println "  ❌ SGD update failed!"

  IO.println ""
  IO.println "╔══════════════════════════════════════════════════════════╗"
  IO.println "║  Testing Complete                                        ║"
  IO.println "╚══════════════════════════════════════════════════════════╝"

end Examples.Tests.TestGPUBackward

def main : IO Unit := Examples.Tests.TestGPUBackward.main
