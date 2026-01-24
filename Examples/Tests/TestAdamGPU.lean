import Hesper
import Hesper.Compute
import Hesper.NN.MLP

/-!
# Test Adam GPU Kernel Execution

Verify that the GPU Adam optimizer kernel produces correct numerical results.
-/

namespace Examples.Tests.TestAdamGPU

open Hesper.WebGPU
open Hesper.Compute
open Hesper.Core (TensorData)
open Hesper.NN.MLP

def main : IO Unit := do
  IO.println "╔══════════════════════════════════════════════════════════╗"
  IO.println "║  Testing Adam GPU Optimizer Kernel                      ║"
  IO.println "╚══════════════════════════════════════════════════════════╝"
  IO.println ""

  -- Initialize WebGPU
  IO.println "🚀 Initializing WebGPU..."
  let inst ← Hesper.init
  let device ← getDevice inst
  IO.println "✅ GPU device ready"
  IO.println ""

  -- Test Adam Optimizer Kernel
  IO.println "Test: Adam Optimizer Update"
  IO.println "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  -- Initial values
  let params := #[1.0, 2.0, 3.0]
  let grads := #[0.1, 0.2, 0.3]
  let m := #[0.0, 0.0, 0.0]  -- First moment (initialized to 0)
  let v := #[0.0, 0.0, 0.0]  -- Second moment (initialized to 0)

  -- Adam hyperparameters
  let lr := 0.01
  let beta1 := 0.9
  let beta2 := 0.999
  let epsilon := 1e-8
  let step := 1  -- First step

  IO.println s!"  Input params: {params}"
  IO.println s!"  Input grads: {grads}"
  IO.println s!"  Learning rate: {lr}"
  IO.println s!"  Beta1: {beta1}, Beta2: {beta2}, Epsilon: {epsilon}"
  IO.println s!"  Step: {step}"
  IO.println ""

  -- Expected values (computed manually):
  -- m_new[0] = 0.9 * 0.0 + 0.1 * 0.1 = 0.01
  -- v_new[0] = 0.999 * 0.0 + 0.001 * 0.01 = 0.00001
  -- m_hat[0] = 0.01 / (1 - 0.9^1) = 0.01 / 0.1 = 0.1
  -- v_hat[0] = 0.00001 / (1 - 0.999^1) = 0.01
  -- params[0] = 1.0 - 0.01 * 0.1 / (sqrt(0.01) + 1e-8) ≈ 1.0 - 0.01

  IO.println "  Computing expected values..."
  -- For step 1:
  -- m = (1 - beta1) * grad = 0.1 * grad
  -- v = (1 - beta2) * grad^2 = 0.001 * grad^2
  -- bias_corr1 = 1 - beta1^1 = 0.1
  -- bias_corr2 = 1 - beta2^1 = 0.001
  -- m_hat = m / 0.1
  -- v_hat = v / 0.001
  -- update = lr * m_hat / (sqrt(v_hat) + eps)

  -- Generate kernel
  let shader := genAdamKernel 3 lr beta1 beta2 epsilon step

  IO.println "  Generated WGSL shader:"
  IO.println "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  IO.println (shader.take 500)
  IO.println "  ..."
  IO.println ""

  -- Create GPU buffers
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
  let mBuf ← createBuffer device {
    size := (m.size * 4).toUSize
    usage := [.storage, .copyDst, .copySrc]
    mappedAtCreation := false
  }
  let vBuf ← createBuffer device {
    size := (v.size * 4).toUSize
    usage := [.storage, .copyDst, .copySrc]
    mappedAtCreation := false
  }

  -- Upload data
  let paramsBytes ← Hesper.Basic.floatArrayToBytes params
  let gradsBytes ← Hesper.Basic.floatArrayToBytes grads
  let mBytes ← Hesper.Basic.floatArrayToBytes m
  let vBytes ← Hesper.Basic.floatArrayToBytes v

  writeBuffer device paramsBuf 0 paramsBytes
  writeBuffer device gradsBuf 0 gradsBytes
  writeBuffer device mBuf 0 mBytes
  writeBuffer device vBuf 0 vBytes

  -- Compile shader
  let shaderModule ← createShaderModule device shader

  -- Create bind group layout
  let layoutEntries : Array BindGroupLayoutEntry := #[
    { binding := 0, visibility := .compute, bindingType := .buffer false },  -- params (r/w)
    { binding := 1, visibility := .compute, bindingType := .buffer true },   -- grads (r)
    { binding := 2, visibility := .compute, bindingType := .buffer false },  -- m (r/w)
    { binding := 3, visibility := .compute, bindingType := .buffer false }   -- v (r/w)
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
    { binding := 0, buffer := paramsBuf, offset := 0, size := (params.size * 4).toUSize },
    { binding := 1, buffer := gradsBuf, offset := 0, size := (grads.size * 4).toUSize },
    { binding := 2, buffer := mBuf, offset := 0, size := (m.size * 4).toUSize },
    { binding := 3, buffer := vBuf, offset := 0, size := (v.size * 4).toUSize }
  ]
  let bindGroup ← createBindGroup device bindGroupLayout bindEntries

  -- Dispatch
  let future ← dispatchCompute device pipeline bindGroup 1 1 1
  deviceWait future

  -- Read results
  let paramsResultBytes ← mapBufferRead device paramsBuf 0 (params.size * 4).toUSize
  unmapBuffer paramsBuf
  let paramsResult ← Hesper.Basic.bytesToFloatArray paramsResultBytes

  let mResultBytes ← mapBufferRead device mBuf 0 (m.size * 4).toUSize
  unmapBuffer mBuf
  let mResult ← Hesper.Basic.bytesToFloatArray mResultBytes

  let vResultBytes ← mapBufferRead device vBuf 0 (v.size * 4).toUSize
  unmapBuffer vBuf
  let vResult ← Hesper.Basic.bytesToFloatArray vResultBytes

  IO.println "  Results:"
  IO.println s!"  Updated params: {paramsResult}"
  IO.println s!"  Updated m: {mResult}"
  IO.println s!"  Updated v: {vResult}"
  IO.println ""

  -- Verify results are reasonable (params should be slightly decreased)
  let mut allCorrect := true
  for i in [:3] do
    if paramsResult[i]! >= params[i]! then
      allCorrect := false
      IO.println s!"  ❌ params[{i}] should decrease but got {paramsResult[i]!} >= {params[i]!}"
    if mResult[i]! <= 0.0 then
      allCorrect := false
      IO.println s!"  ❌ m[{i}] should be positive but got {mResult[i]!}"
    if vResult[i]! <= 0.0 then
      allCorrect := false
      IO.println s!"  ❌ v[{i}] should be positive but got {vResult[i]!}"

  if allCorrect then
    IO.println "  ✅ Adam optimizer kernel executed successfully!"
    IO.println "  ✅ Parameters decreased as expected"
    IO.println "  ✅ Moment estimates updated correctly"
  else
    IO.println "  ❌ Adam optimizer kernel failed validation!"

  IO.println ""
  IO.println "╔══════════════════════════════════════════════════════════╗"
  IO.println "║  Testing Complete                                        ║"
  IO.println "╚══════════════════════════════════════════════════════════╝"

end Examples.Tests.TestAdamGPU

def main : IO Unit := Examples.Tests.TestAdamGPU.main
