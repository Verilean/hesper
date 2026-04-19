import LSpec
import Hesper.Backend.CUDA
import Hesper.CUDA.FFI
import Hesper.Layers.RMSNorm
import Tests.GoldenUnit.Common

/-!
# RMSNorm golden-unit tests

Reference formula (from `llama.cpp/src/llama-graph.cpp::build_norm`):
    y = x / sqrt(mean(x²) + eps) * weight

Gemma 4 uses `eps = 1e-6` and raw weights (no +1 shift).

All allocations are released before the test returns.
-/

namespace Hesper.Tests.GoldenUnit.RMSNorm

open LSpec
open Hesper
open Hesper.Tests.GoldenUnit.Common
open Hesper.Layers

/-- Run hesper's `RMSNorm.forward` on the given f32 input (last-token bytes).
    Releases all GPU buffers before returning. -/
unsafe def runRMSNormSingleToken
    (ctx : CUDAContext) (weightBytes : ByteArray) (inputBytes : ByteArray)
    (dim : Nat) (eps : Float) : IO (Array Float) := do
  let config : RMSNorm.Config := { dim := dim, eps := eps }
  let layer ← RMSNorm.create ctx config weightBytes
  try
    withTempBufFromBytes ctx inputBytes fun inBuf => do
      withTempBuf ctx (dim * 4) fun outBuf => do
        RMSNorm.forward ctx layer inBuf outBuf 1
        let outBytes ← GPUBackend.readBuffer ctx outBuf (dim * 4).toUSize
        pure (byteArrayToF32Array outBytes dim)
  finally
    GPUBackend.freeBuffer ctx layer.scale

/-- Parameterised attn_norm test. -/
unsafe def testAttnNormAtLayer (ctx : CUDAContext) (gguf : Hesper.GGUF.GGUFFile)
    (li : Nat) (inputName : String) (threshold : Float) : IO TestSeq := do
  let dim := gemma4HiddenDim
  let eps := gemma4RmsEps
  let inpFull ← loadFloat32Bin s!"{goldenDir}/{inputName}.bin"
  let inputBytes := lastTokenBytes inpFull dim
  let weightBytes ← extractF32 gguf s!"blk.{li}.attn_norm.weight"
  let expFull ← loadFloat32Bin s!"{goldenDir}/attn_norm-{li}.bin"
  let expected := byteArrayToF32Array (lastTokenBytes expFull dim) dim
  let actual ← runRMSNormSingleToken ctx weightBytes inputBytes dim eps
  let rel := relDiff actual expected
  IO.println s!"[RMSNorm L{li} attn_norm (input={inputName})] rel = {rel}"
  pure (test s!"hesper RMSNorm.forward matches llama.cpp attn_norm-{li} (rel={rel} < {threshold})" (rel < threshold))

unsafe def allTests (ctx : CUDAContext) (gguf : Hesper.GGUF.GGUFFile)
    : IO (List (String × List TestSeq)) := do
  let t0 ← testAttnNormAtLayer ctx gguf 0 "inp_scaled" 1e-5
  let t17 ← testAttnNormAtLayer ctx gguf 17 "l_out-16" 1e-5
  pure [
    ("RMSNorm(attn_norm) L0 last token", [t0]),
    ("RMSNorm(attn_norm) L17 last token", [t17])
  ]

end Hesper.Tests.GoldenUnit.RMSNorm
