import LSpec
import Hesper.Backend.CUDA
import Hesper.CUDA.FFI
import Hesper.Layers.RMSNorm
import Tests.GoldenUnit.Common

/-!
# RMSNorm golden-unit tests

Reference formula (from `llama.cpp/src/llama-graph.cpp::build_norm`):
    y = x / sqrt(mean(x²) + eps) * weight

Gemma 4 uses `eps = 1e-6` and raw weights (no +1 shift, unlike Gemma 3 —
see `convert_hf_to_gguf.py::Gemma4Model.norm_shift` returning 0.0).

## Tests

- `testAttnNormL0`: run hesper's `RMSNorm.forward` on llama.cpp's
  `inp_scaled` (last token) with `blk.0.attn_norm.weight` and compare
  to llama.cpp's `attn_norm-0` (last token).  Threshold: rel < 1e-5
  (f32 numerical floor).
-/

namespace Hesper.Tests.GoldenUnit.RMSNorm

open LSpec
open Hesper.Tests.GoldenUnit.Common
open Hesper.CUDA
open Hesper.Layers

/-- Run hesper's `RMSNorm.forward` on the given f32 input (last-token bytes).
    Returns the output as an Array Float of length `dim`. -/
unsafe def runRMSNormSingleToken
    (ctx : CUDAContext) (weightBytes : ByteArray) (inputBytes : ByteArray)
    (dim : Nat) (eps : Float) : IO (Array Float) := do
  let config : RMSNorm.Config := { dim := dim, eps := eps }
  let layer ← RMSNorm.create ctx config weightBytes
  let inBuf ← GPUBackend.allocBuffer ctx inputBytes.size.toUSize
  GPUBackend.writeBuffer ctx inBuf inputBytes
  let outBuf ← GPUBackend.allocBuffer ctx inputBytes.size.toUSize
  RMSNorm.forward ctx layer inBuf outBuf 1
  let outBytes ← GPUBackend.readBuffer ctx outBuf (dim * 4).toUSize
  pure (byteArrayToF32Array outBytes dim)

unsafe def testAttnNormL0 : IO TestSeq := do
  let dim := gemma4HiddenDim
  let eps := gemma4RmsEps
  -- Inputs: last token of inp_scaled, attn_norm weight from GGUF.
  let inpFull ← loadFloat32Bin s!"{goldenDir}/inp_scaled.bin"
  let inputBytes := lastTokenBytes inpFull dim
  let gguf ← loadGGUF
  let weightBytes ← extractF32 gguf "blk.0.attn_norm.weight"
  -- Expected: last token of attn_norm-0.
  let expFull ← loadFloat32Bin s!"{goldenDir}/attn_norm-0.bin"
  let expected := byteArrayToF32Array (lastTokenBytes expFull dim) dim
  -- Run hesper kernel.
  let ctx ← CUDAContext.init
  let actual ← runRMSNormSingleToken ctx weightBytes inputBytes dim eps
  let rel := relDiff actual expected
  IO.println s!"[RMSNorm L0 attn_norm] rel = {rel}"
  pure (test s!"hesper RMSNorm.forward matches llama.cpp attn_norm-0 (rel={rel} < 1e-5)" (rel < 1e-5))

unsafe def allTests : IO (List (String × List TestSeq)) := do
  let t1 ← testAttnNormL0
  pure [
    ("RMSNorm(attn_norm) L0 last token", [t1])
  ]

end Hesper.Tests.GoldenUnit.RMSNorm
