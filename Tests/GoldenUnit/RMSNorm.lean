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

/-- Parameterised test: run RMSNorm at a given layer using llama.cpp's
    INPUT tensor from that layer and compare against llama.cpp's
    attn_norm output at the same layer.

    Input sources by layer:
    - L0: `inp_scaled` (the prompt-level input to L0)
    - Lₙ (n > 0): `l_out-(n-1)` (previous layer's output, 1-token)
-/
unsafe def testAttnNormAtLayer (ctx : CUDAContext) (li : Nat) (inputName : String)
    (threshold : Float) : IO TestSeq := do
  let dim := gemma4HiddenDim
  let eps := gemma4RmsEps
  let inpFull ← loadFloat32Bin s!"{goldenDir}/{inputName}.bin"
  let inputBytes := lastTokenBytes inpFull dim
  let gguf ← loadGGUF
  let weightBytes ← extractF32 gguf s!"blk.{li}.attn_norm.weight"
  let expFull ← loadFloat32Bin s!"{goldenDir}/attn_norm-{li}.bin"
  let expected := byteArrayToF32Array (lastTokenBytes expFull dim) dim
  let actual ← runRMSNormSingleToken ctx weightBytes inputBytes dim eps
  let rel := relDiff actual expected
  IO.println s!"[RMSNorm L{li} attn_norm (input={inputName})] rel = {rel}"
  pure (test s!"hesper RMSNorm.forward matches llama.cpp attn_norm-{li} (rel={rel} < {threshold})" (rel < threshold))

unsafe def allTests : IO (List (String × List TestSeq)) := do
  let ctx ← CUDAContext.init
  -- L0: attn_norm(inp_scaled).  Input is f32 exact (no prior accumulated
  -- error), so expect f32 numerical floor.
  let t0 ← testAttnNormAtLayer ctx 0 "inp_scaled" 1e-5
  -- L17: attn_norm(l_out-16).  Input is llama.cpp's l_out-16 (f32),
  -- weight is blk.17.attn_norm.weight.  Same kernel as L0 — if rel
  -- diverges from L0, the RMSNorm kernel has some layer-dependent
  -- bug (which it shouldn't; the kernel is layer-agnostic).
  let t17 ← testAttnNormAtLayer ctx 17 "l_out-16" 1e-5
  pure [
    ("RMSNorm(attn_norm) L0 last token", [t0]),
    ("RMSNorm(attn_norm) L17 last token", [t17])
  ]

end Hesper.Tests.GoldenUnit.RMSNorm
