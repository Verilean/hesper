import LSpec
import Hesper.Backend.CUDA
import Hesper.CUDA.FFI
import Hesper.Layers.Linear
import Tests.GoldenUnit.Common

/-!
# Linear (Q4_K dp4a) golden-unit tests

Reference formula (from `llama.cpp/src/models/gemma4-iswa.cpp:55`):
    Qcur = build_lora_mm(model.layers[il].wq, cur)

which under the hood is `ggml_mul_mat(wq, cur)` — Q4_K matmul of
the Q4_K-quantized weight against the f32 input.  llama.cpp CUDA
dispatches its `mul_mat_vec_q` path which (for Q4_K) quantizes
input to Q8_1 and dot-products with dequantized-on-the-fly Q4_K.

Since both hesper's `forwardDP4A` and llama.cpp CUDA's `mul_mat_vec_q`
are Q4_K × Q8_1 dp4a implementations reading the **same** weight
bytes, the answer agrees up to f32 reduction-order noise.
Threshold: rel < 5e-4.  (A stricter f64 oracle would require
dequantizing Q4_K to f64 and re-doing the matmul in f64; deferred
until we need it.  See doc 21 "Two-layer oracle".)

## Tests

- `testWQL0`: run hesper's `Linear.forwardDP4A` (Q4_K dp4a) on
  `attn_norm-0` (last token) with `blk.0.attn_q.weight` and compare
  to `Qcur-0` (last token).
-/

namespace Hesper.Tests.GoldenUnit.Linear

open LSpec
open Hesper.Tests.GoldenUnit.Common
open Hesper.CUDA
open Hesper.Layers

/-- Run hesper's `Linear.forwardDP4A` (single token) and return the
    output as an Array Float.  Enables dp4a for this call. -/
unsafe def runLinearDP4ASingleToken
    (ctx : CUDAContext) (layer : Linear.LinearLayer (GPUBackend.Buf CUDAContext) (GPUBackend.CachedDispatch CUDAContext))
    (inputBytes : ByteArray) (inDim outDim : Nat) : IO (Array Float) := do
  Linear.dp4aEnabled.set true
  Linear.dp4aQ6KEnabled.set true
  let inBuf ← GPUBackend.allocBuffer ctx (inDim * 4).toUSize
  GPUBackend.writeBuffer ctx inBuf inputBytes
  let outBuf ← GPUBackend.allocBuffer ctx (outDim * 4).toUSize
  Linear.forwardDP4A ctx layer inBuf outBuf
  let outBytes ← GPUBackend.readBuffer ctx outBuf (outDim * 4).toUSize
  pure (byteArrayToF32Array outBytes outDim)

unsafe def testWQL0 : IO TestSeq := do
  let inDim := gemma4HiddenDim        -- 2560
  let numHeads := 8
  let headDim := 256                   -- Gemma 4 SWA headDim at L0 (key_length_swa)
  let outDim := numHeads * headDim     -- qDim = 2048
  -- Inputs: last token of attn_norm-0, wQ from GGUF.
  let inFull ← loadFloat32Bin s!"{goldenDir}/attn_norm-0.bin"
  let inputBytes := lastTokenBytes inFull inDim
  let gguf ← loadGGUF
  let ctx ← CUDAContext.init
  let layer ← loadLinear ctx gguf "blk.0.attn_q.weight" inDim outDim
  -- Expected: last token of Qcur-0.
  let expFull ← loadFloat32Bin s!"{goldenDir}/Qcur-0.bin"
  let expected := byteArrayToF32Array (lastTokenBytes expFull outDim) outDim
  -- Run hesper kernel.
  let actual ← runLinearDP4ASingleToken ctx layer inputBytes inDim outDim
  let rel := relDiff actual expected
  IO.println s!"[Linear L0 wQ Q4_K dp4a] rel = {rel}"
  pure (test s!"hesper Linear.forwardDP4A(wQ) matches llama.cpp Qcur-0 (rel={rel} < 5e-4)" (rel < 5e-4))

unsafe def allTests : IO (List (String × List TestSeq)) := do
  let t1 ← testWQL0
  pure [
    ("Linear Q4_K dp4a wQ L0 last token", [t1])
  ]

end Hesper.Tests.GoldenUnit.Linear
