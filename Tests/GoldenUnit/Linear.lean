import LSpec
import Hesper.Backend.CUDA
import Hesper.CUDA.FFI
import Hesper.Layers.Linear
import Tests.GoldenUnit.Common

/-!
# Linear (Q4_K dp4a) golden-unit tests

Reference: `llama.cpp/src/models/gemma4-iswa.cpp:55`
    Qcur = build_lora_mm(model.layers[il].wq, cur)

All allocations are released before the test returns.
-/

namespace Hesper.Tests.GoldenUnit.Linear

open LSpec
open Hesper
open Hesper.Tests.GoldenUnit.Common
open Hesper.Layers

unsafe def runLinearDP4ASingleToken
    (ctx : CUDAContext)
    (layer : Linear.LinearLayer (GPUBackend.Buf Hesper.CUDAContext) (GPUBackend.CachedDispatch Hesper.CUDAContext))
    (inputBytes : ByteArray) (inDim outDim : Nat) : IO (Array Float) := do
  let _ := inDim
  Linear.dp4aEnabled.set true
  Linear.dp4aQ6KEnabled.set true
  withTempBufFromBytes ctx inputBytes fun inBuf => do
    withTempBuf ctx (outDim * 4) fun outBuf => do
      Linear.forwardDP4A ctx layer inBuf outBuf
      let outBytes ← GPUBackend.readBuffer ctx outBuf (outDim * 4).toUSize
      pure (byteArrayToF32Array outBytes outDim)

unsafe def testWQAtLayer (ctx : CUDAContext) (gguf : Hesper.GGUF.GGUFFile)
    (li : Nat) (outDim : Nat) (threshold : Float) : IO TestSeq := do
  let inDim := gemma4HiddenDim
  let inFull ← loadFloat32Bin s!"{goldenDir}/attn_norm-{li}.bin"
  let inputBytes := lastTokenBytes inFull inDim
  let expFull ← loadFloat32Bin s!"{goldenDir}/Qcur-{li}.bin"
  let expected := byteArrayToF32Array (lastTokenBytes expFull outDim) outDim
  let actual ← withLinearLayer ctx gguf s!"blk.{li}.attn_q.weight" inDim outDim fun layer =>
    runLinearDP4ASingleToken ctx layer inputBytes inDim outDim
  let rel := relDiff actual expected
  IO.println s!"[Linear L{li} wQ Q4_K dp4a, outDim={outDim}] rel = {rel}"
  pure (test s!"hesper forwardDP4A(wQ) L{li} matches llama.cpp Qcur-{li} (rel={rel} < {threshold})" (rel < threshold))

unsafe def allTests (ctx : CUDAContext) (gguf : Hesper.GGUF.GGUFFile)
    : IO (List (String × List TestSeq)) := do
  -- L0: SWA, headDim=256, qDim = 8*256 = 2048
  let t0 ← testWQAtLayer ctx gguf 0 2048 5e-4
  -- L17: full-attn, headDim=512, qDim = 8*512 = 4096
  let t17 ← testWQAtLayer ctx gguf 17 4096 5e-4
  pure [
    ("Linear Q4_K dp4a wQ L0 last token", [t0]),
    ("Linear Q4_K dp4a wQ L17 last token", [t17])
  ]

end Hesper.Tests.GoldenUnit.Linear
