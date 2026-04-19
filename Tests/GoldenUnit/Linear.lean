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

/-- Run hesper's `Linear.forwardBatchDP4A` over `seqLen` tokens,
    returning the **last-token** output slice (to compare against
    llama.cpp's last-token Qcur dump). -/
unsafe def runLinearBatchDP4A_lastToken
    (ctx : CUDAContext)
    (layer : Linear.LinearLayer (GPUBackend.Buf Hesper.CUDAContext) (GPUBackend.CachedDispatch Hesper.CUDAContext))
    (inputBytes : ByteArray) (outDim seqLen : Nat) : IO (Array Float) := do
  Linear.dp4aEnabled.set true
  Linear.dp4aQ6KEnabled.set true
  withTempBufFromBytes ctx inputBytes fun inBuf => do
    withTempBuf ctx (outDim * seqLen * 4) fun outBuf => do
      Linear.forwardBatchDP4A ctx layer inBuf outBuf seqLen
      -- Read only the last token's outDim floats.  Column-major
      -- layout: last token starts at (seqLen-1) * outDim floats.
      let outBytes ← GPUBackend.readBuffer ctx outBuf (outDim * seqLen * 4).toUSize
      let lastTokBytes := lastTokenBytes outBytes outDim
      pure (byteArrayToF32Array lastTokBytes outDim)

/-- Batched wQ test at layer `li`, seqLen=5 (full prompt).  Feeds
    llama.cpp's full attn_norm-<li> dump (5 tokens × inDim) through
    hesper's `forwardBatchDP4A`, then compares the last-token output
    to llama.cpp's Qcur-<li> last token. -/
unsafe def testWQBatchAtLayer (ctx : CUDAContext) (gguf : Hesper.GGUF.GGUFFile)
    (li : Nat) (outDim seqLen : Nat) (threshold : Float) : IO TestSeq := do
  let inDim := gemma4HiddenDim
  let inputBytes ← loadFloat32Bin s!"{goldenDir}/attn_norm-{li}.bin"
  -- Verify input has exactly seqLen * inDim floats (== inDim*seqLen*4 bytes)
  if inputBytes.size ≠ inDim * seqLen * 4 then
    throw (IO.userError s!"attn_norm-{li}.bin size={inputBytes.size}, expected {inDim * seqLen * 4} (inDim={inDim}, seqLen={seqLen})")
  let expFull ← loadFloat32Bin s!"{goldenDir}/Qcur-{li}.bin"
  let expected := byteArrayToF32Array (lastTokenBytes expFull outDim) outDim
  let actual ← withLinearLayer ctx gguf s!"blk.{li}.attn_q.weight" inDim outDim fun layer =>
    runLinearBatchDP4A_lastToken ctx layer inputBytes outDim seqLen
  let rel := relDiff actual expected
  IO.println s!"[Linear L{li} wQ BATCHED seqLen={seqLen} outDim={outDim}] rel = {rel}"
  pure (test s!"hesper forwardBatchDP4A(wQ) L{li} seqLen={seqLen} matches llama.cpp Qcur-{li} last tok (rel={rel} < {threshold})" (rel < threshold))

unsafe def allTests (ctx : CUDAContext) (gguf : Hesper.GGUF.GGUFFile)
    : IO (List (String × List TestSeq)) := do
  -- Single-token path
  let t0 ← testWQAtLayer ctx gguf 0 2048 5e-4
  let t17 ← testWQAtLayer ctx gguf 17 4096 5e-4
  -- Batched path: only when llama.cpp's attn_norm dump contains 5 tokens
  -- (which it does for prompt "Hello world how are you" without inp_out_ids
  --  prune at early layers).
  let t0b ← testWQBatchAtLayer ctx gguf 0 2048 5 5e-4
  let t17b ← testWQBatchAtLayer ctx gguf 17 4096 5 5e-4
  pure [
    ("Linear Q4_K dp4a wQ L0 single-token", [t0]),
    ("Linear Q4_K dp4a wQ L17 single-token", [t17]),
    ("Linear Q4_K dp4a wQ L0 BATCHED seqLen=5", [t0b]),
    ("Linear Q4_K dp4a wQ L17 BATCHED seqLen=5", [t17b])
  ]

end Hesper.Tests.GoldenUnit.Linear
