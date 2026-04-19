import LSpec
import Hesper.Backend.CUDA
import Hesper.CUDA.FFI
import Hesper.Layers.Linear
import Hesper.Models.Gemma4
import Tests.GoldenUnit.Common

/-!
# LM-head (Q6_K output projection) golden-unit test

Reference: `llama.cpp/src/models/gemma4-iswa.cpp:229-235` (approx.)
  cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1)
  cb(cur, "result_norm", -1);
  cur = build_lora_mm(model.output, cur)
  cb(cur, "result_output", -1);

Only the last token is evaluated (llama.cpp prunes to `inp_out_ids` at
the last layer).  Dumps:
  - result_norm.bin : [hiddenDim=2560]
  - result_output.bin : [vocabSize=262144]  (Q6_K output projection result)

Feeds `result_norm` through hesper's `Linear.forwardDP4A` (auto-detects
Q6_K for `output.weight`) and compares logits against `result_output`.
-/

namespace Hesper.Tests.GoldenUnit.LMHead

open LSpec
open Hesper
open Hesper.Tests.GoldenUnit.Common
open Hesper.Layers

def gemma4VocabSize : Nat := 262144

unsafe def runLMHeadSingleToken
    (ctx : CUDAContext)
    (layer : Linear.LinearLayer (GPUBackend.Buf CUDAContext) (GPUBackend.CachedDispatch CUDAContext))
    (inputBytes : ByteArray) (hiddenDim vocabSize : Nat) (softcap : Float) : IO (Array Float) := do
  Linear.dp4aEnabled.set true
  Linear.dp4aQ6KEnabled.set true
  let _ := hiddenDim
  withTempBufFromBytes ctx inputBytes fun inBuf => do
    withTempBuf ctx (vocabSize * 4) fun outBuf => do
      Linear.forwardDP4A ctx layer inBuf outBuf
      -- Apply final logit softcap (Gemma 4: 30.0) to match llama.cpp's
      -- result_output which includes softcap.
      if softcap > 0.0 then
        GPUBackend.execute ctx
          (Hesper.Models.Gemma4.logitSoftcapKernel vocabSize softcap)
          [("input", outBuf), ("output", outBuf)]
          (.dispatch1D vocabSize)
      let outBytes ← GPUBackend.readBuffer ctx outBuf (vocabSize * 4).toUSize
      pure (byteArrayToF32Array outBytes vocabSize)

unsafe def testLMHead (ctx : CUDAContext) (gguf : Hesper.GGUF.GGUFFile)
    (threshold : Float) : IO TestSeq := do
  let hiddenDim := gemma4HiddenDim
  let vocabSize := gemma4VocabSize
  let inputBytes ← loadFloat32Bin s!"{goldenDir}/result_norm.bin"
  if inputBytes.size ≠ hiddenDim * 4 then
    throw (IO.userError s!"result_norm.bin size={inputBytes.size}, expected {hiddenDim * 4}")
  let expBytes ← loadFloat32Bin s!"{goldenDir}/result_output.bin"
  if expBytes.size ≠ vocabSize * 4 then
    throw (IO.userError s!"result_output.bin size={expBytes.size}, expected {vocabSize * 4}")
  let expected := byteArrayToF32Array expBytes vocabSize
  -- Gemma 4 uses tied embeddings: lm_head weights = token_embd.weight.
  -- Final logit softcap = 30.0 (gemma4.final_logit_softcapping).
  let actual ← withLinearLayer ctx gguf "token_embd.weight" hiddenDim vocabSize fun layer =>
    runLMHeadSingleToken ctx layer inputBytes hiddenDim vocabSize 30.0
  let rel := relDiff actual expected
  -- Also compute top-k agreement: does argmax match?
  let mut argMaxAct : Nat := 0
  let mut maxAct : Float := -1.0e30
  let mut argMaxExp : Nat := 0
  let mut maxExp : Float := -1.0e30
  for i in [0:vocabSize] do
    let a := actual[i]!
    let e := expected[i]!
    if a > maxAct then maxAct := a; argMaxAct := i
    if e > maxExp then maxExp := e; argMaxExp := i
  IO.println s!"[LMHead Q6_K, vocabSize={vocabSize}] rel = {rel}, argmax: actual={argMaxAct} (logit={maxAct}), expected={argMaxExp} (logit={maxExp})"
  pure (test s!"hesper LM-head Q6_K last-token matches llama.cpp result_output (rel={rel} < {threshold}, argmax match: {argMaxAct} vs {argMaxExp})"
    (rel < threshold && argMaxAct == argMaxExp))

unsafe def allTests (ctx : CUDAContext) (gguf : Hesper.GGUF.GGUFFile)
    : IO (List (String × List TestSeq)) := do
  -- Q6_K dp4a noise floor is ~5e-4 per published benchmarks.
  let t ← testLMHead ctx gguf 5e-3
  pure [("LMHead Q6_K result_output (last token)", [t])]

end Hesper.Tests.GoldenUnit.LMHead
