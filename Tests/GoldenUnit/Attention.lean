import LSpec
import Hesper.Backend.CUDA
import Hesper.CUDA.FFI
import Hesper.Models.Gemma4
import Tests.GoldenUnit.Common

/-!
# Attention kernel golden-unit tests

Reference: `llama.cpp/src/models/gemma4-iswa.cpp:58-61`
    Qcur (reshaped) = ggml_reshape_3d(Qcur, n_embd_head, n_head, n_tokens)
    Qcur_normed = build_norm(Qcur_reshaped, attn_q_norm, nullptr, LLM_NORM_RMS)

Per head h, element d:
    rms_h = sqrt(mean(Qcur[h,:]²) + eps)
    Qcur_normed[h,d] = Qcur[h,d] / rms_h * weight[d]

All allocations are released before the test returns.
-/

namespace Hesper.Tests.GoldenUnit.Attention

open LSpec
open Hesper.Tests.GoldenUnit.Common
open Hesper.CUDA

unsafe def runPerHeadRMSNorm
    (ctx : CUDAContext) (weightBytes : ByteArray) (inputBytes : ByteArray)
    (numHeads headDim : Nat) (eps : Float) : IO (Array Float) := do
  let total := numHeads * headDim
  withTempBufFromBytes ctx inputBytes fun inBuf => do
    withTempBuf ctx (total * 4) fun outBuf => do
      withTempBufFromBytes ctx weightBytes fun wBuf => do
        let wgSize := if headDim < 256 then headDim else 256
        let shader := Hesper.Models.Gemma4.perHeadRMSNormKernel numHeads headDim eps
        GPUBackend.execute ctx shader
          [("input", inBuf), ("weight", wBuf), ("output", outBuf)]
          { numWorkgroups := (numHeads, 1, 1)
            workgroupSize := { x := wgSize, y := 1, z := 1 } : Hesper.ExecConfig }
        let outBytes ← GPUBackend.readBuffer ctx outBuf (total * 4).toUSize
        pure (byteArrayToF32Array outBytes total)

unsafe def testQcurNormedAtLayer (ctx : CUDAContext) (gguf : Hesper.GGUF.GGUFFile)
    (li : Nat) (numHeads headDim : Nat) (threshold : Float) : IO TestSeq := do
  let qDim := numHeads * headDim
  let inFull ← loadFloat32Bin s!"{goldenDir}/Qcur-{li}.bin"
  let inputBytes := lastTokenBytes inFull qDim
  let weightBytes ← extractF32 gguf s!"blk.{li}.attn_q_norm.weight"
  let expFull ← loadFloat32Bin s!"{goldenDir}/Qcur_normed-{li}.bin"
  let expected := byteArrayToF32Array (lastTokenBytes expFull qDim) qDim
  let actual ← runPerHeadRMSNorm ctx weightBytes inputBytes numHeads headDim gemma4RmsEps
  let rel := relDiff actual expected
  IO.println s!"[perHeadRMSNorm L{li} Qcur_normed, numHeads={numHeads}, headDim={headDim}] rel = {rel}"
  pure (test s!"hesper perHeadRMSNormKernel(Q) L{li} matches llama.cpp Qcur_normed-{li} (rel={rel} < {threshold})" (rel < threshold))

unsafe def allTests (ctx : CUDAContext) (gguf : Hesper.GGUF.GGUFFile)
    : IO (List (String × List TestSeq)) := do
  let t0 ← testQcurNormedAtLayer ctx gguf 0 8 256 1e-5
  let t17 ← testQcurNormedAtLayer ctx gguf 17 8 512 1e-5
  pure [
    ("perHeadRMSNorm(Q) L0 last token", [t0]),
    ("perHeadRMSNorm(Q) L17 last token", [t17])
  ]

end Hesper.Tests.GoldenUnit.Attention
