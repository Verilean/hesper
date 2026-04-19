import LSpec
import Hesper.Backend.CUDA
import Hesper.CUDA.FFI
import Hesper.Layers.Linear
import Hesper.Layers.RMSNorm
import Tests.GoldenUnit.Common

/-!
# Output projection (wO) + Post-Attention Norm golden-unit tests

Reference: `llama.cpp/src/models/gemma4-iswa.cpp:92-110`
    cur = build_attn(inp_attn, model.layers[il].wo, ...,
                     Qcur, Kcur, Vcur, ..., kq_scale, il)
    // ... flash-attn produces cur (kqv_out / __fattn__)
    // build_attn internally does: cur = build_lora_mm(wo, cur)
    cur = build_norm(cur, model.layers[il].attn_post_norm, nullptr,
                     LLM_NORM_RMS, il)
    cb(cur, "attn_post_norm", il)

So `attn_post_norm-<li>` = `RMSNorm(wO(__fattn__-<li>))` — pre-residual.
Testing wO+PostNorm as a chain since llama.cpp doesn't dump the pre-norm
post-wO intermediate.

## Layout
- Input (__fattn__): col-major `[headDim, numHeads, seqLen]` = `[qDim, seqLen]`
- wO: `[qDim, hiddenDim]`  (inDim=qDim, outDim=hiddenDim)
- Output (attn_post_norm): col-major `[hiddenDim, seqLen]`

## Tests
- L0 (SWA, qDim=8*256=2048, hiddenDim=2560)
- L17 (full-attn, qDim=8*512=4096, hiddenDim=2560)
-/

namespace Hesper.Tests.GoldenUnit.Oproj

open LSpec
open Hesper
open Hesper.Tests.GoldenUnit.Common
open Hesper.Layers

/-- Run the last-token wO + PostAttnNorm chain.

    Feeds the full `seqLen`-token __fattn__ dump through
    `Linear.forwardBatchDP4A`, then `RMSNorm.forward` (batched numRows=seqLen)
    on the post-wO result, and returns the last-token slice of the
    normalised output. -/
unsafe def runOprojPostNormBatch
    (ctx : CUDAContext)
    (wO : Linear.LinearLayer (GPUBackend.Buf CUDAContext) (GPUBackend.CachedDispatch CUDAContext))
    (postNormLayer : RMSNorm.RMSNorm (GPUBackend.Buf CUDAContext) (GPUBackend.CachedDispatch CUDAContext))
    (fattnBytes : ByteArray) (qDim hiddenDim seqLen : Nat) : IO (Array Float) := do
  Linear.dp4aEnabled.set true
  Linear.dp4aQ6KEnabled.set true
  withTempBufFromBytes ctx fattnBytes fun fattnBuf => do
    withTempBuf ctx (hiddenDim * seqLen * 4) fun wOOutBuf => do
      withTempBuf ctx (hiddenDim * seqLen * 4) fun normedBuf => do
        Linear.forwardBatchDP4A ctx wO fattnBuf wOOutBuf seqLen
        RMSNorm.forward ctx postNormLayer wOOutBuf normedBuf seqLen
        let outBytes ← GPUBackend.readBuffer ctx normedBuf (hiddenDim * seqLen * 4).toUSize
        pure (byteArrayToF32Array (lastTokenBytes outBytes hiddenDim) hiddenDim)

/-- Test `wO + PostAttnNorm` at layer `li`, seqLen=5. -/
unsafe def testOprojPostNormAtLayer (ctx : CUDAContext) (gguf : Hesper.GGUF.GGUFFile)
    (li : Nat) (qDim seqLen : Nat) (threshold : Float) : IO TestSeq := do
  let hiddenDim := gemma4HiddenDim
  let eps := gemma4RmsEps
  let fattnBytes ← loadFloat32Bin s!"{goldenDir}/__fattn__-{li}.bin"
  if fattnBytes.size ≠ qDim * seqLen * 4 then
    throw (IO.userError s!"__fattn__-{li}.bin size={fattnBytes.size}, expected {qDim * seqLen * 4}")
  let expFull ← loadFloat32Bin s!"{goldenDir}/attn_post_norm-{li}.bin"
  if expFull.size ≠ hiddenDim * seqLen * 4 then
    throw (IO.userError s!"attn_post_norm-{li}.bin size={expFull.size}, expected {hiddenDim * seqLen * 4}")
  let expected := byteArrayToF32Array (lastTokenBytes expFull hiddenDim) hiddenDim
  -- Load post-attention-norm weight
  let postNormData ← extractF32 gguf s!"blk.{li}.post_attention_norm.weight"
  let postNormLayer ← RMSNorm.create ctx { dim := hiddenDim, eps := eps } postNormData
  try
    let actual ← withLinearLayer ctx gguf s!"blk.{li}.attn_output.weight" qDim hiddenDim fun wO =>
      runOprojPostNormBatch ctx wO postNormLayer fattnBytes qDim hiddenDim seqLen
    let rel := relDiff actual expected
    IO.println s!"[wO+PostNorm L{li} qDim={qDim} hiddenDim={hiddenDim}] rel = {rel}"
    pure (test s!"hesper wO+PostAttnNorm L{li} last-token matches llama.cpp attn_post_norm-{li} (rel={rel} < {threshold})" (rel < threshold))
  finally
    GPUBackend.freeBuffer ctx postNormLayer.scale

unsafe def allTests (ctx : CUDAContext) (gguf : Hesper.GGUF.GGUFFile)
    : IO (List (String × List TestSeq)) := do
  -- L0: SWA, qDim=2048, threshold=1e-3 (accumulates Q4_K wO noise + f32 norm)
  let t0 ← testOprojPostNormAtLayer ctx gguf 0 2048 5 1e-2
  -- L17: full-attn, qDim=4096
  let t17 ← testOprojPostNormAtLayer ctx gguf 17 4096 5 1e-2
  pure [
    ("wO+PostAttnNorm L0 last-token (SWA)", [t0]),
    ("wO+PostAttnNorm L17 last-token (full-attn)", [t17])
  ]

end Hesper.Tests.GoldenUnit.Oproj
