import LSpec
import Hesper.Backend.CUDA
import Hesper.CUDA.FFI
import Hesper.Models.Gemma4
import Tests.GoldenUnit.Common

/-!
# RoPE golden-unit tests

Reference from `llama.cpp/src/models/gemma4-iswa.cpp:37, 46-48, 63-65`:
    freq_base_l = model.get_rope_freq_base(cparams, il)
    freq_factors = (!is_swa(il)) ? model.layers[il].rope_freqs : nullptr
    Qcur = ggml_rope_ext(Qcur, inp_pos, freq_factors, n_rot_l,
                         rope_type, n_ctx_orig, freq_base_l, ...)

For Gemma 4 E4B:
    freq_base (full-attn) = 1000000
    freq_base (SWA)       = 10000

This MUST produce different theta values for SWA vs full-attn layers,
so `ropeWithFreqFactorsBatchKernel` must know which one to use.

## Tests
- testRopeQL0: SWA layer, freq_base=10000, no freq_factors
- testRopeQL17: full-attn layer, freq_base=1000000, with freq_factors

Layout:
  Input (Qcur_normed): column-major [headDim, numHeads, seqLen]
    i.e. token t's head h element d at offset t*qDim + h*headDim + d
  Output (Qcur_pos): same layout, but with RoPE rotation applied per
    (head, dimPair) using theta = pos / freq_base^(2*dimPair/headDim) / freq_factor

NeoX split-half: pairs are `(x[d], x[d + headDim/2])` within each head.
-/

namespace Hesper.Tests.GoldenUnit.RoPE

open LSpec
open Hesper
open Hesper.Tests.GoldenUnit.Common

/-- Build a freq_factors buffer of ones (for layers that don't use them).
    Caller must free. -/
def allocOnesBuf (ctx : CUDAContext) (n : Nat) : IO (GPUBackend.Buf CUDAContext) := do
  let mut bytes := ByteArray.empty
  for _ in [0:n] do
    -- Float32 1.0 = 0x3F800000 little-endian = 00 00 80 3f
    bytes := bytes.push 0
    bytes := bytes.push 0
    bytes := bytes.push 0x80
    bytes := bytes.push 0x3F
  uploadBuffer ctx bytes

/-- Run hesper's `ropeWithFreqFactorsBatchKernel` at startPos=0 over
    `seqLen` tokens.  Returns the last-token output. -/
unsafe def runRoPEBatchLastToken
    (ctx : CUDAContext) (inputBytes : ByteArray) (freqFactorsBytes : ByteArray)
    (headDim numHeads seqLen : Nat) (ropeBase : Float) : IO (Array Float) := do
  let qDim := numHeads * headDim
  withTempBufFromBytes ctx inputBytes fun inBuf => do
    withTempBuf ctx (qDim * seqLen * 4) fun outBuf => do
      withTempBufFromBytes ctx freqFactorsBytes fun freqBuf => do
        -- params buffer [u32]: startPos = 0
        let startPosBytes := Hesper.WebGPU.BufferOps.uint32ToBytes 0
        withTempBufFromBytes ctx startPosBytes fun paramsBuf => do
          let shader := Hesper.Models.Gemma4.ropeWithFreqFactorsBatchKernel headDim numHeads seqLen ropeBase
          GPUBackend.execute ctx shader
            [("input", inBuf), ("output", outBuf), ("params", paramsBuf), ("freq_factors", freqBuf)]
            (.dispatch1D (numHeads * headDim / 2 * seqLen))
          let outBytes ← GPUBackend.readBuffer ctx outBuf (qDim * seqLen * 4).toUSize
          pure (byteArrayToF32Array (lastTokenBytes outBytes qDim) qDim)

/-- Test Qcur_pos at layer li.  Feeds llama.cpp's Qcur_normed-<li> (full
    5-token dump) into hesper's batched RoPE, compares the last-token
    slice to llama.cpp's Qcur_pos-<li>. -/
unsafe def testRopeQAtLayer (ctx : CUDAContext) (gguf : Hesper.GGUF.GGUFFile)
    (li : Nat) (numHeads headDim seqLen : Nat) (ropeBase : Float)
    (freqFactorsTensor : Option String) (threshold : Float) : IO TestSeq := do
  let qDim := numHeads * headDim
  let inputBytes ← loadFloat32Bin s!"{goldenDir}/Qcur_normed-{li}.bin"
  if inputBytes.size ≠ qDim * seqLen * 4 then
    throw (IO.userError s!"Qcur_normed-{li}.bin size={inputBytes.size}, expected {qDim * seqLen * 4} (qDim={qDim}, seqLen={seqLen})")
  let expFull ← loadFloat32Bin s!"{goldenDir}/Qcur_pos-{li}.bin"
  let expected := byteArrayToF32Array (lastTokenBytes expFull qDim) qDim
  -- freq_factors: either from GGUF (full-attn) or ones (SWA)
  let freqFactorsBytes ← match freqFactorsTensor with
    | some tname => extractF32 gguf tname
    | none =>
      -- Build ones[headDim/2]
      let dimPairs := headDim / 2
      let mut bytes := ByteArray.empty
      for _ in [0:dimPairs] do
        bytes := bytes.push 0
        bytes := bytes.push 0
        bytes := bytes.push 0x80
        bytes := bytes.push 0x3F
      pure bytes
  let actual ← runRoPEBatchLastToken ctx inputBytes freqFactorsBytes headDim numHeads seqLen ropeBase
  let rel := relDiff actual expected
  IO.println s!"[RoPE L{li} Q batched seqLen={seqLen} ropeBase={ropeBase}] rel = {rel}"
  pure (test s!"hesper ropeWithFreqFactorsBatchKernel L{li} seqLen={seqLen} (rel={rel} < {threshold})" (rel < threshold))

unsafe def allTests (ctx : CUDAContext) (gguf : Hesper.GGUF.GGUFFile)
    : IO (List (String × List TestSeq)) := do
  -- L0: SWA layer, headDim=256, freq_base=10000 (gemma4.rope.freq_base_swa),
  -- no freq_factors (Gemma 4 SWA layers pass NULL per gemma4-iswa.cpp:46-49).
  --
  -- Regression: hesper used to pass cfg.ropeTheta=1000000 here, giving
  -- rel=22.87%.  Fixed in commit introducing cfg.ropeBase(li) which
  -- returns ropeThetaSWA for SWA layers.
  let t0 ← testRopeQAtLayer ctx gguf 0 8 256 5 10000 none 1e-4
  -- L17: full-attn, headDim=512, freq_base=1000000, rope_freqs from GGUF
  let t17 ← testRopeQAtLayer ctx gguf 17 8 512 5 1000000 (some "rope_freqs.weight") 1e-4
  pure [
    ("RoPE Q L0 SWA freq_base=10000", [t0]),
    ("RoPE Q L17 full-attn freq_base=1000000", [t17])
  ]

end Hesper.Tests.GoldenUnit.RoPE
