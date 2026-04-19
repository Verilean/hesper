import LSpec
import Hesper.Backend.CUDA
import Hesper.CUDA.FFI
import Hesper.Models.Gemma4
import Hesper.WGSL.FlashAttention
import Tests.GoldenUnit.Common

/-!
# FlashAttention (batched) golden-unit tests

Reference: `llama.cpp/src/models/gemma4-iswa.cpp:83-87`
    cur = build_attn(inp_attn, model.layers[il].wo, nullptr,
                     Qcur, Kcur, Vcur, nullptr, nullptr, kq_scale, il)
where `kq_scale = 1.0` for Gemma 4 (no pre-attn scaling — `f_attention_scale`
defaults to 1.0 per llama-model.cpp:1273).

Layout of `__fattn__-<li>` dump (ggml output of `ggml_flash_attn_ext`):
    ne = { v->ne[0], q->ne[2], q->ne[1], q->ne[3] }
       = { headDim, numHeads, seqLen, 1 }
    col-major offset (t, h, d) = t*numHeads*headDim + h*headDim + d

This matches hesper's `flashAttentionBatchKernel` output layout exactly:
    `output[col * (numHeads * headDim) + h * headDim + d]`

## Tests
- L0 (SWA, numHeads=8, numKVHeads=2, headDim=256)
- L17 (full-attn, numHeads=8, numKVHeads=2, headDim=512)

Per-token attention mask is causal (cacheLen = startPos + col + 1) — the
kernel attends each query to its own prefix of the cache.

## Input pipeline
1. Run `fusedRopeKAndCacheWriteBatchKernel` with llama.cpp's
   `Kcur_normed-<li>` and `Vcur_normed-<li>` dumps → populates the cache
   (startPos=0, seqLen=5). This kernel is already unit-tested at rel=0.
2. Feed llama.cpp's `Qcur_pos-<li>` (post-RoPE Q, 5 tokens) as `q` input.
3. Dispatch `flashAttentionBatchKernel` with grid (numHeads, seqLen, 1).
4. Compare last-token output to last-token slice of `__fattn__-<li>`.
-/

namespace Hesper.Tests.GoldenUnit.FlashAttention

open LSpec
open Hesper
open Hesper.Tests.GoldenUnit.Common

/-- Populate a KV cache by running the batched RoPE-K + cache-write kernel.
    Caller owns `kCacheBuf` and `vCacheBuf`; we return after the dispatch
    completes so the caches are ready for flash attention.

    `newKBytes` / `newVBytes` : llama.cpp's Kcur_normed / Vcur_normed dumps
                                (col-major [headDim, numKVHeads, seqLen]).
    `freqFactorsBytes`        : freq_factors tensor bytes (ones for SWA). -/
unsafe def populateKVCache
    (ctx : CUDAContext)
    (kCacheBuf vCacheBuf : GPUBackend.Buf CUDAContext)
    (newKBytes newVBytes freqFactorsBytes : ByteArray)
    (numKVHeads maxSeqLen headDim seqLen : Nat) (ropeBase : Float) : IO Unit := do
  withTempBufFromBytes ctx newKBytes fun kInBuf => do
    withTempBufFromBytes ctx newVBytes fun vInBuf => do
      withTempBufFromBytes ctx freqFactorsBytes fun freqBuf => do
        let startPosBytes := Hesper.WebGPU.BufferOps.uint32ToBytes 0
        withTempBufFromBytes ctx startPosBytes fun paramsBuf => do
          let shader := Hesper.Models.Gemma4.fusedRopeKAndCacheWriteBatchKernel
            numKVHeads maxSeqLen headDim seqLen ropeBase
          GPUBackend.execute ctx shader
            [("new_k", kInBuf), ("new_v", vInBuf),
             ("k_cache", kCacheBuf), ("v_cache", vCacheBuf),
             ("params", paramsBuf), ("freq_factors", freqBuf)]
            (.dispatch1D (numKVHeads * headDim / 2 * seqLen))

/-- Run the batched flash-attention kernel over `seqLen` query tokens and
    return the **last-token** slice (numHeads * headDim floats). -/
unsafe def runFlashAttnBatchLastToken
    (ctx : CUDAContext)
    (qBytes : ByteArray)
    (newKBytes newVBytes freqFactorsBytes : ByteArray)
    (numHeads numKVHeads maxSeqLen headDim seqLen : Nat)
    (ropeBase scale : Float) : IO (Array Float) := do
  let qDim := numHeads * headDim
  let cacheBytes := numKVHeads * maxSeqLen * headDim * 4
  let zeros : ByteArray := (List.replicate cacheBytes (0 : UInt8)).toByteArray
  withTempBufFromBytes ctx zeros fun kCacheBuf => do
    withTempBufFromBytes ctx zeros fun vCacheBuf => do
      populateKVCache ctx kCacheBuf vCacheBuf newKBytes newVBytes freqFactorsBytes
        numKVHeads maxSeqLen headDim seqLen ropeBase
      withTempBufFromBytes ctx qBytes fun qBuf => do
        withTempBuf ctx (qDim * seqLen * 4) fun outBuf => do
          let startPosBytes := Hesper.WebGPU.BufferOps.uint32ToBytes 0
          withTempBufFromBytes ctx startPosBytes fun paramsBuf => do
            let shader := Hesper.WGSL.FlashAttention.flashAttentionBatchKernel
              numHeads numKVHeads maxSeqLen headDim seqLen scale
            GPUBackend.execute ctx shader
              [("q", qBuf), ("k_cache", kCacheBuf), ("v_cache", vCacheBuf),
               ("output", outBuf), ("params", paramsBuf)]
              ({ numWorkgroups := (numHeads, seqLen, 1) : Hesper.ExecConfig })
            let outBytes ← GPUBackend.readBuffer ctx outBuf (qDim * seqLen * 4).toUSize
            pure (byteArrayToF32Array (lastTokenBytes outBytes qDim) qDim)

/-- Test `flashAttentionBatchKernel` at layer `li`, seqLen=5.

    Feeds llama.cpp's post-RoPE Q (`Qcur_pos-<li>`) and pre-populates the KV
    cache from `Kcur_normed-<li>`/`Vcur_normed-<li>` via the already-
    unit-tested `fusedRopeKAndCacheWriteBatchKernel`.  Compares the
    last-token slice of hesper's flash-attn output against llama.cpp's
    `__fattn__-<li>` last token. -/
unsafe def testFlashAttnAtLayer (ctx : CUDAContext) (gguf : Hesper.GGUF.GGUFFile)
    (li : Nat) (numHeads numKVHeads headDim maxSeqLen seqLen : Nat)
    (ropeBase scale : Float) (freqFactorsTensor : Option String)
    (threshold : Float) : IO TestSeq := do
  let qDim := numHeads * headDim
  let kvDim := numKVHeads * headDim
  -- Load inputs
  let qBytes ← loadFloat32Bin s!"{goldenDir}/Qcur_pos-{li}.bin"
  let kBytes ← loadFloat32Bin s!"{goldenDir}/Kcur_normed-{li}.bin"
  let vBytes ← loadFloat32Bin s!"{goldenDir}/Vcur_normed-{li}.bin"
  if qBytes.size ≠ qDim * seqLen * 4 then
    throw (IO.userError s!"Qcur_pos-{li}.bin size={qBytes.size}, expected {qDim * seqLen * 4}")
  if kBytes.size ≠ kvDim * seqLen * 4 then
    throw (IO.userError s!"Kcur_normed-{li}.bin size={kBytes.size}, expected {kvDim * seqLen * 4}")
  if vBytes.size ≠ kvDim * seqLen * 4 then
    throw (IO.userError s!"Vcur_normed-{li}.bin size={vBytes.size}, expected {kvDim * seqLen * 4}")
  let freqFactorsBytes ← match freqFactorsTensor with
    | some tname => extractF32 gguf tname
    | none =>
      let dimPairs := headDim / 2
      let mut bytes := ByteArray.empty
      for _ in [0:dimPairs] do
        bytes := bytes.push 0; bytes := bytes.push 0
        bytes := bytes.push 0x80; bytes := bytes.push 0x3F
      pure bytes
  -- Expected: last token of __fattn__-<li>
  let expFull ← loadFloat32Bin s!"{goldenDir}/__fattn__-{li}.bin"
  if expFull.size ≠ qDim * seqLen * 4 then
    throw (IO.userError s!"__fattn__-{li}.bin size={expFull.size}, expected {qDim * seqLen * 4}")
  let expected := byteArrayToF32Array (lastTokenBytes expFull qDim) qDim
  -- Run hesper
  let actual ← runFlashAttnBatchLastToken ctx qBytes kBytes vBytes freqFactorsBytes
    numHeads numKVHeads maxSeqLen headDim seqLen ropeBase scale
  let rel := relDiff actual expected
  IO.println s!"[FlashAttn L{li} batched numHeads={numHeads} numKVHeads={numKVHeads} headDim={headDim} seqLen={seqLen} scale={scale}] rel = {rel}"
  pure (test s!"hesper flashAttentionBatchKernel L{li} last-token matches llama.cpp __fattn__-{li} (rel={rel} < {threshold})" (rel < threshold))

unsafe def allTests (ctx : CUDAContext) (gguf : Hesper.GGUF.GGUFFile)
    : IO (List (String × List TestSeq)) := do
  -- L0: SWA, numHeads=8, numKVHeads=2, headDim=256, ropeBase=10000, no freq_factors
  let t0 ← testFlashAttnAtLayer ctx gguf 0 8 2 256 131072 5 10000 1.0 none 1e-3
  -- L17: full-attn, numHeads=8, numKVHeads=2, headDim=512, ropeBase=1000000, with rope_freqs
  let t17 ← testFlashAttnAtLayer ctx gguf 17 8 2 512 131072 5 1000000 1.0
    (some "rope_freqs.weight") 1e-3
  pure [
    ("FlashAttention L0 batched last-token (SWA)", [t0]),
    ("FlashAttention L17 batched last-token (full-attn)", [t17])
  ]

end Hesper.Tests.GoldenUnit.FlashAttention
