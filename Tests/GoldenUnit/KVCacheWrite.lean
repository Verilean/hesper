import LSpec
import Hesper.Backend.CUDA
import Hesper.CUDA.FFI
import Hesper.Models.Gemma4
import Tests.GoldenUnit.Common

/-!
# KV cache write golden-unit tests

`fusedRopeKAndCacheWriteBatchKernel` writes rotated K and plain V into
the KV cache at slots `[kvH, startPos+col, d]`.

KV cache layout: `[numKVHeads, maxSeqLen, headDim]`
  → element (kvH, pos, d) at offset `kvH*maxSeqLen*headDim + pos*headDim + d`.

Reference: after the kernel runs with `startPos=0` and `seqLen=5`, the
cache slots [0..5) should contain:
  k_cache[kvH, t, d] = RoPE(new_k[t,kvH,:])[d]   (llama.cpp's Kcur_pos-<li>[t, kvH, d])
  v_cache[kvH, t, d] = new_v[t, kvH, d]          (llama.cpp's Vcur-<li> without RoPE)

We compare the LAST token's slot (t=seqLen-1) across all KV heads for
one layer.  A bug in `pos`, `startPos`, or the cache layout would
show up here (e.g., if all tokens get written to slot 0, every
token's query would align with token-0's K during decode — which is
exactly the E2E symptom: prompt repeats at decode).

Tests:
- `testKVCacheWriteL0`: SWA L0, numKVHeads=4, headDim=256, ropeBase=10000
- `testKVCacheWriteL17`: full-attn L17, numKVHeads=2, headDim=512, ropeBase=1000000
-/

namespace Hesper.Tests.GoldenUnit.KVCacheWrite

open LSpec
open Hesper
open Hesper.Tests.GoldenUnit.Common

/-- Read one cache slot (kvHead, pos) of size headDim from a
    [numKVHeads, maxSeqLen, headDim] cache buffer (read all bytes,
    slice out the desired region). -/
def sliceCacheSlot (cacheBytes : ByteArray) (numKVHeads maxSeqLen headDim : Nat)
    (kvH pos : Nat) : ByteArray :=
  let _ := numKVHeads
  let offset := (kvH * maxSeqLen * headDim + pos * headDim) * 4
  cacheBytes.extract offset (offset + headDim * 4)

/-- Run hesper's fusedRopeKAndCacheWriteBatchKernel over seqLen tokens
    and return the (k_cache last-token slots, v_cache last-token slots)
    concatenated across all KV heads (total: numKVHeads*headDim floats each). -/
unsafe def runKVCacheWriteBatch
    (ctx : CUDAContext) (newKBytes newVBytes freqFactorsBytes : ByteArray)
    (numKVHeads maxSeqLen headDim seqLen : Nat) (ropeBase : Float)
    : IO (Array Float × Array Float) := do
  let kvDim := numKVHeads * headDim
  let cacheSize := numKVHeads * maxSeqLen * headDim
  withTempBufFromBytes ctx newKBytes fun kInBuf => do
    withTempBufFromBytes ctx newVBytes fun vInBuf => do
      withTempBufFromBytes ctx freqFactorsBytes fun freqBuf => do
        -- Zero-init cache buffers, else we might read garbage on untouched slots
        let zeros : ByteArray := (List.replicate (cacheSize * 4) (0 : UInt8)).toByteArray
        withTempBufFromBytes ctx zeros fun kCacheBuf => do
          withTempBufFromBytes ctx zeros fun vCacheBuf => do
            -- params: startPos = 0
            let startPosBytes := Hesper.WebGPU.BufferOps.uint32ToBytes 0
            withTempBufFromBytes ctx startPosBytes fun paramsBuf => do
              let shader := Hesper.Models.Gemma4.fusedRopeKAndCacheWriteBatchKernel numKVHeads maxSeqLen headDim seqLen ropeBase
              GPUBackend.execute ctx shader
                [("new_k", kInBuf), ("new_v", vInBuf),
                 ("k_cache", kCacheBuf), ("v_cache", vCacheBuf),
                 ("params", paramsBuf), ("freq_factors", freqBuf)]
                (.dispatch1D (numKVHeads * headDim / 2 * seqLen))
              let kBytes ← GPUBackend.readBuffer ctx kCacheBuf (cacheSize * 4).toUSize
              let vBytes ← GPUBackend.readBuffer ctx vCacheBuf (cacheSize * 4).toUSize
              -- Collect slots at pos = seqLen - 1 across all heads
              let mut kLast : Array Float := Array.mkEmpty kvDim
              let mut vLast : Array Float := Array.mkEmpty kvDim
              for kvH in [0:numKVHeads] do
                let kSlot := sliceCacheSlot kBytes numKVHeads maxSeqLen headDim kvH (seqLen - 1)
                let vSlot := sliceCacheSlot vBytes numKVHeads maxSeqLen headDim kvH (seqLen - 1)
                kLast := kLast ++ byteArrayToF32Array kSlot headDim
                vLast := vLast ++ byteArrayToF32Array vSlot headDim
              pure (kLast, vLast)

/-- Collect llama.cpp's expected last-token K (post-RoPE) and V for layer `li`.
    llama.cpp dumps:
      - Kcur_pos-<li>    : [seqLen, numKVHeads, headDim] col-major (post-RoPE K)
      - Vcur_normed-<li> : [seqLen, numKVHeads, headDim] col-major (post qkv_norm V)
    (V_norm: note Gemma 4 has a v_norm weight, so llama dumps Vcur_normed.)

    Hesper's cache receives:
      - K = RoPE(new_k)  where new_k = output of q/k/v norm
      - V = new_v        (no RoPE on V)
    So:
      k_cache[kvH, t, d]  should equal  Kcur_pos-<li>[t, kvH, d]
      v_cache[kvH, t, d]  should equal  Vcur_normed-<li>[t, kvH, d]
-/
def gatherLastTokenPerHead (bytes : ByteArray) (numKVHeads headDim seqLen : Nat) : Array Float := Id.run do
  let kvDim := numKVHeads * headDim
  -- Last token block: byte range [(seqLen-1)*kvDim*4, seqLen*kvDim*4)
  let lastBlock := bytes.extract ((seqLen - 1) * kvDim * 4) (seqLen * kvDim * 4)
  byteArrayToF32Array lastBlock kvDim

unsafe def testKVCacheWriteAtLayer (ctx : CUDAContext) (gguf : Hesper.GGUF.GGUFFile)
    (li : Nat) (numKVHeads headDim maxSeqLen seqLen : Nat) (ropeBase : Float)
    (freqFactorsTensor : Option String) (threshold : Float) : IO (TestSeq × TestSeq) := do
  let kvDim := numKVHeads * headDim
  let newKBytes ← loadFloat32Bin s!"{goldenDir}/Kcur_normed-{li}.bin"
  let newVBytes ← loadFloat32Bin s!"{goldenDir}/Vcur_normed-{li}.bin"
  if newKBytes.size ≠ kvDim * seqLen * 4 then
    throw (IO.userError s!"Kcur_normed-{li}.bin size={newKBytes.size}, expected {kvDim * seqLen * 4}")
  if newVBytes.size ≠ kvDim * seqLen * 4 then
    throw (IO.userError s!"Vcur_normed-{li}.bin size={newVBytes.size}, expected {kvDim * seqLen * 4}")
  let freqFactorsBytes ← match freqFactorsTensor with
    | some tname => extractF32 gguf tname
    | none =>
      let dimPairs := headDim / 2
      let mut bytes := ByteArray.empty
      for _ in [0:dimPairs] do
        bytes := bytes.push 0; bytes := bytes.push 0
        bytes := bytes.push 0x80; bytes := bytes.push 0x3F
      pure bytes
  -- Expected K (from llama.cpp's Kcur_pos) and V (from Vcur_normed, no RoPE on V)
  let kPosBytes ← loadFloat32Bin s!"{goldenDir}/Kcur_pos-{li}.bin"
  let kExpected := gatherLastTokenPerHead kPosBytes numKVHeads headDim seqLen
  let vExpected := gatherLastTokenPerHead newVBytes numKVHeads headDim seqLen
  let (kActual, vActual) ← runKVCacheWriteBatch ctx newKBytes newVBytes freqFactorsBytes
    numKVHeads maxSeqLen headDim seqLen ropeBase
  let kRel := relDiff kActual kExpected
  let vRel := relDiff vActual vExpected
  IO.println s!"[KVCacheWrite L{li} K (post-RoPE), numKVHeads={numKVHeads}, headDim={headDim}] rel = {kRel}"
  IO.println s!"[KVCacheWrite L{li} V (no RoPE),   numKVHeads={numKVHeads}, headDim={headDim}] rel = {vRel}"
  let kTest := test s!"fusedRopeKAndCacheWriteBatchKernel L{li} K at slot seqLen-1 (rel={kRel} < {threshold})" (kRel < threshold)
  let vTest := test s!"fusedRopeKAndCacheWriteBatchKernel L{li} V at slot seqLen-1 (rel={vRel} < {threshold})" (vRel < threshold)
  pure (kTest, vTest)

unsafe def allTests (ctx : CUDAContext) (gguf : Hesper.GGUF.GGUFFile)
    : IO (List (String × List TestSeq)) := do
  -- L0: SWA, numKVHeads=2, headDim=256, ropeBase=10000, no freq_factors
  let (k0, v0) ← testKVCacheWriteAtLayer ctx gguf 0 2 256 131072 5 10000 none 1e-4
  -- L17: full-attn, numKVHeads=2, headDim=512, ropeBase=1000000, with rope_freqs
  let (k17, v17) ← testKVCacheWriteAtLayer ctx gguf 17 2 512 131072 5 1000000 (some "rope_freqs.weight") 1e-4
  pure [
    ("KVCacheWrite L0 K slot seqLen-1 (SWA)", [k0]),
    ("KVCacheWrite L0 V slot seqLen-1 (SWA)", [v0]),
    ("KVCacheWrite L17 K slot seqLen-1 (full-attn)", [k17]),
    ("KVCacheWrite L17 V slot seqLen-1 (full-attn)", [v17])
  ]

end Hesper.Tests.GoldenUnit.KVCacheWrite
