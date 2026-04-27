import Hesper.Backend
import Hesper.Backend.WebGPU
import Hesper.Backend.CUDA
import Hesper.Circuit.IR
import Hesper.Circuit.Lowering
import Hesper.Circuit.Passes
import Hesper.WGSL.Monad
import Hesper.WGSL.Execute
import Hesper.WGSL.Exp
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.Layers.Linear
import Hesper.Layers.RMSNorm
import Hesper.Layers.RoPE
import Hesper.Layers.Embedding
import Hesper.Layers.Softmax
import Hesper.Quantization.Q4_K_M
import Hesper.Layers.MoE
import Hesper.Layers.PerLayerEmbedding
import Hesper.GGUF.Parser
import Hesper.GGUF.Loader
import Hesper.Basic
import Hesper.Logging
import Hesper.WGSL.MatMul
import Hesper.WebGPU.BufferOps
import Hesper.Inference.Sampling
import Hesper.WGSL.FlashAttention
import Hesper.WGSL.FlashAttentionExperiments
import Hesper.Layers.Attention
import Hesper.Models.Gemma4.Config
import Hesper.Circuit.Dispatch_v2
import Hesper.Models.Gemma4.Kernels
import Hesper.Models.Gemma4.Types
import Hesper.Models.Gemma4.Loader

/-!
# Gemma 4 Model Implementation

Implements the Gemma 4 transformer with:
- ISWA (Interleaved Sliding Window Attention)
- Hybrid MoE + dense FFN (MoE deferred to Phase 3)
- Per-layer embeddings (deferred to Phase 4)
- Q/K normalization
- Logit softcapping
- KV cache sharing (deferred to Phase 4)

Reference: llama.cpp/src/models/gemma4-iswa.cpp

## Architecture

```
embed * sqrt(hiddenSize)
for each layer:
  attnNorm -> Q/K/V projections -> Q-norm, K-norm, V-norm -> RoPE -> attention -> postAttnNorm -> + residual
  ffnNorm -> GeGLU FFN -> postFFNNorm -> + residual
  [per_layer_embedding (Phase 4)]
  [layer_scale]
finalNorm -> lm_head -> logit_softcap
```
-/

namespace Hesper.Models.Gemma4

open Hesper.WGSL
open Hesper.WGSL.Monad
open Hesper.WebGPU
open Hesper.WGSL.Execute (PreparedDispatch CompiledKernel)
open Hesper.Layers
open Hesper.Logging (logVerbose)

set_option maxHeartbeats 400000

/-! ## KV Cache State -/

/-- Per-layer KV cache for Gemma 4 -/
structure Gemma4KVCache (BufT : Type) where
  kBuf : BufT    -- [numKVHeads, maxSeqLen, headDim] f32 — used by legacy
                 -- FA + Tiled FA + existing batched RoPE-K paths.
  vBuf : BufT    -- [numKVHeads, maxSeqLen, headDim] f32 — same.
  /-- f16-packed K cache (half byte size).  Byte size = `cacheSize * 2`
      (vs `cacheSize * 4` for kBuf).  Stores u32 with two f16 packed:
      lo = elem at even index, hi = elem at odd index.
      Populated by `fusedRopeKAndCacheWriteKernelF16` when the V11 path
      is enabled (HESPER_FA_V11=1).  Read directly by V11 partial kernel
      via `unpack2x16float`.  Stays zero-init when V11 is off. -/
  kBufF16 : BufT
  /-- f16-packed V cache (half byte size).  Same layout as kBufF16.
      Populated by f32→f16 pack kernel (one extra dispatch per layer
      per token) immediately after V projection writes vBuf, since V
      doesn't go through RoPE so we can't fuse the pack inside a
      transformation kernel. -/
  vBufF16 : BufT

/-- Full inference state -/
structure InferenceState (BufT CacheT : Type) where
  kvCaches : Array (Gemma4KVCache BufT)
  buf1 : BufT          -- [hiddenSize] ping-pong
  buf2 : BufT          -- [hiddenSize] ping-pong
  qBuf : BufT          -- [numHeads * headDim] Q projection output
  kBuf : BufT          -- [numKVHeads * headDim] K projection output
  vBuf : BufT          -- [numKVHeads * headDim] V projection output
  attnOutBuf : BufT    -- [numHeads * headDim] attention output
  gateBuf : BufT       -- [intermediateSize] FFN gate output
  upBuf : BufT         -- [intermediateSize] FFN up output
  geluBuf : BufT       -- [intermediateSize] GELU*up output
  ffnOutBuf : BufT     -- [hiddenSize] FFN down output
  normedBuf : BufT     -- [hiddenSize] normalized output
  attnResidualBuf : BufT  -- [hiddenSize] attn output + residual (between attn and FFN)
  qBuf2 : BufT            -- [maxQDim] alternate Q buffer (for in-place ops)
  kBuf2 : BufT            -- [maxKVDim] alternate K buffer
  vBuf2 : BufT            -- [maxKVDim] alternate V buffer
  normedBuf2 : BufT       -- [hiddenSize] alternate normed buffer
  logitsBuf : BufT     -- [vocabSize]
  logitsBuf2 : BufT    -- [vocabSize] scratch for logit softcap (no aliasing)
  tokenBuf : BufT      -- [1] u32 for single token
  paramsBuf : BufT     -- [2] u32: (pos, cacheLen) for RoPE
  posF32Buf : BufT     -- [1] f32: pos as f32 (for Circuit DSL dynamic offsets)
  -- MoE buffers
  moeRouterOutBuf : BufT    -- [hiddenSize] router preprocessed input
  moeLogitsBuf : BufT       -- [numExperts] router logits
  moeIndicesBuf : BufT      -- [numExpertsUsed] selected expert indices
  moeWeightsBuf : BufT      -- [numExpertsUsed] expert weights
  moeExpertOutBuf : BufT    -- [hiddenSize] combined expert output
  moeExpertGateBuf : BufT   -- [expertFFSize] expert gate projection output
  moeExpertUpBuf : BufT     -- [expertFFSize] expert up projection output
  moeExpertGeluBuf : BufT   -- [expertFFSize] expert GELU*up output
  moeExpertDownBuf : BufT   -- [hiddenSize] single expert down output
  moeNormedBuf : BufT       -- [hiddenSize] pre_norm_2 output for routed experts
  -- Per-layer embedding buffers
  plGateBuf : BufT          -- [embdPerLayer] per-layer gate output
  plProjBuf : BufT          -- [hiddenSize] per-layer projected output
  -- Per-layer input precomputation (computed once per token, used by all layers)
  plTokenSelected : BufT    -- [embdPerLayer * numLayers] tok_embd_per_layer[token] dequantized
  plModelProj : BufT        -- [embdPerLayer * numLayers] per_layer_model_proj @ scaled_embed
  plInputAll : BufT         -- [embdPerLayer * numLayers] final per-layer input (sum, normed, scaled)
  -- Partial buffer for tiled (split-K) flash attention. Pre-allocated
  -- at createInferenceState with size for the maximum tile count.
  flashPartialBuf : BufT
  -- V11 split-K partial output: [numHeads, numSplits, headDim] f32.
  -- numSplits = 8.  Sized for max numHeads * 8 * 256.
  flashPartialOutV11 : BufT
  -- V11 split-K partial meta: [numHeads, numSplits, 2] f32 (max, sum).
  flashPartialMetaV11 : BufT
  -- Small GPU-side scratch for the raw Q6_K bytes of one per-layer
  -- embedding row (~33 KB for Gemma 4 e4b). The full per-layer
  -- embedding table lives on CPU (> WebGPU single-buffer limit); at
  -- decode time we slice the needed row out of the CPU ByteArray,
  -- upload it here, and dequant on-GPU via `q6kSingleRowDequantScaleKernel`.
  -- Sized to the maximum row bytes seen at load time; left as a small
  -- placeholder when per-layer embeddings are absent.
  plRawRowBuf : BufT
  -- Optional: pre-softcap logits buffer for TTT surprise sensor.
  -- When `some`, forwardSingleToken copies logitsBuf here BEFORE
  -- applying logit softcap. When `none` (default), no copy is done
  -- and there is zero performance impact.
  preSoftcapBuf : Option BufT := none
  argmaxBuf : BufT              -- [1] u32 for GPU-side argmax result
  /-- Cached dispatch for the decode-loop argmax kernel.  Without this,
      `GPUBackend.execute` rebuilds PTX / re-resolves the buffer arg
      vector on every call (~150 µs/token × 10 tokens = 1.5 ms wasted). -/
  argmaxCacheRef : IO.Ref (Option CacheT)
  /-- Optional **host-mapped** argmax slot (CUDA only, opt-in via
      `HESPER_DEVICE_ARGMAX=1`).  When set, `(hostPtr, devBuf)`:
      - `devBuf` aliases the same memory as `argmaxBuf` would, but is
        `cuMemHostAlloc`'d so it lives in pinned host memory mapped into
        the device VA.  The argmaxKernel writes the token id to it.
      - `hostPtr` is the raw host pointer.  After one
        `cuStreamSynchronize`, the host can read the u32 with a plain
        memory load — no `cuMemcpyDtoH` (which is implicitly synchronous
        and currently costs ~9.8 ms/token of GPU drain wait).
      Matches llama-cli's `cudaMallocHost`-then-direct-read pattern. -/
  argmaxHostMapped : Option (USize × BufT) := none
  -- doc 58 step B: deferred-argmax-read history buffer.  Each decode
  -- iteration the historyAppendKernel writes the argmax result into
  -- argmaxHistoryBuf[historySlotBuf[0]] and increments the slot.
  -- After the decode loop ends (or every K tokens) we DtoH the whole
  -- history once and recover the per-token IDs.  This breaks the
  -- post-argmax → next-forward GPU idle bubble, since host never has
  -- to wait for the per-iter argmax value.
  argmaxHistoryBuf : BufT
  historySlotBuf   : BufT
  -- Scratch buffer for Q8_1 quantized lmHead input (hiddenSize/32 * 9 u32),
  -- lazily allocated on first dp4a-enabled lmHead call.
  lmHeadQ8Buf : IO.Ref (Option BufT)
  lmHeadQuantizePrepared : IO.Ref (Option CacheT)
  lmHeadDP4APrepared : IO.Ref (Option CacheT)
  -- Pooled Q8_1 scratch buffers used by forwardPrefillBatch, one for the
  -- attention input path and one for the FFN input path.  Stored as
  -- `(ref, sizeBytes)`; we reuse when the request fits, reallocate when it
  -- doesn't.  Eliminates 2 × layers × 2 = ~168 cudaMalloc/cudaFree calls
  -- per decode token at seqLen=1.
  prefillAttnQ8BufRef : IO.Ref (Option (BufT × USize))
  prefillFfnQ8BufRef  : IO.Ref (Option (BufT × USize))
  -- Pooled small per-call scratch buffers used by forwardPrefillBatch.
  -- Each holds `seqLen * 4` bytes of u32 values.  Pool by `(buf, capBytes)`
  -- so we only re-allocate when the next call needs a larger seqLen.
  prefillTokenIdsRef  : IO.Ref (Option (BufT × USize))
  prefillPosRef       : IO.Ref (Option (BufT × USize))
  prefillCacheLenRef  : IO.Ref (Option (BufT × USize))
  prefillColIdxRef    : IO.Ref (Option BufT)
  -- Pooled per-layer-embedding batch tensor `[embdPerLayer * numLayers, seqLen]`.
  -- Only re-allocated when seqLen grows.
  prefillPLInputAllRef : IO.Ref (Option (BufT × USize))
  /-- Pinned host pointer (4 bytes).  See §CUDA Graph notes in the
      InferenceState doc header. -/
  stagingTokenPtr   : USize := 0
  /-- Pinned host pointer (8 bytes — pos @0, cacheLen @4). -/
  stagingParamsPtr  : USize := 0
  /-- Pinned host pointer (4 bytes) for per-layer-embedding row index. -/
  stagingPLRowPtr   : USize := 0
  /-- Pinned host pointer (4 bytes) for the batch-prefill column index. -/
  stagingColIdxPtr  : USize := 0
  /-- Pinned host pointer (4 bytes) for state.posF32Buf (pos as f32 —
      needed by the Circuit DSL scatter addrExpr that writes to the
      KV cache inside fusedRopeKAndCacheWrite). -/
  stagingPosF32Ptr  : USize := 0
  /-- Persistent non-default stream used to avoid implicit synchronous
      behavior of the legacy null stream.  Allocated once at state init.
      When set (HESPER_UNIFIED_STREAM=1), all async H2D copies and
      kernel launches funnel into this stream so they serialise in
      insertion order *without* host-side stalls from sync
      `cuMemcpyHtoD_v2`.  Pinned-slot reuse becomes race-free because
      CUDA's in-stream ordering guarantees the previous copy completes
      before the next host write visible to the device.  0 = disabled
      (legacy null-stream behaviour). -/
  unifiedStream     : USize := 0

/-- Dynamic cache ref store. Lazily creates IO.Ref per unique cacheKey. -/
structure KernelCacheRefs (CacheT : Type) where
  store : IO.Ref (Array (UInt64 × IO.Ref (Option CacheT)))

def KernelCacheRefs.getRef (kcr : KernelCacheRefs CacheT) (key : UInt64) : IO (IO.Ref (Option CacheT)) := do
  let arr ← kcr.store.get
  match arr.find? (fun (k, _) => k == key) with
  | some (_, r) => pure r
  | none =>
    let r ← IO.mkRef none
    kcr.store.modify (·.push (key, r))
    pure r

def createKernelCacheRefs [GPUBackend β] : IO (KernelCacheRefs (GPUBackend.CachedDispatch β)) := do
  pure { store := ← IO.mkRef #[] }

/-- Create inference state with pre-allocated buffers -/
def createInferenceState [GPUBackend β] (ctx : β) (cfg : Config) : IO (InferenceState (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)) := do
  let mkBuf := fun (size : Nat) => GPUBackend.allocBuffer ctx (size * 4).toUSize
  let maxHeadDim := max cfg.headDimFull cfg.headDimSWA
  let maxQDim := cfg.numAttentionHeads * maxHeadDim
  let maxKVDim := (max cfg.numKeyValueHeadsFull cfg.numKeyValueHeadsSWA) * maxHeadDim

  -- Create per-layer KV caches
  let mut kvCaches : Array (Gemma4KVCache (GPUBackend.Buf β)) := #[]
  for li in [0:cfg.numHiddenLayers] do
    let numKVHeads := cfg.numKVHeads li
    let headDim := cfg.headDim li
    let cacheSize := numKVHeads * cfg.maxSeqLen * headDim
    let kBuf ← mkBuf cacheSize
    let vBuf ← mkBuf cacheSize
    -- f16 mirrors for V11 — half the byte size (u16 per element, packed
    -- 2× per u32).  Total: kBuf+vBuf = 8B/elem, +kBufF16+vBufF16 = 4B/elem.
    -- Net VRAM cost: +50% over f32-only path, since both stay live during
    -- the V11 transition.  Once everything moves to f16-only (later
    -- session), the f32 buffers are deleted and net VRAM is -50%.
    let kBufF16 ← GPUBackend.allocBuffer ctx (cacheSize * 2).toUSize
    let vBufF16 ← GPUBackend.allocBuffer ctx (cacheSize * 2).toUSize
    kvCaches := kvCaches.push
      ({ kBuf, vBuf, kBufF16, vBufF16 } : Gemma4KVCache (GPUBackend.Buf β))

  return {
    kvCaches
    buf1 := ← mkBuf cfg.hiddenSize
    buf2 := ← mkBuf cfg.hiddenSize
    qBuf := ← mkBuf maxQDim
    kBuf := ← mkBuf maxKVDim
    vBuf := ← mkBuf maxKVDim
    attnOutBuf := ← mkBuf maxQDim
    gateBuf := ← mkBuf cfg.intermediateSize
    upBuf := ← mkBuf cfg.intermediateSize
    geluBuf := ← mkBuf cfg.intermediateSize
    ffnOutBuf := ← mkBuf cfg.hiddenSize
    normedBuf := ← mkBuf cfg.hiddenSize
    attnResidualBuf := ← mkBuf cfg.hiddenSize
    qBuf2 := ← mkBuf maxQDim
    kBuf2 := ← mkBuf maxKVDim
    vBuf2 := ← mkBuf maxKVDim
    normedBuf2 := ← mkBuf cfg.hiddenSize
    logitsBuf := ← mkBuf cfg.vocabSize
    logitsBuf2 := ← mkBuf cfg.vocabSize
    tokenBuf := ← GPUBackend.allocBuffer ctx (4 : USize)
    paramsBuf := ← GPUBackend.allocBuffer ctx (8 : USize)
    posF32Buf := ← GPUBackend.allocBuffer ctx (4 : USize)
    moeRouterOutBuf := ← mkBuf cfg.hiddenSize
    moeLogitsBuf := ← mkBuf (max cfg.numExperts 1)
    moeIndicesBuf := ← GPUBackend.allocBuffer ctx (max cfg.numExpertsUsed 1 * 4).toUSize
    moeWeightsBuf := ← mkBuf (max cfg.numExpertsUsed 1)
    moeExpertOutBuf := ← mkBuf cfg.hiddenSize
    moeExpertGateBuf := ← mkBuf (max cfg.expertFFSize 1)
    moeExpertUpBuf := ← mkBuf (max cfg.expertFFSize 1)
    moeExpertGeluBuf := ← mkBuf (max cfg.expertFFSize 1)
    moeExpertDownBuf := ← mkBuf cfg.hiddenSize
    moeNormedBuf := ← mkBuf cfg.hiddenSize
    plGateBuf := ← mkBuf (max cfg.embdPerLayer 1)
    plProjBuf := ← mkBuf cfg.hiddenSize
    plTokenSelected := ← mkBuf (max (cfg.embdPerLayer * cfg.numHiddenLayers) 1)
    plModelProj := ← mkBuf (max (cfg.embdPerLayer * cfg.numHiddenLayers) 1)
    plInputAll := ← mkBuf (max (cfg.embdPerLayer * cfg.numHiddenLayers) 1)
    flashPartialBuf := ← FlashAttention.createFlashPartialBuffer ctx
                          cfg.numAttentionHeads cfg.maxSeqLen (max cfg.headDimFull cfg.headDimSWA)
    -- V11 split-K: numHeads * 8 * D * 4 bytes for partial_out, numHeads * 8 * 2 * 4 for meta.
    flashPartialOutV11 := ← GPUBackend.allocBuffer ctx
      (cfg.numAttentionHeads * 8 * (max cfg.headDimFull cfg.headDimSWA) * 4).toUSize
    flashPartialMetaV11 := ← GPUBackend.allocBuffer ctx
      (cfg.numAttentionHeads * 8 * 2 * 4).toUSize
    plRawRowBuf := ← do
      -- Raw Q6_K bytes for one per-layer-embd row:
      --   blocksPerRow = ceil((embdPerLayer * numLayers) / 256)
      --   rowBytes     = blocksPerRow * 210
      let totalPL := cfg.embdPerLayer * cfg.numHiddenLayers
      let blocksPerRow := (totalPL + 255) / 256
      let rowBytes := blocksPerRow * 210
      GPUBackend.allocBuffer ctx (max rowBytes 4).toUSize
    argmaxBuf := ← GPUBackend.allocBuffer ctx (4 : USize)
    argmaxCacheRef := ← GPUBackend.newCacheRef (β := β)
    -- HESPER_DEVICE_ARGMAX=1: skip the per-token cuMemcpyDtoH(4 byte) by
    -- having the argmaxKernel write into a pinned host-mapped slot the
    -- host can read directly.  Closes the 9.8 ms/tok GPU drain bubble
    -- (doc 55).  Falls through to `argmaxBuf` when not set.
    argmaxHostMapped := ← do
      if (← IO.getEnv "HESPER_DEVICE_ARGMAX").isSome then
        let (hostPtr, devPtr) ← Hesper.CUDA.cuMemAllocHostMapped 4
        match ← GPUBackend.bufFromRawDevicePtr ctx devPtr 4 with
        | some buf => pure (some (hostPtr, buf))
        | none =>
          IO.println "[Gemma4] HESPER_DEVICE_ARGMAX requested but backend lacks UVA support; falling back to DtoH"
          pure none
      else pure none
    -- doc 58 step B: 64 K u32 history (covers any reasonable maxTokens).
    -- Slot starts at 0 and is incremented by historyAppendKernel each decode.
    argmaxHistoryBuf := ← GPUBackend.allocBuffer ctx (65536 * 4 : USize)
    historySlotBuf   := ← GPUBackend.allocBuffer ctx (4 : USize)
    lmHeadQ8Buf := ← IO.mkRef none
    lmHeadQuantizePrepared := ← GPUBackend.newCacheRef (β := β)
    lmHeadDP4APrepared := ← GPUBackend.newCacheRef (β := β)
    prefillAttnQ8BufRef := ← IO.mkRef none
    prefillFfnQ8BufRef  := ← IO.mkRef none
    prefillTokenIdsRef  := ← IO.mkRef none
    prefillPosRef       := ← IO.mkRef none
    prefillCacheLenRef  := ← IO.mkRef none
    prefillColIdxRef    := ← IO.mkRef none
    prefillPLInputAllRef := ← IO.mkRef none
    -- Pinned-host staging + unified stream.  Enabled when either
    -- HESPER_CUDA_GRAPHS != "0" (graphs are ON by default, see
    -- forwardLoop's useCudaGraphs decision) or HESPER_UNIFIED_STREAM=1
    -- (single non-null stream for the llama.cpp-style "big call, few ops"
    -- pattern).
    stagingTokenPtr  := ← do
      let graphsOn := match ← IO.getEnv "HESPER_CUDA_GRAPHS" with
        | some "0" => false | _ => true
      if graphsOn || (← IO.getEnv "HESPER_UNIFIED_STREAM").isSome
      then Hesper.CUDA.cuMemAllocHost 4 else pure 0
    stagingParamsPtr := ← do
      let graphsOn := match ← IO.getEnv "HESPER_CUDA_GRAPHS" with
        | some "0" => false | _ => true
      if graphsOn || (← IO.getEnv "HESPER_UNIFIED_STREAM").isSome
      then Hesper.CUDA.cuMemAllocHost 8 else pure 0
    stagingPLRowPtr  := ← do
      let graphsOn := match ← IO.getEnv "HESPER_CUDA_GRAPHS" with
        | some "0" => false | _ => true
      if graphsOn || (← IO.getEnv "HESPER_UNIFIED_STREAM").isSome
      then Hesper.CUDA.cuMemAllocHost 4 else pure 0
    stagingColIdxPtr := ← do
      let graphsOn := match ← IO.getEnv "HESPER_CUDA_GRAPHS" with
        | some "0" => false | _ => true
      if graphsOn || (← IO.getEnv "HESPER_UNIFIED_STREAM").isSome
      then Hesper.CUDA.cuMemAllocHost 4 else pure 0
    stagingPosF32Ptr := ← do
      let graphsOn := match ← IO.getEnv "HESPER_CUDA_GRAPHS" with
        | some "0" => false | _ => true
      if graphsOn || (← IO.getEnv "HESPER_UNIFIED_STREAM").isSome
      then Hesper.CUDA.cuMemAllocHost 4 else pure 0
    unifiedStream    := ← match ← IO.getEnv "HESPER_UNIFIED_STREAM" with
                         | some _ => Hesper.CUDA.cuStreamCreate
                         | none   => pure (0 : USize)
  }

/-- Dump a buffer to a file when HESPER_DUMP_DIR env is set.
    Caller passes the byte count; suffix identifies the checkpoint.
    Flushes any pending CUDA batch queue first so the buffer reflects all
    queued launches.  If there was an active batch, reopens it afterwards. -/
def dumpBuf [GPUBackend β] (ctx : β) (buf : GPUBackend.Buf β) (bytes : USize) (suffix : String) : IO Unit := do
  match ← IO.getEnv "HESPER_DUMP_DIR" with
  | none => pure ()
  | some dir =>
    -- Probe CUDA batch state: if currently batching, queue sync returns
    -- some; we flush + reopen only in that case.  Safe regardless of backend
    -- because endBatch on `none` is a no-op.
    let wasBatching ← Hesper.Backend.isCudaBatching
    GPUBackend.endBatch ctx
    let data ← GPUBackend.readBuffer ctx buf bytes
    IO.FS.writeBinFile s!"{dir}/{suffix}.bin" data
    if wasBatching then
      GPUBackend.beginBatch ctx

/-- Write a small scalar (≤8 bytes) to a device buffer via a pinned-host
    staging slot.  Safe inside CUDA Graph capture: the resulting memcpy
    node holds a stable host pointer.  Outside capture it is identical
    in effect to `writeBufferOffset` (just uses pinned host memory as
    the source).

    * `ctx`        — backend context
    * `dstBuf`     — device buffer
    * `dstOffset`  — byte offset into `dstBuf`
    * `staging`    — `USize` pinned-host pointer allocated at state init
    * `stOffset`   — byte offset inside the staging slot (usually 0)
    * `data`       — the bytes to write

    Global gate: when `skipStagingWrites` is set, the function is a
    no-op.  Token-graph capture uses this so per-step pos/token writes
    don't become memcpy nodes that all read from the same pinned slot
    (which would race to the last captured value). -/
initialize skipStagingWrites : IO.Ref Bool ← IO.mkRef false

def writeScalarViaStaging [GPUBackend β] (ctx : β)
    (dstBuf : GPUBackend.Buf β) (dstOffset : USize)
    (staging : USize) (stOffset : USize)
    (data : ByteArray) : IO Unit := do
  if ← skipStagingWrites.get then
    return
  if staging == 0 then
    -- No pinned slot (e.g. WebGPU, or CUDA_GRAPHS/UNIFIED_STREAM disabled)
    -- — fall back to sync writeBufferOffset.
    GPUBackend.writeBufferOffset ctx dstBuf dstOffset data
  else
    match ← Hesper.cudaCaptureStream.get with
    | some s =>
      match ← GPUBackend.rawDevicePtr ctx dstBuf with
      | some ptr =>
        -- Fused pinned-write + async H2D copy on the stream.  Single
        -- FFI crossing replaces the 2-call sequence (cuWritePinned +
        -- cuMemcpyHtoDFromPinned).  Halves per-scalar Lean→C overhead.
        Hesper.CUDA.cuPinnedWriteAndCopy
          (ptr + dstOffset) staging stOffset data data.size.toUSize s
      | none =>
        -- Backend can't expose a raw ptr; fall back to the normal path.
        GPUBackend.writeBufferOffset ctx dstBuf dstOffset data
    | none =>
      -- Not capturing — just plain writeBufferOffset is fine.
      GPUBackend.writeBufferOffset ctx dstBuf dstOffset data

/-! ## Single-Token Forward Pass -/

/-- Run single-token forward pass through one transformer block.

    Flow (from gemma4-iswa.cpp):
    1. attnNorm(input) → Q/K/V projections → Q-norm, K-norm → attention → postAttnNorm → + residual
    2. ffnNorm(attn_out) → GeGLU FFN → postFFNNorm → + residual
-/
def forwardBlock [GPUBackend β] (ctx : β)
    (block : Gemma4Block (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)) (cfg : Config)
    (inputBuf outputBuf : GPUBackend.Buf β)
    (state : InferenceState (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)) (pos : Nat)
    (kcr : Option (KernelCacheRefs (GPUBackend.CachedDispatch β)) := none)
    (perLayerEmbd : Option (Gemma4PerLayerEmbd (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)) := none)
    (perLayerInput : Option (GPUBackend.Buf β) := none)
    -- doc 58: when true, skip the writeScalarViaStaging of pos /
    -- posF32 / cacheLen.  Caller must guarantee paramsBuf and
    -- posF32Buf already hold the right values (e.g. via
    -- advancePosKernel at the end of the previous forward).
    (skipPosWrite : Bool := false) : IO Unit := do
  let li := block.layerIdx
  let headDim := cfg.headDim li

  -- Helper: cached execute with named cache key. On 2nd+ call for same
  -- kernel, cacheRef hit skips generatePTX entirely (90-330μs → 0μs).
  let ce := fun (name : String) (shader : ShaderM Unit)
      (namedBufs : List (String × GPUBackend.Buf β)) (config : Hesper.ExecConfig) => do
    -- Key includes name + config (numWorkgroups, workgroupSize) to distinguish
    -- same-named kernels with different parameters (e.g., full vs SWA attention)
    match kcr with
    | some k =>
      let key := hash ("gemma4_ce", name, config.numWorkgroups, config.workgroupSize.x, config.workgroupSize.y, config.workgroupSize.z)
      let ref ← k.getRef key
      -- Tag config.funcName with the call-site name so cacheRef-miss traces
      -- show the human-readable `ce "..."` name instead of a hex cacheKey.
      let configNamed : Hesper.ExecConfig := { config with funcName := name }
      GPUBackend.executeWithConfigCached ctx shader namedBufs configNamed key ref
    | none => GPUBackend.execute ctx shader namedBufs config

  -- Step 1+2: Fused attnNorm + Q/K/V projections
  -- Fuses RMSNorm into each matmul: each WG computes RMS on-the-fly (redundant but cheap).
  -- Eliminates the normedBuf global memory write/read round-trip and the attnNorm dispatch.
  -- Local helper: a Gemma RMSNorm via the Circuit DSL.  Builds 4 ops
  -- (reduce + 3 pointwise) which fuseReduceEpilogue collapses to one
  -- dispatch, matching the hand-written `RMSNorm.forward` baseline
  -- but with the kernel generated from ScalarExp instead of being a
  -- hand-maintained ShaderM.  Reuse for any 1D-row RMSNorm site.
  let circuitRMSNorm := fun (tag : String)
      (norm : RMSNorm.RMSNorm (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
      (inB outB : GPUBackend.Buf β) => do
    let key := hash ("circuitRMSNorm-cuda", tag, norm.config.dim, li)
    let ccRef ← Hesper.Circuit.getGlobalCircuitRef (β := β) key
    Hesper.Circuit.runCachedFused ctx ccRef
      (do
        let xT ← Hesper.Circuit.CircuitM.registerExternal
          (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
          inB #[norm.config.dim] .f32 .Global
        let sT ← Hesper.Circuit.CircuitM.registerExternal
          (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
          norm.scale #[norm.config.dim] .f32 .Global
        let _y ← Hesper.Circuit.CircuitM.rmsNorm xT sT norm.config.eps
        pure ())
      [(0, inB), (1, norm.scale), (5, outB)]

  -- Step 1+2 (combined): Attention pre-norm + Q/K/V projections.
  --
  -- The fused path collapses the standalone RMSNorm dispatch INTO the
  -- Q8_1 quantize step of the QKV pipeline (`forwardFusedNormQKV`).
  -- That eliminates the f32 normedBuf round-trip to VRAM (~10 KB/layer)
  -- AND saves one dispatch per layer (4 → 3).  Preconditions: all
  -- three Q/K/V projections Q4_K + inDim divisible by 256 (for dp4a).
  --
  -- Falls back to the prior 4-dispatch sequence otherwise: standalone
  -- RMSNorm via Circuit DSL, then `forwardFusedQKV` reading normedBuf.
  -- HESPER_FUSION_DISABLE=1 forces every Q4_K matmul through the plain
  -- `forwardDP4A` path so the llama.cpp-PTX override (if installed) is
  -- actually used by all QKV/gate/up/down matmuls — otherwise fused
  -- helpers use hesper's own Q8_1 layout and the override only covers
  -- unfused matmuls, producing a mixed-precision regime that garbles
  -- output.  Turn this on together with HESPER_USE_LLAMACPP_PTX=1.
  let disableFusion := (← IO.getEnv "HESPER_FUSION_DISABLE").isSome
  let useFusedQKV := !disableFusion
                  && cfg.hasKV li
                  && block.attention.wQ.quantFormat == .Q4_K
                  && block.attention.wK.quantFormat == .Q4_K
                  && block.attention.wV.quantFormat == .Q4_K
                  && block.attention.wK.config.inDim == block.attention.wQ.config.inDim
                  && block.attention.wV.config.inDim == block.attention.wQ.config.inDim
                  && block.attention.wK.config.outDim == block.attention.wV.config.outDim
  let useFusedNormQKV := !disableFusion
                      && useFusedQKV
                      && block.attention.wQ.config.inDim == block.attnNorm.config.dim
                      && block.attention.wQ.config.inDim % 256 == 0
  -- Shared-KV layer fast path: RMSNorm fused with the single wQ matmul.
  -- Applies when this layer has no own K/V (cfg.hasKV li = false) but still
  -- needs Q, and the shape constraints for dp4a Q8_1 quantize hold.
  let useFusedNormWQ := !disableFusion
                    && !cfg.hasKV li
                    && block.attention.wQ.quantFormat == .Q4_K
                    && block.attention.wQ.config.inDim == block.attnNorm.config.dim
                    && block.attention.wQ.config.inDim % 256 == 0
  if useFusedNormQKV then
    Hesper.WGSL.Execute.withSection "attnNormQKV" do
      let key := hash ("qkvFusedNormDP4A",
        block.attention.wQ.config.inDim, block.attention.wQ.config.outDim,
        block.attention.wK.config.outDim)
      let kvRef ← match kcr with
        | some k => k.getRef key
        | none => IO.mkRef none
      Linear.forwardFusedNormQKV ctx block.attnNorm
        block.attention.wQ block.attention.wK block.attention.wV
        inputBuf state.qBuf state.kBuf state.vBuf kvRef
  else if useFusedNormWQ then
    Hesper.WGSL.Execute.withSection "attnNormWQ" do
      Linear.forwardFusedNormWQ ctx block.attnNorm block.attention.wQ
        inputBuf state.qBuf
  else do
    -- Standalone attnNorm via Circuit DSL.
    Hesper.WGSL.Execute.withSection "attnNorm" do
      circuitRMSNorm "attnNorm" block.attnNorm inputBuf state.normedBuf
    Hesper.WGSL.Execute.withSection "qkvProj" do
      if cfg.hasKV li then
        if disableFusion then
          -- HESPER_FUSION_DISABLE=1: route every matmul through plain
          -- `LinearLayer.forward` so the llama.cpp-PTX override actually
          -- fires for all of Q, K, V (not just the ones Circuit DSL
          -- happens to emit unfused).  Critical for hybrid correctness
          -- bisection: the Circuit DSL path does NOT go through the
          -- `llamaCppDp4aOverride` hook installed by
          -- `Hesper.Layers.Linear.forwardDP4A`.
          Linear.LinearLayer.forward ctx block.attention.wQ state.normedBuf state.qBuf
          Linear.LinearLayer.forward ctx block.attention.wK state.normedBuf state.kBuf
          Linear.LinearLayer.forward ctx block.attention.wV state.normedBuf state.vBuf
        else if useFusedQKV then
          let key := hash ("qkvFusedDP4A",
            block.attention.wQ.config.inDim, block.attention.wQ.config.outDim,
            block.attention.wK.config.outDim)
          let kvRef ← match kcr with
            | some k => k.getRef key
            | none => IO.mkRef none
          Linear.forwardFusedQKV ctx block.attention.wQ block.attention.wK block.attention.wV
            state.normedBuf state.qBuf state.kBuf state.vBuf kvRef
        else
          -- Circuit-DSL: three Q4_K matmuls sharing one input.  Built once,
          -- then `Hesper.Circuit.runCachedFused` runs the Stage 2
          -- `mergeSameDispatch` pass before lowering.  The pass detects
          -- the [matmul wK; matmul wV] pair (same input + same shape) and
          -- merges them into one fusedKV op, mechanically reproducing what
          -- our hand-written `forwardFusedKV` does.
          let key3 := hash ("circuitQKV-fused-cuda", block.attention.wQ.config.inDim,
                            block.attention.wQ.config.outDim,
                            block.attention.wK.config.outDim, li)
          let ccRef3 ← Hesper.Circuit.getGlobalCircuitRef (β := β) key3
          Hesper.Circuit.runCachedFused ctx ccRef3
            (do
              let normed ← Hesper.Circuit.CircuitM.registerExternal
                (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
                state.normedBuf #[cfg.hiddenSize] .f32 .Global
              let _q ← Hesper.Circuit.CircuitM.matmulQ4K normed block.attention.wQ
              let _k ← Hesper.Circuit.CircuitM.matmulQ4K normed block.attention.wK
              let _v ← Hesper.Circuit.CircuitM.matmulQ4K normed block.attention.wV
              pure ())
            [(0, state.normedBuf), (1, state.qBuf), (2, state.kBuf), (3, state.vBuf)]
      else if disableFusion then
        -- Shared-KV layer under HESPER_FUSION_DISABLE: plain path.
        Linear.LinearLayer.forward ctx block.attention.wQ state.normedBuf state.qBuf
      else
        -- No-KV layer: wQ only, via Circuit DSL.  `runCached` builds
        -- the Circuit ONCE per (inDim, outDim, backend), caches the
        -- compiled artifact, and replays.
        let key := hash ("circuitWQ-cuda", block.attention.wQ.config.inDim, block.attention.wQ.config.outDim, li)
        let ccRef ← Hesper.Circuit.getGlobalCircuitRef (β := β) key
        Hesper.Circuit.runCached ctx ccRef
          (do
            let normed ← Hesper.Circuit.CircuitM.registerExternal
              (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
              state.normedBuf #[cfg.hiddenSize] .f32 .Global
            let _q ← Hesper.Circuit.CircuitM.matmulQ4K normed block.attention.wQ
            pure ())
          [(0, state.normedBuf), (1, state.qBuf)]

  -- Step 3: Q-norm, K-norm (per-head RMSNorm)
  let numHeads := cfg.numAttentionHeads
  let numKVHeads := cfg.numKVHeads li
  let wgSize := min headDim 256
  let mkNormConfig := fun (nHeads : Nat) => ({
    numWorkgroups := (nHeads, 1, 1)
    workgroupSize := { x := wgSize, y := 1, z := 1 }
  } : Hesper.ExecConfig)
  let isFull := cfg.isFullAttention li
  Hesper.WGSL.Execute.withSection "qkvNorm" do
    -- When this layer has its own KV, fuse the three per-head norms
    -- (qNorm, kNorm, vNorm) into a single dispatch.  Grid is
    -- `(numHeads, 3, 1)`; `wg_id.y` picks Q/K/V; WGs with
    -- `wg_id.y > 0 && wg_id.x >= numKVHeads` early-return.  Saves 2
    -- dispatches per layer per token.
    --
    -- When the layer shares KV with an earlier block, only qNorm runs
    -- — keep the existing single-dispatch path for that case.
    if cfg.hasKV li then
      ce (if isFull then "qkvNormFull" else "qkvNormSWA")
        (fusedPerHeadQKVNormKernel numHeads numKVHeads headDim cfg.rmsNormEps)
        [("q_in", state.qBuf), ("q_scale", block.attention.qNormWeight), ("q_out", state.qBuf2),
         ("k_in", state.kBuf), ("k_scale", block.attention.kNormWeight), ("k_out", state.kBuf2),
         ("v_in", state.vBuf),                                              ("v_out", state.vBuf2)]
        { numWorkgroups := (numHeads, 3, 1),
          workgroupSize := { x := wgSize, y := 1, z := 1 } : Hesper.ExecConfig }
    else
      ce (if isFull then "qNormFull" else "qNormSWA")
        (perHeadRMSNormKernel numHeads headDim cfg.rmsNormEps)
        [("input", state.qBuf), ("weight", block.attention.qNormWeight), ("output", state.qBuf2)]
        (mkNormConfig numHeads)

  -- Step 4: RoPE on Q and K
  -- Upload position to params buffer (u32 for hand-coded kernels).
  -- pos is layer-invariant within a single forward pass, so write it
  -- only on the first block.  Previously this ran 42× per token,
  -- adding 42 H2D copies / token on the decode critical path.  For
  -- capture mode (HESPER_CUDA_GRAPHS=1) the graph node records the
  -- pinned source pointer once; the generate loop overwrites the slot
  -- before each replay, so the captured node still reads the current
  -- pos regardless of `li`.
  if li == 0 && !skipPosWrite then
    let posBytes := Hesper.WebGPU.BufferOps.uint32ToBytes pos.toUInt32
    writeScalarViaStaging ctx state.paramsBuf 0 state.stagingParamsPtr 0 posBytes
    -- Also upload pos as f32 for Circuit DSL scatter addrExpr.  Routed
    -- through a pinned host slot so CUDA Graph replay picks up the
    -- current pos (not a stale captured value).
    let posF32Bytes ← Hesper.Basic.floatToBytes pos.toFloat
    writeScalarViaStaging ctx state.posF32Buf 0 state.stagingPosF32Ptr 0 posF32Bytes

  Hesper.WGSL.Execute.withSection "rope" do
    -- RoPE on Q: qBuf2 → qBuf
    match block.ropeFreqFactors with
    | some freqFactors =>
      ce s!"ropeFreqQ_{headDim}_base{cfg.ropeBase li}"
        (ropeWithFreqFactorsKernel headDim numHeads (cfg.ropeBase li))
        [("input", state.qBuf2), ("output", state.qBuf), ("params", state.paramsBuf), ("freq_factors", freqFactors)]
        (.dispatch1D (numHeads * headDim / 2))
    | none =>
      let ropeConfig : RoPE.Config := { dim := numHeads * headDim, maxSeqLen := cfg.maxSeqLen, base := cfg.ropeBase li }
      -- NB: `base` is baked into the shader as a literal — must be in cache key.
      ce s!"ropeDynQ_{headDim}_base{cfg.ropeBase li}"
        (RoPE.ropeKernelDynamic ropeConfig 1 1 numHeads headDim)
        [("input", state.qBuf2), ("output", state.qBuf), ("params", state.paramsBuf)]
        (.dispatch1D (numHeads * headDim / 2))

    -- ropeK is fused with KV cache write below (when ropeFreqFactors are available
    -- and we have a KV cache).  When freq factors aren't present we fall back to
    -- the legacy two-kernel path.
    if cfg.hasKV li && block.ropeFreqFactors.isNone then
      let ropeConfig : RoPE.Config := { dim := numKVHeads * headDim, maxSeqLen := cfg.maxSeqLen, base := cfg.ropeBase li }
      ce s!"ropeDynK_{headDim}_{numKVHeads}_base{cfg.ropeBase li}"
        (RoPE.ropeKernelDynamic ropeConfig 1 1 numKVHeads headDim)
        [("input", state.kBuf2), ("output", state.kBuf), ("params", state.paramsBuf)]
        (.dispatch1D (numKVHeads * headDim / 2))

  -- Step 5: Write K/V to cache and compute flash attention
  -- KV-shared layers reuse an earlier layer's cache (see Config.kvCacheLayer).
  let kvLi := cfg.kvCacheLayer li
  if h : kvLi < state.kvCaches.size then
    let kvCache := state.kvCaches[kvLi]
    let kvDim := numKVHeads * headDim
    let cacheLen := pos + 1  -- number of cached positions including current

    -- Write K and V to cache at current position.  When ropeFreqFactors are
    -- available (Gemma 4 default), we use the fused RoPE-K + KV-write kernel
    -- that takes K *before* RoPE (kBuf2) and applies the rotation in-kernel,
    -- saving the separate ropeK dispatch above.
    if cfg.hasKV li then
      Hesper.WGSL.Execute.withSection "kvWrite" do
        match block.ropeFreqFactors with
        | some freqFactors =>
          -- Single fused RoPE-K + KV-write kernel writing the f16 packed cache.
          -- Uses fusedRopeKAndCacheWriteBatchKernelF16 with seqLen=1 so the
          -- prefill batched path and the decode path share one kernel
          -- definition (different shape only).
          ce s!"ropeKAndKvWriteF16_{headDim}_{numKVHeads}_base{cfg.ropeBase li}"
            (Attention.fusedRopeKAndCacheWriteBatchKernelF16
               numKVHeads cfg.maxSeqLen headDim 1 (cfg.ropeBase li))
            [("new_k", state.kBuf2), ("new_v", state.vBuf2),
             ("k_cache_f16", kvCache.kBufF16), ("v_cache_f16", kvCache.vBufF16),
             ("params", state.paramsBuf), ("freq_factors", freqFactors)]
            (.dispatch1D (numKVHeads * (headDim / 2)))
        | none =>
          -- SWA layers: K is already RoPE'd into state.kBuf upstream (the
          -- `if cfg.hasKV li && block.ropeFreqFactors.isNone` branch around
          -- line 696 emits ropeKernelDynamic into state.kBuf).  Just pack
          -- K + V into the f16 cache.
          ce s!"kvWriteF16NoRope_{headDim}_{numKVHeads}"
            (Attention.kvWriteBatchKernelF16 numKVHeads cfg.maxSeqLen headDim 1)
            [("new_k_roped", state.kBuf), ("new_v", state.vBuf2),
             ("k_cache_f16", kvCache.kBufF16), ("v_cache_f16", kvCache.vBufF16),
             ("params", state.paramsBuf)]
            (.dispatch1D (numKVHeads * (headDim / 2)))

    -- Flash attention: Q @ K_cache^T → softmax → @ V_cache → output
    -- Gemma 4 uses hparams.f_attention_scale = 1.0 (NOT the usual 1/sqrt(headDim)),
    -- because the Q-norm RMSNorm already normalizes each head, so the dot product
    -- magnitudes are bounded without the 1/sqrt(headDim) temperature.
    -- See llama.cpp llama-model.cpp:1272 and gemma4-iswa.cpp:94.
    let scale : Float := 1.0
    -- Write cacheLen to params buffer for FlashAttention (params = [pos, cacheLen]).
    -- Layer-invariant within a single forward pass — write only on first
    -- block (same reasoning as pos above; see task #238).
    if li == 0 && !skipPosWrite then
      let cacheLenBytes := Hesper.WebGPU.BufferOps.uint32ToBytes cacheLen.toUInt32
      writeScalarViaStaging ctx state.paramsBuf 4 state.stagingParamsPtr 4 cacheLenBytes
    Hesper.WGSL.Execute.withSection "flashAttn" do
      -- doc 60 Session 5 (V11): split-K + sub-warp partition + f16 K/V cache.
      -- Two-kernel pipeline (partial → combine).  This is the only decode
      -- FlashAttention path now — legacy f32-cache kernels (Dynamic / Tiled
      -- / Vec / Subgroup) were removed when the cache went f16-only.
      --
      -- SWA masking isn't needed: cacheLen is already clamped to
      -- ≤ windowSize for SWA layers upstream.
      let kcrLk := kcr.map (fun k key => k.getRef key)
      FlashAttention.executeFlashAttentionV11 ctx
        state.qBuf kvCache.kBufF16 kvCache.vBufF16 state.paramsBuf
        state.flashPartialOutV11 state.flashPartialMetaV11 state.attnOutBuf
        numHeads numKVHeads cfg.maxSeqLen headDim scale
        (kcrLookup := kcrLk)

    -- Output projection: attnOut [numHeads * headDim] → normedBuf [hiddenSize]
    -- Circuit-DSL: single matmulQ4K op via runCached (build once, replay).
    -- Equivalent to direct LinearLayer.forward; sets up the IR for later
    -- fusion with the post-attn norm chain.
    Hesper.WGSL.Execute.withSection "oProj" do
      -- HESPER_BYPASS_OPROJ=1 short-circuits the Circuit DSL path for this
      -- section. Used as the H4c A/B test (doc 57 §3b.7): result was
      -- runCached 11.79 µs vs bypass 9.90 µs (-1.9 µs/call only). Kept
      -- as a diagnostic toggle since it's behind an env flag.
      let bypass := (← IO.getEnv "HESPER_BYPASS_OPROJ").isSome
      if disableFusion || bypass then
        -- HESPER_FUSION_DISABLE=1: plain path so llama.cpp override fires.
        Linear.LinearLayer.forward ctx block.attention.wO
          state.attnOutBuf state.normedBuf
      else
        let keyO := hash ("circuitWO-cuda", block.attention.wO.config.inDim,
                          block.attention.wO.config.outDim, li)
        let ccRefO ← Hesper.Circuit.getGlobalCircuitRef (β := β) keyO
        Hesper.Circuit.runCached ctx ccRefO
          (do
            let attnOut ← Hesper.Circuit.CircuitM.registerExternal
              (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
              state.attnOutBuf #[block.attention.wO.config.inDim] .f32 .Global
            let _o ← Hesper.Circuit.CircuitM.matmulQ4K attnOut block.attention.wO
            pure ())
          [(0, state.attnOutBuf), (1, state.normedBuf)]
  else
    -- Fallback: skip attention (shouldn't happen)
    Linear.LinearLayer.forward ctx block.attention.wO state.qBuf state.normedBuf

  -- Step 6: Post-attention norm + residual.  Gemma 4 is post-norm:
  -- `attnResidualBuf = RMSNorm(attn_out) * scale + inputBuf`.  Fused into
  -- a single kernel via forwardNormThenAdd to save one dispatch.
  Hesper.WGSL.Execute.withSection "postAttnNorm" do
    let key := hash ("postAttnNormAdd", cfg.hiddenSize)
    let ref ← match kcr with
      | some k => k.getRef key
      | none => IO.mkRef none
    RMSNorm.forwardNormThenAdd ctx block.postAttnNorm
      state.normedBuf inputBuf state.attnResidualBuf ref
  dumpBuf ctx state.attnResidualBuf (cfg.hiddenSize * 4).toUSize s!"single_p{pos}_postAttn_L{li}"

  -- Step 7: FFN (dense or MoE)
  if block.isMoE then do
    -- MoE layer (from gemma4-iswa.cpp:117-169):
    -- 1. Shared expert: ffn_norm → GeGLU FFN → post_norm_1
    RMSNorm.forward ctx block.ffnNorm state.attnResidualBuf state.normedBuf
    Linear.LinearLayer.forward ctx block.ffn.gate state.normedBuf state.gateBuf
    Linear.LinearLayer.forward ctx block.ffn.up state.normedBuf state.upBuf
    ce "geluMul"
      (geluMulKernel cfg.intermediateSize)
      [("gate", state.gateBuf), ("up", state.upBuf), ("output", state.geluBuf)]
      (.dispatch1D cfg.intermediateSize)
    Linear.LinearLayer.forward ctx block.ffn.down state.geluBuf state.ffnOutBuf

    -- Apply post_norm_1 to shared expert output (avoid aliasing)
    match block.moePostNorm1 with
    | some norm =>
      RMSNorm.forward ctx norm state.ffnOutBuf state.normedBuf2
      -- Copy back: normedBuf2 → ffnOutBuf
      ce "pleScale1"
        (PerLayerEmbedding.scaleKernel cfg.hiddenSize 1.0)
        [("input", state.normedBuf2), ("output", state.ffnOutBuf)]
        (.dispatch1D cfg.hiddenSize)
    | none => pure ()

    -- 2. Router: rms_norm(attn_out) * (1/sqrt(n_embd)) * router_scale → logits → softmax → top-K
    match block.moeRouterWeight, block.moeRouterScale with
    | some routerW, some routerS =>
      ce "moeRouterPre"
        (MoE.routerPreprocessKernel cfg.hiddenSize cfg.rmsNormEps)
        [("input", state.attnResidualBuf), ("router_scale", routerS), ("output", state.moeRouterOutBuf)]
        ({ numWorkgroups := (1, 1, 1) : Hesper.ExecConfig })
      -- Router matmul: moeRouterOutBuf [hiddenSize] @ routerW^T → moeLogitsBuf [numExperts]
      let routerMatmulConfig : Hesper.WGSL.MatMul.Config := {
        M := 1, N := cfg.numExperts, K := cfg.hiddenSize
      }
      Hesper.WGSL.MatMul.executeMatMulTranspose ctx state.moeRouterOutBuf routerW state.moeLogitsBuf routerMatmulConfig
      -- Top-K selection
      ce "moeSoftmaxTopK"
        (MoE.softmaxTopKKernel cfg.numExperts cfg.numExpertsUsed)
        [("logits", state.moeLogitsBuf), ("indices", state.moeIndicesBuf), ("weights", state.moeWeightsBuf)]
        (.dispatch1D 1)
    | _, _ => pure ()

    -- 3. Routed experts: ffn_pre_norm_2 → expert GeGLU FFN → weighted sum
    match block.moeGateUpExps, block.moeDownExps, block.moePreNorm2, block.moePostNorm2 with
    | some gateUpExps, some downExps, some preNorm2, some postNorm2 =>
      -- Pre-norm for routed expert input
      RMSNorm.forward ctx preNorm2 state.attnResidualBuf state.moeNormedBuf

      -- Zero the accumulator
      ce "residAddZero"
        (residualAddKernel cfg.hiddenSize)  -- hack: 0 + 0 = 0 (both inputs are same zeroed buf)
        [("a", state.moeExpertOutBuf), ("b", state.moeExpertOutBuf), ("output", state.moeExpertOutBuf)]
        (.dispatch1D cfg.hiddenSize)
      -- Actually zero it properly
      ce "embedScaleZero"
        (embeddingScaleKernel cfg.hiddenSize 0)  -- scale by 0 to zero
        [("input", state.moeExpertOutBuf), ("output", state.moeExpertOutBuf)]
        (.dispatch1D cfg.hiddenSize)

      let moeConfig : MoE.Config := {
        hiddenSize := cfg.hiddenSize
        expertFFSize := cfg.expertFFSize
        numExperts := cfg.numExperts
        numExpertsUsed := cfg.numExpertsUsed
        rmsNormEps := cfg.rmsNormEps
      }

      -- For each selected expert: gate+up → GELU*up → down → weighted accumulate
      for k in [0:cfg.numExpertsUsed] do
        ce s!"moeGate_{k}"
          (MoE.expertGateUpKernel moeConfig k true)
          [("input", state.moeNormedBuf), ("gate_up_weights", gateUpExps),
           ("expert_indices", state.moeIndicesBuf), ("output", state.moeExpertGateBuf)]
          ({ numWorkgroups := (cfg.expertFFSize, 1, 1) : Hesper.ExecConfig })
        ce s!"moeUp_{k}"
          (MoE.expertGateUpKernel moeConfig k false)
          [("input", state.moeNormedBuf), ("gate_up_weights", gateUpExps),
           ("expert_indices", state.moeIndicesBuf), ("output", state.moeExpertUpBuf)]
          ({ numWorkgroups := (cfg.expertFFSize, 1, 1) : Hesper.ExecConfig })
        ce "moeExpertGelu"
          (MoE.expertGeluMulKernel cfg.expertFFSize)
          [("gate", state.moeExpertGateBuf), ("up", state.moeExpertUpBuf), ("output", state.moeExpertGeluBuf)]
          (.dispatch1D cfg.expertFFSize)
        ce s!"moeDown_{k}"
          (MoE.expertDownKernel moeConfig k)
          [("input", state.moeExpertGeluBuf), ("down_weights", downExps),
           ("expert_indices", state.moeIndicesBuf), ("output", state.moeExpertDownBuf)]
          ({ numWorkgroups := (cfg.hiddenSize, 1, 1) : Hesper.ExecConfig })
        ce s!"moeAccum_{k}"
          (MoE.weightedAccumulateKernel cfg.hiddenSize cfg.numExpertsUsed k)
          [("accumulator", state.moeExpertOutBuf), ("expert_output", state.moeExpertDownBuf),
           ("weights", state.moeWeightsBuf)]
          (.dispatch1D cfg.hiddenSize)

      -- post_norm_2 on routed expert output
      -- Avoid aliasing: moeExpertOutBuf → normedBuf2 → moeExpertOutBuf
      RMSNorm.forward ctx postNorm2 state.moeExpertOutBuf state.normedBuf2
      ce "pleScale2"
        (PerLayerEmbedding.scaleKernel cfg.hiddenSize 1.0)
        [("input", state.normedBuf2), ("output", state.moeExpertOutBuf)]
        (.dispatch1D cfg.hiddenSize)

      -- 4. Combined: shared_expert + routed_experts
      ce "residAddMoePost"
        (residualAddKernel cfg.hiddenSize)
        [("a", state.ffnOutBuf), ("b", state.moeExpertOutBuf), ("output", state.ffnOutBuf)]
        (.dispatch1D cfg.hiddenSize)
    | _, _, _, _ => pure ()  -- No MoE weights: shared expert only

    -- Post-FFN norm + residual, fused: output = RMSNorm(ffn_out) * scale + attn_residual.
    let keyFFN := hash ("postFFNNormAdd", cfg.hiddenSize)
    let refFFN ← match kcr with
      | some k => k.getRef keyFFN
      | none => IO.mkRef none
    RMSNorm.forwardNormThenAdd ctx block.postFFNNorm
      state.ffnOutBuf state.attnResidualBuf outputBuf refFFN
  else do
    -- Dense FFN path (GeGLU).
    -- The fused-norm path collapses ffnNorm + Q8_1 + gate+up into 2
    -- dispatches (vs unfused 3): one fused norm-quantize, one fused
    -- gate+up GeGLU matmul that consumes the Q8_1 buffer.  Eliminates
    -- the f32 normedBuf round-trip AND the standalone ffnNorm dispatch.
    -- A/B confirmed 2026-04-16: fused path is 5.6 TPS faster than the
    -- unfused 2-matmul+geluMul alternative (148 µs/layer vs 122 µs/layer
    -- for the heavy kernel).
    let disableFusion := (← IO.getEnv "HESPER_FUSION_DISABLE").isSome
    let useFused := !disableFusion
                  && block.ffn.gate.quantFormat == .Q4_K
                  && block.ffn.up.quantFormat == .Q4_K
                  && block.ffn.gate.config.inDim == block.ffn.up.config.inDim
                  && block.ffn.gate.config.outDim == block.ffn.up.config.outDim
    let useFusedNorm := !disableFusion
                     && useFused
                     && block.ffn.gate.config.inDim == block.ffnNorm.config.dim
                     && block.ffn.gate.config.inDim % 256 == 0
    if useFusedNorm then
      Hesper.WGSL.Execute.withSection "ffnNormGateUp" do
        let key := hash ("ffnGateUpFusedNormDP4A",
          block.ffn.gate.config.inDim, block.ffn.gate.config.outDim)
        let ref ← match kcr with
          | some k => k.getRef key
          | none => IO.mkRef none
        Linear.forwardFusedNormGateUp ctx block.ffnNorm
          block.ffn.gate block.ffn.up
          state.attnResidualBuf state.geluBuf ref
    else do
      Hesper.WGSL.Execute.withSection "ffnNorm" do
        circuitRMSNorm "ffnNorm" block.ffnNorm state.attnResidualBuf state.normedBuf
      Hesper.WGSL.Execute.withSection "ffnGateUpMul" do
        if useFused then
          let key := hash ("ffnGateUpDP4A", block.ffn.gate.config.inDim, block.ffn.gate.config.outDim)
          let ref ← match kcr with
            | some k => k.getRef key
            | none => IO.mkRef none
          Linear.forwardFusedGateUp ctx block.ffn.gate block.ffn.up
            state.normedBuf state.geluBuf ref
        else
          Linear.LinearLayer.forward ctx block.ffn.gate state.normedBuf state.gateBuf
          Linear.LinearLayer.forward ctx block.ffn.up state.normedBuf state.upBuf
          ce "geluMul2"
            (geluMulKernel cfg.intermediateSize)
            [("gate", state.gateBuf), ("up", state.upBuf), ("output", state.geluBuf)]
            (.dispatch1D cfg.intermediateSize)
    -- ffn.down: gelu*up [intermediateSize] → ffnOut [hiddenSize].  Same
    -- Circuit DSL pattern as wO above.
    Hesper.WGSL.Execute.withSection "ffnDown" do
      if disableFusion then
        Linear.LinearLayer.forward ctx block.ffn.down
          state.geluBuf state.ffnOutBuf
      else
        let keyFD := hash ("circuitFFNDown-cuda", block.ffn.down.config.inDim,
                           block.ffn.down.config.outDim, li)
        let ccRefFD ← Hesper.Circuit.getGlobalCircuitRef (β := β) keyFD
        Hesper.Circuit.runCached ctx ccRefFD
          (do
            let gelu ← Hesper.Circuit.CircuitM.registerExternal
              (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
              state.geluBuf #[block.ffn.down.config.inDim] .f32 .Global
            let _o ← Hesper.Circuit.CircuitM.matmulQ4K gelu block.ffn.down
            pure ())
          [(0, state.geluBuf), (1, state.ffnOutBuf)]

    -- Post-FFN norm + residual, fused.  output = RMSNorm(ffn_out) * scale + attn_residual.
    Hesper.WGSL.Execute.withSection "postFFNNorm" do
      let keyFFN2 := hash ("postFFNNormAdd", cfg.hiddenSize)
      let refFFN2 ← match kcr with
        | some k => k.getRef keyFFN2
        | none => IO.mkRef none
      RMSNorm.forwardNormThenAdd ctx block.postFFNNorm
        state.ffnOutBuf state.attnResidualBuf outputBuf refFFN2
  dumpBuf ctx outputBuf (cfg.hiddenSize * 4).toUSize s!"single_p{pos}_postFFN_L{li}"

  -- Step 8: Per-layer embedding (optional, from gemma4-iswa.cpp:192-213)
  -- pe_in = cur (= outputBuf at this point)
  -- gate = GELU(per_layer_inp_gate @ cur)
  -- cur = gate * per_layer_input[layerIdx]
  -- cur = per_layer_proj @ cur
  -- cur = per_layer_post_norm(cur)
  -- output = pe_in + cur
  Hesper.WGSL.Execute.withSection "perLayerEmbd" do
    match perLayerEmbd, perLayerInput with
    | some plEmbd, some plInputAll =>
      let plOffset := li * cfg.embdPerLayer
      let plTotalSize := cfg.embdPerLayer * cfg.numHiddenLayers
      -- Fuse `ple.inpGate` matmul + `ple.geluGateMul` into one dispatch
      -- pair (Q8_1 quantize + fused matmul-with-GELU-slice-mul epilogue).
      -- Saves 1 dispatch per PLE site.  Falls back to the 2-step path
      -- when preconditions fail.
      let useFusedPLGate :=
        plEmbd.inpGate.quantFormat == .Q4_K &&
        plEmbd.inpGate.config.inDim % 256 == 0
      if useFusedPLGate then
        -- Circuit-DSL: one generic `Prim.matmulQ4KWithEpilogue` node
        -- carries the PLE matmul + GELU + slice-mul tail.  Lowering
        -- emits (Q8_1 quantize dispatch) + (fused matmul-epilogue
        -- kernel) — same two dispatches as the prior hand-composed
        -- `forwardFusedPLInpGate`, but from the IR rather than a
        -- duplicated ShaderM kernel.
        --
        -- Epilogue body: `gelu(input 0) * input 1` where
        --   input 0 = matmul dot product (per-row)
        --   input 1 = per_layer_input[plOffset + outIdx]
        Hesper.WGSL.Execute.withSection "ple.inpGateGeluSlice" do
          let key := hash ("circuitPLEInpGateGeluSlice",
            plEmbd.inpGate.config.inDim, plEmbd.inpGate.config.outDim, plOffset)
          let ccRef ← Hesper.Circuit.getGlobalCircuitRef (β := β) key
          Hesper.Circuit.runCached ctx ccRef
            (do
              let x ← Hesper.Circuit.CircuitM.registerExternal
                (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
                outputBuf #[plEmbd.inpGate.config.inDim] .f32 .Global
              let plAll ← Hesper.Circuit.CircuitM.registerExternal
                (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
                plInputAll #[plTotalSize] .f32 .Global
              -- gelu(input 0) * input 1  via tanh approximation.
              let x0 : Hesper.Circuit.ScalarExp := .input 0
              let x3 := .mul (.mul x0 x0) x0
              let inner :=
                .mul (.const 0.7978845608028654)  -- sqrt(2/π)
                     (.add x0 (.mul (.const 0.044715) x3))
              let gelu :=
                .mul (.mul (.const 0.5) x0)
                     (.add (.const 1.0) (.tanh inner))
              let body : Hesper.Circuit.ScalarExp :=
                .mul gelu (.input 1)
              let _out ← Hesper.Circuit.CircuitM.matmulQ4KWithEpilogue
                x plEmbd.inpGate #[plAll] body (epiReadOffsets := #[plOffset])
              pure ())
            -- Tensor ids: 0 = outputBuf external, 1 = plInputAll external,
            -- 2 = matmul-epi output (caller-facing).
            [(0, outputBuf), (1, plInputAll), (2, state.moeRouterOutBuf)]
      else do
        Hesper.WGSL.Execute.withSection "ple.inpGate" do
          Linear.LinearLayer.forward ctx plEmbd.inpGate outputBuf state.plGateBuf
        Hesper.WGSL.Execute.withSection "ple.geluGateMul" do
          ce s!"pleGeluGateMul_{plOffset}"
            (PerLayerEmbedding.geluGateMulSliceKernel cfg.embdPerLayer plTotalSize plOffset)
            [("gate", state.plGateBuf), ("per_layer_input", plInputAll), ("output", state.moeRouterOutBuf)]
            (.dispatch1D cfg.embdPerLayer)
      -- per_layer_proj @ moeRouterOutBuf → plProjBuf [hiddenSize]
      Hesper.WGSL.Execute.withSection "ple.proj" do
        Linear.LinearLayer.forward ctx plEmbd.proj state.moeRouterOutBuf state.plProjBuf
      -- Fused post-norm + residual-add, in place on outputBuf.
      -- When the block has an outScale, fold the layerOutScale tail
      -- multiply into the same kernel — saves one dispatch per layer.
      let skipOutScaleSingle := (← IO.getEnv "HESPER_SKIP_OUTSCALE").isSome
      let outScaleOpt := if skipOutScaleSingle then none else block.outScale
      Hesper.WGSL.Execute.withSection "ple.postNormAdd" do
        match outScaleOpt with
        | some scaleBuf =>
          ce "fusedPLPostScale"
            (fusedPerLayerPostThenScaleKernel cfg.hiddenSize cfg.rmsNormEps)
            [("proj", state.plProjBuf), ("weight", plEmbd.postNorm.scale),
             ("out_scale", scaleBuf), ("residual", outputBuf)]
            { numWorkgroups := (1, 1, 1)
              workgroupSize := { x := 256, y := 1, z := 1 }
              extensions := ["subgroups"]
              : Hesper.ExecConfig }
        | none =>
          ce "fusedPLPost"
            (fusedPerLayerPostKernel cfg.hiddenSize cfg.rmsNormEps)
            [("proj", state.plProjBuf), ("weight", plEmbd.postNorm.scale), ("residual", outputBuf)]
            { numWorkgroups := (1, 1, 1)
              workgroupSize := { x := 256, y := 1, z := 1 }
              extensions := ["subgroups"]
              : Hesper.ExecConfig }
    | _, _ => pure ()
  dumpBuf ctx outputBuf (cfg.hiddenSize * 4).toUSize s!"single_p{pos}_postPLE_L{li}"

  -- Step 9: Layer output scale — previously its own dispatch, now
  -- folded into `ple.postNormAdd` when PLE runs (fusedPLPostScale
  -- kernel).  Fallback dispatch stays below for blocks that DON'T run
  -- PLE (e.g. non-PLE models).  `block.perLayerBlocks[li]` being None
  -- means this layer had no PLE — use the standalone layerOutScale.
  let skipOutScaleFallback := (← IO.getEnv "HESPER_SKIP_OUTSCALE").isSome
  let pleRan := match perLayerEmbd with | some _ => true | none => false
  match if skipOutScaleFallback || pleRan then none else block.outScale with
  | some scale =>
    Hesper.WGSL.Execute.withSection "layerOutScale" do
      let key := hash ("circuitLayerOutScale-cuda", cfg.hiddenSize, li)
      let ccRef ← Hesper.Circuit.getGlobalCircuitRef (β := β) key
      Hesper.Circuit.runCachedFused ctx ccRef
        (do
          let x ← Hesper.Circuit.CircuitM.registerExternal
                    (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
                    outputBuf #[cfg.hiddenSize] .f32 .Global
          let s ← Hesper.Circuit.CircuitM.registerExternal
                    (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
                    scale #[1] .f32 .Global
          let scaled ← Hesper.Circuit.CircuitM.scaleByBroadcast x s
          let _out   ← Hesper.Circuit.CircuitM.map scaled
                         (.mul (.input 0) (.const 1.0))
          pure ())
        -- ids: 0=x (outputBuf), 1=s (scale), 2=scaled (fused away),
        -- 3=final (written back to outputBuf).
        [(0, outputBuf), (1, scale), (3, outputBuf)]
  | none => pure ()

/-! ## Column-major helper kernels for batched prefill -/

/-- GPU-side u32 copy: `dst[dstIdx] = src[params[0]]`.
    srcIdx is read at runtime from `params[0]`.  dstIdx is compile-time
    (always 0 or 1 — only 2 unique kernels per (srcSize, dstSize, dstIdx)). -/
private def copyU32Kernel (srcSize : Nat) (dstSize : Nat) (dstIdx : Nat) : ShaderM Unit := do
  let _src    ← ShaderM.declareInputBuffer "src" (.array (.scalar .u32) srcSize)
  let _params ← ShaderM.declareInputBuffer "params" (.array (.scalar .u32) 1)
  let _dst    ← ShaderM.declareOutputBuffer "dst" (.array (.scalar .u32) dstSize)
  let srcIdx ← ShaderM.readBuffer (ty := .scalar .u32) (n := 1) "params" (Exp.litU32 0)
  let v ← ShaderM.readBuffer (ty := .scalar .u32) (n := srcSize) "src" srcIdx
  ShaderM.writeBuffer (ty := .scalar .u32) "dst" (Exp.litU32 dstIdx) v

/-- Copy column from a column-major batch buffer into a contiguous single-row
    buffer.  Column index is read at runtime from `params[0]` (u32).
    `batch[params[0] * dim + i] → out[i]` for i in [0, dim).
    One kernel JIT'd per (dim, seqLen) pair; colIdx changes via params only. -/
private def columnExtractKernel (dim : Nat) (seqLen : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let i := Exp.vec3X gid
  let totalBatch := dim * seqLen
  let _batch  ← ShaderM.declareInputBuffer "batch" (.array (.scalar .f32) totalBatch)
  let _params ← ShaderM.declareInputBuffer "params" (.array (.scalar .u32) 1)
  let _out    ← ShaderM.declareOutputBuffer "out" (.array (.scalar .f32) dim)
  ShaderM.if_ (Exp.lt i (Exp.litU32 dim)) (do
    let colIdx ← ShaderM.readBuffer (ty := .scalar .u32) (n := 1) "params" (Exp.litU32 0)
    let srcIdx := Exp.add (Exp.mul colIdx (Exp.litU32 dim)) i
    let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalBatch) "batch" srcIdx
    ShaderM.writeBuffer (ty := .scalar .f32) "out" i v
  ) (pure ())

/-- Copy a contiguous single-row buffer into a column of a column-major batch
    buffer.  Column index is read at runtime from `params[0]` (u32).
    `src[i] → batch[params[0] * dim + i]` for i in [0, dim). -/
private def columnInsertKernel (dim : Nat) (seqLen : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let i := Exp.vec3X gid
  let totalBatch := dim * seqLen
  let _src    ← ShaderM.declareInputBuffer "src" (.array (.scalar .f32) dim)
  let _params ← ShaderM.declareInputBuffer "params" (.array (.scalar .u32) 1)
  let _batch  ← ShaderM.declareOutputBuffer "batch" (.array (.scalar .f32) totalBatch)
  ShaderM.if_ (Exp.lt i (Exp.litU32 dim)) (do
    let colIdx ← ShaderM.readBuffer (ty := .scalar .u32) (n := 1) "params" (Exp.litU32 0)
    let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := dim) "src" i
    let dstIdx := Exp.add (Exp.mul colIdx (Exp.litU32 dim)) i
    ShaderM.writeBuffer (ty := .scalar .f32) "batch" dstIdx v
  ) (pure ())

/-! ## Batched Prefill -/

/-- Process all prompt tokens through the model in batch.
    Uses `forwardBatchDP4A` for Q4_K matmuls and `RMSNorm.forward` with
    `numRows` for batch RMSNorm.  Attention remains per-token (extract
    column, run single-token attention, write KV cache).

    Populates the KV caches for all prompt positions and leaves the last
    token's logits in `state.logitsBuf` so that decode can continue from
    position `promptTokens.size`. -/
def forwardPrefillBatch [GPUBackend β] (ctx : β)
    (model : Gemma4Model (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (promptTokens : Array Nat)
    (state : InferenceState (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (kcr : Option (KernelCacheRefs (GPUBackend.CachedDispatch β)) := none)
    (startPos : Nat := 0) : IO Unit := do
  let seqLen := promptTokens.size
  if (← IO.getEnv "HESPER_PREFILL_TRACE").isSome then
    IO.println s!"[prefillBatch] startPos={startPos} seqLen={seqLen} tokens={promptTokens.toList}"
  if seqLen == 0 then return
  -- `startPos` > 0 enables using this function for continuation / decode:
  -- the N tokens in `promptTokens` are written to KV cache slots
  -- `[startPos, startPos+1, ..., startPos+N-1]` and their rotary positions
  -- are `startPos + i`.  For decode use `promptTokens := #[nextToken]`
  -- and `startPos := pos` (N=1 case matches llama.cpp's shape-polymorphic
  -- llama_decode behaviour).

  let cfg := model.config
  let dim := cfg.hiddenSize
  let interSize := cfg.intermediateSize
  let mkBuf := fun (n : Nat) => GPUBackend.allocBuffer ctx (n * 4).toUSize

  -- Cached execute helper (same pattern as forwardBlock / forwardSingleToken).
  let ce := fun (name : String) (shader : ShaderM Unit)
      (namedBufs : List (String × GPUBackend.Buf β)) (config : Hesper.ExecConfig) => do
    match kcr with
    | some k =>
      let key := hash ("gemma4_prefill_ce", name, config.numWorkgroups,
                        config.workgroupSize.x, config.workgroupSize.y, config.workgroupSize.z)
      let ref ← k.getRef key
      let configNamed : Hesper.ExecConfig := { config with funcName := name }
      GPUBackend.executeWithConfigCached ctx shader namedBufs configNamed key ref
    | none => GPUBackend.execute ctx shader namedBufs config

  -- CUDA-Graph-safe write of a u32 into a small device buffer.  In
  -- capture mode the `writeBufferOffset` path would record a memcpy
  -- with the Lean `ByteArray`'s host pointer — moved by GC between
  -- captures.  `writeScalarViaStaging` routes through the pinned-host
  -- slot, giving the graph a stable source address.  Outside capture
  -- it falls back to the normal path.
  --
  -- Single `stagingColIdxPtr` is OK here because:
  --   * The colIdxBuf write sites reachable during unified decode
  --     (seqLen=1, `handledByBatched = true`) all store the SAME
  --     value, `0`, so multiple recorded memcpy nodes sharing the
  --     slot read the same scalar.
  --   * The paramsBuf write (startPos) uses a separate pinned slot
  --     via `state.stagingParamsPtr`.
  let writeColIdxU32 := fun (buf : GPUBackend.Buf β) (v : Nat) => do
    let bytes := Hesper.WebGPU.BufferOps.uint32ToBytes v.toUInt32
    writeScalarViaStaging ctx buf 0 state.stagingColIdxPtr 0 bytes
  let writeParamsU32At := fun (offset : USize) (v : Nat) => do
    let bytes := Hesper.WebGPU.BufferOps.uint32ToBytes v.toUInt32
    writeScalarViaStaging ctx state.paramsBuf offset
      state.stagingParamsPtr offset bytes

  -- ── Allocate prefill-sized batch buffers (column-major) ──────────────
  let batchBuf1 ← mkBuf (dim * seqLen)       -- ping-pong A
  let batchBuf2 ← mkBuf (dim * seqLen)       -- ping-pong B
  let batchNormedBuf ← mkBuf (dim * seqLen)  -- after attnNorm / ffnNorm
  let maxHeadDim := max cfg.headDimFull cfg.headDimSWA
  let maxQDim := cfg.numAttentionHeads * maxHeadDim
  let maxKVDim := (max cfg.numKeyValueHeadsFull cfg.numKeyValueHeadsSWA) * maxHeadDim
  let batchQBuf ← mkBuf (maxQDim * seqLen)
  let batchQRopedBuf ← mkBuf (maxQDim * seqLen)
  let batchKBuf ← mkBuf (maxKVDim * seqLen)
  let batchVBuf ← mkBuf (maxKVDim * seqLen)
  let batchAttnOutBuf ← mkBuf (maxQDim * seqLen)
  let batchOProjBuf ← mkBuf (dim * seqLen)
  let batchAttnResidBuf ← mkBuf (dim * seqLen)
  let batchGateBuf ← mkBuf (interSize * seqLen)
  let batchUpBuf ← mkBuf (interSize * seqLen)
  let batchGeluBuf ← mkBuf (interSize * seqLen)
  let batchFFNOutBuf ← mkBuf (dim * seqLen)
  -- HESPER_ZERO_BATCH=1: zero-init all batch buffers via write of zeros.
  -- Used to rule out uninitialised-memory bugs in unified decode.
  if (← IO.getEnv "HESPER_ZERO_BATCH").isSome then
    let zero4 : ByteArray := ByteArray.mk #[0,0,0,0]
    -- Only write 4B to trigger the path; full zero-init is unnecessary
    -- because every kernel writes every element it reads.  If ZERO_BATCH
    -- FIXES the bug, that means an unexpected kernel IS reading uninit mem.
    -- For now, leave as a no-op marker; full memset wiring requires
    -- extending GPUBackend.  Skipping.
    let _ := zero4
  -- Scaled embedding cache: PLE input uses the embedding (not per-layer output)
  let batchScaledEmbdBuf ← mkBuf (dim * seqLen)
  -- PLE batched scratch: inpGate output, gelu*gate*slice output, proj output.
  let plGateBatchBuf ← mkBuf (cfg.embdPerLayer * seqLen)
  let plMoeOutBatchBuf ← mkBuf (cfg.embdPerLayer * seqLen)
  let plProjBatchBuf ← mkBuf (dim * seqLen)
  -- colIdxBuf is a 4-byte u32 index holder used across ~20 call sites;
  -- pool it on `state` so it's allocated once at first prefill and then
  -- reused for every subsequent call.
  let colIdxBuf ← do
    match ← state.prefillColIdxRef.get with
    | some b => pure b
    | none =>
        let b ← GPUBackend.allocBuffer ctx (4 : USize)
        state.prefillColIdxRef.set (some b)
        pure b

  -- Debug dumps (per-token column extracts + disk writes) are extremely
  -- expensive: 5 loops × 9 tokens × 42 blocks ≈ 1540 wasted dispatches
  -- per prefill when `HESPER_DUMP_DIR` is unset.  Skip the extract+dump
  -- chain entirely in the common case.
  let dumpEnabled := (← IO.getEnv "HESPER_DUMP_DIR").isSome

  -- Attention-path bit-parity harness (Phase 2 item 2 step 1):
  --   HESPER_ATTN_DUMP=<tag>       → dump batchAttnOutBuf after attention,
  --                                   file name `attn_L{li}_<tag>.bin`.
  --   HESPER_FORCE_FALLBACK=1      → force the per-token loop even on
  --                                   layers that would normally take the
  --                                   batched fast path.  Combined with
  --                                   HESPER_ATTN_DUMP=fallback, lets us
  --                                   dump the fallback output for a
  --                                   full-attn layer, then re-run without
  --                                   the flag + HESPER_ATTN_DUMP=batched
  --                                   and diff the two.
  let attnDumpTag ← IO.getEnv "HESPER_ATTN_DUMP"
  let forceFallback := (← IO.getEnv "HESPER_FORCE_FALLBACK").isSome
  let kvDumpDir ← IO.getEnv "HESPER_KVCACHE_DUMP_DIR"
  -- Golden-value harness: dump named intermediate tensors with the SAME
  -- names as llama.cpp's eval-callback dump (`llama.cpp/common/debug.cpp`),
  -- so a simple filename-keyed diff localises the first divergence.
  let goldenDumpDir ← IO.getEnv "HESPER_GOLDEN_DUMP_DIR"
  let dumpGolden : String → GPUBackend.Buf β → Nat → IO Unit :=
    fun name buf nFloats => do
      match goldenDumpDir with
      | some dir =>
        GPUBackend.endBatch ctx
        let data ← GPUBackend.readBuffer ctx buf (nFloats * 4).toUSize
        IO.FS.writeBinFile s!"{dir}/{name}.bin" data
      | none => pure ()
  let stageDumpLayer : Option Nat := (← IO.getEnv "HESPER_ATTN_STAGE_LAYER").bind String.toNat?
  -- Helper: dump `nFloats` f32 elements from `buf` starting at offset 0
  -- to `$HESPER_ATTN_DUMP_DIR/<name>_<tag>.bin` (where <tag> = attnDumpTag).
  -- Active only when both HESPER_ATTN_DUMP and HESPER_ATTN_STAGE_LAYER are
  -- set, and the caller passes `liActive=true`.  Used to localise the
  -- per-stage divergence between batched and fallback attention paths.
  let dumpStage : String → GPUBackend.Buf β → Nat → Bool → IO Unit :=
    fun name buf nFloats liActive => do
      match attnDumpTag with
      | some tag =>
        if liActive then
          let dir := (← IO.getEnv "HESPER_ATTN_DUMP_DIR").getD "/tmp"
          GPUBackend.endBatch ctx
          let data ← GPUBackend.readBuffer ctx buf (nFloats * 4).toUSize
          IO.FS.writeBinFile s!"{dir}/{name}_{tag}.bin" data
      | none => pure ()

  -- NOTE: no beginBatch here — each dispatch fires immediately.
  -- Batching would defer all launches until endBatch, but the per-token
  -- attention loop reads batch matmul outputs mid-stream, requiring
  -- them to be complete.  Individual kernel launches on the default
  -- stream are serialized by CUDA, so this is correct.

  -- ── Pre-upload: token IDs and positions to GPU ──────────────────────
  -- Upload all token IDs and position indices to GPU buffers BEFORE any
  -- kernel dispatch.  This eliminates per-token host→device transfers
  -- inside the per-token attention loop, enabling batch dispatch.
  --
  -- Pool the three small u32 buffers on `state` so they are not
  -- reallocated every call — saves 3 × (cudaMalloc + cudaFree) per
  -- decode token.
  let needBytes : USize := (seqLen * 4).toUSize
  let ensureBuf (ref : IO.Ref (Option (GPUBackend.Buf β × USize)))
      : IO (GPUBackend.Buf β) := do
    match ← ref.get with
    | some (b, cap) =>
        if cap >= needBytes then pure b
        else
          GPUBackend.freeBuffer ctx b
          let b' ← GPUBackend.allocBuffer ctx needBytes
          ref.set (some (b', needBytes))
          pure b'
    | none =>
        let b ← GPUBackend.allocBuffer ctx needBytes
        ref.set (some (b, needBytes))
        pure b
  let tokenIdsBuf ← ensureBuf state.prefillTokenIdsRef
  let posBuf      ← ensureBuf state.prefillPosRef
  let cacheLenBuf ← ensureBuf state.prefillCacheLenRef
  let mut tokBytes : ByteArray := ByteArray.empty
  let mut posBytes : ByteArray := ByteArray.empty
  for i in [0:seqLen] do
    let tokenId := promptTokens[i]!
    tokBytes := tokBytes ++ Hesper.WebGPU.BufferOps.uint32ToBytes tokenId.toUInt32
    -- Absolute rotary position of this token.
    posBytes := posBytes ++ Hesper.WebGPU.BufferOps.uint32ToBytes (startPos + i).toUInt32
  -- cacheLenBuf[i] = startPos + i + 1 (number of KV entries after this token's write)
  let mut clBytes : ByteArray := ByteArray.empty
  for i in [0:seqLen] do
    clBytes := clBytes ++ Hesper.WebGPU.BufferOps.uint32ToBytes (startPos + i + 1).toUInt32
  -- For seqLen=1 (unified decode path), route through pinned+async via
  -- `writeScalarViaStaging` so the copies land on the unified stream
  -- alongside subsequent kernel launches.  For seqLen>1 (prefill) this
  -- is called only a handful of times, so the sync path is acceptable.
  if seqLen == 1 then
    writeScalarViaStaging ctx tokenIdsBuf 0 state.stagingTokenPtr  0 tokBytes
    writeScalarViaStaging ctx posBuf      0 state.stagingParamsPtr 0 posBytes
    writeScalarViaStaging ctx cacheLenBuf 0 state.stagingParamsPtr 4 clBytes
  else
    GPUBackend.writeBuffer ctx tokenIdsBuf tokBytes
    GPUBackend.writeBuffer ctx posBuf posBytes
    GPUBackend.writeBuffer ctx cacheLenBuf clBytes

  -- ── Step 1: Embedding lookup — per token into batch buffer ──────────
  for i in [0:seqLen] do
    -- GPU-side: copy tokenIdsBuf[i] → state.tokenBuf[0]
    writeColIdxU32 colIdxBuf i
    ce s!"copyU32FromTokIds_sl{seqLen}"
      (copyU32Kernel seqLen 1 0)
      [("src", tokenIdsBuf), ("params", colIdxBuf), ("dst", state.tokenBuf)]
      { numWorkgroups := (1, 1, 1), workgroupSize := { x := 1, y := 1, z := 1 } }
    match model.embdFormat with
    | .Q6_K =>
      ce "q6kEmbLookup"
        (Hesper.Quantization.Q6_K.q6kEmbeddingLookupKernel model.config.vocabSize dim)
        [("token_ids", state.tokenBuf), ("embedding_table", model.embedding.embeddingTable), ("output", state.buf1)]
        (.dispatch1D dim)
    | _ =>
      Embedding.forward ctx model.embedding state.tokenBuf state.buf1 1 1
    -- Copy state.buf1 → batchBuf1 column i  (same i, no re-write needed —
    -- colIdxBuf already holds `i` from above).
    ce s!"colInsEmbd_sl{seqLen}"
      (columnInsertKernel dim seqLen)
      [("src", state.buf1), ("params", colIdxBuf), ("batch", batchBuf1)]
      (.dispatch1D dim)

  -- ── Step 1b: Scale embeddings by sqrt(hiddenSize) — batch-wide ──────
  -- embeddingScaleKernel takes a `size` param — we pass dim * seqLen so
  -- it covers all columns in one dispatch.
  let totalHidden := dim * seqLen
  ce "embedScaleBatch"
    (embeddingScaleKernel totalHidden dim)
    [("input", batchBuf1), ("output", batchBuf2)]
    (.dispatch1D totalHidden)
  -- Golden dump: matches llama.cpp's `inp_scaled` tensor.
  dumpGolden "inp_scaled" batchBuf2 totalHidden

  -- Cache the scaled embedding (pre-layer state) for PLE usage inside the block loop.
  -- Single-token path precomputes plInputAll ONCE from the scaled embedding and reuses
  -- across layers; batch path recomputes per token per layer, and MUST use the scaled
  -- embedding (not the current layer output) as the PLE matmul input.
  do
    let totalScaled := dim * seqLen
    let shader : ShaderM Unit := do
      let gid ← ShaderM.globalId
      let i := Exp.vec3X gid
      let _src ← ShaderM.declareInputBuffer "src" (.array (.scalar .f32) totalScaled)
      let _dst ← ShaderM.declareOutputBuffer "dst" (.array (.scalar .f32) totalScaled)
      ShaderM.if_ (Exp.lt i (Exp.litU32 totalScaled)) (do
        let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalScaled) "src" i
        ShaderM.writeBuffer (ty := .scalar .f32) "dst" i v) (pure ())
    ce "batchScaledEmbdCopy" shader
      [("src", batchBuf2), ("dst", batchScaledEmbdBuf)]
      (.dispatch1D totalScaled)

  -- Step 1b: Per-layer input precomputation (BATCHED across all prompt
  -- tokens).  Gemma 4 E4B's per_layer_token_embd needs a plInputAll
  -- vector per token; the value depends only on tokenId (via the
  -- scaled embedding) and NOT on the layer index, so it's safe to
  -- compute once per token before the block loop and reuse for all
  -- 42 layers.  Previously this loop ran per-token-per-layer inside
  -- the block loop (42× redundant recompute).  Now the result is
  -- stored in a `batchPLInputAll : [seqLen × totalPL]` scratch
  -- buffer, and `forwardBlock` reads column `i` from it per token.
  let mut batchPLInputAllOpt : Option (GPUBackend.Buf β) := none
  match model.perLayerEmbdTableGPU, model.perLayerModelProj, model.perLayerProjNorm with
  | some embdTableGPU, some modelProj, some projNorm =>
    let embdPL := model.config.embdPerLayer
    let nLayers := model.config.numHiddenLayers
    let totalPL := embdPL * nLayers
    -- Pooled batched scratch: one totalPL-vector per prompt token.
    -- Re-allocate only when seqLen grows beyond the cached capacity.
    let need := (seqLen * totalPL * 4).toUSize
    let batchPLInputAll ← do
      match ← state.prefillPLInputAllRef.get with
      | some (b, cap) =>
          if cap >= need then pure b
          else
            GPUBackend.freeBuffer ctx b
            let b' ← GPUBackend.allocBuffer ctx need
            state.prefillPLInputAllRef.set (some (b', need))
            pure b'
      | none =>
          let b ← GPUBackend.allocBuffer ctx need
          state.prefillPLInputAllRef.set (some (b, need))
          pure b
    batchPLInputAllOpt := some batchPLInputAll
    let rowBytesU : USize := model.perLayerEmbdRowBytes.toUSize
    for i in [0:seqLen] do
      let tokenId := promptTokens[i]!
      -- UVA on-demand path (HESPER_USE_MMAP=1): table lives in CPU mmap,
      -- pinned + mapped into device VA at load time.  We bypass the
      -- Loader-allocated 1-row scratch buffer entirely: the kernel reads
      -- the row directly from host memory through the unified pointer
      -- (synthesised as a `Buf` at offset = tokenId × rowBytes), which
      -- means kernelTokenId is always 0 and no cuMemcpy fires.
      -- Legacy path: real `tokenId` indexes the full VRAM table.
      let kernelTokenId : Nat := match model.perLayerEmbdMmap with
        | some _ => 0
        | none   => tokenId
      writeScalarViaStaging ctx state.plRawRowBuf 0 state.stagingPLRowPtr 0
        (Hesper.WebGPU.BufferOps.uint32ToBytes kernelTokenId.toUInt32)
      let tableForKernel ← match model.perLayerEmbdMmap with
        | some (_, _, _, devPtr) =>
          let rowDevPtr : USize := devPtr + tokenId.toUSize * rowBytesU
          match ← GPUBackend.bufFromRawDevicePtr ctx rowDevPtr rowBytesU with
          | some b => pure b
          | none   => pure embdTableGPU  -- backend without UVA: fallback
        | none => pure embdTableGPU
      let scaleFactor : Float := Float.sqrt embdPL.toFloat
      ce "q6kDequantScale_pf"
        (Hesper.Quantization.Q6_K.q6kTableRowDequantScaleKernel totalPL scaleFactor
          cfg.vocabSize)
        [("table", tableForKernel), ("params", state.plRawRowBuf), ("output", state.plModelProj)]
        (.dispatch1D totalPL)
      writeColIdxU32 colIdxBuf i
      ce s!"colExtrScaledEmb_sl{seqLen}"
        (columnExtractKernel dim seqLen)
        [("batch", batchBuf2), ("params", colIdxBuf), ("out", state.buf1)]
        (.dispatch1D dim)
      let projConfig : Hesper.WGSL.MatMul.Config := {
        M := 1, N := totalPL, K := dim
      }
      if projConfig.K % 64 == 0 then
        Hesper.WGSL.MatMul.executeMatMulTransposeF16BlockCoop ctx state.buf1 modelProj state.plTokenSelected projConfig
      else
        Hesper.WGSL.MatMul.executeMatMulTransposeF16 ctx state.buf1 modelProj state.plTokenSelected projConfig
      ce "pleScalePL_pf"
        (PerLayerEmbedding.scaleKernel totalPL (1.0 / Float.sqrt dim.toFloat))
        [("input", state.plTokenSelected), ("output", state.plInputAll)]
        (.dispatch1D totalPL)
      ce "chunkedRMSNorm_pf"
        (chunkedRMSNormKernel embdPL nLayers model.config.rmsNormEps)
        [("input", state.plInputAll), ("weight", projNorm.scale), ("output", state.plTokenSelected)]
        { numWorkgroups := (nLayers, 1, 1), workgroupSize := { x := min embdPL 256, y := 1, z := 1 } : Hesper.ExecConfig }
      ce "scaledAdd_pf"
        (scaledAddKernel totalPL (1.0 / Float.sqrt 2.0))
        [("a", state.plTokenSelected), ("b", state.plModelProj), ("output", state.plInputAll)]
        (.dispatch1D totalPL)
      -- Copy the per-token plInputAll into column `i` of the batched
      -- buffer via a params-indexed kernel so the PTX cache key does
      -- NOT embed `i` (otherwise every iteration is a JIT miss).
      writeColIdxU32 colIdxBuf i
      let copyShader : ShaderM Unit := do
        let gid ← ShaderM.globalId
        let k := Exp.vec3X gid
        let _src ← ShaderM.declareInputBuffer "src" (.array (.scalar .f32) totalPL)
        let _params ← ShaderM.declareInputBuffer "params" (.array (.scalar .u32) 1)
        let _dst ← ShaderM.declareOutputBuffer "dst" (.array (.scalar .f32) (seqLen * totalPL))
        ShaderM.if_ (Exp.lt k (Exp.litU32 totalPL)) (do
          let colId ← ShaderM.readBuffer (ty := .scalar .u32) (n := 1) "params" (Exp.litU32 0)
          let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalPL) "src" k
          let dstIdx := Exp.add (Exp.mul colId (Exp.litU32 totalPL)) k
          ShaderM.writeBuffer (ty := .scalar .f32) "dst" dstIdx v) (pure ())
      ce "batchPLInputAllCopy" copyShader
        [("src", state.plInputAll), ("params", colIdxBuf), ("dst", batchPLInputAll)]
        (.dispatch1D totalPL)
    -- After all tokens populated: dump the whole tensor for golden comparison
    dumpGolden s!"inp_per_layer" batchPLInputAll (seqLen * totalPL)
  | _, _, _ => pure ()

  -- ── Step 2: Process transformer blocks ──────────────────────────────
  let mut currentBuf := batchBuf2
  let mut nextBuf := batchBuf1

  -- Dump post-PLE state for each token (extract each column to state.buf1 and dump)
  if dumpEnabled then for i in [0:seqLen] do
    let idxBytes := Hesper.WebGPU.BufferOps.uint32ToBytes i.toUInt32
    GPUBackend.writeBufferOffset ctx colIdxBuf 0 idxBytes
    GPUBackend.execute ctx
      (columnExtractKernel dim seqLen)
      [("batch", currentBuf), ("params", colIdxBuf), ("out", state.buf1)]
      (.dispatch1D dim)
    dumpBuf ctx state.buf1 (dim * 4).toUSize s!"batch_t{i}_postPLE"

  let nBlocksToRun := match ← IO.getEnv "HESPER_PREFILL_LAYERS" with
    | some s => s.toNat!
    | none => model.blocks.size
  -- Phase 2 item 2 Step 3: route SWA layers through the batched RoPE
  -- path by feeding a constant-1.0 freq_factors buffer — equivalent to
  -- the single-token `ropeKernelDynamic` (no freq-factor division).
  -- Verified bit-identical vs the per-token fallback for all 42 layers
  -- on Gemma 4 E4B Q4_K_M (commits c92e377 + caa5c7d).  Can be disabled
  -- with HESPER_UNIFY_ATTN=off for A/B testing.
  let unifyAttnSwa := (← IO.getEnv "HESPER_UNIFY_ATTN").all (· ≠ "off")
  let headDim0 := cfg.headDim 0
  let dimPairs := headDim0 / 2
  let onesBuf ← GPUBackend.allocBuffer ctx (dimPairs * 4).toUSize
  (do
    let oneBytes ← Hesper.WebGPU.BufferOps.floatToBytes 1.0
    let mut onesBytes : ByteArray := ByteArray.empty
    for _ in [0:dimPairs] do
      onesBytes := onesBytes ++ oneBytes
    GPUBackend.writeBuffer ctx onesBuf onesBytes)

  -- Architecture diagnostic: set HESPER_LAYER_PROFILE=1 to print layer mix.
  if (← IO.getEnv "HESPER_LAYER_PROFILE").isSome then
    let mut full := 0
    let mut swa := 0
    let mut shared := 0
    for b in model.blocks do
      if cfg.isFullAttention b.layerIdx then full := full + 1 else swa := swa + 1
      if !cfg.hasKV b.layerIdx then shared := shared + 1
    IO.println s!"[LayerProfile] total={model.blocks.size} full={full} swa={swa} kvShared={shared} ropeFreq_layers={(model.blocks.filter (fun b => b.ropeFreqFactors.isSome)).size}"
  for block in model.blocks.extract 0 nBlocksToRun do
    let li := block.layerIdx
    let headDim := cfg.headDim li
    let numHeads := cfg.numAttentionHeads
    let numKVHeads := cfg.numKVHeads li
    let kvDim := numKVHeads * headDim
    let qDim := numHeads * headDim

    -- ── 2a: RMSNorm + Q/K/V projections (batch) ────────────────────────
    -- Fast path (all Q4_K): fused RMSNorm+Q8_1 quantize → Q8_1 batch matmul.
    -- Fallback (any Q6_K in Q/K/V): standalone RMSNorm → f32 batch matmul.
    let nQ8Blocks := dim / 32
    let allQ4K := block.attention.wQ.quantFormat == .Q4_K
                && (!cfg.hasKV li ||
                    (block.attention.wK.quantFormat == .Q4_K
                     && block.attention.wV.quantFormat == .Q4_K))
    if allQ4K then
      let batchQ8Bytes : USize := (nQ8Blocks * 9 * seqLen * 4).toUSize
      -- Pooled Q8_1 scratch: reuse an existing state buffer if it's large
      -- enough, otherwise free and re-alloc.  Saves ~84 cudaMalloc/free per
      -- decode token (2 × 42 layers).  Pool lifetime: until InferenceState
      -- is dropped (no explicit free here).
      let batchQ8Buf ← do
        match ← state.prefillAttnQ8BufRef.get with
        | some (b, sz) => if sz >= batchQ8Bytes then pure b
                          else
                            GPUBackend.freeBuffer ctx b
                            let b' ← GPUBackend.allocBuffer ctx batchQ8Bytes
                            state.prefillAttnQ8BufRef.set (some (b', batchQ8Bytes))
                            pure b'
        | none =>
          let b ← GPUBackend.allocBuffer ctx batchQ8Bytes
          state.prefillAttnQ8BufRef.set (some (b, batchQ8Bytes))
          pure b
      ce s!"attnNormQ8_1_sl{seqLen}_d{dim}"
        (RMSNorm.fusedRMSNormQ8_1Kernel block.attnNorm.config seqLen 256)
        [("input", currentBuf), ("scale", block.attnNorm.scale), ("output", batchQ8Buf)]
        { workgroupSize := { x := 256 }, numWorkgroups := (seqLen, 1, 1) }
      -- kcr-scoped per-layer refs so prefill (seqLen=N) and unified-decode
      -- (seqLen=1) each get their own cached dispatch and don't collide.
      let wQKey := hash ("q4k-batch-matmul-q8-attn", li, "wQ",
        block.attention.wQ.config.inDim, block.attention.wQ.config.outDim, seqLen)
      let wQRef ← match kcr with
        | some k => k.getRef wQKey |>.map some
        | none   => pure none
      Linear.forwardBatchDP4A_fromQ8 ctx block.attention.wQ batchQ8Buf batchQBuf seqLen
        (refOverride := wQRef)
      if cfg.hasKV li then
        let wKKey := hash ("q4k-batch-matmul-q8-attn", li, "wK",
          block.attention.wK.config.inDim, block.attention.wK.config.outDim, seqLen)
        let wVKey := hash ("q4k-batch-matmul-q8-attn", li, "wV",
          block.attention.wV.config.inDim, block.attention.wV.config.outDim, seqLen)
        let wKRef ← match kcr with
          | some k => k.getRef wKKey |>.map some
          | none   => pure none
        let wVRef ← match kcr with
          | some k => k.getRef wVKey |>.map some
          | none   => pure none
        Linear.forwardBatchDP4A_fromQ8 ctx block.attention.wK batchQ8Buf batchKBuf seqLen
          (refOverride := wKRef)
        Linear.forwardBatchDP4A_fromQ8 ctx block.attention.wV batchQ8Buf batchVBuf seqLen
          (refOverride := wVRef)
    else
      -- Q6_K fallback: RMSNorm into batchNormedBuf as scratch (CANNOT use
      -- batchBuf1 since it may alias currentBuf during ping-pong).  Then
      -- batch matmul in f32.
      -- Use a throwaway ref: batchNormedBuf/currentBuf are per-call allocations
      -- so layer.prepared's cached args would point to stale pointers.
      let attnNormRef ← IO.mkRef none
      RMSNorm.forward ctx block.attnNorm currentBuf batchNormedBuf seqLen
        (refOverride := some attnNormRef)
      Linear.forwardBatchDP4A ctx block.attention.wQ batchNormedBuf batchQBuf seqLen
      if cfg.hasKV li then
        Linear.forwardBatchDP4A ctx block.attention.wK batchNormedBuf batchKBuf seqLen
        Linear.forwardBatchDP4A ctx block.attention.wV batchNormedBuf batchVBuf seqLen
    -- Golden dumps (outside the Q4K/Q6K branch so every layer gets one).
    -- Note: for the Q4K path, attn_norm output is quantized Q8_1 — we dump
    -- the RMSNorm-then-matmul output (Qcur/Kcur/Vcur), not attn_norm.
    -- For the Q6K path we dump batchNormedBuf which IS attn_norm.
    if !allQ4K then dumpGolden s!"attn_norm-{li}" batchNormedBuf (dim * seqLen)
    dumpGolden s!"Qcur-{li}" batchQBuf (qDim * seqLen)
    if cfg.hasKV li then
      dumpGolden s!"Kcur-{li}" batchKBuf (kvDim * seqLen)
      dumpGolden s!"Vcur-{li}" batchVBuf (kvDim * seqLen)

    -- ── 2c: Attention (batched when possible, per-token fallback) ─────
    let wgSize := min headDim 256
    let isFull := cfg.isFullAttention li
    let kvLi := cfg.kvCacheLayer li
    -- Batched path requires:
    --   * cfg.hasKV li (this layer has its own KV cache)
    --   * block.ropeFreqFactors = some (full-attention layers only)
    --   * a valid KV cache slot
    let mut handledByBatched := false
    if hKV : kvLi < state.kvCaches.size then
      -- SWA layers have ropeFreqFactors = none but can share the batched
      -- RoPE-Q/K kernels with a 1.0-filled freq_factors (gated by
      -- HESPER_UNIFY_ATTN=swa).  This only works now that the in-place
      -- RWW bug in ropeWithFreqFactorsBatchKernel is fixed (commit c92e377).
      -- Shared-KV layers (cfg.hasKV li = false) only need a Q-only path.
      let effectiveFreqFactors :=
        if !forceFallback then
          match block.ropeFreqFactors with
          | some ff => some ff
          | none => if unifyAttnSwa then some onesBuf else none
        else none
      match effectiveFreqFactors, cfg.hasKV li with
      | some freqFactors, true =>
        handledByBatched := true
        let kvCache := state.kvCaches[kvLi]

        -- Write startPos to paramsBuf[0]; the kernel treats wgid.y/z as
        -- the per-token offset (absolute position = startPos + col).
        writeParamsU32At 0 startPos

        -- Batched fused QKV norm: grid (numHeads*seqLen, 3, 1).
        ce (if isFull then "qkvNormFullBatch" else "qkvNormSWABatch")
          (fusedPerHeadQKVNormBatchKernel numHeads numKVHeads headDim seqLen cfg.rmsNormEps)
          [("q_in", batchQBuf), ("q_scale", block.attention.qNormWeight), ("q_out", batchQBuf),
           ("k_in", batchKBuf), ("k_scale", block.attention.kNormWeight), ("k_out", batchKBuf),
           ("v_in", batchVBuf),                                            ("v_out", batchVBuf)]
          { numWorkgroups := (numHeads * seqLen, 3, 1),
            workgroupSize := { x := wgSize, y := 1, z := 1 } : Hesper.ExecConfig }
        let stageActive := stageDumpLayer.any (· = li)
        dumpStage s!"Qnormed_L{li}" batchQBuf (qDim * seqLen) stageActive
        dumpStage s!"Knormed_L{li}" batchKBuf (kvDim * seqLen) stageActive
        dumpStage s!"Vnormed_L{li}" batchVBuf (kvDim * seqLen) stageActive
        -- DEBUG: golden-dump pre-RoPE Q/K/V (matches llama.cpp's Qcur_normed / Kcur_normed / Vcur_normed)
        dumpGolden s!"Qcur_normed-{li}" batchQBuf (qDim * seqLen)
        dumpGolden s!"Kcur_normed-{li}" batchKBuf (kvDim * seqLen)
        dumpGolden s!"Vcur_normed-{li}" batchVBuf (kvDim * seqLen)

        -- Batched RoPE-Q (NOT in place — write to a dedicated scratch to
        -- avoid any read-modify-write hazard on the shared Q buffer).
        ce s!"ropeFreqQBatchOut_{headDim}_base{cfg.ropeBase li}"
          (ropeWithFreqFactorsBatchKernel headDim numHeads seqLen (cfg.ropeBase li))
          [("input", batchQBuf), ("output", batchQRopedBuf),
           ("params", state.paramsBuf), ("freq_factors", freqFactors)]
          (.dispatch1D (numHeads * headDim / 2 * seqLen))
        dumpStage s!"Qroped_L{li}" batchQRopedBuf (qDim * seqLen) stageActive
        -- Golden dump: post-RoPE Q (matches llama.cpp's `Qcur_pos-<li>`).
        dumpGolden s!"Qcur_pos-{li}" batchQRopedBuf (qDim * seqLen)

        -- Batched RoPE-K + KV cache write — f16 packed half2 cache.
        -- Single dispatch writes both K (with NeoX RoPE) and V into the
        -- f16 cache used by V11 in decode.
        ce s!"ropeKKvWBatchF16_{headDim}_{numKVHeads}_base{cfg.ropeBase li}"
          (Attention.fusedRopeKAndCacheWriteBatchKernelF16
             numKVHeads cfg.maxSeqLen headDim seqLen (cfg.ropeBase li))
          [("new_k", batchKBuf), ("new_v", batchVBuf),
           ("k_cache_f16", kvCache.kBufF16), ("v_cache_f16", kvCache.vBufF16),
           ("params", state.paramsBuf), ("freq_factors", freqFactors)]
          (.dispatch1D (numKVHeads * (headDim / 2) * seqLen))
        dumpStage s!"Kcache_L{li}" kvCache.kBufF16 (numKVHeads * cfg.maxSeqLen * (headDim / 2)) stageActive
        dumpStage s!"Vcache_L{li}" kvCache.vBufF16 (numKVHeads * cfg.maxSeqLen * (headDim / 2)) stageActive

        -- Batched flash-attention reading f16 K/V cache.
        let scale : Float := 1.0
        ce s!"flashAttnBatchF16_{headDim}_{numKVHeads}"
          (FlashAttention.flashAttentionBatchKernelF16 numHeads numKVHeads cfg.maxSeqLen headDim seqLen scale)
          [("q", batchQRopedBuf),
           ("k_cache_f16", kvCache.kBufF16), ("v_cache_f16", kvCache.vBufF16),
           ("output", batchAttnOutBuf), ("params", state.paramsBuf)]
          ({ numWorkgroups := (numHeads, seqLen, 1) : Hesper.ExecConfig })
        dumpStage s!"attnOut_L{li}" batchAttnOutBuf (qDim * seqLen) stageActive
        -- Golden dump: FlashAttention output pre-Oproj (llama.cpp's `__fattn__-<li>`).
        dumpGolden s!"__fattn__-{li}" batchAttnOutBuf (qDim * seqLen)
      | some freqFactors, false =>
        -- Shared-KV batched path: only Q is computed this layer; K/V
        -- come from an earlier layer's cache at kvLi = cfg.kvCacheLayer li.
        -- Flow: Q-only batched norm → batched RoPE-Q → batched FA.
        handledByBatched := true
        let kvCache := state.kvCaches[kvLi]
        writeParamsU32At 0 startPos
        -- Q-only norm, grid (numHeads, seqLen, 1).
        ce s!"qNormBatch_{headDim}"
          (perHeadRMSNormBatchKernel numHeads headDim seqLen cfg.rmsNormEps)
          [("input", batchQBuf), ("weight", block.attention.qNormWeight),
           ("output", batchQBuf)]
          { numWorkgroups := (numHeads, seqLen, 1),
            workgroupSize := { x := wgSize, y := 1, z := 1 } : Hesper.ExecConfig }
        let stageActive := stageDumpLayer.any (· = li)
        dumpStage s!"Qnormed_L{li}" batchQBuf (qDim * seqLen) stageActive
        -- Batched RoPE-Q → batchQRopedBuf (out-of-place; see commit c92e377).
        ce s!"ropeFreqQBatchOut_{headDim}_base{cfg.ropeBase li}"
          (ropeWithFreqFactorsBatchKernel headDim numHeads seqLen (cfg.ropeBase li))
          [("input", batchQBuf), ("output", batchQRopedBuf),
           ("params", state.paramsBuf), ("freq_factors", freqFactors)]
          (.dispatch1D (numHeads * headDim / 2 * seqLen))
        dumpStage s!"Qroped_L{li}" batchQRopedBuf (qDim * seqLen) stageActive
        -- Golden dump: post-RoPE Q (matches llama.cpp's `Qcur_pos-<li>`).
        dumpGolden s!"Qcur_pos-{li}" batchQRopedBuf (qDim * seqLen)
        -- No K/V writes — cache was populated by the layer at kvLi.
        -- Batched FA reading f16 K/V cache.
        let scale : Float := 1.0
        ce s!"flashAttnBatchF16_{headDim}_{numKVHeads}"
          (FlashAttention.flashAttentionBatchKernelF16 numHeads numKVHeads cfg.maxSeqLen headDim seqLen scale)
          [("q", batchQRopedBuf),
           ("k_cache_f16", kvCache.kBufF16), ("v_cache_f16", kvCache.vBufF16),
           ("output", batchAttnOutBuf), ("params", state.paramsBuf)]
          ({ numWorkgroups := (numHeads, seqLen, 1) : Hesper.ExecConfig })
        dumpStage s!"attnOut_L{li}" batchAttnOutBuf (qDim * seqLen) stageActive
        -- Golden dump: FlashAttention output pre-Oproj (llama.cpp's `__fattn__-<li>`).
        dumpGolden s!"__fattn__-{li}" batchAttnOutBuf (qDim * seqLen)
      | none, _ => pure ()

    if !handledByBatched then
    for i in [0:seqLen] do
      let pos := i

      -- Extract Q column i → state.qBuf
      let colIdxBytes := Hesper.WebGPU.BufferOps.uint32ToBytes i.toUInt32
      GPUBackend.writeBufferOffset ctx colIdxBuf 0 colIdxBytes
      ce s!"colExtractQ_sl{seqLen}_d{qDim}"
        (columnExtractKernel qDim seqLen)
        [("batch", batchQBuf), ("params", colIdxBuf), ("out", state.qBuf)]
        (.dispatch1D qDim)
      -- Extract K, V columns (if this layer has its own KV)
      if cfg.hasKV li then
        ce s!"colExtractK_sl{seqLen}_d{kvDim}"
          (columnExtractKernel kvDim seqLen)
          [("batch", batchKBuf), ("params", colIdxBuf), ("out", state.kBuf)]
          (.dispatch1D kvDim)
        ce s!"colExtractV_sl{seqLen}_d{kvDim}"
          (columnExtractKernel kvDim seqLen)
          [("batch", batchVBuf), ("params", colIdxBuf), ("out", state.vBuf)]
          (.dispatch1D kvDim)

      -- Per-head QKV norms (single token)
      if cfg.hasKV li then
        ce (if isFull then "qkvNormFull_pf" else "qkvNormSWA_pf")
          (fusedPerHeadQKVNormKernel numHeads numKVHeads headDim cfg.rmsNormEps)
          [("q_in", state.qBuf), ("q_scale", block.attention.qNormWeight), ("q_out", state.qBuf2),
           ("k_in", state.kBuf), ("k_scale", block.attention.kNormWeight), ("k_out", state.kBuf2),
           ("v_in", state.vBuf),                                              ("v_out", state.vBuf2)]
          { numWorkgroups := (numHeads, 3, 1),
            workgroupSize := { x := wgSize, y := 1, z := 1 } : Hesper.ExecConfig }
      else
        ce (if isFull then "qNormFull_pf" else "qNormSWA_pf")
          (perHeadRMSNormKernel numHeads headDim cfg.rmsNormEps)
          [("input", state.qBuf), ("weight", block.attention.qNormWeight), ("output", state.qBuf2)]
          { numWorkgroups := (numHeads, 1, 1),
            workgroupSize := { x := wgSize, y := 1, z := 1 } : Hesper.ExecConfig }
      -- Per-stage dumps: fire once at a chosen iter (default = last).
      -- HESPER_ATTN_STAGE_TOKEN picks which token i to dump (0..seqLen-1).
      let targetTok := (← IO.getEnv "HESPER_ATTN_STAGE_TOKEN").bind String.toNat?
        |>.getD (seqLen - 1)
      let stageActive := stageDumpLayer.any (· = li) && i = targetTok
      dumpStage s!"Qnormed_L{li}" state.qBuf2 qDim stageActive
      if cfg.hasKV li then
        dumpStage s!"Knormed_L{li}" state.kBuf2 kvDim stageActive
        dumpStage s!"Vnormed_L{li}" state.vBuf2 kvDim stageActive

      -- RoPE on Q: qBuf2 → qBuf
      -- GPU-side: posBuf[i] → paramsBuf[0]
      -- NOTE: colIdxBuf already holds `i` from the Q/K/V extract above
      -- (extracts always run first this iteration, so skip the redundant
      -- 4-byte HtoD write here — 1 HtoD/token/layer × 42 × 9 = 378 saved).
      ce s!"copyU32Pos_sl{seqLen}"
        (copyU32Kernel seqLen 2 0)
        [("src", posBuf), ("params", colIdxBuf), ("dst", state.paramsBuf)]
        { numWorkgroups := (1, 1, 1), workgroupSize := { x := 1, y := 1, z := 1 } }
      match block.ropeFreqFactors with
      | some freqFactors =>
        ce s!"ropeFreqQ_pf_{headDim}"
          (ropeWithFreqFactorsKernel headDim numHeads (cfg.ropeBase li))
          [("input", state.qBuf2), ("output", state.qBuf), ("params", state.paramsBuf), ("freq_factors", freqFactors)]
          (.dispatch1D (numHeads * headDim / 2))
      | none =>
        let ropeConfig : RoPE.Config := { dim := numHeads * headDim, maxSeqLen := cfg.maxSeqLen, base := cfg.ropeBase li }
        ce s!"ropeDynQ_pf_{headDim}_base{cfg.ropeBase li}"
          (RoPE.ropeKernelDynamic ropeConfig 1 1 numHeads headDim)
          [("input", state.qBuf2), ("output", state.qBuf), ("params", state.paramsBuf)]
          (.dispatch1D (numHeads * headDim / 2))
      dumpStage s!"Qroped_L{li}" state.qBuf qDim stageActive

      -- RoPE on K + KV cache write + flash attention
      if h : kvLi < state.kvCaches.size then
        let kvCache := state.kvCaches[kvLi]
        let cacheLen := pos + 1

        if cfg.hasKV li then
          -- RoPE-K + KV cache write
          match block.ropeFreqFactors with
          | some freqFactors =>
            ce s!"ropeKKvW_pf_{headDim}_{numKVHeads}"
              (Attention.fusedRopeKAndCacheWriteKernel numKVHeads cfg.maxSeqLen headDim kvDim (cfg.ropeBase li))
              [("new_k", state.kBuf2), ("new_v", state.vBuf2),
               ("k_cache", kvCache.kBuf), ("v_cache", kvCache.vBuf),
               ("params", state.paramsBuf), ("freq_factors", freqFactors)]
              (.dispatch1D kvDim)
          | none =>
            -- Separate RoPE-K then KV write
            let ropeConfig : RoPE.Config := { dim := kvDim, maxSeqLen := cfg.maxSeqLen, base := cfg.ropeBase li }
            ce s!"ropeDynK_pf_{headDim}_{numKVHeads}_base{cfg.ropeBase li}"
              (RoPE.ropeKernelDynamic ropeConfig 1 1 numKVHeads headDim)
              [("input", state.kBuf2), ("output", state.kBuf), ("params", state.paramsBuf)]
              (.dispatch1D (numKVHeads * headDim / 2))
            ce s!"kvWrite_pf_{headDim}_{numKVHeads}"
              (Attention.fusedCacheWriteKVKernel numKVHeads cfg.maxSeqLen headDim kvDim)
              [("new_k", state.kBuf), ("new_v", state.vBuf2),
               ("k_cache", kvCache.kBuf), ("v_cache", kvCache.vBuf),
               ("params", state.paramsBuf)]
              (.dispatch1D kvDim)
        if cfg.hasKV li then
          dumpStage s!"Kcache_L{li}" kvCache.kBuf (numKVHeads * cfg.maxSeqLen * headDim) stageActive
          dumpStage s!"Vcache_L{li}" kvCache.vBuf (numKVHeads * cfg.maxSeqLen * headDim) stageActive

        -- Flash attention
        let scale : Float := 1.0
        -- GPU-side: cacheLenBuf[i] → paramsBuf[1] (offset 4 bytes = u32 index 1)
        ce s!"copyU32CacheLen_sl{seqLen}"
          (copyU32Kernel seqLen 2 1)
          [("src", cacheLenBuf), ("params", colIdxBuf), ("dst", state.paramsBuf)]
          { numWorkgroups := (1, 1, 1), workgroupSize := { x := 1, y := 1, z := 1 } }
        if cacheLen > 32 then
          let kcrLk := kcr.map (fun k key => k.getRef key)
          FlashAttention.executeFlashAttentionTiled ctx
            state.qBuf kvCache.kBuf kvCache.vBuf state.attnOutBuf
            numHeads numKVHeads cfg.maxSeqLen headDim cacheLen scale
            (partialBuf := some state.flashPartialBuf)
            (kcrLookup := kcrLk)
        else
          ce s!"flashAttnP_pf_{headDim}_{numKVHeads}"
            (FlashAttention.flashAttentionDynamicParamsKernel numHeads numKVHeads cfg.maxSeqLen headDim scale)
            [("q", state.qBuf), ("k_cache", kvCache.kBuf), ("v_cache", kvCache.vBuf),
             ("output", state.attnOutBuf), ("params", state.paramsBuf)]
            ({ numWorkgroups := (numHeads, 1, 1) : Hesper.ExecConfig })
        dumpStage s!"attnOut_L{li}" state.attnOutBuf qDim stageActive

        -- Insert attnOut into batch buffer for later O-projection.
        -- colIdxBuf still holds `i` from the extracts — skip redundant write.
        ce s!"colInsertAttnOut_sl{seqLen}_d{qDim}"
          (columnInsertKernel qDim seqLen)
          [("src", state.attnOutBuf), ("params", colIdxBuf), ("batch", batchAttnOutBuf)]
          (.dispatch1D qDim)
    else pure ()

    -- Bit-parity harness: dump `batchAttnOutBuf` right after attention.
    -- Use per-layer filenames so full/SWA/shared-KV can be inspected
    -- independently.  Dumps go under `$HESPER_ATTN_DUMP_DIR` (default: /tmp).
    match attnDumpTag with
    | some tag =>
      let dir := (← IO.getEnv "HESPER_ATTN_DUMP_DIR").getD "/tmp"
      GPUBackend.endBatch ctx
      let bytes := (qDim * seqLen * 4).toUSize
      let data ← GPUBackend.readBuffer ctx batchAttnOutBuf bytes
      IO.FS.writeBinFile s!"{dir}/attn_L{li}_{tag}.bin" data
    | none => pure ()

    -- ── 2d: O projection (batch matmul) ──────────────────────────────
    Linear.forwardBatchDP4A ctx block.attention.wO batchAttnOutBuf batchOProjBuf seqLen

    -- ── 2e: Post-attention norm + residual (batched) ─────────────────
    -- `batchAttnResid[i,d] = RMSNorm(oProj[i,:])[d] * scale[d] + currentBuf[i,d]`
    -- One dispatch over seqLen rows (was: 3 dispatches × seqLen).
    let postAttnKey := hash ("postAttnNormAddBatch", cfg.hiddenSize, seqLen)
    let postAttnRef ← match kcr with
      | some k => k.getRef postAttnKey
      | none => IO.mkRef none
    RMSNorm.forwardNormThenAddBatch ctx block.postAttnNorm
      batchOProjBuf currentBuf batchAttnResidBuf seqLen postAttnRef
    -- Golden dump: `attn_out-<li>` (matches llama.cpp's post-attn-residual
    -- tensor — `ggml_add(cur, inpL)` at gemma4-iswa.cpp:112).
    dumpGolden s!"attn_out-{li}" batchAttnResidBuf (dim * seqLen)

    -- Diagnostic: dump currentBuf col 0 at L1 after attention inner loop / after O proj
    if dumpEnabled && li = 1 then
      GPUBackend.writeBufferOffset ctx colIdxBuf 0 (Hesper.WebGPU.BufferOps.uint32ToBytes 0)
      GPUBackend.execute ctx
        (columnExtractKernel dim seqLen)
        [("batch", currentBuf), ("params", colIdxBuf), ("out", state.buf1)]
        (.dispatch1D dim)
      dumpBuf ctx state.buf1 (dim * 4).toUSize s!"batch_t0_currBufL1afterAttnOProj"

    -- Dump post-attn residual for each token (batch diagnostic) — only L0/L1 for brevity
    if dumpEnabled && li ≤ 2 then for i in [0:seqLen] do
      let idxBytes := Hesper.WebGPU.BufferOps.uint32ToBytes i.toUInt32
      GPUBackend.writeBufferOffset ctx colIdxBuf 0 idxBytes
      GPUBackend.execute ctx
        (columnExtractKernel dim seqLen)
        [("batch", batchAttnResidBuf), ("params", colIdxBuf), ("out", state.buf1)]
        (.dispatch1D dim)
      dumpBuf ctx state.buf1 (dim * 4).toUSize s!"batch_t{i}_postAttn_L{li}"

    -- ── 2f: FFN ──────────────────────────────────────────────────────
    -- Skip MoE — use dense FFN path only.

    -- FFN norm + Gate/Up projections (batch)
    -- Fast path (both Q4_K): fused RMSNorm+Q8_1 quantize → Q8_1 batch matmul.
    -- Fallback: standalone RMSNorm → f32 batch matmul.
    let ffnFastDisabled := (← IO.getEnv "HESPER_FFN_FASTPATH_DISABLE").isSome
    let ffnAllQ4K := block.ffn.gate.quantFormat == .Q4_K
                  && block.ffn.up.quantFormat == .Q4_K
                  && !ffnFastDisabled
    if ffnAllQ4K then
      let ffnQ8Bytes : USize := (nQ8Blocks * 9 * seqLen * 4).toUSize
      -- Pooled FFN Q8_1 scratch (separate from the attention-side pool so
      -- they can coexist within a single block's forward without clashing).
      let ffnBatchQ8Buf ← do
        match ← state.prefillFfnQ8BufRef.get with
        | some (b, sz) => if sz >= ffnQ8Bytes then pure b
                          else
                            GPUBackend.freeBuffer ctx b
                            let b' ← GPUBackend.allocBuffer ctx ffnQ8Bytes
                            state.prefillFfnQ8BufRef.set (some (b', ffnQ8Bytes))
                            pure b'
        | none =>
          let b ← GPUBackend.allocBuffer ctx ffnQ8Bytes
          state.prefillFfnQ8BufRef.set (some (b, ffnQ8Bytes))
          pure b
      ce s!"ffnNormQ8_1_sl{seqLen}_d{dim}"
        (RMSNorm.fusedRMSNormQ8_1Kernel block.ffnNorm.config seqLen 256)
        [("input", batchAttnResidBuf), ("scale", block.ffnNorm.scale), ("output", ffnBatchQ8Buf)]
        { workgroupSize := { x := 256 }, numWorkgroups := (seqLen, 1, 1) }
      let gateKey := hash ("q4k-batch-matmul-q8-ffn", li, "gate",
        block.ffn.gate.config.inDim, block.ffn.gate.config.outDim, seqLen)
      let upKey := hash ("q4k-batch-matmul-q8-ffn", li, "up",
        block.ffn.up.config.inDim, block.ffn.up.config.outDim, seqLen)
      let gateRef ← match kcr with
        | some k => k.getRef gateKey |>.map some
        | none   => pure none
      let upRef ← match kcr with
        | some k => k.getRef upKey |>.map some
        | none   => pure none
      Linear.forwardBatchDP4A_fromQ8 ctx block.ffn.gate ffnBatchQ8Buf batchGateBuf seqLen
        (refOverride := gateRef)
      Linear.forwardBatchDP4A_fromQ8 ctx block.ffn.up ffnBatchQ8Buf batchUpBuf seqLen
        (refOverride := upRef)
    else
      -- FFN Q6_K fallback: normedBuf can't be batchBuf1 (ping-pong alias).
      -- Throwaway ref: transient batch buffers.
      let ffnNormRef ← IO.mkRef none
      RMSNorm.forward ctx block.ffnNorm batchAttnResidBuf batchNormedBuf seqLen
        (refOverride := some ffnNormRef)
      -- Golden dump: ffn_norm output (post-RMSNorm, pre-gate/up).
      -- Only visible in the fallback branch; fast path bakes it into Q8_1.
      dumpGolden s!"ffn_norm-{li}" batchNormedBuf (dim * seqLen)
      Linear.forwardBatchDP4A ctx block.ffn.gate batchNormedBuf batchGateBuf seqLen
      Linear.forwardBatchDP4A ctx block.ffn.up batchNormedBuf batchUpBuf seqLen

    -- Golden dumps: pre-GELU gate/up outputs.  Matches llama.cpp's
    -- `ffn_gate-<li>` / `ffn_up-<li>` at gemma4-iswa.cpp (build_ffn
    -- internal; both are [interSize, seqLen] f32 tensors).
    dumpGolden s!"ffn_gate-{li}" batchGateBuf (interSize * seqLen)
    dumpGolden s!"ffn_up-{li}" batchUpBuf (interSize * seqLen)

    -- GELU * up (batch pointwise — dispatch with totalElements = interSize * seqLen)
    let totalInter := interSize * seqLen
    ce s!"geluMulBatch_{li}"
      (geluMulKernel totalInter)
      [("gate", batchGateBuf), ("up", batchUpBuf), ("output", batchGeluBuf)]
      (.dispatch1D totalInter)

    -- Golden dump: GELU(gate) * up (pre-down-proj).  Matches llama.cpp's
    -- `ffn_geglu-<li>` tensor.
    dumpGolden s!"ffn_geglu-{li}" batchGeluBuf (interSize * seqLen)

    -- Down projection (batch matmul)
    Linear.forwardBatchDP4A ctx block.ffn.down batchGeluBuf batchFFNOutBuf seqLen

    -- Golden dump: FFN down-proj output (pre post-FFN-norm+residual).
    -- Matches llama.cpp's `ffn_out-<li>` at gemma4-iswa.cpp:182.
    dumpGolden s!"ffn_out-{li}" batchFFNOutBuf (dim * seqLen)

    -- ── 2g: Post-FFN norm + residual (batched) ───────────────────────
    -- `nextBuf[i,d] = RMSNorm(ffnOut[i,:])[d] * scale[d] + attnResid[i,d]`
    -- One dispatch over seqLen rows (was: 3 dispatches × seqLen).
    let postFFNKey := hash ("postFFNNormAddBatch", cfg.hiddenSize, seqLen)
    let postFFNRef ← match kcr with
      | some k => k.getRef postFFNKey
      | none => IO.mkRef none
    RMSNorm.forwardNormThenAddBatch ctx block.postFFNNorm
      batchFFNOutBuf batchAttnResidBuf nextBuf seqLen postFFNRef
    -- Golden dump: post-FFN residual add; matches llama.cpp's `ffn_post_norm-<li>`
    -- named tensor (cur = ggml_add(cur, attn_out) at gemma4-iswa.cpp:190).
    dumpGolden s!"ffn_post_norm-{li}" nextBuf (dim * seqLen)

    -- Dump post-FFN (pre-PLE) state
    if dumpEnabled then for i in [0:seqLen] do
      let idxBytes := Hesper.WebGPU.BufferOps.uint32ToBytes i.toUInt32
      GPUBackend.writeBufferOffset ctx colIdxBuf 0 idxBytes
      GPUBackend.execute ctx
        (columnExtractKernel dim seqLen)
        [("batch", nextBuf), ("params", colIdxBuf), ("out", state.buf1)]
        (.dispatch1D dim)
      dumpBuf ctx state.buf1 (dim * 4).toUSize s!"batch_t{i}_postFFN_L{li}"

    -- ── 2h: Per-layer embedding + layer output scale (per-token) ──────
    -- Gemma 4 E4B uses per_layer_token_embd: each layer's output gets an
    -- additive embedding that depends on both the layer index and the
    -- token's per-layer input (precomputed in Step 1b).  We also apply
    -- the layer output scale (a single scalar multiply per layer).
    let blockIdx := li
    let skipPLE := (← IO.getEnv "HESPER_SKIP_PLE").isSome
    let plEmbd := if blockIdx < model.perLayerBlocks.size && !skipPLE then
      model.perLayerBlocks[blockIdx]!
    else none
    match plEmbd with
    | some ple =>
      -- Per-layer embedding path — fully batched across seqLen.  Replaces
      -- the old per-token loop (9 × 42 × 6 = 2268 dispatches) with 4
      -- dispatches per layer (matmul + gelu*gate*slice + matmul + norm+add).
      match batchPLInputAllOpt with
      | some batchPL =>
        let plOffset := li * cfg.embdPerLayer
        let plTotalSize := cfg.embdPerLayer * cfg.numHiddenLayers
        -- Golden dump: `pe_in-<li>` = input to PLE (post-FFN-norm+residual).
        -- Matches llama.cpp's `pe_in-<li>` at gemma4-iswa.cpp:195.
        dumpGolden s!"pe_in-{li}" nextBuf (dim * seqLen)
        -- 1) inpGate matmul: nextBuf [dim, seqLen] → plGateBatchBuf [embdPerLayer, seqLen]
        Linear.forwardBatchDP4A ctx ple.inpGate nextBuf plGateBatchBuf seqLen
        dumpGolden s!"ple_gate-{li}" plGateBatchBuf (cfg.embdPerLayer * seqLen)
        -- 2) GELU(gate) * per_layer_input[li_slice] — batched across seqLen
        -- Cache key MUST include plOffset; it's baked into the shader as a
        -- literal, and reusing a stale cached kernel would apply the wrong
        -- layer's slice offset (this was a real bug — see commit log).
        ce s!"pleGeluGateMulSliceBatch_off{plOffset}"
          (PerLayerEmbedding.geluGateMulSliceBatchKernel cfg.embdPerLayer plTotalSize plOffset seqLen)
          [("gate", plGateBatchBuf), ("per_layer_input", batchPL), ("output", plMoeOutBatchBuf)]
          (.dispatch1D (cfg.embdPerLayer * seqLen))
        dumpGolden s!"ple_moe_out-{li}" plMoeOutBatchBuf (cfg.embdPerLayer * seqLen)
        -- 3) proj matmul: plMoeOutBatchBuf [embdPerLayer, seqLen] → plProjBatchBuf [dim, seqLen]
        Linear.forwardBatchDP4A ctx ple.proj plMoeOutBatchBuf plProjBatchBuf seqLen
        dumpGolden s!"ple_proj-{li}" plProjBatchBuf (dim * seqLen)
        -- 4) Fused post-norm + residual add: nextBuf[i,d] = RMSNorm(plProj[i,:])[d] * scale[d] + nextBuf[i,d]
        let plePostKey := hash ("plePostNormAddBatch", cfg.hiddenSize, seqLen)
        let plePostRef ← match kcr with
          | some k => k.getRef plePostKey
          | none => IO.mkRef none
        RMSNorm.forwardNormThenAddBatch ctx ple.postNorm
          plProjBatchBuf nextBuf nextBuf seqLen plePostRef
      | none => pure ()
    | none => pure ()

    -- Dump post-PLE (pre-outScale) state
    if dumpEnabled then for i in [0:seqLen] do
      let idxBytes := Hesper.WebGPU.BufferOps.uint32ToBytes i.toUInt32
      GPUBackend.writeBufferOffset ctx colIdxBuf 0 idxBytes
      GPUBackend.execute ctx
        (columnExtractKernel dim seqLen)
        [("batch", nextBuf), ("params", colIdxBuf), ("out", state.buf1)]
        (.dispatch1D dim)
      dumpBuf ctx state.buf1 (dim * 4).toUSize s!"batch_t{i}_postPLE_L{li}"

    -- Golden dump: post-PLE / pre-outScale.  Matches llama.cpp's
    -- `per_layer_embd_out-<li>` at gemma4-iswa.cpp:209.
    dumpGolden s!"per_layer_embd_out-{li}" nextBuf (dim * seqLen)

    -- Golden dump: pre-outScale state (post-PLE including pe_in residual).
    -- llama.cpp's `out_scaled-<li>` fires at this point but isn't currently
    -- emitted by llama-eval-callback (same tensor gets renamed to "l_out"
    -- by the next `cb` call).  So we dump a hesper-only probe as
    -- `pre_out_scaled-<li>`.
    dumpGolden s!"pre_out_scaled-{li}" nextBuf (dim * seqLen)

    -- Layer output scale (per-token).  In-place broadcast multiply over the
    -- entire [dim, seqLen] batch tensor — 1 dispatch total, no per-column
    -- extract/insert.  Replaces a `for i in [0:seqLen]` chain of 3 dispatches
    -- per column (84 saved at seqLen=1 × 42 layers).
    let skipOutScale := (← IO.getEnv "HESPER_SKIP_OUTSCALE").isSome
    match if skipOutScale then none else block.outScale with
    | some scale =>
      let total := dim * seqLen
      ce s!"batchPrefillOutScale_{dim}"
        (batchBroadcastScaleInPlaceKernel total)
        [("buf", nextBuf), ("scale", scale)]
        (.dispatch1D total)
    | none => pure ()

    -- Per-layer batch Q8_1 buffers are freed inside their respective
    -- Q4_K fast-path branches above.

    -- Golden dump: post-outScale.  Matches llama.cpp's `out_scaled-<li>`
    -- at gemma4-iswa.cpp:218.
    dumpGolden s!"out_scaled-{li}" nextBuf (dim * seqLen)

    -- Swap ping-pong buffers
    let oldCur := currentBuf
    currentBuf := nextBuf
    nextBuf := oldCur

    -- Golden dump: layer output — matches llama.cpp's `l_out-<li>` tensor.
    dumpGolden s!"l_out-{li}" currentBuf (dim * seqLen)

    -- Dump post-layer state for each token
    if dumpEnabled then for i in [0:seqLen] do
      let idxBytes := Hesper.WebGPU.BufferOps.uint32ToBytes i.toUInt32
      GPUBackend.writeBufferOffset ctx colIdxBuf 0 idxBytes
      GPUBackend.execute ctx
        (columnExtractKernel dim seqLen)
        [("batch", currentBuf), ("params", colIdxBuf), ("out", state.buf1)]
        (.dispatch1D dim)
      dumpBuf ctx state.buf1 (dim * 4).toUSize s!"batch_t{i}_afterL{li}"
      -- Also dump previous-buffer (before outScale if it ran) to localize
      GPUBackend.execute ctx
        (columnExtractKernel dim seqLen)
        [("batch", batchAttnResidBuf), ("params", colIdxBuf), ("out", state.buf1)]
        (.dispatch1D dim)
      dumpBuf ctx state.buf1 (dim * 4).toUSize s!"batch_t{i}_attnResidL{li}"

  -- Phase 3 diagnostic: dump all KV caches after prefill (one file per
  -- layer) for comparison against llama.cpp ground truth.  Enabled by
  -- setting HESPER_KVCACHE_DUMP_DIR=<path>.
  match kvDumpDir with
  | some dir =>
    GPUBackend.endBatch ctx
    for li in [0:state.kvCaches.size] do
      if h : li < state.kvCaches.size then
        let kvc := state.kvCaches[li]
        let headDim := cfg.headDim li
        let numKVHeads := cfg.numKVHeads li
        let bytes := (numKVHeads * cfg.maxSeqLen * headDim * 4).toUSize
        let kData ← GPUBackend.readBuffer ctx kvc.kBuf bytes
        let vData ← GPUBackend.readBuffer ctx kvc.vBuf bytes
        IO.FS.writeBinFile s!"{dir}/k_L{li}.bin" kData
        IO.FS.writeBinFile s!"{dir}/v_L{li}.bin" vData
  | none => pure ()

  -- ── Step 3: Extract last token → final norm → lm head ─────────────
  -- Copy last column of currentBuf → state.buf2 (single-token)
  let lastCol := seqLen - 1
  -- For seqLen=1 (unified decode) lastCol is always 0 so this reuses the
  -- shared colIdx slot without value conflict.  For prefill we fall into
  -- the graph-safe helper too but capture doesn't run for prefill.
  writeColIdxU32 colIdxBuf lastCol
  ce s!"colExtractLast_sl{seqLen}_d{dim}"
    (columnExtractKernel dim seqLen)
    [("batch", currentBuf), ("params", colIdxBuf), ("out", state.buf2)]
    (.dispatch1D dim)
  -- Debug: dump state.buf2 (the extracted last-column hidden state that
  -- feeds finalNorm + lm_head).  Compare against single-token path
  -- which writes to state.buf2 via its own column-extract for prefill.
  dumpGolden s!"prefill_buf2_lastcol_seqLen{seqLen}" state.buf2 dim

  -- LM head.
  --
  -- For the Q6_K dp4a path, fuse the final RMSNorm directly into the Q8_1
  -- quantize step — identical pattern to `forwardFusedNormGateUp`.  Saves
  -- one dispatch per token (the standalone RMSNorm) and one ~2560-float
  -- VRAM round-trip (the f32 normed hidden state).  For the fallback
  -- paths (f32 matmul etc.) still run the standalone RMSNorm because
  -- they don't consume Q8_1.
  -- HESPER_DP4A_Q6K_4WARP=1 forces dp4a path even when an f16 lm_head
  -- buffer was prepared at load time.  Same toggle as the decode site.
  let force4WarpQ6K_pf ← do
    match ← IO.getEnv "HESPER_DP4A_Q6K_4WARP" with
    | some "1" => pure true
    | _        => pure false
  match (if force4WarpQ6K_pf then none else model.outputWeightF16) with
  | some f16W =>
    -- Pre-dequantized f16 lm_head — single dispatch, no Q8_1 quantize.
    RMSNorm.forward ctx model.finalNorm state.buf2 state.buf1
    let lmHeadConfig : Hesper.WGSL.MatMul.Config := {
      M := 1, N := cfg.vocabSize, K := cfg.hiddenSize
    }
    Hesper.WGSL.MatMul.executeMatMulTransposeF16BlockCoop ctx state.buf1 f16W state.logitsBuf lmHeadConfig
    dumpGolden "prefill_logits_raw" state.logitsBuf cfg.vocabSize
  | none =>
  match model.embdFormat with
  | .Q6_K =>
    let useSubgroups ← GPUBackend.hasSubgroupSupport ctx
    let dp4aOn ← do
      let a ← Hesper.Layers.Linear.dp4aEnabled.get
      let b ← Hesper.Layers.Linear.dp4aQ6KEnabled.get
      pure (a && b)
    let gridX : Nat := 4096
    let gridY : Nat := (cfg.vocabSize + gridX - 1) / gridX
    if dp4aOn && useSubgroups && cfg.hiddenSize % 32 == 0 then
      let nQ8Blocks := cfg.hiddenSize / 32
      let q8BufBytes : USize := (nQ8Blocks * 9 * 4).toUSize
      let q8Buf ← match ← state.lmHeadQ8Buf.get with
        | some b => pure b
        | none =>
          let b ← GPUBackend.allocBuffer ctx q8BufBytes
          state.lmHeadQ8Buf.set (some b)
          pure b
      -- Fused finalNorm + Q8_1 quantize (one dispatch, no f32 normed buf).
      GPUBackend.executeWithConfigCached ctx
        (Hesper.Layers.RMSNorm.fusedRMSNormQ8_1Kernel model.finalNorm.config)
        [("input", state.buf2), ("scale", model.finalNorm.scale), ("output", q8Buf)]
        { numWorkgroups := (1, 1, 1), workgroupSize := { x := 256, y := 1, z := 1 }
          extensions := ["subgroups"] : Hesper.ExecConfig }
        (hash ("fused-rmsnorm-q8_1-lmhead", cfg.hiddenSize))
        state.lmHeadQuantizePrepared
      let quadCount := (cfg.vocabSize + 3) / 4
      let gridX4 : Nat := 4096
      let gridY4 : Nat := (quadCount + gridX4 - 1) / gridX4
      GPUBackend.executeWithConfigCached ctx
        (Hesper.Layers.Linear.fusedQ6KLinearDP4A4RowKernel
          cfg.hiddenSize cfg.vocabSize gridX4)
        [("weights", model.outputWeight), ("input_q8", q8Buf), ("output", state.logitsBuf)]
        { numWorkgroups := (gridX4, gridY4, 1), workgroupSize := { x := 128, y := 1, z := 1 }
          extensions := ["subgroups"] : Hesper.ExecConfig }
        (hash ("q6k-dp4a-lmhead-4row", cfg.hiddenSize, cfg.vocabSize))
        state.lmHeadDP4APrepared
      -- Golden dump: hesper's prefill logits (pre-softcap).  Matches
      -- llama.cpp's `result_output` top-5 token IDs (including argmax).
      dumpGolden "prefill_logits_raw" state.logitsBuf cfg.vocabSize
    else
      -- Fallback: f32 Q6_K kernel.  Needs standalone RMSNorm since the
      -- f32 matmul can't consume Q8_1.
      RMSNorm.forward ctx model.finalNorm state.buf2 state.buf1
      let shaderF32 := if useSubgroups then
          Hesper.Quantization.Q6_K.fusedQ6KLinearBlockCoopKernel
            cfg.hiddenSize cfg.vocabSize gridX
        else
          Hesper.Quantization.Q6_K.fusedQ6KLinearKernel
            cfg.hiddenSize cfg.vocabSize 256 gridX
      let wgSize := if useSubgroups then 32 else 256
      GPUBackend.execute ctx shaderF32
        [("weights", model.outputWeight), ("input", state.buf1), ("output", state.logitsBuf)]
        { numWorkgroups := (gridX, gridY, 1), workgroupSize := { x := wgSize, y := 1, z := 1 }
          extensions := if useSubgroups then ["subgroups"] else []
          : Hesper.ExecConfig }
  | _ =>
    -- Non-Q6_K fallback: F32 matmul transpose.  Needs standalone RMSNorm.
    RMSNorm.forward ctx model.finalNorm state.buf2 state.buf1
    let lmHeadConfig : Hesper.WGSL.MatMul.Config := {
      M := 1, N := cfg.vocabSize, K := cfg.hiddenSize
    }
    Hesper.WGSL.MatMul.executeMatMulTranspose ctx state.buf1 model.outputWeight state.logitsBuf lmHeadConfig

  -- ── Free prefill batch buffers ─────────────────────────────────────
  GPUBackend.freeBuffer ctx batchBuf1
  GPUBackend.freeBuffer ctx batchBuf2
  GPUBackend.freeBuffer ctx batchNormedBuf
  GPUBackend.freeBuffer ctx batchQBuf
  GPUBackend.freeBuffer ctx batchQRopedBuf
  GPUBackend.freeBuffer ctx onesBuf
  GPUBackend.freeBuffer ctx batchKBuf
  GPUBackend.freeBuffer ctx batchVBuf
  GPUBackend.freeBuffer ctx batchAttnOutBuf
  GPUBackend.freeBuffer ctx batchOProjBuf
  GPUBackend.freeBuffer ctx batchAttnResidBuf
  GPUBackend.freeBuffer ctx batchGateBuf
  GPUBackend.freeBuffer ctx batchUpBuf
  GPUBackend.freeBuffer ctx batchGeluBuf
  GPUBackend.freeBuffer ctx batchFFNOutBuf
  GPUBackend.freeBuffer ctx plGateBatchBuf
  GPUBackend.freeBuffer ctx plMoeOutBatchBuf
  GPUBackend.freeBuffer ctx plProjBatchBuf
  -- NOTE: tokenIdsBuf, posBuf, cacheLenBuf, colIdxBuf, batchPLInputAll
  -- are pooled on `state.prefill*Ref` — NOT freed here.  They live for
  -- the entire InferenceState lifetime.

/-- Run full single-token forward pass through the model.
    Returns logits in state.logitsBuf.

    `skipTokenWrite := true` skips the host-seeded
    `writeScalarViaStaging` for `state.tokenBuf` / `state.plRawRowBuf`.
    Token-graph mode sets this so argmaxKernel's device-side write
    from the previous step feeds the next step without host help. -/
def forwardSingleToken [GPUBackend β] (ctx : β)
    (model : Gemma4Model (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (tokenId : Nat) (pos : Nat)
    (state : InferenceState (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (kcr : Option (KernelCacheRefs (GPUBackend.CachedDispatch β)) := none)
    (skipTokenWrite : Bool := false)
    -- doc 58: when true, skip the writeScalarViaStaging of pos / cacheLen
    -- in forwardBlock.  Caller (e.g. generate's HESPER_DEVICE_FED loop)
    -- is responsible for advancing paramsBuf via advancePosKernel.
    (skipPosWrite : Bool := false) : IO Unit := do
  -- HESPER_FWD_SUBSECT_TRACE=1 prints a per-call breakdown of where the
  -- host CPU spends time inside this forward.  Doc 57 §3: total `forward`
  -- section is ~5 ms steady-state and runs sequential with the GPU drain
  -- on the previous token, so this is the budget we want to attribute
  -- before any restructuring.
  let fwdSubTrace := (← IO.getEnv "HESPER_FWD_SUBSECT_TRACE").isSome
  let tF0 ← if fwdSubTrace then IO.monoNanosNow else pure 0
  -- Step 1: Embedding lookup (format-dependent)
  -- Cached execute helper (same as forwardBlock's ce)
  let ce := fun (name : String) (shader : ShaderM Unit)
      (namedBufs : List (String × GPUBackend.Buf β)) (config : Hesper.ExecConfig) => do
    -- Key includes name + config (numWorkgroups, workgroupSize) to distinguish
    -- same-named kernels with different parameters (e.g., full vs SWA attention)
    match kcr with
    | some k =>
      let key := hash ("gemma4_ce", name, config.numWorkgroups, config.workgroupSize.x, config.workgroupSize.y, config.workgroupSize.z)
      let ref ← k.getRef key
      let configNamed : Hesper.ExecConfig := { config with funcName := name }
      GPUBackend.executeWithConfigCached ctx shader namedBufs configNamed key ref
    | none => GPUBackend.execute ctx shader namedBufs config
  let tokenBytes := Hesper.WebGPU.BufferOps.uint32ToBytes tokenId.toUInt32
  if !skipTokenWrite then
    writeScalarViaStaging ctx state.tokenBuf 0 state.stagingTokenPtr 0 tokenBytes
  Hesper.WGSL.Execute.withSection "embedLookup" do
    match model.embdFormat with
    | .Q6_K =>
      -- Q6_K on-the-fly dequant lookup
      ce "q6kEmbLookup"
        (Hesper.Quantization.Q6_K.q6kEmbeddingLookupKernel model.config.vocabSize model.config.hiddenSize)
        [("token_ids", state.tokenBuf), ("embedding_table", model.embedding.embeddingTable), ("output", state.buf1)]
        (.dispatch1D model.config.hiddenSize)
    | _ =>
      -- F32 / F16 / Q4_K: use existing Embedding.forward (assumes F32 interpretation)
      Embedding.forward ctx model.embedding state.tokenBuf state.buf1 1 1

  -- Scale embeddings by sqrt(hiddenSize)
  -- Cannot alias input/output in WebGPU, so output to buf2
  Hesper.WGSL.Execute.withSection "embedScale" do
    ce "embedScale"
      (embeddingScaleKernel model.config.hiddenSize model.config.hiddenSize)
      [("input", state.buf1), ("output", state.buf2)]
      (.dispatch1D model.config.hiddenSize)

  -- Step 1b: Per-layer input precomputation (gemma4-iswa.cpp:258-311)
  -- The per_layer_token_embd table is too large (>2 GB) for a single GPU buffer
  -- with the current Dawn limits, so we dequant just the input token's row on
  -- CPU and upload (~43 KB).
  Hesper.WGSL.Execute.withSection "perLayerInputPre" do
    match model.perLayerEmbdTableGPU, model.perLayerModelProj, model.perLayerProjNorm with
    | some embdTableGPU, some modelProj, some projNorm =>
      let embdPL := model.config.embdPerLayer
      let nLayers := model.config.numHiddenLayers
      let totalPL := embdPL * nLayers

      -- 1) Dequant the `tokenId` row from Q6_K table.
      --    Two paths:
      --      (a) Full table in VRAM: kernel reads tokenId-th row directly.
      --      (b) On-demand mmap: only the active row lives in `embdTableGPU`
      --          (sized rowBytes); we DMA mmap[tokenId * rowBytes ..] into
      --          it before each forward, and pass tokenId=0 to the kernel.
      Hesper.WGSL.Execute.withSection "plPre.gpuDequant" do
        let rowBytesU : USize := model.perLayerEmbdRowBytes.toUSize
        -- UVA on-demand path: kernel sees a synthesised row-base device
        -- pointer at `devPtr + tokenId × rowBytes`, so it dereferences
        -- the host page through the unified VA mapping with no explicit
        -- cuMemcpy. Legacy path: real tokenId indexes the full VRAM table.
        let kernelTokenId : Nat := match model.perLayerEmbdMmap with
          | some _ => 0
          | none   => tokenId
        let tokenIdBytes := Hesper.WebGPU.BufferOps.uint32ToBytes kernelTokenId.toUInt32
        if !skipTokenWrite then
          writeScalarViaStaging ctx state.plRawRowBuf 0 state.stagingPLRowPtr 0 tokenIdBytes
        let tableForKernel ← match model.perLayerEmbdMmap with
          | some (_, _, _, devPtr) =>
            let rowDevPtr : USize := devPtr + tokenId.toUSize * rowBytesU
            match ← GPUBackend.bufFromRawDevicePtr ctx rowDevPtr rowBytesU with
            | some b => pure b
            | none   => pure embdTableGPU
          | none => pure embdTableGPU
        let scaleFactor : Float := Float.sqrt embdPL.toFloat
        ce "q6kDequantScale"
          (Hesper.Quantization.Q6_K.q6kTableRowDequantScaleKernel totalPL scaleFactor
            model.config.vocabSize)
          [("table", tableForKernel), ("params", state.plRawRowBuf), ("output", state.plModelProj)]
          (.dispatch1D totalPL)

      -- 2) per_layer_model_proj @ buf2 → plTokenSelected
      let projConfig : Hesper.WGSL.MatMul.Config := {
        M := 1, N := totalPL, K := model.config.hiddenSize
      }
      Hesper.WGSL.Execute.withSection "plPre.f16Matmul" do
        if projConfig.K % 64 == 0 then
          Hesper.WGSL.MatMul.executeMatMulTransposeF16BlockCoop ctx state.buf2 modelProj state.plTokenSelected projConfig
        else
          Hesper.WGSL.MatMul.executeMatMulTransposeF16 ctx state.buf2 modelProj state.plTokenSelected projConfig

      -- Fused (scale + chunked RMSNorm + scaledAdd) — one dispatch over
      -- totalPL elements.  The kernel reads `plTokenSelected` (f16Matmul
      -- output, un-scaled), absorbs the (1/√hidden) pre-norm scale via
      -- `scaleSq` in the variance computation, applies the norm + weight,
      -- then adds `plModelProj` scaled by 1/√2.  Replaces three separate
      -- dispatches (pleScalePL, chunkedRMSNorm, scaledAdd) with one.
      Hesper.WGSL.Execute.withSection "plPre.fusedNormAdd" do
        let preScale : Float := 1.0 / Float.sqrt model.config.hiddenSize.toFloat
        let addScale : Float := 1.0 / Float.sqrt 2.0
        ce "plFusedNormAdd"
          (chunkedRMSNormAddScaledKernel embdPL nLayers model.config.rmsNormEps preScale addScale)
          [("input", state.plTokenSelected), ("weight", projNorm.scale),
           ("residual", state.plModelProj), ("output", state.plInputAll)]
          { numWorkgroups := (nLayers, 1, 1), workgroupSize := { x := min embdPL 256, y := 1, z := 1 } : Hesper.ExecConfig }
    | _, _, _ => pure ()

  let tF1_PrePLE ← if fwdSubTrace then IO.monoNanosNow else pure 0

  -- Step 2: Process all transformer blocks (starting from buf2 as current).
  -- If the caller has already started a batch (e.g. a batched prefill
  -- wrapper), we nest inside it instead of starting a new one. Callers that
  -- want per-dispatch GPU timing (e.g. `gemma4-profile`) can flip
  -- `Hesper.Layers.Linear.profilingRef` to `true`, in which case we also
  -- skip our own begin/endBatch so each dispatch auto-syncs.
  let profiling ← Hesper.Layers.Linear.profilingRef.get
  let alreadyBatching ← Hesper.WGSL.Execute.isBatching
  let ownBatch := !profiling && !alreadyBatching
  if ownBatch then GPUBackend.beginBatch ctx

  let mut currentBuf := state.buf2
  let mut nextBuf := state.buf1

  dumpBuf ctx currentBuf (model.config.hiddenSize * 4).toUSize s!"single_p{pos}_postPLE"

  let mut blockIdx := 0
  let skipPLE ← do pure (← IO.getEnv "HESPER_SKIP_PLE").isSome
  let plInputBuf := if model.config.hasPerLayerEmbeddings then some state.plInputAll else none
  for block in model.blocks do
    let plEmbd := if blockIdx < model.perLayerBlocks.size && !skipPLE then
      model.perLayerBlocks[blockIdx]!
    else none
    forwardBlock ctx block model.config currentBuf nextBuf state pos
      (kcr := kcr) (perLayerEmbd := plEmbd) (perLayerInput := plInputBuf)
      (skipPosWrite := skipPosWrite)
    let oldCb := currentBuf
    currentBuf := nextBuf
    nextBuf := oldCb
    dumpBuf ctx currentBuf (model.config.hiddenSize * 4).toUSize s!"single_p{pos}_afterL{blockIdx}"
    blockIdx := blockIdx + 1

  let tF2_BlocksEnd ← if fwdSubTrace then IO.monoNanosNow else pure 0

  -- Step 3: Final norm.  When the Q6_K dp4a lm_head path is available
  -- (Gemma 4's default for embdFormat=Q6_K with dp4a enabled), we defer
  -- emission until lm_head so we can fuse finalNorm + Q8_1 quantize
  -- into one kernel (`fusedRMSNormQ8_1Kernel`).  Otherwise emit the
  -- standalone Circuit-DSL norm so the f32 matmul fallback has a
  -- valid `nextBuf` to read.
  let useFusedNormLmHead ← do
    -- The pre-dequantised f16 lm_head path consumes a finalNorm-output
    -- f32 vector directly — it does not benefit from fusing finalNorm
    -- with Q8_1 quantize.  Use the standalone finalNorm path in that
    -- case so `nextBuf` holds the normed input for `executeMatMulTransposeF16BlockCoop`.
    if model.outputWeightF16.isSome then pure false
    else match model.embdFormat with
    | .Q6_K =>
      let useSubgroups ← GPUBackend.hasSubgroupSupport ctx
      let a ← Hesper.Layers.Linear.dp4aEnabled.get
      let b ← Hesper.Layers.Linear.dp4aQ6KEnabled.get
      pure (a && b && useSubgroups && model.config.hiddenSize % 32 == 0)
    | _ => pure false
  if !useFusedNormLmHead then
    Hesper.WGSL.Execute.withSection "finalNorm" do
      let key := hash ("circuitFinalNorm-cuda", model.finalNorm.config.dim)
      let ccRef ← Hesper.Circuit.getGlobalCircuitRef (β := β) key
      Hesper.Circuit.runCachedFused ctx ccRef
        (do
          let xT ← Hesper.Circuit.CircuitM.registerExternal
            (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
            currentBuf #[model.finalNorm.config.dim] .f32 .Global
          let sT ← Hesper.Circuit.CircuitM.registerExternal
            (BufT := GPUBackend.Buf β) (CacheT := GPUBackend.CachedDispatch β)
            model.finalNorm.scale #[model.finalNorm.config.dim] .f32 .Global
          let _y ← Hesper.Circuit.CircuitM.rmsNorm xT sT model.finalNorm.config.eps
          pure ())
        [(0, currentBuf), (1, model.finalNorm.scale), (5, nextBuf)]

  -- Step 4: LM head matmul (1 × hiddenSize @ hiddenSize × vocabSize)
  Hesper.WGSL.Execute.withSection "lmHead" do
    -- Fast path: pre-dequantized f16 weights → f16 matmul (single dispatch,
    -- no Q8_1 quantize, ~10× speedup over Q6_K dp4a for vocab=262144).
    -- HESPER_DP4A_Q6K_4WARP=1 forces the dp4a 4-warp path even when the
    -- f16 buffer exists — useful for A/B against the f16 fast path.
    let force4WarpQ6K ← do
      match ← IO.getEnv "HESPER_DP4A_Q6K_4WARP" with
      | some "1" => pure true
      | _        => pure false
    match (if force4WarpQ6K then none else model.outputWeightF16) with
    | some f16W =>
      let lmHeadConfig : Hesper.WGSL.MatMul.Config := {
        M := 1, N := model.config.vocabSize, K := model.config.hiddenSize
      }
      Hesper.WGSL.MatMul.executeMatMulTransposeF16BlockCoop ctx nextBuf f16W state.logitsBuf lmHeadConfig
    | none =>
    match model.embdFormat with
    | .Q6_K =>
      let useSubgroups ← GPUBackend.hasSubgroupSupport ctx
      let dp4aOn ← do
        let a ← Hesper.Layers.Linear.dp4aEnabled.get
        let b ← Hesper.Layers.Linear.dp4aQ6KEnabled.get
        pure (a && b)
      let gridX : Nat := 4096
      let gridY : Nat := (model.config.vocabSize + gridX - 1) / gridX
      -- DEBUG: when HESPER_DP4A_Q6K_DEBUG is set, run BOTH the f32 and
      -- the dp4a kernel and dump first 8 logits of each for comparison.
      let debugMode ← do
        match ← IO.getEnv "HESPER_DP4A_Q6K_DEBUG" with
        | some "1" => pure true
        | _ => pure false
      if debugMode && dp4aOn then
        IO.println "[Q6K_DEBUG] Running f32 lmHead for comparison..."
        let shaderF32 := if useSubgroups then
            Hesper.Quantization.Q6_K.fusedQ6KLinearBlockCoopKernel
              model.config.hiddenSize model.config.vocabSize gridX
          else
            Hesper.Quantization.Q6_K.fusedQ6KLinearKernel
              model.config.hiddenSize model.config.vocabSize 256 gridX
        let wgSize := if useSubgroups then 32 else 256
        GPUBackend.execute ctx shaderF32
          [("weights", model.outputWeight), ("input", nextBuf), ("output", state.logitsBuf)]
          { numWorkgroups := (gridX, gridY, 1), workgroupSize := { x := wgSize, y := 1, z := 1 }
            extensions := if useSubgroups then ["subgroups"] else []
            : Hesper.ExecConfig }
        let logitsBytes ← GPUBackend.readBuffer ctx state.logitsBuf (132 * 4 : USize)
        IO.print "[Q6K_DEBUG] f32  logits[5000..5007]: "
        for i in [100:108] do
          let o := i * 4
          let bits := (logitsBytes.get! o).toUInt32 |||
                      ((logitsBytes.get! (o+1)).toUInt32 <<< 8) |||
                      ((logitsBytes.get! (o+2)).toUInt32 <<< 16) |||
                      ((logitsBytes.get! (o+3)).toUInt32 <<< 24)
          let e := (bits >>> 23) &&& 0xFF
          let m := bits &&& (0x7FFFFF : UInt32)
          let s := bits >>> 31
          let v := if e == 0 then 0.0 else
            (1.0 + m.toNat.toFloat / 8388608.0) * Float.pow 2.0 (e.toNat.toFloat - 127.0)
          let v := if s == 1 then -v else v
          IO.print s!"{v} "
        IO.println ""

      let _ := debugMode
      if dp4aOn && useSubgroups && model.config.hiddenSize % 32 == 0 then
        -- dp4a path: fused finalNorm+Q8_1 quantize, then Q6_K × Q8_1 matmul.
        -- The fused kernel consumes the pre-norm hidden state directly
        -- from `currentBuf`, so we deliberately skipped the standalone
        -- finalNorm above (see `useFusedNormLmHead`).  Saves 1 dispatch
        -- and the f32 normed round-trip through `nextBuf`.
        let nQ8Blocks := model.config.hiddenSize / 32
        let q8BufBytes : USize := (nQ8Blocks * 9 * 4).toUSize
        let q8Buf ← match ← state.lmHeadQ8Buf.get with
          | some b => pure b
          | none =>
            let b ← GPUBackend.allocBuffer ctx q8BufBytes
            state.lmHeadQ8Buf.set (some b)
            pure b
        -- Fused finalNorm + Q8_1 quantize.
        GPUBackend.executeWithConfigCached ctx
          (Hesper.Layers.RMSNorm.fusedRMSNormQ8_1Kernel model.finalNorm.config)
          [("input", currentBuf), ("scale", model.finalNorm.scale), ("output", q8Buf)]
          { numWorkgroups := (1, 1, 1), workgroupSize := { x := 256, y := 1, z := 1 }
            extensions := ["subgroups"] : Hesper.ExecConfig }
          (hash ("fused-rmsnorm-q8_1-lmhead", model.config.hiddenSize))
          state.lmHeadQuantizePrepared
        -- Q6_K dp4a matmul (2D grid for vocabSize > 65535).
        -- Variant selection (env, in priority order):
        --   HESPER_DP4A_Q6K_4WARP=1 → 4-warp 1-row coop on K (llama.cpp shape)
        --   HESPER_DP4A_Q6K_1ROW=1  → single-warp     (32 threads, 1 row/WG)
        --   HESPER_DP4A_Q6K_2ROW=1  → 2-warp variant  (64 threads, 2 rows/WG)
        --   default                 → 4-warp 4-rows-per-WG (smem input share)
        let variant ← do
          match ← IO.getEnv "HESPER_DP4A_Q6K_4WARP" with
          | some "1" => pure "4warp"
          | _ =>
            match ← IO.getEnv "HESPER_DP4A_Q6K_1ROW" with
            | some "1" => pure "1row"
            | _ =>
              match ← IO.getEnv "HESPER_DP4A_Q6K_2ROW" with
              | some "1" => pure "2row"
              | _ => pure "4row"
        match variant with
        | "4warp" =>
          -- 1 row per WG, 4 warps cooperate over K.  grid = vocabSize rows.
          -- Use 2-D grid (gridX × gridY) to stay under WebGPU 65535 dim cap.
          let gridX1 : Nat := 4096
          let gridY1 : Nat := (model.config.vocabSize + gridX1 - 1) / gridX1
          GPUBackend.executeWithConfigCached ctx
            (Hesper.Layers.Linear.fusedQ6KLinearDP4A4WarpKernel
              model.config.hiddenSize model.config.vocabSize gridX1)
            [("weights", model.outputWeight), ("input_q8", q8Buf), ("output", state.logitsBuf)]
            { numWorkgroups := (gridX1, gridY1, 1), workgroupSize := { x := 32, y := 4, z := 1 }
              extensions := ["subgroups"]
              -- Distinguish from the Linear dispatcher's 4-warp emit
              -- (same ShaderM, different shape) — see preHash collision
              -- note in Linear.lean's force4Warp branch.
              funcName := s!"q6k_dp4a_4warp_lmhead_{model.config.hiddenSize}_{model.config.vocabSize}"
              : Hesper.ExecConfig }
            (hash ("q6k-dp4a-lmhead-4warp", model.config.hiddenSize, model.config.vocabSize))
            state.lmHeadDP4APrepared
        | "4row" =>
          let quadCount := (model.config.vocabSize + 3) / 4
          let gridX4 : Nat := 4096
          let gridY4 : Nat := (quadCount + gridX4 - 1) / gridX4
          GPUBackend.executeWithConfigCached ctx
            (Hesper.Layers.Linear.fusedQ6KLinearDP4A4RowKernel
              model.config.hiddenSize model.config.vocabSize gridX4)
            [("weights", model.outputWeight), ("input_q8", q8Buf), ("output", state.logitsBuf)]
            { numWorkgroups := (gridX4, gridY4, 1), workgroupSize := { x := 128, y := 1, z := 1 }
              extensions := ["subgroups"] : Hesper.ExecConfig }
            (hash ("q6k-dp4a-lmhead-4row", model.config.hiddenSize, model.config.vocabSize))
            state.lmHeadDP4APrepared
        | "2row" =>
          let pairCount := (model.config.vocabSize + 1) / 2
          let gridX2 : Nat := 4096
          let gridY2 : Nat := (pairCount + gridX2 - 1) / gridX2
          GPUBackend.executeWithConfigCached ctx
            (Hesper.Layers.Linear.fusedQ6KLinearDP4A2RowKernel
              model.config.hiddenSize model.config.vocabSize gridX2)
            [("weights", model.outputWeight), ("input_q8", q8Buf), ("output", state.logitsBuf)]
            { numWorkgroups := (gridX2, gridY2, 1), workgroupSize := { x := 64, y := 1, z := 1 }
              extensions := ["subgroups"] : Hesper.ExecConfig }
            (hash ("q6k-dp4a-lmhead-2row", model.config.hiddenSize, model.config.vocabSize))
            state.lmHeadDP4APrepared
        | _ =>
          GPUBackend.executeWithConfigCached ctx
            (Hesper.Layers.Linear.fusedQ6KLinearDP4AKernel
              model.config.hiddenSize model.config.vocabSize gridX)
            [("weights", model.outputWeight), ("input_q8", q8Buf), ("output", state.logitsBuf)]
            { numWorkgroups := (gridX, gridY, 1), workgroupSize := { x := 32, y := 1, z := 1 }
              extensions := ["subgroups"] : Hesper.ExecConfig }
            (hash ("q6k-dp4a-lmhead", model.config.hiddenSize, model.config.vocabSize))
            state.lmHeadDP4APrepared
        if debugMode then
          -- Read logits[5000..5007] (skip reserved tokens which are often 0)
          let logitsBytes ← GPUBackend.readBuffer ctx state.logitsBuf (5008 * 4 : USize)
          IO.print "[Q6K_DEBUG] dp4a logits[5000..5007]: "
          for i in [5000:5008] do
            let o := i * 4
            let bits := (logitsBytes.get! o).toUInt32 |||
                        ((logitsBytes.get! (o+1)).toUInt32 <<< 8) |||
                        ((logitsBytes.get! (o+2)).toUInt32 <<< 16) |||
                        ((logitsBytes.get! (o+3)).toUInt32 <<< 24)
            let e := (bits >>> 23) &&& 0xFF
            let m := bits &&& (0x7FFFFF : UInt32)
            let s := bits >>> 31
            let v := if e == 0 then 0.0 else
              (1.0 + m.toNat.toFloat / 8388608.0) * Float.pow 2.0 (e.toNat.toFloat - 127.0)
            let v := if s == 1 then -v else v
            IO.print s!"{v} "
          IO.println ""
      else
        -- Original f32 path (block-coop with subgroups or 256-thread tree reduction).
        let shader := if useSubgroups then
            Hesper.Quantization.Q6_K.fusedQ6KLinearBlockCoopKernel
              model.config.hiddenSize model.config.vocabSize gridX
          else
            Hesper.Quantization.Q6_K.fusedQ6KLinearKernel
              model.config.hiddenSize model.config.vocabSize 256 gridX
        let wgSize := if useSubgroups then 32 else 256
        let lmBufs : List (String × GPUBackend.Buf β) :=
          [("weights", model.outputWeight), ("input", state.buf1), ("output", state.logitsBuf)]
        ce "lmHead"
          shader
          lmBufs
          { numWorkgroups := (gridX, gridY, 1)
            workgroupSize := { x := wgSize, y := 1, z := 1 }
            extensions := if useSubgroups then ["subgroups"] else []
            : Hesper.ExecConfig }
    | _ =>
      let lmHeadConfig : Hesper.WGSL.MatMul.Config := {
        M := 1, N := model.config.vocabSize, K := model.config.hiddenSize
      }
      Hesper.WGSL.MatMul.executeMatMulTranspose ctx nextBuf model.outputWeight state.logitsBuf lmHeadConfig

  -- Optional: save pre-softcap logits for TTT surprise sensor.
  -- Only runs when preSoftcapBuf is set (zero cost otherwise).
  match state.preSoftcapBuf with
  | some psBuf =>
    ce "pleScaleVocab"
      (PerLayerEmbedding.scaleKernel model.config.vocabSize 1.0)
      [("input", state.logitsBuf), ("output", psBuf)]
      (.dispatch1D model.config.vocabSize)
  | none => pure ()

  -- Step 5: Logit softcapping (y = scale * tanh(x / scale))
  Hesper.WGSL.Execute.withSection "logitSoftcap" do
    if model.config.logitSoftcapScale > 0.0 then
      ce "logitSoftcap"
        (logitSoftcapKernel model.config.vocabSize model.config.logitSoftcapScale)
        [("input", state.logitsBuf), ("output", state.logitsBuf2)]
        (.dispatch1D model.config.vocabSize)
      ce "pleScaleVocab2"
        (PerLayerEmbedding.scaleKernel model.config.vocabSize 1.0)
        [("input", state.logitsBuf2), ("output", state.logitsBuf)]
        (.dispatch1D model.config.vocabSize)

  if ownBatch then GPUBackend.endBatch ctx

  if fwdSubTrace then
    let tF3 ← IO.monoNanosNow
    let p := (tF1_PrePLE - tF0).toFloat / 1e6
    let b := (tF2_BlocksEnd - tF1_PrePLE).toFloat / 1e6
    let q := (tF3 - tF2_BlocksEnd).toFloat / 1e6
    IO.println s!"[fwdsub] prePLE={p} blocks={b} post={q}"

/-! ## Text Generation -/

/-- GPU argmax: parallel reduction to find token with highest logit.

    Writes maxIdx into `result[historySlot]` for host consumption, and
    mirrors it into both `token[0]` (the single-token lookup buffer
    used by forwardSingleToken) and `token_ids[0]` (the multi-token
    lookup buffer used by forwardPrefillBatch when seqLen=1).  The
    extra writes let the captured decode graph feed the next
    iteration's embedding lookup with no host round-trip.

    `historySlot` is baked into the kernel at ShaderM compile time so
    multiple argmaxKernel instances in the same captured graph write
    to consecutive slots of the same `result` buffer, giving the host
    the full decode history after one graph launch. -/
private def argmaxKernel (vocabSize : Nat) (historySlot : Nat := 0) : ShaderM Unit := do
  let tid ← ShaderM.localId
  let tid := Exp.vec3X tid
  ShaderM.sharedNamed "shared_vals" (.array (.scalar .f32) 256)
  ShaderM.sharedNamed "shared_idxs" (.array (.scalar .u32) 256)
  let _logits ← ShaderM.declareInputBuffer "logits" (.array (.scalar .f32) vocabSize)
  -- `result` is sized generously so a single kernel can hit any slot.
  -- At call time we bind a host-visible history buffer (>= historySlot+1).
  let _result ← ShaderM.declareOutputBuffer "result" (.array (.scalar .u32) (historySlot + 1))
  let _token     ← ShaderM.declareOutputBuffer "token"     (.array (.scalar .u32) 1)
  let _tokenIds  ← ShaderM.declareOutputBuffer "token_ids" (.array (.scalar .u32) 1)
  -- Per-layer embedding row selector: the PLE dequant kernel reads
  -- `plRawRow[0]` as the token index, so token-graph replay needs the
  -- argmax output mirrored here too.
  let _plRawRow  ← ShaderM.declareOutputBuffer "plRawRow" (.array (.scalar .u32) 1)
  ShaderM.varNamed "local_max" (.scalar .f32) (Exp.litF32 (-1.0e38))
  ShaderM.varNamed "local_idx" (.scalar .u32) (Exp.litU32 0)
  ShaderM.loop tid (Exp.litU32 vocabSize) (Exp.litU32 256) fun i => do
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := vocabSize) "logits" i
    ShaderM.if_ (Exp.gt val (Exp.var "local_max")) (do
      ShaderM.assign "local_max" val
      ShaderM.assign "local_idx" i
    ) (pure ())
  ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_vals" tid (Exp.var "local_max")
  ShaderM.writeWorkgroup (ty := .scalar .u32) "shared_idxs" tid (Exp.var "local_idx")
  ShaderM.barrier
  let mut stride := 128
  while stride > 0 do
    ShaderM.if_ (Exp.lt tid (Exp.litU32 stride)) (do
      let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 256) "shared_vals" tid
      let b ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := 256) "shared_vals" (Exp.add tid (Exp.litU32 stride))
      ShaderM.if_ (Exp.gt b a) (do
        ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_vals" tid b
        let bIdx ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := 256) "shared_idxs" (Exp.add tid (Exp.litU32 stride))
        ShaderM.writeWorkgroup (ty := .scalar .u32) "shared_idxs" tid bIdx
      ) (pure ())
    ) (pure ())
    ShaderM.barrier
    stride := stride / 2
  ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
    let maxIdx ← ShaderM.readWorkgroup (ty := .scalar .u32) (n := 256) "shared_idxs" (Exp.litU32 0)
    ShaderM.writeBuffer (ty := .scalar .u32) "result"    (Exp.litU32 historySlot) maxIdx
    ShaderM.writeBuffer (ty := .scalar .u32) "token"     (Exp.litU32 0) maxIdx
    ShaderM.writeBuffer (ty := .scalar .u32) "token_ids" (Exp.litU32 0) maxIdx
    ShaderM.writeBuffer (ty := .scalar .u32) "plRawRow"  (Exp.litU32 0) maxIdx
  ) (pure ())

/-- Device-side history append: copy `src[0]` into `dst[slot[0]]`.
    Used in HESPER_DEVICE_FED decode to record argmax outputs in
    `state.argmaxHistoryBuf` without host sync.  Caller must ensure
    `slot[0]` is a valid index < dst size; we set `slot[0] = pos -
    promptLen` via a host-side initial seed + advancePos increments. -/
private def historyAppendKernel : ShaderM Unit := do
  let tid ← ShaderM.localId
  let tid := Exp.vec3X tid
  let _src  ← ShaderM.declareInputBuffer  "src"  (.array (.scalar .u32) 1)
  let _slot ← ShaderM.declareOutputBuffer "slot" (.array (.scalar .u32) 1)
  let _dst  ← ShaderM.declareOutputBuffer "dst"  (.array (.scalar .u32) 65536)
  ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
    let v ← ShaderM.readBuffer (ty := .scalar .u32) (n := 1) "src" (Exp.litU32 0)
    let s ← ShaderM.readBuffer (ty := .scalar .u32) (n := 1) "slot" (Exp.litU32 0)
    ShaderM.writeBuffer (ty := .scalar .u32) "dst" s v
    ShaderM.writeBuffer (ty := .scalar .u32) "slot" (Exp.litU32 0) (Exp.add s (Exp.litU32 1))
  ) (pure ())

/-- Device-side pos/cacheLen/posF32 advance.
    One thread increments all three scalars in place so the next
    replay of the captured decode graph consumes position `pos+1`
    without any host round-trip.

    Bindings:
      params : u32 × 2   — `[pos, cacheLen]`
      posF32 : f32 × 1   — same value as `pos`, used by RoPE lookups -/
private def advancePosKernel : ShaderM Unit := do
  let tid ← ShaderM.localId
  let tid := Exp.vec3X tid
  let _params ← ShaderM.declareOutputBuffer "params" (.array (.scalar .u32) 2)
  let _posF32 ← ShaderM.declareOutputBuffer "posF32" (.array (.scalar .f32) 1)
  ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
    let p ← ShaderM.readBuffer (ty := .scalar .u32) (n := 2) "params" (Exp.litU32 0)
    let c ← ShaderM.readBuffer (ty := .scalar .u32) (n := 2) "params" (Exp.litU32 1)
    let pNew := Exp.add p (Exp.litU32 1)
    let cNew := Exp.add c (Exp.litU32 1)
    ShaderM.writeBuffer (ty := .scalar .u32) "params" (Exp.litU32 0) pNew
    ShaderM.writeBuffer (ty := .scalar .u32) "params" (Exp.litU32 1) cNew
    ShaderM.writeBuffer (ty := .scalar .f32) "posF32" (Exp.litU32 0) (Exp.toF32 pNew)
  ) (pure ())

private def gpuArgmax [GPUBackend β] (ctx : β)
    (logitsBuf argmaxBuf tokenBuf tokenIdsBuf plRawRowBuf : GPUBackend.Buf β)
    (vocabSize : Nat)
    (cacheRef : IO.Ref (Option (GPUBackend.CachedDispatch β)))
    (hostMapped : Option (USize × GPUBackend.Buf β) := none)
    -- doc 58: when true, fire the argmax kernel but DO NOT sync the
    -- stream / read the result.  Returns 0.  Use this when the host
    -- doesn't need nextToken yet (token-feed is device-side via
    -- tokenBuf) and only needs to read it later for the EOS / output
    -- batch.  Closes the argmax→next-forward GPU idle bubble.
    (deferRead : Bool := false) : IO Nat := do
  -- Pick the result buffer.  When `hostMapped` is set, the kernel writes
  -- straight into pinned host memory we can read with no driver call;
  -- otherwise fall back to the legacy device buffer + cuMemcpyDtoH path.
  let resultBuf := match hostMapped with
    | some (_, devBuf) => devBuf
    | none => argmaxBuf
  GPUBackend.executeWithConfigCached ctx (argmaxKernel vocabSize)
    [ ("logits", logitsBuf), ("result", resultBuf)
    , ("token", tokenBuf), ("token_ids", tokenIdsBuf)
    , ("plRawRow", plRawRowBuf) ]
    { workgroupSize := { x := 256 }, numWorkgroups := (1, 1, 1) }
    (hash ("argmaxKernel", vocabSize)) cacheRef
  if deferRead then
    return 0
  match hostMapped with
  | some (hostPtr, _) =>
    -- Drain the stream so the kernel's `st.global` to the host-mapped
    -- slot is visible to host loads.  This is the same wait that the
    -- legacy `cuMemcpyDtoH(4 byte)` performs implicitly, but accounted
    -- against `cuStreamSynchronize` in nsys traces (matches llama-cli's
    -- shape) and freed of the per-call driver allocation that
    -- cuMemcpyDtoH does.
    Hesper.CUDA.cuStreamSynchronize (0 : USize)
    let v ← Hesper.CUDA.cuReadPinnedU32 hostPtr
    return v.toNat
  | none =>
    let bytes ← GPUBackend.readBuffer ctx argmaxBuf (4 : USize)
    return (Hesper.Basic.bytesToUInt32 bytes 0).toNat

set_option maxHeartbeats 800000 in
/-- Generate tokens from a Gemma 4 model.

    @param device WebGPU device
    @param model Loaded Gemma 4 model
    @param promptTokens Input token IDs
    @param maxTokens Maximum new tokens to generate
    @param eosToken Optional EOS token ID for early stopping
-/
def generate [GPUBackend β] (ctx : β) (model : Gemma4Model (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (promptTokens : Array Nat) (maxTokens : Nat)
    (eosToken : Option Nat := none)
    (extraEosTokens : Array Nat := #[]) : IO (Array Nat) := do
  IO.println s!"[Gemma4] Generating: {promptTokens.size} prompt tokens, max {maxTokens} new tokens"

  -- Create inference state + kernel cache refs.  `kcr` is used by prefill
  -- (seqLen=N), `unifiedKcr` by unified decode (seqLen=1).  They must NOT
  -- share — some `ce` cacheKeys inside `forwardPrefillBatch` don't include
  -- `seqLen`, so a cached prefill dispatch would fire at decode shape and
  -- launch with (18,1,1) workgroups for a (1,1,1)-sized buffer batch → crash
  -- or silent corruption.
  let state ← createInferenceState ctx model.config
  let kcr ← createKernelCacheRefs (β := β)
  let unifiedKcr ← createKernelCacheRefs (β := β)

  let mut tokens := promptTokens

  -- Phase 1: Prefill (process prompt tokens)
  IO.println s!"[Prefill] Processing {promptTokens.size} prompt tokens..."
  let prefillStart ← IO.monoNanosNow
  -- HESPER_BATCH_PREFILL_FORCE=1 uses the batched path even for N=1
  -- prompts (test vehicle for Phase 3 unified decode).
  let forceBatch := (← IO.getEnv "HESPER_BATCH_PREFILL_FORCE").isSome
  let useBatch := (promptTokens.size > 1 || forceBatch)
    && (match ← IO.getEnv "HESPER_BATCH_PREFILL" with | some "0" => false | _ => true)
  if useBatch then
    IO.println s!"[Prefill] Batched path (seqLen={promptTokens.size})"
    forwardPrefillBatch ctx model promptTokens state (kcr := some kcr)
    -- HESPER_PREFILL_TWICE=1: run prefill a second time with the same args
    -- to test whether forwardPrefillBatch is idempotent on shared state.
    if (← IO.getEnv "HESPER_PREFILL_TWICE").isSome then
      IO.println "[Prefill] TWICE mode: calling forwardPrefillBatch again with same args"
      forwardPrefillBatch ctx model promptTokens state (kcr := some kcr)
  else
    for i in [0:promptTokens.size] do
      if i >= model.config.maxSeqLen then break
      forwardSingleToken ctx model promptTokens[i]! i state (kcr := some kcr)
  let prefillEnd ← IO.monoNanosNow
  let prefillMs := (prefillEnd - prefillStart).toFloat / 1_000_000.0
  IO.println s!"[Prefill] Done in {prefillMs} ms"

  -- Phase 2: Decode (generate new tokens)
  -- Reset alloc counter + module-load timer so prefill isn't included
  -- in the decode-only histogram.  `HESPER_ALLOC_TRACE=1` enables
  -- recording (both alloc and module-load).  Printed at end of generate.
  Hesper.resetAllocCounter
  Hesper.resetModuleLoadTimer
  Hesper.resetExecuteImplTimer
  let genStart ← IO.monoNanosNow
  let mut genCount : Nat := 0

  -- CUDA Graph capture+replay: env-gated experimental path.
  -- When HESPER_CUDA_GRAPHS=1, we run the FIRST decode forward pass
  -- inside a relaxed stream capture, harvest the resulting graph, and
  -- replay it on subsequent tokens.  Host-side writes (tokenId, pos)
  -- happen BEFORE each replay via default-stream cuMemcpyHtoD — those
  -- are sync to host and effectively "bake in" at execute time.
  --
  -- IMPORTANT: this relies on the decode forward being shape-stable —
  -- true for Gemma 4's single-token path since all kernels have
  -- compile-time-fixed dispatch shapes (one per-layer set).
  -- CUDA Graphs are now ON by default — empirically +20% TPS at decode time
  -- (memory: project_q6k_lmhead_f16_landed.md, project_115tps_root_cause_2026_04_28.md).
  -- Override with HESPER_CUDA_GRAPHS=0 to opt out (e.g. when debugging dispatch-by-dispatch).
  let useCudaGraphs := match ← IO.getEnv "HESPER_CUDA_GRAPHS" with
    | some "0" => false
    | _        => true
  let tokenGraph   := (← IO.getEnv "HESPER_TOKEN_GRAPH").isSome
  let pipelinedDecode := (← IO.getEnv "HESPER_PIPELINED_DECODE").isSome

  let mut graphExecOpt : Option (Hesper.CUDA.CUgraphExec × Hesper.CUDA.CUstream) := none

  let dispCountEnabled := (← IO.getEnv "HESPER_DISPATCH_COUNT").isSome
  if dispCountEnabled then
    Hesper.WGSL.Execute.sectionProfilingRef.set true

  -- HESPER_TOKEN_GRAPH=1: capture the entire N-token decode loop as a
  -- single CUDA graph and launch it once.  Requires unifiedDecode and
  -- CUDA graphs.  EOS checks are skipped in favour of generating the
  -- full maxTokens (host trims after).
  if tokenGraph && useCudaGraphs then do
    -- 1-shot history buffer: maxTokens u32 slots.
    let historyBuf ← GPUBackend.allocBuffer ctx (maxTokens * 4).toUSize
    -- Pre-capture: put the initial pos / cacheLen / posF32 into the
    -- state buffers via a normal (outside-capture) write so the first
    -- captured kernel invocation reads pos = promptTokens.size.
    let startPos := promptTokens.size
    let posBytes     := Hesper.WebGPU.BufferOps.uint32ToBytes startPos.toUInt32
    let cacheLenBytes := Hesper.WebGPU.BufferOps.uint32ToBytes (startPos + 1).toUInt32
    let posF32Bytes  ← Hesper.Basic.floatToBytes startPos.toFloat
    GPUBackend.writeBufferOffset ctx state.paramsBuf 0 posBytes
    GPUBackend.writeBufferOffset ctx state.paramsBuf 4 cacheLenBytes
    GPUBackend.writeBufferOffset ctx state.posF32Buf 0 posF32Bytes
    -- Pre-capture: seed tokenBuf / plRawRowBuf with the last prefill
    -- token so step 0's embedding lookup picks it up.
    let lastPrefill := promptTokens[promptTokens.size - 1]!
    let lpBytes := Hesper.WebGPU.BufferOps.uint32ToBytes lastPrefill.toUInt32
    GPUBackend.writeBufferOffset ctx state.tokenBuf     0 lpBytes
    GPUBackend.writeBufferOffset ctx state.plRawRowBuf  0 lpBytes

    let stream ← Hesper.CUDA.cuStreamCreate
    Hesper.cudaCaptureStream.set (some stream)
    Hesper.CUDA.cuStreamBeginCapture stream
    skipStagingWrites.set true
    let dispStart ← Hesper.dispatchCounter.get
    for k in [0:maxTokens] do
      let dispBeforeStep ← Hesper.dispatchCounter.get
      forwardSingleToken ctx model 0 (startPos + k) state
        (kcr := some kcr) (skipTokenWrite := true)
      GPUBackend.execute ctx (argmaxKernel model.config.vocabSize k)
        [ ("logits", state.logitsBuf), ("result", historyBuf)
        , ("token", state.tokenBuf), ("token_ids", state.tokenBuf)
        , ("plRawRow", state.plRawRowBuf) ]
        { workgroupSize := { x := 256 }, numWorkgroups := (1, 1, 1) }
      GPUBackend.execute ctx advancePosKernel
        [ ("params", state.paramsBuf), ("posF32", state.posF32Buf) ]
        { workgroupSize := { x := 1 }, numWorkgroups := (1, 1, 1) }
      let dispAfterStep ← Hesper.dispatchCounter.get
      IO.println s!"[token-graph] step {k}: {dispAfterStep - dispBeforeStep} dispatches recorded"
    let dispEnd ← Hesper.dispatchCounter.get
    IO.println s!"[token-graph] total captured: {dispEnd - dispStart} dispatches over {maxTokens} steps = {(dispEnd - dispStart).toFloat / maxTokens.toFloat} /token"
    skipStagingWrites.set false
    let graph ← Hesper.CUDA.cuStreamEndCapture stream
    Hesper.cudaCaptureStream.set none
    let exec ← Hesper.CUDA.cuGraphInstantiate graph
    Hesper.CUDA.cuGraphDestroy graph
    IO.println s!"[Graph] captured full {maxTokens}-token decode graph"
    -- Single launch for the whole decode.
    Hesper.CUDA.cuGraphLaunch exec stream
    Hesper.CUDA.cuStreamSynchronize stream
    -- Read history and append to tokens.  EOS truncation after.
    let historyBytes ← GPUBackend.readBuffer ctx historyBuf (maxTokens * 4).toUSize
    for k in [0:maxTokens] do
      let tok := (Hesper.Basic.bytesToUInt32 historyBytes (k*4)).toNat
      tokens := tokens.push tok
      genCount := genCount + 1
      let mut stop := false
      match eosToken with
      | some eos => if tok == eos then stop := true
      | none => pure ()
      if extraEosTokens.any (· == tok) then stop := true
      if stop then break
    let genEnd ← IO.monoNanosNow
    let genMs := (genEnd - genStart).toFloat / 1000000.0
    let msPerToken := if genCount > 0 then genMs / genCount.toFloat else 0.0
    let tps := if msPerToken > 0 then 1000.0 / msPerToken else 0.0
    IO.println s!"[Gemma4] Generated {genCount} tokens in {genMs} ms ({tps} tokens/sec)"
    return tokens

  -- HESPER_PIPELINED_DECODE=1: like HESPER_TOKEN_GRAPH but without CUDA
  -- Graph capture. Launches argmax + advancePos kernels on the default
  -- stream after each forward; host never reads token value per step.
  -- The async pipeline lets the driver submit kernel N+1 while GPU
  -- still executes kernel N, matching llama.cpp's graphs-OFF behaviour.
  -- EOS check is done in a final batched readback after the loop.
  if pipelinedDecode then do
    let historyBuf ← GPUBackend.allocBuffer ctx (maxTokens * 4).toUSize
    let startPos := promptTokens.size
    let posBytes      := Hesper.WebGPU.BufferOps.uint32ToBytes startPos.toUInt32
    let cacheLenBytes := Hesper.WebGPU.BufferOps.uint32ToBytes (startPos + 1).toUInt32
    let posF32Bytes   ← Hesper.Basic.floatToBytes startPos.toFloat
    GPUBackend.writeBufferOffset ctx state.paramsBuf 0 posBytes
    GPUBackend.writeBufferOffset ctx state.paramsBuf 4 cacheLenBytes
    GPUBackend.writeBufferOffset ctx state.posF32Buf 0 posF32Bytes
    let lastPrefill := promptTokens[promptTokens.size - 1]!
    let lpBytes := Hesper.WebGPU.BufferOps.uint32ToBytes lastPrefill.toUInt32
    GPUBackend.writeBufferOffset ctx state.tokenBuf    0 lpBytes
    GPUBackend.writeBufferOffset ctx state.plRawRowBuf 0 lpBytes
    -- skipStagingWrites must be true so forwardSingleToken's
    -- writeScalarViaStaging calls become no-ops (otherwise they stamp
    -- the first tokenId into state.tokenBuf on every iteration,
    -- defeating the device-side argmax→tokenBuf feedback).
    skipStagingWrites.set true
    for k in [0:maxTokens] do
      forwardSingleToken ctx model 0 (startPos + k) state
        (kcr := some kcr) (skipTokenWrite := true)
      GPUBackend.execute ctx (argmaxKernel model.config.vocabSize k)
        [ ("logits", state.logitsBuf), ("result", historyBuf)
        , ("token", state.tokenBuf), ("token_ids", state.tokenBuf)
        , ("plRawRow", state.plRawRowBuf) ]
        { workgroupSize := { x := 256 }, numWorkgroups := (1, 1, 1) }
      GPUBackend.execute ctx advancePosKernel
        [ ("params", state.paramsBuf), ("posF32", state.posF32Buf) ]
        { workgroupSize := { x := 1 }, numWorkgroups := (1, 1, 1) }
    skipStagingWrites.set false
    -- One sync + one readback for all tokens.
    let historyBytes ← GPUBackend.readBuffer ctx historyBuf (maxTokens * 4).toUSize
    for k in [0:maxTokens] do
      let tok := (Hesper.Basic.bytesToUInt32 historyBytes (k*4)).toNat
      tokens := tokens.push tok
      genCount := genCount + 1
      let mut stop := false
      match eosToken with
      | some eos => if tok == eos then stop := true
      | none => pure ()
      if extraEosTokens.any (· == tok) then stop := true
      if stop then break
    let genEnd ← IO.monoNanosNow
    let genMs := (genEnd - genStart).toFloat / 1_000_000.0
    let msPerToken := if genCount > 0 then genMs / genCount.toFloat else 0.0
    let tps := if msPerToken > 0 then 1000.0 / msPerToken else 0.0
    IO.println s!"[Gemma4] Generated {genCount} tokens in {genMs} ms ({tps} tokens/sec)"
    return tokens

  let perTokTrace := (← IO.getEnv "HESPER_ALLOC_TRACE").isSome
  -- HESPER_UNIFIED_STREAM=1: point `cudaCaptureStream` at a persistent
  -- non-null stream so *all* kernel launches and H2D copies within
  -- `forwardPrefillBatch` / `forwardSingleToken` funnel into the same
  -- stream.  This replaces the default (null) stream's synchronous
  -- semantics: successive ops serialise in stream order without host
  -- stalls, and the `writeScalarViaStaging` pinned+async path fires
  -- instead of the sync `cuMemcpyHtoD_v2` fallback.  Decode sync
  -- happens when generate's argmax reads the result via readBuffer
  -- (implicit stream sync) so output remains correct. -/
  if state.unifiedStream != 0 then
    Hesper.cudaCaptureStream.set (some state.unifiedStream)
  let decodeSectTrace := (← IO.getEnv "HESPER_DECODE_SECT_TRACE").isSome
  -- HESPER_DEVICE_FED=1 (doc 58): drive forwardSingleToken from device-only
  -- state.  Each iteration:
  --   1. argmax kernel writes nextToken to state.tokenBuf / plRawRowBuf
  --   2. advancePosKernel increments paramsBuf[0,1] / posF32Buf
  --   3. forwardSingleToken called with skipTokenWrite=true, skipPosWrite=true
  --      — host doesn't touch any device scalar between iterations
  -- The host still reads `nextToken` once per iteration for EOS check, but
  -- that read can be deferred / batched in a follow-up.  Requires:
  --   * full-VRAM PLE table (not HESPER_USE_MMAP=1).
  let deviceFed := (← IO.getEnv "HESPER_DEVICE_FED").isSome
  if deviceFed && model.perLayerEmbdMmap.isSome then
    IO.println "[Config] HESPER_DEVICE_FED=1 incompatible with HESPER_USE_MMAP=1; falling back to host-fed path"
  let deviceFedActive := deviceFed && model.perLayerEmbdMmap.isNone
  if deviceFedActive then
    IO.println "[Config] HESPER_DEVICE_FED=1: device-fed decode loop active (doc 58)"
  -- Counts how many forwards have run, separate from genCount which is
  -- bumped *before* the forward (to push the next token).  We need the
  -- post-forward count to gate skipPosWrite — the very first forward
  -- still needs the host write because prefill left paramsBuf at
  -- startPos=promptLen-1, not the position we need for token 0.
  let mut decodeForwardsDone : Nat := 0
  -- doc 58 step B: deferred argmax read.  Seeded once before the loop;
  -- historyAppendKernel post-increments after each argmax write.
  if deviceFedActive then do
    let zeroBytes : ByteArray := ByteArray.mk #[0,0,0,0]
    GPUBackend.writeBufferOffset ctx state.historySlotBuf 0 zeroBytes
  for _ in [0:maxTokens] do
    if tokens.size >= model.config.maxSeqLen then break

    -- Reset dispatch counter at the start of each decode iteration — we
    -- want the per-token count (from argmax-read through next-forward).
    let dispBefore ← if dispCountEnabled then Hesper.dispatchCounter.get else pure 0
    let iterT0 ← if perTokTrace || decodeSectTrace then IO.monoNanosNow else pure 0
    let modLoadNsBefore ← if perTokTrace then Hesper.cudaModuleLoadWallNs.get else pure 0

    -- Sample: GPU-side greedy argmax (download 4 bytes instead of 1 MB).
    -- When a captured decode graph exists, argmax has already run as part
    -- of the graph (see capture block below); just read its result.
    let nextToken ← match graphExecOpt with
      | some _ =>
        let bytes ← GPUBackend.readBuffer ctx state.argmaxBuf (4 : USize)
        pure (Hesper.Basic.bytesToUInt32 bytes 0).toNat
      | none =>
        -- tokenIdsBuf / plRawRowBuf feedback keeps the captured graph
        -- self-feeding; when prefillTokenIdsRef isn't allocated (e.g.
        -- non-unified forwardSingleToken path) reuse state.tokenBuf as
        -- a harmless dummy.
        let tokenIdsBuf ← match ← state.prefillTokenIdsRef.get with
          | some (b, _) => pure b
          | none        => pure state.tokenBuf
        let v ← gpuArgmax ctx state.logitsBuf state.argmaxBuf state.tokenBuf tokenIdsBuf
          state.plRawRowBuf model.config.vocabSize state.argmaxCacheRef
          (hostMapped := state.argmaxHostMapped)
          (deferRead := deviceFedActive && decodeForwardsDone >= 1)
        -- doc 58 step B: when deferring, also append the freshly written
        -- argmax to the per-decode history buffer so the host can recover
        -- the token sequence after the decode loop ends.  historyAppend
        -- self-increments historySlotBuf so subsequent calls land at
        -- slot+1 — no per-iter host bookkeeping needed.
        if deviceFedActive && decodeForwardsDone >= 1 then
          GPUBackend.execute ctx historyAppendKernel
            [ ("src",  state.tokenBuf)
            , ("slot", state.historySlotBuf)
            , ("dst",  state.argmaxHistoryBuf) ]
            { workgroupSize := { x := 1 }, numWorkgroups := (1, 1, 1) }
        pure v

    let tArgmaxEnd ← if decodeSectTrace then IO.monoNanosNow else pure 0

    if (← IO.getEnv "HESPER_DECODE_TRACE").isSome then
      IO.println s!"[decode] genCount={genCount} tokens.size(before push)={tokens.size} nextToken={nextToken}"

    tokens := tokens.push nextToken
    genCount := genCount + 1

    -- Check EOS (primary + extras, e.g. Gemma 4's <end_of_turn> = 106)
    let mut stop := false
    match eosToken with
    | some eos => if nextToken == eos then stop := true
    | none => pure ()
    if extraEosTokens.any (· == nextToken) then stop := true
    if stop then break

    let tPostPushEnd ← if decodeSectTrace then IO.monoNanosNow else pure 0

    -- Forward pass for next token
    let newPos := tokens.size - 1
    if newPos < model.config.maxSeqLen then
      match graphExecOpt with
      | some (exec, stream) =>
        -- Replay path.  The captured graph self-feeds via the
        -- argmax→tokenBuf write and the advancePosKernel tail, so NO
        -- host-side buffer updates are needed between iterations — we
        -- just relaunch the same graph.
        let tg0 ← if decodeSectTrace then IO.monoNanosNow else pure 0
        Hesper.CUDA.cuGraphLaunch exec stream
        let tg1 ← if decodeSectTrace then IO.monoNanosNow else pure 0
        Hesper.CUDA.cuStreamSynchronize stream
        let tg2 ← if decodeSectTrace then IO.monoNanosNow else pure 0
        if decodeSectTrace then
          let launchMs := (tg1 - tg0).toFloat / 1e6
          let syncMs   := (tg2 - tg1).toFloat / 1e6
          IO.println s!"[graph] launch={launchMs}ms sync={syncMs}ms"
      | none =>
        -- HESPER_UNIFIED_DECODE=1: route decode through forwardPrefillBatch
        -- with N=1, startPos=newPos.  This is Phase 3 of the llama.cpp
        -- single-path architecture migration (docs/15-llama-single-path.md).
        -- When correct and fast enough, it replaces the separate
        -- forwardSingleToken path entirely.
        let unifiedDecode := (← IO.getEnv "HESPER_UNIFIED_DECODE").isSome
        if useCudaGraphs && genCount == 1 then
          -- Capture path: run decode forward on a capture stream.
          -- IMPORTANT: `cuStreamBeginCapture` intercepts kernel launches —
          -- they are recorded into the graph but do NOT execute.  So the
          -- forward pass captured here does not produce logits.  We must
          -- explicitly launch the instantiated graph once at the end of
          -- this iteration so the first decoded token actually advances
          -- the model state.  Skipping this step causes the next iter's
          -- gpuArgmax to read stale prefill logits → duplicated first
          -- token (the infamous "TheThe" bug).
          let stream ← Hesper.CUDA.cuStreamCreate
          -- BEFORE capture: seed paramsBuf / posF32Buf / tokenBuf / plRawRow
          -- directly (non-captured), so the first captured forward sees
          -- pos=newPos.  These must NOT become memcpy nodes in the graph
          -- because advancePosKernel (at the graph's tail) is the device-
          -- side source of truth for pos on all subsequent replays — any
          -- captured memcpy from a pinned host slot would overwrite it.
          let tokenBytes := Hesper.WebGPU.BufferOps.uint32ToBytes nextToken.toUInt32
          let posBytes := Hesper.WebGPU.BufferOps.uint32ToBytes newPos.toUInt32
          let cacheLenBytes := Hesper.WebGPU.BufferOps.uint32ToBytes (newPos + 1).toUInt32
          let posF32Bytes ← Hesper.Basic.floatToBytes newPos.toFloat
          GPUBackend.writeBufferOffset ctx state.tokenBuf    0 tokenBytes
          GPUBackend.writeBufferOffset ctx state.paramsBuf   0 posBytes
          GPUBackend.writeBufferOffset ctx state.paramsBuf   4 cacheLenBytes
          GPUBackend.writeBufferOffset ctx state.plRawRowBuf 0 tokenBytes
          GPUBackend.writeBufferOffset ctx state.posF32Buf   0 posF32Bytes
          -- Gate staging writes to no-ops for the duration of capture so
          -- `writeScalarViaStaging` inside forwardSingleToken / forwardBlock
          -- does NOT emit captured memcpy nodes.
          skipStagingWrites.set true
          Hesper.cudaCaptureStream.set (some stream)
          Hesper.CUDA.cuStreamBeginCapture stream
          if unifiedDecode then
            forwardPrefillBatch ctx model #[nextToken] state
              (kcr := some unifiedKcr) (startPos := newPos)
          else
            forwardSingleToken ctx model nextToken newPos state
              (kcr := some kcr) (skipTokenWrite := true)
          -- Fold argmax into the captured graph as well — saves one
          -- cuLaunchKernel per token on replay.  Result lands in
          -- `state.argmaxBuf`; the host reads 4 bytes after each replay.
          -- argmax is part of the captured graph so the token id is
          -- deposited into `state.tokenBuf`, `prefillTokenIdsBuf`, and
          -- `state.plRawRowBuf` device-side — no host copy needed
          -- before the next replay.
          let tokenIdsBuf ← match ← state.prefillTokenIdsRef.get with
            | some (b, _) => pure b
            | none        => pure state.tokenBuf
          GPUBackend.execute ctx (argmaxKernel model.config.vocabSize)
            [ ("logits", state.logitsBuf), ("result", state.argmaxBuf)
            , ("token", state.tokenBuf), ("token_ids", tokenIdsBuf)
            , ("plRawRow", state.plRawRowBuf) ]
            { workgroupSize := { x := 256 }, numWorkgroups := (1, 1, 1) }
          -- Advance pos / cacheLen / posF32 by 1 on the device so the
          -- NEXT replay automatically consumes `pos+1` without the host
          -- needing to writeScalarViaStaging them.
          GPUBackend.execute ctx advancePosKernel
            [ ("params", state.paramsBuf), ("posF32", state.posF32Buf) ]
            { workgroupSize := { x := 1 }, numWorkgroups := (1, 1, 1) }
          let graph ← Hesper.CUDA.cuStreamEndCapture stream
          Hesper.cudaCaptureStream.set none
          skipStagingWrites.set false
          let exec ← Hesper.CUDA.cuGraphInstantiate graph
          Hesper.CUDA.cuGraphDestroy graph
          -- Execute the captured graph once for the current token, so
          -- the next loop iteration's argmax sees logits from this token.
          Hesper.CUDA.cuGraphLaunch exec stream
          Hesper.CUDA.cuStreamSynchronize stream
          graphExecOpt := some (exec, stream)
          IO.println s!"[Graph] captured decode graph; subsequent tokens will replay"
        else
          if unifiedDecode then
            -- Pass a decode-local `unifiedKcr`.  The 2nd+ decode tokens then
            -- replay cached dispatches instead of re-generating PTX.  Prefill's
            -- `kcr` is kept separate so its cached (seqLen=18)-shape dispatches
            -- never fire at decode shape.
            forwardPrefillBatch ctx model #[nextToken] state
              (kcr := some unifiedKcr) (startPos := newPos)
            -- Debug for state-dirtying: run it TWICE in a row with same args
            -- and see if 2nd call produces same logits.  If HESPER_DOUBLE_CALL=1.
            if (← IO.getEnv "HESPER_DOUBLE_CALL").isSome then
              let firstArg ← GPUBackend.readBuffer ctx state.logitsBuf (16 : USize)
              forwardPrefillBatch ctx model #[nextToken] state
                (kcr := some unifiedKcr) (startPos := newPos)
              let secondArg ← GPUBackend.readBuffer ctx state.logitsBuf (16 : USize)
              IO.println s!"[doubleCall] genCount={genCount} first={firstArg.toList.take 16} second={secondArg.toList.take 16}"
            if (← IO.getEnv "HESPER_DUMP_LOGITS_UNIFIED").isSome then
              let bytes ← GPUBackend.readBuffer ctx state.logitsBuf (model.config.vocabSize * 4).toUSize
              IO.FS.writeBinFile s!"/tmp/hesper_unified_logits_step{genCount}.bin" bytes
          else
            -- doc 58: when HESPER_DEVICE_FED is on, skip the host-seeded
            -- token / pos writes from genCount=1 onwards — the previous
            -- iteration's argmax kernel wrote nextToken to state.tokenBuf
            -- / plRawRowBuf, and advancePosKernel (added below) inc'd
            -- paramsBuf / posF32Buf.  The very first decode iteration
            -- (genCount=0) still needs host writes because prefill left
            -- paramsBuf at startPos=promptLen-1 and tokenBuf at the prompt's
            -- last token, neither of which match the position we need.
            -- doc 58: from the *second* forward onwards, skip host writes
            -- of token / pos; the previous iter's argmax kernel filled
            -- tokenBuf, and advancePosKernel (below) bumped paramsBuf.
            let dfThisIter := deviceFedActive && decodeForwardsDone >= 1
            let dfTokenSkip := dfThisIter
            let dfPosSkip   := dfThisIter
            forwardSingleToken ctx model nextToken newPos state (kcr := some kcr) (skipTokenWrite := dfTokenSkip) (skipPosWrite := dfPosSkip)
            if deviceFedActive then
              -- Increment pos / cacheLen / posF32 device-side so the *next*
              -- iteration's forward sees the correct values without any host
              -- writeScalarViaStaging.  Same kernel the captured decode
              -- graph uses (line 3081).
              GPUBackend.execute ctx advancePosKernel
                [ ("params", state.paramsBuf), ("posF32", state.posF32Buf) ]
                { workgroupSize := { x := 1 }, numWorkgroups := (1, 1, 1) }
            decodeForwardsDone := decodeForwardsDone + 1
            if (← IO.getEnv "HESPER_DUMP_LOGITS_SINGLE").isSome then
              let bytes ← GPUBackend.readBuffer ctx state.logitsBuf (model.config.vocabSize * 4).toUSize
              IO.FS.writeBinFile s!"/tmp/hesper_single_logits_step{genCount}.bin" bytes

    let tForwardEnd ← if decodeSectTrace then IO.monoNanosNow else pure 0

    -- Print per-token dispatch count.  Captures launches through
    -- `launchKernelMaybeStream` — covers both fresh dispatches and
    -- cached replays.  For captured graph launches, counts just the
    -- cuGraphLaunch (kernel nodes inside the graph are NOT re-invoked
    -- through launchKernelMaybeStream).
    if perTokTrace then
      let iterT1 ← IO.monoNanosNow
      let modLoadNsAfter ← Hesper.cudaModuleLoadWallNs.get
      let iterMs := (iterT1 - iterT0).toFloat / 1e6
      let modMs  := (modLoadNsAfter - modLoadNsBefore).toFloat / 1e6
      IO.println s!"[tok] {genCount}: wall={iterMs} ms, modLoad={modMs} ms"
    if decodeSectTrace then
      let argmaxMs    := (tArgmaxEnd - iterT0).toFloat / 1e6
      let pushEosMs   := (tPostPushEnd - tArgmaxEnd).toFloat / 1e6
      let forwardMs   := (tForwardEnd - tPostPushEnd).toFloat / 1e6
      let totalMs     := (tForwardEnd - iterT0).toFloat / 1e6
      IO.println s!"[sect] tok={genCount} argmax={argmaxMs}ms pushEos={pushEosMs}ms forward={forwardMs}ms total={totalMs}ms"
    if dispCountEnabled then
      let dispAfter ← Hesper.dispatchCounter.get
      IO.println s!"[disp] tok={genCount} dispatches={dispAfter - dispBefore} per-layer={(dispAfter - dispBefore).toFloat / 42}"
      -- On the very first decode iteration (before graph capture),
      -- print per-section breakdown.  Afterward the captured graph
      -- replay doesn't go through withSection so totals stay stable.
      if genCount == 1 then
        let sectionDisp ← Hesper.WGSL.Execute.getSectionDispatches
        -- Sections that wrap other sections (e.g. `perLayerEmbd` wraps
        -- ple.inpGateGeluSlice + ple.proj + ple.postNormAdd) inflate the
        -- count via double-counting.  Skip them by name.
        let wrappers : List String := ["perLayerEmbd", "perLayerInputPre"]
        IO.println "[disp] per-section dispatches (wrappers hidden):"
        let mut innerTotal : Nat := 0
        for (name, count, calls) in sectionDisp do
          if wrappers.contains name then
            IO.println s!"  {name} (wrapper, see children): {count} dispatches over {calls} calls"
          else
            innerTotal := innerTotal + count
            IO.println s!"  {name}: {count} dispatches over {calls} calls"
        IO.println s!"[disp] inner-section sum: {innerTotal}"

  -- Clean up graph resources.
  match graphExecOpt with
  | some (exec, stream) =>
    Hesper.CUDA.cuGraphExecDestroy exec
    Hesper.CUDA.cuStreamDestroy stream
  | none => pure ()

  let genEnd ← IO.monoNanosNow
  let genMs := (genEnd - genStart).toFloat / 1_000_000.0
  let msPerToken := if genCount > 0 then genMs / genCount.toFloat else 0.0
  let tps := if msPerToken > 0 then 1000.0 / msPerToken else 0.0
  IO.println s!"[Gemma4] Generated {genCount} tokens in {genMs} ms ({tps} tokens/sec)"
  -- HESPER_ALLOC_TRACE=1: show sizes of allocBuffer calls during decode.
  -- A cached call site allocates once; a non-cached one allocates
  -- N*genCount times.  Entries with count > 1 are the candidates for
  -- adding an IORef cache.
  if (← IO.getEnv "HESPER_ALLOC_TRACE").isSome then
    Hesper.printAllocHistogram
    Hesper.printModuleLoadStats
    Hesper.printExecuteImplStats

  return tokens

end Hesper.Models.Gemma4
