import Hesper.Backend
import Hesper.Models.Gemma4
import Hesper.Models.Gemma4.LlamaKernels

/-!
# Phase 0 v3 LlamaPath: forwardTokenLlamaCpp (batched-multilayer)

v3 shifts from per-layer dispatch to batched-multilayer: each stub is
launched ONCE per forward (with `gridY = numLayers`), and selects its
layer internally via `blockIdx.y`.

For Phase 0 this means we no longer iterate over layers in host code — we
call `forwardTokenLlamaCpp` once and it emits exactly 17 dispatches.

## Stub dispatch order (one call each per forward)

 1. llamaAttnNormQuantBatchedKernel      — attnNorm + Q8_1 quantize
 2. llamaMulMatWQBatchedKernel            — wQ matmul
 3. llamaMulMatWKBatchedKernel            — wK matmul
 4. llamaMulMatWVBatchedKernel            — wV matmul
 5. llamaVcurNormBatchedKernel            — per-head V norm
 6. llamaRopeQBatchedKernel               — RoPE Q
 7. llamaRopeKBatchedKernel               — RoPE K
 8. llamaSetRowsKBatchedKernel            — K cache write
 9. llamaSetRowsVBatchedKernel            — V cache write
10. llamaFlashAttnBatchedKernel           — attention
11. llamaMulMatWOWithPostNormBatchedKernel— wO + postAttnNorm
12. llamaFfnNormQuantBatchedKernel        — ffnNorm + Q8_1
13. llamaMulMatGateUpGluBatchedKernel     — gate+up+GLU fused
14. llamaFfnDownQuantBatchedKernel        — ffn down input Q8_1
15. llamaMulMatFfnDownWithPostNormBatched — ffnDown + postFFNNorm
16. llamaPleStackBatchedKernel            — PLE chain
17. llamaLOutBatchedKernel                — layerOutScale × PLE + resid

Total: 17 dispatches for the transformer body.
Plus embedding/lmHead/argmax ≈ 3-5 extra.

**Target: ~20 dispatches/forward**, ~6× fewer than llama.cpp's 113/forward
(because llama.cpp does NOT batch across layers).
-/

namespace Hesper.Models.Gemma4

open Hesper
open Hesper.Models.Gemma4.Llama

/-- Emit the llama.cpp-shaped decode forward as batched-multilayer kernels.
    Each kernel runs ONCE with `gridY = numLayers`, internally selecting
    its layer via `blockIdx.y`.  For Phase 0 this gives exactly 17
    dispatches for the transformer body. -/
def forwardTokenLlamaCpp [GPUBackend β] (ctx : β)
    (model : Gemma4Model (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (_tokenId : Nat) (_pos : Nat)
    (state : InferenceState (GPUBackend.Buf β) (GPUBackend.CachedDispatch β)) : IO Unit := do
  let cfg := model.config
  let numLayers := cfg.numHiddenLayers
  let hidden := cfg.hiddenSize
  let inter := cfg.intermediateSize
  let numHeads := cfg.numAttentionHeads
  -- Use layer 0's head dim for the whole batch (Gemma-4 shares the same
  -- headDim across all layers in practice; the kernel can branch if needed).
  let headDim := cfg.headDim 0
  let numKVHeads := cfg.numKVHeads 0
  let qDim := numHeads * headDim
  let kvDim := numKVHeads * headDim
  let maxSeq := cfg.maxSeqLen
  let cacheSize := numKVHeads * maxSeq * headDim

  -- Helper: dispatch a batched stub with gridY = numLayers, workgroupSize=1x1,
  -- gridX chosen so each output element gets a workgroup.  For stubs we
  -- only need ONE thread in the (0, li, 0) WG, so gridX = 1, workgroup
  -- (1,1,1).  Real kernels will pick larger grids.
  let dispatchBatched := fun (shader : Hesper.WGSL.Monad.ShaderM Unit)
                             (bufs : List (String × GPUBackend.Buf β)) => do
    GPUBackend.execute ctx shader bufs
      { workgroupSize := { x := 1, y := 1, z := 1 }
        numWorkgroups := (1, numLayers, 1) : Hesper.ExecConfig }

  -- Alias state buffers as "per-forward scratch, treated as numLayers-sized
  -- tiles".  For Phase 0 stubs we just index into a single linear buffer:
  -- the skeleton doesn't verify correctness, only dispatch count.
  let attnQ8Buf := state.buf1              -- "attn_q8" scratch
  let qBuf := state.qBuf
  let kBuf := state.kBuf
  let vBuf := state.vBuf
  let vNormedBuf := state.vBuf2
  let qRopedBuf := state.qBuf2
  let kRopedBuf := state.kBuf2
  -- KV cache is per-layer; for Phase 0 we just use layer-0's cache for binding.
  let kvCache ← do
    if h : 0 < state.kvCaches.size then pure state.kvCaches[0]
    else throw (IO.userError "forwardTokenLlamaCpp: no KV cache")
  let attnOutBuf := state.attnOutBuf
  let attnResidBuf := state.attnResidualBuf
  let ffnQ8Buf := state.normedBuf          -- "ffn_q8" scratch
  let geluBuf := state.geluBuf
  let geluQ8Buf := state.upBuf             -- "gelu_q8" scratch
  let ffnResidBuf := state.ffnOutBuf
  let pleOutBuf := state.plProjBuf
  let lOutBuf := state.buf2                -- layer output ping-pong

  -- 1. attnNorm + Q8_1 quantize
  dispatchBatched (llamaAttnNormQuantBatchedKernel numLayers hidden)
    [("input", state.buf2), ("attn_q8", attnQ8Buf)]

  -- 2-4. Q/K/V projections
  dispatchBatched (llamaMulMatWQBatchedKernel numLayers hidden qDim)
    [("attn_q8", attnQ8Buf), ("q_out", qBuf)]
  dispatchBatched (llamaMulMatWKBatchedKernel numLayers hidden kvDim)
    [("attn_q8", attnQ8Buf), ("k_out", kBuf)]
  dispatchBatched (llamaMulMatWVBatchedKernel numLayers hidden kvDim)
    [("attn_q8", attnQ8Buf), ("v_out", vBuf)]

  -- 5. V per-head norm
  dispatchBatched (llamaVcurNormBatchedKernel numLayers kvDim)
    [("v_in", vBuf), ("v_normed", vNormedBuf)]

  -- 6-7. RoPE Q/K
  dispatchBatched (llamaRopeQBatchedKernel numLayers qDim)
    [("q_in", qBuf), ("q_roped", qRopedBuf)]
  dispatchBatched (llamaRopeKBatchedKernel numLayers kvDim)
    [("k_in", kBuf), ("k_roped", kRopedBuf)]

  -- 8-9. KV cache writes
  dispatchBatched (llamaSetRowsKBatchedKernel numLayers kvDim cacheSize)
    [("k_new", kRopedBuf), ("k_cache", kvCache.kBuf)]
  dispatchBatched (llamaSetRowsVBatchedKernel numLayers kvDim cacheSize)
    [("v_new", vNormedBuf), ("v_cache", kvCache.vBuf)]

  -- 10. Flash attention
  dispatchBatched (llamaFlashAttnBatchedKernel numLayers qDim qDim)
    [("q_roped", qRopedBuf), ("attn_out", attnOutBuf)]

  -- 11. wO + postAttnNorm + residual
  dispatchBatched (llamaMulMatWOWithPostNormBatchedKernel numLayers qDim hidden)
    [("attn_out", attnOutBuf), ("attn_resid", attnResidBuf)]

  -- 12. FFN norm + Q8_1
  dispatchBatched (llamaFfnNormQuantBatchedKernel numLayers hidden)
    [("attn_resid", attnResidBuf), ("ffn_q8", ffnQ8Buf)]

  -- 13. gate+up+GLU fused
  dispatchBatched (llamaMulMatGateUpGluBatchedKernel numLayers hidden inter)
    [("ffn_q8", ffnQ8Buf), ("gelu_out", geluBuf)]

  -- 14. ffnDown input Q8_1
  dispatchBatched (llamaFfnDownQuantBatchedKernel numLayers inter)
    [("gelu_out", geluBuf), ("gelu_q8", geluQ8Buf)]

  -- 15. ffnDown + postFFNNorm + residual
  dispatchBatched (llamaMulMatFfnDownWithPostNormBatchedKernel numLayers inter hidden)
    [("gelu_q8", geluQ8Buf), ("ffn_resid", ffnResidBuf)]

  -- 16. PLE stack
  dispatchBatched (llamaPleStackBatchedKernel numLayers hidden)
    [("ffn_resid", ffnResidBuf), ("ple_out", pleOutBuf)]

  -- 17. l_out = PLE × scale + residual
  dispatchBatched (llamaLOutBatchedKernel numLayers hidden)
    [("ple_out", pleOutBuf), ("l_out", lOutBuf)]

/-- Stub forward at the **same dispatch shape as the real forwardSingleToken**:
    one stub kernel per layer per op, ~17 ops × 42 layers = ~714 dispatches.
    Used by `gemma4-stub-decode-bench` to measure pure host-side overhead
    (Lean cudaExecuteImpl, kcr lookup, args Array build/expand, withSection
    bookkeeping) **without** any real kernel work. Subtracting this wall
    from the real-forward wall isolates the per-block hesper-specific
    overhead suspected of accounting for the 4 ms gap to llama.cpp's
    graphs-OFF forward host time (doc 57 §3b.1, hypothesis H4b).

    Differences from `forwardTokenLlamaCpp`:
    - The 17 stubs are emitted in a `for li in [0:numLayers]` loop instead
      of `gridY = numLayers`. Total dispatches: numLayers × 17.
    - Each call uses a 1×1×1 workgroup (the stubs are no-ops; we don't
      care about real GPU work).
    - We deliberately drop kcr/cache so each call goes through the
      cold-path of `cudaExecuteImpl` — that's the whole point: we want
      the worst-case host cost the real forward incurs every time it
      misses the dispatch cache. Pass `kcrOpt := some kcr` to measure
      the *cached* host cost instead. -/
def forwardTokenStubPerLayer [GPUBackend β] (ctx : β)
    (model : Gemma4Model (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (state : InferenceState (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (kcrOpt : Option (KernelCacheRefs (GPUBackend.CachedDispatch β)) := none)
    : IO Unit := do
  let cfg := model.config
  let numLayers := cfg.numHiddenLayers
  let hidden := cfg.hiddenSize
  let inter := cfg.intermediateSize
  let numHeads := cfg.numAttentionHeads
  let headDim := cfg.headDim 0
  let numKVHeads := cfg.numKVHeads 0
  let qDim := numHeads * headDim
  let kvDim := numKVHeads * headDim
  let maxSeq := cfg.maxSeqLen
  let cacheSize := numKVHeads * maxSeq * headDim
  let kvCache ← do
    if h : 0 < state.kvCaches.size then pure state.kvCaches[0]
    else throw (IO.userError "forwardTokenStubPerLayer: no KV cache")
  -- Single-layer launch helper: numWorkgroups (1,1,1).  Routed through
  -- `executeWithConfigCached` when a kcr is provided so the per-layer
  -- caching paths the real forward exercises light up identically.
  let dispatchOne := fun (name : String) (shader : Hesper.WGSL.Monad.ShaderM Unit)
                          (bufs : List (String × GPUBackend.Buf β)) (li : Nat) => do
    let cfg : Hesper.ExecConfig :=
      { workgroupSize := { x := 1, y := 1, z := 1 }
        numWorkgroups := (1, 1, 1) : Hesper.ExecConfig }
    match kcrOpt with
    | some k =>
      let key := hash ("stubPerLayer", name, li)
      let ref ← k.getRef key
      GPUBackend.executeWithConfigCached ctx shader bufs cfg key ref
    | none =>
      GPUBackend.execute ctx shader bufs cfg
  let attnQ8Buf := state.buf1
  let qBuf := state.qBuf
  let kBuf := state.kBuf
  let vBuf := state.vBuf
  let vNormedBuf := state.vBuf2
  let qRopedBuf := state.qBuf2
  let kRopedBuf := state.kBuf2
  let attnOutBuf := state.attnOutBuf
  let attnResidBuf := state.attnResidualBuf
  let ffnQ8Buf := state.normedBuf
  let geluBuf := state.geluBuf
  let geluQ8Buf := state.upBuf
  let ffnResidBuf := state.ffnOutBuf
  let pleOutBuf := state.plProjBuf
  let lOutBuf := state.buf2
  for li in [0:numLayers] do
    -- We reuse the same batched-stub kernel set with numLayers=1 so the
    -- shader bodies stay valid; in practice the stubs only touch one
    -- output element so this is fine.
    dispatchOne "attnNormQuant" (llamaAttnNormQuantBatchedKernel 1 hidden)
      [("input", state.buf2), ("attn_q8", attnQ8Buf)] li
    dispatchOne "wQ" (llamaMulMatWQBatchedKernel 1 hidden qDim)
      [("attn_q8", attnQ8Buf), ("q_out", qBuf)] li
    dispatchOne "wK" (llamaMulMatWKBatchedKernel 1 hidden kvDim)
      [("attn_q8", attnQ8Buf), ("k_out", kBuf)] li
    dispatchOne "wV" (llamaMulMatWVBatchedKernel 1 hidden kvDim)
      [("attn_q8", attnQ8Buf), ("v_out", vBuf)] li
    dispatchOne "vNorm" (llamaVcurNormBatchedKernel 1 kvDim)
      [("v_in", vBuf), ("v_normed", vNormedBuf)] li
    dispatchOne "ropeQ" (llamaRopeQBatchedKernel 1 qDim)
      [("q_in", qBuf), ("q_roped", qRopedBuf)] li
    dispatchOne "ropeK" (llamaRopeKBatchedKernel 1 kvDim)
      [("k_in", kBuf), ("k_roped", kRopedBuf)] li
    dispatchOne "kvWriteK" (llamaSetRowsKBatchedKernel 1 kvDim cacheSize)
      [("k_new", kRopedBuf), ("k_cache", kvCache.kBuf)] li
    dispatchOne "kvWriteV" (llamaSetRowsVBatchedKernel 1 kvDim cacheSize)
      [("v_new", vNormedBuf), ("v_cache", kvCache.vBuf)] li
    dispatchOne "flashAttn" (llamaFlashAttnBatchedKernel 1 qDim qDim)
      [("q_roped", qRopedBuf), ("attn_out", attnOutBuf)] li
    dispatchOne "wO" (llamaMulMatWOWithPostNormBatchedKernel 1 qDim hidden)
      [("attn_out", attnOutBuf), ("attn_resid", attnResidBuf)] li
    dispatchOne "ffnNormQuant" (llamaFfnNormQuantBatchedKernel 1 hidden)
      [("attn_resid", attnResidBuf), ("ffn_q8", ffnQ8Buf)] li
    dispatchOne "gateUpGlu" (llamaMulMatGateUpGluBatchedKernel 1 hidden inter)
      [("ffn_q8", ffnQ8Buf), ("gelu_out", geluBuf)] li
    dispatchOne "ffnDownQuant" (llamaFfnDownQuantBatchedKernel 1 inter)
      [("gelu_out", geluBuf), ("gelu_q8", geluQ8Buf)] li
    dispatchOne "ffnDown" (llamaMulMatFfnDownWithPostNormBatchedKernel 1 inter hidden)
      [("gelu_q8", geluQ8Buf), ("ffn_resid", ffnResidBuf)] li
    dispatchOne "pleStack" (llamaPleStackBatchedKernel 1 hidden)
      [("ffn_resid", ffnResidBuf), ("ple_out", pleOutBuf)] li
    dispatchOne "lOut" (llamaLOutBatchedKernel 1 hidden)
      [("ple_out", pleOutBuf), ("l_out", lOutBuf)] li

end Hesper.Models.Gemma4
