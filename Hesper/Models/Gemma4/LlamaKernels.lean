import Hesper.WGSL.Monad
import Hesper.WGSL.Exp

/-!
# Phase 0 v3 LlamaPath stub kernels — batched-multilayer physical kernels

Target: **~113 dispatches/token**, matching llama.cpp's actual physical CUDA
kernel launch count per decode forward (measured via nsys).

Key insight: llama.cpp calls the SAME kernel (e.g. `rms_norm_f32<1024,1,1>`)
many times per forward pass — 42 times for attnNorm alone.  Each of those
call sites is a separate cuLaunchKernel.  nsys counts them as separate
"instances" of one named kernel.

hesper's equivalent: emit ONE stub per **kernel type**, dispatched with a
3-D grid where `gridY = numLayers` (one workgroup row per layer), so the
kernel internally switches on `blockIdx.y` to select which layer's
weights to read.  The physical launch count drops from `N×42` to `N`.

Previously (v2): 17 stubs × 42 layers = 606 dispatches.
This (v3):      17 stubs × 1 (batched across 42) = 17 dispatches.
Plus small non-batched kernels (embedding lookup, lm_head, argmax) ≈ ~10
extra, totaling **~27 kernel launches** for the skeleton itself.

Actual llama.cpp is ~113 because it still emits 42 independent launches for
each of: rms_norm, matmul, rope, set_rows, etc. — fused-multilayer dispatch
is NOT a llama.cpp trick (it's a hesper advantage we can use if it proves
faster).  v3 skeleton stubs are batched-multilayer by default; real kernel
implementations in Phase 1+ can choose either per-layer or multi-layer.

## Stub body convention

Each stub:
- Declares input/output buffers as sized `numLayers * perLayerSize`
- Reads workgroup id y to select layer index
- Writes `out[li*stride + 0] = in[li*stride + 0]` (DCE-safe)
- Grid: `(perLayerCells, numLayers, 1)` so nsys sees ONE kernel with
  one grid dimension of 42

## Per-forward kernel count (v3)

17 llama.cpp-shaped kernel types, each batched across 42 layers → **17
dispatches per forward** for the transformer body.  Plus embedding, PLE
precompute, lm_head, argmax etc. ≈ ~25-30 total.

This is **4× lower than llama.cpp (~113)** because llama.cpp does NOT
use batched-multilayer dispatch.  That's fine — the whole point of hesper
is to take advantage of the unified runtime.

If even 17 proves limiting we can fuse more (e.g. wQ/wK/wV into 1 kernel,
attnNormQuant into the matmul).  For now 17 is the reference.
-/

namespace Hesper.Models.Gemma4.Llama

open Hesper.WGSL
open Hesper.WGSL.Monad
open Hesper.WGSL.Exp

/-- DCE-safe batched-multilayer stub body: thread 0 of workgroup `(0, 0, 0)`
    reads `in[0]` and writes `out[0]`.  The batched dispatch has
    `gridY = numLayers` so 42 workgroups will execute the kernel, but for
    skeleton purposes only the first workgroup does anything observable.
    Buffer size is declared as `1` to sidestep per-layer tiling at the
    stub stage; Phase 1+ kernels will use realistic per-layer tiling. -/
private def stubBodyBatched
    (inBufName outBufName : String)
    (_inTotalSize _outTotalSize _perLayerIn _perLayerOut _numLayers : Nat) : ShaderM Unit := do
  let _in  ← ShaderM.declareInputBuffer  inBufName  (.array (.scalar .f32) 1)
  let _out ← ShaderM.declareOutputBuffer outBufName (.array (.scalar .f32) 1)
  let wgid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let ly := Exp.vec3Y wgid
  let tx := Exp.vec3X lid
  ShaderM.if_ (Exp.eq (Exp.add tx ly) (Exp.litU32 0)) (do
    let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := 1) inBufName (Exp.litU32 0)
    ShaderM.writeBuffer (ty := .scalar .f32) outBufName (Exp.litU32 0) v
  ) (pure ())

/-! ## 1. Batched attn pre-norm + Q8_1 quantize (42-layer dispatch) -/

def llamaAttnNormQuantBatchedKernel (numLayers hiddenSize : Nat) : ShaderM Unit := do
  stubBodyBatched "input" "attn_q8"
    (numLayers * hiddenSize) (numLayers * hiddenSize)
    hiddenSize hiddenSize numLayers

/-! ## 2-4. Q/K/V projections (batched) -/

def llamaMulMatWQBatchedKernel (numLayers hiddenSize outDim : Nat) : ShaderM Unit := do
  stubBodyBatched "attn_q8" "q_out"
    (numLayers * hiddenSize) (numLayers * outDim)
    hiddenSize outDim numLayers

def llamaMulMatWKBatchedKernel (numLayers hiddenSize outDim : Nat) : ShaderM Unit := do
  stubBodyBatched "attn_q8" "k_out"
    (numLayers * hiddenSize) (numLayers * outDim)
    hiddenSize outDim numLayers

def llamaMulMatWVBatchedKernel (numLayers hiddenSize outDim : Nat) : ShaderM Unit := do
  stubBodyBatched "attn_q8" "v_out"
    (numLayers * hiddenSize) (numLayers * outDim)
    hiddenSize outDim numLayers

/-! ## 5. V per-head norm (batched) -/

def llamaVcurNormBatchedKernel (numLayers vDim : Nat) : ShaderM Unit := do
  stubBodyBatched "v_in" "v_normed"
    (numLayers * vDim) (numLayers * vDim) vDim vDim numLayers

/-! ## 6-7. RoPE Q/K (batched) -/

def llamaRopeQBatchedKernel (numLayers qDim : Nat) : ShaderM Unit := do
  stubBodyBatched "q_in" "q_roped"
    (numLayers * qDim) (numLayers * qDim) qDim qDim numLayers

def llamaRopeKBatchedKernel (numLayers kDim : Nat) : ShaderM Unit := do
  stubBodyBatched "k_in" "k_roped"
    (numLayers * kDim) (numLayers * kDim) kDim kDim numLayers

/-! ## 8-9. KV cache writes (batched) -/

def llamaSetRowsKBatchedKernel (numLayers kDim cacheSize : Nat) : ShaderM Unit := do
  stubBodyBatched "k_new" "k_cache"
    (numLayers * kDim) (numLayers * cacheSize) kDim cacheSize numLayers

def llamaSetRowsVBatchedKernel (numLayers vDim cacheSize : Nat) : ShaderM Unit := do
  stubBodyBatched "v_new" "v_cache"
    (numLayers * vDim) (numLayers * cacheSize) vDim cacheSize numLayers

/-! ## 10. FlashAttention (batched across layers) -/

def llamaFlashAttnBatchedKernel (numLayers qDim attnOutDim : Nat) : ShaderM Unit := do
  stubBodyBatched "q_roped" "attn_out"
    (numLayers * qDim) (numLayers * attnOutDim)
    qDim attnOutDim numLayers

/-! ## 11. wO + postAttnNorm + residual (batched) -/

def llamaMulMatWOWithPostNormBatchedKernel (numLayers attnOutDim hiddenSize : Nat) : ShaderM Unit := do
  stubBodyBatched "attn_out" "attn_resid"
    (numLayers * attnOutDim) (numLayers * hiddenSize)
    attnOutDim hiddenSize numLayers

/-! ## 12. FFN pre-norm + Q8_1 quantize (batched) -/

def llamaFfnNormQuantBatchedKernel (numLayers hiddenSize : Nat) : ShaderM Unit := do
  stubBodyBatched "attn_resid" "ffn_q8"
    (numLayers * hiddenSize) (numLayers * hiddenSize)
    hiddenSize hiddenSize numLayers

/-! ## 13. Gate+Up with GLU epilogue (batched) -/

def llamaMulMatGateUpGluBatchedKernel (numLayers hiddenSize interDim : Nat) : ShaderM Unit := do
  stubBodyBatched "ffn_q8" "gelu_out"
    (numLayers * hiddenSize) (numLayers * interDim)
    hiddenSize interDim numLayers

/-! ## 14. ffnDown input re-quantize (batched) -/

def llamaFfnDownQuantBatchedKernel (numLayers interDim : Nat) : ShaderM Unit := do
  stubBodyBatched "gelu_out" "gelu_q8"
    (numLayers * interDim) (numLayers * interDim)
    interDim interDim numLayers

/-! ## 15. ffnDown + postFFNNorm + residual (batched) -/

def llamaMulMatFfnDownWithPostNormBatchedKernel (numLayers interDim hiddenSize : Nat) : ShaderM Unit := do
  stubBodyBatched "gelu_q8" "ffn_resid"
    (numLayers * interDim) (numLayers * hiddenSize)
    interDim hiddenSize numLayers

/-! ## 16. PLE stack (batched) -/

def llamaPleStackBatchedKernel (numLayers hiddenSize : Nat) : ShaderM Unit := do
  stubBodyBatched "ffn_resid" "ple_out"
    (numLayers * hiddenSize) (numLayers * hiddenSize)
    hiddenSize hiddenSize numLayers

/-! ## 17. Layer output: l_out = PLE × scale + residual (batched) -/

def llamaLOutBatchedKernel (numLayers hiddenSize : Nat) : ShaderM Unit := do
  stubBodyBatched "ple_out" "l_out"
    (numLayers * hiddenSize) (numLayers * hiddenSize)
    hiddenSize hiddenSize numLayers

end Hesper.Models.Gemma4.Llama
