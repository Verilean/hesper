import Hesper.Backend
import Hesper.Models.Gemma4
import Hesper.Models.Gemma4.LlamaKernels
import Hesper.Models.Gemma4.LlamaKernelsPrefill
import Hesper.Layers.Embedding
import Hesper.Layers.Linear
import Hesper.Layers.PerLayerEmbedding
import Hesper.Quantization.Q6_K
import Hesper.WebGPU.BufferOps
import Hesper.WGSL.MatMul
import Hesper.Models.Gemma4.Kernels
import Hesper.Models.Gemma4.ScratchPool

/-!
# Phase 0 v3 LlamaPath: forwardPrefillLlamaCpp (loop-faithful stub)

Mirrors **llama.cpp's `llm_build_gemma4_iswa::llm_build_gemma4_iswa`** from
`llama.cpp/src/models/gemma4-iswa.cpp`.  Each logical op in llama.cpp's
graph builder → one stub dispatch here.  Stubs themselves are trivial
(DCE-safe copy); what matters is the **loop structure and call order**.

## Op-per-op correspondence

### Prelude (once, outside the layer loop)
- `build_inp_embd`           → embeddingLookup
- `ggml_scale` (inpL)        → embedScale
- `project_per_layer_inputs`:
  - `ggml_mul_mat` per_layer_model_proj → plPre.matmul
  - `ggml_scale`                         → plPre.scale
  - `ggml_reshape_3d`                    → (view, not a kernel)
  - `build_norm`                         → plPre.rmsNorm
  - `ggml_add`                           → plPre.add
  - `ggml_scale`                         → plPre.scale2
  - `ggml_permute` + `ggml_cont`         → plPre.cont

### Per-layer loop (42 iterations)
- `build_norm` attn_norm
- `build_lora_mm` wQ
- `build_norm` q_norm
- `ggml_rope_ext` Qcur
- [if has_kv] `build_lora_mm` wK
- [if has_kv] `build_lora_mm` wV
- [if has_kv] `build_norm` k_norm
- [if has_kv] `ggml_rms_norm` Vcur
- [if has_kv] `ggml_rope_ext` Kcur
- `build_attn` (flash_attn + wO, ~4 ops internally)
- `build_norm` attn_post_norm
- `ggml_add` attn_out
- `build_norm` ffn_norm
- `build_ffn` (ffn_up + ffn_gate + gelu + mul + ffn_down, 5 ops)
- `build_norm` ffn_post_norm
- `ggml_add` residual
- [PLE block] `build_lora_mm` per_layer_inp_gate
- [PLE block] `ggml_gelu`
- [PLE block] `ggml_mul`
- [PLE block] `build_lora_mm` per_layer_proj
- [PLE block] `build_norm` per_layer_post_norm
- [PLE block] `ggml_add` residual
- [if out_scale] `ggml_mul` scale
- `build_cvec` (usually no-op)

### Post-loop (once)
- `build_norm` output_norm
- `build_lora_mm` lm_head
- [if softcap] 3 ops (scale, tanh, scale)

Target: **~2016 kernels/forward** (matches llama.cpp nsys, graphs disabled).
Actual count depends on how many physical CUDA kernels each ggml op expands
to (mmq = 3 kernels: quantize + matmul + fixup).  We over-represent matmul
as 3 stubs each to approximate the real launch count.
-/

namespace Hesper.Models.Gemma4

open Hesper
open Hesper.Models.Gemma4.Llama
open Hesper.Models.Gemma4.Llama.Prefill

/-- Emit a ggml `mul_mat` op as llama.cpp emits it for prefill:
    `quantize_mmq_q8_1` + `mul_mat_q` + `mul_mat_q_stream_k_fixup` = 3 kernels. -/
private def dispatchMulMat
    [GPUBackend β] (ctx : β)
    (inBuf outBuf : GPUBackend.Buf β) (seqLen K N : Nat) : IO Unit := do
  GPUBackend.execute ctx (prefillQuantizeMmqQ8_1Kernel 1 seqLen K)
    [("mmq_in", inBuf), ("mmq_q8", outBuf)]
    { workgroupSize := { x := 1, y := 1, z := 1 }, numWorkgroups := (1, 1, 1) }
  GPUBackend.execute ctx (prefillMulMatQQ4KKernel 1 seqLen N K)
    [("mmq_q8", outBuf), ("mmq_part", inBuf)]
    { workgroupSize := { x := 1, y := 1, z := 1 }, numWorkgroups := (1, 1, 1) }
  GPUBackend.execute ctx (prefillMulMatQStreamKFixupQ4KKernel 1 seqLen N)
    [("mmq_part", inBuf), ("mmq_out", outBuf)]
    { workgroupSize := { x := 1, y := 1, z := 1 }, numWorkgroups := (1, 1, 1) }

/-- A ggml `build_norm` (RMSNorm) op.  Physical kernel: `rms_norm_f32`. -/
private def dispatchRmsNorm
    [GPUBackend β] (ctx : β)
    (inBuf outBuf : GPUBackend.Buf β) (hidden : Nat) : IO Unit := do
  GPUBackend.execute ctx (llamaAttnNormQuantBatchedKernel 1 hidden)
    [("input", inBuf), ("attn_q8", outBuf)]
    { workgroupSize := { x := 1, y := 1, z := 1 }, numWorkgroups := (1, 1, 1) }

/-- A trivial pointwise op (scale/add/mul/gelu).  1 physical kernel. -/
private def dispatchPointwise
    [GPUBackend β] (ctx : β)
    (inBuf outBuf : GPUBackend.Buf β) (hidden : Nat) : IO Unit := do
  GPUBackend.execute ctx (llamaLOutBatchedKernel 1 hidden)
    [("ple_out", inBuf), ("l_out", outBuf)]
    { workgroupSize := { x := 1, y := 1, z := 1 }, numWorkgroups := (1, 1, 1) }

/-- `build_attn`: expand into flash_attn (ampere_s16816gemm) + soft_max +
    k_set_rows (K) + k_set_rows (V) + wO matmul (3 kernels) + convert_unary ×2.
    ≈ 9 physical kernels per call.  This matches llama.cpp's prefill-path
    attention footprint observed via nsys. -/
private def dispatchBuildAttn
    [GPUBackend β] (ctx : β)
    (qBuf kBuf vBuf outBuf : GPUBackend.Buf β) (seqLen headDim hidden : Nat) : IO Unit := do
  -- K cache write
  GPUBackend.execute ctx (llamaSetRowsKBatchedKernel 1 hidden hidden)
    [("k_new", kBuf), ("k_cache", qBuf)]
    { workgroupSize := { x := 1, y := 1, z := 1 }, numWorkgroups := (1, 1, 1) }
  -- V cache write
  GPUBackend.execute ctx (llamaSetRowsVBatchedKernel 1 hidden hidden)
    [("v_new", vBuf), ("v_cache", qBuf)]
    { workgroupSize := { x := 1, y := 1, z := 1 }, numWorkgroups := (1, 1, 1) }
  -- convert_unary f32→f16 (Q)
  GPUBackend.execute ctx (prefillConvertF32ToF16Kernel 1 hidden)
    [("cv_f32", qBuf), ("cv_f16", outBuf)]
    { workgroupSize := { x := 1, y := 1, z := 1 }, numWorkgroups := (1, 1, 1) }
  -- tensor-core GEMM (QK^T)
  GPUBackend.execute ctx (prefillFlashAttnTensorCoreKernel 1 seqLen seqLen headDim)
    [("q_roped", outBuf), ("attn_out_tc", qBuf)]
    { workgroupSize := { x := 1, y := 1, z := 1 }, numWorkgroups := (1, 1, 1) }
  -- soft_max
  GPUBackend.execute ctx (llamaFlashAttnBatchedKernel 1 hidden hidden)
    [("q_roped", qBuf), ("attn_out", outBuf)]
    { workgroupSize := { x := 1, y := 1, z := 1 }, numWorkgroups := (1, 1, 1) }
  -- tensor-core GEMM (attn × V)
  GPUBackend.execute ctx (prefillFlashAttnTensorCoreKernel 1 seqLen seqLen headDim)
    [("q_roped", outBuf), ("attn_out_tc", qBuf)]
    { workgroupSize := { x := 1, y := 1, z := 1 }, numWorkgroups := (1, 1, 1) }
  -- convert_unary f16→f32
  GPUBackend.execute ctx (prefillConvertF16ToF32Kernel 1 hidden)
    [("cv_f16", qBuf), ("cv_f32", outBuf)]
    { workgroupSize := { x := 1, y := 1, z := 1 }, numWorkgroups := (1, 1, 1) }
  -- wO matmul (3 kernels: quantize + mmq + fixup)
  dispatchMulMat ctx outBuf qBuf seqLen hidden hidden

/-- `build_ffn` (LLM_FFN_GELU, LLM_FFN_PAR): ffn_gate matmul + ffn_up matmul +
    fused gated-gelu + ffn_down matmul.  → 3 + 3 + 1 + 3 = 10 physical kernels.
    (In practice llama.cpp fuses gate×gelu(up) into `unary_gated_op_kernel`.) -/
private def dispatchBuildFfn
    [GPUBackend β] (ctx : β)
    (inBuf outBuf : GPUBackend.Buf β) (seqLen hidden inter : Nat) : IO Unit := do
  -- ffn_gate matmul
  dispatchMulMat ctx inBuf outBuf seqLen hidden inter
  -- ffn_up matmul
  dispatchMulMat ctx inBuf outBuf seqLen hidden inter
  -- fused gate × gelu(up)
  GPUBackend.execute ctx (prefillGatedGeluKernel 1 seqLen inter)
    [("gate_up", inBuf), ("gelu_out", outBuf)]
    { workgroupSize := { x := 1, y := 1, z := 1 }, numWorkgroups := (1, 1, 1) }
  -- ffn_down matmul
  dispatchMulMat ctx outBuf inBuf seqLen inter hidden

/-- Emit the llama.cpp-shaped prefill forward, faithfully mirroring the
    `llm_build_gemma4_iswa` graph builder op sequence.

    Parity-development mode: when `tokenIdsBuf` is provided and points to a
    buffer of u32 token IDs of length ≥ `seqLen`, the prelude performs a
    real embedding lookup + scale.  When `none`, we fall back to the
    earlier dispatch-count-only stub behaviour.

    Set `HESPER_GOLDEN_DUMP_DIR=<dir>` to write each named intermediate
    tensor (matching llama.cpp's `cb()` names) to `<dir>/<name>.bin` for
    side-by-side diffing against `llama-eval-callback`'s dumps.

    Returns the last-token logits buffer (vocabSize f32 elements) when
    a prompt is provided.  Callers can argmax this to pick the next
    token for greedy decoding.

    `startPos` is the rotary / KV-cache offset for the first new token.
    Prefill uses `startPos=0`; decode steps pass the total number of
    previously-seen tokens.

    `persistentCaches`, when provided, overrides per-call KV-cache
    allocation — caches persist across forwards for decode loops.
    Expected length: `numHiddenLayers - numKVSharedLayers` = 24 for E4B.
    Each entry is a pair `(kCache, vCache)`.

    `scratchPool`, when provided, is used for all per-forward transient
    buffers (batch activations, PLE scratches, etc.) — the pool's slots
    are reused across calls, eliminating per-forward cuMemAlloc churn.
    If `none`, each call allocates fresh buffers via `allocBuffer`.

    `paramsBufOverride`, when provided, is used as the paramsBuf (the 4-byte
    u32 holding `startPos` for RoPE / KV-cache kernels).  Callers that want
    to capture this forward into a CUDA graph pass a persistent buffer so
    the graph's kernel args reference a stable device pointer; the caller
    is responsible for writing the current startPos into it before each
    replay.  When the override is provided, this function does NOT write
    to it — the caller owns the update schedule. -/
def forwardPrefillLlamaCpp [GPUBackend β] (ctx : β)
    (model : Gemma4Model (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (seqLen : Nat)
    (state : InferenceState (GPUBackend.Buf β) (GPUBackend.CachedDispatch β))
    (tokenIdsBuf : Option (GPUBackend.Buf β) := none)
    (startPos : Nat := 0)
    (persistentCaches : Option (Array (GPUBackend.Buf β × GPUBackend.Buf β)) := none)
    (scratchPool : Option (ScratchPool β) := none)
    (paramsBufOverride : Option (GPUBackend.Buf β) := none)
    : IO (Option (GPUBackend.Buf β)) := do
  let cfg := model.config
  let numLayers := cfg.numHiddenLayers
  let hidden := cfg.hiddenSize
  let inter := cfg.intermediateSize
  let headDim := cfg.headDim 0
  let vocab := cfg.vocabSize

  -- Golden dump hook: write a buffer's first `nFloats` f32 elements to
  -- `$HESPER_GOLDEN_DUMP_DIR/<name>.bin` when the env var is set.  Names
  -- match llama.cpp's `cb()` tags in `llm_build_gemma4_iswa.cpp`.
  let goldenDumpDir ← IO.getEnv "HESPER_GOLDEN_DUMP_DIR"
  let dumpGolden : String → GPUBackend.Buf β → Nat → IO Unit :=
    fun name buf nFloats => do
      match goldenDumpDir with
      | some dir =>
        GPUBackend.endBatch ctx
        let data ← GPUBackend.readBuffer ctx buf (nFloats * 4).toUSize
        IO.FS.writeBinFile s!"{dir}/{name}.bin" data
      | none => pure ()

  -- buffer aliases (arbitrary — stubs, DCE-safe)
  let buf1 := state.buf1
  let buf2 := state.buf2
  let qBuf := state.qBuf
  let kBuf := state.kBuf
  let vBuf := state.vBuf
  let attnOutBuf := state.attnOutBuf
  let attnResidBuf := state.attnResidualBuf
  let pleOutBuf := state.plProjBuf

  -- Batch buffer: column-major `[hidden, seqLen]` holding embedded tokens
  -- and their per-op transforms.  Allocated fresh each forward so the
  -- stub is isolated from any production state.
  -- Scratch allocator: if the caller provided a pool, reuse its slots
  -- (zero cuMemAlloc in steady state).  Otherwise fall back to direct
  -- cuMemAlloc per call.
  let mkBuf := fun (n : Nat) => match scratchPool with
    | some pool => pool.alloc n
    | none      => GPUBackend.allocBuffer ctx (n * 4).toUSize
  let totalHidden := hidden * seqLen
  let batchBuf1 ← mkBuf totalHidden
  let batchBuf2 ← mkBuf totalHidden

  ------------------------------------------------------------------
  -- Prelude (once)
  ------------------------------------------------------------------

  match tokenIdsBuf with
  | some tokIds =>
    -- Real prelude: embedding lookup (per token) + batch scale by
    -- sqrt(hidden).  Matches `build_inp_embd` + `ggml_scale` in
    -- gemma4-iswa.cpp:10-14.
    --
    -- Per-token loop: for each i in [0, seqLen), copy tokIds[i] into
    -- a 1-element scratch, run Q6_K embedding lookup into a dim-long
    -- scratch, then insert that column into batchBuf1 at column i.
    let colIdxBuf ← mkBuf 1 -- 4 bytes, u32
    let tokenScratch ← mkBuf 1 -- 4 bytes, u32 (overwritten per token)
    let embdScratch ← mkBuf hidden
    for i in [0:seqLen] do
      -- colIdxBuf := i
      let iBytes := Hesper.WebGPU.BufferOps.uint32ToBytes i.toUInt32
      GPUBackend.writeBuffer ctx colIdxBuf iBytes
      -- tokenScratch := tokIds[i]
      GPUBackend.execute ctx (stubCopyU32Kernel seqLen 1 0)
        [("src", tokIds), ("params", colIdxBuf), ("dst", tokenScratch)]
        { workgroupSize := { x := 1, y := 1, z := 1 }, numWorkgroups := (1, 1, 1) }
      -- embdScratch := dequant_q6_k(embedding_table[tokenScratch[0]])
      match model.embdFormat with
      | .Q6_K =>
        GPUBackend.execute ctx
          (Hesper.Quantization.Q6_K.q6kEmbeddingLookupKernel cfg.vocabSize hidden)
          [ ("token_ids", tokenScratch)
          , ("embedding_table", model.embedding.embeddingTable)
          , ("output", embdScratch) ]
          (.dispatch1D hidden)
      | _ =>
        -- Non-Q6_K embedding path: delegate to Embedding.forward.
        -- (Note: this path is reached only for alternate quantizations; the
        -- stub is Q6_K-only per Gemma 4 e4b configuration.)
        Hesper.Layers.Embedding.forward ctx model.embedding tokenScratch embdScratch 1 1
      -- batchBuf1[:, i] := embdScratch
      GPUBackend.execute ctx (stubColumnInsertKernel hidden seqLen)
        [("src", embdScratch), ("params", colIdxBuf), ("batch", batchBuf1)]
        (.dispatch1D hidden)

    -- Batch-wide scale by sqrt(hidden).  batchBuf2 := batchBuf1 * sqrt(hidden).
    GPUBackend.execute ctx (stubEmbedScaleKernel totalHidden hidden)
      [("input", batchBuf1), ("output", batchBuf2)]
      (.dispatch1D totalHidden)

    -- Parity checkpoint: `inp_scaled` must match llama.cpp.
    dumpGolden "inp_scaled" batchBuf2 totalHidden

  | none =>
    -- Dispatch-count-only mode: keep the earlier DCE-safe stub behaviour
    -- so the skeleton can still be run without a real prompt.
    dispatchPointwise ctx buf1 buf2 hidden
    dispatchPointwise ctx buf2 buf1 hidden

  -- project_per_layer_inputs (if tok_embd_per_layer).  Matches
  -- llama.cpp's `project_per_layer_inputs` which is 1 batched matmul +
  -- 1 batched scale + 1 batched chunked-rmsnorm + 1 batched scaled-add,
  -- plus per-token Q6_K dequant for the `inp_per_layer_selected` term.
  let totalPL := cfg.embdPerLayer * numLayers
  let batchPLInputAll ← mkBuf (seqLen * totalPL)
  let batchPLInpSelected ← mkBuf (seqLen * totalPL)  -- Q6_K dequant → batched
  let batchPLProj ← mkBuf (seqLen * totalPL)         -- matmul output → batched
  let batchPLNormed ← mkBuf (seqLen * totalPL)       -- rms-normed → batched
  let plColIdxBuf ← mkBuf 1

  match tokenIdsBuf, model.perLayerEmbdTableGPU, model.perLayerModelProj, model.perLayerProjNorm with
  | some _, some embdTableGPU, some modelProj, some projNorm =>
    let embdPL := cfg.embdPerLayer
    let nLayers := numLayers
    let scaleFactor : Float := Float.sqrt embdPL.toFloat
    -- 1. Per-token Q6_K dequant + scale into [seqLen × totalPL] batch.
    --    (q6kTableRowDequantScaleKernel is per-row; we call it seqLen
    --    times with different token IDs and column-insert the result.
    --    Cheap relative to the matmul, but could be batched in a future
    --    pass if dequantQ6KElement becomes public.)
    let tokenScratch2 ← mkBuf 1
    for i in [0:seqLen] do
      let iBytes := Hesper.WebGPU.BufferOps.uint32ToBytes i.toUInt32
      GPUBackend.writeBuffer ctx plColIdxBuf iBytes
      match tokenIdsBuf with
      | some tokIds =>
        GPUBackend.execute ctx (stubCopyU32Kernel seqLen 1 0)
          [("src", tokIds), ("params", plColIdxBuf), ("dst", tokenScratch2)]
          { workgroupSize := { x := 1, y := 1, z := 1 }, numWorkgroups := (1, 1, 1) }
      | none => pure ()
      -- Dequant row of tok_embd_per_layer directly into slice i of the
      -- batch buffer (via columnExtract-style access).  Use a temp totalPL
      -- scratch then column-insert.
      let tmp ← mkBuf totalPL
      GPUBackend.execute ctx
        (Hesper.Quantization.Q6_K.q6kTableRowDequantScaleKernel totalPL scaleFactor
          cfg.vocabSize)
        [("table", embdTableGPU), ("params", tokenScratch2), ("output", tmp)]
        (.dispatch1D totalPL)
      GPUBackend.execute ctx (stubBatchInsertKernel totalPL (seqLen * totalPL))
        [("src", tmp), ("params", plColIdxBuf), ("dst", batchPLInpSelected)]
        (.dispatch1D totalPL)

    -- 2. ONE batched matmul: per_layer_model_proj × batchBuf2[:, :]
    --    → batchPLProj.  batchBuf2 is [hidden, seqLen] (col-major), the
    --    matmul expects A[M, K] with M=seqLen, K=hidden, N=totalPL.
    --    BlockCoop is M=1 only, so use the general F16 matmul.
    let projConfig : Hesper.WGSL.MatMul.Config := {
      M := seqLen, N := totalPL, K := hidden
    }
    Hesper.WGSL.MatMul.executeMatMulTransposeF16 ctx
      batchBuf2 modelProj batchPLProj projConfig

    -- 3. ONE batched scale by 1/√hidden.
    GPUBackend.execute ctx
      (stubBatchScaleKernel (seqLen * totalPL) (1.0 / Float.sqrt hidden.toFloat))
      [("input", batchPLProj), ("output", batchPLNormed)]
      (.dispatch1D (seqLen * totalPL))

    -- 4. ONE batched chunked RMSNorm: grid (nLayers, seqLen, 1).
    GPUBackend.execute ctx
      (stubChunkedRMSNormBatchKernel embdPL nLayers seqLen cfg.rmsNormEps)
      [("input", batchPLNormed), ("weight", projNorm.scale),
       ("output", batchPLProj)]
      { numWorkgroups := (nLayers, seqLen, 1),
        workgroupSize := { x := min embdPL 256, y := 1, z := 1 } : Hesper.ExecConfig }

    -- 5. ONE batched scaled-add: (normed + inp_per_layer_selected) * (1/√2)
    --    → batchPLInputAll (layout-compatible with slice kernel).
    GPUBackend.execute ctx
      (stubScaledAddBatchKernel (seqLen * totalPL) (1.0 / Float.sqrt 2.0))
      [("a", batchPLProj), ("b", batchPLInpSelected),
       ("output", batchPLInputAll)]
      (.dispatch1D (seqLen * totalPL))
  | _, _, _, _ =>
    -- No PLE table / no prompt: emit the earlier DCE-safe stubs so dispatch
    -- count stays consistent.
    dispatchPointwise ctx buf1 buf2 hidden     -- ggml_get_rows
    dispatchPointwise ctx buf2 buf1 hidden     -- ggml_scale
    dispatchMulMat ctx buf1 buf2 seqLen hidden hidden
    dispatchPointwise ctx buf2 buf1 hidden     -- ggml_scale
    dispatchRmsNorm   ctx buf1 buf2 hidden     -- per_layer_proj_norm
    dispatchPointwise ctx buf2 buf1 hidden     -- ggml_add
    dispatchPointwise ctx buf1 buf2 hidden     -- ggml_scale
    dispatchPointwise ctx buf2 buf1 hidden     -- ggml_cont

  -- build_inp_out_ids → no kernel

  ------------------------------------------------------------------
  -- Per-layer loop (42 iterations)
  ------------------------------------------------------------------
  -- Current-layer input (`inpL` in llama.cpp terminology).  Ping-pong
  -- between two buffers: one holds the current layer's input, the other
  -- receives this layer's output.  Starts with batchBuf2 (scaled embed)
  -- holding L0's input.
  let layerIOBufA := batchBuf2
  let layerIOBufB ← mkBuf totalHidden
  let currentInputRef ← IO.mkRef layerIOBufA
  let nextOutputRef ← IO.mkRef layerIOBufB
  -- Per-layer scratch for normed activations.  Same shape as batchBuf1/2.
  let batchNormedBuf ← mkBuf totalHidden
  -- Q/K/V projection scratch sized for the MAX across all 42 layers (SWA
  -- and Full layers differ in numKVHeads and potentially headDim).  CUDA
  -- ignores declared buffer sizes, so allocating at the max is safe for
  -- every per-layer kernel dispatch.
  let maxHeadDim := max cfg.headDimFull cfg.headDimSWA
  let qDim0 := cfg.numAttentionHeads * maxHeadDim
  let batchQBuf ← mkBuf (qDim0 * seqLen)
  let batchQRopedBuf ← mkBuf (qDim0 * seqLen)
  -- kvDim = numKVHeads * headDim.  Use max of Full / SWA.
  let maxKVHeads := max cfg.numKeyValueHeadsFull cfg.numKeyValueHeadsSWA
  let kvDim0 := maxKVHeads * maxHeadDim
  let batchKBuf ← mkBuf (kvDim0 * seqLen)
  let batchVBuf ← mkBuf (kvDim0 * seqLen)
  let batchKRopedBuf ← mkBuf (kvDim0 * seqLen)
  let batchAttnOutBuf ← mkBuf (qDim0 * seqLen)           -- fattn output
  let batchOProjBuf ← mkBuf (hidden * seqLen)            -- wO output
  let batchAttnPostNormBuf ← mkBuf (hidden * seqLen)     -- after attn_post_norm
  let batchAttnResidBuf ← mkBuf (hidden * seqLen)        -- after residual add (= attn_out)
  let batchFfnNormBuf ← mkBuf (hidden * seqLen)          -- ffn_norm output
  let batchFfnGateBuf ← mkBuf (inter * seqLen)
  let batchFfnUpBuf ← mkBuf (inter * seqLen)
  let batchFfnGegluBuf ← mkBuf (inter * seqLen)
  let batchFfnOutBuf ← mkBuf (hidden * seqLen)
  let batchFfnPostNormBuf ← mkBuf (hidden * seqLen)
  let batchFfnResidBuf ← mkBuf (hidden * seqLen)         -- after FFN residual (= pe_in)
  let batchPleGateBuf ← mkBuf (cfg.embdPerLayer * seqLen)
  let batchPleGeluBuf ← mkBuf (cfg.embdPerLayer * seqLen)
  let batchPleMulBuf ← mkBuf (cfg.embdPerLayer * seqLen)
  let batchPleProjBuf ← mkBuf (hidden * seqLen)
  let batchPleNormBuf ← mkBuf (hidden * seqLen)
  let batchLOutBuf ← mkBuf (hidden * seqLen)
  -- KV caches: one K and V buffer per "own-KV" layer.  Shared-KV
  -- layers (last numKVSharedLayers layers in Gemma 4 E4B) reuse these
  -- via `cfg.kvCacheLayer`.  Sized at maxKVHeads × maxSeqLen × maxHeadDim
  -- so `flashAttentionBatchKernel`'s `kvHead * maxSeqLen * headDim`
  -- stride is always in-bounds.  When `persistentCaches` is provided
  -- (decode loop) we reuse the caller's buffers instead of allocating.
  let maxSeqLenUsed := cfg.maxSeqLen
  let cacheSize := maxKVHeads * maxSeqLenUsed * maxHeadDim
  let ownKVLayers := numLayers - cfg.numKVSharedLayers
  let mut kCaches : Array (GPUBackend.Buf β) := Array.empty
  let mut vCaches : Array (GPUBackend.Buf β) := Array.empty
  match persistentCaches with
  | some arr =>
    for (k, v) in arr do
      kCaches := kCaches.push k
      vCaches := vCaches.push v
  | none =>
    for _ in [0:ownKVLayers] do
      kCaches := kCaches.push (← mkBuf cacheSize)
      vCaches := vCaches.push (← mkBuf cacheSize)
  -- For backwards-compatible naming with earlier single-cache code:
  let kCacheBuf ← if h : 0 < kCaches.size then pure kCaches[0] else mkBuf cacheSize
  let vCacheBuf ← if h : 0 < vCaches.size then pure vCaches[0] else mkBuf cacheSize
  -- Params buffer: holds `startPos` (u32) for RoPE / KV cache indexing.
  -- When `paramsBufOverride` is provided (graph-capture path), use it and
  -- skip the write — caller owns the pre-launch update.  Otherwise alloc
  -- from the pool and write startPos ourselves.
  let paramsBuf ← match paramsBufOverride with
    | some b => pure b
    | none   => mkBuf 1
  match paramsBufOverride with
  | some _ => pure ()
  | none =>
    GPUBackend.writeBuffer ctx paramsBuf
      (Hesper.WebGPU.BufferOps.uint32ToBytes startPos.toUInt32)
  -- Ones buffer: `headDim/2` f32 elements all set to 1.0, used as
  -- `freq_factors` for SWA layers (which llama.cpp calls with nullptr =
  -- equivalent to 1.0 everywhere).  Pattern copied from production.
  let headDim0 := cfg.headDim 0
  let dimPairs0 := headDim0 / 2
  let onesBuf ← mkBuf dimPairs0
  do
    let oneBytes ← Hesper.WebGPU.BufferOps.floatToBytes 1.0
    let mut bytes : ByteArray := ByteArray.empty
    for _ in [0:dimPairs0] do
      bytes := bytes ++ oneBytes
    GPUBackend.writeBuffer ctx onesBuf bytes

  for il in List.range numLayers do
    -- cur = build_norm(inpL, attn_norm)
    -- inpL = `currentInputRef` (scaled embed for L0, previous layer's
    -- `l_out` for L1+).  Output into batchNormedBuf; dump `attn_norm-{il}`.
    let layerInputBuf ← currentInputRef.get
    match tokenIdsBuf with
    | some _ =>
      if h : il < model.blocks.size then
        let block := model.blocks[il]
        Hesper.Layers.RMSNorm.forward ctx block.attnNorm layerInputBuf batchNormedBuf
          seqLen 256 (refOverride := some (← IO.mkRef none))
        dumpGolden s!"attn_norm-{il}" batchNormedBuf totalHidden
      else
        dispatchRmsNorm ctx buf1 buf2 hidden
    | none =>
      dispatchRmsNorm ctx buf1 buf2 hidden

    -- Qcur = build_lora_mm(wq, cur)       [Q4_K batched matmul]
    -- Real implementation for L0 when tokens provided; stub otherwise.
    match tokenIdsBuf with
    | some _ =>
      if h : il < model.blocks.size then
        let block := model.blocks[il]
        Hesper.Layers.Linear.forwardBatchDP4A ctx block.attention.wQ
          batchNormedBuf batchQBuf seqLen
        dumpGolden s!"Qcur-{il}" batchQBuf (qDim0 * seqLen)
      else
        dispatchMulMat ctx buf2 qBuf seqLen hidden hidden
    | none =>
      dispatchMulMat ctx buf2 qBuf seqLen hidden hidden

    -- Qcur = build_norm(Qcur, attn_q_norm)  [per-head RMSNorm across headDim]
    match tokenIdsBuf with
    | some _ =>
      if h : il < model.blocks.size then
        let block := model.blocks[il]
        let numHeads := cfg.numAttentionHeads
        let headDim := cfg.headDim il
        GPUBackend.execute ctx
          (perHeadRMSNormBatchKernel numHeads headDim seqLen cfg.rmsNormEps)
          [("input", batchQBuf), ("weight", block.attention.qNormWeight),
           ("output", batchQBuf)]
          { numWorkgroups := (numHeads, seqLen, 1),
            workgroupSize := { x := (if headDim < 256 then headDim else 256),
                               y := 1, z := 1 } : Hesper.ExecConfig }
        dumpGolden s!"Qcur_normed-{il}" batchQBuf (qDim0 * seqLen)
      else
        dispatchRmsNorm ctx qBuf qBuf hidden
    | none =>
      dispatchRmsNorm ctx qBuf qBuf hidden
    -- ggml_reshape_3d(Qcur)                [no kernel]
    -- Qcur = ggml_rope_ext(Qcur, ...)
    -- For SWA layers llama.cpp passes freq_factors=nullptr; we feed a
    -- ones-filled buffer so `ropeWithFreqFactorsBatchKernel` behaves
    -- identically (pattern matches production's `onesBuf` handling).
    match tokenIdsBuf with
    | some _ =>
      if h : il < model.blocks.size then
        let block := model.blocks[il]
        let numHeads := cfg.numAttentionHeads
        let headDim := cfg.headDim il
        let freqFactors := block.ropeFreqFactors.getD onesBuf
        GPUBackend.execute ctx
          (ropeWithFreqFactorsBatchKernel headDim numHeads seqLen (cfg.ropeBase il))
          [("input", batchQBuf), ("output", batchQRopedBuf),
           ("params", paramsBuf), ("freq_factors", freqFactors)]
          (.dispatch1D (numHeads * headDim / 2 * seqLen))
        dumpGolden s!"Qcur_pos-{il}" batchQRopedBuf (qDim0 * seqLen)
      else
        GPUBackend.execute ctx (llamaRopeQBatchedKernel 1 hidden)
          [("q_in", qBuf), ("q_roped", qBuf)]
          { workgroupSize := { x := 1, y := 1, z := 1 }, numWorkgroups := (1, 1, 1) }
    | none =>
      GPUBackend.execute ctx (llamaRopeQBatchedKernel 1 hidden)
        [("q_in", qBuf), ("q_roped", qBuf)]
        { workgroupSize := { x := 1, y := 1, z := 1 }, numWorkgroups := (1, 1, 1) }

    -- if has_kv (24 of 42 layers)
    -- For stub purposes emit KV branch unconditionally; llama.cpp's 2016
    -- count reflects the full layer mix.

    -- wK / wV / k_norm / v_rms_norm / rope_K: ONLY for own-KV layers.
    -- Shared-KV layers (last numKVSharedLayers layers in Gemma 4 E4B)
    -- reuse the prior layer's KV cache unchanged.
    match tokenIdsBuf with
    | some _ =>
      if h : il < model.blocks.size then
        if cfg.hasKV il then
          let block := model.blocks[il]
          -- Kcur = build_lora_mm(wk, cur)
          Hesper.Layers.Linear.forwardBatchDP4A ctx block.attention.wK
            batchNormedBuf batchKBuf seqLen
          dumpGolden s!"Kcur-{il}" batchKBuf (kvDim0 * seqLen)
          -- Vcur = build_lora_mm(wv, cur)
          Hesper.Layers.Linear.forwardBatchDP4A ctx block.attention.wV
            batchNormedBuf batchVBuf seqLen
          dumpGolden s!"Vcur-{il}" batchVBuf (kvDim0 * seqLen)
          -- ggml_reshape_3d × 2                  [no kernel]
          -- Kcur = build_norm(Kcur, attn_k_norm)
          let numKVH := cfg.numKVHeads il
          let headDim := cfg.headDim il
          GPUBackend.execute ctx
            (perHeadRMSNormBatchKernel numKVH headDim seqLen cfg.rmsNormEps)
            [("input", batchKBuf), ("weight", block.attention.kNormWeight),
             ("output", batchKBuf)]
            { numWorkgroups := (numKVH, seqLen, 1),
              workgroupSize := { x := (if headDim < 256 then headDim else 256),
                                 y := 1, z := 1 } : Hesper.ExecConfig }
          dumpGolden s!"Kcur_normed-{il}" batchKBuf (kvDim0 * seqLen)
          -- Vcur = ggml_rms_norm(Vcur)  [bare RMSNorm, no learned weight]
          GPUBackend.execute ctx
            (stubPerHeadBareRMSNormBatchKernel numKVH headDim seqLen cfg.rmsNormEps)
            [("input", batchVBuf), ("output", batchVBuf)]
            { numWorkgroups := (numKVH, seqLen, 1),
              workgroupSize := { x := (if headDim < 256 then headDim else 256),
                                 y := 1, z := 1 } : Hesper.ExecConfig }
          dumpGolden s!"Vcur_normed-{il}" batchVBuf (kvDim0 * seqLen)
          -- Kcur = ggml_rope_ext(Kcur, ...)
          let freqFactors := block.ropeFreqFactors.getD onesBuf
          GPUBackend.execute ctx
            (ropeWithFreqFactorsBatchKernel headDim numKVH seqLen (cfg.ropeBase il))
            [("input", batchKBuf), ("output", batchKRopedBuf),
             ("params", paramsBuf), ("freq_factors", freqFactors)]
            (.dispatch1D (numKVH * headDim / 2 * seqLen))
          dumpGolden s!"Kcur_pos-{il}" batchKRopedBuf (kvDim0 * seqLen)
      else
        dispatchMulMat ctx buf2 kBuf seqLen hidden hidden
    | none =>
      dispatchMulMat ctx buf2 kBuf seqLen hidden hidden

    -- cur = build_attn(Qcur_pos, Kcur_pos, Vcur_normed, wo, ...)
    -- Sub-steps: KV cache write × 2 (own-KV layers only) + flash_attn + wO matmul
    match tokenIdsBuf with
    | some _ =>
      if h : il < model.blocks.size then
        let block := model.blocks[il]
        let numHeads := cfg.numAttentionHeads
        -- Determine which layer's KV cache to use.  For own-KV layers
        -- this is `il` itself; for shared-KV layers it's an earlier
        -- full-attn or SWA layer per `cfg.kvSharedFromBase`.
        let kvLi := cfg.kvCacheLayer il
        let numKVH := cfg.numKVHeads kvLi
        let headDim := cfg.headDim kvLi
        let kCacheL := if h2 : kvLi < kCaches.size then kCaches[kvLi] else kCacheBuf
        let vCacheL := if h2 : kvLi < vCaches.size then vCaches[kvLi] else vCacheBuf
        -- KV cache writes only for own-KV layers; shared-KV layers read
        -- from the prior layer's cache unchanged.
        if cfg.hasKV il then
          -- 1. K cache write
          GPUBackend.execute ctx
            (stubKVCacheWriteBatchKernel numKVH cfg.maxSeqLen headDim seqLen)
            [("new_data", batchKRopedBuf), ("cache", kCacheL), ("params", paramsBuf)]
            { numWorkgroups := (seqLen, 1, 1),
              workgroupSize := { x := (if (numKVH * headDim) < 256 then numKVH * headDim else 256),
                                 y := 1, z := 1 } : Hesper.ExecConfig }
          -- 2. V cache write
          GPUBackend.execute ctx
            (stubKVCacheWriteBatchKernel numKVH cfg.maxSeqLen headDim seqLen)
            [("new_data", batchVBuf), ("cache", vCacheL), ("params", paramsBuf)]
            { numWorkgroups := (seqLen, 1, 1),
              workgroupSize := { x := (if (numKVH * headDim) < 256 then numKVH * headDim else 256),
                                 y := 1, z := 1 } : Hesper.ExecConfig }
        -- 3. Flash attention (batched across seqLen query tokens).
        -- Use the cache we just wrote (own-KV) or the prior layer's
        -- cache (shared-KV) as K and V inputs.
        let scale : Float := 1.0
        GPUBackend.execute ctx
          (Hesper.WGSL.FlashAttention.flashAttentionBatchKernel numHeads numKVH
             cfg.maxSeqLen headDim seqLen scale)
          [("q", batchQRopedBuf), ("k_cache", kCacheL), ("v_cache", vCacheL),
           ("output", batchAttnOutBuf), ("params", paramsBuf)]
          { numWorkgroups := (numHeads, seqLen, 1),
            workgroupSize := { x := 256, y := 1, z := 1 } : Hesper.ExecConfig }
        dumpGolden s!"__fattn__-{il}" batchAttnOutBuf (qDim0 * seqLen)
        -- 4. wO matmul: [qDim × seqLen] → [hidden × seqLen]
        Hesper.Layers.Linear.forwardBatchDP4A ctx block.attention.wO
          batchAttnOutBuf batchOProjBuf seqLen
      else
        dispatchBuildAttn ctx qBuf kBuf vBuf attnOutBuf seqLen headDim hidden
    | none =>
      dispatchBuildAttn ctx qBuf kBuf vBuf attnOutBuf seqLen headDim hidden

    -- build_norm(cur, attn_post_norm)
    match tokenIdsBuf with
    | some _ =>
      if h : il < model.blocks.size then
        let block := model.blocks[il]
        Hesper.Layers.RMSNorm.forward ctx block.postAttnNorm
          batchOProjBuf batchAttnPostNormBuf seqLen 256
          (refOverride := some (← IO.mkRef none))
      else
        dispatchRmsNorm ctx attnOutBuf buf1 hidden
    | none =>
      dispatchRmsNorm ctx attnOutBuf buf1 hidden

    -- attn_out = ggml_add(cur, inpL)  — residual: post-norm'd wO output +
    -- the current layer's input (llama.cpp line 112).
    match tokenIdsBuf with
    | some _ =>
      GPUBackend.execute ctx (Hesper.Models.Gemma4.residualAddKernel totalHidden)
        [("a", batchAttnPostNormBuf), ("b", layerInputBuf), ("output", batchAttnResidBuf)]
        (.dispatch1D totalHidden)
      dumpGolden s!"attn_out-{il}" batchAttnResidBuf totalHidden
    | none =>
      dispatchPointwise ctx buf1 attnResidBuf hidden

    -- FFN: build_norm(attn_out, ffn_norm)
    match tokenIdsBuf with
    | some _ =>
      if h : il < model.blocks.size then
        let block := model.blocks[il]
        Hesper.Layers.RMSNorm.forward ctx block.ffnNorm
          batchAttnResidBuf batchFfnNormBuf seqLen 256
          (refOverride := some (← IO.mkRef none))
        dumpGolden s!"ffn_norm-{il}" batchFfnNormBuf totalHidden
      else
        dispatchRmsNorm ctx attnResidBuf buf1 hidden
    | none =>
      dispatchRmsNorm ctx attnResidBuf buf1 hidden

    -- build_ffn(...): gate matmul, up matmul, geglu, down matmul
    match tokenIdsBuf with
    | some _ =>
      if h : il < model.blocks.size then
        let block := model.blocks[il]
        -- ffn_gate matmul
        Hesper.Layers.Linear.forwardBatchDP4A ctx block.ffn.gate
          batchFfnNormBuf batchFfnGateBuf seqLen
        dumpGolden s!"ffn_gate-{il}" batchFfnGateBuf (inter * seqLen)
        -- ffn_up matmul
        Hesper.Layers.Linear.forwardBatchDP4A ctx block.ffn.up
          batchFfnNormBuf batchFfnUpBuf seqLen
        dumpGolden s!"ffn_up-{il}" batchFfnUpBuf (inter * seqLen)
        -- GEGLU: gelu(gate) * up
        GPUBackend.execute ctx (stubGegluKernel (inter * seqLen))
          [("gate", batchFfnGateBuf), ("up", batchFfnUpBuf),
           ("output", batchFfnGegluBuf)]
          (.dispatch1D (inter * seqLen))
        dumpGolden s!"ffn_geglu-{il}" batchFfnGegluBuf (inter * seqLen)
        -- ffn_down matmul
        Hesper.Layers.Linear.forwardBatchDP4A ctx block.ffn.down
          batchFfnGegluBuf batchFfnOutBuf seqLen
        dumpGolden s!"ffn_out-{il}" batchFfnOutBuf totalHidden
      else
        dispatchBuildFfn ctx buf1 buf2 seqLen hidden inter
    | none =>
      dispatchBuildFfn ctx buf1 buf2 seqLen hidden inter

    -- build_norm(cur, ffn_post_norm)
    match tokenIdsBuf with
    | some _ =>
      if h : il < model.blocks.size then
        let block := model.blocks[il]
        Hesper.Layers.RMSNorm.forward ctx block.postFFNNorm
          batchFfnOutBuf batchFfnPostNormBuf seqLen 256
          (refOverride := some (← IO.mkRef none))
        dumpGolden s!"ffn_post_norm-{il}" batchFfnPostNormBuf totalHidden
      else
        dispatchRmsNorm ctx buf2 buf1 hidden
    | none =>
      dispatchRmsNorm ctx buf2 buf1 hidden

    -- residual: ggml_add(ffn_post_norm, attn_out) → pe_in
    match tokenIdsBuf with
    | some _ =>
      GPUBackend.execute ctx (Hesper.Models.Gemma4.residualAddKernel totalHidden)
        [("a", batchFfnPostNormBuf), ("b", batchAttnResidBuf),
         ("output", batchFfnResidBuf)]
        (.dispatch1D totalHidden)
      dumpGolden s!"pe_in-{il}" batchFfnResidBuf totalHidden
    | none =>
      dispatchPointwise ctx buf1 attnResidBuf hidden

    -- PLE block (gemma4-iswa.cpp:193-213, always present for Gemma 4).
    -- Input: `pe_in` (= attnResidBuf after FFN residual).  Reads per-layer
    -- input slice from batchPLInputAll; matmuls through inpGate, applies
    -- gelu, multiplies by inp_this_layer slice, matmuls through proj,
    -- applies post_norm, and residual-adds back onto pe_in.
    match tokenIdsBuf with
    | some _ =>
      if h : il < model.blocks.size then
        let _ := model.blocks[il]
        let pleOpt := if h2 : il < model.perLayerBlocks.size then
          model.perLayerBlocks[il] else none
        match pleOpt, model.perLayerEmbdTableGPU with
        | some pleBlock, some _ =>
          let embdPL := cfg.embdPerLayer
          -- (a) inpGate matmul: pe_in [hidden × seqLen] → [embdPL × seqLen]
          Hesper.Layers.Linear.forwardBatchDP4A ctx pleBlock.inpGate
            batchFfnResidBuf batchPleGateBuf seqLen
          -- (b) gelu in-place
          GPUBackend.execute ctx (stubGeluKernel (embdPL * seqLen))
            [("input", batchPleGateBuf), ("output", batchPleGeluBuf)]
            (.dispatch1D (embdPL * seqLen))
          -- (c) extract layer-il slice from batchPLInputAll into plSliceBuf
          let plSliceBuf ← mkBuf (embdPL * seqLen)
          GPUBackend.execute ctx
            (stubPerLayerSliceKernel embdPL seqLen numLayers il)
            [("src", batchPLInputAll), ("dst", plSliceBuf)]
            (.dispatch1D (embdPL * seqLen))
          -- (d) elementwise mul: gelu * inp_this_layer
          GPUBackend.execute ctx (stubMulKernel (embdPL * seqLen))
            [("a", batchPleGeluBuf), ("b", plSliceBuf),
             ("output", batchPleMulBuf)]
            (.dispatch1D (embdPL * seqLen))
          -- (e) proj matmul: [embdPL × seqLen] → [hidden × seqLen]
          Hesper.Layers.Linear.forwardBatchDP4A ctx pleBlock.proj
            batchPleMulBuf batchPleProjBuf seqLen
          -- (f) per_layer_post_norm (RMSNorm across hidden)
          Hesper.Layers.RMSNorm.forward ctx pleBlock.postNorm
            batchPleProjBuf batchPleNormBuf seqLen 256
            (refOverride := some (← IO.mkRef none))
          dumpGolden s!"per_layer_embd_out-{il}" batchPleNormBuf totalHidden
          -- (g) residual: pe_in + pleNorm → batchLOutBuf (scratch before
          -- out_scale).
          GPUBackend.execute ctx (Hesper.Models.Gemma4.residualAddKernel totalHidden)
            [("a", batchFfnResidBuf), ("b", batchPleNormBuf),
             ("output", batchLOutBuf)]
            (.dispatch1D totalHidden)
        | _, _ =>
          -- No PLE block for this layer: just copy pe_in through.
          dispatchPointwise ctx attnResidBuf pleOutBuf hidden
          dispatchPointwise ctx pleOutBuf buf1 hidden
          dispatchPointwise ctx buf1 buf2 hidden
          dispatchMulMat ctx buf2 pleOutBuf seqLen hidden hidden
          dispatchRmsNorm ctx pleOutBuf buf1 hidden
          dispatchPointwise ctx buf1 attnResidBuf hidden
      else
        dispatchMulMat ctx attnResidBuf pleOutBuf seqLen hidden hidden
        dispatchPointwise ctx pleOutBuf buf1 hidden
        dispatchPointwise ctx buf1 buf2 hidden
        dispatchMulMat ctx buf2 pleOutBuf seqLen hidden hidden
        dispatchRmsNorm ctx pleOutBuf buf1 hidden
        dispatchPointwise ctx buf1 attnResidBuf hidden
    | none =>
      dispatchMulMat ctx attnResidBuf pleOutBuf seqLen hidden hidden
      dispatchPointwise ctx pleOutBuf buf1 hidden
      dispatchPointwise ctx buf1 buf2 hidden
      dispatchMulMat ctx buf2 pleOutBuf seqLen hidden hidden
      dispatchRmsNorm ctx pleOutBuf buf1 hidden
      dispatchPointwise ctx buf1 attnResidBuf hidden

    -- layer_scalar: if out_scale → ggml_mul (broadcast scalar over hidden × seqLen)
    -- The scaled result is written to the "next layer input" buffer
    -- (ping-pong), so the next iteration's attn_norm reads from l_out.
    match tokenIdsBuf with
    | some _ =>
      if h : il < model.blocks.size then
        let block := model.blocks[il]
        let nextInputBuf ← nextOutputRef.get
        match block.outScale with
        | some outScale =>
          -- Apply out_scale: nextInputBuf = batchLOutBuf * outScale[0]
          GPUBackend.execute ctx (stubBroadcastScaleKernel totalHidden)
            [("input", batchLOutBuf), ("scale", outScale),
             ("output", nextInputBuf)]
            (.dispatch1D totalHidden)
        | none =>
          -- No out_scale: identity copy via residualAdd(a, 0).  We reuse
          -- the residualAddKernel with b = a; the next layer will compute
          -- attn_norm(2a) instead of attn_norm(a) which breaks parity.
          -- TODO: add a copy kernel.  For now, if out_scale is absent we
          -- fall back to batchLOutBuf directly (set ref without copy).
          GPUBackend.execute ctx (Hesper.Models.Gemma4.residualAddKernel totalHidden)
            [("a", batchLOutBuf), ("b", batchLOutBuf),
             ("output", nextInputBuf)]
            (.dispatch1D totalHidden)
        dumpGolden s!"l_out-{il}" nextInputBuf totalHidden
        -- Swap ping-pong refs: next layer's input becomes current input.
        currentInputRef.set nextInputBuf
        nextOutputRef.set (← pure layerInputBuf)
      else
        dispatchPointwise ctx attnResidBuf buf1 hidden
    | none =>
      dispatchPointwise ctx attnResidBuf buf1 hidden
    -- build_cvec                           [no kernel in practice]

  ------------------------------------------------------------------
  -- Post-loop (once): output_norm + lm_head (+ optional softcap)
  ------------------------------------------------------------------
  match tokenIdsBuf with
  | some _ =>
    -- llama.cpp applies `inp_out_ids` to trim to just the last token
    -- before output_norm.  We do the same: extract column seqLen-1
    -- from the final layer input (currentInputRef) into a hidden-sized
    -- scratch, then run output_norm + lm_head on that single row.
    let finalInputBuf ← currentInputRef.get
    let lastTokenBuf ← mkBuf hidden
    let lastColIdx := seqLen - 1
    let lastColIdxBuf ← mkBuf 1
    GPUBackend.writeBuffer ctx lastColIdxBuf
      (Hesper.WebGPU.BufferOps.uint32ToBytes lastColIdx.toUInt32)
    GPUBackend.execute ctx (stubColumnExtractKernel hidden seqLen)
      [("batch", finalInputBuf), ("params", lastColIdxBuf),
       ("out", lastTokenBuf)]
      (.dispatch1D hidden)

    -- build_norm(cur, output_norm) [RMSNorm on single-token]
    let resultNormBuf ← mkBuf hidden
    Hesper.Layers.RMSNorm.forward ctx model.finalNorm
      lastTokenBuf resultNormBuf 1 256
      (refOverride := some (← IO.mkRef none))
    dumpGolden "result_norm" resultNormBuf hidden

    -- lm_head: Q6_K matmul [hidden → vocabSize] on single-token f32
    let logitsBuf ← mkBuf vocab
    GPUBackend.execute ctx
      (Hesper.Quantization.Q6_K.fusedQ6KLinearKernel hidden vocab)
      [("weights", model.outputWeight), ("input", resultNormBuf),
       ("output", logitsBuf)]
      { numWorkgroups := (vocab, 1, 1),
        workgroupSize := { x := 256, y := 1, z := 1 } : Hesper.ExecConfig }

    -- softcap: logit = softcap * tanh(logit / softcap)  (if f_final_logit_softcapping)
    -- Gemma 4 has softcap=30.0; apply as one fused kernel that matches
    -- llama.cpp's scale + tanh + scale triple at gemma4-iswa.cpp:239-243.
    -- softcap is monotonic-increasing (tanh), so argmax is preserved
    -- either way, but applying it improves result_output parity from
    -- ~11 % (arbitrary scale) to Q6_K-quant-noise scale.
    let softcap := cfg.logitSoftcapScale
    let softcappedBuf ← if softcap > 0.0 then do
      let buf ← mkBuf vocab
      GPUBackend.execute ctx (stubLogitSoftcapKernel vocab softcap)
        [("input", logitsBuf), ("output", buf)]
        (.dispatch1D vocab)
      pure buf
    else pure logitsBuf
    dumpGolden "result_output" softcappedBuf vocab
    return some softcappedBuf
  | none =>
    -- Dispatch-count mode
    dispatchRmsNorm ctx buf1 buf2 hidden
    GPUBackend.execute ctx (prefillQuantizeMmqQ8_1Q6KKernel seqLen hidden)
      [("lm_mmq_in", buf2), ("lm_mmq_q8", buf1)]
      { workgroupSize := { x := 1, y := 1, z := 1 }, numWorkgroups := (1, 1, 1) }
    GPUBackend.execute ctx (prefillMulMatQQ6KKernel seqLen vocab hidden)
      [("lm_mmq_q8", buf1), ("lm_mmq_part", buf2)]
      { workgroupSize := { x := 1, y := 1, z := 1 }, numWorkgroups := (1, 1, 1) }
    GPUBackend.execute ctx (prefillMulMatQStreamKFixupQ6KKernel seqLen vocab)
      [("lm_mmq_part", buf2), ("lm_mmq_out", buf1)]
      { workgroupSize := { x := 1, y := 1, z := 1 }, numWorkgroups := (1, 1, 1) }
    dispatchPointwise ctx buf1 buf2 hidden
    dispatchPointwise ctx buf2 buf1 hidden
    dispatchPointwise ctx buf1 buf2 hidden
    return none

end Hesper.Models.Gemma4
