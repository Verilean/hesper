import Hesper.WGSL.Monad
import Hesper.WGSL.Exp

/-!
# Phase 0 v3 LlamaPath prefill stub kernels — batched-multilayer

Mirrors llama.cpp's **prefill** physical kernels (distinct from decode).
Measured via `GGML_CUDA_DISABLE_GRAPHS=1 nsys` on Gemma 4 E4B Q4_K_M, pp=50,
tg=0: total ≈ 2016 CUDA kernels per prefill forward.

Key prefill-only kernels (not seen in decode):
- `mul_mat_q<Q4_K,64>`             306 instances  (batched Q4_K matmul)
- `mul_mat_q_stream_k_fixup<Q4_K>` 306 instances  (stream-K reduction)
- `quantize_mmq_q8_1`              306 instances  (mmq-layout quantize)
- `ampere_s16816gemm_fp16`          42 instances  (tensor-core flash_attn)
- `ampere_h1688gemm_256x64`          1 instance   (tensor-core lm_head)
- `unary_gated_op_kernel<gelu>`     41 instances  (fused gate×gelu(up))
- `cpy_scalar`                      43 instances  (f32 copy)
- `cutlass_80_wmma_h161616gemm`     35 instances
- `k_compute_batched_ptrs`          84 instances  (cuBLAS batched ptr setup)
- `convert_unary<float,half>`       85 instances
- `convert_unary<half,float>`       43 instances
- `mul_mat_q<Q6_K,64>` / fixup      31+31 (lm_head prefill for N>1)
- plus shared w/ decode: rms_norm, rope_neox, k_set_rows, soft_max,
  k_bin_bcast<mul>, k_bin_bcast<add>, pad_f32, unary_op<gelu>, etc.

v3 strategy: one stub per **distinct kernel type**, batched-multilayer where
applicable (gridY = numLayers).  Real bodies will be filled in later phases.
-/

namespace Hesper.Models.Gemma4.Llama.Prefill

open Hesper.WGSL
open Hesper.WGSL.Monad
open Hesper.WGSL.Exp

/-! ## Host-side support kernels (copied from production — stub isolation)

These utilities (copyU32, columnInsert) are `private` in
`Hesper/Models/Gemma4.lean`.  Per the stub-first-parity plan
(docs/llama-fusion-analysis/33-stub-first-parity-plan.md), production is
read-only reference — we copy these kernel defs here rather than calling
production code, so the stub owns its full kernel set.
-/

/-- Copy 1 u32 element from `src[params[0]]` → `dst[dstIdx]`. -/
def stubCopyU32Kernel (srcSize : Nat) (dstSize : Nat) (dstIdx : Nat) : ShaderM Unit := do
  let _src    ← ShaderM.declareInputBuffer "src" (.array (.scalar .u32) srcSize)
  let _params ← ShaderM.declareInputBuffer "params" (.array (.scalar .u32) 1)
  let _dst    ← ShaderM.declareOutputBuffer "dst" (.array (.scalar .u32) dstSize)
  let srcIdx ← ShaderM.readBuffer (ty := .scalar .u32) (n := 1) "params" (Exp.litU32 0)
  let v ← ShaderM.readBuffer (ty := .scalar .u32) (n := srcSize) "src" srcIdx
  ShaderM.writeBuffer (ty := .scalar .u32) "dst" (Exp.litU32 dstIdx) v

/-- `out[i] = batch[params[0] * dim + i]` for `i in [0, dim)`.
    Column-extract companion to `stubColumnInsertKernel`. -/
def stubColumnExtractKernel (dim : Nat) (seqLen : Nat) : ShaderM Unit := do
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

/-- Extract a single layer's slice `[embdPerLayer, seqLen]` from a
    batched [seqLen × (embdPerLayer * numLayers)] PLE buffer.
    Layout: `plInputAllBatched[col * totalPL + li * embdPerLayer + d]`
    → `out[col * embdPerLayer + d]` for a fixed `li`.
    Grid: `(embdPerLayer * seqLen, 1, 1)`. -/
def stubPerLayerSliceKernel (embdPerLayer seqLen numLayers layerIdx : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid
  let totalPL := embdPerLayer * numLayers
  let totalOut := embdPerLayer * seqLen
  let _src ← ShaderM.declareInputBuffer "src" (.array (.scalar .f32) (seqLen * totalPL))
  let _dst ← ShaderM.declareOutputBuffer "dst" (.array (.scalar .f32) totalOut)
  ShaderM.if_ (Exp.lt idx (Exp.litU32 totalOut)) (do
    let col := Exp.div idx (Exp.litU32 embdPerLayer)
    let d := Exp.sub idx (Exp.mul col (Exp.litU32 embdPerLayer))
    let srcIdx := Exp.add (Exp.mul col (Exp.litU32 totalPL))
                          (Exp.add (Exp.litU32 (layerIdx * embdPerLayer)) d)
    let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := seqLen * totalPL) "src" srcIdx
    ShaderM.writeBuffer (ty := .scalar .f32) "dst" idx v
  ) (pure ())

/-- Element-wise multiply: `out[i] = a[i] * b[i]` over `size` f32 elements. -/
def stubMulKernel (size : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid
  let _a ← ShaderM.declareInputBuffer "a" (.array (.scalar .f32) size)
  let _b ← ShaderM.declareInputBuffer "b" (.array (.scalar .f32) size)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) size)
  ShaderM.if_ (Exp.lt idx (Exp.litU32 size)) (do
    let a ← ShaderM.readBuffer (ty := .scalar .f32) (n := size) "a" idx
    let b ← ShaderM.readBuffer (ty := .scalar .f32) (n := size) "b" idx
    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx (Exp.mul a b)
  ) (pure ())

/-- Element-wise GELU: `out[i] = gelu(x[i])` over `size` f32 elements. -/
def stubGeluKernel (size : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid
  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) size)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) size)
  ShaderM.if_ (Exp.lt idx (Exp.litU32 size)) (do
    let x ← ShaderM.readBuffer (ty := .scalar .f32) (n := size) "input" idx
    let c0 := Exp.litF32 0.7978845608028654
    let c1 := Exp.litF32 0.044715
    let x3 := Exp.mul (Exp.mul x x) x
    let inner := Exp.mul c0 (Exp.add x (Exp.mul c1 x3))
    let gelu := Exp.mul (Exp.mul x (Exp.litF32 0.5))
                        (Exp.add (Exp.litF32 1.0) (Exp.tanh inner))
    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx gelu
  ) (pure ())

/-- Broadcast scalar multiply: `buf[i] *= scale[0]` over `total` elements.
    Used for per-layer output scale (`out_scale`, a single f32 per layer). -/
def stubBroadcastScaleKernel (total : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid
  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) total)
  let _scale ← ShaderM.declareInputBuffer "scale" (.array (.scalar .f32) 1)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) total)
  ShaderM.if_ (Exp.lt idx (Exp.litU32 total)) (do
    let s ← ShaderM.readBuffer (ty := .scalar .f32) (n := 1) "scale" (Exp.litU32 0)
    let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := total) "input" idx
    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx (Exp.mul v s)
  ) (pure ())

/-- Final-logit softcap: `out[i] = softcap * tanh(in[i] / softcap)`.
    Matches llama.cpp's `scale(1/s) + tanh + scale(s)` chain at
    gemma4-iswa.cpp:239-243.  Fused into one kernel because the three
    ops are pointwise and the constant is compile-time known. -/
def stubLogitSoftcapKernel (size : Nat) (softcap : Float) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid
  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) size)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) size)
  ShaderM.if_ (Exp.lt idx (Exp.litU32 size)) (do
    let x ← ShaderM.readBuffer (ty := .scalar .f32) (n := size) "input" idx
    let scaled := Exp.mul x (Exp.litF32 (1.0 / softcap))
    let capped := Exp.mul (Exp.tanh scaled) (Exp.litF32 softcap)
    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx capped
  ) (pure ())

/-- Copy-insert: `dst[params[0] * size + i] = src[i]` for `i in [0, size)`.
    Column-insert variant whose declared buffer size matches `totalDst`
    rather than `size * seqLen` — used for PLE (where `size = totalPL` and
    seqLen is separate). -/
def stubBatchInsertKernel (size totalDst : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let k := Exp.vec3X gid
  let _src ← ShaderM.declareInputBuffer "src" (.array (.scalar .f32) size)
  let _params ← ShaderM.declareInputBuffer "params" (.array (.scalar .u32) 1)
  let _dst ← ShaderM.declareOutputBuffer "dst" (.array (.scalar .f32) totalDst)
  ShaderM.if_ (Exp.lt k (Exp.litU32 size)) (do
    let colId ← ShaderM.readBuffer (ty := .scalar .u32) (n := 1) "params" (Exp.litU32 0)
    let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := size) "src" k
    let dstIdx := Exp.add (Exp.mul colId (Exp.litU32 size)) k
    ShaderM.writeBuffer (ty := .scalar .f32) "dst" dstIdx v
  ) (pure ())

/-- `batch[params[0] * dim + i] = src[i]` for `i in [0, dim)`. -/
def stubColumnInsertKernel (dim : Nat) (seqLen : Nat) : ShaderM Unit := do
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

/-- Multiply each of `size` f32 elements by `sqrt(hiddenSize)`.
    Matches `inp_scaled` in `llama.cpp/src/models/gemma4-iswa.cpp:13`
    (`ggml_scale(inpL, sqrtf(n_embd))`). -/
def stubEmbedScaleKernel (size : Nat) (hiddenSize : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid
  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) size)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) size)
  ShaderM.if_ (Exp.lt idx (Exp.litU32 size)) (do
    let x ← ShaderM.readBuffer (ty := .scalar .f32) (n := size) "input" idx
    let result := Exp.mul x (Exp.litF32 (Float.sqrt hiddenSize.toFloat))
    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx result
  ) (pure ())

/-- Element-wise GEGLU: `out[i] = gelu(gate[i]) * up[i]` over `size` f32 elements.
    Matches llama.cpp's `build_ffn(LLM_FFN_GELU, LLM_FFN_PAR)` after the
    `gate` and `up` matmuls. -/
def stubGegluKernel (size : Nat) : ShaderM Unit := do
  let gid ← ShaderM.globalId
  let idx := Exp.vec3X gid
  let _gate ← ShaderM.declareInputBuffer "gate" (.array (.scalar .f32) size)
  let _up ← ShaderM.declareInputBuffer "up" (.array (.scalar .f32) size)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) size)
  ShaderM.if_ (Exp.lt idx (Exp.litU32 size)) (do
    let g ← ShaderM.readBuffer (ty := .scalar .f32) (n := size) "gate" idx
    let u ← ShaderM.readBuffer (ty := .scalar .f32) (n := size) "up" idx
    -- GELU approx: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let c0 := Exp.litF32 0.7978845608028654   -- sqrt(2/pi)
    let c1 := Exp.litF32 0.044715
    let g3 := Exp.mul (Exp.mul g g) g
    let inner := Exp.mul c0 (Exp.add g (Exp.mul c1 g3))
    let gelu := Exp.mul (Exp.mul g (Exp.litF32 0.5)) (Exp.add (Exp.litF32 1.0) (Exp.tanh inner))
    ShaderM.writeBuffer (ty := .scalar .f32) "output" idx (Exp.mul gelu u)
  ) (pure ())

/-- Batched KV-cache write: scatter `[kvDim, seqLen]` input into the cache
    tensor `[numKVHeads, maxSeqLen, headDim]` starting at `params[0]`.
    Grid: `(seqLen, 1, 1)`; each workgroup handles one token.  Each thread
    handles one `(kvHead, d)` element of `kvDim = numKVHeads * headDim`. -/
def stubKVCacheWriteBatchKernel (numKVHeads maxSeqLen headDim seqLen : Nat) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let col := Exp.vec3X wid
  let tid := Exp.vec3X lid
  let kvDim := numKVHeads * headDim
  let wgSize := if kvDim < 256 then kvDim else 256
  let _newData ← ShaderM.declareInputBuffer "new_data" (.array (.scalar .f32) (kvDim * seqLen))
  let _cache ← ShaderM.declareOutputBuffer "cache" (.array (.scalar .f32) (numKVHeads * maxSeqLen * headDim))
  let _params ← ShaderM.declareInputBuffer "params" (.array (.scalar .u32) 1)
  ShaderM.loop tid (Exp.litU32 kvDim) (Exp.litU32 wgSize) fun i => do
    let kvHead := Exp.div i (Exp.litU32 headDim)
    let d := Exp.sub i (Exp.mul kvHead (Exp.litU32 headDim))
    let startPos ← ShaderM.readBuffer (ty := .scalar .u32) (n := 1) "params" (Exp.litU32 0)
    let pos := Exp.add startPos col
    let srcIdx := Exp.add (Exp.mul col (Exp.litU32 kvDim)) i
    let dstIdx := Exp.add (Exp.mul kvHead (Exp.litU32 (maxSeqLen * headDim)))
                          (Exp.add (Exp.mul pos (Exp.litU32 headDim)) d)
    let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := kvDim * seqLen) "new_data" srcIdx
    ShaderM.writeBuffer (ty := .scalar .f32) "cache" dstIdx v

/-- Per-head bare RMSNorm batched across seqLen.  No learned weight.
    Matches llama.cpp's `ggml_rms_norm(Vcur)` at gemma4-iswa.cpp:82.
    Grid: `(numKVHeads, seqLen, 1)`. -/
def stubPerHeadBareRMSNormBatchKernel (numKVHeads headDim seqLen : Nat) (eps : Float) : ShaderM Unit := do
  let wid ← ShaderM.workgroupId
  let lid ← ShaderM.localId
  let headIdx := Exp.vec3X wid
  let tokIdx  := Exp.vec3Y wid
  let tid := Exp.vec3X lid
  let kvDim := numKVHeads * headDim
  let totalElements := kvDim * seqLen
  let _input ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) totalElements)
  let _output ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) totalElements)
  let wgSize := if headDim < 256 then headDim else 256
  ShaderM.sharedNamed "shared_sum" (.array (.scalar .f32) wgSize)
  let colBase := Exp.mul tokIdx (Exp.litU32 kvDim)
  let headBase := Exp.add colBase (Exp.mul headIdx (Exp.litU32 headDim))
  ShaderM.varNamed "local_sum" (.scalar .f32) (Exp.litF32 0.0)
  let localSum : Exp (.scalar .f32) := Exp.var "local_sum"
  ShaderM.loop tid (Exp.litU32 headDim) (Exp.litU32 wgSize) fun i => do
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalElements) "input" (Exp.add headBase i)
    ShaderM.assign "local_sum" (Exp.add localSum (Exp.mul val val))
  ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_sum" tid localSum
  ShaderM.barrier
  let mut stride := wgSize / 2
  while stride > 0 do
    ShaderM.if_ (Exp.lt tid (Exp.litU32 stride)) (do
      let a ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := wgSize) "shared_sum" tid
      let b ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := wgSize) "shared_sum" (Exp.add tid (Exp.litU32 stride))
      ShaderM.writeWorkgroup (ty := .scalar .f32) "shared_sum" tid (Exp.add a b)
    ) (pure ())
    ShaderM.barrier
    stride := stride / 2
  let sumSq ← ShaderM.readWorkgroup (ty := .scalar .f32) (n := wgSize) "shared_sum" (Exp.litU32 0)
  let rms := Exp.inverseSqrt (Exp.add (Exp.div sumSq (Exp.litF32 headDim.toFloat)) (Exp.litF32 eps))
  ShaderM.loop tid (Exp.litU32 headDim) (Exp.litU32 wgSize) fun i => do
    let val ← ShaderM.readBuffer (ty := .scalar .f32) (n := totalElements) "input" (Exp.add headBase i)
    let normed := Exp.mul val rms
    ShaderM.writeBuffer (ty := .scalar .f32) "output" (Exp.add headBase i) normed

/-- Same DCE-safe stub body as decode path. -/
private def stubBody
    (inBufName outBufName : String) : ShaderM Unit := do
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

/-! ## Prefill-specific matmul pipeline (quantize → mmq → stream-K fixup)

Per layer: wQ, wK, wV, wO, ffn_gate, ffn_up, ffn_down → 7 mmq pipelines.
Each emits 3 physical kernels: `quantize_mmq_q8_1`, `mul_mat_q`, `mul_mat_q_stream_k_fixup`.
Plus PLE's inpGate and proj → 2 more pipelines = 9 pipelines × 3 = 27/layer.

But llama.cpp's 306/306/306 ÷ 42 layers = ~7 per layer → 7 matmul pipelines.
(PLE pathway may use different kernel.)
-/

def prefillQuantizeMmqQ8_1Kernel (_numLayers _M _K : Nat) : ShaderM Unit :=
  stubBody "mmq_in" "mmq_q8"

def prefillMulMatQQ4KKernel (_numLayers _M _N _K : Nat) : ShaderM Unit :=
  stubBody "mmq_q8" "mmq_part"

def prefillMulMatQStreamKFixupQ4KKernel (_numLayers _M _N : Nat) : ShaderM Unit :=
  stubBody "mmq_part" "mmq_out"

/-! ## Prefill attention: tensor-core GEMM-based flash_attn -/

def prefillFlashAttnTensorCoreKernel (_numLayers _qLen _kvLen _headDim : Nat) : ShaderM Unit :=
  stubBody "q_roped" "attn_out_tc"

/-! ## Fused gate×gelu(up) epilogue (prefill-specific) -/

def prefillGatedGeluKernel (_numLayers _M _interDim : Nat) : ShaderM Unit :=
  stubBody "gate_up" "gelu_out"

/-! ## cuBLAS-style batched-pointer setup -/

def prefillComputeBatchedPtrsKernel (_numBatches : Nat) : ShaderM Unit :=
  stubBody "bp_in" "bp_out"

/-! ## f32 scalar copy (temp buffer moves) -/

def prefillCpyScalarKernel (_numLayers _size : Nat) : ShaderM Unit :=
  stubBody "cpy_in" "cpy_out"

/-! ## f16↔f32 type conversions -/

def prefillConvertF32ToF16Kernel (_numLayers _size : Nat) : ShaderM Unit :=
  stubBody "cv_f32" "cv_f16"

def prefillConvertF16ToF32Kernel (_numLayers _size : Nat) : ShaderM Unit :=
  stubBody "cv_f16" "cv_f32"

/-! ## CUTLASS WMMA GEMM (used in some prefill matmul paths) -/

def prefillCutlassWmmaGemmKernel (_M _N _K : Nat) : ShaderM Unit :=
  stubBody "wmma_in" "wmma_out"

/-! ## Tensor-core lm_head (prefill, N>1) -/

def prefillLmHeadTensorCoreKernel (_M _vocabSize _hidden : Nat) : ShaderM Unit :=
  stubBody "lm_in" "lm_logits"

/-! ## Q6_K mmq path (lm_head prefill, N>1) -/

def prefillQuantizeMmqQ8_1Q6KKernel (_M _K : Nat) : ShaderM Unit :=
  stubBody "lm_mmq_in" "lm_mmq_q8"

def prefillMulMatQQ6KKernel (_M _vocabSize _hidden : Nat) : ShaderM Unit :=
  stubBody "lm_mmq_q8" "lm_mmq_part"

def prefillMulMatQStreamKFixupQ6KKernel (_M _vocabSize : Nat) : ShaderM Unit :=
  stubBody "lm_mmq_part" "lm_mmq_out"

end Hesper.Models.Gemma4.Llama.Prefill
