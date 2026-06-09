# 36 — Per-kernel correspondence: hesper ↔ llama.cpp

*Written 2026-04-23 after non-kernel tax closed.  Remaining TPS gap
(59 ms vs 8.6 ms/decode, ~7×) lives entirely inside GPU kernels.  This
table pairs each hesper kernel with its llama.cpp counterpart so
throughput work has a 1:1 target per row.*

## How to read this table

Every row is ONE physical kernel dispatch per layer (or per forward),
in the order they fire along the decode forward.  Columns:

- **llama.cpp op**: name as it appears in `trace_lc.txt` (from
  `llama-eval-callback` on the same Q4_K_M model).
- **llama.cpp kernel**: the CUDA kernel template inside
  `llama.cpp/ggml/src/ggml-cuda/*.cu` that actually runs.
- **hesper callsite**: `file.lean:line` of the `GPUBackend.execute`
  that emits this dispatch.
- **hesper kernel**: the ShaderM kernel name (PTX-level function).
- **Status**: whether hesper's kernel matches llama.cpp's fusion
  shape (`=`), needs improvement (`✗`), or is a stub (`stub`).

Latencies are per-call medians from nsys on the 5-prompt + 10-decode
run (commit `bf57c6b`, graphs mode).  Hesper total = 13 ms/decode of
GPU work; llama.cpp total ≈ 8.3 ms/decode.

## Prefill/embedding prelude (runs once per forward, seqLen times)

| # | llama.cpp op | llama.cpp kernel | hesper callsite | hesper kernel | status |
|---|---|---|---|---|---|
| 0 | `GET_ROWS` (tok_embd) | `get_rows_cuda` (Q6_K) | `LlamaForwardPrefill.lean:295` | `q6kTableRowDequantScaleKernel` | = |
| 1 | `SCALE` (inp_scaled) | `scale_f32` | `LlamaForwardPrefill.lean:312` | `stubEmbedScaleKernel` | stub |
| 2 | `GET_ROWS` per-layer embd | `get_rows_cuda` (Q6_K) | `LlamaForwardPrefill.lean:361` | `q6kTableRowDequantScaleKernel` | = |
| 3 | per_layer_model_proj matmul | `mul_mat_vec_q4_K_q8_1` | `LlamaForwardPrefill.lean:377` | `executeMatMulTransposeF16` | ✗ f16 matmul vs Q4_K matmul (llama.cpp uses Q4_K-fused path) |
| 4 | `SCALE` (inp_per_layer) | `scale_f32` | `LlamaForwardPrefill.lean:381` | `stubBatchScaleKernel` | stub |
| 5 | `RMS_NORM` (inp_per_layer) | `rms_norm_f32` | `LlamaForwardPrefill.lean:387` | `stubChunkedRMSNormBatchKernel` | stub |
| 6 | `ADD` (residual) | `add_f32` | `LlamaForwardPrefill.lean:396` | `stubScaledAddBatchKernel` | stub |

## Per-layer block (runs 42× per decode forward)

### Attention norm + QKV projection

| # | llama.cpp op | llama.cpp kernel | hesper callsite | hesper kernel | status |
|---|---|---|---|---|---|
| 7 | `RMS_NORM` (attn_norm) | `rms_norm_f32` | `LlamaForwardPrefill.lean:516` | `Hesper.Layers.RMSNorm.forward` (decode path: fused RMSNorm+quantize_q8_1) | = (fused) |
| 8 | `MUL_MAT` Qcur-N | `mul_mat_vec_q4_K_q8_1` | `LlamaForwardPrefill.lean:530` | `Linear.forwardBatchDP4A` → `fusedQ4KMLinearDP4A4WarpKernel` | ✗ per-call ~185 µs vs llama.cpp ~40 µs |
| 9 | `RMS_NORM` (Qcur per-head norm) | `rms_norm_f32` | `LlamaForwardPrefill.lean:545` | `perHeadRMSNormBatchKernel` | = |
| 10 | `ROPE` Qcur_pos-N | `rope_norm_f32` | `LlamaForwardPrefill.lean:576` or `:580` | `llamaRopeQBatchedKernel` | = (but check Gemma's neox layout) |
| 11 | `MUL_MAT` Kcur-N | `mul_mat_vec_q4_K_q8_1` | `LlamaForwardPrefill.lean:597` | `Linear.forwardBatchDP4A` | ✗ same as #8 |
| 12 | `MUL_MAT` Vcur-N | `mul_mat_vec_q4_K_q8_1` | `LlamaForwardPrefill.lean:601` | `Linear.forwardBatchDP4A` | ✗ |
| 13 | `RMS_NORM` Kcur (per-head) | `rms_norm_f32` | `LlamaForwardPrefill.lean:608` | `perHeadRMSNormBatchKernel` | = |
| 14 | `RMS_NORM` Vcur_normed-N | `rms_norm_f32` | `LlamaForwardPrefill.lean:617` | `perHeadRMSNormBatchKernel` | = |
| 15 | `ROPE` Kcur_pos-N | `rope_norm_f32` | `LlamaForwardPrefill.lean:626` | `ropeWithFreqFactorsBatchKernel` | = |

### KV cache write + attention

| # | llama.cpp op | llama.cpp kernel | hesper callsite | hesper kernel | status |
|---|---|---|---|---|---|
| 16 | `SET_ROWS` cache_k_l-N | `k_set_rows` | `LlamaForwardPrefill.lean:656` | `llamaSetRowsKBatchedKernel` | = |
| 17 | `SET_ROWS` cache_v_l-N | `k_set_rows` | `LlamaForwardPrefill.lean:663` | `llamaSetRowsVBatchedKernel` | = |
| 18 | `FLASH_ATTN_EXT` __fattn__-N | `fattn-vec-f16` (decode) | `LlamaForwardPrefill.lean:673` | `llamaFlashAttnBatchedKernel` | ✗ per-call ~76 µs vs llama.cpp ~15-25 µs; may be missing wmma/tile path |
| 19 | `MUL_MAT` attn_out (wO) | `mul_mat_vec_q4_K_q8_1` | `LlamaForwardPrefill.lean:682` | `Linear.forwardBatchDP4A` | ✗ same as #8 |
| 20 | `RMS_NORM` postAttnNorm | `rms_norm_f32` | `LlamaForwardPrefill.lean:694` | `Hesper.Layers.RMSNorm.forward` | = |
| 21 | `ADD` residual | `add_f32` | `LlamaForwardPrefill.lean:706` | `residualAddKernel` | = |

### FFN (GEGLU)

| # | llama.cpp op | llama.cpp kernel | hesper callsite | hesper kernel | status |
|---|---|---|---|---|---|
| 22 | `RMS_NORM` ffn_norm | `rms_norm_f32` | `LlamaForwardPrefill.lean:718` | `Hesper.Layers.RMSNorm.forward` | = |
| 23 | `MUL_MAT` ffn_gate-N | `mul_mat_vec_q4_K_q8_1` | `LlamaForwardPrefill.lean:733` | `Linear.forwardBatchDP4A` (fused 4-warp gate+up available) | ✗ |
| 24 | `MUL_MAT` ffn_up-N | `mul_mat_vec_q4_K_q8_1` | `LlamaForwardPrefill.lean:737` | `Linear.forwardBatchDP4A` | ✗ |
| 25 | `GLU` ffn_geglu-N | `glu_swiglu_f32` (Gemma uses GEGLU variant) | `LlamaForwardPrefill.lean:741` | `stubGegluKernel` | stub |
| 26 | `MUL_MAT` ffn_out-N (down) | `mul_mat_vec_q4_K_q8_1` | `LlamaForwardPrefill.lean:747` | `Linear.forwardBatchDP4A` | ✗ |
| 27 | `RMS_NORM` postFFNNorm | `rms_norm_f32` | `LlamaForwardPrefill.lean:760` | `Hesper.Layers.RMSNorm.forward` | = |
| 28 | `ADD` residual | `add_f32` | `LlamaForwardPrefill.lean:772` | `residualAddKernel` | = |

### PLE block (Gemma 4 specific, per-layer)

| # | llama.cpp op | llama.cpp kernel | hesper callsite | hesper kernel | status |
|---|---|---|---|---|---|
| 29 | `MUL_MAT` ple_inp_gate | `mul_mat_vec_q4_K_q8_1` | `LlamaForwardPrefill.lean:795` | `Linear.forwardBatchDP4A` | ✗ |
| 30 | `UNARY` (gelu) | `gelu_f32` | `LlamaForwardPrefill.lean:798` | `stubGeluKernel` | stub |
| 31 | `MUL` ple_gate_mul | `mul_f32` | `LlamaForwardPrefill.lean:808` | `stubMulKernel` | stub |
| 32 | `MUL_MAT` ple_proj | `mul_mat_vec_q4_K_q8_1` | `LlamaForwardPrefill.lean:813` | `Linear.forwardBatchDP4A` | ✗ |
| 33 | `RMS_NORM` ple_post_norm | `rms_norm_f32` | `LlamaForwardPrefill.lean:816` | `Hesper.Layers.RMSNorm.forward` | = |
| 34 | `ADD` residual | `add_f32` | `LlamaForwardPrefill.lean:822` | `residualAddKernel` | = |
| 35 | `MUL` l_out-N (scale) | `mul_f32` | `LlamaForwardPrefill.lean:860` | `stubBroadcastScaleKernel` | stub |
| 36 | `ADD` block-output residual | `add_f32` | `LlamaForwardPrefill.lean:870` | `residualAddKernel` | = |

## Post-loop (runs once per forward, last token only)

| # | llama.cpp op | llama.cpp kernel | hesper callsite | hesper kernel | status |
|---|---|---|---|---|---|
| 37 | column extract (inp_out_ids) | `cpy_f32_f32` | `LlamaForwardPrefill.lean:900` | `stubColumnExtractKernel` | stub |
| 38 | `RMS_NORM` output_norm | `rms_norm_f32` | `LlamaForwardPrefill.lean:907` | `Hesper.Layers.RMSNorm.forward` | = |
| 39 | `MUL_MAT` result_output (lm_head) | `mul_mat_vec_q6_K_q8_1` | `LlamaForwardPrefill.lean:914` | `fusedQ6KLinearKernel` | ✗ lm_head is the single biggest kernel by time (~0.8 ms) |
| 40 | `SCALE` (softcap / 2·tanh) | `scale_f32` + `tanh` fused | `LlamaForwardPrefill.lean:~920` | `stubLogitSoftcapKernel` | stub |

## Hot-list: kernels to attack first

Ordered by `total time / decode`, from the 10-token nsys sample
(kernel hashes resolved via `HESPER_KERNEL_TRACE=1` → label).

| Kernel group | hesper ms/decode | llama.cpp ms/decode | Ratio | Why |
|---|---:|---:|---:|---|
| **Q4_K matmuls** (wQ/wK/wV/wO × 42, plus gate/up/down × 42, plus PLE × 42) | ~37 ms | ~4.5 ms | **8×** | llama.cpp: 4-row+SRAM tile+async pipelined; hesper: 4-warp but no async pipeline + missing multi-row for PLE |
| **FlashAttention** (× 42) | ~3.2 ms | ~0.7 ms | **4.5×** | hesper uses vec-f16 equivalent but missing combine-KV path for decode N=1 |
| **Q6_K lm_head** (× 1) | ~0.8 ms | ~0.5 ms | **1.6×** | Close; probably good first-win for validating the optimization pipeline |
| **RMSNorm** (× 7 per layer × 42 + final 4) | ~1.3 ms | ~0.4 ms | **3.2×** | hesper already fuses quantize_q8_1; gap is inside the norm itself (reduction pattern) |
| **RoPE** (× 2 per own-KV layer) | ~0.4 ms | ~0.2 ms | **2×** | Small, not priority |
| **Everything else** (adds, scales, geglu, set_rows) | ~0.8 ms | ~0.3 ms | 2.7× | Pointwise; cheap to leave |

## Priority order for kernel work

1. **Q4_K matmul** — 37 ms vs 4.5 ms is where the bulk of the gap
   sits.  Three candidates:
   - Port `mul_mat_vec_q` template directly (see
     `llama.cpp/ggml/src/ggml-cuda/mmvq.cu` lines 380-500, the
     `has_fusion=true` branch that inline-quantizes input to Q8_1).
   - Use WMMA (tensor cores) for the decode-hot matmuls.
   - Add a true multi-row (4 or 8 rows) variant for PLE-sized outDim.
   Gain target: **37 → 6 ms (+31 ms/decode)**.

2. **FlashAttention decode path** — hesper's batched kernel isn't
   specialised for the N=1 decode case (single-query, many-keys).
   Port `fattn-vec-f16` or the `fattn-tile` template.
   Gain target: **3.2 → 0.7 ms (+2.5 ms/decode)**.

3. **RMSNorm reduction** — bandwidth-bound kernel; hesper is probably
   doing per-thread atomic or naive reduction.  Switch to warp-shuffle
   + single shared-memory pass.  Gain target: **1.3 → 0.4 ms**.

4. **lm_head Q6_K** — already within 1.6×; one round of cooperative
   tiling puts it at parity.

If 1-3 all land, per-decode drops to ~59 − 33.5 = **25.5 ms/decode →
~39 TPS**.  Remaining gap then has to come from more aggressive
fusion (e.g., fusing the seven RMSNorm dispatches per layer into
fewer ops) and/or tensor-core FLOP density.  Reaching 115 TPS still
requires collapsing the kernel launch count per layer, not just per-
kernel throughput.

## Re-measuring per-kernel latencies

```bash
# 1. Resolve hash → label mapping so kernel names are readable.
HESPER_DP4A=1 HESPER_LLAMA_GRAPHS=1 HESPER_KERNEL_TRACE=1 \
  lake exe gemma4-llama-prefill-skeleton \
    data/gemma-4-e4b-it-Q4_K_M.gguf "Hello world how are you" 5 \
    2>&1 | grep '^\[hs\]' | sort -u > /tmp/hesper_kernel_labels.txt

# 2. nsys (same recipe as doc 35 §B.1).
mkdir -p /tmp/nsys-graphs
HESPER_DP4A=1 HESPER_LLAMA_GRAPHS=1 \
  nsys profile -t cuda,nvtx --stats=false \
    -o /tmp/nsys-graphs/hesper_graphs -f true \
  lake exe gemma4-llama-prefill-skeleton \
    data/gemma-4-e4b-it-Q4_K_M.gguf "Hello world how are you" 10

# 3. Extract per-kernel totals.
nsys stats --report cuda_gpu_kern_sum --format csv \
  /tmp/nsys-graphs/hesper_graphs.nsys-rep > /tmp/hesper_kern.csv

# 4. Cross-reference with llama.cpp:
nsys profile -t cuda,nvtx --stats=false \
  -o /tmp/nsys-graphs/llama_cpp -f true \
  llama.cpp/build/bin/llama-cli -m data/gemma-4-e4b-it-Q4_K_M.gguf \
    -p "Hello world how are you" -n 10 --no-warmup -ngl 99 \
    --seed 0 --single-turn

nsys stats --report cuda_gpu_kern_sum --format csv \
  /tmp/nsys-graphs/llama_cpp.nsys-rep > /tmp/lc_kern.csv
```

Then diff the `Total Time` columns row by row.  Hesper kernel hashes
become stable once the code freezes, so you can reuse
`/tmp/hesper_kernel_labels.txt` for future measurements — just
regenerate on code changes.
