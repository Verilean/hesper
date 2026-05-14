# Gemma 4 — Kernels and their parity tests

This is the source-of-truth map between the CUDA kernels Gemma 4 dispatches
during decode/prefill and the Lean parity tests that guard them.  When you
touch a kernel, run the linked test (`lake exe <name>`) before committing —
each test exits non-zero on parity break.

The "calls/tok" column is from
[`docs/llama-fusion-analysis/bench/head-before.txt`](llama-fusion-analysis/bench/head-before.txt)
(graphs ON, 60-tok decode, "Hello world how are you").  Empty rows are
infrequent or one-shot.

## Hot-path kernels (decode — every token)

| Kernel function | File:line | Used for | calls / tok | hesper µs / call | parity test (`lake exe …`) |
|---|---|---|--:|--:|---|
| `fusedQ4KMLinearDP4A4WarpKernel` | [Linear.lean](../Hesper/Layers/Linear.lean) | Q4_K matmul (qkv, gate, up, wO) | 144 | 28.8 | [`cuda-dp4a-test`](../Tests/CUDA/CUDADP4ATest.lean) |
| `fusedQ4KMLinearDP4A2RowKernel` | [Linear.lean](../Hesper/Layers/Linear.lean) | Q4_K matmul (small outDim variant) | — | — | [`cuda-dp4a-test`](../Tests/CUDA/CUDADP4ATest.lean) |
| `fusedQ4KMLinearDP4AKernel` | [Linear.lean](../Hesper/Layers/Linear.lean) | Q4_K matmul (1-warp fallback) | — | — | [`cuda-dp4a-test`](../Tests/CUDA/CUDADP4ATest.lean) |
| `fusedQ6KLinearDP4AKernel` | [Linear.lean](../Hesper/Layers/Linear.lean) | Q6_K ffn_down (1-warp default) | 21 | 49.9 | [`cuda-q6k-dp4a-test`](../Tests/CUDA/CUDAQ6KDP4ATest.lean) |
| `fusedQ6KLinearDP4A4RowKernel` | [Linear.lean](../Hesper/Layers/Linear.lean) | Q6_K lm_head (4 rows / WG) | — | — | [`cuda-q6k-dp4a-test`](../Tests/CUDA/CUDAQ6KDP4ATest.lean) |
| `fusedQ6KLinearDP4A4WarpKernel` *(opt-in)* | [Linear.lean](../Hesper/Layers/Linear.lean) | Q6_K matmul (4-warp 1-row, llama.cpp shape) | — | — | [`cuda-q6k-4warp-parity`](../Tests/CUDA/CUDAQ6K4WarpParityTest.lean) (single dispatch) + [`cuda-q6k-4warp-graphs`](../Tests/CUDA/CUDAQ6K4WarpGraphsTest.lean) (14× capture+replay) |
| `matMulTransposeF16BlockCoopKernel` | [WGSL/MatMul.lean](../Hesper/WGSL/MatMul.lean) | f16 lm_head (when pre-dequantized) | 1 | 2741 (DRAM-bound) | [`cuda-q6k-to-f16-test`](../Tests/CUDA/Q6KToF16Test.lean) |
| `quantizeQ8_1Kernel` | [Layers/Linear.lean](../Hesper/Layers/Linear.lean) | f32 → Q8_1 input quantize | 80 | 1.0 | covered by `cuda-dp4a-test`, `cuda-q6k-dp4a-test` |
| `fusedRMSNormQ8_1Kernel` | [RMSNorm.lean](../Hesper/Layers/RMSNorm.lean) | Fused finalNorm + Q8_1 quantize for lm_head | 1 | — | end-to-end check (no isolated parity) |
| `fusedPerHeadQKVNormKernel` | [Layers/Linear.lean](../Hesper/Layers/Linear.lean) | Q+K per-head RMSNorm (decode, 1 token) | 24 | 1.4 | end-to-end check |
| `fusedPerHeadQKVNormBatchKernel` | [Layers/Linear.lean](../Hesper/Layers/Linear.lean) | Q+K per-head RMSNorm (prefill batch) | — | — | end-to-end check |
| `ropeWithFreqFactorsKernel` | [Layers/Linear.lean](../Hesper/Layers/Linear.lean) | RoPE Q (decode) | 54 | 1.9 | end-to-end check |
| `ropeWithFreqFactorsBatchKernel` | [Layers/Linear.lean](../Hesper/Layers/Linear.lean) | RoPE Q (prefill batch) | — | — | end-to-end check |
| `fusedRopeKAndCacheWriteKernel` | [Layers/Linear.lean](../Hesper/Layers/Linear.lean) | RoPE-K + KV cache scatter (decode) | 24 | 1.2 | [`cuda-rope-k-f16-test`](../Tests/CUDA/RopeKF16Test.lean) |
| `fusedRopeKAndCacheWriteBatchKernel` | [Layers/Linear.lean](../Hesper/Layers/Linear.lean) | RoPE-K + KV cache scatter (prefill) | — | — | [`cuda-rope-kv-batch-f16-parity`](../Tests/CUDA/RopeKVBatchF16Test.lean) |
| `flashAttentionDynamicParamsKernel` | [WGSL/FlashAttention.lean](../Hesper/WGSL/FlashAttention.lean) | Decode attention (V8 variant) | 41 | 6.7 | [`cuda-fa-golden-test`](../Tests/CUDA/CUDAFlashAttnGoldenTest.lean), [`cuda-fa-v11-parity`](../Tests/CUDA/V11LauncherParityTest.lean) |
| `flashAttentionBatchKernel` | [WGSL/FlashAttention.lean](../Hesper/WGSL/FlashAttention.lean) | Prefill attention (multi-token) | — | — | [`cuda-fa-batch-f16-parity`](../Tests/CUDA/BatchAttnF16Test.lean) |
| `argmaxKernel` | [Models/Gemma4.lean](../Hesper/Models/Gemma4.lean) | Greedy token selection | 167 | 6.5 | end-to-end check (deterministic) |

## Pointwise / fusion / utility kernels (smaller per-tok contribution)

| Kernel | File | Purpose | Test |
|---|---|---|---|
| `geluMulKernel` | [Layers/Linear.lean](../Hesper/Layers/Linear.lean) | FFN GELU(gate) × up | end-to-end |
| `geluGateMulSliceKernel` | [Layers/Linear.lean](../Hesper/Layers/Linear.lean) | Sliced PLE GELU/mul | end-to-end |
| `embeddingScaleKernel` | [Models/Gemma4.lean](../Hesper/Models/Gemma4.lean) | embed × √hiddenSize | end-to-end |
| `kEmbeddingLookupKernel` | [Models/Gemma4.lean](../Hesper/Models/Gemma4.lean) | Token-id → embed row | [`cuda-fa-test`](../Tests/CUDA/CUDAFlashAttnTest.lean) |
| `kTableRowDequantScaleKernel` | [Models/Gemma4.lean](../Hesper/Models/Gemma4.lean) | PLE table row dequant + scale | end-to-end |
| `residualAddKernel`, `scaledAddKernel`, `scaleKernel` | [Layers/Linear.lean](../Hesper/Layers/Linear.lean) | Residual / scaling pointwise | end-to-end |
| `chunkedRMSNormKernel`, `chunkedRMSNormAddScaledKernel` | [Layers/RMSNorm.lean](../Hesper/Layers/RMSNorm.lean) | Standalone RMSNorm (1024-thread) | end-to-end |
| `perHeadRMSNormKernel`, `perHeadRMSNormBatchKernel` | [Layers/Linear.lean](../Hesper/Layers/Linear.lean) | Q/K per-head norm (when not fused) | end-to-end |
| `fusedPerLayerPostKernel`, `fusedPerLayerPostThenScaleKernel` | [Models/Gemma4.lean](../Hesper/Models/Gemma4.lean) | PLE post-layer scale + add | end-to-end |
| `logitSoftcapKernel` | [Models/Gemma4.lean](../Hesper/Models/Gemma4.lean) | Final logit softcap | end-to-end |
| `advancePosKernel`, `historyAppendKernel` | [Models/Gemma4.lean](../Hesper/Models/Gemma4.lean) | Decode loop housekeeping (graphs ON) | end-to-end |
| `splitKReduceKernel` | [Layers/Linear.lean](../Hesper/Layers/Linear.lean) | Split-K matmul partial-sum reduce | end-to-end |
| `kMatmulBatchKernel`, `geluGateMulSliceBatchKernel` | [Layers/Linear.lean](../Hesper/Layers/Linear.lean) | Prefill batch path | end-to-end |

## "End-to-end check" means

Run:
```bash
HESPER_USE_MMAP=1 HESPER_DP4A=1 HESPER_CHAT=1 \
  .lake/build/bin/gemma4-cuda data/gemma-4-e4b-it-Q4_K_M.gguf "Hello" 11
```
Expected output: `Hello! How can I help you today? 😊`.  Any deviation
indicates a regression somewhere on the path (this catches bugs the
isolated parity tests miss — see `feedback_microbench_doesnt_imply_production.md`).

## Process — when you change a kernel

1. **Edit the kernel** in `Hesper/Layers/*.lean` or `Hesper/WGSL/*.lean`.
2. **Run its parity test** (column "parity test" above):
   ```bash
   lake exe cuda-q6k-4warp-parity   # or whichever
   ```
3. **If no parity test exists** (column says "end-to-end check"), write
   one before merging — the pattern is in
   [`Tests/CUDA/CUDAQ6K4WarpParityTest.lean`](../Tests/CUDA/CUDAQ6K4WarpParityTest.lean):
   feed both the new and the baseline kernel the same random input,
   exit non-zero if max rel diff exceeds tolerance.  Register the exe
   in `lakefile.lean` as `lean_exe «cuda-<name>»`.
4. **End-to-end check** at the end:
   ```bash
   HESPER_USE_MMAP=1 HESPER_DP4A=1 HESPER_CHAT=1 \
     .lake/build/bin/gemma4-cuda data/gemma-4-e4b-it-Q4_K_M.gguf "Hello" 11
   ```
5. Then run the canonical bench loop
   ([`scripts/kernel_bench.sh`](../scripts/kernel_bench.sh) `before|after|diff`)
   per [`feedback_kernel_improvement_process.md`](../../.claude/projects/-home-junji-hashimoto-git-hesper-gemma4/memory/feedback_kernel_improvement_process.md).

## Coverage gaps (TODO)

The following hot-path kernels currently rely on **end-to-end check only**
— a dedicated parity test would make regressions easier to bisect.
Highest priority based on call frequency × per-call time:

- `fusedRMSNormQ8_1Kernel` (lm_head fused norm+quant — 1 call but on the
  critical path; subtle bugs are masked by lm_head matmul)
- `fusedPerHeadQKVNormKernel` (24 calls/tok, no isolated test)
- `ropeWithFreqFactorsKernel` decode variant (54 calls/tok, batch
  variant has a parity test but decode does not)
- `geluMulKernel` (FFN gating — silent off-by-one would only show in
  output quality, not crash)
- `argmaxKernel` (correctness affects every output token)

If you touch any of these, write the parity test as part of the change
rather than relying on the e2e check.
