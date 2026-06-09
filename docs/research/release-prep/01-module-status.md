# Module status (release prep — REL-B)

Hesper has accumulated many modules across iterations on different model
families and DSL designs. This document declares each module's status as of
this release.

**Status legend**:
- ✅ **production** — required by the Gemma 4 inference path; covered by `scripts/regression.sh` (26-test suite).
- 🟡 **preview** — implemented + parity-tested but not on the default decode path. Examples: vision/audio kernels, WMMA Tensor Core path. Will graduate to production once wired in and validated end-to-end.
- 🧪 **experimental** — exploratory code from past iterations. Compiles, but not actively used. Keep for now; may be removed in a future release.
- 📚 **infra/lib** — internal infrastructure (logging, FFI, build wiring, test utilities). Always loaded but invisible to end users.

## Top-level modules

| module | status | notes |
|---|---|---|
| `Hesper/Backend.lean`, `Hesper/Backend/`         | ✅ production | GPUBackend typeclass + CUDA + WebGPU impls |
| `Hesper/Basic.lean`                              | 📚 infra | byte/float helpers used everywhere |
| `Hesper/CUDA/`                                    | ✅ production | PTX codegen, FFI, runtime |
| `Hesper/WGSL/`                                    | ✅ production | ShaderM DSL, kernel emission |
| `Hesper/Layers/`                                  | mixed — see per-file below |
| `Hesper/Models/Gemma4*`                           | ✅ production | the model that this release is *named* after |
| `Hesper/Models/BitNet*`                           | 🧪 experimental | older model port; not on the Gemma 4 path |
| `Hesper/GGUF*`                                    | ✅ production | model loader |
| `Hesper/Quantization/`                            | ✅ production | Q4_K, Q6_K, Q8_1 codecs |
| `Hesper/Circuit/`                                 | ✅ production | IR / lowering / passes used by Gemma 4 |
| `Hesper/CircuitV2/`                               | 🟡 preview | IRv2 + Block scope; landed Phase B-F2 layer-0 parity, not wired into decode hot path |
| `Hesper/Tokenizer/`, `Hesper/Tokenizers/`         | ✅ production | SentencePiece + GPT-2 |
| `Hesper/Inference/Sampling.lean`                  | ✅ production | argmax / temperature / top-k |
| `Hesper/Logging.lean`                             | 📚 infra | structured trace |
| `Hesper/Profile*`                                 | 📚 infra | nsys / per-section trace |
| `Hesper/Float16.lean`, `Hesper/Float32.lean`     | 📚 infra | FP type wrappers |
| `Hesper/Compute.lean`                             | 🧪 experimental | early shader-execution prototype, predates Backend typeclass |
| `Hesper/Async.lean`                               | 🧪 experimental | async wrapper, unused on Gemma 4 path |
| `Hesper/Simd.lean`                                | 🧪 experimental | CPU SIMD primitives, unused on GPU path |
| `Hesper/IO/`                                      | 📚 infra | file/stream helpers |
| `Hesper/Core/`                                    | 📚 infra | dim/shape definitions |
| `Hesper/NN/`, `Hesper/Op/`                        | 🧪 experimental | high-level NN ops, predates ShaderM |
| `Hesper/Tensor/`                                  | 🧪 experimental | early tensor abstraction; superseded by ShaderM buffers |
| `Hesper/Optimizer/`                               | 🧪 experimental | training-time optimizers |
| `Hesper/Training/`                                | 🧪 experimental | training loop scaffolding |
| `Hesper/AD/`                                      | 🧪 experimental | autodiff (used by `Examples/DSL/AD*`) |
| `Hesper/TTT/`                                     | 🧪 experimental | Test-Time Training (BitNet-side) |
| `Hesper/LoRA/`                                    | 🧪 experimental | LoRA adapter impl |
| `Hesper/Validation/`                              | 🧪 experimental | dim-check / type-check passes |
| `Hesper/Proofs/`                                  | 🧪 experimental | formal proof scaffolding |
| `Hesper/GLFW*`                                    | 🧪 experimental | windowing for visual demos |
| `Hesper/WebGPU/`                                  | ✅ production | WebGPU buffer ops (called from CUDA path's tokenIdsBuf write) |

## `Hesper/Layers/` per-file

| file | status | notes |
|---|---|---|
| `Linear.lean`        | ✅ production | Q4_K / Q6_K matmul kernels (MMQ2/5, fused gate+up, lm_head) |
| `Attention.lean`     | ✅ production | Q/K/V projection, KV cache, attn dispatch |
| `RMSNorm.lean`       | ✅ production | RMSNorm + fused-Q8_1-quantize |
| `RMSNorm_v2.lean`    | 🟡 preview | IRv2 form; not on Gemma 4 hot path |
| `RoPE.lean`          | ✅ production | RoPE-Q + RoPE-K + KV write |
| `Softmax.lean`       | ✅ production | for non-FlashAttn paths |
| `Embedding.lean`     | ✅ production | token + position embeddings |
| `MoE.lean`           | ✅ production | shared with Gemma 4 expert layout |
| `PerLayerEmbedding.lean` | ✅ production | Gemma 4 PLE on-demand fetch |
| `KVCache_v2.lean`    | 🟡 preview | IRv2-side, not on hot path |
| `TransformerBlock.lean` | 🧪 experimental | early block abstraction; superseded by Gemma4.forwardBlock |
| `BitLinear.lean`, `BitLinearSpec.lean` | 🧪 experimental | BitNet's 1.58-bit linear |
| `FusedFFNSpec.lean`  | 📚 infra | algebraic spec for fused FFN (used in unit tests / Circuit DSL passes) |
| `Vision.lean` (im2col, conv2d, geglu_quick, concat_dim0, permute_4d, matmulF32Naive) | 🟡 preview | byte-parity vs llama.cpp; not on Gemma 4 LLM path. Foundation for SigLIP encoder (~4-5 sessions away from end-to-end image input — see `docs/video-impl-plan/02-siglip-op-coverage.md`) |
| `Audio.lean` (conv_transpose_1d) | 🟡 preview | byte-parity vs llama.cpp; not on Gemma 4 LLM path. Foundation for audio decoders |

## `Examples/` per-target

Active: `gemma4-cuda` (Gemma 4 inference), `Examples/DSL/Gemma4*` parity tests.

Experimental: `BitNet*`, `Examples/SmartKV_Needle*` (TTT), `Examples/MachineLearning/*`, `Examples/DSL/AD*`, `Examples/Generic/*`.

## What's `lean_exe` vs library-only

`lakefile.lean` declares ~50 `lean_exe` targets. The release-relevant ones are:
- `gemma4-cuda` (the headline binary)
- `lake exe` parity tests under `Tests/CUDA/` (the 43-test regression set)
- `transpile-cuda-mmq-q4k-microbench`, etc. for performance work

Everything else is "compile-but-don't-ship" — they exist for development/research, not as user-visible deliverables.

## Recommendation

For this release:
- Ship the file as-is (no deletions).
- This document declares status. End users see only `gemma4-cuda` + `lib/libhesper_*` so the experimental modules don't surface to them.
- Future releases can prune `🧪 experimental` once we've verified no future plans need them.
