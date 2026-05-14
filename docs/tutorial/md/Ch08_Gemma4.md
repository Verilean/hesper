# Chapter 08 — Gemma 4 End-to-End

Gemma 4 is Google's open-weights LLM family. Hesper supports the
**E4B (efficient 4 B-parameter)** instruction-tuned variant via the
CUDA backend. This chapter shows how to load a GGUF file, run greedy
or chat-template decoding, and read the performance counters.

## Prerequisites

- NVIDIA GPU with compute capability ≥ 8.0 (`sm_80`+).
- Driver supporting CUDA Toolkit 12.x.
- A Q4_K_M or Q6_K GGUF file of Gemma 4 E4B (e.g. from Hugging Face).

## Hello, Gemma

```bash
lake -Kgpu=cuda build gemma4-cuda
HESPER_CHAT=1 \
  ./.lake/build/bin/gemma4-cuda data/gemma-4-e4b-it-Q4_K_M.gguf "Hello" 30
```

Expected output (greedy decode, chat template enabled):

```
Hello! How can I help you today? 😊
```

Without `HESPER_CHAT=1`, the model sees the raw prompt and produces
its base-model continuation — useful for sanity-checking the kernel
path but not for chat-style use.

## What runs under the hood

`gemma4-cuda` does this on every token:

1. **Embed.** Look up the token in the embedding table.
2. **Per-layer PLE.** Gemma 4 uses Per-Layer Embeddings — a small extra
   table fetched on demand from CPU mmap to keep VRAM small.
3. **42 transformer blocks.** Each block:
   - Fused RMSNorm + Q8_1 quantize.
   - QKV projection (Q4_K matmul, MMQ5 tile shape for prefill, dp4a
     4-warp for decode).
   - RoPE-K + KV scatter fused.
   - Flash attention V11 (sub-warp partition, K-parallel, split-K).
   - Output projection (Q4_K, 4-warp coop-K).
   - Post-attention RMSNorm + residual fused.
   - Gate / up / GELU-quick (Q4_K matmul, ncols_dst=2).
   - FFN-down (Q6_K matmul, 4-warp 1-row).
4. **LM head.** Q6_K matmul pre-dequantized to F16 for the 256 K
   vocabulary.
5. **Argmax on device.** No DtoH bubble per token.

The whole sequence runs inside a single **CUDA Graph** (default ON), so
host overhead per token is one `cuGraphLaunch` call.

## Performance characteristics

On an RTX 4070 Ti:

| Workload | TPS | Notes |
|---|---|---|
| Decode (32-token prompt) | ~100 | CUDA Graphs ON, MMQ default for prefill |
| Prefill (seqLen 70) | ~17 ms | MMQ5 (llama.cpp-shape tile, mmq_y=128, mmq_x=64) |
| Cold start | +1.4 s | PTX JIT — cubin cache eliminates this on repeats |

The kernel times themselves are within ~3 % of llama.cpp's CUDA backend
(measured separately with `nsys`). The remaining wall-clock gap is host
overlap, not raw matmul throughput.

## Useful environment knobs

```bash
HESPER_CHAT=1                  # apply the IT chat template
HESPER_DP4A=1                  # force dp4a path for decode
HESPER_PREFILL_MMQ2_OFF=1      # disable MMQ for prefill (use dp4a)
HESPER_CUDA_GRAPHS=0           # disable CUDA Graph capture
HESPER_PIN_MMAP=1              # cuMemHostRegister the GGUF mmap region
HESPER_USE_MMAP=1              # mmap the GGUF instead of fread
```

The `HESPER_*` flags are documented in `Hesper/Models/Gemma4/Config.lean`.

## Reading the source

- `Hesper/Models/Gemma4/Gemma4.lean` — top-level forward and decode loop.
- `Hesper/Models/Gemma4/Linear.lean` — Q4_K / Q6_K matmul dispatchers.
- `Hesper/Layers/FlashAttention.lean` — the V11 vec kernel.
- `Hesper/Layers/RMSNorm.lean` — fused RMSNorm + Q8_1 quantize.
- `Hesper/IO/GGUF.lean` — GGUF loader; mmap + on-demand H2D for PLE.
- `Examples/Gemma4/Main.lean` — the `gemma4-cuda` driver.

## Parity test suite

26 parity tests verify each Gemma 4 component against llama.cpp's CPU
reference output, byte-for-byte:

```bash
lake -Kgpu=cuda build gemma4-qproj-parity
lake -Kgpu=cuda build gemma4-ffn-parity
lake -Kgpu=cuda build gemma4-kv-parity
# etc.
```

These tests use `scripts/llama_parity/` to dump ggml-CPU output to a
file and compare it byte-by-byte with the hesper GPU output. They are
how we caught the ShaderM `if_` branch CSE leak and several quantization
off-by-ones during bring-up.

## What's next

- [Chapter 09 — Embedding Hesper in Other Projects](Ch09_Embedding.md):
  add Hesper as a dependency in your own package.
- [Chapter 10 — Architecture](Ch10_Architecture.md): how the pieces in
  this chapter actually fit together, with diagrams.
