# Chapter 07 — BitNet b1.58 End-to-End

This chapter walks through Hesper's first complete inference engine:
BitNet b1.58 2B. Highlights:

- **125 TPS on M4 Max** via WebGPU/Metal.
- **40 TPS on RTX 4070 Ti** via WebGPU/Vulkan.
- LoRA-style instruction fine-tuning with a verified backward pass.

For deeper LoRA training details see
[`docs/LORA_FINETUNING.md`](../../LORA_FINETUNING.md).

## Run inference

```bash
lake exe bitnet-complete --stats
```

Expected output:

```text
> Hello, world!
Hello, world! I'm a 20-year-old college student...

Performance: 125.6 TPS (8.0 ms/token)
  Model: BitNet b1.58 2B (30 layers, 2560 dim, i2_s ternary weights)
```

## What's interesting about BitNet

BitNet b1.58 quantises every weight to `{-1, 0, +1}` (1.58 bits per
weight). The forward pass becomes additions only — no fp multiplies in
the matmul path. The challenge for a GPU implementation is squeezing
useful throughput out of an op that's normally memory-bound by the
weight reads.

Hesper's recipe:

| Optimisation | What it does | Source |
|---|---|---|
| **i2_s ternary kernel** | Pack 4 weights into 8 bits; matmul = popcount-style add | `Hesper/Models/BitNet.lean` |
| **Flash attention** | Fused score + online softmax + apply in one kernel | `Hesper/Layers/FlashAttention.lean` |
| **Fused gate+up+ReLU²×mul** | One FFN dispatch instead of four | `Hesper/Layers/*` |
| **Fused KV cache write** | Score + scatter into one kernel | `Hesper/Layers/Attention.lean` |
| **F16 LM-head matmul** | Shared-memory tile across the 128 K vocab | `Hesper/Models/BitNet.lean` |
| **PreparedDispatch capture** | 99 % pipeline-cache hit rate | `Hesper/Compute.lean` |
| **Single GPU submit/token** | Command-buffer batching | `Examples/BitNetComplete.lean` |
| **KV cache + GQA** | 20 heads / 5 KV heads | `Hesper/Models/BitNet.lean` |

## Sketch of the inference loop

The driver in `Examples/BitNetComplete.lean` wires this up
end-to-end. Schematically:

```text
let dev   ← Hesper.Device.create
let model ← BitNet.load dev "data/bitnet-1.58-2b.bin"
let tok   ← BitNet.Tokenizer.load "data/bitnet-tokenizer.json"

let prompt := "Hello, world!"
let mut state := BitNet.State.init model
let mut tokens := tok.encode prompt

for _ in [0:64] do
  let logits ← BitNet.forward model state tokens.back!
  let next   := BitNet.argmax logits
  tokens := tokens.push next
  state := BitNet.advance state
  IO.print (tok.decode #[next])

IO.println ""
```

(The exact function names live in `Hesper/Models/BitNet.lean` and the
`Examples/BitNetComplete.lean` driver — they're plumbing-heavy and not
worth reproducing verbatim in a tutorial.)

`BitNet.forward` is where every fused kernel actually runs. Internally
it does, per token:

1. Embed the token.
2. For each of the 30 transformer layers:
   - RMSNorm (fused with quant-pack of the input).
   - QKV projection (i2_s matmul, 4-warp coop K).
   - RoPE-Q in place, RoPE-K + KV scatter fused.
   - Flash attention (vec-kernel, K-parallel, sub-warp partition).
   - Output projection (i2_s).
   - Residual + post-attention RMSNorm fused.
   - Gate / up / ReLU² × mul / down (one fused kernel).
3. LM head: F16 shared-memory matmul into the 128 K vocabulary.
4. Argmax → next token.

## LoRA fine-tuning

```bash
lake exe lora-train data/alpaca.jsonl
```

Trains a low-rank adapter on Alpaca-style data, using the verified-AD
layer from Ch03. The training loop uses the same kernels as inference
plus a single backward pass per fused op (see Ch06). LoRA weights save
out as a small adapter file you can swap into the inference binary at
load time.

## Reading the source

Start here:

- `Hesper/Models/BitNet.lean` — the top-level transformer and
  per-layer kernels.
- `Hesper/Layers/FlashAttention.lean` — flash attention shared with
  Gemma 4.
- `Examples/BitNetComplete.lean` — the inference driver wired up to
  `bitnet-complete`.
- `Examples/MachineLearning/` — LoRA training drivers.

## What's next

- [Chapter 08 — Gemma 4 End-to-End](Ch08_Gemma4.md): a larger transformer
  on the CUDA backend with quantised weights (Q4_K_M / Q6_K).
