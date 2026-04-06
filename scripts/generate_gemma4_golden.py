#!/usr/bin/env python3
"""Generate golden values for Gemma 4 validation using llama-cpp-python.

No numpy dependency — uses struct module only (NixOS compatible).

Outputs:
  Tests/golden-values-gemma4/input_tokens.bin   - Token IDs (u32 LE)
  Tests/golden-values-gemma4/logits_output.bin  - Logits (f32 LE, last token only)
  Tests/golden-values-gemma4/metadata.json      - Model info

Usage:
  python3 scripts/generate_gemma4_golden.py data/gemma-4-e4b-it-Q4_K_M.gguf
"""

import sys
import os
import json
import struct
import ctypes
import array

def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "data/gemma-4-e4b-it-Q4_K_M.gguf"
    output_dir = "Tests/golden-values-gemma4"
    os.makedirs(output_dir, exist_ok=True)

    from llama_cpp import Llama

    print(f"Loading model: {model_path}")
    llm = Llama(model_path=model_path, n_ctx=128, n_gpu_layers=0, verbose=True, logits_all=True)

    # Tokenize a short prompt for a meaningful test
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Hello"
    prompt_bytes = prompt.encode("utf-8")
    tokens = llm.tokenize(prompt_bytes, add_bos=True)
    print(f"Prompt: {prompt!r}")
    print(f"Input tokens: {tokens}")

    # Evaluate
    llm.reset()
    llm.eval(tokens)

    # Get logits for last token — llm._scores is a ctypes float array
    # scores shape is [n_tokens, n_vocab], we want last token
    n_vocab = llm.n_vocab()
    print(f"Vocab size: {n_vocab}")

    # Use eval_logits which returns a list of logits for each evaluated token
    # After eval(), llm.eval_logits[-1] has the logits for the last token
    last_logits = llm.eval_logits[-1]
    logits = list(last_logits)

    print(f"Logits length: {len(logits)}")
    print(f"Logits (first 10): {logits[:10]}")

    # Find argmax
    max_val = logits[0]
    max_idx = 0
    for i, v in enumerate(logits):
        if v > max_val:
            max_val = v
            max_idx = i
    print(f"Argmax: token {max_idx} = {max_val}")

    # Save input tokens
    tokens_path = os.path.join(output_dir, "input_tokens.bin")
    with open(tokens_path, "wb") as f:
        for t in tokens:
            f.write(struct.pack("<I", t))
    print(f"Saved input tokens to {tokens_path}")

    # Save logits as f32 LE
    logits_path = os.path.join(output_dir, "logits_output.bin")
    with open(logits_path, "wb") as f:
        for v in logits:
            f.write(struct.pack("<f", v))
    print(f"Saved logits to {logits_path} ({os.path.getsize(logits_path)} bytes)")

    # Save metadata
    metadata = {
        "model": model_path,
        "vocab_size": n_vocab,
        "seq_len": len(tokens),
        "format": "float32_little_endian",
        "source": "llama-cpp-python",
        "tokens": tokens,
        "argmax": max_idx,
        "argmax_value": max_val,
    }
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")
    print("Done!")


if __name__ == "__main__":
    main()
