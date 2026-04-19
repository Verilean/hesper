#!/usr/bin/env python3
"""Inspect Gemma 4 GGUF tokenizer vocab by token id.

Usage: python3 tmp_inspect_vocab.py [token_id ...]

When run with no args, dumps the top-5 tokens from llama.cpp's
/tmp/llama_dump/result_output.bin (final logits, "Hello world how are
you" prompt).  Previously used to confirm that token 236881 = '?', which
is both hesper's and llama.cpp's greedy pick for that prompt — so the
alleged "decode loop bug" was actually correct output.
"""
import sys
sys.path.insert(0, '/home/junji-hashimoto/git/hesper-gemma4/llama.cpp/gguf-py')
from gguf import GGUFReader
import numpy as np

r = GGUFReader('/home/junji-hashimoto/git/hesper-gemma4/data/gemma-4-e4b-it-Q4_K_M.gguf')
tf = [f for f in r.fields.values() if f.name == 'tokenizer.ggml.tokens'][0]

def decode(tid: int) -> str:
    return bytes(tf.parts[tf.data[tid]]).decode('utf-8', errors='replace')

if len(sys.argv) > 1:
    for arg in sys.argv[1:]:
        t = int(arg)
        print(f'token {t:6d}: {decode(t)!r}')
else:
    l = np.fromfile('/tmp/llama_dump/result_output.bin', dtype=np.float32)
    top = np.argsort(-l)[:5]
    print('Top-5 tokens in llama.cpp result_output.bin:')
    for t in top:
        print(f'  token {int(t):6d}: {decode(int(t))!r:>20s}  logit={l[int(t)]:.4f}')
