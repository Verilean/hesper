#!/usr/bin/env python3
"""Unit test #1: RMSNorm (attn_norm) at L0.

Given:
- Input:  llama.cpp's `inp_scaled` (last token slice, 2560 floats)
- Weight: GGUF `blk.0.attn_norm.weight` (2560 floats)

Expected output: llama.cpp's `attn_norm-0` (last token, 2560 floats)

Formula (from llama.cpp build_norm, LLM_NORM_RMS):
    y = ggml_rms_norm(x, eps) * weight
    where ggml_rms_norm(x) = x / sqrt(mean(x²) + eps)

Gemma 4 uses eps = 1e-6 (from GGUF gemma4.attention.layer_norm_rms_epsilon).

Gemma 4 does NOT add +1 to norm weights (Gemma4Model.norm_shift returns 0).

This test has NO hesper dependency — it's pure numpy.  If it fails,
llama.cpp's expected formula is wrong and we need a different reference.
"""
import numpy as np, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../llama.cpp/gguf-py'))
from gguf.gguf_reader import GGUFReader

DIM = 2560
EPS = 1e-6

def load_weight(name):
    r = GGUFReader('data/gemma-4-e4b-it-Q4_K_M.gguf', 'r')
    for t in r.tensors:
        if t.name == name:
            return np.array(t.data, dtype=np.float32)
    raise KeyError(name)

def last_tok(path, dim=DIM):
    arr = np.fromfile(path, dtype=np.float32)
    return arr[-dim:]

def rel_diff(a, b):
    return np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12)

def main():
    x = last_tok('/tmp/llama_dump/inp_scaled.bin').astype(np.float64)
    w = load_weight('blk.0.attn_norm.weight').astype(np.float64)
    expected = last_tok('/tmp/llama_dump/attn_norm-0.bin').astype(np.float64)

    # Reference RMSNorm
    rms = np.sqrt((x**2).mean() + EPS)
    y = (x / rms) * w

    rel = rel_diff(y, expected)
    print(f'[L0 RMSNorm(attn_norm)] python f64 ref vs llama.cpp: rel={rel*100:.8f}%')
    print(f'  x ||={np.linalg.norm(x):.3f}, w ||={np.linalg.norm(w):.3f}, rms={rms:.6f}')
    print(f'  y ||={np.linalg.norm(y):.3f}, expected ||={np.linalg.norm(expected):.3f}')
    if rel < 1e-5:
        print('  PASS (numerical floor)')
    else:
        print('  FAIL — investigate formula or eps')
        # Show first 5 element diffs
        for i in range(5):
            print(f'  [{i}]: y={y[i]:.6f}, expected={expected[i]:.6f}, d={y[i]-expected[i]:+.6e}')

if __name__ == '__main__':
    main()
