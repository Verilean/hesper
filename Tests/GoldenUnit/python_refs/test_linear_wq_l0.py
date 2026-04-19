#!/usr/bin/env python3
"""Python f64 reference for Q4_K matmul (wQ L0).

Purpose: determine whether hesper's forwardDP4A or llama.cpp's
mul_mat_vec_q is "more correct" w.r.t. a dequantized f64 ground
truth.  Both are f32 dp4a implementations, so both have reduction-
order noise.  But a 0.9% diff between them is unusually large —
larger than typical f32 dp4a noise (~1e-4).  This script tells us
which one is closer to the f64 ref.

Expected: both ~1e-4.  If one of them is ~1e-2 and the other
~1e-4, we found a bug in the 1e-2 side.
"""
import numpy as np, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../llama.cpp/gguf-py'))
from gguf.gguf_reader import GGUFReader
from gguf.quants import dequantize

DIM = 2560
QDIM = 2048  # 8 heads × 256 headDim (SWA) at L0

def main():
    # Load inputs
    attn_norm = np.fromfile('/tmp/llama_dump/attn_norm-0.bin', dtype=np.float32).astype(np.float64)
    if attn_norm.size > DIM:
        attn_norm = attn_norm[-DIM:]  # last token
    qcur_llama = np.fromfile('/tmp/llama_dump/Qcur-0.bin', dtype=np.float32).astype(np.float64)
    if qcur_llama.size > QDIM:
        qcur_llama = qcur_llama[-QDIM:]

    # Dequantize Q4_K wQ from GGUF
    r = GGUFReader('data/gemma-4-e4b-it-Q4_K_M.gguf', 'r')
    w = None
    for t in r.tensors:
        if t.name == 'blk.0.attn_q.weight':
            w = dequantize(t.data, t.tensor_type).astype(np.float64)
            break
    assert w is not None
    print(f'w shape: {w.shape}')
    # Expect (out=2048, in=2560) after dequantize reshape

    # f64 matmul reference
    if w.shape == (QDIM, DIM):
        y_ref = w @ attn_norm  # (2048,)
    elif w.shape == (DIM, QDIM):
        y_ref = attn_norm @ w
    else:
        raise RuntimeError(f'unexpected shape {w.shape}')

    # Compare llama's Qcur-0 last token to f64 ref
    rel_llama = np.linalg.norm(qcur_llama - y_ref) / np.linalg.norm(y_ref)
    print(f'\nllama.cpp Qcur-0 (last tok) vs f64 dequant matmul:')
    print(f'  rel = {rel_llama:.6e}')
    print(f'  ||llama||={np.linalg.norm(qcur_llama):.3f}')
    print(f'  ||f64 ref||={np.linalg.norm(y_ref):.3f}')

    # We can't run hesper's kernel from Python, but the LSpec test reported
    # rel vs llama as 9.4e-3.  If f64 ref vs llama is ~1e-4, then hesper vs
    # f64 ref must be ~9.4e-3 also — hesper is the outlier.
    print(f'\nIf llama is within 1e-4 of f64 and hesper is 9.4e-3 from llama,')
    print(f'then hesper is the buggy one (its reduction or quant handling is wrong).')

if __name__ == '__main__':
    main()
