#!/usr/bin/env python3
"""Compare hesper's dumps vs llama.cpp's golden dumps for the prompt
"Hello world how are you" (seqLen=5). Prints rel diff for each layer
of a named tensor, highlighting where hesper first diverges."""
import os, sys
import numpy as np

HESPER = "/tmp/hesper_dump"
LLAMA = "/tmp/llama_dump"

def load(path):
    return np.fromfile(path, dtype=np.float32)

def reldiff(a, b):
    d = a - b
    denom = np.linalg.norm(b)
    return np.linalg.norm(d) / denom if denom > 1e-30 else np.linalg.norm(d)

def last_token(arr, dim):
    return arr[-dim:]

def compare(name_template, dim_per_token, num_layers, layers=None, seqLen=5):
    """Compare per-layer tensor across layers. dim_per_token: floats per token."""
    total = dim_per_token * seqLen
    if layers is None:
        layers = list(range(num_layers))
    print(f"\n=== {name_template} (last-token, dim={dim_per_token}) ===")
    max_rel = 0.0
    worst_layer = -1
    for li in layers:
        hp = f"{HESPER}/{name_template.format(li=li)}.bin"
        lp = f"{LLAMA}/{name_template.format(li=li)}.bin"
        if not os.path.exists(hp):
            print(f"  L{li:2d}: hesper missing {hp}"); continue
        if not os.path.exists(lp):
            continue
        h = load(hp); l = load(lp)
        if h.size != total or l.size != total:
            print(f"  L{li:2d}: size mismatch hesper={h.size} llama={l.size}"); continue
        r_full = reldiff(h, l)
        r_last = reldiff(last_token(h, dim_per_token), last_token(l, dim_per_token))
        tag = " ***" if r_last > 0.05 else ""
        print(f"  L{li:2d}: full={r_full:.4e}  last-tok={r_last:.4e}{tag}")
        if r_last > max_rel:
            max_rel = r_last; worst_layer = li
    print(f"  worst: L{worst_layer} rel={max_rel:.4e}")

N_LAYERS = 42
HIDDEN = 2560

# Walk through the block pipeline in order
compare("inp_scaled", HIDDEN, 1, layers=[0])  # layer-independent, dumped once before first block
compare("attn_norm-{li}", HIDDEN, N_LAYERS, layers=[0,1,2,3,5,10,17,24,41])
compare("Qcur-{li}", 8*256, N_LAYERS, layers=[0,1,2,3])  # SWA qDim=2048
compare("Qcur_pos-{li}", 8*256, N_LAYERS, layers=[0,1,2,3])
compare("__fattn__-{li}", 8*256, N_LAYERS, layers=[0,1,2,3])
compare("attn_post_norm-{li}", HIDDEN, N_LAYERS, layers=[0,1,2,3,5,10,17])
compare("attn_out-{li}", HIDDEN, N_LAYERS, layers=[0,1,2,3,5,10,17])
compare("ffn_norm-{li}", HIDDEN, N_LAYERS, layers=[0,1,2,3,5])
compare("ffn_out-{li}", HIDDEN, N_LAYERS, layers=[0,1,2,3,5])
compare("ffn_post_norm-{li}", HIDDEN, N_LAYERS, layers=[0,1,2,3,5,10,17])
compare("pe_in-{li}", HIDDEN, N_LAYERS, layers=[0,1,2,3,5,10,17])
compare("per_layer_embd_out-{li}", HIDDEN, N_LAYERS, layers=[0,1,2,3,5,10,17])
compare("pre_out_scaled-{li}", HIDDEN, N_LAYERS, layers=[0,1,2,3,5])
compare("out_scaled-{li}", HIDDEN, N_LAYERS, layers=[0,1,2,3,5])
compare("l_out-{li}", HIDDEN, N_LAYERS, layers=[0,1,2,3,4,5,10,17,24,40,41])
