#!/usr/bin/env python3
"""Compare Hesper layer 5 step dumps vs llama.cpp layer 5 dumps."""
import os, struct, math

PAIRS = [
    ("layer5_step_10_attn_norm.bin", "attn_norm-5.bin"),
    ("layer5_step_11_q_proj.bin",    "Qcur-5.bin"),
    ("layer5_step_12_q_norm.bin",    "Qcur_normed-5.bin"),
    ("layer5_step_13_q_rope.bin",    "Qcur_pos-5.bin"),
    ("layer5_step_14_k_proj.bin",    "Kcur-5.bin"),
    ("layer5_step_15_k_norm.bin",    "Kcur_normed-5.bin"),
    ("layer5_step_16_k_rope.bin",    "Kcur_pos-5.bin"),
    ("layer5_step_17_v_proj.bin",    "Vcur-5.bin"),
    ("layer5_step_18_fattn.bin",     "__fattn__-5.bin"),
    ("layer5_step_19_kqv_out.bin",   "kqv_out-5.bin"),
    ("layer5_step_20_attn_post_norm.bin", "attn_post_norm-5.bin"),
    ("layer5_step_21_attn_residual.bin",  "attn_out-5.bin"),
    ("layer5_step_22_ffn_norm.bin",  "ffn_norm-5.bin"),
    ("layer5_step_23_ffn_gate.bin",  "ffn_gate-5.bin"),
    ("layer5_step_24_ffn_up.bin",    "ffn_up-5.bin"),
    ("layer5_step_25_ffn_geglu.bin", "ffn_geglu-5.bin"),
    ("layer5_step_26_ffn_out.bin",   "ffn_out-5.bin"),
    ("layer5_step_27_ffn_post_norm.bin", "ffn_post_norm-5.bin"),
    ("layer5_full_out.bin",          "l_out-5.bin"),
]

def load(p):
    with open(p, "rb") as f: d = f.read()
    return list(struct.unpack(f"<{len(d)//4}f", d))

def cosine(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    cos = dot/(na*nb) if na>0 and nb>0 else 0
    me = max(abs(x-y) for x,y in zip(a,b))
    return cos, me, len(a), len(b)

print(f"{'step':40} {'h_n':>6} {'l_n':>6} {'cosine':>12} {'max_err':>12}  status")
print("-"*90)
for h, l in PAIRS:
    hp = f"/tmp/hesper_dump/{h}"
    lp = f"/tmp/llama_dump/{l}"
    if not os.path.exists(hp): print(f"{h:40} MISSING hesper"); continue
    if not os.path.exists(lp): print(f"{h:40} MISSING llama ({l})"); continue
    H, L = load(hp), load(lp)
    cos, me, hn, ln = cosine(H, L)
    s = "✓" if cos>0.999 else ("~" if cos>0.99 else "✗")
    print(f"{h:40} {hn:>6} {ln:>6} {cos:>12.6f} {me:>12.4e}  {s}")
