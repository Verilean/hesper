#!/usr/bin/env python3
"""Compare Hesper layer-0 intermediate dumps against llama.cpp ground truth.

Workflow:
  1. bash scripts/dump_llama_layer0.sh   # generates /tmp/llama_dump/
  2. lake exe gemma4-layer0-dump         # generates /tmp/hesper_dump/
  3. python3 scripts/compare_layer0.py   # compares the two

Mapping (Hesper step name → llama.cpp tensor name):
  step_02_embed_scaled  → inp_scaled
  step_10_attn_norm     → attn_norm-0
  step_11_q_proj        → Qcur-0
  step_12_q_norm        → Qcur_normed-0
  step_13_q_rope        → Qcur_pos-0
  step_14_k_proj        → Kcur-0
  step_15_k_norm        → Kcur_normed-0
  step_16_k_rope        → Kcur_pos-0
  step_17_v_proj        → Vcur-0
  step_18_v_norm        → Vcur_normed-0
  step_22_o_proj        → kqv_out-0
  step_23_post_attn_norm→ attn_post_norm-0
  step_24_attn_residual → attn_out-0
  step_30_ffn_norm      → ffn_norm-0
  step_31_ffn_gate      → ffn_gate-0
  step_32_ffn_up        → ffn_up-0
  step_33_ffn_gelu      → ffn_geglu-0
  step_34_ffn_down      → ffn_out-0
"""

import os
import struct

LLAMA_DIR = "/tmp/llama_dump"
HESPER_DIR = "/tmp/hesper_dump"

# Hesper step name → (llama.cpp tensor name, expected size)
MAPPING = [
    ("step_01_embed", "embd", 2560),
    ("step_02_embed_scaled", "inp_scaled", 2560),
    ("step_10_attn_norm", "attn_norm-0", 2560),
    ("step_11_q_proj", "Qcur-0", 2048),
    ("step_12_q_norm", "Qcur_normed-0", 2048),
    ("step_13_q_rope", "Qcur_pos-0", 2048),
    ("step_14_k_proj", "Kcur-0", 512),
    ("step_15_k_norm", "Kcur_normed-0", 512),
    ("step_16_k_rope", "Kcur_pos-0", 512),
    ("step_17_v_proj", "Vcur-0", 512),
    ("step_18_v_norm", "Vcur_normed-0", 512),
    ("step_22_o_proj", "kqv_out-0", 2560),
    ("step_23_post_attn_norm", "attn_post_norm-0", 2560),
    ("step_24_attn_residual", "attn_out-0", 2560),
    ("step_30_ffn_norm", "ffn_norm-0", 2560),
    ("step_31_ffn_gate", "ffn_gate-0", 10240),
    ("step_32_ffn_up", "ffn_up-0", 10240),
    ("step_33_ffn_gelu", "ffn_geglu-0", 10240),
    ("step_34_ffn_down", "ffn_out-0", 2560),
]


def load_f32(path):
    with open(path, "rb") as f:
        data = f.read()
    n = len(data) // 4
    return list(struct.unpack(f"<{n}f", data))


def cosine_stats(a, b):
    n = len(a)
    mse = sum((x - y) ** 2 for x, y in zip(a, b)) / n
    max_err = max(abs(x - y) for x, y in zip(a, b))
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        cos = 0.0 if norm_a != norm_b else 1.0
    else:
        cos = dot / (norm_a * norm_b)
    return cos, max_err, mse


def main():
    print(f"Hesper dumps: {HESPER_DIR}")
    print(f"llama.cpp dumps: {LLAMA_DIR}")
    print()

    if not os.path.isdir(LLAMA_DIR):
        print(f"❌ {LLAMA_DIR} not found. Run: bash scripts/dump_llama_layer0.sh")
        return
    if not os.path.isdir(HESPER_DIR):
        print(f"❌ {HESPER_DIR} not found. Run: lake exe gemma4-layer0-dump")
        return

    print(f"{'hesper step':<28} {'llama tensor':<22} {'size':>8} {'cosine':>12} {'max_err':>12}  status")
    print("-" * 100)

    first_divergence = None
    for hesper_name, llama_name, expected_size in MAPPING:
        hesper_path = os.path.join(HESPER_DIR, f"{hesper_name}.bin")
        llama_path = os.path.join(LLAMA_DIR, f"{llama_name}.bin")

        if not os.path.exists(hesper_path):
            print(f"{hesper_name:<28} {llama_name:<22}  HESPER MISSING")
            continue
        if not os.path.exists(llama_path):
            print(f"{hesper_name:<28} {llama_name:<22}  LLAMA MISSING")
            continue

        hesper = load_f32(hesper_path)
        llama = load_f32(llama_path)

        if len(hesper) != len(llama):
            print(f"{hesper_name:<28} {llama_name:<22} hesper={len(hesper)}, llama={len(llama)}  SIZE MISMATCH")
            continue
        if len(llama) != expected_size:
            print(f"{hesper_name:<28} {llama_name:<22} expected={expected_size}, got={len(llama)}  WARN: unexpected size")

        cos, max_err, mse = cosine_stats(llama, hesper)
        status = "✓" if cos > 0.999 else ("~" if cos > 0.99 else "✗")
        if cos <= 0.99 and first_divergence is None:
            first_divergence = hesper_name

        print(f"{hesper_name:<28} {llama_name:<22} {len(hesper):>8} {cos:>12.6f} {max_err:>12.4e}  {status}")

        if cos <= 0.99:
            print(f"    llama[:4]:  {[round(v, 4) for v in llama[:4]]}")
            print(f"    hesper[:4]: {[round(v, 4) for v in hesper[:4]]}")

    print()
    if first_divergence:
        print(f"🎯 First divergence at: {first_divergence}")
    else:
        print("✅ All steps match!")


if __name__ == "__main__":
    main()
