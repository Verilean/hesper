#!/usr/bin/env python3
"""Compare Hesper layer outputs vs llama.cpp layer outputs.

For each layer N, compare /tmp/hesper_dump/forward_layer{N}_out.bin
against /tmp/llama_dump/l_out-{N}.bin.

Shows where divergence first becomes significant.
"""
import os
import struct
import math

LLAMA_DIR = "/tmp/llama_dump"
HESPER_DIR = "/tmp/hesper_dump"

def load(p):
    with open(p, "rb") as f:
        d = f.read()
    return list(struct.unpack(f"<{len(d)//4}f", d))

def cosine(a, b):
    n = len(a)
    mse = sum((x - y) ** 2 for x, y in zip(a, b)) / n
    max_err = max(abs(x - y) for x, y in zip(a, b))
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    cos = dot / (na * nb) if na > 0 and nb > 0 else 0
    return cos, max_err, mse, na, nb

def main():
    print(f"{'layer':>5} {'size':>6} {'cosine':>14} {'max_err':>12} {'norm_l':>10} {'norm_h':>10}  status")
    print("-" * 80)

    first_drop = None
    for i in range(64):  # max possible layers
        llama_path = os.path.join(LLAMA_DIR, f"l_out-{i}.bin")
        hesper_path = os.path.join(HESPER_DIR, f"forward_layer{i}_out.bin")
        if not os.path.exists(llama_path):
            break
        if not os.path.exists(hesper_path):
            print(f"  {i:>3} HESPER MISSING")
            continue
        l = load(llama_path)
        h = load(hesper_path)
        if len(l) != len(h):
            print(f"  {i:>3} SIZE MISMATCH ({len(l)} vs {len(h)})")
            continue
        cos, me, mse, nl, nh = cosine(l, h)
        status = "✓" if cos > 0.999 else ("~" if cos > 0.99 else ("✗" if cos > 0.5 else "✗✗"))
        if cos < 0.99 and first_drop is None:
            first_drop = i
        print(f"  {i:>3} {len(l):>6} {cos:>14.6f} {me:>12.4e} {nl:>10.2f} {nh:>10.2f}  {status}")

    print()
    if first_drop is not None:
        print(f"🎯 Cosine drops below 0.99 first at layer {first_drop}")

if __name__ == "__main__":
    main()
