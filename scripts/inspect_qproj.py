#!/usr/bin/env python3
"""Inspect Q projection differences in detail."""
import struct

def load(path):
    with open(path, "rb") as f:
        data = f.read()
    n = len(data) // 4
    return list(struct.unpack(f"<{n}f", data))

llama = load("/tmp/llama_dump/Qcur-0.bin")
hesper = load("/tmp/hesper_dump/step_11_q_proj.bin")

print(f"sizes: llama={len(llama)}, hesper={len(hesper)}")

# Print every 256th element
print("\nValues at every 256-element boundary:")
print(f"  {'idx':>5} {'llama':>15} {'hesper':>15} {'ratio (h/l)':>15}")
for i in range(0, len(llama), 256):
    l = llama[i]
    h = hesper[i]
    ratio = h / l if abs(l) > 1e-6 else 0
    print(f"  {i:>5} {l:>15.4f} {h:>15.4f} {ratio:>15.4f}")

# Find where ratio first drops
print("\nFirst 16 elements:")
for i in range(16):
    l = llama[i]
    h = hesper[i]
    diff = h - l
    print(f"  {i:>3} llama={l:>12.4f} hesper={h:>12.4f} diff={diff:>10.4f}")

print("\nElements 240-272 (block 0/1 boundary):")
for i in range(240, 272):
    l = llama[i]
    h = hesper[i]
    diff = h - l
    print(f"  {i:>3} llama={l:>12.4f} hesper={h:>12.4f} diff={diff:>10.4f}")

# Block-wise cosine
print("\nCosine per 256-element block:")
import math
for blk in range(0, len(llama) // 256):
    s = blk * 256
    e = s + 256
    l_seg = llama[s:e]
    h_seg = hesper[s:e]
    dot = sum(a * b for a, b in zip(l_seg, h_seg))
    norm_l = math.sqrt(sum(a * a for a in l_seg))
    norm_h = math.sqrt(sum(a * a for a in h_seg))
    cos = dot / (norm_l * norm_h) if norm_l > 0 and norm_h > 0 else 0
    print(f"  block {blk} (elems {s}-{e}): cosine={cos:.4f}")
