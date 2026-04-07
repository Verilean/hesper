#!/usr/bin/env python3
"""Inspect specific divergences in a layer 0 step."""
import sys
import struct

def load(path):
    with open(path, "rb") as f:
        data = f.read()
    n = len(data) // 4
    return list(struct.unpack(f"<{n}f", data))

step = sys.argv[1] if len(sys.argv) > 1 else "step_24_attn_residual"
llama_name = sys.argv[2] if len(sys.argv) > 2 else "attn_out-0"

llama = load(f"/tmp/llama_dump/{llama_name}.bin")
hesper = load(f"/tmp/hesper_dump/{step}.bin")

print(f"Comparing {step} ({len(hesper)}) vs {llama_name} ({len(llama)})")

# Compute per-element absolute and relative errors
errors = [(i, h, l, h - l, abs(h - l) / max(abs(l), 1e-6)) for i, (h, l) in enumerate(zip(hesper, llama))]
# Sort by absolute error (descending)
errors.sort(key=lambda x: -abs(x[3]))

print("\nTop 20 elements by absolute error:")
print(f"  {'idx':>5} {'hesper':>14} {'llama':>14} {'abs_err':>14} {'rel_err':>10}")
for i, h, l, ae, re in errors[:20]:
    print(f"  {i:>5} {h:>14.6f} {l:>14.6f} {ae:>14.6f} {re:>10.4f}")

# Histogram of errors
print("\nError distribution:")
buckets = [0, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0]
counts = [0] * (len(buckets) + 1)
for _, _, _, ae, _ in errors:
    placed = False
    for i, b in enumerate(buckets):
        if abs(ae) < b:
            counts[i] += 1
            placed = True
            break
    if not placed:
        counts[-1] += 1
labels = ["<1e-6", "<1e-4", "<1e-2", "<0.1", "<1", "<10", "<100", ">=100"]
total = len(errors)
for label, cnt in zip(labels, counts):
    pct = 100.0 * cnt / total if total > 0 else 0
    bar = "#" * int(pct / 2)
    print(f"  {label:>8}: {cnt:>6} ({pct:5.1f}%) {bar}")
