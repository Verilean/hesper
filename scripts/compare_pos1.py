#!/usr/bin/env python3
"""Compare Hesper pos=1 layer outputs against llama.cpp pos=1 layer outputs.

llama.cpp was invoked with `-p "Hello world"` so each `l_out-N.bin` has shape
`[hiddenSize, 2]` with pos=0 in the first half and pos=1 in the second half
(ggml is column-major for 2D tensors, so layout is [pos0_h0..pos0_hH-1, pos1_h0..pos1_hH-1]).

Hesper dumps pos=1 layer outputs to `forward_pos1_layer{N}_out.bin` (hiddenSize floats).
"""
import os, struct, math

LLAMA_DIR = "/tmp/llama_dump"
HESPER_DIR = "/tmp/hesper_dump"
HIDDEN = 2560

def load(p):
    with open(p, "rb") as f: d = f.read()
    return list(struct.unpack(f"<{len(d)//4}f", d))

def cosine(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    cos = dot/(na*nb) if na > 0 and nb > 0 else 0
    me = max(abs(x-y) for x, y in zip(a, b))
    return cos, me, na, nb

def main():
    print(f"{'layer':>5} {'cosine':>14} {'max_err':>12} {'norm_l':>10} {'norm_h':>10}  status")
    print("-" * 70)
    first_drop = None
    for i in range(64):
        lp = os.path.join(LLAMA_DIR, f"l_out-{i}.bin")
        hp = os.path.join(HESPER_DIR, f"forward_pos1_layer{i}_out.bin")
        if not os.path.exists(lp):
            break
        if not os.path.exists(hp):
            print(f"  {i:>3} HESPER MISSING")
            continue
        l_full = load(lp)
        if len(l_full) != 2 * HIDDEN:
            print(f"  {i:>3} llama size unexpected: {len(l_full)}")
            continue
        l = l_full[HIDDEN:]  # pos=1 = second half
        h = load(hp)
        if len(h) != HIDDEN:
            print(f"  {i:>3} hesper size unexpected: {len(h)}")
            continue
        cos, me, nl, nh = cosine(l, h)
        status = "✓" if cos > 0.999 else ("~" if cos > 0.99 else ("✗" if cos > 0.5 else "✗✗"))
        if cos < 0.99 and first_drop is None:
            first_drop = i
        print(f"  {i:>3} {cos:>14.6f} {me:>12.4e} {nl:>10.2f} {nh:>10.2f}  {status}")
    print()
    if first_drop is not None:
        print(f"🎯 Cosine drops below 0.99 first at layer {first_drop}")
    else:
        print("All layers at pos=1 match (cosine > 0.99)")

if __name__ == "__main__":
    main()
