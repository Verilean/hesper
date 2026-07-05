#!/usr/bin/env python3
"""Per-layer parity scan: llama-eval-callback text dump vs our golden .bin dumps."""
import re, sys
import numpy as np

REF = '/tmp/llama_ref.txt'
GOLD = '/tmp/e2b_golden'
DIM = 1536
NTOK = 6

# Parse llama ref: node name -> list of rows, each row = (first3, last3)
ref = {}
lines = open(REF).read().splitlines()
i = 0
hdr = re.compile(r'common_debug_cb_eval:\s+(\S.*?) = \(f32\)')
rowre = re.compile(r'\[\s*([-\d.einf]+),\s+([-\d.einf]+),\s+([-\d.einf]+),\s+\.\.\.,\s+([-\d.einf]+),\s+([-\d.einf]+),\s+([-\d.einf]+)\s*\]')
while i < len(lines):
    m = hdr.search(lines[i])
    if m:
        name = m.group(1).strip()
        rows = []
        j = i + 1
        while j < len(lines) and 'sum =' not in lines[j] and not hdr.search(lines[j]):
            rm = rowre.search(lines[j])
            if rm:
                vals = [float(x) for x in rm.groups()]
                rows.append((vals[:3], vals[3:]))
            j += 1
        if name not in ref:  # keep first occurrence
            ref[name] = rows
        i = j
    else:
        i += 1

def cmp_node(llama_name, our_file, tok=None):
    if llama_name not in ref:
        return None
    try:
        ours = np.fromfile(f'{GOLD}/{our_file}', dtype=np.float32).reshape(NTOK, DIM)
    except Exception:
        return None
    rows = ref[llama_name]
    worst = 0.0
    for t in range(min(len(rows), NTOK)):
        if tok is not None and t != tok: continue
        f3, l3 = rows[t]
        o_f3 = ours[t, :3]; o_l3 = ours[t, -3:]
        for a, b in zip(f3 + l3, list(o_f3) + list(o_l3)):
            denom = max(abs(a), abs(b), 1e-3)
            worst = max(worst, abs(a - b) / denom)
    return worst

names = sys.argv[1:] if len(sys.argv) > 1 else ['l_out']
NL = 35
for base in names:
    print(f'--- {base} ---')
    for L in range(NL):
        r = cmp_node(f'{base}-{L}', f'{base}-{L}.bin')
        if r is None:
            # llama uses different name pattern for layer suffix sometimes
            print(f'  L{L:2d}: (missing)')
            continue
        flag = '  <<<' if r > 0.05 else ''
        print(f'  L{L:2d}: rel-diff {r:.4f}{flag}')
