#!/usr/bin/env python3
"""1:1 kernel comparison hesper(graphs ON) vs llama-cli(graphs OFF).

Reads two nsys sqlite files (each captured with --cuda-graph-trace=node),
classifies kernels by gridX × blockX × call count / tok, and prints a
side-by-side table sorted by total kernel time per token.

Usage:
  python scripts/kernel_compare_graphs.py \
    /dev/shm/.../h_g2.sqlite NTOKENS_HS \
    /dev/shm/.../lcli.sqlite NTOKENS_LC
"""
import sqlite3
import sys
from collections import defaultdict


def load_kernels(path: str, decode_t0_ns: int, decode_t1_ns: int):
    """Returns list of (shortName, gridX, gridY, blockX, dur_ns)."""
    conn = sqlite3.connect(path)
    cur = conn.execute("""
        SELECT s.value, k.gridX, k.gridY, k.blockX, k.end - k.start
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id = k.shortName
        WHERE k.start >= ? AND k.end <= ?
    """, (decode_t0_ns, decode_t1_ns))
    return cur.fetchall()


def get_decode_range(path: str, mode: str):
    conn = sqlite3.connect(path)
    if mode == "graphs":
        # decode = bracket of cuGraphLaunch calls
        row = conn.execute("""
            SELECT MIN(r.start), MAX(r.end)
            FROM CUPTI_ACTIVITY_KIND_RUNTIME r
            JOIN StringIds s ON s.id = r.nameId
            WHERE s.value = 'cuGraphLaunch'
        """).fetchone()
    else:
        # llama-cli: decode = last 80% of run timeline (drop load/prefill)
        bounds = conn.execute("""
            SELECT MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL
        """).fetchone()
        if bounds[0] is None:
            return None, None
        span = bounds[1] - bounds[0]
        row = (bounds[0] + int(span * 0.2), bounds[1])
    return row


def classify_lc(shortname: str, gridX: int, blockX: int) -> str:
    """Classify llama.cpp kernel by name."""
    n = shortname
    if "mul_mat_vec_q" in n: return "matmul (Q4_K/Q6_K mmvq)"
    if "mul_mat_q" in n: return "matmul (Q4_K mmq)"
    if "mul_mat_vec_f" in n: return "matmul (f16 vec)"
    if "flash_attn_ext_vec" in n: return "flashAttn"
    if "flash_attn_ext_f16" in n: return "flashAttn"
    if "rms_norm_f32" in n: return "RMSNorm"
    if "quantize_q8_1" in n: return "quantize_q8_1"
    if "rope" in n.lower(): return "RoPE"
    if "soft_max" in n: return "softmax"
    if "k_bin_bcast" in n: return "binary bcast (mul/add)"
    if "k_unary" in n.lower() or "gelu" in n.lower(): return "unary (gelu/...)"
    if "set_rows" in n.lower() or "k_set" in n: return "KV cache write"
    if "argmax" in n.lower() or "k_argmax" in n: return "argmax"
    if "cpy" in n.lower() or "k_cpy" in n: return "copy"
    return f"OTHER ({n[:30]})"


def classify_hs(shortname: str, gridX: int, gridY: int, blockX: int) -> str:
    """Classify hesper kernel by funcName prefix + grid/block."""
    n = shortname
    # Extract human prefix (before the 'k' digit hash)
    prefix = n.split("_k")[0] if "_k" in n else ""
    p = prefix.lower()

    # By prefix
    if "fsdq4kmlnrdp4a" in p or "fsdq4km" in p: return "matmul Q4_K (fused gate+up etc.)"
    if "q6kmm" in p or "lmhd" in p or p == "q6k" or "q6kfn" in p: return "matmul Q6_K"
    if "rpdyn" in p or "rope" in p: return "RoPE"
    if "qkvnrm" in p: return "qkvNorm (RMSNorm fused)"
    if "rmsnrm" in p or "rmsnrmsw" in p: return "RMSNorm"
    if "flshttn" in p: return "flashAttn"
    if "kvwrt" in p or "kvwrite" in p: return "KV cache write"
    if "q8quant" in p or "qntq8" in p: return "quantize_q8_1"
    if "fsdplpstscl" in p: return "PLE post-scale (fused)"
    if "fsdnrmdd" in p or "plfsdnrmdd" in p: return "RMSNorm+add (fused)"
    if "rgmx" in p: return "argmax"
    if "embdlkp" in p or "q6kmblkp" in p or "embd" in p: return "embedding lookup"
    if "embdscl" in p: return "embedScale"
    if "lgtsftcp" in p or "softcap" in p: return "logit softcap"

    # Heuristics for unprefixed (gx, bx)
    if blockX == 128:
        if gridX in (2560, 4096, 10240, 32768): return f"matmul Q4_K bx=128 gx={gridX}"
        if gridX == 128: return "binary bcast"
    if blockX == 256:
        if gridX == 1: return "argmax/reduce single-WG"
        return f"RMSNorm/stub bx=256 gx={gridX}"
    if blockX == 32:
        if gridX in (2560,): return "matmul ffn_down 1-row Q6_K"
    if blockX == 1024:
        return "RMSNorm 1024-block"
    return f"UNCLASSIFIED bx={blockX} gx={gridX} prefix={prefix}"


def aggregate(path: str, mode: str, n_tokens: int):
    t0, t1 = get_decode_range(path, mode)
    if t0 is None: return {}, 0, 0
    rows = load_kernels(path, t0, t1)
    by_class = defaultdict(lambda: [0, 0])  # [total_ns, count]
    for shortname, gx, gy, bx, dur in rows:
        if mode == "graphs":
            c = classify_hs(shortname, gx, gy, bx)
        else:
            c = classify_lc(shortname, gx, bx)
        by_class[c][0] += dur
        by_class[c][1] += 1
    return by_class, (t1 - t0), len(rows)


def main():
    if len(sys.argv) != 5:
        print(__doc__)
        sys.exit(1)
    hs_db, hs_n = sys.argv[1], int(sys.argv[2])
    lc_db, lc_n = sys.argv[3], int(sys.argv[4])

    hs_agg, hs_wall, hs_total = aggregate(hs_db, "graphs", hs_n)
    lc_agg, lc_wall, lc_total = aggregate(lc_db, "lcli", lc_n)

    print(f"# hesper graphs ON: {hs_n} tokens, decode wall {hs_wall/1e6:.1f} ms, "
          f"{hs_total} kernel events ({hs_total/hs_n:.0f}/tok)")
    print(f"# llama-cli graphs OFF: {lc_n} tokens, decode wall {lc_wall/1e6:.1f} ms, "
          f"{lc_total} kernel events ({lc_total/lc_n:.0f}/tok)")
    print()

    # Common class roster: union, sorted by hs total time
    all_classes = sorted(set(hs_agg) | set(lc_agg),
                          key=lambda c: -(hs_agg.get(c, [0,0])[0] + lc_agg.get(c, [0,0])[0]))

    # Print a unified table per class (note hs class names differ from lc; that's OK,
    # we just sort each side independently and align by canonical name where we can)
    canon = {
        "matmul Q4_K (fused gate+up etc.)": "matmul Q4_K",
        "matmul Q4_K bx=128 gx=2560": "matmul Q4_K",
        "matmul Q4_K bx=128 gx=4096": "matmul Q4_K",
        "matmul Q4_K bx=128 gx=10240": "matmul Q4_K",
        "matmul Q4_K bx=128 gx=32768": "matmul Q4_K",
        "matmul Q4_K mmq": "matmul Q4_K",
        "matmul Q6_K": "matmul Q6_K",
        "matmul Q6_K (mmvq)": "matmul Q6_K",
        "matmul ffn_down 1-row Q6_K": "matmul Q6_K (ffn_down)",
        "matmul (Q4_K/Q6_K mmvq)": "matmul (mmvq Q4_K+Q6_K)",
        "matmul (Q4_K mmq)": "matmul Q4_K (mmq)",
        "RMSNorm 1024-block": "RMSNorm",
    }
    def normalize(c):
        if c in canon: return canon[c]
        for prefix, n in canon.items():
            if c.startswith(prefix): return n
        return c

    hs_norm = defaultdict(lambda: [0, 0])
    lc_norm = defaultdict(lambda: [0, 0])
    for c, (t, k) in hs_agg.items():
        n = normalize(c)
        hs_norm[n][0] += t; hs_norm[n][1] += k
    for c, (t, k) in lc_agg.items():
        n = normalize(c)
        lc_norm[n][0] += t; lc_norm[n][1] += k

    rows = []
    for c in sorted(set(hs_norm) | set(lc_norm)):
        h_t, h_k = hs_norm.get(c, (0, 0))
        l_t, l_k = lc_norm.get(c, (0, 0))
        rows.append((c, h_t, h_k, l_t, l_k))
    rows.sort(key=lambda r: -(r[1] + r[3]))

    fmt = "{:<40s}  {:>10s} {:>10s} {:>10s}    {:>10s} {:>10s} {:>10s}"
    print(fmt.format("kernel class", "h_ms/tok", "h_calls/tok", "h_µs/call",
                                       "l_ms/tok", "l_calls/tok", "l_µs/call"))
    print("-" * 110)
    for c, h_t, h_k, l_t, l_k in rows:
        h_ms = h_t / 1e6 / hs_n
        h_per = h_k / hs_n if h_k else 0
        h_us = (h_t / 1e3 / h_k) if h_k else 0
        l_ms = l_t / 1e6 / lc_n
        l_per = l_k / lc_n if l_k else 0
        l_us = (l_t / 1e3 / l_k) if l_k else 0
        print(fmt.format(c[:40],
                         f"{h_ms:.3f}" if h_ms else "-",
                         f"{h_per:.1f}" if h_per else "-",
                         f"{h_us:.1f}" if h_us else "-",
                         f"{l_ms:.3f}" if l_ms else "-",
                         f"{l_per:.1f}" if l_per else "-",
                         f"{l_us:.1f}" if l_us else "-"))
    print("-" * 110)
    h_total = sum(t for _, t, _, _, _ in rows) / 1e6 / hs_n
    l_total = sum(t for _, _, _, t, _ in rows) / 1e6 / lc_n
    print(fmt.format("TOTAL", f"{h_total:.3f}", "", "",
                                f"{l_total:.3f}", "", ""))


if __name__ == "__main__":
    main()
