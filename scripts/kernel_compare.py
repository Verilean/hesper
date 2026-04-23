#!/usr/bin/env python3
"""
Kernel-class 1:1 comparison between hesper stub decode path and llama.cpp.

Produces the table from docs/llama-fusion-analysis/40-kernel-1to1-comparison.md
by running both systems graphs-OFF, profiling with nsys, and aggregating by
kernel class.

Usage:
    scripts/kernel_compare.py [--prompt "Hello world how are you"]
                              [--decodes 10] [--lc-decodes 20]
                              [--out /tmp/kernel_compare]
                              [--skip-build]   # skip lake rebuild
                              [--lc-only]      # only re-run llama.cpp
                              [--hs-only]      # only re-run hesper

Requires:
    - `nsys` on PATH (Nsight Systems)
    - Gemma 4 model at data/gemma-4-e4b-it-Q4_K_M.gguf
    - llama.cpp built at llama.cpp/build/bin/llama-bench
    - `lake exe gemma4-llama-prefill-skeleton` builds

Output: prints comparison table to stdout, writes raw CSVs + per-kernel
detail to {out}/ for follow-up investigation.
"""
import argparse, csv, io, os, re, shutil, subprocess, sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODEL = "data/gemma-4-e4b-it-Q4_K_M.gguf"


def sh(cmd: list[str], env: dict | None = None, check: bool = True, cwd: Path = ROOT,
       quiet: bool = True) -> subprocess.CompletedProcess:
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    if not quiet:
        print(f"$ {' '.join(cmd)}", file=sys.stderr)
    return subprocess.run(cmd, env=full_env, cwd=cwd, check=check,
                          capture_output=True, text=True)


def nsys_kern_csv(nsys_rep: Path) -> list[list[str]]:
    """Return list of rows from `nsys stats --report cuda_gpu_kern_sum`."""
    out = sh(["nsys", "stats", "--force-export=true", "--report",
              "cuda_gpu_kern_sum", "--format", "csv", str(nsys_rep)]).stdout
    rows = []
    header = None
    for row in csv.reader(io.StringIO(out)):
        if header is None:
            if row and row[0] == "Time (%)":
                header = row
            continue
        if len(row) >= 9:
            rows.append(row)
    return rows


# ── llama.cpp classification ─────────────────────────────────────────
def classify_lc(name: str) -> str:
    # CUDA kernel names from ggml-cuda; ggml_type enum: 12=Q4_K, 14=Q6_K
    if "mul_mat_vec_q<(ggml_type)12" in name: return "Q4_K matmul"
    if "mul_mat_vec_q<(ggml_type)14" in name: return "Q6_K matmul"
    if "mul_mat_vec_f" in name:               return "FlashAttn (mul_mat_vec_f)"
    if "rms_norm_f32" in name:                return "RMSNorm"
    if "quantize_q8_1" in name:               return "quantize_q8_1"
    if "rope_neox" in name:                   return "RoPE"
    if "k_bin_bcast" in name:                 return "binary bcast (add/mul)"
    if "k_set_rows" in name:                  return "KV cache write"
    if "soft_max_f32" in name:                return "softmax"
    if "unary_op_kernel" in name:             return "GELU/unary"
    if "softcap_f32" in name:                 return "softcap"
    if "scale_f32" in name:                   return "scale"
    if "pad_f32" in name:                     return "pad"
    if "k_get_rows_float" in name:            return "get_rows"
    # fallback: first word of mangled name
    return name.split("<")[0].split("(")[-1].strip() or "(other)"


# ── hesper classification (grid-based) ───────────────────────────────
def classify_hs(grid: tuple[int, int, int, int] | None) -> str:
    if grid is None:
        return "(untagged)"
    gx, gy, gz, bx = grid
    phase = "decode" if gy == 1 else "prefill"
    # Dispatch identification (block=128 family):
    #   - Q4_K 4-warp: grid=(outDim, 1, 1), block=128, one WG per output row
    #   - Q6_K 4-row:  grid=(outDim/4, 1, 1), block=128, 4 rows per WG
    # Gemma 4 E4B ffn_down is Q6_K (outDim=2560 → grid=640), all other
    # per-layer matmuls are Q4_K (so grid == outDim).
    if bx == 128:
        if gx == 65536:  return f"Q6_K lm_head ({phase})"            # vocab/4
        if gx == 10240:  return f"Q4_K matmul 10240 ({phase})"       # ffn_gate/up  (Q4_K 4-warp)
        if gx == 2560:   return f"Q4_K matmul 2560 ({phase})"        # wO           (Q4_K 4-warp)
        if gx == 2048:   return f"Q4_K matmul 2048 ({phase})"        # wQ           (Q4_K 4-warp)
        if gx == 640:    return f"Q6_K matmul ffn_down ({phase})"    # ffn_down 2560/4 (Q6_K 4-row)
        if gx == 512:    return f"Q4_K matmul 512 ({phase})"         # wK or wV     (Q4_K 4-warp)
        if gx == 256:    return f"Q4_K matmul 256 ({phase})"         # PLE          (Q4_K 4-warp)
        if gx == 4096:   return f"Q4_K matmul 4096 ({phase})"        # perLayerProj (Q4_K 4-warp)
        if gx == 128:    return f"pointwise/residual ({phase})"
        return f"b=128 other gx={gx} ({phase})"
    # 1-warp matmul (prefill path), block=32
    if bx == 32:
        if gx == 10240:  return f"Q4_K matmul 1-warp 10240 ({phase})"
        if gx == 2560:   return f"Q4_K matmul 1-warp 2560 ({phase})"
        if gx == 2048:   return f"Q4_K matmul 1-warp 2048 ({phase})"
        if gx == 80:     return f"quantize_q8_1 hidden ({phase})"
        if gx == 320:    return f"quantize_q8_1 inter ({phase})"
        if gx == 8:      return f"quantize_q8_1 embd ({phase})"
        return f"b=32 other gx={gx} ({phase})"
    # block=256: mostly RMSNorm + pointwise stubs
    if bx == 256:
        return f"RMSNorm/stub b=256 gx={gx} ({phase})"
    return f"b={bx} other ({phase})"


# hesper class → llama.cpp class (for aggregation)
def hs_to_lc_class(c: str) -> str:
    if c.startswith("Q4_K matmul"):        return "Q4_K matmul"
    if c.startswith("Q4_K matmul 1-warp"): return "Q4_K matmul"
    if c.startswith("Q6_K lm_head"):       return "Q6_K matmul"
    if c.startswith("Q6_K matmul"):        return "Q6_K matmul"  # ffn_down etc.
    if c.startswith("quantize_q8_1"):      return "quantize_q8_1"
    if c.startswith("pointwise/residual"): return "binary bcast (add/mul)"
    if c.startswith("RMSNorm/stub"):       return "RMSNorm"
    return "(unmapped)"


# ── hesper profile + hash labels ─────────────────────────────────────
HASH_RE = re.compile(r"\[hs\] (k_\d+)\s+grid=\((\d+),(\d+),(\d+)\)\s+block=\((\d+),\d+,\d+\)")


def collect_hesper_labels(prompt: str, out_dir: Path) -> dict[str, tuple[int, int, int, int]]:
    """Run hesper with HESPER_KERNEL_TRACE=1 to build hash → grid/block map."""
    env = {"HESPER_DP4A": "1", "HESPER_LLAMA_GRAPHS": "0", "HESPER_KERNEL_TRACE": "1"}
    trace = sh(["lake", "exe", "gemma4-llama-prefill-skeleton", MODEL, prompt, "2"],
               env=env, check=False).stderr
    labels: dict[str, tuple[int, int, int, int]] = {}
    for line in trace.splitlines():
        m = HASH_RE.match(line)
        if m:
            h = m.group(1)
            if h not in labels:
                labels[h] = tuple(int(x) for x in m.groups()[1:])
    (out_dir / "hs_labels.txt").write_text("\n".join(
        f"{h} grid=({g[0]},{g[1]},{g[2]}) block=({g[3]},1,1)" for h, g in sorted(labels.items())))
    return labels


def profile_hesper(prompt: str, decodes: int, out_dir: Path) -> Path:
    rep = out_dir / "hesper.nsys-rep"
    env = {"HESPER_DP4A": "1", "HESPER_LLAMA_GRAPHS": "0"}
    sh(["nsys", "profile", "-t", "cuda", "--stats=false",
        "-o", str(rep.with_suffix("")), "-f", "true",
        "lake", "exe", "gemma4-llama-prefill-skeleton", MODEL, prompt, str(decodes)],
       env=env, check=False)
    return rep


def profile_llamacpp(decodes: int, out_dir: Path) -> Path:
    rep = out_dir / "llamacpp.nsys-rep"
    env = {"GGML_CUDA_DISABLE_GRAPHS": "1"}
    sh(["nsys", "profile", "-t", "cuda", "--stats=false",
        "-o", str(rep.with_suffix("")), "-f", "true",
        "llama.cpp/build/bin/llama-bench",
        "-m", MODEL, "-p", "0", "-n", str(decodes), "-r", "1",
        "-ngl", "99", "--no-warmup"],
       env=env, check=False)
    return rep


# ── aggregation ──────────────────────────────────────────────────────
def aggregate_hs(rep: Path, labels: dict, n_decodes: int) -> dict:
    """Returns {class: (total_ns, inst)} for DECODE-PHASE kernels only."""
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in nsys_kern_csv(rep):
        try:
            total_ns = int(r[1]); inst = int(r[2]); name = r[-1]
        except ValueError:
            continue
        if not name.startswith("k_"):
            agg["(untagged)"][0] += total_ns
            agg["(untagged)"][1] += inst
            continue
        grid = labels.get(name)
        c = classify_hs(grid)
        agg[c][0] += total_ns
        agg[c][1] += inst
    return dict(agg)


def aggregate_lc(rep: Path, n_decodes: int) -> dict:
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in nsys_kern_csv(rep):
        try:
            total_ns = int(r[1]); inst = int(r[2]); name = r[-1]
        except ValueError:
            continue
        c = classify_lc(name)
        agg[c][0] += total_ns
        agg[c][1] += inst
    return dict(agg)


# ── reporting ────────────────────────────────────────────────────────
def build_comparison(hs_agg: dict, hs_decodes: int, lc_agg: dict, lc_decodes: int) -> str:
    # Map hs classes → lc classes, summing only DECODE phase
    hs_by_lc: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    for c, (t, inst) in hs_agg.items():
        if "(prefill)" in c:
            continue  # skip prefill kernels for decode comparison
        mapped = hs_to_lc_class(c)
        hs_by_lc[mapped][0] += t / 1e6 / hs_decodes  # ms/decode
        hs_by_lc[mapped][1] += inst / hs_decodes

    lc_by_class: dict[str, tuple[float, float]] = {
        c: (t / 1e6 / lc_decodes, inst / lc_decodes)
        for c, (t, inst) in lc_agg.items()
    }

    # Union of classes, sorted by llama.cpp total ms (biggest impact first)
    classes = sorted(set(hs_by_lc) | set(lc_by_class),
                     key=lambda c: -lc_by_class.get(c, (0, 0))[0])

    lines = []
    lines.append(f"{'=' * 92}")
    lines.append(f"Per-class 1:1 comparison (hesper graphs OFF, {hs_decodes} decodes  "
                 f"vs llama.cpp graphs OFF, {lc_decodes} decodes)")
    lines.append(f"{'=' * 92}")
    lines.append(f"{'category':<30} {'hs ms':>10} {'lc ms':>10} {'ratio':>8}  "
                 f"{'hs inst':>10} {'lc inst':>10}")
    lines.append("-" * 92)
    hs_total = lc_total = 0.0
    for c in classes:
        hs_t, hs_i = hs_by_lc.get(c, (0, 0))
        lc_t, lc_i = lc_by_class.get(c, (0, 0))
        hs_total += hs_t
        lc_total += lc_t
        ratio_str = f"{hs_t / lc_t:>7.2f}×" if lc_t > 0 else ("  —"   if hs_t == 0 else "  ∞ ")
        lines.append(f"{c:<30} {hs_t:>10.3f} {lc_t:>10.3f} {ratio_str}  "
                     f"{hs_i:>10.1f} {lc_i:>10.1f}")
    lines.append("-" * 92)
    ratio = hs_total / lc_total if lc_total > 0 else float("inf")
    lines.append(f"{'TOTAL (ms per decode)':<30} {hs_total:>10.3f} {lc_total:>10.3f} "
                 f"{ratio:>7.2f}×")
    lines.append("")
    lines.append("Notes:")
    lines.append("  - hesper '(unmapped)' = stubs / tiny kernels not yet classified;")
    lines.append("    includes FlashAttn, RoPE, KV write, softmax, GELU, scale, softcap.")
    lines.append("  - Q6_K lm_head in hesper maps to 'Q6_K matmul' here.")
    lines.append("  - Prefill-phase hesper kernels are excluded (decode comparison).")
    return "\n".join(lines)


def dump_hesper_details(hs_agg: dict, n_decodes: int, out_dir: Path) -> None:
    rows = [(t, inst, c) for c, (t, inst) in hs_agg.items()]
    rows.sort(reverse=True)
    with (out_dir / "hesper_decode.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(["total_ns", "per_decode_ms", "inst", "inst_per_decode", "class"])
        for t, inst, c in rows:
            if "(prefill)" in c:
                continue
            w.writerow([t, f"{t/1e6/n_decodes:.4f}", inst,
                        f"{inst/n_decodes:.2f}", c])


def dump_llamacpp_details(lc_agg: dict, n_decodes: int, out_dir: Path) -> None:
    rows = [(t, inst, c) for c, (t, inst) in lc_agg.items()]
    rows.sort(reverse=True)
    with (out_dir / "llamacpp.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(["total_ns", "per_decode_ms", "inst", "inst_per_decode", "class"])
        for t, inst, c in rows:
            w.writerow([t, f"{t/1e6/n_decodes:.4f}", inst,
                        f"{inst/n_decodes:.2f}", c])


# ── main ─────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompt", default="Hello world how are you")
    p.add_argument("--decodes", type=int, default=10,
                   help="number of tokens for hesper (excluding prefill)")
    p.add_argument("--lc-decodes", type=int, default=20,
                   help="llama-bench -n (decode-only token count)")
    p.add_argument("--out", default="/tmp/kernel_compare")
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--lc-only", action="store_true")
    p.add_argument("--hs-only", action="store_true")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for tool in ("nsys", "lake"):
        if not shutil.which(tool):
            print(f"error: {tool} not found on PATH", file=sys.stderr)
            sys.exit(1)
    if not (ROOT / "llama.cpp" / "build" / "bin" / "llama-bench").exists():
        print("error: llama.cpp/build/bin/llama-bench not found", file=sys.stderr)
        sys.exit(1)

    if not args.skip_build:
        print("[build] lake build gemma4-llama-prefill-skeleton ...", file=sys.stderr)
        sh(["lake", "build", "gemma4-llama-prefill-skeleton"], check=False, quiet=False)

    hs_rep = out / "hesper.nsys-rep"
    lc_rep = out / "llamacpp.nsys-rep"

    if not args.lc_only:
        print(f"[hesper] profile: prompt='{args.prompt}', decodes={args.decodes}", file=sys.stderr)
        profile_hesper(args.prompt, args.decodes, out)
        print("[hesper] collect kernel labels via HESPER_KERNEL_TRACE", file=sys.stderr)
        labels = collect_hesper_labels(args.prompt, out)
    else:
        # reuse prior labels
        labels = {}
        lbl = out / "hs_labels.txt"
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                m = re.match(r"(k_\d+)\s+grid=\((\d+),(\d+),(\d+)\)\s+block=\((\d+),\d+,\d+\)", line)
                if m:
                    labels[m.group(1)] = tuple(int(x) for x in m.groups()[1:])

    if not args.hs_only:
        print(f"[llama.cpp] profile: -n {args.lc_decodes}", file=sys.stderr)
        profile_llamacpp(args.lc_decodes, out)

    if not hs_rep.exists() or not lc_rep.exists():
        print("error: missing nsys report(s); re-run without --hs-only/--lc-only", file=sys.stderr)
        sys.exit(1)

    print("[aggregate] classify hesper kernels", file=sys.stderr)
    hs_agg = aggregate_hs(hs_rep, labels, args.decodes)
    dump_hesper_details(hs_agg, args.decodes, out)

    print("[aggregate] classify llama.cpp kernels", file=sys.stderr)
    lc_agg = aggregate_lc(lc_rep, args.lc_decodes)
    dump_llamacpp_details(lc_agg, args.lc_decodes, out)

    print(build_comparison(hs_agg, args.decodes, lc_agg, args.lc_decodes))
    print(f"\ndetails: {out}/hesper_decode.csv, {out}/llamacpp.csv, "
          f"{out}/hs_labels.txt", file=sys.stderr)


if __name__ == "__main__":
    main()
