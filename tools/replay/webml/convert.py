#!/usr/bin/env python3
"""trace.json -> out/ (per-pipeline MSL via tint) + manifest.json for replayer.mm.

Resolves per-pipeline: MSL file, actual entry name, (group,binding)->[[buffer(K)]]
map, workgroup size (override-aware), threadgroup-memory upper bound.
"""
import base64
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TINT = os.environ.get("HESPER_TINT_BIN")
assert TINT, "set HESPER_TINT_BIN"
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

trace = json.load(open(os.path.join(HERE, "trace.json")))
shaders = {s["id"]: s["code"] for s in trace["shaders"]}
pipes = {p["id"]: p for p in trace["pipes"]}

SCALAR = {"f16": 2, "f32": 4, "u32": 4, "i32": 4, "atomic<u32>": 4, "atomic<i32>": 4}

def eval_expr(expr, env):
    expr = expr.strip().rstrip("u")
    expr = re.sub(r"(\d+)u", r"\1", expr)
    names = set(re.findall(r"[A-Za-z_]\w*", expr))
    for n in names:
        if n not in env:
            raise ValueError(f"unknown ident {n} in {expr!r}")
    return int(eval(expr, {"__builtins__": {}}, env))

def override_env(code, constants):
    env = {}
    for m in re.finditer(r"override\s+([A-Za-z_]\w*)\s*(?::\s*\w+)?\s*(?:=\s*([^;]+))?;", code):
        name, default = m.group(1), m.group(2)
        if default is not None:
            try:
                env[name] = eval_expr(default, {})
            except Exception:
                pass
    for m in re.finditer(r"const\s+([A-Za-z_]\w*)\s*(?::\s*\w+)?\s*=\s*([^;]+);", code):
        try:
            env[m.group(1)] = eval_expr(m.group(2), env)
        except Exception:
            pass
    for k, v in (constants or {}).items():
        env[k] = int(v)
    return env

def type_bytes(ty, env):
    ty = ty.strip()
    m = re.match(r"array<\s*(.+)\s*,\s*([^,>]+)\s*>$", ty)
    if m:
        return type_bytes(m.group(1), env) * eval_expr(m.group(2), env)
    m = re.match(r"vec(\d)<\s*(\w+)\s*>$", ty)
    if m:
        n = int(m.group(1))
        return (4 if n == 3 else n) * SCALAR[m.group(2)]
    if ty in SCALAR:
        return SCALAR[ty]
    return 64  # unknown: generous

manifest_pipes = []
pipe_index = {}  # trace pipe id -> manifest index

for pid, p in sorted(pipes.items()):
    code = shaders[p["shaderId"]]
    env = override_env(code, p.get("constants"))
    wgsl_path = os.path.join(OUT, f"p{pid}.wgsl")
    open(wgsl_path, "w").write(code)
    cmd = [TINT, "--format", "msl", wgsl_path]
    if p.get("constants"):
        cmd += ["--overrides", ",".join(f"{k}={int(v)}" for k, v in p["constants"].items())]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[convert] tint FAILED pipe {pid}: {r.stderr[:500]}", file=sys.stderr)
        sys.exit(1)
    msl = r.stdout
    open(os.path.join(OUT, f"p{pid}.msl"), "w").write(msl)

    # WGSL (group,binding) -> var name
    gb2name = {}
    for m in re.finditer(r"@group\((\d+)\)\s*@binding\((\d+)\)[^;]*?\bvar\b[^:;]*?([A-Za-z_]\w*)\s*:", code):
        gb2name[(int(m.group(1)), int(m.group(2)))] = m.group(3)
    # MSL name -> buffer index (+ actual entry name)
    em = re.search(r"kernel void (\w+)\(", msl)
    entry = em.group(1)
    sig = msl[em.end():].split("{", 1)[0]
    name2k = {}
    for m in re.finditer(r"([A-Za-z_]\w*)\s*\[\[buffer\((\d+)\)\]\]", sig):
        name2k[m.group(1)] = int(m.group(2))
    has_tg = "[[threadgroup(0)]]" in sig
    # workgroup size
    wm = re.search(r"@workgroup_size\(([^)]*)\)", code)
    parts = [s.strip() for s in wm.group(1).split(",")] if wm else ["1"]
    while len(parts) < 3:
        parts.append("1")
    wg = [eval_expr(s, env) for s in parts]
    # threadgroup memory upper bound
    tg_bytes = 0
    for m in re.finditer(r"var<workgroup>\s*[A-Za-z_]\w*\s*:\s*([^;]+);", code):
        try:
            tg_bytes += (type_bytes(m.group(1), env) + 15) // 16 * 16
        except Exception:
            tg_bytes += 4096
    pipe_index[pid] = len(manifest_pipes)
    manifest_pipes.append({
        "msl": f"p{pid}.msl", "entry": entry, "wg": wg,
        "tgBytes": tg_bytes if has_tg else 0,
        "gb2k": {f"{g},{b}": name2k[n] for (g, b), n in gb2name.items() if n in name2k},
    })

buffers = {}
ops_out = []
skipped = 0
for op in trace["ops"]:
    pm = pipe_index.get(op["pipe"])
    if pm is None:
        skipped += 1
        continue
    binds = []
    gb2k = manifest_pipes[pm]["gb2k"]
    for g in op["bgs"]:
        for e in g["entries"]:
            k = gb2k.get(f"{g['group']},{e['binding']}")
            if k is None:
                continue  # binding stripped by tint (unused) — skip
            buffers[e["buf"]] = max(buffers.get(e["buf"], 0), e["bufSize"])
            binds.append([k, e["buf"], e["offset"]])
    ops_out.append({"p": pm, "grid": op["grid"], "binds": binds})

manifest = {
    "pipes": manifest_pipes,
    "ops": ops_out,
    "buffers": [{"id": k, "size": v} for k, v in sorted(buffers.items())],
    "contents": trace.get("contents", {}),
}
json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"))
total = sum(b["size"] for b in manifest["buffers"])
print(f"[convert] pipes={len(manifest_pipes)} ops={len(ops_out)} skipped={skipped} "
      f"buffers={len(buffers)} totalBufBytes={total/1e9:.2f}GB contents={len(manifest['contents'])}")
