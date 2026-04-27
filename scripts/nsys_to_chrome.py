#!/usr/bin/env python3
"""
Convert one or more `nsys-rep` files to a single Chrome-trace JSON for
side-by-side viewing.

Each input rep becomes its own thread (process) in the resulting trace,
so hesper graphs OFF / hesper graphs ON / llama.cpp can be compared at
the same wall-time scale.

Usage:
    nsys_to_chrome.py REP[:LABEL] [REP[:LABEL] ...] [--out trace.json]

Each rep is normalised so its first GPU kernel starts at t=0 (so
threads of different absolute start times line up).

Requires:
    nsys  (Nsight Systems) on PATH
    sqlite3 for python (stdlib)

Output JSON is in Chrome's "Trace Event" format (object form).
View with `chrome://tracing` or https://ui.perfetto.dev.
"""
import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

def export_sqlite(rep: Path) -> Path:
    """Run `nsys export -t sqlite REP` if the .sqlite doesn't already exist."""
    sql = rep.with_suffix(".sqlite")
    if not sql.exists():
        subprocess.run(
            ["nsys", "export", "-t", "sqlite", "-o", str(sql), str(rep)],
            check=True,
        )
    return sql

def kernels_of(sql: Path) -> list[tuple[int, int, str]]:
    """Return [(start_ns, dur_ns, name), …] for CUDA kernel launches.

    Falls back to gracefully empty if the rep has no kernel data
    (driver hooks didn't capture them — e.g. some hesper modes).
    """
    con = sqlite3.connect(str(sql))
    cur = con.cursor()
    # CUPTI_ACTIVITY_KIND_KERNEL has start/end + shortName / demangledName ids
    try:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='CUPTI_ACTIVITY_KIND_KERNEL'")
        if not cur.fetchone():
            return []
    except sqlite3.DatabaseError:
        return []
    cur.execute("""
        SELECT k.start, k.end - k.start, COALESCE(s2.value, s.value)
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        LEFT JOIN StringIds s  ON k.shortName = s.id
        LEFT JOIN StringIds s2 ON k.demangledName = s2.id
        ORDER BY k.start
    """)
    return [(int(r[0]), int(r[1]), r[2] or "<unknown>") for r in cur.fetchall()]

def memops_of(sql: Path) -> list[tuple[int, int, str]]:
    con = sqlite3.connect(str(sql))
    cur = con.cursor()
    try:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='CUPTI_ACTIVITY_KIND_MEMCPY'")
        if not cur.fetchone():
            return []
    except sqlite3.DatabaseError:
        return []
    cur.execute("""
        SELECT m.start, m.end - m.start, m.copyKind
        FROM CUPTI_ACTIVITY_KIND_MEMCPY m
        ORDER BY m.start
    """)
    kind_map = {1: "HtoD", 2: "DtoH", 8: "DtoD"}
    return [(int(r[0]), int(r[1]), f"memcpy_{kind_map.get(r[2], r[2])}") for r in cur.fetchall()]

def cuda_api_of(sql: Path) -> list[tuple[int, int, str]]:
    con = sqlite3.connect(str(sql))
    cur = con.cursor()
    try:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='CUPTI_ACTIVITY_KIND_RUNTIME'")
        has_rt = bool(cur.fetchone())
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='CUPTI_ACTIVITY_KIND_DRIVER'")
        has_drv = bool(cur.fetchone())
    except sqlite3.DatabaseError:
        return []
    rows: list[tuple[int, int, str]] = []
    for tbl in [t for t, ok in [("CUPTI_ACTIVITY_KIND_RUNTIME", has_rt), ("CUPTI_ACTIVITY_KIND_DRIVER", has_drv)] if ok]:
        try:
            cur.execute(f"""
                SELECT a.start, a.end - a.start, COALESCE(s.value, '<api>')
                FROM {tbl} a
                LEFT JOIN StringIds s ON a.nameId = s.id
                WHERE (a.end - a.start) > 100  -- skip <100ns noise
                ORDER BY a.start
            """)
            rows += [(int(r[0]), int(r[1]), r[2]) for r in cur.fetchall()]
        except sqlite3.DatabaseError as e:
            print(f"  warn: {tbl} query failed: {e}", file=sys.stderr)
    return rows

def build_trace(reps: list[tuple[str, Path]]) -> dict:
    events: list[dict] = []
    pid = 1
    for label, rep in reps:
        print(f"== {label} == ({rep})", file=sys.stderr)
        sql = export_sqlite(rep)
        kerns = kernels_of(sql)
        memos = memops_of(sql)
        apis  = cuda_api_of(sql)
        print(f"  kernels: {len(kerns)} memops: {len(memos)} apis: {len(apis)}", file=sys.stderr)

        # Normalise: subtract earliest event time across all categories.
        starts = [k[0] for k in kerns + memos + apis]
        if not starts:
            print(f"  (no events, skipping)", file=sys.stderr)
            continue
        t0 = min(starts)

        events.append({"name": "process_name", "ph": "M", "pid": pid, "tid": 0,
                       "args": {"name": label}})
        # GPU kernels on tid=1
        events.append({"name": "thread_name", "ph": "M", "pid": pid, "tid": 1,
                       "args": {"name": "GPU kernels"}})
        for s, d, name in kerns:
            if d <= 0: continue
            events.append({"name": name[:80], "cat": "kernel", "ph": "X",
                           "pid": pid, "tid": 1,
                           "ts": (s - t0) / 1000.0, "dur": d / 1000.0})
        # Memcpy on tid=2
        events.append({"name": "thread_name", "ph": "M", "pid": pid, "tid": 2,
                       "args": {"name": "memcpy"}})
        for s, d, name in memos:
            if d <= 0: continue
            events.append({"name": name, "cat": "memcpy", "ph": "X",
                           "pid": pid, "tid": 2,
                           "ts": (s - t0) / 1000.0, "dur": d / 1000.0})
        # CUDA API on tid=3
        events.append({"name": "thread_name", "ph": "M", "pid": pid, "tid": 3,
                       "args": {"name": "CUDA API"}})
        for s, d, name in apis:
            if d <= 0: continue
            events.append({"name": name[:80], "cat": "api", "ph": "X",
                           "pid": pid, "tid": 3,
                           "ts": (s - t0) / 1000.0, "dur": d / 1000.0})

        pid += 1
    return {"traceEvents": events, "displayTimeUnit": "ms"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+",
                    help="REP[:LABEL] (label defaults to file stem)")
    ap.add_argument("--out", default="trace.json")
    args = ap.parse_args()

    parsed: list[tuple[str, Path]] = []
    for inp in args.inputs:
        if ":" in inp:
            rep, lbl = inp.rsplit(":", 1)
            parsed.append((lbl, Path(rep)))
        else:
            p = Path(inp)
            parsed.append((p.stem, p))
    trace = build_trace(parsed)
    Path(args.out).write_text(json.dumps(trace))
    print(f"wrote {args.out} ({len(trace['traceEvents'])} events)", file=sys.stderr)

if __name__ == "__main__":
    main()
