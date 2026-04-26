#!/usr/bin/env python3
"""nsys sqlite -> Chrome trace event JSON.

Emits Google Trace Event Format (Perfetto / chrome://tracing) with:
  - one process per side (e.g. "hesper" / "llama-cli"), one thread per
    GPU stream and one thread for "CUDA API" (host calls).
  - "phase" tracks (load / prefill / decode) detected from CUDA API
    density along the timeline.
  - per-phase counters: gpu_busy_pct, kernel_count, host_idle_pct.

Usage:
  python3 scripts/nsys_to_chrome_trace.py \
    /dev/shm/perf_compare/h_pLgN.sqlite hesper \
    /dev/shm/perf_compare/lcli_pLgN.sqlite llama-cli \
    -o /tmp/compare.json

Open `/tmp/compare.json` at https://ui.perfetto.dev/ or
chrome://tracing.
"""
import argparse
import json
import os
import sqlite3
import sys
from typing import Iterator


def fetch_strings(conn: sqlite3.Connection) -> dict[int, str]:
    cur = conn.execute("SELECT id, value FROM StringIds")
    return {row[0]: row[1] for row in cur}


def fetch_kernels(conn: sqlite3.Connection, strings: dict[int, str]):
    sql = """SELECT start, end, streamId, shortName, demangledName
             FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"""
    for start, end, stream, short, demangled in conn.execute(sql):
        name = strings.get(short, f"k_{short}")
        yield {
            "ts_ns": start,
            "dur_ns": end - start,
            "stream": stream,
            "name": name,
            "long": strings.get(demangled, name),
        }


def fetch_apis(conn: sqlite3.Connection, strings: dict[int, str]):
    sql = """SELECT start, end, globalTid, nameId
             FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY start"""
    for start, end, tid, name_id in conn.execute(sql):
        name = strings.get(name_id, f"api_{name_id}")
        yield {"ts_ns": start, "dur_ns": end - start, "tid": tid, "name": name}


def fetch_memcpys(conn: sqlite3.Connection):
    sql = """SELECT start, end, streamId, bytes, copyKind
             FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start"""
    for start, end, stream, n_bytes, kind in conn.execute(sql):
        yield {
            "ts_ns": start,
            "dur_ns": end - start,
            "stream": stream,
            "bytes": n_bytes,
            "kind": kind,
        }


def detect_phases(api_events: list[dict], kern_events: list[dict]):
    """Naive 3-phase detector based on inter-API gap.

    load = period before first kernel, prefill = first kernel through the
    last kernel within 50 ms of the next kernel cluster, decode = the
    long uniformly-spaced run after that.

    Returns list of (label, ts_start_ns, ts_end_ns).
    """
    if not kern_events:
        return []
    first_k = kern_events[0]["ts_ns"]
    last_k = kern_events[-1]["ts_ns"] + kern_events[-1]["dur_ns"]
    # Gap-based prefill detection: after the first kernel, look for the
    # *largest* gap between adjacent kernels in the first 200 ms — that
    # gap is typically prefill->decode boundary in llama-cli (sampling
    # readback) and harmless on hesper (decode is uniform, max gap is
    # tiny).
    prefill_end = first_k
    horizon = first_k + int(200e6)
    max_gap = 0
    max_gap_start = first_k
    prev_end = first_k
    for k in kern_events:
        if k["ts_ns"] > horizon:
            break
        gap = k["ts_ns"] - prev_end
        if gap > max_gap:
            max_gap = gap
            max_gap_start = prev_end
        prev_end = k["ts_ns"] + k["dur_ns"]
    if max_gap > int(2e6):  # >2 ms gap qualifies as a phase boundary
        prefill_end = max_gap_start + max_gap // 2
    # Find load->prefill: last API before first kernel
    load_start = api_events[0]["ts_ns"] if api_events else first_k
    return [
        ("load", load_start, first_k),
        ("prefill", first_k, prefill_end),
        ("decode", prefill_end, last_k),
    ]


def gpu_busy_metric(kernels: list[dict], window_ns: int = int(50e6)):
    """Sliding-window GPU-busy ratio sample stream.

    Yields (ts_ns, busy_pct, kernel_rate_per_ms).
    """
    if not kernels:
        return
    t0 = kernels[0]["ts_ns"]
    t1 = kernels[-1]["ts_ns"] + kernels[-1]["dur_ns"]
    step = window_ns // 10
    i_start = 0
    n = len(kernels)
    t = t0
    while t <= t1:
        win_lo = t - window_ns
        win_hi = t
        # advance i_start while behind window
        while i_start < n and kernels[i_start]["ts_ns"] + kernels[i_start]["dur_ns"] < win_lo:
            i_start += 1
        busy = 0
        cnt = 0
        j = i_start
        while j < n and kernels[j]["ts_ns"] < win_hi:
            ks = kernels[j]["ts_ns"]
            ke = ks + kernels[j]["dur_ns"]
            ks = max(ks, win_lo)
            ke = min(ke, win_hi)
            if ke > ks:
                busy += ke - ks
                cnt += 1
            j += 1
        actual = win_hi - max(win_lo, t0)
        if actual > 0:
            yield t, busy / actual * 100.0, cnt
        t += step


def load_lean_trace(path: str | None) -> list[dict]:
    """Load Lean withSection events emitted by `dumpSectionTrace`.

    Format: list of {name, ts_ns, dur_ns, depth}.  ts_ns is in the same
    monotonic clock as nsys' kernel start times (both via clock_gettime
    on Linux), so we don't need a rebase pass beyond the global origin
    rebase already applied to the rest of the events.
    """
    if not path or not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def emit(side: str, sqlite_path: str, t_origin_ns: int,
         lean_trace_path: str | None = None) -> tuple[list[dict], dict]:
    conn = sqlite3.connect(sqlite_path)
    strings = fetch_strings(conn)
    kernels = list(fetch_kernels(conn, strings))
    apis = list(fetch_apis(conn, strings))
    memcpys = list(fetch_memcpys(conn))
    phases = detect_phases(apis, kernels)

    pid = abs(hash(side)) % 100000

    events: list[dict] = []
    # Process / thread metadata
    events.append({"name": "process_name", "ph": "M", "pid": pid, "tid": 0,
                   "args": {"name": side}})
    api_tid = 1
    events.append({"name": "thread_name", "ph": "M", "pid": pid, "tid": api_tid,
                   "args": {"name": "CUDA API"}})
    streams = sorted({k["stream"] for k in kernels} |
                     {m["stream"] for m in memcpys})
    stream_tid = {s: 100 + i for i, s in enumerate(streams)}
    for s, tid in stream_tid.items():
        events.append({"name": "thread_name", "ph": "M", "pid": pid, "tid": tid,
                       "args": {"name": f"GPU stream {s}"}})
    phase_tid = 50
    events.append({"name": "thread_name", "ph": "M", "pid": pid, "tid": phase_tid,
                   "args": {"name": "phase"}})
    counter_tid = 60
    # Lean withSection events get their own thread group, one tid per
    # nesting depth so nested sections stack cleanly in Perfetto.
    lean_tid_base = 300

    # Time origin: Chrome trace uses µs floats; rebase to t_origin so two
    # files line up.
    def to_us(ns: int) -> float:
        return (ns - t_origin_ns) / 1000.0

    # Phase markers as long-duration events on phase_tid
    phase_durations = {label: max(0, end - start) for label, start, end in phases}
    for label, start, end in phases:
        events.append({
            "name": label,
            "ph": "X",
            "pid": pid, "tid": phase_tid,
            "ts": to_us(start),
            "dur": (end - start) / 1000.0,
            "cname": {"load": "thread_state_running",
                      "prefill": "thread_state_runnable",
                      "decode": "thread_state_iowait"}[label],
        })

    # Kernel events on stream threads
    for k in kernels:
        events.append({
            "name": k["name"],
            "ph": "X",
            "pid": pid, "tid": stream_tid[k["stream"]],
            "ts": to_us(k["ts_ns"]),
            "dur": k["dur_ns"] / 1000.0,
            "args": {"long": k["long"][:200]},
        })

    # CUDA API events on api thread
    for a in apis:
        events.append({
            "name": a["name"],
            "ph": "X",
            "pid": pid, "tid": api_tid,
            "ts": to_us(a["ts_ns"]),
            "dur": a["dur_ns"] / 1000.0,
        })

    # Memcpy events on stream threads (separate sub-tid per stream so they
    # don't visually overlap kernels)
    memcpy_tid = {s: 200 + i for i, s in enumerate(streams)}
    for s, tid in memcpy_tid.items():
        events.append({"name": "thread_name", "ph": "M", "pid": pid, "tid": tid,
                       "args": {"name": f"memcpy stream {s}"}})
    for m in memcpys:
        events.append({
            "name": f"memcpy({m['bytes']}B)",
            "ph": "X",
            "pid": pid, "tid": memcpy_tid[m["stream"]],
            "ts": to_us(m["ts_ns"]),
            "dur": m["dur_ns"] / 1000.0,
            "args": {"bytes": m["bytes"], "kind": m["kind"]},
        })

    # Lean withSection events.  Lean uses CLOCK_MONOTONIC (absolute ns
    # since boot, ~1e15 range), while nsys CUPTI emits ns relative to
    # the profiling session start (~1e9 range).  We align the two by
    # rebasing Lean ts so its earliest event lines up with the
    # earliest GPU kernel — both correspond to the first dispatch in
    # prefill, so this is a known-good anchor.
    lean_events = load_lean_trace(lean_trace_path)
    if lean_events and kernels:
        lean_min_ns = min(e["ts_ns"] for e in lean_events)
        nsys_min_ns = kernels[0]["ts_ns"]
        lean_offset = nsys_min_ns - lean_min_ns
        for e in lean_events:
            e["ts_ns"] = e["ts_ns"] + lean_offset
    if lean_events:
        max_depth = max(e["depth"] for e in lean_events)
        for d in range(max_depth + 1):
            events.append({
                "name": "thread_name", "ph": "M",
                "pid": pid, "tid": lean_tid_base + d,
                "args": {"name": f"Lean withSection (depth {d})"},
            })
        for ev in lean_events:
            events.append({
                "name": ev["name"],
                "ph": "X",
                "pid": pid, "tid": lean_tid_base + ev["depth"],
                "ts": to_us(ev["ts_ns"]),
                "dur": ev["dur_ns"] / 1000.0,
            })

    # GPU-busy counter (50 ms sliding window)
    for ts, busy_pct, cnt in gpu_busy_metric(kernels):
        events.append({
            "name": "GPU busy %",
            "ph": "C",
            "pid": pid, "tid": counter_tid,
            "ts": to_us(ts),
            "args": {"busy": round(busy_pct, 2)},
        })

    # Per-phase summary
    summary = {}
    for label, start, end in phases:
        ph_kerns = [k for k in kernels if start <= k["ts_ns"] < end]
        ph_apis = [a for a in apis if start <= a["ts_ns"] < end]
        ph_mems = [m for m in memcpys if start <= m["ts_ns"] < end]
        kernel_ns = sum(k["dur_ns"] for k in ph_kerns)
        wall_ns = end - start
        api_ns = sum(a["dur_ns"] for a in ph_apis)
        summary[label] = {
            "wall_ms": wall_ns / 1e6,
            "kernel_ms": kernel_ns / 1e6,
            "kernel_calls": len(ph_kerns),
            "gpu_busy_pct": (kernel_ns / wall_ns * 100.0) if wall_ns else 0.0,
            "api_calls": len(ph_apis),
            "api_ms": api_ns / 1e6,
            "memcpy_count": len(ph_mems),
            "memcpy_bytes": sum(m["bytes"] for m in ph_mems),
        }

    conn.close()
    return events, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+",
                        help="Pairs of <sqlite> <label>")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--lean-trace",
                        action="append", default=[],
                        help="Lean withSection JSON for one input "
                             "(format: <label>=<path>; may repeat)")
    args = parser.parse_args()
    if len(args.inputs) % 2:
        sys.exit("inputs must be pairs of <sqlite> <label>")
    pairs = list(zip(args.inputs[0::2], args.inputs[1::2]))
    lean_paths: dict[str, str] = {}
    for spec in args.lean_trace:
        if "=" not in spec:
            sys.exit(f"--lean-trace expects <label>=<path>, got {spec!r}")
        label, path = spec.split("=", 1)
        lean_paths[label] = path

    # Find the smallest start across all sqlites so timestamps line up
    origins = []
    for path, _ in pairs:
        c = sqlite3.connect(path)
        row = c.execute(
            "SELECT MIN(start) FROM ("
            "  SELECT MIN(start) start FROM CUPTI_ACTIVITY_KIND_RUNTIME"
            "  UNION ALL"
            "  SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_KERNEL"
            ")").fetchone()
        origins.append(row[0] or 0)
        c.close()
    t_origin = min(origins)

    all_events: list[dict] = []
    summaries = {}
    for path, label in pairs:
        evs, summ = emit(label, path, t_origin,
                         lean_trace_path=lean_paths.get(label))
        all_events.extend(evs)
        summaries[label] = summ

    out = {
        "traceEvents": all_events,
        "displayTimeUnit": "ms",
        "otherData": {"phaseSummary": summaries},
    }
    with open(args.output, "w") as f:
        json.dump(out, f)
    print(f"wrote {args.output} ({os.path.getsize(args.output)/1024/1024:.1f} MiB, {len(all_events)} events)")
    print()
    print("=== phase summary ===")
    for label, ph in summaries.items():
        print(f"-- {label} --")
        for ph_name, m in ph.items():
            print(f"  {ph_name:8s}  wall={m['wall_ms']:7.1f} ms  "
                  f"kern={m['kernel_ms']:7.1f} ms  "
                  f"calls={m['kernel_calls']:5d}  "
                  f"GPU_busy={m['gpu_busy_pct']:5.1f}%  "
                  f"API={m['api_calls']:5d}  "
                  f"memcpy={m['memcpy_count']:5d}")


if __name__ == "__main__":
    main()
