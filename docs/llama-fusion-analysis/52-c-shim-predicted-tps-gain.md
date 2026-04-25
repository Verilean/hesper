# 52 — Predicted TPS gain from C shim (modeled from gap distribution)

*Written 2026-04-24. Before implementing Option B.*

## Goal

Before committing to the ~1-day labor of the C shim, use the gap
distribution from doc 51 to predict how much TPS it should yield. If
the prediction is weak (< 20 TPS gain), reconsider. If strong
(30+ TPS gain), proceed with confidence.

## Input data

From `/tmp/hesper_gaps.txt` and `/tmp/llama_gaps.txt` (nsys inter-API
host gaps, 2026-04-24):

| band | hesper (n, total ms) | llama.cpp (n, total ms) |
|---|---:|---:|
| <500 ns | 11369 / 1.3 | 28699 / 9.9 |
| 0.5-2 µs | 1773 / 1.9 | 13126 / 10.1 |
| 2-5 µs | 315 / 1.1 | 278 / 0.8 |
| **5-12 µs** | **1533 / 14.0** | **24 / 0.2** |
| **12-50 µs** | **822 / 19.9** | **9 / 0.2** |
| **50-500 µs** | **711 / 87.3** | **43 / 10.2** |
| >500 µs | 139 / 5599 | 22 / 1103 |

Bolded rows are the tail bands where hesper wastes 121 ms total that
llama.cpp doesn't spend. Most of the 121 ms falls inside prefill +
decode; the >500 µs band is dominated by model load (mmap / parse).

## Modeled savings (wall-clock)

Treat the C shim as a mechanism that replaces a Lean-side inter-API
event with a C function invocation. The C side cannot trigger Lean
GC / refcount churn / Array realloc during the launch sequence, so
its gap distribution should look like `cuda-launch-bench`'s
microbench: a uniform 1-2 µs floor.

**Conservative scenario**: all gaps ≥ 5 µs get replaced by 2 µs.
Preserves current small-gap distribution (the bottom 3 bands), only
collapses the tail.

- 5-12 µs band saving: 14.0 − (1533 × 2 µs) = 10.9 ms
- 12-50 µs band saving: 19.9 − (822 × 2 µs) = 18.3 ms
- 50-500 µs band saving: 87.3 − (711 × 2 µs) = 85.9 ms
- Total: ~**115 ms saved across the full run**
- Of that, decode-relevant (~1/3): **~30-40 ms/decode-session**
- Over 11 decode tokens: **2.7-3.6 ms/token**
- Current: 15.0 ms/token → predicted: **11-12 ms/token = 83-91 TPS**

**Optimistic scenario**: llama.cpp-like distribution (tail ≈ 0).

- Full 121 ms tail removed (minus 10 ms for llama.cpp tail residual)
- Decode share ~1/3 = 35-45 ms
- Over 11 tokens: **3.2-4.1 ms/token**
- Current 15.0 → predicted **11 ms/token = 90-95 TPS**

**Ceiling**: llama.cpp graphs-OFF TPS at 107. hesper already has
equal per-kernel GPU time, so if *all* tail is eliminated, hesper
reaches the same 107 ± few TPS.

## Summary prediction

| scenario | predicted hesper TPS (graphs OFF) |
|---|---:|
| current baseline | 60 |
| C shim, conservative (tail≥5µs → 2µs) | **83-91** |
| C shim, optimistic (hesper matches llama.cpp tail) | **90-95** |
| theoretical ceiling (GPU-bound) | ~107 (llama.cpp) |

**Expected delivered range: 85-95 TPS (+40-60%).**

## Confidence / caveats

- The model assumes tail gaps are Lean-caused and the C shim eliminates
  them entirely. In practice:
  - Some 5-12 µs gaps may be CUDA driver-side and unavoidable.
  - GC-caused stalls that happen *before* entering the shim still hit
    the gap (we can't move them).
  - Shim scope must cover cacheRef lookup + PendingLaunch path +
    launchKernelMaybeStream; if we leave cacheRef probe on Lean side
    the mid-tail gaps stay.
- The prediction does not account for any new overhead introduced by
  the shim (Lean→C argument marshaling). The per-launch microbench
  already measures 1.12 µs for the C side, so 1-2 µs of marshaling is
  included in the conservative floor.
- Decode-share (1/3 of the run) is an estimate. Could be 1/4 to 1/2.

## Decision

The prediction is **strong enough to justify implementing Option B**.
Worst-case conservative gain is +25-30 TPS, best-case +35-50 TPS.
Either outcome is larger than any single change landed this session
(preHash fix was +17 TPS).

Proceed to shim design and implementation next session.

## Artifact

Raw band analysis:
```
awk '
  $1 < 500   { b0+=$1; n0++; next }
  $1 < 2000  { b1+=$1; n1++; next }
  $1 < 5000  { b2+=$1; n2++; next }
  $1 < 12000 { b3+=$1; n3++; next }
  $1 < 50000 { b4+=$1; n4++; next }
  $1 < 500000 { b5+=$1; n5++; next }
  { b6+=$1; n6++ }
  END { print "..." }
' /tmp/hesper_gaps.txt
```
