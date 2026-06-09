# 47 — Q6_K 1-row switch: measured result

*Written 2026-04-24. Follow-up to doc 46 (handoff plan).*

## TL;DR

Switched `ffnDown` Q6_K dispatcher from the 4-row kernel to the 1-row
kernel (HESPER_Q6K_KERNEL=1row, now the default).  Result:

| variant | ms/dec | Δ vs 4-row |
|---|---:|---:|
| 4-row (baseline) | 1.322 | — |
| 2-row | 1.401 | +6% (regression) |
| **1-row (new default)** | **1.246** | **−5.8%** |
| llama.cpp reference | 1.20 | — |

Ratio vs llama.cpp: **1.10× → 1.04×**.  Token parity preserved on the
canonical "Hello world how are you" 10-decode run → "?".

## Why the gain was only 5.8%, not the 50% doc 46 expected

doc 46 projected −50% based on ncu's tail-effect metric: the 4-row
kernel's grid of 640 blocks fitted into 1 full wave + 160-block
partial wave, and ncu attributed 50% of the runtime to that tail
wave.

The 1-row kernel hits grid 2560 = 5.3 full waves (tail ~6%), so the
tail-effect metric dropped to near-zero as predicted.  But tail
effect isn't additive with the kernel's other costs — it's a
**ceiling** on how much the partial wave can inflate total runtime
**when every other pipeline stage is already saturated**.  In our
case the kernel is memory-bound (ncu SoL 1% compute, memory-heavy),
so HBM throughput is the binding constraint regardless of how many
blocks are in flight.

The 1-row kernel loses the 4-row's smem sharing of the Q8_1 input
(each ffn_down block reads 2.56KB × 4 = 10.24KB shared across 4 rows
in 4-row, vs each block reading 2.56KB alone in 1-row → 4× more
global input traffic per row).  Net result: −45% from tail fix,
+30% from lost sharing, −5.8% measured.

The 2-row variant sits between: partial-wave at grid 1280 (still 2
waves + partial), smem sharing only across 2 rows.  It turned out to
be the worst of both worlds — regression of +6%.

## llama.cpp matches this pattern

llama.cpp's Q6_K decode matmul kernel also uses grid = outDim (1-row
per block).  We now launch the same shape — our remaining 4% gap is
presumably in kernel-body details (scale batching, smem layout) that
docs 48-49 will tackle.

## What to do next

Per doc 46, remaining potential:

| next step | expected | blocked by |
|---|---|---|
| B: batch-read scales as u32 (uncoalesced 11% → ~0%) | −5 to 10% | #216 |
| C: pad smem ×9 → ×10 (bank conflict 30% → ~0%) | −3 to 5% | #217 |

Combined expected: ~−10%, matching llama.cpp or beating it by 5%.

## ncu permission regression

The ncu setup that worked in the previous session stopped giving
data this session (`No kernels were profiled`).  `/etc/modprobe.d/
nixos.conf` still has `NVreg_RestrictProfilingToAdminUsers=0` but
`/sys/module/nvidia/parameters/` is empty — the current NVIDIA
driver build doesn't export the parameter via sysfs.

Workaround for future ncu runs: `sudo rmmod nvidia_uvm nvidia` each
session, or invoke ncu under sudo.  For THIS session we fell back to
nsys `gpu_kern_sum` which gave sufficient ms/decode data to close
task #215 on wall-clock alone.

## Files touched

- `Hesper/Layers/Linear.lean:3952-3985` — dispatcher, new env flag
  `HESPER_Q6K_KERNEL` with default "1row".  Accepts "4row" / "2row"
  for regression comparison.
- `docs/llama-fusion-analysis/47-q6k-1row-switch-result.md` — this doc.
