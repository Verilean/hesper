# Perf Autotune Loop — semi-automatic, headroom-driven kernel optimization

A repeatable flow for closing the gap to llama.cpp (and beyond) on the DiffusionGemma forward — and any
WGSL-generated kernel. The core idea (the JAX/XLA bet): **we generate the kernels, so we know every
stage's shape/FLOP statically — measure the headroom automatically, attack the biggest, validate hard,
re-measure.** No more manual profile-and-guess (which misled us 3× in one session).

```
        ┌─────────────────────────────────────────────────────────────────┐
        │  1. MEASURE headroom (ROOFLINE=1)  →  ranked recoverable_time      │
        │  2. PICK the top row                                              │
        │  3. VARIANT-bench candidate kernels (harness, real shape)         │
        │  4. GOLDEN-validate the kernel (real dims + edge cases)           │
        │  5. INTEGRATE behind a flag                                       │
        │  6. VALIDATE integration (≥2 prompts, diff reg-vs-default)        │
        │  7. MEASURE the win (emb+fwd delta) — land or root-cause          │
        │  8. RE-MEASURE headroom  ───────────────────────────────────────┐ │
        └─────────────────────────────────────────────────────────────────┘ │
                                   ▲                                          │
                                   └──────────────────────────────────────────┘
```

---

## The loop, step by step

### 1. MEASURE headroom (seconds, no cold model load)
```bash
lake build matmul-bench
ROOFLINE=1 ./.lake/build/bin/matmul-bench
```
`forwardRoofline` (Examples/Compute/MatmulBench.lean) micro-benches every forward matmul stage at its real
DiffusionGemma shape, computes each stage's **achievable floor** from its FLOP, and ranks by
**recoverable_time = actual − achievable**. The top row is where the余力 (headroom) is.

- **achievable ceiling, per kernel class** (the realistic target, NOT the theoretical peak):
  - f16 reg-matmul: `0.70 × 15.5 TFLOP/s` (`achF16`)
  - quantized (in-kernel dequant): `0.27 × 15.5` = llama.cpp `mul_mat_id`'s measured 4.25 TFLOP/s (`achQuant`)
- MoE stages get a `0.52` sentinel-skip factor (decode skips ~48% padding tiles).

### 2. PICK the top recoverable row.

### 3. VARIANT-bench candidate kernels (clean, decode is too noisy)
Add a `benchXxx` to the harness (pattern: `benchKernelMs` + real-shape buffers + the kernel's config) that
times each candidate at the **real shape**. Existing comparators: `benchMMQ5GateUp` / `benchQ4kRegGateUp`
(dp4a vs matrix-reg gate/up), `benchWarpDown` / `benchQ8RegDown` (warp vs matrix-reg down). Pick the fastest
— but it is only a candidate until step 6 proves it CORRECT.

### 4. GOLDEN-validate the kernel — at REAL dims + edge cases
`checkFusedQ8Correct` etc. pass at small dims; the decode uses nE=128, large N/K. Extend (e.g.
`checkFusedQ8HighExpert`, `Q8HI=1`) to test **high experts (127), large K, sentinel tiles**. `maxDiff≈0`.

### 5. INTEGRATE behind an env flag (never touch the shipped default until step 7).

### 6. VALIDATE the integration — the discipline that catches what "Paris" hides
```bash
# the kernel under test, on ≥2 prompts:
DG_GROUP=1 DG_QKVRB=1 DG_<FLAG>=1 ./.lake/build/bin/diffusiongemma-decode <gguf> "The capital of France is" 30
DG_GROUP=1 DG_QKVRB=1 DG_<FLAG>=1 ./.lake/build/bin/diffusiongemma-decode <gguf> "The largest planet in our solar system is" 30
# the DEFAULT on the SAME second prompt — they MUST match (same math):
DG_GROUP=1 DG_QKVRB=1                ./.lake/build/bin/diffusiongemma-decode <gguf> "The largest planet in our solar system is" 30
```
**"Paris" alone is an INADEQUATE smoke check** — it passed a kernel that produced garbage on "planet".
Always diff the new path vs the default on ≥2 prompts. If they diverge, the path is wrong even if "Paris"
decodes cleanly.

When it diverges, isolate FAST (no cold-load where possible):
- kernel? → the golden (step 4). If golden passes, NOT the kernel.
- reg-specific? → run the alternative kernel (e.g. warp) in the same path. If it ALSO fails, it's the PATH.
- numerical vs race? → `DG_MOEDIAG` (compares the grouped output vs a per-slot reference in-situ at li==0,
  with a full sync). `maxDiff≈0` ⇒ it is a **race** (the sync masks it); `maxDiff>0` ⇒ a logic bug.

### 7. MEASURE the win + decide
Warm-step `emb+fwd` vs the default. **Land** (flip the default) only when faster AND correct on ≥2 prompts.
Otherwise root-cause (step 6) and either fix or record the dead-end.

### 8. RE-MEASURE headroom (step 1) — confirm recoverable dropped, find the next target. Loop.

---

## Tools

- **Harness** `Examples/Compute/MatmulBench.lean`: `ROOFLINE=1` (the loop), `Q8HI=1` (high-expert golden).
  `benchShape`, `benchKernelMs`, `benchMMQ5GateUp`, `benchWarpDown`, `benchQ8RegDown`, `benchQ4kRegGateUp`,
  `checkFusedQ8Correct`, `checkFusedQ8HighExpert`. `peakFlops=15.5e12` (measured simdgroup-matrix plateau).
- **In-situ numerical diag** `DG_MOEDIAG`: grouped vs per-slot reference (race-vs-logic verdict).
- **Per-component cost** `DG_SKIP_*` / `DG_PROF`: noisy (conflates kernel + surrounding ops + sync) — the
  harness is the clean source of truth; prefer it.

## Hard-won rules (do not relearn these)
1. **"Paris" alone is inadequate** — diff reg-vs-default on ≥2 prompts (step 6).
2. **The harness beats manual profiling** — `DG_SKIP`/`DG_PROF` conflate the kernel with surrounding ops;
   manual profiling gave the WRONG bottleneck 3× in one session. Trust the clean per-kernel bench.
3. **f16 precision is NOT a blocker** — the QKV/O/dense matrix-reg (f16) is correct (`DG_QKVRB` → "Jupiter").
4. **Dawn drops no-wait `flushBatch` splits at batch scale** → long producer→consumer chains
   (geglu→q80→down→scatter→wacc) RACE routing-dependently. `endBatch` (full barrier) fixes it but costs
   3–4 s/step — un-shippable. **A single FUSED kernel (no inter-pass flush) is the only cheap fix.**
5. **Quantized matmuls bottom out at ~0.27 peak** (llama.cpp `mul_mat_id`); don't chase the f16 0.70 floor
   for them.

## Current headroom snapshot (re-run `ROOFLINE=1` to refresh)
```
stage             recoverable   efficiency   status
MoE gate/up MMQ5    215 ms       46% llcpp    matrix-reg is 3.7× faster BUT grouped-path Dawn RACE (blocked)
MoE down warp       192 ms       32% llcpp    matrix-reg is 4.9× faster BUT same Dawn RACE (blocked)
attention QKV+O      40 ms       57% f16      reg ALREADY correct here — SAFE medium-M target
dense gate/up        17 ms       50% f16      reg correct — safe
dense down            8 ms       51% f16      currently warp/bmm — small, reg'able
```

## Backlog / fusion candidates (the high-leverage, JAX/XLA lane)
- **FUSED `down + scatter + wacc`** single kernel — gets the matrix-reg MoE speed (3.7–4.9×, ~400 ms
  recoverable) WITHOUT the racy inter-pass chain. The unblock for the two biggest rows.
- **FUSED `geglu + q80`** — removes element-wise passes (and a race link).
- **Autotuning seed** — generate kernel variants (tile/fragment/unroll as params) → bench all via the
  harness → cache the best per shape. This is the loop's step 3 made automatic.
- **medium-M reg** (attention/dense, 57%→70%) — fewer barriers / M=262 tail; safe, the reg is already correct.

## How to run the loop
Each pass: §1 measure → §2 pick → §3–7 attack the top row → §8 re-measure. One env-flagged kernel per pass;
the shipped default is never regressed. Record every dead-end (a failed variant is data — it narrows the
next pass). See `recipes/DIFFUSIONGEMMA_PERF_PLAN.md` + the project memory for the running log.
