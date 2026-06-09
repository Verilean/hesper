# 48 — RMSNorm 2.99× investigation: call count, not per-call

*Written 2026-04-24, task #219 investigation.*

## Headline

The "RMSNorm 2.99× slower than llama.cpp" figure from kernel_compare
decomposes into:

- **per-call time**: hesper 3.08 µs, llama 2.41 µs (**1.28×**)
- **call count**: hesper 704 (20 decode = 70/decode×10), llama 302 (=15/decode) (**2.33×**)

Per-call is almost parity.  The real gap is **call count** — hesper
emits ~70 RMSNorm invocations per decode, llama.cpp emits ~15.

## Counting where calls come from

hesper classifier counts all `block=256` kernels as RMSNorm.  That
bucket holds both real RMSNorm kernels and pointwise-style stubs with
the same block-size default.  Split:

| kernel hash | ms/dec | inst/dec | gx | notes |
|---|---:|---:|---:|---|
| k_1154709937980003 | 1.07 | 190 | 1 | true RMSNorm (verified via PTX: shared_sum reduce + rsqrt + scale) |
| k_3461724857804993 | 0.38 | 0.9 | 42 | per-layer batched, 42 layers in 1 launch |
| k_3588562057251185 | 0.34 | 35 | 8 | pointwise/stub |
| ... (146 more, each <0.15 ms) | ~0.4 | - | varied | pointwise scale/mul/stubs |
| **block=256 total** | **2.75** | **149 kernels** | | |

So:
- `k_1154709937980003` (gx=1) is the per-row RMSNorm — **190 calls/dec**
  at 5.6 µs/call = 1.07 ms/dec
- The remaining 1.7 ms is spread across ~148 pointwise/stub kernels
  (embedScale, pleScale, residual add, etc.) with 2-8 ms each in
  total, misclassified as "RMSNorm" by the crude block-size filter.

## What "hesper RMSNorm = 2.17 ms" actually measures

kernel_compare's "RMSNorm" row sums all `block=256` kernels.  llama.cpp's
`rms_norm_f32` is strictly RMSNorm only.  So the 2.17 is inflated by
~1.1 ms of unrelated pointwise ops on hesper's side.

**True RMSNorm vs llama.cpp:**
- hesper: 1.07 ms/dec (190 calls @ 5.6 µs)
- llama.cpp: 0.72 ms/dec (42+ calls @ ~13 µs, wider kernels that do
  more work per launch via graph-batching)

Ratio is closer to **1.5×**, not 2.99×.  Reducing the 190 calls is
still the right direction, just a smaller potential win.

## Why 190 calls

The Gemma 4 `LlamaPath` prefill/decode skeleton (
`Hesper/Models/Gemma4/LlamaForwardPrefill.lean`) calls
`RMSNorm.forward` **5 times per layer**:

```
attnNorm, postAttnNorm, ffnNorm, postFFNNorm, plePostNorm
```

42 layers × 5 = **210 RMSNorm calls per decode**, close to the
measured 190 (PLE skipped on some layers).

llama.cpp does roughly the same op count per layer but its graph-based
dispatcher coalesces adjacent ops into ~15 kernel launches via
FlashAttention-style fusion.

## Initially suspected skeleton-only — but it's not

First draft of this doc claimed the 190 call count was specific to
the `LlamaPath` skeleton.  On reflection that's wrong.  The skeleton
is *not* in "stub" state anymore — each of its 19 per-layer ops has
been replaced with a real kernel, and the total dispatch count is
**1492/decode** (confirmed via `HESPER_DISPATCH_COUNT=1`), matching
production hesper's historical call count.

So the 190 RMSNorm calls is *not* skeleton-specific — hesper as a
whole issues 5 RMSNorm invocations per layer (attnNorm, postAttnNorm,
ffnNorm, postFFNNorm, plePostNorm) and loops over 42 layers.
Production `Gemma4.lean forwardBlock` has the same pattern
(see its many `RMSNorm.forward` + `RMSNorm.forwardNormThenAdd` call
sites).

The earlier "LlamaPath v2 Phase 0 skeleton = ~200 dispatches/token"
number was measured against **stub** kernels that did nothing, when
the task was dispatch-count structural matching (task #200).  That
number doesn't apply now that the stubs are real kernels.

## Corrected via GGML_CUDA_DISABLE_FUSION=1 experiment

Ran llama.cpp with fusion disabled.  Result flips the story:

| variant | ms/decode | calls/decode | µs/call |
|---|---:|---:|---:|
| hesper | 1.07 | **190** | 5.6 |
| llama.cpp fusion ON  | 0.73 | 302 | 2.4 |
| llama.cpp fusion OFF | 0.48 | 302 | 1.6 |

Two big corrections:

1. **llama.cpp calls RMSNorm MORE than hesper** (302 vs 190/decode).
   The previous "302" number in kernel_compare wasn't per-decode —
   it was total across 20 decodes (6040/20=302/dec).  My initial
   take that "llama.cpp fuses RMSNorm call count down" was wrong.

2. **The real gap is per-call speed**, not call count.  hesper is
   2.3–3.5× slower per call.  With fusion OFF in llama.cpp (the
   cleaner apples-to-apples comparison), per-call ratio is
   5.6 / 1.6 = **3.5×**.

## What `GGML_CUDA_DISABLE_FUSION=1` does

Not what I guessed.  Fusion here is **kernel-body fusion**: the
rms_norm_f32 kernel's epilogue can absorb an adjacent mul / scale /
add into the same kernel (1 HBM pass instead of 2-3).  Same call
count, more work per call, still a net win: 0.48 → 0.73 ms when
fusion absorbs downstream work (kernel is slower per call but the
next kernel goes away entirely, which shows up elsewhere in the
totals not in rms_norm line).

**Actually it's the opposite direction from what I said**: fusion
makes rms_norm slower per call (absorbs work) but removes N
separate pointwise kernels from the trace.  In our "graphs OFF +
fusion OFF" column, we count the raw rms_norm work alone.

## Real action items

1. **Port llama.cpp's rms_norm fusion** — absorb the adjacent mul
   (by gamma) and add (residual) into the rms_norm kernel.
   hesper has `rmsNormThenAddKernel` already (`Linear.lean:216`).
   Find where dispatcher calls the plain `rmsNormKernel` followed
   by a separate residual-add and switch to the fused variant.

2. **Speed up hesper's rms_norm per-call**.  1.6 µs/call on a 2560-d
   row is very fast — llama.cpp is probably hitting close to L1
   bandwidth bound.  hesper at 5.6 µs/call is 3.5× slower, so
   check ncu (once the permission issue is resolved) for SoL /
   occupancy on `k_1154709937980003`.

Combined potential: 1.07 → ~0.5 ms/dec (from per-call fix alone,
call count left at 190).

## Files touched

- `scripts/kernel_compare.py` — added `Q6_K matmul ffn_down 1row`
  classifier case so the new 1-row Q6_K kernel doesn't get
  misclassified as Q4_K.
- `docs/llama-fusion-analysis/48-rmsnorm-investigation.md` — this doc.
