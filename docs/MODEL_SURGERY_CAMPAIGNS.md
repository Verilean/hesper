# Model-surgery campaigns: closed-form approximation of a frozen transformer

Two campaigns (evidence: [`Verilean/e2b-jamba`](https://github.com/Verilean/e2b-jamba),
[`Verilean/e4b-webgpu`](https://github.com/Verilean/e4b-webgpu) `DEVPLAN-KV.md`)
applied the engine-campaign discipline — pre-registered predictions, golden
gates, ΔKL judging, honest negative results — to model-behavior work on
Gemma-4. They share only the model with the kernel campaigns of the decode
report; their claims are different in kind, so they live here. This document
also records the mathematics, the reason the first campaign HAD to fail the
way it did, and where the same machinery applies with better odds
(quantization, layer compression — §4).

---

## 1. Campaign 4: zero-training Mamba/Jamba-ization of E2B

**Question.** How far can attention→SSM conversion go with ONLY closed-form
algebra (no backprop), starting from an external theory sketch that claimed
"Taylor-Calibrate init ≈ 95% + a cycle-consistency projection absorbs the
rest"? 19 pre-registered predictions; 5 hit, 14 missed with diagnoses.

### 1.1 The substituted block

Original attention block, per head $`h`$ (Gemma-4: RoPE, QK-RMSNorm, GQA):

```math
y_t = W_O\,\mathrm{concat}_h\Big(\sum_s A^{(h)}_{ts} v^{(h)}_s\Big),\qquad
A^{(h)}_{ts} = \mathrm{softmax}_s\big(q^{(h)}_t \cdot k^{(h)}_s/\sqrt{d}\big)
```

with $`q = \mathrm{RoPE}(\mathrm{RMS}_q(W_Q x))`$, $`k, v`$ likewise. The V0
substitute reuses **all four projections** and replaces only the mixing:

```math
\hat y_t = W_O\,\mathrm{concat}_h\Big(\tfrac{\sum_s w_{ts}\,v_s}{\sum_s w_{ts}}\Big),
\qquad w_{ts} = \phi(q_t) \cdot \phi(k_s)\;\gamma_h^{t-s},\qquad
\phi(u)=\mathrm{elu}(u/\sqrt{d})+1
```

which is exactly a diagonal SSM in recurrent form:

```math
S_t = \gamma_h S_{t-1} + \phi(k_t)v_t^{\top},\quad
z_t = \gamma_h z_{t-1} + \phi(k_t),\quad
\hat y_t = \frac{\phi(q_t)^{\top}S_t}{\phi(q_t)^{\top}z_t}.
```

Parameter accounting (measured, layer 5): the block REUSES the original's
~7 M (int4) projections and ADDS ~7.2 M fp32 — capacity was never the
deficit; state size (3 banks × $`d_h^2`$ ≈ 197 K floats/head) is comparable to
the attention window memory (262 K floats/kv-head).

### 1.2 The closed-form toolchain

**Decay init** ("Taylor-Calibrate", done right): from teacher attention maps,
fit per-head $`\gamma_h`$ by log-linear regression of mean attention mass vs
distance; later, per-position selective gates
$`\gamma_t = \sigma(w_g \cdot x_t + b)`$ initialized by ridge regression of
$`\mathrm{logit}(\gamma^*_t)`$ on $`x_t`$, where $`\gamma^*_t = \bar d_t/(1+\bar d_t)`$
and $`\bar d_t = \sum_s A_{ts}(t-s)`$ is the observed mean attended distance.

**Output correction** ("cycle consistency", demystified): ridge regression

```math
W^{*} = \arg\min_W \|X_{\mathrm{out}} - Y W\|_F^2 + \lambda\|W\|_F^2
        = (Y^{\top}Y + \lambda I)^{-1} Y^{\top} X_{\mathrm{out}}
```

on calibration activations; with a $`K`$-channel decay bank the source is the
concatenation $`Y\in\mathbb{R}^{n\times Kd}`$ (capacity ladder, closed form).

**Cascade** (the multi-layer fix): convert layers in ascending single-layer
ΔKL order; fit $`W_i`$ on inputs the CURRENT hybrid actually produces, targets
stay the teacher's clean outputs — each correction is a translator AND a
repair operator for accumulated drift.

**Judge**: everything is selected and scored by
$`\Delta\mathrm{KL} = \mathbb{E}_t\,\mathrm{KL}\big(p_{\mathrm{teacher}}(\cdot|x_{\le t})\,\|\,p_{\mathrm{hybrid}}\big)`$
on held-out text — local MSE was measured to be nearly uncorrelated with
downstream damage (the single most reusable methodological result).

### 1.3 Measured ladder and the wall

| step | 6-layer hybrid | note |
|---|---|---|
| init only | broken text | "95%" falsified: cos ≈ 0.63 |
| + per-layer $`W`$ | 3 layers FREE (ppl ×1.01) | 6+ layers degrade |
| + cascade | ppl ×1.43 | compounding recovered (~30%) |
| + decay bank ($`K{=}3`$) | agreement revives | capacity moves what fitting cannot |
| + selective gate (closed form) | ΔKL 1.48 (best algebra) | −9%, not a breakthrough |
| + light distillation (inserted parts, STE through QAT) | ppl ×1.36 | +5-15%, saturates |
| ALL 13 convertible layers | collapse (ppl ×34) | additive damage, dominated by layer 0 |
| + end-to-end distillation from the collapsed init | worse (×80) | gradients don't descend from collapse |

**Why it had to fail (the owner's diagnosis, confirmed):** the conversion is
mechanical but the two operators are different function classes. Softmax
attention performs *content-addressed, winner-take-most retrieval*: $`A_{ts}`$
can concentrate on one arbitrary position regardless of distance, with
per-token concentration (measured: layer 0 outputs match in direction,
cos 0.77, while magnitudes are wildly off — the row-normalized linear kernel
flattens per-token concentration). The decayed kernel $`w_{ts}`$ factorizes
through a fixed feature inner product times a decay — a *superposition
memory* that cannot express sharp addressing. Every closed-form lever bought
5-15% and composed sub-multiplicatively; the residual is exactly where the
literature (LoLCATs, MOHAWK, Mamba-in-the-Llama) switches to distillation.
Structural constraints recorded: E2B's KV sharing caps conversion at 13/35
layers (2 KV-source layers untouchable); QAT checkpoints kill gradients at
every projection (`round()` — STE wrappers mandatory).

---

## 2. Campaign 5: KV-cache compression (the map pays off)

Campaign 4 said the enemy is retrieval precision → attack the memory, not
the math: importance-scored cache eviction (SnapKV/TOVA/H2O class — absent
from llama.cpp, which ships positional context-shift, cache quantization and
iSWA only). ~80% replication, ~20% new; the new part:

**Mechanism finding.** SnapKV scores keys by the attention of the prompt's
last window. On Gemma-4 this fails at depth: the needle fact ranks 663/4096
at the shallow full layer but **2593/4096 at the deep one** — while scored
by the FIRST GENERATED token's queries it ranks **0-23 at every layer**.
Retrieval-shaped attention appears at generation time, not while reading;
the fix costs one probe token. Python grid (E2B): generation-query scoring
retrieves 9/9 needles at 12.5% budget; positional eviction keeps 3/9.

**Engine port** (A4B WebGPU): sliding layers → ring caches (window+slack,
with per-slot position recovery — the slack matters: an exact-window ring
lets a prefill chunk clobber entries its own earlier tokens need, a real bug
found by the full-cache control); full layers → capacity-capped caches with
in-flight attention-mass scoring and periodic GPU compaction; an
online-softmax attention kernel (running max/sum/acc + log-sum-exp merge)
removes the workgroup-memory context cap. Result: **MAXSEQ 640 → 8192, fixed
decode memory/cost; needle parity with the full cache at 2k/4k/8k (8k budget
= 7.8% of context); budget decode FASTER than full — −14% at 2k, −43% at 8k**
(12.48 vs 21.82 ms/token).

---

## 3. What transfers between the campaigns

The reusable machinery is model-agnostic: (a) capture per-layer teacher IO on
calibration text; (b) closed-form per-layer fits; (c) cascade = fit each layer
on the degraded inputs the modified prefix actually produces; (d) judge by
ΔKL, never by local MSE; (e) hybrid fallback for layers that don't survive;
(f) full-teacher CONTROL runs beside every measurement.

## 4. Where this machinery has better odds (owner's proposal, assessed)

Mamba-ization failed because it crossed function classes. Two targets stay
INSIDE the function class, which changes the odds fundamentally:

**4.1 Post-training BitNet/1.58-bit-ization of Gemma-4.** Quantizing
$`W \to \alpha\,\mathrm{tern}(W)`$ (per-channel scales, weights in
$`\{-1,0,1\}`$) is a *perturbation* of the same linear map, not a substitution.
The toolchain maps one-to-one: closed-form optimal scales per channel;
Hessian-aware rounding on calibration activations (GPTQ's
$`\arg\min_{\hat W}\|X\hat W - XW\|^2`$ is exactly our LS machinery — and our
cascade IS GPTQ's sequential-quantization scheme, re-derived independently);
per-layer ΔKL table → mixed-precision hybrid (sensitive layers stay 4-bit,
the Jamba fallback pattern). Honest prior: pure 1-bit PTQ without any
training is known-hard (BiLLM/OneBit territory); 1.58-bit with per-layer
mixed precision and a light distillation of scales only is a plausible
campaign with real deliverables (memory ÷2.5 vs q4_0 → decode speedup on the
BW-bound engine).

**4.2 Layer compression (prune/merge).** Replace layer $`i`$ (or a pair) by
identity + closed-form $`W`$; judge by ΔKL; cascade the survivors. Same
harness, function class preserved (deep-layer redundancy is well documented).
Both proposals inherit the Campaign-4 lesson intact: judge by KL, cascade the
fits, keep a hybrid escape hatch, and run the full-precision control.
