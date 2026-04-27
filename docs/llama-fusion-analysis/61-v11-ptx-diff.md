# 61 — V11 vs llama.cpp PTX diff (D=256, ncols=1, sm_89)

Date: 2026-04-27. Target: V11 26 µs → 6 µs (4.3× lever needed).

## Setup

- V11 PTX: dump via `HESPER_PTX_DUMP=/tmp/v11_ptx lake exe cuda-flashattn-vec-parity`.
  V11 partial = `v11p_raw.ptx`, V9 partial = `v9p_raw.ptx`, combine = `comb_raw.ptx`.
- llama.cpp PTX: `/tmp/llamacpp_ptx/fattn_vec_f16f16.ptx` (multi-instantiation).
  D=256 ncols=1 softcap=0 entry extracted via awk to
  `/tmp/llamacpp_ptx/llama_d256_ncols1.ptx` (6631 lines).

## Headline numbers (cacheLen=128, 200 iters, RTX 4070 Ti)

| kernel | gpu_time | PTX lines | total instrs (approx) |
|---|---|---|---|
| llama.cpp PTX | 4.42 µs | 6631 | ~2400 |
| V11 partial | 24.3 µs | 12630 | ~10000 |
| V9 partial | 33.1 µs | 1180 | ~1000 |

V11 is 5.5× slower than llama despite generating ~4× more instructions.

## Instruction histogram diff

| op | V11 | llama | ratio |
|---|---|---|---|
| **fma.rn.ftz.f32** | **0** | **778** | ∞ (V11 missing entirely) |
| mul.f32 + add.f32 | 1099 + 1058 = 2157 | 135 + 122 = 257 | 8.4× |
| add.u32 | 2721 | 0 | ∞ (V11 has all addr arith in u32) |
| shl.b32 | 1602 | 18 | 89× |
| mov.f32 | 1894 | 40 | 47× |
| **ld.global.nc.v4.u32** | **0** | **104** | V11 has 0 vec4 loads |
| ld.global.u32 / equiv | 256 | ~104×4 = ~416 | V11 uses scalar |
| ld.shared.f32 | 532 | 58 | 9.2× |
| cvt.f32.f16 | 512 | 768 | 0.7× |
| shfl.sync.bfly.b32 | 31 | 65 | 0.5× |
| bar.sync | 3 | 3 | same |

## Three primary findings

### Finding 1: zero fma fusion (highest ROI)

V11 emits `mul.f32 a b, t1; add.f32 t1 c → r1` as 2 instructions where llama emits
1 `fma.rn.ftz.f32 r1, a, b, c`. **778 fmas vs 0 fmas** → 778 instructions worth
of redundant work in V11's hottest paths.

The hesper PTX backend HAS `fma_f32` (`Hesper/CUDA/PTX.lean:175`), and `Exp.fma`
exists (`CodeGen.lean:621`). The V11 ShaderM source is using `*` and `+` operators
that lower to separate `mul.f32` + `add.f32` instead of constructing `Exp.fma`.

Two routes to fix:
- A. **PTX-level peephole**: `add.f32 (mul.f32 a b) c → fma.rn.f32 a b c`. Catches
  every site, no source rewrite needed. Risk: changes rounding (mul-then-add vs
  fma) but acceptable for FlashAttn (already uses `.ftz.f32` in llama).
- B. **ShaderM-level**: rewrite the V11 inner loop to use `Exp.fma` explicitly.
  Less general but no PTX-level pattern matching.

A is preferable. Target removal: ~778 mul.f32 + 778 add.f32 = 1556 instructions
folded into 778 fmas. Estimated speedup at memory-throughput-bound geometry:
modest (~10-20%) but at compute-bound (which V11 is): potentially 1.5-2×.

Note: ptxas may already do this fusion at JIT time. Need to confirm by reading
SASS (`cuobjdump --dump-sass`). If ptxas already fuses, this lever is closed.

### Finding 2: vec4 load not used

llama uses `ld.global.nc.v4.u32` 104 times (= 416 scalar loads' worth of bandwidth)
per kernel. V11 emits zero vec4 loads — every K element is a scalar `ld.global.u32`.

Hesper has `ld_v4_u32` infrastructure landed in V10 (commit per project_v9_perwarp_gap).
**It just isn't wired into V11**. Action: rewrite V11's K/V load path to use the
v4 primitive — same change V10 did for V9's inner-K rewrite.

Direct gain: 4× fewer global memory transactions, better L1 hit rate, fewer
address computations. Risk: requires K/V buffer alignment to 16 bytes (which f16
naturally satisfies for D=256 since each row is 512 bytes).

### Finding 3: address arithmetic explosion

V11 emits **2721 add.u32 + 1602 shl.b32 = 4323** address-related ALU instructions
vs llama's **18 shl.b32 + 0 add.u32**. This is ~**240× more address arithmetic**.

Root cause: V11's loop is fully unrolled at the meta-level (Lean for-loop iteration
with `Nat`), so each iter materializes a fresh `index = base + i*stride` chain
in PTX. CSE pass would catch the common base-pointer term, but doesn't catch the
linear progression because each iteration's `i*stride` is a literal constant.

Two ways to fix:
- C. **MutPtr / pointer advancement**: instead of `addr = base + i*stride` per
  iter, emit `ptr = base; loop { ...; ptr += stride; }`. ShaderM already has
  `MutPtr` (Step 9b, task #282). V11 doesn't use it.
- D. **PTX peephole for arithmetic progressions**: fold `add.u32 r1, base, c1`
  into a base register that increments. More general but harder to write
  correctly.

Going with C (ShaderM MutPtr) is direct and addresses the structural issue.

## Plan

Three-step session plan in priority order. Each step includes parity check
(`lake exe cuda-flashattn-vec-parity`) and microbench gpu_time measurement.

1. **PTX peephole: mul + add → fma**. Add to `Hesper/CUDA/CodeGen.lean` post-pass
   on emitted Inst stream. Expected: V11 26 → ~16 µs (assuming ptxas wasn't
   already fusing; verify first via cuobjdump --dump-sass).

2. **Wire V10's `ld_v4_u32` into V11 K/V load path**. Edit
   `flashAttentionVecParamsKernelV11` in `Hesper/WGSL/FlashAttentionExperiments.lean`
   to use `ShaderM.readBuffer4` (or the equivalent wrapper). Expected: V11
   ~16 → ~10 µs.

3. **MutPtr in V11 inner K loop**. Replace `index = base + i*stride` pattern
   with pointer-advance pattern. Expected: V11 ~10 → ~7 µs.

Stretch: **Fully vec4 the V (V cache) load path** with f16x2 fma (`fma.rn.f16x2`
already supported, `PTX.lean:180`). Could close to 6 µs if it works.

## Status check before starting step 1

Run `cuobjdump --dump-sass` on V11 cubin and llama.cpp cubin to confirm ptxas
isn't already fusing mul+add into fma. If ptxas DOES fuse, finding 1 is moot
and we go straight to finding 2.

```bash
nvcc -arch=sm_89 -ptx -o /tmp/v11.ptx <V11 PTX>     # already have v11p_raw.ptx
ptxas -arch=sm_89 -o /tmp/v11.cubin /tmp/v11_ptx/v11p_raw.ptx
cuobjdump --dump-sass /tmp/v11.cubin > /tmp/v11.sass
grep -c "FFMA\|FADD\|FMUL" /tmp/v11.sass
```

If FFMA count >> 0, ptxas already fuses → focus on finding 2 first.

## SASS-level reality (run 2026-04-27)

```
ptxas -arch=sm_89 -O3 -o v11p_raw.cubin /tmp/v11_ptx/v11p_raw.ptx
cuobjdump --dump-sass /tmp/v11p_raw.cubin > v11p_raw.sass
```

V11 SASS top mnemonics:

| op | count | meaning |
|---|---|---|
| **FFMA** | **1029** | ptxas already fused mul+add → finding 1 closed |
| HADD2.F32 | 512 | f16x2 add (smem path) |
| IADD3 | 328 | 3-input int add (address) |
| **LDG.E** | **289** | **scalar global load — this is the V11 problem** |
| LDS.128 | 132 | 128-bit smem load (OK) |
| FMUL | 82 | residual unfused |
| FADD | 35 | residual unfused |
| SHFL.BFLY | 31 | warp shuffle |
| STG.E | 34 | global store |
| STS.128 | 8 | 128-bit smem store |
| SHF.L.U32 | 7 | shift |

(llama D=256 sass extraction failed: `.version` directive missing in awk-extracted
single-kernel ptx; need to dump from cubin or supply header. Skipping for now —
ggml's vec_f16f16 D=256 known to use `LDG.E.128` (128-bit) for K/V via the
`__half2` 4× unroll pattern in fattn-vec.cuh.)

## Revised plan (after SASS check)

1. ~~PTX peephole mul+add → fma~~ — **closed** (ptxas already fuses).

2. **(Top priority)** Wire 128-bit global load into V11. V11 issues 289 scalar
   `LDG.E` while ggml uses `LDG.E.128` for K/V tile fetches. ShaderM has the
   `ld_v4_u32` infra (V10). Action: rewrite K/V global-load sites in
   `flashAttentionVecParamsKernelV11` to use 4-wide reads. Expected: V11
   26 → ~12 µs (LDG bandwidth + half the per-element address arith).

3. **(Secondary)** Replace per-iter address materialization with MutPtr loop in
   V11 inner K. ShaderM Step 9b MutPtr already exists; V11 just doesn't use it.
   Targets the IADD3 328 + extra address arithmetic. Expected: V11 ~12 → ~8 µs.

4. **(Stretch)** Pack remaining FFMA chain via f16x2 FMA where the dim is paired.
   `fma.rn.f16x2` is already in PTX backend (PTX.lean:180). Could give the
   final push to ~6 µs.

The 26 → 6 µs target needs a cumulative ~4.3× from steps 2+3+4. Step 2 alone
should validate the direction; if step 2 doesn't yield ≥1.6× we need to
reassess (likely something else is the real bottleneck — re-profile with ncu
to localize).
