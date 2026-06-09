# V11 parity debug — progress notes

## Status: ✅ RESOLVED (commit c5d0f0b)

**Root cause**: PTX CodeGen had no case for `Exp.subgroupShuffleXor`.
Default fallthrough returned an undefined u32 register; surrounding
`add`'s type-mismatch fallback silently dropped the shuffle. So
`acc + subgroupShuffleXor(acc, mask)` lowered to just `acc`.
warpReduceSum 8's 3-step butterfly became a no-op.

**Fix**: add `.subgroupShuffleXor` case in CUDA/CodeGen.lean (after
`.subgroupShuffle`) lowering to `shfl.sync.bfly.b32 dst, src, mask, 31, 0xFFFFFFFF`.
Mask is extracted from the `Exp.litU32 mask` operand (PTX bfly encodes
the offset as an instruction immediate, not a register).

**Verification**:
- vec parity: ALL CASES PASS, V11 max abs diff = 0.0006 (matches V7/V9
  noise level)
- shader-monad unit tests: 73/73 pass
- Hand-computed score for K=0: -1.282, V11 was reporting +0.665 before
  the fix, now matches.

The "12.2× systematic ratio" was lane 0's 32-dim partial dot product
vs the full 256-dim sum; ratio happened to be ~12.2 for this test
data, not anything fundamental.

V8 still fails parity at cacheLen ≥ 33 — separate slot collision bug
from the earlier V8 sub-warp partition attempt, untouched.

---

## Original status (pre-fix, kept for reference)

V11 fails parity at cacheLen ≥ 33 (max abs diff scales with cacheLen,
saturating ~0.45). Bug localized to **Phase 1+2 of V11**, before the
final reduction.

## Reproduction

```bash
lake exe cuda-flashattn-v11-debug
```

Direct A/B of V9 vs V11 partial_out and partial_meta on the same input
buffers. Bypasses the combine kernel (which is shared between V9 and
V11, so innocent).

## Key findings

### cacheLen=8 (passes parity, max diff 0.006)

V9 partial_meta per split:
```
split  V9_max     V9_sum
  0    -1.282     1.000
  1    -1.175     0.902
  ...
  7    -0.533     0.485
```

V11 partial_meta per split:
```
split  V11_max    V11_sum
  0    +0.665     1.000
  1    +0.674     1.000
  ...
  7    +0.726     1.000
```

V11's `sum=1.0` always at cacheLen=8 makes sense (1 K-pos per split,
softmax-normalized = 1). V9's varying sum is itself a separate
oddity — but it gets divided out by combine, so V9 still passes.

V11's **`max` is wrong**: V9=-1.28, V11=+0.66 for the same K-position
0. **Same sign discrepancy.**

partial_out diff at cacheLen=8: split 0 perfect (max diff 0.0). Splits
1-7 diverge by up to 0.6 in some dims. Total downstream parity passes
because combine recovers most of it.

### cacheLen=33 (fails parity, max diff 0.45)

Per-split max:
```
split  V9_max    V11_max   ratio (V9-V0)/(V11-V0)
  0    -0.972    +0.691    ?
  1    -0.544    +0.726    (0.428 / 0.035)
  ...
  7    +2.129    +0.945    
```

V9 max increments by ~0.428 per split. V11 max increments by ~0.035
per split. **Ratio ≈ 12.2× too small.** Same ratio at cacheLen=8
(0.107 vs 0.0088).

partial_out diff: split 0 max abs diff 1.09, growing then decreasing
with split index.

## What's been ruled out

- **Combine kernel**: same combine works for V9 (parity ✓), so isolated to V11 partials
- **Slot encoding in Phase 3 final reduction**: V11's `(warp*4+s)*D + L*32+2*pk` matches the read in final reduction
- **Buffer setup / order of kernel invocation**: swapping V9-then-V11 vs V11-then-V9 produces identical results
- **Q load**: V9 uses dim mapping `[2L, 2L+1, 2L+64, 2L+65, ...]`; V11 uses `[L*32, L*32+1, ..., L*32+31]`. Both internally consistent with K layout.

## Hypotheses for next session

1. **`warpReduceSum 8` produces wrong dot product**. If the 8-way butterfly
   is missing a step or the partition isn't aligned to sub-warp boundaries,
   the per-sub-warp score would be off.

2. **`scoreGated` value seen by `kqMaxNewVar` accumulation is per-sub-warp
   inconsistent**. Since the assign runs uniformly across all 32 lanes but
   `scoreGated` differs per sub-warp, maybe the ShaderM lowering doesn't
   correctly emit per-sub-warp values.

3. **f16 unpacking precision difference between contiguous-block (V11) and
   strided (V9) access patterns**. Algebraically same, but order of
   accumulation could matter for f16 → f32 → reduce-sum chain. Unlikely
   to give 12x error though.

4. **`partialVar` accumulation isn't per-sub-warp local**. If somehow the
   register is shared across sub-warps incorrectly (Lean meta-level
   register naming confusion), one sub-warp's partial would contaminate
   another's.

## Failed attempts this session

- **Try nthreadsKQ=32 in V11** to disable sub-warp partition: PTX gen
  produces `CUDA_ERROR_ILLEGAL_ADDRESS` at module load. Constants like
  `dPerLanePair=4` are not symmetric with the hardcoded `xor 8 / xor 16`
  cross-sub-warp shuffle paths in Phase 2a, breaking the kernel.
  → CAN'T use this as a clean A/B test.

## Next debug steps (for next session)

1. **Add a dedicated debug-output buffer to V11** that captures per-lane
   `partialVar` immediately after `warpReduceSum 8` in Phase 1's iter 0
   (or any specific iter). This requires:
   - A new ShaderM input buffer `debug_partial`
   - A write `debug_partial[warpId*32 + laneId] = partialVar`
   - Compare to V9's `partialVar` at the same iter.

2. **Reduce V11 to a minimal Phase 1 only kernel**: output the per-lane
   `kqRegVar` after Phase 1 (before Phase 2). Compare to V9's per-lane
   `kqRegVar`. If they match, bug is in Phase 2 or later. If they
   differ, bug is in Phase 1 (most likely the sub-warp partition).

3. **Hand-compute** the expected score for K=0 with the test data
   (q[d] = 0.1 + sin(d/64), k[d] = 0.05 + 0.013 + cos(d/53), scale =
   1/sqrt(256)). Compare with V9 (-1.28) and V11 (+0.66). Whichever
   value matches the hand-computed one is correct; the other is the
   bug.

4. **Inspect V11's PTX directly** for the dot product accumulation —
   maybe there's a CodeGen issue where `partialVar` accumulation
   is dropped in some path. Look at the `partialVar` assignment lowering.

## Empirical pattern observed

For both cacheLen=8 and cacheLen=33:
```
V9 max increment per split / V11 max increment per split = 12.2
```

This 12.2 ratio is suspicious. Possible interpretations:
- Effective dim count reduced by ~12.2× (full 256 dims → ~21 dims)?
- Score divided by ~12 somewhere?
- Score multiplied by something between 0 and 1 (e.g. position fraction)?

`12.2 ≈ 256 / 21`. `12.2 ≈ 4 * pi`. `12.2 ≈ sqrt(150)`. None obvious.
