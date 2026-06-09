# Phase F2 / F2.5 / F2.6 — Handoff Document

**Session end**: 2026-04-21
**Next-session start goal**: close the remaining 5.04 parity gap, then run F3 TPS benchmark.

## Where we are

| Phase | Status | max \|err\| | blocks |
|---|---|---|---|
| F1 — bundle plumbing | ✅ shipped | n/a | — |
| F2 — GPU exec skeleton | ✅ shipped | 12.67 | 4 |
| F2.5 — wO + PostAttnNormAdd | ✅ shipped | 8.09 | 6 |
| F2.6 — PLE-off reference | ✅ this session | **5.04** | 6 |
| F2.7 (next) — close 5.04 residual | ⏳ next session | 0 target | 6 |
| F3 (next) — TPS benchmark | ⏳ blocked on F2.7 | — | — |

Parity tracking: **12.67 → 8.09 → 5.04**.  Still 5.04 to close.

## How to reproduce the current number

```bash
HESPER_SKIP_OUTSCALE=1 lake exe gemma4-monolith-layer-parity
```

This disables `block.outScale` in `forwardBlock`'s fallback path so the
reference output matches a PLE-less + outScale-less pipeline.  Note:
`HESPER_SKIP_OUTSCALE=1` must be set as a **shell env var** — we don't
set it from Lean because `IO.setEnv` isn't exposed in this stdlib.

## Remaining 5.04 — suspected causes, in order of likelihood

### 1. wO matmul kernel-variant mismatch  (most likely top contributor)

**Production path**: `forwardBlock` line 765 calls
`Hesper.Circuit.CircuitM.matmulQ4K attnOut block.attention.wO` via
`runCached`.  Internally this goes through `Circuit.Lowering` which
may pick the **2-row DP4A kernel** (outDim ≤ 5120 ∧ even).

**Monolith path**: `runMonolithicGraph` on `GemmaAttnOutProj` calls
`Hesper.Layers.Linear.LinearLayer.forward ctx bundle.wO inBuf outBuf`.
This goes through `forwardDP4A` which ALSO picks the 2-row variant
under the same precondition — so they SHOULD hit the same kernel.
But hesh-key differences in `dp4aMatmulPrepared` between the two
callers could cause JIT to re-compile to a different PTX.

**Debug**: dump the WGSL for both paths; run with HESPER_KERNEL_TRACE=1;
compare the two call sites' cached kernel hashes.

### 2. attnResidualBuf aliasing

My driver wires `baseTensorId+5 → state.attnResidualBuf`.  Production's
`forwardBlock` uses `state.attnResidualBuf` as **BOTH** the
postAttnNorm output AND the FFN input.  Monolith IR does the same —
blocks 4 (PostAttnNormAdd) writes `attnResidId`, block 5 (FFN) reads
it.  No aliasing issue expected here, but worth checking that
Monolith's FFN doesn't overwrite it before FFN completes.

### 3. normedBuf reuse

My driver wires `baseTensorId+4 → state.normedBuf` as the wO output.
Production `forwardBlock` also writes wO output to `state.normedBuf`
(line 772 `matmulQ4K attnOut block.attention.wO` writes to outBuf[1]
= normedBuf).  Then `forwardNormThenAdd` at line 785 reads
`state.normedBuf` as `layer_out`.  My Monolith does the same thing.
Again, no issue expected.

### 4. layerOutScale still not skipped?

`HESPER_SKIP_OUTSCALE=1` is being set at shell level but I should
confirm the env var is actually being read by `forwardBlock`.
Let me check the exact code path once more:

- Line 1084: `let skipOutScaleFallback := (← IO.getEnv "HESPER_SKIP_OUTSCALE").isSome`
- Line 1086: `match if skipOutScaleFallback || pleRan then none else block.outScale with`

`pleRan = false` (we passed perLayerEmbd=none).  So `skipOutScaleFallback`
must be true, meaning `IO.getEnv "HESPER_SKIP_OUTSCALE"` must return
`some _`.  The shell export should handle that.  But to be 100% sure,
next session should **dump `refArr` against an Monolith-generated
output at each intermediate stage** (qBuf, kBuf, vBuf, attnOut, wOOut,
attnResid, ffnOut, final out) — whichever diverges first is the
smoking gun.

### 5. Numerical drift from slightly different op order

The Monolith dispatches FlashAttn with `cacheLen = maxSeqLen`
(approximation; see `runMonolithicGraph` FlashAttention case).
Production reads cacheLen from `paramsBuf` at runtime (`pos+1`).
At pos=0 these might differ — **maxSeqLen=8192 vs cacheLen=1** means
the Monolith is computing attention over much more (zeroed?) cache
than production.

**This is likely the #1 actual cause.**  Fix: either write the
correct cacheLen to paramsBuf (production does via
`writeScalarViaStaging`) and let `executeFlashAttentionTiled` read it
at runtime, OR pass `cacheLen = pos + 1` as a literal in the
Monolith dispatch.

## Concrete F2.7 plan (next session, ~30-60 min to close parity)

### Step 1: stage-by-stage intermediate dump

Add dump-and-diff after each Monolith stage:

```lean
-- After GemmaAttentionMonolith
let qAfterMono ← readF32BufMono ctx state.qBuf (numHeads * headDim)
-- ... compare with production's state.qBuf at the same stage
```

Production doesn't directly expose stage-by-stage buffers, so either:
(a) add `dumpBuf` calls to production `forwardBlock` at each stage
and save intermediates, OR
(b) split production into two calls (attention only, then FFN) and
compare after each.

### Step 2: fix FlashAttn cacheLen

Change `runMonolithicGraph` FlashAttention case:
```
bundle.maxSeqLen     -- OLD (wrong)
(pos + 1)            -- NEW (what production uses)
```

But wait — `pos` isn't in the bundle.  It's in the Monolith node
(`FlashAttention layerKey pos`).  The node already carries pos.
Just pass `pos + 1` to `executeFlashAttentionTiled`.

### Step 3: confirm HESPER_SKIP_OUTSCALE works

Run with and without the env var; confirm that the reference output
changes.  If not, investigate why.

### Step 4: if parity still off after 1+2+3

The remaining is either:
- wO kernel-variant PTX hash divergence (→ unify the cache keys)
- Some paramsBuf-related state not reset between runs
- A subtle bug in how I wire `state.buf2` vs production's expected
  input buffer

Instrument with sum-of-squares or mean at each stage to narrow.

### Success criterion

`max |err| < 1e-4` for layer 5 with HESPER_SKIP_OUTSCALE=1 and
perLayerEmbd=none.  Document the residual 5.04 → 0 progression.

## F3 (after parity): TPS benchmark plan

Once parity holds:

1. Write `Examples/DSL/Gemma4MonolithTPS.lean`.
2. Build the whole-token graph (42 layers × 6 Monolith blocks = 252
   logical nodes) via `forwardTokenLazyMonolith`.
3. Run token 0 eager; run token 1 through `captureMonolithicGraph`;
   time subsequent tokens via `cuGraphLaunch` replay.
4. Compare:
   - v1 eager (no graphs)
   - v1 + `HESPER_CUDA_GRAPHS=1`
   - Monolith + capture

Expected outcome (honest):
- Monolith + capture ≈ v1 + CUDA_GRAPHS (same kernel set, same
  replay mechanism).  Small win if IRv2 setup is cleaner.
- Do NOT expect to outperform llama.cpp.  PTX per-kernel efficiency
  is the dominant remaining factor (see doc 28 §10).

If v2 capture is slower than v1 CUDA_GRAPHS: investigate why host-
side setup differs.  Most likely culprit is some non-captured
`IO.Ref` read inside `runMonolithicGraph` that forces
host-side synchronisation during replay.

## Files touched this session

Changed:
- `Hesper/Circuit/IRv2.lean` — added `GemmaAttnOutProj`, `PostAttnNormAdd` nodes
- `Hesper/Circuit/Dispatch_v2.lean` — wired both node runtimes; extended analyzer
- `Hesper/Circuit/Lowering_v2.lean` — extended exhaustive match
- `Hesper/Models/Gemma4_v2.lean` — 6-block `forwardLayerLazyMonolith` + 9-slot
  `forwardTokenLazyMonolith`
- `Examples/DSL/Gemma4MonolithLayerParity.lean` — full GPU harness, 6-block
  wiring, PLE-off reference
- `docs/llama-fusion-analysis/30-f2-handoff.md` (this file)

Unchanged (stable from prior sessions):
- `Hesper/Models/Gemma4Bridge.lean` (F1 bundle extractor)
- All B1–B9 parity PoCs still pass

## Build status at session end

```
$ lake build gemma4-monolith-layer-parity
✔ [115/115] Built «gemma4-monolith-layer-parity»:exe

$ HESPER_SKIP_OUTSCALE=1 lake exe gemma4-monolith-layer-parity | tail -3
[Parity] max |err| over 2560 elems = 5.037946
DIVERGE: 5.037946 — PLE + layerOutScale not yet modelled
```

Build is green.  GPU runtime is stable (no crashes, no NaN).  Bundle
resolution works for all 42 layers.  The 5.04 residual is the only
remaining blocker to declaring F2 complete.

## Strategic reminder (from doc 29)

**"The BlockGraph is the execution plan.  Do not let the host
negotiate with the GPU during a token."**

Phase F3 is about *proving* this on the clock.  Expect the IRv2
capture + replay path to match production's CUDA-Graphs path in TPS.
Outperforming production is not the goal; being *as fast* while
expressed as a 4-node-per-layer BlockGraph (vs 30+ ad-hoc IO calls
in `forwardBlock`) is the architectural win.
