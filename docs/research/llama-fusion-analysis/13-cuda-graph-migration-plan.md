---
title: "13 — CUDA Graph replay: pinned-host migration plan"
date: 2026-04-18
status: planning
---

# Decode-path CUDA Graph replay: why we haven't landed it yet

## What works today

- FFI bindings for stream capture, graph instantiate, graph launch
  (`Hesper/CUDA/FFI.lean` + `native/cuda_bridge.cpp`, commits
  `4068ae3` and later).
- `Backend/CUDA.lean` routes every `cuLaunchKernel` and every
  `writeBufferOffset` through the capture stream when
  `cudaCaptureStream` is `some s`.
- Pinned-host primitives (`cuMemAllocHost`, `cuWritePinned`,
  `cuMemcpyHtoDFromPinned`).  Commit `786f59b`.
- `Models/Gemma4.lean`'s `generate` captures the first decode token
  when `HESPER_CUDA_GRAPHS=1`.  This does NOT regress TPS
  (still 52 TPS on Hello-world decode).

## What blocks replay

`writeBufferOffset (buf : CUDABuffer) (data : ByteArray)` records a
pointer to `data`'s heap storage into the captured memcpy node.
Lean's runtime can free that ByteArray at any point after the FFI
call returns, so on replay CUDA reads freed memory and aborts with
`CUDA_ERROR_ILLEGAL_ADDRESS`.

## Migration plan

### Step 1 — persistent pinned slots in `InferenceState`

Add three `USize` fields:

```lean
-- in InferenceState:
stagingTokenBuf : USize     -- 4 bytes, pinned host
stagingParamsBuf : USize    -- 8 bytes, pinned host (pos + cacheLen)
stagingPLRowBuf : USize     -- 4 bytes, pinned host
```

Allocated via `cuMemAllocHost` in `createInferenceState`, freed in a
cleanup path (or leaked until process exit — tiny memory).

### Step 2 — replace 6 writeBufferOffset call sites

Every site that writes one of {tokenBuf, paramsBuf[0], paramsBuf[4],
plRawRowBuf} converts to:

```lean
cuWritePinned state.stagingTokenBuf 0 tokenBytes 4
cuMemcpyHtoDFromPinned state.tokenBuf.ptr state.stagingTokenBuf 0 4 captureStream
```

Exact sites (from `grep -n writeBufferOffset .. tokenBuf|paramsBuf|plRawRowBuf`):

| line | target buffer | offset |
|------|---------------|--------|
| 1892 | paramsBuf | 0 |
| 2042 | paramsBuf | 4 |
| 2593 | plRawRowBuf | 0 |
| 2702 | paramsBuf | 0 |
| 2990 | plRawRowBuf | 0 |
| 3230 | tokenBuf | 0 |
| 3266 | plRawRowBuf | 0 |

### Step 3 — wire the replay path

Once the 7 sites above use pinned sources, the capture graph holds
stable pointers.  In `generate`:

```lean
if replay then
  cuWritePinned state.stagingTokenBuf 0 tokenBytes 4
  cuWritePinned state.stagingParamsBuf 0 posBytes 4
  cuWritePinned state.stagingParamsBuf 4 cacheLenBytes 4
  cuGraphLaunch exec stream
  cuStreamSynchronize stream
  -- argmax readback (unchanged, uses cuMemcpyDtoH — not captured)
```

### Step 4 — validate

- `HESPER_CUDA_GRAPHS=0` → 52 TPS (baseline)
- `HESPER_CUDA_GRAPHS=1` → target 75-85 TPS (10ms/tok host saved)
- Confirm same decode output bit-identical vs baseline

## Why haven't we finished it?

Pure time budget — step 2 is a 7-site mechanical edit plus a bit of
state plumbing, which is straightforward but needs care.  The session
ran out before that finished.

## Broader DSL (Phase A) answer

Circuit DSL v2 (`Hesper/CircuitV2/`) has a planned `StagingBuf (n : Nat)`
type that makes "source of a capturable write" a visible requirement
in the type system.  Migrating to that is the long-term correctness
fix; the steps above are the short-term fix that unlocks measurement.

## Expected decode TPS ceiling

From nsys: GPU=15 ms/tok, total=25 ms/tok, host=10 ms/tok.

With CUDA Graphs we should reach ~**65 TPS** (15 ms/tok + small driver
cost ≈ 15.5 ms/tok).  llama.cpp's 119 TPS implies their GPU time is
also much lower — they get tensor cores (MMA) on Q4_K matmul, which
hesper doesn't yet.  Next optimisation after CUDA Graphs: port the
mmvq has_fusion path for decode (see doc 12 §3.4).
