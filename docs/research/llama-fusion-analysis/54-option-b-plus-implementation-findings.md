**SUPERSEDED by 57-host-overhead-canonical.md** — left for history; do not
follow conclusions here without cross-checking §3 of doc 57.

# 54 — Option B+ implementation findings and next steps

*Written 2026-04-24. Continuation of docs 51-53.*

## What landed

- `native/cuda_bridge.cpp`: `HesperLaunchDescriptor` pool (4096 slots), FFI
  functions `hesper_desc_reset` / `hesper_desc_register` / `hesper_desc_rebind` /
  `hesper_desc_launch`. Each descriptor deep-copies its arg array into
  `arg_storage` + parallel `arg_ptrs[]` so cuLaunchKernel receives the
  right `void**`.
- `Hesper/CUDA/FFI.lean`: extern bindings for the 4 new ops.
- `Hesper/Backend/CUDA.lean`: opt-in integration via `HESPER_DESC_PATH=1`.
  Side-table `cudaDescIdMap : Array (USize × USize)` keyed by a hash of
  `(cacheKey, args)`.  Two helpers factored out (`tryDescLaunch`,
  `tryDescRegisterAndLaunch`) to keep the GPUBackend instance body small
  (one attempt to add a field to `CUDACachedDispatch` blew the elaboration
  heartbeat budget — side-table avoids that).
- `Examples/Gemma4CUDA.lean`: `set_option maxHeartbeats 800000` — the
  instance body was already near the 200K limit.
- `Tests/CUDA/CUDALaunchBench.lean`: L4 measurement (descriptor launch).
  Reports 1.11 µs/call, same as L0 raw FFI — tight-loop microbench
  doesn't exercise the tail.

## Correctness test results

```
# Baseline (HESPER_DESC_PATH unset)
./gemma4-cuda "Hello" 30 →
  Hello! How can I help you today? 😊<turn|>   @ 58 TPS  ✓

# Desc path, 1 token
HESPER_DESC_PATH=1 ./gemma4-cuda "Hello" 1 →
  Hello   ✓   (first decode token correct)

# Desc path, 3 tokens
HESPER_DESC_PATH=1 ./gemma4-cuda "Hello" 3 →
  Hello hello hello   ✗   (token 2+ degenerate)

# Desc path, 30 tokens
HESPER_DESC_PATH=1 ./gemma4-cuda "Hello" 30 →
  Hello hello hello hello hello ... (×30)   ✗   21 TPS
```

Token-1 matches baseline bit-for-bit; token-2 onward emits the same
token repeatedly. TPS drops because cache misses on the non-matching
hash cause repeated descriptor registrations.

## Diagnosis

The naive "register-on-first-sighting, replay-by-(cacheKey,args-hash)"
model doesn't match hesper's reality:

1. **Some call sites rebind buffers across calls** — the args array
   differs legitimately between decode steps. Each new args tuple
   creates a new descriptor, bloating the pool and, worse, replaying
   whichever descriptor was registered first for that shape when the
   hash happens to collide.

2. **Some kernels consume dynamic scalars via buffer pointers** — e.g.
   `pos`, `cacheLen`, `tokenId` are written into pinned/device buffers
   and then the buffer pointer is passed as an arg. Pointer stays
   stable, but the **buffer content changes per step**. This is fine
   in principle (register once, replay many times), and most hesper
   kernels use this pattern.

3. **Buffer-rotation call sites** — a handful of sites legitimately
   use different buffers per call (e.g. prefill staging slots, some
   KV write paths). These must `hesper_desc_rebind` between calls,
   not register fresh. The current integration has no rebind wiring.

## Root cause of the degenerate-token output

Almost certainly (3): a decode-state-advance buffer (most likely the
argmax→`tokenBuf` feedback, or a pos-advance write target) is being
captured at step 1 and replayed unchanged at step N. The descriptor
fires the right kernel with the right weights but against a frozen
feedback slot, so every step produces the same token.

Confirming this would take another ~30 minutes with
`HESPER_KERNEL_TRACE=1` + a targeted log of "which descriptor fired
with what args", but the fix is the same regardless: go through the
design doc's **pinned-scalar + rebind** story instead of the
blanket-register path.

## Addendum (same session): rebind-every-call variant

Added `lean_hesper_desc_launch_with_args` FFI that writes fresh arg
pointers into the descriptor's `arg_storage` in C (zero Lean heap)
then fires `cuLaunchKernel`. Switched `tryDescLaunch` to always use
this path and keyed the side-table by `cacheKey` alone (rebind
handles per-call buffer changes).

Also added a missing `cudaBatchQueue.isSome → return false` guard in
`tryDescLaunch` so the descriptor path doesn't race the batch queue.

Results:

```
baseline              : 59.9 / 58.4 / 59.8 TPS  — correct output
HESPER_DESC_PATH=1    : 52.7 / 53.7 / 52.8 TPS  — correct output
```

**Correctness is now preserved** (same decoded text as baseline), but
throughput is **~10% slower**. The rebind FFI adds one cross-language
transition and a `lean_unbox_usize` loop per launch, and the Lean side
still runs the full `declaredNames.foldlM + namedBuffers.find?` args
resolution. The descriptor path is strictly *additional* work on top
of the normal args build — it doesn't skip any of the Lean-side list
walking.

This matches [project_argmax_not_bottleneck.md]'s pattern: microbench
"per-launch overhead" improvements don't translate to real TPS when
the actual bottleneck is Lean-side *tail* latency (GC / refcount /
Array growth events), not per-call steady-state cost. An FFI boundary
doesn't help when Lean still does the same list walks either side of
it.

### Inter-API gap distribution (measured)

Ran nsys with `-t cuda,osrt`, extracted consecutive-call gaps per
thread from `CUPTI_ACTIVITY_KIND_RUNTIME`, sorted per-thread:

```
                 baseline         desc path        delta
n gaps:          16662            16662            same
p50:                110 ns           106 ns        ~same (FFI overhead invisible here)
p90:              12.4 µs          14.5 µs        +17%
p95:                51 µs            65 µs        +27%
p99:               265 µs           268 µs         ~same
p99.9:            1.20 ms          1.21 ms         ~same
> 10µs gaps:      2399             2633           +10%
> 100µs gaps:      544              679           +25%
```

Not only did desc path not *reduce* the tail, it **increased** heavy-
tail event frequency — 544 → 679 events above 100 µs, a 25% jump.
Likely cause: each `descLaunchWithArgs` call now takes an `Array
USize` that's freshly built and FFI-passed, so Lean's refcount system
sees one more `Array` object per launch being dec_ref'd through the
extern boundary. The p50 is unchanged (median Lean→FFI crossing is
still fast), but refcount events cluster on shared allocator state,
so more of them = more stalls in the tail.

### What this says about GC pressure

GC pressure did **not** decrease. The Lean-side work — `declaredNames`
list walk, `namedBuffers.find?` per name, Array construction — is
entirely unchanged; we just added a second Array pass-through on top.
The descriptor pool's `arg_storage` lives in C and never touches Lean
GC, but the args we write into it come from a Lean `Array USize` that
is built and freed per call just like before.

### Why this was the wrong FFI granularity

Each call passes 4-8 pointers across the FFI. That's a unit of work
so small that the boundary-crossing cost (inc_ref / dec_ref on the
`Array` object, `lean_unbox_usize` per element in C) roughly equals
the work itself — hence "FFI that does nothing net". A useful FFI
boundary needs to either:

1. **Reduce the boundary count dramatically** — e.g. one FFI call per
   decode step that triggers the whole 840-launch schedule in C,
   instead of 840 FFI calls per step; or
2. **Carry a proportionally large job across the boundary** — a full
   transformer block's worth of kernel launches, etc.

"1 kernel = 1 FFI" is the smallest-possible granularity and the
least useful one. The C side still has to read the args per launch,
and Lean still had to assemble them; nothing was amortised. A proper
B+ implementation would move the schedule walker into C (the
`for id in block.scheduleIds` loop from doc 53 §Runtime) so the FFI
carries the *whole block's* work, not a single kernel's args.

### What would actually be faster

Skip the args-resolution loop entirely on the repeat path:

- Pre-resolve args at init time (once per call site) and keep the
  CUdeviceptr list in the descriptor — never re-resolve in Lean.
- Pinned-scalar slots (`stagingTokenPtr`, etc.) for the few dynamic
  values, written via `cuWritePinned` (already present).
- C-owned schedule iteration — `for id in block.scheduleIds do
  hesper_launch_by_id state id stream`.

That is exactly doc 53's "Option B+ proper" design. The opportunistic
path doesn't get there because args resolution lives in Lean and
can't be amortised without a schedule walker.

## What doc 53 actually proposed (and this didn't do)

Re-reading §Initialisation-time wiring:

> During `createInferenceState`: Walk every `ce`/`executeWithConfigCached`
> call site in the intended forward, [...] resolve each named buffer to
> its current `CUdeviceptr`, and register a descriptor entry via
> `hesper_register_descriptor(state, ...)` which returns a stable
> `size_t descriptorId`.

And §Dynamic values:

> Lean writes 4-8 bytes to the pinned slot (no Lean heap, single
> `cuWritePinned` FFI). C descriptors that need these scalars carry a
> pointer to the pinned slot as their argument. No `cuMemcpyHtoD` per
> launch — the GPU reads the slot at kernel launch via regular device
> memory (pinned memory is mappable).

And §Buffer updates between tokens:

> For the few sites where a buffer pointer might change [...], expose
> `hesper_rebind_descriptor(id, slot, new_ptr)` and call it only at
> the specific transitions.

The implemented path instead registers descriptors opportunistically
inside `executeWithConfigCached` without walking the schedule at init
time and without any rebind plumbing. That was the fastest path to a
proof-of-concept but runs straight into correctness on token 2.

## Next session plan (proper B+)

1. **Pick one forward-block phase** (e.g. the 6-kernel attention path)
   and drive *that* through the descriptor table explicitly, building
   the schedule at `createInferenceState` time with known `(func,
   buffers)` per kernel.
2. **Identify the 2-3 dynamic-scalar sites** (pos, cacheLen, tokenId
   feedback). Route them through `stagingTokenPtr` / `stagingParamsPtr`
   which already exist (see Gemma4.lean:InferenceState) — the pinned
   slot pointer becomes the kernel's arg, and Lean only writes 4-8
   bytes to the pinned mirror per step via `cuWritePinned`.
3. **Run the partially-descriptorised decode** side-by-side with the
   existing `executeWithConfigCached` path for the rest, via a
   `useDescSchedule : Bool` gate. Measure TPS and bit-parity on each
   phase migration.
4. Only after the explicit schedule builds + rebinds work, evaluate
   whether a general-purpose opportunistic desc-path inside
   `executeWithConfigCached` is even desirable (it might not be — the
   opportunistic path conflates "kernel identity" with "call-site
   identity + dynamic binding" and that distinction is what made this
   attempt go wrong).

## Status of code

The current code is **opt-in safe**: with `HESPER_DESC_PATH` unset,
baseline output and TPS (58) are unchanged. The new FFI bindings and
C descriptor pool are dormant unless explicitly enabled. No rollback
is required — the work can resume against this scaffold.

Decision: leave the opt-in scaffold in place. Next session starts from
step 1 above (explicit schedule build for attention phase), not from
the opportunistic helper route.
