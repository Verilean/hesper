# 53 — Metadata-free forward design (Option B+)

*Written 2026-04-24. Proposed next major work-item.*

## Premise

During `forward`, Lean touches **no computation data**. Tensors, weights,
activations all live on GPU as `CUdeviceptr` (a plain `USize` from Lean's
perspective). What Lean *does* touch is **launch metadata**: kernel
identity, buffer name→pointer mapping, dispatch shape, scalar params
(pos, cacheLen, token id).

`perf report` self-time on the hot path (post-preHash fix):

| symbol | self % | nature |
|---|---:|---|
| `__memmove_avx` | 30% | ByteArray copies (params, args) |
| `lean_dec_ref_cold` | 8% | GC child object dec_ref |
| `lean_copy_expand_array` | 5% | `Array.push` realloc |
| `Array_forIn...cudaExecuteImpl_spec__0` | 2.3% | buffer resolve loop |
| `lean_apply_1` | 1.4% | closure invoke |
| `List.find?` | 1.2% | namedBuffers lookup |

**Roughly 48% of decode CPU time is spent managing metadata that has no
computational meaning.** None of it corresponds to the forward formula;
it's all plumbing for "how do we describe this launch to the driver."

This is what the "Option B narrow C shim" tries to address at a single
hot site. This doc proposes a stronger move: **pull metadata ownership
out of the Lean GC entirely.**

## Design

### Principle

> Forward is a fixed schedule of kernel launches with a small handful of
> dynamic scalars. The schedule and most of its parameters are
> determined at inference-state initialisation time. Lean owns the
> compile-time representation (ShaderM, Exp, Prim, pullback); C owns
> the runtime representation (descriptor table, pointer arrays, pinned
> scalar slots).

### Runtime layout (C-owned)

```c
// native/cuda_bridge.cpp (sketch)
typedef struct {
  CUfunction func;
  uint32_t gx, gy, gz;
  uint32_t bx, by, bz;
  uint32_t n_args;
  CUdeviceptr* args;   // pre-resolved buffer pointers (stable across calls)
} HesperLaunchDescriptor;

typedef struct {
  size_t n_descriptors;
  HesperLaunchDescriptor* descs;     // pooled, malloc'd once at init
  CUdeviceptr* arg_storage_base;     // flat area backing all descs' args
  // pinned host mirrors for dynamic scalars (pos, cacheLen, tokenId,
  // posF32): C refreshes GPU buffers from these between launches as
  // needed, reading Lean-written values via lean_hesper_write_scalar.
} HesperInferenceState;

extern "C" lean_obj_res hesper_launch_by_id(
    size_t state_handle, size_t desc_id, size_t stream_or_zero);
```

### Initialisation-time wiring (Lean-owned → C-resident)

During `createInferenceState`:

1. Walk every `ce`/`executeWithConfigCached` call site in the intended
   forward (or equivalent: fold the forward's `ShaderM Unit` into a
   flat list of (ShaderM computation, grid, block, buffer-bindings)).
2. For each, compile PTX once, resolve each named buffer to its
   current `CUdeviceptr`, and register a descriptor entry via
   `hesper_register_descriptor(state, ...)` which returns a stable
   `size_t descriptorId`.
3. Lean keeps a `Array Nat` (or a compile-time-known indexing scheme)
   of descriptor IDs — each `forwardBlock` becomes
   `for id in block.scheduleIds do hesper_launch_by_id state id stream`.

### Runtime (Lean hot path = FFI loop)

```
-- sketch
def forwardBlock ... : IO Unit := do
  -- Update the handful of dynamic scalars once per block.
  hesper_write_pos state pos cacheLen
  -- Then iterate a pre-built schedule of descriptor ids.
  for id in block.scheduleIds do
    hesper_launch_by_id state id (← cudaCaptureStream.get).getD 0
```

No `List`, no `ByteArray`, no `s!"..."`, no `Array.push`, no closures
allocated per launch. The only per-launch Lean work is array indexing
and one FFI call.

### Dynamic values: pos, cacheLen, tokenId, posF32

These are the only things that actually vary per forward. hesper already
has `stagingTokenPtr` / `stagingParamsPtr` / `stagingPosF32Ptr` /
`stagingPLRowPtr` / `stagingColIdxPtr` — pinned host slots that were
created for the CUDA Graphs path but are underutilised on the graphs-OFF
path. Use them:

- Lean writes 4-8 bytes to the pinned slot (no Lean heap, single `cuWritePinned` FFI).
- C descriptors that need these scalars carry a pointer to the pinned
  slot as their argument. No `cuMemcpyHtoD` per launch — the GPU reads
  the slot at kernel launch via regular device memory (pinned memory is
  mappable).

### Buffer updates between tokens

For most decode kernels the bindings are stable (`state.buf1`,
`state.qBuf`, KV cache, etc., all allocated once in
`createInferenceState`). For the few sites where a buffer pointer might
change (pre-`forwardPrefillBatch` ones, if routed here), expose
`hesper_rebind_descriptor(id, slot, new_ptr)` and call it only at the
specific transitions.

## AD impact: none

AD works on the Lean-side `ShaderM`/`Exp`/`Prim` representation. That
lives at initialisation time when descriptors are built. Forward (and
backward!) kernels are compiled from the same representation and
registered as separate descriptor IDs. The backward pass gets exactly
the same metadata-free treatment; training benefits equally.

## Predicted TPS gain

Ballpark, starting from doc 52's prediction for Option B narrow shim:

| optimisation | predicted TPS (graphs OFF) |
|---|---:|
| current baseline | 60 |
| Option B (narrow shim on launch) | 85-95 |
| **Option B+ (metadata-free forward)** | **95-105** |
| llama.cpp graphs-OFF ceiling | 107 |

Rationale: the narrow shim only kills the tail at the launch site.
The metadata-free version kills the *source* of the tail — the
allocations and dec_refs happening between launches. perf self-time
items totalling ~48% of CPU should drop to near zero, and the p90/p99
gap distribution should collapse toward microbench levels.

### Risk of overshoot? No — GPU kernel time is still hesper's floor.

hesper's GPU kernel budget per token was measured at 10.3 ms/tok vs
llama.cpp's 7.8 ms/tok (2.5 ms gap is per-kernel compute difference,
orthogonal to metadata). So even a perfect host-side removal leaves
about 10-11 ms/tok → ~95 TPS. Higher than that is blocked by the
remaining per-kernel compute gap, which is a separate Q4_K / fusion
optimisation problem (tracked in other tasks).

## Implementation plan (next session)

1. **Descriptor API skeleton** — `native/cuda_bridge.cpp`:
   `hesper_state_create`, `hesper_register_descriptor`,
   `hesper_launch_by_id`. Returns `lean_obj_res` / accepts `USize`
   handles. No Lean heap allocation on the launch path.

2. **Pinned scalar slot plumbing** — already largely present via
   `stagingTokenPtr` etc. in `Gemma4.lean:InferenceState`. Ensure they
   are allocated unconditionally (not gated on `HESPER_CUDA_GRAPHS`),
   and wire `cuWritePinned` as the only Lean-side update path for
   pos/cacheLen/tokenId.

3. **Migrate one forward block section first** — pick `forwardBlock`'s
   attention path (6-8 kernels). Build the descriptor list at
   `createInferenceState`, replace in-situ `ce` calls with
   `hesper_launch_by_id`. Measure TPS.

4. **If TPS improves as predicted**, migrate the rest of
   `forwardBlock`, `forwardSingleToken`, PLE, lmHead.

5. **Keep the old `ce` path behind `HESPER_SLOW_PATH=1`** during
   migration so we can diff bit-parity.

## Validation

- Bit-parity: compare output vs baseline on "Hello" prompt.
- Gap distribution: re-run nsys with same flags as doc 51, show
  p90/p99 collapsed to microbench levels.
- perf self-time: memmove / dec_ref_cold / copy_expand_array should
  drop dramatically in decode region.
- TPS: direct measurement via existing benchmark.

## Why this matters beyond one optimisation

This design also separates "what to compute" (Lean, verified, typed,
differentiable) from "how to drive the GPU efficiently" (C, mutable,
GC-free, tight). That's the natural split for a verified-AD
framework that also wants to be fast — Lean is a great compiler
front-end, a poor runtime executor for hot per-call metadata. Every
future feature (multi-GPU, CUDA Graphs, distributed training) will
benefit from having the runtime metadata layer already in C.
