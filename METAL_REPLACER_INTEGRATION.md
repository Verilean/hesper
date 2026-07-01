# metal_replacer integration — swap llama.cpp's tuned Metal kernels for our WGSL hot kernels

## Goal
Replace the 2-3 hottest generated WGSL→Tint→Metal kernels (MoE `mul_mat_id`, dense `mul_mm`, flash-attn)
with llama.cpp's hand-tuned Metal kernels — getting their efficiency directly and bypassing the WGSL/Tint
limits (occupancy control, coalescing) that this session hit repeatedly. Keep the WGSL DSL + verification +
portability for everything else.

## Feasibility (CONFIRMED)
Our pipeline is WGSL → Dawn (Tint→Metal internally) → `wgpu::ComputePipeline` (native/bridge.cpp:1441,1722).
Dawn exposes the underlying Metal objects:
- **`dawn::native::metal::GetMTLDevice(WGPUDevice)`** — PUBLIC (dawn/native/MetalBackend.h:51). ✅
- **`dawn::native::metal::Buffer::GetMTLBuffer()`** — internal (src/dawn/native/metal/BufferMTL.h:49); reach
  it via `ToBackend(FromAPI(wgpuBuffer))->GetMTLBuffer()` (our bridge already links Dawn's native libs).

## Two mechanisms (pick after seeing metal_replacer's actual API)

### (A) Dawn-interop bypass — dispatch the custom Metal kernel ourselves for hot kernels
- Get `MTLDevice` (GetMTLDevice) + the `MTLBuffer`s (GetMTLBuffer) for the kernel's wgpu::Buffers.
- Compile llama.cpp's `mul_mat_id`/`mul_mm` Metal source → `MTLComputePipelineState` (cache it).
- Dispatch via a Metal command buffer, on **Dawn's own MTLCommandQueue** so ordering holds (queue is
  internal — get it via ToBackend(device)->GetMTLQueue() or dispatch + `WaitForCommandsToBeScheduled`).
- HARD PART: cross-API sync. Cleanest is same-queue submit ordering; fallback is a full
  `WaitForCommandsToBeScheduled` around the hot dispatch (a real barrier — measure the cost, cf. the
  endBatch ~287ms lesson — but it replaces a slow kernel with a fast one, so may still net-win).

### (B) DYLD interpose (likely what metal_replacer does) — swap the MTLFunction under Dawn
- Interpose `-[MTLDevice newComputePipelineStateWithFunction:...]` (or library creation); when Dawn compiles
  OUR kernel, return a pipeline built from llama.cpp's Metal function instead.
- Dawn's dispatch/sync/bind-groups are UNCHANGED — no cross-API sync issue.
- HARD PART: **ABI match**. The swapped function must accept our kernel's exact buffer bindings (indices,
  layouts, the input/weight formats). llama.cpp's `mul_mat_id` has a different ABI → write a thin Metal
  wrapper with OUR binding signature that adapts to llama.cpp's kernel, OR make our WGSL kernel emit the
  bindings in llama.cpp's order. Identify the swap by the generated Metal function name / a shader hash.

## Recommendation
Start with **(A)** for ONE kernel (the MoE gate/up `mul_mat_id`, the biggest recoverable) as a PoC — it's
self-contained (we control the dispatch, no interpose magic) and directly validates the interop + the win.
If the cross-API sync is too costly, switch to (B).

## Concrete steps (fresh focused session)
1. **PoC interop**: new `native/metal_replace.mm` (Objective-C++). `lean_hesper_mtl_device_name(device)` →
   GetMTLDevice → `[dev name]` → prove the Metal handle is live. Wire into CMakeLists (add the .mm, link
   `-framework Metal -framework Foundation`). Rebuild is FAST (Dawn is cached in .lake/build/dawn-install;
   only hesper_native recompiles). Confirm from Lean.
2. **Buffer bridge**: `GetMTLBuffer` for our wgpu::Buffers (needs BufferMTL.h + ToBackend — internal include
   path from dawn-src). PoC: read back a buffer's MTLBuffer.contents and compare to a known value.
3. **Custom kernel dispatch**: compile a TRIVIAL Metal kernel (copy A→B), dispatch on Dawn's queue with the
   bridged MTLBuffers, verify ordering vs a Dawn dispatch. This validates mechanism (A) end-to-end.
4. **Drop in llama.cpp `mul_mat_id`**: extract the exact kernel + its arg struct from
   refs/llama-diffusion-gemma ggml-metal.metal; build its arg buffer from our routing/weights; dispatch;
   validate "Paris"+"Jupiter" vs the per-slot default; measure emb+fwd.
5. Loop (PERF_AUTOTUNE_LOOP.md): measure → next hottest (mul_mm, flash-attn).

## Why this is the right call (this session's evidence)
- WGSL can't control occupancy through Tint (our BM=64 failed; the fused kernel hit padding + Dawn race).
- llama.cpp's Metal kernels ARE the target (~2.5× on the same GPU, same simdgroup_matrix — no AMX, no magic).
- Replacing 2-3 hot kernels with tuned Metal is far higher-ROI than re-deriving them in WGSL, and preserves
  the DSL/verification/portability for the other ~90% of kernels.
