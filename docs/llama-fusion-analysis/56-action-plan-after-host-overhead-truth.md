# Action plan after the host-overhead breakdown (2026-04-26)

Doc 55 (`55-host-overhead-breakdown-2026-04-25.md`) showed that the 8 ms
wall gap between hesper (~60 TPS) and llama-cli (~110 TPS) is NOT Lean
runtime overhead but **GPU-side**: 2.5 ms of pure kernel time + 5.5 ms of
GPU drain wait that the host blocks on while waiting for the last kernel
of the forward to finish.

This document fixes the priority order for closing that gap.

## Budget per workstream

| # | Workstream                                | Budget    | Expected ΔTPS | Status |
|--:|-------------------------------------------|----------:|--------------:|--------|
| 1 | Kernel speed (Q4_K matmul → llama.cpp parity, task #47) | -2.5 ms/tok | +18 TPS  | active |
| 2 | On-device argmax + token feedback (retry of #229)        | **0 ms wall (drain just renames itself)** | **+0.2 TPS measured** | **DEAD END — kept as infra only** |
| 3 | Lean Array overhead in `Hesper.Circuit.CompiledCircuit.replay` | -0.75 ms/tok | +3 TPS | deferred |
| — | Stretch: skip the `cuMemcpyDtoH(4 byte)` entirely (graphs OFF pipelined) | requires #2 | — | follow-on |

The numbers add up to +50-60 TPS, taking us from 60 → 110-120 TPS. They
are **independent**: #1 shrinks the kernel that #2's drain is waiting on,
#2 unblocks the host so kernel speed bottleneck is exposed cleanly. Either
ordering works; we run them in parallel because they touch different
files.

## Workstream 1 — Kernel speed (task #47)

**Goal**: hesper Q4_K matmul kernels run at the same per-call ms as
llama.cpp's `mul_mat_q` template specialization for Q4_K, on the
gemma-4-e4b-it-Q4_K_M.gguf hot path.

**Hot kernels** (from prior `perf` and `nsys` work):
- `q4kFusedNormQKVKernel`     — qkvNorm + Q4_K matmul (3 outputs)
- `q4kFusedGateUpKernel`      — gate + up Q4_K matmul (2 outputs)
- `q4kBlockCoopKernel`        — generic Q4_K matmul (FFN down, others)

llama.cpp's reference is `ggml/src/ggml-cuda/mmq.cu` template
`mul_mat_q<QK_K, Q4_K, ...>`, which:
1. Loads Q8_1 tiles cooperatively into smem with **f16 half2** packing
   (we already match this — task #146).
2. Issues `dp4a` per QI4_K (32) sub-block.
3. **Pipelines** loads via `__pipeline_memcpy_async` with software
   double-buffering (Ampere+ feature).
4. Uses `cudaFuncSetAttribute(MaxDynamicSharedMemorySize, 64 KB)` to spill
   weight tiles into the larger smem pool, raising occupancy.

**Steps**:
1. ncu the three hot kernels with `--set roofline --section
   SchedulerStats,WarpStateStats,MemoryWorkloadAnalysis` and capture the
   stall reasons. Save reports under `docs/llama-fusion-analysis/57-...`.
2. Compare against the llama.cpp PTX dump (already extracted in earlier
   tasks; check `Hesper/Backend/LlamaCppPTX.lean` registry).
3. Apply the dominant fix from the diff. Almost certainly one of:
   - `__pipeline_memcpy_async` for Q8_1 tile loads (Ampere SM_86+ has it,
     RTX 4070 Ti is SM_89). Hesper currently does sync `ld.global` →
     `st.shared`.
   - Bigger smem allocation per block to fit two Q8_1 tile copies.
   - Better fastdiv for the `(blockIdx, ...)` → row/col mapping.
4. Validate with `Examples/DSL/Gemma4MonolithLayerParity.lean` after each
   change. Bit-parity must hold against the production forward.
5. Re-run `scripts/perf_compare.sh both` to confirm the kernel-time row
   drops to within 5 % of llama-cli's.

**Stop condition**: hesper GPU kernel time ≤ 9.0 ms/token.

## ⚠️ Workstream 2 retired (2026-04-26)

Implemented as `HESPER_DEVICE_ARGMAX=1` and measured: **wall unchanged
(59.4 → 59.6 TPS, +0.2)**. nsys shows `cuMemcpyDtoH(4 byte)` going to 0 and
`cuStreamSynchronize` rising to exactly 9.8 ms — i.e. the same wait, just
attributed to a different driver row. The `DtoH = 9.8 ms` in doc 55 was
the GPU drain itself, NOT a copy cost; renaming the API doesn't shorten
the wait.

llama-cli's `cuStreamSync = 6.8 ms` < hesper's 9.8 ms by exactly the
kernel-time delta (8.4 vs 10.9 ms). The drain pins itself to the kernel
finish time. The only knob that actually moves the drain is shrinking
the kernel: workstream **#1**.

The host-mapped infrastructure stays in tree (clean attribution makes
nsys traces easier to read, and it's needed for graphs-ON capture
safety) but it is not on the TPS path. See `feedback_dtoh_is_drain.md`.

## Workstream 2 (retired) — On-device argmax + token feedback (retry of #229)

**The bubble**: nsys shows 9.8 ms/token spent inside `cuMemcpyDtoH_v2`,
which is hesper reading the 4-byte argmax result back to host so that the
*next* token's prefill can use it. The driver implicitly synchronizes the
stream, so the host blocks on GPU drain. llama-cli avoids this by

1. Running argmax as a kernel that writes directly to a host-mapped
   `cudaMallocHost`'d buffer (no DtoH copy needed).
2. Issuing an explicit `cudaStreamSynchronize` for fence semantics, then
   reading from the host-mapped buffer with no driver involvement.

**Why #229 was reverted**: that attempt confused per-token wall with
total-token denominator (the pipeline finished N tokens but TPS was
computed against N+1 because EOS hit at a different point). With doc 55's
methodology (`HESPER_IGNORE_EOS=1`, `--single-turn` on llama-cli, fixed
N) we can compare cleanly.

**Steps**:
1. Add a host-mapped pinned ring slot for the argmax result. New
   `Hesper.CUDA.allocPinnedHostMapped` FFI (analogous to the existing
   pinned ring buffer for writeBufferOffset).
2. Write the argmax kernel output directly to the host-mapped slot via
   the device pointer obtained from `cuMemHostGetDevicePointer`.
3. After the per-token forward, call `cuStreamSynchronize` once
   (explicit drain — same semantics as llama-cli's). Read the slot from
   host as a normal memory load — no driver call.
4. Wire the next-token's `tokenId` parameter to read from this slot
   (input is already a small u32 register-style param).
5. Behind `HESPER_DEVICE_ARGMAX=1` env so it can be A/B'd against the
   current sync-DtoH path.

**Stop condition**: `Host: DtoH (sync read)` row in
`scripts/perf_compare.sh` output drops from 9.8 ms to <0.5 ms (just the
unblocked memory load) and `Host: cuStreamSync` rises to ≈ kernel-time
delta. Wall / token drops by 6-7 ms. Output identical to current path on
HESPER_CHAT prompts.

**Risk**: writeCombining behaviour of host-mapped memory means random
host reads can be slow. We mitigate by reading sequentially (ring buffer)
and, if needed, using a separate non-WC read-back buffer.

## Workstream 3 — Lean Array overhead in `replay` (deferred)

`Hesper.Circuit.CompiledCircuit.replay` iterates an `Array
(CompiledStmt × ...)` per token. The args Array gets `expand`'d for COW
on every iteration because the dispatcher writes back resolved buffer
pointers. `lean_copy_expand_array` is 6.6 % of cycles, `lean_dec_ref_cold`
7.6 %.

The fix is structural: pre-allocate the args ByteArray once at compile
time, then mutate it in-place via the FFI dispatch path (the args are
just CUdeviceptr values, no Lean pointers, so refcounting is unnecessary).

**Defer** until after #1 and #2 close, because:
- Bounded gain (≤ 0.75 ms wall, +3 TPS).
- The Array layout is shared with the IRv2 replay path, so changing it
  affects both. Better done after the kernel-side gains stabilize.

## Sequencing this session

Run #1 and #2 in parallel:
- #1 work: ncu + kernel rewrite, all in
  `Hesper/CUDA/{CodeGen,PTX}.lean` and `Hesper/Models/Gemma4/Kernels.lean`.
- #2 work: FFI + `Hesper/Models/Gemma4.lean` decode loop.

They don't touch the same files, so we land them as independent commits
and re-measure with `scripts/perf_compare.sh` after each.

After both: #3 if and only if there is still > 5 ms gap to llama-cli.
