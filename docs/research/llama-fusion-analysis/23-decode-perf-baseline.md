# Decode Perf Baseline (2026-04-20, correctness = done)

After the correctness campaign closed (all 16 unit tests passing,
HESPER_CHAT=1 produces coherent output), this is the current decode
perf baseline on the same RTX 4070 Ti + Gemma 4 E4B Q4_K_M:

## Measured

| Mode | TPS |
|------|-----|
| Default (no CUDA Graphs) | 41.2 |
| `HESPER_CUDA_GRAPHS=1`   | **65.8** (+60%) |

Target (task #51): **120 TPS**. Gap: ~55 TPS to close.

## Profile under CUDA Graphs (`nsys`, 30 decode tokens)

**Host-side (cuda_api_sum)**:

| Time (%) | Total | Calls | Name |
|----------|-------|-------|------|
| 34.0 | 337 ms | 30 | `cuStreamSynchronize` |
| 34.0 | 337 ms | 1,478 | `cuMemcpyHtoD_v2` (capture + replay) |
| 7.8  | 77 ms  | 1,454 | `cuMemAlloc_v2` (init) |
| 0.4  | 4 ms   | 29 | `cuGraphLaunch` (avg 140μs) |

**GPU (cuda_gpu_kern_sum, top hitters)**:

| GPU ms | Instances | Approx name |
|--------|-----------|-------------|
| 58.5 (total `main`, sum of below) | 2,510 | — |
| 17.2 | 315 | 15 calls/tok → most-called matmul variant |
| 11.5 | 21 | 1/tok × ~550μs → **Q6_K lmHead** |
| 4.0  | 35 | PLE-related? (42 layers / some batch size) |

GPU kernel budget per decode token ≈ **~2.5 ms**, yet actual decode is
~11 ms/token. **8.5 ms/token lost to dispatch + sync overhead.**

## Root cause: argmax forces per-token GPU↔host sync

```
loop:
  gpuArgmax:
    [argmax kernel dispatch] ─sync readBuffer 4B─> nextToken  ← 11ms stall
  update 5 pinned staging slots (token/pos/cacheLen/plRow/posF32)
  cuGraphLaunch (graph contains: embeddingLookup, 42 blocks, lmHead)
  cuStreamSynchronize  ─────wait────────────────> GPU done
```

`Hesper.Models.Gemma4.gpuArgmax` calls `GPUBackend.readBuffer` on a
4-byte argmax result.  `readBuffer` is synchronous, so each token
incurs a full CPU↔GPU round-trip wait.

## Plan (task #122 full version)

To close the 8.5 ms/token gap we need an argmax→embedding path that
stays on the GPU:

1. **Keep argmax on GPU**.  After lmHead + softcap, run `argmaxKernel`
   inside the decode graph, writing to `state.argmaxBuf` (u32 × 1).
2. **Wire embedding lookup to read from argmaxBuf**.  Today
   `q6kEmbeddingLookupKernel` reads from `state.tokenBuf`; change the
   decode capture to bind `state.argmaxBuf` instead (the shader
   already accepts any u32 × 1 buffer as `token_ids`).
3. **Same trick for PLE** (`plRawRowBuf` contains the same tokenId).
   Either reuse argmaxBuf or add a single-byte GPU-side copy.
4. **EOS / end-of-stream check**: poll `state.argmaxBuf` via a
   pinned-host async DtoH copy *after* each graph launch, but let the
   stream run ahead — i.e. check EOS one token late.  Net latency cost:
   one wasted generation per stop event (~10 ms), amortised over the
   full response.

Expected outcome: decode shrinks from 11 ms/tok toward the GPU kernel
sum (~2.5 ms/tok), pushing TPS from 65.8 → ~150+ (cache-warm case).

## Related tasks

- #122 Decode TPS 40 → 115 TPS — this is the main vehicle.
- #127 Phase C2 Wire CUDA Graphs into decode loop — already in_progress,
  brought TPS 41 → 65.8.  The sync-free argmax is its next sub-step.
- #49 Reduce dispatch overhead — completed for the per-kernel level;
  this is the remaining host-side chunk.

## Artifacts

- /tmp/hesper_profile.nsys-rep — the 30-token nsys trace.
- Reproduce baseline:
    HESPER_DP4A=1 HESPER_CHAT=1 HESPER_CUDA_GRAPHS=1 \
      lake exe gemma4-cuda data/gemma-4-e4b-it-Q4_K_M.gguf "..." 50
