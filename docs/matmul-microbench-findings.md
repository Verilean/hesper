# Q4_K Matmul Microbenchmark Findings

Written after ~2 weeks of optimisation attempts on hesper's Q4_K/Q6_K
dp4a matmul kernels, trying to close the gap to llama.cpp (115 TPS CUDA,
109 TPS Vulkan) and to explain why hesper sits at 43.7 TPS.

## Setup

Target shape: `inDim=2560, outDim=10240` (the fused-gate+up kernel).
Hardware: RTX 4070 Ti, peak DRAM 504 GB/s, peak fp16 ~80 TFLOPS.
Benchmark approach: seven kernels (K1..K7), each strictly more work than
the previous, both warm-L2 (single weight buffer) and cold-L2 (rotating
8 × 14 MB buffers, total 112 MB >> 48 MB L2).

## Results

| Kernel | warm-L2 | cold-L2 | BW achieved | peak% |
|---|---|---|---|---|
| K1 pure weight stream | 13.7 µs | 33.1 µs | 446 GB/s | **88%** |
| K2 + dequant | 13.2 | 33.2 | 445 | 88% |
| K3 + dp4a | 13.2 | 33.2 | 445 | 88% |
| K4 + subgroup reduce | 13.9 | 33.3 | 442 | 88% |
| K5 (2 weight buffers) | — | **63.3** | 466 | **92%** |
| K6 full fused (via `executeWithConfig`) | — | 299 | 99 | 20% |
| **K6b full fused (via `executeWithConfigCached`)** | — | **65.2** | **452** | **90%** |
| K7 hand-written PTX, `fma.rn.f16x2` | — | 33.2 | 445 | 88% |
| Real `fusedQ4KMGateUpDP4AKernel` (1-row) | — | **142** | — | — |
| Real `fusedQ4KMGateUpDP4A4RowKernel` | — | **124** | — | — |

## Findings

### 1. Matmul itself is already at ~90% of DRAM peak

K1 (pure weight sum) and K6b (full fused gate+up with cache path) both
hit ~445 GB/s against a 504 GB/s peak.  There is essentially no slack
left on the matmul bandwidth side.

### 2. `fp16x2` packed FMA is irrelevant for this workload

K7 — a hand-written PTX variant with 8 chained `fma.rn.f16x2` per inner
loop iteration — lands at exactly the same 33 µs as K1/K4.  The compute
time for 26 M FMAs at 80 TFLOPS is ~0.3 µs, totally hidden by the
weight-load stream.

**Conclusion: fp16 FMA extension to the DSL will not help.  The kernel
is not compute-limited.**

### 3. The real gap is "kernel-dispatch overhead around the matmul"

K6 (non-cached) vs K6b (cached) measured 299 µs vs 65 µs for the same
computational work on the same inputs.  The 234 µs difference is the
per-call cost of `executeWithConfig`'s PTX hash lookup, argument
resolution, and internal bookkeeping.  The real kernel
(`executeWithConfigCached` path) avoids most of this, landing at 142 µs
— which is still 2.2× slower than K6b's 65 µs.

The remaining 77 µs gap between K6b (65 µs, isolated microbench) and
the real kernel (142 µs, running inside 1370 kernels/token) is explained
by **inter-kernel interference**: other kernels running between gate+up
invocations evict weights from L2, forcing the next gate+up to re-read
from DRAM.  In isolation, the same weight is already warm; in real
inference, it isn't.

### 4. Raw-PTX JIT path works

`k7Ptx : String` was loaded via `cuModuleLoadData` and dispatched via
`cuLaunchKernel` with no ShaderM involvement.  This establishes that
**native-PTX kernels can slot into hesper without DSL changes**, which
is the technical foundation for the VerifiedNativePTX plan.

## What this means for optimisation strategy

What does *not* move the needle:
- **L2 access-policy windows** (tried, TPS-neutral — a single pinned
  layer is evicted by the other 41 layers' worth of weights)
- **fp16/fp16x2 DSL extension** (compute is not the bottleneck)
- **Smaller scale caches, byte packing, loop unrolling** (all already
  absorbed by ptxas, empirically neutral)

What *would* help, in order of expected ROI:
1. **Fewer kernels per token** (currently 1370; llama.cpp ~190).
   Candidates: fused Q/K/V proj, fused attention + O proj, fused
   RMSNorm + matmul.  Each fusion reduces L2 eviction pressure.
2. **CUDA graphs** to cut `cuLaunchKernel` overhead and possibly keep
   warmer weights by batched submission.
3. **`mma` (Tensor Core) path** for Q4_K — needs VerifiedNativePTX
   because WGSL has no equivalent.  Large effort, possibly 2×+ gain.

What needs verification but probably helps:
- **Switching `executeWithConfig` internals** to avoid the 234 µs
  non-cached overhead surprise (lazy-init the cache entry even when the
  caller didn't pass a ref).  This would speed up many small kernels
  that currently take the non-cached path.

## Raw artefacts

- `Tests/CUDA/CUDAMatmulMicrobench.lean` — seven benchmarks, both warm
  and cold-L2 variants.  Reproducible as `lake exe cuda-matmul-microbench`.

Last updated: 2026-04-15.
