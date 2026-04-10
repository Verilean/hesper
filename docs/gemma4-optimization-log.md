# Gemma 4 e4b Decode Optimization Log

RTX 4070 Ti, Gemma 4 e4b (Q4_K_M, 42 layers, hiddenSize=2560), single-token decode (M=1).

## Benchmark Comparison

| Backend | Prompt (t/s) | Decode (t/s) | vs Hesper |
|---|---|---|---|
| llama.cpp CUDA | 423.7 | **115.1** | 4.94x |
| llama.cpp Vulkan | 20.3 | **42.5** | 1.82x |
| **Hesper (WebGPU/Dawn)** | — | **23.3** | 1.00x |
| Hesper (start of session) | — | 6.5 | 0.28x |

Session improvement: **6.5 -> 23.3 TPS = 3.58x**.

Remaining gap to llama.cpp Vulkan: **1.82x** (attributable to Tint->SPIR-V
translation overhead, Dawn dispatch overhead, kernel fusion differences, and
Q4_K inner-loop pattern differences).

## Theoretical Limits (M=1 Decode, RTX 4070 Ti)

- Total Q4_K + Q6_K weight: ~3.6 GB
- DRAM peak bandwidth: 504 GB/s
- Memory-bound floor: 3.6 / 504 = **7.1 ms/tok = 141 TPS**
- Achieved pure-read BW (microbench): **490 GB/s (97% of peak)**
- Practical ceiling with ~35% effective BW: **~40 TPS**
- For 80+ TPS: requires M>1 decode batching (vLLM-style continuous batching)

## Current Profile Breakdown (23.3 TPS = ~43 ms/tok inference)

Profile mode (unbatched, ~51 ms/tok):

| Section | ms/tok | % | Category |
|---|---|---|---|
| ffnDown (Linear) | 8.93 | 17.5% | Q4_K matmul |
| flashAttn | 8.32 | 16.3% | Attention |
| perLayerEmbd | 5.13 | 10.1% | Per-layer embedding |
| ffnGateUp (Linear) | 4.98 | 9.8% | Q4_K matmul |
| lmHead | 4.05 | 7.9% | Q6_K matmul |
| qkvProj (Linear) | 3.16 | 6.2% | Q4_K matmul |
| qkvNorm | 3.05 | 6.0% | Per-head RMSNorm |
| rope | 2.30 | 4.5% | RoPE |
| postAttnNorm+resid | 1.64 | 3.2% | RMSNorm + add |
| postFFNNorm+resid | 1.63 | 3.2% | RMSNorm + add |
| oProj (Linear) | 1.49 | 2.9% | Q4_K matmul |
| ffnGeluMul | 0.88 | 1.7% | Elementwise |
| attnNorm | 0.83 | 1.6% | RMSNorm |
| ffnNorm | 0.75 | 1.5% | RMSNorm |
| kvWrite | 0.62 | 1.2% | KV cache update |
| perLayerInputPre | 0.27 | 0.5% | Per-layer precompute |
| other | ~3 | ~6% | logitSoftcap, embed, etc. |

## Techniques That Worked

### 1. Block-Cooperative Q4_K Kernel (+50% TPS)

**Problem**: Old kernel assigned 1 thread = 1 Q4_K block (256 elements).
With stride-32 loop and `blocksPerRow = inDim/256`, lanes fell idle when
`blocksPerRow < 32`. For Gemma 4 (inDim=2560, blocksPerRow=10), only
10/32 = 31% of the subgroup was active. For KV projections (inDim=256),
only 1/32 = 3%.

**Fix**: Partition work *inside* each block across 32 lanes. Lane `tid`
owns sub-block pair `tid/8` and u32 index `tid%8`, reading exactly 1
qs u32 per block and computing 8 FMAs. All 32 lanes are always active.

**Result**: 6.5 -> 9.8 TPS. Per-call speedups: 2560x10240 = 2.67x,
256x2560 = 9.1x.

### 2. Software-Pipelined Weight Prefetch (+33% TPS)

**Problem**: Microbenchmark showed pure DRAM read at 490 GB/s (97% peak),
but Q4_K kernel with dequant chain ran at only 17-35 GB/s (3-7% peak).
The dequant instruction chain (shift -> and -> int2float -> mul -> sub ->
FMA) serialised with memory loads, preventing latency hiding.

**Fix**: Depth-1 software pipelining. Five per-block u32s (dmU32, sc0,
sc1, sc2, qsU32) are held in mutable `var next*` registers, prefetched
one block ahead. Each iteration snapshots `curr* = next*`, then issues
`next* = read(block+1)` BEFORE the dequant+FMA chain that consumes
`curr*`. The SPIR-V backend sees the next-block loads as independent
writes that can overlap with the current-block compute.

**Result**: 9.8 -> 14.2 TPS. Per-call: 2560x10240 = 3.19x,
10240x2560 = 1.40x.

### 3. Q6_K Block-Coop + SW Pipeline for lmHead (+23% TPS)

**Problem**: lmHead (Q6_K, 1x2560x262144, ~550 MB weights) was using the
old stride-32 subgroup kernel. Section profiling revealed it consumed
12.88 ms/tok — previously invisible because forwardSingleToken wasn't
instrumented.

**Fix**: Ported the block-cooperative + depth-1 SW pipelining pattern from
Q4_K to Q6_K. Each lane owns `l = tid` in each of the 2 chunks of a
Q6_K block (6 per-lane bytes). Shared d/scales are broadcast-loaded.

**Result**: lmHead 12.88 -> 4.05 ms (3.18x). Total 14.2 -> 17.5 TPS.

### 4. GPU Q6_K Row Dequant Replacing CPU (+35% TPS)

**Problem**: Section profiling revealed `perLayerInputPre` consumed 14.1
ms/tok. Inner profiling showed the F16 matmul was only 0.15 ms — the
real cost was **CPU-side Q6_K dequant** (Lean scalar loop + Array.map +
floatArrayToBytes) at 10.18 ms/tok for 40320 elements.

**Fix**: Upload the raw 33 KB Q6_K row bytes to a small GPU scratch
buffer (`ByteArray.extract` + `writeBuffer`, microseconds), then run
`q6kSingleRowDequantScaleKernel` on GPU (1 thread per element) to
dequant + scale directly into the target f32 buffer.

**Result**: perLayerInputPre 14.1 -> 0.27 ms (53x). Total 17.5 -> 23.7 TPS.

### 5. Fused postNorm + residualAdd (dispatch reduction)

**Problem**: Per-layer embedding's 3-dispatch chain (RMSNorm postNorm ->
copy-back -> residualAdd) consumed 2.63 ms/tok for trivial elementwise
work, dominated by dispatch overhead (~15 us per dispatch x 42 layers).

**Fix**: `fusedPerLayerPostKernel` — 1 workgroup of 256 threads,
subgroup-based RMS reduction (1 barrier vs 8), computes norm + adds into
the residual buffer in place. Saves 84 dispatches/token.

**Result**: 3 dispatches -> 1, perLayerEmbd 5.65 -> 5.13 ms. TPS unchanged
(dispatch savings too small to measure in batched inference mode).

### 6. Tiled Split-K Flash Attention (infrastructure)

Wired existing `flashAttentionTiledPhase1/Phase2` to full-attention layers.
Activates for `cacheLen > 32`. Not impactful in the 10-token decode test
(cacheLen <= 10), but ready for longer sequences.

## Techniques That Did NOT Work

### 1. FFN Gate+Up Fusion (0% gain)

Fused gate + up Q4_K matmul into one dispatch to share input loads.
Result: 0% speedup. Tint compiler emitted two independent inlined dot
product bodies with no register/memory sharing. M=1 decode is
compute-bound on dequant math, not memory-bound on input reuse.

### 2. vec4 Input Loads (0% gain)

Declared input buffer as `array<vec4<f32>>` for 16-byte coalesced loads.
Result: 0% speedup. Tint already coalesces consecutive f32 loads
automatically from the scalar version.

### 3. unpack4xU8 Vectorized Byte Extraction (0% gain)

Replaced per-byte shift+mask with WGSL `unpack4xU8` intrinsic.
Result: 0% speedup. Tint already optimised the constant-indexed
shift+mask pattern to equivalent bitfield extraction.

### 4. Dual Accumulator for ILP (0% gain)

Split the accumulator into even/odd block accumulators to break the
serial FMA dependency chain. Result: 0% speedup. Tint/NVIDIA's
compiler already performed this optimisation automatically.

### 5. smin Accumulator Pattern from llama.cpp (0% gain)

Separated the scale-weighted and min-weighted accumulation into two
independent FMA chains (as done in llama.cpp's mul_mat_vec_q4_k.comp).
Result: 0% speedup. Again, Tint's SPIR-V -> PTX compilation already
achieves equivalent instruction scheduling.

### 6. Multi-Row per Workgroup (mixed, net ~0%)

Doubled WG size to 64 (2 subgroups), each subgroup handles one output
row. Result: some shapes improved (outDim > inDim: -4% to -16%), others
regressed (outDim < inDim: +7%). Net TPS within noise. Suspected cause:
runtime `outIdx = pairIdx*2 + (tid >> 5)` computation lifts per-block
address calc into the inner loop.

### 7. Depth-2 Software Pipelining (0% gain over depth-1)

Extended prefetch depth from 1 to 2 blocks ahead. Result: 0% additional
gain. The compiler was already extracting depth-2+ scheduling from the
depth-1 code.

### 8. Split-K Q4_K Matmul (0-3% gain)

Split ffnDown's K dimension by 2/4/8 to increase WG count from 2560 to
5120/10240/20480. Despite reaching 3.5+ waves (same as ffnGateUp which
runs at 335 GB/s), ffnDown remained at 50 GB/s. The wave-count hypothesis
was disproven — L2 cache locality differences between shapes, not wave
count, explained the ffnGateUp vs ffnDown gap.

### 9. Subgroup-Only Flash Attention (-4% TPS)

Replaced 256-thread tree-reduction FA with 32-thread (1 subgroup)
barrier-free variant. Result: slightly slower. With only 32 heads,
dispatch is 32 WGs regardless of thread count. The 256-thread version
provided more latent warps for DRAM latency hiding despite its barrier
overhead.

**Important DSL lesson learned**: Lean `let` bindings build Exp trees that
get re-inlined at every use site. Without explicit `ShaderM.varNamed`
materialisation, `subgroupAdd(partialDot)` was emitted 8+ times per loop
iteration, causing a 10x regression. All complex kernel intermediates must
be materialised via `varNamed`.

## Key Insights

1. **Tint/NVIDIA already does micro-optimisation well**: ILP, CSE,
   instruction scheduling, vectorised loads. Manual attempts to replicate
   these had zero effect. Only **architectural changes** (thread
   utilisation, load/compute overlap, CPU->GPU offload) produced gains.

2. **Microbenchmarking is essential**: The GPU fixed-cost bench
   (`Bench/GpuFixedCost.lean`) conclusively proved that DRAM BW was
   available (490 GB/s) but the Q4_K kernel was using only 7%.
   The dequant-burdened variant proved the load->compute serialisation
   hypothesis, directly motivating SW pipelining.

3. **Section profiling catches hidden costs**: `perLayerInputPre` (14 ms,
   CPU dequant) and `lmHead` (13 ms, old Q6_K kernel) were invisible
   until section timers were added to `forwardSingleToken`. Together they
   accounted for 27 ms/tok — more than ffnDown.

4. **Dispatch overhead matters in profile mode but not batched**: 15 us
   per dispatch x 1030 dispatches = 15.5 ms overhead in unbatched
   profiling, but ~0 in batched inference. The profile/inference TPS gap
   (19.4 vs 23.3) is almost entirely this.

5. **M=1 decode is fundamentally memory-bound**: All weights are read
   once per token with no reuse. The only path to 80+ TPS is batched
   decode (M > 1) where weight traffic is amortised across multiple tokens.

## Remaining Optimisation Targets

| Target | Current ms | Potential ms | Expected gain | Difficulty |
|---|---|---|---|---|
| qkvNorm 3->1 kernel | 3.05 | ~1 | ~2 ms | Easy |
| RoPE Q+K -> 1 kernel | 2.30 | ~1 | ~1 ms | Easy |
| postAttnNorm+resid fusion | 1.64 | ~0.5 | ~1 ms | Easy |
| postFFNNorm+resid fusion | 1.63 | ~0.5 | ~1 ms | Easy |
| flashAttn (long seq) | 8.32 | ~4 | ~4 ms | Medium |
| ffnDown (cache locality) | 8.93 | ~6 | ~3 ms | Hard |
| **Total potential** | | | **~12 ms** | |

Potential: 43 ms -> 31 ms = **32 TPS** (vs llama.cpp Vulkan 42.5 TPS).
