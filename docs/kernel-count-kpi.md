# KPI: kernels/token

Tracked metric: **GPU kernel dispatches per output token**, measured via
`nsys stats --report cuda_gpu_kern_sum` on a 30-token decode of
`data/gemma-4-e4b-it-Q4_K_M.gguf` with prompt "Hello world".

Why this KPI: per-token walltime on single-token decode is dominated by
kernel launch + L2 interference overhead, not per-kernel compute.
Empirically, each kernel launch costs ~10 µs of wall-clock
(launch + minimum GPU residency); 1000 extra kernels/token ≈ 10 ms/token
≈ 50 TPS worth of headroom.

## Measured

| engine | date | total (30 tok) | **per-token** | vs llama.cpp |
|---|---|---:|---:|---:|
| llama.cpp CUDA | 2026-04-15 | 5,615 | **187** | 1.00× (target) |
| hesper (session start) | 2026-04-15 | 43,987 | 1,466 | 7.8× |
| hesper (Q6_K ffn_down dp4a) | " | — | ~1,400 est. | 7.5× |
| hesper (fused gate+up) | " | — | ~1,400 | 7.5× |
| hesper (fused KV) | " | 40,097 | 1,336 | 7.1× |
| hesper (fused Q/K/V share q8_1) | " | ~38,800 est. | ~1,295 | 6.9× |
| hesper (fused post-norm) | " | 36,827 | 1,227 | 6.6× |
| hesper (fused wK+wV) | " | 40,097 | 1,336 | 7.1× |
| hesper (fused QKV q8_1 share) | " | ~38,800 | ~1,295 | 6.9× |
| hesper (fused post-norm) | " | 36,827 | 1,227 | 6.6× |
| hesper (fused RoPE-K+KVwrite) | " | 36,696 | 1,223 | 6.5× |
| hesper (fused RoPE-K+KVwrite) | " | 36,696 | 1,223 | 6.5× |
| hesper (Circuit DSL: wO via runCached) | 2026-04-15 | 35,326 | 1,178 | 6.3× |
| hesper (Circuit DSL: layerScale+pleScale3 auto-fused via ScalarExp) | 2026-04-15 | 34,191 | 1,140 | 6.1× |
| hesper (Circuit DSL: 3 RMSNorm sites via reduce-with-epilogue fusion) | 2026-04-15 | 33,997 | 1,133 | 6.1× |
| hesper (fused RMSNorm+Q8_1 for attnNorm→wQKV + ffnNorm→gate+up) | 2026-04-15 | 32,202 | 1,073 | 5.7× |
| hesper (fused 3-in-1 per-head qkvNorm)                           | 2026-04-15 | 30,721 | 1,024 | 5.5× |
| hesper (PLE inpGate matmul fused with GELU+slice-mul epilogue)   | 2026-04-15 | 29,261 |   975 | 5.2× |
| **hesper (current)** | " | **29,261** | **975** | **5.2×** |

**1000 barrier broken.**  The PLE `inpGate → geluGateMul` pair is a
matmul followed by a pointwise tail (`GELU(x) * perLayerInput[plOffset + i]`).
`fusedQ4KMLinearDP4AGeluSliceKernel` inlines the tail into the
matmul's write-out (lane 0 reads one extra f32, applies tanh-approx
GELU to the dot product, multiplies in, writes).  One fewer dispatch
per PLE site × ~33 layers = −49 kernels/tok, decode bit-identical.

Kernel body is a copy of `fusedQ4KMLinearDP4AKernel`'s matmul phase
with a custom write-out — a deliberate Option B macro-Prim.  A future
`Prim.matmulQ4KWithEpilogue` would let us remove the duplication by
parameterising the shared lowering with a `ScalarExp` tail.

Per-head qNorm+kNorm+vNorm previously ran as 3 separate dispatches per
`hasKV` layer.  A single hand-composed kernel with grid
`(numHeads, 3, 1)` now multiplexes them via `wg_id.y` — q uses its
scale (8 heads), k uses its scale (4 heads, WGs with x>=4 early-return),
v is bare (4 heads).  `wg_id.x/y` are workgroup-uniform so the
early-return branch never straddles a barrier.

Savings: −49 kernels/tok, +0.4 TPS (47.8).  Decode bit-identical, and
`fused-qkv-norm-gpu-test` proves the GPU output matches a per-head
CPU reference to f32 precision on all three Q/K/V output buffers.

First real cross-domain fusion: RMSNorm (global sum-of-squares reduce)
merged with the Q8_1 quantize step (per-block max-abs reduce) of the
subsequent matmul.  Single-WG kernel: 256 lanes first compute the
RMS-normalised value, then split into 8 warps that each handle one
32-element Q8_1 block via subgroup max-abs reduction.  Eliminates the
f32 normedBuf VRAM round-trip (~10 KB/layer/token) AND one standalone
dispatch per fused site.

Wired at `attnNorm → wQ+wK+wV` and `ffnNorm → gate+up` → net −60
kernels/tok (47.4 TPS vs 46.1 before).  Decode bit-identical.

`finalNorm`, `attnNorm`, and `ffnNorm` (~67 sites/tok) now go through
`CircuitM.rmsNorm`, which lowers as 4 ops (reduce + 3 pointwise) and
the `fuseReduceEpilogue` compiler pass collapses them to a single
GPU dispatch — matching the hand-written `RMSNorm.forward` baseline
in dispatch count, but with the kernel **generated from `ScalarExp`**
rather than being a hand-maintained ShaderM.  The net kernels/tok
delta is ~0 (each site already cost 1 dispatch); the value is the
architectural pivot — adding norm+matmul-quantize fusion is now an
IR rewrite, not a new hand-written kernel.

The last row is the **first production fusion driven entirely by the
element-wise compiler pass** (`fusePointwise`), not a hand-written
macro kernel.  The `layerScaleKernel + scaleKernel` chain in
`Gemma4.forwardBlock` is expressed as two CircuitM.pointwise ops and
collapses to one dispatch via β-reduction on `ScalarExp`.  Adding the
next pointwise fusion is now a data change to the IR, not a new
ShaderM.

## Corresponding TPS

| engine | TPS | wall-clock 30 tok |
|---|---:|---:|
| llama.cpp CUDA | 115 | ~260 ms |
| llama.cpp Vulkan | 109 | ~275 ms |
| hesper (session start) | 31.6 | 949 ms |
| hesper (current) | 46.6 | 643 ms |

## Kernel count breakdown (llama.cpp, 30 tok, 187/token)

- `mul_mat_q<Q4_K, ncols=16>` × 306: prefill matmuls, 16 tokens per kernel
- `mul_mat_vec_q<Q4_K, ncols=4>` × 306: 4-token prefill
- `mul_mat_vec_q<Q4_K, ncols=2>` × 306: 2-token prefill
- `mul_mat_vec_q<Q4_K, ncols=1>` × 186: **single-token decode** (~6/token)
- `mul_mat_vec_q<Q6_K, ncols=1>` × 37: Q6_K (ffn_down + lm_head)
- `rms_norm_f32` × 498+86+67: **~21/token** RMSNorms
- `rope_neox` × 55: <2/token
- `flash_attn_ext` × 35+35+35+35: <5/token (one per layer?)
- `quantize_q8_1` × 301+306: input quantize (~20/token)
- Various k_bin_bcast, k_set_rows, soft_max, gelu, etc.

llama.cpp's **187/token** is mostly: 6 matmul-decode + 21 RMSNorms +
<5 FA + ~40 misc ≈ 70-80 per-layer-per-token operations in flight at
once, with layer pipelining.

## Remaining gap for hesper: 1227 → 187 = 1040 to go

Candidate sources still in hesper's 1227/token:
- Attention: Q/K/V + O = still ~4 dispatches × 42 layers = **168**
  - Possible: fuse O-projection into flash attention output write
- FFN: 1 fused gate+up + 1 ffn_down = **~84**
- RMSNorm (attnNorm, qNorm, kNorm, vNorm, ffnNorm) = ~**210**
  - Fuse attnNorm into fusedQKV quantize path (biggest remaining win)
- Per-layer embedding ops = ~**168** (per_layer_input_gate, proj, etc)
- KV cache write = **84**
  - Already fused K+V in hesper (1 kernel per layer)
- RoPE = **84**
- FlashAttention = ~**42**
- Residual add + norm = already fused to **~84** (from 168)
- Misc quantize/scale/copy = rest

**Fattest remaining blocks** to fuse:
1. **attnNorm + fusedQKV quantize** — move the pre-attn RMSNorm inside
   the Q8_1 quantize pass. Saves 42/token.
2. **qNorm + kNorm + vNorm (per-head RMSNorm)** — 3 kernels × 42 = 126
   currently. Could fold into a single "per-head qkvNorm" kernel that
   handles all three in parallel per layer. Saves 84/token.
3. **Attention output → residual → next-layer inputBuf** — the output
   of attention feeds directly into `forwardNormThenAdd`. There's room
   to fuse the attention output projection with the residual add.
4. **RoPE + KV cache write** — both operate on the same Q/K buffers
   in sequence; could be one kernel.

Even aggressive fusion unlikely to get below ~400/token without CUDA
graphs + persistent kernel architecture. llama.cpp's 187 benefits from
much-larger kernels (mul_mat_q does whole prefill rows in one call) and
from CUDA graphs in recent builds.

Last updated: 2026-04-15.

## Per-kernel execution time profile (2026-04-16)

**Key finding: kernel count is NOT the bottleneck.**

Measured dispatch counts:
  llama.cpp Vulkan (Gemma 4 E4B Q4_K_M): **1,186 kernels/tok**, 98 TPS
  llama.cpp CUDA  (same):                   187 kernels/tok,  115 TPS
  hesper (current):                         975 kernels/tok,  49 TPS

hesper already has FEWER dispatches than Vulkan (975 vs 1186) but 2×
slower.  The gap is per-dispatch execution time, not count.

**Top 5 hot kernels (hesper, nsys decode 30 tokens):**

| Rank | Total ms |  %  | /tok | Avg µs | Kernel                                  |
|-----:|---------:|----:|-----:|-------:|-----------------------------------------|
|    1 | 167      | 34% |  45  |  122.6 | q4k-gate-up-dp4a-4row (ffn gate+up)    |
|    2 |  54      | 11% |  23  |   78.9 | q4k-dp4a-matmul-2row (wO)              |
|    3 |  53      | 11% |  23  |   78.4 | q6k-dp4a-matmul-4row (ffn_down)        |
|    4 |  40      |  8% |   1  | 1259   | q6k-dp4a-matmul-4row (lm_head)         |
|    5 |  21      |  4% |  38  |   18.1 | q4k-dp4a-matmul-2row (wQ)              |

**FFN gate+up analysis:**
- Theoretical memory-bound time: 29.4 MB / 400 GB/s sustained = 74 µs
- Hesper actual: **122 µs (60% efficiency)**
- llama.cpp Vulkan's 2-dispatch equivalent (2 × mul_mat_vec_q4_k): estimated ~74 µs

hesper's "fused" gate+up kernel is **~1.6× slower than the naive
2-dispatch path on Vulkan**.  Candidate causes: 4-row cooperative
smem access patterns, register pressure (34 regs), Q8_1 smem
bandwidth bottleneck at 4 warps × 72 KB.

Closing this single kernel's gap (122 → 74 µs) would save 2.2 ms/tok
= **+10 TPS**.  More than any fusion work remaining.

## Experiment A (2026-04-16): Q4_K 4-row smem-sharing kernel — **NO EFFECT**

Hypothesis: wO's 47 GB/s vs theoretical 400 GB/s meant it was
DRAM-bandwidth-starved; adding smem input sharing (like the existing
gate+up and Q6_K 4-row kernels) would give a 3× speedup.

Implementation: `fusedQ4KMLinearDP4A4RowKernel` with cooperative
smem Q8_1 input staging; wired into `forwardDP4A` and
`forwardFusedQKV` when `outDim % 4 == 0`.

Measured:
  wO (m=2560, k=2560):   78.9 → 78.8 µs  (no change)
  wQ (m=2048, k=2560):   18.1 → 17.7 µs  (noise)
  wK+wV (m=1024):        17.0 → 16.6 µs  (noise)
  Decode TPS:            48.4 → 48.6     (noise)

**Hypothesis refuted.**  The 47 GB/s figure measured
`weight_bytes / kernel_time`, but these small matrices (3.7 MB for
wO) sit entirely in the 48 MB L2 cache after the first access.
Second and subsequent calls hit L2, so DRAM bandwidth is not the
bottleneck — the kernels are compute-limited or warp-scheduling
limited.  smem staging just adds a redundant copy.

Change reverted.  Next candidate: eliminate Q8_1 pre-quantize
(llama.cpp Vulkan's approach) — save the 45 Q8_1 quantize dispatches
per token AND the Q8_1-round-trip latency, at the cost of rewriting
the matmul kernels to consume f32 directly (longer but more generic).
