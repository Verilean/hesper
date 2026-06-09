# Q4_K MMQ port plan — batched prefill speedup

## Problem

hesper prefill 86 ms / 19 tokens vs llama 12.86 ms (6.7× slower). Root cause:
hesper's batched matmul is MMVQ-style (1 warp per output element, gridX*gridY
workgroups). llama.cpp at batch>8 switches to MMQ (tile-based GEMM, multi-row
× multi-col tiles per WG, smem-staged X/Y).

## Approach: minimal MMQ for Q4_K

Don't port llama's full templated MMQ (4500 lines). Build the smallest viable
form: one specialized kernel for Gemma 4's prefill shapes, scale up tile sizes
once parity holds.

### Phase 1: skeleton (mmq_y=32, mmq_x=8)

Output tile: 32 rows × 8 cols per WG. WG threads = 256 (8 warps).
- Grid: `(outDim/32, ceil(seqLen/8), 1)`
- Each thread accumulates `(mmq_y/warp_size) × (mmq_x/nwarps) = 1 × 1 = 1` output.

Smem layout per WG:
- X tile: `mmq_y × MMQ_TILE_NE_K` ints (Q4 quantized) +
  `mmq_y × (MMQ_TILE_NE_K/QI4_K)` half2 (dm) +
  `mmq_y × (MMQ_TILE_NE_K/8)` ints (scales)
  ≈ 32 × 32 × 4 + 32 × 1 × 4 + 32 × 4 × 4 = 4096 + 128 + 512 = 4.7 KB
- Y tile: `mmq_x × MMQ_TILE_Y_K` (= mmq_x × (MMQ_TILE_NE_K + MMQ_TILE_NE_K/QI8_1)) ints
  ≈ 8 × 36 × 4 = 1.2 KB

K-loop: outer iterates over Q4_K super-blocks (256-K each), inner iterates
within the super-block (k01 += QR4_K * VDR_Q4_K_Q8_1_MMQ = 2*2 = 4).

### Phase 2: parity check

Build single-layer parity test against existing `forwardBatchDP4A_fromQ8`.
Single Q4_K layer, fixed inDim=2560, outDim=2560, seqLen=8 (matches mmq_x).

### Phase 3: bench

If parity holds, bench end-to-end vs 1-warp baseline. Target: 28 ms gate/up
batched → 8-12 ms (3-4× speedup matching llama.cpp tile structure).

### Phase 4: scale up

If Phase 3 wins, try mmq_y=64, mmq_x=16 for better tile reuse. Add need_check
masking for OOB tiles when seqLen not divisible by mmq_x.

## Out of scope this session

- Tensor cores / MMA path (sm_89 has them but DP4A is what we already use).
- Q6_K MMQ (ffn_down). Same approach but Q6_K's super-block is more complex.
- Full template parameterization. Hardcode mmq_y=32, mmq_x=8 first.

## Files

New: `Hesper/Layers/Q4KMMQ.lean` — kernel + helpers.
Edit: `Hesper/Layers/Linear.lean::forwardBatchDP4A_fromQ8` — add MMQ branch
behind `HESPER_PREFILL_MMQ=1`.
New: `Examples/DSL/Gemma4Q4KMMQParity.lean` — single-layer bit-parity test.

## Risk

Smem layout + threadIdx mapping is the bug-prone part. Plan to bisect by
disabling output write and printing partial sums for thread 0 if parity fails.
