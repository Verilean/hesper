# 39 — PTX rename reveals: Q6_K lm_head is 88% of decode wall time

*Written 2026-04-23.  After the PTX-symbol rename (commit `35da713`)
made every kernel visible to nsys, one kernel jumped to the top
and dwarfs everything else.*

## What the rename revealed

nsys summary, hesper graphs-OFF, 5 forwards (1 prefill + 4 decodes):

| Kernel                   | Grid              | Per-call   | Total   | Identity            |
|--------------------------|-------------------|-----------:|--------:|---------------------|
| **k_6587556783081778**   | **(262144, 1, 1)**| **51.1 ms**| 255.6 ms| **Q6_K lm_head**    |
| k_1433627607784000       | (10240, 1, 1)     |     60 µs  |  20.3 ms| Q4_K ffn_gate/up    |
| k_4973880045649474       | (10240, 5, 1)     |    170 µs  |  14.3 ms| prefill ffn gate/up |
| k_7031743127946451       | (640, 1, 1)       |     67 µs  |  12.6 ms| Q4_K wK/wV combined |
| k_1114518596643478       | (2560, 5, 1)      |    173 µs  |   3.6 ms| prefill Q4_K        |
| k_1257906884153183       | (2560, 1, 1)      |     39 µs  |   3.3 ms| Q4_K wO / ffn_down  |
| k_3461724857804993       | (42, 1, 1)        |    417 µs  |   1.7 ms| ?                   |
| k_3149194083895788       | (2048, 1, 1)      |     14 µs  |   2.0 ms| Q4_K wQ             |
| everything else          | various           |       —    | ~20 ms  |                     |

**Per decode, lm_head alone = 51.1 ms.**  Wall time per decode = 58 ms.
**That's 88% of the decode wall time spent in ONE kernel.**

## Why it's so slow

`Hesper/Models/Gemma4/LlamaForwardPrefill.lean:914` dispatches lm_head as:

```lean
GPUBackend.execute ctx
  (Hesper.Quantization.Q6_K.fusedQ6KLinearKernel hidden vocab)
  [("weights", model.outputWeight), ("input", resultNormBuf),
   ("output", logitsBuf)]
  { numWorkgroups := (vocab, 1, 1),        -- vocab = 262144
    workgroupSize := { x := 256, y := 1, z := 1 } }
```

That's **262,144 workgroups × 256 threads = 67 M threads per call**,
one output row per workgroup.  Each workgroup does a hidden=2560
reduction via 256 threads cooperating.

Problems:
1. **Under-utilised workgroups**: 256 threads for a 2560-element
   reduction = 10 elements/thread — memory-bandwidth bound but each
   thread does a serial walk.  llama.cpp's `mul_mat_vec_q<Q6_K>` uses
   4 rows per workgroup with shared-mem Q8_1 cooperation.
2. **Massive kernel launch overhead from 262 K WGs**: even at ~1 µs
   scheduler overhead per wave, 262 K WGs / (60 SM × 48 max-WG per
   SM) = ~91 waves = ~90 µs of pure launch-latency cost.  Small
   compared to 51 ms but still wasted.
3. **No multi-row tiling**: weights are re-read for every of the 262 K
   output rows independently, destroying any L2 residency.

## Expected win from fixing lm_head

llama.cpp's equivalent Q6_K mul_mat_vec_q at vocab=262144 runs in
~2.15 ms/call (42.99 ms / 20 decodes from the nograph bench).

If we port their multi-row pattern to hesper's fusedQ6KLinearKernel
and get close to 2 ms/call:

- Per-decode: 58 ms → **~9 ms** (since 51 ms of lm_head → 2 ms)
- TPS: 11.2 → **~70 TPS** projected

That's **the single biggest TPS lever** in the codebase right now.

## What to change

`Hesper/Quantization/Q6_K.lean` (`fusedQ6KLinearKernel` definition):
adopt llama.cpp's `mul_mat_vec_q<Q6_K, 1, 4>` pattern — 4 rows per
workgroup, 128 threads (4 warps × 32), Q8_1 input cached in shared
memory and reused across 4 rows.

llama.cpp reference: `llama.cpp/ggml/src/ggml-cuda/mmvq.cu` —
`mul_mat_vec_q<QK_K, VDR_Q6_K, QI6_K, ...>`.  The `vdr=1` +
`nwarps=4` + `ncols_dst=1` template specialisation is the decode-hot
shape.

Hesper probably already has a `fusedQ6KLinearDP4A4RowKernel` from
earlier work — check `Hesper/Layers/Linear.lean:3802-3809`.  The
routing logic at `forwardDP4A` uses it for `outDim % 4 == 0` — which
262144 satisfies.  So why isn't it active here?

**Bug**: `LlamaForwardPrefill.lean:914` calls
`fusedQ6KLinearKernel` directly instead of going through
`Hesper.Layers.Linear.forwardDP4A`, so the 4-row optimisation is
bypassed.  Easy fix: re-route through the Linear wrapper OR swap
directly to `fusedQ6KLinearDP4A4RowKernel`.

## Action

1. Confirm `fusedQ6KLinearDP4A4RowKernel` is correct for vocab=262144
   (`262144 % 4 == 0` ✓).
2. Change `LlamaForwardPrefill.lean:914` to use that kernel (same
   signature, drop-in swap).
3. Re-measure.  If lm_head per-call drops from 51 ms to ~5 ms (still
   behind llama.cpp but 10× better), wall time drops 58 → ~15 ms =
   **60+ TPS**.

This eclipses the doc 36 Q4_K priority by a wide margin.  Q4_K total
was 8 ms of the 58 ms budget; lm_head is 51 ms.
