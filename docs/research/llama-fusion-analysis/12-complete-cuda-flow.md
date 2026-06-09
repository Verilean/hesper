---
title: "12 — llama.cpp CUDA complete flow (Gemma 4 E4B, RTX 4070 Ti)"
date: 2026-04-18
status: reference
---

# Complete llama.cpp CUDA flow for Gemma 4 E4B

Source analysis spanning `llama.cpp/src/{llama-kv-cache,llama-graph,models/gemma4-iswa}.cpp`
and `llama.cpp/ggml/src/ggml-cuda/{fattn*,mmvq,mmq,rope,norm,unary,cpy,quantize}.cu`.

Reference compile: `build/bin/llama-bench -m gemma-4-e4b-it-Q4_K_M.gguf -p 512 -n 128`
gives **pp512 = 8327 tok/s, tg128 = 119 TPS** on RTX 4070 Ti (sm_89).

Target for hesper: reproduce the same numerical results (bit-level where
possible) and the same performance envelope.

## 1. KV cache

### 1.1 Physical layout

Per layer the K tensor is allocated in
`llama-kv-cache.cpp:197-198, 1132-1150`:

```
type_k = F16 (default via llama-context.cpp:2905)
shape  = [n_embd_k_gqa, kv_size, n_stream]
       = [numKVHeads * headDim, context_length, 1]   -- Gemma 4 E4B: [512, ctx, 1]
stride = row-major (standard GGML)
```

View at attention time (`get_k`, line 1144):

```
ggml_view_4d shape = [headDim, numKVHeads, kv_len, n_stream]
                   = [256, 2, kv_len, 1]
```

### 1.2 V layout: transposed iff FA is off

`llama-kv-cache.cpp:1152-1182`:

- **FA enabled (default for Gemma 4):** V is **not** transposed.  Shape
  matches K: `[numKVHeads*headDim, kv_size, 1]`.
- **FA disabled:** V is physically stored as
  `[kv_size, numKVHeads, headDim, 1]` to make mat-vec over the cache
  cache-friendly during decode.

**Implication for hesper:** with FA enabled we can use the same row-major
`[numKVHeads, max_seq, headDim]` layout llama.cpp uses — no special-case
per-layer transpose needed.

### 1.3 Cache writes: `ggml_set_rows`

Both prefill and decode use **the same path**:

- K write: `ggml_set_rows(k_view, k_cur, k_idxs)` (line 1216)
- V write: `ggml_set_rows(v_view, v_cur, v_idxs)` (line 1272; for v_trans
  the shape is `[1, numKVHeads * n_tokens]` element-wise scattered)

`k_cur` is `K_proj` already reshaped and RoPE'd.  The indices
`k_idxs` / `v_idxs` are a small u32 buffer indicating where each of the
N new tokens goes into the ring buffer.

**At the ggml op level**, RoPE and set_rows are **distinct** ops.  But in
the CUDA backend (`ggml_cuda_op_rope_fused`, `rope.cu:3766`) the RoPE
kernel can write directly to the output indexed by row, effectively
fusing RoPE+set_rows into **one dispatch** when the ggml graph has the
pattern `RoPE → VIEW → SET_ROWS`.  This is one of the key fusions we are
missing.

### 1.4 SWA layers

`llama-hparams.h:299`: `is_swa(il)` reads a bitmask set from GGUF
`gemma4.attention.sliding_window_pattern`.

**llama.cpp does NOT allocate a smaller cache for SWA layers.**  Both
full and SWA layers use the same context-length cache; the SWA window
is applied **only as a dense mask** at attention time
(`is_masked_swa()`, hparams.h:316-343).

### 1.5 Gemma-4-specific

From `models/gemma4-iswa.cpp`:

1. Q, K, **and V** all get RMS-norm (lines 60/81/82).  V has no
   dedicated scale weight — pure RMS.
2. Q-norm and K-norm are applied **before** RoPE (line 87 does RoPE on
   the normed tensors).
3. `f_attention_scale = 1.0` (model loader at `llama-model.cpp:4262`) —
   no pre-softmax scaling (unlike Gemma 2).

## 2. Flash attention

### 2.1 Kernel selection

`ggml-cuda/fattn.cu:299-498` dispatch for Gemma 4 E4B
(headDim=256, K/V=F16, Q=F32, sm_89):

| batch | best_kernel | config |
|-------|-------------|--------|
| prefill (Q.ne[1]≥8) | `MMA_F16`, `DKQ=256 DV=256 ncols=8` | 128 threads, nbatch_fa=64, occupancy 2 |
| decode  (Q.ne[1]=1) | `MMA_F16`, ncols=1                  | same tile, 1 query per block |

Both paths go through `template-instances/fattn-mma-f16-*` templates.
There is no vector or tile kernel used for Gemma 4 E4B on sm_89 — MMA
wins because head_dim=256 passes the divisibility checks and Turing+
tensor cores are available.

### 2.2 MMA prefill geometry

`fattn-mma-f16.cuh:469-750`:

- **One CUDA block** processes `ncols=8` query tokens × `nbatch_fa=64`
  KV positions per iteration; it streams over the full KV range.
- Tile load from K/V uses `ldmatrix` into tensor-core fragments; the
  matmul uses `mma.sync.m8n8k4` (fp16 operands, fp32 accum).
- **Online softmax**, two-pass: per K-tile, update rowmax, rescale the
  running rowsum, apply to `VKQ`.  Reduction is warp-level via
  `__shfl_xor_sync` (`fattn-mma-f16.cuh:669`).
- Smem: ≈ 20 KB/block.  Occupancy 2 on sm_89 (96 KB/SM).

Mask is a **dense F16 tensor** pre-filled by `build_attn_inp_kq_mask`
(`llama-graph.cpp:2365-2384`) — SWA and causal both go through the same
`mask[jt, kt]` lookup with `-inf`-encoded masking.

### 2.3 Why it's fast

Measured dominance (based on file-level instruction patterns):

1. Tensor cores do both `Q × K^T` and `attn × V` (two MMA passes).
2. `ncols=8` means one K load is shared across 8 Q rows — amortises the
   memory bandwidth on the K cache by 8×.
3. SMEM K/V reuse: tile loaded once, both matmul passes read it.
4. No explicit SWA branch — the dense mask folds SWA into the same
   causal path, so the kernel is one hot path.

## 3. MatMul

### 3.1 Dispatch

`ggml-cuda.cu:2309-2320`:

| M (tokens) | kernel family | file |
|------------|---------------|------|
| M = 1..8   | `mul_mat_vec_q` (MMVQ) | mmvq.cu:391 |
| M > 8      | `mul_mat_q` (MMQ)       | mmq.cu:74  |

Both operate on Q4_K × Q8_1; src1 (f32 activation) is quantised to Q8_1
first.

### 3.2 Q8_1 quantize

Standalone kernel: `quantize_q8_1` (`quantize.cu:5`, 256 threads/block,
QK8_1=32).  **NOT fused** with the matmul — it runs as its own
dispatch before each matmul.  Scratch buffer is allocated once per
forward pass and re-used across matmuls in that pass.

### 3.3 Q4_K × Q8_1 inner loop

`vecdotq.cuh:502-524`, `vec_dot_q4_K_q8_1_impl_vmmq`:

- VDR = 2 → each thread processes 2 × (int of 4 Q4_K nibbles) × (int of
  4 Q8_1 bytes).
- For each of the `QR4_K = 4` sub-blocks: 2 `dp4a` calls producing
  `dot1` (quant × quant) and `dot2` (quant × running-sum, for the min
  subtract).
- Per Q4_K block (256 elements): 8 dp4a + 4 dequant-scale extractions.
- Final contribution: `dm.x * Σ(d8ᵢ * dot1ᵢ * scᵢ) − dm.y * Σ(d8ᵢ * dot2ᵢ * mᵢ)`.

### 3.4 MMVQ epilogue fusion

`mmvq.cu:435-580` has a template parameter `has_fusion` that enables:

- Optional `x_bias` addition.
- Optional **gate ∘ up** fusion: if the matmul is one of the two halves
  of a SwiGLU/GeGLU, the kernel can compute the gate mat-vec **inside
  the same kernel** and apply the activation, so the epilogue yields
  `gate_act(gate) * up` directly.

This is the `ffn_up + ffn_gate + activation + mul` → one dispatch path
we currently miss in hesper's dense FFN.

### 3.5 MMQ prefill

`mmq.cu` uses `MMQ_ITER_K = 256` tile sizes; both X (Q4_K) and Y (Q8_1)
are loaded into smem once per K-tile and reused across all M rows in
the block.  Tensor cores (mma.sync) on Volta+.

## 4. One transformer layer: dispatches

Walking through `llm_build_gemma4_iswa` (lines 30-226) and mapping each
ggml op to CUDA kernels:

| # | op (ggml) | CUDA kernel | notes |
|---|-----------|-------------|-------|
| 1 | RMS_NORM (attn_norm) | rms_norm fused (norm.cu) | |
| 2 | MUL_MAT (Q) | quantize_q8_1 + mmq/mmvq | 2 dispatches |
| 3 | RMS_NORM (Q-norm) | rms_norm_fused (maps mul) | 1 |
| 4 | ROPE_EXT (Q) | rope | 1 |
| 5 | MUL_MAT (K) | quantize_q8_1 + mmq/mmvq | 2 |
| 6 | RMS_NORM (K-norm) | rms_norm_fused | 1 |
| 7 | ROPE_EXT (K) → SET_ROWS | **rope_fused writes cache directly** | **1 combined** |
| 8 | MUL_MAT (V) | quantize_q8_1 + mmq/mmvq | 2 |
| 9 | RMS_NORM (V) → SET_ROWS | norm + scatter (may fuse) | 1–2 |
| 10 | FLASH_ATTN_EXT | fattn-mma-f16 | 1 |
| 11 | MUL_MAT (O) | quantize_q8_1 + mmq/mmvq | 2 |
| 12 | RMS_NORM+MUL+ADD (post-attn+res) | **rms_norm_fused_add** | **1 combined** |
| 13 | RMS_NORM (ffn_norm) | rms_norm_fused | 1 |
| 14 | MUL_MAT (up) + MUL_MAT (gate) + GEGLU | **mmvq with has_fusion** | **1–2 combined** |
| 15 | MUL_MAT (down) | quantize_q8_1 + mmq/mmvq | 2 |
| 16 | RMS_NORM+MUL+ADD (post-ffn+res) | rms_norm_fused_add | 1 |

Count (conservative, with all fusions enabled): **~20 dispatches / layer / token**.

× 42 layers + ~10 per-layer embedding + final norm + lm_head ≈ **~860
dispatches / token** for a basic decode path.  llama.cpp reports ~200
after CUDA Graphs eliminate the boilerplate in-graph copies — we didn't
audit the graph-capture snapshots but the high-water mark before
capture is consistent with this.

### 4.1 Fusions llama.cpp has that hesper doesn't (yet)

Ranked by dispatch savings per layer:

1. **ROPE + KV-cache write** (`rope_fused`, rope.cu:3766).  Hesper has a
   per-token version but not the batched prefill variant merged in.
2. **RMS_NORM + MUL + ADD** (`rms_norm_fused_add`, norm.cu:3995).  Used
   twice per layer (post-attn, post-ffn).
3. **GEGLU split fused with matmul epilogue** (mmvq.cu:435-580 with
   `has_fusion=true`).  Turns 3 dispatches into 1–2.
4. **Multi-ADD fusion** (`fused_add`, ggml-cuda.cu:3797).  Chains up to
   7 add ops; residual add chains benefit.
5. **CUDA Graphs** (`ggml_cuda_graph_*`).  Captured on stable
   (shape, seq-len); re-launched on subsequent tokens of the same
   shape.  Primary decode TPS win.

## 5. CUDA Graphs

`ggml-cuda.cu:2930-3102, 3540-3682`:

- Entry: `ggml_cuda_graph_compatibility` decides whether capture is
  possible (rejects split buffers + dynamic shapes).
- Capture is keyed by graph pointer + shape hash; **re-captures on
  sequence-length changes**.
- On replay: `cuGraphLaunch` instead of per-op `cuLaunchKernel`, cutting
  the ≈ 1.2 µs host overhead we pay per dispatch.
- Streams: single default stream, with optional named streams for
  `concurrent_events` regions only.

For hesper's decode (stable shape after prefill), this is the single
biggest easy win — task #96 was already marked completed but the
analysis says we still have 1040 dispatches/tok, so either it's not
wired into the current Gemma 4 path or it's re-capturing every token.
Worth re-checking.

## 6. Where to go next

This doc is enough to design Phase C (hesper v2 implementation) and
Phase D (llama.cpp-parity rewrite).  Specifically:

- KV cache layout in hesper: switch to `[numKVHeads, max_seq, headDim]`
  F16 to match llama.cpp.  Rope must produce the same numerical values,
  and cache writes must land at the same byte offsets.
- Attention kernel: port MMA-F16 pattern, ncols=8 prefill / ncols=1
  decode.  Requires fp16 tensor-core intrinsics in ShaderM (or a direct
  PTX emission path).
- MatMul: we already have a dp4a Q4_K kernel; the epilogue fusion path
  (bias + gate + activation) is missing.
- Decode-critical fusions: rms_norm_fused_add, rope+cache-write,
  geglu_split.  These together should move us from ~860 to ~200
  dispatches/tok.
- CUDA Graphs: verify whether the current codepath actually captures,
  and if not, wire it.

The v2 Circuit DSL design (`Hesper/CircuitV2/IR.lean`) should be
evaluated against each of these targets — if any require primitives v2
can't express, that's a design fix before we commit to the rewrite.
