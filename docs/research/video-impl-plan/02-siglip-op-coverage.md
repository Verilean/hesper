# Gemma 4 SigLIP encoder op coverage map

Generated 2026-05-05 by running `llama-mtmd-debug -p encode --image cb -n 128`
on `data/mmproj-gemma-4-e4b-it-f16.gguf`.

## Encoder structure (from graph trace)

- 16 transformer blocks (FLASH_ATTN_EXT × 16, GEGLU_QUICK × 16)
- ~7 norms / layer = 112 RMS_NORM + 1 final norm = 113
- 7 matmuls / layer (Q/K/V/wO/gate/up/down) = 112 + projector head = 114
- RoPE applied per layer (Q + K + maybe a sub-warp split, ~4× per layer = 64)
- Output: pooled tokens + projector → **2560-dim embeddings**

**Gemma 4 SigLIP uses RMSNorm, NOT LayerNorm.** This was a surprise — most CLIP/SigLIP models use LayerNorm. Gemma 4's variant uses RMSNorm everywhere (consistent with the LLM side).

## Op coverage vs hesper

| ggml op | count | hesper status |
|---|---:|---|
| CLAMP                  | 226 | ⚠ inline pointwise (Exp.min + Exp.max), no dedicated kernel — easy |
| MUL_MAT                | 114 | ✅ many variants (f16/Q4_K/Q6_K) ready |
| RMS_NORM               | 113 | ✅ `Hesper/Layers/RMSNorm.lean` production-ready |
| MUL (binary)           | 96  | ✅ pointwise mul |
| RESHAPE                | 69  | ✅ logical-only (no kernel) |
| VIEW                   | 66  | ✅ logical-only |
| ROPE                   | 64  | ✅ `Hesper/Layers/RoPE.lean` |
| PERMUTE                | 49  | ⚠ requires cpy + index permute |
| ADD                    | 34  | ✅ binary pointwise |
| CPY                    | 32  | ⚠ f32↔f16 etc — used in Gemma 4 decode |
| CONCAT                 | 32  | ❌ no hesper kernel |
| GEGLU_QUICK            | 16  | ❌ activation x * sigmoid(1.702*x) * gate, no dedicated impl |
| FLASH_ATTN_EXT         | 16  | ✅ V11 (works for decode + prefill) |
| CONT                   | 4   | ⚠ memory-layout fixup, same as cpy |
| TRANSPOSE              | 3   | ⚠ logical, but PERMUTE+CONT pattern |
| SCALE                  | 2   | ✅ pointwise mul-by-const |
| GET_ROWS               | 2   | ✅ embedding lookup pattern |

## Verdict: what's missing

### Must build for end-to-end SigLIP

1. **GEGLU_QUICK kernel** (1 layer FFN activation, 16× per encode pass)
   - Algorithm: `out[i] = (gate[i] * sigmoid(1.702 * gate[i])) * up[i]`
   - Similar to existing GELU code in `Linear.lean:2001` but uses sigmoid * x (not 0.5*x*(1+tanh)).
   - Effort: ~40 LoC kernel + parity test, 1 session.

2. **CONCAT kernel** (32× per encode)
   - Concatenate two tensors along an axis. May be used to merge patch + position
     embeddings or layer outputs.
   - Effort: ~50 LoC, 1 session.

3. **PERMUTE/CPY for arbitrary axis order** (49 + 32 = 81× per encode)
   - hesper Gemma 4 KV cache write uses `scatter` (custom layout). For SigLIP we
     need a generic 4D permute. Effort: medium, ~80 LoC.

4. **CLAMP fused with surrounding ops** (226× — most ops!)
   - Soft-capping pattern: `clamp(x, -cap, +cap) * scale`. Already a hot
     path. Best handled as inline `Exp.min/Exp.max` — fold into matmul
     epilogue or pointwise chain. Effort: small if framework supports.

### What SHOULD work as-is

5. **MUL_MAT**, **RMS_NORM**, **MUL**, **ADD**, **ROPE**, **FLASH_ATTN_EXT**,
   **GET_ROWS**, **SCALE** — already in hesper for Gemma 4 LLM side.

### What's unexpectedly absent

- No CONV_2D in graph! Patch embedding is **a single MUL_MAT** with input
  pre-reshaped to `[n_patches, n_embd*patch_h*patch_w]`. **My im2col port
  is not used by Gemma 4 SigLIP.**
  → still useful for other CLIP/ViT variants, but not the immediate target.
- No LayerNorm! All RMS norm.
  → my "LayerNorm needed" hypothesis from earlier was wrong.

## Estimated effort to "Gemma 4 image input works in hesper"

Assuming hesper Gemma 4 LLM works (it does):

1. **GEGLU_QUICK** kernel (~40 LoC + test) — 1 session
2. **CONCAT** kernel (~50 LoC + test) — 1 session
3. **PERMUTE** kernel for 4D tensors (~80 LoC + test) — 1 session
4. **clamp fused into matmul epilogue** — 1 session (use existing pointwise
   fusion pass)
5. **SigLIP encoder driver in Lean** stitching the 16 blocks together
   (model loader for mmproj GGUF + forward) — 2-3 sessions
6. **Image preprocessing** (224×224 RGB → patch-reshape tensor; resize +
   normalize). Needs `stb_image.h` or similar via FFI — 1 session
7. **End-to-end parity test**: run an image through hesper SigLIP, dump 2560-dim
   embedding, compare against `llama-mtmd-debug` golden — 1 session

**Total: ~6-8 sessions.**

## Recommended sequence

```
1. GEGLU_QUICK kernel + parity vs ggml ── 1 session
2. CONCAT kernel + parity                ── 1 session
3. PERMUTE kernel + parity                ── 1 session
4. SigLIP encoder forward (driver Lean)   ── 2-3 sessions
5. Image preprocessing                    ── 1 session
6. End-to-end golden vs llama-mtmd-debug  ── 1 session
```

After step 6, hesper will accept a real PNG/JPG and produce embeddings
that **byte-for-byte (or within FP noise) match llama.cpp's** for the
same input — which is the actual "image works" milestone.
