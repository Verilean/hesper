# DiffusionGemma (diffusion-gemma, "Dg_Rc0P1_Patched") — verified architecture from GGUF

## Global / model-level
- arch=diffusion-gemma, 30 blocks, d_model=2816, vocab=262144
- token_embd [2816,262144] Q6_K — TIED (no separate output.weight)
- output_norm [2816] F32 ; rope_freqs [256] F32
- final_logit_softcapping=30.0 ; rms_eps=1e-6
- **attention.causal=False** ; **diffusion.canvas_length=256** ; mask_token_id=4
- SELF-CONDITIONING block (diffusion): self_cond_pre_norm[2816],
  self_cond_gate[2816,2112]Q4K, self_cond_up[2816,2112]Q4K, self_cond_down[2112,2816]Q6K  (GeGLU)

## Attention — DUAL geometry, per-layer
- head_count=16 (all layers)
- SWA layers (sliding_window_pattern=True): head_dim=256, kv_heads=8
    attn_q[2816,4096=16x256] attn_k[2816,2048=8x256] attn_v[2816,2048]Q6K attn_output[4096,2816]
    attn_q_norm[256] attn_k_norm[256]  (per-head RMSNorm)
- FULL/global layers (idx 5,11,17,23,29; pattern=False): head_dim=512, kv_heads=2
    attn_q[2816,8192=16x512] attn_k[2816,1024=2x512] attn_output[8192,2816]
    attn_q_norm[512] attn_k_norm[512]
    *** NO attn_v.weight ***  <-- UNRESOLVED: how is V formed for global layers?
- sliding_window=1024 ; rope freq 1e6 (full) / 1e4 (swa); rope dim 512/256

## Per-layer FFN — DENSE + MoE IN PARALLEL (every layer)
- Dense GeGLU: ffn_norm[2816], ffn_gate[2816,2112]Q4K, ffn_up[2816,2112]Q4K, ffn_down[2112,2816]Q8_0
- MoE 128 experts top-8: ffn_gate_inp[2816,128]F32 (+ .scale[2816]),
    ffn_gate_up_exps[2816,1408=2x704,128]Q4K, ffn_down_exps[704,2816,128]Q8_0 (+ .scale[128])
- Norms present: attn_norm, post_attention_norm, ffn_norm,
    post_ffw_norm, post_ffw_norm_1, post_ffw_norm_2, pre_ffw_norm_2
- Scales: layer_output_scale[1], enc_layer_output_scale[1]

## Decode: masked diffusion, bidirectional, no KV cache, 256-token canvas

## EXACT FORWARD (from llama.cpp PR #24423: src/models/diffusion-gemma.cpp + gemma4-common.h)
Baseline = "UNIFIED" single no-cache bidirectional forward over [prompt|canvas], zero self-conditioning.
Self-cond MLP and the PREFILL/DECODE prompt-KV store are OPTIONAL optimizations — skip for v1.

### Embeddings (region split at P = n_tokens - canvas_length)
inpL = tok_embd[tokens] * sqrt(n_embd)            # ScaledWordEmbedding
  prompt rows (q<P): use scaled embed as-is
  canvas rows (q>=P): inpL = rms_norm(inpL, eps)  # NO scale weight (zero-SC). (+self-cond if enabled)

### Per block il  (n_head=16; SWA: head_dim256/kv8; FULL idx5,11,17,23,29: head_dim512/kv2)
cur = rmsnorm(inpL, attn_norm)
# Q: Qcur = wq@cur; reshape[hd,n_head,T]; q_norm = RMSNorm(attn_q_norm[hd]); RoPE(freq full=1e6/swa=1e4,
#    partial n_rot, full layers use rope_freqs as freq_factors)
# K/V: Kcur=wk@cur; Vcur = wv@cur IF wv else Kcur(raw k_proj);  reshape[hd,kv,T]
#    K: k_norm=RMSNorm(attn_k_norm[hd]) then RoPE.  V: rms_norm(no scale), NO rope, NO k_norm.
cur = attn(wo, Q, K, V, kq_scale = f_attention_scale = 1.0)   # GQA; region-aware additive mask
cur = rmsnorm(cur, attn_post_norm)
attn_out = cur + inpL                                          # residual
cur = ffn_moe(attn_out)                                        # see below (all layers are MoE here)
# region-aware per-layer scalar (1-elem):
cur *= (prompt rows: enc_out_scale ; canvas rows: out_scale)
inpL = cur

### ffn_moe(attn_out)  — dense shared-expert + 128-expert MoE in PARALLEL
cur_mlp = rmsnorm(attn_out, ffn_norm)
cur_mlp = GeGLU(cur_mlp; gate,up,down; act=GELU, parallel)     # down(gelu(gate@x) * (up@x))
cur_mlp = rmsnorm(cur_mlp, ffn_post_norm_1)
cur_moe = rmsnorm(attn_out, ffn_pre_norm_2)
# router logits on attn_out (custom): tmp=rmsnorm_noscale(attn_out); tmp*=1/sqrt(n_embd);
#   tmp*=ffn_gate_inp_s[n_embd];  logits = ffn_gate_inp@tmp  -> [128]
cur_moe = moe_ffn(cur_moe, experts gate_up fused, top_k=8, softmax gating, GELU)
cur_moe = rmsnorm(cur_moe, ffn_post_norm_2)
cur = cur_mlp + cur_moe
cur = rmsnorm(cur, ffn_post_norm)
cur = cur + attn_out                                           # residual

### Final
cur = rmsnorm(inpL, output_norm)
logits = output @ cur            # output tied to tok_embd (TENSOR_DUPLICATED)
logits = softcap * tanh(logits / softcap)   # softcap=30

### Region-aware attention mask (UNIFIED square [T,T], additive 0/-inf)
P = n_tokens - canvas_length.  q_is_canvas = q>=P.
  prompt query (q<P): causal over prompt only (k<P and k<=q), SWA-clipped on sliding layers; never canvas.
  canvas query (q>=P): bidirectional. global layer: all prompt+canvas.
     sliding layer: all canvas + prompt positions k >= P-(n_swa-1).
hparams: causal_attn=false, f_attention_scale=1.0, n_swa=1024(sliding_window), swa_type=STANDARD.

### Diffusion decode loop (examples/diffusion + common_params_diffusion)
default 128 steps; top-k anneal (top_k_start..top_k_end); entropy/confidence remask
(t_min,t_max,entropy_bound,stability_threshold,confidence_threshold); canvas=256; mask token id 4.
