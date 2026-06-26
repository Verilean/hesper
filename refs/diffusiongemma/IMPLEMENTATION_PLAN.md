# DiffusionGemma — Revised Implementation Plan

Branch: `feat/diffusiongemma` (18 commits). Model: `diffusiongemma-26B-A4B-it-Q4_K_M.gguf`.
Architecture spec: `refs/diffusiongemma/ARCH_NOTES.md`. Reference: `~/git/llama-dg` (llama.cpp PR #24423), ggml at `~/git/llama.cpp/build`.

## 0. Current state — DONE & VALIDATED
- **Architecture fully decoded** (Gemma4 backbone + bidirectional attn + diffusion decode + self-cond; see ARCH_NOTES).
- **Loader**: 16.8 GB → Metal as a reused `Gemma4Model` (`Hesper/Models/DiffusionGemma/Loader.lean`); optional-V global layers, dual scales, tied LM head.
- **CPU reference** (`Reference.lean`) + tiny test — validated oracle.
- **Every compute kernel validated on Metal vs ggml/CPU** (maxAbsErr ≈ 0): RMSNorm, Q4_K, Q6_K, **Q8_0** (new), **Q4_K-expert / Q8_0-expert** (new, dynamic expertIdx), GQA attention core, softmax, RoPE, GeGLU, softcap. Parity exes: `diffusiongemma-*-parity`. Dumpers: `scripts/llama_parity/dump_dg_*`.
- **Milestone 1 — GPU-resident forward skeleton** (`DiffusionGemmaForwardGPU.lean`): preallocate buffers once + `beginBatch`/`endBatch` (single submission) + ping-pong → **30-layer loop runs on Metal, NO crash**; dense skeleton finite at 3 layers. This fixed the per-op-dispatch resource-accumulation crash.
- Reference generation works: `~/git/llama-dg/build/bin/llama-diffusion-cli` → coherent text, 54 tok/s.

## 1. Hard constraints (lessons — do NOT repeat)
1. **No per-op host round-trips.** They accumulate GPU dispatch resources → crash ~layer 3. Forward MUST be GPU-resident, one `beginBatch`/`endBatch`.
2. **Gemma4's forward is NOT Metal-reusable.** `generate`, `forwardPrefillBatch`, AND `forwardSingleToken`/`forwardBlock` all throw "CUDA is not available" — the orchestration (InferenceState/KV-cache/stream) is CUDA-coupled even though kernel dispatches default to WGSL. Don't try to call them.
3. **Don't hand-roll kernels for the batch.** Reuse Hesper's validated batched kernels through the framework's dispatch mechanism.

## 2. CRITICAL PATH — Phase 0: crack the batch-dispatch problem (BLOCKING, do first)
**Symptom:** in `beginBatch`, the `Layers.*` ops (`LinearLayer.forward`, `RMSNorm.forward`, `forwardNormThenAdd`) sequence correctly (skeleton finite), but adding a `geluMul` dispatch — even Gemma4's *proven* `geluMulKernel` via `executeWithConfigCached` — gives **NaN**. So a matmul→down chain works, but inserting a custom dispatch breaks it.

**Untested hypotheses (try in order):**
- (a) I reused ONE `gpref` across all 30 layers → geluMul REPLAYS (layers 1-29). The working `Layers.*` ops used a FRESH per-layer ref (record every layer). **Test geluMul with a fresh `IO.mkRef none` per layer** (match the skeleton pattern). ← most likely.
- (b) Missing inter-dispatch barrier between the geluMul write (`sGeglu`) and the `down` read. Check whether the WebGPU backend inserts barriers per dispatch, and whether mixing fresh-ref (Layers.*) and replay-ref (geluMul) dispatches in one batch is the issue.
- (c) Study **`BitNet.forward` + `TransformerBlock.forward`** (`Hesper/Models/BitNet.lean:280`, `Hesper/Layers/TransformerBlock.lean`): how does it sequence its block's dependent dispatches (norm→attn→FFN→geluMul) in one batch correctly on Metal? Replicate that exact mechanism (it works — BitNet runs on Metal). Likely a shared `CachedLayerBuffers` + consistent dispatch path.

**Done when:** a GPU-resident block `RMSNorm→gate/up→geluMul→down→postFFNNorm+residual` gives **FINITE** logits at 30 layers (today it's NaN with geglu).

## 3. Phase 1 — real block (GPU-resident, one batch)
On the Phase-0-fixed pattern, build `forwardBlock` GPU-resident:
- **Attention** (start seqLen=1: attn_out = V → GQA-broadcast → wO; need a small broadcast op via the framework mechanism) + `postAttnNorm` + residual (`forwardNormThenAdd`).
- **Dense FFN**: `ffn_norm → gate/up → geluMul → down → post_ffw_norm_1`.
- **MoE**: router (rmsnorm-noscale × 1/√d × gate_inp_s → `ffn_gate_inp` → softmax top-8) — needs a GPU softmax+top-8 (reuse Gemma4's MoE routing kernels `moeRouterOut/Indices/Weights`, dispatched through the Phase-0 mechanism). Experts: validated `fusedQ4KMExpertKernel`/`fusedQ8_0ExpertKernel` (dynamic expertIdx). `pre_ffw_norm_2`/`post_ffw_norm_2`.
- Combine dense+MoE → `ffn_post_norm` → +residual → ×`out_scale` (canvas) / `enc_out_scale` (prompt).
- **lm_head**: tied Q6_K; TILE the dispatch (vocab 262144 > WebGPU 65535 workgroups/dim).
- **Done when:** finite 30-layer logits on the real model.

## 4. Phase 2 — correctness gate (bidirectional canvas forward)
- Implement the **bidirectional batched forward** over `[prompt | canvas]` (NOT seqLen=1; diffusion-gemma is non-causal): region-aware mask (prompt-causal / canvas-bidirectional), region embeddings (canvas = rmsnorm-noscale of scaled embed), per-token dual scale.
- Generate golden full-forward logits: `~/git/llama-dg/build/bin/llama-diffusion-gemma-eval <model> <prompt_ids.i32> <canvas_ids.i32> <out_logits.bin>`.
- **Validate** Hesper logits vs golden (this is the real correctness gate).

## 5. Phase 3 — diffusion decode + tokenizer
- Entropy/confidence decode loop (canvas of mask_token=4, ~13 entropy-bound steps, top-k anneal, remask). Mirror `~/git/llama-dg/examples/diffusion`.
- gemma4 SentencePiece tokenizer (encode prompt, decode output).
- **Done when:** native Hesper end-to-end text generation on Metal.

## 6. Phase 4 — performance (targets: effective 200 / in-step 1000 TPS @ ~13 steps = ~3.7× llama.cpp)
- **Batch all 256 canvas positions per dispatch** (batched matmul kernels — the big win; also drastically fewer dispatches).
- Fused kernels (RMSNorm+matmul, gate+up+gelu), subgroup/blockcoop variants, F32 router kernel, batched expert kernels.
- Step count FIXED = reference (~13); achieve targets by KERNEL OPTIMIZATION only.
- TPS harness reporting both effective and in-step throughput.

## Key files
- Reuse pattern: `Hesper/Models/BitNet.lean:280` (forward), `Hesper/Layers/TransformerBlock.lean`, `Attention.lean`, `MoE.lean` (batched; BitLinear — study the dispatch mechanism, not the weights).
- Batched kernels to reuse: `geluMulKernel` (`Gemma4/Kernels.lean:26`), Gemma4 MoE kernels; `ce` helper + `KernelCacheRefs` (`Gemma4.lean:476`).
- Mine (validated): expert kernels in `Hesper/Layers/Linear.lean`; `Reference.lean` (oracle); golden dumpers + parity exes.
- Gotchas: `GPUBackend.execute` buffer list must use `::`/`List.nil` (not `[...]` — `(expr)[...]` parses as index under `open Hesper.WebGPU`).

## Phase 0 — progress log
- Hypothesis (a) FRESH per-layer ref for geluMul: TESTED → still NaN. Ruled out.
- geluMulKernel + dispatch verified CORRECT (1 thread/elem, buffers gate/up/output, dispatch ceil(ffn/256)×256). Kernel is not the bug.
- So root cause = (b)/(c): a batch sync/barrier or dispatch-mechanism subtlety when mixing a raw `executeWithConfigCached` op (geluMul) with the `Layers.*` ops inside one `beginBatch`. Gemma4 routes its ENTIRE forward through `ce`; mixing paths breaks it.
- NEXT: study `BitNet.forward` + `TransformerBlock.forward` to see how it sequences its block's geluMul-equivalent with matmuls in one batch (barriers? all-through-one-mechanism?), then route ALL custom ops through that. Likely: don't mix — build the whole block through one consistent ce-like dispatch wrapper + a shared CachedLayerBuffers, exactly like BitNet/Gemma4.

## Phase 0 — SOLVED (commit 3a480a0)
ROOT CAUSE: NOT batching/binding/sync. The gelu tanh-approx `tanh(0.798*(g+0.0447*g³))`
feeds huge arguments for large gate values, and **Metal's tanh = (exp(2x)-1)/(exp(2x)+1)
= Inf/Inf = NaN** for large x (doesn't saturate). The geglu parity test used small inputs
so never triggered it. FIX: clamp the tanh argument to [-10,10] (tanh(±10)≈±1, negligible err).
Debug path that pinned it: single-input copy works → both inputs individually finite →
`g + 0*u` finite (binding/sync fine) → only the gelu *expression* NaNs → dumped WGSL (correct)
→ tanh overflow.  Real dense FFN now GPU-resident, 3-layer finite.
ACTION ITEM: audit ALL tanh-based kernels for Metal overflow — esp. **logit softcap**
(`scale*tanh(x/scale)`); clamp `x/scale` similarly. Also the real GeGLU kernel used in
Phase 1 must carry this clamp (don't reuse the unclamped Gemma4 geluMulKernel on Metal).
30-layer still overflows = dense-only skeleton (no attention/out_scale) — that's Phase 1, not a bug.

## Phase 1 — COMPLETE (commit, 23 commits)
Full real block runs GPU-resident in ONE batch on Metal, 30-layer → FINITE logits.
Block (seqLen=1): attn (V-reuse: wO(broadcastVNormK(wV(norm)))) + postAttnNorm residual;
dense FFN (clamped GeGLU); MoE [routerPrepK→routerMatVecK(F32)→top8K(shared-mem softmax+top-8
→idxs[8]+wts[8])→8×(fusedQ4KMExpertKernel gateup slot=e → gegluMergedK → fusedQ8_0ExpertKernel
down slot=e → waccK)→moePostNorm2]; combine addK(curMlp+curMoe)→postFFNNorm→+residual→×out_scale.
Expert kernels gained (paramsLen,slot) to read params[slot]=e-th top-8 index; parity unaffected.
KEY: dense-only diverged (3 finite/5 NaN) because model runs dense+MoE in PARALLEL every layer.
DISPATCH COUNT: ~38/layer ×30 ≈ 1140 — runs fine in one batch (no crash), but SLOW (router
prep/matvec have per-thread O(dim) reductions). Phase 4 will optimize (batch 256 positions, fuse).
NEXT: Phase 2 — validate logits vs llama-diffusion-gemma-eval. NOTE: current forward is
seqLen=1 single-position; the golden is the full bidirectional canvas, so Phase 2 needs the
batched bidirectional forward (all positions, region mask) — seqLen=1 won't match the golden.
argmax(slice)=0 currently (unvalidated synthetic input; lm_head sliced to 32768).
