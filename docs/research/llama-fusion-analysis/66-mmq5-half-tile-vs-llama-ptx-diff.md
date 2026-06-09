# MMQ5 half-tile vs llama mmq_x=64: PTX structural diff

Date: 2026-05-03

Follows up on the half-tile MMQ5 rewrite (`feedback_mmq5_default_blocked.md` updates 3+).
Earlier "PTX 25× larger" measurement was based on a wrong llama-side cut (treated the
whole `mmq_q4k.ptx` as one entry, but it actually contains every `mmq_x ∈ {8,16,24,…,128}
× need_check ∈ {0,1}` template instance). This doc establishes the correct comparison.

## Reproducible extraction

```bash
# llama side: extract entry _Z9mul_mat_qIL9ggml_type12ELi64ELb0EE (mmq_x=64, need_check=false)
sed -n '69628,77263p' /tmp/llamacpp_ptx/mmq_q4k.ptx > /tmp/llama_mmq_x64.ptx

# hesper side
HESPER_DP4A=1 HESPER_PTX_DUMP=/tmp/hesper_ptx_mmq5 HESPER_PREFILL_MMQ5=1 \
  lake exe gemma4-cuda data/gemma-4-e4b-it-Q4_K_M.gguf \
    "Please explain Turing machines briefly in plain English for a beginner who has never heard the term before today." 3

# Identify MMQ5 instances by signature: 4247 lines, 64 dp4a, 2 bar.sync
for f in /tmp/hesper_ptx_mmq5/*.ptx; do
  dp4a=$(grep -c "dp4a" "$f"); bar=$(grep -c "bar\.sync" "$f")
  if [ "$dp4a" -ge 50 ] && [ "$bar" -eq 2 ]; then echo "$(wc -l < $f) $f"; fi
done
```

## Side-by-side metrics

| metric | hesper MMQ5 (half-tile) | llama mmq_x=64 | hesper / llama |
|---|---:|---:|---:|
| PTX lines | **4,247** | **7,636** | **0.56×** |
| PTX bytes | 129 KB | 245 KB | 0.53× |
| dp4a (explicit) | 64 | 0 | — |
| inline-asm blocks | 0 | 1,148 | — |
| ld.shared | 240 | 356 | 0.67× |
| st.shared | **2** | 111 | **0.02×** |
| ld.global | **2** | 83 | **0.02×** |
| bar.sync | **2** | **13** | **0.15×** |
| `.reg .b32` decls | 1 (RegArray) | 1 | — |
| `$L_*` runtime loops | 0 | 0 | — |

## What the metrics mean

### PTX size is no longer the bottleneck
hesper PTX is now **smaller** than llama's. The earlier narrative ("hesper 25× larger
→ JIT cost dominates → static unroll is wrong") was built on the wrong llama-side
baseline. After the half-tile rewrite, PTX size is 0.56× of llama and JIT cost
should be lower than llama's, not higher. The 937 ms cold-PTX-JIT we measured for
hesper MMQ5 is **roughly proportional to PTX bytes** (129 KB), not pathologically slow.

### The real structural gaps
The four metrics where hesper deviates >5× from llama:

1. **`bar.sync`: 2 vs 13 (6.5× fewer in hesper)**
   - llama runs a multi-stage pipeline: load X stage 1, sync, load X stage 2 + start
     compute on stage 1, sync, etc. — overlapping global→smem transfers with the dp4a
     compute on the previous stage's data.
   - hesper runs a flat schedule: cooperative-load all X+Y → one big sync → all dp4a
     → final sync → write. Memory and compute don't overlap.

2. **`ld.global`: 2 vs 83 (40× fewer in hesper)**
   - llama streams from global through L1 in many small chunks per K iteration
     (83 over the whole K loop = ~2 per K-step at K-iters≈40). Some chunks land
     in registers directly without going through smem.
   - hesper does 2 monolithic global reads at the start (cooperative X tile + Y tile),
     then everything happens in smem.

3. **`st.shared`: 2 vs 111 (55× fewer in hesper)**
   - Same root cause as #2. llama writes back to smem in many small chunks across the
     compute stages (multi-stage pipeline). hesper has one batched store.

4. **inline-asm blocks: 0 vs 1148**
   - llama emits dp4a, mma.sync, and load instructions as raw inline-asm to give ptxas
     finer scheduling control. hesper goes through ShaderM CodeGen's dp4a primitive,
     which produces standard `dp4a.s32` ops. Likely a small effect compared to #1-3.

## Action implications

The dominant remaining gap (warm 1.6 ms/tok hesper vs 0.25 ms/tok llama, ≈6.4×) is
**memory-pipelining-bound**, not size-bound or instruction-mix-bound:

1. **Top action — multi-stage cp.async pipeline.** hesper currently does a single
   monolithic `cooperative-load → barrier → compute → barrier` loop. llama overlaps
   the next K-iteration's global→smem with the current K-iteration's compute via
   `cp.async` (sm_80+) and 13 barriers per K outer iteration. Estimated impact: 2-3×.
   - Requires: `Inst.cp_async` in `Hesper/CUDA/PTX.lean` + `Exp.cpAsync` + ShaderM
     wrapper. ~200 lines of CodeGen + ~250 lines of kernel rewrite. 1-2 sessions.
   - Unblocks a category of work: any future MMQ-style kernel can use the same
     pipeline pattern.

2. **Second action — register-resident accumulators per K-iter.** llama keeps the
   per-K dp4a partial sums in registers and only transfers between registers and
   smem at K-stage boundaries. hesper materializes everything into smem. Estimated
   impact: 1.2-1.5×.
   - Requires: rework the inner dot to treat acc as register-only across the inner
     8 jIter × 2 iIter unroll, not smem-staged. Already mostly the case post-half-tile;
     audit needed.

3. **Third action — inline-asm dp4a clusters.** Cluster 4-8 dp4a's with manual
   register packing into one `asm volatile` block. Estimated impact: <1.2×.
   - Requires: `Exp.inlineAsmDp4aCluster` or a peephole pass that fuses neighbouring
     dp4a's into a single asm block.

4. ~~**Reduce PTX size further (runtime loop conversion).**~~ Not necessary. hesper
   PTX is already smaller than llama's. The "Lean static unroll → runtime loop"
   refactor (~250 lines, 2-3 sessions) would not improve perf and may hurt by
   reducing ptxas's optimization scope.

## Key takeaway for next-session work

**Focus the kernel rewrite on the memory pipeline, not on instruction-mix or unroll
strategy.** Specifically: target `cp.async` + multi-stage barriers as the next concrete
step. The half-tile rewrite has done its job (shrunk PTX, validated parity, kept warm
TPS at MMQ2 baseline) — the next 6.4× requires touching how the kernel moves data
through the memory hierarchy, not how it lays out instructions.

## Progress checklist

- [x] **PTX Inst variants landed** (`Hesper/CUDA/PTX.lean`):
  - `Inst.cp_async_cg_shared_global (smemAddr : RegU32) (globalAddr : RegU64) (bytes : Nat)`
  - `Inst.cp_async_commit_group`
  - `Inst.cp_async_wait_group (n : Nat)`
  - All three render to correct PTX strings; `lake build gemma4-cuda` clean.

- [ ] **Exp + ShaderM API design** (open):
  - Need to decide: should `cpAsync` be an `Exp (.scalar .u32)` returning unit
    (like `workgroupBarrier`), or a pure ShaderM side-effect like `barrier`?
    Workgroup-barrier is currently both (`Exp.workgroupBarrier` exists, plus
    `ShaderM.barrier` wraps it).
  - smem-address operand: take from `Exp.sharedAddr (sym : String)` (lowers via
    `mov_shared_addr`)? Or threaded through the existing `sharedNamed` declaration
    machinery?
  - global-address operand: cp.async wants u64 ptr-to-element. Existing
    `readBuffer` returns *typed values* via `Exp.index`, not raw addresses.
    Need new `Exp.bufferAddr (buf : String) (idx : Exp (.scalar .u32))` that
    yields a u64 address without dereferencing.
  - **Auto-CSE pass safety**: cp.async is a side-effect. The auto-CSE pass
    (`#293`) hoists pure expressions; cp.async with the same `(smem, global)`
    operands at different K-iters is *not* CSE-able. Verify that hoist-prevention
    triggers for it (probably needs a side-effect tag in `Exp`).

- [ ] **Smoke test** (open): standalone kernel — copy 1 KB from global → smem via
  3 stages of cp.async + commit + wait_group, then read back. Bit-parity vs the
  equivalent `ld.global; st.shared` sequence.

- [ ] **MMQ5 multi-stage pipeline rewrite** (open): wrap the K-outer loop with
  prefetch (next-iter cp.async) + commit + wait(N=1) + compute (current-iter
  data). Target 13 bar.sync, matching llama. ~250 lines kernel, careful parity
  check on all 4 dispatcher shapes.
