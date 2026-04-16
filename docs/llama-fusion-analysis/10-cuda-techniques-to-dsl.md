---
title: "10 — llama.cpp CUDA techniques → Circuit DSL lowering plan"
date: 2026-04-16
status: draft
---

# llama.cpp CUDA techniques → Circuit DSL mapping

Companion to `docs/llama-fusion-analysis/09-hybrid-prototype.md`.  Phase 0
hybrid path confirmed llama.cpp's kernels are achievable on our hardware.
This doc catalogues the **techniques** those kernels use and maps each to a
concrete DSL step: expressible today / needs extension / structurally
incompatible.

Source survey: 54 techniques + 5 "surprise" patterns catalogued across
`llama.cpp/ggml/src/ggml-cuda/{mmvq,vecdotq,quantize,mmq,common,cp-async}.*`.
See survey notes below the mapping.

## Mapping principle

Every technique goes in exactly one of these buckets:

| Bucket | Meaning |
|---|---|
| **E** | **E**xpressible today using existing Circuit DSL primitives |
| **P1** | Phase 1 — new primitive needed, design clear, 1-5 days each |
| **P2** | Phase 2 — new **IR concept** needed (tile, async pipeline, tensor cores), 1-3 weeks |
| **R** | **R**untime or launch-config only; not a kernel-body concern |
| **N** | **N**ot applicable (backend-specific, e.g. AMD MFMA) |

The **one-by-one porting flow** is: pick one **P1** technique, add the
primitive, write a minimal GPU test that exercises it, land it.  Repeat.
P2s need design discussion before implementation.

## Mapping tables

### A. Data layout / bit manipulation

| # | Technique | DSL status | Notes |
|---|---|---|---|
| T-01 | unpack_ksigns (multiply-by-0x01010101 broadcast) | **E** | ScalarExp `.mul`/`.bitAnd` suffice |
| T-02 | `__byte_perm` chained for 4-bit nibble lookup | **P1** | New `ScalarExp.bytePerm a b sel` → PTX `prmt.b32` |
| T-03 | 12-byte 6-bit scales+mins unpack (`& 0x3f3f`) | **E** | `ScalarExp.bitAnd`, `.shr`, `.or` — already present |
| T-04 | MXFP4 interleaved nibble layout | **E** | Pure data layout, no compute change |
| T-05 | `__low2float(half2)` partial dequant | **E** | Already have `ScalarExp.f16ToF32` + bitcast u32→half2 low |
| T-06 | 5-bit quant reassembly (qh shifts) | **E** | ScalarExp shift/mask |
| T-07 | iq2_xxs grid table with XOR sign absorb | **P1** | Need `BufferHint.constTable` for ROM-resident grid tables |
| T-08 | Q3_K bit-shifted scale extraction | **E** | |
| T-09 | MMQ interleaved layout | **E** | Data layout |
| T-10 | `__vsubss4` (per-byte signed saturating sub) | **P1** | New `ScalarExp.vsubss4` → PTX `vsub.u32.u32.u32.sat.s32.s32` |
| T-11 | FP4 E2M1 exponent (e8m0) encode | **P1** | New `ScalarExp.floatToE8M0` + helper table |
| T-12 | `__byte_perm` for FP4 pack | **P1** | Same as T-02 |

### B. Compute

| # | Technique | DSL status | Notes |
|---|---|---|---|
| T-13 | dp4a core instruction | **E** | `ScalarExp.dp4a` exists (used by Q4_K/Q6_K forward) |
| T-14 | `dp4a(0x01010101, u, 0)` for Σq8 | **E** | Emerges naturally from existing `ScalarExp.dp4a` if CSE elides the constant load; if it doesn't, add `ScalarExp.sumBytes u` as sugar |
| T-15 | `ldmatrix` smem→reg tile load | **P2** | Requires "tile group" concept in IR (block-cooperative register file abstraction); not expressible with per-lane ScalarExp alone |
| T-16 | `mma.sync` 16×8×16 tile matmul | **P2** | Same: tile group + new Prim `Prim.mmaTile` |
| T-17 | AMD MFMA | **N** | AMD-specific; we target sm_89 |
| T-18 | AMD WMMA | **N** | AMD-specific |
| T-19 | VDR-templated vec_dot | **E** | Already achieved via pointwise unroll + CSE |
| T-20 | per-type vec_dot function pointers | **R** | Runtime dispatch; our generic `Prim.matmulQ4KWithEpilogue` + per-type lowering handles this |

### C. Memory access

| # | Technique | DSL status | Notes |
|---|---|---|---|
| T-21 | **`cp.async.cg` async smem load** | **P1** | **KEY**.  New `Prim.asyncCopyToSmem` + `Prim.asyncWait`; or ShaderM level-4 ops `asyncCopy` / `asyncWait`.  Enables software pipelining. |
| T-22 | L2 cache hints in cp.async (`.L2::256B`) | **P1** | Enum on `BufferHint` / on `asyncCopyToSmem` variant |
| T-23 | vectorized `float4` / `int4` loads | **E** | ScalarExp currently pessimises to scalar; add `BufferHint.vec4` to request 128-bit coalesced load |
| T-24 | `char4` quantized write | **E** | Same — vec4 store variant |
| T-25 | `__low2float` — see T-05 | **E** | |
| T-26 | Coalesced smem→reg via reinterpret | **E** | Using `ShaderM.sharedAlloc` + vec4 hint |
| T-27 | **fastdiv precompute uint3** | **P1** | Very useful: new `ScalarExp.fastdiv n (mp, L)` where `(mp, L)` are kernel params computed host-side.  Avoids division in inner loop (e.g. channel/batch decomposition in reduceScatterEpilogue-like patterns). |
| T-28 | Cooperative smem prefetch of Q8_1 | **E** | We already do this in Q4_K/Q6_K 4-warp kernels; existing `ShaderM.sharedAlloc` path |
| T-29 | **L2 persistent cache** | **R** | Already implemented — see `cuSetL2PersistLimit` / `cuSetL2AccessWindow` in Hesper/CUDA/FFI.lean |

### D. Reductions & sync

| # | Technique | DSL status | Notes |
|---|---|---|---|
| T-30 | warp_reduce_sum via `__shfl_xor_sync` | **E** | Level-2 `WarpExp.reduceSum` |
| T-31 | warp_reduce_sum via `__reduce_add_sync` (Ampere+) | **P1** | Hardware intrinsic, marginal gain; add as optional lowering for `WarpExp.reduceSum` when T is i32 + cc≥80 |
| T-32 | block_reduce (warp + smem + warp) | **E** | Level-3 `Prim.reduceScatterEpilogue` already does this |
| T-33 | warp prefix inclusive scan | **P1** | New `WarpExp.prefixSum` (butterfly via `__shfl_up`) |
| T-34 | `__syncthreads()` | **E** | ShaderM barrier |
| T-35 | cooperative_groups grid sync | **P2** | Cross-workgroup sync; huge DSL change; only needed for very-large softmax — defer |
| T-36 | warp-reduce alibi slope | **E** | Part of RoPE / positional bias — already composable |
| T-37 | `__shfl_sync` variants | **E** | Level-2 `WarpExp.shuffleXor` / `.shuffleUp` |

### E. Launch / scheduling

| # | Technique | DSL status | Notes |
|---|---|---|---|
| T-38 | calc_nwarps arch table | **R** | Launch config only; host side selects workgroup shape |
| T-39 | calc_rows_per_block | **R** | Same |
| T-40 | `should_use_small_k` heuristic | **R** | Per-shape autotune decision (our TODO-P4) |
| T-41 | MMVQ_PARAMETERS per-arch tables | **R** | |
| T-42 | `get_mmvq_mmid_max_batch` | **R** | |
| T-43 | `__launch_bounds__` | **P1** | New `ExecConfig.launchBounds := some (maxThreads, minBlocksPerSM)` — currently not emitted. |
| T-44 | 3D MMQ grid dims | **E** | ExecConfig.numWorkgroups is already 3-tuple |
| T-45 | quantize_mmq launch shape | **E** | |
| T-46 | fastdiv_values packing | **P1** | Tied to T-27 — kernel-param uint3 passed through launch config |

### F. Specialization

| # | Technique | DSL status | Notes |
|---|---|---|---|
| T-47 | `template <ggml_type>` | **E** | Our `CircuitM.matmulQ4K` / `.matmulQ6K` etc. already specialize by type |
| T-48 | `template <int ncols_dst, bool has_fusion>` | **E** | ncols_dst=1 for decode; has_fusion covered by `Prim.matmulQ4KWithEpilogue` |
| T-49 | `template <bool small_k>` | **P1** | New flag on our matmul Prims → different lowering (nwarps×rpb trade) |
| T-50 | Q8_1 layout enum (D4/DS4/D2S6) | **P1** | Part of MMQ path; not needed for decode; defer |
| T-51 | tile_A/B/C struct templates | **P2** | Comes with T-15/T-16 tile concept |
| T-52 | switch ncols_dst unroll | **E** | Lean `match` + specialization |
| T-53 | switch type unroll | **E** | |
| T-54 | has_fusion dispatch | **E** | Matmul-with-epilogue already does this |

## Bucket totals

- **E**  (expressible today):             27 techniques
- **P1** (new primitive, clear design):   14 techniques
- **P2** (new IR concept):                 4 techniques (tile + mma.sync + grid-sync)
- **R**  (runtime/launch):                 7 techniques
- **N**  (n/a, AMD-specific):              2 techniques
- **Total**: 54

> Most of the wins sit in **P1** — these are the one-at-a-time ports the
> user asked for.

## Porting flow (one-at-a-time loop)

Each P1 follows this 5-step loop:

1. **Specify**.  One paragraph in `docs/circuit-dsl-todos.md` pinning down
   the new primitive's signature, the PTX it emits, and what kernel pattern
   it enables.
2. **DSL extension**.  Add to `Hesper/Circuit/IR.lean` (new `ScalarExp`/
   `WarpExp`/`Prim` constructor) + case in `Hesper/Circuit/Lowering.lean`.
3. **Minimal GPU test** in `Tests/Circuit/` — one kernel exercises the new
   primitive in isolation, compared against a CPU reference.
4. **Apply to hot kernel**.  Pick the one production kernel that the new
   primitive unlocks (e.g. cp.async → ffn_down memory-latency fix).
5. **Measure**.  Per-kernel `ncu` before/after.  If no improvement, revert
   the production change but keep the DSL primitive (no cost to future
   users).

Estimated cycle: **2-4 days per P1**, with perf regression test as gate.

## P1 priority ordering

Ranked by (expected TPS gain) × (DSL complexity cost⁻¹):

1. **T-27 fastdiv primitive** — 2 days, used by *every* multi-dim index calc; +0-1 TPS alone, but unblocks clean lowering for future tiling passes.
2. **T-21/T-22 cp.async + L2 hints** — 4-5 days, enables software-pipelined ffn_down.  Target +5-10 TPS.
3. **T-43 `__launch_bounds__`** — 1 day.  Marginal occupancy boost.  Cheap to add.
4. **T-23/T-24 vec4 load/store hint** — 2 days.  Implicit via `BufferHint.vec4`. +1-3 TPS in Q8_1 quantize.
5. **T-02/T-12 `bytePerm`** — 2 days.  Opens IQ-family quant support (future models) + cleaner Q3_K unpack.
6. **T-10 `vsubss4`** — 1 day.  Q3_K inner loop speedup.
7. **T-31 `__reduce_add_sync`** — 0.5 day.  Drop-in replacement in `WarpExp.reduceSum` when T=i32 on sm_80+.  Free.
8. **T-14 `sumBytes` sugar** — 0.5 day.  Could also emerge from CSE of T-13 calls; worth checking first whether ptxas already emits the same instruction.
9. **T-33 `WarpExp.prefixSum`** — 2 days.  Needed for FlashAttention online softmax.
10. **T-49 small_k flag on matmul Prims** — 2 days. Covers the "K too small for usual occupancy" case; modest gain for attention projections.

## P2s (design first, implement later)

1. **Tile group concept (T-15/T-16/T-51)** — 2-3 weeks.  The biggest gap.
   Requires a new IR level (between Level 3 and Level 4) that owns a tile's
   register file + smem tile and exposes `loadFromGlobal`, `mma`, `store`.
   Prerequisite for real tensor-core matmul.  Prefill-blocking.
2. **Grid-wide sync (T-35)** — defer.  Only enables large-softmax via
   `cooperative_groups::this_grid()`; niche.

## Structural-difficulty analysis: what's actually hard?

The **hard** parts aren't the instructions — those are typed wrappers around
PTX, one-day primitives.  The hard part is the abstractions they need:

- **`cp.async`** changes from a synchronous `smem[i] = global[i]` statement to
  a *two-phase* operation (issue, await).  Our ScalarExp is statement-level
  and per-lane; async operations break that mental model.  The fix is to add
  two new `ShaderM` ops (`asyncCopy`, `asyncWait`) at Level 4 and have `Prim.
  matmulQ4KWithEpilogue` optionally lower via the pipelined pattern.  Not
  hard — just a second code path in the matmul lowering.

- **`ldmatrix` + `mma.sync`** is the only real architectural mismatch.  Our
  Level-4 ScalarExp describes *one lane's* computation; `mma.sync` is inherently
  an 8-lane (or 32-lane) cooperating instruction that exchanges values via
  `tile_A[i][j]` indexed across lanes.  We need a "tile group" abstraction:
  `BlockExp.defineTile T (i j k)`, `BlockExp.tileMMA a b c`.  The tile holds
  per-lane register slots but the programmer sees tile indices, not lanes.
  This is the biggest single DSL extension.  Until it exists, tensor-core
  kernels stay in hand-written ShaderM at Level 4.

- **Autotuning / launch-config tables** is a Phase-1 or Phase-2 question
  independent of kernel techniques — already tracked as TODO-P4.

Everything else is "add a new leaf to the AST, wire it through the printer,
write a GPU test."  No architectural mismatch.

## Appendix — full 54-technique survey

See `/home/junji-hashimoto/git/hesper-gemma4/docs/llama-fusion-analysis/11-technique-catalogue.md` (stub; agent report archived there if user wants the raw enumeration).
