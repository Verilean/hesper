# Changelog

## v0.7 — Gemma 4 (2026-05-05)

The headline release: **Gemma 4 inference on CUDA**, ~95-100 TPS decode on
RTX 4070 Ti against the Q4_K_M quantization. ~635 commits since v0.6
(BitNet/M4-Max). The major themes below.

### Highlights
- **Gemma 4 (`google/gemma-3-4b-it`) end-to-end** on the CUDA PTX backend.
  Ships as `lake exe gemma4-cuda` with mmap GGUF loader, Q4_K_M weights,
  Q6_K lm_head/ffn_down, FlashAttention V11, and CUDA Graphs by default.
- **CUDA PTX backend** (new). The WGSL/Vulkan path that BitNet shipped on
  is still supported; CUDA is the new tier-1 path. A common
  `GPUBackend` typeclass abstracts both.
- **Circuit DSL + IRv2** (new). Two-layer DSL above ShaderM:
  Circuit (Prim graph) and IRv2 (BlockGraph). Both compile down to the
  same PTX/WGSL emit path. Used internally for kernel fusion.
- **Vision/audio kernel previews** (new, not on Gemma 4 LLM path). im2col,
  conv2d, conv-transpose-1d, GEGLU_QUICK, CONCAT, PERMUTE 4D — all with
  byte-for-byte parity vs llama.cpp's CPU output. Foundation for an
  upcoming SigLIP encoder + audio decoders.

### Backend / runtime
- New `Hesper/CUDA/` family: PTX codegen, FFI bridge (`native/cuda_bridge.cpp`),
  module load + cubin disk cache, CUDA Graphs capture/replay.
- `GPUBackend` typeclass refactor: every Lean call site now reads
  `ctx : GPUBackend.Ctx`; CUDA and WebGPU implementations both satisfy it.
- mmap GGUF loader: `HESPER_USE_MMAP=1` keeps the file mapped, FFI-managed
  finalizer; saves ~700 ms cold-start and ~2.8 GB RSS.
- VRAM pool with power-of-2 free-list (round threshold 1 MB);
  steady-state VRAM 8189 → 5561 MiB.
- Per-Layer-Embedding (PLE) on-demand row fetch + UVA host-mapped path:
  the 2.2 GB Q6_K table stays on CPU mmap, 1-row H2D per token. Further
  drop to 3435 MiB.
- Cubin disk cache (`~/.cache/hesper/cubin`): cold JIT 48 ms → warm 34 ms.

### Decode + prefill performance (Gemma 4)
**Decode**: ~95-100 TPS, RTX 4070 Ti, graphs ON. The major levers were:

- **CUDA Graphs default ON** (commit `34c85a3`): graphs OFF 56.3 TPS →
  graphs ON 76.5 TPS (+20 TPS).
- **Q6_K lm_head pre-dequant to packed half2** at load time +
  f16 matmul: graphs ON 67.6 → 80.9 TPS (+13.3, +20%).
- **Q4_K f16 native cvt**: replace fp16ToF32 arithmetic decode
  (15-op) with `cvt.f32.f16` (`unpack2x16float`, 2-op) across all 12
  Q4_K dp4a kernel sites. 96.4 → 104.5 TPS in long-prompt decode.
- **Q6_K 4-warp cooperative reduction fix**: corrected cross-warp merge.
  Default ON path: +25% TPS.
- **FlashAttention V11**: V7 softmax + V8 sub-warp partition + split-K +
  128-bit LDG.E.128/LDS.128 + register+shuffle vkq aggregation. Wired
  into production with f16 K/V cache. Final: 26→7.74 µs (3.4× journey).
- RMSNorm: workgroup=1024 (matches llama.cpp); 0.93 → 0.46 ms/dec.
- On-device argmax with host-mapped result: closes the 9.8 ms DtoH
  bubble per token.

**Prefill**: MMQ tile-GEMM (Q4_K + Q6_K), llama.cpp shape:
- MMQ2 (smem-staged X tile, mmq_y=32, mmq_x=8): 1.12-1.17× faster
  baseline-vs-prior across seqLen 16-40.
- MMQ5 (full llama shape, mmq_y=128, mmq_x=64, X+Y smem): half-tile
  variant landed; default behind `HESPER_PREFILL_MMQ5` flag (kept off
  because warm-time profile is compute-bound; see
  `project_mmq5_warm_bottleneck.md`).
- Q6_K batched prefill (`q6kMatmulBatchKernel`): 1239 → 21 dispatches,
  -39% prefill at seqLen=59 combined with MMQ2.
- cp.async (sm_80+) primitives + MMQ7 multi-stage prefetch: landed,
  parity OK, but warm perf negative due to compute-bound profile.
  Kept as infra (`Inst.cp_async_*`); MMQ6/7/8 kernels removed in cleanup.

### Kernel fusion (Circuit DSL + IRv2)
- Prim taxonomy: `pointwise`, `reduceLastAxis`, `matmulQ4K`, `scatter`
  (unified write/v-cache), `view + writeSlice`, `reduceScatterEpilogue`,
  `matmulQ4KWithEpilogue`. All have lowering passes to ShaderM.
- Fusion passes: `fusePointwise`, `fuseReduceIntoQuantize`,
  `fuseMatmulEpilogue`, `fuseWriteDestination`, `mergeSameDispatch`.
- IRv2 (Phase B-F): pure-data BlockGraph + Block-scope reductions.
  Layer-0 monolith parity vs production decode (max |err| = 0).
  Currently preview, not on hot path; will graduate when fusion passes
  match production hand-fused dispatch count.
- ShaderM ergonomics (Steps 4-9): lane/warp helpers
  (`laneId`/`warpId`/`subWarpSplit`), `warpReduceSum`,
  `Ptr ty`/`MutPtr` abstraction, LICM, comparison operators (`<ᵉ ==ᵉ`),
  `softmaxOnline`, `warpBarrier`.
- Auto-CSE pass on Exp tree (`Hesper/CUDA/CodeGen.lean`); 2026-05-05
  fix for if_ branch leakage (sreg/imm/exp caches now scoped per branch).

### Codegen / DSL
- `Inst.dot4I8Packed` (dp4a), `Inst.fma_rn_f16x2`, `Inst.cp_async_*`,
  `Inst.subgroupMatrix*` (WMMA m16n16k16 fp16/fp32), `Inst.bfe_u32`,
  `Inst.subSatS8x4` (`__vsubss4`), `Inst.ld_u8`/`u16`.
- Exp: `dot4I8Packed`, `unpack2x16float`, `subSatS8x4`, `bufferAddr`,
  `toF32U`, `subgroupShuffleXor` family.
- `BufferHint.readOnly` → `ld.global.nc` (read-only cache).
- `ShaderM.scope`: emit `{ }` block scope for register reuse +
  block-scoped `.reg` declarations.
- WMMA Phase 4b: runtime `ShaderM.loop` with in-place register rebind for
  `c_frag`. PTX 1569 chars (vs 3139 for 4-iter unroll), max |err|=0.0.
- KernelStub extractor (`tools/StubExtract/`): generates 3 views —
  prefill (52 kernels), audio (9), video (10) — from llama.cpp source.
  Production non-connected; aids stub triage.

### Tokenizer + sampling
- BPE/SentencePiece tokenizer fix + interactive mode + repetition penalty.
- `Hesper/Inference/Sampling.lean`: argmax / temperature / top-k.

### Quality / testing
- 26-test regression suite (`scripts/regression.sh`); ~3-5 min on RTX
  4070 Ti when builds are cached. Stops at first failure or
  `--continue`. Strengthened detection: catches non-zero exit codes
  + "unknown executable" silent passes (caught 17 silently-passing
  transpile entries that were removed).
- 9 Gemma 4 layer-0 parity tests against real GGUF weights, all max
  |err| = 0.0.
- llama.cpp PTX execution path (loader + arg packing + launch helpers)
  for direct hesper-vs-llama PTX comparison.

### Removals / cleanups
- `Hesper/Transpile/CUDA/` (CUDA→ShaderM transpiler) — abandoned 2026-05-01.
  Architecture mismatch: needed macro-extraction, not detail lowering.
  Reverted Goal B (Stages 2-5) when prefill failed to hit baseline.
  Production untouched; no public API change.
- MMQ6, MMQ7, MMQ8 kernels (cp.async-pipelined Q4_K matmul) — removed
  after warm-time profile showed compute-bound, not memory-bound. cp.async
  Inst variants remain.
- Dead transpile lean_exe entries from `lakefile.lean` (~17 entries).
- `Examples/` cleanup: `BitNet*` moved to experimental status, gemma4
  parity tests promoted under `Examples/DSL/`.

### Repository hygiene (release prep)
- `Hesper/Layers/Vision.lean` + `Audio.lean` marked PREVIEW status with
  in-file headers explaining what's needed for full-path graduation.
- `docs/release-prep/01-module-status.md` — every `Hesper/` module
  declared as production / preview / experimental / infra.
- `docs/release-prep/02-platform-support.md` — Linux+CUDA tier-1;
  macOS/no-CUDA documented as docs-only.
- `lakefile.lean`: 81 hardcoded CUDA link literals centralised into
  `cudaExeArgs` (no-op on Linux; one-line change for future macOS).
- `.gitignore` covers macOS metadata, build artifacts, vendored deps,
  node tooling, model data.

### Known limitations (carried forward)
- **macOS / no-CUDA build**: lib builds, exes won't link. Stub library
  not implemented this release. See `docs/release-prep/02-platform-support.md`.
- **Tensor Core (mma.sync int8)**: WMMA fp16 wired (Phases 1-4c
  landed); int8 mma.sync blocked on PTX 8.7 vs driver 565
  (`feedback_nvcc_ptx_version_vs_driver.md`). Tracked in tasks #338-339.
- **MMQ5 default-off** in prefill: parity OK but 3.3× slower than MMQ2
  on RTX 4070 Ti (`feedback_mmq5_default_blocked.md`). MMQ2 is the
  default; MMQ5 reachable via `HESPER_PREFILL_MMQ5=1`.
- **Vision/audio**: parity-tested but not on any production path.
- **TPS ceiling without int8 Tensor Cores**: warm decode is compute-bound
  on the dp4a chain. Reaching llama.cpp's ~119 TPS likely requires
  mma.sync (`project_mmq5_warm_bottleneck.md`).

### Migration notes
- Targeting BitNet?  v0.6 path still works; CUDA backend is additive.
- Targeting Gemma 4?  `lake exe gemma4-cuda <gguf-path> <prompt> <max-tokens>`
  is the entry point. `HESPER_CHAT=1` for IT chat-template wrapping.
- Lake target list grew. Notable new exes: `gemma4-cuda`,
  `gemma4-{qproj,qkv,ffn,postffn,kv,k,kv-multi,ropeq,q4k-mmq}-parity`,
  `cuda-{matmul,bitlinear,fa-golden,bitnet-golden,...}-test`,
  `wmma-gpu-parity-test`, `transpile-cuda-mmq-q4k-microbench`.

### Acknowledgments
This release builds on llama.cpp's CUDA backend (used as ground-truth
reference and via direct PTX execution for parity tests).
