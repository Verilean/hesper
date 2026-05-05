# Release prep plan

## Context

`origin/main` is 631 commits behind `feature/f16-kv-cache` (the working branch). The working tree has ~40 uncommitted deletions (transpile cleanup leftovers) and many untracked artifacts. We need:

1. A clean, releasable state on a release branch.
2. CHANGELOG documenting the major themes since the last release.
3. Triage on which experimental code stays / goes / gets feature-gated.
4. Build verification on platforms without CUDA (current CI surface unknown).

## Major themes since last release (preview from `git log`)

The 631 commits cluster into:

- **Gemma 4 production**: CUDA backend, MMQ Q4_K matmul (multiple iterations), Q6_K lm_head, FlashAttn V11, RMSNorm, fused kernels, Circuit DSL, IRv2, Monolith bundling, mmap GGUF loader, CUDA Graphs default ON, decode TPS path → ~95-100 TPS.
- **Vision/Audio (NEW)**: im2col, conv2d, conv_transpose_1d, GEGLU_QUICK, CONCAT, PERMUTE 4D — all with byte-parity vs llama.cpp.
- **Transpile experiments (REMOVED)**: 600+ lines under `Hesper/Transpile/CUDA/` were deleted in the cp.async cleanup phase. Working-tree shows the deletions still uncommitted.
- **WMMA (Tensor Core) prototype**: Phase 1-4c landed but not wired into production.
- **CodeGen**: scoped-cache fix for if_ branches (just landed today).

## Tasks

### REL-A: stage uncommitted deletions + clean working tree
Survey the 40 `D` lines in `git status` and the untracked top-level junk (`._hesper-ttt`, `debug_tiled_matmul.ptx`, `elf.o`, `tmp_decode_vs_prefill.py`, `data/gguf/`, `dawn/`, `hesper-ttt/`, `hesper-ttt.tgz`, `llama.cpp/`, `node_modules/`, `package*.json`, etc.). For each:
- transpile/* deletions → commit (already finalised by abandonment memo).
- root-level binary / debug dumps → `.gitignore` entries + delete.
- `llama.cpp/` (vendored test harness) — decide: keep as submodule, vendor properly, or move out.
- `dawn/` (build cache?) — likely `.gitignore`.
- `data/gguf/` and `hesper-ttt*` — model data should not live in repo; `.gitignore`.

### REL-B: triage experimental code
Audit each path under `Hesper/`, mark each file as one of:
- **production** (Gemma 4 inference path): keep.
- **experimental but useful** (e.g. WMMA, Circuit DSL v2): keep + feature-gate or document as "experimental, not in default decode path".
- **dead** (unreachable from any current `lean_exe` target): remove.

Likely candidates for deletion:
- `Hesper/TTT/*` (TTT model wasn't shipped — verify reachability).
- `Hesper/LlamaPath/*` if it exists (LlamaPath v1 was superseded by v2).
- old IRv2 PoC scaffolding if duplicated.
- `Examples/Gemma4LlamaPrefillSkeleton.lean` (the modified one) — is it production or scratch?

### REL-C: video/audio production-readiness review
Goal: decide whether `Hesper/Layers/{Vision,Audio}.lean` are "shipped" or "preview".
- Vision (`im2col`, `matmulF32Naive`, `conv2d`, `concat_dim0`, `permute_4d`, `geglu_quick`): all parity-tested vs llama.cpp byte-for-byte. Wired into NO Gemma 4 production path yet. Decision: ship as "preview / experimental" with a docs note.
- Audio (`conv_transpose_1d`): parity-tested. Single kernel; rest of audio family is blocked. Same: "preview".
- Update `README.md` (or create) to clarify status.

### REL-D: CUDA-less build verification
Lakefile already has `if !(← cudaLib.pathExists)` guards. Verify by:
- `nix-shell --run "lake build hesper"` on a no-CUDA shell (or temporarily hide cuda libs).
- Ensure all `lean_exe` that depend on `libhesper_cuda.a` are clearly marked.
- Check Mac case: `cuda_bridge.cpp` is Linux-only per line 145 comment, but the lakefile may still try to link `-lcuda`. Either:
  - Refuse to build CUDA exes on darwin (lakefile guard).
  - Move CUDA-dependent exes into a separate `lakefile-cuda.lean`-style sub-package.
- Document the matrix: "what builds on what platform".

### REL-E: regression suite covers + CHANGELOG draft
- `scripts/regression.sh` currently runs 43 tests. Verify all CUDA-dependent ones are in this set, and that pure-Lean tests (if any) run separately on Mac.
- Draft `CHANGELOG.md` with sections per theme (Gemma 4 production, Vision/audio kernels, CodeGen, etc.).
- Tag: `v0.X` after merge.

### REL-F: branch + PR
- Squash the 631-commit feature branch? No — too much history loss. Keep commits, merge as a single PR with the CHANGELOG as the merge-commit description.
- Create release branch `release/v0.X` based on cleaned state.
- Open PR `feature/f16-kv-cache → main` once REL-A through REL-E are done.

## Sequencing

```
A (clean tree) ── B (code triage) ── C (V/A review) ── D (no-CUDA build) ── E (CHANGELOG) ── F (PR)
```

Each step gates the next. A and B are mechanical; C is a docs decision; D is the substantive verification; E is writing.

## Estimated effort

- A: 1 session (mechanical)
- B: 1-2 sessions (audit + delete)
- C: 1 session (docs only)
- D: 1-2 sessions (test platforms; may need adjustments)
- E: 1 session (writing)
- F: 1 session (assembly + PR)

**Total: 6-8 sessions** to land a clean release.
