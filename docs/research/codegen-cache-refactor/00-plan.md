# Plan: separate per-region CSE cache from path-independent register bindings

## Problem

`Hesper/CUDA/CodeGen.lean::stmtToPTX` lowers `ifStmt` (and `forLoop`) without snapshotting/restoring the CSE state. As a result, a register cached inside `thenBody` is only assigned on the threads that took the then-path; if the same `Exp` shape appears inside `elseBody`, the codegen reuses the cached register, but on else-path threads that register is unassigned and reads garbage.

A naive fix (snapshot+restore `expCache`, then drop after the join) was attempted at commit c63c305 and rolled back: it caused `CUDA_ERROR_INVALID_ADDRESS_SPACE` in `concat_dim0_f32_kernel`. The reason: `expCache` conflates two semantically distinct things —

1. **Per-region CSE keys** — "Exp `X` was computed; reuse register `r`" — only valid within the current dominator region.
2. **Register bindings** — once `r` is allocated in `freshU32`/etc., its number is path-independent (regs are static SSA names emitted in PTX header). Downstream code that *recomputes the same `Exp`* must produce the same `r` for the cache to be correct, OR allocate a fresh `r` and re-emit the arithmetic. Pure cache reset forces re-emit, which is correct in isolation — but if the recomputed `Exp` is large (buffer base + offset arithmetic), the caller may have stored the *result* in an outer var that the codegen then reads back. The mismatch between "register held by varMap entry" vs "register the recompute produced" causes wrong addresses → `INVALID_ADDRESS_SPACE`.

Actually re-reading: that's not quite right. Let me re-state cleanly.

## Why the naive fix failed (concrete)

In hesper PTX, `freshU32` etc. allocate a new register *number* each call. Two calls to `freshU32` give `%r10` and `%r11`. After PTX is emitted, both regs are listed in the function's `.reg` header.

`expCache` short-circuits redundant compute: if `Exp X` was lowered to `%r10`, the second time we see `X` we return `%r10` instead of allocating `%r11` and re-emitting the same arithmetic.

When you snapshot+restore `expCache` around an `if_`:
- Then-branch lowers `X → %r10` and caches it. `%r10` is the register that holds X for the then-path.
- Else-branch (after restore) lowers `X → %r11` (cache miss), and `%r11` holds X for the else-path.
- Post-if region lowers `X` again. With reset cache, it allocates `%r12` and **emits the arithmetic a third time**. This recompute uses path-independent inputs, so it's actually correct.

That's the correctness story. **So why did concat fail?**

Hypothesis (untested, since the failure was at runtime not at PTX emit): `expCache` is also keyed on Exps that contain `Exp.var "buffer_base"` references. For a u64 buffer base, the cache miss path does NOT allocate a new register — it does `lookupVar` against `varMap`, which is path-independent. So that path was correct.

But `expCache` may also cache compound Exps containing reads from variable assigns (`Stmt.assign`). After the if-branch, an assign inside one branch invalidates the cache (existing `.assign` clears it). That's already handled.

The actual breakage may have been somewhere else — perhaps an interaction with `sregCache` or `immCache` which were left untouched. Need to bisect concretely.

## Investigation needed before implementing

Before re-attempting the fix, capture:
1. **PTX diff** between concat with naive fix (broken) and concat with hoist (working). What instruction changed in else-branch / post-if?
2. **Register allocation stats**: how many regs in each version? If the broken version is missing a register, that confirms a missed cache miss path.
3. **A minimal failing test**: a 1-buffer if/else kernel that breaks the same way. Today we only have concat (3-buffer, 2-input), which is large.

## Plan

Five tasks, small + measurable:

### Task A: minimal repro test (1 session)
- Write `Tests/CUDA/CUDAIfBranchCSEMicrotest.lean`: 1 input buffer `x`, 1 output `dst`, simple if/else where each branch computes `dst[i] = x[i] op_X` vs `dst[i] = x[i] op_Y` (no shared sub-exprs across branches BUT both touch `i`).
- Run with current CodeGen → should PASS (this is the baseline).
- Add a deliberate cross-branch shared sub-expression (e.g. `i*2` in both branches with different ops) → re-run → if PASS, our hypothesis is wrong; if FAIL, we have a deterministic minimal repro.
- Commit the test (always-PASS version) as a regression sentinel.

### Task B: PTX-level instrumentation (1 session)
- Add `HESPER_DEBUG_CSE=1` env var in `expToPTX`: when set, log every cache hit + miss with the Exp key and the register returned, both to stderr.
- Run the minimal repro with cache hit logging, with the broken naive fix re-applied locally (do not commit). Capture: which Exp had a cross-branch hit that should have been a miss?
- This pins down whether the bug is `expCache`, `sregCache`, or `immCache`.

### Task C: scoped cache refactor (1-2 sessions)
- Add `cacheStack : List CacheSnapshot` to `GenState` where `CacheSnapshot = { exp : List ..., sreg : List ..., imm : List ... }`.
- Add helper `pushCache`, `popCache` ops. `pushCache` saves current cache layer; `popCache` discards the top layer + restores the one below.
- Rewrite `stmtToPTX` for `ifStmt`:
  ```
  pushCache
  lower thenBody
  popCache  -- discards then-only cache entries
  pushCache
  lower elseBody
  popCache
  ```
- For `sregCache`/`immCache`: leave them on `GenState` directly (path-independent, kernel-invariant — safe to share).
- For `expCache`: that's the only one that needs scoping.

### Task D: validate against full test matrix (1 session)
- Run `scripts/regression.sh` (43 tests must remain green).
- Run `cuda-concat-dim0-vs-llama` with offset compute INSIDE branches (revert the user-side hoist) — must PASS.
- Run `cuda-geglu-quick-vs-llama`, `cuda-permute-4d-vs-llama` — must PASS.
- Run a Gemma 4 decode end-to-end (`gemma4-cuda data/... "Hello" 30`) — must produce the expected token sequence.

### Task E: simplify call sites (1 session)
- Now that hoist is no longer needed, simplify `concat_dim0_f32_kernel` back to the natural shape (offsets inside branches).
- Update `feedback_if_branches_hoist_offsets.md` to mark the workaround as no longer required.
- Update inline comments in `Hesper/Layers/Vision.lean`.

## Total scope

**5-6 sessions** if everything goes smoothly. Each task is small enough to commit independently, and Tasks A+B answer the prerequisite question ("what exactly is breaking?") before we attempt the fix.

## Safety notes

- Never commit the naive fix without a regression test that catches its failure mode (which is why Task A comes first).
- If Task C is mid-way and breaks something, we can revert `CodeGen.lean` and the user-side hoist remains as the workaround.
- `Tests/Transpile/CUDARmsNormFullSmoke.lean` and similar bigger-kernel parity tests under `Tests/CUDA/` are the canary. Always run a few before saying "looks fine".
