# WGSL Codegen Optimization — closing the WGSL/Tint-vs-hand-MSL gap

**The principle (proven, not assumed):** our DSL emits WGSL → **Tint** translates it to MSL → Apple's
**Metal compiler** produces the binary. Tint is a *transpiler that performs NO optimizations* (official
Dawn stance, Corentin Wallez: it only applies correctness transforms + polyfills + robustness; the
2.5-year IR work was about *translation speed*, not output quality). The Metal compiler *does* optimize —
but only the "simple", low-risk passes. **So the quality of the generated MSL is a direct function of the
WGSL WE emit.** We generate the kernels, so we own their performance. There is no MSL-only feature and no
"binary magic": llama.cpp, our hand-MSL (`native/metal_replace.mm`), and our WGSL path all feed the SAME
Metal compiler MSL *source*. A measured gap between them is purely WGSL-generation quality.

**Do NOT jump to a native-MSL backend.** We measured the q4k gate/up kernel at 1.37× hand-MSL and closed
most of it (→1.15×) with a one-line-per-guard WGSL change. The gap is closeable in the DSL.

---

## What Metal RECOVERS (do not waste time "fixing" these in WGSL)

Metal's compiler reliably does these; changing the WGSL to pre-do them measures NULL. **Metal's optimizer
is strong on *local* transforms — it recovers most emitted-code verbosity.** Two verified null results:

- **Strength reduction** — `a / 2^k` → `>>`, `a % 2^k` → `&`. Tint emits guarded integer div/mod (231 div
  + 151 mod in the q4k kernel!) but Metal inlines, const-props the literal divisor, folds the div-by-zero
  `select`, and strength-reduces. Emitting `>>`/`&` from the DSL measured **0 change**. (Verified 2026-07.)
- **SROA of constant-indexed local arrays** — `array<subgroup_matrix_result, 4>` accessed `Cx[0..3]` with
  the create-temp + write-back pattern (H2). Converting to 4 named scalar fragments *eliminated the
  `tint_array` from the MSL* but measured **0 change** (6.08→6.10 ms) — Metal SROAs the opaque-type array
  into registers and copy-props the temp+write-back away. So array-vs-named-register accumulators do NOT
  matter on Metal.
- **Constant folding**, basic **dead-code elimination**, **function inlining**, **copy propagation**.

## What Metal does NOT recover (THE lever — fix this in WGSL)

The ONE local transform Metal reliably fails at is **CSE of a subexpression duplicated across a large
function body** — everything else it recovers (above). So the single high-value WGSL fix is:

- **Common-subexpression duplication from Lean-`let` (THE lever).** In the ShaderM DSL, `let x := expr` is
  a *Lean* binding = **substitution**: every use of `x` re-emits the whole `expr` tree into the WGSL. A
  loop-invariant guard reused across fragments/blocks expanded **57×** in the q4k MSL, and Metal did NOT
  CSE the repeated (read-only) buffer-load + compare. **Fix:** bind reused / loop-invariant subexpressions
  with `ShaderM.let'` (emits a WGSL `let x = expr;`, computed once). q4k: 7.22→6.08 ms, 1.37×→**1.15×**,
  bit-identical. This is *systemic* — every kernel that reuses a Lean-`let` value has it. This one class
  accounts for essentially all of the recoverable codegen gap on Metal (the div/mod and accumulator-array
  "pathologies" are visible in the MSL but cost nothing — Metal recovers them).
- **Robustness bounds-clamps** (separate axis, ~30% of the *raw* gap). Removable via the
  `HESPER_DISABLE_ROBUSTNESS` env gate (OFF by default — enabling globally makes OOB *undefined behavior*
  for any unguarded kernel; DG decode kernels are DSL-guarded so it's safe there).

**⇒ The playbook is narrow and cheap: hunt duplicated subexpressions (esp. loop-invariant guards and
buffer reads reused via Lean-`let`) and bind them with `let'`.** Don't bother rewriting div/mod, unrolling,
or accumulator storage — Metal already handles those.

### Where the pathology actually lives (scope the effort — measured 2026-07-04)
The duplicated-guard pathology is NOT everywhere. It appears in the **grouped-MoE kernels** whose sentinel /
ragged-skip guard (`tileExpert[tile] < nExpert`, `frag0/frag1`) is reused across every fragment/block
(q4k: 57×; the generic grouped reg matmul: 209×). It does **NOT** appear in the **non-grouped** matmuls
(attention QKV/O-proj, dense gate/up/down, lm_head) — they take a compile-time weight offset, no guard.
Two consequences:
- The deployed DiffusionGemma decode runs the grouped-MoE kernels as **hand-MSL** (`mslQ4kDispatch` etc.),
  so H1 on the WGSL q4k kernel helps only the `DG_NOMSL` portable fallback, not the default path.
- Applying `let'` to the grouped reg matmul measured **0 ms** at the deployed MoE shape (M=6208) — the
  guard is amortized under a compute-bound matmul already sitting *below* its roofline floor. It helps only
  when the guard is a real fraction of a small kernel's time.
- **So there is no broad decode-wide `let'` win to harvest.** The remaining WGSL headroom in the deployed
  decode is **tile efficiency at small M** (ROOFLINE: QKV-KV full 4.9×, dense gate/up 3.2×, O-proj 1.9×
  above floor at M=262) — a *tile-granularity* problem (see `PERF_AUTOTUNE_LOOP.md`), NOT a codegen-CSE one.

**Bottom line: `let'`/CSE is the right tool for grouped kernels with reused guards; verify the pathology
exists (dump + count) AND that the kernel isn't already at floor before spending effort.** A general
auto-CSE codegen pass would be correct but buys ~0 on the deployed path (no duplication there to remove).

---

## The method (repeatable — don't re-derive it each time)

1. **Dump the Tint-generated MSL.** `HESPER_DUMP_MSL=1` on any run routes Dawn's `dump_shaders` output to
   stderr between `===MSLDUMP-BEGIN/END===`. (`HESPER_DUMP_MSL_NORENAME=1` keeps WGSL names but SIGTRAPs
   large kernels — usually leave it off.) The `msl-poc` exe is the cleanest single-kernel source.
2. **Diff vs the hand-MSL** (`native/metal_replace.mm`, e.g. `kQ4kMslTemplate`). Count the pathologies
   above: `grep -c '< 128u'` (guard dup), `grep -c 'v_3(' / '\bv('` (div/mod), `tint_array<simdgroup`
   (array accumulators).
3. **Fix in the WGSL** (`let'` binding, scalar decls, …). One change at a time.
4. **Measure with the PoC ratio oracle:** `msl-poc` reports `WGSL(Tint) ms | MSL(native) ms | ratio |
   maxDiff`. The hand-MSL number is the 1.0× target. `HESPER_DISABLE_ROBUSTNESS=1` for the robustness-off
   comparison.
5. **Golden-gate: `maxDiff` must be unchanged (bit-identical).** All these fixes are semantics-preserving;
   a changed `maxDiff` means a bug, not a win.
6. **Validate on the real decode** (`scripts/dg_eval.sh`, 8-prompt gate — must stay 8/8) before commit.

### The general fix (supersedes manual `let'`)
The DSL should **auto-CSE**: detect reused `Exp` subtrees and emit one WGSL `let` instead of duplicating.
That turns the manual, per-kernel `let'` work into a codegen pass benefiting every kernel at once. Until
that lands, apply `let'` to loop-invariant / multiply-used values by hand.

---

## Caveats (hard-won — ignore at your peril)

- **Measurement asymmetry.** The PoC's WGSL side is *wall-clock* (`monoMsNow`, includes CPU encode +
  sync); the MSL side is *pure GPU* (`cb.GPUEndTime-GPUStartTime`). Encode is ~4% (0.285 ms/iter at the
  q4k shape) and the real decode already batches it away — so compare the *GPU* portions. A wall-clock
  ratio slightly overstates the codegen gap. Splitting encode vs submit+wait confirmed the core gap is
  real GPU compute, not dispatch overhead.
- **Batching is necessary but not sufficient.** Running in one batched command stream removes the ~4%
  per-dispatch encode overhead (the real decode does this) but does nothing for kernel codegen quality —
  the guard-dup / accumulator wins are the bigger lever and are orthogonal to batching.
- **Never trust a perf number without `pgrep -f decode|msl-poc == 0` and a cool box.** Back-to-back builds
  warm the machine ~1.5×; orphan runs corrupt both timing and determinism.
- **Never `kill -9` a decode/PoC mid-GPU-work** — it wedges Metal and corrupts all later runs. Use
  `timeout`/SIGTERM and let runs finish.
- **`maxDiff` bit-identical is the correctness gate** for any codegen-quality change. If it moves, revert.

---

## Results log

| Change | kernel | WGSL ms | ratio (robustness off) | golden |
|---|---|---|---|---|
| baseline | q4k gate/up reg | 7.22 | 1.37× | — |
| H1: guard `let'` (57×→1×) | q4k gate/up reg | **6.08** | **1.15×** | bit-identical ✅ landed |
| H2: scalar accumulators | q4k gate/up reg | 6.10 | 1.156× | NULL (Metal SROAs the array) — reverted |
| strength-reduction (÷/%→>>/&) | q4k gate/up reg | 7.24 | 1.37× | NULL (Metal recovers) — reverted |

Hand-MSL target: 5.28 ms = 1.0×. H1 captured essentially all the recoverable gap; residual 1.15× is near
the WGSL-through-Tint floor.
