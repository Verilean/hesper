# Next-architecture playbook: environment per use case

*Companion to `docs/ANATOMY_OF_DECODE_REPORT.md` §7. Every recommendation below is
tied to a measurement in this record, not taste.*

One stack correction to the common advice first: **Deno's WebGPU is wgpu/naga (Rust),
not Dawn/Tint** — a different WGSL→MSL compiler than Chrome. Usually fine (both are
~no-op transpilers; the Metal compiler does the real work), but if the deploy target
is Chrome, winners should be spot-validated on a Dawn path. Dawn itself ships Node
bindings (`dawn.node`), which gives a CLI loop with *Chrome's exact* compiler stack.

## The environment menu (thin → thinner)

| environment | stack | TAT | when |
|---|---|---|---|
| **Deno + WebGPU** | JS/TS → wgpu/naga → Metal | seconds, one command | default agent sandbox: `deno run` per iteration, no browser, no locks |
| **Node + dawn.node** | JS → Dawn/Tint → Metal | seconds | same loop but bit-faithful to Chrome (deploy target = browser) |
| **Headless Chrome** | the real app → Dawn/Tint | seconds after page setup | fidelity runs + tracing *other people's* engines (the §9c monkey-patch method); NOT the inner loop — profile locks, collectors, lifecycle friction (we hit all of it) |
| **Python + wgpu-py** | Python → wgpu-native | seconds | when goldens come from PyTorch anyway: generate reference tensors and bit-compare in one process |
| **bare-Metal harness** | MSL strings → Metal (0 layers) | seconds (~4 ms recompile, disk cache) | Apple-final targets; already built (`tools/replay/webml/replayer.mm` pattern); tint CLI offline if WGSL is the source language |

## Use case 1 — autonomous kernel-optimization agent

**Environment: Deno (or the bare-Metal harness for Apple-final).** CLI beats browser
for the agent loop: single command per iteration, clean stdout, no SingletonLock /
collector-server lifecycle (§9c cost us several detours). Architecture per report §7:

1. Agent sees ONLY: tensor shapes, the golden, the kernel source string, and last
   timing. No host code in context (P2a).
2. Golden gate before timing, every iteration (maxDiff kill switch — this record's
   golden gates caught every real bug; raw text/token equality only for e2e).
3. Timing = GPU wall min-of-N on the harness, **cold-stream by default** (§13:
   rotate weight clones past the SLC; warm numbers flattered by 13–26 %).
4. Aim the search at STRUCTURE, not parameters: the measured pool is ~2.4 ms of
   kernel-shape/fusion quality (§4.2) vs ≤3 % in parameter space (§13). Prompt tasks
   as "fuse these two ops / change the operand format", not "try workgroup sizes" —
   parameters are a cheap inner sweep the harness automates (0.118 s/variant).

## Use case 2 — Copilot-style (human-driven) kernel work

Same harness, plus two things the autonomous loop doesn't need: a persistent watch
mode (`deno run --watch` re-times on save — keeps the human's loop at editor cadence)
and the per-class profiler view (§12) so the human picks targets by measured budget,
not intuition. The incumbent-guard idea (§ autotune) applies to humans too: the
deployed config always competes before a "win" ships.

## Use case 3 — new-model bring-up (correctness first)

**Environment: Python + wgpu-py, with the reference engine as oracle.** Bring-up is
dominated by dtype/layout surprises (this record: Q5_K-vs-Q6_K, BF16-as-F16,
F32-as-Q4_K — 3 of 4 E2B bugs), and the fastest debugging method we found was
layer-bisect against llama.cpp's eval-callback plus "dump GPU state, continue on
CPU". Python puts the reference (HF/PyTorch), the parser, and the bit-compare in one
process. Read the reference implementation FIRST (principle 7 — violated three times
in this record, each violation cost a detour).

## Use case 4 — research (TTT / inference-time learning)

**PyTorch, with fake-quant in the eval from day one** (a bf16-validated signal may
die on a Q4 frozen base). Deployment path when a recipe stabilizes: server → vLLM
(multi-LoRA fits per-session fast weights); local → llama.cpp fork (~4–6 hand
kernels: transposed-quant matvec, adapter outer products, optimizer step). Nothing in
this use case wants a new engine.

## Use case 5 — production deployment on Apple/local

**llama.cpp (or the app's existing engine), not a new runtime.** §4.2's law: the
runtime is a ±0.5 ms rounding term — ship kernels into an engine that already has
distribution. If the winning kernels came out of the WGSL search loop, port the
*structure* (the record shows structure ports faithfully: our llama-port hit 90–97 %
isolated) — don't port the stack.

## Use case 6 — verification (the Lean salvage)

**Verify the trace, not the generator** (§7-P3). After the structure freezes, capture
the token's dispatch list (the frozen-replay artifact: a few hundred (kernel, buffer,
range, grid) tuples) and prove bounds / no-writable-aliasing / race-freedom over that
finite object in Lean — a deployment certificate, off the exploration loop. The bug
classes it would have caught here: 11 clamp-write races, the silently-dropped
writable-aliasing dispatch, grid-roundup OOB writes.

## Anti-recommendations (each paid for in this record)

- **Don't write an interpreter/bindings on top of Dawn.** Two foreign layers (Dawn,
  Tint) cost us: a serial-dispatch hardcode, an MSL-printer miscompile, silently
  swallowed validation errors, and weeks of instrument-building to see through them.
- **Don't put the kernel language behind a compiled host.** Structure edits must not
  pay a build (§5: the 8–10 min loop is the single largest explanatory variable).
- **Don't trust warm isolated benchmarks** (13–26 % flattery) or **racy concurrent
  timings** (the 3.43 ms mirage) — cold-stream + hazard-correct or it didn't happen.
- **Don't chase scheduling.** Concurrency, reordering, dispatch-count: all measured
  ≲0.6 ms across three engines for M=1 decode.
