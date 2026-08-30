# wgsl-check — static bounds checker for WGSL and MSL-subset compute kernels

One tool, two front ends, one set of analyses. This is the safety foundation
that lets the thin-Metal backend run with robustness and runtime validation
OFF: every kernel in the hot path is statically checked instead.

## Why it exists

Commit `54a2a60` fixed four GPU races that cost weeks. The common class: a
dispatch grid rounds up past the logical element count; the excess threads'
stores are out-of-bounds; WGSL robustness CLAMPS them onto the last element,
silently racing its owner — non-deterministic garbage, no crash, invisible to
every runtime tool. Tint/Dawn deliberately do not check this: they guarantee
the MACHINE is safe for all programs; whether THIS program's grid covers THIS
buffer is a logical-shape question only the dispatch site knows. wgsl-check
takes the dispatch manifest (grid + buffer element counts) and answers it.

## What it checks

- **OOB writes** (the race class above) and **OOB reads** (v1.1 — the
  clamp-vs-predication cross-compiler divergence class)
- **simdgroup/subgroup-matrix load/store footprints** (base + rows×stride
  extent for 8×8 fragments) — the WMMA-tail heap-stomp class (a real
  production bug, R32) is caught statically; the production q4k stores prove
  in-bounds tight-by-one
- Method: parse the compute entry → u32 interval analysis with taint,
  guard refinement (if / early-return / for bounds / else-if chains,
  let-bound bool guards, `v+const` subjects, select-under-refinement),
  lazy lets, builtin-component refinement, the mod idiom `x-(x/c)*c`,
  dead-branch pruning; evaluate every access index against manifest bounds.

## Front ends

- **WGSL**: the DSL-generated kernels are checked at their WGSL source —
  this also covers the tint-translated kernels the Metal backend runs
  (translation is semantics-preserving; the trust boundary is explicit).
- **MSL subset** (M-Metal Stage 3): hand-written Metal kernels get the same
  analyses. Signature scan maps `[[buffer(n)]]`/thread attrs; bodies are
  normalized (single-return inline helpers substituted, ternary→select,
  C decls→let/var). Route with `"lang": "msl"`; `"wg"` required in the
  manifest (workgroup size is not in MSL source).

## How to run

- Ad hoc: `wgsl-check --manifest manifest.json` (see tools/WgslCheck/Main.lean
  header for the manifest schema; buffer sizes can come from a JSTrace).
- Production gate: `bash scripts/msl_check.sh` — extracts the hand kernels
  from native/metal_replace.mm with production dims (provenance documented
  line-by-line in the script header), runs the fixtures (which MUST fail)
  and the production kernels (which must not). Exit 2 = the suite itself
  is broken.
- Fixtures: `specs/wgslcheck/` (3 intentional FAILs + fixed twins),
  `specs/mslcheck/` (scalar OOB, simdgroup tail overrun, unguarded fan-out).

## Track record (experiment ledger: e4b-webgpu DG_PORT_LOG.md)

- 0 false FAILs on two full production engines (DG 1426-dispatch step,
  A4B 144 dispatches / 96 kernels)
- The trace-completeness sibling (e4b-webgpu scripts/dgtrace-validate.py,
  same read/write-set analysis) machine-found the two invisible-writer trace
  holes that blocked the Chrome lab for weeks (R32)
- Every kernel shipped this season (flash-attention, fused elementwise,
  Q6K warp, directB variants) passed through it pre-integration

## Honest unsoundness boundaries

- Pointer locals unsupported (absent from our hand kernels; the vendored
  ggml template kernels are OUTSIDE the subset — they are gated by golden +
  eval instead, and the parser growth is open work)
- Multi-statement helper bodies are not analyzed (bare buffer arguments
  surface as WARN)
- Threadgroup-memory bounds unchecked (device buffers only)
- Data-dependent indirection (e.g. counting-sort outputs) is reported as
  WARN — provable only with the producer's postcondition (see
  specs/ChunkCap.lean for one such discharge)
- Manifests are per-configuration (dims baked); regenerate for other shapes
