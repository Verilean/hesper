# Replay methods — bare-Metal timing of any engine's decode token

The methodology behind DEVPLAN §8/§9: capture ONE decode token's dispatch sequence
from an engine, then re-encode it in a minimal Metal harness and time GPU wall
(GPUStartTime→GPUEndTime, 20 iters, report min/avg). Because the harness is the same
for every engine, the numbers are directly comparable — this is what produced the
three-engine table (webml 3.90 ms / llama.cpp 6.31 / Hesper 6.78, §9c).

**Universal caveats**

- **Timing-only.** Replays run on garbage/synthesized buffer contents. Valid because
  the kernels are fixed-trip-count in the *weights*; but anything read from a
  *uniform/params* buffer (positions, cache lengths, dims) DOES steer loops — those
  buffers must carry real traced values (see the webml mirror below).
- **Concurrent-no-barrier numbers are a race mirage.** They overlap through true
  dependencies no correct engine can ignore. All three engines' decode dataflows are
  ~86–96 % barriered under honest hazard analysis; report serial (and, if needed,
  hazard-barrier) numbers as the real ones. (§8/§9: concurrency is worth ≲0.6 ms on
  this hardware for every engine.)
- Measure cool, `pgrep` clean, never `kill -9` a process mid-GPU-work.

---

## 1. Hesper (in-tree, env-gated)

The harness lives in `native/metal_replace.mm` (replay_* FFIs) + capture hook in
`Hesper/WGSL/Execute.lean` (`executeShaderNamed`) + `Hesper/WGSL/NativeReplay.lean`.
MSL comes from the tint CLI run on the exact WGSL we hand Dawn — build tint once:

```sh
cmake -S .lake/build/dawn-src -B /tmp/tint-build -DCMAKE_BUILD_TYPE=Release \
  -DTINT_BUILD_CMD_TOOLS=ON -DTINT_BUILD_TESTS=OFF \
  -DTINT_BUILD_SPV_READER=OFF -DTINT_BUILD_SPV_WRITER=OFF \
  -DTINT_BUILD_GLSL_WRITER=OFF -DTINT_BUILD_HLSL_WRITER=OFF \
  -DTINT_BUILD_MSL_WRITER=ON -DTINT_BUILD_WGSL_READER=ON -DTINT_BUILD_WGSL_WRITER=ON
cmake --build /tmp/tint-build --target tint_cmd_tint_cmd -j 10
```

**Phase A (timing replay of one captured token):**

```sh
HESPER_CUDA_GRAPHS=0 HESPER_NATIVE_REPLAY=1 HESPER_TINT_BIN=/tmp/tint-build/tint \
  ./.lake/build/bin/gemma4-inference data/gemma-4-E2B-it-Q4_K_M.gguf "prompt" 8
```

`HESPER_NATIVE_REPLAY=<n>` captures decode forward #n (n≥1 so pipelines are warm);
after generate it prints serial / concurrent-no-barrier / layer-barrier /
hazard-barrier timings. Expect `misses=0` in the capture report — a miss means a
kernel's MSL entry or a buffer name failed to resolve.

**Phase B (real native execution, correctness-gated):**

```sh
HESPER_CUDA_GRAPHS=0 HESPER_NATIVE_DECODE=1 HESPER_TINT_BIN=... \
  ./.lake/build/bin/gemma4-inference ... 32
```

From forward #n on, the hook records ops and SKIPS Dawn; `commitToken` executes the
token on Dawn's own MTLCommandQueue (single-queue FIFO keeps ordering with Dawn's
staging writes/readbacks). `HESPER_NATIVE_DECODE_MODE` = 0 serial (fastest host) /
3 hazard-concurrent. Gate: the generated token sequence must equal the Dawn path.

Gotchas encoded in the implementation: Tint may RENAME entry points (`main` → `v`) —
parse `kernel void NAME(` from the MSL; the tint CLI assigns its own `[[buffer(i)]]`
order — parse the entry signature, never assume declaration order; key the PSO cache
on the caller's u64 cacheKey, NOT a hash of the ~10 KB MSL string (572 records/token
made that megabytes of hashing); FFI error strings must be
`lean_mk_io_user_error`-wrapped or the top-level printer segfaults.

## 2. llama.cpp (fork patch, `llama/ggml-metal-replay.patch`)

llama.cpp funnels ALL Metal encoding through 8 wrapper functions in
`ggml/src/ggml-metal/ggml-metal-device.m` — the patch records the raw command stream
(setPipeline/setBytes/setBuffer/setThreadgroupMemoryLength/dispatch/barrier, objects
retained) and replays it at that encoder's `end_encoding`:

```sh
cd refs/llama.cpp-diffusiongemma && git apply /path/to/ggml-metal-replay.patch
cmake --build build --target llama-cli -j 10

# discovery pass: one line per encoder with its dispatch count
GGML_METAL_REPLAY_LOG=1 build/bin/llama-cli -m model.gguf -p "..." -n 6 --temp 0 -no-cnv --no-warmup < /dev/null
# E2B: each decode token = TWO encoders (84 + 770 dispatches). Replay both:
GGML_METAL_REPLAY=9 build/bin/llama-cli ...   # prints serial + as-recorded-concurrent
GGML_METAL_REPLAY=8 build/bin/llama-cli ...
```

**Deadlock gotcha:** llama pre-enqueues its command buffers, so the replay MUST
commit on a PRIVATE MTLCommandQueue — committing on llama's queue waits behind an
enqueued-but-uncommitted CB forever. Never upstream this patch (llama.cpp does not
accept AI-generated PRs; local instrumentation only).

Their own ablation toggles need no patch at all:
`GGML_METAL_CONCURRENCY_DISABLE`, `GGML_METAL_FUSION_DISABLE`,
`GGML_METAL_GRAPH_OPTIMIZE_DISABLE` + `llama-bench -p 0 -n 64 -r 3` (§9 table).

## 3. webml / any WebGPU app in a browser (`webml/`)

No source access needed — monkey-patch the WebGPU API around the real app:

1. Mirror the app locally (e.g. the HF space's `index.html` + JS) into a directory
   with `webml/trace.html`, `collector.py`, `convert.py`, `replayer.mm`.
2. `trace.html` installs hooks BEFORE importing the app module:
   `createShaderModule/ComputePipeline(+Async)/BindGroup/Buffer`,
   `setPipeline/setBindGroup/dispatchWorkgroups`, PLUS a **content mirror for
   buffers ≤ 64 KiB** via `queue.writeBuffer` and `getMappedRange`/`unmap` hooks —
   uniform values drive loop trip counts; zero-filled params would fake the timing.
   It then drives the app's own API (here `Gemma4Mobile.load` → `warmup` →
   `generate`), flips `tracing` on for exactly one steady token (iteration N of the
   token stream), and POSTs `{ops, pipes, shaders, contents}` to the collector.
3. `python3 collector.py` (port 8917) serves the directory and saves `trace.json`.
4. Headless Chrome runs the real app — WebGPU works headless:
   `chrome --headless=new --use-angle=metal --enable-unsafe-webgpu
   --user-data-dir=<profile>` (the profile caches the multi-GB model for reruns;
   one profile = one Chrome at a time — SingletonLock).
5. `HESPER_TINT_BIN=... python3 convert.py` — runs tint per pipeline
   (`--overrides` carries GPUProgrammableStage constants), parses WGSL
   `@group/@binding var NAME` and the MSL entry signature to build the
   (group,binding)→`[[buffer(K)]]` map (tint STRIPS unused bindings — skip those),
   resolves `@workgroup_size` (override-aware) and a threadgroup-memory upper bound,
   and emits `out/manifest.json`.
6. `clang++ -fobjc-arc -std=c++17 replayer.mm -framework Metal -framework Foundation
   -o replayer && ./replayer out/manifest.json` — allocates the traced buffer set
   (2.42 GB for webml E2B), uploads the mirrored uniform contents, encodes the token
   serial + concurrent, prints min/avg.

Validation: webml's replayed serial 3.90 ms matched its real in-browser ~4.0 ms/token.
