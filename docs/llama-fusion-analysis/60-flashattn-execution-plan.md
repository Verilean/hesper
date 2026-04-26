# FlashAttention 高速化マルチセッション計画

2026-04-26 起案. doc 59 の Option B + C を複数セッションに分けて完遂する計画.

## ゴール

decode flashAttn kernel time を hesper 1.12 ms/tok → 0.21 ms/tok (llama.cpp parity).
graphs ON での全体 TPS を 81.9 → ≥ 95 (理論上限近く).

## 計測条件 (frozen)

```bash
HESPER_DP4A=1 HESPER_USE_MMAP=1 HESPER_IGNORE_EOS=1 HESPER_CUDA_GRAPHS=1 \
  lake exe gemma4-cuda data/gemma-4-e4b-it-Q4_K_M.gguf "Hello world how are you" 60
```
- 指標 1: TPS (tokens/sec)
- 指標 2: scripts/kernel_compare_graphs.py の **flashAttn 行 µs/call**
- 指標 3: bit/cosine parity vs baseline output
- 各セッション末尾で 3 指標を doc に追記

## セッション構成

### Session 1: Option B Phase 1 — vec kernel skeleton + dispatcher 切替

**目標**: llama.cpp 風の "1 thread per K position, warp-shuffle reduce" 構造を 1 つだけ実装し、HESPER_FA_VEC=1 で opt-in 可能にする. f32 KV のまま. bit-parity 確保.

**実装**:
1. ShaderM 側調査:
   - 2D thread block (block_dim.x=warp_size, block_dim.y=nwarps) サポート → ない場合 1D で 32 lane × 4 warp manual emulation
   - `subgroupAdd` (warp shuffle reduce) 使用箇所をサンプル抜粋
2. 新 kernel `flashAttentionVecKernel` を `Hesper/WGSL/FlashAttention.lean` に追加:
   - gridX = numHeads, gridY = 1 (split-K は Phase 2 で), gridZ = 1
   - workgroupSize = 128 (4 warps = 4 × 32)
   - 各 thread が異なる k position を分担して q·K[k] dot product (warp-内 D 軸並列で 32 thread shuffle reduce)
   - online softmax は warp-private のまま、最後に warp 間で combine
   - V·attn は同様
3. dispatcher gate: `if (← IO.getEnv "HESPER_FA_VEC").isSome` で新 kernel に切替, default は既存 tiled
4. bit-parity test (HESPER_FA_VEC=0 vs =1 で同じ token 列を出すか)
5. trace で flashAttn 行の µs/call を測る

**Definition of Done**: bit-parity ✓ + flashAttn µs/call が 32.6 → < 15 (2× 以上) になっていれば Phase 2 へ. それ未満なら kernel 内ボトルネックを ncu で見て調整.

### Session 2: Option B Phase 2 — split-K (gridY > 1) + combine kernel

**目標**: cacheLen が大きいほど gridY を増やして K 軸並列度を上げる. partial を combine kernel で集約.

**実装**:
1. flashAttentionVecKernel を gridY 軸対応に変更 (start = blockIdx.y * nthreads, stride = gridDim.y * nthreads)
2. partial buffer + combine kernel (既存 phase2 を流用 or 新規) で gridY 個の partial を softmax-aware merge
3. dispatcher で `gridY = ceil(cacheLen / 128)` を計算
4. trace で flashAttn 0.5 ms/tok 以下を狙う

**Definition of Done**: bit-parity ✓ + flashAttn ≤ 0.5 ms/tok + TPS 90+

### Session 3: Option C Phase 1 — K cache f16 化

**目標**: K cache を f32 → f16. V cache は f32 のまま (段階的移行で blast radius 制御).

**実装**:
1. `Gemma4KVCache` 構造体 / state allocation を f16 サイズ (×0.5) に. 互換性のため `kBufF16Mode : Bool` flag をつけて opt-in
2. RoPE-K + scatter kernel (現在 f32 出力) を f16 出力版で別途用意, dispatcher で切替
3. flash attn kernel (Vec, Tiled 両方) で `k_cache` を f16 として読み (`fp16ToF32` cvt 挿入)
4. KV cache の読み書きの整合性を再確認 (forwardPrefillBatch 経路含む)
5. **bit parity → cosine similarity > 0.999 へ判定基準変更**: テスト用に `HESPER_KV_F16=1` で 60 token 出してログを diff. 出力 token 列が **数 token 違っても correctness OK** とする (greedy decode の small numerical drift)
6. trace で flashAttn ms/tok の動きを記録

**Definition of Done**: cosine sim > 0.999 (or 出力 ≥ 95% 一致) + KV cache VRAM 半減 + flashAttn ms/tok 改善が観測

### Session 4: Option C Phase 2 — V cache f16 化

**目標**: V cache も f16 へ. flash attn kernel に対するメモリ帯域を 50% に.

**実装**:
1. forwardBlock の Vcur scatter (現在 f32) を f16 出力版に (RMSNorm-V の出力を直接 f16 へ)
2. flash attn kernel `v_cache` を f16 として load
3. forwardPrefillBatch 側も同様
4. 全テスト (mono layer parity, e2e decode)

**Definition of Done**: cosine sim > 0.999 + V cache VRAM 半減 + flashAttn が memory-bound 部分 2× の効果

### Session 5: 最終測定 + ドキュメント

1. baseline / Option B-only / Option C-only / Option B+C の 4 way 測定
2. nsys + kernel_compare_graphs.py で 1:1 比較
3. doc 59 / 60 に最終結果追記
4. memory `project_flashattn_decode_5x_gap.md` を CLOSED に更新
5. CLAUDE.md 風に summary を残す

## 段階的にコミットする方針

各 Session の Definition of Done を満たした時点で commit. opt-in env flag を残すので revert は不要 — 旧経路は env 未設定で残る. flashAttn 系の env: `HESPER_FA_VEC=1`, `HESPER_KV_F16=1`, `HESPER_KV_F16_V=1`.

## 失敗時の Rollback

- Session 1 で bit-parity 取れない → 中断, Phase 0 (1-warp 1-tile, full sequential) で完成度を下げて再構築
- Session 2 で gridY > 1 で精度 drift → split-K の combine 関数を再検証 (online softmax の rescale)
- Session 3-4 で cosine sim < 0.999 → cvt path のバグ; 1 layer のみ f16 化して bisect

## 各セッション開始時にやること

1. このドキュメント読む
2. baseline 測定 (上の frozen コマンド)
3. memory 関連メモを読む (`project_flashattn_decode_5x_gap.md`)
4. 当該 Session の "実装" を順に
5. Definition of Done 判定 + 数値追記

## このセッション (kickoff) でやること

Session 1 の Step 1 (ShaderM 側調査) のみ実施し doc に追記する.
それ以上は次セッション以降.

## kickoff session 結果 (2026-04-26)

### ShaderM 側調査結果

**1. 2D thread block サポート**: ✅ あり
- `Hesper/WGSL/Shader.lean:52`: `WorkgroupSize { x : Nat, y : Nat := 1, z : Nat := 1 }`
- `ExecConfig.workgroupSize := { x := 32, y := 4 }` で llama.cpp 風
  `block_dim(warp_size, nwarps)` 構造が表現可能
- `ShaderM.localId` は `Exp (.vec3 .u32)` を返すので `vec3X` (lane id within warp)
  と `vec3Y` (warp id within block) を独立に取れる

**2. subgroup primitives**: ✅ あり
- `Hesper/WGSL/Exp.lean:265`: `subgroupAdd : Exp t → Exp t` (warp-内 sum reduction)
- `Hesper/WGSL/Exp.lean:259-260`: `subgroupBroadcast`, `subgroupBroadcastFirst`
- 既存の RMSNorm warp-shuffle で実証済み (doc 49)
- needsSubgroups flag が `Hesper/WGSL/Monad.lean:48` にあって自動的に WGSL
  の `enable subgroups;` が emit される

**3. 既存 Q6_K dp4a kernel に warp-内 partial reduction が既にある**:
- `Hesper/Layers/Linear.lean` の fusedQ4K kernels に `subgroupAdd` 使用例
- 同パターンを flash-attn 用にコピー流用可

### Session 1 開始時の必要なもの (準備完了)

- ShaderM API: 2D wg + subgroupAdd (両方✅)
- 比較対象: `llama.cpp/ggml/src/ggml-cuda/fattn-vec.cuh` (template instantiation:
  `<D=256, ncols=1, K_type=GGML_TYPE_F16, V_type=GGML_TYPE_F16, false>`)
  ※ ただし f32 KV のまま実装. f16 化は Session 3-4.
- 検証ツール: `scripts/kernel_compare_graphs.py` (graphs ON 1:1 比較)
- 既存テスト: per-layer flash attn parity test がもしあれば bisect に使う

### Session 1 で書く新コード見積

- 新 kernel `flashAttentionVecKernelF32` (~150 lines, ShaderM)
- dispatcher 切り替え (HESPER_FA_VEC=1) (~10 lines)
- Lean 側 `executeFlashAttentionVecF32` 関数 (~30 lines)
- 計 200 行程度

### Session 1 進捗 (2026-04-26)

**WIP commit `0076815`**: skeleton landed but **NOT correct yet**.

- TPS regression: 82.1 → 70.1 (HESPER_FA_VEC=1) — algorithmic bug
- Output diverges from baseline ⇒ silent correctness regression
- Suspected root causes (still TBD, no parity test yet to bisect):
  - `shared_warp_sums` not zeroed between K iterations; cross-warp
    subgroupAdd may pull stale lanes >= numWarps
  - cross-warp broadcast: lane 0 of warp 0 produces canonical sum but
    we publish via `tid==0` write — confirm warp 0 of all heads agrees

### Lesson from Session 1: gemma4 TAT is too long for bisect

End-to-end gemma4 60-token decode is ~6 min per iteration (build 5 min
+ measure 1 min).  Bisecting kernel correctness against the legacy
kernel via gemma4 is impractical.

**Pivot**: build a **CUDA unit test** that compares
`flashAttentionVecParamsKernel` against `flashAttentionDynamicParamsKernel`
on the same Q/K/V/params inputs at multiple sizes (e.g. cacheLen ∈
{1, 8, 32, 64, 128}, headDim=256, numHeads=2 ish).  Existing pattern:
`Tests/CUDA/CUDAFlashAttnTest.lean` does this for sub-kernels.

DoD update: **bit-parity unit test passes first, then enable in gemma4**.
That keeps the iteration cycle <30s instead of 6min.

### Next concrete action (Session 1 continued)

1. Add a new test file `Tests/CUDA/CUDAFlashAttnVecParityTest.lean`
   that runs both kernels at small sizes and asserts max abs diff <
   1e-5
2. Run it to bisect — fix the warp_sums clear / cross-warp issue
3. Re-enable in gemma4 with confidence

### Session 1 continued (2026-04-26 part 2)

**Unit test landed** (`Tests/CUDA/CUDAFlashAttnVecParityTest.lean`,
exe `cuda-flashattn-vec-parity`, ~30 s): 17 cases
covering Gemma 4 geometry (numHeads=8, numKVHeads=1, headDim=256,
cacheLen 5-100). **All PASS, max abs diff < 1e-5**.

**Conclusion**: the vec kernel is bit-parity with the legacy
`flashAttentionDynamicParamsKernel` — the suspected
shared_warp_sums / cross-warp bug from the earlier WIP commit was
NOT the issue.  Earlier `oldMax := maxScore` → `oldMaxVar ←
ShaderM.var maxScore` fix already corrected it.

**Re-tested in Gemma 4**:

| variant | graphs OFF (60 tok) | graphs ON (30 tok) | output |
|---|---:|---:|---|
| baseline | 60.5 TPS | 77.3 TPS | "looking that world how are world..." |
| HESPER_FA_VEC=1 | 58.6 TPS | 73.8 TPS | (graphs OFF) **identical** to baseline; (graphs ON) **DIVERGES at token 2+** |

So:
1. **graphs OFF** : output identical to baseline; minor TPS regression
   (60.5 → 58.6, -3%) — conservative warp-shuffle reduce isn't
   actually faster than the 256-thread tree reduce on this kernel
2. **graphs ON**  : output **diverges at token 2+** despite the kernel
   being bit-parity — points at a CUDA-Graph capture / replay
   interaction, **not** a kernel-correctness bug.  Token 0 (prefill
   path, no graph) and token 1 (capture pass) match; token 2+ is
   replay and that's where the divergence appears.

### Open issue for next session: Graph-capture divergence

Symptom: vec kernel runs correctly outside CUDA Graph capture, but
graph replay produces wrong output.  Hypotheses (untested):
- Buffer binding ordering vs the legacy `ce`-based dispatch was
  recorded differently, so the graph node's bound pointer table is
  permuted
- `flashAttentionDynamicParamsKernel` declares `params` as
  `declareStorageBuffer ... .read`, vec kernel as the same — should
  be equivalent
- workgroupSize 128 vs 256 affects something about the capture's
  shared-memory plan
- The `sharedNamed "shared_q"` / `"shared_warp_sums"` / `"shared_out"`
  shared allocations differ in size from the legacy kernel's set;
  CUDA Graphs may capture a static smem-size attribute that mismatches
  on replay

### Verdict on Session 1 — call it: kernel correct, perf flat

Conservative warp-shuffle reduce (Option B Phase 0) **does not move
the needle** even when working correctly: -3% on graphs-OFF.  The K-
serial structure remains and the per-K barrier count drops only
modestly (8→2) — the kernel was **memory-bound**, not barrier-bound.

The actual win will require Session 2's **K-parallel** layout
(multiple warps process different K positions simultaneously, llama.cpp
style).  Session 1 is closed as **infrastructure landed (test +
opt-in env), no perf gain yet**.

### Files

- `Tests/CUDA/CUDAFlashAttnVecParityTest.lean` (new)
- `lakefile.lean` registers `lean_exe «cuda-flashattn-vec-parity»`
- `Hesper/WGSL/FlashAttention.lean`: `flashAttentionVecParamsKernel` +
  `executeFlashAttentionVecParams`
- `Hesper/Models/Gemma4.lean`: `HESPER_FA_VEC=1` dispatcher gate

### Microbench (added at end of Session 1, 2026-04-26)

Extended `cuda-flashattn-vec-parity` exe with a micro-benchmark mode
that loops each kernel 200× under Gemma-4 geometry (nH=8, nKV=1,
D=256), syncs once, prints avg µs/call.  Total exe runtime ~30 s.

Results (200 iters/case):

| cacheLen | legacy µs | vec µs | speedup |
|---:|---:|---:|---:|
| 8   | 152.6 | 106.0 | **1.44×** |
| 32  | 147.1 | 101.5 | 1.45× |
| 64  | 151.1 | 106.5 | 1.42× |
| 128 | 152.4 | 112.7 | 1.35× |
| 200 | 152.3 | 174.7 | **0.87× (slower!)** |

Findings:
- vec is **1.4× faster than legacy** for cacheLen ≤ 128 (the entire
  Gemma 4 decode range up to ~100)
- vec **regresses at cacheLen 200** — the K-serial loop becomes
  expensive when the loop has many iterations and the warp-shuffle
  cross-warp gather doesn't amortise
- Absolute Lean-bench numbers (152 µs vs nsys 32.6 µs) include host
  per-launch overhead; only the **ratio** is comparable

Why the 1.4× microbench speedup didn't show up as TPS gain in Gemma 4:
flashAttn is only 14% of decode wall (1.12 ms/tok of 7.92 ms).  A
1.4× kernel speedup gives ~0.34 ms/tok savings ≈ +3-4 TPS, which is
within run-to-run noise on the e2e benchmark and was masked by some
Gemma-4-specific interaction (likely occupancy / smem usage hitting
neighbouring kernels under graph capture).

Implication for Session 2 design:
- Need **K-parallel** layout that wins also at long cacheLen (current
  vec is K-serial, regresses at cacheLen=200)
- Microbench will continue to give 30s TAT iteration on raw kernel
  speed, decoupled from Gemma-4 wall-time noise

## After Session 5: PTX-level closure (post-doc-60)

Sessions 1-5 are **ShaderM-level** — we mirror llama.cpp's algorithm
(K-parallel, K/V f16, online softmax with split-K combine) in the
high-level DSL.  After all 5 land, we close the gap at the PTX layer.
Same pattern as `project_q6k_ncu_session.md` /
`project_ptx_codegen_improvements.md`:

1. Dump hesper's generated PTX:
   `HESPER_DUMP_PTX=1 lake exe cuda-flashattn-vec-parity` →
   `/tmp/hesper-flashattn.ptx`
2. Extract llama.cpp's flash_attn_ext_vec cubin/PTX
   (`cuobjdump --dump-ptx` from llama-cli's loaded module)
3. **Diff at instruction level**: ld.global.nc usage (cache hint),
   register count, ldmatrix/cp.async if available, smem bank-conflict
   stride, fp16x2 packing
4. Patch ShaderM codegen for any missing PTX optimisation
   (BufferHint .readOnly already done, doc 55), or fall back to a
   hand-written PTX module wired through `LlamaCppPTX` (see
   `Hesper/Backend/LlamaCppPTX.lean` for the loader pattern)
5. Re-bench microbench → confirm hesper µs/call ≤ llama.cpp µs/call

Don't start (1) until Sessions 1-5 produce a kernel that's
**algorithmically** equivalent to llama.cpp's vec kernel.  Optimising
PTX of an inferior algorithm wastes time.

### Running llama.cpp's PTX in our microbench (idea, no code yet)

Hesper already has a llama.cpp PTX loader (`Hesper/Backend/LlamaCppPTX.lean`)
that reads `mmvq.ptx` / `quantize.ptx`, calls `cuModuleLoadData`,
resolves `cuModuleGetFunction` by mangled symbol, and provides typed
launch helpers.  **The same machinery can host
`flash_attn_ext_vec`** as soon as we have:
1. The mangled symbol of the desired template instantiation
   (D=256, ncols=1, K=GGML_TYPE_F16, V=GGML_TYPE_F16, no_softcap)
2. A `flash.ptx` file, obtained one of:
   - `nvcc -ptx llama.cpp/ggml/src/ggml-cuda/fattn-vec.cu -...` (need
     to fix include paths)
   - `cuobjdump --dump-ptx llama.cpp/build/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/fattn-vec.cu.o`
   - or extract from the loaded shared object at runtime

Once both are in place, extend `cuda-flashattn-vec-parity` with a
**third bench column** (`llama_us`) that runs llama.cpp's PTX with
the same Q/K/V/params buffers.  Resulting table:

```
cacheLen=N: legacy=A µs   vec=B µs   llama_ptx=C µs
                 ratio_legacy_vs_llama=A/C   ratio_vec_vs_llama=B/C
```

This lets us measure the **algorithmic-only delta** (vec vs llama
both running on the same GPU through cuLaunchKernel, so launch
overhead cancels) and gives a hard ceiling for what hesper kernel
work can achieve.  If even the ShaderM-equivalent algorithm runs at
> 1.2× of llama.cpp's PTX, the residual is PTX codegen — start
Section "After Session 5".

This is a simple ~50-line addition to the test file once the PTX
file and symbol are in hand.  Adding to Session 5 (final
measurement) deliverables.

### llama.cpp PTX in microbench — RESULT (2026-04-26)

Implemented (#262).  PTX extracted via:
```bash
cuobjdump --extract-ptx fattn-vec-instance-f16-f16.cu.5.sm_80.ptx \
  llama.cpp/build/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/fattn-vec-instance-f16-f16.cu.o
sed -i 's/^\.version 8\.7/.version 8.6/' fattn_vec_f16f16.ptx  # driver 565.77 caps at PTX 8.6
mv fattn_vec_f16f16.ptx /tmp/llamacpp_ptx/
```

Mangled symbol: `_Z18flash_attn_ext_vecILi256ELi1EL9ggml_type1ELS0_1ELb0EE...`
(D=256, ncols=1, K=f16, V=f16, no_softcap — `Hesper.LlamaCppPTX.fattnVecF16F16D256Symbol`).

Launch from Lean: `Hesper.LlamaCppPTX.launchFlashAttnVecF16F16D256` packs
the 38-param signature; uses `cuLaunchKernelRaw` directly (no ShaderM /
PTX-hash / module-cache path).

Result table (Gemma 4 geometry, nH=8, nKV=1, D=256, RTX 4070 Ti, 200 iters):

```
cacheLen | legacy µs | vec µs  | llama µs | vec/llama
   8     |  150.5    | 103.7   |   4.2    |   24.8×
  32     |  150.5    | 104.2   |   4.2    |   24.7×
  64     |  150.4    | 104.5   |   4.2    |   25.0×
 128     |  150.7    | 112.6   |   4.2    |   26.6×
 200     |  155.2    | 174.7   |   6.2    |   28.3×
```

**Interpretation**:
- llama.cpp PTX is **24–28× faster** than hesper's vec kernel measured as
  wall-clock per call. The doc 59 "5.4× gap" estimate (from
  kernel_compare graphs-ON) was a different measurement — that one looks
  only at graph-ON GPU kernel time, while this one is per-call wall
  including the GPUBackend.execute path on the launch side.
- llama_us is roughly flat across cacheLen 8–128 (~4.2 µs) — kernel work
  fits in a single 256-thread block reading ≤ 128 K positions. At
  cacheLen=200 it grows to 6.2 µs (one extra warp wave through the K loop).
- hesper vec is roughly flat 8–128 (~104 µs) too. The absolute *delta*
  between hesper and llama is roughly constant (~100 µs) across cacheLen,
  while llama itself only changes by ~2 µs (8→200). So the constant 100 µs
  is **not pure GPU** — if it were, hesper would scale similarly with
  cacheLen. It's most likely a mix of:
  - GPUBackend.execute host path (PTX hash, cache lookup, arg pack)
  - Kernel launch+sync overhead amortised differently than llama's raw
    cuLaunchKernel
  - Some genuinely slower kernel time
  Need follow-up: per-call sync, or HESPER_KERNEL_TRACE to break out
  cuLaunchKernel time vs GPU time.
- At cacheLen=200, hesper vec degrades to 174 µs while legacy stays at
  155 µs (the cross-over noted in earlier microbench runs). Likely cause:
  vec's 128-thread block doesn't cover all K positions in one warp wave
  past cacheLen=192, so the inner K loop runs multiple iterations on the
  same 32-thread axis. Session 2 (split-K, multiple blocks per head) is
  the natural fix.

**Action items unblocked by this measurement**:
1. The PTX-level closure work (post-Session 5) now has a working harness.
   The same path can host any other llama.cpp kernel for direct timing.
2. Session 2 split-K gates on closing the cacheLen=200 regression.
3. Session 3-4 f16 K/V conversion will be measured against the now-loaded
   llama.cpp kernel as a stable upper-bound reference.
4. Real Gemma 4 decode TPS comparison after Session 5 remains the final
   verdict, since microbench wall-clock includes per-call host costs that
   aren't always present in the captured-graph decode path.

### Raw-launch column + ncu profile (2026-04-27, #263 + #264)

Added a 4th microbench column ("raw") that compiles the vec ShaderM once,
takes the resulting CUfunction, pre-resolves buffer args, and times bare
`cuLaunchKernel` in the iteration loop — bypassing GPUBackend.execute's
ShaderM-exec / preHash / cache-lookup / buffer-name resolution path. Also
added `cuda-flashattn-ncu-driver` (slim single-kernel driver, easier to
target with `ncu --kernel-name`).

Result table (Gemma 4 geometry, RTX 4070 Ti, 200 iters, microbench):

```
cacheLen | legacy | vec   | raw    | llama | vec/raw  | raw/llama
   8     | 149.9  | 103.7 |   8.4  |   4.3 |  12.3×   |   2.0×
  32     | 149.9  | 103.9 |  29.4  |   4.3 |   3.5×   |   6.8×
  64     | 149.8  | 104.1 |  57.0  |   4.3 |   1.8×   |  13.2×
 128     | 151.2  | 112.7 | 107.5  |   3.9 |   1.05×  |  27.5×
 200     | 151.2  | 158.8 | 158.3  |   5.8 |   1.00×  |  27.3×
```

**Decomposition**:
- **`vec − raw` ≈ 95 µs at short cacheLen, → 0 at cacheLen=128+**: pure
  GPUBackend.execute host-path overhead (ShaderM eval ×2, preHash, name
  resolve). Vanishes once GPU work exceeds it. NOT the bottleneck for the
  long-cacheLen decode regime, but a fixed tax for short-cacheLen calls.
- **`raw − llama`**: pure GPU kernel-time gap. Scales linearly with K
  (8→158 µs over cacheLen 8→200) while llama is flat (3.9→5.8 µs).
  **27× at cacheLen=128.** This is the algorithm/implementation gap.

### ncu metrics (cacheLen=128, 30 launches, last 3 averaged)

| Metric | hesper raw | llama.cpp | h/l |
|---|---|---|---|
| `gpu__time_duration` | 184.9 µs | 5.82 µs | **31.8×** |
| `sm__throughput.pct` | 0.93% | 1.73% | 0.5× |
| `gpu__compute_memory_throughput.pct` | 0.93% | 7.23% | 7.8× |
| `smsp__warps_active.pct` | 8.33% | 8.30% | ~1× |
| `launch__waves_per_multiprocessor` | 0.01 | 0.07 | 7× |
| `launch__registers_per_thread` | 21 | 201 | 9.6× |
| `launch__shared_mem_per_block_static` | 2.18 KB | 16.64 KB | 7.6× |
| `launch__grid_size` | 8 | 8 | 1× |

Both kernels launch 8 blocks (one per Q-head, matching `numHeads=8`,
`numKVHeads=1`). RTX 4070 Ti has **80 SMs**, so both are deeply
under-occupied (waves 0.07 means ~6 SMs busy, 74 idle). **Increasing
SM utilisation = Session 2 (split-K) — same on both kernels in
principle.**

But the bigger surprise is the instruction count:

| Metric | hesper | llama | h/l |
|---|---|---|---|
| `inst_executed.sum` | 872,256 | 55,520 | **15.7×** |
| `pipe_fma.sum` | 389,472 | 41,184 | **9.5×** |
| `op_global_ld.sum` | 20,576 | 2,336 | **8.8×** |
| `op_shared_ld.sum` | 41,024 | 1,344 | **30.5×** |
| `op_shared_st.sum` | 16,512 | 448 | **36.9×** |
| `pipe_lsu.sum` | 119,168 | 5,600 | **21.3×** |

**Conclusion**: the gap is *NOT* primarily SM utilisation (both run 8
blocks). It's that hesper's vec kernel **executes 16× more
instructions** to compute the same result:
- 30× more shared-memory traffic — hesper round-trips through smem to
  reduce across warps; llama keeps results in registers and uses warp
  shuffle.
- 9× more FMA — likely no CSE, no loop unroll, redundant rsqrt /
  rescale / max recomputation per K iteration.
- 9× more global load — f32 K/V instead of f16 (Session 3-4), plus
  not vectorising loads.

**Per-thread register count** is also revealing: hesper 21 vs llama
**201**. llama's kernel keeps a huge VKQ accumulator + KQ_max/KQ_sum
per-thread in registers, never spilling to smem. hesper's kernel leans
on smem because Lean ShaderM doesn't naturally express per-thread
multi-element accumulators that llama's CUDA template specialises into.

**Implication for Sessions 2–5**:
- Session 2 (split-K) helps SM utilisation, but both hesper and llama
  start at the same 8-block / 0.07-wave point — **split-K alone won't
  close the 32× gap**, because the per-thread instruction-count gap
  multiplies on top.
- The real lever is **register-resident accumulators + warp-shuffle
  reduce** (currently hidden in the kernel as smem ping-pong). That's
  partly Session 1's intent but didn't go deep enough — the Session 1
  vec kernel still uses `ShaderM.writeWorkgroup` cross-warp gather
  (line "shared_warp_sums" smem write) which the metrics catch as 30×
  shared traffic.
- f32 → f16 K/V (Sessions 3-4) only addresses the 8.8× global-load gap.

**Updated session priorities**:
1. **Session 2′ (revised)**: refactor vec inner loop to keep
   `KQ_max`/`KQ_sum`/`VKQ[]` in `ShaderM.var` (PTX register) only,
   and reduce across warps via *only* `Exp.subgroupAdd`/`subgroupMax`
   (no smem gather). This shifts work from LSU to ALU and is the main
   instruction-count lever — could give 10× or more on its own.
   Verify with ncu re-run that `op_shared_*` drops 10×+.
2. **Session 3 (K f16) + 4 (V f16)**: addresses the 8.8× global-load
   gap. Less impactful than Session 2′ but cumulative.
3. **Session 5**: split-K + final TPS measurement.

**Decision**: Session 2 (split-K) as originally planned is *not* the
highest lever. Renaming to "Session 2′: per-thread accumulators +
warp-only reduce". Split-K becomes a follow-up if SM utilisation
turns out to be the residual gap after instruction count is closed.
