# Q4_K gate+up shape diff: hesper 4-row vs llama ncols_dst=2

## Measurement (60-token decode, 2026-04-28)

```
hesper k_1387739930045770 (ffnNormGateUp section)
   grid=(2560,1,1) block=(128,1,1) regs=35 smem=2880  68.45 µs/call × 60/tok = 4.10 ms/tok
   = fusedQ4KMGateUpDP4A4RowKernel (4 rows × 1 warp each, smem-shared input)

llama mul_mat_vec_q<Q4_K, ncols_dst=2>
   grid=(5120,1,1) block=(32,4,1) regs=66 smem=1536   33.78 µs/call (gate+up combined)
```

**hesper: 68.45 µs vs llama: 33.78 µs ≈ 2.03× slow.**

## Why hesper is slow

| | rows/WG | warps/WG | K-split | wave count | per-warp K work |
|---|---|---|---|---|---|
| llama `ncols_dst=2` | 1 row × 2 cols | 4 cooperative | each warp = K/4 | grid=5120 → 61 waves | K=2560/4 = 640 per warp |
| hesper `4Row` | 4 rows | 1 each | warp does whole K | grid=2560 → 30 waves | K=2560 per warp |

- hesper の per-warp 仕事量が **4×**多い。
- wave 数が **0.5×**しかない。
- smem input reuse は有るが、**4-row × 1-warp** 構造が出力当たりの時間を間延びさせている。

## llama's `ncols_dst=2` design (reference)

`ggml/src/ggml-cuda/mmvq.cu` の `mul_mat_vec_q<Q4_K, /*ncols_dst=*/2, /*has_fusion=*/false, /*type_acc=*/false>`:

- 1 WG = 128 thread = 4 warps cooperative on K
- 1 WG が 1 row の output[outIdx][0..1] (= gate と up の同じ row) を計算
  - つまり **同じ K input を smem で共有しつつ 2 つの weight tensor (gate, up) で別々に dot**
  - K loop の inner で `acc[0]` と `acc[1]` 2 列を同時に積算 → 入力読み再利用の代わりに **register pressure を許容**
- Cross-warp reduce: 各 warp の partial を warp 0 経由で smem 1 場所に集める
- grid = outDim × ncols_dst の合計でなく、**outDim 単位で WG 1 つ × 列 2 つ**

## ROI

- ffn_gate+up が 4.10 ms/tok の最大 Q4_K bucket。
- llama 並み (33.78 µs/call) になれば **4.10 → 2.03 ms/tok ≈ -2.07 ms/tok**
- 9.0 → 6.93 ms/tok kernel = **theoretical TPS 89.6 → 116 (-23 ms wall/60tok)**
- 実測増分は CUDA Graphs ON のためカーネル時間以外の wall も影響するが、+10〜15 TPS は十分視野。

## Plan

新 kernel `fusedQ4KMGateUpDP4A4WarpNcols2Kernel`:
- block=(32,4,1) = 128 thread = 4 warps coop K
- grid=(outDim, 1, 1) = 10240 → wave 122 (現 30 の 4×)
- smem に Q8_1 input 1 セット
- 各 warp が K/4 = 640 を担当
- Inner loop: `acc_gate += dp4a(w_gate, q8); acc_up += dp4a(w_up, q8)` — 2 列を同じ K iteration で
- Cross-warp reduce: 4 warps の (acc_gate, acc_up) を warp 0 で集約
- Tail: `output[outIdx] = silu(acc_gate) * acc_up` (同じ row、SiLU+mul fused)

注意:
- 現 4-row 1-warp は wO/post/PLE には NG (`feedback_q4k_4row_wave_count.md`) だが gate+up は outDim=10240 で wave 十分。
- ncols_dst=2 は両 tensor の K と outDim が一致する必要 — gate/up は ggml と同じ条件 OK。

## Test plan

1. parity test: `Examples/DSL/Gemma4FFNParity.lean` (already存在) を新 variant 経由で実行
2. micro-bench: per-call µs 計測
3. e2e: `gemma4-cuda "Hello" 30` で `Hello! How can I help you today? 😊` 出力確認
4. TPS: 89.6 → ?
