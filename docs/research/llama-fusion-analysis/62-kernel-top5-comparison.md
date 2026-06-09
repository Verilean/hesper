# Kernel-by-kernel comparison vs llama.cpp (graphs OFF, 60 tok decode)

Generated 2026-04-27 with hesper at commit 8e47536 (f16 K/V cache + V11 default).

## Setup

```bash
# hesper graphs OFF (65.2 TPS, 920 ms wall)
HESPER_DP4A=1 HESPER_CHAT=1 nsys profile -o /tmp/hesper_f16_off \
  ./.lake/build/bin/gemma4-cuda data/gemma-4-e4b-it-Q4_K_M.gguf \
  "Hello, how are you doing today? Tell me a story." 60

# llama.cpp graphs OFF (~110 TPS)
GGML_CUDA_DISABLE_GRAPHS=1 nsys profile -o /tmp/lc_f16_off \
  llama.cpp/build/bin/llama-cli --no-warmup --no-mmap -no-cnv -fa on -ngl 99 \
  -m data/gemma-4-e4b-it-Q4_K_M.gguf \
  -p "Hello, how are you doing today? Tell me a story." -n 60 \
  --temp 0 --top-k 1
```

## Top 15 hesper kernel classes

| class | total ms | inst | avg µs | #hashes |
|---|---|---|---|---|
| Q4_K wO matmul                         | **253.18** | 8785  |  28.82 | 5  |
| Q6_K ffn_down 1-row matmul             |   75.51    | 1756  |  43.00 | 1  |
| Q6_K lm_head                           |   72.99    |   84  | 868.95 | 1  |
| Q4_K perLayer matmul (4096 outDim)     |   68.38    |   60  |1139.60 | 1  |
| FA combine / final reduce              |   67.09    |10159  |   6.60 | 5  |
| b=32 gx=2560 gy=24 (FA partial decode) |   28.92    |  105  | 275.43 | 4  |
| RoPE / small kernel                    |   19.74    | 2509  |   7.87 | 42 |
| Q4_K small matmul (prefill)            |   17.70    | 2510  |   7.05 | 2  |
| RMSNorm 1024-block                     |   14.55    | 1432  |  10.16 | 1  |
| fused PLE post-scale                   |   11.63    | 2509  |   4.64 | 1  |
| b=32 gx=10752 gy=1                     |    9.84    |   84  | 117.19 | 1  |
| Q4_K wK/wV 1-warp matmul               |    9.09    | 1464  |   6.21 | 2  |
| RMSNorm 2048-block                     |    7.37    |  418  |  17.64 | 1  |
| Q4_K wQ matmul                         |    6.41    |  660  |   9.72 | 1  |
| Q4_K wQ 1-warp matmul (prefill)        |    6.14    |   35  | 175.56 | 2  |

## Top 12 llama.cpp kernel classes

| class | total ms | inst | avg µs | #hashes |
|---|---|---|---|---|
| Q4_K matmul                            | **272.21** |16430  |  16.57 | 5  |
| Q6_K matmul                            |  131.20    | 2013  |  65.18 | 3  |
| RMSNorm                                |   46.69    |18719  |   2.49 | 4  |
| quantize_q8_1                          |   18.81    |18443  |   1.02 | 1  |
| FlashAttn vec (decode)                 |   13.36    | 2065  |   6.47 | 1  |
| MMQ matmul (prefill)                   |    8.37    |  674  |  12.42 | 4  |
| FlashAttn (other)                      |    8.29    | 1078  |   7.69 | 12 |
| binary bcast (add/mul)                 |    6.75    | 5329  |   1.27 | 2  |
| MMV f32 matmul                         |    6.71    |   59  | 113.75 | 1  |
| RoPE                                   |    4.46    | 4092  |   1.09 | 2  |
| KV cache write                         |    3.47    | 2976  |   1.16 | 1  |
| GELU/unary                             |    2.32    | 2603  |   0.89 | 1  |

## Side-by-side roll-up — **Top 5 levers**

| category                  | hesper ms | llama.cpp ms | ratio  | hs avg µs | lc avg µs | hs inst | lc inst |
|---------------------------|----------:|-------------:|:-----:|----------:|----------:|--------:|--------:|
| **FlashAttn**             |   68.58   |    21.65     | **3.17x** |  6.45    |  6.89    | 10626   |  3143   |
| **RoPE / KV write**       |   19.74   |     7.93     | **2.49x** |  7.87    |  1.12    |  2509   |  7068   |
| Q4_K matmul (decode hot)  |  354.27   |   272.21     | 1.30x  | 27.87    | 16.57    | 12710   | 16430   |
| pointwise / embed         |   12.38   |     9.07     | 1.37x  |  4.68    |  1.14    |  2646   |  7932   |
| Q6_K matmul               |  148.50   |   131.20     | 1.13x  | 80.71    | 65.18    |  1840   |  2013   |
| **(other / unclassified)** | 76.79   |    16.99     | **4.52x** |       |          |         |          |
| RMSNorm                   |   22.22   |    46.69     | 0.48x  | 11.74    |  2.49    |  1892   | 18719   |
| quantize_q8_1             |   12.00   |    18.81     | 0.64x  |  1.15    |  1.02    | 10444   | 18443   |
| **GRAND TOTAL**           |  714.48   |   524.55     | **1.36x** |        |         |         |         |

## 5 つの優先レバー

### 1. FlashAttn — 3.17x slower (47 ms gap = 0.78 ms/tok = +5 TPS potential)

V11 を入れても **69 ms vs 22 ms**. inst 数が 10626 vs 3143 (3.4×). 原因:
- split-K = 8 で **partial + combine の 2 dispatch / FA call** → llama.cpp は 1 dispatch (`flash_attn_ext_vec`)
- combine kernel が大量に呼ばれてる: "FA combine / final reduce" が 67 ms
- llama.cpp は decode で **flash_attn_ext_vec** の 1 段で完結

**改善案**:
- split-K を **動的に選ぶ**: cacheLen < 64 では split=1 (V7 互換) にして dispatch 半減
- combine kernel と partial kernel を 1 つに fuse (cacheLen ≤ 32 のとき)

### 2. RoPE / KV write — 2.49x slower (12 ms gap = 0.2 ms/tok = +1 TPS)

我々: 2509 inst × 7.87 µs = 20 ms.  
llama.cpp: 7068 inst × 1.12 µs = 8 ms (ただし我々の 1 dispatch を **3 dispatch (RoPE-Q + RoPE-K + k_set_rows)** に分散).

per-call は llama.cpp が **7倍速い** (1.12 µs vs 7.87 µs). 我々の `fusedRopeKAndCacheWriteBatchKernelF16` が重い:
- 4 つの f32 read + RoPE 計算 + 2 つの f16 write
- llama.cpp の `k_set_rows` は単純な copy (RoPE は別 kernel `rope_neox`)

**改善案**: 
- 大きすぎる fused kernel を分割するか、 もっと小さい thread granularity (1 thread = 4 dim pair など) に
- ncu 取って memory throughput 確認

### 3. Q4_K matmul — 1.30x slower (82 ms gap = 1.4 ms/tok = +9 TPS!) **最大レバー**

per-call avg 28 µs vs 17 µs (1.7x slower per call). 全 12710 dispatches に効く。これは memory にも task #47 として in_progress.

主犯: Q4_K wO matmul = 253 ms (我々最大の塊). llama.cpp の `mul_mat_vec_q<Q4_K>` の方が出力幅 outDim 当たり 60-65% の時間。

**改善案**:
- ncu で hesper Q4_K kernel の memory throughput / IPC を比較 (memory `project_real_perf_table_2026_04_25.md` で 1.74× kernel speed gap が指摘されてる)
- llama.cpp の mmvq.cu の inner-K iteration unroll 度合いを参考に
- **task #47 の継続**

### 4. unclassified 77 ms — 何かの未確認 kernel

classifier がカバーできてない hash 多数 → 27 ms が `b=32 gx=2560 gy=24` (4 hashes), 9.8 ms が `b=32 gx=10752 gy=1`. ↑これらは**prefill 用 kernel**っぽい (gy>1 なら prefill). decode 中も走ってるなら無駄.

**改善案**: kernel trace を per-section 区別して分類完成、 prefill 用が decode で走ってないか確認 (走ってたら最大 28 ms 削減).

### 5. Q6_K lm_head — 73 ms (8% wall)

1 inst × 870 µs/token (= 1.4 ms/tok). 巨大 vocab (262144) × 2560 hidden の Q6_K matmul. これは **正しく重い**.

llama.cpp の Q6_K (131 ms total / 2013 inst = 65 µs avg) の方が短い: lm_head 含めても avg 65 µs しかない. 多分 llama.cpp は lm_head を **batched (top_k で全 vocab 計算しない)** か、 **mmq path** (block batched) に流してる.

**改善案**: llama.cpp の lm_head dispatch がどの kernel に行ってるか調査. `mul_mat_q` (MMQ) に分岐してる可能性. **decode で argmax だけなら top vocab まで計算する必要が無い** が、 hesper も llama.cpp も full vocab 出してるので fair compare.

## Wall vs kernel time の差

- hesper: 920 ms wall, 714 ms kernel (= 78% GPU busy. 22% idle = 206 ms)
- llama.cpp: ?? ms wall, 525 ms kernel

llama.cpp の wall を取って GPU busy% で比較すると、 host overhead の差が分かる. (TPS 65 vs 110 = wall 1.69x. kernel 1.36x. 残り 1.24x が host overhead = argmax DtoH = `feedback_dtoh_is_drain.md` の通り).

## TPS 改善見積

| 改善                          | 削減 (ms/tok) | TPS gain |
|-------------------------------|--------------:|---------:|
| Q4_K matmul → 17µs/call (#1)  |  +1.4 ms      | +9 TPS   |
| FlashAttn → split=1 dynamic  |  +0.8 ms      | +5 TPS   |
| Q6_K lm_head → 600µs         |  +0.5 ms      | +3 TPS   |
| RoPE-K → 3µs/call            |  +0.2 ms      | +1 TPS   |
| **合計** (kernel-only)       |  +2.9 ms      | **+18 TPS** |

graphs OFF baseline 65 TPS + 18 = **83 TPS**. 残りは host overhead (DtoH = device-side argmax ですでに #250 で取り組み済).

graphs ON では現在 94 TPS なので, kernel 改善で **graphs ON 110+ TPS = llama.cpp 同等**になる見込み.

## 計測ファイル

- `/tmp/hesper_f16_off.nsys-rep` (920 ms wall)
- `/tmp/lc_f16_off.nsys-rep`
- `/tmp/hs_labels.tsv` (hash → grid/block, 311 行)
- `/tmp/cmp4.py` (集計スクリプト)
