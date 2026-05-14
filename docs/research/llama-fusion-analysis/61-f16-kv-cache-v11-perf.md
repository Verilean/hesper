# f16 K/V cache + V11 production wiring — TPS analysis

## TL;DR

- f16 cache + V11 wiring **大きく効いた** が **graphs ON 限定**.
- graphs OFF は **argmax DtoH (8.6 ms/tok = 46% of wall)** に律速されてて kernel 高速化が拾えない.
- graphs ON だと 67.6 → **93.6 TPS** (+38%).
- llama.cpp 110 TPS まで残り 16%.

## Bench (Gemma 4 E4B Q4_K_M, RTX 4070 Ti, 60 tok decode)

| | f32 baseline (前) | f16 + V11 (今) | Δ |
|---|---|---|---|
| graphs OFF | 60.7 | **65.2** | +7.4% |
| graphs ON  | 67.6 | **93.6** | **+38%** |

## OFF → ON gap の正体 (nsys API summary)

graphs OFF (920 ms wall):
| API | total | calls | avg/call |
|---|---|---|---|
| **cuMemcpyDtoH_v2** | **517 ms (46%)** | 60 | **8.6 ms** |
| cuMemcpyHtoDAsync | 320 ms (28%) | 1169 | 274 µs |
| cuLaunchKernel | 75 ms (7%) | 59482 | 1.3 µs |

graphs ON (641 ms wall):
| API | total | calls | avg/call |
|---|---|---|---|
| cuLaunchKernel | 6.8 ms | 5204 | 1.3 µs |
| **cuGraphLaunch** | **1.9 ms** | 60 | **31 µs** |
| cuGraphInstantiate | 1.9 ms | 1 | once |

`cuMemcpyDtoH_v2` が ON では top 5 に入ってない (graph 内に取り込まれた). これが **46% wall を消した**根本要因.

graphs OFF で 8.6 ms/call は memcpy 自体ではなく **GPU drain の待ち時間**. 4 byte 読むのに 8.6 ms かかるわけがない — argmax の結果を読むには forward 完了を待つ → forward の wall time そのものが現れる.

memory にも記録あり:
- `project_argmax_dtoh_is_28pct.md` (perf children-sort で 28% — 計測法違いだが同じ現象)
- `feedback_dtoh_is_drain.md` (DtoH = drain, host-mapped でも解決せず)

## V11 の効果が graphs ON で効く理由

graphs OFF は GPU drain bound なので、 kernel が速くなっても drain が同じ ⇒ wall 短縮 limited.

graphs ON は graph 全体が submission 1回 (cuGraphLaunch 31 µs/tok) で、 GPU がbacking-off せず連続実行. V11 の kernel 短縮 (~3 ms/tok 削減) がそのまま wall に跳ねる.

## 残り課題

1. **lmHead Q6_K dp4a** が GPU 時間 73 ms / 60 tok = 1.2 ms/tok = **TPS の 9%** を消費している. 1 dispatch / token で 877 µs. これが次の big lever 候補.
2. graphs OFF の DtoH bottleneck は task #250 で device-side argmax が landed しているはずが、 default OFF. HESPER_DEVICE_ARGMAX=1 をデフォルト化検討 (memory `feedback_dtoh_is_drain.md` で「DtoH を cuStreamSync に変えるだけ」とあったが要再検証).
3. llama.cpp 110 TPS まで 16% gap. graphs ON 状態で残るボトルネックは:
   - flashAttn: V11 partial+combine = 26 µs × 168 calls = 4.4 ms/tok
   - lmHead: 877 µs/tok
   - その他 kernels: ~5 ms

## 計測ファイル

- `/tmp/hesper_f16_off.nsys-rep` (graphs OFF, 60 tok)
- `/tmp/hesper_f16_on.nsys-rep` (graphs ON, 60 tok)

## コマンド

```bash
# OFF
HESPER_DP4A=1 HESPER_CHAT=1 \
  ./.lake/build/bin/gemma4-cuda data/gemma-4-e4b-it-Q4_K_M.gguf \
  "Hello, how are you doing today? Tell me a story." 60

# ON
HESPER_DP4A=1 HESPER_CHAT=1 HESPER_CUDA_GRAPHS=1 \
  ./.lake/build/bin/gemma4-cuda data/gemma-4-e4b-it-Q4_K_M.gguf \
  "Hello, how are you doing today? Tell me a story." 60
```
