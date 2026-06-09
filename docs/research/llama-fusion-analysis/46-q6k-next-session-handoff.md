# 46 — Q6_K 次セッションへのハンドオフ

*Written 2026-04-24.*

## 状況

Q6_K ffn_down matmul を llama.cpp と PTX+ncu で比較し、真のボトル
ネックが特定された。PTX peephole はここまで (doc 41-44)、次は
**launch config と memory access pattern** を変える実装フェーズ。

## 現在の数値 (baseline)

| 指標 | hesper 4-row | llama.cpp |
|---|---:|---:|
| ms/decode (ffn_down) | 1.32 | 1.20 |
| ratio | 1.10× | 1.00× |
| inner-loop PTX | 203 inst | 79 inst |
| grid / block | (640,1,1) / 128 | (2560,1,1) / 128 |

Token parity 維持中: "Hello world how are you" → 236881 "?"

## ncu で判明した真のボトルネック (doc 45)

| 要因 | 測定値 | 潜在削減 |
|---|---|---:|
| **Tail effect** | grid 640 = 480 full + 160 partial = 実行の50% がテール | **−50%** |
| **Uncoalesced global** | 11% (243k/2.19M sectors) — byte-granularity load の犠牲 | −10% |
| **Uncoalesced shared** | 30% (205k/672k wavefronts) — ×9 stride の bank conflict | −5% |
| FP32 peak 1% | compute idle | — |
| Occupancy 52.5%/66.7% | shared memory 制限 | — |

**Bottleneck: Memory > Compute** (ncu の SoL chart)。compute は 1% しか
使ってないので、ALU 改善は無駄。memory pattern を直せ。

## 実装プラン (優先度順)

### A. Tail effect 解消 (最大 ROI)

RTX 4070 Ti: 60 SM, 4070 Ti で 1 wave = 480 blocks (4-row kernel で smem
66.7% occupancy の場合)。

**選択肢:**

1. **1-row kernel に戻す** — grid=2560, 5.3 waves, tail 5%
   - Pros: llama.cpp と同じ launch config, tail 縮小
   - Cons: smem input sharing を失う → 各 block が 2.56KB 再読
   - llama.cpp が 1-row で成立している事実 → 試す価値大
   - hesper に既存: `fusedQ6KLinearDP4AKernel` (Linear.lean:2675)

2. **8-row kernel** — grid=320, partial wave 320 blocks
   - 4-row よりマシだが依然 partial
   - 実装コスト: 4-row の改変

3. **4-row のまま block size を変える**
   - smem 使用量を減らして 720-block/wave alignment
   - 実装コスト: 中

**推奨:** 1 を試す (一番簡単に A/B 比較できる)。
dispatcher (`Linear.lean:3931`) で 4-row → 1-row に切り替えて
token parity + ncu + kernel_compare 取る。

### B. Uncoalesced global 解消

現状: hesper は scale を `readByte` で 1-byte ずつ読む (doc 42-43)。
llama.cpp は 4 scale を 1 u32 load + bfe で 1 sector に収める。

**実装:**

`Linear.lean:3135` 近辺の `readByte` 用法:
```lean
let sc0Byte ← readByte blockByteBase (Exp.add (Exp.litU32 192) scaleOff)
let sc1Byte ← readByte blockByteBase (Exp.add (Exp.litU32 192) (Exp.add scaleOff (Exp.litU32 4)))
```

これを u32 1 つで読んで bfe 2 回に変える:

```lean
-- Q6_K block byte 192..208 is 16 × i8 scales.  Read 4 bytes at a time
-- (each warp thread hits a 4-byte aligned offset within the 16-byte range).
-- scaleOff is in [0, 4), so (192 + scaleOff) is NOT 4-byte aligned; read
-- the aligned u32 covering bytes 192..195 or 196..199 etc and bfe.
let scU32 ← readBuffer (ty := .scalar .u32) "weights"
  (shiftRight (add blockByteBase (litU32 192)) (litU32 2))  -- u32 word idx
-- extract scaleOff-th byte: bfe.u32 dst, scU32, scaleOff*8, 8
let sc0Byte := Exp.extractBits scU32 (mul scaleOff (litU32 8)) (litU32 8)
...
```

注意: `blockByteBase` が 2-byte aligned なので (210 × blockIdx), 192
offset を足しても 2-byte aligned のまま。u32 load は **HW が misalign
を tolerate** するが、coalescing が落ちる。**実は llama.cpp は block
pointer が 4-byte aligned になるよう padding している**。
  → まず `block_q6_K*` の alignment を確認して、alignment を揃える
    ことから始めるのが正解かもしれない。

**推奨:** A を先に試す。B は alignment 絡みで慎重に。

### C. SMEM bank conflict 解消

Q8_1 input staging が `s_input_q8[q8Block × 9 + elem]` で ×9 stride。
32 banks の場合 bank = (idx × 4) % 32、×9 はmod 32 で 9/25/17/1/18/...
でほぼ均等に散るように見えるが 9 と 32 の GCD=1 で実は良い? ncu は
30% uncoalesced と言うので何か詰まってる。

要調査:
- `stride=10` vs `stride=9` vs `stride=12` を A/B
- それとも `q8Sub0 + elemOff` の elem index 順序が悪い?

**推奨:** A, B の後で測定。

## 検証プロトコル

各変更で:

1. **Parity check** — 毎回実行:
   ```bash
   HESPER_DP4A=1 HESPER_LLAMA_GRAPHS=0 \
     lake exe gemma4-llama-prefill-skeleton \
     data/gemma-4-e4b-it-Q4_K_M.gguf "Hello world how are you" 2
   # Expected: completion: "?"
   ```

2. **ms/decode measurement**:
   ```bash
   mkdir -p /tmp/nsys_after && HESPER_DP4A=1 HESPER_LLAMA_GRAPHS=0 \
     nsys profile -t cuda --stats=false \
     -o /tmp/nsys_after/run -f true \
     lake exe gemma4-llama-prefill-skeleton \
     data/gemma-4-e4b-it-Q4_K_M.gguf "Hello world how are you" 10

   # Find Q6_K ffn_down by instance count (check if 294 is still correct
   # after the kernel body change — may have changed)
   python3 -c "
   import csv, subprocess, io
   out = subprocess.run(['nsys', 'stats', '--force-export=true',
     '--report', 'cuda_gpu_kern_sum', '--format', 'csv',
     '/tmp/nsys_after/run.nsys-rep'], capture_output=True, text=True).stdout
   for row in csv.reader(io.StringIO(out)):
       if len(row) >= 9 and 'k_' in row[-1]:
           try: inst = int(row[2])
           except: continue
           if 200 < inst < 400:  # ffn_down range
               print(f'{int(row[1])/1e6/10:.3f} ms/dec  inst={inst}  {row[-1]}')
   "
   ```

3. **ncu roofline check** (effects on memory-boundedness):
   ```bash
   NCU=/nix/store/.../target/linux-desktop-glibc_2_11_3-x64/ncu
   # hash 再取得 (kernel PTX body 変更で hash 変わる):
   HESPER_DP4A=1 HESPER_LLAMA_GRAPHS=0 HESPER_KERNEL_TRACE=1 \
     lake exe gemma4-llama-prefill-skeleton \
     data/gemma-4-e4b-it-Q4_K_M.gguf "Hello world how are you" 2 2>&1 \
     | grep "grid=(640," | sort -u  # or (2560,...) if 1-row
   # Then ncu with the new hash
   $NCU --set detailed --kernel-name regex:"k_NEWHASH" \
     --launch-skip 40 --launch-count 20 \
     -o /tmp/q6k_after_ncu -f \
     lake exe gemma4-llama-prefill-skeleton \
     data/gemma-4-e4b-it-Q4_K_M.gguf "Hello world how are you" 5
   ```

## 環境メモ

- **ncu permission**: `sudo rmmod nvidia_uvm nvidia` 済み。
  module reload 後 `NVreg_RestrictProfilingToAdminUsers=0` 反映 →
  sudo なしで ncu 使える
- **ncu binary path**: `/nix/store/06xmh0nkqff42zljxgjcjikjfh9ia8vc-cuda12.8-nsight_compute-2025.1.1.2/target/linux-desktop-glibc_2_11_3-x64/ncu` を直接呼ぶ (wrapper script は "Failed to add rules search path" エラーで --csv 出力が空になる)
- **前回 section file 問題の workaround**: 直接 binary を呼ぶと --csv の出力は stderr に混ざって出てくる。grep で抽出可

## 既存の関連ファイル

- **Q6_K 4-row kernel**: `Hesper/Layers/Linear.lean:3106` (`fusedQ6KLinearDP4A4RowKernel`)
- **Q6_K 2-row kernel**: `Hesper/Layers/Linear.lean:2912` (`fusedQ6KLinearDP4A2RowKernel`)
- **Q6_K 1-row kernel**: `Hesper/Layers/Linear.lean:2675` (`fusedQ6KLinearDP4AKernel`)
- **dispatcher**: `Hesper/Layers/Linear.lean:3931` (`forwardDP4A` の Q6_K 分岐)
- **readByte/readU16/read4Bytes**: `Linear.lean:3135-3170` 付近 (3 か所の kernel で copy)
- **llama.cpp PTX override**: `Hesper/Backend/LlamaCppPTX.lean`
  - 有効化: `HESPER_USE_LLAMACPP_PTX=1 HESPER_LLAMACPP_Q4K=0 HESPER_LLAMACPP_Q6K=1`
  - Prefill skeleton から autoInstall 済み: `Examples/Gemma4LlamaPrefillSkeleton.lean`

## 参照 docs

- `41-q6k-ld-global-diff.md` — 元の PTX 比較
- `42-q6k-u8-u16-primitives.md` — ShaderM u8/u16 primitive 追加
- `43-q6k-fp16-cvt-and-real-gap.md` — fp16ToF32 → cvt.f32.f16 化
- `44-q6k-addr-cse.md` — PTX peephole と diminishing returns
- `45-q6k-ncu-rootcause.md` — ncu で memory-bound + tail effect 特定
- `46-q6k-next-session-handoff.md` — このファイル

## 期待される最終到達点

| stage | ms/decode | ratio |
|---|---:|---:|
| current (doc 44) | 1.32 | 1.10× |
| +A (1-row, tail fix) | ~0.75 | 0.63× |
| +B (batch scales) | ~0.68 | 0.57× |
| +C (smem pad) | ~0.65 | 0.54× |
| llama.cpp baseline | 1.20 | 1.00× |

A だけで llama.cpp を抜く可能性が高い。B/C は余技。
