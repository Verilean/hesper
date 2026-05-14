# Decode bubble の構造的解消計画 (graphs OFF)

2026-04-26.

## 観測 (再確認)

`scripts/nsys_to_chrome_trace.py` で生成した chrome trace
(`/tmp/compare_trace*.json`) を Perfetto で読むと、hesper の decode は
以下の構造になっている:

```
[ launch dispatch x ~880 ] → cuStreamSynchronize(9.8 ms drain)
                              ↑ host が argmax 結果を待つ
                            → [ Lean: embedLookup, embedScale, ...,
                                 perLayerInputPre, attnNorm, qkvProj, ...,
                                 (42 layer 分 × 約 14 section) ]
                              ↑ 5 ms かけて次トークンの forward を組み立てる
                              ↑ この間 GPU は idle
                            → [ launch dispatch x ~880 ] → 次の sync
```

llama.cpp は逆: cuLaunchKernel 大量 → sync (host が GPU の **前** を走る)。
hesper は host が GPU の **後ろ** を走っている。

## 数値 (60-token decode, graphs OFF, "Hello world how are you")

| variant | TPS | sync | GPU busy | gaps>1ms |
|---|---:|---:|---:|---:|
| baseline | 61.7 | 16 | 65.7% | 17 (87 ms) |
| HESPER_DEVICE_ARGMAX=1 | 59.1 | 16 | 65.7% | 17 (87 ms) |
| HESPER_PIPELINED_DECODE=1 | **50.4** | 0 | **56.3%** | **41 (219 ms)** |
| llama-cli | 111.6 | 1640 | 82.1% | (small) |

PIPELINED は sync は消したが bubble が悪化 (17→41) して TPS も悪化。
sync を消すことが目的ではない、host が GPU を待たない構造が必要。

## 根本原因 (`forwardSingleToken` の host-Nat 依存)

`Hesper/Models/Gemma4.lean` の `forwardSingleToken (tokenId : Nat)
(pos : Nat) ...` で host-Nat を要求している箇所:

| # | Lean expression | line | impact | 既存対策 |
|---:|---|---:|---|---|
| 1 | `tokenBytes := uint32ToBytes tokenId.toUInt32`; `writeScalarViaStaging ctx state.tokenBuf …` | 2645-2647 | 毎 forward 先頭で host-side u32 コピー | `skipTokenWrite := true` で skip 可 |
| 2 | `kernelTokenId := match perLayerEmbdMmap with some _ => 0 \| none => tokenId`; `writeScalarViaStaging ctx state.plRawRowBuf …` | 2691-2696 | PLE row id を device 書込 | `skipTokenWrite` で skip 可 |
| 3 | `let rowDevPtr := devPtr + tokenId.toUSize * rowBytesU` | 2699 | **mmap UVA path で row pointer 自体を host-Nat から計算** | 未対策 — UVA path では skipTokenWrite でも計算が host 依存 |
| 4 | `let posBytes := uint32ToBytes pos.toUInt32`; paramsBuf 0 に書込 | 619-620 | RoPE / attn が pos を読む | (graphs ON: advancePosKernel で device-side) |
| 5 | `posF32Bytes ← floatToBytes pos.toFloat`; posF32Buf に書込 | 624-625 | Circuit DSL 動的 offset | (同上) |
| 6 | KV cache `writeScalarViaStaging cacheLen` (= pos+1) | 776 | flashAttn 用 | 同上 |

**graphs ON では既に解決済み**: capture 中に embedLookup の代わりに
argmax が `state.tokenBuf` に device-side で書き込み、advancePosKernel
が `paramsBuf` を inc して、次の replay は device-side で正しい
state を見る。host は何もしない。

**graphs OFF で現状失敗する経路**: pipelined にしても、上記 #3 (UVA
mmap path での row ptr 計算) は host-Nat に依存していて skip
できない。さらに pos 系 (#4-6) も毎 forward で writeScalarViaStaging
が走り、このコピーの順序保証のために stream が暗黙に待たされている
(これが PIPELINED で bubble が増えた原因と推定 — sync を消しても
write の順序待ちが残る)。

## 提案: forwardSingleTokenDeviceFed (新規 entry point)

graphs OFF でも host-Nat 依存を完全排除した経路を作る。

### 必要条件

1. **token id は device-only**: `state.tokenBuf` (および UVA mmap
   path 用に PLE row 選択を device kernel が行う形に) を真実とする。
   host は forwardSingleTokenDeviceFed に **`tokenId` を引数として
   渡さない**。
2. **pos も device-only**: `state.paramsBuf[0]` (= pos) と
   `state.posF32Buf` を真実とする。advancePosKernel の graphs-OFF
   版を導入し、forward の末尾に挿入。
3. **PLE mmap UVA path の host-ptr 計算を kernel 内に移動**: row 選択
   は kernel 内で `tokenBuf` を読んで `devPtr + tokenId * rowBytes`
   を計算する形にする。これは UVA mapping の特性上 device-side で
   できる (devPtr 自体は host-known で kernel に定数として渡せる、
   tokenId だけ device buffer から読む)。

### 実装ステップ

**A. Lean 側 forwardSingleTokenDeviceFed**

新しい entry point `forwardSingleTokenDeviceFed` を `Hesper/Models/Gemma4.lean`
に追加。`forwardSingleToken` のコピーから始め、以下を変更:

- 引数から `tokenId : Nat`, `pos : Nat` を削除
- `if !skipTokenWrite then writeScalarViaStaging ... tokenBuf` ブロック
  を削除 (常に skip)
- `kernelTokenId` 計算と plRawRowBuf 書込を削除 (kernel が
  state.tokenBuf を読む)
- pos 関連の writeScalarViaStaging を削除 (advancePos が devid)
- forward の末尾で `advancePosKernelGraphsOff` を呼ぶ
  (paramsBuf[0]++, posF32Buf++)

**B. Q6_K row dequant kernel に tokenBuf を渡す**

`Hesper.Quantization.Q6_K.q6kTableRowDequantScaleKernel` を変更:

- 現状: `params` (= plRawRowBuf, tokenId が書かれている) を読む
- 変更: そのまま使える (もう既に device-side で正しい)

→ 変更不要。kernel が plRawRowBuf を読む契約を維持し、Lean 側で
`writeScalarViaStaging` の代わりに **token id を device buffer から
コピーする 1 dispatch** を追加。または PLE 入口で
`copyU32Kernel(src=tokenBuf, dst=plRawRowBuf)` を発行する。

**C. UVA mmap path の row pointer 計算**

これが最難関。`bufFromRawDevicePtr` は **host から渡された ptr** を
ラップするので、kernel 起動時に row ptr が決まっている必要がある。

選択肢:
- (a) **UVA mmap path を諦める**: 該当パスを使わない設定で進める
  (HESPER_USE_MMAP=1 の代わりに full VRAM)。VRAM コスト +2 GB の代わりに
  decode の bubble を取れるなら割が合う可能性。
- (b) **row 全体を毎 forward でコピー**: kernel 内で table 全体を view
  して `tokenBuf` で row を選ぶ。Q6_K dequant kernel の input shape
  を変える。VRAM コストはそのまま。
- (c) **pointer table device-side**: vocab 個分の row pointer を
  device buffer に置き、kernel が `tokenBuf` で indirect index する。
  vocab=262144 × 8 byte = 2 MB の pointer table。

(c) が一番きれい。ただし PLE table を VRAM に持っていない場合
host-mapped UVA pointer の配列を作る必要があるので、初期化コードが
重くなる。

**まずは (a)** で進めて、bubble がきちんと閉じることを trace で
確認 → その後 (c) を実装するのが堅い。

**D. generate ループ側**

```lean
-- 旧
forwardSingleToken ctx model nextToken newPos state (kcr := some kcr)

-- 新 (graphs OFF, decode 経路)
forwardSingleTokenDeviceFed ctx model state (kcr := some kcr)
```

`nextToken` も `newPos` も渡さない。`tokens.push nextToken` は
**EOS check のためだけ** に必要なので、最初の数トークンだけ host で
拾い、その後は **K トークンまとめて** 取りに行く設計にする (例:
8 トークン毎に sync して push)。EOS が間に挟まったら break — 多めに
生成して後で trim する llama.cpp 流のパターンも可。

### 期待効果

trace の bubble 17 個 × 5ms = 85 ms / 1 sec ≈ 9% を回収できれば
TPS 61.7 → ~67 TPS。さらに sync 16 → 数回に減れば +10 TPS 程度
追加期待。トータル **graphs OFF で 80 TPS 級** が現実的なゴール。

### 検証

各ステップで `scripts/perf_compare.sh` + `scripts/nsys_to_chrome_trace.py`
を回し、

1. inter-kernel gap p99 (今 0.58 µs) を維持
2. >1ms gap が 17 個 → 0 に近づくか
3. GPU busy が 65.7% → 90%+ に上がるか

の 3 点を毎回確認。frozen な計測コマンドで出す。

## 落穴 (memo から)

- HESPER_PIPELINED_DECODE は landed 済み (`project_pipelined_decode_attempt.md`)
  だが TPS 41 vs 43 の noise レベル → 構造に踏み込まないとダメな証拠。
- DtoH を消しても drain は cuStreamSync に名前が変わるだけ
  (`feedback_dtoh_is_drain.md`). sync の **回数を減らす** ことが本質。
- bufferArray (`project_shaderm_buffer_array.md`) は Phase 2b で
  fusedQ4KMLinearDP4A4Warp のみ実装。残り kernel に拡張すれば
  per-token dispatch を 880 → ~50 程度に削減可能だが、それは
  別軸の最適化。

## 次のセッションでやること

1. forwardSingleTokenDeviceFed の skeleton (option (a) UVA 諦め版) を
   実装
2. advancePosKernelGraphsOff (paramsBuf[0]++, posF32Buf++) を作る
3. generate に新 entry point を繋ぎ込む (HESPER_DEVICE_FED=1 で opt-in)
4. 1 token decode → bit-parity を既存 path と確認
5. 60 token decode → trace で bubble 解消を確認
6. EOS check を K=8 token batched にする (correctness)
7. trace で >1ms gap が消えたら numbers 取って commit

## 結果 2026-04-26 (Step 1-5 まで実施)

**Step 1-3 landed**: HESPER_DEVICE_FED=1 toggle 追加。`forwardBlock`/`forwardSingleToken` に `skipPosWrite` フラグ追加。decode loop で 2 回目以降の forward は host writeScalarViaStaging を skip し、advancePosKernel で device-side に paramsBuf/posF32Buf を inc。

**Step 4 ✓ Bit-parity**: `Hello! How can I help you today? 😊` がbaseline と一致。

**Step 5 ✗ Bubble 解消ならず**:

| | baseline | DEVICE_FED |
|---|---:|---:|
| TPS (60 tok, ignoreEos) | 63.5 | 63.6 |
| decode steady-state GPU busy (deciles 2-10) | 65.7% | 66.9% |
| kernel time / decile-9 window | 585.0 ms | 589.0 ms |
| wall / decile-9 window | 869.2 ms | 880.9 ms |

つまり writeScalarViaStaging を消しても bubble は変わらない。

## なぜ Step 5 が動かなかったか

writeScalarViaStaging は pinned-async で 0.5 µs/call。42 layer × ~3 writes = 126 calls/tok × 0.5 µs = 63 µs/tok を消したに過ぎない。trace で観測された 5 ms/tok bubble は別物。

bubble の本体は `project_perf_2026_04_26.md` で既に測定済みの **5.6 µs/dispatch host floor × 880 dispatch = 4.93 ms/tok**。これは:
- forwardSingleToken / forwardBlock × 42 layer の Lean 関数呼び出しオーバーヘッド
- ce/withSection で毎 dispatch hash 計算 + cacheRef lookup
- match/let block のたびに Array.find?, lean_dec_ref, etc.

これらは graphs-OFF で **毎 token Lean が同じ traversal を再実行する** から発生する。argmax/pos の H2D を消しても Lean traversal は減らない。

## 本当に必要だったもの

doc 58 §の冒頭の認識は半分正しいが片手落ちだった:
- ✓ host-Nat 依存を消す (token / pos / cacheLen) — 副作用は小さい
- ✗ Lean traversal そのものを消す — これが本質

**Lean traversal を消す方法**:
1. **CUDA Graph capture** (現行 HESPER_CUDA_GRAPHS=1): forwardSingleToken
   を 1 回 capture すれば host work がほぼゼロに。実際 80-100 TPS 達成。
2. **bufferArray + 1-dispatch-many-layers** (Phase 2b 部分実装): 全 42
   layer 分を 1 つの kernel で処理。Lean が 42 回ループする必要なし。
3. **AOT Lean → C コード生成**: forwardSingleToken の中身を Lean elab 時に
   全部畳み込んで 1 つの IO action にする。

(1) が一番現実的だが「graphs OFF で勝つ」目標から外れる。(2) は完成
させれば graphs OFF でも勝てる可能性。(3) は重い。

## 結論

「graphs OFF で 80 TPS」 = bufferArray を全 dispatch に拡張するか、
forwardSingleToken の Lean 表現を flatten する必要がある。
host scalar write を消すだけでは届かない。

writeScalarViaStaging skip は**死荷重ではない** (TPS regression なし、
将来的な構造変更の前提条件として有用) ので landed のまま残す。
