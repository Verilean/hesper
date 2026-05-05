/-! # KernelStub — マクロ抽出のための **特徴的な行の集合**

設計方針 (2026-05-01 改訂):
- CUDA kernel の本格 parser は書かない
- 行ベースの grep 相当 — 特徴的なキーワード (template, __launch_bounds__,
  __shared__, kbx/kb0/mmq_x/nwarps を含む for-loop, __syncthreads, <<<>>>) を拾うだけ
- 出力は「kernel ごとの特徴行の bag」+ shape カウンタ
- 失敗時は空、絶対 panic しない

2026-05-01 v2 (Stage A+E): launch site と launch_bounds は構造化して
grid/block/smem 別 field に分解。pretty printer を「stub らしい」 layout に書き直し。
-/

namespace Hesper.StubExtract

/-- `__launch_bounds__(maxThreads, minBlocksPerSM)` を分解. -/
structure LaunchBounds where
  /-- 第1引数 (= max threads per block). 式のまま、評価しない. -/
  maxThreads : String
  /-- 第2引数 (= min blocks per SM). 省略時 none. -/
  minBlocksPerSM : Option String := none
  /-- 元の `__launch_bounds__(...)` 全体. -/
  raw : String
  deriving Repr, Inhabited

/-- ホスト側 `kernel<<<grid, block, smem, stream>>>` の launch site を分解. -/
structure LaunchSite where
  /-- grid 式 (`<<<...>>>` の 1 番目). 例: `block_nums`, `dim3(nb1, nb2, ne12*ne13)`. -/
  grid : String
  /-- block 式 (2 番目). 例: `dim3(WARP_SIZE, nwarps, 1)`, `256`. -/
  block : String
  /-- 動的 smem (3 番目). 省略時 none ("0" は some "0"). -/
  smem : Option String := none
  /-- stream (4 番目). -/
  stream : Option String := none
  /-- 元の `<<<...>>>` 中身 (整形前). -/
  raw : String
  deriving Repr, Inhabited

/-- Kernel の種別: `__global__` entry point か `__device__` helper か. -/
inductive KernelKind | «global» | device deriving Repr, Inhabited, BEq

/-- 1 つの kernel から拾った特徴的な行の集合. -/
structure KernelStub where
  /-- 関数名 (template 名 / specialization なし). -/
  name : String
  /-- `__global__` (entry) か `__device__` (helper) か. -/
  kind : KernelKind := .«global»
  /-- 関数定義のあるファイル. -/
  file : String := ""
  /-- 関数頭の行番号. -/
  headerLine : Nat := 0
  /-- `static __global__`, `__device__ __forceinline__` 等。生文字列のリスト. -/
  attrs : Array String := #[]
  /-- `template <...>` の中身 (生文字列、複数なら最も近いもの). -/
  templ : Option String := none
  /-- template params を `(name, type)` のリストに分解した結果。
      e.g. `ggml_type type, int mmq_x, bool need_check` →
           `[(type, ggml_type), (mmq_x, int), (need_check, bool)]`.
      `default` 値は型側に残す (`bool need_check = false` → type=`bool`, name=`need_check = false`). -/
  templParams : Array (String × String) := #[]
  /-- `__launch_bounds__(...)` の構造化結果. -/
  launchBounds : Option LaunchBounds := none
  /-- `__shared__` / `extern __shared__` の宣言行 (生文字列). -/
  smemDecls : Array String := #[]
  /-- 特徴的な for-loop (kbx/kb0/mmq_*/nwarps/warp_size を含む) の生文字列 + 役割タグ.
      タグは loop variable / step pattern から推定:
      - "K-outer" : `kbx0` / `kb0` / `kbx` を含む — K 軸 (matmul 内側次元) のタイル境界 outer
      - "K-inner" : `kqs` / `vdr` / `qi` を含む — quant block 内の K reduction
      - "X-coop"  : `i0` で `nwarps*warp_size` step — 行 (X tile) 協調ロード
      - "Y-coop"  : `j0` で `nwarps*warp_size` step — 列 (Y tile) 協調ロード
      - "warp"    : `w` ループ — warp 間 reduce
      - "row"     : `i` を `mmq_y` まで — accumulator iterate
      - "col"     : `j` を `mmq_x` まで — accumulator iterate
      - "?"       : 上のどれにも当たらない. -/
  charLoops : Array (String × String) := #[]
  /-- body 内の `__syncthreads()` の出現回数. -/
  syncCount : Nat := 0
  /-- ホスト側 launch site (構造化結果). -/
  launchSites : Array LaunchSite := #[]
  /-- body 内で呼び出している関数 (関数名のみ、count 付き). -/
  callees : Array (String × Nat) := #[]
  deriving Repr, Inhabited

/-! ## Pretty printer -/

/-- launch site を 1 行に整形 (詳細 dump 用). `grid: ... | block: ... | smem: ... | stream: ...`. -/
def LaunchSite.format (ls : LaunchSite) : String :=
  let smemStr := match ls.smem with | some s => s | none => "0"
  let streamStr := match ls.stream with | some s => s | none => "default"
  s!"grid={ls.grid}  block={ls.block}  smem={smemStr}  stream={streamStr}"

/-- launch bounds を 1 行に整形. -/
def LaunchBounds.format (lb : LaunchBounds) : String :=
  match lb.minBlocksPerSM with
  | some n => s!"max_threads={lb.maxThreads}, min_blocks_per_sm={n}"
  | none   => s!"max_threads={lb.maxThreads}"

/-- 1 つの stub を読みやすい multi-line text に整形 (stub-like layout). -/
def KernelStub.dump (s : KernelStub) : String := Id.run do
  let mut out := ""
  let kindTag := match s.kind with | .«global» => "" | .device => " [device]"
  out := out ++ s!"=== {s.name}{kindTag} ({s.file}:{s.headerLine}) ===\n"
  if s.attrs.size > 0 then
    out := out ++ s!"  attrs:    {String.intercalate " " s.attrs.toList}\n"
  match s.templ with
  | some t =>
    if s.templParams.isEmpty then
      out := out ++ s!"  template: <{t}>\n"
    else
      out := out ++ s!"  template ({s.templParams.size}):\n"
      for (n, ty) in s.templParams do
        out := out ++ s!"    {n} : {ty}\n"
  | none => pure ()
  match s.launchBounds with
  | some lb => out := out ++ s!"  occ:      {lb.format}\n"
  | none => pure ()
  if s.launchSites.size > 0 ∧ s.kind == .«global» then
    out := out ++ "  launches:\n"
    for ls in s.launchSites do
      out := out ++ s!"    {ls.format}\n"
  if s.smemDecls.size > 0 then
    out := out ++ s!"  smem ({s.smemDecls.size}):\n"
    for d in s.smemDecls do
      out := out ++ s!"    {d}\n"
  if s.charLoops.size > 0 then
    out := out ++ s!"  loops ({s.charLoops.size}):\n"
    for (tag, l) in s.charLoops do
      out := out ++ s!"    [{tag}] {l}\n"
  if s.syncCount > 0 then
    out := out ++ s!"  syncs:    {s.syncCount}×\n"
  if s.callees.size > 0 then
    out := out ++ s!"  callees ({s.callees.size}):\n"
    for (n, c) in s.callees do
      out := out ++ s!"    {n} ×{c}\n"
  out

/-- 1 行サマリ (markdown table の row 用). -/
def KernelStub.tableRow (s : KernelStub) : String :=
  let templStr := match s.templ with
    | some t => "`" ++ t ++ "`"
    | none => "—"
  let lbStr := match s.launchBounds with
    | some lb =>
      match lb.minBlocksPerSM with
      | some n => s!"`{lb.maxThreads}` / {n}"
      | none   => s!"`{lb.maxThreads}`"
    | none => "—"
  -- Block dim from the first launch site (most kernels have one canonical block shape).
  let blockStr := match s.launchSites[0]? with
    | some ls => "`" ++ ls.block ++ "`"
    | none => "—"
  let smemStr := if s.smemDecls.isEmpty then "—" else s!"{s.smemDecls.size} decl"
  let loopStr := if s.charLoops.isEmpty then "—" else s!"{s.charLoops.size} loop"
  let syncStr := if s.syncCount == 0 then "—" else s!"{s.syncCount}×"
  let siteStr := if s.launchSites.isEmpty then "—" else s!"{s.launchSites.size} site"
  s!"| `{s.name}` | `{s.file}` | {templStr} | {lbStr} | {blockStr} | {smemStr} | {loopStr} | {syncStr} | {siteStr} |"

/-- Kernel が "characteristic" — **何か** 拾えていれば true. これで pointwise 系の
    つまらない kernel (cpy, add 等) を report からふるい落とす.
    device helper は callee として参照される必要があるため別経路 (extractor 側) で
    filter — ここでは shape 持ち (smem/loop/sync) かどうかだけで判定. -/
def KernelStub.isInteresting (s : KernelStub) : Bool :=
  s.launchBounds.isSome ∨ !s.smemDecls.isEmpty ∨ !s.charLoops.isEmpty ∨ s.syncCount > 0

end Hesper.StubExtract
