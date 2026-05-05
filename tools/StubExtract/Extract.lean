import StubExtract.Stub
import Std.Data.HashMap

/-! # Stub extractor — 特徴的な行を grep して bag に詰めるだけ

設計:
- C++ parser は書かない
- 行ベースの substring match で「特徴的な」行を拾う
- kernel の境界は `{` / `}` の depth カウントだけで近似
- ファイル単位 / ディレクトリ単位の collect API を提供
-/

namespace Hesper.StubExtract

/-- 単純な substring 包含 (Lean 標準の List 比較で実装). -/
def hasSubstr (haystack needle : String) : Bool :=
  if needle.isEmpty then true
  else
    let nChars := needle.toList
    let nlen := nChars.length
    let rec go : List Char → Bool
      | [] => false
      | (c :: rest) =>
        let chars := c :: rest
        if chars.length < nlen then false
        else if chars.take nlen == nChars then true
        else go rest
    go haystack.toList

/-! ## 各特徴行を抜き出すヘルパー -/

/-- `template <...>` の `<...>` 中身 (最初の出現のみ). -/
private def grabTemplate (line : String) : Option String := Id.run do
  if !hasSubstr line "template <" then return none
  let parts := line.splitOn "template <"
  match parts with
  | _ :: rest :: _ =>
    let chars := rest.toList
    let mut depth : Int := 1
    let mut buf : List Char := []
    for c in chars do
      if c == '<' then depth := depth + 1; buf := c :: buf
      else if c == '>' then
        depth := depth - 1
        if depth == 0 then break
        buf := c :: buf
      else buf := c :: buf
    return some (String.ofList buf.reverse).trim
  | _ => return none

/-- depth-aware top-level comma split. `(a, b(c, d), e)` → `["a", "b(c, d)", "e"]`.
    `<...>` 内の `,` も depth として扱う (template instance を split しないため). -/
private def splitArgsTopLevel (s : String) : List String := Id.run do
  let mut depth : Int := 0
  let mut acc : List Char := []
  let mut out : List String := []
  for c in s.toList do
    if c == '(' ∨ c == '<' ∨ c == '[' then
      depth := depth + 1
      acc := c :: acc
    else if c == ')' ∨ c == '>' ∨ c == ']' then
      depth := depth - 1
      acc := c :: acc
    else if c == ',' ∧ depth == 0 then
      out := (String.ofList acc.reverse).trim :: out
      acc := []
    else
      acc := c :: acc
  out := (String.ofList acc.reverse).trim :: out
  return out.reverse

/-- `__launch_bounds__(maxThreads, minBlocksPerSM)` を構造化して取り出す. -/
private def grabLaunchBounds (line : String) : Option LaunchBounds := Id.run do
  if !hasSubstr line "__launch_bounds__" then return none
  let parts := line.splitOn "__launch_bounds__"
  match parts with
  | _ :: rest :: _ =>
    let chars := rest.toList
    let chars := chars.dropWhile (· != '(')
    match chars with
    | '(' :: rest =>
      let mut depth : Int := 1
      let mut buf : List Char := []
      for c in rest do
        if c == '(' then depth := depth + 1; buf := c :: buf
        else if c == ')' then
          depth := depth - 1
          if depth == 0 then break
          buf := c :: buf
        else
          buf := c :: buf
      let inner := String.ofList buf.reverse
      let args := splitArgsTopLevel inner
      let raw := s!"__launch_bounds__({inner})"
      match args with
      | [m]      => return some { maxThreads := m, minBlocksPerSM := none, raw }
      | [m, mn]  => return some { maxThreads := m, minBlocksPerSM := some mn, raw }
      | m :: mn :: _ => return some { maxThreads := m, minBlocksPerSM := some mn, raw }
      | []       => return none
    | _ => return none
  | _ => return none

/-- template の中身 (e.g. `ggml_type type, int mmq_x, bool need_check = false, typename T`) を
    `[(name, type)]` に分解。default 値は name に括る (e.g. `(need_check = false, bool)`). -/
def parseTemplParams (s : String) : Array (String × String) := Id.run do
  let parts := splitArgsTopLevel s
  let mut out : Array (String × String) := #[]
  for p in parts do
    let p := p.trim
    if p.isEmpty then continue
    -- separate default value (`= ...`) from declarator
    let (declStr, defaultStr) :=
      match p.splitOn " = " with
      | decl :: rest => (decl.trim, String.intercalate " = " rest)
      | [] => (p, "")
    -- declStr is now `T name` or `typename T` or just `bool` (positional template arg).
    let chars := declStr.toList
    let mut depth : Int := 0
    let mut splitIdx : Int := -1
    let mut idx : Nat := 0
    for c in chars do
      if c == '<' ∨ c == '(' ∨ c == '[' then depth := depth + 1
      else if c == '>' ∨ c == ')' ∨ c == ']' then depth := depth - 1
      else if depth == 0 ∧ (c == ' ' ∨ c == '\t') then
        splitIdx := Int.ofNat idx
      idx := idx + 1
    let (ty, nm) :=
      if splitIdx < 0 then
        (declStr, "")  -- single token like `bool`, `typename` etc → treat as type-only
      else
        let k := splitIdx.toNat
        let l := (String.ofList (chars.take k)).trim
        let r := (String.ofList (chars.drop (k + 1))).trim
        (l, r)
    -- re-attach default to name for visibility
    let nm :=
      if defaultStr.isEmpty then nm
      else if nm.isEmpty then s!"= {defaultStr}"
      else s!"{nm} = {defaultStr}"
    out := out.push (nm, ty)
  return out

/-- 「特徴的な」for-loop か? — kbx / kb0 / mmq_x / mmq_y / nwarps / warp_size を含むなら yes. -/
private def isCharLoop (line : String) : Bool :=
  let trimmed := line.trim
  if !(trimmed.startsWith "for (" ∨ trimmed.startsWith "for(") then false
  else
    hasSubstr line "kbx" ∨ hasSubstr line "kb0" ∨
    hasSubstr line "mmq_x" ∨ hasSubstr line "mmq_y" ∨
    hasSubstr line "nwarps" ∨ hasSubstr line "warp_size"

/-- `__shared__` / `extern __shared__` 宣言行か? -/
private def isSmemDecl (line : String) : Bool :=
  hasSubstr line "__shared__"

/-- 直前数行から `#pragma unroll [N]` / `// #pragma unroll` を検出して
    その for-loop の unroll 戦略タグを返す.
    - `#pragma unroll`        → "unroll"
    - `#pragma unroll N`      → "unroll(N)"  (N==1 は "no-unroll" として読み替え)
    - `// #pragma unroll`     → "no-unroll" (作者が意図的に runtime に残した)
    - 何もなし                 → "" (タグ無し). -/
def detectUnrollPragma (lines : Array String) (forIdx : Nat) : String := Id.run do
  -- 直前 3 行までスキャン (ブランクと中括弧は飛ばす)
  let mut k : Int := Int.ofNat forIdx - 1
  let mut budget : Nat := 3
  while k ≥ 0 ∧ budget > 0 do
    let line := lines[k.toNat]!.trim
    if line.isEmpty ∨ line == "{" then
      k := k - 1
      continue
    if line.startsWith "//" ∧ hasSubstr line "#pragma unroll" then
      return "no-unroll"
    if line.startsWith "#pragma unroll" then
      let rest := (line.drop 14).trim
      if rest.isEmpty then return "unroll"
      if rest == "1" then return "no-unroll"
      return s!"unroll({rest})"
    -- それ以外の non-blank 行に到達 → pragma 無し
    return ""
  return ""

/-- 特徴的 for-loop に役割タグを付ける (heuristic). 行内の loop var と step から推定. -/
def classifyLoop (line : String) : String :=
  -- K-outer: kbx, kb0 (super-block index across K)
  if hasSubstr line "kbx0" ∨ hasSubstr line "kb0" ∨ hasSubstr line "kbx" then "K-outer"
  -- K-inner: vdr / qi / kqs (within-block K reduction)
  else if hasSubstr line " vdr" ∨ hasSubstr line " qi" ∨ hasSubstr line "kqs" then "K-inner"
  -- coop loads with i0 + nwarps*warp_size step → X tile
  else if hasSubstr line "i0 = 0" ∧ (hasSubstr line "nwarps*warp_size" ∨ hasSubstr line "nwarps * warp_size") then "X-coop"
  -- coop loads with j0 + nwarps*warp_size step → Y tile
  else if hasSubstr line "j0 = 0" ∧ (hasSubstr line "nwarps*warp_size" ∨ hasSubstr line "nwarps * warp_size") then "Y-coop"
  -- warp aggregation
  else if hasSubstr line "for (int w = 0" ∨ hasSubstr line "w < nwarps" then "warp"
  -- row / col accumulator iteration
  else if hasSubstr line "i < mmq_y" then "row"
  else if hasSubstr line "j < mmq_x" then "col"
  -- D-axis (head dim) loops in flash_attn
  else if hasSubstr line "DKQ" ∨ hasSubstr line "DVp" ∨ hasSubstr line "DV" then "D-axis"
  -- ncols loops
  else if hasSubstr line "j0 < ncols" then "ncols"
  -- generic l0 with nwarps*warp_size step (mid-level coop loads).
  else if hasSubstr line "l0 = 0" ∧ (hasSubstr line "nwarps * warp_size" ∨ hasSubstr line "nwarps*warp_size") then "tile-coop"
  else "?"

/-- これらは callee として拾わない (control flow / cuda intrinsic / 演算 / 型キャスト系 / pp). -/
private def calleeBlacklist : List String :=
  ["if", "for", "while", "switch", "return", "do", "else", "case",
   "sizeof", "static_cast", "reinterpret_cast", "const_cast", "dynamic_cast",
   "defined",  -- preprocessor `#if defined(X)`
   "__syncthreads", "__syncwarp", "__threadfence", "__threadfence_block",
   "__shfl", "__shfl_sync", "__shfl_xor_sync", "__shfl_up_sync", "__shfl_down_sync",
   "__ballot_sync", "__any_sync", "__all_sync", "__activemask",
   "__half2float", "__float2half", "__half22float2", "__float22half2_rn",
   "__hadd2", "__hmul2", "__hfma2", "__hadd", "__hmul", "__hfma", "__hsub", "__hsub2",
   "__low2half", "__high2half", "__low2float", "__high2float",
   "__dp4a", "__vsubss4", "__vsub4", "__vadd4", "__byte_perm", "__funnelshift_l", "__funnelshift_r",
   "__expf", "__logf", "__fmul_rn", "__fadd_rn", "__fmaf_rn",
   "min", "max", "abs", "fabsf", "sqrtf", "rsqrtf", "expf", "logf", "powf", "sinf", "cosf",
   "make_uint2", "make_uint4", "make_int2", "make_int4", "make_half2", "make_float2", "make_float4",
   "asm", "T", "type", "U", "Tdst", "Tin", "Tout",
   "GGML_ASSERT", "GGML_UNUSED", "GGML_UNUSED_VARS",
   "NO_DEVICE_CODE", "NEW_MMA_AVAILABLE",
   "static_assert", "constexpr",
   "make_uint3", "fastmodulo", "fastdiv",
   -- C-style casts that look like calls
   "int", "float", "double", "char", "short", "long",
   "uint8_t", "uint16_t", "uint32_t", "uint64_t",
   "int8_t", "int16_t", "int32_t", "int64_t",
   "size_t", "ptrdiff_t", "half", "half2", "bfloat16",
   "__builtin_assume", "__align__", "__restrict__",
   "tanhf", "fmaxf", "fminf", "floorf", "ceilf"]

/-- 行を走査して `IDENT(` または `IDENT<...>(` を検出し、IDENT を return.
    blacklist のものは除外する. 1 行で複数あれば全部. -/
private def grabCallees (line : String) : Array String := Id.run do
  let mut out : Array String := #[]
  let chars := line.toList
  let n := chars.length
  let arr := chars.toArray
  let mut i : Nat := 0
  while i < n do
    let c := arr[i]!
    -- skip string / char literals (very rough)
    if c == '"' ∨ c == '\'' then
      let q := c
      i := i + 1
      while i < n ∧ arr[i]! != q do
        if arr[i]! == '\\' then i := i + 1
        i := i + 1
      i := i + 1
      continue
    -- skip line comment
    if c == '/' ∧ i + 1 < n ∧ arr[i+1]! == '/' then
      break
    -- identifier?
    if c.isAlpha ∨ c == '_' then
      let mut j := i
      while j < n ∧ (arr[j]!.isAlphanum ∨ arr[j]! == '_') do
        j := j + 1
      let ident := String.ofList (arr.toList.drop i |>.take (j - i))
      -- after the ident, skip optional `<...>` template instance, then check for `(`
      let mut k := j
      if k < n ∧ arr[k]! == '<' then
        let mut depth : Int := 1
        k := k + 1
        while k < n ∧ depth > 0 do
          let ck := arr[k]!
          if ck == '<' then depth := depth + 1
          else if ck == '>' then depth := depth - 1
          else if ck == ';' ∨ ck == '\n' then
            depth := -1   -- bail; this wasn't a template after all
          k := k + 1
        if depth != 0 then
          i := j
          continue
      -- skip whitespace between `>` (or ident end) and `(`
      while k < n ∧ (arr[k]! == ' ' ∨ arr[k]! == '\t') do
        k := k + 1
      if k < n ∧ arr[k]! == '(' then
        -- It's a call. Filter blacklist + leading underscore-only tokens we want.
        if !(calleeBlacklist.contains ident) ∧ ident.length > 1 ∧ !ident.all (fun c => c.isDigit) then
          if !out.contains ident then out := out.push ident
      i := j
      continue
    i := i + 1
  return out

/-- `dim3(a, b, c)` のような wrap を剥がす — top-level が `dim3(...)` ぴったりなら中身を返す.
    そうでなければそのまま. -/
private def stripDim3 (s : String) : String :=
  let t := s.trim
  if t.startsWith "dim3(" ∧ t.endsWith ")" then
    let inner := (t.drop 5).dropRight 1
    inner.trim
  else t

/-- `<<<grid, block, smem, stream>>>` の中身を `LaunchSite` に分解 (最初の出現のみ).
    smem/stream 省略時は none. depth-aware split で `dim3(a, b, c)` の comma を保護. -/
private def grabLaunchSite (line : String) : Option LaunchSite :=
  if !hasSubstr line "<<<" then none
  else
    let p1 := line.splitOn "<<<"
    match p1 with
    | _ :: rest :: _ =>
      match rest.splitOn ">>>" with
      | cfg :: _ =>
        let raw := cfg.trim
        let args := splitArgsTopLevel raw
        match args with
        | g :: b :: s :: t :: _ =>
          some { grid := g, block := b, smem := some s, stream := some t, raw }
        | [g, b, s] =>
          some { grid := g, block := b, smem := some s, stream := none, raw }
        | [g, b] =>
          some { grid := g, block := b, smem := none, stream := none, raw }
        | _ => none
      | _ => none
    | _ => none

/-- `__global__ void NAME(` の NAME を取り出す (1 行に複数あっても 1 つだけ).
    `__global__ void __launch_bounds__(...)` のように attribute が挟まる場合は
    paren-skip して次の identifier を取る (NAME は `__launch_bounds__` ではない).
    NAME が見つからない (= 次行に持ち越し) 場合は none を返し、caller が
    multi-line で対処する想定. -/
private def grabKernelHeaderName (line : String) : Option String := Id.run do
  if !hasSubstr line "__global__ void " then return none
  let p := line.splitOn "__global__ void "
  match p with
  | _ :: rest :: _ =>
    let mut chars := rest.toList.dropWhile (fun c => c == ' ')
    -- Skip optional `__launch_bounds__(...)` attribute (or any other __X__(...)).
    while !chars.isEmpty ∧ (chars.head!.isAlphanum ∨ chars.head! == '_') do
      let cand := String.ofList (chars.takeWhile (fun c => c.isAlphanum ∨ c == '_'))
      if cand.startsWith "__" ∧ cand.endsWith "__" then
        -- attribute-like keyword: skip its name + the following parens
        chars := chars.dropWhile (fun c => c.isAlphanum ∨ c == '_')
        chars := chars.dropWhile (· == ' ')
        if chars.isEmpty ∨ chars.head! != '(' then break
        let mut depth : Int := 1
        chars := chars.tail!
        while !chars.isEmpty ∧ depth > 0 do
          let c := chars.head!
          if c == '(' then depth := depth + 1
          else if c == ')' then depth := depth - 1
          chars := chars.tail!
        chars := chars.dropWhile (· == ' ')
        continue
      -- This is the actual function name.
      return some cand
    return none
  | _ => return none

/-- `static __global__`, `__forceinline__` 等のキーワードを行から拾う. -/
private def grabAttrs (line : String) : Array String := Id.run do
  let mut out : Array String := #[]
  for kw in ["__device__", "__global__", "__forceinline__", "__host__", "static", "inline", "extern"] do
    if hasSubstr line kw then out := out.push kw
  return out

/-- `__device__` 関数の header 行から関数名を抜き出す.
    パターン: `... __device__ [__forceinline__|inline] [void|RetType] NAME(...)`.
    NAME が同じ行にない場合 (next-line) は none を返し、caller が multi-line で対処する想定. -/
private def grabDeviceHeaderName (line : String) : Option String := Id.run do
  -- Must contain `__device__` and `(` and not be `__global__` (those are handled separately).
  if !hasSubstr line "__device__" then return none
  if hasSubstr line "__global__" then return none
  -- Skip `extern "C"` style and macro-only lines.
  if hasSubstr line "#define" then return none
  -- Find `__device__` keyword end, then walk past attribute words to the function name.
  let parts := line.splitOn "__device__"
  match parts with
  | _ :: rest :: _ =>
    let mut chars := rest.toList.dropWhile (fun c => c == ' ')
    -- skip optional `__forceinline__`, `inline`, return type tokens.
    -- We do this generically: take ident chars, peek what follows.
    -- We want the LAST identifier before `(`.
    -- Walk forward collecting identifiers; the one immediately before `(` is the name.
    let mut lastIdent : String := ""
    while !chars.isEmpty do
      let c := chars.head!
      if c.isAlpha ∨ c == '_' then
        let ident := String.ofList (chars.takeWhile (fun c => c.isAlphanum ∨ c == '_'))
        chars := chars.dropWhile (fun c => c.isAlphanum ∨ c == '_')
        chars := chars.dropWhile (· == ' ')
        -- Skip attribute-like tokens (`__X__`).
        if ident.startsWith "__" ∧ ident.endsWith "__" ∧ !chars.isEmpty ∧ chars.head! == '(' then
          -- attribute with parens: skip its args
          let mut depth : Int := 1
          chars := chars.tail!
          while !chars.isEmpty ∧ depth > 0 do
            let ck := chars.head!
            if ck == '(' then depth := depth + 1
            else if ck == ')' then depth := depth - 1
            chars := chars.tail!
          chars := chars.dropWhile (· == ' ')
          continue
        lastIdent := ident
        -- If next char is `(`, this ident is the function name.
        if !chars.isEmpty ∧ chars.head! == '(' then
          return some lastIdent
      else if c == '*' ∨ c == '&' ∨ c == ',' then
        chars := chars.tail!
        chars := chars.dropWhile (· == ' ')
      else if c == '<' then
        -- Skip template instantiation `Foo<...>`
        let mut depth : Int := 1
        chars := chars.tail!
        while !chars.isEmpty ∧ depth > 0 do
          let ck := chars.head!
          if ck == '<' then depth := depth + 1
          else if ck == '>' then depth := depth - 1
          chars := chars.tail!
        chars := chars.dropWhile (· == ' ')
      else
        break
    return none
  | _ => return none

/-! ## ファイル単位の抽出 -/

/-- ファイル全体を読んで各 kernel 関数の stub を作る.
    kernel 境界は `{` `}` の depth で判定する。1 関数の body 範囲内で出会った
    特徴行を `KernelStub` フィールドに集める。 -/
def extractFromFile (path : String) : IO (Array KernelStub) := do
  let src ← IO.FS.readFile path
  let lines := (src.splitOn "\n").toArray

  -- Pass 1: 各 kernel/device の header 行と body 範囲 [headerIdx, endIdx) を求める.
  -- name が同じ行に出ない場合 (e.g. `__global__ void __launch_bounds__(...)` の後に
  -- 改行して関数名がくる) は、次行を joining して再試行する.
  let mut headers : Array (Nat × String × KernelKind) := #[]
  for h : i in [0:lines.size] do
    have : i < lines.size := h.upper
    let l := lines[i]
    if hasSubstr l "__global__ void " then
      match grabKernelHeaderName l with
      | some name => headers := headers.push (i, name, .«global»)
      | none =>
        -- multi-line: try joining with up to 3 following lines
        let mut joined := l
        let mut k := i + 1
        let mut found : Option String := none
        for _ in [0:3] do
          if k ≥ lines.size then break
          joined := joined ++ " " ++ lines[k]!.trim
          match grabKernelHeaderName joined with
          | some name => found := some name; break
          | none => k := k + 1
        match found with
        | some name => headers := headers.push (i, name, .«global»)
        | none => pure ()
    else if hasSubstr l "__device__" ∧ !hasSubstr l "__global__" then
      -- device function header. require parens for function-like form (skip variable decls).
      if hasSubstr l "(" ∨ (i + 1 < lines.size ∧ hasSubstr lines[i+1]! "(") then
        match grabDeviceHeaderName l with
        | some name => headers := headers.push (i, name, .device)
        | none =>
          let mut joined := l
          let mut k := i + 1
          let mut found : Option String := none
          for _ in [0:3] do
            if k ≥ lines.size then break
            joined := joined ++ " " ++ lines[k]!.trim
            match grabDeviceHeaderName joined with
            | some name => found := some name; break
            | none => k := k + 1
          match found with
          | some name => headers := headers.push (i, name, .device)
          | none => pure ()

  let mut stubs : Array KernelStub := #[]
  for (hIdx, name, kind) in headers do
    -- 前 30 行を遡って template / launch_bounds / attrs を拾う.
    let lookbackStart : Nat := if hIdx > 30 then hIdx - 30 else 0
    let mut templ : Option String := none
    let mut lb : Option LaunchBounds := none
    let mut attrs : Array String := #[]
    for j in [lookbackStart:hIdx + 1] do
      let l := lines[j]!
      -- attrs from any of these lines
      for a in grabAttrs l do
        if !attrs.contains a then attrs := attrs.push a
      match grabTemplate l with
      | some t => templ := some t
      | none => pure ()
      match grabLaunchBounds l with
      | some s => lb := some s
      | none => pure ()

    -- body 範囲: hIdx から次の '}' で depth が 0 に戻るまで
    let mut depth : Int := 0
    let mut started : Bool := false
    let mut endIdx : Nat := hIdx
    for j in [hIdx:lines.size] do
      let l := lines[j]!
      for c in l.toList do
        if c == '{' then depth := depth + 1; started := true
        else if c == '}' then depth := depth - 1
      if started ∧ depth ≤ 0 then
        endIdx := j
        break

    -- body 内の特徴行を bag に詰める
    let mut smem : Array String := #[]
    let mut loops : Array (String × String) := #[]
    let mut syncCount : Nat := 0
    let mut calleeCounts : Std.HashMap String Nat := {}
    let mut calleeOrder : Array String := #[]
    -- 自分自身は callee に含めない (recursion がある場合の noise を抑える)
    let selfName := name
    for j in [hIdx:endIdx + 1] do
      let l := lines[j]!
      let lt := l.trim
      if isSmemDecl lt then smem := smem.push lt
      if isCharLoop lt then
        let baseTag := classifyLoop lt
        let unrollTag := detectUnrollPragma lines j
        let combined := if unrollTag.isEmpty then baseTag else s!"{baseTag}, {unrollTag}"
        loops := loops.push (combined, lt)
      if hasSubstr lt "__syncthreads" then syncCount := syncCount + 1
      -- callee scan: optionally join with next line so `Foo<T>\n(...)` pattern is caught.
      let scanLine :=
        if j + 1 < lines.size ∧ lt.endsWith ">" ∧
           (lines[j+1]!.trim.startsWith "(" ∨ lines[j+1]!.trim.startsWith "        (") then
          lt ++ " " ++ lines[j+1]!.trim
        else lt
      for callee in grabCallees scanLine do
        if callee == selfName then continue
        let cnt := calleeCounts.getD callee 0
        if cnt == 0 then calleeOrder := calleeOrder.push callee
        calleeCounts := calleeCounts.insert callee (cnt + 1)
    let calleesArr := calleeOrder.map (fun n => (n, calleeCounts.getD n 0))

    let templParams := match templ with
      | some t => parseTemplParams t
      | none => #[]
    stubs := stubs.push {
      name, kind, file := path, headerLine := hIdx + 1,
      attrs, templ, templParams, launchBounds := lb,
      smemDecls := smem, charLoops := loops, syncCount,
      launchSites := #[],  -- filled below
      callees := calleesArr
    }
  return stubs

/-- 文字列が「単純な識別子」(英数字+`_`) かどうか. これが true のものだけ
    変数名として lookback で解決を試みる (`dim3(...)` や `WARP_SIZE` は対象外). -/
private def isSimpleIdent (s : String) : Bool :=
  let t := s.trim
  if t.isEmpty then false
  else t.toList.all (fun c => c.isAlphanum ∨ c == '_')

/-- 行から `dim3 NAME(args)` の args を抜き出す. -/
private def grabDim3Decl (line : String) (name : String) : Option String := Id.run do
  -- patterns: `dim3 name(...)`, `const dim3 name(...)`, `dim3 name = dim3(...);`
  let needles := #[s!"dim3 {name}(", s!"dim3 {name} ("]
  for needle in needles do
    if hasSubstr line needle then
      let parts := line.splitOn needle
      match parts with
      | _ :: rest :: _ =>
        let chars := rest.toList
        -- already past the `(`, so rest is the args
        let mut depth : Int := 1
        let mut buf : List Char := []
        for c in chars do
          if c == '(' then depth := depth + 1; buf := c :: buf
          else if c == ')' then
            depth := depth - 1
            if depth == 0 then break
            buf := c :: buf
          else
            buf := c :: buf
        return some s!"dim3({(String.ofList buf.reverse).trim})"
      | _ => pure ()
  -- pattern: `dim3 name = dim3(...)`
  if hasSubstr line s!"dim3 {name} =" then
    let parts := line.splitOn "= "
    match parts with
    | _ :: rhs :: _ =>
      let t := rhs.trim
      let t := if t.endsWith ";" then t.dropRight 1 else t
      return some t.trim
    | _ => pure ()
  return none

/-- 行から `int NAME = ...;`, `size_t NAME = ...;`, `const T NAME = ...;` の RHS を抜く.
    式は `;` か行末まで. -/
private def grabAssignRhs (line : String) (name : String) : Option String := Id.run do
  -- crude: find ` NAME = ` and take rest until `;`
  let needle := s!" {name} = "
  if !hasSubstr line needle then return none
  let parts := line.splitOn needle
  match parts with
  | _ :: rhs :: _ =>
    let t := rhs.trim
    let t := if t.contains ';' then
              (t.splitOn ";").head!.trim
            else t
    return some t
  | _ => return none

/-- launch 行より上の lookback 範囲で変数 `name` を解決. dim3 → assign の順で探す.
    解決できなければそのまま `name` を返す. 探索範囲は 200 行 (host helper の長さ余裕). -/
private def resolveVar (lines : Array String) (curIdx : Nat) (name : String) : String := Id.run do
  if !isSimpleIdent name then return name
  let lookbackStart : Nat := if curIdx > 200 then curIdx - 200 else 0
  -- iterate backward from curIdx-1 to lookbackStart
  let mut i : Nat := curIdx
  while i > lookbackStart do
    i := i - 1
    let l := lines[i]!
    match grabDim3Decl l name with
    | some r => return r
    | none =>
      match grabAssignRhs l name with
      | some r => return r
      | none => pure ()
  return name

/-- ファイル全体から kernel alias 代入を集める.
    パターン:
      `... fattn_kernel = flash_attn_ext_vec<...>;`
      `fattn_kernel_t fattn_kernel = flash_attn_tile<...>;`
    返値: alias name → actual kernel names のマップ (同じ alias 名が複数 kernel に再使用される
    ファイルもあるため Array). -/
def collectKernelAliases (path : String) (kernelNames : Array String) : IO (Std.HashMap String (Array String)) := do
  let src ← IO.FS.readFile path
  let mut acc : Std.HashMap String (Array String) := {}
  for line in src.splitOn "\n" do
    for n in kernelNames do
      let needles := #[s!"= {n}<", s!"= {n}("]
      for needle in needles do
        if hasSubstr line needle then
          let lhsRest := (line.splitOn "=").head!.trim
          let toks := lhsRest.splitOn " "
          let alias := (toks.getLast!).trim
          if isSimpleIdent alias ∧ alias != n then
            let cur := acc.getD alias #[]
            if !cur.contains n then
              acc := acc.insert alias (cur.push n)
          break
  return acc

/-- ファイル全体で `kernelName<<<...>>>` または `kernelName<...><<<...>>>` の
    launch site を集める (template instance も含めて 1 つの kernel に集約).
    各 site を `LaunchSite` として返す. block/grid/smem の引数が **単純識別子** なら
    lookback で `dim3 NAME(...)` や `const T NAME = ...;` を探して解決する.
    `aliasMap` で alias 名 → 実 kernel 名のリストを解決する.
    alias が複数 kernel に map される場合 (e.g. `fattn_kernel` → vec/tile/mma),
    helper file の launch site は全候補に attribute される (どれでも同じ shape を共有). -/
def collectLaunchSites (path : String) (kernelNames : Array String)
    (aliasMap : Std.HashMap String (Array String) := {}) :
    IO (Std.HashMap String (Array LaunchSite)) := do
  let src ← IO.FS.readFile path
  let lines := (src.splitOn "\n").toArray
  let mut acc : Std.HashMap String (Array LaunchSite) := {}
  for h : i in [0:lines.size] do
    have : i < lines.size := h.upper
    let l := lines[i]
    match grabLaunchSite l with
    | none => pure ()
    | some site0 =>
      -- resolve grid/block/smem if they are bare identifiers
      let gridR  := stripDim3 (resolveVar lines i site0.grid)
      let blockR := stripDim3 (resolveVar lines i site0.block)
      let smemR  := site0.smem.map (fun s => resolveVar lines i s)
      let site := { site0 with grid := gridR, block := blockR, smem := smemR }
      -- 直接の kernel 名でマッチ?
      let mut matchedNames : Array String := #[]
      for n in kernelNames do
        if hasSubstr l (n ++ "<<<") ∨ hasSubstr l (n ++ "<") then
          matchedNames := matchedNames.push n
          break
      -- マッチしなかった: alias 越しに解決. multi-target alias なら全候補に追加する
      -- (helper の同じ launch line が全 alias 候補に共通使用される).
      if matchedNames.isEmpty then
        match l.splitOn "<<<" with
        | pre :: _ =>
          let chars := pre.toList.reverse
          let chars := chars.dropWhile (fun c => c == ' ' ∨ c == '\t')
          let identRev := chars.takeWhile (fun c => c.isAlphanum ∨ c == '_')
          let ident := String.ofList identRev.reverse
          if !ident.isEmpty then
            match aliasMap.get? ident with
            | some actuals => matchedNames := actuals
            | none => pure ()
        | [] => pure ()
      for n in matchedNames do
        let cur := acc.getD n #[]
        acc := acc.insert n (cur.push site)
  return acc

/-- 1 ファイル full pipeline: stubs + launch sites を結合 (alias なし). -/
def extractFileWithSites (path : String) : IO (Array KernelStub) := do
  let stubs ← extractFromFile path
  let names := stubs.map (·.name)
  let sites ← collectLaunchSites path names
  return stubs.map (fun s => { s with launchSites := sites.getD s.name #[] })

/-! ## ディレクトリ単位 -/

/-- `.cu` / `.cuh` ファイルを再帰的に集める. -/
partial def walkSourceDir (dir : String) : IO (Array String) := do
  let entries ← System.FilePath.readDir dir
  let mut out : Array String := #[]
  for e in entries do
    let p := e.path.toString
    if (← e.path.isDir) then
      out := out ++ (← walkSourceDir p)
    else if p.endsWith ".cu" ∨ p.endsWith ".cuh" then
      out := out.push p
  return out

/-- ディレクトリ配下の全 kernel stub を抽出 (interesting only).
    2-pass:
    - Pass 1: 全 file から stub (without launch sites) + alias map を集める
    - Pass 2: 全 file から launch site を集め、alias map で template-helper 経由 launch も解決. -/
def extractAllInDir (dir : String) (interestingOnly : Bool := true) : IO (Array KernelStub) := do
  let files ← walkSourceDir dir
  let mut allKernelNames : Array String := #[]
  -- First pass: stubs only
  let mut perFileStubs : Array (String × Array KernelStub) := #[]
  for f in files do
    let stubs ← extractFromFile f
    perFileStubs := perFileStubs.push (f, stubs)
    for s in stubs do
      if !allKernelNames.contains s.name then
        allKernelNames := allKernelNames.push s.name
  -- Pass 2: alias map union — alias 名 → {actual kernel names}.
  -- 同じ alias 名 (e.g. `fattn_kernel`) が複数 kernel に再使用される場合、
  -- helper file の launch site は全候補に attribute される (どれも同じ shape を共有).
  let mut sitesByKernel : Std.HashMap String (Array LaunchSite) := {}
  let mut aliasUnion : Std.HashMap String (Array String) := {}
  for f in files do
    let aliases ← collectKernelAliases f allKernelNames
    for (k, vs) in aliases.toList do
      let cur := aliasUnion.getD k #[]
      let merged := vs.foldl (fun a v => if a.contains v then a else a.push v) cur
      aliasUnion := aliasUnion.insert k merged
  for f in files do
    let sites ← collectLaunchSites f allKernelNames aliasUnion
    for (k, v) in sites.toList do
      let cur := sitesByKernel.getD k #[]
      sitesByKernel := sitesByKernel.insert k (cur ++ v)
  -- Combine: attach launch sites to each stub
  let mut acc : Array KernelStub := #[]
  -- Collect names of `__device__` functions reachable from a `__global__` call tree.
  -- Without this filter the report fills with thousands of irrelevant helpers
  -- (raw decode primitives, type-check helpers, etc.).
  let mut reachable : Std.HashMap String Unit := {}
  let mut frontier : Array String := #[]
  -- Seed with callees of every interesting `__global__` kernel.
  for (_, stubs) in perFileStubs do
    for s in stubs do
      if s.kind == .«global» ∧ (!interestingOnly ∨ s.isInteresting) then
        for (callee, _) in s.callees do
          if !reachable.contains callee then
            reachable := reachable.insert callee ()
            frontier := frontier.push callee
  -- BFS expansion through device-function call graph.
  while !frontier.isEmpty do
    let mut nextFrontier : Array String := #[]
    for n in frontier do
      for (_, stubs) in perFileStubs do
        for s in stubs do
          if s.name == n ∧ s.kind == .device then
            for (callee, _) in s.callees do
              if !reachable.contains callee then
                reachable := reachable.insert callee ()
                nextFrontier := nextFrontier.push callee
    frontier := nextFrontier
  -- Emit. global: filtered by isInteresting (existing). device: must be reachable
  -- AND have shape content (smem / loops / sync) — leaf compute helpers without
  -- shape are filtered out via isInteresting.
  for (_, stubs) in perFileStubs do
    for s in stubs do
      let s' := { s with launchSites := sitesByKernel.getD s.name #[] }
      let include' : Bool := match s.kind with
        | .«global» => !interestingOnly ∨ s'.isInteresting
        | .device   => reachable.contains s.name ∧ s'.isInteresting
      if include' then
        acc := acc.push s'
  return acc

end Hesper.StubExtract
