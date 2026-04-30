import Hesper.Transpile.CUDA.Lex
import Hesper.Transpile.CUDA.AST

/-! # CUDA → ShaderM transpiler — recursive-descent parser

Parses the token stream from `Lex` into `AST.CExpr`. Phase 1 covers
expressions; statements/declarations come later.

Operator precedence (high → low):
  postfix  a[i]  a.b  a->b  a()  a++ a--
  unary    -a !a ~a ++a --a *a &a  (T)a
  *  /  %
  +  -
  <<  >>
  <  <=  >  >=
  ==  !=
  &
  ^
  |
  &&
  ||
  ?:                  right-assoc
  =  += -= ...        right-assoc
  ,
-/
namespace Hesper.Transpile.CUDA

/-- Parser state: token array + cursor. -/
structure ParseState where
  toks : Array Spanned
  i    : Nat := 0

abbrev ParseM := ExceptT String (StateM ParseState)

@[inline] def peek : ParseM Tok := do
  let s ← get
  if h : s.i < s.toks.size then pure s.toks[s.i].tok else pure .eof

@[inline] def bump : ParseM Unit := modify fun s => { s with i := s.i + 1 }

@[inline] def consume : ParseM Tok := do
  let t ← peek; bump; pure t

def expectPunct (p : String) : ParseM Unit := do
  match (← peek) with
  | .punct s =>
    if s == p then bump
    else throw s!"expected punct '{p}', got '{s}'"
  | t => throw s!"expected punct '{p}', got {repr t}"

def matchPunct (p : String) : ParseM Bool := do
  match (← peek) with
  | .punct s => if s == p then bump; pure true else pure false
  | _ => pure false

/-- Heuristic: this identifier starts a cast type. -/
def isLikelyTypeStart : String → Bool
  | "int" | "uint" | "uint8_t" | "uint16_t" | "uint32_t" | "uint64_t"
  | "int8_t" | "int16_t" | "int32_t" | "int64_t"
  | "char" | "short" | "long" | "unsigned" | "signed"
  | "float" | "double" | "void" | "bool"
  | "half" | "half2" | "float2" | "float4"
  | "const" | "volatile" => true
  | _ => false

mutual

partial def parsePrimary : ParseM CExpr := do
  match (← peek) with
  | .num s => bump; pure (CExpr.numLit s)
  | .fnum s => bump; pure (CExpr.floatLit s)
  | .str  s =>
    -- Treat string literal as an opaque numLit (we don't model strings;
    -- the kernel paths we care about don't operate on them — they only
    -- appear in error/log messages we drop).
    bump; pure (CExpr.numLit s!"\"{s}\"")
  | .ident name =>
    bump
    -- C++ qualified name `A::B::C` — fold the chain into a single ident
    -- so downstream lookups can match by full name (or strip the
    -- prefix in the env). Stop before any `<` template head or `(`.
    let mut name := name
    while (← matchPunct "::") do
      match (← consume) with
      | .ident s => name := name ++ "::" ++ s
      | t => throw s!"parsePrimary: '::' must be followed by ident, got {repr t}"
    -- Check for `name<TArgs>(...)` — templated call.  Heuristic: peek
    -- a `<` followed eventually by a matching `>` and then `(`. If we
    -- find that pattern within a small lookahead, consume the
    -- `<...>` (we drop the template args here — template values are
    -- expected to be folded by the user's `Env.consts` registration
    -- or `Env.inlines` rewrite key by name).
    let isTemplCall ← do
      match (← peek) with
      | .punct "<" =>
        let st0 ← get
        let savedI := st0.i
        -- Forward scan: count nested `<>` and find a `>` that lands
        -- right before `(`.  Cap the scan at 32 tokens.
        let mut depth : Int := 0
        let mut j := savedI
        let cap := savedI + 32
        let mut found := false
        let toks := st0.toks
        while !found ∧ j < cap ∧ j < toks.size do
          match toks[j]!.tok with
          | .punct "<" => depth := depth + 1; j := j + 1
          | .punct ">" =>
            depth := depth - 1
            if depth == 0 ∧ j + 1 < toks.size
               ∧ (toks[j+1]!.tok == .punct "("
                  ∨ toks[j+1]!.tok == .punct "<<<") then
              found := true
            else
              j := j + 1
          | .eof => j := cap
          | _ => j := j + 1
        if found then
          -- Consume tokens through the matching '>' (inclusive).
          set { st0 with i := j + 1 }
          pure true
        else
          pure false
      | _ => pure false
    -- Helper: consume `<<<...>>>` launch config if present
    let eatLaunchConfig : ParseM Unit := do
      if (← matchPunct "<<<") then
        let mut depth : Int := 0
        while true do
          match (← peek) with
          | .punct ">>>" =>
            if depth == 0 then bump; break else bump; depth := depth - 1
          | .punct "<<<" => bump; depth := depth + 1
          | .punct "(" => bump; depth := depth + 1
          | .punct ")" => bump; depth := depth - 1
          | .eof => break
          | _ => bump
    if isTemplCall then
      eatLaunchConfig
      expectPunct "("
      let args ← parseArgList
      expectPunct ")"
      pure (CExpr.call name args)
    else
      eatLaunchConfig
      match (← peek) with
      | .punct "(" =>
        bump
        let args ← parseArgList
        expectPunct ")"
        pure (CExpr.call name args)
      | .punct "{" =>
        -- C++ aggregate / value-init: `T{a, b, c}`. Parse the brace
        -- list as a call argument list (we don't model designated
        -- initializers `.field = v` as anything special — those become
        -- expression arguments which the lowerer can recognise via
        -- name binding).
        bump
        let args ← parseArgList
        expectPunct "}"
        pure (CExpr.call name args)
      | _ => pure (CExpr.ident name)
  | .punct "(" =>
    bump
    -- Try cast: `(<type>)<expr>`. Heuristic: peek-ahead to find `)`
    -- and decide based on what follows.
    let s ← get
    let savedI := s.i
    let mut isCast := false
    let mut tyText : String := ""
    let mut endIdx : Nat := savedI
    -- Walk type-like tokens
    let mut j := savedI
    let mut bail := false
    while !bail ∧ j < s.toks.size do
      match s.toks[j]!.tok with
      | .ident t =>
        if isLikelyTypeStart t ∨ tyText ≠ "" then
          tyText := if tyText.isEmpty then t else tyText ++ " " ++ t
          j := j + 1
        else
          bail := true
      | .punct "*" =>
        if tyText ≠ "" then
          tyText := tyText ++ " *"
          j := j + 1
        else bail := true
      | .punct ")" =>
        -- end of cast iff next token starts a unary/primary
        if j + 1 < s.toks.size ∧ tyText ≠ "" then
          match s.toks[j+1]!.tok with
          | .num _ | .fnum _ | .ident _ | .punct "(" | .punct "-" | .punct "!"
          | .punct "~" | .punct "&" | .punct "*" =>
            isCast := true
          | _ => pure ()
        endIdx := j + 1
        bail := true
      | _ => bail := true
    if isCast then
      set { s with i := endIdx }
      let inner ← parseUnary
      pure (CExpr.cast tyText inner)
    else
      let e ← parseExpr
      expectPunct ")"
      pure e
  | t => throw s!"parsePrimary: unexpected {repr t}"

partial def parseArgList : ParseM (Array CExpr) := do
  match (← peek) with
  | .punct ")" => pure #[]
  | _ =>
    let head ← parseAssign
    let mut args : Array CExpr := #[head]
    let rec loop (acc : Array CExpr) : ParseM (Array CExpr) := do
      if (← matchPunct ",") then
        let e ← parseAssign
        loop (acc.push e)
      else
        pure acc
    loop args

partial def parsePostfix : ParseM CExpr := do
  parsePostfix' (← parsePrimary)

partial def parsePostfix' (acc : CExpr) : ParseM CExpr := do
  match (← peek) with
  | .punct "[" =>
    bump
    let idx ← parseExpr
    expectPunct "]"
    parsePostfix' (CExpr.index acc idx)
  | .punct "." =>
    bump
    match (← consume) with
    | .ident f => parsePostfix' (CExpr.member acc f)
    | t => throw s!"expected field name after '.', got {repr t}"
  | .punct "->" =>
    bump
    match (← consume) with
    | .ident f => parsePostfix' (CExpr.arrow acc f)
    | t => throw s!"expected field name after '->', got {repr t}"
  | .punct "++" => bump; parsePostfix' (CExpr.unop .postInc acc)
  | .punct "--" => bump; parsePostfix' (CExpr.unop .postDec acc)
  | .punct "<<<" =>
    -- CUDA kernel launch config: `kernel<<<grid, block[, smem[, stream]]>>>(args)`.
    -- We swallow the launch config tokens (we don't model launch shape
    -- here — the lowerer wires kernels via ShaderM) and recur on the
    -- following call form.
    bump
    let mut depth : Int := 0
    while true do
      match (← peek) with
      | .punct ">>>" =>
        if depth == 0 then bump; break else bump; depth := depth - 1
      | .punct "<<<" => bump; depth := depth + 1
      | .punct "(" => bump; depth := depth + 1
      | .punct ")" => bump; depth := depth - 1
      | .eof => break
      | _ => bump
    parsePostfix' acc
  | _ => pure acc

partial def parseUnary : ParseM CExpr := do
  match (← peek) with
  | .punct "-" => bump; let e ← parseUnary; pure (CExpr.unop .neg e)
  | .punct "!" => bump; let e ← parseUnary; pure (CExpr.unop .logNot e)
  | .punct "~" => bump; let e ← parseUnary; pure (CExpr.unop .bitNot e)
  | .punct "++" => bump; let e ← parseUnary; pure (CExpr.unop .preInc e)
  | .punct "--" => bump; let e ← parseUnary; pure (CExpr.unop .preDec e)
  | .punct "*" => bump; let e ← parseUnary; pure (CExpr.unop .deref e)
  | .punct "&" => bump; let e ← parseUnary; pure (CExpr.unop .addrOf e)
  | .punct "+" => bump; parseUnary
  | _ => parsePostfix

/-- Helper: parse a left-associative binary level. -/
partial def parseBinL (sub : ParseM CExpr) (ops : List (String × BinOp))
    : ParseM CExpr := do
  let lhs ← sub
  parseBinL' sub ops lhs

partial def parseBinL' (sub : ParseM CExpr) (ops : List (String × BinOp))
    (acc : CExpr) : ParseM CExpr := do
  match (← peek) with
  | .punct s =>
    match ops.find? (fun (sym, _) => sym == s) with
    | some (_, op) =>
      bump
      let rhs ← sub
      parseBinL' sub ops (CExpr.binop op acc rhs)
    | none => pure acc
  | _ => pure acc

partial def parseMul : ParseM CExpr :=
  parseBinL parseUnary [("*", .mul), ("/", .div), ("%", .mod)]

partial def parseAdd : ParseM CExpr :=
  parseBinL parseMul [("+", .add), ("-", .sub)]

partial def parseShift : ParseM CExpr :=
  parseBinL parseAdd [("<<", .shl), (">>", .shr)]

partial def parseRel : ParseM CExpr :=
  parseBinL parseShift [("<", .lt), ("<=", .le), (">", .gt), (">=", .ge)]

partial def parseEq : ParseM CExpr :=
  parseBinL parseRel [("==", .eq), ("!=", .ne)]

partial def parseBitAnd : ParseM CExpr :=
  parseBinL parseEq [("&", .bitAnd)]

partial def parseBitXor : ParseM CExpr :=
  parseBinL parseBitAnd [("^", .bitXor)]

partial def parseBitOr : ParseM CExpr :=
  parseBinL parseBitXor [("|", .bitOr)]

partial def parseLogAnd : ParseM CExpr :=
  parseBinL parseBitOr [("&&", .logAnd)]

partial def parseLogOr : ParseM CExpr :=
  parseBinL parseLogAnd [("||", .logOr)]

partial def parseTernary : ParseM CExpr := do
  let c ← parseLogOr
  match (← peek) with
  | .punct "?" =>
    bump
    let t ← parseAssign
    expectPunct ":"
    let e ← parseAssign
    pure (CExpr.ternary c t e)
  | _ => pure c

partial def parseAssign : ParseM CExpr := do
  let lhs ← parseTernary
  let assignOf : String → Option BinOp
    | "="   => some .assign
    | "+="  => some .addAssign
    | "-="  => some .subAssign
    | "*="  => some .mulAssign
    | "/="  => some .divAssign
    | "%="  => some .modAssign
    | "<<=" => some .shlAssign
    | ">>=" => some .shrAssign
    | "&="  => some .andAssign
    | "|="  => some .orAssign
    | "^="  => some .xorAssign
    | _ => none
  match (← peek) with
  | .punct s =>
    match assignOf s with
    | some op => bump; let rhs ← parseAssign; pure (CExpr.binop op lhs rhs)
    | none => pure lhs
  | _ => pure lhs

partial def parseExpr : ParseM CExpr := do
  let lhs ← parseAssign
  parseExpr' lhs

partial def parseExpr' (acc : CExpr) : ParseM CExpr := do
  if (← matchPunct ",") then
    let b ← parseAssign
    parseExpr' (CExpr.comma acc b)
  else pure acc

end -- mutual

/-- Parse a complete CUDA expression from a string. -/
def parseExprStr (src : String) : Except String CExpr :=
  let toks := lex src
  let st : ParseState := { toks }
  match (parseExpr.run.run st) with
  | (.error e, _) => .error e
  | (.ok e, st') =>
    -- Allow trailing eof.
    if h : st'.i < st'.toks.size then
      match st'.toks[st'.i].tok with
      | .eof => .ok e
      | t => .error s!"trailing tokens after expression: {repr t}"
    else
      .ok e

/-! ## Statement parsing (Phase 3) -/

/-- Type-keyword set: tokens that can start a declaration. -/
def isTypeKW (s : String) : Bool :=
  s == "int" ∨ s == "uint" ∨ s == "float" ∨ s == "double"
  ∨ s == "bool" ∨ s == "char" ∨ s == "short" ∨ s == "long"
  ∨ s == "unsigned" ∨ s == "signed" ∨ s == "void"
  ∨ s == "uint8_t" ∨ s == "uint16_t" ∨ s == "uint32_t" ∨ s == "uint64_t"
  ∨ s == "int8_t" ∨ s == "int16_t" ∨ s == "int32_t" ∨ s == "int64_t"
  ∨ s == "half" ∨ s == "half2" ∨ s == "float2" ∨ s == "float4"

/-- Skip optional `const` / `volatile` / `constexpr` / `static` /
    `inline` qualifiers, plus CUDA-specific `__restrict__` / `__device__`
    / `__forceinline__`, accumulating into `tyText`. -/
partial def consumeQualifiers (tyText : String) : ParseM String := do
  match (← peek) with
  | .ident s =>
    if s == "const" ∨ s == "volatile" ∨ s == "constexpr"
       ∨ s == "static" ∨ s == "inline"
       ∨ s == "__restrict__" ∨ s == "__device__"
       ∨ s == "__forceinline__" ∨ s == "__host__" then
      bump
      consumeQualifiers (if tyText.isEmpty then s else tyText ++ " " ++ s)
    else pure tyText
  | _ => pure tyText

/-- Heuristic: keywords that should NOT be treated as type names even
    though they appear after a type-spec might-be-starting position.
    Things like control-flow keywords or `template`. -/
def isReservedNonType (s : String) : Bool :=
  s == "if" ∨ s == "else" ∨ s == "for" ∨ s == "while" ∨ s == "do"
  ∨ s == "return" ∨ s == "switch" ∨ s == "case" ∨ s == "default"
  ∨ s == "break" ∨ s == "continue" ∨ s == "goto"
  ∨ s == "template" ∨ s == "typename" ∨ s == "struct" ∨ s == "union"
  ∨ s == "class" ∨ s == "enum" ∨ s == "typedef" ∨ s == "namespace"
  ∨ s == "using" ∨ s == "static_cast" ∨ s == "reinterpret_cast"
  ∨ s == "const_cast" ∨ s == "dynamic_cast" ∨ s == "sizeof"
  ∨ s == "new" ∨ s == "delete" ∨ s == "this" ∨ s == "true" ∨ s == "false"
  ∨ s == "nullptr"

/-- Read a type spec like `const int *`. Stops just before the variable
    name token. Returns the (possibly empty) type string.

    Phase 11: accepts user-defined type idents (`ggml_type`, `block_q4_K`,
    `cudaStream_t`, ...) so full-file parsing isn't blocked by missing
    names. Also handles `T<args>` template instantiations as types and
    optional `[...]` array suffixes via the caller. -/
partial def parseTypeSpec : ParseM String := do
  -- Optional leading qualifiers
  let mut ty ← consumeQualifiers ""
  -- Base type: any ident that isn't a reserved non-type keyword.
  match (← peek) with
  | .ident s =>
    if isReservedNonType s then
      throw s!"parseTypeSpec: expected type, got '{s}'"
    bump
    let mut s := s
    -- Optional `::` qualified type (`std::array`, `ns::T`)
    while (← matchPunct "::") do
      match (← consume) with
      | .ident s2 => s := s ++ "::" ++ s2
      | t => throw s!"parseTypeSpec: '::' must be followed by ident, got {repr t}"
    ty := if ty.isEmpty then s else ty ++ " " ++ s
    -- Optional secondary type words (e.g. `unsigned int`, `long long`).
    -- Only continue when the next ident is itself a built-in type kw —
    -- user-defined types stand alone.
    while ← (do match (← peek) with
                | .ident s2 => pure (isTypeKW s2)
                | _ => pure false) do
      match (← peek) with
      | .ident s2 => bump; ty := ty ++ " " ++ s2
      | _ => pure ()
    -- Optional `<...>` template arguments after a user-defined type
    -- (e.g. `tile<16, 8, int>`). Skip them as text.
    if (← matchPunct "<") then
      let mut depth : Int := 1
      ty := ty ++ "<"
      while depth > 0 do
        match (← peek) with
        | .punct "<" => bump; depth := depth + 1; ty := ty ++ "<"
        | .punct ">" => bump; depth := depth - 1; ty := ty ++ ">"
        | .eof => break
        | t => bump; ty := ty ++ (match t with
              | .ident s => s
              | .num s => s
              | .fnum s => s
              | .str s => s
              | .punct s => s
              | .eof => "")
  | t => throw s!"parseTypeSpec: expected type, got {repr t}"
  -- Optional trailing qualifiers + pointer markers
  ty ← consumeQualifiers ty
  while (← matchPunct "*") ∨ (← matchPunct "&") do
    ty := ty ++ " *"
    ty ← consumeQualifiers ty
  pure ty

mutual

/-- Parse a single statement. -/
partial def parseStmt : ParseM CStmt := do
  -- Skip any C++11 `[[X]]` attributes leading the statement (inline to
  -- stay inside the mutual block).
  let rec skipAttrs : ParseM Unit := do
    let st ← get
    if st.i + 1 < st.toks.size
       ∧ st.toks[st.i]!.tok == .punct "["
       ∧ st.toks[st.i+1]!.tok == .punct "[" then
      bump; bump
      let mut depth : Int := 0
      while true do
        match (← peek) with
        | .eof => break
        | .punct "[" => bump; depth := depth + 1
        | .punct "]" =>
          if depth == 0 then
            bump
            match (← peek) with
            | .punct "]" => bump
            | _ => pure ()
            break
          else bump; depth := depth - 1
        | _ => bump
      skipAttrs
  skipAttrs
  match (← peek) with
  | .punct "{" =>
    bump
    let mut stmts : Array CStmt := #[]
    while !(← matchPunct "}") do
      let s ← parseStmt
      stmts := stmts.push s
    pure (CStmt.block stmts)
  | .punct "#" =>
    -- `#pragma unroll` etc. — consume tokens until next ; or { or }.
    -- Lexer doesn't track newlines, so we approximate by stopping at
    -- those punct boundaries.
    bump
    let text ← eatPragmaTokens ""
    pure (CStmt.pragma text.trim)
  | .ident kw =>
    if kw == "if" then
      bump
      -- Handle `if constexpr (cond)` — the `constexpr` token is just
      -- a marker before the condition.
      let isConstexpr ← match (← peek) with
        | .ident "constexpr" => bump; pure true
        | _ => pure false
      expectPunct "("
      let c ← parseExpr
      expectPunct ")"
      let thn ← parseStmt
      let els ← match (← peek) with
        | .ident "else" => bump
                           -- `else if` chains: parse next statement
                           -- (which may itself be another `if constexpr`).
                           let s ← parseStmt; pure (some s)
        | _ => pure none
      pure (if isConstexpr then CStmt.ifConstexpr c thn els else CStmt.if_ c thn els)
    else if kw == "static_assert" then
      bump; expectPunct "("
      -- Skip everything up to matching ')'.
      let mut depth := 1
      while depth > 0 do
        match (← consume) with
        | .punct "(" => depth := depth + 1
        | .punct ")" => depth := depth - 1
        | .eof => throw "parseStmt: unmatched '(' in static_assert"
        | _ => pure ()
      expectPunct ";"
      pure CStmt.staticAssert
    else if kw == "extern" then
      -- `extern __shared__ T name[];`
      bump
      match (← peek) with
      | .ident "__shared__" => bump
      | t => throw s!"parseStmt: expected '__shared__' after 'extern', got {repr t}"
      let ty ← parseTypeSpec
      match (← consume) with
      | .ident name =>
        expectPunct "["; expectPunct "]"; expectPunct ";"
        pure (CStmt.externSharedArr ty name)
      | t => throw s!"parseStmt: expected name in 'extern __shared__', got {repr t}"
    else if kw == "for" then
      bump; expectPunct "("
      -- init
      let initStmt ← match (← peek) with
        | .punct ";" => bump; pure none
        | .ident s =>
          if isTypeKW s ∨ s == "const" ∨ s == "constexpr"
             ∨ s == "volatile" ∨ s == "static" then
            -- `int k = 0` / `constexpr int k = 0`
            let s ← parseDeclWithSemi
            pure (some s)
          else
            -- expression init
            let e ← parseExpr
            expectPunct ";"
            pure (some (CStmt.expr e))
        | _ =>
          let e ← parseExpr
          expectPunct ";"
          pure (some (CStmt.expr e))
      -- cond
      let cond ← match (← peek) with
        | .punct ";" => bump; pure none
        | _ =>
          let c ← parseExpr
          expectPunct ";"
          pure (some c)
      -- step
      let step ← match (← peek) with
        | .punct ")" => bump; pure none
        | _ =>
          let e ← parseExpr
          expectPunct ")"
          pure (some e)
      let body ← parseStmt
      pure (CStmt.for_ initStmt cond step body)
    else if kw == "while" then
      bump; expectPunct "("
      let c ← parseExpr
      expectPunct ")"
      let body ← parseStmt
      pure (CStmt.while_ c body)
    else if kw == "return" then
      bump
      let e ← match (← peek) with
        | .punct ";" => pure none
        | _ => let e ← parseExpr; pure (some e)
      expectPunct ";"
      pure (CStmt.return_ e)
    else if kw == "break" then
      bump; expectPunct ";"; pure CStmt.break_
    else if kw == "continue" then
      bump; expectPunct ";"; pure CStmt.continue_
    else if kw == "switch" then
      -- Coarse: parse `switch (expr) { ... }` and lower to a sequential
      -- `block` of inner stmts. We don't model case dispatch; the caller
      -- can post-process via env.inlines if exact dispatch is needed.
      bump; expectPunct "("
      let _ ← parseExpr
      expectPunct ")"
      expectPunct "{"
      let mut acc : Array CStmt := #[]
      while true do
        match (← peek) with
        | .punct "}" => bump; break
        | .eof => break
        | .ident "case" =>
          -- skip `case <expr> :`
          bump
          let _ ← parseExpr
          expectPunct ":"
        | .ident "default" =>
          bump; expectPunct ":"
        | _ =>
          let s ← parseStmt
          acc := acc.push s
      pure (CStmt.block acc)
    else if kw == "case" ∨ kw == "default" then
      -- Stray label outside switch (shouldn't happen but be lenient):
      -- skip the label and recurse on the following statement.
      bump
      if kw == "case" then
        let _ ← parseExpr
        expectPunct ":"
      else
        expectPunct ":"
      parseStmt
    else if kw == "goto" then
      bump
      match (← consume) with
      | .ident _ => expectPunct ";"; pure (CStmt.block #[])
      | t => throw s!"goto: expected label, got {repr t}"
    else if kw == "__syncthreads" then
      bump; expectPunct "("; expectPunct ")"; expectPunct ";"
      pure CStmt.sync_
    else if kw == "__shared__" then
      bump
      parseDeclWithSemiStorage Storage.shared
    else if kw == "__constant__" then
      bump
      parseDeclWithSemiStorage Storage.constant
    else if kw == "typedef" ∨ kw == "using" then
      -- Function-local `typedef T name;` / `using name = T;` — skip
      -- the line. We don't model type aliases; users are expected to
      -- inline the underlying type in the env.
      bump
      while true do
        match (← peek) with
        | .punct ";" => bump; break
        | .eof => break
        | _ => bump
      pure (CStmt.block #[])
    else if isTypeKW kw ∨ kw == "const" ∨ kw == "volatile"
            ∨ kw == "constexpr" ∨ kw == "static" ∨ kw == "inline"
            ∨ kw == "__restrict__" ∨ kw == "__device__"
            ∨ kw == "__forceinline__" ∨ kw == "__host__" then
      parseDeclWithSemi
    else
      -- Heuristic: `<ident> <ident>` (possibly with `*`/`&`/`<...>` in
      -- between) is a typed local-variable declaration, e.g.
      --   `tile_C tile_a;`,  `block_q4_K * b = ...;`,
      --   `std::array<int,4> arr = ...;`.
      let st ← get
      let toks := st.toks
      let mut isDecl := false
      let mut k := st.i + 1
      let cap := k + 12
      let mut depth : Int := 0
      while k < toks.size ∧ k < cap do
        match toks[k]!.tok with
        | .punct "<" | .punct "(" | .punct "[" =>
          depth := depth + 1; k := k + 1
        | .punct ">" | .punct ")" | .punct "]" =>
          depth := depth - 1; k := k + 1
        | .punct "*" | .punct "&" =>
          if depth == 0 then k := k + 1 else k := k + 1
        | .punct "::" => k := k + 1
        | .ident _ =>
          if depth == 0 then
            -- Found a second ident at depth 0 → decl pattern
            -- Distinguish from `func(args)` (depth jumped to 1 then 0) by
            -- checking the previous token: if previous was `)` we are
            -- past a function-call expression, NOT a decl.
            let prevIsCloser :=
              k > 0 ∧
                ((toks[k-1]!.tok == Tok.punct ")") ∨
                 (toks[k-1]!.tok == Tok.punct "]"))
            if !prevIsCloser then
              isDecl := true
            break
          else k := k + 1
        | _ => break
      if isDecl then
        parseDeclWithSemi
      else
        -- Expression statement (e.g. `acc += foo;`, `func();`)
        let e ← parseExpr
        expectPunct ";"
        pure (CStmt.expr e)
  | .eof => throw "parseStmt: unexpected eof"
  | t =>
    -- Expression statement starting with non-ident (e.g. `(int)x` or `*p`)
    let e ← parseExpr
    expectPunct ";"
    pure (CStmt.expr e)

/-- Read pragma-line tokens until we hit a statement boundary. -/
partial def eatPragmaTokens (acc : String) : ParseM String := do
  match (← peek) with
  | .eof => pure acc
  | .punct s =>
    if s == "{" ∨ s == "}" ∨ s == ";" then pure acc
    else bump; eatPragmaTokens (acc ++ " " ++ s)
  | .ident s => bump; eatPragmaTokens (acc ++ " " ++ s)
  | .num s => bump; eatPragmaTokens (acc ++ " " ++ s)
  | _ => pure acc

/-- Parse a declaration, expecting a trailing `;`. Handles
    `int k = 0;`, `int qs[N] = {...};` (no init list yet),
    `int x;`, `int * p = ...;`. Multi-declarators (`int a, b;`) NYI.
    The leading `__shared__` / `__constant__` keyword (if any) is
    captured in `storage`. -/
partial def parseDeclWithSemiStorage (storage : Storage) : ParseM CStmt := do
  let ty ← parseTypeSpec
  match (← consume) with
  | .ident name =>
    -- Optional array size: `name[N]`
    if (← matchPunct "[") then
      let szExpr ← parseExpr
      expectPunct "]"
      -- Optional `= {...}` initializer (we accept and ignore for now)
      if (← matchPunct "=") then
        if (← matchPunct "{") then
          -- skip to matching '}'
          let mut depth := 1
          while depth > 0 do
            match (← consume) with
            | .punct "{" => depth := depth + 1
            | .punct "}" => depth := depth - 1
            | .eof => throw "parseDecl: unmatched '{' in array init"
            | _ => pure ()
        else
          let _ ← parseAssign
      expectPunct ";"
      pure (CStmt.declArr storage ty name szExpr)
    else
      let init ← if (← matchPunct "=") then
        let e ← parseAssign
        pure (some e)
      else if (← matchPunct "{") then
        -- Brace-initialiser: `T name{a, b, c};` or `T name{};`. Parse
        -- as a no-op call expression to capture intent — the lowerer
        -- can decide what to do with empty brace-init (often just
        -- zero-init).
        let args ← parseArgList
        expectPunct "}"
        pure (some (CExpr.call ty args))
      else
        -- C++ direct-init declarator: `T name(a, b, c);` (e.g.
        -- `dim3 num_blocks(x, y, z);`). Only fire when `ty` is a
        -- user-defined type name (NOT a built-in scalar) — we don't
        -- want to misparse `int func(args);` (a function declaration)
        -- as a decl-with-init. Builtin types use `=` for init.
        let isUserType :=
          ¬ (ty == "int" ∨ ty == "uint" ∨ ty == "float" ∨ ty == "double"
             ∨ ty == "bool" ∨ ty == "char" ∨ ty == "short" ∨ ty == "long"
             ∨ ty == "void" ∨ ty.endsWith "*")
        if isUserType ∧ (← matchPunct "(") then
          let args ← parseArgList
          expectPunct ")"
          pure (some (CExpr.call ty args))
        else pure none
      expectPunct ";"
      pure (CStmt.decl storage ty name init)
  | t => throw s!"parseDeclWithSemi: expected ident, got {repr t}"

partial def parseDeclWithSemi : ParseM CStmt :=
  parseDeclWithSemiStorage Storage.none

end -- mutual

/-- Parse a complete CUDA statement (or block) from a string. -/
def parseStmtStr (src : String) : Except String CStmt :=
  let toks := lex src
  let st : ParseState := { toks }
  match (parseStmt.run.run st) with
  | (.error e, _) => .error e
  | (.ok s, st') =>
    if h : st'.i < st'.toks.size then
      match st'.toks[st'.i].tok with
      | .eof => .ok s
      | t => .error s!"trailing tokens after statement: {repr t}"
    else
      .ok s

/-! ## Function-prototype parsing (Phase 4) -/

/-- Try to consume a C++11 attribute specifier `[[ ... ]]` at the
    current position. Returns true if one was eaten. Handles balanced
    `[ ]` inside the attribute body (e.g. `[[gnu::aligned(16)]]`). -/
partial def tryEatCxxAttr : ParseM Bool := do
  let st ← get
  let i := st.i
  let n := st.toks.size
  if i + 1 < n then
    let t0 := st.toks[i]!.tok
    let t1 := st.toks[i+1]!.tok
    if t0 == .punct "[" ∧ t1 == .punct "[" then
      bump; bump
      -- consume balanced until matching `]]`
      let mut depth : Int := 0
      while true do
        match (← peek) with
        | .eof => break
        | .punct "[" => bump; depth := depth + 1
        | .punct "]" =>
          if depth == 0 then
            bump
            -- expect another `]`
            match (← peek) with
            | .punct "]" => bump
            | _ => pure ()
            break
          else
            bump; depth := depth - 1
        | _ => bump
      return true
  pure false

/-- Skip CUDA function attributes: `__device__`, `__forceinline__`,
    `__global__`, `inline`, `static`, etc. Also accepts attribute-like
    constructs with parenthesised args, e.g. `__launch_bounds__(N, 1)`,
    and C++11 `[[X]]` attributes.
    Returns the list of attributes captured. -/
partial def parseAttrs : ParseM (Array String) := do
  let rec loop (acc : Array String) : ParseM (Array String) := do
    -- Eat any C++11 `[[X]]` attributes at this position first
    if (← tryEatCxxAttr) then
      loop (acc.push "[[attr]]")
    else
      match (← peek) with
      | .ident s =>
        if s == "__device__" ∨ s == "__forceinline__" ∨ s == "__global__"
           ∨ s == "__host__" ∨ s == "inline" ∨ s == "static"
           ∨ s == "extern" ∨ s == "constexpr" then
          bump; loop (acc.push s)
        else if s == "__launch_bounds__" ∨ s == "__align__"
                ∨ s == "__attribute__" then
          -- attribute(args)  — consume balanced parens after the keyword
          bump
          if (← matchPunct "(") then
            let mut depth : Int := 1
            while depth > 0 do
              match (← peek) with
              | .punct "(" => bump; depth := depth + 1
              | .punct ")" => bump; depth := depth - 1
              | .eof => break
              | _ => bump
          loop (acc.push s)
        else pure acc
      | _ => pure acc
  loop #[]

/-- Parse `<T1 a, T2 b = default, ...>` after an initial `template`.
    Position is just past the keyword. Default values (`= expr`) are
    accepted and dropped — kernel call sites give explicit args. -/
partial def parseTemplateParams : ParseM (Array CTemplParam) := do
  expectPunct "<"
  if (← matchPunct ">") then pure #[]
  else
    let parseOne : ParseM CTemplParam := do
      -- Type token (treat `int`, `bool`, `ggml_type`, `typename` etc. as opaque)
      match (← consume) with
      | .ident ty =>
        -- `typename T` is the C++ form; `ty` is just a tag here.
        match (← consume) with
        | .ident name =>
          -- Optional default: `= <expr-up-to-comma-or-gt>`. We skip
          -- balanced tokens until the next top-level `,` or `>`.
          if (← matchPunct "=") then
            let mut depth : Int := 0
            while true do
              match (← peek) with
              | .punct "(" | .punct "[" | .punct "<" =>
                bump; depth := depth + 1
              | .punct ")" | .punct "]" =>
                bump; depth := depth - 1
              | .punct ">" =>
                if depth == 0 then break else bump; depth := depth - 1
              | .punct "," =>
                if depth == 0 then break else bump
              | .eof => break
              | _ => bump
          pure { ty, name }
        | t => throw s!"parseTemplateParams: expected name, got {repr t}"
      | t => throw s!"parseTemplateParams: expected type ident, got {repr t}"
    let head ← parseOne
    let rec loop (acc : Array CTemplParam) : ParseM (Array CTemplParam) := do
      if (← matchPunct ",") then
        let p ← parseOne; loop (acc.push p)
      else pure acc
    let result ← loop #[head]
    expectPunct ">"
    pure result

/-- Parse a single function parameter: `<type> <name> [= <default>]`.
    The default expression is parsed if present; we accept it but the
    lowering can ignore it for kernels invoked with explicit arguments.

    Also accepts an optional `[]` / `[N]` suffix after the name (treated
    as a pointer-equivalent for purposes of type-text capture). -/
partial def parseParam : ParseM CParam := do
  -- Eat any leading `[[X]]` attributes on the parameter
  let rec skipParamAttrs : ParseM Unit := do
    let st ← get
    if st.i + 1 < st.toks.size
       ∧ st.toks[st.i]!.tok == .punct "["
       ∧ st.toks[st.i+1]!.tok == .punct "[" then
      bump; bump
      let mut depth : Int := 0
      while true do
        match (← peek) with
        | .eof => break
        | .punct "[" => bump; depth := depth + 1
        | .punct "]" =>
          if depth == 0 then
            bump
            match (← peek) with
            | .punct "]" => bump
            | _ => pure ()
            break
          else bump; depth := depth - 1
        | _ => bump
      skipParamAttrs
  skipParamAttrs
  let ty ← parseTypeSpec
  -- Anonymous parameter (rare but legal: `void f(int)`)
  match (← peek) with
  | .punct "," | .punct ")" =>
    pure { ty, name := "_anon" }
  | _ =>
    match (← consume) with
    | .ident name =>
      -- Optional `[N]` / `[]` suffix
      let mut tyAdj := ty
      while (← matchPunct "[") do
        tyAdj := tyAdj ++ " *"
        -- skip everything until matching `]`
        let mut depth : Int := 1
        while depth > 0 do
          match (← peek) with
          | .punct "[" => bump; depth := depth + 1
          | .punct "]" => bump; depth := depth - 1
          | .eof => break
          | _ => bump
      -- Optional `= <default-expr>`. The default expression is at
      -- assignment precedence and stops at `,` or `)`.
      if (← matchPunct "=") then
        let d ← parseAssign
        pure { ty := tyAdj, name, default? := some d }
      else
        pure { ty := tyAdj, name }
    | t => throw s!"parseParam: expected name, got {repr t}"

/-- Parse `( <param>, <param>, ... )`. Position is just past `(`. -/
partial def parseParams : ParseM (Array CParam) := do
  if (← matchPunct ")") then pure #[]
  else
    let head ← parseParam
    let rec loop (acc : Array CParam) : ParseM (Array CParam) := do
      if (← matchPunct ",") then
        let p ← parseParam; loop (acc.push p)
      else pure acc
    let result ← loop #[head]
    expectPunct ")"
    pure result

/-- Parse a CUDA function definition:

    [template <...>] [attrs] retTy name(params) { body }

    `attrs` covers `__device__`, `__global__`, `__forceinline__`, etc.
    `retTy` is captured as a string. -/
partial def parseFunction : ParseM CFunction := do
  let templParams ← match (← peek) with
    | .ident "template" => bump; parseTemplateParams
    | _ => pure #[]
  let attrs ← parseAttrs
  -- Return type may be `void`, `int`, `static int`, etc. Reuse
  -- `parseTypeSpec` which handles primitive types + qualifiers.
  let retTy ← parseTypeSpec
  match (← consume) with
  | .ident name =>
    -- C++ member-function definition: `RetTy ClassName::method(...) { ... }`.
    let mut name := name
    while (← matchPunct "::") do
      match (← consume) with
      | .ident s => name := name ++ "::" ++ s
      | t => throw s!"parseFunction: '::' must be followed by ident, got {repr t}"
    expectPunct "("
    let params ← parseParams
    -- Optional `const` / `noexcept` / etc. qualifiers after the param list
    let mut keepGoing := true
    while keepGoing do
      match (← peek) with
      | .ident s =>
        if s == "const" ∨ s == "noexcept" ∨ s == "override" ∨ s == "final" then
          bump
        else
          keepGoing := false
      | _ => keepGoing := false
    -- Optional member initializer list `: a(x), b(y)` before `{`.
    if (← matchPunct ":") then
      while true do
        match (← peek) with
        | .punct "{" => break
        | .eof => break
        | _ => bump
    let body ← parseStmt   -- expects a `{ ... }` block
    pure { attrs, templParams, retTy, name, params, body }
  | t => throw s!"parseFunction: expected function name, got {repr t}"

/-- Parse a complete CUDA function from a string. -/
def parseFunctionStr (src : String) : Except String CFunction :=
  let toks := lex src
  let st : ParseState := { toks }
  match (parseFunction.run.run st) with
  | (.error e, _) => .error e
  | (.ok f, st') =>
    if h : st'.i < st'.toks.size then
      match st'.toks[st'.i].tok with
      | .eof => .ok f
      | t => .error s!"trailing tokens after function: {repr t}"
    else
      .ok f

/-! ## Translation-unit parsing (Phase 11)

Parses a full `.cu` / `.cuh` source file. Skips top-level constructs we
can't lower (struct/typedef/enum definitions, `extern "C"` blocks,
namespaces, `using` declarations, top-level `template <...>` aliases,
forward function declarations without a body) and collects per-item
results so the caller can report which functions parsed and which did
not. Preprocessor directives are stripped by the lexer. -/

/-- Outcome of parsing one top-level item. -/
inductive TUItem where
  | function (f : CFunction)
  /-- Skipped non-function: keyword that started it + reason. -/
  | skipped  (kind : String) (info : String)
  /-- Parse error inside what looked like a function. -/
  | failed   (header : String) (err : String)
  deriving Inhabited

/-- Walk forward, balancing `()`, `[]`, `{}`, until depth returns to 0
    AND we hit one of the terminator punct tokens (typically `;` or
    `}` at depth 0). Used to skip past unparseable items. -/
partial def skipBalancedUntil (terms : List String) : ParseM Unit := do
  let mut depth : Int := 0
  let mut steps : Nat := 0
  while steps < 1_000_000 do
    steps := steps + 1
    match (← peek) with
    | .eof => return
    | .punct p =>
      if p == "(" ∨ p == "[" ∨ p == "{" then
        depth := depth + 1; bump
      else if p == ")" ∨ p == "]" ∨ p == "}" then
        if depth > 0 then
          depth := depth - 1; bump
          if depth == 0 ∧ terms.contains p then return
        else
          -- spurious closer at depth 0; consume and stop
          bump; return
      else if depth == 0 ∧ terms.contains p then
        bump; return
      else
        bump
    | _ => bump

/-- Lookahead: scan forward (without modifying state) to determine if
    the upcoming top-level item is a function definition (i.e. a
    parenthesised parameter list followed by a `{` brace). Returns the
    detected item kind:
      "function" — parse with parseFunction
      "extern-c" — `extern "C" { ... }` block; descend into its body
      "namespace"— `namespace X { ... }` block; descend
      "struct"   — `struct/union/class/enum` definition (skip)
      "typedef"  — `typedef ...;` (skip)
      "using"    — `using X = ...;` (skip)
      "template" — top-level `template <...>` ALIAS (skip if not function)
      "decl"     — top-level variable declaration `... ;` (skip)
      "unknown"  — could not classify -/
partial def classifyTopLevel : ParseM String := do
  let st ← get
  let toks := st.toks
  let mut j := st.i
  let n := toks.size
  -- Skip leading attributes / template heads / qualifiers in the lookahead
  let mut sawTemplate := false
  while j < n do
    match toks[j]!.tok with
    | .ident "template" =>
      sawTemplate := true
      j := j + 1
      -- skip balanced <...>
      if j < n ∧ toks[j]!.tok == .punct "<" then
        let mut d : Int := 1
        j := j + 1
        while d > 0 ∧ j < n do
          match toks[j]!.tok with
          | .punct "<" => d := d + 1
          | .punct ">" => d := d - 1
          | _ => pure ()
          j := j + 1
    | .ident s =>
      if s == "__device__" ∨ s == "__forceinline__" ∨ s == "__global__"
         ∨ s == "__host__" ∨ s == "inline" ∨ s == "static"
         ∨ s == "constexpr" then
        j := j + 1
      else
        break
    | _ => break
  if j >= n then return "unknown"
  -- Classify by leading keyword
  match toks[j]!.tok with
  | .ident "extern" =>
    -- `extern "C" { ... }` — punct after extern is a string-as-ident
    -- but our lexer doesn't handle string literals. Check for `extern`
    -- followed by `{` after a punct skip. Conservatively treat as
    -- extern-c if a `{` appears within 4 tokens at the same depth.
    return "extern-c"
  | .ident "namespace" => return "namespace"
  | .ident "struct" | .ident "union" | .ident "class" | .ident "enum" =>
    -- Could be `struct Foo { ... };` (skip) OR `struct Foo func(...) { ... }`
    -- (function returning Foo). Disambiguate by scanning forward.
    -- For simplicity: scan forward for `{` vs `(`. If `(` appears first
    -- without a `{`, it's a function; else it's a definition to skip.
    let mut k := j + 1
    let cap := k + 64
    while k < n ∧ k < cap do
      match toks[k]!.tok with
      | .punct "(" => return "function"
      | .punct "{" => return "struct"
      | .punct ";" => return "decl"
      | _ => pure ()
      k := k + 1
    return "unknown"
  | .ident "typedef" => return "typedef"
  | .ident "using" => return "using"
  | _ =>
    -- Generic: scan forward for `(` vs `=` vs `;` vs `{`.
    -- A function has the shape `<type> <name> ( ... ) { ... }`.
    -- A var decl is `<type> <name> [= ...] ;`.
    let mut k := j
    let cap := k + 256
    while k < n ∧ k < cap do
      match toks[k]!.tok with
      | .punct "(" =>
        -- ensure a `{` appears after the matching `)` — that's a fn
        let mut d : Int := 1
        let mut m := k + 1
        while d > 0 ∧ m < n do
          match toks[m]!.tok with
          | .punct "(" => d := d + 1
          | .punct ")" => d := d - 1
          | _ => pure ()
          m := m + 1
        if m < n ∧ toks[m]!.tok == .punct "{" then return "function"
        else return "decl"
      | .punct "=" => return "decl"
      | .punct ";" => return "decl"
      | .punct "{" => return "decl"
      | _ => pure ()
      k := k + 1
    return "unknown"

/-- Capture the first ~6 tokens at the current position as a string,
    for use in error messages. -/
partial def headerSnippet : ParseM String := do
  let st ← get
  let toks := st.toks
  let n := min toks.size (st.i + 8)
  let mut acc : String := ""
  let mut k := st.i
  while k < n do
    let s := match toks[k]!.tok with
      | .ident s => s
      | .num s => s
      | .fnum s => s
      | .str s => s!"\"{s}\""
      | .punct s => s
      | .eof => ""
    acc := acc ++ s ++ " "
    k := k + 1
  pure acc.trim

/-- Skip a top-level item we can't lower, returning a brief tag. -/
partial def skipTopLevelItem (kind : String) : ParseM TUItem := do
  let header ← headerSnippet
  match kind with
  | "extern-c" =>
    -- consume `extern`
    bump
    -- consume an optional ident token (e.g. `"C"` lexed weirdly) until `{`
    let mut steps := 0
    while steps < 8 do
      steps := steps + 1
      match (← peek) with
      | .punct "{" => break
      | .eof => return .skipped "extern-c" header
      | _ => bump
    -- enter block: parse contents recursively until the matching `}`.
    -- Caller handles this; here we just descend and return marker.
    return .skipped "extern-c-open" header
  | "namespace" =>
    -- skip until `{`, then mark open (caller descends)
    let mut steps := 0
    while steps < 16 do
      steps := steps + 1
      match (← peek) with
      | .punct "{" => break
      | .eof => return .skipped "namespace" header
      | _ => bump
    return .skipped "namespace-open" header
  | "struct" | "typedef" | "using" | "decl" =>
    skipBalancedUntil [";", "}"]
    return .skipped kind header
  | _ =>
    skipBalancedUntil [";", "}"]
    return .skipped "unknown" header

/-- Parse one top-level item, recovering on parse errors by jumping
    past the next `;`/`}` and continuing. -/
partial def parseTopLevelItem : ParseM TUItem := do
  let kind ← classifyTopLevel
  if kind == "function" then
    let header ← headerSnippet
    -- snapshot state for recovery
    let st0 ← get
    match (← (parseFunction : ParseM CFunction).run.run st0) with
    | (.ok f, st1) =>
      set st1
      return .function f
    | (.error e, _) =>
      -- recover: from snapshot, skip past the function body
      set st0
      -- consume tokens until we land just past a top-level `}` or `;`
      skipBalancedUntil ["}", ";"]
      return .failed header e
  else
    skipTopLevelItem kind

/-- Parse a translation unit: a sequence of top-level items. Descends
    into `extern "C" { ... }` and `namespace X { ... }` blocks. -/
partial def parseTranslationUnit : ParseM (Array TUItem) := do
  let mut acc : Array TUItem := #[]
  let mut steps : Nat := 0
  while steps < 100_000 do
    steps := steps + 1
    match (← peek) with
    | .eof => break
    | .punct "}" =>
      -- end of an enclosing block (extern-c / namespace) — let caller handle
      bump
      -- treat as benign
      pure ()
    | _ =>
      let item ← parseTopLevelItem
      acc := acc.push item
  pure acc

/-- Parse a complete CUDA translation unit (`.cu`/`.cuh`) from a
    string. Returns per-item results — never fails as a whole. -/
def parseTranslationUnitStr (src : String) : Array TUItem :=
  let toks := lex src
  let st : ParseState := { toks }
  match (parseTranslationUnit.run.run st) with
  | (.ok arr, _) => arr
  | (.error e, st') =>
    -- shouldn't happen — parseTopLevelItem swallows errors — but
    -- preserve diagnostic
    let leftover := st'.toks.size - st'.i
    #[TUItem.failed s!"<TU error after {st'.i}/{st'.toks.size} toks, {leftover} left>" e]

/-- Parse a block of statements (sequence). -/
def parseStmtsStr (src : String) : Except String (Array CStmt) :=
  let toks := lex src
  let st : ParseState := { toks }
  let act : ParseM (Array CStmt) := do
    let mut acc : Array CStmt := #[]
    while (← peek) != .eof do
      let s ← parseStmt
      acc := acc.push s
    pure acc
  match (act.run.run st) with
  | (.error e, _) => .error e
  | (.ok arr, _) => .ok arr

end Hesper.Transpile.CUDA
