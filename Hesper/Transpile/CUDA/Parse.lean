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
  | .ident name =>
    bump
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
            if depth == 0 ∧ j + 1 < toks.size ∧ toks[j+1]!.tok == .punct "(" then
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
    if isTemplCall then
      expectPunct "("
      let args ← parseArgList
      expectPunct ")"
      pure (CExpr.call name args)
    else
      match (← peek) with
      | .punct "(" =>
        bump
        let args ← parseArgList
        expectPunct ")"
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

/-- Skip optional `const` / `volatile` qualifiers, accumulating into `tyText`. -/
partial def consumeQualifiers (tyText : String) : ParseM String := do
  match (← peek) with
  | .ident s =>
    if s == "const" ∨ s == "volatile" ∨ s == "__restrict__" ∨ s == "__device__" then
      bump
      consumeQualifiers (if tyText.isEmpty then s else tyText ++ " " ++ s)
    else pure tyText
  | _ => pure tyText

/-- Read a type spec like `const int *`. Stops just before the variable
    name token. Returns the (possibly empty) type string. -/
partial def parseTypeSpec : ParseM String := do
  -- Optional leading qualifiers
  let mut ty ← consumeQualifiers ""
  -- Base type: must be at least one ident that's a type keyword
  match (← peek) with
  | .ident s =>
    if isTypeKW s then
      bump
      ty := if ty.isEmpty then s else ty ++ " " ++ s
      -- Optional secondary types (e.g. `unsigned int`)
      while ← (do match (← peek) with
                  | .ident s2 => pure (isTypeKW s2)
                  | _ => pure false) do
        match (← peek) with
        | .ident s2 => bump; ty := ty ++ " " ++ s2
        | _ => pure ()
    else
      throw s!"parseTypeSpec: expected type, got '{s}'"
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
          if isTypeKW s ∨ s == "const" then
            -- `int k = 0`
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
    else if kw == "__syncthreads" then
      bump; expectPunct "("; expectPunct ")"; expectPunct ";"
      pure CStmt.sync_
    else if kw == "__shared__" then
      bump
      parseDeclWithSemiStorage Storage.shared
    else if kw == "__constant__" then
      bump
      parseDeclWithSemiStorage Storage.constant
    else if isTypeKW kw ∨ kw == "const" ∨ kw == "volatile"
            ∨ kw == "__restrict__" ∨ kw == "__device__" then
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

/-- Skip CUDA function attributes: `__device__`, `__forceinline__`,
    `__global__`, `inline`, `static`, etc. Returns the list of
    attributes captured. -/
partial def parseAttrs : ParseM (Array String) := do
  let rec loop (acc : Array String) : ParseM (Array String) := do
    match (← peek) with
    | .ident s =>
      if s == "__device__" ∨ s == "__forceinline__" ∨ s == "__global__"
         ∨ s == "__host__" ∨ s == "inline" ∨ s == "static"
         ∨ s == "extern" ∨ s == "constexpr" then
        bump; loop (acc.push s)
      else pure acc
    | _ => pure acc
  loop #[]

/-- Parse `<T1 a, T2 b, ...>` after an initial `template`. Position is
    just past the keyword. -/
partial def parseTemplateParams : ParseM (Array CTemplParam) := do
  expectPunct "<"
  if (← matchPunct ">") then pure #[]
  else
    let parseOne : ParseM CTemplParam := do
      -- Type token (treat `int`, `bool`, `ggml_type` etc. as opaque ident)
      match (← consume) with
      | .ident ty =>
        match (← consume) with
        | .ident name => pure { ty, name }
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
    lowering can ignore it for kernels invoked with explicit arguments. -/
partial def parseParam : ParseM CParam := do
  let ty ← parseTypeSpec
  match (← consume) with
  | .ident name =>
    -- Optional `= <default-expr>`. The default expression is at
    -- assignment precedence and stops at `,` or `)`.
    if (← matchPunct "=") then
      let d ← parseAssign
      pure { ty, name, default? := some d }
    else
      pure { ty, name }
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
    expectPunct "("
    let params ← parseParams
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
