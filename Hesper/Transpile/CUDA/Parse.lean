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

end Hesper.Transpile.CUDA
