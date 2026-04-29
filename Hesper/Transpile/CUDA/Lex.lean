/-! # CUDA → ShaderM transpiler — lexer

Small lexer for the CUDA C++ subset that appears in llama.cpp's
`mmq.cuh`, `mmvq.cu`, `vecdotq.cuh`, `quantize.cu`. Not a full C
lexer — only what the kernels we care about actually use.

Tokens:
  ident   identifiers (and keywords; the parser disambiguates)
  num     integer literal (decimal or hex), suffix `[uUlL]+` allowed
  fnum    float literal (`1.5f`, `0.0f`, `1e-3f`)
  punct   1- to 3-char operator/punctuation
  eof

Whitespace and `//` and `/* */` comments are skipped.
-/
namespace Hesper.Transpile.CUDA

inductive Tok where
  | ident (s : String)
  | num   (s : String)
  | fnum  (s : String)
  | punct (s : String)
  | eof
  deriving Repr, Inhabited, BEq

structure Pos where
  line : Nat := 1
  col  : Nat := 1
  deriving Repr, Inhabited, BEq

structure Spanned where
  tok : Tok
  pos : Pos
  deriving Repr, Inhabited

private def isIdentStart (c : Char) : Bool := c.isAlpha ∨ c = '_'
private def isIdentCont (c : Char) : Bool := c.isAlphanum ∨ c = '_'
private def isHexDigit (c : Char) : Bool :=
  c.isDigit ∨ ('a' ≤ c ∧ c ≤ 'f') ∨ ('A' ≤ c ∧ c ≤ 'F')

/-- 2-char operators we recognise (anything not listed becomes two
    1-char punct tokens). -/
private def known2 : Array String := #[
  "<<", ">>", "<=", ">=", "==", "!=", "&&", "||",
  "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=",
  "++", "--", "->"]

private def known3 : Array String := #["<<=", ">>=", "..."]

/-- Lex a CUDA source string into a token array. Always succeeds —
    unrecognised characters become single-char `punct` tokens. -/
partial def lex (src : String) : Array Spanned := Id.run do
  let mut toks : Array Spanned := #[]
  let mut i : String.Pos := 0
  let mut line : Nat := 1
  let mut col : Nat := 1
  let endPos := src.endPos

  let advance (i : String.Pos) (line col : Nat) (n : Nat := 1) : String.Pos × Nat × Nat := Id.run do
    let mut i := i
    let mut line := line
    let mut col := col
    for _ in [0:n] do
      if i < endPos then
        let c := src.get i
        i := src.next i
        if c = '\n' then line := line + 1; col := 1
        else col := col + 1
    pure (i, line, col)

  while i < endPos do
    let c := src.get i
    -- Skip whitespace
    if c = ' ' ∨ c = '\t' ∨ c = '\r' ∨ c = '\n' then
      let (i', l', c') := advance i line col 1
      i := i'; line := l'; col := c'
      continue
    -- Skip comments
    if c = '/' ∧ src.next i < endPos then
      let c2 := src.get (src.next i)
      if c2 = '/' then
        -- line comment
        let mut j := src.next (src.next i)
        while j < endPos ∧ src.get j ≠ '\n' do
          j := src.next j
        col := col + (j.byteIdx - i.byteIdx)
        i := j
        continue
      if c2 = '*' then
        -- block comment
        let mut j := src.next (src.next i)
        let mut l := line
        let mut cc := col + 2
        while j < endPos do
          let ch := src.get j
          if ch = '*' ∧ src.next j < endPos ∧ src.get (src.next j) = '/' then
            j := src.next (src.next j); cc := cc + 2; break
          if ch = '\n' then l := l + 1; cc := 1 else cc := cc + 1
          j := src.next j
        i := j; line := l; col := cc
        continue
    let pos : Pos := { line, col }
    -- Identifier
    if isIdentStart c then
      let mut acc := ""
      let mut j := i
      while j < endPos ∧ isIdentCont (src.get j) do
        acc := acc.push (src.get j); j := src.next j
      let len := j.byteIdx - i.byteIdx
      let (i', l', c') := advance i line col len
      i := j; line := l'; col := c'
      toks := toks.push { tok := .ident acc, pos }
      continue
    -- Number (integer or float)
    if c.isDigit ∨ (c = '.' ∧ src.next i < endPos ∧ (src.get (src.next i)).isDigit) then
      let mut acc := ""
      let mut j := i
      let mut isFloat := false
      -- Hex prefix?
      if c = '0' ∧ src.next i < endPos
         ∧ (src.get (src.next i) = 'x' ∨ src.get (src.next i) = 'X') then
        acc := "0".push (src.get (src.next i))
        j := src.next (src.next i)
        while j < endPos ∧ isHexDigit (src.get j) do
          acc := acc.push (src.get j); j := src.next j
      else
        -- Decimal integer part
        while j < endPos ∧ (src.get j).isDigit do
          acc := acc.push (src.get j); j := src.next j
        -- Fractional part
        if j < endPos ∧ src.get j = '.' then
          isFloat := true
          acc := acc.push '.'; j := src.next j
          while j < endPos ∧ (src.get j).isDigit do
            acc := acc.push (src.get j); j := src.next j
        -- Exponent
        if j < endPos ∧ (src.get j = 'e' ∨ src.get j = 'E') then
          isFloat := true
          acc := acc.push (src.get j); j := src.next j
          if j < endPos ∧ (src.get j = '+' ∨ src.get j = '-') then
            acc := acc.push (src.get j); j := src.next j
          while j < endPos ∧ (src.get j).isDigit do
            acc := acc.push (src.get j); j := src.next j
        -- f/F suffix marks float
        if j < endPos ∧ (src.get j = 'f' ∨ src.get j = 'F') then
          isFloat := true
          acc := acc.push (src.get j); j := src.next j
      -- Integer suffixes (u/U/l/L)
      while j < endPos
            ∧ (src.get j = 'u' ∨ src.get j = 'U'
               ∨ src.get j = 'l' ∨ src.get j = 'L') do
        acc := acc.push (src.get j); j := src.next j
      let len := j.byteIdx - i.byteIdx
      let (i', l', c') := advance i line col len
      i := j; line := l'; col := c'
      let t := if isFloat then Tok.fnum acc else Tok.num acc
      toks := toks.push { tok := t, pos }
      continue
    -- Punctuation: try 3-char, then 2-char, then 1-char.
    let p1 := src.next i
    let c2 := if p1 < endPos then some (src.get p1) else none
    let p2 := if p1 < endPos then src.next p1 else p1
    let c3 := if p2 < endPos then some (src.get p2) else none
    let three : Option String := match c2, c3 with
      | some a, some b => some (String.mk [c, a, b])
      | _, _ => none
    let two : Option String := match c2 with
      | some a => some (String.mk [c, a])
      | _ => none
    if let some t := three then
      if known3.contains t then
        let (i', l', c') := advance i line col 3
        i := i'; line := l'; col := c'
        toks := toks.push { tok := .punct t, pos }
        continue
    if let some t := two then
      if known2.contains t then
        let (i', l', c') := advance i line col 2
        i := i'; line := l'; col := c'
        toks := toks.push { tok := .punct t, pos }
        continue
    -- Single-char punct
    let (i', l', c') := advance i line col 1
    i := i'; line := l'; col := c'
    toks := toks.push { tok := .punct (String.mk [c]), pos }
  toks := toks.push { tok := .eof, pos := { line, col } }
  pure toks

end Hesper.Transpile.CUDA
