/-!
# Float String Parser

Parses float strings supporting:
- Decimal: "3.14", "0.001", "100"
- Scientific: "1e-4", "2.5e3", "1.0E-7"
- Signs: "-0.5", "+1e3", "1e+2", "1e-4"
-/

namespace Hesper.Training.ParseFloat

/-- Check if a character is a digit -/
private def isDigit (c : Char) : Bool :=
  c.toNat >= '0'.toNat && c.toNat <= '9'.toNat

/-- Find the position of 'e' or 'E' in a string, if any -/
private def findExpSep (s : String) : Option String.Pos := Id.run do
  let mut i := 0
  for c in s.toList do
    if c == 'e' || c == 'E' then
      return some ⟨i⟩
    i := i + c.utf8Size
  return none

/-- Find the position of '.' in a string, if any -/
private def findDot (s : String) : Option String.Pos := Id.run do
  let mut i := 0
  for c in s.toList do
    if c == '.' then
      return some ⟨i⟩
    i := i + c.utf8Size
  return none

/-- Parse a float string supporting decimal and scientific notation.
    Returns 0.0 for unparseable input. -/
def parseFloat (s : String) : Float :=
  let s := s.trim
  if s.isEmpty then 0.0
  else
    -- Split on 'e'/'E'
    match findExpSep s with
    | some ePos =>
      let mantissaStr := s.extract 0 ePos
      let expStr := s.extract ⟨ePos.byteIdx + 1⟩ ⟨s.utf8ByteSize⟩
      let mantissa := parseMantissa mantissaStr
      let expVal := parseSignedInt expStr
      mantissa * Float.pow 10.0 expVal
    | none =>
      parseMantissa s
where
  /-- Parse optional sign + digits + optional decimal -/
  parseMantissa (s : String) : Float :=
    let (sign, rest) := if s.startsWith "-" then (-1.0, s.drop 1)
      else if s.startsWith "+" then (1.0, s.drop 1)
      else (1.0, s)
    match findDot rest with
    | some dotPos =>
      let intStr := rest.extract 0 dotPos
      let fracStr := rest.extract ⟨dotPos.byteIdx + 1⟩ ⟨rest.utf8ByteSize⟩
      let intVal := (intStr.toNat?.getD 0).toFloat
      let fracVal := (fracStr.toNat?.getD 0).toFloat
      let fracDiv := Float.pow 10.0 fracStr.length.toFloat
      sign * (intVal + fracVal / fracDiv)
    | none =>
      sign * (rest.toNat?.getD 0).toFloat
  /-- Parse a signed integer string -/
  parseSignedInt (s : String) : Float :=
    let (sign, rest) := if s.startsWith "-" then (-1.0, s.drop 1)
      else if s.startsWith "+" then (1.0, s.drop 1)
      else (1.0, s)
    sign * (rest.toNat?.getD 0).toFloat

end Hesper.Training.ParseFloat
