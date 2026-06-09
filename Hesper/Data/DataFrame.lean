import Hesper.Core.Float32Array

/-!
# `Hesper.Data.DataFrame` — minimal CSV-backed data analysis primitives

This module is the **Option A MVP** for tutorial Ch11: just enough of a
DataFrame API to port the Sabela CaliforniaHousing example to Lean.

Scope (intentionally small):
* Schema-on-read CSV loading with `Float` / `String` / `Option Float`.
* Row-level selection (`take`, `randomSplit`).
* Column-level projection (`select`, `exclude`, `columnNames`,
  `dimensions`).
* Missing-value imputation (`impute`).
* Derived columns (`derive`, `deriveMany`) computed from scalars per row.
* Min-max normalisation across all numeric columns (`normalizeFeatures`).
* Markdown pretty-printer for Jupyter (`toMarkdown`).
* GPU upload (`toTensor`) producing a `Float32Array` ready for matmul.

Out of scope (Option B will add it on top):
* Typed column references via Lean macros.
* `Column` algebra (`+ - * /` lifted to columns).
* GroupBy / Join / Arrow interop.

The API is shaped after
[DataHaskell's `dataframe`](https://github.com/mchav/dataframe) (the
library Sabela wraps), so users coming from that world should find it
familiar.  Naming choices follow Sabela where they map cleanly to Lean.

## Acknowledgements

The data-analysis surface (column-oriented design, `impute / derive /
randomSplit / normalizeFeatures` helpers, min-max normalisation
choice, walk-through structure of the California Housing example)
follows
[DataHaskell's `dataframe`](https://github.com/mchav/dataframe)
and
[Sabela's port to Hasktorch](https://github.com/DataHaskell/sabela).
Implementation is fresh Lean 4 code; semantics, normalisation policy
and the example layout are adapted from those projects.  Thanks to
their authors.
-/

namespace Hesper.Data

/-- A cell in a `DataFrame`.  `missing` represents NA / NaN. -/
inductive Value
  | f64     : Float → Value
  | text    : String → Value
  | missing : Value
  deriving Inhabited, Repr

namespace Value

/-- Project to `Float?`.  `text` is treated as missing. -/
def toFloat? : Value → Option Float
  | .f64 x => some x
  | _      => none

/-- Project to `String?`.  Numerics are stringified. -/
def toString? : Value → Option String
  | .text s => some s
  | .f64 x  => some (toString x)
  | .missing => none

/-- Pretty-print for `toMarkdown`. -/
def render : Value → String
  | .f64 x   => toString x
  | .text s  => s
  | .missing => "?"

end Value

/-- Inferred column type.  `optF64` means the column contains at least
    one NA — used to format header cells like the Haskell `Maybe Double`. -/
inductive ColumnType
  | f64
  | optF64
  | text
  deriving Inhabited, Repr, BEq

namespace ColumnType

def isNumeric : ColumnType → Bool
  | .f64    => true
  | .optF64 => true
  | .text   => false

def label : ColumnType → String
  | .f64    => "Float"
  | .optF64 => "Float?"
  | .text   => "String"

end ColumnType

/-- A row-oriented data frame.  All rows have the same length as
    `columns`; cell `(row, col)` is `rows[row]![col]!`.

    The row-oriented layout makes `derive` and `impute` cheap to express
    (one closure per row) at the cost of a strided access pattern for
    column reductions.  At the scale Ch11 needs (≤ 20k rows × 11 cols)
    this is irrelevant; a column-oriented redesign is left for Option B. -/
structure DataFrame where
  columns : Array String
  schema  : Array ColumnType
  rows    : Array (Array Value)
  deriving Inhabited

namespace DataFrame

/-- `(numRows, numCols)`.  Matches `D.dimensions` in Sabela. -/
def dimensions (df : DataFrame) : Nat × Nat :=
  (df.rows.size, df.columns.size)

def columnNames (df : DataFrame) : Array String := df.columns

/-- Look up a column index by name.  `none` if absent. -/
def colIndex? (df : DataFrame) (name : String) : Option Nat :=
  df.columns.findIdx? (· == name)

def colIndex! (df : DataFrame) (name : String) : Nat :=
  match df.colIndex? name with
  | some i => i
  | none   => panic! s!"DataFrame: unknown column '{name}'"

/-- Project to a single column as `Array Value`. -/
def column (df : DataFrame) (name : String) : Array Value :=
  let i := df.colIndex! name
  df.rows.map (·[i]!)

/-- All `Float` values in a column, skipping NAs / non-numeric. -/
def numericColumn (df : DataFrame) (name : String) : Array Float :=
  (df.column name).filterMap Value.toFloat?

/-- The first `n` rows.  Like `D.take`. -/
def take (n : Nat) (df : DataFrame) : DataFrame :=
  { df with rows := df.rows.extract 0 (min n df.rows.size) }

/-- Keep only the named columns, in the given order. -/
def select (cols : Array String) (df : DataFrame) : DataFrame :=
  let idxs := cols.map (df.colIndex! ·)
  { columns := cols
    schema  := idxs.map (df.schema[·]!)
    rows    := df.rows.map (fun row => idxs.map (row[·]!)) }

/-- Drop the named columns. -/
def exclude (cols : Array String) (df : DataFrame) : DataFrame :=
  -- `Array.contains` is O(n) but `cols` is tiny in practice (≤ tutorial
  -- footprint).  If this turns into a hot path we can swap for a HashSet.
  let keep := df.columns.filter (! cols.contains ·)
  df.select keep

/-- Replace every `missing` in `colName` with `value`.  The column type
    is promoted from `optF64` to `f64`. -/
def impute (colName : String) (value : Float) (df : DataFrame) : DataFrame :=
  let i := df.colIndex! colName
  let rows' := df.rows.map (fun row =>
    match row[i]! with
    | .missing => row.set! i (.f64 value)
    | _        => row)
  let schema' := df.schema.set! i .f64
  { df with rows := rows', schema := schema' }

/-- Mean of a numeric column, ignoring NAs.  Returns `0` if there are
    no non-NA cells (so it's safe to call before `impute`). -/
def meanMaybe (colName : String) (df : DataFrame) : Float :=
  let xs := df.numericColumn colName
  if xs.isEmpty then 0.0 else xs.foldl (·+·) 0.0 / xs.size.toFloat

private def floatMin (a b : Float) : Float := if a < b then a else b
private def floatMax (a b : Float) : Float := if a > b then a else b

/-- Minimum of a numeric column, ignoring NAs. -/
def minColumn (colName : String) (df : DataFrame) : Float :=
  let xs := df.numericColumn colName
  if xs.isEmpty then 0.0 else xs.foldl floatMin xs[0]!

/-- Maximum of a numeric column, ignoring NAs. -/
def maxColumn (colName : String) (df : DataFrame) : Float :=
  let xs := df.numericColumn colName
  if xs.isEmpty then 0.0 else xs.foldl floatMax xs[0]!

/-- Append a new column computed from each row.  The whole `Array Value`
    is passed to the closure so it can reference any other column by
    index/name.  The new column's type is inferred from the first
    non-missing cell. -/
def derive (newCol : String) (f : Array Value → Value) (df : DataFrame) : DataFrame :=
  let newCells := df.rows.map f
  let firstNonMissing : Option Value :=
    newCells.findSome? fun v =>
      match v with
      | .missing => none
      | other    => some other
  let ty : ColumnType :=
    match firstNonMissing with
    | some (.f64 _)  =>
      if newCells.any (fun v => match v with | .missing => true | _ => false)
      then .optF64 else .f64
    | some (.text _) => .text
    | _              => .optF64
  { columns := df.columns.push newCol
    schema  := df.schema.push ty
    rows    := Array.zipWith (fun row v => Array.push row v) df.rows newCells }

/-- `derive` several columns in sequence.  Each subsequent closure sees
    the columns added by earlier closures. -/
def deriveMany (specs : Array (String × (Array Value → Value))) (df : DataFrame) : DataFrame :=
  specs.foldl (fun acc (n, f) => acc.derive n f) df

/-- Helper: derive a `Float` column from a function on the **named**
    columns of each row.  Looks up the name once. -/
def deriveFloat (newCol : String) (deps : Array String)
    (f : Array Float → Float) (df : DataFrame) : DataFrame :=
  let idxs := deps.map (df.colIndex! ·)
  df.derive newCol fun row =>
    let xs := idxs.map (fun i =>
      match row[i]! with
      | .f64 x => x
      | _      => 0.0)
    Value.f64 (f xs)

/-- Min-max normalise every numeric column into `[0, 1]`.  Constant
    columns (`max == min`) are left at `0`.  Mirrors Sabela's
    `normalizeFeatures` helper. -/
def normalizeFeatures (df : DataFrame) : DataFrame :=
  let idxs := df.columns.zip df.schema
    |>.filterMap (fun (n, t) => if t.isNumeric then some n else none)
  idxs.foldl (init := df) fun acc name =>
    let lo := acc.minColumn name
    let hi := acc.maxColumn name
    let span := hi - lo
    let i := acc.colIndex! name
    let rows' := acc.rows.map fun row =>
      match row[i]! with
      | .f64 x =>
        let y := if span == 0.0 then 0.0 else (x - lo) / span
        row.set! i (.f64 y)
      | other => row.set! i other
    { acc with rows := rows' }

/-! ### Random split -/

/-- Simple linear congruential generator — deterministic, good enough
    for shuffling rows in a tutorial.  Don't use for crypto. -/
private partial def lcg (seed : UInt64) : UInt64 :=
  seed * 6364136223846793005 + 1442695040888963407

/-- Fisher-Yates shuffle of `0..n-1` using LCG seeded with `seed`. -/
private def shuffleIndices (n : Nat) (seed : UInt64) : Array Nat := Id.run do
  let mut a : Array Nat := Array.range n
  let mut s := seed
  for i in [0:n] do
    let span := n - i
    let j := i + (if span == 0 then 0 else s.toNat % span)
    s := lcg s
    let tmp := a[i]!
    a := a.set! i a[j]!
    a := a.set! j tmp
  return a

/-- Split into (train, test) with `frac` of rows going to train.  Order
    is randomised with `seed` for reproducibility, matching Sabela's
    `D.randomSplit (mkStdGen seed) frac`. -/
def randomSplit (seed : UInt64) (frac : Float) (df : DataFrame) : DataFrame × DataFrame :=
  let n := df.rows.size
  let idx := shuffleIndices n seed
  let nTrain := (n.toFloat * frac).toUInt32.toNat
  let trainRows := idx.extract 0 nTrain |>.map (df.rows[·]!)
  let testRows  := idx.extract nTrain n |>.map (df.rows[·]!)
  ( { df with rows := trainRows }
  , { df with rows := testRows  } )

/-! ### Pretty-printing -/

/-- GFM-style markdown table with header type annotations like
    Sabela's `total_rooms<br>Double`.  Suitable for `displayMarkdown`
    in a Jupyter cell. -/
def toMarkdown (df : DataFrame) : String := Id.run do
  let header := df.columns.zip df.schema
    |>.map (fun (n, t) => s!"{n}<br>{t.label}")
    |>.toList
    |> String.intercalate " | "
  let sep := List.replicate df.columns.size "---" |> String.intercalate " | "
  let body := df.rows.toList.map fun row =>
    row.toList.map Value.render |> String.intercalate " | "
  let lines := [s!"| {header} |", s!"| {sep} |"]
            ++ body.map (fun r => s!"| {r} |")
  return String.intercalate "\n" lines

/-- Like `toMarkdown` but caps the row count.  Handy in cells where a
    20k-row dump would overwhelm Jupyter. -/
def toMarkdownHead (n : Nat) (df : DataFrame) : String :=
  (df.take n).toMarkdown

/-- Minimal HTML escape — handles the three characters that matter for
    table cell text.  We don't escape quotes because we never inject
    cell text into an attribute. -/
private def htmlEscape (s : String) : String :=
  s.foldl (init := "") fun acc c =>
    match c with
    | '&' => acc ++ "&amp;"
    | '<' => acc ++ "&lt;"
    | '>' => acc ++ "&gt;"
    | _   => acc.push c

private def cellHtml : Value → String
  | .f64 x   => toString x
  | .text s  => htmlEscape s
  | .missing => "<em>NA</em>"

/-- HTML table renderer.  Use this with xeus-lean's `#html` command to
    get a real `<table>` (markdown round-trip via `IO.println` only
    yields raw text in `#eval` cells).  Column headers include the
    inferred type, matching `toMarkdown`. -/
def toHtml (df : DataFrame) : String := Id.run do
  let header := df.columns.zip df.schema
    |>.toList
    |>.map (fun (n, t) => s!"<th>{htmlEscape n}<br><small>{t.label}</small></th>")
    |> String.intercalate ""
  let body := df.rows.toList.map fun row =>
    let tds := row.toList.map (fun v => s!"<td>{cellHtml v}</td>") |> String.intercalate ""
    s!"<tr>{tds}</tr>"
  let bodyStr := String.intercalate "" body
  return s!"<table><thead><tr>{header}</tr></thead><tbody>{bodyStr}</tbody></table>"

/-- Like `toHtml` but caps the row count. -/
def toHtmlHead (n : Nat) (df : DataFrame) : String :=
  (df.take n).toHtml

/-! ### CSV loader (minimal) -/

/-- Parse one CSV line, handling double-quoted strings with embedded
    commas.  Returns the cells as raw strings — `readCsv` converts them
    to `Value` once the schema is known. -/
private def parseCsvLine (line : String) : Array String := Id.run do
  let mut cells : Array String := #[]
  let mut cur : String := ""
  let mut inQuote : Bool := false
  for ch in line.toList do
    if ch == '"' then
      inQuote := !inQuote
    else if ch == ',' && !inQuote then
      cells := cells.push cur
      cur := ""
    else
      cur := cur.push ch
  cells := cells.push cur
  return cells

/-! ### Local float parser

    `Hesper.Training.ParseFloat.parseFloat` works on older Lean — its
    `String.Pos.mk ⟨...⟩` and `String.trim` usages don't survive the
    4.28 API churn.  Re-implement a small parser inline that handles
    California-Housing-style decimals + scientific notation. -/

/-- Parse a non-negative decimal mantissa (no sign, no exponent). -/
private def parseUnsignedMantissa (s : String) : Option Float := Id.run do
  let cs := s.toList
  if cs.isEmpty then return none
  let mut intPart : Nat := 0
  let mut fracPart : Nat := 0
  let mut fracLen : Nat := 0
  let mut seenDot : Bool := false
  let mut seenDigit : Bool := false
  for c in cs do
    if c == '.' then
      if seenDot then return none
      seenDot := true
    else if c.isDigit then
      seenDigit := true
      let d := c.toNat - '0'.toNat
      if seenDot then
        fracPart := fracPart * 10 + d
        fracLen := fracLen + 1
      else
        intPart := intPart * 10 + d
    else
      return none
  if !seenDigit then return none
  let f := intPart.toFloat + fracPart.toFloat / Float.pow 10.0 fracLen.toFloat
  return some f

/-- Parse a signed integer (digits with optional leading `-`/`+`).
    Returns the value as a `Float` so the caller doesn't have to
    juggle a missing `Int.toFloat` in Lean 4.28. -/
private def parseSignedIntFloat (s : String) : Option Float := Id.run do
  if s.isEmpty then return none
  let (sign, rest) : Float × String :=
    if s.startsWith "-" then (-1.0, (s.drop 1).toString)
    else if s.startsWith "+" then (1.0, (s.drop 1).toString)
    else (1.0, s)
  if rest.isEmpty then return none
  let mut n : Nat := 0
  for c in rest.toList do
    if c.isDigit then n := n * 10 + (c.toNat - '0'.toNat)
    else return none
  return some (sign * n.toFloat)

/-- Parse a float in decimal or scientific notation; `none` if the
    string doesn't look like a number. -/
private def parseFloat? (raw : String) : Option Float := Id.run do
  let s := raw
  let (sign, rest) : Float × String :=
    if s.startsWith "-" then (-1.0, (s.drop 1).toString)
    else if s.startsWith "+" then (1.0, (s.drop 1).toString)
    else (1.0, s)
  -- Split on 'e' / 'E'
  let (mantStr, expStr) : String × String :=
    match rest.toList.findIdx? (fun c => c == 'e' || c == 'E') with
    | some i => ((rest.take i).toString, (rest.drop (i + 1)).toString)
    | none   => (rest, "")
  match parseUnsignedMantissa mantStr with
  | none => return none
  | some m =>
    if expStr.isEmpty then
      return some (sign * m)
    else
      match parseSignedIntFloat expStr with
      | none   => return none
      | some e => return some (sign * m * Float.pow 10.0 e)

/-- True iff `s` looks like the start of a number.  Cheap pre-check
    before calling `parseFloat?`. -/
private def looksNumeric (s : String) : Bool :=
  match s.toList with
  | []     => false
  | c :: _ => c.isDigit || c == '-' || c == '+' || c == '.'

/-- Try to parse a cell as `Float`.  Returns `Value.f64` on success,
    `Value.missing` if the cell is empty / "NA" / "NaN" / "null",
    `Value.text` otherwise. -/
private def parseCell (raw : String) : Value :=
  let s := raw.trimAscii.toString
  if s.isEmpty || s == "NA" || s == "NaN" || s == "null" then
    .missing
  else if looksNumeric s then
    match parseFloat? s with
    | some x => .f64 x
    | none   => .text s
  else
    .text s

/-- Infer the type of a column from its parsed cells. -/
private def inferType (col : Array Value) : ColumnType :=
  let hasMissing := col.any (· matches .missing)
  let hasText    := col.any (· matches .text _)
  if hasText then .text
  else if hasMissing then .optF64
  else .f64

/-- Read a CSV file with a header row.  Type inference: a column is
    `text` if any non-empty cell fails to parse as `Float`, `optF64` if
    all non-empty cells are floats but at least one cell is missing,
    `f64` otherwise.  Lines are split on `\n` (also handles `\r\n`). -/
def readCsv (path : System.FilePath) : IO DataFrame := do
  let text ← IO.FS.readFile path
  let lines := text.replace "\r\n" "\n" |>.splitOn "\n" |>.filter (! ·.isEmpty)
  match lines with
  | []           => return { columns := #[], schema := #[], rows := #[] }
  | header :: body =>
    let cols := parseCsvLine header
    let rows := body.toArray.map fun ln =>
      let cells := parseCsvLine ln
      cells.map parseCell
    -- transpose to infer per-column type without holding 2× memory
    let schema := cols.size.fold (init := #[]) fun i _ acc =>
      acc.push (inferType (rows.map (·.getD i .missing)))
    return { columns := cols, schema := schema, rows := rows }

/-! ### Tensor export -/

/-- Flatten the numeric columns of `df` into a row-major `Float32Array`
    suitable for upload as a feature matrix.  Non-numeric and missing
    cells are encoded as `0.0` — call `impute` first to control that
    behaviour.  The returned shape is `(numRows, numNumericCols)`. -/
def toTensor (df : DataFrame) : Array Float × Nat × Nat :=
  let numericIdx := df.schema.zipIdx.filterMap (fun (t, i) =>
    if t.isNumeric then some i else none)
  let nCols := numericIdx.size
  let xs : Array Float := df.rows.flatMap fun row =>
    numericIdx.map fun i =>
      match row[i]! with
      | .f64 x => x
      | _      => 0.0
  (xs, df.rows.size, nCols)

end DataFrame

end Hesper.Data
