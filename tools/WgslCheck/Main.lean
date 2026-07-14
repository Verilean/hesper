import Lean.Data.Json
/-!
# wgsl-check v0 — static write-bounds checker for WGSL compute kernels

Checks the bug class that cost weeks in commit 54a2a60: a dispatch grid
rounded up past the logical element count launches excess threads whose
stores are OOB; WGSL robustness CLAMPS them onto the last element, silently
racing its owner (non-deterministic garbage, no crash).

Tint/Dawn deliberately do not check this: they guarantee the machine is
safe for ALL programs; whether THIS program's grid covers THIS buffer is a
logical-shape question only the dispatch manifest knows.

Usage:
  wgsl-check --manifest manifest.json

Manifest:
  { "dispatches": [ { "kernel": "path/to/k.wgsl", "entry": "main",
      "grid": [gx, gy, gz],
      "bindings": [ { "name": "outBuf", "elems": 553344 } ] } ] }

Method: parse the compute entry (declarations, builtins, let/var, if
(incl. early-return guards), for-loops, stores); evaluate each store index
as a u32 interval under guard refinements; compare against the bound
buffer's element count.

Verdicts per store:
  FAIL — max possible index >= elems (excess threads clamp-write: race)
  WARN — index interval unknown (checker can't bound it; inspect manually)
  ok   — provably in bounds, or syntactically guarded by `idx < N`, N <= elems

v0 scope: integer index arithmetic (+ - * / % << >> & min max clamp select),
1-D indexed stores, literal workgroup sizes. Sound as a bug-finder, not a
verifier: `loop`/`while` bodies and compound mutation degrade to WARN, never
to a silent pass of an unbounded index.
-/
open Lean (Json)

namespace WgslCheck

def U32MAX : Nat := 4294967295

/-- u32 interval (values are non-negative; `hi` capped at U32MAX). -/
structure IVal where
  lo : Nat
  hi : Nat
deriving Repr, Inhabited, BEq

def IVal.top : IVal := ⟨0, U32MAX⟩
def IVal.const (n : Nat) : IVal := ⟨n, n⟩
def IVal.cap (a : IVal) : IVal := ⟨min a.lo U32MAX, min a.hi U32MAX⟩

def IVal.add (a b : IVal) : IVal := IVal.cap ⟨a.lo + b.lo, a.hi + b.hi⟩
def IVal.mul (a b : IVal) : IVal := IVal.cap ⟨a.lo * b.lo, a.hi * b.hi⟩

/-- u32 subtraction wraps; only precise when no operand pair can wrap. -/
def IVal.sub (a b : IVal) : IVal :=
  if b.hi ≤ a.lo then ⟨a.lo - b.hi, a.hi - b.lo⟩ else IVal.top

def IVal.div (a b : IVal) : IVal :=
  if b.lo > 0 then ⟨a.lo / b.hi, a.hi / b.lo⟩ else ⟨0, a.hi⟩

def IVal.mod (a b : IVal) : IVal :=
  if b.lo > 0 then
    if a.hi < b.lo then a else ⟨0, b.hi - 1⟩
  else ⟨0, a.hi⟩

def IVal.band (a b : IVal) : IVal := ⟨0, min a.hi b.hi⟩
def IVal.shl (a : IVal) (c : Nat) : IVal := IVal.cap ⟨a.lo <<< c, a.hi <<< c⟩
def IVal.shr (a : IVal) (c : Nat) : IVal := ⟨a.lo >>> c, a.hi >>> c⟩
def IVal.imin (a b : IVal) : IVal := ⟨min a.lo b.lo, min a.hi b.hi⟩
def IVal.imax (a b : IVal) : IVal := ⟨max a.lo b.lo, max a.hi b.hi⟩
def IVal.hull (a b : IVal) : IVal := ⟨min a.lo b.lo, max a.hi b.hi⟩

-- ============================== tokens ===============================

inductive Tok where
  | ident (s : String)
  | num (n : Nat)
  | flt
  | sym (s : String)
deriving Repr, BEq, Inhabited

structure PTok where
  tok : Tok
  line : Nat
deriving Repr, Inhabited

def twoCharSyms : List String :=
  ["<<", ">>", "<=", ">=", "==", "!=", "&&", "||",
   "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "->"]

partial def tokenize (s : String) : Array PTok := Id.run do
  let cs := s.data.toArray
  let n := cs.size
  let mut out : Array PTok := #[]
  let mut i := 0
  let mut line := 1
  while i < n do
    let c := cs[i]!
    if c = '\n' then line := line + 1; i := i + 1
    else if c.isWhitespace then i := i + 1
    else if c = '/' ∧ i + 1 < n ∧ cs[i+1]! = '/' then
      while i < n ∧ cs[i]! ≠ '\n' do i := i + 1
    else if c = '/' ∧ i + 1 < n ∧ cs[i+1]! = '*' then
      i := i + 2
      while i + 1 < n ∧ ¬(cs[i]! = '*' ∧ cs[i+1]! = '/') do
        if cs[i]! = '\n' then line := line + 1
        i := i + 1
      i := i + 2
    else if c.isAlpha ∨ c = '_' then
      let start := i
      while i < n ∧ (cs[i]!.isAlphanum ∨ cs[i]! = '_') do i := i + 1
      out := out.push ⟨.ident (String.mk (cs.toList.drop start |>.take (i - start))), line⟩
    else if c.isDigit then
      let start := i
      let mut isFloat := false
      if c = '0' ∧ i + 1 < n ∧ (cs[i+1]! = 'x' ∨ cs[i+1]! = 'X') then
        i := i + 2
        while i < n ∧ (cs[i]!.isDigit ∨ ('a' ≤ cs[i]! ∧ cs[i]! ≤ 'f') ∨ ('A' ≤ cs[i]! ∧ cs[i]! ≤ 'F')) do
          i := i + 1
      else
        while i < n ∧ cs[i]!.isDigit do i := i + 1
        if i < n ∧ cs[i]! = '.' then
          isFloat := true; i := i + 1
          while i < n ∧ cs[i]!.isDigit do i := i + 1
        if i < n ∧ (cs[i]! = 'e' ∨ cs[i]! = 'E') then
          isFloat := true; i := i + 1
          if i < n ∧ (cs[i]! = '+' ∨ cs[i]! = '-') then i := i + 1
          while i < n ∧ cs[i]!.isDigit do i := i + 1
      let lit := String.mk (cs.toList.drop start |>.take (i - start))
      -- suffix
      if i < n ∧ (cs[i]! = 'f' ∨ cs[i]! = 'h') then isFloat := true; i := i + 1
      else if i < n ∧ (cs[i]! = 'u' ∨ cs[i]! = 'i') then i := i + 1
      if isFloat then
        out := out.push ⟨.flt, line⟩
      else
        let v := if lit.startsWith "0x" ∨ lit.startsWith "0X" then
          (lit.drop 2).foldl (fun a c =>
            a * 16 + (if c.isDigit then c.toNat - '0'.toNat
                      else if 'a' ≤ c ∧ c ≤ 'f' then c.toNat - 'a'.toNat + 10
                      else c.toNat - 'A'.toNat + 10)) 0
          else lit.toNat!
        out := out.push ⟨.num v, line⟩
    else
      let two := if i + 1 < n then String.mk [c, cs[i+1]!] else ""
      if twoCharSyms.contains two then
        out := out.push ⟨.sym two, line⟩; i := i + 2
      else
        out := out.push ⟨.sym (String.mk [c]), line⟩; i := i + 1
  return out

-- ============================ expressions ============================

inductive Expr where
  | num (n : Nat)
  | flt
  | var (s : String)
  | member (e : Expr) (m : String)
  | index (e : Expr) (i : Expr)
  | call (f : String) (args : List Expr)
  | bin (op : String) (a b : Expr)
  | un (op : String) (a : Expr)
deriving Repr, BEq, Inhabited

def genericCtors : List String :=
  ["vec2", "vec3", "vec4", "array", "bitcast", "ptr", "atomic",
   "mat2x2", "mat2x3", "mat2x4", "mat3x2", "mat3x3", "mat3x4",
   "mat4x2", "mat4x3", "mat4x4"]

def binLevels : List (List String) :=
  [["||"], ["&&"], ["<", "<=", ">", ">=", "==", "!="],
   ["|"], ["^"], ["&"], ["<<", ">>"], ["+", "-"], ["*", "/", "%"]]

structure Prs where
  ts : Array PTok

def Prs.tok (p : Prs) (i : Nat) : Tok :=
  if h : i < p.ts.size then p.ts[i].tok else .sym "<eof>"

def Prs.isSym (p : Prs) (i : Nat) (s : String) : Bool := p.tok i == .sym s

/-- skip a balanced `< ... >` generic argument list starting at `<`. -/
partial def skipGeneric (p : Prs) (i : Nat) : Nat := Id.run do
  let mut d := 0
  let mut j := i
  while j < p.ts.size do
    match p.tok j with
    | .sym "<" => d := d + 1; j := j + 1
    | .sym ">" =>
      d := d - 1; j := j + 1
      if d = 0 then return j
    | .sym ">>" =>
      d := d - 2; j := j + 1
      if d ≤ 0 then return j
    | _ => j := j + 1
  return j

mutual

partial def parsePrimary (p : Prs) (i : Nat) : Option (Expr × Nat) := do
  match p.tok i with
  | .num n => some (.num n, i + 1)
  | .flt => some (.flt, i + 1)
  | .ident s =>
    if genericCtors.contains s ∧ p.isSym (i+1) "<" then
      let j := skipGeneric p (i + 1)
      if p.isSym j "(" then
        let (args, k) ← parseArgs p (j + 1)
        parsePostfix p (.call s args) k
      else none
    else if p.isSym (i+1) "(" then
      let (args, k) ← parseArgs p (i + 2)
      parsePostfix p (.call s args) k
    else
      parsePostfix p (.var s) (i + 1)
  | .sym "(" =>
    let (e, j) ← parseExpr p (i + 1)
    if p.isSym j ")" then parsePostfix p e (j + 1) else none
  | .sym "-" => do let (e, j) ← parsePrimary p (i + 1); some (.un "-" e, j)
  | .sym "!" => do let (e, j) ← parsePrimary p (i + 1); some (.un "!" e, j)
  | .sym "~" => do let (e, j) ← parsePrimary p (i + 1); some (.un "~" e, j)
  | .sym "&" => do let (e, j) ← parsePrimary p (i + 1); some (.un "&" e, j)
  | .sym "*" => do let (e, j) ← parsePrimary p (i + 1); some (.un "*" e, j)
  | _ => none

partial def parseArgs (p : Prs) (i : Nat) : Option (List Expr × Nat) := do
  if p.isSym i ")" then return ([], i + 1)
  let (e, j) ← parseExpr p i
  if p.isSym j "," then
    let (rest, k) ← parseArgs p (j + 1)
    some (e :: rest, k)
  else if p.isSym j ")" then some ([e], j + 1)
  else none

partial def parsePostfix (p : Prs) (e : Expr) (i : Nat) : Option (Expr × Nat) := do
  if p.isSym i "." then
    match p.tok (i+1) with
    | .ident m => parsePostfix p (.member e m) (i + 2)
    | _ => none
  else if p.isSym i "[" then
    let (ix, j) ← parseExpr p (i + 1)
    if p.isSym j "]" then parsePostfix p (.index e ix) (j + 1) else none
  else
    some (e, i)

partial def parseLevel (p : Prs) (lvl : Nat) (i : Nat) : Option (Expr × Nat) := do
  if h : lvl ≥ binLevels.length then parsePrimary p i
  else
    let ops := binLevels[lvl]!
    let mut r ← parseLevel p (lvl + 1) i
    let mut go := true
    while go do
      let (e, j) := r
      match p.tok j with
      | .sym s =>
        if ops.contains s then
          match parseLevel p (lvl + 1) (j + 1) with
          | some (rhs, k) => r := (.bin s e rhs, k)
          | none => go := false
        else go := false
      | _ => go := false
    return r

partial def parseExpr (p : Prs) (i : Nat) : Option (Expr × Nat) :=
  parseLevel p 0 i

end

-- ========================= kernel structure ==========================

structure StorageVar where
  name : String
  group : Nat
  binding : Nat
  readWrite : Bool
deriving Repr, Inhabited

structure Entry where
  name : String
  wg : Nat × Nat × Nat
  builtins : List (String × String)  -- param name → builtin name
  bodyStart : Nat                    -- token index just after `{`
deriving Repr, Inhabited

structure Kernel where
  toks : Array PTok
  storages : List StorageVar
  entries : List Entry
deriving Inhabited

partial def scanKernel (ts : Array PTok) : Kernel := Id.run do
  let p : Prs := ⟨ts⟩
  let mut storages : List StorageVar := []
  let mut entries : List Entry := []
  let mut curGroup := 0
  let mut curBinding := 0
  let mut curWg : Nat × Nat × Nat := (1, 1, 1)
  let mut isCompute := false
  let mut i := 0
  while i < ts.size do
    match p.tok i with
    | .sym "@" =>
      match p.tok (i+1) with
      | .ident "group" =>
        if let .num n := p.tok (i+3) then curGroup := n
        i := i + 5
      | .ident "binding" =>
        if let .num n := p.tok (i+3) then curBinding := n
        i := i + 5
      | .ident "compute" => isCompute := true; i := i + 2
      | .ident "workgroup_size" =>
        let mut dims : Array Nat := #[]
        let mut j := i + 3
        while ¬(p.isSym j ")") ∧ j < ts.size do
          if let .num n := p.tok j then dims := dims.push n
          j := j + 1
        curWg := (dims.getD 0 1, dims.getD 1 1, dims.getD 2 1)
        i := j + 1
      | _ => i := i + 2
    | .ident "var" =>
      -- var<storage[, access]> name : ...
      if p.isSym (i+1) "<" ∧ p.tok (i+2) == .ident "storage" then
        let mut j := i + 3
        let mut rw := false
        if p.isSym j "," then
          if p.tok (j+1) == .ident "read_write" then rw := true
          j := j + 2
        if p.isSym j ">" then
          if let .ident nm := p.tok (j+1) then
            storages := ⟨nm, curGroup, curBinding, rw⟩ :: storages
        i := j + 2
      else i := i + 1
    | .ident "fn" =>
      match p.tok (i+1) with
      | .ident fname =>
        -- params
        let mut j := i + 3
        let mut builtins : List (String × String) := []
        let mut depth := 1
        while depth > 0 ∧ j < ts.size do
          if p.isSym j "(" then depth := depth + 1
          else if p.isSym j ")" then depth := depth - 1
          else if p.isSym j "@" ∧ p.tok (j+1) == .ident "builtin" then
            if let .ident bname := p.tok (j+3) then
              if let .ident pname := p.tok (j+5) then
                builtins := (pname, bname) :: builtins
            j := j + 5
          j := j + 1
        -- body
        if p.isSym j "{" then
          if isCompute then
            entries := ⟨fname, curWg, builtins, j + 1⟩ :: entries
        isCompute := false
        i := j + 1
      | _ => i := i + 1
    | _ => i := i + 1
  return ⟨ts, storages, entries⟩

-- ============================ analysis ===============================

inductive Sev where | fail | warn
deriving Repr, BEq

structure Finding where
  sev : Sev
  line : Nat
  buf : String
  msg : String

structure Ctx where
  p : Prs
  storages : List StorageVar
  elemsOf : String → Option Nat          -- buffer name → element count
  bufVal : String → Nat → Option Nat     -- known buffer contents (from trace)
  grid : Nat × Nat × Nat
  wg : Nat × Nat × Nat
  builtins : List (String × String)

structure WSt where
  env : List (String × IVal)      -- innermost first
  sguards : List (Expr × IVal)    -- known: expr < bound
  finds : Array Finding

def envGet (st : WSt) (x : String) : Option IVal :=
  st.env.lookup x

def envSet (st : WSt) (x : String) (v : IVal) : WSt :=
  { st with env := (x, v) :: st.env.filter (·.1 ≠ x) }

def dim (t : Nat × Nat × Nat) (c : Nat) : Nat :=
  match c with | 0 => t.1 | 1 => t.2.1 | _ => t.2.2

def builtinComp (cx : Ctx) (bname : String) (c : Nat) : IVal :=
  match bname with
  | "global_invocation_id" => ⟨0, dim cx.grid c * dim cx.wg c - 1⟩
  | "local_invocation_id" => ⟨0, dim cx.wg c - 1⟩
  | "workgroup_id" => ⟨0, dim cx.grid c - 1⟩
  | "num_workgroups" => .const (dim cx.grid c)
  | _ => .top

partial def evalE (cx : Ctx) (st : WSt) : Expr → IVal
  | .num n => .const n
  | .flt => .top
  | .var x =>
    match envGet st x with
    | some v => v
    | none =>
      match cx.builtins.lookup x with
      | some "local_invocation_index" =>
        ⟨0, dim cx.wg 0 * dim cx.wg 1 * dim cx.wg 2 - 1⟩
      | some "subgroup_invocation_id" => ⟨0, 127⟩
      | some "subgroup_size" => ⟨1, 128⟩
      | _ => .top
  | .member (.var x) m =>
    let c := if m = "x" then 0 else if m = "y" then 1 else if m = "z" then 2 else 0
    match cx.builtins.lookup x with
    | some b => builtinComp cx b c
    | none => .top
  | .member _ _ => .top
  | .index (.var b) ie =>
    -- a read from a buffer whose contents the manifest supplies (e.g. a
    -- params UBO recorded by the trace) evaluates to a constant
    let iv := evalE cx st ie
    if iv.lo = iv.hi then
      match cx.bufVal b iv.lo with
      | some v => .const v
      | none => .top
    else .top
  | .index _ _ => .top
  | .call f args =>
    let vs := args.map (evalE cx st)
    match f, vs with
    | "min", [a, b] => a.imin b
    | "max", [a, b] => a.imax b
    | "clamp", [_, a, b] => ⟨a.lo, b.hi⟩
    | "select", [a, b, _] => a.hull b
    | "u32", [a] => a
    | "i32", [a] => a
    | "arrayLength", [_] => .top
    | _, _ => .top
  | .bin op a b =>
    let va := evalE cx st a
    let vb := evalE cx st b
    match op with
    | "+" => va.add vb
    | "-" => va.sub vb
    | "*" => va.mul vb
    | "/" => va.div vb
    | "%" => va.mod vb
    | "&" => va.band vb
    | "|" => IVal.cap ⟨max va.lo vb.lo, va.hi + vb.hi⟩
    | "^" => IVal.cap ⟨0, va.hi + vb.hi⟩
    | "<<" => if vb.lo = vb.hi then va.shl vb.lo else .top
    | ">>" => if vb.lo = vb.hi then va.shr vb.lo else ⟨0, va.hi⟩
    | _ => .top
  | .un op a =>
    match op with
    | "&" | "*" => evalE cx st a
    | _ => .top

/-- refine `st` with condition `c` known to be `pos`. -/
partial def applyCond (cx : Ctx) (c : Expr) (pos : Bool) (st : WSt) : WSt :=
  match c, pos with
  | .bin "&&" a b, true => applyCond cx b true (applyCond cx a true st)
  | .bin "||" a b, false => applyCond cx b false (applyCond cx a false st)
  | .un "!" a, _ => applyCond cx a (¬pos) st
  | .bin op a b, _ =>
    -- normalize to a strict/loose upper or lower bound on `a`
    let vb := evalE cx st b
    let va := evalE cx st a
    let refineHi (x : Expr) (bound : Nat) (g : Option (Expr × IVal)) (s : WSt) : WSt :=
      let s := match g with
        | some sg => { s with sguards := sg :: s.sguards }
        | none => s
      match x with
      | .var v =>
        let cur := (envGet s v).getD .top
        envSet s v ⟨cur.lo, min cur.hi bound⟩
      | _ => s
    let refineLo (x : Expr) (bound : Nat) (s : WSt) : WSt :=
      match x with
      | .var v =>
        let cur := (envGet s v).getD .top
        envSet s v ⟨max cur.lo bound, cur.hi⟩
      | _ => s
    match op, pos with
    | "<", true => refineHi a (vb.hi - 1) (some (a, vb)) st
    | "<=", true => refineHi a vb.hi (some (a, ⟨vb.lo + 1, vb.hi + 1⟩)) st
    | ">=", false => refineHi a (vb.hi - 1) (some (a, vb)) st   -- !(a>=b) → a<b
    | ">", false => refineHi a vb.hi (some (a, ⟨vb.lo + 1, vb.hi + 1⟩)) st
    | ">=", true => refineLo a vb.lo st
    | ">", true => refineLo a (vb.lo + 1) st
    | "<", false => refineLo a vb.lo st
    | "<=", false => refineLo a (vb.lo + 1) st
    | "==", true =>
      let st := refineHi a vb.hi none st
      let st := refineLo a vb.lo st
      -- symmetric: refine b by a
      (match b with
       | .var v =>
         let cur := (envGet st v).getD .top
         envSet st v ⟨max cur.lo va.lo, min cur.hi va.hi⟩
       | _ => st)
    | _, _ => st
  | _, _ => st

def checkStore (cx : Ctx) (st : WSt) (bufName : String) (idx : Expr)
    (line : Nat) : WSt :=
  match cx.elemsOf bufName with
  | none => st
  | some elems =>
    -- syntactic guard: some guard `e < B` with e == idx and B ≤ elems
    let guarded := st.sguards.any fun (e, b) => e == idx ∧ b.hi ≤ elems
    if guarded then st
    else
      let iv := evalE cx st idx
      if iv.hi < elems then st
      else if iv.hi < U32MAX then
        { st with finds := st.finds.push ⟨.fail, line, bufName,
          s!"store index max {iv.hi} ≥ {elems} elems — excess threads \
             clamp-write the last element, racing its owner (guard the store)"⟩ }
      else
        { st with finds := st.finds.push ⟨.warn, line, bufName,
          s!"store index cannot be bounded by the checker (elems {elems}) — \
             inspect manually"⟩ }

/-- skip one statement (to `;` at depth 0); returns position after `;`. -/
partial def skipStmt (p : Prs) (i : Nat) : Nat := Id.run do
  let mut j := i
  let mut d := 0
  while j < p.ts.size do
    match p.tok j with
    | .sym "(" | .sym "[" => d := d + 1; j := j + 1
    | .sym ")" | .sym "]" => d := d - 1; j := j + 1
    | .sym "{" => d := d + 1; j := j + 1
    | .sym "}" =>
      if d = 0 then return j   -- statement ran into block end
      d := d - 1; j := j + 1
    | .sym ";" =>
      if d = 0 then return j + 1
      j := j + 1
    | _ => j := j + 1
  return j

/-- does the block starting at `i` (just after `{`) consist solely of
`return;` / `continue;` / `break;`? -/
def isBailBlock (p : Prs) (i : Nat) : Bool :=
  (p.tok i == .ident "return" ∨ p.tok i == .ident "continue" ∨
   p.tok i == .ident "break") ∧ p.isSym (i+1) ";" ∧ p.isSym (i+2) "}"

mutual

/-- walk statements until the matching `}`; returns state and position
after the `}`. -/
partial def walkBlock (cx : Ctx) (st0 : WSt) (i0 : Nat) : WSt × Nat := Id.run do
  let p := cx.p
  let mut st := st0
  let mut i := i0
  while i < p.ts.size do
    match p.tok i with
    | .sym "}" => return (st, i + 1)
    | .ident "let" | .ident "const" =>
      match p.tok (i+1) with
      | .ident x =>
        let mut j := i + 2
        if p.isSym j ":" then
          -- skip type up to `=`
          while ¬(p.isSym j "=") ∧ ¬(p.isSym j ";") ∧ j < p.ts.size do
            j := j + 1
        if p.isSym j "=" then
          match parseExpr p (j + 1) with
          | some (e, k) =>
            st := envSet st x (evalE cx st e)
            i := if p.isSym k ";" then k + 1 else skipStmt p k
          | none => i := skipStmt p (j + 1)
        else i := skipStmt p j
      | _ => i := skipStmt p (i + 1)
    | .ident "var" =>
      match p.tok (i+1) with
      | .ident x =>
        let mut j := i + 2
        if p.isSym j ":" then
          while ¬(p.isSym j "=") ∧ ¬(p.isSym j ";") ∧ j < p.ts.size do
            j := j + 1
        if p.isSym j "=" then
          match parseExpr p (j + 1) with
          | some (e, k) =>
            st := envSet st x (evalE cx st e)
            i := if p.isSym k ";" then k + 1 else skipStmt p k
          | none => i := skipStmt p (j + 1)
        else
          st := envSet st x .top
          i := skipStmt p j
      | _ => i := skipStmt p (i + 1)
    | .ident "if" =>
      match parseCondAfterIf cx st i with
      | some (cond, bodyStart) =>
        if isBailBlock p bodyStart then
          -- early-exit guard: continue with ¬cond
          st := applyCond cx cond false st
          i := bodyStart + 3
          -- optional else after a bail block: walk it under cond
          if p.tok i == .ident "else" ∧ p.isSym (i+1) "{" then
            let stT := applyCond cx cond true { st with finds := st.finds }
            let (st', k) := walkBlock cx stT (i + 2)
            st := { st with finds := st'.finds }
            i := k
        else
          let stT := applyCond cx cond true st
          let (st', k) := walkBlock cx stT bodyStart
          st := { st with finds := st'.finds }
          i := k
          -- else / else if
          if p.tok i == .ident "else" then
            if p.isSym (i+1) "{" then
              let stF := applyCond cx cond false st
              let (st'', k2) := walkBlock cx stF (i + 2)
              st := { st with finds := st''.finds }
              i := k2
            else
              -- else if …: walk as a fresh statement under ¬cond (findings only)
              let stF := applyCond cx cond false st
              let (st'', k2) := walkStmtAt cx stF (i + 1)
              st := { st with finds := st''.finds }
              i := k2
      | none => i := skipStmt p (i + 1)
    | .ident "for" =>
      -- for (var x = a; x < b; …) { body }
      let mut j := i + 2
      let mut loopVar : Option (String × IVal) := none
      if p.tok j == .ident "var" then
        if let .ident x := p.tok (j+1) then
          let mut k := j + 2
          if p.isSym k ":" then
            while ¬(p.isSym k "=") ∧ k < p.ts.size do k := k + 1
          if p.isSym k "=" then
            if let some (einit, k2) := parseExpr p (k + 1) then
              let vinit := evalE cx st einit
              if p.isSym k2 ";" then
                if let some (econd, _) := parseExpr p (k2 + 1) then
                  match econd with
                  | .bin "<" (.var y) b =>
                    if y = x then
                      let vb := evalE cx st b
                      loopVar := some (x, ⟨vinit.lo, max vinit.hi (vb.hi - 1)⟩)
                  | .bin "<=" (.var y) b =>
                    if y = x then
                      let vb := evalE cx st b
                      loopVar := some (x, ⟨vinit.lo, max vinit.hi vb.hi⟩)
                  | _ => pure ()
      -- find the loop body `{`
      let mut k := i + 1
      let mut d := 0
      while k < p.ts.size do
        match p.tok k with
        | .sym "(" => d := d + 1; k := k + 1
        | .sym ")" =>
          d := d - 1; k := k + 1
          if d = 0 then break
        | _ => k := k + 1
      if p.isSym k "{" then
        let stB := match loopVar with
          | some (x, v) => envSet st x v
          | none => st
        let (st', k2) := walkBlock cx stB (k + 1)
        st := { st with finds := st'.finds }
        i := k2
      else i := skipStmt p k
    | .ident "loop" | .ident "while" =>
      -- walk body for stores; env entering is unchanged (vars mutated inside
      -- are handled conservatively by assignment → eval-under-current-env)
      let mut k := i + 1
      while ¬(p.isSym k "{") ∧ ¬(p.isSym k ";") ∧ k < p.ts.size do k := k + 1
      if p.isSym k "{" then
        let (st', k2) := walkBlock cx st (k + 1)
        st := { st with finds := st'.finds }
        i := k2
      else i := skipStmt p k
    | .sym "{" =>
      let (st', k) := walkBlock cx st (i + 1)
      st := { st with finds := st'.finds }
      i := k
    | .ident "return" | .ident "continue" | .ident "break"
    | .ident "continuing" | .ident "discard" =>
      i := skipStmt p (i + 1)
    | .ident x =>
      -- store `buf[e] … = …;` or assignment `x = e;` or call `f(…);`
      if p.isSym (i+1) "[" ∧ cx.storages.any (fun s => s.name = x ∧ s.readWrite) then
        match parseExpr p (i + 2) with
        | some (idx, j) =>
          if p.isSym j "]" then
            -- must be an assignment (not a read inside a larger stmt): the
            -- statement STARTED with the buffer name, so it is a store.
            let line := if h : i < p.ts.size then p.ts[i].line else 0
            st := checkStore cx st x idx line
          i := skipStmt p i
        | none => i := skipStmt p i
      else if p.isSym (i+1) "=" then
        match parseExpr p (i + 2) with
        | some (e, k) =>
          st := envSet st x (evalE cx st e)
          i := if p.isSym k ";" then k + 1 else skipStmt p k
        | none => i := skipStmt p (i + 2)
      else if p.isSym (i+1) "+=" ∨ p.isSym (i+1) "-=" ∨ p.isSym (i+1) "*=" ∨
              p.isSym (i+1) "/=" ∨ p.isSym (i+1) "%=" then
        st := envSet st x .top
        i := skipStmt p (i + 2)
      else
        i := skipStmt p i
    | _ => i := skipStmt p i
  return (st, i)

/-- walk exactly one statement at `i` (used for `else if`). -/
partial def walkStmtAt (cx : Ctx) (st : WSt) (i : Nat) : WSt × Nat :=
  let p := cx.p
  if p.tok i == .ident "if" then
    match parseCondAfterIf cx st i with
    | some (cond, bodyStart) =>
      let stT := applyCond cx cond true st
      let (st', k) := walkBlock cx stT bodyStart
      let st := { st with finds := st'.finds }
      if p.tok k == .ident "else" then
        if p.isSym (k+1) "{" then
          let (st'', k2) := walkBlock cx st (k + 2)
          ({ st with finds := st''.finds }, k2)
        else
          let (st'', k2) := walkStmtAt cx st (k + 1)
          ({ st with finds := st''.finds }, k2)
      else (st, k)
    | none => (st, skipStmt p (i + 1))
  else (st, skipStmt p i)

partial def parseCondAfterIf (cx : Ctx) (_st : WSt) (i : Nat) :
    Option (Expr × Nat) := do
  let p := cx.p
  guard (p.isSym (i+1) "(")
  let (cond, j) ← parseExpr p (i + 2)
  guard (p.isSym j ")")
  guard (p.isSym (j+1) "{")
  return (cond, j + 2)

end

-- ============================== driver ===============================

structure Binding where
  name : Option String
  group : Option Nat
  bindingNo : Option Nat
  elems : Nat
  values : List (Option Nat)   -- known contents (index → value), [] if unknown

structure Dispatch where
  kernel : String
  entry : String
  grid : Nat × Nat × Nat
  bindings : List Binding

def jNat (j : Json) : Option Nat := j.getNat?.toOption
def jStr (j : Json) : Option String := j.getStr?.toOption
def jArr (j : Json) : Option (Array Json) := j.getArr?.toOption
def jGet (j : Json) (k : String) : Option Json := (j.getObjVal? k).toOption

def parseManifest (j : Json) : Option (List Dispatch) := do
  let ds ← jArr (← jGet j "dispatches")
  ds.toList.mapM fun d => do
    let kernel ← jStr (← jGet d "kernel")
    let entry := (jGet d "entry" >>= jStr).getD "main"
    let g ← jArr (← jGet d "grid")
    let grid := ((g.getD 0 (Json.num 1) |> jNat).getD 1,
                 (g.getD 1 (Json.num 1) |> jNat).getD 1,
                 (g.getD 2 (Json.num 1) |> jNat).getD 1)
    let bs ← jArr (← jGet d "bindings")
    let bindings ← bs.toList.mapM fun b => do
      let elems ← jNat (← jGet b "elems")
      let values := match jGet b "values" >>= jArr with
        | some vs => vs.toList.map jNat
        | none => []
      return { name := jGet b "name" >>= jStr,
               group := jGet b "group" >>= jNat,
               bindingNo := jGet b "binding" >>= jNat,
               elems, values : Binding }
    return { kernel, entry, grid, bindings : Dispatch }

def findBinding (storages : List StorageVar) (bs : List Binding)
    (bufName : String) : Option Binding := Id.run do
  for b in bs do
    match b.name with
    | some n => if n = bufName then return some b
    | none =>
      if let (some g, some bd) := (b.group, b.bindingNo) then
        if storages.any (fun s => s.name = bufName ∧ s.group = g ∧ s.binding = bd) then
          return some b
  return none

def resolveElems (storages : List StorageVar) (bs : List Binding)
    (bufName : String) : Option Nat :=
  (findBinding storages bs bufName).map (·.elems)

def resolveVal (storages : List StorageVar) (bs : List Binding)
    (bufName : String) (idx : Nat) : Option Nat := do
  let b ← findBinding storages bs bufName
  (b.values.getD idx none)

def sevStr : Sev → String
  | .fail => "FAIL"
  | .warn => "WARN"

def main (args : List String) : IO UInt32 := do
  let manifestPath ← match args with
    | ["--manifest", p] => pure p
    | [p] => pure p
    | _ =>
      IO.eprintln "usage: wgsl-check --manifest manifest.json"
      return 2
  let mtext ← IO.FS.readFile manifestPath
  let some dispatches := (Json.parse mtext).toOption >>= parseManifest
    | IO.eprintln "wgsl-check: cannot parse manifest"; return 2
  let mdir := System.FilePath.parent manifestPath |>.getD "."
  let mut nFail := 0
  let mut nWarn := 0
  for d in dispatches do
    let kpath := if System.FilePath.isAbsolute d.kernel then
        System.FilePath.mk d.kernel else mdir / d.kernel
    let src ← IO.FS.readFile kpath
    let k := scanKernel (tokenize src)
    let some e := k.entries.find? (·.name = d.entry)
      | IO.println s!"{d.kernel}: entry `{d.entry}` not found — skipped"
        nWarn := nWarn + 1
        continue
    let cx : Ctx := {
      p := ⟨k.toks⟩, storages := k.storages,
      elemsOf := resolveElems k.storages d.bindings,
      bufVal := resolveVal k.storages d.bindings,
      grid := d.grid, wg := e.wg, builtins := e.builtins }
    let st0 : WSt := { env := [], sguards := [], finds := #[] }
    let (st, _) := walkBlock cx st0 e.bodyStart
    let (gx, gy, gz) := d.grid
    let (wx, wy, wz) := e.wg
    if st.finds.isEmpty then
      IO.println s!"{d.kernel} @{d.entry} grid {gx}x{gy}x{gz} wg {wx}x{wy}x{wz}: ok"
    else
      for f in st.finds do
        IO.println s!"{d.kernel}:{f.line} [{sevStr f.sev}] buffer `{f.buf}` \
                      (grid {gx}x{gy}x{gz} wg {wx}x{wy}x{wz}): {f.msg}"
        if f.sev == .fail then nFail := nFail + 1 else nWarn := nWarn + 1
  IO.println s!"wgsl-check: {nFail} FAIL, {nWarn} WARN"
  return (if nFail > 0 then 1 else 0)

end WgslCheck

def main (args : List String) : IO UInt32 := WgslCheck.main args
