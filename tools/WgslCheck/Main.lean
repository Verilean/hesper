import Lean.Data.Json
/-!
# wgsl-check — static bounds checker for WGSL **and MSL-subset** compute kernels

M-Metal Stage 3 added an MSL front end (see the "MSL front end" section):
hand-written Metal kernels get the same interval/taint/guard OOB analyses as
the generated WGSL ones. MSL dispatches are routed by `"lang": "msl"` (or a
`.metal`/`.msl` kernel path) and REQUIRE `"wg": [x,y,z]` in the manifest
(workgroup size is not in MSL source). One-command production gate:
`scripts/msl_check.sh` (extracts the hand kernels from native/metal_replace.mm,
substitutes production dims, runs fixtures + kernels; fixtures live in
specs/mslcheck/).

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

/-- u32 interval. `t` marks a bound TAINTED by unbounded memory reads
(atomicLoad, unknown buffer contents): such an over-approximate hi must
demote a violation to WARN — the real invariant is data-dependent and
outside the interval domain. Thread-id/params-derived bounds stay clean. -/
structure IVal where
  lo : Nat
  hi : Nat
  t : Bool := false
deriving Repr, Inhabited, BEq

def IVal.top : IVal := ⟨0, U32MAX, false⟩
def IVal.unk : IVal := ⟨0, U32MAX, true⟩
def IVal.const (n : Nat) : IVal := ⟨n, n, false⟩
def IVal.cap (a : IVal) : IVal := ⟨min a.lo U32MAX, min a.hi U32MAX, a.t⟩

def IVal.add (a b : IVal) : IVal := IVal.cap ⟨a.lo + b.lo, a.hi + b.hi, a.t ∨ b.t⟩
def IVal.mul (a b : IVal) : IVal := IVal.cap ⟨a.lo * b.lo, a.hi * b.hi, a.t ∨ b.t⟩

/-- u32 subtraction wraps; only precise when no operand pair can wrap. -/
def IVal.sub (a b : IVal) : IVal :=
  if b.hi ≤ a.lo then ⟨a.lo - b.hi, a.hi - b.lo, a.t ∨ b.t⟩
  else ⟨0, U32MAX, a.t ∨ b.t⟩

def IVal.div (a b : IVal) : IVal :=
  if b.lo > 0 then ⟨a.lo / b.hi, a.hi / b.lo, a.t ∨ b.t⟩ else ⟨0, a.hi, a.t⟩

def IVal.mod (a b : IVal) : IVal :=
  if b.lo > 0 then
    if a.hi < b.lo then a else ⟨0, b.hi - 1, b.t⟩   -- bound comes from b
  else ⟨0, a.hi, a.t⟩

def IVal.band (a b : IVal) : IVal :=
  ⟨0, min a.hi b.hi, if a.hi ≤ b.hi then a.t else b.t⟩
def IVal.shl (a : IVal) (c : Nat) : IVal := IVal.cap ⟨a.lo <<< c, a.hi <<< c, a.t⟩
def IVal.shr (a : IVal) (c : Nat) : IVal := ⟨a.lo >>> c, a.hi >>> c, a.t⟩
def IVal.imin (a b : IVal) : IVal :=
  ⟨min a.lo b.lo, min a.hi b.hi, if a.hi ≤ b.hi then a.t else b.t⟩
def IVal.imax (a b : IVal) : IVal := ⟨max a.lo b.lo, max a.hi b.hi, a.t ∨ b.t⟩
def IVal.hull (a b : IVal) : IVal := ⟨min a.lo b.lo, max a.hi b.hi, a.t ∨ b.t⟩

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
   "mat4x2", "mat4x3", "mat4x4",
   -- MSL front end: templated calls that appear in value positions
   "as_type", "make_filled_simdgroup_matrix"]

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
  elemBytes : Nat := 0   -- bytes per indexed element (0 = unknown)
deriving Repr, Inhabited

def scalarBytes : String → Nat
  | "f32" | "u32" | "i32" => 4
  | "f16" => 2
  | _ => 0

/-- element size of the declared type starting at `i` (after the `:`).
Handles `array<T[, N]>`, `vecN<T>`, scalars, `atomic<u32>`. -/
def typeElemBytes (p : Prs) (i : Nat) : Nat :=
  let base (j : Nat) : Nat :=
    match p.tok j with
    | .ident "atomic" => 4
    | .ident v =>
      if v = "vec2" ∨ v = "vec3" ∨ v = "vec4" then
        let n := if v = "vec2" then 2 else if v = "vec3" then 3 else 4
        match p.tok (j+2) with
        | .ident s => n * scalarBytes s
        | _ => 0
      else scalarBytes v
    | _ => 0
  match p.tok i with
  | .ident "array" => if p.isSym (i+1) "<" then base (i+2) else 0
  | _ => base i

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
      -- var<storage[, access]> name : T   |   var<uniform> name : T
      if p.isSym (i+1) "<" ∧
         (p.tok (i+2) == .ident "storage" ∨ p.tok (i+2) == .ident "uniform") then
        let mut j := i + 3
        let mut rw := false
        if p.isSym j "," then
          if p.tok (j+1) == .ident "read_write" then rw := true
          j := j + 2
        if p.isSym j ">" then
          if let .ident nm := p.tok (j+1) then
            let eb := if p.isSym (j+2) ":" then typeElemBytes p (j+3) else 0
            storages := ⟨nm, curGroup, curBinding, rw, eb⟩ :: storages
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

def IVal.inter (a b : IVal) : IVal :=
  ⟨max a.lo b.lo, min a.hi b.hi, if a.hi ≤ b.hi then a.t else b.t⟩

/-- `let` bindings stay symbolic (re-evaluated at USE site, so guard
refinements between definition and use take effect); `var`/loop bindings are
eager intervals. -/
inductive EnvVal where
  | expr (e : Expr)
  | ival (v : IVal)

structure WSt where
  env : List (String × EnvVal)    -- innermost first
  refin : List (String × IVal)    -- guard refinements: var name or "vec.comp"
  sguards : List (Expr × IVal)    -- known: expr < bound
  finds : Array Finding

def envSetLet (st : WSt) (x : String) (e : Expr) : WSt :=
  { st with env := (x, .expr e) :: st.env.filter (·.1 ≠ x),
            refin := st.refin.filter (·.1 ≠ x) }

def envSetVar (st : WSt) (x : String) (v : IVal) : WSt :=
  { st with env := (x, .ival v) :: st.env.filter (·.1 ≠ x),
            refin := st.refin.filter (·.1 ≠ x) }

def refinOf (st : WSt) (key : String) : IVal :=
  (st.refin.lookup key).getD .top

def refinAdd (st : WSt) (key : String) (v : IVal) : WSt :=
  { st with refin := (key, (refinOf st key).inter v) :: st.refin.filter (·.1 ≠ key) }

/-- refinement key for guard subjects: plain vars and builtin components. -/
def refinKey : Expr → Option String
  | .var v => some v
  | .member (.var v) m => some (v ++ "." ++ m)
  | _ => none

def dim (t : Nat × Nat × Nat) (c : Nat) : Nat :=
  match c with | 0 => t.1 | 1 => t.2.1 | _ => t.2.2

def builtinComp (cx : Ctx) (bname : String) (c : Nat) : IVal :=
  match bname with
  | "global_invocation_id" => ⟨0, dim cx.grid c * dim cx.wg c - 1, false⟩
  | "local_invocation_id" => ⟨0, dim cx.wg c - 1, false⟩
  | "workgroup_id" => ⟨0, dim cx.grid c - 1, false⟩
  | "num_workgroups" => .const (dim cx.grid c)
  | _ => .unk

mutual

partial def evalE (cx : Ctx) (st : WSt) : Expr → IVal
  | .num n => .const n
  | .flt => .unk
  | .var x =>
    let base := match st.env.lookup x with
      | some (.ival v) => v
      | some (.expr e) => evalE cx st e
      | none =>
        match cx.builtins.lookup x with
        | some "local_invocation_index" =>
          ⟨0, dim cx.wg 0 * dim cx.wg 1 * dim cx.wg 2 - 1, false⟩
        | some "subgroup_invocation_id" => ⟨0, 127, false⟩
        | some "subgroup_size" => ⟨1, 128, false⟩
        -- MSL scalar-declared builtins (`uint gid [[thread_position_in_grid]]`):
        -- component 0 of the corresponding WGSL builtin
        | some b => builtinComp cx b 0
        | _ => .unk
    base.inter (refinOf st x)
  | .member (.var x) m =>
    let c := if m = "x" then 0 else if m = "y" then 1 else if m = "z" then 2 else 0
    let base := match cx.builtins.lookup x with
      | some b => builtinComp cx b c
      | none => .unk
    base.inter (refinOf st (x ++ "." ++ m))
  | .member _ _ => .unk
  | .index (.var b) ie =>
    -- a read from a buffer whose contents the manifest supplies (e.g. a
    -- params UBO recorded by the trace) evaluates to a constant
    let iv := evalE cx st ie
    if iv.lo = iv.hi then
      match cx.bufVal b iv.lo with
      | some v => .const v
      | none => .unk
    else .unk
  | .index _ _ => .unk
  | .call f args =>
    match f, args with
    -- WGSL select(f, t, cond): evaluate each branch under the matching
    -- condition refinement (the MSL ternary rewrite lands here; enables
    -- `k0 < 16 ? k0 : k0 - 16` to bound instead of wrapping to top)
    | "select", [fe, te, ce] =>
      (evalE cx (applyCond cx ce false st) fe).hull
        (evalE cx (applyCond cx ce true st) te)
    | _, _ =>
    let vs := args.map (evalE cx st)
    match f, vs with
    | "min", [a, b] => a.imin b
    | "max", [a, b] => a.imax b
    | "clamp", [_, a, b] => ⟨a.lo, b.hi, a.t ∨ b.t⟩
    | "u32", [a] => a
    | "i32", [a] => a
    -- MSL functional casts: value-preserving for interval purposes
    | "uint", [a] => a
    | "int", [a] => a
    | "ushort", [a] => a
    | "short", [a] => a
    | "arrayLength", [_] => .unk
    | _, _ => .unk
  | .bin op a b =>
    let va := evalE cx st a
    let vb := evalE cx st b
    match op with
    | "+" => va.add vb
    | "-" =>
      -- mod idiom: `x - (x / c) * c` = `x % c` (Tint/codegen emit this form)
      match b with
      | .bin "*" (.bin "/" a' c) c' =>
        if a' == a ∧ c' == c then va.mod (evalE cx st c) else va.sub vb
      | .bin "*" c' (.bin "/" a' c) =>
        if a' == a ∧ c' == c then va.mod (evalE cx st c) else va.sub vb
      | _ => va.sub vb
    | "*" => va.mul vb
    | "/" => va.div vb
    | "%" => va.mod vb
    | "&" => va.band vb
    | "|" => IVal.cap ⟨max va.lo vb.lo, va.hi + vb.hi, va.t ∨ vb.t⟩
    | "^" => IVal.cap ⟨0, va.hi + vb.hi, va.t ∨ vb.t⟩
    | "<<" => if vb.lo = vb.hi then va.shl vb.lo else .unk
    | ">>" => if vb.lo = vb.hi then va.shr vb.lo else ⟨0, va.hi, va.t⟩
    | _ => .unk
  | .un op a =>
    match op with
    | "&" | "*" => evalE cx st a
    | _ => .unk

/-- refine `st` with condition `c` known to be `pos`. -/
partial def applyCond (cx : Ctx) (c : Expr) (pos : Bool) (st : WSt) : WSt :=
  match c, pos with
  | .bin "&&" a b, true => applyCond cx b true (applyCond cx a true st)
  | .bin "||" a b, false => applyCond cx b false (applyCond cx a false st)
  | .un "!" a, _ => applyCond cx a (¬pos) st
  -- a condition that is a let-bound boolean (`const bool ok = x < N; if (ok)`)
  -- resolves to its defining expression — common in the hand-MSL kernels
  | .var x, _ =>
    (match st.env.lookup x with
     | some (.expr e) => applyCond cx e pos st
     | _ => st)
  | .bin op a b, _ =>
    -- normalize to a strict/loose upper or lower bound on `a`
    let vb := evalE cx st b
    let va := evalE cx st a
    let refineHi (x : Expr) (bound : Nat) (g : Option (Expr × IVal)) (s : WSt) : WSt :=
      let s := match g with
        | some sg => { s with sguards := sg :: s.sguards }
        | none => s
      match refinKey x with
      | some k => refinAdd s k ⟨0, bound, false⟩
      | none =>
        -- `v + c < bound` ⟹ `v < bound - c` (MSL ternary guards on k0+1u etc.)
        match x with
        | .bin "+" (.var v) (.num n) | .bin "+" (.num n) (.var v) =>
          if bound ≥ n then refinAdd s v ⟨0, bound - n, false⟩ else s
        | _ => s
    let refineLo (x : Expr) (bound : Nat) (s : WSt) : WSt :=
      match refinKey x with
      | some k => refinAdd s k ⟨bound, U32MAX, false⟩
      | none =>
        match x with
        | .bin "+" (.var v) (.num n) | .bin "+" (.num n) (.var v) =>
          refinAdd s v ⟨bound - n, U32MAX, false⟩
        | _ => s
    match op, pos with
    | "<", true => refineHi a (vb.hi - 1) (some (a, vb)) st
    | "<=", true => refineHi a vb.hi (some (a, ⟨vb.lo + 1, vb.hi + 1, vb.t⟩)) st
    | ">=", false => refineHi a (vb.hi - 1) (some (a, vb)) st   -- !(a>=b) → a<b
    | ">", false => refineHi a vb.hi (some (a, ⟨vb.lo + 1, vb.hi + 1, vb.t⟩)) st
    | ">=", true => refineLo a vb.lo st
    | ">", true => refineLo a (vb.lo + 1) st
    | "<", false => refineLo a vb.lo st
    | "<=", false => refineLo a (vb.lo + 1) st
    | "==", true =>
      let st := refineHi a vb.hi none st
      let st := refineLo a vb.lo st
      match refinKey b with
      | some k => refinAdd st k va
      | none => st
    | _, _ => st
  | _, _ => st

end

/-- decide a condition statically when the intervals are decisive
(kills template-disabled branches like `if (0u == 1u)`). -/
partial def condTruth (cx : Ctx) (st : WSt) : Expr → Option Bool
  | .un "!" a => (condTruth cx st a).map (!·)
  | .bin "&&" a b =>
    match condTruth cx st a, condTruth cx st b with
    | some false, _ | _, some false => some false
    | some true, some true => some true
    | _, _ => none
  | .bin "||" a b =>
    match condTruth cx st a, condTruth cx st b with
    | some true, _ | _, some true => some true
    | some false, some false => some false
    | _, _ => none
  | .bin op a b =>
    let va := evalE cx st a
    let vb := evalE cx st b
    match op with
    | "<" => if va.hi < vb.lo then some true
             else if va.lo ≥ vb.hi then some false else none
    | "<=" => if va.hi ≤ vb.lo then some true
              else if va.lo > vb.hi then some false else none
    | ">" => if va.lo > vb.hi then some true
             else if va.hi ≤ vb.lo then some false else none
    | ">=" => if va.lo ≥ vb.hi then some true
              else if va.hi < vb.lo then some false else none
    | "==" => if va.lo = va.hi ∧ vb.lo = vb.hi ∧ va.lo = vb.lo then some true
              else if va.hi < vb.lo ∨ vb.hi < va.lo then some false else none
    | "!=" => if va.hi < vb.lo ∨ vb.hi < va.lo then some true
              else if va.lo = va.hi ∧ vb.lo = vb.hi ∧ va.lo = vb.lo then some false
              else none
    | _ => none
  | _ => none

/-- skip a `{ … }` block starting just after its `{`; returns position after
the matching `}`. -/
partial def skipBlock (p : Prs) (i : Nat) : Nat := Id.run do
  let mut d := 1
  let mut j := i
  while j < p.ts.size do
    match p.tok j with
    | .sym "{" => d := d + 1; j := j + 1
    | .sym "}" =>
      d := d - 1; j := j + 1
      if d = 0 then return j
    | _ => j := j + 1
  return j

/-- v1.1: OOB READ detection. Reads clamp (old Tint) or predicate-to-zero
(newer Tint robustness) — an OOB read is therefore a CROSS-COMPILER VALUE
DIVERGENCE, silent on both. Walk an expression tree and bounds-check every
index into a known storage buffer. -/
partial def checkReads (cx : Ctx) (st : WSt) (line : Nat) : Expr → WSt := fun e =>
  match e with
  | .index (.var b) ie =>
    let st := checkReads cx st line ie
    match cx.elemsOf b with
    | none => st
    | some elems =>
      let guarded := st.sguards.any fun (g, bnd) => g == ie ∧ bnd.hi ≤ elems
      if guarded then st
      else
        let iv := evalE cx st ie
        if iv.hi < elems then st
        else if ¬iv.t then
          { st with finds := st.finds.push ⟨.fail, line, b,
            s!"READ index max {iv.hi} ≥ {elems} elems — OOB reads clamp on \
               old Tint but predicate-to-zero on newer robustness: silent \
               cross-compiler value divergence"⟩ }
        else
          { st with finds := st.finds.push ⟨.warn, line, b,
            s!"READ index cannot be bounded (elems {elems}; data-dependent) \
               — potential cross-compiler OOB-read divergence"⟩ }
  | .index a ie => checkReads cx (checkReads cx st line a) line ie
  | .bin _ a b => checkReads cx (checkReads cx st line a) line b
  | .un _ a => checkReads cx st line a
  | .member a _ => checkReads cx st line a
  | .call f args =>
    -- a storage buffer passed BARE to an opaque (non-inlined) helper means
    -- reads inside that helper are invisible to this analysis — surface it
    let st := args.foldl (init := st) fun acc x =>
      match x with
      | .var b =>
        if cx.storages.any (·.name = b) then
          { acc with finds := acc.finds.push ⟨.warn, line, b,
            s!"buffer passed to helper `{f}` — reads inside it are UNCHECKED \
               (inline the helper or verify manually)"⟩ }
        else acc
      | _ => acc
    args.foldl (fun acc x => checkReads cx acc line x) st
  | _ => st

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
      else if ¬iv.t then
        -- bound derived from thread-id/params arithmetic: trustworthy
        { st with finds := st.finds.push ⟨.fail, line, bufName,
          s!"store index max {iv.hi} ≥ {elems} elems — excess threads \
             clamp-write the last element, racing its owner (guard the store)"⟩ }
      else
        -- tainted by unbounded memory reads: the real invariant is
        -- data-dependent and outside the interval domain
        { st with finds := st.finds.push ⟨.warn, line, bufName,
          s!"store index cannot be bounded (elems {elems}; data-dependent) — \
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
            let line := if h : j < p.ts.size then p.ts[j].line else 0
            st := checkReads cx st line e
            st := envSetLet st x e
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
            let line := if h : j < p.ts.size then p.ts[j].line else 0
            st := checkReads cx st line e
            st := envSetVar st x (evalE cx st e)
            i := if p.isSym k ";" then k + 1 else skipStmt p k
          | none => i := skipStmt p (j + 1)
        else
          st := envSetVar st x .top
          i := skipStmt p j
      | _ => i := skipStmt p (i + 1)
    | .ident "if" =>
      match parseCondAfterIf cx st i with
      | some (cond, bodyStart) =>
        if condTruth cx st cond == some false then
          -- template-disabled branch: dead code, skip; walk any else as live
          let k := skipBlock p bodyStart
          if p.tok k == .ident "else" ∧ p.isSym (k+1) "{" then
            let (st', k2) := walkBlock cx st (k + 2)
            st := { st with finds := st'.finds }
            i := k2
          else if p.tok k == .ident "else" then
            let (st', k2) := walkStmtAt cx st (k + 1)
            st := { st with finds := st'.finds }
            i := k2
          else
            i := k
        else if condTruth cx st cond == some true then
          if isBailBlock p bodyStart then
            -- unconditional bail: the rest of this block is dead
            let mut k := skipBlock p bodyStart
            while ¬(p.isSym k "}") ∧ k < p.ts.size do k := skipStmt p k
            return (st, k + 1)
          else
            let (st', k) := walkBlock cx st bodyStart
            st := { st with finds := st'.finds }
            i := if p.tok k == .ident "else" then
                   (if p.isSym (k+1) "{" then skipBlock p (k+2) else skipStmt p (k+1))
                 else k
        else if isBailBlock p bodyStart then
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
                  -- inside the body the condition has just been checked, so
                  -- the loop variable is bounded by the condition alone
                  match econd with
                  | .bin "<" (.var y) b =>
                    if y = x then
                      let vb := evalE cx st b
                      loopVar := some (x, ⟨vinit.lo, vb.hi - 1, vb.t⟩)
                  | .bin "<=" (.var y) b =>
                    if y = x then
                      let vb := evalE cx st b
                      loopVar := some (x, ⟨vinit.lo, vb.hi, vb.t⟩)
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
          | some (x, v) => envSetVar st x v
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
    | .ident "simdgroup_store" | .ident "simdgroup_load" =>
      -- MSL 8×8 fragment memory ops: footprint = base + 7·stride + 7 elements.
      -- `simdgroup_store(frag, &buf[e], stride)` — the R32 WMMA-tail heap-stomp
      -- class lived exactly here on the WGSL side.
      let isStore := p.tok i == .ident "simdgroup_store"
      let line := if h : i < p.ts.size then p.ts[i].line else 0
      if p.isSym (i+1) "(" ∧ p.isSym (i+3) "," ∧ p.isSym (i+4) "&" then
        match p.tok (i+5) with
        | .ident buf =>
          if p.isSym (i+6) "[" then
            match parseExpr p (i + 7) with
            | some (e, j) =>
              if p.isSym j "]" ∧ p.isSym (j+1) "," then
                match parseExpr p (j + 2) with
                | some (strideE, _) =>
                  let footMax : Expr :=
                    .bin "+" e (.bin "+" (.bin "*" (.num 7) strideE) (.num 7))
                  if cx.storages.any (·.name = buf) then
                    if isStore then
                      st := checkStore cx st buf footMax line
                    else
                      st := checkReads cx st line (.index (.var buf) footMax)
                  -- threadgroup arrays: not modeled (out of scope)
                | none => pure ()
            | none => pure ()
        | _ => pure ()
      i := skipStmt p i
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
          st := envSetVar st x (evalE cx st e)
          i := if p.isSym k ";" then k + 1 else skipStmt p k
        | none => i := skipStmt p (i + 2)
      else if p.isSym (i+1) "+=" ∨ p.isSym (i+1) "-=" ∨ p.isSym (i+1) "*=" ∨
              p.isSym (i+1) "/=" ∨ p.isSym (i+1) "%=" then
        st := envSetVar st x .top
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
        -- the else side sees ¬cond (essential for else-if chains: the final
        -- else must carry every negated condition of the chain)
        let stF := applyCond cx cond false st
        if p.isSym (k+1) "{" then
          let (st'', k2) := walkBlock cx stF (k + 2)
          ({ st with finds := st''.finds }, k2)
        else
          let (st'', k2) := walkStmtAt cx stF (k + 1)
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

-- ========================== MSL front end ============================
/-! MSL-subset front end (M-Metal Stage 3): normalizes a hand-written MSL
kernel into the token shape `walkBlock` consumes, so the interval/taint/
guard/OOB analyses are shared verbatim with WGSL.

Subset: one `kernel void` entry with `[[buffer(n)]]` / thread-position
attributes; file-scope `constant T X = …;` (become prelude `let`s);
single-`return` `inline` helpers (token-inlined at call sites); C
declarations incl. multi-declarator lines; ternaries (rewritten to WGSL
`select`, whose evaluator refines each branch under the condition);
`for (T i = …)` headers.

Workgroup size is not in MSL source — the manifest must supply `"wg"`.

Unsoundness boundaries (deliberate, documented):
 * pointer locals (`device T* p = …`) unsupported (absent from our kernels);
 * reads inside NON-inlined (multi-statement) helpers are not analyzed —
   a bare storage-buffer argument to one raises a WARN;
 * threadgroup-memory bounds are unchecked (device buffers only);
 * `simdgroup_load/store` footprint model: 8×8 fragment = base + 7·stride+7
   elements (our kernels use default origins only);
 * statements the walker cannot classify are skipped, never silently passed
   as verified — the tool stays a bug-finder, not a verifier. -/

def mslScalarBytes : String → Nat
  | "float" => 4 | "uint" => 4 | "int" => 4
  | "half" => 2 | "ushort" => 2 | "short" => 2
  | "uchar" => 1 | "char" => 1
  | _ => 0

def mslTypeNames : List String :=
  ["uint", "int", "float", "half", "bool", "ushort", "short", "uchar", "char",
   "uint2", "uint3", "uint4", "int2", "int3", "int4",
   "float2", "float3", "float4", "half2", "half4", "size_t"]

def mslBuiltinMap : String → Option String
  | "thread_position_in_grid" => some "global_invocation_id"
  | "threadgroup_position_in_grid" => some "workgroup_id"
  | "thread_position_in_threadgroup" => some "local_invocation_id"
  | "thread_index_in_threadgroup" => some "local_invocation_index"
  | "thread_index_in_simdgroup" => some "subgroup_invocation_id"
  | _ => none

structure MslHelper where
  name : String
  params : List String
  body : Array PTok      -- return-expression tokens (no `return`, no `;`)

/-- find the position AFTER the `)`/`]`/`}` matching the opener at `i0`. -/
partial def mslMatch (ts : Array PTok) (i0 : Nat) (o c : String) : Nat := Id.run do
  let p : Prs := ⟨ts⟩
  let mut d := 0
  let mut j := i0
  while j < ts.size do
    if p.isSym j o then d := d + 1
    else if p.isSym j c then
      d := d - 1
      if d = 0 then return j + 1
    j := j + 1
  return j

/-- token-level helper inlining: replace `h(a1, …)` with the helper's
return expression, each param occurrence substituted by `( argToks )`. -/
partial def mslInline (helpers : List MslHelper) (ts : Array PTok) : Array PTok := Id.run do
  let p : Prs := ⟨ts⟩
  let mut out : Array PTok := #[]
  let mut i := 0
  let mut changed := false
  while i < ts.size do
    match p.tok i with
    | .ident nm =>
      match helpers.find? (·.name = nm) with
      | some h =>
        if p.isSym (i+1) "(" then
          -- split args at top-level commas
          let endP := mslMatch ts (i+1) "(" ")"
          let mut args : Array (Array PTok) := #[]
          let mut cur : Array PTok := #[]
          let mut d := 0
          let mut j := i + 2
          while j < endP - 1 do
            if p.isSym j "(" ∨ p.isSym j "[" then d := d + 1
            else if p.isSym j ")" ∨ p.isSym j "]" then d := d - 1
            if p.isSym j "," ∧ d = 0 then
              args := args.push cur; cur := #[]
            else
              cur := cur.push ts[j]!
            j := j + 1
          if cur.size > 0 then args := args.push cur
          -- substitute
          let ln := ts[i]!.line
          out := out.push ⟨.sym "(", ln⟩
          for bt in h.body do
            match bt.tok with
            | .ident pn =>
              match h.params.idxOf? pn with
              | some ai =>
                out := out.push ⟨.sym "(", bt.line⟩
                out := out ++ (args.getD ai #[])
                out := out.push ⟨.sym ")", bt.line⟩
              | none => out := out.push bt
            | _ => out := out.push bt
          out := out.push ⟨.sym ")", ln⟩
          i := endP
          changed := true
        else
          out := out.push ts[i]!; i := i + 1
      | none => out := out.push ts[i]!; i := i + 1
    | _ => out := out.push ts[i]!; i := i + 1
  if changed then mslInline helpers out else out

/-- rewrite one ternary `cond ? a : b` → `select(b, a, cond)`; returns none
when no `?` remains. -/
partial def mslTernaryOnce (ts : Array PTok) : Option (Array PTok) := Id.run do
  let p : Prs := ⟨ts⟩
  -- first `?`
  let mut q : Option Nat := none
  for i in [0:ts.size] do
    if p.isSym i "?" then q := some i; break
  let some qi := q | return none
  -- cond start: scan left
  let mut cs := qi
  let mut d := 0
  let mut k := qi
  while k > 0 do
    k := k - 1
    if p.isSym k ")" ∨ p.isSym k "]" then d := d + 1
    else if p.isSym k "(" ∨ p.isSym k "[" then
      if d = 0 then cs := k + 1; break
      d := d - 1
    else if d = 0 ∧ (p.isSym k "," ∨ p.isSym k ";" ∨ p.isSym k "=" ∨
                     p.isSym k "{" ∨ p.isSym k "}" ∨ p.isSym k ":" ∨
                     p.tok k == .ident "return" ∨ p.tok k == .ident "let" ∨
                     p.tok k == .ident "var") then
      cs := k + 1; break
    if k = 0 then cs := 0
  -- matching `:`
  let mut colon := qi
  let mut pd := 0
  let mut tn := 0
  let mut j := qi + 1
  while j < ts.size do
    if p.isSym j "(" ∨ p.isSym j "[" then pd := pd + 1
    else if p.isSym j ")" ∨ p.isSym j "]" then pd := pd - 1
    else if p.isSym j "?" ∧ pd = 0 then tn := tn + 1
    else if p.isSym j ":" ∧ pd = 0 then
      if tn = 0 then colon := j; break
      tn := tn - 1
    j := j + 1
  -- branch2 end
  let mut b2e := ts.size
  pd := 0
  tn := 0
  j := colon + 1
  while j < ts.size do
    if p.isSym j "(" ∨ p.isSym j "[" then pd := pd + 1
    else if (p.isSym j ")" ∨ p.isSym j "]") then
      if pd = 0 then b2e := j; break
      pd := pd - 1
    else if p.isSym j "?" ∧ pd = 0 then tn := tn + 1
    else if p.isSym j ":" ∧ pd = 0 then
      if tn = 0 then b2e := j; break
      tn := tn - 1
    else if (p.isSym j "," ∨ p.isSym j ";") ∧ pd = 0 then b2e := j; break
    j := j + 1
  let ln := ts[qi]!.line
  let cond := ts.extract cs qi
  let b1 := ts.extract (qi+1) colon
  let b2 := ts.extract (colon+1) b2e
  let mut out := ts.extract 0 cs
  out := out.push ⟨.ident "select", ln⟩
  out := out.push ⟨.sym "(", ln⟩
  out := out ++ b2
  out := out.push ⟨.sym ",", ln⟩
  out := out ++ b1
  out := out.push ⟨.sym ",", ln⟩
  out := out ++ cond
  out := out.push ⟨.sym ")", ln⟩
  out := out ++ ts.extract b2e ts.size
  return some out

partial def mslTernary (ts : Array PTok) : Array PTok :=
  match mslTernaryOnce ts with
  | some ts' => mslTernary ts'
  | none => ts

/-- C declaration rewrite: `[const] T x = e[, y = e2]* ;` → `let/var x = e ;
let/var y = e2 ; …`; `for ( [const] T i = …` → `for ( var i = …`. -/
partial def mslDecls (ts : Array PTok) : Array PTok := Id.run do
  let p : Prs := ⟨ts⟩
  let mut out : Array PTok := #[]
  let mut i := 0
  let mut stmtStart := true
  while i < ts.size do
    let isForHead := p.tok i == .ident "for" ∧ p.isSym (i+1) "("
    if isForHead then
      out := out.push ts[i]!
      out := out.push ts[i+1]!
      i := i + 2
      -- optional decl in the for-init
      let j0 := if p.tok i == .ident "const" then i + 1 else i
      match p.tok j0 with
      | .ident t =>
        if mslTypeNames.contains t then
          match p.tok (j0+1) with
          | .ident _ =>
            out := out.push ⟨.ident "var", ts[i]!.line⟩
            i := j0 + 1   -- drop const+type, keep the name onward
          | _ => pure ()
        else pure ()
      | _ => pure ()
      stmtStart := false
    else if stmtStart then
      let (isConst, j0) := if p.tok i == .ident "const" then (true, i + 1) else (false, i)
      let mut consumed := false
      match p.tok j0 with
      | .ident t =>
        if mslTypeNames.contains t then
          match p.tok (j0+1) with
          | .ident _ =>
            -- declaration; split multi-declarators at top-level commas
            let kw : Tok := if isConst then .ident "let" else .ident "var"
            let ln := ts[j0]!.line
            out := out.push ⟨kw, ln⟩
            let mut j := j0 + 1
            let mut d := 0
            while j < ts.size do
              if p.isSym j "(" ∨ p.isSym j "[" then d := d + 1
              else if p.isSym j ")" ∨ p.isSym j "]" then d := d - 1
              if p.isSym j ";" ∧ d = 0 then
                out := out.push ts[j]!; j := j + 1; break
              else if p.isSym j "," ∧ d = 0 then
                out := out.push ⟨.sym ";", ts[j]!.line⟩
                out := out.push ⟨kw, ts[j]!.line⟩
                j := j + 1
              else
                out := out.push ts[j]!; j := j + 1
            i := j
            consumed := true
          | _ => pure ()
        else pure ()
      | _ => pure ()
      if ¬consumed then
        stmtStart := p.isSym i ";" ∨ p.isSym i "{" ∨ p.isSym i "}"
        out := out.push ts[i]!
        i := i + 1
      else
        stmtStart := true
    else
      stmtStart := p.isSym i ";" ∨ p.isSym i "{" ∨ p.isSym i "}"
      out := out.push ts[i]!
      i := i + 1
  return out

/-- parse an MSL kernel file into the shared `Kernel` shape. `wg` comes from
the manifest (MSL has no in-source workgroup size). -/
partial def scanKernelMsl (ts : Array PTok) (wg : Nat × Nat × Nat) : Kernel := Id.run do
  let p : Prs := ⟨ts⟩
  let mut consts : Array (String × Array PTok) := #[]
  let mut helpers : List MslHelper := []
  let mut storages : List StorageVar := []
  let mut builtins : List (String × String) := []
  let mut entryName := ""
  let mut body : Array PTok := #[]
  let mut i := 0
  while i < ts.size do
    match p.tok i with
    | .ident "constant" =>
      -- constant T NAME = <toks> ;
      match p.tok (i+2) with
      | .ident nm =>
        if p.isSym (i+3) "=" then
          let mut j := i + 4
          let mut ex : Array PTok := #[]
          while ¬(p.isSym j ";") ∧ j < ts.size do
            ex := ex.push ts[j]!; j := j + 1
          consts := consts.push (nm, ex)
          i := j + 1
        else i := i + 1
      | _ => i := i + 1
    | .ident "inline" =>
      -- inline RET name ( params ) { … }  — record single-`return` bodies
      let mut j := i + 1
      -- return type: one ident (float2 etc.)
      match p.tok (j+1) with
      | .ident nm =>
        let po := j + 2
        if p.isSym po "(" then
          let pe := mslMatch ts po "(" ")"
          -- params: last ident of each comma-group
          let mut params : List String := []
          let mut cur : Option String := none
          let mut k := po + 1
          let mut d := 0
          while k < pe - 1 do
            if p.isSym k "(" then d := d + 1
            else if p.isSym k ")" then d := d - 1
            if p.isSym k "," ∧ d = 0 then
              if let some c := cur then params := params ++ [c]
              cur := none
            else if let .ident pn := p.tok k then
              cur := some pn
            k := k + 1
          if let some c := cur then params := params ++ [c]
          if p.isSym pe "{" then
            let be := mslMatch ts pe "{" "}"
            -- single-return body?
            if p.tok (pe+1) == .ident "return" then
              let mut ex : Array PTok := #[]
              let mut m := pe + 2
              while ¬(p.isSym m ";") ∧ m < be do
                ex := ex.push ts[m]!; m := m + 1
              if m + 2 ≥ be then   -- `; }` ends the body: single return
                helpers := ⟨nm, params, ex⟩ :: helpers
            i := be
          else i := pe
        else i := j + 2
      | _ => i := i + 1
    | .ident "kernel" =>
      -- kernel void NAME ( params ) { body }
      match p.tok (i+2) with
      | .ident nm =>
        entryName := nm
        let po := i + 3
        let pe := mslMatch ts po "(" ")"
        -- split params at top commas; classify each
        let mut groups : Array (Array PTok) := #[]
        let mut cur : Array PTok := #[]
        let mut d := 0
        let mut k := po + 1
        while k < pe - 1 do
          if p.isSym k "(" ∨ p.isSym k "[" then d := d + 1
          else if p.isSym k ")" ∨ p.isSym k "]" then d := d - 1
          if p.isSym k "," ∧ d = 0 then
            groups := groups.push cur; cur := #[]
          else
            cur := cur.push ts[k]!
          k := k + 1
        if cur.size > 0 then groups := groups.push cur
        for g in groups do
          let gp : Prs := ⟨g⟩
          let idents := g.filterMap fun t =>
            match t.tok with | .ident s => some s | _ => none
          let hasStar := g.any (·.tok == .sym "*")
          -- attribute contents: idents inside `[[ … ]]`
          let mut attr : Option String := none
          let mut bindNo : Option Nat := none
          for ai in [0:g.size] do
            if gp.isSym ai "[" ∧ gp.isSym (ai+1) "[" then
              match gp.tok (ai+2) with
              | .ident a =>
                attr := some a
                if gp.isSym (ai+3) "(" then
                  if let .num n := gp.tok (ai+4) then bindNo := some n
              | _ => pure ()
          -- param name: last ident BEFORE the attribute brackets
          let mut nameIdx : Option String := none
          for ai in [0:g.size] do
            if gp.isSym ai "[" then break
            if let .ident s := gp.tok ai then
              if ¬(["device", "constant", "const", "threadgroup"].contains s)
                 ∧ ¬(mslTypeNames.contains s) then
                nameIdx := some s
          -- fall back: last type-ish ident before [[ is the name for
          -- `uint tid [[…]]` (tid not in type list — covered above)
          match attr with
          | some "buffer" =>
            let rw := ¬ idents.contains "const"
            let tys := idents.filter (mslScalarBytes · > 0)
            let eb := match tys.back? with | some t => mslScalarBytes t | none => 0
            match nameIdx, bindNo with
            | some nm2, some b =>
              if hasStar then
                storages := ⟨nm2, 0, b, rw, eb⟩ :: storages
            | _, _ => pure ()
          | some a =>
            match mslBuiltinMap a, nameIdx with
            | some wb, some nm2 => builtins := (nm2, wb) :: builtins
            | _, _ => pure ()
          | none => pure ()
        if p.isSym pe "{" then
          let be := mslMatch ts pe "{" "}"
          body := ts.extract (pe + 1) be   -- includes the closing `}`
          i := be
        else i := pe
      | _ => i := i + 1
    | _ => i := i + 1
  -- prelude: constants as lets (inlined + ternary-rewritten like the body)
  let mut prelude : Array PTok := #[]
  for (nm, ex) in consts do
    let ln := if h : 0 < ex.size then ex[0]!.line else 1
    prelude := prelude.push ⟨.ident "let", ln⟩
    prelude := prelude.push ⟨.ident nm, ln⟩
    prelude := prelude.push ⟨.sym "=", ln⟩
    prelude := prelude ++ ex
    prelude := prelude.push ⟨.sym ";", ln⟩
  let normBody := mslDecls (mslTernary (mslInline helpers body))
  let toks := (mslTernary prelude) ++ normBody
  -- bodyStart 0: the walk covers the prelude lets THEN the body, ending at
  -- the kernel's own closing `}` (carried inside normBody)
  return { toks := toks, storages := storages,
           entries := [⟨entryName, wg, builtins, 0⟩] }

-- ============================== driver ===============================

structure Binding where
  name : Option String
  group : Option Nat
  bindingNo : Option Nat
  elems : Option Nat           -- element count, or
  bytes : Option Nat           -- byte size (elems derived from the declared type)
  values : List (Option Nat)   -- known contents (index → value), [] if unknown

structure Dispatch where
  kernel : String
  entry : String
  grid : Nat × Nat × Nat
  bindings : List Binding
  lang : Option String := none          -- "msl" routes to the MSL front end
  wg : Option (Nat × Nat × Nat) := none -- REQUIRED for msl (not in source)

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
    let bindings := bs.toList.filterMap fun b => do
      let values := match jGet b "values" >>= jArr with
        | some vs => vs.toList.map jNat
        | none => []
      let elems := jGet b "elems" >>= jNat
      let bytes := jGet b "bytes" >>= jNat
      if elems.isNone ∧ bytes.isNone then none else
      return { name := jGet b "name" >>= jStr,
               group := jGet b "group" >>= jNat,
               bindingNo := jGet b "binding" >>= jNat,
               elems, bytes, values : Binding }
    let wg := match jGet d "wg" >>= jArr with
      | some w => some ((w.getD 0 (Json.num 1) |> jNat).getD 1,
                        (w.getD 1 (Json.num 1) |> jNat).getD 1,
                        (w.getD 2 (Json.num 1) |> jNat).getD 1)
      | none => none
    return { kernel, entry, grid, bindings, lang := jGet d "lang" >>= jStr, wg : Dispatch }

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
    (bufName : String) : Option Nat := do
  let b ← findBinding storages bs bufName
  match b.elems with
  | some e => some e
  | none => do
    let bytes ← b.bytes
    let sv ← storages.find? (·.name = bufName)
    if sv.elemBytes > 0 then some (bytes / sv.elemBytes) else none

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
    let isMsl := d.lang == some "msl" ∨ d.kernel.endsWith ".metal" ∨ d.kernel.endsWith ".msl"
    let k ← if isMsl then
        match d.wg with
        | some wg => pure (scanKernelMsl (tokenize src) wg)
        | none =>
          IO.println s!"{d.kernel}: MSL dispatch needs \"wg\" in the manifest — skipped"
          nWarn := nWarn + 1
          continue
      else pure (scanKernel (tokenize src))
    let some e := k.entries.find? (·.name = d.entry)
      | IO.println s!"{d.kernel}: entry `{d.entry}` not found — skipped"
        nWarn := nWarn + 1
        continue
    let cx : Ctx := {
      p := ⟨k.toks⟩, storages := k.storages,
      elemsOf := resolveElems k.storages d.bindings,
      bufVal := resolveVal k.storages d.bindings,
      grid := d.grid, wg := e.wg, builtins := e.builtins }
    let st0 : WSt := { env := [], refin := [], sguards := [], finds := #[] }
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
