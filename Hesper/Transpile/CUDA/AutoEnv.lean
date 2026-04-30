import Hesper.Transpile.CUDA.AST
import Hesper.Transpile.CUDA.Parse
import Hesper.Transpile.CUDA.Lower
import Hesper.Transpile.CUDA.MMQEnv

/-! # Two-pass auto Env generation

Phase 12 added a hand-written `mmqDefaultEnv` covering arch-specific
helper inlines (`mmq_get_nwarps_device`, `min`, `max`, ...) for the
RTX 4070 Ti target. But each CUDA kernel still needed a per-kernel
hand-written env for its `__global__` parameter buffers and
`__device__` helper functions.

This module closes that gap with a **2-pass pipeline**:

  * Pass 1 — symbol collection: walk the parsed `Array TUItem`,
    record (a) every `__device__` / `__global__` function signature,
    (b) every parameter that's a pointer type, (c) every standalone
    helper `static T f() { return <numLit>; }` with a constant body.
  * Pass 2 — env synthesis: build `Env.bufs` from the collected
    pointer-typed parameters of `targetFn`, and `Env.inlines` from
    the constant-bodied helpers + `mmqHelperInlines`.

This is what `nvcc` does implicitly via its symbol-table traversal —
we just had to write it down once.

Usage:
```
let items := parseTranslationUnitStr src
let env := autoEnv items "vec_dot_q4_K_q8_1_dp4a"
match lowerStmt env body with ...
```

-/
namespace Hesper.Transpile.CUDA

open Hesper.WGSL

/-! ## Pass 1 — symbol collection -/

/-- A sketch of one function gathered from the TU. -/
structure FnSig where
  name   : String
  params : Array CParam
  /-- If the body is `{ return <numLit>; }`, we record the literal
      so Pass 2 can register an inline. -/
  constReturn? : Option String := none
  deriving Inhabited

/-- Detect a single-statement `return <numLit>;` (with optional cast
    or unary-neg wrapper). Returns the literal text on success.
    Non-recursive: handles `{ return <numLit>; }` by peeking inside a
    1-stmt block once. -/
def detectConstReturnStmt : CStmt → Option String
  | .return_ (some (.numLit s)) => some s
  | .return_ (some (.cast _ (.numLit s))) => some s
  | .return_ (some (.unop .neg (.numLit s))) => some s!"-{s}"
  | _ => none

def detectConstReturn : CStmt → Option String
  | .block stmts =>
    if h : stmts.size = 1 then detectConstReturnStmt (stmts[0]'(by simp [h]))
    else none
  | s => detectConstReturnStmt s

/-- Walk the items and collect all function signatures. -/
def collectFns (items : Array TUItem) : Array FnSig := Id.run do
  let mut acc : Array FnSig := #[]
  for it in items do
    match it with
    | .function f =>
      acc := acc.push {
        name := f.name,
        params := f.params,
        constReturn? := detectConstReturn f.body }
    | _ => pure ()
  acc

/-! ## Pass 2 — env synthesis -/

/-- Strip `const`, `volatile`, `__restrict__`, `__device__`,
    `__constant__` qualifiers from a type string and return the
    "core" type (still a string). -/
def stripQualifiers (ty : String) : String :=
  let parts := (ty.splitOn " ").filter fun p =>
    p ≠ "const" ∧ p ≠ "volatile" ∧ p ≠ "__restrict__"
    ∧ p ≠ "__device__" ∧ p ≠ "__constant__" ∧ p ≠ ""
  String.intercalate " " parts

/-- Classify a CUDA pointer-element type to a hesper `WGSLType`.
    Returns `none` when we don't recognise the element type yet —
    the caller can either skip the binding or fall through to a
    synthetic u32 buffer (we do the latter, since most CUDA kernel
    pointers eventually decompose to packed u32 reads). -/
def classifyPointerElem (ty : String) : WGSLType :=
  let core := stripQualifiers ty
  -- Drop trailing `*` markers. We don't differentiate `T *` vs
  -- `T **` here; smem layouts in mmq.cuh always single-indirect.
  let core := core.replace "*" ""
  let core := core.trim
  match core with
  | "int" | "int32_t"           => .scalar .i32
  | "uint" | "uint32_t" | "unsigned" | "unsigned int" => .scalar .u32
  | "float"                     => .scalar .f32
  | "char" | "int8_t" | "uint8_t" => .scalar .u32  -- byte-packed
  | "short" | "int16_t" | "uint16_t" => .scalar .u32
  | "half" | "half2"            => .scalar .u32
  | "float2" | "float4"         => .scalar .u32
  | _ =>
    -- Unknown struct types like `block_q4_K *` or `block_q8_1 *` —
    -- treat as opaque u32 buffer (kernel will index by byte offsets
    -- via env.structFields if the user registers them; otherwise
    -- the kernel body's first struct-member access will fail at
    -- lower time with a clear "structFields not registered" error).
    .scalar .u32

/-- True if a parameter type is a pointer (raw `T *` or `T &` form). -/
def isPointerParam (ty : String) : Bool :=
  ty.endsWith "*" ∨ ty.endsWith " *" ∨ ty.endsWith "&" ∨ ty.endsWith " &"
  ∨ (ty.splitOn " ").any (fun p => p == "*" ∨ p == "&")

/-- Build `Env.bufs` from a target function's parameter list. Every
    pointer-typed parameter `T * name` becomes a buffer binding
    `name → { name := s!"{name}_buf", elemTy := classifyPointerElem T }`.
    Non-pointer params are left unbound (caller likely passes them
    through env.u32/i32 by separate mechanisms, e.g. template params). -/
def bufsOfParams (params : Array CParam) : String → Option BufBinding :=
  fun n =>
    -- Linear search is fine — kernel-arg lists are <30 entries.
    params.foldl (init := none) fun acc p =>
      match acc with
      | some _ => acc  -- already found
      | none =>
        if p.name == n ∧ isPointerParam p.ty then
          some { name := s!"{p.name}_buf",
                 elemTy := classifyPointerElem p.ty,
                 offset? := none }
        else
          none

/-- Build an `Env.inlines`-style rewrite from a list of
    constant-bodied helpers. Each `f` with `f.constReturn? = some "8"`
    becomes a 0-arg inline that returns `numLit "8"`. -/
def inlinesOfConstFns (fns : Array FnSig) : String → Array CExpr → Option CExpr :=
  fun fn _args =>
    fns.foldl (init := none) fun acc f =>
      match acc with
      | some _ => acc
      | none =>
        match f.constReturn? with
        | some s => if f.name == fn then some (CExpr.numLit s) else none
        | none => none

/-- Compose two `inlines` resolvers: try the first, fall back to the
    second. Used to layer the auto-collected const-fn inlines on top
    of the static `mmqHelperInlines`. -/
def chainInlines
    (a : String → Array CExpr → Option CExpr)
    (b : String → Array CExpr → Option CExpr)
    : String → Array CExpr → Option CExpr :=
  fun fn args => match a fn args with
    | some e => some e
    | none => b fn args

/-- Two-pass `Env` builder.

    Given the parsed translation unit and the *target* function name
    (the kernel we want to lower), produce an `Env` with:

    * `bufs` — auto-built from the target function's pointer params
      (kernel argument buffers).
    * `inlines` — auto-built from every standalone const-return helper
      in the TU, layered over `mmqHelperInlines`.
    * `threadIdx*` / `blockIdx*` / `blockDim*` builtins from
      `mmqDefaultMembers`.

    Returns `none` when the target function is not present in the TU. -/
def autoEnv (items : Array TUItem) (targetFn : String) : Option Env := Id.run do
  let fns := collectFns items
  let mut target : Option FnSig := none
  for f in fns do
    if f.name == targetFn then target := some f
  match target with
  | none => return none
  | some t =>
    let autoBufs := bufsOfParams t.params
    let autoInlines := inlinesOfConstFns fns
    return some { mmqDefaultMembers with
      bufs := autoBufs
      inlines := chainInlines autoInlines mmqHelperInlines }

/-- Convenience: like `autoEnv` but returns `mmqDefaultEnv` when the
    target isn't found. Useful for the lower-coverage map driver
    where every function in the TU is its own target. -/
def autoEnvFor (items : Array TUItem) (target : CFunction) : Env :=
  let fns := collectFns items
  let autoBufs := bufsOfParams target.params
  let autoInlines := inlinesOfConstFns fns
  { mmqDefaultMembers with
    bufs := autoBufs
    inlines := chainInlines autoInlines mmqHelperInlines }

/-! ## Pass 0 — `#define` const scan

The lexer strips `#define` directives entirely, so simple macro
constants like `#define VDR_Q4_0_Q8_1_MMVQ 2` get lost.  We rescue
them with a one-pass linewise text scan: any line of the form
`#define IDENT <integer-literal>` becomes a `(IDENT, n)` entry that
the lower's `env.consts` resolver can find.

This unblocks local-array decls whose size is a `#define` (e.g.
`int v[VDR_Q4_0_Q8_1_MMVQ];`) and `for (i = 0; i < CONST; ++i)`
loops whose bound is a `#define`. -/

/-- Parse the operand of a `#define NAME <text>` line as an Int when
    `<text>` is a bare decimal/hex integer literal. Returns `none` for
    function-like macros, multi-token operands, or non-integer values. -/
def parseDefineLineRhs (rhs : String) : Option Int :=
  let core := rhs.trim
  if core.isEmpty then none
  else
    -- Reject anything that doesn't look like a single int literal.
    let isOK : Bool := core.all (fun c =>
      c.isDigit ∨ c == '-' ∨ c == 'x' ∨ c == 'X' ∨ c == 'u' ∨ c == 'U'
      ∨ c == 'l' ∨ c == 'L' ∨ ('a' ≤ c ∧ c ≤ 'f')
      ∨ ('A' ≤ c ∧ c ≤ 'F'))
    if !isOK then none
    else
      let (sign, body) :=
        if core.startsWith "-" then ((-1 : Int), core.drop 1) else ((1 : Int), core)
      match parseIntLit body with
      | .ok n => some (sign * (n : Int))
      | .error _ => none

/-- Scan source text for `#define IDENT <int>` lines. Function-like
    macros (`#define F(x) ...`) and non-integer values are skipped. -/
def collectDefines (src : String) : Array (String × Int) := Id.run do
  let mut acc : Array (String × Int) := #[]
  for line in src.splitOn "\n" do
    let line := line.trim
    if !line.startsWith "#" then continue
    -- Drop the leading `#` and any whitespace, expect `define`.
    let rest := (line.drop 1).trim
    if !rest.startsWith "define" then continue
    let rest := (rest.drop 6).trim
    -- Identifier portion (alpha/_/digits); stop at first non-ident char.
    let chars := rest.toList
    let mut nameAcc : String := ""
    let mut i : Nat := 0
    let mut ok : Bool := true
    while i < chars.length do
      let c := chars[i]!
      if c.isAlpha ∨ c == '_' ∨ (i > 0 ∧ c.isDigit) then
        nameAcc := nameAcc.push c
        i := i + 1
      else
        break
    -- Skip function-like macros: `#define NAME(...)`.
    if i < chars.length ∧ chars[i]! == '(' then ok := false
    if !ok ∨ nameAcc.isEmpty then continue
    let rhs := (rest.drop nameAcc.length).trim
    match parseDefineLineRhs rhs with
    | some n => acc := acc.push (nameAcc, n)
    | none   => continue
  acc

/-- Build a `consts` resolver from the collected `#define` pairs,
    layered over an existing resolver `prior` (so caller-provided
    consts still win). -/
def constsOfDefines (defs : Array (String × Int))
    (prior : String → Option Int) : String → Option Int :=
  fun n => match prior n with
    | some v => some v
    | none =>
      defs.foldl (init := none) fun acc (k, v) =>
        match acc with
        | some _ => acc
        | none => if k == n then some v else none

/-- `autoEnvFor` extended with `#define` constant pickup. Pass the
    source string in addition to the parsed TU and target — defines
    found in the source feed `Env.consts`, so things like
    `int v[VDR_Q4_0_Q8_1_MMVQ];` lower with the constant array size. -/
def autoEnvForWithDefines
    (src : String) (items : Array TUItem) (target : CFunction) : Env :=
  let baseEnv := autoEnvFor items target
  let defs := collectDefines src
  { baseEnv with consts := constsOfDefines defs baseEnv.consts }

/-- Multi-source variant: scan defines from all given source strings
    (typically the file plus its `#include` chain) and merge them all
    into `env.consts`. Useful for vecdotq.cuh which depends on
    `QR2_K`, `QI4_K` etc defined in `ggml-common.h`. -/
def autoEnvForWithMultiDefines
    (srcs : Array String) (items : Array TUItem) (target : CFunction) : Env :=
  let baseEnv := autoEnvFor items target
  let allDefs : Array (String × Int) := srcs.foldl (init := #[]) fun acc s =>
    acc ++ collectDefines s
  { baseEnv with consts := constsOfDefines allDefs baseEnv.consts }

end Hesper.Transpile.CUDA
