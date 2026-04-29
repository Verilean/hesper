import Hesper.Transpile.CUDA.AST
import Hesper.Transpile.CUDA.Lower
import Hesper.WGSL.Exp
import Hesper.WGSL.Monad

/-! # CUDA → ShaderM transpiler — Phase 3 statement lowering

Lowers `CStmt` (CUDA surface AST statements) into a `ShaderM Unit`
action that builds the kernel body.

Coverage:
  - `__syncthreads()` → `ShaderM.barrier`
  - `if (c) thn` / `if (c) thn else els` → `ShaderM.if_`
  - `for (int k = a; k < b; k += s) body`
       → `ShaderM.loop a b s fun k => body`
  - declaration `int x = expr;` → `ShaderM.varNamed`
  - assignment `x = expr;`, `x += expr;`, `x &= expr;` → `ShaderM.assign`
  - block `{ s1; s2; ... }`
  - expression statement (call) — currently only `__syncthreads`-style
    bare calls, others raise an error
  - pragma — preserved as a no-op for transparency

The lowering is **type-driven** by the CUDA declared type:
  - `int x` / `int32_t` → i32 var
  - `unsigned`, `uint32_t`, `unsigned int` → u32
  - `float`, `f32` → f32

Compound assignment lowering:
  - `acc += e` → `assign acc (acc + e)` (re-binds the variable)
  - same for `-=`, `*=`, `&=`, `|=`, `^=`

The for-loop pattern we accept matches what shows up in MMQ kernels:

  for (int k0 = 0; k0 < MMQ_TILE_NE_K; k0 += STEP) { ... }

We extract `init = 0`, `bound = MMQ_TILE_NE_K` (as a u32 expr), and
`step = STEP` (literal), then emit `ShaderM.loop`.
-/
namespace Hesper.Transpile.CUDA

open Hesper.WGSL Hesper.WGSL.Monad

/-! ## Type detection from CUDA type strings -/

/-- Strip `const`/`volatile`/`__restrict__` qualifiers and pointer
    markers, then classify the remaining base type. -/
def classifyTy (s : String) : Option WGSLType :=
  -- Quick approach: look for keywords in the type string.
  if s.endsWith "*" ∨ s.endsWith "&" then none  -- pointer types unsupported
  else if s.endsWith "uint32_t" ∨ s.endsWith "unsigned" ∨ s == "unsigned int"
       ∨ s.endsWith "u32" ∨ s == "uint" then
    some (.scalar .u32)
  else if s.endsWith "int32_t" ∨ s.endsWith "int" ∨ s == "i32"
       ∨ s == "signed int" ∨ s == "signed" then
    some (.scalar .i32)
  else if s.endsWith "float" ∨ s.endsWith "f32" then
    some (.scalar .f32)
  else if s.endsWith "double" then
    some (.scalar .f32)  -- map double to f32
  else if s == "bool" then
    some (.scalar .bool)
  else
    none

/-! ## Statement lowering

We have separate lowering for the three scalar types we currently
support (u32 / i32 / f32), unified by dispatch on declared type. -/

mutual

/-- Lower a single statement to a ShaderM action. -/
partial def lowerStmt (env : Env) : CStmt → Except String (ShaderM Unit)
  | .sync_ => .ok ShaderM.barrier
  | .pragma _ => .ok (pure ())  -- preserve presence as no-op
  | .break_ => .error "lowerStmt: 'break' not supported"
  | .continue_ => .error "lowerStmt: 'continue' not supported"
  | .return_ _ => .error "lowerStmt: 'return' inside body not supported"
  | .block stmts => do
    let actions ← stmts.toList.mapM (lowerStmt env)
    .ok (actions.foldl (fun acc a => acc *> a) (pure ()))
  | .if_ c thn els => do
    let condE ← lowerBool env c
    let thnA ← lowerStmt env thn
    let elsA ← match els with
      | some s => lowerStmt env s
      | none => .ok (pure ())
    .ok (ShaderM.if_ condE thnA elsA)
  | .for_ initOpt condOpt stepOpt body =>
    lowerFor env initOpt condOpt stepOpt body
  | .while_ _ _ => .error "lowerStmt: 'while' not yet supported (use 'for')"
  | .decl ty name initOpt =>
    match classifyTy ty with
    | some (.scalar .u32) => do
      let init ← match initOpt with
        | some e => lowerU32 env e
        | none => .ok (Exp.litU32 0)
      .ok (ShaderM.varNamed name (.scalar .u32) init)
    | some (.scalar .i32) => do
      let init ← match initOpt with
        | some e => lowerI32 env e
        | none => .ok (Exp.litI32 0)
      .ok (ShaderM.varNamed name (.scalar .i32) init)
    | some (.scalar .f32) => do
      let init ← match initOpt with
        | some e => lowerF32 env e
        | none => .ok (Exp.litF32 0.0)
      .ok (ShaderM.varNamed name (.scalar .f32) init)
    | _ => .error s!"lowerStmt: unsupported decl type '{ty}'"
  | .declArr ty name _szExpr =>
    -- `__shared__ int x[N]` — Phase 4 will handle smem; for now accept
    -- and emit a comment-only no-op so we can at least *parse* kernels
    -- containing them.
    .error s!"lowerStmt: array decl '{ty} {name}[…]' not yet supported (Phase 4)"
  | .expr e =>
    -- Expression statement: usually an assignment.
    lowerExprStmt env e

/-- Lower `for (init; cond; step) body` into a `ShaderM.loop`. We
    require the standard pattern:
      init  = `int k = <start>;`
      cond  = `k < <end>` (or `<=`)
      step  = `k += <stepLit>` or `k++`. -/
partial def lowerFor (env : Env)
    (initOpt : Option CStmt) (condOpt : Option CExpr) (stepOpt : Option CExpr)
    (body : CStmt) : Except String (ShaderM Unit) := do
  -- Init must be `<int|uint> k = <expr>;`
  let (iname, startExpr) ← match initOpt with
    | some (.decl _ name (some e)) => .ok (name, e)
    | some (.expr (.binop .assign (.ident name) e)) => .ok (name, e)
    | _ => .error "lowerFor: expected `int k = e` style init"
  -- Cond: `k < bound` or `k <= bound`
  let (boundExpr, boundIsLE) ← match condOpt with
    | some (.binop .lt (.ident n) b) =>
      if n == iname then .ok (b, false)
      else .error s!"lowerFor: loop var mismatch in cond ('{n}' vs '{iname}')"
    | some (.binop .le (.ident n) b) =>
      if n == iname then .ok (b, true)
      else .error s!"lowerFor: loop var mismatch in cond ('{n}' vs '{iname}')"
    | _ => .error "lowerFor: expected `k < bound` / `k <= bound`"
  -- Step: `k += s` or `k++`
  let stepExpr : CExpr ← match stepOpt with
    | some (.binop .addAssign (.ident n) s) =>
      if n == iname then .ok s
      else .error s!"lowerFor: loop var mismatch in step ('{n}' vs '{iname}')"
    | some (.unop .postInc (.ident n)) | some (.unop .preInc (.ident n)) =>
      if n == iname then .ok (.numLit "1")
      else .error s!"lowerFor: loop var mismatch in step ('{n}' vs '{iname}')"
    | _ => .error "lowerFor: expected `k += s` or `k++`"
  -- Lower start, bound, step as u32 (CUDA `int` loop counters → u32 is
  -- fine because llama.cpp loops are non-negative).
  let s ← lowerU32 env startExpr
  let bExpr ← lowerU32 env boundExpr
  -- ShaderM.loop is a half-open range; if cond was `<=`, add 1 to bound.
  let bound := if boundIsLE then Exp.add bExpr (Exp.litU32 1) else bExpr
  let stepE ← lowerU32 env stepExpr
  -- ShaderM.loop generates a fresh loop-var name (`i0`, `i1`, ...) and
  -- gives us its Exp. We bind that Exp to `iname` in *all three* env
  -- type slots — `lowerXxx` only checks its own type slot, and the
  -- loop var is conceptually polymorphic (`int k` in CUDA can flow
  -- into i32, u32, or f32 contexts via implicit conversion).
  let bodyOf (k : Exp (.scalar .u32)) : Except String (ShaderM Unit) :=
    let env' : Env := {
      u32 := fun n => if n == iname then some k             else env.u32 n,
      i32 := fun n => if n == iname then some (Exp.toI32 k) else env.i32 n,
      f32 := fun n => if n == iname then some (Exp.toF32 k) else env.f32 n
    }
    lowerStmt env' body
  -- We need `bodyOf` to produce a fresh ShaderM action *per* call to
  -- the loop body, so each iteration sees the correct loop-var Exp.
  -- Convert Except (ShaderM Unit) into ShaderM Unit by panicking on
  -- error (errors at this layer have already been validated by the
  -- expression-level lowerings above).
  .ok do
    ShaderM.loop s bound stepE fun k =>
      match bodyOf k with
      | .ok a => a
      | .error msg => panic! s!"lowerFor body: {msg}"

/-- Lower a CUDA bool expression. Phase 3 supports comparisons (`a < b`)
    and short-circuit `&&` / `||`. -/
partial def lowerBool (env : Env) : CExpr → Except String (Exp (.scalar .bool))
  | .binop op a b =>
    let cmp (mk : ∀ {t : WGSLType}, Exp t → Exp t → Exp (.scalar .bool))
        : Except String (Exp (.scalar .bool)) := do
      -- Try u32 first, then i32, then f32.
      match lowerU32 env a, lowerU32 env b with
      | .ok ae, .ok be => .ok (mk ae be)
      | _, _ =>
        match lowerI32 env a, lowerI32 env b with
        | .ok ae, .ok be => .ok (mk ae be)
        | _, _ =>
          match lowerF32 env a, lowerF32 env b with
          | .ok ae, .ok be => .ok (mk ae be)
          | _, _ => .error "lowerBool: cmp operands type-mismatch"
    match op with
    | .lt => cmp Exp.lt
    | .le => cmp Exp.le
    | .gt => cmp Exp.gt
    | .ge => cmp Exp.ge
    | .eq => cmp Exp.eq
    | .ne => cmp Exp.ne
    | .logAnd => do
      let ae ← lowerBool env a; let be ← lowerBool env b
      .ok (Exp.and ae be)
    | .logOr => do
      let ae ← lowerBool env a; let be ← lowerBool env b
      .ok (Exp.or ae be)
    | _ => .error s!"lowerBool: unsupported binop {repr op}"
  | .unop .logNot a => do
    let ae ← lowerBool env a; .ok (Exp.not ae)
  | _ => .error "lowerBool: only comparisons / bool ops supported"

/-- Lower an expression statement. Most are assignments. -/
partial def lowerExprStmt (env : Env) (e : CExpr) : Except String (ShaderM Unit) :=
  match e with
  | .binop .assign (.ident name) rhs =>
    -- Determine type from env presence; default to u32. Allow
    -- numeric-literal RHS to flow into i32/f32 contexts.
    if env.f32 name |>.isSome then
      match lowerF32 env rhs with
      | .ok v => .ok (ShaderM.assign name v)
      | .error _ =>
        match lowerI32 env rhs with
        | .ok v => .ok (ShaderM.assign name (Exp.toF32 v))
        | .error _ =>
          lowerU32 env rhs |>.map fun v =>
            ShaderM.assign name (Exp.toF32 v)
    else if env.i32 name |>.isSome then
      match lowerI32 env rhs with
      | .ok v => .ok (ShaderM.assign name v)
      | .error _ =>
        lowerU32 env rhs |>.map fun v =>
          ShaderM.assign name (Exp.toI32 v)
    else
      lowerU32 env rhs |>.map fun v => ShaderM.assign name v
  | .binop .addAssign (.ident name) rhs =>
    lowerCompoundAssign env name rhs Exp.add Exp.add Exp.add
  | .binop .subAssign (.ident name) rhs =>
    lowerCompoundAssign env name rhs Exp.sub Exp.sub Exp.sub
  | .binop .mulAssign (.ident name) rhs =>
    lowerCompoundAssign env name rhs Exp.mul Exp.mul Exp.mul
  | .binop .andAssign (.ident name) rhs => do
    -- Bitwise AND: u32 only.
    let v ← lowerU32 env rhs
    .ok (ShaderM.assign name (Exp.bitAnd (Exp.var name) v))
  | .binop .orAssign (.ident name) rhs => do
    let v ← lowerU32 env rhs
    .ok (ShaderM.assign name (Exp.bitOr (Exp.var name) v))
  | .binop .xorAssign (.ident name) rhs => do
    let v ← lowerU32 env rhs
    .ok (ShaderM.assign name (Exp.bitXor (Exp.var name) v))
  | .call "__syncthreads" #[] => .ok ShaderM.barrier
  | _ => .error s!"lowerExprStmt: unsupported expression statement"

partial def lowerCompoundAssign (env : Env) (name : String) (rhs : CExpr)
    (mkU32 : Exp (.scalar .u32) → Exp (.scalar .u32) → Exp (.scalar .u32))
    (mkI32 : Exp (.scalar .i32) → Exp (.scalar .i32) → Exp (.scalar .i32))
    (mkF32 : Exp (.scalar .f32) → Exp (.scalar .f32) → Exp (.scalar .f32))
    : Except String (ShaderM Unit) :=
  if env.f32 name |>.isSome then
    -- f32 += : try f32, then i32→f32, then u32→f32.
    match lowerF32 env rhs with
    | .ok v => .ok (ShaderM.assign name (mkF32 (Exp.var name) v))
    | .error _ =>
      match lowerI32 env rhs with
      | .ok v => .ok (ShaderM.assign name (mkF32 (Exp.var name) (Exp.toF32 v)))
      | .error _ =>
        lowerU32 env rhs |>.map fun v =>
          ShaderM.assign name (mkF32 (Exp.var name) (Exp.toF32 v))
  else if env.i32 name |>.isSome then
    -- i32 += : try i32, then u32→i32 cast.
    match lowerI32 env rhs with
    | .ok v => .ok (ShaderM.assign name (mkI32 (Exp.var name) v))
    | .error _ =>
      lowerU32 env rhs |>.map fun v =>
        ShaderM.assign name (mkI32 (Exp.var name) (Exp.toI32 v))
  else
    lowerU32 env rhs |>.map fun v =>
      ShaderM.assign name (mkU32 (Exp.var name) v)

end -- mutual

end Hesper.Transpile.CUDA
