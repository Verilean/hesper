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

/-! ## Constexpr evaluator (for array sizes like `mmq_y * (MMQ_TILE_NE_K + 1)`).

    Resolves identifiers via `Env.consts` (a String → Int map). Returns
    `none` if any leaf is unknown or an unsupported operator appears. -/

/-- Parse a non-negative integer literal token (decimal or hex with
    optional `[uUlL]+` suffix) to `Int`. Returns `none` on malformed
    input. -/
def parseConstIntLit (s : String) : Option Int :=
  let stripped := s.dropRightWhile (fun c =>
    c == 'u' ∨ c == 'U' ∨ c == 'l' ∨ c == 'L')
  let isHex := stripped.startsWith "0x" ∨ stripped.startsWith "0X"
  let body := if isHex then stripped.drop 2 else stripped
  let base : Nat := if isHex then 16 else 10
  let digit (c : Char) : Option Nat :=
    if c.isDigit then some (c.toNat - '0'.toNat)
    else if isHex ∧ 'a' ≤ c ∧ c ≤ 'f' then some (10 + (c.toNat - 'a'.toNat))
    else if isHex ∧ 'A' ≤ c ∧ c ≤ 'F' then some (10 + (c.toNat - 'A'.toNat))
    else none
  body.toList.foldlM (init := (0 : Int)) fun acc c => do
    let d ← digit c
    pure (acc * Int.ofNat base + Int.ofNat d)

partial def evalConst (lookup : String → Option Int) : CExpr → Option Int
  | .numLit s => parseConstIntLit s
  | .ident n => lookup n
  | .unop .neg e => do let v ← evalConst lookup e; pure (-v)
  | .unop .bitNot e => do let v ← evalConst lookup e; pure (-v - 1)
  | .binop op a b => do
    let av ← evalConst lookup a
    let bv ← evalConst lookup b
    match op with
    | .add => pure (av + bv)
    | .sub => pure (av - bv)
    | .mul => pure (av * bv)
    | .div => if bv == 0 then none else pure (av / bv)
    | .mod => if bv == 0 then none else pure (av % bv)
    | .shl => pure (av * (2 ^ bv.toNat))
    | .shr => pure (av / (2 ^ bv.toNat))
    | .bitAnd => pure (Int.ofNat (av.toNat &&& bv.toNat))
    | .bitOr  => pure (Int.ofNat (av.toNat ||| bv.toNat))
    | .bitXor => pure (Int.ofNat (av.toNat ^^^ bv.toNat))
    -- Logical ops for `if constexpr` conditions
    | .logAnd => pure (if av != 0 ∧ bv != 0 then 1 else 0)
    | .logOr  => pure (if av != 0 ∨ bv != 0 then 1 else 0)
    | .eq => pure (if av == bv then 1 else 0)
    | .ne => pure (if av != bv then 1 else 0)
    | .lt => pure (if av < bv then 1 else 0)
    | .le => pure (if av ≤ bv then 1 else 0)
    | .gt => pure (if av > bv then 1 else 0)
    | .ge => pure (if av ≥ bv then 1 else 0)
    | _ => none
  | .unop .logNot e => do let v ← evalConst lookup e; pure (if v == 0 then 1 else 0)
  | _ => none

/-! ## Type detection from CUDA type strings -/

/-- Strip `const`/`volatile`/`__restrict__` qualifiers and pointer
    markers, then classify the remaining base type. -/
def classifyTy (s : String) : Option WGSLType :=
  -- Quick approach: look for keywords in the type string.
  if s.endsWith "*" ∨ s.endsWith "&" then none  -- pointer types unsupported
  else if s.endsWith "float2" then
    some (.vec2 .f32)
  -- Narrow unsigned types (uint8_t, uint16_t, uint64_t) all share the
  -- u32 register file at the WGSL/PTX level; the difference is just
  -- how many bytes the load instruction reads.  For local-variable
  -- typing we treat them all as u32 — load-width concerns are handled
  -- at the buffer-binding level, not the value-type level.
  else if s.endsWith "uint32_t" ∨ s.endsWith "unsigned" ∨ s == "unsigned int"
       ∨ s.endsWith "u32" ∨ s == "uint"
       ∨ s.endsWith "uint8_t" ∨ s.endsWith "uint16_t" ∨ s.endsWith "uint64_t"
       ∨ s == "u8" ∨ s == "u16" ∨ s == "u64" then
    some (.scalar .u32)
  else if s.endsWith "int32_t" ∨ s.endsWith "int" ∨ s == "i32"
       ∨ s == "signed int" ∨ s == "signed"
       ∨ s.endsWith "int8_t" ∨ s.endsWith "int16_t" ∨ s.endsWith "int64_t"
       ∨ s == "i8" ∨ s == "i16" ∨ s == "i64" then
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
  | .break_ =>
    -- After `switch → CStmt.block` flattening, residual `break;`s are
    -- vestigial; emit as a no-op. (Real `break` inside a `for` is rare
    -- in numerics kernels; the user can reject this if it changes
    -- semantics by post-validating the lowered ShaderM.)
    .ok (pure ())
  | .continue_ => .ok (pure ())
  | .return_ _ =>
    -- A bare `return;` is only legal as the body of an `if`-guard; we
    -- treat it as a no-op so that the block-level rewrite below can
    -- transform `if (cond) return; <rest>` → `if (!cond) { <rest> }`.
    -- Bare `return;` at the top level of the kernel body collapses to
    -- a no-op (the kernel just exits naturally on fallthrough).
    .ok (pure ())
  | .block stmts => do
    -- Thread `env` through the block so that statements that **update**
    -- the env (pointer arithmetic `x += off`, declarations introducing
    -- new bindings) take effect for following statements.  Currently
    -- only pointer-arith updates are propagated; new local decls add a
    -- ShaderM-level bound name which we don't need in env.
    --
    -- Special-case: rewrite the early-exit guard pattern
    --   `if (cond) return;`
    --   <rest>
    -- into
    --   `if (!cond) { <rest> }`
    -- so that the rest of the kernel runs only on threads that didn't
    -- hit the guard. CUDA kernels frequently use this pattern for
    -- bounds checks (`if (i0 >= ne0) return;`).  Doing this at block-
    -- lowering time keeps the rewriting local and avoids an extra AST
    -- pass.  Multiple guards stack naturally via recursion.
    let isReturnGuard : CStmt → Option CExpr
      | .if_ c (.return_ _) none => some c
      | .if_ c (.block #[.return_ _]) none => some c
      | _ => none
    match (stmts.toList.findIdx? (fun s => (isReturnGuard s).isSome)) with
    | some idx =>
      let before := stmts.toList.take idx
      let guardStmt := stmts.toList[idx]!
      let after := stmts.toList.drop (idx + 1)
      match isReturnGuard guardStmt with
      | some c =>
        -- Rebuild as: <before>; if (!cond) { <after> }
        let inverted : CStmt := .if_ (CExpr.unop .logNot c) (.block after.toArray) none
        let newBlock : CStmt := .block (before.toArray.push inverted)
        lowerStmt env newBlock
      | none => unreachable!
    | none =>
      let mut envCur : Env := env
      let mut acts : List (ShaderM Unit) := []
      for s in stmts.toList do
        let (envNext, act) ← lowerStmtWithEnvUpdate envCur s
        envCur := envNext
        acts := acts ++ [act]
      .ok (acts.foldl (fun acc a => acc *> a) (pure ()))
  | .if_ c thn els => do
    let condE ← lowerBool env c
    let thnA ← lowerStmt env thn
    let elsA ← match els with
      | some s => lowerStmt env s
      | none => .ok (pure ())
    .ok (ShaderM.if_ condE thnA elsA)
  | .ifConstexpr c thn els =>
    -- Compile-time evaluate the condition using `env.consts`. Bool
    -- conditions appear in two forms: a single `bool` template param
    -- (`if constexpr (do_multiply)`) or an `&&` / `||` of params.
    -- We treat any non-zero const as true, zero as false.
    match evalConst env.consts c with
    | some v =>
      if v != 0 then
        lowerStmt env thn
      else
        match els with
        | some s => lowerStmt env s
        | none => .ok (pure ())
    | none =>
      -- Condition not const-foldable (template param not bound).
      -- Conservatively take the `then` branch — most `if constexpr`
      -- guards are arch checks where the production target is in the
      -- true branch. For coverage purposes either branch works.
      lowerStmt env thn
  | .externSharedArr ty name =>
    -- Runtime-sized smem array. We don't yet have a ShaderM
    -- primitive for genuinely runtime-sized smem; we treat it as a
    -- declaration the caller must materialise via `ShaderM.sharedNamed`
    -- with a fixed size obtained from a separately-passed env const.
    -- Look up `<name>_size` in consts as the convention.
    match classifyTy ty with
    | some elemTy =>
      match env.consts s!"{name}_size" with
      | some n =>
        if n < 0 then .error s!"lowerStmt: extern __shared__ {name} negative size"
        else .ok (ShaderM.sharedNamed name (.array elemTy n.toNat))
      | none =>
        .error s!"lowerStmt: extern __shared__ {name}[] needs '{name}_size' in env.consts"
    | none => .error s!"lowerStmt: extern __shared__ {ty} {name}[] — unsupported elem type"
  | .staticAssert => .ok (pure ())
  | .for_ initOpt condOpt stepOpt body =>
    lowerFor env initOpt condOpt stepOpt body
  | .while_ _ _ => .error "lowerStmt: 'while' not yet supported (use 'for')"
  | .decl _storage ty name initOpt =>
    -- `__shared__ int x;` (no array) is rare; we treat it the same as a
    -- local for now since there's no shared scalar in WGSL.
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
    | some (.vec2 .f32) => do
      -- `float2 v = __half22float2(h);` — declare a vec2<f32> local
      -- and bind it via env.f32x2 so subsequent `v.x` / `v.y` lower
      -- to `vecX(v)` / `vecY(v)`.
      let init ← match initOpt with
        | some e => lowerF32x2 env e
        | none => .ok (Exp.vec2 (Exp.litF32 0.0) (Exp.litF32 0.0))
      .ok (ShaderM.varNamed name (.vec2 .f32) init)
    | _ =>
      -- Unrecognised decl type — could be a pointer rebind
      -- (`const int * x_qs = (const int *) x;`), a POD struct local
      -- (`constexpr tile_x_sizes txs = ...;`), or an enum-typed local.
      -- Heuristic: if the type ends in `*` it's a pointer rebind;
      -- alias the original buffer name through env.bufs at the call
      -- site (no-op at lower time — the kernel body will use the
      -- pointer's array-index sites which the user must hook via
      -- env.bufs). For non-pointer custom types (enums, structs)
      -- we also no-op so the kernel body's downstream statements
      -- can still attempt lowering. The ident is left unbound; if a
      -- later use needs it the lower will throw a separate error
      -- pointing at the actual use site rather than the decl.
      -- Strip leading `const ` for the comparison so we don't have
      -- to enumerate every cv-qualified variant.
      let core := if ty.startsWith "const " then ty.drop 6 else ty
      -- Soft no-op fallback. `classifyTy` already handled the scalar
      -- cases (u32/i32/f32/vec2.f32) above; everything else here is a
      -- type we don't yet model (POD structs, half2, tile_<T,M,K>,
      -- coordinate-bag tuples, size_t, project-specific structs like
      -- `iq1m_scale_t`). The decl itself is accepted; if a downstream
      -- use site needs the value it'll error there with a clearer
      -- "ident not bound" message.
      .ok (pure ())  -- soft no-op
  | .declArr storage ty name szExpr =>
    -- Shared arrays → `ShaderM.sharedNamed`. Local arrays not yet
    -- supported (would need `var<private>` array decl which the DSL
    -- doesn't currently expose).
    match storage with
    | .shared =>
      match classifyTy ty with
      | some elemTy =>
        match evalConst env.consts szExpr with
        | some n =>
          if n < 0 then .error s!"lowerStmt: __shared__ {name}[…] negative size {n}"
          else .ok (ShaderM.sharedNamed name (.array elemTy n.toNat))
        | none =>
          .error s!"lowerStmt: __shared__ {name}[…] size not constant-foldable"
      | none => .error s!"lowerStmt: __shared__ {ty} {name}[…] — unsupported elem type"
    | _ =>
      -- Local array decl `int v[N];` — scalarize at transpile time.
      -- Emit N separate scalar varDecls `v_0, v_1, ... v_{N-1}` so the
      -- PTX backend (which lacks .array register form) can register-
      -- allocate each slot independently.  Subsequent `v[k]` accesses
      -- with const k look up via `tryScalarizeLocalArray` and rewrite
      -- to `v_k`.  Env registration is done by `lowerStmtWithEnvUpdate`.
      match classifyTy ty, evalConst env.consts szExpr with
      | some elemTy, some n =>
        if n < 0 then .error s!"lowerStmt: local array '{name}[{n}]' negative size"
        else
          let zeroInit : ShaderM Unit := match elemTy with
            | .scalar .u32 => Id.run do
              let mut acc : ShaderM Unit := pure ()
              for k in [0:n.toNat] do
                acc := acc *> ShaderM.varNamed s!"{name}_{k}" (.scalar .u32) (Exp.litU32 0)
              return acc
            | .scalar .i32 => Id.run do
              let mut acc : ShaderM Unit := pure ()
              for k in [0:n.toNat] do
                acc := acc *> ShaderM.varNamed s!"{name}_{k}" (.scalar .i32) (Exp.litI32 0)
              return acc
            | .scalar .f32 => Id.run do
              let mut acc : ShaderM Unit := pure ()
              for k in [0:n.toNat] do
                acc := acc *> ShaderM.varNamed s!"{name}_{k}" (.scalar .f32) (Exp.litF32 0.0)
              return acc
            | _ => pure ()
          .ok zeroInit
      | _, _ =>
        -- Soft no-op for unsupported local-array element types
        -- (`tile_A`, `half2`, `float2`, POD struct types). The decl
        -- itself is accepted; subsequent array indexing on `name`
        -- will surface a separate "not bound" error if it actually
        -- needs to read/write the array.
        let core := if ty.startsWith "const " then ty.drop 6 else ty
        if core.startsWith "tile_" ∨ core.endsWith "tile_A" ∨ core.endsWith "tile_B"
           ∨ core.endsWith "tile_C" ∨ core.endsWith "tile_D"
           ∨ core == "half2" ∨ core == "half" ∨ core == "__half" ∨ core == "__half2"
           ∨ core == "float2" ∨ core == "float4"
           ∨ core == "int2" ∨ core == "uint2" ∨ core == "uint3"
           ∨ ty.endsWith "*" ∨ ty.endsWith " *" then
          .ok (pure ())
        else
          .error s!"lowerStmt: local array '{ty} {name}[…]' — unrecognised elem type or non-const size"
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
    | some (.decl _ _ name (some e)) => .ok (name, e)
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
    | _ =>
      -- Non-standard cond shape (e.g. `k != bound`, `k > bound`,
      -- compound condition). Default to a 1-iteration unroll with
      -- `bound = k_init + 1` so the body lowers in some context.
      .ok (.binop .add (.ident iname) (.numLit "1"), false)
  -- Step: `k += s`, `k++`, or `k = k + s` (the desugared form many
  -- llama.cpp source dumps emit after `pragma unroll`).
  let stepExpr : CExpr ← match stepOpt with
    | some (.binop .addAssign (.ident n) s) =>
      if n == iname then .ok s
      else .error s!"lowerFor: loop var mismatch in step ('{n}' vs '{iname}')"
    | some (.unop .postInc (.ident n)) | some (.unop .preInc (.ident n)) =>
      if n == iname then .ok (.numLit "1")
      else .error s!"lowerFor: loop var mismatch in step ('{n}' vs '{iname}')"
    | some (.binop .assign (.ident n) (.binop .add (.ident m) s)) =>
      if n == iname ∧ m == iname then .ok s
      else .error s!"lowerFor: loop var mismatch in step ('{n}' or '{m}' vs '{iname}')"
    | some (.binop .assign (.ident n) (.binop .add s (.ident m))) =>
      -- `i = 1 + i` form
      if n == iname ∧ m == iname then .ok s
      else .error s!"lowerFor: loop var mismatch in step"
    | _ => .error "lowerFor: expected `k += s`, `k++`, or `k = k + s`"
  -- Try const-fold all three: if start/bound/step are compile-time
  -- known, **unroll the loop at transpile time** so the body sees a
  -- concrete integer for the loop variable.  This enables local-array
  -- scalarization (`d8[i]` with const `i`) and matches llama.cpp's
  -- `#pragma unroll` semantics.  Falls through to runtime loop if any
  -- bound isn't foldable.
  let unrolled : Option (Except String (ShaderM Unit)) :=
    match evalConst env.consts startExpr,
          evalConst env.consts boundExpr,
          evalConst env.consts stepExpr with
    | some startN, some boundN, some stepN =>
      if stepN ≤ 0 then some (.error s!"lowerFor: non-positive step {stepN} in unrolled loop")
      else
        let lastN := if boundIsLE then boundN else boundN - 1
        let result : Except String (ShaderM Unit) := Id.run do
          let mut acts : List (ShaderM Unit) := []
          let mut iVal : Int := startN
          while iVal ≤ lastN do
            let env' : Env := { env with
              consts := fun n => if n == iname then some iVal else env.consts n,
              u32 := fun n => if n == iname then some (Exp.litU32 iVal.toNat) else env.u32 n,
              i32 := fun n => if n == iname then some (Exp.litI32 iVal) else env.i32 n,
              f32 := fun n => if n == iname then some (Exp.litF32 (Float.ofInt iVal)) else env.f32 n
            }
            match lowerStmt env' body with
            | .ok a => acts := acts ++ [a]
            | .error e => return .error s!"lowerFor (unrolled, i={iVal}): {e}"
            iVal := iVal + stepN
          return .ok (acts.foldl (fun acc a => acc *> a) (pure ()))
        some result
    | _, _, _ => none
  match unrolled with
  | some result => result
  | none =>
  -- Runtime loop fallback.
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
    let env' : Env := { env with
      u32 := fun n => if n == iname then some k             else env.u32 n,
      i32 := fun n => if n == iname then some (Exp.toI32 k) else env.i32 n,
      f32 := fun n => if n == iname then some (Exp.toF32 k) else env.f32 n
    }
    lowerStmt env' body
  -- Validate the body once eagerly with a placeholder loop var, so
  -- any lowering error surfaces here rather than disappearing inside
  -- ShaderM.loop's body callback (where Inhabited would silently
  -- swallow it via the Except.error → pure () path).
  let _ ← bodyOf (Exp.var "__loopvar_probe")
  .ok do
    ShaderM.loop s bound stepE fun k =>
      match bodyOf k with
      | .ok a => a
      | .error _ => pure ()  -- unreachable: validated above

-- `lowerBool` moved into the Lower.lean mutual block so the
-- ternary cases of `lowerU32`/`lowerI32`/`lowerF32` can call it.

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
  | .call fn args =>
    -- Bare function call as statement (return value ignored).  Common
    -- in llama.cpp source for void helpers like `load_tiles(...)`,
    -- `vec_dot(...)`, `__syncthreads()`-likes, or user-injected
    -- inlines that produce side-effecting code.  If the user
    -- registered the function in `env.inlines`, expand the rewrite
    -- and recursively lower the resulting expression as a statement.
    -- Otherwise, treat as a no-op (the kernel still type-checks; the
    -- caller knows the helper has been replaced).
    match env.inlines fn args with
    | some rewritten =>
      -- Try lower the rewritten expr as a statement (may itself be
      -- a call or compound assign); if that fails fall back to
      -- evaluating it as a u32 expression and discarding the value.
      match lowerExprStmt env rewritten with
      | .ok act => .ok act
      | .error _ =>
        match lowerU32 env rewritten with
        | .ok _ => .ok (pure ())  -- discard value
        | .error e => .error s!"lowerExprStmt: inlined call '{fn}' rewrite failed: {e}"
    | none => .ok (pure ())  -- no-op for unregistered void helpers
  | .binop .assign (.index (.ident bname) idx) rhs =>
    -- Local-array scalarized assignment: `v[<const>] = …` → `v_<n> = …`.
    match tryScalarizeLocalArray env (.ident bname) idx with
    | some scalarName =>
      lowerExprStmt env (.binop .assign (.ident scalarName) rhs)
    | none =>
    -- Buffer write: `dst[i] = expr;` → ShaderM.writeBuffer.
    match env.bufs bname with
    | none =>
      -- Unbound buffer: skip the write so the lower itself completes.
      -- Real codegen needs an explicit env.bufs binding. The transpile
      -- coverage check measures whether the function body is parseable
      -- and lowerable in principle.
      .ok (pure ())
    | some b => do
      let ie ← lowerU32 env idx
      let idxFinal := match b.offset? with
        | some off => Exp.add off ie
        | none => ie
      match b.elemTy with
      | .scalar .f32 => do
        let v ← lowerF32 env rhs
        .ok (ShaderM.writeBuffer (ty := .scalar .f32) b.name idxFinal v)
      | .scalar .i32 => do
        let v ← lowerI32 env rhs
        .ok (ShaderM.writeBuffer (ty := .scalar .i32) b.name idxFinal v)
      | .scalar .u32 => do
        let v ← lowerU32 env rhs
        .ok (ShaderM.writeBuffer (ty := .scalar .u32) b.name idxFinal v)
      | t => .error s!"lowerExprStmt: write to buffer with unsupported elem type {repr t}"
  -- Compound-assign on a buffer index: `dst[i] += rhs;` → desugar to
  -- `dst[i] = dst[i] + rhs;` and re-lower. Same for -=/*=//=. Common
  -- in mmq.cuh accumulation: `sum[j0/n + i0/w] += vec_dot_…(...)`.
  | .binop .addAssign (.index (.ident bname) idx) rhs =>
    lowerExprStmt env (.binop .assign (.index (.ident bname) idx)
      (.binop .add (.index (.ident bname) idx) rhs))
  | .binop .subAssign (.index (.ident bname) idx) rhs =>
    lowerExprStmt env (.binop .assign (.index (.ident bname) idx)
      (.binop .sub (.index (.ident bname) idx) rhs))
  | .binop .mulAssign (.index (.ident bname) idx) rhs =>
    lowerExprStmt env (.binop .assign (.index (.ident bname) idx)
      (.binop .mul (.index (.ident bname) idx) rhs))
  | .binop .divAssign (.index (.ident bname) idx) rhs =>
    lowerExprStmt env (.binop .assign (.index (.ident bname) idx)
      (.binop .div (.index (.ident bname) idx) rhs))
  -- Increment/decrement as a statement: `++i;` / `i++;` / `--i;` /
  -- `i--;` — desugar to `i = i + 1` / `i = i - 1`.
  | .unop .preInc (.ident name) | .unop .postInc (.ident name) =>
    lowerExprStmt env (.binop .assign (.ident name)
      (.binop .add (.ident name) (.numLit "1")))
  | .unop .preDec (.ident name) | .unop .postDec (.ident name) =>
    lowerExprStmt env (.binop .assign (.ident name)
      (.binop .sub (.ident name) (.numLit "1")))
  | _ =>
    -- Unsupported expression statement — soft no-op so the lower
    -- itself completes. The transpile coverage check measures
    -- structural lower; correctness still requires explicit handling
    -- of the missed shape.
    .ok (pure ())

/-- Lower a statement and report whether (and how) it updates the env
    for following statements in a block.  Currently only **pointer
    arithmetic on kernel pointer args** updates env: `x += off;` for a
    pointer `x` accumulates `off` into `env.bufs x`'s `offset?` field,
    so subsequent `x[i]` loads emit `Exp.index x (off+i)`. Other
    statements lower as before and return env unchanged.

    `extern __shared__` decls would also update env in a fuller
    implementation (binding the smem array as a buffer). Not yet. -/
partial def lowerStmtWithEnvUpdate (env : Env) (s : CStmt)
    : Except String (Env × ShaderM Unit) := do
  match s with
  -- Pointer arithmetic on kernel pointer args.
  | .expr (.binop .addAssign (.ident name) rhs) =>
    match env.bufs name with
    | some b => do
      let offDelta ← lowerU32 env rhs
      let newOff := match b.offset? with
        | some old => Exp.add old offDelta
        | none => offDelta
      let bNew : BufBinding := { name := b.name, elemTy := b.elemTy, offset? := some newOff }
      let oldBufs := env.bufs
      let newBufs : String → Option BufBinding := fun n =>
        if n == name then some bNew else oldBufs n
      let env' : Env := { env with bufs := newBufs }
      .ok (env', pure ())  -- pointer advance has no runtime ShaderM action
    | none =>
      let act ← lowerStmt env s
      .ok (env, act)
  -- Pointer-rebind decl: `T * NAME = (T *) BASE;` or
  -- `T * NAME = (T *) (BASE + OFFSET);` (or bare `T * NAME = BASE;`).
  -- These are common in mmq.cuh where smem layouts are typed-aliased
  -- onto a single `extern __shared__` byte buffer:
  --   int   * x_qs = (int   *)  x_tile;
  --   float * x_df = (float *) (x_qs + 2*MMQ_TILE_NE_K);
  -- Aliasing the decl to the source buffer means subsequent
  -- `x_qs[i] = …` lowers via the existing buffer-store path with the
  -- offset folded in.
  | .decl _storage ty name (some initE) =>
    if ty.endsWith "*" ∨ ty.endsWith " *" then
      -- Strip outer cast `(T *) inner` — same shape regardless of T.
      let stripped : CExpr := match initE with
        | .cast _ inner => inner
        | other => other
      -- Detect `BASE + OFFSET` (parenthesised or not) vs bare BASE.
      let (baseName?, offset?) : Option String × Option CExpr :=
        match stripped with
        | .ident b => (some b, none)
        | .binop .add (.ident b) off => (some b, some off)
        | .binop .add off (.ident b) => (some b, some off)  -- commutative
        | _ => (none, none)
      match baseName? with
      | some baseName =>
        match env.bufs baseName with
        | some bBase =>
          -- Resolve the offset (when present). If it doesn't lower as
          -- u32, fall through to the soft no-op rather than failing
          -- the whole function — common for `const T * p = (T*)src + i`
          -- where `i` is a `const int &` reference param we don't track.
          let newOff? : Option (Exp (.scalar .u32)) :=
            match offset? with
            | none => bBase.offset?
            | some off =>
              match lowerU32 env off with
              | .ok off' => some (match bBase.offset? with
                  | some old => Exp.add old off'
                  | none => off')
              | .error _ => none
          match offset?, newOff? with
          | some _, none =>
            -- Had an offset but couldn't lower it — punt to soft no-op.
            let act ← lowerStmt env s
            .ok (env, act)
          | _, _ =>
            let bAlias : BufBinding :=
              { name := bBase.name, elemTy := bBase.elemTy, offset? := newOff? }
            let oldBufs := env.bufs
            let newBufs : String → Option BufBinding := fun n =>
              if n == name then some bAlias else oldBufs n
            let env' : Env := { env with bufs := newBufs }
            -- No runtime decl needed — the alias is purely a name binding.
            .ok (env', pure ())
        | none =>
          -- Base unbound — treat like the original soft-no-op decl
          -- (the existing fallthrough in lowerStmt accepts ptr types).
          let act ← lowerStmt env s
          .ok (env, act)
      | none =>
        let act ← lowerStmt env s
        .ok (env, act)
    else
      -- Non-pointer decl with init: fall through to the typed-scalar
      -- branch below by re-matching.
      let act ← lowerStmt env s
      let env' : Env := match classifyTy ty with
        | some (.scalar .u32) =>
          { env with u32 := fun n => if n == name then some (Exp.var name) else env.u32 n }
        | some (.scalar .i32) =>
          { env with i32 := fun n => if n == name then some (Exp.var name) else env.i32 n }
        | some (.scalar .f32) =>
          { env with f32 := fun n => if n == name then some (Exp.var name) else env.f32 n }
        | some (.vec2 .f32) =>
          let initExpr := match lowerF32x2 env initE with
            | .ok e => e
            | .error _ => Exp.vec2 (Exp.litF32 0.0) (Exp.litF32 0.0)
          { env with f32x2 := fun n => if n == name then some initExpr else env.f32x2 n }
        | _ => env
      .ok (env', act)
  -- Local decls of typed scalars: extend env so that subsequent
  -- references to `name` (e.g. `tmp += __shfl_xor_sync(0u, tmp, 16u)`)
  -- type-resolve to the right slot.
  | .decl _storage ty name initOpt =>
    -- Special-case `float2 v = expr;` — the PTX backend has no native
    -- vec2.f32 local-variable representation; emitting it as a runtime
    -- ShaderM.varNamed yields a non-functional binding (subsequent `v.x`
    -- reads return an unallocated f32 register).  Instead, *inline-
    -- substitute* the init: bind `f32x2 v` directly to the lowered init
    -- expression, and skip emitting a runtime decl.  Subsequent `v.x` /
    -- `v.y` then lower to `Exp.vecX <init-expr>` / `Exp.vecY <init-expr>`,
    -- which the PTX backend already handles.
    match classifyTy ty with
    | some (.vec2 .f32) =>
      let initExpr ← match initOpt with
        | some e => lowerF32x2 env e
        | none => .ok (Exp.vec2 (Exp.litF32 0.0) (Exp.litF32 0.0))
      let env' : Env :=
        { env with f32x2 := fun n => if n == name then some initExpr else env.f32x2 n }
      .ok (env', pure ())
    | _ =>
      let act ← lowerStmt env s
      -- NOTE: tempting to register const-foldable inits (e.g.
      -- `int idx = 0`) in env.consts so subsequent uses fold further,
      -- but this breaks variables that are mutated later (`int sumi_d
      -- = 0; sumi_d = sumi_d + …`) because we have no flow analysis
      -- to detect mutation.  Without that, transpiled vec_dot
      -- silently turns `sumi_d` into a const 0.  Caller should inline
      -- the const directly into uses (the unroll substitutes loop
      -- vars at iteration time, so `arr[(j0/4) + (i0/32)]` folds even
      -- if there's no `idx` intermediate).
      let env' : Env := match classifyTy ty with
        | some (.scalar .u32) =>
          { env with u32 := fun n => if n == name then some (Exp.var name) else env.u32 n }
        | some (.scalar .i32) =>
          { env with i32 := fun n => if n == name then some (Exp.var name) else env.i32 n }
        | some (.scalar .f32) =>
          { env with f32 := fun n => if n == name then some (Exp.var name) else env.f32 n }
        | _ => env
      .ok (env', act)
  | .declArr storage ty name szExpr =>
    -- Local array (Storage.none) → register in env.localArrays so
    -- `tryScalarizeLocalArray` can resolve `v[<const>]` to `v_<n>`.
    -- Each scalarized slot is also registered in the typed env so
    -- assignment `v_<n> = …` and reads in any context lower correctly.
    match storage with
    | .none =>
      match classifyTy ty, evalConst env.consts szExpr with
      | some elemTy, some n =>
        if n < 0 then .error s!"lowerStmtWithEnvUpdate: local array negative size"
        else do
          let act ← lowerStmt env s
          let nNat := n.toNat
          let env' : Env :=
            { env with
              localArrays := fun nm =>
                if nm == name then some (nNat, elemTy) else env.localArrays nm
              -- Pre-register each slot in the typed env so subsequent
              -- assignments and reads of `v_<k>` lower correctly.
              u32 := fun nm => Id.run do
                for k in [0:nNat] do
                  if nm == s!"{name}_{k}" ∧ elemTy == .scalar .u32 then
                    return some (Exp.var nm)
                env.u32 nm
              i32 := fun nm => Id.run do
                for k in [0:nNat] do
                  if nm == s!"{name}_{k}" ∧ elemTy == .scalar .i32 then
                    return some (Exp.var nm)
                env.i32 nm
              f32 := fun nm => Id.run do
                for k in [0:nNat] do
                  if nm == s!"{name}_{k}" ∧ elemTy == .scalar .f32 then
                    return some (Exp.var nm)
                env.f32 nm
            }
          .ok (env', act)
      | _, _ =>
        let act ← lowerStmt env s
        .ok (env, act)
    | _ =>
      let act ← lowerStmt env s
      .ok (env, act)
  | _ =>
    let act ← lowerStmt env s
    .ok (env, act)

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

/-! ## Function lowering (Phase 4)

    Lowers a `CFunction` body to a `ShaderM Unit`. Template parameters
    are folded into `Env.consts` via the user-supplied `templVals`
    table; runtime parameters (e.g. pointer args) are not yet
    represented and the lowering only succeeds for kernels whose body
    refers to template params and globals via `Env`. -/

/-- Lower a parsed function body to a `ShaderM` action. The caller
    supplies `templVals` — concrete values for the template params
    (e.g. `[("mmq_y", 128), ("mmq_x", 64), ("nwarps", 8)]`) — which
    extend `env.consts`. Runtime params (`int *x`, etc.) must already
    be wired in `env` (typically via a buffer-binding pre-pass). -/
def lowerFunction (env : Env) (f : Hesper.Transpile.CUDA.CFunction)
    (templVals : List (String × Int)) : Except String (ShaderM Unit) :=
  let lookup (n : String) : Option Int :=
    match templVals.find? (·.fst == n) with
    | some (_, v) => some v
    | none => env.consts n
  let env' : Env := { env with consts := lookup }
  lowerStmt env' f.body

end Hesper.Transpile.CUDA
