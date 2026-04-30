/-! # CUDA → ShaderM transpiler — surface AST

A small AST covering the subset of CUDA C++ used by llama.cpp's
quantized-matmul kernels.

This is a "surface" AST — types are tracked as strings (e.g. `"int"`,
`"const int * __restrict__"`) and resolved later when lowering to
Hesper `Exp`/`ShaderM`. Template arguments (e.g. `mmq_y`, `mmq_x`)
appear as plain identifiers; constexpr-eval handles them at lowering
time.
-/
namespace Hesper.Transpile.CUDA

/-- Unary operators we encode. -/
inductive UnOp where
  | neg | bitNot | logNot | preInc | preDec | postInc | postDec | addrOf | deref
  deriving Repr, Inhabited, BEq

/-- Binary operators, including assignment. The lowering pass treats
    `assign` (and the compound forms) as `CStmt.assign` rather than a
    pure expression. -/
inductive BinOp where
  | add | sub | mul | div | mod
  | shl | shr
  | bitAnd | bitOr | bitXor
  | logAnd | logOr
  | eq | ne | lt | le | gt | ge
  | assign
  | addAssign | subAssign | mulAssign | divAssign | modAssign
  | shlAssign | shrAssign
  | andAssign | orAssign  | xorAssign
  deriving Repr, Inhabited, BEq

/-- Expression AST. -/
inductive CExpr where
  | numLit  (s : String)              -- raw text from lexer ("0x0F", "42u", ...)
  | floatLit (s : String)
  | ident   (name : String)
  | unop    (op : UnOp) (e : CExpr)
  | binop   (op : BinOp) (a b : CExpr)
  | ternary (c t e : CExpr)
  | index   (arr : CExpr) (i : CExpr) -- `a[i]`
  | member  (obj : CExpr) (f : String) -- `a.f`
  | arrow   (obj : CExpr) (f : String) -- `a->f`
  | call    (fn : String) (args : Array CExpr)
  | cast    (ty : String) (e : CExpr) -- `(int) e` — type kept as string
  | comma   (a b : CExpr)
  deriving Repr, Inhabited

/-- Storage qualifier for declarations. `none` = ordinary local /
    register; `shared` = CUDA `__shared__` (workgroup memory in WGSL);
    `constant` = CUDA `__constant__` (uniform in WGSL). -/
inductive Storage where
  | none
  | shared
  | constant
  deriving Repr, Inhabited, BEq

/-- Statement AST. -/
inductive CStmt where
  | expr      (e : CExpr)
  | decl      (storage : Storage) (ty : String) (name : String) (init : Option CExpr)
  | declArr   (storage : Storage) (ty : String) (name : String) (sizeExpr : CExpr)
  | block     (stmts : Array CStmt)
  | if_       (cond : CExpr) (thn : CStmt) (els : Option CStmt)
  | for_      (init : Option CStmt) (cond : Option CExpr) (step : Option CExpr) (body : CStmt)
  | while_    (cond : CExpr) (body : CStmt)
  | return_   (e : Option CExpr)
  | break_
  | continue_
  | sync_                    -- `__syncthreads()`
  | pragma    (s : String)   -- `#pragma unroll` etc.
  /-- `if constexpr (cond) thn [else els]` — branch is selected at
      lowering time using compile-time constants (template params). -/
  | ifConstexpr (cond : CExpr) (thn : CStmt) (els : Option CStmt)
  /-- `extern __shared__ T name[];` — runtime-sized smem array. The
      element type is captured; the size is supplied at launch time and
      the lowering binds it to a ShaderM buffer of unknown extent. -/
  | externSharedArr (ty : String) (name : String)
  /-- `static_assert(...)` — accepted and dropped (compile-time only). -/
  | staticAssert
  deriving Repr, Inhabited

/-- A function parameter (type kept as string; resolved at lowering).
    `default?` captures CUDA default arguments (`= nullptr`,
    `= make_uint3(0,0,0)`) so the parser can accept them and the
    lowering can ignore irrelevant defaulted ptr/struct args. -/
structure CParam where
  ty : String
  name : String
  default? : Option CExpr := none
  deriving Repr, Inhabited

/-- A template parameter, e.g. `template <int mmq_y, bool need_check>`. -/
structure CTemplParam where
  ty : String                 -- "int", "bool", "ggml_type", ...
  name : String
  deriving Repr, Inhabited

/-- Function definition. -/
structure CFunction where
  attrs       : Array String  -- e.g. `__device__`, `__forceinline__`, `__global__`
  templParams : Array CTemplParam
  retTy       : String
  name        : String
  params      : Array CParam
  body        : CStmt
  deriving Repr, Inhabited

end Hesper.Transpile.CUDA
