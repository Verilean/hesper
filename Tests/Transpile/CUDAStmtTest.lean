import Hesper.Transpile.CUDA
import Hesper.Transpile.CUDA.LowerStmt
import Hesper.Transpile.CUDA.Parse
import Hesper.WGSL.Monad
import Hesper.WGSL.Shader

/-! # Phase 3 statement transpiler test

Lex + parse + lower a CUDA statement (or block of statements) to a
`ShaderM Unit`, then run the action and compare the resulting WGSL.

Each test compares the *output WGSL string* against an expected
hand-written counterpart, since `ShaderM Unit` actions are opaque
functions but their effect (the emitted Stmts) is observable.
-/
namespace Hesper.Transpile.CUDA.StmtTest

open Hesper.WGSL Hesper.WGSL.Monad Hesper.Transpile.CUDA

/-- Render the WGSL body emitted by a ShaderM action. -/
def renderShader (m : ShaderM Unit) : String :=
  let st := ShaderM.exec m
  String.join (st.stmts.map (·.toWGSL 0))

/-- Parse + lower a CUDA stmt string and render its WGSL. Returns
    the rendered string, or an error message on failure. -/
def transpileStmt (env : Env) (src : String) : Except String String := do
  let cstmt ← parseStmtStr src
  let action ← lowerStmt env cstmt
  .ok (renderShader action)

def assertWGSL (label : String) (env : Env) (src expected : String) : IO Unit := do
  match transpileStmt env src with
  | .ok wgsl =>
    -- Trim whitespace per line for forgiving comparison.
    let normalize (s : String) : String :=
      s.splitOn "\n" |>.map (·.trim) |>.filter (· ≠ "") |> String.intercalate "\n"
    let g := normalize wgsl
    let e := normalize expected
    if g = e then
      IO.println s!"PASS  {label}"
    else
      IO.println s!"FAIL  {label}"
      IO.println s!"  CUDA src: {src}"
      IO.println s!"  got:\n{g}"
      IO.println s!"  expected:\n{e}"
  | .error err =>
    IO.println s!"FAIL  {label}: {err}"

def main : IO Unit := do
  IO.println "=== Phase 3 CUDA → ShaderM statement transpile tests ==="

  -- 1. __syncthreads()
  assertWGSL "__syncthreads" Env.empty
    "__syncthreads();"
    "workgroupBarrier();"

  -- 2. simple int decl
  assertWGSL "int decl"  Env.empty
    "int x = 42;"
    "var x: i32 = 42i;"

  -- 3. uint decl
  assertWGSL "uint decl" Env.empty
    "uint32_t y = 0x0F;"
    "var y: u32 = 15u;"

  -- 4. simple compound assign
  --    acc += a;   →   acc = acc + a;
  assertWGSL "acc += a" { f32 := fun n => if n == "acc" then some (Exp.var "acc") else none }
    "acc += a;"
    "acc = (acc + a);"

  -- 5. for-loop with int counter
  --    for (int k = 0; k < 4; k += 1) {
  --      sum += k;
  --    }
  -- ShaderM.loop generates a fresh loop var (i0) and we bind k → i0
  -- in the body env. sum is i32 so we cast u32 k → i32.
  assertWGSL "for-loop sum" { i32 := fun n => if n == "sum" then some (Exp.var "sum") else none }
    "for (int k = 0; k < 4; k += 1) { sum += k; }"
    "for (var i0: u32 = 0u; (i0 < 4u); i0 = (i0 + 1u)) {
       sum = (sum + i32(i0));
     }"

  -- 6. if-else
  assertWGSL "if/else"
    { i32 := fun n => if n == "y" then some (Exp.var "y") else none,
      u32 := fun n => if n == "x" then some (Exp.var "x") else none }
    "if (x < 4) { y = 1; } else { y = 2; }"
    "if ((x < 4u)) {
       y = 1i;
     } else {
       y = 2i;
     }"

  IO.println "=== done ==="

end Hesper.Transpile.CUDA.StmtTest

def main : IO Unit := Hesper.Transpile.CUDA.StmtTest.main
