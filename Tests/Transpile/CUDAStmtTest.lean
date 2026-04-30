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

/-- Render the WGSL body emitted by a ShaderM action. Includes
    `var<workgroup>` declarations in the output so tests can observe
    `__shared__` lowering. -/
def renderShader (m : ShaderM Unit) : String :=
  let st := ShaderM.exec m
  let sharedDecls : String :=
    st.sharedVars.foldl (init := "") fun acc (n, t) =>
      acc ++ s!"var<workgroup> {n}: {t.toWGSL};\n"
  let body : String := String.join (st.stmts.map (·.toWGSL 0))
  sharedDecls ++ body

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
  -- Phase 9: const-foldable bounds → unroll at transpile time.
  -- The body is duplicated 4 times with k = 0, 1, 2, 3 substituted.
  -- This matches `#pragma unroll` semantics in CUDA.  Runtime-loop
  -- form preserved for non-const bounds.
  assertWGSL "for-loop sum (Phase 9 unrolled)"
    { i32 := fun n => if n == "sum" then some (Exp.var "sum") else none }
    "for (int k = 0; k < 4; k += 1) { sum += k; }"
    "sum = (sum + 0i);
     sum = (sum + 1i);
     sum = (sum + 2i);
     sum = (sum + 3i);"

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

  IO.println ""
  IO.println "=== Phase 4: __shared__ smem decls ==="

  -- 7. simple __shared__ array with literal size.
  assertWGSL "shared int x[128]" Env.empty
    "__shared__ int x[128];"
    "var<workgroup> x: array<i32, 128>;"

  -- 8. shared array with constexpr size:
  --      mmq_y = 128, MMQ_TILE_NE_K = 32
  --      __shared__ int x_qs[mmq_y * (MMQ_TILE_NE_K + 1)];
  -- expected size = 128 * 33 = 4224.
  let constsEnv : Env := {
    consts := fun n =>
      if n == "mmq_y"          then some 128
      else if n == "MMQ_TILE_NE_K" then some 32
      else none
  }
  assertWGSL "shared with constexpr size" constsEnv
    "__shared__ int x_qs[mmq_y * (MMQ_TILE_NE_K + 1)];"
    "var<workgroup> x_qs: array<i32, 4224>;"

  -- 9. shared float array (RMSNorm-style smem reduction buffer).
  assertWGSL "shared float buf" Env.empty
    "__shared__ float buf[256];"
    "var<workgroup> buf: array<f32, 256>;"

  -- 10. function transpile — template params fold into smem size.
  -- This mirrors the shape of llama.cpp's `mul_mat_q_process_tile`:
  --
  --   template <int mmq_y, int mmq_x>
  --   __device__ void foo() {
  --     __shared__ int x_qs[mmq_y * (mmq_x + 1)];
  --     __syncthreads();
  --   }
  --
  -- With mmq_y=128, mmq_x=32 → 128*33 = 4224.
  let funcSrc :=
    "template <int mmq_y, int mmq_x> __device__ void foo() {
       __shared__ int x_qs[mmq_y * (mmq_x + 1)];
       __syncthreads();
     }"
  match parseFunctionStr funcSrc with
  | .ok f =>
    match lowerFunction Env.empty f [("mmq_y", 128), ("mmq_x", 32)] with
    | .ok action =>
      let normalize (s : String) : String :=
        s.splitOn "\n" |>.map (·.trim) |>.filter (· ≠ "") |> String.intercalate "\n"
      let g := normalize (renderShader action)
      let e := normalize "var<workgroup> x_qs: array<i32, 4224>;
        workgroupBarrier();"
      if g = e then IO.println "PASS  function transpile (template + smem)"
      else
        IO.println "FAIL  function transpile (template + smem)"
        IO.println s!"  got:\n{g}"
        IO.println s!"  expected:\n{e}"
    | .error err => IO.println s!"FAIL  function lowerFunction: {err}"
  | .error err => IO.println s!"FAIL  function parse: {err}"

  IO.println "=== done ==="

end Hesper.Transpile.CUDA.StmtTest

def main : IO Unit := Hesper.Transpile.CUDA.StmtTest.main
