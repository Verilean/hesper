import Hesper.Transpile.CUDA
import Hesper.Transpile.CUDA.Parse
import Hesper.Transpile.CUDA.LowerStmt
import Hesper.WGSL.Monad
import Hesper.WGSL.Shader

/-! # Phase 5 smoke: how close can we get to a real llama.cpp helper?

We attempt to transpile the inner-loop body of
`vec_dot_q4_K_q8_1_impl_vmmq` (vecdotq.cuh:502). The original signature
takes pointer params (`const int * v`, `const int * u`, ...) which the
Phase 4 transpiler cannot bind yet — so we *manually* substitute scalar
forms of those array reads in the source string and check whether the
*pure compute* lowers cleanly.

This is a "does it actually work?" reality check, not a parity test.
-/
namespace Hesper.Transpile.CUDA.VecDotSmoke

open Hesper.WGSL Hesper.WGSL.Monad Hesper.Transpile.CUDA

def renderShader (m : ShaderM Unit) : String :=
  let st := ShaderM.exec m
  let sharedDecls : String :=
    st.sharedVars.foldl (init := "") fun acc (n, t) =>
      acc ++ s!"var<workgroup> {n}: {t.toWGSL};\n"
  let body : String := String.join (st.stmts.map (·.toWGSL 0))
  sharedDecls ++ body

/- Body of vec_dot_q4_K_q8_1_impl_vmmq — array refs flattened to env
   scalars. We test the i=0 iteration only (no `pragma unroll for`). -/
def vmmqInner : String := "{
  int v0i = (v0 >> 0) & 0x0F0F0F0F;
  int v1i = (v1 >> 0) & 0x0F0F0F0F;
  int dot1 = __dp4a(v1i, u1, __dp4a(v0i, u0, 0));
  int dot2 = __dp4a(0x01010101, u1, __dp4a(0x01010101, u0, 0));
  sumf_d += d8 * dot1;
  sumf_m += d8 * dot2;
}"

/- Outer loop with QR4_K=2 unrolled via for. Tests for-loop + __dp4a
   together — this is the actual control structure we'd transpile. -/
def vmmqWithLoop : String := "
{
  for (int i = 0; i < 2; i = i + 1) {
    int v0i = v0 & 0x0F0F0F0F;
    int dot1 = __dp4a(v0i, u0, 0);
    sumf_d += d8 * dot1;
  }
}"

def envFor : Env := {
  i32 := fun n => match n with
    | "v0" | "v1" | "u0" | "u1" => some (Exp.var n)
    | _ => none
  f32 := fun n => match n with
    | "d8" | "sumf_d" | "sumf_m" => some (Exp.var n)
    | _ => none
}

def tryTranspile (label : String) (src : String) : IO Unit := do
  IO.println s!"--- {label} ---"
  match parseStmtStr src with
  | .ok stmt =>
    match lowerStmt envFor stmt with
    | .ok act =>
      IO.println "PARSE+LOWER OK"
      IO.println (renderShader act)
    | .error e => IO.println s!"LOWER ERROR: {e}"
  | .error e => IO.println s!"PARSE ERROR: {e}"
  IO.println ""

/- The actual llama.cpp body, with `int *v`, `int *u` bound as
   buffers via `Env.bufs`. This is much closer to a real port. -/
def vmmqRealPointers : String := "
{
  int v0i = (v[0] >> 0) & 0x0F0F0F0F;
  int v1i = (v[1] >> 0) & 0x0F0F0F0F;
  int dot1 = __dp4a(v1i, u[1], __dp4a(v0i, u[0], 0));
  sumf_d += d8 * dot1;
}
"

/- The full vec_dot_q4_K_q8_1_impl_vmmq inner accumulator loop
   (vecdotq.cuh:506-519). QR4_K is folded to 2 via Env.consts.
   Note: we drop the `#pragma unroll` (transpiler accepts it as a
   no-op) and the trailing `__half22float2(dm4)` post-loop math
   (member access not yet supported — DSL gap candidate). -/
def vmmqFull : String := "
{
  for (int i = 0; i < QR4_K; i = i + 1) {
    int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
    int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;
    int dot1 = __dp4a(v1i, u[2*i+1], __dp4a(v0i, u[2*i+0], 0));
    int dot2 = __dp4a(0x01010101, u[2*i+1], __dp4a(0x01010101, u[2*i+0], 0));
    sumf_d += d8[i] * (dot1 * sc[i]);
    sumf_m += d8[i] * (dot2 * mv[i]);
  }
}
"

def envWithBufs : Env := {
  bufs := fun n => match n with
    | "v" => some { name := "v_buf", elemTy := .scalar .i32 }
    | "u" => some { name := "u_buf", elemTy := .scalar .i32 }
    | _ => none
  f32 := fun n => match n with
    | "d8" | "sumf_d" => some (Exp.var n)
    | _ => none
}

/- Env for the full vmmq transpile. `v`, `u` are int* (i32 buffer);
   `sc`, `mv` are uint8_t* (u32 buffer with byte-packed values, but
   indexed as u32 here for simplicity); `d8` is float*. QR4_K is a
   compile-time constant from llama.cpp's common.cuh. -/
def envFull : Env := {
  bufs := fun n => match n with
    | "v"  => some { name := "v_buf",  elemTy := .scalar .i32 }
    | "u"  => some { name := "u_buf",  elemTy := .scalar .i32 }
    | "sc" => some { name := "sc_buf", elemTy := .scalar .u32 }
    | "mv" => some { name := "mv_buf", elemTy := .scalar .u32 }
    | "d8" => some { name := "d8_buf", elemTy := .scalar .f32 }
    | _ => none
  f32 := fun n => match n with
    | "sumf_d" | "sumf_m" => some (Exp.var n)
    | _ => none
  consts := fun n => if n == "QR4_K" then some 2 else none
}

def tryTranspileEnv (env : Env) (label : String) (src : String) : IO Unit := do
  IO.println s!"--- {label} ---"
  match parseStmtStr src with
  | .ok stmt =>
    match lowerStmt env stmt with
    | .ok act =>
      IO.println "PARSE+LOWER OK"
      IO.println (renderShader act)
    | .error e => IO.println s!"LOWER ERROR: {e}"
  | .error e => IO.println s!"PARSE ERROR: {e}"
  IO.println ""

/- Q6_K vec_dot inner body (vecdotq.cuh:625-638) — like vmmqFull but
   with `__vsubss4`. Tests Phase 5b's new transpiler primitive. -/
def q6kInner : String := "
{
  for (int i = 0; i < QR6_K; i = i + 1) {
    int sc = scales[4*i];
    int vil = (vl >> (4*i)) & 0x0F0F0F0F;
    int vih = ((vh >> (4*i)) << 4) & 0x30303030;
    int vi = __vsubss4(vil | vih, 0x20202020);
    sumf += d8[i] * (__dp4a(vi, u[i], 0) * sc);
  }
}
"

def envQ6K : Env := {
  bufs := fun n => match n with
    | "u"      => some { name := "u_buf",       elemTy := .scalar .i32 }
    | "scales" => some { name := "scales_buf",  elemTy := .scalar .u32 }
    | "d8"     => some { name := "d8_buf",      elemTy := .scalar .f32 }
    | _ => none
  i32 := fun n => match n with
    | "vl" | "vh" => some (Exp.var n)
    | _ => none
  f32 := fun n => match n with
    | "sumf" => some (Exp.var "sumf")
    | _ => none
  consts := fun n => if n == "QR6_K" then some 2 else none
}

def main : IO Unit := do
  IO.println "=== Phase 5 reality check ==="
  tryTranspile "vmmq inner (i=0 only, no loop)" vmmqInner
  tryTranspile "vmmq with for loop"             vmmqWithLoop
  tryTranspileEnv envWithBufs "vmmq with real pointer params" vmmqRealPointers
  tryTranspileEnv envFull "vmmq FULL (QR4_K loop, all pointers)" vmmqFull
  IO.println "--- Q6_K vec_dot mmvq (with __vsubss4) ---"
  tryTranspileEnv envQ6K "Q6_K full inner loop" q6kInner
  tryTranspileEnv envFull "FULL min: just one decl in loop"
    "{ for (int i = 0; i < QR4_K; i = i + 1) { int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F; } }"
  tryTranspileEnv envFull "FULL +1: two decls"
    "{ for (int i = 0; i < QR4_K; i = i + 1) { int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F; int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F; } }"
  tryTranspileEnv envFull "FULL +2: dp4a chained"
    "{ for (int i = 0; i < QR4_K; i = i + 1) { int v0i = v[0]; int dot1 = __dp4a(v0i, u[0], 0); } }"
  tryTranspileEnv envFull "FULL +3: f32 mul into compound +="
    "{ for (int i = 0; i < QR4_K; i = i + 1) { int dot1 = 0; sumf_d += d8[i] * dot1; } }"
  tryTranspileEnv envFull "FULL +4: u[2*i+0] index"
    "{ for (int i = 0; i < QR4_K; i = i + 1) { int dot1 = __dp4a(0, u[2*i+0], 0); } }"
  tryTranspileEnv envFull "FULL +5: dot1 * sc[i]"
    "{ for (int i = 0; i < QR4_K; i = i + 1) { int dot1 = 0; sumf_d += d8[i] * (dot1 * sc[i]); } }"
  IO.println "=== done ==="

end Hesper.Transpile.CUDA.VecDotSmoke

def main : IO Unit := Hesper.Transpile.CUDA.VecDotSmoke.main
