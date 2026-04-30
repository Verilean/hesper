import Hesper.Transpile.CUDA
import Hesper.Transpile.CUDA.Parse
import Hesper.Transpile.CUDA.LowerStmt
import Hesper.WGSL.Monad
import Hesper.WGSL.Shader

/-! # Phase 9 smoke: vec_dot_q4_K_q8_1 — full body verbatim

This is the *full* outer wrapper from llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh:816.
Constants pre-substituted: QR4_K=2, QI8_1=4 (these are #defines that
the parser doesn't evaluate; the user expands them in env.consts or
inlines manually).

The body exercises every Phase 8 feature plus the ones we still need:
  - local arrays:  `int v[2]; int u[4]; float d8[2];`
  - struct member with arithmetic: `bq4_K->qs + 16*bq8_offset + 4*((iqs/2)%4)`
  - typed pointer cast: `(const int *)(...)` then `q4[0]`, `q4[4]`
  - second struct: `bq8_1 + bq8_offset + i` then `bq8i->ds`, `bq8i->qs`
  - inline call: `vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8)`
    via Env.inlines
-/
namespace Hesper.Transpile.CUDA.Q4KFullWrapperSmoke

open Hesper.WGSL Hesper.WGSL.Monad Hesper.Transpile.CUDA

/-- Distilled wrapper: QR4_K, QI8_1 expanded; comments stripped.
    sc/m treated as flat u32 for now (Phase 9b will handle the
    `(const uint8_t *)aux` cast properly). -/
def fullBody : String :=
"{
  int v[2];
  int u[4];
  float d8[2];

  const int bq8_offset = 2 * ((iqs/2) / 2);

  v[0] = bq4_K_qs_int[0];
  v[1] = bq4_K_qs_int[4];

  uint16_t aux0;
  uint16_t aux1;
  const int j = bq8_offset / 2;
  if (j < 2) {
    aux0 = bq4_K_scales_u16[j + 0] & 0x3f3f;
    aux1 = bq4_K_scales_u16[j + 2] & 0x3f3f;
  } else {
    aux0 = ((bq4_K_scales_u16[j + 2] >> 0) & 0x0f0f) | ((bq4_K_scales_u16[j - 2] & 0xc0c0) >> 2);
    aux1 = ((bq4_K_scales_u16[j + 2] >> 4) & 0x0f0f) | ((bq4_K_scales_u16[j - 0] & 0xc0c0) >> 2);
  }

  for (int i = 0; i < 2; i = i + 1) {
    d8[i] = bq8_d_buf[bq8_offset + i];
    u[2*i + 0] = bq8_qs_int_buf[(bq8_offset + i) * 8 + 0];
    u[2*i + 1] = bq8_qs_int_buf[(bq8_offset + i) * 8 + 4];
  }

  result_packed = bq4_K->dm;
  result_v0 = v[0];
  result_v1 = v[1];
  result_u0 = u[0];
  result_d80 = d8[0];
  result_aux0 = aux0;
  result_aux1 = aux1;
}"

def envFor : Env := {
  bufs := fun n => match n with
    | "bq4_K_qs_int"     => some { name := "x_qs_buf",     elemTy := .scalar .i32 }
    | "bq4_K_scales_u16" => some { name := "x_scales_buf", elemTy := .scalar .u32 }
    | "bq8_d_buf"        => some { name := "y_d_buf",      elemTy := .scalar .f32 }
    | "bq8_qs_int_buf"   => some { name := "y_qs_buf",     elemTy := .scalar .i32 }
    | _ => none
  i32 := fun n => match n with
    | "iqs" => some (Exp.var "iqs")
    | _ => none
  f32 := fun n => match n with
    | "result_v0" | "result_v1" | "result_u0" | "result_d80"
    | "result_aux0" | "result_aux1"
      => some (Exp.var n)
    | _ => none
  u32 := fun n => match n with
    | "result_packed" => some (Exp.var "result_packed")
    | _ => none
  structFieldU32 := fun base field => match base, field with
    | "bq4_K", "dm" => some (Exp.var "bq4_K_dm_packed_load")
    | _, _ => none
}

def renderShader (m : ShaderM Unit) : String :=
  let st := ShaderM.exec m
  String.join (st.stmts.map (·.toWGSL 0))

def main : IO Unit := do
  IO.println "═══ Phase 9: vec_dot_q4_K_q8_1 full wrapper ═══"
  match parseStmtStr fullBody with
  | .error e => IO.println s!"PARSE ERROR: {e}"
  | .ok stmt =>
    IO.println "PARSE OK"
    match lowerStmt envFor stmt with
    | .error e => IO.println s!"LOWER ERROR: {e}"
    | .ok act =>
      IO.println "LOWER OK"
      IO.println "--- generated WGSL ---"
      IO.println (renderShader act)
      IO.println "═══ done ═══"

end Hesper.Transpile.CUDA.Q4KFullWrapperSmoke

def main : IO Unit := Hesper.Transpile.CUDA.Q4KFullWrapperSmoke.main
