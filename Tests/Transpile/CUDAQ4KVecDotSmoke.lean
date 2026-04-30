import Hesper.Transpile.CUDA
import Hesper.Transpile.CUDA.Parse
import Hesper.Transpile.CUDA.LowerStmt
import Hesper.WGSL.Monad
import Hesper.WGSL.Shader

/-! # Phase 7 smoke: transpile vec_dot_q4_K_q8_1_impl_mmq

Pulled verbatim from llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh:527.
This is the INNER-DOT body of llama.cpp's Q4_K mmq matmul — the
hot loop running ~1280 calls per Gemma 4 prefill token.

What the transpiler needs that's new vs Phase 6:
  - `uint8_t *` array indexing → byte load (Exp.readBufferByte)
  - `half2 *` array indexing → u32 read + unpack2x16float
  - `__half22float2(h)` → unpack2x16float
  - `float2.x`/`.y` member access → vec2 .x/.y
  - References (`const half2 & dm4`) — treat as scalar passed by value
  - `ggml_cuda_dp4a` (alias of `__dp4a`)
-/
namespace Hesper.Transpile.CUDA.Q4KVecDotSmoke

open Hesper.WGSL Hesper.WGSL.Monad Hesper.Transpile.CUDA

def renderShader (m : ShaderM Unit) : String :=
  let st := ShaderM.exec m
  let body : String := String.join (st.stmts.map (·.toWGSL 0))
  body

/-- The actual vec_dot body, hand-typed.  We've inlined macro values:
    QR4_K=2, VDR_Q4_K_Q8_1_MMQ=8, QI8_1=4 → outer loop count = 2*8/4 = 4. -/
def vecDotBody : String :=
"{
  float sumf_d = 0.0f;
  float sumf_m = 0.0f;

  for (int i = 0; i < 4; i = i + 1) {
    int sumi_d = 0;
    for (int j = 0; j < 4; j = j + 1) {
      sumi_d = ggml_cuda_dp4a((v[j] >> (4 * i)) & 0x0F0F0F0F, u[i * 4 + j], sumi_d);
    }

    const float2 ds8f = __half22float2(ds8[i]);

    sumf_d = sumf_d + ds8f.x * (sc[i] * sumi_d);
    sumf_m = sumf_m + ds8f.y * m[i];
  }

  const float2 dm4f = __half22float2(dm4);

  result = dm4f.x * sumf_d - dm4f.y * sumf_m;
}"

def envFor : Env := {
  bufs := fun n => match n with
    | "v"   => some { name := "v_buf",   elemTy := .scalar .i32 }
    | "u"   => some { name := "u_buf",   elemTy := .scalar .i32 }
    | "sc"  => some { name := "sc_buf",  elemTy := .scalar .u32 }  -- byte buffer; we pad to u32
    | "m"   => some { name := "m_buf",   elemTy := .scalar .u32 }
    | "ds8" => some { name := "ds8_buf", elemTy := .scalar .u32 }  -- half2 packed as u32
    | _ => none
  f32 := fun n => match n with
    | "result" => some (Exp.var "result")
    | "dm4"    => some (Exp.var "dm4_packed")  -- half2 dm4 we pass as packed u32 → unpack
    | _ => none
}

def main : IO Unit := do
  IO.println "=== Phase 7 smoke: vec_dot_q4_K_q8_1_impl_mmq ==="
  match parseStmtStr vecDotBody with
  | .error e =>
    IO.println s!"PARSE ERROR: {e}"
  | .ok stmt =>
    IO.println "PARSE OK"
    match lowerStmt envFor stmt with
    | .error e =>
      IO.println s!"LOWER ERROR: {e}"
    | .ok act =>
      IO.println "LOWER OK"
      IO.println "--- generated WGSL body ---"
      IO.println (renderShader act)
      IO.println "=== done ==="

end Hesper.Transpile.CUDA.Q4KVecDotSmoke

def main : IO Unit := Hesper.Transpile.CUDA.Q4KVecDotSmoke.main
