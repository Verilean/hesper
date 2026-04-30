import Hesper.Transpile.CUDA
import Hesper.Transpile.CUDA.Parse
import Hesper.Transpile.CUDA.LowerStmt
import Hesper.WGSL.Monad
import Hesper.WGSL.Shader

/-! # Phase 6 smoke: transpile llama.cpp rms_norm_f32

Goal: take a hand-flattened version of llama.cpp's `rms_norm_f32`
(template specialization `<block_size=1024, do_multiply=false,
do_add=false>`, single-channel single-sample, with `block_reduce`
inlined as a warp-shuffle reduce) and run it through the transpiler.

This exercises Phase 6 features:
  * `if constexpr (false)` blocks elided (do_multiply / do_add are 0)
  * pointer arithmetic on kernel args (`x += row * stride_row`)
  * builtin member access (`threadIdx.x`, `blockIdx.x`)
  * `rsqrtf` intrinsic
  * `__shfl_xor_sync` warp shuffle (manually written in source)
  * extern __shared__ (declared, kept tiny via `s_sum_size = 32`)

The static_assert is parsed and dropped.
-/
namespace Hesper.Transpile.CUDA.RmsNormSmoke

open Hesper.WGSL Hesper.WGSL.Monad Hesper.Transpile.CUDA

/-- Render a ShaderM action to its WGSL body for visual inspection. -/
def renderShader (m : ShaderM Unit) : String :=
  let st := ShaderM.exec m
  let sharedDecls : String :=
    st.sharedVars.foldl (init := "") fun acc (n, t) =>
      acc ++ s!"var<workgroup> {n}: {t.toWGSL};\n"
  let body : String := String.join (st.stmts.map (·.toWGSL 0))
  sharedDecls ++ body

/-- Hand-flattened body of llama.cpp `rms_norm_f32<1024, false, false>`
    for the no-MoE single-channel single-sample case. The original code
    (norm.cu line 75) has `block_reduce<>` calls — we manually inline a
    warp-only sum-reduce for the smoke test (block_size=32 so a single
    warp suffices; in the real specialization with 1024 we'd need a
    cross-warp tree, but that's a separate exercise). -/
def rmsNormBody : String :=
"{
  static_assert(do_add == 0, \"smoke\");
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  x += row * stride_row;
  dst += row * ncols;

  if constexpr (do_multiply) {
    const int mul_row = 0;
    mul += mul_row * mul_stride_row;
  }
  if constexpr (do_add) {
    const int add_row = 0;
    add += add_row * add_stride_row;
  }

  float tmp = 0.0f;
  for (int col = tid; col < ncols; col += block_size) {
    const float xi = x[col];
    tmp += xi * xi;
  }

  // Manually-inlined warp-reduce-sum (block_size = 32):
  tmp += __shfl_xor_sync(0u, tmp, 16u);
  tmp += __shfl_xor_sync(0u, tmp, 8u);
  tmp += __shfl_xor_sync(0u, tmp, 4u);
  tmp += __shfl_xor_sync(0u, tmp, 2u);
  tmp += __shfl_xor_sync(0u, tmp, 1u);

  const float mean = tmp / ncols;
  const float scale = rsqrtf(mean + eps);

  for (int col = tid; col < ncols; col += block_size) {
    if constexpr (do_multiply && do_add) {
      dst[col] = scale * x[col] * mul[0] + add[0];
    } else if constexpr (do_multiply) {
      dst[col] = scale * x[col] * mul[0];
    } else {
      dst[col] = scale * x[col];
    }
  }
}"

/-- Build an env that:
  * binds the kernel pointer args `x`, `dst` as f32 buffers (mul/add
    are unused because do_multiply/do_add = 0 — they don't appear after
    constexpr folding)
  * binds `threadIdx.x` and `blockIdx.x` to ShaderM-style locals
  * binds `block_size = 32`, `do_multiply = 0`, `do_add = 0` as consts
  * supplies `ncols`, `stride_row` etc. as runtime u32 idents -/
def envFor : Env := {
  bufs := fun n => match n with
    | "x"   => some { name := "x_buf", elemTy := .scalar .f32 }
    | "dst" => some { name := "dst_buf", elemTy := .scalar .f32 }
    | _ => none
  -- Builtin thread/block identifiers — bind to plain Exp.var so the
  -- WGSL output is readable. In a real codegen they'd be
  -- `Exp.vec3X (Exp.var "__local_id")` etc.
  threadIdxX := some (Exp.var "threadIdx_x")
  blockIdxX  := some (Exp.var "blockIdx_x")
  -- f32 runtime params
  f32 := fun n => match n with
    | "eps" => some (Exp.var "eps")
    | _ => none
  -- u32 runtime params (ncols, stride_row, mul_stride_row, etc.)
  u32 := fun n => match n with
    | "ncols" | "stride_row" | "mul_stride_row" | "add_stride_row" =>
      some (Exp.var n)
    | _ => none
  -- Compile-time constants from the template specialization.
  consts := fun n => match n with
    | "block_size"  => some 32
    | "do_multiply" => some 0
    | "do_add"      => some 0
    | _ => none
}

def main : IO Unit := do
  IO.println "=== Phase 6 smoke: rms_norm_f32<32, false, false> ==="
  match parseStmtStr rmsNormBody with
  | .error e =>
    IO.println s!"PARSE ERROR: {e}"
    IO.Process.exit 1
  | .ok stmt =>
    IO.println "PARSE OK"
    match lowerStmt envFor stmt with
    | .error e =>
      IO.println s!"LOWER ERROR: {e}"
      IO.Process.exit 1
    | .ok act =>
      IO.println "LOWER OK"
      IO.println "--- generated WGSL body ---"
      IO.println (renderShader act)
      IO.println "=== done ==="

end Hesper.Transpile.CUDA.RmsNormSmoke

def main : IO Unit := Hesper.Transpile.CUDA.RmsNormSmoke.main
