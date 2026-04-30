import Hesper.Transpile.CUDA
import Hesper.Transpile.CUDA.Parse
import Hesper.Transpile.CUDA.LowerStmt
import Hesper.WGSL.Monad
import Hesper.WGSL.Shader

/-! # Phase 6 follow-up: transpile rms_norm_f32 with `block_reduce<>` inlined

This is the "no manual flattening" version of CUDARmsNormSmoke.lean.
Same source body as before, except the warp-reduce stage now uses
llama.cpp's actual call:

  tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

To make this lower, we register two inline rewrites in `Env.inlines`:

  block_reduce(val, _smem)  →  warp_reduce_sum(val)
  warp_reduce_sum(val)      →  ((((val + shfl_xor(16))
                                 + shfl_xor(8))
                                 + shfl_xor(4))
                                 + shfl_xor(2))
                                 + shfl_xor(1)

These rewrites approximate the actual templated specializations for
`<SUM, block_size=32, float>` (single-warp path; multi-warp adds the
smem-staged tree which we don't model here).

For block_size>32 the real `block_reduce` would also stage through
shared memory.  This smoke covers the "fits in one warp" case which
is exactly what llama.cpp uses for soft_max and small RMSNorm.
-/
namespace Hesper.Transpile.CUDA.RmsNormFullSmoke

open Hesper.WGSL Hesper.WGSL.Monad Hesper.Transpile.CUDA

def renderShader (m : ShaderM Unit) : String :=
  let st := ShaderM.exec m
  let body : String := String.join (st.stmts.map (·.toWGSL 0))
  body

/-- Build the warp-reduce-sum butterfly as a CExpr.  Five __shfl_xor's
    over offsets 16/8/4/2/1.  Caller passes `val` as a `CExpr` (typically
    just `CExpr.ident name`). -/
def warpReduceSumExpr (val : CExpr) : CExpr := Id.run do
  let shuf (v : CExpr) (lane : Nat) : CExpr :=
    .call "__shfl_xor_sync" #[.numLit "0xffffffff", v, .numLit (toString lane)]
  let mut e := val
  for o in [16, 8, 4, 2, 1] do
    e := .binop .add e (shuf e o)
  return e

/-- The kernel body — same as the manual-flatten test, but uses
    `block_reduce<...>` directly. -/
def rmsNormBody : String :=
"{
  static_assert(do_add == 0, \"smoke\");
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  x += row * stride_row;
  dst += row * ncols;

  if constexpr (do_multiply) {
    mul += 0;
  }

  float tmp = 0.0f;
  for (int col = tid; col < ncols; col += block_size) {
    const float xi = x[col];
    tmp += xi * xi;
  }

  tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

  const float mean = tmp / ncols;
  const float scale = rsqrtf(mean + eps);

  for (int col = tid; col < ncols; col += block_size) {
    if constexpr (do_multiply) {
      dst[col] = scale * x[col] * mul[0];
    } else {
      dst[col] = scale * x[col];
    }
  }
}"

def envFor : Env := {
  bufs := fun n => match n with
    | "x"   => some { name := "x_buf", elemTy := .scalar .f32 }
    | "dst" => some { name := "dst_buf", elemTy := .scalar .f32 }
    | _ => none
  threadIdxX := some (Exp.var "threadIdx_x")
  blockIdxX  := some (Exp.var "blockIdx_x")
  f32 := fun n => match n with
    | "eps" => some (Exp.var "eps")
    | _ => none
  u32 := fun n => match n with
    | "ncols" | "stride_row" => some (Exp.var n)
    | _ => none
  consts := fun n => match n with
    | "block_size"  => some 32
    | "do_multiply" => some 0
    | "do_add"      => some 0
    | _ => none
  inlines := fun fn args => match fn, args.toList with
    -- block_reduce<SUM, block_size>(val, smem) — for block_size <= 32
    -- (the WARP_SIZE path), the real body collapses to just
    -- warp_reduce_sum(val). The shared-memory arg is unused.
    | "block_reduce", [val, _smem] =>
      some (.call "warp_reduce_sum" #[val])
    -- warp_reduce_sum<float>(val) — unrolled butterfly.
    | "warp_reduce_sum", [val] =>
      some (warpReduceSumExpr val)
    | _, _ => none
}

def main : IO Unit := do
  IO.println "=== Phase 6 follow-up: rms_norm_f32 with block_reduce<> inlined ==="
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

end Hesper.Transpile.CUDA.RmsNormFullSmoke

def main : IO Unit := Hesper.Transpile.CUDA.RmsNormFullSmoke.main
