import Hesper.Transpile.CUDA
import Hesper.Transpile.CUDA.Parse
import Hesper.Transpile.CUDA.LowerStmt
import Hesper.WGSL.Monad
import Hesper.WGSL.Shader

/-! # Prefill kernel transpile probe

Asks the question: of the kernels llama.cpp uses on the prefill path,
which ones can the current transpiler (Phase 1..7) handle, and what's
the first failure for those it can't?

Each candidate is a *body* extracted (and lightly massaged) from
llama.cpp's source.  We don't try to be 100% faithful — we strip
template params, reference-out-params, and other features the
transpiler's surface AST doesn't yet model, then report whether
the resulting body parses + lowers.

Prefill hot kernels (per llama.cpp/ggml/src/ggml-cuda/):
  1. quantize_q8_1 (quantize.cu:5)         — input quantize
  2. vec_dot_q4_K_q8_1_impl_mmq (vecdotq.cuh:527) — already done (#346)
  3. vec_dot_q6_K_q8_1_impl_mmvq (vecdotq.cuh:589) — Phase 5b done
  4. rms_norm_f32 (norm.cu)                 — Phase 6 done
  5. rope_neox body (rope.cu:115)          — try
  6. mul_mat_vec_q inner block (mmvq.cu)   — try
  7. flash_attn_vec inner (fattn-vec.cuh)  — try

This probe focuses on the *new* candidates (1, 5, 6, 7) and reports
their transpile status with the failure mode for each.
-/
namespace Hesper.Transpile.CUDA.PrefillProbe

open Hesper.WGSL Hesper.WGSL.Monad Hesper.Transpile.CUDA

/-! ## Candidate 1: quantize_q8_1 body (input Q8_1 quantize)

Original (llama.cpp/ggml/src/ggml-cuda/quantize.cu:5). We strip
`int64_t` (use `int`), drop the `block_q8_1 *` cast (this is the
POD-struct gap), and remove the multi-dim indexing — leaving the
core arithmetic. -/
def quantizeQ81Simplified : String :=
"{
  const int i0 = blockDim.x * blockIdx.x + threadIdx.x;

  if (i0 >= ne0) {
    return;
  }

  const int i_cont = blockIdx.y * ne0 + i0;
  const int ib  = i_cont / 32;
  const int iqs = i_cont - ib * 32;

  const float xi = i0 < ne00 ? x[i0] : 0.0f;
  float amax = fabsf(xi);
  float sum = xi;

  amax = warp_reduce_max(amax);
  sum  = warp_reduce_sum(sum);

  const float d = amax / 127.0f;

  qs[i_cont] = amax;
  if (iqs == 0) {
    ds[ib] = d;
  }
}"

/-! ## Candidate 2: rope_neox dispatch body

Picks just the rotation arithmetic — i.e. assuming `cos_theta` /
`sin_theta` already computed. Gemma 4 uses NeoX layout. -/
def ropeNeoxRotation : String :=
"{
  const int row_dst = blockDim.x * blockIdx.x + threadIdx.x;
  const int i0 = 2 * (blockDim.y * blockIdx.y + threadIdx.y);

  if (i0 >= ne00) {
    return;
  }

  const int idst = i0 / 2 + row_dst * stride_out;
  const int ix   = i0 / 2 + row_dst * stride_in;

  const float theta_base = pos_val * theta_scale_pow_i0;
  const float cos_theta = cosf(theta_base);
  const float sin_theta = sinf(theta_base);

  const float x0 = x[ix];
  const float x1 = x[ix + n_dims_half];

  dst[idst]                = x0 * cos_theta - x1 * sin_theta;
  dst[idst + n_dims_half]  = x0 * sin_theta + x1 * cos_theta;
}"

/-! ## Candidate 3: mul_mat_vec_q inner accumulator

The body inside `mul_mat_vec_q`'s `for (int kbx = …)` — this is the
hot path that calls `vec_dot_q4_K_q8_1` per block.  Already covered
by Phase 7 vec_dot transpile, but here we test that the *outer*
loop structure also lowers. -/
def mmvqOuterLoop : String :=
"{
  float tmp = 0.0f;
  for (int kbx = 0; kbx < blocks_per_row; kbx = kbx + 1) {
    const int ibx = row * blocks_per_row + kbx;
    const int iby = kbx * blocks_per_q8_1;
    tmp = tmp + vec_dot(ibx, iby);
  }
  result = tmp;
}"

/-! ## Candidate 4: flash_attn_vec online softmax inner

The compute-heavy core of fattn-vec.cuh. -/
def fattnVecOnlineSoftmax : String :=
"{
  float kqmax_new = kqmax;
  float KQ_max_scale = 1.0f;

  for (int j = 0; j < D; j = j + 1) {
    const float k_val = K[k_offset + j];
    sum = sum + Q[j] * k_val;
  }

  if (sum > kqmax_new) {
    kqmax_new = sum;
    KQ_max_scale = expf(kqmax - kqmax_new);
  }

  const float val = expf(sum - kqmax_new);
  out[tid] = val * KQ_max_scale;
}"

def envSimple : Env := {
  threadIdxX := some (Exp.litU32 0)
  threadIdxY := some (Exp.litU32 0)
  blockIdxX  := some (Exp.litU32 0)
  blockIdxY  := some (Exp.litU32 0)
  blockDimX  := some (Exp.litU32 32)
  blockDimY  := some (Exp.litU32 1)
  consts := fun n => match n with
    | "ne0" | "ne00" => some 4096
    | "stride_out" | "stride_in" => some 4096
    | "n_dims_half" => some 64
    | "blocks_per_row" => some 128
    | "blocks_per_q8_1" => some 8
    | "row" => some 0
    | "D" => some 128
    | _ => none
  bufs := fun n => match n with
    | "x" | "qs" => some { name := "x_buf", elemTy := .scalar .f32 }
    | "ds" => some { name := "ds_buf", elemTy := .scalar .f32 }
    | "dst" => some { name := "dst_buf", elemTy := .scalar .f32 }
    | "K" => some { name := "k_buf", elemTy := .scalar .f32 }
    | "Q" => some { name := "q_buf", elemTy := .scalar .f32 }
    | "out" => some { name := "out_buf", elemTy := .scalar .f32 }
    | _ => none
  f32 := fun n => match n with
    | "result" => some (Exp.var "result")
    | "kqmax" => some (Exp.var "kqmax")
    | "sum" => some (Exp.var "sum")
    | "pos_val" => some (Exp.var "pos_val")
    | "theta_scale_pow_i0" => some (Exp.var "theta_scale_pow_i0")
    | "k_offset" => some (Exp.var "k_offset_f32")
    | "tid" => some (Exp.var "tid_f32")
    | _ => none
  u32 := fun n => match n with
    | "k_offset" => some (Exp.var "k_offset")
    | "tid" => some (Exp.var "tid")
    | _ => none
  inlines := fun fn args => match fn, args.toList with
    -- warp_reduce_max / warp_reduce_sum collapse to identity for the probe
    -- (we're testing that the *call site* parses, not the reduction itself).
    | "warp_reduce_max", [v] => some v
    | "warp_reduce_sum", [v] => some v
    | "vec_dot", [_, _] => some (.numLit "0")
    | _, _ => none
}

def probe (name : String) (body : String) : IO Unit := do
  IO.println s!"--- {name} ---"
  match parseStmtStr body with
  | .error e =>
    IO.println s!"  ✖ PARSE ERROR: {e}"
  | .ok stmt =>
    match lowerStmt envSimple stmt with
    | .error e =>
      IO.println s!"  ✖ LOWER ERROR: {e}"
    | .ok _ =>
      IO.println s!"  ✓ PARSE + LOWER OK"

def main : IO Unit := do
  IO.println "═══ Prefill kernel transpile probe ═══"
  IO.println ""
  probe "quantize_q8_1 (simplified)" quantizeQ81Simplified
  probe "rope_neox rotation"          ropeNeoxRotation
  probe "mul_mat_vec_q outer loop"    mmvqOuterLoop
  probe "flash_attn_vec online softmax" fattnVecOnlineSoftmax
  IO.println ""
  IO.println "═══ done ═══"

end Hesper.Transpile.CUDA.PrefillProbe

def main : IO Unit := Hesper.Transpile.CUDA.PrefillProbe.main
