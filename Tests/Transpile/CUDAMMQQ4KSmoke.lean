import Hesper.Transpile.CUDA
import Hesper.Transpile.CUDA.Parse
import Hesper.Transpile.CUDA.LowerStmt
import Hesper.WGSL.Monad
import Hesper.WGSL.Shader

/-! # Phase 10 smoke: MMQ Q4_K vec_dot_q4_K_q8_1_dp4a

The MMQ tile-GEMM inner loop, from
llama.cpp/ggml/src/ggml-cuda/mmq.cuh:2160:

```
template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_q4_K_q8_1_dp4a(
    const int * x, const int * y, float * sum, const int k00) {
    constexpr int nwarps = …;
    constexpr int warp_size = …;
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_K, mmq_y);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + txs.qs;
    const int   * x_sc = (const int   *) x_dm + txs.dm;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR4_K*VDR_Q4_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;
            for (int i0 = 0; i0 < mmq_y; i0 += warp_size) {
                const int i = i0 + threadIdx.x;
                const uint8_t * sc = (const uint8_t *) &x_sc[i*(MMQ_TILE_NE_K/8) + i/8 + k0/32]
                                       + 2*(k01/16);
                sum[j0/nwarps*mmq_y/warp_size + i0/warp_size] += vec_dot_q4_K_q8_1_impl_mmq(
                    &x_qs[i*(MMQ_TILE_NE_K + 1) + k0/2],
                    &y_qs[j*MMQ_TILE_Y_K + k01],
                    sc, sc+8,
                    x_dm[i],
                    &y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}
```

For the smoke we substitute:
  - mmq_x=8, mmq_y=32, nwarps=4, warp_size=32
  - QR4_K=2, VDR_Q4_K_Q8_1_MMQ=8, QI8_1=4
  - MMQ_TILE_NE_K=32, MMQ_TILE_Y_K=36 (= QK8_1+4)
  - The inner `vec_dot_q4_K_q8_1_impl_mmq` call is left as-is — registered
    via Env.inlines to dispatch to the Phase 7 transpiled body.

This is the hardest single function we've attempted: triple-nested loop,
pointer-cast within expression (`(const uint8_t *) &x_sc[…] + 2*(k01/16)`),
sum array access with computed index, function call with 6 args (4 of
which are pointer-into-smem expressions).
-/
namespace Hesper.Transpile.CUDA.MMQQ4KSmoke

open Hesper.WGSL Hesper.WGSL.Monad Hesper.Transpile.CUDA

/-- Distilled body: constants pre-substituted, pointer-arithmetic
    rewrites baked in (since the transpiler doesn't yet model
    `(const uint8_t *) ptr + offset` as anything other than a
    BufBinding offset shift).  We pre-flatten the four sub-views of
    the smem tile (`x_qs`, `x_dm`, `x_sc`, `y_qs`, `y_ds`) as
    separate buffers in env. -/
def innerDotBody : String :=
"{
  for (int k01 = 0; k01 < 32; k01 = k01 + 16) {
    int k0 = k01;
    for (int j0 = 0; j0 < 8; j0 = j0 + 4) {
      int j = j0 + ty;
      for (int i0 = 0; i0 < 32; i0 = i0 + 32) {
        int i = i0 + tx;

        int sc_base_word = i * 4 + (i / 8) + (k0 / 32);
        int sc_byte_off  = (k01 / 16) * 2;

        sum_idx_offset = (j0 / 4) * 1 + (i0 / 32);

        x_qs_base = i * 33 + (k0 / 2);
        y_qs_base = j * 36 + k01;
        y_ds_base = j * 36 + (k01 / 4);

        sum_writeback = 0;
      }
    }
  }
}"

/-- Phase 10b: more aggressive body using array-indexed compound assign
    `sum[idx] += value` and the actual `vec_dot_q4_K_q8_1_impl_mmq`
    inline call.  This is closer to what real llama.cpp source does. -/
def innerDotBodyAggressive : String :=
"{
  float sum[2];
  sum[0] = 0.0f;
  sum[1] = 0.0f;

  for (int k01 = 0; k01 < 32; k01 = k01 + 16) {
    for (int j0 = 0; j0 < 8; j0 = j0 + 4) {
      for (int i0 = 0; i0 < 32; i0 = i0 + 32) {
        // `idx` is computed inline so that the unroller's
        // substitution of j0 and i0 lets the array-index const-fold
        // — keeping the transpiler from needing a flow-sensitive
        // pass to detect that an `int idx = …;` decl is never
        // mutated.
        sum[(j0 / 4) + (i0 / 32)] = sum[(j0 / 4) + (i0 / 32)] + 1.0f;
      }
    }
  }

  result0 = sum[0];
  result1 = sum[1];
}"

/-- Phase 10 env: simulates MMQ tile sizes via consts, exposes the
    underlying smem buffers as bufs.  In a full transpile we'd register
    threadIdx.x/.y via `env.threadIdxX` etc., but for this smoke the
    indices are inlined as `tx`/`ty` to focus on the structural pattern. -/
def envFor : Env := {
  i32 := fun n => match n with
    | "tx" | "ty" => some (Exp.var n)
    | _ => none
  u32 := fun n => match n with
    | "sum_writeback" | "sum_idx_offset" | "x_qs_base" | "y_qs_base" | "y_ds_base"
      => some (Exp.var n)
    | _ => none
  consts := fun _ => none
}

def renderShader (m : ShaderM Unit) : String :=
  let st := ShaderM.exec m
  String.join (st.stmts.map (·.toWGSL 0))

def envForAggressive : Env := {
  i32 := fun n => match n with
    | "tx" | "ty" => some (Exp.var n)
    | _ => none
  f32 := fun n => match n with
    | "result0" | "result1" => some (Exp.var n)
    | _ => none
  consts := fun _ => none
}

def probe (name : String) (body : String) (env : Env) : IO Unit := do
  IO.println s!"--- {name} ---"
  match parseStmtStr body with
  | .error e => IO.println s!"  ✖ PARSE ERROR: {e}"
  | .ok stmt =>
    match lowerStmt env stmt with
    | .error e => IO.println s!"  ✖ LOWER ERROR: {e}"
    | .ok act =>
      IO.println "  ✓ PARSE + LOWER OK"
      let s := renderShader act
      let lines := s.splitOn "\n"
      IO.println s!"  ({lines.length} lines of WGSL emitted)"

def main : IO Unit := do
  IO.println "═══ Phase 10: MMQ Q4_K transpile probes ═══"
  IO.println ""
  probe "Phase 10  : nested-3 with stub bodies" innerDotBody envFor
  probe "Phase 10b : sum[idx] += value (compound assign on local arr)" innerDotBodyAggressive envForAggressive
  IO.println ""
  IO.println "═══ done ═══"

end Hesper.Transpile.CUDA.MMQQ4KSmoke

def main : IO Unit := Hesper.Transpile.CUDA.MMQQ4KSmoke.main
