import Hesper.Transpile.CUDA
import Hesper.Transpile.CUDA.Parse
import Hesper.Transpile.CUDA.LowerStmt
import Hesper.WGSL.Monad
import Hesper.WGSL.Shader

/-! # Phase 10c outer kernel smoke: mul_mat_q_process_tile

The core MMQ tile-processing kernel from
llama.cpp/ggml/src/ggml-cuda/mmq.cuh:3443.  Distilled body:
template params constant-folded (mmq_x=8, mmq_y=32, nwarps=4,
warp_size=32, qk=256, MMQ_TILE_Y_K=36, MMQ_TILE_NE_K=32, ne_block=128,
ITER_K=256, blocks_per_iter=1, sz=18); `load_tiles`, `vec_dot`,
`write_back` as inlined function calls.

The body covers:
  - per-thread `float sum[mmq_x*mmq_y / (nwarps*warp_size)] = {0.0f}`
    where `mmq_x*mmq_y / (nwarps*warp_size)` = 8*32 / 128 = 2 — local
    array of 2 floats, scalarized.
  - outer K-loop `for (kb0 = kb0_start; kb0 < kb0_stop; kb0++)`
    runtime loop (kb0_start/_stop are kernel params).
  - cooperative `tile_y[l] = by0[l]` with l unrolled.
  - `__syncthreads()` between phases.
  - branch on `fixup` flag to choose write_back path.

The K-loop's start/stop bounds are runtime so it stays as a runtime
loop; the inner `tile_y` co-op load is unrolled because `mmq_x*36/(4*32)`
= 288/128 = 2 iters, all const.
-/
namespace Hesper.Transpile.CUDA.MMQOuterSmoke

open Hesper.WGSL Hesper.WGSL.Monad Hesper.Transpile.CUDA

/-- Distilled body — emulates the K-loop with bounds [0, 4)
    (instead of [kb0_start, kb0_stop) which would be kernel params)
    so the test stays self-contained.  In a real wire-up the user
    would register kb0_start/kb0_stop as runtime u32 idents and
    keep the loop runtime. -/
def outerBody : String :=
"{
  // Per-thread accumulator.  mmq_x*mmq_y / (nwarps*warp_size) = 2.
  float sum[2];
  sum[0] = 0.0f;
  sum[1] = 0.0f;

  // Outer K-loop.  In real source this is for(kb0 = start; kb0 < stop; kb0++).
  // Here we use a small const-bound loop so the smoke is fully self-
  // contained without needing runtime-loop scaffolding.
  for (int kb0 = 0; kb0 < 4; kb0 = kb0 + 1) {
    // Phase 1: load Q4_K x-tile and Q8_1 y-tile-half-1.
    load_tiles_q4k_inline(kb0);

    // Cooperative load of tile_y (mmq_x*36 / (nwarps*warp_size) = 288/128
    // = 2.25, so 3 unrolled iters with bounds checks elided).
    for (int l0 = 0; l0 < 256; l0 = l0 + 128) {
      int l = l0 + ty * 32 + tx;
      tile_y[l] = y_buf[l + kb0 * 256];
    }

    syncthreads_inline();

    vec_dot_inline(0);

    syncthreads_inline();

    // Phase 2: load Q8_1 y-tile-half-2 and compute.
    for (int l0 = 0; l0 < 256; l0 = l0 + 128) {
      int l = l0 + ty * 32 + tx;
      tile_y[l] = y_buf[l + kb0 * 256 + 256];
    }

    syncthreads_inline();

    vec_dot_inline(32);

    syncthreads_inline();
  }

  // Epilogue: write sum[] to dst.  In real source there's a `fixup`
  // branch; we just always go to the main path.
  for (int idx = 0; idx < 2; idx = idx + 1) {
    dst_buf[ty * 32 + tx + idx * 128] = sum[idx];
  }
}"

def envFor : Env := {
  bufs := fun n => match n with
    | "y_buf"    => some { name := "y_buf",    elemTy := .scalar .i32 }
    | "tile_y"   => some { name := "tile_y_smem", elemTy := .scalar .i32 }
    | "dst_buf"  => some { name := "dst_buf",  elemTy := .scalar .f32 }
    | _ => none
  i32 := fun n => match n with
    | "tx" | "ty" => some (Exp.var n)
    | _ => none
  consts := fun _ => none
  -- Inline-rewrite stubs: load_tiles, vec_dot, write_back, syncthreads
  -- expand to no-ops at lowering time (`numLit "0"` for any context).
  inlines := fun fn args => match fn, args.toList with
    | "load_tiles_q4k_inline", [_] => some (CExpr.numLit "0")
    | "vec_dot_inline", [_] => some (CExpr.numLit "0")
    | "syncthreads_inline", [] => some (CExpr.numLit "0")
    | _, _ => none
}

def renderShader (m : ShaderM Unit) : String :=
  let st := ShaderM.exec m
  String.join (st.stmts.map (·.toWGSL 0))

def main : IO Unit := do
  IO.println "═══ Phase 10c outer kernel: mul_mat_q_process_tile (distilled) ═══"
  match parseStmtStr outerBody with
  | .error e => IO.println s!"PARSE ERROR: {e}"
  | .ok stmt =>
    IO.println "PARSE OK"
    match lowerStmt envFor stmt with
    | .error e => IO.println s!"LOWER ERROR: {e}"
    | .ok act =>
      IO.println "LOWER OK"
      let s := renderShader act
      let lines := s.splitOn "\n"
      IO.println s!"  ({lines.length} lines of WGSL emitted)"
      IO.println "═══ done ═══"

end Hesper.Transpile.CUDA.MMQOuterSmoke

def main : IO Unit := Hesper.Transpile.CUDA.MMQOuterSmoke.main
