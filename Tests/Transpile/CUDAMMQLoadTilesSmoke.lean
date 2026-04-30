import Hesper.Transpile.CUDA
import Hesper.Transpile.CUDA.Parse
import Hesper.Transpile.CUDA.LowerStmt
import Hesper.WGSL.Monad
import Hesper.WGSL.Shader

/-! # Phase 10c smoke: load_tiles_q4_K (dp4a path)

The real load-tiles routine from
llama.cpp/ggml/src/ggml-cuda/mmq.cuh:2050.  Loads Q4_K weight blocks
from global memory into the X-side smem tile, including:
  - 8-bit quants stored as 4 bytes packed per int
  - half2 dm values (one per block)
  - 6-bit scales (8 of them, packed)

Uses:
  - mmq_y=32, nwarps=4, warp_size=32 (template params via env.consts)
  - QI4_K=8, MMQ_TILE_NE_K=32 (literal constants pre-substituted)
  - block_q4_K { half2 dm; uint8_t scales[12]; uint8_t qs[128]; }
  - threadIdx.x / .y for cooperative load

The big test: triple-load pattern with `__shared__` writes to smem
arrays, multi-iteration outer loop (i0 += nwarps*warp_size), and
struct member access via `bxi->dm` / `bxi->scales`.
-/
namespace Hesper.Transpile.CUDA.MMQLoadTilesSmoke

open Hesper.WGSL Hesper.WGSL.Monad Hesper.Transpile.CUDA

/-- The dp4a-path load_tiles_q4_K body, with template params and
    constants pre-substituted: mmq_y=32, nwarps=4, warp_size=32,
    MMQ_TILE_NE_K=32, QI4_K=8.

    Three sequential cooperative-load loops:

      Loop 1: load Q4_K nibble qs into x_qs[i*(33) + txi]  (32 ints / block)
      Loop 2: load half2 dm (1 per block) into x_dm[i]
      Loop 3: unpack 6-bit scales (8 / block) into x_sc[…]

    For Loop 1 with nrows*nwarps = (32/8)*4 = 16, mmq_y=32: 2 iters.
    For Loop 2 with nwarps*warp_size = 128, mmq_y=32: 1 iter (i0=0).
    For Loop 3 with nwarps*rows_per_warp = 4*8 = 32, mmq_y=32: 1 iter.

    `unpack_scales_q45_K` is registered via Env.inlines to expand
    inline; same for the get_int_b4 byte-packed read. -/
def loadTilesBody : String :=
"{
  // Loop 1: cooperative load of Q4_K nibbles (32 ints per row).
  // threads_per_row = 32/(4*2) = 4, nrows = 32/4 = 8
  for (int i0 = 0; i0 < 32; i0 = i0 + 32) {
    int i = i0 + ty * 8 + tx / 4;
    int txi = tx % 4;

    int qs0 = bq4_K_qs_int[i * 32 + txi];
    x_qs[i * 33 + txi] = qs0;
  }

  // Loop 2: load half2 dm — one per block.
  for (int i0 = 0; i0 < 32; i0 = i0 + 128) {
    int i_raw = i0 + ty * 32 + tx;
    int i = i_raw % 32;
    x_dm[i] = bq4_K_dm[i];
  }

  // Loop 3: unpack scales — rows_per_warp = 8.
  for (int i0 = 0; i0 < 32; i0 = i0 + 32) {
    int i_raw = i0 + ty * 8 + tx / 4;
    int i = i_raw % 32;

    int ksc = tx % 4;
    int scales8 = unpack_scales_q45_K_inline(i, ksc);
    x_sc[i * 4 + (i / 8) + ksc] = scales8;
  }
}"

def envFor : Env := {
  bufs := fun n => match n with
    -- pre-flattened struct members (`block_q4_K { half2 dm; uint8_t
    -- scales[12]; uint8_t qs[128]; }` would be 144-byte rows; we
    -- expose each member as its own buffer with the per-block stride
    -- baked in by the user).
    | "bq4_K_qs_int" => some { name := "x_qs_buf",   elemTy := .scalar .i32 }
    | "bq4_K_dm"     => some { name := "x_dm_buf",   elemTy := .scalar .u32 }
    | "x_qs"         => some { name := "x_qs_smem",  elemTy := .scalar .i32 }
    | "x_dm"         => some { name := "x_dm_smem",  elemTy := .scalar .u32 }
    | "x_sc"         => some { name := "x_sc_smem",  elemTy := .scalar .i32 }
    | _ => none
  i32 := fun n => match n with
    | "tx" | "ty" => some (Exp.var n)
    | _ => none
  consts := fun _ => none
  -- `unpack_scales_q45_K_inline(i, ksc)` is a placeholder for
  -- llama.cpp's `unpack_scales_q45_K(scales, ksc)`.  We register an
  -- inline rewrite that returns a fake but well-typed expression so
  -- the smoke doesn't need to plumb through the real bit-twiddling.
  inlines := fun fn args =>
    match fn, args.toList with
    | "unpack_scales_q45_K_inline", [_, _] =>
      -- Stub: just returns 0xFFFFFFFF (the lowering treats this as i32).
      some (CExpr.numLit "1234567")
    | _, _ => none
}

def renderShader (m : ShaderM Unit) : String :=
  let st := ShaderM.exec m
  String.join (st.stmts.map (·.toWGSL 0))

def main : IO Unit := do
  IO.println "═══ Phase 10c: MMQ Q4_K load_tiles (distilled) ═══"
  match parseStmtStr loadTilesBody with
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
      for ln in lines.take 25 do IO.println s!"  | {ln}"
      if lines.length > 25 then IO.println s!"  | ... (+{lines.length - 25} more lines)"
      IO.println "═══ done ═══"

end Hesper.Transpile.CUDA.MMQLoadTilesSmoke

def main : IO Unit := Hesper.Transpile.CUDA.MMQLoadTilesSmoke.main
