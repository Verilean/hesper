import Hesper.Transpile.CUDA
import Hesper.Transpile.CUDA.Parse
import Hesper.Transpile.CUDA.LowerStmt
import Hesper.WGSL.Monad
import Hesper.WGSL.Shader
import Hesper.CUDA.CodeGen
import Hesper.CUDA.Buffer
import Hesper.Basic

/-! # Phase 6 GPU parity: transpiled rms_norm runs on real GPU

Closes the loop: parse llama.cpp's `rms_norm_f32` body → ShaderM →
PTX → JIT-load on RTX 4070 Ti → run with random f32 input → compare
against a CPU reference.

Shape: `nrows = 4`, `ncols = 128`, `block_size = 32` (single warp).
Block grid is `(nrows, 1, 1)`, block dim `(32, 1, 1)`.

The transpiled body refers to `x_buf` / `dst_buf` (the buffer names we
chose in the env), `threadIdx_x` / `blockIdx_x` (variable names we
introduce into the ShaderM scope before lowering), `ncols` /
`stride_row` (we'll bind these as compile-time constants =128), and
`eps` (a runtime f32 we'll bind to a literal).

For block_size=32 the warp-shuffle path collapses to a single
butterfly — exactly what we want.
-/
namespace Hesper.Transpile.CUDA.RmsNormGPU

open Hesper Hesper.WGSL Hesper.WGSL.Monad Hesper.CUDA Hesper.CUDA.CodeGen Hesper.Transpile.CUDA

/-- Transpiled body — the same source as in CUDARmsNormFullSmoke. -/
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

/-- Warp-reduce-sum butterfly (5 levels). Same as full-smoke test. -/
def warpReduceSumExpr (val : CExpr) : CExpr := Id.run do
  let shuf (v : CExpr) (lane : Nat) : CExpr :=
    .call "__shfl_xor_sync" #[.numLit "0xffffffff", v, .numLit (toString lane)]
  let mut e := val
  for o in [16, 8, 4, 2, 1] do
    e := .binop .add e (shuf e o)
  return e

/-- Build the full kernel: declare buffers and ShaderM-level local
    bindings for `threadIdx_x` / `blockIdx_x`, then run the transpiled
    body inside that scope. -/
def buildKernel (ncols : Nat) (eps : Float) : ShaderM Unit := do
  -- Buffers (parameter order: input, output).
  let _x   ← ShaderM.declareReadOnlyBuffer "x_buf"   (.array (.scalar .f32) (ncols * 1024))
  let _dst ← ShaderM.declareOutputBuffer  "dst_buf" (.array (.scalar .f32) (ncols * 1024))

  -- Bind threadIdx.x / blockIdx.x as ShaderM-level locals so the
  -- transpiled body's `Exp.var "threadIdx_x"` / `"blockIdx_x"`
  -- references resolve to actual local IDs.
  let lid ← ShaderM.localId
  let wid ← ShaderM.workgroupId
  ShaderM.varNamed "threadIdx_x" (.scalar .u32) (Exp.vec3X lid)
  ShaderM.varNamed "blockIdx_x"  (.scalar .u32) (Exp.vec3X wid)
  -- Runtime eps as a varNamed so the transpiler's `Exp.var "eps"` resolves.
  ShaderM.varNamed "eps" (.scalar .f32) (Exp.litF32 eps)

  let envFor : Env := {
    bufs := fun n => match n with
      | "x"   => some { name := "x_buf",   elemTy := .scalar .f32 }
      | "dst" => some { name := "dst_buf", elemTy := .scalar .f32 }
      | _ => none
    threadIdxX := some (Exp.var "threadIdx_x")
    blockIdxX  := some (Exp.var "blockIdx_x")
    f32 := fun n => match n with
      | "eps" => some (Exp.var "eps")
      | _ => none
    -- ncols and stride_row are compile-time constants here; bind them
    -- through env.consts so `for (col=tid; col < ncols; col += block_size)`
    -- folds the bound to litU32 ncols.
    consts := fun n => match n with
      | "block_size"  => some 32
      | "do_multiply" => some 0
      | "do_add"      => some 0
      | "ncols"       => some (Int.ofNat ncols)
      | "stride_row"  => some (Int.ofNat ncols)
      | _ => none
    inlines := fun fn args => match fn, args.toList with
      | "block_reduce", [val, _smem] => some (.call "warp_reduce_sum" #[val])
      | "warp_reduce_sum", [val] => some (warpReduceSumExpr val)
      | _, _ => none
  }

  -- Parse + lower the source body once at Lean compile time.
  let stmt := match parseStmtStr rmsNormBody with
    | .ok s => s
    | .error e => panic! s!"parseStmtStr failed: {e}"
  match lowerStmt envFor stmt with
  | .ok act => act
  | .error e => panic! s!"lowerStmt failed: {e}"

/-- CPU reference: y = x * rsqrt(mean(x²) + eps). -/
def cpuRmsNorm (input : Array Float) (ncols : Nat) (eps : Float) : Array Float := Id.run do
  let nrows := input.size / ncols
  let mut out : Array Float := Array.mkEmpty input.size
  for r in [0:nrows] do
    let mut sumSq : Float := 0.0
    for c in [0:ncols] do
      let v := input[r*ncols + c]!
      sumSq := sumSq + v * v
    let mean := sumSq / ncols.toFloat
    let scale := 1.0 / (mean + eps).sqrt
    for c in [0:ncols] do
      out := out.push (scale * input[r*ncols + c]!)
  return out

unsafe def main : IO Unit := do
  IO.println "═══ Phase 6 GPU parity: transpiled rms_norm on RTX 4070 Ti ═══"
  let (_dev, _ctx) ← initCUDA

  let ncols : Nat := 128
  let nrows : Nat := 4
  let eps : Float := 1e-6
  IO.println s!"  shape: nrows={nrows} ncols={ncols} eps={eps}"

  let kernel := buildKernel ncols eps
  let ptx := generatePTX
    (funcName := "rms_norm_transpiled")
    (workgroupSize := { x := 32, y := 1, z := 1 })
    (computation := kernel)
    (targetArch := "sm_89")
  IO.println s!"  PTX size: {ptx.length} chars"

  let cudaMod ← cuModuleLoadData ptx
  IO.println "  ptxas JIT: OK ✓"
  let func ← cuModuleGetFunction cudaMod "rms_norm_transpiled"

  -- Build deterministic input.
  let total := nrows * ncols
  let inArr : Array Float := (List.range total).toArray.map (fun i =>
    Float.sin (i.toFloat * 0.07) * 0.5 + 0.1)
  let inBytes ← Hesper.Basic.floatArrayToBytes inArr

  let inBuf ← createCUDABuffer (4 * total).toUSize
  let outBuf ← createCUDABuffer (4 * total).toUSize
  writeCUDABuffer inBuf inBytes
  IO.println "  buffers initialized"

  -- Launch: nrows blocks × 32 threads.
  cuLaunchKernel func nrows.toUInt32 1 1 32 1 1 0 #[inBuf.ptr, outBuf.ptr]
  let resultBytes ← readCUDABufferFull outBuf
  let resultArr := Hesper.Basic.bytesToFloatArrayPure resultBytes
  IO.println "  kernel launched + result fetched"

  -- CPU reference.
  let refArr := cpuRmsNorm inArr ncols eps

  -- Compare.
  let mut maxErr : Float := 0.0
  let mut numWrong : Nat := 0
  for i in [0:total] do
    let err := (resultArr[i]! - refArr[i]!).abs
    if err > maxErr then maxErr := err
    if err > 1e-3 then numWrong := numWrong + 1

  IO.println s!"  out[0..3]   = {resultArr[0]!} {resultArr[1]!} {resultArr[2]!} {resultArr[3]!}"
  IO.println s!"  ref[0..3]   = {refArr[0]!} {refArr[1]!} {refArr[2]!} {refArr[3]!}"
  IO.println s!"  out[r=1, 0..3] = {resultArr[ncols]!} {resultArr[ncols+1]!} {resultArr[ncols+2]!} {resultArr[ncols+3]!}"
  IO.println s!"  ref[r=1, 0..3] = {refArr[ncols]!} {refArr[ncols+1]!} {refArr[ncols+2]!} {refArr[ncols+3]!}"
  IO.println s!"  max |err| = {maxErr}"
  IO.println s!"  num wrong (err > 1e-3) = {numWrong}"

  freeCUDABuffer inBuf
  freeCUDABuffer outBuf

  if maxErr < 1e-3 then
    IO.println "✓ PASSED — transpiled rms_norm matches CPU reference"
  else
    IO.println "✗ FAILED — output diverges from CPU reference"
    IO.Process.exit 1

end Hesper.Transpile.CUDA.RmsNormGPU

unsafe def main : IO Unit := Hesper.Transpile.CUDA.RmsNormGPU.main
