import Hesper
import Hesper.WGSL.Execute
import Hesper.WGSL.Monad
import Hesper.WGSL.MatMul
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WebGPU.BufferOps
import Hesper.Float16

open Hesper.WebGPU
open Hesper.WGSL
abbrev SM := Hesper.WGSL.Monad.ShaderM
open Hesper.WGSL.Monad.ShaderM (
  declareInputBuffer declareOutputBuffer
  declareMatrixLeftArray declareMatrixRightArray declareMatrixResultArray
  loadMatrixLeft loadMatrixRight matrixMultiplyAccumulate storeMatrixResult)

set_option maxRecDepth 8000

namespace Examples.Compute.WMMA8x8Test

/-- One 8×8 × 8×8 → 8×8 f16 subgroup matmul (confirms 8×8 is supported on this device). -/
def probe8 : SM Unit := do
  let _A ← declareInputBuffer "A" (.array (.scalar .f16) 64)
  let _B ← declareInputBuffer "B" (.array (.scalar .f16) 64)
  let _C ← declareOutputBuffer "C" (.array (.scalar .f16) 64)
  declareMatrixLeftArray "Ax" .f16 8 8 1 Exp.subgroupMatrixZeroLeft
  declareMatrixRightArray "Bx" .f16 8 8 1 Exp.subgroupMatrixZeroRight
  declareMatrixResultArray "Cx" .f16 8 8 1 Exp.subgroupMatrixZeroResult
  loadMatrixLeft (st := .f16) (m := 8) (k := 8) "Ax" 0 "A" (Exp.litU32 0) (Exp.litU32 8)
  loadMatrixRight (st := .f16) (k := 8) (n := 8) "Bx" 0 "B" (Exp.litU32 0) (Exp.litU32 8)
  matrixMultiplyAccumulate (st := .f16) (m := 8) (k := 8) (n := 8) "Cx" 0 "Ax" 0 "Bx" 0
  storeMatrixResult (st := .f16) (m := 8) (n := 8) "Cx" 0 "C" (Exp.litU32 0) (Exp.litU32 8)

def floatsToF16Bytes (arr : Array Float) : IO ByteArray := do
  let f16 ← Hesper.Float16.fromFloatArray (FloatArray.mk arr); pure f16.data

def f16BytesToFloats (bytes : ByteArray) : IO (Array Float) := do
  match Hesper.Float16.fromBytes bytes with
  | none => throw (IO.userError "f16 bytes must be 2-byte aligned")
  | some arr =>
    let fa ← Hesper.Float16.toFloatArray arr
    let mut out : Array Float := #[]
    for i in [:fa.size] do out := out.push (fa.get! i)
    pure out

def main : IO Unit := do
  IO.println "=== 8×8 WMMA test (Apple matrix units) ==="
  let inst ← Hesper.init
  let device ← getDevice inst
  let hasSm ← Execute.hasSubgroupMatrixSupport device
  let hasF16 ← Execute.hasShaderF16Support device
  IO.println s!"  SubgroupMatrix={hasSm} ShaderF16={hasF16}"
  if !hasSm || !hasF16 then IO.println "  ✗ features missing"; return
  let mkBuf (usz : USize) : IO Buffer := createBuffer device {
    size := usz, usage := [.storage, .copyDst, .copySrc], mappedAtCreation := false }
  let cfg8 : Execute.ExecutionConfig := {
    funcName := "main", workgroupSize := { x := 32, y := 1, z := 1 }, numWorkgroups := (1, 1, 1),
    extensions := ["f16", "chromium_experimental_subgroup_matrix"],
    diagnostics := [("off", "chromium.subgroup_matrix_uniformity")] }
  -- ---- Test 1: single 8×8 ----
  let mut aD : Array Float := #[]; let mut bD : Array Float := #[]
  for i in [:8] do for _j in [:8] do aD := aD.push (i+1 : Nat).toFloat
  for _i in [:8] do for j in [:8] do bD := bD.push (j+1 : Nat).toFloat
  let aBuf ← mkBuf 128; let bBuf ← mkBuf 128; let cBuf ← mkBuf 128
  writeBuffer device aBuf 0 (← floatsToF16Bytes aD)
  writeBuffer device bBuf 0 (← floatsToF16Bytes bD)
  let bufs1 : List (String × Buffer) := [("A",aBuf),("B",bBuf),("C",cBuf)]
  Execute.executeShaderNamed device probe8 bufs1 cfg8
  let c ← f16BytesToFloats (← mapBufferRead device cBuf 0 128)
  let mut ok := 0
  for i in [:8] do for j in [:8] do
    let exp := (8*(i+1)*(j+1) : Nat).toFloat
    if (exp - c.getD (i*8+j) 0.0).abs < 0.5 then ok := ok+1
  let v1 := if ok==64 then "✅ 8×8 subgroup matrix WORKS" else "❌ failed"
  IO.println s!"  [single 8×8] {ok}/64 match → {v1}"
  if ok != 64 then return
  -- ---- Test 2: tiled matMulTransposeF16WMMA8x8Kernel, M=N=K=64 ----
  let M := 64; let N := 64; let K := 64
  let cfgT : MatMul.Config := { M := M, N := N, K := K }
  -- A[m,k] = (m+k)%7 - 3 ; B[n,k] = (n*3+k)%5 - 2   (small signed)
  let af := fun (m k : Nat) => (((m+k)%7 : Nat).toFloat - 3.0)
  let bf := fun (n k : Nat) => (((n*3+k)%5 : Nat).toFloat - 2.0)
  let mut aT : Array Float := #[]; let mut bT : Array Float := #[]
  for m in [:M] do for k in [:K] do aT := aT.push (af m k)
  for n in [:N] do for k in [:K] do bT := bT.push (bf n k)
  let aBufT ← mkBuf (M*K*4).toUSize           -- f32
  let bBufT ← mkBuf (N*K*2).toUSize           -- f16-packed (= u32 [N,K/2])
  let cBufT ← mkBuf (M*N*4).toUSize           -- f32
  writeBuffer device aBufT 0 (← Hesper.Basic.floatArrayToBytes aT)
  writeBuffer device bBufT 0 (← floatsToF16Bytes bT)
  let cfgRun : Execute.ExecutionConfig := {
    funcName := "main", workgroupSize := { x := 32, y := 1, z := 1 }, numWorkgroups := (N/8, M/8, 1),
    extensions := ["f16", "chromium_experimental_subgroup_matrix"],
    diagnostics := [("off", "chromium.subgroup_matrix_uniformity")] }
  let bufs2 : List (String × Buffer) := [("a",aBufT),("b",bBufT),("c",cBufT)]
  Execute.executeShaderNamed device (MatMul.matMulTransposeF16WMMA8x8Kernel cfgT) bufs2 cfgRun
  let cT ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device cBufT 0 (M*N*4).toUSize)
  let mut ok2 := 0; let mut shown := 0
  for m in [:M] do for n in [:N] do
    let mut s := 0.0
    for k in [:K] do s := s + (af m k) * (bf n k)
    let got := cT.getD (m*N+n) 0.0
    if (s - got).abs < 1.0 then ok2 := ok2+1
    else if shown < 3 then IO.println s!"    ✗ [{m},{n}] exp={s} got={got}"; shown := shown+1
  let v2 := if ok2==M*N then "✅ matMulTransposeF16WMMA8x8Kernel CORRECT" else "❌"
  IO.println s!"  [tiled 64³] {ok2}/{M*N} match → {v2}"

  -- ---- Test 3: GFLOPS benchmark (proof the matrix units are fast) ----
  let MB := 1024
  let cfgB : MatMul.Config := { M := MB, N := MB, K := MB }
  let mut aB : Array Float := #[]; let mut bB : Array Float := #[]
  for _i in [:MB*MB] do aB := aB.push 0.01
  for _i in [:MB*MB] do bB := bB.push 0.01
  let aBufB ← mkBuf (MB*MB*4).toUSize
  let bBufB ← mkBuf (MB*MB*2).toUSize
  let cBufB ← mkBuf (MB*MB*4).toUSize
  writeBuffer device aBufB 0 (← Hesper.Basic.floatArrayToBytes aB)
  writeBuffer device bBufB 0 (← floatsToF16Bytes bB)
  let cfgRunB : Execute.ExecutionConfig := {
    funcName := "main", workgroupSize := { x := 32, y := 1, z := 1 }, numWorkgroups := (MB/8, MB/8, 1),
    extensions := ["f16", "chromium_experimental_subgroup_matrix"],
    diagnostics := [("off", "chromium.subgroup_matrix_uniformity")] }
  let bufsB : List (String × Buffer) := [("a",aBufB),("b",bBufB),("c",cBufB)]
  -- warmup
  Execute.executeShaderNamed device (MatMul.matMulTransposeF16WMMA8x8Kernel cfgB) bufsB cfgRunB
  let iters := 30
  let t0 ← IO.monoMsNow
  for _ in [0:iters] do
    Execute.executeShaderNamed device (MatMul.matMulTransposeF16WMMA8x8Kernel cfgB) bufsB cfgRunB
  let t1 ← IO.monoMsNow
  let secs := (t1 - t0).toFloat / 1000.0 / iters.toFloat
  let flop := 2.0 * MB.toFloat * MB.toFloat * MB.toFloat
  let gflops := flop / secs / 1.0e9
  IO.println s!"  [bench 1024³ × {iters}] {secs*1000.0} ms/iter → {gflops} GFLOPS (M4 Max f16 peak ~34000; forward currently ~170 effective)"

  -- ---- Test 4: register-blocked WMMA (TM=TN=4 → 32×32 output/workgroup) ----
  let TM := 2; let TN := 2
  -- correctness on 64³ (reuse aBufT/bBufT)
  let cBufR ← mkBuf (M*N*4).toUSize
  let cfgRunR64 : Execute.ExecutionConfig := {
    funcName := "main", workgroupSize := { x := 32, y := 1, z := 1 },
    numWorkgroups := (N/(8*TN), M/(8*TM), 1),
    extensions := ["f16", "chromium_experimental_subgroup_matrix"],
    diagnostics := [("off", "chromium.subgroup_matrix_uniformity")] }
  let bufsR : List (String × Buffer) := [("a",aBufT),("b",bBufT),("c",cBufR)]
  Execute.executeShaderNamed device (MatMul.matMulTransposeF16WMMA8x8RegKernel cfgT TM TN) bufsR cfgRunR64
  let cR ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device cBufR 0 (M*N*4).toUSize)
  let mut ok4 := 0
  for m in [:M] do for n in [:N] do
    let mut sm := 0.0
    for k in [:K] do sm := sm + (af m k) * (bf n k)
    if (sm - cR.getD (m*N+n) 0.0).abs < 1.0 then ok4 := ok4+1
  let v4 := if ok4==M*N then "✅ reg-blocked CORRECT" else "❌"
  IO.println s!"  [reg 64³] {ok4}/{M*N} → {v4}"
  -- benchmark on 1024³
  let cfgRunRB : Execute.ExecutionConfig := {
    funcName := "main", workgroupSize := { x := 32, y := 1, z := 1 },
    numWorkgroups := (MB/(8*TN), MB/(8*TM), 1),
    extensions := ["f16", "chromium_experimental_subgroup_matrix"],
    diagnostics := [("off", "chromium.subgroup_matrix_uniformity")] }
  Execute.executeShaderNamed device (MatMul.matMulTransposeF16WMMA8x8RegKernel cfgB TM TN) bufsB cfgRunRB
  let t2 ← IO.monoMsNow
  for _ in [0:iters] do
    Execute.executeShaderNamed device (MatMul.matMulTransposeF16WMMA8x8RegKernel cfgB TM TN) bufsB cfgRunRB
  let t3 ← IO.monoMsNow
  let secsR := (t3 - t2).toFloat / 1000.0 / iters.toFloat
  let gflopsR := flop / secsR / 1.0e9
  IO.println s!"  [reg-bench 1024³ × {iters}] {secsR*1000.0} ms/iter → {gflopsR} GFLOPS (naive was ~503; f16 peak ~34000)"

  -- ---- Test 5: subgroup-blocked WMMA ----
  let STM := 2; let STN := 2
  let cBufS ← mkBuf (M*N*4).toUSize
  let cfgRunS64 : Execute.ExecutionConfig := {
    funcName := "main", workgroupSize := { x := STM*STN*32, y := 1, z := 1 },
    numWorkgroups := (N/(8*STN), M/(8*STM), 1),
    extensions := ["f16", "chromium_experimental_subgroup_matrix"],
    diagnostics := [("off", "chromium.subgroup_matrix_uniformity")] }
  let bufsS : List (String × Buffer) := [("a",aBufT),("b",bBufT),("c",cBufS)]
  Execute.executeShaderNamed device (MatMul.matMulTransposeF16WMMA8x8SgKernel cfgT STM STN) bufsS cfgRunS64
  let cS ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device cBufS 0 (M*N*4).toUSize)
  let mut ok5 := 0
  for m in [:M] do for n in [:N] do
    let mut sm := 0.0
    for k in [:K] do sm := sm + (af m k) * (bf n k)
    if (sm - cS.getD (m*N+n) 0.0).abs < 1.0 then ok5 := ok5+1
  let v5 := if ok5==M*N then "✅ subgroup-blocked CORRECT" else "❌"
  IO.println s!"  [sg 64³] {ok5}/{M*N} → {v5}"
  let cfgRunSB : Execute.ExecutionConfig := {
    funcName := "main", workgroupSize := { x := STM*STN*32, y := 1, z := 1 },
    numWorkgroups := (MB/(8*STN), MB/(8*STM), 1),
    extensions := ["f16", "chromium_experimental_subgroup_matrix"],
    diagnostics := [("off", "chromium.subgroup_matrix_uniformity")] }
  Execute.executeShaderNamed device (MatMul.matMulTransposeF16WMMA8x8SgKernel cfgB STM STN) bufsB cfgRunSB
  let t4 ← IO.monoMsNow
  for _ in [0:iters] do
    Execute.executeShaderNamed device (MatMul.matMulTransposeF16WMMA8x8SgKernel cfgB STM STN) bufsB cfgRunSB
  let t5 ← IO.monoMsNow
  let secsS := (t5 - t4).toFloat / 1000.0 / iters.toFloat
  let gflopsS := flop / secsS / 1.0e9
  IO.println s!"  [sg-bench 1024³ × {iters}] {secsS*1000.0} ms/iter → {gflopsS} GFLOPS (naive ~503, reg slower)"

  -- ---- Test 6: K-batched WMMA (BK=8) ----
  let BK := 8
  let cBufK ← mkBuf (M*N*4).toUSize
  let cfgRunK64 : Execute.ExecutionConfig := {
    funcName := "main", workgroupSize := { x := 32, y := 1, z := 1 }, numWorkgroups := (N/8, M/8, 1),
    extensions := ["f16", "chromium_experimental_subgroup_matrix"],
    diagnostics := [("off", "chromium.subgroup_matrix_uniformity")] }
  let bufsK : List (String × Buffer) := [("a",aBufT),("b",bBufT),("c",cBufK)]
  Execute.executeShaderNamed device (MatMul.matMulTransposeF16WMMA8x8KbKernel cfgT BK) bufsK cfgRunK64
  let cK ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device cBufK 0 (M*N*4).toUSize)
  let mut ok6 := 0
  for m in [:M] do for n in [:N] do
    let mut sm := 0.0
    for k in [:K] do sm := sm + (af m k) * (bf n k)
    if (sm - cK.getD (m*N+n) 0.0).abs < 1.0 then ok6 := ok6+1
  let v6 := if ok6==M*N then "✅ K-batched CORRECT" else "❌"
  IO.println s!"  [kb 64³] {ok6}/{M*N} → {v6}"
  let cfgRunKB : Execute.ExecutionConfig := {
    funcName := "main", workgroupSize := { x := 32, y := 1, z := 1 }, numWorkgroups := (MB/8, MB/8, 1),
    extensions := ["f16", "chromium_experimental_subgroup_matrix"],
    diagnostics := [("off", "chromium.subgroup_matrix_uniformity")] }
  Execute.executeShaderNamed device (MatMul.matMulTransposeF16WMMA8x8KbKernel cfgB BK) bufsB cfgRunKB
  let t6 ← IO.monoMsNow
  for _ in [0:iters] do
    Execute.executeShaderNamed device (MatMul.matMulTransposeF16WMMA8x8KbKernel cfgB BK) bufsB cfgRunKB
  let t7 ← IO.monoMsNow
  let secsK := (t7 - t6).toFloat / 1000.0 / iters.toFloat
  let gflopsK := flop / secsK / 1.0e9
  IO.println s!"  [kb-bench 1024³ × {iters}] {secsK*1000.0} ms/iter → {gflopsK} GFLOPS (naive ~555)"

end Examples.Compute.WMMA8x8Test

def main : IO Unit := Examples.Compute.WMMA8x8Test.main
