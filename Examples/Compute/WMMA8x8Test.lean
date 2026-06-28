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
  let cfgDump : Execute.ExecutionConfig := {
    funcName := "main", workgroupSize := { x := 128, y := 1, z := 1 }, numWorkgroups := (1,1,1),
    extensions := ["f16","chromium_experimental_subgroup_matrix"],
    diagnostics := [("off","chromium.subgroup_matrix_uniformity")] }
  let wgslRB := Execute.compileToWGSL (MatMul.matMulTransposeF16WMMARegKernel { M := 1024, N := 1024, K := 1024 }) cfgDump.funcName cfgDump.workgroupSize cfgDump.extensions cfgDump.diagnostics
  IO.FS.writeFile "/tmp/regblk.wgsl" wgslRB
  IO.println "  (wrote /tmp/regblk.wgsl)"
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

  -- ---- Test 7: forward shapes (does the 1024³ rate hold at the real matmul sizes?) ----
  let shapes : List (Nat × Nat × Nat × String) :=
    [(288, 4224, 2816, "dense gate/up M=288"),
     (288, 2816, 2112, "dense down     M=288"),
     (32,  4224, 2816, "MoE expert  M=32 (per-expert-ish)"),
     (288, 262144, 2816, "lm_head     M=288 (big N)")]
  for shp in shapes do
    let (sm, sn, sk, lbl) := shp
    let cf : MatMul.Config := { M := sm, N := sn, K := sk }
    let aBy ← Hesper.Basic.floatArrayToBytes (Array.replicate (sm*sk) 0.01)
    let bBy ← floatsToF16Bytes (Array.replicate (sn*sk) 0.01)
    let aF ← mkBuf (sm*sk*4).toUSize; let bF ← mkBuf (sn*sk*2).toUSize; let cF ← mkBuf (sm*sn*4).toUSize
    writeBuffer device aF 0 aBy; writeBuffer device bF 0 bBy
    let cfgF : Execute.ExecutionConfig := {
      funcName := "main", workgroupSize := { x := 32, y := 1, z := 1 }, numWorkgroups := (sn/8, sm/8, 1),
      extensions := ["f16", "chromium_experimental_subgroup_matrix"],
      diagnostics := [("off", "chromium.subgroup_matrix_uniformity")] }
    let bufsF : List (String × Buffer) := [("a",aF),("b",bF),("c",cF)]
    Execute.executeShaderNamed device (MatMul.matMulTransposeF16WMMA8x8Kernel cf) bufsF cfgF
    let tF0 ← IO.monoMsNow
    for _ in [0:10] do Execute.executeShaderNamed device (MatMul.matMulTransposeF16WMMA8x8Kernel cf) bufsF cfgF
    let tF1 ← IO.monoMsNow
    let secsF := (tF1 - tF0).toFloat / 1000.0 / 10.0
    let gf := 2.0 * sm.toFloat * sn.toFloat * sk.toFloat / secsF / 1.0e9
    IO.println s!"  [{lbl}] {secsF*1000.0} ms → {gf} GFLOPS"

  -- ---- Test 8: dispatch-overhead floor (is fusion worth it?) ----
  let trivialK : SM Unit := do
    let _x ← declareInputBuffer "x" (.array (.scalar .f32) 64)
    let _y ← declareOutputBuffer "y" (.array (.scalar .f32) 64)
    let gid ← Hesper.WGSL.Monad.ShaderM.globalId
    let i := Exp.vec3X gid
    Hesper.WGSL.Monad.ShaderM.if_ (Exp.lt i (Exp.litU32 64)) (do
      let v ← Hesper.WGSL.Monad.ShaderM.readBuffer (ty := .scalar .f32) (n := 64) "x" i
      Hesper.WGSL.Monad.ShaderM.writeBuffer (ty := .scalar .f32) "y" i v) (pure ())
  let xB ← mkBuf 256; let yB ← mkBuf 256
  let cfgTriv : Execute.ExecutionConfig := {
    funcName := "main", workgroupSize := { x := 64, y := 1, z := 1 }, numWorkgroups := (1, 1, 1),
    extensions := [], diagnostics := [] }
  let bufsTriv : List (String × Buffer) := [("x",xB),("y",yB)]
  let nDisp := 1500
  -- warm (compile pipeline)
  Execute.beginBatch device
  Execute.executeShaderNamed device trivialK bufsTriv cfgTriv
  Execute.endBatch device
  let td0 ← IO.monoMsNow
  Execute.beginBatch device
  for _ in [0:nDisp] do Execute.executeShaderNamed device trivialK bufsTriv cfgTriv
  Execute.endBatch device
  let td1 ← IO.monoMsNow
  let perDisp := (td1 - td0).toFloat / nDisp.toFloat
  IO.println s!"  [dispatch floor] {nDisp} trivial dispatches in 1 batch = {(td1-td0)} ms total → {perDisp} ms/dispatch"
  IO.println s!"  → forward has ~1500 dispatches/step; estimated dispatch-overhead floor = {perDisp * 1500.0} ms of the ~2500 ms step"

  -- ---- Test 9: llama.cpp-style register-blocked WMMA (64x32 tile, 4 sg, 4x2 reg block) ----
  let cBufRB ← mkBuf (M*N*4).toUSize
  let cfgRB64 : Execute.ExecutionConfig := {
    funcName := "main", workgroupSize := { x := 128, y := 1, z := 1 }, numWorkgroups := (N/32, M/64, 1),
    extensions := ["f16", "chromium_experimental_subgroup_matrix"],
    diagnostics := [("off", "chromium.subgroup_matrix_uniformity")] }
  let bufsRB : List (String × Buffer) := [("a",aBufT),("b",bBufT),("c",cBufRB)]
  Execute.executeShaderNamed device (MatMul.matMulTransposeF16WMMARegKernel cfgT) bufsRB cfgRB64
  let cRB ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device cBufRB 0 (M*N*4).toUSize)
  let mut ok9 := 0
  for m in [:M] do for n in [:N] do
    let mut sm := 0.0
    for k in [:K] do sm := sm + (af m k) * (bf n k)
    if (sm - cRB.getD (m*N+n) 0.0).abs < 1.0 then ok9 := ok9+1
  let v9 := if ok9==M*N then "✅ reg-blocked(llama.cpp構造) CORRECT" else "❌"
  IO.println s!"  [regblk 64³] {ok9}/{M*N} → {v9}"
  if ok9 == M*N then
    let cfgRBB : Execute.ExecutionConfig := {
      funcName := "main", workgroupSize := { x := 128, y := 1, z := 1 }, numWorkgroups := (MB/32, MB/64, 1),
      extensions := ["f16", "chromium_experimental_subgroup_matrix"],
      diagnostics := [("off", "chromium.subgroup_matrix_uniformity")] }
    Execute.executeShaderNamed device (MatMul.matMulTransposeF16WMMARegKernel cfgB) bufsB cfgRBB
    let tr0 ← IO.monoMsNow
    for _ in [0:iters] do Execute.executeShaderNamed device (MatMul.matMulTransposeF16WMMARegKernel cfgB) bufsB cfgRBB
    let tr1 ← IO.monoMsNow
    let secsRB := (tr1 - tr0).toFloat / 1000.0 / iters.toFloat
    let gflopsRB := flop / secsRB / 1.0e9
    IO.println s!"  [regblk 1024³ × {iters}] {secsRB*1000.0} ms/iter → {gflopsRB} GFLOPS  (naive ~500; 8× = ~4000)"

  -- ---- Test 10: reg kernel with M-bounds (M not %64) + forward shapes ----
  -- correctness at M=96 (=64+32, partial last tile), N=64, K=64
  let Mb := 96; let Nb := 64; let Kb := 64
  let cfgb : MatMul.Config := { M := Mb, N := Nb, K := Kb }
  let mut aBd : Array Float := #[]; let mut bBd : Array Float := #[]
  for m in [:Mb] do for k in [:Kb] do aBd := aBd.push (af m k)
  for n in [:Nb] do for k in [:Kb] do bBd := bBd.push (bf n k)
  let aBb ← mkBuf (Mb*Kb*4).toUSize; let bBb ← mkBuf (Nb*Kb*2).toUSize; let cBb ← mkBuf (Mb*Nb*4).toUSize
  writeBuffer device aBb 0 (← Hesper.Basic.floatArrayToBytes aBd)
  writeBuffer device bBb 0 (← floatsToF16Bytes bBd)
  let cfgb96 : Execute.ExecutionConfig := {
    funcName := "main", workgroupSize := { x := 128, y := 1, z := 1 }, numWorkgroups := ((Nb+31)/32, (Mb+63)/64, 1),
    extensions := ["f16","chromium_experimental_subgroup_matrix"], diagnostics := [("off","chromium.subgroup_matrix_uniformity")] }
  let bufsb : List (String × Buffer) := [("a",aBb),("b",bBb),("c",cBb)]
  Execute.executeShaderNamed device (MatMul.matMulTransposeF16WMMARegKernel cfgb) bufsb cfgb96
  let cBr ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device cBb 0 (Mb*Nb*4).toUSize)
  let mut okb := 0
  for m in [:Mb] do for n in [:Nb] do
    let mut sm := 0.0
    for k in [:Kb] do sm := sm + (af m k) * (bf n k)
    if (sm - cBr.getD (m*Nb+n) 0.0).abs < 1.0 then okb := okb+1
  let vb := if okb==Mb*Nb then "✅ M-bounds CORRECT" else "❌"
  IO.println s!"  [regblk M=96 bounds] {okb}/{Mb*Nb} → {vb}"
  -- GFLOPS at forward shapes
  let fshapes : List (Nat × Nat × Nat × String) :=
    [(288, 2112, 2816, "dense gate/up"), (288, 2816, 2112, "dense down"),
     (32, 1408, 2816, "MoE expert M=32"), (288, 262144, 2816, "lm_head")]
  for shp in fshapes do
    let (sm, sn, sk, lbl) := shp
    let cf : MatMul.Config := { M := sm, N := sn, K := sk }
    let aF ← mkBuf (sm*sk*4).toUSize; let bF ← mkBuf (sn*sk*2).toUSize; let cF ← mkBuf (sm*sn*4).toUSize
    writeBuffer device aF 0 (← Hesper.Basic.floatArrayToBytes (Array.replicate (sm*sk) 0.01))
    writeBuffer device bF 0 (← floatsToF16Bytes (Array.replicate (sn*sk) 0.01))
    let cfgF : Execute.ExecutionConfig := {
      funcName := "main", workgroupSize := { x := 128, y := 1, z := 1 }, numWorkgroups := ((sn+31)/32, (sm+63)/64, 1),
      extensions := ["f16","chromium_experimental_subgroup_matrix"], diagnostics := [("off","chromium.subgroup_matrix_uniformity")] }
    let bufsF : List (String × Buffer) := [("a",aF),("b",bF),("c",cF)]
    Execute.executeShaderNamed device (MatMul.matMulTransposeF16WMMARegKernel cf) bufsF cfgF
    let tf0 ← IO.monoMsNow
    for _ in [0:10] do Execute.executeShaderNamed device (MatMul.matMulTransposeF16WMMARegKernel cf) bufsF cfgF
    let tf1 ← IO.monoMsNow
    let secsF := (tf1 - tf0).toFloat / 1000.0 / 10.0
    let gf := 2.0 * sm.toFloat * sn.toFloat * sk.toFloat / secsF / 1.0e9
    IO.println s!"  [regblk {lbl}] {secsF*1000.0} ms → {gf} GFLOPS  (naive was ~200 at M=288)"

  -- ---- Test 11: weightRowOffset path (chunked, like lm_head chunks 1-7) ----
  -- N=64 split into 2 chunks of 32; reg kernel with weightRowOffset, compare to CPU
  let mut ok11 := 0
  for ch in [:2] do
    let cBufC ← mkBuf (M*32*4).toUSize
    let cfgC : MatMul.Config := { M := M, N := 32, K := K }   -- chunk width 32, K=64
    let cfgCe : Execute.ExecutionConfig := {
      funcName := "main", workgroupSize := { x := 128, y := 1, z := 1 }, numWorkgroups := ((32+31)/32, (M+63)/64, 1),
      extensions := ["f16","chromium_experimental_subgroup_matrix"], diagnostics := [("off","chromium.subgroup_matrix_uniformity")] }
    let bufsC : List (String × Buffer) := [("a",aBufT),("b",bBufT),("c",cBufC)]
    -- weightRowOffset = ch*32, weightRows = N(=64 full)
    Execute.executeShaderNamed device (MatMul.matMulTransposeF16WMMARegKernel cfgC (ch*32) N) bufsC cfgCe
    let cC ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device cBufC 0 (M*32*4).toUSize)
    for m in [:M] do for n in [:32] do
      let mut sm := 0.0
      for k in [:K] do sm := sm + (af m k) * (bf (ch*32+n) k)
      if (sm - cC.getD (m*32+n) 0.0).abs < 1.0 then ok11 := ok11+1
  let v11 := if ok11==M*64 then "✅ weightRowOffset CORRECT" else "❌ weightRowOffset BUG"
  IO.println s!"  [regblk chunked] {ok11}/{M*64} → {v11}"

end Examples.Compute.WMMA8x8Test

def main : IO Unit := Examples.Compute.WMMA8x8Test.main
