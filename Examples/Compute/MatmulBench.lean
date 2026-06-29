import Hesper
import Hesper.Backend
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WebGPU.BufferOps
import Hesper.Basic
import Hesper.WGSL.MatMul
import Hesper.Quantization.Q4_K_M

open Hesper.WebGPU
open Hesper.WGSL (Exp)

namespace Examples.Compute.MatmulBench

/-- f32 → f16 half2 u32 pack (same as the decode's packF32ToF16B). -/
def packF32ToF16 (nOut : Nat) : Hesper.WGSL.Monad.ShaderM Unit := do
  let gid ← Hesper.WGSL.Monad.ShaderM.globalId; let i := Exp.vec3X gid
  let _in ← Hesper.WGSL.Monad.ShaderM.declareReadOnlyBuffer "fin" (.array (.scalar .f32) (nOut*2))
  let _out ← Hesper.WGSL.Monad.ShaderM.declareOutputBuffer "fout" (.array (.scalar .u32) nOut)
  Hesper.WGSL.Monad.ShaderM.if_ (Exp.lt i (Exp.litU32 nOut)) (do
    let a := Exp.index (Exp.var "fin" : Exp (.array (.scalar .f32) (nOut*2))) (Exp.mul i (Exp.litU32 2))
    let b := Exp.index (Exp.var "fin" : Exp (.array (.scalar .f32) (nOut*2))) (Exp.add (Exp.mul i (Exp.litU32 2)) (Exp.litU32 1))
    Hesper.WGSL.Monad.ShaderM.writeBuffer (ty := .scalar .u32) "fout" i (Exp.pack2x16float (Exp.vec2 a b))) (pure ())

/-- Roofline micro-benchmark for the register-blocked subgroup-matrix matmul at the DiffusionGemma
    forward shapes. Times the kernel in isolation (no e2e), computes the hardware floor
    (max(weight+act+out bytes / BW, 2·M·N·K FLOPs / peak)), and reports the distance from the floor —
    so the kernel furthest above its floor is the one to attack. Weights are garbage (matmul timing
    is data-independent); correctness is covered by WMMA8x8Test / grouped-down-test (golden, no e2e). -/

def mkBuf (device : Device) (n : Nat) : IO Buffer :=
  createBuffer device { size := (n*4).toUSize, usage := [.storage, .copyDst, .copySrc], mappedAtCreation := false }

-- M4 Max: ~34 TFLOP/s f16 (matrix units), ~400 GB/s unified memory.
def peakFlops : Float := 15.5e12   -- MEASURED simdgroup_matrix peak (probe plateaus ~15.5; the 34
                                   -- f16-ALU figure is regular FMA, which the matrix instruction can't reach)
def peakBW    : Float := 400.0e9

def benchShape (device : Device) (name : String) (M N K : Nat) : IO Float := do
  let aBuf ← mkBuf device (M*K)        -- A: f32 [M,K]   (activation)
  let bBuf ← mkBuf device (N*(K/2))    -- B: f16-packed [N,K/2] u32  (weight)
  let cBuf ← mkBuf device (M*N)        -- C: f32 [M,N]   (output)
  let bufs : List (String × Buffer) := [("a",aBuf),("b",bBuf),("c",cBuf)]
  let kern := Hesper.WGSL.MatMul.matMulTransposeF16WMMARegKernel { M := M, N := N, K := K } 0 N
  let cfg : Hesper.ExecConfig := {
    numWorkgroups := ((N+31)/32, (M+63)/64, 1), workgroupSize := {x:=128},
    extensions := ["f16","chromium_experimental_subgroup_matrix"],
    diagnostics := [("off","chromium.subgroup_matrix_uniformity")] }
  let r ← IO.mkRef none
  Hesper.GPUBackend.executeWithConfigCached device kern bufs cfg 1 r   -- warmup (compile)
  let iters := 100
  -- batch the iterations (one submit, no per-call sync) so the time reflects the kernel, not the
  -- submit+wait round-trip — closer to how the matmul runs inside the batched forward.
  let t0 ← IO.monoMsNow
  Hesper.GPUBackend.beginBatch device
  for _ in [0:iters] do Hesper.GPUBackend.executeWithConfigCached device kern bufs cfg 1 r
  Hesper.GPUBackend.endBatch device
  let t1 ← IO.monoMsNow
  let ms := (t1-t0).toFloat / iters.toFloat
  let flops := 2.0 * M.toFloat * N.toFloat * K.toFloat
  let gflops := flops / (ms/1000.0) / 1.0e9
  let bytes := (N*(K/2)*4 + M*K*4 + M*N*4).toFloat   -- f16 weight (u32 packed) + f32 act + f32 out
  let floorMemMs  := bytes / peakBW * 1000.0
  let floorCompMs := flops / peakFlops * 1000.0
  let floor := max floorMemMs floorCompMs
  let bound := if floorMemMs > floorCompMs then "MEM" else "COMPUTE"
  let gbs := bytes / (ms/1000.0) / 1.0e9
  let compPct := 100.0*gflops*1.0e9/peakFlops
  let memPct := 100.0*gbs/(peakBW/1.0e9)
  IO.println s!"{name} [M={M} N={N} K={K}]: {ms}ms | compute {compPct}% ({gflops} GFLOPS) | memory {memPct}% ({gbs} GB/s) | {bound}-bound, {ms/floor}× above floor"
  pure ms

/-- Golden correctness: known f32 W → GPU pack → f16 → reg-matmul, vs a CPU matmul. Tests the
    pack + reg-matmul + B-layout path (same path the QKV reg uses, minus the Q4_K dequant). -/
def checkCorrect (device : Device) : IO Unit := do
  let M := 64; let N := 32; let K := 64
  let wf := fun (n k : Nat) => (((n+k) % 7 : Nat).toFloat) - 3.0
  let af := fun (m k : Nat) => (((m*2+k) % 5 : Nat).toFloat) - 2.0
  let mut wA : Array Float := #[]
  for n in [0:N] do for k in [0:K] do wA := wA.push (wf n k)
  let mut aA : Array Float := #[]
  for m in [0:M] do for k in [0:K] do aA := aA.push (af m k)
  let wf32 ← mkBuf device (N*K); let bf16 ← mkBuf device (N*(K/2))
  let aBuf ← mkBuf device (M*K); let cBuf ← mkBuf device (M*N)
  writeBuffer device wf32 0 (← Hesper.Basic.floatArrayToBytes wA)
  writeBuffer device aBuf 0 (← Hesper.Basic.floatArrayToBytes aA)
  let rp ← IO.mkRef none
  Hesper.GPUBackend.executeWithConfigCached device (packF32ToF16 (N*(K/2)))
    (("fin",wf32)::("fout",bf16)::List.nil) { numWorkgroups := ((N*(K/2)+255)/256,1,1), workgroupSize := {x:=256} } 1 rp
  let cfg : Hesper.ExecConfig := {
    numWorkgroups := ((N+31)/32, (M+63)/64, 1), workgroupSize := {x:=128},
    extensions := ["f16","chromium_experimental_subgroup_matrix"],
    diagnostics := [("off","chromium.subgroup_matrix_uniformity")] }
  let rr ← IO.mkRef none
  Hesper.GPUBackend.executeWithConfigCached device (Hesper.WGSL.MatMul.matMulTransposeF16WMMARegKernel { M := M, N := N, K := K } 0 N)
    (("a",aBuf)::("b",bf16)::("c",cBuf)::List.nil) cfg 1 rr
  let gpu ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device cBuf 0 (M*N*4).toUSize)
  let mut md := 0.0; let mut bad := 0
  for m in [0:M] do for n in [0:N] do
    let mut g := 0.0
    for k in [0:K] do g := g + (af m k) * (wf n k)
    let d := (gpu.getD (m*N+n) 0.0 - g).abs
    if d > md then md := d
    if d > 0.5 then bad := bad+1
  let v := if bad==0 then "✅ pack+reg+layout CORRECT" else "❌ WRONG"
  IO.println s!"[correctness M={M} N={N} K={K}] maxDiff={md} bad={bad}/{M*N} → {v}; sample gpu={gpu.getD 0 0.0}"

/-- Golden correctness for the GROUPED reg-matmul (MoE gate/up): each 64-row M-tile uses its
    tileExpert[tile] expert weight. -/
def checkGroupedCorrect (device : Device) : IO Unit := do
  let M := 128; let N := 32; let K := 64; let nE := 2   -- 2 tiles of 64 rows → experts 0,1
  let wf := fun (e n k : Nat) => (((e*3+n+k) % 7 : Nat).toFloat) - 3.0
  let af := fun (m k : Nat) => (((m*2+k) % 5 : Nat).toFloat) - 2.0
  let mut wA : Array Float := #[]
  for e in [0:nE] do for n in [0:N] do for k in [0:K] do wA := wA.push (wf e n k)
  let mut aA : Array Float := #[]
  for m in [0:M] do for k in [0:K] do aA := aA.push (af m k)
  let wf32 ← mkBuf device (nE*N*K); let bf16 ← mkBuf device (nE*N*(K/2))
  let aBuf ← mkBuf device (M*K); let cBuf ← mkBuf device (M*N); let teBuf ← mkBuf device (M/64)
  writeBuffer device wf32 0 (← Hesper.Basic.floatArrayToBytes wA)
  writeBuffer device aBuf 0 (← Hesper.Basic.floatArrayToBytes aA)
  let mut teB : ByteArray := ByteArray.empty
  for t in [0:M/64] do teB := teB.push t.toUInt8; teB := teB.push 0; teB := teB.push 0; teB := teB.push 0
  writeBuffer device teBuf 0 teB
  let rp ← IO.mkRef none
  Hesper.GPUBackend.executeWithConfigCached device (packF32ToF16 (nE*N*(K/2)))
    (("fin",wf32)::("fout",bf16)::List.nil) { numWorkgroups := ((nE*N*(K/2)+255)/256,1,1), workgroupSize := {x:=256} } 1 rp
  let cfg : Hesper.ExecConfig := {
    numWorkgroups := ((N+31)/32, (M+63)/64, 1), workgroupSize := {x:=128},
    extensions := ["f16","chromium_experimental_subgroup_matrix"],
    diagnostics := [("off","chromium.subgroup_matrix_uniformity")] }
  let rr ← IO.mkRef none
  Hesper.GPUBackend.executeWithConfigCached device (Hesper.WGSL.MatMul.matMulTransposeF16WMMARegKernel { M := M, N := N, K := K } 0 0 true nE)
    (("a",aBuf)::("b",bf16)::("c",cBuf)::("tileExpert",teBuf)::List.nil) cfg 1 rr
  let gpu ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device cBuf 0 (M*N*4).toUSize)
  let mut md := 0.0; let mut bad := 0
  for m in [0:M] do for n in [0:N] do
    let e := m / 64
    let mut g := 0.0
    for k in [0:K] do g := g + (af m k) * (wf e n k)
    let d := (gpu.getD (m*N+n) 0.0 - g).abs
    if d > md then md := d
    if d > 0.5 then bad := bad+1
  let v := if bad==0 then "✅ grouped reg-matmul CORRECT" else "❌ WRONG"
  IO.println s!"[grouped M={M} N={N} K={K} nE={nE}] maxDiff={md} bad={bad}/{M*N} → {v}; sample m0={gpu.getD 0 0.0} m64={gpu.getD (64*N) 0.0}"

/-- Pure-MMA throughput probe: load one A,B 8×8 tile, then do `iters` back-to-back
    matrixMultiplyAccumulate into 8 INDEPENDENT accumulators (no memory in the loop) → the raw
    subgroup-matrix peak GFLOPS, to calibrate what % the real kernels hit. -/
def mmaPeakProbe (iters nAcc : Nat) : Hesper.WGSL.Monad.ShaderM Unit := do
  let lid ← Hesper.WGSL.Monad.ShaderM.localId; let tid := Exp.vec3X lid
  let _a ← Hesper.WGSL.Monad.ShaderM.declareInputBuffer "a" (.array (.scalar .f32) 64)
  let _c ← Hesper.WGSL.Monad.ShaderM.declareOutputBuffer "c" (.array (.scalar .f32) 64)
  Hesper.WGSL.Monad.ShaderM.sharedNamed "sA" (.array (.scalar .f16) 64)
  Hesper.WGSL.Monad.ShaderM.sharedNamed "sB" (.array (.scalar .f16) 64)
  Hesper.WGSL.Monad.ShaderM.if_ (Exp.lt tid (Exp.litU32 64)) (do
    let v ← Hesper.WGSL.Monad.ShaderM.readBuffer (ty := .scalar .f32) (n := 64) "a" tid
    Hesper.WGSL.Monad.ShaderM.writeWorkgroup (ty := .scalar .f16) "sA" tid (Exp.toF16 v)
    Hesper.WGSL.Monad.ShaderM.writeWorkgroup (ty := .scalar .f16) "sB" tid (Exp.toF16 v)) (pure ())
  Hesper.WGSL.Monad.ShaderM.barrier
  Hesper.WGSL.Monad.ShaderM.declareMatrixLeftArray  "Ax" .f16 8 8 1 Exp.subgroupMatrixZeroLeft
  Hesper.WGSL.Monad.ShaderM.declareMatrixRightArray "Bx" .f16 8 8 1 Exp.subgroupMatrixZeroRight
  Hesper.WGSL.Monad.ShaderM.declareMatrixResultArray "Cx" .f32 8 8 nAcc Exp.subgroupMatrixZeroResult
  Hesper.WGSL.Monad.ShaderM.loadMatrixLeft  (st := .f16) (m := 8) (k := 8) "Ax" 0 "sA" (Exp.litU32 0) (Exp.litU32 8)
  Hesper.WGSL.Monad.ShaderM.loadMatrixRight (st := .f16) (k := 8) (n := 8) "Bx" 0 "sB" (Exp.litU32 0) (Exp.litU32 8)
  Hesper.WGSL.Monad.ShaderM.loop (Exp.litU32 0) (Exp.litU32 iters) (Exp.litU32 1) fun _ => do
    for i in [0:nAcc] do
      Hesper.WGSL.Monad.ShaderM.matrixMultiplyAccumulateMixed (inSt := .f16) (outSt := .f32) (m := 8) (k := 8) (n := 8) "Cx" i "Ax" 0 "Bx" 0
  for i in [0:nAcc] do
    Hesper.WGSL.Monad.ShaderM.storeMatrixResult (st := .f32) (m := 8) (n := 8) "Cx" i "c" (Exp.litU32 0) (Exp.litU32 8)

def benchPeak (device : Device) : IO Unit := do
  let iters := 2048; let nWg := 1024; let wgThreads := 128
  let aBuf ← mkBuf device 64; let cBuf ← mkBuf device 64
  let cfg : Hesper.ExecConfig := {
    numWorkgroups := (nWg,1,1), workgroupSize := {x:=wgThreads},
    extensions := ["f16","chromium_experimental_subgroup_matrix"],
    diagnostics := [("off","chromium.subgroup_matrix_uniformity")] }
  IO.println "=== MMA PEAK PROBE (pure subgroup-matrix, no memory in loop) — sweep accumulators ==="
  for nAcc in [2, 4, 8, 16, 32] do
    let r ← IO.mkRef none
    Hesper.GPUBackend.executeWithConfigCached device (mmaPeakProbe iters nAcc) (("a",aBuf)::("c",cBuf)::List.nil) cfg 1 r
    let reps := 30
    let t0 ← IO.monoMsNow
    Hesper.GPUBackend.beginBatch device
    for _ in [0:reps] do Hesper.GPUBackend.executeWithConfigCached device (mmaPeakProbe iters nAcc) (("a",aBuf)::("c",cBuf)::List.nil) cfg 1 r
    Hesper.GPUBackend.endBatch device
    let t1 ← IO.monoMsNow
    let ms := (t1-t0).toFloat / reps.toFloat
    let flop := iters.toFloat * nAcc.toFloat * (8.0*8.0*8.0*2.0) * nWg.toFloat * (wgThreads.toFloat/32.0)
    let tflops := flop / (ms/1000.0) / 1.0e12
    IO.println s!"  nAcc={nAcc}: {ms} ms → {tflops} TFLOP/s ({100.0*tflops/34.0}% of 34 f16-ALU peak)"

/-- Golden for the FUSED Q4_K-dequant grouped reg-matmul: build valid Q4_K weight bytes, get the
    reference f32 weights via the (already-validated) dequantQ4KMKernel, CPU-matmul them, and compare
    to the fused kernel (which dequants the SAME bytes in-kernel). f16 weight-rounding tolerance. -/
def checkFusedQ4KCorrect (device : Device) : IO Unit := do
  let M := 128; let N := 32; let K := 256; let nE := 2     -- 2 M-tiles → experts 0,1; 1 Q4_K block/row
  let bRows := nE * N
  let mut bytes : ByteArray := ByteArray.empty
  for r in [0:bRows] do
    bytes := (((bytes.push 0x00).push 0x3C).push 0x00).push 0x30   -- d=1.0 (0x3C00), dmin=0.125 (0x3000)
    for i in [0:12]  do bytes := bytes.push (((r + i*3) % 12 + 1).toUInt8)    -- 6-bit sub-scales/mins
    for i in [0:128] do bytes := bytes.push (((r*7 + i*13) % 256).toUInt8)    -- 4-bit quants
  let qbuf ← mkBuf device (bytes.size / 4)
  writeBuffer device qbuf 0 bytes
  let wbuf ← mkBuf device (bRows*K)
  let rd ← IO.mkRef none
  Hesper.GPUBackend.executeWithConfigCached device (Hesper.Quantization.Q4_K_M.dequantQ4KMKernel (bRows*K))
    (("data",qbuf)::("output",wbuf)::List.nil) { numWorkgroups := ((bRows*K+255)/256,1,1), workgroupSize := {x:=256} } 1 rd
  let wF32 ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device wbuf 0 (bRows*K*4).toUSize)
  let af := fun (m k : Nat) => (((m*2+k) % 5 : Nat).toFloat) - 2.0
  let mut aA : Array Float := #[]
  for m in [0:M] do for k in [0:K] do aA := aA.push (af m k)
  let aBuf ← mkBuf device (M*K); writeBuffer device aBuf 0 (← Hesper.Basic.floatArrayToBytes aA)
  let teBuf ← mkBuf device (M/64)
  let mut teB : ByteArray := ByteArray.empty
  for t in [0:M/64] do teB := (((teB.push (t.toUInt8)).push 0).push 0).push 0
  writeBuffer device teBuf 0 teB
  let cBuf ← mkBuf device (M*N)
  let cfg : Hesper.ExecConfig := {
    numWorkgroups := ((N+31)/32, (M+63)/64, 1), workgroupSize := {x:=128},
    extensions := ["f16","chromium_experimental_subgroup_matrix"],
    diagnostics := [("off","chromium.subgroup_matrix_uniformity")] }
  let rr ← IO.mkRef none
  Hesper.GPUBackend.executeWithConfigCached device (Hesper.Quantization.Q4_K_M.q4kMatmulGroupedRegKernel M N K nE)
    (("a",aBuf)::("b",qbuf)::("c",cBuf)::("tileExpert",teBuf)::List.nil) cfg 1 rr
  let gpu ← Hesper.Basic.bytesToFloatArray (← mapBufferRead device cBuf 0 (M*N*4).toUSize)
  let mut md := 0.0; let mut maxRel := 0.0; let mut bad := 0
  for m in [0:M] do for n in [0:N] do
    let e := m/64
    let mut g := 0.0
    for k in [0:K] do g := g + (af m k) * (wF32.getD ((e*N+n)*K + k) 0.0)
    let d := (gpu.getD (m*N+n) 0.0 - g).abs
    let rel := d / (g.abs + 1.0)
    if d > md then md := d
    if rel > maxRel then maxRel := rel
    if rel > 0.02 then bad := bad+1
  let v := if bad==0 then "✅ fused Q4_K reg CORRECT (f16-round tol)" else "❌ WRONG"
  IO.println s!"[fused-q4k M={M} N={N} K={K} nE={nE}] maxDiff={md} maxRel={maxRel} bad(rel>2%)={bad}/{M*N} → {v}; sample m0={gpu.getD 0 0.0} m64={gpu.getD (64*N) 0.0}"

def main : IO Unit := do
  IO.println "=== reg-matmul roofline micro-bench (DiffusionGemma forward shapes, M=262 tokens) ==="
  let inst ← Hesper.init
  let device ← getDevice inst
  checkFusedQ4KCorrect device
  benchPeak device
  checkCorrect device
  checkGroupedCorrect device
  IO.println "=== SIZE SWEEP (square) — does efficiency climb toward 80% as size grows? ==="
  let _ ← benchShape device "512^3 " 512 512 512
  let _ ← benchShape device "1024^3" 1024 1024 1024
  let _ ← benchShape device "2048^3" 2048 2048 2048
  let _ ← benchShape device "3072^3" 3072 3072 3072
  let _ ← benchShape device "4096^3" 4096 4096 4096
  IO.println "=== forward shapes ==="
  -- per-shape time; then the per-step forward SUM (5 full-attn layers + 25 SWA + lm_head),
  -- = the integration ceiling: the real forward can't beat the sum of its matmul GPU times.
  let qF  ← benchShape device "QKV wQ (full) " 262 8192 2816
  let qS  ← benchShape device "QKV wQ (SWA)  " 262 4096 2816
  let kvF ← benchShape device "QKV wK/V(full)" 262 1024 2816
  let kvS ← benchShape device "QKV wK/V (SWA)" 262 2048 2816
  let oF  ← benchShape device "O-proj (full) " 262 2816 8192
  let oS  ← benchShape device "O-proj (SWA)  " 262 2816 4096
  let gu  ← benchShape device "dense gate/up " 262 2112 2816
  let mge ← benchShape device "MoE gate/up   " 6400 1408 2816   -- grouped: maxPadded(~64-pad) rows
  let lm  ← benchShape device "lm_head chunk " 262 32768 2816
  let attn := 5.0*(qF + 2.0*kvF + oF) + 25.0*(qS + 2.0*kvS + oS)
  let dense := 30.0*(2.0*gu)
  let moe := 30.0*mge
  let lmTot := 8.0*lm
  IO.println s!"=== per-step matmul SUM (integration ceiling) ==="
  IO.println s!"  attention QKV+O = {attn} ms | dense gate/up = {dense} ms | MoE gate/up = {moe} ms | lm_head = {lmTot} ms"
  IO.println s!"  TOTAL forward matmul = {attn+dense+moe} ms (+ lm_head {lmTot}) — best case if ALL reg-matmul"

end Examples.Compute.MatmulBench

def main : IO Unit := Examples.Compute.MatmulBench.main
