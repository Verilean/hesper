import Hesper
import Hesper.Backend
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WebGPU.BufferOps
import Hesper.Basic
import Hesper.WGSL.MatMul

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
def peakFlops : Float := 34.0e12
def peakBW    : Float := 400.0e9

def benchShape (device : Device) (name : String) (M N K : Nat) : IO Unit := do
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
  let iters := 50
  let t0 ← IO.monoMsNow
  for _ in [0:iters] do Hesper.GPUBackend.executeWithConfigCached device kern bufs cfg 1 r
  let t1 ← IO.monoMsNow
  let ms := (t1-t0).toFloat / iters.toFloat
  let flops := 2.0 * M.toFloat * N.toFloat * K.toFloat
  let gflops := flops / (ms/1000.0) / 1.0e9
  let bytes := (N*(K/2)*4 + M*K*4 + M*N*4).toFloat   -- f16 weight (u32 packed) + f32 act + f32 out
  let floorMemMs  := bytes / peakBW * 1000.0
  let floorCompMs := flops / peakFlops * 1000.0
  let floor := max floorMemMs floorCompMs
  let bound := if floorMemMs > floorCompMs then "MEM" else "COMPUTE"
  IO.println s!"{name} [M={M} N={N} K={K}]: {ms} ms/iter, {gflops} GFLOPS ({100.0*gflops*1.0e9/peakFlops}% peak) | floor={floor}ms ({bound}: mem {floorMemMs} / comp {floorCompMs}) → {ms/floor}× above floor"

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

def main : IO Unit := do
  IO.println "=== reg-matmul roofline micro-bench (DiffusionGemma forward shapes, M=262 tokens) ==="
  let inst ← Hesper.init
  let device ← getDevice inst
  checkCorrect device
  checkGroupedCorrect device
  benchShape device "QKV wQ (full) " 262 8192 2816
  benchShape device "QKV wQ (SWA)  " 262 4096 2816
  benchShape device "QKV wK/V (SWA)" 262 2048 2816
  benchShape device "O-proj (full) " 262 2816 8192
  benchShape device "dense gate/up " 262 2112 2816
  benchShape device "dense down    " 262 2816 2112
  benchShape device "lm_head chunk " 262 32768 2816

end Examples.Compute.MatmulBench

def main : IO Unit := Examples.Compute.MatmulBench.main
