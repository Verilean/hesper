import Hesper
import Hesper.Backend
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WebGPU.BufferOps
import Hesper.Basic
import Hesper.WGSL.MatMul

open Hesper.WebGPU

namespace Examples.Compute.MatmulBench

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

def main : IO Unit := do
  IO.println "=== reg-matmul roofline micro-bench (DiffusionGemma forward shapes, M=262 tokens) ==="
  let inst ← Hesper.init
  let device ← getDevice inst
  benchShape device "QKV wQ (full) " 262 8192 2816
  benchShape device "QKV wQ (SWA)  " 262 4096 2816
  benchShape device "QKV wK/V (SWA)" 262 2048 2816
  benchShape device "O-proj (full) " 262 2816 8192
  benchShape device "dense gate/up " 262 2112 2816
  benchShape device "dense down    " 262 2816 2112
  benchShape device "lm_head chunk " 262 32768 2816

end Examples.Compute.MatmulBench

def main : IO Unit := Examples.Compute.MatmulBench.main
