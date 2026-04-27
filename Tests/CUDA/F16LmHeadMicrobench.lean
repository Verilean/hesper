import Hesper.Backend.CUDA
import Hesper.CUDA.FFI
import Hesper.CUDA.Buffer
import Hesper.WGSL.MatMul
import Hesper.WGSL.MatMulF16Variants

set_option maxRecDepth 2048

/-!
# F16 lm_head matmul microbenchmark

Same shape as Gemma 4's pre-dequantised LM head (M=1, K=2560, N=262144).
Compares per-call GPU time across block-size variants and against the
existing `matMulTransposeF16BlockCoopKernel` baseline.

llama.cpp at this shape uses `mul_mat_vec_f<half,float,1,256>` (block_size=256,
8 warps cooperative), measured at ~114 µs/call in the doc 63 nsys trace.

Usage: `lake exe cuda-f16-lmhead-microbench`
-/

open Hesper
open Hesper.CUDA
open Hesper.WGSL.Monad (ShaderM)

private def f64ToF32Bits (f : Float) : UInt32 :=
  let b := f.toBits
  let s := (b >>> 63) &&& 1
  let e64 := (b >>> 52) &&& 0x7FF
  if e64 == 0 then 0
  else
    let eUnb : Int := Int.ofNat e64.toNat - 1023
    let e32i : Int := eUnb + 127
    if e32i ≤ 0 then 0
    else if e32i ≥ 255 then (s.toUInt32 <<< 31) ||| ((0xFF : UInt32) <<< 23)
    else
      let m32 : UInt32 := ((b >>> 29).toUInt32) &&& (0x7FFFFF : UInt32)
      let e32 : UInt32 := e32i.toNat.toUInt32
      (s.toUInt32 <<< 31) ||| (e32 <<< 23) ||| m32

private def packF32 (arr : Array Float) : ByteArray :=
  arr.foldl (init := ByteArray.empty) fun acc f =>
    let b := f64ToF32Bits f
    acc.push b.toUInt8 |>.push (b >>> 8).toUInt8
        |>.push (b >>> 16).toUInt8 |>.push (b >>> 24).toUInt8

private def lcg (seed : UInt64) : UInt64 := seed * 6364136223846793005 + 1442695040888963407

private def benchVariant
    (ctx : CUDAContext) (name : String)
    (kernel : ShaderM Unit)
    (cfg : Hesper.WGSL.MatMul.Config)
    (workgroupSize : Nat)
    (aBuf bBuf cBuf : CUDABuffer) : IO Unit := do
  let execConfig : Hesper.ExecConfig := {
    numWorkgroups := (cfg.N, 1, 1)
    workgroupSize := { x := workgroupSize, y := 1, z := 1 }
    extensions := ["subgroups"]
  }
  let cacheKey : UInt64 := hash ("f16-lmhead-bench", name, cfg.N, cfg.K, workgroupSize)
  let cacheRef ← GPUBackend.newCacheRef (β := CUDAContext)

  -- Warmup (3 iters).
  for _ in List.range 3 do
    GPUBackend.executeWithConfigCached ctx kernel
      [("a", aBuf), ("b", bBuf), ("c", cBuf)] execConfig cacheKey cacheRef
  let _ ← Hesper.CUDA.cuMemcpyDtoH cBuf.ptr 4

  -- Timed loop.
  let iters := 30
  let t0 ← IO.monoNanosNow
  for _ in List.range iters do
    GPUBackend.executeWithConfigCached ctx kernel
      [("a", aBuf), ("b", bBuf), ("c", cBuf)] execConfig cacheKey cacheRef
  let _ ← Hesper.CUDA.cuMemcpyDtoH cBuf.ptr 4
  let t1 ← IO.monoNanosNow
  let perCallUs : Float := (t1 - t0).toFloat / (iters.toFloat * 1000.0)
  -- BW: weight read = N * K/2 * 4 bytes (packed half2 u32).
  let totalWB : Float := (cfg.N * cfg.K / 2 * 4).toFloat
  let bwGBs : Float := totalWB / (perCallUs * 1e-6) / 1e9
  IO.println s!"  {name}: {perCallUs} µs/call  ({bwGBs} GB/s)"

def main : IO Unit := do
  IO.println "════════════════════════════════════════════════════════════"
  IO.println " F16 lm_head matmul microbenchmark"
  IO.println " Shape: M=1, K=2560, N=262144 (Gemma 4 E4B vocab × hidden)"
  IO.println "════════════════════════════════════════════════════════════"
  IO.println "[startup] init CUDA..."
  let cfg : Hesper.WGSL.MatMul.Config := { M := 1, N := 262144, K := 2560 }
  let ctx ← CUDAContext.init
  let aBytes : USize := (cfg.K * 4).toUSize
  let bBytes : USize := (cfg.N * cfg.K / 2 * 4).toUSize  -- packed half2 u32
  let cBytes : USize := (cfg.N * 4).toUSize

  IO.println s!"a: {aBytes / 1024} KB"
  IO.println s!"b: {bBytes / 1024 / 1024} MB"
  IO.println s!"c: {cBytes / 1024 / 1024} MB"
  IO.println ""

  IO.println "[alloc] aBuf..."
  let aBuf ← GPUBackend.allocBuffer ctx aBytes
  IO.println "[alloc] bBuf..."
  let bBuf ← GPUBackend.allocBuffer ctx bBytes
  IO.println "[alloc] cBuf..."
  let cBuf ← GPUBackend.allocBuffer ctx cBytes
  IO.println "[alloc] done"

  -- Fill A with deterministic pseudo-random f32.
  let mut aArr := Array.mkEmpty cfg.K
  for i in [0:cfg.K] do
    aArr := aArr.push (Float.sin (i.toFloat * 0.01))
  GPUBackend.writeBuffer ctx aBuf (packF32 aArr)
  -- B: leave as default-allocated (zeros) — content doesn't affect timing,
  -- only memory traffic patterns.

  IO.println "── Per-call timing (warmup 3 + timed 30) ──"
  IO.println "[bench] BlockCoop bs=32..."
  benchVariant ctx "[Hesper] BlockCoop bs=32 (current default)"
    (Hesper.WGSL.MatMul.matMulTransposeF16BlockCoopKernel cfg) cfg 32 aBuf bBuf cBuf
  IO.println "[bench] RowBlock bs=32..."
  benchVariant ctx "[Hesper] RowBlock  bs=32 "
    (Hesper.WGSL.MatMul.matMulTransposeF16RowBlockKernel cfg 32) cfg 32 aBuf bBuf cBuf
  IO.println "[bench] RowBlock bs=64..."
  benchVariant ctx "[Hesper] RowBlock  bs=64 "
    (Hesper.WGSL.MatMul.matMulTransposeF16RowBlockKernel cfg 64) cfg 64 aBuf bBuf cBuf
  IO.println "[bench] RowBlock bs=128..."
  benchVariant ctx "[Hesper] RowBlock  bs=128"
    (Hesper.WGSL.MatMul.matMulTransposeF16RowBlockKernel cfg 128) cfg 128 aBuf bBuf cBuf
  IO.println "[bench] RowBlock bs=256..."
  benchVariant ctx "[Hesper] RowBlock  bs=256"
    (Hesper.WGSL.MatMul.matMulTransposeF16RowBlockKernel cfg 256) cfg 256 aBuf bBuf cBuf

  IO.println ""
  IO.println "── Reference ──"
  IO.println "  llama.cpp mul_mat_vec_f<half,float,1,256>: ~114 µs/call (doc 63 nsys)"
  IO.println ""
  IO.println "Notes:"
  IO.println "  - Per-call time includes Lean wrapper + cuLaunchKernel + GPU exec"
  IO.println "  - llama.cpp 114 µs is in-process nsys timing (kernel only)"
  IO.println "  - Memory BW: cfg.N * cfg.K / 2 * 4 bytes / time"
  IO.println "  - Theoretical peak: ~504 GB/s (RTX 4070 Ti GDDR6X)"

  GPUBackend.freeBuffer ctx aBuf
  GPUBackend.freeBuffer ctx bBuf
  GPUBackend.freeBuffer ctx cBuf
