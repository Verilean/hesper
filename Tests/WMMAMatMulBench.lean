import Hesper
import Hesper.WGSL.MatMul
import Hesper.WGSL.Execute
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WebGPU.BufferOps
import Hesper.Float16
import Hesper.Basic

/-!
# WMMA MatMul Microbenchmark

Compares the non-WMMA `matMulTransposeF16Kernel` against the new
`matMulTransposeF16WMMAKernel` across a range of shapes, reporting
ms/call and GFLOP/s for each.

Both kernels compute `C = A @ B^T` with A : f32, B : f16 packed in u32,
C : f32. Dispatches are batched via `beginBatch/endBatch` so we measure
aggregate GPU time rather than per-call driver round-trips.
-/

open Hesper.WebGPU
open Hesper.WGSL

namespace Tests.WMMAMatMulBench

structure Shape where
  M : Nat
  N : Nat
  K : Nat
  name : String

def uploadF32 (device : Device) (arr : Array Float) : IO Buffer := do
  let buf ← createBuffer device {
    size := (arr.size * 4).toUSize
    usage := [.storage, .copyDst, .copySrc]
    mappedAtCreation := false
  }
  let bytes ← Hesper.Basic.floatArrayToBytes arr
  writeBuffer device buf 0 bytes
  pure buf

def uploadF16AsU32 (device : Device) (arr : Array Float) : IO Buffer := do
  let fa : FloatArray := FloatArray.mk arr
  let f16 ← Hesper.Float16.fromFloatArray fa
  let buf ← createBuffer device {
    size := f16.data.size.toUSize
    usage := [.storage, .copyDst, .copySrc]
    mappedAtCreation := false
  }
  writeBuffer device buf 0 f16.data
  pure buf

def benchShape (device : Device) (shape : Shape) (warmups iters : Nat) : IO Unit := do
  -- Deterministic trivial data — values don't matter for timing.
  let mut aData : Array Float := Array.empty
  for i in [:shape.M * shape.K] do
    aData := aData.push ((i % 17).toFloat * 0.01 - 0.08)
  let mut bData : Array Float := Array.empty
  for i in [:shape.N * shape.K] do
    bData := bData.push ((i % 19).toFloat * 0.01 - 0.09)
  let aBuf ← uploadF32 device aData
  let bBuf ← uploadF16AsU32 device bData
  let cBuf ← createBuffer device {
    size := (shape.M * shape.N * 4).toUSize
    usage := [.storage, .copyDst, .copySrc]
    mappedAtCreation := false
  }

  let cfg : MatMul.Config := { M := shape.M, N := shape.N, K := shape.K }
  let nonWmmaCfg : Execute.ExecutionConfig :=
    Execute.ExecutionConfig.dispatch1D (shape.M * shape.N)
  let wmmaCfg : Execute.ExecutionConfig := {
    funcName := "main"
    workgroupSize := { x := 32, y := 1, z := 1 }
    numWorkgroups := (shape.N / 16, shape.M / 16, 1)
    extensions := ["f16", "chromium_experimental_subgroup_matrix"]
    diagnostics := [("off", "chromium.subgroup_matrix_uniformity")]
  }

  let nonWmmaBufs : List (String × Buffer) :=
    [("a", aBuf), ("b", bBuf), ("c", cBuf)]
  let wmmaBufs : List (String × Buffer) :=
    [("a", aBuf), ("b", bBuf), ("c", cBuf)]

  -- Cache keys: without these, executeShaderNamed regenerates the WGSL
  -- source on every call, which for the WMMA kernel is a ~500KB string that
  -- takes tens of milliseconds to emit and hash — dominating per-call time
  -- even on cache hits.
  let nonWmmaKey : UInt64 := hash ("mm-f16", shape.M, shape.K, shape.N)
  let wmmaKey    : UInt64 := hash ("mm-wmma", shape.M, shape.K, shape.N)
  let runNonWmma : IO Unit := do
    Execute.executeShaderNamed device
      (MatMul.matMulTransposeF16Kernel cfg)
      nonWmmaBufs nonWmmaCfg (some nonWmmaKey)
  let runWmma : IO Unit := do
    Execute.executeShaderNamed device
      (MatMul.matMulTransposeF16WMMAKernel cfg)
      wmmaBufs wmmaCfg (some wmmaKey)

  let flops := 2.0 * shape.M.toFloat * shape.N.toFloat * shape.K.toFloat

  let timeOne (kind : String) (run : IO Unit) : IO Unit := do
    -- Warmup
    Execute.beginBatch device
    for _ in [:warmups] do run
    Execute.endBatch device
    let _ ← Hesper.WebGPU.BufferOps.downloadFloatArray device cBuf 1
    -- Timed
    let start ← IO.monoNanosNow
    Execute.beginBatch device
    for _ in [:iters] do run
    Execute.endBatch device
    let _ ← Hesper.WebGPU.BufferOps.downloadFloatArray device cBuf 1
    let stop ← IO.monoNanosNow
    let ms := (stop - start).toFloat / 1_000_000.0
    let msPerCall := ms / iters.toFloat
    let gflops := flops / (msPerCall * 1_000_000.0)
    IO.println s!"    {kind}  {msPerCall} ms/call  {gflops} GFLOP/s"

  IO.println s!"  {shape.name}  M={shape.M} K={shape.K} N={shape.N}  (FLOPs={flops.toString})"
  timeOne "non-WMMA" runNonWmma
  timeOne "WMMA    " runWmma

def shapes : List Shape :=
  [ { M := 16,  K := 2560, N := 640,  name := "gemma4-down-ish-M16" }
  , { M := 32,  K := 2560, N := 640,  name := "gemma4-down-ish-M32" }
  , { M := 64,  K := 2560, N := 640,  name := "gemma4-down-ish-M64" }
  , { M := 128, K := 2560, N := 640,  name := "gemma4-down-ish-M128" }
  , { M := 16,  K := 2560, N := 2560, name := "gemma4-square-M16" }
  , { M := 32,  K := 2560, N := 2560, name := "gemma4-square-M32" }
  , { M := 64,  K := 2560, N := 2560, name := "gemma4-square-M64" }
  , { M := 128, K := 2560, N := 2560, name := "gemma4-square-M128" }
  , { M := 256, K := 2560, N := 2560, name := "gemma4-square-M256" }
  ]

def main : IO Unit := do
  IO.println "═══════════════════════════════════════════════"
  IO.println "  WMMA MatMul Microbenchmark"
  IO.println "═══════════════════════════════════════════════"
  let inst ← Hesper.init
  let device ← Hesper.WebGPU.getDevice inst
  let hasSm  ← Execute.hasSubgroupMatrixSupport device
  let hasF16 ← Execute.hasShaderF16Support device
  if !hasSm || !hasF16 then
    IO.println "  ⚠ Adapter lacks SubgroupMatrix or ShaderF16 — skipping."
    return
  IO.println ""
  for s in shapes do
    benchShape device s 5 50

end Tests.WMMAMatMulBench

def main : IO Unit := Tests.WMMAMatMulBench.main
