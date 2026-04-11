import Hesper.Backend
import Hesper.Backend.CUDA
import Hesper.Backend.WebGPU
import Hesper.Layers.Embedding
import Hesper.Layers.RMSNorm
import Hesper.TTT.Kernels
import Hesper.WGSL.MatMul
import Hesper.WebGPU.Device
import Hesper

/-!
# WebGPU vs CUDA PTX Benchmark

Compares inference performance on the same kernels using both backends.
Tests: matVec, RMSNorm, Embedding, and a multi-layer pipeline.
-/

open Hesper
open Hesper.TTT.Kernels

-- Float helpers
private def f64ToF32Bits (f : Float) : UInt32 :=
  let b := f.toBits; let s := (b >>> 63) &&& 1; let e := (b >>> 52) &&& 0x7FF
  let m := b &&& 0x000FFFFFFFFFFFFF
  if e == 0 then 0
  else let e32 : Int := e.toNat - 1023 + 127
    if e32 <= 0 then 0 else if e32 >= 255 then (s.toUInt32 <<< 31) ||| ((0xFF : UInt32) <<< 23)
    else (s.toUInt32 <<< 31) ||| (e32.toNat.toUInt32 <<< 23) ||| ((m >>> 29).toUInt32 &&& (0x7FFFFF : UInt32))

private def packFloats (arr : Array Float) : ByteArray :=
  arr.foldl (fun acc f => let bits := f64ToF32Bits f
    acc.push bits.toUInt8 |>.push (bits>>>8).toUInt8 |>.push (bits>>>16).toUInt8 |>.push (bits>>>24).toUInt8
  ) ByteArray.empty

/-- Benchmark a single operation, return average μs over N runs -/
def benchmarkOp (name : String) (warmup : Nat) (runs : Nat) (op : IO Unit) : IO Float := do
  -- Warmup
  for _ in List.range warmup do op
  -- Timed runs
  let t0 ← IO.monoNanosNow
  for _ in List.range runs do op
  let t1 ← IO.monoNanosNow
  let avgUs := (t1 - t0).toFloat / (runs.toFloat * 1000.0)
  IO.println s!"  {name}: {avgUs} μs/run ({runs} runs)"
  return avgUs

/-- Run benchmarks on a specific backend -/
def runBenchmarks [GPUBackend β] (ctx : β) (backendName : String) : IO (Array (String × Float)) := do
  IO.println s!"\n═══ {backendName} Backend ═══"
  let mut results : Array (String × Float) := #[]

  -- ── Bench 1: matVec (256 → 256) ──
  let dim := 256
  let wBuf ← GPUBackend.allocBuffer ctx (dim * dim * 4).toUSize
  let xBuf ← GPUBackend.allocBuffer ctx (dim * 4).toUSize
  let yBuf ← GPUBackend.allocBuffer ctx (dim * 4).toUSize
  let wData := Array.range (dim * dim) |>.map (fun i => (i % 7).toFloat * 0.01)
  let xData := Array.range dim |>.map (fun i => (i % 5).toFloat * 0.1)
  GPUBackend.writeBuffer ctx wBuf (packFloats wData)
  GPUBackend.writeBuffer ctx xBuf (packFloats xData)

  let t ← benchmarkOp "matVec 256×256" 10 100 do
    executeMatVec ctx wBuf xBuf yBuf dim dim
  results := results.push ("matVec 256×256", t)

  GPUBackend.freeBuffer ctx wBuf
  GPUBackend.freeBuffer ctx xBuf
  GPUBackend.freeBuffer ctx yBuf

  -- ── Bench 2: matVec (2560 → 2560) — BitNet-scale ──
  let dim2 := 2560
  let wBuf2 ← GPUBackend.allocBuffer ctx (dim2 * dim2 * 4).toUSize
  let xBuf2 ← GPUBackend.allocBuffer ctx (dim2 * 4).toUSize
  let yBuf2 ← GPUBackend.allocBuffer ctx (dim2 * 4).toUSize
  let wData2 := Array.range (dim2 * dim2) |>.map (fun i => (i % 11).toFloat * 0.001)
  let xData2 := Array.range dim2 |>.map (fun i => (i % 7).toFloat * 0.01)
  GPUBackend.writeBuffer ctx wBuf2 (packFloats wData2)
  GPUBackend.writeBuffer ctx xBuf2 (packFloats xData2)

  let t2 ← benchmarkOp "matVec 2560×2560" 5 50 do
    executeMatVec ctx wBuf2 xBuf2 yBuf2 dim2 dim2
  results := results.push ("matVec 2560×2560", t2)

  GPUBackend.freeBuffer ctx wBuf2
  GPUBackend.freeBuffer ctx xBuf2
  GPUBackend.freeBuffer ctx yBuf2

  -- ── Bench 3: RMSNorm (2560 dim) ──
  let normDim := 2560
  let scaleBuf ← GPUBackend.allocBuffer ctx (normDim * 4).toUSize
  GPUBackend.writeBuffer ctx scaleBuf (packFloats (Array.replicate normDim 1.0))
  let normLayer ← Hesper.Layers.RMSNorm.create ctx
    { dim := normDim, eps := 1.0e-5 } (packFloats (Array.replicate normDim 1.0))
  let inBuf ← GPUBackend.allocBuffer ctx (normDim * 4).toUSize
  let outBuf ← GPUBackend.allocBuffer ctx (normDim * 4).toUSize
  GPUBackend.writeBuffer ctx inBuf (packFloats (Array.range normDim |>.map (fun i => i.toFloat * 0.01)))

  let t3 ← benchmarkOp "RMSNorm 2560" 10 100 do
    Hesper.Layers.RMSNorm.forward ctx normLayer inBuf outBuf
  results := results.push ("RMSNorm 2560", t3)

  GPUBackend.freeBuffer ctx scaleBuf
  GPUBackend.freeBuffer ctx inBuf
  GPUBackend.freeBuffer ctx outBuf

  -- ── Bench 4: vecAdd (10000 elements) ──
  let n := 10000
  let aBuf ← GPUBackend.allocBuffer ctx (n * 4).toUSize
  let bBuf ← GPUBackend.allocBuffer ctx (n * 4).toUSize
  let cBuf ← GPUBackend.allocBuffer ctx (n * 4).toUSize

  let t4 ← benchmarkOp "vecAdd 10000" 10 200 do
    executeVecAdd ctx aBuf bBuf cBuf n
  results := results.push ("vecAdd 10000", t4)

  GPUBackend.freeBuffer ctx aBuf
  GPUBackend.freeBuffer ctx bBuf
  GPUBackend.freeBuffer ctx cBuf

  -- ── Bench 5: Pipeline (embed → RMSNorm → matVec) ──
  let vocabSize := 256; let hiddenDim := 512
  let embData := Array.range (vocabSize * hiddenDim) |>.map (fun i => (i % 13).toFloat * 0.01)
  let embLayer ← Hesper.Layers.Embedding.createFromFloat32 ctx
    { vocabSize, dim := hiddenDim } (packFloats embData)
  let normLayer2 ← Hesper.Layers.RMSNorm.create ctx
    { dim := hiddenDim, eps := 1.0e-5 } (packFloats (Array.replicate hiddenDim 1.0))
  let wPipeBuf ← GPUBackend.allocBuffer ctx (vocabSize * hiddenDim * 4).toUSize
  GPUBackend.writeBuffer ctx wPipeBuf (packFloats embData)
  let tokenBuf ← GPUBackend.allocBuffer ctx 4
  GPUBackend.writeBuffer ctx tokenBuf (ByteArray.mk #[5, 0, 0, 0])
  let hidBuf ← GPUBackend.allocBuffer ctx (hiddenDim * 4).toUSize
  let normBuf2 ← GPUBackend.allocBuffer ctx (hiddenDim * 4).toUSize
  let logitBuf ← GPUBackend.allocBuffer ctx (vocabSize * 4).toUSize

  let t5 ← benchmarkOp "Pipeline embed→norm→matVec" 5 50 do
    Hesper.Layers.Embedding.forward ctx embLayer tokenBuf hidBuf 1 1
    Hesper.Layers.RMSNorm.forward ctx normLayer2 hidBuf normBuf2
    executeMatVec ctx wPipeBuf normBuf2 logitBuf vocabSize hiddenDim
  results := results.push ("Pipeline", t5)

  GPUBackend.freeBuffer ctx wPipeBuf
  GPUBackend.freeBuffer ctx tokenBuf
  GPUBackend.freeBuffer ctx hidBuf
  GPUBackend.freeBuffer ctx normBuf2
  GPUBackend.freeBuffer ctx logitBuf

  return results

def main : IO Unit := do
  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║  WebGPU vs CUDA PTX Benchmark               ║"
  IO.println "╚══════════════════════════════════════════════╝"

  -- WebGPU
  let inst ← Hesper.init
  let device ← Hesper.WebGPU.getDevice inst
  let webgpuResults ← runBenchmarks device "WebGPU"

  -- CUDA
  let cudaCtx ← Hesper.CUDAContext.init
  let cudaResults ← runBenchmarks cudaCtx "CUDA PTX"

  -- Comparison
  IO.println "\n═══ Comparison ═══"
  IO.println "Benchmark                      WebGPU (μs)     CUDA (μs)       Speedup"
  IO.println (String.mk (List.replicate 75 '-'))
  for i in [:webgpuResults.size] do
    let (name, wt) := webgpuResults[i]!
    let (_, ct) := cudaResults[i]!
    let speedup := wt / ct
    IO.println s!"  {name}: WebGPU={wt} μs, CUDA={ct} μs, {speedup}×"
