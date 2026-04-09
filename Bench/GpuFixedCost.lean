import Hesper
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WGSL.Monad
import Hesper.WGSL.Execute
import Hesper.WGSL.Exp

/-!
# GPU Fixed-Cost Microbenchmark

Measures the components of per-dispatch cost to isolate where the
Gemma 4 Q4_K mat-vec kernel's ~0.43 ms/call time is actually being
spent. Five experiments:

1. **1×1 empty dispatch** (1 WG × 1 thread, trivial write): pure
   dispatch + deviceWait round-trip.
2. **N×1 empty dispatch** (N WGs × 1 thread, trivial write): scaling
   cost of WG launch.
3. **N×32 empty dispatch** (N WGs × 32 threads, trivial write): same
   as #2 but with full subgroup — adds subgroup launch cost.
4. **Memory-bandwidth test** (N WGs × 32 threads, each reads S f32
   from a big buffer and sum-writes): measures actual achievable
   memory bandwidth with the block-coop's access pattern.
5. **Compute-bound test** (N WGs × 32 threads, each does K FMAs on
   registers, then writes): measures achievable FMA throughput.

All experiments dispatch the kernel REP times unbatched (each call
`deviceWait`s), divide total time by REP.
-/

open Hesper.WebGPU
open Hesper.WGSL.Monad (ShaderM)
open Hesper.WGSL

-- ========================================================================
-- Kernel definitions
-- ========================================================================

/-- Minimal kernel: one thread writes 0.0 to `out[0]`. -/
def emptyKernel : Hesper.WGSL.Monad.ShaderM Unit := do
  let _ ← ShaderM.declareOutputBuffer "out" (.array (.scalar .f32) 1)
  let lid ← ShaderM.localId
  let wid ← ShaderM.workgroupId
  let tid := Exp.vec3X lid
  let w := Exp.vec3X wid
  -- Only (WG 0, thread 0) writes, rest return. Still every thread runs
  -- this branch, so WG launch cost is faithfully charged.
  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) (Exp.eq w (Exp.litU32 0))) (do
    ShaderM.writeBuffer (ty := .scalar .f32) "out" (Exp.litU32 0) (Exp.litF32 0.0)
  ) (pure ())

/-- Memory-bandwidth kernel: each WG reads `stride` f32 from `input`
    (strided like the block-coop kernel) and sums into a per-lane acc.
    Lane 0 writes the subgroup-reduced sum. The actual sum is
    meaningless — the point is to force real memory traffic. -/
def memBwKernel (stride : Nat) : Hesper.WGSL.Monad.ShaderM Unit := do
  let _ ← ShaderM.declareInputBuffer "input" (.array (.scalar .f32) stride)
  let _ ← ShaderM.declareOutputBuffer "out" (.array (.scalar .f32) 1)
  let lid ← ShaderM.localId
  let wid ← ShaderM.workgroupId
  let tid := Exp.vec3X lid
  let w := Exp.vec3X wid
  ShaderM.varNamed "acc" (.scalar .f32) (Exp.litF32 0.0)
  let acc : Exp (.scalar .f32) := Exp.var "acc"
  -- Strided read: thread reads input[tid], input[tid+32], ...,
  -- input[tid + 32*(stride/32 - 1)].
  let iters := stride / 32
  for i in [0:iters] do
    let idx := Exp.add tid (Exp.litU32 (i * 32))
    let v ← ShaderM.readBuffer (ty := .scalar .f32) (n := stride) "input" idx
    ShaderM.assign "acc" (Exp.add acc v)
  ShaderM.varNamed "total" (.scalar .f32) (Exp.subgroupAdd acc)
  let total : Exp (.scalar .f32) := Exp.var "total"
  -- Only WG 0 + lane 0 writes (prevents wasting DRAM bw on stores,
  -- and the `total` use keeps the reads live).
  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) (Exp.eq w (Exp.litU32 0))) (do
    ShaderM.writeBuffer (ty := .scalar .f32) "out" (Exp.litU32 0) total
  ) (pure ())

/-- Compute-bound kernel: K inlined FMA ops on register-only data, no
    memory reads (beyond one initial constant), then a write. -/
def computeKernel (k : Nat) : Hesper.WGSL.Monad.ShaderM Unit := do
  let _ ← ShaderM.declareOutputBuffer "out" (.array (.scalar .f32) 1)
  let lid ← ShaderM.localId
  let wid ← ShaderM.workgroupId
  let tid := Exp.vec3X lid
  let w := Exp.vec3X wid
  ShaderM.varNamed "a" (.scalar .f32) (Exp.litF32 1.0)
  let a0 : Exp (.scalar .f32) := Exp.var "a"
  -- K independent FMA accumulations. Use `tid` as a live input so the
  -- compiler cannot fold the whole thing to a constant.
  let tf := Exp.toF32 tid
  for _ in [0:k] do
    ShaderM.assign "a" (Exp.add (Exp.mul a0 (Exp.litF32 1.0000001)) tf)
  ShaderM.if_ (Exp.and (Exp.eq tid (Exp.litU32 0)) (Exp.eq w (Exp.litU32 0))) (do
    ShaderM.writeBuffer (ty := .scalar .f32) "out" (Exp.litU32 0) a0
  ) (pure ())

-- ========================================================================
-- Experiment runner
-- ========================================================================

/-- Run an experiment REP times unbatched (each dispatch is
    `deviceWait`-synced) and return the mean time per dispatch in ms. -/
def runExperiment
    (device : Device)
    (label : String)
    (shader : Hesper.WGSL.Monad.ShaderM Unit)
    (namedBuffers : List (String × Buffer))
    (numWorkgroups : Nat × Nat × Nat)
    (wgSize : Nat)
    (extensions : List String)
    (rep : Nat) : IO Unit := do
  let execConfig : Execute.ExecutionConfig := {
    numWorkgroups := numWorkgroups
    workgroupSize := { x := wgSize, y := 1, z := 1 }
    extensions := extensions
  }
  -- Warmup: compile pipeline + first few dispatches.
  for _ in [0:3] do
    Execute.executeShaderNamed device shader namedBuffers execConfig
  -- Timed loop.
  let start ← IO.monoNanosNow
  for _ in [0:rep] do
    Execute.executeShaderNamed device shader namedBuffers execConfig
  let stop ← IO.monoNanosNow
  let totalNs := stop - start
  let perCallUs := (totalNs.toFloat / 1000.0) / rep.toFloat
  IO.println s!"  {label}: {perCallUs} µs/call ({rep} reps)"

-- ========================================================================
-- Main
-- ========================================================================

def main : IO Unit := do
  IO.println "═══════════════════════════════════════════════════════════"
  IO.println "  GPU Fixed-Cost Microbenchmark (RTX 4070 Ti / Dawn)"
  IO.println "═══════════════════════════════════════════════════════════"
  let inst ← Hesper.init
  let device ← getDevice inst

  -- Shared output buffer (4 bytes is enough, we only write out[0]).
  let outBuf ← createBuffer device {
    size := 16
    usage := [.storage, .copySrc, .copyDst]
    mappedAtCreation := false
  }

  -- Shared input buffer large enough for all memBw tests (16 MB).
  let inSize : USize := 16 * 1024 * 1024
  let inBuf ← createBuffer device {
    size := inSize
    usage := [.storage, .copySrc, .copyDst]
    mappedAtCreation := false
  }

  let rep := 200
  let outBufs : List (String × Buffer) := [("out", outBuf)]
  IO.println ""
  IO.println "─── 1. Dispatch-cost scaling ───"
  -- 1×1 empty dispatch: pure RTT
  runExperiment device "1 WG × 1 thread (empty)"
    emptyKernel outBufs (1, 1, 1) 1 ([] : List String) rep
  -- Scaling WG count, 1 thread each
  runExperiment device "256 WG × 1 thread (empty)"
    emptyKernel outBufs (256, 1, 1) 1 ([] : List String) rep
  runExperiment device "2560 WG × 1 thread (empty)"
    emptyKernel outBufs (2560, 1, 1) 1 ([] : List String) rep
  runExperiment device "10240 WG × 1 thread (empty)"
    emptyKernel outBufs (10240, 1, 1) 1 ([] : List String) rep

  IO.println ""
  IO.println "─── 2. Dispatch scaling × 32 threads ───"
  runExperiment device "1 WG × 32 threads (empty)"
    emptyKernel outBufs (1, 1, 1) 32 ([] : List String) rep
  runExperiment device "256 WG × 32 threads (empty)"
    emptyKernel outBufs (256, 1, 1) 32 ([] : List String) rep
  runExperiment device "2560 WG × 32 threads (empty)"
    emptyKernel outBufs (2560, 1, 1) 32 ([] : List String) rep
  runExperiment device "10240 WG × 32 threads (empty)"
    emptyKernel outBufs (10240, 1, 1) 32 ([] : List String) rep

  IO.println ""
  IO.println "─── 3. Memory bandwidth (strided read) ───"
  -- 2560 f32 / WG, 2560 WGs → 2560² f32 ≈ 26 MB
  let memBufs : List (String × Buffer) := [("input", inBuf), ("out", outBuf)]
  let wg1 : Nat × Nat × Nat := (2560, 1, 1)
  let wg2 : Nat × Nat × Nat := (10240, 1, 1)
  let subExt : List String := ["subgroups"]
  runExperiment device "2560 WG × 32 th, read 2560 f32 each"
    (memBwKernel 2560) memBufs wg1 32 subExt rep
  runExperiment device "2560 WG × 32 th, read 10240 f32 each"
    (memBwKernel 10240) memBufs wg1 32 subExt rep
  runExperiment device "10240 WG × 32 th, read 2560 f32 each"
    (memBwKernel 2560) memBufs wg2 32 subExt rep

  IO.println ""
  IO.println "─── 4. Compute bound (inline FMAs) ───"
  -- 10240 WG × 32 thread × 256 FMA ≈ 80M FMA, ~2 µs at peak
  runExperiment device "10240 WG × 32 th, 256 FMAs each"
    (computeKernel 256) outBufs (10240, 1, 1) 32 ([] : List String) rep
  -- 10240 × 32 × 1024 FMA ≈ 320M FMA, ~8 µs at peak
  runExperiment device "10240 WG × 32 th, 1024 FMAs each"
    (computeKernel 1024) outBufs (10240, 1, 1) 32 ([] : List String) rep

  IO.println ""
  IO.println "─── Interpretation ───"
  IO.println "  Per-dispatch cost = value from §1/§2 'empty' experiments."
  IO.println "  Realised memory BW = (WG × 32 × stride × 4B) / time."
  IO.println "  Achievable FMA/s   = (WG × 32 × K) / time."
  IO.println "  Compare with Q4_K ffnDown: 0.434 ms/call, 14.75 MB weight, 52M FMA."
