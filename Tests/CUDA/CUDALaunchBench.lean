import Hesper.Backend.CUDA
import Hesper.CUDA.FFI
import Hesper.CUDA.Buffer
import Hesper.WGSL.Monad
import Hesper.WGSL.Exp

set_option maxRecDepth 2048

/-!
# Per-launch cost micro-bench

Measure launch overhead at each layer of the hesper stack to find out
where the 14.7 µs/call (graphs OFF) is actually spent. Each layer adds
one piece of the hesper hot path:

  L0 raw FFI         `cuLaunchKernel` direct. Baseline.
  L1 maybeStream     + `cudaCaptureStream.get` IORef check.
  L2 replayCached    + `bumpDispatchOnEmit` + `traceLaunch` + batchQueue check.
  L3 executeCached   + `cacheRef.get` + per-call `namedBuffers.find?` resolve.

Run (graphs OFF):
  lake build cuda-launch-bench
  ./.lake/build/bin/cuda-launch-bench
-/

open Hesper
open Hesper.CUDA
open Hesper.WGSL
open Hesper.WGSL.Monad (ShaderM)

/-- Trivial kernel: 1 thread, 1 workgroup, writes [0] once.
    Enough to not be DCE'd, minimum possible GPU compute. -/
def noopKernel : ShaderM Unit := do
  let lid ← ShaderM.localId
  let tid := Exp.vec3X lid
  let _o ← ShaderM.declareOutputBuffer "output" (.array (.scalar .f32) 1)
  ShaderM.if_ (Exp.eq tid (Exp.litU32 0)) (do
    ShaderM.writeBuffer (ty := .scalar .f32) "output" (Exp.litU32 0) (Exp.litF32 1.0)
  ) (pure ())

def main : IO Unit := do
  let ctx ← CUDAContext.init
  let oBuf ← GPUBackend.allocBuffer ctx 4
  let config : Hesper.ExecConfig := {
    numWorkgroups := (1, 1, 1), workgroupSize := { x := 32, y := 1, z := 1 }
    funcName := "noop_bench"
  }
  let iters : Nat := 20000

  -- Warm up PTX module cache + get a CUfunction for raw FFI.
  let kcr ← GPUBackend.newCacheRef (β := CUDAContext)
  let cacheKey : UInt64 := 0xB0B0B0B0
  for _ in List.range 10 do
    GPUBackend.executeWithConfigCached ctx noopKernel
      [("output", oBuf)] config cacheKey kcr
  let _ ← Hesper.CUDA.cuMemcpyDtoH oBuf.ptr 4

  -- Extract CUfunction from cache for raw FFI bench.
  let cached ← kcr.get
  let func ← match cached with
    | some c => pure c.func
    | none   => throw (IO.userError "warm-up failed to populate cache")
  let args : Array USize := #[oBuf.ptr]

  IO.println "=== per-launch cost (graphs OFF, iters=20000) ==="
  IO.println ""

  -- L0: raw cuLaunchKernel FFI.
  let _ ← Hesper.CUDA.cuMemcpyDtoH oBuf.ptr 4  -- drain pending
  let t0 ← IO.monoNanosNow
  for _ in List.range iters do
    Hesper.CUDA.cuLaunchKernel func 1 1 1 32 1 1 0 args
  let _ ← Hesper.CUDA.cuMemcpyDtoH oBuf.ptr 4
  let t1 ← IO.monoNanosNow
  let l0us : Float := (t1 - t0).toFloat / (iters.toFloat * 1000.0)
  IO.println s!"L0 raw cuLaunchKernel FFI      : {l0us} µs/call"

  -- L2: replayCached (full prepared-dispatch path).
  let c ← match cached with
    | some x => pure x
    | none   => throw (IO.userError "no cache after warmup")
  let _ ← Hesper.CUDA.cuMemcpyDtoH oBuf.ptr 4
  let t4 ← IO.monoNanosNow
  for _ in List.range iters do
    GPUBackend.replayCached ctx c (1, 1, 1)
  let _ ← Hesper.CUDA.cuMemcpyDtoH oBuf.ptr 4
  let t5 ← IO.monoNanosNow
  let l2us : Float := (t5 - t4).toFloat / (iters.toFloat * 1000.0)
  IO.println s!"L2 replayCached                : {l2us} µs/call"

  -- L3: executeWithConfigCached (full hot path with buffer resolve).
  let _ ← Hesper.CUDA.cuMemcpyDtoH oBuf.ptr 4
  let t6 ← IO.monoNanosNow
  for _ in List.range iters do
    GPUBackend.executeWithConfigCached ctx noopKernel
      [("output", oBuf)] config cacheKey kcr
  let _ ← Hesper.CUDA.cuMemcpyDtoH oBuf.ptr 4
  let t7 ← IO.monoNanosNow
  let l3us : Float := (t7 - t6).toFloat / (iters.toFloat * 1000.0)
  IO.println s!"L3 executeWithConfigCached     : {l3us} µs/call"

  -- L4: descriptor launch (metadata-free, Option B+).
  -- Register once, fire by id — no Array alloc, no closure, no buffer resolve.
  Hesper.CUDA.descReset
  let descId ← Hesper.CUDA.descRegister func 1 1 1 32 1 1 0 args
  let _ ← Hesper.CUDA.cuMemcpyDtoH oBuf.ptr 4
  let t8 ← IO.monoNanosNow
  for _ in List.range iters do
    Hesper.CUDA.descLaunch descId 0
  let _ ← Hesper.CUDA.cuMemcpyDtoH oBuf.ptr 4
  let t9 ← IO.monoNanosNow
  let l4us : Float := (t9 - t8).toFloat / (iters.toFloat * 1000.0)
  IO.println s!"L4 descriptor launch           : {l4us} µs/call"

  IO.println ""
  IO.println s!"  L2 overhead vs L0: {l2us - l0us} µs/call"
  IO.println s!"  L3 overhead vs L2: {l3us - l2us} µs/call (= buffer resolve + cacheRef.get)"
  IO.println s!"  L4 vs L0          : {l4us - l0us} µs/call (descriptor minus raw FFI)"
  IO.println s!"  Total Lean wrapper : {l3us - l0us} µs/call"
