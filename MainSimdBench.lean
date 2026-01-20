import Hesper.Simd

/-!
# SIMD CPU Backend Benchmark

Demonstrates SIMD-accelerated vector operations with performance measurements.

**Note:** Array sizes are limited to ≤50K elements due to Lean's Array.range performance.
For production workloads, generate data from external sources or use FFI-based initialization.
-/

open Hesper.Simd

def getTimeNs : IO UInt64 := do
  let t ← IO.monoNanosNow
  return t.toUInt64

def benchmark (label : String) (f : Unit → α) : IO UInt64 := do
  -- Warm-up
  let _ := f ()

  -- Measurement
  let start ← getTimeNs
  let _ := f ()
  let finish ← getTimeNs

  let duration := finish - start
  let ms := duration.toFloat / 1_000_000.0
  IO.println s!"{label}: {ms}ms"
  return duration

def testSize (generateTestArray : Nat → FloatArray) (size : Nat) : IO Unit := do
  IO.println s!"\n─── Size: {size} elements ({size * 8} bytes) ───"

  let a := generateTestArray size
  let b := generateTestArray size

  -- Benchmark naive implementation
  let naiveTime ← benchmark "  Naive (Lean)" fun _ => naiveAdd a b

  -- Benchmark SIMD implementation
  let simdTime ← benchmark "  SIMD (FFI) " fun _ => simdAdd a b

  -- Calculate speedup
  let speedup := naiveTime.toFloat / simdTime.toFloat
  IO.println s!"  Speedup: {speedup}x"

  -- Verify correctness
  let c := naiveAdd a b
  let d := simdAdd a b
  if verifyEqual c d then
    IO.println "  ✓ Results verified"
  else
    IO.println "  ✗ Results differ!"

def main : IO Unit := do
  IO.println "╔═══════════════════════════════════════════════╗"
  IO.println "║   Hesper SIMD CPU Backend Benchmark          ║"
  IO.println "╚═══════════════════════════════════════════════╝"

  let backend ← backendInfo
  IO.println s!"\nBackend: {backend}"

  -- Test sizes (limited by Lean's Array.range performance)
  testSize generateTestArray 1_000
  testSize generateTestArray 10_000
  testSize generateTestArray 50_000

  IO.println "\n╔═══════════════════════════════════════════════╗"
  IO.println "║   Benchmark Complete                          ║"
  IO.println "╚═══════════════════════════════════════════════╝"

where
  generateTestArray (size : Nat) : FloatArray :=
    let data := Array.range size |>.map (·.toFloat)
    FloatArray.mk data
