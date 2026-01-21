import Hesper.Simd

/-!
# SIMD CPU Backend Benchmark

Compares SIMD-optimized vector addition against pure Lean implementation.

Demonstrates FFI integration with architecture-specific optimizations:
- **AVX2** on x86_64 (8x parallelism)
- **NEON** on ARM64 (4x parallelism)
-/

open Hesper.Simd

/-- High-precision timer (nanoseconds) -/
def getTimeNs : IO UInt64 := do
  let start ← IO.monoNanosNow
  return start.toUInt64

/-- Generate test FloatArray with values 0.0, 1.0, 2.0, ... -/
def generateTestArray (size : Nat) : FloatArray :=
  let data := Array.range size |>.map (·.toFloat)
  FloatArray.mk data

/-- Benchmark a computation and return duration in nanoseconds -/
def benchmark (label : String) (f : Unit → α) : IO UInt64 := do
  -- Warm-up run
  let _ := f ()

  -- Actual measurement
  let start ← getTimeNs
  let _ := f ()
  let finish ← getTimeNs

  let duration := finish - start
  IO.println s!"{label}: {duration}ns ({duration.toFloat / 1_000_000.0}ms)"
  return duration

def main : IO Unit := do
  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║   Hesper SIMD CPU Backend Benchmark         ║"
  IO.println "╚══════════════════════════════════════════════╝"
  IO.println ""

  -- Display backend information
  try
    let backend ← backendInfo
    IO.println s!"Backend: {backend}"
  catch e =>
    IO.println s!"Backend info failed: {e}"
  IO.println ""

  -- Test sizes
  let sizes := [1000, 10_000, 100_000, 1_000_000, 10_000_000]

  for size in sizes do
    IO.println s!"─── Vector Size: {size} elements ({size * 4} bytes) ───"

    -- Generate test data
    let a := generateTestArray size
    let b := generateTestArray size

    -- Benchmark naive (pure Lean) implementation
    let naiveDuration ← benchmark "  Naive (Lean)" fun _ =>
      naiveAdd a b

    -- Benchmark SIMD implementation
    let simdDuration ← benchmark "  SIMD (FFI) " fun _ =>
      simdAdd a b

    -- Calculate speedup
    let speedup := naiveDuration.toFloat / simdDuration.toFloat
    IO.println s!"  Speedup: {speedup}x"

    -- Verify correctness
    let naiveResult := naiveAdd a b
    let simdResult := simdAdd a b

    if verifyEqual naiveResult simdResult then
      IO.println "  ✓ Results verified (within epsilon)"
    else
      IO.println "  ✗ Results differ!"

    IO.println ""

  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║   Benchmark Complete                         ║"
  IO.println "╚══════════════════════════════════════════════╝"
