import Hesper.Simd
import Hesper.Float32
import Hesper.Float16

/-!
# Comprehensive SIMD Performance Benchmarks

Tests all three precision levels (Float64/Float32/Float16) across various array sizes
to measure SIMD performance improvements.
-/

open Hesper.Simd Hesper.Float32 Hesper.Float16

structure BenchResult where
  size : Nat
  naiveTimeMs : Float
  simdTimeMs : Float
  speedup : Float
  deriving Repr

def timeOperation (iterations : Nat) (op : Unit → α) : IO Float := do
  let startTime ← IO.monoNanosNow
  for _ in [0:iterations] do
    let _ := op ()
  let endTime ← IO.monoNanosNow
  let totalNs := (endTime - startTime).toFloat
  let totalMs := totalNs / 1000000.0  -- Convert ns to ms
  return totalMs / iterations.toFloat

def benchmarkFloat64 (size : Nat) (iterations : Nat) : IO BenchResult := do
  -- Generate test data
  let a := FloatArray.mk (Array.range size |>.map (·.toFloat))
  let b := FloatArray.mk (Array.range size |>.map (fun i => (i + 1).toFloat))

  -- Benchmark naive
  let naiveTime ← timeOperation iterations (fun () => naiveAdd a b)

  -- Benchmark SIMD
  let simdTime ← timeOperation iterations (fun () => simdAdd a b)

  let speedup := naiveTime / simdTime

  return { size, naiveTimeMs := naiveTime, simdTimeMs := simdTime, speedup }

def benchmarkFloat32 (size : Nat) (iterations : Nat) : IO BenchResult := do
  -- Generate test data
  let a64 := FloatArray.mk (Array.range size |>.map (·.toFloat))
  let b64 := FloatArray.mk (Array.range size |>.map (fun i => (i + 1).toFloat))
  let a := Hesper.Float32.fromFloatArray a64
  let b := Hesper.Float32.fromFloatArray b64

  -- Benchmark naive (convert to F64, add, convert back)
  let naiveTime ← timeOperation iterations (fun () =>
    let a64 := Hesper.Float32.toFloatArray a
    let b64 := Hesper.Float32.toFloatArray b
    let c64 := naiveAdd a64 b64
    Hesper.Float32.fromFloatArray c64
  )

  -- Benchmark SIMD
  let simdTime ← timeOperation iterations (fun () => Hesper.Float32.simdAdd a b)

  let speedup := naiveTime / simdTime

  return { size, naiveTimeMs := naiveTime, simdTimeMs := simdTime, speedup }

def benchmarkFloat16 (size : Nat) (iterations : Nat) : IO BenchResult := do
  -- Generate test data
  let a64 := FloatArray.mk (Array.range size |>.map (·.toFloat))
  let b64 := FloatArray.mk (Array.range size |>.map (fun i => (i + 1).toFloat))
  let a ← Hesper.Float16.fromFloatArray a64
  let b ← Hesper.Float16.fromFloatArray b64

  -- Benchmark naive (convert to F64, add, convert back)
  let naiveTime ← timeOperation iterations (fun () => do
    let a64 ← Hesper.Float16.toFloatArray a
    let b64 ← Hesper.Float16.toFloatArray b
    let c64 := naiveAdd a64 b64
    Hesper.Float16.fromFloatArray c64
  )

  -- Benchmark SIMD
  let simdTime ← timeOperation iterations (fun () => Hesper.Float16.simdAdd a b)

  let speedup := naiveTime / simdTime

  return { size, naiveTimeMs := naiveTime, simdTimeMs := simdTime, speedup }

def printResult (result : BenchResult) : IO Unit := do
  IO.println s!"  Size: {result.size} | Naive: {result.naiveTimeMs}ms | SIMD: {result.simdTimeMs}ms | Speedup: {result.speedup}x"

def printHeader (precision : String) : IO Unit := do
  IO.println s!"\n═══ {precision} Precision ═══"
  IO.println "  Size         | Naive Time   | SIMD Time    | Speedup"
  IO.println "  ──────────────────────────────────────────────────────"

def calculateMeanSpeedup (results : List BenchResult) : Float :=
  if results.isEmpty then 0.0
  else
    let total := results.foldl (fun acc r => acc + r.speedup) 0.0
    total / results.length.toFloat

def main : IO Unit := do
  IO.println "╔═══════════════════════════════════════════════════════╗"
  IO.println "║   Hesper SIMD Comprehensive Performance Benchmark   ║"
  IO.println "╚═══════════════════════════════════════════════════════╝"

  -- Display backend
  let backend ← backendInfo
  IO.println s!"\nBackend: {backend}"

  -- Test sizes: small to very large
  let sizes := [
    (100, 10000),      -- Small: 800 bytes, 10k iterations
    (1000, 5000),      -- Medium: 8 KB, 5k iterations
    (10000, 1000),     -- Large: 80 KB, 1k iterations
    (100000, 100),     -- Very Large: 800 KB, 100 iterations
    (1000000, 10)      -- Huge: 8 MB, 10 iterations
  ]

  -- Benchmark Float64
  printHeader "Float64 (8 bytes/element)"
  let mut f64Results := []
  for pair in sizes do
    let size := pair.1
    let iters := pair.2
    let result ← benchmarkFloat64 size iters
    printResult result
    f64Results := result :: f64Results
  let f64Mean := calculateMeanSpeedup f64Results.reverse
  IO.println s!"  Mean Speedup: {f64Mean}x"

  -- Benchmark Float32
  printHeader "Float32 (4 bytes/element)"
  let mut f32Results := []
  for pair in sizes do
    let size := pair.1
    let iters := pair.2
    let result ← benchmarkFloat32 size iters
    printResult result
    f32Results := result :: f32Results
  let f32Mean := calculateMeanSpeedup f32Results.reverse
  IO.println s!"  Mean Speedup: {f32Mean}x"

  -- Benchmark Float16 (only if hardware supports it)
  let hasFP16 ← Hesper.Float16.hasHardwareSupport
  let mut f16MeanOpt : Option Float := none
  if hasFP16 then
    printHeader "Float16 (2 bytes/element)"
    let mut f16Results := []
    for pair in sizes do
      let size := pair.1
      let iters := pair.2
      let result ← benchmarkFloat16 size iters
      printResult result
      f16Results := result :: f16Results
    let f16Mean := calculateMeanSpeedup f16Results.reverse
    IO.println s!"  Mean Speedup: {f16Mean}x"
    f16MeanOpt := some f16Mean
  else
    IO.println "\n⚠ Float16 hardware not available - skipping FP16 benchmarks"

  -- Summary
  IO.println "\n╔═══════════════════════════════════════════════════════╗"
  IO.println "║   Summary                                            ║"
  IO.println "╚═══════════════════════════════════════════════════════╝"
  IO.println s!"Float64 Mean Speedup: {f64Mean}x"
  IO.println s!"Float32 Mean Speedup: {f32Mean}x"
  match f16MeanOpt with
  | some f16Mean => IO.println s!"Float16 Mean Speedup: {f16Mean}x"
  | none => pure ()

  IO.println "\n✓ Benchmark complete"
