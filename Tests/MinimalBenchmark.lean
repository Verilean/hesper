import Hesper
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WebGPU.Errors
import Hesper.WebGPU.Types

namespace Test

open Hesper.WebGPU

/-- High-precision timer using C++ chrono -/
@[extern "lean_hesper_get_time_ns"]
opaque getTimeNs : IO UInt64

/-- Benchmark result with timing and throughput metrics -/
structure BenchmarkResult where
  name : String
  durationNs : UInt64
  operations : UInt64
  bytesTransferred : UInt64
  deriving Repr

/-- Format benchmark result as human-readable string -/
def BenchmarkResult.format (r : BenchmarkResult) : String :=
  let durationMs := r.durationNs.toFloat / 1_000_000.0
  s!"{r.name}: {durationMs} ms"

/-- Run a benchmark with timing -/
def benchmark (name : String) (ops : UInt64) (bytes : UInt64) (action : IO Unit) : IO BenchmarkResult := do
  let startNs ← getTimeNs
  action
  let endNs ← getTimeNs
  let durationNs := endNs - startNs
  pure { name, durationNs, operations := ops, bytesTransferred := bytes }

def testDeviceInNamespace : IO Unit := do
  IO.println "=== Minimal Device Test ==="
  Hesper.init
  IO.println "After init"

  IO.println "Creating device..."
  let device ← getDevice
  IO.println "After getDevice"

  -- Try running a simple benchmark
  let r ← benchmark "Simple test" 1 0 do
    IO.println "Inside benchmark"
    pure ()
  IO.println (r.format)

  IO.println "Test complete!"

end Test

def main : IO Unit := do
  Test.testDeviceInNamespace
