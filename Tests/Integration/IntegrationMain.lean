import Hesper
import Tests.Integration.TestHarness
import Tests.Integration.ComputePipeline
import Tests.Integration.BufferOperations

/-!
# Integration Test Suite Runner

Runs all integration tests and reports results.

Test Categories:
- Compute Pipeline Tests (5 tests)
- Buffer Operations Tests (8 tests)

Total: 13 integration tests
-/

namespace Hesper.Tests.Integration

open Hesper.WebGPU
open Hesper.Tests.Integration

def main : IO Unit := do
  IO.println "╔══════════════════════════════════════════════════════════╗"
  IO.println "║          Hesper Integration Test Suite                  ║"
  IO.println "╚══════════════════════════════════════════════════════════╝"
  IO.println ""

  -- Initialize GPU
  IO.println "🚀 Initializing WebGPU..."
  let (_, device) ← initGPU
  IO.println "✅ GPU initialized"
  IO.println ""

  let mut stats : TestStats := {}

  -- Run Compute Pipeline Tests
  IO.println "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  IO.println "Category: Compute Pipeline Tests"
  IO.println "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  let computeResults ← ComputePipeline.runAll device
  for (result, time) in computeResults do
    printResult result time
    stats := stats.addResult result time

  IO.println ""

  -- Run Buffer Operations Tests
  IO.println "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  IO.println "Category: Buffer Operations Tests"
  IO.println "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  let bufferResults ← BufferOperations.runAll device
  for (result, time) in bufferResults do
    printResult result time
    stats := stats.addResult result time

  IO.println ""

  -- Print final statistics
  printStats stats

  -- Exit with appropriate code
  IO.Process.exit (exitCode stats)

end Hesper.Tests.Integration

def main : IO Unit := Hesper.Tests.Integration.main
