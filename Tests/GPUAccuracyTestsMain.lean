import Tests.GPUAccuracyTests
import LSpec.Main

/-!
# GPU Accuracy Tests Runner

Runs CPU vs GPU numerical accuracy integration tests and reports results.
-/

def main : IO UInt32 := do
  let results ← Tests.GPUAccuracyTests.allTests
  LSpec.Main.runTests results
