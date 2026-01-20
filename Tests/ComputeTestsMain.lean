import Tests.ComputeTests
import LSpec

/-!
# Compute Tests Runner

Standalone runner for compute pipeline tests.
-/

def main : IO UInt32 := do
  let tests ← Tests.ComputeTests.allTests
  LSpec.lspecIO (.ofList tests) ([] : List String)
