import Tests.NumericalTests
import LSpec

/-!
# Numerical Tests Runner

Standalone runner for numerical accuracy tests.
-/

def main : IO UInt32 := do
  let tests ← Tests.NumericalTests.allTests
  LSpec.lspecIO (.ofList tests) ([] : List String)
