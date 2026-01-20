import Tests.ErrorHandling
import LSpec

/-!
# Error Handling Tests Runner

Standalone runner for error handling tests.
-/

def main : IO UInt32 := do
  let tests ← Tests.ErrorHandling.allTests
  LSpec.lspecIO (.ofList tests) ([] : List String)
