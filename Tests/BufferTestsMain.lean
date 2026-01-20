import Tests.BufferTests
import LSpec

/-!
# Buffer Tests Runner

Standalone runner for buffer operation tests.
-/

def main : IO UInt32 := do
  let tests ← Tests.BufferTests.allTests
  LSpec.lspecIO (.ofList tests) ([] : List String)
