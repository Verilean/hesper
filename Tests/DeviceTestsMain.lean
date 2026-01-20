import Tests.DeviceTests
import LSpec

/-!
# Device Tests Runner

Standalone runner for device operation tests.
-/

def main : IO UInt32 := do
  let tests ← Tests.DeviceTests.allTests
  LSpec.lspecIO (.ofList tests) ([] : List String)
