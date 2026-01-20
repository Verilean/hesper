import Tests.WGSLDSLTests
import LSpec

/-!
# WGSL DSL Tests Runner

Standalone runner for WGSL DSL tests.
-/

def main : IO UInt32 := do
  let tests ← Tests.WGSLDSLTests.allTests
  LSpec.lspecIO (.ofList tests) ([] : List String)
