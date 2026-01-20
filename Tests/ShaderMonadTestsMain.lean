import Tests.ShaderMonadTests
import LSpec

/-!
# ShaderM Monad Tests Runner

Standalone runner for ShaderM monad tests.
-/

def main : IO UInt32 := do
  let tests ← Tests.ShaderMonadTests.allTests
  LSpec.lspecIO (.ofList tests) ([] : List String)
