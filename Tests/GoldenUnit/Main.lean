import LSpec
import Tests.GoldenUnit.RMSNorm
import Tests.GoldenUnit.Linear

/-!
# Gemma4 unit-test runner

Single LSpec exe `gemma4-unit-tests` that bundles every kernel
unit test.  Add new test modules here.
-/

unsafe def main : IO UInt32 := do
  let g1 ← Hesper.Tests.GoldenUnit.RMSNorm.allTests
  let g2 ← Hesper.Tests.GoldenUnit.Linear.allTests
  LSpec.lspecIO (.ofList (g1 ++ g2)) ([] : List String)
