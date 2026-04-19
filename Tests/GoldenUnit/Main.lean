import LSpec
import Tests.GoldenUnit.RMSNorm

/-!
# Gemma4 unit-test runner

Single LSpec exe `gemma4-unit-tests` that bundles every kernel
unit test.  Add new test modules here.
-/

unsafe def main : IO UInt32 := do
  let g1 ← Hesper.Tests.GoldenUnit.RMSNorm.allTests
  LSpec.lspecIO (.ofList g1) ([] : List String)
