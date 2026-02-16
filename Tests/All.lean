import LSpec
import Tests.ErrorHandling
import Tests.DeviceTests
import Tests.BufferTests
import Tests.ComputeTests
import Tests.WGSLDSLTests
import Tests.NumericalTests
import Tests.ShaderMonadTests

/-!
# Hesper Test Suite Runner

Runs all test suites and reports results.

## Test Categories:
- **Error Handling**: Error message formatting and propagation
- **Device Operations**: Adapter enumeration, device creation, multi-GPU
- **Buffer Operations**: Buffer creation, usage flags, lifecycle
- **Compute Pipeline**: Shader compilation, pipeline creation, execution

## Usage:
```
lake exe test-all
```
-/

open LSpec

def main : IO UInt32 := do
  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║   Hesper Comprehensive Test Suite           ║"
  IO.println "╚══════════════════════════════════════════════╝"
  IO.println ""

  -- Collect all tests
  let errorTests ← Tests.ErrorHandling.allTests
  let deviceTests ← Tests.DeviceTests.allTests
  let bufferTests ← Tests.BufferTests.allTests
  let computeTests ← Tests.ComputeTests.allTests
  let wgslDslTests ← Tests.WGSLDSLTests.allTests
  let numericalTests ← Tests.NumericalTests.allTests
  let shaderMonadTests ← Tests.ShaderMonadTests.allTests

  -- Combine all tests
  let allTests := errorTests ++ deviceTests ++ bufferTests ++ computeTests ++ wgslDslTests ++ numericalTests ++ shaderMonadTests

  IO.println s!"Running {allTests.length} test suites...\n"

  -- Run with LSpec
  let exitCode ← LSpec.lspecIO (.ofList allTests) ([] : List String)

  IO.println "\n╔══════════════════════════════════════════════╗"
  if exitCode == 0 then
    IO.println "║   ✅ All Tests Passed!                       ║"
  else
    IO.println "║   ❌ Some Tests Failed                       ║"
  IO.println "╚══════════════════════════════════════════════╝"

  pure exitCode
