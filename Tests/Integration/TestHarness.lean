import Hesper
import Hesper.Compute
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer

/-!
# Integration Test Harness

Common utilities for integration tests:
- GPU initialization and cleanup
- Buffer creation helpers
- Numerical comparison with tolerance
- Performance timing
- Test result formatting
-/

namespace Hesper.Tests.Integration

open Hesper.WebGPU
open Hesper.Compute

/-- Test result type -/
inductive TestResult where
  | pass : String → TestResult
  | fail : String → String → TestResult  -- test name, error message
  | skip : String → String → TestResult  -- test name, reason
  deriving Repr

/-- Test statistics -/
structure TestStats where
  passed : Nat := 0
  failed : Nat := 0
  skipped : Nat := 0
  totalTime : Float := 0.0
  deriving Repr

def TestStats.total (s : TestStats) : Nat :=
  s.passed + s.failed + s.skipped

def TestStats.addResult (stats : TestStats) (result : TestResult) (time : Float) : TestStats :=
  match result with
  | .pass _ => { stats with passed := stats.passed + 1, totalTime := stats.totalTime + time }
  | .fail _ _ => { stats with failed := stats.failed + 1, totalTime := stats.totalTime + time }
  | .skip _ _ => { stats with skipped := stats.skipped + 1 }

/-- Initialize GPU for testing -/
def initGPU : IO (Hesper.WebGPU.Instance × Device) := do
  let inst ← Hesper.init
  let device ← getDevice inst
  return (inst, device)

/-- Create buffer with data upload -/
def createBufferWithData (device : Device) (data : Array Float) : IO Buffer := do
  let size := (data.size * 4).toUSize
  let buffer ← createBuffer device {
    size := size
    usage := [.storage, .copyDst, .copySrc]
    mappedAtCreation := false
  }
  let bytes ← Hesper.Basic.floatArrayToBytes data
  writeBuffer device buffer 0 bytes
  return buffer

/-- Read buffer contents as float array -/
def readBufferAsFloats (device : Device) (buffer : Buffer) (count : Nat) : IO (Array Float) := do
  let size := (count * 4).toUSize
  let bytes ← mapBufferRead device buffer 0 size
  unmapBuffer buffer
  Hesper.Basic.bytesToFloatArray bytes

/-- Compare float arrays with tolerance -/
def compareFloatArrays (expected actual : Array Float) (tolerance : Float := 1e-5) : Bool :=
  if expected.size != actual.size then
    false
  else
    let diffs := expected.zip actual |>.map fun (e, a) => Float.abs (e - a)
    diffs.all (· <= tolerance)

/-- Assert float arrays are equal within tolerance -/
def assertFloatsEqual (name : String) (expected actual : Array Float) (tolerance : Float := 1e-5) : TestResult :=
  if compareFloatArrays expected actual tolerance then
    .pass name
  else
    let diff := expected.zip actual |>.map fun (e, a) => s!"{e} vs {a} (diff: {Float.abs (e - a)})"
    .fail name s!"Arrays differ:\n  Expected: {expected}\n  Actual: {actual}\n  Differences: {diff}"

/-- Run test with timing -/
def runTimedTest (name : String) (test : IO TestResult) : IO (TestResult × Float) := do
  let startTime ← IO.monoMsNow
  let result ← test
  let endTime ← IO.monoMsNow
  let elapsed := (endTime - startTime).toFloat / 1000.0  -- Convert to seconds
  return (result, elapsed)

/-- Print test result -/
def printResult (result : TestResult) (time : Float) : IO Unit := do
  match result with
  | .pass name =>
      IO.println s!"  ✅ {name} ({time}s)"
  | .fail name msg =>
      IO.println s!"  ❌ {name} ({time}s)"
      IO.println s!"     Error: {msg}"
  | .skip name reason =>
      IO.println s!"  ⏭️  {name} (skipped: {reason})"

/-- Print final test statistics -/
def printStats (stats : TestStats) : IO Unit := do
  IO.println ""
  IO.println "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  IO.println s!"Total Tests: {stats.total}"
  IO.println s!"  Passed:  {stats.passed} ✅"
  IO.println s!"  Failed:  {stats.failed} ❌"
  IO.println s!"  Skipped: {stats.skipped} ⏭️"
  IO.println s!"Total Time: {stats.totalTime}s"
  IO.println "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  if stats.failed > 0 then
    IO.println s!"❌ {stats.failed} test(s) FAILED"
  else if stats.passed == stats.total then
    IO.println "✅ All tests PASSED!"
  else
    IO.println s!"⚠️  Some tests were skipped"

/-- Exit code based on test results -/
def exitCode (stats : TestStats) : UInt8 :=
  if stats.failed > 0 then 1 else 0

end Hesper.Tests.Integration
