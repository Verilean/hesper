import LSpec
import Hesper
import Hesper.WebGPU.Errors
import Hesper.WebGPU.Device

/-!
# Error Handling Tests

Tests for the comprehensive error handling system in Hesper.
Validates that errors are properly created, formatted, and propagated.
-/

namespace Tests.ErrorHandling

open Hesper.WebGPU
open LSpec

-- Helper function to check if a string contains a substring
def containsSubstr (s : String) (sub : String) : Bool :=
  (s.splitOn sub).length > 1

-- Test error message formatting
def testDeviceErrorMessages : TestSeq :=
  test "DeviceError.NoAdaptersFound has informative message" (
    let err := DeviceError.NoAdaptersFound
    let msg := err.toString
    containsSubstr msg "Recovery"
  ) ++

  test "DeviceError.InvalidAdapterIndex includes requested and available counts" (
    let err := DeviceError.InvalidAdapterIndex 5 2
    let msg := err.toString
    containsSubstr msg "5" && containsSubstr msg "2" && containsSubstr msg "Recovery"
  ) ++

  test "DeviceError.DeviceCreationFailed includes adapter index and reason" (
    let err := DeviceError.DeviceCreationFailed 0 "out of memory"
    let msg := err.toString
    containsSubstr msg "0" && containsSubstr msg "out of memory"
  )

-- Test buffer error messages
def testBufferErrorMessages : TestSeq :=
  test "BufferError.AllocationFailed includes size and reason" (
    let err := BufferError.AllocationFailed 1024 "insufficient GPU memory"
    let msg := err.toString
    containsSubstr msg "1024" && containsSubstr msg "insufficient GPU memory" && containsSubstr msg "Recovery"
  ) ++

  test "BufferError.OutOfBounds includes requested offset and buffer size" (
    let err := BufferError.OutOfBounds 2048 1024
    let msg := err.toString
    containsSubstr msg "2048" && containsSubstr msg "1024"
  ) ++

  test "BufferError.MappingFailed includes reason" (
    let err := BufferError.MappingFailed "buffer not created with MAP_READ"
    let msg := err.toString
    containsSubstr msg "MAP_READ"
  )

-- Test shader error messages
def testShaderErrorMessages : TestSeq :=
  test "ShaderError.CompilationFailed includes source preview and errors" (
    let source := "invalid wgsl code here"
    let errors := "syntax error at line 1"
    let err := ShaderError.CompilationFailed source errors
    let msg := err.toString
    containsSubstr msg "syntax error" && containsSubstr msg "invalid wgsl"
  ) ++

  test "ShaderError.ValidationFailed includes reason" (
    let err := ShaderError.ValidationFailed "missing entry point"
    let msg := err.toString
    containsSubstr msg "missing entry point"
  )

-- Test WebGPUError wrapper
def testWebGPUErrorWrapper : TestSeq :=
  test "WebGPUError.Device wraps device errors correctly" (
    let devErr := DeviceError.NoAdaptersFound
    let err := WebGPUError.Device devErr
    let msg := err.toString
    containsSubstr msg "[Device Error]" && containsSubstr msg "No GPU adapters found"
  ) ++

  test "WebGPUError.Buffer wraps buffer errors correctly" (
    let bufErr := BufferError.AllocationFailed 512 "test reason"
    let err := WebGPUError.Buffer bufErr
    let msg := err.toString
    containsSubstr msg "[Buffer Error]" && containsSubstr msg "512"
  ) ++

  test "WebGPUError.Shader wraps shader errors correctly" (
    let shdErr := ShaderError.ValidationFailed "test validation"
    let err := WebGPUError.Shader shdErr
    let msg := err.toString
    containsSubstr msg "[Shader Error]" && containsSubstr msg "test validation"
  )

-- Test backend type conversion
def testBackendTypes : TestSeq :=
  test "BackendType.toString converts Metal correctly" (
    BackendType.Metal.toString == "Metal"
  ) ++

  test "BackendType.toString converts Vulkan correctly" (
    BackendType.Vulkan.toString == "Vulkan"
  ) ++

  test "BackendType.toString converts D3D12 correctly" (
    BackendType.D3D12.toString == "D3D12"
  )

-- Test helper functions (simplified - just test that they compile and run)
def testErrorHelpers : TestSeq :=
  test "DeviceError.noAdapters creates an error" (
    let err := DeviceError.noAdapters
    let msg := err.toString
    containsSubstr msg "No GPU adapters found"
  ) ++

  test "BufferError.allocationFailed creates an error with size" (
    let err := BufferError.allocationFailed 2048 "test"
    let msg := err.toString
    containsSubstr msg "2048"
  )

-- Integration tests with actual device operations (IO-based)
def testDeviceOperations : IO TestSeq := do
  -- Initialize Hesper
  let inst ← Hesper.init

  -- Test adapter count (should succeed)
  let count ← getAdapterCount inst
  let testCount := test "getAdapterCount returns valid count" (count >= 0)

  -- Test getting adapter info for valid index (should succeed if adapters exist)
  let testValidAdapter ← if count > 0 then do
    let info ← getAdapterInfo inst 0
    pure $ test "getAdapterInfo with valid index succeeds" (info.name.length > 0)
  else
    pure $ test "skipped (no adapters)" true

  -- Test getting adapter info for invalid index (should fail with proper error)
  let testInvalidAdapter ← try
    let _ ← getAdapterInfo inst 999
    pure $ test "getAdapterInfo with invalid index should fail" false
  catch _ =>
    -- Just verify that an error was thrown, don't inspect the message
    pure $ test "getAdapterInfo with invalid index throws error" true

  pure (testCount ++ testValidAdapter ++ testInvalidAdapter)

-- All tests combined
def allTests : IO (List (String × List TestSeq)) := do
  IO.println "Running Error Handling Tests..."

  let pureTests :=
    ("Device Error Messages", [testDeviceErrorMessages]) ::
    ("Buffer Error Messages", [testBufferErrorMessages]) ::
    ("Shader Error Messages", [testShaderErrorMessages]) ::
    ("WebGPUError Wrapper", [testWebGPUErrorWrapper]) ::
    ("Backend Types", [testBackendTypes]) ::
    ("Error Helper Functions", [testErrorHelpers]) ::
    []

  let ioTests ← testDeviceOperations
  let allTestsList := pureTests ++ [("Device Operations (IO)", [ioTests])]

  pure allTestsList

end Tests.ErrorHandling

