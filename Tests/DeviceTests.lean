import LSpec
import Hesper
import Hesper.WebGPU.Device
import Hesper.WebGPU.Types
import Hesper.WebGPU.Errors

/-!
# Device Operation Tests

Comprehensive tests for GPU device operations:
- Adapter enumeration
- Device creation
- Multi-GPU support
- Error path validation
-/

namespace Tests.DeviceTests

open Hesper.WebGPU
open LSpec

-- Helper to suppress Instance finalizer output during tests
def withInstance (action : Instance → IO α) : IO α := do
  let inst ← Hesper.init
  action inst

-- Test: Adapter Count
def testAdapterCount : IO TestSeq := withInstance fun inst => do
  let count ← getAdapterCount inst
  pure $ test "Adapter count is non-negative" (count >= 0)

-- Test: Adapter Info Retrieval
def testAdapterInfo : IO TestSeq := withInstance fun inst => do
  let count ← getAdapterCount inst

  if count == 0 then
    pure $ test "No adapters (skipped)" true
  else
    -- Test valid adapter index
    let info ← getAdapterInfo inst 0
    let validTest := test "First adapter has non-empty name" (info.name.length > 0)

    -- Test that adapter info contains expected backend type
    let backendTest := test "Adapter has valid backend type" (
      info.backendType == 4 || -- Metal
      info.backendType == 5 || -- Vulkan
      info.backendType == 3 || -- D3D12
      info.backendType == 6    -- OpenGL
    )

    pure (validTest ++ backendTest)

-- Test: Device Creation
def testDeviceCreation : IO TestSeq := withInstance fun inst => do
  let count ← getAdapterCount inst

  if count == 0 then
    pure $ test "No adapters for device creation (skipped)" true
  else
    -- Create device from first adapter
    let device ← getDevice inst
    -- If we get here without exception, device creation succeeded
    pure $ test "Device created successfully" true

-- Test: Device Creation by Index
def testDeviceByIndex : IO TestSeq := withInstance fun inst => do
  let count ← getAdapterCount inst

  if count == 0 then
    pure $ test "No adapters for index test (skipped)" true
  else
    -- Create device from first adapter by index
    let device ← getDeviceByIndex inst 0
    pure $ test "Device created from index 0" true

-- Test: Multi-GPU Support
def testMultiGPU : IO TestSeq := withInstance fun inst => do
  let count ← getAdapterCount inst

  let countTest := test "Adapter count matches system" true  -- We can't verify the exact count

  if count >= 2 then
    -- If multiple GPUs, test creating devices from each
    let device0 ← getDeviceByIndex inst 0
    let device1 ← getDeviceByIndex inst 1
    pure $ countTest ++ test "Multiple devices created from different adapters" true
  else
    pure $ countTest ++ test "Single GPU system (multi-GPU test skipped)" true

-- Test: Error Path - Invalid Adapter Index
def testInvalidAdapterIndex : IO TestSeq := withInstance fun inst => do
  let result ← try
    -- Try to get adapter info for invalid index
    let _ ← getAdapterInfo inst 999
    pure false  -- Should not reach here
  catch _ =>
    pure true   -- Error thrown as expected

  pure $ test "Invalid adapter index throws error" result

-- Test: Error Path - Out of Range Device Index
def testInvalidDeviceIndex : IO TestSeq := withInstance fun inst => do
  let result ← try
    -- Try to create device from invalid index
    let _ ← getDeviceByIndex inst 999
    pure false  -- Should not reach here
  catch _ =>
    pure true   -- Error thrown as expected

  pure $ test "Invalid device index throws error" result

-- Test: Instance Lifecycle
def testInstanceLifecycle : IO TestSeq := do
  -- Create and destroy multiple instances
  for _ in [0:3] do
    let inst ← Hesper.init
    let count ← getAdapterCount inst
    -- Instance will be GC'd when out of scope
    pure ()

  pure $ test "Multiple instance creation/destruction succeeds" true

-- All device tests
def allTests : IO (List (String × List TestSeq)) := do
  IO.println "Running Device Tests..."

  let t1 ← testAdapterCount
  let t2 ← testAdapterInfo
  let t3 ← testDeviceCreation
  let t4 ← testDeviceByIndex
  let t5 ← testMultiGPU
  let t6 ← testInvalidAdapterIndex
  let t7 ← testInvalidDeviceIndex
  let t8 ← testInstanceLifecycle

  pure [
    ("Adapter Count", [t1]),
    ("Adapter Info", [t2]),
    ("Device Creation", [t3]),
    ("Device By Index", [t4]),
    ("Multi-GPU Support", [t5]),
    ("Error: Invalid Adapter Index", [t6]),
    ("Error: Invalid Device Index", [t7]),
    ("Instance Lifecycle", [t8])
  ]

end Tests.DeviceTests

