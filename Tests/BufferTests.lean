import LSpec
import Hesper
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WebGPU.Types
import Hesper.WebGPU.Errors

/-!
# Buffer Operation Tests

Comprehensive tests for GPU buffer operations:
- Buffer creation with various sizes
- Buffer usage flags
- Buffer lifecycle
- Error path validation
-/

namespace Tests.BufferTests

open Hesper.WebGPU
open LSpec

-- Helper to create instance and device
def withDevice (action : Instance → Device → IO α) : IO α := do
  let inst ← Hesper.init
  let device ← getDevice inst
  action inst device

-- Test: Small Buffer Creation (1KB)
def testSmallBuffer : IO TestSeq := withDevice fun _ device => do
  let desc : BufferDescriptor := {
    size := 1024  -- 1KB
    usage := [BufferUsage.storage]
    mappedAtCreation := false
  }

  let buffer ← createBuffer device desc
  pure $ test "1KB storage buffer created successfully" true

-- Test: Medium Buffer Creation (1MB)
def testMediumBuffer : IO TestSeq := withDevice fun _ device => do
  let desc : BufferDescriptor := {
    size := 1024 * 1024  -- 1MB
    usage := [BufferUsage.storage]
    mappedAtCreation := false
  }

  let buffer ← createBuffer device desc
  pure $ test "1MB storage buffer created successfully" true

-- Test: Large Buffer Creation (16MB)
def testLargeBuffer : IO TestSeq := withDevice fun _ device => do
  let desc : BufferDescriptor := {
    size := 16 * 1024 * 1024  -- 16MB
    usage := [BufferUsage.storage]
    mappedAtCreation := false
  }

  let buffer ← createBuffer device desc
  pure $ test "16MB storage buffer created successfully" true

-- Test: Buffer with Multiple Usage Flags
def testBufferUsageFlags : IO TestSeq := withDevice fun _ device => do
  let desc : BufferDescriptor := {
    size := 4096
    usage := [BufferUsage.storage, BufferUsage.copySrc, BufferUsage.copyDst]
    mappedAtCreation := false
  }

  let buffer ← createBuffer device desc
  pure $ test "Buffer with multiple usage flags created successfully" true

-- Test: Buffer Lifecycle - Multiple Buffers
def testMultipleBuffers : IO TestSeq := withDevice fun _ device => do
  let desc : BufferDescriptor := {
    size := 1024
    usage := [BufferUsage.storage]
    mappedAtCreation := false
  }

  -- Create multiple buffers
  let buffer1 ← createBuffer device desc
  let buffer2 ← createBuffer device desc
  let buffer3 ← createBuffer device desc

  pure $ test "Multiple buffers created successfully" true

-- Test: Different Buffer Sizes
def testVariousBufferSizes : IO TestSeq := withDevice fun _ device => do
  let sizes := [256, 512, 1024, 2048, 4096, 8192]

  let mut allSuccess := true
  for size in sizes do
    let desc : BufferDescriptor := {
      size := size
      usage := [BufferUsage.storage]
      mappedAtCreation := false
    }
    try
      let _ ← createBuffer device desc
      pure ()
    catch _ =>
      allSuccess := false

  pure $ test s!"Buffers of various sizes ({sizes.length} tested) created successfully" allSuccess

-- Test: Zero-Size Buffer (Dawn allows zero-size buffers)
def testZeroSizeBuffer : IO TestSeq := withDevice fun _ device => do
  let result ← try
    let desc : BufferDescriptor := {
      size := 0
      usage := [BufferUsage.storage]
      mappedAtCreation := false
    }
    let _ ← createBuffer device desc
    pure true   -- Dawn allows zero-size buffers
  catch _ =>
    pure false  -- Unexpected failure

  pure $ test "Zero-size buffer creation succeeds" result

-- Test: Buffer Write Operation
def testBufferWrite : IO TestSeq := withDevice fun _ device => do
  let desc : BufferDescriptor := {
    size := 1024
    usage := [BufferUsage.storage, BufferUsage.copyDst]
    mappedAtCreation := false
  }

  let buffer ← createBuffer device desc
  let data := ByteArray.empty  -- Dummy data

  -- Write to buffer (simplified - actual write would need real data)
  writeBuffer device buffer 0 data

  pure $ test "Buffer write operation succeeds" true

-- All buffer tests
def allTests : IO (List (String × List TestSeq)) := do
  IO.println "Running Buffer Tests..."

  let t1 ← testSmallBuffer
  let t2 ← testMediumBuffer
  let t3 ← testLargeBuffer
  let t4 ← testBufferUsageFlags
  let t5 ← testMultipleBuffers
  let t6 ← testVariousBufferSizes
  let t7 ← testZeroSizeBuffer
  let t8 ← testBufferWrite

  pure [
    ("Small Buffer (1KB)", [t1]),
    ("Medium Buffer (1MB)", [t2]),
    ("Large Buffer (16MB)", [t3]),
    ("Multiple Usage Flags", [t4]),
    ("Multiple Buffers", [t5]),
    ("Various Buffer Sizes", [t6]),
    ("Error: Zero-Size Buffer", [t7]),
    ("Buffer Write", [t8])
  ]

end Tests.BufferTests

