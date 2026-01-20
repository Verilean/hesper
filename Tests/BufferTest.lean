import Hesper
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WebGPU.Types

def testSmallBuffer : IO Unit := do
  IO.println "Testing small buffer creation..."
  let inst ← Hesper.init

  let device ← Hesper.WebGPU.getDevice inst
  IO.println "Device created"

  let desc : Hesper.WebGPU.BufferDescriptor := {
    size := 1024  -- 1KB
    usage := [Hesper.WebGPU.BufferUsage.storage]
    mappedAtCreation := false
  }

  IO.println "Creating 1KB buffer..."
  IO.println s!"DEBUG: desc.size = {desc.size}"
  IO.println s!"DEBUG: desc.mappedAtCreation = {desc.mappedAtCreation}"
  try
    let buffer ← Hesper.WebGPU.createBuffer device desc
    IO.println "Buffer created successfully!"
  catch e =>
    IO.println s!"ERROR caught: {e}"

def main : IO Unit := do
  testSmallBuffer
