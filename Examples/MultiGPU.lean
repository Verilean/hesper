import Hesper
import Hesper.WebGPU.Device

/-!
# Multi-GPU Support Demo

This example demonstrates how to enumerate and select GPUs in a multi-GPU system.
-/

namespace Examples.MultiGPU

open Hesper.WebGPU

def main : IO Unit := do
  IO.println "╔════════════════════════════════════════╗"
  IO.println "║   Hesper Multi-GPU Support Demo       ║"
  IO.println "╚════════════════════════════════════════╝"
  IO.println ""

  -- Initialize Hesper
  Hesper.init

  -- List all available GPUs
  IO.println "Listing available GPU adapters:"
  listAdapters

  IO.println ""

  -- Get adapter count
  let count ← getAdapterCount

  if count == 0 then
    IO.println "❌ No GPU adapters found!"
    return ()

  -- Test creating device from first GPU
  IO.println "Testing device creation from GPU 0:"
  let device0 ← getDeviceByIndex 0
  IO.println "✓ Device created from GPU 0"

  -- If multiple GPUs available, test second GPU
  if count > 1 then
    IO.println ""
    IO.println "Testing device creation from GPU 1:"
    let device1 ← getDeviceByIndex 1
    IO.println "✓ Device created from GPU 1"

  IO.println ""
  IO.println "✓ Multi-GPU demo complete!"

end Examples.MultiGPU

def main : IO Unit := Examples.MultiGPU.main
