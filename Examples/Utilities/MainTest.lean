import Hesper
import Hesper.WebGPU.Types
import Hesper.WebGPU.Device

/-!
# Minimal WebGPU Wrapper Test

Tests just the Device FFI to verify linking works.
-/

open Hesper.WebGPU

def main : IO Unit := do
  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║   Testing WebGPU Wrapper FFI                 ║"
  IO.println "╚══════════════════════════════════════════════╝"
  IO.println ""

  -- Initialize WebGPU first
  IO.println "Initializing WebGPU..."
  Hesper.init
  IO.println "Init complete!"
  IO.println ""

  -- Test basic device creation
  IO.println "Getting GPU device..."
  let device ← getDevice
  IO.println "Device obtained successfully!"
  IO.println s!"Device handle: {device.handle}"

  IO.println ""
  IO.println "╔══════════════════════════════════════════════╗"
  IO.println "║   FFI Test Complete!                         ║"
  IO.println "╚══════════════════════════════════════════════╝"
