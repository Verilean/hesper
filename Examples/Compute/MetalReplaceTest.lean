import Hesper
import Hesper.WebGPU.Device

open Hesper.WebGPU

/-- metal_replacer STEP 1 PoC: prove the Dawn→Metal interop is live by reading the MTLDevice (behind the
    WGPUDevice, via Dawn's GetMTLDevice) from Lean. If this prints the real GPU name, the foundation for
    swapping in llama.cpp's tuned Metal kernels works. See METAL_REPLACER_INTEGRATION.md. -/
def main : IO Unit := do
  IO.println "=== metal_replacer STEP 1 PoC — Dawn → Metal interop ==="
  let inst ← Hesper.init
  let device ← getDevice inst
  let info ← mtlDeviceName device
  IO.println s!"✅ MTLDevice via dawn::native::metal::GetMTLDevice: {info}"
