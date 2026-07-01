import Hesper
import Hesper.WebGPU.Device
import Hesper.WebGPU.Buffer
import Hesper.WebGPU.BufferOps
import Hesper.Basic

open Hesper.WebGPU

/-- metal_replacer STEP 1+2 PoC — validate the Dawn→Metal interop foundation:
    STEP 1: the MTLDevice behind the WGPUDevice (GetMTLDevice).
    STEP 2: the MTLBuffer behind a Dawn buffer (reinterpret to metal::Buffer + GetMTLBuffer) — the length
            must match our allocation, proving we can hand our data to llama.cpp's Metal kernels with no
            copies. See METAL_REPLACER_INTEGRATION.md. -/
def main : IO Unit := do
  IO.println "=== metal_replacer STEP 1+2 PoC — Dawn → Metal interop ==="
  let inst ← Hesper.init
  let device ← getDevice inst
  let info ← mtlDeviceName device
  IO.println s!"✅ STEP 1  MTLDevice: {info}"
  -- STEP 2: create a 16-float buffer, write known data, probe the underlying MTLBuffer.
  let buf ← createBuffer device { size := (16*4).toUSize, usage := [.storage, .copyDst, .copySrc], mappedAtCreation := false }
  let data ← Hesper.Basic.floatArrayToBytes #[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0]
  writeBuffer device buf 0 data
  let probe ← mtlBufferProbe buf
  IO.println s!"✅ STEP 2  buffer bridge (expect length=64): {probe}"
