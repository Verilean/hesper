import Hesper.Backend
import Hesper.WebGPU.Types
import Hesper.WebGPU.Buffer
import Hesper.WGSL.Execute

/-!
# WebGPU Backend Instance

Implements `GPUBackend` for the Dawn/WebGPU backend.
-/

namespace Hesper

open Hesper.WebGPU
open Hesper.WGSL.Execute (ExecutionConfig executeShaderNamed)

instance : GPUBackend Device where
  Buf := Buffer
  execute device computation namedBuffers config := do
    let wgpuConfig : ExecutionConfig := {
      funcName := config.funcName
      workgroupSize := config.workgroupSize
      numWorkgroups := config.numWorkgroups
    }
    executeShaderNamed device computation namedBuffers wgpuConfig
  allocBuffer device size := do
    let desc : BufferDescriptor := {
      size := size
      usage := [.storage, .copyDst, .copySrc]
      mappedAtCreation := false
    }
    createBuffer device desc
  freeBuffer _device _buf := pure ()  -- WebGPU GC handles this
  writeBuffer device buf data :=
    Hesper.WebGPU.writeBuffer device buf 0 data
  readBuffer device buf size :=
    Hesper.WebGPU.mapBufferRead device buf 0 size

end Hesper
