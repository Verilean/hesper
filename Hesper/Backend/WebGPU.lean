import Hesper.Backend
import Hesper.WebGPU.Types
import Hesper.WebGPU.Buffer
import Hesper.WGSL.Execute

/-!
# WebGPU Backend Instance
-/

namespace Hesper

open Hesper.WebGPU
open Hesper.WGSL.Execute (ExecutionConfig executeShaderNamed)

instance : GPUBackend Device where
  Buf := Buffer
  executeKernel device computation namedBuffers funcName workgroupSize numWorkgroups := do
    let config : ExecutionConfig := {
      funcName, workgroupSize, numWorkgroups
    }
    executeShaderNamed device computation namedBuffers config
  allocBuffer device size := do
    createBuffer device {
      size := size
      usage := [.storage, .copyDst, .copySrc]
      mappedAtCreation := false
    }
  freeBuffer _device _buf := pure ()
  writeBuffer device buf data :=
    Hesper.WebGPU.writeBuffer device buf 0 data
  readBuffer device buf size :=
    Hesper.WebGPU.mapBufferRead device buf 0 size

end Hesper
