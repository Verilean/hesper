import Hesper.Backend
import Hesper.WebGPU.Types
import Hesper.WebGPU.Buffer
import Hesper.WGSL.Execute

/-!
# WebGPU Backend Instance
-/

namespace Hesper

open Hesper.WebGPU
open Hesper.WGSL.Execute

instance : GPUBackend Device where
  Buf := Buffer
  CachedDispatch := PreparedDispatch
  executeKernel device computation namedBuffers funcName workgroupSize numWorkgroups :=
    executeShaderNamed device computation namedBuffers
      { funcName, workgroupSize, numWorkgroups }
  executeKernelCached device computation namedBuffers funcName workgroupSize numWorkgroups cacheKey cacheRef :=
    executeShaderNamed device computation namedBuffers
      { funcName, workgroupSize, numWorkgroups }
      (some cacheKey) (some cacheRef)
  replayCached device cached dims :=
    replayPreparedDispatch device cached dims.1 dims.2.1 dims.2.2
  allocBuffer device size :=
    createBuffer device { size, usage := [.storage, .copyDst, .copySrc], mappedAtCreation := false }
  allocBufferUsage device size _usage :=
    createBuffer device { size, usage := [.storage, .copyDst, .copySrc], mappedAtCreation := false }
  freeBuffer _device _buf := pure ()
  writeBuffer device buf data :=
    Hesper.WebGPU.writeBuffer device buf 0 data
  writeBufferOffset device buf offset data :=
    Hesper.WebGPU.writeBuffer device buf offset data
  readBuffer device buf size :=
    mapBufferRead device buf 0 size
  newCacheRef := IO.mkRef none

end Hesper
