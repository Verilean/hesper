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
open Hesper.WGSL (WorkgroupSize)

@[reducible] instance : GPUBackend Device where
  Buf := Buffer
  CachedDispatch := PreparedDispatch
  CompiledKernel := Hesper.WGSL.Execute.CompiledKernel
  executeWithConfig device computation namedBuffers (config : Hesper.ExecConfig) :=
    executeShaderNamed device computation namedBuffers
      { funcName := config.funcName, workgroupSize := config.workgroupSize,
        numWorkgroups := config.numWorkgroups,
        extensions := config.extensions, diagnostics := config.diagnostics }
  executeWithConfigCached device computation namedBuffers (config : Hesper.ExecConfig) cacheKey cacheRef :=
    executeShaderNamed device computation namedBuffers
      { funcName := config.funcName, workgroupSize := config.workgroupSize,
        numWorkgroups := config.numWorkgroups,
        extensions := config.extensions, diagnostics := config.diagnostics }
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
  buildKernel device computation funcName workgroupSize numWorkgroups :=
    Hesper.WGSL.Execute.buildKernel device computation
      { funcName, workgroupSize, numWorkgroups }
  dispatchCompiledKernel device kernel buffers numWorkgroups cacheRef := do
    let bg ← Hesper.WGSL.Execute.bindKernelDirect device kernel buffers
    match cacheRef with
    | some ref => ref.set (some (kernel.prepare bg))
    | none => pure ()
    Hesper.WGSL.Execute.dispatchKernel device kernel bg numWorkgroups
  hasSubgroupSupport device := Hesper.WGSL.Execute.hasSubgroupSupport device
  hasShaderF16Support device := Hesper.WGSL.Execute.hasShaderF16Support device
  newCacheRef := IO.mkRef none
  beginBatch device := Hesper.WGSL.Execute.beginBatch device
  endBatch device := Hesper.WGSL.Execute.endBatch device

end Hesper
