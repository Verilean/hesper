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

/-- Cached once: HESPER_KEYED_PIPELINES=0 disables authoritative cache keys. -/
private def keyedPipelinesEnabled : IO Bool := do
  match ← IO.getEnv "HESPER_KEYED_PIPELINES" with
  | some "0" => pure false
  | _ => pure true

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
    -- The caller's cacheKey is AUTHORITATIVE (mirrors the CUDA backend, which
    -- skips codegen entirely on a key hit): the first call compiles the WGSL
    -- and registers the pipeline under the key; later calls skip WGSL
    -- regeneration (compileToWGSL per dispatch was ~100-200 µs — the dominant
    -- decode host cost at ~600 dispatches/token). Buffers are still re-bound
    -- per call via the (key, buffers) bind-group cache, so ping-pong buffer
    -- call sites stay correct. Contract (same as CUDA): a cacheKey must
    -- uniquely identify the generated WGSL — bake-varying params into the key.
    -- cacheKey=0 means "no key" (hash the generated WGSL as before).
    do
      -- HESPER_KEYED_PIPELINES=0: fall back to hashing the regenerated WGSL
      -- per call (slow but collision-proof) — A/B tool for key-collision hunts.
      let keyed ← keyedPipelinesEnabled
      executeShaderNamed device computation namedBuffers
        { funcName := config.funcName, workgroupSize := config.workgroupSize,
          numWorkgroups := config.numWorkgroups,
          extensions := config.extensions, diagnostics := config.diagnostics }
        (if cacheKey == 0 || !keyed then none else some cacheKey) (some cacheRef)
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
  buildKernel device computation (config : Hesper.ExecConfig) :=
    Hesper.WGSL.Execute.buildKernel device computation
      { funcName := config.funcName, workgroupSize := config.workgroupSize,
        numWorkgroups := config.numWorkgroups,
        extensions := config.extensions, diagnostics := config.diagnostics }
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
