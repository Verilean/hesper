import Hesper.Backend
import Hesper.CUDA.FFI
import Hesper.CUDA.Buffer
import Hesper.CUDA.CodeGen

/-!
# CUDA Backend Instance
-/

namespace Hesper

open Hesper.CUDA
open Hesper.CUDA.CodeGen
open Hesper.WGSL.Monad (ShaderM)

structure CUDAContext where
  ctx : CUcontext
  deriving Inhabited

def CUDAContext.init : IO CUDAContext := do
  cuDriverInit
  let count ← cuDeviceCount
  if count == 0 then throw (IO.userError "No CUDA devices found")
  let dev ← cuDeviceGet 0
  let name ← cuDeviceName dev
  let cc ← cuComputeCapability dev
  let mem ← cuTotalMem dev
  IO.println s!"[CUDA] Device: {name}, SM {cc / 10}.{cc % 10}, {mem / (1024*1024)} MB"
  let ctx ← cuCtxCreate dev
  return ⟨ctx⟩

/-- CUDA cached dispatch — compiled function + buffer args for replay. -/
structure CUDACachedDispatch where
  func : CUfunction
  sourceHash : UInt64
  args : Array USize  -- device pointers for replay

initialize cudaModuleCache : IO.Ref (Array (UInt64 × CUfunction)) ← IO.mkRef #[]

private def cudaExecuteImpl (computation : ShaderM Unit) (namedBuffers : List (String × CUDABuffer))
    (funcName : String) (workgroupSize : Hesper.WGSL.WorkgroupSize)
    (numWorkgroups : Nat × Nat × Nat) : IO CUfunction := do
  let ptx := generatePTX funcName workgroupSize computation
  let sourceHash := hash ptx
  let cache ← cudaModuleCache.get
  match cache.find? (fun e => e.1 == sourceHash) with
  | some (_, f) => return f
  | none =>
    let cudaMod ← cuModuleLoadData ptx
    let f ← cuModuleGetFunction cudaMod funcName
    cudaModuleCache.modify (·.push (sourceHash, f))
    return f

private def cudaLaunchWithBuffers (func : CUfunction) (namedBuffers : List (String × CUDABuffer))
    (computation : ShaderM Unit) (workgroupSize : Hesper.WGSL.WorkgroupSize)
    (numWorkgroups : Nat × Nat × Nat) : IO Unit := do
  let state := Hesper.WGSL.Monad.ShaderM.exec computation
  let declaredNames := state.declaredBuffers.map (·.1)
  let args ← declaredNames.foldlM (init := #[]) fun acc name => do
    match namedBuffers.find? (fun p => p.1 == name) with
    | some (_, buf) => return acc.push buf.ptr
    | none => throw (IO.userError s!"CUDA execute: missing buffer '{name}'")
  let (gx, gy, gz) := numWorkgroups
  cuLaunchKernel func
    gx.toUInt32 gy.toUInt32 gz.toUInt32
    workgroupSize.x.toUInt32 workgroupSize.y.toUInt32 workgroupSize.z.toUInt32
    0 args

/-- CUDA compiled kernel — just the JIT'd function. -/
structure CUDACompiledKernel where
  func : CUfunction
  declaredNames : Array String

instance : GPUBackend CUDAContext where
  Buf := CUDABuffer
  CachedDispatch := CUDACachedDispatch
  CompiledKernel := CUDACompiledKernel
  executeWithConfig _ctx computation namedBuffers config := do
    -- CUDA ignores extensions/diagnostics
    let func ← cudaExecuteImpl computation namedBuffers config.funcName config.workgroupSize config.numWorkgroups
    cudaLaunchWithBuffers func namedBuffers computation config.workgroupSize config.numWorkgroups
  executeWithConfigCached _ctx computation namedBuffers config _cacheKey cacheRef := do
    let cached ← cacheRef.get
    let func ← match cached with
    | some c => pure c.func
    | none => do
      let f ← cudaExecuteImpl computation namedBuffers config.funcName config.workgroupSize config.numWorkgroups
      -- Collect buffer args for replay
      let state := Hesper.WGSL.Monad.ShaderM.exec computation
      let declaredNames := state.declaredBuffers.map (·.1)
      let args ← declaredNames.foldlM (init := #[]) fun acc name => do
        match namedBuffers.find? (fun p => p.1 == name) with
        | some (_, buf) => return acc.push buf.ptr
        | none => return acc
      cacheRef.set (some { func := f, sourceHash := hash (generatePTX config.funcName config.workgroupSize computation), args })
      pure f
    cudaLaunchWithBuffers func namedBuffers computation config.workgroupSize config.numWorkgroups
  replayCached _ctx cached dims := do
    let (gx, gy, gz) := dims
    cuLaunchKernel cached.func
      gx.toUInt32 gy.toUInt32 gz.toUInt32 1 1 1 0 cached.args
  allocBuffer _ctx size := createCUDABuffer size
  freeBuffer _ctx buf := freeCUDABuffer buf
  writeBuffer _ctx buf data := writeCUDABuffer buf data
  readBuffer _ctx buf size := readCUDABuffer buf size
  buildKernel _ctx computation funcName workgroupSize _numWorkgroups := do
    let ptx := generatePTX funcName workgroupSize computation
    let sourceHash := hash ptx
    let cache ← cudaModuleCache.get
    let func ← match cache.find? (fun e => e.1 == sourceHash) with
    | some (_, f) => pure f
    | none => do
      let cudaMod ← cuModuleLoadData ptx
      let f ← cuModuleGetFunction cudaMod funcName
      cudaModuleCache.modify (·.push (sourceHash, f))
      pure f
    let state := Hesper.WGSL.Monad.ShaderM.exec computation
    pure { func, declaredNames := state.declaredBuffers.map (·.1) |>.toArray }
  dispatchCompiledKernel _ctx kernel buffers numWorkgroups _cacheRef := do
    let args := buffers.map (·.ptr)
    let (gx, gy, gz) := numWorkgroups
    cuLaunchKernel kernel.func
      gx.toUInt32 gy.toUInt32 gz.toUInt32 1 1 1 0 args
  hasSubgroupSupport _ctx := pure true   -- CUDA always has warp shuffle
  hasShaderF16Support _ctx := pure true  -- sm_89 has native f16
  newCacheRef := IO.mkRef none

end Hesper
