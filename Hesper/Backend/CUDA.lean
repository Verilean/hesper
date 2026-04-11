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

/-- CUDA backend context. -/
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

initialize cudaModuleCache : IO.Ref (Array (UInt64 × CUfunction)) ← IO.mkRef #[]

instance : GPUBackend CUDAContext where
  Buf := CUDABuffer
  executeKernel _ctx computation namedBuffers funcName workgroupSize numWorkgroups := do
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
  allocBuffer _ctx size := createCUDABuffer size
  freeBuffer _ctx buf := freeCUDABuffer buf
  writeBuffer _ctx buf data := writeCUDABuffer buf data
  readBuffer _ctx buf size := readCUDABuffer buf size

end Hesper
