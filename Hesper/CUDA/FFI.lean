/-!
# CUDA Driver API FFI Bindings

Minimal bindings to the CUDA Driver API for PTX JIT compilation
and kernel execution. No nvcc or CUDA runtime needed — only the
NVIDIA driver (libcuda.so).

Usage:
  cuDriverInit
  let dev ← cuDeviceGet 0
  let ctx ← cuCtxCreate dev
  let mod ← cuModuleLoadData ptxString
  let func ← cuModuleGetFunction mod "main"
  let buf ← cuMalloc 1024
  cuLaunchKernel func (gridX, gridY, gridZ) (blockX, blockY, blockZ) 0 #[buf]
-/

namespace Hesper.CUDA

-- Opaque handles (stored as USize / Nat)
abbrev CUdevice := UInt32
abbrev CUcontext := USize
abbrev CUmodule := USize
abbrev CUfunction := USize
abbrev CUdeviceptr := USize

/-! ## Driver initialization -/

@[extern "lean_hesper_cuda_init"]
opaque cuDriverInit : IO Unit

@[extern "lean_hesper_cuda_device_count"]
opaque cuDeviceCount : IO Nat

@[extern "lean_hesper_cuda_device_get"]
opaque cuDeviceGet (idx : UInt32) : IO CUdevice

@[extern "lean_hesper_cuda_device_name"]
opaque cuDeviceName (dev : CUdevice) : IO String

@[extern "lean_hesper_cuda_compute_capability"]
opaque cuComputeCapability (dev : CUdevice) : IO Nat

@[extern "lean_hesper_cuda_total_mem"]
opaque cuTotalMem (dev : CUdevice) : IO USize

/-! ## Context -/

@[extern "lean_hesper_cuda_ctx_create"]
opaque cuCtxCreate (dev : CUdevice) : IO CUcontext

@[extern "lean_hesper_cuda_ctx_destroy"]
opaque cuCtxDestroy (ctx : CUcontext) : IO Unit

/-! ## Module (PTX JIT) -/

@[extern "lean_hesper_cuda_module_load_data"]
opaque cuModuleLoadData (ptxSource : @& String) : IO CUmodule

@[extern "lean_hesper_cuda_module_get_function"]
opaque cuModuleGetFunction (mod : CUmodule) (funcName : @& String) : IO CUfunction

@[extern "lean_hesper_cuda_module_unload"]
opaque cuModuleUnload (mod : CUmodule) : IO Unit

/-! ## Utilities -/

@[extern "lean_hesper_fast_string_hash"]
opaque fastStringHash (s : @& String) : IO USize

/-! ## Memory -/

@[extern "lean_hesper_cuda_malloc"]
opaque cuMalloc (size : USize) : IO CUdeviceptr

@[extern "lean_hesper_cuda_free"]
opaque cuFree (ptr : CUdeviceptr) : IO Unit

@[extern "lean_hesper_cuda_memcpy_htod"]
opaque cuMemcpyHtoD (dst : CUdeviceptr) (src : @& ByteArray) (offset : USize) (size : USize) : IO Unit

@[extern "lean_hesper_cuda_memcpy_dtoh"]
opaque cuMemcpyDtoH (src : CUdeviceptr) (size : USize) : IO ByteArray

@[extern "lean_hesper_cuda_memset"]
opaque cuMemset (ptr : CUdeviceptr) (size : USize) : IO Unit

/-! ## Kernel launch -/

@[extern "lean_hesper_cuda_launch_kernel"]
opaque cuLaunchKernel
    (func : CUfunction)
    (gridX gridY gridZ : UInt32)
    (blockX blockY blockZ : UInt32)
    (sharedMem : UInt32)
    (args : @& Array USize)  -- device pointers
    : IO Unit

end Hesper.CUDA
