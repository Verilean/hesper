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

/-! ## Memory-mapped file I/O -/

/-- mmap a file. Returns (pointer, size). -/
@[extern "lean_hesper_mmap_file"]
opaque mmapFile (path : @& String) : IO (USize × USize)

@[extern "lean_hesper_munmap"]
opaque munmap (ptr : USize) (size : USize) : IO Unit

/-- Copy a slice from mmapped memory to a Lean ByteArray (for metadata). -/
@[extern "lean_hesper_mmap_slice_to_bytes"]
opaque mmapSliceToBytes (ptr : USize) (offset : USize) (size : USize) : IO ByteArray

/-- Copy mmapped data directly to GPU buffer (zero Lean-side copy). -/
@[extern "lean_hesper_mmap_to_gpu"]
opaque mmapToGPU (mmapPtr : USize) (offset : USize) (gpuPtr : USize) (size : USize) : IO Unit

/-- Fast file read: mmap + memcpy. Faster than IO.FS.readBinFile for large files. -/
@[extern "lean_hesper_read_file_fast"]
opaque readFileFast (path : @& String) : IO ByteArray

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

/-! ## L2 cache persistence (Ampere+ / Ada / Hopper, CC ≥ 8.0) -/

/-- Set the persisting-L2 limit on the current context.  The value is
    clamped to the device's `MAX_PERSISTING_L2_CACHE_SIZE` attribute.
    Returns the effective limit the driver applied. -/
@[extern "lean_hesper_cuda_set_l2_persist_limit"]
opaque cuSetL2PersistLimit (size : USize) : IO USize

/-- Install a persisting access-policy window on the default (null) stream.
    Subsequent launches on this stream will prefer keeping
    `[ptr, ptr+size)` in L2 across kernel boundaries. -/
@[extern "lean_hesper_cuda_set_l2_access_window"]
opaque cuSetL2AccessWindow (ptr : USize) (size : USize) : IO Unit

/-- Evict all persisting lines from L2 and return the cache to its default
    non-persistent behaviour. -/
@[extern "lean_hesper_cuda_reset_l2_persisting_cache"]
opaque cuResetL2PersistingCache : IO Unit

end Hesper.CUDA
