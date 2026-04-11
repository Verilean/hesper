import Hesper.CUDA.FFI

namespace Hesper.CUDA

structure CUDABuffer where
  ptr : CUdeviceptr
  size : USize

def initCUDA : IO (CUdevice × CUcontext) := do
  cuDriverInit
  let count ← cuDeviceCount
  if count == 0 then throw (IO.userError "No CUDA devices found")
  let dev ← cuDeviceGet 0
  let name ← cuDeviceName dev
  let cc ← cuComputeCapability dev
  let mem ← cuTotalMem dev
  IO.println s!"[CUDA] Device: {name}, SM {cc / 10}.{cc % 10}, {mem / (1024*1024)} MB"
  let ctx ← cuCtxCreate dev
  return (dev, ctx)

def createCUDABuffer (size : USize) : IO CUDABuffer := do
  let ptr ← cuMalloc size
  cuMemset ptr size
  return { ptr, size }

def writeCUDABuffer (buf : CUDABuffer) (data : ByteArray) : IO Unit :=
  cuMemcpyHtoD buf.ptr data 0 data.size.toUSize

def readCUDABuffer (buf : CUDABuffer) (size : USize) : IO ByteArray :=
  cuMemcpyDtoH buf.ptr size

def readCUDABufferFull (buf : CUDABuffer) : IO ByteArray :=
  cuMemcpyDtoH buf.ptr buf.size

def freeCUDABuffer (buf : CUDABuffer) : IO Unit :=
  cuFree buf.ptr

end Hesper.CUDA
