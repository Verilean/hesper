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

/-! ## Device buffer pool

nsys showed `cuMemAlloc_v2` firing **1383 times per 11-prompt + 10-decode
inference** vs llama.cpp's 4.  Each call is O(20–30 µs) in the driver
and the bumped total was ~46 ms of host time.  llama.cpp reuses its VRAM
via a VMM pool.  hesper's pool here is simpler: free'd buffers go onto
a free-list keyed by rounded-up size bucket; `createCUDABuffer` first
tries to pop from the bucket, falling back to `cuMalloc` only on miss.

Rounding: size → next multiple of 256 bytes (CUDA alignment), capped
bucket count via power-of-two rounding for sizes > 4 KB.  This keeps
small (per-call scratch) buffers high-hit-rate without wasting VRAM on
large weight buffers that rarely repeat. -/

/-- Maps rounded bucket-size → array of free device pointers.  Each
    entry is a stack; pop on alloc, push on free. -/
initialize bufferPool : IO.Ref (Array (USize × Array CUdeviceptr)) ← IO.mkRef #[]

/-- Toggle for disabling the pool (debugging).  Default: enabled. -/
initialize bufferPoolEnabled : IO.Ref Bool ← IO.mkRef true

/-- Threshold above which we DO NOT round to power-of-two buckets.
    Buffers larger than this are allocated at exact size (no pool reuse,
    no rounding waste) — they are typically model weights that live for
    the whole session, so the hit-rate of the bucket pool would be zero
    anyway and the rounding waste is significant (up to 2×).

    1 MB chosen because: gemma4 weight tensors are 2-150 MB; transient
    decode scratch is < 100 KB. -/
private def largeAllocThreshold : USize := 1024 * 1024

/-- Round a size up to a power-of-two bucket (min 256 bytes) for small
    buffers.  Returns `size` unchanged for buffers ≥ `largeAllocThreshold`
    so model weights don't waste up to 2× VRAM each.  The pool now only
    services small transient buffers where the round-up waste is bounded
    in absolute bytes. -/
private partial def roundBucketSize (size : USize) : USize :=
  if size >= largeAllocThreshold then size
  else if size <= 256 then 256
  else
    let n := size.toNat
    let rec go (x : Nat) : Nat := if x >= n then x else go (x * 2)
    (go 256).toUSize

/-- Reset the pool: actually free every pooled device pointer.  Call
    on shutdown or when the cache has gone stale. -/
def resetBufferPool : IO Unit := do
  let pool ← bufferPool.get
  for (_, ptrs) in pool do
    for p in ptrs do
      try cuFree p catch _ => pure ()
  bufferPool.set #[]

/-- Try to pop a free buffer of (at least) `size` bytes from the pool.
    Returns `some (ptr, capacity)` if one is available, else `none`. -/
private def tryPopFromPool (size : USize) : IO (Option (CUdeviceptr × USize)) := do
  if ¬ (← bufferPoolEnabled.get) then return none
  let bucket := roundBucketSize size
  let pool ← bufferPool.get
  let idx? := pool.findIdx? (fun p => p.1 == bucket)
  match idx? with
  | none => return none
  | some i =>
    let (sz, ptrs) := pool[i]!
    if ptrs.size == 0 then return none
    let ptr := ptrs.back!
    let newPtrs := ptrs.pop
    bufferPool.set (pool.set! i (sz, newPtrs))
    return some (ptr, sz)

/-- Push a buffer back onto the pool, keyed by its rounded bucket.
    Skips pooling for very large buffers (> 256 MB) to avoid VRAM
    bloat from one-off model tensors. -/
private def pushToPool (ptr : CUdeviceptr) (size : USize) : IO Bool := do
  if ¬ (← bufferPoolEnabled.get) then return false
  if size > 256 * 1024 * 1024 then return false
  let bucket := roundBucketSize size
  bufferPool.modify fun pool =>
    match pool.findIdx? (fun p => p.1 == bucket) with
    | some i =>
      let (sz, ptrs) := pool[i]!
      pool.set! i (sz, ptrs.push ptr)
    | none => pool.push (bucket, #[ptr])
  return true

def createCUDABuffer (size : USize) : IO CUDABuffer := do
  match ← tryPopFromPool size with
  | some (ptr, bucket) =>
    -- Reusing memory: zero it to keep createCUDABuffer's invariant
    -- (callers expect zeroed memory, matching the original cuMemset).
    cuMemset ptr bucket
    return { ptr, size := bucket }
  | none =>
    let bucket := roundBucketSize size
    let ptr ← cuMalloc bucket
    cuMemset ptr bucket
    return { ptr, size := bucket }

def writeCUDABuffer (buf : CUDABuffer) (data : ByteArray) : IO Unit :=
  cuMemcpyHtoD buf.ptr data 0 data.size.toUSize

def readCUDABuffer (buf : CUDABuffer) (size : USize) : IO ByteArray :=
  cuMemcpyDtoH buf.ptr size

def readCUDABufferFull (buf : CUDABuffer) : IO ByteArray :=
  cuMemcpyDtoH buf.ptr buf.size

def freeCUDABuffer (buf : CUDABuffer) : IO Unit := do
  let pooled ← pushToPool buf.ptr buf.size
  unless pooled do cuFree buf.ptr

end Hesper.CUDA
