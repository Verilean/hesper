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

/-! ## Persistent mmap with GC-managed lifetime

`MMappedFile` is an opaque Lean external object backed by a C-side
`mmap` region.  Its finalizer (registered on the external class at
registration time) calls `munmap` when the last Lean reference is
dropped — matching llama.cpp's `unique_ptr<llama_mmap>` RAII pattern.

Unlike the legacy `mmapFile` above, this path never copies the file
into a `ByteArray`.  Tensor weights are kept as `(mmap, offset, size)`
triples and uploaded to the GPU via `cuMemcpyHtoDAsync` directly from
the mmap region — saving ~5 GB of Lean-heap allocation and enabling
async H2D (the mmap region stays alive until all in-flight copies
finish, because the stream sync point still holds a Lean reference). -/

/-- Opaque handle to a memory-mapped file.  GC-finalised with
    `munmap`. -/
opaque MMappedFileImpl : NonemptyType
def MMappedFile : Type := MMappedFileImpl.type
instance : Nonempty MMappedFile := MMappedFileImpl.property

@[extern "lean_hesper_mmap_open_persistent"]
opaque mmapOpen (path : @& String) : IO MMappedFile

@[extern "lean_hesper_mmap_size"]
opaque mmapSize (h : @& MMappedFile) : IO USize

/-- Copy a slice of the mmap into a fresh Lean ByteArray.  Use only
    for metadata / parsing; tensor weights should use
    `cuMemcpyHtoDFromMmap` instead. -/
@[extern "lean_hesper_mmap_slice_to_bytes_persistent"]
opaque mmapSliceToBytesPersistent (h : @& MMappedFile) (offset : USize)
    (size : USize) : IO ByteArray

/-- H2D copy from mmap region directly to GPU.  When `stream = 0`,
    issues the sync `cuMemcpyHtoD`; otherwise async on the given
    stream.  Caller must keep the `MMappedFile` referenced until the
    stream has synchronised (Lean GC automatically does this via any
    structure that holds both the GPU buffer handle and the mmap). -/
@[extern "lean_hesper_cuda_memcpy_htod_from_mmap"]
opaque cuMemcpyHtoDFromMmap (dst : USize) (h : @& MMappedFile)
    (offset : USize) (size : USize) (stream : USize) : IO Unit

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

/-! ## Metadata-free launch descriptor table (Option B+)

These register kernel launch metadata (func, grid/block, arg ptrs) in
C-owned storage, then fire by numeric id.  The Lean side holds only a
USize id per kernel and makes exactly one FFI call per launch — no
`Array USize` allocation, no `ByteArray` for params, no closure for
buffer resolution.  See `docs/llama-fusion-analysis/53-metadata-free-
forward-design.md`.
-/

/-- Reset the descriptor pool.  Call once when rebuilding an inference
    schedule (not per token). -/
@[extern "lean_hesper_desc_reset"]
opaque descReset : IO Unit

/-- Register a descriptor with immutable grid/block and a snapshot of
    buffer pointers.  Returns the descriptor id. -/
@[extern "lean_hesper_desc_register"]
opaque descRegister
    (func : CUfunction)
    (gridX gridY gridZ : UInt32)
    (blockX blockY blockZ : UInt32)
    (sharedMem : UInt32)
    (args : @& Array USize)
    : IO USize

/-- Rebind one buffer pointer in an existing descriptor (used only for
    sites whose buffers genuinely change between calls). -/
@[extern "lean_hesper_desc_rebind"]
opaque descRebind (descId : USize) (slot : USize) (newPtr : USize) : IO Unit

/-- Fast launch via descriptor id.  This is the Lean-heap-free hot
    path.  `stream=0` = default stream. -/
@[extern "lean_hesper_desc_launch"]
opaque descLaunch (descId : USize) (stream : USize) : IO Unit

/-- Fused rebind-all-args + launch.  Used when the descriptor's
    buffer pointers may differ between calls (e.g. same logical call
    site across 42 layers with per-layer weights).  Writes each arg
    into the C-owned `arg_storage` in place — still zero Lean heap
    past the FFI boundary.  `args.size` must equal the descriptor's
    `n_args`. -/
@[extern "lean_hesper_desc_launch_with_args"]
opaque descLaunchWithArgs
    (descId : USize) (stream : USize) (args : Array USize) : IO Unit

/-- Raw-bytes launch for external PTX (e.g. llama.cpp) whose kernels take
    mixed-type args (uint3 structs, f16 scalars, ...).  `argBytes` holds
    packed arg values; `argOffsets[i]` = byte offset of the i-th arg.
    CUDA receives `void**` where each entry points at `argBytes + offset`. -/
@[extern "lean_hesper_cuda_launch_kernel_raw"]
opaque cuLaunchKernelRaw
    (func : CUfunction)
    (gridX gridY gridZ : UInt32)
    (blockX blockY blockZ : UInt32)
    (sharedMem : UInt32)
    (argBytes : @& ByteArray)
    (argOffsets : @& Array USize)
    : IO Unit

/-- Stream-aware variant of `cuLaunchKernelRaw`.  Required when the caller
    is inside `cuStreamBeginCapture` — launches on the default stream are
    NOT captured into the graph, so their execution ordering diverges from
    the captured hesper kernels on graph replay, producing garbage output.
    `stream = 0` (default stream) matches the old behaviour. -/
@[extern "lean_hesper_cuda_launch_kernel_raw_on_stream"]
opaque cuLaunchKernelRawOnStream
    (func : CUfunction)
    (gridX gridY gridZ : UInt32)
    (blockX blockY blockZ : UInt32)
    (sharedMem : UInt32)
    (argBytes : @& ByteArray)
    (argOffsets : @& Array USize)
    (stream : CUstream)
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

/-! ## CUDA Graphs

Capture a sequence of kernel launches once, then replay the whole graph
per decode token with a single driver call.  llama.cpp uses this to
amortise the ~1.2 µs/dispatch host overhead that dominates our ~10
ms/tok host budget.  See docs/llama-fusion-analysis/12-complete-cuda-flow.md
§5 for the llama.cpp reference flow.

Opaque handles are `size_t` on the C side; we marshall as `USize`. -/

/-- `cudaStream_t` handle (null stream ≡ 0). -/
def CUstream : Type := USize
/-- `cudaGraph_t` handle (capture product). -/
def CUgraph : Type := USize
/-- `cudaGraphExec_t` handle (instantiated graph ready to launch). -/
def CUgraphExec : Type := USize

/-- Create a dedicated non-blocking stream on which subsequent launches
    can be captured.  Call once at context init. -/
@[extern "lean_hesper_cuda_stream_create"]
opaque cuStreamCreate : IO CUstream

/-- Create a blocking (default-synchronising) stream.  Ops on this
    stream implicitly sync with the null stream, so `readBuffer`
    (which goes through the null stream) observes prior work
    without an explicit `cuStreamSynchronize`. -/
@[extern "lean_hesper_cuda_stream_create_default"]
opaque cuStreamCreateDefault : IO CUstream

@[extern "lean_hesper_cuda_stream_destroy"]
opaque cuStreamDestroy (stream : CUstream) : IO Unit

/-- Begin graph capture on `stream`.  All `cuLaunchKernelOnStream`
    launches between this call and `cuStreamEndCapture` are recorded
    into the produced `cudaGraph_t`. -/
@[extern "lean_hesper_cuda_stream_begin_capture"]
opaque cuStreamBeginCapture (stream : CUstream) : IO Unit

/-- End capture and return the captured graph. -/
@[extern "lean_hesper_cuda_stream_end_capture"]
opaque cuStreamEndCapture (stream : CUstream) : IO CUgraph

/-- Turn a captured graph into an executable that the driver can replay
    with one launch.  Must be called before the first `cuGraphLaunch`. -/
@[extern "lean_hesper_cuda_graph_instantiate"]
opaque cuGraphInstantiate (graph : CUgraph) : IO CUgraphExec

@[extern "lean_hesper_cuda_graph_exec_destroy"]
opaque cuGraphExecDestroy (exec : CUgraphExec) : IO Unit

@[extern "lean_hesper_cuda_graph_destroy"]
opaque cuGraphDestroy (graph : CUgraph) : IO Unit

/-- Replay a previously-instantiated graph on `stream`.  Returns when
    the kernels have been submitted (they still run asynchronously). -/
@[extern "lean_hesper_cuda_graph_launch"]
opaque cuGraphLaunch (exec : CUgraphExec) (stream : CUstream) : IO Unit

/-- Launch a kernel onto a specific stream — needed so we can capture
    the launches into a graph (the default `cuLaunchKernel` goes to the
    null stream, which is not captureable). -/
@[extern "lean_hesper_cuda_launch_kernel_on_stream"]
opaque cuLaunchKernelOnStream
    (func : CUfunction)
    (gridX gridY gridZ : UInt32)
    (blockX blockY blockZ : UInt32)
    (sharedMem : UInt32)
    (stream : CUstream)
    (args : @& Array USize)
    : IO Unit

/-- Synchronise a stream.  Required after `cuGraphLaunch` when the
    caller wants to read back results. -/
@[extern "lean_hesper_cuda_stream_synchronize"]
opaque cuStreamSynchronize (stream : CUstream) : IO Unit

/-- Host→device memcpy on an explicit stream.  Needed so writes
    issued DURING stream capture get recorded as memcpy nodes in the
    resulting graph (rather than forcing a sync).  The driver captures
    the (src-host-ptr, dst-device-ptr, size) triple; on each replay
    the memcpy re-reads `src` from host memory, so subsequent tokens'
    values flow through without re-capturing. -/
@[extern "lean_hesper_cuda_memcpy_htod_async"]
opaque cuMemcpyHtoDAsync (dst : CUdeviceptr) (src : @& ByteArray)
    (offset : USize) (size : USize) (stream : CUstream) : IO Unit

/-! ## Pinned host memory (staging buffers for CUDA Graphs)

`ByteArray` is Lean-GC'd and its address is not stable across
`writeBufferOffset` calls.  CUDA Graph capture records the *pointer*,
so replay against a freed ByteArray tombstones with
`CUDA_ERROR_ILLEGAL_ADDRESS`.  The correct source for capturable
writes is **page-locked (pinned) host memory** allocated via
`cuMemHostAlloc` and held for the whole session.  llama.cpp uses the
same trick for its per-token scalar uploads. -/

/-- Allocate `size` bytes of pinned host memory.  Returns the host
    virtual-address as a `USize`.  The memory survives until
    `cuMemFreeHost` is called (or the process exits). -/
@[extern "lean_hesper_cuda_mem_alloc_host"]
opaque cuMemAllocHost (size : USize) : IO USize

@[extern "lean_hesper_cuda_mem_free_host"]
opaque cuMemFreeHost (hostPtr : USize) : IO Unit

/-- Write a small scalar (≤ 8 bytes) into a pinned host buffer.  The
    data is plain Lean bytes; the C++ side memcpys them into the
    pinned region.  No GPU involvement.  Use before every graph
    launch to update a captured memcpy node's source. -/
@[extern "lean_hesper_cuda_write_pinned"]
opaque cuWritePinned (hostPtr : USize) (offset : USize)
    (src : @& ByteArray) (size : USize) : IO Unit

/-- Host→device memcpy where the source is a pinned host pointer
    (stable across the session).  Safe to use inside stream capture;
    the graph records the host pointer, and every replay reads the
    current contents. -/
@[extern "lean_hesper_cuda_memcpy_htod_from_pinned"]
opaque cuMemcpyHtoDFromPinned
    (dst : CUdeviceptr) (hostPtr : USize) (offset : USize)
    (size : USize) (stream : CUstream) : IO Unit

/-- Fused: write `src` bytes to pinned slot `(hostPtr+offset)` then
    immediately queue an async H2D copy to `(dst)` on `stream`.  One
    FFI crossing instead of two (cuWritePinned + cuMemcpyHtoDFromPinned),
    eliminating ~half the Lean→C boundary overhead on hot per-scalar
    write sites. -/
@[extern "lean_hesper_cuda_pinned_write_and_copy"]
opaque cuPinnedWriteAndCopy
    (dst : CUdeviceptr) (hostPtr : USize) (offset : USize)
    (src : @& ByteArray) (size : USize) (stream : CUstream) : IO Unit

end Hesper.CUDA
