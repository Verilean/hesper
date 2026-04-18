// CUDA Driver API bridge for Hesper PTX JIT backend.
// Direct linking against -lcuda. Uses cuda.h for correct types.
// NOTE: Lean 4 IO FFI does NOT pass a world token argument.

#include <lean/lean.h>
#include <cuda.h>
#include <cstring>
#include <cstdio>
#include <cstdlib>

static lean_obj_res cuda_error(CUresult err, const char* func) {
    const char* errName = nullptr;
    const char* errStr = nullptr;
    cuGetErrorName(err, &errName);
    cuGetErrorString(err, &errStr);
    char buf[512];
    snprintf(buf, sizeof(buf), "%s failed: %s - %s", func,
             errName ? errName : "unknown", errStr ? errStr : "unknown");
    return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(buf)));
}

#define CUDA_CHECK(call, func) \
    do { CUresult err = (call); if (err != CUDA_SUCCESS) return cuda_error(err, func); } while(0)

// ============================================================================
// Device management
// ============================================================================

extern "C" lean_obj_res lean_hesper_cuda_init() {
    CUDA_CHECK(cuInit(0), "cuInit");
    return lean_io_result_mk_ok(lean_box(0));
}

extern "C" lean_obj_res lean_hesper_cuda_device_count() {
    int count = 0;
    CUDA_CHECK(cuDeviceGetCount(&count), "cuDeviceGetCount");
    return lean_io_result_mk_ok(lean_box(count));
}

extern "C" lean_obj_res lean_hesper_cuda_device_get(uint32_t idx) {
    CUdevice dev;
    CUDA_CHECK(cuDeviceGet(&dev, idx), "cuDeviceGet");
    return lean_io_result_mk_ok(lean_box(dev));
}

extern "C" lean_obj_res lean_hesper_cuda_device_name(uint32_t dev) {
    char name[256];
    CUDA_CHECK(cuDeviceGetName(name, sizeof(name), dev), "cuDeviceGetName");
    return lean_io_result_mk_ok(lean_mk_string(name));
}

extern "C" lean_obj_res lean_hesper_cuda_compute_capability(uint32_t dev) {
    int major, minor;
    CUDA_CHECK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev),
               "cuDeviceGetAttribute(major)");
    CUDA_CHECK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev),
               "cuDeviceGetAttribute(minor)");
    return lean_io_result_mk_ok(lean_box(major * 10 + minor));
}

extern "C" lean_obj_res lean_hesper_cuda_total_mem(uint32_t dev) {
    size_t totalMem;
    CUDA_CHECK(cuDeviceTotalMem(&totalMem, dev), "cuDeviceTotalMem");
    return lean_io_result_mk_ok(lean_box_usize(totalMem));
}

// ============================================================================
// Context
// ============================================================================

extern "C" lean_obj_res lean_hesper_cuda_ctx_create(uint32_t dev) {
    CUcontext ctx;
    CUDA_CHECK(cuCtxCreate(&ctx, 0, dev), "cuCtxCreate");
    return lean_io_result_mk_ok(lean_box_usize((size_t)ctx));
}

extern "C" lean_obj_res lean_hesper_cuda_ctx_destroy(size_t ctx_val) {
    CUDA_CHECK(cuCtxDestroy((CUcontext)ctx_val), "cuCtxDestroy");
    return lean_io_result_mk_ok(lean_box(0));
}

// ============================================================================
// Module (PTX JIT)
// ============================================================================

extern "C" lean_obj_res lean_hesper_cuda_module_load_data(b_lean_obj_arg ptx_str) {
    const char* ptx = lean_string_cstr(ptx_str);
    CUmodule mod;
    // Use cuModuleLoadDataEx with max optimization level (4)
    CUjit_option opts[] = { CU_JIT_OPTIMIZATION_LEVEL };
    void* optVals[] = { (void*)(uintptr_t)4 };
    CUresult err = cuModuleLoadDataEx(&mod, ptx, 1, opts, optVals);
    if (err != CUDA_SUCCESS) {
        const char* errName = nullptr;
        cuGetErrorName(err, &errName);
        char buf[1024];
        size_t showLen = strlen(ptx); if (showLen > 500) showLen = 500;
        snprintf(buf, sizeof(buf), "cuModuleLoadData failed: %s\nPTX (%zu chars):\n%.*s",
                 errName ? errName : "unknown", showLen, (int)showLen, ptx);
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(buf)));
    }
    return lean_io_result_mk_ok(lean_box_usize((size_t)mod));
}

extern "C" lean_obj_res lean_hesper_cuda_module_get_function(size_t mod_val, b_lean_obj_arg func_name) {
    CUfunction func;
    CUDA_CHECK(cuModuleGetFunction(&func, (CUmodule)mod_val, lean_string_cstr(func_name)),
               "cuModuleGetFunction");
    return lean_io_result_mk_ok(lean_box_usize((size_t)func));
}

extern "C" lean_obj_res lean_hesper_cuda_module_unload(size_t mod_val) {
    CUDA_CHECK(cuModuleUnload((CUmodule)mod_val), "cuModuleUnload");
    return lean_io_result_mk_ok(lean_box(0));
}

// ============================================================================
// Memory
// ============================================================================

extern "C" lean_obj_res lean_hesper_cuda_malloc(size_t size) {
    CUdeviceptr ptr;
    CUDA_CHECK(cuMemAlloc(&ptr, size), "cuMemAlloc");
    return lean_io_result_mk_ok(lean_box_usize((size_t)ptr));
}

extern "C" lean_obj_res lean_hesper_cuda_free(size_t ptr_val) {
    CUDA_CHECK(cuMemFree((CUdeviceptr)ptr_val), "cuMemFree");
    return lean_io_result_mk_ok(lean_box(0));
}

extern "C" lean_obj_res lean_hesper_cuda_memcpy_htod(
    size_t dst_val, b_lean_obj_arg src_bytes, size_t offset, size_t size
) {
    CUDA_CHECK(cuMemcpyHtoD((CUdeviceptr)dst_val + offset,
                             lean_sarray_cptr(src_bytes), size), "cuMemcpyHtoD");
    return lean_io_result_mk_ok(lean_box(0));
}

extern "C" lean_obj_res lean_hesper_cuda_memcpy_dtoh(size_t src_val, size_t size) {
    CUDA_CHECK(cuCtxSynchronize(), "cuCtxSynchronize (before readback)");
    lean_obj_res arr = lean_alloc_sarray(1, size, size);
    CUDA_CHECK(cuMemcpyDtoH(lean_sarray_cptr(arr), (CUdeviceptr)src_val, size),
               "cuMemcpyDtoH");
    return lean_io_result_mk_ok(arr);
}

extern "C" lean_obj_res lean_hesper_cuda_memset(size_t ptr_val, size_t size) {
    CUDA_CHECK(cuMemsetD8((CUdeviceptr)ptr_val, 0, size), "cuMemsetD8");
    return lean_io_result_mk_ok(lean_box(0));
}

// ============================================================================
// L2 cache persistence (compute capability ≥ 8.0: Ampere+, Ada, Hopper)
// Lets the driver keep specific address ranges in L2 across kernel launches.
// ============================================================================

// Query the device's MAX_PERSISTING_L2_CACHE_SIZE and set the current
// context's CU_LIMIT_PERSISTING_L2_CACHE_SIZE to that maximum (or `size` if
// smaller).  Returns the effective limit applied.
extern "C" lean_obj_res lean_hesper_cuda_set_l2_persist_limit(size_t size) {
    CUdevice dev;
    CUDA_CHECK(cuCtxGetDevice(&dev), "cuCtxGetDevice");
    int maxL2 = 0;
    CUDA_CHECK(cuDeviceGetAttribute(&maxL2,
        CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE, dev),
        "cuDeviceGetAttribute(MAX_PERSISTING_L2_CACHE_SIZE)");
    size_t limit = size;
    if ((size_t)maxL2 > 0 && limit > (size_t)maxL2) limit = (size_t)maxL2;
    CUDA_CHECK(cuCtxSetLimit(CU_LIMIT_PERSISTING_L2_CACHE_SIZE, limit),
               "cuCtxSetLimit(PERSISTING_L2_CACHE_SIZE)");
    return lean_io_result_mk_ok(lean_box_usize(limit));
}

// Install a persisting access-policy window on the default (null) stream.
// Subsequent kernel launches on this stream will prefer keeping `[ptr, ptr+size)`
// resident in L2.  hitRatio=1.0 marks 100% of the range as persisting; bytes
// beyond ::MAX_ACCESS_POLICY_WINDOW_SIZE are silently clipped by the driver.
extern "C" lean_obj_res lean_hesper_cuda_set_l2_access_window(size_t ptr_val, size_t size) {
    CUstreamAttrValue attr = {};
    attr.accessPolicyWindow.base_ptr = (void*)ptr_val;
    attr.accessPolicyWindow.num_bytes = size;
    attr.accessPolicyWindow.hitRatio = 1.0f;
    attr.accessPolicyWindow.hitProp = CU_ACCESS_PROPERTY_PERSISTING;
    attr.accessPolicyWindow.missProp = CU_ACCESS_PROPERTY_STREAMING;
    CUDA_CHECK(cuStreamSetAttribute(
        /*hStream=*/0,
        CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW,
        &attr),
        "cuStreamSetAttribute(ACCESS_POLICY_WINDOW)");
    return lean_io_result_mk_ok(lean_box(0));
}

// Clear the persisting access window on the default stream.  Use before
// swapping to a new window so the old one doesn't keep its L2 allocation.
extern "C" lean_obj_res lean_hesper_cuda_reset_l2_persisting_cache() {
    CUDA_CHECK(cuCtxResetPersistingL2Cache(), "cuCtxResetPersistingL2Cache");
    return lean_io_result_mk_ok(lean_box(0));
}

// ============================================================================
// Kernel launch
// ============================================================================

extern "C" lean_obj_res lean_hesper_cuda_launch_kernel(
    size_t func_val,
    uint32_t gx, uint32_t gy, uint32_t gz,
    uint32_t bx, uint32_t by, uint32_t bz,
    uint32_t smem,
    b_lean_obj_arg arg_ptrs
) {
    size_t n = lean_array_size(arg_ptrs);
    CUdeviceptr* ptrs = (CUdeviceptr*)malloc(n * sizeof(CUdeviceptr));
    void** args = (void**)malloc(n * sizeof(void*));
    for (size_t i = 0; i < n; i++) {
        ptrs[i] = (CUdeviceptr)lean_unbox_usize(lean_array_get_core(arg_ptrs, i));
        args[i] = &ptrs[i];
    }
    CUDA_CHECK(cuLaunchKernel((CUfunction)func_val,
        gx, gy, gz, bx, by, bz,
        smem, nullptr, args, nullptr), "cuLaunchKernel");
    // No sync here — async launch. Sync happens in readBuffer.
    free(ptrs); free(args);
    return lean_io_result_mk_ok(lean_box(0));
}

// Raw-arg launch: for external PTX (e.g. llama.cpp's mul_mat_vec_q) whose
// kernels take mixed-type args (void*, u32, uint3, structs).  `arg_bytes`
// is a single ByteArray of packed arg values; `arg_offsets` gives the
// byte offset of each arg within that buffer.  CUDA's cuLaunchKernel
// takes `void**` where each entry points at an arg's value storage.
extern "C" lean_obj_res lean_hesper_cuda_launch_kernel_raw(
    size_t func_val,
    uint32_t gx, uint32_t gy, uint32_t gz,
    uint32_t bx, uint32_t by, uint32_t bz,
    uint32_t smem,
    b_lean_obj_arg arg_bytes,
    b_lean_obj_arg arg_offsets
) {
    size_t n = lean_array_size(arg_offsets);
    uint8_t* base = lean_sarray_cptr(arg_bytes);
    void** args = (void**)malloc(n * sizeof(void*));
    for (size_t i = 0; i < n; i++) {
        size_t off = lean_unbox_usize(lean_array_get_core(arg_offsets, i));
        args[i] = (void*)(base + off);
    }
    CUDA_CHECK(cuLaunchKernel((CUfunction)func_val,
        gx, gy, gz, bx, by, bz,
        smem, nullptr, args, nullptr), "cuLaunchKernelRaw");
    free(args);
    return lean_io_result_mk_ok(lean_box(0));
}

// ============================================================================
// Fast string hash (FNV-1a, ~4 GB/s on modern CPU)
// ============================================================================

extern "C" lean_obj_res lean_hesper_fast_string_hash(b_lean_obj_arg s) {
    const char* data = lean_string_cstr(s);
    size_t len = lean_string_size(s) - 1;
    // Full hash — correctness over speed (衝突はsilent corruption)
    size_t hash_len = len;
    uint64_t h = 14695981039346656037ULL;
    h ^= (uint64_t)len;
    h *= 1099511628211ULL;
    for (size_t i = 0; i < hash_len; i++) {
        h ^= (uint8_t)data[i];
        h *= 1099511628211ULL;
    }
    return lean_io_result_mk_ok(lean_box_usize((size_t)h));
}

// ============================================================================
// Memory-mapped file I/O
// ============================================================================

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// mmap a file, return (pointer, size) as two USize values packed in an array
extern "C" lean_obj_res lean_hesper_mmap_file(b_lean_obj_arg path_str) {
    const char* path = lean_string_cstr(path_str);
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string("mmap: open failed")));
    }
    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("mmap: fstat failed")));
    }
    size_t size = st.st_size;
    void* ptr = mmap(nullptr, size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
    close(fd);
    if (ptr == MAP_FAILED) {
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("mmap: mmap failed")));
    }
    // Return as pair (ptr, size)
    lean_obj_res pair = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(pair, 0, lean_box_usize((size_t)ptr));
    lean_ctor_set(pair, 1, lean_box_usize(size));
    return lean_io_result_mk_ok(pair);
}

extern "C" lean_obj_res lean_hesper_munmap(size_t ptr, size_t size) {
    munmap((void*)ptr, size);
    return lean_io_result_mk_ok(lean_box(0));
}

// Copy a slice of mmapped memory to a Lean ByteArray (for small metadata)
extern "C" lean_obj_res lean_hesper_mmap_slice_to_bytes(size_t ptr, size_t offset, size_t size) {
    lean_obj_res arr = lean_alloc_sarray(1, size, size);
    memcpy(lean_sarray_cptr(arr), (const char*)ptr + offset, size);
    return lean_io_result_mk_ok(arr);
}

// Copy a slice of mmapped memory directly to GPU (zero Lean-side copy)
extern "C" lean_obj_res lean_hesper_mmap_to_gpu(size_t mmap_ptr, size_t offset, size_t gpu_ptr, size_t size) {
    CUDA_CHECK(cuMemcpyHtoD((CUdeviceptr)gpu_ptr, (const char*)mmap_ptr + offset, size), "mmap_to_gpu");
    return lean_io_result_mk_ok(lean_box(0));
}

// Fast file read using mmap + memcpy (avoids Lean's IO.FS.readBinFile overhead)
extern "C" lean_obj_res lean_hesper_read_file_fast(b_lean_obj_arg path_str) {
    const char* path = lean_string_cstr(path_str);
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("readFileFast: open failed")));
    }
    struct stat st;
    fstat(fd, &st);
    size_t size = st.st_size;
    
    // mmap with MAP_POPULATE to prefault all pages
    void* src = mmap(nullptr, size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
    close(fd);
    if (src == MAP_FAILED) {
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("readFileFast: mmap failed")));
    }
    
    // Allocate Lean ByteArray and copy (memcpy from mmap is fast — pages already in memory)
    lean_obj_res arr = lean_alloc_sarray(1, size, size);
    memcpy(lean_sarray_cptr(arr), src, size);
    munmap(src, size);
    
    return lean_io_result_mk_ok(arr);
}

// ============================================================================
// CUDA Graphs — amortise per-dispatch launch overhead across decode tokens.
// ============================================================================

extern "C" lean_obj_res lean_hesper_cuda_stream_create() {
    CUstream s;
    CUDA_CHECK(cuStreamCreate(&s, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
    return lean_io_result_mk_ok(lean_box_usize((size_t)s));
}

extern "C" lean_obj_res lean_hesper_cuda_stream_destroy(size_t stream_val) {
    CUDA_CHECK(cuStreamDestroy((CUstream)stream_val), "cuStreamDestroy");
    return lean_io_result_mk_ok(lean_box(0));
}

extern "C" lean_obj_res lean_hesper_cuda_stream_begin_capture(size_t stream_val) {
    CUDA_CHECK(cuStreamBeginCapture((CUstream)stream_val, CU_STREAM_CAPTURE_MODE_RELAXED),
               "cuStreamBeginCapture");
    return lean_io_result_mk_ok(lean_box(0));
}

extern "C" lean_obj_res lean_hesper_cuda_stream_end_capture(size_t stream_val) {
    CUgraph graph;
    CUDA_CHECK(cuStreamEndCapture((CUstream)stream_val, &graph),
               "cuStreamEndCapture");
    return lean_io_result_mk_ok(lean_box_usize((size_t)graph));
}

extern "C" lean_obj_res lean_hesper_cuda_graph_instantiate(size_t graph_val) {
    CUgraphExec exec;
    // cuGraphInstantiate signature takes (exec*, graph, flags) on recent
    // CUDA; older drivers need the 4-arg form.  Use the 2-arg helper
    // that's stable across 11.x / 12.x.
    CUDA_CHECK(cuGraphInstantiate(&exec, (CUgraph)graph_val, 0),
               "cuGraphInstantiate");
    return lean_io_result_mk_ok(lean_box_usize((size_t)exec));
}

extern "C" lean_obj_res lean_hesper_cuda_graph_exec_destroy(size_t exec_val) {
    CUDA_CHECK(cuGraphExecDestroy((CUgraphExec)exec_val), "cuGraphExecDestroy");
    return lean_io_result_mk_ok(lean_box(0));
}

extern "C" lean_obj_res lean_hesper_cuda_graph_destroy(size_t graph_val) {
    CUDA_CHECK(cuGraphDestroy((CUgraph)graph_val), "cuGraphDestroy");
    return lean_io_result_mk_ok(lean_box(0));
}

extern "C" lean_obj_res lean_hesper_cuda_graph_launch(size_t exec_val,
                                                      size_t stream_val) {
    CUDA_CHECK(cuGraphLaunch((CUgraphExec)exec_val, (CUstream)stream_val),
               "cuGraphLaunch");
    return lean_io_result_mk_ok(lean_box(0));
}

extern "C" lean_obj_res lean_hesper_cuda_stream_synchronize(size_t stream_val) {
    CUDA_CHECK(cuStreamSynchronize((CUstream)stream_val), "cuStreamSynchronize");
    return lean_io_result_mk_ok(lean_box(0));
}

extern "C" lean_obj_res lean_hesper_cuda_launch_kernel_on_stream(
    size_t func_val,
    uint32_t gx, uint32_t gy, uint32_t gz,
    uint32_t bx, uint32_t by, uint32_t bz,
    uint32_t smem,
    size_t stream_val,
    b_lean_obj_arg arg_ptrs
) {
    size_t n = lean_array_size(arg_ptrs);
    CUdeviceptr* ptrs = (CUdeviceptr*)malloc(n * sizeof(CUdeviceptr));
    void** args = (void**)malloc(n * sizeof(void*));
    for (size_t i = 0; i < n; i++) {
        ptrs[i] = (CUdeviceptr)lean_unbox_usize(lean_array_get_core(arg_ptrs, i));
        args[i] = &ptrs[i];
    }
    CUDA_CHECK(cuLaunchKernel((CUfunction)func_val,
        gx, gy, gz, bx, by, bz,
        smem, (CUstream)stream_val, args, nullptr), "cuLaunchKernelOnStream");
    free(ptrs); free(args);
    return lean_io_result_mk_ok(lean_box(0));
}

extern "C" lean_obj_res lean_hesper_cuda_memcpy_htod_async(
    size_t dst_ptr,
    b_lean_obj_arg src_bytes,
    size_t offset,
    size_t size,
    size_t stream_val
) {
    const uint8_t* src = lean_sarray_cptr(src_bytes);
    CUDA_CHECK(cuMemcpyHtoDAsync((CUdeviceptr)(dst_ptr + offset), src, size, (CUstream)stream_val),
               "cuMemcpyHtoDAsync");
    return lean_io_result_mk_ok(lean_box(0));
}

// ============================================================================
// Pinned host memory for CUDA Graph-capture-safe writes
// ============================================================================

extern "C" lean_obj_res lean_hesper_cuda_mem_alloc_host(size_t size) {
    void* ptr = nullptr;
    CUDA_CHECK(cuMemHostAlloc(&ptr, size, CU_MEMHOSTALLOC_PORTABLE),
               "cuMemHostAlloc");
    return lean_io_result_mk_ok(lean_box_usize((size_t)ptr));
}

extern "C" lean_obj_res lean_hesper_cuda_mem_free_host(size_t host_ptr) {
    CUDA_CHECK(cuMemFreeHost((void*)host_ptr), "cuMemFreeHost");
    return lean_io_result_mk_ok(lean_box(0));
}

extern "C" lean_obj_res lean_hesper_cuda_write_pinned(
    size_t host_ptr,
    size_t offset,
    b_lean_obj_arg src_bytes,
    size_t size
) {
    const uint8_t* src = lean_sarray_cptr(src_bytes);
    memcpy((uint8_t*)host_ptr + offset, src, size);
    return lean_io_result_mk_ok(lean_box(0));
}

extern "C" lean_obj_res lean_hesper_cuda_memcpy_htod_from_pinned(
    size_t dst_ptr,
    size_t host_ptr,
    size_t offset,
    size_t size,
    size_t stream_val
) {
    CUDA_CHECK(cuMemcpyHtoDAsync((CUdeviceptr)dst_ptr,
                                 (const void*)(host_ptr + offset),
                                 size, (CUstream)stream_val),
               "cuMemcpyHtoDFromPinned");
    return lean_io_result_mk_ok(lean_box(0));
}
