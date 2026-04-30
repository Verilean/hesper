// CUDA Driver API bridge for Hesper PTX JIT backend.
// Direct linking against -lcuda. Uses cuda.h for correct types.
// NOTE: Lean 4 IO FFI does NOT pass a world token argument.

#include <lean/lean.h>
#include <cuda.h>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cerrno>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

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

// Simple 64-bit FNV-1a hash for cubin filename key.
static uint64_t fnv1a64(const char* s, size_t n) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= (uint8_t)s[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

// Cache directory for cubin artifacts.  Keyed by FNV64(ptx).  Avoids
// JIT on warm starts — `cuModuleLoadData(cubin)` is O(µs) vs cubin
// compilation O(200µs) × 259 kernels = 50 ms.
static const char* hesper_cubin_cache_dir() {
    static char dir[512] = {0};
    if (dir[0] == '\0') {
        const char* env = getenv("HESPER_CUBIN_CACHE");
        if (env && *env) { snprintf(dir, sizeof(dir), "%s", env); }
        else {
            const char* home = getenv("HOME");
            if (home) snprintf(dir, sizeof(dir), "%s/.cache/hesper/cubin", home);
            else      snprintf(dir, sizeof(dir), "/tmp/hesper-cubin");
        }
    }
    return dir;
}

static void hesper_mkdir_p(const char* path) {
    char buf[512];
    snprintf(buf, sizeof(buf), "%s", path);
    for (char* p = buf + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            int r = mkdir(buf, 0755);
            if (getenv("HESPER_CUBIN_DEBUG")) {
                fprintf(stderr, "[cubin] mkdir %s rc=%d errno=%d\n", buf, r, errno);
            }
            *p = '/';
        }
    }
    int r = mkdir(buf, 0755);
    if (getenv("HESPER_CUBIN_DEBUG")) {
        fprintf(stderr, "[cubin] mkdir %s rc=%d errno=%d\n", buf, r, errno);
    }
}

extern "C" lean_obj_res lean_hesper_cuda_module_load_data(b_lean_obj_arg ptx_str) {
    const char* ptx = lean_string_cstr(ptx_str);
    size_t ptx_len = strlen(ptx);
    uint64_t key = fnv1a64(ptx, ptx_len);

    // 1. Try loading cached cubin from disk.
    char cubin_path[640];
    snprintf(cubin_path, sizeof(cubin_path), "%s/%016lx.cubin",
             hesper_cubin_cache_dir(), (unsigned long)key);
    FILE* f = fopen(cubin_path, "rb");
    if (f) {
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        if (sz > 0 && sz < (1 << 24)) {  // sanity: < 16 MB
            void* buf = malloc(sz);
            if (buf) {
                size_t rd = fread(buf, 1, sz, f);
                fclose(f);
                if (rd == (size_t)sz) {
                    CUmodule mod;
                    CUresult ferr = cuModuleLoadData(&mod, buf);
                    free(buf);
                    if (ferr == CUDA_SUCCESS) {
                        return lean_io_result_mk_ok(lean_box_usize((size_t)mod));
                    }
                    // Fall through to JIT on load failure (possibly
                    // stale cubin from a different driver/arch).
                } else {
                    free(buf);
                }
            } else {
                fclose(f);
            }
        } else {
            fclose(f);
        }
    }

    // 2. JIT compile.  Request CU_JIT_TARGET from current device.
    CUmodule mod;
    CUjit_option opts[] = { CU_JIT_OPTIMIZATION_LEVEL };
    void* optVals[] = { (void*)(uintptr_t)4 };
    CUresult err = cuModuleLoadDataEx(&mod, ptx, 1, opts, optVals);
    if (err != CUDA_SUCCESS) {
        const char* errName = nullptr;
        cuGetErrorName(err, &errName);
        char buf[1024];
        size_t showLen = ptx_len; if (showLen > 500) showLen = 500;
        snprintf(buf, sizeof(buf), "cuModuleLoadData failed: %s\nPTX (%zu chars):\n%.*s",
                 errName ? errName : "unknown", showLen, (int)showLen, ptx);
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(buf)));
    }

    // 3. Extract cubin and write to disk for future runs.
    //    cuLinkCreate + cuLinkAddData(PTX) + cuLinkComplete gives us
    //    cubin bytes without needing to round-trip through module.
    static int cache_dir_created = 0;
    if (!cache_dir_created) {
        hesper_mkdir_p(hesper_cubin_cache_dir());
        cache_dir_created = 1;
    }
    CUlinkState link;
    CUresult lres = cuLinkCreate(0, nullptr, nullptr, &link);
    if (lres == CUDA_SUCCESS) {
        CUresult ares = cuLinkAddData(link, CU_JIT_INPUT_PTX, (void*)ptx, ptx_len + 1,
                                       "kern", 0, nullptr, nullptr);
        if (ares == CUDA_SUCCESS) {
            void* cubin = nullptr;
            size_t cubin_size = 0;
            CUresult cres = cuLinkComplete(link, &cubin, &cubin_size);
            if (cres == CUDA_SUCCESS && cubin && cubin_size > 0) {
                FILE* fo = fopen(cubin_path, "wb");
                if (fo) {
                    fwrite(cubin, 1, cubin_size, fo);
                    fclose(fo);
                } else if (getenv("HESPER_CUBIN_DEBUG")) {
                    fprintf(stderr, "[cubin] fopen write failed: %s\n", cubin_path);
                }
            } else if (getenv("HESPER_CUBIN_DEBUG")) {
                fprintf(stderr, "[cubin] cuLinkComplete rc=%d size=%zu\n", cres, cubin_size);
            }
        } else if (getenv("HESPER_CUBIN_DEBUG")) {
            fprintf(stderr, "[cubin] cuLinkAddData rc=%d\n", ares);
        }
        cuLinkDestroy(link);
    } else if (getenv("HESPER_CUBIN_DEBUG")) {
        fprintf(stderr, "[cubin] cuLinkCreate rc=%d\n", lres);
    }
    return lean_io_result_mk_ok(lean_box_usize((size_t)mod));
}

// Variant of cuModuleLoadData that takes raw bytes (for cubin or fatbin
// loading where the source is binary, not UTF-8 text). Skips the JIT and
// disk-cache paths; the input is assumed to already be a fully compiled
// cubin/fatbin/PTX-as-bytes that the driver can load directly.
extern "C" lean_obj_res lean_hesper_cuda_module_load_data_bytes(b_lean_obj_arg ba) {
    const uint8_t* data = lean_sarray_cptr(ba);
    size_t len = lean_sarray_size(ba);
    CUmodule mod;
    CUresult err = cuModuleLoadData(&mod, data);
    if (err != CUDA_SUCCESS) {
        const char* errName = nullptr;
        cuGetErrorName(err, &errName);
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "cuModuleLoadData (bytes, %zu B) failed: %s",
                 len, errName ? errName : "unknown");
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

// Raise a kernel's dynamic shared-memory limit. Required for kernels
// that request > 48 KB of dynamic smem (e.g. llama.cpp's mmq tile-GEMM
// at mmq_y=128, mmq_x=64, ~46-50 KB needed).
extern "C" lean_obj_res lean_hesper_cuda_func_set_max_dynamic_smem(size_t func_val, size_t bytes) {
    CUDA_CHECK(cuFuncSetAttribute((CUfunction)func_val,
                                   CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                   (int)bytes),
               "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)");
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
    // Use cuMemcpyHtoDAsync on stream 0 (legacy default) instead of the
    // sync cuMemcpyHtoD.  Semantics match: pageable-source HtoDAsync is
    // synchronous internally, and stream 0 serialises with cuLaunchKernel
    // on the null stream.  This eliminates the entire `cuMemcpyHtoD_v2`
    // API row from nsys traces, matching llama.cpp's all-Async profile.
    CUDA_CHECK(cuMemcpyHtoDAsync((CUdeviceptr)dst_val + offset,
                                  lean_sarray_cptr(src_bytes), size,
                                  /* stream = legacy default */ 0),
               "cuMemcpyHtoDAsync(legacy)");
    return lean_io_result_mk_ok(lean_box(0));
}

extern "C" lean_obj_res lean_hesper_cuda_memcpy_dtoh(size_t src_val, size_t size) {
    // cuMemcpyDtoH itself is synchronous (waits for all prior work on the
    // default stream), so the explicit cuCtxSynchronize is redundant and
    // was adding an extra host→driver round trip (~8.8 ms × 10 tokens =
    // 88 ms / decode run observed via nsys).  Removed; cuMemcpyDtoH alone
    // is correct and waits just as long as needed.
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

extern "C" lean_obj_res lean_hesper_cuda_launch_kernel_raw_on_stream(
    size_t func_val,
    uint32_t gx, uint32_t gy, uint32_t gz,
    uint32_t bx, uint32_t by, uint32_t bz,
    uint32_t smem,
    b_lean_obj_arg arg_bytes,
    b_lean_obj_arg arg_offsets,
    size_t stream_val
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
        smem, (CUstream)stream_val, args, nullptr), "cuLaunchKernelRawOnStream");
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
    CUDA_CHECK(cuMemcpyHtoDAsync((CUdeviceptr)gpu_ptr,
                                  (const char*)mmap_ptr + offset, size,
                                  /* stream = legacy default */ 0),
               "mmap_to_gpu(async)");
    return lean_io_result_mk_ok(lean_box(0));
}

// ============================================================================
// Persistent mmap with Lean GC-managed lifetime.
//
// Unlike `lean_hesper_read_file_fast` (which mmap+memcpy+munmap and returns a
// Lean ByteArray), this holds the mmap open and exposes a raw `void*` via a
// Lean external object.  When the last reference is dropped, the finalizer
// runs `munmap`.  Tensor slices are created as offsets into the mmap; they
// share a Lean reference to the mmap handle so the region stays live while
// any slice is in use — this lets us pass `mmap_ptr + offset` directly to
// `cuMemcpyHtoDAsync` without copying through Lean heap.
// ============================================================================

struct hesper_mmap_handle {
    void*  addr;
    size_t size;
    bool   pinned;  // true if the region was cuMemHostRegister'd
};

static void hesper_mmap_finalize(void* obj) {
    auto* h = static_cast<hesper_mmap_handle*>(obj);
    if (h && h->addr && h->addr != MAP_FAILED) {
        if (h->pinned) {
            // Best-effort unregister; ignore errors (process is exiting).
            cuMemHostUnregister(h->addr);
        }
        munmap(h->addr, h->size);
    }
    delete h;
}

static void hesper_mmap_foreach(void*, b_lean_obj_arg) {
    // No Lean references owned.
}

static lean_external_class* hesper_mmap_class = nullptr;

static lean_external_class* get_mmap_class() {
    if (!hesper_mmap_class) {
        hesper_mmap_class = lean_register_external_class(
            hesper_mmap_finalize, hesper_mmap_foreach);
    }
    return hesper_mmap_class;
}

/// Open a file as mmap.  Returns a Lean external object; munmap runs when GC
/// reclaims it.  MAP_POPULATE prefaults all pages (same as readFileFast) to
/// avoid page-fault latency on first access.
extern "C" lean_obj_res lean_hesper_mmap_open_persistent(b_lean_obj_arg path_str) {
    const char* path = lean_string_cstr(path_str);
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string("mmapOpen: open failed")));
    }
    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string("mmapOpen: fstat failed")));
    }
    size_t size = st.st_size;
    void* addr = mmap(nullptr, size, PROT_READ,
                      MAP_PRIVATE | MAP_POPULATE, fd, 0);
    close(fd);
    if (addr == MAP_FAILED) {
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string("mmapOpen: mmap failed")));
    }
    // Register the mmap region as pinned host memory so subsequent
    // cuMemcpyHtoDAsync calls are *truly* asynchronous (not blocked on
    // pageable-source sync).  CU_MEMHOSTREGISTER_READ_ONLY tells the
    // driver the file region won't be written by host — matches PROT_READ.
    // On failure (out of pinnable RAM, kernel limit, etc.) fall back to
    // unpinned semantics; copies still work, just block host.
    // Optionally register the mmap region as pinned host memory so subsequent
    // cuMemcpyHtoDAsync calls actually overlap with kernel work.  This costs
    // ~200 ms/GB at startup (page locking syscall), but the resulting H2D
    // becomes truly async (we measured 311 ms → 1.8 ms total transfer time
    // for 982 calls when pinned).  Disabled by default because the pin
    // syscall is sequential and not amortised on short sessions.  Set
    // HESPER_PIN_MMAP=1 to opt in (long-running servers, batch jobs).
    bool pinned = false;
    if (getenv("HESPER_PIN_MMAP") != nullptr) {
        CUresult rc = cuMemHostRegister(addr, size,
            CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_READ_ONLY);
        if (rc == CUDA_SUCCESS) {
            pinned = true;
            fprintf(stderr, "[mmap] registered %.2f MB as pinned host memory\n",
                    size / (1024.0 * 1024.0));
        } else {
            const char* errStr = nullptr;
            cuGetErrorString(rc, &errStr);
            fprintf(stderr, "[mmap] cuMemHostRegister failed (%s); copies will be sync\n",
                    errStr ? errStr : "unknown");
        }
        fflush(stderr);
    }
    auto* h = new hesper_mmap_handle{addr, size, pinned};
    lean_object* ext = lean_alloc_external(get_mmap_class(), h);
    return lean_io_result_mk_ok(ext);
}

/// Get the total size of an mmap'd file.
extern "C" lean_obj_res lean_hesper_mmap_size(b_lean_obj_arg h_obj) {
    auto* h = static_cast<hesper_mmap_handle*>(lean_get_external_data(h_obj));
    return lean_io_result_mk_ok(lean_box_usize(h->size));
}

/// Pin a sub-range of an mmap region as page-locked + map it into the CUDA
/// unified-VA space, returning the *device-side* pointer that aliases the
/// host range.  Kernels can dereference that pointer directly via global
/// loads — the driver pulls pages over PCIe on demand (the same trick
/// llama.cpp's getrows kernel uses for `tok_embd_per_layer`).
///
/// Cost: ~200 ms/GB on first call (mlock + cuMemHostRegister), once at
/// model load.  No per-token cuMemcpy needed afterwards.
///
/// Caller may pass any offset/size; we page-align internally and return
/// the device pointer corresponding to the *requested* offset (not the
/// page-aligned region start).  Size is also rounded up to cover the
/// requested range.
///
/// Returns the device pointer (`USize`) on success.  On failure (OOM,
/// kernel limits, already-registered overlap), returns a Lean IO error.
extern "C" lean_obj_res lean_hesper_mmap_register_region(
        b_lean_obj_arg h_obj, size_t offset, size_t size) {
    auto* h = static_cast<hesper_mmap_handle*>(lean_get_external_data(h_obj));
    if (offset + size > h->size) {
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string("mmapRegisterRegion: out of range")));
    }
    constexpr size_t PAGE = 4096;
    // Snap [offset, offset+size) outward to page boundaries so cuMemHostRegister
    // sees an aligned region; clamp the upper end to the mmap size to avoid
    // registering bytes past the file.
    size_t region_off = offset & ~(PAGE - 1);
    size_t region_end = ((offset + size + PAGE - 1) & ~(PAGE - 1));
    if (region_end > h->size) region_end = h->size;
    size_t region_size = region_end - region_off;
    void* region_ptr = (char*)h->addr + region_off;
    CUresult rc = cuMemHostRegister(region_ptr, region_size,
        CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_READ_ONLY);
    if (rc != CUDA_SUCCESS) {
        const char* errStr = nullptr;
        cuGetErrorString(rc, &errStr);
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "mmapRegisterRegion: cuMemHostRegister failed: %s",
                 errStr ? errStr : "unknown");
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(buf)));
    }
    CUdeviceptr region_dev = 0;
    rc = cuMemHostGetDevicePointer(&region_dev, region_ptr, 0);
    if (rc != CUDA_SUCCESS) {
        cuMemHostUnregister(region_ptr);
        const char* errStr = nullptr;
        cuGetErrorString(rc, &errStr);
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "mmapRegisterRegion: cuMemHostGetDevicePointer failed: %s",
                 errStr ? errStr : "unknown");
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(buf)));
    }
    // Shift the returned pointer by the in-page offset so the caller gets
    // a device pointer that aliases the requested host range exactly.
    size_t in_page_off = offset - region_off;
    return lean_io_result_mk_ok(lean_box_usize((size_t)region_dev + in_page_off));
}

/// Copy a slice of the mmap to a fresh Lean ByteArray.  Used for metadata
/// parsing which needs Lean-side operations; tensor bodies should NOT use
/// this — use the GPU-direct path below.
extern "C" lean_obj_res lean_hesper_mmap_slice_to_bytes_persistent(
        b_lean_obj_arg h_obj, size_t offset, size_t size) {
    auto* h = static_cast<hesper_mmap_handle*>(lean_get_external_data(h_obj));
    if (offset + size > h->size) {
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string("mmapSliceToBytes: out of range")));
    }
    lean_obj_res arr = lean_alloc_sarray(1, size, size);
    memcpy(lean_sarray_cptr(arr), (char*)h->addr + offset, size);
    return lean_io_result_mk_ok(arr);
}

/// H2D copy straight from mmap to GPU, async on the given stream.  The Lean
/// caller must keep the mmap handle live (typically by holding it in the
/// same structure as the returned GPU buffer handle) until all in-flight
/// async copies complete — stream-synchronize guarantees that.
extern "C" lean_obj_res lean_hesper_cuda_memcpy_htod_from_mmap(
        size_t dst_ptr, b_lean_obj_arg h_obj, size_t offset,
        size_t size, size_t stream_val) {
    auto* h = static_cast<hesper_mmap_handle*>(lean_get_external_data(h_obj));
    if (offset + size > h->size) {
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string("cuMemcpyHtoDFromMmap: out of range")));
    }
    void* src = (char*)h->addr + offset;
    // Always use cuMemcpyHtoDAsync.  When stream=0 it goes onto the
    // legacy default stream — same ordering as cuMemcpyHtoD, but counted
    // as Async in nsys traces, matching llama.cpp's all-Async profile.
    CUDA_CHECK(cuMemcpyHtoDAsync((CUdeviceptr)dst_ptr, src, size,
                                  (CUstream)stream_val),
               stream_val == 0 ? "cuMemcpyHtoDAsync(mmap, legacy)"
                                : "cuMemcpyHtoDAsync(mmap)");
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

// Create a stream with default (blocking) semantics: synchronises
// implicitly with the null stream.  Used for `cudaDefaultStream`
// so that readBuffer / cuMemcpyDtoH on the null stream observes
// prior H2D + kernel launches submitted on this stream without an
// explicit `cuStreamSynchronize`.
extern "C" lean_obj_res lean_hesper_cuda_stream_create_default() {
    CUstream s;
    CUDA_CHECK(cuStreamCreate(&s, CU_STREAM_DEFAULT), "cuStreamCreate(default)");
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

// Host-mapped pinned allocation.  Returns a Lean Pair (host_ptr, dev_ptr)
// where the device pointer aliases the same memory through CUDA's unified
// VA — kernels can ld/st it directly via global ops, and the host can read
// the result with no driver call once the producing stream has been
// synchronised (vs. the cuMemcpyDtoH(4 byte) sync that costs ~9.8 ms/tok
// on the per-token argmax path because the driver implicitly drains the
// stream).
extern "C" lean_obj_res lean_hesper_cuda_mem_alloc_host_mapped(size_t size) {
    void* host_ptr = nullptr;
    // PORTABLE | DEVICEMAP: usable from any context, mapped into the
    // device's VA space.  No WRITECOMBINED — argmax is also read by host,
    // and WC penalises host reads heavily.
    CUDA_CHECK(cuMemHostAlloc(&host_ptr, size,
                CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP),
               "cuMemHostAlloc(MAPPED)");
    CUdeviceptr dev_ptr = 0;
    CUresult rc = cuMemHostGetDevicePointer(&dev_ptr, host_ptr, 0);
    if (rc != CUDA_SUCCESS) {
        cuMemFreeHost(host_ptr);
        const char* errStr = nullptr;
        cuGetErrorString(rc, &errStr);
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "cuMemAllocHostMapped: cuMemHostGetDevicePointer failed: %s",
                 errStr ? errStr : "unknown");
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(buf)));
    }
    // Pack as a Lean Prod (host_ptr, dev_ptr) — both as USize.
    lean_object* pair = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(pair, 0, lean_box_usize((size_t)host_ptr));
    lean_ctor_set(pair, 1, lean_box_usize((size_t)dev_ptr));
    return lean_io_result_mk_ok(pair);
}

// Read a u32 from a pinned host pointer.  No driver call.  Caller must
// have synchronised the stream that wrote the value (e.g. via
// cuStreamSynchronize).
extern "C" lean_obj_res lean_hesper_cuda_read_pinned_u32(size_t host_ptr) {
    uint32_t v = *(volatile uint32_t*)host_ptr;
    return lean_io_result_mk_ok(lean_box_uint32(v));
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

/*
 * Fused: write to pinned slot + async H2D copy + stream dispatch, all in a
 * single FFI crossing.  Reduces Lean→FFI call-count overhead from 3-4
 * per scalar write to 1.  Pinned slot lifetime is managed by the caller;
 * this function just writes `size` bytes from `src_bytes` into the slot at
 * `host_ptr + offset` and immediately queues an async HtoD copy.
 */
extern "C" lean_obj_res lean_hesper_cuda_pinned_write_and_copy(
    size_t dst_ptr,
    size_t host_ptr,
    size_t offset,
    b_lean_obj_arg src_bytes,
    size_t size,
    size_t stream_val
) {
    const uint8_t* src = lean_sarray_cptr(src_bytes);
    memcpy((uint8_t*)host_ptr + offset, src, size);
    CUDA_CHECK(cuMemcpyHtoDAsync((CUdeviceptr)dst_ptr,
                                 (const void*)((uint8_t*)host_ptr + offset),
                                 size, (CUstream)stream_val),
               "cuMemcpyHtoDAsync(pinned_write_and_copy)");
    return lean_io_result_mk_ok(lean_box(0));
}

// ============================================================================
// Launch descriptor table (Option B+ metadata-free forward)
//
// Forward metadata (kernel handle, grid/block, argument pointer list) lives
// in C-owned storage so the Lean hot path between launches allocates nothing.
// See docs/llama-fusion-analysis/53-metadata-free-forward-design.md.
// ============================================================================

struct HesperLaunchDescriptor {
    CUfunction func;
    uint32_t gx, gy, gz;
    uint32_t bx, by, bz;
    uint32_t smem;
    uint32_t n_args;
    // Args are stored as an array of CUdeviceptr. For cuLaunchKernel we need
    // `void**` where each element points at the value storage — so we keep a
    // parallel `arg_ptrs` array of `void*` each pointing into `arg_storage`.
    CUdeviceptr* arg_storage;
    void** arg_ptrs;
};

// Fixed-capacity global table. Chosen large enough for a single Gemma-4 decode
// schedule (~900 launches). Grows only at init time; no realloc during forward.
#define HESPER_DESC_CAPACITY 4096
static HesperLaunchDescriptor g_descriptors[HESPER_DESC_CAPACITY];
static uint32_t g_descriptor_count = 0;

// Reset the descriptor pool (call when rebuilding the schedule, e.g. new
// inference state).
extern "C" lean_obj_res lean_hesper_desc_reset() {
    for (uint32_t i = 0; i < g_descriptor_count; i++) {
        free(g_descriptors[i].arg_storage);
        free(g_descriptors[i].arg_ptrs);
    }
    g_descriptor_count = 0;
    return lean_io_result_mk_ok(lean_box(0));
}

// Register a descriptor; returns its id as a boxed USize.
// `args` is a Lean Array USize (CUdeviceptr values).
extern "C" lean_obj_res lean_hesper_desc_register(
    size_t func_val,
    uint32_t gx, uint32_t gy, uint32_t gz,
    uint32_t bx, uint32_t by, uint32_t bz,
    uint32_t smem,
    b_lean_obj_arg args
) {
    if (g_descriptor_count >= HESPER_DESC_CAPACITY) {
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string("hesper_desc_register: descriptor pool exhausted")));
    }
    size_t n = lean_array_size(args);
    HesperLaunchDescriptor* d = &g_descriptors[g_descriptor_count];
    d->func = (CUfunction)func_val;
    d->gx = gx; d->gy = gy; d->gz = gz;
    d->bx = bx; d->by = by; d->bz = bz;
    d->smem = smem;
    d->n_args = (uint32_t)n;
    d->arg_storage = (CUdeviceptr*)malloc(n * sizeof(CUdeviceptr));
    d->arg_ptrs = (void**)malloc(n * sizeof(void*));
    for (size_t i = 0; i < n; i++) {
        d->arg_storage[i] = (CUdeviceptr)lean_unbox_usize(lean_array_get_core(args, i));
        d->arg_ptrs[i] = &d->arg_storage[i];
    }
    uint32_t id = g_descriptor_count++;
    return lean_io_result_mk_ok(lean_box_usize((size_t)id));
}

// Rebind a single argument slot of an existing descriptor. Used when a buffer
// pointer changes between calls (rare in hesper's decode path).
extern "C" lean_obj_res lean_hesper_desc_rebind(
    size_t desc_id, size_t slot, size_t new_ptr
) {
    if (desc_id >= g_descriptor_count) {
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string("hesper_desc_rebind: invalid descriptor id")));
    }
    HesperLaunchDescriptor* d = &g_descriptors[desc_id];
    if (slot >= d->n_args) {
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string("hesper_desc_rebind: slot out of range")));
    }
    d->arg_storage[slot] = (CUdeviceptr)new_ptr;
    // arg_ptrs[slot] already points at arg_storage[slot], unchanged.
    return lean_io_result_mk_ok(lean_box(0));
}

// Fast-path launch: pure C, no Lean heap touched between the FFI entry and
// cuLaunchKernel. `stream_val=0` means default stream.
extern "C" lean_obj_res lean_hesper_desc_launch(size_t desc_id, size_t stream_val) {
    if (desc_id >= g_descriptor_count) {
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string("hesper_desc_launch: invalid descriptor id")));
    }
    HesperLaunchDescriptor* d = &g_descriptors[desc_id];
    CUDA_CHECK(cuLaunchKernel(d->func,
        d->gx, d->gy, d->gz,
        d->bx, d->by, d->bz,
        d->smem, (CUstream)stream_val, d->arg_ptrs, nullptr),
        "cuLaunchKernel(desc)");
    return lean_io_result_mk_ok(lean_box(0));
}

// Fused rebind-all-args + launch.  Used when buffer pointers are
// allowed to change between calls (e.g. same logical call site but
// different layer weights).  Writes each arg into `arg_storage` in
// place — still zero Lean heap allocation past the FFI boundary.
extern "C" lean_obj_res lean_hesper_desc_launch_with_args(
    size_t desc_id, size_t stream_val, b_lean_obj_arg args
) {
    if (desc_id >= g_descriptor_count) {
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string("hesper_desc_launch_with_args: invalid descriptor id")));
    }
    HesperLaunchDescriptor* d = &g_descriptors[desc_id];
    size_t n = lean_array_size(args);
    if (n != d->n_args) {
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string("hesper_desc_launch_with_args: arg count mismatch")));
    }
    for (size_t i = 0; i < n; i++) {
        d->arg_storage[i] = (CUdeviceptr)lean_unbox_usize(lean_array_get_core(args, i));
    }
    CUDA_CHECK(cuLaunchKernel(d->func,
        d->gx, d->gy, d->gz,
        d->bx, d->by, d->bz,
        d->smem, (CUstream)stream_val, d->arg_ptrs, nullptr),
        "cuLaunchKernel(desc-rebind)");
    return lean_io_result_mk_ok(lean_box(0));
}
