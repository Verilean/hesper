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
    CUresult err = cuModuleLoadData(&mod, ptx);
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
    CUDA_CHECK(cuCtxSynchronize(), "cuCtxSynchronize");
    free(ptrs); free(args);
    return lean_io_result_mk_ok(lean_box(0));
}
