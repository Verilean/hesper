// CUDA Driver API bridge for Hesper PTX JIT backend.
// Uses the CUDA Driver API (libcuda.so) directly — no nvcc or cudart needed.
// PTX strings are JIT-compiled at runtime via cuModuleLoadData.

#include <lean/lean.h>
#include <cuda.h>
#include <cstring>
#include <iostream>
#include <vector>

// ============================================================================
// Error handling
// ============================================================================

static lean_obj_res cuda_error(CUresult err, const char* func) {
    const char* errName = nullptr;
    const char* errStr = nullptr;
    cuGetErrorName(err, &errName);
    cuGetErrorString(err, &errStr);
    std::string msg = std::string(func) + " failed: " +
        (errName ? errName : "unknown") + " - " +
        (errStr ? errStr : "unknown");
    return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(msg.c_str())));
}

#define CUDA_CHECK(call, func) \
    do { CUresult err = (call); if (err != CUDA_SUCCESS) return cuda_error(err, func); } while(0)

// ============================================================================
// Device management
// ============================================================================

// Initialize CUDA driver
extern "C" lean_obj_res lean_hesper_cuda_init(lean_obj_arg /* world */) {
    CUDA_CHECK(cuInit(0), "cuInit");
    return lean_io_result_mk_ok(lean_box(0));
}

// Get device count
extern "C" lean_obj_res lean_hesper_cuda_device_count(lean_obj_arg /* world */) {
    int count = 0;
    CUDA_CHECK(cuDeviceGetCount(&count), "cuDeviceGetCount");
    return lean_io_result_mk_ok(lean_box(count));
}

// Get device by index
extern "C" lean_obj_res lean_hesper_cuda_device_get(uint32_t idx, lean_obj_arg /* world */) {
    CUdevice dev;
    CUDA_CHECK(cuDeviceGet(&dev, idx), "cuDeviceGet");
    return lean_io_result_mk_ok(lean_box(dev));
}

// Get device name
extern "C" lean_obj_res lean_hesper_cuda_device_name(uint32_t dev, lean_obj_arg /* world */) {
    char name[256];
    CUDA_CHECK(cuDeviceGetName(name, sizeof(name), dev), "cuDeviceGetName");
    return lean_io_result_mk_ok(lean_mk_string(name));
}

// Create context
extern "C" lean_obj_res lean_hesper_cuda_ctx_create(uint32_t dev, lean_obj_arg /* world */) {
    CUcontext ctx;
    CUDA_CHECK(cuCtxCreate(&ctx, 0, dev), "cuCtxCreate");
    // Store as opaque pointer in Lean external object
    return lean_io_result_mk_ok(lean_box((size_t)ctx));
}

// Destroy context
extern "C" lean_obj_res lean_hesper_cuda_ctx_destroy(size_t ctx_val, lean_obj_arg /* world */) {
    CUcontext ctx = (CUcontext)ctx_val;
    CUDA_CHECK(cuCtxDestroy(ctx), "cuCtxDestroy");
    return lean_io_result_mk_ok(lean_box(0));
}

// ============================================================================
// Module (PTX JIT compilation)
// ============================================================================

// Load PTX string → compiled module
extern "C" lean_obj_res lean_hesper_cuda_module_load_data(
    b_lean_obj_arg ptx_str, lean_obj_arg /* world */
) {
    const char* ptx = lean_string_cstr(ptx_str);

    CUmodule mod;
    CUresult err = cuModuleLoadData(&mod, ptx);
    if (err != CUDA_SUCCESS) {
        // Include PTX compilation error details
        const char* errName = nullptr;
        cuGetErrorName(err, &errName);
        std::string msg = "cuModuleLoadData failed: ";
        msg += errName ? errName : "unknown";
        msg += "\nPTX source (first 500 chars):\n";
        msg += std::string(ptx, std::min(strlen(ptx), (size_t)500));
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(msg.c_str())));
    }

    return lean_io_result_mk_ok(lean_box((size_t)mod));
}

// Get function from module
extern "C" lean_obj_res lean_hesper_cuda_module_get_function(
    size_t mod_val, b_lean_obj_arg func_name, lean_obj_arg /* world */
) {
    CUmodule mod = (CUmodule)mod_val;
    const char* name = lean_string_cstr(func_name);
    CUfunction func;
    CUDA_CHECK(cuModuleGetFunction(&func, mod, name), "cuModuleGetFunction");
    return lean_io_result_mk_ok(lean_box((size_t)func));
}

// Unload module
extern "C" lean_obj_res lean_hesper_cuda_module_unload(size_t mod_val, lean_obj_arg /* world */) {
    CUmodule mod = (CUmodule)mod_val;
    CUDA_CHECK(cuModuleUnload(mod), "cuModuleUnload");
    return lean_io_result_mk_ok(lean_box(0));
}

// ============================================================================
// Memory management
// ============================================================================

// Allocate device memory
extern "C" lean_obj_res lean_hesper_cuda_malloc(size_t size, lean_obj_arg /* world */) {
    CUdeviceptr ptr;
    CUDA_CHECK(cuMemAlloc(&ptr, size), "cuMemAlloc");
    return lean_io_result_mk_ok(lean_box((size_t)ptr));
}

// Free device memory
extern "C" lean_obj_res lean_hesper_cuda_free(size_t ptr_val, lean_obj_arg /* world */) {
    CUdeviceptr ptr = (CUdeviceptr)ptr_val;
    CUDA_CHECK(cuMemFree(ptr), "cuMemFree");
    return lean_io_result_mk_ok(lean_box(0));
}

// Host → Device copy
extern "C" lean_obj_res lean_hesper_cuda_memcpy_htod(
    size_t dst_val, b_lean_obj_arg src_bytes,
    size_t offset, size_t size, lean_obj_arg /* world */
) {
    CUdeviceptr dst = (CUdeviceptr)dst_val;
    const uint8_t* src = lean_sarray_cptr(src_bytes);
    CUDA_CHECK(cuMemcpyHtoD(dst + offset, src, size), "cuMemcpyHtoD");
    return lean_io_result_mk_ok(lean_box(0));
}

// Device → Host copy
extern "C" lean_obj_res lean_hesper_cuda_memcpy_dtoh(
    size_t src_val, size_t size, lean_obj_arg /* world */
) {
    CUdeviceptr src = (CUdeviceptr)src_val;
    lean_obj_res arr = lean_alloc_sarray(1, size, size);
    uint8_t* dst = lean_sarray_cptr(arr);
    CUDA_CHECK(cuMemcpyDtoH(dst, src, size), "cuMemcpyDtoH");
    return lean_io_result_mk_ok(arr);
}

// Zero-fill device memory
extern "C" lean_obj_res lean_hesper_cuda_memset(
    size_t ptr_val, size_t size, lean_obj_arg /* world */
) {
    CUdeviceptr ptr = (CUdeviceptr)ptr_val;
    CUDA_CHECK(cuMemsetD8(ptr, 0, size), "cuMemsetD8");
    return lean_io_result_mk_ok(lean_box(0));
}

// ============================================================================
// Kernel launch
// ============================================================================

// Launch kernel with array of device pointer arguments
extern "C" lean_obj_res lean_hesper_cuda_launch_kernel(
    size_t func_val,
    uint32_t gridX, uint32_t gridY, uint32_t gridZ,
    uint32_t blockX, uint32_t blockY, uint32_t blockZ,
    uint32_t shared_mem,
    b_lean_obj_arg arg_ptrs,  // Array of USize (device pointers)
    lean_obj_arg /* world */
) {
    CUfunction func = (CUfunction)func_val;
    size_t nargs = lean_array_size(arg_ptrs);

    // Build kernel argument array: each arg is a pointer to a CUdeviceptr
    std::vector<CUdeviceptr> ptrs(nargs);
    std::vector<void*> args(nargs);
    for (size_t i = 0; i < nargs; i++) {
        ptrs[i] = (CUdeviceptr)lean_unbox(lean_array_get_core(arg_ptrs, i));
        args[i] = &ptrs[i];
    }

    CUDA_CHECK(cuLaunchKernel(
        func,
        gridX, gridY, gridZ,
        blockX, blockY, blockZ,
        shared_mem, /*stream=*/0,
        args.data(), /*extra=*/nullptr
    ), "cuLaunchKernel");

    // Synchronize (simple mode — no stream management yet)
    CUDA_CHECK(cuCtxSynchronize(), "cuCtxSynchronize");

    return lean_io_result_mk_ok(lean_box(0));
}

// ============================================================================
// Device properties
// ============================================================================

// Get compute capability
extern "C" lean_obj_res lean_hesper_cuda_compute_capability(
    uint32_t dev, lean_obj_arg /* world */
) {
    int major, minor;
    CUDA_CHECK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev),
               "cuDeviceGetAttribute(major)");
    CUDA_CHECK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev),
               "cuDeviceGetAttribute(minor)");
    // Pack as major * 10 + minor (e.g., 89 for sm_89)
    return lean_io_result_mk_ok(lean_box(major * 10 + minor));
}

// Get total memory
extern "C" lean_obj_res lean_hesper_cuda_total_mem(
    uint32_t dev, lean_obj_arg /* world */
) {
    size_t totalMem;
    CUDA_CHECK(cuDeviceTotalMem(&totalMem, dev), "cuDeviceTotalMem");
    return lean_io_result_mk_ok(lean_box(totalMem));
}
