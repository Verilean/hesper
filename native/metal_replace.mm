// metal_replace.mm — Objective-C++ bridge to Dawn's underlying Metal objects, for swapping llama.cpp's
// tuned Metal kernels in for our hottest WGSL-generated kernels. See METAL_REPLACER_INTEGRATION.md.
//
// STEP 1 (this file, PoC): prove the interop is live — get the MTLDevice behind the WGPUDevice and return
// its name to Lean. Dawn exposes GetMTLDevice(WGPUDevice) publicly (dawn/native/MetalBackend.h).
//
// Compiled as Objective-C++ (.mm). Linked into hesper_native alongside bridge.cpp.

#include <lean/lean.h>
#include <webgpu/webgpu.h>
#include <webgpu/webgpu_cpp.h>
#include <dawn/native/MetalBackend.h>
#import <Metal/Metal.h>

// Same extraction as bridge.cpp's EXTRACT_DEVICE_PTR: the device Lean struct holds the wgpu::Device* as its
// external data at ctor field 0.
static inline wgpu::Device* mr_extract_device(b_lean_obj_arg device_obj) {
    return static_cast<wgpu::Device*>(lean_get_external_data(lean_ctor_get(device_obj, 0)));
}
static inline wgpu::Buffer* mr_extract_buffer(b_lean_obj_arg buffer_obj) {
    return static_cast<wgpu::Buffer*>(lean_get_external_data(lean_ctor_get(buffer_obj, 0)));
}

// MINIMAL declaration of dawn::native::metal::Buffer just to call GetMTLBuffer() — avoids pulling in the
// deep internal headers (BufferMTL.h → Buffer.h → the whole Dawn object model). GetMTLBuffer() is an
// out-of-line non-virtual accessor (BufferMTL.mm:182), so the mangled symbol resolves from the linked
// Dawn static lib, and `this` (the reinterpret_cast'd WGPUBuffer C-handle = the concrete metal::Buffer*)
// gives the right member offsets. Dawn's C handles ARE the internal object pointers (BufferBase-first).
namespace dawn::native::metal {
class Buffer {
 public:
    id<MTLBuffer> GetMTLBuffer() const;
};
}  // namespace dawn::native::metal

// STEP 2: bridge our Dawn buffer to its underlying MTLBuffer. Prove it by reporting the MTLBuffer's length
// (must match our allocation) + its contents (readable on unified memory).
extern "C" lean_obj_res lean_hesper_mtl_buffer_probe(b_lean_obj_arg buffer_obj, lean_obj_res /* unit */) {
    wgpu::Buffer* buf = mr_extract_buffer(buffer_obj);
    if (!buf || !buf->Get()) {
        return lean_io_result_mk_error(lean_mk_string("metal_replace: buffer is invalid"));
    }
    auto* mb = reinterpret_cast<dawn::native::metal::Buffer*>(buf->Get());
    id<MTLBuffer> mtl = mb->GetMTLBuffer();
    if (!mtl) {
        return lean_io_result_mk_ok(lean_mk_string("(null MTLBuffer)"));
    }
    NSUInteger len = [mtl length];
    NSUInteger mode = (NSUInteger)[mtl storageMode];
    void* c = [mtl contents];
    char b[256];
    if (c) {
        float* f = (float*)c;
        snprintf(b, sizeof(b), "MTLBuffer OK: length=%lu storageMode=%lu contents[0..2]=%.3f,%.3f,%.3f",
                 (unsigned long)len, (unsigned long)mode, f[0], f[1], f[2]);
    } else {
        snprintf(b, sizeof(b), "MTLBuffer OK: length=%lu storageMode=%lu contents=nil(private)",
                 (unsigned long)len, (unsigned long)mode);
    }
    return lean_io_result_mk_ok(lean_mk_string(b));
}

extern "C" lean_obj_res lean_hesper_mtl_device_name(b_lean_obj_arg device_obj, lean_obj_res /* unit */) {
    wgpu::Device* device = mr_extract_device(device_obj);
    if (!device || !device->Get()) {
        return lean_io_result_mk_error(lean_mk_string("metal_replace: device is invalid"));
    }
    id<MTLDevice> mtl = dawn::native::metal::GetMTLDevice(device->Get());
    if (!mtl) {
        return lean_io_result_mk_ok(lean_mk_string("(null MTLDevice)"));
    }
    // Prove we can reach live Metal properties through the Dawn handle.
    const char* name = [[mtl name] UTF8String];
    char buf[512];
    snprintf(buf, sizeof(buf), "%s | maxThreadsPerThreadgroup=%lu | hasUnifiedMemory=%d",
             name ? name : "(no name)",
             (unsigned long)[mtl maxThreadsPerThreadgroup].width,
             (int)[mtl hasUnifiedMemory]);
    return lean_io_result_mk_ok(lean_mk_string(buf));
}
