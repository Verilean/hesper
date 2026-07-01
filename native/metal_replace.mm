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
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <chrono>

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

// STEP 3: dispatch a CUSTOM Metal kernel (out[i] = in[i]*2) on the Dawn-backed MTLBuffers. Validates
// mechanism (A) end-to-end — a hand-written Metal kernel running on OUR data. Caller must sync the Dawn
// write of `in` first (e.g. a mapBufferRead), and read `out` back via Dawn afterwards; we waitUntilCompleted
// so the output is materialized before returning.
extern "C" lean_obj_res lean_hesper_metal_dispatch_mul2(
    b_lean_obj_arg device_obj, b_lean_obj_arg inBuf_obj, b_lean_obj_arg outBuf_obj,
    uint32_t n, lean_obj_res /* unit */) {
    wgpu::Device* device = mr_extract_device(device_obj);
    wgpu::Buffer* inB = mr_extract_buffer(inBuf_obj);
    wgpu::Buffer* outB = mr_extract_buffer(outBuf_obj);
    if (!device || !device->Get() || !inB || !inB->Get() || !outB || !outB->Get()) {
        return lean_io_result_mk_error(lean_mk_string("metal_dispatch: invalid args"));
    }
    id<MTLDevice> mtl = dawn::native::metal::GetMTLDevice(device->Get());
    id<MTLBuffer> mIn  = reinterpret_cast<dawn::native::metal::Buffer*>(inB->Get())->GetMTLBuffer();
    id<MTLBuffer> mOut = reinterpret_cast<dawn::native::metal::Buffer*>(outB->Get())->GetMTLBuffer();
    if (!mtl || !mIn || !mOut) {
        return lean_io_result_mk_error(lean_mk_string("metal_dispatch: null Metal objects"));
    }
    NSError* err = nil;
    NSString* src = @"#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void mul2(device const float* inp [[buffer(0)]], device float* outp [[buffer(1)]], "
        "uint i [[thread_position_in_grid]]) { outp[i] = inp[i] * 2.0f; }";
    id<MTLLibrary> lib = [mtl newLibraryWithSource:src options:nil error:&err];
    if (!lib) {
        return lean_io_result_mk_error(lean_mk_string([[NSString stringWithFormat:@"metal_dispatch: compile failed: %@", err] UTF8String]));
    }
    id<MTLFunction> fn = [lib newFunctionWithName:@"mul2"];
    id<MTLComputePipelineState> pso = [mtl newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        return lean_io_result_mk_error(lean_mk_string("metal_dispatch: pipeline failed"));
    }
    id<MTLCommandQueue> q = [mtl newCommandQueue];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:mIn  offset:0 atIndex:0];
    [enc setBuffer:mOut offset:0 atIndex:1];
    NSUInteger tg = MIN((NSUInteger)n, pso.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    return lean_io_result_mk_ok(lean_box(0));
}

// STEP 4 (measurement): the CEILING — Apple's tuned MPS f16 matmul (C = A · Bᵀ, matching our reg-matmul's
// A[M,K] × B[N,K]) on the MTLDevice behind our WGPUDevice. Batched `iters` encodes in one command buffer
// (like our batched WGSL bench). Returns "ms/call | GFLOPS | %of15.5peak" so we can diff vs the WGSL reg
// (harness benchShape) at the same shape — quantifies how far WGSL→Tint→Metal is from tuned Metal.
extern "C" lean_obj_res lean_hesper_mps_matmul_bench(
    b_lean_obj_arg device_obj, uint32_t M, uint32_t N, uint32_t K, uint32_t iters, lean_obj_res /* unit */) {
    wgpu::Device* device = mr_extract_device(device_obj);
    if (!device || !device->Get()) return lean_io_result_mk_error(lean_mk_string("mps: invalid device"));
    id<MTLDevice> mtl = dawn::native::metal::GetMTLDevice(device->Get());
    if (!mtl) return lean_io_result_mk_error(lean_mk_string("mps: null MTLDevice"));
    id<MTLCommandQueue> q = [mtl newCommandQueue];
    id<MTLBuffer> bA = [mtl newBufferWithLength:(NSUInteger)M*K*2 options:MTLResourceStorageModePrivate];
    id<MTLBuffer> bB = [mtl newBufferWithLength:(NSUInteger)N*K*2 options:MTLResourceStorageModePrivate];
    id<MTLBuffer> bC = [mtl newBufferWithLength:(NSUInteger)M*N*2 options:MTLResourceStorageModePrivate];
    MPSMatrixDescriptor* dA = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K rowBytes:(NSUInteger)K*2 dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor* dB = [MPSMatrixDescriptor matrixDescriptorWithRows:N columns:K rowBytes:(NSUInteger)K*2 dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor* dC = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N rowBytes:(NSUInteger)N*2 dataType:MPSDataTypeFloat16];
    MPSMatrix* mA = [[MPSMatrix alloc] initWithBuffer:bA descriptor:dA];
    MPSMatrix* mB = [[MPSMatrix alloc] initWithBuffer:bB descriptor:dB];
    MPSMatrix* mC = [[MPSMatrix alloc] initWithBuffer:bC descriptor:dC];
    MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc]
        initWithDevice:mtl transposeLeft:NO transposeRight:YES
        resultRows:M resultColumns:N interiorColumns:K alpha:1.0 beta:0.0];
    // warmup
    { id<MTLCommandBuffer> cb = [q commandBuffer]; [mm encodeToCommandBuffer:cb leftMatrix:mA rightMatrix:mB resultMatrix:mC]; [cb commit]; [cb waitUntilCompleted]; }
    auto t0 = std::chrono::high_resolution_clock::now();
    { id<MTLCommandBuffer> cb = [q commandBuffer];
      for (uint32_t i = 0; i < iters; i++) [mm encodeToCommandBuffer:cb leftMatrix:mA rightMatrix:mB resultMatrix:mC];
      [cb commit]; [cb waitUntilCompleted]; }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / (double)iters;
    double gflops = 2.0 * (double)M * N * K / (ms / 1000.0) / 1e9;
    double pct = gflops * 1e9 / 15.5e12 * 100.0;
    char b[256];
    snprintf(b, sizeof(b), "%.4f ms/call | %.1f GFLOPS | %.1f%% of 15.5T peak", ms, gflops, pct);
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
