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
#include <cstring>

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

// ===========================================================================================
// MSL 1-kernel PoC (macOS DEBUG/REFERENCE, like the rest of metal_replacer — NOT production):
// hand-written native-Metal port of q4kMatmulGroupedRegIndexedKernel (the deployed MoE gate/up:
// Q4_K in-kernel dequant + simdgroup-matrix 32x32 tiles + indexed A-load + sentinel-tile skip +
// ragged 8-row sub-tile skip). SAME algorithm as the WGSL kernel — measures what native codegen
// (occupancy control, no Tint) buys on identical data. Timed with MTLCommandBuffer GPU timestamps.
// ===========================================================================================
static const char* kQ4kMslTemplate = R"MSL(
#include <metal_stdlib>
using namespace metal;

constant uint M = @M@u;
constant uint N = @N@u;
constant uint K = @K@u;
constant uint NEXP = @NEXP@u;
constant uint SRCROWS = @SRCROWS@u;
constant uint RSU = (K/256u)*36u;
constant uint NUMBLK = K/256u;

inline uint sm_byte(uint p, uint s0, uint s1, uint s2) {
  return (p < 4u) ? ((s0 >> (p*8u)) & 0xFFu)
       : (p < 8u) ? ((s1 >> ((p-4u)*8u)) & 0xFFu)
                  : ((s2 >> ((p-8u)*8u)) & 0xFFu);
}
inline float2 scale_min(uint j, uint s0, uint s1, uint s2) {
  if (j < 4u) {
    return float2(float(sm_byte(j, s0,s1,s2) & 63u), float(sm_byte(j+4u, s0,s1,s2) & 63u));
  }
  uint sc = (sm_byte(j+4u, s0,s1,s2) & 0xFu) | ((sm_byte(j-4u, s0,s1,s2) >> 6u) << 4u);
  uint m  = (sm_byte(j+4u, s0,s1,s2) >> 4u)  | ((sm_byte(j,    s0,s1,s2) >> 6u) << 4u);
  return float2(float(sc), float(m));
}

kernel void q4k_grouped_reg_indexed(
    device const float* src [[buffer(0)]],
    device const uint*  idx [[buffer(1)]],
    device const uint*  b   [[buffer(2)]],
    device float*       c   [[buffer(3)]],
    device const uint*  te  [[buffer(4)]],
    device const uint*  trs [[buffer(5)]],
    uint2 wid [[threadgroup_position_in_grid]],
    uint  tid [[thread_index_in_threadgroup]])
{
  threadgroup half  shared_A[32*32];
  threadgroup half  shared_B[32*32];
  threadgroup float shared_dq[32*18];

  simdgroup_half8x8  Ax0, Ax1, Bx0, Bx1;
  simdgroup_float8x8 Cx0 = make_filled_simdgroup_matrix<float,8,8>(0.0f);
  simdgroup_float8x8 Cx1 = make_filled_simdgroup_matrix<float,8,8>(0.0f);
  simdgroup_float8x8 Cx2 = make_filled_simdgroup_matrix<float,8,8>(0.0f);
  simdgroup_float8x8 Cx3 = make_filled_simdgroup_matrix<float,8,8>(0.0f);

  const uint rowBase = wid.y * 32u, colBase = wid.x * 32u;
  const uint sgitg = tid / 32u;
  const uint sgRow = sgitg % 2u, sgCol = sgitg / 2u;
  const uint mOff = sgRow*16u, nOff = sgCol*16u;
  const uint teRaw = te[wid.y];
  const bool isActive = teRaw < NEXP;
  const uint e = isActive ? teRaw : (NEXP - 1u);
  const uint wro = e * N;
  const uint trRaw = trs[wid.y];
  const bool frag0 = isActive && (sgRow*16u < trRaw);
  const bool frag1 = isActive && (sgRow*16u + 8u < trRaw);
  const uint tr8 = ((trRaw + 7u)/8u)*8u;

  for (uint blockIdx = 0u; blockIdx < NUMBLK; blockIdx++) {
    if (tid < 32u && isActive) {
      uint row = wro + colBase + tid;
      uint bb = row*RSU + blockIdx*36u;
      uint dm = b[bb];
      float d    = float(as_type<half>(ushort(dm & 0xFFFFu)));
      float dmin = float(as_type<half>(ushort(dm >> 16u)));
      uint s0 = b[bb+1u], s1 = b[bb+2u], s2 = b[bb+3u];
      uint base = tid*18u;
      shared_dq[base] = d; shared_dq[base+1u] = dmin;
      for (uint j = 0u; j < 8u; j++) {
        float2 sm = scale_min(j, s0, s1, s2);
        shared_dq[base+2u+j] = sm.x; shared_dq[base+10u+j] = sm.y;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint jSub = 0u; jSub < 8u; jSub++) {
      uint kBase = blockIdx*256u + jSub*32u;
      uint chunk = jSub/2u; bool isHigh = (jSub % 2u) == 1u;
      if (isActive) {
        for (uint s = 0u; s < 8u; s++) {
          uint flat = tid + s*128u;
          uint m = flat/32u, k = flat % 32u;
          if (m < tr8) {
            uint tok = idx[rowBase + m];
            float x = src[tok*K + kBase + k];
            uint blk = (m/8u)*4u + k/8u;
            uint within = (m % 8u)*8u + (k % 8u);
            shared_A[blk*64u + within] = half(x);
          }
        }
      }
      if (isActive) {
        for (uint s = 0u; s < 4u; s++) {
          uint u = tid + s*128u;
          uint n = u/16u, kpair = u % 16u;
          uint row = wro + colBase + n;
          uint bbase = row*RSU + blockIdx*36u;
          uint dqb = n*18u;
          float d = shared_dq[dqb], dmin = shared_dq[dqb+1u];
          float sc = shared_dq[dqb+2u+jSub], mv = shared_dq[dqb+10u+jSub];
          uint k0 = kpair*2u;
          uint qsByteIdx0 = chunk*32u + k0;
          uint qsU32a = b[bbase + 4u + qsByteIdx0/4u];
          uint qsByte0 = (qsU32a >> ((qsByteIdx0 % 4u)*8u)) & 0xFFu;
          uint qsByteIdx1 = qsByteIdx0 + 1u;
          uint qsU32b = b[bbase + 4u + qsByteIdx1/4u];
          uint qsByte1 = (qsU32b >> ((qsByteIdx1 % 4u)*8u)) & 0xFFu;
          float q0 = float(isHigh ? (qsByte0 >> 4u) : (qsByte0 & 0xFu));
          float q1 = float(isHigh ? (qsByte1 >> 4u) : (qsByte1 & 0xFu));
          float y0 = d*sc*q0 - dmin*mv;
          float y1 = d*sc*q1 - dmin*mv;
          uint ktile = k0/8u, kr = k0 % 8u;
          uint blkB = ktile*4u + n/8u;
          uint baseB = blkB*64u + (n % 8u);
          shared_B[baseB + kr*8u]        = half(y0);
          shared_B[baseB + (kr+1u)*8u]   = half(y1);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint k8 = 0u; k8 < 4u; k8++) {
        if (frag0) {
          uint blkA0 = (sgRow*2u)*4u + k8;
          simdgroup_load(Ax0, &shared_A[blkA0*64u], 8u);
          uint blkB0 = k8*4u + (sgCol*2u);
          simdgroup_load(Bx0, &shared_B[blkB0*64u], 8u);
          uint blkB1 = k8*4u + (sgCol*2u + 1u);
          simdgroup_load(Bx1, &shared_B[blkB1*64u], 8u);
          simdgroup_multiply_accumulate(Cx0, Ax0, Bx0, Cx0);
          simdgroup_multiply_accumulate(Cx1, Ax0, Bx1, Cx1);
        }
        if (frag1) {
          uint blkA1 = (sgRow*2u + 1u)*4u + k8;
          simdgroup_load(Ax1, &shared_A[blkA1*64u], 8u);
          simdgroup_multiply_accumulate(Cx2, Ax1, Bx0, Cx2);
          simdgroup_multiply_accumulate(Cx3, Ax1, Bx1, Cx3);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  if (frag0) {
    simdgroup_store(Cx0, &c[(rowBase+mOff)*N + colBase+nOff], N);
    simdgroup_store(Cx1, &c[(rowBase+mOff)*N + colBase+nOff+8u], N);
  }
  if (frag1) {
    simdgroup_store(Cx2, &c[(rowBase+mOff+8u)*N + colBase+nOff], N);
    simdgroup_store(Cx3, &c[(rowBase+mOff+8u)*N + colBase+nOff+8u], N);
  }
}
)MSL";

// Bench the hand-written MSL kernel on the given Dawn buffers. Returns "msPerIter" as a string.
// Caller must have synced all input writes (a prior Dawn readback/endBatch) and written `c` once
// (Dawn lazy-clear gotcha) before calling. iters dispatches go into ONE command buffer with a
// buffer-scope memory barrier between them (serialized like Dawn's inter-pass barriers); time =
// (GPUEndTime-GPUStartTime)/iters from the command buffer's GPU timestamps.
extern "C" lean_obj_res lean_hesper_msl_q4k_bench(
    b_lean_obj_arg device_obj,
    b_lean_obj_arg src_obj, b_lean_obj_arg idx_obj, b_lean_obj_arg b_obj,
    b_lean_obj_arg c_obj, b_lean_obj_arg te_obj, b_lean_obj_arg tr_obj,
    uint32_t Mv, uint32_t Nv, uint32_t Kv, uint32_t nExpert, uint32_t srcRows,
    uint32_t iters, lean_obj_res /* unit */) {
    wgpu::Device* device = mr_extract_device(device_obj);
    if (!device || !device->Get()) return lean_io_result_mk_error(lean_mk_string("msl_q4k: invalid device"));
    id<MTLDevice> mtl = dawn::native::metal::GetMTLDevice(device->Get());
    wgpu::Buffer* bufs[6] = { mr_extract_buffer(src_obj), mr_extract_buffer(idx_obj),
                              mr_extract_buffer(b_obj),   mr_extract_buffer(c_obj),
                              mr_extract_buffer(te_obj),  mr_extract_buffer(tr_obj) };
    id<MTLBuffer> mb[6];
    for (int i = 0; i < 6; i++) {
        if (!bufs[i] || !bufs[i]->Get()) return lean_io_result_mk_error(lean_mk_string("msl_q4k: invalid buffer"));
        mb[i] = reinterpret_cast<dawn::native::metal::Buffer*>(bufs[i]->Get())->GetMTLBuffer();
        if (!mb[i]) return lean_io_result_mk_error(lean_mk_string("msl_q4k: null MTLBuffer"));
    }
    std::string msl(kQ4kMslTemplate);
    auto subst = [&](const char* tok, uint32_t v) {
        std::string t(tok); std::string r = std::to_string(v);
        size_t pos;
        while ((pos = msl.find(t)) != std::string::npos) msl.replace(pos, t.size(), r);
    };
    subst("@M@", Mv); subst("@N@", Nv); subst("@K@", Kv);
    subst("@NEXP@", nExpert); subst("@SRCROWS@", srcRows);
    // MSL_PROBE: timing diagnostic — stub the Q4_K qs global reads + bit-extract in the B-load so
    // shared_B is filled from the cached scale only. Removes the per-jSub qs global traffic + dequant
    // math but keeps the loop structure, A-load, and simdgroup compute. probe<<real ⇒ dequant-bound
    // (headroom via shared-qs hoist / async load); probe≈real ⇒ compute/A-load-bound (near ceiling).
    if (getenv("MSL_PROBE")) {
        const std::string needle =
            "          uint qsByteIdx0 = chunk*32u + k0;";
        const std::string upto =
            "          float q1 = float(isHigh ? (qsByte1 >> 4u) : (qsByte1 & 0xFu));";
        size_t a = msl.find(needle), z = msl.find(upto);
        if (a != std::string::npos && z != std::string::npos)
            msl.replace(a, (z + upto.size()) - a, "          float q0 = 1.0f, q1 = 1.0f;");
    }
    NSError* err = nil;
    MTLCompileOptions* opts = [MTLCompileOptions new];
    id<MTLLibrary> lib = [mtl newLibraryWithSource:[NSString stringWithUTF8String:msl.c_str()] options:opts error:&err];
    if (!lib) {
        return lean_io_result_mk_error(lean_mk_string(
            [[NSString stringWithFormat:@"msl_q4k compile failed: %@", err] UTF8String]));
    }
    id<MTLFunction> fn = [lib newFunctionWithName:@"q4k_grouped_reg_indexed"];
    id<MTLComputePipelineState> pso = [mtl newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) return lean_io_result_mk_error(lean_mk_string("msl_q4k: pipeline failed"));
    id<MTLCommandQueue> q = [mtl newCommandQueue];
    MTLSize grid = MTLSizeMake((Nv + 31) / 32, (Mv + 31) / 32, 1);
    MTLSize tg = MTLSizeMake(128, 1, 1);
    auto encodeOnce = [&](id<MTLComputeCommandEncoder> enc) {
        [enc setComputePipelineState:pso];
        for (int i = 0; i < 6; i++) [enc setBuffer:mb[i] offset:0 atIndex:i];
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    };
    { // warmup (also materializes the output for the golden readback)
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        encodeOnce(enc);
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            return lean_io_result_mk_error(lean_mk_string("msl_q4k: warmup GPU error"));
        }
    }
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    for (uint32_t i = 0; i < iters; i++) {
        encodeOnce(enc);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];   // serialize like Dawn's inter-pass barriers
    }
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusError) {
        return lean_io_result_mk_error(lean_mk_string("msl_q4k: timed GPU error"));
    }
    double ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0 / (double)iters;
    char out[64];
    snprintf(out, sizeof(out), "%.4f", ms);
    return lean_io_result_mk_ok(lean_mk_string(out));
}

// HOT-PATH dispatch of the MSL q4k kernel (production path for DG_MSL, ~1.61× vs WGSL/Tint).
// PSO compiled ONCE (single-slot cache — all 30 layers share dims) on a persistent queue.
// Encode + commit, NO CPU wait. ORDERING CONTRACT (why this is safe without waits): Dawn's Metal
// buffers are device-allocated with default options = MTLResourceHazardTrackingModeTracked, so
// Metal serializes command buffers touching the same buffers in COMMIT order. Caller must
// flushBatch (commit Dawn's producer encoder) BEFORE this call; Dawn work recorded after runs in
// a later-committed cb → ordered after ours by the same hazard tracking.
static id<MTLComputePipelineState> g_q4kPso = nil;
static id<MTLCommandQueue> g_mslQueue = nil;
static uint32_t g_q4kKey[5] = {0,0,0,0,0};

extern "C" lean_obj_res lean_hesper_msl_q4k_dispatch(
    b_lean_obj_arg device_obj,
    b_lean_obj_arg src_obj, b_lean_obj_arg idx_obj, b_lean_obj_arg b_obj,
    b_lean_obj_arg c_obj, b_lean_obj_arg te_obj, b_lean_obj_arg tr_obj,
    uint32_t Mv, uint32_t Nv, uint32_t Kv, uint32_t nExpert, uint32_t srcRows,
    lean_obj_res /* unit */) {
    wgpu::Device* device = mr_extract_device(device_obj);
    if (!device || !device->Get()) return lean_io_result_mk_error(lean_mk_string("msl_q4k_dispatch: invalid device"));
    id<MTLDevice> mtl = dawn::native::metal::GetMTLDevice(device->Get());
    wgpu::Buffer* bufs[6] = { mr_extract_buffer(src_obj), mr_extract_buffer(idx_obj),
                              mr_extract_buffer(b_obj),   mr_extract_buffer(c_obj),
                              mr_extract_buffer(te_obj),  mr_extract_buffer(tr_obj) };
    id<MTLBuffer> mb[6];
    for (int i = 0; i < 6; i++) {
        if (!bufs[i] || !bufs[i]->Get()) return lean_io_result_mk_error(lean_mk_string("msl_q4k_dispatch: invalid buffer"));
        mb[i] = reinterpret_cast<dawn::native::metal::Buffer*>(bufs[i]->Get())->GetMTLBuffer();
        if (!mb[i]) return lean_io_result_mk_error(lean_mk_string("msl_q4k_dispatch: null MTLBuffer"));
    }
    uint32_t key[5] = {Mv, Nv, Kv, nExpert, srcRows};
    if (!g_q4kPso || memcmp(key, g_q4kKey, sizeof(key)) != 0) {
        std::string msl(kQ4kMslTemplate);
        auto subst = [&](const char* tok, uint32_t v) {
            std::string t(tok); std::string r = std::to_string(v);
            size_t pos;
            while ((pos = msl.find(t)) != std::string::npos) msl.replace(pos, t.size(), r);
        };
        subst("@M@", Mv); subst("@N@", Nv); subst("@K@", Kv);
        subst("@NEXP@", nExpert); subst("@SRCROWS@", srcRows);
        NSError* err = nil;
        MTLCompileOptions* opts = [MTLCompileOptions new];
        id<MTLLibrary> lib = [mtl newLibraryWithSource:[NSString stringWithUTF8String:msl.c_str()] options:opts error:&err];
        if (!lib) return lean_io_result_mk_error(lean_mk_string(
            [[NSString stringWithFormat:@"msl_q4k_dispatch compile failed: %@", err] UTF8String]));
        id<MTLFunction> fn = [lib newFunctionWithName:@"q4k_grouped_reg_indexed"];
        id<MTLComputePipelineState> pso = [mtl newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) return lean_io_result_mk_error(lean_mk_string("msl_q4k_dispatch: pipeline failed"));
        g_q4kPso = pso;
        memcpy(g_q4kKey, key, sizeof(key));
    }
    if (!g_mslQueue) g_mslQueue = [mtl newCommandQueue];
    // Hazard tracking orders execution by COMMIT order — but Dawn's submit can lag the wgpu call.
    // WaitForCommandsToBeScheduled blocks until Dawn's producer command buffers are committed and
    // scheduled (NOT completed — µs-class), making our commit provably later. Empirically required:
    // without it the pipeline races (NOBATCH+MSL converges fine; batched+MSL garbaged).
    dawn::native::metal::WaitForCommandsToBeScheduled(device->Get());
    id<MTLCommandBuffer> cb = [g_mslQueue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:g_q4kPso];
    for (int i = 0; i < 6; i++) [enc setBuffer:mb[i] offset:0 atIndex:i];
    [enc dispatchThreadgroups:MTLSizeMake((Nv + 31) / 32, (Mv + 31) / 32, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc endEncoding];
    [cb commit];   // no wait — hazard tracking orders vs Dawn's committed work
    return lean_io_result_mk_ok(lean_box(0));
}

// ===========================================================================================
// MSL port of q8MatmulGroupedRegIndexedScatterKernel — the deployed MoE DOWN (Q8_0 weights,
// 34B/block = f16 scale + 32 int8; A = the grouped geglu output read directly; ragged sub-tile
// skip + sentinel skip; C scatter-on-store through pos/slot into dst[slot,NTOK,N]). SAME
// algorithm as the WGSL kernel; same ordering contract as the q4k dispatch above.
// ===========================================================================================
static const char* kQ8DownMslTemplate = R"MSL(
#include <metal_stdlib>
using namespace metal;

constant uint M = @M@u;
constant uint N = @N@u;
constant uint K = @K@u;
constant uint NEXP = @NEXP@u;
constant uint NUSED = @NUSED@u;
constant uint NTOK = @NTOK@u;
constant uint RSB = (K/32u)*34u;   // row stride in BYTES (34B per 32-K Q8_0 block, NOT u32-aligned)
constant uint NUMBLK = K/32u;

inline uint rd_byte(device const uint* b, uint bo) {
  return (b[bo >> 2u] >> ((bo & 3u)*8u)) & 0xFFu;
}

kernel void q8_down_indexed_scatter(
    device const float* a    [[buffer(0)]],
    device const uint*  b    [[buffer(1)]],
    device const uint*  te   [[buffer(2)]],
    device const uint*  trs  [[buffer(3)]],
    device const uint*  pos  [[buffer(4)]],
    device const uint*  slot [[buffer(5)]],
    device float*       dst  [[buffer(6)]],
    uint2 wid [[threadgroup_position_in_grid]],
    uint  tid [[thread_index_in_threadgroup]])
{
  threadgroup half  shared_A[32*32];
  threadgroup half  shared_B[32*32];
  threadgroup float shared_d[32];
  threadgroup float shared_C[32*32];

  simdgroup_half8x8  Ax0, Ax1, Bx0, Bx1;
  simdgroup_float8x8 Cx0 = make_filled_simdgroup_matrix<float,8,8>(0.0f);
  simdgroup_float8x8 Cx1 = make_filled_simdgroup_matrix<float,8,8>(0.0f);
  simdgroup_float8x8 Cx2 = make_filled_simdgroup_matrix<float,8,8>(0.0f);
  simdgroup_float8x8 Cx3 = make_filled_simdgroup_matrix<float,8,8>(0.0f);

  const uint rowBase = wid.y * 32u, colBase = wid.x * 32u;
  const uint sgitg = tid / 32u;
  const uint sgRow = sgitg % 2u, sgCol = sgitg / 2u;
  const uint mOff = sgRow*16u, nOff = sgCol*16u;
  const uint teRaw = te[wid.y];
  const bool isActive = teRaw < NEXP;
  const uint e = isActive ? teRaw : (NEXP - 1u);
  const uint wro = e * N;
  const uint trRaw = trs[wid.y];
  const bool frag0 = isActive && (sgRow*16u < trRaw);
  const bool frag1 = isActive && (sgRow*16u + 8u < trRaw);
  const uint tr8 = ((trRaw + 7u)/8u)*8u;

  for (uint blockIdx = 0u; blockIdx < NUMBLK; blockIdx++) {
    uint blkByte = blockIdx*34u;
    if (tid < 32u && isActive) {
      uint row = wro + colBase + tid;
      uint bb = row*RSB + blkByte;
      uint lo = rd_byte(b, bb), hi = rd_byte(b, bb + 1u);
      shared_d[tid] = float(as_type<half>(ushort(lo | (hi << 8u))));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (isActive) {
      for (uint s = 0u; s < 8u; s++) {
        uint flat = tid + s*128u;
        uint m = flat/32u, k = flat % 32u;
        if (m < tr8) {
          float x = a[(rowBase + m)*K + blockIdx*32u + k];
          uint blk = (m/8u)*4u + k/8u;
          uint within = (m % 8u)*8u + (k % 8u);
          shared_A[blk*64u + within] = half(x);
        }
      }
      for (uint s = 0u; s < 4u; s++) {
        uint u = tid + s*128u;
        uint n = u/16u, kpair = u % 16u;
        uint row = wro + colBase + n;
        uint bb = row*RSB + blkByte;
        float d = shared_d[n];
        uint k0 = kpair*2u;
        uint q0b = rd_byte(b, bb + 2u + k0);
        uint q1b = rd_byte(b, bb + 3u + k0);
        float y0 = d * (float(q0b) - 256.0f*float(q0b >> 7u));
        float y1 = d * (float(q1b) - 256.0f*float(q1b >> 7u));
        uint ktile = k0/8u, kr = k0 % 8u;
        uint blkB = ktile*4u + n/8u;
        uint baseB = blkB*64u + (n % 8u);
        shared_B[baseB + kr*8u]      = half(y0);
        shared_B[baseB + (kr+1u)*8u] = half(y1);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k8 = 0u; k8 < 4u; k8++) {
      if (frag0) {
        uint blkA0 = (sgRow*2u)*4u + k8;
        simdgroup_load(Ax0, &shared_A[blkA0*64u], 8u);
        uint blkB0 = k8*4u + (sgCol*2u);
        simdgroup_load(Bx0, &shared_B[blkB0*64u], 8u);
        uint blkB1 = k8*4u + (sgCol*2u + 1u);
        simdgroup_load(Bx1, &shared_B[blkB1*64u], 8u);
        simdgroup_multiply_accumulate(Cx0, Ax0, Bx0, Cx0);
        simdgroup_multiply_accumulate(Cx1, Ax0, Bx1, Cx1);
      }
      if (frag1) {
        uint blkA1 = (sgRow*2u + 1u)*4u + k8;
        simdgroup_load(Ax1, &shared_A[blkA1*64u], 8u);
        simdgroup_multiply_accumulate(Cx2, Ax1, Bx0, Cx2);
        simdgroup_multiply_accumulate(Cx3, Ax1, Bx1, Cx3);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (frag0) {
    simdgroup_store(Cx0, &shared_C[mOff*32u + nOff], 32u);
    simdgroup_store(Cx1, &shared_C[mOff*32u + nOff + 8u], 32u);
  }
  if (frag1) {
    simdgroup_store(Cx2, &shared_C[(mOff+8u)*32u + nOff], 32u);
    simdgroup_store(Cx3, &shared_C[(mOff+8u)*32u + nOff + 8u], 32u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (isActive) {
    for (uint s = 0u; s < 8u; s++) {
      uint flat = tid + s*128u;
      uint r = flat/32u, c = flat % 32u;
      uint rowGlobal = rowBase + r;
      uint slotR = slot[rowGlobal];
      if (slotR < NUSED) {
        uint posR = pos[rowGlobal];
        dst[slotR*(NTOK*N) + posR*N + colBase + c] = shared_C[r*32u + c];
      }
    }
  }
}
)MSL";

static id<MTLComputePipelineState> g_q8dPso = nil;
static uint32_t g_q8dKey[6] = {0,0,0,0,0,0};

extern "C" lean_obj_res lean_hesper_msl_q8down_dispatch(
    b_lean_obj_arg device_obj,
    b_lean_obj_arg a_obj, b_lean_obj_arg b_obj, b_lean_obj_arg te_obj, b_lean_obj_arg tr_obj,
    b_lean_obj_arg pos_obj, b_lean_obj_arg slot_obj, b_lean_obj_arg dst_obj,
    uint32_t Mv, uint32_t Nv, uint32_t Kv, uint32_t nExpert, uint32_t nUsed, uint32_t nTok,
    lean_obj_res /* unit */) {
    wgpu::Device* device = mr_extract_device(device_obj);
    if (!device || !device->Get()) return lean_io_result_mk_error(lean_mk_string("msl_q8down: invalid device"));
    id<MTLDevice> mtl = dawn::native::metal::GetMTLDevice(device->Get());
    wgpu::Buffer* bufs[7] = { mr_extract_buffer(a_obj),   mr_extract_buffer(b_obj),
                              mr_extract_buffer(te_obj),  mr_extract_buffer(tr_obj),
                              mr_extract_buffer(pos_obj), mr_extract_buffer(slot_obj),
                              mr_extract_buffer(dst_obj) };
    id<MTLBuffer> mb[7];
    for (int i = 0; i < 7; i++) {
        if (!bufs[i] || !bufs[i]->Get()) return lean_io_result_mk_error(lean_mk_string("msl_q8down: invalid buffer"));
        mb[i] = reinterpret_cast<dawn::native::metal::Buffer*>(bufs[i]->Get())->GetMTLBuffer();
        if (!mb[i]) return lean_io_result_mk_error(lean_mk_string("msl_q8down: null MTLBuffer"));
    }
    uint32_t key[6] = {Mv, Nv, Kv, nExpert, nUsed, nTok};
    if (!g_q8dPso || memcmp(key, g_q8dKey, sizeof(key)) != 0) {
        std::string msl(kQ8DownMslTemplate);
        auto subst = [&](const char* tok, uint32_t v) {
            std::string t(tok); std::string r = std::to_string(v);
            size_t p;
            while ((p = msl.find(t)) != std::string::npos) msl.replace(p, t.size(), r);
        };
        subst("@M@", Mv); subst("@N@", Nv); subst("@K@", Kv);
        subst("@NEXP@", nExpert); subst("@NUSED@", nUsed); subst("@NTOK@", nTok);
        NSError* err = nil;
        MTLCompileOptions* opts = [MTLCompileOptions new];
        id<MTLLibrary> lib = [mtl newLibraryWithSource:[NSString stringWithUTF8String:msl.c_str()] options:opts error:&err];
        if (!lib) return lean_io_result_mk_error(lean_mk_string(
            [[NSString stringWithFormat:@"msl_q8down compile failed: %@", err] UTF8String]));
        id<MTLFunction> fn = [lib newFunctionWithName:@"q8_down_indexed_scatter"];
        id<MTLComputePipelineState> pso = [mtl newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) return lean_io_result_mk_error(lean_mk_string("msl_q8down: pipeline failed"));
        g_q8dPso = pso;
        memcpy(g_q8dKey, key, sizeof(key));
    }
    if (!g_mslQueue) g_mslQueue = [mtl newCommandQueue];
    dawn::native::metal::WaitForCommandsToBeScheduled(device->Get());
    id<MTLCommandBuffer> cb = [g_mslQueue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:g_q8dPso];
    for (int i = 0; i < 7; i++) [enc setBuffer:mb[i] offset:0 atIndex:i];
    [enc dispatchThreadgroups:MTLSizeMake((Nv + 31) / 32, (Mv + 31) / 32, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc endEncoding];
    [cb commit];   // no wait — hazard tracking orders vs Dawn's committed work
    return lean_io_result_mk_ok(lean_box(0));
}


// ============ Q5_0 MoE down (MSL) — the 16/30 Q5_0 layers were on the WGSL warp fallback ============
// Same structure as kQ8DownMslTemplate; only the B dequant differs. Q5_0 block = 22 bytes:
// d(f16, bytes 0-1) + qh(u32 LE, bytes 2-5) + qs(16 bytes, 6-21). weight k (0..31):
// nibble = k<16 ? qs[k]&0xF : qs[k-16]>>4; hbit=(qh>>k)&1; x = (float(nibble|hbit<<4) - 16) * d.
static const char* kQ5DownMslTemplate = R"MSL(
#include <metal_stdlib>
using namespace metal;

constant uint M = @M@u;
constant uint N = @N@u;
constant uint K = @K@u;
constant uint NEXP = @NEXP@u;
constant uint NUSED = @NUSED@u;
constant uint NTOK = @NTOK@u;
constant uint RSB = (K/32u)*22u;   // row stride in BYTES (22B per 32-K Q5_0 block)
constant uint NUMBLK = K/32u;

inline uint rd_byte(device const uint* b, uint bo) {
  return (b[bo >> 2u] >> ((bo & 3u)*8u)) & 0xFFu;
}

kernel void q5_down_indexed_scatter(
    device const float* a    [[buffer(0)]],
    device const uint*  b    [[buffer(1)]],
    device const uint*  te   [[buffer(2)]],
    device const uint*  trs  [[buffer(3)]],
    device const uint*  pos  [[buffer(4)]],
    device const uint*  slot [[buffer(5)]],
    device float*       dst  [[buffer(6)]],
    uint2 wid [[threadgroup_position_in_grid]],
    uint  tid [[thread_index_in_threadgroup]])
{
  threadgroup half  shared_A[32*32];
  threadgroup half  shared_B[32*32];
  threadgroup float shared_d[32];
  threadgroup uint  shared_qh[32];
  threadgroup float shared_C[32*32];

  simdgroup_half8x8  Ax0, Ax1, Bx0, Bx1;
  simdgroup_float8x8 Cx0 = make_filled_simdgroup_matrix<float,8,8>(0.0f);
  simdgroup_float8x8 Cx1 = make_filled_simdgroup_matrix<float,8,8>(0.0f);
  simdgroup_float8x8 Cx2 = make_filled_simdgroup_matrix<float,8,8>(0.0f);
  simdgroup_float8x8 Cx3 = make_filled_simdgroup_matrix<float,8,8>(0.0f);

  const uint rowBase = wid.y * 32u, colBase = wid.x * 32u;
  const uint sgitg = tid / 32u;
  const uint sgRow = sgitg % 2u, sgCol = sgitg / 2u;
  const uint mOff = sgRow*16u, nOff = sgCol*16u;
  const uint teRaw = te[wid.y];
  const bool isActive = teRaw < NEXP;
  const uint e = isActive ? teRaw : (NEXP - 1u);
  const uint wro = e * N;
  const uint trRaw = trs[wid.y];
  const bool frag0 = isActive && (sgRow*16u < trRaw);
  const bool frag1 = isActive && (sgRow*16u + 8u < trRaw);
  const uint tr8 = ((trRaw + 7u)/8u)*8u;

  for (uint blockIdx = 0u; blockIdx < NUMBLK; blockIdx++) {
    uint blkByte = blockIdx*22u;
    if (tid < 32u && isActive) {
      uint row = wro + colBase + tid;
      uint bb = row*RSB + blkByte;
      uint lo = rd_byte(b, bb), hi = rd_byte(b, bb + 1u);
      shared_d[tid] = float(as_type<half>(ushort(lo | (hi << 8u))));
      shared_qh[tid] = rd_byte(b, bb + 2u) | (rd_byte(b, bb + 3u) << 8u)
                     | (rd_byte(b, bb + 4u) << 16u) | (rd_byte(b, bb + 5u) << 24u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (isActive) {
      for (uint s = 0u; s < 8u; s++) {
        uint flat = tid + s*128u;
        uint m = flat/32u, k = flat % 32u;
        if (m < tr8) {
          float x = a[(rowBase + m)*K + blockIdx*32u + k];
          uint blk = (m/8u)*4u + k/8u;
          uint within = (m % 8u)*8u + (k % 8u);
          shared_A[blk*64u + within] = half(x);
        }
      }
      for (uint s = 0u; s < 4u; s++) {
        uint u = tid + s*128u;
        uint n = u/16u, kpair = u % 16u;
        uint row = wro + colBase + n;
        uint bb = row*RSB + blkByte;
        float d = shared_d[n];
        uint qh = shared_qh[n];
        uint k0 = kpair*2u;
        uint qs0 = rd_byte(b, bb + 6u + (k0 < 16u ? k0 : k0 - 16u));
        uint qs1 = rd_byte(b, bb + 6u + (k0+1u < 16u ? k0+1u : k0+1u - 16u));
        uint nib0 = (k0 < 16u) ? (qs0 & 0xFu) : ((qs0 >> 4u) & 0xFu);
        uint nib1 = (k0+1u < 16u) ? (qs1 & 0xFu) : ((qs1 >> 4u) & 0xFu);
        uint q50 = nib0 | (((qh >> k0) & 1u) << 4u);
        uint q51 = nib1 | (((qh >> (k0+1u)) & 1u) << 4u);
        float y0 = d * (float(q50) - 16.0f);
        float y1 = d * (float(q51) - 16.0f);
        uint ktile = k0/8u, kr = k0 % 8u;
        uint blkB = ktile*4u + n/8u;
        uint baseB = blkB*64u + (n % 8u);
        shared_B[baseB + kr*8u]      = half(y0);
        shared_B[baseB + (kr+1u)*8u] = half(y1);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k8 = 0u; k8 < 4u; k8++) {
      if (frag0) {
        uint blkA0 = (sgRow*2u)*4u + k8;
        simdgroup_load(Ax0, &shared_A[blkA0*64u], 8u);
        uint blkB0 = k8*4u + (sgCol*2u);
        simdgroup_load(Bx0, &shared_B[blkB0*64u], 8u);
        uint blkB1 = k8*4u + (sgCol*2u + 1u);
        simdgroup_load(Bx1, &shared_B[blkB1*64u], 8u);
        simdgroup_multiply_accumulate(Cx0, Ax0, Bx0, Cx0);
        simdgroup_multiply_accumulate(Cx1, Ax0, Bx1, Cx1);
      }
      if (frag1) {
        uint blkA1 = (sgRow*2u + 1u)*4u + k8;
        simdgroup_load(Ax1, &shared_A[blkA1*64u], 8u);
        simdgroup_multiply_accumulate(Cx2, Ax1, Bx0, Cx2);
        simdgroup_multiply_accumulate(Cx3, Ax1, Bx1, Cx3);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (frag0) {
    simdgroup_store(Cx0, &shared_C[mOff*32u + nOff], 32u);
    simdgroup_store(Cx1, &shared_C[mOff*32u + nOff + 8u], 32u);
  }
  if (frag1) {
    simdgroup_store(Cx2, &shared_C[(mOff+8u)*32u + nOff], 32u);
    simdgroup_store(Cx3, &shared_C[(mOff+8u)*32u + nOff + 8u], 32u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (isActive) {
    for (uint s = 0u; s < 8u; s++) {
      uint flat = tid + s*128u;
      uint r = flat/32u, c = flat % 32u;
      uint rowGlobal = rowBase + r;
      uint slotR = slot[rowGlobal];
      if (slotR < NUSED) {
        uint posR = pos[rowGlobal];
        dst[slotR*(NTOK*N) + posR*N + colBase + c] = shared_C[r*32u + c];
      }
    }
  }
}
)MSL";

static id<MTLComputePipelineState> g_q5dPso = nil;
static uint32_t g_q5dKey[6] = {0,0,0,0,0,0};

extern "C" lean_obj_res lean_hesper_msl_q5down_dispatch(
    b_lean_obj_arg device_obj,
    b_lean_obj_arg a_obj, b_lean_obj_arg b_obj, b_lean_obj_arg te_obj, b_lean_obj_arg tr_obj,
    b_lean_obj_arg pos_obj, b_lean_obj_arg slot_obj, b_lean_obj_arg dst_obj,
    uint32_t Mv, uint32_t Nv, uint32_t Kv, uint32_t nExpert, uint32_t nUsed, uint32_t nTok,
    lean_obj_res /* unit */) {
    wgpu::Device* device = mr_extract_device(device_obj);
    if (!device || !device->Get()) return lean_io_result_mk_error(lean_mk_string("msl_q5down: invalid device"));
    id<MTLDevice> mtl = dawn::native::metal::GetMTLDevice(device->Get());
    wgpu::Buffer* bufs[7] = { mr_extract_buffer(a_obj),   mr_extract_buffer(b_obj),
                              mr_extract_buffer(te_obj),  mr_extract_buffer(tr_obj),
                              mr_extract_buffer(pos_obj), mr_extract_buffer(slot_obj),
                              mr_extract_buffer(dst_obj) };
    id<MTLBuffer> mb[7];
    for (int i = 0; i < 7; i++) {
        if (!bufs[i] || !bufs[i]->Get()) return lean_io_result_mk_error(lean_mk_string("msl_q5down: invalid buffer"));
        mb[i] = reinterpret_cast<dawn::native::metal::Buffer*>(bufs[i]->Get())->GetMTLBuffer();
        if (!mb[i]) return lean_io_result_mk_error(lean_mk_string("msl_q5down: null MTLBuffer"));
    }
    uint32_t key[6] = {Mv, Nv, Kv, nExpert, nUsed, nTok};
    if (!g_q5dPso || memcmp(key, g_q5dKey, sizeof(key)) != 0) {
        std::string msl(kQ5DownMslTemplate);
        auto subst = [&](const char* tok, uint32_t v) {
            std::string t(tok); std::string r = std::to_string(v);
            size_t p;
            while ((p = msl.find(t)) != std::string::npos) msl.replace(p, t.size(), r);
        };
        subst("@M@", Mv); subst("@N@", Nv); subst("@K@", Kv);
        subst("@NEXP@", nExpert); subst("@NUSED@", nUsed); subst("@NTOK@", nTok);
        NSError* err = nil;
        MTLCompileOptions* opts = [MTLCompileOptions new];
        id<MTLLibrary> lib = [mtl newLibraryWithSource:[NSString stringWithUTF8String:msl.c_str()] options:opts error:&err];
        if (!lib) return lean_io_result_mk_error(lean_mk_string(
            [[NSString stringWithFormat:@"msl_q5down compile failed: %@", err] UTF8String]));
        id<MTLFunction> fn = [lib newFunctionWithName:@"q5_down_indexed_scatter"];
        id<MTLComputePipelineState> pso = [mtl newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) return lean_io_result_mk_error(lean_mk_string("msl_q5down: pipeline failed"));
        g_q5dPso = pso;
        memcpy(g_q5dKey, key, sizeof(key));
    }
    if (!g_mslQueue) g_mslQueue = [mtl newCommandQueue];
    dawn::native::metal::WaitForCommandsToBeScheduled(device->Get());
    id<MTLCommandBuffer> cb = [g_mslQueue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:g_q5dPso];
    for (int i = 0; i < 7; i++) [enc setBuffer:mb[i] offset:0 atIndex:i];
    [enc dispatchThreadgroups:MTLSizeMake((Nv + 31) / 32, (Mv + 31) / 32, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc endEncoding];
    [cb commit];   // no wait — hazard tracking orders vs Dawn's committed work
    return lean_io_result_mk_ok(lean_box(0));
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
