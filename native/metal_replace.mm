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
#include <atomic>
#include <unordered_set>

// DG_GPUBUSY: MSL-path accounting. wait = time in WaitForCommandsToBeScheduled (Dawn→MSL handoff);
// enc = command-buffer build+encode+commit (CPU); gpu = kernel GPU time (GPUEnd-GPUStart, via a
// completed handler). count = MSL dispatches. Read+reset by lean_hesper_msl_busy_read.
static std::atomic<uint64_t> g_msl_count{0};
static std::atomic<uint64_t> g_msl_wait_ns{0};
static std::atomic<uint64_t> g_msl_enc_ns{0};
static std::atomic<uint64_t> g_msl_gpu_ns{0};

static inline void mslAccountGpu(id<MTLCommandBuffer> cb) {
    [cb addCompletedHandler:^(id<MTLCommandBuffer> c) {
        g_msl_gpu_ns.fetch_add((uint64_t)((c.GPUEndTime - c.GPUStartTime) * 1e9), std::memory_order_relaxed);
    }];
}

extern "C" lean_obj_res lean_hesper_msl_busy_read(lean_obj_res /* unit */) {
    char b[256];
    snprintf(b, sizeof(b), "msl_count=%llu wait=%.2fms enc=%.2fms gpu=%.2fms",
             (unsigned long long)g_msl_count.exchange(0),
             g_msl_wait_ns.exchange(0) / 1e6, g_msl_enc_ns.exchange(0) / 1e6,
             g_msl_gpu_ns.exchange(0) / 1e6);
    return lean_io_result_mk_ok(lean_mk_string(b));
}

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

// MINIMAL decls to reach Dawn's OWN MTLCommandQueue (same reinterpret trick). Committing the MSL
// command buffers to Dawn's queue instead of a separate one makes commit-order == execution-order
// on a single Metal queue → correct producer→MSL→consumer ordering WITHOUT the per-dispatch
// WaitForCommandsToBeScheduled CPU block (~7ms × 60/step = 430ms/step, the whole gap to llama.cpp).
// Symbols verified in libwebgpu_dawn.a:
//   dawn::native::metal::Queue::GetPendingCommandContext(dawn::native::ExecutionQueueBase::SubmitMode)
//   dawn::native::metal::CommandRecordingContext::GetCommands()
// Passive submitMode is peek-only (no SetNeedsSubmit); GetCommands returns the current pending
// MTLCommandBuffer, whose -commandQueue IS Dawn's mCommandQueue.
namespace dawn::native {
class ExecutionQueueBase {
 public:
    enum class SubmitMode { Normal, Passive };
};
namespace metal {
class CommandRecordingContext {
 public:
    id<MTLCommandBuffer> GetCommands();
};
class Queue {
 public:
    CommandRecordingContext* GetPendingCommandContext(dawn::native::ExecutionQueueBase::SubmitMode);
};
}  // namespace metal
}  // namespace dawn::native

// Dawn's own MTLCommandQueue (cached). WGPUQueue C-handle == QueueBase* == metal::Queue* (single
// inheritance, QueueBase-first). Returns nil if Dawn hasn't prepared a command buffer yet.
static id<MTLCommandQueue> g_dawnQueue = nil;
static id<MTLCommandQueue> mr_dawn_queue(wgpu::Device* device) {
    if (g_dawnQueue) return g_dawnQueue;
    WGPUQueue qh = device->GetQueue().Get();
    if (!qh) return nil;
    auto* dq = reinterpret_cast<dawn::native::metal::Queue*>(qh);
    auto* ctx = dq->GetPendingCommandContext(dawn::native::ExecutionQueueBase::SubmitMode::Passive);
    if (!ctx) return nil;
    id<MTLCommandBuffer> cb = ctx->GetCommands();
    if (!cb) return nil;
    g_dawnQueue = [cb commandQueue];
    return g_dawnQueue;
}

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
    // Commit to Dawn's OWN queue → commit-order == execution-order on one Metal queue → correct
    // producer→MSL→consumer ordering (the decode flushBatch-commits Dawn producers before this call
    // and consumers after) with NO WaitForCommandsToBeScheduled CPU block. Falls back to a separate
    // queue + the scheduling wait only if Dawn's queue isn't reachable yet (first dispatch). Set
    // DG_MSLSEPQUEUE to force the old separate-queue+wait path.
    auto _tw0 = std::chrono::steady_clock::now();
    static const bool sepQueue = getenv("DG_MSLSEPQUEUE") != nullptr;
    id<MTLCommandQueue> mslQ = sepQueue ? nil : mr_dawn_queue(device);
    if (!mslQ) {
        if (!g_mslQueue) g_mslQueue = [mtl newCommandQueue];
        dawn::native::metal::WaitForCommandsToBeScheduled(device->Get());
        mslQ = g_mslQueue;
    }
    auto _tw1 = std::chrono::steady_clock::now();
    id<MTLCommandBuffer> cb = [mslQ commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:g_q4kPso];
    for (int i = 0; i < 6; i++) [enc setBuffer:mb[i] offset:0 atIndex:i];
    [enc dispatchThreadgroups:MTLSizeMake((Nv + 31) / 32, (Mv + 31) / 32, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc endEncoding];
    mslAccountGpu(cb);
    [cb commit];   // no wait — hazard tracking orders vs Dawn's committed work
    g_msl_count.fetch_add(1, std::memory_order_relaxed);
    g_msl_wait_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(_tw1 - _tw0).count(), std::memory_order_relaxed);
    g_msl_enc_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - _tw1).count(), std::memory_order_relaxed);
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
constant uint FUSED = @FUSED@u;   // 1 = buffer(0) is the raw grouped gate/up [M,2K]; geglu inline

inline uint rd_byte(device const uint* b, uint bo) {
  return (b[bo >> 2u] >> ((bo & 3u)*8u)) & 0xFFu;
}

// tanh-GELU geglu (matches gegluMergedB / q8FusedGegluDownScatterKernel): gate*0.5*(1+tanh(...))*up
inline float geglu_row(device const float* gu, uint row, uint kG) {
  float gate = gu[row*(2u*K) + kG];
  float up   = gu[row*(2u*K) + K + kG];
  float g3 = gate*gate*gate;
  float inner = clamp(0.7978845608f*(gate + 0.044715f*g3), -10.0f, 10.0f);
  return 0.5f*gate*(1.0f + tanh(inner)) * up;
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
          uint kG = blockIdx*32u + k;
          float x = FUSED ? geglu_row(a, rowBase + m, kG) : a[(rowBase + m)*K + kG];
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
static uint32_t g_q8dKey[7] = {0,0,0,0,0,0,0};

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
    uint32_t fused = getenv("DG_FUSEDOWN") != nullptr ? 1u : 0u;
    uint32_t key[7] = {Mv, Nv, Kv, nExpert, nUsed, nTok, fused};
    if (!g_q8dPso || memcmp(key, g_q8dKey, sizeof(key)) != 0) {
        std::string msl(kQ8DownMslTemplate);
        auto subst = [&](const char* tok, uint32_t v) {
            std::string t(tok); std::string r = std::to_string(v);
            size_t p;
            while ((p = msl.find(t)) != std::string::npos) msl.replace(p, t.size(), r);
        };
        subst("@M@", Mv); subst("@N@", Nv); subst("@K@", Kv);
        subst("@NEXP@", nExpert); subst("@NUSED@", nUsed); subst("@NTOK@", nTok);
        subst("@FUSED@", fused);
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
    // Commit to Dawn's own queue (commit-order == execution-order, no scheduling wait); fall back to
    // a separate queue + WaitForCommandsToBeScheduled if Dawn's queue isn't reachable. See mr_dawn_queue.
    auto _tw0 = std::chrono::steady_clock::now();
    static const bool sepQueue = getenv("DG_MSLSEPQUEUE") != nullptr;
    id<MTLCommandQueue> mslQ = sepQueue ? nil : mr_dawn_queue(device);
    if (!mslQ) {
        if (!g_mslQueue) g_mslQueue = [mtl newCommandQueue];
        dawn::native::metal::WaitForCommandsToBeScheduled(device->Get());
        mslQ = g_mslQueue;
    }
    auto _tw1 = std::chrono::steady_clock::now();
    id<MTLCommandBuffer> cb = [mslQ commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:g_q8dPso];
    for (int i = 0; i < 7; i++) [enc setBuffer:mb[i] offset:0 atIndex:i];
    [enc dispatchThreadgroups:MTLSizeMake((Nv + 31) / 32, (Mv + 31) / 32, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc endEncoding];
    mslAccountGpu(cb);
    [cb commit];   // no wait — hazard tracking orders vs Dawn's committed work
    g_msl_count.fetch_add(1, std::memory_order_relaxed);
    g_msl_wait_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(_tw1 - _tw0).count(), std::memory_order_relaxed);
    g_msl_enc_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - _tw1).count(), std::memory_order_relaxed);
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
constant uint FUSED = @FUSED@u;   // 1 = buffer(0) is the raw grouped gate/up [M,2K]; geglu inline

inline uint rd_byte(device const uint* b, uint bo) {
  return (b[bo >> 2u] >> ((bo & 3u)*8u)) & 0xFFu;
}

inline float geglu_row(device const float* gu, uint row, uint kG) {
  float gate = gu[row*(2u*K) + kG];
  float up   = gu[row*(2u*K) + K + kG];
  float g3 = gate*gate*gate;
  float inner = clamp(0.7978845608f*(gate + 0.044715f*g3), -10.0f, 10.0f);
  return 0.5f*gate*(1.0f + tanh(inner)) * up;
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
          uint kG = blockIdx*32u + k;
          float x = FUSED ? geglu_row(a, rowBase + m, kG) : a[(rowBase + m)*K + kG];
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
static uint32_t g_q5dKey[7] = {0,0,0,0,0,0,0};

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
    uint32_t fused = getenv("DG_FUSEDOWN") != nullptr ? 1u : 0u;
    uint32_t key[7] = {Mv, Nv, Kv, nExpert, nUsed, nTok, fused};
    if (!g_q5dPso || memcmp(key, g_q5dKey, sizeof(key)) != 0) {
        std::string msl(kQ5DownMslTemplate);
        auto subst = [&](const char* tok, uint32_t v) {
            std::string t(tok); std::string r = std::to_string(v);
            size_t p;
            while ((p = msl.find(t)) != std::string::npos) msl.replace(p, t.size(), r);
        };
        subst("@M@", Mv); subst("@N@", Nv); subst("@K@", Kv);
        subst("@NEXP@", nExpert); subst("@NUSED@", nUsed); subst("@NTOK@", nTok);
        subst("@FUSED@", fused);
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
    // Commit to Dawn's own queue (commit-order == execution-order, no scheduling wait); fall back to
    // a separate queue + WaitForCommandsToBeScheduled if Dawn's queue isn't reachable. See mr_dawn_queue.
    auto _tw0 = std::chrono::steady_clock::now();
    static const bool sepQueue = getenv("DG_MSLSEPQUEUE") != nullptr;
    id<MTLCommandQueue> mslQ = sepQueue ? nil : mr_dawn_queue(device);
    if (!mslQ) {
        if (!g_mslQueue) g_mslQueue = [mtl newCommandQueue];
        dawn::native::metal::WaitForCommandsToBeScheduled(device->Get());
        mslQ = g_mslQueue;
    }
    auto _tw1 = std::chrono::steady_clock::now();
    id<MTLCommandBuffer> cb = [mslQ commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:g_q5dPso];
    for (int i = 0; i < 7; i++) [enc setBuffer:mb[i] offset:0 atIndex:i];
    [enc dispatchThreadgroups:MTLSizeMake((Nv + 31) / 32, (Mv + 31) / 32, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc endEncoding];
    mslAccountGpu(cb);
    [cb commit];   // no wait — hazard tracking orders vs Dawn's committed work
    g_msl_count.fetch_add(1, std::memory_order_relaxed);
    g_msl_wait_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(_tw1 - _tw0).count(), std::memory_order_relaxed);
    g_msl_enc_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - _tw1).count(), std::memory_order_relaxed);
    return lean_io_result_mk_ok(lean_box(0));
}

// SINGLE-STREAM: encode the MSL gate/up AND the fused MSL down into ONE command buffer (two compute
// encoders, ONE commit) — testing whether same-cb execution removes the ~220ms/step inter-cb handoff
// bubbles. Two compute encoders in one cb are implicitly ordered + memory-coherent on Metal, so the
// gate/up→sGatheredGU→down chain is correct with no explicit barrier. Requires the FUSED down (reads
// sGatheredGU + inline geglu) so there is no Dawn work between the two MSL kernels. isQ5 selects the
// down kernel. Buffers: gate/up {src,idx,b(guE),c(sGatheredGU),te,tr}; down reuses c as its A and idx
// as its pos, plus {b(dnE),slot,dst}. Shares the g_q4k/q8d/q5d PSO caches with the separate dispatches.
extern "C" lean_obj_res lean_hesper_msl_gateup_down_onecb(
    b_lean_obj_arg device_obj,
    b_lean_obj_arg guSrc_obj, b_lean_obj_arg guIdx_obj, b_lean_obj_arg guB_obj, b_lean_obj_arg guC_obj,
    b_lean_obj_arg te_obj, b_lean_obj_arg tr_obj,
    b_lean_obj_arg dnB_obj, b_lean_obj_arg dnSlot_obj, b_lean_obj_arg dnDst_obj,
    uint32_t maxPadded, uint32_t guN, uint32_t guK, uint32_t nExpert, uint32_t srcRows,
    uint32_t dnN, uint32_t dnK, uint32_t nUsed, uint32_t nTok, uint32_t isQ5,
    lean_obj_res /* unit */) {
    wgpu::Device* device = mr_extract_device(device_obj);
    if (!device || !device->Get()) return lean_io_result_mk_error(lean_mk_string("msl_onecb: invalid device"));
    id<MTLDevice> mtl = dawn::native::metal::GetMTLDevice(device->Get());
    b_lean_obj_arg objs[9] = {guSrc_obj, guIdx_obj, guB_obj, guC_obj, te_obj, tr_obj, dnB_obj, dnSlot_obj, dnDst_obj};
    id<MTLBuffer> Mb[9];
    for (int i = 0; i < 9; i++) {
        wgpu::Buffer* wb = mr_extract_buffer(objs[i]);
        if (!wb || !wb->Get()) return lean_io_result_mk_error(lean_mk_string("msl_onecb: invalid buffer"));
        Mb[i] = reinterpret_cast<dawn::native::metal::Buffer*>(wb->Get())->GetMTLBuffer();
        if (!Mb[i]) return lean_io_result_mk_error(lean_mk_string("msl_onecb: null MTLBuffer"));
    }
    id<MTLBuffer> guSrc=Mb[0], guIdx=Mb[1], guB=Mb[2], guC=Mb[3], te=Mb[4], tr=Mb[5], dnB=Mb[6], dnSlot=Mb[7], dnDst=Mb[8];
    auto substOf = [](std::string& msl, const char* t, uint32_t v){
        std::string tk(t), r = std::to_string(v); size_t p;
        while ((p = msl.find(tk)) != std::string::npos) msl.replace(p, tk.size(), r);
    };
    // gate/up PSO (shares g_q4kPso cache)
    uint32_t gkey[5] = {maxPadded, guN, guK, nExpert, srcRows};
    if (!g_q4kPso || memcmp(gkey, g_q4kKey, sizeof(gkey)) != 0) {
        std::string msl(kQ4kMslTemplate);
        substOf(msl,"@M@",maxPadded); substOf(msl,"@N@",guN); substOf(msl,"@K@",guK);
        substOf(msl,"@NEXP@",nExpert); substOf(msl,"@SRCROWS@",srcRows);
        NSError* err=nil; MTLCompileOptions* opts=[MTLCompileOptions new];
        id<MTLLibrary> lib=[mtl newLibraryWithSource:[NSString stringWithUTF8String:msl.c_str()] options:opts error:&err];
        if(!lib) return lean_io_result_mk_error(lean_mk_string([[NSString stringWithFormat:@"msl_onecb q4k compile: %@",err] UTF8String]));
        id<MTLFunction> fn=[lib newFunctionWithName:@"q4k_grouped_reg_indexed"];
        id<MTLComputePipelineState> pso=[mtl newComputePipelineStateWithFunction:fn error:&err];
        if(!pso) return lean_io_result_mk_error(lean_mk_string("msl_onecb: q4k pipeline failed"));
        g_q4kPso=pso; memcpy(g_q4kKey,gkey,sizeof(gkey));
    }
    // down PSO (fused=1), q8 or q5 (shares g_q8dPso/g_q5dPso caches)
    uint32_t dkey[7] = {maxPadded, dnN, dnK, nExpert, nUsed, nTok, 1u};
    id<MTLComputePipelineState> downPso = nil;
    if (isQ5) {
        if (!g_q5dPso || memcmp(dkey, g_q5dKey, sizeof(dkey)) != 0) {
            std::string msl(kQ5DownMslTemplate);
            substOf(msl,"@M@",maxPadded); substOf(msl,"@N@",dnN); substOf(msl,"@K@",dnK);
            substOf(msl,"@NEXP@",nExpert); substOf(msl,"@NUSED@",nUsed); substOf(msl,"@NTOK@",nTok); substOf(msl,"@FUSED@",1u);
            NSError* err=nil; MTLCompileOptions* opts=[MTLCompileOptions new];
            id<MTLLibrary> lib=[mtl newLibraryWithSource:[NSString stringWithUTF8String:msl.c_str()] options:opts error:&err];
            if(!lib) return lean_io_result_mk_error(lean_mk_string([[NSString stringWithFormat:@"msl_onecb q5 compile: %@",err] UTF8String]));
            id<MTLFunction> fn=[lib newFunctionWithName:@"q5_down_indexed_scatter"];
            id<MTLComputePipelineState> pso=[mtl newComputePipelineStateWithFunction:fn error:&err];
            if(!pso) return lean_io_result_mk_error(lean_mk_string("msl_onecb: q5 pipeline failed"));
            g_q5dPso=pso; memcpy(g_q5dKey,dkey,sizeof(dkey));
        }
        downPso = g_q5dPso;
    } else {
        if (!g_q8dPso || memcmp(dkey, g_q8dKey, sizeof(dkey)) != 0) {
            std::string msl(kQ8DownMslTemplate);
            substOf(msl,"@M@",maxPadded); substOf(msl,"@N@",dnN); substOf(msl,"@K@",dnK);
            substOf(msl,"@NEXP@",nExpert); substOf(msl,"@NUSED@",nUsed); substOf(msl,"@NTOK@",nTok); substOf(msl,"@FUSED@",1u);
            NSError* err=nil; MTLCompileOptions* opts=[MTLCompileOptions new];
            id<MTLLibrary> lib=[mtl newLibraryWithSource:[NSString stringWithUTF8String:msl.c_str()] options:opts error:&err];
            if(!lib) return lean_io_result_mk_error(lean_mk_string([[NSString stringWithFormat:@"msl_onecb q8 compile: %@",err] UTF8String]));
            id<MTLFunction> fn=[lib newFunctionWithName:@"q8_down_indexed_scatter"];
            id<MTLComputePipelineState> pso=[mtl newComputePipelineStateWithFunction:fn error:&err];
            if(!pso) return lean_io_result_mk_error(lean_mk_string("msl_onecb: q8 pipeline failed"));
            g_q8dPso=pso; memcpy(g_q8dKey,dkey,sizeof(dkey));
        }
        downPso = g_q8dPso;
    }
    // ONE command buffer, TWO encoders, ONE commit
    auto _tw0 = std::chrono::steady_clock::now();
    static const bool sepQueue = getenv("DG_MSLSEPQUEUE") != nullptr;
    id<MTLCommandQueue> mslQ = sepQueue ? nil : mr_dawn_queue(device);
    if (!mslQ) { if (!g_mslQueue) g_mslQueue = [mtl newCommandQueue]; dawn::native::metal::WaitForCommandsToBeScheduled(device->Get()); mslQ = g_mslQueue; }
    auto _tw1 = std::chrono::steady_clock::now();
    id<MTLCommandBuffer> cb = [mslQ commandBuffer];
    id<MTLComputeCommandEncoder> e1 = [cb computeCommandEncoder];
    [e1 setComputePipelineState:g_q4kPso];
    { id<MTLBuffer> gb[6] = {guSrc, guIdx, guB, guC, te, tr}; for (int i=0;i<6;i++) [e1 setBuffer:gb[i] offset:0 atIndex:i]; }
    [e1 dispatchThreadgroups:MTLSizeMake((guN+31)/32, (maxPadded+31)/32, 1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    [e1 endEncoding];
    id<MTLComputeCommandEncoder> e2 = [cb computeCommandEncoder];
    [e2 setComputePipelineState:downPso];
    { id<MTLBuffer> db[7] = {guC, dnB, te, tr, guIdx, dnSlot, dnDst}; for (int i=0;i<7;i++) [e2 setBuffer:db[i] offset:0 atIndex:i]; }
    [e2 dispatchThreadgroups:MTLSizeMake((dnN+31)/32, (maxPadded+31)/32, 1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    [e2 endEncoding];
    mslAccountGpu(cb);
    [cb commit];
    g_msl_count.fetch_add(1, std::memory_order_relaxed);
    g_msl_wait_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(_tw1 - _tw0).count(), std::memory_order_relaxed);
    g_msl_enc_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - _tw1).count(), std::memory_order_relaxed);
    return lean_io_result_mk_ok(lean_box(0));
}

// DEVPLAN M1 — occupancy probe: compile a Tint-dumped MSL source with the REAL Metal compiler and
// report the pipeline's resource stats. Same source + same compiler as the Dawn path, so the numbers
// match what the deployed pipeline gets: maxTotalThreadsPerThreadgroup is the register-pressure
// inverse signal (drops when the kernel spills), threadExecutionWidth confirms simd width (32 on
// Apple), staticThreadgroupMemoryLength is the shared-memory footprint. This is the MANDATORY
// resource column of the autotune sweep (never report speed without it).
// ===========================================================================================
// M3 de-risk probe: does MTLDispatchTypeConcurrent recover the ~14 µs/dispatch serialization
// that Dawn's hardcoded Serial encoder costs us? Compiles a DUMPED Tint-MSL kernel (same code
// Dawn runs), records `nDispatches` back-to-back in ONE encoder with the requested dispatch
// type and NO barriers (racing writes are fine — timing only), returns GPU wall ms.
// ===========================================================================================
extern "C" lean_obj_res lean_hesper_msl_concurrent_probe(
    b_lean_obj_arg device_obj, b_lean_obj_arg msl_obj,
    b_lean_obj_arg b0_obj, b_lean_obj_arg b1_obj, b_lean_obj_arg b2_obj,
    uint32_t nDispatches, uint32_t gridX, uint32_t gridY, uint32_t wgX,
    uint8_t concurrent,
    lean_obj_res /* unit */) {
    @autoreleasepool {
        wgpu::Device* device = mr_extract_device(device_obj);
        if (!device || !device->Get()) return lean_io_result_mk_error(lean_mk_string("conc_probe: invalid device"));
        id<MTLDevice> mtl = dawn::native::metal::GetMTLDevice(device->Get());
        const char* mslSrc = lean_string_cstr(msl_obj);
        if (!mslSrc || mslSrc[0] == '\0')
            return lean_io_result_mk_error(lean_mk_string("conc_probe: empty MSL (set HESPER_DUMP_MSL=1)"));
        wgpu::Buffer* bufs[3] = { mr_extract_buffer(b0_obj), mr_extract_buffer(b1_obj), mr_extract_buffer(b2_obj) };
        id<MTLBuffer> mb[3];
        for (int i = 0; i < 3; i++) {
            if (!bufs[i] || !bufs[i]->Get()) return lean_io_result_mk_error(lean_mk_string("conc_probe: invalid buffer"));
            mb[i] = reinterpret_cast<dawn::native::metal::Buffer*>(bufs[i]->Get())->GetMTLBuffer();
        }
        NSError* err = nil;
        MTLCompileOptions* opts = [MTLCompileOptions new];
        id<MTLLibrary> lib = [mtl newLibraryWithSource:[NSString stringWithUTF8String:mslSrc] options:opts error:&err];
        if (!lib) return lean_io_result_mk_error(lean_mk_string(
            [[NSString stringWithFormat:@"conc_probe compile failed: %@", err] UTF8String]));
        id<MTLFunction> fn = [lib newFunctionWithName:@"dawn_entry_point"];
        if (!fn) return lean_io_result_mk_error(lean_mk_string("conc_probe: no dawn_entry_point"));
        id<MTLComputePipelineState> pso = [mtl newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) return lean_io_result_mk_error(lean_mk_string("conc_probe: PSO failed"));
        if (!g_mslQueue) g_mslQueue = [mtl newCommandQueue];
        id<MTLCommandBuffer> cb = [g_mslQueue commandBuffer];
        MTLComputePassDescriptor* desc = [MTLComputePassDescriptor computePassDescriptor];
        desc.dispatchType = concurrent ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial;
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDescriptor:desc];
        [enc setComputePipelineState:pso];
        for (int i = 0; i < 3; i++) [enc setBuffer:mb[i] offset:0 atIndex:i];
        for (uint32_t d = 0; d < nDispatches; d++) {
            [enc dispatchThreadgroups:MTLSizeMake(gridX, gridY, 1)
                threadsPerThreadgroup:MTLSizeMake(wgX, 1, 1)];
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        double gpuMs = ([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0;
        char buf[64];
        snprintf(buf, sizeof(buf), "%.4f", gpuMs);
        return lean_io_result_mk_ok(lean_mk_string(buf));
    }
}

// ===========================================================================================
// Native GPU timing for the autotune engine (HESPER_NATIVE_BENCH=1): benching tiny matvecs
// through Dawn measures Dawn's ~35 µs/dispatch overhead, not the kernel (native serial
// 13.5 µs vs 47 µs through Dawn for the SAME kernel — measured). This runs `nDispatches`
// of the variant's dumped Tint-MSL back-to-back in one SERIAL native encoder and returns
// the honest serialized GPU ms. Buffers come as a Lean Array in BINDING ORDER.
// ===========================================================================================
extern "C" lean_obj_res lean_hesper_msl_bench_serial(
    b_lean_obj_arg device_obj, b_lean_obj_arg msl_obj, b_lean_obj_arg bufs_array,
    uint32_t nDispatches, uint32_t gridX, uint32_t gridY, uint32_t gridZ, uint32_t wgX,
    lean_obj_res /* unit */) {
    @autoreleasepool {
        wgpu::Device* device = mr_extract_device(device_obj);
        if (!device || !device->Get()) return lean_io_result_mk_error(lean_mk_string("bench_serial: invalid device"));
        id<MTLDevice> mtl = dawn::native::metal::GetMTLDevice(device->Get());
        const char* mslSrc = lean_string_cstr(msl_obj);
        if (!mslSrc || mslSrc[0] == '\0')
            return lean_io_result_mk_error(lean_mk_string("bench_serial: empty MSL (set HESPER_DUMP_MSL=1)"));
        size_t nBufs = lean_array_size(bufs_array);
        std::vector<id<MTLBuffer>> mb(nBufs);
        for (size_t i = 0; i < nBufs; i++) {
            wgpu::Buffer* b = mr_extract_buffer(lean_array_get_core(bufs_array, i));
            if (!b || !b->Get()) return lean_io_result_mk_error(lean_mk_string("bench_serial: invalid buffer"));
            mb[i] = reinterpret_cast<dawn::native::metal::Buffer*>(b->Get())->GetMTLBuffer();
        }
        // PSO cache keyed by MSL hash — refine re-benches the same variants.
        static std::unordered_map<size_t, id<MTLComputePipelineState>> psoCache;
        size_t key = std::hash<std::string>{}(std::string(mslSrc));
        id<MTLComputePipelineState> pso = nil;
        auto it = psoCache.find(key);
        if (it != psoCache.end()) pso = it->second;
        else {
            NSError* err = nil;
            MTLCompileOptions* opts = [MTLCompileOptions new];
            id<MTLLibrary> lib = [mtl newLibraryWithSource:[NSString stringWithUTF8String:mslSrc] options:opts error:&err];
            if (!lib) return lean_io_result_mk_error(lean_mk_string(
                [[NSString stringWithFormat:@"bench_serial compile failed: %@", err] UTF8String]));
            id<MTLFunction> fn = [lib newFunctionWithName:@"dawn_entry_point"];
            if (!fn) return lean_io_result_mk_error(lean_mk_string("bench_serial: no dawn_entry_point"));
            pso = [mtl newComputePipelineStateWithFunction:fn error:&err];
            if (!pso) return lean_io_result_mk_error(lean_mk_string("bench_serial: PSO failed"));
            psoCache[key] = pso;
        }
        if (!g_mslQueue) g_mslQueue = [mtl newCommandQueue];
        id<MTLCommandBuffer> cb = [g_mslQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];   // serial (default)
        [enc setComputePipelineState:pso];
        for (size_t i = 0; i < nBufs; i++) [enc setBuffer:mb[i] offset:0 atIndex:i];
        for (uint32_t d = 0; d < nDispatches; d++) {
            [enc dispatchThreadgroups:MTLSizeMake(gridX, gridY, gridZ)
                threadsPerThreadgroup:MTLSizeMake(wgX, 1, 1)];
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        double gpuMs = ([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0;
        char buf[64];
        snprintf(buf, sizeof(buf), "%.4f", gpuMs);
        return lean_io_result_mk_ok(lean_mk_string(buf));
    }
}

extern "C" lean_obj_res lean_hesper_msl_occupancy_probe(
    b_lean_obj_arg device_obj, b_lean_obj_arg msl_obj, lean_obj_res /* unit */) {
    @autoreleasepool {
        wgpu::Device* device = mr_extract_device(device_obj);
        if (!device || !device->Get()) {
            return lean_io_result_mk_error(lean_mk_string("occupancy_probe: device is invalid"));
        }
        id<MTLDevice> mtl = dawn::native::metal::GetMTLDevice(device->Get());
        if (!mtl) {
            return lean_io_result_mk_error(lean_mk_string("occupancy_probe: null MTLDevice"));
        }
        const char* mslSrc = lean_string_cstr(msl_obj);
        if (!mslSrc || mslSrc[0] == '\0') {
            return lean_io_result_mk_error(lean_mk_string(
                "occupancy_probe: empty MSL source (is HESPER_DUMP_MSL=1 set and a pipeline compiled?)"));
        }
        NSError* err = nil;
        MTLCompileOptions* opts = [MTLCompileOptions new];
        id<MTLLibrary> lib = [mtl newLibraryWithSource:[NSString stringWithUTF8String:mslSrc]
                                               options:opts error:&err];
        if (!lib) {
            NSString* msg = [NSString stringWithFormat:@"occupancy_probe: newLibraryWithSource failed: %@",
                             err ? err.localizedDescription : @"(no error info)"];
            return lean_io_result_mk_error(lean_mk_string([msg UTF8String]));
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"dawn_entry_point"];
        if (!fn) {
            return lean_io_result_mk_error(lean_mk_string(
                "occupancy_probe: entry point 'dawn_entry_point' not found in dumped MSL"));
        }
        id<MTLComputePipelineState> pso = [mtl newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) {
            NSString* msg = [NSString stringWithFormat:@"occupancy_probe: PSO creation failed: %@",
                             err ? err.localizedDescription : @"(no error info)"];
            return lean_io_result_mk_error(lean_mk_string([msg UTF8String]));
        }
        char buf[128];
        snprintf(buf, sizeof(buf), "maxThreads=%lu execWidth=%lu tgMem=%lu",
                 (unsigned long)pso.maxTotalThreadsPerThreadgroup,
                 (unsigned long)pso.threadExecutionWidth,
                 (unsigned long)pso.staticThreadgroupMemoryLength);
        return lean_io_result_mk_ok(lean_mk_string(buf));
    }
}

// ===========================================================================================
// Experiment 2 Phase A: NATIVE REPLAY of a captured decode-token dispatch sequence.
// Falsification test of the post-mortem's C2 ("the gap is the dispatch layer, not kernels"):
// the Lean side captures every dispatch of ONE decode token (Tint-CLI MSL + buffers in MSL
// binding order + grid/wg dims + threadgroup bytes), then we replay the whole token in ONE
// native command buffer, timing Serial vs MTLDispatchTypeConcurrent (no-barrier upper bound,
// and per-layer-barrier realistic bound via barrier markers). Timing-only: the replay runs on
// the live Dawn buffers AFTER decode finishes, so values are garbage but layout is identical —
// all our kernels are fixed-trip-count, so timing is value-independent.
// ===========================================================================================
struct HesperReplayOp {
    id<MTLComputePipelineState> pso;   // nil => barrier marker
    std::vector<id<MTLBuffer>> bufs;
    uint32_t gx, gy, gz, tx, ty, tz;
    uint32_t tgBytes;
    uint32_t writeMask;  // bit i set => bufs[i] is written (WGSL read_write binding)
};
// Heap-allocated: never destroyed, so no static-destruction-order issues with ARC members.
static std::vector<HesperReplayOp>* g_replayOps = nullptr;
static std::unordered_map<size_t, id<MTLComputePipelineState>>* g_replayPsoCache = nullptr;
// Stashed at record time so replay_run needs no device handle (callable from
// backend-generic Lean code).
static id<MTLDevice> g_replayMtlDevice = nil;
static wgpu::Device* g_replayDawnDevice = nullptr;  // for mr_dawn_queue in replay_exec

static inline std::vector<HesperReplayOp>& hesper_replay_ops() {
    if (!g_replayOps) g_replayOps = new std::vector<HesperReplayOp>();
    return *g_replayOps;
}

extern "C" lean_obj_res lean_hesper_replay_reset(lean_obj_res /* unit */) {
    hesper_replay_ops().clear();
    return lean_io_result_mk_ok(lean_box(0));
}

extern "C" lean_obj_res lean_hesper_replay_barrier(lean_obj_res /* unit */) {
    HesperReplayOp op{};
    op.pso = nil;
    hesper_replay_ops().push_back(std::move(op));
    return lean_io_result_mk_ok(lean_box(0));
}

// Record one dispatch. `bufs_array` must already be in MSL [[buffer(i)]] order (the Lean side
// parses the Tint-CLI entry-point signature). `entry` = the MSL kernel function name.
// `key` = the Lean-side authoritative cache key: the PSO cache is keyed on it directly so the
// steady-state record cost is a u64 map probe — NOT a copy+hash of the ~10 KB MSL string
// (572 records/token made that megabytes of hashing per token).
extern "C" lean_obj_res lean_hesper_replay_record(
    b_lean_obj_arg device_obj, b_lean_obj_arg msl_obj, b_lean_obj_arg entry_obj,
    b_lean_obj_arg bufs_array,
    uint32_t gx, uint32_t gy, uint32_t gz,
    uint32_t tx, uint32_t ty, uint32_t tz,
    uint32_t tgBytes, uint32_t writeMask, uint64_t key, lean_obj_res /* unit */) {
    @autoreleasepool {
        wgpu::Device* device = mr_extract_device(device_obj);
        if (!device || !device->Get()) return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("replay_record: invalid device")));
        id<MTLDevice> mtl = dawn::native::metal::GetMTLDevice(device->Get());
        g_replayMtlDevice = mtl;
        g_replayDawnDevice = device;

        if (!g_replayPsoCache) g_replayPsoCache = new std::unordered_map<size_t, id<MTLComputePipelineState>>();
        id<MTLComputePipelineState> pso = nil;
        auto it = g_replayPsoCache->find(key);
        if (it != g_replayPsoCache->end()) pso = it->second;
        else {
            const char* mslSrc = lean_string_cstr(msl_obj);
            const char* entry = lean_string_cstr(entry_obj);
            if (!mslSrc || mslSrc[0] == '\0') return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("replay_record: empty MSL")));
            NSError* err = nil;
            MTLCompileOptions* opts = [MTLCompileOptions new];
            id<MTLLibrary> lib = [mtl newLibraryWithSource:[NSString stringWithUTF8String:mslSrc] options:opts error:&err];
            if (!lib) return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(
                [[NSString stringWithFormat:@"replay_record compile failed (%s): %@", entry, err] UTF8String])));
            id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:entry]];
            if (!fn) return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(
                [[NSString stringWithFormat:@"replay_record: entry '%s' not found", entry] UTF8String])));
            pso = [mtl newComputePipelineStateWithFunction:fn error:&err];
            if (!pso) return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(
                [[NSString stringWithFormat:@"replay_record: PSO failed (%s): %@", entry, err] UTF8String])));
            (*g_replayPsoCache)[key] = pso;
        }

        HesperReplayOp op{};
        op.pso = pso;
        size_t nBufs = lean_array_size(bufs_array);
        op.bufs.resize(nBufs);
        for (size_t i = 0; i < nBufs; i++) {
            wgpu::Buffer* b = mr_extract_buffer(lean_array_get_core(bufs_array, i));
            if (!b || !b->Get()) return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("replay_record: invalid buffer")));
            op.bufs[i] = reinterpret_cast<dawn::native::metal::Buffer*>(b->Get())->GetMTLBuffer();
        }
        op.gx = gx; op.gy = gy; op.gz = gz;
        op.tx = tx; op.ty = ty; op.tz = tz;
        op.tgBytes = tgBytes;
        op.writeMask = writeMask;
        hesper_replay_ops().push_back(std::move(op));
        return lean_io_result_mk_ok(lean_box(0));
    }
}

// Encode the recorded sequence into ONE compute encoder. mode: 0 = Serial,
// 1 = Concurrent no barriers, 2 = Concurrent + barriers at recorded layer markers,
// 3 = Concurrent + AUTOMATIC hazard barriers (llama.cpp ggml_mem_ranges-style,
//     whole-buffer granularity: barrier before an op that reads a buffer written
//     since the last barrier, or writes a buffer read/written since the last barrier).
// Returns the number of barriers inserted.
static size_t hesper_replay_encode(id<MTLComputeCommandEncoder> enc, uint32_t mode,
                                   std::vector<HesperReplayOp>& ops) {
    size_t barriers = 0;
    std::unordered_set<void*> readSet, writeSet;
    for (auto& op : ops) {
        if (!op.pso) {
            if (mode == 2) { [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; barriers++; }
            continue;
        }
        if (mode == 3) {
            bool hazard = false;
            for (size_t i = 0; i < op.bufs.size() && !hazard; i++) {
                void* p = (__bridge void*)op.bufs[i];
                bool writes = (i < 32) && ((op.writeMask >> i) & 1u);
                if (writeSet.count(p)) hazard = true;                       // RAW / WAW
                else if (writes && readSet.count(p)) hazard = true;         // WAR
            }
            if (hazard) {
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                barriers++;
                readSet.clear();
                writeSet.clear();
            }
            for (size_t i = 0; i < op.bufs.size(); i++) {
                void* p = (__bridge void*)op.bufs[i];
                if ((i < 32) && ((op.writeMask >> i) & 1u)) writeSet.insert(p);
                else readSet.insert(p);
            }
        }
        [enc setComputePipelineState:op.pso];
        for (size_t i = 0; i < op.bufs.size(); i++) [enc setBuffer:op.bufs[i] offset:0 atIndex:i];
        if (op.tgBytes > 0)
            [enc setThreadgroupMemoryLength:((op.tgBytes + 15u) & ~15u) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(op.gx, op.gy, op.gz)
            threadsPerThreadgroup:MTLSizeMake(op.tx, op.ty, op.tz)];
    }
    return barriers;
}

// Execute the recorded token ONCE as the REAL computation (Exp 2 Phase B): the command
// buffer goes to DAWN'S OWN MTLCommandQueue when reachable (commit order == execution
// order with Dawn's staging writes / readbacks on a single queue), falling back to our
// private queue. Waits for completion. Returns "ms=<gpu ms> barriers=<n> queue=<dawn|own>".
extern "C" lean_obj_res lean_hesper_replay_exec(
    uint32_t mode, lean_obj_res /* unit */) {
    @autoreleasepool {
        id<MTLDevice> mtl = g_replayMtlDevice;
        if (!mtl) return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("replay_exec: nothing recorded")));
        auto& ops = hesper_replay_ops();
        size_t nOps = 0;
        for (auto& op : ops) if (op.pso) nOps++;
        if (nOps == 0) return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("replay_exec: nothing recorded")));
        id<MTLCommandQueue> q = g_replayDawnDevice ? mr_dawn_queue(g_replayDawnDevice) : nil;
        bool onDawnQueue = (q != nil);
        if (!q) {
            if (!g_mslQueue) g_mslQueue = [mtl newCommandQueue];
            q = g_mslQueue;
        }
        id<MTLCommandBuffer> cb = [q commandBuffer];
        MTLComputePassDescriptor* desc = [MTLComputePassDescriptor computePassDescriptor];
        desc.dispatchType = (mode == 0) ? MTLDispatchTypeSerial : MTLDispatchTypeConcurrent;
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDescriptor:desc];
        size_t barriers = hesper_replay_encode(enc, mode, ops);
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if ([cb status] == MTLCommandBufferStatusError) {
            return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(
                [[NSString stringWithFormat:@"replay_exec: command buffer error: %@", [cb error]] UTF8String])));
        }
        double ms = ([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0;
        char buf[128];
        snprintf(buf, sizeof(buf), "ms=%.4f barriers=%zu queue=%s", ms, barriers, onDawnQueue ? "dawn" : "own");
        return lean_io_result_mk_ok(lean_mk_string(buf));
    }
}

// Replay the recorded sequence. mode: 0 = Serial, 1 = Concurrent (no barriers — timing upper
// bound), 2 = Concurrent + memoryBarrier at recorded layer markers. Runs `iters` command
// buffers back-to-back; returns "count=<ops> min=<ms> avg=<ms>" (GPU wall per iteration).
extern "C" lean_obj_res lean_hesper_replay_run(
    uint32_t mode, uint32_t iters, lean_obj_res /* unit */) {
    @autoreleasepool {
        id<MTLDevice> mtl = g_replayMtlDevice;
        if (!mtl) return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("replay_run: nothing recorded (no device)")));
        auto& ops = hesper_replay_ops();
        size_t nOps = 0;
        for (auto& op : ops) if (op.pso) nOps++;
        if (nOps == 0) return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string("replay_run: nothing recorded")));
        if (!g_mslQueue) g_mslQueue = [mtl newCommandQueue];
        if (iters == 0) iters = 1;
        double best = 1e30, sum = 0.0;
        size_t barriers = 0;
        for (uint32_t r = 0; r < iters; r++) {
            id<MTLCommandBuffer> cb = [g_mslQueue commandBuffer];
            MTLComputePassDescriptor* desc = [MTLComputePassDescriptor computePassDescriptor];
            desc.dispatchType = (mode == 0) ? MTLDispatchTypeSerial : MTLDispatchTypeConcurrent;
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDescriptor:desc];
            barriers = hesper_replay_encode(enc, mode, ops);
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if ([cb status] == MTLCommandBufferStatusError) {
                return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(
                    [[NSString stringWithFormat:@"replay_run: command buffer error: %@", [cb error]] UTF8String])));
            }
            double ms = ([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0;
            best = std::min(best, ms);
            sum += ms;
        }
        char buf[160];
        snprintf(buf, sizeof(buf), "count=%zu barriers=%zu min=%.4f avg=%.4f", nOps, barriers, best, sum / iters);
        return lean_io_result_mk_ok(lean_mk_string(buf));
    }
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
